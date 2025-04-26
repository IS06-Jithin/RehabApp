import math, time, sqlite3, statistics
from pathlib import Path
from datetime import datetime
from typing import List, Optional

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi import APIRouter

# ────────────────────────────────────────────────────────────────────────────────
# Constants & Model Setup (same as standalone script)────────────────────────────
# -------------------------------------------------------------------------------
SCRIPT_DIR = Path().resolve()
CKPT_PATH = SCRIPT_DIR / "model/pose_quality_best.pt"

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
)
print("► device =", DEVICE)

POSE_LANDMARK_NAMES = [
    "NOSE",
    "LEFT_EYE_INNER",
    "LEFT_EYE",
    "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER",
    "RIGHT_EYE",
    "RIGHT_EYE_OUTER",
    "LEFT_EAR",
    "RIGHT_EAR",
    "MOUTH_LEFT",
    "MOUTH_RIGHT",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_PINKY",
    "RIGHT_PINKY",
    "LEFT_INDEX",
    "RIGHT_INDEX",
    "LEFT_THUMB",
    "RIGHT_THUMB",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
    "LEFT_HEEL",
    "RIGHT_HEEL",
    "LEFT_FOOT_INDEX",
    "RIGHT_FOOT_INDEX",
]
N_JOINTS = len(POSE_LANDMARK_NAMES)

EXERCISE_MAP = {
    1: "Arm-abduction",
    2: "Arm-VW",
    3: "Push-ups",
    4: "Leg-abduction",
    5: "Lunge",
    6: "Squat",
}
NUM_EX = len(EXERCISE_MAP)

ERR_JOINTS = [
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "SPINE",
    "HEAD",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]
JOINT_LABELS = [j.replace("_", " ").lower() for j in ERR_JOINTS]

JOINT_TRIPLETS = {
    "LEFT_ELBOW": (11, 13, 15),
    "RIGHT_ELBOW": (12, 14, 16),
    "LEFT_SHOULDER": (13, 11, 23),
    "RIGHT_SHOULDER": (14, 12, 24),
    "LEFT_HIP": (11, 23, 25),
    "RIGHT_HIP": (12, 24, 26),
    "LEFT_KNEE": (23, 25, 27),
    "RIGHT_KNEE": (24, 26, 28),
    "SPINE": (23, 11, 12),
    "HEAD": (11, 0, 12),
    "LEFT_WRIST": (13, 15, 19),
    "RIGHT_WRIST": (14, 16, 20),
    "LEFT_ANKLE": (25, 27, 31),
    "RIGHT_ANKLE": (26, 28, 32),
}

SEQ_LEN = 16
IN_DIM = N_JOINTS * 3
TH_ERR_DEG = 10  # threshold for joint‑angle advice

# ────────────────────────────────────────────────────────────────────────────────
# Model definition (same architecture as trainer)────────────────────────────────
# -------------------------------------------------------------------------------


class KeypointEncoder(nn.Module):
    def __init__(self, in_dim: int, embed: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, embed, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x.unsqueeze(2)))
        x = torch.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)


class PoseQualityNetKP(nn.Module):
    def __init__(self, in_dim: int, num_ex: int, hidden: int = 256, ex_emb: int = 64):
        super().__init__()
        self.encoder = KeypointEncoder(in_dim)
        self.lstm = nn.LSTM(
            512, hidden, num_layers=2, batch_first=True, bidirectional=True
        )
        feat_dim = hidden * 2
        self.ex_emb = nn.Sequential(
            nn.Linear(num_ex, ex_emb), nn.ReLU(), nn.Linear(ex_emb, ex_emb)
        )
        self.cls_head = nn.Linear(feat_dim + ex_emb, 2)
        self.err_head = nn.Linear(feat_dim + ex_emb, len(ERR_JOINTS))
        self.ex_head = nn.Linear(feat_dim, num_ex)

    def forward(self, seq: torch.Tensor, ex_1h: torch.Tensor):
        B, T, _ = seq.shape
        feats = torch.stack([self.encoder(seq[:, t]) for t in range(T)], dim=1)
        out, _ = self.lstm(feats)
        g = out.mean(1)
        ex_e = self.ex_emb(ex_1h)
        h = torch.cat([g, ex_e], dim=1)
        logits_q = self.cls_head(h)
        err_hat = self.err_head(h)
        logits_ex = self.ex_head(g)
        return logits_q, err_hat, logits_ex


# ────────────────────────────────────────────────────────────────────────────────
# Load weights────────────────────────────────────────────────────────────────────
# -------------------------------------------------------------------------------
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"{CKPT_PATH} not found")
print("Loading model …")
model = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(model, dict):
    _m = PoseQualityNetKP(IN_DIM, NUM_EX).to(DEVICE)
    _m.load_state_dict(model)
    model = _m
model.eval()
print("✓ model ready")

# ────────────────────────────────────────────────────────────────────────────────
# FastAPI setup───────────────────────────────────────────────────────────────────
# -------------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ────────────────────────────────────────────────────────────────────────────────
# Helper: joint advice builder (shared with live script)──────────────────────────
# -------------------------------------------------------------------------------


def build_advice(errs: np.ndarray) -> str:
    bad_idxs = np.argsort(np.abs(errs))[::-1][:3]
    joints = [JOINT_LABELS[i] for i in bad_idxs if abs(errs[i]) >= TH_ERR_DEG]
    if not joints:
        return "Check your form."
    if len(joints) == 1:
        joint_str = joints[0]
    elif len(joints) == 2:
        joint_str = f"{joints[0]} and {joints[1]}"
    else:
        joint_str = f"{joints[0]}, {joints[1]} and {joints[2]}"
    return f"Adjust your {joint_str} properly."


# ────────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint (accepts the *same payload* the front‑end sends)─────────────
# -------------------------------------------------------------------------------


@app.websocket("/ws/infer")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # session-level accumulators for KPIs
    correct_reps = 0
    wrong_reps = 0
    per_joint_sum = np.zeros(len(ERR_JOINTS), np.float32)
    per_joint_n = 0

    while True:
        try:
            msg = await ws.receive_json()
        except Exception:
            break  # client closed

        if msg.get("label") == "stop":
            # ─ send session summary and close ─
            mean_joint_err = (per_joint_sum / max(1, per_joint_n)).tolist()
            await ws.send_json(
                {
                    "type": "summary",
                    "correct": correct_reps,
                    "total": correct_reps + wrong_reps,
                    "joint_errors": mean_joint_err,
                }
            )
            break

        if msg.get("label") != "keypoint_sequence":
            await ws.send_json({"type": "error", "msg": "unknown label"})
            continue

        kp = np.asarray(msg["keypoints"], np.float32)
        exid = int(msg["exercise_id"])
        if kp.shape != (SEQ_LEN, N_JOINTS, 3):
            continue

        seq = torch.tensor(kp.reshape(SEQ_LEN, -1), device=DEVICE).unsqueeze(0)
        ex1 = F.one_hot(
            torch.tensor([exid - 1], device=DEVICE), len(EXERCISE_MAP)
        ).float()

        with torch.no_grad():
            log_q, err_hat, log_ex = model(seq, ex1)

        q_pred = log_q.argmax(1).item()
        ex_pred = log_ex.argmax(1).item() + 1
        errs = err_hat.squeeze().cpu().numpy()

        if ex_pred != exid:
            fb = f"Wrong exercise! Looks like {EXERCISE_MAP.get(ex_pred,'')}."
            sug = ""
        else:
            if q_pred == 1:
                fb, sug = "You're on the right track!", ""
                correct_reps += 1
            else:
                fb, sug = "You're doing it wrongly!", build_advice(errs)
                wrong_reps += 1

        per_joint_sum += np.abs(errs)
        per_joint_n += 1
        avg_dev = float(np.mean(np.abs(errs)))

        await ws.send_json(
            {
                "type": "progress",
                "feedback": fb,
                "suggestion": sug,
                "avg_error": avg_dev,
                "correct": correct_reps,
                "total": correct_reps + wrong_reps,
            }
        )


# ────────────────────────────── feedback DB setup ─────────────────────────────
DB_PATH = Path("feedback.sqlite")


def get_db():
    """yields a sqlite3 connection per-request and closes it afterwards"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """create table once"""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS feedback (
                   id            INTEGER PRIMARY KEY AUTOINCREMENT,
                   ts            TEXT    NOT NULL,
                   ease_of_use   INTEGER NOT NULL,
                   accuracy      INTEGER NOT NULL,
                   satisfaction  INTEGER NOT NULL,
                   comments      TEXT
               )"""
        )


init_db()


# ─────────────────────────────── Feedback schema ─────────────────────────────
class FeedbackIn(BaseModel):
    ease_of_use: int = Field(..., ge=1, le=5)
    accuracy: int = Field(..., ge=1, le=5)
    satisfaction: int = Field(..., ge=1, le=5)
    comments: Optional[str] = None


class FeedbackOut(FeedbackIn):
    id: int
    ts: str


class FeedbackSummary(BaseModel):
    count: int
    avg_ease: float
    avg_accuracy: float
    avg_satisf: float


# ───────────────────────────── Feedback endpoints ────────────────────────────
@app.post("/feedback", response_model=FeedbackOut, status_code=201)
def create_feedback(fb: FeedbackIn, db=Depends(get_db)):
    ts = datetime.utcnow().isoformat()
    cur = db.execute(
        "INSERT INTO feedback(ts,ease_of_use,accuracy,satisfaction,comments)"
        " VALUES (?,?,?,?,?)",
        (ts, fb.ease_of_use, fb.accuracy, fb.satisfaction, fb.comments),
    )
    db.commit()
    return FeedbackOut(id=cur.lastrowid, ts=ts, **fb.dict())


@app.get("/feedback", response_model=List[FeedbackOut])
def list_feedback(db=Depends(get_db)):
    rows = db.execute("SELECT * FROM feedback ORDER BY id DESC").fetchall()
    return [FeedbackOut(**row) for row in rows]


@app.get("/feedback/summary", response_model=FeedbackSummary)
def feedback_summary(db=Depends(get_db)):
    rows = db.execute(
        "SELECT ease_of_use,accuracy,satisfaction FROM feedback"
    ).fetchall()
    if not rows:
        return FeedbackSummary(count=0, avg_ease=0, avg_accuracy=0, avg_satisf=0)
    ease, acc, sat = zip(*rows)
    return FeedbackSummary(
        count=len(rows),
        avg_ease=round(statistics.mean(ease), 2),
        avg_accuracy=round(statistics.mean(acc), 2),
        avg_satisf=round(statistics.mean(sat), 2),
    )


@app.delete("/feedback/{fid}", status_code=204)
def delete_feedback(fid: int, db=Depends(get_db)):
    cur = db.execute("DELETE FROM feedback WHERE id=?", (fid,))
    db.commit()
    if cur.rowcount == 0:
        raise HTTPException(404, "feedback id not found")
