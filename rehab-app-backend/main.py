import math, time, sqlite3, statistics
from pathlib import Path
from datetime import datetime
from typing import List, Optional
from collections import deque, Counter
from typing import List, Optional, Union

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
TH_ERR_DEG = 8  # joint-angle advice cut-off
TH_EX_CONF = 0.65  # exercise-mismatch confidence
TH_Q_WRONG = 0.60  # wrong-form confidence
BUF_SIZE = 5  # temporal stabilisation window
IDX_CORRECT = 1  # index mapping in logits_q
IDX_WRONG = 0

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

    correct_reps = 0
    wrong_reps = 0
    per_joint_sum = np.zeros(len(ERR_JOINTS), np.float32)
    per_joint_n = 0
    start_ts = time.time()

    # ★ ADDED: temporal-stabilisation buffers
    state_buf: deque[str] = deque(maxlen=BUF_SIZE)
    stable_state: Optional[str] = None  # "good" | "bad_form" | "wrong_ex"
    last_wrong_pred: Optional[int] = None  # remember culprit ex ID
    # buffer of wrong-exercise IDs to allow fast refresh
    wrong_id_buf: deque[int] = deque(maxlen=BUF_SIZE)

    def mode(buf: deque[int]) -> Optional[int]:
        """
        Return the most common value if it appears ≥2 times;
        otherwise return None (guards against a single noisy frame).
        """
        if not buf:
            return None
        val, freq = Counter(buf).most_common(1)[0]
        return val if freq >= 2 else None

    stable_state: Optional[str] = None  # "good" | "bad_form" | "wrong_ex" | None

    def decide_state(ex_match: bool, prob_wrong: float) -> str:
        if not ex_match:
            return "wrong_ex"
        return "bad_form" if prob_wrong >= TH_Q_WRONG else "good"

    def majority_state(buf: deque) -> Optional[Union[int, str]]:
        cnt = Counter(buf)
        return cnt.most_common(1)[0][0] if cnt else "good"

    while True:
        try:
            msg = await ws.receive_json()
        except Exception:
            break  # client closed

        if msg.get("label") == "stop":
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

        # ───────── probabilities & predictions ─────────
        prob_q = torch.softmax(log_q, dim=-1)[0]
        prob_ex = torch.softmax(log_ex, dim=-1)[0]

        ex_pred = int(prob_ex.argmax().item()) + 1
        ex_conf = float(prob_ex[ex_pred - 1])
        ex_match = (ex_pred == exid) or (ex_conf < TH_EX_CONF)

        prob_wrong = float(prob_q[IDX_WRONG])  # idx-0 == wrong
        errs_abs = np.abs(err_hat.squeeze().cpu().numpy())

        # ───────── temporal stabilisation ─────────
        # ───────── temporal stabilisation ─────────
        instant_state = decide_state(ex_match, prob_wrong)
        state_buf.append(instant_state)

        # ★ NEW – track wrong-exercise predictions
        if instant_state == "wrong_ex":
            wrong_id_buf.append(ex_pred)
        else:
            wrong_id_buf.clear()

        new_stable = majority_state(state_buf)
        state_changed = new_stable != stable_state and state_buf.count(new_stable) >= (
            BUF_SIZE // 2 + 1
        )
        if state_changed:
            stable_state = new_stable  # flip the global flag

        # ★ UPDATED – update culprit whenever we are stably in wrong_ex
        if stable_state == "wrong_ex":
            culprit = mode(wrong_id_buf)
            if culprit is not None:
                last_wrong_pred = culprit
        elif stable_state == "good":
            correct_reps += 1
            last_wrong_pred = None
        elif stable_state == "bad_form":
            wrong_reps += 1
            last_wrong_pred = None

        # ───────── feedback text  (uses stable_state) ─────────
        if stable_state == "wrong_ex":
            culprit = last_wrong_pred or ex_pred  # fall back if ever None
            fb = f"Wrong exercise! Looks like {EXERCISE_MAP.get(culprit, '')}."
            sug = ""
        elif stable_state == "bad_form":
            fb = "You're doing it wrongly!"
            sug = build_advice(errs_abs)
        else:  # "good"
            fb = "You're on the right track!"
            sug = ""

        # ───────── histogram logic (unchanged) ─────────
        track = (time.time() - start_ts) >= 10.0
        if track and ex_match and stable_state == "bad_form":
            per_joint_sum += errs_abs
            per_joint_n += 1

        mean_so_far = (
            (per_joint_sum / per_joint_n).tolist()
            if per_joint_n
            else [0.0] * len(ERR_JOINTS)
        )

        mean_now = per_joint_sum / max(1, per_joint_n)
        top3_idx = np.argsort(mean_now)[::-1][:3]
        top3_labels = [JOINT_LABELS[i] for i in top3_idx]

        await ws.send_json(
            {
                "type": "progress",
                "feedback": fb,
                "suggestion": sug,
                "avg_error": float(errs_abs.mean()),
                "correct": correct_reps,
                "total": correct_reps + wrong_reps,
                "joint_errors": errs_abs.tolist(),
                "joint_errors_mean": mean_so_far,
                "top_joints": top3_labels,
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
