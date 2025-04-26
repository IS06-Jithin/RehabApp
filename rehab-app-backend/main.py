import math
import time
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    return f"Please adjust your {joint_str} properly."


# ────────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint (accepts the *same payload* the front‑end sends)─────────────
# -------------------------------------------------------------------------------


@app.websocket("/ws/infer")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    progress_history: list[dict] = []

    while True:
        try:
            data = await ws.receive_json()
        except Exception:
            break  # client disconnected

        label = data.get("label")

        # ------------------------------------------------------------------
        # The React client only sends one type: "keypoint_sequence"
        # containing a *full* 16‑frame tensor plus exercise_id.
        # ------------------------------------------------------------------
        if label != "keypoint_sequence":
            await ws.send_json(
                {
                    "feedback": "Unknown message label",
                    "suggestion": "",
                    "progress": {"timestamps": [], "deviations": []},
                }
            )
            continue

        keypoints = data.get("keypoints", [])
        exercise_id = int(data.get("exercise_id", 0))
        if exercise_id not in EXERCISE_MAP:
            await ws.send_json(
                {
                    "feedback": "Invalid exercise ID",
                    "suggestion": "",
                    "progress": {"timestamps": [], "deviations": []},
                }
            )
            continue

        # shape check --------------------------------------------------------
        kp_np = np.asarray(keypoints, dtype=np.float32)
        if kp_np.shape != (SEQ_LEN, N_JOINTS, 3):
            await ws.send_json(
                {
                    "feedback": "Keypoints shape invalid",
                    "suggestion": "",
                    "progress": {"timestamps": [], "deviations": []},
                }
            )
            continue

        # inference ----------------------------------------------------------
        seq_tensor = torch.tensor(
            kp_np.reshape(SEQ_LEN, -1), dtype=torch.float32, device=DEVICE
        ).unsqueeze(0)
        ex_1h = F.one_hot(
            torch.tensor([exercise_id - 1], device=DEVICE), NUM_EX
        ).float()

        with torch.no_grad():
            log_q, err_hat, log_ex = model(seq_tensor, ex_1h)

        q_pred = log_q.argmax(1).item()
        ex_pred = log_ex.argmax(1).item() + 1
        errs = err_hat.squeeze().cpu().numpy()

        # feedback construction ---------------------------------------------
        if ex_pred != exercise_id:
            pred_name = EXERCISE_MAP.get(ex_pred, f"Ex {ex_pred}")
            feedback = f"Wrong exercise! It looks like you're doing {pred_name}."
            suggestion = ""
        else:
            if q_pred == 1:
                feedback = "You're on the right track!"
                suggestion = ""
            else:
                feedback = "You're doing it wrongly!"
                suggestion = build_advice(errs)

        avg_dev = float(np.mean(np.abs(errs)))
        progress_history.append(
            {
                "timestamp": int(time.time() * 1000),
                "deviation": avg_dev,
                "classification": feedback,
            }
        )

        await ws.send_json(
            {
                "feedback": feedback,
                "suggestion": suggestion,
                "progress": {
                    "timestamps": [p["timestamp"] for p in progress_history][-20:],
                    "deviations": [p["deviation"] for p in progress_history][-20:],
                },
            }
        )