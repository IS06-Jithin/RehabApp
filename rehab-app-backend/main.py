# main.py
import os
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn

# ─── Constants and Model Setup ───
SCRIPT_DIR = Path().resolve()
CKPT_PATH = SCRIPT_DIR / "model/pose_quality_best.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")

EXERCISE_MAP = {1:"Arm-abduction", 2:"Arm-VW", 3:"Push-ups",
                4:"Leg-abduction", 5:"Lunge", 6:"Squat"}
NUM_EX = len(EXERCISE_MAP)

POSE_LANDMARK_NAMES = [  # standard 33 joints in MediaPipe
    'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
    'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
    'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
    'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
    'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
    'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
    'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
    'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
    'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
]
N_JOINTS = len(POSE_LANDMARK_NAMES)
SEQ_LEN = 16
IN_DIM = N_JOINTS * 3
VIS_THRESH = 0.8
TH_ERR_DEG = 10

ERR_JOINTS = [
    "LEFT_ELBOW","RIGHT_ELBOW","LEFT_SHOULDER","RIGHT_SHOULDER",
    "LEFT_HIP","RIGHT_HIP","LEFT_KNEE","RIGHT_KNEE",
    "SPINE","HEAD","LEFT_WRIST","RIGHT_WRIST","LEFT_ANKLE","RIGHT_ANKLE"
]
JOINT_LABELS = [j.replace('_', ' ').lower() for j in ERR_JOINTS]

# ─── Helper Definitions ───
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
    "RIGHT_ANKLE": (26, 28, 32)
}

def required_visible(visibility_array, exercise_id):
    """ Check if all required joints for a given exercise are visible """
    EXERCISE_JOINTS = {
        1: {"LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW"},
        2: {"LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "SPINE"},
        3: {"RIGHT_SHOULDER", "RIGHT_ELBOW", "SPINE"},
        4: {"LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "SPINE"},
        5: {"LEFT_HIP", "RIGHT_HIP", "RIGHT_KNEE"},
        6: {"LEFT_HIP", "RIGHT_HIP", "LEFT_KNEE", "RIGHT_KNEE", "SPINE"},
    }
    required_joints = EXERCISE_JOINTS.get(exercise_id, {"LEFT_SHOULDER", "RIGHT_SHOULDER", "SPINE"})

    indices = set()
    for joint in required_joints:
        if joint in JOINT_TRIPLETS:
            indices.update(JOINT_TRIPLETS[joint])

    return all(visibility_array[i] >= VIS_THRESH for i in indices)

# ─── Model Definitions ───
class KeypointEncoder(nn.Module):
    def __init__(self, in_dim: int, embed: int = 512):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, 128, 3, padding=1)
        self.conv2 = nn.Conv1d(128, embed, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = torch.relu(self.conv1(x.unsqueeze(2)))
        x = torch.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)

class PoseQualityNetKP(nn.Module):
    def __init__(self, in_dim: int, num_ex: int, hidden: int = 256, ex_emb: int = 64):
        super().__init__()
        self.encoder = KeypointEncoder(in_dim)
        self.lstm = nn.LSTM(512, hidden, num_layers=2, batch_first=True, bidirectional=True)
        feat_dim = hidden * 2
        self.ex_emb = nn.Sequential(
            nn.Linear(num_ex, ex_emb), nn.ReLU(),
            nn.Linear(ex_emb, ex_emb)
        )
        self.cls_head = nn.Linear(feat_dim + ex_emb, 2)
        self.err_head = nn.Linear(feat_dim + ex_emb, len(ERR_JOINTS))
        self.ex_head = nn.Linear(feat_dim, num_ex)

    def forward(self, seq, ex_1hot):
        B, T, _ = seq.shape
        feats = torch.stack([self.encoder(seq[:, t]) for t in range(T)], dim=1)
        out, _ = self.lstm(feats)
        g = out.mean(1)
        ex_e = self.ex_emb(ex_1hot)
        h = torch.cat([g, ex_e], dim=1)
        logits_q = self.cls_head(h)
        err_hat = self.err_head(h)
        logits_ex = self.ex_head(g)
        return logits_q, err_hat, logits_ex

# ─── Load model ───
if not CKPT_PATH.exists():
    raise FileNotFoundError(f"{CKPT_PATH} not found")
print("Loading model …")
model = torch.load(CKPT_PATH, map_location=DEVICE)
if isinstance(model, dict):
    m = PoseQualityNetKP(IN_DIM, NUM_EX).to(DEVICE)
    m.load_state_dict(model)
    model = m
model.eval()
print("✓ model ready")

# ─── FastAPI setup ───
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/infer")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    progress_history = []

    try:
        while True:
            data = await websocket.receive_json()

            if data.get("label") == "keypoint_sequence":
                keypoints = data["keypoints"]
                user_ex_id = int(data["exercise_id"])

                keypoints_np = np.array(keypoints, dtype=np.float32)
                if keypoints_np.shape != (SEQ_LEN, N_JOINTS, 3):
                    await websocket.send_json({
                        "feedback": "Keypoints shape invalid",
                        "suggestion": "",
                        "progress": {"timestamps": [], "deviations": []}
                    })
                    continue

                visibilities = np.array([kp[2] for kp in keypoints_np[-1]])
                seq = torch.tensor(keypoints_np.reshape(SEQ_LEN, -1),
                                   dtype=torch.float32, device=DEVICE).unsqueeze(0)
                ex_1h = F.one_hot(torch.tensor([user_ex_id-1], device=DEVICE), NUM_EX).float()

                if not required_visible(visibilities, user_ex_id):
                    feedback = "Adjust your posture"
                    suggestion = ""

                    await websocket.send_json({
                        "feedback": feedback,
                        "suggestion": suggestion,
                        "progress": {"timestamps": [], "deviations": []}
                    })
                    continue  # ← Very important! Don't proceed if posture not correct

                # Now only if visible -> infer
                with torch.no_grad():
                    log_q, err_hat, log_ex = model(seq, ex_1h)

                q_pred = log_q.argmax(1).item()
                ex_pred = log_ex.argmax(1).item() + 1
                errs = err_hat.squeeze().cpu().numpy()

                if ex_pred != user_ex_id:
                    pred_name = EXERCISE_MAP.get(ex_pred, f"Exercise {ex_pred}")
                    feedback = f"Wrong exercise! It looks like you're doing {pred_name}."
                    suggestion = ""
                else:
                    if q_pred == 1:
                        feedback = "You're on the right track!"
                        suggestion = ""
                    else:
                        feedback = "You're doing it wrongly!"
                        bad_idxs = np.argsort(np.abs(errs))[::-1][:3]
                        joints = [JOINT_LABELS[i] for i in bad_idxs if abs(errs[i]) >= TH_ERR_DEG]
                        if joints:
                            if len(joints) == 1:
                                joint_str = joints[0]
                            elif len(joints) == 2:
                                joint_str = f"{joints[0]} and {joints[1]}"
                            else:
                                joint_str = f"{joints[0]}, {joints[1]} and {joints[2]}"
                            suggestion = f"Please adjust your {joint_str} properly."
                        else:
                            suggestion = "Check your form."

                avg_dev = float(np.mean(np.abs(errs)))
                progress_history.append({
                    "timestamp": int(time.time() * 1000),
                    "deviation": avg_dev,
                    "classification": feedback
                })

                await websocket.send_json({
                    "feedback": feedback,
                    "suggestion": suggestion,
                    "progress": {
                        "timestamps": [p["timestamp"] for p in progress_history][-20:],
                        "deviations": [p["deviation"] for p in progress_history][-20:],
                    }
                })

    except Exception as e:
        print(f"WebSocket closed: {e}")
