"""
Rehab App Backend

This module serves as the backend for the Rehab App, providing API endpoints for
exercise management and real-time feedback.
"""

from fastapi import FastAPI, WebSocket
import json
import torch
import torch.nn as nn
import sqlite3
from datetime import datetime

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dummy CNN-LSTM Model (replace with your actual model)
class CNNLSTMModel(nn.Module):
    def __init__(self):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(99, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.lstm = nn.LSTM(64, 50, batch_first=True)  # corrected to 64 here
        self.fc = nn.Linear(50, 2)  # Output: [classification, deviation_score]

    def forward(self, x):
        batch_size, seq_len, features = x.size()
        x = x.permute(0, 2, 1)  # [batch, features, seq_len] for CNN
        x = self.cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, features] for LSTM
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take last LSTM output
        return x


model = CNNLSTMModel()
model.eval()  # Assume model is trained; load weights if available

# SQLite database setup
conn = sqlite3.connect("rehab_progress.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute(
    """CREATE TABLE IF NOT EXISTS progress
                  (id INTEGER PRIMARY KEY, timestamp TEXT, classification TEXT, deviation REAL)"""
)
conn.commit()


@app.get("/")
async def root():
    return {"message": "Welcome to Rehab App Backend"}


@app.post("/start-exercise")
async def start_exercise():
    return {"message": "Exercise started"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    buffer = []
    sequence_length = 10

    while True:
        data = await websocket.receive_text()
        keypoints = json.loads(data)
        # Flatten keypoints: 33 landmarks * 3 coords = 99 features
        flat_keypoints = [coord for landmark in keypoints for coord in landmark]
        buffer.append(flat_keypoints)

        if len(buffer) > sequence_length:
            buffer.pop(0)

        if len(buffer) == sequence_length:
            # Prepare input tensor
            input_tensor = torch.tensor([buffer], dtype=torch.float32)  # [1, 10, 99]
            with torch.no_grad():
                output = model(input_tensor)
                classification_score, deviation_score = output[0].tolist()
                is_correct = classification_score > 0  # Dummy threshold
                deviation = max(0, deviation_score)  # Ensure non-negative

            # Generate feedback
            feedback = {}
            if not is_correct or deviation > 10:  # Example threshold
                feedback["message"] = "Straighten your back"  # Simplified feedback
                feedback["deviations"] = {"torso": deviation}
            else:
                feedback["message"] = "Correct pose"

            # Store progress
            timestamp = datetime.now().isoformat()
            cursor.execute(
                "INSERT INTO progress (timestamp, classification, deviation) VALUES (?, ?, ?)",
                (timestamp, "correct" if is_correct else "incorrect", deviation),
            )
            conn.commit()

            # Send feedback to frontend
            await websocket.send_text(json.dumps(feedback))


@app.get("/progress")
async def get_progress():
    cursor.execute(
        "SELECT timestamp, classification, deviation FROM progress ORDER BY timestamp"
    )
    rows = cursor.fetchall()
    return [
        {"timestamp": r[0], "classification": r[1], "deviation": r[2]} for r in rows
    ]
