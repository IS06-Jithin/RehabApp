"""
Rehab App Backend

This module serves as the backend for the Rehab App, providing API endpoints for
exercise management and real-time feedback.
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import torch
import torch.nn as nn
import sqlite3
from datetime import datetime
import logging  # Import logging

from fastapi.middleware.cors import CORSMiddleware

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Model Definition ---
# Corrected for 132 features (33 landmarks * 4 coords [x,y,z,v])
class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        input_features=132,
        cnn_out_channels=64,
        lstm_hidden_size=50,
        num_classes=2,
    ):
        super(CNNLSTMModel, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_features, cnn_out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(
                2
            ),  # This halves the sequence length dimension for the output channels
        )
        # Calculate the effective sequence length after MaxPool1d if needed, but LSTM input size is channel based
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, batch_first=True)
        self.fc = nn.Linear(
            lstm_hidden_size, num_classes
        )  # Output: [classification, deviation_score]

    def forward(self, x):
        # Input x shape: [batch_size, seq_len, features] = [1, 10, 132]
        x = x.permute(0, 2, 1)  # [batch, features, seq_len] = [1, 132, 10] for CNN
        x = self.cnn(
            x
        )  # Output shape depends on padding/stride, channels = cnn_out_channels (64)
        # Example shape after CNN+Pool: [1, 64, 5] (if seq_len halved by pooling)
        x = x.permute(
            0, 2, 1
        )  # [batch, new_seq_len, cnn_out_channels] = [1, 5, 64] for LSTM
        x, _ = self.lstm(x)  # Output shape: [1, 5, lstm_hidden_size=50]
        # Use the output of the last time step
        x = self.fc(x[:, -1, :])  # Shape: [1, num_classes=2]
        return x


# --- Initialize Model ---
model = CNNLSTMModel(input_features=132)  # Use 132 features
model.eval()
logger.info("Model initialized (expecting 132 features).")
# Consider loading pre-trained weights here:
# try:
#     model.load_state_dict(torch.load("your_model_weights.pth"))
#     logger.info("Loaded pre-trained model weights.")
# except FileNotFoundError:
#     logger.warning("Model weights file not found. Using initialized weights.")
# except Exception as e:
#      logger.error(f"Error loading model weights: {e}")


# --- Database Setup ---
try:
    conn = sqlite3.connect("rehab_progress.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS progress
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, classification TEXT, deviation REAL)"""
    )
    conn.commit()
    logger.info("Database connection established and table verified.")
except sqlite3.Error as e:
    logger.error(f"Database error: {e}")
    # Handle DB connection failure gracefully? Exit?
    conn = None  # Indicate DB failure


# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to Rehab App Backend"}


@app.post("/start-exercise")
async def start_exercise():
    # This endpoint might not be strictly needed if using WebSocket 'action' messages
    logger.info("Received request for /start-exercise")
    return {"message": "Exercise started acknowledgement"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(
        f"WebSocket connection accepted from {websocket.client.host}:{websocket.client.port}"
    )
    buffer = []  # Buffer to hold the sequence of flattened frames
    sequence_length = 10  # Number of frames required for model input

    try:
        while True:
            data = await websocket.receive_text()
            try:
                # Assume data is the list of frames sent by frontend
                received_frames = json.loads(data)

                # Check if it's a control message (like stop/start) instead of keypoints
                if isinstance(received_frames, dict) and "action" in received_frames:
                    logger.info(f"Received action: {received_frames['action']}")
                    # Handle actions if needed (e.g., reset buffer)
                    if received_frames["action"] == "start_exercise":
                        buffer = []  # Clear buffer on new start
                    continue  # Skip keypoint processing for action messages

                # Ensure received_frames is a list (sequence)
                if not isinstance(received_frames, list):
                    logger.warning(
                        f"Received non-list data, skipping: {type(received_frames)}"
                    )
                    continue

                # Process each frame in the received sequence
                # (Frontend sends 10 frames at once, so this loop runs 10 times)
                for frame in received_frames:
                    # Ensure frame is a list of landmarks
                    if not isinstance(frame, list):
                        logger.warning(
                            f"Invalid frame data type in sequence: {type(frame)}"
                        )
                        continue  # Skip this frame

                    # --- Corrected Flattening ---
                    # Flatten landmarks within *this* frame
                    # Each landmark is [x, y, z, v] -> 4 coords
                    flat_keypoints_for_frame = []
                    for landmark in frame:
                        if isinstance(landmark, list) and len(landmark) == 4:
                            flat_keypoints_for_frame.extend(landmark)
                        else:
                            logger.warning(f"Invalid landmark data: {landmark}")
                            # Handle error: maybe append zeros or skip frame?
                            # Appending zeros to maintain structure:
                            flat_keypoints_for_frame.extend([0.0] * 4)

                    # Check if flattened frame has the correct number of features (33*4 = 132)
                    if len(flat_keypoints_for_frame) != 132:
                        logger.warning(
                            f"Incorrect feature count for frame: {len(flat_keypoints_for_frame)}, expected 132. Skipping frame."
                        )
                        continue  # Skip this frame

                    # Add the correctly flattened frame to the buffer
                    buffer.append(flat_keypoints_for_frame)

                    # Maintain buffer size
                    if len(buffer) > sequence_length:
                        buffer.pop(0)

                    # --- Model Prediction (if buffer is full) ---
                    if len(buffer) == sequence_length:
                        logger.debug(
                            f"Buffer full ({len(buffer)} frames). Running prediction."
                        )
                        # Prepare input tensor: [batch_size, seq_len, features]
                        input_tensor = torch.tensor(
                            [buffer], dtype=torch.float32
                        )  # Shape: [1, 10, 132]

                        feedback = {}  # Prepare feedback dict
                        try:
                            with torch.no_grad():
                                output = model(input_tensor)
                                # Assuming output[0] is tensor([class_score, dev_score])
                                classification_score, deviation_score = output[
                                    0
                                ].tolist()

                            # Dummy logic for classification/deviation
                            is_correct = classification_score > 0.5  # Example threshold
                            deviation = max(0, deviation_score)  # Ensure non-negative

                            logger.debug(
                                f"Prediction - Correct: {is_correct}, Deviation: {deviation:.2f}"
                            )

                            # Generate feedback message
                            if not is_correct or deviation > 10:  # Example threshold
                                feedback["message"] = (
                                    "Straighten your back"  # Simplified feedback
                                )
                                feedback["deviations"] = {
                                    "torso": deviation
                                }  # Example deviation key
                            else:
                                feedback["message"] = "Correct pose"

                            # Store progress in DB (if connected)
                            if conn:
                                try:
                                    timestamp = datetime.now().isoformat()
                                    cursor.execute(
                                        "INSERT INTO progress (timestamp, classification, deviation) VALUES (?, ?, ?)",
                                        (
                                            timestamp,
                                            "correct" if is_correct else "incorrect",
                                            deviation,
                                        ),
                                    )
                                    conn.commit()
                                except sqlite3.Error as db_err:
                                    logger.error(
                                        f"Failed to insert progress into DB: {db_err}"
                                    )
                            else:
                                logger.warning(
                                    "Database not connected. Progress not saved."
                                )

                        except Exception as model_err:
                            logger.error(
                                f"Error during model prediction or processing: {model_err}"
                            )
                            feedback["message"] = "Error analyzing pose."

                        # Send feedback (even if error occurred)
                        await websocket.send_text(json.dumps(feedback))
                        logger.debug(f"Sent feedback: {feedback}")

                        # Optionally clear buffer after prediction or let it slide
                        # buffer.pop(0) # If using sliding window where each frame triggers prediction once full

            except json.JSONDecodeError:
                logger.warning(f"Received invalid JSON data: {data}")
                await websocket.send_text(
                    json.dumps({"message": "Error: Invalid data format received."})
                )
            except Exception as e:
                logger.error(
                    f"Error processing received data: {e}", exc_info=True
                )  # Log full traceback
                # Send generic error message to frontend
                await websocket.send_text(
                    json.dumps({"message": "An error occurred on the server."})
                )

    except WebSocketDisconnect:
        logger.info(
            f"WebSocket disconnected from {websocket.client.host}:{websocket.client.port}"
        )
    except Exception as e:
        # Catch unexpected errors in the main loop
        logger.error(f"Unexpected error in WebSocket handler: {e}", exc_info=True)
    finally:
        # Optional: Log final disconnect reason if available
        logger.info("WebSocket endpoint finished.")


@app.get("/progress")
async def get_progress():
    if not conn:
        return {"error": "Database not connected"}
    try:
        cursor.execute(
            "SELECT timestamp, classification, deviation FROM progress ORDER BY timestamp DESC LIMIT 100"  # Get recent progress
        )
        rows = cursor.fetchall()
        return [
            {"timestamp": r[0], "classification": r[1], "deviation": r[2]} for r in rows
        ]
    except sqlite3.Error as db_err:
        logger.error(f"Failed to fetch progress from DB: {db_err}")
        return {"error": "Failed to fetch progress"}


# Optional: Add cleanup for DB connection on shutdown
# @app.on_event("shutdown")
# def shutdown_event():
#     if conn:
#         conn.close()
#         logger.info("Database connection closed.")
