from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import torch
import numpy as np
import sqlite3
from datetime import datetime
import logging
from inference import infer_keypoints, DEVICE, N_ERR
from pathlib import Path
from model_loader import load_model
import sys
from fastapi.middleware.cors import CORSMiddleware

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
MODEL_PATH = "model/kp_pose_quality_windows_ex.pth"
model = None
try:
    sys.modules["__mp_main__"] = load_model
    model = load_model(state_dict_path=MODEL_PATH)
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}")
except Exception as e:
    logger.error(f"Unexpected error loading model: {e}")

# --- Database Setup ---
DB_PATH = "rehab_progress.db"
conn = None
cursor = None
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS progress
           (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, classification TEXT, deviation REAL, exercise_id INTEGER)"""
    )
    conn.commit()
    logger.info(f"Database connection established to '{DB_PATH}' and table verified.")
except sqlite3.Error as e:
    logger.error(f"Database error connecting to '{DB_PATH}': {e}")
    conn = None
    cursor = None

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Welcome to Rehab App Backend"}

@app.post("/start-exercise")
async def start_exercise():
    logger.info("Received request for /start-exercise")
    return {"message": "Exercise started acknowledgement"}

@app.get("/progress")
async def get_progress(exercise_id: int = None):
    if not conn or not cursor:
        logger.error("Attempted to get progress but database is not connected.")
        return {"error": "Database not available"}
    try:
        query = "SELECT timestamp, classification, deviation, exercise_id FROM progress"
        params = []
        if exercise_id is not None:
            query += " WHERE exercise_id = ?"
            params.append(exercise_id)
        query += " ORDER BY timestamp DESC LIMIT 100"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [
            {"timestamp": r[0], "classification": r[1], "deviation": r[2], "exercise_id": r[3]} for r in rows
        ]
    except sqlite3.Error as db_err:
        logger.error(f"Failed to fetch progress from DB: {db_err}")
        return {"error": "Failed to fetch progress data"}

# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host
    logger.info(f"WebSocket connection established from {client_host}")
    current_exercise_id = 1

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received raw data from {client_host}: {data[:200]}...")

            try:
                received_data = json.loads(data)
                logger.debug(f"Parsed data: {received_data}")

                if "action" in received_data:
                    action = received_data["action"]
                    logger.info(f"Received action: {action} from {client_host}")
                    if action == "start_exercise":
                        if "exercise_id" in received_data:
                            current_exercise_id = int(received_data["exercise_id"])
                            logger.info(f"Exercise ID set to {current_exercise_id} by client {client_host}")
                        await websocket.send_text(json.dumps({"message": f"Exercise {current_exercise_id} started."}))
                    elif action == "stop_exercise":
                        await websocket.send_text(json.dumps({"message": "Exercise stopped."}))

                elif "label" in received_data and received_data["label"] == "keypoint_sequence":
                    if model is None:
                        logger.warning("Model not loaded. Cannot perform inference.")
                        await websocket.send_text(json.dumps({"feedback": "Error", "suggestion": "Model not available.", "errors": [0.0]*N_ERR}))
                        continue

                    keypoints = received_data.get("keypoints")
                    ex_id_from_msg = received_data.get("exercise_id")
                    if ex_id_from_msg is not None:
                        try:
                            current_exercise_id = int(ex_id_from_msg)
                        except ValueError:
                            logger.warning(f"Invalid exercise_id '{ex_id_from_msg}'. Using last known: {current_exercise_id}")

                    if keypoints:
                        logger.info(f"Processing keypoint sequence for exercise {current_exercise_id} from {client_host}")
                        feedback, suggestion, err_values = infer_keypoints(model, keypoints, current_exercise_id)

                        errors_list = err_values.tolist() if isinstance(err_values, np.ndarray) else [0.0] * N_ERR
                        response = {
                            "feedback": feedback,
                            "suggestion": suggestion,
                            "errors": errors_list,
                            "exercise_id": current_exercise_id
                        }
                        await websocket.send_text(json.dumps(response))
                        logger.debug(f"Sent feedback to {client_host}: {response}")

                        if feedback != "Error" and conn and cursor:
                            try:
                                overall_deviation = np.mean(np.abs(err_values)) if N_ERR > 0 else 0.0
                                timestamp = datetime.now().isoformat()
                                cursor.execute(
                                    "INSERT INTO progress (timestamp, classification, deviation, exercise_id) VALUES (?, ?, ?, ?)",
                                    (timestamp, feedback, float(overall_deviation), current_exercise_id)
                                )
                                conn.commit()
                                logger.debug("Logged progress to database.")
                            except sqlite3.Error as db_err:
                                logger.error(f"Failed to log progress to DB: {db_err}")
                            except Exception as log_err:
                                logger.error(f"Unexpected error during DB logging: {log_err}")

                    else:
                        logger.warning(f"Received empty keypoints from {client_host}.")
                        await websocket.send_text(json.dumps({"feedback": "Info", "suggestion": "No keypoint data received.", "errors": [0.0]*N_ERR}))

                else:
                    logger.warning(f"Received unknown message structure from {client_host}: {received_data}")
                    await websocket.send_text(json.dumps({"message": "Unknown message format."}))

            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {client_host}: {data}")
                await websocket.send_text(json.dumps({"message": "Error: Invalid JSON received."}))
            except Exception as e:
                logger.exception(f"Error processing message from {client_host}: {e}")
                await websocket.send_text(json.dumps({"message": f"Error processing data: {e}"}))

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from {client_host}")
    except Exception as e:
        logger.error(f"Unexpected error in WebSocket connection with {client_host}: {e}", exc_info=True)
    finally:
        logger.info(f"Closing WebSocket connection handler for {client_host}")

# --- Cleanup on Shutdown ---
@app.on_event("shutdown")
def shutdown_event():
    if conn:
        conn.close()
        logger.info("Database connection closed.")