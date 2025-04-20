

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json
import torch
import numpy as np # Import numpy
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
# --- End Logging Setup ---

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model at App Startup ---
MODEL_PATH = "model/kp_pose_quality_windows_ex.pth"
model = None
try:
    sys.modules["__mp_main__"] = load_model
    model = load_model(state_dict_path=MODEL_PATH)
except FileNotFoundError:
    logger.error(f"Model file not found at {MODEL_PATH}…")
except Exception as e:
    logger.error(f"Unexpected error loading model: {e}")


# --- Database Setup  ---
DB_PATH = "rehab_progress.db"
conn = None
cursor = None
try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS progress
                    (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, classification TEXT, deviation REAL)"""
    )
    conn.commit()
    logger.info(f"Database connection established to '{DB_PATH}' and table verified.")
except sqlite3.Error as e:
    logger.error(f"Database error connecting to '{DB_PATH}': {e}")
    # Handle DB connection failure gracefully
    conn = None
    cursor = None


# --- API Endpoints (Keep / and /progress as is) ---
@app.get("/")
async def root():
    return {"message": "Welcome to Rehab App Backend"}


@app.post("/start-exercise")
async def start_exercise():
    # This endpoint might not be strictly needed if using WebSocket 'action' messages
    logger.info("Received request for /start-exercise")
    return {"message": "Exercise started acknowledgement"}


# --- WebSocket for Exercise Feedback ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_host = websocket.client.host
    logger.info(f"WebSocket connection established from {client_host}")
    current_exercise_id = 1 # Default exercise ID

    try:
        while True:
            data = await websocket.receive_text()
            logger.debug(f"Received raw data from {client_host}: {data[:200]}...") # Log snippet

            try:
                received_data = json.loads(data)
                logger.debug(f"Parsed data: {received_data}")

                # Handle different message types (actions vs keypoints)
                if "action" in received_data:
                    action = received_data["action"]
                    logger.info(f"Received action: {action} from {client_host}")
                    if action == "start_exercise":
                        # Reset state or log start if needed
                        # You might receive exercise_id here too
                        if "exercise_id" in received_data:
                             current_exercise_id = int(received_data["exercise_id"])
                             logger.info(f"Exercise ID set to {current_exercise_id} by client {client_host}")
                        await websocket.send_text(json.dumps({"message": f"Exercise {current_exercise_id} started."}))
                    elif action == "stop_exercise":
                        # Log stop if needed
                         await websocket.send_text(json.dumps({"message": "Exercise stopped."}))
                    # Add more actions if needed

                elif "label" in received_data and received_data["label"] == "keypoint_sequence":
                    if model is None:
                         logger.warning("Model not loaded. Cannot perform inference.")
                         await websocket.send_text(json.dumps({"feedback": "Error", "suggestion": "Model not available.", "errors": [0.0]*N_ERR}))
                         continue # Skip inference if model isn't loaded

                    keypoints = received_data.get("keypoints")
                    # --- Get exercise_id from the message if available, otherwise use current ---
                    # This assumes frontend sends exercise_id WITH keypoints if it changes
                    ex_id_from_msg = received_data.get("exercise_id")
                    if ex_id_from_msg is not None:
                        try:
                            current_exercise_id = int(ex_id_from_msg)
                        except ValueError:
                            logger.warning(f"Invalid exercise_id '{ex_id_from_msg}' received in keypoint message. Using last known: {current_exercise_id}")


                    if keypoints:
                        logger.info(f"Processing keypoint sequence for exercise {current_exercise_id} from {client_host}")
                        # Call the corrected inference function
                        feedback, suggestion, err_values = infer_keypoints(model, keypoints, current_exercise_id)

                        # Ensure err_values is serializable (should be numpy array from inference)
                        if isinstance(err_values, np.ndarray):
                            errors_list = err_values.tolist()
                        else:
                            logger.warning(f"err_values is not a numpy array: {type(err_values)}. Sending zeros.")
                            errors_list = [0.0] * N_ERR # Default error list

                        response = {
                            "feedback": feedback,
                            "suggestion": suggestion,
                            "errors": errors_list
                        }
                        await websocket.send_text(json.dumps(response))
                        logger.debug(f"Sent feedback to {client_host}: {response}")

                        # --- Optional: Database Logging ---
                        if feedback != "Error" and conn and cursor:
                            try:
                                # Calculate overall deviation (e.g., mean absolute error)
                                overall_deviation = np.mean(np.abs(err_values)) if N_ERR > 0 else 0.0
                                timestamp = datetime.now().isoformat()
                                cursor.execute(
                                    "INSERT INTO progress (timestamp, classification, deviation) VALUES (?, ?, ?)",
                                    (timestamp, feedback, float(overall_deviation)) # Ensure deviation is float
                                )
                                conn.commit()
                                logger.debug("Logged progress to database.")
                            except sqlite3.Error as db_err:
                                logger.error(f"Failed to log progress to DB: {db_err}")
                            except Exception as log_err:
                                logger.error(f"Unexpected error during DB logging: {log_err}")
                        # --- End DB Logging ---

                    else:
                        logger.warning(f"Received keypoint_sequence label but 'keypoints' field was missing or empty from {client_host}.")
                        await websocket.send_text(json.dumps({"feedback": "Info", "suggestion": "No keypoint data received.", "errors": [0.0]*N_ERR}))
                else:
                     logger.warning(f"Received unknown message structure from {client_host}: {received_data}")
                     await websocket.send_text(json.dumps({"message": "Unknown message format."}))


            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from {client_host}: {data}")
                await websocket.send_text(json.dumps({"message": "Error: Invalid JSON received."}))
            except Exception as e:
                logger.exception(f"Error processing message from {client_host}: {e}") # Log full traceback
                try:
                    await websocket.send_text(json.dumps({"message": f"Error processing data: {e}"}))
                except Exception as send_err:
                     logger.error(f"Failed to send error message back to client {client_host}: {send_err}")


    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected from {client_host}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in the WebSocket connection with {client_host}: {e}", exc_info=True)
    finally:
        logger.info(f"Closing WebSocket connection handler for {client_host}")
        # Any cleanup specific to this connection could go here 


@app.get("/progress")
async def get_progress():
    if not conn or not cursor:
        logger.error("Attempted to get progress but database is not connected.")
        return {"error": "Database not available"}
    try:
        cursor.execute(
            "SELECT timestamp, classification, deviation FROM progress ORDER BY timestamp DESC LIMIT 100"
        )
        rows = cursor.fetchall()
        # Format results for JSON response
        return [
            {"timestamp": r[0], "classification": r[1], "deviation": r[2]} for r in rows
        ]
    except sqlite3.Error as db_err:
        logger.error(f"Failed to fetch progress from DB: {db_err}")
        return {"error": "Failed to fetch progress data"}


# Optional: Add cleanup for DB connection on shutdown
@app.on_event("shutdown")
def shutdown_event():
    if conn:
        conn.close()
        logger.info("Database connection closed.")