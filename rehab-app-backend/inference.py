# inference.py

import torch
import numpy as np
import torch.nn.functional as F
import json
import logging # Import logging

logger = logging.getLogger(__name__) # Get logger instance

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EXERCISES = 6  # Make sure this matches training and frontend selection
SEQUENCE_LENGTH = 16 # Expected number of frames in the sequence
NUM_JOINTS = 33    # Expected number of joints per frame
NUM_COORDS = 3     # Expected coordinates per joint (x, y, z)
EXPECTED_DIM_PER_FRAME = NUM_JOINTS * NUM_COORDS # Should be 99

ERR_JOINTS = [
    "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_SHOULDER", "RIGHT_SHOULDER",
    "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE",
    "SPINE", "HEAD", # Ensure this order matches training if applicable
]

N_ERR = len(ERR_JOINTS)
# --- End Constants ---

# --- Inference Logic (Corrected) ---
def infer_keypoints(model, keypoints_sequence, exercise_id):
    """
    Performs inference on a sequence of keypoints.

    Args:
        model: The loaded PyTorch model.
        keypoints_sequence: A list of lists of lists representing the keypoint sequence.
                           Expected structure: [frame1, frame2, ..., frameT]
                           where frame = [joint1, joint2, ..., jointN]
                           and joint = [x, y, z].
                           Expected dimensions: (SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS) -> (16, 33, 3)
        exercise_id: The ID of the exercise being performed (1-based).

    Returns:
        A tuple containing:
        - feedback (str): "Correct" or "Incorrect".
        - suggestion (str): Specific correction advice if incorrect, empty otherwise.
        - err_values (np.ndarray): Array of error values for specified joints.
    """
    if not isinstance(keypoints_sequence, list):
        logger.error(f"Invalid keypoints_sequence type: {type(keypoints_sequence)}. Expected list.")
        return "Error", "Invalid keypoint data type received.", np.zeros(N_ERR)

    if len(keypoints_sequence) != SEQUENCE_LENGTH:
        logger.error(f"Invalid sequence length: {len(keypoints_sequence)}. Expected {SEQUENCE_LENGTH}.")
        return "Error", f"Received {len(keypoints_sequence)} frames, expected {SEQUENCE_LENGTH}.", np.zeros(N_ERR)

    try:
        # 1. Convert to NumPy array and validate shape
        keypoints_array = np.array(keypoints_sequence, dtype=np.float32)
        if keypoints_array.shape != (SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS):
            logger.error(
                f"Unexpected shape: {keypoints_array.shape}. "
                f"Expected {(SEQUENCE_LENGTH, NUM_JOINTS, NUM_COORDS)}."
            )
            return "Error", "Incorrect keypoint data structure.", np.zeros(N_ERR)

        # 2. Flatten each frame: (16, 33, 3) -> (16, 99)
        keypoints_flat = keypoints_array.reshape(SEQUENCE_LENGTH, -1)
        if keypoints_flat.shape[1] != EXPECTED_DIM_PER_FRAME:
            logger.error(
                f"Unexpected flattened dim: {keypoints_flat.shape[1]}. "
                f"Expected {EXPECTED_DIM_PER_FRAME}."
            )
            return "Error", "Keypoint flattening failed.", np.zeros(N_ERR)

        # 3. Determine the device from the model
        device = next(model.parameters()).device

        # 4. Build sequence tensor on that device: (1, 16, 99)
        seq_tensor = (
            torch.from_numpy(keypoints_flat)
            .unsqueeze(0)
            .to(device)
        )

        # 5. Validate and one-hot encode exercise ID on same device
        if not (1 <= exercise_id <= NUM_EXERCISES):
            logger.error(f"Invalid exercise_id: {exercise_id}. Must be 1–{NUM_EXERCISES}.")
            exercise_id = 1  # fallback

        ex_index = torch.tensor([exercise_id - 1], device=device)
        ex_1hot = F.one_hot(ex_index, num_classes=NUM_EXERCISES).float()  # (1, NUM_EXERCISES)

        # 6. Inference
        model.eval()
        with torch.no_grad():
            logits, err_hat = model(seq_tensor, ex_1hot)
            predicted_class = logits.argmax(dim=1).item()
            err_values = err_hat.squeeze(0).cpu().numpy()

        # 7. Build feedback
        feedback = "Correct" if predicted_class == 1 else "Incorrect"
        suggestion = ""
        if predicted_class == 0 and N_ERR > 0:
            if len(err_values) == len(ERR_JOINTS):
                idx = np.argmax(np.abs(err_values))
                joint = ERR_JOINTS[idx].replace("_", " ")
                suggestion = f"Check your {joint} position."
            else:
                logger.warning(
                    f"err_values length {len(err_values)} != ERR_JOINTS {len(ERR_JOINTS)}"
                )
                suggestion = "Please check your overall form."

        logger.info(f"Inference result – Feedback: {feedback}, Suggestion: '{suggestion}'")
        return feedback, suggestion, err_values

    except Exception as e:
        logger.exception(f"Error during inference: {e}")
        return "Error", "An error occurred during inference.", np.zeros(N_ERR)
