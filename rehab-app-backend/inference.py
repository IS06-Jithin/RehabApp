# inference.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from mediapipe.python.solutions.pose import PoseLandmark
# Import necessary items from model_loader
# Ensure model_loader.py is in the same directory or Python path
from model_loader import PoseQualityNetKP, KeypointEncoder, NUM_EXERCISES

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available() else
    torch.device("cuda") if torch.cuda.is_available() else
    torch.device("cpu")
)

# Exercise Info
EXERCISE_MAP = {
    1: "Arm abduction", 2: "Arm VW", 3: "Push-ups",
    4: "Leg abduction", 5: "Leg lunge", 6: "Squats"
}
# NUM_EXERCISES is imported from model_loader

# Joint Info
JOINT_NAMES = [lm.name for lm in PoseLandmark]
N_JOINTS = len(JOINT_NAMES) # Should be 33

JOINT_TRIPLETS = {
    "LEFT_ELBOW":   (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    "RIGHT_ELBOW":  (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    "LEFT_SHOULDER":  (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
    "RIGHT_SHOULDER": (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    "LEFT_HIP":   (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
    "RIGHT_HIP":  (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
    "LEFT_KNEE":  (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
    "RIGHT_KNEE": (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
    "SPINE":      (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
    "HEAD":       (PoseLandmark.LEFT_SHOULDER, PoseLandmark.NOSE, PoseLandmark.RIGHT_SHOULDER),
    "LEFT_WRIST": (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_INDEX),
    "RIGHT_WRIST":(PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST, PoseLandmark.RIGHT_INDEX),
    "LEFT_ANKLE": (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_FOOT_INDEX),
    "RIGHT_ANKLE":(PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_FOOT_INDEX),
}
ERR_JOINTS = list(JOINT_TRIPLETS.keys())
N_ERR = len(ERR_JOINTS) # Should be 14

# --- Constants related to original script's visibility checks (Not used in this function) ---
# These landmark sets were used in the original script's `extract_keypoints` function
# to determine visibility (`all_required_visible`) *before* calling the model.
# Since this `infer_keypoints` function does not receive visibility data, these
# specific checks cannot be replicated here. The model itself uses the same input
# features regardless, differentiating exercises via the `ex_1hot` tensor.
REQUIRED_LANDMARK_INDICES_FOR_ERRORS = set()
for _, landmarks in JOINT_TRIPLETS.items():
    for landmark in landmarks:
        REQUIRED_LANDMARK_INDICES_FOR_ERRORS.add(landmark.value)

PUSHUP_JOINT_TRIPLETS = { # Original subset for Push-ups (Ex ID 3) visibility
    "RIGHT_ELBOW": (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
    "RIGHT_SHOULDER": (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
    "SPINE": (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
}
PUSHUP_REQUIRED_LANDMARKS = set()
for landmarks in PUSHUP_JOINT_TRIPLETS.values():
    for landmark in landmarks:
        PUSHUP_REQUIRED_LANDMARKS.add(landmark.value)
# --- End constants related to visibility ---


# Inference Parameters
SEQUENCE_LENGTH = 16
NUM_COORDS = 3 # x, y, z world landmarks
ERROR_THRESHOLD = 1e-3 # Threshold for considering an error significant enough for suggestion
IN_DIM = N_JOINTS * NUM_COORDS #3 coordinates


# --- Core Inference Logic ---
def infer_keypoints(model: PoseQualityNetKP,
                    keypoints_sequence: list | np.ndarray,
                    exercise_id: int):
    """
    Performs inference on a sequence of 3D world keypoints using the provided model.

    Note: This function replicates the model inference and feedback generation
    logic based *only* on the keypoint data and exercise ID. It does *not*
    replicate the visibility checks (including the push-up specific one) from
    the original standalone script's `extract_keypoints` function, as it does
    not receive visibility data as input.

    Args:
        model: The loaded and initialized PoseQualityNetKP model, set to eval mode.
        keypoints_sequence: A list or NumPy array expected to represent 16 frames
                           of 33 joints with (x, y, z) world coordinates.
                           Expected shape: (SEQUENCE_LENGTH, N_JOINTS, NUM_COORDS).
        exercise_id: Integer ID of the exercise (1 to NUM_EXERCISES).

    Returns:
        Tuple[str, str, np.ndarray]:
            - feedback (str): "Correct", "Incorrect", "World landmarks missing" (if input is all zeros),
                              or "Error" (for validation/processing issues).
            - suggestion (str): Corrective suggestion if feedback is "Incorrect", otherwise empty.
            - err_values (np.ndarray): Array of predicted error values (shape: (N_ERR,)).
                                       Zeros if inference couldn't run or input was invalid.
    """
    feedback = "Error"  # Default state indicating a processing problem
    suggestion = ""
    err_values = np.zeros(N_ERR, dtype=np.float32)

    try:
        # 1. Validate keypoints_sequence type and length
        if not isinstance(keypoints_sequence, (list, np.ndarray)):
            logger.error(f"Invalid keypoints_sequence type: {type(keypoints_sequence)}. Expected list or ndarray.")
            return feedback, "Invalid keypoint data type.", err_values

        if len(keypoints_sequence) != SEQUENCE_LENGTH:
            logger.error(f"Invalid sequence length: {len(keypoints_sequence)}. Expected {SEQUENCE_LENGTH}.")
            return feedback, f"Requires {SEQUENCE_LENGTH} frames, received {len(keypoints_sequence)}.", err_values

        # 2. Convert to NumPy array and validate shape
        # Expects input like: [[ [x,y,z], [x,y,z], ... 33 joints ], ... 16 frames]
        keypoints_array = np.array(keypoints_sequence, dtype=np.float32)
        expected_shape = (SEQUENCE_LENGTH, N_JOINTS, NUM_COORDS)
        if keypoints_array.shape != expected_shape:
            logger.error(f"Unexpected keypoints shape: {keypoints_array.shape}. Expected {expected_shape}.")
            # Optionally, handle pre-flattened input if absolutely necessary, but it's cleaner if the API caller sends the structured data.
            # if keypoints_array.ndim == 2 and keypoints_array.shape == (SEQUENCE_LENGTH, IN_DIM): ...
            return feedback, "Incorrect keypoint data structure.", err_values

        # 3. Check for valid keypoints (non-NaN, finite values)
        if not np.all(np.isfinite(keypoints_array)):
            logger.error("Keypoints contain NaN or infinite values.")
            return feedback, "Invalid keypoint values.", err_values

        # 4. Validate exercise ID
        if not isinstance(exercise_id, int) or not (1 <= exercise_id <= NUM_EXERCISES):
            logger.error(f"Invalid exercise_id: {exercise_id}. Must be int between 1 and {NUM_EXERCISES}.")
            return feedback, f"Invalid exercise ID ({exercise_id}).", err_values

        # 5. Check if keypoints are valid (not all zeros) - Simulates missing world landmarks
        # This check corresponds to the state where MediaPipe might detect a pose
        # but fails to provide world landmarks, or if the input data itself is zeroed out.
        if np.all(keypoints_array == 0):
            logger.warning("Keypoints array contains only zeros. Simulating missing world landmarks.")
            return "World landmarks missing", "Valid 3D keypoints not detected.", err_values

        # --- Data Preparation for Model ---

        # 6. Flatten each frame: (16, 33, 3) -> (16, 99)
        keypoints_flat = keypoints_array.reshape(SEQUENCE_LENGTH, -1) # Reshape to (16, 99)
        if keypoints_flat.shape[1] != IN_DIM:
            # This check ensures the flattening resulted in the expected feature dimension (99)
            logger.error(f"Unexpected flattened dimension: {keypoints_flat.shape[1]}. Expected {IN_DIM}.")
            return feedback, "Keypoint flattening failed.", err_values

        # 7. Prepare tensors for the model
        seq_tensor = torch.from_numpy(keypoints_flat).unsqueeze(0).to(DEVICE)  # Shape: (1, 16, 99)
        ex_index = torch.tensor([exercise_id - 1], device=DEVICE) # Exercise ID is 1-based
        ex_1hot = F.one_hot(ex_index, num_classes=NUM_EXERCISES).float()  # Shape: (1, NUM_EXERCISES)

        # --- Model Inference ---
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad(): # Disable gradient calculations for inference
            # The model's forward pass expects (B, T, D) where D=IN_DIM=99
            logits, err_hat = model(seq_tensor, ex_1hot)
            # logits shape: (1, 2) -> [Incorrect Score, Correct Score]
            # err_hat shape: (1, N_ERR) -> (1, 14) Error values

            predicted_class = logits.argmax(dim=1).item() # 0 for Incorrect, 1 for Correct
            # Squeeze batch dim, move to CPU, convert to NumPy array
            err_values = err_hat.squeeze(0).cpu().numpy() # Shape: (N_ERR,) -> (14,)

        # 8. Validate error values shape (sanity check)
        if err_values.shape != (N_ERR,):
            logger.error(f"Unexpected err_values shape after inference: {err_values.shape}. Expected ({N_ERR},).")
            return feedback, "Model error output mismatch.", err_values # Return default error values

        # --- Feedback and Suggestion Generation (Mirrors original logic) ---

        # 9. Set feedback based on prediction
        feedback = "Correct" if predicted_class == 1 else "Incorrect"

        # 10. Generate suggestion if Incorrect, mimicking the original script's logic
        if predicted_class == 0:  # Incorrect prediction
            # Check if any error value is significantly different from zero
            if np.any(np.abs(err_values) > ERROR_THRESHOLD):
                max_error_idx = np.argmax(np.abs(err_values)) # Find index of largest absolute error
                joint_with_error = ERR_JOINTS[max_error_idx]  # Get the name of the joint
                max_error = err_values[max_error_idx]         # Get the signed error value
                # Format the suggestion string
                suggestion = f"Check {joint_with_error.replace('_', ' ')} (Dev: {max_error:+.1f}°)"
            else:
                # If classified as incorrect but no specific joint error stands out
                suggestion = "Check Form"
        else:
            # Correct prediction, no suggestion needed
            suggestion = ""

        # Log successful inference at INFO level for monitoring
        logger.info(f"Inference successful - ExID: {exercise_id}, Feedback: {feedback}, Suggestion: '{suggestion}'")
        return feedback, suggestion, err_values

    except Exception as e:
        # Catch any unexpected errors during the process
        logger.exception(f"An unexpected error occurred during inference execution: {e}")
        # Return the default error state and empty suggestion/zero errors
        return "Error", f"Internal error during inference: {str(e)}", err_values