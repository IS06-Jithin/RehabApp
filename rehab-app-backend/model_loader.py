import sys
# Tell Python “when you see __mp_main__, look here instead”
sys.modules["__mp_main__"] = sys.modules[__name__]

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys # To potentially add paths if classes are in separate files

# --- Constants (Required by the model definition) ---
#  Exerciseses (Ex1…Ex6)
NUM_EXERCISES = 6

ERR_JOINTS   = [
  "LEFT_ELBOW","RIGHT_ELBOW",
  "LEFT_SHOULDER","RIGHT_SHOULDER",
  "LEFT_HIP","RIGHT_HIP",
  "LEFT_KNEE","RIGHT_KNEE",
  "SPINE","HEAD",
  "LEFT_WRIST", "RIGHT_WRIST",
  "LEFT_ANKLE", "RIGHT_ANKLE"
]
N_ERR = len(ERR_JOINTS)   # 14

ERR_COLS = [f"err_{i}" for i in range(N_ERR)]

# --- Model Class Definitions ---
# These MUST be defined *before* calling torch.load with weights_only=False
# They must EXACTLY match the definitions used when the model was saved.

class KeypointEncoder(nn.Module):
    def __init__(self, in_dim:int, embed:int=512):
        super().__init__()
        # Ensure kernel_size, padding etc. match the saved model's architecture
        self.conv1 = nn.Conv1d(in_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, embed, kernel_size=3, padding=1)
        self.pool  = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D); treat as (B, D, 1) for Conv1d
        if x.dim() == 2:
             x = x.unsqueeze(2)                 # → (B, D, 1)
        elif x.dim() != 3 or x.shape[2] != 1:
             # If input is already (B, D, L), ensure L=1 or handle appropriately
             if x.dim() == 3 and x.shape[2] > 1:
                  print("Warning: KeypointEncoder received unexpected input shape:", x.shape)
                  if x.shape[2] != 1:
                      raise ValueError(f"Encoder expected input like (B, D) or (B, D, 1), got {x.shape}")
             elif x.dim() != 3:
                 raise ValueError(f"Encoder expected input like (B, D) or (B, D, 1), got {x.shape}")
        # Proceed with convolutions
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.pool(x).squeeze(-1)    # → (B, embed)


class PoseQualityNetKP(nn.Module):
    def __init__(self,
                 in_dim: int, # e.g., 99 (33 joints * 3 coords)
                 num_ex: int, # e.g., 6
                 hidden: int = 256, # Must match saved model
                 ex_emb: int = 64,  # Must match saved model
                 embed: int = 512): # Must match saved model (output dim of encoder)
        super().__init__()
        # keypoint feature extractor
        self.encoder = KeypointEncoder(in_dim, embed=embed)

        # sequence model
        self.lstm = nn.LSTM(
            input_size=embed, # Input size matches encoder output
            hidden_size=hidden,
            num_layers=2,     # Must match saved model
            batch_first=True,
            bidirectional=True # Must match saved model
        )
        feat_dim = hidden * 2 # Because bidirectional=True

        # exercise embedding MLP
        self.ex_emb = nn.Sequential(
            nn.Linear(num_ex, ex_emb),
            nn.ReLU(),
            nn.Linear(ex_emb, ex_emb)
        )

        # final heads
        self.cls_head = nn.Linear(feat_dim + ex_emb, 2) # 2 classes: incorrect, correct
        self.err_head = nn.Linear(feat_dim + ex_emb, N_ERR) # N_ERR outputs

    def forward(self,
                seq:     torch.Tensor,  # Expected shape (B, T, D), e.g., (B, 16, 99)
                ex_1hot: torch.Tensor   # Expected shape (B, num_ex)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 1) keypoint → sequence feats
        B, T, D = seq.shape
        frame_embeddings = []
        for t in range(T):
            frame_data = seq[:, t, :] # Get data for frame t: (B, D)
            frame_embedding = self.encoder(frame_data) # Output: (B, embed)
            frame_embeddings.append(frame_embedding)

        feats = torch.stack(frame_embeddings, dim=1) # Stacks embeddings along time: (B, T, embed)

        # 2) sequence model (LSTM)
        out, _ = self.lstm(feats) # Output shape (B, T, 2*hidden)

        # Aggregate LSTM outputs (e.g., mean pooling over time)
        g = out.mean(dim=1)       # Output shape (B, 2*hidden)

        # 3) exercise embed
        ex_e = self.ex_emb(ex_1hot) # Output shape (B, ex_emb)

        # 4) concat and heads
        h = torch.cat([g, ex_e], dim=1) # Output shape (B, 2*hidden + ex_emb)
        logits = self.cls_head(h)       # Output shape (B, 2)
        err_hat = self.err_head(h)      # Output shape (B, N_ERR)

        return logits, err_hat

# --- Model Loading Function ---

def load_model(
    state_dict_path: str = "model/kp_pose_quality_windows_ex.pth",
    device_str:     str | None = None
) -> nn.Module | None:
    """
    Instantiate a PoseQualityNetKP, load weights from a .pth state-dict,
    move to `device`, switch to eval(), and return.

    Args:
      state_dict_path: path to the saved state_dict (torch.save(model.state_dict()))
      device_str:      optional override ("cpu","cuda","mps")
    """
    # 1) Resolve paths & device
    sd_file = Path(state_dict_path)
    if not sd_file.exists():
        print(f"❌ State‐dict not found at {sd_file}")
        return None

    device = torch.device(device_str) if device_str \
             else (torch.device("mps") if torch.backends.mps.is_available()
                   else torch.device("cuda") if torch.cuda.is_available()
                   else torch.device("cpu"))
    print(f"Loading weights to device {device}")

    # 2) Instantiate EXACT same architecture trained
    IN_DIM = 33 * 3
    model = PoseQualityNetKP(
        in_dim=IN_DIM,
        num_ex=NUM_EXERCISES,
        hidden=256,
        ex_emb=64,
        embed=512
    ).to(device)

    # 3) Load the state_dict
    sd = torch.load(sd_file, map_location=device)
    model.load_state_dict(sd)
    print(f"✅ Loaded weights from {sd_file}")

    # 4) Finalize
    model.eval()
    return model