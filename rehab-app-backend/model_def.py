import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys # To potentially add paths if classes are in separate files

# --- Constants (Required by the model definition) ---
# Ensure these match the configuration of the *specific model file* being loaded
ERR_JOINTS   = [
  "LEFT_ELBOW","RIGHT_ELBOW",
  "LEFT_SHOULDER","RIGHT_SHOULDER",
  "LEFT_HIP","RIGHT_HIP",
  "LEFT_KNEE","RIGHT_KNEE",
  "SPINE","HEAD",
]
N_ERR = len(ERR_JOINTS)
NUM_EXERCISES = 6 # Example: Must match the 'num_ex' the model was trained with

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