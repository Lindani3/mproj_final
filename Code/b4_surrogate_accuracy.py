"""
Block 4 — Surrogate Accuracy: Predicted vs Analytical IRS Value.

Loads the trained Model 1a GRU (checkpoint best.pt) and runs inference
on the test split of model1a_dataset.csv.

Plots: predicted V_IRS vs analytical V_IRS (scatter), with the identity
line, R², and RMSE annotated.

Architecture (from train_1a.py):
  - Input: (B, 20) discount factors at payment dates + (B, 4) scalars [a, sigma, t_k, r_t]
  - Hidden: stacked GRU layers (sizes from checkpoint config)
  - Output: scalar V_IRS

Data:   ../../mproj/model1a_dataset.csv
Model:  ../../mproj/checkpoints/model1a/best.pt
Output: ../../results/b4_surrogate_accuracy.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.abspath(os.path.join(HERE, "..", "..", "mproj", "model1a_dataset.csv"))
CKPT     = os.path.abspath(os.path.join(HERE, "..", "..", "mproj", "checkpoints", "model1a", "best.pt"))
OUT_DIR  = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DISC_COLS = [f"P0_{j:02d}" for j in range(1, 21)]   # P0_01 .. P0_20
SCALAR_COLS = ["a", "sigma", "t_k", "r_t"]
TARGET_COL  = "V_irs"


# ── Model definition (mirrors train_1a.py) ────────────────────────────────────
class SwapNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, n_scalar, dropout=0.0):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.num_layers   = len(hidden_sizes)
        self.grus = nn.ModuleList()
        for k in range(self.num_layers):
            in_size = input_size if k == 0 else hidden_sizes[k - 1]
            self.grus.append(
                nn.GRU(input_size=in_size, hidden_size=hidden_sizes[k],
                       num_layers=1, batch_first=True)
            )
        self.drop = nn.Dropout(p=dropout)
        total_features = hidden_sizes[-1] + n_scalar
        self.fc = nn.Sequential(
            nn.Linear(total_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_seq, x_scalar):
        x = x_seq.unsqueeze(-1)
        for k, gru in enumerate(self.grus):
            x, h = gru(x)
            if k < self.num_layers - 1:
                x = self.drop(x)
        h_last   = h.squeeze(0)
        combined = torch.cat([h_last, x_scalar], dim=-1)
        return self.fc(combined).squeeze(-1)


# ── Load checkpoint ───────────────────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT}")
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
cfg  = ckpt["model_config"]
print(f"  hidden_sizes = {cfg['hidden_sizes']}  |  n_scalar = {cfg['n_scalar']}")

model = SwapNet(
    input_size   = cfg.get("input_size", 1),
    hidden_sizes = cfg["hidden_sizes"],
    n_scalar     = cfg["n_scalar"],
    dropout      = cfg.get("dropout", 0.0),
)
model.load_state_dict(ckpt["model_state"])
model.to(DEVICE)
model.eval()

# ── Load test data ─────────────────────────────────────────────────────────────
print(f"Loading dataset: {DATA}")
df   = pd.read_csv(DATA)
test = df[df["split"] == "test"].copy()
print(f"  Test samples: {len(test):,}")

X_disc   = torch.tensor(test[DISC_COLS].values, dtype=torch.float32, device=DEVICE)
X_scalar = torch.tensor(test[SCALAR_COLS].values, dtype=torch.float32, device=DEVICE)
y_true   = test[TARGET_COL].values

# ── Inference ─────────────────────────────────────────────────────────────────
BATCH = 4096
preds = []
with torch.no_grad():
    for start in range(0, len(X_disc), BATCH):
        out = model(X_disc[start:start+BATCH], X_scalar[start:start+BATCH])
        preds.append(out.cpu().numpy())
y_pred = np.concatenate(preds)

# ── Metrics ───────────────────────────────────────────────────────────────────
rmse  = np.sqrt(np.mean((y_pred - y_true) ** 2))
ss_res = np.sum((y_true - y_pred) ** 2)
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
r2    = 1.0 - ss_res / ss_tot
print(f"  RMSE = {rmse:.6f}  |  R² = {r2:.6f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
# Sub-sample for visibility (scatter with 1M points is too dense)
N_PLOT = min(5000, len(y_true))
idx    = np.random.default_rng(0).choice(len(y_true), N_PLOT, replace=False)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_true[idx], y_pred[idx], s=4, alpha=0.4, color="#1f77b4", label="Test samples")

lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
ax.plot(lims, lims, "k--", lw=1.2, label="Identity ($y = x$)")

ax.set_xlabel("Analytical $V_{\\mathrm{IRS}}$", fontsize=12)
ax.set_ylabel("Predicted $\\hat{V}_{\\mathrm{IRS}}$", fontsize=12)
ax.set_title("Model 1a: Surrogate Accuracy — Predicted vs Analytical", fontsize=11)
ax.text(0.05, 0.92,
        f"$R^2 = {r2:.4f}$\n$\\mathrm{{RMSE}} = {rmse:.5f}$",
        transform=ax.transAxes, fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b4_surrogate_accuracy.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
