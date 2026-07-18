"""
Block 4 — Greeks Accuracy: Surrogate Jacobian vs QuantLib Bump-and-Reprice.

Model 1b maps yield-curve inputs (y0..y9) to V_IRS and its key-rate DV01s.
The dataset stores dv01_0..dv01_9 (partial derivatives dV/dy_j) computed
analytically during data generation.

This script:
  1. Loads the Model 1b test split.
  2. Loads trained Model 1b checkpoint and computes predicted DV01s via
     torch.autograd (Jacobian of output w.r.t. curve inputs).
  3. Compares predicted vs stored analytical DV01s per tenor knot.
  4. Plots: per-tenor RMSE bar chart + scatter (predicted vs analytical)
     for each of the 10 tenor buckets.

Data:   ../../mproj/model1b_dataset.csv
Model:  ../../mproj/checkpoints/model1b/best.pt
Output: ../../results/b4_greeks_accuracy.png
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA     = os.path.abspath(os.path.join(HERE, "..", "..", "mproj", "model1b_dataset.csv"))
CKPT     = os.path.abspath(os.path.join(HERE, "..", "..", "mproj", "checkpoints", "model1b", "best.pt"))
OUT_DIR  = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CURVE_COLS  = [f"y{j}" for j in range(10)]           # y0 .. y9
SCALAR_COLS = ["a", "sigma", "t_k"]                   # model1b scalars
DV01_COLS   = [f"dv01_{j}" for j in range(10)]        # analytical DV01s
TARGET_COL  = "V_irs"
TENORS      = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0,
               12.0, 15.0, 20.0]                       # adjust to actual knots if needed


# ── Model definition (mirrors model1b architecture, same GRU backbone) ────────
class SwapNetB(nn.Module):
    """
    Model 1b: input is yield-curve values (y0..y9) as a sequence,
    scalar conditioning on (a, sigma, t_k).
    """
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

model = SwapNetB(
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
test = df[df["split"] == "test"].copy().reset_index(drop=True)
print(f"  Test samples: {len(test):,}")

n_scalar_actual = len(SCALAR_COLS)
# Adjust scalar columns if model used different features
if cfg["n_scalar"] == 4 and "r_t" in df.columns:
    SCALAR_COLS = ["a", "sigma", "t_k", "r_t"]

X_curve  = torch.tensor(test[CURVE_COLS].values, dtype=torch.float32, device=DEVICE)
X_scalar = torch.tensor(test[SCALAR_COLS[:cfg["n_scalar"]]].values,
                         dtype=torch.float32, device=DEVICE)
y_dv01   = test[DV01_COLS].values   # shape (N_test, 10)

# ── Compute Jacobian via autograd ─────────────────────────────────────────────
print("Computing surrogate Jacobians via autograd...")
BATCH = 512
all_jac = []

for start in range(0, len(X_curve), BATCH):
    xc = X_curve[start:start+BATCH].detach().requires_grad_(True)
    xs = X_scalar[start:start+BATCH].detach()
    out = model(xc, xs)
    # Jacobian: dV/dy_j for each sample
    jac = torch.zeros(out.shape[0], X_curve.shape[1], device=DEVICE)
    for i in range(out.shape[0]):
        grad = torch.autograd.grad(out[i], xc, retain_graph=True)[0]
        jac[i] = grad[i]
    all_jac.append(jac.detach().cpu().numpy())

pred_dv01 = np.vstack(all_jac)   # shape (N_test, 10)

# ── Per-tenor RMSE ────────────────────────────────────────────────────────────
rmse_per_tenor = np.sqrt(np.mean((pred_dv01 - y_dv01) ** 2, axis=0))
print("\nPer-tenor RMSE (DV01):")
for j, rmse in enumerate(rmse_per_tenor):
    print(f"  y{j}: {rmse:.6f}")

# ── Plot: RMSE bar chart + scatter for 2 representative tenors ────────────────
fig = plt.figure(figsize=(14, 5))
gs  = fig.add_gridspec(1, 3, width_ratios=[2, 1, 1])

# Panel 1: RMSE per tenor
ax0 = fig.add_subplot(gs[0])
tenor_labels = [f"$y_{{{j}}}$" for j in range(10)]
ax0.bar(tenor_labels, rmse_per_tenor, color="#1f77b4", alpha=0.85)
ax0.set_xlabel("Yield-curve knot", fontsize=11)
ax0.set_ylabel("RMSE (DV01)", fontsize=11)
ax0.set_title("Per-Knot DV01 RMSE — Model 1b", fontsize=11)
ax0.grid(True, axis="y", alpha=0.25)

# Panel 2: scatter for knot 6 (typically 5Y — largest DV01)
for panel_idx, knot_idx in enumerate([6, 9]):
    ax = fig.add_subplot(gs[panel_idx + 1])
    N_PLOT = min(2000, len(y_dv01))
    idx_s  = np.random.default_rng(0).choice(len(y_dv01), N_PLOT, replace=False)
    ax.scatter(y_dv01[idx_s, knot_idx], pred_dv01[idx_s, knot_idx],
               s=5, alpha=0.4, color="#d62728")
    lims = [min(y_dv01[:, knot_idx].min(), pred_dv01[:, knot_idx].min()),
            max(y_dv01[:, knot_idx].max(), pred_dv01[:, knot_idx].max())]
    ax.plot(lims, lims, "k--", lw=1.2)
    ax.set_xlabel(f"Analytical $\\partial V/\\partial y_{{{knot_idx}}}$", fontsize=10)
    ax.set_ylabel(f"Predicted", fontsize=10)
    ax.set_title(f"DV01 knot $y_{{{knot_idx}}}$\nRMSE={rmse_per_tenor[knot_idx]:.5f}",
                 fontsize=10)
    ax.grid(True, alpha=0.25)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b4_greeks_accuracy.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
