"""
Block 4 — Training and Validation Loss Curves.

Plots train and validation MSE vs epoch for Model 1a (GRU on discount-factor
inputs) and Model 1b (GRU on yield-curve inputs), side by side, log-scale y-axis.

Data: ../../mproj/checkpoints/model1a/history.csv
      ../../mproj/checkpoints/model1b/history.csv

Output: ../../results/b4_training_history.png
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

HERE     = os.path.dirname(os.path.abspath(__file__))
CKPT_1A  = os.path.abspath(os.path.join(HERE, "..", "..", "mproj", "checkpoints", "model1a", "history.csv"))
CKPT_1B  = os.path.abspath(os.path.join(HERE, "..", "..", "mproj", "checkpoints", "model1b", "history.csv"))
OUT_DIR  = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

h1a = pd.read_csv(CKPT_1A)
h1b = pd.read_csv(CKPT_1B)

print(f"Model 1a: {len(h1a)} epochs  |  final val_mse = {h1a['val_mse'].iloc[-1]:.6f}")
print(f"Model 1b: {len(h1b)} epochs  |  final val_mse = {h1b['val_mse'].iloc[-1]:.6f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

CONFIGS = [
    (h1a, "Model 1a — GRU on Discount-Factor Inputs"),
    (h1b, "Model 1b — GRU on Yield-Curve Inputs"),
]

for ax, (h, title) in zip(axes, CONFIGS):
    ax.semilogy(h["epoch"], h["train_mse"], lw=1.8, color="#1f77b4",
                label="Training MSE")
    ax.semilogy(h["epoch"], h["val_mse"],   lw=1.8, color="#d62728",
                ls="--", label="Validation MSE")

    # Mark best (lowest val_mse) epoch
    best_ep  = h.loc[h["val_mse"].idxmin(), "epoch"]
    best_mse = h["val_mse"].min()
    ax.axvline(best_ep, color="grey", lw=1.0, ls=":", alpha=0.8)
    ax.annotate(
        f"Best epoch {best_ep}\nval MSE={best_mse:.4f}",
        xy=(best_ep, best_mse),
        xytext=(best_ep + max(1, len(h) // 10), best_mse * 3),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color="grey"),
        color="grey",
    )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE (log scale)", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b4_training_history.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
