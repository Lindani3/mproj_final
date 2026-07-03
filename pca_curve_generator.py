"""
Script 2 of 2: PCA-based synthetic yield curve generator.

Requires the output of pca_analysis.py (run that first and confirm results).

Method:
  1. Re-run PCA on feds200628.csv at the hw_lib TENORS grid.
  2. Retain K=5 principal components.
  3. Sample N_CURVES synthetic factor scores via quasi-random Sobol sequences,
     scaled to the empirical factor distribution (variance x SCALE).
  4. Reconstruct synthetic SplineCurve objects: y = mu + z @ E_K.T
  5. For each synthetic curve compute:
       - k_par  (par IRS rate, eq:k_par)
       - DV01_j = (k_par(bumped_j) - k_par) / BUMP   for j = 0..9
  6. Write to pca_synthetic_dataset.csv.

Output columns:
  y0..y9       zero rates at TENORS (decimal)
  k_par        par swap rate (decimal)
  dv01_0..9    sensitivity to each nodal rate (decimal/decimal)
"""

import numpy as np
import pandas as pd
from scipy.special import ndtri
from hw_lib import svensson, TENORS, SplineCurve, k_par, pay_dates

# ── Configuration ─────────────────────────────────────────────────────────────

N_COMPONENTS  = 5          # PCs to retain
N_CURVES      = 20_000     # synthetic curves
SCALE         = 1.5        # inflate historical variance for stress coverage
BUMP          = 1e-4       # 1 bp bump for DV01
SEED          = 42
LAM           = 0.10       # Hull-White mean reversion (matches validate scripts)
ETA           = 0.015      # Hull-White volatility

OUTPUT_CSV = "pca_synthetic_dataset.csv"

# ── 1. Load and evaluate historical curves ────────────────────────────────────

feds = pd.read_csv("feds200628.csv", skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
feds = feds[["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]].dropna()

params = feds.values
N = len(params)
X = np.zeros((N, len(TENORS)))
for i in range(N):
    b0, b1, b2, b3, t1, t2 = params[i]
    X[i] = svensson(TENORS, b0, b1, b2, b3, t1, t2) / 100.0

print(f"Historical curves loaded: {N} dates")

# ── 2. PCA ────────────────────────────────────────────────────────────────────

mu  = X.mean(axis=0)
Xc  = X - mu
C   = (Xc.T @ Xc) / (N - 1)

vals, vecs = np.linalg.eigh(C)
idx  = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]

E_K = vecs[:, :N_COMPONENTS]    # (10, K)
Z   = Xc @ E_K                  # (N, K) historical scores

total_var = vals.sum()
cum = 0.0
print(f"\nVariance explained by {N_COMPONENTS} PCs:")
for k in range(N_COMPONENTS):
    cum += vals[k] / total_var
    print(f"  PC{k+1}: {vals[k]/total_var*100:.2f}%  cumulative {cum*100:.2f}%")

# ── 3. Sample factor scores via Sobol + Cholesky ─────────────────────────────
# Quasi-random Sobol in [0,1]^K, mapped to correlated normal via Cholesky
# of the empirical factor covariance scaled by SCALE.

Z_cov = np.cov(Z.T) * SCALE      # (K, K)
L     = np.linalg.cholesky(Z_cov)

rng = np.random.default_rng(SEED)

# Sobol engine — scipy's qmc.Sobol
from scipy.stats import qmc
sobol = qmc.Sobol(d=N_COMPONENTS, scramble=True, seed=SEED)
u     = sobol.random(N_CURVES)                          # (N_CURVES, K) in (0,1)
u     = np.clip(u, 1e-9, 1 - 1e-9)
z_std = ndtri(u)                                        # standard normal
Z_syn = z_std @ L.T                                     # (N_CURVES, K)

# ── 4. Reconstruct synthetic nodal zero rates ─────────────────────────────────

Y_syn = mu + Z_syn @ E_K.T       # (N_CURVES, 10), decimal
Y_syn = np.clip(Y_syn, 1e-4, 0.20)   # floor 1bp, ceiling 20%

print(f"\nSynthetic curves: {Y_syn.shape}")
print(f"Rate range: [{Y_syn.min()*100:.3f}%, {Y_syn.max()*100:.3f}%]")

# ── 5. Compute k_par and DV01 for each synthetic curve ───────────────────────

dates = pay_dates()   # quarterly, 0.25..10.0

# Column names
y_cols    = [f"y{j}" for j in range(len(TENORS))]
dv01_cols = [f"dv01_{j}" for j in range(len(TENORS))]
all_cols  = y_cols + ["k_par"] + dv01_cols

print(f"\nPricing {N_CURVES:,} curves (k_par + {len(TENORS)} DV01s each) ...")

rows = []
for i, y_nodes in enumerate(Y_syn):
    sc   = SplineCurve(y_nodes, LAM, ETA)
    K0   = k_par(sc, dates)

    dvs = np.zeros(len(TENORS))
    for j in range(len(TENORS)):
        sc_b  = sc.bumped(j, BUMP)
        dvs[j] = (k_par(sc_b, dates) - K0) / BUMP

    row = list(y_nodes) + [K0] + list(dvs)
    rows.append(row)

    if (i + 1) % 4000 == 0:
        print(f"  {i+1:>6}/{N_CURVES} done")

# ── 6. Write CSV ──────────────────────────────────────────────────────────────

df = pd.DataFrame(rows, columns=all_cols)
df.to_csv(OUTPUT_CSV, index=False, float_format="%.8f")

print(f"\nWritten {len(df):,} rows to {OUTPUT_CSV}")
print("\nk_par summary (%):")
print((df["k_par"] * 100).describe().round(4).to_string())
