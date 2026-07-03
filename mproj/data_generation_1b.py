"""
data_generation_1b.py
Model 1b: IRS dataset with Monte Carlo simulation labels.

Input vector  x_i = [y(T_1),...,y(T_10), a, sigma, t_k, r_t]   (14 inputs)
Label         y_i = V_hat_IRS(t_k, r_t; K^par)                   scalar  O(M^{-1/2}) noise
Sensitivity   d_i = dV_hat/dr_t  (pathwise differentiation)       scalar

Identical sampling to Model 1a. The only difference is that labels are
estimated by simulating M short-rate paths rather than using the affine
closed-form formula. This introduces O(M^{-1/2}) label noise, isolating
its effect on surrogate accuracy when compared against Model 1a.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.special import ndtri

from curves import SplineCurve, svensson, TENORS
from irs    import IRS
from labels import MCLabels

# ── Configuration ──────────────────────────────────────────────────────────────
N                = 100_000
A_LO             = 0.01
A_HI             = 0.30
E_LO             = 0.001
E_HI             = 0.030
RT_SIGMA         = 2.0
SEED             = 42
OUTPUT           = "model1b_dataset.csv"
CHUNK            = 1_000
DATA_PATH        = "feds200628.csv"

M_PATHS          = 1_000     # MC paths per label
STEPS_PER_PERIOD = 10        # fine grid steps per quarterly period

swap = IRS(tenor=10.0, freq=0.25)
gen  = MCLabels(M=M_PATHS, seed=SEED, steps_per_period=STEPS_PER_PERIOD)
J    = len(TENORS)

# ── 1. Load historical curves ──────────────────────────────────────────────────
print("Loading FEDS200628 historical curves ...")
feds = pd.read_csv(DATA_PATH, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")[["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]].dropna()

raw    = feds.values
N_hist = len(raw)

X_hist = np.zeros((N_hist, J))
for i in range(N_hist):
    b0, b1, b2, b3, t1, t2 = raw[i]
    X_hist[i] = svensson(TENORS, b0, b1, b2, b3, t1, t2) / 100.0

print(f"  Dates  : {N_hist}  ({feds.index.min().date()} to {feds.index.max().date()})")
print(f"  Rates  : [{X_hist.min()*100:.3f}%, {X_hist.max()*100:.3f}%]")

# ── 2. Latin Hypercube Sampling  (identical to Model 1a) ──────────────────────
print(f"\nSampling {N:,} points via LHS (5 dims) ...")
U = qmc.LatinHypercube(d=5, seed=SEED).random(n=N)

curve_idx = np.floor(U[:, 0] * N_hist).astype(int).clip(0, N_hist - 1)
Y_curves  = X_hist[curve_idx]

a_arr   = A_LO + U[:, 1] * (A_HI - A_LO)
e_arr   = E_LO + U[:, 2] * (E_HI - E_LO)

n_dates = len(swap.payment_dates)
t_k_idx = np.floor(U[:, 3] * n_dates).astype(int).clip(0, n_dates - 1)
t_k_arr = swap.payment_dates[t_k_idx]

z_rt = ndtri(np.clip(U[:, 4], 1e-9, 1 - 1e-9))

# ── 3. Label generation ────────────────────────────────────────────────────────
col_y    = [f"y{j}" for j in range(J)]
all_cols = col_y + ["a", "sigma", "t_k", "k_par", "r_t", "V_irs", "delta_rt"]

rows = []
print(f"\nM={M_PATHS} paths, {STEPS_PER_PERIOD} steps/period per label")
print(f"{'i':>8}  {'a':>6}  {'sigma':>7}  {'t_k':>5}  {'r_t%':>7}  {'K%':>7}  {'V_irs':>10}  {'delta':>10}")

for i in range(N):
    y_nodes = Y_curves[i]
    a       = float(a_arr[i])
    sigma   = float(e_arr[i])
    t_k     = float(t_k_arr[i])

    sc = SplineCurve(y_nodes, a, sigma)
    K  = swap.k_par(sc)

    mu_rt = float(sc.alpha(t_k)[0])
    nu_rt = float(np.sqrt(sc.var_r(t_k)[0]))
    r_t   = float(np.clip(mu_rt + RT_SIGMA * nu_rt * z_rt[i], -0.05, 0.25))

    V, delta = gen._labels(swap, sc, t_k, r_t, K)

    rows.append(list(y_nodes) + [a, sigma, t_k, K, r_t, V, delta])

    if (i + 1) % CHUNK == 0:
        print(f"{i+1:>8d}  {a:>6.3f}  {sigma:>7.4f}  {t_k:>5.2f}  "
              f"{r_t*100:>7.3f}  {K*100:>7.4f}  {V:>10.6f}  {delta:>10.6f}")

# ── 4. Assemble, split, write ──────────────────────────────────────────────────
df   = pd.DataFrame(rows, columns=all_cols)
rng  = np.random.default_rng(SEED + 1)
perm = rng.permutation(N)
n_tr = int(0.70 * N)
n_va = int(0.15 * N)

split = np.full(N, "test")
split[perm[:n_tr]]            = "train"
split[perm[n_tr:n_tr + n_va]] = "val"
df["split"] = split

df.to_csv(OUTPUT, index=False, float_format="%.8f")
print(f"\nWritten {len(df):,} rows x {len(df.columns)} cols -> {OUTPUT}")

# ── 5. Summary ─────────────────────────────────────────────────────────────────
print(f"\nV_irs  : mean={df['V_irs'].mean():.4f}  std={df['V_irs'].std():.4f}"
      f"  min={df['V_irs'].min():.4f}  max={df['V_irs'].max():.4f}")
print(f"delta  : mean={df['delta_rt'].mean():.4f}  std={df['delta_rt'].std():.4f}")
print(f"K^par  : mean={df['k_par'].mean()*100:.4f}%")
print(f"V>0    : {(df['V_irs']>0).mean()*100:.1f}%   V<0: {(df['V_irs']<0).mean()*100:.1f}%")
print(f"\nSplit  :\n{df['split'].value_counts().to_string()}")
