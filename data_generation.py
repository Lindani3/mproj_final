"""
Model 1a: Closed-Form IRS Label Generation (alg:cf)

Yield curves come directly from the FEDS200628 historical dataset (no PCA).
(a, sigma, t_k, r_t) are sampled via Latin Hypercube Sampling.

Input vector  x_i  = [y(T_1), ..., y(T_10), a, sigma, t_k, r_t]   (14 inputs)
Label         y_i  = V_IRS(t_k, r_t ; K^par(curve, a, sigma))       scalar
Sensitivities g_i  = [dV / dy(T_j)]_{j=1..10}                       10 one-sided FD

LHS dimensions (5):
  0  ->  curve index   (uniform quantile -> integer index into X_hist)
  1  ->  a             (uniform in [A_LO, A_HI])
  2  ->  sigma         (uniform in [E_LO, E_HI])
  3  ->  t_k           (quantile -> one of 40 quarterly dates)
  4  ->  z             (standard normal -> r_t = alpha(t_k) + RT_SIGMA * nu(t_k) * z)

r_t is an independent input and is held fixed during the bump-and-revalue.
B(t_k, T_j) depends only on a, so it does not change under yield-curve bumps.
Only K^par and A(t_k, T_j) are recomputed for each bump.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.special import ndtri

from hw_lib import svensson, TENORS, SplineCurve, k_par, irs_value, pay_dates

# ── Configuration ─────────────────────────────────────────────────────────────

N        = 100_000     # total samples
A_LO     = 0.01        # HW mean reversion lower bound
A_HI     = 0.30        # HW mean reversion upper bound
E_LO     = 0.001       # HW vol lower bound
E_HI     = 0.030       # HW vol upper bound
RT_SIGMA = 2.0         # r_t drawn within +/- RT_SIGMA * nu(t_k)
DELTA    = 1e-4        # 1 bp bump for FD sensitivities
SEED     = 42
OUTPUT   = "model1a_dataset.csv"
CHUNK    = 10_000      # progress print interval

dates_all = pay_dates()   # 40 quarterly dates: 0.25, 0.50, ..., 10.0
J         = len(TENORS)   # 10

# ── 1.  Load all historical curves from FEDS200628 ────────────────────────────

print("Loading FEDS200628 historical curves ...")
feds = pd.read_csv("feds200628.csv", skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")[["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]].dropna()

raw    = feds.values          # (N_hist, 6) Svensson parameters
N_hist = len(raw)

print(f"  Usable dates : {N_hist}")
print(f"  Date range   : {feds.index.min().date()}  to  {feds.index.max().date()}")

# Evaluate zero rates at TENORS for every historical date
X_hist = np.zeros((N_hist, J))
for i in range(N_hist):
    b0, b1, b2, b3, t1, t2 = raw[i]
    X_hist[i] = svensson(TENORS, b0, b1, b2, b3, t1, t2) / 100.0   # decimal

print(f"  Rate range   : [{X_hist.min()*100:.3f}%, {X_hist.max()*100:.3f}%]")

# ── 2.  Latin Hypercube Sampling in 5 dimensions ──────────────────────────────

print(f"\nSampling N={N:,} points via LHS (5 dims) ...")
sampler = qmc.LatinHypercube(d=5, seed=SEED)
U       = sampler.random(n=N)      # (N, 5) uniform in [0, 1]

# dim 0: curve index (sample with replacement from historical dates)
curve_idx = np.floor(U[:, 0] * N_hist).astype(int).clip(0, N_hist - 1)
Y_curves  = X_hist[curve_idx]      # (N, 10) nodal zero rates

# dim 1: a (HW mean reversion)
a_arr = A_LO + U[:, 1] * (A_HI - A_LO)

# dim 2: sigma (HW vol)
e_arr = E_LO + U[:, 2] * (E_HI - E_LO)

# dim 3: t_k (monitoring date, discrete)
n_dates   = len(dates_all)
t_k_idx   = np.floor(U[:, 3] * n_dates).astype(int).clip(0, n_dates - 1)
t_k_arr   = dates_all[t_k_idx]

# dim 4: z -> r_t (standard normal quantile, applied after curve/HW are known)
z_rt = ndtri(np.clip(U[:, 4], 1e-9, 1 - 1e-9))

print(f"  a     range  : [{a_arr.min():.4f}, {a_arr.max():.4f}]")
print(f"  sigma range  : [{e_arr.min():.5f}, {e_arr.max():.5f}]")
print(f"  t_k   range  : [{t_k_arr.min():.2f}, {t_k_arr.max():.2f}]")
print(f"  y     range  : [{Y_curves.min()*100:.3f}%, {Y_curves.max()*100:.3f}%]")

# ── 3.  Label generation ──────────────────────────────────────────────────────

col_y    = [f"y{j}" for j in range(J)]
col_g    = [f"g{j}" for j in range(J)]
all_cols = col_y + ["a", "sigma", "t_k", "k_par", "r_t", "V_irs"] + col_g

rows = []

print(f"\nGenerating labels ...")
print(f"{'i':>8}  {'a':>6}  {'sigma':>7}  {'t_k':>5}  "
      f"{'r_t%':>7}  {'K%':>7}  {'V_irs':>10}  {'max|g|':>10}")

for i in range(N):

    y_nodes = Y_curves[i]
    a       = float(a_arr[i])
    sigma   = float(e_arr[i])
    t_k     = float(t_k_arr[i])
    k_idx   = int(t_k_idx[i]) + 1     # payments passed: t_k is the k_idx-th date

    # Build base SplineCurve and compute K^par
    sc = SplineCurve(y_nodes, a, sigma)
    K0 = k_par(sc, dates_all)

    # Short rate at t_k: r_t = alpha(t_k) + RT_SIGMA * nu(t_k) * z
    mu_rt = float(sc.alpha(t_k)[0])
    nu_rt = float(np.sqrt(sc.var_r(t_k)[0]))
    r_t   = float(np.clip(mu_rt + RT_SIGMA * nu_rt * z_rt[i], -0.05, 0.25))

    # Base IRS value (closed-form)
    V0 = float(irs_value(sc, t_k, r_t, k_idx, K0, dates_all))

    # FD sensitivities: bump y(T_j) by +Delta, hold r_t fixed
    g = np.zeros(J)
    for j in range(J):
        sc_b  = sc.bumped(j, DELTA)
        K_b   = k_par(sc_b, dates_all)
        V_b   = float(irs_value(sc_b, t_k, r_t, k_idx, K_b, dates_all))
        g[j]  = (V_b - V0) / DELTA

    rows.append(list(y_nodes) + [a, sigma, t_k, K0, r_t, V0] + list(g))

    if (i + 1) % CHUNK == 0:
        print(f"{i+1:>8d}  {a:>6.3f}  {sigma:>7.4f}  {t_k:>5.2f}  "
              f"{r_t*100:>7.3f}  {K0*100:>7.4f}  {V0:>10.6f}  "
              f"{float(np.max(np.abs(g))):>10.4f}")

# ── 4.  Assemble and write ────────────────────────────────────────────────────

df = pd.DataFrame(rows, columns=all_cols)

# Train / validation / test split (70 / 15 / 15)
rng   = np.random.default_rng(SEED + 1)
perm  = rng.permutation(N)
n_tr  = int(0.70 * N)
n_va  = int(0.15 * N)
split = np.full(N, "test")
split[perm[:n_tr]]           = "train"
split[perm[n_tr:n_tr + n_va]] = "val"
df["split"] = split

df.to_csv(OUTPUT, index=False, float_format="%.8f")
print(f"\nWritten {len(df):,} rows x {len(df.columns)} cols to {OUTPUT}")

# ── 5.  Summary statistics ────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print("\nV_irs (label):")
print(df["V_irs"].describe().round(6).to_string())

print("\nK^par (%):")
print((df["k_par"] * 100).describe().round(4).to_string())

print("\nr_t (%):")
print((df["r_t"] * 100).describe().round(4).to_string())

print("\nFraction V > 0 (receiver in-the-money): "
      f"{(df['V_irs'] > 0).mean()*100:.1f}%")
print(f"Fraction V < 0 (receiver out-of-money): "
      f"{(df['V_irs'] < 0).mean()*100:.1f}%")

print("\nSplit counts:")
print(df["split"].value_counts().to_string())

print("\nSensitivity g9 (10-year tenor):")
print(df["g9"].describe().round(6).to_string())
