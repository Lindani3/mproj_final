import numpy as np
import pandas as pd
from hw_lib import SplineCurve, TENORS, k_par, irs_value, pay_dates

# Fixed parameters
FLAT_RATE = 0.05
A         = 0.10
SIGMA     = 0.015
M         = 1000    # simulated r_t per monitoring date
SEED      = 42

# Flat yield curve: y(T_j) = 0.05 for all tenors
y_nodes   = np.full(len(TENORS), FLAT_RATE)
curve     = SplineCurve(y_nodes, A, SIGMA)
dates     = pay_dates()
K0        = k_par(curve, dates)

print(f"r_0   = {curve.r0*100:.4f}%")
print(f"K^par = {K0*100:.4f}%")

rng  = np.random.default_rng(SEED)
rows = []

for k_idx, t_k in enumerate(dates):
    mu_r  = float(curve.alpha(t_k)[0])
    var_r = float(curve.var_r(t_k)[0])
    nu_r  = np.sqrt(var_r)

    # Simulate M realisations of r_t from Q-distribution
    r_samples = mu_r + nu_r * rng.standard_normal(M)

    for r_t in r_samples:
        r_t  = float(np.clip(r_t, -0.05, 0.25))
        V    = float(irs_value(curve, t_k, r_t, k_idx + 1, K0, dates))
        rows.append([t_k, mu_r, nu_r, r_t, V])

df = pd.DataFrame(rows, columns=["t_k", "alpha_tk", "nu_tk", "r_t", "V_irs"])
df.to_csv("experiment_flat.csv", index=False, float_format="%.8f")

# Summary per monitoring date
print(f"\n{'t_k':>5}  {'alpha(t)%':>10}  {'nu(t)%':>8}  "
      f"{'E[V]':>10}  {'std[V]':>10}  {'EE':>10}")
for t_k, grp in df.groupby("t_k"):
    print(f"{t_k:>5.2f}  {grp.alpha_tk.iloc[0]*100:>10.4f}  "
          f"{grp.nu_tk.iloc[0]*100:>8.4f}  "
          f"{grp.V_irs.mean():>10.6f}  "
          f"{grp.V_irs.std():>10.6f}  "
          f"{(grp.V_irs.clip(lower=0)).mean():>10.6f}")

