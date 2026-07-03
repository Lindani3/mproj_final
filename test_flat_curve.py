"""
Test: FlatCurve + IRS + ClosedFormLabels (1a) + MCLabels (1b)
Flat curve at r0=1.25%, 10-year quarterly IRS.
"""

import numpy as np
import pandas as pd
from hw_lib import FlatCurve, IRS, ClosedFormLabels, MCLabels

pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 120)

# ── Parameters ────────────────────────────────────────────────────────────────
r0  = 0.0125   # 1.25% flat curve
lam = 0.10
eta = 0.015

curve = FlatCurve(r0=r0, lam=lam, eta=eta)
swap  = IRS(tenor=10, freq=0.25, notional=1.0)
K     = swap.k_par(curve)

print("=" * 70)
print("FLAT CURVE + IRS SETUP")
print("=" * 70)
print(f"r0       : {r0*100:.4f}%")
print(f"lambda   : {lam}")
print(f"eta      : {eta}")
print(f"K^par    : {K*100:.6f}%")

# ── Calibration check ─────────────────────────────────────────────────────────
print("\n--- Calibration check: bond_price(0, T, r0) == exp(-r0*T) ---")
tenors = np.array([1.0, 2.0, 5.0, 10.0])
P_model  = curve.bond_price(0.0, tenors, r0)
P_market = curve.discount_factor(tenors)
for T, pm, pth in zip(tenors, P_model, P_market):
    print(f"  T={T:4.1f}  model={pm:.8f}  market={pth:.8f}  err={abs(pm-pth):.2e}")

# ── t = 0 cashflow table ──────────────────────────────────────────────────────
print("\n--- Cashflow table at t=0 ---")
table = swap.cashflow_table(curve, t_k=0.0, r_t=r0, K=K)
print(table.to_string(index=False))

# ── Model 1a — closed-form labels ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 1a  — ClosedFormLabels")
print("=" * 70)

gen1a = ClosedFormLabels()

# t = 0: par swap must have V = 0
V0 = gen1a.value(swap, curve, t_k=0.0, r_t=r0, K=K)
D0 = gen1a.delta(swap, curve, t_k=0.0, r_t=r0, K=K)
print(f"\nt=0  V = {V0:.3e}  (expect ~0)   delta = {D0:.6f}")

# t > 0: scan across monitoring dates and rate scenarios
print(f"\n{'t_k':>6}  {'r_t%':>7}  {'V_1a':>12}  {'delta_1a':>12}")
for t_k in [1.0, 3.0, 5.0, 7.0, 9.0]:
    for r_t in [r0 - 0.005, r0, r0 + 0.005]:
        k_idx = int(round(t_k / swap.freq))
        V = gen1a.value(swap, curve, t_k=t_k, r_t=r_t, K=K)
        D = gen1a.delta(swap, curve, t_k=t_k, r_t=r_t, K=K)
        print(f"{t_k:>6.1f}  {r_t*100:>7.3f}  {V:>12.6f}  {D:>12.6f}")

# ── Model 1b — MC labels ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("MODEL 1b  — MCLabels  (M=5000, steps_per_period=20)")
print("=" * 70)

gen1b = MCLabels(M=5000, seed=42, steps_per_period=20)

print(f"\n{'t_k':>6}  {'r_t%':>7}  {'V_1a':>12}  {'V_1b':>12}  "
      f"{'err(bps)':>10}  {'d_1a':>10}  {'d_1b':>10}")

for t_k in [1.0, 3.0, 5.0, 7.0, 9.0]:
    for r_t in [r0 - 0.005, r0, r0 + 0.005]:
        Va = gen1a.value(swap, curve, t_k=t_k, r_t=r_t, K=K)
        Vb = gen1b.value(swap, curve, t_k=t_k, r_t=r_t, K=K)
        Da = gen1a.delta(swap, curve, t_k=t_k, r_t=r_t, K=K)
        Db = gen1b.delta(swap, curve, t_k=t_k, r_t=r_t, K=K)
        err_bps = (Vb - Va) * 10_000
        print(f"{t_k:>6.1f}  {r_t*100:>7.3f}  {Va:>12.6f}  {Vb:>12.6f}  "
              f"{err_bps:>10.2f}  {Da:>10.6f}  {Db:>10.6f}")

# ── t = 5 cashflow table at a bumped rate ─────────────────────────────────────
print("\n--- Cashflow table at t=5, r_t = r0 + 50bps ---")
table5 = swap.cashflow_table(curve, t_k=5.0, r_t=r0 + 0.005, K=K)
print(table5.to_string(index=False))
