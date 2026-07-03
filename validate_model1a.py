"""
Validation 2: Model 1a (closed-form IRS valuation), per the K=K^par(x)
construction agreed for Section 3.2/3.4.

Checks:
  (a) t=0 degeneracy: V(0;K)=0 and dV(0)/dy(T_j)=0 for every tenor j,
      identically, regardless of (a, sigma) -- confirms the algebraic
      argument from the discussion.
  (b) t_k>0: V(t_k,r0;K) is non-zero and varies with t_k.
  (c) FD curve sensitivities g_{i,j} (one-sided, alg:cf) at t_k>0 agree with
      a two-sided reference to within O(Delta) truncation error, with K
      recomputed under each bumped curve (K_i^{(j)}).
"""

import numpy as np
from hw_lib import load_curve, Curve, SplineCurve, TENORS, pay_dates, k_par, irs_value

CURVE_DATE = "2026-03-27"
A_HW, SIGMA_HW = 0.10, 0.015
DELTA = 0.0001

dates = pay_dates()
params = load_curve(CURVE_DATE)
mkt = Curve(params, lam=A_HW, eta=SIGMA_HW)
y_nodes = mkt.nodal_rates(TENORS)

base = SplineCurve(y_nodes, A_HW, SIGMA_HW)
K0 = k_par(base, dates)

print("=" * 78)
print("VALIDATION 2: Model 1a closed-form")
print("=" * 78)
print(f"Curve date = {CURVE_DATE}, a={A_HW}, sigma={SIGMA_HW}")
print(f"r0 = {base.r0*100:.4f}%   K = K^par(x) = {K0*100:.4f}%")

# (a) t=0 degeneracy ------------------------------------------------------
print("\n--- (a) t=0 degeneracy ---")
V0 = irs_value(base, 0.0, base.r0, 0, K0, dates)
print(f"V(0; K^par) = {V0:.3e}  (expect ~0)")

print(f"{'tenor T_j':>10} {'dV(0)/dy(T_j)  one-sided FD':>30}")
max_grad0 = 0.0
for j, Tj in enumerate(TENORS):
    bumped = base.bumped(j, DELTA)
    Kb = k_par(bumped, dates)
    Vb = irs_value(bumped, 0.0, bumped.r0, 0, Kb, dates)
    grad = (Vb - V0) / DELTA
    max_grad0 = max(max_grad0, abs(grad))
    print(f"{Tj:>10.4f} {grad:>30.3e}")
print(f"max|dV(0)/dy(T_j)| = {max_grad0:.3e}  (expect ~0)")

# (b) t_k>0 values ----------------------------------------------------------
print("\n--- (b) V(t_k, r0; K) for t_k>0 ---")
t_check = [1.0, 3.0, 5.0, 7.0, 9.0]
print(f"{'t_k':>6} {'k_idx':>6} {'V(t_k,r0;K)':>14}")
for t in t_check:
    k_idx = int(round(t / 0.25))
    V = irs_value(base, t, base.r0, k_idx, K0, dates)
    print(f"{t:>6.2f} {k_idx:>6d} {V:>14.6f}")

# (c) FD sensitivities at t_k>0, K recomputed under bump --------------------
print("\n--- (c) FD curve sensitivities at t_k=5y, K recomputed per bump ---")
t = 5.0
k_idx = int(round(t / 0.25))
V_base = irs_value(base, t, base.r0, k_idx, K0, dates)
print(f"{'tenor T_j':>10} {'one-sided g':>14} {'two-sided g':>14} {'rel.diff':>10}")
for j, Tj in enumerate(TENORS):
    up = base.bumped(j, DELTA)
    dn = base.bumped(j, -DELTA)
    Kup, Kdn = k_par(up, dates), k_par(dn, dates)
    Vup = irs_value(up, t, up.r0, k_idx, Kup, dates)
    Vdn = irs_value(dn, t, dn.r0, k_idx, Kdn, dates)
    g_one = (Vup - V_base) / DELTA
    g_two = (Vup - Vdn) / (2 * DELTA)
    reldiff = abs(g_one - g_two) / (abs(g_two) + 1e-12)
    print(f"{Tj:>10.4f} {g_one:>14.4f} {g_two:>14.4f} {reldiff:>10.2e}")

print("\nPASS criteria: V(0)~0, max|grad V(0)|~0 (both < 1e-6); "
      "V(t_k>0) clearly nonzero; one-sided vs two-sided FD agree to a few %.")
