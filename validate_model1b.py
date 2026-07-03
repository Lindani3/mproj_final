"""
Validation 3: Model 1b (Monte Carlo IRS valuation).

K is fixed once at K^par(x) from the t=0 curve (eq:k_par), as established for
Models 1b/2, and reused at every simulated r_{t_k}.

Checks:
  (a) E[V(t_k,r_{t_k})] from MC matches the closed-form lognormal-moment
      expectation E[A(t_k,T_j) exp(-B(t_k,T_j) r_{t_k})]
        = A(t_k,T_j) exp(-B mu_k + 0.5 B^2 nu_k^2),
      with mu_k=E^Q[r_tk] (curve.alpha), nu_k^2=Var^Q[r_tk] (curve.var_r).
  (b) MC standard error scales as O(M^{-1/2}).
"""

import numpy as np
from hw_lib import (load_curve, Curve, SplineCurve, TENORS, pay_dates,
                     k_par, irs_value, simulate_short_rate, coupon_weights)

CURVE_DATE = "2026-03-27"
A_HW, SIGMA_HW = 0.10, 0.015

dates = pay_dates()
params = load_curve(CURVE_DATE)
mkt = Curve(params, lam=A_HW, eta=SIGMA_HW)
y_nodes = mkt.nodal_rates(TENORS)
base = SplineCurve(y_nodes, A_HW, SIGMA_HW)
K0 = k_par(base, dates)

print("=" * 78)
print("VALIDATION 3: Model 1b Monte Carlo")
print("=" * 78)
print(f"Curve date = {CURVE_DATE}, a={A_HW}, sigma={SIGMA_HW}, K={K0*100:.4f}%")


def exact_expectation(curve, t_k, k_idx, K, dates):
    remaining = dates[k_idx:]
    if len(remaining) == 0:
        return 0.0
    c = coupon_weights(K, dates)[k_idx:]
    mu = curve.alpha(t_k)[0]
    nu2 = curve.var_r(t_k)[0]
    B = curve.B(t_k, remaining)
    A = curve.A(t_k, remaining)
    EP = A * np.exp(-B * mu + 0.5 * B**2 * nu2)
    return 1.0 - np.sum(c * EP)


# (a) MC mean vs exact expectation, all monitoring dates --------------------
M = 10_000
r_paths = simulate_short_rate(base, M, seed=42, dates=dates)

print(f"\n--- (a) MC (M={M}) vs lognormal-moment expectation, all monitoring dates ---")
print(f"{'t_k':>6} {'k_idx':>6} {'E[V] exact':>12} {'MC mean':>12} {'diff':>10} {'MC stderr':>10}")
checks = [4, 12, 20, 28, 36, 39]
for k_idx in checks:
    t_k = dates[k_idx - 1]
    V_exact = exact_expectation(base, t_k, k_idx, K0, dates)
    V_mc_paths = irs_value(base, t_k, r_paths[:, k_idx - 1], k_idx, K0, dates)
    V_mc = V_mc_paths.mean()
    se = V_mc_paths.std(ddof=1) / np.sqrt(M)
    print(f"{t_k:>6.2f} {k_idx:>6d} {V_exact:>12.6f} {V_mc:>12.6f} "
          f"{V_mc - V_exact:>10.2e} {se:>10.2e}")

# (b) standard error scaling with M -----------------------------------------
print("\n--- (b) MC standard error scaling, t_k=5y ---")
k_idx = 20
t_k = dates[k_idx - 1]
V_exact = exact_expectation(base, t_k, k_idx, K0, dates)
print(f"E[V(5y)] exact = {V_exact:.6f}")
print(f"{'M':>8} {'MC mean':>12} {'|err|':>10} {'stderr':>10} {'stderr*sqrt(M)':>16}")
for M_test in [1000, 5000, 10000, 50000, 200000]:
    rp = simulate_short_rate(base, M_test, seed=123, dates=dates)
    Vp = irs_value(base, t_k, rp[:, k_idx - 1], k_idx, K0, dates)
    se = Vp.std(ddof=1) / np.sqrt(M_test)
    print(f"{M_test:>8d} {Vp.mean():>12.6f} {abs(Vp.mean()-V_exact):>10.2e} "
          f"{se:>10.2e} {se*np.sqrt(M_test):>16.4e}")

print("\nPASS criteria: MC mean within ~2-3 stderr of exact expectation at "
      "every monitoring date; stderr*sqrt(M) approximately constant (O(M^-1/2)).")
