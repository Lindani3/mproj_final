"""
Validation 1: K^par construction (eq:k_par) and the par-pricing identity.

For a sample of curve dates and Hull-White parameter pairs (a, sigma), check:
  1. Calibration identity: A(0,T) exp(-B(0,T) r0) == P_market(0,T)  (no-arbitrage)
  2. V(0; K^par) == 0  (par swap at inception, eq derived from eq:k_par)

Both should hold to machine precision (~1e-12 to 1e-15) for every curve and
every (a, sigma) pair, since K^par is an algebraic identity, not an
approximation.
"""

import numpy as np
from hw_lib import load_curve, Curve, pay_dates, k_par, irs_value

CURVE_DATES = [
    "2010-01-04",
    "2014-06-16",
    "2018-12-28",
    "2020-03-23",   # COVID stress
    "2022-10-21",   # 2022 hiking cycle
    "2024-01-02",
    "2026-03-27",   # date used in test_ee_comparison.py
]

HW_PARAMS = [
    (0.05, 0.010),
    (0.10, 0.015),
    (0.20, 0.005),
    (0.001, 0.001),  # near-degenerate a, smallest sigma in our sampling range
]

dates = pay_dates()

print("=" * 78)
print("VALIDATION 1: K^par construction and par-pricing identity")
print("=" * 78)
print(f"{'Curve date':>12} {'(a, sigma)':>16} {'r0 (%)':>8} {'K^par (%)':>10} "
      f"{'max|calib err|':>15} {'|V(0;K^par)|':>14}")
print("-" * 78)

worst_calib = 0.0
worst_v0 = 0.0

for cd in CURVE_DATES:
    params = load_curve(cd)
    for a, sigma in HW_PARAMS:
        curve = Curve(params, lam=a, eta=sigma)
        K = k_par(curve, dates)
        calib_err = curve.calibration_error(dates).max()
        V0 = irs_value(curve, 0.0, curve.r0, 0, K, dates)

        worst_calib = max(worst_calib, calib_err)
        worst_v0 = max(worst_v0, abs(V0))

        print(f"{cd:>12} ({a:>5.3f},{sigma:>6.3f}) {curve.r0*100:>8.4f} "
              f"{K*100:>10.4f} {calib_err:>15.3e} {abs(V0):>14.3e}")

print("-" * 78)
print(f"Worst calibration error across all (curve, a, sigma): {worst_calib:.3e}")
print(f"Worst |V(0;K^par)| across all (curve, a, sigma)     : {worst_v0:.3e}")
print()
if worst_calib < 1e-8 and worst_v0 < 1e-8:
    print("PASS: all curves price to par at t=0 and HW calibration is exact.")
else:
    print("FAIL: identity violated, see worst-case rows above.")
