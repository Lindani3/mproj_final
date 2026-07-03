"""
show_cashflows.py
Cashflow table demo using FlatCurve + IRS.
Shows t=0 (par swap) and t=5 (in-the-money) for a 10Y quarterly IRS.
"""

import numpy as np
import pandas as pd
from curves import FlatCurve
from irs    import IRS
from labels import ClosedFormLabels

pd.set_option("display.float_format", "{:.6f}".format)
pd.set_option("display.max_columns", 20)
pd.set_option("display.width", 130)

# ── Contract and curve setup ───────────────────────────────────────────────────
r0  = 0.0125   # 1.25% flat
lam = 0.10
eta = 0.015

curve = FlatCurve(r0=r0, lam=lam, eta=eta)
swap  = IRS(tenor=10.0, freq=0.25, notional=1.0)
K     = swap.k_par(curve)
gen   = ClosedFormLabels()

print("=" * 80)
print("FLAT CURVE  IRS SETUP")
print("=" * 80)
print(f"  r0      : {r0*100:.4f}%")
print(f"  lambda  : {lam}")
print(f"  eta     : {eta}")
print(f"  K^par   : {K*100:.6f}%")
print(f"  Tenor   : {swap.tenor:.0f}Y  freq: quarterly  notional: {swap.notional:.0f}")

# ── t = 0: par swap (V must be zero) ──────────────────────────────────────────
print("\n" + "=" * 80)
print("CASHFLOW TABLE  t=0  r_t = r0 = 1.25%  (par swap — V = 0)")
print("=" * 80)

V0 = gen.value(swap, curve, t_k=0.0, r_t=r0, K=K)
D0 = gen.delta(swap, curve, t_k=0.0, r_t=r0, K=K)
print(f"\n  V(0, r0) = {V0:.3e}   delta = {D0:.6f}\n")
print(swap.cashflow_table(curve, t_k=0.0, r_t=r0, K=K).to_string(index=False))

# ── t = 5: rates have risen 50 bps (payer is in the money) ────────────────────
t_k  = 5.0
r_hi = r0 + 0.005   # 1.75%

print("\n" + "=" * 80)
print(f"CASHFLOW TABLE  t={t_k:.0f}  r_t = {r_hi*100:.2f}%  (rates up 50 bps, payer ITM)")
print("=" * 80)

V5_hi = gen.value(swap, curve, t_k=t_k, r_t=r_hi, K=K)
D5_hi = gen.delta(swap, curve, t_k=t_k, r_t=r_hi, K=K)
print(f"\n  V({t_k:.0f}, {r_hi*100:.2f}%) = {V5_hi:.6f}   delta = {D5_hi:.6f}\n")
print(swap.cashflow_table(curve, t_k=t_k, r_t=r_hi, K=K).to_string(index=False))

# ── t = 5: rates have fallen 50 bps (payer is out of the money) ───────────────
r_lo = r0 - 0.005   # 0.75%

print("\n" + "=" * 80)
print(f"CASHFLOW TABLE  t={t_k:.0f}  r_t = {r_lo*100:.2f}%  (rates down 50 bps, payer OTM)")
print("=" * 80)

V5_lo = gen.value(swap, curve, t_k=t_k, r_t=r_lo, K=K)
D5_lo = gen.delta(swap, curve, t_k=t_k, r_t=r_lo, K=K)
print(f"\n  V({t_k:.0f}, {r_lo*100:.2f}%) = {V5_lo:.6f}   delta = {D5_lo:.6f}\n")
print(swap.cashflow_table(curve, t_k=t_k, r_t=r_lo, K=K).to_string(index=False))

# ── Value scan across time and rate ───────────────────────────────────────────
print("\n" + "=" * 80)
print("VALUE SCAN  V(t_k, r_t)  across monitoring dates and rate scenarios")
print("=" * 80)
print(f"\n{'t_k':>5}  {'r_t%':>7}  {'V':>12}  {'delta':>10}  {'ITM/OTM':>8}")

for t_k in [0.25, 1.0, 2.0, 5.0, 7.0, 9.0, 9.75]:
    for r_t in [r0 - 0.01, r0 - 0.005, r0, r0 + 0.005, r0 + 0.01]:
        V = gen.value(swap, curve, t_k=t_k, r_t=r_t, K=K)
        D = gen.delta(swap, curve, t_k=t_k, r_t=r_t, K=K)
        tag = "ITM" if V > 0 else ("OTM" if V < 0 else "ATM")
        print(f"{t_k:>5.2f}  {r_t*100:>7.3f}  {V:>12.6f}  {D:>10.6f}  {tag:>8}")
    print()
