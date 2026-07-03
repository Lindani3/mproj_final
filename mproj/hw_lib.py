"""
hw_lib.py
Compatibility shim — re-exports everything from the four module files
so that existing scripts (data_generation.py, validate_model1a.py, etc.)
continue to work without import changes.

New code should import directly from:
    hw_pricer, curves, irs, labels
"""

import numpy as np
import pandas as pd

from hw_pricer import HWPricer
from curves    import (FlatCurve, Curve, SplineCurve,
                       svensson, load_curve, TENORS, DATA_PATH)
from irs       import IRS
from labels    import ClosedFormLabels, MCLabels

# ── Constants ──────────────────────────────────────────────────────────────────
TAU_PAY = 0.25
T_N     = 10.0

# ── Free functions (kept for backward compatibility) ──────────────────────────

def pay_dates(tau_pay=TAU_PAY, t_n=T_N):
    return np.arange(tau_pay, t_n + 1e-10, tau_pay)


def k_par(curve, dates=None, tau_pay=TAU_PAY):
    if dates is None:
        dates = pay_dates(tau_pay)
    P0 = curve.discount_factor(dates)
    return (1.0 - P0[-1]) / (tau_pay * P0.sum())


def coupon_weights(K, dates=None, tau_pay=TAU_PAY):
    if dates is None:
        dates = pay_dates(tau_pay)
    c      = np.full(len(dates), K * tau_pay)
    c[-1] += 1.0
    return c


def irs_value(curve, t_k, r_t, k_idx, K, dates=None, tau_pay=TAU_PAY):
    if dates is None:
        dates = pay_dates(tau_pay)
    remaining = dates[k_idx:]
    if len(remaining) == 0:
        return np.zeros_like(r_t) if hasattr(r_t, "__len__") else 0.0
    c     = coupon_weights(K, dates, tau_pay)[k_idx:]
    r_arr = np.atleast_1d(r_t)
    P     = curve.bond_price(t_k, remaining, r_arr[:, None])
    V     = 1.0 - np.sum(c * P, axis=-1)
    return V if hasattr(r_t, "__len__") else V[0]


def simulate_short_rate(curve, M, seed, dates=None, tau_pay=TAU_PAY):
    if dates is None:
        dates = pay_dates(tau_pay)
    rng     = np.random.default_rng(seed)
    K_mon   = len(dates)
    r_paths = np.zeros((M, K_mon))
    r_prev  = np.full(M, curve.r0)
    for k, t_next in enumerate(dates):
        t_prev = dates[k - 1] if k > 0 else 0.0
        dt     = t_next - t_prev
        e_lam  = np.exp(-curve.lam * dt)
        mu     = curve.fwd_rate(t_next)[0] + (r_prev - curve.fwd_rate(t_prev)[0]) * e_lam
        nu     = curve.eta * np.sqrt((1.0 - np.exp(-2.0 * curve.lam * dt)) / (2.0 * curve.lam))
        r_paths[:, k] = mu + nu * rng.standard_normal(M)
        r_prev = r_paths[:, k]
    return r_paths
