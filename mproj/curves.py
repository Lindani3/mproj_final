"""
curves.py
Yield curve classes: FlatCurve, Curve (Svensson), SplineCurve.

All three inherit HWPricer and implement the curve interface:
    discount_factor(T), fwd_rate(T), zero_rate(T), r0, lam, eta

Switching curves requires only changing which object is constructed —
IRS, ClosedFormLabels, and MCLabels all accept any curve object.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from hw_pricer import HWPricer

DATA_PATH = "feds200628.csv"
TENORS    = np.array([1/12, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


# ── Svensson helpers ──────────────────────────────────────────────────────────

def svensson(T, b0, b1, b2, b3, t1, t2):
    """Svensson (1994) zero rate (%) at maturity T (years)."""
    T    = np.atleast_1d(np.asarray(T, dtype=float))
    safe = np.where(T < 1e-10, 1e-10, T)
    e1   = np.exp(-safe / t1)
    e2   = np.exp(-safe / t2)
    t1_  = (1.0 - e1) / (safe / t1)
    t2_  = t1_ - e1
    t3_  = (1.0 - e2) / (safe / t2) - e2
    y    = b0 + b1 * t1_ + b2 * t2_ + b3 * t3_
    return np.where(T < 1e-10, b0 + b1, y)


def load_curve(curve_date, data_path=DATA_PATH):
    """Load Svensson parameters for a given date from FEDS200628."""
    feds         = pd.read_csv(data_path, skiprows=9, na_values="NA")
    feds["Date"] = pd.to_datetime(feds["Date"])
    feds         = feds.set_index("Date")
    row = feds.loc[curve_date]
    return dict(
        beta0=row["BETA0"], beta1=row["BETA1"],
        beta2=row["BETA2"], beta3=row["BETA3"],
        tau1=row["TAU1"],   tau2=row["TAU2"],
    )


# ── Curve classes ─────────────────────────────────────────────────────────────

class FlatCurve(HWPricer):
    """
    Flat yield curve: P(0,T) = exp(-r0*T), fwd_rate = r0 everywhere.
    Used for Model 1a/1b testing before moving to historical curves.
    """

    def __init__(self, r0, lam, eta):
        self.r0  = float(r0)
        self.lam = float(lam)
        self.eta = float(eta)

    def zero_rate(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.full_like(T, self.r0)

    def discount_factor(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.exp(-self.r0 * T)

    def fwd_rate(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.full_like(T, self.r0)


class Curve(HWPricer):
    """Svensson parametric yield curve + Hull-White parameters."""

    def __init__(self, params, lam, eta):
        self.b0, self.b1 = params["beta0"], params["beta1"]
        self.b2, self.b3 = params["beta2"], params["beta3"]
        self.t1, self.t2 = params["tau1"],  params["tau2"]
        self.lam = float(lam)
        self.eta = float(eta)
        self.r0  = (self.b0 + self.b1) / 100.0

    def zero_rate(self, T):
        return svensson(T, self.b0, self.b1, self.b2,
                        self.b3, self.t1, self.t2) / 100.0

    def discount_factor(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.exp(-self.zero_rate(T) * T)

    def fwd_rate(self, T, dt=1e-7):
        T  = np.atleast_1d(np.asarray(T, dtype=float))
        Tl = np.maximum(T - dt, 1e-8)
        Tr = T + dt
        return self.zero_rate(T) + T * (self.zero_rate(Tr) - self.zero_rate(Tl)) / (Tr - Tl)

    def nodal_rates(self, tenors=TENORS):
        return self.zero_rate(tenors)


class SplineCurve(HWPricer):
    """
    Yield curve defined by nodal zero rates on TENORS via cubic spline.
    bumped(j, delta) returns a new SplineCurve with y_nodes[j] += delta,
    used for finite-difference curve sensitivities in data generation.
    """

    def __init__(self, y_nodes, lam, eta, tenors=TENORS):
        self.tenors  = np.asarray(tenors, dtype=float)
        self.y_nodes = np.asarray(y_nodes, dtype=float)
        self.lam     = float(lam)
        self.eta     = float(eta)
        self._spline  = CubicSpline(self.tenors, self.y_nodes, extrapolate=True)
        self._dspline = self._spline.derivative()
        self.r0      = float(self._spline(0.0))

    def zero_rate(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return self._spline(np.maximum(T, 0.0))

    def discount_factor(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.exp(-self.zero_rate(T) * T)

    def fwd_rate(self, T):
        T  = np.atleast_1d(np.asarray(T, dtype=float))
        Tc = np.maximum(T, 0.0)
        return self.zero_rate(Tc) + Tc * self._dspline(Tc)

    def bumped(self, j, delta):
        """New SplineCurve with y_nodes[j] shifted by delta."""
        y     = self.y_nodes.copy()
        y[j] += delta
        return SplineCurve(y, self.lam, self.eta, self.tenors)
