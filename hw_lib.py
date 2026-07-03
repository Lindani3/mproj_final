"""
Shared Hull-White / IRS library used by the validation scripts.

Extracted and consolidated from hull_white.ipynb. All functions operate on a
single market curve (a row of feds200628.csv) and a chosen pair of
Hull-White parameters (lam, eta).
"""

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

DATA_PATH = "feds200628.csv"

# Tenor grid used for the input vector {y(T_j)} (proposal eq:input_vec)
TENORS = np.array([1/12, 0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])


def load_curve(curve_date, data_path=DATA_PATH):
    """Load Svensson parameters for a given date from the FEDS200628 dataset."""
    feds = pd.read_csv(data_path, skiprows=9, na_values="NA")
    feds["Date"] = pd.to_datetime(feds["Date"])
    feds = feds.set_index("Date")
    row = feds.loc[curve_date]
    return dict(
        beta0=row["BETA0"], beta1=row["BETA1"], beta2=row["BETA2"], beta3=row["BETA3"],
        tau1=row["TAU1"], tau2=row["TAU2"],
    )


def svensson(T, b0, b1, b2, b3, t1, t2):
    """Svensson (1994) zero rate (%) at maturity T (years)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    safe = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-safe / t1)
    e2 = np.exp(-safe / t2)
    term1 = (1.0 - e1) / (safe / t1)
    term2 = term1 - e1
    term3 = (1.0 - e2) / (safe / t2) - e2
    y = b0 + b1 * term1 + b2 * term2 + b3 * term3
    return np.where(T < 1e-10, b0 + b1, y)


class HWPricer:
    """
    Mixin providing Hull-White bond pricing and Q-dynamics given that the
    subclass exposes self.lam, self.eta, self.r0, self.zero_rate(T),
    self.discount_factor(T) and self.fwd_rate(T).
    """

    # ---- Hull-White bond pricing --------------------------------------
    def B(self, t, T):
        return (1.0 - np.exp(-self.lam * (T - t))) / self.lam

    def lnA(self, t, T):
        Bt = self.B(t, T)
        conv = (self.eta**2 / (4.0 * self.lam)) * Bt**2 * (1.0 - np.exp(-2.0 * self.lam * t))
        return np.log(self.discount_factor(T) / self.discount_factor(t)) + Bt * self.fwd_rate(t) - conv

    def A(self, t, T):
        return np.exp(self.lnA(t, T))

    def bond_price(self, t, T, r_t):
        return self.A(t, T) * np.exp(-self.B(t, T) * r_t)

    def calibration_error(self, tenors):
        """|A(0,T) exp(-B(0,T) r0) - P_market(0,T)|, should be ~0 (no-arbitrage)."""
        P_model = self.A(0.0, tenors) * np.exp(-self.B(0.0, tenors) * self.r0)
        P_market = self.discount_factor(tenors)
        return np.abs(P_model - P_market)

    # ---- Q-measure dynamics of r_t -------------------------------------
    def alpha(self, t):
        """E^Q[r_t] = f(0,t) + (eta^2/(2 lam^2)) (1-exp(-lam t))^2."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return self.fwd_rate(t) + (self.eta**2 / (2.0 * self.lam**2)) * (1.0 - np.exp(-self.lam * t)) ** 2

    def var_r(self, t):
        """Var^Q[r_t] = eta^2/(2 lam) (1-exp(-2 lam t))."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return (self.eta**2 / (2.0 * self.lam)) * (1.0 - np.exp(-2.0 * self.lam * t))


class Curve(HWPricer):
    """A market yield curve (Svensson) plus a Hull-White (lam, eta) pair."""

    def __init__(self, params, lam, eta):
        self.b0, self.b1, self.b2, self.b3 = (
            params["beta0"], params["beta1"], params["beta2"], params["beta3"]
        )
        self.t1, self.t2 = params["tau1"], params["tau2"]
        self.lam = lam
        self.eta = eta
        self.r0 = (self.b0 + self.b1) / 100.0

    def zero_rate(self, T):
        return svensson(T, self.b0, self.b1, self.b2, self.b3, self.t1, self.t2) / 100.0

    def discount_factor(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.exp(-self.zero_rate(T) * T)

    def fwd_rate(self, T, dt=1e-7):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        Tl = np.maximum(T - dt, 1e-8)
        Tr = T + dt
        return self.zero_rate(T) + T * (self.zero_rate(Tr) - self.zero_rate(Tl)) / (Tr - Tl)

    def nodal_rates(self, tenors=TENORS):
        """Zero rates at the input-vector tenor grid {y(T_j)}."""
        return self.zero_rate(tenors)


class SplineCurve(HWPricer):
    """
    Curve represented by its nodal zero rates {y(T_j)} on TENORS, via cubic
    spline interpolation. Used to realise the +/-Delta tenor bumps in
    eq:neighbourhood: bumping y_nodes[j] by Delta and rebuilding the spline
    perturbs the curve only in the neighbourhood of T_j.
    """

    def __init__(self, y_nodes, lam, eta, tenors=TENORS):
        self.tenors = np.asarray(tenors, dtype=float)
        self.y_nodes = np.asarray(y_nodes, dtype=float)
        self.lam = lam
        self.eta = eta
        self._spline = CubicSpline(self.tenors, self.y_nodes, extrapolate=True)
        self._dspline = self._spline.derivative()
        self.r0 = float(self._spline(0.0))

    def zero_rate(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return self._spline(np.maximum(T, 0.0))

    def discount_factor(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        return np.exp(-self.zero_rate(T) * T)

    def fwd_rate(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        Tc = np.maximum(T, 0.0)
        return self.zero_rate(Tc) + Tc * self._dspline(Tc)

    def bumped(self, j, delta):
        """Return a new SplineCurve with y_nodes[j] += delta."""
        y = self.y_nodes.copy()
        y[j] += delta
        return SplineCurve(y, self.lam, self.eta, self.tenors)


# ---- IRS contract -------------------------------------------------------
TAU_PAY = 0.25
T_N = 10.0


def pay_dates(tau_pay=TAU_PAY, t_n=T_N):
    return np.arange(tau_pay, t_n + 1e-10, tau_pay)


def k_par(curve, dates=None, tau_pay=TAU_PAY):
    """K^par(x) via eq:k_par: (1 - P(0,T_n)) / (tau * sum P(0,T_j))."""
    if dates is None:
        dates = pay_dates(tau_pay)
    P0 = curve.discount_factor(dates)
    return (1.0 - P0[-1]) / (tau_pay * P0.sum())


def coupon_weights(K, dates=None, tau_pay=TAU_PAY):
    """c_j = K*tau for j<n, c_n = 1 + K*tau (unit notional, receive-fixed annuity form)."""
    if dates is None:
        dates = pay_dates(tau_pay)
    c = np.full(len(dates), K * tau_pay)
    c[-1] += 1.0
    return c


def irs_value(curve, t_k, r_t, k_idx, K, dates=None, tau_pay=TAU_PAY):
    """
    V(t_k; r_t) = (1 - P(t_k,T_n)) - K*tau*sum_{j>k} P(t_k,T_j)
                = 1 - sum_{j>k} c_j P(t_k,T_j)   (unit notional)
    k_idx: number of payment dates already passed (0 for t_k=0).
    r_t may be scalar or array.
    """
    if dates is None:
        dates = pay_dates(tau_pay)
    remaining = dates[k_idx:]
    if len(remaining) == 0:
        return np.zeros_like(r_t) if hasattr(r_t, "__len__") else 0.0
    c = coupon_weights(K, dates, tau_pay)[k_idx:]
    r_arr = np.atleast_1d(r_t)
    P = curve.bond_price(t_k, remaining, r_arr[:, None])
    V = 1.0 - np.sum(c * P, axis=-1)
    return V if hasattr(r_t, "__len__") else V[0]


def simulate_short_rate(curve, M, seed, dates=None, tau_pay=TAU_PAY):
    """Exact Gaussian simulation of r_t under Q at each monitoring date."""
    if dates is None:
        dates = pay_dates(tau_pay)
    rng = np.random.default_rng(seed)
    K_mon = len(dates)
    r_paths = np.zeros((M, K_mon))
    r_prev = np.full(M, curve.r0)
    for k, t_next in enumerate(dates):
        t_prev = dates[k - 1] if k > 0 else 0.0
        dt = t_next - t_prev
        e_lam = np.exp(-curve.lam * dt)
        mu = curve.fwd_rate(t_next)[0] + (r_prev - curve.fwd_rate(t_prev)[0]) * e_lam
        nu = curve.eta * np.sqrt((1.0 - np.exp(-2.0 * curve.lam * dt)) / (2.0 * curve.lam))
        r_paths[:, k] = mu + nu * rng.standard_normal(M)
        r_prev = r_paths[:, k]
    return r_paths


# =============================================================================
# FlatCurve
# =============================================================================

class FlatCurve(HWPricer):
    """
    Flat yield curve: P(0,T) = exp(-r0*T), fwd_rate(T) = r0 everywhere.
    Inherits HW bond pricing from HWPricer.
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


# =============================================================================
# IRS
# =============================================================================

class IRS:
    """
    Fixed-for-floating interest rate swap.
    Payer convention: receive floating, pay fixed.
    V(t,r) = 1 - sum_j c_j * P(t, T_j, r)
    where c_j = K*tau for j < n, c_n = 1 + K*tau.
    """

    def __init__(self, tenor, freq=0.25, notional=1.0):
        self.tenor         = float(tenor)
        self.freq          = float(freq)
        self.notional      = float(notional)
        self.payment_dates = np.arange(freq, tenor + 1e-10, freq)
        self.prev_dates    = np.concatenate([[0.0], self.payment_dates[:-1]])
        self.tau           = self.payment_dates - self.prev_dates

    def _coupon_weights(self, K):
        """c_j = K*tau for j < n,  c_n = 1 + K*tau."""
        c      = np.full(len(self.payment_dates), K * self.freq)
        c[-1] += 1.0
        return c

    def k_par(self, curve):
        """
        Par fixed rate from the no-arbitrage condition:
        K = (P(0,T_0) - P(0,T_n)) / (freq * sum_j P(0,T_j))
        With T_0 = 0: P(0,0) = 1.
        """
        P = curve.discount_factor(self.payment_dates)
        return (1.0 - float(P[-1])) / (self.freq * float(P.sum()))

    def value(self, curve, t_k, r_t, K):
        """
        IRS MtM at (t_k, r_t) using closed-form bond prices from curve.
        r_t may be scalar or 1-D array for vectorised pricing.
        """
        k_idx     = int(np.searchsorted(self.payment_dates, t_k + 1e-9))
        remaining = self.payment_dates[k_idx:]
        if len(remaining) == 0:
            return 0.0
        c     = self._coupon_weights(K)[k_idx:]
        r_arr = np.atleast_1d(np.asarray(r_t, dtype=float))
        P     = curve.bond_price(t_k, remaining, r_arr[:, None])  # (N_r, N_dates)
        V     = 1.0 - np.sum(c * P, axis=-1)
        return float(V[0]) if np.ndim(r_t) == 0 else V

    def cashflow_table(self, curve, t_k, r_t, K):
        """
        Cashflow table for remaining payment dates after t_k.
        Returns a DataFrame with fixed CF, float CF, net CF, discount
        factors, and present values, plus a TOTAL footer row.
        """
        k_idx     = int(np.searchsorted(self.payment_dates, t_k + 1e-9))
        remaining = self.payment_dates[k_idx:]
        tau_rem   = self.tau[k_idx:]
        prev_rem  = np.concatenate([[t_k], remaining[:-1]])

        if len(remaining) == 0:
            return pd.DataFrame()

        N        = self.notional
        r_scalar = float(np.atleast_1d(r_t)[0])

        P_j    = curve.bond_price(t_k, remaining, r_scalar)   # (n,)
        P_prev = curve.bond_price(t_k, prev_rem,  r_scalar)   # (n,)

        fixed_cf = N * K * tau_rem
        float_cf = N * (P_prev / P_j - 1.0)
        net_cf   = float_cf - fixed_cf

        pv_fixed = fixed_cf * P_j
        pv_float = float_cf * P_j
        pv_net   = pv_float - pv_fixed

        table = pd.DataFrame({
            "T_j":      remaining,
            "tau_j":    tau_rem,
            "Fixed_CF": fixed_cf,
            "Float_CF": float_cf,
            "Net_CF":   net_cf,
            "P(t,Tj)":  P_j,
            "PV_Fixed": pv_fixed,
            "PV_Float": pv_float,
            "PV_Net":   pv_net,
        })

        footer = pd.DataFrame([{
            "T_j":      "TOTAL",
            "tau_j":    tau_rem.sum(),
            "Fixed_CF": fixed_cf.sum(),
            "Float_CF": float_cf.sum(),
            "Net_CF":   net_cf.sum(),
            "P(t,Tj)":  np.nan,
            "PV_Fixed": pv_fixed.sum(),
            "PV_Float": pv_float.sum(),
            "PV_Net":   pv_net.sum(),
        }])

        return pd.concat([table, footer], ignore_index=True)


# =============================================================================
# ClosedFormLabels  —  Model 1a
# =============================================================================

class ClosedFormLabels:
    """
    Exact label generation using the affine HW bond price formula.
    No simulation — labels are exact up to floating-point precision.
    """

    def value(self, irs, curve, t_k, r_t, K):
        """IRS MtM via closed-form affine bond prices."""
        return irs.value(curve, t_k, r_t, K)

    def delta(self, irs, curve, t_k, r_t, K):
        """
        Closed-form rate delta:
        dV/dr_t = sum_j c_j * B(t_k, T_j) * P(t_k, T_j, r_t)
        """
        k_idx     = int(np.searchsorted(irs.payment_dates, t_k + 1e-9))
        remaining = irs.payment_dates[k_idx:]
        if len(remaining) == 0:
            return 0.0
        c        = irs._coupon_weights(K)[k_idx:]
        r_scalar = float(np.atleast_1d(r_t)[0])
        P        = curve.bond_price(t_k, remaining, r_scalar)
        B        = curve.B(t_k, remaining)
        return float((c * B * P).sum())


# =============================================================================
# MCLabels  —  Model 1b
# =============================================================================

class MCLabels:
    """
    Simulation-based label generation under Hull-White Q-dynamics.
    Uses exact Gaussian transitions on a fine grid.
    Labels carry O(M^{-1/2}) noise.
    """

    def __init__(self, M=1000, seed=42, steps_per_period=10):
        self.M              = int(M)
        self.seed           = int(seed)
        self.steps_per_period = int(steps_per_period)

    def _simulate(self, curve, t_k, r_t, remaining):
        """
        Simulate M paths from r_t at t_k on a fine grid that includes
        every date in remaining. Returns:
            bond_prices : (M, len(remaining))  exp(-integral r ds) per path
            B_weights   : (len(remaining),)    B(t_k, T_j) for pathwise delta
        """
        rng   = np.random.default_rng(self.seed)
        dt    = (remaining[0] - t_k) / self.steps_per_period
        T_max = remaining[-1]

        # Fine grid that always includes the payment dates exactly
        fine  = np.arange(t_k + dt, T_max + 1e-10, dt)
        grid  = np.unique(np.concatenate([fine, remaining]))
        grid  = np.sort(grid[grid > t_k - 1e-9])

        # Index in grid where each payment date falls
        pay_idx = np.array(
            [np.searchsorted(grid, T_j - 1e-9) for T_j in remaining]
        )

        r_prev    = np.full(self.M, float(r_t))
        int_r     = np.zeros(self.M)
        t_prev    = t_k
        integrals = np.zeros((self.M, len(remaining)))
        pay_ptr   = 0

        for k, t_next in enumerate(grid):
            dt_k  = t_next - t_prev
            e_lam = np.exp(-curve.lam * dt_k)
            f_cur = float(curve.fwd_rate(np.array([t_next]))[0])
            f_prv = float(curve.fwd_rate(np.array([t_prev]))[0])
            mu    = f_cur + (r_prev - f_prv) * e_lam
            nu    = curve.eta * np.sqrt(
                max((1.0 - np.exp(-2.0 * curve.lam * dt_k))
                    / (2.0 * curve.lam), 0.0)
            )
            r_curr = mu + nu * rng.standard_normal(self.M)

            # Trapezoidal integral approximation
            int_r += 0.5 * (r_prev + r_curr) * dt_k

            if pay_ptr < len(remaining) and k == pay_idx[pay_ptr]:
                integrals[:, pay_ptr] = int_r
                pay_ptr += 1

            r_prev = r_curr
            t_prev = t_next

        bond_prices = np.exp(-integrals)
        B_weights   = curve.B(t_k, remaining)
        return bond_prices, B_weights

    def _labels(self, irs, curve, t_k, r_t, K):
        """Single simulation pass returning both value and delta."""
        k_idx     = int(np.searchsorted(irs.payment_dates, t_k + 1e-9))
        remaining = irs.payment_dates[k_idx:]
        if len(remaining) == 0:
            return 0.0, 0.0
        c = irs._coupon_weights(K)[k_idx:]

        bond_prices, B_weights = self._simulate(curve, t_k, r_t, remaining)

        V_paths     = 1.0 - (bond_prices * c).sum(axis=1)
        delta_paths = (bond_prices * c * B_weights).sum(axis=1)

        return float(V_paths.mean()), float(delta_paths.mean())

    def value(self, irs, curve, t_k, r_t, K):
        v, _ = self._labels(irs, curve, t_k, r_t, K)
        return v

    def delta(self, irs, curve, t_k, r_t, K):
        _, d = self._labels(irs, curve, t_k, r_t, K)
        return d
