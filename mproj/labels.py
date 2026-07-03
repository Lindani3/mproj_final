"""
labels.py
Label generators for Model 1a and Model 1b.

ClosedFormLabels (Model 1a)
    Exact labels via the affine HW bond price formula.
    No simulation noise — labels are exact up to floating-point precision.

MCLabels (Model 1b)
    Simulation-based labels under HW Q-dynamics.
    Uses exact Gaussian transitions on a fine time grid.
    Labels carry O(M^{-1/2}) noise.
"""

import numpy as np


class ClosedFormLabels:
    """Model 1a: exact label generation."""

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


class MCLabels:
    """Model 1b: simulation-based label generation."""

    def __init__(self, M=1000, seed=42, steps_per_period=10):
        self.M                = int(M)
        self.seed             = int(seed)
        self.steps_per_period = int(steps_per_period)

    def _simulate(self, curve, t_k, r_t, remaining):
        """
        Simulate M paths from r_t at t_k on a fine grid that includes
        every date in remaining. Uses exact Gaussian transitions and
        trapezoidal integration for integral of r.

        Returns:
            bond_prices : (M, len(remaining))
            B_weights   : (len(remaining),)  B(t_k, T_j) for pathwise delta
        """
        rng   = np.random.default_rng(self.seed)
        dt    = (remaining[0] - t_k) / self.steps_per_period
        T_max = remaining[-1]

        fine  = np.arange(t_k + dt, T_max + 1e-10, dt)
        grid  = np.unique(np.concatenate([fine, remaining]))
        grid  = np.sort(grid[grid > t_k - 1e-9])

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
        """
        Single simulation pass returning both value and delta.
        Avoids running the simulation twice.
        """
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
