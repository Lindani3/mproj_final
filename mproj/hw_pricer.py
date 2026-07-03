"""
hw_pricer.py
Hull-White bond pricing mixin.

HWPricer requires subclasses to supply:
    discount_factor(T), fwd_rate(T), r0, lam, eta
It provides all HW bond pricing and Q-measure moment formulas.
"""

import numpy as np


class HWPricer:

    def B(self, t, T):
        """B(t,T) = (1 - exp(-lam*(T-t))) / lam"""
        return (1.0 - np.exp(-self.lam * (T - t))) / self.lam

    def lnA(self, t, T):
        Bt   = self.B(t, T)
        conv = (self.eta**2 / (4.0 * self.lam)) * Bt**2 * (1.0 - np.exp(-2.0 * self.lam * t))
        return (np.log(self.discount_factor(T) / self.discount_factor(t))
                + Bt * self.fwd_rate(t) - conv)

    def A(self, t, T):
        return np.exp(self.lnA(t, T))

    def bond_price(self, t, T, r_t):
        """P(t,T;r_t) = A(t,T) * exp(-B(t,T) * r_t)"""
        return self.A(t, T) * np.exp(-self.B(t, T) * r_t)

    def calibration_error(self, tenors):
        """|P_model(0,T) - P_market(0,T)| at each tenor."""
        P_model  = self.A(0.0, tenors) * np.exp(-self.B(0.0, tenors) * self.r0)
        P_market = self.discount_factor(tenors)
        return np.abs(P_model - P_market)

    def alpha(self, t):
        """E^Q[r_t] = f(0,t) + (eta^2 / 2*lam^2) * (1 - exp(-lam*t))^2"""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return (self.fwd_rate(t)
                + (self.eta**2 / (2.0 * self.lam**2))
                * (1.0 - np.exp(-self.lam * t))**2)

    def var_r(self, t):
        """Var^Q[r_t] = (eta^2 / 2*lam) * (1 - exp(-2*lam*t))"""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        return (self.eta**2 / (2.0 * self.lam)) * (1.0 - np.exp(-2.0 * self.lam * t))
