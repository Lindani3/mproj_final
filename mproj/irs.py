"""
irs.py
Interest Rate Swap contract class.

Payer convention: receive floating, pay fixed.
V(t,r) = 1 - sum_j c_j * P(t, T_j, r)
where c_j = K*tau for j < n,  c_n = 1 + K*tau.

Accepts any curve object that implements bond_price and discount_factor.
"""

import numpy as np
import pandas as pd


class IRS:

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
        K = (1 - P(0,T_n)) / (freq * sum_j P(0,T_j))
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
        P     = curve.bond_price(t_k, remaining, r_arr[:, None])
        V     = 1.0 - np.sum(c * P, axis=-1)
        return float(V[0]) if np.ndim(r_t) == 0 else V

    def cashflow_table(self, curve, t_k, r_t, K):
        """
        Cashflow table for all remaining payment dates after t_k.
        Columns: T_j, tau_j, Fixed_CF, Float_CF, Net_CF,
                 P(t,Tj), PV_Fixed, PV_Float, PV_Net.
        Last row is a TOTAL footer.
        """
        k_idx     = int(np.searchsorted(self.payment_dates, t_k + 1e-9))
        remaining = self.payment_dates[k_idx:]
        tau_rem   = self.tau[k_idx:]
        prev_rem  = np.concatenate([[t_k], remaining[:-1]])

        if len(remaining) == 0:
            return pd.DataFrame()

        N        = self.notional
        r_scalar = float(np.atleast_1d(r_t)[0])

        P_j    = curve.bond_price(t_k, remaining, r_scalar)
        P_prev = curve.bond_price(t_k, prev_rem,  r_scalar)

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
