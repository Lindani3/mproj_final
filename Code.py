import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Contract parameters
R0    = 0.05
A_HW  = 0.10
S_HW  = 0.05
TAU   = 0.5
N_PAY = 20
T_EXP = 1.0
NOTL  = 1_000_000.0

PAY_DATES = T_EXP + np.arange(1, N_PAY + 1) * TAU   # 1.5, 2.0, ..., 11.0
BERM_EXER = np.arange(1.0, 10.0, 1.0)                # 1, 2, ..., 9

# For a flat curve the instantaneous forward rate is constant
F0 = R0

def P0(T):
    return np.exp(-R0 * np.asarray(T, dtype=float))


# Hull-White bond pricing: P(t,T;r) = A(t,T) exp(-B(t,T) r)

def hw_B(t, T):
    return (1.0 - np.exp(-A_HW * (T - t))) / A_HW

def hw_lnA(t, T):
    Bt = hw_B(t, T)
    # Convexity correction ensures P(0,T;r0) = P_market(0,T) exactly
    return (np.log(P0(T) / P0(t))
            + Bt * F0
            - (S_HW**2 / (4.0 * A_HW)) * (1.0 - np.exp(-2.0 * A_HW * t)) * Bt**2)

def hw_bond(t, T, r):
    t = float(t)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    return np.exp(hw_lnA(t, T)) * np.exp(-hw_B(t, T) * r)

def alpha_hw(t):
    # E^Q[r_t] under the flat-curve specialisation of Hull-White
    t = np.asarray(t, dtype=float)
    return F0 + (S_HW**2 / (2.0 * A_HW**2)) * (1.0 - np.exp(-A_HW * t))**2


# Par rate and cashflows

def par_rate():
    Ps = P0(PAY_DATES)
    return (P0(T_EXP) - Ps[-1]) / (TAU * Ps.sum())

K_PAR = par_rate()

X_CF = np.full(N_PAY, NOTL * K_PAR * TAU)
X_CF[-1] += NOTL    # final cashflow carries notional redemption


def intrinsic(t, r):
    """Payer swap value at (t, r) for remaining cashflows."""
    r   = np.atleast_1d(np.asarray(r, dtype=float))
    idx = PAY_DATES > t + 1e-10
    if not idx.any():
        return np.zeros(len(r))
    Ti   = PAY_DATES[idx]
    Xi   = X_CF[idx]
    Pmat = hw_bond(t, Ti[np.newaxis, :], r[:, np.newaxis])
    return NOTL - (Xi * Pmat).sum(axis=1)


# European swaption — Jamshidian decomposition

def european_jamshidian():
    """
    Reduces the coupon bond option to a portfolio of ZBOs via the
    critical rate r*, then prices each ZBO in closed form.
    """
    t0 = T_EXP

    def g(r):
        return np.sum(X_CF * hw_bond(t0, PAY_DATES, r)) - NOTL

    r_star  = brentq(g, -0.5, 0.5, xtol=1e-12)
    P0_t0   = P0(t0)
    vol_fac = np.sqrt((1.0 - np.exp(-2.0 * A_HW * t0)) / (2.0 * A_HW))

    V = 0.0
    for i, Ti in enumerate(PAY_DATES):
        Ki    = hw_bond(t0, Ti, r_star)
        Fi    = P0(Ti) / P0_t0
        sig_P = S_HW * hw_B(t0, Ti) * vol_fac
        d1    = (np.log(Fi / Ki) + 0.5 * sig_P**2) / sig_P
        d2    = d1 - sig_P
        zbo   = P0_t0 * (Ki * norm.cdf(-d2) - Fi * norm.cdf(-d1))
        V    += X_CF[i] * zbo

    return V, r_star


# Bermudan swaption — LSMC (Longstaff & Schwartz 2001)

def bermudan_lsmc(M=100_000, seed=42):
    """
    Simulate M paths of r_t exactly (Gaussian transitions), then work
    backwards regressing discounted continuation values on Laguerre
    polynomials of r_t. Antithetic variates halve the variance at no
    extra simulation cost.
    """
    rng  = np.random.default_rng(seed)
    exer = BERM_EXER
    m    = len(exer)

    # Simulate short rate at each exercise date
    r_paths = np.zeros((M, m))
    r_prev  = np.full(M, R0)
    t_prev  = 0.0

    for idx, t_next in enumerate(exer):
        dt_step = t_next - t_prev
        e_lam   = np.exp(-A_HW * dt_step)
        mu_r    = r_prev * e_lam + alpha_hw(t_next) - alpha_hw(t_prev) * e_lam
        nu_r    = S_HW * np.sqrt((1.0 - np.exp(-2.0 * A_HW * dt_step)) / (2.0 * A_HW))
        z = rng.standard_normal(M // 2)
        z = np.concatenate([z, -z])
        r_paths[:, idx] = mu_r + nu_r * z
        r_prev = r_paths[:, idx]
        t_prev = t_next

    def laguerre_basis(r):
        r = np.asarray(r, dtype=float)
        e = np.exp(-r / 2.0)
        return np.column_stack([
            e,
            e * (1.0 - r),
            e * (1.0 - 2.0*r + 0.5*r**2),
        ])

    intr = np.zeros((M, m))
    for idx, t in enumerate(exer):
        intr[:, idx] = intrinsic(t, r_paths[:, idx])

    h = np.maximum(intr[:, -1], 0.0)

    for idx in range(m - 2, -1, -1):
        dt_s = exer[idx + 1] - exer[idx]
        disc = np.exp(-r_paths[:, idx] * dt_s)
        Y    = disc * h

        itm = intr[:, idx] > 0.0
        if itm.sum() < 10:
            h = Y
            continue

        # Regress on ITM paths only — Longstaff-Schwartz design choice
        Phi   = laguerre_basis(r_paths[itm, idx])
        beta  = np.linalg.lstsq(Phi, Y[itm], rcond=None)[0]
        C_hat = Phi @ beta

        h_new      = Y.copy()
        ex_mask    = intr[itm, idx] >= C_hat
        h_new[itm] = np.where(ex_mask, intr[itm, idx], Y[itm])
        h = h_new

    disc0 = np.exp(-R0 * exer[0])
    V0    = disc0 * h.mean()
    se    = disc0 * h.std() / np.sqrt(M)
    return V0, se


if __name__ == "__main__":
    print(f"\n  Flat curve {R0*100:.0f}%  |  a={A_HW}  sigma={S_HW}"
          f"  |  K={K_PAR*100:.4f}%  N=R{NOTL:,.0f}\n")

    V_eur, r_star = european_jamshidian()
    print(f"  European  (Jamshidian)    R{V_eur:>12,.2f}   r* = {r_star*100:.4f}%\n")

    V_lsmc = None
    for M_paths in [10_000, 50_000, 100_000]:
        V_lsmc, se = bermudan_lsmc(M=M_paths)
        print(f"  Bermudan  (LSMC M={M_paths:<7,}) R{V_lsmc:>12,.2f}   se=R{se:,.0f}")

    print(f"\n  Early exercise premium  (LSMC) R{V_lsmc - V_eur:,.2f}\n")
