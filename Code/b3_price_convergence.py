"""
b3_price_convergence.py
=======================
Convergence study: Hull-White affine (closed-form) IRS value versus
Monte Carlo simulation.

Research question
-----------------
By risk-neutral pricing theory, the closed-form affine formula and an inner
Monte Carlo simulation are two representations of the same expectation.
This script verifies numerically that the MC estimator converges to the
closed-form value as the number of simulation paths M increases.

Instrument
----------
10-year payer IRS (pay fixed, receive floating), semi-annual payments,
struck at par at t = 0.

Model
-----
Hull-White one-factor model:
    dr_t = (theta(t) - a * r_t) dt + sigma dW_t

with mean-reversion speed a = 0.10 and short-rate volatility sigma = 0.015.
The drift function theta(t) is calibrated to match the Svensson (1994) initial
term structure exactly (see Section 3 below).

Affine bond price (Hull-White, 1990)
-------------------------------------
    P(t, T | r_t) = A(t, T) * exp(-B(t, T) * r_t)

    B(t, T)   = [1 - exp(-a*(T-t))] / a

    ln A(t,T) = ln[P(0,T)/P(0,t)] - B(t,T)*f(0,t)
                - (sigma^2 / 4a) * B(t,T)^2 * (1 - exp(-2a*t))

where P(0, cdot) and f(0, cdot) are the market discount factor and
instantaneous forward rate, respectively, from the Svensson curve.

Payer IRS value (closed form)
------------------------------
At evaluation time t_eval coinciding with a semi-annual reset date:

    V_payer(t_eval, r_t) = [1 - P(t_eval, T_n)]
                           - K * tau * sum_{T_j > t_eval} P(t_eval, T_j)

The floating leg value 1 - P(t_eval, T_n) follows from the telescoping
identity for single-curve pricing.

Monte Carlo method
------------------
From (t_eval, r_t), simulate M inner paths via Euler-Maruyama:

    r_{s+dt} = r_s + (theta(s) - a * r_s) * dt + sigma * sqrt(dt) * Z,
                                                   Z ~ N(0,1)

At each payment date T_j on the path:
  - Floating cashflow: float_CF = 1 / P_aff(T_{j-1}, T_j | r_{T_{j-1}}) - 1
  - Fixed cashflow:    fixed_CF = K * tau
  - Net payer CF:      net_CF   = float_CF - fixed_CF

The floating cashflow uses the affine formula evaluated at the *simulated*
r_{T_{j-1}}, reproducing the LIBOR setting mechanism under the model.

Each cashflow is discounted to t_eval using the path numeraire:

    D(t_eval, T_j) = exp(-sum_{steps from t_eval to T_j} r_s * dt)

The MC estimate is the mean across paths:

    V_MC = (1/M) * sum_{omega=1}^{M} sum_j net_CF_j(omega) * D_j(omega)

Theoretical guarantees
-----------------------
By the tower property of conditional expectations:

    E^Q[D(t_eval, T_j) / P_aff(T_{j-1}, T_j | r_{T_{j-1}})]
        = P_aff(t_eval, T_{j-1} | r_{t_eval})

from which the floating leg telescopes to 1 - P_aff(t_eval, T_n | r_{t_eval}).
Therefore V_MC is an unbiased estimator of V_payer (closed form), and the
two values must agree in the limit M -> infinity.

References
----------
Hull, J. and White, A. (1990). Pricing interest-rate-derivative securities.
    Review of Financial Studies, 3(4), 573-592.
Svensson, L.E.O. (1994). Estimating and Interpreting Forward Interest Rates:
    Sweden 1992-1994. IMF Working Paper 94/114.
Gurkaynak, R.S., Sack, B. and Wright, J.H. (2007). The U.S. Treasury yield
    curve: 1961 to the present. Journal of Monetary Economics, 54(8), 2291-2304.
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")   # non-interactive backend; safe for script execution

# =============================================================================
# 0.  File paths
# =============================================================================
DATA_PATH = (
    "/home/lindani/Documents/Masters Research/Mproj/FinalResults/Code/feds200628.csv"
)
OUTPUT_PATH = (
    "/home/lindani/Documents/Masters Research/Mproj/FinalResults/results/"
    "hw_price_convergence.png"
)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# =============================================================================
# 1.  Hull-White parameters
# =============================================================================
HW_A     = 0.10    # mean-reversion speed
HW_SIGMA = 0.015   # short-rate volatility

# =============================================================================
# 2.  Svensson (1994) term structure
#
#  Continuously compounded zero-coupon yield (Gurkaynak, Sack, Wright, 2007):
#
#    y(0,t) = beta0
#             + beta1 * g1(t, tau1)
#             + beta2 * [g1(t, tau1) - exp(-t/tau1)]
#             + beta3 * [g1(t, tau2) - exp(-t/tau2)]
#
#  where g1(t, tau) = (1 - exp(-t/tau)) / (t/tau).
#
#  Instantaneous forward rate (derivative of t*y w.r.t. t):
#
#    f(0,t) = beta0
#             + beta1 * exp(-t/tau1)
#             + beta2 * (t/tau1) * exp(-t/tau1)
#             + beta3 * (t/tau2) * exp(-t/tau2)
#
#  Discount factor:  P(0,t) = exp(-y(0,t) * t)
# =============================================================================

def load_svensson_params(filepath: str) -> dict:
    """
    Parse the Federal Reserve GSW dataset and return Svensson parameters
    from the last complete observation row.

    The dataset stores yields in percent; this function converts to decimals.
    """
    df = pd.read_csv(filepath, skiprows=9)
    df = df.dropna(subset=["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"])
    row = df.iloc[-1]
    return {
        "beta0": float(row["BETA0"]) / 100.0,
        "beta1": float(row["BETA1"]) / 100.0,
        "beta2": float(row["BETA2"]) / 100.0,
        "beta3": float(row["BETA3"]) / 100.0,
        "tau1":  float(row["TAU1"]),
        "tau2":  float(row["TAU2"]),
        "date":  str(row["Date"]),
    }


def svensson_yield(t: np.ndarray, p: dict) -> np.ndarray:
    """Continuously compounded zero-coupon yield y(0,t) in decimal form."""
    t = np.asarray(t, dtype=float)
    # Guard against division by zero at t = 0; limit is beta0 + beta1.
    ts = np.where(t < 1e-10, 1e-10, t)
    e1 = np.exp(-ts / p["tau1"])
    e2 = np.exp(-ts / p["tau2"])
    g1 = (1.0 - e1) / (ts / p["tau1"])
    g2 = (1.0 - e2) / (ts / p["tau2"])
    return (
        p["beta0"]
        + p["beta1"] * g1
        + p["beta2"] * (g1 - e1)
        + p["beta3"] * (g2 - e2)
    )


def svensson_forward(t: np.ndarray, p: dict) -> np.ndarray:
    """Instantaneous forward rate f(0,t) in decimal form."""
    t = np.asarray(t, dtype=float)
    ts = np.where(t < 1e-10, 1e-10, t)
    e1 = np.exp(-ts / p["tau1"])
    e2 = np.exp(-ts / p["tau2"])
    return (
        p["beta0"]
        + p["beta1"] * e1
        + p["beta2"] * (ts / p["tau1"]) * e1
        + p["beta3"] * (ts / p["tau2"]) * e2
    )


def market_discount(t: np.ndarray, p: dict) -> np.ndarray:
    """Market discount factor P(0,t) = exp(-y(0,t) * t)."""
    t = np.asarray(t, dtype=float)
    return np.exp(-svensson_yield(t, p) * t)


# =============================================================================
# 3.  Hull-White drift function theta(t)
#
#  Calibrating to the initial term structure requires:
#
#    theta(t) = df(0,t)/dt + a * f(0,t)
#               + (sigma^2 / 2a) * (1 - exp(-2a*t))
#
#  The time derivative df(0,t)/dt is evaluated by central finite differences
#  applied to the Svensson formula.
# =============================================================================

def hw_theta(t: np.ndarray, p: dict) -> np.ndarray:
    """
    Hull-White drift function theta(t), calibrated to match f(0,t).
    Accepts arrays for vectorised evaluation over the simulation grid.
    """
    t = np.asarray(t, dtype=float)
    h = 1e-5
    dfdt = (svensson_forward(t + h, p) - svensson_forward(t - h, p)) / (2.0 * h)
    return (
        dfdt
        + HW_A * svensson_forward(t, p)
        + (HW_SIGMA**2 / (2.0 * HW_A)) * (1.0 - np.exp(-2.0 * HW_A * t))
    )


# =============================================================================
# 4.  Hull-White affine bond price
#
#    B(t,T)    = [1 - exp(-a*(T-t))] / a
#
#    ln A(t,T) = ln[P(0,T)/P(0,t)] - B(t,T)*f(0,t)
#                - (sigma^2 / 4a) * B(t,T)^2 * (1 - exp(-2a*t))
#
#    P(t,T|r)  = exp(ln A(t,T) - B(t,T)*r)
# =============================================================================

def _hw_B(t: float, T) -> np.ndarray:
    """B(t,T) coefficient; T may be a scalar or an array."""
    return (1.0 - np.exp(-HW_A * (np.asarray(T) - t))) / HW_A


def _hw_lnA(t: float, T, p: dict) -> np.ndarray:
    """
    ln A(t,T) for the affine bond price.
    t is a scalar; T may be scalar or array.
    All market quantities (P(0,cdot), f(0,cdot)) depend only on the
    Svensson parameters, not on the simulated short-rate path.
    """
    T   = np.asarray(T, dtype=float)
    B   = _hw_B(t, T)
    P0T = market_discount(T, p)
    P0t = float(market_discount(t, p))
    f0t = float(svensson_forward(t, p))
    return (
        np.log(P0T / P0t)
        + B * f0t
        - (HW_SIGMA**2 / (4.0 * HW_A)) * B**2 * (1.0 - np.exp(-2.0 * HW_A * t))
    )


def hw_bond_price(t: float, T, r_t, p: dict) -> np.ndarray:
    """
    Compute P(t, T | r_t) via the Hull-White affine formula.

    Parameters
    ----------
    t   : float         Evaluation time (scalar).
    T   : array-like    Maturity date(s).
    r_t : scalar or (M,) array
          Short rate at t. When r_t is a 1-D array of shape (M,) and T is
          a 1-D array of shape (n,), the output has shape (M, n).
    p   : dict          Svensson parameters.

    Returns
    -------
    ndarray  Bond price(s).
    """
    T    = np.asarray(T, dtype=float)
    B    = _hw_B(t, T)       # shape (n,) or scalar
    lnA  = _hw_lnA(t, T, p)  # shape (n,) or scalar
    r_t  = np.asarray(r_t)

    if r_t.ndim == 0:
        # Scalar r_t: straightforward.
        return np.exp(lnA - B * r_t)
    else:
        # Vector r_t of shape (M,): broadcast to produce shape (M, n).
        # outer(r_t, B) has shape (M, n); lnA has shape (n,).
        return np.exp(lnA[np.newaxis, :] - np.outer(r_t, B))


# =============================================================================
# 5.  IRS setup: 10-year payer, semi-annual, par at t = 0
#
#  Payment schedule: T_1, T_2, ..., T_n = 0.5, 1.0, ..., 10.0
#  Day-count fraction: tau = 0.5 (ACT/365 approximation for semi-annual)
#
#  Par fixed rate at t = 0:
#    K = (1 - P(0, T_n)) / (tau * sum_{j=1}^{n} P(0, T_j))
# =============================================================================
MATURITY   = 10.0
TAU        = 0.5
T_PAY      = np.arange(TAU, MATURITY + 1e-9, TAU)   # shape (20,)


def par_swap_rate(p: dict) -> float:
    """Par fixed rate K at t = 0 under single-curve pricing."""
    P = market_discount(T_PAY, p)
    return float((1.0 - P[-1]) / (TAU * P.sum()))


# =============================================================================
# 6.  Closed-form (affine) payer IRS value
#
#  When t_eval coincides with a reset date, the floating leg value collapses
#  to 1 - P(t_eval, T_n) by the telescoping identity, giving:
#
#    V_payer(t_eval, r_t)
#        = [1 - P(t_eval, T_n | r_t)]
#          - K * tau * sum_{T_j > t_eval} P(t_eval, T_j | r_t)
# =============================================================================

def irs_affine(t_eval: float, r_t: float, K: float, p: dict) -> float:
    """Closed-form payer IRS value via Hull-White affine bond prices."""
    rem   = T_PAY[T_PAY > t_eval + 1e-9]   # remaining payment dates
    bonds = hw_bond_price(t_eval, rem, r_t, p)   # shape (n_rem,)
    return float((1.0 - bonds[-1]) - K * TAU * bonds.sum())


# =============================================================================
# 7.  Monte Carlo payer IRS value
#
#  Inner simulation from t_eval to T_n using Euler-Maruyama.
#  The time grid uses SUB_STEPS substeps per semi-annual period so that
#  payment dates fall exactly on grid nodes (no interpolation needed).
#
#  At each payment date T_j on each path:
#    P(T_{j-1}, T_j | r_{T_{j-1}}) is computed via the affine formula
#    using the simulated short rate at T_{j-1}.  This is the model bond
#    price used by the LIBOR setting mechanism.
#
#    net_CF_j = [1 / P(T_{j-1}, T_j | r_{T_{j-1}}) - 1] - K * tau
#
#    disc_j   = exp(-integral_{t_eval}^{T_j} r_s ds)
#             ~ exp(sum_{steps to T_j} r_s * (-dt))   [left-endpoint rule]
#
#    V_MC = (1/M) * sum_{omega} sum_j net_CF_j(omega) * disc_j(omega)
# =============================================================================
SUB_STEPS = 12   # substeps per TAU period; dt = TAU / SUB_STEPS = 1/24 year


def irs_mc(
    t_eval: float,
    r_t:    float,
    K:      float,
    p:      dict,
    M:      int,
    rng:    np.random.Generator,
) -> float:
    """
    Monte Carlo estimate of the payer IRS value at (t_eval, r_t).

    Parameters
    ----------
    t_eval : float              Evaluation time in years; must be a reset date.
    r_t    : float              Short rate at t_eval.
    K      : float              Fixed swap rate.
    p      : dict               Svensson parameters.
    M      : int                Number of inner simulation paths.
    rng    : np.random.Generator  Pre-seeded generator for reproducibility.

    Returns
    -------
    float  Monte Carlo estimate of V_payer(t_eval, r_t).
    """
    dt   = TAU / SUB_STEPS          # step size (years)
    rem  = T_PAY[T_PAY > t_eval + 1e-9]
    n_pay = len(rem)
    if n_pay == 0:
        return 0.0

    # Build uniform time grid from t_eval to MATURITY.
    # The small epsilon in the upper bound ensures MATURITY is included.
    grid    = np.arange(t_eval, MATURITY + dt * 0.1, dt)
    n_steps = len(grid) - 1

    # Map each payment date to its grid index.
    # Because dt divides TAU exactly, np.round gives the correct integer.
    pay_to_grid = {
        j: int(np.round((rem[j] - t_eval) / dt))
        for j in range(n_pay)
    }
    grid_to_pay = {v: k for k, v in pay_to_grid.items()}   # reverse lookup

    # Precompute theta on the left endpoints of each Euler interval.
    theta_vec = hw_theta(grid[:-1], p)   # shape (n_steps,)

    # ---- Path state --------------------------------------------------------
    r    = np.full(M, r_t, dtype=float)   # short rate at current step
    logD = np.zeros(M, dtype=float)        # log D(t_eval, current time)

    # Storage: short rate and log-discount at each payment date.
    r_at_pay    = np.zeros((M, n_pay), dtype=float)
    logD_at_pay = np.zeros((M, n_pay), dtype=float)

    sqrt_dt = np.sqrt(dt)

    # ---- Euler-Maruyama loop -----------------------------------------------
    for i in range(n_steps):
        # Accumulate discount using r at the *left* endpoint (step i).
        logD -= r * dt
        # Euler-Maruyama update.
        Z  = rng.standard_normal(M)
        r  = r + (theta_vec[i] - HW_A * r) * dt + HW_SIGMA * sqrt_dt * Z
        # Record state whenever we land on a payment date.
        if (i + 1) in grid_to_pay:
            j               = grid_to_pay[i + 1]
            r_at_pay[:, j]    = r
            logD_at_pay[:, j] = logD

    # ---- Cashflow aggregation ----------------------------------------------
    V_paths = np.zeros(M, dtype=float)

    for j in range(n_pay):
        T_j   = rem[j]
        T_jm1 = t_eval if j == 0 else rem[j - 1]

        # Short rate at T_{j-1}: either the initial r_t (j=0) or the
        # simulated value stored at the previous payment date.
        r_jm1 = np.full(M, r_t) if j == 0 else r_at_pay[:, j - 1]

        # Affine bond price P(T_{j-1}, T_j | r_{T_{j-1}}).
        # lnA and B are deterministic; only r_jm1 is path-dependent.
        B_val   = float(_hw_B(T_jm1, T_j))
        lnA_val = float(_hw_lnA(T_jm1, T_j, p))
        P_bond  = np.exp(lnA_val - B_val * r_jm1)   # shape (M,)

        # Net payer cashflow at T_j.
        net_cf = (1.0 / P_bond - 1.0) - K * TAU

        # Discount factor from t_eval to T_j (path-dependent).
        disc = np.exp(logD_at_pay[:, j])

        V_paths += net_cf * disc

    return float(np.mean(V_paths))


# =============================================================================
# 8.  Main: evaluate convergence at three representative points
# =============================================================================

def main() -> None:
    # --- Load initial term structure ----------------------------------------
    p   = load_svensson_params(DATA_PATH)
    r0  = float(svensson_forward(1e-8, p))   # f(0, 0+) = beta0 + beta1
    K   = par_swap_rate(p)

    print("=" * 65)
    print(f"Svensson parameters (date: {p['date']})")
    print(f"  beta0 = {p['beta0']*100:+.4f}%   beta1 = {p['beta1']*100:+.4f}%")
    print(f"  beta2 = {p['beta2']*100:+.4f}%   beta3 = {p['beta3']*100:+.4f}%")
    print(f"  tau1  = {p['tau1']:.4f}           tau2  = {p['tau2']:.4f}")
    print(f"Initial short rate  r0 = f(0,0+) = {r0*100:.4f}%")
    print(f"Par swap rate       K  = {K*100:.6f}%")
    print("=" * 65)

    # --- Simulation settings ------------------------------------------------
    M_list = [1_000, 5_000, 10_000, 50_000]
    rng    = np.random.default_rng(seed=42)

    # --- Evaluation points --------------------------------------------------
    # Each point is (t_eval, r_t) where t_eval is a semi-annual reset date.
    eval_points = [
        {
            "tag":     "t=1Y, r_t=r0",
            "t_latex": r"$t=1Y,\;\;r_t=r_0$",
            "t":       1.0,
            "r":       r0,
        },
        {
            "tag":     "t=5Y, r_t=r0+1%",
            "t_latex": r"$t=5Y,\;\;r_t=r_0+1\%$",
            "t":       5.0,
            "r":       r0 + 0.01,
        },
        {
            "tag":     "t=8Y, r_t=r0-0.5%",
            "t_latex": r"$t=8Y,\;\;r_t=r_0-0.5\%$",
            "t":       8.0,
            "r":       r0 - 0.005,
        },
    ]

    results = []

    for ep in eval_points:
        t_eval   = ep["t"]
        r_t      = ep["r"]
        V_affine = irs_affine(t_eval, r_t, K, p)
        V_mc_seq = []
        print(f"\n{ep['tag']}  |  r_t = {r_t*100:.4f}%  |  V_affine = {V_affine:+.6f}")
        for M in M_list:
            V_mc = irs_mc(t_eval, r_t, K, p, M, rng)
            V_mc_seq.append(V_mc)
            print(f"  M={M:>6,}  V_MC={V_mc:+.6f}  err={V_mc - V_affine:+.2e}")
        results.append({**ep, "V_affine": V_affine, "V_mc": V_mc_seq})

    # --- Summary table ------------------------------------------------------
    header = (
        f"\n{'Evaluation point':<26}"
        f"{'V_affine':>12}"
        + "".join(f"{'V_MC('+str(M//1000)+'k)':>12}" for M in M_list)
        + f"{'Err@50k':>12}"
    )
    sep = "=" * (26 + 12 + 12 * len(M_list) + 12)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for res in results:
        mc_str = "".join(f"{v:>12.6f}" for v in res["V_mc"])
        err    = res["V_mc"][-1] - res["V_affine"]
        print(f"{res['tag']:<26}{res['V_affine']:>12.6f}{mc_str}{err:>12.2e}")
    print(sep)

    # --- Plot: convergence of V_MC to V_affine ------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Hull-White IRS pricing: affine closed form vs Monte Carlo convergence\n"
        r"10-year payer IRS, semi-annual, $a=0.10$, $\sigma=0.015$, "
        "Svensson initial curve",
        fontsize=11,
    )

    M_arr = np.array(M_list, dtype=float)

    for ax, res in zip(axes, results):
        V_aff = res["V_affine"]
        V_mc  = np.array(res["V_mc"])

        # Convert to percent of notional for legibility.
        ax.semilogx(
            M_arr, V_mc * 100,
            marker="o", color="#1f77b4", linewidth=1.5, markersize=6,
            label=r"$V_\mathrm{MC}(M)$", zorder=3,
        )
        ax.axhline(
            V_aff * 100, color="#d62728", linestyle="--", linewidth=1.5,
            label=r"$V_\mathrm{affine}$", zorder=2,
        )
        # Shaded 1 bp tolerance band around the affine reference.
        ax.axhspan(
            (V_aff - 1e-4) * 100, (V_aff + 1e-4) * 100,
            alpha=0.15, color="#d62728", label=r"$\pm 1$ bp band",
        )

        ax.set_title(res["t_latex"], fontsize=10)
        ax.set_xlabel("Number of paths $M$", fontsize=9)
        ax.set_ylabel("IRS value (% of notional)", fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, which="both", alpha=0.3, linestyle=":")
        ax.set_xticks(M_arr)
        ax.set_xticklabels([f"{int(m):,}" for m in M_arr], fontsize=7)

        # Annotate the signed error at 50 000 paths.
        err_50k = res["V_mc"][-1] - V_aff
        ax.annotate(
            f"err = {err_50k:+.2e}",
            xy=(M_arr[-1], res["V_mc"][-1] * 100),
            xytext=(-75, 18),
            textcoords="offset points",
            fontsize=8,
            color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c", lw=0.8),
        )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to:\n  {OUTPUT_PATH}")


# =============================================================================
if __name__ == "__main__":
    main()
