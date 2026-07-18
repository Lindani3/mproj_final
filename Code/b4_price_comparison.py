"""
b4_price_comparison.py
======================
Systematic comparison of two Hull-White IRS pricing representations across
a 9 x 5 evaluation grid of (monitoring date t, short rate r_t) points.

Pricing equations
-----------------
Equation 8.8  — closed-form affine payer IRS value:

    V_affine(t, r_t) = [1 - P(t, T_n | r_t)]
                       - K^par * tau * sum_{T_j > t} P(t, T_j | r_t)

where the Hull-White affine bond price is:

    P(t, T | r_t)  = A(t, T) * exp(-B(t, T) * r_t)
    B(t, T)         = [1 - exp(-a*(T-t))] / a
    ln A(t, T)      = ln[P(0,T)/P(0,t)] - B(t,T) * f(0,t)
                      - (sigma^2 / 4a) * B(t,T)^2 * (1 - exp(-2a*t))

Equation 8.13 — simulation-based payer IRS value:

    V_MC(t, r_t) = (1/M) * sum_{i=1}^{M} sum_{T_j > t}
                   CF_j^(i) * D^(i)(t, T_j)

where:
    CF_j^(i) = [1 / P(T_{j-1}, T_j | r^(i)_{T_{j-1}}) - 1] - K^par * tau
    D^(i)(t, T_j) = exp( -sum_{s in [t, T_j]} r^(i)_s * dt )

and M = 10,000 inner Euler-Maruyama paths are simulated from (t, r_t).

Theoretical equivalence
------------------------
By the tower property of conditional expectations, V_MC is an unbiased
estimator of V_affine. Finite-sample discrepancies reflect MC sampling error,
O(1/sqrt(M)), and Euler-Maruyama discretisation error, O(dt). Both vanish in
the limits M -> inf and dt -> 0.

Evaluation grid
---------------
  Monitoring dates  : t in {1, 2, 3, 4, 5, 6, 7, 8, 9}  years
  Short-rate levels : r_t in {r0-2%, r0-1%, r0, r0+1%, r0+2%}
  Total             : 45 evaluation points

Model
-----
Hull-White one-factor:  dr_t = (theta(t) - a * r_t) dt + sigma dW_t
Parameters:  a = 0.10,  sigma = 0.015
Initial term structure: Svensson (1994), last complete row of FEDS dataset.

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
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")

# =============================================================================
# 0.  File paths
# =============================================================================
DATA_PATH = (
    "/home/lindani/Documents/Masters Research/Mproj/FinalResults/Code/feds200628.csv"
)
OUTPUT_PATH = (
    "/home/lindani/Documents/Masters Research/Mproj/FinalResults/results/"
    "b4_price_comparison.png"
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

    Yields in the dataset are stored in percent; this function converts
    them to decimal form.
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
    t  = np.asarray(t, dtype=float)
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
    t  = np.asarray(t, dtype=float)
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
#  The time derivative df(0,t)/dt is evaluated via central finite differences
#  applied to the Svensson formula.
# =============================================================================

def hw_theta(t: np.ndarray, p: dict) -> np.ndarray:
    """
    Hull-White drift function theta(t), calibrated to match f(0,t).
    Accepts arrays for vectorised evaluation over the simulation grid.
    """
    t    = np.asarray(t, dtype=float)
    h    = 1e-5
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
    """B(t,T) sensitivity coefficient; T may be scalar or array."""
    return (1.0 - np.exp(-HW_A * (np.asarray(T) - t))) / HW_A


def _hw_lnA(t: float, T, p: dict) -> np.ndarray:
    """
    ln A(t,T) for the affine bond price.
    t is a scalar; T may be scalar or array.
    All market quantities (P(0,.),f(0,.)) depend only on the Svensson
    parameters, not on the simulated short-rate path.
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
    t   : float           Evaluation time (scalar).
    T   : array-like      Maturity date(s).
    r_t : scalar or (M,) array
          Short rate at t. When r_t is 1-D with shape (M,) and T is 1-D
          with shape (n,), the result has shape (M, n).
    p   : dict            Svensson parameters.
    """
    T   = np.asarray(T, dtype=float)
    B   = _hw_B(t, T)
    lnA = _hw_lnA(t, T, p)
    r_t = np.asarray(r_t)

    if r_t.ndim == 0:
        return np.exp(lnA - B * r_t)
    else:
        # r_t shape (M,), T shape (n,) -> result shape (M, n)
        return np.exp(lnA[np.newaxis, :] - np.outer(r_t, B))


# =============================================================================
# 5.  IRS setup: 10-year payer, semi-annual, par at t = 0
#
#  Payment schedule: T_1, ..., T_n = 0.5, 1.0, ..., 10.0
#  Day-count fraction: tau = 0.5 (ACT/365 approximation for semi-annual)
#
#  Par fixed rate:  K = (1 - P(0, T_n)) / (tau * sum_{j} P(0, T_j))
# =============================================================================
MATURITY  = 10.0
TAU       = 0.5
T_PAY     = np.arange(TAU, MATURITY + 1e-9, TAU)   # shape (20,)
SUB_STEPS = 12   # substeps per TAU period; dt = TAU / SUB_STEPS = 1/24 yr
                 # payment dates fall exactly on simulation grid nodes


def par_swap_rate(p: dict) -> float:
    """Par fixed rate K at t = 0 under single-curve pricing."""
    P = market_discount(T_PAY, p)
    return float((1.0 - P[-1]) / (TAU * P.sum()))


# =============================================================================
# 6.  Closed-form (affine) payer IRS value  — Equation 8.8
#
#  When t_eval coincides with a reset date, the floating leg collapses
#  to 1 - P(t_eval, T_n) via the telescoping identity:
#
#    V_payer(t_eval, r_t)
#        = [1 - P(t_eval, T_n | r_t)]
#          - K * tau * sum_{T_j > t_eval} P(t_eval, T_j | r_t)
# =============================================================================

def irs_affine(t_eval: float, r_t: float, K: float, p: dict) -> float:
    """
    Closed-form payer IRS value at (t_eval, r_t) via affine bond prices.
    Implements Equation 8.8.
    """
    rem   = T_PAY[T_PAY > t_eval + 1e-9]
    bonds = hw_bond_price(t_eval, rem, r_t, p)   # shape (n_rem,)
    return float((1.0 - bonds[-1]) - K * TAU * bonds.sum())


# =============================================================================
# 7.  Monte Carlo payer IRS value  — Equation 8.13
#
#  Simulates M inner Euler-Maruyama paths from (t_eval, r_t) to MATURITY.
#  Because SUB_STEPS = 12 divides TAU exactly, every payment date T_j
#  falls on a simulation grid node — no interpolation is needed.
#
#  At payment date T_j:
#    P(T_{j-1}, T_j | r_{T_{j-1}}) is the affine bond price at the
#    simulated short rate at T_{j-1}.  This is the model LIBOR fixing.
#
#    net_CF_j = [1 / P(T_{j-1}, T_j | r_{T_{j-1}}) - 1] - K * tau
#
#    disc_j   = exp(-integral_{t_eval}^{T_j} r_s ds)
#             ~ exp(sum_{steps} r_s * (-dt))   [left-endpoint rule]
#
#  All M random increments are pre-generated in one call to avoid
#  repeated overhead inside the time-stepping loop.
# =============================================================================

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
    Implements Equation 8.13.

    Parameters
    ----------
    t_eval : float               Evaluation time (years); must be a reset date.
    r_t    : float               Short rate at t_eval.
    K      : float               Fixed swap rate.
    p      : dict                Svensson parameters.
    M      : int                 Number of inner simulation paths.
    rng    : np.random.Generator Pre-seeded generator for reproducibility.
    """
    dt    = TAU / SUB_STEPS
    rem   = T_PAY[T_PAY > t_eval + 1e-9]
    n_pay = len(rem)
    if n_pay == 0:
        return 0.0

    # Uniform time grid from t_eval to MATURITY.
    # The small safety margin ensures MATURITY is always included.
    grid    = np.arange(t_eval, MATURITY + dt * 0.1, dt)
    n_steps = len(grid) - 1

    # Map each payment date to its corresponding grid index.
    # Because dt divides TAU exactly, rounding gives the correct integer.
    pay_to_grid = {
        j: int(np.round((rem[j] - t_eval) / dt))
        for j in range(n_pay)
    }
    grid_to_pay = {v: k for k, v in pay_to_grid.items()}

    # Precompute theta at the left endpoint of each Euler interval (vectorised).
    theta_vec = hw_theta(grid[:-1], p)   # shape (n_steps,)

    # ---- Path state ----------------------------------------------------------
    r    = np.full(M, r_t, dtype=float)   # short rate at current step
    logD = np.zeros(M, dtype=float)        # accumulated log-discount factor

    r_at_pay    = np.zeros((M, n_pay), dtype=float)
    logD_at_pay = np.zeros((M, n_pay), dtype=float)

    # Pre-generate all standard-normal draws: shape (n_steps, M).
    # This avoids repeated generator calls inside the loop.
    Z_all   = rng.standard_normal((n_steps, M))
    sqrt_dt = np.sqrt(dt)

    # ---- Euler-Maruyama loop -------------------------------------------------
    for i in range(n_steps):
        # Accumulate discount using r at the left endpoint (step i).
        logD -= r * dt
        # Euler-Maruyama update.
        r = r + (theta_vec[i] - HW_A * r) * dt + HW_SIGMA * sqrt_dt * Z_all[i]
        # Record state when we land on a payment date.
        if (i + 1) in grid_to_pay:
            j                  = grid_to_pay[i + 1]
            r_at_pay[:, j]    = r
            logD_at_pay[:, j] = logD

    # ---- Cashflow aggregation ------------------------------------------------
    V_paths = np.zeros(M, dtype=float)

    for j in range(n_pay):
        T_jm1 = t_eval if j == 0 else rem[j - 1]
        T_j   = rem[j]

        # Short rate at T_{j-1}: initial value for j=0, simulated otherwise.
        r_jm1 = np.full(M, r_t) if j == 0 else r_at_pay[:, j - 1]

        # Affine bond price P(T_{j-1}, T_j | r_{T_{j-1}}) for each path.
        B_val   = float(_hw_B(T_jm1, T_j))
        lnA_val = float(_hw_lnA(T_jm1, T_j, p))
        P_bond  = np.exp(lnA_val - B_val * r_jm1)   # shape (M,)

        # Net payer cashflow: floating minus fixed, per unit notional.
        net_cf = (1.0 / P_bond - 1.0) - K * TAU

        # Path discount factor from t_eval to T_j.
        disc = np.exp(logD_at_pay[:, j])

        V_paths += net_cf * disc

    return float(np.mean(V_paths))


# =============================================================================
# 8.  Heatmap plot
# =============================================================================

def _draw_panel(
    ax,
    data:        np.ndarray,
    title:       str,
    cmap:        str,
    vmin:        float,
    vmax:        float,
    fmt:         str,
    cb_label:    str,
    x_labels:    list,
    y_labels:    list,
    center:      float = None,
) -> None:
    """
    Draw a single annotated heatmap panel.

    Parameters
    ----------
    ax        : matplotlib Axes object.
    data      : 2-D array of shape (n_r, n_t); rows = r_t levels, cols = t.
    title     : panel title string.
    cmap      : colormap name.
    vmin/vmax : color scale limits.
    fmt       : Python format string for cell annotations.
    cb_label  : colorbar axis label.
    x_labels  : column (t) tick labels.
    y_labels  : row (r_t) tick labels.
    center    : if not None, use TwoSlopeNorm (diverging colormap).
    """
    if center is not None:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        im   = ax.imshow(data, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    else:
        im   = ax.imshow(
            data, aspect="auto", origin="lower",
            cmap=cmap, vmin=vmin, vmax=vmax,
        )

    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Monitoring date $t$ (years)", fontsize=9)
    ax.set_ylabel("Short rate $r_t$", fontsize=9)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(cb_label, fontsize=8)

    # Cell annotations: choose text colour by normalised colour intensity.
    n_r, n_t = data.shape
    denom     = max(abs(vmax - vmin), 1e-12)
    for ir in range(n_r):
        for it in range(n_t):
            val       = data[ir, it]
            intensity = abs(val - (center if center is not None else vmin)) / denom
            txt_color = "white" if intensity > 0.60 else "black"
            ax.text(
                it, ir, f"{val:{fmt}}",
                ha="center", va="center",
                fontsize=6.5, color=txt_color,
            )


def plot_heatmaps(
    V_aff:    np.ndarray,
    V_mc:     np.ndarray,
    ABS_ERR:  np.ndarray,
    T_MON:    np.ndarray,
    R_SHIFTS: np.ndarray,
    r0:       float,
    K:        float,
    p:        dict,
) -> None:
    """
    Produce three-panel heatmap figure and save to OUTPUT_PATH.

    Panel 1  V_affine(t, r_t)          — closed-form, Eq 8.8
    Panel 2  V_MC(t, r_t)              — simulation,  Eq 8.13
    Panel 3  |V_MC - V_affine|         — absolute error
    """
    # Convert all values to percent of notional for display legibility.
    V_aff_pct = V_aff   * 100.0
    V_mc_pct  = V_mc    * 100.0
    err_pct   = ABS_ERR * 100.0

    # Symmetric colour limits for the two value panels.
    v_abs_max = max(abs(V_aff_pct).max(), abs(V_mc_pct).max())
    v_abs_max = float(np.ceil(v_abs_max * 10.0) / 10.0)   # round to 0.1

    # Upper colour limit for the error panel, rounded up.
    err_max = float(np.ceil(err_pct.max() * 1000.0) / 1000.0)   # round to 0.001

    # Axis labels.
    x_labels = [f"{int(t)}" for t in T_MON]
    y_labels  = [
        r"$r_0 - 2\%$",
        r"$r_0 - 1\%$",
        r"$r_0$",
        r"$r_0 + 1\%$",
        r"$r_0 + 2\%$",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    fig.suptitle(
        "Hull-White payer IRS: closed-form (Eq 8.8) vs Monte Carlo (Eq 8.13)\n"
        rf"10-year payer, semi-annual, $a={HW_A}$, $\sigma={HW_SIGMA}$, "
        rf"$K^{{\mathrm{{par}}}}={K*100:.4f}\%$,  Svensson curve ({p['date']})",
        fontsize=11,
    )

    # Panel 1: V_affine
    _draw_panel(
        ax       = axes[0],
        data     = V_aff_pct,
        title    = r"$V_{\mathrm{affine}}(t,\,r_t)$ — Eq 8.8",
        cmap     = "RdBu_r",
        vmin     = -v_abs_max,
        vmax     =  v_abs_max,
        fmt      = ".2f",
        cb_label = "Value (% of notional)",
        x_labels = x_labels,
        y_labels = y_labels,
        center   = 0.0,
    )

    # Panel 2: V_MC
    _draw_panel(
        ax       = axes[1],
        data     = V_mc_pct,
        title    = r"$V_{\mathrm{MC}}(t,\,r_t)$ — Eq 8.13  ($M=10\,000$)",
        cmap     = "RdBu_r",
        vmin     = -v_abs_max,
        vmax     =  v_abs_max,
        fmt      = ".2f",
        cb_label = "Value (% of notional)",
        x_labels = x_labels,
        y_labels = y_labels,
        center   = 0.0,
    )

    # Panel 3: absolute error
    _draw_panel(
        ax       = axes[2],
        data     = err_pct,
        title    = r"$|V_{\mathrm{MC}} - V_{\mathrm{affine}}|$ — absolute error",
        cmap     = "YlOrRd",
        vmin     = 0.0,
        vmax     = err_max,
        fmt      = ".3f",
        cb_label = "Abs. error (% of notional)",
        x_labels = x_labels,
        y_labels = y_labels,
        center   = None,
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"\nHeatmap saved to:\n  {OUTPUT_PATH}")


# =============================================================================
# 9.  Main: evaluate the 9 x 5 grid and report
# =============================================================================

def main() -> None:

    # ---- Load initial term structure ----------------------------------------
    p  = load_svensson_params(DATA_PATH)
    r0 = float(svensson_forward(1e-8, p))   # f(0, 0+) = beta0 + beta1
    K  = par_swap_rate(p)

    print("=" * 72)
    print(f"Svensson parameters  (date: {p['date']})")
    print(f"  beta0 = {p['beta0']*100:+.4f}%   beta1 = {p['beta1']*100:+.4f}%")
    print(f"  beta2 = {p['beta2']*100:+.4f}%   beta3 = {p['beta3']*100:+.4f}%")
    print(f"  tau1  = {p['tau1']:.4f}            tau2  = {p['tau2']:.4f}")
    print(f"Initial short rate  r0 = f(0, 0+) = {r0*100:.4f}%")
    print(f"Par swap rate       K  = {K*100:.6f}%")
    print("=" * 72)

    # ---- Evaluation grid ----------------------------------------------------
    T_MON    = np.arange(1.0, 10.0, 1.0)               # [1, 2, ..., 9]
    R_SHIFTS = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    R_GRID   = r0 + R_SHIFTS                            # absolute short rates

    n_t = len(T_MON)    # 9
    n_r = len(R_GRID)   # 5
    M   = 10_000

    print(f"\nGrid: {n_t} monitoring dates x {n_r} short-rate levels = {n_t*n_r} points")
    print(f"MC paths per point: M = {M:,}")
    print(f"Short-rate levels: {', '.join(f'{r*100:.2f}%' for r in R_GRID)}")
    print()

    # Results arrays: shape (n_r, n_t) with r_t on rows, t on columns.
    # Row 0 = r0-2% (bottom of heatmap), row 4 = r0+2% (top).
    V_aff = np.zeros((n_r, n_t))
    V_mc  = np.zeros((n_r, n_t))

    print(
        f"{'Point':>5}  {'t':>5}  {'r_t':>8}  "
        f"{'V_affine':>12}  {'V_MC':>12}  {'|error|':>10}"
    )
    print("-" * 60)

    point = 0
    for i_t, t_eval in enumerate(T_MON):
        for i_r, r_t in enumerate(R_GRID):
            point += 1

            # Deterministic, unique seed for reproducibility per grid point.
            # Seed ranges from 1 to 45; no seed collision possible.
            seed = int(i_t * n_r + i_r + 1)
            rng  = np.random.default_rng(seed=seed)

            v_aff = irs_affine(t_eval, r_t, K, p)
            v_mc  = irs_mc(t_eval, r_t, K, p, M, rng)

            V_aff[i_r, i_t] = v_aff
            V_mc[i_r, i_t]  = v_mc

            print(
                f"{point:>5}  {t_eval:>5.0f}  {r_t*100:>+7.2f}%  "
                f"{v_aff:>+12.6f}  {v_mc:>+12.6f}  {abs(v_mc - v_aff):>10.2e}"
            )

    ABS_ERR = np.abs(V_mc - V_aff)

    # ---- Summary table: per monitoring date ---------------------------------
    print()
    print("=" * 72)
    print("Summary: absolute error |V_MC - V_affine| per monitoring date")
    print(
        f"{'t (yrs)':>8}  {'Mean abs error':>20}  {'Max abs error':>20}"
        f"  {'Mean (bps)':>12}  {'Max (bps)':>12}"
    )
    print("-" * 72)
    for i_t, t_eval in enumerate(T_MON):
        col  = ABS_ERR[:, i_t]
        mean = col.mean()
        mx   = col.max()
        print(
            f"{t_eval:>8.0f}  {mean:>20.6f}  {mx:>20.6f}"
            f"  {mean*10000:>12.2f}  {mx*10000:>12.2f}"
        )
    print("=" * 72)

    # ---- Overall statistics across all 45 points ----------------------------
    signed = V_mc - V_aff
    rmse   = float(np.sqrt(np.mean(signed**2)))
    mae    = float(ABS_ERR.mean())
    mxae   = float(ABS_ERR.max())

    print(f"\nOverall statistics across all {n_t * n_r} evaluation points:")
    print(f"  RMSE                : {rmse:.6f}  ({rmse*10000:.2f} bps of notional)")
    print(f"  Mean absolute error : {mae:.6f}  ({mae*10000:.2f} bps of notional)")
    print(f"  Max  absolute error : {mxae:.6f}  ({mxae*10000:.2f} bps of notional)")

    # ---- Heatmap plots ------------------------------------------------------
    plot_heatmaps(V_aff, V_mc, ABS_ERR, T_MON, R_SHIFTS, r0, K, p)


# =============================================================================
if __name__ == "__main__":
    main()
