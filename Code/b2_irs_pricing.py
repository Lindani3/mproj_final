"""
Block 2 — IRS Pricing Under Hull-White: Affine vs Monte Carlo.

Implements two valuation methods for a 10-year payer IRS under Hull-White:

  Method 1 — Affine Term Structure (closed-form):
    P(t,T) = A(t,T) * exp(-B(t,T) * r_t)
    V_IRS(t, r_t) = P(t,T0) - P(t,Tn) - K^par * tau * sum_j P(t,Tj)

  Method 2 — Monte Carlo simulation:
    r_{t+dt} = r_t + (theta(t) - a*r_t)*dt + sigma*sqrt(dt)*Z
    V_IRS^(i)(t_k) evaluated analytically at each simulated r_{t_k}^(i)

Both methods use the same Svensson initial term structure.
The plot shows the distribution of MC swap values at selected monitoring
dates, overlaid with the analytical value at the mean simulated short rate.

Output: ../../results/b2_irs_pricing.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql

HERE    = os.path.dirname(os.path.abspath(__file__))
FEDS    = os.path.join(HERE, "feds200628.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

SVEN_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]

# Hull-White parameters
HW_A     = 0.10
HW_SIGMA = 0.015

# IRS setup
T_N    = 10.0
TAU    = 0.5
N_PAY  = 20
PAY_DATES = np.arange(TAU, T_N + TAU / 2, TAU)   # T_1, ..., T_20

# Monte Carlo setup
N_PATHS = 10000
N_MON   = 120                                      # monthly monitoring dates
N_STEPS = N_MON                                    # one step per monitoring month
SEED    = 42
DT      = T_N / N_STEPS                           # = 1/12


# =============================================================================
# Svensson helpers
# =============================================================================
def svensson(T, b0, b1, b2, b3, t1, t2):
    T   = np.atleast_1d(np.asarray(T, dtype=float))
    s   = np.where(T < 1e-10, 1e-10, T)
    e1  = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


def disc(T, params):
    """Discount factor P(0,T) from Svensson zero rates."""
    y = svensson(T, *params) / 100.0
    return np.exp(-y * T)


def fwd_rate(t, params, h=1e-4):
    """Instantaneous forward rate f(0,t) via central differences."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    lp = np.log(disc(np.maximum(t - h, 1e-8), params))
    ln = np.log(disc(t + h, params))
    return -(ln - lp) / (2 * h)


# =============================================================================
# Method 1: Affine Term Structure
# =============================================================================
def hw_B(t, T, a):
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_lnA(t, T, a, sigma, params):
    """ln A(t,T) from Hull-White affine formula."""
    B_   = hw_B(t, T, a)
    P0T  = disc(T, params)
    P0t  = disc(np.maximum(t, 1e-10), params)
    f0t  = fwd_rate(np.maximum(t, 1e-10), params)
    return (np.log(P0T / P0t)
            - B_ * f0t
            - (sigma**2 / (4.0 * a)) * B_**2 * (1.0 - np.exp(-2.0 * a * t)))


def hw_bond(t, T, r_t, a, sigma, params):
    """P(t,T) = A(t,T) * exp(-B(t,T) * r_t)."""
    B_   = hw_B(t, T, a)
    lnA_ = hw_lnA(t, T, a, sigma, params)
    return np.exp(lnA_ - B_ * r_t)


def irs_value(t, r_t, pay_dates, k_par, a, sigma, params):
    """Payer IRS value at (t, r_t) via affine formula."""
    remaining = pay_dates[pay_dates > t + 1e-10]
    if len(remaining) == 0:
        return 0.0
    p_vals  = hw_bond(t, remaining, r_t, a, sigma, params)
    pv_flt  = 1.0 - p_vals[-1]       # P(t,t) - P(t,T_n) = 1 - P(t,T_n)
    pv_fix  = k_par * TAU * np.sum(p_vals)
    return pv_flt - pv_fix


# =============================================================================
# Load FEDS data
# =============================================================================
feds     = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds     = feds.set_index("Date")
valid    = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]
params   = row[SVEN_COLS].values

# r0 = lim_{T->0} y(0,T) = beta0 + beta1  (Diebold and Li, 2006)
r0 = (row["BETA0"] + row["BETA1"]) / 100.0

# Par rate at t=0
dfs_pay = disc(PAY_DATES, params)
k_par   = (1.0 - dfs_pay[-1]) / (TAU * np.sum(dfs_pay))

print(f"Calibration date : {cal_date.strftime('%d %B %Y')}")
print(f"r0               : {r0*100:.4f}%")
print(f"K^par            : {k_par*100:.4f}%")

# V_IRS at t=0 should be zero (par condition)
v0 = irs_value(0.0, r0, PAY_DATES, k_par, HW_A, HW_SIGMA, params)
print(f"V_IRS(0, r0)     : {v0:.2e}  (should be ~0)")

# =============================================================================
# Method 2: Monte Carlo via QuantLib
# =============================================================================
ref_date = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

def _tenor_date(rd, t):
    if t < 1.0:
        return rd + ql.Period(int(round(t * 12)), ql.Months)
    return rd + ql.Period(int(t), ql.Years)

tenors     = np.array([0.25, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=float)
zero_rates = svensson(tenors, *params) / 100.0
ql_dates   = [_tenor_date(ref_date, t) for t in tenors]
zero_curve = ql.ZeroCurve(ql_dates, zero_rates.tolist(),
                           ql.Actual365Fixed(), ql.NullCalendar(),
                           ql.Linear(), ql.Continuous)
yts_handle = ql.YieldTermStructureHandle(zero_curve)
hw_process = ql.HullWhiteProcess(yts_handle, HW_A, HW_SIGMA)

rng      = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(N_STEPS, ql.UniformRandomGenerator(SEED))
)
path_gen = ql.GaussianPathGenerator(hw_process, T_N, N_STEPS, rng, False)

monitor  = np.linspace(0.0, T_N, N_MON + 1)   # t_k = k/12, k=0..120
all_V    = np.zeros((N_PATHS, len(monitor)))
all_r    = np.zeros((N_PATHS, len(monitor)))

print(f"\nRunning {N_PATHS:,} Monte Carlo paths...")
for i in range(N_PATHS):
    path = path_gen.next().value()
    for k, t_k in enumerate(monitor):
        idx         = min(k, N_STEPS)
        r_tk        = path[idx]
        all_r[i, k] = r_tk
        all_V[i, k] = irs_value(t_k, r_tk, PAY_DATES, k_par,
                                 HW_A, HW_SIGMA, params)
    if (i + 1) % 2000 == 0:
        print(f"  path {i+1:,}/{N_PATHS:,}")

print("Done.")

# =============================================================================
# Plot: swap value distribution at selected dates
# =============================================================================
plot_dates = [0, 24, 60, 96, 120]   # t = 0, 2, 5, 8, 10 years
labels     = ["$t=0$", "$t=2Y$", "$t=5Y$", "$t=8Y$", "$t=10Y$"]
colors     = plt.cm.tab10(np.linspace(0, 0.45, len(plot_dates)))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: swap value distributions (violin)
parts = axes[0].violinplot(
    [all_V[:, k] for k in plot_dates],
    positions=range(len(plot_dates)),
    showmedians=True, showextrema=False
)
for pc, col in zip(parts["bodies"], colors):
    pc.set_facecolor(col)
    pc.set_alpha(0.6)

# Overlay affine value at mean simulated r_t
for j, k in enumerate(plot_dates):
    t_k    = monitor[k]
    r_mean = np.mean(all_r[:, k])
    v_aff  = irs_value(t_k, r_mean, PAY_DATES, k_par, HW_A, HW_SIGMA, params)
    axes[0].scatter(j, v_aff, marker="D", color="black", zorder=5, s=40,
                    label="Affine at $\\bar{r}_{t_k}$" if j == 0 else "")

axes[0].axhline(0, color="k", lw=0.8, ls=":")
axes[0].set_xticks(range(len(plot_dates)))
axes[0].set_xticklabels(labels, fontsize=10)
axes[0].set_ylabel("$V_{\\mathrm{IRS}}(t_k, r_{t_k})$", fontsize=11)
axes[0].set_title("Distribution of MC Swap Values at Selected Dates\n"
                   "(diamond = affine value at mean $r_{t_k}$)", fontsize=10)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.25)

# Right: mean swap value — MC vs affine along the path mean
mean_r_path = np.mean(all_r, axis=0)
v_affine    = np.array([
    irs_value(monitor[k], mean_r_path[k], PAY_DATES, k_par, HW_A, HW_SIGMA, params)
    for k in range(len(monitor))
])
v_mc_mean   = np.mean(all_V, axis=0)

axes[1].plot(monitor, v_affine,  "b-",  lw=2.0, label="Affine at $\\bar{r}_{t_k}$")
axes[1].plot(monitor, v_mc_mean, "r--", lw=1.8, label="MC mean $V_{\\mathrm{IRS}}$")
axes[1].axhline(0, color="k", lw=0.8, ls=":")
axes[1].set_xlabel("Monitoring date $t_k$ (years)", fontsize=11)
axes[1].set_ylabel("$V_{\\mathrm{IRS}}$", fontsize=11)
axes[1].set_title("Affine vs Monte Carlo Mean Swap Value", fontsize=10)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.25)
axes[1].set_xlim(0, T_N)

plt.suptitle(
    f"IRS Pricing Under Hull-White  |  $a={HW_A}$, $\\sigma={HW_SIGMA}$, "
    f"$K^{{\\mathrm{{par}}}}={k_par*100:.4f}\\%$  |  "
    f"{cal_date.strftime('%d %B %Y')}",
    fontsize=10, y=1.01
)
plt.tight_layout()
out = os.path.join(OUT_DIR, "b2_irs_pricing.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
