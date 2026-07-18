"""
Block 3 — Jamshidian vs Monte Carlo Expected Exposure Comparison.

Compares EE(t_k) computed by:
  (a) Jamshidian decomposition — closed-form swaption formula under HW1F
  (b) Monte Carlo simulation   — QuantLib HullWhiteProcess

If ee_comparison_results.csv is present in FinalResults/, that data is
plotted directly. Otherwise both methods are computed from scratch.

EE via Jamshidian:
  EE(t_k) = sum_{j: T_j > t_k} [ K^par * tau * ZBP(t_k, T_j) - ZBC(t_k, T_j) ]
  where ZBP/ZBC are zero-bond put/call prices from the analytical HW formula
  (see Brigo & Mercurio, 2006, Chapter 3).

Output: ../../results/b3_ee_comparison.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import QuantLib as ql

HERE    = os.path.dirname(os.path.abspath(__file__))
FEDS    = os.path.join(HERE, "feds200628.csv")
CSV_IN  = os.path.abspath(os.path.join(HERE, "..", "ee_comparison_results.csv"))
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

SVEN_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
HW_A     = 0.10
HW_SIGMA = 0.015
T_N      = 10.0
TAU      = 0.5       # IRS payment frequency (semi-annual)
N_MON    = 120       # monitoring: monthly over 10Y
N_PATHS  = 20000
N_STEPS  = 120       # one MC step per monitoring month
SEED     = 42


def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


# ── Hull-White analytical helpers ─────────────────────────────────────────────
def hw_B(t, T, a):
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_sigma_P(t, T, a, sigma):
    """Volatility of log P(t,T) in Hull-White."""
    return sigma * hw_B(t, T, a) * np.sqrt((1.0 - np.exp(-2.0 * a * t)) / (2.0 * a))


def hw_bond_flat(t, T, r_t, a, sigma, r0):
    B_  = hw_B(t, T, a)
    lnA = (B_ - (T - t)) * (r0 - sigma**2 / (2.0 * a**2)) \
          - sigma**2 * B_**2 / (4.0 * a)
    return np.exp(lnA - B_ * r_t)


def zb_option_flat(t, T, S, K_strike, option_type, a, sigma, r0):
    """
    Price at t=0 of a European option on the zero-coupon bond P(t,T).
    Matures at time t, the bond matures at S > T.
    K_strike: strike price on the bond.
    Brigo & Mercurio (2006), Proposition 3.1.
    """
    P_tT  = hw_bond_flat(0.0, t, r0, a, sigma, r0)   # P(0,t) for flat curve = exp(-r0*t)
    P_tS  = hw_bond_flat(0.0, S, r0, a, sigma, r0)
    sig_P = hw_sigma_P(t, S, a, sigma)                 # sigma_P(t,S)

    if sig_P < 1e-12 or t < 1e-10:
        return max((P_tS - K_strike * P_tT) if option_type == "call" else
                   (K_strike * P_tT - P_tS), 0.0)

    h = (np.log(P_tS / (K_strike * P_tT)) / sig_P) + sig_P / 2.0
    if option_type == "call":
        return P_tS * st.norm.cdf(h) - K_strike * P_tT * st.norm.cdf(h - sig_P)
    else:
        return K_strike * P_tT * st.norm.cdf(-(h - sig_P)) - P_tS * st.norm.cdf(-h)


def ee_jamshidian(t_k, pay_dates, k_par, a, sigma, r0):
    """
    EE(t_k) via Jamshidian decomposition.
    EE = sum of zero-bond call options with strikes X_j (Brigo & Mercurio, §3.3).
    """
    if t_k >= T_N - 1e-10:
        return 0.0
    remaining = pay_dates[pay_dates > t_k + 1e-10]
    if len(remaining) == 0:
        return 0.0

    # Jamshidian strike: X_j = k_par*tau for j < n, X_n = 1 + k_par*tau
    strikes = k_par * TAU * np.ones(len(remaining))
    strikes[-1] += 1.0   # final notional on last bond

    # EE = sum_j ZBC(0; t_k, T_j, X_j) where ZBC = call on zero-coupon bond
    ee = sum(
        zb_option_flat(t_k, t_k, T_j, X_j, "call", a, sigma, r0)
        for T_j, X_j in zip(remaining, strikes)
    )
    return ee


def swap_value_flat(t, r_t, pay_dates, k_par, a, sigma, r0):
    remaining = pay_dates[pay_dates > t + 1e-10]
    if len(remaining) == 0:
        return 0.0
    p_vals   = hw_bond_flat(t, remaining, r_t, a, sigma, r0)
    pv_float = 1.0 - p_vals[-1]
    pv_fixed = k_par * TAU * np.sum(p_vals)
    return pv_float - pv_fixed


# ── Load curve ────────────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]
ref_date = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

r0 = (row["BETA0"] + row["BETA1"]) / 100.0
pay_dates  = np.arange(TAU, T_N + TAU / 2, TAU)      # semi-annual: 0.5..10.0
dfs        = np.exp(-r0 * pay_dates)
k_par      = (1.0 - dfs[-1]) / (TAU * np.sum(dfs))
monitor    = np.linspace(0.0, T_N, N_MON + 1)        # monthly: 0, 1/12, ..., 10.0

# ── Use CSV if available; otherwise compute ───────────────────────────────────
if os.path.isfile(CSV_IN):
    print(f"Loading pre-computed results from {CSV_IN}")
    df_csv      = pd.read_csv(CSV_IN)
    monitor     = df_csv["monitoring_date"].values
    ee_jams_arr = df_csv["EE_2a_jamshidian"].values
    ee_mc_arr   = df_csv["EE_2b_mc"].values
else:
    print("ee_comparison_results.csv not found — computing both methods...")

    # Jamshidian
    ee_jams_arr = np.array([ee_jamshidian(t, pay_dates, k_par, HW_A, HW_SIGMA, r0)
                             for t in monitor])

    # MC
    flat_ts    = ql.FlatForward(ref_date, ql.QuoteHandle(ql.SimpleQuote(r0)),
                                 ql.Actual365Fixed(), ql.Continuous)
    yts_handle = ql.YieldTermStructureHandle(flat_ts)
    hw_process = ql.HullWhiteProcess(yts_handle, HW_A, HW_SIGMA)
    rng        = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(N_STEPS, ql.UniformRandomGenerator(SEED))
    )
    path_gen   = ql.GaussianPathGenerator(hw_process, T_N, N_STEPS, rng, False)

    all_V = np.zeros((N_PATHS, len(monitor)))
    for i in range(N_PATHS):
        path = path_gen.next().value()
        for k, t_k in enumerate(monitor):
            # Step k aligns exactly with month k (N_STEPS = N_MON)
            idx   = min(k, path.length() - 1)
            r_t_k = path[idx]
            all_V[i, k] = swap_value_flat(t_k, r_t_k, pay_dates, k_par,
                                           HW_A, HW_SIGMA, r0)
        if (i + 1) % 5000 == 0:
            print(f"  path {i+1}/{N_PATHS}")

    ee_mc_arr = np.mean(np.maximum(all_V, 0.0), axis=0)
    print("Done.")

diff_bps = (ee_mc_arr - ee_jams_arr) * 10000.0

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(monitor, ee_jams_arr * 100, "b-",  lw=2.0, label="Jamshidian (analytical)")
ax1.plot(monitor, ee_mc_arr   * 100, "r--", lw=1.8, label=f"Monte Carlo ($N={N_PATHS:,}$)")
ax2.bar(monitor, diff_bps, width=0.35, alpha=0.25, color="grey",
        label="Difference (bps, right axis)")

ax1.set_xlabel("Monitoring date $t_k$ (years)", fontsize=12)
ax1.set_ylabel("EE (% of notional)", fontsize=12)
ax2.set_ylabel("MC $-$ Jamshidian (bps)", fontsize=10, color="grey")
ax1.set_xlim(0, T_N)
ax1.grid(True, alpha=0.25)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
ax1.set_title(
    "Expected Exposure: Jamshidian vs Monte Carlo\n"
    f"$a={HW_A}$, $\\sigma={HW_SIGMA}$, $r_0={r0*100:.2f}\\%$",
    fontsize=11,
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b3_ee_comparison.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"  Max |diff| = {np.max(np.abs(diff_bps)):.2f} bps")
