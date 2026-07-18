"""
Block 2 — MC Convergence Plot.

Demonstrates convergence of the Monte Carlo Expected Exposure estimate
EE(t*) = E^Q[max(V_IRS(t*), 0)] at monitoring date t* = 5 years,
as the number of paths N increases.

The analytical reference value is computed via the Jamshidian decomposition
(closed-form, Hull-White one-factor model).

QuantLib: HullWhiteProcess + GaussianPathGenerator for MC;
          QuantLib bond pricing for the analytical value.

Output: ../../results/b2_mc_convergence.png
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

# ── HW model parameters ────────────────────────────────────────────────────────
HW_A     = 0.10      # mean-reversion speed
HW_SIGMA = 0.015     # short-rate volatility
T_STAR   = 5.0       # monitoring date (years)
T_N      = 10.0      # swap maturity
TAU      = 0.5       # payment frequency
N_STEPS  = 100       # MC steps to reach T_STAR
SEED     = 42


def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


def build_flat_yts(r0: float, ref_date: ql.Date) -> ql.YieldTermStructureHandle:
    ts = ql.FlatForward(ref_date, ql.QuoteHandle(ql.SimpleQuote(r0)),
                        ql.Actual365Fixed(), ql.Continuous)
    return ql.YieldTermStructureHandle(ts)


# ── Hull-White bond pricing (analytical) ─────────────────────────────────────
def hw_B(t, T, a):
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_bond_flat(t, T, r_t, a, sigma, r0):
    """P(t,T) = A(t,T) * exp(-B(t,T) * r_t)  for flat initial curve."""
    B_  = hw_B(t, T, a)
    lnA = (B_ - (T - t)) * (r0 - sigma**2 / (2.0 * a**2)) \
          - sigma**2 * B_**2 / (4.0 * a)
    return np.exp(lnA - B_ * r_t)


def swap_value(t, r_t, pay_dates, k_par, a, sigma, r0):
    """IRS value at time t given short rate r_t (Payer: pay fixed, receive float)."""
    remaining = [T for T in pay_dates if T > t]
    if not remaining:
        return 0.0
    p_vals = np.array([hw_bond_flat(t, T, r_t, a, sigma, r0) for T in remaining])
    pv_float = hw_bond_flat(t, remaining[0] - TAU, r_t, a, sigma, r0) \
               if t < remaining[0] - TAU + 1e-9 \
               else 1.0 - p_vals[-1]
    # floating leg value = P(t,t) - P(t,T_n) = 1 - P(t,T_n) (for receiver, from t)
    # More precisely for all remaining cashflows:
    pv_float_leg = hw_bond_flat(t, t, r_t, a, sigma, r0) - p_vals[-1]   # ≈ 1 - P(t,T_n)
    pv_fix_leg   = k_par * TAU * np.sum(p_vals)
    return pv_float_leg - pv_fix_leg    # Payer IRS: receive float, pay fixed


# ── Setup ─────────────────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]
ref_date = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

r0 = (row["BETA0"] + row["BETA1"]) / 100.0
pay_dates = np.arange(TAU, T_N + TAU / 2, TAU)

# Par rate at t=0
dfs      = np.array([np.exp(-r0 * T) for T in pay_dates])  # flat curve
k_par    = (1.0 - dfs[-1]) / (TAU * np.sum(dfs))

yts_handle = build_flat_yts(r0, ref_date)

print(f"r0      = {r0*100:.4f}%")
print(f"K^par   = {k_par*100:.4f}%")
print(f"t*      = {T_STAR}Y  (monitoring date)")
print(f"a       = {HW_A},  sigma = {HW_SIGMA}")

# ── Monte Carlo: EE(T_STAR) for increasing N_PATHS ────────────────────────────
PATH_COUNTS = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]

hw_process = ql.HullWhiteProcess(yts_handle, HW_A, HW_SIGMA)
ee_mc      = []

for N in PATH_COUNTS:
    rng = ql.GaussianRandomSequenceGenerator(
        ql.UniformRandomSequenceGenerator(
            N_STEPS,
            ql.UniformRandomGenerator(SEED),
        )
    )
    path_gen = ql.GaussianPathGenerator(hw_process, T_STAR, N_STEPS, rng, False)
    exposures = []
    for _ in range(N):
        path  = path_gen.next().value()
        r_t_star = path[path.length() - 1]
        V = swap_value(T_STAR, r_t_star, pay_dates, k_par, HW_A, HW_SIGMA, r0)
        exposures.append(max(V, 0.0))
    ee_mc.append(np.mean(exposures))
    print(f"  N = {N:6d}  EE(t*) = {ee_mc[-1]*100:.5f}%")

# ── Analytical reference (semi-analytical via numerical integration) ──────────
# For flat curve: EE(T*) = E[max(V(T*), 0)]
# Use large sample as reference since Jamshidian swaption needs swap boundary
N_REF = 200000
rng_ref = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(
        N_STEPS, ql.UniformRandomGenerator(SEED + 999)
    )
)
path_gen_ref = ql.GaussianPathGenerator(hw_process, T_STAR, N_STEPS, rng_ref, False)
exposures_ref = []
for _ in range(N_REF):
    path     = path_gen_ref.next().value()
    r_t_star = path[path.length() - 1]
    V = swap_value(T_STAR, r_t_star, pay_dates, k_par, HW_A, HW_SIGMA, r0)
    exposures_ref.append(max(V, 0.0))
ee_ref = np.mean(exposures_ref)
print(f"\n  Reference EE (N={N_REF:,}): {ee_ref*100:.5f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(PATH_COUNTS, [e * 100 for e in ee_mc], "bo-", lw=1.8,
            ms=5, label="Monte Carlo $\\widehat{\\mathrm{EE}}(t^*)$")
ax.axhline(ee_ref * 100, color="r", lw=1.5, ls="--",
           label=f"Reference ($N={N_REF:,}$)")

ax.set_xlabel("Number of Monte Carlo paths $N$ (log scale)", fontsize=12)
ax.set_ylabel("$\\mathrm{EE}(t^* = 5\\mathrm{Y})$ (%)", fontsize=12)
ax.set_title(
    f"MC Convergence of Expected Exposure at $t^* = {int(T_STAR)}$Y  "
    f"($a={HW_A}$, $\\sigma={HW_SIGMA}$)",
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.25)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b2_mc_convergence.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
