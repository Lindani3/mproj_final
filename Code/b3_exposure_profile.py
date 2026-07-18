"""
Block 3 — EPE and ENE Profiles.

Computes Expected Positive Exposure (EPE) and Expected Negative Exposure (ENE)
of a 10-year Payer IRS under the Hull-White one-factor model via Monte Carlo.

  EPE(t_k) = E^Q[ max(V_IRS(t_k), 0) ]
  ENE(t_k) = E^Q[ min(V_IRS(t_k), 0) ]

IRS payment dates : semi-annual, T_j = 0.5j, j = 1..20.
Monitoring dates  : monthly,     t_k = k/12, k = 0, 1, ..., 120.
QuantLib: HullWhiteProcess + GaussianPathGenerator.

Output: ../../results/b3_exposure_profile.png
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

HW_A     = 0.10
HW_SIGMA = 0.015
T_N      = 10.0
TAU      = 0.5       # IRS payment frequency (semi-annual)
N_MON    = 120       # monitoring steps: monthly over 10Y (120 months)
N_PATHS  = 10000
N_STEPS  = 120       # one MC step per monitoring month — aligns path grid exactly
SEED     = 42


def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


def hw_B(t, T, a):
    return (1.0 - np.exp(-a * (T - t))) / a


def hw_bond_flat(t, T, r_t, a, sigma, r0):
    """Analytical HW bond price for flat initial curve."""
    B_  = hw_B(t, T, a)
    lnA = (B_ - (T - t)) * (r0 - sigma**2 / (2.0 * a**2)) \
          - sigma**2 * B_**2 / (4.0 * a)
    return np.exp(lnA - B_ * r_t)


def swap_value_flat(t, r_t, pay_dates, k_par, a, sigma, r0):
    """Payer IRS value at (t, r_t): receive float, pay fixed."""
    remaining = pay_dates[pay_dates > t + 1e-10]
    if len(remaining) == 0:
        return 0.0
    p_vals     = hw_bond_flat(t, remaining, r_t, a, sigma, r0)
    pv_float   = 1.0 - p_vals[-1]     # P(t,t) - P(t,T_n) for the remaining float leg
    pv_fixed   = k_par * TAU * np.sum(p_vals)
    return pv_float - pv_fixed


# ── Load FEDS, setup ─────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]
ref_date = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

r0 = (row["BETA0"] + row["BETA1"]) / 100.0
flat_ts    = ql.FlatForward(ref_date, ql.QuoteHandle(ql.SimpleQuote(r0)),
                             ql.Actual365Fixed(), ql.Continuous)
yts_handle = ql.YieldTermStructureHandle(flat_ts)

pay_dates     = np.arange(TAU, T_N + TAU / 2, TAU)          # semi-annual: 0.5..10.0
dfs           = np.exp(-r0 * pay_dates)
k_par         = (1.0 - dfs[-1]) / (TAU * np.sum(dfs))
monitor_dates = np.linspace(0.0, T_N, N_MON + 1)            # monthly: 0, 1/12, ..., 10.0

print(f"r0 = {r0*100:.4f}%  |  K^par = {k_par*100:.4f}%  |  N_paths = {N_PATHS}")
print(f"Payment dates : semi-annual ({len(pay_dates)} dates)")
print(f"Monitoring    : monthly ({len(monitor_dates)} dates)")
print("Simulating MC paths...")

# ── Monte Carlo ───────────────────────────────────────────────────────────────
hw_process = ql.HullWhiteProcess(yts_handle, HW_A, HW_SIGMA)

# Generate N_PATHS of the full 10-year path (N_STEPS steps over T_N years)
rng = ql.GaussianRandomSequenceGenerator(
    ql.UniformRandomSequenceGenerator(N_STEPS, ql.UniformRandomGenerator(SEED))
)
path_gen = ql.GaussianPathGenerator(hw_process, T_N, N_STEPS, rng, False)

# Storage: (N_PATHS, len(monitor_dates))
all_V = np.zeros((N_PATHS, len(monitor_dates)))

for i in range(N_PATHS):
    path = path_gen.next().value()
    for k, t_k in enumerate(monitor_dates):
        # Each MC step = 1 month = T_N/N_STEPS, so step index = k exactly
        idx   = min(k, path.length() - 1)
        r_t_k = path[idx]
        all_V[i, k] = swap_value_flat(t_k, r_t_k, pay_dates, k_par, HW_A, HW_SIGMA, r0)

    if (i + 1) % 2000 == 0:
        print(f"  path {i+1:5d}/{N_PATHS}")

epe = np.mean(np.maximum(all_V, 0.0), axis=0)
ene = np.mean(np.minimum(all_V, 0.0), axis=0)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(monitor_dates, epe * 100, "b-",  lw=2.0, label="EPE = $E^Q[\\max(V_{t_k},0)]$")
ax.plot(monitor_dates, ene * 100, "r--", lw=2.0, label="ENE = $E^Q[\\min(V_{t_k},0)]$")
ax.fill_between(monitor_dates, epe * 100, alpha=0.15, color="blue")
ax.fill_between(monitor_dates, ene * 100, alpha=0.15, color="red")
ax.axhline(0, color="k", lw=0.8, ls=":")

ax.set_xlabel("Monitoring date $t_k$ (years)", fontsize=12)
ax.set_ylabel("Exposure (% of notional)", fontsize=12)
ax.set_xlim(0, T_N)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.25)
ax.set_title(
    f"EPE and ENE Profile — 10Y Payer IRS under Hull-White\n"
    f"$a={HW_A}$, $\\sigma={HW_SIGMA}$, $N={N_PATHS:,}$ paths, "
    f"$r_0={r0*100:.2f}\\%$, $K^{{\\mathrm{{par}}}}={k_par*100:.4f}\\%$",
    fontsize=11,
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b3_exposure_profile.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")
print(f"  Peak EPE: {np.max(epe)*100:.4f}% at t = {monitor_dates[np.argmax(epe)]:.1f}Y")
