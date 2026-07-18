"""
Block 2 — Fixed vs Floating Cashflow Profile.

Bar chart showing the fixed and floating cashflows at each semi-annual
payment date T_j = 0.5j, j=1..20, for the 10-year IRS.

Fixed cashflows:    CF_fix(T_j)  = K^par * tau * N  (constant)
Floating cashflows: CF_flt(T_j)  = (P(0,T_{j-1})/P(0,T_j) - 1) * N
                                  = forward rate over [T_{j-1}, T_j] * tau * N

Both expressed per unit notional (N=1).
QuantLib ZeroCurve provides the discount factors via yts.discount(T).

Output: ../../results/b2_cashflow_profile.png
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
TAU       = 0.5
T_N       = 10.0


def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


def build_yts(row, ref_date):
    grid  = [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    rates = svensson(np.array(grid),
                     row["BETA0"], row["BETA1"],
                     row["BETA2"], row["BETA3"],
                     row["TAU1"],  row["TAU2"]) / 100.0
    ql_dates = [ref_date + ql.Period(int(round(g * 12)), ql.Months) for g in grid]
    curve = ql.ZeroCurve(
        ql_dates, rates.tolist(),
        ql.Actual365Fixed(), ql.NullCalendar(),
        ql.Linear(), ql.Continuous,
    )
    return ql.YieldTermStructureHandle(curve)


# ── Load FEDS ─────────────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]
ref_date = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

yts_handle = build_yts(row, ref_date)
yts        = yts_handle.currentLink()

# ── Cashflows ─────────────────────────────────────────────────────────────────
pay_dates  = np.arange(TAU, T_N + TAU / 2, TAU)   # 0.5, 1.0, ..., 10.0
reset_dates = np.concatenate([[0.0], pay_dates[:-1]])  # 0.0, 0.5, ..., 9.5

df_T  = np.array([yts.discount(T) for T in pay_dates])
df_T0 = np.array([yts.discount(T) for T in reset_dates])

# Forward rate L_j = (P(0, T_{j-1}) / P(0, T_j) - 1) / tau
fwd_rates = (df_T0 / df_T - 1.0) / TAU

# Par rate
annuity   = TAU * np.sum(df_T)
k_par     = (1.0 - df_T[-1]) / annuity

# Cashflows (per unit notional)
cf_fix = k_par * TAU * np.ones(len(pay_dates))
cf_flt = fwd_rates * TAU

print(f"Calibration date : {cal_date.strftime('%d %B %Y')}")
print(f"K^par            : {k_par*100:.4f}%")
print(f"Fixed CF (const) : {cf_fix[0]*100:.4f}% per period")

# ── Plot ──────────────────────────────────────────────────────────────────────
width = 0.22
x     = np.arange(len(pay_dates))

fig, ax = plt.subplots(figsize=(12, 5))

bars_fix = ax.bar(x - width / 2, cf_fix * 100, width,
                  label="Fixed leg $K^{\\mathrm{par}} \\cdot \\tau$",
                  color="#1f77b4", alpha=0.85)
bars_flt = ax.bar(x + width / 2, cf_flt * 100, width,
                  label="Floating leg $L_j \\cdot \\tau$",
                  color="#d62728", alpha=0.85)

ax.set_xlabel("Payment date $T_j$ (years)", fontsize=12)
ax.set_ylabel("Cashflow (% of notional)", fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels([f"{T:.1f}" for T in pay_dates], rotation=45, fontsize=8)
ax.legend(fontsize=10)
ax.grid(True, axis="y", alpha=0.25)
ax.set_title(
    f"10-Year IRS Cashflow Profile — Svensson Curve ({cal_date.strftime('%d %B %Y')})\n"
    f"$K^{{\\mathrm{{par}}}} = {k_par*100:.4f}\\%$, $\\tau = {TAU}$, $N=1$",
    fontsize=11,
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b2_cashflow_profile.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
