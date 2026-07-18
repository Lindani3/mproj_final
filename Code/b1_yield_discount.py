"""
Block 1 — Figure 2: Dual-Axis Zero Rate and Discount Factor.

Uses the most recent FEDS date with valid Svensson parameters.
Left y-axis: continuously-compounded zero rate y(0,T) in percent.
Right y-axis: discount factor P(0,T) = exp(-y(0,T)*T).
QuantLib ZeroCurve provides the query interface.

Output: ../../results/b1_yield_discount.png
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


def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1)
    ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


def build_ql_curve(row, ref_date: ql.Date) -> ql.ZeroCurveHandle:
    """Build a QuantLib ZeroCurve handle from a FEDS row."""
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


# ── Load FEDS, pick most recent valid date ────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]

row       = valid.iloc[-1]
cal_date  = valid.index[-1]
ref_date  = ql.Date(int(cal_date.day), int(cal_date.month), int(cal_date.year))
ql.Settings.instance().evaluationDate = ref_date

yts_handle = build_ql_curve(row, ref_date)
yts        = yts_handle.currentLink()

# ── Evaluate zero rates and discount factors via QuantLib ─────────────────────
T_grid = np.linspace(0.083, 30, 600)
dc     = ql.Actual365Fixed()

zero_rates = np.array([
    yts.zeroRate(t, dc, ql.Continuous, ql.Annual).rate() * 100.0
    for t in T_grid
])
disc_factors = np.array([yts.discount(t) for t in T_grid])

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

l1, = ax1.plot(T_grid, zero_rates,   "b-",  lw=2.0, label="Zero rate $y(0,T)$")
l2, = ax2.plot(T_grid, disc_factors, "r--", lw=2.0, label="Discount factor $P(0,T)$")

ax1.set_xlabel("Maturity $T$ (years)", fontsize=12)
ax1.set_ylabel("Zero rate (%)", color="b", fontsize=12)
ax2.set_ylabel("Discount factor $P(0,T)$", color="r", fontsize=12)
ax1.tick_params(axis="y", labelcolor="b")
ax2.tick_params(axis="y", labelcolor="r")
ax1.set_xlim(0, 30)
ax1.grid(True, alpha=0.25)

lines  = [l1, l2]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper right", fontsize=11)
ax1.set_title(
    f"US Treasury Svensson Yield Curve — {cal_date.strftime('%d %B %Y')}",
    fontsize=12,
)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b1_yield_discount.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
print(f"  Calibration date : {cal_date.strftime('%d %B %Y')}")
print(f"  Short rate r(0)  : {zero_rates[0]:.4f}%")
print(f"  P(0, 10Y)        : {yts.discount(10.0):.6f}")
