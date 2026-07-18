"""
Block 1 — Figure 1: Five Svensson Yield Curves.

Selects five historically distinct dates from the FEDS200628 dataset
representing different term-structure regimes (normal, humped, flat,
inverted, steep) and plots zero-coupon yield curves.

Output: ../../results/b1_yield_curves.png
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

# ── Svensson (1994) zero rate (in percent) ────────────────────────────────────
def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1)
    e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1)
    ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


# ── QuantLib curve builder (used to verify / can be extended for pricing) ─────
def build_ql_curve(params: dict, ref_date: ql.Date) -> ql.ZeroCurve:
    """Return a QuantLib ZeroCurve from Svensson parameters."""
    grid = [0.083, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30]
    rates = svensson(np.array(grid),
                     params["b0"], params["b1"],
                     params["b2"], params["b3"],
                     params["t1"], params["t2"]) / 100.0
    ql_dates = [ref_date + ql.Period(int(round(g * 12)), ql.Months) for g in grid]
    return ql.ZeroCurve(
        ql_dates, rates.tolist(),
        ql.Actual365Fixed(), ql.NullCalendar(),
        ql.Linear(), ql.Continuous,
    )


# ── Load FEDS data ────────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]

# Five representative regimes — edit dates if any are missing in your dataset
REGIMES = {
    "Normal (2014-01-02)":   "2014-01-02",
    "Humped (2006-08-01)":   "2006-08-01",
    "Flat (2000-12-29)":     "2000-12-29",
    "Inverted (2000-07-03)": "2000-07-03",
    "Steep (2010-08-02)":    "2010-08-02",
}

T = np.linspace(0.083, 30, 600)

fig, ax = plt.subplots(figsize=(10, 5.5))
colours = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]

for (label, d), col in zip(REGIMES.items(), colours):
    # fall back to nearest available date if exact date not in dataset
    try:
        row = valid.loc[d]
    except KeyError:
        nearest = valid.index[valid.index.searchsorted(pd.Timestamp(d))]
        row = valid.iloc[valid.index.get_loc(nearest)]
        print(f"  {d} not found; using {valid.index[valid.index.searchsorted(pd.Timestamp(d))]}")

    y = svensson(T, row["BETA0"], row["BETA1"],
                 row["BETA2"], row["BETA3"],
                 row["TAU1"],  row["TAU2"])
    ax.plot(T, y, lw=1.8, color=col, label=label)

ax.set_xlabel("Maturity $T$ (years)", fontsize=12)
ax.set_ylabel("Zero-coupon yield (%)", fontsize=12)
ax.set_xlim(0, 30)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(True, alpha=0.25)
plt.tight_layout()

out = os.path.join(OUT_DIR, "b1_yield_curves.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
