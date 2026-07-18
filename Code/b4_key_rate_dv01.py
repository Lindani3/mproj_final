"""
Block 4 — Analytical Key-Rate DV01 Profile.

Computes the analytical key-rate DV01 of a 10-year Payer IRS across
all 10 yield-curve knots using QuantLib bump-and-reprice.

Method:
  KR_DV01_j = [V(curve + 1bp at knot j) - V(base curve)] / (0.0001)

The curve is a SplineCurve (cubic spline) at payment-date knots, consistent
with the model1b data-generation setup (10 knots at 1..10Y in 1Y steps
matching the payment-date grid).

The base curve is taken from the most recent FEDS date (Svensson parameters
evaluated at the 10 knot tenors).

Output: ../../results/b4_key_rate_dv01.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import QuantLib as ql
from scipy.interpolate import CubicSpline

HERE    = os.path.dirname(os.path.abspath(__file__))
FEDS    = os.path.join(HERE, "feds200628.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

SVEN_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
TAU       = 0.5
T_N       = 10.0
DELTA     = 1e-4    # 1 bp bump
KNOT_TENORS = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])


def svensson(T, b0, b1, b2, b3, t1, t2):
    T  = np.atleast_1d(np.asarray(T, dtype=float))
    s  = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


class SplineCurveQL:
    """Cubic-spline zero curve backed by scipy, queried via scipy for speed."""
    def __init__(self, knot_tenors, y_nodes):
        self.knots   = knot_tenors
        self.y_nodes = y_nodes
        self._cs     = CubicSpline(knot_tenors, y_nodes, extrapolate=True)

    def zero_rate(self, T):
        return float(self._cs(max(T, 0.0)))

    def discount(self, T):
        return np.exp(-self.zero_rate(T) * T)

    def bumped(self, j, delta):
        y_new    = self.y_nodes.copy()
        y_nodes_bumped = y_new
        y_nodes_bumped[j] += delta
        return SplineCurveQL(self.knots, y_nodes_bumped)

    def par_rate(self, pay_dates):
        dfs     = np.array([self.discount(T) for T in pay_dates])
        annuity = TAU * np.sum(dfs)
        return (1.0 - dfs[-1]) / annuity

    def irs_value(self, pay_dates, k_par, t=0.0, r_t=None):
        """IRS value at t=0 using discount factors from the spline curve."""
        remaining = pay_dates[pay_dates > t + 1e-10]
        if len(remaining) == 0:
            return 0.0
        dfs    = np.array([self.discount(T) for T in remaining])
        pv_flt = self.discount(t) - dfs[-1]   # P(0,t) - P(0,T_n) — simplified
        pv_fix = k_par * TAU * np.sum(dfs)
        return pv_flt - pv_fix


# ── Load FEDS ─────────────────────────────────────────────────────────────────
feds = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
valid = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]

# ── Build base spline curve ────────────────────────────────────────────────────
y_nodes_base = svensson(
    KNOT_TENORS,
    row["BETA0"], row["BETA1"],
    row["BETA2"], row["BETA3"],
    row["TAU1"],  row["TAU2"],
) / 100.0   # decimal

base_curve = SplineCurveQL(KNOT_TENORS, y_nodes_base)
pay_dates  = np.arange(TAU, T_N + TAU / 2, TAU)
k_par      = base_curve.par_rate(pay_dates)
V_base     = base_curve.irs_value(pay_dates, k_par)

print(f"Calibration date : {cal_date.strftime('%d %B %Y')}")
print(f"K^par            : {k_par*100:.4f}%")
print(f"V_base at t=0    : {V_base:.8f}  (should be ~0 for par swap)")

# ── Bump-and-reprice for each knot ────────────────────────────────────────────
kr_dv01 = np.zeros(len(KNOT_TENORS))
for j in range(len(KNOT_TENORS)):
    bumped_curve = base_curve.bumped(j, DELTA)
    V_bumped     = bumped_curve.irs_value(pay_dates, k_par)
    kr_dv01[j]   = (V_bumped - V_base) / DELTA
    print(f"  Knot {KNOT_TENORS[j]:.1f}Y  KR-DV01 = {kr_dv01[j]*10000:.4f} bps/1bp")

# Also compute total DV01 (parallel shift of 1bp)
y_par_bumped = y_nodes_base + DELTA
par_curve    = SplineCurveQL(KNOT_TENORS, y_par_bumped)
dv01_total   = (par_curve.irs_value(pay_dates, k_par) - V_base) / DELTA
print(f"\nTotal DV01 (parallel 1bp shift): {dv01_total*10000:.4f} bps")
print(f"Sum of KR-DV01s:                  {kr_dv01.sum()*10000:.4f} bps")

# ── Plot ──────────────────────────────────────────────────────────────────────
tenor_labels = [f"{int(T)}Y" for T in KNOT_TENORS]
colours      = ["#d62728" if d < 0 else "#1f77b4" for d in kr_dv01]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(tenor_labels, kr_dv01 * 10000, color=colours, alpha=0.85)
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("Yield-curve knot (tenor)", fontsize=12)
ax.set_ylabel("Key-Rate DV01 (bp per 1bp shift)", fontsize=12)
ax.set_title(
    f"Key-Rate DV01 Profile — 10Y Payer IRS\n"
    f"Svensson Curve, {cal_date.strftime('%d %B %Y')}, "
    f"$K^{{\\mathrm{{par}}}}={k_par*100:.4f}\\%$",
    fontsize=11,
)
ax.grid(True, axis="y", alpha=0.25)

# Annotate bars with values
for bar, val in zip(bars, kr_dv01 * 10000):
    ax.text(bar.get_x() + bar.get_width() / 2.0,
            val + (0.3 if val >= 0 else -0.6),
            f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top",
            fontsize=8)

plt.tight_layout()
out = os.path.join(OUT_DIR, "b4_key_rate_dv01.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
