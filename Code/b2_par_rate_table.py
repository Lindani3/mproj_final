"""
Block 2 — Par Rate Table.

Computes the par swap rate K^par(T) for standard maturities using the
QuantLib VanillaSwap (single-curve, DiscountingSwapEngine) built on top
of the Svensson zero curve from the most recent FEDS date.

The par rate satisfies:  K^par(T) = (1 - P(0,T)) / sum_{j=1}^{2T} 0.5 * P(0, 0.5j)

Outputs:
  ../../results/b2_par_rate_table.csv   (values for inspection)
  ../../results/b2_par_rate_table.tex   (ready-to-include LaTeX tabular)
"""
import os
import numpy as np
import pandas as pd
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


def par_rate_analytical(yts, tenor_years: float, tau: float = 0.5) -> float:
    """
    Analytical par rate for a vanilla IRS (single-curve, no credit spread):
        K^par = (P(0,0) - P(0,T)) / (tau * sum_{j=1}^{n} P(0, j*tau))
    """
    n       = int(round(tenor_years / tau))
    pay_df  = [yts.discount(j * tau) for j in range(1, n + 1)]
    annuity = tau * sum(pay_df)
    k_par   = (1.0 - pay_df[-1]) / annuity
    return k_par


def par_rate_ql(yts_handle, ref_date, tenor_years: int) -> float:
    """
    Par rate from QuantLib VanillaSwap.fairRate() — single-curve approach.
    Floating index uses the same discount curve as forecast curve.
    """
    calendar = ql.NullCalendar()
    dc_fixed = ql.Thirty360(ql.Thirty360.BondBasis)
    dc_float = ql.Actual360()
    settle   = ref_date                              # no spot lag (continuous-time model)
    maturity = ref_date + ql.Period(tenor_years, ql.Years)

    fixed_schedule = ql.Schedule(
        settle, maturity,
        ql.Period(ql.Semiannual),
        calendar,
        ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Forward, False,
    )
    float_schedule = ql.Schedule(
        settle, maturity,
        ql.Period(ql.Semiannual),
        calendar,
        ql.Unadjusted, ql.Unadjusted,
        ql.DateGeneration.Forward, False,
    )
    # Dummy fixed rate; fairRate() will override this
    swap = ql.VanillaSwap(
        ql.VanillaSwap.Payer,
        1.0,                        # unit notional
        fixed_schedule, 0.05,       # placeholder fixed rate
        dc_fixed,
        float_schedule,
        ql.IborIndex("Ibor", ql.Period(ql.Semiannual), 0,
                     ql.USDCurrency(), calendar, ql.Unadjusted, False, dc_float,
                     yts_handle),
        0.0,                        # no spread on floating leg
        dc_float,
    )
    engine = ql.DiscountingSwapEngine(yts_handle)
    swap.setPricingEngine(engine)
    return swap.fairRate()


# ── Load FEDS, pick most recent valid date ────────────────────────────────────
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

# ── Compute par rates ─────────────────────────────────────────────────────────
TENORS = [1, 2, 3, 5, 7, 10]
records = []
for T in TENORS:
    k_anal = par_rate_analytical(yts, T)
    k_ql   = par_rate_ql(yts_handle, ref_date, T)
    p_T    = yts.discount(float(T))
    records.append({
        "Tenor (yr)":          T,
        "P(0,T)":              round(p_T,    6),
        "K^par analytical (%)": round(k_anal * 100, 4),
        "K^par QL (%)":         round(k_ql   * 100, 4),
    })

df = pd.DataFrame(records)
print(df.to_string(index=False))
print(f"\nCalibration date: {cal_date.strftime('%d %B %Y')}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_out = os.path.join(OUT_DIR, "b2_par_rate_table.csv")
df.to_csv(csv_out, index=False)
print(f"Saved: {csv_out}")

# ── Save LaTeX tabular ────────────────────────────────────────────────────────
tex_out = os.path.join(OUT_DIR, "b2_par_rate_table.tex")
latex_df = df[["Tenor (yr)", "P(0,T)", "K^par analytical (%)"]].copy()
latex_df.columns = ["Tenor (yr)", "$P(0,T)$", "$K^{\\mathrm{par}}$ (\\%)"]

with open(tex_out, "w") as f:
    f.write("% Par rate table — auto-generated by b2_par_rate_table.py\n")
    f.write(f"% Calibration date: {cal_date.strftime('%d %B %Y')}\n")
    f.write("\\begin{tabular}{@{}ccc@{}}\n")
    f.write("\\toprule\n")
    f.write("Tenor (yr) & $P(0,T)$ & $K^{\\mathrm{par}}$ (\\%) \\\\\n")
    f.write("\\midrule\n")
    for _, r in latex_df.iterrows():
        f.write(f"{int(r['Tenor (yr)'])} & "
                f"{r['$P(0,T)$']:.6f} & "
                f"{r['$K^{{\\mathrm{{par}}}}$ (\\%)']:.4f} \\\\\n")
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
print(f"Saved: {tex_out}")
