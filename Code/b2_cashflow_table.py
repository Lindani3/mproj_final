"""
Block 2 — IRS Cashflow Table: Fixed vs Floating at t=0.

For a 10-year payer IRS initiated at par under the Svensson initial term
structure, computes at each semi-annual payment date T_j:

  Fixed CF   = K^par * tau
  Floating CF = P(0, T_{j-1}) - P(0, T_j)   [single-curve, no-arbitrage]
  Net CF      = Floating CF - Fixed CF         [from payer's perspective]
  PV(Net CF)  = Net CF * P(0, T_j)

Sum of PV(Net CF) = 0 verifies the par condition.

Outputs:
  ../../results/b2_cashflow_table.csv
  ../../results/b2_cashflow_table.tex
  ../../results/b2_cashflow_profile.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HERE    = os.path.dirname(os.path.abspath(__file__))
FEDS    = os.path.join(HERE, "feds200628.csv")
OUT_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results"))
os.makedirs(OUT_DIR, exist_ok=True)

SVEN_COLS = ["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]
T_N  = 10.0
TAU  = 0.5
N    = 20   # payment dates


def svensson(T, b0, b1, b2, b3, t1, t2):
    T   = np.atleast_1d(np.asarray(T, dtype=float))
    s   = np.where(T < 1e-10, 1e-10, T)
    e1  = np.exp(-s / t1);  e2 = np.exp(-s / t2)
    ph1 = (1.0 - e1) / (s / t1);  ph2 = ph1 - e1
    ph3 = (1.0 - e2) / (s / t2) - e2
    y   = b0 + b1 * ph1 + b2 * ph2 + b3 * ph3
    return np.where(T < 1e-10, b0 + b1, y)


def disc(T, params):
    y = svensson(T, *params) / 100.0
    return np.exp(-y * T)


# ── Load FEDS ─────────────────────────────────────────────────────────────────
feds     = pd.read_csv(FEDS, skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds     = feds.set_index("Date")
valid    = feds[feds[SVEN_COLS].notna().all(axis=1) & (feds["TAU1"] > 0)]
row      = valid.iloc[-1]
cal_date = valid.index[-1]
params   = row[SVEN_COLS].values

r0 = (row["BETA0"] + row["BETA1"]) / 100.0

# Payment and accrual start dates
pay_dates   = np.arange(TAU, T_N + TAU / 2, TAU)         # T_1 .. T_20
start_dates = np.concatenate([[0.0], pay_dates[:-1]])     # T_0=0, T_1, .., T_19

# Discount factors
P_pay   = disc(pay_dates,   params)   # P(0, T_j),     j=1..20
P_start = disc(start_dates, params)   # P(0, T_{j-1}), j=1..20  (P(0,0)=1)

# Par rate
k_par = (1.0 - P_pay[-1]) / (TAU * np.sum(P_pay))

# Cashflows
fixed_cf   = np.full(N, k_par * TAU)
float_cf   = P_start - P_pay                # P(0,T_{j-1}) - P(0,T_j)
net_cf     = float_cf - fixed_cf
pv_net     = net_cf * P_pay

print(f"Calibration date : {cal_date.strftime('%d %B %Y')}")
print(f"r0               : {r0*100:.4f}%")
print(f"K^par            : {k_par*100:.4f}%")
print(f"Sum PV(Net CF)   : {np.sum(pv_net):.6e}  (par condition: should be ~0)")

# ── Build DataFrame ───────────────────────────────────────────────────────────
df = pd.DataFrame({
    "T_j":        pay_dates,
    "P(0,T_j)":   P_pay,
    "Fixed CF":   fixed_cf,
    "Float CF":   float_cf,
    "Net CF":     net_cf,
    "PV(Net CF)": pv_net,
})

# ── Save CSV ──────────────────────────────────────────────────────────────────
csv_out = os.path.join(OUT_DIR, "b2_cashflow_table.csv")
df.to_csv(csv_out, index=False, float_format="%.6f")
print(f"Saved CSV: {csv_out}")

# ── Save LaTeX table ──────────────────────────────────────────────────────────
tex_out = os.path.join(OUT_DIR, "b2_cashflow_table.tex")
with open(tex_out, "w") as f:
    f.write("\\begin{tabular}{@{}rrrrrr@{}}\n")
    f.write("\\toprule\n")
    f.write("$T_j$ & $P(0,T_j)$ & Fixed CF & Float CF & Net CF & PV(Net CF) \\\\\n")
    f.write("\\midrule\n")
    for _, r in df.iterrows():
        f.write(
            f"{r['T_j']:.1f} & {r['P(0,T_j)']:.4f} & "
            f"{r['Fixed CF']*100:.4f}\\% & {r['Float CF']*100:.4f}\\% & "
            f"{r['Net CF']*100:.4f}\\% & {r['PV(Net CF)']*10000:.4f}bp \\\\\n"
        )
    f.write("\\midrule\n")
    f.write(
        f"\\multicolumn{{5}}{{r}}{{\\textbf{{Total PV(Net CF)}}}} & "
        f"{np.sum(pv_net)*10000:.4f}bp \\\\\n"
    )
    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
print(f"Saved LaTeX: {tex_out}")

# ── Plot ──────────────────────────────────────────────────────────────────────
width = 0.18
x     = pay_dates

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8),
                                gridspec_kw={"height_ratios": [2, 1]})

# Top: fixed vs floating vs net cashflows
ax1.bar(x - width, fixed_cf * 100, width=width, color="#1f77b4",
        alpha=0.85, label="Fixed CF = $K^{\\mathrm{par}}\\tau$")
ax1.bar(x,         float_cf * 100, width=width, color="#d62728",
        alpha=0.85, label="Float CF = $P(0,T_{j-1}) - P(0,T_j)$")
ax1.bar(x + width, net_cf   * 100, width=width, color="#2ca02c",
        alpha=0.85, label="Net CF (Float $-$ Fixed)")
ax1.axhline(0, color="k", lw=0.8, ls=":")
ax1.set_ylabel("Cash flow (% of notional)", fontsize=11)
ax1.set_title(
    f"IRS Cashflow Profile at $t=0$  |  $K^{{\\mathrm{{par}}}}={k_par*100:.4f}\\%$  |  "
    f"{cal_date.strftime('%d %B %Y')}",
    fontsize=10
)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2)
ax1.set_xlim(0, T_N + TAU)

# Bottom: discount factors
ax2.plot(pay_dates, P_pay, "o-", color="#1f77b4", lw=1.8, ms=4,
         label="$P(0,T_j)$ — Svensson")
ax2.set_xlabel("Payment date $T_j$ (years)", fontsize=11)
ax2.set_ylabel("Discount factor", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.2)
ax2.set_xlim(0, T_N + TAU)

plt.tight_layout()
png_out = os.path.join(OUT_DIR, "b2_cashflow_profile.png")
plt.savefig(png_out, dpi=150, bbox_inches="tight")
print(f"Saved plot: {png_out}")

# ── Print summary table ───────────────────────────────────────────────────────
print(f"\n{'T_j':>5} {'P(0,Tj)':>9} {'Fixed CF%':>10} "
      f"{'Float CF%':>10} {'Net CF%':>10} {'PV(Net) bp':>12}")
print("-" * 62)
for _, r in df.iterrows():
    print(f"{r['T_j']:>5.1f} {r['P(0,T_j)']:>9.4f} "
          f"{r['Fixed CF']*100:>10.4f} {r['Float CF']*100:>10.4f} "
          f"{r['Net CF']*100:>10.4f} {r['PV(Net CF)']*10000:>12.4f}")
print("-" * 62)
print(f"{'Total':>47} {np.sum(pv_net)*10000:>12.4f}")
