"""
EE Comparison: Model 2a (Jamshidian closed-form) vs Model 2b (Monte Carlo)
One curve date: 2026-03-27 from FEDS200628 dataset.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

# ── 1. Load FEDS data ─────────────────────────────────────────────────────────
feds = pd.read_csv('feds200628.csv', skiprows=9, na_values='NA')
feds['Date'] = pd.to_datetime(feds['Date'])
feds = feds.set_index('Date')
row = feds.loc['2026-03-27']

beta0, beta1, beta2, beta3 = row['BETA0'], row['BETA1'], row['BETA2'], row['BETA3']
tau1, tau2 = row['TAU1'], row['TAU2']

print("=" * 65)
print("EE COMPARISON: MODEL 2a (JAMSHIDIAN) vs MODEL 2b (MONTE CARLO)")
print("=" * 65)
print(f"Curve date : 2026-03-27  (FEDS200628 dataset)")
print(f"Svensson   : β0={beta0:.4f}  β1={beta1:.4f}  β2={beta2:.4f}  β3={beta3:.4f}")
print(f"             τ1={tau1:.4f}  τ2={tau2:.4f}")

# ── 2. Yield curve functions ──────────────────────────────────────────────────
def svensson(T, b0, b1, b2, b3, t1, t2):
    T = np.atleast_1d(np.asarray(T, dtype=float))
    safe = np.where(T < 1e-10, 1e-10, T)
    e1 = np.exp(-safe / t1); e2 = np.exp(-safe / t2)
    term1 = (1 - e1) / (safe / t1)
    y = b0 + b1*term1 + b2*(term1 - e1) + b3*((1-e2)/(safe/t2) - e2)
    return np.where(T < 1e-10, b0 + b1, y)

def zero_rate(T):
    return svensson(T, beta0, beta1, beta2, beta3, tau1, tau2) / 100.0

def discount_factor(T):
    T = np.atleast_1d(np.asarray(T, dtype=float))
    return np.exp(-zero_rate(T) * T)

def fwd_rate(T, dt=1e-7):
    T = np.atleast_1d(np.asarray(T, dtype=float))
    Tl = np.maximum(T - dt, 1e-10); Tr = T + dt
    return zero_rate(T) + T * (zero_rate(Tr) - zero_rate(Tl)) / (Tr - Tl)

r0  = (beta0 + beta1) / 100.0
lam = 0.10
eta = 0.015

print(f"\nHull-White : λ={lam}  η={eta}")
print(f"r_0        : {r0*100:.4f}%  (β0+β1, Svensson overnight limit)")

# ── 3. Hull-White bond price functions ───────────────────────────────────────
def B(t, T):
    return (1.0 - np.exp(-lam * (T - t))) / lam

def lnA(t, T):
    Bt = B(t, T)
    conv = (eta**2 / (4*lam)) * Bt**2 * (1 - np.exp(-2*lam*t))
    return np.log(discount_factor(T) / discount_factor(t)) + Bt*fwd_rate(t) - conv

def bond_price(t, T, r_t):
    return np.exp(lnA(t, T)) * np.exp(-B(t, T) * r_t)

# ── 4. IRS specification ──────────────────────────────────────────────────────
Tn      = 10.0
tau_pay = 0.25
pay_dates = np.arange(tau_pay, Tn + 1e-10, tau_pay)
n_pay   = len(pay_dates)
P0_pay  = discount_factor(pay_dates)
K_atm   = (1.0 - P0_pay[-1]) / (tau_pay * P0_pay.sum())

# coupon weights: c_j = K*tau for j<n, c_n = 1 + K*tau
c = np.full(n_pay, K_atm * tau_pay)
c[-1] += 1.0

print(f"\nIRS        : T_n={Tn}yr  quarterly  n={n_pay}  K_ATM={K_atm*100:.4f}%")
print(f"V(0) check : {1.0 - P0_pay[-1] - K_atm*tau_pay*P0_pay.sum():.2e}")

# ── 5. Closed-form IRS at monitoring date ─────────────────────────────────────
def irs_cf(t_k, r_t, k_idx):
    """V(t_k; r_t) = 1 - sum_j c_j P(t_k, T_j; r_t)  [remaining j > k]"""
    remaining_dates = pay_dates[k_idx:]
    rem_c = c[k_idx:]
    if len(remaining_dates) == 0:
        return np.zeros_like(r_t) if hasattr(r_t, '__len__') else 0.0
    P = bond_price(t_k, remaining_dates, r_t[:, None] if hasattr(r_t,'__len__') else r_t)
    return 1.0 - np.sum(rem_c * P, axis=-1)

# ── 6. Model 2a: Jamshidian closed-form EE ───────────────────────────────────
def ee_jamshidian(t_k, k_idx):
    remaining_dates = pay_dates[k_idx:]
    rem_c = c[k_idx:]
    if len(remaining_dates) == 0:
        return 0.0

    # Mean and std of r_{t_k} under Q
    mu  = fwd_rate(t_k)[0] + (eta**2 / (2*lam**2)) * (1 - np.exp(-lam*t_k))**2
    nu  = eta * np.sqrt((1 - np.exp(-2*lam*t_k)) / (2*lam))

    # Critical rate r* via bisection: sum c_j P(t_k, T_j; r*) = 1
    def coupon_bond(r):
        return np.sum(rem_c * bond_price(t_k, remaining_dates, r)) - 1.0

    # Search bounds: find bracket
    try:
        r_lo, r_hi = mu - 8*nu, mu + 8*nu
        if coupon_bond(r_lo) * coupon_bond(r_hi) > 0:
            return 0.0   # no crossing: EE=0 (deep OOM) or saturated
        r_star = brentq(coupon_bond, r_lo, r_hi, xtol=1e-12)
    except ValueError:
        return 0.0

    # d factors
    d0 = (mu - r_star) / nu
    Bj = B(t_k, remaining_dates)
    dj = d0 - Bj * nu

    # Convexity-adjusted bond prices P̃(0, T_j)
    conv_adj = np.exp(-Bj * (eta**2 / (2*lam**2)) * (1 - np.exp(-lam*t_k))**2)
    P_tilde  = discount_factor(remaining_dates) * conv_adj

    # EE = P(0,t_k)*Φ(d0) - Σ c_j * P̃(0,T_j) * Φ(d_j)
    EE = discount_factor(t_k)[0] * norm.cdf(d0) - np.sum(rem_c * P_tilde * norm.cdf(dj))
    return max(EE, 0.0)

# ── 7. Model 2b: Monte Carlo EE ───────────────────────────────────────────────
M = 50_000
np.random.seed(42)

# ── Correct Q-measure simulation using the Hull-White x-process ───────────────
# r_t = x_t + alpha(t),  x_t follows OU with mean 0 under Q
# alpha(t) = f(0,t) + eta^2/(2*lam^2) * (1-exp(-lam*t))^2
# Exact transition: x_{k+1} = x_k * exp(-lam*dt) + noise
#                  r_{k+1}  = x_{k+1} + alpha(t_{k+1})
def alpha(t):
    t = np.atleast_1d(np.asarray(t, dtype=float))
    return fwd_rate(t) + (eta**2 / (2*lam**2)) * (1 - np.exp(-lam*t))**2

r_paths = np.zeros((M, n_pay))
x_prev  = np.full(M, r0 - alpha(0.0)[0])  # x_0 = r_0 - alpha(0) = 0 since r_0=f(0,0)

for k, t_next in enumerate(pay_dates):
    t_prev = pay_dates[k-1] if k > 0 else 0.0
    dt     = t_next - t_prev
    e_lam  = np.exp(-lam * dt)
    nu_k   = eta * np.sqrt((1 - np.exp(-2*lam*dt)) / (2*lam))
    x_next = x_prev * e_lam + nu_k * np.random.standard_normal(M)
    r_paths[:, k] = x_next + alpha(t_next)[0]
    x_prev = x_next

V_paths = np.zeros((M, n_pay))
for k in range(n_pay):
    t_k = pay_dates[k]
    remaining_dates = pay_dates[k+1:]
    rem_c = c[k+1:]
    if len(remaining_dates) == 0:
        continue
    r_k  = r_paths[:, k]
    Pk_n = bond_price(t_k, Tn, r_k)
    Bk   = B(t_k, remaining_dates)
    Ak   = np.exp(lnA(t_k, remaining_dates))
    Pk_j = Ak * np.exp(-Bk * r_k[:, None])
    V_paths[:, k] = 1.0 - np.sum(rem_c * Pk_j, axis=1)

P0_mon  = discount_factor(pay_dates)
EE_mc   = P0_mon * np.mean(np.maximum(V_paths, 0.0), axis=0)

# ── 8. Compute Jamshidian EE profile ─────────────────────────────────────────
print(f"\nRunning Jamshidian closed-form for {n_pay} monitoring dates...")
EE_jam = np.array([ee_jamshidian(pay_dates[k], k+1) for k in range(n_pay)])
print("Done.")

# ── Verify MC short rate mean vs theoretical Q-measure mean ──────────────────
print("\nShort rate mean check (MC vs Q-theory):")
print(f"{'Date':>6}  {'MC mean%':>10}  {'Q-theory%':>12}  {'Diff(bps)':>10}")
print("-" * 44)
for k in [3, 7, 15, 19, 27, 39]:
    t = pay_dates[k]
    mc_mean  = r_paths[:, k].mean() * 100
    q_theory = alpha(t)[0] * 100
    print(f"{t:>6.2f}  {mc_mean:>10.4f}  {q_theory:>12.4f}  {(mc_mean-q_theory)*100:>+10.4f}")

# ── 9. Results table ──────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Date':>6}  {'EE_2a (Jamshidian)':>20}  {'EE_2b (MC M=50k)':>18}  {'Diff (bps)':>12}")
print("-" * 65)
for k in range(n_pay):
    a, b = EE_jam[k], EE_mc[k]
    diff_bps = (b - a) * 1e4
    print(f"{pay_dates[k]:>6.2f}  {a:>20.8f}  {b:>18.8f}  {diff_bps:>+12.4f}")

print("=" * 65)
max_diff = np.abs(EE_mc - EE_jam).max() * 1e4
peak_jam = EE_jam.max()
peak_mc  = EE_mc.max()
rel_err  = np.abs(EE_mc - EE_jam).max() / peak_jam * 100

print(f"\nSummary statistics")
print(f"  Peak EE_2a (Jamshidian) : {peak_jam:.6f}")
print(f"  Peak EE_2b (MC, M=50k)  : {peak_mc:.6f}")
print(f"  Max |diff|              : {max_diff:.4f} bps")
print(f"  Max relative error      : {rel_err:.4f}%")
print(f"  MC std error at peak    : {P0_mon[np.argmax(EE_jam)] * np.std(np.maximum(V_paths[:,np.argmax(EE_jam)],0))/np.sqrt(M)*1e4:.4f} bps")

# ── 10. Save results to CSV ───────────────────────────────────────────────────
results = pd.DataFrame({
    'monitoring_date' : pay_dates,
    'EE_2a_jamshidian': EE_jam,
    'EE_2b_mc'        : EE_mc,
    'diff_bps'        : (EE_mc - EE_jam) * 1e4,
})
results.to_csv('ee_comparison_results.csv', index=False)
print(f"\nResults saved to ee_comparison_results.csv")
