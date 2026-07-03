"""
Script 1 of 2: PCA analysis of historical US Treasury zero curves.

Data source : feds200628.csv (Gürkaynak, Sack & Wright 2007)
Tenor grid  : TENORS from hw_lib.py (10 points)

Run this script first. Paste the printed output back so we can
confirm the PCA results before generating synthetic curves.
"""

import numpy as np
import pandas as pd
from hw_lib import svensson, TENORS

# ── 1. Load Svensson parameters ───────────────────────────────────────────────

feds = pd.read_csv("feds200628.csv", skiprows=9, na_values="NA")
feds["Date"] = pd.to_datetime(feds["Date"])
feds = feds.set_index("Date")
feds = feds[["BETA0","BETA1","BETA2","BETA3","TAU1","TAU2"]].dropna()

print(f"Dates loaded  : {len(feds)}")
print(f"Date range    : {feds.index.min().date()} → {feds.index.max().date()}")

# ── 2. Evaluate zero rates at the 10-tenor grid ───────────────────────────────
# X shape: (N, 10), decimal units (not %)

params = feds.values   # (N, 6): b0,b1,b2,b3,t1,t2
N = len(params)
X = np.zeros((N, len(TENORS)))
for i in range(N):
    b0, b1, b2, b3, t1, t2 = params[i]
    X[i] = svensson(TENORS, b0, b1, b2, b3, t1, t2) / 100.0

print(f"\nZero rate matrix X : {X.shape}  (decimal)")
print(f"Tenor grid (years) : {TENORS}")

print("\nRate range per tenor (%):")
for j, t in enumerate(TENORS):
    print(f"  T={t:5.4f}y : [{X[:,j].min()*100:.3f}, {X[:,j].max()*100: .3f}]%")

# ── 3. PCA from first principles ──────────────────────────────────────────────

mu  = X.mean(axis=0)          # (10,)
Xc  = X - mu                  # (N, 10)
C   = (Xc.T @ Xc) / (N - 1)  # (10, 10) sample covariance

vals, vecs = np.linalg.eigh(C)   # ascending
idx  = np.argsort(vals)[::-1]
vals = vals[idx]
vecs = vecs[:, idx]               # columns = eigenvectors

# ── 4. Variance explained ─────────────────────────────────────────────────────

total = vals.sum()
labels = ["Level","Slope","Curvature","PC4","PC5","PC6","PC7","PC8","PC9","PC10"]

print("\n" + "="*56)
print(f"{'PC':<5} {'Eigenvalue':>12}  {'Var%':>7}  {'Cum%':>7}  Label")
print("="*56)
cum = 0.0
for k in range(10):
    cum += vals[k] / total
    print(f"PC{k+1:<3}  {vals[k]:>12.8f}  "
          f"{vals[k]/total*100:>6.3f}%  {cum*100:>6.3f}%  {labels[k]}")

# ── 5. Eigenvector loadings ───────────────────────────────────────────────────

print("\nEigenvector loadings (first 5 PCs, rows=PC, cols=tenors):")
header = "".join(f"{t:>9.4f}" for t in TENORS)
print(f"{'':7}  {header}")
for k in range(5):
    row = "".join(f"{vecs[j,k]:>+9.5f}" for j in range(len(TENORS)))
    print(f"PC{k+1:<4}   {row}")

# ── 6. Factor score statistics ────────────────────────────────────────────────

E5 = vecs[:, :5]      # (10, 5)
Z  = Xc @ E5          # (N, 5) historical factor scores

print("\nHistorical factor score statistics (decimal):")
print(f"{'PC':<5}  {'mean':>10}  {'std':>10}  {'min':>10}  {'max':>10}")
for k in range(5):
    print(f"PC{k+1:<4}  {Z[:,k].mean():>10.6f}  {Z[:,k].std():>10.6f}  "
          f"{Z[:,k].min():>10.6f}  {Z[:,k].max():>10.6f}")

print("\nMean yield curve (%):")
for j, t in enumerate(TENORS):
    print(f"  T={t:6.4f}y : {mu[j]*100:.4f}%")
