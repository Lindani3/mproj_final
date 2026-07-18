"""
c3_datagen_2.py
===============
Generate training data for Model 2 (EPE — expected positive exposure).

For each (FEDS date, a, sigma, t_k):
    label = E[max(V_MC(t_k), 0)]

Outer paths record r_{t_k} for each monitoring date.
Inner MC computes IRS value from each outer path's r_{t_k}.
EPE is the mean of max(V, 0) across all outer paths.

Grid
----
  16,160 FEDS curves  x  8 a-values  x  8 sigma-values  x  21 t_k
  = ~21.7 M samples   (N_OUTER=200, M_INNER=50  =>  M=10,000 paths per t_k)

Inputs
------
  X_disc   (N, 20)  P(0, T_j) at T_PAY = [0.5, 1.0, ..., 10.0]
  X_scalar (N,  3)  [a, sigma, t_k]

Output
------
  data/train_2.h5  keys: X_disc, X_scalar, y_price
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hw_utils import (
    T_MONITOR, T_PAY, TAU, SUB_STEPS, MATURITY,
    compute_irs_pv, hw_theta, load_all_svensson_params,
    market_discount, par_swap_rate, simulate_hw_paths, svensson_forward,
)

CODE_DIR  = os.path.dirname(os.path.abspath(__file__))
FEDS_FILE = os.path.join(CODE_DIR, "feds200628.csv")

A_GRID     = np.linspace(0.01, 0.30, 8)
SIGMA_GRID = np.linspace(0.005, 0.030, 8)
N_A        = len(A_GRID)
N_SIG      = len(SIGMA_GRID)
N_MON      = len(T_MONITOR)
CHUNK      = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model 2 data generation — EPE labels")
    p.add_argument("--feds",    type=str, default=FEDS_FILE)
    p.add_argument("--out",     type=str,
                   default=os.path.join(CODE_DIR, "data", "train_2.h5"))
    p.add_argument("--n_outer", type=int, default=200,
                   help="Outer Euler-Maruyama paths per (FEDS, a, sigma)")
    p.add_argument("--m_inner", type=int, default=50,
                   help="Inner MC replicates per outer path")
    p.add_argument("--seed",    type=int, default=2)
    return p.parse_args()


def outer_simulation(
    r0:      float,
    p:       dict,
    a:       float,
    sigma:   float,
    n_outer: int,
    rng:     np.random.Generator,
) -> np.ndarray:
    """
    Simulate n_outer Euler-Maruyama paths from t=0 to t=10.
    Returns r_at_mon: shape (n_outer, 21) — short rate at each T_MONITOR date.
    """
    dt      = TAU / SUB_STEPS
    t_max   = float(T_MONITOR[-1])
    grid    = np.arange(0.0, t_max + dt * 0.1, dt)
    n_steps = len(grid) - 1

    mon_to_step = {k: int(np.round(t / dt)) for k, t in enumerate(T_MONITOR)}
    step_to_mon = {v: k for k, v in mon_to_step.items()}

    theta_vec = hw_theta(grid[:-1], p, a, sigma)
    r         = np.full(n_outer, r0, dtype=float)
    r_at_mon  = np.zeros((n_outer, N_MON), dtype=float)
    r_at_mon[:, 0] = r0
    sqrt_dt   = np.sqrt(dt)

    for i in range(n_steps):
        r = r + (theta_vec[i] - a * r) * dt + sigma * sqrt_dt * rng.standard_normal(n_outer)
        if (i + 1) in step_to_mon:
            r_at_mon[:, step_to_mon[i + 1]] = r

    return r_at_mon


def main() -> None:
    args  = parse_args()
    rng   = np.random.default_rng(args.seed)
    M     = args.n_outer * args.m_inner
    BATCH = N_A * N_SIG * N_MON

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    all_params = load_all_svensson_params(args.feds)
    n_feds     = len(all_params)
    total      = n_feds * BATCH

    print(f"FEDS curves   : {n_feds:,}")
    print(f"a  values     : {N_A}   ({A_GRID[0]:.3f} to {A_GRID[-1]:.3f})")
    print(f"sig values    : {N_SIG}   ({SIGMA_GRID[0]:.4f} to {SIGMA_GRID[-1]:.4f})")
    print(f"t_k dates     : {N_MON}  (0.0 to 10.0 semi-annual)")
    print(f"n_outer       : {args.n_outer}   m_inner={args.m_inner}   M={M:,}")
    print(f"Total samples : {total:,}")
    print(f"Output        : {args.out}")

    with h5py.File(args.out, "w") as hf:
        ds_disc   = hf.create_dataset("X_disc",   shape=(0, 20),
                                       maxshape=(None, 20), dtype="float32",
                                       chunks=(CHUNK, 20), compression="gzip")
        ds_scalar = hf.create_dataset("X_scalar", shape=(0, 3),
                                       maxshape=(None, 3),  dtype="float32",
                                       chunks=(CHUNK, 3),  compression="gzip")
        ds_price  = hf.create_dataset("y_price",  shape=(0,),
                                       maxshape=(None,),    dtype="float32",
                                       chunks=(CHUNK,),    compression="gzip")

        offset = 0
        t0     = time.time()

        for fi, p in enumerate(all_params):
            r0      = float(svensson_forward(1e-8, p))
            K       = par_swap_rate(p)
            d_knots = market_discount(T_PAY, p).astype(np.float32)   # (20,)

            batch_disc   = []
            batch_scalar = []
            batch_price  = []

            for a in A_GRID:
                for sigma in SIGMA_GRID:
                    r_outer = outer_simulation(r0, p, a, sigma, args.n_outer, rng)

                    for ki, t_k in enumerate(T_MONITOR):
                        if t_k < 1e-8 or t_k >= MATURITY - 1e-8:
                            label = 0.0
                        else:
                            r_tk    = r_outer[:, ki]
                            r_batch = np.repeat(r_tk, args.m_inner)

                            r_at_pay, logD_at_pay, rem = simulate_hw_paths(
                                t_eval=t_k, r_t_batch=r_batch,
                                p=p, a=a, sigma=sigma, rng=rng,
                            )
                            V_paths = compute_irs_pv(
                                t_eval=t_k, r_t_batch=r_batch,
                                r_at_pay=r_at_pay, logD_at_pay=logD_at_pay,
                                rem=rem, K=K, p=p, a=a, sigma=sigma,
                            )
                            V_outer = V_paths.reshape(args.n_outer, args.m_inner).mean(axis=1)
                            label   = float(np.maximum(V_outer, 0.0).mean())

                        batch_disc.append(d_knots)
                        batch_scalar.append(np.array([a, sigma, t_k], dtype=np.float32))
                        batch_price.append(label)

            n = len(batch_price)
            X_disc_b   = np.stack(batch_disc).astype(np.float32)
            X_scalar_b = np.stack(batch_scalar).astype(np.float32)
            y_price_b  = np.array(batch_price, dtype=np.float32)

            ds_disc.resize(offset + n, axis=0)
            ds_scalar.resize(offset + n, axis=0)
            ds_price.resize(offset + n, axis=0)

            ds_disc[offset:offset + n]   = X_disc_b
            ds_scalar[offset:offset + n] = X_scalar_b
            ds_price[offset:offset + n]  = y_price_b

            offset += n

            if (fi + 1) % 200 == 0 or fi == 0 or fi == n_feds - 1:
                elapsed = time.time() - t0
                rate    = (fi + 1) / max(elapsed, 1)
                eta     = (n_feds - fi - 1) / max(rate, 1)
                print(f"  [{fi+1:>5}/{n_feds}]  samples={offset:>12,}  "
                      f"elapsed={elapsed:>6.0f}s  eta={eta:>6.0f}s")

    print(f"\nSaved {offset:,} samples to {args.out}")


if __name__ == "__main__":
    main()
