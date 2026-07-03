import numpy as np

# ==========================
# Model parameters
# ==========================
lambda_hw = 0.03      # mean reversion speed
eta = 0.01            # volatility
r0 = 0.04             # initial short rate

# ==========================
# Time grid
# ==========================
T = 10.0              # total horizon (years)
dt = 0.25             # quarterly steps
times = np.arange(0.0, T + dt, dt)
n_steps = len(times)

# ==========================
# Forward rate curve (flat for simplicity)
# ==========================
def f0(t):
    return 0.04        # constant forward rate

# ==========================
# Exact transition for one step
# ==========================
def next_rate(r_curr, t_curr, t_next, Z):
    dt_step = t_next - t_curr
    exp_neg = np.exp(-lambda_hw * dt_step)
    mean = f0(t_next) + (r_curr - f0(t_curr)) * exp_neg
    std = eta * np.sqrt((1 - np.exp(-2 * lambda_hw * dt_step)) / (2 * lambda_hw))
    return mean + std * Z

# ==========================
# Simulation of 10,000 paths
# ==========================
np.random.seed(42)
n_paths = 10000
paths = np.zeros((n_paths, n_steps))
paths[:, 0] = r0

for i in range(n_steps - 1):
    Z = np.random.normal(0, 1, n_paths)
    for m in range(n_paths):
        paths[m, i+1] = next_rate(paths[m, i], times[i], times[i+1], Z[m])

# ==========================
# (Optional) Quick statistics at each time
# ==========================
mean_rates = np.mean(paths, axis=0)
std_rates = np.std(paths, axis=0)

print("Simulation complete: 10,000 paths over", n_steps, "time points")
print("\nFirst 5 time points (t, mean short rate, std):")
for j in range(min(5, n_steps)):
    print(f"t = {times[j]:.2f}: mean = {mean_rates[j]:.6f}, std = {std_rates[j]:.6f}")