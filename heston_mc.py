import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import quad
import time

# Model parameters
S0 = 100 # Initial price
v0 = 0.04 # Initial variance
kappa = 2.0 # Mean reversion speed
theta = 0.04 # Long-term variance
xi = 0.3 # Volatility of variance
rho = -0.7 # Correlation between price and variance
r = 0.05 # Risk-free interest rate
T = 1.0 # Time to maturity (1 year)
K = 100 # Strike price

# Simulation parameters
steps = 252 # Daily steps
simulations = 10000
dt = T/steps

# Correlated Brownian motions
np.random.seed(42)

Z1 = np.random.standard_normal((steps, simulations)) # Brownian for variance
Z2 = np.random.standard_normal((steps, simulations)) # Independent Brownian
Z_S = rho * Z1 + np.sqrt(1 - rho**2) * Z2 # Correlated Brownian for price

# Initialize paths
S = np.zeros((steps + 1, simulations))
v = np.zeros((steps + 1, simulations))
S[0] = S0
v[0] = v0

# Heston simulations - truncation scheme to avoid negative variance
@njit(cache=True)
def simulate_heston(S, v, Z1, Z_S, kappa, theta, xi, r, dt, steps):
    for t in range(1, steps + 1):
        v_pos = np.maximum(v[t-1], 0)
        v[t] = v[t-1] + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos * dt) * Z1[t-1]
        S[t] = S[t-1] * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z_S[t-1])
    return S, v

# Run simulation 
start = time.time()
S, v = simulate_heston(S, v, Z1, Z_S, kappa, theta, xi, r, dt , steps)
print(f"Simulation time: {time.time() - start:.2f}s")

# European call option pricing - discounted expected payoff
payoffs = np.maximum(S[-1] - K, 0) # Payoff at maturity (max(S_T - K, 0))
call_price = np.exp(-r * T) * np.mean(payoffs) # Discounted expected payoff (E[payoff] * e^(-rT))

# Analytical Heston valoration
# Heston characteristic function
def heston_characteristic(phi, S0, v0, kappa, theta, xi, rho, r, T, j):
    i = complex(0, 1)

    if j == 1:
        u = 0.5
        b = kappa - rho * xi
    else:
        u = -0.5
        b = kappa
    
    a = kappa * theta
    d = np.sqrt((rho *  xi * i * phi -b)**2 - xi**2 * (2 * u * i * phi - phi**2))
    g = (b - rho * xi * i * phi + d) / (b - rho * xi * i * phi - d)

    C = r * i * phi * T + (a / xi**2) * ((b - rho * xi * i * phi + d) *  T - 2 * np.log((1 - g * np.exp(d * T)) / (1 - g)))
    D = ((b - rho * xi * i * phi + d) / xi**2) * ((1 - np.exp(d * T)) / (1 - g * np.exp(d * T)))

    return np.exp(C + D * v0 + i * phi * np.log(S0))

# Heston probabilities via numerical integration
def heston_P(j, S0, v0, kappa, theta, xi, rho, r, T, K):
    integrand = lambda phi: np.real(
        np.exp(-complex(0,1) * phi * np.log(K)) *
        heston_characteristic(phi, S0, v0, kappa, theta, xi, rho, r, T, j) /
        (complex(0, 1) * phi)
    )
    integral, _ = quad(integrand, 1e-6, 100)
    return 0.5 + integral / np.pi

# Analytical call price via Heston formula
def heston_call_analytical(S0, v0, kappa, theta, xi, rho, r, T, K):
    P1 = heston_P(1, S0, v0, kappa, theta, xi, rho, r, T, K)
    P2 = heston_P(2, S0, v0, kappa, theta, xi, rho, r, T, K)
    return S0 * P1 - K * np.exp(-r * T) * P2

# Analytical call price
analytical_price = heston_call_analytical(S0, v0, kappa, theta, xi, rho, r, T, K)

# Plot convergence
def plot_convergence(S0, v0, kappa, theta, xi, rho, r, T, K, steps, total_sims, analytical, batch_size=1000):
    iterations = np.arange(batch_size, total_sims + 1, batch_size)
    running_prices = []
    running_sum = 0

    print("Computing convergence plot...")
    for i in range(len(iterations)):
        # Simulate a batch
        Z1_b = np.random.standard_normal((steps, batch_size))
        Z2_b = np.random.standard_normal((steps, batch_size))
        Z_S_b = rho * Z1_b + np.sqrt(1 - rho**2) * Z2_b
    
        S_b = np.zeros((steps + 1, batch_size))
        v_b = np.zeros((steps + 1, batch_size))
        S_b[0] = S0
        v_b[0] = v0

        S_b, v_b = simulate_heston(S_b, v_b, Z1_b, Z_S_b, kappa, theta, xi, r, dt, steps) 

        payoffs_b = np.maximum(S_b[-1] - K, 0)
        price_b = np.exp(-r * T) * np.mean(payoffs_b)

        running_sum += price_b
        running_prices.append(running_sum / (i + 1))


    plt.figure(figsize=(12, 6))
    plt.plot(iterations, running_prices, label="Monte Carlo (Running Average)", color="#1f77b4", lw=2)
    plt.axhline(y=analytical, color="#d62728", linestyle="--", label=f"Analytical Price (${analytical:.4f})")
    plt.title("Monte Carlo Convergence vs Analytical Price (Heston Call)", fontsize=14)
    plt.xlabel("Number of simulations")
    plt.ylabel("Option price($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Delta calculation via finite differences (central difference scheme)
epsilon = S0 * 0.01 # 1% of S0

price_up = heston_call_analytical(S0 + epsilon, v0, kappa, theta, xi, rho, r, T, K)
price_down = heston_call_analytical(S0 - epsilon, v0, kappa, theta, xi, rho, r, T, K)

delta = (price_up - price_down) / (2 * epsilon)

# Results
print(f"\nHeston Model Results")
print("*" * 35)
print(f"Call Price (Monte Carlo): ${call_price:.4f}")
print(f"Call Price (Analytical): ${analytical_price:.4f}")
print(f"Difference: ${abs(call_price - analytical_price):.4f}")
print(f"Delta: {delta:.4f}")

# Visualize paths
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), sharex= True)

# Price paths 
ax1.plot(S[:, :20], alpha=0.5, linewidth=0.8)
ax1.set_title("Heston Model - Asset Price Paths (20 simulations)")
ax1.set_ylabel("Price ($)")
ax1.axhline(S0, color="black", linestyle="--", linewidth=1.5, label="Initial Price")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Variance paths
ax2.plot(np.sqrt(np.maximum(v[:, :20],0)), alpha=0.5, linewidth=0.8)
ax2.set_title("Heston Model - Volatility Paths (20 simulations)")
ax2.set_ylabel("Volatility")
ax2.axhline(np.sqrt(theta), color="black", linestyle="--", linewidth=1.5, label="Long-term Vol")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlabel("Days")

plt.tight_layout()
plt.show()

plot_convergence(S0, v0, kappa, theta, xi, rho, r, T, K, steps, 50000, analytical_price)
