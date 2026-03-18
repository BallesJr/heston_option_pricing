# HESTON OPTION PRICING

This project extends the classic Black-Scholes framework by implementing the Heston stochastic volatility model. Unlike Black-Scholes, where volatility is constant, Heston models volatility as a mean-reverting stochastic process, leading to a more realistic price distribution. The analytical solution requires numerical integration over a complex characteristic function.

## FEATURES

**Monte Carlo simulation**: Simulates 10000 different paths that the price can follow.

**Analytical price**: Calculates the fair price using the analytical formula, which includes the complex integral.

**Price and volatility visualization**: Generates a visual representation of both price and volatility paths for the first 20 simulations.

**Convergence plot**: Plots the running average to check that the simulation tends to the analytical optimal price.

## RESULTS

The convergence plot shows how the Law of Large Numbers works because with 50000 simulations the average price and the analytical price are almost exactly the same.

_Call price (Monte Carlo)_: $10.3962

_Call price (Analytical)_: $10.3942

_Difference_: $0.0020

_Delta_: 0.6917

## REQUIREMENTS

`numpy, matplotlib, numba, scipy`

## EXECUTION

Execute `python heston_mc.py`
