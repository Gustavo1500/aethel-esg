# Parameter Calibration Guide

Before starting an economic projection, the Aethel engine parameters should be calibrated against historical series. The calibrator uses Maximum Likelihood Estimation (MLE) and Ordinary Least Squares (OLS) to estimate parameters.

## Calibration Process Flow

```text
[Historical CSV Data]
         │
         ├──► Inflation Series ──────► [InflationCalibrator] ──┐
         ├──► Short Rate Series ─────► [RatesCalibrator] ──────┼──► [SimulatorConfig]
         └──► Equity Return Series ──► [EquityCalibrator] ─────┘
```

---

## Data Frequency & Scaling Requirements

To obtain mathematically stable and realistic parameters, the historical data input must strictly adhere to the following properties:

1. **Monthly Observations:** Input series must be sampled on a monthly frequency. The underlying parameter derivations scale the transition matrix assuming a constant time step of $dt = 1/12$.
2. **Annualized Interest & Inflation Rates:** Rates should be represented in decimal annualized format (e.g., $4.5\%$ nominal annualized short rate must be input as `0.045`).
3. **Simple Monthly Equity Returns:** Equity return series must be periodic monthly returns (not annualized). For example, a $1.2\%$ monthly gain is input as `0.012`.
4. **Minimum Observations Constraint:** The Merton Jump-Diffusion MLE optimization requires a minimum of **10 monthly historical return observations** to initialize and avoid solver divergence. Passing fewer observations will trigger a `ValueError`.

---

## Two Calibration Methodologies

The `MarketCalibrator` provides two approaches to parameterizing the Cox-Ingersoll-Ross (CIR) interest rate model:

### 1. Time-Series Estimation (Default)
When provided only with historical short rates, the engine uses discrete-time regression analysis over successive transitions to solve for mean reversion speed ($\theta_r$), long-term target rate ($\mu_r$), and volatility ($\sigma_r$).

### 2. Cross-Sectional Yield Curve Fitting (Advanced)
If you provide an observed historical yield curve term structure and its corresponding tenors, the engine bypasses time-series estimation. Instead, it matches analytical CIR term structure formulas to the target yield curve across all specified maturities using numerical optimization (L-BFGS-B). This ensures that the generated yield curves remain consistent with current or historical interest rate environments.

---

## Step-by-Step Code Example

The code snippet below demonstrates how to configure and execute both the default time-series calibration and the advanced cross-sectional yield curve fitting, followed by exporting the results to a JSON preset.

```python
import numpy as np
import pandas as pd
import json
from aethel import MarketCalibrator

# 1. Load historical data
# Expected columns: 'inflation', 'short_rate', 'equity_return'
df = pd.read_csv("historical_market_data.csv")

# Ensure the minimum data length constraint is satisfied
if len(df) < 10:
    raise ValueError("Historical data must contain at least 10 monthly observations.")

# 2. Instantiate the unified calibrator
calibrator = MarketCalibrator()

# --- METHOD A: Default Time-Series Calibration ---
print("Fitting Aethel ESG models using time-series transitions...")
calibrated_config = calibrator.fit(
    historical_inflation=df["inflation"].values,
    historical_rates=df["short_rate"].values,
    historical_equity_returns=df["equity_return"].values
)

# --- METHOD B: Cross-Sectional Yield Curve Fitting (Optional) ---
# Suppose you have a target yield curve observed at the final historical data point:
# Maturity tenors: 3-month, 1-year, 5-year, 10-year
target_tenors = np.array([0.25, 1.0, 5.0, 10.0])
# Observed yields for those tenors: 4.2%, 4.5%, 4.8%, 5.1%
target_yield_curve = np.array([0.042, 0.045, 0.048, 0.051])

print("Fitting interest rate parameters directly to target yield curve...")
calibrated_config_curve = calibrator.fit(
    historical_inflation=df["inflation"].values,
    historical_rates=df["short_rate"].values,
    historical_equity_returns=df["equity_return"].values,
    historical_yield_curve=target_yield_curve,  # Maps directly to rates solver
    tenors=target_tenors
)

# 3. Configure projection parameters on the preferred configuration
calibrated_config_curve.duration_years = 30
calibrated_config_curve.num_scenarios = 1000
calibrated_config_curve.seed = 42

# 4. Save config dictionary to a JSON file for deployment
config_dict = calibrated_config_curve.to_dict()
with open("calibrated_world_preset.json", "w") as f:
    json.dump(config_dict, f, indent=4)

print(f"Calibrated Equity Volatility (Continuous): {calibrated_config_curve.gbm_sigma_val * 100:.2f}%")
print(f"Calibrated Inflation Target (OU Mean): {calibrated_config_curve.ou_mu * 100:.2f}%")
print(f"Calibrated CIR Long Rate Target (Mean):  {calibrated_config_curve.cir_mu_val * 100:.2f}%")
```
