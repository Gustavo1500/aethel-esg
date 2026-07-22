# 🌀 Aethel: Economic Scenario Generator

Aethel is an actuarial-grade Economic Scenario Generator (ESG) engine written in Python. This repository contains the open-source core library, which implements high-performance stochastic asset-liability and macroeconomic models.

While the mathematical engine is open-source here for local modeling, we also provide a cloud-hosted, multi-tenant enterprise API. The SaaS infrastructure features automated usage tracking, global regional presets, and persistent state management.

*   **Live Interactive Demo:** [View the Interactive ESG Dashboard](https://aethel-esg.vercel.app/) *(Hosted directly via this repo's `index.html`)*
*   **Request API Access:** Get a cloud API key at [Tally Key Request Form](https://tally.so/r/eqj5Eq).

---

## Core Capabilities

Aethel generates correlated, multi-path stochastic simulations of macroeconomic indicators and asset classes to support:

*   **Asset-Liability Management (ALM)** and solvency evaluations.
*   **Retirement Decumulation Profiling** (Sequence of Returns Risk modeling).
*   **Dynamic Portfolio Projections** under varying inflationary and interest rate regimes.
*   **Yield Curve Term-Structure Forecasting** (Nominal and Real).

### The Three-Factor Stochastic Framework
The underlying mathematical engine models the economy using a system of continuous-time stochastic processes:
1.  **Equities (Merton Jump-Diffusion)**: Captures log-return growth dynamics combined with asymmetric Poisson-driven market shocks (jumps) to represent systemic downturns.
2.  **Short Rates (Cox-Ingersoll-Ross / CIR)**: Models mean-reverting nominal short-term interest rates with a square-root diffusion term to preserve mathematical consistency.
3.  **Inflation (Ornstein-Uhlenbeck / Shifted-CIR)**: Simulates mean-reverting consumer price dynamics with feedback loops representing central bank monetary policy adjustments.

---

## ⚙️ Installation

The core open-source library can be installed locally to execute simulations.

### Prerequisites
*   Python 3.9 or higher

### Local Setup
Clone the repository and install the local package:

```bash
# Clone the repository
git clone https://github.com/gustavo1500/aethel-esg.git

# Install the package locally with visualization and optimization dependencies
cd aethel-esg

# Install core library along with Numba for optimization, and Plotly for charts
pip install -e ."[numba,plots]"
```

---

## ⚡ Quickstart Code Examples

### 1. Calibrate Parameters from Historical Data
Before running a simulation, the engine parameters are typically calibrated against historical data. This step estimates continuous and jump-diffusion parameters from historical records using Maximum Likelihood Estimation (MLE) and OLS.

```python
import numpy as np
from aethel import MarketCalibrator

# 1. Generate or load historical monthly observations (minimum 10 months required)
np.random.seed(42)
history_length_months = 120

historical_inflation = np.random.normal(0.03, 0.01, history_length_months)
historical_rates = np.random.normal(0.04, 0.015, history_length_months)
historical_equity_returns = np.random.normal(0.008, 0.05, history_length_months)

# 2. Fit the ESG models to obtain calibrated parameters
calibrator = MarketCalibrator()
calibrated_config = calibrator.fit(
    historical_inflation=historical_inflation,
    historical_rates=historical_rates,
    historical_equity_returns=historical_equity_returns
)

# 3. Configure the projection horizon and scenario count on the calibrated config
calibrated_config.duration_years = 10
calibrated_config.num_scenarios = 250
calibrated_config.seed = 123

print(f"Calibrated Equity Volatility (GBM): {calibrated_config.gbm_sigma_val * 100:.2f}%")
print(f"Calibrated Inflation Target (OU): {calibrated_config.ou_mu * 100:.2f}%")
```

### 2. Run an Economic Simulation
Initialize the simulator with the calibrated configuration (or a customized configuration) to extract simulated economic trajectories.

```python
from aethel import MarketSimulator, SimulationResults

# 1. Run simulation engine using the calibrated configuration from Step 1
simulator = MarketSimulator(calibrated_config)
scenarios = simulator.run()

# 2. Compile and analyze statistical paths
results = SimulationResults(scenarios)
median_equity_growth = results.query("equity_growth", stat="median", year=10.0)

print(f"Median compounding factor of $1.00 after 10 years: ${median_equity_growth:.2f}")
```

### 3. Simulate Retirement Portfolio Decumulation
Evaluate retirement nest-egg survival rates under path-dependent sequences of returns using simulated outcomes.

```python
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

# Initialize a standard baseline configuration
config = SimulatorConfig(duration_years=30, num_scenarios=1000, seed=42)
simulator = MarketSimulator(config)
results = SimulationResults(simulator.run())

# Simulate a 4% rule decumulation with 15% gains tax, 25 bps fee drag, and cash-first liquidation
decum_results = results.simulate_decumulation(
    initial_balance=1_000_000.0,
    initial_monthly_withdrawal=3333.33,  # ~$40,000 annually
    portfolio_weights={"equity": 0.60, "fixed_income": 0.40},
    liquidation_strategy="cash_first",
    frictional_drag_annual=0.0025,
    tax_on_gains_rate=0.15
)

final_solvency = decum_results["probability_of_success"][-1]
print(f"Portfolio Solvency Probability at Year 30: {final_solvency * 100:.1f}%")
```

---

## 📁 Repository Structure

```text
actuarial_esg/
├── pyproject.toml                   # Python package metadata & dependencies
├── requirements.txt                 # Minimum requirement specifications
├── presets/
│   └── usa.json                     # Pre-calibrated United States macroeconomic baseline parameters
├── demo/
│   ├── logo.png                     # Logo branding asset
│   ├── demo_database.json           # Compact pre-calculated scenario database
│   └── generate_demo_cache.py       # Pre-calculates and compiles demo_database.json
│   └── index.html                   # The frontend of the interactive demo

├── examples/                        # Specialized mathematical modeling templates
│   ├── basic_simulation.py          # Quickstart run of economic projections
│   ├── run_calibration.py           # Calibration & parameters export from historical CSVs
│   ├── portfolio_decumulation_demo.py # Compares constant-mix vs cash-first with guardrails
│   ├── fire_integration.py          # Calibrates and outputs dataframes to support FIRE models
│   ├── run_chunked_simulation.py    # Memory-safe execution of high-horizon simulations
│   └── visualize_scenarios.py       # Interactive Plotly dashboard assembly
├── src/
│   └── aethel/                      # Packaged Python library source code
│       ├── calibration/             # Parameter estimation logic (OLS, MLE solvers)
│       ├── engine/                  # Core stochastic processes & simulation loops
│       └── output/                  # Downstream metrics, portfolio, & decumulation engines
└── tests/
    └── test_engine.py               # Core regression & boundary condition mathematical assertions
    └── test_robustness.py           # Robustness test of the project - both tests have 88% project coverage
```

---

## Developer Performance Optimizations

The Aethel library includes options for memory management and performance scaling:

*   **Memory-Safe Lazy Evaluation**: Generating term structures (nominal and real yields) for multiple maturities across thousands of scenarios is memory-intensive. The engine uses a `LazyScenarioList` to keep primary arrays contiguous in memory, deriving yield curves dynamically only when accessed.
*   **Hardware-Aware Concurrency**: The simulation detects host RAM and CPU capacity to automatically partition simulations into memory-safe blocks, avoiding concurrent page-fault issues in shared environments.
*   **Fastmath Compilations**: If `numba` is detected, the engine runs JIT-compiled loops with parallel execution capabilities.

---

## 💻 SaaS Cloud Infrastructure

The enterprise cloud platform wraps the Aethel engine in a production API:

*   **Application Server**: Deployed as a scalable FastAPI service hosted on **Render**.
*   **Database Engine**: Backed by **Neon (PostgreSQL)**, utilizing a serverless architecture to manage user state, transactional API usage metrics, and cryptographically hashed API keys (SHA-256).
*   **Hosted Calibration**: The enterprise API handles raw historical CSV uploads, running mathematical convergence checks on cloud hardware to output customized JSON presets.

### Requesting API Access
API access keys are available for testing and commercial production. You can request a key and view subscription tiers through our application intake form:

👉 **[Tally API Key Request Form](https://tally.so/r/eqj5Eq)**

---

## License

The Aethel engine open-sourced here is released under the [MIT License](https://opensource.org/licenses/MIT). All SaaS architecture, cloud multi-tenant database models, and deployment configurations remain proprietary.
