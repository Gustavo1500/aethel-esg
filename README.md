# Actuarial ESG: Economic Scenario Generator

An actuarial-grade Economic Scenario Generator (ESG) engine written in Python. This repository contains the open-source core library (`actuarial_esg`), which implements high-performance stochastic asset-liability and macroeconomic models.

While the mathematical engine is open-source here for local modeling, we also provide a cloud-hosted, multi-tenant enterprise API. The SaaS infrastructure features automated usage tracking, global regional presets, and persistent state management.

*   **Live Interactive Demo:** [View the Interactive ESG Dashboard](https://aethel-api.onrender.com) *(Hosted directly via this repo's `index.html`)*
*   **Request API Access:** Get a cloud API key at [Tally Key Request Form](https://tally.so/r/eqj5Eq).

---

## Core Capabilities

Actuarial ESG generates correlated, multi-path stochastic simulations of macroeconomic indicators and asset classes to support:

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

## Installation

You can install the core open-source library locally to execute simulations on your own hardware.

### Prerequisites
*   Python 3.9 or higher

### Local Setup
Clone the repository and install the local package:

```bash
# Navigate to the open-source engine directory
cd actuarial_esg

# Install core library along with Numba for optimization, and plots for Plotly charts
pip install -e ."[numba,plots]"
```

---

## Quickstart Code Examples

### 1. Run a Basic Economic Simulation
This script initializes the simulator with nominal interest rate and inflation starting values and extracts the resulting trajectories.

```python
import numpy as np
from actuarial_esg import SimulatorConfig, MarketSimulator, SimulationResults

# 1. Define macroeconomic starting states and horizon
config = SimulatorConfig(
    duration_years=10,       # 10-year projection horizon
    num_scenarios=250,       # 250 simulated futures
    seed=123,                # Seed for reproducible outputs
    initial_cdi=0.08,        # Start with risk-free rate at 8.0%
    initial_ipca=0.045       # Start with YoY inflation at 4.5%
)

# 2. Run simulation engine
simulator = MarketSimulator(config)
scenarios = simulator.run()

# 3. Compile and analyze statistical paths
results = SimulationResults(scenarios)
median_equity_growth = results.query("equity_growth", stat="median", year=10.0)

print(f"Median compounding factor of $1.00 after 10 years: ${median_equity_growth:.2f}")
```

### 2. Simulate Retirement Portfolio Decumulation
Evaluate retirement nest-egg survival rates under path-dependent sequences of returns.

```python
from actuarial_esg import SimulatorConfig, MarketSimulator, SimulationResults

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

## Repository Structure

```text
actuarial_esg/
├── index.html                       # Tailwind/Plotly SPA for GitHub Pages interactive demo
├── build_presets.py                 # Fetches macroeconomic historical data to calibrate regional files
├── pyproject.toml                   # Python package metadata & dependencies
├── requirements.txt                 # Minimum requirement specifications
├── presets/
│   └── usa.json                     # Pre-calibrated United States macroeconomic baseline parameters
├── demo/
│   ├── logo.png                     # Logo branding asset
│   ├── demo_database.json           # Compact pre-calculated scenario database used by index.html
│   └── generate_demo_cache.py       # Combinatorial script to pre-calculate and compile demo_database.json
├── examples/                        # Specialized mathematical modeling templates
│   ├── basic_simulation.py          # Quickstart run of economic projections
│   ├── run_calibration.py           # Calibration & parameters export from historical CSVs
│   ├── compare_region.py            # Compares retirement solvency curve across global configurations
│   ├── portfolio_decumulation_demo.py # Compares constant-mix vs cash-first with guardrails
│   ├── fire_integration.py          # Calibrates and outputs dataframes to support FIRE models
│   ├── run_chunked_simulation.py    # Memory-safe execution of high-horizon simulations
│   └── visualize_scenarios.py       # Interactive Plotly dashboard assembly
├── src/
│   └── actuarial_esg/               # Packaged Python library source code
│       ├── calibration/             # Parameter estimation logic (OLS, MLE solvers)
│       ├── engine/                  # Core stochastic processes & simulation loops
│       └── output/                  # Downstream metrics, portfolio, & decumulation engines
└── tests/
    └── test.py                      # Core regression & boundary condition mathematical assertions
```

---

## Developer Performance Optimizations

The open-source core includes advanced memory-management and CPU-optimization paradigms:

*   **Memory-Safe Lazy Evaluation**: Generating term structures (nominal and real yields) for multiple maturities across thousands of scenarios is memory-intensive. The engine uses a `LazyScenarioList` to keep primary arrays contiguous in memory, deriving yield curves dynamically only when accessed.
*   **Hardware-Aware Concurrency**: The simulation detects host RAM and CPU capacity to automatically partition simulations into memory-safe blocks, avoiding concurrent page-fault issues in shared environments.
*   **Fastmath Compilations**: If `numba` is detected, the engine runs fully JIT-compiled loops with parallel execution capabilities.

---

## SaaS Cloud Infrastructure

The closed-source enterprise cloud platform wraps this engine in a production API:

*   **Application Server**: Deployed as a scalable FastAPI service hosted on **Render**.
*   **Database Engine**: Backed by **Neon (PostgreSQL)**, utilizing a serverless architecture to manage user state, transactional API usage metrics, and cryptographically hashed API keys (SHA-256).
*   **Hosted Calibration**: The enterprise API handles raw historical CSV uploads, running mathematical convergence checks on cloud hardware to output customized JSON presets.

### Requesting API Access
API access keys are available for testing and commercial production. You can request a key and view subscription tiers through our application intake form:

👉 **[Tally API Key Request Form](https://tally.so/r/eqj5Eq)**

---

## License

The engine open-sourced here is released under the [MIT License](https://opensource.org/licenses/MIT). All SaaS architecture, cloud multi-tenant database models, and deployment configurations remain proprietary.
