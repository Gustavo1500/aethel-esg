# Aethel Economic Scenario Generator (ESG)

Aethel is an actuarial-grade Economic Scenario Generator (ESG) engine written in Python. It implements high-performance continuous-time stochastic processes to generate correlated multi-path simulations of core macroeconomic indicators and asset classes.

*   **Live Interactive Demo:** [View the Interactive ESG Dashboard](https://aethel-esg.vercel.app/) *(Hosted via the Aethel simulation cache)*

---

## Primary Applications

Aethel is optimized for modeling long-term economic trajectories, helping risk managers and financial planners conduct:

*   **Asset-Liability Management (ALM)** and solvency evaluations.
*   **Retirement Decumulation Profiling** (Sequence of Returns Risk modeling).
*   **Dynamic Portfolio Projections** under varying inflationary and interest rate regimes.
*   **Yield Curve Term-Structure Forecasting** (Nominal and Real).

---

## Aethel Cloud API (SaaS Platform)

While the mathematical modeling engine can be executed locally, we offer a fully managed **Multi-Tenant SaaS API** designed to transition stochastic simulations from local scripts to scalable, cloud-hosted architecture. 

The API layer wraps the Aethel core engine in a production-grade Web Service built with **FastAPI**, hosted on **Render**, and backed by a **Neon (PostgreSQL)** database for persistent tenant, billing, and secure api-key management.

### Cloud-Only Features & Enterprise Capabilities

*   **Premium Pre-Calibrated Presets:** Run simulations instantly using built-in, historically calibrated macroeconomic baselines for major global regions, including **USA**, **Europe**, **Japan**, **Brazil**, and **World**.
*   **Expert Actuarial Boundaries:** Regional presets include structural overrides (such as deflation floors and nominal rate boundaries) designed to preserve simulation stability in extreme interest rate regimes.
*   **Enterprise Calibration Lifecycle:** Upload raw historical CSV datasets containing interest rates, inflation, and equity returns directly via the `/api/v1/calibrate` endpoint to fit parameters on cloud hardware and save them as permanent, custom presets.
*   **Resource & Quota Management:** Compute usage is tracked automatically using a robust **Compute Unit** billing metric, derived as:

    $$
    \text{Compute Units} = \text{Scenarios} \times \text{Projection Years} \times 12
    $$

*   **Tenant Access Control:** Subscription tiers (`Free`, `Professional`, `Enterprise`) enforce API rate limits and monthly cumulative compute budgets. Authentication is secured using crytographically hashed SHA-256 API keys.

> **[Launch Live Interactive Dashboard](https://aethel-esg.vercel.app/)** | **[Request API Access via Tally](https://tally.so/r/eqj5Eq)**

---

## High-Performance Architectural Features

To manage the high memory demands of multi-scenario simulations, the engine implements several performance optimizations:

*   **Hardware-Aware Concurrency:** The simulation detects available host RAM and CPU capacity to automatically partition simulations into memory-safe blocks, preventing page-fault issues.
*   **Memory-Safe Lazy Evaluation:** Generating term structures (nominal and real yields) for multiple maturities across thousands of scenarios is memory-intensive. The engine uses a `LazyScenarioList` to keep primary arrays contiguous in memory, deriving yield curves dynamically only when accessed.
*   **Fastmath Compilations:** If `numba` is detected, the engine runs JIT-compiled loops with parallel execution capabilities.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Gustavo1500/aethel-esg.git

cd aethel-esg

# Install the package locally with visualization and optimization dependencies
pip install -e ."[numba,plots]"
```
