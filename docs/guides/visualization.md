# Scenario Visualization

The Aethel output engine features a built-in, Plotly-based interactive visualization suite. These plotting methods are available directly as helper methods on the `SimulationResults` class. They generate standard `plotly.graph_objects.Figure` instances, which can be modified, rendered in Jupyter Notebooks, displayed in browsers, or exported directly to HTML.

---

## 1. Visualizer Catalog

The visualization engine organizes outcomes into four primary analytical perspectives:

| Method | Type | Financial Interpretation |
| :--- | :--- | :--- |
| **`plot_fan_chart()`** | Probability Cone | Displays the median trajectory surrounded by $5\%-95\%$ and $25\%-75\%$ confidence bands. Helps evaluate the widening range of uncertainty over long horizons. |
| **`plot_scenario_paths()`** | Spaghetti Chart | Draws a configurable sample of individual, randomized future trajectories alongside the expected mathematical mean. Useful for showcasing path-dependency and volatility. |
| **`plot_yield_curve_evolution()`** | Term Structure | Illustrates how the nominal or real yield curve (tenor vs. average yield) shifts and flattens across milestone years. |
| **`plot_horizon_distribution()`** | Risk Histogram | Compiles a relative density histogram at a specific future year. Automatically calculates and displays **Value-at-Risk (VaR)** and **Tail Value-at-Risk (TVaR)**. |

---

## 2. API Method Signatures

### Fan Chart
Generates a confidence-band projection for any queryable metric:
```python
def plot_fan_chart(
    self,
    metric: str,
    tenor: Optional[float] = None,
    annualized: bool = False,
    title: Optional[str] = None
) -> plotly.graph_objects.Figure:
```

### Scenario Paths
Displays a subset of raw paths:
```python
def plot_scenario_paths(
    self,
    metric: str,
    num_paths: int = 15,
    tenor: Optional[float] = None,
    annualized: bool = False,
    title: Optional[str] = None
) -> plotly.graph_objects.Figure:
```

### Yield Curve Evolution
Tracks term structure shifts across specific time milestones:
```python
def plot_yield_curve_evolution(
    self,
    years_milestones: List[float] = [0.0, 1.0, 5.0, 15.0, 30.0],
    real: bool = False,
    title: Optional[str] = None
) -> plotly.graph_objects.Figure:
```

### Horizon Distribution
Plots a distribution density profile with tail-risk metrics:
```python
def plot_horizon_distribution(
    self,
    metric: str,
    target_year: float,
    tenor: Optional[float] = None,
    annualized: bool = False,
    bins: int = 40,
    title: Optional[str] = None
) -> plotly.graph_objects.Figure:
```

---

## 3. Practical Implementation Example

The script below demonstrates how to initialize a simulation, generate all four chart types, and export them as self-contained interactive HTML widgets.

```python
import numpy as np
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

# 1. Execute a baseline economic simulation
config = SimulatorConfig(duration_years=30, num_scenarios=1000, seed=42)
simulator = MarketSimulator(config)
results = SimulationResults(simulator.run())

# =====================================================================
# CHART 1: CPI Fan Chart
# =====================================================================
# Displays inflation compounding over time with confidence bands.
fig_fan = results.plot_fan_chart(
    metric="cpi", 
    title="Cumulative Inflation Projection (CPI Index)"
)
fig_fan.write_html("chart_cpi_fan.html")

# =====================================================================
# CHART 2: Short Rate Paths
# =====================================================================
# Displays 20 randomized interest rate paths and their expected mean.
fig_paths = results.plot_scenario_paths(
    metric="short_rate", 
    num_paths=20, 
    title="Stochastic Interest Rate Trajectories"
)
fig_paths.write_html("chart_short_rate_paths.html")

# =====================================================================
# CHART 3: Yield Curve Evolution (Nominal)
# =====================================================================
# Compiles average nominal term structure curves at Years 0, 5, 15, and 30.
fig_yields = results.plot_yield_curve_evolution(
    years_milestones=[0.0, 5.0, 15.0, 30.0],
    real=False,
    title="Nominal Yield Curve Term Structure Evolution"
)
fig_yields.write_html("chart_yield_curves.html")

# =====================================================================
# CHART 4: Equity Growth Outcome Distribution (Tail-Risk)
# =====================================================================
# Analyzes cumulative equity returns of $1 at Year 30.
# The visualizer automatically detects downside risk metrics for asset growth,
# calculating 95% VaR and 95% TVaR (expected tail loss).
fig_dist = results.plot_horizon_distribution(
    metric="equity_growth",
    target_year=30.0,
    bins=50,
    title="Distribution of Equity Growth Outcomes at Year 30"
)
fig_dist.write_html("chart_equity_distribution_y30.html")

# =====================================================================
# CHART 5: Decumulation Portfolio Value Distribution
# =====================================================================
# First, run a decumulation sequence to populate decumulation caches
results.simulate_decumulation(
    initial_balance=1000000.0,
    initial_monthly_withdrawal=4000.0,
    portfolio_weights={"equity": 0.60, "fixed_income": 0.40}
)

# Plot the distribution of remaining balances at Year 15
fig_decum_dist = results.plot_horizon_distribution(
    metric="decumulation_balance",
    target_year=15.0,
    bins=45,
    title="Remaining Retirement Balance Outcomes at Year 15"
)
fig_decum_dist.write_html("chart_decumulation_y15.html")

print("All charts have been exported successfully as interactive HTML pages.")
```

---

## 4. Customizing Graph Layouts

Since the visualizer helper methods return standard `plotly.graph_objects.Figure` containers, you can alter styles, colors, axes, and legends before rendering or exporting using Plotly's standard updates API:

```python
# Customize an existing figure's aesthetic parameters
fig_fan.update_layout(
    template="plotly_dark",  # Switch to dark mode
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

# Display inside an active browser session or Jupyter cell
fig_fan.show()
```
