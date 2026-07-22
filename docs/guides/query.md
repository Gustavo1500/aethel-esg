# Querying Simulation Results

The `SimulationResults` class provides a unified, highly optimized query interface (`SimulationResults.query()`) to extract, slice, transform, and analyze generated scenario data. This guide explains how to use the query engine to retrieve raw scenario paths, calculate statistical moments, and analyze portfolio performance.

---

## 1. Query Method Signature

The `query` method uses the following signature:

```python
def query(
    self,
    metric: str,
    stat: str = "raw",
    time: Optional[Union[str, float, int, List[float]]] = None,
    year: Optional[Union[str, float, int, List[float]]] = None,
    month: Optional[Union[str, int, List[int]]] = None,
    step: Optional[Union[str, int, List[int]]] = None,
    tenor: Optional[float] = None,
    annualized: bool = False,
    portfolio_weights: Optional[Dict[str, float]] = None
) -> Union[np.ndarray, float]:
```

---

## 2. Valid Query Parameters

### Metric Selection (`metric`)
The query engine maps several case-insensitive string aliases to the underlying scenario arrays:

| Category | String Aliases | Required Extra Parameters | Description |
| :--- | :--- | :--- | :--- |
| **Equities** | `"equity_returns"`, `"returns"` | None | Monthly stock returns ($R_{E, t}$) |
| | `"equity_growth"`, `"growth"` | None | Cumulative growth factor of $1.00 base |
| **Inflation** | `"cpi"`, `"inflation_index"` | None | Cumulative CPI index ($1.00 initial base) |
| | `"inflation_rate"`, `"inflation"` | None | Annualized monthly inflation rate ($\pi_t$) |
| **Interest Rates** | `"short_rate"`, `"rate"` | None | Annualized short-term interest rate ($r_t$) |
| | `"nominal_yield"`, `"real_yield"` | `tenor` | Analytical bond yields for a given maturity |
| **Portfolios** | `"portfolio_returns"` | `portfolio_weights` | Weighted composite return of assets |
| | `"portfolio_growth"` | `portfolio_weights` | Compounded growth of the weighted portfolio |
| **Decumulation** | `"decumulation_balance"` | None | Balance trajectory (requires prior simulation) |
| | `"decumulation_withdrawal"` | None | Withdrawal trajectory (requires prior simulation) |

### Statistical Operations (`stat`)
The `stat` parameter collapses the scenario dimension (Dimension 1) into a single path or scalar:

*   **`"raw"` (Default):** Returns the full 2D array of individual scenario paths (Slices are shaped as `[Time, Scenarios]`).
*   **`"mean"`:** Calculates the arithmetic mean across all scenarios at each step.
*   **`"median"`:** Calculates the 50th percentile across all scenarios.
*   **`"std"`, `"vol"`, `"volatility"`:** Calculates the standard deviation of outcomes.
*   **`"p<N>"`:** Calculates an arbitrary percentile where `<N>` is between `0.0` and `100.0` (e.g., `"p5"`, `"p95"`, `"p25"`).

### Time Slicing options
To retrieve data at specific intervals, use **exactly one** of the following mutually exclusive keyword parameters:

*   **`step`:** Integer monthly step index (e.g., `step=120` or `step="all"`).
*   **`month`:** Integer monthly index (synonym for `step`).
*   **`year`:** Float or integer representing the time in years (e.g., `year=10.0` or `year=[1.0, 5.0, 10.0]`).
*   **`time`:** Legacy float alias for `year`.

---

## 3. Query Execution Examples

The following practical code examples show how to run queries on simulation results.

### Example 1: Basic Macroeconomic Extraction
This example demonstrates how to query interest rate and inflation distributions at specific horizons.

```python
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

# 1. Run a 30-year simulation
config = SimulatorConfig(duration_years=30, num_scenarios=1000, seed=42)
simulator = MarketSimulator(config)
results = SimulationResults(simulator.run())

# 2. Query median inflation rate at Year 10
median_inflation_y10 = results.query("inflation", stat="median", year=10.0)
print(f"Median Inflation (Year 10): {median_inflation_y10 * 100:.2f}%")

# 3. Query the 95th percentile of the short-term interest rate at Year 20
rate_p95_y20 = results.query("short_rate", stat="p95", year=20.0)
print(f"95th Percentile Short Rate (Year 20): {rate_p95_y20 * 100:.2f}%")

# 4. Get the complete average nominal yield curve at Year 30
yields_y30 = {
    t: results.query("nominal_yield", stat="mean", year=30.0, tenor=t)
    for t in config.tenors
}
print("Expected Yield Curve at Year 30:")
for tenor, y in yields_y30.items():
    print(f"  - {tenor:5.2f}Y Tenor: {y * 100:.2f}%")
```

---

### Example 2: Analyzing Portfolios and Compounded Annual Growth (CAGR)
This example uses the built-in portfolio-blending logic and annualization transforms to compute CAGRs for custom asset allocations.

```python
# Configure weights for a balanced portfolio (60% Equity, 40% Fixed Income)
weights = {"equity": 0.60, "fixed_income": 0.40}

# 1. Extract the raw compounded growth trajectories of the blended portfolio
raw_growth_paths = results.query(
    "portfolio_growth",
    stat="raw",
    year=15.0,
    portfolio_weights=weights
)
print(f"Compounded Growth shape at Year 15: {raw_growth_paths.shape}") # returns a 1D array of 1000 scenarios

# 2. Extract annualized growth rates directly using the annualized transform
# This automatically converts cumulative growth into CAGR percentages: (Growth ^ (1/Years)) - 1
median_cagr_y15 = results.query(
    "portfolio_growth",
    stat="median",
    year=15.0,
    annualized=True,
    portfolio_weights=weights
)
downside_cagr_y15 = results.query(
    "portfolio_growth",
    stat="p5",
    year=15.0,
    annualized=True,
    portfolio_weights=weights
)

print(f"Median 15-Year Portfolio CAGR:   {median_cagr_y15 * 100:.2f}%")
print(f"Worst-case (P5) Portfolio CAGR:  {downside_cagr_y15 * 100:.2f}%")
```

---

### Example 3: Slicing Decumulation Paths after Simulation
To query decumulation metrics, the decumulation simulator must be executed first to populate the results cache.

```python
# 1. Run decumulation simulation
results.simulate_decumulation(
    initial_balance=1000000.0,
    initial_monthly_withdrawal=4000.0,
    portfolio_weights={"equity": 0.60, "fixed_income": 0.40}
)

# 2. Query median remaining balance trajectory at multiple milestones
milestone_years = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
median_balances = results.query("decumulation_balance", stat="median", year=milestone_years)

print("Median Portfolio Decumulation Path:")
for y, bal in zip(milestone_years, median_balances):
    print(f"  - Year {int(y)}: ${bal:,.2f}")
```
