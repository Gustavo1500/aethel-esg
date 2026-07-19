import numpy as np
import pandas as pd
from aethel import MarketSimulator, SimulatorConfig, SimulationResults


def run_basic_simulation():
    print("==================================================")
    print("        Aethel ESG - Basic Simulation             ")
    print("==================================================")

    # 1. Initialize a custom configuration using standardized terms
    config = SimulatorConfig(
        duration_years=10,       # 10-year projection horizon
        num_scenarios=250,       # Run 250 simulated futures
        seed=123,                # Set seed for reproducibility
        initial_rate=0.12,       # Start with short rate (e.g., 12.0%)
        initial_inflation=0.06,  # Start with inflation (e.g., 6.0%)
        lambda_J=0.15,           # Lower jump frequency (15% chance per year)
    )

    print(f"Configured simulation for {config.duration_years} years.")
    print(f"Initial State -> Rate: {config.initial_rate*100:.1f}%, Inflation: {config.initial_inflation*100:.1f}%")

    # 2. Run the simulation
    simulator = MarketSimulator(config)
    raw_scenarios = simulator.run()

    # 3. Wrap results using the helper wrapper
    results = SimulationResults(raw_scenarios)

    # 4. Analyze Equity Performance across scenarios using the query engine
    equity_stats = results.query("equity_growth", stat="raw", year=10.0)

    print("\n[Analysis] Equity Portfolio Growth of $1.00:")
    print("-" * 50)
    print(f"  - Average Final Value (Year 10):  ${np.mean(equity_stats):.2f}")
    print(f"  - Median Final Value (Year 10):   ${np.median(equity_stats):.2f}")
    print(f"  - Worst 5% Scenario (Downside):   ${np.percentile(equity_stats, 5):.2f}")
    print(f"  - Best 5% Scenario (Upside):      ${np.percentile(equity_stats, 95):.2f}")

    # 5. Analyze Nominal Yield Curves at the final step using query
    mean_yields = [
        results.query("nominal_yield", stat="mean", year=10.0, tenor=t)
        for t in config.tenors
    ]

    print("\n[Analysis] Expected Nominal Yield Curve at Year 10:")
    print("-" * 50)
    yield_table = pd.DataFrame({
        "Tenor (Years)": config.tenors,
        "Expected Nominal Yield": [f"{y*100:.2f}%" for y in mean_yields]
    })
    print(yield_table.to_string(index=False))
    print("==================================================")


if __name__ == "__main__":
    run_basic_simulation()
