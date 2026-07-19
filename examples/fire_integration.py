import numpy as np
import pandas as pd
from aethel import MarketSimulator, MarketCalibrator, SimulationResults


def generate_mock_historical_data(months: int = 120):
    np.random.seed(101)
    historical_inflation = 0.045 + np.random.normal(0, 0.005, months)
    historical_rates = historical_inflation + 0.04 + np.random.normal(0, 0.008, months)
    historical_rates = np.maximum(0.01, historical_rates)
    historical_equity = 0.11 / 12.0 + np.random.normal(0.0, 0.05, months)
    historical_curve = np.array([0.08, 0.085, 0.09, 0.098, 0.105, 0.11, 0.112, 0.115])

    return historical_inflation, historical_rates, historical_equity, historical_curve


def run_full_pipeline():
    print("==================================================")
    print(" Aethel ESG - Complete Pipeline Execution Demo    ")
    print("==================================================")

    # Step 1: Generate/Import historical market data
    months_of_history = 120
    hist_inf, hist_rat, hist_eq, hist_curve = generate_mock_historical_data(months=months_of_history)
    print(f"Loaded {months_of_history} months of historical calibration data.")

    # Step 2: Calibrate the ESG parameters
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0])
    calibrator = MarketCalibrator()

    print("\n[Calibration] Fitting models to historical records...")
    calibrated_config = calibrator.fit(
        historical_inflation=hist_inf,
        historical_rates=hist_rat,
        historical_equity_returns=hist_eq,
        historical_yield_curve=hist_curve,
        tenors=tenors
    )

    calibrated_config.duration_years = 30  # Horizon for FIRE evaluation
    calibrated_config.num_scenarios = 500
    print("[Calibration] Success. New configuration generated.")

    # Step 3: Run Simulation Engine
    simulator = MarketSimulator(calibrated_config)
    raw_scenarios = simulator.run()

    # Step 4: Wrap outputs and process
    results = SimulationResults(raw_scenarios)
    print("\n[Results] Wrapping outputs complete.")

    # Step 5: Export a single path for a FIRE simulator run-through
    scenario_df = results.to_pandas(scenario_idx=0)
    print("\nSample Scenario 0 path structure (first 5 months):")
    print(scenario_df[["stock_returns", "cpis", "deposit_rates", "real_yield_10.0y"]].head())

    # Step 6: Query standard percentiles to estimate risk boundaries
    print("\nInflation Index Cumulative Path distribution (at year 10):")
    print(f"  - 10th percentile (low inflation):  {results.query('cpi', stat='p10', year=10.0):.4f}")
    print(f"  - 50th percentile (median):         {results.query('cpi', stat='p50', year=10.0):.4f}")
    print(f"  - 90th percentile (high inflation): {results.query('cpi', stat='p90', year=10.0):.4f}")
    print("==================================================")


if __name__ == "__main__":
    run_full_pipeline()
