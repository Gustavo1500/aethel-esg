import json
import os
import numpy as np
import pandas as pd
from aethel import MarketCalibrator, SimulatorConfig


def create_mocksample_csv(filename: str):
    """Generates a mock CSV file containing historical market records."""
    np.random.seed(42)
    rows = 120  # 10 years of monthly data

    # Generate mock variables using standardized names
    short_rate = np.clip(np.random.normal(0.09, 0.015, rows), 0.02, 0.15)
    inflation = np.clip(short_rate - 0.04 + np.random.normal(0, 0.005, rows), -0.01, 0.10)
    equity_returns = np.random.normal(0.008, 0.05, rows)

    df = pd.DataFrame({
        "date": pd.date_range(end="2026-01-01", periods=rows, freq="ME"),
        "short_rate": short_rate,
        "inflation": inflation,
        "equity_return": equity_returns
    })
    df.to_csv(filename, index=False)
    print(f"[File] Generated sample data file: '{filename}'")


def run_calibration_example():
    print("==================================================")
    print("      Aethel ESG - Calibration & Saving           ")
    print("==================================================")

    csv_filename = "historical_market_data.csv"
    json_filename = "calibrated_config.json"

    # 1. Ensure historical CSV exists
    create_mocksample_csv(csv_filename)

    # 2. Load historical series using Pandas
    df = pd.read_csv(csv_filename)
    print(f"Successfully loaded {len(df)} months of historical data.")

    # 3. Instantiate calibrator and fit parameters
    calibrator = MarketCalibrator()
    print("\n[Calibration] Fitting models to historical series...")

    calibrated_config = calibrator.fit(
        historical_inflation=df["inflation"].values,
        historical_rates=df["short_rate"].values,
        historical_equity_returns=df["equity_return"].values
    )

    config_dict = calibrated_config.to_dict()

    # 5. Save config dictionary to a JSON file
    with open("calibrated_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    print(f"[File] Calibrated parameters saved to: '{json_filename}'")

    # 6. Display a summary of the calibrated parameters
    print("\n[Summary] Calibration Outputs:")
    print("-" * 50)
    print(f"  - Calibrated Equity Volatility (GBM): {config_dict['gbm_sigma_val']*100:.2f}%")
    print(f"  - Calibrated Inflation target (OU):   {config_dict['ou_mu']*100:.2f}%")
    print(f"  - Calibrated Inflation Volatility:    {config_dict['ou_sigma_val']*100:.2f}%")
    print(f"  - Calibrated Rate Target Mean (CIR):  {config_dict['cir_mu_val']*100:.2f}%")
    print(f"  - Calibrated Rate Volatility (CIR):   {config_dict['cir_sigma_val']*100:.2f}%")
    print("-" * 50)

    # Clean up mock CSV
    if os.path.exists(csv_filename):
        os.remove(csv_filename)

    print("Calibration pipeline complete.")
    print("==================================================")


if __name__ == "__main__":
    run_calibration_example()
