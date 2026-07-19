import os
import json
import numpy as np
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

# --- SYMMETRIC PARAMETER GRID (HORIZON FIXED AT 50 YEARS) ---
HORIZON_YEARS = 50
INFLATION_TARGETS = [0.02, 0.04, 0.06] # 2%, 4%, and 6% structural targets
WITHDRAWALS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
ALLOCATION_MIXES = [20, 30, 40, 50, 60, 70, 80, 90, 100]  # Representing Equity %
STRATEGIES = ["constant_mix", "cash_first_guardrail"]

DOWNSAMPLE_FACTOR = 3  # Downsample to quarterly steps to optimize JSON size
PRESETS_DIR = "presets"
OUTPUT_FILE = "demo/demo_database.json"


def load_usa_config() -> SimulatorConfig:
    """Loads default USA parameters from presets directory."""
    path = os.path.join(PRESETS_DIR, "usa.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing 'usa.json' in {PRESETS_DIR}. Run build_presets.py first.")

    with open(path, "r") as f:
        preset_dict = json.load(f)
    return SimulatorConfig.from_dict(preset_dict)


def sanitize_rates(arr) -> list[float]:
    """Rounds interest rates/probabilities to 4 decimals for space efficiency."""
    return [round(float(x), 4) for x in arr]


def sanitize_balances(arr) -> list[int]:
    """Saves significant storage space by rounding balances to the nearest integer."""
    return [int(round(float(x))) for x in arr]


def make_spending_guardrail(initial_monthly_withdrawal: float, threshold: float = 400000.0, reduction: float = 0.80):
    """Factory function returning a vectorized scenario-level spending guardrail."""
    def variable_spending_guardrail(balance, cpi_factor, step, deposit_rate):
        base_w = initial_monthly_withdrawal * cpi_factor
        is_depleted = balance < threshold
        return np.where(is_depleted, base_w * reduction, base_w)
    return variable_spending_guardrail


def main():
    print("==================================================")
    print(" Aethel ESG - Compiling Multi-Target Demo Cache   ")
    print("==================================================")

    config = load_usa_config()
    config.duration_years = HORIZON_YEARS
    config.num_scenarios = 1000  # High scenario count for high-quality statistics
    config.seed = 42
    config.max_workers = 4

    db = {
        "metadata": {
            "preset": "usa",
            "scenarios": config.num_scenarios,
            "horizon_years": HORIZON_YEARS,
            "inflation_targets": INFLATION_TARGETS,
            "withdrawals": WITHDRAWALS,
            "allocations": ALLOCATION_MIXES,
            "strategies": STRATEGIES,
            "downsample_factor": DOWNSAMPLE_FACTOR
        },
        "economic_data": {},
        "decumulation_data": {}
    }

    # Generate economic base paths per inflation target
    for target in INFLATION_TARGETS:
        target_pct = int(round(target * 100))
        print(f"\n -> Generating Base Economic Paths for {target_pct}% Inflation Target...")
        
        # Override structural inflation mean parameters in parameters config
        config.ou_mu = target
        
        simulator = MarketSimulator(config)
        raw_scenarios = simulator.run()
        results = SimulationResults(raw_scenarios)

        steps_count = config.steps
        full_timeline = np.arange(steps_count + 1) / 12.0
        downsampled_timeline = full_timeline[::DOWNSAMPLE_FACTOR].tolist()

        # Extract downsampled macro-economic quantiles
        cpi_p5 = sanitize_rates(results.query("cpi", stat="p5", step="all")[::DOWNSAMPLE_FACTOR])
        cpi_p50 = sanitize_rates(results.query("cpi", stat="p50", step="all")[::DOWNSAMPLE_FACTOR])
        cpi_p95 = sanitize_rates(results.query("cpi", stat="p95", step="all")[::DOWNSAMPLE_FACTOR])
        cpi_mean = sanitize_rates(results.query("cpi", stat="mean", step="all")[::DOWNSAMPLE_FACTOR])

        yield_p5 = sanitize_rates(results.query("rate", stat="p5", step="all")[::DOWNSAMPLE_FACTOR])
        yield_p50 = sanitize_rates(results.query("rate", stat="p50", step="all")[::DOWNSAMPLE_FACTOR])
        yield_p95 = sanitize_rates(results.query("rate", stat="p95", step="all")[::DOWNSAMPLE_FACTOR])
        yield_mean = sanitize_rates(results.query("rate", stat="mean", step="all")[::DOWNSAMPLE_FACTOR])

        db["economic_data"][f"target_{target_pct}"] = {
            "timeline": sanitize_rates(downsampled_timeline),
            "cpi": {"p5": cpi_p5, "p25": cpi_p50, "p50": cpi_p50, "p75": cpi_p50, "p95": cpi_p95, "mean": cpi_mean},
            "yield": {"p5": yield_p5, "p25": yield_p50, "p50": yield_p50, "p75": yield_p50, "p95": yield_p95, "mean": yield_mean}
        }

        # Run combinatorial decumulations for this target
        for w_amount in WITHDRAWALS:
            for eq_mix in ALLOCATION_MIXES:
                portfolio_weights = {
                    "equity": eq_mix / 100.0,
                    "fixed_income": (100 - eq_mix) / 100.0
                }

                for strategy in STRATEGIES:
                    combo_key = f"{w_amount}w_{eq_mix}a_{strategy}_{target_pct}t"
                    print(f"    * Node: Inflation Target {target_pct}% | ${w_amount}/mo | {eq_mix}% Equity | Strategy: {strategy}")

                    if strategy == "cash_first_guardrail":
                        guardrail_rule = make_spending_guardrail(
                            initial_monthly_withdrawal=w_amount,
                            threshold=400000.0,
                            reduction=0.80
                        )
                        decum_results = results.simulate_decumulation(
                            initial_balance=1000000.0,
                            initial_monthly_withdrawal=w_amount,
                            portfolio_weights=portfolio_weights,
                            liquidation_strategy="cash_first",
                            frictional_drag_annual=0.0025,
                            tax_on_gains_rate=0.15,
                            withdrawal_policy=guardrail_rule
                        )
                    else:
                        decum_results = results.simulate_decumulation(
                            initial_balance=1000000.0,
                            initial_monthly_withdrawal=w_amount,
                            portfolio_weights=portfolio_weights,
                            liquidation_strategy="constant_mix",
                            frictional_drag_annual=0.0025,
                            tax_on_gains_rate=0.15,
                            withdrawal_policy=None
                        )

                    solvency_curve = sanitize_rates(decum_results["probability_of_success"][::DOWNSAMPLE_FACTOR])

                    balances_at_steps = decum_results["balances"][::DOWNSAMPLE_FACTOR]
                    p5_balances = sanitize_balances(np.percentile(balances_at_steps, 5.0, axis=1))
                    p50_balances = sanitize_balances(np.percentile(balances_at_steps, 50.0, axis=1))
                    p95_balances = sanitize_balances(np.percentile(balances_at_steps, 95.0, axis=1))

                    db["decumulation_data"][combo_key] = {
                        "solvency": solvency_curve,
                        "balance_p5": p5_balances,
                        "balance_p50": p50_balances,
                        "balance_p95": p95_balances
                    }

        results.cleanup()

    # Save to highly-compressed JSON using minified separators
    with open(OUTPUT_FILE, "w") as f:
        json.dump(db, f, separators=(',', ':'))

    print("\n==================================================")
    print(f" SUCCESS! Target Database generated: {OUTPUT_FILE}")
    print(f" Size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
    print("==================================================")


if __name__ == "__main__":
    main()
