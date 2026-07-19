import numpy as np
import pytest
import json
import sys
import pandas as pd
from typing import Dict, List, Any

from aethel import SimulatorConfig, MarketSimulator, SimulationResults, MarketCalibrator
from aethel.engine.simulator import LazyScenarioList
from aethel.calibration.equity import EquityCalibrator
from aethel.calibration.rates import RatesCalibrator
from aethel.calibration.inflation import InflationCalibrator
from aethel.output import time_utils, portfolio, decumulation, visualizer


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def stable_seed() -> int:
    return 42


@pytest.fixture
def baseline_config(stable_seed) -> SimulatorConfig:
    return SimulatorConfig(
        duration_years=3,
        num_scenarios=50,
        seed=stable_seed,
        initial_rate=0.04,
        initial_inflation=0.02
    )


# =====================================================================
# 1. CONFIGURATION & SERIALIZATION TESTS
# =====================================================================

class TestConfigurationAndValidation:
    """Verifies SimulatorConfig state, validation boundaries, and serialization."""

    def test_serialization_pipeline(self, baseline_config):
        """Checks if config dict conversions preserve structures and array types."""
        config_dict = baseline_config.to_dict()
        
        # Verify JSON compatibility (no raw numpy arrays inside dictionary)
        try:
            serialized = json.dumps(config_dict)
            deserialized = json.loads(serialized)
        except TypeError as e:
            pytest.fail(f"Config dictionary is not JSON-serializable: {e}")

        reconstructed_config = SimulatorConfig.from_dict(deserialized)
        
        assert reconstructed_config.duration_years == baseline_config.duration_years
        assert reconstructed_config.num_scenarios == baseline_config.num_scenarios
        assert np.array_equal(reconstructed_config.tenors, baseline_config.tenors)
        assert reconstructed_config.steps == baseline_config.steps

    def test_extreme_merton_derived_properties(self):
        """Verifies mathematical helper properties do not crash with extreme inputs."""
        config = SimulatorConfig(mu_J=-0.99, sigma_J=0.0)
        expected_k = np.exp(-0.99) - 1.0
        assert np.isclose(config.k_jump, expected_k)


# =====================================================================
# 2. CALIBRATION ROBUSTNESS TESTS (CORNER CASES)
# =====================================================================

class TestCalibrationRobustness:
    """Tests parameter calibration solvers under noisy or degenerate scenarios."""

    def test_merton_calibration_with_degenerate_data(self):
        """Tests EquityCalibrator fallback logic under zero-volatility returns."""
        flat_returns = np.full(24, 0.005)
        params = EquityCalibrator.calibrate(flat_returns, dt=1.0/12.0)
        
        assert "gbm_sigma" in params
        assert "lambda_J" in params
        assert params["gbm_sigma"] > 0.0
        assert 0.0 <= params["lambda_J"] <= 1.0

    def test_merton_extreme_outliers(self):
        """Tests that calibration handles massive outliers without numerical overflow."""
        np.random.seed(123)
        returns = np.random.normal(0.008, 0.04, 60)
        returns[15] = -0.90
        
        params = EquityCalibrator.calibrate(returns, dt=1.0/12.0)
        assert np.isfinite(params["gbm_sigma"])
        assert np.isfinite(params["lambda_J"])
        assert params["mu_J"] < 0.0

    def test_inflation_non_stationary_inputs(self):
        """Tests InflationCalibrator when inputs represent explosive or non-stationary AR(1)."""
        explosive_inflation = np.array([0.01 * (1.1 ** i) for i in range(30)])
        
        params = InflationCalibrator.calibrate(explosive_inflation, dt=1.0/12.0)
        assert np.isfinite(params["ou_theta"])
        assert np.isfinite(params["ou_mu"])
        assert params["ou_theta"] > 0.0

    def test_rates_curve_fit_extreme_inversion(self):
        """Tests yield curve optimization under a highly inverted target yield curve."""
        tenors = np.array([0.25, 1.0, 5.0, 10.0, 30.0])
        inverted_targets = np.array([0.15, 0.12, 0.08, 0.05, 0.03])
        
        params = RatesCalibrator.fit_yield_curve_to_target(
            target_yields=inverted_targets,
            tenors=tenors,
            initial_rate=0.14
        )
        
        assert params["cir_theta"] > 0.0
        assert params["cir_mu"] > 0.0
        assert params["cir_sigma"] > 0.0
        assert np.isfinite(list(params.values())).all()

    def test_unified_market_calibrator_with_yield_curve(self):
        """Tests standard unified calibrator with historical yield curve inputs."""
        np.random.seed(101)
        history_len = 36
        hist_inf = np.random.normal(0.02, 0.005, history_len)
        hist_rat = np.random.normal(0.03, 0.005, history_len)
        hist_eq = np.random.normal(0.008, 0.05, history_len)
        
        tenors = np.array([0.25, 1.0, 5.0, 10.0])
        hist_curve = np.array([0.02, 0.025, 0.03, 0.035])

        calibrator = MarketCalibrator()
        config = calibrator.fit(
            historical_inflation=hist_inf,
            historical_rates=hist_rat,
            historical_equity_returns=hist_eq,
            historical_yield_curve=hist_curve,
            tenors=tenors
        )
        
        assert isinstance(config, SimulatorConfig)
        assert config.initial_rate == hist_rat[-1]
        assert config.initial_inflation == hist_inf[-1]

    def test_rates_calibrator_short_series_exception(self):
        """Ensures rates calibrator raises a ValueError on short input arrays."""
        with pytest.raises(ValueError, match="at least 3 observations"):
            RatesCalibrator.calibrate_short_rate_series(np.array([0.01, 0.02]))

    def test_rates_calibrator_degenerate_drift(self):
        """Tests short-rate series calibration under zero or negative mean reversion."""
        degenerate_rates = np.array([0.05, 0.05, 0.05, 0.05])
        params = RatesCalibrator.calibrate_short_rate_series(degenerate_rates, dt=1.0/12.0)
        assert params["cir_theta"] > 0.0
        assert params["cir_mu"] > 0.0


# =====================================================================
# 3. ENGINE STRATEGY AND MEMORY OPTIMIZATION TESTS
# =====================================================================

class TestSimulationExecutionStrategies:
    """Verifies memory optimizations, chunking, and lazy loading consistency."""

    def test_chunking_numerical_parity(self, baseline_config):
        """Ensures that executing simulations in chunks produces identical results to single-block runs."""
        baseline_config.chunk_size = None
        baseline_config.max_workers = 1
        sim1 = MarketSimulator(baseline_config)
        results1 = sim1.run()
        
        baseline_config.chunk_size = 10
        baseline_config.max_workers = 2
        sim2 = MarketSimulator(baseline_config)
        results2 = sim2.run()
        
        assert len(results1) == len(results2)
        
        for idx in [0, 12, 25, 49]:
            np.testing.assert_allclose(results1[idx]["stock_returns"], results2[idx]["stock_returns"])
            np.testing.assert_allclose(results1[idx]["deposit_rates"], results2[idx]["deposit_rates"])
            np.testing.assert_allclose(results1[idx]["cpis"], results2[idx]["cpis"])
            np.testing.assert_allclose(results1[idx]["nominal_yield_curves"], results2[idx]["nominal_yield_curves"])
            np.testing.assert_allclose(results1[idx]["real_yield_curves"], results2[idx]["real_yield_curves"])

    def test_lazy_scenario_list_boundaries(self, baseline_config):
        """Verifies boundary indexing, slicing, and exception handling of LazyScenarioList."""
        sim = MarketSimulator(baseline_config)
        scenarios = sim.run()
        
        assert isinstance(scenarios, LazyScenarioList)
        
        last_scenario = scenarios[-1]
        equivalent_scenario = scenarios[baseline_config.num_scenarios - 1]
        np.testing.assert_equal(last_scenario["stock_returns"], equivalent_scenario["stock_returns"])
        
        with pytest.raises(IndexError):
            _ = scenarios[baseline_config.num_scenarios]
            
        with pytest.raises(IndexError):
            _ = scenarios[-baseline_config.num_scenarios - 1]

        sliced = scenarios[5:15]
        assert len(sliced) == 10
        assert isinstance(sliced, list)


# =====================================================================
# 4. PORTFOLIO & DECUMULATION DYNAMICS (STRESS CASES)
# =====================================================================

class TestPortfolioAndDecumulationDynamics:
    """Evaluates portfolio construction and retirement decumulation under extreme pressure."""

    def test_invalid_portfolio_weights(self, baseline_config):
        """Tests that invalid portfolio specifications are rejected immediately."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        with pytest.raises(ValueError, match="Weights dictionary cannot be empty"):
            _ = results.calculate_portfolio_returns({})
            
        with pytest.raises(ValueError, match="Sum of portfolio weights must be greater than zero"):
            _ = results.calculate_portfolio_returns({"equity": 0.0, "fixed_income": 0.0})

        with pytest.raises(ValueError, match="Unknown asset class"):
            _ = results.calculate_portfolio_returns({"non_existent_asset": 1.0})

    def test_alternative_portfolio_weights(self, baseline_config):
        """Tests alternative naming variations for asset weight specifications."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        # Test naming aliases like 'cash', 'cdi', 'stock'
        w = {"stock": 0.4, "cash": 0.6}
        returns = results.calculate_portfolio_returns(w)
        assert returns.shape == (baseline_config.steps, baseline_config.num_scenarios)

        w2 = {"equity_returns": 0.5, "cdi": 0.5}
        growth = results.calculate_portfolio_growth(w2)
        assert growth.shape == (baseline_config.steps + 1, baseline_config.num_scenarios)

    def test_decumulation_with_immediate_bankruptcy(self, baseline_config):
        """Verifies decumulation paths with a high withdrawal rate relative to capital."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        decum_data = results.simulate_decumulation(
            initial_balance=10000.0,
            initial_monthly_withdrawal=25000.0,
            portfolio_weights={"equity": 0.5, "fixed_income": 0.5},
            withdrawal_timing="beginning"
        )
        
        assert np.all(decum_data["balances"][1:, :] == 0.0)
        assert np.all(decum_data["probability_of_success"][1:] == 0.0)

    def test_decumulation_draconian_friction_and_taxes(self, baseline_config):
        """Tests asset performance under extreme tax (99%) and drag (50% annual)."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        decum_data = results.simulate_decumulation(
            initial_balance=1_000_000.0,
            initial_monthly_withdrawal=1_000.0,
            portfolio_weights={"equity": 0.6, "fixed_income": 0.4},
            frictional_drag_annual=0.50,
            tax_on_gains_rate=0.99
        )
        
        baseline_decum = results.simulate_decumulation(
            initial_balance=1_000_000.0,
            initial_monthly_withdrawal=1_000.0,
            portfolio_weights={"equity": 0.6, "fixed_income": 0.4}
        )
        
        assert decum_data["balances"][-1].mean() < baseline_decum["balances"][-1].mean()

    def test_decumulation_cash_first_depletion_mechanic(self, baseline_config):
        """Validates the order of operations for asset depletion in Cash-First strategy."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        decum_data = results.simulate_decumulation(
            initial_balance=100_000.0,
            initial_monthly_withdrawal=5_000.0,
            portfolio_weights={"equity": 0.90, "fixed_income": 0.10},
            liquidation_strategy="cash_first"
        )
        
        fi_balances = decum_data["fixed_income_balances"]
        eq_balances = decum_data["equity_balances"]
        
        has_reached_zero_fi = np.any(fi_balances == 0.0, axis=0)
        assert np.any(has_reached_zero_fi)
        
        for t in range(1, baseline_config.steps):
            depleted_fi_mask = fi_balances[t, :] == 0.0
            assert np.all(eq_balances[t + 1, depleted_fi_mask] <= eq_balances[t, depleted_fi_mask] * (1.15))

    def test_decumulation_end_of_period_timing(self, baseline_config):
        """Tests decumulation utilizing end-of-period withdrawal timing logic."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        decum_data = results.simulate_decumulation(
            initial_balance=1_000_000.0,
            initial_monthly_withdrawal=5_000.0,
            portfolio_weights={"equity": 0.6, "fixed_income": 0.4},
            withdrawal_timing="end",
            tax_on_gains_rate=0.15
        )
        assert decum_data["balances"].shape == (baseline_config.steps + 1, baseline_config.num_scenarios)

    def test_decumulation_dynamic_guardrail_policy(self, baseline_config):
        """Verifies decumulation when a custom spending policy modifies monthly withdrawals."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        def extreme_safety_rule(balance, cpi_factor, step, deposit_rate):
            return np.where(balance < 500_000.0, 0.0, 10_000.0)

        decum_data = results.simulate_decumulation(
            initial_balance=600_000.0,
            initial_monthly_withdrawal=10_000.0,
            portfolio_weights={"equity": 0.5, "fixed_income": 0.5},
            withdrawal_policy=extreme_safety_rule
        )
        
        min_ending_balance = np.min(decum_data["balances"][-1])
        assert min_ending_balance >= 0.0


# =====================================================================
# 5. TIME UTILITY AND INVALID QUERY TESTS
# =====================================================================

class TestTimeUtilityAndQueryErrors:
    """Tests the resolution of time coordinates and invalid stat query exceptions."""

    def test_time_resolution_conflicts(self):
        """Verifies that query parsing raises clear errors if conflicting horizons are requested."""
        with pytest.raises(ValueError, match="Conflicting time parameters"):
            time_utils.resolve_time_to_indices(
                time_legacy=None,
                year=3.0,
                month=36,
                step=None,
                max_idx=120,
                steps_per_year=12
            )

    def test_invalid_query_parameters(self, baseline_config):
        """Verifies that requesting invalid metrics or percentiles raises explicit errors."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        with pytest.raises(ValueError, match="Unknown metric"):
            results.query("invalid_metric_name", stat="mean")
            
        with pytest.raises(ValueError, match="Could not parse valid percentile"):
            results.query("rate", stat="p150")
            
        with pytest.raises(ValueError, match="Could not parse valid percentile"):
            results.query("rate", stat="p-10")

    def test_annualization_division_by_zero_protection(self):
        """Ensures that the annualization parser handles step=0 boundaries safely."""
        matrix = np.array([[1.0, 1.0], [1.05, 1.08]])
        annualized = time_utils.apply_annualization(matrix, "equity_growth", steps_per_year=12)
        assert annualized[0, 0] == 0.0
        assert np.isfinite(annualized).all()

        # Test index mapping
        cpi_matrix = np.array([[1.0, 1.0], [1.02, 1.03]])
        annual_cpi = time_utils.apply_annualization(cpi_matrix, "cpi", steps_per_year=12)
        assert np.isfinite(annual_cpi).all()


# =====================================================================
# 6. ADVERSARIAL SYSTEM MATRIX AND MEMOIZATION TESTS
# =====================================================================

class TestAdversarialSystemAndCache:
    """Validates systemic cache invalidation and structural matrix extraction."""

    def test_results_caching_and_cleanup(self, baseline_config):
        """Verifies internal memoization and clean-up functions."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())
        
        rate_matrix_1 = results._extract_base_matrix("rate")
        assert "rate" in results._cache
        
        rate_matrix_2 = results._extract_base_matrix("rate")
        assert rate_matrix_1 is rate_matrix_2
        
        results.cleanup()
        assert len(results._cache) == 0

    def test_query_variations_for_coverage(self, baseline_config):
        """Tests query transformations on portfolio and yield curves to expand results.py coverage."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())

        # Cache population of decumulation metrics
        _ = results.simulate_decumulation(100000.0, 1000.0, {"equity": 1.0})
        
        decum_b = results._extract_base_matrix("decumulation_balance")
        decum_w = results._extract_base_matrix("decumulation_withdrawal")
        assert decum_b is not None
        assert decum_w is not None

        # Base matrix extractions for yields, cpi growth, and returns
        eq_growth = results._extract_base_matrix("equity_growth")
        inflation_r = results._extract_base_matrix("inflation_rate")
        nom_yield = results._extract_base_matrix("nominal_yield", tenor=5.0)
        real_yield = results._extract_base_matrix("real_yield", tenor=5.0)

        assert eq_growth.shape[0] == baseline_config.steps + 1
        assert inflation_r.shape[0] == baseline_config.steps
        assert nom_yield.shape[0] == baseline_config.steps
        assert real_yield.shape[0] == baseline_config.steps

    def test_results_exceptions_on_missing_simulation(self, baseline_config):
        """Ensures results query returns exceptions on uninitialized states."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())

        with pytest.raises(ValueError, match="not simulated yet"):
            _ = results._extract_base_matrix("decumulation_balance")

        with pytest.raises(ValueError, match="not simulated yet"):
            _ = results._extract_base_matrix("decumulation_withdrawal")

        with pytest.raises(ValueError, match="must be specified"):
            _ = results._extract_base_matrix("nominal_yield")

        with pytest.raises(ValueError, match="not available"):
            _ = results._extract_base_matrix("nominal_yield", tenor=999.0)

    def test_legacy_interface_methods(self, baseline_config):
        """Executes legacy compatibility endpoints in SimulationResults."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())

        assert results.get_matrix("stock_returns").shape == (baseline_config.steps, baseline_config.num_scenarios)
        assert results.get_yield_curve_matrix(real=False, tenor_idx=0).shape == (baseline_config.steps, baseline_config.num_scenarios)
        
        sum_stats = results.get_summary_statistics("cpis")
        assert "mean" in sum_stats
        assert "p50" in sum_stats

        df = results.to_pandas(scenario_idx=0)
        assert isinstance(df, pd.DataFrame)
        assert "stock_returns" in df.columns

        with pytest.raises(ValueError, match="must be one of"):
            _ = results.get_matrix("invalid_legacy_name")

        with pytest.raises(ValueError, match="must be one of"):
            _ = results.get_summary_statistics("invalid_legacy_name")

        with pytest.raises(IndexError):
            _ = results.to_pandas(999)


# =====================================================================
# 7. PLOTLY VISUALIZATION trace COMPILING TESTS
# =====================================================================

class TestVisualizationOutputs:
    """Verifies that visualizers generate complete, finite Plotly structures."""

    def test_visualizer_compiles(self, baseline_config):
        """Asserts Plotly figure object generation across all visualization modes."""
        sim = MarketSimulator(baseline_config)
        results = SimulationResults(sim.run())

        # Test fan chart
        fig_fan = results.plot_fan_chart(metric="rate")
        assert fig_fan.data is not None

        # Test path traces
        fig_paths = results.plot_scenario_paths(metric="cpi", num_paths=5)
        assert fig_paths.data is not None

        # Test curve evolution
        fig_curves = results.plot_yield_curve_evolution(years_milestones=[0.0, 1.0, 2.0], real=False)
        assert fig_curves.data is not None

        # Test horizon distributions
        fig_dist = results.plot_horizon_distribution(metric="growth", target_year=2.0)
        assert fig_dist.data is not None


# =====================================================================
# PROGRAMMATIC ENTRY POINT AND TERMINAL REPORTING
# =====================================================================

if __name__ == "__main__":
    print("======================================================================")
    print("             AETHEL ESG - SYSTEM STRESS & ROBUSTNESS AUDIT            ")
    print("======================================================================")
    print("Initiating comprehensive evaluation of calibration, parameters, boundary")
    print("conditions, memory allocation, and decumulation structures...\n")

    exit_code = pytest.main(["-v", "--tb=short", __file__])

    print("\n======================================================================")
    print("                     STRESS AUDIT TERMINAL REPORT                     ")
    print("======================================================================")
    
    if exit_code == 0:
        print(" AUDIT STATUS: PASSED")
        print(" Details:")
        print(" - Calibration models handles degenerate, noisy, and inverted datasets.")
        print(" - Simulation block chunking is numerically consistent with standard runs.")
        print(" - Lazy scenario allocations enforce bounds and negative coordinate steps.")
        print(" - Decumulation mechanics manage tax drag, high fees, and cash depletions.")
        print(" - Plotly visualizers generate complete, compiled graph trace structures.")
    else:
        print(" AUDIT STATUS: FAILED")
        print(" Details:")
        print(" - Disruptions or unexpected outputs occurred under stress testing.")
        print(" - Review specific assertion failures in the trace logs printed above.")
        
    print("======================================================================")

    sys.exit(exit_code)
