import numpy as np
import pytest
from typing import Dict, List

from aethel import SimulatorConfig, MarketSimulator, SimulationResults
from aethel.engine.loops import HAS_NUMBA, run_simulation_loop_numpy


@pytest.fixture
def standard_config() -> SimulatorConfig:
    """Provides a standard baseline configuration for testing."""
    return SimulatorConfig(
        duration_years=5,
        num_scenarios=20,
        seed=101,
        initial_rate=0.08,
        initial_inflation=0.045
    )


class TestMacroeconomicEngineStructure:
    """Tests dimensions, reproducibility, and structural attributes of the ESG."""

    def test_output_shapes_and_keys(self, standard_config):
        """Verifies that all expected variables are generated with correct shapes."""
        simulator = MarketSimulator(standard_config)
        scenarios = simulator.run()

        expected_keys = {
            "stock_returns",
            "cpis",
            "deposit_rates",
            "nominal_yield_curves",
            "real_yield_curves",
            "tenors"
        }

        assert len(scenarios) == standard_config.num_scenarios
        expected_steps = standard_config.steps  # 5 years * 12 = 60 steps

        for idx, scenario in enumerate(scenarios):
            assert expected_keys.issubset(scenario.keys()), f"Scenario {idx} is missing keys."
            assert len(scenario["stock_returns"]) == expected_steps
            assert len(scenario["deposit_rates"]) == expected_steps
            assert len(scenario["cpis"]) == expected_steps
            assert scenario["nominal_yield_curves"].shape == (expected_steps + 1, len(standard_config.tenors))
            assert scenario["real_yield_curves"].shape == (expected_steps + 1, len(standard_config.tenors))
            assert np.array_equal(scenario["tenors"], standard_config.tenors)

    def test_reproducibility(self, standard_config):
        """Ensures that identical seeds yield identical trajectories (determinism)."""
        sim1 = MarketSimulator(standard_config)
        results1 = sim1.run()

        sim2 = MarketSimulator(standard_config)
        results2 = sim2.run()

        for s1, s2 in zip(results1, results2):
            np.testing.assert_allclose(s1["stock_returns"], s2["stock_returns"])
            np.testing.assert_allclose(s1["deposit_rates"], s2["deposit_rates"])
            np.testing.assert_allclose(s1["cpis"], s2["cpis"])
            np.testing.assert_allclose(s1["nominal_yield_curves"], s2["nominal_yield_curves"])
            np.testing.assert_allclose(s1["real_yield_curves"], s2["real_yield_curves"])


class TestBoundaryConstraints:
    """Tests the economic boundary conditions and policy floors implemented in loops."""

    def test_interest_rate_non_negativity(self):
        """Verifies that simulated interest rates respect the absolute floor (0.001) for t > 0."""
        config = SimulatorConfig(
            duration_years=3,
            num_scenarios=10,
            initial_rate=-0.05,  # Intentional negative value to test recovery
            cir_mu_val=-0.02,
            mu_min=0.001
        )
        simulator = MarketSimulator(config)
        scenarios = simulator.run()
        results = SimulationResults(scenarios)

        short_rates = results.query("short_rate", stat="raw", step="all")

        # We slice out the initial input step (t=0) to verify loop transition enforcement
        simulated_rates = short_rates[1:]
        assert np.all(simulated_rates >= 0.001), "Short rate violated the absolute lower boundary during simulation."

    def test_inflation_minimum_boundary(self):
        """Verifies inflation rate paths respect the pi_min lower policy threshold for t > 0."""
        pi_min_floor = -0.015
        config = SimulatorConfig(
            duration_years=3,
            num_scenarios=10,
            initial_inflation=-0.05,  # Intentional negative value to test recovery
            ou_mu=-0.03,
            pi_min=pi_min_floor
        )
        simulator = MarketSimulator(config)
        scenarios = simulator.run()
        results = SimulationResults(scenarios)

        inflation_rates = results.query("inflation_rate", stat="raw", step="all")

        # We slice out the initial input step (t=0) to verify loop transition enforcement
        simulated_inflation = inflation_rates[1:]
        assert np.all(simulated_inflation >= pi_min_floor - 1e-7), "Inflation rate fell below pi_min floor during simulation."


class TestYieldCurveTermStructure:
    """Audits interest rate term structure consistency, Fisher real relations, and NaNs."""

    def test_yields_contain_no_nans_or_infinites(self, standard_config):
        """Ensures analytical yield derivations do not yield numerical errors."""
        simulator = MarketSimulator(standard_config)
        scenarios = simulator.run()

        for s in scenarios:
            assert np.all(np.isfinite(s["nominal_yield_curves"]))
            assert np.all(np.isfinite(s["real_yield_curves"]))

    def test_fisher_relation_inequality(self, standard_config):
        """Tests that real yields are generally lower than nominal yields when long-term inflation is positive."""
        simulator = MarketSimulator(standard_config)
        scenarios = simulator.run()

        for s in scenarios:
            nominal = s["nominal_yield_curves"]
            real = s["real_yield_curves"]
            assert np.any(nominal > real), "Real yields should be lower than nominal yields under inflation."


class TestAssetPricingAndMertonJumps:
    """Tests stock return behavior, Equity Risk Premium, and Jump-Diffusion shocks."""

    def test_equity_growth_with_positive_erp(self, standard_config):
        """Ensures that stock growth compounds positively on average over time under standard ERP."""
        simulator = MarketSimulator(standard_config)
        scenarios = simulator.run()
        results = SimulationResults(scenarios)

        mean_final_growth = results.query("growth", stat="mean", year=5.0)
        assert mean_final_growth > 1.0, "Average equity growth index did not compound above initial value."

    def test_merton_jump_impact(self):
        """Verifies that high jump frequency triggers heavy-tailed downside return distribution."""
        config_no_jumps = SimulatorConfig(
            duration_years=3, num_scenarios=50, seed=42,
            lambda_J=0.0, mu_J=0.0, sigma_J=0.0
        )
        sim_no_jumps = MarketSimulator(config_no_jumps)
        results_no_jumps = SimulationResults(sim_no_jumps.run())
        returns_no_jumps = results_no_jumps.query("returns", stat="raw", step="all")

        config_with_jumps = SimulatorConfig(
            duration_years=3, num_scenarios=50, seed=42,
            lambda_J=2.0, mu_J=-0.35, sigma_J=0.10
        )
        sim_with_jumps = MarketSimulator(config_with_jumps)
        results_with_jumps = SimulationResults(sim_with_jumps.run())
        returns_with_jumps = results_with_jumps.query("returns", stat="raw", step="all")

        min_no_jump = np.min(returns_no_jumps)
        min_with_jump = np.min(returns_with_jumps)

        assert min_with_jump < min_no_jump, "Jump-diffusion failed to produce deep outlier returns."


@pytest.mark.skipif(not HAS_NUMBA, reason="Numba is not installed or available.")
class TestNumbaEquivalence:
    """Ensures mathematical parity between Numba-compiled and NumPy-vectorized loop engines."""

    def test_loop_results_identical(self, standard_config):
        """
        Verifies that Numba-accelerated and standard NumPy-vectorized loops produce identical trajectories.
        Relative tolerance is configured to handle sub-epsilon variations introduced by hardware fastmath optimizations.
        """
        sim = MarketSimulator(standard_config)
        results_numba = sim.run()

        from aethel.engine import simulator as sim_module
        original_has_numba = sim_module.HAS_NUMBA
        try:
            sim_module.HAS_NUMBA = False
            sim_numpy_override = MarketSimulator(standard_config)
            results_numpy = sim_numpy_override.run()
        finally:
            sim_module.HAS_NUMBA = original_has_numba

        for s_numba, s_numpy in zip(results_numba, results_numpy):
            np.testing.assert_allclose(s_numba["stock_returns"], s_numpy["stock_returns"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(s_numba["deposit_rates"], s_numpy["deposit_rates"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(s_numba["cpis"], s_numpy["cpis"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(s_numba["nominal_yield_curves"], s_numpy["nominal_yield_curves"], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(s_numba["real_yield_curves"], s_numpy["real_yield_curves"], rtol=1e-10, atol=1e-12)


# --- PROGRAMMATIC ENTRY POINT AND TERMINAL REPORTING ---
if __name__ == "__main__":
    import sys

    print("======================================================================")
    print("             ACTUARIAL ESG - MACROECONOMIC ENGINE REPORT              ")
    print("======================================================================")
    print("Initiating verification on engine dimensions, limits, and pricing...\n")

    exit_code = pytest.main(["-v", "--tb=short", __file__])

    print("\n======================================================================")
    if exit_code == 0:
        print(" VERIFICATION RESULT: PASSED ")
        print(" All mathematical bounds, loops, and structure checks are verified.")
    else:
        print(" VERIFICATION RESULT: FAILED ")
        print(" Issues detected in the macroeconomic calculations. Review tracebacks above.")
    print("======================================================================")

    sys.exit(exit_code)
    