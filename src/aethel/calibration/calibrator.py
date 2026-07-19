import numpy as np
from typing import Optional

from aethel.engine.parameters import SimulatorConfig
from aethel.calibration.inflation import InflationCalibrator
from aethel.calibration.rates import RatesCalibrator
from aethel.calibration.equity import EquityCalibrator


class MarketCalibrator:
    """
    Unified API orchestrating model-parameter calibration
    from raw input historical series.
    """

    def __init__(self, base_config: Optional[SimulatorConfig] = None):
        self.config = base_config if base_config is not None else SimulatorConfig()

    def fit(
        self,
        historical_inflation: np.ndarray,
        historical_rates: np.ndarray,
        historical_equity_returns: np.ndarray,
        historical_yield_curve: Optional[np.ndarray] = None,
        tenors: Optional[np.ndarray] = None
    ) -> SimulatorConfig:
        """
        Calibrates model configurations based on historical data.
        Returns a new SimulatorConfig containing the calibrated parameters.
        """
        # 1. Calibrate Ornstein-Uhlenbeck (OU) Inflation parameters
        inf_params = InflationCalibrator.calibrate(historical_inflation, dt=self.config.dt)

        # 2. Calibrate CIR Short Rate Parameters
        if historical_yield_curve is not None and tenors is not None:
            # Match yield curve structure
            initial_rate = historical_rates[-1]
            rate_params = RatesCalibrator.fit_yield_curve_to_target(
                target_yields=historical_yield_curve,
                tenors=tenors,
                initial_rate=initial_rate
            )
        else:
            # Fallback to time-series analysis
            rate_params = RatesCalibrator.calibrate_short_rate_series(historical_rates, dt=self.config.dt)

        # 3. Calibrate Merton Jump Diffusion parameters
        eq_params = EquityCalibrator.calibrate(historical_equity_returns, dt=self.config.dt)

        # 4. Construct a new configuration inheriting non-calibrated variables
        new_config = SimulatorConfig(
            duration_years=self.config.duration_years,
            num_scenarios=self.config.num_scenarios,
            seed=self.config.seed,
            tenors=self.config.tenors,
            mu_min=self.config.mu_min,
            pi_min=self.config.pi_min,
            alpha_smooth=self.config.alpha_smooth,
            beta_drag=self.config.beta_drag,
            eta_erp=self.config.eta_erp,
            lambda_irp=self.config.lambda_irp,
            kappa_irp=self.config.kappa_irp,
            initial_rate=float(historical_rates[-1]),
            initial_inflation=float(historical_inflation[-1]),

            # Calibrated variables
            ou_mu=inf_params["ou_mu"],
            lambda_J=eq_params["lambda_J"],
            mu_J=eq_params["mu_J"],
            sigma_J=eq_params["sigma_J"]
        )

        # Storing directly as attributes to allow the Simulator to read them
        new_config.cir_theta_val = rate_params["cir_theta"]
        new_config.cir_sigma_val = rate_params["cir_sigma"]
        new_config.cir_mu_val = rate_params["cir_mu"]

        new_config.ou_theta_val = inf_params["ou_theta"]
        new_config.ou_sigma_val = inf_params["ou_sigma"]

        new_config.gbm_sigma_val = eq_params["gbm_sigma"]

        return new_config
