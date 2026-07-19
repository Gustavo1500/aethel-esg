import numpy as np
from scipy.optimize import minimize


class RatesCalibrator:
    """
    Calibrates CIR short-rate parameter matrices and expectations
    using historical short-rates and yield curve term structures.
    """

    @staticmethod
    def calibrate_short_rate_series(historical_rates: np.ndarray, dt: float = 1.0 / 12.0) -> dict:
        """
        Performs discrete-time estimation of the CIR drift and diffusion
        parameters based on historical short rate transitions.
        """
        r = np.asarray(historical_rates, dtype=np.float64)
        if len(r) < 3:
            raise ValueError("Historical short rate series must have at least 3 observations.")

        # CIR: dx_t = theta * (mu - x_t) * dt + sigma * sqrt(x_t) * dW_t
        r_t = r[:-1]
        r_next = r[1:]

        y = (r_next - r_t) / np.sqrt(np.maximum(1e-5, r_t))
        x1 = dt / np.sqrt(np.maximum(1e-5, r_t))
        x2 = -dt * np.sqrt(np.maximum(1e-5, r_t))

        X = np.column_stack((x1, x2))

        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        theta_mu = beta[0]
        theta = beta[1]

        if theta <= 0.0:
            theta = 0.25
        mu = np.maximum(0.01, theta_mu / theta)

        predicted = theta * (mu - r_t) * dt
        actual_diff = r_next - r_t
        residuals = (actual_diff - predicted) / np.sqrt(np.maximum(1e-5, r_t))
        sigma = np.sqrt(np.var(residuals, ddof=1) / dt)

        return {
            "cir_theta": float(theta),
            "cir_mu": float(mu),
            "cir_sigma": float(np.clip(sigma, 0.01, 0.20))
        }

    @staticmethod
    def fit_yield_curve_to_target(
        target_yields: np.ndarray,
        tenors: np.ndarray,
        initial_rate: float
    ) -> dict:
        """
        Calibrates CIR parameters by matching analytical yield formulas
        to an observed target yield curve using optimization.
        """
        target = np.asarray(target_yields, dtype=np.float64)
        tau = np.asarray(tenors, dtype=np.float64)

        def objective(params):
            theta, mu, sigma = params
            if theta <= 0.01 or mu <= 0.001 or sigma <= 0.001:
                return 1e10

            h = np.sqrt(theta**2 + 2.0 * sigma**2)
            denominator = (theta + h) * (np.exp(h * tau) - 1.0) + 2.0 * h
            base_A = (2.0 * h * np.exp((theta + h) * tau / 2.0)) / denominator
            B_tau = (2.0 * (np.exp(h * tau) - 1.0)) / denominator

            log_base_A_div_tau = np.log(base_A) / tau
            B_tau_div_tau = B_tau / tau

            power_factor = (2.0 * theta * mu) / (sigma**2)
            model_yields = initial_rate * B_tau_div_tau - power_factor * log_base_A_div_tau

            return np.mean((target - model_yields) ** 2)

        init_guess = [0.25, np.mean(target), 0.08]
        bounds = [(0.01, 2.0), (0.005, 0.25), (0.005, 0.25)]

        res = minimize(objective, init_guess, method="L-BFGS-B", bounds=bounds)
        if not res.success:
            return {"cir_theta": 0.25, "cir_mu": float(np.mean(target)), "cir_sigma": 0.08}

        return {
            "cir_theta": float(res.x[0]),
            "cir_mu": float(res.x[1]),
            "cir_sigma": float(res.x[2])
        }
    