import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class EquityCalibrator:
    """
    Calibrates Merton Jump Diffusion parameters using Maximum Likelihood Estimation (MLE) 
    from historical asset return paths.
    """

    @staticmethod
    def calibrate(historical_returns: np.ndarray, dt: float = 1.0 / 12.0) -> dict:
        """
        Finds drift, diffusion, and jump parameters using historical returns.
        Uses a truncated infinite series representation of the Merton density function.
        """
        returns = np.asarray(historical_returns, dtype=np.float64)
        log_returns = np.log(1.0 + returns)
        
        if len(log_returns) < 10:
            raise ValueError("Merton MLE requires a minimum of 10 return observations.")

        # Numerical approximation truncation level for Poisson jump events
        K_max = 5 

        # Parameters to fit: [mu, sigma (diffusion), lambda_J, mu_J, sigma_J]
        def negative_log_likelihood(params):
            mu, sigma, lmbda, mu_J, sigma_J = params
            
            # Boundary constraints for physical realism
            if sigma <= 0.001 or lmbda < 0.0 or sigma_J <= 0.001:
                return 1e10
            if lmbda > 5.0:  # Restrict extreme unphysical jump rates
                return 1e10

            # Calculate Merton density for each observation
            likelihood_sum = np.zeros_like(log_returns)
            for k in range(K_max):
                poisson_prob = np.exp(-lmbda * dt) * ((lmbda * dt) ** k) / math.factorial(k)
                
                # Distribution parameters conditional on k jumps
                mean_k = (mu - 0.5 * sigma**2 - lmbda * (np.exp(mu_J + 0.5 * sigma_J**2) - 1.0)) * dt + k * mu_J
                variance_k = (sigma**2) * dt + k * (sigma_J**2)
                
                likelihood_sum += poisson_prob * norm.pdf(log_returns, loc=mean_k, scale=np.sqrt(variance_k))

            # Numerical stability adjustment
            likelihood_sum = np.maximum(likelihood_sum, 1e-12)
            return -np.sum(np.log(likelihood_sum))

        # Initial assumptions via standard sample moments
        sample_mean = np.mean(log_returns) / dt
        sample_std = np.std(log_returns) / np.sqrt(dt)

        # Assumptions: [mu, sigma, lambda_J, mu_J, sigma_J]
        init_guess = [sample_mean, sample_std * 0.80, 0.20, -0.10, 0.10]
        bounds = [
            (-0.5, 0.5),          # Drift range
            (0.01, 0.5),          # Continuous volatility
            (0.0, 2.0),           # Annual jump rate (0 to 2 occurrences per year)
            (-0.4, 0.1),          # Average jump return impact
            (0.01, 0.3)           # Jump impact uncertainty
        ]

        res = minimize(negative_log_likelihood, init_guess, method="L-BFGS-B", bounds=bounds)

        if not res.success:
            print("Warning: Fallback applied, solver failed to converge")
            # FALLBACK
            # Identify "outlier" months that are likely jump events (e.g., movements > 1.5 standard deviations)
            threshold = 1.5 * sample_std
            jumps = log_returns[np.abs(log_returns - np.mean(log_returns)) > threshold]
            
            if len(jumps) > 0:
                # Estimate jump characteristics directly from these outlier events
                est_lambda = len(jumps) / (len(log_returns) * dt)
                est_mu_J = float(np.mean(jumps))
                est_sigma_J = float(np.clip(np.std(jumps), 0.01, 0.20))
            else:
                # Defaults if no outliers are found
                est_lambda = 0.20
                est_mu_J = -0.10
                est_sigma_J = 0.05

            return {
                "gbm_sigma": float(sample_std * 0.90), # Leave room for the jump variance
                "lambda_J": float(np.clip(est_lambda, 0.05, 1.0)),
                "mu_J": float(np.clip(est_mu_J, -0.30, 0.0)),
                "sigma_J": est_sigma_J
            }

        return {
            "gbm_sigma": float(res.x[1]),
            "lambda_J": float(res.x[2]),
            "mu_J": float(res.x[3]),
            "sigma_J": float(res.x[4])
        }