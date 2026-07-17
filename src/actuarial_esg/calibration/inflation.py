import numpy as np


class InflationCalibrator:
    """
    Calibrates Shifted-CIR / Ornstein-Uhlenbeck (OU) inflation parameters 
    using historical monthly inflation (IPCA) series via OLS AR(1) mapping.
    """
    
    @staticmethod
    def calibrate(historical_ipca: np.ndarray, dt: float = 1.0 / 12.0) -> dict:
        """
        Fits an AR(1) process to the inflation series and maps the parameters 
        to continuous-time Ornstein-Uhlenbeck parameters.
        """
        ipca = np.asarray(historical_ipca, dtype=np.float64)
        if len(ipca) < 3:
            raise ValueError("Historical inflation series must contain at least 3 historical points.")

        # Prepare lagged series
        x_t = ipca[1:]
        x_lag = ipca[:-1]

        # Fit OLS: x_t = a + b * x_lag + e
        poly = np.polyfit(x_lag, x_t, deg=1)
        b = poly[0]
        a = poly[1]

        residuals = x_t - (a + b * x_lag)
        residual_var = np.var(residuals, ddof=2)

        # Enforce stability constraints
        if b <= 0.0 or b >= 1.0:
            # Fallback to realistic bounds if non-stationary
            b = np.clip(b, 0.01, 0.99)

        # Map AR(1) to continuous-time OU
        theta_ou = -np.log(b) / dt
        mu_ou = a / (1.0 - b)
        
        # Continuous variance mapping
        sigma_ou_sq = residual_var * (2.0 * theta_ou) / (1.0 - b**2)
        sigma_ou = np.sqrt(np.maximum(1e-6, sigma_ou_sq))

        return {
            "ou_theta": float(theta_ou),
            "ou_mu": float(mu_ou),
            "ou_sigma": float(sigma_ou)
        }