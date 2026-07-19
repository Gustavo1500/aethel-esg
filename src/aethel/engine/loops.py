import numpy as np

try:
    from numba import njit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

if HAS_NUMBA:
    @njit(fastmath=True, cache=True, nogil=True)
    def run_simulation_loop_numba(
        steps, num_scenarios, dt,
        Z_corr, jump_shocks_all,
        ou_mu, pi_target, r_real_target, gamma, base_erp,
        cir_theta_dt, cir_sigma_sqrt_dt, ou_theta_dt, ou_sigma_sqrt_dt,
        gbm_sigma_sqrt_dt, drift_adjustment,
        rate_paths, inflation_paths, y_paths, smoothed_inflation_paths,
        mu_rate_paths, y_target_paths, equity_returns,
        mu_min, pi_min, beta_drag, alpha_smooth, eta_erp
    ):
        for t in range(steps):
            Z_t = Z_corr[t]
            rate_t = rate_paths[t]
            y_t = y_paths[t]
            smoothed_inflation_t = smoothed_inflation_paths[t]

            for i in range(num_scenarios):
                r_t = rate_t[i]
                s_inflation_t = smoothed_inflation_t[i]

                # Policy Short Rate Target
                mu_r = r_real_target[i] + s_inflation_t + gamma[i] * (s_inflation_t - pi_target[i])
                if mu_r < mu_min:
                    mu_r = mu_min
                mu_rate_paths[t, i] = mu_r

                # CIR Step (Short Rate)
                rate_next = (
                    r_t
                    + cir_theta_dt[i] * (mu_r - r_t)
                    + cir_sigma_sqrt_dt[i] * np.sqrt(max(1e-5, r_t)) * Z_t[0, i]
                )
                if rate_next < 0.001:
                    rate_next = 0.001
                rate_paths[t+1, i] = rate_next

                # Monetary Drag & Inflation State Target
                real_rate_gap = r_t - s_inflation_t
                y_target = (ou_mu[i] - pi_min) - beta_drag * real_rate_gap
                if y_target < 0.0:
                    y_target = 0.0
                y_target_paths[t, i] = y_target

                # Shifted-CIR Step
                y_val = y_t[i]
                y_next = (
                    y_val
                    + ou_theta_dt[i] * (y_target - y_val)
                    + ou_sigma_sqrt_dt[i] * np.sqrt(max(1e-5, y_val)) * Z_t[1, i]
                )
                if y_next < 0.0:
                    y_next = 0.0
                y_paths[t+1, i] = y_next

                inflation_next = y_next + pi_min
                inflation_paths[t+1, i] = inflation_next

                # Update smoothed inflation
                s_inflation_next = alpha_smooth * inflation_next + (1.0 - alpha_smooth) * s_inflation_t
                smoothed_inflation_paths[t+1, i] = s_inflation_next

                # Equity dynamic ERP
                dynamic_erp = base_erp[i] + eta_erp * (s_inflation_t - pi_target[i])
                if dynamic_erp < 0.001:
                    dynamic_erp = 0.001
                drift_dt = (r_t + dynamic_erp) * dt

                # Equity returns using pre-generated jump shocks
                equity_returns[t, i] = np.exp(
                    drift_dt
                    + drift_adjustment[i]
                    + gbm_sigma_sqrt_dt[i] * Z_t[2, i]
                    + jump_shocks_all[t, i]
                ) - 1.0
else:
    run_simulation_loop_numba = None


def run_simulation_loop_numpy(
    steps, num_scenarios, dt,
    Z_corr, jump_shocks_all,
    ou_mu, pi_target, r_real_target, gamma, base_erp,
    cir_theta_dt, cir_sigma_sqrt_dt, ou_theta_dt, ou_sigma_sqrt_dt,
    gbm_sigma_sqrt_dt, drift_adjustment,
    rate_paths, inflation_paths, y_paths, smoothed_inflation_paths,
    mu_rate_paths, y_target_paths, equity_returns,
    mu_min, pi_min, beta_drag, alpha_smooth, eta_erp
):
    """Fallback loop using vectorized NumPy arrays."""
    for t in range(steps):
        Z_t = Z_corr[t]

        rate_t = rate_paths[t]
        y_t = y_paths[t]
        smoothed_inflation_t = smoothed_inflation_paths[t]

        mu_rate_t = np.maximum(
            mu_min,
            r_real_target + smoothed_inflation_t + gamma * (smoothed_inflation_t - pi_target)
        )
        mu_rate_paths[t] = mu_rate_t

        rate_next = (
            rate_t
            + cir_theta_dt * (mu_rate_t - rate_t)
            + cir_sigma_sqrt_dt * np.sqrt(np.maximum(1e-5, rate_t)) * Z_t[0]
        )
        rate_paths[t+1] = np.maximum(0.001, rate_next)

        real_rate_gap_t = rate_t - smoothed_inflation_t
        y_target_t = np.maximum(0.0, (ou_mu - pi_min) - beta_drag * real_rate_gap_t)
        y_target_paths[t] = y_target_t

        y_next = (
            y_t
            + ou_theta_dt * (y_target_t - y_t)
            + ou_sigma_sqrt_dt * np.sqrt(np.maximum(1e-5, y_t)) * Z_t[1]
        )
        y_paths[t+1] = np.maximum(0.0, y_next)
        inflation_paths[t+1] = y_paths[t+1] + pi_min

        smoothed_inflation_paths[t+1] = (
            alpha_smooth * inflation_paths[t+1]
            + (1.0 - alpha_smooth) * smoothed_inflation_t
        )

        dynamic_erp_t = np.maximum(0.001, base_erp + eta_erp * (smoothed_inflation_t - pi_target))
        drift_t_dt = (rate_t + dynamic_erp_t) * dt

        equity_returns[t] = np.exp(
            drift_t_dt
            + drift_adjustment
            + gbm_sigma_sqrt_dt * Z_t[2]
            + jump_shocks_all[t]
        ) - 1.0
    