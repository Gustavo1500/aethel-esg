import os
import sys
import numpy as np
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor

from aethel.engine.parameters import SimulatorConfig
from aethel.engine.loops import (
    HAS_NUMBA,
    run_simulation_loop_numba,
    run_simulation_loop_numpy
)


def get_available_system_ram() -> int:
    """
    Detects available system RAM in bytes using standard library fallback logic.
    Returns 4 GB (4,294,967,296 bytes) as a safe fallback if OS queries fail.
    """
    # 1. Windows Detection (GlobalMemoryStatusEx)
    if sys.platform.startswith("win"):
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_uint64),
                    ("ullAvailPhys", ctypes.c_uint64),
                    ("ullTotalPageFile", ctypes.c_uint64),
                    ("ullAvailPageFile", ctypes.c_uint64),
                    ("ullTotalVirtual", ctypes.c_uint64),
                    ("ullAvailVirtual", ctypes.c_uint64),
                    ("ullAvailExtendedVirtual", ctypes.c_uint64)
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullAvailPhys
        except Exception:
            pass

    # 2. Linux Detection (/proc/meminfo)
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if "MemAvailable" in line:
                        return int(line.split()[1]) * 1024  # KB to Bytes
                    if "MemFree" in line:
                        return int(line.split()[1]) * 1024
        except Exception:
            pass

    # 3. macOS Detection (sysctl hw.memsize)
    elif sys.platform.startswith("darwin"):
        try:
            import subprocess
            res = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
            total_mem = int(res.strip())
            return int(total_mem * 0.5)
        except Exception:
            pass

    # 4. Fallback Default (4 GB of RAM)
    return 4 * 1024 * 1024 * 1024


class LazyScenarioList:
    """
    A list-like adapter wrapping pre-allocated contiguous arrays.
    Saves memory by loading or slicing array regions on demand,
    fully matching standard Python list expectations.
    """
    def __init__(
        self,
        equity_returns: np.ndarray,
        cpis: np.ndarray,
        deposit_rates: np.ndarray,
        rate_paths: np.ndarray,
        mu_rate_paths: np.ndarray,
        inflation_paths: np.ndarray,
        y_target_paths: np.ndarray,
        cir_theta: np.ndarray,
        cir_sigma: np.ndarray,
        ou_theta: np.ndarray,
        ou_sigma: np.ndarray,
        tenors: np.ndarray,
        pi_min: float,
        lambda_irp: float,
        kappa_irp: float
    ):
        self.equity_returns = equity_returns
        self.cpis = cpis
        self.deposit_rates = deposit_rates
        self.rate_paths = rate_paths
        self.mu_rate_paths = mu_rate_paths
        self.inflation_paths = inflation_paths
        self.y_target_paths = y_target_paths
        self.cir_theta = cir_theta
        self.cir_sigma = cir_sigma
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.tenors = tenors
        self.pi_min = pi_min
        self.lambda_irp = lambda_irp
        self.kappa_irp = kappa_irp
        self.num_scenarios = len(equity_returns)

    def __len__(self) -> int:
        return self.num_scenarios

    def __getitem__(self, idx: Union[int, slice]) -> Any:
        if isinstance(idx, slice):
            start, stop, step = idx.indices(self.num_scenarios)
            return [self[i] for i in range(start, stop, step)]

        if idx < 0:
            idx += self.num_scenarios
        if idx < 0 or idx >= self.num_scenarios:
            raise IndexError("Scenario index out of range.")

        # Dynamically generate yield curves for the requested scenario index on-the-fly
        nominal_yields = self._generate_scenario_nominal_yields(idx)
        real_yields = self._generate_scenario_real_yields(idx, nominal_yields)

        return {
            "stock_returns": self.equity_returns[idx],
            "cpis": self.cpis[idx],
            "deposit_rates": self.deposit_rates[idx],
            "nominal_yield_curves": nominal_yields,
            "real_yield_curves": real_yields,
            "tenors": self.tenors
        }

    def __iter__(self):
        for i in range(self.num_scenarios):
            yield self[i]

    def cleanup(self) -> None:
        """Symmetrical clean-up method."""
        pass

    def _generate_scenario_nominal_yields(self, idx: int) -> np.ndarray:
        r_path = self.rate_paths[:, idx]
        mu_path = self.mu_rate_paths[:, idx]
        theta = self.cir_theta[idx]
        sigma = self.cir_sigma[idx]
        tenors = self.tenors

        h = np.sqrt(theta ** 2 + 2.0 * (sigma ** 2))
        denominator = (theta + h) * (np.exp(h * tenors) - 1.0) + 2.0 * h
        base_A = (2.0 * h * np.exp((theta + h) * tenors / 2.0)) / denominator
        B_tau = (2.0 * (np.exp(h * tenors) - 1.0)) / denominator

        log_base_A_div_tau = np.log(base_A) / tenors
        B_tau_div_tau = B_tau / tenors
        safe_sigma_sq = max(1e-6, sigma ** 2)
        power_factor = (2.0 * theta * mu_path) / safe_sigma_sq

        yields = r_path[:, np.newaxis] * B_tau_div_tau[np.newaxis, :]
        yields -= power_factor[:, np.newaxis] * log_base_A_div_tau[np.newaxis, :]
        return yields

    def _generate_scenario_real_yields(self, idx: int, nominal_yields: np.ndarray) -> np.ndarray:
        inflation_path = self.inflation_paths[:, idx]
        mu_local_path = self.y_target_paths[:, idx] + self.pi_min
        theta = self.ou_theta[idx]
        sigma = self.ou_sigma[idx]
        tenors = self.tenors

        theta_tau = theta * tenors
        factor = np.where(
            theta_tau > 1e-4,
            (1.0 - np.exp(-theta_tau)) / theta_tau,
            1.0 - 0.5 * theta_tau + (theta_tau ** 2) / 6.0
        )

        diff = inflation_path - mu_local_path
        irp = (self.lambda_irp * sigma) * (1.0 - np.exp(-self.kappa_irp * tenors))

        yields_real = nominal_yields - mu_local_path[:, np.newaxis]
        yields_real -= diff[:, np.newaxis] * factor[np.newaxis, :]
        yields_real -= irp[np.newaxis, :]
        return yields_real

    def __repr__(self) -> str:
        return f"LazyScenarioList(scenarios={self.num_scenarios}, steps={self.equity_returns.shape[1]})"


class MarketSimulator:
    """
    An Economic Scenario Generator (ESG) that orchestrates parameters,
    concurrent random path generation, and analytical yield derivation.
    """
    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config if config is not None else SimulatorConfig()

    def run(self) -> LazyScenarioList:
        """
        Executes the simulation engine, dynamically choosing between single-block
        or memory-safe chunked execution based on available host RAM.
        """
        cfg = self.config

        # 1. Deterministic generation of scenario-level parameters
        master_rng = np.random.default_rng(cfg.seed)

        ou_mu_mean = cfg.ou_mu if cfg.ou_mu is not None else 0.055
        ou_mu = np.clip(master_rng.normal(ou_mu_mean, 0.008, cfg.num_scenarios), ou_mu_mean - 0.02, ou_mu_mean + 0.02)
        pi_target = ou_mu

        r_real_mean = cfg.cir_mu_val - ou_mu_mean if (cfg.cir_mu_val is not None) else 0.050
        r_real_mean = np.clip(r_real_mean, 0.02, 0.08)
        r_real_target = np.clip(master_rng.normal(r_real_mean, 0.008, cfg.num_scenarios), r_real_mean - 0.02, r_real_mean + 0.02)

        gamma = master_rng.uniform(0.3, 0.7, cfg.num_scenarios)
        base_erp = master_rng.uniform(0.005, 0.025, cfg.num_scenarios)

        cir_sigma_mean = cfg.cir_sigma_val if cfg.cir_sigma_val is not None else 0.08
        cir_sigma = np.clip(master_rng.normal(cir_sigma_mean, 0.01, cfg.num_scenarios), cir_sigma_mean - 0.02, cir_sigma_mean + 0.02)

        ou_sigma_mean = cfg.ou_sigma_val if cfg.ou_sigma_val is not None else 0.015
        ou_sigma = np.clip(master_rng.normal(ou_sigma_mean, 0.003, cfg.num_scenarios), ou_sigma_mean - 0.005, ou_sigma_mean + 0.007)

        gbm_sigma_mean = cfg.gbm_sigma_val if cfg.gbm_sigma_val is not None else 0.18
        gbm_sigma = np.clip(master_rng.normal(gbm_sigma_mean, 0.02, cfg.num_scenarios), gbm_sigma_mean - 0.04, gbm_sigma_mean + 0.04)

        cir_theta_val = cfg.cir_theta_val if cfg.cir_theta_val is not None else 0.25
        cir_theta = np.full(cfg.num_scenarios, cir_theta_val)

        ou_theta_val = cfg.ou_theta_val if cfg.ou_theta_val is not None else 0.35
        ou_theta = np.full(cfg.num_scenarios, ou_theta_val)

        # 2. Derive global random variables correlation matrix
        rho_12 = np.clip(master_rng.normal(0.40, 0.05), 0.25, 0.55)
        rho_13 = np.clip(master_rng.normal(-0.15, 0.05), -0.25, -0.05)
        rho_23 = np.clip(master_rng.normal(-0.10, 0.05), -0.20, 0.00)

        correlation_matrix = np.array([
            [1.00, rho_12, rho_13],
            [rho_12, 1.00, rho_23],
            [rho_13, rho_23, 1.00]
        ])
        L = np.linalg.cholesky(correlation_matrix)

        # 3. Dynamic Hardware-Aware Concurrency & Safety Buffer
        avail_ram = get_available_system_ram()

        os_safety_buffer = 4 * 1024 * 1024 * 1024
        usable_ram = max(1 * 1024 * 1024 * 1024, avail_ram - os_safety_buffer)

        peak_mem_needed = 136 * cfg.steps * cfg.num_scenarios

        if cfg.max_workers is not None:
            max_workers = cfg.max_workers
        else:
            system_cpus = os.cpu_count() or 1
            if usable_ram < 8 * 1024 * 1024 * 1024:
                max_workers = min(2, system_cpus)
            else:
                max_workers = min(4, system_cpus)

        # 4. Defensive Triggering Rules
        if cfg.chunk_size is not None:
            chunk_size = cfg.chunk_size
            use_chunking = True
            print(f"[ESG] Manual override: Chunked execution activated (chunk size: {chunk_size}).")
        else:
            use_chunking = (cfg.num_scenarios > 20000) or (peak_mem_needed > (0.25 * usable_ram))
            if use_chunking:
                temp_mem_per_scenario = 64 * cfg.steps
                total_temp_limit = 256 * 1024 * 1024
                temp_limit_per_worker = total_temp_limit / max_workers

                chunk_size = int(temp_limit_per_worker / temp_mem_per_scenario)
                chunk_size = min(5000, max(100, chunk_size))

                print(f"[ESG] High memory projection detected ({peak_mem_needed / (1024**3):.2f} GB estimated, {avail_ram / (1024**3):.2f} GB system available).")
                print(f"[ESG] Defensive dynamic chunking active (chunk size: {chunk_size}, active workers: {max_workers}).")
            else:
                chunk_size = cfg.num_scenarios

        num_chunks = int(np.ceil(cfg.num_scenarios / chunk_size))

        # 5. Symmetrical Master Outputs Pre-Allocation (In-Memory)
        rate_paths = np.zeros((cfg.steps + 1, cfg.num_scenarios), dtype=np.float64)
        inflation_paths = np.zeros((cfg.steps + 1, cfg.num_scenarios), dtype=np.float64)
        y_target_paths = np.zeros((cfg.steps + 1, cfg.num_scenarios), dtype=np.float64)
        mu_rate_paths = np.zeros((cfg.steps + 1, cfg.num_scenarios), dtype=np.float64)

        equity_returns = np.zeros((cfg.steps, cfg.num_scenarios), dtype=np.float64)
        cpis = np.zeros((cfg.steps, cfg.num_scenarios), dtype=np.float64)
        deposit_rates = np.zeros((cfg.steps, cfg.num_scenarios), dtype=np.float64)

        # 6. Dispatch threads filling master array views directly
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for j in range(num_chunks):
                start_idx = j * chunk_size
                end_idx = min(start_idx + chunk_size, cfg.num_scenarios)

                futures.append(
                    executor.submit(
                        self._run_chunk,
                        start_idx=start_idx,
                        end_idx=end_idx,
                        seed_offset=cfg.seed + start_idx,
                        ou_mu_chunk=ou_mu[start_idx:end_idx],
                        r_real_target_chunk=r_real_target[start_idx:end_idx],
                        gamma_chunk=gamma[start_idx:end_idx],
                        base_erp_chunk=base_erp[start_idx:end_idx],
                        cir_sigma_chunk=cir_sigma[start_idx:end_idx],
                        ou_sigma_chunk=ou_sigma[start_idx:end_idx],
                        gbm_sigma_chunk=gbm_sigma[start_idx:end_idx],
                        cir_theta_chunk=cir_theta[start_idx:end_idx],
                        ou_theta_chunk=ou_theta[start_idx:end_idx],
                        L=L,
                        equity_returns_view=equity_returns[:, start_idx:end_idx],
                        cpis_view=cpis[:, start_idx:end_idx],
                        deposit_rates_view=deposit_rates[:, start_idx:end_idx],
                        rate_paths_view=rate_paths[:, start_idx:end_idx],
                        inflation_paths_view=inflation_paths[:, start_idx:end_idx],
                        y_target_paths_view=y_target_paths[:, start_idx:end_idx],
                        mu_rate_paths_view=mu_rate_paths[:, start_idx:end_idx]
                    )
                )

            for fut in futures:
                fut.result()

        print("[ESG] Core processing phase complete.")
        return LazyScenarioList(
            equity_returns=np.ascontiguousarray(equity_returns.T),
            cpis=np.ascontiguousarray(cpis.T),
            deposit_rates=np.ascontiguousarray(deposit_rates.T),
            rate_paths=rate_paths,
            mu_rate_paths=mu_rate_paths,
            inflation_paths=inflation_paths,
            y_target_paths=y_target_paths,
            cir_theta=cir_theta,
            cir_sigma=cir_sigma,
            ou_theta=ou_theta,
            ou_sigma=ou_sigma,
            tenors=cfg.tenors,
            pi_min=cfg.pi_min,
            lambda_irp=cfg.lambda_irp,
            kappa_irp=cfg.kappa_irp
        )

    def _run_chunk(
        self,
        start_idx: int,
        end_idx: int,
        seed_offset: int,
        ou_mu_chunk: np.ndarray,
        r_real_target_chunk: np.ndarray,
        gamma_chunk: np.ndarray,
        base_erp_chunk: np.ndarray,
        cir_sigma_chunk: np.ndarray,
        ou_sigma_chunk: np.ndarray,
        gbm_sigma_chunk: np.ndarray,
        cir_theta_chunk: np.ndarray,
        ou_theta_chunk: np.ndarray,
        L: np.ndarray,
        equity_returns_view: np.ndarray,
        cpis_view: np.ndarray,
        deposit_rates_view: np.ndarray,
        rate_paths_view: np.ndarray,
        inflation_paths_view: np.ndarray,
        y_target_paths_view: np.ndarray,
        mu_rate_paths_view: np.ndarray
    ):
        """Worker thread compiling and running an isolated block of scenarios."""
        cfg = self.config
        chunk_size = end_idx - start_idx
        sqrt_dt = np.sqrt(cfg.dt)

        Z_raw_flat = np.empty((3, cfg.steps * chunk_size))
        jump_shocks_all = np.zeros((cfg.steps, chunk_size), dtype=np.float64)

        for i in range(chunk_size):
            scenario_seed = seed_offset + i
            local_rng = np.random.default_rng(scenario_seed)

            Z_raw_flat[:, i * cfg.steps : (i + 1) * cfg.steps] = local_rng.normal(0.0, 1.0, (3, cfg.steps))

            jumps = local_rng.poisson(cfg.lambda_J * cfg.dt, cfg.steps)
            active_idx = np.where(jumps > 0)[0]
            for t_step in active_idx:
                num_jumps = jumps[t_step]
                jump_shocks_all[t_step, i] = np.sum(local_rng.normal(cfg.mu_J, cfg.sigma_J, num_jumps))

        Z_corr = (L @ Z_raw_flat).reshape(3, chunk_size, cfg.steps).transpose(2, 0, 1)
        Z_corr = np.ascontiguousarray(Z_corr)

        y_paths = np.empty((cfg.steps + 1, chunk_size))
        smoothed_inflation_paths = np.empty((cfg.steps + 1, chunk_size))

        # Direct writes initialization into master view pointers
        rate_paths_view[0] = cfg.initial_rate
        inflation_paths_view[0] = cfg.initial_inflation
        y_paths[0] = inflation_paths_view[0] - cfg.pi_min
        smoothed_inflation_paths[0] = inflation_paths_view[0]

        cir_theta_dt = cir_theta_chunk * cfg.dt
        cir_sigma_sqrt_dt = cir_sigma_chunk * sqrt_dt
        ou_theta_dt = ou_theta_chunk * cfg.dt
        ou_sigma_sqrt_dt = ou_sigma_chunk * sqrt_dt

        gbm_sigma_sq_dt = 0.5 * (gbm_sigma_chunk ** 2) * cfg.dt
        gbm_sigma_sqrt_dt = gbm_sigma_chunk * sqrt_dt
        drift_adjustment = - (cfg.lambda_J * cfg.k_jump) * cfg.dt - gbm_sigma_sq_dt

        if HAS_NUMBA:
            run_simulation_loop_numba(
                cfg.steps, chunk_size, cfg.dt,
                Z_corr, jump_shocks_all,
                ou_mu_chunk, ou_mu_chunk, r_real_target_chunk, gamma_chunk, base_erp_chunk,
                cir_theta_dt, cir_sigma_sqrt_dt, ou_theta_dt, ou_sigma_sqrt_dt,
                gbm_sigma_sqrt_dt, drift_adjustment,
                rate_paths_view, inflation_paths_view, y_paths, smoothed_inflation_paths,
                mu_rate_paths_view, y_target_paths_view, equity_returns_view,
                cfg.mu_min, cfg.pi_min, cfg.beta_drag, cfg.alpha_smooth, cfg.eta_erp
            )
        else:
            run_simulation_loop_numpy(
                cfg.steps, chunk_size, cfg.dt,
                Z_corr, jump_shocks_all,
                ou_mu_chunk, ou_mu_chunk, r_real_target_chunk, gamma_chunk, base_erp_chunk,
                cir_theta_dt, cir_sigma_sqrt_dt, ou_theta_dt, ou_sigma_sqrt_dt,
                gbm_sigma_sqrt_dt, drift_adjustment,
                rate_paths_view, inflation_paths_view, y_paths, smoothed_inflation_paths,
                mu_rate_paths_view, y_target_paths_view, equity_returns_view,
                cfg.mu_min, cfg.pi_min, cfg.beta_drag, cfg.alpha_smooth, cfg.eta_erp
            )

        mu_rate_paths_view[cfg.steps] = mu_rate_paths_view[cfg.steps - 1]
        y_target_paths_view[cfg.steps] = y_target_paths_view[cfg.steps - 1]

        # Direct in-place writes for monthly outputs (Zero memory copies)
        deposit_rates_view[:] = (1.0 + np.maximum(-0.99, rate_paths_view[:-1, :])) ** (1.0 / 12.0) - 1.0
        inflation_monthly_all = (1.0 + np.maximum(-0.99, inflation_paths_view[:-1, :])) ** (1.0 / 12.0) - 1.0
        cpis_view[:] = np.cumprod(1.0 + inflation_monthly_all, axis=0)
