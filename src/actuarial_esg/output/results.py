import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Callable

from . import time_utils, portfolio, decumulation, visualizer


class SimulationResults:
    """
    A unified analysis, post-processing, and visualization engine for ESG outputs.
    Maintains a robust state-caching framework and delegates operations to submodules.
    """

    def __init__(self, raw_scenarios: List[Dict[str, np.ndarray]]):
        if not raw_scenarios:
            raise ValueError("Scenario list cannot be empty.")

        self.scenarios = raw_scenarios
        self.num_scenarios = len(raw_scenarios)
        self.steps = len(raw_scenarios[0]["stock_returns"])
        self.tenors = list(raw_scenarios[0]["tenors"])
        self.steps_per_year = 12

        # Performance Optimization: Cache base matrices to prevent redundant calculations
        self._cache: Dict[str, np.ndarray] = {}

    def query(
        self,
        metric: str,
        stat: str = "raw",
        time: Optional[Union[str, float, int, List[float]]] = None,
        year: Optional[Union[str, float, int, List[float]]] = None,
        month: Optional[Union[str, int, List[int]]] = None,
        step: Optional[Union[str, int, List[int]]] = None,
        tenor: Optional[float] = None,
        annualized: bool = False,
        portfolio_weights: Optional[Dict[str, float]] = None
    ) -> Union[np.ndarray, float]:
        """
        Extracts, transforms, and calculates statistical parameters from the simulation.
        """
        # 1. Fetch base 2D matrix (steps, scenarios)
        matrix = self._extract_base_matrix(metric, tenor, portfolio_weights)

        # 2. Apply annualization transformations if requested
        if annualized:
            matrix = time_utils.apply_annualization(matrix, metric, self.steps_per_year)

        # 3. Resolve time inputs to indices
        step_indices = time_utils.resolve_time_to_indices(
            time_legacy=time,
            year=year,
            month=month,
            step=step,
            max_idx=len(matrix) - 1,
            steps_per_year=self.steps_per_year
        )

        # 4. Handle time horizon slicing
        if step_indices is not None:
            matrix = matrix[step_indices, :]

        # 5. Collapse dimension 1 (scenarios) according to the requested statistic
        return self._apply_statistics(matrix, stat)

    # --- Portfolio Blending Logic (Delegated) ---

    def calculate_portfolio_returns(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Calculates the blended monthly return series for a given portfolio allocation.
        """
        return portfolio.calculate_portfolio_returns(self, weights)

    def calculate_portfolio_growth(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Calculates the blended cumulative growth series of $1.00 for a given portfolio allocation.
        """
        return portfolio.calculate_portfolio_growth(self, weights)

    # --- Decumulation (Withdrawal) Simulator (Delegated) ---

    def simulate_decumulation(
        self,
        initial_balance: float,
        initial_monthly_withdrawal: float,
        portfolio_weights: Dict[str, float],
        withdrawal_timing: str = "beginning",
        inflate_withdrawals: bool = True,
        frictional_drag_annual: float = 0.0,
        tax_on_gains_rate: float = 0.0,
        liquidation_strategy: str = "constant_mix",
        withdrawal_policy: Optional[Callable[[np.ndarray, np.ndarray, int, np.ndarray], np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Simulates the monthly decumulation of a starting balance.
        """
        return decumulation.simulate_decumulation(
            self,
            initial_balance,
            initial_monthly_withdrawal,
            portfolio_weights,
            withdrawal_timing,
            inflate_withdrawals,
            frictional_drag_annual,
            tax_on_gains_rate,
            liquidation_strategy,
            withdrawal_policy
        )

    # --- Optimized Base Matrix Retrieval (With Memoization) ---

    def _extract_base_matrix(
        self,
        metric: str,
        tenor: Optional[float] = None,
        portfolio_weights: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        m = metric.lower().strip()

        # Build stable cache key for portfolio blends
        if portfolio_weights is not None:
            sorted_weights = sorted(portfolio_weights.items())
            weights_str = "_".join([f"{k}:{v:.4f}" for k, v in sorted_weights])
            cache_key = f"{m}_{weights_str}"
        else:
            cache_key = f"{m}_{tenor}" if tenor is not None else m

        # Return cached array if already computed to avoid redundant loops
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check for optimized LazyScenarioList processing
        from actuarial_esg.engine.simulator import LazyScenarioList
        is_lazy = isinstance(self.scenarios, LazyScenarioList)

        if m == "portfolio_returns":
            res = self.calculate_portfolio_returns(portfolio_weights)

        elif m == "portfolio_growth":
            res = self.calculate_portfolio_growth(portfolio_weights)

        elif m == "decumulation_balance":
            if "decumulation_balance" not in self._cache:
                raise ValueError(
                    "Decumulation balance not simulated yet. Please run "
                    "simulate_decumulation(...) first to populate the cache."
                )
            return self._cache["decumulation_balance"]

        elif m == "decumulation_withdrawal":
            if "decumulation_withdrawal" not in self._cache:
                raise ValueError(
                    "Decumulation withdrawal not simulated yet. Please run "
                    "simulate_decumulation(...) first to populate the cache."
                )
            return self._cache["decumulation_withdrawal"]

        elif m in {"equity_returns", "returns"}:
            if is_lazy:
                res = self.scenarios.equity_returns.T
            else:
                res = np.column_stack([s["stock_returns"] for s in self.scenarios])

        elif m in {"equity_growth", "growth"}:
            returns = self._extract_base_matrix("returns")
            growth = np.vstack([np.zeros(self.num_scenarios), returns])
            res = np.cumprod(1.0 + growth, axis=0)

        elif m in {"cpi", "inflation_index"}:
            if is_lazy:
                res = self.scenarios.cpis.T
            else:
                res = np.column_stack([s["cpis"] for s in self.scenarios])

        elif m in {"inflation_rate", "inflation"}:
            cpis_matrix = self._extract_base_matrix("cpi")
            cpis_padded = np.vstack([np.ones(self.num_scenarios), cpis_matrix])
            with np.errstate(divide='ignore', invalid='ignore'):
                monthly_rates = (cpis_padded[1:, :] / cpis_padded[:-1, :]) - 1.0
            res = (1.0 + monthly_rates) ** 12 - 1.0

        elif m in {"short_rate", "cdi", "deposit_rates"}:
            if is_lazy:
                res = (1.0 + self.scenarios.deposit_rates.T) ** 12 - 1.0
            else:
                monthly_rates = np.column_stack([s["deposit_rates"] for s in self.scenarios])
                res = (1.0 + monthly_rates) ** 12 - 1.0

        elif m in {"nominal_yield", "real_yield"}:
            if tenor is None:
                raise ValueError(f"A 'tenor' must be specified when querying '{metric}'. Available: {self.tenors}")
            if tenor not in self.tenors:
                raise ValueError(f"Tenor {tenor} not available. Choose from: {self.tenors}")

            if is_lazy:
                if m == "real_yield":
                    res = self._generate_real_yields_for_tenor(tenor)[:-1, :]
                else:
                    res = self._generate_cir_yields_for_tenor(tenor)[:-1, :]
            else:
                tenor_idx = self.tenors.index(tenor)
                key = "real_yield_curves" if m == "real_yield" else "nominal_yield_curves"
                res = np.column_stack([s[key][:-1, tenor_idx] for s in self.scenarios])

        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose from correct keywords.")

        self._cache[cache_key] = res
        return res

    # --- On-The-Fly Vectorized Tenor Yield Derivations ---

    def _generate_cir_yields_for_tenor(self, tenor: float) -> np.ndarray:
        """Derives CIR yields for a single tenor across all scenarios instantly."""
        r = self.scenarios.cdi_paths.T
        mu = self.scenarios.mu_cdi_paths.T
        theta = self.scenarios.cir_theta
        sigma = self.scenarios.cir_sigma

        h = np.sqrt(theta ** 2 + 2.0 * (sigma ** 2))
        denominator = (theta + h) * (np.exp(h * tenor) - 1.0) + 2.0 * h
        base_A = (2.0 * h * np.exp((theta + h) * tenor / 2.0)) / denominator
        B_tau = (2.0 * (np.exp(h * tenor) - 1.0)) / denominator

        log_base_A_div_tenor = np.log(base_A) / tenor
        B_tau_div_tenor = B_tau / tenor
        safe_sigma_sq = np.maximum(1e-6, sigma ** 2)
        power_factor = (2.0 * theta[:, np.newaxis] * mu) / safe_sigma_sq[:, np.newaxis]

        yields = r * B_tau_div_tenor[:, np.newaxis]
        yields -= power_factor * log_base_A_div_tenor[:, np.newaxis]
        return yields.T

    def _generate_real_yields_for_tenor(self, tenor: float) -> np.ndarray:
        """Derives Fisher real yields for a single tenor across all scenarios on-the-fly."""
        yields_nominal = self._generate_cir_yields_for_tenor(tenor)

        ipca = self.scenarios.ipca_paths
        mu_local = self.scenarios.y_target_paths + self.scenarios.pi_min

        theta = self.scenarios.ou_theta
        sigma = self.scenarios.ou_sigma

        theta_tau = theta * tenor
        factor = np.where(
            theta_tau > 1e-4,
            (1.0 - np.exp(-theta_tau)) / theta_tau,
            1.0 - 0.5 * theta_tau + (theta_tau ** 2) / 6.0
        )

        diff = ipca - mu_local
        irp = (self.scenarios.lambda_irp * sigma) * (1.0 - np.exp(-self.scenarios.kappa_irp * tenor))

        yields_real = yields_nominal - mu_local
        yields_real -= diff * factor[np.newaxis, :]
        yields_real -= irp[np.newaxis, :]
        return yields_real

    def _apply_statistics(self, matrix: np.ndarray, stat: str) -> Union[np.ndarray, float]:
        s = stat.lower().strip()

        if s == "raw":
            return matrix[0] if matrix.shape[0] == 1 else matrix

        if s == "mean":
            aggregated = np.mean(matrix, axis=1)
        elif s == "median":
            aggregated = np.percentile(matrix, 50.0, axis=1)
        elif s in {"std", "vol", "volatility"}:
            aggregated = np.std(matrix, axis=1)
        elif s.startswith("p"):
            try:
                percentile_val = float(s[1:])
                if not (0.0 <= percentile_val <= 100.0):
                    raise ValueError
                aggregated = np.percentile(matrix, percentile_val, axis=1)
            except ValueError:
                raise ValueError(f"Could not parse valid percentile from '{stat}'.")
        else:
            raise ValueError(f"Unknown statistic '{stat}'.")

        return float(aggregated[0]) if len(aggregated) == 1 else aggregated

    # --- Backward Compatibility Methods ---

    def get_matrix(self, metric: str) -> np.ndarray:
        valid_map = {
            "stock_returns": "returns",
            "cpis": "cpi",
            "deposit_rates": "deposit_rates"
        }
        mapped = valid_map.get(metric)
        if not mapped:
            raise ValueError(f"Metric '{metric}' must be one of {list(valid_map.keys())}")
        return self._extract_base_matrix(mapped, tenor=None)

    def get_yield_curve_matrix(self, real: bool = False, tenor_idx: int = 0) -> np.ndarray:
        tenor = self.tenors[tenor_idx]
        metric = "real_yield" if real else "nominal_yield"
        return self._extract_base_matrix(metric, tenor=tenor)

    def get_summary_statistics(self, metric: str, percentiles: List[float] = [5.0, 50.0, 95.0]) -> Dict[str, np.ndarray]:
        valid_map = {"cpis": "cpi", "deposit_rates": "deposit_rates", "stock_returns": "returns"}
        mapped = valid_map.get(metric)
        if not mapped:
            raise ValueError(f"Metric must be one of {list(valid_map.keys())}")

        stats = {}
        for p in percentiles:
            stats[f"p{int(p)}"] = self.query(mapped, stat=f"p{p}", step="all")
        stats["mean"] = self.query(mapped, stat="mean", step="all")
        return stats

    def to_pandas(self, scenario_idx: int) -> pd.DataFrame:
        if scenario_idx < 0 or scenario_idx >= self.num_scenarios:
            raise IndexError(f"Scenario index must be between 0 and {self.num_scenarios - 1}")

        scenario = self.scenarios[scenario_idx]
        df = pd.DataFrame({
            "step": np.arange(self.steps),
            "stock_returns": scenario["stock_returns"],
            "cpis": scenario["cpis"],
            "deposit_rates": scenario["deposit_rates"]
        })

        for idx, tenor in enumerate(self.tenors):
            df[f"nominal_yield_{tenor}y"] = scenario["nominal_yield_curves"][:-1, idx]
            df[f"real_yield_{tenor}y"] = scenario["real_yield_curves"][:-1, idx]

        return df.set_index("step")

    def cleanup(self) -> None:
        """Clears cached matrices."""
        self._cache.clear()
        if hasattr(self.scenarios, "cleanup"):
            self.scenarios.cleanup()

    # --- Plotly Interactive Plotting Helpers (Delegated) ---

    def plot_fan_chart(
        self,
        metric: str,
        tenor: Optional[float] = None,
        annualized: bool = False,
        title: Optional[str] = None
    ) -> Any:
        return visualizer.plot_fan_chart(self, metric, tenor, annualized, title)

    def plot_scenario_paths(
        self,
        metric: str,
        num_paths: int = 15,
        tenor: Optional[float] = None,
        annualized: bool = False,
        title: Optional[str] = None
    ) -> Any:
        return visualizer.plot_scenario_paths(self, metric, num_paths, tenor, annualized, title)

    def plot_yield_curve_evolution(
        self,
        years_milestones: List[float] = [0.0, 1.0, 5.0, 15.0, 30.0],
        real: bool = False,
        title: Optional[str] = None
    ) -> Any:
        return visualizer.plot_yield_curve_evolution(self, years_milestones, real, title)

    def plot_horizon_distribution(
        self,
        metric: str,
        target_year: float,
        tenor: Optional[float] = None,
        annualized: bool = False,
        bins: int = 40,
        title: Optional[str] = None
    ) -> Any:
        return visualizer.plot_horizon_distribution(self, metric, target_year, tenor, annualized, bins, title)
