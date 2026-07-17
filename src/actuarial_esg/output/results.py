import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Optional, Callable


class SimulationResults:
    """
    A unified analysis, post-processing, and visualization engine for ESG outputs.
    Provides an expressive querying interface alongside optimized interactive plotting helpers.
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
        Supports time filtering via 'year', 'month', 'step', or legacy 'time' parameter.
        
        Supports portfolio blending queries using 'portfolio_weights'.
        """
        # 1. Fetch base 2D matrix (steps, scenarios)
        matrix = self._extract_base_matrix(metric, tenor, portfolio_weights)

        # 2. Apply annualization transformations if requested
        if annualized:
            matrix = self._apply_annualization(matrix, metric)

        # 3. Resolve time inputs to indices
        step_indices = self._resolve_time_to_indices(
            time_legacy=time,
            year=year,
            month=month,
            step=step,
            max_idx=len(matrix) - 1
        )

        # 4. Handle time horizon slicing
        if step_indices is not None:
            matrix = matrix[step_indices, :]

        # 5. Collapse dimension 1 (scenarios) according to the requested statistic
        return self._apply_statistics(matrix, stat)

    # --- Portfolio Blending Logic ---

    def calculate_portfolio_returns(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Calculates the blended monthly return series for a given portfolio allocation.
        Assumes constant-mix monthly rebalancing.

        Parameters
        ----------
        weights : dict
            Asset allocation weights, e.g., {"equity": 0.60, "fixed_income": 0.40}.
            Supported keys:
              - 'equity', 'stock', 'returns', 'equity_returns' -> Stock returns
              - 'fixed_income', 'cash', 'cdi', 'deposit_rates', 'fixed_income_short' -> CDI returns
        """
        if not weights:
            raise ValueError("Weights dictionary cannot be empty.")

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("Sum of portfolio weights must be greater than zero.")
        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        blended_returns = np.zeros((self.steps, self.num_scenarios), dtype=np.float64)

        for asset, weight in normalized_weights.items():
            a = asset.lower().strip()
            if a in {"equity", "stock", "returns", "equity_returns"}:
                asset_returns = self._extract_base_matrix("returns")
            elif a in {"fixed_income", "cash", "cdi", "deposit_rates", "fixed_income_short"}:
                # Extract monthly returns directly from scenario properties
                from actuarial_esg.engine.simulator import LazyScenarioList
                if isinstance(self.scenarios, LazyScenarioList):
                    asset_returns = self.scenarios.deposit_rates.T
                else:
                    asset_returns = np.column_stack([s["deposit_rates"] for s in self.scenarios])
            else:
                raise ValueError(
                    f"Unknown asset class '{asset}'. Supported assets: "
                    "['equity', 'stock', 'fixed_income', 'cash', 'cdi']"
                )

            blended_returns += weight * asset_returns

        return blended_returns

    def calculate_portfolio_growth(self, weights: Dict[str, float]) -> np.ndarray:
        """
        Calculates the blended cumulative growth series of $1.00 for a given portfolio allocation.
        Assumes constant-mix monthly rebalancing.
        """
        returns = self.calculate_portfolio_returns(weights)
        # Growth has length self.steps + 1 (t=0 starting at 1.0)
        growth_pad = np.vstack([np.zeros(self.num_scenarios), returns])
        return np.cumprod(1.0 + growth_pad, axis=0)

    # --- Decumulation (Withdrawal) Simulator ---

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
        Simulates the monthly decumulation of a starting balance, supporting dual-bucket asset 
        tracking, generic tax-on-gains (cost-basis drag), frictional drag, and custom withdrawal policies.

        Parameters
        ----------
        initial_balance : float
            Starting balance at t=0 (e.g., 1000000).
        initial_monthly_withdrawal : float
            Initial monthly withdrawal amount (e.g., 4000).
        portfolio_weights : dict
            Asset allocation weights, e.g., {"equity": 0.60, "fixed_income": 0.40}.
        withdrawal_timing : str, default "beginning"
            Timing of withdrawals:
              - 'beginning' / 'advance': Withdrawal is subtracted at the start of each month.
              - 'end' / 'arrears': Withdrawal is subtracted at the end of each month.
        inflate_withdrawals : bool, default True
            If True, monthly withdrawals are adjusted by the simulated CPI index.
        frictional_drag_annual : float, default 0.0
            Annualized frictional expense drag (e.g., 0.00344 for 34.4 bps drag), deducted monthly.
        tax_on_gains_rate : float, default 0.0
            Tax rate applied only to profitable gains exceeding the average cost-basis (e.g., 0.15 for 15%).
            Defaults to 0.0 (no tax applied).
        liquidation_strategy : str, default "constant_mix"
            Algorithm used to withdraw capital and rebalance:
              - 'constant_mix': Withdraw proportionally and rebalance back to target weights monthly.
              - 'cash_first': Withdraw from cash/fixed-income first. Sell equities only if cash is exhausted.
        withdrawal_policy : callable, optional
            A vectorized rule to dynamically compute target net withdrawals.
            Must match signature: f(balance_vector, cpi_vector, step, short_rate_vector) -> target_withdrawal_vector

        Returns
        -------
        dict
            A dictionary containing:
              - 'balances': Array of shape (steps + 1, num_scenarios) of total portfolio values.
              - 'equity_balances': Array of shape (steps + 1, num_scenarios) of equity values.
              - 'fixed_income_balances': Array of shape (steps + 1, num_scenarios) of fixed income values.
              - 'withdrawals': Array of shape (steps, num_scenarios) of actual withdrawals made (net).
              - 'taxes_paid': Array of shape (steps, num_scenarios) of taxes paid.
              - 'probability_of_success': Array of shape (steps + 1,) indicating percentage of solvent scenarios.
        """
        from actuarial_esg.engine.simulator import LazyScenarioList
        is_lazy = isinstance(self.scenarios, LazyScenarioList)

        # 1. Fetch raw underlying asset returns
        equity_returns = self._extract_base_matrix("returns")
        if is_lazy:
            deposit_returns = self.scenarios.deposit_rates.T
        else:
            deposit_returns = np.column_stack([s["deposit_rates"] for s in self.scenarios])

        cpis = self._extract_base_matrix("cpi")

        # 2. Pre-allocate state arrays
        balances = np.zeros((self.steps + 1, self.num_scenarios), dtype=np.float64)
        equity_balances = np.zeros((self.steps + 1, self.num_scenarios), dtype=np.float64)
        fixed_income_balances = np.zeros((self.steps + 1, self.num_scenarios), dtype=np.float64)
        withdrawals = np.zeros((self.steps, self.num_scenarios), dtype=np.float64)
        taxes_paid = np.zeros((self.steps, self.num_scenarios), dtype=np.float64)

        # 3. Setup starting state
        total_weights = sum(portfolio_weights.values())
        eq_weight = portfolio_weights.get("equity", 0.60) / total_weights if "equity" in portfolio_weights else 0.60
        fi_weight = 1.0 - eq_weight

        balances[0, :] = initial_balance
        equity_balances[0, :] = initial_balance * eq_weight
        fixed_income_balances[0, :] = initial_balance * fi_weight

        cost_basis = np.full(self.num_scenarios, initial_balance, dtype=np.float64)
        timing = withdrawal_timing.lower().strip()
        frictional_drag_monthly = frictional_drag_annual / 12.0

        for t in range(self.steps):
            # Resolve inflation factor
            if inflate_withdrawals and t > 0:
                cpi_factor = cpis[t - 1, :]
            else:
                cpi_factor = np.ones(self.num_scenarios, dtype=np.float64)

            # Determine baseline spending net target
            if withdrawal_policy is not None:
                target_w_net = withdrawal_policy(balances[t, :], cpi_factor, t, deposit_returns[t, :])
            else:
                target_w_net = np.full(self.num_scenarios, initial_monthly_withdrawal, dtype=np.float64) * cpi_factor

            # Core state transition
            eq_pre = equity_balances[t, :] * (1.0 + equity_returns[t, :] - frictional_drag_monthly)
            fi_pre = fixed_income_balances[t, :] * (1.0 + deposit_returns[t, :] - frictional_drag_monthly)
            w_pre = eq_pre + fi_pre

            # Tax drag calculation
            if tax_on_gains_rate > 0:
                # Standard cost-basis accounting applied on profitable gains
                gain_ratio = np.zeros(self.num_scenarios)
                profitable_idx = (w_pre > cost_basis) & (cost_basis > 0)
                gain_ratio[profitable_idx] = (w_pre[profitable_idx] - cost_basis[profitable_idx]) / w_pre[profitable_idx]
                target_w_gross = target_w_net / (1.0 - tax_on_gains_rate * gain_ratio)
            else:
                target_w_gross = target_w_net

            if timing in {"beginning", "advance"}:
                # 1. Beginning of Month Withdrawal
                actual_gross = np.minimum(balances[t, :], target_w_gross)
                actual_net = actual_gross * (target_w_net / np.maximum(1e-9, target_w_gross))
                tax = actual_gross - actual_net

                # Track withdrawals and taxes
                withdrawals[t, :] = actual_net
                taxes_paid[t, :] = tax

                # Adjust cost basis proportional to asset redemption
                basis_drawn = cost_basis * (actual_gross / np.maximum(1e-9, balances[t, :]))
                cost_basis = np.maximum(0.0, cost_basis - basis_drawn)

                # Deduct withdrawal using selected strategy
                if liquidation_strategy.lower().strip() == "cash_first":
                    fi_rem = np.maximum(0.0, fixed_income_balances[t, :] - actual_gross)
                    unmet_gross = np.maximum(0.0, actual_gross - fixed_income_balances[t, :])
                    eq_rem = np.maximum(0.0, equity_balances[t, :] - unmet_gross)

                    # Compound the remaining assets monthly
                    equity_balances[t + 1, :] = eq_rem * (1.0 + equity_returns[t, :] - frictional_drag_monthly)
                    fixed_income_balances[t + 1, :] = fi_rem * (1.0 + deposit_returns[t, :] - frictional_drag_monthly)
                else:
                    # 'constant_mix' (rebalance monthly)
                    remaining = np.maximum(0.0, balances[t, :] - actual_gross)
                    equity_balances[t + 1, :] = remaining * eq_weight * (1.0 + equity_returns[t, :] - frictional_drag_monthly)
                    fixed_income_balances[t + 1, :] = remaining * fi_weight * (1.0 + deposit_returns[t, :] - frictional_drag_monthly)

                balances[t + 1, :] = equity_balances[t + 1, :] + fixed_income_balances[t + 1, :]

            else:
                # 2. End of Month Withdrawal (arrears)
                actual_gross = np.minimum(w_pre, target_w_gross)
                actual_net = actual_gross * (target_w_net / np.maximum(1e-9, target_w_gross))
                tax = actual_gross - actual_net

                withdrawals[t, :] = actual_net
                taxes_paid[t, :] = tax

                # Adjust basis
                basis_drawn = cost_basis * (actual_gross / np.maximum(1e-9, w_pre))
                cost_basis = np.maximum(0.0, cost_basis - basis_drawn)

                if liquidation_strategy.lower().strip() == "cash_first":
                    fi_rem = np.maximum(0.0, fi_pre - actual_gross)
                    unmet_gross = np.maximum(0.0, actual_gross - fi_pre)
                    eq_rem = np.maximum(0.0, eq_pre - unmet_gross)

                    equity_balances[t + 1, :] = eq_rem
                    fixed_income_balances[t + 1, :] = fi_rem
                else:
                    remaining = np.maximum(0.0, w_pre - actual_gross)
                    equity_balances[t + 1, :] = remaining * eq_weight
                    fixed_income_balances[t + 1, :] = remaining * fi_weight

                balances[t + 1, :] = equity_balances[t + 1, :] + fixed_income_balances[t + 1, :]

        # Track solvency over time
        probability_of_success = np.mean(balances > 1e-4, axis=1)

        # Cache results for plotting and metrics queries
        self._cache["decumulation_balance"] = balances
        self._cache["decumulation_withdrawal"] = withdrawals

        return {
            "balances": balances,
            "equity_balances": equity_balances,
            "fixed_income_balances": fixed_income_balances,
            "withdrawals": withdrawals,
            "taxes_paid": taxes_paid,
            "probability_of_success": probability_of_success
        }

    # --- Time Parameter Mapping Logic ---

    def _resolve_time_to_indices(
        self,
        time_legacy: Optional[Any],
        year: Optional[Any],
        month: Optional[Any],
        step: Optional[Any],
        max_idx: int
    ) -> Optional[List[int]]:
        """
        Maps year, month, step, or legacy time filters to internal row indices.
        Ensures that exactly one parameter filter is supplied.
        """
        provided = {
            "time": time_legacy,
            "year": year,
            "month": month,
            "step": step
        }
        active = {k: v for k, v in provided.items() if v is not None}

        if len(active) > 1:
            raise ValueError(
                f"Conflicting time parameters provided: {list(active.keys())}. "
                "Specify exactly one of 'year', 'month', 'step', or legacy 'time'."
            )

        if len(active) == 0:
            return None

        key, val = list(active.items())[0]

        # Check for "all" bypass
        if isinstance(val, str) and val.lower() == "all":
            return None

        if key == "step":
            raw = np.atleast_1d(val)
            return list(np.clip(raw.astype(int), 0, max_idx))

        elif key == "month":
            # Month 0 maps to Step 0 (initial state)
            raw = np.atleast_1d(val)
            return list(np.clip(raw.astype(int), 0, max_idx))

        elif key == "year" or key == "time":
            # Year 0 maps to Step 0 (initial state)
            raw = np.atleast_1d(val)
            indices = np.round(raw * self.steps_per_year).astype(int)
            return list(np.clip(indices, 0, max_idx))

        return None

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
            # Growth has length self.steps + 1 (t=0 to t=steps)
            growth = np.vstack([np.zeros(self.num_scenarios), returns])
            res = np.cumprod(1.0 + growth, axis=0)

        elif m in {"cpi", "inflation_index"}:
            if is_lazy:
                res = self.scenarios.cpis.T
            else:
                res = np.column_stack([s["cpis"] for s in self.scenarios])

        elif m in {"inflation_rate", "inflation"}:
            # Convert cumulative CPI back to periodic annualized rates
            cpis = self._extract_base_matrix("cpi")
            cpis_padded = np.vstack([np.ones(self.num_scenarios), cpis])
            with np.errstate(divide='ignore', invalid='ignore'):
                monthly_rates = (cpis_padded[1:, :] / cpis_padded[:-1, :]) - 1.0
            res = (1.0 + monthly_rates) ** 12 - 1.0

        elif m in {"short_rate", "cdi", "deposit_rates"}:
            if is_lazy:
                # Geometric annualization of monthly CDI rates
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
                    # Derive real yields for the single tenor on-the-fly (excluding final step)
                    res = self._generate_real_yields_for_tenor(tenor)[:-1, :]
                else:
                    # Derive nominal yields for the single tenor on-the-fly (excluding final step)
                    res = self._generate_cir_yields_for_tenor(tenor)[:-1, :]
            else:
                tenor_idx = self.tenors.index(tenor)
                key = "real_yield_curves" if m == "real_yield" else "nominal_yield_curves"
                res = np.column_stack([s[key][:-1, tenor_idx] for s in self.scenarios])

        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose from: "
                             "returns, growth, cpi, inflation, cdi, nominal_yield, real_yield, "
                             "portfolio_returns, portfolio_growth, decumulation_balance, decumulation_withdrawal")

        self._cache[cache_key] = res
        return res

    # --- On-The-Fly Vectorized Tenor Yield Derivations ---

    def _generate_cir_yields_for_tenor(self, tenor: float) -> np.ndarray:
        """Derives CIR yields for a single tenor across all scenarios instantly."""
        r = self.scenarios.cdi_paths.T       # shape (num_scenarios, steps + 1)
        mu = self.scenarios.mu_cdi_paths.T   # shape (num_scenarios, steps + 1)
        theta = self.scenarios.cir_theta     # shape (num_scenarios,)
        sigma = self.scenarios.cir_sigma     # shape (num_scenarios,)

        h = np.sqrt(theta ** 2 + 2.0 * (sigma ** 2))
        denominator = (theta + h) * (np.exp(h * tenor) - 1.0) + 2.0 * h
        base_A = (2.0 * h * np.exp((theta + h) * tenor / 2.0)) / denominator
        B_tau = (2.0 * (np.exp(h * tenor) - 1.0)) / denominator

        log_base_A_div_tenor = np.log(base_A) / tenor
        B_tau_div_tenor = B_tau / tenor
        safe_sigma_sq = np.maximum(1e-6, sigma ** 2)
        power_factor = (2.0 * theta[:, np.newaxis] * mu) / safe_sigma_sq[:, np.newaxis]

        # Vectorized multiplication and broadcasting across all scenarios
        yields = r * B_tau_div_tenor[:, np.newaxis]
        yields -= power_factor * log_base_A_div_tenor[:, np.newaxis]
        return yields.T  # returns shape (steps + 1, num_scenarios)

    def _generate_real_yields_for_tenor(self, tenor: float) -> np.ndarray:
        """Derives Fisher real yields for a single tenor across all scenarios on-the-fly."""
        yields_nominal = self._generate_cir_yields_for_tenor(tenor)  # shape (steps + 1, num_scenarios)

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

    # --- Slicing and Statistics Symmetrical Logic ---

    def _apply_annualization(self, matrix: np.ndarray, metric: str) -> np.ndarray:
        m = metric.lower().strip()
        if m in {"equity_growth", "growth", "portfolio_growth"}:
            steps = np.arange(len(matrix))[:, np.newaxis]
            years = steps / self.steps_per_year
            with np.errstate(divide='ignore', invalid='ignore'):
                annualized = np.power(matrix, 1.0 / np.maximum(1e-9, years)) - 1.0
            annualized[0, :] = 0.0
            return annualized

        elif m in {"cpi", "inflation_index"}:
            steps = np.arange(1, len(matrix) + 1)[:, np.newaxis]
            years = steps / self.steps_per_year
            return np.power(matrix, 1.0 / years) - 1.0

        return matrix

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
                raise ValueError(f"Could not parse valid percentile from '{stat}'. Try formats like 'p5', 'p95'.")
        else:
            raise ValueError(f"Unknown statistic '{stat}'. Choose from: 'raw', 'mean', 'median', 'std', 'pXX'.")

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

    # --- Plotly Interactive Plotting Helpers ---

    def _import_plotly(self):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            return go, make_subplots
        except ImportError:
            raise ImportError(
                "Plotly is required for these visualizers. Install it using 'pip install plotly'."
            )

    def plot_fan_chart(
        self,
        metric: str,
        tenor: Optional[float] = None,
        annualized: bool = False,
        title: Optional[str] = None
    ) -> Any:
        go, _ = self._import_plotly()

        median = self.query(metric, stat="median", step="all", tenor=tenor, annualized=annualized)
        p5 = self.query(metric, stat="p5", step="all", tenor=tenor, annualized=annualized)
        p25 = self.query(metric, stat="p25", step="all", tenor=tenor, annualized=annualized)
        p75 = self.query(metric, stat="p75", step="all", tenor=tenor, annualized=annualized)
        p95 = self.query(metric, stat="p95", step="all", tenor=tenor, annualized=annualized)

        years = np.arange(len(median)) / self.steps_per_year

        fig = go.Figure()

        # 5% to 95% Confidence Band
        fig.add_trace(go.Scatter(
            x=years, y=p5, line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=years, y=p95, fill='tonexty', fillcolor='rgba(70, 130, 180, 0.15)',
            line=dict(width=0), name='5% - 95% Interval'
        ))

        # 25% to 75% Confidence Band
        fig.add_trace(go.Scatter(
            x=years, y=p25, line=dict(width=0), showlegend=False, hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=years, y=p75, fill='tonexty', fillcolor='rgba(70, 130, 180, 0.3)',
            line=dict(width=0), name='25% - 75% Interval'
        ))

        # Median Path
        fig.add_trace(go.Scatter(
            x=years, y=median, line=dict(color='rgb(12, 35, 115)', width=3),
            name='Median Path'
        ))

        fig.update_layout(
            title=title if title else f"Fan Chart: {metric.replace('_', ' ').title()}",
            xaxis_title="Projection Horizon (Years)",
            yaxis_title="Value" if not annualized else "Annualized Rate",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig

    def plot_scenario_paths(
        self,
        metric: str,
        num_paths: int = 15,
        tenor: Optional[float] = None,
        annualized: bool = False,
        title: Optional[str] = None
    ) -> Any:
        go, _ = self._import_plotly()

        raw_paths = self.query(metric, stat="raw", step="all", tenor=tenor, annualized=annualized)
        mean_path = self.query(metric, stat="mean", step="all", tenor=tenor, annualized=annualized)

        years = np.arange(len(mean_path)) / self.steps_per_year
        paths_to_plot = min(num_paths, raw_paths.shape[1])

        fig = go.Figure()

        # Add raw paths
        for i in range(paths_to_plot):
            fig.add_trace(go.Scatter(
                x=years, y=raw_paths[:, i],
                line=dict(color='rgba(120, 120, 120, 0.35)', width=1),
                showlegend=False, name=f"Path {i}"
            ))

        # Highlight expected mean path
        fig.add_trace(go.Scatter(
            x=years, y=mean_path,
            line=dict(color='black', width=3, dash='dash'),
            name='Expected Mean'
        ))

        fig.update_layout(
            title=title if title else f"Scenario Sample Paths: {metric.replace('_', ' ').title()}",
            xaxis_title="Projection Horizon (Years)",
            yaxis_title="Value" if not annualized else "Annualized Rate",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig

    def plot_yield_curve_evolution(
        self,
        years_milestones: List[float] = [0.0, 1.0, 5.0, 15.0, 30.0],
        real: bool = False,
        title: Optional[str] = None
    ) -> Any:
        go, _ = self._import_plotly()

        metric = "real_yield" if real else "nominal_yield"
        fig = go.Figure()

        for y in years_milestones:
            yields_for_tenors = [
                self.query(metric, stat="mean", year=y, tenor=t)
                for t in self.tenors
            ]
            fig.add_trace(go.Scatter(
                x=self.tenors, y=np.array(yields_for_tenors) * 100,
                mode='lines+markers', name=f"Year {y}"
            ))

        curve_type = "Real" if real else "Nominal"
        fig.update_layout(
            title=title if title else f"Expected {curve_type} Yield Curve Evolution",
            xaxis_title="Tenor (Years)",
            yaxis_title="Average Yield (%)",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig

    def plot_horizon_distribution(
        self,
        metric: str,
        target_year: float,
        tenor: Optional[float] = None,
        annualized: bool = False,
        bins: int = 40,
        title: Optional[str] = None
    ) -> Any:
        go, _ = self._import_plotly()

        raw_data = self.query(metric, stat="raw", year=target_year, tenor=tenor, annualized=annualized)
        mean_val = np.mean(raw_data)
        med_val = np.median(raw_data)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=np.arange(len(raw_data)), y=raw_data,
            mode='markers', marker=dict(color='steelblue', opacity=0.75),
            name="Scenarios"
        ) if metric == "yield" else go.Histogram(
            x=raw_data, nbinsx=bins, histnorm='density',
            marker_color='steelblue', opacity=0.75, name="Scenario Density"
        ))

        fig.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=2,
                      annotation_text=f"Mean: {mean_val:.4f}", annotation_position="top right")
        fig.add_vline(x=med_val, line_dash="dot", line_color="green", line_width=2,
                      annotation_text=f"Median: {med_val:.4f}", annotation_position="top left")

        fig.update_layout(
            title=title if title else f"Distribution of {metric.replace('_', ' ').title()} at Year {target_year}",
            xaxis_title="Value" if not annualized else "Annualized Rate",
            yaxis_title="Density",
            template="plotly_white",
            margin=dict(l=40, r=40, t=60, b=40)
        )

        return fig
