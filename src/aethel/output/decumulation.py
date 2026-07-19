from typing import Dict, Optional, Callable, Any
import numpy as np

def simulate_decumulation(
    results,
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
    Simulates monthly retirement portfolio decumulation with drag, taxes, and dynamic policies.
    """
    from aethel.engine.simulator import LazyScenarioList
    is_lazy = isinstance(results.scenarios, LazyScenarioList)

    equity_returns = results._extract_base_matrix("returns")
    if is_lazy:
        deposit_returns = results.scenarios.deposit_rates.T
    else:
        deposit_returns = np.column_stack([s["deposit_rates"] for s in results.scenarios])

    cpis = results._extract_base_matrix("cpi")

    balances = np.zeros((results.steps + 1, results.num_scenarios), dtype=np.float64)
    equity_balances = np.zeros((results.steps + 1, results.num_scenarios), dtype=np.float64)
    fixed_income_balances = np.zeros((results.steps + 1, results.num_scenarios), dtype=np.float64)
    withdrawals = np.zeros((results.steps, results.num_scenarios), dtype=np.float64)
    taxes_paid = np.zeros((results.steps, results.num_scenarios), dtype=np.float64)

    total_weights = sum(portfolio_weights.values())
    eq_weight = portfolio_weights.get("equity", 0.60) / total_weights if "equity" in portfolio_weights else 0.60
    fi_weight = 1.0 - eq_weight

    balances[0, :] = initial_balance
    equity_balances[0, :] = initial_balance * eq_weight
    fixed_income_balances[0, :] = initial_balance * fi_weight

    cost_basis = np.full(results.num_scenarios, initial_balance, dtype=np.float64)
    timing = withdrawal_timing.lower().strip()
    frictional_drag_monthly = frictional_drag_annual / 12.0

    for t in range(results.steps):
        if inflate_withdrawals and t > 0:
            cpi_factor = cpis[t - 1, :]
        else:
            cpi_factor = np.ones(results.num_scenarios, dtype=np.float64)

        if withdrawal_policy is not None:
            target_w_net = withdrawal_policy(balances[t, :], cpi_factor, t, deposit_returns[t, :])
        else:
            target_w_net = np.full(results.num_scenarios, initial_monthly_withdrawal, dtype=np.float64) * cpi_factor

        eq_pre = equity_balances[t, :] * (1.0 + equity_returns[t, :] - frictional_drag_monthly)
        fi_pre = fixed_income_balances[t, :] * (1.0 + deposit_returns[t, :] - frictional_drag_monthly)
        w_pre = eq_pre + fi_pre

        if tax_on_gains_rate > 0:
            gain_ratio = np.zeros(results.num_scenarios)
            profitable_idx = (w_pre > cost_basis) & (cost_basis > 0)
            gain_ratio[profitable_idx] = (w_pre[profitable_idx] - cost_basis[profitable_idx]) / w_pre[profitable_idx]
            target_w_gross = target_w_net / (1.0 - tax_on_gains_rate * gain_ratio)
        else:
            target_w_gross = target_w_net

        if timing in {"beginning", "advance"}:
            actual_gross = np.minimum(balances[t, :], target_w_gross)
            actual_net = actual_gross * (target_w_net / np.maximum(1e-9, target_w_gross))
            tax = actual_gross - actual_net

            withdrawals[t, :] = actual_net
            taxes_paid[t, :] = tax

            basis_drawn = cost_basis * (actual_gross / np.maximum(1e-9, balances[t, :]))
            cost_basis = np.maximum(0.0, cost_basis - basis_drawn)

            if liquidation_strategy.lower().strip() == "cash_first":
                fi_rem = np.maximum(0.0, fixed_income_balances[t, :] - actual_gross)
                unmet_gross = np.maximum(0.0, actual_gross - fixed_income_balances[t, :])
                eq_rem = np.maximum(0.0, equity_balances[t, :] - unmet_gross)

                equity_balances[t + 1, :] = eq_rem * (1.0 + equity_returns[t, :] - frictional_drag_monthly)
                fixed_income_balances[t + 1, :] = fi_rem * (1.0 + deposit_returns[t, :] - frictional_drag_monthly)
            else:
                remaining = np.maximum(0.0, balances[t, :] - actual_gross)
                equity_balances[t + 1, :] = remaining * eq_weight * (1.0 + equity_returns[t, :] - frictional_drag_monthly)
                fixed_income_balances[t + 1, :] = remaining * fi_weight * (1.0 + deposit_returns[t, :] - frictional_drag_monthly)

            balances[t + 1, :] = equity_balances[t + 1, :] + fixed_income_balances[t + 1, :]

        else:
            actual_gross = np.minimum(w_pre, target_w_gross)
            actual_net = actual_gross * (target_w_net / np.maximum(1e-9, target_w_gross))
            tax = actual_gross - actual_net

            withdrawals[t, :] = actual_net
            taxes_paid[t, :] = tax

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

    probability_of_success = np.mean(balances > 1e-4, axis=1)

    results._cache["decumulation_balance"] = balances
    results._cache["decumulation_withdrawal"] = withdrawals

    return {
        "balances": balances,
        "equity_balances": equity_balances,
        "fixed_income_balances": fixed_income_balances,
        "withdrawals": withdrawals,
        "taxes_paid": taxes_paid,
        "probability_of_success": probability_of_success
    }
