from typing import Dict, Any
import numpy as np

def calculate_portfolio_returns(results, weights: Dict[str, float]) -> np.ndarray:
    """
    Evaluates monthly composite returns for a multi-asset allocation setting.
    """
    if not weights:
        raise ValueError("Weights dictionary cannot be empty.")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("Sum of portfolio weights must be greater than zero.")
    normalized_weights = {k: v / total_weight for k, v in weights.items()}

    blended_returns = np.zeros((results.steps, results.num_scenarios), dtype=np.float64)

    for asset, weight in normalized_weights.items():
        a = asset.lower().strip()
        if a in {"equity", "stock", "returns", "equity_returns"}:
            asset_returns = results._extract_base_matrix("returns")
        elif a in {"fixed_income", "cash", "cdi", "deposit_rates", "fixed_income_short"}:
            from actuarial_esg.engine.simulator import LazyScenarioList
            if isinstance(results.scenarios, LazyScenarioList):
                asset_returns = results.scenarios.deposit_rates.T
            else:
                asset_returns = np.column_stack([s["deposit_rates"] for s in results.scenarios])
        else:
            raise ValueError(
                f"Unknown asset class '{asset}'. Supported assets: "
                "['equity', 'stock', 'fixed_income', 'cash', 'cdi']"
            )

        blended_returns += weight * asset_returns

    return blended_returns

def calculate_portfolio_growth(results, weights: Dict[str, float]) -> np.ndarray:
    """
    Compiles compounded asset growth of $1.00 starting principal.
    """
    returns = calculate_portfolio_returns(results, weights)
    growth_pad = np.vstack([np.zeros(results.num_scenarios), returns])
    return np.cumprod(1.0 + growth_pad, axis=0)
