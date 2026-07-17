from typing import Optional, Union, List, Any
import numpy as np

def resolve_time_to_indices(
    time_legacy: Optional[Any],
    year: Optional[Any],
    month: Optional[Any],
    step: Optional[Any],
    max_idx: int,
    steps_per_year: int
) -> Optional[List[int]]:
    """
    Resolves legacy time, year, month, or step selections to specific index offsets.
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

    if isinstance(val, str) and val.lower() == "all":
        return None

    if key == "step":
        raw = np.atleast_1d(val)
        return list(np.clip(raw.astype(int), 0, max_idx))

    elif key == "month":
        raw = np.atleast_1d(val)
        return list(np.clip(raw.astype(int), 0, max_idx))

    elif key in ("year", "time"):
        raw = np.atleast_1d(val)
        indices = np.round(raw * steps_per_year).astype(int)
        return list(np.clip(indices, 0, max_idx))

    return None

def apply_annualization(matrix: np.ndarray, metric: str, steps_per_year: int) -> np.ndarray:
    """
    Transforms multi-period cumulative indicators into annualized percentages.
    """
    m = metric.lower().strip()
    if m in {"equity_growth", "growth", "portfolio_growth"}:
        steps = np.arange(len(matrix))[:, np.newaxis]
        years = steps / steps_per_year
        with np.errstate(divide='ignore', invalid='ignore'):
            annualized = np.power(matrix, 1.0 / np.maximum(1e-9, years)) - 1.0
        annualized[0, :] = 0.0
        return annualized

    elif m in {"cpi", "inflation_index"}:
        steps = np.arange(1, len(matrix) + 1)[:, np.newaxis]
        years = steps / steps_per_year
        return np.power(matrix, 1.0 / years) - 1.0

    return matrix
