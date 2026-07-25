import os
import json
import numpy as np
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

# --- SYMMETRIC PARAMETER GRID (HORIZON FIXED AT 50 YEARS) ---
HORIZON_YEARS = 50
INFLATION_TARGETS = [0.02, 0.04, 0.06]
WITHDRAWALS = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000]
ALLOCATION_MIXES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
STRATEGIES = ["constant_mix", "cash_first_guardrail"]

DOWNSAMPLE_FACTOR = 3
N_SAMPLE_PATHS = 24          # Spaghetti-plot paths stored per inflation target
INITIAL_BALANCE = 1_000_000.0
PRESETS_DIR = "presets"
OUTPUT_FILE = "demo/demo_database.json"


def load_preset_config() -> SimulatorConfig:
    path = os.path.join(PRESETS_DIR, "world.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing 'world.json' in {PRESETS_DIR}. Run build_presets.py first.")
    with open(path, "r") as f:
        return SimulatorConfig.from_dict(json.load(f))


def sanitize_rates(arr) -> list[float]:
    return [round(float(x), 4) for x in arr]


def sanitize_balances(arr) -> list[int]:
    return [int(round(float(x))) for x in arr]


def make_spending_guardrail(initial_monthly_withdrawal, threshold=400000.0, reduction=0.80):
    def variable_spending_guardrail(balance, cpi_factor, step, deposit_rate):
        base_w = initial_monthly_withdrawal * cpi_factor
        return np.where(balance < threshold, base_w * reduction, base_w)
    return variable_spending_guardrail


def safe_query(results, metric, stat):
    """Query a stat; gracefully fall back to p50 if the percentile is unsupported."""
    try:
        return results.query(metric, stat=stat, step="all")
    except Exception:
        return results.query(metric, stat="p50", step="all")


def try_query(results, metric, stat):
    try:
        return results.query(metric, stat=stat, step="all")
    except Exception:
        return None


def extract_sample_paths(raw_scenarios, n=N_SAMPLE_PATHS, seed=7):
    """Best-effort extraction of raw per-scenario equity-growth paths for the
    frontend spaghetti plot. Returns [] if the internal layout is unknown."""
    candidates = ["equity_growth", "equity", "equity_growth_factor", "equity_path"]
    matrix = None
    try:
        first = raw_scenarios[0]
        for cand in candidates:
            vals = None
            if isinstance(first, dict) and cand in first:
                vals = [s[cand] for s in raw_scenarios]
            elif hasattr(first, cand):
                vals = [getattr(s, cand) for s in raw_scenarios]
            if vals is not None:
                matrix = np.asarray(vals, dtype=float)
                break
    except Exception:
        matrix = None
    if matrix is None or matrix.ndim != 2:
        return []
    if matrix.shape[0] != len(raw_scenarios) and matrix.shape[1] == len(raw_scenarios):
        matrix = matrix.T
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(matrix.shape[0], size=min(n, matrix.shape[0]), replace=False))
    sampled = matrix[idx][:, ::DOWNSAMPLE_FACTOR]
    return [[round(float(v), 4) for v in row] for row in sampled]


def ruin_distribution(balances) -> dict:
    """Histogram of the year each scenario's balance first hits zero."""
    arr = np.asarray(balances, dtype=float)
    if arr.ndim != 2:
        return {}
    depleted = arr <= 0.0
    ever = depleted.any(axis=0)
    first_idx = depleted.argmax(axis=0)
    years = np.minimum((first_idx // 12).astype(int), HORIZON_YEARS)
    hist = np.bincount(years[ever], minlength=HORIZON_YEARS + 1).astype(int).tolist()
    return {"ruin_hist": hist, "survived": int((~ever).sum())}


def economic_block(results, raw_scenarios, downsampled_timeline) -> dict:
    stats = ["p5", "p25", "p50", "p75", "p95", "mean"]
    block = {
        "timeline": sanitize_rates(downsampled_timeline),
        "cpi": {s: sanitize_rates(safe_query(results, "cpi", s)[::DOWNSAMPLE_FACTOR]) for s in stats},
        "yield": {s: sanitize_rates(safe_query(results, "rate", s)[::DOWNSAMPLE_FACTOR]) for s in stats},
    }
    # Equity-growth fan (try common metric names)
    eq = {}
    for s in stats:
        arr = try_query(results, "equity_growth", s)
        if arr is None:
            arr = try_query(results, "equity", s)
        if arr is not None:
            eq[s] = sanitize_rates(arr[::DOWNSAMPLE_FACTOR])
    if len(eq) == len(stats):
        block["equity"] = eq
    samples = extract_sample_paths(raw_scenarios)
    if samples:
        block["samples_equity"] = samples
    return block


def main():
    print("=" * 50)
    print(" Aethel ESG - Compiling Demo Cache v2")
    print("=" * 50)

    config = load_preset_config()
    config.duration_years = HORIZON_YEARS
    config.num_scenarios = 1000
    config.seed = 42
    config.max_workers = 4

    db = {
        "metadata": {
            "version": 2,
            "preset": "world",
            "scenarios": config.num_scenarios,
            "horizon_years": HORIZON_YEARS,
            "initial_balance": INITIAL_BALANCE,
            "frictional_drag_annual": 0.0025,
            "tax_on_gains_rate": 0.15,
            "inflation_targets": INFLATION_TARGETS,
            "withdrawals": WITHDRAWALS,
            "allocations": ALLOCATION_MIXES,
            "strategies": STRATEGIES,
            "downsample_factor": DOWNSAMPLE_FACTOR,
        },
        "economic_data": {},
        "decumulation_data": {},
    }

    for target in INFLATION_TARGETS:
        target_pct = int(round(target * 100))
        print(f"\n -> Base Economic Paths @ {target_pct}% Inflation Target...")
        config.ou_mu = target

        simulator = MarketSimulator(config)
        raw_scenarios = simulator.run()
        results = SimulationResults(raw_scenarios)

        full_timeline = np.arange(config.steps + 1) / 12.0
        db["economic_data"][f"target_{target_pct}"] = economic_block(
            results, raw_scenarios, full_timeline[::DOWNSAMPLE_FACTOR].tolist()
        )

        for w_amount in WITHDRAWALS:
            for eq_mix in ALLOCATION_MIXES:
                weights = {"equity": eq_mix / 100.0, "fixed_income": (100 - eq_mix) / 100.0}
                for strategy in STRATEGIES:
                    combo_key = f"{w_amount}w_{eq_mix}a_{strategy}_{target_pct}t"
                    print(f"    * {target_pct}% | ${w_amount}/mo | {eq_mix}% EQ | {strategy}")

                    kwargs = dict(
                        initial_balance=INITIAL_BALANCE,
                        initial_monthly_withdrawal=w_amount,
                        portfolio_weights=weights,
                        frictional_drag_annual=0.0025,
                        tax_on_gains_rate=0.15,
                    )
                    if strategy == "cash_first_guardrail":
                        decum = results.simulate_decumulation(
                            liquidation_strategy="cash_first",
                            withdrawal_policy=make_spending_guardrail(w_amount),
                            **kwargs,
                        )
                    else:
                        decum = results.simulate_decumulation(
                            liquidation_strategy="constant_mix",
                            withdrawal_policy=None,
                            **kwargs,
                        )

                    balances_ds = decum["balances"][::DOWNSAMPLE_FACTOR]
                    node = {
                        "solvency": sanitize_rates(decum["probability_of_success"][::DOWNSAMPLE_FACTOR]),
                        "balance_p5": sanitize_balances(np.percentile(balances_ds, 5.0, axis=1)),
                        "balance_p50": sanitize_balances(np.percentile(balances_ds, 50.0, axis=1)),
                        "balance_p95": sanitize_balances(np.percentile(balances_ds, 95.0, axis=1)),
                    }
                    node.update(ruin_distribution(decum["balances"]))
                    db["decumulation_data"][combo_key] = node

        results.cleanup()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(db, f, separators=(",", ":"))

    print("\n" + "=" * 50)
    print(f" SUCCESS! Database written: {OUTPUT_FILE}")
    print(f" Size: {os.path.getsize(OUTPUT_FILE) / 1024 / 1024:.2f} MB")
    print("=" * 50)


if __name__ == "__main__":
    main()
