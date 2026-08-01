import os
import sys
import time
import platform
import multiprocessing
import numpy as np
import pandas as pd

from aethel import SimulatorConfig, MarketSimulator, SimulationResults
from aethel.engine.loops import HAS_NUMBA
from aethel.engine.simulator import get_available_system_ram


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title:^68}")
    print("=" * 70)


def collect_system_info() -> dict:
    """Gathers hardware and software environment details."""
    try:
        ram_gb = get_available_system_ram() / (1024 ** 3)
    except Exception:
        ram_gb = 0.0

    return {
        "OS": f"{platform.system()} {platform.release()}",
        "Python Version": sys.version.split()[0],
        "CPU Cores": multiprocessing.cpu_count(),
        "Available RAM (GB)": f"{ram_gb:.2f} GB" if ram_gb > 0 else "Unknown",
        "Numba JIT Enabled": HAS_NUMBA,
    }


def run_simulation_benchmark(num_scenarios: int, duration_years: int, seed: int = 42) -> dict:
    """Measures simulation execution time and calculates throughput metrics."""
    config = SimulatorConfig(
        duration_years=duration_years,
        num_scenarios=num_scenarios,
        seed=seed
    )
    
    total_steps = config.steps  # months
    total_calculations = num_scenarios * total_steps

    t0 = time.perf_counter()
    simulator = MarketSimulator(config)
    raw_scenarios = simulator.run()
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    scenarios_per_sec = num_scenarios / elapsed
    steps_per_sec = total_calculations / elapsed

    return {
        "Scenarios": num_scenarios,
        "Horizon (Years)": duration_years,
        "Total Steps / Path": total_steps,
        "Time Elapsed (s)": elapsed,
        "Scenarios / Sec": scenarios_per_sec,
        "Steps / Sec": steps_per_sec,
        "Results Object": raw_scenarios
    }


def run_query_benchmark(results_obj, num_queries: int = 5) -> dict:
    """Measures the latency of analytical extraction and lazy yield generation."""
    results = SimulationResults(results_obj)
    
    # Run once to warm up any potential JIT or initial caching structures
    _ = results.query("equity_growth", stat="median", year=10.0)
    
    t0 = time.perf_counter()
    for i in range(num_queries):
        # Queries that trigger dynamic yield curve derivations
        _ = results.query("nominal_yield", stat="mean", year=15.0, tenor=10.0)
        _ = results.query("real_yield", stat="p5", year=25.0, tenor=30.0)
        # Standard macroeconomic queries
        _ = results.query("cpi", stat="p95", step="all")
        _ = results.query("rate", stat="median", step="all")
    t1 = time.perf_counter()
    
    elapsed_total = t1 - t0
    avg_query_time_ms = (elapsed_total / (num_queries * 4)) * 1000
    
    results.cleanup()
    return {
        "Total Queries Profiled": num_queries * 4,
        "Total Query Time (s)": elapsed_total,
        "Avg Query Latency (ms)": avg_query_time_ms
    }


def run_decumulation_benchmark(results_obj, num_runs: int = 5) -> dict:
    """Measures performance of the portfolio cashflow and tax liquidation engine."""
    results = SimulationResults(results_obj)
    weights = {"equity": 0.60, "fixed_income": 0.40}
    
    t0 = time.perf_counter()
    for _ in range(num_runs):
        _ = results.simulate_decumulation(
            initial_balance=1_000_000.0,
            initial_monthly_withdrawal=4000.0,
            portfolio_weights=weights,
            frictional_drag_annual=0.0025,
            tax_on_gains_rate=0.15,
            liquidation_strategy="cash_first"
        )
        # Clear the cache between runs to prevent instant hits
        results.cleanup()
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    avg_decum_time_ms = (elapsed / num_runs) * 1000
    
    return {
        "Scenarios Decumulated": results.num_scenarios,
        "Avg Decumulation Time (ms)": avg_decum_time_ms,
        "Decumulations / Sec": results.num_scenarios / (elapsed / num_runs)
    }


def main():
    print_header("Aethel ESG Performance Benchmark Suite")
    
    # 1. System Diagnosis
    sys_info = collect_system_info()
    print("Environment Specifications:")
    for k, v in sys_info.items():
        print(f"  - {k:22}: {v}")
    
    # 2. Performance Scaling Matrix
    # We evaluate across three typical production scale levels
    benchmarks_configs = [
        {"scenarios": 1000, "years": 30, "label": "Small Scale (ALM Baseline)"},
        {"scenarios": 5000, "years": 40, "label": "Medium Scale (Retirement Study)"},
        {"scenarios": 20000, "years": 50, "label": "Large Scale (Stress Portfolio)"}
    ]
    
    sim_results = []
    query_results = []
    decum_results = []
    
    for config in benchmarks_configs:
        label = config["label"]
        n_scen = config["scenarios"]
        years = config["years"]
        
        print_header(f"Benchmarking: {label} ({n_scen:,} paths, {years} years)")
        
        # A. Run Engine Simulation
        print("  -> Executing stochastic loop...")
        sim_metrics = run_simulation_loop_with_feedback(n_scen, years)
        sim_results.append({
            "Label": label,
            "Scenarios": n_scen,
            "Horizon": f"{years}y",
            "Time (s)": f"{sim_metrics['Time Elapsed (s)']:.3f}s",
            "Scenarios / s": f"{sim_metrics['Scenarios / Sec']:.1f}",
            "Steps / s": f"{sim_metrics['Steps / Sec']:,.0f}"
        })
        
        # B. Run Query & Lazy Evaluation Engine
        print("  -> Profiling analytical query engine (lazy structures)...")
        q_metrics = run_query_benchmark(sim_metrics["Results Object"])
        query_results.append({
            "Label": label,
            "Scenarios": n_scen,
            "Avg Latency (ms)": f"{q_metrics['Avg Query Latency (ms)']:.2f} ms"
        })
        
        # C. Run Decumulation Engine
        print("  -> Profiling portfolio liquidation & decumulation simulator...")
        d_metrics = run_decumulation_benchmark(sim_metrics["Results Object"])
        decum_results.append({
            "Label": label,
            "Scenarios": n_scen,
            "Avg Decumulation Time (ms)": f"{d_metrics['Avg Decumulation Time (ms)']:.1f} ms",
            "Throughput (Scenarios / s)": f"{d_metrics['Decumulations / Sec']:,.0f}"
        })

    # 3. Report Summaries
    print_header("Benchmark Summary - Core Simulation Loop")
    df_sim = pd.DataFrame(sim_results)
    print(df_sim.to_string(index=False))
    
    print_header("Benchmark Summary - Query Latency")
    df_q = pd.DataFrame(query_results)
    print(df_q.to_string(index=False))

    print_header("Benchmark Summary - Decumulation Operations")
    df_d = pd.DataFrame(decum_results)
    print(df_d.to_string(index=False))
    
    print("\nBenchmark completed successfully.")


def run_simulation_loop_with_feedback(num_scenarios: int, duration_years: int) -> dict:
    """Helper execution wrapper to gracefully run and measure the simulation."""
    return run_simulation_benchmark(num_scenarios, duration_years)


if __name__ == "__main__":
    main()