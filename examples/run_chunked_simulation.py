import time
import numpy as np
from actuarial_esg import SimulatorConfig, MarketSimulator, SimulationResults


def run_massive_simulation_demo():
    print("======================================================================")
    print("          ACTUARIAL ESG - HIGH-SCALE IN-MEMORY SIMULATION             ")
    print("======================================================================")

    # 1. Setup simulation parameters
    # Dynamic RAM-Aware Guard activate and partition the workload automatically!
    total_scenarios = 100000
    duration_years = 120
    seed = 42

    print(f"Configuring projection: {duration_years} years ({duration_years * 12} steps)")
    print(f"Total scenario target:  {total_scenarios:,} scenarios")
    print("Initializing execution engine...")

    # 2. Initialize configuration (letting the simulator handle concurrency and chunking)
    config = SimulatorConfig(
        duration_years=duration_years,
        num_scenarios=total_scenarios,
        seed=seed
    )

    # 3. Run simulation engine (Fast Vectorized Implementation with Adaptive RAM Guard)
    t0 = time.time()
    simulator = MarketSimulator(config)
    raw_scenarios = simulator.run()
    t1 = time.time()
    print(f"✓ Simulation execution finished in: {t1 - t0:.2f} seconds.")

    # 4. Bind output to results container
    results = SimulationResults(raw_scenarios)

    print("\n[Query] Extracting median trajectory paths (50th percentile) across all scenarios...")

    # 5. Extract median macroeconomic variables using the optimized on-the-fly query engine
    years = np.arange(duration_years * 12) / 12

    # Short rates and inflation (Primary Axis)
    cdi_median = results.query("cdi", stat="median", step="all")
    inflation_median = results.query("inflation", stat="median", step="all")

    # Long-term yields (Primary Axis) - derived on-the-fly instantly
    yield_nom_10y = results.query("nominal_yield", stat="median", step="all", tenor=10.0)
    yield_real_10y = results.query("real_yield", stat="median", step="all", tenor=10.0)

    # Cumulative equity growth on $1.00 base (Secondary Axis)
    years_growth = np.arange(duration_years * 12 + 1) / 12
    equity_growth_median = results.query("growth", stat="median", step="all")

    print("[Visualization] Assembling interactive Plotly chart...")

    # 6. Construct professional double-axis Plotly chart
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Primary Axis: Annualized yields and rates (%)
    fig.add_trace(
        go.Scatter(
            x=years,
            y=cdi_median * 100,
            name="CDI Short Rate (Median)",
            line=dict(color="#D32F2F", width=2.5, dash="solid")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=inflation_median * 100,
            name="IPCA Inflation Rate (Median)",
            line=dict(color="#00796B", width=2.5, dash="dash")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=yield_nom_10y * 100,
            name="10Y Nominal Yield (Median)",
            line=dict(color="#1976D2", width=2.0, dash="dot")
        ),
        secondary_y=False
    )

    fig.add_trace(
        go.Scatter(
            x=years,
            y=yield_real_10y * 100,
            name="10Y Real Yield (Median)",
            line=dict(color="#7B1FA2", width=2.0, dash="dashdot")
        ),
        secondary_y=False
    )

    # Secondary Axis: Equity Growth Factor ($)
    fig.add_trace(
        go.Scatter(
            x=years_growth,
            y=equity_growth_median,
            name="Equity Growth of $1.00 (Median)",
            line=dict(color="#388E3C", width=3.0, dash="solid")
        ),
        secondary_y=True
    )

    # Style layout according to actuarial presentation standards
    fig.update_layout(
        title=dict(
            text=f"<b>Expected Macroeconomic Medians ({total_scenarios:,} Scenario Simulation)</b>",
            font=dict(size=18, color="#212121"),
            x=0.05
        ),
        xaxis=dict(
            title="Projection Horizon (Years)",
            gridcolor="#E0E0E0",
            showgrid=True,
            linecolor="#9E9E9E"
        ),
        yaxis=dict(
            title="Annualized Rates / Yields (%)",
            gridcolor="#F5F5F5",
            showgrid=True,
            ticksuffix="%",
            linecolor="#9E9E9E"
        ),
        yaxis2=dict(
            title="Cumulative Equity Growth ($)",
            tickprefix="$",
            showgrid=False,
            linecolor="#9E9E9E"
        ),
        template="plotly_white",
        height=650,
        width=1200,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=60, t=100, b=60)
    )

    # Save HTML file on disk and launch the active browser window
    output_html = "esg_massive_median_paths.html"
    fig.write_html(output_html)
    print(f"✓ Interactive HTML chart exported to '{output_html}'")

    try:
        fig.show()
    except Exception:
        print("Note: Automated browser opening bypassed (headless or terminal execution).")

    # 7. Clean up memory allocations
    results.cleanup()
    print("✓ Resource clean-up complete.")

    print("\n======================================================================")
    print(" IN-MEMORY ESG EXECUTION PIPELINE COMPLETE ")
    print("======================================================================")


if __name__ == "__main__":
    run_massive_simulation_demo()
