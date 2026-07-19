import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from aethel import MarketSimulator, SimulatorConfig, SimulationResults


def run_visualization_suite():
    print("==================================================")
    print("      Aethel ESG - Interactive Dashboard          ")
    print("==================================================")

    # 1. Setup a standard 30-year configuration with standardized initial inputs
    config = SimulatorConfig(
        duration_years=30,
        num_scenarios=500,
        seed=42,
        initial_rate=0.105,      # Starting short rate at 10.5%
        initial_inflation=0.045  # Starting inflation at 4.5%
    )

    print(f"Configured simulation for {config.duration_years} years.")
    print(f"Running engine over {config.num_scenarios} scenarios...")

    # 2. Execute simulation
    simulator = MarketSimulator(config)
    raw_scenarios = simulator.run()

    # 3. Wrap results
    results = SimulationResults(raw_scenarios)

    print("\n[Visualization] Assembling unified dashboard layout...")

    # Generate the 4 individual charts
    fig_fan = results.plot_fan_chart(metric="cpi", title="CPI Fan Chart (Projections of $1.00)")
    fig_paths = results.plot_scenario_paths(metric="rate", num_paths=15, title="Sample Short Rate Trajectories")
    fig_curves = results.plot_yield_curve_evolution(years_milestones=[0.0, 1.0, 5.0, 15.0, 30.0], title="Expected Term Structure Evolution")
    fig_dist = results.plot_horizon_distribution(metric="equity_growth", target_year=30.0, bins=50, title="Distribution of Equity Growth at Year 30")

    # 4. Initialize a 2x2 Subplot Figure
    dashboard = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "CPI Fan Chart (Projections of $1.00)",
            "Sample Short Rate Trajectories",
            "Expected Term Structure Evolution",
            "Distribution of Equity Growth ($1.00 base) at Year 30"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.08
    )

    # Map Traces into target Subplot Quadrants
    for trace in fig_fan.data:
        dashboard.add_trace(trace, row=1, col=1)

    for trace in fig_paths.data:
        dashboard.add_trace(trace, row=1, col=2)

    for trace in fig_curves.data:
        dashboard.add_trace(trace, row=2, col=1)

    for trace in fig_dist.data:
        dashboard.add_trace(trace, row=2, col=2)

    # 5. Extract statistics for vertical lines
    mean_val = results.query("equity_growth", stat="mean", year=30.0)
    med_val = results.query("equity_growth", stat="median", year=30.0)

    dashboard.add_vline(x=mean_val, line_dash="dash", line_color="red", line_width=1.5,
                        row=2, col=2, annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
    dashboard.add_vline(x=med_val, line_dash="dot", line_color="green", line_width=1.5,
                        row=2, col=2, annotation_text=f"Median: {med_val:.2f}", annotation_position="top left")

    # 6. Apply coordinated axis labels and layout spacing
    dashboard.update_xaxes(title_text="Projection Horizon (Years)", row=1, col=1)
    dashboard.update_yaxes(title_text="Index Value", row=1, col=1)

    dashboard.update_xaxes(title_text="Projection Horizon (Years)", row=1, col=2)
    dashboard.update_yaxes(title_text="Annualized Rate", row=1, col=2)

    dashboard.update_xaxes(title_text="Tenor (Years)", row=2, col=1)
    dashboard.update_yaxes(title_text="Average Yield (%)", row=2, col=1)

    dashboard.update_xaxes(title_text="Growth Factor ($)", row=2, col=2)
    dashboard.update_yaxes(title_text="Density", row=2, col=2)

    dashboard.update_layout(
        title_text="Aethel Economic Scenario Generator (ESG) Dashboard",
        height=950,
        width=1450,
        template="plotly_white",
        showlegend=True,
        margin=dict(l=50, r=50, t=100, b=50)
    )

    dashboard.show()

    print("\nDashboard deployed. Review the active web browser tab for the consolidated layout.")
    print("==================================================")


if __name__ == "__main__":
    run_visualization_suite()
