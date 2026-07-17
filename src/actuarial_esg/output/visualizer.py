from typing import List, Optional, Any
import numpy as np

def _import_plotly():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        return go, make_subplots
    except ImportError:
        raise ImportError(
            "Plotly is required for these visualizers. Install it using 'pip install plotly'."
        )

def plot_fan_chart(
    results,
    metric: str,
    tenor: Optional[float] = None,
    annualized: bool = False,
    title: Optional[str] = None
) -> Any:
    go, _ = _import_plotly()

    median = results.query(metric, stat="median", step="all", tenor=tenor, annualized=annualized)
    p5 = results.query(metric, stat="p5", step="all", tenor=tenor, annualized=annualized)
    p25 = results.query(metric, stat="p25", step="all", tenor=tenor, annualized=annualized)
    p75 = results.query(metric, stat="p75", step="all", tenor=tenor, annualized=annualized)
    p95 = results.query(metric, stat="p95", step="all", tenor=tenor, annualized=annualized)

    years = np.arange(len(median)) / results.steps_per_year

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years, y=p5, line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=years, y=p95, fill='tonexty', fillcolor='rgba(70, 130, 180, 0.15)',
        line=dict(width=0), name='5% - 95% Interval'
    ))

    fig.add_trace(go.Scatter(
        x=years, y=p25, line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=years, y=p75, fill='tonexty', fillcolor='rgba(70, 130, 180, 0.3)',
        line=dict(width=0), name='25% - 75% Interval'
    ))

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
    results,
    metric: str,
    num_paths: int = 15,
    tenor: Optional[float] = None,
    annualized: bool = False,
    title: Optional[str] = None
) -> Any:
    go, _ = _import_plotly()

    raw_paths = results.query(metric, stat="raw", step="all", tenor=tenor, annualized=annualized)
    mean_path = results.query(metric, stat="mean", step="all", tenor=tenor, annualized=annualized)

    years = np.arange(len(mean_path)) / results.steps_per_year
    paths_to_plot = min(num_paths, raw_paths.shape[1])

    fig = go.Figure()

    for i in range(paths_to_plot):
        fig.add_trace(go.Scatter(
            x=years, y=raw_paths[:, i],
            line=dict(color='rgba(120, 120, 120, 0.35)', width=1),
            showlegend=False, name=f"Path {i}"
        ))

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
    results,
    years_milestones: List[float] = [0.0, 1.0, 5.0, 15.0, 30.0],
    real: bool = False,
    title: Optional[str] = None
) -> Any:
    go, _ = _import_plotly()

    metric = "real_yield" if real else "nominal_yield"
    fig = go.Figure()

    for y in years_milestones:
        yields_for_tenors = [
            results.query(metric, stat="mean", year=y, tenor=t)
            for t in results.tenors
        ]
        fig.add_trace(go.Scatter(
            x=results.tenors, y=np.array(yields_for_tenors) * 100,
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
    results,
    metric: str,
    target_year: float,
    tenor: Optional[float] = None,
    annualized: bool = False,
    bins: int = 40,
    title: Optional[str] = None
) -> Any:
    go, _ = _import_plotly()

    raw_data = results.query(metric, stat="raw", year=target_year, tenor=tenor, annualized=annualized)
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
