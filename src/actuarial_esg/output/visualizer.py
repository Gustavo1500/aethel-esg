from typing import List, Optional, Any, Tuple, Callable
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

def _get_metric_formatter(metric: str, annualized: bool) -> Tuple[Callable[[float], str], str, str]:
    """
    Returns an appropriate text-formatter, Plotly hover-template format string,
    and y-axis tick-format based on the characteristics of the target metric.
    """
    m = metric.lower()
    is_percent = annualized or any(x in m for x in ["rate", "yield", "cdi", "inflation", "return", "drag", "tax"])
    is_currency = any(x in m for x in ["balance", "nest_egg", "withdrawal", "tax_paid"])
    
    if is_percent:
        return lambda v: f"{v * 100:.2f}%", "%{y:.2%}", ".1%"
    elif is_currency:
        return lambda v: f"${v:,.2f}", "$%{y:,.2f}", "$,.0f"
    else:
        return lambda v: f"{v:.2f}", "%{y:.2f}", ".2f"

def plot_fan_chart(
    results,
    metric: str,
    tenor: Optional[float] = None,
    annualized: bool = False,
    title: Optional[str] = None
) -> Any:
    go, _ = _import_plotly()
    formatter, y_format, tick_format = _get_metric_formatter(metric, annualized)

    # 1. Fetch statistical percentiles
    median = results.query(metric, stat="median", step="all", tenor=tenor, annualized=annualized)
    p5 = results.query(metric, stat="p5", step="all", tenor=tenor, annualized=annualized)
    p25 = results.query(metric, stat="p25", step="all", tenor=tenor, annualized=annualized)
    p75 = results.query(metric, stat="p75", step="all", tenor=tenor, annualized=annualized)
    p95 = results.query(metric, stat="p95", step="all", tenor=tenor, annualized=annualized)

    years = np.arange(len(median)) / results.steps_per_year
    fig = go.Figure()

    # 2. Add Confidence Interval Areas with modern transparencies & Legend Grouping
    # 5% to 95% Confidence Band (Outer)
    fig.add_trace(go.Scatter(
        x=years, y=p5, line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=years, y=p95, fill='tonexty', fillcolor='rgba(203, 213, 225, 0.25)',
        line=dict(width=0), name='5% - 95% Confidence Band', legendgroup="ci"
    ))

    # 25% to 75% Confidence Band (Inner)
    fig.add_trace(go.Scatter(
        x=years, y=p25, line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=years, y=p75, fill='tonexty', fillcolor='rgba(148, 163, 184, 0.45)',
        line=dict(width=0), name='25% - 75% Confidence Band', legendgroup="ci"
    ))

    # 3. Add solid Deep Slate Median Path
    fig.add_trace(go.Scatter(
        x=years, y=median, line=dict(color='#0F172A', width=3),
        name='Median Trajectory', hovertemplate=y_format
    ))

    # 4. Premium layout updates (Inter Font, Slate Gridlines, Spaced Margins)
    fig.update_layout(
        title=dict(
            text=title if title else f"Projections: {metric.replace('_', ' ').title()}",
            font=dict(size=16, color="#0F172A", family="Inter, system-ui, sans-serif")
        ),
        xaxis=dict(
            title="Projection Horizon (Years)",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False
        ),
        yaxis=dict(
            title="Value" if not annualized else "Annualized Rate",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False,
            tickformat=tick_format
        ),
        template="plotly_white",
        hovermode="x unified",
        margin=dict(l=50, r=50, t=100, b=50),  # Increased t to 100 for title breathing room
        legend=dict(
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#334155")  # Decreased to 10 for de-cluttering
        )
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
    formatter, y_format, tick_format = _get_metric_formatter(metric, annualized)

    raw_paths = results.query(metric, stat="raw", step="all", tenor=tenor, annualized=annualized)
    mean_path = results.query(metric, stat="mean", step="all", tenor=tenor, annualized=annualized)

    years = np.arange(len(mean_path)) / results.steps_per_year
    paths_to_plot = min(num_paths, raw_paths.shape[1])

    fig = go.Figure()

    # 1. Add delicate, transparent background trajectories
    for i in range(paths_to_plot):
        fig.add_trace(go.Scatter(
            x=years, y=raw_paths[:, i],
            mode="lines",
            line=dict(color='rgba(148, 163, 184, 0.25)', width=1.2),
            showlegend=False, name=f"Scenario Path {i}",
            hovertemplate=y_format
        ))

    # 2. Highlight the Expected Mean as a bold, dashed Slate Line
    fig.add_trace(go.Scatter(
        x=years, y=mean_path,
        mode="lines",
        line=dict(color='#475569', width=2.5, dash='dash'),
        name='Expected Mean',
        hovertemplate=y_format
    ))

    # 3. Modern Layout Tuning
    fig.update_layout(
        title=dict(
            text=title if title else f"Scenario Sample Paths: {metric.replace('_', ' ').title()}",
            font=dict(size=16, color="#0F172A", family="Inter, system-ui, sans-serif")
        ),
        xaxis=dict(
            title="Projection Horizon (Years)",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False
        ),
        yaxis=dict(
            title="Value" if not annualized else "Annualized Rate",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False,
            tickformat=tick_format
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=100, b=50),  # Increased t to 100 for spacing
        legend=dict(
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#334155")  # Decreased to 10 for de-cluttering
        )
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

    # Dynamic Consulting Color Cycle transitioning from soft Slate to bold Emerald/Teal
    colors_preset = ["#94A3B8", "#64748B", "#475569", "#0F172A", "#0D9488", "#F59E0B"]

    for idx, y in enumerate(years_milestones):
        yields_for_tenors = [
            results.query(metric, stat="mean", year=y, tenor=t)
            for t in results.tenors
        ]
        
        color = colors_preset[idx % len(colors_preset)]
        fig.add_trace(go.Scatter(
            x=results.tenors, y=np.array(yields_for_tenors) * 100,
            mode='lines+markers',
            name=f"Horizon: Year {y:.1f}",
            line=dict(color=color, width=2.2),
            marker=dict(size=6, symbol="circle"),
            hovertemplate="Tenor %{x}y: %{y:.2f}%"
        ))

    curve_type = "Real" if real else "Nominal"
    fig.update_layout(
        title=dict(
            text=title if title else f"Expected {curve_type} Yield Curve Evolution",
            font=dict(size=16, color="#0F172A", family="Inter, system-ui, sans-serif")
        ),
        xaxis=dict(
            title="Tenor (Years)",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False
        ),
        yaxis=dict(
            title="Average Yield (%)",
            ticksuffix="%",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=100, b=50),  # Increased t to 100 for spacing
        legend=dict(
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=10, color="#334155")  # Decreased to 10 for de-cluttering
        )
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
    formatter, y_format, tick_format = _get_metric_formatter(metric, annualized)

    # 1. Fetch raw data points and calculate summary statistics
    raw_data = results.query(metric, stat="raw", year=target_year, tenor=tenor, annualized=annualized)
    mean_val = np.mean(raw_data)
    med_val = np.median(raw_data)

    # 2. Determine Risk Direction and compute Value-at-Risk / Tail Value-at-Risk
    is_downside_risk = any(x in metric.lower() for x in ["growth", "return", "balance", "nest_egg", "withdrawal"])

    if is_downside_risk:
        var_val = np.percentile(raw_data, 5.0)
        tvar_val = np.mean(raw_data[raw_data <= var_val])
        risk_label = "95% Value-at-Risk (VaR)"
        risk_color = "#EF4444"  # Red
    else:
        var_val = np.percentile(raw_data, 95.0)
        tvar_val = np.mean(raw_data[raw_data >= var_val])
        risk_label = "95% Value-at-Risk (Upside VaR)"
        risk_color = "#F97316"  # Orange

    fig = go.Figure()

    # 3. Add modern Density Histogram with smooth borders
    fig.add_trace(go.Histogram(
        x=raw_data, nbinsx=bins, histnorm='density',
        marker=dict(color='#64748B', line=dict(color='#F8FAFC', width=0.5)),
        opacity=0.85, name="Scenario Outcome Density"
    ))

    # 4. Add key statistical vertical markers
    fig.add_vline(x=mean_val, line_dash="dash", line_color="#475569", line_width=1.5,
                  annotation_text=f"Mean: {formatter(mean_val)}", annotation_position="top right")
    fig.add_vline(x=med_val, line_dash="dot", line_color="#0F172A", line_width=1.5,
                  annotation_text=f"Median: {formatter(med_val)}", annotation_position="top left")
    fig.add_vline(x=var_val, line_dash="solid", line_color=risk_color, line_width=2.0)

    # 5. Create a professional floating metric annotation card
    summary_text = (
        f"<b>PROJECTION METRIC PROFILE (Y{target_year:.1f})</b><br>"
        f"Mean: {formatter(mean_val)}<br>"
        f"Median: {formatter(med_val)}<br>"
        f"Std Dev: {np.std(raw_data):.2%}<br>"
        f"<span style='color:{risk_color};'><b>{risk_label}:</b> {formatter(var_val)}</span><br>"
        f"<span style='color:{risk_color};'><b>95% TVaR:</b> {formatter(tvar_val)}</span>"
    )

    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.98, y=0.95,
        text=summary_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.92)",
        bordercolor="#E2E8F0",
        borderwidth=1.5,
        borderpad=8,
        font=dict(size=11, color="#334155", family="Inter, system-ui, sans-serif")
    )

    # 6. Apply Tail risk shading under the curve
    shapes = []
    if is_downside_risk:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=min(raw_data), x1=var_val,
            y0=0, y1=1,
            fillcolor="rgba(239, 68, 68, 0.08)",
            line_width=0, layer="below"
        ))
    else:
        shapes.append(dict(
            type="rect", xref="x", yref="paper",
            x0=var_val, x1=max(raw_data),
            y0=0, y1=1,
            fillcolor="rgba(249, 115, 22, 0.08)",
            line_width=0, layer="below"
        ))

    # 7. Apply structural styling
    fig.update_layout(
        title=dict(
            text=title if title else f"Outcome Distribution of {metric.replace('_', ' ').title()} at Year {target_year}",
            font=dict(size=16, color="#0F172A", family="Inter, system-ui, sans-serif")
        ),
        xaxis=dict(
            title="Value" if not annualized else "Annualized Rate",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False,
            tickformat=tick_format
        ),
        yaxis=dict(
            title="Relative Density",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False
        ),
        template="plotly_white",
        margin=dict(l=50, r=50, t=100, b=50),
        shapes=shapes
    )

    return fig
