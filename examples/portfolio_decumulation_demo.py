import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from actuarial_esg import SimulatorConfig, MarketSimulator, SimulationResults


def run_comprehensive_decumulation_demo():
    print("======================================================================")
    print("      ACTUARIAL ESG - COMPREHENSIVE RETIREMENT DECUMULATION DEMO      ")
    print("======================================================================")

    # 1. Setup economic simulation configuration
    config = SimulatorConfig(
        duration_years=40,      # Retirement planning horizon
        num_scenarios=10000,     # Run 10,000 simulated futures
        seed=42,                # Ensure reproducibility
        initial_cdi=0.105,      # Starting short-rate at 10.5%
        initial_ipca=0.045      # Starting annual inflation at 4.5%
    )

    print(f"Configuring ESG projection: {config.duration_years} years ({config.steps} months)")
    print(f"Generating paths for {config.num_scenarios:,} scenarios...")
    
    simulator = MarketSimulator(config)
    raw_scenarios = simulator.run()
    results = SimulationResults(raw_scenarios)
    print("✓ Economic Scenario Generation complete.")

    # 2. Setup Baseline Parameters
    portfolio_weights = {"equity": 0.60, "fixed_income": 0.40}
    initial_balance = 1_000_000.0
    initial_monthly_withdrawal = 4_000.0

    # 3. Define a Vectorized Custom Spending Policy (Case 4)
    # If the portfolio balance falls below $350k, we scale spending back by 20% to prevent ruin.
    def variable_spending_guardrail(balance, cpi_factor, step, deposit_rate):
        base_w = initial_monthly_withdrawal * cpi_factor
        is_depleted = balance < 350_000.0
        # Returns a vectorized array of withdrawals matching the current state of each scenario
        return np.where(is_depleted, base_w * 0.80, base_w)

    print("\n[Simulation] Evaluating 4 Comparative Decumulation Strategies...")

    # --- CASE 1: Baseline (Unregulated, Constant Mix, Untaxed, No Drag) ---
    case_1 = results.simulate_decumulation(
        initial_balance=initial_balance,
        initial_monthly_withdrawal=initial_monthly_withdrawal,
        portfolio_weights=portfolio_weights,
        liquidation_strategy="constant_mix"
    )

    # --- CASE 2: Constant Mix with Taxes & Frictional Drag ---
    case_2 = results.simulate_decumulation(
        initial_balance=initial_balance,
        initial_monthly_withdrawal=initial_monthly_withdrawal,
        portfolio_weights=portfolio_weights,
        liquidation_strategy="constant_mix",
        frictional_drag_annual=0.0034,  # 34 bps annual drag
        tax_on_gains_rate=0.15          # 15% capital gains tax on profits
    )

    # --- CASE 3: Cash-First with Taxes & Frictional Drag ---
    case_3 = results.simulate_decumulation(
        initial_balance=initial_balance,
        initial_monthly_withdrawal=initial_monthly_withdrawal,
        portfolio_weights=portfolio_weights,
        liquidation_strategy="cash_first",  # Sequence of Returns risk mitigation
        frictional_drag_annual=0.0034,
        tax_on_gains_rate=0.15
    )

    # --- CASE 4: Cash-First with Guardrail Policy, Taxes, and Drag ---
    case_4 = results.simulate_decumulation(
        initial_balance=initial_balance,
        initial_monthly_withdrawal=initial_monthly_withdrawal,
        portfolio_weights=portfolio_weights,
        liquidation_strategy="cash_first",
        frictional_drag_annual=0.0034,
        tax_on_gains_rate=0.15,
        withdrawal_policy=variable_spending_guardrail  # Dynamic withdrawal policy
    )

    # 4. Generate Comparative Metrics
    comparison_data = []
    cases = [
        ("Case 1: Baseline (Constant-Mix, No Friction/Tax)", case_1),
        ("Case 2: Constant-Mix (with Drag & 15% Gains Tax)", case_2),
        ("Case 3: Cash-First Liquidation (with Drag & Tax)", case_3),
        ("Case 4: Variable Spending Guardrail (Cash-First, Tax, Drag)", case_4),
    ]

    for name, data in cases:
        solvency_y30 = data["probability_of_success"][-1] * 100
        median_ending_val = np.percentile(data["balances"][-1, :], 50)
        mean_tax_paid = np.mean(np.sum(data["taxes_paid"], axis=0))
        mean_withdrawn_real = np.mean(np.sum(data["withdrawals"], axis=0))

        comparison_data.append({
            "Strategy": name,
            "Solvency Y30 (%)": f"{solvency_y30:.1f}%",
            "Median Ending Nest Egg": f"${median_ending_val:,.2f}",
            "Mean Lifetime Taxes Paid": f"${mean_tax_paid:,.2f}",
            "Mean Total Lifetime Net Received": f"${mean_withdrawn_real:,.2f}"
        })

    # Display Comparative Table
    df = pd.DataFrame(comparison_data)
    print("\n" + "=" * 115)
    print(f"| {f'DECUMULATION STRATEGY COMPARATIVE PERFORMANCE METRICS':^111} |")
    print("=" * 115)
    print(df.to_string(index=False))
    print("=" * 115 + "\n")

    # 5. Build Comparative Chart
    print("[Dashboard] Building multi-path interactive comparison charts...")
    years = np.arange(config.steps + 1) / 12

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Nest Egg Survival/Solvency Probability over Horizon",
            "Median Real Nest Egg Depletion Paths (P50)"
        ),
        horizontal_spacing=0.10
    )

    colors = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c"]

    for idx, (name, data) in enumerate(cases):
        short_name = name.split(" (")[0]
        color = colors[idx]

        # Left Panel (1, 1): Solvency curves over time
        fig.add_trace(
            go.Scatter(
                x=years,
                y=data["probability_of_success"] * 100,
                mode="lines",
                name=f"{short_name} Solvency",
                line=dict(color=color, width=2.5),
                hovertemplate=f"<b>{short_name}</b><br>Year %{{x:.1f}}: %{{y:.1f}}% Solvent<extra></extra>"
            ),
            row=1, col=1
        )

        # Right Panel (1, 2): Median real depletion path
        p50_bal = np.percentile(data["balances"], 50, axis=1)
        fig.add_trace(
            go.Scatter(
                x=years,
                y=p50_bal,
                mode="lines",
                name=f"{short_name} Balances (P50)",
                line=dict(color=color, width=2.5, dash="dash" if idx > 1 else "solid"),
                showlegend=False,
                hovertemplate=f"<b>{short_name}</b><br>Year %{{x:.1f}}: $%{{y:,.2f}}<extra></extra>"
            ),
            row=1, col=2
        )

    # Style Dashboard Layout
    fig.update_xaxes(title_text="Horizon (Years)", row=1, col=1)
    fig.update_yaxes(title_text="Solvency Success Rate (%)", ticksuffix="%", range=[0, 105], row=1, col=1)

    fig.update_xaxes(title_text="Horizon (Years)", row=1, col=2)
    fig.update_yaxes(title_text="Portfolio Value ($)", tickprefix="$", row=1, col=2)

    fig.update_layout(
        title_text="<b>Actuarial ESG - Comparative Decumulation Analysis Dashboard</b>",
        height=550,
        width=1350,
        template="plotly_white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=60, t=100, b=60)
    )

    output_html = "esg_advanced_retirement_comparisons.html"
    fig.write_html(output_html)
    print(f"✓ Advanced interactive retirement comparison dashboard saved to '{output_html}'")

    try:
        fig.show()
    except Exception:
         print("Note: Automated browser opening bypassed (headless or terminal execution).")

    # Clean up results cache
    results.cleanup()


if __name__ == "__main__":
    run_comprehensive_decumulation_demo()
