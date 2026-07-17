import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from actuarial_esg import SimulatorConfig, MarketSimulator, SimulationResults

# --- DECUMULATION PARAMETERS ---
INITIAL_BALANCE = 1_000_000.0
ANNUAL_WITHDRAWAL_RATE = 0.04  # 4% Rule
MONTHLY_WITHDRAWAL = (INITIAL_BALANCE * ANNUAL_WITHDRAWAL_RATE) / 12.0
PORTFOLIO_WEIGHTS = {"equity": 0.60, "fixed_income": 0.40}  # Classic 60/40

# Projection Settings
YEARS = 40
NUM_SCENARIOS = 5000  # Higher scenario count for smoother solvency curves
PRESETS_DIR = "presets"

# Elegant consulting palette for chart lines
COLORS = {
    "usa": "#3B82F6",      # Bright Blue
    "europe": "#10B981",   # Emerald Green
    "japan": "#8B5CF6",    # Violet
    "world": "#64748B",    # Slate Grey
    "brazil": "#F59E0B"    # Amber/Gold
}

def load_preset_config(name: str) -> SimulatorConfig:
    """Loads regional preset and overrides runtime-specific parameters."""
    path = os.path.join(PRESETS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preset file not found: {path}. Run build_presets.py first.")
        
    with open(path, "r") as f:
        preset_dict = json.load(f)
        
    # Reconstruct config
    config = SimulatorConfig.from_dict(preset_dict)
    
    # Override standard horizon and scenario parameters for uniform comparison
    config.duration_years = YEARS
    config.num_scenarios = NUM_SCENARIOS
    config.seed = 42 # Force identical seed to compare models cleanly
    
    return config

def run_regional_decumulation(name: str):
    """Executes market simulation and decumulation for a single region."""
    print(f" -> Simulating {name.upper()} ({NUM_SCENARIOS:,} scenarios)...")
    config = load_preset_config(name)
    
    # Run ESG
    simulator = MarketSimulator(config)
    raw_scenarios = simulator.run()
    results = SimulationResults(raw_scenarios)
    
    # Simulate Retirement Portfolio Decumulation
    decum_results = results.simulate_decumulation(
        initial_balance=INITIAL_BALANCE,
        initial_monthly_withdrawal=MONTHLY_WITHDRAWAL,
        portfolio_weights=PORTFOLIO_WEIGHTS,
        liquidation_strategy="constant_mix",
        frictional_drag_annual=0.0025,  # 25 bps annual fee drag
        tax_on_gains_rate=0.15          # 15% capital gains tax on liquidated gains
    )
    
    # Clean up results cache to conserve memory
    results.cleanup()
    
    return decum_results["probability_of_success"]

def main():
    print("==================================================")
    print("   Actuarial ESG - Regional Solvency Comparison   ")
    print("==================================================")
    
    regions = ["usa", "europe", "japan", "world", "brazil"]
    solvency_curves = {}
    
    # 1. Run all regions
    for r in regions:
        try:
            solvency_curves[r] = run_regional_decumulation(r)
        except Exception as e:
            print(f" ✗ Error running {r}: {str(e)}")
            
    if not solvency_curves:
        print("No regions were successfully simulated. Exiting.")
        return

    # 2. Build Interactive Plotly Line Chart
    print("\n[Dashboard] Generating comparative line chart...")
    fig = go.Figure()
    
    timeline_years = np.arange(YEARS * 12 + 1) / 12.0
    
    # Add solvency curves to chart
    for r, curve in solvency_curves.items():
        fig.add_trace(
            go.Scatter(
                x=timeline_years,
                y=curve * 100,  # Convert to percentage
                mode="lines",
                name=f"{r.upper()} (60/40 Portfolio)",
                line=dict(color=COLORS.get(r, "#000000"), width=2.5),
                hovertemplate=f"<b>{r.upper()}</b><br>Year %{{x:.1f}}: %{{y:.1f}}% Solvent<extra></extra>"
            )
        )
        
    # Reference benchmarks
    fig.add_hline(y=80, line_dash="dash", line_color="#94A3B8", line_width=1.5,
                  annotation_text="80% Solvency Target", annotation_position="bottom right")
    fig.add_hline(y=50, line_dash="dot", line_color="#EF4444", line_width=1.5,
                  annotation_text="50% Solvency (Median Depletion Point)", annotation_position="bottom right")

    # Layout Aesthetics
    fig.update_layout(
        title=dict(
            text=f"<b>Comparative Solvency Probability over {YEARS}-Year Retirement</b><br>"
                 f"<span style='font-size: 12px; color: #64748B;'>Standard 4% Initial Withdrawal Rate (Inflated Monthly), 60/40 Asset Allocation, 15% Gains Tax</span>",
            font=dict(family="Inter, system-ui, sans-serif", size=16, color="#0F172A")
        ),
        xaxis=dict(
            title="Retirement Horizon (Years)",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False,
            range=[0, YEARS]
        ),
        yaxis=dict(
            title="Portfolio Solvency Probability (%)",
            ticksuffix="%",
            gridcolor="#F1F5F9",
            showgrid=True,
            linecolor="#E2E8F0",
            zeroline=False,
            range=[-2, 105]
        ),
        template="plotly_white",
        height=650,
        width=1100,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="right",
            x=1,
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(size=11, color="#334155")
        ),
        margin=dict(l=65, r=50, t=120, b=65)
    )

    # 3. Save to HTML and launch
    output_filename = "regional_solvency_comparison.html"
    fig.write_html(output_filename)
    print(f"✓ Interactive chart saved to: '{output_filename}'")
    
    # 4. Print Summary Terminal Table
    print("\n" + "=" * 60)
    print(f"| {'REGIONAL SOLVENCY METRICS SUMMARY (Y30 & Y40)':^56} |")
    print("=" * 60)
    
    table_data = []
    for r, curve in solvency_curves.items():
        sol_y30 = curve[30 * 12] * 100
        sol_y40 = curve[40 * 12] * 100
        table_data.append({
            "Region": r.upper(),
            "Solvency @ Yr 30": f"{sol_y30:.1f}%",
            "Solvency @ Yr 40": f"{sol_y40:.1f}%"
        })
        
    summary_df = pd.DataFrame(table_data)
    print(summary_df.to_string(index=False))
    print("=" * 60)

    try:
        fig.show()
    except Exception:
        print("Note: Automatic browser opening skipped (headless or container execution environment).")

if __name__ == "__main__":
    main()
