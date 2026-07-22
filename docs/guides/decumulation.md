# Retirement Decumulation Simulator

The Aethel decumulation engine models monthly retirement portfolio cash flows under stochastic pathways. This guide explains the core liquidation strategies, cost-basis accounting, taxation mechanics, withdrawal timing, and how to apply custom spending policies using code examples.

---

## 1. Liquidation Strategies

When executing withdrawals from a multi-asset portfolio, the engine supports two distinct liquidation methods:

### Strategy A: Constant Mix (`constant_mix`)
* **Behavior:** The portfolio is rebalanced back to its target weights (e.g., $60\%$ Equity / $40\%$ Fixed Income) at the start of each month.
* **Execution:** Withdrawals are taken proportionally from all asset classes to maintain the target asset mix.
* **Mathematical representation of the target allocation:**
  <div class="arithmatex">
  \[
  E_{t} = B_{t} \times w_{\text{equity}}, \quad F_{t} = B_{t} \times (1 - w_{\text{equity}})
  \]
  </div>

### Strategy B: Cash First (`cash_first`)
* **Behavior:** Withdrawals are drawn from the fixed-income allocation first. Equity holdings are left untouched unless the fixed-income balance is entirely depleted.
* **Execution:** If the required withdrawal exceeds the available fixed income, the remaining balance is liquidated from the equity portion. This strategy acts as a buffer against sequence-of-returns risk during equity market drawdowns.

---

## 2. Mathematical Framework

The simulation updates portfolio values, taxes, fees, and cost basis monthly using the following sequential steps:

### Inflation Adjustment
If `inflate_withdrawals=True` is configured, the nominal net withdrawal target for month $t$ is adjusted based on the Cumulative Consumer Price Index ($\text{CPI}$) from the preceding month:

<div class="arithmatex">
\[
W_{\text{net}, t} = W_{\text{initial}} \times \text{CPI}_{t-1}
\]
</div>

### Frictional Drag Application
Investment management fees or structural advisor costs are represented by the annualized parameter $f_{\text{annual}}$. This is converted to a monthly rate and applied directly as a deduction from asset returns:

<div class="arithmatex">
\[
f_m = \frac{f_{\text{annual}}}{12}
\]
</div>

### Cost-Basis and Capital Gains Taxation
When a capital gains tax rate ($\tau$) is applied via `tax_on_gains_rate`, the engine tracks the cumulative unliquidated principal (cost basis, $C_t$).

If the pre-withdrawal portfolio value ($W_{\text{pre}}$) is greater than the remaining cost basis ($C_t$), a profit has occurred. The proportion of the portfolio representing taxable gain is defined by the **Gain Ratio**:

<div class="arithmatex">
\[
\text{Gain Ratio}_t = \max\left(0, \frac{W_{\text{pre}} - C_t}{W_{\text{pre}}}\right)
\]
</div>

To ensure the retiree receives the exact target net withdrawal ($W_{\text{net}, t}$) after tax has been withheld, the required **Gross Withdrawal** is calculated as:

<div class="arithmatex">
\[
W_{\text{gross}, t} = \frac{W_{\text{net}, t}}{1.0 - (\tau \times \text{Gain Ratio}_t)}
\]
</div>

The cost basis is then reduced proportionally to the share of the total portfolio liquidated:

<div class="arithmatex">
\[
C_{t+1} = C_t - C_t \times \left(\frac{W_{\text{gross}, t}}{W_{\text{pre}}}\right)
\]
</div>

> **Note on Cost-Basis Accounting Scope:** 
> The decumulation engine uses an **aggregate portfolio-wide cost basis** (modeled after the average cost method across total balances) rather than asset-specific or lot-specific tracking (such as FIFO, LIFO, or HIFO). All tax calculations assume that withdrawals represent a proportional mix of capital return and capital gains across the entire portfolio balance ($W_{\text{pre}}$).

---

## 3. Withdrawal Timing Logic

The parameter `withdrawal_timing` determines the chronological order of transactions within each monthly step:

### Beginning of Period (`"beginning"` / `"advance"`)
Withdrawals are subtracted from asset balances at the *start* of the month before market growth or fees are applied.
1. Withdrawals are deducted from current balances ($E_t$, $F_t$) to establish remaining post-withdrawal balances.
2. Cost basis is adjusted downward based on starting balances.
3. Remaining balances are multiplied by the asset return rates for that period:
   <div class="arithmatex">
   \[
   E_{t+1} = E_{\text{remaining}} \times (1 + R_{E, t} - f_m)
   \]
   \[
   F_{t+1} = F_{\text{remaining}} \times (1 + R_{F, t} - f_m)
   \]
   </div>

### End of Period (`"end"`)
Asset balances grow first, and withdrawals are subtracted from the grown balances at the *end* of the month.
1. Asset values are updated with monthly returns and fees:
   <div class="arithmatex">
   \[
   E_{\text{pre}} = E_t \times (1 + R_{E, t} - f_m)
   \]
   \[
   F_{\text{pre}} = F_t \times (1 + R_{F, t} - f_m)
   \]
   </div>
2. Withdrawals are executed against $E_{\text{pre}}$ and $F_{\text{pre}}$.
3. Cost basis is adjusted downward based on the pre-withdrawal values.

---

## 4. Practical Implementation Examples

Below are three code examples demonstrating how to configure and execute decumulation simulations under different planning scenarios.

### Example 1: Standard 4% Rule with Constant Mix Rebalancing
This example runs a baseline decumulation simulation using a 60/40 portfolio rebalanced monthly, adjusting withdrawals for inflation without taxes or fee drag.

```python
import numpy as np
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

# 1. Generate underlying macroeconomic paths
config = SimulatorConfig(duration_years=30, num_scenarios=2000, seed=42)
simulator = MarketSimulator(config)
results = SimulationResults(simulator.run())

# 2. Run decumulation simulation (4% rule on $1M = $40,000 annually)
decum_baseline = results.simulate_decumulation(
    initial_balance=1000000.0,
    initial_monthly_withdrawal=3333.33,
    portfolio_weights={"equity": 0.60, "fixed_income": 0.40},
    liquidation_strategy="constant_mix",
    withdrawal_timing="beginning",
    inflate_withdrawals=True
)

# 3. Analyze probability of success over the horizon
solvency_by_year = decum_baseline["probability_of_success"]
print(f"Year 10 Solvency: {solvency_by_year[120] * 100:.2f}%")
print(f"Year 20 Solvency: {solvency_by_year[240] * 100:.2f}%")
print(f"Year 30 Solvency: {solvency_by_year[360] * 100:.2f}%")
```

---

### Example 2: Comparing Constant Mix vs. Cash-First Liquidation
This example evaluates the impact of changing the liquidation strategy to Cash-First, preserving equities during market fluctuations.

```python
import numpy as np
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

config = SimulatorConfig(duration_years=30, num_scenarios=2000, seed=123)
simulator = MarketSimulator(config)
results = SimulationResults(simulator.run())

# Strategy A: Constant Mix
res_constant_mix = results.simulate_decumulation(
    initial_balance=1000000.0,
    initial_monthly_withdrawal=4500.0, # $54k annual withdrawal rate
    portfolio_weights={"equity": 0.60, "fixed_income": 0.40},
    liquidation_strategy="constant_mix"
)

# Strategy B: Cash First
res_cash_first = results.simulate_decumulation(
    initial_balance=1000000.0,
    initial_monthly_withdrawal=4500.0,
    portfolio_weights={"equity": 0.60, "fixed_income": 0.40},
    liquidation_strategy="cash_first"
)

success_constant = res_constant_mix["probability_of_success"][-1] * 100
success_cash = res_cash_first["probability_of_success"][-1] * 100

print(f"Solvency Rate (Constant Mix): {success_constant:.2f}%")
print(f"Solvency Rate (Cash-First):   {success_cash:.2f}%")
```

---

### Example 3: SWR with Capital Gains Tax, Management Fees, and Spending Guardrails
This example simulates a highly realistic scenario incorporating a 15% capital gains tax, a 35 basis point annual asset management fee, and a dynamic spending policy that reduces withdrawals if the portfolio value falls below a specific threshold.

```python
import numpy as np
from aethel import SimulatorConfig, MarketSimulator, SimulationResults

config = SimulatorConfig(duration_years=30, num_scenarios=2000, seed=42)
simulator = MarketSimulator(config)
results = SimulationResults(simulator.run())

# Define a vectorized spending guardrail rule
def dynamic_spending_policy(balance, cpi_factor, step, deposit_rate):
    """
    Standard monthly withdrawal is $4,000 (adjusted for inflation).
    If the portfolio balance falls below $300,000, withdrawals are reduced by 25%.
    """
    baseline_withdrawal = 4000.0 * cpi_factor
    is_below_threshold = balance < 300000.0
    return np.where(is_below_threshold, baseline_withdrawal * 0.75, baseline_withdrawal)

# Execute decumulation simulation with drag, taxes, and the policy
decum_advanced = results.simulate_decumulation(
    initial_balance=1000000.0,
    initial_monthly_withdrawal=4000.0,
    portfolio_weights={"equity": 0.70, "fixed_income": 0.30},
    liquidation_strategy="cash_first",
    withdrawal_timing="beginning",
    inflate_withdrawals=True,
    frictional_drag_annual=0.0035,   # 35 bps annual drag
    tax_on_gains_rate=0.15,          # 15% capital gains tax
    withdrawal_policy=dynamic_spending_policy
)

# Extract and report performance
success_rate = decum_advanced["probability_of_success"][-1] * 100
final_balances = decum_advanced["balances"][-1, :]
median_surviving_balance = np.median(final_balances[final_balances > 0])

print(f"Final Solvency Rate: {success_rate:.2f}%")
print(f"Median Remaining Balance (Surviving Scenarios): ${median_surviving_balance:,.2f}")
```
