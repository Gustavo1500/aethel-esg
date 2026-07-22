# Mathematical Framework

The Aethel engine generates correlated, multi-path stochastic simulations of macroeconomic indicators and asset classes using three continuous-time stochastic processes.

---

## 1. Short Rates: Cox-Ingersoll-Ross (CIR) Model

Nominal short-term interest rates are modeled using the CIR square-root diffusion process. This model enforces mean reversion and prevents the nominal short rate from dropping below zero under standard configurations:

$$
dr_t = \theta_r (\mu_{r,t} - r_t) dt + \sigma_r \sqrt{r_t} dW_{1,t}
$$

Where:

*   \(r_t\) is the nominal short rate at time \(t\).
*   \(\theta_r\) is the speed of mean reversion.
*   \(\mu_{r,t}\) is the time-varying rate target, representing central bank monetary intervention.
*   \(\sigma_r\) is the short-rate volatility parameter.
*   \(W_{1,t}\) is a standard Brownian motion.

### Central Bank Policy Rule
The parameter \(\mu_{r,t}\) is not static. It dynamically adjusts according to a Taylor-like feedback rule based on the smoothed inflation rate:

$$
\mu_{r,t} = r^{\text{real}}_{\text{target}, t} + s_t + \gamma_t (s_t - \pi_{\text{target}, t})
$$

Where:
*   \(r^{\text{real}}_{\text{target}, t}\) is the structural real interest rate target.
*   \(s_t\) is the smoothed (exponentially moving-averaged) inflation rate.
*   \(\pi_{\text{target}, t}\) is the explicit target inflation rate.
*   \(\gamma_t\) is the policy response coefficient to inflation deviations.

---

## 2. Inflation: Ornstein-Uhlenbeck (OU) / Shifted-CIR Model

Consumer price index (CPI) dynamics are simulated with a mean-reverting process that incorporates central bank monetary policy feedback loops:

$$
dy_t = \theta_{\pi} (y_{\text{target},t} - y_t) dt + \sigma_{\pi} \sqrt{y_t} dW_{2,t}
$$

Where the actual annualized inflation rate \(\pi_t\) is derived as:

$$
\pi_t = y_t + \pi_{\text{min}}
$$

The feedback variable \(y_{\text{target},t}\) represents target-adjusting central bank behavior. It dynamically tightens or loosens based on the nominal interest rate gap to model economic drag:

$$
y_{\text{target},t} = (\mu_{\pi} - \pi_{\text{min}}) - \beta_{\text{drag}} (r_t - s_t)
$$

Where:

*   \(\mu_{\pi}\) is the long-term structural inflation target.
*   \(\pi_{\text{min}}\) is the minimum possible inflation rate (deflation floor).
*   \(s_t\) is the smoothed inflation rate at time \(t\).
*   \(\beta_{\text{drag}}\) represents the drag coefficient of monetary intervention.
*   \(W_{2,t}\) is a standard Brownian motion.

---

## 3. Equities: Merton Jump-Diffusion Model

Stock returns are modeled utilizing a Merton Jump-Diffusion formulation. This structure combines log-return growth dynamics with asymmetric Poisson-driven market shocks (jumps) to represent systemic downturns:

$$
\frac{dS_t}{S_{t^-}} = (r_t + \eta_{\text{ERP},t}) dt + \sigma_S dW_{3,t} + d\left( \sum_{i=1}^{N_t} (Y_i - 1) \right)
$$

Where:

*   \(\sigma_S\) is the continuous volatility of the stock process.
*   \(N_t\) is a Poisson process with jump intensity parameter \(\lambda_J\).
*   \(Y_i\) is the random jump size factor, where log-jump returns are normally distributed: \(\ln(Y_i) \sim \mathcal{N}(\mu_J, \sigma_J^2)\).
*   \(W_{3,t}\) is a standard Brownian motion.

### Dynamic Equity Risk Premium
The Equity Risk Premium \(\eta_{\text{ERP},t}\) adjusts dynamically to reflect the macroeconomic regime, expanding or contracting based on the deviation of inflation from the policy target:

$$
\eta_{\text{ERP},t} = \eta_{\text{base}} + \eta_{\text{erp}} (s_t - \pi_{\text{target}, t})
$$

### Discrete Time Step Log-Return Representation
For discrete-time path generation, the log-return over a time step \(dt\) is computed as:

$$
\ln\left(\frac{S_{t+dt}}{S_t}\right) = \left( r_t + \eta_{\text{ERP},t} - \lambda_J k_J - \frac{1}{2}\sigma_S^2 \right) dt + \sigma_S \sqrt{dt} Z_{3,t} + \sum_{i=1}^{N_{dt}} \ln(Y_i)
$$

Where:

*   \(Z_{3,t} \sim \mathcal{N}(0, 1)\) is the standard normal shock for the continuous component.
*   \(N_{dt} \sim \text{Poisson}(\lambda_J dt)\) is the number of jump events occurring in the interval.
*   \(k_J\) is the expected relative jump size.

The expected relative jump size \(k_J\) is derived analytically as:

$$
k_J = \mathbb{E}[Y_i - 1] = \exp\left(\mu_J + \frac{1}{2}\sigma_J^2\right) - 1
$$

---

## 4. Stochastic Correlation Matrix

The three standard Brownian motions (\(W_{1,t}\), \(W_{2,t}\), and \(W_{3,t}\)) are correlated using a Cholesky decomposition of the symmetric correlation matrix:

$$
\Sigma = \begin{pmatrix} 1 & \rho_{12} & \rho_{13} \\ \rho_{12} & 1 & \rho_{23} \\ \rho_{13} & \rho_{23} & 1 \end{pmatrix}
$$

Where:

*   \(\rho_{12}\) represents the short rate and inflation correlation.
*   \(\rho_{13}\) represents the short rate and equity return correlation.
*   \(\rho_{23}\) represents the inflation and equity return correlation.

---

## 5. Term Structure and Yield Curve Derivation

The Aethel engine generates entire term structures for nominal and real risk-free rates across arbitrary maturities (tenors) on-the-fly. This is done analytically to preserve memory capacity across high scenario counts.

### Nominal Yield Curve (Analytical CIR Formulation)
Under the Cox-Ingersoll-Ross framework, the price of a nominal zero-coupon bond maturing in \(\tau\) years, \(P(t, \tau)\), can be evaluated analytically. The nominal yield \(Y(t, \tau)\) for tenor \(\tau\) at time \(t\) is defined as:

$$
Y(t, \tau) = -\frac{\ln P(t, \tau)}{\tau} = \frac{B(\tau)}{\tau} r_t - \left( \frac{2\theta_r \mu_{r,t}}{\sigma_r^2} \right) \frac{\ln A_{\text{base}}(\tau)}{\tau}
$$

Where the components \(h\), \(A_{\text{base}}(\tau)\), and \(B(\tau)\) are derived as:

$$
h = \sqrt{\theta_r^2 + 2\sigma_r^2}
$$

$$
A_{\text{base}}(\tau) = \frac{2 h \exp\left( \frac{(\theta_r + h)\tau}{2} \right)}{(\theta_r + h)(\exp(h\tau) - 1) + 2h}
$$

$$
B(\tau) = \frac{2(\exp(h\tau) - 1)}{(\theta_r + h)(\exp(h\tau) - 1) + 2h}
$$

Here, the time-varying short-rate target \(\mu_{r,t}\) reflects path-dependent monetary policy shifts, yielding a dynamic nominal yield curve term structure.

### Real Yield Curve (Stochastic Fisher Relation)
Real yield curves are constructed by adjusting the analytical nominal term structure for expected inflation and an inflation risk premium over the holding period:

$$
Y_{\text{real}}(t, \tau) = Y(t, \tau) - \mathbb{E}_t[\bar{\pi}_{t, t+\tau}] - \operatorname{IRP}(\tau)
$$

The calculation of the real yield curve relies on the following two core term-structure components:

#### Expected Average Inflation
Expected average inflation over the tenor horizon \(\tau\) is derived from the mean-reverting properties of the underlying inflation process:

$$
\mathbb{E}_t[\bar{\pi}_{t, t+\tau}] = \mu_{\pi, t} + (\pi_t - \mu_{\pi, t}) \Lambda(\tau)
$$

Where \(\mu_{\pi, t} = y_{\text{target}, t} + \pi_{\text{min}}\) is the local central bank inflation target and \(\pi_t\) is the current spot inflation rate. The average reversion coefficient \(\Lambda(\tau)\) over the interval \(\tau\) is defined as:

$$
\Lambda(\tau) = \begin{cases} 
  \frac{1 - \exp(-\theta_{\pi} \tau)}{\theta_{\pi} \tau} & \text{if } \theta_{\pi} \tau > 10^{-4} \\ 
  1 - \frac{1}{2}\theta_{\pi} \tau + \frac{1}{6}(\theta_{\pi} \tau)^2 & \text{if } \theta_{\pi} \tau \le 10^{-4} 
\end{cases}
$$

*(Note: The second branch uses a Taylor series expansion to maintain numerical stability and avoid division-by-zero errors when handling very short tenors.)*

#### Inflation Risk Premium
The inflation risk premium compensates investors for the term-structure uncertainty of inflation, parameterized using a decay rate:

$$
\operatorname{IRP}(\tau) = \lambda_{\text{IRP}} \sigma_{\pi} \left( 1 - \exp(-\kappa_{\text{IRP}} \tau) \right)
$$

Where:

*   \(\lambda_{\text{IRP}}\) is the constant market price of inflation risk.
*   \(\kappa_{\text{IRP}}\) is the risk premium speed of adjustment parameter.
*   \(\sigma_{\pi}\) is the continuous inflation volatility parameter.
