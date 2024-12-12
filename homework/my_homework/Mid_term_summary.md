# Homework 1 / Mean-Variance optimization
## Notes
* Risk-Less Portfolio:
    - With 2 assets: if $\rho = -1$, with a  ${w_s}/{w_b} = {\sigma_b}/{\sigma_s}$ you can achive $\sigma_p^{2} = 0$
    - With n assets: $\rho = 0$ due to with n-large $\sigma_p^{2} \rightarrow \rho\sigma^{2}$

* Systematic and Idiosyncratic risk:
    - Systematic : Risk that you can't diversify ($\rho$)
        for a portfolio with n assets $\sigma_p^{2} \rightarrow \rho\sigma^{2}$
    - Idiosyncratic risk: Can be eliminated through diversification ($\sigma^{2}$)

* *Tangency Porfolio*: Allocates more weight to assets that:
    - Have higher mean returns
    - Have lower volatility (variance)
    - Have lowe covariance with other assets 

* Invert Covariance matrix ($\Sigma^{-1}$) is highly unstable with large number of securities (determinant close to zero or nearly singular). High sensitivity to changes in input data $\rightarrow$ In-sample vs out-sample results (weights). In order to "correct" this, you can use a regularization method (such as add the diagonal and divided the result by two), this shrinks the off-diagonal elements (covariance)

* *Two-stage optimization approach*: In order to split the optimization problem into layers, within a group of security, returns characteristics should be similars and asset classes (diferent problems) must be sufficiently distinct $\rightarrow$ Eventhough, you could achive a local optimal result instead a global

* For a Mean-Variance optimization without a risk-free asset:
    $$
    r_p = r_t \omega + r_v (1 - \omega)
    $$

    $$
    r_p = r_t \omega + r_v - r_v \omega
    $$

    $$
    r_p - r_v = (r_t - r_v) \omega
    $$

    $$
    \omega = \frac{r_p - r_v}{(r_t - r_v)}
    $$
# Homework 2 & 3
## Hedging and Tracking

* Net Exposure - Basis: It is the error of hedgeing/tracking an asset with a combination of others ($\epsilon$). You want to do this because:
    - Non-tradable exposure
    - Liquid security but costly to short
* Only with one asset:
    $$\epsilon_t = r_t^{i} -hr_t^{j} $$
    Basis risk: $$\sigma_\epsilon^{2} = \sigma_i^{2} + h^{2}\sigma_j^{2} - 2h\sigma_i\sigma_j\rho_{i,j}$$
    Therefore:
    $$ h* = \arg\min_h{\sigma_\epsilon^{2}} $$
    $$ h* = {\sigma_i}/{\sigma_j}\rho_{i,j}$$
* For multiple risks:
    - Just the linear regression
* Include the interception for hedging ($\alpha$):
    - Including an intercept lets the $\beta$ focus on matching return variation and not just the level 
    - If $\alpha$ is excluded, $\beta$ also adjust the magnitude
    - *Keep Point*: If we expect the difference in mean returns to persist out-of-sample, we should not include $\alpha$
* Market-Hedged Position
    - When we hedge the market and invest in a particular asset, our return becomes $\alpha + \epsilon_t$. In order to eliminate the volatility from $\beta^{i,m}r_t^{m}$
    $$r_t^{i} - \beta^{i,m} r_t^{m} = \alpha + \epsilon_t$$
    - If we have a positive return ($\alpha$) and a low variance ($\sigma_\epsilon$), we get an statistical arbitrage
* Hedging vs Traking
    - Hedging portfolio:
        - Large $\alpha$ (especially out-of-sample)
        - Low $\sigma_\epsilon$ (Basis)
    - Tracking portfolio:
        - $\alpha = 0$, to ensure goog mimici
        - Low $\sigma_\epsilon$ (tracking error),for high $R^{2}$
    - Information Ratio (IR): ${\alpha}/{\sigma_\epsilon}$
* Study Case Notes:
    - Two types of Alternative ETFs:
        - Alternative Asset Class: Such as illiquid investments (Real Estate or Private Equity). Alternative assets typically have different regulations and feestructures
        - Alternative Strategy: Unconventional approache to generating returns: Shorting or leverage, event bases investing 

## Value at Risk (VaR)

* Historical VaR: does not assume any distribution, it uses the empirical cumulative distribution function, so we can calculate the quantile directly from historical data
    - Benefits:
        - No assumptions about the distribution of the returns
        - Ease implementation
    - Drawbacks:
        - Requires a lot of data to get a good estimate for extreme events (VaR q = 0.1%)
        - Assume iid returns
* Parametric VaR: Assume normal distribution due to the simplicity in calculation. $$r^{VaR_{q,t}} = \mu_t +z_q\sigma_t$$
    - Benefits:
        - Statistical power: you can get a better estimation with less data available 
        - Easy calculation
    - Drawbacks:
        - If you assume normal distribution, you're missing how to model propertly the tails of the returns (skewness and kurtosis) [Bias]
        - Assume iid returns
* Hit Ratio:
$$Hit Ratio = \frac{count(r_t < VaR(q)_t)}{n} $$
* Hit Ratio Error:
$$Hit Ratio = |\frac{Hit Ratio}{q} - 1| $$
* Market Beta: $\beta$
* Treynor Ratio (measures the level of returns per unit of market risk): $(E[r]-r_f)/ \beta$
* Information ratio: $\alpha / \sigma_\epsilon$ 