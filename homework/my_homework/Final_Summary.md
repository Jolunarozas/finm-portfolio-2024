# Homework 1 / Mean-Variance optimization
## Notes
* Risk-Less Portfolio:
    - With 2 assets: if $\rho = -1$, with a  ${w_s}/{w_b} = {\sigma_b}/{\sigma_s}$ you can achive $\sigma_p^{2} = 0$
    - With n assets: $\rho = 0$ due to with n-large $\sigma_p^{2} \rightarrow \rho\sigma^{2}$

* Systematic and Idiosyncratic risk:
    - **Systematic** : Risk that you can't diversify ($\rho$)
        for a portfolio with n assets $\sigma_p^{2} \rightarrow \rho\sigma^{2}$
    - **Idiosyncratic**: Can be eliminated through diversification ($\sigma^{2}$)

* *Tangency Porfolio*: Allocates more weight to assets that:
    - Have higher mean returns
    - Have lower volatility (variance)
    - Have lower covariance with other assets 

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


* When you don't have the risk-free available, the min-variance portfolio and tangency portfolio are crucial, depending of the target return is your allocation in this two portfolios. **Weightes sum 1**.

* Now, if you have a risk-free asset available, all the mean-variance investors should hold a combination of risk-free and the tangency portfolio (Capital Market Line - CML), the slope of this Line is the max Sharpe Ratio that you can achive. If you have a target return greater than the tangency portfolio return, you should take a leverage on the risk free (more than 100% on tancency portfolio). **Total weight can be different than 1**.


## Harvard Case

* The policy of the portfolio would only change if: 
    - Changes in the goals or risk tolerance of the University
    - Change in capital market assumptions (1)
    - The emergence of a new asset class in the market (2)

* TIPS (inflation-linked bonds) showed a higher real yields than regular US Treasuries. So the argument was (1) and (2) for reduce dramatically the allocation on US Equity and US Treasurym investing on TIPS as a new asset class

* The team believed a 4% yields was reasonable for TIPS' expected real returns

* Harvard splitted their portfolio into layers, which is not optimal if there is correlation between asset classes. Within a group of securities, return characteristics must be highly homogeneous and asset classes must be sufficiently distinct.

* Only based on the Sharpe Ratio of TIPS is insufficient information to decide whether TIPS should be classified as a separate asset class. It's importan the correlation and characteristics of the asset.
    - Removing TIPS didn't show a significat impact on the optimization 
    - However, changing the expected returns can increase their portfolio weight significantly -> Due to the covariance sensitivity


# Homework 2 & 3

## Hedging and Tracking

* Net Exposure - Basis: It is the error of hedging an asset with a combination of others ($\epsilon$). You want to do this because:

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

* Performance evaluation: always include the intercept

* Hedging vs Traking

    - Hedging portfolio:
        - Large $\alpha$ (especially out-of-sample)
        - Low $\sigma_\epsilon$ (Basis)
    - Tracking portfolio:
        - $\alpha = 0$, to ensure good mimicry
        - Low $\sigma_\epsilon$ (tracking error),for high $R^{2}$
    - Information Ratio (IR) [${\alpha}/{\sigma_\epsilon}$]: Tradeoff between obtaining extra return $\alpha$ at the cost of $\epsilon$
    - Gross Notional of the hedge = results.params.abs().sum() (including intercept if that is the model)

## Value at Risk (VaR)

* **Historical VaR**: does not assume any distribution, it uses the empirical cumulative distribution function, so we can calculate the quantile directly from historical data
    - Benefits:
        - No assumptions about the distribution of the returns
        - Ease implementation
    - Drawbacks:
        - Requires a lot of data to get a good estimate for extreme events (VaR q = 0.1%)
        - Assume iid returns
* **Parametric VaR**: Assume normal distribution due to the simplicity in calculation. $$r^{VaR_{q,t}} = \mu_t +z_q\sigma_t$$
    - Benefits:
        - Statistical power: you can get a better estimation with less data available 
        - Easy calculation
    - Drawbacks:
        - If you assume normal distribution, you're missing how to model propertly the tails of the returns (skewness and kurtosis), particulary for small samples [**Bias**]
        - Assume iid returns
* Generally, parametric VaR outperforms empirical VaR

* Hit Ratio: Measures the precentage of times the next period's returns is worse than the VaR estimate

$$Hit Ratio = \frac{count(r_t < VaR(q)_t)}{n} $$
* Hit Ratio Error:
$$Hit Ratio = |\frac{Hit Ratio}{q} - 1| $$
* Market Beta: $\beta$
* Treynor Ratio (measures the level of returns per unit of market risk): $(E[r]-r_f)/ \beta$
* Information ratio: $\alpha / \sigma_\epsilon$ 

## Study Case Notes - ProShares Hedge Replication ETF

- Two types of Alternative ETFs:
    - Alternative Asset Class: Such as illiquid investments (Real Estate or Private Equity). Alternative assets typically have different regulations and feestructures
    - Alternative Strategy: Unconventional approache to generating returns: Shorting or leverage, event bases investing 
- ProShares:
    - Allowed investors to take hedging and speculative position without directly using derivatives
    - When the volatility is high, the price of the index and ETF can diverge due to ETFs mostly has a daily reset
    - Hedge fund replication ETF (HDG == ETF ; HRFI Hedge Fund Research index == INDEX ; Merill Lynch Factor Model - INDEX ;  Merill Lynch Factor Model Exchange Series (adjusted assets) - ProShare tried to replicated) offered:
        - Exposure to Hedge Funds (diversification)
        - Transparency
        - Daily liquidity
        - Low fees and taxation
    - Results: 
        - HDG aiming to provide access to $\beta$ rather than generate $\alpha$
        - Presented multifactor regression with high multicollinearity
        - Model achived a high R-Squared, but difference on level, skewness and kurtosis
        - HFRI had better SR, but HDG exhibited less negative skweness and lower excess kurtosis


# Homework 4 / CAPM
## Notes DFA Case

- Explote small/micro-cap equity opportunities  and in value stocks (high book to market ratio) -> Based on academic research and lack of these products
- These excess returns in the market are earned by bearing risk
- Based on the assumption of efficiency of markets (all information all ready incorporated to the price) -> They don't rely on individual equity analysis or macroeconomic strategies
- DFA's funds follow a index-like strategies but with a component of active management 
- On the 80s, DFA faced a challenge due to deep recession that particularly affected to small companies -> After this episode, the premium of "size factor" was unattractive and the growth stocks outperformed value


## Notes Theory 

### CAPM : 

$$ E[r^{i}] = \beta^{i,m}E[r^{m}] + \epsilon^{i}  \space (1)$$
$$ \beta^{i,m} = \frac{Cov(r^{i},r^{m})}{Var(r^{m})} \space (2)$$
$$ \frac{ E[r^{i}] }{\sigma^{i}} = \rho^{i,m}\frac{ E[r^{m}] }{\sigma^{m}} \implies (SR)^{i} = \rho^{i,m}(SR)^{m}  \space (3)$$
$$ Teynor Ratio = \frac{ E[r^{i}] }{\beta^{i,m}} = E[r^m] \space \forall i \space (4)$$



- (1) This implies that market beta is the only risk associated with higher returns

- (3) The $\max(SR^i) = 1(SR)^m$

- (4) If the CAPM were true, the Teynor Ratios would be the same for all the assets and constant equal to the expected market premium

- Alphas would also be equal to zero (mean-absolute-error := $MAE = \frac{1}{n} \sum_{i=1}^{n} |\alpha_i| \space != 0  $), then the Information Ratio should be zero too

- On **time-series regressions**, the $R^2$ only tells whether the market is a good hedge for the asset, but **it does not address the validity of CAPM** (or any other factor model) -> It's irrelevant! The important part in the time-series is $\alpha$

- If CAPM were true, with a **cross-sectional validation**: $$\mu_i = \eta + \lambda_m\beta^{i,m} + \epsilon$$
    - **$R^2$ should be high**: dependent variable (portfolio mean returns) should be explained entirely by the level of market risk ($\beta^{i,m}$ estimated in the time-series regressions) 
    - $\lambda_m$ should be equal the **market premium**
    - The intercept ($\eta$) **should be zero**, reflecting that one should not get any excess mean retruns if they take no market risk

### Fama-French 3 factor model

$$ E[r^{i}] = \beta^{i,m}E[r^{m}] + \beta^{i,s}E[SMB] + \beta^{i,v}E[HML] + \epsilon^{i} $$

- Size (SMB) : Small minus big market capitalization
- Value (HML) : High (value) minus low (grown) book-to-market ratio.    
    - **Value** -> Low price relative to some firm fundamentals. 
    - **Growth** -> High valuation with respect to these measures

### How to test a time-varying betas

This comes from Fama-MacBeth at the end of Lecture 4. 

1. Estimate a the time-series regression using a rolling window or other methods, we want an estimate of $\beta_{t}^i$, that is, the beta for asset $i$ at time $t$.
2. Run a cross-sectional regression for each time, $t$. That is, calculate the cross-section regression using the $\beta_{t}$, for each time $t$. We do this to calculate $\lambda$ and $v$.
3. We can then take the sample means of each of the estimates to get the final estimates of $\lambda$ and $v$.

Let's assume we are using t-window rolling regression. For TS regression we have N * (T - t+1) regressions where N is the number of assets and T is the total number of time periods. For CS regression we have (T-t+1) regression for each of the time periods. Thus, the total number of regressions is (N + 1) * (T - t+1)

# Homework 5 / Multi-Factor models


## Notes Case (Smart Beta)

- By taking lon-short position, FF tried to reduce the correlation between the factors

- Smart-beta EFT's: Weighting schemes were based on firms' financial characteristics or properties. Combination of passive and active investing, they track a factor index and will thus be long-short (careful, ETF for regulation can't take a significant portion in short position, therefor these ETF are "tiled" to the factor charasteristics)

- Value (`HML`) is realtively redundant for the FF5, the correlation with Investment Factor (`CMA`) is almost 70% and the correlation with Quality Factor (`RMW`) is 22%

## Theory Notes

$$ CAPM := E[r^{i}] = \beta^{i,m}E[r^{m}] + \epsilon^{i}  \space $$ 
$$ FF-3F := E[r^{i}] = \beta^{i,m}E[r^{m}] + \beta^{i,s}E[SMB] + \beta^{i,v}E[HML] + \epsilon^{i} $$
$$ FF-5F := E[r^{i}] = \beta^{i,m}E[r^{m}] + \beta^{i,s}E[SMB] + \beta^{i,v}E[HML] + \beta^{i,quality}E[RMW] + \beta^{i,inv}E[CMA] + \epsilon^{i} $$
$$ AQR := E[r^{i}] = \beta^{i,m}E[r^{m}] + \beta^{i,v}E[HML] + \beta^{i,quality}E[RMW] + \beta^{i,mom}E[UMD] + \epsilon^{i} $$

- The factors are constructed using:

    - **Size Factor** (Small minus Big -- SMB): Small cap minus big cap. Split stocks into 5 quantiles, long smallest quantile and short largest quantile (by market cap).

    - **Value Factor** (High minus Low -- HML): High book-to-market ratio minus low book-to-market ratio. Split stocks into 5 quantiles, long highest quantile and short lowest quantile (by book-to-market ratio).

    - **Quality Factor** (Robust minus Weak -- RMW): Robust operating profitability minus weak operating profitability (operating profitability = revenius - costs). Split stocks into 5 quantiles, long highest quantile and short lowest quantile (by operating profitability).

    - **Investment Factor** (Conservative minus Aggressive -- CMA): Conservative investment (low levels of investment) minus aggressive investment. Split stocks into 5 quantiles, long lowest quantile and short lowest highest (by investment).

    - **Momentum** (Up minus Down -- UMD): Up minus down. Split stocks into 5 quantiles, long highest quantile and short lowest quantile (by returns in previous year).

- Recall from class/homework that `UMD` (Momentum) is negatively correlated to all the other factors

- Depending on how we construct the factor, their charasteristics can change: For instance, on HW6 whe we only used 1 decile (top/bottom of assets returns), we get more return but also more vol, ie less diversification (less assets) and higher idiosincratic risk

- The tangency portfolio helps us identify the most important factors. But LASSO for instance, can't be use for this propurse, because LASSO in time-series maximmizes the $R^2$ and our goal is to minimize $\alpha$

- In the **time-series** analysis of a **Pricing Factor Model**, it is not a problem to have a **R-Squared = 0**, it means that the beta is zero, therefor the expected return of the asset is the risk-free rate. But if the **alpha is different than zero is a problem**, it means that the asset is on averege getting compensated more/less than the risk-free due to risk non-associated with the factors

- In the **Linear Factor Descomposition (LFD)** it is a **problem to have a R-Squared close to zero**, it means that your regressors (factors) are no able to capture variability of the target asset. Therefore, **you can't excecute a correct hedge** and if you're **tracking** instrument is expected to have a **very poor performance**. In LFD is not a problem to have an alpha different to zero



- The Time-Series MAE (= $\alpha$) larger for factor models with more factor in constrast to Cross-Sectional (focus on $R^2$ and MAE = avg absolute error). 


![alt text](image.png)


- **Arbitrage Pricing Theory (APT)**: links Linear Factor Decomposition with linear factor pricing models. If factors work well for decomposition, they should work well for Linear Factor Model. 
    - For a LFD we have some excess-returns factors which works well 

    - Assumptions: 
        1. The **residuals** are uncorrelated across regressions
        2. There is not arbitrage ($\alpha = 0$)

    - The idiosincratic risk of the asset depends only on the residuals variance ($var(\epsilon_t) = \frac{1}{n}\sigma^2_e$), so if the number of asset goes to infinitive, the assets have a perfect factor structure with no idiosincratic risk -> The assets returns can be perfectly replicated by the factors
    - Perfect Linear Factor Decomposition $\implies$ Perfect Linear Factor Model (but not the opposite)
# Homework 6

## Case Notes - AQR Momentum

- AQR Momentum product is a mutual fund, allowing to retail investor to take this type of exposition. it has a Long-only legal limitation, retricting the use of short positions or leverage and they have to be ready to return capital at the end of the trading day (Open-end)

- This mutual fund wil not track exactly the momentum index due to: (1) the Long-Only construction,  of FF (Long-Short position) -> high correlation with the market (90%); (2)  the index assume a monthly rebalance (high transaction cost) while AQR used a quarterly rebalance; (3) FF Factor used all listed stocks and AQR has to use liquid stocks 

- They construct the momentum portfolio using returns from t-12 to t-2, due to the winning/losing stocks between t-1 and t may show short-term reversal

## Case Notes - Barnstable approach

- Assumptions:  log iid returns, normally distributed
- The average period return r of cumulative returns can be represented as: 

$$ r \sim N(\mu,\frac{\sigma^2}{h}) $$
- The probability of the asset underperforming the risk-free:

$$ P(r_f > r) = \phi(-\sqrt{h}\frac{\mu-r_f}{\sigma}) $$
- Implications:
    - Sharpe ratio scales with time, meaning that in the long-run we would expect to have a higher Sharpe, and thus it would be a safer investment. Additionally, we'll have lower standard error of average returns.

    - Riskier because volatility also scales with time! That means our cumulative returns will have higher volatility (despite a higher Sharpe) and thus they could be considered riskier in the long run. We also have parameter uncertainty, meaning that we are less certain about the parameters in the long run, so we have "model risk" in the long run.

- They took advantange of this by: 

    1. Selling puts on the S&P 500 index, which were at a premium (due to vol skew).
    2. Selling shares on a pool of S&P 500 stocks, where (1) class of shares would return the minimum of the S&P return and 6%, and (2) would return whatever was left over from (1). They can then sell (1) and keep (2).

Application Example: Probability that MKT will outperform HML in the following 5y:

First, we assume log-returns of MKT and HML are iid multivariate normal.

We want

$$P\left(\Pi_{t=1}^{60} (1+r_t^{MKT}) > \Pi_{t=1}^{60} (1+r_t^{HML})\right)$$

Denote log-return $\mathrm{r} = \log(1+r)$ and let $\mathrm{r}^P = \mathrm{r}^{MKT} - \mathrm{r}^{HML}$.



$$
\begin{aligned}
P\left(\Pi_{t=1}^{60} (1+r_t^{MKT}) > \Pi_{t=1}^{60} (1+r_t^{HML})\right) 
&= P\left(\sum_{t=1}^{60} \mathrm{r}_t^{MKT} > \sum_{t=1}^{60} \mathrm{r}_t^{HML}\right) \\
&= P\left(\sum_{t=1}^{60} (\mathrm{r}_t^{MKT} - \mathrm{r}_t^{HML}) > 0 \right) \\
&= P\left(\sum_{t=1}^{60} \mathrm{r}_t^{P} > 0 \right) \\
&= P\left(\bar{\mathrm{r}}_t^{P} > 0 \right) \\
&= 1- \Phi\left(\sqrt{60}\frac{-\mu}{\sigma}\right) = \Phi\left(\sqrt{60}\frac{\mu}{\sigma}\right)
\end{aligned}
$$

The last equation comes from

$$\bar{\mathrm{r}}^P \sim \mathcal{N}\left(\mu, \frac{1}{60}\sigma^2\right)$$

where

$$\mu = \mu_{MKT} - \mu_{HML}$$ 
$$\sigma^2 = \sigma_{MKT}^2 + \sigma_{HML}^2 - 2\sigma_{MKT, HML}$$

or can directly calculate $\mu$ and $\sigma$ from the spread between the two.

In case of using annualized $\mu$ and $\sigma$, the distribution should be the average of 5 years of annualized returns
$$\bar{\mathrm{r}}_{annual}^P \sim \mathcal{N}\left(12\mu, \frac{12}{5}\sigma^2\right)$$


$$
\Phi\left(\frac{12\mu}{\sqrt{12/5}\sigma}\right) = \Phi\left(\sqrt{60}\frac{\mu}{\sigma}\right)
$$

Which is equivalent to the above.

## Notes - Time diversification

We assume iid logarithms returns in this context

### Mean

- Cumulative return is the sum of returns 
- Expected return for a given period h is : $E[r_{t,t+h}] = h\mu$

### Variance - Autoregresive model AR(1)

$$ cov[r_t,r_{t+i}] = \rho^i\sigma^2 \implies corr[r_t,r_{t+i}] = \rho^i $$

- If $\rho^i = 1$, the standard desviation grows linearly with time: $h\sigma \implies SR(r_{t,t+h}) = SR(r_t)$
- If $\rho^i = 0$, the variance grows linearly with time: $h\sigma^2 \implies SR(r_{t,t+h}) = \sqrt{h}SR(r_t)$ 
- If $\rho^i = -1$, the return becomes riskless and the variance is 0
- If $|\rho^i| < 1$, the **Sharpe Ratio increases over longer horizons**, indicating risk reduction over time for a unit of excess return: $SR(r_{t,t+h}) > SR(r_t)$


# Homework 7 

## Case - GMO

- They based their desicions on macroeconomic signals, the market tends to overreact on the short-term, but in the long-term the value will align with the fundamentals

- They estimated the equity risk premium in the long term and the expected return on sticks over the intermediate term. Equities deserved risk premium as they "lose money at bad times"

- GMO was interested in the expected return in the *intermediate* term because market prices can significantly differ from fundamental value, which leads to the intermediate term return possibly being different from the long-run required return - 7 years forecast

- Surviving as a contratian: The market can remain irrational longer than you can remain liquid -> Lose too much and/or your clients take away their invests. GMO took this position with tech-firms on 90's -> losing moning in the rally, but avoiding the tech-bubble after

- GMO belive that the fundamentals have changed on the last years, so the past 40-50 years can't provide an accurate representation of futures returns

- They used: `dividend-to-price ratio (DP)`, `earnings-to-price (EP)` multiple expansion/contraction, the change in the `profit margin`, and `sales growth` as its predicting variables. Specifically, it believes that the `profit margin` and the `P/E multiple` have **long-term steady values**, and that in the **long-term returns are driven** by `sales growth` and the *required* `dividend yield`.

- Their fund, `GMWAX` performed worse than the SPY: (1) High correlation ; (2) Lower Sharpe Ratio (in the case period and the years after) ; (3) Worse tail risk (kurtosis and skewness, VaR and CVaR after adjusting for vol were much worse)


## Class Notes - Forcasting 

- Two views: 
    - **Classical view** risk premium doesn't change over time: $E_t[r_{t+1}] = E[r]$ similar to think $log(\frac{P_{t+1}}{P_t}) = constant + \epsilon_{t+1} $ (random walk with a drift)
    - **Risk premium changes over time**, there is a function that explains the expected return: $E_t[r_{t+1}] = f(X_t)$
        - Example: Linear model, $r_{t+1} = \alpha + \beta x_t + \epsilon_{t+1}$ The **$R^2$ is crucial** in this type of time series forcast, even if $\beta$ is significant, if $R^2$ is low even when the effect exists, it is practically insignificant 
- Dividend-Yield Forecasting
    - Dividend-Yield = dividend-price ratio ($\frac{D_t}{P_t}$)
    $$ R_{t+1} = \frac{D_t}{P_t}\frac{D_{t+1}}{D_t} + \frac{P_{t+1}}{P_t} $$
    $$ E_t[R_{t+k}] = DP_tE_t[\frac{D_{t+k}}{D_t}] + E_t[\frac{P_{t+k}}{P_t}] $$
    - **Classical View**: increase in DP implies a dicrease of the dividend grows -> Expected return constant over time
    - **Empirical results**: increase in DP implies an increase on the expected returns in the long-term horizon. This is holds because DP has a high autoregressive coefficiient at monthly frequencies -> Compouding this effect leads to higher returns for assets with higher DP
        - Similar effects with Earnings/Price (EP) ratio


# Homework 8 


## Case Notes - Long-Term Capital Management

- Tried to trade mispricing instruments (arbitrage): (1) Relative values ; (2) Covergence trades
- Use leverage and Trades focus mainly on Fixed income and credit, but also a portion on Equity 
- High negative skeness -> target of 3% return, but in extreme scenarios have an unlimited posible losses
- Advantages: Efficient financing ; Fund Size ; Collaterallization (pay lower haircuts -> higher leverage) ; Long-term horizon ; Liquidity and Hedging (against default risk) 
- Manage of risks:
    - Collateral haircuts: almost 100% financing and stress test for the haircuts
    - Repo maturity: Manage their aggregate repo maturity (usually they used long repos - 6 to 12 months)
    - Equity redemption: could lead to unwind their positions at unfavorable rates. They restrict the redemptions
    - Loan access: Access to loan without any relation to their fund performance 
    - Liquidity: Asked for daily basis collateral, Haircuts worst case scenario and adjusting security correlations
- Performance: 
    - Similar return and vol than SPY, but a little higher SR, also a low beta. In this sence is attractive, but the Skewness was more negative and Kurtosis was higher than SPY, making more risk in extrem scenarios


## Class Notes - FX Carry Trade

- Carry: investing in assets with higher cash flow
- Return on investing on a Foreign Currency = $\frac{S_{t+1}}{S_t}R^{foreing}_{t+1}$ 
- $S_t$ represetns the foreign exchange rate, expressed as USD per foreign currency -> $S_t$ raise -> weak dollar
- **Covered Interest Parity (CIP)**: Difference between the **forward and spot** rate is determined by the difference in risk-free rates
$$ f_t - s_t = r^{dollars}_{t,t+1} - r^{f}_{t,t+1} $$
    - This is a **non-arbtrage relationship** (law of one price), not a model. Empirical data supports this relationship, except during periods of extreme market stress
    - **Not Arbitrage Condition**
- **Uncovered Interest Parity (UIP)**: The **expected growth** in the foreign exchange rate is equal to the difference in risk-free rates
$$ ln(E[S_{t+1}]) - s_t = r^{dollars}_{t,t+1} - r^{f}_{t,t+1} $$
    - Logic: If the rate is higher in one country, money should flow there, causing its currency to appreciate
    - Based under expectation not realized values
    - If UIP holds -> Forward rate would be the best predictor of future prices 
    - **Model**
- Testing - Should result on $\beta = 1$, $\alpha = 0$ and $R^2$ depends on the model, for **CIP we expected a high $R^2$** while **UIP focus on the average relationship**: 
    - Interest rate differencial: 
        $$ s_{t+1} - s_t = \alpha +\beta(r^{dollar}_{t,t+1} - r^{f}_{t,t+1}) + \epsilon_{t+1}$$
    - Forward premium
        $$ s_{t+1} - s_t = \alpha +\beta(f^s_t - s_t) + \epsilon_{t+1}$$
    - If: 
        * $\beta < 1$ and high $R^2$, worthwhile to favor the higher interest rate side on average
        * $\beta < 0$, could get benefits from both, high interest rate or FX changes
        * $\beta < 1$, the FX change more than offsets the interest rate differential
- Carry Trades FX: 
    - Taking positions in currencies based on their carry (interest rate differential)
    - Profits from the fact that **UIP generally does not hold**
    - has a high negative skewness -> When it fails, the losses are big
- Carry Trade: Going long assets with higher yield/dividends and shorting those with lower yields

