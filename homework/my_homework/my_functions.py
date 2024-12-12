
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import skew,kurtosis,norm
import statsmodels.api as sm

def get_metric_returns(returns, weights=[], adj_factor=12, VaR_q=5):


    # If weights are provided, compute the weighted returns
    if len(weights) == 0:
        port_metrics = returns.copy()
    else:
        port_metrics = returns @ weights
        port_metrics = pd.DataFrame(port_metrics, columns=['Portfolio'])

    # Initialize the result DataFrame
    result = pd.DataFrame()

    # Compute Mean, Volatility, Sharpe Ratio, Skew, Excess Kurtosis
    if len(weights) == 0:
        port_metrics_r = pd.DataFrame({
            "Mean": port_metrics.mean() * adj_factor,
            "Volatility": port_metrics.std() * np.sqrt(adj_factor)
        })
        port_metrics_r["Sharpe_Ratio"] = (port_metrics.mean() / port_metrics.std()) * np.sqrt(adj_factor)
        port_metrics_r["Skew"] = port_metrics.apply(skew)
        port_metrics_r["Excess Kurtosis"] = port_metrics.apply(kurtosis, fisher=True, bias=False)
    else:
        asset = 'Portfolio'
        port_metrics_r = pd.DataFrame({
            "Mean": [port_metrics[asset].mean() * adj_factor],
            "Volatility": [port_metrics[asset].std() * np.sqrt(adj_factor)]
        }, index=[asset])
        port_metrics_r["Sharpe_Ratio"] = (port_metrics[asset].mean() / port_metrics[asset].std()) * np.sqrt(adj_factor)
        port_metrics_r["Skew"] = skew(port_metrics[asset])
        port_metrics_r["Excess Kurtosis"] = kurtosis(port_metrics[asset], fisher=True, bias=False)

    # Compute VaR, CVaR, Max Drawdown, and related metrics
    for asset in port_metrics.columns:
        data_aux = port_metrics[[asset]].copy()
        VaR = np.percentile(sorted(data_aux.values.flatten()), q=VaR_q)
        CVaR = data_aux[data_aux[asset] <= VaR].mean().values[0]

        data_aux_acum_return = (data_aux + 1).cumprod()
        data_aux_max_cum_return = data_aux_acum_return.cummax()
        data_aux_drawdown = ((data_aux_acum_return - data_aux_max_cum_return) / data_aux_max_cum_return)
        max_drawdown = data_aux_drawdown.min().values[0]
        max_drawdown_date = data_aux_drawdown.idxmin().values[0]
        peak_idx = data_aux_max_cum_return.idxmax().values[0]
        recovery_data = data_aux_drawdown[data_aux_drawdown.index >= max_drawdown_date]
        recovery_idx = recovery_data[recovery_data[asset] >= -0.00001].first_valid_index()
        duration = (recovery_idx - max_drawdown_date).days if recovery_idx else np.nan

        aux_result = pd.DataFrame(
            [[VaR, CVaR, max_drawdown, max_drawdown_date, peak_idx, recovery_idx, duration]],
            columns=["VaR", "CVaR", "Max Drawdown", "Bottom", "Peak", "Recovery", "Duration (days)"],
            index=[asset]
        )
        result = pd.concat([result, aux_result], axis=0)

    # Merge the two sets of metrics
    metrics = pd.merge(port_metrics_r, result, left_index=True, right_index=True, how="left")
    return metrics

def weights_tang(return_db, adj_factor = 12):
    sigma = (return_db.cov()*adj_factor)
    mu_excess = (return_db.mean()*adj_factor)
    vector = np.ones(len(mu_excess))
    w_tan = (np.linalg.inv(sigma) @ mu_excess )/(np.transpose(vector) @ np.linalg.inv(sigma) @ mu_excess)
    weights_db = pd.DataFrame({"w_tan": w_tan})
    weights_db.index = return_db.columns
    return weights_db

def weights_tag_reg(return_db, adj_factor = 12, diagonal_factor = 2, denominator = 3):
    sigma = (return_db.cov()*adj_factor)
    sigma_reg = (sigma + diagonal_factor*np.diag(np.diag(sigma)))/denominator
    mu_excess = (return_db.mean()*adj_factor)
    vector = np.ones(len(mu_excess))
    w_tan = (np.linalg.inv(sigma_reg) @ mu_excess )/(np.transpose(vector) @ np.linalg.inv(sigma_reg) @ mu_excess)
    weights_db = pd.DataFrame({"w_tan_reg": w_tan})
    weights_db.index = return_db.columns
    return weights_db

def weights_equally_weighted(return_db):
    n = len(return_db.columns)
    weights_db = pd.DataFrame({"w_equally_weighted": [1/n]*n})
    weights_db.index = return_db.columns
    return weights_db

def weights_GMV(return_db, adj_factor = 12):
    sigma = (return_db.cov()*adj_factor)
    mu_excess = np.ones(len(return_db.columns))
    vector = np.ones(len(return_db.columns))
    w_tan = (np.linalg.inv(sigma) @ mu_excess )/(np.transpose(vector) @ np.linalg.inv(sigma) @ mu_excess)
    weights_db = pd.DataFrame({"w_GMV": w_tan})
    weights_db.index = return_db.columns
    return weights_db

def mv_portfolio_excess_returns(return_db, mu_target, adj_factor = 12):
    w_tan = weights_tang(return_db, adj_factor)
    w_tangency_scale = ((mu_target/(return_db.mean() @ w_tan)) * w_tan)
    return w_tangency_scale

def mv_portfolio_total_returns(return_db, mu_target, adj_factor = 12):
    w_tan = weights_tang(return_db, adj_factor)
    w_GMV = weights_GMV(return_db, adj_factor)
    rho = (mu_target - (return_db.mean() @ w_GMV).values[0])/((return_db.mean() @ w_tan).values[0] -( return_db.mean() @ w_GMV).values[0])
    w_MV_target_no_risk = (rho*w_tan.values + (1-rho)*w_GMV)
    w_MV_target_no_risk.columns = ['w_MV']
    return w_MV_target_no_risk

def benchmark_regresion(data,benchmark = "SPY US Equity",adj = 12):
    result = pd.DataFrame()
    for asset in data.drop([benchmark],axis=1).columns:
        X = sm.add_constant(data[benchmark])
        y = data[asset]
        mod = sm.OLS(y, X).fit()
        inter, beta = mod.params.values[0], mod.params.values[1]
        rsquare = mod.rsquared
        std_errors= mod.resid.std()
        mae = np.mean(np.abs(mod.predict(X) - y))
        TR = (y.mean()/beta)*adj
        IR = (inter/std_errors)*np.sqrt(adj)
        Sortino = y.mean()/data[data[asset]<0][asset].std() * np.sqrt(adj)
        aux_result = pd.DataFrame([[inter*adj,beta,rsquare,std_errors,y.mean()*adj,TR,IR,Sortino,mae]],columns=["Alpha-Adj","Beta","R-square","std_errors","Asset_mean","Treynor Ratio","Information Ratio","Sortino Ratio","MAE"], index = [asset])
        result = pd.concat([result,aux_result],axis=0)
    return result

def time_series_regression(data, y_asset, x_asset, adj=12, constant=True):
    Y = data[y_asset]
    X = data[x_asset]

    if constant:
        X = sm.add_constant(X)

    mod = sm.OLS(Y, X).fit()

    if constant:
        inter = mod.params.values[0]
        betas = mod.params[1:].values
    else:
        inter = np.nan
        betas = mod.params.values

    rsquare = mod.rsquared
    std_errors = mod.resid.std()
    tracking_error = mod.resid.std() * np.sqrt(adj)

    # Calculate Treynor Ratio, Information Ratio, and Tracking Error
    treynor_ratio = (adj * Y.mean()) / mod.params[0] if not constant else (adj * Y.mean()) / mod.params[1]
    info_ratio = (
        np.sqrt(adj) * inter / std_errors if std_errors != 0 and constant else np.nan
    )

    # Calculate Sortino Ratio
    downside_std = data[data[y_asset] < 0][y_asset].std()  # Downside standard deviation
    sortino_ratio = (Y.mean() / downside_std) * np.sqrt(adj) if downside_std > 0 else np.nan

    # Construct the metrics dataframe conditionally
    if constant:
        metrics_columns = [
            "Alpha", "Alpha Adj",
            *([f"{col}" for col in x_asset]),
            "R-square", "std_errors", "Tracking_error",
            "Treynor Ratio", "Information Ratio", "Sortino Ratio"
        ]
        metrics_values = [
            inter, inter * adj,
            *betas,
            rsquare, std_errors, tracking_error,
            treynor_ratio, info_ratio, sortino_ratio
        ]
    else:
        metrics_columns = [
            *([f"{col}" for col in x_asset]),
            "R-square", "std_errors", "Tracking_error",
            "Treynor Ratio", "Sortino Ratio"
        ]
        metrics_values = [
            *betas,
            rsquare, std_errors, tracking_error,
            treynor_ratio, sortino_ratio
        ]

    metrics = pd.DataFrame([metrics_values], columns=metrics_columns, index=[y_asset])
    return metrics



# VaR Functions

def CVaR_parametric(data, alpha = 0.05):
    return data.std()*(-norm.pdf(1.65)/alpha)

def Rolling_window_VaR_CVaR(data, asset, alpha = 0.05, m = 52):
    data = data[[asset]]
    data["Rolling_Vol"] = np.sqrt((data.shift()[asset]**2).rolling(m).mean())
    data["Rolling_VaR"] = data["Rolling_Vol"]*(-1.65)#round(norm.ppf(alpha),2)
    data["Rolling_CVaR"] = data["Rolling_Vol"]*(-norm.pdf(round(norm.ppf(alpha)))/alpha)
    Hit_Ratio = data.loc[data[asset] < data["Rolling_VaR"],asset].count()/len(data["Rolling_VaR"].dropna())
    return data, Hit_Ratio

def Parametric_VaR(data, alpha = 0.05):
    return data.std()*(-norm.pdf(1.65)/alpha)

def Historical_VaR_Expanding(data, alpha = 0.05):
    return data.shift().expanding(min_periods = 60).quantile(alpha)

def Historical_VaR_Rolling(data, alpha = 0.05, m = 52):
    return data.shift().rolling(m).quantile(alpha)

def Historical_CVaR_Expanding(data, alpha = 0.05):
    return data.shift().expanding(min_periods = 60).apply(lambda x: x[x < x.quantile(alpha)].mean())

def Historical_CVaR_Rolling(data, alpha = 0.05, m = 52):
    return data.shift().rolling(m).apply(lambda x: x[x < x.quantile(alpha)].mean())

def Normal_VaR_Rolling(data, z_score = -2.33, m = 52):
    return np.sqrt((data.shift()**2).rolling(m).mean())*z_score

def Historical_VaR(data, alpha = 0.05):
    return data.shift().quantile(alpha)

def ewma(returns, theta = 0.94, sigma_zero = 0.2/np.sqrt(252)):
    ewma_var = [sigma_zero**2]
    for i in range(len(returns)):
        var = ewma_var[-1]*theta + (returns.iloc[i]**2)*(1-theta)
        ewma_var.append(var)
    ewma_var = np.sqrt(ewma_var[1:])
    return ewma_var