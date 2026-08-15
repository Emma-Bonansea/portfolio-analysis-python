#1.IMPORTS

import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from scipy.optimize import minimize


# 2. DATA COLLECTION & PARAMETERS

risk_free_data = pdr.DataReader("DGS3MO","fred",start="2021-01-01")
risk_free_data=risk_free_data.dropna()
risk_free_daily=(risk_free_data["DGS3MO"]/100) / 252
tickers = ["AAPL","JPM","MC.PA","TTE.PA","SAN.PA"]
data = yf.download(tickers,start="2021-01-01",auto_adjust=True)
benchmark_data=yf.download("^GSPC",start="2021-01-01",auto_adjust=True)
cac_data=yf.download("^FCHI",start="2021-01-01",auto_adjust=True)


# 3. BENCHMARK ANALYSIS

benchmark_close=benchmark_data["Close"]
cac_close=cac_data["Close"]
benchmark_normalized=benchmark_close/benchmark_close.iloc[0]*100
cac_normalized=cac_close/cac_close.iloc[0]*100
cac_returns=cac_close.pct_change().dropna()
cac_risk_free_daily = risk_free_daily.reindex(cac_returns.index).ffill()
cac_excess_returns = (cac_returns["^FCHI"] - cac_risk_free_daily)
cac_annual_return=cac_returns.mean().iloc[0]*252
cac_annual_volatility=cac_returns.std().iloc[0]*(252**0.5)
cac_total_return=(1 + cac_returns["^FCHI"]).prod() - 1
cac_years=(cac_returns.index[-1]-cac_returns.index[0]).days / 365.25
cac_cagr=(1 + cac_total_return)**(1/cac_years)-1
cac_cumulative=(1 + cac_returns["^FCHI"]).cumprod() * 100
cac_running_max=cac_cumulative.cummax()
cac_drawdown=cac_cumulative / cac_running_max - 1
cac_max_drawdown = cac_drawdown.min()
benchmark_returns=benchmark_close.pct_change().dropna()
benchmark_risk_free_daily = risk_free_daily.reindex(benchmark_returns.index).ffill()
benchmark_excess_returns = benchmark_returns["^GSPC"] - benchmark_risk_free_daily
benchmark_annual_return=benchmark_returns.mean().iloc[0]*252
benchmark_annual_volatility=benchmark_returns.std().iloc[0]*(252**0.5)
benchmark_total_return = (1 + benchmark_returns["^GSPC"]).prod() - 1
benchmark_years=(benchmark_returns.index[-1]-benchmark_returns.index[0]).days / 365.25
benchmark_cagr = (1 + benchmark_total_return) ** (1 / benchmark_years) - 1
benchmark_cumulative = (1 + benchmark_returns["^GSPC"]).cumprod() * 100
benchmark_running_max = benchmark_cumulative.cummax()
benchmark_drawdown = benchmark_cumulative / benchmark_running_max - 1
benchmark_max_drawdown = benchmark_drawdown.min()
cac_sharpe=(cac_excess_returns.mean()/cac_excess_returns.std())*(252**0.5)
benchmark_sharpe=(benchmark_excess_returns.mean()/benchmark_excess_returns.std())*(252**0.5)


# 4. CURRENT PORTFOLIO ANALYSIS

close_price = data["Close"]
daily_returns =  close_price.pct_change()
daily_returns =  daily_returns.dropna()
daily_volatility = daily_returns.std()
annual_volatility = daily_volatility * (252**0.5)
correlation_matrix=daily_returns.corr()
weights=np.array([0.20,0.20,0.20,0.20,0.20])
portfolio_daily_returns=daily_returns.dot(weights)
portfolio_annual_return=portfolio_daily_returns.mean()*252
portfolio_annual_volatility=portfolio_daily_returns.std()*(252**0.5)

risk_free_daily = risk_free_daily.reindex(portfolio_daily_returns.index).ffill()
portfolio_excess_returns=portfolio_daily_returns-risk_free_daily
aligned_excess_returns = pd.concat([portfolio_excess_returns.rename("Portfolio"),benchmark_excess_returns.rename("S&P 500")],axis=1,join="inner").dropna()

sharpe_ratio = (portfolio_excess_returns.mean()/ portfolio_excess_returns.std()) * (252**0.5)
normalized_prices=close_price/close_price.iloc[0]*100
portfolio_cumulative=(1+portfolio_daily_returns).cumprod()*100
running_max=portfolio_cumulative.cummax()
drawdown=portfolio_cumulative/running_max-1
max_drawdown=drawdown.min()
total_return=(1+portfolio_daily_returns).prod()-1
years=(portfolio_daily_returns.index[-1]-portfolio_daily_returns.index[0]).days/365.25
cagr=(1+total_return)**(1/years)-1


# 5. RISK METRICS

var_95=portfolio_daily_returns.quantile(0.05)
cvar_95=portfolio_daily_returns[portfolio_daily_returns<=var_95].mean()
covariance=aligned_excess_returns["Portfolio"].cov(aligned_excess_returns["S&P 500"])
market_variance = aligned_excess_returns["S&P 500"].var()
portfolio_beta = covariance / market_variance
alpha_daily =(aligned_excess_returns["Portfolio"]-portfolio_beta * aligned_excess_returns["S&P 500"])
portfolio_alpha = alpha_daily.mean() * 252


# 6. PORTFOLIO OPTIMIZATION

mean_daily_returns = daily_returns.mean()
cov_matrix = daily_returns.cov()
def portfolio_return(weights):
    return np.dot(mean_daily_returns, weights) * 252


def portfolio_volatility(weights):
    return np.sqrt(
        np.dot(
            weights.T,
            np.dot(cov_matrix * 252, weights)
        )
    )


def negative_sharpe(weights):
    daily_portfolio_returns = daily_returns.dot(weights)
    aligned_risk_free = risk_free_daily.reindex(
        daily_portfolio_returns.index
    ).ffill()

    excess_returns = daily_portfolio_returns - aligned_risk_free

    sharpe = (
        excess_returns.mean()
        / excess_returns.std()
    ) * (252 ** 0.5)

    return -sharpe
initial_weights = np.array([1 / len(tickers)] * len(tickers))
bounds = tuple((0, 1) for _ in range(len(tickers)))
constraints={"type": "eq","fun": lambda weights: np.sum(weights) - 1}
max_sharpe_result = minimize(negative_sharpe,initial_weights,method="SLSQP",bounds=bounds,constraints=constraints)
exact_max_sharpe_weights = max_sharpe_result.x
exact_max_sharpe_return = portfolio_return(exact_max_sharpe_weights)
exact_max_sharpe_volatility = portfolio_volatility(exact_max_sharpe_weights)
exact_max_sharpe_ratio = -max_sharpe_result.fun
min_vol_result = minimize(portfolio_volatility,initial_weights,method="SLSQP",bounds=bounds,constraints=constraints)
exact_min_vol_weights = min_vol_result.x
exact_min_vol_return = portfolio_return(exact_min_vol_weights)
exact_min_vol_volatility = portfolio_volatility(exact_min_vol_weights)
daily_min_vol_returns = daily_returns.dot(exact_min_vol_weights)
aligned_risk_free_min_vol = risk_free_daily.reindex(daily_min_vol_returns.index).ffill()
exact_min_vol_excess_returns = (daily_min_vol_returns - aligned_risk_free_min_vol)
exact_min_vol_sharpe = (exact_min_vol_excess_returns.mean()/ exact_min_vol_excess_returns.std()) * (252 ** 0.5)

exact_max_sharpe_daily_returns = daily_returns.dot(exact_max_sharpe_weights)
exact_max_sharpe_cumulative = (1 + exact_max_sharpe_daily_returns).cumprod()
exact_max_sharpe_total_return = (1 + exact_max_sharpe_daily_returns).prod() - 1
exact_max_sharpe_years = (exact_max_sharpe_daily_returns.index[-1]- exact_max_sharpe_daily_returns.index[0]).days / 365.25
exact_max_sharpe_cagr = (1 + exact_max_sharpe_total_return) ** (1 / exact_max_sharpe_years) - 1
exact_max_sharpe_running_max = exact_max_sharpe_cumulative.cummax()
exact_max_sharpe_drawdown = (exact_max_sharpe_cumulative/ exact_max_sharpe_running_max- 1)
exact_max_sharpe_max_drawdown = exact_max_sharpe_drawdown.min()
exact_min_vol_daily_returns = daily_returns.dot(exact_min_vol_weights)
exact_min_vol_cumulative = (1 + exact_min_vol_daily_returns).cumprod()
exact_min_vol_total_return = (1 + exact_min_vol_daily_returns).prod() - 1
exact_min_vol_years = (exact_min_vol_daily_returns.index[-1]- exact_min_vol_daily_returns.index[0]).days / 365.25
exact_min_vol_cagr = (1 + exact_min_vol_total_return) ** (1 / exact_min_vol_years) - 1
exact_min_vol_running_max = exact_min_vol_cumulative.cummax()
exact_min_vol_drawdown = (exact_min_vol_cumulative/ exact_min_vol_running_max- 1)
exact_min_vol_max_drawdown = exact_min_vol_drawdown.min()


# 7. EFFICIENT FRONTIER & MONTE CARLO SIMULATION

max_asset_return = (mean_daily_returns * 252).max()

target_returns = np.linspace(
    exact_min_vol_return,
    max_asset_return,
    50
)

frontier_returns = []
frontier_volatilities = []

for target_return in target_returns:

    frontier_constraints = (
        {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
        },
        {
            "type": "eq",
            "fun": lambda weights, target=target_return:
                portfolio_return(weights) - target
        }
    )

    result = minimize(
        portfolio_volatility,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=frontier_constraints
    )

    if result.success:
        frontier_returns.append(target_return)
        frontier_volatilities.append(
            portfolio_volatility(result.x)
        )

np.random.seed(42)
num_portfolios = 10000

portfolio_returns = []
portfolio_volatilities = []
portfolio_sharpes = []


for _ in range(num_portfolios):
    random_weights = np.random.random(len(tickers))
    random_weights = random_weights / random_weights.sum()

    random_return = np.dot(mean_daily_returns, random_weights) * 252

    random_volatility = np.sqrt(
        np.dot(
            random_weights.T,
            np.dot(cov_matrix * 252, random_weights)
        )
    )

    random_risk_free_daily = risk_free_daily.reindex(
        daily_returns.index
    ).ffill()

    random_excess_returns = (
        daily_returns.dot(random_weights) - random_risk_free_daily
    )

    random_sharpe = (
        random_excess_returns.mean()
        / random_excess_returns.std()
    ) * (252**0.5)

    portfolio_returns.append(random_return)
    portfolio_volatilities.append(random_volatility)
    portfolio_sharpes.append(random_sharpe)


portfolio_returns = np.array(portfolio_returns)
portfolio_volatilities = np.array(portfolio_volatilities)
portfolio_sharpes = np.array(portfolio_sharpes)


# 8. FINAL RESULTS TABLE

final_comparison = pd.DataFrame({
    "Metric": [
        "Annualized Mean Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "CAGR",
        "Maximum Drawdown"
    ],

    "Current Portfolio": [
        f"{portfolio_annual_return * 100:.2f}%",
        f"{portfolio_annual_volatility * 100:.2f}%",
        f"{sharpe_ratio:.2f}",
        f"{cagr * 100:.2f}%",
        f"{max_drawdown * 100:.2f}%"
    ],

    "Max Sharpe": [
        f"{exact_max_sharpe_return * 100:.2f}%",
        f"{exact_max_sharpe_volatility * 100:.2f}%",
        f"{exact_max_sharpe_ratio:.2f}",
        f"{exact_max_sharpe_cagr * 100:.2f}%",
        f"{exact_max_sharpe_max_drawdown * 100:.2f}%"
    ],

    "Min Volatility": [
        f"{exact_min_vol_return * 100:.2f}%",
        f"{exact_min_vol_volatility * 100:.2f}%",
        f"{exact_min_vol_sharpe:.2f}",
        f"{exact_min_vol_cagr * 100:.2f}%",
        f"{exact_min_vol_max_drawdown * 100:.2f}%"
    ],

    "S&P 500": [
        f"{benchmark_annual_return * 100:.2f}%",
        f"{benchmark_annual_volatility * 100:.2f}%",
        f"{benchmark_sharpe:.2f}",
        f"{benchmark_cagr * 100:.2f}%",
        f"{benchmark_max_drawdown * 100:.2f}%"
    ],

    "CAC 40": [
        f"{cac_annual_return * 100:.2f}%",
        f"{cac_annual_volatility * 100:.2f}%",
        f"{cac_sharpe:.2f}",
        f"{cac_cagr * 100:.2f}%",
        f"{cac_max_drawdown * 100:.2f}%"
    ]
})


# 9. VISUALIZATIONS

ax=normalized_prices.plot(figsize=(12,6))
portfolio_cumulative.plot(ax=ax,label="portfolio", linewidth=3)
benchmark_normalized["^GSPC"].plot(ax=ax,label="S&P 500", linewidth=3)
cac_normalized["^FCHI"].plot(ax=ax,label="CAC 40", linewidth=3)
plt.title("Stock performance - base 100")
plt.xlabel("Date")
plt.ylabel("Base 100")
plt.grid(True)
plt.legend()
plt.show()
drawdown.plot(figsize=(12,6))
plt.title("Portfolio Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown")
plt.grid(True)
plt.show()

fig, ax = plt.subplots(figsize=(8,6))
heatmap=ax.imshow(correlation_matrix)
ax.set_xticks(range(len(correlation_matrix.columns)))
ax.set_yticks(range(len(correlation_matrix.columns)))
ax.set_xticklabels(correlation_matrix.columns)
ax.set_yticklabels(correlation_matrix.columns)
plt.title("Correlation Matrix")
for i in range(len(correlation_matrix.columns)):
    for j in range (len(correlation_matrix.columns)):
        ax.text(j,i,f"{correlation_matrix.iloc[i,j]:.2f}", ha="center", va="center")
plt.colorbar(heatmap)
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(portfolio_volatilities*100,portfolio_returns*100,c=portfolio_sharpes)
plt.scatter(portfolio_annual_volatility*100,portfolio_annual_return*100,marker="o",s=120,label="Current Portfolio")
plt.scatter(exact_max_sharpe_volatility * 100,exact_max_sharpe_return * 100,marker="*",s=200,label="Exact Max Sharpe Portfolio")
plt.scatter(exact_min_vol_volatility * 100,exact_min_vol_return * 100,marker="X",s=150,label="Exact Minimum Volatility Portfolio")
plt.plot(np.array(frontier_volatilities) * 100,np.array(frontier_returns) * 100,linewidth=3,label="Efficient Frontier")
plt.xlabel("Annualized Volatility (%)")
plt.ylabel("Annualized Return (%)")
plt.title("Portfolio Optimization & Efficient Frontier")
plt.colorbar(label="Sharpe Ratio")
plt.legend()
plt.grid(True)
plt.show()


# 10. CONSOLE OUTPUT

print(f"Historical VaR 95%: {var_95*100:.2f}%")
print(f"Historical CVaR 95%: {cvar_95*100:.2f}%")
print(f"Portfolio Beta vs S&P 500: {portfolio_beta:.2f}")
print(f"Portfolio Alpha vs S&P 500: {portfolio_alpha*100:.2f}%")

print("\nEXACT MAX SHARPE PORTFOLIO")
print((pd.Series(exact_max_sharpe_weights, index=tickers) * 100).round(2))
print(f"Annualized Return: {exact_max_sharpe_return * 100:.2f}%")
print(f"Annualized Volatility: {exact_max_sharpe_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {exact_max_sharpe_ratio:.2f}")

print("\nEXACT MINIMUM VOLATILITY PORTFOLIO")
print((pd.Series(exact_min_vol_weights, index=tickers) * 100).round(2))
print(f"Annualized Return: {exact_min_vol_return * 100:.2f}%")
print(f"Annualized Volatility: {exact_min_vol_volatility * 100:.2f}%")
print(f"Sharpe Ratio: {exact_min_vol_sharpe:.2f}")

print("\nFINAL PORTFOLIO COMPARISON")
print(final_comparison.to_string(index=False))