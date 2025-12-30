# ===============================
# Milestone 2 – Data Processing and Calculation
# Project: Crypto Risk Analysis
# Assets: BTC, ETH, SOL, ADA, DOGE
# Benchmark: BTC
# Data: Daily Closing Prices
# ===============================

# --------- Libraries ---------
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# --------- Configuration ---------
assets = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD"
}

start_date = "2021-01-01"
end_date = None   # till today
TRADING_DAYS = 252

# ===============================
# Part A: Data Preparation
# ===============================

price_data = pd.DataFrame()

for name, ticker in assets.items():
    df = yf.download(ticker, start=start_date, end=end_date)
    price_data[name] = df['Close']

# Handle missing values
price_data = price_data.dropna()

# Ensure date format and sorting
price_data.index = pd.to_datetime(price_data.index)
price_data = price_data.sort_index()

print("Clean Price Dataset:\n", price_data.head())

# ===============================
# Part B: Daily Log Returns
# ===============================

log_returns = np.log(price_data / price_data.shift(1))
log_returns = log_returns.dropna()

print("\nDaily Log Returns (Sample):\n", log_returns.head())

# ===============================
# Part C: Statistical Measures
# ===============================

# ---- Volatility ----
daily_volatility = log_returns.std()
annual_volatility = daily_volatility * np.sqrt(TRADING_DAYS)

# ---- Sharpe Ratio (Rf = 0) ----
mean_daily_return = log_returns.mean()
sharpe_ratio = (mean_daily_return * TRADING_DAYS) / annual_volatility

# ---- Beta (Benchmark = BTC) ----
betas = {}
market_returns = log_returns['BTC']

for asset in log_returns.columns:
    cov = np.cov(log_returns[asset], market_returns)[0][1]
    var = np.var(market_returns)
    betas[asset] = cov / var

# ---- Metrics Table ----
metrics = pd.DataFrame({
    'Annual Volatility': annual_volatility,
    'Sharpe Ratio': sharpe_ratio,
    'Beta (vs BTC)': pd.Series(betas)
})

print("\nRisk Metrics Table:\n", metrics)

# ===============================
# Part D: Moving Average & Rolling Volatility
# ===============================

moving_avg_30 = price_data.rolling(window=30).mean()
rolling_vol_30 = log_returns.rolling(window=30).std() * np.sqrt(TRADING_DAYS)

# ===============================
# Part E: Visualization
# ===============================

# ---- Volatility Bar Chart ----
plt.figure()
annual_volatility.plot(kind='bar')
plt.title('Annualized Volatility Comparison')
plt.ylabel('Volatility')
plt.xlabel('Cryptocurrency')
plt.show()

# ---- Rolling Volatility Line Chart ----
plt.figure()
for asset in rolling_vol_30.columns:
    plt.plot(rolling_vol_30.index, rolling_vol_30[asset], label=asset)

plt.title('30-Day Rolling Volatility')
plt.ylabel('Volatility')
plt.xlabel('Date')
plt.legend()
plt.show()

# ===============================
# Part F: Interpretation & Inference
# ===============================

most_volatile = annual_volatility.idxmax()
best_sharpe = sharpe_ratio.idxmax()

print("\nMost Volatile Asset:", most_volatile)
print("Best Risk-Adjusted Return (Sharpe):", best_sharpe)
print("\nBeta Interpretation:")
print(metrics['Beta (vs BTC)'])

# ===============================
# Final Outputs
# ===============================

# Save processed datasets
price_data.to_csv("crypto_prices_cleaned.csv")
log_returns.to_csv("crypto_log_returns.csv")
metrics.to_csv("crypto_risk_metrics.csv")

print("\nFiles saved:")
print("- crypto_prices_cleaned.csv")
print("- crypto_log_returns.csv")
print("- crypto_risk_metrics.csv")
