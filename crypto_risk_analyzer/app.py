import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Live Crypto Risk Analyzer", layout="wide")

st.title("📊 Live Crypto Volatility & Risk Analyzer")

# Auto refresh every 30 seconds
st.markdown("⏱ Auto-refresh every **30 seconds**")
st.experimental_set_query_params(refresh=int(time.time()))
st_autorefresh = st.empty()

# -------------------------
# Select Cryptocurrencies
# -------------------------
assets = {
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Solana": "SOL-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD"
}

selected_assets = st.multiselect(
    "Select Cryptocurrencies",
    list(assets.keys()),
    default=list(assets.keys())
)

# -------------------------
# Fetch Live Data
# -------------------------
@st.cache_data(ttl=30)
def load_data(tickers):
    data = yf.download(tickers, period="1d", interval="1m", auto_adjust=True)
    return data["Close"]

data = load_data([assets[a] for a in selected_assets])

# -------------------------
# Calculate Volatility
# -------------------------
returns = data.pct_change().dropna()
volatility = returns.std() * np.sqrt(1440)  # intraday annualized

# -------------------------
# Risk Classification
# -------------------------
def classify(v):
    if v < 0.02:
        return "Low Risk"
    elif v < 0.05:
        return "Medium Risk"
    else:
        return "High Risk"

risk_df = pd.DataFrame({
    "Asset": volatility.index,
    "Volatility": volatility.values,
})
risk_df["Risk Level"] = risk_df["Volatility"].apply(classify)

# -------------------------
# DISPLAY
# -------------------------
st.subheader("📈 Live Risk Table")
st.dataframe(risk_df)

st.subheader("📊 Risk Distribution")
st.bar_chart(risk_df.set_index("Asset")["Volatility"])

st.success("Live data updated automatically every 30 seconds ✅")
