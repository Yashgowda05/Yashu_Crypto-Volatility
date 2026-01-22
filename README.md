# Yashu_Crypto-Volatility


A comprehensive cryptocurrency volatility and risk analysis project with interactive dashboards.  
This project analyzes historical and live crypto price data, calculates returns, risk metrics, and visualizes insights using Python and Streamlit.

---

## 📌 Features

### 🔹 Milestone 1 – BTC Data Analysis
- Fetch 1-year BTC-USD historical price data using `yfinance`.
- Calculate daily returns and cumulative returns.
- Compute volatility and Value at Risk (VaR).
- Save raw and processed data.
- Visualizations:
  - BTC price trend
  - Return distribution
  - Cumulative returns

### 🔹 Milestone 2 – Crypto Risk Analysis Dashboard
- Multi-crypto support: BTC, ETH, SOL, ADA, DOGE, LTC, XRP, DOT.
- Calculate Sharpe ratio, beta vs BTC, rolling volatility.
- Classify risk as Low, Medium, High.
- Interactive dashboard with Plotly & Streamlit.

### 🔹 Milestone 3 – Interactive Visualization Dashboard
- Price trend and rolling volatility over time.
- Risk vs Return scatter plots.
- Radar chart for risk profile.
- KPI metrics and comparative analysis.

### 🔹 Milestone 4 – Risk Classification & Reporting
- Classify assets into risk buckets.
- Visual risk cards and distribution charts.
- Downloadable reports (CSV & PDF).
- Summary of project milestones and completion status.

---

## 📊 Dashboards and Screenshots

You can view dashboard images in the `images/` folder:

- Milestone 1: `cumulative_returns.png`, `price_trend.png`, `return_distribution.png`  
- Milestone 2: `milestone2_dashboard.png`, `milestone2p1.png`, `milestone2p2.png`  
- Milestone 3: `milestone3_dashboard.png`, `milestone3dfp.png`, `milestone3p1.png` ...  
- Milestone 4: `milestone4_dashboard.png`, `milestone4dfp.png`, `milestone4p1.png` ...

*(Refer to the `images/` folder in the repo for all screenshots.)*

---

## 🛠 Technologies Used

- Python 3.x  
- `yfinance`, `requests`  
- `pandas`, `numpy`  
- `matplotlib`, `seaborn`, `plotly`  
- `streamlit`  
- `FPDF`  

---

## 🚀 Installation

1. Clone the repository:

```bash
git clone https://github.com/arunachalamhemalatha/Hemalatha_Crypto_Volatility_And_RiskAnalyzer.git
cd Hemalatha_Crypto_Volatility_And_RiskAnalyzer
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Run the main dashboard:

bash
Copy code
streamlit run app.py
Optional: Run milestone dashboards individually:

bash
Copy code
streamlit run milestone_2_dashboard.py
streamlit run milestone_3_dashboard.py
streamlit run milestone_4_dashboard.py
👩‍💻 Author
Hemalatha

📌 Notes

Live data is fetched from the CoinGecko API.

Demonstrates real-time price fetching, risk metric calculation, and interactive analytics.

Exportable reports include CSV and PDF.
