# StockBuddy Atelier — Quantitative Market Intelligence Engine

StockBuddy Atelier is an institutional-grade quantitative finance platform built with Flask, scikit-learn, PyTorch/TensorFlow, and a visual terminal inspector. It processes multi-decade historical market datasets via live `yfinance` pipelines to discover market regimes, run vector cosine scenario matching, and backtest tactical quantitative allocations in real-time.

---

## ⚡ Key Engineering Features

- **Unsupervised K-Means Market Regime Discovery ($k=6$)**:
  - Classifies market conditions (Bullish Recovery, Momentum Breakout, Overbought Sideways, Volatile Neutral, Cyclical Pullback, Structural Bear Stress) using scaled feature vectors (RSI, 20d Volatility, EMA Slope, MACD Histogram).
- **Interactive Terminal Pipeline Inspector (7-Stage Spec Console)**:
  - Transparent visual data flow (`Input ➔ Process ➔ Output`) and parameter matrices detailing every step from data ingestion to quantitative portfolio backtesting.
- **Vector Nearest-Neighbor Scenario Matching ($k$-NN)**:
  - Computes Cosine Distance matrix dot products across 15+ years of daily technical vectors to return the top 5 historical market scenarios matching today's condition.
- **Tactical Allocation Strategy Backtester**:
  - Simulates dynamic equity curve allocations (100% Bull, 50% Neutral, 0% Bear) against Buy & Hold benchmark on $10,000 portfolio base.
- **Deep Learning Neural Signal Studio**:
  - Out-of-sample directional signal benchmarking using LSTM, GRU, and Transformer self-attention architectures trained on log returns $r_t = \ln(P_t / P_{t-1})$.
- **Quick Date Range Selection & Downsampling**:
  - Supports 1Y, 3Y, 5Y, 10Y, 15Y historical shortcuts with dynamic 500-point timeline downsampling for zero-latency UI rendering.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask REST API, Pandas, NumPy, scikit-learn, yfinance
- **Deep Learning**: TensorFlow / PyTorch (LSTM, GRU, Transformer)
- **Frontend**: Vanilla JS (ES6+), Modern Vanilla CSS, Chart.js 4, Google Fonts (Outfit, Inter, JetBrains Mono)
- **Architecture**: Decoupled RESTful API + Reactive Atelier Dashboard

---

## 🚀 Quickstart

1. **Install Dependencies**:
   ```bash
   pip install flask pandas numpy scikit-learn yfinance tensorflow
   ```

2. **Launch Application**:
   ```bash
   python app.py
   ```

3. **Access Dashboard**:
   Open browser at `http://localhost:8080` (or target host port).