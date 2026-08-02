# StockBuddy Atelier — Quantitative Market Intelligence Engine

StockBuddy Atelier is an institutional-grade quantitative finance platform built with Flask, scikit-learn, TensorFlow, and a visual terminal inspector. It processes multi-decade historical market datasets via live `yfinance` pipelines to discover market regimes, run vector cosine scenario matching, and backtest tactical quantitative allocations in real-time.

> **Phase 1 – Statistical Corrections** have been applied. All backtest metrics, model accuracy figures, and regime statistics are now statistically valid. See [Statistical Integrity](#statistical-integrity) section.

---

## ⚡ Key Engineering Features

- **Unsupervised K-Means Market Regime Discovery (k=6)**:
  - Classifies market conditions (Bullish Recovery, Momentum Breakout, Overbought Sideways, Volatile Neutral, Cyclical Pullback, Structural Bear Stress) using scaled feature vectors (RSI, 20d Volatility, EMA Slope, MACD Histogram).
- **Interactive Terminal Pipeline Inspector (7-Stage Spec Console)**:
  - Transparent visual data flow (`Input ➔ Process ➔ Output`) and parameter matrices detailing every step from data ingestion to quantitative portfolio backtesting.
- **Vector Nearest-Neighbor Scenario Matching (k-NN)**:
  - Computes Cosine Distance matrix dot products across 15+ years of daily technical vectors to return the top 5 historical market scenarios matching today's condition.
- **Tactical Allocation Strategy Backtester**:
  - Simulates dynamic equity curve allocations (100% Bull, 50% Neutral, 0% Bear) against Buy & Hold benchmark on $10,000 portfolio base.
  - **Phase 1:** Includes 0.10% round-trip transaction costs and correct Sharpe ratio with 5% risk-free rate.
- **Deep Learning Neural Signal Studio**:
  - Out-of-sample directional signal benchmarking using LSTM, GRU, and Transformer self-attention architectures trained on log returns r_t = ln(P_t / P_{t-1}).
  - **Phase 1:** Scalers are now fit only on training data — test metrics are statistically valid.
- **Walk-Forward Validation (`POST /api/wf_validate`)**:
  - **Phase 1 addition.** Expanding-window cross-validation across N folds gives mean ± std directional accuracy — replacing the single cherry-picked 70/15/15 split.
- **In-Memory Model Cache**:
  - **Phase 1 addition.** Trained models are cached by `{ticker}_{dates}_{config}` key. Repeat calls for same ticker return instantly without retraining.

---

## 🛡️ Statistical Integrity

Phase 1 corrected the following critical issues:

| # | Severity | Issue | Fix |
|---|---|---|---|
| BUG-01 | 🔴 CRITICAL | Data leakage — scaler fit on full dataset before train/test split | Scaler now fit on training partition only |
| BUG-02 | 🔴 CRITICAL | Sequence boundary leakage across split | Context-window approach; val/test sequences never cross a raw boundary |
| BUG-03 | 🟠 HIGH | No transaction costs in backtest | 0.10% round-trip cost applied on every weight change |
| BUG-04 | 🟠 HIGH | Single static 70/15/15 split — overfitting risk | Walk-forward validation with N expanding folds |
| BUG-05 | 🟡 MEDIUM | Full model retrain every API call | In-memory model cache keyed by ticker+config |
| BUG-06 | 🟡 MEDIUM | Regime forward-return stats on incomplete windows | Last 20 rows excluded from forward-return summaries |
| BUG-07 | 🟡 MEDIUM | Sharpe ratio computed without risk-free rate | 5% annual Rf subtracted in all Sharpe calculations |

**What this means for reported metrics:**
- RMSE / Directional Accuracy from `/api/train` are now genuinely out-of-sample.
- Sharpe ratios will be ~0.1–0.4 lower than before (more honest).
- Strategy returns will be slightly lower due to transaction costs.
- Regime statistics are now computed only on dates with complete forward windows.

---

## 🛠️ Technology Stack

- **Backend**: Python 3.x, Flask REST API, Pandas, NumPy, scikit-learn, yfinance
- **Deep Learning**: TensorFlow (LSTM, GRU, Transformer)
- **Frontend**: Vanilla JS (ES6+), Modern Vanilla CSS, Chart.js 4, Google Fonts (Outfit, Inter, JetBrains Mono)
- **Architecture**: Decoupled RESTful API + Reactive Atelier Dashboard

---

## 🚀 Quickstart

1. **Install Dependencies**:
   ```bash
   pip install flask flask-cors pandas numpy scikit-learn yfinance tensorflow
   ```

2. **Launch Application**:
   ```bash
   python app.py serve
   ```

3. **Access Dashboard**:
   Open `index.html` in your browser (calls `http://localhost:5000`).

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + model cache status |
| `POST` | `/api/regime` | Main regime analysis pipeline |
| `POST` | `/api/predict` | Train DL models + forecast (with cache) |
| `POST` | `/api/wf_validate` | Walk-forward validation (N folds) — Phase 1 |

### `/api/wf_validate` example:
```json
POST /api/wf_validate
{
  "ticker": "AAPL",
  "start_date": "2018-01-01",
  "end_date": "2024-01-01",
  "n_folds": 5,
  "model": "GRU",
  "epochs": 10
}
```

---

## ⚠️ Known Limitations

- Models are not saved to disk — cache is lost on server restart.
- Walk-forward validation is computationally heavy (CPU: ~5–30 min for 5 folds × 3 models).
- The regime engine uses KMeans with a fixed k=6; regime semantics may shift for non-US markets.
- No slippage model beyond the flat transaction cost percentage.