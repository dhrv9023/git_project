# StockBuddy — AI Context Document

## What This Project Is

StockBuddy is a **quantitative stock market intelligence web application**. It is a two-file project:
- `app.py` — A Python Flask backend (~1100 lines)
- `index.html` — A self-contained frontend dashboard (vanilla JS + CSS, ~2700 lines)

The app lets a user type any stock ticker (e.g., `AAPL`, `TSLA`, `INFY.NS`) and instantly get a quantitative analysis of that stock's current market condition, historical regimes, and AI-generated price signals — all rendered in a premium dark-themed dashboard UI.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, Flask-CORS |
| Data | `yfinance` (live market data), Pandas, NumPy |
| ML / Quant | scikit-learn (KMeans, MinMaxScaler, cosine_similarity), TensorFlow (LSTM, GRU, Transformer) |
| Frontend | Vanilla HTML/CSS/JS (ES6+), Chart.js 4 |
| Fonts | Google Fonts — Outfit, Inter, JetBrains Mono |
| Architecture | Decoupled REST API + Single-Page App (SPA) dashboard |

---

## How to Run

```bash
# 1. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn yfinance tensorflow

# 2. Start the Flask API server
python app.py serve

# 3. Open the frontend
# Just open index.html in a browser (it calls http://localhost:8080)
```

---

## Backend Architecture (`app.py`)

The backend exposes **three REST API endpoints**, all accepting POST with a JSON body containing `{ "ticker": "AAPL", "start_date": "...", "end_date": "..." }`.

### `POST /api/regime` — Main Analysis Endpoint
This is the core endpoint. It runs the full quantitative pipeline:

1. **Data Fetch** — Downloads OHLCV data via `yfinance`
2. **Preprocessing** — Sorts, deduplicates, IQR outlier filtering, winsorizes tails
3. **Feature Engineering** — Computes: RSI(14), EMA(20), MACD(12,26,9), Log Returns, Day of Week
4. **Regime Classification (KMeans, k=6)** — Clusters daily market conditions into 6 semantic regimes:
   - 0: Trending Bull
   - 1: Overbought / Exhaustion
   - 2: Sideways / Choppy
   - 3: Recovery / Bounce
   - 4: Downtrend / Bear
   - 5: High Volatility / Stress
5. **Risk Score (0–10)** — Composite of: vol stress, RSI extremity, trend misalignment, MACD divergence
6. **Historical Scenario Matching** — Finds top-5 most similar historical dates using Cosine Similarity on feature vectors
7. **Quantitative Backtest** — Simulates a regime-based allocation strategy (100% equity in bull, 50% in neutral, 0% in bear) vs Buy & Hold benchmark

**Response JSON keys:** `ticker`, `current_regime`, `risk_score`, `alert`, `indicators`, `timeline`, `regime_stats`, `similar_scenarios`, `quant_backtest`

### `POST /api/train` — Deep Learning Signal Training
Trains LSTM, GRU, and Transformer models on historical log-return sequences and returns prediction accuracy metrics (RMSE, MAE, Directional Accuracy, R2).

**Training config:**
- Sequence length: 90 days
- Train/Val/Test split: 70/15/15
- Target: next-step log return → reconstructed as price
- Ensemble: equal-weight average of all 3 models
- Output: per-model metrics + confidence intervals (±1.96 std)

### `GET /api/health`
Simple health check, returns `{ "status": "ok" }`.

---

## Frontend Architecture (`index.html`)

Single self-contained HTML file. No build tools or NPM.

### Key Sections / Panels

1. **Hero Command Studio** — Ticker search bar at top. User types a ticker and hits Enter/Search. Triggers `/api/regime`.
2. **Regime Intelligence Panel** — Shows the current market regime badge, Risk Condition Score (gauge), and a condition alert (GREEN/YELLOW/RED).
3. **Regime Timeline Chart** — Color-coded timeline of all historical regime classifications. Built with Chart.js.
4. **Live Technical Indicators** — RSI, EMA20, MACD, Close price telemetry cards.
5. **Historical Scenario Matching Panel** — Shows top-5 cosine-similar historical dates with their forward 10-day price path chart and forward return.
6. **Regime Statistics Table** — Per-regime median/mean forward returns at 5d, 10d, 20d horizons, and % of time price went up.
7. **Quantitative Backtest Panel** — Equity curve chart comparing regime strategy vs Buy & Hold. Shows Sharpe ratio, max drawdown, total return, win rate.
8. **Neural Signal Studio** — "Train AI Models" button that triggers `/api/train`. Shows LSTM/GRU/Transformer/Ensemble metrics in a table, plus a prediction vs actual chart.
9. **Execution Pipeline Inspector** — A visual 7-stage diagram showing the data flow of the entire quantitative pipeline (for transparency).

---

## Current State & Known Info

- The server **runs on port 8080** by default (`python app.py serve`)
- The frontend is a **static file** — just open `index.html` directly in a browser
- The frontend calls `http://localhost:8080` (hardcoded) — change this if deploying remotely
- GPU support is auto-detected; mixed precision (float16) is enabled if a GPU is found
- The project is stored at: `/home/dhruv/Desktop/stockbuddy/StockBuddy/`
- Git remote: `https://github.com/dhrv9023/git_project.git` (branch: `main`)

---

## File Map

```
StockBuddy/
├── app.py          # Flask backend — all Python logic (data, ML, API)
├── index.html      # Frontend SPA — all UI, charts, interactions
├── README.md       # Project overview
└── PROJECT_SUMMARY.md  # This file
```

---

## Key Design Decisions

- **No database** — All analysis is computed live on-demand from yfinance
- **No separate frontend framework** — Pure vanilla JS in a single HTML file for zero-dependency simplicity
- **Regime IDs are semantically ordered** — KMeans raw cluster IDs are remapped so that regime 0 is always the most bullish (by a scoring function: `2*Ret10 + 1.5*RSI_dev - 3*Vol`)
- **Training is on-demand** — Deep learning models are trained fresh on each `/api/train` call (not persisted to disk)
- **Confidence intervals** — Computed as ±1.96 × std of model predictions (model disagreement as proxy for uncertainty)
