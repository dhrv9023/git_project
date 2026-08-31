# StockBuddy — Platform User Guide

> **Who is this for?** Anyone using or integrating with the StockBuddy platform,
> including analysts, developers, and API consumers. This guide explains every
> section of the dashboard and what happens behind the scenes when you use it.

---

## Table of Contents

1. [Getting Started](#1-getting-started)
2. [Dashboard — Live Market Overview](#2-dashboard--live-market-overview)
3. [Regime Analysis](#3-regime-analysis)
4. [AI Predict — Deep Learning Forecast](#4-ai-predict--deep-learning-forecast)
5. [Walk-Forward Validation](#5-walk-forward-validation)
6. [Quant Research](#6-quant-research)
7. [AI Intelligence](#7-ai-intelligence)
8. [Portfolio Optimizer](#8-portfolio-optimizer)
9. [Settings](#9-settings)
10. [API Reference (Quick)](#10-api-reference-quick)
11. [Deployment Options](#11-deployment-options)

---

## 1. Getting Started

### Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python app.py serve
# or: gunicorn app:flask_app --bind 0.0.0.0:5000

# Open the dashboard
open http://localhost:5000
```

### Docker

```bash
docker-compose up --build
# Dashboard: http://localhost:80
```

### Environment Variables

Copy `.env.example` to `.env` and set:
```env
ENVIRONMENT=development     # development | staging | production
PORT=5000
SENTRY_DSN=                 # optional Sentry error tracking
LOG_FORMAT=text             # text | json
SB_MODEL_STALE_DAYS=7       # days before auto-retrain
```

---

## 2. Dashboard — Live Market Overview

### What you see

- **Live Quote Ticker** — last price + 1-day percentage change for up to 8 symbols
- **Current Regime Badge** — the K-Means cluster that today's market fits into
- **Risk Score Gauge** — a 0–10 composite risk indicator
- **Condition Alert** — GREEN / YELLOW / RED with interpretation text
- **Latest Technical Indicators** — RSI, EMA20, MACD, Close

### What happens behind the scenes

**Live Quotes:** `GET /api/v1/quotes?tickers=AAPL,NVDA,TSLA,SPY`

- Fetches `yf.Ticker(t).fast_info` for each symbol
- Results cached server-side for 60 seconds to avoid hammering yfinance
- Returns `last_price`, `previousClose`, and derived `change_pct`

**Regime Badge + Risk Score:** Loaded from the last `POST /api/regime` call.
See [Regime Analysis](#3-regime-analysis) for details.

---

## 3. Regime Analysis

### What you see

- **Market Regime Indicator** — which of 6 named regimes the market is in today
- **Risk Score Timeline** — a line chart of 0-10 daily risk scores over history
- **Regime Timeline** — a coloured bar showing which regime was active each day
- **Regime Performance Table** — for each regime: how many days occurred historically, median 5/10/20 day forward returns, and % of times those returns were positive
- **Similar Historical Scenarios** — top 5 past dates whose technical fingerprint is most similar to today, with their subsequent 10-day trajectories
- **Quant Backtest Chart** — regime-allocation strategy equity curve vs buy-and-hold

### Behind the scenes: `POST /api/regime`

Request:
```json
{ "ticker": "AAPL", "start_date": "2021-01-01", "end_date": "2024-12-31" }
```

**Step 1 — Data fetch**
yfinance downloads OHLCV. Data is cleaned: duplicates removed, gaps forward-filled,
extreme returns (outside 3x IQR) dropped, and tails winsorized at p1/p99.

**Step 2 — Feature engineering**
12 technical indicators computed from the cleaned Close price:
- RSI(14): momentum oscillator 0-100
- EMA(20): trend direction indicator
- MACD (12/26/9): trend + momentum crossover signal
- LogReturn: `log(Close[t] / Close[t-1])` — stationarity-preserving return
- DayOfWeek: captures weekly seasonality

**Step 3 — Regime classification (K-Means)**
5 features are extracted: RSI deviation from 50, MACD histogram normalized by price,
EMA slope, 20-day rolling volatility, and 10-day price return.

These are standardized (zero-mean, unit-variance) and fed into K-Means with 6 clusters.
Clusters are semantically ordered by a composite score so the labels are reproducible:
- Regime 0 (Trending Bull): high Ret10, positive RSI, low vol
- Regime 1 (Overbought/Exhaustion): high RSI, elevated vol
- Regime 2 (Sideways/Choppy): near-zero returns, no momentum
- Regime 3 (Recovery/Bounce): recovering RSI, mean-reversion setup
- Regime 4 (Downtrend/Bear): negative returns, bearish momentum
- Regime 5 (High Vol/Stress): extreme volatility, regime instability

**Step 4 — Risk Score (0–10)**
Four components combined:
- Vol Stress (0-3): how much 20-day vol exceeds 90-day average
- Momentum Stress (0-3): RSI extremity from 50 in either direction
- Trend Misalignment (0-2): Close below EMA20
- MACD Divergence (0-2): MACD histogram negative

Smoothed with a 5-day rolling mean. Bear (4) and Stress (5) regimes are floored at 7.5.

**Step 5 — Regime forward-return statistics**
For each regime, all historical occurrences are collected. For each occurrence,
the 5-day, 10-day, and 20-day forward returns are measured. Only dates with a
FULL forward window are included (last 20 days excluded to avoid incomplete windows).

Statistics returned: median, mean, % positive, sample count.

**Step 6 — Historical scenario matching**
The current day's feature vector (RSI_dev, MACD_hist_norm, EMA_slope, Vol20, Ret10)
is compared via cosine similarity to every historical day's vector.
Top 5 most similar days are returned with their actual subsequent 10-day price paths
(normalized to 100 at day 0), so you can see what "usually happened next" in
similar conditions.

**Step 7 — Quant backtest**
A regime-allocation strategy is simulated:
- Regimes 0/3: 100% equity (fully invested)
- Regimes 1/2: 50% equity, 50% cash
- Regimes 4/5: 0% equity (all cash)

Transaction costs of 0.1% (configurable) are applied every time the allocation
weight changes. The resulting equity curve is compared against buy-and-hold.
Sharpe ratio uses the excess return over a 5% annual risk-free rate.

---

## 4. AI Predict — Deep Learning Forecast

### What you see

- **Model Training Progress** — LSTM, GRU, and Transformer training in the background
- **Actual vs Predicted Chart** — the ensemble's price predictions overlaid on actuals
  for the held-out test set, with a 95% confidence band
- **5-Day Forward Forecast** — autoregressive rolling prediction beyond the last known price
- **Individual Model Metrics** — RMSE, MAE, MAPE, R2, Directional Accuracy for each model
- **Training Loss Curves** — val_loss and train_loss per epoch

### Two modes

**Quick (v1) — trains from scratch:**
`POST /api/predict` submits a synchronous training job. It trains all three models,
runs ensemble evaluation, and returns all results in one response. Training takes
2–10 minutes depending on epochs and hardware.

**Production (v2) — cached inference:**
`POST /api/v2/train` — submits an async background training job, returns immediately
with a `job_id`. Poll `GET /api/v2/jobs/{job_id}` for status.
Once training is done (status="done"), `POST /api/v2/predict` loads the saved model
from disk and runs inference in seconds (cached for 1 hour).

### Behind the scenes: model training

**Target variable:** `LogReturn.shift(-1)` — tomorrow's log-return.
The model predicts log-returns, not prices. Prices are reconstructed as:
`price[t+1] = price[t] * exp(predicted_logret)`

This avoids the non-stationarity of raw prices.

**Train/Val/Test split (70/15/15 chronological):**
- Training: oldest 70% of data (only these rows are used to fit the scaler)
- Validation: next 15% (used for early stopping and LR scheduling)
- Test: final 15% (never seen during training; reported as the evaluation result)

**Scaler discipline:**
The MinMaxScaler is fit ONLY on training rows and applied via `transform()` to val
and test. This prevents future data from leaking into normalization.

**Context windows:**
Val and test sequences prepend the last 90 (sequence_length) rows of the preceding
partition as lookback context. This ensures even `val_seq[0]` has a full lookback
window without wasting data.

**Model architectures:**
- LSTM: Two stacked layers (128 and 64 units) with recurrent dropout and L2 reg
- GRU: Same topology as LSTM but GRU cells — 20-30% faster on CPU
- Transformer: Two self-attention encoder blocks with 4 heads and learnable position
  embeddings; best for capturing long-range temporal patterns

All use Huber loss (delta=1.0) — less sensitive to outlier log-return spikes than MSE.

**Training callbacks:**
- EarlyStopping: stops if val_loss doesn't improve for 5 epochs, restores best weights
- ReduceLROnPlateau: halves LR if val_loss stalls for 5 epochs
- LearningRateScheduler: 10% linear warmup + cosine decay

**Ensemble:**
Each model's predictions are averaged with equal weights (1/N). The spread (std) of
model predictions forms the confidence interval: `ensemble ± 1.96 * std`.

**5-day autoregressive forecast:**
Starting from the last known sequence, the ensemble predicts day+1, then feeds that
prediction back as the new most-recent observation, repeating for 5 steps.
The Close column of the sequence is updated using the predicted price (scaled back
using the train-fitted MinMaxScaler).

---

## 5. Walk-Forward Validation

### What you see

- **Per-Fold Metrics** — RMSE, MAE, R2, Directional Accuracy for each fold
- **Summary Statistics** — mean ± std across all folds
- **Interpretation Text** — explains what the results mean in context

### What it tests and why it matters

A single 70/15/15 split evaluates the model on exactly ONE historical period.
If that period happens to be calm or trending, metrics look artificially good.

Walk-forward validation tests the model across N disjoint out-of-sample periods:

```
Fold 1: Train on [0, 20%]    -> Test on [20%, 40%]
Fold 2: Train on [0, 40%]    -> Test on [40%, 60%]  (expanding anchor)
Fold 3: Train on [0, 60%]    -> Test on [60%, 80%]
...
```

Each test block was NEVER seen during that fold's training. The scaler is re-fit
from scratch for each fold using only that fold's training rows.

If Directional Accuracy is consistently above 55% across all folds, there is likely
genuine predictive signal. Values near 50% indicate near-random.

### Behind the scenes: `POST /api/wf_validate`

Request:
```json
{
  "ticker": "AAPL",
  "n_folds": 5,
  "model": "GRU",
  "epochs": 10,
  "batch_size": 32
}
```

Each fold: fetch raw data once -> for this fold's training window, fit scaler ->
scale train partition -> create training sequences -> train model -> evaluate on
test block (with context window prepended) -> record metrics.

---

## 6. Quant Research

### What you see

A full institutional-grade quantitative report with 17 metrics organized into sections:
- Performance Summary: Sharpe, Sortino, Calmar, CAGR, Max Drawdown
- Market Attribution: Jensen's Alpha, Beta vs SPY
- Information Coefficient: IC value + t-test significance
- Rolling Analytics: 30-day volatility and 60-day Sharpe charts
- Monte Carlo Simulation: 1,000 future path fan chart (5th/50th/95th percentile)
- Feature Importance: which indicators matter most for regime clustering
- Regime Transition Matrix: probability of switching between regimes

### Behind the scenes: `POST /api/v5/quant`

Request:
```json
{ "ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2024-12-31" }
```

**Information Coefficient (IC):**
Spearman rank correlation between next-day log-returns (actual) and same-day
log-returns (used as a proxy signal). Values > 0.05 with p < 0.05 indicate
statistically significant predictive signal.

**Alpha and Beta:**
OLS regression: `daily_return = alpha + beta * spy_return`. Alpha is Jensen's alpha —
the return unexplained by market exposure. Beta measures sensitivity to the market.

**Monte Carlo:**
1,000 paths simulated using Geometric Brownian Motion with drift = historical mean
log-return and volatility = historical daily std. Each path covers 252 trading days.
The 5th/50th/95th percentile paths are returned for visualization.

**Regime Transition Matrix:**
For each consecutive pair of days, the regime transition is recorded.
`P[i, j] = P(next day is regime j | today is regime i)`.
Persistent diagonal values (e.g., P[0,0] = 0.85) mean the regime is sticky (tends
to persist). Off-diagonal transitions reveal regime switching dynamics.

---

## 7. AI Intelligence

### What you see

- **AI Market Brief** — an executive summary narrative about the stock's condition
- **XAI Feature Attribution** — a bar chart showing which technical indicators
  are driving the AI's signal (SHAP-style weights)
- **Sentiment Score** — VADER-based sentiment analysis
- **Sector Heatmap** — 1-day performance of major sector ETFs (XLK, XLF, XLV, etc.)
- **Macro Calendar** — upcoming economic events stub

### Behind the scenes: `POST /api/v7/ai/explain`

Request:
```json
{
  "ticker": "AAPL",
  "regime": "Trending Bull",
  "signal": "UP",
  "confidence": 0.72,
  "last_price": 182.50,
  "return_1m": 3.8,
  "volatility_20d": 16.4,
  "rsi": 58.2
}
```

The `AIMarketSynthesizer` generates:
1. A headline describing regime + confidence
2. An executive summary incorporating price, returns, vol, RSI, and regime
3. 4 key takeaways (regime win-rate, RSI interpretation, vol assessment, recommendation)
4. XAI feature attributions — the contribution of each indicator to the signal
   (RSI deviation from 50, vol above/below 15%, EMA slope, regime vector distance, MACD)

**Note:** This uses structured template synthesis, not an external LLM API.

**Sector Heatmap** — `GET /api/v7/market/intelligence?ticker=AAPL`
Fetches live 1-day returns for XLK, XLF, XLV, XLE, XLY, XLI from yfinance.
Results cached server-side for 5 minutes.

---

## 8. Portfolio Optimizer

### What you see

- **Efficient Frontier Chart** — scatter plot of 200 random portfolios in
  (annualized vol, annualized return) space, coloured by Sharpe ratio
- **Optimal Portfolios** — highlighted dots for Minimum Variance and Maximum Sharpe
- **Allocation Table** — the weight assigned to each asset in each optimal portfolio

### Behind the scenes: `POST /api/v7/portfolio/optimize`

Request:
```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "SPY"],
  "risk_free_rate": 0.05,
  "start_date": "2022-01-01",
  "end_date": "2024-01-01"
}
```

**Step 1 — Fetch historical returns:**
Log-returns are computed from daily Close prices for each ticker.
If yfinance fails for a ticker, synthetic returns from a normal distribution are used
as a fallback to prevent the whole request from failing.

**Step 2 — Covariance matrix:**
The sample covariance matrix of daily log-returns is computed across all tickers.

**Step 3 — Efficient frontier sampling:**
200 random weight vectors are generated (Dirichlet distributed, so they sum to 1).
For each portfolio: annualized return = `mean(returns) * 252`, annualized vol =
`sqrt(weights @ cov_matrix @ weights * 252)`, Sharpe = `(ret - rf) / vol`.

**Step 4 — Optimal portfolios:**
- Minimum variance: the weight vector minimizing portfolio variance
- Maximum Sharpe: the weight vector maximizing `(ret - rf) / vol`

---

## 9. Settings

The Settings panel controls defaults for all other panels:

| Setting | Effect |
|---------|--------|
| Ticker | Stock symbol used in all API calls |
| Start Date | Historical data start date |
| End Date | Historical data end date |
| Model | Which model to train/validate (LSTM / GRU / Transformer / All) |
| Epochs | Training epochs (default: 20, walk-forward: 10) |
| Sequence Length | Lookback window in days (default: 90) |
| Future Days | Autoregressive forecast horizon (default: 5) |
| Force Retrain | Bypass server-side model cache |

---

## 10. API Reference (Quick)

All endpoints return JSON. Base URL: `http://localhost:5000`

### Health
```
GET  /health           Container liveness probe (always 200 if running)
GET  /ready            Readiness probe — checks storage, circuit breaker, queue
GET  /api/health       Full system health + registry stats
```

### v1 — Core Analytics
```
GET  /api/v1/quotes?tickers=AAPL,NVDA   Live quotes for up to 8 tickers
POST /api/regime                         Regime analysis + risk score + backtest
POST /api/predict                        Train + evaluate + forecast (synchronous)
POST /api/wf_validate                    Walk-forward N-fold validation
```

### v2 — Production ML Pipeline
```
POST /api/v2/train                       Enqueue async training job -> returns job_id
GET  /api/v2/jobs/{job_id}              Poll training job status
GET  /api/v2/jobs                        List all training jobs
POST /api/v2/predict                     Cached inference from saved model
GET  /api/v2/registry                    Full model registry
GET  /api/v2/registry/{ticker}           All versions for one ticker
DELETE /api/v2/cache                     Flush prediction cache (both layers)
GET  /api/v2/metrics                     System health and performance counters
```

### v3 — Distributed Systems
```
GET  /api/v3/metrics                    Prometheus text or JSON metrics
GET  /api/v3/queue                       Priority queue status + DLQ contents
POST /api/v3/queue/dlq/{id}/requeue     Manually retry a dead-lettered job
POST /api/v3/train                       Enqueue via priority queue with retry policy
POST /api/v3/predict                     Cached inference + circuit-breaker protection
GET  /api/v3/breakers                   All circuit breaker states
POST /api/v3/breakers/{name}/reset      Manually reset a circuit breaker
GET  /api/v3/rate-limiter               Per-client token bucket status
```

### v5 — Quantitative Research
```
POST /api/v5/quant   Full 17-metric report (IC, Alpha, Monte Carlo, transitions...)
```

### v7 — AI Intelligence
```
POST /api/v7/ai/explain              AI market narrative + XAI attributions
POST /api/v7/portfolio/optimize      Markowitz mean-variance optimization
GET  /api/v7/market/intelligence     Sector heatmap + sentiment
GET  /api/v7/alerts                  List user alerts
POST /api/v7/alerts                  Create new alert rule
POST /api/v7/auth/login              Get session token
```

### Monitoring
```
GET  /metrics                        Prometheus scrape endpoint
```

---

## 11. Deployment Options

### Render (recommended)
`render.yaml` is pre-configured. Connect your GitHub repo to Render and it
will auto-deploy on every push.

### Railway
`railway.json` is pre-configured for automatic detection.

### Docker / Self-hosted
```bash
docker-compose up -d

# Scale workers
docker-compose up -d --scale app=2
```

### Gunicorn (production WSGI)
```bash
gunicorn app:flask_app \
  --bind 0.0.0.0:5000 \
  --workers 2 \
  --timeout 300 \
  --access-logfile -
```

> **Timeout:** Training can take several minutes. Set `--timeout 300` or higher,
> or use async training (`/api/v2/train` or `/api/v3/train`) which returns immediately.

### Prometheus + Grafana
Add to your `prometheus.yml`:
```yaml
scrape_configs:
  - job_name: stockbuddy
    static_configs:
      - targets: ['localhost:5000']
```

The `/metrics` endpoint exposes all counters, histograms, and gauges in Prometheus
text format.
