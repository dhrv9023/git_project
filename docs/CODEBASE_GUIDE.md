# StockBuddy — Codebase Guide

> **Who is this for?** Any developer reading the source code for the first time. This document walks you through every module, why it exists, and how it connects to everything else.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Map](#2-repository-map)
3. [Architecture in One Diagram](#3-architecture-in-one-diagram)
4. [Core Modules (`core/`)](#4-core-modules-core)
5. [ML Modules (`ml/`)](#5-ml-modules-ml)
6. [App Layer (`app/`)](#6-app-layer-app)
7. [Storage Layer (`storage/`)](#7-storage-layer-storage)
8. [The Monolith Entry Point (`app.py`)](#8-the-monolith-entry-point-apppy)
9. [Front-end (`index.html`)](#9-front-end-indexhtml)
10. [Configuration Reference](#10-configuration-reference)
11. [Phase History and Bug Fixes](#11-phase-history-and-bug-fixes)
12. [Data Flow: End-to-End Request](#12-data-flow-end-to-end-request)
13. [Known Limitations](#13-known-limitations)

---

## 1. Project Overview

StockBuddy is a quantitative market intelligence engine. It:

- Fetches OHLCV stock data from **yfinance**
- Engineers 12 technical features (RSI, EMA, MACD, LogReturn, etc.)
- Trains three deep learning models (**LSTM, GRU, Transformer**) on historical log-returns
- Ensemble-averages predictions and reconstructs next-day price estimates
- Classifies market into 6 **regimes** via K-Means clustering
- Runs a **regime-allocation backtest** against buy-and-hold
- Exposes all of this via a **Flask REST API** (v1 through v7 versioned routes)
- Serves a **single-page dashboard** (index.html)

The project went through 7 development phases:

| Phase | Feature Added |
|-------|--------------|
| 1 | Statistical correctness: data leakage fixes, proper Sharpe, transaction costs, walk-forward validation |
| 2 | Production ML pipeline: model registry, background training, disk persistence, inference cache, scheduler |
| 3 | Distributed systems: circuit breaker, rate limiter, priority job queue, Prometheus metrics |
| 4 | UI/UX: the index.html dashboard |
| 5 | Quant research platform: 17 institutional-grade metrics |
| 6 | Deployment hardening: JSON logging, Sentry, security headers, health/ready probes |
| 7 | AI platform: LLM narrative synthesis, XAI attributions, portfolio optimizer, market sentiment |

---

## 2. Repository Map

```
StockBuddy/
├── app.py                    # Legacy monolith entry point (Gunicorn: app:flask_app)
├── index.html                # Single-file SPA dashboard (94 KB)
│
├── core/                     # Framework-agnostic infrastructure
│   ├── config.py             # AppConfig dataclass — every runtime setting
│   ├── metrics.py            # Prometheus-compatible counters/histograms/gauges
│   ├── circuit_breaker.py    # Three-state circuit breaker
│   └── rate_limiter.py       # Token bucket + sliding window rate limiter
│
├── ml/                       # Machine learning pipeline
│   ├── features.py           # RSI, EMA, MACD, LogReturn + split_and_scale_data
│   ├── models.py             # build_lstm / build_gru / build_transformer factories
│   ├── trainer.py            # BackgroundTrainer — async ThreadPool training
│   ├── registry.py           # ModelRegistry — JSON-backed model versioning
│   ├── inference.py          # InferenceCache (L1 memory + L2 disk) + InferenceEngine
│   ├── scheduler.py          # RetrainingScheduler — daemon thread for stale model refresh
│   ├── queue.py              # PriorityJobQueue — heapq + DLQ + retry policy
│   ├── batch_predictor.py    # BatchPredictor — micro-batch inference
│   ├── quant_analytics.py    # 17 institutional metrics (IC, Sharpe, Monte Carlo, ...)
│   ├── ai_intelligence.py    # LLM narrative, XAI, portfolio optimization
│   └── sentiment.py          # VADER-based news sentiment engine
│
├── app/                      # Modular Flask application (blueprint architecture)
│   ├── __init__.py           # create_app() — composition root
│   ├── api/                  # Route blueprints
│   │   ├── v1_routes.py      # /api/regime, /api/predict, /api/wf_validate
│   │   ├── v2_routes.py      # /api/v2/train, /api/v2/predict, /api/v2/registry
│   │   ├── v3_routes.py      # /api/v3/metrics, /api/v3/queue, /api/v3/breakers
│   │   ├── v5_routes.py      # /api/v5/quant
│   │   ├── v7_routes.py      # /api/v7/ai/explain, /api/v7/portfolio
│   │   ├── health_routes.py  # /health, /ready
│   │   ├── auth_routes.py    # /api/v7/auth/login + JWT setup
│   │   └── workspace_routes.py  # /api/workspace/*
│   ├── domain/
│   │   ├── models.py         # Pure data classes
│   │   └── exceptions.py     # DataFetchError, InsufficientDataError, etc.
│   ├── repositories/
│   │   └── market_data_repo.py  # yfinance -> cleaned DataFrame
│   ├── services/
│   │   ├── backtest_service.py   # Signal and regime backtests + metrics
│   │   ├── regime_service.py     # K-Means regime classification + risk scoring
│   │   ├── comparison_service.py # Multi-ticker comparison
│   │   └── training_service.py   # Wrapper around BackgroundTrainer
│   ├── auth/
│   │   └── auth_service.py   # JWT + bcrypt user management
│   └── middleware/
│       ├── error_handlers.py # Flask error -> JSON response
│       └── observability.py  # Request timing + rate limit middleware
│
├── storage/
│   ├── model_store.py        # Save/load Keras models + scalers + metadata
│   └── workspace_store.py    # Persist user-saved analyses
│
├── tests/
│   ├── unit/                 # Fast pure-Python tests (no network, no GPU)
│   ├── integration/          # Flask test client — all endpoints
│   └── test_ai_intelligence.py
│
├── docs/
│   ├── ENGINEERING_DESIGN_DOCUMENT.md
│   ├── PRODUCT_REQUIREMENTS_DOCUMENT.md
│   ├── CODEBASE_GUIDE.md         <- this file
│   └── PLATFORM_USER_GUIDE.md    <- how to use the platform
│
├── model_artifacts/          # Runtime: Keras models + registry.json
├── requirements.txt
├── pyproject.toml            # Project metadata + pytest config
├── Dockerfile
└── docker-compose.yml
```

---

## 3. Architecture in One Diagram

```
Browser (index.html)
        | REST/JSON
        v
+------------------------------------------------------------------+
|  Flask  (app.py or app/__init__.py create_app)                   |
|  +------------------------------------------------------------+   |
|  |  Middleware                                                |   |
|  |  - Rate limiter (token bucket + sliding window)           |   |
|  |  - Prometheus metrics (before/after_request hooks)        |   |
|  |  - Security headers (HSTS, CSP, X-Frame-Options)          |   |
|  +------------------------------------------------------------+   |
|  +------------------------------------------------------------+   |
|  |  Route Blueprints                                          |   |
|  |  v1: regime / predict / wf_validate                       |   |
|  |  v2: train / registry / predict (cached)                  |   |
|  |  v3: metrics / queue / breakers                           |   |
|  |  v5: quant research report                                |   |
|  |  v7: AI explain / portfolio / market intelligence         |   |
|  +------------------------------------------------------------+   |
|                           |                                       |
|         +-----------------+-----------------+                     |
|         v                 v                 v                     |
|   RegimeService    InferenceEngine   TrainingService              |
|         |                |                  |                     |
|         v                |                  v                     |
|  MarketDataRepo    InferenceCache    BackgroundTrainer            |
|    (yfinance)      (L1 mem + L2 disk) (ThreadPoolExecutor)       |
|         |                |                  |                     |
|  CircuitBreaker           |            ModelStore                 |
|  (yfinance guard)         |          (disk artifacts)            |
|                           |                  |                    |
|                      ModelRegistry <---------+                    |
|                     (registry.json)                               |
|                                                                   |
|  RetrainingScheduler (daemon) -> polls registry, submits jobs    |
+------------------------------------------------------------------+
```

---

## 4. Core Modules (`core/`)

### `core/config.py` — AppConfig

A single `@dataclass` holding every runtime setting. `AppConfig.from_env()` reads
environment variables at startup. The global singleton `CFG` is imported everywhere.

Key settings:

| Setting | Default | Meaning |
|---------|---------|---------|
| `sequence_length` | 90 | LSTM lookback window (days) |
| `train_split` | 0.70 | Training data fraction |
| `risk_free_rate_annual` | 0.05 | 5% T-bill rate for Sharpe |
| `transaction_cost_pct` | 0.001 | 0.1% round-trip cost per trade |
| `model_stale_days` | 7 | Days before a model triggers auto-retrain |
| `inference_cache_ttl_s` | 3600 | Prediction cache TTL (1 hour) |

Environment override convention: `SB_{FIELD_NAME_UPPER}=value`
Standard overrides: `PORT`, `HOST`, `ENVIRONMENT`, `SENTRY_DSN`, `LOG_FORMAT`.

---

### `core/metrics.py` — Prometheus Metrics

A zero-dependency Prometheus-compatible metrics collector.

Types: `Counter` (monotonically increasing), `Histogram` (latency distribution),
`Gauge` (current value). The module-level `REGISTRY` singleton collects them all.

Pre-registered metrics:

| Metric | Type | Purpose |
|--------|------|---------|
| `http_requests_total` | Counter | Per method/endpoint/status |
| `http_latency_seconds` | Histogram | Per endpoint |
| `training_jobs_total` | Counter | Per status |
| `inference_latency_seconds` | Histogram | Per cache layer |
| `cache_hits_total` | Counter | Per layer (memory/disk) |
| `job_queue_depth` | Gauge | Priority queue backlog |
| `model_versions_total` | Gauge | Total registry versions |

The `Timer` context manager records elapsed time into a Histogram:
```python
with Timer(inference_latency_s, cache_layer="memory"):
    result = predict(...)
```

Scrape endpoint: `GET /metrics` returns Prometheus text format.
`GET /api/v3/metrics` returns JSON with circuit breaker and queue stats.

---

### `core/circuit_breaker.py` — CircuitBreaker

A three-state machine protecting against cascading failures when yfinance is degraded.

States:
- CLOSED: normal — all calls pass through
- OPEN: failure rate exceeded threshold — all calls blocked immediately
- HALF_OPEN: after `reset_timeout_s`, one probe call is allowed

Transition logic:
```
CLOSED --(failure_rate >= threshold)--> OPEN --(timeout elapsed)--> HALF_OPEN
                                              ^-- probe fails          |
CLOSED <--------------------------- probe succeeds --------------------+
```

Key defaults: `failure_threshold=0.5`, `window_size=20`, `reset_timeout_s=60`.

Named breakers are stored globally; access via `get_breaker("yfinance")`.

---

### `core/rate_limiter.py` — RateLimiter

Composite limiter: Token Bucket (burst) + Sliding Window (strict). Both must allow
for a request to proceed.

- Token Bucket: allows bursts up to `burst_capacity=20`; refills at 5 tokens/second
- Sliding Window: strict limit of `window_max=30` requests per 60-second window

Applied in Flask `before_request` middleware to all POST/DELETE endpoints.

---

## 5. ML Modules (`ml/`)

### `ml/features.py` — Feature Engineering

Public functions:
- `compute_rsi(series, 14)` — RSI via rolling mean of gains/losses
- `compute_ema(series, 20)` — Exponential Moving Average
- `compute_macd(series)` — MACD line, signal, histogram (12/26/9 defaults)
- `engineer_features(df)` — Adds all indicators to raw OHLCV DataFrame
- `split_and_scale_data(...)` — Chronological split + scaler discipline

**CRITICAL — BUG-01 fix:**
The scaler is NEVER fit on the full dataset. Fitting on full data leaks future price
extremes into normalization space, making test metrics look artificially better.
The scaler is fit ONLY on training rows. `transform()` (not `fit_transform()`) is
applied to val and test.

**Context windows — BUG-02 fix:**
Without context, `val_seq[0]` would have no lookback rows. We prepend the last
`seq_len` rows of the preceding partition as lookback context. These rows are
already correctly scaled (no leakage).

The 12 input features (`FEATURE_COLS`):
```python
["Open", "High", "Low", "Close", "Volume",
 "RSI14", "EMA20", "MACD", "MACD_signal", "MACD_hist",
 "DayOfWeek", "LogReturn"]
```

Target variable: `LogReturn.shift(-1)` — predicting tomorrow's log-return (not price
directly) because log-returns are stationary; prices are not.

---

### `ml/models.py` — Model Builders

Three compiled Keras models via `build_model(model_type, input_shape, learning_rate)`:

| Model | Architecture |
|-------|-------------|
| LSTM | Input -> LSTM(128) -> Dropout(0.4) -> LSTM(64) -> Dropout(0.5) -> Dense(64,relu) -> Dense(1) |
| GRU | Same topology as LSTM but GRU cells — faster on CPU |
| Transformer | 2-block encoder with MultiHeadAttention(4 heads) + learnable position embeddings + GlobalAvgPool |

All use Huber loss (delta=1.0) — more robust to log-return spikes than MSE.

`make_callbacks()` produces:
- EarlyStopping(patience=5, restore_best_weights=True) on val_loss
- ReduceLROnPlateau(factor=0.5, patience=5) on val_loss
- LearningRateScheduler: 10% linear warmup + cosine decay

---

### `ml/trainer.py` — BackgroundTrainer

Decouples HTTP request lifecycle from training. POST /api/v2/train returns `job_id`
immediately. Actual training runs in a ThreadPoolExecutor(max_workers=2) worker.

Job lifecycle:
```
POST /train -> submit() -> job_id returned (status=queued)
     |-> worker thread: _run_job()
               |-> status = "running"
               |-> fetch data -> split/scale -> train all models
               |-> evaluate on test set
               |-> save artifacts to ModelStore
               |-> update ModelRegistry (status="ready")
               |-> status = "done" (or "failed" on exception)
```

Polling: `GET /api/v2/jobs/{job_id}` returns `TrainingJob.to_dict()`.

---

### `ml/registry.py` — ModelRegistry

A thread-safe, file-backed model registry. Every training run creates a version record.

Tracks:
- Which version is `latest` (most recently trained)
- Which is `best` (lowest ensemble RMSE)
- Full provenance: config, dates, metrics, timestamps

File: `model_artifacts/registry.json` (written atomically via `os.replace`).

Status lifecycle: `queued -> training -> ready -> stale -> failed`

---

### `ml/inference.py` — InferenceCache + InferenceEngine

**InferenceCache** — two-level cache:
- L1 (memory dict): sub-millisecond lookup, keyed by `pred_{ticker}_{version}_{start}_{end}`
- L2 (disk pickle): survives process restart; same key hashed to MD5 filename
- TTL: 3600 seconds (configurable)
- `DELETE /api/v2/cache` flushes both layers

**InferenceEngine.predict()** pipeline:
1. Resolve version string ("best"/"latest"/"v3") -> registry record
2. Cache lookup -> if hit, return immediately (with `from_cache=True`)
3. Load model artifacts from ModelStore
4. Rebuild feature matrix + split/scale
5. Run `model.predict()` on test partition
6. Compute ensemble + confidence intervals (`mean ± 1.96 * std`)
7. Generate 5-day autoregressive rolling forecast
8. Cache result; return

---

### `ml/scheduler.py` — RetrainingScheduler

A daemon thread that polls the registry every `scheduler_interval_s` (default: 3600s).

For each READY model older than `model_stale_days` (default: 7 days):
1. Mark the old version as `stale` in the registry
2. Evict it from the inference artifact cache
3. Submit a new training job via BackgroundTrainer

This ensures models don't drift as market conditions change.

---

### `ml/queue.py` — PriorityJobQueue

A heapq-based priority queue with retry logic and a Dead-Letter Queue (DLQ).

Features:
- Jobs submitted with `priority` (0=urgent, 5=normal, 10=low)
- `RetryPolicy(max_retries=3, base_delay_s=5, max_delay_s=300)` with exponential backoff
- Failed jobs after max retries go to DLQ
- `POST /api/v3/queue/dlq/{job_id}/requeue` for manual retry

---

### `ml/quant_analytics.py` — Quantitative Research

17 metrics computed by `compute_quant_research_report(ticker, start, end)`:

| # | Metric | Method |
|---|--------|--------|
| 1 | Information Coefficient | Spearman rank correlation (predicted vs actual returns) |
| 2 | Sharpe Ratio | `(mean_excess_return / std) * sqrt(252)` |
| 3 | Sortino Ratio | Downside deviation denominator |
| 4 | Calmar Ratio | CAGR / max_drawdown |
| 5 | CAGR | Compound Annual Growth Rate |
| 6 | Alpha (Jensen's) | OLS residual: `r = alpha + beta * r_spy` |
| 7 | Beta | OLS slope vs SPY benchmark |
| 8 | Max Drawdown | `min((equity - peak) / peak)` |
| 9 | Rolling Vol (30d) | 30-day rolling std annualised |
| 10 | Rolling Sharpe (60d) | 60-day rolling Sharpe |
| 11 | Walk-Forward Opt | IC across 5 threshold parameter settings |
| 12 | Cross-Validation | TimeSeriesSplit IC stability |
| 13 | IC t-test | Is IC statistically > 0? |
| 14 | Regime Confidence | KMeans centroid distance as confidence proxy |
| 15 | Monte Carlo | 1,000 paths x 252 days via GBM |
| 16 | Feature Importance | Permutation importance vs KMeans inertia |
| 17 | Transition Matrix | Markov P[regime_i -> regime_j] |

---

### `ml/ai_intelligence.py` — AI Platform (Phase 7)

**AIMarketSynthesizer:** Generates structured narrative market summaries from
quantitative signals (regime, RSI, volatility, return). Uses template synthesis —
no external LLM API required.

**PortfolioOptimizer:** Markowitz Mean-Variance Optimization.
Given tickers + return series:
- Minimum variance portfolio
- Maximum Sharpe portfolio
- Efficient frontier (200 random portfolios)

**MarketSentimentEngine:** Returns VADER-based sentiment scores, live sector ETF
heatmap data (XLK, XLF, XLV, XLE, XLY, XLI), and macro economic calendar.

**WorkspaceManager:** In-memory CRUD for user alert rules.

---

## 6. App Layer (`app/`)

### `app/__init__.py` — Composition Root

The production entry point. `create_app(cfg?)` wires every layer together in
dependency-inversion order (infrastructure -> repository -> service -> routes).

Blueprint registration order:
1. `health_bp` — `/health`, `/ready`
2. `auth_bp` — JWT init + auth routes
3. `v1_bp` — `/api/regime`, `/api/predict`, `/api/wf_validate`
4. `v2_bp` — training jobs, registry, cached inference
5. `v3_bp` — metrics, queue, circuit breakers
6. `workspace_bp` — saved analyses
7. `v5_bp` — quant research (optional import)
8. `v7_bp` — AI intelligence (optional import)

Module-level `flask_app = create_app()` is the Gunicorn WSGI entry point.

---

### `app/repositories/market_data_repo.py`

Single responsibility: fetch, clean, and engineer features from yfinance.

`build_feature_matrix(ticker, start, end, seq_len)` returns:
```python
{
  "X_raw": ndarray,           # (N, 12) raw feature matrix
  "y_raw": ndarray,           # (N,) raw log-return targets
  "dates_raw": DatetimeIndex,
  "base_prices_raw": ndarray, # Close at each t (for price reconstruction)
  "feature_cols": list,
  "close_feature_index": int,
}
```

All yfinance calls go through the `yfinance` circuit breaker.
Raises `DataFetchError` or `InsufficientDataError` so callers handle domain errors.

---

### `app/services/backtest_service.py`

**`run_signal_backtest(pred, actual, initial_capital)`:**
- Signal: `sign(pred[t] - pred[t-1])` — long if predicted direction is up
- BUG-07 fix: Sharpe uses excess returns over daily risk-free rate

**`run_regime_backtest(close, regimes)`:**
- 100% equity in Bull (0) / Recovery (3) regimes
- 50% equity in Sideways (2) / Overbought (1)
- 0% in Bear (4) / Stress (5)
- BUG-03 fix: transaction costs on every weight change

**`calculate_metrics(y_true, y_pred)`:** RMSE, MAE, MAPE, R2, Directional Accuracy

---

### `app/services/regime_service.py`

**`classify_regimes(df, n_clusters=6)`:**
1. Build 5-feature matrix: `[RSI_dev, MACD_hist_norm, EMA_slope, Vol20, Ret10]`
2. Fit KMeans(n_clusters=6) on standardized features
3. Order clusters by score: `2*ret10 + 1.5*rsi_dev - 3*vol`
4. Map raw IDs -> semantic IDs (0=Bull ... 5=Stress)

**`compute_risk_score(df, labels)`:** 0-10 composite from vol ratio, RSI extremity,
EMA misalignment, MACD direction. Bear/Stress regimes floored at 7.5.

**`find_similar_historical_scenarios(df, feat, labels, top_k=5)`:**
Cosine similarity between today's feature vector and all historical vectors.
Returns top 5 matches with 10-day forward price trajectories.

---

## 7. Storage Layer (`storage/`)

### `storage/model_store.py`

Saves and loads Keras model artifacts to `model_artifacts/{TICKER}/{version}/`:
```
model_artifacts/
  AAPL/
    v1/
      LSTM.keras
      GRU.keras
      Transformer.keras
      scaler_X.pkl
      scaler_y.pkl
      metadata.json
```

### `storage/workspace_store.py`

Persists user-saved analysis workspaces to `model_artifacts/user_data/{id}.json`.
Supports CRUD operations for the dashboard's "Save Analysis" feature.

---

## 8. The Monolith Entry Point (`app.py`)

`app.py` is the original 2,166-line monolith that grew through all 7 phases.
It serves as:
1. Standalone script (`python app.py` -> runs training pipeline + shows plots)
2. Flask API server (`python app.py serve`)
3. Gunicorn entry point (`app:flask_app`)

Key functions inside `app.py`:

| Function | Purpose |
|----------|---------|
| `prepare_data()` | Fetch + preprocess + engineer features |
| `split_and_scale_data()` | Chronological split + scaler discipline |
| `train_models()` | Train LSTM/GRU/Transformer with callbacks |
| `evaluate_and_ensemble()` | Predict on test set, equal-weight ensemble |
| `run_backtest()` | Signal-based strategy backtest |
| `classify_regimes()` | K-Means regime clustering |
| `compute_risk_score()` | 0-10 composite risk indicator |
| `compute_quant_backtest()` | Regime-allocation vs buy-and-hold |
| `walk_forward_validate()` | 5-fold expanding window out-of-sample validation |
| `create_app()` | Compose and return Flask app with all routes |

> **Note:** `app/` package provides a cleaner, modular version of the same logic.
> New feature work should go into `app/` modules, not `app.py`.

---

## 9. Front-end (`index.html`)

A 94 KB single-file SPA. Uses React (CDN), Chart.js, and Tailwind CSS.
All API calls go to relative paths so it works with any backend host.

Panels:
- **Dashboard:** Live quotes, regime indicator, risk score, alert badge
- **Regime Analysis:** K-Means timeline, similar historical scenarios, quant backtest chart
- **AI Predict:** Training/prediction, ensemble chart, confidence bands, 5-day forecast
- **Walk-Forward:** N-fold validation results across time
- **Quant Research:** Full 17-metric institutional report
- **AI Intelligence:** LLM narrative, XAI feature attribution, sector heatmap
- **Portfolio Optimizer:** Markowitz efficient frontier
- **Settings:** Ticker, date range, model/epoch selection

---

## 10. Configuration Reference

Full list of environment variables:

| Env Var | AppConfig Field | Default |
|---------|----------------|---------|
| `SB_EPOCHS` | `epochs` | `20` |
| `SB_SEQUENCE_LENGTH` | `sequence_length` | `90` |
| `SB_BATCH_SIZE` | `batch_size` | `16` |
| `SB_TRAIN_SPLIT` | `train_split` | `0.70` |
| `PORT` | `port` | `5000` |
| `HOST` | `host` | `0.0.0.0` |
| `ENVIRONMENT` | `environment` | `development` |
| `SENTRY_DSN` | `sentry_dsn` | `""` |
| `LOG_FORMAT` | `log_format` | `text` |
| `SB_MODEL_STALE_DAYS` | `model_stale_days` | `7` |
| `SB_RISK_FREE_RATE_ANNUAL` | `risk_free_rate_annual` | `0.05` |
| `SB_TRANSACTION_COST_PCT` | `transaction_cost_pct` | `0.001` |
| `SB_MAX_WORKER_THREADS` | `max_worker_threads` | `2` |
| `SB_INFERENCE_CACHE_TTL_S` | `inference_cache_ttl_s` | `3600` |

---

## 11. Phase History and Bug Fixes

BUG-XX comments in the codebase refer to these corrections:

| Bug ID | Issue | Fix |
|--------|-------|-----|
| BUG-01 | Scaler fit on full dataset; future price extremes leaked into training normalization | Fit scaler ONLY on training partition |
| BUG-02 | Val/test sequences had no lookback context | Context window: prepend last `seq_len` rows of preceding partition |
| BUG-03 | Backtest ignored transaction costs | `cost = tc * abs(weight[t] - weight[t-1])` on every weight change |
| BUG-04 | Single 70/15/15 split tested on only one historical period | Walk-forward expanding window validation (5 folds) |
| BUG-05 | Full model retrain on every /api/predict request | In-memory MODEL_CACHE keyed by config hash |
| BUG-06 | Forward return stats computed on dates without full forward window | Exclude last 20 dates from regime statistics |
| BUG-07 | Sharpe ratio did not subtract risk-free rate | `Sharpe = (E[r] - Rf_daily) / std(r) * sqrt(252)` |

Additional fixes applied during final audit:

| Bug | Issue | Fix |
|-----|-------|-----|
| Scheduler NameError | `scheduler` referenced before instantiation in `app.py`'s `create_app()` | Added `scheduler = RetrainingScheduler(registry, trainer, engine, CFG)` before `scheduler.start()` |
| Dead code in `main()` | `pass` statement at line 1166 made all subsequent print/training code unreachable | Removed the `pass` statement |
| Redundant isinstance | `isinstance(v, (float, int)) and isinstance(v, float)` | Simplified to `isinstance(v, float)` |

---

## 12. Data Flow: End-to-End Request

**Example: `POST /api/regime` with `{"ticker": "AAPL"}`**

```
1. Flask before_request middleware
   |-> rate_limiter.allow(client_ip) - token bucket + sliding window check
   |-> record flask.g.t_start

2. api_regime() route handler
   |-> fetch_data_yfinance("AAPL", start, end)
         |-> MarketDataRepository.fetch_raw()
               |-> yf.download() [protected by yfinance circuit breaker]
   |-> preprocess_data(raw_df)         # sort, dedup, IQR filter, winsorise
   |-> engineer_features(df)           # RSI14, EMA20, MACD, LogReturn, DayOfWeek
   |-> classify_regimes(df, n=6)       # KMeans -> semantic ordering
   |-> compute_risk_score(df, labels)  # 0-10 composite risk
   |-> compute_regime_stats(df, labels)# fwd 5/10/20d return distributions
   |-> find_similar_historical_scenarios() # cosine similarity lookup
   |-> compute_quant_backtest(df, labels)  # regime strategy vs buy-and-hold
   |-> jsonify(result)

3. Flask after_request middleware
   |-> http_latency_seconds.observe(elapsed, endpoint="/api/regime")
   |-> http_requests_total.inc(method="POST", endpoint="...", status="200")
   |-> apply security headers (CSP, X-Content-Type-Options, etc.)
```

---

## 13. Known Limitations

1. **JWT auth is a mock** — `v7_auth_login` returns a fake token string, not a real
   signed JWT. Do not use in production without replacing with `flask-jwt-extended`.

2. **yfinance rate limits** — Many concurrent requests can trigger yfinance throttling.
   The circuit breaker prevents cascading failures but not initial throttling.

3. **In-process model cache** — `MODEL_CACHE` in `app.py` is per-process. Gunicorn
   with multiple workers will have separate caches. The disk-backed InferenceCache
   survives restarts.

4. **Monte Carlo runs synchronously** — 1,000 paths x 252 days on the request thread.
   For high-traffic deployments, consider offloading to a background job.

5. **Sentiment engine** — `ml/sentiment.py` uses VADER (rule-based). It works for
   simple headlines but struggles with nuanced financial language.

6. **`app.py` vs `app/` duality** — The project has two create_app() functions.
   `app/__init__.py`'s version is the correct production one with full DI and
   blueprints. `app.py`'s version is the legacy monolith kept for backward
   compatibility.
