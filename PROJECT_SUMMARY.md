# StockBuddy — AI Context Document

## What This Project Is

StockBuddy is a **production-grade quantitative stock market intelligence platform** built across two engineering phases:

- **Phase 1 — Statistical Corrections:** Eliminated data leakage, lookahead bias, missing transaction costs, and incorrect Sharpe ratios.
- **Phase 2 — Production ML Pipeline:** Added disk-persistent models, model registry, async background training, two-level inference cache, auto-retraining scheduler, and config management.

### File Structure

```
StockBuddy/
├── app.py                   # Flask entry point — thin router + orchestration (~1570 lines)
├── index.html               # Frontend SPA — vanilla JS + CSS + Chart.js (~2700 lines)
│
├── core/                    # Shared utilities
│   └── config.py            # AppConfig dataclass + SB_* env-var overrides
│
├── ml/                      # ML operations package
│   ├── registry.py          # ModelRegistry — JSON-backed versioning + best-model tracking
│   ├── trainer.py           # BackgroundTrainer — async ThreadPool job queue
│   ├── inference.py         # InferenceEngine + two-level cache (memory + disk)
│   └── scheduler.py         # RetrainingScheduler — daemon staleness poller
│
├── storage/
│   └── model_store.py       # ModelStore — Keras model + scaler disk persistence
│
├── model_artifacts/         # Created at runtime
│   ├── registry.json        # Model registry (all tickers, versions, metrics)
│   ├── {TICKER}/{version}/  # Saved models + scalers + metadata per version
│   └── inference_cache/     # Disk-layer prediction cache (pickle, TTL=1h)
│
├── README.md
└── PROJECT_SUMMARY.md       # This file
```

---

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python 3, Flask, Flask-CORS |
| Data | `yfinance` (live market data), Pandas, NumPy |
| ML / Quant | scikit-learn (KMeans, MinMaxScaler, cosine_similarity), TensorFlow (LSTM, GRU, Transformer) |
| Model Persistence | TensorFlow SavedModel format (`.keras`), pickle (scalers) |
| Background Jobs | `concurrent.futures.ThreadPoolExecutor` (daemon threads) |
| Frontend | Vanilla HTML/CSS/JS (ES6+), Chart.js 4 |
| Fonts | Google Fonts — Outfit, Inter, JetBrains Mono |
| Architecture | Decoupled REST API v1 (Phase 1) + v2 (Phase 2) + SPA Dashboard |

---

## How to Run

```bash
# 1. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn yfinance tensorflow

# 2. Start the Flask API server (Phase 2 production mode)
python app.py serve

# 3. Optional: override config via environment variables
SB_EPOCHS=30 SB_MODEL_STALE_DAYS=3 python app.py serve

# 4. Open the frontend
# Open index.html in a browser (calls http://localhost:5000)
```

---

## API Endpoints

### v1 Routes (Phase 1 — backward compatible)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + cache + registry stats |
| `POST` | `/api/regime` | Full regime analysis (KMeans, risk score, backtest) |
| `POST` | `/api/predict` | Train DL models + inference (with session cache) |
| `POST` | `/api/wf_validate` | Walk-forward validation (N expanding folds) |

### v2 Routes (Phase 2 — production pipeline)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/train` | **Async** training job — returns `job_id` immediately (202) |
| `GET` | `/api/v2/jobs/{id}` | Poll training job status (queued → running → done/failed) |
| `GET` | `/api/v2/jobs` | List all jobs + active count |
| `GET` | `/api/v2/registry` | Full model registry (all tickers, all versions) |
| `GET` | `/api/v2/registry/{ticker}` | Ticker's versions, best model, latest model |
| `POST` | `/api/v2/predict` | Cached inference — **never retrains**, uses saved model |
| `DELETE` | `/api/v2/cache` | Flush inference cache (both memory + disk) |
| `GET` | `/api/v2/metrics` | System health: disk usage, jobs, cache stats, scheduler |

---

## Backend Pipeline: `POST /api/regime`

1. **Data Fetch** — Downloads OHLCV via `yfinance`
2. **Preprocessing** — Sort, deduplicate, IQR outlier filtering, winsorize tails
3. **Feature Engineering** — RSI(14), EMA(20), MACD(12,26,9), Log Returns, Day of Week
4. **Regime Classification (KMeans, k=6):**
   - 0: Trending Bull | 1: Overbought/Exhaustion | 2: Sideways/Choppy
   - 3: Recovery/Bounce | 4: Downtrend/Bear | 5: High Volatility/Stress
5. **Risk Score (0–10)** — Vol stress + RSI extremity + trend misalignment + MACD divergence
6. **Regime Stats** — Forward returns at 5d/10d/20d per regime *(Phase 1: last 20d excluded)*
7. **Scenario Matching** — Top-5 cosine-similar historical dates with forward price paths
8. **Quantitative Backtest** — Regime strategy vs Buy & Hold *(Phase 1: with Rf + tx costs)*

---

## Backend Pipeline: `POST /api/v2/train` → `POST /api/v2/predict`

### Training (async)
1. `POST /api/v2/train` enqueues a `TrainingJob` → returns `job_id` in <50ms
2. Worker thread runs: `prepare_data` → `split_and_scale_data` → `train_models` → `evaluate_and_ensemble`
3. Artifacts saved to `model_artifacts/{TICKER}/{version}/`
4. Registry updated: version promoted to `latest`; if RMSE < previous best, promoted to `best`

### Inference (cached)
1. `POST /api/v2/predict` → `InferenceEngine.predict()`
2. L1 check: in-memory dict (< 1ms if hit)
3. L2 check: disk pickle (< 5ms if hit, TTL = 1h)
4. Cache miss: load Keras model from disk + run inference (~2s, **no training**)
5. Result cached in both layers; returned to client

---

## Phase 1 — Statistical Corrections (7 bugs fixed)

| # | Bug | Fix |
|---|---|---|
| BUG-01 | Scaler fit on full dataset (data leakage) | `scaler.fit()` on training partition only |
| BUG-02 | Sequence boundary leakage across split | Context-window approach for val/test sequences |
| BUG-03 | No transaction costs in backtest | 0.10% round-trip applied on every weight change |
| BUG-04 | Single static 70/15/15 split | Walk-forward validation with N expanding folds |
| BUG-05 | Full retrain on every API call | In-memory `MODEL_CACHE` keyed by config hash |
| BUG-06 | Regime stats computed on incomplete forward windows | Last 20 rows excluded from forward-return summaries |
| BUG-07 | Sharpe computed without risk-free rate | 5% annual Rf subtracted in all Sharpe calculations |

---

## Configuration Management

All settings in `core/config.py` (`AppConfig` dataclass). Override any value via env vars:

| Env Variable | Default | Description |
|---|---|---|
| `SB_EPOCHS` | `20` | Training epochs |
| `SB_BATCH_SIZE` | `16` | Training batch size |
| `SB_MODEL_STALE_DAYS` | `7` | Days until model auto-retrains |
| `SB_INFERENCE_CACHE_TTL_S` | `3600` | Prediction cache TTL (seconds) |
| `SB_MAX_WORKER_THREADS` | `2` | Concurrent training jobs |
| `SB_RISK_FREE_RATE_ANNUAL` | `0.05` | Sharpe ratio risk-free rate |
| `SB_TRANSACTION_COST_PCT` | `0.001` | Backtest round-trip cost |
| `SB_PORT` | `5000` | Flask server port |

---

## Key Design Decisions

- **No database** — Registry and cache use JSON + pickle files; production upgrade path is Redis/Postgres
- **Backward-compatible versioning** — All v1 routes untouched; v2 routes added in parallel
- **Async-by-default training** — Training jobs never block HTTP request handling
- **Best-model semantics** — `best` pointer tracks lowest Ensemble RMSE across all versions; survives retrains
- **Atomic registry writes** — `registry.json` written via `tmp → os.replace()` (atomic on POSIX)
- **Daemon scheduler** — Auto-retraining thread exits with the main process; no cleanup required
- **Confidence intervals** — ±1.96 × std of model disagreement (ensemble spread as uncertainty proxy)
- **Regime IDs semantically ordered** — KMeans raw clusters remapped so regime 0 = most bullish

---

## Current State

- Server runs on **port 5000** (`python app.py serve`)
- Frontend calls `http://localhost:5000` — update if deploying remotely
- GPU auto-detected; mixed precision (float16) enabled if available
- Project path: `/home/dhruv/Desktop/stockbuddy/StockBuddy/`
- Git remote: `https://github.com/dhrv9023/git_project.git` (branch: `main`)
