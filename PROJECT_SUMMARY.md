# StockBuddy — AI Context Document

## What This Project Is

StockBuddy is a **production-grade quantitative stock market intelligence platform** built across three engineering phases:

- **Phase 1 — Statistical Corrections:** Eliminated data leakage, lookahead bias, missing transaction costs, and incorrect Sharpe ratios.
- **Phase 2 — Production ML Pipeline:** Added disk-persistent models, model registry, async background training, two-level inference cache, auto-retraining scheduler, and config management.
- **Phase 3 — Distributed Systems:** Added circuit breaker, rate limiter, Prometheus metrics, priority job queue with DLQ, batched inference, and LRU cache — scalable from solo use to enterprise load.

### File Structure

```
StockBuddy/
├── app.py                   # Flask entry point — v1/v2/v3 routes (~1822 lines)
├── index.html               # Frontend SPA — vanilla JS + CSS + Chart.js (~2700 lines)
│
├── core/                    # Shared utilities
│   ├── config.py            # AppConfig dataclass + SB_* env-var overrides (50+ params)
│   ├── circuit_breaker.py   # 3-state FSM circuit breaker + global registry
│   ├── rate_limiter.py      # Token bucket + sliding window rate limiter
│   └── metrics.py           # Prometheus Counter/Histogram/Gauge + 15 pre-registered metrics
│
├── ml/                      # ML operations package
│   ├── registry.py          # ModelRegistry — JSON-backed versioning + best-model tracking
│   ├── trainer.py           # BackgroundTrainer — async ThreadPool job queue
│   ├── inference.py         # InferenceEngine + LRU L1 + disk L2 cache
│   ├── scheduler.py         # RetrainingScheduler — daemon staleness poller
│   ├── queue.py             # PriorityJobQueue — min-heap + DLQ + retry backoff
│   └── batch_predictor.py   # BatchPredictor — micro-batch async inference engine
│
├── storage/
│   └── model_store.py       # ModelStore — parallel Keras model + scaler disk persistence
│
├── tests/
│   └── load_test.py         # Throughput/latency benchmark + unit microbenchmarks
│
├── model_artifacts/         # Created at runtime (gitignored)
│   ├── registry.json        # Model registry (all tickers, versions, metrics)
│   ├── {TICKER}/{version}/  # Saved models + scalers + metadata per version
│   └── inference_cache/     # Disk-layer prediction cache (pickle, TTL=1h)
│
├── PHASE_1_STATISTICAL_CORRECTIONS.md
├── PHASE_2_PRODUCTION_PIPELINE.md
├── PHASE_3_DISTRIBUTED_SCALE.md
├── StockBuddy_Portfolio_Review.docx
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
| Resilience | Circuit breaker (3-state FSM), token bucket rate limiter |
| Observability | Prometheus-compatible metrics (Counter, Histogram, Gauge, `/metrics` endpoint) |
| Job Queue | Min-heap priority queue, exponential backoff retry, dead-letter queue |
| Inference | Micro-batch `BatchPredictor`, LRU L1 cache + zlib-compressed L2 disk cache |
| Frontend | Vanilla HTML/CSS/JS (ES6+), Chart.js 4 |
| Fonts | Google Fonts — Outfit, Inter, JetBrains Mono |
| Architecture | Decoupled REST API (v1/v2/v3) + Single-Page App dashboard |

---

## How to Run

```bash
# 1. Install dependencies
pip install flask flask-cors pandas numpy scikit-learn yfinance tensorflow

# 2. Start the Flask server (all v1/v2/v3 routes)
python app.py serve

# 3. Optional: override config via environment variables
SB_EPOCHS=30 SB_MODEL_STALE_DAYS=3 SB_RATE_LIMIT_BURST=50 python app.py serve

# 4. Open the frontend
# Open index.html in a browser (calls http://localhost:5000)

# 5. Run benchmarks (no server needed)
python tests/load_test.py --unit
```

---

## API Endpoints

### v1 Routes (Phase 1 — backward compatible)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `POST` | `/api/regime` | Full regime analysis (KMeans, risk score, backtest) |
| `POST` | `/api/predict` | Train DL models + inference (with session cache) |
| `POST` | `/api/wf_validate` | Walk-forward validation (N expanding folds) |

### v2 Routes (Phase 2 — production pipeline)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v2/train` | Async training job — returns `job_id` immediately (202) |
| `GET` | `/api/v2/jobs/{id}` | Poll job status |
| `GET` | `/api/v2/jobs` | List all jobs |
| `GET` | `/api/v2/registry` | Full model registry |
| `GET` | `/api/v2/registry/{ticker}` | Ticker's versions, best/latest |
| `POST` | `/api/v2/predict` | Cached inference — uses saved model |
| `DELETE` | `/api/v2/cache` | Flush inference cache |
| `GET` | `/api/v2/metrics` | System health snapshot |

### v3 Routes (Phase 3 — distributed systems)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/v3/train` | Priority queue training with retry + DLQ |
| `POST` | `/api/v3/predict` | Metered + circuit-broken inference |
| `GET` | `/api/v3/queue` | Queue depth, all jobs, DLQ contents |
| `POST` | `/api/v3/queue/dlq/{id}/requeue` | Manual DLQ retry |
| `GET` | `/api/v3/metrics` | Prometheus text or JSON telemetry |
| `GET` | `/api/v3/breakers` | All circuit breaker states |
| `POST` | `/api/v3/breakers/{name}/reset` | Operator circuit reset |
| `GET` | `/api/v3/rate-limiter` | Per-IP token bucket status |
| `GET` | `/metrics` | Standard Prometheus scrape endpoint |

---

## Phase 1 — Statistical Corrections (7 bugs fixed)

| # | Bug | Fix |
|---|---|---|
| BUG-01 | Scaler fit on full dataset (data leakage) | `scaler.fit()` on training partition only |
| BUG-02 | Sequence boundary leakage across split | Context-window approach for val/test |
| BUG-03 | No transaction costs in backtest | 0.10% round-trip applied on every weight change |
| BUG-04 | Single static 70/15/15 split | Walk-forward validation with N expanding folds |
| BUG-05 | Full retrain on every API call | In-memory `MODEL_CACHE` keyed by config hash |
| BUG-06 | Regime stats computed on incomplete forward windows | Last 20 rows excluded |
| BUG-07 | Sharpe computed without risk-free rate | 5% annual Rf subtracted |

---

## Phase 3 — 10 Bottlenecks Fixed

| # | Bottleneck | Before | After |
|---|---|---|---|
| B1 | yfinance failure cascade | Server crash | Circuit breaker (3-state FSM) |
| B2 | Feature recompute | O(N×M) per call | O(ΔN×M) incremental + parallel fetch |
| B3 | FIFO job queue | O(1) no priority | O(log N) min-heap + 4 priority levels |
| B4 | Registry full JSON parse | O(N) every lookup | O(1) incremental counters |
| B5 | Sequential model I/O | O(K) serial | O(1) parallel + SHA256 integrity |
| B6 | Batch size = 1 inference | N × T | N/B × T, up to 16× faster |
| B7 | Unbounded memory cache | OOM under load | O(1) LRU + zlib disk compression |
| B8 | No circuit breaker | Failure storms | 1,073 ns/call overhead |
| B9 | No rate limiting | Flood attacks | Token bucket + sliding window |
| B10 | No observability | Flying blind | 15 metrics + Prometheus `/metrics` |

### Benchmark Results (unit, no server required)
```bash
python tests/load_test.py --unit
```
```
circuit_breaker.call():       1,073 ns/call  (930K RPS)
TokenBucketLimiter.allow():     582 ns/call  (1.7M RPS)
Counter.inc():                  717 ns/call  (1.4M RPS)
```

---

## Configuration Management

All settings in `core/config.py`. Override any value via `SB_*` env vars:

| Env Variable | Default | Description |
|---|---|---|
| `SB_EPOCHS` | `20` | Training epochs |
| `SB_BATCH_SIZE` | `16` | Training batch size |
| `SB_MODEL_STALE_DAYS` | `7` | Days until auto-retrain |
| `SB_INFERENCE_CACHE_TTL_S` | `3600` | Prediction cache TTL |
| `SB_MAX_WORKER_THREADS` | `2` | Concurrent training jobs |
| `SB_FETCH_PARALLELISM` | `4` | Parallel yfinance fetch threads |
| `SB_CB_FAILURE_THRESHOLD` | `0.5` | Circuit breaker failure rate |
| `SB_CB_RESET_TIMEOUT_S` | `60` | Seconds before half-open probe |
| `SB_RATE_LIMIT_BURST` | `20` | Token bucket capacity |
| `SB_RATE_LIMIT_RATE` | `5.0` | Tokens per second refill |
| `SB_BATCH_PREDICTOR_MAX_BATCH` | `32` | Sequences per forward pass |
| `SB_JOB_MAX_RETRIES` | `3` | DLQ retry attempts |
| `SB_CACHE_MAX_MEMORY_ENTRIES` | `100` | LRU cache max size |
| `SB_RISK_FREE_RATE_ANNUAL` | `0.05` | Sharpe ratio Rf |
| `SB_TRANSACTION_COST_PCT` | `0.001` | Backtest round-trip cost |
| `SB_PORT` | `5000` | Flask server port |

---

## Current State

- Server runs on **port 5000** (`python app.py serve`)
- Frontend calls `http://localhost:5000` — update if deploying remotely
- GPU auto-detected; mixed precision (float16) enabled if available
- Project path: `/home/dhruv/Desktop/stockbuddy/StockBuddy/`
- Git remote: `https://github.com/dhrv9023/git_project.git` (branch: `main`)
- All 3 phases pushed and live on GitHub
