# StockBuddy — Engineering Design Document
**Version:** 2.0  **Date:** 2026-08-10  **Author:** Senior Engineering Review

---

## 1. Executive Summary

StockBuddy is a quantitative market intelligence platform that combines deep-learning price prediction (LSTM/GRU/Transformer ensemble), unsupervised regime classification (K-Means), and Markowitz portfolio optimisation into a REST API served by Flask + Gunicorn.

**Current state:** ~55% of production engineering practices are implemented. Infrastructure (Docker, Prometheus, circuit breakers) is solid. Application architecture (SOLID, service layer, repository pattern, DI, testing) is largely absent. The primary artefact `app.py` is 2,104 lines — a God Object with circular `sys.modules` coupling that makes the codebase untestable.

**Goal of this refactor:** Decompose `app.py` into a clean layered architecture without changing any external API contracts.

---

## 2. Current Architecture Problems

| Problem | Impact |
|---|---|
| `app.py` is 2,104 lines — routes + ML + backtest + utilities all in one file | Untestable, violates SRP |
| `trainer.py` & `inference.py` call `sys.modules.get("__main__").prepare_data` | Circular coupling, untestable |
| Old `CONFIG = {}` dict coexists with `CFG` dataclass | Dual source of truth |
| Route handlers contain inline business logic | Violates SRP/DI |
| No service classes (`PredictionService`, `RegimeService`, etc.) | No service layer |
| No repository abstraction over data access | No repository pattern |
| Bare `except Exception` everywhere | Poor observability |
| No domain exception hierarchy | Error handling is opaque |
| `requirements.txt` uses `>=` ranges, no lockfile | Non-reproducible builds |
| No `ruff`/`black`/`mypy`/`pre-commit` | No code quality gate |
| 3 shallow test files, zero mocks | ~5% coverage |

---

## 3. Target Architecture

### 3.1 Folder Structure

```
StockBuddy/
├── app/                          # ← NEW: application package
│   ├── __init__.py               # create_app() factory
│   ├── api/                      # ← NEW: route blueprints only
│   │   ├── __init__.py
│   │   ├── v1_routes.py          # /api/regime, /api/predict, /api/wf_validate
│   │   ├── v2_routes.py          # /api/v2/*
│   │   ├── v3_routes.py          # /api/v3/*
│   │   ├── v5_routes.py          # /api/v5/*
│   │   ├── v7_routes.py          # /api/v7/*
│   │   └── health_routes.py      # /health, /ready, /metrics
│   ├── services/                 # ← NEW: business logic (service layer)
│   │   ├── __init__.py
│   │   ├── prediction_service.py # Orchestrates data → features → model → result
│   │   ├── regime_service.py     # Orchestrates data → regime classification
│   │   ├── backtest_service.py   # Runs backtests, computes Sharpe etc.
│   │   ├── training_service.py   # Wraps BackgroundTrainer
│   │   └── portfolio_service.py  # Markowitz optimisation
│   ├── repositories/             # ← NEW: data access layer
│   │   ├── __init__.py
│   │   ├── market_data_repo.py   # yfinance fetching + preprocessing
│   │   └── model_artifact_repo.py # load/save model artifacts
│   ├── domain/                   # ← NEW: domain models & exceptions
│   │   ├── __init__.py
│   │   ├── exceptions.py         # StockBuddyError hierarchy
│   │   └── models.py             # PredictionResult, RegimeResult dataclasses
│   └── middleware/               # ← NEW: Flask middleware
│       ├── __init__.py
│       ├── error_handlers.py     # Global exception → JSON response
│       ├── security.py           # Security headers
│       └── observability.py      # before/after_request metrics
│
├── core/                         # (existing — keep)
│   ├── config.py                 # AppConfig dataclass (MODIFY: remove CONFIG dict)
│   ├── circuit_breaker.py        # (keep)
│   ├── metrics.py                # (keep)
│   └── rate_limiter.py           # (keep)
│
├── ml/                           # (existing — refactor internals)
│   ├── features.py               # ← NEW: RSI, EMA, MACD, engineer_features()
│   ├── models.py                 # ← NEW: build_lstm, build_gru, build_transformer
│   ├── trainer.py                # (MODIFY: remove sys.modules coupling)
│   ├── inference.py              # (MODIFY: accept injected services)
│   ├── registry.py               # (keep)
│   ├── scheduler.py              # (keep)
│   ├── queue.py                  # (keep)
│   ├── ai_intelligence.py        # (keep)
│   ├── quant_analytics.py        # (keep)
│   └── batch_predictor.py        # (keep)
│
├── storage/
│   └── model_store.py            # (keep)
│
├── tests/
│   ├── conftest.py               # ← NEW: pytest fixtures, DI wiring
│   ├── unit/                     # ← NEW
│   │   ├── test_features.py      # RSI, MACD, EMA unit tests
│   │   ├── test_backtest.py      # Sharpe, transaction costs
│   │   ├── test_regime.py        # classify_regimes, risk_score
│   │   ├── test_config.py        # AppConfig env overrides
│   │   └── test_exceptions.py    # Domain exception hierarchy
│   ├── integration/              # ← NEW
│   │   ├── test_prediction_service.py
│   │   ├── test_regime_service.py
│   │   └── test_api_contracts.py # Flask test client
│   └── load_test.py              # (keep)
│
├── docs/
│   └── ENGINEERING_DESIGN_DOCUMENT.md  # this file
│
├── .github/workflows/
│   └── ci-cd.yml                 # (MODIFY: add ruff, mypy, coverage)
│
├── pyproject.toml                # ← NEW: replaces requirements.txt for tooling config
├── requirements.txt              # (MODIFY: pin exact versions)
├── requirements-dev.txt          # ← NEW: dev/test dependencies
├── .pre-commit-config.yaml       # ← NEW: ruff, black, mypy hooks
├── Dockerfile                    # (keep — already good)
├── docker-compose.yml            # (MODIFY: add dev profile)
└── app.py                        # ← REPLACED by app/__init__.py
```

### 3.2 Layered Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    HTTP / WSGI Layer                    │
│         Flask Blueprints (app/api/*.py)                 │
│  Routes only: parse request → call service → jsonify   │
└───────────────────────────┬─────────────────────────────┘
                            │ calls
┌───────────────────────────▼─────────────────────────────┐
│                    Service Layer                        │
│         app/services/*.py                               │
│  Business logic, orchestration, error translation       │
│  PredictionService | RegimeService | BacktestService    │
└──────────────┬─────────────────────────┬────────────────┘
               │ calls                   │ calls
┌──────────────▼──────────┐  ┌──────────▼───────────────┐
│   Repository Layer      │  │      ML Pipeline Layer    │
│   app/repositories/     │  │      ml/*.py              │
│   MarketDataRepo        │  │      ModelRegistry        │
│   ModelArtifactRepo     │  │      InferenceEngine      │
└──────────────┬──────────┘  └──────────┬───────────────┘
               │ uses                    │ uses
┌──────────────▼─────────────────────────▼───────────────┐
│                  Infrastructure Layer                   │
│   core/ (config, circuit_breaker, metrics, rate_limiter)│
│   storage/ (model_store)                                │
│   External: yfinance, TensorFlow, sklearn               │
└─────────────────────────────────────────────────────────┘
```

### 3.3 Dependency Injection Strategy

Python has no IoC container by default. We use **constructor injection** — every service receives its dependencies via `__init__`. The `create_app()` factory in `app/__init__.py` is the **composition root** — the single place that wires everything together.

```python
# app/__init__.py  (composition root)
def create_app(cfg: AppConfig = None) -> Flask:
    cfg = cfg or AppConfig.from_env()

    # Infrastructure
    store    = ModelStore(cfg.model_artifacts_dir)
    registry = ModelRegistry(cfg.registry_path)
    cache    = InferenceCache(cfg.cache_dir, cfg.inference_cache_ttl_s)

    # Repositories
    market_repo   = MarketDataRepository()
    artifact_repo = ModelArtifactRepository(store)

    # Services  ← receive dependencies, never import app.py globals
    prediction_svc = PredictionService(registry, artifact_repo, cache, market_repo, cfg)
    regime_svc     = RegimeService(market_repo, cfg)
    backtest_svc   = BacktestService(cfg)
    training_svc   = TrainingService(registry, store, cfg)

    # Flask app
    flask_app = Flask(__name__)
    register_blueprints(flask_app, prediction_svc, regime_svc, backtest_svc, training_svc)
    register_middleware(flask_app, cfg)
    return flask_app
```

**Why this matters:** `trainer.py` currently uses `sys.modules.get("__main__").prepare_data` — a pattern that is impossible to unit-test and breaks under `pytest`. With DI, `TrainingService` receives a `MarketDataRepository` instance and calls `market_repo.fetch(ticker)` — fully mockable.

---

## 4. Class Diagrams

### 4.1 Service Layer

```
┌─────────────────────────────┐
│      PredictionService      │
├─────────────────────────────┤
│ - registry: ModelRegistry   │
│ - artifact_repo: IArtifact  │
│ - cache: InferenceCache     │
│ - market_repo: IMarketData  │
│ - cfg: AppConfig            │
├─────────────────────────────┤
│ + predict(ticker,start,end) │
│   → PredictionResult        │
│ + walk_forward_validate(…)  │
│   → WalkForwardResult       │
└─────────────────────────────┘

┌─────────────────────────────┐
│       RegimeService         │
├─────────────────────────────┤
│ - market_repo: IMarketData  │
│ - cfg: AppConfig            │
├─────────────────────────────┤
│ + classify(ticker,start,end)│
│   → RegimeResult            │
└─────────────────────────────┘

┌─────────────────────────────┐
│      BacktestService        │
├─────────────────────────────┤
│ - cfg: AppConfig            │
├─────────────────────────────┤
│ + run(pred,actual,dates,cap)│
│   → BacktestResult          │
└─────────────────────────────┘

┌─────────────────────────────┐
│      TrainingService        │
├─────────────────────────────┤
│ - trainer: BackgroundTrainer│
│ - registry: ModelRegistry   │
├─────────────────────────────┤
│ + submit_job(ticker,…)→str  │
│ + get_job(job_id)           │
│ + list_jobs()               │
└─────────────────────────────┘
```

### 4.2 Repository Interfaces

```
«interface»
IMarketDataRepository
  + fetch(ticker, start, end) → pd.DataFrame
  + preprocess(df) → pd.DataFrame
  + engineer_features(df) → pd.DataFrame

«interface»
IModelArtifactRepository
  + load(ticker, version) → ArtifactBundle | None
  + save(ticker, version, bundle) → str
  + delete(ticker, version)

MarketDataRepository implements IMarketDataRepository
  # wraps yfinance + preprocess_data + engineer_features

ModelArtifactRepository implements IModelArtifactRepository
  # wraps storage.ModelStore
```

### 4.3 Domain Exception Hierarchy

```
StockBuddyError(Exception)
├── DataFetchError          # yfinance failure, empty data
├── InsufficientDataError   # too few rows for sequences
├── ModelNotFoundError      # no ready model in registry
├── TrainingError           # model.fit() failure
├── ConfigurationError      # bad env var, missing key
└── CacheError              # disk I/O failure
```

---

## 5. API Flow

### 5.1 POST /api/v2/train

```
Client → POST /api/v2/train {ticker, epochs}
         │
         ▼
   v2_routes.py (Blueprint)
   parse + validate request
         │
         ▼
   TrainingService.submit_job(ticker, …)
         │
         ▼
   BackgroundTrainer.submit(…) → job_id (non-blocking)
         │
         ▼
   ThreadPoolExecutor worker thread:
     MarketDataRepository.fetch(ticker)
     → engineer_features()
     → split_and_scale_data()
     → build_lstm/gru/transformer()
     → model.fit()
     → evaluate_and_ensemble()
     → ModelArtifactRepository.save()
     → ModelRegistry.update_status("ready")
         │
         ▼
   Client ← 202 {job_id, status: "queued"}

Client → GET /api/v2/jobs/{job_id}
         │
         ▼
   TrainingService.get_job(job_id) → TrainingJob.to_dict()
   Client ← 200 {status: "done", version, metrics}
```

### 5.2 POST /api/regime

```
Client → POST /api/regime {ticker, start_date, end_date}
         │
         ▼
   v1_routes.py (Blueprint)
         │
         ▼
   RegimeService.classify(ticker, start, end)
     MarketDataRepository.fetch()   [circuit-breaker protected]
     → preprocess_data()
     → engineer_features()
     → classify_regimes()           [KMeans]
     → compute_risk_score()
     → compute_regime_stats()
     → find_similar_scenarios()
     → compute_quant_backtest()
     → RegimeResult dataclass
         │
         ▼
   v1_routes.py: jsonify(result.to_dict())
   Client ← 200 {current_regime, risk_score, timeline, …}
```

---

## 6. Logging Pipeline

```
Application Code
      │  log.info("…") / log.error("…", exc_info=True)
      ▼
Python logging.Logger (structured)
      │
      ▼
JSONFormatter (production) / TextFormatter (dev)
      │
   ┌──┴──────────────────────┐
   ▼                         ▼
stdout (container)      Sentry SDK (errors only)
   │                         │
   ▼                         ▼
Docker log driver       Sentry.io dashboard
   │
   ▼
CloudWatch / Datadog / Loki (infrastructure)
```

**Log levels:**
- `DEBUG` — cache hit/miss, individual predictions (dev only)
- `INFO` — job submitted/done, model version promoted, server start
- `WARNING` — circuit breaker state change, cache write error
- `ERROR` — training failed, data fetch failed (with full stack trace)
- `CRITICAL` — startup failure (registry unreachable, etc.)

---

## 7. Testing Strategy

### 7.1 Test Pyramid

```
        ┌──────────┐
        │  E2E/Load │  5%  (load_test.py — Locust)
        └──────────┘
      ┌────────────────┐
      │  Integration   │  25%  (Flask test client, real services, mocked yfinance)
      └────────────────┘
    ┌────────────────────────┐
    │       Unit Tests        │  70%  (pure functions, mocked dependencies)
    └────────────────────────┘
```

### 7.2 Unit Test Coverage Targets

| Module | Key Tests |
|---|---|
| `ml/features.py` | RSI window=14 known values; MACD signal crossover; EMA span alignment |
| `ml/models.py` | build_lstm output shape; build_gru trainable params |
| `app/services/backtest_service.py` | Sharpe with Rf=5%; transaction cost deduction; MDD calculation |
| `app/services/regime_service.py` | classify_regimes returns 6 labels; risk_score 0–10 bounds |
| `app/domain/exceptions.py` | Exception hierarchy `isinstance` checks |
| `core/config.py` | Env var overrides; bool parsing; int/float casting |

### 7.3 Integration Tests

| Test | Strategy |
|---|---|
| `POST /api/regime` | Mock `MarketDataRepository.fetch()` → synthetic DataFrame; assert response keys |
| `POST /api/v2/train` | Submit job; assert 202; poll until done with real tiny dataset |
| `GET /health` | Assert 200 + `status: healthy` |
| `GET /ready` | Assert 200 when storage writable |
| Rate limiter | Send 25 POSTs from same IP; assert 429 on 21st |

### 7.4 Pytest Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --tb=short --cov=app --cov=ml --cov=core --cov-report=term-missing --cov-fail-under=70"
markers = [
    "unit: fast, no I/O",
    "integration: uses Flask test client",
    "slow: trains models"
]
```

---

## 8. CI Pipeline

```yaml
# Triggered on: push to main/develop, all PRs

Pipeline:
  1. lint-and-format       (ruff check, black --check)
  2. type-check            (mypy --strict app/ ml/ core/)
  3. unit-tests            (pytest tests/unit/ --cov-fail-under=70)
  4. integration-tests     (pytest tests/integration/)
  5. docker-build          (docker build --target runner .)
  6. container-healthcheck (docker run + curl /health + /ready)
  7. deploy                (Render webhook — main branch only)
```

**New additions vs current:**
- Steps 1, 2 are new (ruff + mypy)
- Step 3 split from step 4 (unit vs integration)
- Coverage gate (70%) is new

---

## 9. Deployment Workflow

```
Developer → git push → GitHub
                │
                ▼
         GitHub Actions CI (steps 1–6)
                │
        ┌───────┴───────┐
        │ PR branch      │ main branch
        │ stops at step 6│     │
        └───────────────┘     ▼
                         Step 7: Render deploy hook
                              │
                              ▼
                    Render pulls image
                    docker build (multi-stage)
                    replaces running container
                    /health probe must pass
                    traffic routed to new container
```

### 9.1 Docker Compose (dev profile)

```yaml
# docker-compose.yml additions
profiles:
  dev:
    app:
      environment:
        - ENVIRONMENT=development
        - LOG_FORMAT=text
        - DEBUG=true
      volumes:
        - .:/app          # live-reload mount
```

### 9.2 Environment Matrix

| Variable | development | staging | production |
|---|---|---|---|
| `ENVIRONMENT` | development | staging | production |
| `LOG_FORMAT` | text | json | json |
| `DEBUG` | true | false | false |
| `SB_EPOCHS` | 5 | 20 | 20 |
| `SENTRY_DSN` | — | optional | required |
| `HSTS header` | off | off | on |

---

## 10. Implementation Phases

### Phase A — Foundation (no behaviour change)
1. Extract `ml/features.py` (RSI, EMA, MACD, engineer_features)
2. Extract `ml/models.py` (build_lstm, build_gru, build_transformer)
3. Remove `sys.modules` coupling from `trainer.py` and `inference.py`
4. Remove duplicate `CONFIG` dict from `app.py`

### Phase B — Domain + Repositories
5. Create `app/domain/exceptions.py` — exception hierarchy
6. Create `app/domain/models.py` — result dataclasses
7. Create `app/repositories/market_data_repo.py`
8. Create `app/repositories/model_artifact_repo.py`

### Phase C — Service Layer
9. Create `app/services/prediction_service.py`
10. Create `app/services/regime_service.py`
11. Create `app/services/backtest_service.py`
12. Create `app/services/training_service.py`

### Phase D — Route Blueprints
13. Create `app/api/v1_routes.py` through `v7_routes.py`
14. Create `app/api/health_routes.py`
15. Create `app/middleware/` (error_handlers, security, observability)
16. Create `app/__init__.py` (composition root — `create_app()`)

### Phase E — Testing
17. `tests/conftest.py` — fixtures, mocks
18. `tests/unit/` — 6 test modules
19. `tests/integration/` — 3 test modules

### Phase F — Code Quality & CI
20. `pyproject.toml` — ruff, black, mypy, pytest config
21. `requirements-dev.txt` — dev dependencies
22. `.pre-commit-config.yaml`
23. Update `.github/workflows/ci-cd.yml`

---

## 11. Implementation Status Tracker

| Phase | Item | Status |
|---|---|---|
| A | Extract `ml/features.py` | ⬜ Pending |
| A | Extract `ml/models.py` | ⬜ Pending |
| A | Remove `sys.modules` coupling | ⬜ Pending |
| A | Remove duplicate `CONFIG` dict | ⬜ Pending |
| B | `app/domain/exceptions.py` | ⬜ Pending |
| B | `app/domain/models.py` | ⬜ Pending |
| B | `app/repositories/market_data_repo.py` | ⬜ Pending |
| B | `app/repositories/model_artifact_repo.py` | ⬜ Pending |
| C | `app/services/prediction_service.py` | ⬜ Pending |
| C | `app/services/regime_service.py` | ⬜ Pending |
| C | `app/services/backtest_service.py` | ⬜ Pending |
| C | `app/services/training_service.py` | ⬜ Pending |
| D | All route blueprints | ⬜ Pending |
| D | Middleware | ⬜ Pending |
| D | `app/__init__.py` (composition root) | ⬜ Pending |
| E | `tests/conftest.py` | ⬜ Pending |
| E | `tests/unit/` (6 modules) | ⬜ Pending |
| E | `tests/integration/` (3 modules) | ⬜ Pending |
| F | `pyproject.toml` | ⬜ Pending |
| F | `.pre-commit-config.yaml` | ⬜ Pending |
| F | Updated CI/CD workflow | ⬜ Pending |
