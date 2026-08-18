# StockBuddy — Master Project Summary & Architecture Context

## What This Project Is

StockBuddy is a **production-grade quantitative market intelligence and AI financial analytics platform** built across seven engineering phases and a P1 feature sprint:

- **Phase 1 — Statistical Corrections:** Eliminated data leakage (scalers fit on training data only), lookahead bias, missing transaction friction (0.10%), and benchmarked Sharpe against a 5% annual risk-free rate.
- **Phase 2 — Production ML Pipeline:** Disk-persistent models with SHA-256 validation, semantic model registry, async background training, two-level TTL inference cache, auto-retraining daemon scheduler, and typed AppConfig.
- **Phase 3 — High-Throughput Distributed Infrastructure:** 3-state circuit breaker FSM, IP token-bucket rate limiter, Prometheus metrics exposition (`/metrics`), priority min-heap job queue with dead-letter queue, and batched inference.
- **Phase 4 — Terminal UI/UX:** Responsive Bloomberg/TradingView-grade dark terminal interface, glassmorphism, execution pipeline workflow, keyboard shortcuts (`⌘K` / `?`), condition alerts, and live telemetry polling.
- **Phase 5 — Quantitative Research Engine:** 17 institutional quant metrics (Sharpe, Sortino, Calmar, Max Drawdown, CAGR, Alpha, Beta, Rolling Volatility, Rolling Sharpe, Walk-Forward Optimisation, Cross-Validation IC Stability, Statistical Significance, Deflated Sharpe Ratio, Probability of Backtest Overfitting, Monte Carlo 1,000 paths, Permutation Feature Importance, and 6×6 Markov Transition Matrix).
- **Phase 6 — Production Deployment & Cloud Infrastructure:** Multi-stage Dockerfile, Nginx reverse proxy with SSL/TLS termination & rate limiting, security headers (CSP, HSTS, X-Frame-Options), Render Blueprint, Railway manifest, GitHub Actions CI/CD, container health probes (`/health`, `/ready`), structured JSON logging, and Sentry error tracking.
- **Phase 7 & P1 Innovation Suite:** Deterministic XAI feature attributions, multi-stock comparative matrix, VADER financial sentiment engine, Markowitz Mean-Variance Portfolio Optimization, JWT authentication with bcrypt password hashing, and workspace layout persistence.

---

## Codebase Architecture

```
StockBuddy/
├── app/                          # Modular Application Package
│   ├── __init__.py               # create_app() composition root & DI wiring
│   ├── api/                      # REST Blueprint Controllers
│   │   ├── auth_routes.py        # /api/auth (JWT login, register, profile)
│   │   ├── health_routes.py      # /health, /ready, /metrics
│   │   ├── v1_routes.py          # /api/regime, /api/predict, /api/wf_validate
│   │   ├── v2_routes.py          # /api/v2/train, /jobs, /registry, /compare
│   │   ├── v3_routes.py          # /api/v3/train, /queue, /breakers, /metrics
│   │   ├── v5_routes.py          # /api/v5/quant (17 institutional metrics)
│   │   ├── v7_routes.py          # /api/v7/explain, /portfolio-optimize, /market-intelligence, /sentiment
│   │   └── workspace_routes.py   # /api/workspaces (layouts, watchlists, alerts)
│   ├── auth/                     # Authentication & Security
│   │   └── auth_service.py       # AuthService (bcrypt, JWT tokens, RBAC)
│   ├── domain/                   # Domain Models & Exceptions
│   │   ├── exceptions.py         # Domain exception hierarchy (StockBuddyError)
│   │   └── models.py             # Typed dataclasses (RegimeResult, etc.)
│   ├── middleware/               # HTTP Interceptors
│   │   ├── error_handlers.py     # Global error handling & JSON format
│   │   └── observability.py      # Request latency & status code instrumentation
│   ├── repositories/             # Data Access Layer
│   │   └── market_data_repo.py   # MarketDataRepository (yfinance, cleaning, features)
│   └── services/                 # Business Logic Services
│       ├── backtest_service.py   # Backtesting & risk metrics calculation
│       ├── comparison_service.py # Multi-stock normalisation & correlation
│       ├── regime_service.py     # KMeans regime classification & scoring
│       └── training_service.py   # Background training orchestration
│
├── core/                         # Shared Distributed Primitives
│   ├── circuit_breaker.py        # 3-state FSM circuit breaker + global registry
│   ├── config.py                 # AppConfig dataclass + SB_* env-var overrides
│   ├── metrics.py                # Prometheus Counter/Histogram/Gauge with RLock thread-safety
│   └── rate_limiter.py           # Token bucket + sliding window rate limiter
│
├── ml/                           # ML & Quant Package
│   ├── ai_intelligence.py        # AIMarketSynthesizer, PortfolioOptimizer, Sentiment
│   ├── batch_predictor.py        # Micro-batch async inference engine
│   ├── features.py               # Pure indicator calculations (RSI, EMA, MACD, etc.)
│   ├── inference.py              # InferenceEngine + LRU L1 + disk L2 cache
│   ├── models.py                 # Deep learning model builders (LSTM, GRU, Transformer)
│   ├── quant_analytics.py        # 17 quant metrics, Monte Carlo, DSR, PBO
│   ├── queue.py                  # PriorityJobQueue min-heap + DLQ
│   ├── registry.py               # ModelRegistry JSON versioning & metadata
│   ├── scheduler.py              # RetrainingScheduler daemon
│   ├── sentiment.py              # VADER financial sentiment analyser
│   └── trainer.py                # BackgroundTrainer async worker pool
│
├── storage/                      # Persistence Layer
│   ├── model_store.py            # ModelStore (atomic disk persistence & checksums)
│   └── workspace_store.py        # WorkspaceStore (user layouts, watchlists, alerts)
│
├── nginx/                        # Nginx Reverse Proxy & SSL
│   ├── nginx.conf                # SSL termination, rate limits, security headers
│   └── generate-ssl.sh           # Local self-signed certificate generator
│
├── tests/                        # Fast Deterministic Test Suite (130 Tests)
│   ├── conftest.py               # Global network mocking & test doubles
│   ├── integration/
│   │   └── test_api_contracts.py # API contract integration tests
│   ├── unit/
│   │   ├── test_backtest.py      # Backtest math & Sharpe testing
│   │   ├── test_config.py        # Config override tests
│   │   ├── test_exceptions.py    # Domain exception hierarchy tests
│   │   ├── test_features.py      # Zero-leakage indicator tests
│   │   └── test_p1_features.py   # Auth, Workspace, & Comparison tests
│   ├── test_ai_intelligence.py   # XAI, Portfolio Optimization, Sentiment tests
│   ├── test_deployment.py        # Health, readiness, & security header tests
│   └── load_test.py              # High-throughput benchmark script
│
├── Dockerfile                    # Multi-stage production container
├── docker-compose.yml            # Gunicorn + Nginx HTTPS stack
├── index.html                    # Institutional Financial Terminal UI
├── requirements.txt              # Production pinned dependencies
└── pyproject.toml                # Build & pytest configuration
```

---

## Technology Stack

| Layer | Technologies | Key Rationale |
|---|---|---|
| **API Framework** | Python 3.10+, Flask, Gunicorn | Lightweight, low-overhead microservice foundation |
| **Data & Vector Ops** | Pandas, NumPy, SciPy, `yfinance` | High-performance array operations and time-series manipulation |
| **Machine Learning** | scikit-learn (KMeans, GMM, PCA, StandardScaler) | Fast, deterministic unsupervised clustering & dimensionality reduction |
| **Deep Learning** | TensorFlow 2.x (LSTM, GRU, Transformer Encoder) | Multi-architecture sequence forecasting with recurrent regularisation |
| **Security & Auth** | PyJWT, bcrypt, HSTS, CSP | Industry-standard password hashing and stateless token authentication |
| **Distributed Primitives** | Custom 3-state Circuit Breaker, Token-Bucket Rate Limiter | Resilience against upstream API failures and DDoS protection |
| **Observability** | Custom Prometheus Registry (`/metrics`), Sentry, JSON Logging | Zero-dependency scrapeable metrics and structured telemetry |
| **Frontend** | Vanilla JS (ES6+), Vanilla CSS (Glassmorphism), Chart.js 4 | Zero bloated frameworks, ultra-fast 60 FPS terminal rendering |
| **Testing** | pytest, pytest-asyncio, unittest.mock | 130 comprehensive unit & integration tests running in <6s |

---

## Key Performance & Stability Metrics

- **Test Suite Execution**: 130 tests execute in **5.48 seconds** with 100% offline isolation.
- **Inference Latency**: L1 memory cache hit < **1ms**; L2 disk cache hit < **5ms**; cold neural net inference ~ **80ms**.
- **Quant Calculation Latency**: Full 17-metric quantitative report with 1,000-path Monte Carlo in < **120ms**.
- **Circuit Breaker Trip Threshold**: 50% failure rate over 20-sample sliding window; 30s half-open probe recovery.
- **Token Bucket Rate Limiting**: 20 requests burst capacity with 2.0 req/s sustained token refill per IP.
