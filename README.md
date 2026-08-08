# StockBuddy Atelier — Quantitative Market Intelligence Engine

StockBuddy Atelier is an institutional-grade quantitative finance platform built with Flask, scikit-learn, TensorFlow, and a visual terminal inspector. It processes multi-decade historical market datasets via live `yfinance` pipelines to discover market regimes, run vector cosine scenario matching, and backtest tactical quantitative allocations in real-time.

> **Phases 1–6 complete.** Features statistical corrections, production ML pipeline, distributed microservice infrastructure, Atelier design UX, institutional quant research engine (17 metrics), containerization (Docker & Nginx HTTPS), and CI/CD deployment pipelines. See [PHASE_6_DEPLOYMENT.md](PHASE_6_DEPLOYMENT.md) for full deployment documentation.

---

## ⚡ Architecture & Progression (Phases 1 – 6)

- **Phase 1 — Statistical Corrections**: Fixed data leakage (scalers fit on train split only), context boundary windowing, transaction costs (0.10%), 5% risk-free Sharpe ratio, expanding walk-forward validation.
- **Phase 2 — Production ML Pipeline**: Disk model store (`ModelStore`), model registry (`ModelRegistry`), LRU/Disk inference cache (`InferenceCache`), background training pool (`BackgroundTrainer`), and automated retraining scheduler.
- **Phase 3 — Distributed Scale**: Parallel yfinance downloads, circuit breaker (`core/circuit_breaker.py`), IP token-bucket rate limiter, priority job queue (`PriorityJobQueue`), and Prometheus `/metrics` instrumentation.
- **Phase 4 — Usability & Atelier UX**: Responsive Atelier dark UI, execution pipeline workflow, keyboard shortcuts (`⌘K`), dark mode toggle, error states, and telemetry badges.
- **Phase 5 — Quantitative Research Platform**: 17 institutional quant metrics (Sharpe, Sortino, Calmar, Max Drawdown, CAGR, Alpha, Beta, Rolling Vol, Rolling Sharpe, Walk-Forward Optimisation, Cross-Validation IC Stability, Statistical Significance, Regime Confidence, Monte Carlo 1,000 paths, Permutation Feature Importance, 6×6 Markov Transition Matrix).
- **Phase 6 — Production Deployment & Cloud Infrastructure**: Multi-stage Docker, Nginx reverse proxy with SSL termination & TLS 1.2/1.3, production security headers, Render Blueprint (`render.yaml`), Railway manifest (`railway.json`), GitHub Actions CI/CD (`ci-cd.yml`), container health probes (`/health`, `/ready`), structured JSON logging, and Sentry error tracking.

---

## 🛡️ Statistical Integrity & Verification

Phase 1 & Phase 5 enforce statistical integrity across the analytics suite:

| Component | Metric / Behavior | Guarantee |
|---|---|---|
| **Data Partitioning** | Scaler Fitting | Fit exclusively on training split; zero lookahead bias |
| **Backtesting** | Transaction Friction | 0.10% round-trip cost subtracted on portfolio rebalances |
| **Risk Metrics** | Sharpe Ratio | Calculated with 5% annual risk-free rate benchmark |
| **Validation** | Walk-Forward Fold | 5-fold expanding window cross-validation |
| **Factor Testing** | Information Coefficient | Spearman rank correlation with t-test p-value significance |

---

## 🛠️ Technology Stack

- **Backend**: Python 3.10+, Flask REST API, Gunicorn, Pandas, NumPy, scikit-learn, SciPy, yfinance
- **Deep Learning**: TensorFlow (LSTM, GRU, Transformer)
- **Container & Proxy**: Docker, Docker Compose, Nginx (TLS 1.2/1.3, Rate Limiting, Security Headers)
- **Observability**: Prometheus Client (`/metrics`), Sentry Error Tracking, Structured JSON Logging
- **CI/CD & Cloud**: GitHub Actions, Render Blueprint (`render.yaml`), Railway (`railway.json`)
- **Frontend**: Vanilla JS (ES6+), Vanilla CSS, Chart.js 4, Google Fonts (Outfit, Inter, JetBrains Mono)

---

## 🚀 Deployment Quickstart

### 1. Docker Compose (Gunicorn + Nginx + HTTPS)
```bash
# Generate self-signed TLS certificates for local Nginx SSL
./nginx/generate-ssl.sh

# Start multi-container production stack
docker-compose up -d --build
```
Access the application at `https://localhost` or test health at `https://localhost/health`.

### 2. Standard Local Python Server
```bash
pip install -r requirements.txt
cp .env.example .env
python app.py serve
```

---

## 📡 API Endpoints Summary

| Method | Endpoint | Phase | Description |
|---|---|---|---|
| `GET` | `/health` | Phase 6 | Container Liveness Probe (K8s/Docker/Render) |
| `GET` | `/ready` | Phase 6 | Container Readiness Probe (Storage & Circuit Breaker) |
| `GET` | `/metrics` | Phase 3 | Prometheus scrape endpoint |
| `POST` | `/api/v5/quant` | Phase 5 | Full 17-metric quantitative research report |
| `POST` | `/api/v3/train` | Phase 3 | Priority job queue async ML training |
| `POST` | `/api/predict` | Phase 2 | Cached deep learning inference forecast |
| `POST` | `/api/regime` | Phase 1 | K-Means regime classification & backtest |

---

## 📑 Detailed Phase Documentation

- 📘 [Phase 1 — Statistical Corrections Documentation](PHASE_1_STATISTICAL_CORRECTIONS.md)
- 📙 [Phase 2 — Production ML Pipeline Architecture](PHASE_2_PRODUCTION_PIPELINE.md)
- 📗 [Phase 3 — Distributed Scale & Observability](PHASE_3_DISTRIBUTED_SCALE.md)
- 📓 [Phase 4 — Usability & UI/UX Design](PHASE_4_UI_UX_DESIGN.md)
- 📕 [Phase 5 — Quantitative Research Engine Documentation](PHASE_5_QUANT_RESEARCH.md)
- 📁 [Phase 6 — Production Deployment Manual](PHASE_6_DEPLOYMENT.md)