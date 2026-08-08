# Phase 6 — Production Infrastructure & Deployment Engineering Report

> **Institutional-Grade Quantitative Research Engine Deployment Manual**  
> *StockBuddy Platform — Phase 6: Production Infrastructure, Containerization, Nginx, HTTPS, CI/CD & Observability*

---

## Executive Summary

Phase 6 upgrades StockBuddy from a locally executable research prototype into an **institutional, deployment-ready quantitative intelligence web application**. The platform is hardened with container isolation (Docker multi-stage builds), production WSGI orchestration (Gunicorn with async worker threads), reverse-proxy security & TLS termination (Nginx with HTTPS & security headers), 1-click cloud platform blueprints (Render & Railway), automated CI/CD pipelines (GitHub Actions), health & readiness probes (`/health`, `/ready`), structured JSON logging, and Sentry error tracking.

---

## System Architecture

```
                          ┌───────────────────────────┐
                          │   Internet / Web Clients  │
                          └─────────────┬─────────────┘
                                        │
                               (HTTPS Port 443 / 80)
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │            Nginx Reverse Proxy            │
                  │  - SSL/TLS Termination (TLS 1.2/1.3)     │
                  │  - Security Headers (HSTS, CSP, X-Frame)  │
                  │  - Token-Bucket Rate Limiting             │
                  │  - Gzip Compression & Asset Caching       │
                  └─────────────────────┬─────────────────────┘
                                        │
                             (Internal HTTP Port 5000)
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │          Gunicorn WSGI Application        │
                  │  - 4 Worker Processes / 2 Threads each    │
                  │  - Non-Root `appuser` Execution           │
                  └─────────────────────┬─────────────────────┘
                                        │
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │             Flask Core Engine             │
                  │  - /health (Liveness) & /ready (Readiness)│
                  │  - Structured JSON Logging Formatter      │
                  │  - Sentry Error Tracking Hook             │
                  │  - Prometheus /metrics Exporter           │
                  │  - v1-v5 Quantitative Research API      │
                  └───────────────────────────────────────────┘
```

---

## Section 1: Environment Variables & Configuration

StockBuddy utilizes centralized, environment-aware configuration managed via `core/config.py` with automatic `.env` file resolution.

### Primary Environment Variables

| Variable | Type | Default | Description |
|---|---|---|---|
| `ENVIRONMENT` / `SB_ENVIRONMENT` | `string` | `development` | Environment mode (`development`, `staging`, `production`). |
| `PORT` / `SB_PORT` | `integer` | `5000` | Server HTTP port. |
| `HOST` / `SB_HOST` | `string` | `0.0.0.0` | Bind IP host address. |
| `LOG_FORMAT` / `SB_LOG_FORMAT` | `string` | `text` | Logging output format (`text` or `json`). Set to `json` for production. |
| `SENTRY_DSN` / `SB_SENTRY_DSN` | `string` | `""` | Sentry DSN key for automatic error tracking & APM. |
| `SECURITY_HEADERS_ENABLED` | `boolean` | `true` | Enables automatic HTTP security headers middleware. |
| `SB_FETCH_PARALLELISM` | `integer` | `4` | Number of concurrent worker threads for historical data downloads. |
| `SB_MAX_WORKER_THREADS` | `integer` | `4` | Background model training thread pool capacity. |
| `SB_INFERENCE_CACHE_TTL_S` | `integer` | `3600` | Inference cache expiration TTL in seconds (default 1 hr). |
| `SB_CB_FAILURE_THRESHOLD` | `float` | `0.5` | Circuit breaker failure ratio trigger for `yfinance`. |

---

## Section 2: Local Setup Guide

### Prerequisites
- Python 3.10+
- `pip` & `virtualenv`

### Step-by-Step Execution
```bash
# 1. Clone & navigate to project root
cd StockBuddy

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install production & development dependencies
pip install -r requirements.txt

# 4. Copy template configuration
cp .env.example .env

# 5. Start Flask development server
python app.py serve
```
Access the application at `http://localhost:5000`.

---

## Section 3: Containerization & Docker Setup

StockBuddy utilizes a hardened, multi-stage Docker build producing a lean container image based on `python:3.10-slim`.

### Single Container Build & Run
```bash
# Build production Docker image
docker build -t stockbuddy:v6.0 .

# Run container with environment overrides
docker run -d \
  --name stockbuddy_app \
  -p 5000:5000 \
  -e ENVIRONMENT=production \
  -e LOG_FORMAT=json \
  stockbuddy:v6.0

# Inspect container health probe status
docker inspect --format='{{json .State.Health}}' stockbuddy_app | jq
```

---

## Section 4: Production Setup (Gunicorn + Nginx + HTTPS)

### Multi-Container Stack Orchestration via Docker Compose
The `docker-compose.yml` stack provisions both the Gunicorn Flask application and the Nginx SSL reverse proxy.

```bash
# 1. Generate local self-signed TLS certificates for Nginx SSL
./nginx/generate-ssl.sh

# 2. Launch multi-container stack in detached mode
docker-compose up -d --build

# 3. View container logs
docker-compose logs -f app
```

### Access Endpoints
- **HTTPS Web Console & Quant Lab**: `https://localhost/`
- **Container Liveness Probe**: `https://localhost/health`
- **Container Readiness Probe**: `https://localhost/ready`
- **Prometheus Telemetry Metrics**: `https://localhost/metrics`

### Security Headers Enforced by Nginx & Flask
- `Strict-Transport-Security: max-age=31536000; includeSubDomains`
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: SAMEORIGIN`
- `X-XSS-Protection: 1; mode=block`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Content-Security-Policy: default-src 'self' ...`

---

## Section 5: Cloud Deployment Blueprint (Render & Railway)

### Option A: Render Cloud Deployment
1. Log into [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** → **Blueprint**.
3. Connect your GitHub repository (`dhrv9023/git_project`).
4. Render automatically detects `render.yaml` and provisions the Web Service using Docker.
5. Environment variables `PORT`, `HOST`, `ENVIRONMENT=production`, `LOG_FORMAT=json` are configured automatically.
6. The health check probe is set to `/health`.

### Option B: Railway Cloud Deployment
1. Log into [Railway.app](https://railway.app/).
2. Click **New Project** → **Deploy from GitHub repo**.
3. Select `StockBuddy`.
4. Railway detects `railway.json` and builds via `Dockerfile`.
5. Add custom domain or use Railway auto-generated domain.

---

## Section 6: CI/CD Pipeline (GitHub Actions)

The repository includes an automated workflow (`.github/workflows/ci-cd.yml`) that executes on every pull request and push to `main`:

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ 🧪 Lint & Test  │ ────► │ 🐳 Docker Build │ ────► │ 🚀 Cloud Deploy │
│ - Pyflakes      │       │ - Multi-Stage   │       │ - Render Webhook│
│ - Pytest        │       │ - Health Check  │       │   (main branch) │
└─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **Lint & Test**: Runs static syntax verification (`pyflakes`) and unit tests (`pytest`).
2. **Docker Build & Health Probe Verification**: Builds container image, launches container, and verifies `curl http://localhost:5000/health` returns HTTP 200 OK.
3. **Automated Cloud Deploy**: Triggers the Render continuous deployment webhook upon successful build on `main`.

---

## Section 7: Health Checks & Observability

### Liveness Probe (`GET /health`)
Used by Kubernetes, Render, Railway, and Nginx to determine if the container process is alive.
```json
{
  "status": "healthy",
  "timestamp": "2026-08-08T21:32:00.000000+00:00",
  "uptime_seconds": 342.15,
  "environment": "production",
  "version": "6.0.0"
}
```

### Readiness Probe (`GET /ready`)
Verifies system readiness before routing live user traffic:
1. **Storage Check**: Verifies `model_artifacts` directory is present and writeable.
2. **Circuit Breaker Check**: Checks `yfinance` circuit breaker status (`CLOSED`, `HALF-OPEN`, or `OPEN`).
3. **Queue Health**: Checks background priority job queue status.

---

## Section 8: Production Scaling Strategy

1. **Stateless Horizontal Scaling**:
   - The Flask container maintains model state in mounted persistent volumes or object storage (`model_artifacts/`).
   - Multiple container instances can be run behind an AWS ALB or Nginx load balancer.
2. **Async Background Model Training**:
   - Model training requests are offloaded to `PriorityJobQueue` worker threads (`ml/queue.py`), preventing API request blocking.
3. **Inference Caching**:
   - LRU memory cache + disk cache (`ml/inference.py`) serves cached predictions in <5ms.

---

## Section 9: Operational Troubleshooting Guide

| Issue | Symptom | Cause | Solution |
|---|---|---|---|
| **Port 5000 in use** | `OSError: [Errno 98] Address already in use` | Another process listening on port 5000. | Run `lsof -i :5000` and kill process, or set `PORT=5001`. |
| **Circuit Breaker Open** | `503 Service Unavailable` on ticker download | `yfinance` API rate limit or outage. | Check `/ready` endpoint status. Wait 60s for automatic half-open probe reset. |
| **Permission Denied in Docker** | Container exits immediately on boot | Non-root `appuser` cannot write to `/app/model_artifacts`. | Verify `Dockerfile` permissions: `chown -R appuser:appgroup /app`. |
| **SSL Handshake Failure** | Browser warning when loading `https://localhost` | Self-signed certificate used locally. | Click "Proceed to localhost (unsafe)" in browser, or install domain SSL via Let's Encrypt / Certbot. |

---

## Summary of Completed Deliverables

- [x] **Dockerfile**: Multi-stage production container build with non-root security.
- [x] **.dockerignore**: Excludes unneeded source files and binaries.
- [x] **docker-compose.yml**: Multi-container stack (Gunicorn + Nginx reverse proxy).
- [x] **Nginx Config & SSL**: TLS 1.2/1.3, rate-limiting zones, security headers, gzip, static caching.
- [x] **Cloud Manifests**: `render.yaml`, `railway.json`, and `.env.example`.
- [x] **Health Probes**: `/health` (liveness) and `/ready` (readiness).
- [x] **Security Middleware**: Auto-applies standard security headers to every response.
- [x] **Logging & Error Tracking**: Structured JSON log option + Sentry DSN hook.
- [x] **CI/CD**: GitHub Actions workflow with linting, testing, Docker build & container health probe.
- [x] **Documentation**: Complete engineering manual in `PHASE_6_DEPLOYMENT.md`.
