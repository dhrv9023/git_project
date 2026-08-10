# StockBuddy — Product Requirements Document (PRD)
**Version:** 2.0 | **Date:** 2026-08-10 | **Owner:** Engineering

---

## Feature Audit Status Legend
| Symbol | Meaning |
|---|---|
| ✅ **BUILT** | Fully implemented in codebase |
| ⚠️ **STUB** | Class/function exists but uses synthetic/hardcoded data |
| ❌ **MISSING** | Not implemented at all |

---

## Master Feature Status Table

| Feature | Status | Location |
|---|---|---|
| LLM-generated market summaries | ⚠️ STUB | `ml/ai_intelligence.py::AIMarketSynthesizer` — template strings, no real LLM |
| AI explanations / XAI | ⚠️ STUB | Same — SHAP weights are hardcoded `random.uniform` |
| Portfolio optimization | ✅ BUILT | `ml/ai_intelligence.py::PortfolioOptimizer` — real Markowitz MPT |
| Sentiment analysis | ⚠️ STUB | `MarketSentimentEngine` — returns `random.uniform` scores |
| News integration | ⚠️ STUB | Headlines are hardcoded strings, no real news API |
| Sector heatmaps | ⚠️ STUB | Hardcoded 6 sectors in `get_market_sentiment()` |
| Economic calendar | ⚠️ STUB | 4 hardcoded events, no live calendar API |
| Alert system | ⚠️ STUB | `WorkspaceManager` — in-memory list, no persistence, no triggering |
| Watchlists | ⚠️ STUB | Implied via `WorkspaceManager.save_workspace()` — no dedicated model |
| Risk scoring | ✅ BUILT | `RegimeService._compute_risk_score()` — real rolling volatility/RSI formula |
| Explainable AI | ⚠️ STUB | Feature attributions in `AIMarketSynthesizer` are formula-based, not SHAP |
| Scenario simulation | ❌ MISSING | Not implemented anywhere |
| Portfolio recommendations | ⚠️ STUB | `PortfolioOptimizer` returns weights but no personalized recommendations |
| User authentication | ⚠️ STUB | `v7_auth_login` returns a fake JWT string — no real auth, no password hashing |
| Saved workspaces | ⚠️ STUB | `WorkspaceManager._workspaces` — Python class dict, lost on restart |
| Multi-stock comparison | ❌ MISSING | No endpoint or service exists |
| WebSocket live updates | ❌ MISSING | No socket.io or SSE implementation |

---

## PRD — F01: LLM-Generated Market Summaries

### Problem Statement
Users cannot quickly interpret raw quantitative signals. Natural-language summaries lower the expertise barrier and increase decision speed.

### Current Status: ⚠️ STUB
`AIMarketSynthesizer.generate_narrative_summary()` uses Python f-strings. It produces plausible text but is not backed by an LLM — no model weights, no API call.

### User Stories
- As a **portfolio manager**, I want a one-paragraph AI brief on any ticker so I can make allocation decisions without reading 15 charts.
- As a **retail investor**, I want plain-English explanations of why the model is bullish/bearish.

### Technical Architecture
```
POST /api/v7/ai/explain
      │
      ▼
AIMarketSynthesizer.generate_narrative_summary()
      │
      ├─ [Current] f-string template only
      └─ [Target] Google Gemini API / OpenAI API
            │  prompt = system_context + quant_signals_json
            │  response = llm.generate(prompt)
            ▼
         Structured JSON: headline + summary + takeaways + attributions
```

### What Needs to Be Built
1. `GEMINI_API_KEY` env variable + `core/config.py` field
2. `ml/llm_client.py` — async wrapper around `google.generativeai` SDK
3. Prompt template with market context injection
4. Fallback to current f-string template if API key not set

### API Endpoints
| Method | Path | Description |
|---|---|---|
| POST | `/api/v7/ai/explain` | ✅ Exists — needs LLM backend |
| GET | `/api/v7/ai/explain/{ticker}` | ❌ Missing — GET shorthand |

### Database Schema
No DB needed — stateless generation. Cache response for 1h by ticker+regime key.

### Success Metrics
- Narrative generated in < 2s (p95)
- User engagement: avg session time +30% after launch
- Zero hallucinated ticker names (prompt guard)

### Testing Plan
- Unit: mock LLM client, assert response keys present
- Integration: call real Gemini with test prompt, assert no exception
- Regression: golden-file test — same input → same template output

### Future Roadmap
- Multi-model comparison (Gemini vs GPT-4o)
- User feedback thumbs up/down to fine-tune prompts

---

## PRD — F02: Explainable AI (XAI) Feature Attribution

### Problem Statement
Model predictions are black boxes. Users (especially institutional) need to know *why* the model is predicting up or down.

### Current Status: ⚠️ STUB
`feature_attributions` in `AIMarketSynthesizer` uses `round(float(random.uniform(-0.1, 0.15)), 4)` for MACD weight — this is **randomly generated on every call**, not real attribution.

### User Stories
- As a **quant analyst**, I want SHAP values for each feature so I can validate model behaviour.
- As a **compliance officer**, I want an audit trail explaining each prediction.

### Technical Architecture
```
POST /api/v7/ai/explain
      │
      ▼
PredictionService.predict()   ← returns model + input sequence
      │
      ▼
XAIService.compute_attributions(model, X_input)
      │
      ├─ SHAP DeepExplainer / GradientExplainer
      │  shap_values = explainer.shap_values(X_input[:1])
      │
      └─ Waterfall dict: {feature_name: shap_value}
```

### What Needs to Be Built
1. `pip install shap` in `requirements.txt`
2. `ml/xai.py` — `ShapExplainer` class wrapping `shap.DeepExplainer`
3. Cache SHAP values alongside model artifacts in `ModelArtifactRepository`

### API Endpoints
| Method | Path | Description |
|---|---|---|
| POST | `/api/v7/ai/explain` | Extend response with real SHAP values |
| GET | `/api/v7/xai/{ticker}/attributions` | ❌ Missing — per-ticker SHAP summary |

### Testing Plan
- Unit: `assert len(shap_values) == n_features`
- Property: `abs(shap_values).sum() ≈ prediction_delta` (SHAP completeness axiom)

### Success Metrics
- SHAP completeness axiom satisfied within 1% tolerance
- Attribution renders in < 500ms

---

## PRD — F03: Multi-Stock Comparison Dashboard

### Problem Statement
Users need to compare multiple tickers side by side — performance, regime, risk score — without making N separate API calls.

### Current Status: ❌ MISSING
No endpoint or service exists for multi-stock comparison.

### User Stories
- As a **fund manager**, I want to compare AAPL, MSFT, GOOGL on one screen.
- As a **trader**, I want to see which of my watchlist stocks is in the best regime right now.

### Technical Architecture
```
POST /api/v2/compare
  body: { tickers: ["AAPL","MSFT","GOOGL"], start, end }
        │
        ▼
  ComparisonService.compare(tickers, start, end)
        │  Parallel execution via ThreadPoolExecutor
        ├─ RegimeService.classify("AAPL", ...)
        ├─ RegimeService.classify("MSFT", ...)
        └─ RegimeService.classify("GOOGL", ...)
        │
        ▼
  Merge results → normalized price series → correlation matrix → response
```

### What Needs to Be Built
1. `app/services/comparison_service.py` — `ComparisonService`
2. `app/api/v2_routes.py` — `POST /api/v2/compare` endpoint
3. Normalised price index (base=100 at start date) for all tickers
4. Pearson correlation matrix from `LogReturn` series

### API Endpoints
| Method | Path | Description | Status |
|---|---|---|---|
| POST | `/api/v2/compare` | Multi-stock comparison | ❌ Missing |

### Database Schema
None — fully stateless. Results cached by sorted ticker list + date range.

### Success Metrics
- Response for 5 tickers in < 5s
- Correlation matrix is symmetric (test invariant)

---

## PRD — F04: Portfolio Optimization

### Problem Statement
Users want optimal asset allocation given a set of tickers, not just prediction signals.

### Current Status: ✅ BUILT (real Markowitz MPT)
`PortfolioOptimizer.optimize_portfolio()` implements Monte Carlo efficient frontier. Exposed via `POST /api/v7/portfolio/optimize`.

### Gap
- Uses `random` log-return data as fallback instead of raising `DataFetchError`
- No persistence of saved portfolios
- No rebalancing alerts
- `num_portfolios=50` Monte Carlo — should be `scipy.optimize` for precision

### Improvements Needed
1. Replace Monte Carlo with `scipy.optimize.minimize` (SLSQP) for exact tangency portfolio
2. Add `POST /api/v7/portfolio/save` to persist portfolio configs via `WorkspaceManager`
3. Add `GET /api/v7/portfolio/{id}/rebalance` to compute drift since last optimisation

### API Endpoints
| Method | Path | Status |
|---|---|---|
| POST | `/api/v7/portfolio/optimize` | ✅ Exists |
| POST | `/api/v7/portfolio/save` | ❌ Missing |
| GET | `/api/v7/portfolio/{id}/rebalance` | ❌ Missing |

### Success Metrics
- Max Sharpe portfolio Sharpe ≥ equal-weight Sharpe (mathematical invariant — testable)
- Optimization completes in < 1s for up to 20 assets

---

## PRD — F05: Sector Heatmaps

### Problem Statement
Macro-level sector rotation context is critical for allocation decisions but requires expensive data feeds.

### Current Status: ⚠️ STUB
`MarketSentimentEngine.get_market_sentiment()` returns a hardcoded list of 6 sectors with static `change_pct` values.

### What Needs to Be Built
1. `ml/sector_data.py` — Fetch S&P 500 sector ETF prices (XLK, XLF, XLV, XLE, XLY, XLI, XLP, XLU, XLRE, XLB) via yfinance
2. Compute 1D, 5D, 1M returns for each ETF
3. Map each stock ticker to its GICS sector via a static lookup table
4. Cache sector data for 15 minutes (market hours only)

### API Endpoints
| Method | Path | Status |
|---|---|---|
| GET | `/api/v7/market/intelligence` | ✅ Exists — stub |
| GET | `/api/v7/sectors` | ❌ Missing — dedicated endpoint |

---

## PRD — F06: Sentiment Analysis & News Integration

### Problem Statement
Price-only models miss the impact of earnings calls, Fed speeches, and geopolitical events.

### Current Status: ⚠️ STUB
Headlines are hardcoded strings. Sentiment scores are `random.uniform(0.15, 0.78)`.

### Technical Architecture (Target)
```
GET /api/v7/market/intelligence?ticker=AAPL
      │
      ▼
NewsAggregator.fetch(ticker)
      │  ├─ Alpha Vantage News API  (free tier: 25 req/day)
      │  ├─ Yahoo Finance RSS feed  (free, no key)
      │  └─ Finnhub News API        (free tier: 60 req/min)
      │
      ▼
SentimentScorer.score(headlines)
      │  FinBERT model (ProsusAI/finbert via HuggingFace Transformers)
      │  Input: headline text
      │  Output: {positive, negative, neutral} probabilities
      ▼
Aggregate weighted score by recency decay
```

### What Needs to Be Built
1. `ml/news_aggregator.py` — news fetching with circuit breaker
2. `ml/sentiment_scorer.py` — FinBERT inference (or VADER as lightweight fallback)
3. `ALPHA_VANTAGE_API_KEY`, `FINNHUB_API_KEY` env vars in `AppConfig`
4. News cache: 30-minute TTL by ticker

### API Endpoints
| Method | Path | Status |
|---|---|---|
| GET | `/api/v7/market/intelligence` | ✅ Exists — stub |
| GET | `/api/v7/news/{ticker}` | ❌ Missing |

### Success Metrics
- FinBERT F1 > 0.80 on Financial PhraseBank benchmark
- News fetch latency < 800ms (p95)

---

## PRD — F07: Economic Calendar

### Current Status: ⚠️ STUB
4 hardcoded FOMC/CPI events in `MarketSentimentEngine.get_market_sentiment()`.

### What Needs to Be Built
1. Integrate **Econdb** or **Investing.com** API for live economic events
2. `ml/economic_calendar.py` — `EconomicCalendarService`
3. Filter events by currency (USD only for US stocks), impact level (HIGH/MEDIUM)
4. Auto-refresh daily at midnight UTC via `RetrainingScheduler`

---

## PRD — F08: WebSocket Live Updates

### Problem Statement
The dashboard currently requires manual refresh. Institutional traders need live price updates and regime-change notifications without polling.

### Current Status: ❌ MISSING
No WebSocket, SSE, or any real-time mechanism exists.

### Technical Architecture
```
Client WebSocket ws://host/ws/live
      │
      ▼
Flask-SocketIO server (eventlet async worker)
      │
      ├─ Room: "ticker:{AAPL}"  ← price updates every 5s
      │         yfinance.download(period="1d", interval="1m") polling
      │
      ├─ Room: "alerts:{user_id}"  ← alert trigger notifications
      │
      └─ Room: "regime:{ticker}"  ← regime change events
```

### What Needs to Be Built
1. `pip install flask-socketio eventlet` → `requirements.txt`
2. `app/websocket/` package:
   - `live_prices.py` — background thread polling yfinance 1m bars
   - `alert_dispatcher.py` — evaluates alert conditions on each tick
3. `app/__init__.py` — replace `Flask(__name__)` with `SocketIO(Flask(__name__))`
4. Gunicorn worker class: `eventlet` (update `Dockerfile`)

### API Endpoints (WebSocket Events)
| Event | Direction | Payload |
|---|---|---|
| `subscribe_ticker` | Client → Server | `{ticker, interval}` |
| `price_update` | Server → Client | `{ticker, price, volume, timestamp}` |
| `regime_change` | Server → Client | `{ticker, old_regime, new_regime}` |
| `alert_triggered` | Server → Client | `{alert_id, ticker, condition, value}` |

### Success Metrics
- Price update latency < 1s after yfinance data available
- Supports 100 concurrent subscribers without degradation

---

## PRD — F09: Alert System

### Problem Statement
Users want to be notified when price, RSI, or regime conditions are met — without constantly watching the dashboard.

### Current Status: ⚠️ STUB
`WorkspaceManager.create_alert()` stores alerts in a Python class-level list (`_alerts = []`) which is **lost on every server restart** and **never evaluated against real prices**.

### What Needs to Be Built
1. **Persistence**: Store alerts in JSON file via `ModelStore` or SQLite
2. **Evaluation engine**: `AlertEvaluator` background thread that:
   - Runs every 60s during market hours
   - Fetches latest price for each watched ticker
   - Evaluates conditions (`PRICE_ABOVE`, `RSI_ABOVE`, `REGIME_CHANGE`, etc.)
   - Dispatches via WebSocket `alert_triggered` event
3. **Alert types**: `PRICE_ABOVE`, `PRICE_BELOW`, `RSI_ABOVE`, `RSI_BELOW`, `REGIME_CHANGE`, `DRAWDOWN_EXCEEDS`

### API Endpoints
| Method | Path | Status |
|---|---|---|
| POST | `/api/v7/alerts` | ✅ Exists — no persistence |
| GET | `/api/v7/alerts` | ✅ Exists — in-memory only |
| DELETE | `/api/v7/alerts/{id}` | ❌ Missing |
| PATCH | `/api/v7/alerts/{id}` | ❌ Missing |

---

## PRD — F10: Watchlists

### Current Status: ⚠️ STUB
`WorkspaceManager.save_workspace()` can store a watchlist config but the structure is generic and lost on restart.

### What Needs to Be Built
1. Dedicated `Watchlist` model (not generic workspace config)
2. JSON persistence via `ModelStore`
3. `GET /api/v7/watchlist` → return list of tickers with latest regime + risk score
4. Batch fetch: one yfinance call for all tickers in watchlist

### API Endpoints
| Method | Path | Status |
|---|---|---|
| GET | `/api/v7/watchlist` | ❌ Missing |
| POST | `/api/v7/watchlist` | ❌ Missing |
| DELETE | `/api/v7/watchlist/{ticker}` | ❌ Missing |

---

## PRD — F11: Risk Scoring

### Current Status: ✅ BUILT
`RegimeService._compute_risk_score()` computes a real rolling 0–10 risk score using volatility ratio, RSI extremity, trend alignment, and MACD divergence. Returned in `/api/regime` response.

### Gap
- No dedicated `/api/risk/{ticker}` endpoint — score is buried in regime response
- No portfolio-level risk score (weighted sum of individual scores)
- No historical risk score time series endpoint

### What Needs to Be Built
1. `GET /api/v2/risk/{ticker}` — standalone risk score endpoint
2. `POST /api/v7/portfolio/risk` — portfolio-level VaR + risk score

---

## PRD — F12: Scenario Simulation

### Problem Statement
Users want to ask "what happens to my portfolio if AAPL drops 20%?" or "what if the Fed raises rates?"

### Current Status: ❌ MISSING
Not implemented anywhere in the codebase.

### Technical Architecture
```
POST /api/v7/scenario
  body: {
    portfolio: {AAPL: 0.4, MSFT: 0.3, GOOGL: 0.3},
    shocks: [
      {ticker: "AAPL", shock_pct: -20},
      {factor: "interest_rate", shock_bps: +50}
    ]
  }
      │
      ▼
ScenarioService.simulate(portfolio, shocks)
      │
      ├─ Apply price shocks to position weights
      ├─ Recalculate covariance with shocked correlations
      ├─ Compute delta PnL, new VaR, new max drawdown
      └─ Return: {original_metrics, shocked_metrics, impact_summary}
```

### What Needs to Be Built
1. `app/services/scenario_service.py` — `ScenarioService`
2. `app/api/v7_routes.py` — `POST /api/v7/scenario`
3. Factor shock library (interest rate → bond proxy beta, oil → energy correlation)

---

## PRD — F13: User Authentication

### Current Status: ⚠️ STUB
`POST /api/v7/auth/login` returns a **fake** JWT string constructed as:
```python
token = f"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{username}.sb_v7_session"
```
This is **not a real JWT** — no signature, no expiry, no secret key. Any username is accepted.

### What Needs to Be Built
1. `pip install Flask-JWT-Extended bcrypt`
2. `app/auth/` package:
   - `auth_service.py` — `hash_password`, `verify_password`, `create_token`, `decode_token`
   - `decorators.py` — `@jwt_required` wrapper
3. User store — JSON file via `ModelStore` (no DB required for portfolio project)
4. Token expiry: 24h access token, 7d refresh token

### API Endpoints
| Method | Path | Status |
|---|---|---|
| POST | `/api/v7/auth/login` | ⚠️ Fake JWT |
| POST | `/api/v7/auth/register` | ❌ Missing |
| POST | `/api/v7/auth/refresh` | ❌ Missing |
| POST | `/api/v7/auth/logout` | ❌ Missing |

---

## PRD — F14: Saved Workspaces

### Current Status: ⚠️ STUB
`WorkspaceManager._workspaces` is a class-level dict — lost on every restart.

### What Needs to Be Built
1. `storage/workspace_store.py` — JSON file persistence (same pattern as `ModelStore`)
2. Workspace schema: `{user_id, name, tickers[], layout, chart_settings, alerts[], watchlist[]}`
3. Wire into `WorkspaceManager` so load/save goes to disk

### API Endpoints
| Method | Path | Status |
|---|---|---|
| GET | `/api/v7/workspaces` | ❌ Missing |
| POST | `/api/v7/workspaces` | ❌ Missing |
| GET | `/api/v7/workspaces/{name}` | ❌ Missing |
| PUT | `/api/v7/workspaces/{name}` | ❌ Missing |
| DELETE | `/api/v7/workspaces/{name}` | ❌ Missing |

---

## Implementation Priority Matrix

| Feature | Effort | Portfolio Impact | Priority |
|---|---|---|---|
| Real sentiment scoring (FinBERT/VADER) | M | High | 🔴 P1 |
| Multi-stock comparison | S | High | 🔴 P1 |
| Real JWT auth | M | High | 🔴 P1 |
| Workspace persistence | S | High | 🔴 P1 |
| Real LLM summaries (Gemini API) | S | High | 🟡 P2 |
| SHAP XAI attributions | M | High | 🟡 P2 |
| Scenario simulation | L | Medium | 🟡 P2 |
| Alert persistence + evaluation | M | Medium | 🟡 P2 |
| WebSocket live updates | L | Medium | 🟠 P3 |
| Real sector data (ETF prices) | S | Medium | 🟠 P3 |
| Scipy portfolio optimization | S | Low | 🟠 P3 |
| Economic calendar API | M | Low | 🟠 P3 |

**Effort:** S = < 1 day, M = 1–3 days, L = 3–7 days

---

## Next Implementation Steps (P1 Sprint)

```
Step 1: Multi-stock comparison (new ComparisonService + endpoint)
Step 2: Workspace persistence (JSON file storage)
Step 3: Real JWT auth (Flask-JWT-Extended)
Step 4: FinBERT/VADER sentiment scoring
Step 5: Gemini API for LLM summaries
Step 6: SHAP XAI attributions
Step 7: Scenario simulation service
Step 8: WebSocket live prices
```
