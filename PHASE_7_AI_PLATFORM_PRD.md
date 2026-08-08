# Phase 7 Product Requirement Document (PRD): Institutional AI Financial Intelligence Platform

> **Document Version:** 7.0.0  
> **Target Audience:** Quantitative Researchers, AI Engineers, Product Managers, Senior Architects  
> **Status:** APPROVED & IMPLEMENTED  

---

## 1. Executive Summary & Problem Statement

### 1.1 Problem Statement
Modern quantitative finance platforms frequently suffer from three primary deficiencies:
1. **Black-Box AI Models:** Deep learning time-series models (LSTMs, Transformers) output numerical price arrays or directional probabilities without providing human-interpretable rationale or feature contribution weights.
2. **Fragmented Quantitative Tools:** Portfolio optimization (Markowitz Modern Portfolio Theory), macro sentiment aggregation, and market regime tracking are fragmented across separate tools, creating friction for quantitative decision-making.
3. **Lack of Explainability & Risk Context:** Quants and portfolio managers cannot easily inspect why an AI model predicts a specific market outcome or how regime shifts alter optimal asset allocation.

### 1.2 Solution Overview
StockBuddy Phase 7 transforms the core platform into a unified **AI Financial Intelligence Platform**. It introduces:
- **LLM-Generated Market Summaries:** Automated natural-language synthesis converting quantitative metrics into executive commentary.
- **Explainable AI (XAI) Feature Attribution:** SHAP-style relative contribution metrics explaining exact indicators driving model forecasts.
- **Markowitz Portfolio Optimization:** Efficient Frontier curve simulation, Tangency Portfolio (Maximum Sharpe Ratio), and Minimum Volatility allocation matrices across custom asset universes.
- **Sector Heatmaps & Macro Calendar:** Live tracking of 6 major market sectors and high-impact macroeconomic events (FOMC, CPI, Non-Farm Payrolls).
- **Alert System & Workspace Management:** User-defined price/RSI/regime triggers and session persistence.

---

## 2. Feature Specifications & User Stories

### Feature 1: LLM Market Synthesis & Narrative Generator
- **User Story:** *As a Quantitative Portfolio Manager, I want natural-language executive briefs of complex market metrics so that I can quickly synthesize market state without deciphering raw arrays.*
- **Capabilities:** Rule-directed prompt template synthesis generating headline, executive summary, key takeaways, and tactical recommendation.

### Feature 2: Explainable AI (XAI) Feature Attribution
- **User Story:** *As a Model Validator, I want to see which technical indicators (RSI, Volatility, EMA Slope) impacted a prediction so that I can audit model sanity and prevent data leakage.*
- **Capabilities:** SHAP-style permutation feature weights displayed as a horizontal waterfall bar chart.

### Feature 3: Multi-Stock Markowitz Portfolio Optimization
- **User Story:** *As an Asset Manager, I want to compute optimal portfolio weights across a multi-stock universe so that I can maximize Sharpe ratio for a given risk tolerance.*
- **Capabilities:** Monte Carlo Efficient Frontier scatter plot, Tangency Portfolio, Minimum Variance Portfolio, and Equal Weight comparisons.

### Feature 4: Sector Heatmap & News Sentiment
- **User Story:** *As a Tactical Trader, I want a real-time heatmap of sector performance and sentiment so that I can spot sector rotation early.*
- **Capabilities:** 6-sector status tiles (Tech, Financials, Healthcare, Energy, Consumer, Industrials) with daily % change, sentiment score, and active regime state.

### Feature 5: Macro Economic Calendar
- **User Story:** *As a Risk Officer, I want an integrated macro calendar so that I can hedge positions before major volatility events like CPI or Fed Rate decisions.*
- **Capabilities:** Upcoming economic events with consensus estimates and volatility impact ratings (HIGH / MEDIUM / LOW).

---

## 3. Technical Architecture & Data Flow

```
                      ┌──────────────────────────────────────────────┐
                      │        Browser Dashboard (index.html)        │
                      │  - AI Copilot Executive Brief                │
                      │  - XAI Feature Attribution Waterfall         │
                      │  - Markowitz Efficient Frontier Scatter      │
                      │  - Sector Heatmap Grid                       │
                      └──────────────────────┬───────────────────────┘
                                             │
                                   HTTP REST API Requests
                                             │
                      ┌──────────────────────▼───────────────────────┐
                      │          Flask Backend (app.py v7)           │
                      └───────┬──────────────┬──────────────┬────────┘
                              │              │              │
                              ▼              ▼              ▼
                     ┌───────────┐  ┌───────────┐  ┌───────────┐
                     │  v7/ai/   │  │ v7/port/  │  │ v7/market/│
                     │  explain  │  │ optimize  │  │ intel     │
                     └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
                           │              │              │
                           └──────────────┼──────────────┘
                                          ▼
                     ┌──────────────────────────────────────────┐
                     │         ml/ai_intelligence.py            │
                     │  - AIMarketSynthesizer                   │
                     │  - PortfolioOptimizer (Markowitz MPT)    │
                     │  - MarketSentimentEngine                 │
                     │  - WorkspaceManager & Alert Engine       │
                     └──────────────────────────────────────────┘
```

---

## 4. Database & Workspace Schema

### 4.1 Saved Workspace Schema (`JSON`)
```json
{
  "user_id": "quant_user_01",
  "workspace_name": "Tech_Sector_Alpha",
  "updated_at": "2026-08-08T21:48:00Z",
  "config": {
    "primary_ticker": "AAPL",
    "portfolio_assets": ["AAPL", "MSFT", "GOOGL", "SPY"],
    "risk_free_rate": 0.05,
    "active_alerts": [
      {
        "alert_id": "alt_1723145000",
        "ticker": "AAPL",
        "condition_type": "RSI_ABOVE",
        "threshold": 70.0,
        "status": "ACTIVE"
      }
    ]
  }
}
```

### 4.2 User Authentication Token Schema (`JWT / Mock Session`)
```json
{
  "status": "success",
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.quant_user.sb_v7_session",
  "user": {
    "username": "quant_user",
    "role": "Institutional Quant",
    "permissions": ["read", "write", "execute_models", "portfolio_opt"]
  }
}
```

---

## 5. API Endpoints Specification

| Endpoint | Method | Input Payload | Response Output | Description |
| :--- | :--- | :--- | :--- | :--- |
| `/api/v7/ai/explain` | `POST` | `{"ticker": "AAPL", "confidence": 0.76}` | Executive summary, takeaways, feature attributions | Generates LLM narrative summary & XAI feature attributions |
| `/api/v7/portfolio/optimize` | `POST` | `{"tickers": ["AAPL", "MSFT"], "risk_free_rate": 0.05}` | Efficient frontier points, Tangency weights, Min Vol weights | Computes Markowitz Mean-Variance Optimal Allocations |
| `/api/v7/market/intelligence` | `GET` | `?ticker=AAPL` | Sector heatmap, news headlines, macro calendar | Returns sector heatmap, news sentiment, and economic calendar |
| `/api/v7/alerts` | `GET / POST` | `{"ticker": "AAPL", "threshold": 70.0}` | Active alert rules list | Creates and manages price/regime alert rules |
| `/api/v7/auth/login` | `POST` | `{"username": "quant_user"}` | JWT Token + User profile | User session authentication |

---

## 6. UI Mockup & Visual Layout

```
+-----------------------------------------------------------------------------------+
| StockBuddy | Architecture | Regime Engine | Deep Learning | ⚗️ Quant Lab | 🤖 AI Intelligence |
+-----------------------------------------------------------------------------------+
| 🤖 AI Financial Intelligence Command Center                                       |
| [Primary Ticker: AAPL] [Portfolio Assets: AAPL, MSFT, GOOGL, SPY] [Run AI Engine]  |
+--------------------------------------------------+--------------------------------+
| 🧠 LLM Market Executive Brief                    | 🔍 Explainable AI (XAI)        |
| AI Brief: AAPL demonstrates Bullish bias...      | RSI (14-Day)     [=========>   ] |
| - Regime: Momentum Breakout (76% confidence)     | 20d Volatility   [<====      ] |
| - Technical Driver: RSI at 58.2                  | EMA Slope        [======>    ] |
+--------------------------------------------------+--------------------------------+
| 📈 Markowitz Efficient Frontier                  | ⚖️ Optimal Allocations         |
| [Scatter Plot: Return vs Volatility]              | AAPL : Max Sharpe 35% | Min Vol 20%
| • Tangency Portfolio (Max Sharpe)                | MSFT : Max Sharpe 40% | Min Vol 45%
| • Minimum Volatility Portfolio                   | GOOGL: Max Sharpe 15% | Min Vol 20%
+--------------------------------------------------+--------------------------------+
| 🔥 Sector Heatmap (6 Sectors)                    | 📅 Macro Economic Calendar     |
| • Info Tech : +1.42% (Bullish Breakout)          | • FOMC Rate Decision (Aug 12)  |
| • Financials: +0.85% (Bullish Recovery)          | • US CPI Inflation   (Aug 15)  |
+--------------------------------------------------+--------------------------------+
```

---

## 7. Testing & Quality Assurance Plan

### 7.1 Automated Unit & Integration Tests
- **`tests/test_ai_intelligence.py`**: Validates `AIMarketSynthesizer`, `PortfolioOptimizer`, `MarketSentimentEngine`, and `WorkspaceManager`.
- **`tests/test_deployment.py`**: Verifies health probes (`/health`, `/ready`), security headers, and config environment loading.

### 7.2 Static Analysis & Zero-Warning Linting
- **`pyflakes app.py ml/`**: Enforces zero missing imports, zero unused variables, and clean variable scoping.

---

## 8. Success Metrics & Key Performance Indicators (KPIs)

1. **Portfolio Sharpe Improvement:** Optimization engine delivers $\ge 15\%$ higher Sharpe ratio compared to equal-weighted benchmarks.
2. **API Latency:** `/api/v7/ai/explain` and `/api/v7/portfolio/optimize` respond in $< 450\text{ ms}$.
3. **Codebase Compliance:** 100% test coverage for core math engines and 0 pyflakes linting warnings.

---

## 9. Future Engineering Roadmap

- **Phase 8 (Q4 2026):** Real-time WebSocket streaming for tick-level order book depth and live trade execution feeds.
- **Phase 9 (Q1 2027):** Reinforcement Learning (PPO/DQN) portfolio rebalancing agents with transaction fee constraints.
