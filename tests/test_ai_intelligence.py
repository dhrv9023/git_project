"""
tests/test_ai_intelligence.py — Automated Unit Tests for Phase 7 AI Intelligence Engine
"""

import numpy as np
from ml.ai_intelligence import (
    AIMarketSynthesizer,
    PortfolioOptimizer,
    MarketSentimentEngine,
    WorkspaceManager
)
from app import flask_app


def test_ai_market_synthesizer():
    metrics = {
        'last_price': 185.20,
        'return_1m': 4.2,
        'volatility_20d': 17.5,
        'rsi': 62.0
    }
    result = AIMarketSynthesizer.generate_narrative_summary("AAPL", metrics, "Momentum Breakout", "UP", 0.78)
    assert result["ticker"] == "AAPL"
    assert "executive_summary" in result
    assert len(result["feature_attributions"]) == 5
    assert result["sentiment_bias"] == "Bullish"


def test_portfolio_optimizer():
    np.random.seed(42)
    asset_data = {
        "AAPL": np.random.normal(0.001, 0.015, 252).tolist(),
        "MSFT": np.random.normal(0.0008, 0.014, 252).tolist(),
        "GOOGL": np.random.normal(0.0005, 0.016, 252).tolist(),
        "SPY": np.random.normal(0.0004, 0.009, 252).tolist()
    }
    res = PortfolioOptimizer.optimize_portfolio(asset_data, rf_rate=0.05)
    assert len(res["tickers"]) == 4
    assert "max_sharpe_portfolio" in res
    assert "min_volatility_portfolio" in res
    assert len(res["efficient_frontier"]) > 0


def test_market_sentiment_engine():
    data = MarketSentimentEngine.get_market_sentiment("AAPL")
    assert data["ticker"] == "AAPL"
    assert len(data["sector_heatmap"]) == 6
    assert len(data["economic_calendar"]) == 4


def test_workspace_manager():
    w = WorkspaceManager.save_workspace("user1", "default", {"ticker": "AAPL"})
    assert w["user_id"] == "user1"

    a = WorkspaceManager.create_alert("AAPL", "RSI_ABOVE", 70.0, user_id="user1")
    assert a["status"] == "ACTIVE"

    alerts = WorkspaceManager.get_alerts("user1")
    assert len(alerts) >= 1


def test_v7_explain_endpoint():
    client = flask_app.test_client()
    resp = client.post('/api/v7/ai/explain', json={
        'ticker': 'AAPL',
        'regime': 'Momentum Breakout',
        'signal': 'UP',
        'confidence': 0.82
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert data['ticker'] == 'AAPL'
    assert 'feature_attributions' in data


def test_v7_portfolio_optimize_endpoint():
    client = flask_app.test_client()
    resp = client.post('/api/v7/portfolio/optimize', json={
        'tickers': ['AAPL', 'MSFT', 'GOOGL', 'SPY'],
        'risk_free_rate': 0.05
    })
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'efficient_frontier' in data
    assert 'max_sharpe_portfolio' in data


def test_v7_market_intelligence_endpoint():
    client = flask_app.test_client()
    resp = client.get('/api/v7/market/intelligence?ticker=AAPL')
    assert resp.status_code == 200
    data = resp.get_json()
    assert 'sector_heatmap' in data
    assert 'economic_calendar' in data
