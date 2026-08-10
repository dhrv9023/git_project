"""app/api/v7_routes.py — AI Intelligence & Portfolio platform."""
from __future__ import annotations
from flask import Blueprint, jsonify, request
from core.config import AppConfig


def make_v7_blueprint(market_repo, cfg: AppConfig) -> Blueprint:
    bp = Blueprint("v7", __name__, url_prefix="/api/v7")

    @bp.route("/ai/explain", methods=["POST"])
    def v7_ai_explain():
        body = request.get_json(silent=True) or {}
        ticker = body.get("ticker", "AAPL").upper().strip()
        regime = body.get("regime", "Momentum Breakout")
        signal = body.get("signal", "UP")
        confidence = float(body.get("confidence", 0.72))
        metrics = {
            "last_price": float(body.get("last_price", 182.50)),
            "return_1m": float(body.get("return_1m", 3.8)),
            "volatility_20d": float(body.get("volatility_20d", 16.4)),
            "rsi": float(body.get("rsi", 58.2)),
        }
        from ml.ai_intelligence import AIMarketSynthesizer
        return jsonify(AIMarketSynthesizer.generate_narrative_summary(ticker, metrics, regime, signal, confidence)), 200

    @bp.route("/portfolio/optimize", methods=["POST"])
    def v7_portfolio_optimize():
        import numpy as np
        body = request.get_json(silent=True) or {}
        tickers = body.get("tickers", ["AAPL", "MSFT", "GOOGL", "SPY"])
        rf_rate = float(body.get("risk_free_rate", 0.05))
        if isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if len(tickers) < 2:
            tickers = ["AAPL", "MSFT", "GOOGL", "SPY"]
        start_date = body.get("start_date", "2022-01-01")
        end_date = body.get("end_date", "2024-01-01")
        asset_returns = {}
        for t in tickers:
            try:
                df = market_repo.fetch_raw(t, start_date, end_date)
                import numpy as _np
                rets = _np.diff(_np.log(df["Close"].values))
                asset_returns[t] = rets.tolist()
            except Exception:
                np.random.seed(abs(hash(t)) % 1000)
                asset_returns[t] = np.random.normal(0.0005, 0.015, 252).tolist()
        from ml.ai_intelligence import PortfolioOptimizer
        return jsonify(PortfolioOptimizer.optimize_portfolio(asset_returns, rf_rate=rf_rate)), 200

    @bp.route("/market/intelligence", methods=["GET"])
    def v7_market_intelligence():
        ticker = request.args.get("ticker", "AAPL").upper().strip()
        from ml.ai_intelligence import MarketSentimentEngine
        return jsonify(MarketSentimentEngine.get_market_sentiment(ticker)), 200

    @bp.route("/alerts", methods=["GET", "POST"])
    def v7_alerts():
        from ml.ai_intelligence import WorkspaceManager
        if request.method == "POST":
            body = request.get_json(silent=True) or {}
            alert = WorkspaceManager.create_alert(
                body.get("ticker", "AAPL").upper(),
                body.get("condition_type", "RSI_ABOVE"),
                float(body.get("threshold", 70.0)),
            )
            return jsonify(alert), 201
        return jsonify({"alerts": WorkspaceManager.get_alerts()}), 200

    @bp.route("/auth/login", methods=["POST"])
    def v7_auth_login():
        body = request.get_json(silent=True) or {}
        username = body.get("username", "quant_user")
        token = f"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{username}.sb_v7_session"
        return jsonify({
            "status": "success", "token": token,
            "user": {"username": username, "role": "Institutional Quant",
                     "permissions": ["read", "write", "execute_models", "portfolio_opt"]},
        }), 200

    return bp
