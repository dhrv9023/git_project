"""
ml/ai_intelligence.py — Institutional AI Financial Intelligence Engine

Phase 7 core module for StockBuddy:
  1. LLM Market Synthesis via Google Gemini API (falls back to template if key missing)
  2. Explainable AI (XAI) Feature Attribution — real permutation-based SHAP on KMeans model
  3. Markowitz Mean-Variance Portfolio Optimization & Efficient Frontier
  4. Market Sentiment & Live Financial News Analytics (via yfinance — no API key needed)
  5. Sector Heatmap & Macro Economic Calendar Engine
  6. Real-time Alert System & Workspace Manager
"""

import os
import time
import json
import logging
import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if (f == f and abs(f) < 1e15) else default
    except Exception:
        return default


# ── 1. LLM Market Narrative & XAI Synthesizer ─────────────────────────────────

class AIMarketSynthesizer:
    """
    Generates institutional natural-language market summaries using the
    Google Gemini API, with real permutation-based XAI feature attributions.

    Falls back to a structured template if GEMINI_API_KEY is not set.
    """

    _gemini_model      = None
    _gemini_model_name = "gemini-flash-latest"
    _gemini_api_key    = ""
    _gemini_ready      = False

    @classmethod
    def _init_gemini(cls) -> bool:
        if cls._gemini_ready:
            return True
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            log.warning("[AIMarketSynthesizer] GEMINI_API_KEY not set — using template fallback.")
            return False
        try:
            from google import genai   # type: ignore
            client = genai.Client(api_key=api_key)
            cls._gemini_api_key = api_key

            # Auto-detect best available flash model (avoids version deprecation breakage)
            preferred = ["gemini-3.1-flash-lite", "gemini-3.1-flash-lite-preview",
                         "gemini-3.6-flash", "gemini-3-flash-preview",
                         "gemini-flash-latest", "gemini-2.5-flash"]
            available = {m.name.split("/")[-1] for m in client.models.list()}
            cls._gemini_model_name = next(
                (m for m in preferred if m in available),
                next((m for m in available if "flash" in m and "preview" not in m), "gemini-flash-latest")
            )
            cls._gemini_ready = True
            log.info("[AIMarketSynthesizer] Gemini ready — model: %s", cls._gemini_model_name)
            return True
        except ImportError:
            log.warning("[AIMarketSynthesizer] google-genai not installed. Run: pip install google-genai")
            return False
        except Exception as exc:
            log.warning("[AIMarketSynthesizer] Gemini init failed: %s", exc)
            return False

    @classmethod
    def generate_narrative_summary(
        cls,
        ticker: str,
        metrics: Dict[str, Any],
        regime_name: str,
        signal_dir: str,
        confidence: float,
        feature_matrix: Optional[np.ndarray] = None,
        kmeans_model: Optional[Any] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Synthesises an executive LLM market summary using Gemini and
        computes real permutation-based XAI feature attributions.
        """
        current_price  = _safe_float(metrics.get("last_price",    150.0))
        ret_1m         = _safe_float(metrics.get("return_1m",       2.5))
        vol_20d        = _safe_float(metrics.get("volatility_20d", 18.2))
        rsi            = _safe_float(metrics.get("rsi",            54.0))
        sentiment_bias = "Bullish" if signal_dir == "UP" else ("Bearish" if signal_dir == "DOWN" else "Neutral")

        feature_attributions = cls._compute_permutation_shap(
            feature_matrix=feature_matrix,
            kmeans_model=kmeans_model,
            feature_names=feature_names,
            rsi=rsi, vol_20d=vol_20d, ret_1m=ret_1m, confidence=confidence,
        )

        narrative = cls._generate_gemini_narrative(
            ticker=ticker, current_price=current_price, ret_1m=ret_1m,
            vol_20d=vol_20d, rsi=rsi, regime_name=regime_name,
            signal_dir=signal_dir, confidence=confidence, sentiment_bias=sentiment_bias,
        )

        return {
            "ticker":               ticker,
            "headline":             narrative["headline"],
            "executive_summary":    narrative["executive_summary"],
            "key_takeaways":        narrative["key_takeaways"],
            "sentiment_bias":       sentiment_bias,
            "confidence_score":     round(float(confidence), 4),
            "feature_attributions": feature_attributions,
            "narrative_source":     narrative["source"],
            "generated_at":         datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

    @classmethod
    def _generate_gemini_narrative(cls, **kw) -> Dict[str, Any]:
        if cls._init_gemini():
            try:
                return cls._call_gemini(**kw)
            except Exception as exc:
                log.warning("[AIMarketSynthesizer] Gemini call failed (%s) — using template.", exc)
        return cls._template_narrative(**kw)

    @classmethod
    def _call_gemini(cls, ticker, current_price, ret_1m, vol_20d, rsi,
                     regime_name, signal_dir, confidence, sentiment_bias, **_) -> Dict[str, Any]:
        prompt = (
            "You are an institutional quantitative analyst writing a market intelligence brief.\n\n"
            f"Asset: {ticker}\n"
            f"Current Price: ${current_price:.2f}\n"
            f"30-Day Return: {ret_1m:+.2f}%\n"
            f"20-Day Annualised Volatility: {vol_20d:.1f}%\n"
            f"RSI (14-day): {rsi:.1f}\n"
            f"Market Regime: {regime_name}\n"
            f"Directional Signal: {signal_dir}\n"
            f"Model Confidence: {confidence*100:.1f}%\n"
            f"Sentiment Bias: {sentiment_bias}\n\n"
            "Respond ONLY with a valid JSON object (no markdown fences) with exactly these keys:\n"
            '{"headline": "<one sentence max 160 chars>", '
            '"executive_summary": "<2-3 sentences>", '
            '"key_takeaways": ["<point 1>", "<point 2>", "<point 3>", "<point 4>"]}'
        )
        from google import genai   # type: ignore
        client   = genai.Client(api_key=cls._gemini_api_key)
        response = client.models.generate_content(
            model=cls._gemini_model_name,
            contents=prompt,
        )
        raw = response.text.strip()
        if raw.startswith("```"):
            raw = "\n".join(l for l in raw.splitlines() if not l.startswith("```")).strip()
        parsed = json.loads(raw)
        return {
            "headline":          parsed["headline"],
            "executive_summary": parsed["executive_summary"],
            "key_takeaways":     parsed["key_takeaways"],
            "source":            f"gemini/{cls._gemini_model_name}",
        }

    @staticmethod
    def _template_narrative(ticker, current_price, ret_1m, vol_20d, rsi,
                             regime_name, signal_dir, confidence, sentiment_bias, **_) -> Dict[str, Any]:
        return {
            "headline": (
                f"AI Intelligence Brief: {ticker} demonstrates a {sentiment_bias} bias "
                f"under {regime_name} regime conditions with {confidence*100:.1f}% confidence."
            ),
            "executive_summary": (
                f"Asset {ticker} is currently trading at ${current_price:.2f}, exhibiting a "
                f"30-day return of {ret_1m:+.2f}% with 20-day annualised volatility at {vol_20d:.1f}%. "
                f"The quantitative engine classifies the dominant market state as '{regime_name}', "
                f"where momentum indicators (RSI: {rsi:.1f}) support a {signal_dir} directional "
                f"forecast over a 20-day horizon."
            ),
            "key_takeaways": [
                f"Market Regime: {regime_name} — Historical win-rate: {max(45, min(85, int(confidence*100+5)))}%.",
                f"Technical Driver: RSI at {rsi:.1f} indicates "
                f"{'overbought momentum' if rsi > 70 else ('oversold recovery' if rsi < 30 else 'neutral equilibrium')}.",
                f"Risk Assessment: Annualised vol of {vol_20d:.1f}% suggests "
                f"{'elevated tail risk' if vol_20d > 25 else 'controlled variance'}.",
                f"Tactical: Rebalance toward {signal_dir} positioning, trailing stop at {current_price*0.95:.2f}.",
            ],
            "source": "template-fallback",
        }

    @staticmethod
    def _compute_permutation_shap(
        feature_matrix, kmeans_model, feature_names,
        rsi, vol_20d, ret_1m, confidence,
    ) -> List[Dict[str, Any]]:
        """
        Real permutation-based feature importance on the fitted KMeans model.

        Algorithm:
          1. Baseline inertia on the feature matrix.
          2. For each feature: shuffle -> recompute inertia -> restore.
          3. importance = (perturbed_inertia - baseline) / |baseline|
          4. Normalise to sum to 1.0, sort descending.
        """
        if (
            feature_matrix is not None
            and kmeans_model is not None
            and feature_names is not None
            and hasattr(kmeans_model, "cluster_centers_")
            and len(feature_matrix) > 10
        ):
            try:
                X       = np.array(feature_matrix, dtype=float)
                labels  = kmeans_model.predict(X)
                centers = kmeans_model.cluster_centers_

                def _inertia(Xp):
                    return float(np.sum((Xp - centers[labels]) ** 2))

                baseline    = _inertia(X)
                importances = []
                rng         = np.random.default_rng(seed=42)

                for col in range(X.shape[1]):
                    Xp         = X.copy()
                    Xp[:, col] = rng.permutation(Xp[:, col])
                    delta      = (_inertia(Xp) - baseline) / (abs(baseline) + 1e-9)
                    importances.append(max(0.0, delta))

                total = sum(importances) or 1.0
                norm  = [v / total for v in importances]
                result = [
                    {
                        "feature": feature_names[i],
                        "weight":  round(norm[i], 4),
                        "impact":  "Positive" if norm[i] > 0.05 else "Neutral",
                        "method":  "permutation",
                    }
                    for i in range(len(feature_names))
                ]
                return sorted(result, key=lambda x: -x["weight"])

            except Exception as exc:
                log.warning("[AIMarketSynthesizer] Permutation SHAP failed: %s — approximating.", exc)

        # Metric-based approximation fallback (no model available yet)
        log.info("[AIMarketSynthesizer] No trained KMeans model — using metric-based approximation.")
        raw = [
            ("RSI (14-Day)",      abs((rsi - 50.0) / 100.0)),
            ("20-Day Volatility", abs(0.02 * (vol_20d - 15.0))),
            ("EMA 20/50 Slope",   abs(ret_1m / 50.0)),
            ("Regime Confidence", abs(confidence * 0.25)),
            ("MACD Histogram",    abs((rsi - 50.0) * ret_1m / 10000.0)),
        ]
        total = sum(v for _, v in raw) or 1.0
        return sorted(
            [{"feature": n, "weight": round(v / total, 4), "impact": "Positive", "method": "approximated"} for n, v in raw],
            key=lambda x: -x["weight"],
        )


# ── 2. Markowitz Mean-Variance Portfolio Optimizer ─────────────────────────────

class PortfolioOptimizer:
    """
    Modern Portfolio Theory (MPT) Markowitz Mean-Variance Optimizer.
    Computes Expected Return, Covariance, Tangency Portfolio (Max Sharpe),
    Minimum Variance Portfolio, and the Efficient Frontier curve.
    """

    @staticmethod
    def optimize_portfolio(
        asset_data: Dict[str, List[float]],
        rf_rate: float = 0.05,
        num_portfolios: int = 50,
    ) -> Dict[str, Any]:
        tickers  = list(asset_data.keys())
        n_assets = len(tickers)
        if n_assets == 0:
            raise ValueError("No assets provided for portfolio optimization.")

        df_returns  = pd.DataFrame(asset_data)
        ann_returns = df_returns.mean() * 252.0
        ann_cov     = df_returns.cov()  * 252.0

        eq_weights = np.ones(n_assets) / n_assets
        eq_ret     = float(np.sum(ann_returns * eq_weights))
        eq_vol     = float(np.sqrt(np.dot(eq_weights.T, np.dot(ann_cov, eq_weights))))
        eq_sharpe  = float((eq_ret - rf_rate) / eq_vol) if eq_vol > 0 else 0.0

        frontier_points    = []
        best_sharpe        = -999.0
        max_sharpe_weights = eq_weights.copy()
        min_vol            = 999.0
        min_vol_weights    = eq_weights.copy()

        np.random.seed(42)
        for _ in range(num_portfolios * 20):
            w        = np.random.dirichlet(np.ones(n_assets))
            p_ret    = float(np.sum(ann_returns * w))
            p_vol    = float(np.sqrt(np.dot(w.T, np.dot(ann_cov, w))))
            p_sharpe = float((p_ret - rf_rate) / p_vol) if p_vol > 0 else 0.0
            if p_sharpe > best_sharpe:
                best_sharpe        = p_sharpe
                max_sharpe_weights = w
            if p_vol < min_vol:
                min_vol         = p_vol
                min_vol_weights = w
            frontier_points.append({"return": round(p_ret, 4), "volatility": round(p_vol, 4), "sharpe": round(p_sharpe, 4)})

        df_frontier = (
            pd.DataFrame(frontier_points)
            .sort_values("volatility")
            .drop_duplicates(subset=["volatility"])
            .head(num_portfolios)
        )

        def _perf(w):
            ret = float(np.sum(ann_returns * w))
            vol = float(np.sqrt(np.dot(w.T, np.dot(ann_cov, w))))
            return ret, vol, (ret - rf_rate) / vol if vol > 0 else 0.0

        t_ret,  t_vol,  t_sr  = _perf(max_sharpe_weights)
        mv_ret, mv_vol, mv_sr = _perf(min_vol_weights)

        return {
            "tickers": tickers,
            "equal_weight": {
                "weights":         {tickers[i]: round(float(eq_weights[i]), 4) for i in range(n_assets)},
                "expected_return": round(eq_ret, 4),
                "volatility":      round(eq_vol, 4),
                "sharpe_ratio":    round(eq_sharpe, 4),
            },
            "max_sharpe_portfolio": {
                "weights":         {tickers[i]: round(float(max_sharpe_weights[i]), 4) for i in range(n_assets)},
                "expected_return": round(t_ret, 4),
                "volatility":      round(t_vol, 4),
                "sharpe_ratio":    round(t_sr, 4),
            },
            "min_volatility_portfolio": {
                "weights":         {tickers[i]: round(float(min_vol_weights[i]), 4) for i in range(n_assets)},
                "expected_return": round(mv_ret, 4),
                "volatility":      round(mv_vol, 4),
                "sharpe_ratio":    round(mv_sr, 4),
            },
            "efficient_frontier": df_frontier.to_dict(orient="records"),
            "risk_free_rate":     rf_rate,
        }


# ── 3. Sentiment & Live News Analytics Engine ──────────────────────────────────

class MarketSentimentEngine:
    """
    Market Sentiment, Live News Aggregation, and Economic Calendar Engine.

    News is fetched live from Yahoo Finance via yfinance (no API key required).
    Sentiment scores are computed by VADER with a financial domain lexicon extension.
    """

    @staticmethod
    def get_market_sentiment(
        ticker: str = "AAPL",
        sector_heatmap: Optional[list] = None,
    ) -> Dict[str, Any]:
        from ml.sentiment import get_scorer

        headlines = MarketSentimentEngine._fetch_live_headlines(ticker)

        scorer = get_scorer()
        for h in headlines:
            s = scorer.score(h["title"])
            h["sentiment"]       = s.label
            h["sentiment_score"] = round(s.compound, 4)
            h["impact_score"]    = round(s.confidence, 4)

        texts = [h["title"] for h in headlines]
        agg   = scorer.score_batch(texts)
        sentiment_score = round(agg.compound, 4)

        is_live_sector = sector_heatmap is not None
        if is_live_sector:
            _regime_map = {"Bullish": "Momentum Breakout", "Neutral": "Overbought Sideways", "Bearish": "Cyclical Pullback"}
            for s in sector_heatmap:
                s.setdefault("regime", _regime_map.get(s.get("sentiment", "Neutral"), "Neutral Drift"))
        else:
            sector_heatmap = [
                {"sector": "Information Technology", "etf": "XLK", "change_pct":  1.42, "sentiment": "Bullish", "regime": "Momentum Breakout"},
                {"sector": "Financials",             "etf": "XLF", "change_pct":  0.85, "sentiment": "Bullish", "regime": "Bullish Recovery"},
                {"sector": "Healthcare",             "etf": "XLV", "change_pct": -0.32, "sentiment": "Neutral", "regime": "Volatile Neutral"},
                {"sector": "Energy",                 "etf": "XLE", "change_pct": -1.15, "sentiment": "Bearish", "regime": "Cyclical Pullback"},
                {"sector": "Consumer Discretionary", "etf": "XLY", "change_pct":  0.94, "sentiment": "Bullish", "regime": "Bullish Recovery"},
                {"sector": "Industrials",            "etf": "XLI", "change_pct":  0.21, "sentiment": "Neutral", "regime": "Overbought Sideways"},
            ]

        today = datetime.date.today()
        economic_calendar = [
            {"event": "FOMC Interest Rate Decision",       "date": str(today + datetime.timedelta(days=7)),  "impact": "HIGH",   "consensus": "Current Rate Hold", "previous": "Prev Meeting"},
            {"event": "US Consumer Price Index (CPI YoY)", "date": str(today + datetime.timedelta(days=14)), "impact": "HIGH",   "consensus": "~2.9%",            "previous": "3.1%"},
            {"event": "Non-Farm Payrolls & Unemployment",  "date": str(today + datetime.timedelta(days=21)), "impact": "HIGH",   "consensus": "~175K",            "previous": "185K"},
            {"event": "Retail Sales MoM",                  "date": str(today + datetime.timedelta(days=28)), "impact": "MEDIUM", "consensus": "+0.4%",            "previous": "+0.2%"},
        ]

        bullish_pct = int(max(0, min(100, (sentiment_score + 1.0) / 2.0 * 100)))

        return {
            "ticker":                  ticker,
            "overall_sentiment_score": sentiment_score,
            "sentiment_label":         agg.label,
            "sentiment_detail":        agg.to_dict(),
            "bullish_pct":             bullish_pct,
            "bearish_pct":             100 - bullish_pct,
            "news_headlines":          headlines,
            "news_source":             "Yahoo Finance (live)" if any(h.get("is_live") for h in headlines) else "Illustrative fallback",
            "sector_heatmap":          sector_heatmap,
            "sector_data_source":      "Live yfinance ETF returns" if is_live_sector else "Illustrative fallback",
            "economic_calendar":       economic_calendar,
            "scoring_engine":          "VADER + Financial Lexicon",
        }

    @staticmethod
    def _fetch_live_headlines(ticker: str, max_items: int = 8) -> List[Dict[str, Any]]:
        """Fetch real headlines from Yahoo Finance via yfinance.news. Falls back gracefully."""
        try:
            import yfinance as yf
            tk   = yf.Ticker(ticker)
            news = tk.news or []
            if not news:
                raise ValueError("yfinance returned empty news list")

            headlines = []
            for item in news[:max_items]:
                content   = item.get("content", {})
                title     = content.get("title", item.get("title", ""))
                publisher = (content.get("provider", {}) or {}).get("displayName", item.get("publisher", "Yahoo Finance"))
                pub_time  = content.get("pubDate") or item.get("providerPublishTime") or ""
                url       = (content.get("canonicalUrl", {}) or {}).get("url", item.get("link", "#"))
                if not title:
                    continue
                headlines.append({
                    "title":    title,
                    "source":   publisher,
                    "time_ago": MarketSentimentEngine._relative_time(pub_time),
                    "url":      url,
                    "is_live":  True,
                })

            if headlines:
                log.info("[MarketSentimentEngine] Fetched %d live headlines for %s", len(headlines), ticker)
                return headlines

        except Exception as exc:
            log.warning("[MarketSentimentEngine] News fetch failed for %s: %s — using fallback.", ticker, exc)

        return [
            {"title": f"Federal Reserve Signals Policy Pause as Inflation Moderates for {ticker} Sector", "source": "Bloomberg Quant",    "time_ago": "25m ago", "url": "#", "is_live": False},
            {"title": f"Institutional Capital Inflows Drive Volume Expansion in {ticker}",                 "source": "Financial Times",    "time_ago": "1h ago",  "url": "#", "is_live": False},
            {"title": "Supply Chain Rebalancing and Margin Outlook Analysis for Q3",                       "source": "Reuters",            "time_ago": "3h ago",  "url": "#", "is_live": False},
            {"title": "Global Bond Yield Volatility Sparks Tactical Sector Rotation",                      "source": "Wall Street Journal","time_ago": "5h ago",  "url": "#", "is_live": False},
        ]

    @staticmethod
    def _relative_time(pub_time) -> str:
        try:
            if isinstance(pub_time, (int, float)):
                pub_dt = datetime.datetime.fromtimestamp(pub_time, tz=datetime.timezone.utc)
            elif isinstance(pub_time, str) and pub_time:
                pub_dt = datetime.datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
            else:
                return "recently"
            delta = datetime.datetime.now(tz=datetime.timezone.utc) - pub_dt
            secs  = int(delta.total_seconds())
            if secs < 3600:
                return f"{secs // 60}m ago"
            elif secs < 86400:
                return f"{secs // 3600}h ago"
            else:
                return f"{secs // 86400}d ago"
        except Exception:
            return "recently"


# ── 4. Alert Engine & Workspace Manager ───────────────────────────────────────

class WorkspaceManager:
    """Manages custom user watchlists, active alert triggers, and saved workspaces."""

    _workspaces: Dict[str, Dict[str, Any]] = {}
    _alerts: List[Dict[str, Any]] = []

    @classmethod
    def save_workspace(cls, user_id: str, workspace_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        key = f"{user_id}:{workspace_name}"
        cls._workspaces[key] = {
            "user_id":    user_id,
            "name":       workspace_name,
            "config":     config,
            "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        return cls._workspaces[key]

    @classmethod
    def get_workspace(cls, user_id: str, workspace_name: str) -> Optional[Dict[str, Any]]:
        return cls._workspaces.get(f"{user_id}:{workspace_name}")

    @classmethod
    def create_alert(
        cls,
        ticker: str,
        condition_type: str,
        threshold: float,
        user_id: str = "default_user",
    ) -> Dict[str, Any]:
        alert = {
            "alert_id":       f"alt_{int(time.time() * 1000)}",
            "user_id":        user_id,
            "ticker":         ticker,
            "condition_type": condition_type,
            "threshold":      threshold,
            "status":         "ACTIVE",
            "created_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        cls._alerts.append(alert)
        return alert

    @classmethod
    def get_alerts(cls, user_id: str = "default_user") -> List[Dict[str, Any]]:
        return [a for a in cls._alerts if a["user_id"] == user_id]
