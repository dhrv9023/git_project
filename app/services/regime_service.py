"""
app/services/regime_service.py — Market regime classification service.

Engineering decisions:
  - Receives MarketDataRepository via constructor injection.
  - Returns RegimeResult domain object — no Flask coupling.
  - All regime logic (KMeans, risk scoring, stats) lives here, not in routes.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from app.domain.models import RegimeResult
from app.repositories.market_data_repo import IMarketDataRepository
from core.config import AppConfig

log = logging.getLogger(__name__)

REGIME_PROFILES = {
    0: {"name": "Trending Bull",          "color": "#00ff88", "description": "Strong uptrend with aligned momentum. Historical bias is bullish."},
    1: {"name": "Overbought / Exhaustion","color": "#ffaa00", "description": "Extended rally. RSI elevated and momentum diverging — pullback risk."},
    2: {"name": "Sideways / Choppy",      "color": "#888888", "description": "No clear directional bias. Trend-following strategies underperform."},
    3: {"name": "Recovery / Bounce",      "color": "#00aaff", "description": "Recovering from oversold condition. Mean-reversion setup forming."},
    4: {"name": "Downtrend / Bear",       "color": "#ff3366", "description": "Clear downtrend in progress. Momentum is bearish."},
    5: {"name": "High Volatility / Stress","color": "#ff6600", "description": "Elevated fear and regime instability. Reduce risk exposure."},
}


class RegimeService:
    """Classifies market regimes and computes risk scores.

    Engineering decision: KMeans is fitted fresh on each request because
    regime boundaries shift as new data arrives. Caching the cluster model
    would introduce stale-regime risk. The computation is fast (~50ms) so
    request-time fitting is acceptable.
    """

    def __init__(
        self,
        market_repo: IMarketDataRepository,
        cfg: AppConfig,
    ) -> None:
        self.market_repo = market_repo
        self.cfg = cfg

    def classify(self, ticker: str, start: str, end: str) -> RegimeResult:
        """Full regime analysis pipeline.

        Raises:
            DataFetchError: upstream yfinance failure.
            InsufficientDataError: not enough rows for clustering.
        """
        data = self.market_repo.build_feature_matrix(ticker, start, end, self.cfg.sequence_length)
        df = data["df"]

        labels, feat = self._classify_regimes(df)
        df_aligned = df.loc[df.index.intersection(labels.index)]
        risk_scores = self._compute_risk_score(df_aligned, labels)
        risk_scores = risk_scores.reindex(labels.index).ffill().fillna(5.0)
        regime_stats = self._compute_regime_stats(df_aligned, labels)

        current_regime_id = int(labels.iloc[-1])
        current_risk_score = float(risk_scores.iloc[-1] or 5.0)
        current_alert = self._get_condition_alert(current_risk_score, current_regime_id)
        current_profile = REGIME_PROFILES[current_regime_id]
        latest_row = df.iloc[-1]

        def _s(v):
            import math
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v

        indicators = {
            "rsi": _s(latest_row.get("RSI14")),
            "ema20": _s(latest_row.get("EMA20")),
            "macd": _s(latest_row.get("MACD")),
            "macd_hist": _s(latest_row.get("MACD_hist")),
            "close": _s(latest_row.get("Close")),
            "logret": _s(latest_row.get("LogReturn")),
        }

        timeline_dates = labels.index
        sample_step = max(1, len(timeline_dates) // 500)
        sampled_idx = list(range(0, len(timeline_dates), sample_step))
        if sampled_idx[-1] != len(timeline_dates) - 1:
            sampled_idx.append(len(timeline_dates) - 1)

        price_close = df_aligned["Close"]
        timeline = []
        for i in sampled_idx:
            d = timeline_dates[i]
            rid = int(labels.iloc[i])
            timeline.append({
                "date": d.strftime("%Y-%m-%d"),
                "regime_id": rid,
                "regime_name": REGIME_PROFILES[rid]["name"],
                "regime_color": REGIME_PROFILES[rid]["color"],
                "risk_score": _s(float(risk_scores.iloc[i])),
                "close": _s(float(price_close.get(d, np.nan))),
            })

        stats_out = {}
        for rid, s in regime_stats.items():
            def _ss(stat): return {k: (_s(v) if not isinstance(v, (list, dict)) else v) for k, v in stat.items()}
            stats_out[str(rid)] = {
                "id": s["id"], "name": s["name"], "color": s["color"],
                "description": s["description"], "total_days": s["total_days"],
                "fwd_5d": _ss(s["fwd_5d"]),
                "fwd_10d": _ss(s["fwd_10d"]),
                "fwd_20d": _ss(s["fwd_20d"]),
            }

        from app.services.backtest_service import BacktestService
        bt_svc = BacktestService(self.cfg)
        quant_backtest = bt_svc.run_regime_backtest(df_aligned["Close"], labels)
        similar_matches = self._find_similar_scenarios(df_aligned, feat, labels)

        return RegimeResult(
            ticker=ticker,
            current_regime={
                "id": current_regime_id,
                "name": current_profile["name"],
                "color": current_profile["color"],
                "description": current_profile["description"],
            },
            risk_score=current_risk_score,
            alert=current_alert,
            indicators=indicators,
            regime_stats=stats_out,
            timeline=timeline,
            similar_matches=similar_matches,
            quant_backtest=quant_backtest,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Vol20"] = out["LogReturn"].rolling(20).std()
        out["RSI_dev"] = out["RSI14"] - 50.0
        out["MACD_hist_norm"] = out["MACD_hist"] / (out["Close"].rolling(20).mean().replace(0, 1) * 0.01 + 1e-9)
        out["EMA_slope"] = (out["EMA20"] - out["EMA20"].shift(10)) / (out["EMA20"].shift(10).replace(0, 1) + 1e-9) * 100.0
        out["Ret10"] = out["Close"].pct_change(10) * 100.0
        cols = ["RSI_dev", "MACD_hist_norm", "EMA_slope", "Vol20", "Ret10"]
        return out[cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

    def _classify_regimes(self, df: pd.DataFrame, n_clusters: int = 6):
        feat = self._build_regime_features(df)
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat.values)
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        raw_labels = km.fit_predict(feat_scaled)

        centers = km.cluster_centers_
        ret10_idx = list(feat.columns).index("Ret10")
        rsi_idx = list(feat.columns).index("RSI_dev")
        vol_idx = list(feat.columns).index("Vol20")
        scores = 2.0 * centers[:, ret10_idx] + 1.5 * centers[:, rsi_idx] - 3.0 * centers[:, vol_idx]
        sorted_clusters = np.argsort(-scores)
        mapping = {int(orig): int(sem) for sem, orig in enumerate(sorted_clusters)}
        semantic_labels = np.array([mapping[l] for l in raw_labels], dtype=int)
        return pd.Series(semantic_labels, index=feat.index, name="regime"), feat

    def _compute_risk_score(self, df: pd.DataFrame, labels: pd.Series) -> pd.Series:
        df = df.copy()
        df["Vol20"] = df["LogReturn"].rolling(20).std()
        df["Vol90"] = df["LogReturn"].rolling(90).std()
        vol_ratio = (df["Vol20"] / df["Vol90"].replace(0, 1e-9)).clip(0.5, 3.0)
        vol_stress = ((vol_ratio - 0.5) / 2.5 * 3.0).clip(0, 3.0)
        rsi_extremity = ((df["RSI14"] - 50).abs() / 50.0 * 3.0).clip(0, 3.0)
        trend_align = np.where(df["Close"] < df["EMA20"], 2.0, 0.0)
        macd_div = np.where(df["MACD_hist"] < 0, 2.0, 0.0)
        raw = vol_stress.values + rsi_extremity.values + trend_align + macd_div
        score = pd.Series(np.clip(raw, 0, 10), index=df.index).rolling(5).mean()
        if labels is not None:
            common = score.index.intersection(labels.index)
            for idx in common:
                if labels[idx] in (4, 5):
                    score[idx] = max(score[idx], 7.5)
        return score.clip(0, 10)

    def _compute_regime_stats(self, df: pd.DataFrame, labels: pd.Series) -> dict:
        MAX_HORIZON = 20
        common_idx = df.index.intersection(labels.index)
        close = df.loc[common_idx, "Close"]
        eligible_cutoff = close.index[-MAX_HORIZON] if len(close) > MAX_HORIZON else close.index[0]
        stats = {}
        for regime_id in range(6):
            regime_dates = labels[labels == regime_id].index
            regime_dates = regime_dates.intersection(close.index)
            regime_dates = regime_dates[regime_dates < eligible_cutoff]
            fwd_5, fwd_10, fwd_20 = [], [], []
            for d in regime_dates:
                loc = close.index.get_loc(d)
                for horizon, store in [(5, fwd_5), (10, fwd_10), (20, fwd_20)]:
                    end_loc = loc + horizon
                    if end_loc < len(close):
                        store.append(float((close.iloc[end_loc] / close.iloc[loc] - 1.0) * 100.0))

            def _summarise(lst):
                if not lst:
                    return {"median": None, "mean": None, "pct_positive": None, "count": 0, "samples": []}
                arr = np.array(lst)
                return {"median": float(np.median(arr)), "mean": float(np.mean(arr)),
                        "pct_positive": float((arr > 0).mean() * 100), "count": len(arr),
                        "samples": [round(v, 2) for v in arr[-10:].tolist()]}

            profile = REGIME_PROFILES.get(regime_id, {"name": f"Regime {regime_id}", "color": "#888", "description": ""})
            stats[regime_id] = {
                "id": regime_id, "name": profile["name"], "color": profile["color"],
                "description": profile["description"], "total_days": len(regime_dates),
                "fwd_5d": _summarise(fwd_5), "fwd_10d": _summarise(fwd_10), "fwd_20d": _summarise(fwd_20),
            }
        return stats

    def _find_similar_scenarios(self, df: pd.DataFrame, feat: pd.DataFrame, labels: pd.Series, top_k: int = 5) -> list:
        from sklearn.metrics.pairwise import cosine_similarity
        if len(feat) < 30:
            return []
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat.values)
        target_vec = feat_scaled[-1].reshape(1, -1)
        cand_indices = range(0, len(feat_scaled) - 15)
        sims = cosine_similarity(target_vec, feat_scaled[cand_indices])[0]
        top_positions = np.argsort(-sims)[:top_k]
        matches = []
        close_series = df.loc[feat.index, "Close"]
        for pos in top_positions:
            idx = cand_indices[pos]
            matched_date = feat.index[idx]
            start_loc = close_series.index.get_loc(matched_date)
            fwd_prices = close_series.iloc[start_loc:start_loc + 11].tolist()
            fwd_ret = float((fwd_prices[-1] / fwd_prices[0] - 1.0) * 100.0) if len(fwd_prices) >= 2 else 0.0
            base = fwd_prices[0] if fwd_prices else 1.0
            normalized = [round(float(p / base * 100.0), 2) for p in fwd_prices]
            rid = int(labels.iloc[idx])
            profile = REGIME_PROFILES.get(rid, {"name": "Unknown", "color": "#888"})
            matches.append({
                "date": matched_date.strftime("%Y-%m-%d"),
                "similarity_pct": round(float(sims[pos] * 100.0), 1),
                "regime_name": profile["name"],
                "regime_color": profile["color"],
                "fwd_10d_ret": round(fwd_ret, 2),
                "price_path": normalized,
            })
        return matches

    @staticmethod
    def _get_condition_alert(risk_score: float, regime_id: int) -> dict:
        if regime_id in (0, 3) and risk_score < 4.5:
            return {"level": "GREEN", "label": "Favorable Conditions", "color": "#00ff88",
                    "description": "Regime and indicators aligned positively."}
        elif regime_id in (4, 5) or risk_score >= 7.0:
            return {"level": "RED", "label": "Elevated Risk", "color": "#ff3366",
                    "description": "High risk conditions. Historical odds favor caution."}
        else:
            return {"level": "YELLOW", "label": "Watch / Wait", "color": "#ffaa00",
                    "description": "Mixed signals. Wait for clearer confirmation."}
