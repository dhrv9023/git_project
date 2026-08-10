"""
app/domain/models.py — Result dataclasses for the service layer.

Engineering decisions:
  - Dataclasses chosen over TypedDict because they support default_factory,
    validation logic in __post_init__, and are easier to mock in tests.
  - to_dict() methods produce JSON-safe dicts (no NaN/Inf, all floats rounded).
  - Domain models decouple services from Flask's jsonify() — services return
    typed objects; routes call .to_dict() before jsonify().
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


def _safe(v: Any) -> Any:
    """Convert NaN/Inf floats to None for JSON serialisation."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _safe_list(lst: list) -> list:
    return [_safe(v) for v in lst]


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    ticker: str
    version_used: str
    from_cache: bool
    cache_key: str
    dates: list[str]
    actual: list[float | None]
    predictions: dict[str, list[float | None]]
    confidence_lower: list[float | None]
    confidence_upper: list[float | None]
    metrics: dict[str, dict[str, float | None]]
    backtest_equity: list[float | None]
    backtest_buy_signals: list[float | None]
    backtest_sell_signals: list[float | None]
    backtest_metrics: dict[str, float | None]
    future_dates: list[str] = field(default_factory=list)
    future_predictions: list[float | None] = field(default_factory=list)
    histories: dict[str, Any] = field(default_factory=dict)
    model_path: str | None = None

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "version_used": self.version_used,
            "from_cache": self.from_cache,
            "cache_key": self.cache_key,
            "dates": self.dates,
            "actual": _safe_list(self.actual),
            "predictions": {k: _safe_list(v) for k, v in self.predictions.items()},
            "confidence": {
                "lower": _safe_list(self.confidence_lower),
                "upper": _safe_list(self.confidence_upper),
            },
            "metrics": {
                k: {mk: _safe(mv) for mk, mv in m.items()}
                for k, m in self.metrics.items()
            },
            "backtest": {
                "equity": _safe_list(self.backtest_equity),
                "buy_signals": _safe_list(self.backtest_buy_signals),
                "sell_signals": _safe_list(self.backtest_sell_signals),
                "metrics": {mk: _safe(mv) for mk, mv in self.backtest_metrics.items()},
            },
            "future": {
                "dates": self.future_dates,
                "predictions": _safe_list(self.future_predictions),
            },
            "histories": self.histories,
            "model_path": self.model_path,
        }


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    ticker: str
    current_regime: dict
    risk_score: float
    alert: dict
    indicators: dict[str, float | None]
    regime_stats: dict
    timeline: list[dict]
    similar_matches: list[dict]
    quant_backtest: dict

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "current_regime": self.current_regime,
            "risk_score": round(_safe(self.risk_score) or 5.0, 2),
            "alert": self.alert,
            "indicators": {k: _safe(v) for k, v in self.indicators.items()},
            "regime_stats": self.regime_stats,
            "timeline": self.timeline,
            "similar_matches": self.similar_matches,
            "quant_backtest": self.quant_backtest,
        }


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    model_type: str
    n_folds: int
    folds: list[dict]
    summary: dict[str, float]
    interpretation: str

    def to_dict(self) -> dict:
        return {
            "model": self.model_type,
            "n_folds": self.n_folds,
            "folds": self.folds,
            "summary": self.summary,
            "interpretation": self.interpretation,
        }
