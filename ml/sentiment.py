"""
ml/sentiment.py — Real sentiment scoring using VADER + financial lexicon extension.

Engineering decisions:
  - VADER (Valence Aware Dictionary and sEntiment Reasoner) chosen over FinBERT
    because it runs on CPU with no GPU, installs in <1s, and scores at ~50k
    texts/second — suitable for real-time news tickers.
  - Extended with a custom financial domain lexicon: words like "bullish",
    "breakout", "recession", "downgrade" that VADER's generic lexicon misses.
  - SentimentScorer is a singleton (instantiated once, shared across requests)
    to avoid re-loading the lexicon on every call.
  - Recency-decay weighting: recent headlines have higher weight when computing
    the aggregate score.
  - Falls back to neutral (0.0) if vaderSentiment is not installed — never
    crashes the application.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Financial domain lexicon extension for VADER
# Positive words that VADER under-weights in a financial context
# ---------------------------------------------------------------------------

FINANCIAL_LEXICON: dict[str, float] = {
    # Strongly positive
    "bullish": 3.0, "breakout": 2.5, "outperform": 2.5, "upgrade": 2.5,
    "beat": 2.0, "record": 2.0, "surge": 2.5, "rally": 2.5,
    "momentum": 1.5, "recovery": 2.0, "rebound": 2.0, "growth": 1.5,
    "dividend": 1.5, "buyback": 1.5, "acquisition": 1.0,
    # Strongly negative
    "bearish": -3.0, "selloff": -3.0, "crash": -3.5, "recession": -3.0,
    "downgrade": -2.5, "miss": -2.0, "decline": -1.5, "layoffs": -2.5,
    "bankruptcy": -4.0, "default": -3.5, "fraud": -4.0, "investigation": -2.5,
    "warning": -2.0, "volatility": -1.0, "uncertainty": -1.5,
    "downside": -2.0, "headwinds": -2.0, "disappointing": -2.5,
    # Mildly positive
    "stable": 0.5, "resilient": 1.0, "confident": 1.0,
    # Mildly negative
    "concerns": -1.0, "risks": -0.5, "challenges": -0.5,
}


@dataclass
class SentimentScore:
    compound: float          # -1.0 (most negative) to +1.0 (most positive)
    positive: float          # fraction positive (0–1)
    negative: float          # fraction negative (0–1)
    neutral: float           # fraction neutral (0–1)
    label: str               # "Bullish" | "Bearish" | "Neutral"
    confidence: float        # abs(compound) — strength of signal

    def to_dict(self) -> dict:
        return {
            "compound": round(self.compound, 4),
            "positive": round(self.positive, 4),
            "negative": round(self.negative, 4),
            "neutral": round(self.neutral, 4),
            "label": self.label,
            "confidence": round(self.confidence, 4),
        }


class SentimentScorer:
    """VADER-based sentiment scorer with financial lexicon extension.

    Usage:
        scorer = SentimentScorer()
        score = scorer.score("Apple beats earnings expectations, raises guidance")
        # SentimentScore(compound=0.72, label='Bullish', ...)
    """

    _instance: "SentimentScorer | None" = None

    def __init__(self) -> None:
        self._vader = None
        self._available = False
        self._load()

    @classmethod
    def get(cls) -> "SentimentScorer":
        """Singleton accessor."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
            # Inject financial lexicon
            self._vader.lexicon.update(FINANCIAL_LEXICON)
            self._available = True
            log.info("SentimentScorer: VADER loaded with %d financial terms", len(FINANCIAL_LEXICON))
        except ImportError:
            log.warning(
                "vaderSentiment not installed — sentiment scoring disabled. "
                "Run: pip install vaderSentiment"
            )
            self._available = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, text: str) -> SentimentScore:
        """Score a single text string."""
        if not self._available or self._vader is None or not text.strip():
            return SentimentScore(0.0, 0.0, 0.0, 1.0, "Neutral", 0.0)

        scores = self._vader.polarity_scores(text)
        compound = float(scores["compound"])
        return SentimentScore(
            compound=compound,
            positive=float(scores["pos"]),
            negative=float(scores["neg"]),
            neutral=float(scores["neu"]),
            label=self._label(compound),
            confidence=abs(compound),
        )

    def score_batch(
        self,
        texts: Sequence[str],
        recency_weights: Sequence[float] | None = None,
    ) -> SentimentScore:
        """Aggregate sentiment across multiple texts with optional recency decay.

        Args:
            texts:           List of headline strings.
            recency_weights: Optional weights (higher = more recent).
                             Defaults to exponential decay: most recent gets
                             highest weight.

        Returns:
            Weighted aggregate SentimentScore.
        """
        if not texts:
            return SentimentScore(0.0, 0.0, 0.0, 1.0, "Neutral", 0.0)

        if recency_weights is None:
            # Exponential decay: index 0 = most recent
            n = len(texts)
            recency_weights = [math.exp(-0.3 * i) for i in range(n)]

        scores = [self.score(t) for t in texts]
        total_w = sum(recency_weights)
        if total_w == 0:
            total_w = 1.0

        w_compound = sum(s.compound * w for s, w in zip(scores, recency_weights)) / total_w
        w_pos = sum(s.positive * w for s, w in zip(scores, recency_weights)) / total_w
        w_neg = sum(s.negative * w for s, w in zip(scores, recency_weights)) / total_w
        w_neu = sum(s.neutral * w for s, w in zip(scores, recency_weights)) / total_w

        return SentimentScore(
            compound=w_compound,
            positive=w_pos,
            negative=w_neg,
            neutral=w_neu,
            label=self._label(w_compound),
            confidence=abs(w_compound),
        )

    @staticmethod
    def _label(compound: float) -> str:
        if compound >= 0.05:
            return "Bullish"
        if compound <= -0.05:
            return "Bearish"
        return "Neutral"


# ---------------------------------------------------------------------------
# Module-level singleton for use across the app
# ---------------------------------------------------------------------------

_scorer: SentimentScorer | None = None


def get_scorer() -> SentimentScorer:
    global _scorer
    if _scorer is None:
        _scorer = SentimentScorer()
    return _scorer


def score_text(text: str) -> dict:
    """Convenience function: score a single text and return dict."""
    return get_scorer().score(text).to_dict()


def score_headlines(headlines: list[str]) -> dict:
    """Convenience function: score a list of headlines with recency decay."""
    return get_scorer().score_batch(headlines).to_dict()
