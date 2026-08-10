"""
app/domain/exceptions.py — Domain exception hierarchy.

Engineering decisions:
  - All exceptions inherit from StockBuddyError so callers can catch the
    entire domain with a single except clause when needed.
  - Named subclasses give HTTP error handlers enough information to map
    exceptions to appropriate status codes without coupling domain code to Flask.
  - HTTP mapping lives in app/middleware/error_handlers.py, NOT here.
"""
from __future__ import annotations


class StockBuddyError(Exception):
    """Base exception for all StockBuddy domain errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def to_dict(self) -> dict:
        result = {"error": self.message}
        if self.detail:
            result["detail"] = self.detail
        return result


class DataFetchError(StockBuddyError):
    """Raised when market data cannot be retrieved from yfinance.

    Maps to HTTP 502 (Bad Gateway) — the upstream data provider failed.
    """


class InsufficientDataError(StockBuddyError):
    """Raised when the fetched data has too few rows for model training.

    Maps to HTTP 422 (Unprocessable Entity).
    """

    def __init__(self, required: int, available: int) -> None:
        super().__init__(
            f"Insufficient data: {available} rows available, {required} required.",
            detail=f"required={required} available={available}",
        )
        self.required = required
        self.available = available


class ModelNotFoundError(StockBuddyError):
    """Raised when no trained model exists for the requested ticker/version.

    Maps to HTTP 404.
    """

    def __init__(self, ticker: str, version: str = "best") -> None:
        super().__init__(
            f"No ready model found for {ticker} (version='{version}'). "
            "Submit a training job via POST /api/v2/train first.",
            detail=f"ticker={ticker} version={version}",
        )
        self.ticker = ticker
        self.version = version


class TrainingError(StockBuddyError):
    """Raised when a model training job fails.

    Maps to HTTP 500.
    """


class ConfigurationError(StockBuddyError):
    """Raised when required configuration is missing or invalid.

    Maps to HTTP 500.
    """


class CacheError(StockBuddyError):
    """Raised on inference cache read/write failures.

    Maps to HTTP 500 but is usually non-fatal (caller can proceed without cache).
    """
