"""
app/repositories/market_data_repo.py — Market data access layer.

Engineering decisions:
  - Implements IMarketDataRepository interface so tests can inject a mock.
  - All yfinance calls go through the circuit breaker defined in core/.
  - preprocess_data and engineer_features are delegated to ml.features,
    keeping this class thin (SRP: only data access + basic cleaning).
  - Raises domain exceptions (DataFetchError, InsufficientDataError) so
    callers never need to handle yfinance-specific errors.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd

from app.domain.exceptions import DataFetchError, InsufficientDataError
from ml.features import engineer_features, FEATURE_COLS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Interface (structural subtyping via Protocol)
# ---------------------------------------------------------------------------

class IMarketDataRepository(Protocol):
    def fetch_raw(self, ticker: str, start: str, end: str) -> pd.DataFrame: ...
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def build_feature_matrix(
        self,
        ticker: str,
        start: str,
        end: str,
        sequence_length: int,
    ) -> dict: ...


# ---------------------------------------------------------------------------
# Implementation
# ---------------------------------------------------------------------------

class MarketDataRepository:
    """Fetches, cleans, and engineers features from yfinance data.

    Engineering decision: the circuit breaker is applied at this layer —
    not in the service — because the circuit breaker is a data-access concern
    (protecting the upstream provider), not a business logic concern.
    """

    def fetch_raw(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Download OHLCV data from yfinance with circuit-breaker protection.

        Raises:
            DataFetchError: if yfinance returns empty data or raises.
        """
        try:
            import yfinance as yf
            df = yf.download(
                tickers=ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
        except Exception as exc:
            raise DataFetchError(
                f"yfinance download failed for {ticker}",
                detail=str(exc),
            ) from exc

        if df.empty:
            raise DataFetchError(
                f"No data returned for {ticker} between {start} and {end}. "
                "Check ticker symbol and date range."
            )

        # Flatten MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            try:
                df = df.xs(ticker, axis=1, level=-1)
            except Exception:
                df.columns = [
                    "_".join(str(x) for x in col if x is not None)
                    for col in df.columns
                ]

        # Normalise column names
        df = df.rename(columns={c: c.title() for c in df.columns})
        if "Close" not in df.columns and "Adj Close" in df.columns:
            df["Close"] = df["Adj Close"]

        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                df[col] = df["Close"] if col != "Volume" else 0.0

        return df[required].dropna()

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data: sort, dedup, fill gaps, remove outlier returns, winsorise."""
        cleaned = df.copy().sort_index()
        cleaned = cleaned[~cleaned.index.duplicated(keep="first")]
        cleaned = cleaned.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()

        returns = cleaned["Close"].pct_change()
        ret_no_na = returns.dropna()
        if not ret_no_na.empty:
            q1, q3 = ret_no_na.quantile([0.25, 0.75])
            iqr = q3 - q1
            mask = (returns.between(q1 - 3 * iqr, q3 + 3 * iqr)) | returns.isna()
            cleaned = cleaned.loc[mask]

        num_cols = cleaned.select_dtypes(include=[np.number]).columns
        cleaned[num_cols] = cleaned[num_cols].clip(
            lower=cleaned[num_cols].quantile(0.01),
            upper=cleaned[num_cols].quantile(0.99),
            axis=1,
        )
        return cleaned.dropna()

    def build_feature_matrix(
        self,
        ticker: str,
        start: str,
        end: str,
        sequence_length: int,
    ) -> dict:
        """Full pipeline: fetch → preprocess → engineer features → arrays.

        Returns dict with keys:
            X_raw, y_raw, dates_raw, base_prices_raw,
            sequence_length, feature_cols,
            close_feature_index, logret_feature_index

        Raises:
            InsufficientDataError: if fewer rows than 2 * sequence_length.
        """
        raw = self.fetch_raw(ticker, start, end)
        raw = self.preprocess(raw)
        df = engineer_features(raw)

        min_rows = sequence_length * 2 + 10
        if len(df) < min_rows:
            raise InsufficientDataError(required=min_rows, available=len(df))

        target_series = df["LogReturn"].shift(-1).dropna()
        df.columns = df.columns.astype(str)
        available_cols = [c for c in FEATURE_COLS if c in df.columns]

        X_df = df.loc[target_series.index, available_cols]
        base_prices = df.loc[target_series.index, "Close"]

        log.debug(
            "Feature matrix built: ticker=%s  shape=%s  rows=%d",
            ticker, X_df.shape, len(X_df),
        )

        return {
            "X_raw": X_df.values.astype("float32"),
            "y_raw": target_series.values.astype("float32"),
            "dates_raw": target_series.index,
            "base_prices_raw": base_prices.values,
            "df": df,
            "sequence_length": sequence_length,
            "feature_cols": available_cols,
            "close_feature_index": (
                available_cols.index("Close") if "Close" in available_cols else 0
            ),
            "logret_feature_index": (
                available_cols.index("LogReturn")
                if "LogReturn" in available_cols
                else len(available_cols) - 1
            ),
        }
