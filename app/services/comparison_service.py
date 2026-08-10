"""
app/services/comparison_service.py — Multi-stock comparison service.

Engineering decisions:
  - ThreadPoolExecutor for parallel data fetching (one thread per ticker).
  - All tickers normalised to a base-100 index so price levels are comparable.
  - Pearson correlation matrix computed on LogReturn series (not price levels)
    to avoid spurious correlations from co-trending prices.
  - Returns a typed dict — routes call json.dumps on the result.
  - Raises DataFetchError if ALL tickers fail; partial failures are tolerated
    and flagged in the `errors` field so the dashboard can still render.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from app.domain.exceptions import DataFetchError, InsufficientDataError
from app.repositories.market_data_repo import MarketDataRepository
from core.config import AppConfig
from ml.features import engineer_features

log = logging.getLogger(__name__)

_MIN_ROWS = 30          # minimum rows needed for correlation to be meaningful
_MAX_TICKERS = 10       # hard cap to protect server resources
_NORMALISE_BASE = 100.0 # base value for normalised price index


class ComparisonService:
    """Fetches and compares multiple tickers in parallel.

    Constructor injection: receives MarketDataRepository so tests can
    inject a mock without any yfinance calls.
    """

    def __init__(self, market_repo: MarketDataRepository, cfg: AppConfig) -> None:
        self.market_repo = market_repo
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> dict[str, Any]:
        """Fetch, normalise, and compare multiple tickers.

        Args:
            tickers: List of ticker symbols (max 10).
            start:   ISO date string e.g. "2022-01-01".
            end:     ISO date string e.g. "2024-01-01".

        Returns:
            Dict with keys:
              - tickers:          list of successfully fetched tickers
              - errors:           dict mapping failed tickers to error messages
              - normalised_prices: {ticker: [{date, value}]} base-100 index
              - returns:          {ticker: [{date, ret_pct}]} daily % returns
              - correlation:      N×N correlation matrix of log returns
              - summary:          {ticker: {total_return, volatility, sharpe, max_drawdown, regime}}
              - date_range:       {start, end, trading_days}

        Raises:
            DataFetchError: if ALL tickers fail to fetch.
        """
        tickers = [t.upper().strip() for t in tickers[:_MAX_TICKERS] if t.strip()]
        if not tickers:
            raise DataFetchError("No valid tickers provided.")

        results: dict[str, pd.DataFrame] = {}
        errors: dict[str, str] = {}

        # Parallel fetch — one thread per ticker
        with ThreadPoolExecutor(max_workers=min(len(tickers), 4)) as pool:
            futures = {
                pool.submit(self._fetch_one, ticker, start, end): ticker
                for ticker in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    df = future.result()
                    results[ticker] = df
                except (DataFetchError, InsufficientDataError) as exc:
                    log.warning("ComparisonService: %s failed — %s", ticker, exc.message)
                    errors[ticker] = exc.message
                except Exception as exc:
                    log.warning("ComparisonService: %s unexpected error — %s", ticker, exc)
                    errors[ticker] = str(exc)

        if not results:
            raise DataFetchError(
                "All tickers failed to fetch.",
                detail="; ".join(f"{t}: {e}" for t, e in errors.items()),
            )

        # Find common trading dates across all tickers
        common_dates = None
        for df in results.values():
            if common_dates is None:
                common_dates = df.index
            else:
                common_dates = common_dates.intersection(df.index)

        if common_dates is None or len(common_dates) < _MIN_ROWS:
            raise InsufficientDataError(required=_MIN_ROWS, available=len(common_dates or []))

        # Align all DataFrames to common dates
        aligned: dict[str, pd.DataFrame] = {
            t: df.loc[common_dates] for t, df in results.items()
        }

        normalised = self._normalise_prices(aligned)
        daily_returns = self._daily_returns(aligned)
        correlation = self._correlation_matrix(aligned)
        summary = self._summary_stats(aligned)

        return {
            "tickers": list(results.keys()),
            "errors": errors,
            "normalised_prices": normalised,
            "returns": daily_returns,
            "correlation": correlation,
            "summary": summary,
            "date_range": {
                "start": common_dates[0].strftime("%Y-%m-%d"),
                "end": common_dates[-1].strftime("%Y-%m-%d"),
                "trading_days": len(common_dates),
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_one(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        """Fetch and engineer features for a single ticker."""
        raw = self.market_repo.fetch_raw(ticker, start, end)
        raw = self.market_repo.preprocess(raw)
        df = engineer_features(raw)
        if len(df) < _MIN_ROWS:
            raise InsufficientDataError(required=_MIN_ROWS, available=len(df))
        return df

    def _normalise_prices(self, aligned: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
        """Return base-100 normalised price series for each ticker."""
        out: dict[str, list[dict]] = {}
        for ticker, df in aligned.items():
            close = df["Close"]
            base = close.iloc[0]
            normalised = (close / base * _NORMALISE_BASE).round(2)
            out[ticker] = [
                {"date": d.strftime("%Y-%m-%d"), "value": float(v)}
                for d, v in normalised.items()
            ]
        return out

    def _daily_returns(self, aligned: dict[str, pd.DataFrame]) -> dict[str, list[dict]]:
        """Return daily % return series for each ticker."""
        out: dict[str, list[dict]] = {}
        for ticker, df in aligned.items():
            pct = df["Close"].pct_change().fillna(0.0) * 100.0
            out[ticker] = [
                {"date": d.strftime("%Y-%m-%d"), "return_pct": round(float(v), 4)}
                for d, v in pct.items()
            ]
        return out

    def _correlation_matrix(self, aligned: dict[str, pd.DataFrame]) -> dict[str, Any]:
        """Compute Pearson correlation matrix on log returns."""
        tickers = list(aligned.keys())
        log_rets = pd.DataFrame({
            t: aligned[t]["LogReturn"] for t in tickers
        }).dropna()

        corr = log_rets.corr(method="pearson").round(4)

        # Serialise as list-of-lists with labels
        return {
            "tickers": tickers,
            "matrix": corr.values.tolist(),
            "labels": {
                t: {other: round(float(corr.loc[t, other]), 4) for other in tickers}
                for t in tickers
            },
        }

    def _summary_stats(self, aligned: dict[str, pd.DataFrame]) -> dict[str, dict]:
        """Per-ticker summary: total return, annualised vol, Sharpe, max drawdown."""
        rf_daily = self.cfg.risk_free_rate_annual / 252.0
        out: dict[str, dict] = {}
        for ticker, df in aligned.items():
            close = df["Close"]
            log_ret = df["LogReturn"].dropna()

            total_return = float((close.iloc[-1] / close.iloc[0] - 1.0) * 100.0)
            ann_vol = float(log_ret.std() * np.sqrt(252) * 100.0)
            ann_ret = float(log_ret.mean() * 252 * 100.0)
            excess = log_ret - rf_daily
            sharpe = float(excess.mean() / (excess.std() + 1e-9) * np.sqrt(252))

            # Max drawdown
            cumulative = (1 + log_ret).cumprod()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak) / peak
            max_dd = float(drawdown.min() * 100.0)

            # Latest RSI and regime from most recent data point
            latest_rsi = float(df["RSI14"].iloc[-1]) if "RSI14" in df.columns else None

            out[ticker] = {
                "total_return_pct": round(total_return, 2),
                "annualised_return_pct": round(ann_ret, 2),
                "annualised_volatility_pct": round(ann_vol, 2),
                "sharpe_ratio": round(sharpe, 4),
                "max_drawdown_pct": round(max_dd, 2),
                "latest_rsi": round(latest_rsi, 1) if latest_rsi is not None else None,
                "latest_close": round(float(close.iloc[-1]), 2),
                "start_close": round(float(close.iloc[0]), 2),
            }
        return out
