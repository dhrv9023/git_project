"""
tests/conftest.py — Shared pytest fixtures with dependency injection.

Engineering decision: all fixtures use concrete test doubles (not mocks by
default) so tests catch real integration bugs while remaining fast.
Fixtures that need network isolation receive a MockMarketDataRepository.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import AppConfig


# ---------------------------------------------------------------------------
# Config fixture — test config with tiny epoch counts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_cfg() -> AppConfig:
    """Minimal config for fast tests. No network, temp dirs overridden by patches."""
    return AppConfig(
        epochs=2,
        batch_size=8,
        sequence_length=10,
        train_split=0.7,
        val_split=0.15,
        model_artifacts_dir="/tmp/sb_test_artifacts",
        inference_cache_dir="inference_cache",
        max_worker_threads=1,
        environment="test",
    )


# ---------------------------------------------------------------------------
# Synthetic OHLCV data
# ---------------------------------------------------------------------------

def make_synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100.0 * np.cumprod(1 + rng.normal(0.0003, 0.015, n))
    noise = rng.uniform(0.98, 1.02, n)
    df = pd.DataFrame(
        {
            "Open": close * rng.uniform(0.99, 1.01, n),
            "High": close * rng.uniform(1.00, 1.03, n),
            "Low": close * rng.uniform(0.97, 1.00, n),
            "Close": close,
            "Volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
        },
        index=dates,
    )
    return df


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return make_synthetic_ohlcv()


@pytest.fixture
def engineered_df(synthetic_ohlcv) -> pd.DataFrame:
    from ml.features import engineer_features
    return engineer_features(synthetic_ohlcv)


# ---------------------------------------------------------------------------
# Global yfinance & repository mock fixture to ensure zero network in tests
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_external_network(monkeypatch):
    """Globally mock yfinance and data fetchers to guarantee fast, offline tests."""
    import yfinance as yf
    import app
    from app.repositories.market_data_repo import MarketDataRepository
    
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: make_synthetic_ohlcv(300))
    if hasattr(app, "fetch_data_yfinance"):
        monkeypatch.setattr(app, "fetch_data_yfinance", lambda *args, **kwargs: make_synthetic_ohlcv(300))
    monkeypatch.setattr(MarketDataRepository, "fetch_raw", lambda self, ticker, start, end: make_synthetic_ohlcv(300))


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def test_app(test_cfg):
    """Flask test app with DI-injected test config."""
    from unittest.mock import patch

    with patch("yfinance.download", return_value=make_synthetic_ohlcv(300)):
        from app import create_app
        app = create_app(test_cfg)
        app.config["TESTING"] = True
        return app


@pytest.fixture
def client(test_app):
    with test_app.test_client() as c:
        yield c
