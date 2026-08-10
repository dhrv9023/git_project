"""
tests/unit/test_features.py — Unit tests for ml/features.py

Engineering decision: pure function unit tests — no I/O, no mocks.
Known-value assertions validate computation correctness.
"""
import numpy as np
import pandas as pd
import pytest

from ml.features import (
    compute_rsi, compute_ema, compute_macd,
    create_sequences, split_and_scale_data, engineer_features,
)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestComputeRsi:
    def test_output_bounded_0_100(self):
        series = pd.Series(np.random.uniform(100, 200, 100))
        rsi = compute_rsi(series, window=14)
        valid = rsi.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_all_up_days_rsi_near_100(self):
        """If price always rises, RSI should approach 100."""
        series = pd.Series(np.linspace(100, 200, 60))
        rsi = compute_rsi(series, window=14).dropna()
        assert rsi.iloc[-1] > 90

    def test_all_down_days_rsi_near_0(self):
        """If price always falls, RSI should approach 0."""
        series = pd.Series(np.linspace(200, 100, 60))
        rsi = compute_rsi(series, window=14).dropna()
        assert rsi.iloc[-1] < 10

    def test_returns_series_same_index(self):
        series = pd.Series(np.random.rand(50), index=pd.date_range("2020-01-01", periods=50))
        rsi = compute_rsi(series)
        assert rsi.index.equals(series.index)


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestComputeEma:
    def test_output_length_matches_input(self):
        series = pd.Series(np.random.rand(100))
        ema = compute_ema(series, span=20)
        assert len(ema) == len(series)

    def test_ema_smooths_series(self):
        """EMA std should be less than raw series std."""
        series = pd.Series(np.random.randn(200) * 10 + 100)
        ema = compute_ema(series, span=20)
        assert ema.std() < series.std()


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

class TestComputeMacd:
    def test_output_columns(self):
        series = pd.Series(np.random.rand(100) * 100 + 100)
        macd_df = compute_macd(series)
        assert set(macd_df.columns) == {"MACD", "MACD_signal", "MACD_hist"}

    def test_hist_equals_macd_minus_signal(self):
        series = pd.Series(np.random.rand(100) * 100 + 100)
        df = compute_macd(series)
        diff = (df["MACD"] - df["MACD_signal"] - df["MACD_hist"]).abs()
        assert diff.max() < 1e-10

    def test_index_preserved(self):
        idx = pd.date_range("2020-01-01", periods=100)
        series = pd.Series(np.random.rand(100) * 100, index=idx)
        df = compute_macd(series)
        assert df.index.equals(idx)


# ---------------------------------------------------------------------------
# Sequences
# ---------------------------------------------------------------------------

class TestCreateSequences:
    def test_output_shapes(self):
        X = np.random.rand(100, 5).astype("float32")
        y = np.random.rand(100).astype("float32")
        Xs, ys = create_sequences(X, y, seq_len=10)
        assert Xs.shape == (90, 10, 5)
        assert ys.shape == (90,)

    def test_dtypes_float32(self):
        X = np.random.rand(50, 3).astype("float64")
        y = np.random.rand(50).astype("float64")
        Xs, ys = create_sequences(X, y, seq_len=5)
        assert Xs.dtype == np.float32
        assert ys.dtype == np.float32


# ---------------------------------------------------------------------------
# Split and scale
# ---------------------------------------------------------------------------

class TestSplitAndScaleData:
    def test_no_data_leakage_scaler_fit_on_train_only(self):
        """Scaler data_min_ must equal training partition minimum, not overall min."""
        np.random.seed(0)
        X = np.random.rand(300, 5).astype("float32")
        X[200:] *= 10  # future data has much higher values
        y = np.random.rand(300).astype("float32")
        dates = pd.bdate_range("2020-01-01", periods=300)
        base = np.ones(300)

        result = split_and_scale_data(X, y, dates, base, 0.7, 0.15, 10)
        scaler_X = result["scaler_X"]
        # Scaler was fitted on first 210 rows — data_min_ should be from there
        train_min = X[:210].min(axis=0)
        assert np.allclose(scaler_X.data_min_, train_min, atol=1e-5)

    def test_all_three_splits_present(self):
        X = np.random.rand(200, 4).astype("float32")
        y = np.random.rand(200).astype("float32")
        dates = pd.bdate_range("2020-01-01", periods=200)
        base = np.ones(200)
        result = split_and_scale_data(X, y, dates, base, 0.7, 0.15, 5)
        assert "train" in result and "val" in result and "test" in result

    def test_sequences_are_float32(self):
        X = np.random.rand(100, 3).astype("float32")
        y = np.random.rand(100).astype("float32")
        dates = pd.bdate_range("2020-01-01", periods=100)
        base = np.ones(100)
        result = split_and_scale_data(X, y, dates, base, 0.7, 0.15, 5)
        X_tr, y_tr, _, _ = result["train"]
        assert X_tr.dtype == np.float32
        assert y_tr.dtype == np.float32


# ---------------------------------------------------------------------------
# Engineer features
# ---------------------------------------------------------------------------

class TestEngineerFeatures:
    def test_required_columns_present(self, engineered_df):
        for col in ["RSI14", "EMA20", "MACD", "MACD_signal", "MACD_hist", "LogReturn", "DayOfWeek"]:
            assert col in engineered_df.columns, f"Missing: {col}"

    def test_no_nulls_after_engineering(self, engineered_df):
        assert not engineered_df.isnull().any().any()

    def test_day_of_week_range(self, engineered_df):
        assert engineered_df["DayOfWeek"].between(0, 6).all()
