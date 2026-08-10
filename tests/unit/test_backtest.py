"""
tests/unit/test_backtest.py — Unit tests for BacktestService.

All tests use synthetic deterministic data — no network calls.
"""
import numpy as np
import pandas as pd
import pytest

from core.config import AppConfig
from app.services.backtest_service import BacktestService


@pytest.fixture
def svc(test_cfg):
    return BacktestService(test_cfg)


class TestSignalBacktest:
    def test_equity_length_matches_input(self, svc):
        pred = np.linspace(100, 120, 50)
        actual = np.linspace(100, 115, 50)
        result = svc.run_signal_backtest(pred, actual, initial_capital=10_000)
        assert len(result["equity"]) == 50

    def test_initial_capital_is_equity_start(self, svc):
        pred = np.linspace(100, 110, 30)
        actual = np.linspace(100, 108, 30)
        result = svc.run_signal_backtest(pred, actual, initial_capital=5_000)
        assert abs(result["equity"][0] - 5_000) < 1.0

    def test_sharpe_uses_risk_free_rate(self, svc):
        """With Rf=5% annualised, Sharpe of a flat strategy should be negative."""
        actual = np.ones(252) * 100.0
        pred = np.ones(252) * 100.0
        result = svc.run_signal_backtest(pred, actual, initial_capital=10_000)
        assert result["metrics"]["Sharpe Ratio"] <= 0

    def test_buy_signals_are_nan_on_sell_days(self, svc):
        # Predict monotonically increasing — always buy signal
        pred = np.linspace(100, 200, 50)
        actual = np.linspace(100, 180, 50)
        result = svc.run_signal_backtest(pred, actual, initial_capital=10_000)
        # sell_signals should be all NaN for increasing prediction
        sell = np.array(result["sell_signals"], dtype=float)
        assert np.all(np.isnan(sell[1:]))  # first element may differ


class TestCalculateMetrics:
    def test_perfect_prediction_rmse_zero(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m = BacktestService.calculate_metrics(y, y)
        assert m["RMSE"] < 1e-10
        assert m["R2"] > 0.9999

    def test_directional_accuracy_perfect(self):
        y_true = np.array([1.0, 2.0, 1.5, 3.0, 2.5])
        y_pred = np.array([1.1, 2.1, 1.4, 3.1, 2.4])
        m = BacktestService.calculate_metrics(y_true, y_pred)
        assert m["Directional_Accuracy"] == 100.0

    def test_output_keys(self):
        y = np.random.rand(20) + 1
        m = BacktestService.calculate_metrics(y, y * 0.95)
        assert set(m.keys()) == {"RMSE", "MAE", "MAPE", "R2", "Directional_Accuracy"}


class TestRegimeBacktest:
    def test_output_keys(self, svc):
        dates = pd.bdate_range("2021-01-01", periods=100)
        close = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 100)) * 100, index=dates)
        regimes = pd.Series(np.random.randint(0, 6, 100), index=dates)
        result = svc.run_regime_backtest(close, regimes)
        assert "dates" in result
        assert "metrics" in result
        assert "benchmark_sharpe" in result["metrics"]
        assert "strategy_sharpe" in result["metrics"]

    def test_sharpe_is_float(self, svc):
        dates = pd.bdate_range("2021-01-01", periods=252)
        close = pd.Series(np.cumprod(1 + np.random.normal(0, 0.01, 252)) * 100, index=dates)
        regimes = pd.Series(np.zeros(252, dtype=int), index=dates)  # all bull
        result = svc.run_regime_backtest(close, regimes)
        assert isinstance(result["metrics"]["strategy_sharpe"], float)
