"""
app/services/backtest_service.py — Backtesting and performance analytics.

Engineering decisions:
  - Pure service: receives arrays, returns typed result dict.
  - No I/O, no Flask imports — fully unit-testable with synthetic data.
  - BUG-07 FIX: Sharpe ratio uses risk-free rate from AppConfig.
  - BUG-03 FIX: Transaction costs applied on weight changes.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from core.config import AppConfig

log = logging.getLogger(__name__)


class BacktestService:
    """Runs backtests and computes portfolio performance metrics.

    Injected with AppConfig so risk_free_rate_annual and
    transaction_cost_pct come from one source of truth.
    """

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Simple signal backtest (used by /api/predict)
    # ------------------------------------------------------------------

    def run_signal_backtest(
        self,
        pred: np.ndarray,
        actual: np.ndarray,
        initial_capital: float,
    ) -> dict:
        """Direction-signal backtest: go long when predicted price rises.

        BUG-07 FIX: excess returns = strategy returns - daily risk-free rate.
        """
        pred_shift = np.roll(pred, 1)
        pred_shift[0] = pred_shift[1]
        signal = np.sign(pred - pred_shift)

        ret = np.concatenate([[0.0], np.diff(actual) / (actual[:-1] + 1e-9)])
        strat_ret = ret * signal
        equity = initial_capital * (1 + strat_ret).cumprod()

        rf_daily = self.cfg.risk_free_rate_annual / 252.0
        excess = strat_ret - rf_daily
        sharpe = float(np.mean(excess) / (np.std(excess) + 1e-9) * np.sqrt(252))

        return {
            "equity": equity.tolist(),
            "buy_signals": np.where(signal > 0, actual, np.nan).tolist(),
            "sell_signals": np.where(signal < 0, actual, np.nan).tolist(),
            "metrics": {
                "Sharpe Ratio": sharpe,
                "Total Return (%)": float((equity[-1] / equity[0] - 1) * 100),
                "Buy & Hold Return (%)": float((actual[-1] / actual[0] - 1) * 100),
            },
        }

    # ------------------------------------------------------------------
    # Regime strategy backtest (used by /api/regime)
    # ------------------------------------------------------------------

    def run_regime_backtest(
        self,
        close: pd.Series,
        regimes: pd.Series,
    ) -> dict:
        """Regime-allocation strategy vs buy-and-hold.

        Allocation matrix:
          Regime 0 (Bull) / 3 (Recovery)  → 100% equity
          Regime 1 (Overbought) / 2 (Sideways) → 50% equity
          Regime 4 (Bear) / 5 (Stress)    →  0% equity (cash)

        BUG-03 FIX: transaction_cost_pct applied on |weight_change|.
        BUG-07 FIX: excess Sharpe with daily risk-free rate.
        """
        common = close.index.intersection(regimes.index)
        close = close.loc[common]
        regimes = regimes.loc[common]
        daily_ret = close.pct_change().fillna(0.0)

        regime_weights = {0: 1.0, 3: 1.0, 1: 0.5, 2: 0.5, 4: 0.0, 5: 0.0}
        weights = regimes.map(regime_weights).fillna(0.0)

        tc = self.cfg.transaction_cost_pct
        weight_changes = weights.diff().abs().fillna(0.0)
        transaction_costs = weight_changes * tc

        strat_ret = weights.shift(1).fillna(0.0) * daily_ret - transaction_costs

        init_cap = self.cfg.initial_capital
        bh_equity = init_cap * (1.0 + daily_ret).cumprod()
        strat_equity = init_cap * (1.0 + strat_ret).cumprod()

        def _mdd(eq: pd.Series) -> float:
            peak = eq.cummax()
            return float(((eq - peak) / peak).min() * 100.0)

        rf_daily = self.cfg.risk_free_rate_annual / 252.0

        def _sharpe(ret: pd.Series) -> float:
            excess = ret - rf_daily
            std = excess.std()
            return float(excess.mean() / std * np.sqrt(252)) if std > 0 else 0.0

        active = strat_ret[weights.shift(1) > 0]
        win_rate = float((active > 0).mean() * 100.0) if len(active) > 0 else 0.0

        n_trades = int((weight_changes > 0).sum())
        step = max(1, len(common) // 200)

        return {
            "dates": [d.strftime("%Y-%m-%d") for d in common[::step]],
            "benchmark_equity": [round(float(v), 2) for v in bh_equity.iloc[::step]],
            "strategy_equity": [round(float(v), 2) for v in strat_equity.iloc[::step]],
            "metrics": {
                "benchmark_total_return": round(float((bh_equity.iloc[-1] / init_cap - 1) * 100), 2),
                "strategy_total_return": round(float((strat_equity.iloc[-1] / init_cap - 1) * 100), 2),
                "benchmark_max_drawdown": round(_mdd(bh_equity), 2),
                "strategy_max_drawdown": round(_mdd(strat_equity), 2),
                "benchmark_sharpe": round(_sharpe(daily_ret), 2),
                "strategy_sharpe": round(_sharpe(strat_ret), 2),
                "strategy_win_rate": round(win_rate, 1),
                "total_transaction_cost_pct": round(float(transaction_costs.sum() * 100), 4),
                "n_trades": n_trades,
                "risk_free_rate_annual": self.cfg.risk_free_rate_annual,
            },
        }

    # ------------------------------------------------------------------
    # Metrics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """Compute RMSE, MAE, MAPE, R², Directional Accuracy."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        eps = 1e-9
        mape = float(np.abs((y_true - y_pred) / (y_true + eps)).mean() * 100.0)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + eps
        r2 = float(1 - ss_res / ss_tot)
        da = float((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean() * 100.0)
        return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "Directional_Accuracy": da}
