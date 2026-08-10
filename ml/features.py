"""
ml/features.py — Technical indicator computation and feature engineering pipeline.

Engineering decisions:
  - Pure functions only: no global state, no I/O — fully unit-testable.
  - All functions take pd.Series / pd.DataFrame and return same types.
  - Type hints on every public function signature.
  - Extracted from app.py to break circular sys.modules coupling.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing via EWM)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD line, signal line, and histogram.

    Returns a DataFrame with columns: MACD, MACD_signal, MACD_hist.
    Uses np.asarray().ravel() to avoid scalar-construction errors with
    MultiIndex DataFrames from yfinance.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    return pd.DataFrame(
        {
            "MACD": np.asarray(macd).ravel(),
            "MACD_signal": np.asarray(signal_line).ravel(),
            "MACD_hist": np.asarray(hist).ravel(),
        },
        index=series.index,
    )


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

FEATURE_COLS: List[str] = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI14", "EMA20",
    "MACD", "MACD_signal", "MACD_hist",
    "DayOfWeek", "LogReturn",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a raw OHLCV DataFrame.

    Engineering decision: features computed on the full available history
    before the train/val/test split. This is safe because indicators are
    purely historical (no look-ahead). The scaler is fitted ONLY on the
    training partition downstream (see split_and_scale_data).
    """
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["LogReturn"] = out["Close"].apply(np.log).diff()
    out["RSI14"] = compute_rsi(out["Close"], 14)
    out["EMA20"] = compute_ema(out["Close"], 20)
    macd_df = compute_macd(out["Close"])
    out = pd.concat([out, macd_df], axis=1)
    out["DayOfWeek"] = pd.DatetimeIndex(out.index).dayofweek
    return out.dropna()


# ---------------------------------------------------------------------------
# Sequence creation
# ---------------------------------------------------------------------------

def create_sequences(
    data_array: np.ndarray,
    target_array: np.ndarray,
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert flat feature matrix into overlapping (X, y) sequences.

    Args:
        data_array:   (N, F) scaled feature matrix
        target_array: (N,)  scaled target vector
        seq_len:      lookback window length

    Returns:
        X: (N - seq_len, seq_len, F)
        y: (N - seq_len,)
    """
    X, y = [], []
    for i in range(seq_len, len(data_array)):
        X.append(data_array[i - seq_len : i])
        y.append(target_array[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# Train/val/test split with correct scaler discipline
# ---------------------------------------------------------------------------

def split_and_scale_data(
    X_raw: np.ndarray,
    y_raw: np.ndarray,
    dates: pd.Index,
    base_prices: np.ndarray,
    train_split: float,
    val_split: float,
    sequence_length: int,
) -> dict:
    """Chronological split → fit scalers on train only → create sequences.

    BUG-01 FIX: Scaler is fitted ONLY on the training partition. Fitting on
    the full dataset leaks future price extremes into normalization space,
    inflating evaluation metrics artificially.

    BUG-02 FIX: Context windows prepend the last seq_len rows of the
    preceding partition so val/test sequences never have a cold lookback.
    """
    n = len(X_raw)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    seq_len = sequence_length

    X_tr_raw = X_raw[:n_train]
    X_va_raw = X_raw[n_train : n_train + n_val]
    X_te_raw = X_raw[n_train + n_val :]

    y_tr_raw = y_raw[:n_train]
    y_va_raw = y_raw[n_train : n_train + n_val]
    y_te_raw = y_raw[n_train + n_val :]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_tr_raw)
    scaler_y.fit(y_tr_raw.reshape(-1, 1))

    X_tr = scaler_X.transform(X_tr_raw)
    X_va = scaler_X.transform(X_va_raw)
    X_te = scaler_X.transform(X_te_raw)

    y_tr = scaler_y.transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_va = scaler_y.transform(y_va_raw.reshape(-1, 1)).ravel()
    y_te = scaler_y.transform(y_te_raw.reshape(-1, 1)).ravel()

    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len)
    dates_tr = dates[seq_len:n_train]
    base_tr = base_prices[seq_len:n_train]

    X_va_ctx = np.concatenate([X_tr[-seq_len:], X_va], axis=0)
    y_va_ctx = np.concatenate([y_tr[-seq_len:], y_va], axis=0)
    X_va_seq, y_va_seq = create_sequences(X_va_ctx, y_va_ctx, seq_len)
    dates_va = dates[n_train : n_train + n_val]
    base_va = base_prices[n_train : n_train + n_val]

    X_te_ctx = np.concatenate([X_va[-seq_len:], X_te], axis=0)
    y_te_ctx = np.concatenate([y_va[-seq_len:], y_te], axis=0)
    X_te_seq, y_te_seq = create_sequences(X_te_ctx, y_te_ctx, seq_len)
    dates_te = dates[n_train + n_val :]
    base_te = base_prices[n_train + n_val :]

    log.debug(
        "Sequences — train:%d  val:%d  test:%d",
        len(X_tr_seq), len(X_va_seq), len(X_te_seq),
    )
    return {
        "train": (X_tr_seq, y_tr_seq, dates_tr, base_tr),
        "val": (X_va_seq, y_va_seq, dates_va, base_va),
        "test": (X_te_seq, y_te_seq, dates_te, base_te),
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
    }


# ---------------------------------------------------------------------------
# Scaling helpers
# ---------------------------------------------------------------------------

def inverse_target_only(
    pred_scaled: np.ndarray,
    scaler: MinMaxScaler,
    num_features: int,
    target_index: int,
) -> np.ndarray:
    """Inverse-transform a 1-D scaled target using a multi-feature scaler."""
    mat = np.zeros((len(pred_scaled), num_features), dtype=np.float32)
    mat[:, target_index] = pred_scaled
    return scaler.inverse_transform(mat)[:, target_index]


def scale_single_feature(
    value: float,
    scaler: MinMaxScaler,
    feature_index: int,
) -> float:
    """Scale one raw feature value using an already-fitted scaler."""
    if hasattr(scaler, "mean_") and hasattr(scaler, "scale_"):
        return float(
            (value - scaler.mean_[feature_index])
            / (scaler.scale_[feature_index] + 1e-12)
        )
    if hasattr(scaler, "data_min_"):
        return float(
            (value - scaler.data_min_[feature_index]) * scaler.scale_[feature_index]
        )
    row = np.zeros((1, feature_index + 1), dtype=np.float32)
    row[0, feature_index] = value
    return float(scaler.transform(row)[0, feature_index])
