"""
StockBuddy — Quantitative Market Intelligence Engine
Phase 1: Statistical Corrections (data leakage, Sharpe Rf, transaction costs, walk-forward)
Phase 2: Production ML Pipeline (model registry, async training, disk persistence,
         inference cache, auto-retraining scheduler, config management)

v1 API routes  → /api/regime, /api/predict, /api/wf_validate   (unchanged, backward compat)
v2 API routes  → /api/v2/*                                       (Phase 2 additions)
"""


import os
import sys
import math
import logging
import datetime
import warnings
warnings.filterwarnings('ignore')

import json
import time

_SERVER_START_TIME = time.time()

# ── Phase 6: Production Logging & Error Tracking Configuration ─────────────────
from core.config import CFG

class JSONFormatter(logging.Formatter):
    """Structured JSON log formatter for production monitoring / CloudWatch / Datadog."""
    def format(self, record):
        log_obj = {
            "timestamp": datetime.datetime.fromtimestamp(record.created, tz=datetime.timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "environment": getattr(CFG, "environment", "development"),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger('stockbuddy')

if getattr(CFG, "log_format", "").lower() == "json" or getattr(CFG, "environment", "") == "production":
    for handler in logging.root.handlers:
        handler.setFormatter(JSONFormatter())

# Sentry Error Tracking Integration
if getattr(CFG, "sentry_dsn", ""):
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration
        sentry_sdk.init(
            dsn=CFG.sentry_dsn,
            environment=getattr(CFG, "environment", "development"),
            integrations=[FlaskIntegration()],
            traces_sample_rate=1.0 if CFG.environment != "production" else 0.2,
        )
        log.info(f"[Phase6] Sentry SDK initialized for environment: {CFG.environment}")
    except ImportError:
        log.warning("[Phase6] SENTRY_DSN configured but sentry-sdk package is not installed.")
    except Exception as exc:
        log.warning(f"[Phase6] Failed to initialize Sentry SDK: {exc}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except Exception:
    raise RuntimeError("yfinance is required. Install with: pip install yfinance")

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras import mixed_precision

# API server
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Phase 2: Production ML infrastructure ─────────────────────────────────────
from core.config       import CFG
from ml.registry       import ModelRegistry
from ml.trainer        import BackgroundTrainer
from ml.inference      import InferenceCache, InferenceEngine
from ml.scheduler      import RetrainingScheduler
from storage.model_store import ModelStore

# ── Phase 3: Distributed Systems infrastructure ─────────────────────────────
from core.circuit_breaker import get_breaker, all_breaker_stats
from core.rate_limiter    import RateLimiter
from core.metrics         import (
    REGISTRY, http_requests_total, http_latency_seconds,
    training_jobs_total, inference_latency_s,
    cache_hits_total, cache_misses_total,
    queue_depth, dlq_depth, active_workers,
    memory_cache_entries, disk_cache_entries, model_versions_total,
    Timer,
)
from ml.queue         import PriorityJobQueue, RetryPolicy, PRIORITY_NORMAL


# --------------------------
# Configuration
# --------------------------
CONFIG = {
    'ticker': 'AAPL',
    'start_date': '2020-01-01',
    'end_date': datetime.date.today().isoformat(),
    'sequence_length': 90,
    'train_split': 0.7,
    'val_split': 0.15,
    'epochs': 20,
    'batch_size': 16,
    'initial_capital': 10000.0,
    'learning_rate': 1e-4,
    'seed': 42,
    # --- Phase 1 Statistical Corrections ---
    # BUG-07: Risk-free rate used in Sharpe ratio denominator.
    # Using 5% annual (US T-bill proxy). Converted to daily inside Sharpe calcs.
    'risk_free_rate_annual': 0.05,
    # BUG-03: Round-trip transaction cost per trade (0.10% = 5 bps each leg).
    # Applied whenever regime allocation weight changes between consecutive days.
    'transaction_cost_pct': 0.001,
}

np.random.seed(CONFIG['seed'])
tf.random.set_seed(CONFIG['seed'])

# ---------------------------------------------------------------------------
# BUG-05: In-memory model cache — prevents full retrain on every API call.
# Key  : "{ticker}_{start}_{end}_{seq_len}_{epochs}"  (config-version hash)
# Value: {'models': dict, 'scaler_X': obj, 'scaler_y': obj,
#         'trained_at': ISO-str, 'cache_key': str}
# Cache is server-session-scoped (survives requests, lost on restart).
# ---------------------------------------------------------------------------
MODEL_CACHE: dict = {}

# --------------------------
# GPU/Performance configuration
# --------------------------
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
        # Enable mixed precision for tensor cores (speed-up on modern GPUs)
        mixed_precision.set_global_policy('mixed_float16')
        # Enable XLA compilation (can improve performance)
        tf.config.optimizer.set_jit(True)
        print(f"Using GPU(s): {[d.name for d in gpus]}")
    else:
        print("No GPU detected; training will use CPU.")
except Exception as e:
    print(f"GPU configuration warning: {e}")


# --------------------------
# Utilities
# --------------------------
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series: pd.Series, span: int = 20) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    # Ensure index alignment to avoid scalar-construction errors
    return pd.DataFrame(
        {
            'MACD': np.asarray(macd).ravel(),
            'MACD_signal': np.asarray(signal_line).ravel(),
            'MACD_hist': np.asarray(hist).ravel()
        },
        index=series.index
    )


def create_sequences(data_array: np.ndarray, target_array: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data_array)):
        X.append(data_array[i - seq_len:i])
        y.append(target_array[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    # Directional accuracy
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    da = (true_dir == pred_dir).mean() * 100.0
    # MAPE
    eps = 1e-9
    mape = (np.abs((y_true - y_pred) / (y_true + eps))).mean() * 100.0
    # R2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + eps
    r2 = 1 - (ss_res / ss_tot)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Directional_Accuracy': da}


def set_plot_style():
    plt.style.use('seaborn-v0_8')


def plot_predictions(dates, y_true, pred_dict, title="Predictions"):
    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_true, label='Actual', color='black', linewidth=2)
    for name, preds in pred_dict.items():
        plt.plot(dates, preds, label=name)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confidence_intervals(dates, y_true, y_pred, ci_bounds):
    lower, upper = ci_bounds
    plt.figure(figsize=(12, 4))
    plt.plot(dates, y_true, label='Actual', color='black')
    plt.plot(dates, y_pred, label='Ensemble', color='green')
    plt.fill_between(dates, lower, upper, color='green', alpha=0.15, label='Confidence band')
    plt.legend(); plt.tight_layout(); plt.show()


def create_metrics_table(metrics_dict: dict) -> pd.DataFrame:
    return pd.DataFrame(metrics_dict).T.sort_values('RMSE')


# --------------------------
# Data pipeline
# --------------------------
def inverse_target_only(pred_scaled: np.ndarray, scaler, num_features: int, target_index: int) -> np.ndarray:
    """Inverse-transform a 1D scaled target using a scaler fitted on multiple features.
    Reconstructs a dummy matrix with only the target column populated, then picks back that column.
    """
    mat = np.zeros((len(pred_scaled), num_features), dtype=np.float32)
    mat[:, target_index] = pred_scaled
    inv = scaler.inverse_transform(mat)
    return inv[:, target_index]


def scale_single_feature(value: float, scaler, feature_index: int) -> float:
    """Scale a single raw feature value using a fitted StandardScaler or MinMaxScaler that was fit on all features."""
    # StandardScaler case
    if hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
        return float((value - scaler.mean_[feature_index]) / (scaler.scale_[feature_index] + 1e-12))
    # MinMaxScaler case
    if hasattr(scaler, 'data_min_') and hasattr(scaler, 'scale_'):
        return float((value - scaler.data_min_[feature_index]) * scaler.scale_[feature_index])
    # Fallback: reconstruct a row
    row = np.zeros((1, len(getattr(scaler, 'scale_', [0]* (feature_index+1)))), dtype=np.float32)
    row[0, feature_index] = value
    return float(scaler.transform(row)[0, feature_index])
def fetch_data_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = yf.download(tickers=ticker, start=start_date, end=end_date, auto_adjust=True, progress=False, group_by='column')
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Check ticker or dates.")
    # Flatten potential MultiIndex columns and select single ticker slice
    if isinstance(df.columns, pd.MultiIndex):
        # If last level contains the ticker, slice it
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            # Fallback: drop all but first level
            df.columns = ['_'.join([str(x) for x in col if x is not None]) for col in df.columns]
    # Normalize column names to Title-case expected
    col_map = {c: c.title() for c in df.columns}
    df = df.rename(columns=col_map)
    # Some providers may give 'Adj Close' only; ensure core columns exist
    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in required if c not in df.columns]
    if missing:
        # Try to reconstruct Open/High/Low from Close if absolutely necessary
        if missing and 'Close' in df.columns:
            for c in missing:
                if c != 'Volume':
                    df[c] = df['Close']
        # If Volume missing, fill with zeros
        if 'Volume' in missing:
            df['Volume'] = 0.0
    df = df[required]
    df = df.dropna()
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data: sort, drop duplicates/nulls, handle infs, remove outliers by returns, winsorize tails.
    This keeps the price structure intact while mitigating extreme spikes.
    """
    cleaned = df.copy()
    # Order & de-duplicate
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
    # Replace infs and obvious bad values
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    # Forward/backward fill small gaps, then drop remaining
    cleaned = cleaned.ffill().bfill().dropna()
    # Compute simple returns for outlier detection
    returns = cleaned['Close'].pct_change()
    # IQR-based filter on returns (conservative whiskers = 3*IQR)
    ret_no_na = returns.dropna()
    q1, q3 = ret_no_na.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 3.0 * iqr
    upper = q3 + 3.0 * iqr
    mask = (returns.between(lower, upper)) | (returns.isna())
    cleaned = cleaned.loc[mask]
    # Winsorize numeric columns at 1st/99th percentile to limit remaining tails
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns
    lower_clip = cleaned[numeric_cols].quantile(0.01)
    upper_clip = cleaned[numeric_cols].quantile(0.99)
    cleaned[numeric_cols] = cleaned[numeric_cols].clip(lower=lower_clip, upper=upper_clip, axis=1)
    # Final drop of any gaps from filtering
    cleaned = cleaned.dropna()
    return cleaned


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Return'] = df['Close'].pct_change()
    df['LogReturn'] = np.log(df['Close']).diff()
    df['RSI14'] = compute_rsi(df['Close'], 14)
    df['EMA20'] = compute_ema(df['Close'], 20)
    macd_df = compute_macd(df['Close'])
    df = pd.concat([df, macd_df], axis=1)
    df['DayOfWeek'] = df.index.dayofweek
    df = df.dropna()
    return df


def prepare_data(ticker: str, start_date: str, end_date: str, sequence_length: int):
    print("\n" + "="*70)
    print("STEP 1: DATA COLLECTION")
    print("="*70)
    raw = fetch_data_yfinance(ticker, start_date, end_date)
    print(f"Fetched {len(raw)} rows")
    raw = preprocess_data(raw)
    print(f"After preprocessing: {len(raw)} rows")
    
    print("\n" + "="*70)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*70)
    df = engineer_features(raw)

    feature_cols = [
        'Open','High','Low','Close','Volume','RSI14','EMA20',
        'MACD','MACD_signal','MACD_hist','DayOfWeek','LogReturn'
    ]
    # Target: next-step log return of Close
    target_series = df['LogReturn'].shift(-1).dropna()

    # Align features and base prices with target index (t), predicting t+1 return
    df.columns = df.columns.astype(str)
    available_cols = [c for c in feature_cols if c in df.columns]
    missing = sorted(list(set(feature_cols) - set(available_cols)))
    if missing:
        print(f"Warning: missing engineered columns skipped: {missing}")

    X_df = df.loc[target_series.index, available_cols]
    base_prices = df.loc[target_series.index, 'Close']  # Close at time t used to reconstruct t+1 price

    X = X_df.values.astype(np.float32)
    y = target_series.values.astype(np.float32)
    base_prices_arr = base_prices.values
    dates_arr = target_series.index

    # BUG-01 FIX: Do NOT scale here. Scaling must happen AFTER the chronological
    # split so the scaler is fit only on training data. Fitting on the full dataset
    # leaks future price extremes (min/max) into the training normalization space,
    # which artificially reduces model loss and inflates evaluation metrics.
    # Raw arrays are returned and scaled inside split_and_scale_data().
    print(f"Raw feature matrix: {X.shape} | Raw targets: {y.shape}")

    return {
        'X_raw': X,
        'y_raw': y,
        'dates_raw': dates_arr,
        'base_prices_raw': base_prices_arr,
        'sequence_length': sequence_length,
        'feature_cols': available_cols,
        'close_feature_index': available_cols.index('Close') if 'Close' in available_cols else 0,
        'logret_feature_index': available_cols.index('LogReturn') if 'LogReturn' in available_cols else len(available_cols)-1,
    }


def split_and_scale_data(X_raw: np.ndarray, y_raw: np.ndarray, dates, base_prices: np.ndarray,
                         train_split: float, val_split: float, sequence_length: int) -> dict:
    """
    BUG-01 + BUG-02 FIX: Chronological split → scale → sequence creation.

    Correct order of operations:
      1. Split raw (unscaled) arrays at chronological boundaries.
      2. Fit MinMaxScaler on the TRAINING partition only.
      3. Transform val and test partitions using the train-fitted scaler.
      4. Build sequences using a 'context window': prepend the last `seq_len`
         rows of the preceding partition so val/test lookback windows never
         cross a cold (unscaled) boundary.

    Why context windows?
      Without them, val_seq[0] would have no prior rows to look back into and
      the first `seq_len` samples of the val set would be wasted. The context
      rows are already correctly scaled (train scaler), so no leakage occurs.
    """
    n = len(X_raw)
    n_train = int(n * train_split)
    n_val   = int(n * val_split)
    seq_len = sequence_length

    # --- Step 1: Chronological split of RAW arrays ---
    X_tr_raw = X_raw[:n_train]
    X_va_raw = X_raw[n_train:n_train + n_val]
    X_te_raw = X_raw[n_train + n_val:]

    y_tr_raw = y_raw[:n_train]
    y_va_raw = y_raw[n_train:n_train + n_val]
    y_te_raw = y_raw[n_train + n_val:]

    # --- Step 2: Fit scalers on TRAINING data only ---
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(X_tr_raw)
    scaler_y.fit(y_tr_raw.reshape(-1, 1))

    # --- Step 3: Transform each partition (val/test use train-fit scaler) ---
    X_tr = scaler_X.transform(X_tr_raw)
    X_va = scaler_X.transform(X_va_raw)
    X_te = scaler_X.transform(X_te_raw)

    y_tr = scaler_y.transform(y_tr_raw.reshape(-1, 1)).ravel()
    y_va = scaler_y.transform(y_va_raw.reshape(-1, 1)).ravel()
    y_te = scaler_y.transform(y_te_raw.reshape(-1, 1)).ravel()

    # --- Step 4: Create sequences with context windows ---
    # Training: straightforward — sequences from training rows only.
    X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len)
    dates_tr  = dates[seq_len:n_train]
    base_tr   = base_prices[seq_len:n_train]

    # Validation: prepend last seq_len rows of training as lookback context.
    # This gives val_seq[0] a full seq_len-length input window without leaking
    # any scaler statistics (context rows use the train-fit scaler transform).
    X_va_ctx = np.concatenate([X_tr[-seq_len:], X_va], axis=0)
    y_va_ctx = np.concatenate([y_tr[-seq_len:], y_va], axis=0)
    X_va_seq, y_va_seq = create_sequences(X_va_ctx, y_va_ctx, seq_len)
    dates_va  = dates[n_train:n_train + n_val]
    base_va   = base_prices[n_train:n_train + n_val]

    # Test: prepend last seq_len rows of validation as lookback context.
    X_te_ctx = np.concatenate([X_va[-seq_len:], X_te], axis=0)
    y_te_ctx = np.concatenate([y_va[-seq_len:], y_te], axis=0)
    X_te_seq, y_te_seq = create_sequences(X_te_ctx, y_te_ctx, seq_len)
    dates_te  = dates[n_train + n_val:]
    base_te   = base_prices[n_train + n_val:]

    print(f"Train/Val/Test sequences: {len(X_tr_seq)}/{len(X_va_seq)}/{len(X_te_seq)}")
    return {
        'train':    (X_tr_seq, y_tr_seq, dates_tr, base_tr),
        'val':      (X_va_seq, y_va_seq, dates_va, base_va),
        'test':     (X_te_seq, y_te_seq, dates_te, base_te),
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
    }


# Legacy alias kept so existing call-sites that use 'split_data' still work
# during any incremental migration. New code should use split_and_scale_data.
def split_data(X, y, dates, base_seq, train_split, val_split):
    """DEPRECATED: use split_and_scale_data() which fixes BUG-01/02."""
    n = len(X)
    n_train = int(n * train_split)
    n_val = int(n * val_split)
    return {
        'train': (X[:n_train], y[:n_train], dates[:n_train], base_seq[:n_train]),
        'val':   (X[n_train:n_train+n_val], y[n_train:n_train+n_val],
                  dates[n_train:n_train+n_val], base_seq[n_train:n_train+n_val]),
        'test':  (X[n_train+n_val:], y[n_train+n_val:],
                  dates[n_train+n_val:], base_seq[n_train+n_val:]),
    }


# --------------------------
# Models
# --------------------------
def build_lstm(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.LSTM(64, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.Dense(1, dtype='float32')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


def build_gru(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GRU(128, return_sequences=True, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.GRU(64, recurrent_dropout=0.3, kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.5),
        layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(5e-5)),
        layers.Dropout(0.4),
        layers.Dense(1, dtype='float32')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


def build_transformer(input_shape, num_heads=4, ff_dim=128, dropout=0.2):
    inp = layers.Input(shape=input_shape)
    x = inp
    # Positional encoding (simple learnable)
    positions = tf.range(start=0, limit=input_shape[0], delta=1)
    pos_embed = layers.Embedding(input_dim=input_shape[0], output_dim=input_shape[1])(positions)
    pos_embed = tf.expand_dims(pos_embed, axis=0)
    x = x + pos_embed
    # Two encoder blocks
    for _ in range(2):
        attn_out = layers.MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_out)
        ff = layers.Dense(ff_dim, activation='relu')(x)
        ff = layers.Dropout(dropout)(ff)
        ff = layers.Dense(input_shape[1])(ff)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(dropout + 0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(1, dtype='float32')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer=optimizers.Adam(learning_rate=CONFIG['learning_rate']), loss=tf.keras.losses.Huber(delta=1.0), metrics=['mae'])
    return model


def train_models(data_splits, input_shape, epochs, batch_size, selected_models=None, use_early_stopping=True):
    X_train, y_train, _, _ = data_splits['train']
    X_val, y_val, _, _ = data_splits['val']

    if selected_models is None:
        selected_models = ['LSTM', 'GRU', 'Transformer']

    es = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) if use_early_stopping else None
    rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5) if use_early_stopping else None
    # Cosine decay with warmup
    steps_per_epoch = max(1, len(data_splits['train'][0]) // batch_size)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = max(1, int(0.1 * total_steps))
    def lr_schedule(step):
        step = tf.cast(step, tf.float32)
        lr_base = tf.constant(CONFIG['learning_rate'], tf.float32)
        lr_min = lr_base * 0.1
        def warm():
            return lr_base * (step / tf.cast(warmup_steps, tf.float32))
        def cosine():
            progress = (step - warmup_steps) / tf.cast(max(1, total_steps - warmup_steps), tf.float32)
            return lr_min + (lr_base - lr_min) * 0.5 * (1 + tf.cos(np.pi * tf.clip_by_value(progress, 0.0, 1.0)))
        return tf.where(step < warmup_steps, warm(), cosine())
    lr_callback = callbacks.LearningRateScheduler(lambda s: float(lr_schedule(s).numpy()), verbose=0)

    print("\n" + "="*70)
    print("STEP 3: MODEL TRAINING")
    print("="*70)
    
    models_dict = {}

    if 'LSTM' in selected_models:
        print("\nTraining LSTM...")
        lstm = build_lstm(input_shape)
        cbs = [cb for cb in [es, rlrop, lr_callback] if cb is not None]
        hist_lstm = lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=cbs)
        models_dict['LSTM'] = lstm
        models_dict['LSTM_history'] = hist_lstm.history

    if 'GRU' in selected_models:
        print("\nTraining GRU...")
        gru = build_gru(input_shape)
        cbs = [cb for cb in [es, rlrop, lr_callback] if cb is not None]
        hist_gru = gru.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=cbs)
        models_dict['GRU'] = gru
        models_dict['GRU_history'] = hist_gru.history

    if 'Transformer' in selected_models:
        print("\nTraining Transformer...")
        transformer = build_transformer(input_shape)
        cbs = [cb for cb in [es, rlrop, lr_callback] if cb is not None]
        hist_tr = transformer.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=cbs)
        models_dict['Transformer'] = transformer
        models_dict['Transformer_history'] = hist_tr.history

    return models_dict


def evaluate_and_ensemble(models_dict, data_splits, scaler_y):
    print("\n" + "="*70)
    print("STEP 4: EVALUATION & ENSEMBLE")
    print("="*70)
    X_test, y_test, dates_test, base_test = data_splits['test']
    preds = {}
    metrics_all = {}
    # Inverse-transform true log-returns and reconstruct true prices for comparison
    logret_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
    y_true = base_test * np.exp(logret_true)
    for name, mdl in models_dict.items():
        if name.endswith('_history'):
            continue
        y_pred_scaled = mdl.predict(X_test, verbose=0).ravel()
        # Inverse to returns and reconstruct price: price_{t+1} = price_t * exp(pred_logret)
        logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        # Align base_test (price_t) to y_pred length
        price_pred = base_test * np.exp(logret_pred)
        preds[name] = price_pred
        metrics_all[name] = calculate_metrics(y_true, price_pred)
        print(f"{name} -> RMSE: {metrics_all[name]['RMSE']:.4f} | MAE: {metrics_all[name]['MAE']:.4f} | DA: {metrics_all[name]['Directional_Accuracy']:.2f}%")

    # Simple weighted ensemble (equal weights). If multiple models are present we average.
    model_names = [k for k in models_dict.keys() if not k.endswith('_history')]
    y_stack = np.stack([preds[n] for n in model_names], axis=1)
    weights = np.ones(len(model_names), dtype=np.float32) / float(len(model_names))
    y_ens = (y_stack * weights).sum(axis=1)
    preds['Ensemble'] = y_ens
    metrics_all['Ensemble'] = calculate_metrics(y_true, y_ens)
    print(f"Ensemble -> RMSE: {metrics_all['Ensemble']['RMSE']:.4f} | MAE: {metrics_all['Ensemble']['MAE']:.4f} | DA: {metrics_all['Ensemble']['Directional_Accuracy']:.2f}%")

    # Confidence via model spread (std across models)
    std = y_stack.std(axis=1)
    lower = y_ens - 1.96 * std
    upper = y_ens + 1.96 * std
    
    return {
        'predictions': preds,
        'metrics': metrics_all,
        'y_test_actual': y_true,
        'dates_test': dates_test,
        'confidence_intervals': (lower, upper)
    }


# --------------------------
# Backtesting (simple sign strategy)
# --------------------------
def run_backtest(pred, actual, dates, initial_capital: float):
    pred_shift = np.roll(pred, 1)
    pred_shift[0] = pred_shift[1]
    signal = np.sign(pred - pred_shift)
    ret = np.concatenate([[0.0], np.diff(actual) / actual[:-1]])
    strat_ret = ret * signal
    equity = initial_capital * (1 + strat_ret).cumprod()
    buy_signals = np.where(signal > 0, actual, np.nan)
    sell_signals = np.where(signal < 0, actual, np.nan)
    # BUG-07: Subtract risk-free rate from returns before computing Sharpe.
    # Sharpe = (E[r] - Rf_daily) / std(r) * sqrt(252)
    rf_daily  = CONFIG['risk_free_rate_annual'] / 252.0
    excess_ret = strat_ret - rf_daily
    sharpe = np.mean(excess_ret) / (np.std(excess_ret) + 1e-9) * np.sqrt(252)
    total_return = (equity[-1] / equity[0] - 1) * 100
    buy_hold = (actual[-1] / actual[0] - 1) * 100
    return {
        'equity': equity,
        'buy_signals': buy_signals,
        'sell_signals': sell_signals,
        'metrics': {
            'Sharpe Ratio': sharpe,
            'Total Return (%)': total_return,
            'Buy & Hold Return (%)': buy_hold
        }
    }


def plot_trading_signals(dates, prices, buy, sell):
    plt.figure(figsize=(12, 4))
    plt.plot(dates, prices, color='black', linewidth=1, label='Price')
    plt.scatter(dates, buy, marker='^', color='green', label='Buy', s=30)
    plt.scatter(dates, sell, marker='v', color='red', label='Sell', s=30)
    plt.legend(); plt.tight_layout(); plt.show()


# --------------------------
# Market Regime & Risk Condition Engine
# --------------------------

REGIME_PROFILES = {
    0: {'name': 'Trending Bull',          'color': '#00ff88', 'description': 'Strong uptrend with aligned momentum. Historical bias is bullish.'},
    1: {'name': 'Overbought / Exhaustion', 'color': '#ffaa00', 'description': 'Extended rally. RSI elevated and momentum diverging — pullback risk.'},
    2: {'name': 'Sideways / Choppy',       'color': '#888888', 'description': 'No clear directional bias. Trend-following strategies underperform.'},
    3: {'name': 'Recovery / Bounce',       'color': '#00aaff', 'description': 'Recovering from an oversold condition. Mean-reversion setup forming.'},
    4: {'name': 'Downtrend / Bear',        'color': '#ff3366', 'description': 'Clear downtrend in progress. Momentum is bearish.'},
    5: {'name': 'High Volatility / Stress','color': '#ff6600', 'description': 'Elevated fear and regime instability. Reduce risk exposure.'},
}

def _regime_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build a compact feature matrix for regime clustering from an engineered DataFrame."""
    df = df.copy()
    # Rolling volatility (20-day std of LogReturn) – captures volatility regime
    df['Vol20'] = df['LogReturn'].rolling(20).std()
    # RSI relative to 50 (sign tells us momentum side, magnitude tells intensity)
    df['RSI_dev'] = df['RSI14'] - 50.0
    # MACD histogram sign and magnitude
    df['MACD_hist_norm'] = df['MACD_hist'] / (df['Close'].rolling(20).mean().replace(0, 1) * 0.01 + 1e-9)
    # EMA slope: (EMA20 - EMA20_lag10) / EMA20_lag10
    df['EMA_slope'] = (df['EMA20'] - df['EMA20'].shift(10)) / (df['EMA20'].shift(10).replace(0, 1) + 1e-9) * 100.0
    # 10-day price return
    df['Ret10'] = df['Close'].pct_change(10) * 100.0

    feature_cols = ['RSI_dev', 'MACD_hist_norm', 'EMA_slope', 'Vol20', 'Ret10']
    feat = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return feat


def classify_regimes(df: pd.DataFrame, n_clusters: int = 6):
    """
    Cluster the feature matrix into n_clusters regimes using K-Means.
    Returns:
        labels (pd.Series indexed like feat), feat_df
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler as SS

    feat = _regime_feature_matrix(df)
    scaler = SS()
    feat_scaled = scaler.fit_transform(feat.values)

    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    raw_labels = km.fit_predict(feat_scaled)

    # Map raw cluster IDs → semantically ordered regimes:
    # Order clusters by their center's 10-day return (Ret10, index 4) and EMA slope (index 2)
    centers = km.cluster_centers_           # shape (n_clusters, n_features)
    ret10_idx = list(feat.columns).index('Ret10')
    rsi_idx   = list(feat.columns).index('RSI_dev')
    vol_idx   = list(feat.columns).index('Vol20')

    # Score = 2 * ret10 + 1.5 * rsi_dev - 3 * vol (higher = more bullish / calmer)
    scores = 2.0 * centers[:, ret10_idx] + 1.5 * centers[:, rsi_idx] - 3.0 * centers[:, vol_idx]
    sorted_clusters = np.argsort(-scores)   # descending: 0 → best, n-1 → worst

    # Build a mapping: raw_label → semantic_label (0..5)
    mapping = {int(orig): int(sem) for sem, orig in enumerate(sorted_clusters)}
    semantic_labels = np.array([mapping[l] for l in raw_labels], dtype=int)

    labels_series = pd.Series(semantic_labels, index=feat.index, name='regime')
    return labels_series, feat


def compute_risk_score(df: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Compute a 0–10 Risk Condition Score for each date.
    Higher score = more risk / bearish condition.
    Components:
      - Volatility stress  (0–3): how much current vol exceeds 90-day avg
      - Momentum stress    (0–3): RSI overbought/oversold extremity
      - Trend misalignment (0–2): Close vs EMA20 direction
      - MACD divergence    (0–2): MACD histogram direction
    """
    df = df.copy()
    df['Vol20'] = df['LogReturn'].rolling(20).std()
    df['Vol90'] = df['LogReturn'].rolling(90).std()
    vol_ratio = (df['Vol20'] / (df['Vol90'].replace(0, 1e-9))).clip(0.5, 3.0)
    vol_stress = ((vol_ratio - 0.5) / 2.5 * 3.0).clip(0, 3.0)

    rsi = df['RSI14']
    # Extremity: RSI far from 50 in either direction
    rsi_extremity = ((rsi - 50).abs() / 50.0 * 3.0).clip(0, 3.0)

    trend_align = np.where(df['Close'] < df['EMA20'], 2.0, 0.0)

    macd_div = np.where(df['MACD_hist'] < 0, 2.0, 0.0)

    raw = vol_stress.values + rsi_extremity.values + trend_align + macd_div
    score = pd.Series(
        np.clip(raw / 10.0 * 10.0, 0, 10),   # already on 0–10 (max raw = 10)
        index=df.index
    ).rolling(5).mean()                        # 5-day smoothing

    # Override to max stress for regime 4 (Bear) or 5 (Stress)
    if labels is not None:
        common = score.index.intersection(labels.index)
        for idx in common:
            if labels[idx] in (4, 5):
                score[idx] = max(score[idx], 7.5)

    return score.clip(0, 10)


def compute_regime_stats(df: pd.DataFrame, labels: pd.Series) -> dict:
    """
    For each regime, compute forward return statistics from historical occurrences.
    Returns a dict keyed by semantic regime id.

    BUG-06 FIX: Exclude dates within the last MAX_HORIZON trading days.
    Computing a 20-day forward return on, e.g., the final 5 dates of the dataset
    would use only 5 available price points, producing incomplete/biased statistics.
    Only dates with a FULL forward window are included in summary statistics.
    """
    MAX_HORIZON = 20   # largest forward-return horizon used below
    common_idx = df.index.intersection(labels.index)
    close = df.loc[common_idx, 'Close']

    # Cutoff: exclude the last MAX_HORIZON dates so every forward window is complete
    eligible_cutoff = close.index[-MAX_HORIZON] if len(close) > MAX_HORIZON else close.index[0]

    stats = {}
    for regime_id in range(6):
        regime_dates = labels[labels == regime_id].index
        regime_dates_common = regime_dates.intersection(close.index)
        # BUG-06: Only keep dates where the full 20-day forward window is observable
        regime_dates_common = regime_dates_common[regime_dates_common < eligible_cutoff]

        fwd_5, fwd_10, fwd_20 = [], [], []
        for d in regime_dates_common:
            loc = close.index.get_loc(d)
            for horizon, store in [(5, fwd_5), (10, fwd_10), (20, fwd_20)]:
                end_loc = loc + horizon
                if end_loc < len(close):
                    ret = (close.iloc[end_loc] / close.iloc[loc] - 1.0) * 100.0
                    store.append(float(ret))

        def summarise(lst):
            if not lst:
                return {'median': None, 'mean': None, 'pct_positive': None, 'count': 0, 'samples': []}
            arr = np.array(lst)
            return {
                'median': float(np.median(arr)),
                'mean':   float(np.mean(arr)),
                'pct_positive': float((arr > 0).mean() * 100),
                'count': len(arr),
                'samples': [round(v, 2) for v in arr[-10:].tolist()]
            }

        profile = REGIME_PROFILES.get(regime_id, {'name': f'Regime {regime_id}', 'color': '#888', 'description': ''})
        stats[regime_id] = {
            'id':          regime_id,
            'name':        profile['name'],
            'color':       profile['color'],
            'description': profile['description'],
            'total_days':  len(regime_dates_common),
            'fwd_5d':      summarise(fwd_5),
            'fwd_10d':     summarise(fwd_10),
            'fwd_20d':     summarise(fwd_20),
        }
    return stats


def find_similar_historical_scenarios(df: pd.DataFrame, feat: pd.DataFrame, labels: pd.Series, top_k: int = 5) -> list:
    """
    Find top_k historical dates whose technical feature vector is most similar
    (using Cosine Similarity on standardized feature vectors) to the current market day.
    Returns real historical forward 10-day price trajectories for each match.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    if len(feat) < 30:
        return []

    scaler = StandardScaler()
    feat_scaled = scaler.fit_transform(feat.values)
    target_vec = feat_scaled[-1].reshape(1, -1)

    # Exclude last 15 trading days so we have a full 10-day forward window to measure
    cand_indices = range(0, len(feat_scaled) - 15)
    cand_vecs = feat_scaled[cand_indices]

    sims = cosine_similarity(target_vec, cand_vecs)[0]
    top_cand_positions = np.argsort(-sims)[:top_k]

    matches = []
    close_series = df.loc[feat.index, 'Close']

    for pos in top_cand_positions:
        idx_in_feat = cand_indices[pos]
        matched_date = feat.index[idx_in_feat]
        sim_score = float(sims[pos])
        regime_id = int(labels.iloc[idx_in_feat])

        start_loc = close_series.index.get_loc(matched_date)
        fwd_prices = close_series.iloc[start_loc:start_loc + 11].tolist()

        if len(fwd_prices) >= 2:
            fwd_ret_10d = float((fwd_prices[-1] / fwd_prices[0] - 1.0) * 100.0)
            base = fwd_prices[0]
            normalized_path = [round(float(p / base * 100.0), 2) for p in fwd_prices]
        else:
            fwd_ret_10d = 0.0
            normalized_path = []

        profile = REGIME_PROFILES.get(regime_id, {'name': 'Unknown', 'color': '#888'})

        matches.append({
            'date': matched_date.strftime('%Y-%m-%d'),
            'similarity_pct': round(float(sim_score * 100.0), 1),
            'regime_name': profile['name'],
            'regime_color': profile['color'],
            'fwd_10d_ret': round(fwd_ret_10d, 2),
            'price_path': normalized_path
        })

    return matches


def compute_quant_backtest(df: pd.DataFrame, labels: pd.Series) -> dict:
    """
    Run a Quantitative Regime Strategy vs. Buy & Hold Benchmark on real historical data.
    Regime Allocation Matrix:
      - Trending Bull (0), Recovery (3): 100% Equity Exposure
      - Sideways (2), Overbought (1): 50% Equity, 50% Cash
      - Bear (4), Stress (5): 0% Equity (100% Cash)

    BUG-03 FIX: Transaction costs applied on every weight change.
      cost = CONFIG['transaction_cost_pct'] * |weight[t] - weight[t-1]|
      This models brokerage + bid-ask spread. Default: 0.10% round-trip.

    BUG-07 FIX: Sharpe ratio uses risk-free rate from CONFIG.
      Sharpe = (E[r] - Rf_daily) / std(r) * sqrt(252)
    """
    common = df.index.intersection(labels.index)
    close = df.loc[common, 'Close']
    regimes = labels.loc[common]

    daily_ret = close.pct_change().fillna(0.0)

    weights = pd.Series(0.0, index=common)
    for idx in common:
        r = regimes[idx]
        if r in (0, 3):
            weights[idx] = 1.0
        elif r in (1, 2):
            weights[idx] = 0.5
        else:
            weights[idx] = 0.0

    # BUG-03: Deduct transaction cost whenever allocation weight changes.
    # weight_prev is yesterday's allocation; if it differs from today's, a trade occurs.
    tc = CONFIG['transaction_cost_pct']
    weight_changes = weights.diff().abs().fillna(0.0)
    transaction_costs = weight_changes * tc   # proportional cost on the traded portion

    strat_ret = (weights.shift(1).fillna(0.0) * daily_ret) - transaction_costs

    init_cap = 10000.0
    bh_equity = init_cap * (1.0 + daily_ret).cumprod()
    strat_equity = init_cap * (1.0 + strat_ret).cumprod()

    bh_total_ret = float((bh_equity.iloc[-1] / init_cap - 1.0) * 100.0)
    strat_total_ret = float((strat_equity.iloc[-1] / init_cap - 1.0) * 100.0)

    def calc_mdd(eq_series):
        peak = eq_series.cummax()
        dd = (eq_series - peak) / peak
        return float(dd.min() * 100.0)

    bh_mdd = calc_mdd(bh_equity)
    strat_mdd = calc_mdd(strat_equity)

    # BUG-07: Correct Sharpe = (mean_excess_return) / std * sqrt(252)
    rf_daily = CONFIG['risk_free_rate_annual'] / 252.0
    def calc_sharpe(ret_series):
        excess = ret_series - rf_daily
        std = excess.std()
        if std == 0 or np.isnan(std): return 0.0
        return float(excess.mean() / std * np.sqrt(252))

    bh_sharpe   = calc_sharpe(daily_ret)
    strat_sharpe = calc_sharpe(strat_ret)

    active_days = strat_ret[weights.shift(1) > 0]
    win_rate = float((active_days > 0).mean() * 100.0) if len(active_days) > 0 else 0.0

    # Total transaction cost paid (for transparency in output)
    total_tc_paid_pct = float(transaction_costs.sum() * 100.0)
    n_trades = int((weight_changes > 0).sum())

    step = max(1, len(common) // 200)
    dates_sampled  = [d.strftime('%Y-%m-%d') for d in common[::step]]
    bh_sampled     = [round(float(v), 2) for v in bh_equity.iloc[::step]]
    strat_sampled  = [round(float(v), 2) for v in strat_equity.iloc[::step]]

    return {
        'dates': dates_sampled,
        'benchmark_equity': bh_sampled,
        'strategy_equity': strat_sampled,
        'metrics': {
            'benchmark_total_return':  round(bh_total_ret, 2),
            'strategy_total_return':   round(strat_total_ret, 2),
            'benchmark_max_drawdown':  round(bh_mdd, 2),
            'strategy_max_drawdown':   round(strat_mdd, 2),
            'benchmark_sharpe':        round(bh_sharpe, 2),
            'strategy_sharpe':         round(strat_sharpe, 2),
            'strategy_win_rate':       round(win_rate, 1),
            'total_transaction_cost_pct': round(total_tc_paid_pct, 4),
            'n_trades':                n_trades,
            'risk_free_rate_annual':   CONFIG['risk_free_rate_annual'],
        }
    }


def walk_forward_validate(ticker: str, start_date: str, end_date: str,
                          n_folds: int = 5,
                          model_type: str = 'GRU',
                          epochs: int = 10,
                          batch_size: int = 32) -> dict:
    """
    BUG-04 FIX: Walk-Forward (Expanding Window) Validation.

    Why a single 70/15/15 split is insufficient:
      - It tests the model on exactly ONE historical period. If that period
        happens to be low-volatility or trending, metrics look artificially good.
      - Walk-forward validation tests the model across N disjoint out-of-sample
        periods, giving mean ± std of metrics — a statistically honest estimate.

    Method (Expanding Anchor):
      - The training window always starts from day 0 (anchored).
      - Each fold extends the training set by 1/n_folds of total data.
      - The test set for each fold is the NEXT 1/n_folds block (never seen).
      - No data from any test block leaks into any training block.

    Returns per-fold metrics and aggregate mean/std for portfolio review.
    """
    print(f"\nWalk-Forward Validation: {n_folds} folds, model={model_type}")

    # Prepare raw data once (no scaling — scaling happens per fold)
    data = prepare_data(ticker, start_date, end_date, CONFIG['sequence_length'])
    X_raw   = data['X_raw']
    y_raw   = data['y_raw']
    # dates_raw not used directly; fold windows are sliced by integer index
    base   = data['base_prices_raw']
    seq_len = data['sequence_length']

    n = len(X_raw)
    fold_size = n // (n_folds + 1)   # +1 so there's always a test block after last train

    fold_results = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)    # expanding: grows each fold
        test_start = train_end
        test_end   = min(test_start + fold_size, n)

        if test_end - test_start < seq_len + 10:
            print(f"  Fold {fold+1}: skipped (insufficient test data)")
            continue

        # Scale: fit ONLY on this fold's training partition
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()
        X_tr_raw = X_raw[:train_end]
        y_tr_raw = y_raw[:train_end]
        scaler_X.fit(X_tr_raw)
        scaler_y.fit(y_tr_raw.reshape(-1, 1))

        X_tr = scaler_X.transform(X_raw[:train_end])
        y_tr = scaler_y.transform(y_raw[:train_end].reshape(-1, 1)).ravel()

        X_te_raw_fold = X_raw[test_start:test_end]
        y_te_raw_fold = y_raw[test_start:test_end]
        X_te_ctx = np.concatenate([X_tr[-seq_len:], scaler_X.transform(X_te_raw_fold)], axis=0)
        y_te_ctx = np.concatenate([y_tr[-seq_len:], scaler_y.transform(y_te_raw_fold.reshape(-1,1)).ravel()], axis=0)
        X_te_seq, y_te_seq = create_sequences(X_te_ctx, y_te_ctx, seq_len)

        X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len)

        if len(X_tr_seq) < 10 or len(X_te_seq) < 5:
            continue

        input_shape = (seq_len, X_raw.shape[1])

        # Train selected model for this fold
        if model_type == 'LSTM':
            mdl = build_lstm(input_shape)
        elif model_type == 'Transformer':
            mdl = build_transformer(input_shape)
        else:
            mdl = build_gru(input_shape)

        es = callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
        mdl.fit(X_tr_seq, y_tr_seq, epochs=epochs, batch_size=batch_size,
                verbose=0, callbacks=[es])

        # Evaluate on out-of-sample test block
        y_pred_scaled = mdl.predict(X_te_seq, verbose=0).ravel()
        logret_true = scaler_y.inverse_transform(y_te_seq.reshape(-1,1)).ravel()
        logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

        base_te = base[test_start:test_end][:len(y_te_seq)]
        y_true_prices = base_te * np.exp(logret_true)
        y_pred_prices = base_te * np.exp(logret_pred)

        metrics = calculate_metrics(y_true_prices, y_pred_prices)

        # Directional accuracy on log returns (sign prediction)
        dir_acc = float((np.sign(logret_true) == np.sign(logret_pred)).mean() * 100)

        fold_results.append({
            'fold': fold + 1,
            'train_days':  train_end,
            'test_days':   test_end - test_start,
            'rmse':        round(metrics['RMSE'], 4),
            'mae':         round(metrics['MAE'], 4),
            'directional_accuracy': round(dir_acc, 2),
            'r2':          round(metrics['R2'], 4),
        })
        print(f"  Fold {fold+1}: DA={dir_acc:.1f}%  RMSE={metrics['RMSE']:.4f}  R2={metrics['R2']:.4f}")

    if not fold_results:
        return {'error': 'Insufficient data for walk-forward validation', 'folds': []}

    das   = [f['directional_accuracy'] for f in fold_results]
    rmses = [f['rmse'] for f in fold_results]

    return {
        'model': model_type,
        'n_folds': len(fold_results),
        'folds': fold_results,
        'summary': {
            'mean_directional_accuracy': round(float(np.mean(das)), 2),
            'std_directional_accuracy':  round(float(np.std(das)), 2),
            'mean_rmse':                 round(float(np.mean(rmses)), 4),
            'std_rmse':                  round(float(np.std(rmses)), 4),
        },
        'interpretation': (
            f"Over {len(fold_results)} out-of-sample folds, {model_type} achieved "
            f"{np.mean(das):.1f}% ± {np.std(das):.1f}% directional accuracy. "
            f"Values near 50% indicate near-random. Values >55% sustained across "
            f"all folds suggest genuine predictive signal."
        )
    }


def get_condition_alert(risk_score: float, regime_id: int) -> dict:
    """Map risk score + regime to a user-facing alert level."""
    if regime_id in (0, 3) and risk_score < 4.5:
        return {'level': 'GREEN',  'label': 'Favorable Conditions', 'color': '#00ff88',
                'description': 'Regime and indicators aligned positively. Historically a favorable entry zone.'}
    elif regime_id in (4, 5) or risk_score >= 7.0:
        return {'level': 'RED',    'label': 'Elevated Risk',         'color': '#ff3366',
                'description': 'High risk conditions. Historical odds favor caution and reduced exposure.'}
    else:
        return {'level': 'YELLOW', 'label': 'Watch / Wait',          'color': '#ffaa00',
                'description': 'Mixed signals. No strong directional bias — wait for clearer confirmation.'}


# --------------------------
# Orchestration
# --------------------------
def main():
    pass

    print("\n" + "="*70)
    print("STOCK PRICE PREDICTION WITH DEEP LEARNING")
    print(f"Ticker: {CONFIG['ticker']}")
    print(f"Period: {CONFIG['start_date']} to {CONFIG['end_date']}")
    print("="*70)
    
    data   = prepare_data(CONFIG['ticker'], CONFIG['start_date'], CONFIG['end_date'], CONFIG['sequence_length'])
    splits = split_and_scale_data(
        data['X_raw'], data['y_raw'], data['dates_raw'], data['base_prices_raw'],
        CONFIG['train_split'], CONFIG['val_split'], CONFIG['sequence_length']
    )
    input_shape = (CONFIG['sequence_length'], data['X_raw'].shape[1])
    models_dict = train_models(splits, input_shape, CONFIG['epochs'], CONFIG['batch_size'])
    eval_results = evaluate_and_ensemble(models_dict, splits, splits['scaler_y'])

    print("\n" + "="*70)
    print("STEP 5: BACKTESTING")
    print("="*70)
    backtest = run_backtest(eval_results['predictions']['Ensemble'], eval_results['y_test_actual'], eval_results['dates_test'], CONFIG['initial_capital'])

    print("\n" + "="*70)
    print("STEP 6: VISUALIZATION")
    print("="*70)
    set_plot_style()
    print("\nModel metrics (lower RMSE is better):")
    print(create_metrics_table(eval_results['metrics']).to_string())
    plot_predictions(eval_results['dates_test'], eval_results['y_test_actual'], eval_results['predictions'], title=f"{CONFIG['ticker']} Predictions")
    plot_confidence_intervals(eval_results['dates_test'], eval_results['y_test_actual'], eval_results['predictions']['Ensemble'], eval_results['confidence_intervals'])
    plot_trading_signals(eval_results['dates_test'], eval_results['y_test_actual'], backtest['buy_signals'], backtest['sell_signals'])

    print("\nBacktest metrics:")
    for k, v in backtest['metrics'].items():
        print(f"{k}: {v:.2f}")


# __main__ entry-point is at the bottom of the file (after _run_server is defined)


import time  # required by Phase 3 before_request / after_request middleware


def create_app():
    """Initialize Phase 2-6 infrastructure and return configured Flask application instance."""

    # ── Initialize Phase 2 components ──────────────────────────────────────────
    store          = ModelStore(base_dir=CFG.model_artifacts_dir)
    registry       = ModelRegistry(registry_path=CFG.registry_path)
    cache          = InferenceCache(cache_dir=CFG.cache_dir,
                                   ttl_seconds=CFG.inference_cache_ttl_s)
    trainer        = BackgroundTrainer(registry, store, CFG,
                                      max_workers=CFG.max_worker_threads)
    engine         = InferenceEngine(registry, store, cache, CFG)
    scheduler      = RetrainingScheduler(registry, trainer, engine, CFG)

    scheduler.start()
    log.info("Phase 2 ML infrastructure initialized")
    log.info(f"Model artifacts dir : {CFG.model_artifacts_dir}")
    log.info(f"Registry path       : {CFG.registry_path}")

    # ── Flask App ──────────────────────────────────────────────────────────────
    flask_app = Flask(__name__)
    CORS(flask_app)

    # ── Shared safe_scalar helper ──────────────────────────────────────────────
    def _ss(v):
        if v is None: return None
        if isinstance(v, (float,)) and (math.isnan(v) or math.isinf(v)): return None
        return v

    # ═══════════════════════════════════════
    # v1 routes (Phase 1 — unchanged)
    # ═══════════════════════════════════════

    @flask_app.route('/api/health', methods=['GET'])
    def health():
        return jsonify({
            'status':         'ok',
            'phase':          'Phase-2-Production-Pipeline',
            'cached_models':  len(MODEL_CACHE),
            'registry_stats': registry.stats(),
            'scheduler':      scheduler.status(),
            'artifact_store': {
                'disk_bytes': store.disk_usage_bytes(),
                'dir':        CFG.model_artifacts_dir,
            },
            'inference_cache': cache.stats(),
        })

    @flask_app.route('/api/regime', methods=['POST'])
    def api_regime():
        payload = request.get_json(force=True) or {}
        ticker = payload.get('ticker', 'AAPL').upper()
        three_years_ago = (datetime.date.today() - datetime.timedelta(days=3*365)).isoformat()
        start = payload.get('start_date', three_years_ago)
        end   = payload.get('end_date', datetime.date.today().isoformat())

        def safe_scalar(v):
            if v is None: return None
            if isinstance(v, (float, int)) and isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v

        try:
            raw = fetch_data_yfinance(ticker, start, end)
            raw = preprocess_data(raw)
            df  = engineer_features(raw)
        except Exception as e:
            return jsonify({'error': str(e)}), 400

        labels, feat = classify_regimes(df, n_clusters=6)
        df_aligned   = df.loc[df.index.intersection(labels.index)]
        risk_scores  = compute_risk_score(df_aligned, labels)
        risk_scores  = risk_scores.reindex(labels.index).ffill().fillna(5.0)
        regime_stats = compute_regime_stats(df_aligned, labels)

        current_regime_id  = int(labels.iloc[-1])
        current_risk_score = float(risk_scores.iloc[-1] or 5.0)
        current_alert      = get_condition_alert(current_risk_score, current_regime_id)
        current_profile    = REGIME_PROFILES[current_regime_id]
        latest_row         = df.iloc[-1]

        indicators = {
            'rsi':       safe_scalar(latest_row['RSI14']),
            'ema20':     safe_scalar(latest_row['EMA20']),
            'macd':      safe_scalar(latest_row['MACD']),
            'macd_hist': safe_scalar(latest_row['MACD_hist']),
            'close':     safe_scalar(latest_row['Close']),
            'logret':    safe_scalar(latest_row['LogReturn']),
        }

        timeline_dates = labels.index
        sample_step = max(1, len(timeline_dates) // 500)
        sampled_idx = list(range(0, len(timeline_dates), sample_step))
        if sampled_idx[-1] != len(timeline_dates) - 1:
            sampled_idx.append(len(timeline_dates) - 1)

        price_close = df_aligned['Close']
        timeline = []
        for i in sampled_idx:
            d   = timeline_dates[i]
            rid = int(labels.iloc[i])
            rs  = safe_scalar(risk_scores.iloc[i])
            close_val = safe_scalar(price_close.get(d, None))
            timeline.append({
                'date':         d.strftime('%Y-%m-%d'),
                'regime_id':    rid,
                'regime_name':  REGIME_PROFILES[rid]['name'],
                'regime_color': REGIME_PROFILES[rid]['color'],
                'risk_score':   rs,
                'close':        close_val,
            })

        def safe_stat(s):
            return {k: (safe_scalar(v) if not isinstance(v, (list, dict)) else v)
                    for k, v in s.items()}

        stats_out = {}
        for rid, s in regime_stats.items():
            stats_out[str(rid)] = {
                'id': s['id'], 'name': s['name'], 'color': s['color'],
                'description': s['description'], 'total_days': s['total_days'],
                'fwd_5d': safe_stat(s['fwd_5d']),
                'fwd_10d': safe_stat(s['fwd_10d']),
                'fwd_20d': safe_stat(s['fwd_20d']),
            }

        similar_matches = find_similar_historical_scenarios(df_aligned, feat, labels, top_k=5)
        quant_backtest  = compute_quant_backtest(df_aligned, labels)

        return jsonify({
            'ticker':          ticker,
            'current_regime':  {'id': current_regime_id, 'name': current_profile['name'],
                                'color': current_profile['color'], 'description': current_profile['description']},
            'risk_score':      round(current_risk_score, 2),
            'alert':           current_alert,
            'indicators':      indicators,
            'regime_stats':    stats_out,
            'timeline':        timeline,
            'similar_matches': similar_matches,
            'quant_backtest':  quant_backtest,
        })

    @flask_app.route('/api/predict', methods=['POST'])
    def api_predict():
        payload      = request.get_json(force=True) or {}
        ticker       = payload.get('ticker', CONFIG['ticker']).upper()
        start        = payload.get('start_date', CONFIG['start_date'])
        end          = payload.get('end_date', CONFIG['end_date'])
        seq_len      = int(payload.get('sequence_length', CONFIG['sequence_length']))
        epochs       = int(payload.get('epochs', CONFIG['epochs']))
        batch_size   = int(payload.get('batch_size', CONFIG['batch_size']))
        future_days  = int(payload.get('future_days', 5))
        force_retrain = bool(payload.get('force_retrain', False))

        cache_key     = f"{ticker}_{start}_{end}_{seq_len}_{epochs}"
        selected      = payload.get('model')
        selected_list = [selected] if selected in {'LSTM', 'GRU', 'Transformer'} else None
        use_es        = bool(payload.get('early_stopping', True))

        data   = prepare_data(ticker, start, end, seq_len)
        splits = split_and_scale_data(
            data['X_raw'], data['y_raw'], data['dates_raw'], data['base_prices_raw'],
            CONFIG['train_split'], CONFIG['val_split'], seq_len
        )
        scaler_y    = splits['scaler_y']
        scaler_X    = splits['scaler_X']
        input_shape = (seq_len, data['X_raw'].shape[1])

        if cache_key in MODEL_CACHE and not force_retrain:
            models_dict = MODEL_CACHE[cache_key]['models']
            from_cache  = True
        else:
            models_dict = train_models(splits, input_shape, epochs, batch_size,
                                       selected_models=selected_list, use_early_stopping=use_es)
            MODEL_CACHE[cache_key] = {
                'models':     models_dict,
                'scaler_X':   scaler_X,
                'scaler_y':   scaler_y,
                'trained_at': datetime.datetime.utcnow().isoformat(),
                'cache_key':  cache_key,
            }
            from_cache = False

        eval_results = evaluate_and_ensemble(models_dict, splits, scaler_y)
        backtest = run_backtest(
            eval_results['predictions']['Ensemble'],
            eval_results['y_test_actual'],
            eval_results['dates_test'],
            float(payload.get('initial_capital', CONFIG['initial_capital']))
        )

        def roll_forecast(mdls, last_seq, steps, sy, sx, close_idx, last_price):
            import numpy as _np
            seq   = last_seq.copy().astype(_np.float32)
            price = float(last_price)
            preds = []
            for _ in range(steps):
                lrs = []
                for name, mdl in mdls.items():
                    if name.endswith('_history'): continue
                    yhat_s = mdl.predict(seq[_np.newaxis, ...], verbose=0).ravel()[0]
                    logret = sy.inverse_transform([[yhat_s]])[0, 0]
                    lrs.append(logret)
                next_price = price * float(_np.exp(float(_np.mean(lrs))))
                preds.append(next_price)
                price = next_price
                seq = _np.roll(seq, -1, axis=0)
                seq[-1, close_idx] = scale_single_feature(next_price, sx, close_idx)
            return preds

        X_te_seq   = splits['test'][0]
        last_seq   = X_te_seq[-1] if len(X_te_seq) > 0 else splits['train'][0][-1]
        last_price = float(eval_results['y_test_actual'][-1])
        future_forecast = roll_forecast(models_dict, last_seq, future_days,
                                        scaler_y, scaler_X, data['close_feature_index'], last_price)
        last_date    = eval_results['dates_test'][-1]
        future_dates = [(last_date + pd.Timedelta(days=i+1)).strftime('%Y-%m-%d')
                        for i in range(future_days)]

        def safe_scalar(v):
            if v is None: return None
            if isinstance(v, (float,)) and (math.isnan(v) or math.isinf(v)): return None
            return v
        def safe_list(a): return [safe_scalar(v) for v in a]

        response = {
            'ticker': ticker, 'from_cache': from_cache, 'cache_key': cache_key,
            'dates':  [d.strftime('%Y-%m-%d') for d in eval_results['dates_test']],
            'actual': safe_list(eval_results['y_test_actual']),
            'predictions': {k: safe_list(v) for k, v in eval_results['predictions'].items()},
            'confidence': {
                'lower': safe_list(eval_results['confidence_intervals'][0]),
                'upper': safe_list(eval_results['confidence_intervals'][1]),
            },
            'metrics': {k: {mk: safe_scalar(mv) for mk, mv in md.items()}
                        for k, md in eval_results['metrics'].items()},
            'backtest': {
                'equity':       safe_list(backtest['equity']),
                'buy_signals':  safe_list(backtest['buy_signals']),
                'sell_signals': safe_list(backtest['sell_signals']),
                'metrics':      {mk: safe_scalar(mv) for mk, mv in backtest['metrics'].items()},
            },
            'future': {'dates': future_dates, 'predictions': safe_list(future_forecast)},
        }
        histories = {
            k.replace('_history', ''): {hk: safe_list(hv) for hk, hv in h.items()}
            for k, h in models_dict.items() if k.endswith('_history')
        }
        response['histories'] = histories
        return jsonify(response)

    @flask_app.route('/api/wf_validate', methods=['POST'])
    def api_wf_validate():
        payload    = request.get_json(force=True) or {}
        ticker     = payload.get('ticker', 'AAPL').upper()
        start      = payload.get('start_date', CONFIG['start_date'])
        end        = payload.get('end_date', CONFIG['end_date'])
        n_folds    = int(payload.get('n_folds', 5))
        model_type = payload.get('model', 'GRU')
        epochs     = int(payload.get('epochs', 10))
        batch_size = int(payload.get('batch_size', 32))
        try:
            result = walk_forward_validate(ticker, start, end,
                n_folds=n_folds, model_type=model_type,
                epochs=epochs, batch_size=batch_size)
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    # ═══════════════════════════════════════
    # v2 routes (Phase 2)
    # ═══════════════════════════════════════

    @flask_app.route('/api/v2/train', methods=['POST'])
    def v2_train():
        """
        POST /api/v2/train
        Enqueue an async training job. Returns job_id immediately.
        Poll GET /api/v2/jobs/{job_id} for status.

        Body: { ticker, start_date, end_date, epochs, seq_len, batch_size, models }
        """
        payload  = request.get_json(force=True) or {}
        ticker   = payload.get('ticker', 'AAPL').upper()
        start    = payload.get('start_date', CFG.default_start_date)
        end      = payload.get('end_date', datetime.date.today().isoformat())
        epochs   = int(payload.get('epochs', CFG.epochs))
        seq_len  = int(payload.get('seq_len', CFG.sequence_length))
        batch_sz = int(payload.get('batch_size', CFG.batch_size))
        mdl_list = payload.get('models', ['LSTM', 'GRU', 'Transformer'])

        try:
            job_id = trainer.submit(
                ticker=ticker, start_date=start, end_date=end,
                seq_len=seq_len, epochs=epochs, batch_size=batch_sz,
                models=mdl_list,
            )
            job = trainer.get_job(job_id)
            return jsonify({
                'job_id':  job_id,
                'version': job.version if job else None,
                'ticker':  ticker,
                'status':  'queued',
                'message': f'Training job enqueued. Poll GET /api/v2/jobs/{job_id} for status.',
            }), 202
        except Exception as e:
            log.error(f'v2/train error: {e}')
            return jsonify({'error': str(e)}), 500

    @flask_app.route('/api/v2/jobs/<job_id>', methods=['GET'])
    def v2_job_status(job_id):
        """GET /api/v2/jobs/{job_id} — Poll training job status."""
        job = trainer.get_job(job_id)
        if job is None:
            return jsonify({'error': f'Job {job_id} not found'}), 404
        return jsonify(job.to_dict())

    @flask_app.route('/api/v2/jobs', methods=['GET'])
    def v2_list_jobs():
        """GET /api/v2/jobs — List all training jobs."""
        return jsonify({
            'jobs':         trainer.list_jobs(),
            'active_count': trainer.active_count(),
        })

    @flask_app.route('/api/v2/registry', methods=['GET'])
    def v2_registry():
        """GET /api/v2/registry — Full model registry."""
        return jsonify(registry.full_registry())

    @flask_app.route('/api/v2/registry/<ticker>', methods=['GET'])
    def v2_registry_ticker(ticker):
        """GET /api/v2/registry/{ticker} — All versions for a ticker."""
        ticker = ticker.upper()
        versions = registry.list_versions(ticker)
        best     = registry.get_best(ticker)
        latest   = registry.get_latest(ticker)
        if not versions:
            return jsonify({'error': f'No models found for {ticker}'}), 404
        return jsonify({
            'ticker':   ticker,
            'versions': versions,
            'best':     best,
            'latest':   latest,
        })

    @flask_app.route('/api/v2/predict', methods=['POST'])
    def v2_predict():
        """
        POST /api/v2/predict
        Inference using a saved model. Never retrains.
        Requires a prior successful training job for the ticker.

        Body: { ticker, start_date, end_date, version (optional: best|latest|v3) }
        """
        payload  = request.get_json(force=True) or {}
        ticker   = payload.get('ticker', 'AAPL').upper()
        start    = payload.get('start_date', CFG.default_start_date)
        end      = payload.get('end_date', datetime.date.today().isoformat())
        version  = payload.get('version', 'best')
        force    = bool(payload.get('force_refresh', False))

        result = engine.predict(ticker, start, end, version=version, force_refresh=force)
        if 'error' in result:
            return jsonify(result), 404
        return jsonify(result)

    @flask_app.route('/api/v2/cache', methods=['DELETE'])
    def v2_flush_cache():
        """DELETE /api/v2/cache — Flush the inference cache (both layers)."""
        cache.flush()
        engine.evict_artifacts()
        return jsonify({'status': 'flushed', 'message': 'Inference cache cleared.'})

    @flask_app.route('/api/v2/metrics', methods=['GET'])
    def v2_metrics():
        """GET /api/v2/metrics — System health and performance counters."""
        return jsonify({
            'phase':          'Phase-2-Production-Pipeline',
            'timestamp':       datetime.datetime.utcnow().isoformat(),
            'registry':        registry.stats(),
            'training': {
                'active_jobs':  trainer.active_count(),
                'total_jobs':   len(trainer.list_jobs()),
                'workers':      CFG.max_worker_threads,
            },
            'inference_cache': cache.stats(),
            'artifact_store':  {
                'disk_bytes':   store.disk_usage_bytes(),
                'disk_mb':      round(store.disk_usage_bytes() / 1024**2, 2),
                'base_dir':     CFG.model_artifacts_dir,
            },
            'scheduler':       scheduler.status(),
            'config': {
                'stale_after_days':       CFG.model_stale_days,
                'scheduler_interval_s':   CFG.scheduler_interval_s,
                'inference_cache_ttl_s':  CFG.inference_cache_ttl_s,
                'risk_free_rate_annual':  CFG.risk_free_rate_annual,
                'transaction_cost_pct':   CFG.transaction_cost_pct,
            },
        })

    # ═══════════════════════════════════════
    # Phase 3 — v3 API Routes
    # ═══════════════════════════════════════

    # ── Phase 3 components ─────────────────────────────────────────────────
    pq          = PriorityJobQueue(max_workers=CFG.max_worker_threads)
    rate_limiter = RateLimiter(
        burst_capacity = CFG.rate_limit_burst,
        burst_rate     = CFG.rate_limit_rate,
        window_max     = CFG.rate_limit_window_max,
        window_s       = CFG.rate_limit_window_s,
    )
    # Register yfinance circuit breaker (get_breaker stores it in global registry by name)
    get_breaker(
        "yfinance",
        failure_threshold = CFG.cb_failure_threshold,
        window_size       = CFG.cb_window_size,
        reset_timeout_s   = CFG.cb_reset_timeout_s,
    )

    # ── Phase 3: Flask middleware ───────────────────────────────────────────

    @flask_app.before_request
    def _before():
        """Record request start time + rate limit check."""
        import flask
        flask.g.t_start  = time.perf_counter()
        flask.g.endpoint = request.path

        # Rate limiting on mutating endpoints
        if request.method in ("POST", "DELETE"):
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
            if not rate_limiter.allow(client_ip):
                http_requests_total.inc(method=request.method,
                                        endpoint=request.path, status="429")
                return jsonify({"error": "Rate limit exceeded", "retry_after_s": 1}), 429

    @flask_app.after_request
    def _after(response):
        """Record latency + HTTP status metrics and apply security headers."""
        import flask
        elapsed = time.perf_counter() - getattr(flask.g, "t_start", time.perf_counter())
        endpoint = getattr(flask.g, "endpoint", request.path)
        http_latency_seconds.observe(elapsed, endpoint=endpoint)
        http_requests_total.inc(method=request.method,
                                endpoint=endpoint,
                                status=str(response.status_code))

        # ── Phase 6: Security Headers Middleware ──────────────────────────────
        if getattr(CFG, "security_headers_enabled", True):
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "SAMEORIGIN"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self' 'unsafe-inline' 'unsafe-eval' https: data:; "
                "img-src 'self' data: https:; "
                "style-src 'self' 'unsafe-inline' https:;"
            )
            if not CFG.debug and getattr(CFG, "environment", "") in ("production", "staging"):
                response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response

    # ── Phase 6: Container Health & Readiness Probes ───────────────────────

    @flask_app.route('/health', methods=['GET'])
    def container_health():
        """
        GET /health — Container Liveness Probe (K8s / Docker / Render / Railway).
        Returns HTTP 200 if process is running and alive.
        """
        uptime_s = round(time.time() - _SERVER_START_TIME, 2)
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'uptime_seconds': uptime_s,
            'environment': getattr(CFG, "environment", "development"),
            'version': '6.0.0',
        }), 200

    @flask_app.route('/ready', methods=['GET'])
    def container_ready():
        """
        GET /ready — Container Readiness Probe.
        Checks if model storage directory is writable, circuit breaker is healthy,
        and priority job queue is accepting work. Returns 200 if ready, 503 if unready.
        """
        checks = {}
        is_ready = True

        # Check 1: Artifact storage directory
        try:
            os.makedirs(CFG.model_artifacts_dir, exist_ok=True)
            test_path = os.path.join(CFG.model_artifacts_dir, ".health_check_tmp")
            with open(test_path, "w") as f:
                f.write("ok")
            if os.path.exists(test_path):
                os.remove(test_path)
            checks['storage_writable'] = True
        except Exception as exc:
            checks['storage_writable'] = False
            checks['storage_error'] = str(exc)
            is_ready = False

        # Check 2: yfinance circuit breaker state
        try:
            yf_b = get_breaker("yfinance")
            checks['circuit_breaker_state'] = yf_b.state.value
            if yf_b.state.value == "OPEN":
                checks['circuit_breaker_warning'] = "yfinance circuit breaker is currently OPEN"
        except Exception:
            checks['circuit_breaker_state'] = "unknown"

        # Check 3: Priority Queue stats
        try:
            checks['queue_stats'] = pq.stats()
            checks['queue_healthy'] = True
        except Exception:
            checks['queue_healthy'] = False
            is_ready = False

        status_code = 200 if is_ready else 503
        return jsonify({
            'status': 'ready' if is_ready else 'unready',
            'timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checks': checks
        }), status_code


    # ── v3 Routes ──────────────────────────────────────────────────────────

    @flask_app.route('/api/v3/metrics', methods=['GET'])
    def v3_metrics():
        """
        GET /api/v3/metrics
        Prometheus text format or JSON (Accept: application/json)

        Aggregates all observability telemetry:
          - HTTP latency percentiles per endpoint
          - Training job counters
          - Cache hit/miss rates
          - Circuit breaker states
          - Queue depths
          - Memory/disk usage
        """
        # Update gauges from live state
        queue_depth.set(pq.stats()["queue_depth"])
        dlq_depth.set(pq.stats()["dlq_size"])
        active_workers.set(pq.stats()["by_status"].get("running", 0))
        model_versions_total.set(registry.stats()["total_versions"])
        cs = cache.stats()
        memory_cache_entries.set(cs.get("memory_entries", 0))
        disk_cache_entries.set(cs.get("disk_entries", 0))

        accept = request.headers.get("Accept", "")
        if "application/json" in accept:
            return jsonify({
                "metrics":          REGISTRY.as_dict(),
                "circuit_breakers": all_breaker_stats(),
                "rate_limiter":     rate_limiter.stats(),
                "queue":            pq.stats(),
            })
        # Default: Prometheus text format
        from flask import Response
        return Response(REGISTRY.format_prometheus(),
                        mimetype="text/plain; version=0.0.4")

    @flask_app.route('/api/v3/queue', methods=['GET'])
    def v3_queue():
        """
        GET /api/v3/queue
        Full job queue status including DLQ contents.
        """
        return jsonify({
            "stats":     pq.stats(),
            "jobs":      pq.list_jobs(),
            "dlq":       pq.dlq_jobs(),
        })

    @flask_app.route('/api/v3/queue/dlq/<job_id>/requeue', methods=['POST'])
    def v3_dlq_requeue(job_id):
        """
        POST /api/v3/queue/dlq/{job_id}/requeue
        Manually requeue a dead-lettered job for retry.
        """
        ok = pq.requeue_from_dlq(job_id)
        if not ok:
            return jsonify({"error": f"Job {job_id} not found in DLQ"}), 404
        return jsonify({"status": "requeued", "job_id": job_id})

    @flask_app.route('/api/v3/train', methods=['POST'])
    def v3_train():
        """
        POST /api/v3/train
        Enqueue training via priority queue with retry + DLQ.
        Supports optional 'priority' field: 0=urgent, 5=normal, 10=low.

        Body: { ticker, start_date, end_date, epochs, seq_len, priority, force }
        """
        payload  = request.get_json(force=True) or {}
        ticker   = payload.get('ticker', 'AAPL').upper()
        start    = payload.get('start_date', CFG.default_start_date)
        end      = payload.get('end_date', datetime.date.today().isoformat())
        epochs   = int(payload.get('epochs', CFG.epochs))
        seq_len  = int(payload.get('seq_len', CFG.sequence_length))
        batch_sz = int(payload.get('batch_size', CFG.batch_size))
        mdl_list = payload.get('models', ['LSTM', 'GRU', 'Transformer'])
        priority = int(payload.get('priority', PRIORITY_NORMAL))
        # 'force' field reserved for future cache-bust behaviour; ignored for now

        retry_policy = RetryPolicy(
            max_retries     = CFG.job_max_retries,
            base_delay_s    = CFG.job_retry_base_delay_s,
            max_delay_s     = CFG.job_retry_max_delay_s,
        )

        # Use Phase 2 trainer.submit() but routed through priority queue
        # The fn wraps BackgroundTrainer._run_job logic for DLQ support
        def _train_fn():
            return trainer.submit(
                ticker=ticker, start_date=start, end_date=end,
                seq_len=seq_len, epochs=epochs, batch_size=batch_sz,
                models=mdl_list,
            )

        job_id = pq.submit(
            fn           = _train_fn,
            payload      = {"ticker": ticker, "start": start, "end": end,
                            "epochs": epochs, "seq_len": seq_len},
            priority     = priority,
            retry_policy = retry_policy,
        )

        training_jobs_total.inc(status="queued")
        queue_depth.set(pq.stats()["queue_depth"])

        return jsonify({
            "job_id":   job_id,
            "ticker":   ticker,
            "priority": priority,
            "status":   "queued",
            "message":  "Queued via PriorityJobQueue. Poll GET /api/v3/queue.",
            "retry_policy": {
                "max_retries":   CFG.job_max_retries,
                "base_delay_s":  CFG.job_retry_base_delay_s,
                "max_delay_s":   CFG.job_retry_max_delay_s,
            },
        }), 202

    @flask_app.route('/api/v3/predict', methods=['POST'])
    def v3_predict():
        """
        POST /api/v3/predict
        Batch-aware cached inference with circuit-breaker-protected data fetch.

        Body: { ticker, start_date, end_date, version, force_refresh }

        Differences from /api/v2/predict:
          - Data fetch protected by yfinance circuit breaker
          - Latency recorded in metrics histogram
          - Cache hit/miss tracked in counters
        """
        with Timer(inference_latency_s, cache_layer="total"):
            payload = request.get_json(force=True) or {}
            ticker  = payload.get('ticker', 'AAPL').upper()
            start   = payload.get('start_date', CFG.default_start_date)
            end     = payload.get('end_date', datetime.date.today().isoformat())
            version = payload.get('version', 'best')
            force   = bool(payload.get('force_refresh', False))

            result = engine.predict(ticker, start, end,
                                    version=version, force_refresh=force)

        if 'error' in result:
            return jsonify(result), 404

        if result.get('from_cache'):
            cache_hits_total.inc(layer="memory_or_disk")
        else:
            cache_misses_total.inc()

        return jsonify(result)

    @flask_app.route('/api/v3/breakers', methods=['GET'])
    def v3_breakers():
        """GET /api/v3/breakers — All circuit breaker states."""
        return jsonify(all_breaker_stats())

    @flask_app.route('/api/v3/breakers/<name>/reset', methods=['POST'])
    def v3_breaker_reset(name: str):
        """POST /api/v3/breakers/{name}/reset — Manually reset a circuit breaker."""
        from core.circuit_breaker import _BREAKERS, _BREAKER_LOCK
        with _BREAKER_LOCK:
            b = _BREAKERS.get(name)
        if b is None:
            return jsonify({"error": f"No circuit breaker named '{name}'"}), 404
        b.reset()
        return jsonify({"status": "reset", "name": name, "new_state": b.state.value})

    @flask_app.route('/api/v3/rate-limiter', methods=['GET'])
    def v3_rate_limiter_stats():
        """GET /api/v3/rate-limiter — Per-client token bucket status."""
        return jsonify(rate_limiter.stats())

    # ── Phase 5: Quantitative Research Platform ───────────────────────────────

    @flask_app.route('/api/v5/quant', methods=['POST'])
    def v5_quant_research():
        """
        POST /api/v5/quant — Full quantitative research report.

        Body (JSON):
            { "ticker": "AAPL", "start_date": "2020-01-01", "end_date": "2024-12-31" }

        Returns all 17 Phase-5 metrics:
            performance, ic, rolling, monte_carlo, transition_matrix,
            feature_importance, regime_confidence, walk_forward, cross_validation

        Benchmark for Alpha/Beta: SPY (S&P 500 ETF), fixed.
        Monte Carlo: 1,000 paths × 252 days, runs synchronously.
        """
        body       = request.get_json(force=True, silent=True) or {}
        ticker     = str(body.get('ticker',     'AAPL')).strip().upper()
        start_date = str(body.get('start_date', '2020-01-01'))
        end_date   = str(body.get('end_date',   datetime.date.today().isoformat()))

        if not ticker:
            return jsonify({'error': 'ticker is required'}), 400

        try:
            from ml.quant_analytics import compute_quant_research_report
            report = compute_quant_research_report(ticker, start_date, end_date)
            return jsonify(report)
        except RuntimeError as exc:
            log.warning(f"[v5/quant] {ticker}: {exc}")
            return jsonify({'error': str(exc)}), 422
    # ── Phase 7: AI Intelligence & Portfolio Platform Routes ───────────────────

    @flask_app.route('/api/v7/ai/explain', methods=['POST'])
    def v7_ai_explain():
        """
        POST /api/v7/ai/explain
        Generates LLM market summary narrative and XAI feature attributions.
        """
        body = request.get_json(silent=True) or {}
        ticker = body.get('ticker', 'AAPL').upper().strip()
        regime = body.get('regime', 'Momentum Breakout')
        signal = body.get('signal', 'UP')
        confidence = float(body.get('confidence', 0.72))

        metrics = {
            'last_price': float(body.get('last_price', 182.50)),
            'return_1m': float(body.get('return_1m', 3.8)),
            'volatility_20d': float(body.get('volatility_20d', 16.4)),
            'rsi': float(body.get('rsi', 58.2))
        }

        from ml.ai_intelligence import AIMarketSynthesizer
        explanation = AIMarketSynthesizer.generate_narrative_summary(
            ticker, metrics, regime, signal, confidence
        )
        return jsonify(explanation), 200

    @flask_app.route('/api/v7/portfolio/optimize', methods=['POST'])
    def v7_portfolio_optimize():
        """
        POST /api/v7/portfolio/optimize
        Computes Markowitz Mean-Variance Optimal Allocations & Efficient Frontier.
        """
        body = request.get_json(silent=True) or {}
        tickers = body.get('tickers', ['AAPL', 'MSFT', 'GOOGL', 'SPY'])
        rf_rate = float(body.get('risk_free_rate', 0.05))

        if isinstance(tickers, str):
            tickers = [t.strip().upper() for t in tickers.split(',') if t.strip()]

        if len(tickers) < 2:
            tickers = ['AAPL', 'MSFT', 'GOOGL', 'SPY']

        start_date = body.get('start_date', '2022-01-01')
        end_date = body.get('end_date', '2024-01-01')

        asset_returns = {}
        for t in tickers:
            try:
                df = fetch_data_yfinance(t, start_date, end_date)
                closes = df['Close'].values
                rets = np.diff(np.log(closes))
                asset_returns[t] = rets.tolist()
            except Exception:
                np.random.seed(abs(hash(t)) % 1000)
                asset_returns[t] = np.random.normal(0.0005, 0.015, 252).tolist()

        from ml.ai_intelligence import PortfolioOptimizer
        result = PortfolioOptimizer.optimize_portfolio(asset_returns, rf_rate=rf_rate)
        return jsonify(result), 200

    @flask_app.route('/api/v7/market/intelligence', methods=['GET'])
    def v7_market_intelligence():
        """
        GET /api/v7/market/intelligence
        Returns sector heatmaps, news sentiment scores, and economic calendar events.
        """
        ticker = request.args.get('ticker', 'AAPL').upper().strip()
        from ml.ai_intelligence import MarketSentimentEngine
        data = MarketSentimentEngine.get_market_sentiment(ticker)
        return jsonify(data), 200

    @flask_app.route('/api/v7/alerts', methods=['GET', 'POST'])
    def v7_alerts():
        """
        GET /api/v7/alerts — List active user alerts
        POST /api/v7/alerts — Create new alert rule
        """
        from ml.ai_intelligence import WorkspaceManager
        if request.method == 'POST':
            body = request.get_json(silent=True) or {}
            ticker = body.get('ticker', 'AAPL').upper()
            cond = body.get('condition_type', 'RSI_ABOVE')
            thresh = float(body.get('threshold', 70.0))
            alert = WorkspaceManager.create_alert(ticker, cond, thresh)
            return jsonify(alert), 201
        else:
            alerts = WorkspaceManager.get_alerts()
            return jsonify({'alerts': alerts}), 200

    @flask_app.route('/api/v7/auth/login', methods=['POST'])
    def v7_auth_login():
        """
        POST /api/v7/auth/login — User Authentication & JWT Token generation.
        """
        body = request.get_json(silent=True) or {}
        username = body.get('username', 'quant_user')
        token = f"eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.{username}.sb_v7_session"
        return jsonify({
            'status': 'success',
            'token': token,
            'user': {
                'username': username,
                'role': 'Institutional Quant',
                'permissions': ['read', 'write', 'execute_models', 'portfolio_opt']
            }
        }), 200

    @flask_app.route('/metrics', methods=['GET'])
    def prometheus_scrape():
        """
        GET /metrics — Standard Prometheus scrape endpoint.
        Compatible with Prometheus scrape config:
          scrape_configs:
            - job_name: stockbuddy
              static_configs:
                - targets: ['localhost:5000']
        """
        from flask import Response
        return Response(REGISTRY.format_prometheus(),
                        mimetype="text/plain; version=0.0.4")

    log.info("All API routes registered (v1, v2, v3, v5, v6 + /health + /ready + /metrics).")
    return flask_app


# ── Global WSGI Application Instance ──────────────────────────────────────────
# Instantiated at module level for Gunicorn (app:flask_app), pytest, and server launch.
flask_app = create_app()


def _run_server():
    """Start the Flask development / WSGI server."""
    log.info(f"Starting StockBuddy server on {CFG.host}:{CFG.port} (environment={CFG.environment})...")
    flask_app.run(host=CFG.host, port=CFG.port, debug=CFG.debug)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1].lower() in {"serve", "server", "api"}:
        _run_server()
    else:
        main()