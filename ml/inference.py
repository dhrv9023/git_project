"""
ml/inference.py — Cached Inference Engine.

Production pattern: prediction is separated from training. When a user
requests a prediction, we:

  1. Check memory cache (sub-millisecond)
  2. Check disk cache (milliseconds)
  3. Load saved model from disk + run inference (seconds)
  4. Fall back to live training only if no saved model exists

This mirrors how Netflix's Merlin or Uber's Michelangelo serve predictions:
models are loaded once into memory and predictions are cached aggressively.

Cache strategy:
  Layer 1 — In-Memory LRU (fastest): keyed by {ticker}_{version}_{window_hash}
  Layer 2 — Disk pickle (persistent): keyed by same key, TTL = 1 hour
  Layer 3 — Live inference (fallback): loads model from disk, no training

Cache invalidation:
  - TTL-based: predictions expire after cfg.inference_cache_ttl_s seconds
  - Explicit: DELETE /api/v2/cache flushes both layers
  - Version-based: new model version auto-expires old predictions
"""

import os
import time
import pickle
import hashlib
import logging
import threading
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ml.registry import ModelRegistry
    from storage.model_store import ModelStore
    from core.config import AppConfig

log = logging.getLogger(__name__)


class InferenceCache:
    """
    Two-level cache: in-memory dict + disk pickle files.

    Thread-safe via RLock.
    """

    def __init__(self, cache_dir: str, ttl_seconds: int = 3600):
        self.cache_dir   = cache_dir
        self.ttl         = ttl_seconds
        self._lock       = threading.RLock()
        self._mem: Dict[str, Tuple[Any, float]] = {}   # key → (value, timestamp)
        os.makedirs(cache_dir, exist_ok=True)

    def _disk_path(self, key: str) -> str:
        # Use a hash so long keys don't exceed filesystem limits
        safe = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{safe}.pkl")

    def get(self, key: str) -> Optional[Any]:
        now = time.time()

        # L1: memory
        with self._lock:
            if key in self._mem:
                val, ts = self._mem[key]
                if now - ts < self.ttl:
                    log.debug(f"Cache L1 hit: {key}")
                    return val
                else:
                    del self._mem[key]

        # L2: disk
        path = self._disk_path(key)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    stored = pickle.load(f)
                ts  = stored.get("ts", 0)
                val = stored.get("val")
                if now - ts < self.ttl:
                    log.debug(f"Cache L2 hit: {key}")
                    with self._lock:
                        self._mem[key] = (val, ts)   # promote to L1
                    return val
                else:
                    os.remove(path)   # expired — delete
            except Exception as e:
                log.warning(f"Disk cache read error ({key}): {e}")

        return None

    def set(self, key: str, value: Any):
        now = time.time()
        with self._lock:
            self._mem[key] = (value, now)

        path = self._disk_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump({"val": value, "ts": now}, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            log.warning(f"Disk cache write error ({key}): {e}")

    def flush(self):
        """Evict everything from both layers."""
        with self._lock:
            self._mem.clear()
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".pkl"):
                try:
                    os.remove(os.path.join(self.cache_dir, fname))
                except OSError:
                    pass
        log.info("Inference cache flushed")

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            mem_size = len(self._mem)
        disk_files = len([f for f in os.listdir(self.cache_dir) if f.endswith(".pkl")])
        return {
            "memory_entries": mem_size,
            "disk_entries":   disk_files,
            "ttl_seconds":    self.ttl,
        }


class InferenceEngine:
    """
    Loads trained models from disk and runs cached inference.

    Lifecycle per request:
      1. Look up best/latest model version in registry
      2. Check cache for {ticker}_{version}_{window_id}
      3. Cache miss → load artifacts from ModelStore
      4. Run model.predict() on the last sequence window
      5. Store result in cache; return to caller
    """

    def __init__(self, registry: Any, store: Any, cache: InferenceCache, cfg: Any):
        self.registry = registry
        self.store    = store
        self.cache    = cache
        self.cfg      = cfg
        # Warm artifact cache: maps version_dir → loaded artifacts
        self._artifact_cache: Dict[str, Any] = {}
        self._artifact_lock  = threading.Lock()

    def _load_artifacts(self, ticker: str, version: str) -> Optional[Dict[str, Any]]:
        """Load model artifacts, using in-process warm cache to avoid repeated disk I/O."""
        key = f"{ticker.upper()}_{version}"
        with self._artifact_lock:
            if key in self._artifact_cache:
                return self._artifact_cache[key]
        arts = self.store.load_artifacts(ticker, version)
        if arts:
            with self._artifact_lock:
                self._artifact_cache[key] = arts
        return arts

    def evict_artifacts(self, ticker: Optional[str] = None):
        """Remove loaded artifacts from warm cache (call after retraining)."""
        with self._artifact_lock:
            if ticker:
                keys = [k for k in self._artifact_cache if k.startswith(ticker.upper())]
                for k in keys:
                    del self._artifact_cache[k]
            else:
                self._artifact_cache.clear()

    def predict(self, ticker: str,
                start_date: str,
                end_date: str,
                version: str = "best",
                force_refresh: bool = False) -> Dict[str, Any]:
        """
        Run full inference pipeline for a ticker.

        Args:
            ticker:        Stock symbol (e.g. "AAPL")
            start_date:    Historical data start for feature computation
            end_date:      Historical data end
            version:       "best", "latest", or specific version string e.g. "v3"
            force_refresh: Skip cache lookup and recompute

        Returns:
            dict with keys: predictions, metrics, confidence, dates,
                            version_used, from_cache, model_path
        """
        import math
        import numpy as np

        ticker = ticker.upper()

        # Resolve version string → actual version record
        if version == "best":
            rec = self.registry.get_best(ticker)
        elif version == "latest":
            rec = self.registry.get_latest(ticker)
        else:
            rec = self.registry.get_record(ticker, version)

        if rec is None or rec.get("status") != "ready":
            return {
                "error": f"No ready model found for {ticker} (requested version='{version}'). "
                         f"Submit a training job via POST /api/v2/train first.",
                "ticker": ticker,
            }

        resolved_version = rec["version"]
        cache_key = f"pred_{ticker}_{resolved_version}_{start_date}_{end_date}"

        # ── Cache lookup ──────────────────────────────────────────────────────
        if not force_refresh:
            cached = self.cache.get(cache_key)
            if cached is not None:
                cached["from_cache"] = True
                cached["cache_key"]  = cache_key
                return cached

        # ── Load artifacts ────────────────────────────────────────────────────
        arts = self._load_artifacts(ticker, resolved_version)
        if arts is None:
            return {"error": f"Artifacts for {ticker}/{resolved_version} not found on disk."}

        models_dict = arts["models"]
        scaler_X    = arts["scaler_X"]
        scaler_y    = arts["scaler_y"]
        metadata    = arts["metadata"] or {}
        seq_len     = metadata.get("seq_len", self.cfg.sequence_length)
        close_idx   = metadata.get("close_feature_index", 0)

        # ── Run data pipeline & evaluation via modular architecture ───────────
        from ml.features import split_and_scale_data, scale_single_feature
        from app.repositories.market_data_repo import MarketDataRepository
        from app.services.backtest_service import BacktestService

        market_repo = MarketDataRepository()
        data = market_repo.build_feature_matrix(ticker, start_date, end_date, seq_len)
        splits = split_and_scale_data(
            data["X_raw"], data["y_raw"], data["dates_raw"], data["base_prices_raw"],
            self.cfg.train_split, self.cfg.val_split, seq_len,
        )

        # ── Evaluate on test partition ────────────────────────────────────────
        X_test, y_test, dates_test, base_test = splits["test"]
        preds = {}
        metrics_all = {}
        bt_svc = BacktestService(self.cfg)

        logret_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_true = base_test * np.exp(logret_true)
        for name, mdl in models_dict.items():
            if name.endswith("_history"):
                continue
            y_pred_scaled = mdl.predict(X_test, verbose=0).ravel()
            logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            price_pred = base_test * np.exp(logret_pred)
            preds[name] = price_pred
            metrics_all[name] = bt_svc.calculate_metrics(y_true, price_pred)

        model_names = [k for k in models_dict.keys() if not k.endswith("_history")]
        y_stack = np.stack([preds[n] for n in model_names], axis=1)
        weights = np.ones(len(model_names), dtype=np.float32) / float(len(model_names))
        y_ens = (y_stack * weights).sum(axis=1)
        preds["Ensemble"] = y_ens
        metrics_all["Ensemble"] = bt_svc.calculate_metrics(y_true, y_ens)

        std = y_stack.std(axis=1) if len(model_names) > 1 else np.zeros_like(y_ens)
        lower = y_ens - 1.96 * std
        upper = y_ens + 1.96 * std

        eval_results = {
            "predictions": preds,
            "metrics": metrics_all,
            "y_test_actual": y_true,
            "dates_test": dates_test,
            "confidence_intervals": (lower, upper),
        }

        # Backtest
        bt = bt_svc.run_signal_backtest(
            eval_results["predictions"]["Ensemble"],
            eval_results["y_test_actual"],
            self.cfg.initial_capital,
        )

        # ── 5-day autoregressive forecast ─────────────────────────────────────
        X_te_seq  = splits["test"][0]
        last_seq  = X_te_seq[-1] if len(X_te_seq) > 0 else splits["train"][0][-1]
        last_price = float(eval_results["y_test_actual"][-1])
        future_days = 5

        seq = last_seq.copy().astype(np.float32)
        price = last_price
        future_preds = []
        for _ in range(future_days):
            lrs = []
            for name, mdl in models_dict.items():
                if name.endswith("_history"): continue
                yhat_s = mdl.predict(seq[np.newaxis, ...], verbose=0).ravel()[0]
                logret = scaler_y.inverse_transform([[yhat_s]])[0, 0]
                lrs.append(logret)
            avg_lr = float(np.mean(lrs))
            price  = price * float(np.exp(avg_lr))
            future_preds.append(round(price, 4))
            seq = np.roll(seq, -1, axis=0)
            seq[-1, close_idx] = scale_single_feature(price, scaler_X, close_idx)

        import pandas as pd
        last_date    = eval_results["dates_test"][-1]
        future_dates = [(last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
                        for i in range(future_days)]

        # ── Serialize ─────────────────────────────────────────────────────────
        def _safe(v):
            if v is None: return None
            if isinstance(v, (float,)) and (math.isnan(v) or math.isinf(v)): return None
            return round(float(v), 6) if isinstance(v, float) else v
        def _sl(a): return [_safe(x) for x in a]

        result = {
            "ticker":        ticker,
            "version_used":  resolved_version,
            "model_path":    rec.get("model_path"),
            "from_cache":    False,
            "cache_key":     cache_key,
            "dates":         [d.strftime("%Y-%m-%d") for d in eval_results["dates_test"]],
            "actual":        _sl(eval_results["y_test_actual"]),
            "predictions":   {k: _sl(v) for k, v in eval_results["predictions"].items()},
            "confidence":    {
                "lower": _sl(eval_results["confidence_intervals"][0]),
                "upper": _sl(eval_results["confidence_intervals"][1]),
            },
            "metrics":       {k: {mk: _safe(mv) for mk, mv in m.items()}
                              for k, m in eval_results["metrics"].items()},
            "backtest": {
                "equity":       _sl(bt["equity"]),
                "buy_signals":  _sl(bt["buy_signals"]),
                "sell_signals": _sl(bt["sell_signals"]),
                "metrics":      {mk: _safe(mv) for mk, mv in bt["metrics"].items()},
            },
            "future": {
                "dates":       future_dates,
                "predictions": future_preds,
            },
        }

        # Store in cache
        self.cache.set(cache_key, result)
        return result
