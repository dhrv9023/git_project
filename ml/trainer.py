"""
ml/trainer.py — Background Training Engine with Job Queue.

Production pattern: training is decoupled from the HTTP request lifecycle.
When a user hits POST /api/v2/train, they receive a job_id immediately.
The actual training happens in a background thread. The client polls
GET /api/v2/jobs/{job_id} for status.

This matches the pattern used by:
  - Google Vertex AI Training Jobs
  - AWS SageMaker Training Jobs
  - Databricks MLflow Projects

Architecture:
  ┌─────────┐   POST /train    ┌──────────────┐   enqueue   ┌──────────────┐
  │  Flask  │ ─────────────── ▶│BackgroundTrainer│ ─────────▶│  ThreadPool  │
  │ Handler │ ◀─── job_id ──── │              │             │  (workers)   │
  └─────────┘                  └──────────────┘             └──────────────┘
       │                              │                             │
  GET /jobs/{id}             JOB_REGISTRY dict              trains + saves
       │◀──── status/result ──────────┘◀───────── done/failed ─────┘
"""

import uuid
import datetime
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """
    Immutable-ish record of one async training request.

    Status lifecycle: queued → running → done
                                       └→ failed
    """
    job_id:       str
    ticker:       str
    start_date:   str
    end_date:     str
    seq_len:      int
    epochs:       int
    batch_size:   int
    models:       list
    status:       str = "queued"         # queued | running | done | failed
    created_at:   str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    started_at:   Optional[str] = None
    completed_at: Optional[str] = None
    version:      Optional[str] = None   # registry version assigned on completion
    result:       Optional[Dict] = None  # summary metrics on success
    error:        Optional[str] = None   # error message on failure

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id":       self.job_id,
            "ticker":       self.ticker,
            "start_date":   self.start_date,
            "end_date":     self.end_date,
            "seq_len":      self.seq_len,
            "epochs":       self.epochs,
            "batch_size":   self.batch_size,
            "models":       self.models,
            "status":       self.status,
            "created_at":   self.created_at,
            "started_at":   self.started_at,
            "completed_at": self.completed_at,
            "version":      self.version,
            "result":       self.result,
            "error":        self.error,
        }


class BackgroundTrainer:
    """
    Manages a pool of worker threads for async model training.

    Jobs are tracked in an in-memory dict (JOB_REGISTRY) and
    persisted to the ModelRegistry + ModelStore on completion.

    Thread safety: job_registry access is protected by _lock.
    """

    def __init__(self, registry, store, cfg, max_workers: int = 2):
        """
        Args:
            registry: ModelRegistry instance
            store:    ModelStore instance
            cfg:      AppConfig instance
            max_workers: max concurrent training threads
        """
        self.registry    = registry
        self.store       = store
        self.cfg         = cfg
        self._pool       = ThreadPoolExecutor(max_workers=max_workers,
                                              thread_name_prefix="sb_trainer")
        self._lock       = threading.Lock()
        self._jobs: Dict[str, TrainingJob] = {}
        log.info(f"BackgroundTrainer ready (workers={max_workers})")

    # ── Submit ────────────────────────────────────────────────────────────────

    def submit(self, ticker: str,
               start_date: str,
               end_date: str,
               seq_len: int   = None,
               epochs: int    = None,
               batch_size: int = None,
               models: list   = None) -> str:
        """
        Enqueue a training job. Returns job_id immediately (non-blocking).
        """
        cfg = self.cfg
        job = TrainingJob(
            job_id     = str(uuid.uuid4()),
            ticker     = ticker.upper(),
            start_date = start_date,
            end_date   = end_date,
            seq_len    = seq_len    or cfg.sequence_length,
            epochs     = epochs     or cfg.epochs,
            batch_size = batch_size or cfg.batch_size,
            models     = models     or ["LSTM", "GRU", "Transformer"],
        )
        with self._lock:
            self._jobs[job.job_id] = job

        # Register in model registry immediately (status=queued)
        version = self.registry.register(job.ticker, {
            "start_date":      job.start_date,
            "end_date":        job.end_date,
            "seq_len":         job.seq_len,
            "epochs":          job.epochs,
            "models":          job.models,
            "job_id":          job.job_id,
            "stale_after_days": cfg.model_stale_days,
        }, status="queued")
        job.version = version

        # Submit to thread pool
        self._pool.submit(self._run_job, job)
        log.info(f"Submitted job {job.job_id} for {job.ticker} version={version}")
        return job.job_id

    # ── Worker ────────────────────────────────────────────────────────────────

    def _run_job(self, job: TrainingJob):
        """
        Actual training logic — runs inside a worker thread.

        Engineering decision: direct imports from ml.features and ml.models
        replace the previous sys.modules hack. This makes the trainer
        independently importable and fully testable without a running Flask app.
        """
        # Mark running
        with self._lock:
            job.status = "running"
            job.started_at = datetime.datetime.utcnow().isoformat()
        self.registry.update_status(job.ticker, job.version, "training")
        log.info(f"[{job.job_id}] Training started: {job.ticker}/{job.version}")

        try:
            # Direct imports — no sys.modules coupling
            from app.repositories.market_data_repo import MarketDataRepository
            from ml.features import split_and_scale_data
            from ml.models import build_model, make_callbacks
            from app.services.backtest_service import BacktestService
            import numpy as np

            cfg = self.cfg
            market_repo = MarketDataRepository()

            # ── Data pipeline ──────────────────────────────────────────────
            data = market_repo.build_feature_matrix(
                job.ticker, job.start_date, job.end_date, job.seq_len
            )
            splits = split_and_scale_data(
                data["X_raw"], data["y_raw"], data["dates_raw"], data["base_prices_raw"],
                cfg.train_split, cfg.val_split, job.seq_len,
            )
            input_shape = (job.seq_len, data["X_raw"].shape[1])

            # ── Train ──────────────────────────────────────────────────────
            selected = [m for m in job.models if m in {"LSTM", "GRU", "Transformer"}]
            if not selected:
                selected = ["LSTM", "GRU", "Transformer"]

            X_train, y_train, _, _ = splits["train"]
            X_val, y_val, _, _ = splits["val"]
            steps_per_epoch = max(1, len(X_train) // job.batch_size)

            models_dict = {}
            for model_type in selected:
                log.info(f"[{job.job_id}] Training {model_type}...")
                model = build_model(model_type, input_shape, cfg.learning_rate)
                cbs = make_callbacks(job.epochs, steps_per_epoch, cfg.learning_rate)
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=job.epochs,
                    batch_size=job.batch_size,
                    verbose=0,
                    callbacks=cbs,
                )
                models_dict[model_type] = model
                models_dict[f"{model_type}_history"] = history.history

            # ── Evaluate (ensemble) ────────────────────────────────────────
            scaler_y = splits["scaler_y"]
            X_test, y_test, dates_test, base_test = splits["test"]
            model_names = [k for k in models_dict if not k.endswith("_history")]
            preds = {}
            bt_svc = BacktestService(cfg)
            for name in model_names:
                y_pred_scaled = models_dict[name].predict(X_test, verbose=0).ravel()
                logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                logret_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
                price_pred = base_test * np.exp(logret_pred)
                preds[name] = price_pred

            y_true_prices = base_test * np.exp(logret_true)
            y_stack = np.stack([preds[n] for n in model_names], axis=1)
            y_ens = y_stack.mean(axis=1)
            preds["Ensemble"] = y_ens

            metrics = {
                name: bt_svc.calculate_metrics(y_true_prices, preds[name])
                for name in list(model_names) + ["Ensemble"]
            }
            eval_results = {"metrics": metrics}

            # Build clean metrics dict (JSON-serializable)
            import math
            def _clean(v):
                if v is None: return None
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
                return round(float(v), 6)

            metrics = {
                model_name: {k: _clean(v) for k, v in m.items()}
                for model_name, m in eval_results["metrics"].items()
            }

            # ── Persist to disk ────────────────────────────────────────────
            metadata = {
                "job_id":      job.job_id,
                "ticker":      job.ticker,
                "version":     job.version,
                "start_date":  job.start_date,
                "end_date":    job.end_date,
                "seq_len":     job.seq_len,
                "epochs":      job.epochs,
                "batch_size":  job.batch_size,
                "train_split": cfg.train_split,
                "val_split":   cfg.val_split,
                "metrics":     metrics,
                "feature_cols": data.get("feature_cols", []),
                "close_feature_index": data.get("close_feature_index", 0),
            }
            model_path = self.store.save_artifacts(
                job.ticker, job.version,
                models_dict, splits["scaler_X"], splits["scaler_y"],
                metadata,
            )

            # ── Update registry ────────────────────────────────────────────
            self.registry.update_status(
                job.ticker, job.version, "ready",
                metrics=metrics, model_path=model_path,
            )

            # ── Finalize job record ────────────────────────────────────────
            with self._lock:
                job.status       = "done"
                job.completed_at = datetime.datetime.utcnow().isoformat()
                job.result       = {
                    "version":    job.version,
                    "model_path": model_path,
                    "metrics":    metrics,
                }
            log.info(f"[{job.job_id}] Training DONE: {job.ticker}/{job.version}")

        except Exception as exc:
            log.exception(f"[{job.job_id}] Training FAILED: {exc}")
            self.registry.update_status(job.ticker, job.version, "failed")
            with self._lock:
                job.status       = "failed"
                job.completed_at = datetime.datetime.utcnow().isoformat()
                job.error        = str(exc)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list:
        with self._lock:
            return [j.to_dict() for j in sorted(
                self._jobs.values(),
                key=lambda j: j.created_at, reverse=True
            )]

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status == "running")

    def shutdown(self):
        """Graceful shutdown — wait for running jobs to finish."""
        log.info("BackgroundTrainer shutting down...")
        self._pool.shutdown(wait=True)
