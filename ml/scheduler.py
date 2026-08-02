"""
ml/scheduler.py — Automatic Retraining Scheduler.

Production pattern: models degrade over time as market conditions shift.
A background daemon thread periodically checks the model registry for
stale models and automatically submits retraining jobs.

This matches:
  - Google Cloud Vertex AI's scheduled retraining pipelines
  - AWS SageMaker Model Monitor + automatic retraining triggers
  - Uber Michelangelo's staleness-based refresh policy

Staleness policy:
  A model is considered STALE if:
    trained_at < (now - cfg.model_stale_days)

  When staleness is detected:
    1. Mark model as 'stale' in registry
    2. Evict model from inference artifact cache
    3. Submit a new training job via BackgroundTrainer

The scheduler runs as a daemon thread — it exits automatically when
the main Flask process exits (no cleanup needed).
"""

import time
import threading
import datetime
import logging
from typing import Optional

log = logging.getLogger(__name__)


class RetrainingScheduler:
    """
    Daemon thread that polls the registry and triggers retraining for stale models.
    """

    def __init__(self, registry, trainer, inference_engine, cfg):
        self.registry         = registry
        self.trainer          = trainer
        self.inference_engine = inference_engine
        self.cfg              = cfg
        self._stop_event      = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start the scheduler daemon thread."""
        if self._thread and self._thread.is_alive():
            log.warning("Scheduler already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="sb_scheduler",
            daemon=True,   # dies with the main process — no explicit cleanup needed
        )
        self._thread.start()
        log.info(f"RetrainingScheduler started (interval={self.cfg.scheduler_interval_s}s, "
                 f"stale_after={self.cfg.model_stale_days}d)")

    def stop(self):
        """Signal the scheduler to stop at next iteration."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        """Main scheduler loop — runs every cfg.scheduler_interval_s seconds."""
        while not self._stop_event.is_set():
            try:
                self._check_and_retrain()
            except Exception as e:
                log.error(f"Scheduler error: {e}", exc_info=True)

            # Sleep in small chunks so stop() is responsive
            for _ in range(self.cfg.scheduler_interval_s):
                if self._stop_event.is_set():
                    return
                time.sleep(1)

    def _check_and_retrain(self):
        """
        Check all READY models for staleness, trigger retraining as needed.
        """
        stale_models = self.registry.get_stale_ready_models(
            stale_after_days=self.cfg.model_stale_days
        )

        if not stale_models:
            log.debug("Scheduler: no stale models found")
            return

        log.info(f"Scheduler: found {len(stale_models)} stale model(s)")

        for rec in stale_models:
            ticker  = rec["ticker"]
            version = rec["version"]
            log.info(f"Scheduler: retraining {ticker}/{version} (trained_at={rec.get('trained_at')})")

            # Mark old version as stale in registry
            self.registry.mark_stale(ticker, version)

            # Evict from inference warm-cache so next request loads the new model
            self.inference_engine.evict_artifacts(ticker)

            # Submit new training job (uses same date config as the stale model)
            try:
                job_id = self.trainer.submit(
                    ticker     = ticker,
                    start_date = rec.get("start_date", self.cfg.default_start_date),
                    end_date   = datetime.date.today().isoformat(),  # always extend to today
                    seq_len    = rec.get("seq_len", self.cfg.sequence_length),
                    epochs     = rec.get("epochs",  self.cfg.epochs),
                    batch_size = self.cfg.batch_size,
                    models     = rec.get("models_trained", ["GRU"]),  # retrain best arch
                )
                log.info(f"Scheduler: submitted retraining job {job_id} for {ticker}")
            except Exception as e:
                log.error(f"Scheduler: failed to submit job for {ticker}: {e}")

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        return {
            "running":          self.is_running,
            "interval_seconds": self.cfg.scheduler_interval_s,
            "stale_after_days": self.cfg.model_stale_days,
        }
