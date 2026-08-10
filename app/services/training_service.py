"""
app/services/training_service.py — Training job orchestration.

Engineering decisions:
  - Wraps BackgroundTrainer and ModelRegistry, exposes a clean interface.
  - ELIMINATES sys.modules coupling: trainer.py now imports from ml.features
    and ml.models directly (no app.py function references).
  - submit_job() returns job_id immediately; caller polls get_job().
"""
from __future__ import annotations

import logging
from typing import Optional

from ml.registry import ModelRegistry
from ml.trainer import BackgroundTrainer
from core.config import AppConfig

log = logging.getLogger(__name__)


class TrainingService:
    """Facade over BackgroundTrainer and ModelRegistry.

    Engineering decision: this is a thin facade — it doesn't add logic,
    it provides a stable interface. If BackgroundTrainer is replaced with
    a Celery-backed implementation, only this class changes.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        trainer: BackgroundTrainer,
        cfg: AppConfig,
    ) -> None:
        self.registry = registry
        self.trainer = trainer
        self.cfg = cfg

    def submit_job(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        seq_len: Optional[int] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        models: Optional[list] = None,
    ) -> str:
        """Enqueue a training job. Returns job_id immediately (non-blocking)."""
        return self.trainer.submit(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            seq_len=seq_len or self.cfg.sequence_length,
            epochs=epochs or self.cfg.epochs,
            batch_size=batch_size or self.cfg.batch_size,
            models=models or ["LSTM", "GRU", "Transformer"],
        )

    def get_job(self, job_id: str):
        return self.trainer.get_job(job_id)

    def list_jobs(self) -> list:
        return self.trainer.list_jobs()

    def active_count(self) -> int:
        return self.trainer.active_count()

    def registry_stats(self) -> dict:
        return self.registry.stats()

    def full_registry(self) -> dict:
        return self.registry.full_registry()

    def list_versions(self, ticker: str) -> list:
        return self.registry.list_versions(ticker.upper())

    def get_best(self, ticker: str):
        return self.registry.get_best(ticker.upper())

    def get_latest(self, ticker: str):
        return self.registry.get_latest(ticker.upper())
