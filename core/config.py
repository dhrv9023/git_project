"""
core/config.py — Centralized configuration management.

Production pattern: all config comes from one place, supports environment
variable overrides, and has typed defaults. No more magic dict literals
scattered across the codebase.
"""

import os
import datetime
from dataclasses import dataclass, field


@dataclass
class AppConfig:
    # ── Data ─────────────────────────────────────────────────────────────────
    default_ticker:     str   = "AAPL"
    default_start_date: str   = "2020-01-01"
    default_end_date:   str   = field(default_factory=lambda: datetime.date.today().isoformat())

    # ── Model training ────────────────────────────────────────────────────────
    sequence_length:    int   = 90
    train_split:        float = 0.70
    val_split:          float = 0.15
    epochs:             int   = 20
    batch_size:         int   = 16
    learning_rate:      float = 1e-4
    seed:               int   = 42

    # ── Finance / backtest ────────────────────────────────────────────────────
    initial_capital:        float = 10_000.0
    risk_free_rate_annual:  float = 0.05    # 5 % T-bill proxy  (Phase 1 BUG-07)
    transaction_cost_pct:   float = 0.001   # 0.10 % round-trip (Phase 1 BUG-03)

    # ── Model persistence (Phase 2) ───────────────────────────────────────────
    model_artifacts_dir:    str   = "model_artifacts"
    registry_filename:      str   = "registry.json"
    inference_cache_dir:    str   = "inference_cache"
    inference_cache_ttl_s:  int   = 3600        # 1 hour
    model_stale_days:       int   = 7           # trigger auto-retrain after this

    # ── Background jobs (Phase 2) ─────────────────────────────────────────────
    max_worker_threads:     int   = 2
    scheduler_interval_s:   int   = 3600        # staleness check every hour

    # ── Server ────────────────────────────────────────────────────────────────
    host:   str = "0.0.0.0"
    port:   int = 5000
    debug:  bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Override any field via environment variables.

        Convention:  SB_{FIELD_NAME_UPPER} = value
        Example:     SB_EPOCHS=50 python app.py serve

        Supports str / int / float / bool fields.
        """
        inst = cls()
        for f_name, f_type in cls.__annotations__.items():
            env_key = f"SB_{f_name.upper()}"
            env_val = os.environ.get(env_key)
            if env_val is not None:
                try:
                    if f_type in (int,):
                        setattr(inst, f_name, int(env_val))
                    elif f_type in (float,):
                        setattr(inst, f_name, float(env_val))
                    elif f_type in (bool,):
                        setattr(inst, f_name, env_val.lower() in ("1", "true", "yes"))
                    else:
                        setattr(inst, f_name, env_val)
                except Exception:
                    pass   # silently keep default if parse fails
        return inst

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @property
    def registry_path(self) -> str:
        return os.path.join(self.model_artifacts_dir, self.registry_filename)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.model_artifacts_dir, self.inference_cache_dir)


# Global singleton — import this everywhere instead of the old CONFIG dict
CFG = AppConfig.from_env()
