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

    # ── Phase 3: Distributed Systems ─────────────────────────────────────────
    # Parallel data fetch: number of concurrent yfinance download threads
    fetch_parallelism:          int   = 4

    # Circuit breaker: yfinance API
    cb_failure_threshold:       float = 0.5     # open if 50% of window fails
    cb_window_size:             int   = 20      # sliding window size
    cb_reset_timeout_s:         float = 60.0    # seconds before half-open probe

    # Rate limiter: per-IP burst + sustained
    rate_limit_burst:           float = 20.0    # token bucket capacity
    rate_limit_rate:            float = 5.0     # tokens per second refill
    rate_limit_window_max:      int   = 30      # max requests per window
    rate_limit_window_s:        float = 60.0    # window duration (seconds)

    # Batch predictor
    batch_predictor_max_batch:  int   = 32      # sequences per forward pass
    batch_predictor_max_wait_ms:float = 50.0    # ms to wait before flushing

    # Retry policy for training jobs
    job_max_retries:            int   = 3
    job_retry_base_delay_s:     float = 5.0
    job_retry_max_delay_s:      float = 300.0

    # Priority job queue
    use_priority_queue:         bool  = True    # Phase 3 queue vs Phase 2 pool

    # Model store: parallel I/O workers
    store_io_workers:           int   = 3

    # Inference cache: max in-memory entries (LRU eviction)
    cache_max_memory_entries:   int   = 100

    # ── Server & Production (Phase 6) ──────────────────────────────────────────
    host:   str = "0.0.0.0"
    port:   int = 5000
    debug:  bool = False

    environment:              str  = "development"  # development | staging | production
    sentry_dsn:               str  = ""
    log_format:               str  = "text"         # text | json
    security_headers_enabled: bool = True

    @classmethod
    def from_env(cls) -> "AppConfig":
        """
        Override any field via environment variables.

        Convention:  SB_{FIELD_NAME_UPPER} = value (or standard env vars like PORT, HOST, SENTRY_DSN, ENVIRONMENT)
        Example:     SB_EPOCHS=50 python app.py serve

        Supports str / int / float / bool fields.
        Automatically loads .env file if available.
        """
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        inst = cls()
        for f_name, f_type in cls.__annotations__.items():
            # Check primary prefix SB_{FIELD} first, then fallback to standard env names
            env_key = f"SB_{f_name.upper()}"
            env_val = os.environ.get(env_key)

            if env_val is None:
                # Direct fallback mapping for standard cloud platform env vars
                std_keys = {
                    "port": "PORT",
                    "host": "HOST",
                    "debug": "DEBUG",
                    "environment": "ENVIRONMENT",
                    "sentry_dsn": "SENTRY_DSN",
                    "log_format": "LOG_FORMAT",
                }
                if f_name in std_keys:
                    env_val = os.environ.get(std_keys[f_name])

            if env_val is not None:
                try:
                    if f_type in (int,):
                        setattr(inst, f_name, int(env_val))
                    elif f_type in (float,):
                        setattr(inst, f_name, float(env_val))
                    elif f_type in (bool,):
                        setattr(inst, f_name, env_val.lower() in ("1", "true", "yes", "on"))
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
