"""
app/__init__.py — Application factory (Composition Root).

Engineering decisions:
  - create_app() is the single composition root — the ONLY place where
    concrete implementations are wired to interfaces.
  - All dependencies flow inward (infrastructure → repositories → services → routes).
  - Accepts an optional AppConfig so tests can inject a test config
    (e.g., with tiny epoch counts, temp directories) without environment variables.
  - Flask app is returned (not instantiated at module level) so it can be
    created multiple times safely in tests.

Usage:
  Production (Gunicorn):  app:flask_app
  Tests:                  from app import create_app; app = create_app(test_cfg)
"""
from __future__ import annotations

import logging
import os
import time
import datetime

from flask import Flask, jsonify
from flask_cors import CORS

from core.config import AppConfig
from core.circuit_breaker import get_breaker
from core.rate_limiter import RateLimiter
from ml.registry import ModelRegistry
from ml.trainer import BackgroundTrainer
from ml.inference import InferenceCache, InferenceEngine
from ml.scheduler import RetrainingScheduler
from storage.model_store import ModelStore
from storage.workspace_store import WorkspaceStore

from app.auth.auth_service import AuthService
from app.repositories.market_data_repo import MarketDataRepository
from app.services.backtest_service import BacktestService
from app.services.comparison_service import ComparisonService
from app.services.regime_service import RegimeService
from app.services.training_service import TrainingService
from app.middleware.error_handlers import register_error_handlers
from app.middleware.observability import register_observability

log = logging.getLogger(__name__)

_SERVER_START_TIME = time.time()


def create_app(cfg: AppConfig | None = None) -> Flask:
    """Create and configure the Flask application.

    This is the Composition Root: all concrete implementations are
    instantiated and wired here. No service or route imports anything
    from this module (dependency inversion).
    """
    cfg = cfg or AppConfig.from_env()

    # ── Infrastructure layer ──────────────────────────────────────────────
    store = ModelStore(base_dir=cfg.model_artifacts_dir)
    registry = ModelRegistry(registry_path=cfg.registry_path)
    cache = InferenceCache(cache_dir=cfg.cache_dir, ttl_seconds=cfg.inference_cache_ttl_s)
    raw_trainer = BackgroundTrainer(registry, store, cfg, max_workers=cfg.max_worker_threads)
    engine = InferenceEngine(registry, store, cache, cfg)
    scheduler = RetrainingScheduler(registry, raw_trainer, engine, cfg)
    scheduler.start()

    rate_limiter = RateLimiter(
        burst_capacity=cfg.rate_limit_burst,
        burst_rate=cfg.rate_limit_rate,
        window_max=cfg.rate_limit_window_max,
        window_s=cfg.rate_limit_window_s,
    )
    get_breaker(
        "yfinance",
        failure_threshold=cfg.cb_failure_threshold,
        window_size=cfg.cb_window_size,
        reset_timeout_s=cfg.cb_reset_timeout_s,
    )

    # ── Repository layer ──────────────────────────────────────────────────
    market_repo = MarketDataRepository()

    # ── Service layer ─────────────────────────────────────────────────────
    regime_svc = RegimeService(market_repo, cfg)
    backtest_svc = BacktestService(cfg)
    training_svc = TrainingService(registry, raw_trainer, cfg)
    comparison_svc = ComparisonService(market_repo, cfg)

    # ── Persistent user data store ───────────────────────────────────
    workspace_store = WorkspaceStore(
        base_dir=os.path.join(cfg.model_artifacts_dir, "user_data")
    )

    # ── Flask application ─────────────────────────────────────────────────────
    flask_app = Flask(__name__)
    CORS(flask_app)

    @flask_app.route("/")
    def serve_index():
        from flask import send_from_directory
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return send_from_directory(root_dir, "index.html")

    # Auth (JWT) — must be initialised before any JWT-protected routes
    from app.api.auth_routes import init_jwt
    init_jwt(flask_app)
    auth_svc = AuthService(
        users_file=os.path.join(cfg.model_artifacts_dir, "user_data", "users.json")
    )

    # Middleware
    register_error_handlers(flask_app)
    register_observability(flask_app, rate_limiter)

    # ── Register blueprints ───────────────────────────────────────────────
    from app.api.health_routes import health_bp
    from app.api.v1_routes import make_v1_blueprint
    from app.api.v2_routes import make_v2_blueprint
    from app.api.v3_routes import make_v3_blueprint
    from app.api.workspace_routes import make_workspace_blueprint
    from app.api.auth_routes import make_auth_blueprint

    flask_app.register_blueprint(health_bp)
    flask_app.register_blueprint(make_auth_blueprint(auth_svc))
    flask_app.register_blueprint(
        make_v1_blueprint(market_repo, regime_svc, backtest_svc, engine, cfg)
    )
    flask_app.register_blueprint(
        make_v2_blueprint(training_svc, engine, cache, store, scheduler, comparison_svc, cfg)
    )
    flask_app.register_blueprint(
        make_v3_blueprint(training_svc, engine, cache, rate_limiter, cfg)
    )
    flask_app.register_blueprint(
        make_workspace_blueprint(workspace_store, cfg)
    )

    # v5 and v7 blueprints (thin wrappers — logic in ml/)
    try:
        from app.api.v5_routes import make_v5_blueprint
        flask_app.register_blueprint(make_v5_blueprint())
    except ImportError:
        log.warning("v5 blueprint not available")

    try:
        from app.api.v7_routes import make_v7_blueprint
        flask_app.register_blueprint(make_v7_blueprint(market_repo, cfg))
    except ImportError:
        log.warning("v7 blueprint not available")

    log.info(
        "StockBuddy app created | env=%s | registry=%s",
        cfg.environment, cfg.registry_path,
    )
    return flask_app


# ── Module-level instance for Gunicorn and pytest ─────────────────────────────
flask_app = create_app()
