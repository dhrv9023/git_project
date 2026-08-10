"""
app/api/health_routes.py — Liveness and readiness probes.

Engineering decision: health endpoints are a Flask Blueprint so they
can be registered independently and tested without any service dependencies.
"""
from __future__ import annotations

import datetime
import os
import time

from flask import Blueprint, jsonify

from core.circuit_breaker import get_breaker

health_bp = Blueprint("health", __name__)

_START_TIME = time.time()


@health_bp.route("/health", methods=["GET"])
def liveness():
    """GET /health — Container liveness probe (K8s / Docker / Render)."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "uptime_seconds": round(time.time() - _START_TIME, 2),
        "version": "2.0.0",
    }), 200


@health_bp.route("/ready", methods=["GET"])
def readiness():
    """GET /ready — Container readiness probe."""
    from core.config import AppConfig
    cfg = AppConfig.from_env()
    checks: dict = {}
    is_ready = True

    try:
        test_path = os.path.join(cfg.model_artifacts_dir, ".health_check_tmp")
        os.makedirs(cfg.model_artifacts_dir, exist_ok=True)
        with open(test_path, "w") as f:
            f.write("ok")
        os.remove(test_path)
        checks["storage_writable"] = True
    except Exception as exc:
        checks["storage_writable"] = False
        checks["storage_error"] = str(exc)
        is_ready = False

    try:
        yf_b = get_breaker("yfinance")
        checks["circuit_breaker_state"] = yf_b.state.value
    except Exception:
        checks["circuit_breaker_state"] = "unknown"

    status_code = 200 if is_ready else 503
    return jsonify({
        "status": "ready" if is_ready else "unready",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checks": checks,
    }), status_code


@health_bp.route("/metrics", methods=["GET"])
def prometheus_scrape():
    """GET /metrics — Prometheus scrape endpoint."""
    from flask import Response
    from core.metrics import REGISTRY
    return Response(REGISTRY.format_prometheus(), mimetype="text/plain; version=0.0.4")
