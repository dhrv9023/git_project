"""
app/middleware/observability.py — before/after request metrics hooks.

Engineering decision: moved out of create_app() so the middleware logic
is independently readable and testable.
"""
from __future__ import annotations

import time

import flask
from flask import Flask, request

from core.metrics import (
    http_latency_seconds,
    http_requests_total,
)


def register_observability(app: Flask, rate_limiter) -> None:
    """Attach request timing and rate-limiting hooks."""

    @app.before_request
    def _before():
        flask.g.t_start = time.perf_counter()
        flask.g.endpoint = request.path
        if request.method in ("POST", "DELETE"):
            client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
            if not rate_limiter.allow(client_ip):
                http_requests_total.inc(method=request.method, endpoint=request.path, status="429")
                from flask import jsonify
                return jsonify({"error": "Rate limit exceeded", "retry_after_s": 1}), 429

    @app.after_request
    def _after(response):
        elapsed = time.perf_counter() - getattr(flask.g, "t_start", time.perf_counter())
        endpoint = getattr(flask.g, "endpoint", request.path)
        http_latency_seconds.observe(elapsed, endpoint=endpoint)
        http_requests_total.inc(
            method=request.method,
            endpoint=endpoint,
            status=str(response.status_code),
        )
        return response
