"""
app/middleware/error_handlers.py — Global exception → JSON response mapping.

Engineering decisions:
  - Domain exceptions are mapped to HTTP status codes here, NOT in services.
  - This is the correct SRP boundary: services raise domain errors,
    the HTTP layer translates them.
  - A catch-all handler returns 500 with a sanitised message (no stack trace
    exposed to the client in production).
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify

from app.domain.exceptions import (
    DataFetchError,
    InsufficientDataError,
    ModelNotFoundError,
    StockBuddyError,
    TrainingError,
    ConfigurationError,
)

log = logging.getLogger(__name__)


def register_error_handlers(app: Flask) -> None:
    """Attach all domain exception handlers to the Flask app."""

    @app.errorhandler(DataFetchError)
    def handle_data_fetch(exc: DataFetchError):
        log.warning("DataFetchError: %s", exc.message)
        return jsonify(exc.to_dict()), 502

    @app.errorhandler(InsufficientDataError)
    def handle_insufficient(exc: InsufficientDataError):
        log.warning("InsufficientDataError: %s", exc.message)
        return jsonify(exc.to_dict()), 422

    @app.errorhandler(ModelNotFoundError)
    def handle_model_not_found(exc: ModelNotFoundError):
        log.info("ModelNotFoundError: %s", exc.message)
        return jsonify(exc.to_dict()), 404

    @app.errorhandler(TrainingError)
    def handle_training(exc: TrainingError):
        log.error("TrainingError: %s", exc.message)
        return jsonify(exc.to_dict()), 500

    @app.errorhandler(ConfigurationError)
    def handle_config(exc: ConfigurationError):
        log.critical("ConfigurationError: %s", exc.message)
        return jsonify(exc.to_dict()), 500

    @app.errorhandler(StockBuddyError)
    def handle_generic_domain(exc: StockBuddyError):
        log.error("StockBuddyError: %s", exc.message)
        return jsonify(exc.to_dict()), 500

    @app.errorhandler(404)
    def handle_not_found(exc):
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(exc):
        return jsonify({"error": "Method not allowed"}), 405

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        log.exception("Unhandled exception: %s", exc)
        return jsonify({"error": "Internal server error"}), 500
