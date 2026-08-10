"""
app/api/v2_routes.py — v2 API: async training, registry, cached inference.
"""
from __future__ import annotations

import datetime
import logging

from flask import Blueprint, jsonify, request

from core.config import AppConfig

log = logging.getLogger(__name__)


def make_v2_blueprint(training_svc, engine, cache, store, scheduler, cfg: AppConfig) -> Blueprint:
    bp = Blueprint("v2", __name__, url_prefix="/api/v2")

    @bp.route("/train", methods=["POST"])
    def v2_train():
        payload = request.get_json(force=True) or {}
        ticker = payload.get("ticker", "AAPL").upper()
        start = payload.get("start_date", cfg.default_start_date)
        end = payload.get("end_date", datetime.date.today().isoformat())
        epochs = int(payload.get("epochs", cfg.epochs))
        seq_len = int(payload.get("seq_len", cfg.sequence_length))
        batch_sz = int(payload.get("batch_size", cfg.batch_size))
        mdl_list = payload.get("models", ["LSTM", "GRU", "Transformer"])

        job_id = training_svc.submit_job(
            ticker=ticker, start_date=start, end_date=end,
            seq_len=seq_len, epochs=epochs, batch_size=batch_sz, models=mdl_list,
        )
        job = training_svc.get_job(job_id)
        return jsonify({
            "job_id": job_id,
            "version": job.version if job else None,
            "ticker": ticker,
            "status": "queued",
            "message": f"Training job enqueued. Poll GET /api/v2/jobs/{job_id} for status.",
        }), 202

    @bp.route("/jobs/<job_id>", methods=["GET"])
    def v2_job_status(job_id: str):
        job = training_svc.get_job(job_id)
        if job is None:
            return jsonify({"error": f"Job {job_id} not found"}), 404
        return jsonify(job.to_dict()), 200

    @bp.route("/jobs", methods=["GET"])
    def v2_list_jobs():
        return jsonify({
            "jobs": training_svc.list_jobs(),
            "active_count": training_svc.active_count(),
        }), 200

    @bp.route("/registry", methods=["GET"])
    def v2_registry():
        return jsonify(training_svc.full_registry()), 200

    @bp.route("/registry/<ticker>", methods=["GET"])
    def v2_registry_ticker(ticker: str):
        ticker = ticker.upper()
        versions = training_svc.list_versions(ticker)
        if not versions:
            return jsonify({"error": f"No models found for {ticker}"}), 404
        return jsonify({
            "ticker": ticker,
            "versions": versions,
            "best": training_svc.get_best(ticker),
            "latest": training_svc.get_latest(ticker),
        }), 200

    @bp.route("/predict", methods=["POST"])
    def v2_predict():
        payload = request.get_json(force=True) or {}
        ticker = payload.get("ticker", "AAPL").upper()
        start = payload.get("start_date", cfg.default_start_date)
        end = payload.get("end_date", datetime.date.today().isoformat())
        version = payload.get("version", "best")
        force = bool(payload.get("force_refresh", False))

        result = engine.predict(ticker, start, end, version=version, force_refresh=force)
        if "error" in result:
            return jsonify(result), 404
        return jsonify(result), 200

    @bp.route("/cache", methods=["DELETE"])
    def v2_flush_cache():
        cache.flush()
        engine.evict_artifacts()
        return jsonify({"status": "flushed"}), 200

    @bp.route("/metrics", methods=["GET"])
    def v2_metrics():
        return jsonify({
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "registry": training_svc.registry_stats(),
            "training": {
                "active_jobs": training_svc.active_count(),
                "total_jobs": len(training_svc.list_jobs()),
            },
            "inference_cache": cache.stats(),
            "artifact_store": {
                "disk_mb": round(store.disk_usage_bytes() / 1024 ** 2, 2),
            },
            "scheduler": scheduler.status(),
        }), 200

    return bp
