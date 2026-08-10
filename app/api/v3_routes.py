"""
app/api/v3_routes.py — v3 API: distributed systems (circuit breakers, priority queue, metrics).
"""
from __future__ import annotations

import datetime
import logging

from flask import Blueprint, Response, jsonify, request

from core.circuit_breaker import _BREAKERS, _BREAKER_LOCK, all_breaker_stats
from core.metrics import (
    REGISTRY, active_workers, cache_hits_total, cache_misses_total,
    disk_cache_entries, dlq_depth, inference_latency_s,
    memory_cache_entries, model_versions_total, queue_depth,
    training_jobs_total, Timer,
)
from ml.queue import PriorityJobQueue, RetryPolicy, PRIORITY_NORMAL
from core.config import AppConfig

log = logging.getLogger(__name__)


def make_v3_blueprint(training_svc, engine, cache, rate_limiter, cfg: AppConfig) -> Blueprint:
    bp = Blueprint("v3", __name__, url_prefix="/api/v3")

    pq = PriorityJobQueue(max_workers=cfg.max_worker_threads)

    @bp.route("/metrics", methods=["GET"])
    def v3_metrics():
        queue_depth.set(pq.stats()["queue_depth"])
        dlq_depth.set(pq.stats()["dlq_size"])
        active_workers.set(pq.stats()["by_status"].get("running", 0))

        accept = request.headers.get("Accept", "")
        if "application/json" in accept:
            return jsonify({
                "metrics": REGISTRY.as_dict(),
                "circuit_breakers": all_breaker_stats(),
                "rate_limiter": rate_limiter.stats(),
                "queue": pq.stats(),
            }), 200
        return Response(REGISTRY.format_prometheus(), mimetype="text/plain; version=0.0.4")

    @bp.route("/queue", methods=["GET"])
    def v3_queue():
        return jsonify({
            "stats": pq.stats(),
            "jobs": pq.list_jobs(),
            "dlq": pq.dlq_jobs(),
        }), 200

    @bp.route("/queue/dlq/<job_id>/requeue", methods=["POST"])
    def v3_dlq_requeue(job_id: str):
        ok = pq.requeue_from_dlq(job_id)
        if not ok:
            return jsonify({"error": f"Job {job_id} not found in DLQ"}), 404
        return jsonify({"status": "requeued", "job_id": job_id}), 200

    @bp.route("/train", methods=["POST"])
    def v3_train():
        payload = request.get_json(force=True) or {}
        ticker = payload.get("ticker", "AAPL").upper()
        start = payload.get("start_date", cfg.default_start_date)
        end = payload.get("end_date", datetime.date.today().isoformat())
        epochs = int(payload.get("epochs", cfg.epochs))
        seq_len = int(payload.get("seq_len", cfg.sequence_length))
        batch_sz = int(payload.get("batch_size", cfg.batch_size))
        mdl_list = payload.get("models", ["LSTM", "GRU", "Transformer"])
        priority = int(payload.get("priority", PRIORITY_NORMAL))

        retry_policy = RetryPolicy(
            max_retries=cfg.job_max_retries,
            base_delay_s=cfg.job_retry_base_delay_s,
            max_delay_s=cfg.job_retry_max_delay_s,
        )

        def _train_fn():
            return training_svc.submit_job(
                ticker=ticker, start_date=start, end_date=end,
                seq_len=seq_len, epochs=epochs, batch_size=batch_sz, models=mdl_list,
            )

        job_id = pq.submit(
            fn=_train_fn,
            payload={"ticker": ticker, "start": start, "end": end},
            priority=priority,
            retry_policy=retry_policy,
        )
        training_jobs_total.inc(status="queued")
        return jsonify({"job_id": job_id, "ticker": ticker, "priority": priority, "status": "queued"}), 202

    @bp.route("/predict", methods=["POST"])
    def v3_predict():
        with Timer(inference_latency_s, cache_layer="total"):
            payload = request.get_json(force=True) or {}
            ticker = payload.get("ticker", "AAPL").upper()
            start = payload.get("start_date", cfg.default_start_date)
            end = payload.get("end_date", datetime.date.today().isoformat())
            version = payload.get("version", "best")
            force = bool(payload.get("force_refresh", False))
            result = engine.predict(ticker, start, end, version=version, force_refresh=force)

        if "error" in result:
            return jsonify(result), 404
        if result.get("from_cache"):
            cache_hits_total.inc(layer="memory_or_disk")
        else:
            cache_misses_total.inc()
        return jsonify(result), 200

    @bp.route("/breakers", methods=["GET"])
    def v3_breakers():
        return jsonify(all_breaker_stats()), 200

    @bp.route("/breakers/<name>/reset", methods=["POST"])
    def v3_breaker_reset(name: str):
        with _BREAKER_LOCK:
            b = _BREAKERS.get(name)
        if b is None:
            return jsonify({"error": f"No circuit breaker named '{name}'"}), 404
        b.reset()
        return jsonify({"status": "reset", "name": name, "new_state": b.state.value}), 200

    @bp.route("/rate-limiter", methods=["GET"])
    def v3_rate_limiter_stats():
        return jsonify(rate_limiter.stats()), 200

    return bp
