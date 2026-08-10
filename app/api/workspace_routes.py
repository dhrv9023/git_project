"""
app/api/workspace_routes.py — Workspace, Watchlist, and Alert endpoints.

Engineering decision: all endpoints read user_id from a JWT token header
(when auth is enabled) or fall back to "default_user" so the API works
without authentication during development.
"""
from __future__ import annotations

import logging
from flask import Blueprint, jsonify, request

log = logging.getLogger(__name__)


def make_workspace_blueprint(workspace_store, cfg) -> Blueprint:
    bp = Blueprint("workspace", __name__, url_prefix="/api/v7")

    def _user_id() -> str:
        """Extract user_id from Bearer token or default to 'default_user'."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            # Real JWT decode happens in P1-C; for now extract embedded user
            try:
                parts = token.split(".")
                if len(parts) >= 2:
                    return parts[1] or "default_user"
            except Exception:
                pass
        return "default_user"

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    @bp.route("/workspaces", methods=["GET"])
    def list_workspaces():
        """GET /api/v7/workspaces — List all saved workspaces for current user."""
        return jsonify({"workspaces": workspace_store.list_workspaces(_user_id())}), 200

    @bp.route("/workspaces", methods=["POST"])
    def save_workspace():
        """POST /api/v7/workspaces — Save or update a workspace."""
        body = request.get_json(silent=True) or {}
        name = body.get("name", "default")
        config = body.get("config", {})
        ws = workspace_store.save_workspace(_user_id(), name, config)
        return jsonify(ws), 201

    @bp.route("/workspaces/<name>", methods=["GET"])
    def get_workspace(name: str):
        ws = workspace_store.get_workspace(_user_id(), name)
        if ws is None:
            return jsonify({"error": f"Workspace '{name}' not found"}), 404
        return jsonify(ws), 200

    @bp.route("/workspaces/<name>", methods=["PUT"])
    def update_workspace(name: str):
        body = request.get_json(silent=True) or {}
        ws = workspace_store.save_workspace(_user_id(), name, body.get("config", {}))
        return jsonify(ws), 200

    @bp.route("/workspaces/<name>", methods=["DELETE"])
    def delete_workspace(name: str):
        deleted = workspace_store.delete_workspace(_user_id(), name)
        if not deleted:
            return jsonify({"error": f"Workspace '{name}' not found"}), 404
        return jsonify({"status": "deleted", "name": name}), 200

    # ------------------------------------------------------------------
    # Watchlists
    # ------------------------------------------------------------------

    @bp.route("/watchlist", methods=["GET"])
    def get_watchlist():
        """GET /api/v7/watchlist — Return current user's watchlist."""
        tickers = workspace_store.get_watchlist(_user_id())
        return jsonify({"watchlist": tickers, "count": len(tickers)}), 200

    @bp.route("/watchlist", methods=["POST"])
    def add_to_watchlist():
        """POST /api/v7/watchlist — Add ticker to watchlist."""
        body = request.get_json(silent=True) or {}
        ticker = str(body.get("ticker", "")).upper().strip()
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400
        tickers = workspace_store.add_to_watchlist(_user_id(), ticker)
        return jsonify({"watchlist": tickers, "added": ticker}), 201

    @bp.route("/watchlist/<ticker>", methods=["DELETE"])
    def remove_from_watchlist(ticker: str):
        """DELETE /api/v7/watchlist/{ticker} — Remove ticker from watchlist."""
        tickers = workspace_store.remove_from_watchlist(_user_id(), ticker.upper())
        return jsonify({"watchlist": tickers, "removed": ticker.upper()}), 200

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    @bp.route("/alerts", methods=["GET"])
    def get_alerts():
        """GET /api/v7/alerts — List alerts for current user."""
        status_filter = request.args.get("status")
        alerts = workspace_store.get_alerts(_user_id(), status=status_filter)
        return jsonify({"alerts": alerts, "count": len(alerts)}), 200

    @bp.route("/alerts", methods=["POST"])
    def create_alert():
        """POST /api/v7/alerts — Create a new price/RSI/regime alert."""
        body = request.get_json(silent=True) or {}
        ticker = str(body.get("ticker", "AAPL")).upper().strip()
        condition_type = str(body.get("condition_type", "PRICE_ABOVE")).upper()
        threshold = float(body.get("threshold", 0.0))

        valid_conditions = {
            "PRICE_ABOVE", "PRICE_BELOW",
            "RSI_ABOVE", "RSI_BELOW",
            "REGIME_CHANGE", "DRAWDOWN_EXCEEDS",
        }
        if condition_type not in valid_conditions:
            return jsonify({"error": f"Invalid condition_type. Choose from: {valid_conditions}"}), 400

        alert = workspace_store.create_alert(_user_id(), ticker, condition_type, threshold)
        return jsonify(alert), 201

    @bp.route("/alerts/<alert_id>", methods=["DELETE"])
    def delete_alert(alert_id: str):
        """DELETE /api/v7/alerts/{alert_id} — Delete an alert."""
        deleted = workspace_store.delete_alert(_user_id(), alert_id)
        if not deleted:
            return jsonify({"error": f"Alert {alert_id} not found"}), 404
        return jsonify({"status": "deleted", "alert_id": alert_id}), 200

    @bp.route("/alerts/<alert_id>", methods=["PATCH"])
    def update_alert(alert_id: str):
        """PATCH /api/v7/alerts/{alert_id} — Deactivate / reactivate an alert."""
        body = request.get_json(silent=True) or {}
        status = str(body.get("status", "ACTIVE")).upper()
        if status not in {"ACTIVE", "PAUSED", "DELETED"}:
            return jsonify({"error": "status must be ACTIVE, PAUSED, or DELETED"}), 400
        alert = workspace_store.update_alert_status(alert_id, status)
        if alert is None:
            return jsonify({"error": f"Alert {alert_id} not found"}), 404
        return jsonify(alert), 200

    return bp
