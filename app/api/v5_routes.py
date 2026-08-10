"""app/api/v5_routes.py — Quantitative research platform."""
from flask import Blueprint, jsonify, request
import datetime


def make_v5_blueprint() -> Blueprint:
    bp = Blueprint("v5", __name__, url_prefix="/api/v5")

    @bp.route("/quant", methods=["POST"])
    def v5_quant_research():
        body = request.get_json(force=True, silent=True) or {}
        ticker = str(body.get("ticker", "AAPL")).strip().upper()
        start_date = str(body.get("start_date", "2020-01-01"))
        end_date = str(body.get("end_date", datetime.date.today().isoformat()))
        if not ticker:
            return jsonify({"error": "ticker is required"}), 400
        from ml.quant_analytics import compute_quant_research_report
        report = compute_quant_research_report(ticker, start_date, end_date)
        return jsonify(report), 200

    return bp
