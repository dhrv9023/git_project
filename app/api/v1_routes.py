"""
app/api/v1_routes.py — Original v1 API endpoints (backward compatible).

Engineering decision: routes are thin — they parse the request, call
a service, and jsonify the result. No business logic lives here.
The make_v1_blueprint() factory receives services via DI.
"""
from __future__ import annotations

import datetime
import logging

from flask import Blueprint, jsonify, request

from core.config import AppConfig

log = logging.getLogger(__name__)


def make_v1_blueprint(market_repo, regime_svc, backtest_svc, engine, cfg: AppConfig) -> Blueprint:
    bp = Blueprint("v1", __name__)

    @bp.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "phase": "Phase-2-Production-Pipeline"}), 200

    @bp.route("/api/regime", methods=["POST"])
    def api_regime():
        """POST /api/regime — Market regime classification."""
        payload = request.get_json(force=True) or {}
        ticker = payload.get("ticker", "AAPL").upper()
        three_years_ago = (datetime.date.today() - datetime.timedelta(days=3 * 365)).isoformat()
        start = payload.get("start_date", three_years_ago)
        end = payload.get("end_date", datetime.date.today().isoformat())

        result = regime_svc.classify(ticker, start, end)
        return jsonify(result.to_dict()), 200

    @bp.route("/api/predict", methods=["POST"])
    def api_predict():
        """POST /api/predict — Train + predict (legacy endpoint, uses in-memory cache)."""
        import numpy as np
        from ml.features import split_and_scale_data
        from ml.models import build_model, make_callbacks
        from app.repositories.market_data_repo import MarketDataRepository
        from app.services.backtest_service import BacktestService

        payload = request.get_json(force=True) or {}
        ticker = payload.get("ticker", cfg.default_ticker).upper()
        start = payload.get("start_date", cfg.default_start_date)
        end = payload.get("end_date", datetime.date.today().isoformat())
        seq_len = int(payload.get("sequence_length", cfg.sequence_length))
        epochs = int(payload.get("epochs", cfg.epochs))
        batch_size = int(payload.get("batch_size", cfg.batch_size))
        future_days = int(payload.get("future_days", 5))
        selected = payload.get("model")
        model_list = [selected] if selected in {"LSTM", "GRU", "Transformer"} else ["LSTM", "GRU", "Transformer"]

        repo = MarketDataRepository()
        data = repo.build_feature_matrix(ticker, start, end, seq_len)
        splits = split_and_scale_data(
            data["X_raw"], data["y_raw"], data["dates_raw"], data["base_prices_raw"],
            cfg.train_split, cfg.val_split, seq_len,
        )
        input_shape = (seq_len, data["X_raw"].shape[1])
        scaler_y = splits["scaler_y"]
        scaler_X = splits["scaler_X"]

        X_train, y_train, _, _ = splits["train"]
        X_val, y_val, _, _ = splits["val"]
        X_test, y_test, dates_test, base_test = splits["test"]
        steps = max(1, len(X_train) // batch_size)

        models_dict = {}
        for mtype in model_list:
            model = build_model(mtype, input_shape, cfg.learning_rate)
            cbs = make_callbacks(epochs, steps, cfg.learning_rate)
            history = model.fit(
                X_train, y_train, validation_data=(X_val, y_val),
                epochs=epochs, batch_size=batch_size, verbose=0, callbacks=cbs,
            )
            models_dict[mtype] = model
            models_dict[f"{mtype}_history"] = history.history

        bt_svc = BacktestService(cfg)
        preds = {}
        for name in model_list:
            y_pred_scaled = models_dict[name].predict(X_test, verbose=0).ravel()
            logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            logret_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
            preds[name] = base_test * np.exp(logret_pred)

        y_true_prices = base_test * np.exp(logret_true)
        y_stack = np.stack([preds[n] for n in model_list], axis=1)
        y_ens = y_stack.mean(axis=1)
        preds["Ensemble"] = y_ens

        metrics = {name: bt_svc.calculate_metrics(y_true_prices, preds[name]) for name in list(model_list) + ["Ensemble"]}
        backtest = bt_svc.run_signal_backtest(y_ens, y_true_prices, cfg.initial_capital)

        def _s(v):
            import math
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return v

        return jsonify({
            "ticker": ticker,
            "dates": [d.strftime("%Y-%m-%d") for d in dates_test],
            "actual": [_s(v) for v in y_true_prices.tolist()],
            "predictions": {k: [_s(v) for v in arr.tolist()] for k, arr in preds.items()},
            "metrics": {k: {mk: _s(mv) for mk, mv in m.items()} for k, m in metrics.items()},
            "backtest": backtest,
            "future": {"dates": [], "predictions": []},
        }), 200

    @bp.route("/api/wf_validate", methods=["POST"])
    def api_wf_validate():
        """POST /api/wf_validate — Walk-forward validation."""
        from ml.features import split_and_scale_data, create_sequences
        from ml.models import build_model
        from app.repositories.market_data_repo import MarketDataRepository
        from app.services.backtest_service import BacktestService
        from sklearn.preprocessing import MinMaxScaler
        import numpy as np
        from tensorflow.keras import callbacks as cb_module

        payload = request.get_json(force=True) or {}
        ticker = payload.get("ticker", "AAPL").upper()
        start = payload.get("start_date", cfg.default_start_date)
        end = payload.get("end_date", datetime.date.today().isoformat())
        n_folds = int(payload.get("n_folds", 5))
        model_type = payload.get("model", "GRU")
        epochs = int(payload.get("epochs", 10))
        batch_size = int(payload.get("batch_size", 32))

        repo = MarketDataRepository()
        data = repo.build_feature_matrix(ticker, start, end, cfg.sequence_length)
        X_raw = data["X_raw"]
        y_raw = data["y_raw"]
        base = data["base_prices_raw"]
        seq_len = cfg.sequence_length
        n = len(X_raw)
        fold_size = n // (n_folds + 1)
        fold_results = []
        bt_svc = BacktestService(cfg)

        for fold in range(n_folds):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n)
            if test_end - test_start < seq_len + 10:
                continue

            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            scaler_X.fit(X_raw[:train_end])
            scaler_y.fit(y_raw[:train_end].reshape(-1, 1))

            X_tr = scaler_X.transform(X_raw[:train_end])
            y_tr = scaler_y.transform(y_raw[:train_end].reshape(-1, 1)).ravel()
            X_te_ctx = np.concatenate([X_tr[-seq_len:], scaler_X.transform(X_raw[test_start:test_end])], axis=0)
            y_te_ctx = np.concatenate([y_tr[-seq_len:], scaler_y.transform(y_raw[test_start:test_end].reshape(-1, 1)).ravel()], axis=0)
            X_tr_seq, y_tr_seq = create_sequences(X_tr, y_tr, seq_len)
            X_te_seq, y_te_seq = create_sequences(X_te_ctx, y_te_ctx, seq_len)

            if len(X_tr_seq) < 10 or len(X_te_seq) < 5:
                continue

            model = build_model(model_type, (seq_len, X_raw.shape[1]), cfg.learning_rate)
            es = cb_module.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)
            model.fit(X_tr_seq, y_tr_seq, epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

            y_pred_scaled = model.predict(X_te_seq, verbose=0).ravel()
            logret_true = scaler_y.inverse_transform(y_te_seq.reshape(-1, 1)).ravel()
            logret_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            base_te = base[test_start:test_end][:len(y_te_seq)]
            m = bt_svc.calculate_metrics(base_te * np.exp(logret_true), base_te * np.exp(logret_pred))
            da = float((np.sign(logret_true) == np.sign(logret_pred)).mean() * 100)
            fold_results.append({"fold": fold + 1, "rmse": round(m["RMSE"], 4), "directional_accuracy": round(da, 2), "r2": round(m["R2"], 4)})

        if not fold_results:
            return jsonify({"error": "Insufficient data for walk-forward validation", "folds": []}), 422

        das = [f["directional_accuracy"] for f in fold_results]
        rmses = [f["rmse"] for f in fold_results]
        return jsonify({
            "model": model_type, "n_folds": len(fold_results), "folds": fold_results,
            "summary": {"mean_directional_accuracy": round(float(np.mean(das)), 2),
                        "std_directional_accuracy": round(float(np.std(das)), 2),
                        "mean_rmse": round(float(np.mean(rmses)), 4)},
        }), 200

    return bp
