"""
ml/registry.py — Model Registry backed by a JSON file.

Production pattern: every model training run is tracked in a central registry.
This is the single source of truth for:
  - Which model versions exist for a ticker
  - Which version is currently "best" (lowest RMSE, highest DA)
  - Whether a model is stale and needs retraining
  - Full training provenance (config, dates, metrics)

Inspired by MLflow Model Registry and Google Vertex AI Model Registry,
but with zero external dependencies — just a thread-safe JSON file.

Schema (registry.json):
{
  "AAPL": {
    "latest": "v3",
    "best": "v3",
    "versions": {
      "v1": { <ModelRecord> },
      "v2": { <ModelRecord> },
      "v3": { <ModelRecord> }
    }
  }
}

ModelRecord fields:
  version, model_id, ticker, start_date, end_date, seq_len, epochs,
  status (queued|training|ready|stale|failed),
  metrics { model_name: { RMSE, DA, R2 } },
  created_at, trained_at, model_path, stale_after_days, is_stale
"""

import os
import json
import threading
import datetime
import logging
from typing import Optional, Dict, Any, List

log = logging.getLogger(__name__)

# Valid lifecycle states (modelled after Kubeflow Pipeline run states)
VALID_STATUSES = {"queued", "training", "ready", "stale", "failed"}


class ModelRegistry:
    """
    Thread-safe, file-backed model registry.

    All public methods acquire a lock before reading/writing registry.json
    so the background trainer and Flask request threads can safely share it.
    """

    def __init__(self, registry_path: str = "model_artifacts/registry.json"):
        self.registry_path = registry_path
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        self._lock = threading.RLock()
        self._ensure_file()

    # ── File I/O ──────────────────────────────────────────────────────────────

    def _ensure_file(self):
        if not os.path.exists(self.registry_path):
            with open(self.registry_path, "w") as f:
                json.dump({}, f, indent=2)

    def _read(self) -> dict:
        with open(self.registry_path) as f:
            return json.load(f)

    def _write(self, data: dict):
        tmp = self.registry_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, self.registry_path)   # atomic on POSIX

    # ── Version management ────────────────────────────────────────────────────

    def _next_version(self, ticker_data: dict) -> str:
        """Generate next version string: v1, v2, v3, ..."""
        existing = list(ticker_data.get("versions", {}).keys())
        if not existing:
            return "v1"
        nums = []
        for v in existing:
            try:
                nums.append(int(v.lstrip("v")))
            except ValueError:
                pass
        return f"v{max(nums) + 1}"

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, ticker: str, config: Dict[str, Any],
                 status: str = "queued") -> str:
        """
        Create a new model version record and return the version string.
        Called when a training job is submitted.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            if ticker not in data:
                data[ticker] = {"latest": None, "best": None, "versions": {}}
            version = self._next_version(data[ticker])
            model_id = f"{ticker}_{datetime.date.today().isoformat()}_{version}"
            record = {
                "version":        version,
                "model_id":       model_id,
                "ticker":         ticker,
                "start_date":     config.get("start_date", ""),
                "end_date":       config.get("end_date", ""),
                "seq_len":        config.get("seq_len", 90),
                "epochs":         config.get("epochs", 20),
                "models_trained": config.get("models", ["LSTM", "GRU", "Transformer"]),
                "status":         status,
                "metrics":        {},
                "created_at":     datetime.datetime.utcnow().isoformat(),
                "trained_at":     None,
                "model_path":     None,
                "stale_after_days": config.get("stale_after_days", 7),
                "is_stale":       False,
                "job_id":         config.get("job_id", None),
            }
            data[ticker]["versions"][version] = record
            self._write(data)
            log.info(f"Registered {ticker}/{version} (status={status})")
            return version

    def update_status(self, ticker: str, version: str,
                      status: str,
                      metrics: Optional[Dict] = None,
                      model_path: Optional[str] = None):
        """Update status + optional metrics after training completes or fails."""
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            rec = data.get(ticker, {}).get("versions", {}).get(version)
            if rec is None:
                raise KeyError(f"No record for {ticker}/{version}")
            rec["status"] = status
            if metrics:
                rec["metrics"] = metrics
            if model_path:
                rec["model_path"] = model_path
            if status == "ready":
                rec["trained_at"] = datetime.datetime.utcnow().isoformat()
                # Promote to latest
                data[ticker]["latest"] = version
                # Promote to best if this has lower ensemble RMSE
                best_v = data[ticker].get("best")
                if best_v is None:
                    data[ticker]["best"] = version
                else:
                    best_rec = data[ticker]["versions"].get(best_v, {})
                    best_rmse = best_rec.get("metrics", {}).get("Ensemble", {}).get("RMSE", float("inf"))
                    new_rmse  = metrics.get("Ensemble", {}).get("RMSE", float("inf")) if metrics else float("inf")
                    if new_rmse < best_rmse:
                        data[ticker]["best"] = version
            self._write(data)
            log.info(f"Updated {ticker}/{version} → status={status}")

    def mark_stale(self, ticker: str, version: str):
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            rec = data.get(ticker, {}).get("versions", {}).get(version)
            if rec:
                rec["status"]   = "stale"
                rec["is_stale"] = True
                self._write(data)
                log.info(f"Marked {ticker}/{version} as stale")

    def get_record(self, ticker: str, version: str) -> Optional[Dict]:
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            return data.get(ticker, {}).get("versions", {}).get(version)

    def get_latest(self, ticker: str) -> Optional[Dict]:
        """Return the most recently trained READY record, or None."""
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            tdata = data.get(ticker, {})
            latest_v = tdata.get("latest")
            if not latest_v:
                return None
            return tdata.get("versions", {}).get(latest_v)

    def get_best(self, ticker: str) -> Optional[Dict]:
        """Return the record with best (lowest) ensemble RMSE."""
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            tdata = data.get(ticker, {})
            best_v = tdata.get("best")
            if not best_v:
                return None
            return tdata.get("versions", {}).get(best_v)

    def list_versions(self, ticker: str) -> List[Dict]:
        ticker = ticker.upper()
        with self._lock:
            data = self._read()
            versions = data.get(ticker, {}).get("versions", {})
            return sorted(versions.values(), key=lambda r: r.get("created_at", ""))

    def all_tickers(self) -> List[str]:
        with self._lock:
            return list(self._read().keys())

    def full_registry(self) -> dict:
        with self._lock:
            return self._read()

    def get_stale_ready_models(self, stale_after_days: int = 7) -> List[Dict]:
        """
        Return all READY models whose 'trained_at' is older than stale_after_days.
        Used by the scheduler to trigger automatic retraining.
        """
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=stale_after_days)
        stale = []
        with self._lock:
            data = self._read()
            for ticker, tdata in data.items():
                for version, rec in tdata.get("versions", {}).items():
                    if rec.get("status") != "ready":
                        continue
                    trained_at_str = rec.get("trained_at")
                    if not trained_at_str:
                        continue
                    try:
                        trained_at = datetime.datetime.fromisoformat(trained_at_str)
                        if trained_at < cutoff:
                            stale.append(rec)
                    except ValueError:
                        pass
        return stale

    def stats(self) -> dict:
        """Return aggregate counts for the /api/v2/metrics endpoint."""
        with self._lock:
            data = self._read()
        total_versions = sum(len(t.get("versions", {})) for t in data.values())
        by_status: Dict[str, int] = {}
        for tdata in data.values():
            for rec in tdata.get("versions", {}).values():
                s = rec.get("status", "unknown")
                by_status[s] = by_status.get(s, 0) + 1
        return {
            "total_tickers":  len(data),
            "total_versions": total_versions,
            "by_status":      by_status,
        }
