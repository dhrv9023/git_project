"""
storage/workspace_store.py — Persistent JSON storage for workspaces, watchlists, and alerts.

Engineering decisions:
  - Same file-locking pattern as ModelStore (threading.Lock) for thread safety.
  - Each entity type (workspaces, watchlists, alerts) lives in its own JSON file
    so concurrent reads/writes don't block each other.
  - Atomic writes via write-to-temp + os.replace so a crash mid-write never
    corrupts the existing file.
  - IDs are generated as {entity_type}_{timestamp_ms} — no UUID dependency.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any

log = logging.getLogger(__name__)

_LOCKS: dict[str, threading.Lock] = {}
_LOCK_REGISTRY = threading.Lock()


def _get_lock(path: str) -> threading.Lock:
    with _LOCK_REGISTRY:
        if path not in _LOCKS:
            _LOCKS[path] = threading.Lock()
        return _LOCKS[path]


def _read(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        log.warning("workspace_store: corrupt file %s — resetting", path)
        return {}


def _write(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp)
        raise


class WorkspaceStore:
    """Thread-safe, file-backed store for workspaces, watchlists, and alerts.

    Args:
        base_dir: Root directory for storage (default: model_artifacts_dir/user_data)
    """

    def __init__(self, base_dir: str = "model_artifacts/user_data") -> None:
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, filename: str) -> str:
        return os.path.join(self.base_dir, filename)

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def save_workspace(self, user_id: str, name: str, config: dict) -> dict:
        path = self._path("workspaces.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            key = f"{user_id}:{name}"
            workspace = {
                "user_id": user_id,
                "name": name,
                "config": config,
                "updated_at": _now(),
            }
            data[key] = workspace
            _write(path, data)
        log.debug("workspace saved: %s", key)
        return workspace

    def get_workspace(self, user_id: str, name: str) -> dict | None:
        path = self._path("workspaces.json")
        with _get_lock(path):
            data = _read(path)
        return data.get(f"{user_id}:{name}")

    def list_workspaces(self, user_id: str) -> list[dict]:
        path = self._path("workspaces.json")
        with _get_lock(path):
            data = _read(path)
        return [v for k, v in data.items() if k.startswith(f"{user_id}:")]

    def delete_workspace(self, user_id: str, name: str) -> bool:
        path = self._path("workspaces.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            key = f"{user_id}:{name}"
            if key not in data:
                return False
            del data[key]
            _write(path, data)
        return True

    # ------------------------------------------------------------------
    # Watchlists
    # ------------------------------------------------------------------

    def get_watchlist(self, user_id: str) -> list[str]:
        path = self._path("watchlists.json")
        with _get_lock(path):
            data = _read(path)
        return data.get(user_id, [])

    def add_to_watchlist(self, user_id: str, ticker: str) -> list[str]:
        path = self._path("watchlists.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            tickers = data.get(user_id, [])
            ticker = ticker.upper().strip()
            if ticker not in tickers:
                tickers.append(ticker)
                data[user_id] = tickers
                _write(path, data)
        return data.get(user_id, tickers)

    def remove_from_watchlist(self, user_id: str, ticker: str) -> list[str]:
        path = self._path("watchlists.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            tickers = data.get(user_id, [])
            ticker = ticker.upper().strip()
            if ticker in tickers:
                tickers.remove(ticker)
                data[user_id] = tickers
                _write(path, data)
        return data.get(user_id, tickers)

    # ------------------------------------------------------------------
    # Alerts
    # ------------------------------------------------------------------

    def create_alert(self, user_id: str, ticker: str, condition_type: str, threshold: float) -> dict:
        path = self._path("alerts.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            alert_id = f"alt_{int(time.time() * 1000)}"
            alert: dict[str, Any] = {
                "alert_id": alert_id,
                "user_id": user_id,
                "ticker": ticker.upper(),
                "condition_type": condition_type,
                "threshold": threshold,
                "status": "ACTIVE",
                "triggered_at": None,
                "created_at": _now(),
            }
            data[alert_id] = alert
            _write(path, data)
        return alert

    def get_alerts(self, user_id: str, status: str | None = None) -> list[dict]:
        path = self._path("alerts.json")
        with _get_lock(path):
            data = _read(path)
        alerts = [v for v in data.values() if v.get("user_id") == user_id]
        if status:
            alerts = [a for a in alerts if a.get("status") == status]
        return sorted(alerts, key=lambda a: a["created_at"], reverse=True)

    def update_alert_status(self, alert_id: str, status: str, triggered_at: str | None = None) -> dict | None:
        path = self._path("alerts.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            if alert_id not in data:
                return None
            data[alert_id]["status"] = status
            if triggered_at:
                data[alert_id]["triggered_at"] = triggered_at
            _write(path, data)
        return data[alert_id]

    def delete_alert(self, user_id: str, alert_id: str) -> bool:
        path = self._path("alerts.json")
        lock = _get_lock(path)
        with lock:
            data = _read(path)
            if alert_id not in data or data[alert_id].get("user_id") != user_id:
                return False
            del data[alert_id]
            _write(path, data)
        return True

    def get_all_active_alerts(self) -> list[dict]:
        """Return all ACTIVE alerts across all users — used by AlertEvaluator."""
        path = self._path("alerts.json")
        with _get_lock(path):
            data = _read(path)
        return [v for v in data.values() if v.get("status") == "ACTIVE"]


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
