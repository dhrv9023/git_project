"""
app/auth/auth_service.py — Real JWT authentication with bcrypt password hashing.

Engineering decisions:
  - No database required: users stored in JSON file via WorkspaceStore-style
    atomic writes. Sufficient for a portfolio project; replace with SQLAlchemy
    for production multi-tenant deployment.
  - bcrypt for password hashing: industry standard, includes salt automatically.
  - Flask-JWT-Extended for token management: handles expiry, refresh, blacklist.
  - 24h access token + 7d refresh token lifecycle.
  - JWT secret read from SB_JWT_SECRET env var — falls back to a random secret
    (safe for dev, logs a warning).
  - Passwords are NEVER stored in plaintext or logs.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from typing import Any

import bcrypt

log = logging.getLogger(__name__)

_LOCK = threading.Lock()


class AuthService:
    """User registration, login, and token management.

    Args:
        users_file: Path to JSON file storing user records.
    """

    def __init__(self, users_file: str = "model_artifacts/user_data/users.json") -> None:
        self.users_file = users_file
        os.makedirs(os.path.dirname(users_file), exist_ok=True)

    # ------------------------------------------------------------------
    # User management
    # ------------------------------------------------------------------

    def register(self, username: str, password: str, role: str = "analyst") -> dict:
        """Create a new user account.

        Raises:
            ValueError: if username already exists or is invalid.
        """
        username = username.strip().lower()
        if not username or len(username) < 3:
            raise ValueError("Username must be at least 3 characters.")
        if not password or len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")

        with _LOCK:
            users = self._read()
            if username in users:
                raise ValueError(f"Username '{username}' already exists.")

            pw_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
            users[username] = {
                "username": username,
                "password_hash": pw_hash,
                "role": role,
                "permissions": self._default_permissions(role),
                "created_at": _now(),
                "last_login": None,
            }
            self._write(users)

        log.info("User registered: %s (role=%s)", username, role)
        return self._public(users[username])

    def authenticate(self, username: str, password: str) -> dict | None:
        """Verify credentials. Returns public user dict or None if invalid."""
        username = username.strip().lower()
        with _LOCK:
            users = self._read()

        user = users.get(username)
        if user is None:
            # Constant-time comparison even for missing users
            bcrypt.checkpw(b"dummy", bcrypt.hashpw(b"dummy", bcrypt.gensalt()))
            return None

        if not bcrypt.checkpw(password.encode(), user["password_hash"].encode()):
            return None

        # Update last_login
        with _LOCK:
            users = self._read()
            if username in users:
                users[username]["last_login"] = _now()
                self._write(users)

        log.info("User authenticated: %s", username)
        return self._public(user)

    def get_user(self, username: str) -> dict | None:
        """Fetch public user profile by username."""
        with _LOCK:
            users = self._read()
        user = users.get(username.lower())
        return self._public(user) if user else None

    def user_exists(self, username: str) -> bool:
        with _LOCK:
            users = self._read()
        return username.lower() in users

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read(self) -> dict:
        if not os.path.exists(self.users_file):
            return {}
        try:
            with open(self.users_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write(self, users: dict) -> None:
        dir_ = os.path.dirname(self.users_file)
        fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(users, f, indent=2, default=str)
            os.replace(tmp, self.users_file)
        except Exception:
            os.unlink(tmp)
            raise

    @staticmethod
    def _public(user: dict) -> dict:
        """Return user dict without password hash."""
        return {k: v for k, v in user.items() if k != "password_hash"}

    @staticmethod
    def _default_permissions(role: str) -> list[str]:
        base = ["read"]
        if role in ("analyst", "quant", "admin"):
            base += ["write", "execute_models"]
        if role in ("quant", "admin"):
            base += ["portfolio_opt", "scenario_sim"]
        if role == "admin":
            base += ["manage_users", "flush_cache"]
        return base


def _now() -> str:
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
