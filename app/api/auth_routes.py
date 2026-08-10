"""
app/api/auth_routes.py — Real JWT authentication endpoints.

Engineering decisions:
  - Flask-JWT-Extended handles token signing, expiry, and refresh.
  - JWT secret loaded from SB_JWT_SECRET env var; falls back to random
    secret (dev-only — logs a warning so it's never silently insecure).
  - /register is open; /login is open; all other routes require @jwt_required.
  - Refresh token endpoint allows clients to get new access tokens without
    re-entering credentials.
  - Token blacklisting not implemented (stateless JWTs); add Redis-backed
    blocklist for production.
"""
from __future__ import annotations

import logging
import os

from flask import Blueprint, jsonify, request
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    jwt_required,
)

from app.auth.auth_service import AuthService

log = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__, url_prefix="/api/v7/auth")


def init_jwt(app) -> JWTManager:
    """Configure Flask-JWT-Extended on the Flask app.

    Engineering decision: JWT_SECRET_KEY is read from env SB_JWT_SECRET.
    If not set, a random 32-byte key is generated and a WARNING is logged.
    This means tokens are invalidated on every server restart — acceptable
    for dev, but SB_JWT_SECRET MUST be set in production.
    """
    secret = os.environ.get("SB_JWT_SECRET")
    if not secret:
        import secrets
        secret = secrets.token_hex(32)
        log.warning(
            "SB_JWT_SECRET not set — using ephemeral random JWT secret. "
            "All tokens will be invalidated on server restart. "
            "Set SB_JWT_SECRET in production."
        )
    app.config["JWT_SECRET_KEY"] = secret
    app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 86_400      # 24 hours
    app.config["JWT_REFRESH_TOKEN_EXPIRES"] = 604_800    # 7 days
    app.config["JWT_ALGORITHM"] = "HS256"
    return JWTManager(app)


def make_auth_blueprint(auth_svc: AuthService) -> Blueprint:
    """Factory that returns the auth blueprint with injected AuthService."""

    @auth_bp.route("/register", methods=["POST"])
    def register():
        """POST /api/v7/auth/register — Create a new user account.

        Body: { username, password, role? }
        Roles: analyst (default), quant, admin
        """
        body = request.get_json(silent=True) or {}
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", "")).strip()
        role = str(body.get("role", "analyst")).strip().lower()

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400
        if role not in ("analyst", "quant", "admin"):
            return jsonify({"error": "role must be analyst, quant, or admin"}), 400

        try:
            user = auth_svc.register(username, password, role)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 409

        access_token = create_access_token(identity=username)
        refresh_token = create_refresh_token(identity=username)

        return jsonify({
            "message": f"User '{username}' registered successfully.",
            "user": user,
            "access_token": access_token,
            "refresh_token": refresh_token,
        }), 201

    @auth_bp.route("/login", methods=["POST"])
    def login():
        """POST /api/v7/auth/login — Authenticate and receive JWT tokens.

        Body: { username, password }
        Returns: { access_token, refresh_token, user }
        """
        body = request.get_json(silent=True) or {}
        username = str(body.get("username", "")).strip()
        password = str(body.get("password", "")).strip()

        if not username or not password:
            return jsonify({"error": "username and password are required"}), 400

        user = auth_svc.authenticate(username, password)
        if user is None:
            # Intentionally vague — don't reveal whether username exists
            return jsonify({"error": "Invalid credentials"}), 401

        access_token = create_access_token(identity=username)
        refresh_token = create_refresh_token(identity=username)

        return jsonify({
            "message": "Login successful",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "Bearer",
            "expires_in": 86400,
            "user": user,
        }), 200

    @auth_bp.route("/refresh", methods=["POST"])
    @jwt_required(refresh=True)
    def refresh():
        """POST /api/v7/auth/refresh — Get new access token using refresh token.

        Header: Authorization: Bearer <refresh_token>
        """
        identity = get_jwt_identity()
        new_access = create_access_token(identity=identity)
        return jsonify({
            "access_token": new_access,
            "token_type": "Bearer",
            "expires_in": 86400,
        }), 200

    @auth_bp.route("/me", methods=["GET"])
    @jwt_required()
    def me():
        """GET /api/v7/auth/me — Get current user profile (requires token)."""
        identity = get_jwt_identity()
        user = auth_svc.get_user(identity)
        if user is None:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"user": user}), 200

    @auth_bp.route("/logout", methods=["POST"])
    @jwt_required()
    def logout():
        """POST /api/v7/auth/logout — Client-side logout (stateless).

        Engineering note: Stateless JWTs cannot be invalidated server-side
        without a blocklist. This endpoint exists for client compatibility —
        clients should discard the token on their side.
        To add server-side invalidation, implement a Redis-backed blocklist
        in Flask-JWT-Extended's token_in_blocklist_loader.
        """
        return jsonify({
            "message": "Logged out. Discard your token on the client side.",
            "note": "Stateless JWT — token remains valid until expiry unless blocklist is enabled."
        }), 200

    return auth_bp
