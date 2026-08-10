"""app/middleware/__init__.py"""
from app.middleware.error_handlers import register_error_handlers  # noqa: F401
from app.middleware.observability import register_observability  # noqa: F401
