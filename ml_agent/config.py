"""Centralized, environment-driven configuration for the ML Agent.

Production readiness rule #1: never scatter tunables across modules. Every
hard-coded default lives here and can be overridden with an environment
variable, so the same codebase can run in dev, CI and production without code
changes.

All values are read lazily (on access) so tests / tooling can set ``os.environ``
between calls.
"""
from __future__ import annotations

import os
from typing import Callable, Optional


def _get_int(name: str, default: int) -> int:
    """Parse an integer env var, falling back to ``default`` on bad input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    """Parse a float env var, falling back to ``default`` on bad input."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _get_bool(name: str, default: bool) -> bool:
    """Parse a boolean env var (true/1/yes/on)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ============ Logging ============

LOG_LEVEL = _get_str("MLAGENT_LOG_LEVEL", "INFO")
LOG_FORMAT = _get_str(
    "MLAGENT_LOG_FORMAT",
    "%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
)


# ============ Database engine / pooling ============

# For multi-threaded / concurrent request servers (e.g. waitress/gunicorn) a
# single shared engine must pool connections. These control the pool bounds.
DB_POOL_SIZE = _get_int("MLAGENT_DB_POOL_SIZE", 10)
DB_MAX_OVERFLOW = _get_int("MLAGENT_DB_MAX_OVERFLOW", 20)
DB_POOL_RECYCLE = _get_int("MLAGENT_DB_POOL_RECYCLE", 1800)  # seconds
DB_POOL_TIMEOUT = _get_float("MLAGENT_DB_POOL_TIMEOUT", 30.0)  # seconds
DB_POOL_PRE_PING = _get_bool("MLAGENT_DB_POOL_PRE_PING", True)

# Fallback query timeout (seconds) applied when a caller does not pass one and
# the processor has no explicit timeout configured.
DB_QUERY_TIMEOUT: Optional[float] = _get_float("MLAGENT_DB_QUERY_TIMEOUT", 0.0) or None
if not DB_QUERY_TIMEOUT:
    DB_QUERY_TIMEOUT = None

# SQLite-specific: how long (seconds) to wait on a locked database file before
# giving up, and whether to allow cross-thread usage of a single connection.
SQLITE_BUSY_TIMEOUT = _get_int("MLAGENT_SQLITE_BUSY_TIMEOUT", 30_000)  # ms
SQLITE_CHECK_SAME_THREAD = _get_bool("MLAGENT_SQLITE_CHECK_SAME_THREAD", False)

# Connection socket timeouts (seconds) for network databases. Applied to
# PostgreSQL and MySQL. None leaves driver defaults.
DB_CONNECT_TIMEOUT: Optional[float] = _get_float("MLAGENT_DB_CONNECT_TIMEOUT", 0.0) or None
if not DB_CONNECT_TIMEOUT:
    DB_CONNECT_TIMEOUT = None


# ============ Web API ============

WEB_HOST = _get_str("MLAGENT_WEB_HOST", "127.0.0.1")
WEB_PORT = _get_int("MLAGENT_WEB_PORT", 5000)
WEB_MODELS_DIR = _get_str("MLAGENT_WEB_MODELS_DIR", "ml_agent_models")
WEB_MAX_CONTENT_LENGTH = _get_int("MLAGENT_WEB_MAX_CONTENT_LENGTH", 50 * 1024 * 1024)
WEB_MAX_NOTIFICATIONS = _get_int("MLAGENT_WEB_MAX_NOTIFICATIONS", 10)
WEB_TRUST_PROXY = _get_bool("MLAGENT_WEB_TRUST_PROXY", False)


def get_upload_dir() -> str:
    """Directory for database / model uploads (created on demand)."""
    return _get_str("MLAGENT_WEB_UPLOAD_DIR", os.path.join(_default_data_dir(), "uploads"))


def _default_data_dir() -> str:
    """Cross-platform data directory for runtime artifacts (uploads/stores)."""
    if not hasattr(_default_data_dir, "_cache"):
        base = os.environ.get("MLAGENT_DATA_DIR")
        if not base:
            home = os.path.expanduser("~")
            base = os.path.join(home, ".mlagent", "data")
        _default_data_dir._cache = base  # type: ignore[attr-defined]
    return _default_data_dir._cache  # type: ignore[attr-defined]


# ============ Training / scaling ============

TRAIN_DEFAULT_CV_FOLDS = _get_int("MLAGENT_TRAIN_CV_FOLDS", 5)
TRAIN_DEFAULT_N_JOBS = _get_int("MLAGENT_TRAIN_N_JOBS", 1)
# Early-stop: stop evaluating candidate models after N models without improvement.
TRAIN_EARLY_STOP_PATIENCE = _get_int("MLAGENT_TRAIN_EARLY_STOP_PATIENCE", 4)


# ============ WSGI / deployment ============

WSGI_THREADS = _get_int("MLAGENT_WSGI_THREADS", 8)
WSGI_WORKERS = _get_int("MLAGENT_WSGI_WORKERS", 1)