"""Gunicorn configuration for production (Linux / macOS only).

Usage:
    gunicorn -c gunicorn.conf.py wsgi:app

On Windows prefer waitress (a pure-Python, thread-pool WSGI server):
    waitress-serve --listen=127.0.0.1:5000 --threads=8 wsgi:app
"""
import multiprocessing
import os


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Bind address. Defaults to localhost for safety in production.
bind = os.environ.get("MLAGENT_WEB_BIND", "127.0.0.1:5000")

# One worker per CPU is a good starting point for CPU-bound ML work.
workers = _int_env("MLAGENT_WSGI_WORKERS", multiprocessing.cpu_count())
threads = _int_env("MLAGENT_WSGI_THREADS", 4)

# Handles concurrent requests per worker without additional processes.
worker_class = "gthread"
worker_connections = 1000

# Long-running (async) endpoints should return quickly, but some ready-heavy
# routes (e.g. one-shot training) can take a while; allow generous timeouts.
timeout = _int_env("MLAGENT_WSGI_TIMEOUT", 300)
graceful_timeout = 30

# Master/workers inherit the already-imported app (faster boot).
preload_app = True
max_requests = 1000
max_requests_jitter = 100

accesslog = "-"
errorlog = "-"
loglevel = os.environ.get("MLAGENT_LOG_LEVEL", "info").lower()
