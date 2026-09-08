"""Rate limiting with a shared Redis backend (with in-process fallback).

Under a single process the default in-memory token bucket is sufficient. Under
multi-process gunicorn/Celery you want a single shared counter, so this module
uses a Redis sorted-set (``ZREMRANGEBYSCORE`` + ``ZCARD``) keyed by a fixed
window, and expires each key automatically.

If Redis is not installed or ``MLAGENT_REDIS_URL`` is unset it transparently
falls back to a thread-safe in-process fixed-window limiter (same semantics,
per-process scope).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


class RateLimiter:
    """Fixed-window per-key rate limiter (Redis-backed when available)."""

    def __init__(self, redis_url: Optional[str] = None) -> None:
        self._redis: Optional[Any] = None
        if redis_url:
            try:
                import redis  # type: ignore[import-untyped]

                self._redis = redis.from_url(
                    redis_url, socket_connect_timeout=1, socket_timeout=2
                )
                # Verify connectivity so we don't silently fail under load;
                # on error fall back to the in-process limiter.
                self._redis.ping()
            except Exception:
                self._redis = None

        self._lock = threading.Lock()
        self._buckets: Dict[str, Deque[float]] = {}
        self._strip: Deque[str] = deque()

    @property
    def backend(self) -> str:
        return "redis" if self._redis is not None else "local"

    def allow(self, key: str, limit: int, window: float = 60.0) -> bool:
        """Return True if the call is allowed, False if rate-limited."""
        if limit <= 0:
            return True
        if self._redis is not None:
            return self._allow_redis(key, limit, window)
        return self._allow_local(key, limit, window)

    def _allow_redis(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        rk = f"rl:{key}"
        try:
            pipe = self._redis.pipeline()
            pipe.zremrangebyscore(rk, 0, now - window)
            pipe.zcard(rk)
            count = pipe.execute()[-1]
            if int(count) >= limit:
                return False
            pipe = self._redis.pipeline()
            pipe.zadd(rk, {f"{now:.6f}-{key}": now})
            pipe.expire(rk, int(window) + 5)
            pipe.execute()
            return True
        except Exception:
            # Redis hiccup: degrade gracefully to the local limiter.
            return self._allow_local(key, limit, window)

    def _allow_local(self, key: str, limit: int, window: float) -> bool:
        now = time.time()
        with self._lock:
            dq = self._buckets.get(key)
            if dq is None:
                dq = deque()
                self._buckets[key] = dq
                self._strip.append(key)
            # Bound memory: drop oldest keys beyond the cache ceiling.
            while self._strip and len(self._buckets) > 10_000:
                self._buckets.pop(self._strip.popleft(), None)
            while dq and now - dq[0] >= window:
                dq.popleft()
            if len(dq) >= limit:
                return False
            dq.append(now)
            return True