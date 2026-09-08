"""Shared session registry for stateless/horizontally-scaled deployments.

Stores per-session *metadata* (most importantly the DB connection string) in a
shared backing store so that:

- a stale worker that lost its in-memory session can re-hydrate it by
  re-connecting to the same database (survives restarts / worker recycling), and
- a multi-node LB can route/rebuild a session on any worker possessing the key.

Backends (in priority order):
  1. Redis  (``MLAGENT_REDIS_URL``)  — shared across processes/nodes.
  2. Local SQLite (``MLAGENT_SESSIONS_DB_PATH``) — durable, single-node.
In-memory dict used only if neither is available.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, Optional


class SessionStore:
    """Thread-safe session registry keyed by session id -> metadata dict."""

    def __init__(self, redis_url: Optional[str] = None,
                 sqlite_path: Optional[str] = None,
                 ttl: int = 0) -> None:
        self._ttl = ttl
        self._redis: Optional[Any] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._local: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        if redis_url:
            try:
                import redis  # type: ignore[import-untyped]

                self._redis = redis.from_url(
                    redis_url, socket_connect_timeout=1, socket_timeout=2
                )
                self._redis.ping()
            except Exception:
                self._redis = None

        if self._redis is None and sqlite_path:
            try:
                basedir = os.path.dirname(os.path.abspath(sqlite_path))
                os.makedirs(basedir, exist_ok=True)
                self._conn = sqlite3.connect(sqlite_path, check_same_thread=False)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS sessions ("
                    " id TEXT PRIMARY KEY, data TEXT, updated_at REAL)"
                )
                self._conn.commit()
            except Exception:
                self._conn = None

    @property
    def backend(self) -> str:
        if self._redis is not None:
            return "redis"
        if self._conn is not None:
            return "sqlite"
        return "memory"

    # ---- public API -------------------------------------------------------
    def put(self, session_id: str, data: Dict[str, Any]) -> None:
        data = dict(data)
        data.setdefault("created_at", time.time())
        data["last_seen"] = time.time()
        if self._redis is not None:
            key = f"session:{session_id}"
            try:
                self._redis.set(key, json.dumps(data))
                if self._ttl > 0:
                    self._redis.expire(key, self._ttl)
            except Exception:
                self._local[session_id] = dict(data)
            return
        if self._conn is not None:
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO sessions (id,data,updated_at) VALUES (?,?,?)",
                    (session_id, json.dumps(data), time.time()),
                )
                self._conn.commit()
                self._local[session_id] = dict(data)  # cheap in-memory cache too
            return
        self._local[session_id] = dict(data)

    def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._redis is not None:
            key = f"session:{session_id}"
            try:
                raw = self._redis.get(key)
                if raw:
                    data = json.loads(raw)
                    return self._expired(data) or data
            except Exception:
                pass
        if self._conn is not None:
            with self._lock:
                if session_id in self._local:
                    d = self._local[session_id]
                    return None if self._expired(d) else d
                row = self._conn.execute(
                    "SELECT data FROM sessions WHERE id=?", (session_id,)
                ).fetchone()
            if row:
                try:
                    d = json.loads(row[0])
                    if self._expired(d):
                        return None
                    self._local[session_id] = d
                    return d
                except Exception:
                    return None
            return None
        d = self._local.get(session_id)
        return None if self._expired(d) else d

    def delete(self, session_id: str) -> None:
        self._local.pop(session_id, None)
        if self._redis is not None:
            try:
                self._redis.delete(f"session:{session_id}")
                return
            except Exception:
                pass
        if self._conn is not None:
            with self._lock:
                self._conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
                self._conn.commit()

    def touch(self, session_id: str) -> None:
        d = self.get(session_id)
        if d is not None:
            d["last_seen"] = time.time()
            self.put(session_id, d)

    def close(self) -> None:
        self._local.clear()
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ---- internals ---------------------------------------------------------
    def _expired(self, data: Optional[Dict[str, Any]]) -> bool:
        if data is None or self._ttl <= 0:
            return False
        last = data.get("last_seen") or data.get("created_at") or 0
        return (time.time() - last) > self._ttl