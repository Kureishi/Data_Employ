"""Thread-safe, SQLite-backed job store.

Background jobs (training / op jobs) previously lived only in in-memory dicts
and vanished on restart, orphaning every client. This store write-throughs each
job transition to SQLite (WAL mode) so jobs survive a process restart; a
startup routine marks any job still ``running`` as ``interrupted``.

The job store is optional: if no path is configured (``MLAGENT_JOBS_DB_PATH``)
callers should use an in-memory fallback (the API keeps a live dict anyway).
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    operation  TEXT,
    status     TEXT NOT NULL,
    params     TEXT,
    result     TEXT,
    progress   TEXT,
    error      TEXT,
    created_at REAL,
    updated_at REAL
);
"""

_ROW_COLUMNS = ("id", "kind", "operation", "status", "params", "result",
                "progress", "error", "created_at", "updated_at")


def _js(x: Any) -> Optional[str]:
    return json.dumps(x, default=str) if x is not None else None


class JobStore:
    """Minimal durable, thread-safe job registry backed by SQLite."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = threading.Lock()
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            with self._lock:
                self._conn.execute(_SCHEMA)
                self._conn.commit()
                # Migrate older DBs that predate the `progress` column.
                cols = [r[1] for r in self._conn.execute("PRAGMA table_info(jobs)")]
                if cols and "progress" not in cols:
                    self._conn.execute(
                        "ALTER TABLE jobs ADD COLUMN progress TEXT"
                    )
                    self._conn.commit()

    @property
    def enabled(self) -> bool:
        return self._conn is not None

    def put(self, job_id: str, kind: str, operation: Optional[str],
            status: str, params: Optional[Dict[str, Any]] = None,
            result: Optional[Any] = None, error: Optional[str] = None,
            created_at: Optional[float] = None) -> None:
        now = created_at or time.time()
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs (id,kind,operation,status,params,"
                "result,error,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (job_id, kind, operation, status, _js(params), _js(result),
                 error, now, now),
            )
            self._conn.commit()

    def update(self, job_id: str, **fields: Any) -> None:
        if not self._conn or not fields:
            return
        fields = dict(fields)
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        values: List[Any] = []
        for k in fields:
            v = fields[k]
            # JSON-serialize anything that isn't a scalar so sqlite can bind it;
            # keep the schema column ("params"/"result"/"progress"...) intact.
            if k in ("params", "result", "progress"):
                values.append(_js(v))
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                values.append(_js(v))
            else:
                values.append(v)
        values.append(job_id)
        with self._lock:
            self._conn.execute(
                f"UPDATE jobs SET {cols} WHERE id=?", tuple(values)
            )
            self._conn.commit()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        if not self._conn:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def list(self, kind: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._conn:
            return []
        with self._lock:
            if kind:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE kind=? ORDER BY created_at DESC",
                    (kind,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC"
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def mark_stale_interrupted(self) -> None:
        """Mark any job still running/cancelling as interrupted (we crashed)."""
        if not self._conn:
            return
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status='interrupted', updated_at=? "
                "WHERE status IN ('running','cancelling')",
                (time.time(),),
            )
            self._conn.commit()

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    @staticmethod
    def _row_to_dict(row: Any) -> Dict[str, Any]:
        d = dict(zip(_ROW_COLUMNS, row))
        for k in ("params", "result", "progress"):
            try:
                d[k] = json.loads(d[k]) if d[k] else None
            except Exception:
                d[k] = None
        return d