"""
ML Agent Web API - Flask backend exposing the ML Agent as a REST API.

Run:
    python web_api.py --db sample_company.db
    python web_api.py --db sqlite:///sample_company.db --port 8080

Then open http://localhost:5000 in your browser.
"""
import argparse
import logging
import os
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file, Response
from werkzeug.utils import secure_filename

from ml_agent import MLAgent
from ml_agent import config as cfg
from ml_agent.job_store import JobStore
from ml_agent.logging_utils import configure_logging

log = logging.getLogger("ml_agent.web_api")

# ============ App setup ============

app = Flask(
    __name__,
    static_folder="web/static",
    template_folder="web/templates",
)

# Cap request body size so a single client can't exhaust memory (scalability + DoS).
app.config["MAX_CONTENT_LENGTH"] = cfg.WEB_MAX_CONTENT_LENGTH

# When behind a load balancer / reverse proxy, honor X-Forwarded-* headers so
# request.client and url_for produce the correct external scheme/host.
if cfg.WEB_TRUST_PROXY:
    from werkzeug.middleware.proxy_fix import ProxyFix

    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Global agent state. A single agent holds mutable state (current_df,
# model_selector, ...). AGENT_LOCK serializes *state-mutating* operations from
# background threads (connect/disconnect/train/op jobs). Pure reads stay
# lock-free. AGENT is the backward-compatible *default* session; named sessions
# (Tier 3/C3) are stored in SESSIONS keyed by session id.
AGENT: Optional[MLAgent] = None
AGENT_LOCK = threading.RLock()

# ---- Session isolation (C3) -------------------------------------------------
# Clients may pass an X-Session-Id header to get an isolated agent per session
# (enables per-session / horizontally-scaled workflows). The default session is
# the legacy `AGENT` global so existing clients keep working.
SESSIONS: Dict[str, MLAgent] = {}
SESSION_LOCK = threading.RLock()
DEFAULT_SESSION_ID = "default"


def _current_session_id() -> str:
    sid = request.headers.get("X-Session-Id", "").strip()
    return sid if sid else DEFAULT_SESSION_ID


# ---- Async job subsystem (A3 bounded, A5 durable) ---------------------------
# Operation jobs run on a bounded thread pool (instead of one raw daemon thread
# per job) so a flood of requests cannot spawn unbounded threads. A semaphore
# enforces a hard cap; submissions beyond it receive 429.
JOB_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, cfg.MAX_CONCURRENT_JOBS))
JOB_SEM = threading.BoundedSemaphore(max(1, cfg.MAX_CONCURRENT_JOBS))

# Durable job store (SQLite) so jobs survive restarts. Optional; in-memory only
# when MLAGENT_JOBS_DB_PATH is unset.
JOBS_DB = JobStore(cfg.get_jobs_db_path())
JOBS_DB.mark_stale_interrupted()

# Async training-job registry (id -> job dict)
TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}
TRAIN_LOCK = threading.Lock()

# Generic async operation-job registry (analyze/synthesize/monitor/anomaly/...)
OP_JOBS: Dict[str, Dict[str, Any]] = {}
OP_LOCK = threading.Lock()
NOTIFICATIONS: List[Dict[str, Any]] = []
MAX_NOTIFICATIONS = cfg.WEB_MAX_NOTIFICATIONS

# Graceful-shutdown flag (C2): when set, new jobs/requests are refused and the
# server drains in-flight work before exiting.
RUNNING = True
SHUTDOWN_LOCK = threading.Lock()

# Lightweight request metrics for /status (C1).
METRICS_LOCK = threading.Lock()
METRICS = {"requests": 0, "errors": 0, "by_path": {}, "started_at": time.time()}

# Upload directory for database / model files (configurable via env for prod).
UPLOAD_DIR = cfg.get_upload_dir()
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_DB_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}
ALLOWED_MODEL_EXTENSIONS = {".joblib", ".pkl", ".pickle"}

# ============ Helpers ============


def _get_agent(session_id: Optional[str] = None) -> MLAgent:
    """Return the agent for the current session (or a named one).

    When ``session_id`` is omitted the request's ``X-Session-Id`` header is
    used; the default session maps to the legacy global ``AGENT`` so existing
    clients keep working unchanged.
    """
    global AGENT
    sid = session_id or _current_session_id()
    if sid == DEFAULT_SESSION_ID:
        if AGENT is None:
            raise RuntimeError("No database connected. POST /api/connect first.")
        return AGENT
    with SESSION_LOCK:
        agent = SESSIONS.get(sid)
    if agent is None:
        raise RuntimeError(
            "Unknown or expired session. POST /api/connect with an "
            "X-Session-Id header to create one."
        )
    return agent


def _make_agent(connection: str) -> MLAgent:
    """Build, connect and configure a new MLAgent (used on connect/upload)."""
    agent = MLAgent(connection)
    _init_agent_stores(agent)
    # Persistent (disk-backed) analysis cache, shared across workers (B3).
    agent.enable_disk_cache()
    return agent


def _connect_session(session_id: str, connection: str) -> MLAgent:
    """Create (or replace) the agent for a session and connect it to the DB."""
    global AGENT
    if session_id == DEFAULT_SESSION_ID:
        if AGENT is not None:
            try:
                AGENT.close()
            except Exception:
                pass
        AGENT = _make_agent(connection)
        return AGENT
    with SESSION_LOCK:
        existing = SESSIONS.pop(session_id, None)
    if existing is not None:
        try:
            existing.close()
        except Exception:
            pass
    agent = _make_agent(connection)
    with SESSION_LOCK:
        SESSIONS[session_id] = agent
    return agent


def _close_session(session_id: str) -> None:
    global AGENT
    if session_id == DEFAULT_SESSION_ID:
        if AGENT is not None:
            try:
                AGENT.close()
            except Exception:
                pass
        AGENT = None
        return
    with SESSION_LOCK:
        agent = SESSIONS.pop(session_id, None)
    if agent is not None:
        try:
            agent.close()
        except Exception:
            pass


def _init_agent_stores(agent: MLAgent) -> None:
    """Set up per-database persistent stores for saved queries and profiles."""
    try:
        cs = agent.db.connection_string or "db"
        slug = "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in cs)[:60]
        if not slug:
            slug = "db"
        store_dir = os.path.join(UPLOAD_DIR, "stores")
        os.makedirs(store_dir, exist_ok=True)
        agent.db.set_query_store(os.path.join(store_dir, f"{slug}_queries.json"))
        agent.db.set_profile_store(os.path.join(store_dir, f"{slug}_profiles.json"))
        agent.experiments.set_store(os.path.join(store_dir, f"{slug}_experiments.json"))
        agent.recipes.set_store(os.path.join(store_dir, f"{slug}_recipes.json"))
        agent.snapshots.set_store(os.path.join(store_dir, f"{slug}_snapshots.json"))
        agent.models_dir = os.path.join(store_dir, "models")
    except Exception:
        pass


def _jsonable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable representation."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float):
            # Handle NaN / infinity
            if pd.isna(obj):
                return None
        return obj
    return str(obj)


def _error(message: str, status: int = 400) -> tuple:
    return jsonify({"error": str(message)}), status


def _success(**kwargs) -> Response:
    return jsonify({"success": True, **kwargs})


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 100) -> str:
    """Convert a DataFrame to an HTML table string."""
    return df.head(max_rows).to_html(classes="display nowrap", index=False)


# ============ Auth + rate limiting + metrics + health (Tier 2) ============


def _is_rate_limited(key: str, limit_per_min: int) -> bool:
    """Fixed-window in-memory rate limiter (per key). Returns True if the
    caller exceeded the per-minute allowance for that key."""
    if limit_per_min <= 0:
        return False
    now = time.time()
    window = 60.0
    with METRICS_LOCK:
        buckets = getattr(_is_rate_limited, "_buckets", {})
        strip = getattr(_is_rate_limited, "_strip", deque())
        dq = buckets.get(key)
        if dq is None:
            dq = deque()
            buckets[key] = dq
            strip.append(key)
        # drop stale keys to avoid unbounded memory growth
        while strip and len(buckets) > 10_000:
            stale = strip.popleft()
            buckets.pop(stale, None)
        while dq and now - dq[0] >= window:
            dq.popleft()
        if len(dq) >= limit_per_min:
            _is_rate_limited._buckets = buckets
            _is_rate_limited._strip = strip
            return True
        dq.append(now)
        _is_rate_limited._buckets = buckets
        _is_rate_limited._strip = strip
        return False


def _client_key() -> str:
    return request.remote_addr or "unknown"


@app.before_request
def _apply_auth_and_limits() -> Optional[Response]:
    # Liveness probes and static assets bypass auth/limits.
    if request.path.startswith("/health") or request.path == "/":
        return None
    # Optional shared bearer token (D1).
    if cfg.API_TOKEN:
        auth = request.headers.get("Authorization", "")
        supplied = auth[7:] if auth.startswith("Bearer ") else request.headers.get("X-API-Token", "")
        if supplied != cfg.API_TOKEN:
            return _error("Unauthorized.", status=401)
    # Rate-limit mutating API calls (POST/PUT/DELETE) per client IP.
    if request.method in ("POST", "PUT", "DELETE") and cfg.RATE_LIMIT_PER_MINUTE > 0:
        if _is_rate_limited(f"{_client_key()}:{request.method}:{request.path}", cfg.RATE_LIMIT_PER_MINUTE):
            return _error("Rate limit exceeded. Slow down and retry later.", status=429)
    return None


@app.before_request
def _metrics_and_shutdown() -> Optional[Response]:
    if not RUNNING and not request.path.startswith("/health"):
        return _error("Server is shutting down. No new requests.", status=503)
    return None


@app.route("/health", methods=["GET"])
def health() -> Response:
    """Liveness probe for load balancers / orchestrators."""
    return jsonify({"status": "ok", "time": time.time()})


@app.route("/health/ready", methods=["GET"])
def health_ready() -> Response:
    """Readiness probe: 200 when connected + (optionally) a model is loaded."""
    try:
        agent = _get_agent()
        ready = agent.db.engine is not None
        body: Dict[str, Any] = {"ready": ready, "connected": True}
        if ready:
            body["tables"] = agent.list_tables()
        code = 200 if ready else 503
        return jsonify(body), code
    except Exception:
        return jsonify({"ready": False, "connected": False}), 503


@app.route("/status", methods=["GET"])
def status() -> Response:
    """Operational metrics: request counters, active jobs, session count."""
    with METRICS_LOCK:
        metrics = {
            "requests": METRICS["requests"],
            "errors": METRICS["errors"],
            "uptime_s": round(time.time() - METRICS["started_at"], 1),
            "by_path": dict(sorted(METRICS["by_path"].items(), key=lambda kv: -kv[1])[:20]),
        }
    with OP_LOCK:
        active = sum(1 for j in OP_JOBS.values() if j.get("status") == "running")
    with SESSION_LOCK:
        session_count = len(SESSIONS) + (1 if AGENT is not None else 0)
    return jsonify({
        "ok": True,
        "metrics": metrics,
        "active_op_jobs": active,
        "sessions": session_count,
        "concurrency_limit": cfg.MAX_CONCURRENT_JOBS,
        "shutting_down": not RUNNING,
    })


# ============ Async operation jobs + notifications (Tier 2: async everything) ============


def _add_notification(message: str, ntype: str = "info", detail: str = "") -> None:
    with OP_LOCK:
        NOTIFICATIONS.append({
            "message": message,
            "type": ntype,
            "detail": detail,
            "time": time.time(),
        })
        if len(NOTIFICATIONS) > MAX_NOTIFICATIONS:
            del NOTIFICATIONS[: len(NOTIFICATIONS) - MAX_NOTIFICATIONS]


def _get_op_handler(op_name: str):
    return OP_HANDLERS.get(op_name)


class JobBusyError(Exception):
    """Raised when the worker pool is saturated (returns HTTP 429)."""


def _start_op_job(op_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Launch a bounded, durable async operation job.

    Returns the job record. Raises JobBusyError when the concurrency cap is hit.
    """
    agent = _get_agent()
    handler = _get_op_handler(op_name)
    if handler is None:
        raise ValueError(f"Unknown operation: {op_name}")

    # A3: refuse new submissions once the worker pool is saturated.
    if not JOB_SEM.acquire(blocking=False):
        raise JobBusyError(
            f"Too many background jobs running (limit {cfg.MAX_CONCURRENT_JOBS}). "
            "Wait for one to finish and retry."
        )

    job_id = f"op_{uuid.uuid4().hex[:8]}"  # A1: collision-safe id
    started_at = time.time()
    job: Dict[str, Any] = {
        "id": job_id,
        "operation": op_name,
        "status": "running",
        "result": None,
        "error": None,
        "started_at": started_at,
    }
    # A5: persist so the job survives a restart and is inspectable.
    JOBS_DB.put(job_id, "op", op_name, "running", params=params, created_at=started_at)

    def worker() -> None:
        try:
            with AGENT_LOCK:
                result = handler(agent, params)
            # Enforce a soft wall-clock ceiling: report timeout if exceeded.
            if time.time() - started_at > cfg.JOB_MAX_WALLCLOCK:
                job["status"] = "timed_out"
                job["error"] = f"Job exceeded {cfg.JOB_MAX_WALLCLOCK}s wall-clock limit."
                JOBS_DB.update(job_id, status="timed_out", error=job["error"])
                return
            with OP_LOCK:
                job["result"] = result
            job["status"] = "done"
            JOBS_DB.update(job_id, status="done", result=result)
            _add_notification(f"{op_name} completed", "success", op_name)
        except Exception as e:
            job["status"] = "error"
            job["error"] = str(e)
            JOBS_DB.update(job_id, status="error", error=str(e))
            _add_notification(f"{op_name} failed: {e}", "error", op_name)
        finally:
            JOB_SEM.release()

    with OP_LOCK:
        OP_JOBS[job_id] = job
    JOB_EXECUTOR.submit(worker)
    return job


# Handlers: (agent, params) -> JSON-serializable result dict.
def _op_analyze(agent, params):
    return {"analysis": _jsonable(agent.analyze(
        params.get("target_column"), params.get("type", "summary")
    ))}


def _op_anomaly(agent, params):
    result = agent.detect_anomalies(
        table=params.get("table") or None,
        method=params.get("method", "isolation_forest"),
        contamination=float(params.get("contamination", 0.1)),
    )
    return {"anomaly": _jsonable(result)}


def _op_synthesize(agent, params):
    base = params.get("table")
    if not base:
        raise ValueError("table is required.")
    agent.synthesize_features(
        base,
        include_counts=bool(params.get("include_counts", True)),
        include_aggregates=bool(params.get("include_aggregates", True)),
    )
    summary = agent.get_feature_synthesizer_summary()
    return {
        "summary": summary,
        "columns": list(agent.current_df.columns),
        "rows": int(len(agent.current_df)),
        "data": _jsonable(agent.current_df.head(100)),
    }


def _op_monitor_capture(agent, params):
    result = agent.capture_monitor_reference()
    return {"rows": len(agent.current_df) if agent.current_df is not None else 0,
            "features": result.get("features", [])}


def _op_monitor_check(agent, params):
    table = params.get("table")
    if not table:
        raise ValueError("table is required.")
    return {"drift": _jsonable(agent.monitor_live(table, feature_columns=params.get("features")))}


def _op_repredict(agent, params):
    table = params.get("table")
    if not table:
        raise ValueError("table is required.")
    res = agent.repredict_table(
        table, limit=params.get("limit"), contamination=float(params.get("contamination", 0.05))
    )
    return {
        "predictions": _jsonable(res["predictions"]),
        "drift": _jsonable(res["drift"]),
        "anomaly": res["anomaly"],
        "rows": res["rows"],
        "columns": list(res["predictions"].columns),
    }


def _op_recipe_apply(agent, params):
    name = params.get("name")
    if not name:
        raise ValueError("recipe name is required.")
    results = agent.apply_recipe(
        name,
        target=params.get("target"),
        tuning=params.get("tuning"),
        n_jobs=params.get("n_jobs"),
    )
    return {"training": _serialize_training(results), "recipe": name}


def _op_batch_whatif(agent, params):
    feature = params.get("feature")
    if not feature:
        raise ValueError("feature is required.")
    result = agent.batch_what_if(feature, params.get("value"))
    return {"batch_whatif": _jsonable(result)}


OP_HANDLERS: Dict[str, Any] = {
    "analyze": _op_analyze,
    "anomaly": _op_anomaly,
    "synthesize": _op_synthesize,
    "monitor_capture": _op_monitor_capture,
    "monitor_check": _op_monitor_check,
    "repredict": _op_repredict,
    "recipe_apply": _op_recipe_apply,
    "batch_whatif": _op_batch_whatif,
}


# ============ Routes ============


@app.route("/", methods=["GET"])
def index():
    """Serve the web UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/state", methods=["GET"])
def api_state():
    """Get the current agent state summary."""
    try:
        agent = _get_agent()
        state: Dict[str, Any] = {"connected": True}

        if agent.db.engine is not None:
            state["tables"] = agent.list_tables()

        if agent.current_df is not None:
            state["loaded_data"] = {
                "shape": [agent.current_df.shape[0], agent.current_df.shape[1]],
                "columns": list(agent.current_df.columns),
            }

        state["target_column"] = agent.target_column

        if agent.predictor is not None:
            state["model"] = {
                "task_type": agent.predictor.model.task_type,
                "best_model": agent.predictor.model.best_model_name,
                "best_cv_score": agent.predictor.model.best_score,
                "test_metrics": _jsonable(agent.predictor.model.metrics),
                "feature_columns": agent.predictor.model.feature_columns,
            }

        if agent.llm is not None:
            state["llm"] = agent.llm_check()

        return jsonify(state)
    except RuntimeError:
        return jsonify(_success(connected=False))


@app.route("/api/connect", methods=["POST"])
def api_connect():
    """Connect to a database. With an X-Session-Id header (or body session_id),
    the agent is created in its own isolated session (C3). Otherwise the
    default (legacy) global agent is used."""
    global AGENT
    data = request.get_json(silent=True) or {}
    connection = data.get("connection") or data.get("db")
    if not connection:
        return _error("Connection string or SQLite file path is required.")
    session_id = (data.get("session_id") or _current_session_id()).strip()

    # Default session: reuse an existing agent already on this connection.
    if session_id == DEFAULT_SESSION_ID and AGENT is not None:
        try:
            existing = _get_agent()
            if existing.db.connection_string:
                existing_cs = existing.db.connection_string.replace("sqlite:///", "")
                if existing_cs == connection or existing.db.connection_string == connection:
                    return _success(
                        connected=True,
                        session_id=session_id,
                        tables=existing.list_tables(),
                        message="Already connected.",
                    )
        except Exception:
            pass

    try:
        agent = _connect_session(session_id, connection)
        tables = agent.list_tables()
        return _success(
            connected=True,
            session_id=session_id,
            tables=tables,
            message=f"Connected to {connection}",
        )
    except Exception as e:
        _close_session(session_id)
        return _error(f"Failed to connect: {e}")


@app.route("/api/upload-db", methods=["POST"])
def api_upload_db():
    """Upload a database file and connect (optionally to a named session)."""
    global AGENT
    if "file" not in request.files:
        return _error("No file uploaded. Use multipart/form-data with a 'file' field.")

    file = request.files["file"]
    if file.filename == "":
        return _error("No file selected.")

    # Validate extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_DB_EXTENSIONS:
        return _error(
            f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_DB_EXTENSIONS))}"
        )

    # Save the uploaded file to a safe location
    filename = secure_filename(file.filename)
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    session_id = (request.form.get("session_id") or _current_session_id()).strip()
    try:
        agent = _connect_session(session_id, save_path)
        tables = agent.list_tables()
        return _success(
            connected=True,
            session_id=session_id,
            tables=tables,
            message=f"Connected to uploaded database: {filename}",
            path=save_path,
        )
    except Exception as e:
        _close_session(session_id)
        return _error(f"Failed to connect to uploaded database: {e}")


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    """Disconnect from the database (the current session's agent)."""
    _close_session(_current_session_id())
    return _success(message="Disconnected.")


# ========== Tables ==========


@app.route("/api/tables", methods=["GET"])
def api_tables():
    try:
        agent = _get_agent()
        return _success(tables=agent.list_tables())
    except Exception as e:
        return _error(str(e))


@app.route("/api/tables/<table>/schema", methods=["GET"])
def api_table_schema(table: str):
    try:
        agent = _get_agent()
        schema = agent.db.get_schema(table)
        return _success(table=table, columns=schema)
    except Exception as e:
        return _error(str(e))


@app.route("/api/tables/<table>/preview", methods=["GET"])
def api_table_preview(table: str):
    try:
        agent = _get_agent()
        limit = request.args.get("limit", default=50, type=int)
        df = agent.db.load_table(table, limit=limit)
        return _success(
            table=table,
            rows=len(df),
            columns=[{"name": c} for c in df.columns],
            data=_jsonable(df),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/overview", methods=["GET"])
def api_overview():
    try:
        agent = _get_agent()
        return _success(overview=agent.get_database_overview())
    except Exception as e:
        return _error(str(e))


@app.route("/api/load", methods=["POST"])
def api_load():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        limit = data.get("limit")

        if not table:
            return _error("Table name is required.")

        df = agent.load_table(table, limit=limit)
        return _success(
            loaded=True,
            table=table,
            rows=len(df),
            columns=list(df.columns),
            data=_jsonable(df.head(100)),
            preview=_df_to_html_table(df),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/query", methods=["POST"])
def api_query():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        if not query:
            return _error("SQL query is required.")
        df = agent.execute_query(query)
        return _success(
            query=query,
            rows=len(df),
            columns=list(df.columns),
            data=_jsonable(df.head(100)),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/load-query", methods=["POST"])
def api_load_query():
    """Load a custom SQL query as the current working dataset."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        if not query:
            return _error("SQL query is required.")

        agent.load_query_as_data(query)
        return _success(
            loaded=True,
            source=f"query: {query}",
            rows=len(agent.current_df),
            columns=list(agent.current_df.columns),
            data=_jsonable(agent.current_df.head(100)),
            preview=_df_to_html_table(agent.current_df),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/query/validate", methods=["POST"])
def api_query_validate():
    """Validate a SQL statement (schema-aware, read-only) without running it."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        query = data.get("query")
        if not query:
            return _error("SQL query is required.")
        return _success(validation=agent.validate_sql(query))
    except Exception as e:
        return _error(str(e))


# ========== SQL-specific features (column typing / sampling / joins) ==========


@app.route("/api/columns/<table>/types", methods=["GET"])
def api_column_types(table: str):
    try:
        agent = _get_agent()
        types = agent.get_column_types(table)
        return _success(table=table, columns=types)
    except Exception as e:
        return _error(str(e))


@app.route("/api/relationships", methods=["GET"])
def api_relationships():
    try:
        agent = _get_agent()
        return _success(relationships=agent.get_relationships())
    except Exception as e:
        return _error(str(e))


@app.route("/api/load-sample", methods=["POST"])
def api_load_sample():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        if not table:
            return _error("table is required.")
        df = agent.load_table_sample(
            table,
            fraction=data.get("fraction", 0.1),
            columns=data.get("columns"),
            limit=data.get("limit"),
            method=data.get("method", "auto"),
        )
        return _success(loaded=True, table=table, rows=len(df),
                        columns=list(df.columns), data=_jsonable(df.head(100)))
    except Exception as e:
        return _error(str(e))


@app.route("/api/load-join", methods=["POST"])
def api_load_join():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        tables = data.get("tables")
        if not tables or not isinstance(tables, list):
            return _error("tables (list) is required.")
        df = agent.load_auto_join(tables, data.get("join_type", "inner"))
        return _success(loaded=True, tables=tables, rows=len(df),
                        columns=list(df.columns), data=_jsonable(df.head(100)))
    except Exception as e:
        return _error(str(e))


@app.route("/api/query/build-join", methods=["POST"])
def api_build_join():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        tables = data.get("tables")
        if not tables or not isinstance(tables, list):
            return _error("tables (list) is required.")
        built = agent.db.build_join_query(tables, data.get("join_type", "inner"))
        return _success(query=built["query"], joins=built["joins"])
    except Exception as e:
        return _error(str(e))


# ========== Saved query library (Feature 2) ==========


@app.route("/api/query/save", methods=["POST"])
def api_query_save():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        query = data.get("query")
        if not name or not query:
            return _error("Both 'name' and 'query' are required.")
        key = agent.save_query(name, query, data.get("description", ""))
        return _success(saved=True, name=key)
    except Exception as e:
        return _error(str(e))


@app.route("/api/query/list", methods=["GET"])
def api_query_list():
    try:
        agent = _get_agent()
        return _success(queries=agent.list_queries())
    except Exception as e:
        return _error(str(e))


@app.route("/api/query/get/<name>", methods=["GET"])
def api_query_get(name: str):
    try:
        agent = _get_agent()
        q = agent.get_query(name)
        if q is None:
            return _error("Query not found.", status=404)
        return _success(query=q)
    except Exception as e:
        return _error(str(e))


@app.route("/api/query/delete/<name>", methods=["POST"])
def api_query_delete(name: str):
    try:
        agent = _get_agent()
        deleted = agent.delete_query(name)
        if not deleted:
            return _error("Query not found.", status=404)
        return _success(deleted=True)
    except Exception as e:
        return _error(str(e))


# ========== Schema drift profiling (Feature 7) ==========


@app.route("/api/profile/capture", methods=["POST"])
def api_profile_capture():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        profile = agent.capture_profile(table=data.get("table"), name=data.get("name", ""))
        return _success(profile=profile)
    except Exception as e:
        return _error(str(e))


@app.route("/api/profile/list", methods=["GET"])
def api_profile_list():
    try:
        agent = _get_agent()
        return _success(profiles=agent.list_profiles())
    except Exception as e:
        return _error(str(e))


@app.route("/api/profile/compare", methods=["POST"])
def api_profile_compare():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        a, b = data.get("a"), data.get("b")
        if not a or not b:
            return _error("Both 'a' and 'b' profile names are required.")
        diff = agent.compare_profiles(a, b)
        return _success(diff=diff)
    except Exception as e:
        return _error(str(e))


# ========== Reusable pipeline recipes ==========


@app.route("/api/recipe/save", methods=["POST"])
def api_recipe_save():
    """Save the current data source + preprocessing pipeline as a named recipe."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        if not name:
            return _error("name is required.")
        saved = agent.save_recipe(
            name=name,
            description=data.get("description", ""),
            target=data.get("target_column"),
            task_type=data.get("task_type"),
            tuning=data.get("tuning"),
            n_jobs=data.get("n_jobs"),
            auto_prepare=bool(data.get("auto_prepare", False)),
        )
        return _success(saved=True, name=saved)
    except Exception as e:
        return _error(str(e))


@app.route("/api/recipe/list", methods=["GET"])
def api_recipe_list():
    try:
        agent = _get_agent()
        return _success(recipes=agent.list_recipes())
    except Exception as e:
        return _error(str(e))


@app.route("/api/recipe/get/<name>", methods=["GET"])
def api_recipe_get(name: str):
    try:
        agent = _get_agent()
        r = agent.get_recipe(name)
        if r is None:
            return _error("Recipe not found.", status=404)
        return _success(recipe=r)
    except Exception as e:
        return _error(str(e))


@app.route("/api/recipe/delete/<name>", methods=["POST"])
def api_recipe_delete(name: str):
    try:
        agent = _get_agent()
        ok = agent.delete_recipe(name)
        if not ok:
            return _error("Recipe not found.", status=404)
        return _success(deleted=True)
    except Exception as e:
        return _error(str(e))


@app.route("/api/recipe/apply", methods=["POST"])
def api_recipe_apply():
    """Reproduce a saved recipe synchronously (train a fresh model)."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        name = data.get("name")
        if not name:
            return _error("name is required.")
        results = agent.apply_recipe(
            name,
            target=data.get("target_column"),
            tuning=data.get("tuning"),
            n_jobs=data.get("n_jobs"),
        )
        return _success(training=_serialize_training(results), recipe=name)
    except Exception as e:
        return _error(str(e))


# ============ Result snapshots & diff (Feature: snapshot) ============


@app.route("/api/snapshots", methods=["GET"])
def api_snapshots_list():
    """List saved result snapshots."""
    try:
        agent = _get_agent()
        snaps = agent.list_snapshots()
        return _success(snapshots=snaps)
    except Exception as e:
        return _error(str(e))


@app.route("/api/snapshots", methods=["POST"])
def api_snapshots_save():
    """Save a JSON-serializable result snapshot under a name."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        name = agent.save_snapshot(
            data.get("name") or "",
            data.get("kind") or "manual",
            data.get("payload") or {},
            meta=data.get("meta") or {},
        )
        return _success(name=name)
    except Exception as e:
        return _error(str(e))


@app.route("/api/snapshots/<name>", methods=["GET"])
def api_snapshots_get(name: str):
    try:
        agent = _get_agent()
        snap = agent.get_snapshot(name)
        if snap is None:
            return _error("Snapshot not found.", status=404)
        return _success(snapshot=snap)
    except Exception as e:
        return _error(str(e))


@app.route("/api/snapshots/<name>", methods=["DELETE"])
def api_snapshots_delete(name: str):
    try:
        agent = _get_agent()
        deleted = agent.delete_snapshot(name)
        if not deleted:
            return _error("Snapshot not found.", status=404)
        return _success(deleted=True, name=name)
    except Exception as e:
        return _error(str(e))


@app.route("/api/snapshots/diff", methods=["POST"])
def api_snapshots_diff():
    """Field-level comparison of two named snapshots."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        a, b = data.get("a"), data.get("b")
        if not a or not b:
            return _error("Both snapshot names (a and b) are required.")
        diff = agent.diff_snapshots(a, b)
        return _success(diff=diff)
    except Exception as e:
        return _error(str(e))


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Update DB query safety settings (read-only mode / timeout)."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        if "read_only" in data:
            agent.db.read_only = bool(data["read_only"])
        if "timeout" in data:
            agent.db.query_timeout = data.get("timeout")
        if "n_jobs" in data:
            try:
                agent.set_n_jobs(int(data["n_jobs"]))
            except (TypeError, ValueError):
                pass
        return _success(
            read_only=agent.db.read_only,
            timeout=agent.db.query_timeout,
            n_jobs=agent.n_jobs,
        )
    except Exception as e:
        return _error(str(e))


# ========== Preprocessing ==========


@app.route("/api/preprocess", methods=["POST"])
def api_preprocess():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        operations = data.get("operations", [])
        if not isinstance(operations, list) or not operations:
            return _error("operations must be a non-empty list.")

        agent.apply_preprocessing(operations)
        summary = agent.get_preprocessing_summary()
        return _success(
            applied=True,
            summary=summary,
            data=_jsonable(agent.current_df.head(100)),
            columns=list(agent.current_df.columns),
            rows=len(agent.current_df),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/preprocess-summary", methods=["GET"])
def api_preprocess_summary():
    try:
        agent = _get_agent()
        return _success(summary=agent.get_preprocessing_summary())
    except Exception as e:
        return _error(str(e))


@app.route("/api/save-preprocessed-db", methods=["POST"])
def api_save_preprocessed_db():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        output = data.get("output_path", "preprocessed.db")
        table_name = data.get("table_name")
        include_original = data.get("include_original_tables", False)

        path = agent.save_preprocessed_db(
            output_path=output,
            table_name=table_name,
            include_original_tables=bool(include_original),
        )
        return _success(saved_path=path)
    except Exception as e:
        return _error(str(e))


# ========== Analysis ==========


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        analysis_type = data.get("type", "summary")
        target = data.get("target_column")
        use_cache = bool(data.get("use_cache", True))

        result = agent.analyze(target_column=target, analysis_type=analysis_type,
                               use_cache=use_cache)
        return _success(analysis=result, type=analysis_type)
    except Exception as e:
        return _error(str(e))


@app.route("/api/suggest-targets", methods=["GET"])
def api_suggest_targets():
    """Return ranked heuristic target-column suggestions for the current data."""
    try:
        agent = _get_agent()
        k = request.args.get("k", default=5, type=int)
        suggestions = agent.suggest_targets(k=k)
        top = suggestions[0] if suggestions else None
        return _success(suggestions=suggestions, top=top)
    except Exception as e:
        return _error(str(e))


@app.route("/api/auto-prepare", methods=["POST"])
def api_auto_prepare():
    """Best-effort automatic data preparation (drops constant/high-cardinality)."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        target = data.get("target_column")
        result = agent.auto_prepare(target_column=target)
        return _success(
            prepared=result["prepared"],
            rows=result["rows"],
            kept_columns=result["kept_columns"],
            dropped_columns=result["dropped_columns"],
            columns=list(agent.current_df.columns),
            data=_jsonable(agent.current_df.head(100)),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/feature-health", methods=["POST"])
def api_feature_health():
    """Report near-constant / high-cardinality / collinear features (filtering hints)."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        result = agent.check_feature_health(
            target_column=data.get("target_column"),
            corr_threshold=float(data.get("corr_threshold", 0.95)),
        )
        return _success(health=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/anomaly/detect", methods=["POST"])
def api_anomaly_detect():
    """Run unsupervised anomaly detection on a table or the loaded dataset."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        result = agent.detect_anomalies(
            table=data.get("table"),
            method=data.get("method", "isolation_forest"),
            contamination=float(data.get("contamination", 0.1)),
            features=data.get("features"),
        )
        return _success(anomaly=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/monitor/capture", methods=["POST"])
def api_monitor_capture():
    """Capture a feature-distribution reference from the loaded dataset."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        result = agent.capture_monitor_reference(feature_columns=data.get("features"))
        return _success(reference=result, rows=result.get("rows"))
    except Exception as e:
        return _error(str(e))


@app.route("/api/monitor/check", methods=["POST"])
def api_monitor_check():
    """Compare a live table's feature distributions against the reference."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        if not table:
            return _error("table is required.")
        result = agent.monitor_live(table, feature_columns=data.get("features"))
        return _success(drift=result)
    except Exception as e:
        return _error(str(e))


# ========== Training ==========


def _serialize_training(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-serializable representation of training results."""
    return {
        "task_type": results["task_type"],
        "target_column": results["target_column"],
        "best_model": results["best_model"],
        "best_cv_score": results["best_cv_score"],
        "model_scores": results["model_scores"],
        "test_metrics": _jsonable(results["test_metrics"]),
        "feature_columns": results["feature_columns"],
        "numeric_columns": results["numeric_columns"],
        "categorical_columns": results["categorical_columns"],
        "target_was_encoded": results.get("target_was_encoded", False),
        "class_mapping": results.get("class_mapping"),
        "feature_importance": (
            _jsonable(results["feature_importance"])
            if results.get("feature_importance") is not None else None
        ),
        "confusion_matrix": results.get("confusion_matrix"),
        "classification_report": results.get("classification_report"),
        "best_params": results.get("best_params"),
        "tuning": results.get("tuning"),
        "early_stopped": results.get("early_stopped", False),
        "experiment": results.get("experiment"),
        "model_path": results.get("model_path"),
    }


@app.route("/api/train", methods=["POST"])
def api_train():
    """Start an asynchronous training job and return its id immediately."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        target = data.get("target_column")
        task_type = data.get("task_type")
        table = data.get("table_name")
        tuning = data.get("tuning") if data.get("tuning") in ("off", "quick", "full") else None
        n_jobs_raw = data.get("n_jobs")
        n_jobs = int(n_jobs_raw) if n_jobs_raw else agent.n_jobs
        if n_jobs < 1:
            n_jobs = 1

        early_stop_raw = data.get("early_stop")
        early_stop = int(early_stop_raw) if early_stop_raw else None
        if early_stop is not None and early_stop < 1:
            early_stop = None

        if not target:
            return _error("target_column is required.")

        with TRAIN_LOCK:
            for job in TRAIN_JOBS.values():
                if job.get("status") in ("running", "cancelling"):
                    return _error("A training job is already running. Cancel it first or wait.")

        # Single concurrent training job per process, and refuse while draining.
        with TRAIN_LOCK:
            for job in TRAIN_JOBS.values():
                if job.get("status") in ("running", "cancelling"):
                    return _error("A training job is already running. Cancel it first or wait.")
        if not RUNNING:
            return _error("Server is shutting down. No new training.", status=503)

        job_id = f"train_{uuid.uuid4().hex[:8]}"  # A1: collision-safe id
        stop_flag = [False]
        started_at = time.time()
        job: Dict[str, Any] = {
            "id": job_id,
            "status": "running",
            "target": target,
            "progress": {"current": 0, "total": 0, "model": "", "scores": {}},
            "error": None,
            "training": None,
        }
        JOBS_DB.put(job_id, "train", None, "running",
                    params={"target_column": target}, created_at=started_at)

        def progress_cb(info: Dict[str, Any]) -> None:
            job["progress"] = info
            with TRAIN_LOCK:
                pass

        def should_stop() -> bool:
            # Honor explicit cancellation (set via /api/train/cancel on this
            # shared job dict) AND the wall-clock ceiling.
            if stop_flag[0] or job.get("status") == "cancelling":
                return True
            return time.time() - started_at > cfg.JOB_MAX_WALLCLOCK

        def worker() -> None:
            try:
                # A2: snapshot the training data under a *brief* lock, then run
                # the (long) model search off-lock so connect/disconnect/other
                # requests are never blocked for the whole fit.
                snapshot = None
                with AGENT_LOCK:
                    if table:
                        agent.load_table(table)
                    if agent.current_df is not None:
                        snapshot = agent.current_df.copy()
                if snapshot is None:
                    raise RuntimeError("No data loaded to train on.")
                results = agent.train(
                    target_column=target,
                    task_type=task_type,
                    progress_callback=progress_cb,
                    should_stop=should_stop,
                    tuning=tuning,
                    n_jobs=n_jobs,
                    early_stop=early_stop,
                    X=snapshot,
                )
                job["status"] = "done"
                job["training"] = _serialize_training(results)
                job["progress"] = {
                    "current": job["progress"]["total"] or 1,
                    "total": job["progress"]["total"] or 1,
                    "model": job["progress"].get("model", ""),
                    "scores": job["progress"].get("scores", {}),
                }
                JOBS_DB.update(job_id, status="done")
            except InterruptedError as e:
                job["status"] = "cancelled"
                job["error"] = str(e)
                JOBS_DB.update(job_id, status="cancelled", error=str(e))
            except RuntimeError as e:
                if time.time() - started_at > cfg.JOB_MAX_WALLCLOCK:
                    job["status"] = "timed_out"
                    job["error"] = f"Training exceeded {cfg.JOB_MAX_WALLCLOCK}s."
                    JOBS_DB.update(job_id, status="timed_out", error=job["error"])
                else:
                    job["status"] = "error"
                    job["error"] = str(e)
                    JOBS_DB.update(job_id, status="error", error=str(e))
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)
                JOBS_DB.update(job_id, status="error", error=str(e))

        with TRAIN_LOCK:
            TRAIN_JOBS[job_id] = job
        threading.Thread(target=worker, daemon=True).start()

        return _success(job_id=job_id, status=job["status"])
    except Exception as e:
        return _error(str(e))


@app.route("/api/train/status/<job_id>", methods=["GET"])
def api_train_status(job_id: str):
    try:
        with TRAIN_LOCK:
            job = TRAIN_JOBS.get(job_id)
        if job is None:
            stored = JOBS_DB.get(job_id)
            if stored is None:
                return _error("Job not found.", status=404)
            return jsonify({
                "job_id": job_id,
                "status": stored.get("status"),
                "target": (stored.get("params") or {}).get("target_column"),
                "progress": {},
                "error": stored.get("error"),
                "training": None,
                "persisted": True,
            })
        return jsonify({
            "job_id": job_id,
            "status": job["status"],
            "target": job["target"],
            "progress": job["progress"],
            "error": job["error"],
            "training": job["training"],
        })
    except Exception as e:
        return _error(str(e))


@app.route("/api/train/cancel/<job_id>", methods=["POST"])
def api_train_cancel(job_id: str):
    try:
        with TRAIN_LOCK:
            job = TRAIN_JOBS.get(job_id)
            if job is not None and job["status"] == "running":
                job["status"] = "cancelling"
        if job is None:
            return _error("Job not found.", status=404)
        if job["status"] == "cancelling":
            JOBS_DB.update(job_id, status="cancelling")
        return _success(status=job["status"], message="Cancellation requested. Stopping after the current model…")
    except Exception as e:
        return _error(str(e))


@app.route("/api/op/start", methods=["POST"])
def api_op_start():
    """Start an async operation job (analyze/synthesize/monitor/anomaly/
    repredict/recipe-apply/batch-whatif) and return its id immediately."""
    try:
        data = request.get_json(silent=True) or {}
        op_name = data.get("operation")
        params = data.get("params") or {}
        job = _start_op_job(op_name, params)
        return _success(job_id=job["id"], status=job["status"], operation=op_name)
    except JobBusyError as e:
        return _error(str(e), status=429)
    except Exception as e:
        return _error(str(e))


@app.route("/api/op/status/<job_id>", methods=["GET"])
def api_op_status(job_id: str):
    try:
        with OP_LOCK:
            job = OP_JOBS.get(job_id)
        # Survive a restart: fall back to the durable job store (A5).
        if job is None:
            stored = JOBS_DB.get(job_id)
            if stored is None:
                return _error("Operation job not found.", status=404)
            return jsonify({
                "job_id": job_id,
                "operation": stored.get("operation"),
                "status": stored.get("status"),
                "error": stored.get("error"),
                "result": stored.get("result"),
                "persisted": True,
            })
        return jsonify({
            "job_id": job_id,
            "operation": job["operation"],
            "status": job["status"],
            "error": job["error"],
            "result": job["result"],
        })
    except Exception as e:
        return _error(str(e))


@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    """Drain the most-recent async-job notifications (completion signals)."""
    try:
        with OP_LOCK:
            items = list(NOTIFICATIONS)
            NOTIFICATIONS.clear()
        return _success(notifications=items)
    except Exception as e:
        return _error(str(e))


@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    try:
        agent = _get_agent()
        info = agent.get_model_info()
        info["feature_importance"] = _jsonable(agent.get_feature_importance())
        return _success(model=info)
    except Exception as e:
        return _error(str(e))


# ========== Prediction ==========


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        rows = data.get("data")
        if rows is None:
            return _error("data field is required. Provide a JSON object or array of objects.")

        predictions = agent.predict(rows)
        return _success(
            predictions=_jsonable(predictions),
            columns=list(predictions.columns),
            rows=len(predictions),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/explain/prediction", methods=["POST"])
def api_explain_prediction():
    """Explain a single prediction by feature contributions."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        row = data.get("data")
        top_n = data.get("top_n", 5)
        if not row:
            return _error("data field is required (a single JSON object).")
        return _success(explanation=agent.explain_prediction(row, top_n=top_n))
    except Exception as e:
        return _error(str(e))


@app.route("/api/explain/whatif", methods=["POST"])
def api_explain_whatif():
    """Recompute a prediction after changing one feature (what-if)."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        row = data.get("data")
        feature = data.get("feature")
        value = data.get("value")
        if not row or not feature:
            return _error("data and feature are required.")
        return _success(whatif=agent.what_if(row, feature, value))
    except Exception as e:
        return _error(str(e))


@app.route("/api/explain/batch-whatif", methods=["POST"])
def api_explain_batch_whatif():
    """Perturb one feature across the whole loaded dataset and summarise the effect."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        feature = data.get("feature")
        if not feature:
            return _error("feature is required.")
        result = agent.batch_what_if(feature, data.get("value"))
        return _success(batch_whatif=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/repredict", methods=["POST"])
def api_repredict():
    """Re-score a (changed) table with the trained model, flagging drift + anomalies."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        if not table:
            return _error("table is required.")
        res = agent.repredict_table(
            table,
            limit=data.get("limit"),
            contamination=float(data.get("contamination", 0.05)),
        )
        return _success(
            predictions=_jsonable(res["predictions"]),
            columns=list(res["predictions"].columns),
            rows=res["rows"],
            drift=_jsonable(res["drift"]) if res["drift"] else None,
            anomaly=res["anomaly"],
            source=res["source"],
        )
    except Exception as e:
        return _error(str(e))


# ---------- Batch prediction ----------


@app.route("/api/predict/table", methods=["POST"])
def api_predict_table():
    """Run the trained model on rows of a database table, optionally paginated.

    Supports ``limit`` + ``offset`` for large tables so a single request never
    materializes the whole table + prediction set in memory at once.
    """
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        limit = data.get("limit")
        offset = data.get("offset") or 0
        if not table:
            return _error("table is required.")

        try:
            offset = max(0, int(offset))
            if limit:
                limit = max(1, int(limit))
        except (TypeError, ValueError):
            return _error("limit/offset must be integers.")

        total = agent.db.get_row_count(table)
        df = agent.load_table(table, limit=limit, offset=offset)
        predictions = agent.predict(df)
        return _success(
            predictions=_jsonable(predictions),
            columns=list(predictions.columns),
            rows=len(predictions),
            total=total,
            offset=offset,
            limit=limit,
            source=f"table:{table}",
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/predict/current", methods=["POST"])
def api_predict_current():
    """Run the trained model on the currently loaded dataset."""
    try:
        agent = _get_agent()
        if agent.current_df is None:
            return _error("No data loaded. Load a table or query first.")
        predictions = agent.predict(agent.current_df)
        return _success(
            predictions=_jsonable(predictions),
            columns=list(predictions.columns),
            rows=len(predictions),
            source="current",
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/predict/upload", methods=["POST"])
def api_predict_upload():
    """Upload a CSV of feature rows and run the trained model on them."""
    try:
        agent = _get_agent()
        if "file" not in request.files:
            return _error("No file uploaded. Use multipart/form-data with a 'file' field.")
        file = request.files["file"]
        if file.filename == "":
            return _error("No file selected.")
        if not os.path.splitext(file.filename)[1].lower() in (".csv", ".txt"):
            return _error("Please upload a .csv file.")

        df = pd.read_csv(file)
        if df.empty:
            return _error("Uploaded CSV is empty.")

        predictions = agent.predict(df)
        return _success(
            predictions=_jsonable(predictions),
            columns=list(predictions.columns),
            rows=len(predictions),
            source=f"csv:{file.filename}",
        )
    except Exception as e:
        return _error(str(e))


# ---------- Export ----------


@app.route("/api/export/csv", methods=["GET"])
def api_export_csv():
    """Stream the loaded dataset as a CSV download (chunked, low memory)."""
    try:
        agent = _get_agent()
        if agent.current_df is None:
            return _error("No data loaded. Load a table or query first.")
        df = agent.current_df

        def _generate() -> Any:
            yield df.iloc[:0].to_csv(index=False)  # header row
            chunk = 10_000
            for start in range(0, len(df), chunk):
                yield df.iloc[start:start + chunk].to_csv(index=False, header=False)

        return Response(
            _generate(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=exported_data.csv"
            },
        )
    except Exception as e:
        return _error(str(e))


# ========== Model persistence ==========


@app.route("/api/save-model", methods=["POST"])
def api_save_model():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        path = data.get("path", "model.joblib")
        agent.save_model(path)
        return _success(saved_path=path)
    except Exception as e:
        return _error(str(e))


@app.route("/api/load-model", methods=["POST"])
def api_load_model():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        path = data.get("path")
        if not path:
            return _error("path is required.")
        if not os.path.exists(path):
            return _error(f"Model file not found: {path}")

        agent.load_model(path)
        return _success(
            loaded=True,
            target_column=agent.target_column,
            model_info=agent.get_model_info(),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/model/download", methods=["GET"])
def api_model_download():
    try:
        path = request.args.get("path", "model.joblib")
        if not os.path.exists(path):
            return _error(f"Model file not found: {path}", status=404)
        return send_file(
            os.path.abspath(path),
            as_attachment=True,
            download_name=os.path.basename(path),
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/upload-model", methods=["POST"])
def api_upload_model():
    try:
        agent = _get_agent()
        if "file" not in request.files:
            return _error("No file uploaded. Use multipart/form-data with a 'file' field.")

        file = request.files["file"]
        if file.filename == "":
            return _error("No file selected.")

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_MODEL_EXTENSIONS:
            return _error(
                f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_MODEL_EXTENSIONS))}"
            )

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        agent.load_model(save_path)
        return _success(
            loaded=True,
            target_column=agent.target_column,
            model_info=agent.get_model_info(),
        )
    except Exception as e:
        return _error(str(e))


# ========== Relational deep-feature synthesis ==========


@app.route("/api/synthesize", methods=["POST"])
def api_synthesize():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        if not table:
            return _error("table is required.")
        df = agent.synthesize_features(
            table,
            max_rows=data.get("max_rows"),
            include_counts=data.get("include_counts", True),
            include_aggregates=data.get("include_aggregates", True),
        )
        return _success(
            loaded=True, table=table,
            rows=len(df), columns=list(df.columns),
            data=_jsonable(df.head(100)),
            summary=_jsonable(agent.get_feature_synthesizer_summary()),
        )
    except Exception as e:
        return _error(str(e))


# ========== Versioned experiments ==========


@app.route("/api/experiments", methods=["GET"])
def api_experiments_list():
    try:
        agent = _get_agent()
        return _success(experiments=_jsonable(agent.experiments.list()))
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/<eid>", methods=["GET"])
def api_experiments_get(eid: str):
    try:
        agent = _get_agent()
        exp = agent.experiments.get(eid)
        if exp is None:
            return _error("Experiment not found.", status=404)
        return _success(experiment=_jsonable(exp))
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/<eid>/champion", methods=["POST"])
def api_experiments_champion(eid: str):
    try:
        agent = _get_agent()
        ok = agent.experiments.set_champion(eid)
        if not ok:
            return _error("Experiment not found.", status=404)
        return _success(champion=agent.experiments.champion())
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/champion", methods=["GET"])
def api_experiments_get_champion():
    try:
        agent = _get_agent()
        return _success(champion=agent.experiments.champion())
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/<eid>/promote", methods=["POST"])
def api_experiments_promote(eid: str):
    """Promote an experiment to champion if it beats the current one, and
    reload the champion model so it becomes the live predictor."""
    try:
        agent = _get_agent()
        result = agent.promote_experiment(eid)
        return _success(**_jsonable(result))
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/rollback", methods=["POST"])
def api_experiments_rollback():
    """Revert the champion to the previous experiment and reload its model."""
    try:
        agent = _get_agent()
        result = agent.rollback_experiment()
        return _success(**_jsonable(result))
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/<eid>", methods=["DELETE"])
def api_experiments_delete(eid: str):
    try:
        agent = _get_agent()
        ok = agent.experiments.delete(eid)
        if not ok:
            return _error("Experiment not found.", status=404)
        return _success(deleted=True)
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/compare", methods=["POST"])
def api_experiments_compare():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        a, b = data.get("a"), data.get("b")
        if not a or not b:
            return _error("Both 'a' and 'b' experiment ids are required.")
        return _success(diff=_jsonable(agent.experiments.compare(a, b)))
    except Exception as e:
        return _error(str(e))


@app.route("/api/experiments/propose", methods=["POST"])
def api_experiments_propose():
    """Find the best previous completed experiment matching the given data
    source and/or target, so the user can reuse it instead of retraining."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        data_source = data.get("data_source") or data.get("table")
        target = data.get("target_column")
        found = agent.find_experiment(data_source=data_source, target_column=target)
        return _success(found=found is not None, experiment=_jsonable(found) if found else None)
    except Exception as e:
        return _error(str(e))


# ========== LLM ==========


@app.route("/api/llm/enable", methods=["POST"])
def api_llm_enable():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        agent.enable_llm(
            base_url=data.get("base_url", "http://localhost:1234/v1"),
            model=data.get("model"),
            timeout=data.get("timeout", 60),
        )
        return _success(enabled=True, status=agent.llm_check())
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/check", methods=["GET"])
def api_llm_check():
    try:
        agent = _get_agent()
        if agent.llm is None:
            agent.enable_llm()
        return _success(status=agent.llm_check())
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/sql", methods=["POST"])
def api_llm_sql():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        question = data.get("question")
        if not question:
            return _error("question is required.")

        result = agent.llm_generate_sql(question)
        return _success(result=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/suggest-target", methods=["POST"])
def api_llm_suggest_target():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        result = agent.llm_suggest_target(preferred=data.get("preferred"))
        return _success(result=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/suggest-preprocessing", methods=["POST"])
def api_llm_suggest_preprocessing():
    try:
        agent = _get_agent()
        result = agent.llm_suggest_preprocessing()
        return _success(result=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/apply-preprocessing", methods=["POST"])
def api_llm_apply_preprocessing():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        operations = data.get("operations")
        if isinstance(operations, list) and operations:
            agent.apply_preprocessing(operations)
        else:
            suggestion = agent.llm_suggest_preprocessing()
            if "error" in suggestion:
                return _error(suggestion["error"])
            ops = suggestion.get("operations", [])
            if not ops:
                return _error("No preprocessing operations recommended.")
            agent.apply_preprocessing(ops)

        return _success(
            applied=True,
            summary=agent.get_preprocessing_summary(),
            data=_jsonable(agent.current_df.head(100)) if agent.current_df is not None else None,
        )
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/explain-results", methods=["POST"])
def api_llm_explain_results():
    try:
        agent = _get_agent()
        result = agent.llm_explain_results()
        return _success(result=result)
    except Exception as e:
        return _error(str(e))


@app.route("/api/llm/explain-predictions", methods=["POST"])
def api_llm_explain_predictions():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        predictions = data.get("predictions")
        top_n = data.get("top_n", 5)

        if predictions is None:
            return _error("predictions field is required.")

        pred_df = pd.DataFrame(predictions)
        result = agent.llm_explain_predictions(pred_df, top_n=top_n)
        return _success(result=result)
    except Exception as e:
        return _error(str(e))


# ============ Request logging (structured, production-friendly) ============


@app.before_request
def _log_request_start() -> None:
    request._start_time = time.monotonic()
    if not cfg.DISABLE_REQUEST_LOG:
        log.info("-> %s %s", request.method, request.path)


@app.after_request
def _log_request_end(resp: Response) -> Response:
    with METRICS_LOCK:
        METRICS["requests"] += 1
        if resp.status_code >= 400:
            METRICS["errors"] += 1
        key = f"{request.method} {request.path}"
        METRICS["by_path"][key] = METRICS["by_path"].get(key, 0) + 1
    ms = (time.monotonic() - getattr(request, "_start_time", time.monotonic())) * 1000
    if not cfg.DISABLE_REQUEST_LOG:
        log.info("<- %s %s %s (%.1f ms)", request.method, request.path, resp.status_code, ms)
    return resp


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Agent Web API")
    parser.add_argument(
        "--database", "--db", "-db",
        help="SQL database connection string or SQLite file path (e.g., sample_company.db)",
    )
    parser.add_argument(
        "--host", default=cfg.WEB_HOST,
        help="Host to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--port", type=int, default=cfg.WEB_PORT,
        help="Port to bind (default: %(default)s)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run the Flask development server instead of the production WSGI server",
    )
    args = parser.parse_args()

    configure_logging()

    global AGENT
    if args.database:
        try:
            AGENT = MLAgent(args.database)
            log.info("Connected to database: %s", args.database)
        except Exception as e:
            log.error("Failed to connect to database: %s", e)

    _install_signal_handlers()

    # Production default: serve via waitress (cross-platform, thread pool) so
    # the API can handle concurrent requests without the Flask dev server.
    if not args.debug:
        try:
            from waitress import serve

            log.info(
                "ML Agent Web API (waitress) at http://%s:%s with %d threads",
                args.host, args.port, cfg.WSGI_THREADS,
            )
            try:
                serve(
                    app,
                    host=args.host,
                    port=args.port,
                    threads=cfg.WSGI_THREADS,
                    channel_timeout=300,
                )
            finally:
                _shutdown()
            return
        except ImportError:
            log.warning(
                "waitress not installed; falling back to the Flask dev server. "
                "Install it with: pip install waitress"
            )
        except Exception as e:  # pragma: no cover - defensive
            log.warning("waitress failed to start (%s); using Flask dev server", e)

    log.info("ML Agent Web API (dev server) at http://%s:%s", args.host, args.port)
    try:
        app.run(host=args.host, port=args.port, debug=args.debug, threaded=cfg.WSGI_THREADS > 1)
    finally:
        _shutdown()


def _install_signal_handlers() -> None:
    """On SIGTERM/SIGINT, refuse new work and let in-flight jobs drain (C2)."""
    try:
        import signal

        def _on_stop(signum, frame):  # noqa: ANN001
            global RUNNING
            with SHUTDOWN_LOCK:
                RUNNING = False
            log.warning("Received %s; draining in-flight jobs and shutting down.", signum)

        signal.signal(signal.SIGTERM, _on_stop)
        signal.signal(signal.SIGINT, _on_stop)
    except Exception:  # pragma: no cover - non-POSIX platforms
        pass


def _shutdown() -> None:
    """Drain background work and release resources on exit."""
    global RUNNING
    with SHUTDOWN_LOCK:
        RUNNING = False
    # Give in-flight jobs a bounded window to finish (short ops), then close.
    try:
        JOB_EXECUTOR.shutdown(wait=True, cancel_futures=False)
    except Exception:
        pass
    for session in list(SESSIONS.values()):
        try:
            session.close()
        except Exception:
            pass
    if AGENT is not None:
        try:
            AGENT.close()
        except Exception:
            pass
    try:
        JOBS_DB.close()
    except Exception:
        pass
    log.info("Shutdown complete.")
