"""
ML Agent Web API - Flask backend exposing the ML Agent as a REST API.

Run:
    python web_api.py --db sample_company.db
    python web_api.py --db sqlite:///sample_company.db --port 8080

Then open http://localhost:5000 in your browser.
"""
import argparse
import os
import sys
import tempfile
import threading
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, send_file, Response
from werkzeug.utils import secure_filename

from ml_agent import MLAgent

# ============ App setup ============

app = Flask(
    __name__,
    static_folder="web/static",
    template_folder="web/templates",
)

# Global agent state
AGENT: Optional[MLAgent] = None

# Async training-job registry (id -> job dict)
TRAIN_JOBS: Dict[str, Dict[str, Any]] = {}
TRAIN_LOCK = threading.Lock()

# Upload directory for database files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ml_agent_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_DB_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}
ALLOWED_MODEL_EXTENSIONS = {".joblib", ".pkl", ".pickle"}

# ============ Helpers ============


def _get_agent() -> MLAgent:
    """Return the shared agent instance."""
    global AGENT
    if AGENT is None:
        raise RuntimeError("No database connected. POST /api/connect first.")
    return AGENT


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
    """Connect to a database, replacing any existing connection."""
    global AGENT
    data = request.get_json(silent=True) or {}
    connection = data.get("connection") or data.get("db")
    if not connection:
        return _error("Connection string or SQLite file path is required.")

    # If we already have an agent with the same connection, just return state
    if AGENT is not None:
        try:
            existing = _get_agent()
            if existing.db.connection_string:
                existing_cs = existing.db.connection_string.replace("sqlite:///", "")
                if existing_cs == connection or existing.db.connection_string == connection:
                    return _success(
                        connected=True,
                        tables=existing.list_tables(),
                        message="Already connected.",
                    )
        except Exception:
            pass

    # Create new agent
    if AGENT is not None:
        try:
            AGENT.close()
        except Exception:
            pass

    try:
        AGENT = MLAgent(connection)
        _init_agent_stores(AGENT)
        tables = AGENT.list_tables()
        return _success(
            connected=True,
            tables=tables,
            message=f"Connected to {connection}",
        )
    except Exception as e:
        AGENT = None
        return _error(f"Failed to connect: {e}")


@app.route("/api/upload-db", methods=["POST"])
def api_upload_db():
    """Upload a database file and connect to it."""
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

    # Close any existing agent
    if AGENT is not None:
        try:
            AGENT.close()
        except Exception:
            pass

    try:
        AGENT = MLAgent(save_path)
        _init_agent_stores(AGENT)
        tables = AGENT.list_tables()
        return _success(
            connected=True,
            tables=tables,
            message=f"Connected to uploaded database: {filename}",
            path=save_path,
        )
    except Exception as e:
        AGENT = None
        return _error(f"Failed to connect to uploaded database: {e}")


@app.route("/api/disconnect", methods=["POST"])
def api_disconnect():
    """Disconnect from the database."""
    global AGENT
    if AGENT is not None:
        try:
            AGENT.close()
        except Exception:
            pass
    AGENT = None
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
        return _success(
            read_only=agent.db.read_only,
            timeout=agent.db.query_timeout,
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

        result = agent.analyze(target_column=target, analysis_type=analysis_type)
        return _success(analysis=result, type=analysis_type)
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

        if not target:
            return _error("target_column is required.")

        with TRAIN_LOCK:
            for job in TRAIN_JOBS.values():
                if job.get("status") in ("running", "cancelling"):
                    return _error("A training job is already running. Cancel it first or wait.")

        job_id = f"train_{len(TRAIN_JOBS) + 1}"
        stop_flag = [False]
        job: Dict[str, Any] = {
            "id": job_id,
            "status": "running",
            "target": target,
            "progress": {"current": 0, "total": 0, "model": "", "scores": {}},
            "error": None,
            "training": None,
        }

        def progress_cb(info: Dict[str, Any]) -> None:
            job["progress"] = info
            with TRAIN_LOCK:
                pass

        def should_stop() -> bool:
            return bool(stop_flag[0])

        def worker() -> None:
            try:
                results = agent.train(
                    target_column=target,
                    task_type=task_type,
                    table_name=table,
                    progress_callback=progress_cb,
                    should_stop=should_stop,
                    tuning=tuning,
                )
                job["status"] = "done"
                job["training"] = _serialize_training(results)
                job["progress"] = {
                    "current": job["progress"]["total"] or 1,
                    "total": job["progress"]["total"] or 1,
                    "model": job["progress"].get("model", ""),
                    "scores": job["progress"].get("scores", {}),
                }
            except InterruptedError as e:
                job["status"] = "cancelled"
                job["error"] = str(e)
            except Exception as e:
                job["status"] = "error"
                job["error"] = str(e)

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
            return _error("Job not found.", status=404)
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
        return _success(status=job["status"], message="Cancellation requested. Stopping after the current model…")
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


# ---------- Batch prediction ----------


@app.route("/api/predict/table", methods=["POST"])
def api_predict_table():
    """Run the trained model on every row of a database table."""
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        table = data.get("table")
        limit = data.get("limit")
        if not table:
            return _error("table is required.")

        df = agent.load_table(table, limit=limit)
        predictions = agent.predict(df)
        return _success(
            predictions=_jsonable(predictions),
            columns=list(predictions.columns),
            rows=len(predictions),
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
    """Download the currently loaded dataset as CSV."""
    try:
        agent = _get_agent()
        if agent.current_df is None:
            return _error("No data loaded. Load a table or query first.")
        csv_str = agent.current_df.to_csv(index=False)
        return Response(
            csv_str,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="ML Agent Web API")
    parser.add_argument(
        "--database", "--db", "-db",
        help="SQL database connection string or SQLite file path (e.g., sample_company.db)",
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port", type=int, default=5000,
        help="Port to bind (default: 5000)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable Flask debug mode",
    )
    args = parser.parse_args()

    global AGENT
    if args.database:
        try:
            AGENT = MLAgent(args.database)
            print(f"Connected to database: {args.database}", file=sys.stderr)
        except Exception as e:
            print(f"Failed to connect to database: {e}", file=sys.stderr)

    print(f"ML Agent Web API running at http://{args.host}:{args.port}", file=sys.stderr)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
