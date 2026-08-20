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
from typing import Any, Dict, List, Optional

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response
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

# Upload directory for database files
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "ml_agent_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_DB_EXTENSIONS = {".db", ".sqlite", ".sqlite3"}

# ============ Helpers ============


def _get_agent() -> MLAgent:
    """Return the shared agent instance."""
    global AGENT
    if AGENT is None:
        raise RuntimeError("No database connected. POST /api/connect first.")
    return AGENT


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


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        agent = _get_agent()
        data = request.get_json(silent=True) or {}
        target = data.get("target_column")
        task_type = data.get("task_type")
        table = data.get("table_name")

        if not target:
            return _error("target_column is required.")

        results = agent.train(
            target_column=target,
            task_type=task_type,
            table_name=table,
        )

        # Build serializable results
        serializable = {
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
        }

        return _success(training=serializable)
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
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
