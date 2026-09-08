"""Operation handlers executed by background jobs (thread or subprocess).

Kept in this module (not web_api) so the same code can run inside a child
process for hard-killable background jobs: the factory functions take an
``MLAgent`` and a params dict and return a JSON-serializable result, and this
module has no Flask/web dependencies.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from ml_agent.agent import MLAgent


def jsonable(obj: Any) -> Any:
    """Convert an object to a JSON-serializable representation."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        if isinstance(obj, float) and pd.isna(obj):
            return None
        return obj
    return str(obj)


def serialize_training(results: Dict[str, Any]) -> Dict[str, Any]:
    """Build a JSON-serializable representation of model-selection results."""
    return {
        "task_type": results["task_type"],
        "target_column": results["target_column"],
        "best_model": results["best_model"],
        "best_cv_score": results["best_cv_score"],
        "model_scores": results["model_scores"],
        "test_metrics": jsonable(results["test_metrics"]),
        "feature_columns": results["feature_columns"],
        "numeric_columns": results["numeric_columns"],
        "categorical_columns": results["categorical_columns"],
        "target_was_encoded": results.get("target_was_encoded", False),
        "class_mapping": results.get("class_mapping"),
        "feature_importance": (
            jsonable(results["feature_importance"])
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


# ---- Handlers: (agent, params) -> JSON-serializable dict -------------------


def _op_analyze(agent: MLAgent, params: Dict[str, Any]):
    return {"analysis": jsonable(agent.analyze(
        params.get("target_column"), params.get("type", "summary")
    ))}


def _op_anomaly(agent: MLAgent, params: Dict[str, Any]):
    result = agent.detect_anomalies(
        table=params.get("table") or None,
        method=params.get("method", "isolation_forest"),
        contamination=float(params.get("contamination", 0.1)),
    )
    return {"anomaly": jsonable(result)}


def _op_synthesize(agent: MLAgent, params: Dict[str, Any]):
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
        "data": jsonable(agent.current_df.head(100)),
    }


def _op_monitor_capture(agent: MLAgent, params: Dict[str, Any]):
    result = agent.capture_monitor_reference()
    return {"rows": len(agent.current_df) if agent.current_df is not None else 0,
            "features": result.get("features", [])}


def _op_monitor_check(agent: MLAgent, params: Dict[str, Any]):
    table = params.get("table")
    if not table:
        raise ValueError("table is required.")
    return {"drift": jsonable(agent.monitor_live(table, feature_columns=params.get("features")))}


def _op_repredict(agent: MLAgent, params: Dict[str, Any]):
    table = params.get("table")
    if not table:
        raise ValueError("table is required.")
    res = agent.repredict_table(
        table, limit=params.get("limit"), contamination=float(params.get("contamination", 0.05))
    )
    return {
        "predictions": jsonable(res["predictions"]),
        "drift": jsonable(res["drift"]),
        "anomaly": res["anomaly"],
        "rows": res["rows"],
        "columns": list(res["predictions"].columns),
    }


def _op_recipe_apply(agent: MLAgent, params: Dict[str, Any]):
    name = params.get("name")
    if not name:
        raise ValueError("recipe name is required.")
    results = agent.apply_recipe(
        name,
        target=params.get("target"),
        tuning=params.get("tuning"),
        n_jobs=params.get("n_jobs"),
    )
    return {"training": serialize_training(results), "recipe": name}


def _op_batch_whatif(agent: MLAgent, params: Dict[str, Any]):
    feature = params.get("feature")
    if not feature:
        raise ValueError("feature is required.")
    result = agent.batch_what_if(feature, params.get("value"))
    return {"batch_whatif": jsonable(result)}


# Exposed name used by the web layer and the subprocess worker.
handlers: Dict[str, Any] = {
    "analyze": _op_analyze,
    "anomaly": _op_anomaly,
    "synthesize": _op_synthesize,
    "monitor_capture": _op_monitor_capture,
    "monitor_check": _op_monitor_check,
    "repredict": _op_repredict,
    "recipe_apply": _op_recipe_apply,
    "batch_whatif": _op_batch_whatif,
}