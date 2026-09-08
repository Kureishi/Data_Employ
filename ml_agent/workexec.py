"""Child-process entrypoints for hard-killable background jobs.

The parent (web_api) spawns a subprocess that calls :func:`entry`; that is the
only symbol the child imports, so spawning does not re-import the heavy Flask
app. The child:

- re-connects to the DB from a connection string (the MLAgent is not pickled),
- re-hydrates the working dataset from a ``data_spec`` (table/query),
- runs the job (train or an op handler),
- reports lifecycle / progress / result through the durable SQLite JobStore.

The parent can then poll the store, honour cancellations, and — if a fit is
genuinely stuck and the soft wall-clock (checked between fits) is unresponsive —
hard-kill the subprocess (terminate/kill).
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional

from ml_agent import MLAgent
from ml_agent.job_store import JobStore

# Ops are imported inside the worker functions to keep import time low even if
# only training is being run in a subprocess.


def _upd(store: Optional[JobStore], job_id: str, **kwargs: Any) -> None:
    if store is not None:
        try:
            store.update(job_id, **kwargs)
        except Exception:
            pass


def _rehydrate(agent: MLAgent, data_spec: Optional[Dict[str, Any]]) -> None:
    if not data_spec:
        return
    try:
        typ = data_spec.get("type")
        if typ == "table":
            agent.load_table(data_spec.get("table"), limit=data_spec.get("limit"))
        elif typ == "query":
            agent.load_query_as_data(data_spec.get("query"))
    except Exception:
        # Data rehydration is best-effort; individual handlers re-check.
        pass


def _poll_cancel(store: Optional[JobStore], job_id: str, started: float,
                 max_wallclock: int) -> bool:
    if store is not None:
        try:
            rec = store.get(job_id)
            if rec:
                if rec.get("status") == "cancelling" or rec.get("cancel_requested"):
                    return True
        except Exception:
            pass
    return (time.time() - started) > max_wallclock


def _run_train(kw: Dict[str, Any]) -> None:
    store = JobStore(kw.get("jobs_db_path")) if kw.get("jobs_db_path") else None
    job_id = kw["job_id"]
    started = time.time()
    max_wallclock = int(kw.get("max_wallclock") or 3600)

    def should_stop() -> bool:
        return _poll_cancel(store, job_id, started, max_wallclock)

    def progress_cb(info: Dict[str, Any]) -> None:
        _upd(store, job_id, progress=info)

    try:
        agent = MLAgent(kw["connection"])
        agent.models_dir = kw.get("models_dir")  # match parent's model dir
        _rehydrate(agent, kw.get("data_spec"))
        results = agent.train(
            target_column=kw["target"],
            task_type=kw.get("task_type"),
            progress_callback=progress_cb,
            should_stop=should_stop,
            tuning=kw.get("tuning"),
            n_jobs=kw.get("n_jobs"),
            early_stop=kw.get("early_stop"),
        )
        from ml_agent.ops import serialize_training

        payload = serialize_training(results)
        _upd(store, job_id,
             status="done",
             result=payload,
             progress=results.get("progress", {}))
    except Exception as e:
        _upd(store, job_id, status="error", error=str(e))


def _run_op(kw: Dict[str, Any]) -> None:
    store = JobStore(kw.get("jobs_db_path")) if kw.get("jobs_db_path") else None
    job_id = kw["job_id"]
    try:
        from ml_agent.ops import handlers

        op_name = kw["op_name"]
        handler = handlers.get(op_name)
        if handler is None:
            raise ValueError(f"Unknown operation: {op_name}")
        agent = MLAgent(kw["connection"])
        _rehydrate(agent, kw.get("data_spec"))
        result = handler(agent, kw.get("params") or {})
        _upd(store, job_id, status="done", result=result)
    except Exception as e:
        _upd(store, job_id, status="error", error=str(e))


def entry(kind: str, kw: Dict[str, Any]) -> None:
    """Spawn target for a child process executing ``kind`` in ['train','op']."""
    if kind == "train":
        _run_train(kw)
    else:
        _run_op(kw)