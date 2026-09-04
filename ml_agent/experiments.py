"""Versioned experiment tracking for the ML Agent.

Records the lineage of each training run: the exact data source (table / SQL /
sample seed / feature synthesis), the preprocessing and tuning config, model
metrics, and a champion/rollback mechanism. Persisted to JSON, keyed by
database connection, so runs can be reproduced and compared later.
"""
import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional


class ExperimentTracker:
    """Stores, lists, compares, and champions training experiments."""

    def __init__(self, store_path: Optional[str] = None):
        self._store_path = store_path
        self._experiments: Dict[str, dict] = {}
        self._champion_id: Optional[str] = None
        self._prev_champion_id: Optional[str] = None
        if store_path and os.path.exists(store_path):
            try:
                with open(store_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._experiments = data.get("experiments", {})
                self._champion_id = data.get("champion") or None
            except Exception:
                self._experiments = {}

    def set_store(self, path: str) -> None:
        self._store_path = path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                self._experiments = data.get("experiments", {})
                self._champion_id = data.get("champion") or None
            except Exception:
                self._experiments = {}

    def _persist(self) -> None:
        if not self._store_path:
            return
        try:
            with open(self._store_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "experiments": self._experiments,
                        "champion": self._champion_id,
                    },
                    fh, indent=2,
                )
        except Exception:
            pass

    def start(
        self,
        data_source: str,
        target_column: str,
        task_type: Optional[str] = None,
        tuning: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create and return a new experiment id."""
        eid = f"exp_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self._experiments[eid] = {
            "id": eid,
            "created": time.time(),
            "data_source": data_source,
            "target_column": target_column,
            "task_type": task_type,
            "tuning": tuning,
            "meta": meta or {},
            "status": "running",
            "metrics": None,
            "best_model": None,
            "best_cv_score": None,
            "best_params": None,
            "model_path": None,
        }
        self._persist()
        return eid

    def finish(
        self,
        eid: str,
        training_results: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ) -> None:
        exp = self._experiments.get(eid)
        if exp is None:
            return
        exp["status"] = "done"
        if training_results:
            exp.update({
                "task_type": training_results.get("task_type", exp["task_type"]),
                "best_model": training_results.get("best_model"),
                "best_cv_score": training_results.get("best_cv_score"),
                "best_params": training_results.get("best_params"),
                "metrics": training_results.get("test_metrics"),
                "model_scores": training_results.get("model_scores"),
            })
        exp["model_path"] = model_path
        self._persist()

    def fail(self, eid: str, error: str) -> None:
        exp = self._experiments.get(eid)
        if exp:
            exp["status"] = "error"
            exp["error"] = error
            self._persist()

    def list(self) -> List[Dict[str, Any]]:
        return sorted(
            self._experiments.values(),
            key=lambda e: e.get("created", 0), reverse=True,
        )

    def get(self, eid: str) -> Optional[Dict[str, Any]]:
        return self._experiments.get(eid)

    def set_champion(self, eid: str) -> bool:
        if eid not in self._experiments:
            return False
        if eid != self._champion_id:
            self._prev_champion_id = self._champion_id
            self._champion_id = eid
            self._persist()
        return True

    def champion(self):
        return self._experiments.get(self._champion_id) if self._champion_id else None

    def prev_champion(self):
        return (
            self._experiments.get(self._prev_champion_id)
            if self._prev_champion_id else None
        )

    def set_model_path(self, eid: str, path: Optional[str]) -> None:
        exp = self._experiments.get(eid)
        if exp:
            exp["model_path"] = path
            self._persist()

    def promote_if_better(self, eid: str) -> Dict[str, Any]:
        """Automatically promote an experiment if it beats the current champion.

        Uses ``best_cv_score`` (accuracy for classification, R2 for regression —
        higher is better). Returns a decision dict describing the outcome.
        """
        exp = self._experiments.get(eid)
        if exp is None:
            raise ValueError(f"Experiment not found: {eid}")

        prev = self._experiments.get(self._champion_id) if self._champion_id else None
        if prev is None or prev.get("best_cv_score") is None:
            # No champion yet (or champion has no score) — promote unconditionally.
            self.set_champion(eid)
            return {
                "decision": "promoted",
                "reason": "first_champion",
                "previous": prev,
                "champion": self.champion(),
            }

        new_score = exp.get("best_cv_score")
        old_score = prev.get("best_cv_score")
        if new_score is None:
            return {
                "decision": "kept",
                "reason": "no_score",
                "previous": prev,
                "champion": self.champion(),
            }

        # higher is better (accuracy / R2)
        if new_score >= old_score:
            self.set_champion(eid)
            return {
                "decision": "promoted",
                "reason": "better",
                "previous": prev,
                "champion": self.champion(),
            }

        return {
            "decision": "kept",
            "reason": "not_better",
            "previous": prev,
            "champion": self.champion(),
        }

    def rollback(self) -> Dict[str, Any]:
        """Revert the champion to the previous one (if any). Returns a dict
        describing whether a rollback occurred and the restored champion."""
        if self._prev_champion_id and self._prev_champion_id in self._experiments:
            old = self._experiments.get(self._champion_id) if self._champion_id else None
            self._champion_id = self._prev_champion_id
            self._prev_champion_id = None
            self._persist()
            return {
                "rolled_back": True,
                "previous": old,
                "champion": self._experiments.get(self._champion_id),
            }
        return {
            "rolled_back": False,
            "previous": None,
            "champion": self.champion(),
        }

    def delete(self, eid: str) -> bool:
        existed = eid in self._experiments
        if existed:
            del self._experiments[eid]
            if eid == self._champion_id:
                self._champion_id = None
            if eid == self._prev_champion_id:
                self._prev_champion_id = None
            self._persist()
        return existed

    def compare(self, a: str, b: str) -> Dict[str, Any]:
        ea, eb = self._experiments.get(a), self._experiments.get(b)
        if not ea or not eb:
            raise ValueError("Both experiment ids must exist.")
        return {
            "a": ea, "b": eb,
            "a_better": bool(
                ea.get("best_cv_score") is not None and eb.get("best_cv_score") is not None
                and ea["best_cv_score"] >= eb["best_cv_score"]
            ),
        }