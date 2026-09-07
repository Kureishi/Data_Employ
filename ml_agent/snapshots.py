"""Named result snapshots for before/after and side-by-side comparison.

A snapshot stores a JSON-serializable result payload (an analysis report, a
training summary, a prediction batch, a recipe reference, ...) under a
user-chosen name, persisted per-database. ``diff_snapshots`` compares two
snapshots' payloads field by field, reporting numeric deltas and
equal/changed flags, so users can see e.g. "before vs after preprocessing" or
"model A vs model B".
"""
import json
import os
import time
from typing import Any, Dict, List, Optional


def _norm(value: Any) -> Any:
    """Best-effort comparison normalisation (treat NaN/None as equal)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


class SnapshotStore:
    """Load, save, list, delete, and diff named result snapshots."""

    def __init__(self, store_path: Optional[str] = None):
        self._store_path: Optional[str] = None
        self._snapshots: Dict[str, Dict[str, Any]] = {}
        if store_path:
            self.set_store(store_path)

    def set_store(self, path: str) -> None:
        self._store_path = path
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._snapshots = json.load(f)
            except Exception:
                self._snapshots = {}

    def _persist(self) -> None:
        if not self._store_path:
            return
        try:
            with open(self._store_path, "w", encoding="utf-8") as f:
                json.dump(self._snapshots, f, indent=2)
        except Exception:
            pass

    def save(
        self,
        name: str,
        kind: str,
        payload: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save (or overwrite) a snapshot under ``name``. Returns the name."""
        name = (name or "").strip() or time.strftime("snapshot_%Y%m%d_%H%M%S")
        self._snapshots[name] = {
            "name": name,
            "kind": kind or "manual",
            "created": time.time(),
            "created_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
            "payload": payload,
        }
        self._persist()
        return name

    def list(self) -> List[Dict[str, Any]]:
        snaps = list(self._snapshots.values())
        snaps.sort(key=lambda s: s.get("created") or 0)
        return snaps

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self._snapshots.get(name)

    def delete(self, name: str) -> bool:
        if name in self._snapshots:
            del self._snapshots[name]
            self._persist()
            return True
        return False

    def diff(self, a: str, b: str) -> Dict[str, Any]:
        """Compare two snapshots' payloads and report field-level changes."""
        sa, sb = self._snapshots.get(a), self._snapshots.get(b)
        if not sa or not sb:
            raise ValueError("Both snapshot names must exist.")
        return _diff_payloads(a, b, sa.get("payload") or {}, sb.get("payload") or {},
                              meta_a=sa, meta_b=sb)


def _diff_payloads(a: str, b: str, pa: Dict, pb: Dict,
                   meta_a: Optional[Dict] = None, meta_b: Optional[Dict] = None,
                   path: str = "", depth: int = 0) -> Dict[str, Any]:
    changes = []
    seen = set()
    for k in list(pa.keys()) + list(pb.keys()):
        if k in seen:
            continue
        seen.add(k)
        va, vb = pa.get(k), pb.get(k)
        full = f"{path}.{k}" if path else str(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            sub = _diff_payloads(a, b, va, vb, path=full, depth=depth + 1)
            if sub.get("changed"):
                changes.append({"field": full, "kind": "nested", "changed": True,
                                "summary": f"{sub.get('change_count', 0)} change(s)"})
            continue
        if (isinstance(va, (int, float)) and isinstance(vb, (int, float))
                and not isinstance(va, bool) and not isinstance(vb, bool)):
            delta = vb - va
            if delta == 0:
                continue
            changes.append({"field": full, "kind": "numeric", "changed": True,
                            "old": va, "new": vb, "delta": float(delta)})
            continue
        na, nb = _norm(va), _norm(vb)
        if na == nb:
            continue
        changes.append({"field": full, "kind": "other", "changed": True,
                        "old": na, "new": nb})

    return {
        "a": a, "b": b,
        "kind_a": (meta_a or {}).get("kind"),
        "kind_b": (meta_b or {}).get("kind"),
        "created_a": (meta_a or {}).get("created_iso"),
        "created_b": (meta_b or {}).get("created_iso"),
        "changed": len(changes) > 0,
        "change_count": len(changes),
        "changes": changes,
    }