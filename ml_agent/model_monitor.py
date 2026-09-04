"""Model drift monitoring for the ML Agent.

Captures a reference distribution for the features used in training, then
compares a live (production) table's feature distributions against it using
the Population Stability Index (PSI). Shrinking/rising PSI per feature flags
drift before it degrades prediction quality.
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# PSI thresholds: <0.1 stable, 0.1..0.25 moderate, >0.25 drift
STABLE, MODERATE = 0.1, 0.25


class ModelMonitor:
    """Compute and compare feature-drift statistics (PSI)."""

    def __init__(self, n_bins: int = 10, min_fraction: float = 1e-3) -> None:
        self.n_bins = n_bins
        self.min_fraction = min_fraction
        self.reference: Optional[Dict[str, Any]] = None

    def capture_reference(
        self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compute per-feature reference distributions from `df`."""
        if df is None or len(df) == 0:
            raise RuntimeError("Cannot capture a reference from empty data.")
        feats = feature_columns or list(df.columns)
        reference: Dict[str, Any] = {}
        for col in feats:
            if col not in df.columns:
                continue
            s = df[col]
            if pd.api.types.is_numeric_dtype(s.dtype):
                s_clean = s.dropna()
                if s_clean.nunique() < 2:
                    reference[col] = {"type": "constant", "value": float(s_clean.median())}
                    continue
                edges = np.quantile(s_clean, np.linspace(0, 1, self.n_bins + 1))
                counts, _ = np.histogram(s_clean, bins=edges)
                reference[col] = {
                    "type": "numeric",
                    "edges": [float(e) for e in edges],
                    "expected": self._normalize(counts),
                }
            else:
                vc = s.value_counts(normalize=True).to_dict()
                reference[col] = {
                    "type": "categorical",
                    "categories": list(vc.keys()),
                    "expected": [float(vc[k]) for k in vc],
                }
        self.reference = {"features": reference, "rows": int(len(df))}
        return self.reference

    def monitor(self, live_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare each feature in `live_df` against the stored reference (PSI)."""
        if self.reference is None:
            raise RuntimeError("No reference captured yet. Call capture_reference() first.")
        reference = self.reference["features"]
        results: List[Dict[str, Any]] = []
        for col, ref in reference.items():
            if col not in live_df.columns:
                results.append({
                    "feature": col, "psi": None, "status": "missing",
                    "message": "Feature missing from live data.",
                })
                continue
            s = live_df[col]
            if ref["type"] == "constant":
                change = str(s.median()) != str(ref["value"])
                psi = 1.0 if change else 0.0
                results.append({
                    "feature": col, "psi": round(float(psi), 4),
                    "status": "drift" if change else "stable",
                    "message": "Reference feature is constant; value changed." if change
                    else "Reference feature is constant.",
                })
                continue
            if ref["type"] == "numeric":
                actual_counts, _ = np.histogram(s.dropna(), bins=np.array(ref["edges"]))
                actual = self._normalize(actual_counts)
            else:
                actual = self._cat_proportions(s, ref["categories"])
            psi = self._psi(ref["expected"], actual)
            status = "stable" if psi < STABLE else ("moderate" if psi < MODERATE else "drift")
            results.append({
                "feature": col,
                "psi": round(float(psi), 4),
                "status": status,
                "expected": [round(float(x), 4) for x in ref["expected"]],
                "actual": [round(float(x), 4) for x in actual],
            })

        drifted = [r for r in results if r.get("status") == "drift"]
        moderate = [r for r in results if r.get("status") == "moderate"]
        max_psi = max((r.get("psi") or 0) for r in results) if results else 0.0
        return {
            "rows_live": int(len(live_df)),
            "rows_reference": int(self.reference["rows"]),
            "overall_status": "drift" if drifted else ("moderate" if moderate else "stable"),
            "n_drifted": len(drifted),
            "n_moderate": len(moderate),
            "max_psi": round(float(max_psi), 4),
            "features": results,
            "drift_threshold": MODERATE,
            "moderate_threshold": STABLE,
        }

    def _normalize(self, counts) -> List[float]:
        total = float(np.sum(counts))
        if total <= 0:
            return [0.0] * len(list(counts))
        arr = [max(float(c) / total, self.min_fraction) for c in counts]
        s = float(sum(arr))
        return [a / s for a in arr]

    def _cat_proportions(self, s: pd.Series, categories: List) -> List[float]:
        props = []
        total = max(float(len(s)), 1)
        for cat in categories:
            props.append(max(float((s == cat).sum()) / total, self.min_fraction))
        total_p = float(sum(props))
        return [p / total_p for p in props]

    @staticmethod
    def _psi(expected: List[float], actual: List[float]) -> float:
        psi = 0.0
        for e, a in zip(expected, actual):
            e = max(e, 1e-6)
            a = max(a, 1e-6)
            psi += (a - e) * np.log(a / e)
        return float(psi)