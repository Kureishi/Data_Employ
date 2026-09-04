"""Anomaly detection for tables loaded into the ML Agent.

Runs an unsupervised anomaly scorer (Isolation Forest by default) over the
numeric columns and reports which features push each flagged row away from
the bulk of the data. Also surfaces global feature importance from the
forest so users can see which columns define "normal."
"""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """Detect anomalies in a DataFrame and explain their drivers."""

    SCORE_COL = "anomaly_score"
    FLAG_COL = "is_anomaly"

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.feature_importance: Optional[pd.Series] = None

    def detect(
        self,
        df: pd.DataFrame,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        features: Optional[List[str]] = None,
        random_state: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Score rows for anomalousness and annotate the DataFrame.

        Returns:
            {"data": [...], "features": [...], "n_anomalies": int,
             "contamination": float, "importance": [...],
             "top_drivers": {...}, "summary": {...}}
        """
        rs = random_state if random_state is not None else self.random_state
        if df is None or len(df) == 0:
            raise RuntimeError("No data available to scan for anomalies.")

        work = df.copy()
        if features:
            use = [f for f in features if f in work.columns]
        else:
            use = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c].dtype)]
            use = [c for c in use if work[c].notna().sum() > 0 and work[c].nunique() > 1]

        if not use:
            raise RuntimeError(
                "No usable numeric feature columns found for anomaly detection. "
                "Pass an explicit feature list or ensure at least two numeric columns exist."
            )

        model_data = work[use].apply(pd.to_numeric, errors="coerce").fillna(work[use].median())

        if method != "isolation_forest":
            raise ValueError(f"Unsupported anomaly method: {method} (supported: isolation_forest)")

        iso = IsolationForest(
            contamination=float(contamination),
            random_state=rs,
            n_estimators=100,
        )
        iso.fit(model_data)
        decision = iso.decision_function(model_data)  # higher = more normal
        anomaliness = -decision                      # higher = more anomalous
        lo, hi = float(np.min(anomaliness)), float(np.max(anomaliness))
        score01 = np.where(
            hi > lo,
            (anomaliness - lo) / (hi - lo),
            np.zeros(len(df)),
        )
        pred = iso.predict(model_data)  # -1 = outlier, 1 = inlier

        result = work.copy()
        result[self.SCORE_COL] = np.round(score01, 4)
        result[self.FLAG_COL] = (pred == -1)

        # Which features deviate most for each flagged row (robust z-score)
        z = model_data.sub(model_data.mean()).div(model_data.std().replace(0, np.nan)).abs()
        # Global importance = mean absolute standardized deviation per feature
        z_mean = z.mean().fillna(0.0)
        tot = float(z_mean.sum()) or 1.0
        self.feature_importance = (z_mean / tot).round(5)
        top_drivers: Dict[str, List[str]] = {}
        if pred[pred == -1].size:
            zarr = z.to_numpy()
            for i in np.where(pred == -1)[0][:200]:
                order = np.argsort(zarr[i])[::-1]
                top = [
                    {"feature": use[j], "abs_z": round(float(zarr[i][j]), 3)}
                    for j in order[:5] if not np.isnan(zarr[i][j])
                ]
                top_drivers[str(int(i))] = top

        n_anom = int(pred[pred == -1].size)
        return {
            "data": _jsonify(result),
            "features": use,
            "n_anomalies": n_anom,
            "total": int(len(df)),
            "contamination": float(contamination),
            "importance": [
                {"feature": f, "importance": round(float(v), 4)}
                for f, v in self.feature_importance.sort_values(ascending=False).items()
            ],
            "top_drivers": top_drivers,
            "summary": {
                "anomaly_rate": round(float(n_anom / max(len(df), 1)), 4),
                "score_column": self.SCORE_COL,
                "flag_column": self.FLAG_COL,
            },
        }


def _jsonify(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame to JSON-ready records (handles NaN & numpy types)."""
    records = df.to_dict(orient="records")
    out = []
    for r in records:
        row = {}
        for k, v in r.items():
            if isinstance(v, np.integer):
                v = int(v)
            elif isinstance(v, np.floating):
                v = float(v)
            try:
                if pd.isna(v):
                    v = None
            except Exception:
                pass
            row[str(k)] = v
        out.append(row)
    return out