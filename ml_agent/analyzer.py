"""Data analysis module for the ML Agent."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import json


class DataAnalyzer:
    """Performs statistical analysis and insights on data."""

    def __init__(self, df: pd.DataFrame, target_column: Optional[str] = None):
        self.df = df
        self.target_column = target_column

    def get_basic_stats(self) -> Dict[str, Any]:
        """Get basic statistics for the DataFrame."""
        numeric_cols = [
            c for c in self.df.columns
            if pd.api.types.is_numeric_dtype(self.df[c].dtype)
        ]
        categorical_cols = [
            c for c in self.df.columns
            if not pd.api.types.is_numeric_dtype(self.df[c].dtype)
        ]

        stats = {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "missing_values": int(self.df.isna().sum().sum()),
            "duplicate_rows": int(self.df.duplicated().sum()),
        }

        if numeric_cols:
            stats["numeric_summary"] = self.df[numeric_cols].describe().to_dict()

        return stats

    def get_correlations(self) -> Dict[str, Any]:
        """Get correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}

        corr = numeric_df.corr()
        corr_dict = {}
        for col in corr.columns:
            corr_dict[col] = {
                c: round(float(v), 4)
                for c, v in corr[col].items()
            }
        return corr_dict

    def get_column_insights(self) -> List[Dict[str, Any]]:
        """Get insights for each column."""
        insights = []
        for col in self.df.columns:
            col_data = self.df[col]
            info = {"column": col, "dtype": str(col_data.dtype), "null_count": int(col_data.isna().sum())}

            if pd.api.types.is_numeric_dtype(col_data.dtype):
                info.update({
                    "mean": round(float(col_data.mean()), 4),
                    "std": round(float(col_data.std()), 4),
                    "min": round(float(col_data.min()), 4),
                    "max": round(float(col_data.max()), 4),
                    "skew": round(float(col_data.skew()), 4),
                })
            else:
                info.update({
                    "unique_values": int(col_data.nunique()),
                    "top_values": col_data.value_counts().head(5).to_dict(),
                })
            insights.append(info)
        return insights

    def get_target_analysis(self) -> Dict[str, Any]:
        """Analyze the target column relationship with features."""
        if not self.target_column or self.target_column not in self.df.columns:
            return {"error": "Target column not specified or not found"}

        target = self.df[self.target_column]
        result = {
            "target_column": self.target_column,
            "dtype": str(target.dtype),
            "null_count": int(target.isna().sum()),
        }

        if pd.api.types.is_numeric_dtype(target.dtype):
            result["distribution"] = {
                "mean": round(float(target.mean()), 4),
                "median": round(float(target.median()), 4),
                "std": round(float(target.std()), 4),
                "min": round(float(target.min()), 4),
                "max": round(float(target.max()), 4),
                "skew": round(float(target.skew()), 4),
            }
        else:
            vc = target.value_counts()
            result["distribution"] = {
                str(k): int(v) for k, v in vc.head(10).items()
            }
            result["unique_classes"] = int(target.nunique())

        # Correlations with numeric features
        numeric_target = pd.to_numeric(target, errors="coerce")
        if numeric_target.notna().sum() > 10:
            corr_with_target = {}
            for col in self.df.select_dtypes(include=[np.number]).columns:
                if col != self.target_column:
                    corr = self.df[col].corr(numeric_target)
                    if not np.isnan(corr):
                        corr_with_target[col] = round(float(corr), 4)
            result["correlations_with_numeric_features"] = dict(
                sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True)
            )

        return result

    def get_data_health_report(self) -> Dict[str, Any]:
        """Produce a data-quality / health report with warnings and per-column stats."""
        df = self.df
        report: Dict[str, Any] = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "missing_total": int(df.isna().sum().sum()),
            "missing_cells_pct": round(float(df.isna().mean().mean() * 100), 2),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_pct": round(float(df.duplicated().mean() * 100), 2),
            "column_health": [],
            "warnings": [],
            "class_balance": None,
        }

        for col in df.columns:
            s = df[col]
            missing = int(s.isna().sum())
            missing_pct = round(float(s.isna().mean() * 100), 2)
            unique = int(s.nunique())
            info = {
                "column": col,
                "dtype": str(s.dtype),
                "null_count": missing,
                "null_pct": missing_pct,
                "unique_values": unique,
                "cardinality": (round(unique / len(s), 4) if len(s) else 0),
            }

            if not pd.api.types.is_numeric_dtype(s.dtype):
                info["top_values"] = s.value_counts().head(5).to_dict()

            # Warn on high missing
            if missing > 0:
                report["warnings"].append(
                    f"Column '{col}' has {missing} missing value(s) ({missing_pct}%)."
                )
            # Warn on high cardinality object column
            if (not pd.api.types.is_numeric_dtype(s.dtype)) and unique > 50:
                report["warnings"].append(
                    f"Column '{col}' is high-cardinality ({unique} unique values) — check if it's an ID."
                )
            # Constant column
            if unique == 1:
                report["warnings"].append(
                    f"Column '{col}' is constant — it provides no predictive signal."
                )

            report["column_health"].append(info)

        # Class balance for a categorical target
        if self.target_column and self.target_column in df.columns:
            target = df[self.target_column]
            if not pd.api.types.is_numeric_dtype(target.dtype) or (
                pd.api.types.is_numeric_dtype(target.dtype) and target.nunique() <= 10
            ):
                counts = target.value_counts()
                proportions = (counts / len(target)).round(4).to_dict()
                report["class_balance"] = {
                    str(k): {"count": int(v), "proportion": float(proportions.get(k, 0))}
                    for k, v in counts.items()
                }
                if counts.min() < len(target) * 0.1 and counts.max() > 0:
                    report["warnings"].append(
                        "Imbalanced target detected — consider using class weighting or stratified sampling."
                    )

        if report["missing_cells_pct"] > 5:
            report["warnings"].append(
                f"Overall missing-cell rate is {report['missing_cells_pct']}% — consider imputation."
            )
        if report["duplicate_rows"] > 0:
            report["warnings"].append(
                f"Found {report['duplicate_rows']} duplicate rows ({report['duplicate_pct']}%)."
            )

        report["status"] = "warning" if report["warnings"] else "ok"
        return report

    def get_summary_report(self) -> Dict[str, Any]:
        """Get a comprehensive summary report."""
        return {
            "basic_stats": self.get_basic_stats(),
            "correlations": self.get_correlations(),
            "column_insights": self.get_column_insights(),
            "target_analysis": self.get_target_analysis(),
        }