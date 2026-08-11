"""Prediction module for the ML Agent."""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from .model_selector import ModelSelector
from .analyzer import DataAnalyzer


class Predictor:
    """Makes predictions and provides analysis using the trained model."""

    def __init__(self, model_selector: ModelSelector):
        self.model = model_selector

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            X: DataFrame with the same feature columns used for training.

        Returns:
            DataFrame with original data plus prediction column(s).
        """
        if self.model.pipeline is None:
            raise RuntimeError("Model not trained yet. Call select() on ModelSelector first.")

        result = X.copy()
        pred = self.model.predict(X)

        if self.model.task_type == "classification":
            # Map numeric predictions back to original class labels if encoded
            if self.model.label_encoder is not None:
                labels = self.model.label_encoder.inverse_transform(pred.astype(int))
                result["prediction"] = labels
            else:
                result["prediction"] = pred

            # Add probabilities
            try:
                proba = self.model.predict_proba(X)
                # Use the model's actual trained classes
                if self.model.label_encoder is not None:
                    classes = self.model.label_encoder.classes_
                else:
                    classes = self.model.pipeline.named_steps["model"].classes_
                for i, cls in enumerate(classes):
                    result[f"prob_{cls}"] = proba[:, i]
            except Exception:
                pass
        else:
            result["prediction"] = pred

        return result

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a prediction for a single record.

        Args:
            data: Dictionary mapping feature names to values.

        Returns:
            Dictionary with prediction result.
        """
        df = pd.DataFrame([data])
        result = self.predict(df)
        row = result.iloc[0].to_dict()

        pred = row.pop("prediction")
        output = {"prediction": pred}
        if self.model.label_encoder is not None:
            # Convert back
            try:
                sample = pd.DataFrame([data])
                numeric_pred = self.model.predict(sample)[0]
                output["prediction_label"] = self.model.label_encoder.inverse_transform(
                    [int(numeric_pred)]
                )[0]
            except Exception:
                pass

        # Add probabilities if present
        probs = {k: round(float(v), 4) for k, v in row.items() if k.startswith("prob_")}
        if probs:
            output["probabilities"] = probs

        return output

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the trained model."""
        return self.model.feature_importance

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        return {
            "task_type": self.model.task_type,
            "target_column": self.model.target_column,
            "best_model": self.model.best_model_name,
            "best_cv_score": self.model.best_score,
            "test_metrics": self.model.metrics,
            "feature_columns": self.model.feature_columns,
            "class_mapping": (
                dict(enumerate(self.model.label_encoder.classes_))
                if self.model.label_encoder else None
            ),
        }

    def generate_analysis(
        self,
        df: pd.DataFrame,
        target_column: str,
        analysis_type: str = "summary",
    ) -> Dict[str, Any]:
        """
        Generate analysis report on the data.

        Args:
            df: Input DataFrame.
            target_column: Target column name.
            analysis_type: "summary", "correlations", "insights", or "target".

        Returns:
            Analysis report dictionary.
        """
        analyzer = DataAnalyzer(df, target_column)
        if analysis_type == "summary":
            return analyzer.get_summary_report()
        elif analysis_type == "correlations":
            return analyzer.get_correlations()
        elif analysis_type == "insights":
            return analyzer.get_column_insights()
        elif analysis_type == "target":
            return analyzer.get_target_analysis()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    def format_prediction_output(self, df: pd.DataFrame, predictions: pd.DataFrame) -> str:
        """Format predictions as a readable string."""
        rows = []
        headers = ["#"] + list(predictions.columns)
        for i, (_, row) in enumerate(predictions.iterrows(), 1):
            formatted = [str(i)]
            for col in predictions.columns:
                val = row[col]
                if isinstance(val, float):
                    formatted.append(f"{val:.4f}")
                else:
                    formatted.append(str(val))
            rows.append(formatted)
        return self._format_table(headers, rows)

    def _format_table(self, headers: List[str], rows: List[List[str]]) -> str:
        """Format data as a simple ASCII table."""
        widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                widths[i] = max(widths[i], len(cell))

        lines = []
        header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-+-".join("-" * w for w in widths))
        for row in rows:
            lines.append(" | ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
        return "\n".join(lines)