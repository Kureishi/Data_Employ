"""Main ML Agent orchestrator."""
import json
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from .database import DatabaseProcessor
from .model_selector import ModelSelector
from .predictor import Predictor
from .analyzer import DataAnalyzer
from .preprocessor import DataPreprocessor


class MLAgent:
    """
    An intelligent agent that:
    1. Processes SQL databases (potentially with multiple tables)
    2. Determines the best machine learning model for the data
    3. Conducts analysis and produces predictions per user request
    """

    def __init__(
        self,
        connection_string: Optional[str] = None,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize the ML Agent.

        Args:
            connection_string: SQLAlchemy connection string or SQLite file path.
            task_type: "regression", "classification", or None for auto-detection.
            test_size: Fraction of data for test set.
            cv_folds: Number of cross-validation folds.
            random_state: Random seed for reproducibility.
        """
        self.db = DatabaseProcessor(connection_string)
        self.task_type = task_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.current_table: Optional[str] = None
        self.current_df: Optional[pd.DataFrame] = None
        self.model_selector: Optional[ModelSelector] = None
        self.predictor: Optional[Predictor] = None
        self.analyzer: Optional[DataAnalyzer] = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.target_column: Optional[str] = None

    # ========== Database Operations ==========

    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        return self.db.get_tables()

    def get_database_overview(self) -> Dict[str, Any]:
        """Get a comprehensive overview of the database."""
        return self.db.get_database_overview()

    def get_table_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all tables."""
        return self.db.get_table_summary()

    def load_table(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Load a table into memory."""
        self.current_table = table_name
        self.current_df = self.db.load_table(table_name, limit)
        return self.current_df

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a custom SQL query."""
        return self.db.execute_query(query)

    def load_query_as_data(self, query: str) -> pd.DataFrame:
        """Load query results as the current working dataset."""
        self.current_df = self.db.execute_query(query)
        self.current_table = f"query: {query[:50]}..."
        return self.current_df

    # ========== Preprocessing Operations ==========

    def preprocess(self, df: Optional[pd.DataFrame] = None) -> DataPreprocessor:
        """
        Get a preprocessor for the current dataset.

        Args:
            df: Optional DataFrame to preprocess (default: current dataset).

        Returns:
            DataPreprocessor instance.
        """
        data = df if df is not None else self.current_df
        if data is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")
        self.preprocessor = DataPreprocessor(data)
        return self.preprocessor

    def apply_preprocessing(self, operations: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Apply a list of preprocessing operations to the current dataset.

        Args:
            operations: List of dicts with 'op' and parameters.
                Example: [{"op": "dropna"}, {"op": "fillna", "method": "median"}]

        Returns:
            Preprocessed DataFrame.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")

        pre = self.preprocess(self.current_df)
        for op in operations:
            op_copy = dict(op)
            op_name = op_copy.pop("op")
            method = getattr(pre, op_name, None)
            if method is None:
                raise ValueError(f"Unknown preprocessing operation: {op_name}")
            method(**op_copy)

        self.current_df = pre.get_data()
        return self.current_df

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get a summary of preprocessing operations performed."""
        if self.preprocessor is None:
            return {"operations": [], "total_operations": 0}
        return self.preprocessor.get_summary()

    # ========== Analysis Operations ==========

    def analyze(self, target_column: Optional[str] = None, analysis_type: str = "summary") -> Dict[str, Any]:
        """
        Perform data analysis on the current dataset.

        Args:
            target_column: Target column for analysis (optional).
            analysis_type: "summary", "correlations", "insights", or "target".

        Returns:
            Analysis report.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")

        self.analyzer = DataAnalyzer(self.current_df, target_column or self.target_column)
        if analysis_type == "summary":
            return self.analyzer.get_summary_report()
        elif analysis_type == "correlations":
            return self.analyzer.get_correlations()
        elif analysis_type == "insights":
            return self.analyzer.get_column_insights()
        elif analysis_type == "target":
            return self.analyzer.get_target_analysis()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    # ========== Model Training ==========

    def train(
        self,
        target_column: str,
        task_type: Optional[str] = None,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the best model for the given target column.

        Args:
            target_column: Column to predict.
            task_type: "regression", "classification", or None for auto-detection.
            table_name: Table to use (if not already loaded).

        Returns:
            Model selection results with best model and metrics.
        """
        if table_name:
            self.load_table(table_name)

        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or provide table_name.")

        if target_column not in self.current_df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        self.target_column = target_column
        self.model_selector = ModelSelector(
            target_column=target_column,
            task_type=task_type or self.task_type,
            test_size=self.test_size,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
        )

        results = self.model_selector.select(self.current_df)
        self.predictor = Predictor(self.model_selector)
        return results

    # ========== Prediction Operations ==========

    def predict(self, data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Make predictions on new data.

        Args:
            data: DataFrame, single dict, or list of dicts with feature values.

        Returns:
            DataFrame with predictions.
        """
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data

        return self.predictor.predict(df)

    def predict_single(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a prediction for a single record."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.predictor.predict_single(data)

    # ========== Model Management ==========

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.predictor.get_model_info()

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Get feature importance from the trained model."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.predictor.get_feature_importance()

    def save_model(self, path: str) -> None:
        """Save the trained model to disk."""
        if self.model_selector is None:
            raise RuntimeError("Model not trained. Call train() first.")
        self.model_selector.save_model(path)

    def load_model(self, path: str) -> None:
        """Load a trained model from disk."""
        self.model_selector = ModelSelector(
            target_column="", task_type=None,
            test_size=self.test_size, cv_folds=self.cv_folds,
            random_state=self.random_state,
        )
        self.model_selector.load_model(path)
        self.target_column = self.model_selector.target_column
        self.predictor = Predictor(self.model_selector)

    # ========== Utility ==========

    def format_results(self, results: Dict[str, Any]) -> str:
        """Format model selection results as a readable string."""
        lines = []
        lines.append("=" * 60)
        lines.append("MODEL SELECTION RESULTS")
        lines.append("=" * 60)
        lines.append(f"Task Type: {results['task_type']}")
        lines.append(f"Target Column: {results['target_column']}")
        lines.append(f"Best Model: {results['best_model']}")
        lines.append(f"Best CV Score: {results['best_cv_score']}")
        lines.append("")
        lines.append("Model Scores (sorted):")
        for name, score in results["model_scores"].items():
            lines.append(f"  {name}: {score}")
        lines.append("")
        lines.append("Test Metrics:")
        for metric, value in results["test_metrics"].items():
            lines.append(f"  {metric}: {value:.4f}")
        lines.append("")
        lines.append(f"Feature Columns: {', '.join(results['feature_columns'])}")
        if results.get("class_mapping"):
            lines.append(f"Class Mapping: {results['class_mapping']}")
        if results.get("feature_importance") is not None and len(results["feature_importance"]) > 0:
            lines.append("")
            lines.append("Top 10 Feature Importances:")
            imp = results["feature_importance"].head(10)
            for _, row in imp.iterrows():
                lines.append(f"  {row['feature']}: {row['importance']:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def close(self) -> None:
        """Close the database connection."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()