"""Main ML Agent orchestrator."""
import json
from typing import Any, Dict, List, Optional, Union
import pandas as pd
from .database import DatabaseProcessor
from .model_selector import ModelSelector
from .predictor import Predictor
from .analyzer import DataAnalyzer
from .preprocessor import DataPreprocessor
from .llm_advisor import LLMAdvisor
from .relational import RelationalFeatureSynthesizer
from .experiments import ExperimentTracker
from .sql_validation import SQLValidator
from .anomaly import AnomalyDetector
from .model_monitor import ModelMonitor


class MLAgent:
    """
    An intelligent agent that:
    1. Processes SQL databases (potentially with multiple tables)
    2. Determines the best machine learning model for the data
    3. Conducts analysis and produces predictions per user request
    4. Uses a local LLM (LM Studio) for natural-language insights
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
        self.synthesizer = RelationalFeatureSynthesizer(self.db)
        self.experiments = ExperimentTracker()
        self.validator = SQLValidator(self.db)
        self.anomaly = AnomalyDetector(random_state=random_state)
        self.monitor = ModelMonitor()
        self.llm: Optional[LLMAdvisor] = None

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

    def validate_sql(self, query: str) -> Dict[str, Any]:
        """Validate a SQL statement against the schema (read-only safety)."""
        return self.validator.validate(query, read_only=self.db.read_only)

    # ========== SQL-specific helpers (column typing / sampling / joins) ==========

    def get_column_types(self, table: str, sample_limit: int = 1000) -> List[Dict[str, Any]]:
        """Return semantic column types (id/fk/date/boolean/numeric/...) for a table."""
        return self.db.get_column_types(table, sample_limit=sample_limit)

    def load_table_sample(
        self, table: str, fraction: float = 0.1, columns: Optional[List[str]] = None,
        limit: Optional[int] = None, method: str = "auto",
    ) -> pd.DataFrame:
        """Load a random sample of a table, sampled inside the DB."""
        df = self.db.load_table_sample(
            table, fraction=fraction, columns=columns, limit=limit, method=method
        )
        self.current_table = table
        self.current_df = df
        return df

    def load_auto_join(self, tables: List[str], join_type: str = "inner") -> pd.DataFrame:
        """Load data by auto-joining related tables via FK metadata."""
        df = self.db.load_auto_join(tables, join_type)
        self.current_table = "+".join(tables)
        self.current_df = df
        return df

    def get_relationships(self) -> List[Dict[str, Any]]:
        """Return FK relationships across all tables (for ER diagrams / query builder)."""
        rels = []
        for t in self.list_tables():
            for fk in self.db.get_foreign_keys(t):
                rels.append({
                    "from_table": t,
                    "from_columns": fk.get("constrained_columns", []),
                    "to_table": fk.get("referred_table"),
                    "to_columns": fk.get("referred_columns", []),
                })
        return rels

    # ========== Relational deep-feature synthesis (Feature: deep features) ==========

    def synthesize_features(
        self, base_table: str, max_rows=None, include_counts=True, include_aggregates=True
    ) -> pd.DataFrame:
        """Generate deep features by aggregating related tables, and set as current data."""
        result = self.synthesizer.synthesize(
            base_table, max_rows=max_rows,
            include_counts=include_counts, include_aggregates=include_aggregates,
        )
        self.current_table = base_table
        self.current_df = result["data"]
        self._last_features = result["features"]
        self._last_feature_joins = result["joins"]
        return self.current_df

    def get_feature_synthesizer_summary(self) -> Dict[str, Any]:
        return {
            "features": getattr(self, "_last_features", []),
            "joins": getattr(self, "_last_feature_joins", []),
        }

    # ========== Saved query library (Feature 2) ==========

    def save_query(self, name: str, query: str, description: str = "") -> str:
        return self.db.save_query(name, query, description)

    def list_queries(self) -> List[Dict[str, Any]]:
        return self.db.list_queries()

    def get_query(self, name: str) -> Optional[Dict[str, Any]]:
        return self.db.get_query(name)

    def delete_query(self, name: str) -> bool:
        return self.db.delete_query(name)

    # ========== Schema drift profiling (Feature 7) ==========

    def set_profile_store(self, path: str) -> None:
        self.db.set_profile_store(path)

    def capture_profile(self, table: Optional[str] = None, name: str = "") -> Dict[str, Any]:
        return self.db.capture_profile(table=table, name=name)

    def list_profiles(self) -> List[str]:
        return self.db.list_profiles()

    def compare_profiles(self, a: str, b: str) -> Dict[str, Any]:
        return self.db.compare_profiles(a, b)

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

    def save_preprocessed_db(
        self,
        output_path: str,
        table_name: Optional[str] = None,
        include_original_tables: bool = False,
    ) -> str:
        """
        Save the preprocessed dataset as a new SQL database, leaving the original untouched.

        Args:
            output_path: Path for the new database file (e.g., "preprocessed.db").
            table_name: Name for the table in the new database.
                Defaults to the original table name or "preprocessed_data".
            include_original_tables: If True, also copy all original tables from the
                source database into the new database alongside the preprocessed data.

        Returns:
            Path to the created database file.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")

        # Determine table name
        if table_name is None:
            table_name = self.current_table or "preprocessed_data"
            # Clean up query-based table names
            if table_name.startswith("query:"):
                table_name = "preprocessed_data"

        # Normalize output path to SQLite
        if not output_path.endswith((".db", ".sqlite", ".sqlite3")):
            output_path = f"{output_path}.db"

        import sqlite3

        conn = sqlite3.connect(output_path)
        try:
            # Write the preprocessed dataset
            self.current_df.to_sql(table_name, conn, index=False, if_exists="replace")

            # Optionally copy original tables
            if include_original_tables and self.db.engine is not None:
                for orig_table in self.db.get_tables():
                    if orig_table == table_name:
                        continue
                    orig_df = self.db.load_table(orig_table)
                    orig_df.to_sql(orig_table, conn, index=False, if_exists="replace")
        finally:
            conn.close()

        return output_path

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
        elif analysis_type == "health":
            return self.analyzer.get_data_health_report()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

    def detect_anomalies(
        self,
        table: Optional[str] = None,
        method: str = "isolation_forest",
        contamination: float = 0.1,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run anomaly detection on a table (or the loaded dataset).

        If table is given, it is loaded first. Returns a dict from
        AnomalyDetector with score/flag columns, importance, and drivers.
        """
        detector = self.anomaly
        if table:
            df = self.load_table(table)
        else:
            if self.current_df is None:
                raise RuntimeError("No data loaded. Pass a table name or load data first.")
            df = self.current_df
        return detector.detect(
            df, method=method, contamination=contamination, features=features,
        )

    def capture_monitor_reference(
        self, feature_columns: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Capture a reference feature distribution from the loaded dataset
        (the training data) for drift monitoring."""
        if self.current_df is None:
            raise RuntimeError("No data loaded. Load the training dataset first.")
        use = feature_columns or list(self.current_df.columns)
        return self.monitor.capture_reference(self.current_df, feature_columns=use)

    def monitor_live(self, table: str, feature_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compare a live table's feature distributions against the reference,
        computing per-feature PSI drift."""
        if self.monitor.reference is None:
            raise RuntimeError(
                "No reference captured yet. Call capture_monitor_reference() after loading training data."
            )
        live = self.db.load_table(table)
        return self.monitor.monitor(live)

    # ========== Model Training ==========

    def train(
        self,
        target_column: str,
        task_type: Optional[str] = None,
        table_name: Optional[str] = None,
        progress_callback: Optional[Any] = None,
        should_stop: Optional[Any] = None,
        tuning: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the best model for the given target column.

        Args:
            target_column: Column to predict.
            task_type: "regression", "classification", or None for auto-detection.
            table_name: Table to use (if not already loaded).
            progress_callback: Optional callback invoked as models are evaluated.
            should_stop: Optional callable returning True to cancel training.
            tuning: "off", "quick", or "full" — hyperparameter-tune the best model.

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

        data_source = table_name or self.current_table or "loaded_data"
        eid = self.experiments.start(
            data_source=data_source,
            target_column=target_column,
            task_type=task_type or self.task_type,
            tuning=tuning,
            meta={"rows": int(len(self.current_df)), "columns": list(self.current_df.columns)},
        )
        try:
            results = self.model_selector.select(
                self.current_df,
                progress_callback=progress_callback,
                should_stop=should_stop,
                tuning=tuning,
            )
        except Exception as e:
            self.experiments.fail(eid, str(e))
            raise
        self.experiments.finish(eid, training_results=results)
        results["experiment"] = eid
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

    def explain_prediction(self, data: Dict[str, Any], top_n: int = 5) -> Dict[str, Any]:
        """Explain a single prediction by feature contributions."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.predictor.explain_prediction(data, top_n=top_n)

    def what_if(self, data: Dict[str, Any], feature: str, value: Any) -> Dict[str, Any]:
        """Re-predict after changing one feature (perturbation analysis)."""
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")
        return self.predictor.what_if(data, feature, value)

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

    # ========== LLM Advisor Operations ==========

    def enable_llm(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: Optional[str] = None,
        timeout: int = 60,
    ) -> None:
        """
        Enable the LLM advisor backed by LM Studio.

        Args:
            base_url: LM Studio server URL (default: http://localhost:1234/v1).
            model: Optional model name to use.
            timeout: Request timeout in seconds.
        """
        self.llm = LLMAdvisor(
            base_url=base_url,
            model=model,
            timeout=timeout,
        )

    def llm_check(self) -> Dict[str, Any]:
        """Check LLM advisor availability. Returns dict with 'available' and 'detail'."""
        if self.llm is None:
            return {
                "available": False,
                "detail": "LLM advisor not enabled.",
            }
        available, detail = self.llm.check_connection()
        return {"available": available, "detail": detail}

    def llm_generate_sql(self, question: str) -> Dict[str, Any]:
        """
        Convert a natural-language question into SQL using the local LLM.

        Requires an active database connection.
        """
        if self.llm is None:
            self.enable_llm()
        overview = self.db.get_database_overview()
        result = self.llm.generate_sql(question, overview["tables"])
        if result.get("sql"):
            result["validation"] = self.validate_sql(result["sql"])
        return result

    def llm_suggest_target(self, preferred: Optional[str] = None) -> Dict[str, Any]:
        """
        Recommend a target column (and task type) using the LLM.

        Falls back to heuristics if the LLM server is unreachable.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")
        if self.llm is None:
            self.enable_llm()
        return self.llm.suggest_target(self.current_df, preferred=preferred)

    def llm_suggest_preprocessing(self) -> Dict[str, Any]:
        """
        Recommend a preprocessing operation chain using the LLM.

        Falls back to heuristics if the LLM server is unreachable.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")
        if self.llm is None:
            self.enable_llm()
        return self.llm.suggest_preprocessing(
            self.current_df,
            target_column=self.target_column,
        )

    def llm_explain_results(self, results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Produce a plain-language interpretation of model training results."""
        if self.llm is None:
            self.enable_llm()
        if results is None:
            if self.model_selector is None:
                raise RuntimeError("No training results available. Call train() first.")
            results = self.model_selector.get_result_summary()
        return self.llm.explain_results(results)

    def llm_explain_predictions(
        self,
        predictions: pd.DataFrame,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Explain prediction rows in plain language using the LLM."""
        if self.llm is None:
            self.enable_llm()
        model_info = None
        if self.predictor is not None:
            model_info = self.predictor.get_model_info()
        return self.llm.explain_predictions(
            predictions,
            model_info=model_info,
            top_n=top_n,
        )

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