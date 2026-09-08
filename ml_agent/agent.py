"""Main ML Agent orchestrator."""
import hashlib
import json
import os
import threading
import time
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
from .recipes import RecipeStore
from .snapshots import SnapshotStore


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
        self.models_dir: Optional[str] = None
        self.validator = SQLValidator(self.db)
        self.anomaly = AnomalyDetector(random_state=random_state)
        self.monitor = ModelMonitor()
        self.n_jobs: int = 1
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()
        self._cache_dir: Optional[str] = None
        self._suggested_targets: List[Dict[str, Any]] = []
        self._cache_token: Optional[str] = None
        self.llm: Optional[LLMAdvisor] = None

        # Recipe / reproducibility state
        self.recipes = RecipeStore()
        self.snapshots = SnapshotStore()
        self._last_operations: List[Dict[str, Any]] = []
        self._last_synth_base: Optional[str] = None
        self._last_synth_counts: bool = True
        self._last_synth_aggs: bool = True

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

    def load_table(self, table_name: str, limit: Optional[int] = None,
                   offset: Optional[int] = None) -> pd.DataFrame:
        """Load a table (optionally paginated with limit/offset) into memory."""
        self.current_table = table_name
        self.current_df = self.db.load_table(table_name, limit, offset)
        self._invalidate_analysis_cache()
        return self.current_df

    def execute_query(self, query: str) -> pd.DataFrame:
        """Execute a custom SQL query."""
        return self.db.execute_query(query)

    def load_query_as_data(self, query: str) -> pd.DataFrame:
        """Load query results as the current working dataset."""
        self.current_df = self.db.execute_query(query)
        self.current_table = f"query: {query[:50]}..."
        self._invalidate_analysis_cache()
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
        self._invalidate_analysis_cache()
        return df

    def load_auto_join(self, tables: List[str], join_type: str = "inner") -> pd.DataFrame:
        """Load data by auto-joining related tables via FK metadata."""
        df = self.db.load_auto_join(tables, join_type)
        self.current_table = "+".join(tables)
        self.current_df = df
        self._invalidate_analysis_cache()
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
        self._invalidate_analysis_cache()
        self._last_features = result["features"]
        self._last_feature_joins = result["joins"]
        self._last_synth_base = base_table
        self._last_synth_counts = include_counts
        self._last_synth_aggs = include_aggregates
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

    # ========== Reusable pipeline recipes (Feature: recipe) ==========

    def set_recipe_store(self, path: str) -> None:
        self.recipes.set_store(path)

    def save_recipe(
        self,
        name: str,
        description: str = "",
        target: Optional[str] = None,
        task_type: Optional[str] = None,
        tuning: Optional[str] = None,
        n_jobs: Optional[int] = None,
        auto_prepare: bool = False,
    ) -> str:
        """Capture the current data source + preprocessing as a reusable recipe.

        The recipe stores the table (or relational synthesis config) that
        produced the current dataset, the exact preprocessing operations, and
        the training configuration, so it can be reproduced later.
        """
        synth = None
        if getattr(self, "_last_synth_base", None):
            synth = {
                "base_table": self._last_synth_base,
                "include_counts": self._last_synth_counts,
                "include_aggregates": self._last_synth_aggs,
            }
        if synth is None and not self.current_table:
            raise RuntimeError(
                "No data source available. Load a table or synthesize features first."
            )
        source = synth["base_table"] if synth else self.current_table
        recipe = {
            "name": name or time.strftime("recipe_%Y%m%d_%H%M%S"),
            "description": description,
            "data_source": source,
            "synth": synth,
            "preprocessing": getattr(self, "_last_operations", []),
            "target": target or self.target_column,
            "task_type": task_type or self.task_type,
            "tuning": tuning,
            "n_jobs": n_jobs,
            "auto_prepare": bool(auto_prepare),
        }
        return self.recipes.save(recipe)

    def list_recipes(self) -> List[Dict[str, Any]]:
        return self.recipes.list()

    def get_recipe(self, name: str) -> Optional[Dict[str, Any]]:
        return self.recipes.get(name)

    def delete_recipe(self, name: str) -> bool:
        return self.recipes.delete(name)

    def apply_recipe(
        self,
        name: str,
        target: Optional[str] = None,
        tuning: Optional[str] = None,
        n_jobs: Optional[int] = None,
        should_stop: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Reproduce a saved recipe on its data source and train a fresh model.

        Reloads the source (table or relational synthesis), re-applies the
        stored preprocessing operations, optionally auto-prepares, and trains
        with the recipe's (or overridden) target/tuning.
        """
        recipe = self.recipes.get(name)
        if recipe is None:
            raise ValueError(f"Recipe '{name}' not found. Save it first.")

        synth = recipe.get("synth")
        if synth and synth.get("base_table"):
            self.synthesize_features(
                synth["base_table"],
                include_counts=synth.get("include_counts", True),
                include_aggregates=synth.get("include_aggregates", True),
            )
        elif recipe.get("data_source"):
            self.load_table(recipe["data_source"])
        else:
            raise RuntimeError("Recipe has no data source.")

        operations = recipe.get("preprocessing") or []
        if operations:
            self.apply_preprocessing(operations)

        tgt = target or recipe.get("target")
        if not tgt:
            raise RuntimeError(
                "Recipe has no target column. Provide one via 'target'."
            )
        if recipe.get("auto_prepare"):
            self.auto_prepare(tgt)

        tuning_val = tuning if tuning is not None else recipe.get("tuning")
        n_jobs_val = n_jobs if n_jobs is not None else recipe.get("n_jobs")
        results = self.train(
            target_column=tgt,
            task_type=recipe.get("task_type"),
            tuning=tuning_val,
            n_jobs=n_jobs_val,
            should_stop=should_stop,
        )
        results["recipe"] = name
        return results

    # ========== Named result snapshots & diff (Feature: snapshot) ==========

    def set_snapshot_store(self, path: str) -> None:
        self.snapshots.set_store(path)

    def save_snapshot(
        self, name: str, kind: str, payload: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a JSON-serializable result snapshot under a name."""
        return self.snapshots.save(name, kind, payload, meta=meta)

    def list_snapshots(self) -> List[Dict[str, Any]]:
        return self.snapshots.list()

    def get_snapshot(self, name: str) -> Optional[Dict[str, Any]]:
        return self.snapshots.get(name)

    def delete_snapshot(self, name: str) -> bool:
        return self.snapshots.delete(name)

    def diff_snapshots(self, a: str, b: str) -> Dict[str, Any]:
        return self.snapshots.diff(a, b)

    # ========== Feature health / smarter filtering (Feature: feature-health) ==========

    def check_feature_health(
        self,
        target_column: Optional[str] = None,
        corr_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """Report feature-filtering health for the loaded dataset.

        Detects near-constant columns, high-cardinality ID/free-text columns,
        and highly-collinear numeric feature pairs (multicollinearity) so the
        user can drop uninformative / redundant features before training.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() first.")
        df = self.current_df
        target = target_column or self.target_column
        rows = int(len(df))
        if rows == 0:
            return {"rows": 0, "near_constant": [], "high_cardinality": [],
                    "multicollinearity": [], "warnings": []}

        near_constant = []
        high_cardinality = []
        for col in df.columns:
            if col == target:
                continue
            s = df[col]
            nunique = int(s.nunique(dropna=False))
            if nunique <= 1:
                near_constant.append({"column": col, "unique_values": nunique,
                                      "reason": "constant"})
            elif 2 <= nunique <= 5 and rows >= 50:
                top = float(s.value_counts(normalize=True, dropna=False).iloc[0])
                if top >= 0.99:
                    near_constant.append({"column": col, "unique_values": nunique,
                                          "reason": "near-constant", "dominance": round(top, 3)})
            if (not pd.api.types.is_numeric_dtype(s.dtype)) and nunique > max(100, int(rows * 0.95)):
                high_cardinality.append({"column": col, "unique_values": nunique})

        # Multicollinearity among candidate numeric features.
        num_cols = [c for c in df.columns if c != target
                    and pd.api.types.is_numeric_dtype(df[c].dtype)]
        conflicts = []
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    aa, bb = num_cols[i], num_cols[j]
                    c = corr.loc[aa, bb]
                    if c is None or (isinstance(c, float) and c != c):  # NaN
                        continue
                    if abs(float(c)) >= corr_threshold:
                        std_a = float(df[aa].std()) or 0.0
                        std_b = float(df[bb].std()) or 0.0
                        # Suggest dropping the lower-variance one.
                        suggested = aa if std_a <= std_b else bb
                        conflicts.append({
                            "feature_a": aa, "feature_b": bb,
                            "correlation": round(float(c), 4),
                            "suggested_drop": suggested,
                        })

        warnings = [
            f"{len(near_constant)} near-constant column(s)",
            f"{len(high_cardinality)} high-cardinality column(s)",
            f"{len(conflicts)} highly-correlated pair(s)",
        ]
        return {
            "rows": rows,
            "target_column": target,
            "near_constant": near_constant,
            "high_cardinality": high_cardinality,
            "multicollinearity": conflicts,
            "warnings": warnings,
        }

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
        ops_applied = []
        for op in operations:
            op_copy = dict(op)
            op_name = op_copy.pop("op")
            method = getattr(pre, op_name, None)
            if method is None:
                raise ValueError(f"Unknown preprocessing operation: {op_name}")
            method(**op_copy)
            ops_applied.append({"op": op_name, **op_copy})

        self._last_operations = ops_applied
        self.current_df = pre.get_data()
        self._invalidate_analysis_cache()
        return self.current_df

    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get a summary of preprocessing operations performed."""
        if self.preprocessor is None:
            return {"operations": [], "total_operations": 0}
        return self.preprocessor.get_summary()

    def auto_prepare(self, target_column: Optional[str] = None) -> Dict[str, Any]:
        """Best-effort automatic data preparation.

        Drops constant columns and high-cardinality ID/object columns (that are
        not the target) so the trainer doesn't waste time on them. Missing values
        are handled upstream by the model preprocessing pipeline.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")

        df = self.current_df
        target = target_column or self.target_column
        dropped = []
        keep = []
        row_count = len(df)

        for col in df.columns:
            s = df[col]
            nunique = s.nunique(dropna=False)
            # Constant column (no signal) — drop unless it's the target
            if nunique <= 1:
                if col != target:
                    dropped.append({"column": col, "reason": "constant"})
                    continue
            # Near-constant (one value dominates 99%+) — drop unless target
            elif 2 <= nunique <= 5 and len(df) >= 50:
                top = s.value_counts(normalize=True, dropna=False).iloc[0]
                if top >= 0.99 and col != target:
                    dropped.append({"column": col, "reason": "near-constant"})
                    continue
            # High-cardinality object column (~ID / free text) — drop unless target
            elif (not pd.api.types.is_numeric_dtype(s.dtype)) and nunique > max(100, int(len(s) * 0.95)):
                if col != target:
                    dropped.append({"column": col, "reason": "high-cardinality"})
                    continue
            keep.append(col)

        if keep != list(df.columns):
            self.current_df = df[keep]
            self._invalidate_analysis_cache()

        return {
            "prepared": True,
            "rows": row_count,
            "kept_columns": keep,
            "dropped_columns": dropped,
            "target_column": target,
        }

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

    def analyze(self, target_column: Optional[str] = None, analysis_type: str = "summary",
                use_cache: bool = True) -> Dict[str, Any]:
        """
        Perform data analysis on the current dataset.

        Args:
            target_column: Target column for analysis (optional).
            analysis_type: "summary", "correlations", "insights", "health", or "target".
            use_cache: Whether to reuse a cached result for identical inputs.

        Returns:
            Analysis report.
        """
        if self.current_df is None:
            raise RuntimeError("No data loaded. Call load_table() or load_query_as_data() first.")

        cache_key = f"{analysis_type}|{target_column}|{self._data_token()}"
        cached = self._analysis_cache_get(cache_key)
        if use_cache and cached is not None:
            return cached

        self.analyzer = DataAnalyzer(self.current_df, target_column or self.target_column)
        if analysis_type == "summary":
            result = self.analyzer.get_summary_report()
        elif analysis_type == "correlations":
            result = self.analyzer.get_correlations()
        elif analysis_type == "insights":
            result = self.analyzer.get_column_insights()
        elif analysis_type == "target":
            result = self.analyzer.get_target_analysis()
        elif analysis_type == "health":
            result = self.analyzer.get_data_health_report()
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        self._analysis_cache_set(cache_key, result)
        return result

    # -- Persistent (disk-backed) analysis cache (B3) -----------------------

    def enable_disk_cache(self, cache_dir: Optional[str] = None) -> None:
        """Point the analysis cache at a directory for persistence across
        restarts. Analysis results are cheap to recompute but expensive under
        load; persisting them (keyed by data fingerprint + analysis type) makes
        repeated deep analysis nearly free and is shared across workers."""
        from . import config as cfg  # local to avoid import cost at module load

        self._cache_dir = cache_dir or cfg.get_cache_dir()
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)

    def disable_disk_cache(self) -> None:
        self._cache_dir = None

    def _cache_path(self, cache_key: str) -> Optional[str]:
        if not self._cache_dir:
            return None
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return os.path.join(self._cache_dir, f"{digest}.json")

    def _analysis_cache_get(self, cache_key: str) -> Optional[Any]:
        with self._cache_lock:
            if cache_key in self._analysis_cache:
                return self._analysis_cache[cache_key]
        path = self._cache_path(cache_key)
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            return payload.get("result")
        except Exception:
            return None

    def _analysis_cache_set(self, cache_key: str, result: Any) -> None:
        with self._cache_lock:
            self._analysis_cache[cache_key] = result
        path = self._cache_path(cache_key)
        if not path:
            return
        try:
            import json as _json

            with open(path, "w", encoding="utf-8") as fh:
                _json.dump({"key": cache_key, "result": result}, fh, default=str)
        except Exception:
            # Disk cache is best-effort; never fail a request because of it.
            pass

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
        n_jobs: Optional[int] = None,
        early_stop: Optional[int] = None,
        X: Optional[pd.DataFrame] = None,
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
            X: Optional pre-loaded training DataFrame. When provided, training
                uses X and does NOT overwrite ``self.current_df``. This enables
                the web layer to snapshot data under a brief lock and run the
                (potentially long) model search off-lock.

        Returns:
            Model selection results with best model and metrics.
        """
        if table_name:
            self.load_table(table_name)

        df = X if X is not None else self.current_df
        if df is None:
            raise RuntimeError(
                "No data loaded. Call load_table() or provide table_name / X."
            )

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in data.")

        self.target_column = target_column
        self.model_selector = ModelSelector(
            target_column=target_column,
            task_type=task_type or self.task_type,
            test_size=self.test_size,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            n_jobs=n_jobs or self.n_jobs,
        )

        data_source = table_name or self.current_table or "loaded_data"
        eid = self.experiments.start(
            data_source=data_source,
            target_column=target_column,
            task_type=task_type or self.task_type,
            tuning=tuning,
            meta={"rows": int(len(df)), "columns": list(df.columns)},
        )
        try:
            results = self.model_selector.select(
                df,
                progress_callback=progress_callback,
                should_stop=should_stop,
                tuning=tuning,
                early_stop=early_stop,
            )
        except Exception as e:
            self.experiments.fail(eid, str(e))
            raise
        self.experiments.finish(eid, training_results=results)
        self.predictor = Predictor(self.model_selector)

        # Auto-save the trained model so a champion can be reloaded on rollback.
        model_path = self._experiment_model_path(eid)
        try:
            self.model_selector.save_model(model_path)
        except Exception:
            model_path = None
        if model_path:
            self.experiments.set_model_path(eid, model_path)
            results["model_path"] = model_path

        results["experiment"] = eid
        return results

    def _experiment_model_path(self, eid: str) -> str:
        """Return a deterministic per-experiment model save path."""
        base = self.models_dir
        if not base:
            store = getattr(self.experiments, "_store_path", None)
            if store:
                base = os.path.join(os.path.dirname(store), "models")
            else:
                base = "models"
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, f"{eid}.joblib")

    def promote_experiment(self, eid: str) -> Dict[str, Any]:
        """Promote an experiment to champion if it beats the current one, and
        reload the champion's persisted model so predictions use it."""
        decision = self.experiments.promote_if_better(eid)
        champion = decision.get("champion")
        reloaded = False
        if champion and champion.get("model_path"):
            try:
                self.load_model(champion["model_path"])
                reloaded = True
            except Exception:
                reloaded = False
        decision["reloaded"] = reloaded
        return decision

    def rollback_experiment(self) -> Dict[str, Any]:
        """Revert the champion to the previous experiment and reload its model."""
        result = self.experiments.rollback()
        champion = result.get("champion")
        reloaded = False
        if result.get("rolled_back") and champion and champion.get("model_path"):
            try:
                self.load_model(champion["model_path"])
                reloaded = True
            except Exception:
                reloaded = False
        result["reloaded"] = reloaded
        return result

    # ========== Caching / smart defaults (Tier 1) ==========

    def _invalidate_analysis_cache(self) -> None:
        """Drop cached analysis results when the working dataset changes."""
        self._analysis_cache = {}
        self._cache_token = None

    def _data_token(self) -> str:
        """A cheap fingerprint of the current dataset for cache keys."""
        if self.current_df is None:
            return "none"
        import json
        cols = [str(c) for c in self.current_df.columns]
        dtypes = [str(self.current_df[c].dtype) for c in self.current_df.columns]
        return json.dumps([len(self.current_df), cols, dtypes]) + f"|base={self.current_table}"

    def set_n_jobs(self, n_jobs: int) -> None:
        """Set the default number of parallel workers used for training/tuning."""
        try:
            self.n_jobs = max(1, int(n_jobs))
        except (TypeError, ValueError):
            self.n_jobs = 1

    def suggest_targets(self, k: int = 5, df: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """Suggest likely target columns for prediction.

        Uses heuristics (dtype, cardinality, name, missing rate) rather than the
        LLM, so it works offline and instantly. Columns that are obviously IDs,
        dates, or foreign keys are excluded.
        """
        data = df if df is not None else self.current_df
        if data is None:
            raise RuntimeError("No data loaded. Call load_table() or provide a DataFrame.")

        token = self._data_token()
        if token != self._cache_token:
            self._cache_token = token
            self._analysis_cache = {}

        id_hint = ("_id", "id", "_key", "key", "uuid", "hash", "code")
        date_hint = ("_date", "date", "_time", "time", "created", "updated", "_at")

        candidates = []
        for col in data.columns:
            name = col.lower()
            if name.endswith(id_hint) or "_id" in name:
                continue
            if any(h in name for h in date_hint):
                continue
            s = data[col]
            nunique = s.nunique()
            null_rate = float(s.isna().mean())

            # Determine likely task type (mirrors ModelSelector._detect_task_type)
            if not pd.api.types.is_numeric_dtype(s.dtype):
                task = "classification"
            else:
                if nunique <= 10:
                    task = "classification"
                elif pd.api.types.is_integer_dtype(s.dtype) and nunique <= 50:
                    mn, mx = (int(s.min()), int(s.max())) if nunique else (0, 0)
                    rng = mx - mn + 1
                    coverage = (nunique / rng) if rng > 0 else 0
                    task = "regression" if coverage > 0.5 else "classification"
                else:
                    task = "regression"

            # Score: very high cardinality + heavy missing penalised.
            score = 10.0
            score -= min(nunique / max(len(s), 1) * 20, 8)
            score -= null_rate * 20
            if task == "classification":
                if 2 <= nunique <= 20:
                    score += 3
                elif nunique > 50:
                    score -= 4
            target_hint = ("salary", "price", "amount", "score", "rating", "value",
                          "income", "cost", "label", "class", "target", "outcome")
            if any(h in name for h in target_hint):
                score += 2

            candidates.append({
                "column": col,
                "task_type": task,
                "cardinality": int(nunique),
                "null_rate": round(null_rate, 4),
                "score": round(score, 3),
                "dtype": str(s.dtype),
            })

        candidates.sort(key=lambda c: (-c["score"], c["column"]))
        self._suggested_targets = candidates[: k if k > 0 else None]
        return candidates[: k if k > 0 else None]

    def find_experiment(
        self, data_source: Optional[str] = None, target_column: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Return the best previous completed experiment matching the given
        data source and/or target, or None."""
        best = None
        for e in self.experiments.list():
            if e.get("status") != "done":
                continue
            if data_source and e.get("data_source") != data_source:
                continue
            if target_column and e.get("target_column") != target_column:
                continue
            if best is None or (e.get("best_cv_score") or 0) >= (best.get("best_cv_score") or 0):
                best = e
        return best

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

    def batch_what_if(
        self,
        feature: str,
        value: Any,
        df: Optional[pd.DataFrame] = None,
        top_rows: int = 20,
    ) -> Dict[str, Any]:
        """Perturb a feature across a whole dataset and summarise the effect.

        Returns a summary (mean delta for regression, % class-changed for
        classification) plus a sample of per-row before/after predictions.
        """
        if self.predictor is None:
            raise RuntimeError("Model not trained. Call train() first.")
        data = df if df is not None else self.current_df
        if data is None:
            raise RuntimeError("No data to test. Pass df or load data first.")
        return self.predictor.batch_what_if(feature, value, data, top_rows=top_rows)

    def repredict_table(
        self,
        table: str,
        limit: Optional[int] = None,
        contamination: float = 0.05,
    ) -> Dict[str, Any]:
        """Re-score every row of a (likely changed) production table with the
        trained model, flagging drift against the training reference and rows
        that look anomalous.

        Returns predictions plus optional drift (PSI) and anomaly summaries.
        """
        if self.predictor is None:
            raise RuntimeError("No trained model to re-score with. Train or load a model first.")
        if not table:
            raise ValueError("table is required.")

        live = self.db.load_table(table, limit=limit)
        predictions = self.predictor.predict(live)

        drift = None
        try:
            feats = (
                self.model_selector.feature_columns
                if self.model_selector is not None else None
            )
            if self.current_df is not None and feats:
                valid = [f for f in feats if f in self.current_df.columns]
                if valid:
                    self.monitor.capture_reference(self.current_df, feature_columns=valid)
            if self.monitor.reference is not None:
                drift = self.monitor.monitor(live)
        except Exception:
            drift = None

        anomaly = None
        try:
            an = self.anomaly.detect(live, contamination=float(contamination))
            anomaly = {
                "n_anomalies": an.get("n_anomalies"),
                "total": an.get("total"),
                "features": an.get("features"),
                "importance": an.get("importance"),
                "top_drivers": an.get("top_drivers"),
            }
        except Exception:
            anomaly = None

        return {
            "predictions": predictions,
            "drift": drift,
            "anomaly": anomaly,
            "source": f"table:{table}",
            "rows": int(len(predictions)),
        }

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