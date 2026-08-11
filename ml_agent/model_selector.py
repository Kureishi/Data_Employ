"""Model selection module for the ML Agent."""
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_squared_error, mean_absolute_error, roc_auc_score,
)
from sklearn.base import BaseEstimator

warnings.filterwarnings("ignore")


class ModelSelector:
    """Automatically selects the best ML model for the given data."""

    # Model candidates per task type
    REGRESSION_MODELS: Dict[str, BaseEstimator] = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=42, max_depth=10),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
        "Extra Trees": ExtraTreesRegressor(random_state=42, n_estimators=100),
        "K-Neighbors": KNeighborsRegressor(n_neighbors=5),
        "SVR": SVR(kernel="rbf"),
    }

    CLASSIFICATION_MODELS: Dict[str, BaseEstimator] = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42, max_depth=10),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "Extra Trees": ExtraTreesClassifier(random_state=42, n_estimators=100),
        "K-Neighbors": KNeighborsClassifier(n_neighbors=5),
        "SVC": SVC(kernel="rbf", probability=True),
    }

    def __init__(
        self,
        target_column: str,
        task_type: Optional[str] = None,
        test_size: float = 0.2,
        cv_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Args:
            target_column: Name of the target/prediction column.
            task_type: "regression", "classification", or None for auto-detection.
            test_size: Fraction of data for test set.
            cv_folds: Number of cross-validation folds.
            random_state: Random seed for reproducibility.
        """
        self.target_column = target_column
        self.task_type = task_type
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.random_state = random_state

        self.feature_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numeric_columns: List[str] = []
        self.label_encoder: Optional[LabelEncoder] = None
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[BaseEstimator] = None
        self.best_score: float = 0.0
        self.model_scores: Dict[str, float] = {}
        self.metrics: Dict[str, Any] = {}
        self.pipeline: Optional[Pipeline] = None
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_importance: Optional[pd.DataFrame] = None

    def _detect_task_type(self, y: pd.Series) -> str:
        """Auto-detect whether the task is regression or classification."""
        if self.task_type:
            return self.task_type

        if not pd.api.types.is_numeric_dtype(y.dtype):
            return "classification"

        # Numeric heuristics
        unique_count = y.nunique()
        if unique_count <= 10:
            return "classification"

        if pd.api.types.is_integer_dtype(y.dtype) and unique_count <= 50:
            # Check if the integer values densely cover their range.
            # Dense coverage suggests a count/continuous variable (regression),
            # sparse coverage suggests categorical labels (classification).
            min_val, max_val = y.min(), y.max()
            value_range = max_val - min_val + 1
            coverage = unique_count / value_range if value_range > 0 else 0
            if coverage > 0.5:
                return "regression"
            return "classification"

        return "regression"

    def _prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target, separating categorical and numeric columns."""
        features = df.drop(columns=[self.target_column])
        target = df[self.target_column]

        # Remove constant columns and ID-like columns
        constants = []
        for col in features.columns:
            if features[col].nunique(dropna=False) <= 1:
                constants.append(col)
            elif not pd.api.types.is_numeric_dtype(features[col].dtype) and features[col].nunique() == len(features[col]):
                constants.append(col)  # high-cardinality object column (likely ID)
            elif pd.api.types.is_numeric_dtype(features[col].dtype):
                # Detect auto-increment primary key columns:
                # must be unique (all distinct) AND sequential starting from 1.
                # Foreign keys (repeated values) are kept as meaningful features.
                non_null = features[col].dropna()
                if len(non_null) == len(features) and non_null.nunique() == len(features):
                    unique_vals = sorted(non_null.unique())
                    is_sequential = (
                        len(unique_vals) > 1
                        and unique_vals[0] == 1
                        and np.all(np.diff(unique_vals) == 1)
                    )
                    if is_sequential and col.lower().endswith(("_id", "id")):
                        constants.append(col)
        features = features.drop(columns=constants)

        self.feature_columns = list(features.columns)

        # Separate column types
        self.numeric_columns = []
        self.categorical_columns = []
        for col in features.columns:
            if not pd.api.types.is_numeric_dtype(features[col].dtype):
                self.categorical_columns.append(col)
            else:
                self.numeric_columns.append(col)

        return features, target

    def _build_preprocessor(self) -> ColumnTransformer:
        """Build preprocessing pipeline for numeric and categorical features."""
        transformers = []

        if self.numeric_columns:
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", num_pipeline, self.numeric_columns))

        if self.categorical_columns:
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat", cat_pipeline, self.categorical_columns))

        return ColumnTransformer(transformers, remainder="drop")

    def _encode_target(self, y: pd.Series) -> Tuple[pd.Series, bool]:
        """Encode categorical target if needed. Returns (encoded_y, was_encoded)."""
        if self.task_type == "classification" and not pd.api.types.is_numeric_dtype(y.dtype):
            self.label_encoder = LabelEncoder()
            return pd.Series(self.label_encoder.fit_transform(y), index=y.index), True
        return y, False

    def _get_models(self) -> Dict[str, BaseEstimator]:
        return (
            self.CLASSIFICATION_MODELS if self.task_type == "classification"
            else self.REGRESSION_MODELS
        )

    def _get_metric(self):
        return "accuracy" if self.task_type == "classification" else "r2"

    def _evaluate_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate all candidate models with cross-validation."""
        scores = {}
        models = self._get_models()
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            if self.task_type == "classification"
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        )

        for name, model in models.items():
            try:
                pipeline = Pipeline([
                    ("preprocessor", self.preprocessor),
                    ("model", model),
                ])
                score = cross_val_score(
                    pipeline, X, y, cv=cv,
                    scoring=self._get_metric(), n_jobs=-1,
                ).mean()
                scores[name] = score
            except Exception:
                continue

        return scores

    def _get_feature_importance(self, model: BaseEstimator, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract feature importance if the model supports it."""
        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                return None

            # Get transformed feature names (numeric first, then categorical,
            # matching the ColumnTransformer transformer order)
            feature_names = []
            if self.numeric_columns:
                feature_names.extend(self.numeric_columns)
            if self.categorical_columns:
                ohe = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
                cat_names = ohe.get_feature_names_out(self.categorical_columns)
                feature_names.extend(cat_names)

            if len(feature_names) != len(importances):
                return None

            imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            return imp_df.sort_values("importance", ascending=False)
        except Exception:
            return None

    def select(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Automatically select the best model for the given data.

        Args:
            df: DataFrame containing features and target column.

        Returns:
            Dictionary with task type, best model, scores, and feature importance.
        """
        X, y = self._prepare_data(df)

        if X.shape[1] == 0:
            raise ValueError("No usable feature columns found after removing constants/IDs.")

        # Detect task type
        self.task_type = self._detect_task_type(y)
        y, was_encoded = self._encode_target(y)

        # Build preprocessor
        self.preprocessor = self._build_preprocessor()

        # Evaluate models
        self.model_scores = self._evaluate_models(X, y)

        if not self.model_scores:
            raise ValueError("No models could be evaluated successfully.")

        # Select best model
        self.best_model_name = max(self.model_scores, key=self.model_scores.get)
        self.best_score = self.model_scores[self.best_model_name]

        # Fit best model on full training split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state,
            stratify=y if self.task_type == "classification" and y.nunique() > 1 else None,
        )
        self._last_X_test = X_test

        best_model = self._get_models()[self.best_model_name]
        self.pipeline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", best_model),
        ])
        self.pipeline.fit(X_train, y_train)

        # Compute test metrics
        y_pred = self.pipeline.predict(X_test)
        self.metrics = self._compute_metrics(y_test, y_pred)

        # Feature importance
        self.feature_importance = self._get_feature_importance(
            self.pipeline.named_steps["model"], X_train
        )

        return self.get_result_summary(was_encoded)

    def _compute_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, Any]:
        """Compute evaluation metrics for the test set."""
        if self.task_type == "classification":
            # Handle binary vs multiclass
            try:
                y_pred_proba = self.pipeline.predict_proba(self._last_X_test)
            except Exception:
                y_pred_proba = None

            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            }
            if y_pred_proba is not None and y_true.nunique() == 2:
                try:
                    metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba[:, 1])
                except Exception:
                    pass
            return metrics
        else:
            return {
                "r2": r2_score(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                "mae": mean_absolute_error(y_true, y_pred),
            }

    def get_result_summary(self, was_encoded: bool = False) -> Dict[str, Any]:
        """Get a summary of the model selection results."""
        return {
            "task_type": self.task_type,
            "target_column": self.target_column,
            "best_model": self.best_model_name,
            "best_cv_score": round(self.best_score, 4),
            "model_scores": {k: round(v, 4) for k, v in sorted(
                self.model_scores.items(), key=lambda x: x[1], reverse=True
            )},
            "test_metrics": self.metrics,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "target_was_encoded": was_encoded,
            "class_mapping": (
                dict(enumerate(self.label_encoder.classes_)) if self.label_encoder else None
            ),
            "feature_importance": self.feature_importance,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the selected model."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet. Call select() first.")
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained yet. Call select() first.")
        if self.task_type != "classification":
            raise ValueError("predict_proba is only available for classification tasks.")
        return self.pipeline.predict_proba(X)

    def save_model(self, path: str) -> None:
        """Save the trained pipeline to disk."""
        import joblib
        if self.preprocessor and self.categorical_columns:
            try:
                ohe = self.preprocessor.named_transformers_["cat"].named_steps["onehot"]
                self._saved_cat_names = ohe.get_feature_names_out(self.categorical_columns)
            except Exception:
                self._saved_cat_names = None
        joblib.dump({
            "pipeline": self.pipeline,
            "task_type": self.task_type,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "best_model_name": self.best_model_name,
            "best_score": self.best_score,
            "metrics": self.metrics,
            "label_encoder": self.label_encoder,
            "feature_importance": (
                self.feature_importance.to_dict(orient="list")
                if self.feature_importance is not None else None
            ),
            "class_mapping": (
                dict(enumerate(self.label_encoder.classes_)) if self.label_encoder else None
            ),
        }, path)

    def load_model(self, path: str) -> None:
        """Load a trained pipeline from disk."""
        import joblib
        data = joblib.load(path)
        self.pipeline = data["pipeline"]
        self.task_type = data["task_type"]
        self.target_column = data["target_column"]
        self.feature_columns = data["feature_columns"]
        self.numeric_columns = data.get("numeric_columns", [])
        self.categorical_columns = data.get("categorical_columns", [])
        self.best_model_name = data["best_model_name"]
        self.best_score = data["best_score"]
        self.metrics = data.get("metrics", {})
        self.label_encoder = data["label_encoder"]

        # Restore feature importance DataFrame
        fi_data = data.get("feature_importance")
        if fi_data:
            self.feature_importance = pd.DataFrame(fi_data)

        # Extract preprocessor from pipeline for feature importance extraction
        try:
            self.preprocessor = self.pipeline.named_steps["preprocessor"]
        except Exception:
            self.preprocessor = None
