"""
Data preprocessing module for the ML Agent.
Provides common data cleaning and transformation operations.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


class DataPreprocessor:
    """Performs common data preprocessing operations on DataFrames."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df
        self.operations_log: List[Dict[str, Any]] = []
        self.fitted_transformers: Dict[str, Any] = {}

    def set_data(self, df: pd.DataFrame) -> None:
        """Set the working DataFrame."""
        self.df = df.copy()

    def get_data(self) -> pd.DataFrame:
        """Get the current DataFrame."""
        if self.df is None:
            raise RuntimeError("No data set. Call set_data() first.")
        return self.df

    def _log(self, operation: str, details: Dict[str, Any]) -> None:
        """Log a preprocessing operation."""
        self.operations_log.append({"operation": operation, **details})

    # ========== Column Operations ==========

    def drop_columns(self, columns: List[str]) -> pd.DataFrame:
        """Drop specified columns."""
        df = self.get_data()
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        result = df.drop(columns=columns)
        self._log("drop_columns", {"columns": columns, "rows": len(result), "cols": len(result.columns)})
        self.df = result
        return result

    def keep_columns(self, columns: List[str]) -> pd.DataFrame:
        """Keep only the specified columns."""
        df = self.get_data()
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        result = df[columns]
        self._log("keep_columns", {"columns": columns, "rows": len(result), "cols": len(result.columns)})
        self.df = result
        return result

    def rename_columns(self, mapping: Dict[str, str]) -> pd.DataFrame:
        """Rename columns using a mapping dict."""
        df = self.get_data()
        result = df.rename(columns=mapping)
        self._log("rename_columns", {"mapping": mapping})
        self.df = result
        return result

    # ========== Missing Value Handling ==========

    def dropna(self, subset: Optional[List[str]] = None, how: str = "any") -> pd.DataFrame:
        """Drop rows with missing values."""
        df = self.get_data()
        before = len(df)
        result = df.dropna(subset=subset, how=how)
        self._log("dropna", {"subset": subset, "how": how, "rows_dropped": before - len(result)})
        self.df = result
        return result

    def fillna(self, value: Any = None, method: str = "mean", columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Fill missing values.

        Args:
            value: Constant value to fill with (if method='constant').
            method: 'mean', 'median', 'mode', 'zero', 'constant', 'ffill', 'bfill'.
            columns: Columns to fill (default: all).
        """
        df = self.get_data()
        result = df.copy()
        cols = columns if columns else list(df.columns)

        for col in cols:
            if col not in result.columns:
                continue
            if method == "mean":
                if pd.api.types.is_numeric_dtype(result[col].dtype):
                    result[col] = result[col].fillna(result[col].mean())
            elif method == "median":
                if pd.api.types.is_numeric_dtype(result[col].dtype):
                    result[col] = result[col].fillna(result[col].median())
            elif method == "mode":
                mode_val = result[col].mode()
                if len(mode_val) > 0:
                    result[col] = result[col].fillna(mode_val[0])
            elif method == "zero":
                result[col] = result[col].fillna(0)
            elif method == "constant":
                result[col] = result[col].fillna(value)
            elif method == "ffill":
                result[col] = result[col].fillna(method="ffill")
            elif method == "bfill":
                result[col] = result[col].fillna(method="bfill")
            else:
                raise ValueError(f"Unknown fill method: {method}")

        self._log("fillna", {"method": method, "columns": cols, "value": value})
        self.df = result
        return result

    # ========== Row Operations ==========

    def drop_duplicates(self, subset: Optional[List[str]] = None) -> pd.DataFrame:
        """Drop duplicate rows."""
        df = self.get_data()
        before = len(df)
        result = df.drop_duplicates(subset=subset)
        self._log("drop_duplicates", {"subset": subset, "rows_dropped": before - len(result)})
        self.df = result
        return result

    def filter_rows(self, condition: str) -> pd.DataFrame:
        """
        Filter rows using a pandas query expression.

        Example: "salary > 50000 and age < 40"
        """
        df = self.get_data()
        before = len(df)
        result = df.query(condition)
        self._log("filter_rows", {"condition": condition, "rows_kept": len(result), "rows_dropped": before - len(result)})
        self.df = result
        return result

    def sample_rows(self, n: Optional[int] = None, frac: Optional[float] = None, random_state: int = 42) -> pd.DataFrame:
        """Sample rows from the dataset."""
        df = self.get_data()
        result = df.sample(n=n, frac=frac, random_state=random_state)
        self._log("sample_rows", {"n": n, "frac": frac, "rows": len(result)})
        self.df = result
        return result

    def head_rows(self, n: int = 100) -> pd.DataFrame:
        """Keep only the first n rows."""
        df = self.get_data()
        result = df.head(n)
        self._log("head_rows", {"n": n, "rows": len(result)})
        self.df = result
        return result

    # ========== Scaling ==========

    def scale_numeric(self, method: str = "standard", columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Scale numeric columns.

        Args:
            method: 'standard' (z-score) or 'minmax'.
            columns: Columns to scale (default: all numeric).
        """
        df = self.get_data()
        result = df.copy()
        cols = columns if columns else [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c].dtype)
        ]
        cols = [c for c in cols if c in df.columns]

        if not cols:
            self._log("scale_numeric", {"method": method, "columns": [], "note": "no numeric columns"})
            return result

        scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        result[cols] = scaler.fit_transform(result[cols])
        self.fitted_transformers[f"scale_{method}"] = (scaler, cols)
        self._log("scale_numeric", {"method": method, "columns": cols})
        self.df = result
        return result

    # ========== Categorical Encoding ==========

    def encode_categorical(self, method: str = "label", columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical columns.

        Args:
            method: 'label' (integer labels) or 'onehot' (dummy variables).
            columns: Columns to encode (default: all categorical).
        """
        df = self.get_data()
        result = df.copy()
        cols = columns if columns else [
            c for c in df.columns
            if not pd.api.types.is_numeric_dtype(df[c].dtype)
        ]
        cols = [c for c in cols if c in df.columns]

        if not cols:
            self._log("encode_categorical", {"method": method, "columns": [], "note": "no categorical columns"})
            return result

        if method == "label":
            for col in cols:
                le = LabelEncoder()
                result[col] = le.fit_transform(result[col].astype(str))
                self.fitted_transformers[f"label_{col}"] = le
        elif method == "onehot":
            result = pd.get_dummies(result, columns=cols, prefix=cols, drop_first=False)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

        self._log("encode_categorical", {"method": method, "columns": cols})
        self.df = result
        return result

    # ========== Feature Engineering ==========

    def create_ratio(self, numerator: str, denominator: str, new_column: str) -> pd.DataFrame:
        """Create a ratio feature from two columns."""
        df = self.get_data()
        if numerator not in df.columns or denominator not in df.columns:
            raise ValueError(f"Columns not found: {numerator}, {denominator}")
        result = df.copy()
        result[new_column] = result[numerator] / result[denominator].replace(0, np.nan)
        self._log("create_ratio", {"numerator": numerator, "denominator": denominator, "new_column": new_column})
        self.df = result
        return result

    def create_product(self, col1: str, col2: str, new_column: str) -> pd.DataFrame:
        """Create a product feature from two columns."""
        df = self.get_data()
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Columns not found: {col1}, {col2}")
        result = df.copy()
        result[new_column] = result[col1] * result[col2]
        self._log("create_product", {"col1": col1, "col2": col2, "new_column": new_column})
        self.df = result
        return result

    def create_difference(self, col1: str, col2: str, new_column: str) -> pd.DataFrame:
        """Create a difference feature from two columns."""
        df = self.get_data()
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Columns not found: {col1}, {col2}")
        result = df.copy()
        result[new_column] = result[col1] - result[col2]
        self._log("create_difference", {"col1": col1, "col2": col2, "new_column": new_column})
        self.df = result
        return result

    def create_bins(self, column: str, bins: int, new_column: Optional[str] = None) -> pd.DataFrame:
        """Bin a numeric column into categorical bins."""
        df = self.get_data()
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")
        result = df.copy()
        out_col = new_column or f"{column}_binned"
        result[out_col] = pd.cut(result[column], bins=bins)
        self._log("create_bins", {"column": column, "bins": bins, "new_column": out_col})
        self.df = result
        return result

    # ========== Type Conversion ==========

    def convert_dtype(self, column: str, dtype: str) -> pd.DataFrame:
        """Convert a column to a different data type."""
        df = self.get_data()
        if column not in df.columns:
            raise ValueError(f"Column not found: {column}")
        result = df.copy()
        result[column] = result[column].astype(dtype)
        self._log("convert_dtype", {"column": column, "dtype": dtype})
        self.df = result
        return result

    # ========== Transform (apply fitted transformers to new data) ==========

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted transformers (scalers, label encoders) to new data.
        Used for preprocessing prediction data consistently with training data.
        """
        result = df.copy()

        # Apply label encoders
        for key, le in self.fitted_transformers.items():
            if key.startswith("label_"):
                col = key[6:]
                if col in result.columns:
                    known = set(le.classes_)
                    result[col] = result[col].astype(str).map(
                        lambda v: le.transform([v])[0] if v in known else -1
                    )

        # Apply scalers
        for key, value in self.fitted_transformers.items():
            if key.startswith("scale_"):
                scaler, cols = value
                existing_cols = [c for c in cols if c in result.columns]
                if existing_cols:
                    result[existing_cols] = scaler.transform(result[existing_cols])

        return result

    # ========== Utility ==========

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of preprocessing operations performed."""
        return {
            "operations": self.operations_log,
            "total_operations": len(self.operations_log),
            "current_shape": (
                {"rows": len(self.df), "columns": len(self.df.columns)}
                if self.df is not None else None
            ),
        }

    def reset(self) -> None:
        """Reset the preprocessor."""
        self.df = None
        self.operations_log = []
        self.fitted_transformers = {}