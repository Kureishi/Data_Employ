"""
LLM advisor module for the ML Agent.
Provides natural-language intelligence using a local LLM served by LM Studio.

LM Studio exposes an OpenAI-compatible API at http://localhost:1234/v1,
so all requests here use the /chat/completions endpoint without shipping
the full openai SDK — only `requests` is needed.

Every method degrades gracefully: if the LLM server is unreachable or
times out, a heuristic fallback result is returned instead of crashing.
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# Default LM Studio local server
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TIMEOUT = 60


class LLMAdvisorError(Exception):
    """Raised when the LM Studio server cannot fulfill a request."""


class LLMAdvisor:
    """
    An advisor that uses a local LLM (LM Studio) for:
      1. Natural language -> SQL generation
      2. Target column recommendation
      3. Preprocessing / feature-engineering suggestions
      4. Plain-language interpretation of model results
      5. Prediction narratives
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ):
        """
        Args:
            base_url: LM Studio server URL (e.g. http://localhost:1234/v1).
            model: Optional model name to ping. If None, LM Studio's loaded model is used.
            timeout: Request timeout in seconds.
            temperature: LLM sampling temperature (lower = more deterministic).
            max_tokens: Maximum tokens for the response.
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._available: Optional[bool] = None

    # ========== Connection Management ==========

    def check_connection(self) -> Tuple[bool, str]:
        """Check if the LM Studio server is reachable. Returns (available, detail)."""
        try:
            resp = requests.get(f"{self.base_url}/models", timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                self._available = True
                if self.model:
                    models = [m for m in models if self.model.lower() in m.lower()]
                detail = models[0] if models else "Server reachable"
                return True, detail
            return False, f"HTTP {resp.status_code}"
        except Exception as e:
            self._available = False
            return False, str(e)

    def is_available(self) -> bool:
        """Return whether the LLM server responded during the last check."""
        return bool(self._available)

    # --------------------------------------------------------------------------
    # Core /chat/completions request
    # --------------------------------------------------------------------------

    def _chat(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Send a chat completion request to LM Studio.

        Args:
            messages: OpenAI-style message list: [{"role": "...", "content": "..."}]
            response_format: Optional {"type": "json_object"} for structured output.

        Returns:
            The assistant text response.

        Raises:
            LLMAdvisorError: If the server cannot be reached or returns an error.
        """
        payload: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.model:
            payload["model"] = self.model
        if response_format:
            payload["response_format"] = response_format

        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
        except requests.exceptions.ConnectionError:
            self._available = False
            raise LLMAdvisorError(
                "Cannot reach LM Studio. Start LM Studio and load a model, "
                f"or verify the server at {self.base_url}"
            )
        except requests.exceptions.Timeout:
            self._available = False
            raise LLMAdvisorError(
                f"LM Studio request timed out after {self.timeout}s. "
                "The model may still be loading or the prompt is too large."
            )

        if resp.status_code != 200:
            raise LLMAdvisorError(
                f"LM Studio returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        data = resp.json()
        self._available = True
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as e:
            raise LLMAdvisorError(f"Unexpected LM Studio response format: {e}")

    @staticmethod
    def _parse_json(text: str) -> Optional[Dict[str, Any]]:
        """Best-effort JSON extraction from LLM output."""
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find the first JSON object/array in the text
            for start, end in [("\{", "\}"), ("\[", "\]")]:
                match = re.search(f"{start}.*{end}", text, re.DOTALL)
                if match:
                    try:
                        return json.loads(match.group(0))
                    except json.JSONDecodeError:
                        continue
        return None

    def _summarize_dataframe(self, df: pd.DataFrame, sample_rows: int = 5) -> str:
        """Create a compact textual description of a DataFrame for LLM context."""
        lines = [
            f"Dataset: {len(df)} rows x {len(df.columns)} columns",
            "Columns:",
        ]
        for col in df.columns:
            dtype = str(df[col].dtype)
            nunique = df[col].nunique(dropna=True)
            nulls = int(df[col].isna().sum())
            stats = f"dtype={dtype}, unique={nunique}, nulls={nulls}"
            if pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].notna().any():
                stats += f", min={float(df[col].min()):.2f}, max={float(df[col].max()):.2f}, mean={float(df[col].mean()):.2f}"
            else:
                top = df[col].value_counts().head(5)
                vals = ", ".join(f"{k}({v})" for k, v in top.items())
                stats += f", top={vals}"
            lines.append(f"  - {col}: {stats}")
        lines.append(f"Sample rows (first {sample_rows}):")
        lines.append(df.head(sample_rows).to_string(max_colwidth=40))
        return "\n".join(lines)

    # --------------------------------------------------------------------------
    # 1. Natural Language -> SQL
    # --------------------------------------------------------------------------

    def generate_sql(
        self,
        question: str,
        tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Convert a natural-language question into SQL given the database schema.

        Args:
            question: The natural-language question.
            tables: List of table dicts from DatabaseProcessor.get_database_overview()["tables"]

        Returns:
            {"sql": ..., "explanation": ...} or {"error": ...} if LLM unavailable.
        """
        # Build schema description
        schema_desc = []
        for t in tables:
            cols = ", ".join(
                f"{c['name']} {c['type']}{' [PK]' if c.get('primary_key') else ''}"
                for c in t.get("columns", [])
            )
            schema_desc.append(f"Table {t['name']} ({t.get('row_count', '?')} rows): {cols}")
            if t.get("foreign_keys"):
                for fk in t["foreign_keys"]:
                    schema_desc.append(
                        f"  FK: {fk.get('constrained_columns')} -> "
                        f"{fk.get('referred_table')}.{fk.get('referred_columns')}"
                    )

        schema_text = "\n".join(schema_desc)

        system_prompt = (
            "You are a SQL expert. Given a database schema and a user question, "
            "write an SQL query that answers the question. "
            'Respond with ONLY valid JSON in this exact format: '
            '{"sql": "<the SQL query>", "explanation": "<short explanation>"}. '
            "Do not include markdown fences. Use standard SQL compatible with SQLite."
        )
        user_prompt = (
            f"Database schema:\n{schema_text}\n\n"
            f"User question: {question}\n\n"
            "Return the JSON now."
        )

        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json(raw)
            if parsed and parsed.get("sql"):
                return {
                    "sql": parsed["sql"],
                    "explanation": parsed.get("explanation", ""),
                }
            return {"error": f"Could not parse LLM SQL response: {raw[:200]}"}
        except LLMAdvisorError as e:
            return {"error": str(e)}

    # --------------------------------------------------------------------------
    # 2. Target Column Recommendation
    # --------------------------------------------------------------------------

    def suggest_target(
        self,
        df: pd.DataFrame,
        preferred: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recommend which column to use as the prediction target.

        Args:
            df: The training DataFrame.
            preferred: Optional column the user wants to predict (skips LLM).

        Returns:
            dict with "target_column", "task_type", "reasoning".
        """
        if preferred and preferred in df.columns:
            return {
                "target_column": preferred,
                "task_type": self._heuristic_task_type(df[preferred]),
                "reasoning": f"User-specified target column '{preferred}'.",
                "source": "user",
            }

        summary = self._summarize_dataframe(df, sample_rows=3)

        system_prompt = (
            "You are an ML data-scientist advisor. Given a dataset summary, "
            "recommend which single column should be the prediction target. "
            "Prefer columns that are: not auto-increment IDs, not free-text, "
            "not constants, and have meaningful variance. "
            'Respond with ONLY JSON: {"target_column": "...", '
            '"task_type": "regression" or "classification", '
            '"reasoning": "<brief explanation>"}. '
            "If no column makes a good target, set target_column to the "
            "column with the most signal. Do not use markdown fences."
        )

        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Dataset:\n{summary}"},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json(raw)
            if parsed and parsed.get("target_column") in df.columns:
                task = parsed.get("task_type", "auto")
                if task not in ("regression", "classification"):
                    task = "auto"
                return {
                    "target_column": parsed["target_column"],
                    "task_type": task,
                    "reasoning": parsed.get("reasoning", ""),
                    "source": "llm",
                }
            return {
                "error": f"LLM recommended column not in dataset: {raw[:200]}",
                "source": "llm",
            }
        except LLMAdvisorError as e:
            # Heuristic fallback
            return self._heuristic_suggest_target(df, error=str(e))

    def _heuristic_suggest_target(self, df: pd.DataFrame, error: str = "") -> Dict[str, Any]:
        """Heuristic fallback: pick the numeric column with highest variance."""
        best_col = None
        best_score = -1.0
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col].dtype):
                continue
            non_null = df[col].dropna()
            if len(non_null) < 2:
                continue
            if non_null.nunique() == 1:
                continue
            # Skip auto-increment PK-like columns
            unique_vals = sorted(non_null.unique())
            if (
                len(non_null) == len(df)
                and non_null.nunique() == len(df)
                and len(unique_vals) > 1
                and unique_vals[0] == 1
            ):
                continue
            score = float(non_null.std())
            if score > best_score:
                best_score = score
                best_col = col

        if best_col is None:
            for col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col].dtype) and df[col].nunique() <= 10:
                    best_col = col
                    break

        if best_col is None:
            return {"error": "No suitable target column found heuristically."}

        return {
            "target_column": best_col,
            "task_type": self._heuristic_task_type(df[best_col]),
            "reasoning": (
                f"LLM unavailable ({error}) - heuristic fallback: chosen column "
                f"'{best_col}' with the highest variance as the strongest signal."
            ),
            "source": "heuristic",
        }

    @staticmethod
    def _heuristic_task_type(s: pd.Series) -> str:
        """Match ModelSelector's auto-detection heuristic."""
        if not pd.api.types.is_numeric_dtype(s.dtype):
            return "classification"
        unique_count = s.nunique()
        if unique_count <= 10:
            return "classification"
        if pd.api.types.is_integer_dtype(s.dtype) and unique_count <= 50:
            min_val, max_val = s.min(), s.max()
            value_range = max_val - min_val + 1
            coverage = unique_count / value_range if value_range > 0 else 0
            return "regression" if coverage > 0.5 else "classification"
        return "regression"

    # --------------------------------------------------------------------------
    # 3. Preprocessing / Feature Engineering Recommendations
    # --------------------------------------------------------------------------

    def suggest_preprocessing(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Recommend a chain of preprocessing operations based on the data profile.

        Available operations (match DataPreprocessor):
        dropna, fillna(method), drop_columns(cols), keep_columns(cols),
        drop_duplicates, filter_rows(condition), scale_numeric(standard|minmax),
        encode_categorical(label|onehot), sample(n), head(n),
        create_ratio(num, den, new), create_product(c1, c2, new),
        create_difference(c1, c2, new), create_bins(col, bins)

        Returns:
            dict: {"operations": [...], "rationale": "..."}
        """
        summary = self._summarize_dataframe(df, sample_rows=3)

        system_prompt = (
            "You are a data-science feature-engineering expert. Given a dataset "
            "summary and optional target column, recommend a chain of up to 5 "
            "preprocessing operations from this exact list of op dicts:"
            "\n"
            '  {"op": "dropna"}\n'
            '  {"op": "fillna", "method": "mean|median|mode|zero"}\n'
            '  {"op": "drop_columns", "columns": ["col"]}\n'
            '  {"op": "keep_columns", "columns": ["col1", "col2"]}\n'
            '  {"op": "drop_duplicates"}\n'
            '  {"op": "filter_rows", "condition": "col > 0"}\n'
            '  {"op": "scale_numeric", "method": "standard|minmax", '
            '"exclude": ["<target>"]}\n'
            '  {"op": "encode_categorical", "method": "label|onehot"}\n'
            '  {"op": "create_ratio", "numerator": "a", "denominator": "b", '
            '"new_column": "ratio_ab"}\n'
            '  {"op": "create_product", "col1": "a", "col2": "b", '
            '"new_column": "prod_ab"}\n'
            '  {"op": "create_difference", "col1": "a", "col2": "b", '
            '"new_column": "diff_ab"}\n'
            '  {"op": "create_bins", "column": "a", "bins": 5}\n'
            "\n"
            "Rules: never scale or drop the target column; only reference column "
            "names that actually exist; return an empty operations list if the "
            "data is already clean. Return ONLY JSON: "
            '{"operations": [...], "preferred": "<why these ops>"}. '
            "No markdown fences."
        )

        user_prompt = (
            f"Target column: {target_column or 'NONE - pick the most useful target'}\n\n"
            f"Dataset:\n{summary}\n"
        )

        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json(raw)
            if parsed is None:
                return {"error": f"Could not parse LLM response: {raw[:200]}"}
            ops = parsed.get("operations", [])
            # Sanitize: reject ops referencing the target
            if target_column:
                ops = [
                    op for op in ops
                    if all(
                        str(v) != target_column
                        for k, v in op.items()
                        if k in ("columns", "exclude") or k in ("numerator", "denominator", "col1", "col2", "column")
                    )
                ]
            return {
                "operations": ops,
                "preferred": parsed.get("preferred", ""),
                "source": "llm",
            }
        except LLMAdvisorError as e:
            return self._heuristic_suggest_preprocessing(df, target_column, error=str(e))

    def _heuristic_suggest_preprocessing(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        """Simple fallback: drop constants/IDs/free-text, fillna, scale numeric (excluding target)."""
        ops: List[Dict[str, Any]] = []
        for col in df.columns:
            if col == target_column:
                continue
            col_data = df[col]
            non_null = col_data.dropna()
            try:
                unique_vals = sorted(non_null.unique())
            except TypeError:
                unique_vals = []

            is_auto_pk = (
                len(non_null) == len(df)
                and non_null.nunique() == len(df)
                and len(unique_vals) > 1
                and unique_vals[0] == 1
                and pd.api.types.is_numeric_dtype(col_data.dtype)
                and col.lower().endswith(("_id", "id"))
            )

            if col_data.nunique(dropna=False) <= 1:
                ops.append({"op": "drop_columns", "columns": [col]})
            elif is_auto_pk:
                ops.append({"op": "drop_columns", "columns": [col]})
            elif not pd.api.types.is_numeric_dtype(col_data.dtype) and non_null.nunique() == len(df):
                ops.append({"op": "drop_columns", "columns": [col]})

        null_count = int(df.isna().sum().sum())
        if null_count > 0:
            ops.append({"op": "fillna", "method": "median" if target_column else "mean"})

        numeric_count = sum(
            1 for c in df.columns
            if c != target_column and pd.api.types.is_numeric_dtype(df[c].dtype)
        )
        if numeric_count > 1:
            scale_op: Dict[str, Any] = {"op": "scale_numeric", "method": "standard"}
            if target_column:
                scale_op["exclude"] = [target_column]
            ops.append(scale_op)

        return {
            "operations": ops,
            "preferred": (
                f"LLM unavailable ({error}). Heuristic fallback applied "
                f"{len(ops)} standard cleaning operations."
            ),
            "source": "heuristic",
        }

    # --------------------------------------------------------------------------
    # 4. Model Result Interpretation
    # --------------------------------------------------------------------------

    def explain_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce a plain-language interpretation of model training results.

        Args:
            results: The dict returned by ModelSelector.select() /
                     MLAgent.train().

        Returns:
            {"explanation": "...", "highlights": [...]} or {"error": ...}
        """
        # Build a compact, serializable summary of the results
        summary_lines = [
            f"Task type: {results.get('task_type')}",
            f"Target column: {results.get('target_column')}",
            f"Best model: {results.get('best_model')}",
            f"Best cross-validation score: {results.get('best_cv_score')}",
            "Test metrics:",
        ]
        for k, v in (results.get("test_metrics") or {}).items():
            if isinstance(v, (int, float)):
                summary_lines.append(f"  {k} = {float(v):.4f}")
        summary_lines.append(f"Feature columns: {results.get('feature_columns')}")
        imp = results.get("feature_importance")
        if imp is not None and len(imp) > 0:
            imp_head = imp.head(10)
            summary_lines.append("Top feature importances:")
            for _, row in imp_head.iterrows():
                summary_lines.append(
                    f"  {row['feature']}: {row['importance']:.4f}"
                )

        system_prompt = (
            "You are a machine learning results interpreter. Given the training "
            "summary, write a clear plain-language interpretation for a business "
            "user. Explain what the metrics mean, whether the model is performing "
            "well, which features matter most, and any caveats. "
            'Return ONLY JSON: {"explanation": "<paragraph>", '
            '"highlights": ["<point1>", "<point2>"]}. No markdown fences.'
        )

        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "Training results:\n" + "\n".join(summary_lines)},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json(raw) or {"explanation": raw, "highlights": []}
            return {
                "explanation": parsed.get("explanation", ""),
                "highlights": parsed.get("highlights", []),
            }
        except LLMAdvisorError as e:
            return {"error": str(e)}

    # --------------------------------------------------------------------------
    # 5. Prediction Narratives
    # --------------------------------------------------------------------------

    def explain_predictions(
        self,
        predictions: pd.DataFrame,
        model_info: Optional[Dict[str, Any]] = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """
        Explain predictions in plain language.

        Args:
            predictions: DataFrame with prediction columns.
            model_info: Optional model info dict from MLAgent.get_model_info().
            top_n: Only explain the first n rows to keep the prompt small.

        Returns:
            {"narratives": ["...", ...]} or {"error": ...}
        """
        n = min(top_n, len(predictions))
        sub = predictions.head(n)

        info_lines = []
        if model_info:
            info_lines.append(f"Model: {model_info.get('best_model')}")
            info_lines.append(f"Task: {model_info.get('task_type')}")
            info_lines.append(f"Target: {model_info.get('target_column')}")
            tm = model_info.get("test_metrics", {})
            if tm:
                info_lines.append(
                    "Test metrics: " + ", ".join(
                        f"{k}={v:.4f}" for k, v in tm.items() if isinstance(v, (int, float))
                    )
                )

        rows_text = sub.to_string(max_colwidth=60)

        system_prompt = (
            "You are a business analytics assistant. Given sample prediction rows "
            "(each may include the input features and the predicted value), "
            "explain each row in 1-2 sentences of plain language: why the predicted "
            "value is high/low/high-confidence, and which input feature drove it. "
            'Return ONLY JSON: {"narratives": ["row1...", "row2...", ...]} matching '
            f"the number of rows (exactly {n}). No markdown fences."
        )
        user_prompt = "\n".join(info_lines + ["", "Prediction rows:", rows_text])

        try:
            raw = self._chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            parsed = self._parse_json(raw)
            if parsed and parsed.get("predictions"):
                return {"narratives": parsed["predictions"]}
            return {"error": f"Could not parse LLM response: {raw[:200]}"}
        except LLMAdvisorError as e:
            return {"error": str(e)}


# ------------------------------------------------------------------------------
# Convenience factory
# ------------------------------------------------------------------------------

def create_llm_advisor(**kwargs) -> LLMAdvisor:
    """Create an LLMAdvisor. Included for discoverability in the package."""
    return LLMAdvisor(**kwargs)