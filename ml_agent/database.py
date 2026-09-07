"""Database processing module for the ML Agent."""
import json
import os
import re
import time
from typing import Any, Dict, List, Optional
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine


class DatabaseProcessor:
    """Processes SQL databases to extract schema and load data."""

    def __init__(self, connection_string: Optional[str] = None):
        self.connection_string = self._normalize(connection_string) if connection_string else None
        self.engine: Optional[Engine] = None
        if self.connection_string:
            self._connect()
        self._profiles: Dict[str, Any] = {}
        self._profiles_path: Optional[str] = None
        self.read_only = True
        self.query_timeout: Optional[float] = None
        self._queries: Dict[str, Any] = {}
        self._queries_path: Optional[str] = None

    def _normalize(self, cs: str) -> str:
        if "://" in cs:
            return cs
        if cs.endswith((".db", ".sqlite", ".sqlite3")):
            return f"sqlite:///{cs}"
        return cs

    def _connect(self) -> None:
        from . import config as cfg  # local import: avoid import cost at module load

        try:
            url = self.connection_string
            backend = (url.split("://", 1)[0] or "sqlite").lower()
            kwargs: Dict[str, Any] = {}

            if backend == "sqlite":
                # SQLite needs a connect_args timeout so concurrent writers/readers
                # wait on the in-process busy lock instead of failing immediately,
                # and allow the pooled connection to be used across threads when
                # the server runs threaded (waitress/gunicorn threads).
                kwargs = {
                    "connect_args": {
                        "timeout": cfg.SQLITE_BUSY_TIMEOUT / 1000.0,
                        "check_same_thread": cfg.SQLITE_CHECK_SAME_THREAD,
                    }
                }
            else:
                # Network databases: build a bounded pool and pre-ping idle
                # connections so a dropped MySQL/Postgres connection is detected
                # and replaced instead of raising mid-request.
                kwargs = {
                    "pool_size": cfg.DB_POOL_SIZE,
                    "max_overflow": cfg.DB_MAX_OVERFLOW,
                    "pool_recycle": cfg.DB_POOL_RECYCLE,
                    "pool_timeout": cfg.DB_POOL_TIMEOUT,
                    "pool_pre_ping": cfg.DB_POOL_PRE_PING,
                }
                if cfg.DB_CONNECT_TIMEOUT is not None:
                    kwargs.setdefault(
                        "connect_args",
                        {"connect_timeout": cfg.DB_CONNECT_TIMEOUT},
                    )

            self.engine = create_engine(url, **kwargs)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
            self.engine = None
            raise ConnectionError(f"Failed to connect to database: {e}")

    def get_tables(self) -> List[str]:
        return inspect(self.engine).get_table_names()

    def get_schema(self, table: str) -> List[Dict[str, Any]]:
        cols = inspect(self.engine).get_columns(table)
        return [{
            "name": c["name"], "type": str(c["type"]),
            "nullable": c.get("nullable", True),
            "default": c.get("default"),
            "primary_key": c.get("primary_key", False),
        } for c in cols]

    def get_primary_keys(self, table: str) -> List[str]:
        return inspect(self.engine).get_pk_constraint(table).get("constrained_columns", [])

    def get_foreign_keys(self, table: str) -> List[Dict[str, Any]]:
        return inspect(self.engine).get_foreign_keys(table)

    # ========== Column-type detection & auto-typing ==========

    _ID_PATTERNS = (
        re.compile(r"(^|_)(id|ids|key|keys|uuid|guid|sk|hk|pk)(_|$)"),
        re.compile(r"^(id|uuid|gtin|ean|empno|deptno|sku)$", re.I),
        re.compile(r"(^|_)(row_?num|serial|seq|id)(_|$)", re.I),
    )
    _FK_PATTERNS = (
        re.compile(r"(.*)_(id|ids)$", re.I),
        re.compile(r"(.*)(Id|ID)$"),
    )
    _DATE_PATTERNS = (
        re.compile(r"(^|_)(date|dt)(_|$)", re.I),
        re.compile(r"(^|_)(.*_at|created|updated|timestamp|time)$", re.I),
        re.compile(r"(^|_)date(_|$)", re.I),
    )
    _BOOL_WORDS = {"is_", "has_", "flag", "enabled", "active", "valid", "deleted", "bool"}

    @staticmethod
    def _looks_like_bool_name(name: str) -> bool:
        low = name.lower()
        return any(low.startswith(w) or low.endswith(("_flag", "_enabled", "_active")) for w in ("is_", "has_"))

    def detect_column_type(
        self,
        name: str,
        sql_type: str = "",
        pandas_dtype: Optional[str] = None,
        unique_count: Optional[int] = None,
        sample_size: Optional[int] = None,
    ) -> str:
        """Heuristically classify a column into a semantic type.

        Returns one of: "primary_key", "foreign_key", "id", "date",
        "categorical", "boolean", "numeric_ratio", "numeric", "text".
        """
        low = name.lower()
        dtype_str = (pandas_dtype or sql_type or "").lower()

        # Date
        if "datetime" in dtype_str or "timestamp" in dtype_str or "date" in dtype_str:
            return "date"
        for pat in self._DATE_PATTERNS:
            if pat.search(name):
                return "date"

        # Boolean names
        if self._looks_like_bool_name(name) or "bool" in dtype_str:
            return "boolean"

        # Primary / foreign keys by name
        if low == "id" or low.endswith((".id")):
            return "primary_key"
        if low.endswith("_id") or re.search(r"_id$", low):
            # FK if it references something (has a base name before _id)
            base = low[:-3]
            if base:
                return "foreign_key"
            return "primary_key"

        # Numeric ratio-typed columns (0..1 scales, percentages)
        if any(tok in low for tok in ("_ratio", "_rate", "_pct", "_prop", "_score_")):
            return "numeric_ratio"

        # Numeric
        if pandas_dtype:
            if pandas_dtype.startswith(("int", "float", "uint", "int64", "float64")):
                return "numeric"
        if any(tok in dtype_str for tok in ("int", "float", "double", "decimal", "numeric")):
            return "numeric"

        # High-cardinality text
        if (pandas_dtype and pandas_dtype.startswith("object")) or "varchar" in dtype_str or "text" in dtype_str:
            if unique_count is not None and sample_size is not None:
                if unique_count > max(50, sample_size * 0.8):
                    return "id"
            return "text"

        return "categorical"

    @staticmethod
    def _is_pk_name(name: str) -> bool:
        low = name.lower()
        return low == "id" or low.endswith(".id") or low == "pk" or low.endswith("_pk")

    def get_column_types(self, table: str, sample_limit: int = 1000) -> List[Dict[str, Any]]:
        """Return per-column semantic types for a table, using a sample."""
        schema = self.get_schema(table)
        sample = self.load_table(table, limit=sample_limit)
        pk_cols = set(self.get_primary_keys(table))
        fk_info = self.get_foreign_keys(table)
        fk_cols = set()
        for fk in fk_info:
            fk_cols.update(fk.get("constrained_columns") or [])

        results = []
        for col in schema:
            name = col["name"]
            sql_type = col["type"]
            pd_dtype = str(sample[name].dtype) if name in sample.columns else ""
            nunique = int(sample[name].nunique()) if name in sample.columns else 0
            sem = "primary_key" if name in pk_cols else ("foreign_key" if name in fk_cols else None)
            if sem is None:
                sem = self.detect_column_type(
                    name, sql_type=sql_type, pandas_dtype=pd_dtype,
                    unique_count=nunique, sample_size=len(sample),
                )
            results.append({
                "name": name,
                "type": sql_type,
                "dtype": pd_dtype,
                "semantic_type": sem,
                "nullable": col.get("nullable", True),
                "primary_key": name in pk_cols,
                "foreign_key": name in fk_cols,
            })
        return results

    def get_row_count(self, table: str) -> int:
        with self.engine.connect() as conn:
            return conn.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar()

    def get_table_summary(self) -> List[Dict[str, Any]]:
        summary = []
        for t in self.get_tables():
            try:
                summary.append({
                    "table": t,
                    "row_count": self.get_row_count(t),
                    "column_count": len(self.get_schema(t)),
                    "columns": [c["name"] for c in self.get_schema(t)],
                })
            except Exception:
                continue
        return summary

    def load_table(self, table: str, limit: Optional[int] = None) -> pd.DataFrame:
        q = f'SELECT * FROM "{table}"'
        if limit:
            q += f" LIMIT {limit}"
        return pd.read_sql_query(q, self.engine)

    # ========== Read-only safety + query timeout (Feature 6) ==========

    _WRITE_KEYWORDS = ("insert", "update", "delete", "drop", "alter", "create",
                       "truncate", "grant", "revoke", "replace", "vacuum", "merge")

    def _is_write_query(self, query: str) -> bool:
        first = query.lstrip().lstrip("(").strip().lower()
        head = re.split(r"\s+", first, maxsplit=1)[0]
        return head in self._WRITE_KEYWORDS

    def execute_query(
        self,
        query: str,
        read_only: Optional[bool] = None,
        timeout: Optional[float] = None,
    ) -> pd.DataFrame:
        """Execute a SQL query with read-only enforcement and optional timeout.

        If read_only is None, uses this processor's `read_only` attribute
        (default True). If timeout is None, uses `self.query_timeout`.
        """
        ro = self.read_only if read_only is None else read_only
        to = self.query_timeout if timeout is None else timeout
        if ro and self._is_write_query(query):
            raise PermissionError(
                "Read-only mode is enabled. Write statements (INSERT/UPDATE/DELETE/DROP...) are blocked."
            )

        holder: Dict[str, Any] = {}
        def _run():
            try:
                holder["df"] = pd.read_sql_query(query, self.engine)
            except Exception as e:  # noqa
                holder["error"] = e

        import threading as _th
        t = _th.Thread(target=_run, daemon=True)
        t.start()
        if to:
            t.join(timeout=to)
            if t.is_alive():
                raise TimeoutError(f"Query exceeded timeout of {to} seconds.")
        else:
            t.join()
        if "error" in holder:
            raise holder["error"]
        return holder["df"]

    def get_database_overview(self) -> Dict[str, Any]:
        overview = {
            "connection_string": self.connection_string,
            "table_count": len(self.get_tables()),
            "tables": [],
        }
        for t in self.get_tables():
            overview["tables"].append({
                "name": t,
                "row_count": self.get_row_count(t),
                "columns": self.get_schema(t),
                "primary_keys": self.get_primary_keys(t),
                "foreign_keys": self.get_foreign_keys(t),
            })
        return overview

    def close(self) -> None:
        if self.engine:
            self.engine.dispose()
            self.engine = None

    # ========== DB-native sampling / chunking (Feature 3) ==========

    def estimate_row_count(self, table: str) -> int:
        """Get a fast row-count estimate where the dialect supports it."""
        try:
            dialect = str(self.engine.url.get_backend_name()).lower()
            if dialect == "postgresql":
                with self.engine.connect() as conn:
                    return int(conn.execute(text(f"SELECT reltuples::bigint FROM pg_class WHERE relname='{table}'")).scalar() or 0)
            if dialect == "sqlite":
                return self.get_row_count(table)
        except Exception:
            pass
        return self.get_row_count(table)

    def load_table_sample(
        self,
        table: str,
        fraction: float = 0.1,
        columns: Optional[List[str]] = None,
        limit: Optional[int] = None,
        method: str = "auto",
        seed: int = 42,
    ) -> pd.DataFrame:
        """Load a random sample of rows from a table, doing sampling in the DB.

        Args:
            table: Table name.
            fraction: Fraction of rows to sample (0 < fraction <= 1).
            columns: Optional subset of columns (pushdown).
            limit: Optional hard row cap.
            method: "auto" | "tablesample" | "random".
            seed: Seed for reproducible random ordering.
        """
        cols = "*" if not columns else ", ".join(f'"{c}"' for c in columns)
        dialect = str(self.engine.url.get_backend_name()).lower()

        if method == "tablesample" or (method == "auto" and dialect in ("postgresql", "sqlite")):
            pct = max(1e-6, min(fraction * 100.0, 100.0))
            try:
                if dialect == "postgresql":
                    sql = f'SELECT {cols} FROM "{table}" TABLESAMPLE BERNOULLI ({pct})'
                else:  # sqlite: order by random() + limit
                    sql = f'SELECT {cols} FROM "{table}" ORDER BY random() LIMIT {limit or 1000}'
                if limit and dialect == "postgresql":
                    sql += f" LIMIT {limit}"
                return pd.read_sql_query(sql, self.engine)
            except Exception:
                pass

        n = limit if limit else max(1, int(self.get_row_count(table) * fraction))
        sql = f'SELECT {cols} FROM "{table}" ORDER BY random() LIMIT {n}'
        return pd.read_sql_query(sql, self.engine)

    def load_table_columns(
        self, table: str, columns: List[str], limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Load only specific columns from a table (column pushdown)."""
        cols = ", ".join(f'"{c}"' for c in columns)
        sql = f'SELECT {cols} FROM "{table}"'
        if limit:
            sql += f" LIMIT {limit}"
        return pd.read_sql_query(sql, self.engine)

    # ========== Multi-table auto-join (Feature 5) ==========

    def _collect_relationships(self, tables: List[str]) -> List[Dict[str, Any]]:
        """Gather FK relationships restricted to the given set of tables."""
        rels = []
        tset = set(tables)
        for t in tables:
            try:
                for fk in self.get_foreign_keys(t):
                    ref = fk.get("referred_table")
                    if ref in tset:
                        rels.append({
                            "table": t,
                            "ref_table": ref,
                            "col": fk["constrained_columns"][0] if fk.get("constrained_columns") else "",
                            "referred_columns": list(fk.get("referred_columns") or []),
                        })
            except Exception:
                pass
        return rels

    def build_join_query(
        self,
        tables: List[str],
        join_type: str = "inner",
    ) -> Dict[str, Any]:
        """Return an SQL query that joins the given tables via FK metadata.

        Each FK edge is between an *owner* table (has the FK column) and a
        *referenced* table. The query aliases every table and selects
        `TableAlias__column` to avoid column-name collisions. Returns
        {"query", "joins"}; joins is a list of edge descriptions.
        """
        if len(tables) < 1:
            raise ValueError("At least one table is required.")
        main = tables[0]
        mgmt = join_type.lower()
        if mgmt not in ("inner", "left"):
            mgmt = "inner"
        clause_word = "LEFT" if mgmt == "left" else "INNER"

        rels = self._collect_relationships(tables)
        used = {main}
        aliases = {main: "t0"}
        joins = []
        joins_sql = []
        alias_counter = [0]

        def _alias(table: str) -> str:
            if table in aliases:
                return aliases[table]
            alias_counter[0] += 1
            a = f"t{alias_counter[0]}"
            aliases[table] = a
            return a

        pending = list(tables)
        pending.remove(main)

        while pending:
            added = None
            for t in list(pending):
                edge = None
                for rel in rels:
                    if rel["table"] in used and rel["ref_table"] == t:
                        edge = rel
                        break
                if edge is not None:
                    owner = edge["table"]
                    fk_col = edge["col"]
                    ref_col = edge["referred_columns"][0] if edge.get("referred_columns") else fk_col
                    a = _alias(t)
                    clause = (f'{clause_word} JOIN "{t}" {a} ON '
                              f'{aliases[owner]}."{fk_col}" = {a}."{ref_col}"')
                    joins.append({"table": t, "via": f"{owner}.{fk_col} = {t}.{ref_col}"})
                    joins_sql.append(clause)
                    used.add(t)
                    pending.remove(t)
                    added = t
                    break
            if added is not None:
                continue
            # Second pass: 't' references an already-used table
            for t in list(pending):
                edge = None
                for rel in rels:
                    if rel["table"] == t and rel["ref_table"] in used:
                        edge = rel
                        break
                if edge is not None:
                    fk_col = edge["col"]
                    ref_col = edge["referred_columns"][0] if edge.get("referred_columns") else fk_col
                    ref_t = edge["ref_table"]
                    a = _alias(t)
                    clause = (f'{clause_word} JOIN "{t}" {a} ON '
                              f'{a}."{fk_col}" = {aliases[ref_t]}."{ref_col}"')
                    joins.append({"table": t, "via": f"{t}.{fk_col} = {ref_t}.{ref_col}"})
                    joins_sql.append(clause)
                    used.add(t)
                    pending.remove(t)
                    added = t
                    break
            if added is None:
                # No FK connects remaining tables; stop to avoid cartesian.
                break

        # Explicit, aliased SELECT to avoid ambiguous column names.
        main_alias = aliases[main]
        select_parts = []
        for table in [main] + [j["table"] for j in joins]:
            a = aliases[table]
            for col in self.get_schema(table):
                select_parts.append(f'{a}."{col["name"]}" AS "{table}__{col["name"]}"')

        query = (
            "SELECT " + ", ".join(select_parts) +
            f' FROM "{main}" {main_alias} ' + " ".join(joins_sql) + ";"
        )
        return {"query": query, "joins": joins}

    def load_auto_join(self, tables: List[str], join_type: str = "inner") -> pd.DataFrame:
        """Load data by auto-joining the given tables via FK metadata."""
        built = self.build_join_query(tables, join_type)
        return self.execute_query(built["query"])

    # ========== Schema drift profiling (Feature 7) ==========

    def set_profile_store(self, path: str) -> None:
        """Point the profile store at a JSON file backing persistent profiles."""
        self._profiles_path = path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._profiles = json.load(f)
            except Exception:
                self._profiles = {}

    def capture_profile(
        self, table: Optional[str] = None, name: str = "", sample_limit: int = 2000
    ) -> Dict[str, Any]:
        """Capture a schema/data-quality snapshot for drift comparison."""
        tables = [table] if table else self.get_tables()
        profile = {"timestamp": time.time(), "name": name or "profile", "tables": []}
        for t in tables:
            try:
                nrows = self.get_row_count(t)
                sample = self.load_table(t, limit=sample_limit)
                cols = []
                for c in self.get_schema(t):
                    colname = c["name"]
                    s = sample[colname] if colname in sample.columns else None
                    cols.append({
                        "name": colname, "type": c["type"], "nullable": c.get("nullable", True),
                        "null_pct": round(float(s.isna().mean() * 100), 2) if s is not None and len(s) else None,
                        "unique_count": int(s.nunique()) if s is not None else None,
                        "dtype": str(s.dtype) if s is not None else "",
                    })
                profile["tables"].append({"table": t, "row_count": nrows, "columns": cols})
            except Exception:
                continue
        self._profiles[name or "profile"] = profile
        if self._profiles_path:
            try:
                with open(self._profiles_path, "w", encoding="utf-8") as f:
                    json.dump(self._profiles, f, indent=2)
            except Exception:
                pass
        return profile

    def list_profiles(self) -> List[str]:
        return list(self._profiles.keys())

    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        return self._profiles.get(name)

    def compare_profiles(self, a: str, b: str) -> Dict[str, Any]:
        """Compare two captured profiles and summarize schema/data drift."""
        pa, pb = self._profiles.get(a), self._profiles.get(b)
        if not pa or not pb:
            raise ValueError("Both profile names must exist.")
        ta = {t["table"]: t for t in pa["tables"]}
        tb = {t["table"]: t for t in pb["tables"]}
        diff = {"from": a, "to": b, "added_tables": [], "removed_tables": [],
                "changed_tables": [], "summary": []}
        for t in tb:
            if t not in ta:
                diff["added_tables"].append(t)
                diff["summary"].append(f"Table added: {t}")
        for t in ta:
            if t not in tb:
                diff["removed_tables"].append(t)
                diff["summary"].append(f"Table removed: {t}")
        for t in tb:
            if t not in ta:
                continue
            changes = []
            if ta[t]["row_count"] != tb[t]["row_count"]:
                changes.append(f"row_count {ta[t]['row_count']} -> {tb[t]['row_count']}")
            ca = {c["name"]: c for c in ta[t]["columns"]}
            for bcol in tb[t]["columns"]:
                acol = ca.get(bcol["name"])
                if acol is None:
                    changes.append(f"column added: {bcol['name']}")
                elif acol["null_pct"] is not None and bcol["null_pct"] is not None and abs(acol["null_pct"] - bcol["null_pct"]) > 5:
                    changes.append(f"'{bcol['name']}' null% {acol['null_pct']} -> {bcol['null_pct']}")
            for acol in ta[t]["columns"]:
                if acol["name"] not in {c["name"] for c in tb[t]["columns"]}:
                    changes.append(f"column removed: {acol['name']}")
            if changes:
                diff["changed_tables"].append({"table": t, "changes": changes})
                diff["summary"].append(f"Table '{t}' changed: {'; '.join(changes)}")
        return diff

    # ========== Saved query library (Feature 2) ==========

    def set_query_store(self, path: str) -> None:
        """Point the saved-query store at a JSON file (persistent)."""
        self._queries_path = path
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    self._queries = json.load(fh)
            except Exception:
                self._queries = {}

    def save_query(self, name: str, query: str, description: str = "") -> str:
        """Save a named query. Returns the query name."""
        key = name.strip()
        if not key:
            raise ValueError("Query name is required.")
        self._queries[key] = {
            "query": query,
            "description": description,
            "saved_at": time.time(),
        }
        self._persist_queries()
        return key

    def list_queries(self) -> List[Dict[str, Any]]:
        return [
            {"name": k, "query": v.get("query"), "description": v.get("description", "")}
            for k, v in self._queries.items()
        ]

    def get_query(self, name: str) -> Optional[Dict[str, Any]]:
        v = self._queries.get(name)
        return {"name": name, "query": v.get("query"), "description": v.get("description", "")} if v else None

    def delete_query(self, name: str) -> bool:
        existed = name in self._queries
        if existed:
            del self._queries[name]
            self._persist_queries()
        return existed

    def _persist_queries(self) -> None:
        if self._queries_path:
            try:
                with open(self._queries_path, "w", encoding="utf-8") as fh:
                    json.dump(self._queries, fh, indent=2)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()