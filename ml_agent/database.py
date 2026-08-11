"""Database processing module for the ML Agent."""
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

    def _normalize(self, cs: str) -> str:
        if cs.endswith((".db", ".sqlite", ".sqlite3")):
            return f"sqlite:///{cs}"
        return cs

    def _connect(self) -> None:
        try:
            self.engine = create_engine(self.connection_string)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
        except Exception as e:
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

    def execute_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.engine)

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

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()