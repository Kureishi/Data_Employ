"""Safe SQL validation for the ML Agent.

Checks a SQL statement against the live schema so that user- or
LLM-generated SQL is safe to run: verifies table/column references,
flags write statements (respecting read-only safety), and warns about
expensive or ambiguous patterns (cartesian joins, missing LIMIT, SELECT *).
"""
import re
from typing import Any, Dict, List

# SQL write verbs that must never run when read-only safety is on.
_WRITE_KEYWORDS = (
    "insert", "update", "delete", "drop", "alter", "create",
    "truncate", "grant", "revoke", "replace", "vacuum", "merge",
)

_FROM_OR_JOIN = re.compile(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_.$]*)\b", re.I)
_REF = re.compile(r"[\"`]?([A-Za-z_][A-Za-z0-9_]*)[\"`]?\s*\(", re.I)

_RESERVED = {
    "select", "from", "where", "join", "on", "and", "or", "group", "by",
    "order", "limit", "as", "distinct", "count", "sum", "avg", "min", "max",
    "case", "when", "then", "else", "end", "having", "union", "all", "left",
    "right", "inner", "outer", "cross", "not", "null", "is", "in", "like",
    "between", "exists", "asc", "desc",
}


class SQLValidator:
    """Validates a SQL statement against the connected database schema."""

    def __init__(self, db: Any) -> None:
        self.db = db

    def _tables(self) -> set:
        try:
            return set(self.db.get_tables())
        except Exception:
            return set()

    def _schema_columns(self, table: str) -> set:
        try:
            return {c["name"] for c in self.db.get_schema(table)}
        except Exception:
            return set()

    def statement_type(self, sql: str) -> str:
        """Return 'select', 'write', or 'unknown' for the statement head."""
        head = re.split(r"\s+", sql.lstrip().lstrip("(").strip(), maxsplit=1)[0].lower()
        if head in _WRITE_KEYWORDS:
            return "write"
        if head == "select":
            return "select"
        return "unknown"

    def _referenced_tables(self, sql: str) -> List[str]:
        """Names of tables referenced by FROM/JOIN clauses (deduplicated)."""
        found = []
        # Match FROM <clause> up to a following keyword, and each JOIN <table>
        from_match = re.search(r"\bfrom\s+([A-Za-z_\"]+[\w\s,.\"]*?)(?=\b(?:where|group\s+by|order\s+by|join|having|limit|union)\b|;|$)", sql, re.I | re.S)
        candidates = []
        if from_match:
            candidates.append(from_match.group(1))
        for m in re.finditer(r"\bjoin\s+([A-Za-z_\"][\w\.\"]*)", sql, re.I):
            candidates.append(m.group(1))
        for token in candidates:
            token = token.strip().strip("\"'`")
            for chunk in token.split(","):
                raw = chunk.strip().strip("\"'`")
                raw = re.sub(r"\s+", " ", raw)
                name = raw.split()[0] if raw else ""
                name = name.split(".")[-1] if "." in name else name
                name = name.strip("\"'`")
                if name and name.lower() not in {"select"} and name not in found:
                    found.append(name)
        return found

    def validate(self, sql: str, read_only: bool = True) -> Dict[str, Any]:
        """Validate a SQL statement and return a structured report."""
        sql = (sql or "").strip()
        errors: List[str] = []
        warnings: List[str] = []
        notes: List[str] = []

        if not sql:
            return {
                "sql": sql, "statement_type": "unknown", "valid": False,
                "errors": ["Empty SQL statement."], "warnings": [],
                "tables": [], "notes": [],
            }

        stype = self.statement_type(sql)
        if stype == "write":
            where = "Blocked by read-only mode." if read_only else "Write statements are allowed only when read-only mode is disabled."
            errors.append(f"Write statement detected ({stype}). {where}")
        elif stype == "unknown":
            warnings.append("Statement type could not be recognised (expected SELECT).")

        known_tables = self._tables()
        tables = self._referenced_tables(sql)
        for t in tables:
            if known_tables and t not in known_tables:
                errors.append(f"Unknown table '{t}' referenced in query.")

        # Column reference check against the referenced tables' schemas
        columns = {c.group(1).strip().strip("\"`") for c in _REF.finditer(sql)} - _RESERVED
        if tables:
            known_cols = set()
            for t in tables:
                if t in known_tables:
                    known_cols |= self._schema_columns(t)
            for col in columns:
                if known_cols and col not in known_cols:
                    warnings.append(
                        f"Column '{col}' not found in the referenced table(s) — it may be a typo or alias."
                    )
        return self._finish(sql, stype, errors, warnings, notes, tables, len(columns))

    def _finish(self, sql, stype, errors, warnings, notes, tables, col_count):
        upper = sql.upper()
        join_count = len(re.findall(r"\bJOIN\b", upper, re.I))
        on_count = len(re.findall(r"\bON\b", upper, re.I))
        if len(tables) > 1 and join_count > on_count:
            warnings.append(
                "Multiple tables are referenced with fewer ON conditions than JOINs — "
                "this may cause an unintended cartesian (cross) join."
            )
        if re.search(r"\bSELECT\s+\*", upper, re.I):
            warnings.append(
                "SELECT * fetches all columns — prefer selecting only the columns you need."
            )
        if "WHERE" not in upper and "GROUP BY" not in upper and len(tables) == 1 and col_count == 0:
            warnings.append(
                "Query has no WHERE clause — it will scan/return every row of the table."
            )
        if "LIMIT" not in upper:
            notes.append("No LIMIT clause — consider capping the result size on large tables.")
        if tables:
            notes.append("Tables referenced: " + ", ".join(tables))

        return {
            "sql": sql, "statement_type": stype,
            "valid": len(errors) == 0,
            "errors": errors, "warnings": warnings,
            "tables": tables, "notes": notes,
        }