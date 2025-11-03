"""Utility tools for the SQL agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import re
import time
from typing import Deque, Optional

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

DENY_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b", re.I
)
HAS_LIMIT_TAIL_RE = re.compile(r"(?is)\blimit\b\s+\d+(\s*,\s*\d+)?\s*;?\s*$")
LIMIT_VALUE_RE = re.compile(r"limit\s+(?P<rows>\d+)(\s*,\s*\d+)?", re.IGNORECASE)

MAX_QUERIES_PER_WINDOW = 8
QUERY_WINDOW_SECONDS = 10.0
_QUERY_HISTORY: Deque[float] = deque(maxlen=MAX_QUERIES_PER_WINDOW)


@dataclass
class RuntimeContext:
    """Runtime dependencies available to tools."""

    db: SQLDatabase
    max_rows: int = 5


def _rate_limited() -> Optional[str]:
    """Enforce a simple process-wide rate limit for data access tools."""

    now = time.monotonic()
    if len(_QUERY_HISTORY) == _QUERY_HISTORY.maxlen:
        oldest = _QUERY_HISTORY[0]
        elapsed = now - oldest
        if elapsed < QUERY_WINDOW_SECONDS:
            retry_after = QUERY_WINDOW_SECONDS - elapsed
            return (
                "Error: Too many SQL requests in a short period. "
                f"Try again in {retry_after:.1f} seconds."
            )
        _QUERY_HISTORY.popleft()
    _QUERY_HISTORY.append(now)
    return None


def _safe_sql(query: str, max_rows: int) -> str:
    query = query.strip()
    if query.count(";") > 1 or (query.endswith(";") and ";" in query[:-1]):
        return "Error: multiple statements are not allowed."
    query = query.rstrip(";").strip()

    if not query.lower().startswith("select"):
        return "Error: only SELECT statements are allowed."
    if DENY_RE.search(query):
        return "Error: DML/DDL detected. Only read-only queries are permitted."

    match = LIMIT_VALUE_RE.search(query)
    if match:
        rows = int(match.group("rows"))
        if rows > max_rows:
            return (
                "Error: Requested LIMIT exceeds the configured maximum "
                f"({max_rows})."
            )
    elif not HAS_LIMIT_TAIL_RE.search(query):
        query += f" LIMIT {max_rows}"
    return query


@tool
def execute_sql(query: str) -> str:
    """Execute a read-only SQLite SELECT query (auto-limited)."""

    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db
    max_rows = runtime.context.max_rows

    rate_error = _rate_limited()
    if rate_error:
        return rate_error

    safe_query = _safe_sql(query, max_rows)
    if safe_query.startswith("Error:"):
        return safe_query
    try:
        return db.run(safe_query)
    except Exception as exc:  # pragma: no cover - database error surface only
        return f"Error: {exc}"


@tool
def list_tables() -> str:
    """Return available table names in the database."""

    runtime = get_runtime(RuntimeContext)
    tables = runtime.context.db.get_usable_table_names()
    if not tables:
        return "No accessible tables were found."
    return "\n".join(sorted(tables))


@tool
def describe_table(table_name: str) -> str:
    """Describe the columns for a specific table."""

    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db

    rate_error = _rate_limited()
    if rate_error:
        return rate_error

    table_name = table_name.strip()
    if not table_name:
        return "Error: table_name must not be empty."

    try:
        pragma_query = f"PRAGMA table_info({table_name});"
        results = db.run(pragma_query)
        if not results:
            return f"Table '{table_name}' not found."
        header = "cid | name | type | notnull | default | pk"
        rows = [header, "-" * len(header)]
        rows.extend(
            " | ".join(str(value) for value in row)
            for row in results
        )
        return "\n".join(rows)
    except Exception as exc:  # pragma: no cover - database error surface only
        return f"Error: {exc}"


SQL_TOOLS = [execute_sql, list_tables, describe_table]

__all__ = [
    "RuntimeContext",
    "execute_sql",
    "list_tables",
    "describe_table",
    "SQL_TOOLS",
    "MAX_QUERIES_PER_WINDOW",
    "QUERY_WINDOW_SECONDS",
]
