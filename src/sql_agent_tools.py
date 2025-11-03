"""Utility tools for the SQL agent."""

from __future__ import annotations

from dataclasses import dataclass
import re

from langchain_community.utilities import SQLDatabase
from langchain_core.tools import tool
from langgraph.runtime import get_runtime

DENY_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|ALTER|DROP|CREATE|REPLACE|TRUNCATE)\b", re.I
)


@dataclass
class RuntimeContext:
    """Runtime dependencies available to tools."""

    db: SQLDatabase


def _ensure_read_only(query: str) -> str:
    query = query.strip().rstrip(";").strip()
    if not query:
        return "Error: query must not be empty."

    if not query.lower().startswith("select"):
        return "Error: only SELECT statements are allowed."
    if DENY_RE.search(query):
        return "Error: DML/DDL detected. Only read-only queries are permitted."

    return query


@tool
def execute_sql(query: str) -> str:
    """Execute a read-only SQLite SELECT query."""

    runtime = get_runtime(RuntimeContext)
    db = runtime.context.db
    safe_query = _ensure_read_only(query)
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
]
