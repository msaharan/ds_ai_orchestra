"""Tooling for the data science LangGraph agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import shorten
from typing import Any, Mapping

from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from .datasets import DatasetCatalog
from .data_prep import ProfileConfig, profile_dataset


MAX_PREVIEW_ROWS = 100
MAX_ANALYSIS_ROWS = 5000


@dataclass
class DataScienceContext:
    """Runtime dependencies wired into the data science tools."""

    catalog: DatasetCatalog

    def load_dataset(self, name: str, *, limit: int | None = None) -> Any:
        overrides: Mapping[str, Any] | None = None
        if limit is not None:
            overrides = {"nrows": limit}
        return self.catalog.load(name, overrides=overrides)


def _get_context() -> DataScienceContext:
    runtime = get_runtime(DataScienceContext)
    return runtime.context


@tool
def list_datasets(verbose: bool = False) -> str:
    """List datasets available in the catalog."""

    context = _get_context()
    lines = []
    for entry in context.catalog.entries():
        summary = entry.name
        if entry.format:
            summary += f" ({entry.format.value})"
        if verbose and entry.description:
            summary += f": {entry.description}"
        lines.append(summary)
    return "\n".join(lines) if lines else "No datasets registered in the catalog."


@tool
def preview_dataset(dataset_name: str, limit: int = 5) -> str:
    """Return a small sample from a dataset."""

    context = _get_context()
    limit = max(1, min(limit, MAX_PREVIEW_ROWS))
    frame = context.load_dataset(dataset_name, limit=limit)
    preview = frame.head(limit).to_string(index=False)
    shape_info = f"rows={frame.shape[0]}, columns={frame.shape[1]}"
    return f"Dataset: {dataset_name}\nShape: {shape_info}\nSample (limit={limit}):\n{preview}"


@tool
def profile_dataset_tool(dataset_name: str, sample_size: int = 5) -> str:
    """Generate a cached dataset profile summarising columns and basic stats."""

    context = _get_context()
    config = ProfileConfig(sample_size=sample_size)
    profile = profile_dataset(context.catalog.get(dataset_name), config=config)
    return json.dumps(profile.to_dict(), indent=2)


@tool
def analyze_dataset(dataset_name: str, objective: str = "summary") -> str:
    """Perform lightweight exploratory analysis for the given dataset."""

    context = _get_context()
    objective_lower = objective.lower()
    frame = context.load_dataset(dataset_name, limit=MAX_ANALYSIS_ROWS)

    results: list[str] = []

    if "missing" in objective_lower or "null" in objective_lower:
        missing = frame.isnull().sum()
        missing_summary = missing[missing > 0]
        if missing_summary.empty:
            results.append("No missing values detected in the loaded sample.")
        else:
            results.append("Missing values per column:\n" + missing_summary.to_string())

    if "describe" in objective_lower or "summary" in objective_lower or "stats" in objective_lower:
        try:
            described = frame.describe(include="all", datetime_is_numeric=True)
        except TypeError:
            described = frame.describe(include="all")
        results.append("Descriptive statistics:\n" + described.to_string())

    if "top" in objective_lower or "frequency" in objective_lower:
        top_lines: list[str] = []
        for column in frame.select_dtypes(include=["object", "category"]).columns:
            value_counts = frame[column].value_counts().head(5)
            formatted = value_counts.to_string()
            top_lines.append(f"Column '{column}' top values:\n{formatted}")
        if top_lines:
            results.extend(top_lines)

    if not results:
        preview_text = frame.head(5).to_string(index=False)
        results.append(
            "Objective not recognised; returning default preview.\n"
            f"Preview:\n{preview_text}"
        )

    heading = f"Analysis summary for '{dataset_name}' (objective='{objective}')"
    return heading + "\n\n" + "\n\n".join(results)


DATA_SCIENCE_TOOLS = [
    list_datasets,
    preview_dataset,
    profile_dataset_tool,
    analyze_dataset,
]


__all__ = [
    "DataScienceContext",
    "DATA_SCIENCE_TOOLS",
    "analyze_dataset",
    "list_datasets",
    "profile_dataset_tool",
    "preview_dataset",
]

