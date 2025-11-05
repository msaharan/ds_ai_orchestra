"""Tooling for the data science LangGraph agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from langchain_core.tools import tool
from langgraph.runtime import get_runtime

from .datasets import DatasetCatalog
from .data_prep import ProfileConfig, profile_dataset
from .ds_agent_templates import available_task_types, get_task_template, template_catalog


MAX_PREVIEW_ROWS = 100
MAX_ANALYSIS_ROWS = 5000
MAX_MERGE_PREVIEW_ROWS = 20

_VALID_JOIN_TYPES = {"inner", "left", "right", "outer", "cross"}


@dataclass
class DataScienceContext:
    """Runtime dependencies wired into the data science tools."""

    catalog: DatasetCatalog

    def load_dataset(self, name: str, *, limit: int | None = None) -> Any:
        overrides: Mapping[str, Any] | None = None
        if limit is not None:
            overrides = {"nrows": limit}

        try:
            frame = self.catalog.load(name, overrides=overrides)
        except TypeError as exc:
            if limit is None or "nrows" not in str(exc):
                raise
            frame = self.catalog.load(name)

        if limit is not None:
            frame = frame.head(limit)
        return frame


def _get_context() -> DataScienceContext:
    runtime = get_runtime(DataScienceContext)
    return runtime.context


def _normalise_join_columns(value: str) -> list[str]:
    parts = [item.strip() for item in value.split(",")]
    columns = [item for item in parts if item]
    if not columns:
        raise ValueError("Join column specification must not be empty.")
    return columns


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


@tool
def list_task_templates(include_tips: bool = False) -> str:
    """Return the catalog of predefined analytical task templates in JSON form."""

    catalog_payload: list[Mapping[str, Any]] = []
    for entry in template_catalog():
        payload = dict(entry)
        if not include_tips:
            payload.pop("tips", None)
        catalog_payload.append(payload)
    return json.dumps({"templates": catalog_payload}, indent=2)


@tool
def task_template_details(task_type: str, include_tips: bool = True) -> str:
    """Fetch a specific task template, returning structured guidance for the agent."""

    try:
        template = get_task_template(task_type)
    except KeyError:
        choices = ", ".join(available_task_types())
        return (
            f"Unknown task template '{task_type}'. "
            f"Available templates: {choices}."
        )
    payload: Mapping[str, Any] = template.to_dict()
    if not include_tips:
        payload = {k: v for k, v in payload.items() if k != "tips"}
    return json.dumps(payload, indent=2)


@tool
def merge_datasets(
    left_dataset: str,
    right_dataset: str,
    *,
    left_on: str,
    right_on: str | None = None,
    how: str = "inner",
    limit: int = 10,
) -> str:
    """Join two catalog datasets on the specified keys and preview the merged sample."""

    context = _get_context()

    join_type = how.lower()
    if join_type not in _VALID_JOIN_TYPES:
        options = ", ".join(sorted(_VALID_JOIN_TYPES))
        return f"Unsupported join type '{how}'. Choose from: {options}."

    try:
        left_keys = _normalise_join_columns(left_on)
    except ValueError as exc:
        return f"Invalid left join keys: {exc}"

    right_key_arg = right_on if right_on is not None else left_on
    try:
        right_keys = _normalise_join_columns(right_key_arg)
    except ValueError as exc:
        return f"Invalid right join keys: {exc}"

    left_frame = context.load_dataset(left_dataset, limit=MAX_ANALYSIS_ROWS)
    right_frame = context.load_dataset(right_dataset, limit=MAX_ANALYSIS_ROWS)

    missing_left = [column for column in left_keys if column not in left_frame.columns]
    if missing_left:
        missing = ", ".join(missing_left)
        return (
            f"Columns not found in left dataset '{left_dataset}': {missing}."
            " Verify the join keys."
        )

    missing_right = [column for column in right_keys if column not in right_frame.columns]
    if missing_right:
        missing = ", ".join(missing_right)
        return (
            f"Columns not found in right dataset '{right_dataset}': {missing}."
            " Verify the join keys."
        )

    left_key_arg: str | list[str]
    right_key_arg_final: str | list[str]

    if len(left_keys) == 1:
        left_key_arg = left_keys[0]
    else:
        left_key_arg = left_keys

    if len(right_keys) == 1:
        right_key_arg_final = right_keys[0]
    else:
        right_key_arg_final = right_keys

    try:
        merged = left_frame.merge(
            right_frame,
            how=join_type,
            left_on=left_key_arg,
            right_on=right_key_arg_final,
            suffixes=("_left", "_right"),
        )
    except Exception as exc:  # pragma: no cover - defensive fallback
        return f"Failed to merge datasets: {exc}"

    preview_limit = max(1, min(limit, MAX_MERGE_PREVIEW_ROWS))
    preview_table = merged.head(preview_limit).to_string(index=False)

    metadata_lines = [
        f"Left shape: rows={left_frame.shape[0]}, columns={left_frame.shape[1]}",
        f"Right shape: rows={right_frame.shape[0]}, columns={right_frame.shape[1]}",
        f"Merged shape: rows={merged.shape[0]}, columns={merged.shape[1]}",
        f"Join type: {join_type}",
        f"Join keys: left={left_keys}, right={right_keys}",
    ]

    return (
        "\n".join(metadata_lines)
        + "\n\nPreview (limit="
        + str(preview_limit)
        + "):\n"
        + preview_table
    )


DATA_SCIENCE_TOOLS = [
    list_datasets,
    preview_dataset,
    profile_dataset_tool,
    analyze_dataset,
    list_task_templates,
    task_template_details,
    merge_datasets,
]


__all__ = [
    "DataScienceContext",
    "DATA_SCIENCE_TOOLS",
    "analyze_dataset",
    "list_datasets",
    "list_task_templates",
    "merge_datasets",
    "profile_dataset_tool",
    "preview_dataset",
    "task_template_details",
]

