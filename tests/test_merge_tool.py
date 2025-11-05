import json

import pandas as pd

from src.datasets import DatasetCatalog
from src.ds_agent_tools import DataScienceContext, merge_datasets


def _make_catalog(tmp_path):
    left_frame = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "country": ["us", "fr", "de"],
        }
    )
    right_frame = pd.DataFrame(
        {
            "identifier": [1, 2, 4],
            "sales": [10.0, 12.5, 7.2],
        }
    )

    csv_path = tmp_path / "left.csv"
    parquet_path = tmp_path / "right.parquet"

    left_frame.to_csv(csv_path, index=False)
    right_frame.to_parquet(parquet_path, index=False)

    manifest_path = tmp_path / "catalog.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": {
                    "left_dataset": {
                        "uri": str(csv_path),
                        "format": "csv",
                    },
                    "right_dataset": {
                        "uri": str(parquet_path),
                        "format": "parquet",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    catalog = DatasetCatalog(manifest_path=manifest_path, base_path=tmp_path)
    return catalog


def test_merge_datasets_success(monkeypatch, tmp_path):
    catalog = _make_catalog(tmp_path)
    context = DataScienceContext(catalog=catalog)
    monkeypatch.setattr("src.ds_agent_tools._get_context", lambda: context)

    output = merge_datasets.invoke(
        {
            "left_dataset": "left_dataset",
            "right_dataset": "right_dataset",
            "left_on": "id",
            "right_on": "identifier",
            "how": "left",
            "limit": 3,
        }
    )

    assert "Merged shape" in output
    assert "Join type: left" in output
    assert "left_dataset" not in output  # ensure we don't leak reprs
    assert "id" in output
    assert "sales" in output


def test_merge_datasets_unknown_join(monkeypatch, tmp_path):
    catalog = _make_catalog(tmp_path)
    context = DataScienceContext(catalog=catalog)
    monkeypatch.setattr("src.ds_agent_tools._get_context", lambda: context)

    output = merge_datasets.invoke(
        {
            "left_dataset": "left_dataset",
            "right_dataset": "right_dataset",
            "left_on": "id",
            "right_on": "identifier",
            "how": "invalid",
        }
    )

    assert "Unsupported join type" in output


def test_merge_datasets_missing_column(monkeypatch, tmp_path):
    catalog = _make_catalog(tmp_path)
    context = DataScienceContext(catalog=catalog)
    monkeypatch.setattr("src.ds_agent_tools._get_context", lambda: context)

    output = merge_datasets.invoke(
        {
            "left_dataset": "left_dataset",
            "right_dataset": "right_dataset",
            "left_on": "does_not_exist",
            "right_on": "identifier",
        }
    )

    assert "Columns not found in left dataset" in output

