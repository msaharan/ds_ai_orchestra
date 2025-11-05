"""Unified CLI entrypoint for selecting SQL or data science agents."""

from __future__ import annotations

import argparse
from typing import Iterable


def parse_args(argv: Iterable[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Unified agent launcher", add_help=False)
    parser.add_argument(
        "--agent",
        choices=["sql", "data_science"],
        default="sql",
        help="Select which agent to launch (default: %(default)s)",
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show help for this launcher or the selected agent.",
    )
    known, remaining = parser.parse_known_args(list(argv) if argv is not None else None)
    return known, remaining


def main(argv: Iterable[str] | None = None) -> None:
    known, remaining = parse_args(argv)

    if known.agent == "sql":
        from . import sql_agent

        if known.help:
            sql_agent.parse_cli_args(["--help"])  # Will exit after printing help
            return
        sql_agent.main(remaining)
    else:
        from . import ds_agent

        if known.help:
            ds_agent.parse_cli_args(["--help"])  # Will exit after printing help
            return
        ds_agent.main(remaining)


if __name__ == "__main__":  # pragma: no cover - invoked manually
    main()

