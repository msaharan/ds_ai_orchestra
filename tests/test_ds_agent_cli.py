from types import SimpleNamespace

import pytest

from src.ds_agent import build_system_prompt, parse_cli_args


class _StubCatalog:
    def entries(self):
        yield SimpleNamespace(name="alpha")
        yield SimpleNamespace(name="beta")


def test_build_system_prompt_includes_mcp_summary():
    prompt = build_system_prompt(_StubCatalog(), mcp_summary="remote_fetch, s3_export")
    assert "remote_fetch" in prompt
    assert "s3_export" in prompt


def test_parse_cli_args_accepts_mcp_config():
    args = parse_cli_args(
        [
            "--mcp-config",
            '{"servers": {"demo": {"transport": "stdio", "command": "echo", "args": ["hi"]}}}',
        ]
    )
    assert args.mcp_config.startswith("{")


def test_parse_cli_args_conflicting_stream_flags():
    with pytest.raises(SystemExit):
        parse_cli_args(["--event-stream", "--no-stream"])

