"""Helpers for loading MCP-backed tools."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.tools import StructuredTool, Tool


def _ensure_asyncio() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@dataclass
class MCPConfig:
    """Configuration for connecting to one or more MCP servers."""

    servers: Dict[str, Dict[str, Any]]
    use_standard_content_blocks: bool = True


DEFAULT_TIME_MCP = MCPConfig(
    servers={
        "time": {
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "@theo.foobar/mcp-time"],
        }
    }
)


def load_mcp_config(value: str) -> MCPConfig:
    """Parse an MCP configuration from JSON content or a file path."""

    path = Path(value)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = json.loads(value)

    if not isinstance(data, dict) or "servers" not in data:
        raise ValueError("MCP configuration must contain a 'servers' object")
    use_standard = data.get("use_standard_content_blocks", True)
    servers = data["servers"]
    if not isinstance(servers, dict) or not servers:
        raise ValueError("MCP configuration requires at least one server definition")
    return MCPConfig(servers=servers, use_standard_content_blocks=use_standard)


def load_mcp_tools(config: MCPConfig) -> List[Any]:
    """Connect to configured MCP servers and return LangChain Tool objects."""

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError(
            "langchain-mcp-adapters is required for MCP integration"
        ) from exc

    async def _load() -> List[Any]:
        try:
            client = MultiServerMCPClient(
                config.servers,
                use_standard_content_blocks=config.use_standard_content_blocks,
            )
        except TypeError:
            client = MultiServerMCPClient(config.servers)
        tools = await client.get_tools()
        return [_wrap_mcp_tool(tool) for tool in tools]

    loop = _ensure_asyncio()
    if loop.is_running():  # pragma: no cover - not expected in CLI
        return loop.run_until_complete(_load())
    return loop.run_until_complete(_load())


def merge_tools(*groups: Iterable[Any]) -> List[Any]:
    tools: List[Any] = []
    for group in groups:
        if group:
            tools.extend(list(group))
    return tools


def _wrap_mcp_tool(tool: Any) -> Any:
    """Ensure MCP tools expose a synchronous interface for LangGraph."""

    if hasattr(tool, "run") and getattr(tool, "run", None):
        return tool

    name = getattr(tool, "name", "mcp_tool")
    description = getattr(tool, "description", "MCP tool")
    args_schema = getattr(tool, "args_schema", None)

    async def _arun(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    def _run(**kwargs: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(tool.ainvoke(kwargs))
        else:  # pragma: no cover - not expected in threads without loop
            future = asyncio.run_coroutine_threadsafe(tool.ainvoke(kwargs), loop)
            return future.result()

    if args_schema is not None:
        return StructuredTool.from_function(
            func=_run,
            coroutine=_arun,
            name=name,
            description=description,
            args_schema=args_schema,
        )

    def _run_simple(tool_input: Any) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(tool.ainvoke(tool_input))
        else:  # pragma: no cover
            future = asyncio.run_coroutine_threadsafe(
                tool.ainvoke(tool_input), loop
            )
            return future.result()

    async def _arun_simple(tool_input: Any) -> Any:
        return await tool.ainvoke(tool_input)

    return Tool.from_function(
        func=_run_simple,
        coroutine=_arun_simple,
        name=name,
        description=description,
    )


__all__ = [
    "MCPConfig",
    "DEFAULT_TIME_MCP",
    "load_mcp_config",
    "load_mcp_tools",
    "merge_tools",
]
