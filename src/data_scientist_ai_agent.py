"""SQL agent for studio."""

from __future__ import annotations

import argparse
import asyncio
import atexit
import os
import re
from datetime import datetime, timezone
import json
import pathlib
import sys
import time
from typing import Any, Optional, Tuple, Type

import requests
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - optional dependency
    SqliteSaver = None  # type: ignore[assignment]

try:
    from redis.asyncio import Redis  # type: ignore
    from langgraph.checkpoint.redis import RedisSaver  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Redis = None  # type: ignore[assignment]
    RedisSaver = None  # type: ignore[assignment]

from pydantic import BaseModel, ValidationError
from .data_scientist_ai_agent_mcp import (
    DEFAULT_TIME_MCP,
    MCPConfig,
    load_mcp_config,
    load_mcp_tools,
)
from .data_scientist_ai_agent_tools import (
    MAX_QUERIES_PER_WINDOW,
    QUERY_WINDOW_SECONDS,
    RuntimeContext,
    SQL_TOOLS,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load .env file - this file is mandatory
from dotenv import load_dotenv
env_path = PROJECT_ROOT / ".env"
if not env_path.exists():
    raise FileNotFoundError(
        f"The .env file is mandatory but was not found at {env_path}. "
        f"Please create a .env file in the project root directory. "
        f"You can copy example.env to .env as a starting point."
    )
load_dotenv(env_path)

try:
    from .env_utils import doublecheck_env
except ImportError:  # pragma: no cover - optional outside primary project root
    doublecheck_env = None  # type: ignore[assignment]


CHINOOK_URL = "https://storage.googleapis.com/benchmarks-artifacts/chinook/Chinook.db"
DEFAULT_DB_PATH = pathlib.Path(__file__).resolve().parents[2] / "Chinook.db"
DEFAULT_MODEL = "openai:gpt-3.5-turbo"


class ConversationLogger:
    """Append structured message payloads to a JSONL file."""

    def __init__(self, path: pathlib.Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self,
        message: BaseMessage,
        *,
        direction: str,
        step_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "direction": direction,
            "message": serialize_message(message),
        }
        if step_metadata:
            entry["step_metadata"] = step_metadata
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, default=str))
            handle.write("\n")


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    """Convert a LangChain message into a JSON-serializable payload."""

    payload: dict[str, Any] = {
        "type": message.__class__.__name__,
        "role": getattr(message, "type", None),
        "content": message.content,
        "additional_kwargs": message.additional_kwargs,
    }
    response_metadata = getattr(message, "response_metadata", None)
    if response_metadata:
        payload["response_metadata"] = response_metadata
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    metadata = getattr(message, "metadata", None)
    if metadata:
        payload["metadata"] = metadata
    return payload


class InvoiceSummary(BaseModel):
    """Structured representation of an invoice answer."""

    customer_name: Optional[str] = None
    invoice_total: Optional[float] = None
    currency: Optional[str] = None
    notes: Optional[str] = None


STRUCTURED_PROMPTS = {
    "invoice_summary": (
        InvoiceSummary,
        "When answering, return ONLY valid JSON matching this schema::\n"
        "{\n"
        '  "customer_name": string | null,\n'
        '  "invoice_total": number | null,\n'
        '  "currency": string | null,  # ISO currency code such as "USD" or "EUR"\n'
        '  "notes": string | null\n'
        "}\n"
        "Use null when a field is unknown. Do not include extra commentary outside the JSON.",
        (
            "Respond with ONLY JSON conforming to the InvoiceSummary schema described above. Do not add prose or explanations. "
            "Do not translate field names. If currency is unspecified, default to 'USD'."
        ),
    )
}

DEFAULT_TONE_PROMPT = "\n- Provide concise answers with clear takeaways."
DEFAULT_RESPONSE_SUFFIX = "Respond in English with concise, decision-ready phrasing."

AGG_FUNCTION_MARKERS = ("max(", "min(", "count(", "sum(", "avg(", "distinct ")


CUSTOMER_IDENTITY_PATTERNS = [
    re.compile(r"\b(?:this is|ik ben|i am)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:same\s+for|for|about|regarding|voor)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b",
        re.IGNORECASE,
    ),
]


def extract_customer_identity(text: str) -> Optional[Tuple[str, str]]:
    for pattern in CUSTOMER_IDENTITY_PATTERNS:
        match = pattern.search(text)
        if match:
            first, last = match.group(1), match.group(2)
            return first.capitalize(), last.capitalize()
    return None


def is_query_risky(query: str) -> bool:
    normalized = query.lower()
    if "sqlite_master" in normalized or "pragma" in normalized:
        return True
    if "select *" in normalized:
        return True
    has_limit = " limit " in normalized or normalized.rstrip().endswith("limit 1")
    has_aggregate = any(func in normalized for func in AGG_FUNCTION_MARKERS)
    if "customer" in normalized:
        has_first = "firstname" in normalized
        has_last = "lastname" in normalized
        if not (has_first and has_last):
            return True
    if not has_limit and not has_aggregate:
        return True
    return False


def build_system_prompt(
    db: SQLDatabase,
    max_rows: int,
    mcp_summary: Optional[str] = None,
    structured_instructions: Optional[str] = None,
) -> str:
    schema = db.get_table_info()
    mcp_lines = f"\n- MCP tools available: {mcp_summary}" if mcp_summary else ""
    structured_lines = (
        "\n- Structured output requirement: " + structured_instructions
        if structured_instructions
        else ""
    )
    tone_lines = DEFAULT_TONE_PROMPT
    locale_lines = "\n- Respond in English with concise, decision-ready phrasing."

    return f"""You are a careful SQLite analyst.
{tone_lines}{locale_lines}

Authoritative schema (do not invent columns/tables):
{schema}

Rules:
- Think step-by-step.
- When you need data, call the tool `execute_sql` with ONE SELECT query.
- Use `list_tables` to refresh yourself on the tables that exist.
- Use `describe_table` to inspect a table's columns before querying it.
- Read-only only; no INSERT/UPDATE/DELETE/ALTER/DROP/CREATE/REPLACE/TRUNCATE.
- SQL responses are limited to {max_rows} rows unless the user has explicit authorization.
- If the tool returns 'Error:', revise the SQL and try again.
- Limit the number of attempts to 5. 
- If you are not successful after 5 attempts, return a note to the user.
- Prefer explicit column lists; avoid SELECT *.
- If the user identifies themselves (e.g. "This is <First Last>"), ensure every SQL query filters on that customer's first and last name. If you cannot match the name, ask for clarification before answering.
- Data tools are rate limited to {MAX_QUERIES_PER_WINDOW} queries every {QUERY_WINDOW_SECONDS:.0f} seconds. Consolidate requests when possible.{mcp_lines}{structured_lines}
"""


def build_agent_and_context(
    model_name: str = DEFAULT_MODEL,
    db_path: pathlib.Path = DEFAULT_DB_PATH,
    *,
    max_rows: int = 5,
    mcp_tools: Optional[list[Any]] = None,
    checkpointer: Optional[Any] = None,
    structured_instructions: Optional[str] = None,
    hitl_enabled: bool = False,
) -> tuple[Any, RuntimeContext]:
    """Construct the SQL agent alongside its runtime context."""

    ensure_database(db_path)
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    runtime_context = RuntimeContext(db=db, max_rows=max_rows)

    base_tools = list(SQL_TOOLS)
    mcp_tool_names: Optional[str] = None
    if mcp_tools:
        base_tools.extend(mcp_tools)
        mcp_tool_names = ", ".join(
            sorted(tool.name for tool in mcp_tools if getattr(tool, "name", None))
        )

    middleware = []
    if hitl_enabled:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={
                    "execute_sql": {"allowed_decisions": ["approve", "reject"]}
                }
            )
        )

    agent = create_agent(
        model=init_chat_model(model_name),
        tools=base_tools,
        system_prompt=build_system_prompt(
            db,
            max_rows,
            mcp_tool_names,
            structured_instructions,
        ),
        context_schema=RuntimeContext,
        checkpointer=checkpointer or InMemorySaver(),
        middleware=middleware,
    )

    return agent, runtime_context


def ensure_database(db_path: pathlib.Path = DEFAULT_DB_PATH) -> pathlib.Path:
    """Download the Chinook database if it does not already exist."""

    if db_path.exists():
        print(f"{db_path} already exists, skipping download.")
        return db_path

    db_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(CHINOOK_URL, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to download the database. Status code: {response.status_code}"
        )

    db_path.write_bytes(response.content)
    print(f"File downloaded and saved as {db_path}")
    return db_path


def validate_environment(example_env: str = "example.env") -> None:
    """Optionally ensure expected environment variables are present."""

    if doublecheck_env is None:  # pragma: no cover - optional helper
        return

    try:
        doublecheck_env(example_env)
    except Exception as exc:  # pragma: no cover - informative warning only
        print(f"Warning: environment validation failed: {exc}")


def run_cli(
    agent: Any,
    runtime_context: RuntimeContext,
    *,
    thread_id: str = "demo-thread-1",
    logger: Optional[ConversationLogger] = None,
    event_stream: bool = False,
    disable_stream: bool = False,
    structured_model: Optional[Type[BaseModel]] = None,
    structured_suffix: Optional[str] = None,
    hitl_enabled: bool = False,
    hitl_auto_approve: bool = False,
) -> None:
    """Interactive REPL for the SQL agent."""

    config = {"configurable": {"thread_id": thread_id}}
    turn_index = 0
    active_customer: Optional[Tuple[str, str]] = None

    if hitl_enabled and event_stream:
        print("[HITL] Event streaming is disabled while human-in-the-loop mode is active.")
        event_stream = False

    print("=" * 60)
    print("SQL Agent with Memory - Interactive Mode")
    print("Type 'quit' or 'exit' to end the conversation")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in {"quit", "exit", "q"}:
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            turn_index += 1
            timestamp = datetime.now(timezone.utc).isoformat()
            header = f"--- Turn {turn_index} | thread={thread_id} | {timestamp} ---"
            print(header)
            extracted = extract_customer_identity(user_input)
            if extracted is not None:
                active_customer = extracted
            content = user_input
            if structured_suffix:
                content = f"{user_input}\n\n{structured_suffix}"
            if active_customer is not None:
                first, last = active_customer
                customer_instruction = (
                    "Active customer context: {first} {last}. Every SQL query MUST include "
                    "filters on Customer.FirstName = '{first}' AND Customer.LastName = '{last}'. "
                    "If no rows match, ask the user which customer they mean rather than "
                    "guessing."
                ).format(first=first, last=last)
                content = f"{content}\n\n{customer_instruction}"
            else:
                if structured_model is not None:
                    content = (
                        f"{content}\n\nIf the request does not specify a first and last name, produce JSON with "
                        "all fields null except `notes`, clarifying that you require the customer's "
                        "full name before querying."
                    )
                else:
                    content = (
                        f"{content}\n\nIf the request does not specify a first and last name, ask the user to "
                        "identify the customer before querying."
                    )
            if structured_model is None:
                content = f"{content}\n\n{DEFAULT_RESPONSE_SUFFIX}"

            user_message = HumanMessage(
                content=content,
                additional_kwargs={
                    "source": "cli",
                    "turn_index": turn_index,
                    "timestamp": timestamp,
                },
            )
            if logger:
                logger.log(
                    user_message,
                    direction="outgoing",
                    step_metadata={
                        "turn_index": turn_index,
                        "active_customer": " ".join(active_customer) if active_customer else None,
                    },
                )

            start = time.perf_counter()
            final_ai_message: Optional[BaseMessage] = None

            print()
            if hitl_enabled:
                result = agent.invoke(
                    {"messages": [user_message]},
                    config=config,
                    context=runtime_context,
                )
                result = resolve_hitl_interrupts(
                    agent,
                    result,
                    config,
                    runtime_context,
                    auto_approve=hitl_auto_approve,
                )
                messages = result.get("messages", []) if isinstance(result, dict) else []
                if messages:
                    final_ai_message = messages[-1]
                    if structured_model is None:
                        final_ai_message.pretty_print()
            elif event_stream:
                final_ai_message = asyncio.run(
                    _event_stream_interaction(
                        agent,
                        user_message,
                        config=config,
                        runtime_context=runtime_context,
                        turn_index=turn_index,
                    )
                )
            elif disable_stream:
                result = agent.invoke(
                    {"messages": [user_message]},
                    config=config,
                    context=runtime_context,
                )
                messages = result.get("messages", []) if isinstance(result, dict) else []
                if messages:
                    final_ai_message = messages[-1]
                    if structured_model is None:
                        final_ai_message.pretty_print()
            else:
                for step in agent.stream(
                    {"messages": [user_message]},
                    config,
                    context=runtime_context,
                    stream_mode="values",
                ):
                    latest_message = step["messages"][-1]
                    final_ai_message = latest_message
                    if logger:
                        step_metadata = {
                            "turn_index": turn_index,
                            "step_type": step.get("type") if isinstance(step, dict) else None,
                            "active_customer": " ".join(active_customer)
                            if active_customer
                            else None,
                        }
                        logger.log(
                            latest_message,
                            direction="incoming",
                            step_metadata=step_metadata,
                        )
                    if structured_model is None:
                        latest_message.pretty_print()

            elapsed = time.perf_counter() - start
            print(f"\n[elapsed: {elapsed:.2f}s]\n")

            if logger and final_ai_message is not None:
                logger.log(
                    final_ai_message,
                    direction="incoming",
                    step_metadata={
                        "turn_index": turn_index,
                        "elapsed_seconds": elapsed,
                        "active_customer": " ".join(active_customer)
                        if active_customer
                        else None,
                    },
                )

            if structured_model is not None and final_ai_message is not None:
                _display_structured_output(structured_model, final_ai_message)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


async def _event_stream_interaction(
    agent: Any,
    user_message: HumanMessage,
    *,
    config: dict[str, Any],
    runtime_context: RuntimeContext,
    turn_index: int,
) -> Optional[BaseMessage]:
    """Stream detailed agent events to the console and return the final AI message."""

    final_ai_message: Optional[BaseMessage] = None
    saw_text_chunk = False

    async for event in agent.astream_events(  # type: ignore[attr-defined]
        {"messages": [user_message]},
        config=config,
        context=runtime_context,
    ):
        event_type = event.get("event") if isinstance(event, dict) else None
        data = event.get("data") if isinstance(event, dict) else None
        if data is None:
            data = {}
        name = event.get("name") or data.get("name") if isinstance(event, dict) else None

        if event_type in {"on_chain_start", "on_graph_start"}:
            print(f"🔁 Chain start: {name}")
        elif event_type in {"on_chain_end", "on_graph_end"}:
            print("🔚 Chain end")
            output = data.get("output") if isinstance(data, dict) else None
            if isinstance(output, dict):
                messages = (
                    output.get("messages")
                    or output.get("return_value")
                    or output.get("result")
                )
                if isinstance(messages, list) and messages:
                    candidate = messages[-1]
                    if isinstance(candidate, BaseMessage):
                        final_ai_message = candidate
        elif event_type == "on_tool_start":
            inputs = None
            if isinstance(data, dict):
                inputs = data.get("input") or data.get("inputs")
            print(f"⚙️  Tool start: {name or 'unknown'} | inputs={inputs}")
        elif event_type == "on_tool_end":
            output = data.get("output") if isinstance(data, dict) else None
            print(f"✅ Tool end: {name or 'unknown'} | output={output}")
        elif event_type in {"on_chat_model_stream", "on_llm_stream"}:
            text = None
            if isinstance(data, dict):
                chunk = data.get("chunk")
                if isinstance(chunk, BaseMessage):
                    text = chunk.content
                elif isinstance(chunk, dict):
                    text = chunk.get("content") or chunk.get("text")
                elif hasattr(chunk, "content"):
                    text = getattr(chunk, "content", None)
            if text:
                print(text, end="", flush=True)
                saw_text_chunk = True
        elif event_type in {"on_chat_model_end", "on_llm_end"}:
            output = data.get("output") if isinstance(data, dict) else None
            if isinstance(output, dict):
                generations = output.get("generations")
                if generations and generations[0]:
                    final_chunk = generations[0][0]
                    message = getattr(final_chunk, "message", None)
                    if isinstance(message, BaseMessage):
                        final_ai_message = message

    print()
    if final_ai_message is not None and not saw_text_chunk:
        final_ai_message.pretty_print()
    return final_ai_message


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def resolve_hitl_interrupts(
    agent: Any,
    result: dict,
    config: dict[str, Any],
    runtime_context: RuntimeContext,
    *,
    auto_approve: bool,
) -> dict:
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"][-1].value
        action_requests = _get_attr(interrupt, "action_requests", [])
        if not isinstance(action_requests, (list, tuple)):
            action_requests = [action_requests]

        decisions = []
        tool_calls = _get_attr(interrupt, "tool_calls", [])
        if isinstance(tool_calls, dict):
            tool_calls = [tool_calls]

        for idx, request in enumerate(action_requests):
            description = _get_attr(request, "description", "Approval required")
            kwargs = _get_attr(request, "kwargs", {}) or _get_attr(request, "args", {}) or {}
            query = kwargs.get("query") or kwargs.get("sql") or kwargs.get("tool_input")
            allowed = _get_attr(request, "allowed_decisions", None)
            call_id = (
                _get_attr(request, "tool_call_id")
                or _get_attr(request, "id")
                or _get_attr(_get_attr(request, "tool_call", {}), "id")
                or _get_attr(_get_attr(request, "tool_call", {}), "tool_call_id")
            )
            if call_id is None and 0 <= idx < len(tool_calls):
                call_id = (
                    _get_attr(tool_calls[idx], "id")
                    or _get_attr(tool_calls[idx], "tool_call_id")
                )
            if call_id is None:
                for tc in tool_calls:
                    if (
                        _get_attr(tc, "name") == _get_attr(request, "name")
                        and _get_attr(tc, "args") == kwargs
                    ):
                        call_id = _get_attr(tc, "id") or _get_attr(tc, "tool_call_id")
                        if call_id:
                            break
            if call_id is None:
                print("[HITL] Warning: tool_call_id missing; raw request:")
                print(request)
            allowed_types: list[str] = []
            if allowed:
                for entry in allowed:
                    if isinstance(entry, dict):
                        allowed_types.append(entry.get("type"))
                    else:
                        allowed_types.append(getattr(entry, "type", entry))
            if not allowed_types:
                allowed_types = ["approve", "reject"]

            if auto_approve and isinstance(query, str) and not is_query_risky(query):
                print("[HITL] Auto-approving SQL query:")
                print(query)
                decision: dict[str, Any] = {"type": "approve"}
                if call_id:
                    decision["tool_call_id"] = call_id
                decisions.append(decision)
                continue

            print("-" * 60)
            print(f"[HITL] Approval required: {description}")
            if isinstance(query, str):
                print(f"[HITL] Proposed query:\n{query}")
            print(f"[HITL] Allowed decisions: {', '.join(allowed_types)}")

            decision: Optional[dict[str, Any]] = None
            while decision is None:
                choice = input("[HITL] Enter decision ([a]pprove/[r]eject): ").strip().lower()
                if choice in {"a", "approve"} and "approve" in allowed_types:
                    decision = {"type": "approve"}
                elif choice in {"r", "reject"} and "reject" in allowed_types:
                    reason = input("[HITL] Optional rejection reason: ").strip()
                    decision = {"type": "reject", "message": reason or None}
                else:
                    print("[HITL] Invalid choice. Please enter 'a' to approve or 'r' to reject.")

                if decision and call_id:
                    decision["tool_call_id"] = call_id

            decisions.append(decision)

        result = agent.invoke(
            Command(resume={"decisions": decisions}),
            config=config,
            context=runtime_context,
        )

    return result


def load_checkpointer(args: argparse.Namespace) -> Optional[Any]:
    backend = args.memory_backend
    if backend == "memory":
        return None
    if backend == "sqlite":
        if SqliteSaver is None:
            raise RuntimeError(
                "SQLite memory backend requested but langgraph's sqlite extras are not installed. "
                "Install with `pip install 'langgraph[sqlite]'` and retry."
            )
        path = args.memory_path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        context = SqliteSaver.from_conn_string(str(path))
        if hasattr(context, "__enter__"):
            saver = context.__enter__()
            atexit.register(context.__exit__, None, None, None)
            return saver
        return context
    if backend == "redis":
        if RedisSaver is None:
            raise RuntimeError(
                "Redis memory backend requested but redis extras are not installed. "
                "Install with `pip install 'langgraph[redis]' redis` and retry."
            )
        context = RedisSaver.from_conn_string(args.redis_url)
        if hasattr(context, "__enter__"):
            saver = context.__enter__()
            atexit.register(context.__exit__, None, None, None)
            return saver
        return context
    raise ValueError(f"Unsupported memory backend: {backend}")


def parse_cli_args() -> argparse.Namespace:
    """Collect command-line arguments for the interactive agent."""

    parser = argparse.ArgumentParser(description="Run the SQL agent from the terminal.")
    parser.add_argument(
        "--thread-id",
        default="demo-thread-1",
        help="Identifier used for checkpointed memory (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model identifier passed to init_chat_model (default: %(default)s)",
    )
    parser.add_argument(
        "--hitl",
        action="store_true",
        help="Enable human-in-the-loop approvals for SQL tool calls.",
    )
    parser.add_argument(
        "--hitl-auto-approve",
        action="store_true",
        help="Automatically approve low-risk SQL queries when HITL is enabled.",
    )
    parser.add_argument(
        "--db-path",
        type=pathlib.Path,
        default=DEFAULT_DB_PATH,
        help="Path to the Chinook SQLite database (default: %(default)s)",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=5,
        help="Maximum rows permitted in SQL responses (default: %(default)s)",
    )
    parser.add_argument(
        "--enable-mcp-time",
        action="store_true",
        help="Enable the sample MCP time server tools (connects via npx).",
    )
    parser.add_argument(
        "--mcp-config",
        help="Path or JSON string describing MCP servers (overrides --enable-mcp-time).",
    )
    parser.add_argument(
        "--log-path",
        type=pathlib.Path,
        help="Optional path to a JSONL file for recording conversation turns.",
    )
    parser.add_argument(
        "--example-env",
        default="example.env",
        help="Path to the example env file used for validation (default: %(default)s)",
    )
    parser.add_argument(
        "--memory-backend",
        choices=["memory", "sqlite", "redis"],
        default="memory",
        help="Checkpoint backend: in-memory (default), sqlite, or redis.",
    )
    parser.add_argument(
        "--memory-path",
        type=pathlib.Path,
        default=pathlib.Path("sql_agent_memory.db"),
        help="SQLite file for --memory-backend=sqlite (default: %(default)s)",
    )
    parser.add_argument(
        "--redis-url",
        default="redis://localhost:6379/0",
        help="Redis URL when --memory-backend=redis (default: %(default)s)",
    )
    parser.add_argument(
        "--reset-memory",
        action="store_true",
        help="Clear existing persisted memory before starting the session.",
    )
    parser.add_argument(
        "--structured-output",
        choices=["none", "invoice_summary"],
        default="none",
        help="Emit structured JSON for certain tasks (default: none).",
    )
    parser.add_argument(
        "--event-stream",
        action="store_true",
        help="Stream detailed LangGraph events (tool starts, token chunks).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and wait for the final answer only.",
    )
    return parser.parse_args()


def get_mcp_tools(args: argparse.Namespace) -> Optional[list[Any]]:
    if args.mcp_config:
        try:
            config = load_mcp_config(args.mcp_config)
            return load_mcp_tools(config)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"Warning: failed to load MCP tools: {exc}")
    elif args.enable_mcp_time:
        try:
            return load_mcp_tools(DEFAULT_TIME_MCP)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"Warning: failed to load MCP time tools: {exc}")
    return None


def get_structured_config(name: str) -> Tuple[Optional[Type[BaseModel]], Optional[str], Optional[str]]:
    if name == "none":
        return None, None, None
    config = STRUCTURED_PROMPTS.get(name)
    if config is None:
        raise ValueError(f"Unsupported structured output mode: {name}")
    return config


def _display_structured_output(model: Type[BaseModel], message: BaseMessage) -> None:
    content = (
        message.content if isinstance(message.content, str) else json.dumps(message.content)
    )
    try:
        parsed = model.model_validate_json(content)
    except ValidationError as exc:
        print("Structured output validation failed:")
        print(exc)
        return

    print(f"Structured ({model.__name__}):")
    print(parsed.model_dump_json(indent=2))

    if isinstance(parsed, InvoiceSummary):
        summary_bits = []
        if parsed.customer_name:
            summary_bits.append(f"Customer: {parsed.customer_name}")
        if parsed.invoice_total is not None:
            currency = parsed.currency or ""
            summary_bits.append(f"Total: {parsed.invoice_total} {currency}".strip())
        if parsed.notes:
            summary_bits.append(f"Notes: {parsed.notes}")
        if summary_bits:
            print("Human-readable summary:")
            for bit in summary_bits:
                print(f"- {bit}")


if __name__ == "__main__":
    args = parse_cli_args()

    if args.event_stream and args.no_stream:
        raise SystemExit("--event-stream and --no-stream are mutually exclusive")

    structured_model, structured_instructions, structured_suffix = get_structured_config(
        args.structured_output
    )

    validate_environment(args.example_env)
    mcp_tools: Optional[list[Any]] = None
    if args.mcp_config:
        try:
            config = load_mcp_config(args.mcp_config)
            mcp_tools = load_mcp_tools(config)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"Warning: failed to load MCP tools: {exc}")
    elif args.enable_mcp_time:
        try:
            mcp_tools = load_mcp_tools(DEFAULT_TIME_MCP)
        except Exception as exc:  # pragma: no cover - optional feature
            print(f"Warning: failed to load MCP time tools: {exc}")

    if mcp_tools:
        tool_names = ", ".join(sorted(tool.name for tool in mcp_tools if getattr(tool, "name", None)))
        print(f"Loaded MCP tools: {tool_names}")

    checkpointer = load_checkpointer(args)
    if args.reset_memory:
        backend = args.memory_backend
        try:
            if backend == "sqlite":
                if SqliteSaver is None:
                    raise RuntimeError("SQLite backend unavailable; cannot reset.")
                db_path = args.memory_path.expanduser().resolve()
                if db_path.exists():
                    os.remove(db_path)
                    print("SQLite memory store deleted.")
            elif backend == "redis":
                if RedisSaver is None:
                    raise RuntimeError("Redis backend unavailable; cannot reset.")
                context = RedisSaver.from_conn_string(args.redis_url)
                if hasattr(context, "__enter__"):
                    saver = context.__enter__()
                else:
                    saver = context
                try:
                    saver.clear()
                finally:
                    if hasattr(context, "__exit__"):
                        context.__exit__(None, None, None)
                print("Redis memory store cleared.")
            else:
                print("In-memory backend selected; nothing to reset.")
        except Exception as exc:  # pragma: no cover - best effort cleanup
            print(f"Warning: failed to reset memory store: {exc}")
        else:
            if backend != "memory":
                checkpointer = load_checkpointer(args)
            print(
                "Memory store reset. Re-establish context (e.g. restate the scenario) on the first turn."
            )

    agent, runtime_context = build_agent_and_context(
        model_name=args.model,
        db_path=args.db_path,
        max_rows=args.max_rows,
        mcp_tools=mcp_tools,
        checkpointer=checkpointer,
        structured_instructions=structured_instructions,
        hitl_enabled=args.hitl,
    )

    logger = ConversationLogger(args.log_path) if args.log_path else None
    run_cli(
        agent,
        runtime_context,
        thread_id=args.thread_id,
        logger=logger,
        event_stream=args.event_stream,
        disable_stream=args.no_stream,
        structured_model=structured_model,
        structured_suffix=structured_suffix,
        hitl_enabled=args.hitl,
        hitl_auto_approve=args.hitl_auto_approve,
    )