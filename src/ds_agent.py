"""LangGraph-powered data science agent CLI."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import pathlib
import sys
import time
from typing import Any, Iterable, Optional

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from .datasets import DatasetCatalog, load_catalog
from .ds_agent_tools import DATA_SCIENCE_TOOLS, DataScienceContext


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_MODEL = "openai:gpt-4o-mini"


def build_system_prompt(catalog: DatasetCatalog) -> str:
    dataset_names = ", ".join(entry.name for entry in catalog.entries())
    return (
        "You are a careful data science analyst.\n"
        "- Always think step-by-step.\n"
        "- Use the provided tools to list datasets, preview samples, profile columns,"
        " and perform lightweight exploratory analysis.\n"
        "- Avoid destructive actions and never write to disk.\n"
        "- Keep row outputs concise (prefer <= 20 rows) unless explicitly instructed.\n"
        f"Known datasets: {dataset_names or 'none registered.'}\n"
    )


def build_agent(
    *,
    model_name: str,
    catalog: DatasetCatalog,
    checkpointer: Optional[Any] = None,
) -> tuple[Any, DataScienceContext]:
    context = DataScienceContext(catalog=catalog)
    agent = create_agent(
        model=init_chat_model(model_name),
        tools=list(DATA_SCIENCE_TOOLS),
        system_prompt=build_system_prompt(catalog),
        context_schema=DataScienceContext,
        checkpointer=checkpointer or InMemorySaver(),
    )
    return agent, context


def run_cli(
    agent: Any,
    runtime_context: DataScienceContext,
    *,
    thread_id: str = "ds-thread-1",
    event_stream: bool = False,
    disable_stream: bool = False,
) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    turn_index = 0

    print("=" * 60)
    print("Data Science Agent - Interactive Mode")
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
            user_message = HumanMessage(content=user_input)

            start = time.perf_counter()
            final_ai_message: Optional[BaseMessage] = None

            print()
            if event_stream:
                final_ai_message = asyncio.run(
                    _event_stream_interaction(
                        agent,
                        user_message,
                        config=config,
                        runtime_context=runtime_context,
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
                    latest_message.pretty_print()

            elapsed = time.perf_counter() - start
            print(f"\n[elapsed: {elapsed:.2f}s]\n")

            if final_ai_message is not None and not disable_stream and not event_stream:
                final_ai_message.pretty_print()

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
    runtime_context: DataScienceContext,
) -> Optional[BaseMessage]:
    final_ai_message: Optional[BaseMessage] = None
    async for event in agent.astream_events(  # type: ignore[attr-defined]
        {"messages": [user_message]},
        config=config,
        context=runtime_context,
    ):
        event_type = event.get("event") if isinstance(event, dict) else None
        data = event.get("data") if isinstance(event, dict) else None
        if event_type in {"on_chat_model_stream", "on_llm_stream"}:
            chunk = data.get("chunk") if isinstance(data, dict) else None
            text = getattr(chunk, "content", None) if chunk is not None else None
            if isinstance(text, str):
                print(text, end="", flush=True)
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
    return final_ai_message


def parse_cli_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the data science agent.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--catalog",
        type=pathlib.Path,
        default=None,
        help="Optional path to dataset catalog (defaults to data/catalog.json)",
    )
    parser.add_argument(
        "--thread-id",
        default="ds-thread-1",
        help="Identifier for checkpointed memory (default: %(default)s)",
    )
    parser.add_argument(
        "--event-stream",
        action="store_true",
        help="Stream detailed LangGraph events (token chunks).",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming and wait for final answer only.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.event_stream and args.no_stream:
        parser.error("--event-stream and --no-stream are mutually exclusive")
    return args


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_cli_args(argv)
    catalog = load_catalog(args.catalog)
    agent, context = build_agent(
        model_name=args.model,
        catalog=catalog,
    )
    run_cli(
        agent,
        context,
        thread_id=args.thread_id,
        event_stream=args.event_stream,
        disable_stream=args.no_stream,
    )


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()

