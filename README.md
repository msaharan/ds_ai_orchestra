# Data Scientist AI Agents Orchestra

Building an orchestra of AI agents to automate data science workflows. The project currently ships two LangGraph-powered assistants:

- **SQL agent**: conversational analyst backed by the Chinook sample database with enforced read-only SQL.
- **Data science agent**: tabular exploration assistant that loads datasets from a manifest, profiles columns, joins sources, and surfaces observations through curated tool calls.

## Highlights

- LangGraph-powered agents with step-by-step planning, tool arbitration, and checkpointed memory
- Dataset catalog + loader abstraction supporting CSV, Parquet, and SQLite sources
- Profiling, preview, and merge tools with structured logging to `logs/ds_agent.log`
- Memory backends for ephemeral sessions (`memory`) or persistence (`sqlite`, `redis`)
- Built-in SQL guardrails (read-only enforcement) and structured JSON output presets
- Optional Model Context Protocol (MCP) integration for remote SQL or dataset services
- Conversation logging and LangGraph event streaming for observability

## Requirements

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- OpenAI-compatible API key exposed as `OPENAI_API_KEY` in `.env`
- Optional libraries: `pyarrow` for Parquet support (already pinned in requirements), `langchain-mcp-adapters` for MCP integration, LangSmith credentials for tracing, Redis or SQLite extras for memory backends
- Optional services: Redis (for `--memory-backend redis`), Node.js/npm (`npx`) for the sample MCP time server, MCP servers exposing remote datasets

> The entry point refuses to start if `.env` is missing. Copy `example.env` and populate the required keys before launching.

## Installation

```bash
git clone https://github.com/msharan/data_scientist_ai_agent.git
cd data_scientist_ai_agent
uv sync  # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp example.env .env
# edit .env to set OPENAI_API_KEY=<your key>
```

The first run downloads `Chinook.db` automatically to the `data/` directory unless you point `--db-path` elsewhere.

## Run the Agents

```bash
# Unified launcher (defaults to the SQL agent)
python -m src.main

# Explicit agent selection
python -m src.main --agent sql
python -m src.main --agent data_science

# Wrapper scripts
./run.sh                       # SQL agent shortcut
uv run python -m src.main --agent data_science
```

Helpful flags (SQL agent):

- `--model openai:gpt-4o-mini` – choose any chat model supported by `langchain.chat_models.init_chat_model`
- `--structured-output invoice_summary` – return validated JSON for invoice summaries
- `--memory-backend sqlite --memory-path ./state/agent.db` – persist conversation context between runs
- `--enable-mcp-time` or `--mcp-config path/to/config.json` – attach MCP tool servers
- `--log-path run/conversation.jsonl` – capture every user and agent message for auditing
- `--event-stream` / `--no-stream` – toggle LangGraph streaming (mutually exclusive)

Data science agent extras:

- `--catalog data/catalog.json` – override the dataset manifest location
- `--mcp-config path/to/config.json` – merge MCP dataset services into the toolkit
- Toolbelt includes `list_datasets`, `preview_dataset`, `profile_dataset_tool`, `analyze_dataset`, `merge_datasets`, and task template helpers
- Structured telemetry written to `logs/ds_agent.log` for dataset loads and tool usage

See `USAGE.md` for the full CLI reference, workflow walkthroughs, and troubleshooting tips.

### Example Conversation

```
You: How many customers do we have?
Agent: We have 59 customers in the database.

You: This is Julia Barnett
Agent: Hello Julia! How can I help you today?

You: What's my total spending?
Agent: Your total spending is $43.86.
```

## Project Layout

- `src/sql_agent.py` – SQL agent CLI (LangGraph wiring, memory, structured output handling)
- `src/sql_agent_tools.py` – SQL tool implementations and runtime context
- `src/sql_agent_mcp.py` – MCP configuration loader and adapters shared across agents
- `src/ds_agent.py` – Data science agent CLI and MCP hook-up
- `src/ds_agent_tools.py` – Dataset tooling (catalog integration, profiling, merge, logging)
- `src/ds_agent_templates.py` – Task templates guiding exploratory workflows
- `src/datasets/` – Dataset abstractions, loader registry, and manifest CLI
- Sample datasets: the bundled catalog references the Booking.com Multi-Destination Trips dataset. Download the CSV files from [bookingcom/ml-dataset-mdt](https://github.com/bookingcom/ml-dataset-mdt) into `data/bookingcom/` before running the data science agent.
- `src/data_prep/profile.py` – Profiling logic and caching utilities

Refer to `ARCHITECTURE.md` for a deeper system walkthrough and safety model.

## Related Documentation

- `ARCHITECTURE.md` – runtime topology, safety layers, and extension points
- `USAGE.md` – CLI flags, launch recipes, and troubleshooting
- `local/plan_focus_data_science.md` – example analysis prompts and workflows
- `local/implement_ds_agent_plan.md` – roadmap and progress tracker for the data science agent

External references: [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Model Context Protocol](https://modelcontextprotocol.io/), [LangSmith](https://docs.smith.langchain.com/)
