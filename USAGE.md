# Running the Agents

Start with the setup snapshot in `README.md` to install dependencies and configure environment variables. This document focuses on day-to-day invocation, supported flags, dataset workflows, and troubleshooting tips for both the SQL and Data Science agents.

## Quick Start

### Using the shell script (SQL agent, default)

```bash
# Run with default settings
./run.sh

# Run with custom options
./run.sh --model "openai:gpt-4"
```

### Launch through the unified CLI

```bash
# SQL agent (default)
python -m src.main

# Explicitly select SQL agent
python -m src.main --agent sql

# Data Science agent
python -m src.main --agent data_science

# Forward any additional flags after the agent selection
python -m src.main --agent data_science --model "openai:gpt-4o-mini"
```

### Using uv (recommended if you have uv installed)

```bash
# Run with default settings
uv run python -m src.main

# Run with custom options
uv run python -m src.main --agent data_science --model "openai:gpt-4o-mini"
```

### Using Python directly

```bash
# Activate virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the agent
python -m src.main --agent data_science

# Or with custom options
python -m src.main --agent sql --model "openai:gpt-4"
```

## Command Line Options (SQL Agent)

### Basic Options

- `--thread-id TEXT`: Identifier for checkpointed memory (default: `demo-thread-1`)
- `--model TEXT`: Model identifier passed to LangChain (default: `openai:gpt-3.5-turbo`)
- `--db-path PATH`: Path to the Chinook SQLite database (default: `data/Chinook.db`)
- `--example-env PATH`: Alternate `.env` template checked during startup (default: `example.env`)

### Memory & State

- `--memory-backend {memory,sqlite,redis}`: Choose checkpoint storage (default: `memory`)
- `--memory-path PATH`: SQLite file for `--memory-backend=sqlite` (default: `sql_agent_memory.db`)
- `--redis-url TEXT`: Redis URL when `--memory-backend=redis` (default: `redis://localhost:6379/0`)
- `--reset-memory`: Clears the selected backend before launching (best effort for SQLite/Redis)

### MCP Tools

- `--enable-mcp-time`: Attach the sample MCP time server (requires `npx`)
- `--mcp-config PATH_OR_JSON`: Load a custom MCP configuration; takes precedence over `--enable-mcp-time`

### Output & Observability

- `--structured-output {none,invoice_summary}`: Validate responses against a Pydantic schema
- `--log-path PATH`: Write every conversation turn to a JSONL file
- `--event-stream`: Stream LangGraph events (tool invocations, token batches)
- `--no-stream`: Disable incremental token streaming and print only the final reply
  > Tip: `--event-stream` and `--no-stream` are mutually exclusive; omit both for default token streaming.

### Language

Responses are produced in English; the system prompt enforces concise answers with key takeaways.

## Data Science Agent Options

```bash
python -m src.main --agent data_science [--model MODEL_NAME] [--catalog PATH] [--thread-id THREAD]
```

- `--model`: LangChain chat model identifier (default: `openai:gpt-4o-mini`).
- `--catalog`: Override the dataset manifest path (defaults to `data/catalog.json`).
- `--thread-id`: Checkpoint identifier used by LangGraph memory (default: `ds-thread-1`).
- `--event-stream` / `--no-stream`: Inherit the same semantics as the SQL agent.
- `--mcp-config`: JSON string or file path describing MCP dataset services (requires `langchain-mcp-adapters`).

### Dataset catalog workflow

1. Register datasets in `data/catalog.json` (see `src/datasets/catalog.py` for schema). Each dataset entry must include `uri` or `path`, optional `format`, `read_options`, and `metadata`.
2. Discover datasets via CLI:

```bash
python -m src.datasets.catalog list --verbose
python -m src.datasets.catalog describe bookingcom_train
python -m src.datasets.catalog load bookingcom_train --limit 5
```

> The bundled examples in `data/catalog.json` point at the Booking.com Multi-Destination Trips dataset hosted on GitHub (`https://github.com/bookingcom/ml-dataset-mdt`). Download the CSV files from that repository and place them under `data/bookingcom/` before running the agent.

3. At runtime, the agent exposes tools:
   - `list_datasets` to enumerate catalog entries.
   - `preview_dataset` to sample rows (capped at 100).
   - `profile_dataset_tool` to generate cached column summaries.
   - `merge_datasets` to join catalog datasets across formats.

### Remote dataset integration (Model Context Protocol)

```bash
python -m src.main --agent data_science --mcp-config path/to/config.json
```

- `--mcp-config` accepts either a JSON file or inline JSON string defining MCP servers. Each entry specifies a transport (e.g., `stdio`) and the command used to launch the remote service.
- MCP dataset tools are merged into the data science toolkit; their names are surfaced in the system prompt so the agent understands which remote connectors are available.
- The same configuration can be supplied to the SQL agent for hybrid workloads.
- Ensure `langchain-mcp-adapters` is installed and any external binaries referenced by the MCP config are available on `PATH`.

### Logging and telemetry

The data science tooling records structured events to `logs/ds_agent.log`. Entries include dataset load duration, row/column counts, and tool usage metadata. Inspect the log when debugging agent behaviour or to measure profiling latency.

### Sample scripted tests

Run the focused pytest suites to validate dataset tooling:

```bash
pytest tests/test_merge_tool.py
pytest tests/test_logging_usage.py
```

Both tests rely on temporary catalogs and should pass without external datasets.

## Examples

### Basic query
```bash
./run.sh
# Then type: "Show me the top 5 artists by number of tracks"
```

### With GPT-4
```bash
./run.sh --model "openai:gpt-4"
```

### Launch the data science agent with cached profiling
```bash
python -m src.main --agent data_science --model "openai:gpt-4o-mini"
# inside the session, request "profile bookingcom_train"
```

### With persistent SQLite memory
```bash
./run.sh --memory-backend sqlite --memory-path my_agent_memory.db
```

### With MCP time tools
```bash
./run.sh --enable-mcp-time
```

### Reset memory and start fresh
```bash
./run.sh --memory-backend sqlite --reset-memory
```

### Emit structured JSON
```bash
./run.sh --structured-output invoice_summary
```

### Log conversations to file
```bash
./run.sh --log-path conversations.jsonl
```

## Troubleshooting

### Module not found error
If you get import errors, make sure you're running from the project root directory and using the `-m` flag to run as a module.

### Database not found
The CLI downloads `Chinook.db` to the `data/` directory on first run. If that fails, ensure outgoing network access and rerun, or provide a local copy with `--db-path`.

### Environment variables not loaded
The launcher requires `.env` to exist. Copy `example.env` to `.env` and populate `OPENAI_API_KEY` (plus any optional LangSmith keys). Use `--example-env` to validate a different template.

### Optional dependencies missing
- SQLite checkpoints require `pip install 'langgraph[sqlite]'`
- Redis checkpoints require `pip install 'langgraph[redis]' redis`
- MCP integration requires `pip install langchain-mcp-adapters`
- The synchronous MCP loader now expects to run outside any active asyncio event loop. If you are extending the project from async code, import and await `src.sql_agent_mcp.aload_mcp_tools` instead of calling the blocking helper.

### Data science agent cannot find datasets
- Confirm `data/catalog.json` exists and is readable.
- Use `python -m src.datasets.catalog list` to check registration and file paths.
- Relative `uri` entries are resolved against the project root; absolute paths are supported.

### Dataset profiling takes too long
- Profiles are cached under `data/profiles/<dataset>.json`. Delete files to force refresh.
- Large parquet files may trigger a fallback load strategy; monitor `logs/ds_agent.log` for `dataset_load_retry` messages and adjust catalog `read_options` (e.g., columns, filters).
- If MCP-backed datasets appear slow to respond, confirm the remote server is reachable and that the MCP configuration is correct.

