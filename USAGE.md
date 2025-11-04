# Running the Data Science AI Agent

Start with the setup snapshot in `README.md` to install dependencies and configure environment variables. This document focuses on day-to-day invocation, supported flags, and troubleshooting tips.

## Quick Start

### Using the shell script (recommended)

```bash
# Run with default settings
./run.sh

# Run with custom options
./run.sh --model "openai:gpt-4"
```

### Using uv (recommended if you have uv installed)

```bash
# Run with default settings
uv run python -m src.sql_agent

# Run with custom options
uv run python -m src.sql_agent --model "openai:gpt-4"
```

### Using Python directly

```bash
# Activate virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the agent
python -m src.sql_agent

# Or with custom options
python -m src.sql_agent --model "openai:gpt-4"
```

## Command Line Options

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

