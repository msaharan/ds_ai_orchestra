# Running the Data Science AI Agent

Start with the setup snapshot in `README.md` to install dependencies and configure environment variables. This document focuses on day-to-day invocation, supported flags, and troubleshooting tips.

## Quick Start

### Using the shell script (recommended)

```bash
# Run with default settings
./run.sh

# Run with custom options
./run.sh --model "openai:gpt-4" --persona friendly
```

### Using uv (recommended if you have uv installed)

```bash
# Run with default settings
uv run python -m src.data_scientist_ai_agent

# Run with custom options
uv run python -m src.data_scientist_ai_agent --model "openai:gpt-4" --persona friendly
```

### Using Python directly

```bash
# Activate virtual environment first
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the agent
python -m src.data_scientist_ai_agent

# Or with custom options
python -m src.data_scientist_ai_agent --model "openai:gpt-4" --persona friendly
```

## Command Line Options

### Basic Options

- `--thread-id TEXT`: Identifier for checkpointed memory (default: `demo-thread-1`)
- `--model TEXT`: Model identifier (default: `openai:gpt-3.5-turbo`)
- `--db-path PATH`: Path to Chinook SQLite database (default: `./Chinook.db`)
- `--max-rows INT`: Maximum rows in SQL responses (default: `5`)

### Persona and Localization

- `--persona {default,friendly,executive,analytical}`: Adjust assistant's tone/personality
- `--locale TEXT`: Preferred locale/language code (default: `en-US`)

### Human-in-the-Loop

- `--hitl`: Enable human-in-the-loop approvals for SQL tool calls
- `--hitl-auto-approve`: Automatically approve low-risk SQL queries when HITL is enabled

### Memory/Checkpointing

- `--memory-backend {memory,sqlite,redis}`: Checkpoint backend (default: `memory`)
- `--memory-path PATH`: SQLite file for `--memory-backend=sqlite` (default: `sql_agent_memory.db`)
- `--redis-url TEXT`: Redis URL when `--memory-backend=redis` (default: `redis://localhost:6379/0`)
- `--reset-memory`: Clear existing persisted memory before starting

### MCP Tools

- `--enable-mcp-time`: Enable the sample MCP time server tools
- `--mcp-config PATH`: Path or JSON string describing MCP servers

### Output Options

- `--log-path PATH`: Optional path to JSONL file for recording conversations
- `--event-stream`: Stream detailed LangGraph events
- `--no-stream`: Disable streaming and wait for final answer only
- `--structured-output {none,invoice_summary}`: Emit structured JSON for certain tasks

### Other Options

- `--example-env PATH`: Path to example env file for validation (default: `example.env`)

## Examples

### Basic query
```bash
./run.sh
# Then type: "Show me the top 5 artists by number of tracks"
```

### With GPT-4 and friendly persona
```bash
./run.sh --model "openai:gpt-4" --persona friendly
```

### With persistent SQLite memory
```bash
./run.sh --memory-backend sqlite --memory-path my_agent_memory.db
```

### With human-in-the-loop approval
```bash
./run.sh --hitl
```

### With MCP time tools
```bash
./run.sh --enable-mcp-time
```

### Reset memory and start fresh
```bash
./run.sh --memory-backend sqlite --reset-memory
```

### Log conversations to file
```bash
./run.sh --log-path conversations.jsonl
```

## Troubleshooting

### Module not found error
If you get import errors, make sure you're running from the project root directory and using the `-m` flag to run as a module.

### Database not found
Ensure `Chinook.db` exists in the project root, or specify a custom path with `--db-path`.

### Environment variables not loaded
Make sure your `.env` file is in the project root directory.

