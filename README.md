# Data Scientist AI Agents Orchestra

Building an orchestra of AI agents to automate data science. Current functionality: conversational SQL assistant built with LangGraph that delivers concise, decision-ready answers backed by the Chinook sample database. The agent enforces read-only access and can be extended with Model Context Protocol (MCP) tools.

## Highlights

- LangGraph-powered agent with step-by-step SQL planning and tool arbitration
- Memory backends for ephemeral sessions (`memory`) or persistence (`sqlite`, `redis`)
- Built-in guardrails: read-only SQL enforcement via input validation
- Structured JSON output via `--structured-output` presets
- MCP integration for augmenting the SQL toolkit with external services
- Conversation logging and event streaming for observability

## Requirements

- Python 3.11–3.13
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- OpenAI-compatible API key exposed as `OPENAI_API_KEY` in `.env`
- Optional: Redis (for `--memory-backend redis`), Node.js/npm (`npx`) for the sample MCP time server, `langchain-mcp-adapters` for MCP integration, LangSmith credentials for tracing

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

## Run the Agent

```bash
# Quick start (uses defaults from .env)
python -m src.sql_agent

# Unified launcher (select SQL or data science agent)
python -m src.main --agent sql
python -m src.main --agent data_science

# Wrapper options
./run.sh
uv run python -m src.sql_agent
```

Helpful flags:

- `--model openai:gpt-4o-mini` – pick any model supported by `langchain.chat_models.init_chat_model`
- `--structured-output invoice_summary` – return validated JSON for invoice summaries
- `--memory-backend sqlite --memory-path ./state/agent.db` – persist conversation context between runs
- `--enable-mcp-time` or `--mcp-config path/to/config.json` – attach MCP tool servers
- `--log-path run/conversation.jsonl` – capture every user and agent message for auditing
- `--event-stream` / `--no-stream` – switch between detailed LangGraph streaming and single-response mode (choose at most one)

See `USAGE.md` for the full CLI reference, troubleshooting, and workflow examples.

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

- `src/sql_agent.py` – CLI entry point for the SQL agent (LangGraph wiring, memory, structured output handling)
- `src/sql_agent_tools.py` – SQL tool implementations and runtime context
- `src/sql_agent_mcp.py` – MCP configuration loader and adapters

Refer to `ARCHITECTURE.md` for a deeper system walkthrough and safety model.

## Related Documentation

- `ARCHITECTURE.md` – runtime topology, safety layers, and extension points
- `USAGE.md` – CLI flags, launch recipes, and troubleshooting
- `local/plan_focus_data_science.md` – example analysis prompts and workflows

External references: [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Model Context Protocol](https://modelcontextprotocol.io/), [LangSmith](https://docs.smith.langchain.com/)
