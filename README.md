# Data Scientist AI Agent

A LangGraph-based SQL agent that provides conversational access to SQLite databases with memory, human-in-the-loop approvals, and MCP extensibility.

## Capabilities

- Conversational SQL agent built on LangGraph and LangChain
- Memory backends for ephemeral or persistent state (in-memory, SQLite, Redis)
- Guardrails for read-only SQL execution, rate limiting, and optional approvals
- Structured JSON output when schemas are registered
- Tooling extensibility through Model Context Protocol (MCP) servers and custom Python tools

## Quick Start

```bash
python -m src.data_scientist_ai_agent
```

- Use `./run.sh` or `uv run python -m src.data_scientist_ai_agent` if you prefer wrapper scripts.
- See `USAGE.md` for additional launch options, persona settings, and flag descriptions.

### Example conversation

```
You: How many customers do we have?
Agent: We have 59 customers in the database.

You: This is Julia Barnett
Agent: Hello Julia! How can I help you today?

You: What's my total spending?
Agent: Your total spending is $43.86.
```

## Setup Snapshot

1. Install Python 3.11–3.13 and [`uv`](https://docs.astral.sh/uv/) (or `pip`).
2. Clone the repository and sync dependencies:
   ```bash
   git clone https://github.com/msharan/data_scientist_ai_agent.git
   cd data_scientist_ai_agent
   uv sync
   ```
3. Copy `.env` and add API keys:
   ```bash
   cp example.env .env
   ```
   Add `OPENAI_API_KEY` (required) and optional LangSmith or alternative model keys.
4. For MCP notebook examples, ensure Node.js and `npx` are installed.

Detailed environment, notebook, and LangSmith instructions live in `USAGE.md`.

## Architecture & Extensibility

- High-level system design and component descriptions are in `ARCHITECTURE.md`.
- That document also covers safety controls, personalization, and how to register custom tools or MCP integrations.

## Security Highlights

- Read-only SQL validation with enforced LIMIT clauses
- Global rate limiting (default 8 queries / 10 seconds)
- Optional human-in-the-loop approvals for risky queries

Full security guidance appears in `ARCHITECTURE.md`.

## Additional Resources

- `USAGE.md` – command-line flags, run scripts, and walkthroughs
- `ARCHITECTURE.md` – system internals and extension patterns
- `USAGE.md` also links to notebook workflows and LangSmith setup
- `CONTRIBUTING.md` (if present) – contribution process

External references: [LangChain](https://python.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Model Context Protocol](https://modelcontextprotocol.io/), [LangSmith](https://docs.smith.langchain.com/)
