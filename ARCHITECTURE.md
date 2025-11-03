# Architecture Documentation

## System Overview

```mermaid
graph TD
    User --> CLI[CLI interface]
    CLI --> Agent{AI agent (LangGraph)}
    Agent --> Prompt[System prompt & persona]
    Agent --> ToolSelect[Tool selection]
    ToolSelect --> SQLTools[SQL toolkit]
    ToolSelect --> MCPTools[MCP toolkit]
    SQLTools --> RateLimit[Rate limiter]
    MCPTools --> RateLimit
    RateLimit --> Safety[Query validator]
    Safety --> Runtime[(Runtime context: database, max_rows)]
    Runtime --> Database[(SQLite database)]
    Agent --> HITL[Human-in-the-loop middleware]
    HITL --> Decision[Approve or reject query]
    Decision --> Runtime
    Runtime --> Result[Query results]
    Result --> Agent
    Agent --> Memory[(Checkpointer: in-memory, SQLite, Redis)]
    Memory --> Agent
    Agent --> Logger[Conversation logger]
    Agent --> Output[Response to user]
    Database --> Result
```

## Core Components

### Agent layer
- Built with LangGraph and LangChain; each conversation runs inside a graph state machine.
- Maintains thread-scoped state with message history and runtime context.
- Supports middleware hooks for human approval and logging.

### Tooling

#### SQL toolkit (`src/data_scientist_ai_agent_tools.py`)
- `execute_sql`: read-only SELECT execution with automatic LIMIT insertion.
- `list_tables`: enumerates database tables.
- `describe_table`: PRAGMA-based schema inspection.

#### MCP toolkit (`src/data_scientist_ai_agent_mcp.py`)
- Loads Model Context Protocol servers defined in configuration files.
- Wraps async MCP handlers for synchronous use inside the agent.
- Merges MCP tools with the SQL toolkit when the runtime starts.

### Memory and state
- `InMemorySaver`: default, optimized for local development.
- `SqliteSaver`: persists state to `sql_agent_memory.db`.
- `RedisSaver`: recommended for multi-instance deployments.
- Thread IDs allow conversations to resume with prior context; `--reset-memory` clears state.

### Safety controls
- `_safe_sql` enforces SELECT-only execution, single statements, and per-query LIMITs.
- Rate limiting (`_rate_limited`) defaults to eight queries every ten seconds.
- HITL middleware pauses before execution for manual approval or automatic risk checks.
- Access to `sqlite_master`, PRAGMA commands, and unrestricted `SELECT *` queries is blocked.

### Personalization
- Personas are defined through `PERSONA_SNIPPETS` and `PERSONA_SUFFIXES`.
- Locale codes (for example `en-US`) flow into the system prompt to adjust language.
- Customer identity tracking keeps per-thread context for personalization and filtering.

### Structured output
- Pydantic models (for example `InvoiceSummary`) describe JSON responses.
- `STRUCTURED_PROMPTS` maps schema keys to `(model, instructions, suffix)` tuples.
- Passing `--structured-output <key>` prompts the agent to validate responses before returning them.

## Extension Points

- Register additional Python tools and append them to the SQL toolkit.
- Provide an MCP configuration file via `--mcp-config` to attach external services.
- Implement custom middleware (logging, policy enforcement, analytics) and insert it into the LangGraph runtime.
- Extend personas or structured outputs by adjusting dictionaries in `src/data_scientist_ai_agent.py`.

## Security Architecture

### Defense in depth
1. Query validation: regex checks and statement counting block writes and multi-statements.
2. Rate limiting: process-wide guard prevents burst traffic.
3. Human approval: optional manual review or risk-based auto-approval.
4. Database isolation: read-only connections and enforced row limits.

### Threat model
**Protected:** SQL injection through multi-statement payloads, unbounded data exfiltration, schema mutation, accidental `SELECT *` without LIMIT.  
**Not protected:** Slow but legitimate analytical queries, deliberate human overrides, vulnerabilities inside external databases or MCP servers.

## Database Schema

The agent targets the Chinook sample database.

**Catalog:** `Artist`, `Album`, `Track`, `Genre`, `MediaType`.  
**Customers & Sales:** `Customer`, `Employee`, `Invoice`, `InvoiceLine`.  
**Playlists:** `Playlist`, `PlaylistTrack`.

Key relationships mirror the Chinook ERD (for example an `Album` has many `Track` records). Typical analytical queries include tracking top selling tracks or customer lifetime value.

### Observability
- Every message, tool call, and error is recorded as JSON Lines via the conversation logger.
- CLI flags enable three modes: streaming, event streaming (`--event-stream`), and non-streaming (`--no-stream`).
- Optional LangSmith instrumentation provides trace-level visibility.
