# alzabo

A CLI tool and MCP server for searching Claude Code and Codex JSONL transcripts with hybrid search (BM25 + vector).

## Install

```bash
uv tool install alzabo
```

## CLI Usage

```bash
alzabo search "oauth token refresh"         # search turns (text output)
alzabo search "vector db" --sessions        # group results by session
alzabo search "terraform" --format json     # JSON output
alzabo list --format jsonl                  # list all sessions as JSONL
alzabo read <session-id> --turn 0           # read a specific turn
alzabo status                               # index stats
alzabo serve                                # start MCP server
alzabo extract --stats                      # tool call extraction stats
```

### Subcommands

| Subcommand | Description |
|---|---|
| `search QUERY` | Search turns. `--sessions` groups by session. `--mode {hybrid,bm25,vector}`, `--context-window N` |
| `list` | List conversations. `--source`, `--project`, `--start-date`, `--end-date`, `--offset`, `--limit` |
| `read SESSION_ID` | Read a conversation. `--turn N` for a single turn. `--compact`, `--include-records` |
| `status` | Show index stats: session counts, turn count, embeddings state, last reindex time |
| `serve` | Start the MCP server. `--watch`/`--no-watch`, `--debounce-seconds` |
| `extract` | Extract structured tool call records. `--stats`, `--tool`, `--category`, `--errors-only` |

### Global flags

These can be placed after the subcommand:

- `--format {text,json,jsonl}`: output format (default: `text`)
- `--no-cache`: skip disk cache, always reindex from source files
- `--transcripts-dir`: Claude transcript directory (default: `~/.claude/projects`)
- `--codex-dir`: Codex session directory (default: `~/.codex/sessions`)
- `--quiet`: suppress progress logs for cleaner LLM/agent consumption

### Disk cache

On first run, alzabo indexes all transcripts and saves a cache to `~/.cache/alzabo/`. Subsequent runs load from cache if no source files have changed, making startup near-instant. Use `--no-cache` to force a fresh reindex.

## MCP Server

### Claude Code config

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "alzabo": {
      "command": "uvx",
      "args": ["alzabo", "serve"]
    }
  }
}
```

To test the local checkout instead of a published tool, use an absolute project path:

```json
{
  "mcpServers": {
    "alzabo": {
      "command": "uvx",
      "args": ["--directory", "/absolute/path/to/alzabo", "--from", ".", "alzabo", "serve"]
    }
  }
}
```

You can also bypass `uvx` if you prefer:

```json
{
  "mcpServers": {
    "alzabo": {
      "command": "uv",
      "args": ["run", "--project", "/absolute/path/to/alzabo", "alzabo", "serve"]
    }
  }
}
```

After updating your MCP config, restart the MCP host and then verify:

```bash
uvx --directory /absolute/path/to/alzabo --from . alzabo --help
```

### MCP tools

All tools return plain text.

| Tool | Description |
|------|-------------|
| `search_conversations(query, ...)` | Search individual turns with optional inline context turns |
| `search_sessions(query, ...)` | Search grouped by session |
| `list_conversations(...)` | Browse indexed sessions with compact metadata |
| `read_turn(session_id, turn_number, ...)` | Read a single turn, including extracted signals |
| `read_conversation(session_id, ...)` | Read a conversation sequentially |
| `index_status()` | Report index counts, directories, watch state, and last reindex time |

## What gets indexed

- User prompts and assistant text
- Claude `tool_use` inputs and `tool_result` payloads
- Codex function-call arguments and `function_call_output` payloads
- Extracted signals per turn: tools, file paths, shell commands, and error snippets

## Search modes

- `hybrid`: Reciprocal Rank Fusion of BM25 and vector search (default)
- `bm25`: keyword-only search
- `vector`: semantic search using `model2vec`

## Data sources

| Source | Directory | Format |
|--------|-----------|--------|
| Claude Code | `~/.claude/projects/**/*.jsonl` | `user`, `assistant`, `summary` |
| Codex | `~/.codex/sessions/**/*.jsonl` | `session_meta`, `event_msg`, `response_item` |

Codex session IDs are prefixed with `codex:` to avoid collisions with Claude session IDs.

## Entry points

| Command | Target | Purpose |
|---|---|---|
| `alzabo` | `alzabo.main_cli:main` | Unified CLI with subcommands |
| `alzabo-serve` | `alzabo.cli:main` | Backward-compat MCP server (same as `alzabo serve`) |
| `alzabo-extract` | `alzabo.extract_cli:main` | Standalone extraction tool |

## Development

```bash
uv run pytest
```

Project layout:

- `src/alzabo/main_cli.py`: unified CLI entry point with subparsers
- `src/alzabo/cli.py`: MCP server startup, watcher wiring
- `src/alzabo/server.py`: MCP tool definitions
- `src/alzabo/index.py`: indexing, ranking, state management
- `src/alzabo/parsers.py`: transcript parsing and signal extraction
- `src/alzabo/render.py`: plain-text output formatting
- `src/alzabo/output.py`: output format dispatch (text/json/jsonl)
- `src/alzabo/cache.py`: disk cache for fast startup
- `src/alzabo/extract.py`: structured tool call extraction
- `src/alzabo/extract_cli.py`: standalone extract CLI
