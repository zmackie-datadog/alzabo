# alzabo

An MCP server that indexes Claude Code and Codex JSONL transcripts and exposes compact, plain-text tools for search and read workflows.

## Install

```bash
uvx alzabo
```

Or install as a tool:

```bash
uv tool install alzabo
```

## Claude Code MCP config

Add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "alzabo": {
      "command": "uvx",
      "args": ["alzabo"]
    }
  }
}
```

Optional flags:

- `--transcripts-dir`: root folder to scan for Claude `*.jsonl` transcripts. Default: `~/.claude/projects`
- `--codex-dir`: root folder to scan for Codex `*.jsonl` sessions. Default: `~/.codex/sessions`
- `--watch` / `--no-watch`: enable or disable filesystem watching for automatic reindex
- `--debounce-seconds`: debounce window before auto-reindexing after file changes

## MCP tools

All tools return plain text.

| Tool | Description |
|------|-------------|
| `search_conversations(query, limit, session_id, source, project, start_date, end_date, mode, context_window)` | Search individual turns with optional inline context turns |
| `search_sessions(query, limit, source, project, start_date, end_date, mode)` | Search grouped by session |
| `list_conversations(source, project, start_date, end_date, limit, offset)` | Browse indexed sessions with compact metadata |
| `read_turn(session_id, turn_number, include_records, include_content)` | Read a single turn, including extracted signals |
| `read_conversation(session_id, offset, limit, include_records, include_content, compact)` | Read a conversation sequentially |
| `index_status()` | Report index counts, directories, watch state, and last reindex time |

## What gets indexed

- user prompts and assistant text
- Claude `tool_use` inputs and `tool_result` payloads
- Codex function-call arguments and `function_call_output` payloads
- extracted signals per turn: tools, file paths, shell commands, and error snippets

This makes exact command, file, and failure lookups searchable instead of relying only on assistant prose.

## Search modes

- `hybrid`: Reciprocal Rank Fusion of BM25 and vector search
- `bm25`: keyword-only search
- `vector`: semantic search using `model2vec`

## Data sources

| Source | Directory | Format |
|--------|-----------|--------|
| Claude Code | `~/.claude/projects/**/*.jsonl` | `user`, `assistant`, `summary` |
| Codex | `~/.codex/sessions/**/*.jsonl` | `session_meta`, `event_msg`, `response_item` |

Codex session IDs are prefixed with `codex:` to avoid collisions with Claude session IDs.

## Development

```bash
uv run pytest
```

Project layout:

- `src/alzabo/cli.py`: startup, args, watcher wiring
- `src/alzabo/server.py`: MCP tool definitions
- `src/alzabo/index.py`: indexing, ranking, state management
- `src/alzabo/parsers.py`: transcript parsing and signal extraction
- `src/alzabo/render.py`: plain-text output formatting
- `tests/`: test suite

## Deferred Work

Deferred standalone extraction and validation ideas live in `ROADMAP.md`.
