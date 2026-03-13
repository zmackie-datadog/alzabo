# alzabo

MCP server that indexes Claude Code and Codex JSONL transcripts for search and read workflows.

## Project Structure

- `src/alzabo/cli.py` — startup, args, watcher wiring
- `src/alzabo/server.py` — MCP tool definitions
- `src/alzabo/index.py` — indexing, ranking, state management
- `src/alzabo/parsers.py` — transcript parsing and signal extraction
- `src/alzabo/render.py` — plain-text output formatting
- `tests/` — test suite

## Development

- **Python 3.12+** required
- Use `uv` for dependency management and running
- Run tests: `uv run pytest`
- Run server: `uv run alzabo`

## Conventions

- All MCP tools return plain text, not JSON
- Search supports three modes: `hybrid` (default), `bm25`, `vector`
- Codex session IDs are prefixed with `codex:` to avoid collisions
- Prefer self-contained `uv` scripts for tooling
- Keep changes minimal and focused — don't over-engineer
- Validate work by running tests after changes

## Data Sources

| Source | Directory | Format |
|--------|-----------|--------|
| Claude Code | `~/.claude/projects/**/*.jsonl` | `user`, `assistant`, `summary` |
| Codex | `~/.codex/sessions/**/*.jsonl` | `session_meta`, `event_msg`, `response_item` |
