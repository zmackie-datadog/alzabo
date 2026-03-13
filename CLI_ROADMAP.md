# Alzabo CLI-First Refactor Roadmap

## Context

Alzabo is currently MCP-only. The AI agent ecosystem is trending toward CLI + skills as the primary interface, with MCP as an optional integration layer. This roadmap makes alzabo CLI-first while keeping MCP as a first-class adapter.

**Target CLI:**
```bash
alzabo search "oauth callback bug"              # search turns
alzabo search --sessions "vector search"         # search grouped by session
alzabo list --source codex --project alzabo      # browse sessions
alzabo read <session-id>                         # read conversation
alzabo read <session-id> --turn 17               # read single turn
alzabo status                                    # index stats
alzabo serve                                     # start MCP server (current behavior)
alzabo extract                                   # tool call extraction (current alzabo-extract)
```

## Steps

### Step 1: Decouple `server.py` from module-level singletons
- Replace `manager = TranscriptIndexManager()` and `server = FastMCP("alzabo")` with `create_mcp_server(manager)` factory
- Update `cli.py` to construct manager and pass it
- Update `tests/test_server.py` fixtures
- **Verify:** `uv run pytest` passes, MCP server still starts with `uv run alzabo`

### Step 2: Add `as_dict()` to result dataclasses
- Add `as_dict()` to `SearchResultSet`, `SessionResultSet`, `ConversationPage`, `IndexStatus`, `TurnSearchResult`, `SessionSearchResult` in `index.py`
- **Verify:** unit tests for each `as_dict()` method

### Step 3: Create output formatting layer (`output.py`)
- Dispatch on format: text → existing `render.py`, json → `as_dict()` + `json.dumps`, jsonl → one line per item
- **Verify:** test JSON/JSONL validity for each result type

### Step 4: Create disk-based index cache (`cache.py`)
- Serialize index to `~/.cache/alzabo/` (manifest.json + turns.json + embeddings.npy)
- Staleness check via file mtime comparison
- Allow `TranscriptIndexManager` to accept pre-built `Index`
- **Verify:** cache roundtrip, staleness detection, corrupt cache recovery

### Step 5: Create unified CLI entry point (`main_cli.py`)
- argparse with subparsers: search, list, read, status, serve, extract
- `serve` delegates to refactored `cli.py`, `extract` delegates to `extract_cli.py`
- Common flags: --transcripts-dir, --codex-dir, --format, --no-cache
- **Verify:** each subcommand works end-to-end

### Step 6: Update entry points in `pyproject.toml`
- Change `alzabo` entry point to `alzabo.main_cli:main`
- Keep `alzabo-extract` for one release cycle
- **Verify:** `uvx alzabo search "test"` works

### Step 7: Backward compatibility + polish
- Detect old-style bare `alzabo --watch` and route to `serve` with deprecation warning
- Update README to lead with CLI usage
- **Verify:** old MCP config still works with `alzabo serve`

## Status
- Step 1: ✅
- Step 2: ✅
- Step 3: ✅
- Step 4: ✅
- Step 5: ✅
- Step 6: ✅
- Step 7: ✅
