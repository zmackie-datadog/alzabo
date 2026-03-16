# Simplify to CLI-only architecture

## Context

`alzabo search` is broken: the daemon dies immediately because `_start_daemon_process()` spawns `python -m alzabo.cli` with `stdin=DEVNULL`, but `run_mcp_server()` calls `server.run()` (FastMCP stdio) which reads stdin, gets EOF, and exits — killing the IPC server. Every CLI invocation falls back to full reindex, loading the embedding model each time.

Rather than fix the daemon, we're simplifying to CLI-only. MCP is token-heavy; the CLI with disk cache is fast enough (~100ms for fresh cache, no model load). Claude Code can invoke `alzabo search` via Bash.

## Files to delete

| File | Reason |
|---|---|
| `src/alzabo/server.py` | MCP server tool definitions (FastMCP) |
| `src/alzabo/ipc.py` | IPC bridge (server + client) |
| `src/alzabo/cli.py` | MCP server entry point (`alzabo serve` / `alzabo-serve`) |
| `tests/test_ipc.py` | IPC tests |

## Files to modify

### 1. `src/alzabo/main_cli.py`

**Remove** (~200 lines):
- Imports: `subprocess`, `time`
- Constants: `_COMMANDS`, `_LEGACY_SERVE_FLAGS`, `_DAEMON_START_*` (lines 13-25)
- Daemon lifecycle functions (lines 28-153): `_daemon_start_lock_path`, `_is_stale_lock`, `_acquire_daemon_start_lock`, `_release_daemon_start_lock`, `_is_daemon_running`, `_start_daemon_process`, `_wait_for_daemon_ready`, `_ensure_daemon_running`
- IPC functions (lines 167-209): `_should_fallback_from_ipc`, `_send_ipc_request`
- `cmd_daemon_status()` (lines 212-246)
- Legacy serve functions (lines 249-305): `_build_legacy_serve_parser`, `_run_legacy_serve`, `_is_legacy_serve_invocation`
- `cmd_serve()` (lines 546-560)

**Simplify** command handlers — remove IPC try/fallback, call `_get_manager()` directly:
```python
def cmd_search(args):
    manager = _get_manager(args)
    # ... search logic unchanged
```

Same for `cmd_list`, `cmd_read`, `cmd_status`.

**Simplify** `build_parser()`:
- Remove `serve` and `daemon-status` subparsers
- Remove `--no-daemon` flag from shared parser

**Simplify** `main()`:
- Remove `_is_legacy_serve_invocation` check

### 2. `tests/test_main_cli.py`

**Remove** tests (~400 lines):
- `test_serve_parses` (line 54)
- `test_daemon_status_parses` (line 67)
- `test_no_daemon_flag` (line 98)
- `test_search_prefers_ipc_when_available` (line 596)
- `test_daemon_status_reports_running` (line 626)
- `test_daemon_status_reports_unavailable` (line 655)
- `test_fallback_when_ipc_server_unavailable` (line 675)
- `test_search_starts_daemon_when_ipc_missing` (line 716)
- `test_start_daemon_process_invokes_cli_entrypoint` (line 769)
- `test_start_daemon_process_includes_watch_and_quiet_flags` (line 807)
- `test_acquire_daemon_start_lock_serializes_starts` (line 835)
- `test_ensure_daemon_waits_for_existing_starter` (line 847)
- `TestLegacyCompatibility` class (lines 878-966) — legacy serve and serve delegation tests

**Keep** `test_extract_subcommand_delegates_to_extract_cli` (line 968) — move out of `TestLegacyCompatibility` to standalone function.

### 3. `tests/test_server.py`

**Remove** only `TestStartupLatency` class (line 439) — tests MCP handshake via `alzabo-serve`.

**Keep** everything else — `TestHelpers`, `TestParsers`, `TestIndexBuilding`, `TestManagerQueries`, `TestAsDict`, `TestReindex` all test `index.py`/`render.py`/`parsers.py`, not the MCP server.

### 4. `pyproject.toml`

- Remove `alzabo-serve` entry point (line 34)
- Remove `mcp[cli]>=1.0` dependency (line 24)
- Remove `watchdog>=4.0` dependency (line 28)
- Update description to "CLI tool for searching Claude Code and Codex transcripts."

## Execution order

1. Delete `server.py`, `ipc.py`, `cli.py`, `tests/test_ipc.py`
2. Simplify `main_cli.py` (remove daemon/IPC/serve/legacy code)
3. Update `tests/test_main_cli.py` (remove deleted-code tests)
4. Update `tests/test_server.py` (remove `TestStartupLatency`)
5. Update `pyproject.toml` (deps + entry points)
6. Run `uv run pytest tests/` to verify

## Verification

1. `uv run pytest tests/` — all tests pass
2. `uv run alzabo search "test query" --quiet` — loads from disk cache, no daemon spawn
3. Run twice in a row — second invocation should NOT print "loading embedding model" if cache is fresh
4. `uv run alzabo status` — shows index stats
5. `uv run alzabo list --limit 3` — lists sessions
