# alzabo

CLI tool for searching and exploring Claude Code and Codex session transcripts.

## Quick reference

```bash
alzabo search "error handling" --quiet            # BM25 keyword search (fast, default)
alzabo search "error handling" --mode hybrid      # BM25 + vector semantic search
alzabo search "terraform" --sessions --limit 5    # group results by session
alzabo search "auth" --project csm-pde --source claude --start-date 2026-03-01
alzabo list --limit 10 --quiet                    # list recent sessions
alzabo read <session-id> --compact --quiet        # read a session
alzabo read <session-id> --turn 3 --quiet         # read a specific turn
alzabo reindex                                    # full index rebuild
alzabo status --quiet                             # index stats
alzabo extract --stats                            # tool call extraction stats
alzabo --version                                  # show version
```

All commands accept `--format text|json|jsonl` and `--quiet`.

## Project structure

| File | Purpose |
|------|---------|
| `src/alzabo/main_cli.py` | CLI entry point, argument parsing, subcommands, stale-while-revalidate cache loading |
| `src/alzabo/index.py` | Index dataclasses, BM25/vector search, JSONL parsing, TranscriptIndexManager |
| `src/alzabo/cache.py` | Pickle cache save/load, slim index stripping, manifest-based change detection |
| `src/alzabo/output.py` | Output formatting (text/json/jsonl) |
| `src/alzabo/extract_cli.py` | Tool call extraction from transcripts |
| `src/alzabo/parsers.py` | Claude/Codex record parsing and signal extraction |
| `tests/` | Test suite (`uv run pytest tests/ -v`) |
| `scripts/verify_incremental_reindex.sh` | End-to-end incremental reindex verification |

## Architecture

- **Cache**: `~/.cache/alzabo/` — pickle index with pre-built BM25 + numpy embeddings
- **Source data**: `~/.claude/projects/**/*.jsonl` (Claude) + `~/.codex/sessions/**/*.jsonl` (Codex)
- **Stale-while-revalidate**: commands load the cached index immediately and defer incremental rebuilds until after output prints. Next invocation sees updated data. Use `alzabo reindex` for a forced full rebuild.
- **`read` loads content on demand** — the cache stores only metadata; full turn content is re-parsed from source JSONL files.
- **Search modes**: `bm25` (default, fast), `hybrid` (BM25 + vector via RRF), `vector` (semantic only). Hybrid/vector load the `model2vec` embedding model on first use.

## Data sources

| Source | Directory | Format |
|--------|-----------|--------|
| Claude Code | `~/.claude/projects/**/*.jsonl` | `user`, `assistant`, `summary` |
| Codex | `~/.codex/sessions/**/*.jsonl` | `session_meta`, `event_msg`, `response_item` |

Codex session IDs are prefixed with `codex:` to avoid collisions.

## Development

- **Python 3.12+** required
- Use `uv` for dependency management and running
- Run tests: `uv run pytest tests/ -v`
- CLI help: `uv run alzabo --help`
- Keep changes minimal and focused — don't over-engineer
- Validate work by running tests after changes
