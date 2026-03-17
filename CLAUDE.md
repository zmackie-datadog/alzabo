# alzabo

CLI tool for searching and exploring Claude Code and Codex session transcripts.

## Quick reference

```bash
# Search (BM25, fast — default)
alzabo search "error handling" --quiet

# Search with semantic matching (loads embedding model, slower)
alzabo search "error handling" --quiet --mode hybrid

# Filter by project, source, date
alzabo search "auth" --project csm-pde --source claude --start-date 2026-03-01

# Group results by session instead of individual turns
alzabo search "terraform" --sessions --limit 5

# List recent sessions
alzabo list --limit 10 --quiet

# Read a specific session
alzabo read <session-id> --compact --quiet

# Read a specific turn
alzabo read <session-id> --turn 3 --quiet

# Rebuild the index (run when you want fresh data)
alzabo reindex

# Check index status
alzabo status --quiet
```

## Output formats

All commands accept `--format text|json|jsonl`. Use `--format json` for structured output when parsing programmatically. Use `--quiet` to suppress stderr progress logs.

## Architecture

- **Cache**: `~/.cache/alzabo/` — pickle index with pre-built BM25 + numpy embeddings
- **Source data**: `~/.claude/projects/**/*.jsonl` (Claude) + `~/.codex/sessions/**/*.jsonl` (Codex)
- **Search never reindexes** — it loads the cache as-is. Run `alzabo reindex` explicitly when you want fresh data.
- **`read` loads content on demand** — the cache stores only metadata; full turn content is re-parsed from source JSONL files.

## Development

```bash
uv run pytest tests/ -v          # run tests
uv run alzabo --help             # CLI help
bash scripts/verify_cli_only.sh  # end-to-end verification
```

## Key files

| File | Purpose |
|------|---------|
| `src/alzabo/index.py` | Index dataclasses, BM25/vector search, JSONL parsing, TranscriptIndexManager |
| `src/alzabo/cache.py` | Pickle cache save/load, slim index stripping, manifest management |
| `src/alzabo/main_cli.py` | CLI entry point, argument parsing, subcommands |
| `src/alzabo/output.py` | Output formatting (text/json/jsonl) |
| `src/alzabo/extract_cli.py` | Tool call extraction from transcripts |
| `src/alzabo/parsers.py` | Claude/Codex record parsing |
