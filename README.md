# alzabo

CLI tool for searching and exploring Claude Code and Codex session transcripts with hybrid search (BM25 + vector).

## Install

```bash
uv tool install git+https://github.com/zmackie-datadog/alzabo
```

## Usage

```bash
alzabo search "oauth token refresh"         # search turns (BM25, fast)
alzabo search "error handling" --mode hybrid # BM25 + vector (loads model, slower)
alzabo search "vector db" --sessions         # group results by session
alzabo search "terraform" --format json      # JSON output
alzabo list --format jsonl                   # list all sessions as JSONL
alzabo read <session-id> --turn 0            # read a specific turn
alzabo read <session-id> --compact           # compact session view
alzabo status                                # index stats
alzabo reindex                               # rebuild index from source files
alzabo extract --stats                       # tool call extraction stats
```

### Subcommands

| Subcommand | Description |
|---|---|
| `search QUERY` | Search turns. `--sessions` groups by session. `--mode {hybrid,bm25,vector}` (default: bm25), `--context-window N` |
| `list` | List conversations. `--source`, `--project`, `--start-date`, `--end-date`, `--offset`, `--limit` |
| `read SESSION_ID` | Read a conversation. `--turn N` for a single turn. `--compact`, `--include-records` |
| `status` | Show index stats: session counts, turn count, embeddings state, last reindex time |
| `reindex` | Rebuild the search index from source JSONL files |
| `extract` | Extract structured tool call records. `--stats`, `--tool`, `--category`, `--errors-only` |

### Global flags

- `--format {text,json,jsonl}`: output format (default: `text`)
- `--cache-dir`: override cache directory (default: `~/.cache/alzabo`)
- `--transcripts-dir`: Claude transcript directory (default: `~/.claude/projects`)
- `--codex-dir`: Codex session directory (default: `~/.codex/sessions`)
- `--quiet`: suppress progress logs for cleaner LLM/agent consumption

### Disk cache

alzabo builds a pickle-based index on first `reindex` and saves it to `~/.cache/alzabo/`. Search loads the cache as-is and never reindexes automatically -- run `alzabo reindex` explicitly when you want fresh data.

## Search modes

- `bm25` (default): keyword-only search, fast
- `hybrid`: Reciprocal Rank Fusion of BM25 and vector search (loads `model2vec` embeddings on first use)
- `vector`: semantic search using `model2vec`

## What gets indexed

- User prompts and assistant text
- Claude `tool_use` inputs and `tool_result` payloads
- Codex function-call arguments and `function_call_output` payloads
- Extracted signals per turn: tools, file paths, shell commands, and error snippets

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
| `alzabo-extract` | `alzabo.extract_cli:main` | Standalone extraction tool |

## Development

```bash
uv run pytest tests/ -v
```
