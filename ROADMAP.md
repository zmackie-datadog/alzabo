# alzabo roadmap

This file is intentionally a later-phase backlog, not an implementation checklist for the current iteration.

## Extraction (done)

`alzabo-extract` CLI extracts structured tool call records from Claude Code and Codex JSONL transcripts. Each record includes tool name, category, input, output, error status, session/project context, and correlation IDs.

This is the data foundation for the eval pipeline below.

## Validation TODO

Core hypothesis:

- transcript search helps coding agents resume prior work and recover exact commands, files, and failure context faster than transcript-blind operation

### Phase 1 — Labeled data & error analysis (next)

- collect real queries agents made to alzabo (via `alzabo-extract` filtering for `mcp__alzabo__*` tool calls)
- manually review ~50-100 query-result pairs, categorize failure modes
- build a labeled query set (~100 entries) stored as JSON in-repo
- see git history for prior eval audit findings and sampling strategies

### Phase 2 — Evaluators & regression gate

- code-based evaluators for objective failures (wrong project, stale results)
- LLM judge for subjective relevance (only after labeled examples exist)
- validate judges with TPR/TNR on held-out labels
- pytest regression test asserting Recall@5 above threshold

### Phase 3 — ODP backend & benchmark

- add `extract_from_odp()` querying `llmobs` track via TrinoSQL for duration_ms and production traces
- a task set where the only variable is whether alzabo is available
- candidate metrics: Recall@5, MRR, task success rate, completion time, search/read round trips per task
- candidate benchmark categories: interrupted-task resumption, prior-fix recovery, cross-session synthesis

Current status:

- extraction module implemented (Phase 0)
- no validation work implemented yet
