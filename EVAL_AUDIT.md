# Eval Audit: alzabo

**Date:** 2026-03-11

## Context

alzabo is a transcript search MCP server for Claude Code/Codex sessions. The core hypothesis (from ROADMAP.md) is that transcript search helps agents resume prior work faster. The `alzabo-extract` module extracts structured tool call records as the foundation for a future eval pipeline.

Current eval infrastructure: essentially none. This is acknowledged in ROADMAP.md. The audit below identifies what matters most to build first.

## Findings (by priority)

### 1. No labeled data (Critical)

No labeled query-result pairs exist. This blocks all quality measurement — cannot compute Recall@5, MRR, validate judges, or detect regressions.

**Fix:** Build a 100-query labeled set from real usage. Sampling strategies:
- Search transcripts for `mcp__alzabo__search_conversations` calls — these are real queries agents made
- Find sessions where agents resumed work across sessions; the second session's initial query is a natural test case
- Use `alzabo-extract` output to generate queries from error/fix patterns

Store as a JSON file in-repo.

### 2. No error analysis (High)

No one has systematically reviewed what alzabo returns and categorized failures. ROADMAP lists generic metrics (Recall@5, MRR) but no application-grounded failure categories.

Possible failure modes: wrong project, stale/superseded solutions, wrong turn within correct session, too much context diluting the useful part, terminology mismatch.

**Fix:** Run error-analysis on ~50-100 real search results. Categorize failures before building evaluators.

### 3. No search quality evaluators (High)

The test suite (37 tests) validates mechanical correctness: JSONL parsing, tool correlation, search plumbing, CLI output. None measures whether results are useful.

**Fix:** Build after error analysis. Many will be code-based:
- Wrong project: check returned session's project matches query context
- Stale results: check for more recent sessions on same topic
- Relevance: LLM judge, but only after labeled examples define "relevant"

### 4. No review interface (Medium)

No structured way to review search results for labeling. MCP tools return plain text, making human review tedious.

**Fix:** Build a browser-based annotation UI. Input: query + top-5 results. Output: per-result binary relevance + failure category tag.

### 5. No regression gate (Medium)

Changing the embedding model, BM25 params, turn boundaries, or search text construction has no quality feedback.

**Fix:** Once labeled data exists, add a pytest test that asserts Recall@5 stays above a threshold.

### 6. No judge validation (Not applicable yet)

No judges built yet. Correct state — judges come after error analysis.

**Fix:** When judges are built, validate with TPR/TNR on held-out human labels (~50 Pass + ~50 Fail per judge).

## What exists and is OK

- Test suite validates mechanical correctness (parsing, serialization, search plumbing)
- `alzabo-extract --stats` provides usage statistics (tool counts, error rates)
- Data structures (TurnSignals, SearchResultSet, ToolCallRecord) can support evaluation
- BM25 + vector similarity (RRF fusion) is the retrieval mechanism, not being conflated with quality metrics
- Domain expert (developer/primary user) is available as reviewer

## Recommended sequence

1. **Now:** Use `alzabo-extract` + transcript search to collect real queries agents made to alzabo
2. **Next:** Error-analysis on ~50 query-result pairs, tag failures
3. **Then:** Build labeled query set (100 entries) stored in-repo
4. **Then:** Add regression test computing Recall@5 on labeled set
5. **Later:** LLM judges only for failure modes code can't check; validate each one
