# transcript-search roadmap

This file is intentionally a later-phase backlog, not an implementation checklist for the current iteration.

## Standalone Extraction Plan

Trigger criteria:

- `transcript-search/` has no imports from sibling repo directories
- the subproject can be installed and tested from its own `pyproject.toml`
- CI is already scoped to the subproject path

What moves later:

- `transcript-search/README.md`
- `transcript-search/ROADMAP.md`
- `transcript-search/pyproject.toml`
- `transcript-search/transcript-search.py`
- `transcript-search/src/transcript_search/`
- `transcript-search/tests/`
- `.github/workflows/transcript-search.yml`

Post-move tasks:

- replace in-repo path references in docs with standalone repo paths
- decide whether to keep the wrapper script or use only the package entrypoint
- decide whether to publish a release artifact or keep it as a local-only tool
- move any future fixtures or sample transcripts that are specific to this tool

## Validation TODO

Core hypothesis:

- transcript search helps coding agents resume prior work and recover exact commands, files, and failure context faster than transcript-blind operation

Candidate future metrics:

- `Recall@5`
- `MRR`
- task success rate
- completion time
- number of search/read round trips per task

Candidate future benchmark categories:

- interrupted-task resumption
- prior-fix recovery
- cross-session synthesis

Future benchmark assets to prepare:

- a query set with expected session and turn matches
- a task set where the only variable is whether `transcript-search` is available
- simple scoring rules for success, latency, and provenance

Current status:

- none of the above validation work is implemented in this phase
