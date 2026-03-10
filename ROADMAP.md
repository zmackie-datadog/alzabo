# alzabo roadmap

This file is intentionally a later-phase backlog, not an implementation checklist for the current iteration.

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
- a task set where the only variable is whether `alzabo` is available
- simple scoring rules for success, latency, and provenance

Current status:

- none of the above validation work is implemented in this phase
