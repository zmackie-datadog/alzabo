#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "mcp[cli]>=1.0",
#     "rank-bm25",
#     "watchdog>=4.0",
#     "model2vec",
#     "numpy",
# ]
# ///
"""Compatibility wrapper for the packaged transcript-search server.

Run:
  uv run transcript-search.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from transcript_search.cli import main


if __name__ == "__main__":
    main()
