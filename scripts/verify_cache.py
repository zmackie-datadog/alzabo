#!/usr/bin/env python
"""Verify cache reuse for alzabo CLI commands."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path


def run_once(binary: str, args: list[str]) -> dict[str, object]:
    proc = subprocess.run(
        shlex.split(binary) + args,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "code": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify alzabo cache load/reuse behavior.")
    parser.add_argument("--binary", default="alzabo", help="CLI command to execute (default: alzabo)")
    parser.add_argument("--transcripts-dir", default="", help="Source directory for Claude JSONL transcripts.")
    parser.add_argument("--codex-dir", default="", help="Source directory for Codex JSONL sessions.")
    parser.add_argument("--cache-dir", default="", help="Directory to use for index cache (defaults to ~/.cache/alzabo).")
    parser.add_argument("--output-dir", default=str(Path.cwd() / "tmp" / "alzabo-cache-verify"), help="Where to write run artifacts.")
    parser.add_argument("--expect-turns", action="store_true", help="Create a tiny Claude transcript with one turn (requires model for indexing).")
    parser.add_argument(
        "--preserve-cache",
        action="store_true",
        help="Do not clear cache directory before the run.",
    )
    parser.add_argument("--quiet", action="store_true", help="Pass --quiet to alzabo runs.")
    opts = parser.parse_args()
    output_dir = Path(opts.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcripts_dir = Path(opts.transcripts_dir or output_dir / "transcripts")
    codex_dir = Path(opts.codex_dir or output_dir / "codex")
    cache_dir = Path(opts.cache_dir or output_dir / "cache")
    if not opts.preserve_cache and cache_dir.exists():
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir)
        else:
            cache_dir.unlink()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    codex_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use a tiny transcript so the command is fast and deterministic.
    if opts.expect_turns:
        project_dir = transcripts_dir / "proj"
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "session.jsonl").write_text(
            "\n".join(
                [
                    '{"type":"user","sessionId":"verify-session","timestamp":"2026-03-13T00:00:00Z","message":{"content":"hello"}}',
                    '{"type":"assistant","sessionId":"verify-session","timestamp":"2026-03-13T00:00:01Z","message":{"content":[{"type":"text","text":"confirmed"}]}}',
                ]
            )
            + "\n"
        )

    common_args = [
        "status",
        "--format", "json",
        "--transcripts-dir", str(transcripts_dir),
        "--codex-dir", str(codex_dir),
        "--cache-dir", str(cache_dir),
    ]
    if opts.quiet:
        common_args.append("--quiet")

    first = run_once(opts.binary, common_args)
    (output_dir / "run1.stdout").write_text(first["stdout"])
    (output_dir / "run1.stderr").write_text(first["stderr"])

    second = run_once(opts.binary, common_args)
    (output_dir / "run2.stdout").write_text(second["stdout"])
    (output_dir / "run2.stderr").write_text(second["stderr"])

    cached = second["code"] == 0 and (
        "loading from cache..." in second["stderr"]
        or "cache loaded" in second["stderr"]
    )
    reindexed = "indexing transcripts..." in second["stderr"]

    manifest_path = cache_dir / "manifest.json"
    summary = {
        "cache_dir": str(cache_dir),
        "transcripts_dir": str(transcripts_dir),
        "codex_dir": str(codex_dir),
        "run1": {
            "returncode": first["code"],
            "indexing_logged": "indexing transcripts..." in first["stderr"],
            "cache_loaded": "loading from cache..." in first["stderr"] or "cache loaded" in first["stderr"],
        },
        "run2": {
            "returncode": second["code"],
            "indexing_logged": reindexed,
            "cache_loaded": cached,
            "cache_save_failed": "cache save failed" in second["stderr"],
        },
        "manifest_exists": manifest_path.exists(),
        "cache_reused": cached and not reindexed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"wrote verification artifacts to {output_dir}")
    print(json.dumps(summary, indent=2))

    if summary["run2"]["returncode"] != 0:
        print("verification failed: second run exited non-zero")
        return 1
    if not summary["manifest_exists"]:
        print("verification failed: cache manifest was not created")
        return 1
    if not summary["cache_reused"]:
        print("verification failed: second run did not appear to load from cache")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
