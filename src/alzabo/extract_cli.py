"""CLI entry point for alzabo-extract: structured tool call extraction."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from .extract import ToolCallRecord, extract_all


def _print_stats(records: list[ToolCallRecord]) -> None:
    if not records:
        print("No tool call records found.")
        return

    total = len(records)
    error_count = sum(1 for r in records if r.is_error)
    tool_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    error_tools: Counter[str] = Counter()
    project_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for rec in records:
        tool_counts[rec.tool_name] += 1
        category_counts[rec.tool_category] += 1
        project_counts[rec.project] += 1
        source_counts[rec.source] += 1
        if rec.is_error:
            error_tools[rec.tool_name] += 1

    print(f"=== alzabo-extract stats ===")
    print(f"total tool calls: {total}")
    print(f"errors: {error_count} ({error_count * 100 / total:.1f}%)")
    print()

    print("by source:")
    for source, count in source_counts.most_common():
        print(f"  {source}: {count}")
    print()

    print("by category:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")
    print()

    print(f"top 15 tools:")
    for tool, count in tool_counts.most_common(15):
        err = error_tools.get(tool, 0)
        err_pct = f" ({err} errors)" if err else ""
        print(f"  {tool}: {count}{err_pct}")
    print()

    if error_tools:
        print("top error tools:")
        for tool, count in error_tools.most_common(10):
            print(f"  {tool}: {count}")
        print()

    print(f"top 10 projects:")
    for proj, count in project_counts.most_common(10):
        print(f"  {proj}: {count}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()
    run_extract(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        tool_filter=args.tool,
        category_filter=args.category,
        project_filter=args.project,
        session_filter=args.session,
        errors_only=args.errors_only,
        stats=args.stats,
        limit=args.limit,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract structured tool call records from Claude Code and Codex transcripts."
    )
    parser.add_argument(
        "--transcripts-dir",
        default=str(Path.home() / ".claude" / "projects"),
        help="Root directory for Claude .jsonl transcripts.",
    )
    parser.add_argument(
        "--codex-dir",
        default=str(Path.home() / ".codex" / "sessions"),
        help="Root directory for Codex .jsonl sessions.",
    )
    parser.add_argument("--tool", default="", help="Filter by tool name substring.")
    parser.add_argument("--category", default="", choices=["builtin", "mcp", "bash", "agent", ""],
                        help="Filter by tool category.")
    parser.add_argument("--project", default="", help="Filter by project name substring.")
    parser.add_argument("--session", default="", help="Filter by session ID.")
    parser.add_argument("--errors-only", action="store_true", help="Only show error records.")
    parser.add_argument("--stats", action="store_true", help="Print summary stats instead of JSONL.")
    parser.add_argument("--limit", type=int, default=0, help="Max records to output (0 = unlimited).")
    return parser


def run_extract(
    *,
    transcripts_dir: Path,
    codex_dir: Path,
    tool_filter: str = "",
    category_filter: str = "",
    project_filter: str = "",
    session_filter: str = "",
    errors_only: bool = False,
    stats: bool = False,
    limit: int = 0,
) -> None:
    filters = {
        "tool_filter": tool_filter,
        "category_filter": category_filter,
        "project_filter": project_filter,
        "session_filter": session_filter,
        "errors_only": errors_only,
    }

    if stats:
        records: list[ToolCallRecord] = []
        for rec in extract_all(transcripts_dir, codex_dir, **filters):
            records.append(rec)
            if limit and len(records) >= limit:
                break
        _print_stats(records)
    else:
        count = 0
        for rec in extract_all(transcripts_dir, codex_dir, **filters):
            print(rec.to_jsonl())
            count += 1
            if limit and count >= limit:
                break
        print(f"# {count} records", file=sys.stderr)


if __name__ == "__main__":
    main()
