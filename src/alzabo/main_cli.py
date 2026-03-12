"""Unified CLI entry point for alzabo: search, list, read, status, serve, extract."""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path


def _get_manager(args: argparse.Namespace) -> "TranscriptIndexManager":
    """Build a TranscriptIndexManager, using disk cache when possible."""
    from .cache import is_cache_fresh, load_cache, save_cache
    from .index import TranscriptIndexManager, _log

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()

    manager = TranscriptIndexManager()
    manager.configure(transcripts_dir=transcripts_dir, codex_dir=codex_dir, watch_enabled=False)

    no_cache = getattr(args, "no_cache", False)

    if not no_cache and is_cache_fresh(transcripts_dir, codex_dir):
        _log("loading from cache...")
        index = load_cache()
        if index is not None:
            manager.set_index(index)
            return manager
        _log("cache load failed, falling back to reindex")

    _log("indexing transcripts...")
    manager.reindex()

    if not no_cache:
        save_cache(manager._index, transcripts_dir, codex_dir)

    return manager


def cmd_search(args: argparse.Namespace) -> None:
    from .output import format_search_results, format_session_results

    manager = _get_manager(args)
    fmt = args.format

    if args.sessions:
        result = manager.search_sessions(
            query=args.query,
            limit=args.limit,
            source=args.source,
            project=args.project,
            start_date=args.start_date,
            end_date=args.end_date,
            mode=args.mode,
        )
        print(format_session_results(result, fmt))
    else:
        result = manager.search_conversations(
            query=args.query,
            limit=args.limit,
            source=args.source,
            project=args.project,
            start_date=args.start_date,
            end_date=args.end_date,
            mode=args.mode,
            context_window=args.context_window,
        )
        print(format_search_results(result, fmt))


def cmd_list(args: argparse.Namespace) -> None:
    from .output import format_conversation_page

    manager = _get_manager(args)
    page = manager.list_conversations(
        source=args.source,
        project=args.project,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        offset=args.offset,
    )
    print(format_conversation_page(page, args.format))


def cmd_read(args: argparse.Namespace) -> None:
    from .output import format_conversation, format_turn

    manager = _get_manager(args)
    fmt = args.format

    if args.turn is not None:
        try:
            turn = manager.get_turn(args.session_id, args.turn)
        except KeyError:
            print(f"error: session not found: {args.session_id}", file=sys.stderr)
            sys.exit(1)
        except IndexError:
            print(f"error: turn out of range: {args.turn}", file=sys.stderr)
            sys.exit(1)
        print(format_turn(turn, fmt, include_records=args.include_records, include_content=args.include_content))
    else:
        try:
            convo = manager.get_conversation(args.session_id)
        except KeyError:
            print(f"error: session not found: {args.session_id}", file=sys.stderr)
            sys.exit(1)
        print(
            format_conversation(
                convo,
                fmt,
                offset=args.offset,
                limit=args.limit,
                include_records=args.include_records,
                include_content=args.include_content,
                compact=args.compact,
            )
        )


def cmd_status(args: argparse.Namespace) -> None:
    from .output import format_index_status

    manager = _get_manager(args)
    status = manager.get_index_status()
    print(format_index_status(status, args.format))


def cmd_serve(args: argparse.Namespace) -> None:
    from watchdog.observers import Observer

    from .cli import TranscriptChangeHandler
    from .index import TranscriptIndexManager, _log
    from .server import create_mcp_server

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()

    manager = TranscriptIndexManager()
    manager.configure(transcripts_dir=transcripts_dir, codex_dir=codex_dir, watch_enabled=args.watch)

    bg = threading.Thread(target=manager.reindex, daemon=True)
    bg.start()

    observer: Observer | None = None
    if args.watch:
        handler = TranscriptChangeHandler(manager, args.debounce_seconds)
        observer = Observer()
        scheduled = False
        for path in (transcripts_dir, codex_dir):
            if path.exists() and path.is_dir():
                observer.schedule(handler, str(path), recursive=True)
                scheduled = True
        if scheduled:
            observer.daemon = True
            observer.start()
        else:
            _log("watch requested but no transcript directories exist; watcher disabled")
            observer = None

    server = create_mcp_server(manager)
    try:
        server.run()
    finally:
        if observer is not None:
            observer.stop()
            observer.join(timeout=2)


def cmd_extract(args: argparse.Namespace) -> None:
    from .extract import extract_all
    from .extract_cli import _print_stats

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()

    filters = {
        "tool_filter": args.tool,
        "category_filter": args.category,
        "project_filter": args.project,
        "session_filter": args.session,
        "errors_only": args.errors_only,
    }

    if args.stats:
        records = []
        for rec in extract_all(transcripts_dir, codex_dir, **filters):
            records.append(rec)
            if args.extract_limit and len(records) >= args.extract_limit:
                break
        _print_stats(records)
    else:
        count = 0
        for rec in extract_all(transcripts_dir, codex_dir, **filters):
            print(rec.to_jsonl())
            count += 1
            if args.extract_limit and count >= args.extract_limit:
                break
        print(f"# {count} records", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alzabo", description="Search and explore Claude Code and Codex transcripts.")

    # Shared parent with global flags — lets them appear before or after subcommand
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--transcripts-dir",
        default=str(Path.home() / ".claude" / "projects"),
        help="Root directory for Claude .jsonl transcripts.",
    )
    shared.add_argument(
        "--codex-dir",
        default=str(Path.home() / ".codex" / "sessions"),
        help="Root directory for Codex .jsonl sessions.",
    )
    shared.add_argument("--format", choices=["text", "json", "jsonl"], default="text", help="Output format.")
    shared.add_argument("--no-cache", action="store_true", help="Skip disk cache, always reindex.")

    subparsers = parser.add_subparsers(dest="command")

    # --- search ---
    p_search = subparsers.add_parser("search", parents=[shared], help="Search transcripts.")
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("--sessions", action="store_true", help="Group results by session.")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--source", default="")
    p_search.add_argument("--project", default="")
    p_search.add_argument("--start-date", default="")
    p_search.add_argument("--end-date", default="")
    p_search.add_argument("--mode", choices=["hybrid", "bm25", "vector"], default="hybrid")
    p_search.add_argument("--context-window", type=int, default=0)
    p_search.set_defaults(func=cmd_search)

    # --- list ---
    p_list = subparsers.add_parser("list", parents=[shared], help="List conversations.")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.add_argument("--offset", type=int, default=0)
    p_list.add_argument("--source", default="")
    p_list.add_argument("--project", default="")
    p_list.add_argument("--start-date", default="")
    p_list.add_argument("--end-date", default="")
    p_list.set_defaults(func=cmd_list)

    # --- read ---
    p_read = subparsers.add_parser("read", parents=[shared], help="Read a conversation or turn.")
    p_read.add_argument("session_id", help="Session ID.")
    p_read.add_argument("--turn", type=int, default=None, help="Specific turn number.")
    p_read.add_argument("--offset", type=int, default=0)
    p_read.add_argument("--limit", type=int, default=20)
    p_read.add_argument("--include-records", action="store_true")
    p_read.add_argument("--include-content", action=argparse.BooleanOptionalAction, default=True)
    p_read.add_argument("--compact", action="store_true")
    p_read.set_defaults(func=cmd_read)

    # --- status ---
    p_status = subparsers.add_parser("status", parents=[shared], help="Show index status.")
    p_status.set_defaults(func=cmd_status)

    # --- serve ---
    p_serve = subparsers.add_parser("serve", parents=[shared], help="Start MCP server.")
    p_serve.add_argument("--watch", action=argparse.BooleanOptionalAction, default=True)
    p_serve.add_argument("--debounce-seconds", type=float, default=2.0)
    p_serve.set_defaults(func=cmd_serve)

    # --- extract ---
    p_extract = subparsers.add_parser("extract", parents=[shared], help="Extract structured tool call records.")
    p_extract.add_argument("--tool", default="", help="Filter by tool name substring.")
    p_extract.add_argument(
        "--category", default="", choices=["builtin", "mcp", "bash", "agent", ""],
        help="Filter by tool category.",
    )
    p_extract.add_argument("--project", default="", help="Filter by project name.")
    p_extract.add_argument("--session", default="", help="Filter by session ID.")
    p_extract.add_argument("--errors-only", action="store_true")
    p_extract.add_argument("--stats", action="store_true", help="Print summary stats instead of JSONL.")
    p_extract.add_argument("--extract-limit", type=int, default=0, help="Max records (0 = unlimited).")
    p_extract.set_defaults(func=cmd_extract)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
