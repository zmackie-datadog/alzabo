"""Unified CLI entry point for alzabo: search, list, read, status, reindex, extract."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import click

_PENDING_UPDATE: DeferredUpdate | None = None


@dataclass
class DeferredUpdate:
    """A pending incremental index refresh scheduled after command output."""

    cached_index: "Index"
    manifest: dict
    transcripts_dir: Path
    codex_dir: Path
    reindex_at: str


def _resolve_cache_dir(args: SimpleNamespace) -> Path:
    from . import cache as cache_mod

    cache_dir = getattr(args, "cache_dir", "")
    if not cache_dir:
        cache_dir = os.environ.get("ALZABO_CACHE_DIR", "")
    if cache_dir:
        cache_mod.set_cache_dir(cache_dir)
    return cache_mod.get_cache_dir()


def _load_manager(args: SimpleNamespace) -> "TranscriptIndexManager":
    """Build a TranscriptIndexManager, using disk cache when possible.

    Loads from cache first; if cache is stale, defers incremental update until after
    output is rendered.
    If no cache exists, performs a full reindex (cold start only).
    """
    from .cache import (
        is_cache_recently_checked,
        load_cache_bundle,
        save_cache,
        set_log_enabled as set_cache_log_enabled,
    )
    from .index import (
        TranscriptIndexManager,
        _log,
        set_log_enabled as set_index_log_enabled,
    )

    logging_enabled = not args.quiet
    set_index_log_enabled(logging_enabled)
    set_cache_log_enabled(logging_enabled)

    _resolve_cache_dir(args)

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()

    manager = TranscriptIndexManager()
    manager.configure(transcripts_dir=transcripts_dir, codex_dir=codex_dir, watch_enabled=False)

    global _PENDING_UPDATE
    _PENDING_UPDATE = None
    no_cache = getattr(args, "no_cache", False)

    if not no_cache:
        cache_bundle = load_cache_bundle()
        if cache_bundle is not None:
            cached_index, manifest = cache_bundle
            cache_reindex_at = manifest.get("reindex_at", "")
            same_directories = (
                manifest.get("transcripts_dir") == str(transcripts_dir)
                and manifest.get("codex_dir") == str(codex_dir)
            )
            if same_directories:
                _log("loading from cache...")
                manager.set_index(cached_index, reindex_at=cache_reindex_at)

                if is_cache_recently_checked(manifest, 30.0):
                    _log("cache checked recently; skipping source scan")
                    return manager

                _PENDING_UPDATE = DeferredUpdate(
                    cached_index=cached_index,
                    manifest=manifest,
                    transcripts_dir=transcripts_dir,
                    codex_dir=codex_dir,
                    reindex_at=cache_reindex_at,
                )
                return manager

    # Cold start: no usable cache, do a full reindex
    _log("indexing transcripts...")
    manager.reindex()

    if not no_cache:
        save_cache(manager._index, transcripts_dir, codex_dir)

    return manager


def _flush_deferred_update() -> None:
    global _PENDING_UPDATE
    if _PENDING_UPDATE is None:
        return

    from .cache import (
        collect_source_files,
        changed_source_files,
        save_cache,
        touch_cache_checked_at,
    )
    from .index import _log, rebuild_index_incrementally

    pending = _PENDING_UPDATE
    _PENDING_UPDATE = None

    current_files = collect_source_files(pending.transcripts_dir, pending.codex_dir)
    previous_files = pending.manifest.get("source_files", {})
    if not isinstance(previous_files, dict):
        previous_files = {}

    changed_files = changed_source_files(previous_files, current_files)
    if not changed_files:
        touch_cache_checked_at(pending.transcripts_dir, pending.codex_dir)
        return

    _log(f"{len(changed_files)} files changed, updating index...")
    incremental_index = rebuild_index_incrementally(
        pending.cached_index,
        changed_files,
        transcripts_dir=pending.transcripts_dir,
        codex_dir=pending.codex_dir,
    )
    if incremental_index is None:
        touch_cache_checked_at(pending.transcripts_dir, pending.codex_dir)
        return

    save_cache(
        incremental_index,
        pending.transcripts_dir,
        pending.codex_dir,
        reindex_at=pending.reindex_at,
    )


def cmd_search(args: SimpleNamespace) -> None:
    from .output import format_search_results, format_session_results

    manager = _load_manager(args)
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


def cmd_list(args: SimpleNamespace) -> None:
    from .output import format_conversation_page

    manager = _load_manager(args)
    page = manager.list_conversations(
        source=args.source,
        project=args.project,
        start_date=args.start_date,
        end_date=args.end_date,
        limit=args.limit,
        offset=args.offset,
    )
    print(format_conversation_page(page, args.format))


def cmd_read(args: SimpleNamespace) -> None:
    from .output import format_conversation, format_turn
    from .index import load_conversation_content

    manager = _load_manager(args)
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

        if args.include_content and turn.user_content is None and turn.source_file:
            convo = load_conversation_content(turn.session_id, {turn.source_file})
            if convo is not None:
                for t in convo.turns:
                    if t.turn_number == turn.turn_number:
                        turn = t
                        break

        print(format_turn(turn, fmt, include_records=args.include_records, include_content=args.include_content))
    else:
        try:
            convo = manager.get_conversation(args.session_id)
        except KeyError:
            print(f"error: session not found: {args.session_id}", file=sys.stderr)
            sys.exit(1)

        if args.include_content and convo.turns and convo.turns[0].user_content is None:
            source_files = {t.source_file for t in convo.turns if t.source_file}
            if source_files:
                full_convo = load_conversation_content(convo.session_id, source_files)
                if full_convo is not None:
                    convo = full_convo

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


def cmd_status(args: SimpleNamespace) -> None:
    from .output import format_index_status

    manager = _load_manager(args)
    status = manager.get_index_status()
    print(format_index_status(status, args.format))


def cmd_reindex(args: SimpleNamespace) -> None:
    """Explicit reindex: rebuild the cache from source files."""
    from .cache import save_cache, set_log_enabled as set_cache_log_enabled
    from .index import TranscriptIndexManager, _log, set_log_enabled as set_index_log_enabled

    logging_enabled = not args.quiet
    set_index_log_enabled(logging_enabled)
    set_cache_log_enabled(logging_enabled)

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()

    manager = TranscriptIndexManager()
    manager.configure(transcripts_dir=transcripts_dir, codex_dir=codex_dir, watch_enabled=False)

    total = manager.reindex()
    save_cache(manager._index, transcripts_dir, codex_dir)
    _log(f"reindex complete: {total} turns cached")


def cmd_extract(args: SimpleNamespace) -> None:
    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    codex_dir = Path(args.codex_dir).expanduser().resolve()
    from .extract_cli import run_extract

    run_extract(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        tool_filter=args.tool,
        category_filter=args.category,
        project_filter=args.project,
        session_filter=args.session,
        errors_only=args.errors_only,
        stats=args.stats,
        limit=args.extract_limit,
    )


def _get_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("alzabo")
    except PackageNotFoundError:
        return "0.0.0-dev"


def _global_options(func):
    defaults = {
        "transcripts_dir": str(Path.home() / ".claude" / "projects"),
        "codex_dir": str(Path.home() / ".codex" / "sessions"),
    }

    decorators = [
        click.option(
            "--transcripts-dir",
            default=defaults["transcripts_dir"],
            show_default=True,
            help=(
                "Root directory for Claude transcripts. This can be any folder containing "
                "Claude-style JSONL files; default is ~/.claude/projects."
            ),
        ),
        click.option(
            "--codex-dir",
            default=defaults["codex_dir"],
            show_default=True,
            help=(
                "Root directory for Codex session transcripts. This can be any folder "
                "containing Codex-style JSONL sessions; default is ~/.codex/sessions."
            ),
        ),
        click.option(
            "--format",
            "format_",
            default="text",
            show_default=True,
            type=click.Choice(["text", "json", "jsonl"]),
            help=(
                "Output format for command payloads. 'text' prints human-readable output, "
                "'json' prints JSON objects, and 'jsonl' prints newline-delimited JSON lines."
            ),
        ),
        click.option(
            "--no-cache",
            is_flag=True,
            default=False,
            help=(
                "Bypass cache loads and writes for this run. Forces a full reindex before "
                "running the command (except extract/reindex which do not use the cache)."
            ),
        ),
        click.option(
            "--cache-dir",
            default="",
            show_default=True,
            help=(
                "Override cache directory path. When empty, ALZABO_CACHE_DIR is consulted "
                "and then ~/.cache/alzabo is used."
            ),
        ),
        click.option(
            "--quiet",
            is_flag=True,
            default=False,
            help=(
                "Silence progress logging and status messages while still printing command output."
            ),
        ),
    ]
    for decorator in reversed(decorators):
        func = decorator(func)
    return func


def _namespace_for_command(
    *,
    query: str | None = None,
    **kwargs,
) -> SimpleNamespace:
    args = {
        "quiet": kwargs.pop("quiet"),
        "cache_dir": kwargs.pop("cache_dir"),
        "transcripts_dir": kwargs.pop("transcripts_dir"),
        "codex_dir": kwargs.pop("codex_dir"),
        "no_cache": kwargs.pop("no_cache"),
        "format": kwargs.pop("format_"),
    }
    args["query"] = query
    args.update(kwargs)
    return SimpleNamespace(**args)


@click.group(invoke_without_command=True)
@click.version_option(_get_version(), "--version")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Search and explore Claude Code and Codex transcripts from the terminal."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        ctx.exit(1)


@cli.command("search", help="Search transcripts.")
@_global_options
@click.argument("query", metavar="QUERY")
@click.option(
    "--sessions",
    is_flag=True,
    help="Group search results by session instead of returning individual turns.",
)
@click.option(
    "--limit",
    default=10,
    show_default=True,
    type=int,
    help=(
        "Maximum number of results to return. For session mode this limits sessions; "
        "for default mode it limits turns."
    ),
)
@click.option(
    "--source",
    default="",
    help="Filter results to one source, e.g. 'claude' or 'codex'. Empty means both.",
)
@click.option(
    "--project",
    default="",
    help="Filter results to one project name. Empty means all projects.",
)
@click.option(
    "--start-date",
    default="",
    help=(
        "Filter results to sessions/turns on/after this ISO timestamp. "
        "Format should be YYYY-MM-DD or full ISO datetime."
    ),
)
@click.option(
    "--end-date",
    default="",
    help=(
        "Filter results to sessions/turns before this ISO timestamp. "
        "Format should be YYYY-MM-DD or full ISO datetime."
    ),
)
@click.option(
    "--mode",
    default="bm25",
    show_default=True,
    type=click.Choice(["bm25", "hybrid", "vector"]),
    help=(
        "Search ranking mode. 'bm25' uses keyword matching and is fastest. "
        "'hybrid' combines BM25 with semantic vector search via reciprocal-rank fusion. "
        "'vector' uses semantic search only and requires vector model initialization."
    ),
)
@click.option(
    "--context-window",
    "context_window",
    default=0,
    type=int,
    help=(
        "Number of surrounding turns to include as extra context around each matching turn. "
        "A value of 0 returns exact matches only."
    ),
)
def search(
    query: str,
    sessions: bool,
    limit: int,
    source: str,
    project: str,
    start_date: str,
    end_date: str,
    mode: str,
    context_window: int,
    transcripts_dir: str,
    codex_dir: str,
    format_: str,
    no_cache: bool,
    cache_dir: str,
    quiet: bool,
) -> None:
    args = _namespace_for_command(
        query=query,
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        format_=format_,
        no_cache=no_cache,
        cache_dir=cache_dir,
        quiet=quiet,
        sessions=sessions,
        limit=limit,
        source=source,
        project=project,
        start_date=start_date,
        end_date=end_date,
        mode=mode,
        context_window=context_window,
    )
    cmd_search(args)


@cli.command("list", help="List conversations.")
@_global_options
@click.option(
    "--limit",
    default=20,
    show_default=True,
    type=int,
    help="Maximum number of conversations to include per page.",
)
@click.option(
    "--offset",
    default=0,
    show_default=True,
    type=int,
    help="Number of conversations to skip before the returned page.",
)
@click.option(
    "--source",
    default="",
    help="Filter list results to one source, e.g. 'claude' or 'codex'. Empty means both.",
)
@click.option(
    "--project",
    default="",
    help="Filter list results to one project name. Empty means all projects.",
)
@click.option(
    "--start-date",
    default="",
    help="Include only turns/conversations after this date (ISO format preferred).",
)
@click.option(
    "--end-date",
    default="",
    help="Include only turns/conversations before this date (ISO format preferred).",
)
def list_convos(
    limit: int,
    offset: int,
    source: str,
    project: str,
    start_date: str,
    end_date: str,
    transcripts_dir: str,
    codex_dir: str,
    format_: str,
    no_cache: bool,
    cache_dir: str,
    quiet: bool,
) -> None:
    args = _namespace_for_command(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        format_=format_,
        no_cache=no_cache,
        cache_dir=cache_dir,
        quiet=quiet,
        limit=limit,
        offset=offset,
        source=source,
        project=project,
        start_date=start_date,
        end_date=end_date,
    )
    cmd_list(args)


@cli.command("read", help="Read a conversation or turn.")
@_global_options
@click.argument("session_id", metavar="SESSION_ID")
@click.option(
    "--turn",
    type=int,
    default=None,
    help="Return only one turn by numeric index instead of a full conversation.",
)
@click.option(
    "--offset",
    default=0,
    show_default=True,
    type=int,
    help="Offset within the conversation when printing turns (applies only when --turn is not set).",
)
@click.option(
    "--limit",
    default=20,
    show_default=True,
    type=int,
    help="Max turns to print when reading a full conversation.",
)
@click.option(
    "--include-records",
    is_flag=True,
    help="Include low-level record objects in each turn output.",
)
@click.option(
    "--include-content/--no-include-content",
    default=True,
    show_default=True,
    help=(
        "Whether to print the message content for each turn. Use --no-include-content "
        "to suppress long payloads while keeping summary metadata."
    ),
)
@click.option(
    "--compact",
    is_flag=True,
    help="Use compact output for conversation blocks and turns.",
)
def read(
    session_id: str,
    turn: int | None,
    offset: int,
    limit: int,
    include_records: bool,
    include_content: bool,
    compact: bool,
    transcripts_dir: str,
    codex_dir: str,
    format_: str,
    no_cache: bool,
    cache_dir: str,
    quiet: bool,
) -> None:
    args = _namespace_for_command(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        format_=format_,
        no_cache=no_cache,
        cache_dir=cache_dir,
        quiet=quiet,
        session_id=session_id,
        turn=turn,
        offset=offset,
        limit=limit,
        include_records=include_records,
        include_content=include_content,
        compact=compact,
    )
    cmd_read(args)


@cli.command("status", help="Show index status.")
@_global_options
def status(
    transcripts_dir: str,
    codex_dir: str,
    format_: str,
    no_cache: bool,
    cache_dir: str,
    quiet: bool,
) -> None:
    args = _namespace_for_command(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        format_=format_,
        no_cache=no_cache,
        cache_dir=cache_dir,
        quiet=quiet,
    )
    cmd_status(args)


@cli.command("reindex", help="Rebuild the index from source files.")
@_global_options
def reindex(
    transcripts_dir: str,
    codex_dir: str,
    format_: str,
    no_cache: bool,
    cache_dir: str,
    quiet: bool,
) -> None:
    args = _namespace_for_command(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        format_=format_,
        no_cache=no_cache,
        cache_dir=cache_dir,
        quiet=quiet,
    )
    cmd_reindex(args)


@cli.command("extract", help="Extract structured tool call records.")
@_global_options
@click.option(
    "--tool",
    default="",
    help="Filter by tool name substring; empty means all tools.",
)
@click.option(
    "--category",
    default="",
    type=click.Choice(["", "builtin", "mcp", "bash", "agent"]),
    show_default=False,
    help=(
        "Filter by tool category. Options are builtin, mcp, bash, agent, or empty for all."
    ),
)
@click.option(
    "--project",
    default="",
    help="Filter by project name substring.",
)
@click.option(
    "--session",
    default="",
    help="Filter by exact session ID.",
)
@click.option(
    "--errors-only",
    is_flag=True,
    help="Only include tool call records where execution failed.",
)
@click.option(
    "--stats",
    is_flag=True,
    help="Instead of JSONL output, print summary statistics.",
)
@click.option(
    "--extract-limit",
    default=0,
    show_default=True,
    type=int,
    help="Maximum number of records to emit. Zero means unlimited.",
)
def extract(
    tool: str,
    category: str,
    project: str,
    session: str,
    errors_only: bool,
    stats: bool,
    extract_limit: int,
    transcripts_dir: str,
    codex_dir: str,
    format_: str,
    no_cache: bool,
    cache_dir: str,
    quiet: bool,
) -> None:
    args = _namespace_for_command(
        transcripts_dir=transcripts_dir,
        codex_dir=codex_dir,
        format_=format_,
        no_cache=no_cache,
        cache_dir=cache_dir,
        quiet=quiet,
        tool=tool,
        category=category,
        project=project,
        session=session,
        errors_only=errors_only,
        stats=stats,
        extract_limit=extract_limit,
    )
    cmd_extract(args)


def main() -> None:
    try:
        cli()
    finally:
        sys.stdout.flush()
        _flush_deferred_update()


if __name__ == "__main__":
    main()
