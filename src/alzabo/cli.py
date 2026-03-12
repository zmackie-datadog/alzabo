from __future__ import annotations

import argparse
import threading
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .index import TranscriptIndexManager, _log
from .server import create_mcp_server


class TranscriptChangeHandler(FileSystemEventHandler):
    def __init__(self, manager: TranscriptIndexManager, debounce_seconds: float) -> None:
        self._manager = manager
        self._debounce_seconds = debounce_seconds
        self._debounce_lock = threading.Lock()
        self._debounce_timer: threading.Timer | None = None

    def _is_jsonl_event(self, event: FileSystemEvent) -> bool:
        if event.is_directory:
            return False
        paths = [getattr(event, "src_path", "")]
        dest_path = getattr(event, "dest_path", "")
        if dest_path:
            paths.append(dest_path)
        return any(path.endswith(".jsonl") for path in paths)

    def _schedule_reindex(self) -> None:
        with self._debounce_lock:
            if self._debounce_timer is not None:
                self._debounce_timer.cancel()
            self._debounce_timer = threading.Timer(self._debounce_seconds, self._manager.reindex)
            self._debounce_timer.daemon = True
            self._debounce_timer.start()

    def on_created(self, event: FileSystemEvent) -> None:
        if self._is_jsonl_event(event):
            self._schedule_reindex()

    def on_modified(self, event: FileSystemEvent) -> None:
        if self._is_jsonl_event(event):
            self._schedule_reindex()

    def on_deleted(self, event: FileSystemEvent) -> None:
        if self._is_jsonl_event(event):
            self._schedule_reindex()

    def on_moved(self, event: FileSystemEvent) -> None:
        if self._is_jsonl_event(event):
            self._schedule_reindex()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search Claude Code and Codex transcript JSONL files with hybrid search."
    )
    parser.add_argument(
        "--transcripts-dir",
        default=str(Path.home() / ".claude" / "projects"),
        help="Root directory to recursively scan for Claude .jsonl transcripts.",
    )
    parser.add_argument(
        "--codex-dir",
        default=str(Path.home() / ".codex" / "sessions"),
        help="Root directory to recursively scan for Codex .jsonl sessions.",
    )
    parser.add_argument(
        "--watch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Watch transcript files and auto-reindex on changes (default: true).",
    )
    parser.add_argument(
        "--debounce-seconds",
        type=float,
        default=2.0,
        help="Reindex debounce delay when watch mode is enabled.",
    )
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
