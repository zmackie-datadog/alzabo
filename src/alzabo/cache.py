"""Disk cache for the transcript index. Avoids re-parsing + re-embedding on startup."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .index import (
    Conversation,
    Index,
    Turn,
    TurnSignals,
    _EMBED_DIM,
)

CACHE_VERSION = 2
CACHE_DIR = Path.home() / ".cache" / "alzabo"
_LOG_ENABLED = True


def set_log_enabled(enabled: bool) -> None:
    global _LOG_ENABLED
    _LOG_ENABLED = bool(enabled)


def get_log_enabled() -> bool:
    return _LOG_ENABLED


def get_cache_dir() -> Path:
    return CACHE_DIR


def set_cache_dir(path: str | Path) -> None:
    global CACHE_DIR
    CACHE_DIR = Path(path).expanduser().resolve()


def _log(msg: str) -> None:
    if not _LOG_ENABLED:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[alzabo {ts}] {msg}", file=sys.stderr)


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "mtime": round(stat.st_mtime, 6),
        "size": stat.st_size,
    }


def _normalize_manifest_entry(value: Any) -> tuple[float, int | None] | None:
    if isinstance(value, dict):
        mtime = value.get("mtime")
        size = value.get("size")
        if not isinstance(mtime, (int, float)):
            return None
        if size is None:
            return (float(mtime), None)
        if not isinstance(size, (int, float)):
            return None
        return (float(mtime), int(size))
    if isinstance(value, (int, float)):
        return (float(value), None)
    return None


def collect_source_files(transcripts_dir: Path, codex_dir: Path) -> dict[str, dict[str, Any]]:
    """Collect file metadata for all transcript JSONL files."""
    files: dict[str, dict[str, Any]] = {}
    for d in (transcripts_dir, codex_dir):
        if not d.exists():
            continue
        for f in d.rglob("*.jsonl"):
            files[str(f.resolve())] = _file_signature(f)
    return files


def changed_source_files(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> set[str]:
    """Return files added, removed, or changed by mtime/size."""
    previous_norm: dict[str, tuple[float, int | None]] = {}
    for path, value in previous.items():
        try:
            previous_norm[str(Path(path).resolve())] = _normalize_manifest_entry(value)
        except OSError:
            continue

    current_norm = {str(Path(path).resolve()): _normalize_manifest_entry(value) for path, value in current.items()}

    changed: set[str] = set()
    previous_paths = set(previous_norm)
    current_paths = set(current_norm)
    for path in previous_paths ^ current_paths:
        changed.add(path)

    for path in previous_paths & current_paths:
        previous_sig = previous_norm[path]
        current_sig = current_norm[path]
        if previous_sig is None or current_sig is None:
            changed.add(path)
            continue
        previous_mtime, previous_size = previous_sig
        current_mtime, current_size = current_sig
        if previous_size is not None and current_size is not None and previous_size != current_size:
            changed.add(path)
            continue
        if abs(previous_mtime - current_mtime) > 0.01:
            changed.add(path)
    return changed


def partition_changed_files_by_stability(
    changed_files: set[str],
    current_files: dict[str, Any],
    *,
    debounce_seconds: float,
    now: float | None = None,
) -> tuple[set[str], set[str]]:
    """Split changed files into settled vs unstable based on recent mtime.

    Returns (settled, unstable).
    """
    if not changed_files or debounce_seconds <= 0:
        return set(changed_files), set()

    settled: set[str] = set()
    unstable: set[str] = set()
    current_time = now if now is not None else time.time()

    for path in changed_files:
        if path not in current_files:
            settled.add(path)
            continue

        entry = current_files[path]
        if not isinstance(entry, dict):
            settled.add(path)
            continue
        mtime = entry.get("mtime")
        if mtime is None or not isinstance(mtime, (int, float)):
            settled.add(path)
            continue

        if current_time - float(mtime) >= debounce_seconds:
            settled.add(path)
        else:
            unstable.add(path)

    return settled, unstable


def load_cache_bundle() -> tuple[Index, dict[str, Any]] | None:
    """Load cache index and manifest if both files are valid."""
    try:
        manifest_path = CACHE_DIR / "manifest.json"
        turns_path = CACHE_DIR / "turns.json"
        embeddings_path = CACHE_DIR / "embeddings.npy"
        if not manifest_path.exists() or not turns_path.exists() or not embeddings_path.exists():
            return None

        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") != CACHE_VERSION:
            return None

        data = json.loads(turns_path.read_text())
        index = _deserialize_index(data)

        embeddings = np.load(str(embeddings_path))
        if embeddings.shape[0] == len(index.turns) and (embeddings.ndim == 1 or embeddings.shape[1] == _EMBED_DIM):
            index.embeddings = embeddings
        elif embeddings.shape[0] > 0:
            _log(f"cache embeddings shape mismatch: {embeddings.shape} vs {len(index.turns)} turns, skipping")

        index.build()
        _log(f"cache loaded ({len(index.turns)} turns)")
        return index, manifest
    except Exception as e:
        _log(f"cache load failed: {e}")
        return None


def is_cache_fresh(transcripts_dir: Path, codex_dir: Path) -> bool:
    manifest_path = CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False

    if manifest.get("version") != CACHE_VERSION:
        return False
    if manifest.get("transcripts_dir") != str(transcripts_dir):
        return False
    if manifest.get("codex_dir") != str(codex_dir):
        return False

    cached_files = manifest.get("source_files", {})
    current_files = collect_source_files(transcripts_dir, codex_dir)
    return not changed_source_files(cached_files, current_files)


def save_cache(index: Index, transcripts_dir: Path, codex_dir: Path, *, reindex_at: str | None = None) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        # Serialize turns grouped by conversation
        data = _serialize_index(index)
        turns_path = CACHE_DIR / "turns.json"
        turns_path.write_text(json.dumps(data, separators=(",", ":")))

        # Save embeddings
        embeddings_path = CACHE_DIR / "embeddings.npy"
        np.save(str(embeddings_path), index.embeddings)

        # Write manifest last (commit marker)
        manifest = {
            "version": CACHE_VERSION,
            "transcripts_dir": str(transcripts_dir),
            "codex_dir": str(codex_dir),
            "source_files": collect_source_files(transcripts_dir, codex_dir),
            "turn_count": len(index.turns),
            "reindex_at": reindex_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        manifest_path = CACHE_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _log(f"cache saved ({len(index.turns)} turns)")
    except OSError as e:
        _log(f"cache save failed: {e}")


def load_cache() -> Index | None:
    bundle = load_cache_bundle()
    if bundle is None:
        return None
    return bundle[0]


def _serialize_index(index: Index) -> list[dict[str, Any]]:
    """Serialize conversations with nested turns."""
    convos: list[dict[str, Any]] = []
    for convo in index.conversations.values():
        turns_data = []
        for turn in convo.turns:
            turns_data.append({
                "turn_number": turn.turn_number,
                "timestamp": turn.timestamp,
                "user_content": turn.user_content,
                "assistant_content": turn.assistant_content,
                "tool_results": turn.tool_results,
                "summary": turn.summary,
                "search_text": turn.search_text,
                "records": turn.records,
                "source_file": turn.source_file,
                "signals": {
                    "tools": turn.signals.tools,
                    "files": turn.signals.files,
                    "commands": turn.signals.commands,
                    "errors": turn.signals.errors,
                },
            })
        convos.append({
            "session_id": convo.session_id,
            "project": convo.project,
            "branch": convo.branch,
            "slug": convo.slug,
            "source": convo.source,
            "summary": convo.summary,
            "first_timestamp": convo.first_timestamp,
            "last_timestamp": convo.last_timestamp,
            "turns": turns_data,
        })
    return convos


def _deserialize_index(data: list[dict[str, Any]]) -> Index:
    """Reconstruct Index from serialized data."""
    index = Index()
    for convo_data in data:
        convo = Conversation(
            session_id=convo_data["session_id"],
            project=convo_data.get("project", ""),
            branch=convo_data.get("branch", ""),
            slug=convo_data.get("slug", ""),
            source=convo_data.get("source", "claude"),
            summary=convo_data.get("summary", ""),
            first_timestamp=convo_data.get("first_timestamp", ""),
            last_timestamp=convo_data.get("last_timestamp", ""),
        )
        for turn_data in convo_data.get("turns", []):
            signals_data = turn_data.get("signals", {})
            signals = TurnSignals(
                tools=signals_data.get("tools", []),
                files=signals_data.get("files", []),
                commands=signals_data.get("commands", []),
                errors=signals_data.get("errors", []),
            )
            turn = Turn(
                session_id=convo.session_id,
                turn_number=turn_data["turn_number"],
                timestamp=turn_data.get("timestamp", ""),
                project=convo.project,
                branch=convo.branch,
                slug=convo.slug,
                source=convo.source,
                user_content=turn_data.get("user_content"),
                assistant_content=turn_data.get("assistant_content", []),
                tool_results=turn_data.get("tool_results", []),
                summary=turn_data.get("summary", ""),
                signals=signals,
                records=turn_data.get("records", []),
                search_text=turn_data.get("search_text", ""),
                source_file=turn_data.get("source_file", ""),
            )
            convo.turns.append(turn)
            index.turns.append(turn)
            index.corpus.append(turn.search_text.lower().split())

        index.conversations[convo.session_id] = convo
    return index
