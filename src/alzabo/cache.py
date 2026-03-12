"""Disk cache for the transcript index. Avoids re-parsing + re-embedding on startup."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .index import (
    Conversation,
    Index,
    Turn,
    TurnSignals,
    _EMBED_DIM,
)

CACHE_VERSION = 1
CACHE_DIR = Path.home() / ".cache" / "alzabo"


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[alzabo {ts}] {msg}", file=sys.stderr)


def _source_files(transcripts_dir: Path, codex_dir: Path) -> dict[str, float]:
    """Collect {path: mtime} for all JSONL files in both dirs."""
    files: dict[str, float] = {}
    for d in (transcripts_dir, codex_dir):
        if not d.exists():
            continue
        for f in d.rglob("*.jsonl"):
            files[str(f)] = f.stat().st_mtime
    return files


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
    current_files = _source_files(transcripts_dir, codex_dir)

    if set(cached_files.keys()) != set(current_files.keys()):
        return False

    for path, mtime in current_files.items():
        if abs(cached_files.get(path, 0) - mtime) > 0.01:
            return False

    return True


def save_cache(index: Index, transcripts_dir: Path, codex_dir: Path) -> None:
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
            "source_files": _source_files(transcripts_dir, codex_dir),
            "turn_count": len(index.turns),
        }
        manifest_path = CACHE_DIR / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        _log(f"cache saved ({len(index.turns)} turns)")
    except OSError as e:
        _log(f"cache save failed: {e}")


def load_cache() -> Index | None:
    try:
        turns_path = CACHE_DIR / "turns.json"
        embeddings_path = CACHE_DIR / "embeddings.npy"
        if not turns_path.exists() or not embeddings_path.exists():
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
        return index
    except Exception as e:
        _log(f"cache load failed: {e}")
        return None


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
            )
            convo.turns.append(turn)
            index.turns.append(turn)
            index.corpus.append(turn.search_text.lower().split())

        index.conversations[convo.session_id] = convo
    return index
