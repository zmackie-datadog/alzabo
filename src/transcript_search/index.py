from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from .parsers import (
    ParsedContent,
    parse_claude_record,
    parse_codex_function_call,
    parse_codex_function_output,
    parse_codex_message_content,
)

_EMBED_DIM = 512
_embed_model = None


def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[transcript-search {ts}] {msg}", file=sys.stderr)


def _compact(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v or v == 0}


def _dumps(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from model2vec import StaticModel

        _log("loading embedding model minishlab/potion-retrieval-32M...")
        _embed_model = StaticModel.from_pretrained("minishlab/potion-retrieval-32M")
        _log("embedding model loaded")
    return _embed_model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_embed_model()
    vecs = model.encode(texts)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.where(norms == 0, 1, norms)


def parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        normalized = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        return None


def normalize_project(raw: str) -> str:
    for sep in ("-DataDog-", "-datadog-"):
        pos = raw.find(sep)
        if pos != -1:
            return raw[pos + len(sep):]
    return raw


def extract_project(jsonl_file: Path) -> str:
    project = jsonl_file.parent.name
    if project == "subagents" and len(jsonl_file.parents) >= 4:
        return normalize_project(jsonl_file.parent.parent.parent.name)
    return normalize_project(project)


def _tool_result_only(content: Any) -> bool:
    if not isinstance(content, list) or not content:
        return False
    return all(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)


def _merge_signal_lists(target: dict[str, list[str]], parsed: ParsedContent) -> None:
    for attr in ("tools", "files", "commands", "errors"):
        current = target[attr]
        for value in getattr(parsed, attr):
            if value not in current:
                current.append(value)


@dataclass
class TurnSignals:
    tools: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @classmethod
    def from_map(cls, signal_map: dict[str, list[str]]) -> "TurnSignals":
        return cls(
            tools=list(signal_map.get("tools", [])),
            files=list(signal_map.get("files", [])),
            commands=list(signal_map.get("commands", [])),
            errors=list(signal_map.get("errors", [])),
        )

    def is_empty(self) -> bool:
        return not (self.tools or self.files or self.commands or self.errors)


@dataclass
class Turn:
    session_id: str
    turn_number: int
    timestamp: str
    project: str
    branch: str
    slug: str
    source: str
    user_content: str | list[Any] | None
    assistant_content: list[Any]
    tool_results: list[Any]
    summary: str
    signals: TurnSignals
    records: list[dict[str, Any]]
    search_text: str

    @property
    def tool_uses(self) -> list[str]:
        return self.signals.tools

    def as_dict(self, include_records: bool = False, include_content: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "turn_number": self.turn_number,
            "timestamp": self.timestamp,
            "project": self.project,
            "branch": self.branch,
            "slug": self.slug,
            "source": self.source,
            "summary": self.summary,
            "tool_uses": self.tool_uses,
        }
        if include_content:
            payload["user_content"] = self.user_content
            payload["assistant_content"] = self.assistant_content
            payload["tool_results"] = self.tool_results
        if include_records:
            payload["records"] = strip_signatures(self.records)
        if not self.signals.is_empty():
            payload["signals"] = _compact(
                {
                    "files": self.signals.files,
                    "commands": self.signals.commands,
                    "errors": self.signals.errors,
                }
            )
        return _compact(payload)


@dataclass
class Conversation:
    session_id: str
    project: str
    branch: str
    slug: str
    source: str = "claude"
    summary: str = ""
    first_timestamp: str = ""
    last_timestamp: str = ""
    turns: list[Turn] = field(default_factory=list)

    def as_metadata(self) -> dict[str, Any]:
        return _compact(
            {
                "session_id": self.session_id,
                "project": self.project,
                "branch": self.branch,
                "slug": self.slug,
                "source": self.source,
                "summary": self.summary,
                "first_timestamp": self.first_timestamp,
                "last_timestamp": self.last_timestamp,
                "turn_count": len(self.turns),
                "first_user_prompt": self.first_user_prompt(),
                "top_tools": self.top_tools(),
                "error_count": self.error_count(),
            }
        )

    def first_user_prompt(self) -> str:
        for turn in self.turns:
            content = turn.user_content
            if isinstance(content, str) and content.strip():
                return content.strip()[:200]
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
                        text = block["text"].strip()
                        if text:
                            return text[:200]
        return ""

    def top_tools(self, limit: int = 3) -> list[str]:
        counts: dict[str, int] = {}
        for turn in self.turns:
            for tool in turn.tool_uses:
                counts[tool] = counts.get(tool, 0) + 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [f"{name}({count})" for name, count in ranked[:limit]]

    def error_count(self) -> int:
        return sum(len(turn.signals.errors) for turn in self.turns)


@dataclass
class Index:
    turns: list[Turn] = field(default_factory=list)
    corpus: list[list[str]] = field(default_factory=list)
    conversations: dict[str, Conversation] = field(default_factory=dict)
    bm25: BM25Okapi | None = None
    embeddings: np.ndarray = field(default_factory=lambda: np.empty((0, _EMBED_DIM), dtype=np.float32))

    def build(self) -> None:
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)


@dataclass
class TurnSearchResult:
    turn: Turn
    score: float
    context: list[Turn] = field(default_factory=list)


@dataclass
class SessionSearchResult:
    conversation: Conversation
    best_score: float
    best_turn_number: int
    best_turn_summary: str
    matching_turns: int


@dataclass
class ConversationPage:
    items: list[Conversation]
    total: int
    offset: int
    end: int
    next_offset: int | None


@dataclass
class SearchResultSet:
    query: str
    mode: str
    effective_mode: str
    items: list[TurnSearchResult]


@dataclass
class SessionResultSet:
    query: str
    mode: str
    effective_mode: str
    items: list[SessionSearchResult]


@dataclass
class IndexStatus:
    transcripts_dir: str
    codex_dir: str
    watch_enabled: bool
    total_sessions: int
    claude_sessions: int
    codex_sessions: int
    total_turns: int
    embeddings_ready: bool
    last_reindex_at: str


def strip_signatures(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    import copy

    cleaned = copy.deepcopy(records)
    for rec in cleaned:
        msg = rec.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "thinking":
                block.pop("signature", None)
    return cleaned


def vector_search(query: str, embeddings: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    if embeddings.shape[0] == 0:
        return []
    q_vec = embed_texts([query])[0]
    scores = embeddings @ q_vec
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]


def rrf_fuse(
    bm25_ranked: list[tuple[int, float]],
    vec_ranked: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float]]:
    rrf: dict[int, float] = {}
    for rank, (idx, _score) in enumerate(bm25_ranked):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, (idx, _score) in enumerate(vec_ranked):
        rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(rrf.items(), key=lambda item: item[1], reverse=True)


def _turn_in_date_range(turn: Turn, start_date: str, end_date: str) -> bool:
    turn_dt = parse_timestamp(turn.timestamp)
    if turn_dt is None:
        return not start_date and not end_date
    if start_date:
        start_dt = parse_timestamp(start_date)
        if start_dt is None or turn_dt < start_dt:
            return False
    if end_date:
        end_dt = parse_timestamp(end_date)
        if end_dt is None or turn_dt > end_dt:
            return False
    return True


def _conversation_in_date_range(convo: Conversation, start_date: str, end_date: str) -> bool:
    first_dt = parse_timestamp(convo.first_timestamp)
    last_dt = parse_timestamp(convo.last_timestamp)
    if start_date:
        start_dt = parse_timestamp(start_date)
        if start_dt is None:
            return False
        if last_dt is not None and last_dt < start_dt:
            return False
    if end_date:
        end_dt = parse_timestamp(end_date)
        if end_dt is None:
            return False
        if first_dt is not None and first_dt > end_dt:
            return False
    return True


def _finalize_turn(convo: Conversation, current_turn: dict[str, Any], turn_number: int) -> Turn:
    search_parts = [part for part in current_turn["search_parts"] if part]
    text = "\n".join(search_parts).strip()
    signals = TurnSignals.from_map(current_turn["signals"])
    return Turn(
        session_id=convo.session_id,
        turn_number=turn_number,
        timestamp=str(current_turn.get("timestamp", "")),
        project=convo.project,
        branch=convo.branch,
        slug=convo.slug,
        source=convo.source,
        user_content=current_turn.get("user_content"),
        assistant_content=list(current_turn.get("assistant_content", [])),
        tool_results=list(current_turn.get("tool_results", [])),
        summary=text[:300],
        signals=signals,
        records=list(current_turn.get("records", [])),
        search_text=text,
    )


def build_codex_index(sessions_dir: Path) -> tuple[Index, int]:
    idx = Index()
    if not sessions_dir.exists():
        return idx, 0

    for jsonl_file in sessions_dir.rglob("*.jsonl"):
        try:
            records: list[dict[str, Any]] = []
            with open(jsonl_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            continue

        if not records:
            continue

        session_id = ""
        cwd = ""
        for record in records:
            if record.get("type") == "session_meta":
                payload = record.get("payload", {})
                session_id = str(payload.get("id", ""))
                cwd = str(payload.get("cwd", ""))
                break

        if not session_id:
            session_id = jsonl_file.stem

        convo = Conversation(
            session_id=f"codex:{session_id}",
            project=normalize_project(Path(cwd).name) if cwd else jsonl_file.stem,
            branch="",
            slug="",
            source="codex",
        )
        current_turn: dict[str, Any] | None = None

        for record in records:
            rtype = str(record.get("type", ""))
            payload = record.get("payload", {})
            payload_type = str(payload.get("type", ""))
            ts = str(record.get("timestamp", ""))

            if ts:
                if not convo.first_timestamp or ts < convo.first_timestamp:
                    convo.first_timestamp = ts
                if not convo.last_timestamp or ts > convo.last_timestamp:
                    convo.last_timestamp = ts

            if rtype == "event_msg" and payload_type == "user_message":
                if current_turn is not None:
                    convo.turns.append(_finalize_turn(convo, current_turn, len(convo.turns)))
                user_text = str(payload.get("message", "")).strip()
                if not convo.summary and user_text:
                    convo.summary = user_text[:200]
                current_turn = {
                    "timestamp": ts,
                    "user_content": user_text,
                    "assistant_content": [],
                    "tool_results": [],
                    "records": [record],
                    "search_parts": [user_text] if user_text else [],
                    "signals": {"tools": [], "files": [], "commands": [], "errors": []},
                }
                continue

            if rtype == "response_item" and payload_type == "message" and payload.get("role") == "assistant":
                parsed = parse_codex_message_content(payload.get("content"))
                if current_turn is None:
                    current_turn = {
                        "timestamp": ts,
                        "user_content": None,
                        "assistant_content": [],
                        "tool_results": [],
                        "records": [],
                        "search_parts": [],
                        "signals": {"tools": [], "files": [], "commands": [], "errors": []},
                    }
                current_turn["assistant_content"].append(payload.get("content"))
                current_turn["records"].append(record)
                if parsed.text:
                    current_turn["search_parts"].append(parsed.text)
                _merge_signal_lists(current_turn["signals"], parsed)
                if not current_turn.get("timestamp") and ts:
                    current_turn["timestamp"] = ts
                continue

            if rtype == "response_item" and payload_type == "function_call":
                parsed = parse_codex_function_call(payload)
                if current_turn is None:
                    current_turn = {
                        "timestamp": ts,
                        "user_content": None,
                        "assistant_content": [],
                        "tool_results": [],
                        "records": [],
                        "search_parts": [],
                        "signals": {"tools": [], "files": [], "commands": [], "errors": []},
                    }
                current_turn["records"].append(record)
                if parsed.text:
                    current_turn["search_parts"].append(parsed.text)
                _merge_signal_lists(current_turn["signals"], parsed)
                if not current_turn.get("timestamp") and ts:
                    current_turn["timestamp"] = ts
                continue

            if rtype == "response_item" and payload_type == "function_call_output":
                parsed = parse_codex_function_output(payload)
                if current_turn is None:
                    current_turn = {
                        "timestamp": ts,
                        "user_content": None,
                        "assistant_content": [],
                        "tool_results": [],
                        "records": [],
                        "search_parts": [],
                        "signals": {"tools": [], "files": [], "commands": [], "errors": []},
                    }
                current_turn["tool_results"].append(payload.get("output"))
                current_turn["records"].append(record)
                if parsed.text:
                    current_turn["search_parts"].append(parsed.text)
                _merge_signal_lists(current_turn["signals"], parsed)
                if not current_turn.get("timestamp") and ts:
                    current_turn["timestamp"] = ts
                continue

            if rtype == "response_item" and current_turn is not None:
                fallback = ParsedContent(text=_dumps(payload)[:1200])
                current_turn["records"].append(record)
                if fallback.text:
                    current_turn["search_parts"].append(fallback.text)

        if current_turn is not None:
            convo.turns.append(_finalize_turn(convo, current_turn, len(convo.turns)))

        if not convo.summary and convo.turns:
            convo.summary = convo.turns[0].summary[:200]

        if convo.turns:
            idx.conversations[convo.session_id] = convo
            for turn in convo.turns:
                idx.turns.append(turn)
                idx.corpus.append(turn.search_text.lower().split())

    return idx, len(idx.turns)


def build_claude_index(base_dir: Path) -> tuple[Index, int]:
    idx = Index()
    if not base_dir.exists():
        return idx, 0

    grouped: dict[str, list[tuple[dict[str, Any], str, str, str, str]]] = {}
    fallback_counter = 0

    for jsonl_file in base_dir.rglob("*.jsonl"):
        project = extract_project(jsonl_file)
        try:
            with open(jsonl_file, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    rtype = str(record.get("type", ""))
                    if rtype not in {"user", "assistant", "summary"}:
                        continue

                    sid = str(record.get("sessionId", ""))
                    if not sid:
                        sid = f"fallback:{jsonl_file.name}:{fallback_counter}"
                        fallback_counter += 1

                    grouped.setdefault(sid, []).append(
                        (
                            record,
                            project,
                            str(record.get("gitBranch", "")),
                            str(record.get("slug", "")),
                            str(record.get("timestamp", "")),
                        )
                    )
        except OSError:
            continue

    for session_id, rows in grouped.items():
        rows.sort(key=lambda row: row[4] or "")
        convo = Conversation(session_id=session_id, project="", branch="", slug="")
        current_turn: dict[str, Any] | None = None

        for record, project, branch, slug, ts in rows:
            rtype = str(record.get("type", ""))
            if project and not convo.project:
                convo.project = project
            if branch and not convo.branch:
                convo.branch = branch
            if slug and not convo.slug:
                convo.slug = slug
            if ts:
                if not convo.first_timestamp or ts < convo.first_timestamp:
                    convo.first_timestamp = ts
                if not convo.last_timestamp or ts > convo.last_timestamp:
                    convo.last_timestamp = ts

            if rtype == "summary":
                summary_text = record.get("summary")
                if isinstance(summary_text, str) and summary_text.strip() and not convo.summary:
                    convo.summary = summary_text.strip()
                continue

            if rtype == "user":
                parsed = parse_claude_record(record)
                message = record.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if _tool_result_only(content) and current_turn is not None:
                    current_turn["tool_results"].append(content)
                    current_turn["records"].append(record)
                    if parsed.text:
                        current_turn["search_parts"].append(parsed.text)
                    _merge_signal_lists(current_turn["signals"], parsed)
                    continue

                if current_turn is not None:
                    convo.turns.append(_finalize_turn(convo, current_turn, len(convo.turns)))

                if not convo.summary and parsed.text:
                    convo.summary = parsed.text[:200]
                current_turn = {
                    "timestamp": ts,
                    "user_content": content,
                    "assistant_content": [],
                    "tool_results": [],
                    "records": [record],
                    "search_parts": [parsed.text] if parsed.text else [],
                    "signals": {"tools": [], "files": [], "commands": [], "errors": []},
                }
                _merge_signal_lists(current_turn["signals"], parsed)
                continue

            if rtype == "assistant":
                parsed = parse_claude_record(record)
                message = record.get("message", {})
                content = message.get("content") if isinstance(message, dict) else None
                if current_turn is None:
                    current_turn = {
                        "timestamp": ts,
                        "user_content": None,
                        "assistant_content": [],
                        "tool_results": [],
                        "records": [],
                        "search_parts": [],
                        "signals": {"tools": [], "files": [], "commands": [], "errors": []},
                    }

                current_turn["assistant_content"].append(content)
                current_turn["records"].append(record)
                if parsed.text:
                    current_turn["search_parts"].append(parsed.text)
                _merge_signal_lists(current_turn["signals"], parsed)
                if not current_turn.get("timestamp") and ts:
                    current_turn["timestamp"] = ts

        if current_turn is not None:
            convo.turns.append(_finalize_turn(convo, current_turn, len(convo.turns)))

        if not convo.summary and convo.turns:
            convo.summary = convo.turns[0].summary[:200]

        idx.conversations[session_id] = convo
        for turn in convo.turns:
            idx.turns.append(turn)
            idx.corpus.append(turn.search_text.lower().split())

    idx.build()
    return idx, len(idx.turns)


class TranscriptIndexManager:
    def __init__(self) -> None:
        self._index_lock = threading.RLock()
        self._reindex_lock = threading.Lock()
        self._index_ready = threading.Event()
        self._index = Index()
        self._transcripts_dir = Path.home() / ".claude" / "projects"
        self._codex_sessions_dir = Path.home() / ".codex" / "sessions"
        self._watch_enabled = True
        self._last_reindex_at = ""

    def configure(self, transcripts_dir: Path, codex_dir: Path, watch_enabled: bool) -> None:
        self._transcripts_dir = transcripts_dir
        self._codex_sessions_dir = codex_dir
        self._watch_enabled = watch_enabled

    def ensure_index(self) -> None:
        self._index_ready.wait()

    def reindex(self) -> int:
        if not self._reindex_lock.acquire(blocking=False):
            _log("reindex already in progress, skipping")
            return 0
        try:
            return self._reindex_inner()
        finally:
            self._reindex_lock.release()

    def _reindex_inner(self) -> int:
        claude_idx, claude_count = build_claude_index(self._transcripts_dir)
        codex_idx, codex_count = build_codex_index(self._codex_sessions_dir)

        claude_idx.turns.extend(codex_idx.turns)
        claude_idx.corpus.extend(codex_idx.corpus)
        claude_idx.conversations.update(codex_idx.conversations)
        claude_idx.build()

        total = claude_count + codex_count
        _log(f"indexed {total} turns (claude={claude_count}, codex={codex_count})")

        if claude_idx.turns:
            t0 = time.monotonic()
            texts = [turn.search_text[:2000] for turn in claude_idx.turns]
            claude_idx.embeddings = embed_texts(texts)
            elapsed = time.monotonic() - t0
            _log(f"embedded {len(texts)} turns in {elapsed:.1f}s")

        with self._index_lock:
            self._index = claude_idx
            self._last_reindex_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self._index_ready.set()
        return total

    def _snapshot(self) -> tuple[Index, Path, Path, bool, str]:
        with self._index_lock:
            return (
                self._index,
                self._transcripts_dir,
                self._codex_sessions_dir,
                self._watch_enabled,
                self._last_reindex_at,
            )

    def search_conversations(
        self,
        query: str,
        limit: int = 10,
        session_id: str = "",
        source: str = "",
        project: str = "",
        start_date: str = "",
        end_date: str = "",
        mode: str = "hybrid",
        context_window: int = 0,
    ) -> SearchResultSet:
        self.ensure_index()
        index, *_ = self._snapshot()

        if not index.turns:
            return SearchResultSet(query=query, mode=mode, effective_mode=mode, items=[])

        project_lower = project.lower() if project else ""
        vectors_ready = index.embeddings.shape[0] > 0
        effective_mode = mode
        if mode in {"vector", "hybrid"} and not vectors_ready:
            effective_mode = "bm25"

        def passes(turn: Turn) -> bool:
            if session_id and turn.session_id != session_id:
                return False
            if source and turn.source != source:
                return False
            if project_lower and project_lower not in turn.project.lower():
                return False
            if (start_date or end_date) and not _turn_in_date_range(turn, start_date, end_date):
                return False
            return True

        cap = max(limit, 1)
        pool_size = cap * 5
        bm25_ranked: list[tuple[int, float]] = []
        if effective_mode in {"bm25", "hybrid"} and index.bm25 is not None:
            scores = index.bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            for idx, score in ranked:
                if score <= 0:
                    break
                if passes(index.turns[idx]):
                    bm25_ranked.append((idx, float(score)))
                if len(bm25_ranked) >= pool_size:
                    break

        vec_ranked: list[tuple[int, float]] = []
        if effective_mode in {"vector", "hybrid"} and vectors_ready:
            for idx, score in vector_search(query, index.embeddings, pool_size * 2):
                if passes(index.turns[idx]):
                    vec_ranked.append((idx, score))
                if len(vec_ranked) >= pool_size:
                    break

        if effective_mode == "hybrid":
            fused = rrf_fuse(bm25_ranked, vec_ranked)
        elif effective_mode == "bm25":
            fused = bm25_ranked
        else:
            fused = vec_ranked

        items: list[TurnSearchResult] = []
        for idx, score in fused:
            turn = index.turns[idx]
            context: list[Turn] = []
            if context_window > 0:
                convo = index.conversations.get(turn.session_id)
                if convo is not None:
                    start = max(turn.turn_number - context_window, 0)
                    end = min(turn.turn_number + context_window + 1, len(convo.turns))
                    context = convo.turns[start:end]
            items.append(TurnSearchResult(turn=turn, score=score, context=context))
            if len(items) >= cap:
                break
        return SearchResultSet(query=query, mode=mode, effective_mode=effective_mode, items=items)

    def search_sessions(
        self,
        query: str,
        limit: int = 5,
        source: str = "",
        project: str = "",
        start_date: str = "",
        end_date: str = "",
        mode: str = "hybrid",
    ) -> SessionResultSet:
        self.ensure_index()
        index, *_ = self._snapshot()

        if not index.turns:
            return SessionResultSet(query=query, mode=mode, effective_mode=mode, items=[])

        project_lower = project.lower() if project else ""
        vectors_ready = index.embeddings.shape[0] > 0
        effective_mode = mode
        if mode in {"vector", "hybrid"} and not vectors_ready:
            effective_mode = "bm25"

        def passes(turn: Turn) -> bool:
            if source and turn.source != source:
                return False
            if project_lower and project_lower not in turn.project.lower():
                return False
            if (start_date or end_date) and not _turn_in_date_range(turn, start_date, end_date):
                return False
            return True

        pool_size = max(limit, 5) * 20
        bm25_ranked: list[tuple[int, float]] = []
        if effective_mode in {"bm25", "hybrid"} and index.bm25 is not None:
            scores = index.bm25.get_scores(query.lower().split())
            ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
            for idx, score in ranked:
                if score <= 0:
                    break
                if passes(index.turns[idx]):
                    bm25_ranked.append((idx, float(score)))
                if len(bm25_ranked) >= pool_size:
                    break

        vec_ranked: list[tuple[int, float]] = []
        if effective_mode in {"vector", "hybrid"} and vectors_ready:
            for idx, score in vector_search(query, index.embeddings, pool_size * 2):
                if passes(index.turns[idx]):
                    vec_ranked.append((idx, score))
                if len(vec_ranked) >= pool_size:
                    break

        if effective_mode == "hybrid":
            fused = rrf_fuse(bm25_ranked, vec_ranked)
        elif effective_mode == "bm25":
            fused = bm25_ranked
        else:
            fused = vec_ranked

        session_scores: dict[str, dict[str, Any]] = {}
        for idx, score in fused:
            turn = index.turns[idx]
            existing = session_scores.get(turn.session_id)
            if existing is None or score > existing["best_score"]:
                session_scores[turn.session_id] = {
                    "best_score": float(score),
                    "best_turn_number": turn.turn_number,
                    "best_turn_summary": turn.summary,
                    "matching_turns": (existing["matching_turns"] if existing else 0) + 1,
                }
            else:
                existing["matching_turns"] += 1

        ranked_sessions = sorted(session_scores.items(), key=lambda item: item[1]["best_score"], reverse=True)
        items: list[SessionSearchResult] = []
        for session_id, info in ranked_sessions[: max(limit, 1)]:
            convo = index.conversations.get(session_id)
            if convo is None:
                continue
            items.append(
                SessionSearchResult(
                    conversation=convo,
                    best_score=float(info["best_score"]),
                    best_turn_number=int(info["best_turn_number"]),
                    best_turn_summary=str(info["best_turn_summary"]),
                    matching_turns=int(info["matching_turns"]),
                )
            )
        return SessionResultSet(query=query, mode=mode, effective_mode=effective_mode, items=items)

    def list_conversations(
        self,
        source: str = "",
        project: str = "",
        start_date: str = "",
        end_date: str = "",
        limit: int = 20,
        offset: int = 0,
    ) -> ConversationPage:
        self.ensure_index()
        index, *_ = self._snapshot()
        conversations = list(index.conversations.values())
        if source:
            conversations = [convo for convo in conversations if convo.source == source]
        if project:
            project_lower = project.lower()
            conversations = [convo for convo in conversations if project_lower in convo.project.lower()]
        if start_date or end_date:
            conversations = [
                convo for convo in conversations if _conversation_in_date_range(convo, start_date, end_date)
            ]

        conversations.sort(key=lambda convo: convo.last_timestamp, reverse=True)
        total = len(conversations)
        safe_offset = max(offset, 0)
        safe_limit = max(limit, 1)
        end = min(safe_offset + safe_limit, total)
        page = conversations[safe_offset:end]
        next_offset = end if end < total else None
        return ConversationPage(items=page, total=total, offset=safe_offset, end=end, next_offset=next_offset)

    def get_turn(self, session_id: str, turn_number: int) -> Turn:
        self.ensure_index()
        index, *_ = self._snapshot()
        convo = index.conversations.get(session_id)
        if convo is None:
            raise KeyError(f"session not found: {session_id}")
        if turn_number < 0 or turn_number >= len(convo.turns):
            raise IndexError(f"turn out of range: {turn_number}")
        return convo.turns[turn_number]

    def get_conversation(self, session_id: str) -> Conversation:
        self.ensure_index()
        index, *_ = self._snapshot()
        convo = index.conversations.get(session_id)
        if convo is None:
            raise KeyError(f"session not found: {session_id}")
        return convo

    def get_index_status(self) -> IndexStatus:
        self.ensure_index()
        index, transcripts_dir, codex_dir, watch_enabled, last_reindex_at = self._snapshot()
        claude_sessions = sum(1 for convo in index.conversations.values() if convo.source == "claude")
        codex_sessions = sum(1 for convo in index.conversations.values() if convo.source == "codex")
        return IndexStatus(
            transcripts_dir=str(transcripts_dir),
            codex_dir=str(codex_dir),
            watch_enabled=watch_enabled,
            total_sessions=len(index.conversations),
            claude_sessions=claude_sessions,
            codex_sessions=codex_sessions,
            total_turns=len(index.turns),
            embeddings_ready=index.embeddings.shape[0] > 0,
            last_reindex_at=last_reindex_at,
        )
