"""Output formatting dispatch: text, json, jsonl."""

from __future__ import annotations

import json
from typing import Any

from .index import (
    Conversation,
    ConversationPage,
    IndexStatus,
    SearchResultSet,
    SessionResultSet,
    Turn,
)
from .render import (
    render_conversation,
    render_index_status,
    render_list_conversations,
    render_search_conversations,
    render_search_sessions,
    render_turn,
)


def _json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"))


def _jsonl_items(items: list[dict[str, Any]]) -> str:
    return "\n".join(json.dumps(item, separators=(",", ":")) for item in items)


def format_search_results(result: SearchResultSet, fmt: str) -> str:
    if fmt == "json":
        return _json(result.as_dict())
    if fmt == "jsonl":
        return _jsonl_items([i.as_dict() for i in result.items])
    return render_search_conversations(result)


def format_session_results(result: SessionResultSet, fmt: str) -> str:
    if fmt == "json":
        return _json(result.as_dict())
    if fmt == "jsonl":
        return _jsonl_items([i.as_dict() for i in result.items])
    return render_search_sessions(result)


def format_conversation_page(page: ConversationPage, fmt: str) -> str:
    if fmt == "json":
        return _json(page.as_dict())
    if fmt == "jsonl":
        return _jsonl_items([c.as_metadata() for c in page.items])
    return render_list_conversations(page)


def format_conversation(
    convo: Conversation,
    fmt: str,
    offset: int = 0,
    limit: int = 20,
    include_records: bool = False,
    include_content: bool = True,
    compact: bool = False,
) -> str:
    if fmt == "json":
        total = len(convo.turns)
        safe_offset = max(offset, 0)
        safe_limit = max(limit, 1)
        end = min(safe_offset + safe_limit, total)
        turns_slice = convo.turns[safe_offset:end]
        return _json(
            {
                "session_id": convo.session_id,
                "metadata": convo.as_metadata(),
                "offset": safe_offset,
                "limit": safe_limit,
                "turns": [
                    t.as_dict(include_records=include_records, include_content=include_content) for t in turns_slice
                ],
            }
        )
    if fmt == "jsonl":
        total = len(convo.turns)
        safe_offset = max(offset, 0)
        safe_limit = max(limit, 1)
        end = min(safe_offset + safe_limit, total)
        turns_slice = convo.turns[safe_offset:end]
        return _jsonl_items(
            [t.as_dict(include_records=include_records, include_content=include_content) for t in turns_slice]
        )
    return render_conversation(
        convo,
        offset=offset,
        limit=limit,
        include_records=include_records,
        include_content=include_content,
        compact=compact,
    )


def format_turn(
    turn: Turn,
    fmt: str,
    include_records: bool = False,
    include_content: bool = True,
) -> str:
    if fmt in ("json", "jsonl"):
        return _json(turn.as_dict(include_records=include_records, include_content=include_content))
    return render_turn(turn, include_records=include_records, include_content=include_content)


def format_index_status(status: IndexStatus, fmt: str) -> str:
    if fmt in ("json", "jsonl"):
        return _jsonl_items([status.as_dict()]) if fmt == "jsonl" else _json(status.as_dict())
    return render_index_status(status)
