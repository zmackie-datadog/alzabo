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
    TurnSignals,
    parse_timestamp,
    strip_signatures,
)
from .parsers import fmt_claude_content, fmt_claude_content_list, parse_claude_content


def _ts_short(ts: str) -> str:
    if not ts:
        return ""
    dt = parse_timestamp(ts)
    if dt is None:
        return ts[:16] + "Z" if len(ts) > 16 else ts
    return dt.strftime("%Y-%m-%dT%H:%MZ")


def _ts_date(ts: str) -> str:
    if not ts:
        return ""
    dt = parse_timestamp(ts)
    if dt is None:
        return ts[:10] if len(ts) >= 10 else ts
    return dt.strftime("%Y-%m-%d")


def _compact_json(value: Any, max_chars: int = 4000) -> str:
    if isinstance(value, str):
        text = value.strip()
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        except TypeError:
            text = str(value)
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def _render_tool_results(items: list[Any]) -> str:
    rendered = [_compact_json(item, max_chars=4000) for item in items]
    return "\n".join(item for item in rendered if item)


def _user_text(turn: Turn) -> str:
    if isinstance(turn.user_content, str):
        return turn.user_content.strip()
    if isinstance(turn.user_content, list):
        return parse_claude_content(turn.user_content).text
    return ""


def _signal_line(signals: TurnSignals) -> str:
    if signals.is_empty():
        return ""
    parts: list[str] = []
    if signals.tools:
        parts.append(f"tools={', '.join(signals.tools[:4])}")
    if signals.files:
        parts.append(f"files={', '.join(signals.files[:3])}")
    if signals.commands:
        parts.append(f"commands={', '.join(signals.commands[:2])}")
    if signals.errors:
        parts.append(f"errors={'; '.join(signals.errors[:2])}")
    return "signals: " + " | ".join(parts)


def _render_context(turns: list[Turn], matched_turn: int) -> list[str]:
    lines = ["  context:"]
    for turn in turns:
        prompt = _user_text(turn)[:160] or turn.summary[:160]
        marker = "*" if turn.turn_number == matched_turn else "-"
        tools_part = f" | tools: {', '.join(turn.tool_uses)}" if turn.tool_uses else ""
        lines.append(f"    {marker} turn {turn.turn_number} | {_ts_short(turn.timestamp)}{tools_part}")
        lines.append(f"      {prompt}")
    return lines


def render_search_conversations(result_set: SearchResultSet) -> str:
    header = f"query: {result_set.query} | mode: {result_set.mode}"
    if result_set.effective_mode != result_set.mode:
        header += f" (using {result_set.effective_mode})"
    header += f" | {len(result_set.items)} results"
    lines = [header]
    for i, item in enumerate(result_set.items, 1):
        turn = item.turn
        lines.append("")
        lines.append(
            f"#{i} [{item.score:.4f}] {turn.project} ({turn.source}) session={turn.session_id} turn={turn.turn_number}"
        )
        lines.append(f"  {_ts_short(turn.timestamp)} | {turn.summary[:150]}")
        signal_line = _signal_line(turn.signals)
        if signal_line:
            lines.append(f"  {signal_line}")
        if item.context:
            lines.extend(_render_context(item.context, turn.turn_number))
    return "\n".join(lines)


def render_search_sessions(result_set: SessionResultSet) -> str:
    header = f"query: {result_set.query} | mode: {result_set.mode}"
    if result_set.effective_mode != result_set.mode:
        header += f" (using {result_set.effective_mode})"
    header += f" | {len(result_set.items)} sessions"
    lines = [header]
    for i, item in enumerate(result_set.items, 1):
        convo = item.conversation
        lines.append("")
        lines.append(
            f"#{i} [{item.best_score:.4f}] {convo.project} ({convo.source}) {len(convo.turns)} turns, {item.matching_turns} matching"
        )
        lines.append(f"  session: {convo.session_id}")
        lines.append(f"  {_ts_date(convo.first_timestamp)} -> {_ts_date(convo.last_timestamp)}")
        if convo.top_tools():
            lines.append(f"  top tools: {', '.join(convo.top_tools())}")
        if convo.error_count():
            lines.append(f"  error count: {convo.error_count()}")
        lines.append(f"  {convo.summary[:150]}")
        lines.append(f"  best turn: {item.best_turn_number} | {item.best_turn_summary[:150]}")
    return "\n".join(lines)


def render_list_conversations(page: ConversationPage) -> str:
    header = f"{page.total} sessions (showing {page.offset + 1}-{page.end}"
    if page.next_offset is not None:
        header += f", next_offset={page.next_offset})"
    else:
        header += ")"
    lines = [header]
    for convo in page.items:
        branch_part = f" | {convo.branch}" if convo.branch else ""
        lines.append("")
        lines.append(f"{convo.project} ({convo.source}) {len(convo.turns)} turns{branch_part}")
        lines.append(f"  {convo.session_id}")
        lines.append(f"  {_ts_date(convo.first_timestamp)} -> {_ts_date(convo.last_timestamp)} | {convo.summary[:150]}")
        first_ask = convo.first_user_prompt()
        if first_ask:
            lines.append(f"  first ask: {first_ask[:120]}")
        if convo.top_tools():
            lines.append(f"  top tools: {', '.join(convo.top_tools())}")
        if convo.error_count():
            lines.append(f"  error count: {convo.error_count()}")
    return "\n".join(lines)


def render_turn(turn: Turn, include_records: bool = False, include_content: bool = True) -> str:
    lines = [
        f"session: {turn.session_id} | turn {turn.turn_number} | {turn.project} ({turn.source})",
        turn.timestamp,
    ]
    if turn.tool_uses:
        lines.append(f"tools: {', '.join(turn.tool_uses)}")
    if turn.summary:
        lines.append(f"summary: {turn.summary[:300]}")
    signal_line = _signal_line(turn.signals)
    if signal_line:
        lines.append(signal_line)

    if include_content:
        lines.append("")
        lines.append("--- user ---")
        if turn.source == "claude":
            lines.append(fmt_claude_content(turn.user_content))
        else:
            lines.append(_user_text(turn))
        lines.append("")
        lines.append("--- assistant ---")
        if turn.source == "claude":
            lines.append(fmt_claude_content_list(turn.assistant_content))
        else:
            lines.append("\n".join(_compact_json(item, max_chars=4000) for item in turn.assistant_content if item))
        if turn.tool_results:
            lines.append("")
            lines.append("--- tool results ---")
            lines.append(_render_tool_results(turn.tool_results))

    if include_records:
        lines.append("")
        lines.append("--- records ---")
        lines.append(json.dumps(strip_signatures(turn.records), separators=(",", ":")))
    return "\n".join(lines)


def render_conversation(
    convo: Conversation,
    offset: int = 0,
    limit: int = 20,
    include_records: bool = False,
    include_content: bool = True,
    compact: bool = False,
) -> str:
    total = len(convo.turns)
    safe_offset = max(offset, 0)
    safe_limit = max(limit, 1)
    end = min(safe_offset + safe_limit, total)
    next_offset = end if end < total else None
    lines = [
        f"session: {convo.session_id} | {convo.project} ({convo.source}) | {total} turns",
        f"{_ts_date(convo.first_timestamp)} -> {_ts_date(convo.last_timestamp)}",
    ]
    showing = f"showing turns {safe_offset}-{end - 1}"
    if next_offset is not None:
        showing += f" (next_offset={next_offset})"
    lines.append(showing)

    for turn in convo.turns[safe_offset:end]:
        lines.append("")
        tools_part = f" | tools: {', '.join(turn.tool_uses)}" if turn.tool_uses else ""
        if compact:
            prompt = _user_text(turn)[:200] or turn.summary[:200]
            lines.append(f"=== turn {turn.turn_number} | {_ts_short(turn.timestamp)}{tools_part} ===")
            signal_line = _signal_line(turn.signals)
            if signal_line:
                lines.append(signal_line)
            lines.append(f"[user] {prompt}")
            continue

        lines.append(f"=== turn {turn.turn_number} | {turn.timestamp} ===")
        if turn.tool_uses:
            lines.append(f"tools: {', '.join(turn.tool_uses)}")
        if turn.summary:
            lines.append(f"summary: {turn.summary[:300]}")
        signal_line = _signal_line(turn.signals)
        if signal_line:
            lines.append(signal_line)

        if include_content:
            lines.append("")
            lines.append("--- user ---")
            if turn.source == "claude":
                lines.append(fmt_claude_content(turn.user_content))
            else:
                lines.append(_user_text(turn))
            lines.append("")
            lines.append("--- assistant ---")
            if turn.source == "claude":
                lines.append(fmt_claude_content_list(turn.assistant_content))
            else:
                lines.append("\n".join(_compact_json(item, max_chars=4000) for item in turn.assistant_content if item))
            if turn.tool_results:
                lines.append("")
                lines.append("--- tool results ---")
                lines.append(_render_tool_results(turn.tool_results))

        if include_records:
            lines.append("")
            lines.append("--- records ---")
            lines.append(json.dumps(strip_signatures(turn.records), separators=(",", ":")))
    return "\n".join(lines)


def render_index_status(status: IndexStatus) -> str:
    lines = [
        "alzabo status",
        f"claude dir: {status.transcripts_dir}",
        f"codex dir: {status.codex_dir}",
        f"watch enabled: {status.watch_enabled}",
        f"sessions: {status.total_sessions} (claude={status.claude_sessions}, codex={status.codex_sessions})",
        f"turns: {status.total_turns}",
        f"embeddings ready: {status.embeddings_ready}",
        f"last reindex: {status.last_reindex_at or 'never'}",
    ]
    return "\n".join(lines)
