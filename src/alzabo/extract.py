"""Structured tool call extraction from Claude Code and Codex JSONL transcripts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .index import extract_project, normalize_project
from .parsers import _compact_json, _extract_errors


@dataclass
class ToolCallRecord:
    session_id: str
    project: str
    source: str  # "claude" | "codex"
    timestamp: str  # ISO 8601
    tool_name: str  # e.g. "Read", "mcp__alzabo__search_conversations"
    tool_category: str  # "builtin" | "mcp" | "bash" | "agent"
    tool_input: dict[str, Any]
    tool_output: str  # truncated result text
    is_error: bool
    error_snippet: str  # first error line if is_error
    duration_ms: int | None  # from ODP only (not available locally)
    turn_number: int
    tool_use_id: str  # correlation ID from JSONL

    def to_jsonl(self) -> str:
        d = asdict(self)
        return json.dumps(d, separators=(",", ":"), ensure_ascii=False)


def classify_tool(name: str) -> str:
    if name.startswith("mcp__"):
        return "mcp"
    if name.lower() in {"bash", "shell"}:
        return "bash"
    if name.lower() in {"agent"}:
        return "agent"
    return "builtin"


def _result_text(content: Any) -> str:
    """Extract text from a tool_result content field."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text", "")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
            elif isinstance(block, str):
                parts.append(block.strip())
        return "\n".join(parts)
    if isinstance(content, dict):
        text = content.get("text", "")
        if isinstance(text, str):
            return text.strip()
    return ""


def extract_from_claude_jsonl(
    base_dir: Path,
    *,
    tool_filter: str = "",
    category_filter: str = "",
    project_filter: str = "",
    session_filter: str = "",
    errors_only: bool = False,
) -> Iterator[ToolCallRecord]:
    """Extract tool call records from Claude Code JSONL transcripts.

    Walks JSONL files, groups records by sessionId, correlates tool_use blocks
    with tool_result blocks via tool_use_id.
    """
    if not base_dir.exists():
        return

    # Group records by session, preserving file-level project info
    grouped: dict[str, list[tuple[dict[str, Any], str]]] = {}

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
                    if rtype not in {"user", "assistant"}:
                        continue
                    sid = str(record.get("sessionId", ""))
                    if not sid:
                        continue
                    grouped.setdefault(sid, []).append((record, project))
        except OSError:
            continue

    for session_id, rows in grouped.items():
        if session_filter and session_id != session_filter:
            continue

        # Sort by timestamp
        rows.sort(key=lambda r: str(r[0].get("timestamp", "")))

        project = ""
        for _, proj in rows:
            if proj:
                project = proj
                break

        if project_filter and project_filter.lower() not in project.lower():
            continue

        # Pending tool_use entries: id -> (name, input, timestamp, turn_number)
        pending: dict[str, tuple[str, dict[str, Any], str, int]] = {}
        turn_number = 0

        for record, _ in rows:
            rtype = str(record.get("type", ""))
            ts = str(record.get("timestamp", ""))
            message = record.get("message", {})
            content = message.get("content") if isinstance(message, dict) else None

            if rtype == "user":
                # Check if this is a plain user message (new turn) vs tool_result
                if isinstance(content, list):
                    has_tool_result = any(
                        isinstance(b, dict) and b.get("type") == "tool_result"
                        for b in content
                    )
                    if not has_tool_result:
                        turn_number += 1

                    # Process tool_result blocks
                    for block in content:
                        if not isinstance(block, dict) or block.get("type") != "tool_result":
                            continue
                        tuid = str(block.get("tool_use_id", ""))
                        if tuid not in pending:
                            continue

                        tool_name, tool_input, tool_ts, tool_turn = pending.pop(tuid)
                        category = classify_tool(tool_name)
                        result_content = block.get("content", "")
                        output_text = _compact_json(_result_text(result_content), max_chars=2000)
                        errors = _extract_errors(result_content, max_items=1)
                        is_error = block.get("is_error", False) or bool(errors)
                        error_snippet = errors[0] if errors else ""

                        rec = ToolCallRecord(
                            session_id=session_id,
                            project=project,
                            source="claude",
                            timestamp=tool_ts or ts,
                            tool_name=tool_name,
                            tool_category=category,
                            tool_input=tool_input,
                            tool_output=output_text,
                            is_error=is_error,
                            error_snippet=error_snippet,
                            duration_ms=None,
                            turn_number=tool_turn,
                            tool_use_id=tuid,
                        )

                        if _passes_filters(rec, tool_filter, category_filter, errors_only):
                            yield rec
                elif isinstance(content, str):
                    turn_number += 1

            elif rtype == "assistant":
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict) or block.get("type") != "tool_use":
                        continue
                    tuid = str(block.get("id", ""))
                    name = str(block.get("name", ""))
                    inp = block.get("input", {})
                    if not isinstance(inp, dict):
                        inp = {}
                    if tuid and name:
                        pending[tuid] = (name, inp, ts, turn_number)

        # Yield any unmatched tool_use entries (no result received)
        for tuid, (tool_name, tool_input, tool_ts, tool_turn) in pending.items():
            category = classify_tool(tool_name)
            rec = ToolCallRecord(
                session_id=session_id,
                project=project,
                source="claude",
                timestamp=tool_ts,
                tool_name=tool_name,
                tool_category=category,
                tool_input=tool_input,
                tool_output="",
                is_error=False,
                error_snippet="",
                duration_ms=None,
                turn_number=tool_turn,
                tool_use_id=tuid,
            )
            if _passes_filters(rec, tool_filter, category_filter, errors_only):
                yield rec


def extract_from_codex_jsonl(
    sessions_dir: Path,
    *,
    tool_filter: str = "",
    category_filter: str = "",
    project_filter: str = "",
    session_filter: str = "",
    errors_only: bool = False,
) -> Iterator[ToolCallRecord]:
    """Extract tool call records from Codex JSONL sessions.

    Matches function_call records with their function_call_output via call_id.
    """
    if not sessions_dir.exists():
        return

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

        # Extract session metadata
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

        full_session_id = f"codex:{session_id}"
        if session_filter and full_session_id != session_filter:
            continue

        project = normalize_project(Path(cwd).name) if cwd else jsonl_file.stem
        if project_filter and project_filter.lower() not in project.lower():
            continue

        # Pending function_call entries: call_id -> (name, input, timestamp, turn_number)
        pending: dict[str, tuple[str, dict[str, Any], str, int]] = {}
        turn_number = 0

        for record in records:
            rtype = str(record.get("type", ""))
            payload = record.get("payload", {})
            payload_type = str(payload.get("type", ""))
            ts = str(record.get("timestamp", ""))

            if rtype == "event_msg" and payload_type == "user_message":
                turn_number += 1
                continue

            if rtype == "response_item" and payload_type == "function_call":
                call_id = str(payload.get("call_id", ""))
                name = str(payload.get("name", ""))
                arguments = payload.get("arguments", "")
                try:
                    inp = json.loads(arguments) if isinstance(arguments, str) and arguments else {}
                except json.JSONDecodeError:
                    inp = {"raw": arguments}
                if not isinstance(inp, dict):
                    inp = {"value": inp}
                if call_id and name:
                    pending[call_id] = (name, inp, ts, turn_number)
                continue

            if rtype == "response_item" and payload_type == "function_call_output":
                call_id = str(payload.get("call_id", ""))
                if call_id not in pending:
                    continue

                tool_name, tool_input, tool_ts, tool_turn = pending.pop(call_id)
                category = classify_tool(tool_name)
                output = payload.get("output", "")
                output_text = _compact_json(output, max_chars=2000) if output else ""
                is_error = bool(payload.get("is_error", False))
                errors = _extract_errors(output, max_items=1) if output else []
                if errors and not is_error:
                    is_error = True
                error_snippet = errors[0] if errors else ""

                rec = ToolCallRecord(
                    session_id=full_session_id,
                    project=project,
                    source="codex",
                    timestamp=tool_ts or ts,
                    tool_name=tool_name,
                    tool_category=category,
                    tool_input=tool_input,
                    tool_output=output_text,
                    is_error=is_error,
                    error_snippet=error_snippet,
                    duration_ms=None,
                    turn_number=tool_turn,
                    tool_use_id=call_id,
                )

                if _passes_filters(rec, tool_filter, category_filter, errors_only):
                    yield rec


def extract_all(
    transcripts_dir: Path,
    codex_dir: Path,
    **filters: Any,
) -> Iterator[ToolCallRecord]:
    """Extract tool call records from both Claude Code and Codex transcripts."""
    yield from extract_from_claude_jsonl(transcripts_dir, **filters)
    yield from extract_from_codex_jsonl(codex_dir, **filters)


def _passes_filters(
    rec: ToolCallRecord,
    tool_filter: str,
    category_filter: str,
    errors_only: bool,
) -> bool:
    if tool_filter and tool_filter.lower() not in rec.tool_name.lower():
        return False
    if category_filter and rec.tool_category != category_filter:
        return False
    if errors_only and not rec.is_error:
        return False
    return True
