from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

_PATH_KEYS = {
    "path",
    "paths",
    "file",
    "files",
    "file_path",
    "filepath",
    "notebook_path",
    "old_path",
    "new_path",
    "cwd",
    "workdir",
    "directory",
    "dirs",
    "target",
}
_COMMAND_KEYS = {"command", "cmd", "shell_command"}
_ERROR_TERMS = ("error", "stderr", "traceback", "failed", "panic", "exception")


@dataclass
class ParsedContent:
    text: str = ""
    tools: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


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


def _walk_strings(value: Any, key: str = "") -> list[tuple[str, str]]:
    strings: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            strings.extend(_walk_strings(child_value, child_key))
    elif isinstance(value, list):
        for item in value:
            strings.extend(_walk_strings(item, key))
    elif isinstance(value, str):
        strings.append((key, value.strip()))
    return strings


def _extract_paths(value: Any) -> list[str]:
    paths: list[str] = []
    for key, text in _walk_strings(value):
        if not text:
            continue
        lowered_key = key.lower()
        if lowered_key in _PATH_KEYS:
            paths.append(text)
            continue
        if "/" in text and len(text) < 300:
            if text.startswith(("/", "./", "../")) or "." in text.rsplit("/", 1)[-1]:
                paths.append(text)
    return _dedupe(paths)


def _extract_commands(value: Any) -> list[str]:
    commands: list[str] = []
    for key, text in _walk_strings(value):
        if key.lower() in _COMMAND_KEYS and text:
            commands.append(text)
    return _dedupe(commands)


def _extract_errors(value: Any, max_items: int = 5) -> list[str]:
    matches: list[str] = []
    for _key, text in _walk_strings(value):
        if not text:
            continue
        for line in text.splitlines():
            cleaned = line.strip()
            lowered = cleaned.lower()
            if cleaned and any(term in lowered for term in _ERROR_TERMS):
                matches.append(cleaned[:240])
    return _dedupe(matches)[:max_items]


def _merge_contents(parts: list[ParsedContent]) -> ParsedContent:
    merged = ParsedContent()
    merged.text = "\n".join(part.text for part in parts if part.text).strip()
    merged.tools = _dedupe([tool for part in parts for tool in part.tools])
    merged.files = _dedupe([path for part in parts for path in part.files])
    merged.commands = _dedupe([command for part in parts for command in part.commands])
    merged.errors = _dedupe([error for part in parts for error in part.errors])
    return merged


def _parsed_from_value(value: Any, *, include_text: bool = True, max_chars: int = 4000) -> ParsedContent:
    text = _compact_json(value, max_chars=max_chars) if include_text else ""
    return ParsedContent(
        text=text,
        files=_extract_paths(value),
        commands=_extract_commands(value),
        errors=_extract_errors(value),
    )


def parse_claude_content(content: Any) -> ParsedContent:
    if isinstance(content, str):
        return ParsedContent(text=content.strip())
    if not isinstance(content, list):
        return ParsedContent()

    parts: list[ParsedContent] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "text" and isinstance(block.get("text"), str):
            text = block["text"].strip()
            if text:
                parts.append(ParsedContent(text=text))
        elif block_type == "tool_use":
            name = str(block.get("name", "")).strip()
            parsed = _parsed_from_value(block.get("input"), include_text=True, max_chars=2000)
            if name:
                parsed.tools = [name]
                parsed.text = "\n".join(part for part in [name, parsed.text] if part).strip()
            parts.append(parsed)
        elif block_type == "tool_result":
            parts.append(_parsed_from_value(block.get("content"), include_text=True))
    return _merge_contents(parts)


def parse_claude_record(record: dict[str, Any]) -> ParsedContent:
    message = record.get("message", {})
    content = message.get("content") if isinstance(message, dict) else None
    parsed = parse_claude_content(content)
    extra = record.get("toolUseResult")
    if extra:
        parsed = _merge_contents([parsed, _parsed_from_value(extra, include_text=True)])
    return parsed


def _render_tool_value(value: Any, max_chars: int = 2000) -> str:
    if value is None:
        return ""
    rendered = _compact_json(value, max_chars=max_chars)
    return rendered.strip()


def fmt_claude_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type == "text" and isinstance(block.get("text"), str):
            value = block["text"].strip()
            if value:
                parts.append(value)
        elif block_type == "tool_use":
            name = block.get("name", "unknown")
            input_text = _render_tool_value(block.get("input"))
            if input_text:
                parts.append(f"[tool_use: {name}] {input_text}")
            else:
                parts.append(f"[tool_use: {name}]")
        elif block_type == "tool_result":
            result_text = _render_tool_value(block.get("content"), max_chars=4000)
            if result_text:
                parts.append(f"[tool_result] {result_text}")
            else:
                parts.append("[tool_result]")
    return "\n".join(part for part in parts if part).strip()


def fmt_claude_content_list(content_list: list[Any]) -> str:
    return "\n".join(part for part in (fmt_claude_content(item) for item in content_list) if part)


def parse_codex_message_content(content: Any) -> ParsedContent:
    if isinstance(content, str):
        return ParsedContent(text=content.strip())
    if not isinstance(content, list):
        return ParsedContent()

    parts: list[ParsedContent] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type", ""))
        if block_type in {"output_text", "input_text"} and isinstance(block.get("text"), str):
            text = block["text"].strip()
            if text:
                parts.append(ParsedContent(text=text))
        else:
            parts.append(_parsed_from_value(block, include_text=True, max_chars=2000))
    return _merge_contents(parts)


def parse_codex_function_call(payload: dict[str, Any]) -> ParsedContent:
    name = str(payload.get("name", "")).strip()
    arguments = payload.get("arguments", "")
    try:
        parsed_args: Any = json.loads(arguments) if isinstance(arguments, str) and arguments else arguments
    except json.JSONDecodeError:
        parsed_args = arguments
    parsed = _parsed_from_value(parsed_args, include_text=True, max_chars=2000)
    if name:
        parsed.tools = [name]
        parsed.text = "\n".join(part for part in [name, parsed.text] if part).strip()
    return parsed


def parse_codex_function_output(payload: dict[str, Any]) -> ParsedContent:
    output = payload.get("output")
    parsed = _parsed_from_value(output, include_text=True)
    if payload.get("is_error") and parsed.text:
        parsed.errors = _dedupe([parsed.text[:240], *parsed.errors])
    return parsed
