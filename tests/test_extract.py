from __future__ import annotations

import json
from pathlib import Path

import pytest

from alzabo.extract import (
    ToolCallRecord,
    classify_tool,
    extract_all,
    extract_from_claude_jsonl,
    extract_from_codex_jsonl,
)
from alzabo.extract_cli import _print_stats


class TestClassifyTool:
    def test_mcp_tool(self):
        assert classify_tool("mcp__alzabo__search_conversations") == "mcp"

    def test_bash_tool(self):
        assert classify_tool("Bash") == "bash"
        assert classify_tool("bash") == "bash"

    def test_agent_tool(self):
        assert classify_tool("Agent") == "agent"

    def test_builtin_tool(self):
        assert classify_tool("Read") == "builtin"
        assert classify_tool("Write") == "builtin"
        assert classify_tool("Edit") == "builtin"
        assert classify_tool("Grep") == "builtin"
        assert classify_tool("Glob") == "builtin"


class TestClaudeExtraction:
    def _write_claude_session(self, tmp_path: Path, session_id: str, records: list[dict]) -> Path:
        project_dir = tmp_path / "test-project"
        project_dir.mkdir(exist_ok=True)
        jsonl_file = project_dir / f"{session_id}.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return tmp_path

    def test_basic_tool_use_result_pairing(self, tmp_path):
        sid = "session-1"
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "Read a file"},
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_001",
                            "name": "Read",
                            "input": {"file_path": "/tmp/test.py"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_001",
                            "content": "file contents here",
                        }
                    ]
                },
            },
        ]
        base = self._write_claude_session(tmp_path, sid, records)
        results = list(extract_from_claude_jsonl(base))

        assert len(results) == 1
        rec = results[0]
        assert rec.session_id == sid
        assert rec.tool_name == "Read"
        assert rec.tool_category == "builtin"
        assert rec.tool_use_id == "tu_001"
        assert rec.tool_input == {"file_path": "/tmp/test.py"}
        assert "file contents here" in rec.tool_output
        assert rec.is_error is False
        assert rec.source == "claude"
        assert rec.project == "test-project"

    def test_multiple_tool_calls_in_one_assistant_message(self, tmp_path):
        sid = "session-2"
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "Do two things"},
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_a",
                            "name": "Read",
                            "input": {"file_path": "/a.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "tu_b",
                            "name": "Bash",
                            "input": {"command": "ls"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_a",
                            "content": "contents of a.py",
                        },
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_b",
                            "content": "file1.txt\nfile2.txt",
                        },
                    ]
                },
            },
        ]
        base = self._write_claude_session(tmp_path, sid, records)
        results = list(extract_from_claude_jsonl(base))

        assert len(results) == 2
        names = {r.tool_name for r in results}
        assert names == {"Read", "Bash"}
        bash_rec = next(r for r in results if r.tool_name == "Bash")
        assert bash_rec.tool_category == "bash"

    def test_error_detection(self, tmp_path):
        sid = "session-err"
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "run tests"},
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_err",
                            "name": "Bash",
                            "input": {"command": "pytest"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_err",
                            "content": "Traceback (most recent call last):\n  File 'test.py'\nAssertionError: failed",
                        }
                    ]
                },
            },
        ]
        base = self._write_claude_session(tmp_path, sid, records)
        results = list(extract_from_claude_jsonl(base))

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].error_snippet  # should have extracted an error line

    def test_is_error_flag_from_block(self, tmp_path):
        sid = "session-flag"
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "do something"},
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_flag",
                            "name": "Read",
                            "input": {"file_path": "/nope"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_flag",
                            "is_error": True,
                            "content": "file not found",
                        }
                    ]
                },
            },
        ]
        base = self._write_claude_session(tmp_path, sid, records)
        results = list(extract_from_claude_jsonl(base))
        assert len(results) == 1
        assert results[0].is_error is True

    def test_mcp_tool_classification(self, tmp_path):
        sid = "session-mcp"
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "search"},
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "tu_mcp",
                            "name": "mcp__alzabo__search_conversations",
                            "input": {"query": "terraform"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "tu_mcp",
                            "content": "found 3 results",
                        }
                    ]
                },
            },
        ]
        base = self._write_claude_session(tmp_path, sid, records)
        results = list(extract_from_claude_jsonl(base))
        assert len(results) == 1
        assert results[0].tool_category == "mcp"


class TestCodexExtraction:
    def _write_codex_session(self, tmp_path: Path, records: list[dict]) -> Path:
        jsonl_file = tmp_path / "session.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return tmp_path

    def test_basic_function_call_output_pairing(self, tmp_path):
        records = [
            {
                "type": "session_meta",
                "payload": {"id": "codex-1", "cwd": "/tmp/my-project"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "find bugs"},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "fc_001",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "pytest", "workdir": "/tmp/my-project"}),
                },
                "timestamp": "2026-01-01T00:00:02Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "fc_001",
                    "output": "all tests passed",
                    "is_error": False,
                },
                "timestamp": "2026-01-01T00:00:03Z",
            },
        ]
        base = self._write_codex_session(tmp_path, records)
        results = list(extract_from_codex_jsonl(base))

        assert len(results) == 1
        rec = results[0]
        assert rec.session_id == "codex:codex-1"
        assert rec.tool_name == "exec_command"
        assert rec.source == "codex"
        assert rec.project == "my-project"
        assert rec.is_error is False
        assert rec.tool_use_id == "fc_001"
        assert rec.tool_input == {"cmd": "pytest", "workdir": "/tmp/my-project"}

    def test_codex_error_detection(self, tmp_path):
        records = [
            {
                "type": "session_meta",
                "payload": {"id": "codex-err", "cwd": "/tmp/proj"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "run it"},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "fc_err",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "make build"}),
                },
                "timestamp": "2026-01-01T00:00:02Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "fc_err",
                    "output": "stderr: build failed with error",
                    "is_error": True,
                },
                "timestamp": "2026-01-01T00:00:03Z",
            },
        ]
        base = self._write_codex_session(tmp_path, records)
        results = list(extract_from_codex_jsonl(base))

        assert len(results) == 1
        assert results[0].is_error is True
        assert results[0].error_snippet


class TestFilters:
    @pytest.fixture
    def claude_dir(self, tmp_path):
        project_dir = tmp_path / "claude" / "test-project"
        project_dir.mkdir(parents=True)
        sid = "filter-session"
        records = [
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "do stuff"},
            },
            {
                "type": "assistant",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a.py"}},
                        {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "ls"}},
                        {"type": "tool_use", "id": "t3", "name": "mcp__foo__bar", "input": {"q": "x"}},
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": sid,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
                        {"type": "tool_result", "tool_use_id": "t2", "content": "stderr: error happened"},
                        {"type": "tool_result", "tool_use_id": "t3", "content": "found it"},
                    ]
                },
            },
        ]
        jsonl_file = project_dir / f"{sid}.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        return tmp_path / "claude"

    def test_filter_by_tool_name(self, claude_dir):
        results = list(extract_from_claude_jsonl(claude_dir, tool_filter="Read"))
        assert len(results) == 1
        assert results[0].tool_name == "Read"

    def test_filter_by_category(self, claude_dir):
        results = list(extract_from_claude_jsonl(claude_dir, category_filter="mcp"))
        assert len(results) == 1
        assert results[0].tool_name == "mcp__foo__bar"

    def test_filter_errors_only(self, claude_dir):
        results = list(extract_from_claude_jsonl(claude_dir, errors_only=True))
        assert len(results) == 1
        assert results[0].tool_name == "Bash"
        assert results[0].is_error is True

    def test_filter_by_session(self, claude_dir):
        results = list(extract_from_claude_jsonl(claude_dir, session_filter="filter-session"))
        assert len(results) == 3
        results_none = list(extract_from_claude_jsonl(claude_dir, session_filter="nonexistent"))
        assert len(results_none) == 0


class TestExtractAll:
    def test_combines_claude_and_codex(self, tmp_path):
        # Claude session
        claude_dir = tmp_path / "claude" / "proj"
        claude_dir.mkdir(parents=True)
        claude_records = [
            {
                "type": "user",
                "sessionId": "cs1",
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "hello"},
            },
            {
                "type": "assistant",
                "sessionId": "cs1",
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {"type": "tool_use", "id": "c1", "name": "Read", "input": {"file_path": "/x"}}
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": "cs1",
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [{"type": "tool_result", "tool_use_id": "c1", "content": "ok"}]
                },
            },
        ]
        (claude_dir / "cs1.jsonl").write_text(
            "\n".join(json.dumps(r) for r in claude_records) + "\n"
        )

        # Codex session
        codex_dir = tmp_path / "codex"
        codex_dir.mkdir()
        codex_records = [
            {
                "type": "session_meta",
                "payload": {"id": "cx1", "cwd": "/tmp/proj"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "go"},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "call_id": "x1",
                    "name": "exec_command",
                    "arguments": "{}",
                },
                "timestamp": "2026-01-01T00:00:02Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "call_id": "x1",
                    "output": "done",
                },
                "timestamp": "2026-01-01T00:00:03Z",
            },
        ]
        (codex_dir / "cx1.jsonl").write_text(
            "\n".join(json.dumps(r) for r in codex_records) + "\n"
        )

        results = list(extract_all(tmp_path / "claude", codex_dir))
        sources = {r.source for r in results}
        assert "claude" in sources
        assert "codex" in sources
        assert len(results) == 2


class TestToJsonl:
    def test_serialization_roundtrip(self):
        rec = ToolCallRecord(
            session_id="s1",
            project="proj",
            source="claude",
            timestamp="2026-01-01T00:00:00Z",
            tool_name="Read",
            tool_category="builtin",
            tool_input={"file_path": "/tmp/x"},
            tool_output="content",
            is_error=False,
            error_snippet="",
            duration_ms=None,
            turn_number=0,
            tool_use_id="tu1",
        )
        line = rec.to_jsonl()
        parsed = json.loads(line)
        assert parsed["tool_name"] == "Read"
        assert parsed["session_id"] == "s1"
        assert parsed["duration_ms"] is None


class TestPrintStats:
    def test_stats_output(self, capsys):
        records = [
            ToolCallRecord(
                session_id="s1", project="p1", source="claude",
                timestamp="2026-01-01T00:00:00Z", tool_name="Read",
                tool_category="builtin", tool_input={}, tool_output="ok",
                is_error=False, error_snippet="", duration_ms=None,
                turn_number=0, tool_use_id="t1",
            ),
            ToolCallRecord(
                session_id="s1", project="p1", source="claude",
                timestamp="2026-01-01T00:00:01Z", tool_name="Bash",
                tool_category="bash", tool_input={"command": "ls"},
                tool_output="stderr: error", is_error=True,
                error_snippet="stderr: error", duration_ms=None,
                turn_number=0, tool_use_id="t2",
            ),
        ]
        _print_stats(records)
        out = capsys.readouterr().out
        assert "total tool calls: 2" in out
        assert "errors: 1" in out
        assert "builtin:" in out
        assert "bash:" in out

    def test_empty_stats(self, capsys):
        _print_stats([])
        out = capsys.readouterr().out
        assert "No tool call records found" in out
