from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

import alzabo.index as idxmod
import alzabo.parsers as parsers
from alzabo.render import (
    render_conversation,
    render_index_status,
    render_list_conversations,
    render_search_conversations,
    render_search_sessions,
    render_turn,
)


def _make_turn(
    session_id: str = "test-session",
    turn_number: int = 0,
    project: str = "test-project",
    source: str = "claude",
    search_text: str = "hello world",
    timestamp: str = "2026-01-01T00:00:00Z",
    signals: idxmod.TurnSignals | None = None,
) -> idxmod.Turn:
    return idxmod.Turn(
        session_id=session_id,
        turn_number=turn_number,
        timestamp=timestamp,
        project=project,
        branch="main",
        slug="test",
        source=source,
        user_content="hello",
        assistant_content=[],
        tool_results=[],
        summary=search_text[:300],
        signals=signals or idxmod.TurnSignals(),
        records=[],
        search_text=search_text,
    )


def _make_convo(session_id: str = "test-session", turns: list[idxmod.Turn] | None = None) -> idxmod.Conversation:
    turns = turns or []
    return idxmod.Conversation(
        session_id=session_id,
        project="test-project",
        branch="main",
        slug="test",
        source="claude",
        summary="test convo",
        first_timestamp="2026-01-01T00:00:00Z",
        last_timestamp="2026-01-01T01:00:00Z",
        turns=turns,
    )


@pytest.fixture
def sample_turns() -> list[idxmod.Turn]:
    return [
        _make_turn(
            search_text="terraform infrastructure as code deployment",
            turn_number=0,
            signals=idxmod.TurnSignals(tools=["Bash"], commands=["terraform apply"]),
        ),
        _make_turn(search_text="debugging authentication login token issues", turn_number=1),
        _make_turn(
            search_text="kubernetes pod networking service mesh",
            turn_number=2,
            signals=idxmod.TurnSignals(files=["k8s/service.yaml"]),
        ),
        _make_turn(search_text="python pytest unit testing best practices", turn_number=3),
        _make_turn(
            search_text="docker container image build optimization",
            turn_number=4,
            signals=idxmod.TurnSignals(errors=["build failed"]),
        ),
    ]


@pytest.fixture
def sample_index(sample_turns) -> idxmod.Index:
    idx = idxmod.Index()
    convo = _make_convo(turns=sample_turns)
    idx.conversations["test-session"] = convo
    for turn in sample_turns:
        idx.turns.append(turn)
        idx.corpus.append(turn.search_text.lower().split())
    idx.build()
    idx.embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.8, 0.2, 0.0],
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 0.8],
        ],
        dtype=np.float32,
    )
    return idx


class TestHelpers:
    def test_parse_timestamp(self):
        assert idxmod.parse_timestamp("2026-01-01T00:00:00Z") is not None

    def test_normalize_project(self):
        assert idxmod.normalize_project("-Users-zander-go-src-github-com-DataDog-csm-pde-tools") == "csm-pde-tools"

    def test_rrf_fuse(self):
        fused = idxmod.rrf_fuse([(0, 2.0), (1, 1.0)], [(1, 0.7), (2, 0.6)])
        assert fused[0][0] == 1


class TestParsers:
    def test_parse_claude_tool_use_extracts_signals(self):
        parsed = parsers.parse_claude_content(
            [
                {
                    "type": "tool_use",
                    "name": "Bash",
                    "input": {
                        "command": "uv run --project transcript-search pytest",
                        "path": "/tmp/work/file.py",
                    },
                }
            ]
        )
        assert "Bash" in parsed.tools
        assert "/tmp/work/file.py" in parsed.files
        assert "uv run --project transcript-search pytest" in parsed.commands

    def test_parse_claude_tool_result_extracts_errors(self):
        parsed = parsers.parse_claude_content(
            [{"type": "tool_result", "content": "Traceback: boom\nstderr: failed to parse"}]
        )
        assert parsed.errors

    def test_parse_codex_function_call_extracts_paths_and_commands(self):
        parsed = parsers.parse_codex_function_call(
            {
                "name": "exec_command",
                "arguments": json.dumps({"cmd": "rg TODO", "workdir": "/tmp/repo"}),
            }
        )
        assert "exec_command" in parsed.tools
        assert "rg TODO" in parsed.commands
        assert "/tmp/repo" in parsed.files

    def test_parse_codex_function_output_extracts_errors(self):
        parsed = parsers.parse_codex_function_output(
            {"output": "stderr: build failed", "is_error": True}
        )
        assert parsed.errors


class TestIndexBuilding:
    def test_build_claude_index_keeps_tool_result_in_same_turn(self, tmp_path):
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()
        session_id = "claude-session"
        records = [
            {
                "type": "user",
                "sessionId": session_id,
                "timestamp": "2026-01-01T00:00:00Z",
                "message": {"content": "Run the command"},
            },
            {
                "type": "assistant",
                "sessionId": session_id,
                "timestamp": "2026-01-01T00:00:01Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "pytest transcript-search/tests/test_server.py"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "sessionId": session_id,
                "timestamp": "2026-01-01T00:00:02Z",
                "message": {
                    "content": [{"type": "tool_result", "content": "stderr: failed assertion"}]
                },
            },
            {
                "type": "assistant",
                "sessionId": session_id,
                "timestamp": "2026-01-01T00:00:03Z",
                "message": {"content": [{"type": "text", "text": "I fixed it."}]},
            },
        ]
        jsonl_file = project_dir / "session.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(record) for record in records) + "\n")

        idx, count = idxmod.build_claude_index(tmp_path)

        assert count == 1
        turn = idx.conversations[session_id].turns[0]
        assert "pytest transcript-search/tests/test_server.py" in turn.search_text
        assert turn.tool_results
        assert turn.signals.errors

    def test_build_codex_index_parses_function_call_output(self, tmp_path):
        records = [
            {
                "type": "session_meta",
                "payload": {"id": "codex-123", "cwd": "/tmp/my-project"},
                "timestamp": "2026-01-01T00:00:00Z",
            },
            {
                "type": "event_msg",
                "payload": {"type": "user_message", "message": "find the failing command"},
                "timestamp": "2026-01-01T00:00:01Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "exec_command",
                    "arguments": json.dumps({"cmd": "pytest", "workdir": "/tmp/my-project"}),
                },
                "timestamp": "2026-01-01T00:00:02Z",
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "function_call_output",
                    "output": "stderr: build failed",
                    "is_error": True,
                },
                "timestamp": "2026-01-01T00:00:03Z",
            },
        ]

        jsonl_file = tmp_path / "session.jsonl"
        jsonl_file.write_text("\n".join(json.dumps(record) for record in records) + "\n")

        idx, count = idxmod.build_codex_index(tmp_path)
        assert count == 1
        turn = idx.conversations["codex:codex-123"].turns[0]
        assert "pytest" in turn.search_text
        assert turn.tool_results
        assert turn.signals.errors


class TestManagerQueries:
    @pytest.fixture(autouse=True)
    def setup_manager(self, sample_index):
        self.manager = idxmod.TranscriptIndexManager()
        self.manager._index = sample_index
        self.manager._index_ready.set()
        self.manager._last_reindex_at = "2026-01-01T00:00:00Z"

    def test_search_conversations_bm25(self):
        result = self.manager.search_conversations(query="terraform", limit=3, mode="bm25")
        rendered = render_search_conversations(result)
        assert "mode: bm25" in rendered
        assert "#1" in rendered

    def test_search_conversations_context_window(self):
        result = self.manager.search_conversations(query="terraform", limit=3, mode="bm25", context_window=1)
        rendered = render_search_conversations(result)
        assert "context:" in rendered
        assert "turn 0" in rendered

    def test_search_sessions(self):
        result = self.manager.search_sessions(query="terraform", limit=3, mode="bm25")
        rendered = render_search_sessions(result)
        assert "test-session" in rendered
        assert "top tools:" in rendered

    def test_list_conversations(self):
        page = self.manager.list_conversations()
        rendered = render_list_conversations(page)
        assert "1 sessions" in rendered
        assert "top tools:" in rendered

    def test_read_turn_renders_signals(self):
        turn = self.manager.get_turn("test-session", 0)
        rendered = render_turn(turn)
        assert "signals:" in rendered

    def test_read_conversation_compact(self):
        convo = self.manager.get_conversation("test-session")
        rendered = render_conversation(convo, compact=True)
        assert "=== turn 0" in rendered

    def test_index_status(self):
        status = self.manager.get_index_status()
        rendered = render_index_status(status)
        assert "alzabo status" in rendered
        assert "last reindex: 2026-01-01T00:00:00Z" in rendered


class TestAsDict:
    def test_index_status_as_dict(self):
        status = idxmod.IndexStatus(
            transcripts_dir="/tmp/claude",
            codex_dir="/tmp/codex",
            watch_enabled=True,
            total_sessions=5,
            claude_sessions=3,
            codex_sessions=2,
            total_turns=20,
            embeddings_ready=True,
            last_reindex_at="2026-01-01T00:00:00Z",
        )
        d = status.as_dict()
        assert json.loads(json.dumps(d))  # JSON-serializable
        assert d["total_sessions"] == 5
        assert d["watch_enabled"] is True

    def test_turn_search_result_as_dict_no_context(self):
        turn = _make_turn(search_text="hello world")
        result = idxmod.TurnSearchResult(turn=turn, score=0.12345)
        d = result.as_dict()
        assert json.loads(json.dumps(d))
        assert d["score"] == 0.1235
        assert "context" not in d
        assert d["turn"]["session_id"] == "test-session"

    def test_turn_search_result_as_dict_with_context(self):
        turn = _make_turn(search_text="main", turn_number=1)
        ctx = _make_turn(search_text="context", turn_number=0)
        result = idxmod.TurnSearchResult(turn=turn, score=0.5, context=[ctx])
        d = result.as_dict()
        assert "context" in d
        assert len(d["context"]) == 1
        # Context turns exclude content
        assert "user_content" not in d["context"][0]

    def test_session_search_result_as_dict(self):
        convo = _make_convo(turns=[_make_turn()])
        result = idxmod.SessionSearchResult(
            conversation=convo,
            best_score=0.99,
            best_turn_number=0,
            best_turn_summary="test summary",
            matching_turns=3,
        )
        d = result.as_dict()
        assert json.loads(json.dumps(d))
        assert d["score"] == 0.99
        assert d["matching_turns"] == 3
        assert d["conversation"]["session_id"] == "test-session"

    def test_search_result_set_as_dict_no_effective_mode(self):
        result_set = idxmod.SearchResultSet(query="test", mode="bm25", effective_mode="bm25", items=[])
        d = result_set.as_dict()
        assert json.loads(json.dumps(d))
        assert d["result_count"] == 0
        assert "effective_mode" not in d

    def test_search_result_set_as_dict_with_effective_mode(self):
        result_set = idxmod.SearchResultSet(query="test", mode="hybrid", effective_mode="bm25", items=[])
        d = result_set.as_dict()
        assert d["effective_mode"] == "bm25"

    def test_session_result_set_as_dict(self):
        result_set = idxmod.SessionResultSet(query="q", mode="hybrid", effective_mode="hybrid", items=[])
        d = result_set.as_dict()
        assert json.loads(json.dumps(d))
        assert d["query"] == "q"
        assert "effective_mode" not in d

    def test_conversation_page_as_dict_no_next(self):
        page = idxmod.ConversationPage(items=[], total=0, offset=0, end=0, next_offset=None)
        d = page.as_dict()
        assert json.loads(json.dumps(d))
        assert "next_offset" not in d
        assert d["total"] == 0

    def test_conversation_page_as_dict_with_next(self):
        convo = _make_convo(turns=[_make_turn()])
        page = idxmod.ConversationPage(items=[convo], total=5, offset=0, end=1, next_offset=1)
        d = page.as_dict()
        assert d["next_offset"] == 1
        assert len(d["items"]) == 1
        assert d["items"][0]["session_id"] == "test-session"


class TestReindex:
    def test_reindex_logs_and_status(self, tmp_path, monkeypatch, capsys):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        project_dir = transcripts / "test-project"
        project_dir.mkdir(parents=True)
        codex.mkdir()
        jsonl = project_dir / "s.jsonl"
        jsonl.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "s1",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "message": {"content": "hello"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "s1",
                            "timestamp": "2026-01-01T00:01:00Z",
                            "message": {"content": [{"type": "text", "text": "hi"}]},
                        }
                    ),
                ]
            )
            + "\n"
        )

        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))
        manager = idxmod.TranscriptIndexManager()
        manager.configure(transcripts, codex, watch_enabled=False)

        manager.reindex()

        captured = capsys.readouterr()
        assert "indexed 1 turns" in captured.err
        assert "embedded 1 turns" in captured.err
        assert manager.get_index_status().total_turns == 1


class TestStartupLatency:
    def test_mcp_handshake_responds_before_indexing_completes(self, tmp_path):
        claude_dir = tmp_path / "claude"
        codex_dir = tmp_path / "codex"
        claude_dir.mkdir()
        codex_dir.mkdir()
        init_msg = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0.1"},
                },
            }
        )
        proc = subprocess.Popen(
            [
                "uv",
                "run",
                "alzabo-serve",
                "--transcripts-dir",
                str(claude_dir),
                "--codex-dir",
                str(codex_dir),
                "--no-watch",
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        deadline = 3.0
        try:
            proc.stdin.write((init_msg + "\n").encode())
            proc.stdin.flush()
            t0 = time.monotonic()
            response_bytes = b""
            while time.monotonic() - t0 < deadline:
                import select

                ready, _, _ = select.select([proc.stdout], [], [], 0.1)
                if ready:
                    chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
                    if chunk:
                        response_bytes += chunk
                        if b'"result"' in response_bytes and b'"protocolVersion"' in response_bytes:
                            break
            assert response_bytes
            assert b'"protocolVersion"' in response_bytes
            assert time.monotonic() - t0 < deadline
        finally:
            proc.terminate()
            proc.wait(timeout=5)
