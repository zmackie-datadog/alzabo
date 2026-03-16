from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pytest

import alzabo.cache as cache_mod
import alzabo.index as idxmod
from alzabo.main_cli import build_parser


class TestParser:
    def test_search_parses(self):
        parser = build_parser()
        args = parser.parse_args(["search", "oauth", "--mode", "bm25", "--limit", "5"])
        assert args.command == "search"
        assert args.query == "oauth"
        assert args.mode == "bm25"
        assert args.limit == 5
        assert args.sessions is False

    def test_search_sessions_flag(self):
        parser = build_parser()
        args = parser.parse_args(["search", "test", "--sessions"])
        assert args.sessions is True

    def test_list_parses(self):
        parser = build_parser()
        args = parser.parse_args(["list", "--format", "json", "--offset", "10"])
        assert args.command == "list"
        assert args.offset == 10
        assert args.format == "json"

    def test_read_parses(self):
        parser = build_parser()
        args = parser.parse_args(["read", "abc-123", "--turn", "2"])
        assert args.command == "read"
        assert args.session_id == "abc-123"
        assert args.turn == 2

    def test_read_no_turn(self):
        parser = build_parser()
        args = parser.parse_args(["read", "abc-123"])
        assert args.turn is None

    def test_status_parses(self):
        parser = build_parser()
        args = parser.parse_args(["status", "--no-cache"])
        assert args.command == "status"
        assert args.no_cache is True

    def test_extract_parses(self):
        parser = build_parser()
        args = parser.parse_args(["extract", "--stats", "--tool", "Bash"])
        assert args.command == "extract"
        assert args.stats is True
        assert args.tool == "Bash"

    def test_global_format_default(self):
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.format == "text"

    def test_global_format_jsonl(self):
        parser = build_parser()
        args = parser.parse_args(["list", "--format", "jsonl"])
        assert args.format == "jsonl"

    def test_global_quiet(self):
        parser = build_parser()
        args = parser.parse_args(["status", "--quiet"])
        assert args.quiet is True

    def test_global_cache_dir(self):
        parser = build_parser()
        args = parser.parse_args(["status", "--cache-dir", "/tmp/alzabo-cache-test"])
        assert args.cache_dir == "/tmp/alzabo-cache-test"

    def test_global_debounce_seconds(self):
        parser = build_parser()
        args = parser.parse_args(["status"])
        assert args.debounce_seconds == 2.0



class TestCacheConfig:
    def test_cache_dir_env(self, tmp_path, monkeypatch):
        parser = build_parser()
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        monkeypatch.setenv("ALZABO_CACHE_DIR", str(tmp_path / "from-env"))
        args = parser.parse_args([
            "status",
            "--no-cache",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])
        from alzabo.main_cli import _get_manager

        _get_manager(args)
        assert cache_mod.get_cache_dir() == (tmp_path / "from-env").resolve()


class TestCacheBypass:
    @pytest.fixture(autouse=True)
    def override_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")

    def test_no_cache_forces_reindex(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        reindex_calls = []
        original_reindex = idxmod.TranscriptIndexManager.reindex

        def tracking_reindex(self):
            reindex_calls.append(True)
            return original_reindex(self)

        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", tracking_reindex)

        parser = build_parser()
        args = parser.parse_args([
            "status", "--no-cache",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        from alzabo.main_cli import _get_manager
        _get_manager(args)
        assert len(reindex_calls) == 1

    def test_fresh_cache_skips_reindex(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        # Pre-populate cache
        idx = idxmod.Index()
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)

        reindex_calls = []
        original_reindex = idxmod.TranscriptIndexManager.reindex

        def tracking_reindex(self):
            reindex_calls.append(True)
            return original_reindex(self)

        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", tracking_reindex)

        parser = build_parser()
        args = parser.parse_args([
            "status",
            "--debounce-seconds", "0",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        from alzabo.main_cli import _get_manager
        _get_manager(args)
        assert len(reindex_calls) == 0

    def test_fresh_cache_skips_embedding(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        embed_calls: list[int] = []

        def fake_embed_texts(texts: list[str]) -> np.ndarray:
            embed_calls.append(len(texts))
            return np.ones((len(texts), 512), dtype=np.float32)

        idx = idxmod.Index()
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)

        parser = build_parser()
        args = parser.parse_args([
            "status",
            "--debounce-seconds", "0",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        from alzabo.main_cli import _get_manager
        monkeypatch.setattr(idxmod, "embed_texts", fake_embed_texts)
        _get_manager(args)
        assert embed_calls == []


class TestIncrementalCache:
    @pytest.fixture(autouse=True)
    def override_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")

    def test_stale_cache_with_unstable_changes_uses_existing_cache(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        session_file = transcripts / "session.jsonl"
        session_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "message": {"content": "first question"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:01:00Z",
                            "message": {"content": [{"type": "text", "text": "first answer"}]},
                        }
                    ),
                ]
            )
            + "\n"
        )

        cached_index, _ = idxmod.build_claude_index(transcripts)
        cache_mod.save_cache(cached_index, transcripts, codex)

        session_file.write_text(
            session_file.read_text()
            + "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:02:00Z",
                            "message": {"content": "second question"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:02:01Z",
                            "message": {"content": [{"type": "text", "text": "second answer"}]},
                        }
                    ),
                ]
            )
        )

        current_files = cache_mod.collect_source_files(transcripts, codex)
        unstable_now = current_files[str(session_file.resolve())]["mtime"] + 0.5
        monkeypatch.setattr(cache_mod.time, "time", lambda: unstable_now)

        def fail_rebuild(
            index: idxmod.Index,
            changed_files: set[str],
            transcripts_dir: Path,
            codex_dir: Path,
        ) -> idxmod.Index:
            raise AssertionError("incremental rebuild should not run while changes are unstable")

        def fail_reindex(self) -> int:
            raise AssertionError("full reindex should not run while changes are unstable")

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", fail_rebuild)
        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", fail_reindex)

        parser = build_parser()
        args = parser.parse_args([
            "status",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
            "--debounce-seconds", "10",
        ])

        from alzabo.main_cli import _get_manager

        manager = _get_manager(args)
        assert manager.get_index_status().total_turns == 1

    def test_stale_cache_uses_incremental_update(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        project_dir = transcripts / "test-project"
        project_dir.mkdir(parents=True)
        codex.mkdir()

        session_file = project_dir / "session.jsonl"
        session_file.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:00:00Z",
                            "message": {"content": "first question"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:00:01Z",
                            "message": {"content": [{"type": "text", "text": "first answer"}]},
                        }
                    ),
                ]
            )
            + "\n"
        )

        cached_index, _ = idxmod.build_claude_index(transcripts)
        cache_mod.save_cache(cached_index, transcripts, codex, reindex_at="2026-01-01T00:00:00Z")

        session_file.write_text(
            session_file.read_text()
            + "\n".join(
                [
                    json.dumps(
                        {
                            "type": "user",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:01:00Z",
                            "message": {"content": "second question"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "assistant",
                            "sessionId": "session-1",
                            "timestamp": "2026-01-01T00:01:01Z",
                            "message": {"content": [{"type": "text", "text": "second answer"}]},
                        }
                    ),
                ]
            ),
        )

        original_rebuild = idxmod.rebuild_index_incrementally
        observed: dict[str, set[str]] = {}

        def spy(index: idxmod.Index, changed_files: set[str], transcripts_dir: Path, codex_dir: Path) -> idxmod.Index:
            observed["files"] = set(changed_files)
            return original_rebuild(index, changed_files, transcripts_dir, codex_dir)

        def fail_reindex(self) -> int:
            raise AssertionError("full reindex should not run when incremental cache path applies")

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", spy)
        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", fail_reindex)

        from alzabo.main_cli import _get_manager, build_parser

        parser = build_parser()
        args = parser.parse_args([
            "status",
            "--debounce-seconds", "0",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        manager = _get_manager(args)
        assert str(session_file.resolve()) in observed.get("files", set())
        assert manager.get_index_status().total_turns == 2

    def test_incremental_reuses_cached_vectors(self, monkeypatch):
        transcripts_dir = Path("/tmp")
        codex_dir = Path("/tmp")

        stale_base = idxmod.Turn(
            session_id="s1",
            turn_number=0,
            timestamp="2026-01-01T00:00:00Z",
            project="p",
            branch="main",
            slug="s1",
            source="claude",
            user_content="question-keep",
            assistant_content=[],
            tool_results=[],
            summary="keep",
            signals=idxmod.TurnSignals(),
            records=[],
            search_text="keep text",
            source_file="/tmp/stale.jsonl",
        )

        stale = idxmod.Turn(
            session_id="s1",
            turn_number=1,
            timestamp="2026-01-01T00:01:00Z",
            project="p",
            branch="main",
            slug="s1",
            source="claude",
            user_content="question-keep",
            assistant_content=[],
            tool_results=[],
            summary="stale",
            signals=idxmod.TurnSignals(),
            records=[],
            search_text="stale text",
            source_file="/tmp/stale.jsonl",
        )

        index = idxmod.Index()
        convo = idxmod.Conversation(
            session_id="s1",
            project="p",
            branch="main",
            slug="s1",
            summary="",
            first_timestamp="2026-01-01T00:00:00Z",
            last_timestamp="2026-01-01T00:01:00Z",
            source="claude",
        )
        convo.turns = [stale_base, stale]
        index.conversations["s1"] = convo
        index.turns = [stale_base, stale]
        index.corpus = ["keep text".split(), "stale text".split()]
        index.embeddings = np.array(
            [
                np.array([1.0] * 512, dtype=np.float32),
                np.array([2.0] * 512, dtype=np.float32),
            ]
        )

        reused = idxmod.Turn(
            session_id="s1",
            turn_number=0,
            timestamp="2026-01-01T00:00:00Z",
            project="p",
            branch="main",
            slug="s1",
            source="claude",
            user_content="question-keep",
            assistant_content=[],
            tool_results=[],
            summary="keep",
            signals=idxmod.TurnSignals(),
            records=[],
            search_text="keep text",
            source_file="/tmp/stale.jsonl",
        )

        refreshed = idxmod.Turn(
            session_id="s1",
            turn_number=1,
            timestamp="2026-01-01T00:01:00Z",
            project="p",
            branch="main",
            slug="s1",
            source="claude",
            user_content="question-refresh",
            assistant_content=[],
            tool_results=[],
            summary="refresh",
            signals=idxmod.TurnSignals(),
            records=[],
            search_text="refresh text",
            source_file="/tmp/stale.jsonl",
        )

        fake_idx = idxmod.Index()
        fake_convo = idxmod.Conversation(
            session_id="s1",
            project="p",
            branch="main",
            slug="s1",
            summary="refresh",
            first_timestamp="2026-01-01T00:01:00Z",
            last_timestamp="2026-01-01T00:01:00Z",
            source="claude",
        )
        fake_convo.turns = [reused, refreshed]
        fake_idx.turns = [reused, refreshed]
        fake_idx.corpus = ["keep text".split(), "refresh text".split()]
        fake_idx.conversations["s1"] = fake_convo

        def fake_build_claude_index_from_files(jsonl_files):
            return fake_idx, 2

        embed_calls: dict[str, int] = {"count": 0, "shapes": []}

        def fake_embed_texts(texts: list[str]) -> np.ndarray:
            embed_calls["count"] = len(texts)
            return np.array([np.array([9.0] * 512, dtype=np.float32) for _ in texts], dtype=np.float32)

        def fake_build_codex_index_from_files(jsonl_files):
            return idxmod.Index(), 0

        monkeypatch.setattr(idxmod, "build_claude_index_from_files", fake_build_claude_index_from_files)
        monkeypatch.setattr(idxmod, "build_codex_index_from_files", fake_build_codex_index_from_files)
        monkeypatch.setattr(idxmod, "embed_texts", fake_embed_texts)

        rebuilt = idxmod.rebuild_index_incrementally(
            index,
            changed_files={"/tmp/stale.jsonl"},
            transcripts_dir=transcripts_dir,
            codex_dir=codex_dir,
        )

        assert rebuilt is not None
        assert embed_calls["count"] == 1
        assert rebuilt is not None
        assert len(rebuilt.turns) == 2
        assert np.allclose(rebuilt.embeddings[0], np.array([1.0] * 512, dtype=np.float32))
        assert np.allclose(rebuilt.embeddings[1], np.array([9.0] * 512, dtype=np.float32))

def test_extract_subcommand_delegates_to_extract_cli(monkeypatch, tmp_path):
    calls: dict[str, object] = {}

    def fake_run_extract(
        *,
        transcripts_dir,
        codex_dir,
        tool_filter: str = "",
        category_filter: str = "",
        project_filter: str = "",
        session_filter: str = "",
        errors_only: bool = False,
        stats: bool = False,
        limit: int = 0,
    ) -> None:
        calls["transcripts_dir"] = transcripts_dir
        calls["codex_dir"] = codex_dir
        calls["tool_filter"] = tool_filter
        calls["stats"] = stats
        calls["limit"] = limit

    monkeypatch.setattr("alzabo.extract_cli.run_extract", fake_run_extract)
    parser = build_parser()
    args = parser.parse_args(["extract", "--tool", "Bash", "--extract-limit", "5"])

    import alzabo.main_cli as main_cli

    main_cli.cmd_extract(args)

    assert calls["tool_filter"] == "Bash"
    assert calls["limit"] == 5


class TestStatusSubcommand:
    def test_status_text_output(self, tmp_path, monkeypatch, capsys):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        parser = build_parser()
        args = parser.parse_args([
            "status", "--no-cache",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        from alzabo.main_cli import cmd_status
        cmd_status(args)

        captured = capsys.readouterr()
        assert "alzabo status" in captured.out

    def test_status_json_output(self, tmp_path, monkeypatch, capsys):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        parser = build_parser()
        args = parser.parse_args([
            "status", "--no-cache", "--format", "json",
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        from alzabo.main_cli import cmd_status
        cmd_status(args)

        captured = capsys.readouterr()
        d = json.loads(captured.out)
        assert "total_turns" in d
