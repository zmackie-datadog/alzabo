from __future__ import annotations

import json
import sys

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

    def test_serve_parses(self):
        parser = build_parser()
        args = parser.parse_args(["serve", "--no-watch"])
        assert args.command == "serve"
        assert args.watch is False

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
            "--transcripts-dir", str(transcripts),
            "--codex-dir", str(codex),
        ])

        from alzabo.main_cli import _get_manager
        _get_manager(args)
        assert len(reindex_calls) == 0


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


class TestLegacyCompatibility:
    def test_watch_only_invokes_serve_with_deprecation_warning(self, monkeypatch, tmp_path, capsys):
        calls: dict[str, object] = {}

        def fake_run_mcp_server(
            *,
            transcripts_dir,
            codex_dir,
            watch: bool = True,
            debounce_seconds: float = 2.0,
            quiet: bool = False,
        ) -> None:
            calls["transcripts_dir"] = transcripts_dir
            calls["codex_dir"] = codex_dir
            calls["watch"] = watch
            calls["debounce_seconds"] = debounce_seconds

        monkeypatch.setattr("alzabo.cli.run_mcp_server", fake_run_mcp_server)
        monkeypatch.setattr(sys, "argv", [
            "alzabo",
            "--watch",
            "--transcripts-dir", str(tmp_path / "c"),
            "--codex-dir", str(tmp_path / "x"),
            "--debounce-seconds", "3",
        ])

        import alzabo.main_cli as main_cli

        main_cli.main()

        captured = capsys.readouterr()
        assert "deprecated" in captured.err.lower()
        assert calls["watch"] is True
        assert calls["debounce_seconds"] == 3.0

    def test_legacy_watch_invocation_respects_quiet(self, monkeypatch, tmp_path, capsys):
        calls: dict[str, object] = {}

        def fake_run_mcp_server(
            *,
            transcripts_dir,
            codex_dir,
            watch: bool = True,
            debounce_seconds: float = 2.0,
            quiet: bool = False,
        ) -> None:
            calls["quiet"] = quiet

        monkeypatch.setattr("alzabo.cli.run_mcp_server", fake_run_mcp_server)
        monkeypatch.setattr(sys, "argv", [
            "alzabo",
            "--watch",
            "--quiet",
            "--transcripts-dir", str(tmp_path / "c"),
            "--codex-dir", str(tmp_path / "x"),
        ])

        import alzabo.main_cli as main_cli

        main_cli.main()

        captured = capsys.readouterr()
        assert "deprecated" in captured.err.lower()
        assert calls["quiet"] is True

    def test_serve_subcommand_delegates_to_cli(self, monkeypatch, tmp_path):
        calls: dict[str, object] = {}

        def fake_run_mcp_server(*, transcripts_dir, codex_dir, watch=True, debounce_seconds=2.0, quiet: bool = False) -> None:
            calls["transcripts_dir"] = transcripts_dir
            calls["codex_dir"] = codex_dir
            calls["watch"] = watch
            calls["quiet"] = quiet

        monkeypatch.setattr("alzabo.cli.run_mcp_server", fake_run_mcp_server)
        monkeypatch.setattr(sys, "argv", [
            "alzabo",
            "serve",
            "--no-watch",
            "--transcripts-dir", str(tmp_path / "claude"),
            "--codex-dir", str(tmp_path / "codex"),
        ])

        import alzabo.main_cli as main_cli

        main_cli.main()

        assert calls["watch"] is False
        assert calls["quiet"] is False

    def test_extract_subcommand_delegates_to_extract_cli(self, monkeypatch, tmp_path):
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
        monkeypatch.setattr(sys, "argv", [
            "alzabo",
            "extract",
            "--tool", "Bash",
            "--extract-limit", "5",
        ])
        import alzabo.main_cli as main_cli

        main_cli.main()

        assert calls["tool_filter"] == "Bash"
        assert calls["limit"] == 5
