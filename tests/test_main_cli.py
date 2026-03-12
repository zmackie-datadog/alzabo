from __future__ import annotations

import json

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
