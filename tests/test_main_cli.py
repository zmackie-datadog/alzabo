from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from click.testing import CliRunner

import alzabo.cache as cache_mod
import alzabo.index as idxmod
from alzabo.main_cli import _flush_deferred_update, _load_manager, cli


def _build_args(**kwargs):
    defaults = {
        "quiet": False,
        "cache_dir": "",
        "transcripts_dir": str(Path.home() / ".claude" / "projects"),
        "codex_dir": str(Path.home() / ".codex" / "sessions"),
        "no_cache": False,
        "format": "text",
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_search_help(runner):
    result = runner.invoke(cli, ["search", "--help"])

    assert result.exit_code == 0
    assert "Search transcripts." in result.output
    assert "Usage: cli search [OPTIONS] QUERY" in result.output
    assert "bm25" in result.output


def test_list_help(runner):
    result = runner.invoke(cli, ["list", "--help"])

    assert result.exit_code == 0
    assert "List conversations." in result.output


def test_read_help(runner):
    result = runner.invoke(cli, ["read", "--help"])

    assert result.exit_code == 0
    assert "Read a conversation or turn." in result.output
    assert "SESSION_ID" in result.output


def test_status_default_format(runner):
    result = runner.invoke(cli, ["status", "--help"])

    assert result.exit_code == 0
    assert "Show index status." in result.output
    assert "default: text" in result.output


def test_global_format_jsonl(runner):
    result = runner.invoke(cli, ["list", "--help"])

    assert result.exit_code == 0
    assert "jsonl" in result.output


def test_global_quiet_option(runner):
    result = runner.invoke(cli, ["status", "--help"])

    assert result.exit_code == 0
    assert "--quiet" in result.output


def test_global_cache_dir_option(runner):
    result = runner.invoke(cli, ["status", "--help"])

    assert result.exit_code == 0
    assert "--cache-dir" in result.output


class TestCacheConfig:
    def test_cache_dir_env(self, tmp_path, monkeypatch, runner):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        monkeypatch.setenv("ALZABO_CACHE_DIR", str(tmp_path / "from-env"))
        _load_manager(
            _build_args(
                no_cache=True,
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert cache_mod.get_cache_dir() == (tmp_path / "from-env").resolve()


class TestCacheBypass:
    @pytest.fixture(autouse=True)
    def override_cache_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
        import alzabo.main_cli as main_cli

        main_cli._PENDING_UPDATE = None

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

        _load_manager(
            _build_args(
                no_cache=True,
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert len(reindex_calls) == 1

    def test_fresh_cache_skips_reindex(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

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

        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
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

        monkeypatch.setattr(idxmod, "embed_texts", fake_embed_texts)
        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert embed_calls == []

    def test_stale_cache_sets_deferred_update(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        old_file = transcripts / "old.jsonl"
        old_file.write_text('{"type":"user","sessionId":"old-session","message":{"content":"old query"}}\n')

        idx = idxmod.Index()
        convo = idxmod.Conversation(session_id="old-session", project="proj", branch="main", slug="old", source="claude")
        old_turn = idxmod.Turn(
            session_id="old-session",
            turn_number=0,
            timestamp="2026-01-01T00:00:00Z",
            project="proj",
            branch="main",
            slug="old",
            source="claude",
            user_content="old query",
            assistant_content=[],
            tool_results=[],
            summary="old query",
            signals=idxmod.TurnSignals(),
            records=[],
            search_text="old query",
            source_file=str(old_file),
        )
        convo.turns.append(old_turn)
        idx.turns.append(old_turn)
        idx.conversations["old-session"] = convo
        idx.corpus.append(["old", "query"])
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)
        cache_mod.touch_cache_checked_at(transcripts, codex, checked_at="2026-01-01T00:00:00Z")

        new_file = transcripts / "new.jsonl"
        new_file.write_text('{"type":"user","sessionId":"new-session","message":{"content":"new query"}}\n')

        import alzabo.main_cli as main_cli

        rebuild_calls: list[tuple[set[str], bool]] = []
        original_rebuild = idxmod.rebuild_index_incrementally

        def tracking_rebuild(index, changed_files, transcripts_dir, codex_dir, skip_embeddings=False):
            rebuild_calls.append((set(changed_files), skip_embeddings))
            return original_rebuild(
                index,
                changed_files,
                transcripts_dir,
                codex_dir,
                skip_embeddings=skip_embeddings,
            )

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", tracking_rebuild)
        manager = main_cli._load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert manager.get_index_status().total_turns == 1
        assert len(rebuild_calls) == 0
        assert main_cli._PENDING_UPDATE is not None
        assert main_cli._PENDING_UPDATE.manifest.get("source_files", {})

    def test_stale_cache_no_changes_returns_cached_index(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        idx = idxmod.Index()
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)
        cache_mod.touch_cache_checked_at(transcripts, codex, checked_at="2026-01-01T00:00:00Z")

        rebuild_calls: list[bool] = []
        original_rebuild = idxmod.rebuild_index_incrementally

        def tracking_rebuild(index, changed_files, transcripts_dir, codex_dir, skip_embeddings=False):
            rebuild_calls.append(True)
            return original_rebuild(
                index,
                changed_files,
                transcripts_dir,
                codex_dir,
                skip_embeddings=skip_embeddings,
            )

        reindex_calls = []
        original_reindex = idxmod.TranscriptIndexManager.reindex

        def tracking_reindex(self):
            reindex_calls.append(True)
            return original_reindex(self)

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", tracking_rebuild)
        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", tracking_reindex)

        import alzabo.main_cli as main_cli

        manager = _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert len(rebuild_calls) == 0
        assert len(reindex_calls) == 0
        assert main_cli._PENDING_UPDATE is not None
        assert manager.get_index_status().total_turns == 0

    def test_cache_debounce_skips_file_scan_within_window(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        idx = idxmod.Index()
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)
        cache_mod.touch_cache_checked_at(transcripts, codex)

        (transcripts / "new.jsonl").write_text('{"type":"user","sessionId":"new-session","message":{"content":"new query"}}\n')

        rebuild_calls: list[bool] = []
        original_rebuild = idxmod.rebuild_index_incrementally

        def tracking_rebuild(index, changed_files, transcripts_dir, codex_dir, skip_embeddings=False):
            rebuild_calls.append(True)
            return original_rebuild(
                index,
                changed_files,
                transcripts_dir,
                codex_dir,
                skip_embeddings=skip_embeddings,
            )

        reindex_calls = []
        original_reindex = idxmod.TranscriptIndexManager.reindex

        def tracking_reindex(self):
            reindex_calls.append(True)
            return original_reindex(self)

        import alzabo.main_cli as main_cli

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", tracking_rebuild)
        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", tracking_reindex)

        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert main_cli._PENDING_UPDATE is None
        assert len(rebuild_calls) == 0
        assert len(reindex_calls) == 0

    def test_flush_deferred_update_rebuilds_incrementally(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        old_file = transcripts / "old.jsonl"
        old_file.write_text('{"type":"user","sessionId":"old-session","message":{"content":"old query"}}\n')

        idx = idxmod.Index()
        convo = idxmod.Conversation(session_id="old-session", project="proj", branch="main", slug="old", source="claude")
        old_turn = idxmod.Turn(
            session_id="old-session",
            turn_number=0,
            timestamp="2026-01-01T00:00:00Z",
            project="proj",
            branch="main",
            slug="old",
            source="claude",
            user_content="old query",
            assistant_content=[],
            tool_results=[],
            summary="old query",
            signals=idxmod.TurnSignals(),
            records=[],
            search_text="old query",
            source_file=str(old_file),
        )
        convo.turns.append(old_turn)
        idx.turns.append(old_turn)
        idx.conversations["old-session"] = convo
        idx.corpus.append(["old", "query"])
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)
        cache_mod.touch_cache_checked_at(transcripts, codex, checked_at="2026-01-01T00:00:00Z")

        new_file = transcripts / "new.jsonl"
        new_file.write_text('{"type":"user","sessionId":"new-session","message":{"content":"new query"}}\n')

        import alzabo.main_cli as main_cli

        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert main_cli._PENDING_UPDATE is not None

        rebuild_calls: list[tuple[set[str], bool]] = []
        original_rebuild = idxmod.rebuild_index_incrementally

        def tracking_rebuild(index, changed_files, transcripts_dir, codex_dir, skip_embeddings=False):
            rebuild_calls.append((set(changed_files), skip_embeddings))
            return original_rebuild(
                index,
                changed_files,
                transcripts_dir,
                codex_dir,
                skip_embeddings=skip_embeddings,
            )

        save_calls: list[tuple[int]] = []
        original_save_cache = cache_mod.save_cache

        def tracking_save_cache(index, transcripts_dir, codex_dir, reindex_at: str | None = None):
            save_calls.append((len(index.turns), str(reindex_at)))
            return original_save_cache(index, transcripts_dir, codex_dir, reindex_at=reindex_at)

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", tracking_rebuild)
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 512), dtype=np.float32))
        monkeypatch.setattr(cache_mod, "save_cache", tracking_save_cache)

        _flush_deferred_update()

        assert len(rebuild_calls) == 1
        assert rebuild_calls[0][0] == {str(new_file.resolve())}
        assert rebuild_calls[0][1] is False
        assert len(save_calls) == 1
        assert save_calls[0][0] == 2
        cached_index, _ = cache_mod.load_cache_bundle()
        assert cached_index is not None
        assert len(cached_index.turns) == 2

    def test_flush_deferred_update_touches_cache_when_no_changes(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        idx = idxmod.Index()
        idx.build()
        idx.embeddings = np.empty((0, 512), dtype=np.float32)
        cache_mod.save_cache(idx, transcripts, codex)
        cache_mod.touch_cache_checked_at(transcripts, codex, checked_at="2026-01-01T00:00:00Z")

        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        import alzabo.main_cli as main_cli

        rebuild_calls: list[tuple[set[str], bool]] = []
        touch_calls: list[tuple[str, str]] = []

        def tracking_rebuild(index, changed_files, transcripts_dir, codex_dir, skip_embeddings=False):
            rebuild_calls.append((set(changed_files), skip_embeddings))
            return idxmod.rebuild_index_incrementally(
                index,
                changed_files,
                transcripts_dir,
                codex_dir,
                skip_embeddings=skip_embeddings,
            )

        original_touch_cache_checked_at = cache_mod.touch_cache_checked_at

        def tracking_touch_cache_checked_at(transcripts_dir: Path, codex_dir: Path, checked_at: str | None = None) -> None:
            touch_calls.append((str(transcripts_dir), str(codex_dir), str(checked_at)))
            return original_touch_cache_checked_at(transcripts_dir, codex_dir, checked_at=checked_at)

        monkeypatch.setattr(idxmod, "rebuild_index_incrementally", tracking_rebuild)
        monkeypatch.setattr(cache_mod, "touch_cache_checked_at", tracking_touch_cache_checked_at)

        assert main_cli._PENDING_UPDATE is not None
        _flush_deferred_update()

        assert len(rebuild_calls) == 0
        assert len(touch_calls) == 1

    def test_fresh_cache_has_no_deferred_update(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        idx = idxmod.Index()
        idx.build()
        cache_mod.save_cache(idx, transcripts, codex)

        import alzabo.main_cli as main_cli

        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert main_cli._PENDING_UPDATE is None

    def test_cold_start_reindexes_synchronously(self, tmp_path, monkeypatch):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        reindex_calls: list[bool] = []
        original_reindex = idxmod.TranscriptIndexManager.reindex

        def tracking_reindex(self):
            reindex_calls.append(True)
            return original_reindex(self)

        monkeypatch.setattr(idxmod.TranscriptIndexManager, "reindex", tracking_reindex)

        _load_manager(
            _build_args(
                transcripts_dir=str(transcripts),
                codex_dir=str(codex),
            )
        )
        assert len(reindex_calls) == 1
        import alzabo.main_cli as main_cli
        assert main_cli._PENDING_UPDATE is None


def test_extract_subcommand_delegates_to_extract_cli(monkeypatch, tmp_path, runner):
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
    transcripts = tmp_path / "claude"
    codex = tmp_path / "codex"
    transcripts.mkdir()
    codex.mkdir()
    result = runner.invoke(
        cli,
        [
            "extract",
            "--tool",
            "Bash",
            "--extract-limit",
            "5",
            "--transcripts-dir",
            str(transcripts),
            "--codex-dir",
            str(codex),
        ],
    )

    assert result.exit_code == 0
    assert calls["tool_filter"] == "Bash"
    assert calls["limit"] == 5


class TestStatusSubcommand:
    def test_status_text_output(self, tmp_path, monkeypatch, runner):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        result = runner.invoke(
            cli,
            [
                "status",
                "--no-cache",
                "--transcripts-dir",
                str(transcripts),
                "--codex-dir",
                str(codex),
            ],
        )

        assert result.exit_code == 0
        assert "alzabo status" in result.output

    def test_status_json_output(self, tmp_path, monkeypatch, runner):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 3), dtype=np.float32))

        result = runner.invoke(
            cli,
            [
                "status",
                "--no-cache",
                "--quiet",
                "--format",
                "json",
                "--transcripts-dir",
                str(transcripts),
                "--codex-dir",
                str(codex),
            ],
        )

        assert result.exit_code == 0
        d = json.loads(result.output)
        assert "total_turns" in d


class TestReindexSubcommand:
    def test_reindex_rebuilds_and_saves(self, tmp_path, monkeypatch, runner):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(idxmod, "embed_texts", lambda texts: np.ones((len(texts), 512), dtype=np.float32))

        result = runner.invoke(
            cli,
            [
                "reindex",
                "--transcripts-dir",
                str(transcripts),
                "--codex-dir",
                str(codex),
            ],
        )
        assert result.exit_code == 0

        # Cache should now exist
        assert (tmp_path / "cache" / "index.pkl").exists()
        assert (tmp_path / "cache" / "manifest.json").exists()


@pytest.fixture
def runner():
    return CliRunner()
