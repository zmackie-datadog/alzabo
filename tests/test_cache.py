from __future__ import annotations

import json
import time

import numpy as np
import pytest

import alzabo.cache as cache_mod
import alzabo.index as idxmod


def _make_index(turn_count: int = 3) -> idxmod.Index:
    idx = idxmod.Index()
    convo = idxmod.Conversation(
        session_id="s1", project="proj", branch="main", slug="test",
        source="claude", summary="test", first_timestamp="2026-01-01T00:00:00Z",
        last_timestamp="2026-01-01T01:00:00Z",
    )
    for i in range(turn_count):
        turn = idxmod.Turn(
            session_id="s1", turn_number=i, timestamp=f"2026-01-01T00:0{i}:00Z",
            project="proj", branch="main", slug="test", source="claude",
            user_content=f"question {i}", assistant_content=[{"type": "text", "text": f"answer {i}"}],
            tool_results=[], summary=f"turn {i}", signals=idxmod.TurnSignals(tools=["Read"] if i == 0 else []),
            records=[], search_text=f"question {i} answer {i}",
        )
        convo.turns.append(turn)
        idx.turns.append(turn)
        idx.corpus.append(turn.search_text.lower().split())

    idx.conversations["s1"] = convo
    idx.build()
    idx.embeddings = np.random.randn(turn_count, 512).astype(np.float32)
    return idx


@pytest.fixture(autouse=True)
def override_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_mod, "CACHE_DIR", tmp_path / "cache")


class TestCacheRoundtrip:
    def test_save_load_roundtrip(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        original = _make_index(3)
        cache_mod.save_cache(original, transcripts, codex)
        loaded = cache_mod.load_cache()

        assert loaded is not None
        assert len(loaded.turns) == len(original.turns)
        assert len(loaded.conversations) == len(original.conversations)
        assert loaded.bm25 is not None

        # Verify turn content preserved
        for orig_t, load_t in zip(original.turns, loaded.turns):
            assert orig_t.session_id == load_t.session_id
            assert orig_t.turn_number == load_t.turn_number
            assert orig_t.summary == load_t.summary
            assert orig_t.user_content == load_t.user_content
            assert orig_t.signals.tools == load_t.signals.tools

    def test_embeddings_shape_preserved(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        original = _make_index(5)
        cache_mod.save_cache(original, transcripts, codex)
        loaded = cache_mod.load_cache()

        assert loaded is not None
        assert loaded.embeddings.shape == original.embeddings.shape
        np.testing.assert_allclose(loaded.embeddings, original.embeddings, atol=1e-6)


class TestCacheStaleness:
    def test_fresh_cache(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()
        jsonl = transcripts / "test.jsonl"
        jsonl.write_text('{"type":"user"}\n')

        idx = _make_index(1)
        cache_mod.save_cache(idx, transcripts, codex)
        assert cache_mod.is_cache_fresh(transcripts, codex) is True

    def test_stale_after_mtime_change(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()
        jsonl = transcripts / "test.jsonl"
        jsonl.write_text('{"type":"user"}\n')

        idx = _make_index(1)
        cache_mod.save_cache(idx, transcripts, codex)

        # Touch the file to change mtime
        time.sleep(0.05)
        jsonl.write_text('{"type":"user"}\n{"type":"assistant"}\n')

        assert cache_mod.is_cache_fresh(transcripts, codex) is False

    def test_stale_after_new_file(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()
        jsonl = transcripts / "test.jsonl"
        jsonl.write_text('{"type":"user"}\n')

        idx = _make_index(1)
        cache_mod.save_cache(idx, transcripts, codex)

        # Add a new file
        new_jsonl = transcripts / "new.jsonl"
        new_jsonl.write_text('{"type":"user"}\n')

        assert cache_mod.is_cache_fresh(transcripts, codex) is False

    def test_no_cache_dir(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()
        assert cache_mod.is_cache_fresh(transcripts, codex) is False


class TestCacheCorruption:
    def test_corrupt_turns_json(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "turns.json").write_text("not valid json{{{")
        (cache_dir / "embeddings.npy").write_bytes(b"")
        (cache_dir / "manifest.json").write_text("{}")

        result = cache_mod.load_cache()
        assert result is None

    def test_missing_embeddings(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "turns.json").write_text("[]")
        # No embeddings.npy

        result = cache_mod.load_cache()
        assert result is None


class TestSetIndex:
    def test_set_index_makes_ready(self):
        manager = idxmod.TranscriptIndexManager()
        assert not manager._index_ready.is_set()

        idx = _make_index(2)
        manager.set_index(idx)

        assert manager._index_ready.is_set()
        status = manager.get_index_status()
        assert status.total_turns == 2
        assert status.last_reindex_at != ""

    def test_init_accepts_prebuilt_index(self):
        idx = _make_index(2)
        manager = idxmod.TranscriptIndexManager(index=idx)
        status = manager.get_index_status()
        assert status.total_turns == 2
        assert status.total_sessions == 1
