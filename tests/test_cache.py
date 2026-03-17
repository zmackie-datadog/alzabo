from __future__ import annotations

import json
import time
from datetime import datetime

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
            source_file="/tmp/test.jsonl",
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

        # Slim cache strips content but preserves metadata
        for orig_t, load_t in zip(original.turns, loaded.turns):
            assert orig_t.session_id == load_t.session_id
            assert orig_t.turn_number == load_t.turn_number
            assert orig_t.summary == load_t.summary
            assert orig_t.signals.tools == load_t.signals.tools
            assert load_t.source_file == orig_t.source_file
            # Content is stripped in slim cache
            assert load_t.user_content is None
            assert load_t.assistant_content == []
            assert load_t.search_text == ""

    def test_load_cache_bundle(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        original = _make_index(3)
        cache_mod.save_cache(original, transcripts, codex, reindex_at="2026-01-01T00:00:00Z")
        bundle = cache_mod.load_cache_bundle()

        assert bundle is not None
        loaded, manifest = bundle
        assert loaded is not None
        assert manifest["version"] == cache_mod.CACHE_VERSION
        assert manifest["reindex_at"] == "2026-01-01T00:00:00Z"
        assert manifest["turn_count"] == 3

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

    def test_bm25_prebuilt_in_cache(self, tmp_path):
        """BM25 should be pre-built in the pickle, no rebuild needed."""
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        original = _make_index(3)
        cache_mod.save_cache(original, transcripts, codex)

        bundle = cache_mod.load_cache_bundle()
        assert bundle is not None
        loaded, _ = bundle

        # BM25 should be ready to use without calling build()
        assert loaded.bm25 is not None
        # "answer 0" only appears in turn 0, so scores should be non-uniform
        scores = loaded.bm25.get_scores(["answer", "0"])
        assert scores[0] != scores[1], f"BM25 should discriminate: {scores}"

    def test_pickle_format_used(self, tmp_path):
        """Verify cache uses index.pkl not turns.json."""
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        original = _make_index(1)
        cache_mod.save_cache(original, transcripts, codex)

        assert (cache_mod.CACHE_DIR / "index.pkl").exists()
        assert (cache_mod.CACHE_DIR / "embeddings.npy").exists()
        assert not (cache_mod.CACHE_DIR / "turns.json").exists()


class TestCacheManifest:
    def test_changed_source_files_detects_updates(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        file_a = transcripts / "a.jsonl"
        file_b = transcripts / "b.jsonl"
        file_a.write_text("alpha")
        baseline = cache_mod.collect_source_files(transcripts, codex)

        same = cache_mod.collect_source_files(transcripts, codex)
        assert cache_mod.changed_source_files(baseline, same) == set()

        # Modify a file: mtime or size changes.
        file_a.write_text("alpha\nmore")
        updated = cache_mod.collect_source_files(transcripts, codex)
        assert cache_mod.changed_source_files(baseline, updated) == {str(file_a.resolve())}

        # Add a file.
        file_b.write_text("beta")
        added = cache_mod.collect_source_files(transcripts, codex)
        assert cache_mod.changed_source_files(updated, added) == {str(file_b.resolve())}

        # Remove a file.
        file_a.unlink()
        removed = cache_mod.collect_source_files(transcripts, codex)
        assert cache_mod.changed_source_files(added, removed) == {str(file_a.resolve())}

    def test_partition_changed_files_by_stability(self):
        changed_files = {
            "/tmp/settled.jsonl",
            "/tmp/unstable.jsonl",
            "/tmp/missing.jsonl",
            "/tmp/non_dict.jsonl",
        }
        current_files = {
            "/tmp/settled.jsonl": {"mtime": 100.0, "size": 1},
            "/tmp/unstable.jsonl": {"mtime": 199.0, "size": 1},
            "/tmp/non_dict.jsonl": "legacy-signature",
        }

        settled, unstable = cache_mod.partition_changed_files_by_stability(
            changed_files=changed_files,
            current_files=current_files,
            debounce_seconds=10.0,
            now=200.0,
        )
        assert settled == {"/tmp/settled.jsonl", "/tmp/missing.jsonl", "/tmp/non_dict.jsonl"}
        assert unstable == {"/tmp/unstable.jsonl"}

    def test_partition_changed_files_by_stability_disabled(self):
        changed_files = {"/tmp/fast.jsonl"}
        current_files = {"/tmp/fast.jsonl": {"mtime": 200.0, "size": 10}}
        settled, unstable = cache_mod.partition_changed_files_by_stability(
            changed_files=changed_files,
            current_files=current_files,
            debounce_seconds=0.0,
            now=200.5,
        )
        assert settled == {"/tmp/fast.jsonl"}
        assert unstable == set()

    def test_cache_manifest_includes_file_signature(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        file_a = transcripts / "a.jsonl"
        file_a.write_text("alpha")

        idx = _make_index(1)
        cache_mod.save_cache(idx, transcripts, codex)
        manifest = json.loads((cache_mod.CACHE_DIR / "manifest.json").read_text())
        _, file_meta = manifest["source_files"].popitem()
        assert "mtime" in file_meta
        assert "size" in file_meta

    def test_cache_checked_timestamp_refreshes_without_rewrite(self, tmp_path):
        transcripts = tmp_path / "claude"
        codex = tmp_path / "codex"
        transcripts.mkdir()
        codex.mkdir()

        idx = _make_index(1)
        cache_mod.save_cache(idx, transcripts, codex)

        manifest_path = cache_mod.CACHE_DIR / "manifest.json"
        index_path = cache_mod.CACHE_DIR / "index.pkl"
        manifest_early = json.loads(manifest_path.read_text())
        index_bytes_early = index_path.read_bytes()
        first_checked_at = manifest_early["cache_checked_at"]

        cache_mod.touch_cache_checked_at(transcripts, codex, checked_at="2026-01-01T00:00:01Z")

        manifest_late = json.loads(manifest_path.read_text())
        index_bytes_late = index_path.read_bytes()
        assert manifest_late["cache_checked_at"] == "2026-01-01T00:00:01Z"
        assert manifest_late["cache_checked_at"] != first_checked_at
        assert index_bytes_late == index_bytes_early

    def test_cache_recently_checked(self):
        manifest = {"cache_checked_at": "2026-01-01T00:00:00Z"}
        checked = cache_mod.is_cache_recently_checked(
            manifest,
            30.0,
            now=datetime.fromisoformat("2026-01-01T00:00:20+00:00"),
        )
        assert checked is True

        stale = cache_mod.is_cache_recently_checked(
            manifest,
            10.0,
            now=datetime.fromisoformat("2026-01-01T00:00:20+00:00"),
        )
        assert stale is False

        never = cache_mod.is_cache_recently_checked(
            manifest, 10.0, now=datetime.fromisoformat("2025-01-01T00:00:00+00:00")
        )
        assert never is False

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
    def test_corrupt_index_pkl(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "index.pkl").write_text("not valid pickle{{{")
        (cache_dir / "embeddings.npy").write_bytes(b"")
        (cache_dir / "manifest.json").write_text(json.dumps({"version": cache_mod.CACHE_VERSION}))

        result = cache_mod.load_cache()
        assert result is None

    def test_missing_embeddings(self, tmp_path):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        (cache_dir / "index.pkl").write_bytes(b"")
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


class TestSlimIndex:
    def test_slim_turn_strips_content(self):
        turn = idxmod.Turn(
            session_id="s1", turn_number=0, timestamp="2026-01-01T00:00:00Z",
            project="proj", branch="main", slug="test", source="claude",
            user_content="big question", assistant_content=[{"type": "text", "text": "big answer"}],
            tool_results=[{"output": "big result"}], summary="short summary",
            signals=idxmod.TurnSignals(tools=["Read"]), records=[{"some": "record"}],
            search_text="big question big answer", source_file="/tmp/test.jsonl",
        )
        slim = cache_mod._slim_turn(turn)
        assert slim.user_content is None
        assert slim.assistant_content == []
        assert slim.tool_results == []
        assert slim.records == []
        assert slim.search_text == ""
        assert slim.summary == "short summary"
        assert slim.signals.tools == ["Read"]
        assert slim.source_file == "/tmp/test.jsonl"

    def test_slim_index_preserves_structure(self):
        idx = _make_index(3)
        slim = cache_mod._slim_index(idx)
        assert len(slim.turns) == 3
        assert len(slim.conversations) == 1
        assert slim.bm25 is not None
        assert slim.corpus == []  # corpus stripped
