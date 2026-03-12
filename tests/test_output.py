from __future__ import annotations

import json

import alzabo.index as idxmod
from alzabo.output import (
    format_conversation,
    format_conversation_page,
    format_index_status,
    format_search_results,
    format_session_results,
    format_turn,
)
from alzabo.render import (
    render_conversation,
    render_index_status,
    render_list_conversations,
    render_search_conversations,
    render_search_sessions,
    render_turn,
)


def _make_turn(**kwargs) -> idxmod.Turn:
    defaults = dict(
        session_id="s1",
        turn_number=0,
        timestamp="2026-01-01T00:00:00Z",
        project="proj",
        branch="main",
        slug="test",
        source="claude",
        user_content="hello",
        assistant_content=[],
        tool_results=[],
        summary="hello world",
        signals=idxmod.TurnSignals(),
        records=[],
        search_text="hello world",
    )
    defaults.update(kwargs)
    return idxmod.Turn(**defaults)


def _make_convo(**kwargs) -> idxmod.Conversation:
    defaults = dict(
        session_id="s1",
        project="proj",
        branch="main",
        slug="test",
        source="claude",
        summary="test convo",
        first_timestamp="2026-01-01T00:00:00Z",
        last_timestamp="2026-01-01T01:00:00Z",
        turns=[_make_turn()],
    )
    defaults.update(kwargs)
    return idxmod.Conversation(**defaults)


class TestFormatSearchResults:
    def _make_result(self) -> idxmod.SearchResultSet:
        turn = _make_turn()
        item = idxmod.TurnSearchResult(turn=turn, score=0.75)
        return idxmod.SearchResultSet(query="test", mode="bm25", effective_mode="bm25", items=[item])

    def test_json_parses(self):
        result = self._make_result()
        out = format_search_results(result, "json")
        d = json.loads(out)
        assert d["query"] == "test"
        assert d["result_count"] == 1

    def test_jsonl_lines_parse(self):
        result = self._make_result()
        out = format_search_results(result, "jsonl")
        lines = out.strip().split("\n")
        assert len(lines) == 1
        d = json.loads(lines[0])
        assert d["score"] == 0.75

    def test_text_matches_render(self):
        result = self._make_result()
        assert format_search_results(result, "text") == render_search_conversations(result)

    def test_empty_result_set(self):
        result = idxmod.SearchResultSet(query="q", mode="bm25", effective_mode="bm25", items=[])
        assert json.loads(format_search_results(result, "json"))["result_count"] == 0
        assert format_search_results(result, "jsonl") == ""


class TestFormatSessionResults:
    def _make_result(self) -> idxmod.SessionResultSet:
        convo = _make_convo()
        item = idxmod.SessionSearchResult(
            conversation=convo, best_score=0.9, best_turn_number=0,
            best_turn_summary="summary", matching_turns=2,
        )
        return idxmod.SessionResultSet(query="q", mode="hybrid", effective_mode="hybrid", items=[item])

    def test_json_parses(self):
        d = json.loads(format_session_results(self._make_result(), "json"))
        assert d["result_count"] == 1

    def test_jsonl_lines_parse(self):
        out = format_session_results(self._make_result(), "jsonl")
        d = json.loads(out.strip())
        assert d["matching_turns"] == 2

    def test_text_matches_render(self):
        r = self._make_result()
        assert format_session_results(r, "text") == render_search_sessions(r)


class TestFormatConversationPage:
    def _make_page(self, next_offset=None) -> idxmod.ConversationPage:
        return idxmod.ConversationPage(
            items=[_make_convo()], total=5, offset=0, end=1, next_offset=next_offset,
        )

    def test_json_parses(self):
        d = json.loads(format_conversation_page(self._make_page(next_offset=1), "json"))
        assert d["total"] == 5
        assert d["next_offset"] == 1

    def test_jsonl_lines_parse(self):
        out = format_conversation_page(self._make_page(), "jsonl")
        d = json.loads(out.strip())
        assert d["session_id"] == "s1"

    def test_text_matches_render(self):
        page = self._make_page()
        assert format_conversation_page(page, "text") == render_list_conversations(page)

    def test_empty_page(self):
        page = idxmod.ConversationPage(items=[], total=0, offset=0, end=0, next_offset=None)
        d = json.loads(format_conversation_page(page, "json"))
        assert d["total"] == 0
        assert format_conversation_page(page, "jsonl") == ""


class TestFormatConversation:
    def test_json_parses(self):
        convo = _make_convo()
        d = json.loads(format_conversation(convo, "json"))
        assert d["session_id"] == "s1"
        assert len(d["turns"]) == 1

    def test_jsonl_parses(self):
        convo = _make_convo()
        out = format_conversation(convo, "jsonl")
        d = json.loads(out.strip())
        assert d["session_id"] == "s1"

    def test_text_matches_render(self):
        convo = _make_convo()
        assert format_conversation(convo, "text") == render_conversation(convo)


class TestFormatTurn:
    def test_json_parses(self):
        turn = _make_turn()
        d = json.loads(format_turn(turn, "json"))
        assert d["session_id"] == "s1"

    def test_text_matches_render(self):
        turn = _make_turn()
        assert format_turn(turn, "text") == render_turn(turn)


class TestFormatIndexStatus:
    def _make_status(self) -> idxmod.IndexStatus:
        return idxmod.IndexStatus(
            transcripts_dir="/tmp/c", codex_dir="/tmp/x", watch_enabled=False,
            total_sessions=3, claude_sessions=2, codex_sessions=1,
            total_turns=10, embeddings_ready=True, last_reindex_at="2026-01-01T00:00:00Z",
        )

    def test_json_parses(self):
        d = json.loads(format_index_status(self._make_status(), "json"))
        assert d["total_turns"] == 10

    def test_text_matches_render(self):
        s = self._make_status()
        assert format_index_status(s, "text") == render_index_status(s)
