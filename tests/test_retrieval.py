"""Tests for chunking and retrieval (unit tests that don't require OpenAI)."""

import pytest

from src.ingestion.parser import parse_transcript, CallTranscript, DialogueTurn
from src.ingestion.chunker import chunk_transcript, Chunk
from src.retrieval.retriever import Retriever, RetrievalResult


def _make_call(num_turns: int = 12, call_id: str = "99") -> CallTranscript:
    """Create a synthetic CallTranscript for testing."""
    turns = []
    for i in range(num_turns):
        mm = i // 60
        ss = i % 60
        turns.append(DialogueTurn(
            timestamp=f"{mm:02d}:{ss:02d}",
            speaker="AE (Jordan)" if i % 2 == 0 else "Prospect (Priya)",
            speaker_name="Jordan" if i % 2 == 0 else "Priya",
            role="AE" if i % 2 == 0 else "Prospect",
            text=f"Turn number {i} with some sample dialogue text.",
        ))

    return CallTranscript(
        call_id=call_id,
        call_type="Test",
        file_name=f"call_{call_id}.txt",
        participants=["Jordan (AE)", "Priya (Prospect)"],
        turns=turns,
        duration=turns[-1].timestamp if turns else "00:00",
    )


# ── Chunking ─────────────────────────────────────────────────────────

class TestChunking:
    def test_basic_chunk_count(self):
        call = _make_call(12)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        assert len(chunks) >= 4

    def test_chunk_overlap_shares_turns(self):
        call = _make_call(10)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        # First chunk ends at turn 4, second starts at turn 3 (overlap=2)
        first_text = chunks[0].text
        second_text = chunks[1].text
        # Turn 3 should appear in both chunks
        assert "Turn number 3" in first_text
        assert "Turn number 3" in second_text

    def test_chunk_has_metadata(self):
        call = _make_call(6)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        for chunk in chunks:
            assert chunk.call_id == "99"
            assert chunk.call_type == "Test"
            assert chunk.start_time
            assert chunk.end_time
            assert len(chunk.speakers) > 0

    def test_empty_call(self):
        call = CallTranscript(
            call_id="0", call_type="Empty", file_name="empty.txt",
            participants=[], turns=[],
        )
        chunks = chunk_transcript(call)
        assert chunks == []

    def test_single_turn(self):
        call = _make_call(1)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        assert len(chunks) == 1

    def test_chunk_text_contains_timestamps(self):
        call = _make_call(5)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        assert "[00:00]" in chunks[0].text

    def test_chunk_index_sequential(self):
        call = _make_call(15)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_speakers_per_chunk(self):
        call = _make_call(6)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        for chunk in chunks:
            assert "Jordan" in chunk.speakers or "Priya" in chunk.speakers

    def test_two_turns_one_chunk(self):
        call = _make_call(2)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        assert len(chunks) == 1
        assert "Turn number 0" in chunks[0].text
        assert "Turn number 1" in chunks[0].text

    def test_exact_chunk_size_no_overlap(self):
        """Exactly turns_per_chunk turns with no overlap -> one chunk."""
        call = _make_call(5)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=0)
        assert len(chunks) == 1

    def test_no_overlap_mode(self):
        call = _make_call(10)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=0)
        assert len(chunks) == 2
        # No shared content
        assert "Turn number 4" in chunks[0].text
        assert "Turn number 4" not in chunks[1].text

    def test_large_overlap(self):
        """Overlap almost equal to chunk size."""
        call = _make_call(8)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=4)
        # step = 1, so many chunks
        assert len(chunks) >= 4

    def test_chunk_start_end_times(self):
        call = _make_call(10)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        for chunk in chunks:
            assert chunk.start_time <= chunk.end_time


# ── Retriever context formatting ─────────────────────────────────────

class TestRetrieverFormatting:
    def test_format_context_empty(self):
        """Empty results should return a no-results message."""
        result = Retriever.format_context(None, [])
        assert "No relevant" in result

    def test_format_context_single(self):
        results = [RetrievalResult(
            text="[00:00] AE: Hello",
            call_id="1", call_type="Demo", chunk_index=0,
            start_time="00:00", end_time="00:05",
            speakers="Jordan", relevance_score=0.95,
        )]
        context = Retriever.format_context(None, results)
        assert "Segment 1" in context
        assert "Call #1" in context
        assert "Demo" in context
        assert "Jordan" in context
        assert "[00:00] AE: Hello" in context

    def test_format_context_multiple(self):
        results = [
            RetrievalResult(
                text="Turn A", call_id="1", call_type="Demo", chunk_index=0,
                start_time="00:00", end_time="00:05",
                speakers="Jordan", relevance_score=0.95,
            ),
            RetrievalResult(
                text="Turn B", call_id="2", call_type="Pricing", chunk_index=3,
                start_time="03:00", end_time="03:10",
                speakers="Maya, Dan", relevance_score=0.85,
            ),
        ]
        context = Retriever.format_context(None, results)
        assert "Segment 1" in context
        assert "Segment 2" in context
        assert "Call #1" in context
        assert "Call #2" in context

    def test_relevance_score_computation(self):
        """RetrievalResult relevance = 1 - distance."""
        r = RetrievalResult(
            text="x", call_id="1", call_type="", chunk_index=0,
            start_time="", end_time="", speakers="",
            relevance_score=1 - 0.3,
        )
        assert abs(r.relevance_score - 0.7) < 0.01


# ── Parser + Chunker integration ────────────────────────────────────

class TestParserChunkerIntegration:
    @pytest.fixture
    def sample_file(self, tmp_path):
        content = (
            "[00:00] AE (Jordan):  Hello.\n\n"
            "[00:05] Prospect (Priya – RevOps Director):  Hi.\n\n"
            "[00:10] AE:  Let me show the demo.\n\n"
            "[00:15] Prospect:  Looks great.\n\n"
            "[00:20] SE (Luis):  Any questions?\n\n"
        )
        fp = tmp_path / "call_5.txt"
        fp.write_text(content, encoding="utf-8")
        return str(fp)

    def test_parse_and_chunk(self, sample_file):
        call = parse_transcript(sample_file)
        chunks = chunk_transcript(call, turns_per_chunk=3, overlap=1)
        assert len(chunks) >= 2
        assert all(c.call_id == "5" for c in chunks)

    def test_chunk_call_type_propagated(self, sample_file):
        call = parse_transcript(sample_file)
        chunks = chunk_transcript(call)
        for chunk in chunks:
            assert chunk.call_type == call.call_type

    def test_all_turns_appear_in_at_least_one_chunk(self, sample_file):
        call = parse_transcript(sample_file)
        chunks = chunk_transcript(call, turns_per_chunk=3, overlap=1)
        all_chunk_text = " ".join(c.text for c in chunks)
        for turn in call.turns:
            assert turn.text in all_chunk_text


# ── Engine action routing (mocked) ──────────────────────────────────

class TestEngineActionRouting:
    """Test _execute_action dispatch without real OpenAI/ChromaDB calls."""

    def _make_engine_stub(self):
        """Create a minimal engine-like object for testing dispatch logic."""
        from unittest.mock import MagicMock
        from src.chatbot.engine import ChatEngine

        engine = object.__new__(ChatEngine)
        engine._store = MagicMock()
        engine._retriever = MagicMock()
        engine._llm = MagicMock()
        return engine

    def test_list_calls_action(self):
        engine = self._make_engine_stub()
        engine._store.list_calls.return_value = [
            {"call_id": "1", "call_type": "Demo", "chunk_count": 10},
        ]
        result = engine._execute_action({"action": "list_calls", "query": ""})
        assert "Call #1" in result
        assert "Demo" in result

    def test_list_calls_empty(self):
        engine = self._make_engine_stub()
        engine._store.list_calls.return_value = []
        result = engine._execute_action({"action": "list_calls", "query": ""})
        assert "No calls" in result

    def test_delete_single_call(self):
        engine = self._make_engine_stub()
        engine._store.delete_call.return_value = 15
        result = engine._execute_action({
            "action": "delete_call", "call_ids": ["3"], "query": "",
        })
        assert "Deleted 15 chunks" in result
        assert "call #3" in result

    def test_delete_multiple_calls(self):
        engine = self._make_engine_stub()
        engine._store.delete_call.side_effect = [10, 8]
        result = engine._execute_action({
            "action": "delete_call", "call_ids": ["1", "6"], "query": "",
        })
        assert "call #1" in result
        assert "call #6" in result

    def test_delete_nonexistent_call(self):
        engine = self._make_engine_stub()
        engine._store.delete_call.return_value = 0
        result = engine._execute_action({
            "action": "delete_call", "call_ids": ["99"], "query": "",
        })
        assert "No data found" in result

    def test_delete_no_call_id(self):
        engine = self._make_engine_stub()
        result = engine._execute_action({
            "action": "delete_call", "call_ids": [], "query": "",
        })
        assert "specify" in result.lower()

    def test_ingest_no_file_path(self):
        engine = self._make_engine_stub()
        result = engine._execute_action({
            "action": "ingest", "file_paths": [], "query": "",
        })
        assert "specify" in result.lower()

    def test_ingest_nonexistent_file(self):
        engine = self._make_engine_stub()
        result = engine._execute_action({
            "action": "ingest",
            "file_paths": ["/nonexistent/path.txt"],
            "query": "",
        })
        assert "File not found" in result

    def test_summarize_action_calls_rag(self):
        engine = self._make_engine_stub()
        engine._retriever.retrieve.return_value = []
        engine._retriever.format_context.return_value = "No results."
        engine._llm.generate.return_value = "Summary here."

        result = engine._execute_action({
            "action": "summarize", "call_ids": ["1"],
            "query": "summarize call 1",
        })
        assert result == "Summary here."
        engine._retriever.retrieve.assert_called_once()

    def test_sentiment_action_calls_rag(self):
        engine = self._make_engine_stub()
        engine._retriever.retrieve.return_value = []
        engine._retriever.format_context.return_value = "No results."
        engine._llm.generate.return_value = "Sentiment analysis."

        result = engine._execute_action({
            "action": "sentiment", "call_ids": [],
            "query": "negative comments about pricing",
        })
        assert result == "Sentiment analysis."

    def test_question_action_calls_rag(self):
        engine = self._make_engine_stub()
        engine._retriever.retrieve.return_value = []
        engine._retriever.format_context.return_value = "No results."
        engine._llm.generate.return_value = "Answer here."

        result = engine._execute_action({
            "action": "question", "call_ids": [],
            "query": "What pricing was discussed?",
        })
        assert result == "Answer here."

    def test_unknown_action_defaults_to_qa(self):
        engine = self._make_engine_stub()
        engine._retriever.retrieve.return_value = []
        engine._retriever.format_context.return_value = ""
        engine._llm.generate.return_value = "Fallback answer."

        result = engine._execute_action({
            "action": "something_new", "query": "test",
        })
        assert result == "Fallback answer."

    def test_multiple_actions_processed(self):
        engine = self._make_engine_stub()
        engine._store.delete_call.return_value = 5
        engine._store.list_calls.return_value = []

        engine._llm.route_query.return_value = [
            {"action": "delete_call", "call_ids": ["1"], "query": ""},
            {"action": "list_calls", "query": ""},
        ]

        result = engine.process_query("delete call 1 and then list calls")
        assert "Deleted" in result
        assert "No calls" in result


# ── Prompt template formatting ───────────────────────────────────────

class TestPromptTemplates:
    def test_qa_prompt_formats(self):
        from src.llm.prompts import QA_PROMPT
        result = QA_PROMPT.format(context="CONTEXT_HERE", query="QUERY_HERE")
        assert "CONTEXT_HERE" in result
        assert "QUERY_HERE" in result

    def test_summary_prompt_formats(self):
        from src.llm.prompts import SUMMARY_PROMPT
        result = SUMMARY_PROMPT.format(context="CONTEXT_HERE")
        assert "CONTEXT_HERE" in result

    def test_sentiment_prompt_formats(self):
        from src.llm.prompts import SENTIMENT_PROMPT
        result = SENTIMENT_PROMPT.format(context="CONTEXT_HERE", query="QUERY_HERE")
        assert "CONTEXT_HERE" in result
        assert "QUERY_HERE" in result

    def test_system_prompt_has_citation_rule(self):
        from src.llm.prompts import SYSTEM_PROMPT
        assert "cite" in SYSTEM_PROMPT.lower()
        assert "Call #" in SYSTEM_PROMPT
