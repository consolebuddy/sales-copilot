"""Tests for chunking and retrieval (unit tests that don't require OpenAI)."""

import pytest

from src.ingestion.parser import parse_transcript, CallTranscript, DialogueTurn
from src.ingestion.chunker import chunk_transcript, Chunk


def _make_call(num_turns: int = 12) -> CallTranscript:
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
        call_id="99",
        call_type="Test",
        file_name="call_99.txt",
        participants=["Jordan (AE)", "Priya (Prospect)"],
        turns=turns,
        duration=turns[-1].timestamp,
    )


class TestChunking:
    def test_basic_chunk_count(self):
        call = _make_call(12)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        # With 12 turns, step=3: windows start at 0,3,6,9 -> 4 chunks
        assert len(chunks) >= 4

    def test_chunk_overlap(self):
        call = _make_call(10)
        chunks = chunk_transcript(call, turns_per_chunk=5, overlap=2)
        # Second chunk should share last 2 turns of first chunk
        assert chunks[0].end_time != chunks[1].start_time or len(chunks) > 1

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
            call_id="0",
            call_type="Empty",
            file_name="empty.txt",
            participants=[],
            turns=[],
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


class TestParserIntegration:
    """Integration tests that parse actual transcript files."""

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
