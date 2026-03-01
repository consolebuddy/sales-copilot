"""Chunk parsed transcripts into overlapping segments for embedding."""

from dataclasses import dataclass, field
import tiktoken

from src.ingestion.parser import CallTranscript, DialogueTurn


@dataclass
class Chunk:
    call_id: str
    call_type: str
    chunk_index: int
    text: str
    start_time: str
    end_time: str
    speakers: list[str]
    metadata: dict = field(default_factory=dict)


def _format_turn(turn: DialogueTurn) -> str:
    """Format a single turn for chunk text."""
    return f"[{turn.timestamp}] {turn.speaker}:  {turn.text}"


def _count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    enc = tiktoken.get_encoding(encoding_name)
    return len(enc.encode(text))


def _split_large_chunk(
    turns: list[DialogueTurn],
    call_id: str,
    call_type: str,
    base_index: int,
    max_tokens: int = 512,
) -> list[Chunk]:
    """Split a group of turns that exceeds max_tokens into smaller chunks."""
    chunks = []
    current_turns: list[DialogueTurn] = []
    current_text_parts: list[str] = []

    for turn in turns:
        formatted = _format_turn(turn)
        tentative = "\n".join(current_text_parts + [formatted])

        if current_turns and _count_tokens(tentative) > max_tokens:
            # Flush current chunk
            speakers = sorted({t.speaker_name for t in current_turns})
            chunks.append(Chunk(
                call_id=call_id,
                call_type=call_type,
                chunk_index=base_index + len(chunks),
                text="\n".join(current_text_parts),
                start_time=current_turns[0].timestamp,
                end_time=current_turns[-1].timestamp,
                speakers=speakers,
            ))
            current_turns = []
            current_text_parts = []

        current_turns.append(turn)
        current_text_parts.append(formatted)

    if current_turns:
        speakers = sorted({t.speaker_name for t in current_turns})
        chunks.append(Chunk(
            call_id=call_id,
            call_type=call_type,
            chunk_index=base_index + len(chunks),
            text="\n".join(current_text_parts),
            start_time=current_turns[0].timestamp,
            end_time=current_turns[-1].timestamp,
            speakers=speakers,
        ))

    return chunks


def chunk_transcript(
    call: CallTranscript,
    turns_per_chunk: int = 5,
    overlap: int = 2,
    max_tokens: int = 512,
) -> list[Chunk]:
    """
    Group consecutive dialogue turns into overlapping chunks.

    Args:
        call: Parsed call transcript.
        turns_per_chunk: Number of turns per chunk.
        overlap: Number of overlapping turns between consecutive chunks.
        max_tokens: Maximum tokens per chunk; splits further if exceeded.
    """
    turns = call.turns
    if not turns:
        return []

    chunks: list[Chunk] = []
    step = max(1, turns_per_chunk - overlap)
    chunk_idx = 0

    i = 0
    while i < len(turns):
        window = turns[i : i + turns_per_chunk]
        text = "\n".join(_format_turn(t) for t in window)

        if _count_tokens(text) > max_tokens:
            sub_chunks = _split_large_chunk(
                window, call.call_id, call.call_type, chunk_idx, max_tokens
            )
            chunks.extend(sub_chunks)
            chunk_idx += len(sub_chunks)
        else:
            speakers = sorted({t.speaker_name for t in window})
            chunks.append(Chunk(
                call_id=call.call_id,
                call_type=call.call_type,
                chunk_index=chunk_idx,
                text=text,
                start_time=window[0].timestamp,
                end_time=window[-1].timestamp,
                speakers=speakers,
            ))
            chunk_idx += 1

        i += step

    return chunks
