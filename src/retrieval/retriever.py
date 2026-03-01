"""Retrieval layer wrapping the vector store."""

from dataclasses import dataclass
from src.storage.vector_store import VectorStore


@dataclass
class RetrievalResult:
    text: str
    call_id: str
    call_type: str
    chunk_index: int
    start_time: str
    end_time: str
    speakers: str
    relevance_score: float


class Retriever:
    def __init__(self, store: VectorStore):
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        call_id: str | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve relevant transcript chunks for a query."""
        hits = self._store.search(query=query, top_k=top_k, call_id=call_id)

        results = []
        for hit in hits:
            meta = hit["metadata"]
            results.append(RetrievalResult(
                text=hit["text"],
                call_id=meta["call_id"],
                call_type=meta.get("call_type", ""),
                chunk_index=meta.get("chunk_index", 0),
                start_time=meta.get("start_time", ""),
                end_time=meta.get("end_time", ""),
                speakers=meta.get("speakers", ""),
                relevance_score=1 - hit["distance"],  # cosine distance -> similarity
            ))
        return results

    def format_context(self, results: list[RetrievalResult]) -> str:
        """Format retrieval results into a context string for the LLM prompt."""
        if not results:
            return "No relevant transcript segments found."

        parts = []
        for i, r in enumerate(results, 1):
            header = (
                f"[Segment {i} | Call #{r.call_id} ({r.call_type}), "
                f"{r.start_time}–{r.end_time} | Speakers: {r.speakers}]"
            )
            parts.append(f"{header}\n{r.text}")
        return "\n\n".join(parts)
