"""ChromaDB vector store for call transcript chunks."""

import chromadb
from openai import OpenAI

import config
from src.ingestion.chunker import Chunk


class VectorStore:
    def __init__(self):
        self._chroma = chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIR)
        self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self._collection = self._chroma.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings via OpenAI."""
        resp = self._openai.embeddings.create(
            input=texts,
            model=config.EMBEDDING_MODEL,
        )
        return [item.embedding for item in resp.data]

    def add_chunks(self, chunks: list[Chunk]) -> int:
        """Store chunks with embeddings and metadata. Returns count added."""
        if not chunks:
            return 0

        ids = [f"{c.call_id}_{c.chunk_index}" for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "call_id": c.call_id,
                "call_type": c.call_type,
                "chunk_index": c.chunk_index,
                "start_time": c.start_time,
                "end_time": c.end_time,
                "speakers": ", ".join(c.speakers),
            }
            for c in chunks
        ]

        embeddings = self._embed(documents)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        call_id: str | None = None,
    ) -> list[dict]:
        """Similarity search. Returns list of dicts with text, metadata, score."""
        query_embedding = self._embed([query])[0]

        where_filter = {"call_id": call_id} if call_id else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return hits

    def list_calls(self) -> list[dict]:
        """Return distinct call_ids with metadata."""
        all_items = self._collection.get(include=["metadatas"])
        calls: dict[str, dict] = {}
        for meta in all_items["metadatas"]:
            cid = meta["call_id"]
            if cid not in calls:
                calls[cid] = {
                    "call_id": cid,
                    "call_type": meta.get("call_type", ""),
                    "chunk_count": 0,
                }
            calls[cid]["chunk_count"] += 1
        return sorted(calls.values(), key=lambda x: x["call_id"])

    def call_exists(self, call_id: str) -> bool:
        """Check if a call is already ingested."""
        results = self._collection.get(
            where={"call_id": call_id},
            limit=1,
        )
        return len(results["ids"]) > 0

    def delete_call(self, call_id: str) -> int:
        """Remove all chunks for a call. Returns count deleted."""
        results = self._collection.get(where={"call_id": call_id})
        ids = results["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
