"""Chatbot engine — orchestrates ingestion, retrieval, and LLM generation."""

import os
import re
import glob

from src.ingestion.parser import parse_transcript
from src.ingestion.chunker import chunk_transcript
from src.storage.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.llm.client import LLMClient
from src.llm.prompts import SYSTEM_PROMPT, QA_PROMPT, SUMMARY_PROMPT, SENTIMENT_PROMPT
import config


class ChatEngine:
    def __init__(self):
        self._store = VectorStore()
        self._retriever = Retriever(self._store)
        self._llm = LLMClient()

    # ── Ingestion ────────────────────────────────────────────────

    def ingest(self, file_path: str) -> str:
        """Parse, chunk, and store a transcript. Returns confirmation message."""
        if not os.path.isfile(file_path):
            return f"File not found: {file_path}"

        call = parse_transcript(file_path)

        if self._store.call_exists(call.call_id):
            return (
                f"Call #{call.call_id} ({call.call_type}) is already ingested. "
                f"Delete it first to re-ingest."
            )

        chunks = chunk_transcript(call)
        count = self._store.add_chunks(chunks)

        participants = ", ".join(call.participants)
        return (
            f"Ingested call #{call.call_id} ({call.call_type})\n"
            f"  Participants: {participants}\n"
            f"  Duration: {call.duration}\n"
            f"  Chunks stored: {count}"
        )

    def auto_ingest(self, transcript_dir: str | None = None) -> list[str]:
        """Ingest all transcripts from a directory. Returns list of messages."""
        directory = transcript_dir or config.TRANSCRIPT_DIR
        messages = []

        pattern = os.path.join(directory, "*.txt")
        files = sorted(glob.glob(pattern))

        if not files:
            messages.append(f"No transcript files found in {directory}")
            return messages

        for fp in files:
            msg = self.ingest(fp)
            messages.append(msg)

        return messages

    # ── Query handling ───────────────────────────────────────────

    def list_calls(self) -> str:
        """Return formatted list of ingested calls."""
        calls = self._store.list_calls()
        if not calls:
            return "No calls ingested yet. Place transcripts in the transcripts/ folder and restart."

        lines = ["Ingested calls:"]
        for c in calls:
            lines.append(
                f"  Call #{c['call_id']}  |  Type: {c['call_type']}  |  Chunks: {c['chunk_count']}"
            )
        return "\n".join(lines)

    def process_query(self, user_input: str) -> str:
        """Route user input to the appropriate handler and return a response."""
        text = user_input.strip()
        lower = text.lower()

        # ── Intent: list calls ───────────────────────────────────
        if re.search(r"\b(list|show)\b.*\b(call|transcript)", lower):
            return self.list_calls()

        # ── Intent: ingest ───────────────────────────────────────
        ingest_match = re.search(
            r"\b(?:ingest|add|import|load)\b.*?(?:from\s+)([\"']?)(\S+\.txt)\1\s*$",
            text,
            re.IGNORECASE,
        )
        if not ingest_match:
            # Also match: "ingest ./transcripts/call_5.txt" (no "from")
            ingest_match = re.search(
                r"\b(?:ingest|add|import|load)\b\s+([\"']?)(\S+\.txt)\1\s*$",
                text,
                re.IGNORECASE,
            )
        if ingest_match:
            path = ingest_match.group(2).strip()
            return self.ingest(path)

        # ── Intent: delete call ──────────────────────────────────
        delete_match = re.search(r"\b(?:delete|remove)\b.*\b(?:call|transcript)\s*[_#]?(\d+)", lower)
        if delete_match:
            call_id = delete_match.group(1)
            count = self._store.delete_call(call_id)
            if count:
                return f"Deleted {count} chunks for call #{call_id}."
            return f"No data found for call #{call_id}."

        # ── Determine call filter ────────────────────────────────
        call_id = None
        call_ref = re.search(r"\bcall\s*#?(\d+)", lower)
        if call_ref:
            call_id = call_ref.group(1)

        # ── Intent: summarise ────────────────────────────────────
        if re.search(r"\b(summar|recap|overview|highlights)\b", lower):
            return self._rag_query(
                query=text,
                prompt_template=SUMMARY_PROMPT,
                call_id=call_id,
                top_k=8,
            )

        # ── Intent: sentiment / negative / positive ──────────────
        if re.search(r"\b(sentiment|negative|positive|concern|objection|pushback|complaint)\b", lower):
            return self._rag_query(
                query=text,
                prompt_template=SENTIMENT_PROMPT,
                call_id=call_id,
                top_k=6,
            )

        # ── Default: QA ──────────────────────────────────────────
        return self._rag_query(
            query=text,
            prompt_template=QA_PROMPT,
            call_id=call_id,
            top_k=5,
        )

    # ── Private helpers ──────────────────────────────────────────

    def _rag_query(
        self,
        query: str,
        prompt_template: str,
        call_id: str | None = None,
        top_k: int = 5,
    ) -> str:
        """Retrieve context, build prompt, call LLM, return answer."""
        results = self._retriever.retrieve(query=query, top_k=top_k, call_id=call_id)
        context = self._retriever.format_context(results)
        user_prompt = prompt_template.format(context=context, query=query)
        return self._llm.generate(SYSTEM_PROMPT, user_prompt)
