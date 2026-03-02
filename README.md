# Sales Call AI Copilot

A RAG-based CLI chatbot that ingests sales call transcripts, stores them with vector embeddings in ChromaDB, and lets users ask natural-language questions with source-cited answers.

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  CLI     │────▶│  ChatEngine  │────▶│  Retriever │────▶│ ChromaDB │
│ (cli.py) │     │ (orchestrator)│     │            │     │ (vectors)│
└──────────┘     └──────┬───────┘     └────────────┘     └──────────┘
                        │
                   ┌────┴─────┐
                   │ LLM Client│──▶ OpenAI GPT-4o-mini
                   └────┬─────┘
                        │
                ┌───────┴────────┐
                │ Intent Router  │  (OpenAI function calling)
                │ ─ list_calls   │
                │ ─ ingest       │
                │ ─ delete_call  │
                │ ─ summarize    │
                │ ─ sentiment    │
                │ ─ question     │
                └────────────────┘
```

**Data flow:**
1. **Ingestion:** Transcript files → Parser → Chunker → Embeddings → ChromaDB
2. **Query:** User input + Conversation History → LLM Intent Router → Action handler → (for RAG: Retrieval + History → LLM) → Response with citations

## Setup

### Prerequisites
- Python 3.11+
- OpenAI API key

### Installation

```bash
cd sales-copilot
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

```bash
python cli.py
```

On every startup, the chatbot scans the `transcripts/` folder and auto-ingests any new transcript files. Previously ingested calls are skipped, so you can simply drop new `.txt` files into the folder.

### Example Commands

| Command | Description |
|---------|-------------|
| `list my call ids` | Show all ingested calls with count |
| `tell me total number of calls done` | Same — any natural phrasing works |
| `summarise call 1` | Summarize a specific call |
| `What pricing was discussed?` | Free-form Q&A across all calls |
| `Give me all negative comments when pricing was mentioned` | Sentiment-aware query |
| `ingest a new call transcript from ./transcripts/call_5.txt` | Add a new transcript |
| `delete call 5` | Remove a call from the store |
| `exit` | Quit the chatbot |

All commands are understood via natural language — there are no rigid keywords to memorize.

### How Intent Routing Works

Instead of brittle regex matching, the chatbot uses **OpenAI function calling** to classify every user message into one of six actions:

| Action | When it triggers |
|--------|-----------------|
| `list_calls` | "list calls", "how many calls", "show transcripts", etc. |
| `ingest` | "ingest ...", "add a transcript from ...", "load ..." |
| `delete_call` | "delete call 5", "remove transcript 3", etc. |
| `summarize` | "summarise call 1", "recap", "give me an overview", etc. |
| `sentiment` | "negative comments", "objections", "concerns about pricing", etc. |
| `question` | Any other question about the call content |

The router also extracts structured parameters (`call_id`, `file_path`) so the engine can act precisely.

## Design Decisions

### Storage Schema (ChromaDB)

Each transcript is parsed into dialogue turns, then grouped into **overlapping chunks** (5 turns per chunk, 2-turn overlap). Each chunk is stored as a ChromaDB document with:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | `{call_id}_{chunk_index}` |
| `document` | string | Formatted transcript text with timestamps |
| `embedding` | vector | OpenAI `text-embedding-3-small` (1536-dim) |
| `call_id` | metadata | Source call identifier |
| `call_type` | metadata | Demo, Pricing, Objection Handling, Negotiation |
| `chunk_index` | metadata | Position within the call |
| `start_time` | metadata | First timestamp in chunk |
| `end_time` | metadata | Last timestamp in chunk |
| `speakers` | metadata | Comma-separated speaker names |

**Why overlapping chunks?** Ensures context isn't lost at chunk boundaries. A question about a topic that spans two adjacent turns will match a chunk containing both.

**Why ChromaDB?** Lightweight, local-first, persistent, with built-in metadata filtering — ideal for a CLI tool without external infrastructure.

### Conversation History

The chatbot maintains a rolling conversation history (last 10 exchanges) so users can ask follow-up questions naturally:

- **"summarise call 1"** → summary response
- **"tell me more about the pricing part"** → the LLM uses history to understand "the pricing part" refers to call 1's summary
- **"what about objections?"** → resolves "what about" from conversation context

History is passed to both the intent router (so it can resolve references like "that call" or "the same one") and the LLM generation step (so RAG answers are contextually aware). The history is bounded to prevent token limit issues.

### Prompt Engineering

- **System prompt** enforces citation format `[Call #id, start-end]`
- **Specialized prompts** for summary, sentiment, and QA modes produce higher-quality, focused responses
- **Context injection** includes segment headers with call metadata so the LLM can cite accurately

### Transcript Parser

- Regex-based parser for the `[MM:SS] Role (Name – Title):  Text` format
- Filters out stage directions (e.g., `*Call ends.*`, `SE (reads on-screen)`) from dialogue turns
- Infers call type (Demo, Pricing, Objection Handling, Negotiation) from filename or content keywords
- Deduplicates participants by name with clean display labels

### Assumptions

- Transcript format: `[MM:SS] Role (Name – Title):  Text`
- Stage directions (e.g., `*Call ends.*`) are stored as metadata, not embedded
- Call IDs are extracted from filenames (first number found)
- All calls are in English (with some Hindi-English mix handled by the LLM)

## Testing

```bash
python -m pytest tests/ -v
```

97 tests covering:

| Area | Tests | Coverage |
|------|-------|----------|
| Parser basics | 6 | Turn count, call ID, duration, timestamps, text content, filename |
| Speaker extraction | 5 | Named speakers, roles, shorthand, multi-speaker, participants |
| `_parse_speaker` | 8 | Name+title, bare role, two-word role, CISO, stage directions |
| `_extract_call_id` | 4 | Underscore, prefix, double-digit, no-number fallback |
| `_infer_call_type` | 9 | All 4 types from filename + content, unknown fallback |
| Speaker validation | 2 | Known roles pass, invalid speakers rejected |
| Stage directions | 4 | Excluded from turns, collected separately, smiling/reads |
| Edge cases | 3 | Empty file, single turn, call ID from single turn |
| Real transcripts | 12 | All 4 sample files: parsing, participants, speakers |
| Chunking | 13 | Overlap, no-overlap, large overlap, timing, speakers, metadata |
| Retriever formatting | 4 | Empty, single, multiple results, relevance score |
| Parser+chunker integration | 3 | End-to-end, type propagation, all turns covered |
| Engine action routing | 13 | All 6 actions + edge cases + multi-action (mocked) |
| Prompt templates | 4 | QA, summary, sentiment formatting, citation rules |
| Conversation history | 7 | Growth, accumulation, trimming, pass-through to router & LLM, ordering |

## Project Structure

```
sales-copilot/
├── cli.py                     # Entry point — REPL with auto-ingestion
├── config.py                  # Environment configuration
├── requirements.txt
├── .env.example
├── transcripts/               # Drop transcript .txt files here
│   ├── call_1.txt             # Demo call
│   ├── call_2.txt             # Pricing call
│   ├── call_3.txt             # Objection handling call
│   └── call_4.txt             # Negotiation call
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # Transcript format parser
│   │   └── chunker.py         # Overlapping chunk generator
│   ├── storage/
│   │   └── vector_store.py    # ChromaDB operations
│   ├── retrieval/
│   │   └── retriever.py       # Similarity search + context formatting
│   ├── llm/
│   │   ├── client.py          # OpenAI wrapper + intent router (function calling)
│   │   └── prompts.py         # Prompt templates (QA, summary, sentiment)
│   └── chatbot/
│       └── engine.py          # Query orchestrator
└── tests/
    ├── test_parser.py
    └── test_retrieval.py
```
