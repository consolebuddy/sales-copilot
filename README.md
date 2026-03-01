# Sales Call AI Copilot

A RAG-based CLI chatbot that ingests sales call transcripts, stores them with vector embeddings in ChromaDB, and lets users ask natural-language questions with source-cited answers.

## Architecture

```
┌──────────┐     ┌──────────────┐     ┌────────────┐     ┌──────────┐
│  CLI     │────▶│  ChatEngine  │────▶│  Retriever │────▶│ ChromaDB │
│ (cli.py) │     │ (orchestrator)│     │            │     │ (vectors)│
└──────────┘     └──────┬───────┘     └────────────┘     └──────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  LLM Client  │──▶ OpenAI GPT-4o-mini
                 └──────────────┘
```

**Data flow:** Transcript files → Parser → Chunker → Embeddings → ChromaDB → Retrieval → LLM → Response with citations

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

On first run, the chatbot auto-ingests all transcripts from the `transcripts/` folder.

### Example Commands

| Command | Description |
|---------|-------------|
| `list my call ids` | Show all ingested calls |
| `summarise call 1` | Summarize a specific call |
| `What pricing was discussed?` | Free-form Q&A across all calls |
| `Give me all negative comments when pricing was mentioned` | Sentiment-aware query |
| `ingest path/to/new_call.txt` | Add a new transcript |
| `exit` | Quit the chatbot |

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

### Prompt Engineering

- **System prompt** enforces citation format `[Call #id, start-end]`
- **Intent detection** routes to specialized prompts (summary, sentiment, QA) for better responses
- **Context injection** includes segment headers with call metadata so the LLM can cite accurately

### Assumptions

- Transcript format: `[MM:SS] Role (Name – Title):  Text`
- Stage directions (e.g., `*Call ends.*`) are stored as metadata, not embedded
- Call IDs are extracted from filenames (first number found)
- All calls are in English (with some Hindi-English mix handled by the LLM)

## Testing

```bash
pytest tests/ -v
```

Tests cover:
- Transcript parsing (turn count, speaker/role extraction, timestamps, stage directions)
- Chunking (overlap, token limits, metadata, edge cases)
- Parser-chunker integration

## Project Structure

```
sales-copilot/
├── cli.py                     # Entry point
├── config.py                  # Environment configuration
├── requirements.txt
├── .env.example
├── transcripts/               # Sample call transcripts
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
│   │   ├── client.py          # OpenAI API wrapper
│   │   └── prompts.py         # Prompt templates
│   └── chatbot/
│       └── engine.py          # Query orchestrator
└── tests/
    ├── test_parser.py
    └── test_retrieval.py
```
