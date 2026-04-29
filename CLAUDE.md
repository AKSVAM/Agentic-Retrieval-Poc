# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A GraphRAG (Graph Retrieval-Augmented Generation) proof-of-concept for procurement document analysis. It extracts entity graphs from procurement documents (Purchase Orders, Invoices, Goods Receipt Notes), then uses multi-hop graph traversal at query time to answer relationship-aware questions.

**No Azure account required.** The project runs fully locally (ChromaDB) + free-tier Gemini API.

## Setup

Copy `.env.template` to `.env` and populate all variables before running any commands.

Required services:
- **Google Gemini API** (free tier) — get key at https://aistudio.google.com/apikey
- **ChromaDB** — local, no account needed, installed via pip

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

**Ingestion pipeline:**
```bash
python ingest.py --setup-indexes            # Create/recreate ChromaDB collections (run once, or to reset)
python ingest.py --data-dir sample_data/    # Ingest all JSON + text files
python ingest.py --file sample_data/json/PO-2024-001.json  # Ingest a single file
```

**Query interface:**
```bash
python query.py                    # Interactive loop
python query.py --query "..."      # Single query then exit
python query.py --no-graphrag      # Force keyword-only mode
python query.py --verbose          # Show traversal thought steps
```

There are no automated tests or CI in this repository.

## Architecture

### Data Flow

1. **Ingest**: Documents are chunked (800 tokens, 100-token overlap), entities and relationships extracted by LLM (Gemini), then both chunks and entities are embedded (3072-dim via `models/gemini-embedding-2`) and stored in ChromaDB collections on disk.
2. **Query**: `QueryRouter` classifies the query. Relationship-oriented queries go to `GraphRAGApproach`; others fall back to keyword search.
3. **GraphRAG traversal** (`GraphRAGApproach.run()`):
   - Extract seed entities from the query via LLM tool call (`graphrag_entity_extract_tools.json`)
   - Semantic search in the entity collection to find initial entity matches
   - Agentic hop planning: LLM decides which related entities to follow next (`graphrag_hop_planning_tools.json`)
   - Repeat for up to `GRAPHRAG_MAX_HOPS` hops
   - Collect source chunks from all traversed entities (max 3 chunks per source file for diversity)
   - Generate final answer with citations

### Key Files

| File | Role |
|---|---|
| `approaches/graphragapproach.py` | Core multi-hop traversal logic — the main algorithm |
| `approaches/queryrouter.py` | Pattern-matching classifier: GraphRAG vs. keyword search |
| `prepdocslib/entityextractor.py` | LLM-based entity extraction during ingestion |
| `prepdocslib/entitysearchmanager.py` | ChromaDB entity collection CRUD and entity merging |
| `approaches/prompts/graphrag_hop_planning.prompty` | Prompt controlling agentic hop decisions |
| `chromadb_data/` | Local ChromaDB persistence directory (auto-created, do not commit) |

### Two ChromaDB Collections

- **Chunk collection** (`procurement-chunks`): Document chunks with content vector (3072-dim) and metadata (document_type, vendor, project, sourcefile).
- **Entity collection** (`procurement-entities`): Extracted entities with `related_entities` (JSON string of linked entity IDs), `source_chunks`, embedding (3072-dim), and `mention_count`. This is the graph.

> ChromaDB metadata values must be scalar (str/int/float/bool). List fields (`source_chunks`, `source_files`, `entity_aliases`, `related_entities`) are stored as JSON strings and deserialized on read.

### Entity Types

`vendor`, `customer`, `po`, `invoice`, `grn`, `project`, `item`, `contact`

### Key Configuration (`.env`)

| Variable | Value | Effect |
|---|---|---|
| `OPENAI_API_KEY` | Gemini API key (`AIza...`) | Authentication for Gemini |
| `OPENAI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai/` | Routes OpenAI SDK to Gemini |
| `AZURE_OPENAI_CHATGPT_DEPLOYMENT` | `models/gemini-2.5-flash` | Chat/tool-calling model |
| `AZURE_OPENAI_EMB_DEPLOYMENT` | `models/gemini-embedding-2` | Embedding model |
| `AZURE_OPENAI_EMB_DIMENSIONS` | `3072` | Output dimension of gemini-embedding-2 |
| `CHROMADB_PATH` | `./chromadb_data` | Local storage path for ChromaDB |
| `AZURE_SEARCH_CHUNK_INDEX` | `procurement-chunks` | ChromaDB collection name for chunks |
| `AZURE_SEARCH_ENTITY_INDEX` | `procurement-entities` | ChromaDB collection name for entities |
| `USE_GRAPHRAG` | `true` | Enable/disable graph traversal entirely |
| `USE_GRAPHRAG_ENTITY_EXTRACTION` | `true` | Extract entities during ingestion |
| `GRAPHRAG_MAX_HOPS` | `3` | Maximum traversal depth |
| `GRAPHRAG_MAX_ENTITIES_PER_HOP` | `10` | Entities fetched per hop |
| `GRAPHRAG_ENTITY_TOP_K` | `5` | Seed entity candidates from vector search |
| `GRAPHRAG_CHUNK_TOP_K` | `15` | Chunks retrieved from entity sources |

### LLM Tool Use Pattern

Both entity extraction and hop planning use function/tool calling — the LLM returns structured JSON through tool definitions in `approaches/prompts/*_tools.json`. If the LLM returns no tool call, the code falls back to keyword-based search rather than failing hard.

### OpenAI SDK Compatibility

The code uses the `openai` Python SDK (`AsyncOpenAI`) pointed at Gemini's OpenAI-compatible endpoint via `OPENAI_BASE_URL`. To switch back to OpenAI API, remove `OPENAI_BASE_URL` and set `OPENAI_API_KEY` to an `sk-...` key.

The `dimensions` parameter is intentionally omitted from embedding calls for Gemini compatibility — the model always returns its native 3072-dim output.
