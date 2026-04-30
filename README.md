# Agentic Retrieval POC — GraphRAG for Procurement

A **GraphRAG** (Graph Retrieval-Augmented Generation) proof-of-concept for procurement document analysis. Extracts an entity graph from procurement documents (Purchase Orders, Invoices, Goods Receipt Notes), then uses **agentic multi-hop graph traversal** at query time to answer relationship-aware questions with citations.

> Runs fully **locally** with **ChromaDB** + **free-tier Google Gemini** — no Azure or paid OpenAI account required.

---

## Why GraphRAG?

Plain vector RAG retrieves chunks by semantic similarity to a query. It struggles with relational questions like:

> *"Which invoices have been received but not yet validated by a GRN for vendor Acme?"*

GraphRAG answers these by:
1. Extracting an **entity graph** (vendors, POs, invoices, GRNs, projects, items…) during ingestion.
2. At query time, an LLM picks **seed entities**, then iteratively decides which related entities to follow (**hop planning**), up to N hops.
3. The final answer is generated only from chunks attached to traversed entities — no semantic noise from unrelated documents.

See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full system diagram, hop loop, and design decisions.

---

## Stack

| Layer | Technology |
|---|---|
| LLM + embeddings | Google Gemini (free tier) via OpenAI-compatible SDK |
| Vector / graph store | ChromaDB (local, on-disk) |
| Backend API | FastAPI + NDJSON streaming |
| Frontend | React + TypeScript + Vite |
| Entity extraction & hop planning | LLM tool calling (function calling) |

---

## Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ (for the frontend)
- A free Google Gemini API key — get one at https://aistudio.google.com/apikey

### 2. Configure
```bash
cp .env.template .env
# edit .env and paste your Gemini key into OPENAI_API_KEY
```

### 3. Install
```bash
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 4. Ingest sample data
```bash
python ingest.py --setup-indexes              # create ChromaDB collections (run once)
python ingest.py --data-dir sample_data/      # ingest all sample JSON + text files
```

### 5. Run

**Backend:**
```bash
uvicorn app:app --reload --port 8000
```

**Frontend (in a second terminal):**
```bash
cd frontend
npm run dev
# open http://localhost:5173
```

**CLI alternative:**
```bash
python query.py                               # interactive
python query.py --query "List all invoices linked to PO-2024-001"
python query.py --verbose                     # show traversal thought steps
```

---

## Project Structure

```
.
├── app.py                     # FastAPI server (POST /search/stream)
├── ingest.py                  # Ingestion CLI
├── query.py                   # Query CLI
├── approaches/
│   ├── graphragapproach.py    # Core multi-hop traversal — main algorithm
│   ├── queryrouter.py         # Pattern classifier: GraphRAG vs. vector
│   └── prompts/               # .prompty files + tool JSON schemas
├── prepdocslib/
│   ├── entityextractor.py     # LLM-based entity extraction (ingestion)
│   └── entitysearchmanager.py # ChromaDB entity CRUD + merging
├── frontend/                  # React UI with thought-step trace + comparison view
├── sample_data/               # Sample POs, invoices, GRNs (JSON + text)
├── schemas/                   # JSON schemas for procurement document types
├── ARCHITECTURE.md            # Full architecture reference (read this!)
├── GRAPH_EXPLORER_PLAN.md     # Roadmap for interactive graph explorer UI
└── CLAUDE.md                  # Repo guide for Claude Code
```

---

## Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Gemini API key (`AIza...`) |
| `OPENAI_BASE_URL` | Gemini OpenAI-compatible endpoint | Routes OpenAI SDK to Gemini |
| `AZURE_OPENAI_CHATGPT_DEPLOYMENT` | `models/gemini-2.5-flash` | Chat / tool-calling model |
| `AZURE_OPENAI_EMB_DEPLOYMENT` | `models/gemini-embedding-2` | Embedding model |
| `AZURE_OPENAI_EMB_DIMENSIONS` | `3072` | Embedding output dimension |
| `CHROMADB_PATH` | `./chromadb_data` | Local ChromaDB storage path |
| `USE_GRAPHRAG` | `true` | Toggle graph traversal entirely |
| `GRAPHRAG_MAX_HOPS` | `3` | Max traversal depth |
| `GRAPHRAG_MAX_ENTITIES_PER_HOP` | `10` | Entities per hop |
| `GRAPHRAG_ENTITY_TOP_K` | `5` | Seed entity candidates from vector search |
| `GRAPHRAG_CHUNK_TOP_K` | `15` | Final chunks retrieved for answer generation |

To switch from Gemini to OpenAI: clear `OPENAI_BASE_URL`, set `OPENAI_API_KEY=sk-...`, and update the model deployments (see commented Option B in `.env.template`).

---

## How the Query Flow Works

```
User query
    │
    ▼
QueryRouter ── regex match? ──► GraphRAG mode
    │  no
    └──► Vector mode (top-K embed search)

GraphRAG mode:
    LLM call ① — extract seed entities from query
    Loop (up to MAX_HOPS):
        Hop 0: semantic search in entity collection
        Hop N: direct fetch of related entity IDs
        LLM call ② — plan_hops: which IDs to follow next?
    Collect source chunks from traversed entities (3 per file cap)
    LLM call ③ — answer generation with citations
    Stream NDJSON events to client
```

Streaming protocol emits `thought_step` events (`entity_extraction`, `hop_discovery`, `hop_planning`, `traversal_complete`, `chunk_retrieval`, `answer_generation`) and a final `answer` event with citations. The frontend visualises each hop as a step card.

---

## API

```
POST /search/stream
Content-Type: application/json

{ "query": "...", "mode": "auto" | "graphrag" | "vector" }
```

Returns `application/x-ndjson` — one JSON event per line. See [`ARCHITECTURE.md`](./ARCHITECTURE.md#streaming-protocol) for the event schema.

```
GET /health  →  {"status": "ok"}
```

---

## Sample Queries

Try these against the bundled `sample_data/`:

- `List all POs issued to vendor Acme Industries`
- `Which invoices fulfill PO-2024-001?`
- `Show all GRNs that validate invoices for project Phoenix`
- `Find purchase orders charged to project Phoenix and their delivery status`

The frontend's **comparison view** runs both GraphRAG and plain vector simultaneously so you can see the difference.

---

## Notes & Limitations

- This is a **POC**. There are no automated tests or CI.
- Entity extraction quality depends on the LLM. The `models/gemini-2.5-flash` free tier is fast but occasionally misses cross-document references.
- ChromaDB metadata must be scalar — list fields are JSON-stringified and parsed on read.
- The `dimensions` parameter is intentionally omitted from embedding calls (Gemini compatibility — it always returns 3072d).
- Sample data is synthetic.

---

## License

This is a proof-of-concept; no license file is included. Contact the repo owner for terms before reuse.
