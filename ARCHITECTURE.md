# GraphRAG POC — Architecture Reference

> Procurement document Q&A using agentic multi-hop graph traversal.  
> Stack: Python · FastAPI · ChromaDB (local) · Gemini API (via OpenAI-compatible SDK)

---

## System Components

```
┌─────────────────────────────────────────────────────────────────┐
│  INGESTION (ingest.py)          QUERY (app.py · FastAPI)        │
│                                                                  │
│  Raw docs                        Browser / CLI                  │
│     │                                │                          │
│  Chunker                        POST /search/stream             │
│  (800 tok, 100 overlap)              │                          │
│     │                           QueryRouter                     │
│     ├──► Embed (Gemini 3072d)   (regex patterns)                │
│     │         │                      │                          │
│     │    chunk_col ◄────────         ├── graphrag ──► GraphRAGApproach
│     │                               └── vector  ──► GraphRAGApproach
│     └──► LLM entity extract                          (_run_vector_streaming)
│               │                                                  │
│          EntitySearchManager                     NDJSON stream   │
│          (merge + upsert)            thought_step events ──────►│
│               │                     answer event ──────────────►│
│          entity_col                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## ChromaDB Collections

| Collection | Key stored per record | Notable fields |
|---|---|---|
| `procurement-chunks` | chunk_id | `content`, `document_type`, `vendor`, `project`, `sourcefile`, embedding (3072d) |
| `procurement-entities` | entity_id (SHA-256 of `type:VALUE`) | `entity_name`, `entity_type`, `related_entities` (JSON), `source_chunks` (JSON), `source_files` (JSON), `mention_count`, embedding (3072d) |

Entity types: `vendor · customer · po · invoice · grn · project · item · contact`

> All list-valued fields are stored as JSON strings (ChromaDB scalar constraint) and deserialized on read.

---

## Ingestion Pipeline

```
 File (JSON or .txt)
        │
        ▼
 ┌─────────────┐
 │  Chunker    │  800-word windows, 100-word overlap
 └──────┬──────┘
        │  chunk_text, chunk_id, sourcefile
        ├────────────────────────────────────────────────────────┐
        ▼                                                        ▼
 ┌──────────────────┐                               ┌────────────────────┐
 │  LLM CALL ①      │  tool: store_entities         │  Embed (Gemini)    │
 │  Entity Extract  │  per-chunk, T=0, max 2000 tok │  3072-dim vector   │
 └──────┬───────────┘                               └────────┬───────────┘
        │ [{entity_name, entity_type, entity_value,          │
        │   entity_aliases, related_entity_refs}]            │
        ▼                                                     ▼
 ┌──────────────────────────────────────┐      ┌─────────────────────┐
 │  EntitySearchManager.upsert_entities │      │  chunk_col.upsert   │
 │  • compute entity_id = SHA-256       │      │  (id, embedding,    │
 │    (type:VALUE)[:32]                 │      │   metadata)         │
 │  • resolve related_entity_refs →     │      └─────────────────────┘
 │    typed entity_ids via pattern      │
 │    (PO-* → po, INV-* → invoice …)   │
 │  • merge with existing (union        │
 │    source_chunks, source_files,      │
 │    related_entities, aliases)        │
 │  • embed entity_name → 3072d         │
 │  • entity_col.upsert                 │
 └──────────────────────────────────────┘
```

**Relationship inference during ingestion** — the LLM returns `related_entity_refs` as document IDs (e.g. `"PO-2024-001"`). The extractor derives both the target entity_id and the relationship type from the pair of entity types:

| from → to | relationship |
|---|---|
| invoice → po | `fulfills` |
| grn → po | `closes` |
| grn → invoice | `validates` |
| po / invoice / grn → vendor | `issued_to / from / received_from` |
| po / invoice → project | `for_project / charged_to` |
| anything else | `related_to` |

---

## Query Pipeline

```
 User query
      │
      ▼
 ┌─────────────┐   regex match on           ┌──────────────────────┐
 │ QueryRouter │──► relationship keywords ──►│  GraphRAGApproach    │
 └─────────────┘   ("all vendors", "compare" │  .run_streaming()    │
       │            "linked", "find all" …)  └──────────────────────┘
       │
       └──► no match ──► still GraphRAGApproach
                         but mode="vector"
                         (_run_vector_streaming — straight top-K embed search)
```

> The frontend can also force either mode independently (comparison view fires both simultaneously).

---

## GraphRAG Traversal — Core Loop

```
 User query
      │
      ▼
 ┌────────────────────────────────────┐
 │  LLM CALL ②  (extract_query_      │  tool: extract_query_entities
 │  entities)                         │  T=0, max 500 tok
 └──────────────────┬─────────────────┘
                    │ [{name, entity_type, search_phrase}]
                    │
           no seeds?├──► vector fallback (keyword_fallback)
                    │
                    ▼
         ┌──────────────────────────────────────────────────────┐
         │  HOP LOOP  (max 3 hops, configurable)                │
         │                                                       │
         │  hop 0                                               │
         │  ┌───────────────────────────────────────────────┐  │
         │  │  Embed seed search_phrase (Gemini 3072d)       │  │
         │  │  entity_col.query  cosine top-K=5             │  │
         │  │  → newly_found entities                        │  │
         │  └───────────────────────────────────────────────┘  │
         │                                                       │
         │  hop N (N > 0)                                       │
         │  ┌───────────────────────────────────────────────┐  │
         │  │  entity_col.get(ids=entity_ids_to_follow)      │  │
         │  │  → newly_found entities  (no embedding call)   │  │
         │  └───────────────────────────────────────────────┘  │
         │                                                       │
         │  After each hop:                                      │
         │  1. Batch-fetch names for unknown related_entity IDs  │
         │     (entity_col.get — one call per hop)               │
         │  2. Emit hop_discovery event (entity graph snapshot)  │
         │  3. Accumulate into all_discovered dict               │
         │                                                       │
         │  ┌───────────────────────────────────────────────┐  │
         │  │  LLM CALL ③  (plan_hops)                      │  │  ◄── per hop, up to max_hops-1 times
         │  │  Input: query + entity summary with names       │  │
         │  │  Output: entity_ids_to_follow[], reasoning      │  │
         │  │  T=0, max 300 tok                               │  │
         │  └───────────────────┬───────────────────────────┘  │
         │                      │                               │
         │            empty list?├──► STOP (traversal_complete) │
         │                      │                               │
         │                      └──► next hop with those IDs    │
         └──────────────────────────────────────────────────────┘
                    │
                    ▼
         Filter hop-0 entities by seed-name match
         (substring check; explicitly-followed entities always kept)
                    │
                    ▼
         Collect source_chunk IDs from relevant entities
                    │
                    ▼
         chunk_col.get(chunk_ids)
         → apply diversity cap (max 3 chunks per source file)
         → top-K=15 chunks
                    │
                    ▼
         ┌────────────────────────────────┐
         │  LLM CALL ④  (answer gen)      │  plain completion, T=0.3
         │  system: procurement analyst   │  max 1500 tok
         │  context: [filename] chunks    │
         │  → answer text + citations     │
         └────────────────────────────────┘
                    │
                    ▼
         NDJSON stream → frontend
```

---

## LLM Calls Summary

| # | When | Tool / Type | Model input | Output | Fallback |
|---|---|---|---|---|---|
| ① | Ingestion, per chunk | `store_entities` tool | chunk text | entity list with refs | skip chunk |
| ② | Per query | `extract_query_entities` tool | user query | seed entities | vector search |
| ③ | Per hop (up to 2×) | `plan_hops` tool | query + entity summary | IDs to follow + reasoning | stop traversal |
| ④ | Per query (final) | plain completion | top-K chunks as context | answer + citations | — |

All calls use `temperature=0` except ④ (`0.3`). All use Gemini via OpenAI-compatible endpoint.

---

## Streaming Protocol

Server sends NDJSON over `POST /search/stream`. Each line is one of:

```jsonc
// Thought step (trace event)
{"type":"thought_step","step_type":"hop_discovery","title":"Hop 0 — entity discovery",
 "description":"Found 5 entities: [...]","entities":[...],"reasoning":null}

// Final answer
{"type":"answer","content":"...","citations":["GRN-2024-001.json",...],"query_type":"graphrag"}

// Error
{"type":"error","message":"..."}
```

`step_type` values: `entity_extraction · hop_discovery · hop_planning · traversal_complete · chunk_retrieval · vector_search · answer_generation · fallback`

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Entity IDs as SHA-256 of `type:VALUE` | Deterministic — same entity extracted from two different files gets the same ID and merges automatically |
| `related_entities` stores IDs not names | IDs are stable; names can vary across documents. Names resolved at query time via batch fetch |
| Hop 0 via semantic search, hops 1+ via direct ID fetch | Seed lookup needs fuzzy matching; subsequent hops follow explicit graph edges |
| Hop-0 noise filter (seed-name substring match) | Embedding search can return semantically-nearby-but-wrong project entities; filter keeps only those that actually match the seed before chunk retrieval |
| Diversity cap (3 chunks/file) | Prevents a single large document from dominating the LLM context |
| QueryRouter pattern-match before GraphRAG | Simple keyword queries (single-entity lookups) don't need graph traversal; avoids 2 extra LLM calls |
