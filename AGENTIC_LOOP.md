# Agentic Loop & Harness — GraphRAG.POC

This document describes the agentic retrieval pipeline in this project: every LLM call, tool call, vector search, and decision point from the moment a user query enters the system to the moment a cited answer is returned.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Building Blocks (Primitives)](#2-building-blocks-primitives)
3. [GraphRAG Agentic Multi-Hop Loop](#3-graphrag-agentic-multi-hop-loop)
4. [Agent Router — Fully Agentic Tool-Calling Loop](#4-agent-router--fully-agentic-tool-calling-loop)
5. [Vector-Only Mode](#5-vector-only-mode)
6. [Streaming & Event Protocol](#6-streaming--event-protocol)
7. [Dry Run — Log Trace Walkthrough](#7-dry-run--log-trace-walkthrough)

---

## 1. System Overview

There are **three retrieval modes**, selected by the `mode` field in the search request:

| Mode | Effective Pipeline | Description |
|---|---|---|
| `auto` (default) | Agent Router | LLM decides which tools to call each turn |
| `graphrag` | GraphRAG Multi-Hop | Structured seed → discover → plan → hop loop |
| `vector` | Vector-Only | Embed query → cosine similarity → answer |

Both the HTTP API (`app.py → POST /search/stream`) and the CLI (`query.py`) share the same pipeline objects. The HTTP server streams NDJSON events (thought steps + final answer); the CLI collects them and prints.

### Entrypoint flow

```
User Query
    │
    ▼
app.py  /search/stream  (or query.py CLI)
    │
    ├── mode = "auto" / "agent"  ──▶  AgentRouter.run_streaming()
    ├── mode = "graphrag"        ──▶  GraphRAGApproach.run_streaming(mode="graphrag")
    └── mode = "vector"          ──▶  GraphRAGApproach.run_streaming(mode="vector")
```

---

## 2. Building Blocks (Primitives)

### 2.1 Retrievers (`approaches/retrievers.py`)

A stateless service layer wrapping ChromaDB and the embedding model. Every retrieval operation in both pipelines goes through this class.

| Method | What it does | Called by |
|---|---|---|
| `embed(text)` | **Embedding API call** — sends text to `models/gemini-embedding-2`, returns a 3072-dim vector | All search methods below |
| `search_entities_by_phrase(seeds)` | For each seed: embed the `search_phrase` → cosine query the **entity collection** → return top-K entity metadata | GraphRAG hop 0, Agent `find_entities_by_name` |
| `fetch_entities_by_ids(ids)` | Direct ChromaDB `get()` by ID on the **entity collection** — no embedding call | GraphRAG hops 1+, Agent `expand_entity_neighbors` |
| `fetch_entity_name_map(ids)` | Batch ID→name lookup on the **entity collection** (metadata only) | GraphRAG (resolve related-entity names for the hop planner), Agent (same) |
| `fetch_chunks(chunk_ids)` | Direct ChromaDB `get()` by ID on the **chunk collection** — returns full document text + metadata. Applies **document diversity filter** (max 3 chunks per source file) | GraphRAG step 3, Agent `fetch_chunks_by_id` |
| `keyword_fallback(query)` | **Embedding API call** + cosine query on the **chunk collection** → top-K chunks. This is the "vector search" path | Vector mode, Agent `vector_search_chunks`, GraphRAG fallback |

**Document diversity filter** (`_apply_document_diversity`): iterates chunks in relevance order but caps at 3 chunks per unique `sourcefile`. This prevents a single long document from dominating the context window.

### 2.2 ChromaDB Collections

| Collection | Content | Key metadata fields |
|---|---|---|
| `procurement-chunks` | Document chunks (800 tokens, 100-token overlap) | `sourcefile`, `document_type`, `vendor`, `project`, content vector |
| `procurement-entities` | Extracted entities (the "graph") | `entity_name`, `entity_type`, `related_entities` (JSON string of `[{entity_id, relationship_type}]`), `source_chunks` (JSON string of chunk IDs), `source_files`, `mention_count` |

The graph structure lives in the `related_entities` metadata field — each entity stores outgoing edges as a JSON array of `{entity_id, relationship_type}` pairs. There is no separate edge store.

### 2.3 Prompt Templates (`approaches/prompts/`)

| File | Used in | Purpose |
|---|---|---|
| `graphrag_entity_extract.prompty` | GraphRAG seed extraction | Instructs the LLM to identify named entities (vendor, project, PO, etc.) in the user query |
| `graphrag_entity_extract_tools.json` | GraphRAG seed extraction | Defines the `extract_query_entities` tool schema — forces structured output of `[{name, entity_type, search_phrase}]` |
| `graphrag_hop_planning.prompty` | GraphRAG hop planner | Shows the LLM all discovered entities and asks: "which related entity IDs should we follow next, or stop?" |
| `graphrag_hop_planning_tools.json` | GraphRAG hop planner | Defines the `plan_hops` tool schema — returns `{entity_ids_to_follow[], reasoning}` |
| `agent_router.prompty` | Agent Router system prompt | Full instructions: available tools, workflow steps, rules (one tool per turn, must fetch chunks before answering) |
| `agent_router_tools.json` | Agent Router | Defines 5 tools: `find_entities_by_name`, `expand_entity_neighbors`, `vector_search_chunks`, `fetch_chunks_by_id`, `final_answer` |

### 2.4 LLM Client

A single `AsyncOpenAI` client configured to route through Gemini's OpenAI-compatible endpoint:

```
OPENAI_BASE_URL = https://generativelanguage.googleapis.com/v1beta/openai/
Chat model    = models/gemini-2.5-flash
Embedding     = models/gemini-embedding-2  (3072 dimensions)
```

All LLM calls go through the OpenAI SDK's `chat.completions.create()` with `tools` + `tool_choice` for structured output.

---

## 3. GraphRAG Agentic Multi-Hop Loop

**File:** `approaches/graphragapproach.py` — method `_run_graphrag_streaming()`

This is a **structured agentic loop**: the code controls the loop structure, but the LLM makes two key decisions via tool calls — (a) which entities are in the query, and (b) whether to keep traversing or stop.

### Step-by-step pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│  Step 1:  SEED ENTITY EXTRACTION                    [LLM Call]  │
│           LLM + extract_query_entities tool                     │
│           Input:  user query                                    │
│           Output: [{name, entity_type, search_phrase}]          │
├──────────────────────────────────────────────────────────────────┤
│  Step 2:  ENTITY DISCOVERY — Hop 0               [Vector Search]│
│           For each seed: embed(search_phrase) →                 │
│           cosine search entity collection → top-K entities      │
│           Output: entity metadata + related_entities edges      │
├──────────────────────────────────────────────────────────────────┤
│  Step 3:  HOP PLANNING                             [LLM Call]   │
│           LLM + plan_hops tool                                  │
│           Input:  user query + summary of all discovered        │
│                   entities (names, types, relationships)        │
│           Output: {entity_ids_to_follow[], reasoning}           │
│           If empty list → STOP traversal                        │
├──────────────────────────────────────────────────────────────────┤
│  Step 4:  ENTITY DISCOVERY — Hop 1..N            [ChromaDB Get] │
│           fetch_entities_by_ids(entity_ids_to_follow)           │
│           → merge into discovered pool → go to Step 3           │
│           Loop terminates when: agent returns [], or max_hops   │
├──────────────────────────────────────────────────────────────────┤
│  Step 5:  CHUNK RETRIEVAL                        [ChromaDB Get] │
│           Collect source_chunk_ids from relevant entities       │
│           fetch_chunks() with document diversity filter         │
├──────────────────────────────────────────────────────────────────┤
│  Step 6:  ANSWER GENERATION                        [LLM Call]   │
│           System: "You are a procurement analyst..."            │
│           User: concatenated chunk text + question              │
│           Output: cited natural-language answer                 │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Step 1 — Seed Entity Extraction

**LLM Call #1** — `_extract_seed_entities()`

```
Model:       models/gemini-2.5-flash
Temperature: 0
Max tokens:  500
Tool:        extract_query_entities  (tool_choice: forced)
```

The prompt (`graphrag_entity_extract.prompty`) lists the 8 entity types (vendor, customer, po, invoice, grn, project, item, contact) and asks the LLM to extract all entities from the user query with a `search_phrase` for each.

**Output:** A list like `[{"name": "Bangalore Data Center Upgrade", "entity_type": "project", "search_phrase": "Bangalore Data Center Upgrade"}]`.

**Fallback:** If the LLM returns no tool call or an empty entity list, the pipeline falls back to `keyword_fallback()` (pure vector search on chunks) and skips the graph traversal entirely.

### 3.2 Step 2 — Hop 0: Entity Discovery via Semantic Search

**Embedding API Call** — `Retrievers.search_entities_by_phrase()`

For each seed entity:
1. Call `embed(search_phrase)` → 3072-dim vector
2. Query the **entity ChromaDB collection** with cosine similarity, `n_results = entity_top_k` (default 5)
3. Optionally filter by `entity_type` if the seed has a known type

Results are deduplicated by `entity_id`. Each returned entity carries:
- `entity_name`, `entity_type`
- `related_entities` — JSON string of edges: `[{entity_id, relationship_type}]`
- `source_chunks` — JSON string of chunk IDs that mention this entity
- `source_files` — the source documents

For the UI, the pipeline also resolves related-entity IDs to names by calling `fetch_entity_name_map()` on any IDs not already in the discovered pool.

### 3.3 Step 3 — Hop Planning (The Agentic Decision)

**LLM Call #2** — `_plan_next_hops()`

```
Model:       models/gemini-2.5-flash
Temperature: 0
Max tokens:  300
Tool:        plan_hops  (tool_choice: forced)
```

The prompt (`graphrag_hop_planning.prompty`) is populated with:
- The original user query
- Current hop number and max hops
- A text summary of **all entities discovered so far**: their IDs, types, names, related entities (with resolved names), and source files

The LLM decides:
- **Follow** → returns a list of `entity_ids_to_follow` (IDs from the `related_entities` of discovered entities)
- **Stop** → returns an empty list with reasoning like "enough information collected"

### 3.4 Step 4 — Subsequent Hops (Repeat)

If the planner returned entity IDs to follow:

1. `fetch_entities_by_ids(entity_ids_to_follow)` — direct ChromaDB `get()`, no embedding needed
2. Merge newly found entities into the `all_discovered` pool
3. Resolve any new unknown related-entity names
4. Go back to Step 3 (hop planning)

The loop runs for at most `max_hops` iterations (default 3). It terminates early if:
- The planner returns an empty `entity_ids_to_follow` list
- No new entities were found at the current hop

### 3.5 Step 5 — Chunk Retrieval with Seed Filtering

After traversal completes, the pipeline collects `source_chunks` from **relevant** entities:

1. **Seed relevance filter**: From hop-0 entities, keep only those whose `entity_name` matches a seed (fuzzy substring match via `_entity_matches_seed`). This filters out noise — hop-0 vector search may return similar-but-unrelated entities.
2. **Followed entities**: All entities discovered in hops 1+ are automatically included (the planner explicitly chose to follow them).
3. If filtering leaves nothing, fall back to all discovered entities.

The union of `source_chunk_ids` from relevant entities is passed to `Retrievers.fetch_chunks()`, which:
1. Fetches chunk documents + metadata from ChromaDB by ID
2. Applies the **document diversity filter** (max 3 chunks per source file, up to `chunk_top_k` total)

### 3.6 Step 6 — Answer Generation

**LLM Call #3** — `_generate_answer()`

```
Model:       models/gemini-2.5-flash
Temperature: 0.3
Max tokens:  1500
No tools (plain completion)
```

The system prompt is:
> "You are a procurement analyst. Answer the user's question using only the document excerpts below. Cite sources using the [filename] labels. If the answer is not in the documents, say so."

The user message contains all retrieved chunks formatted as:

```
[PO-2024-001.json]
<chunk content>

---

[GRN-2024-003.json]
<chunk content>

...

Question: <original user query>
```

The LLM generates a natural-language answer with `[filename]` citations inline.

### Total LLM/API calls for a GraphRAG query

| Call | Type | When |
|---|---|---|
| Seed extraction | LLM (chat + tool) | Always — 1 call |
| Embed seed phrases | Embedding API | 1 per seed entity |
| Hop planning | LLM (chat + tool) | 1 per hop (until stop or max_hops-1) |
| Resolve related names | ChromaDB get | 1 per hop (if unknown IDs exist) |
| Fetch chunks | ChromaDB get | Once after traversal |
| Answer generation | LLM (chat) | Always — 1 call |

Typical total: **3 LLM calls** (extract + 1 plan + answer) + **1 embedding call** + **2-3 ChromaDB reads**.

---

## 4. Agent Router — Fully Agentic Tool-Calling Loop

**File:** `approaches/agentrouter.py` — method `run_streaming()`

This is an **open-ended agentic loop**: the LLM has full autonomy to decide which tools to call, in what order, and when to stop. The harness provides the tools and enforces budget limits.

### Architecture

```
┌───────────────────────────────────────────────────────────────────┐
│                      AGENT TURN LOOP                              │
│                                                                   │
│   messages = [system_prompt, user_query]                          │
│                                                                   │
│   for turn in 1..max_turns:                                       │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │  LLM CALL  (chat + tools, tool_choice="required")        │   │
│   │  On final turn: tool_choice forced to "final_answer"     │   │
│   │                                                           │   │
│   │  Model returns: one or more tool_calls                    │   │
│   └───────────────┬───────────────────────────────────────────┘   │
│                   │                                               │
│                   ▼                                               │
│   ┌───────────────────────────────────────────────────────────┐   │
│   │  DISPATCH  — execute each tool call:                      │   │
│   │                                                           │   │
│   │  find_entities_by_name    → Retrievers.search_entities_   │   │
│   │                             by_phrase()                   │   │
│   │  expand_entity_neighbors  → Retrievers.fetch_entities_    │   │
│   │                             by_ids() + name resolution    │   │
│   │  vector_search_chunks     → Retrievers.keyword_fallback() │   │
│   │  fetch_chunks_by_id       → Retrievers.fetch_chunks()     │   │
│   │  final_answer             → return answer + citations     │   │
│   └───────────────┬───────────────────────────────────────────┘   │
│                   │                                               │
│                   ▼                                               │
│   Append tool result to messages → next turn                      │
│                                                                   │
│   Exit conditions:                                                │
│   • final_answer tool called → yield answer, return               │
│   • No tool call from LLM → fallback synthesis                    │
│   • Turn budget exhausted → fallback synthesis                    │
└───────────────────────────────────────────────────────────────────┘
```

### 4.1 System Prompt & Tools

The system prompt (`agent_router.prompty`) gives the LLM:
- A persona ("procurement analyst")
- Descriptions of all 5 tools with usage examples
- A prescribed workflow: find entities → optionally expand → fetch chunks → answer
- Hard rules: one tool per turn, must fetch chunks before answering, max 2 expansions, don't repeat identical calls

### 4.2 The 5 Agent Tools

| Tool | Embedding call? | ChromaDB call | Purpose |
|---|---|---|---|
| `find_entities_by_name` | Yes (1 per phrase) | Entity collection query | Find entities by name — same as GraphRAG hop 0 |
| `expand_entity_neighbors` | No | Entity collection get | Fetch entities by ID + resolve related names — same as GraphRAG hop 1+ |
| `vector_search_chunks` | Yes (1) | Chunk collection query | Semantic search over document chunks — returns 400-char previews |
| `fetch_chunks_by_id` | No | Chunk collection get | Retrieve full chunk text by ID — fills a chunk pool (max 20 chunks) |
| `final_answer` | No | None | Terminal tool — LLM provides `{answer, citations[]}` |

### 4.3 Turn-by-turn LLM Calls

Each turn:

```
Model:       models/gemini-2.5-flash
Temperature: 0
Max tokens:  800
Tools:       all 5 agent tools
tool_choice: "required" (turns 1..N-1), forced "final_answer" (turn N)
```

The conversation accumulates: `[system, user, assistant+tool_calls, tool_result, assistant+tool_calls, tool_result, ...]`

### 4.4 Safety & Budget Mechanisms

| Mechanism | How |
|---|---|
| **Turn budget** | `max_turns` (default 8). On the final turn, `tool_choice` is forced to `final_answer` so the model must answer. |
| **Chunk pool cap** | `max_chunks` (default 20). `fetch_chunks_by_id` refuses to fetch more once the pool is full. |
| **Duplicate detection** | A `tool_call_log` set tracks `tool_name + serialized_args`. Duplicates return an error hint instead of re-executing. |
| **Tool result truncation** | Tool results are truncated to `max_tool_result_bytes` (default 8000) before being appended to messages. |
| **No-tool-call fallback** | If the LLM returns text instead of a tool call, the agent gathers whatever chunks it has and synthesizes an answer. |

### 4.5 Fallback Synthesis

When the agent exhausts its budget or fails to call a tool:

1. If the chunk pool has entries → use those chunks
2. Else, extract `source_chunk_ids` from entity tool results in the message history → fetch those chunks
3. Else, fall back to `keyword_fallback(query)` (pure vector search)

Then call `_generate_answer()` — the same LLM call as GraphRAG Step 6.

### Typical Agent turn sequence

For a relationship query like *"Which vendor supplied items for the Bangalore Data Center Upgrade?"*:

| Turn | Tool called | What happens |
|---|---|---|
| 1 | `find_entities_by_name` | Searches for "Bangalore Data Center Upgrade" (project) → returns entities with relationships and `source_chunk_ids` |
| 2 | `expand_entity_neighbors` | Follows related vendor entity IDs → gets vendor details with their `source_chunk_ids` |
| 3 | `fetch_chunks_by_id` | Fetches full document text for the relevant chunk IDs |
| 4 | `fetch_chunks_by_id` | Fetches additional chunks if needed |
| 5 | `final_answer` | LLM produces cited answer from the accumulated chunk context |

---

## 5. Vector-Only Mode

**File:** `approaches/graphragapproach.py` — method `_run_vector_streaming()`

The simplest path — no graph traversal, no entity extraction:

```
Step 1:  Embed the query         → Embedding API call (1x)
Step 2:  Cosine search chunks    → ChromaDB chunk collection query (top-K)
Step 3:  Generate answer         → LLM call (same as GraphRAG Step 6)
```

**Total:** 1 embedding call + 1 ChromaDB query + 1 LLM call.

This mode has no relationship awareness — it retrieves chunks that are semantically similar to the query regardless of entity connections.

---

## 6. Streaming & Event Protocol

All modes emit events as NDJSON lines over `POST /search/stream`:

| Event type | Fields | When |
|---|---|---|
| `thought_step` | `title`, `description`, `step_type`, optional `entities[]`, `chunks[]`, `reasoning` | After each pipeline step |
| `answer` | `content`, `citations[]`, `query_type` | Final event — the answer |
| `error` | `message` | On unrecoverable failure |

The `step_type` field allows the frontend to render different UI for different steps:

- `entity_extraction` — seed entities found
- `hop_discovery` — entities discovered at a hop (includes full entity+relationship payload)
- `hop_planning` — agent's hop decision + reasoning
- `traversal_complete` — agent stopped traversal
- `chunk_retrieval` — chunks fetched (includes chunk content payload)
- `answer_generation` — synthesis in progress
- `agent_turn` — an Agent Router tool call was dispatched
- `vector_search` — vector-only mode search step

---

## 7. Dry Run — Log Trace Walkthrough

Below is an annotated walkthrough of a real query execution from `logs/graphrag.log`. This run used the **comparison mode** — the frontend fired two parallel requests: one `mode=graphrag` and one `mode=vector`, both for the same query.

### Query

> *"Which vendor supplied items for the Bangalore Data Center Upgrade?"*

### Run timestamp: 2026-05-05 00:01:20

---

### 7.1 Both pipelines start simultaneously

```
00:01:20  [app] Query received | mode=graphrag effective=graphrag
00:01:20  [GraphRAG] START query='Which vendor supplied items for the Bangalore Data Center Upgrade?'
00:01:20  [app] Query received | mode=vector effective=vector
00:01:20  [Vector] START query='Which vendor supplied items for the Bangalore Data Center Upgrade?'
```

The frontend sent two `POST /search/stream` requests in parallel. The FastAPI server handles them concurrently via async.

---

### 7.2 Vector pipeline (right panel) — fast path

**Step V1 — Embed query + cosine search chunks** (00:01:20 → 00:01:21)

```
00:01:21  [Vector] Retrieved 15 chunks, sources=['INV-2024-001.txt', 'INV-2024-004.txt',
          'PO-2024-003.txt', 'GRN-2024-001.txt', 'INV-2024-002.txt', 'PO-2024-004.txt',
          'INV-2024-003.txt', 'PO-2024-001.json', 'GRN-2024-003.txt', 'PO-2024-001.txt',
          'PO-2024-004.json', 'INV-2024-003.json', 'INV-2024-001.json', 'GRN-2024-001.json',
          'PO-2024-003.json']
```

- **API calls:** 1 embedding call (embed the query) + 1 ChromaDB cosine query
- **Duration:** ~1 second
- **Result:** 15 chunks from 15 unique source files. No entity awareness — it's pulling whatever chunks are most semantically similar to "vendor supplied items Bangalore Data Center Upgrade".

**Step V2 — Generate answer** (00:01:21 → 00:01:25)

```
00:01:25  [Vector] Answer generated | citations=['PO-2024-001.txt', 'PO-2024-001.json',
          'GRN-2024-001.txt', 'INV-2024-001.txt', 'INV-2024-001.json', 'INV-2024-003.txt',
          'PO-2024-003.txt', 'GRN-2024-003.txt', 'GRN-2024-001.json', 'INV-2024-003.json',
          'PO-2024-003.json', 'PO-2024-004.json', 'INV-2024-002.txt', 'INV-2024-004.txt',
          'PO-2024-004.txt']
```

- **API calls:** 1 LLM chat completion (answer generation)
- **Duration:** ~4 seconds
- **Result:** 15 citations. The vector pipeline includes documents from PO-2024-004 (Delhi Sales Office Setup) which is not related to the Bangalore project — this is the noise that GraphRAG's entity-aware retrieval avoids.

```
00:01:25  [app] Query complete | mode=vector | 4 events
```

**Vector pipeline total: ~5 seconds, 4 streamed events.**

---

### 7.3 GraphRAG pipeline (left panel) — agentic path

**Step G1 — Seed Entity Extraction** (00:01:20 → 00:01:23)

```
00:01:23  [GraphRAG] Seeds extracted: ['Bangalore Data Center Upgrade']
```

- **API calls:** 1 LLM call with `extract_query_entities` tool (forced tool_choice)
- **Duration:** ~3 seconds
- **What happened:** The LLM parsed the user query and identified one seed entity:
  ```json
  [{"name": "Bangalore Data Center Upgrade", "entity_type": "project", "search_phrase": "Bangalore Data Center Upgrade"}]
  ```

**Step G2 — Hop 0: Entity Discovery** (00:01:23 → 00:01:23)

```
00:01:23  [GraphRAG] Hop 0 discovery: 5 entities — ['Bangalore Data Center Upgrade',
          'Bangalore Data Center Upgrade Project', 'Mumbai Office Renovation Project',
          'Mumbai Office Renovation', 'Delhi Sales Office Setup Project']
```

- **API calls:** 1 embedding call (embed "Bangalore Data Center Upgrade") + 1 ChromaDB entity collection cosine query (top-5)
- **Duration:** < 1 second
- **What happened:** Semantic search in the entity collection returned 5 entities. The top 2 are direct matches (`Bangalore Data Center Upgrade`, `Bangalore Data Center Upgrade Project`). The other 3 are other projects that are close in embedding space but unrelated — this is where the seed relevance filter (Step G4) will help.

**Step G3 — Hop 0: Hop Planning** (00:01:23 → 00:01:26)

```
00:01:26  [GraphRAG] Hop 0 plan: following 0 entity IDs
00:01:26  [GraphRAG] Agent signalled stop at hop 0
```

- **API calls:** 1 LLM call with `plan_hops` tool (forced tool_choice)
- **Duration:** ~3 seconds
- **What happened:** The LLM was shown all 5 discovered entities with their relationships and decided **no further traversal was needed** — the discovered entities already have `related_entities` edges pointing to vendors, and their `source_chunks` contain enough context. The planner returned `{entity_ids_to_follow: [], reasoning: "..."}`.

This is the **agentic decision** — the LLM judged that the current entity set is sufficient. In a more complex query (e.g., "What items from the Bangalore project's vendor also appear in the Delhi project?"), the planner would return vendor entity IDs to follow into hop 1.

**Step G4 — Chunk Retrieval with Seed Filtering** (00:01:26)

```
00:01:26  [GraphRAG] Chunks retrieved: 11 from 2 entities (filtered from 5),
          sources=['INV-2024-001.txt', 'PO-2024-003.txt', 'GRN-2024-001.txt',
          'INV-2024-003.txt', 'PO-2024-001.json', 'PO-2024-001.txt',
          'GRN-2024-003.json', 'INV-2024-003.json', 'INV-2024-001.json',
          'GRN-2024-001.json', 'PO-2024-003.json']
```

- **API calls:** 1 ChromaDB chunk collection `get()` by IDs
- **Duration:** < 1 second
- **What happened:**
  1. **Seed relevance filter:** Of the 5 hop-0 entities, only 2 match the seed phrase "Bangalore Data Center Upgrade" (substring match). The 3 unrelated projects (Mumbai, Delhi) were filtered out.
  2. **Source chunk collection:** The 2 relevant entities' `source_chunks` yielded chunk IDs.
  3. **Document diversity:** After fetching, the diversity filter (max 3 per file) reduced to 11 chunks from 11 source files.

Notice: compared to the vector pipeline's 15 sources, the GraphRAG pipeline's 11 sources are more focused — no `PO-2024-004` (Delhi project) or `INV-2024-002` noise.

**Step G5 — Answer Generation** (00:01:26 → 00:01:31)

```
00:01:31  [GraphRAG] Answer generated | citations=['GRN-2024-001.json', 'GRN-2024-003.json',
          'INV-2024-001.json', 'INV-2024-003.json', 'PO-2024-001.json', 'PO-2024-003.json',
          'GRN-2024-001.txt', 'INV-2024-001.txt', 'INV-2024-003.txt', 'PO-2024-001.txt',
          'PO-2024-003.txt']
```

- **API calls:** 1 LLM chat completion (answer generation)
- **Duration:** ~5 seconds
- **Result:** 11 citations, all directly relevant to the Bangalore Data Center Upgrade project.

```
00:01:31  [app] Query complete | mode=graphrag | 7 events
```

**GraphRAG pipeline total: ~11 seconds, 7 streamed events.**

---

### 7.4 Side-by-side comparison

| Aspect | Vector | GraphRAG |
|---|---|---|
| Duration | ~5 sec | ~11 sec |
| Streamed events | 4 | 7 |
| LLM calls | 1 (answer) | 3 (extract + plan + answer) |
| Embedding calls | 1 (query) | 1 (seed phrase) |
| ChromaDB reads | 1 (chunk query) | 2 (entity query + chunk get) |
| Chunks retrieved | 15 | 11 |
| Source files cited | 15 | 11 |
| Precision | Lower — includes unrelated projects | Higher — seed filter removes noise |

The GraphRAG pipeline takes roughly 2x longer due to the additional LLM calls for entity extraction and hop planning, but retrieves a more focused set of documents by leveraging the entity graph to filter out unrelated content.

### 7.5 Agent Router dry run (from same log)

The log also shows an Agent Router execution at `2026-05-04 23:58:20`:

```
23:58:20  [app] Query received | mode=auto effective=agent
23:58:38  [Agent] final_answer on turn 5 | citations=['INV-2024-001.json',
          'GRN-2024-001.txt', 'PO-2024-001.txt', 'PO-2024-001.json']
23:58:38  [app] Query complete | mode=agent | 5 events
```

- **Total duration:** ~18 seconds (5 LLM turns)
- **5 events** = the agent made 4 tool calls + 1 `final_answer`
- The likely turn sequence: `find_entities_by_name` → `expand_entity_neighbors` → `fetch_chunks_by_id` → `fetch_chunks_by_id` → `final_answer`
- **4 citations** — the most focused result of all three modes, because the agent had full autonomy to select exactly the chunks it needed

The trade-off: the Agent Router is the slowest (5 sequential LLM roundtrips) but the most precise, while Vector is the fastest but least precise.
