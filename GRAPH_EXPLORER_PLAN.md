# Plan: Graph Explorer Tab

## What we're building

A new **"Graph Explorer"** tab in the existing React frontend that renders every entity in the ChromaDB database as an interactive force-directed graph — nodes for entities, arrows for relationships.

---

## Why

The ChromaDB entity collection already holds a full knowledge graph (built during ingestion), but there's no way to see it. This adds a visual inspector so you can verify what got ingested, explore relationships, and debug traversal paths.

---

## What it will look like

```
┌─────────────────────────────────────────────────────────┐
│  [Search]   [Graph Explorer]                            │
├─────────────────────────────────────────────────────────┤
│  Filter: [✓vendor] [✓po] [✓invoice] [✓grn] [✓project]  │
│                          [Reset zoom]  20 nodes·34 edges │
├──────────────────────────────────┬──────────────────────┤
│                                  │  (click a node)       │
│   Force-directed graph           │  Name: Acme Corp      │
│   (fills the panel, zoomable,    │  Type: vendor         │
│    pannable, click to select)    │  Mentioned: 3×        │
│                                  │  Files: PO-001.json   │
│   ● Acme Corp ──fulfills──► ●   │        GRN-001.json   │
│        vendor            PO-001  │                       │
│                                  │                       │
└──────────────────────────────────┴──────────────────────┘
```

**Node colors:**
| Entity type | Color |
|---|---|
| vendor | purple |
| customer | blue |
| po | orange |
| invoice | red |
| grn | green |
| project | teal |
| item | amber |
| contact | pink |

**Node size** scales with `mention_count` — entities seen more often are bigger circles.

---

## Files that will change

| File | What changes |
|---|---|
| `app.py` | Add one new `GET /graph` endpoint |
| `frontend/src/api/types.ts` | Add 3 new TypeScript interfaces |
| `frontend/src/api/graphClient.ts` | **New file** — fetches `/graph` |
| `frontend/src/pages/GraphPage/GraphPage.tsx` | **New file** — the graph page |
| `frontend/src/pages/GraphPage/GraphPage.module.css` | **New file** — layout styles |
| `frontend/src/App.tsx` | Add tab switcher (Search / Graph Explorer) |
| `frontend/src/App.css` | Add tab bar styles |

---

## Backend change (app.py)

New endpoint:

```
GET /graph
→ { nodes: [...], edges: [...] }
```

**Node shape:**
```json
{ "id": "abc123", "name": "Acme Corp", "type": "vendor", "mention_count": 3, "source_files": ["PO-001.json"] }
```

**Edge shape:**
```json
{ "source": "abc123", "target": "def456", "relationship_type": "fulfills" }
```

It reads every entity from ChromaDB (`entity_col.get()`), parses the `related_entities` JSON field on each one, and returns the flattened nodes + edges. Edges pointing to unknown node IDs are dropped silently.

---

## Frontend change — new library

Install `react-force-graph` (1 package, includes its own TypeScript types, no extra `@types/` install).

---

## Edge cases handled

- **Empty DB** — shows "No entities found — run ingestion first" instead of a blank canvas
- **Loading state** — spinner while the `/graph` fetch is in flight
- **Type filter** — toggling a type hides matching nodes and any edges connected to them
- **Orphaned edges** — relationship refs pointing to entities not in the DB are dropped at the backend

---

## How to verify after implementation

1. Run backend: `uvicorn app:app --reload --port 8000`
2. Hit `GET http://localhost:8000/graph` — should return JSON with nodes and edges
3. Run frontend: `cd frontend && npm run dev`
4. Click "Graph Explorer" tab → graph renders
5. Click a node → sidebar shows its details
6. Toggle a type filter → those nodes disappear
7. If DB is empty → "No entities" message appears (no crash)
