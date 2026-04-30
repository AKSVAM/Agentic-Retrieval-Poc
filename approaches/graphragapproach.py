import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator
from openai import AsyncOpenAI
import chromadb

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _load_tools(name: str) -> list[dict]:
    return json.loads((PROMPTS_DIR / name).read_text(encoding="utf-8"))


def _entity_from_chroma(entity_id: str, meta: dict) -> dict:
    result = {"entity_id": entity_id, **meta}
    for field in ("source_chunks", "source_files", "entity_aliases"):
        if isinstance(result.get(field), str):
            result[field] = json.loads(result[field] or "[]")
    return result


class GraphRAGApproach:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        chat_deployment: str,
        emb_deployment: str,
        emb_dimensions: int,
        chunk_col: chromadb.Collection,
        entity_col: chromadb.Collection,
        max_hops: int = 3,
        max_entities_per_hop: int = 10,
        entity_top_k: int = 5,
        chunk_top_k: int = 15,
    ):
        self.openai = openai_client
        self.chat_deployment = chat_deployment
        self.emb_deployment = emb_deployment
        self.emb_dimensions = emb_dimensions
        self.chunk_col = chunk_col
        self.entity_col = entity_col
        self.max_hops = max_hops
        self.max_entities_per_hop = max_entities_per_hop
        self.entity_top_k = entity_top_k
        self.chunk_top_k = chunk_top_k

        self._extract_prompt = _load_prompt("graphrag_entity_extract.prompty")
        self._extract_tools = _load_tools("graphrag_entity_extract_tools.json")
        self._hop_prompt = _load_prompt("graphrag_hop_planning.prompty")
        self._hop_tools = _load_tools("graphrag_hop_planning_tools.json")

    # -------------------------------------------------------------------------
    # Public entry point
    # -------------------------------------------------------------------------

    async def run(self, user_query: str, security_filter: str | None = None) -> dict:
        thought_steps = []

        # Step 1: extract seed entities
        seeds = await self._extract_seed_entities(user_query)
        thought_steps.append({"title": "Seed entity extraction", "description": f"Found {len(seeds)} seed(s): {[s['name'] for s in seeds]}"})

        if not seeds:
            thought_steps.append({"title": "Fallback", "description": "No entities found — falling back to keyword search."})
            chunks = await self._keyword_fallback(user_query)
            return await self._generate_answer(user_query, chunks, thought_steps, query_type="keyword_fallback")

        # Step 2: agentic hop loop
        all_discovered: dict[str, dict] = {}
        hop0_entity_ids: set[str] = set()
        entity_ids_to_fetch: list[str] = []

        for hop in range(self.max_hops):
            if hop == 0:
                newly_found = await self._search_entities_by_phrase(seeds)
            else:
                if not entity_ids_to_fetch:
                    break
                newly_found = await self._fetch_entities_by_ids(entity_ids_to_fetch)

            new_names = [e.get("entity_name") for e in newly_found if e.get("entity_id") not in all_discovered]
            thought_steps.append({"title": f"Hop {hop} — entity discovery", "description": f"Found {len(newly_found)} entities: {new_names[:8]}"})

            for e in newly_found:
                eid = e.get("entity_id")
                if eid and eid not in all_discovered:
                    all_discovered[eid] = e
                    if hop == 0:
                        hop0_entity_ids.add(eid)

            if hop < self.max_hops - 1:
                entity_ids_to_fetch, _ = await self._plan_next_hops(user_query, all_discovered, hop)
                thought_steps.append({
                    "title": f"Hop {hop} — plan",
                    "description": f"Agent decided to follow {len(entity_ids_to_fetch)} entity IDs next."
                })
                if not entity_ids_to_fetch:
                    thought_steps.append({"title": "Traversal complete", "description": "Agent signalled stop."})
                    break

        # Step 3: fetch chunks from seed-relevant entities only (hop 0 noise filtered)
        relevant_hop0 = {eid for eid in hop0_entity_ids if _entity_matches_seed(all_discovered[eid].get("entity_name", ""), seeds)}
        followed_ids = set(all_discovered.keys()) - hop0_entity_ids
        relevant_ids = (relevant_hop0 | followed_ids) or set(all_discovered.keys())
        chunk_ids: set[str] = set()
        for eid in relevant_ids:
            chunk_ids.update(all_discovered[eid].get("source_chunks") or [])
        chunks = await self._fetch_chunks(list(chunk_ids))
        thought_steps.append({"title": "Chunk retrieval", "description": f"Retrieved {len(chunks)} chunks from {len(relevant_ids)} entities."})

        return await self._generate_answer(user_query, chunks, thought_steps, query_type="graphrag")

    async def run_streaming(
        self, user_query: str, mode: str = "graphrag"
    ) -> AsyncGenerator[dict, None]:
        """Async generator yielding thought_step events then a final answer event."""
        try:
            if mode == "vector":
                async for event in self._run_vector_streaming(user_query):
                    yield event
            else:
                async for event in self._run_graphrag_streaming(user_query):
                    yield event
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    async def _run_graphrag_streaming(self, user_query: str) -> AsyncGenerator[dict, None]:
        logger.info("[GraphRAG] START query=%r", user_query)
        seeds = await self._extract_seed_entities(user_query)
        logger.info("[GraphRAG] Seeds extracted: %s", [s.get("name") for s in seeds])
        yield _ts("Seed entity extraction", f"Found {len(seeds)} seed(s): {[s['name'] for s in seeds]}", "entity_extraction")

        if not seeds:
            logger.info("[GraphRAG] No seeds — falling back to vector search")
            yield _ts("Fallback", "No entities found — falling back to keyword/vector search.", "fallback")
            chunks = await self._keyword_fallback(user_query)
            logger.info("[GraphRAG] Fallback retrieved %d chunks", len(chunks))
            yield _ts("Vector search", f"Retrieved {len(chunks)} chunks via embedding similarity.", "vector_search")
            yield _ts("Generating answer", "Synthesizing answer from retrieved context...", "answer_generation")
            result = await self._generate_answer(user_query, chunks, [], query_type="keyword_fallback")
            yield {"type": "answer", "content": result["answer"], "citations": result["citations"], "query_type": result["query_type"]}
            return

        all_discovered: dict[str, dict] = {}
        hop0_entity_ids: set[str] = set()
        all_name_maps: dict[str, str] = {}
        entity_ids_to_fetch: list[str] = []

        for hop in range(self.max_hops):
            if hop == 0:
                newly_found = await self._search_entities_by_phrase(seeds)
            else:
                if not entity_ids_to_fetch:
                    break
                newly_found = await self._fetch_entities_by_ids(entity_ids_to_fetch)

            new_names = [e.get("entity_name") for e in newly_found if e.get("entity_id") not in all_discovered]
            logger.info("[GraphRAG] Hop %d discovery: %d entities — %s", hop, len(newly_found), new_names)
            _combined = {**all_discovered, **{e["entity_id"]: e for e in newly_found if e.get("entity_id")}}
            related_ids_unknown = _collect_unknown_related_ids(newly_found, _combined)
            hop_name_map = await self._fetch_entity_name_map(related_ids_unknown) if related_ids_unknown else {}
            all_name_maps.update(hop_name_map)
            enriched_lookup = {**_combined, **{eid: {"entity_name": name} for eid, name in all_name_maps.items()}}
            entity_payload = _build_entity_payload(newly_found, enriched_lookup, all_discovered)
            yield _ts(f"Hop {hop} — entity discovery", f"Found {len(newly_found)} entities: {new_names[:8]}", "hop_discovery", entities=entity_payload)

            for e in newly_found:
                eid = e.get("entity_id")
                if eid and eid not in all_discovered:
                    all_discovered[eid] = e
                    if hop == 0:
                        hop0_entity_ids.add(eid)

            if hop < self.max_hops - 1:
                entity_ids_to_fetch, reasoning = await self._plan_next_hops(user_query, all_discovered, hop, all_name_maps)
                logger.info("[GraphRAG] Hop %d plan: following %d entity IDs", hop, len(entity_ids_to_fetch))
                yield _ts(f"Hop {hop} — plan", f"Agent decided to follow {len(entity_ids_to_fetch)} entity IDs next.", "hop_planning", reasoning=reasoning)
                if not entity_ids_to_fetch:
                    logger.info("[GraphRAG] Agent signalled stop at hop %d", hop)
                    yield _ts("Traversal complete", "Agent signalled stop.", "traversal_complete")
                    break

        relevant_hop0 = {eid for eid in hop0_entity_ids if _entity_matches_seed(all_discovered[eid].get("entity_name", ""), seeds)}
        followed_ids = set(all_discovered.keys()) - hop0_entity_ids
        relevant_ids = (relevant_hop0 | followed_ids) or set(all_discovered.keys())
        chunk_ids: set[str] = set()
        for eid in relevant_ids:
            chunk_ids.update(all_discovered[eid].get("source_chunks") or [])
        chunks = await self._fetch_chunks(list(chunk_ids))
        source_files = list({c.get("sourcefile", "") for c in chunks})
        logger.info("[GraphRAG] Chunks retrieved: %d from %d entities (filtered from %d), sources=%s", len(chunks), len(relevant_ids), len(all_discovered), source_files)
        yield _ts("Chunk retrieval", f"Retrieved {len(chunks)} chunks from {len(relevant_ids)} of {len(all_discovered)} entities.", "chunk_retrieval", chunks=_build_chunk_payload(chunks))
        yield _ts("Generating answer", "Synthesizing answer from retrieved context...", "answer_generation")
        result = await self._generate_answer(user_query, chunks, [], query_type="graphrag")
        logger.info("[GraphRAG] Answer generated | citations=%s", result["citations"])
        yield {"type": "answer", "content": result["answer"], "citations": result["citations"], "query_type": result["query_type"]}

    async def _run_vector_streaming(self, user_query: str) -> AsyncGenerator[dict, None]:
        logger.info("[Vector] START query=%r", user_query)
        yield _ts("Query embedding", f"Embedding query → {self.emb_dimensions}-dim vector", "vector_search")
        chunks = await self._keyword_fallback(user_query)
        source_files = list({c.get("sourcefile", "") for c in chunks})
        logger.info("[Vector] Retrieved %d chunks, sources=%s", len(chunks), source_files)
        yield _ts("Top-K chunk retrieval", f"Cosine similarity → top {len(chunks)} chunks (no relationship awareness)", "vector_search", chunks=_build_chunk_payload(chunks))
        yield _ts("Generating answer", "Synthesizing answer from retrieved context...", "answer_generation")
        result = await self._generate_answer(user_query, chunks, [], query_type="vector")
        logger.info("[Vector] Answer generated | citations=%s", result["citations"])
        yield {"type": "answer", "content": result["answer"], "citations": result["citations"], "query_type": result["query_type"]}

    # -------------------------------------------------------------------------
    # Entity index operations
    # -------------------------------------------------------------------------

    async def _extract_seed_entities(self, user_query: str) -> list[dict]:
        prompt = self._extract_prompt.replace("{{user_query}}", user_query)
        try:
            resp = await self.openai.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                tools=self._extract_tools,
                tool_choice={"type": "function", "function": {"name": "extract_query_entities"}},
                temperature=0,
                max_tokens=500,
            )
            tool_calls = resp.choices[0].message.tool_calls
            if not tool_calls:
                return []
            return json.loads(tool_calls[0].function.arguments).get("entities", [])
        except Exception as e:
            logger.warning("Seed extraction failed: %s", e)
            return []

    async def _search_entities_by_phrase(self, seeds: list[dict]) -> list[dict]:
        results = []
        count = self.entity_col.count()
        if count == 0:
            return results

        for seed in seeds:
            phrase = seed.get("search_phrase", seed.get("name", ""))
            entity_type = seed.get("entity_type")

            embedding = await self._embed(phrase)
            n = min(self.entity_top_k, count)
            where = {"entity_type": entity_type} if entity_type and entity_type != "unknown" else None

            try:
                kwargs = dict(
                    query_embeddings=[embedding],
                    n_results=n,
                    include=["metadatas"],
                )
                if where:
                    kwargs["where"] = where
                search_results = self.entity_col.query(**kwargs)
                for eid, meta in zip(search_results["ids"][0], search_results["metadatas"][0]):
                    results.append(_entity_from_chroma(eid, meta))
            except Exception as e:
                logger.warning("Entity semantic search failed for '%s': %s", phrase, e)

        return _deduplicate_by_id(results)

    async def _fetch_entities_by_ids(self, entity_ids: list[str]) -> list[dict]:
        if not entity_ids:
            return []

        results = []
        try:
            fetched = self.entity_col.get(
                ids=entity_ids[: self.max_entities_per_hop],
                include=["metadatas"],
            )
            for eid, meta in zip(fetched["ids"], fetched["metadatas"]):
                results.append(_entity_from_chroma(eid, meta))
        except Exception as e:
            logger.warning("Entity ID fetch failed: %s", e)

        return results

    async def _plan_next_hops(self, user_query: str, discovered: dict[str, dict], hop: int, name_map: dict[str, str] | None = None) -> tuple[list[str], str]:
        summary = _summarize_entities(discovered, name_map or {}, max_chars=3000)
        prompt = (
            self._hop_prompt
            .replace("{{user_query}}", user_query)
            .replace("{{discovered_entities_summary}}", summary)
            .replace("{{hop_number}}", str(hop))
            .replace("{{max_hops}}", str(self.max_hops))
        )
        try:
            resp = await self.openai.chat.completions.create(
                model=self.chat_deployment,
                messages=[{"role": "user", "content": prompt}],
                tools=self._hop_tools,
                tool_choice={"type": "function", "function": {"name": "plan_hops"}},
                temperature=0,
                max_tokens=300,
            )
            tool_calls = resp.choices[0].message.tool_calls
            if not tool_calls:
                return [], ""
            args = json.loads(tool_calls[0].function.arguments)
            reasoning = args.get("reasoning", "")
            logger.info("Hop %d planning reasoning: %s", hop, reasoning)
            return args.get("entity_ids_to_follow", []), reasoning
        except Exception as e:
            logger.warning("Hop planning failed at hop %d: %s", hop, e)
            return [], ""

    async def _fetch_entity_name_map(self, entity_ids: list[str]) -> dict[str, str]:
        if not entity_ids:
            return {}
        result: dict[str, str] = {}
        try:
            fetched = self.entity_col.get(ids=entity_ids[:50], include=["metadatas"])
            for eid, meta in zip(fetched["ids"], fetched["metadatas"]):
                result[eid] = meta.get("entity_name", "")
        except Exception as e:
            logger.warning("Name map fetch failed: %s", e)
        return result

    # -------------------------------------------------------------------------
    # Chunk retrieval
    # -------------------------------------------------------------------------

    async def _fetch_chunks(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []

        chunks = []
        try:
            fetched = self.chunk_col.get(
                ids=chunk_ids[: self.chunk_top_k * 2],
                include=["metadatas", "documents"],
            )
            for cid, meta, doc in zip(fetched["ids"], fetched["metadatas"], fetched["documents"]):
                chunks.append({"id": cid, "content": doc, **meta})
        except Exception as e:
            logger.warning("Chunk fetch failed: %s", e)

        return _apply_document_diversity(chunks, self.chunk_top_k)

    async def _keyword_fallback(self, query: str) -> list[dict]:
        count = self.chunk_col.count()
        if count == 0:
            return []

        embedding = await self._embed(query)
        n = min(self.chunk_top_k, count)

        chunks = []
        try:
            results = self.chunk_col.query(
                query_embeddings=[embedding],
                n_results=n,
                include=["metadatas", "documents"],
            )
            for cid, meta, doc in zip(results["ids"][0], results["metadatas"][0], results["documents"][0]):
                chunks.append({"id": cid, "content": doc, **meta})
        except Exception as e:
            logger.warning("Keyword fallback search failed: %s", e)

        return chunks

    # -------------------------------------------------------------------------
    # Answer generation
    # -------------------------------------------------------------------------

    async def _generate_answer(self, user_query: str, chunks: list[dict], thought_steps: list[dict], query_type: str) -> dict:
        context_parts = []
        citations = []
        for i, c in enumerate(chunks):
            label = c.get("sourcefile", f"source_{i}")
            context_parts.append(f"[{label}]\n{c.get('content', '')}")
            citations.append(label)

        context = "\n\n---\n\n".join(context_parts)
        system_prompt = (
            "You are a procurement analyst. Answer the user's question using only the document excerpts below. "
            "Cite sources using the [filename] labels. If the answer is not in the documents, say so."
        )

        response = await self.openai.chat.completions.create(
            model=self.chat_deployment,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Documents:\n\n{context}\n\nQuestion: {user_query}"},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        answer = response.choices[0].message.content or ""
        return {
            "answer": answer,
            "citations": list(dict.fromkeys(citations)),
            "thought_steps": thought_steps,
            "query_type": query_type,
        }

    # -------------------------------------------------------------------------
    # Embedding helper
    # -------------------------------------------------------------------------

    async def _embed(self, text: str) -> list[float]:
        resp = await self.openai.embeddings.create(
            model=self.emb_deployment,
            input=text,
        )
        return resp.data[0].embedding


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _ts(title: str, description: str, step_type: str, **extra) -> dict:
    return {"type": "thought_step", "title": title, "description": description, "step_type": step_type, **extra}


def _build_entity_payload(newly_found: list[dict], name_lookup: dict[str, dict], already_known: dict[str, dict]) -> list[dict]:
    result = []
    for e in newly_found:
        eid = e.get("entity_id", "")
        if not eid or eid in already_known:
            continue
        related_raw = e.get("related_entities", "[]")
        try:
            rels_list = json.loads(related_raw) if isinstance(related_raw, str) else (related_raw or [])
        except Exception:
            rels_list = []
        relationships = []
        for r in rels_list:
            rid = r.get("entity_id", "")
            rname = name_lookup.get(rid, {}).get("entity_name") or ""
            relationships.append({
                "entity_id": rid,
                "entity_name": rname,
                "relationship_type": r.get("relationship_type", "related_to"),
            })
        result.append({
            "entity_id": eid,
            "entity_name": e.get("entity_name", ""),
            "entity_type": e.get("entity_type", ""),
            "relationships": relationships,
        })
    return result


def _build_chunk_payload(chunks: list[dict]) -> list[dict]:
    return [
        {
            "id": c.get("id", ""),
            "content": c.get("content", ""),
            "sourcefile": c.get("sourcefile", ""),
            "document_type": c.get("document_type", ""),
            "vendor": c.get("vendor", ""),
            "project": c.get("project", ""),
        }
        for c in chunks
    ]


def _deduplicate_by_id(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for item in items:
        eid = item.get("entity_id")
        if eid and eid not in seen:
            seen.add(eid)
            result.append(item)
    return result


def _apply_document_diversity(chunks: list[dict], top_k: int) -> list[dict]:
    seen_files: dict[str, int] = {}
    result = []
    for c in chunks:
        f = c.get("sourcefile", "")
        count = seen_files.get(f, 0)
        if count < 3:
            result.append(c)
            seen_files[f] = count + 1
        if len(result) >= top_k:
            break
    return result


def _summarize_entities(discovered: dict[str, dict], name_map: dict[str, str], max_chars: int = 3000) -> str:
    lines = []
    for eid, e in discovered.items():
        related_raw = e.get("related_entities") or "[]"
        try:
            related = json.loads(related_raw)
            related_summaries = []
            for r in related[:5]:
                rid = r.get("entity_id", "")
                rname = name_map.get(rid, "")
                rtype = r.get("relationship_type", "related_to")
                related_summaries.append(f"{rid}({rname},{rtype})" if rname else f"{rid}({rtype})")
        except Exception:
            related_summaries = []

        line = (
            f"- entity_id={eid} | type={e.get('entity_type')} | "
            f"name={e.get('entity_name')} | "
            f"related_to=[{', '.join(related_summaries)}] | "
            f"files={e.get('source_files', [])[:2]}"
        )
        lines.append(line)
        if sum(len(l) for l in lines) > max_chars:
            break
    return "\n".join(lines)


def _collect_unknown_related_ids(newly_found: list[dict], known: dict[str, dict]) -> list[str]:
    ids: set[str] = set()
    for e in newly_found:
        related_raw = e.get("related_entities", "[]")
        try:
            rels = json.loads(related_raw) if isinstance(related_raw, str) else (related_raw or [])
        except Exception:
            rels = []
        for r in rels:
            rid = r.get("entity_id", "")
            if rid and rid not in known:
                ids.add(rid)
    return list(ids)


def _entity_matches_seed(entity_name: str, seeds: list[dict]) -> bool:
    en = entity_name.lower()
    for seed in seeds:
        sn = seed.get("name", "").lower()
        if sn and (sn in en or en in sn):
            return True
    return False
