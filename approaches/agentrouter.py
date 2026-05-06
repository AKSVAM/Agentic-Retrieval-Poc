import json
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from approaches.graphragapproach import _load_prompt, _load_tools, _ts
from approaches.retrievers import Retrievers

logger = logging.getLogger(__name__)


class AgentRouter:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        chat_deployment: str,
        retrievers: Retrievers,
        max_turns: int = 8,
        max_chunks: int = 20,
        max_tool_result_bytes: int = 8000,
    ):
        self.openai = openai_client
        self.chat_deployment = chat_deployment
        self.retrievers = retrievers
        self.max_turns = max_turns
        self.max_chunks = max_chunks
        self.max_tool_result_bytes = max_tool_result_bytes
        self._system_prompt = (
            _load_prompt("agent_router.prompty")
            .replace("{{max_turns}}", str(max_turns))
            .replace("{{max_chunks}}", str(max_chunks))
        )
        self._tools = _load_tools("agent_router_tools.json")

    async def run(self, user_query: str) -> dict:
        thought_steps = []
        answer = None
        async for event in self.run_streaming(user_query):
            if event.get("type") == "thought_step":
                thought_steps.append(event)
            elif event.get("type") == "answer":
                answer = event
        if answer is None:
            return {
                "answer": "I was unable to find an answer.",
                "citations": [],
                "thought_steps": thought_steps,
                "query_type": "agent_fallback",
            }
        return {
            "answer": answer["content"],
            "citations": answer.get("citations", []),
            "thought_steps": thought_steps,
            "query_type": answer.get("query_type", "agent"),
        }

    async def run_streaming(self, user_query: str) -> AsyncGenerator[dict, None]:
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_query},
        ]
        tool_call_log: set[str] = set()
        chunk_pool: dict[str, dict] = {}
        turn = 0

        while turn < self.max_turns:
            turn += 1

            if turn == self.max_turns:
                tool_choice = {"type": "function", "function": {"name": "final_answer"}}
            else:
                tool_choice = "required"

            try:
                resp = await self.openai.chat.completions.create(
                    model=self.chat_deployment,
                    messages=messages,
                    tools=self._tools,
                    tool_choice=tool_choice,
                    temperature=0,
                    max_tokens=800,
                )
            except Exception as e:
                logger.warning("Agent LLM call failed on turn %d: %s", turn, e)
                break

            msg = resp.choices[0].message
            messages.append(msg.model_dump(exclude_none=True))

            if not msg.tool_calls:
                logger.info("[Agent] No tool call on turn %d — synthesizing from gathered context", turn)
                yield _ts("No tool call", "Model responded with text — synthesizing from gathered context.", "no_tool_fallback")
                chunks = await self._gather_fallback_chunks(chunk_pool, messages, user_query)
                async for event in self._synthesize_from_chunks(user_query, chunks):
                    yield event
                return

            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_result = {"error": "invalid_json_arguments"}
                    logger.warning("[Agent] Turn %d: bad JSON from %s", turn, name)
                else:
                    sig = name + ":" + json.dumps(args, sort_keys=True)
                    if sig in tool_call_log:
                        tool_result = {
                            "error": "duplicate_call",
                            "hint": "Same call already issued. Try different args or call final_answer.",
                        }
                    else:
                        tool_call_log.add(sig)
                        tool_result = await self._dispatch(name, args, chunk_pool)

                if name != "final_answer":
                    desc = _summarize_tool_result(name, args if "args" in dir() else {}, tool_result)
                    yield _ts(
                        f"Turn {turn}: {name}",
                        desc,
                        "agent_turn",
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result)[:self.max_tool_result_bytes],
                })

                if name == "final_answer":
                    answer_text = args.get("answer", "")
                    citations = args.get("citations", [])
                    logger.info("[Agent] final_answer on turn %d | citations=%s", turn, citations)
                    yield {"type": "answer", "content": answer_text, "citations": citations, "query_type": "agent"}
                    return

        yield _ts("Agent budget exhausted", "Synthesizing from gathered context.", "agent_fallback")
        chunks = await self._gather_fallback_chunks(chunk_pool, messages, user_query)
        async for event in self._synthesize_from_chunks(user_query, chunks):
            yield event

    async def _dispatch(self, tool_name: str, args: dict, chunk_pool: dict[str, dict]) -> dict:
        try:
            if tool_name == "find_entities_by_name":
                phrases = args.get("phrases", [])
                entities = await self.retrievers.search_entities_by_phrase(phrases)
                return _format_entities_with_related(entities)

            elif tool_name == "expand_entity_neighbors":
                entity_ids = args.get("entity_ids", [])
                entities = await self.retrievers.fetch_entities_by_ids(entity_ids)
                found_ids = {e["entity_id"] for e in entities}
                all_related_ids = set()
                for e in entities:
                    related_raw = e.get("related_entities", "[]")
                    try:
                        rels = json.loads(related_raw) if isinstance(related_raw, str) else (related_raw or [])
                    except Exception:
                        rels = []
                    for r in rels:
                        rid = r.get("entity_id", "")
                        if rid:
                            all_related_ids.add(rid)
                name_map = await self.retrievers.fetch_entity_name_map(list(all_related_ids)) if all_related_ids else {}
                results = []
                for eid in entity_ids:
                    match = next((e for e in entities if e.get("entity_id") == eid), None)
                    if match:
                        results.append(_format_single_entity(match, name_map))
                    else:
                        results.append({"entity_id": eid, "found": False})
                return {"entities": results}

            elif tool_name == "vector_search_chunks":
                query = args.get("query", "")
                chunks = await self.retrievers.keyword_fallback(query)
                return {"chunks": [
                    {
                        "chunk_id": c.get("id", ""),
                        "content_preview": (c.get("content", "") or "")[:400],
                        "sourcefile": c.get("sourcefile", ""),
                    }
                    for c in chunks
                ]}

            elif tool_name == "fetch_chunks_by_id":
                chunk_ids = args.get("chunk_ids", [])
                remaining = self.max_chunks - len(chunk_pool)
                if remaining <= 0:
                    return {"error": "chunk_pool_full", "limit_reached": True,
                            "hint": f"Pool has {len(chunk_pool)}/{self.max_chunks} chunks. Call final_answer."}
                ids_to_fetch = [cid for cid in chunk_ids if cid not in chunk_pool][:remaining]
                if ids_to_fetch:
                    fetched = await self.retrievers.fetch_chunks(ids_to_fetch)
                    for c in fetched:
                        cid = c.get("id", "")
                        if cid and cid not in chunk_pool:
                            chunk_pool[cid] = c
                result_chunks = []
                for cid in chunk_ids:
                    if cid in chunk_pool:
                        c = chunk_pool[cid]
                        result_chunks.append({
                            "chunk_id": cid,
                            "content": c.get("content", ""),
                            "sourcefile": c.get("sourcefile", ""),
                            "document_type": c.get("document_type", ""),
                            "vendor": c.get("vendor", ""),
                            "project": c.get("project", ""),
                        })
                return {
                    "chunks": result_chunks,
                    "pool_used": len(chunk_pool),
                    "pool_limit": self.max_chunks,
                    "limit_reached": len(chunk_pool) >= self.max_chunks,
                }

            elif tool_name == "final_answer":
                return args

            else:
                return {"error": f"unknown_tool: {tool_name}"}

        except Exception as e:
            logger.warning("[Agent] Dispatch error for %s: %s", tool_name, e)
            return {"error": str(e)}

    async def _gather_fallback_chunks(
        self, chunk_pool: dict[str, dict], messages: list[dict], user_query: str
    ) -> list[dict]:
        if chunk_pool:
            return list(chunk_pool.values())
        chunk_ids = _extract_chunk_ids_from_messages(messages)
        if chunk_ids:
            return await self.retrievers.fetch_chunks(chunk_ids)
        return await self.retrievers.keyword_fallback(user_query)

    async def _synthesize_from_chunks(self, user_query: str, chunks: list[dict]) -> AsyncGenerator[dict, None]:
        if not chunks:
            yield {"type": "answer", "content": "I could not find relevant information to answer your question.", "citations": [], "query_type": "agent_fallback"}
            return

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

        try:
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
        except Exception as e:
            logger.warning("[Agent] Fallback answer generation failed: %s", e)
            answer = "I encountered an error generating an answer."
            citations = []

        yield {"type": "answer", "content": answer, "citations": list(dict.fromkeys(citations)), "query_type": "agent_fallback"}


def _extract_chunk_ids_from_messages(messages: list[dict]) -> list[str]:
    chunk_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        try:
            data = json.loads(msg.get("content", "{}"))
        except (json.JSONDecodeError, TypeError):
            continue
        for entity in data.get("entities", []):
            for cid in entity.get("source_chunk_ids", []):
                if cid:
                    chunk_ids.add(cid)
    return list(chunk_ids)


def _format_entities_with_related(entities: list[dict]) -> dict:
    results = []
    for e in entities:
        results.append(_format_single_entity(e, {}))
    return {"entities": results}


def _format_single_entity(entity: dict, name_map: dict[str, str]) -> dict:
    related_raw = entity.get("related_entities", "[]")
    try:
        rels = json.loads(related_raw) if isinstance(related_raw, str) else (related_raw or [])
    except Exception:
        rels = []
    related = []
    for r in rels:
        rid = r.get("entity_id", "")
        related.append({
            "id": rid,
            "name": name_map.get(rid, r.get("entity_name", "")),
            "rel_type": r.get("relationship_type", "related_to"),
        })
    return {
        "entity_id": entity.get("entity_id", ""),
        "entity_name": entity.get("entity_name", ""),
        "entity_type": entity.get("entity_type", ""),
        "found": True,
        "related": related,
        "source_chunk_ids": entity.get("source_chunks", []),
    }


def _summarize_tool_result(name: str, args: dict, result: dict) -> str:
    if "error" in result:
        return f"Error: {result['error']}"

    if name == "find_entities_by_name":
        entities = result.get("entities", [])
        names = [e.get("entity_name", "?") for e in entities[:5]]
        return f"Found {len(entities)} entities: {names}"

    elif name == "expand_entity_neighbors":
        entities = result.get("entities", [])
        found = sum(1 for e in entities if e.get("found"))
        return f"Expanded {len(entities)} IDs — {found} found, {len(entities) - found} missing"

    elif name == "vector_search_chunks":
        chunks = result.get("chunks", [])
        top = chunks[0]["sourcefile"] if chunks else "none"
        return f"Vector search → {len(chunks)} chunk previews (top: {top})"

    elif name == "fetch_chunks_by_id":
        chunks = result.get("chunks", [])
        pool_used = result.get("pool_used", 0)
        pool_limit = result.get("pool_limit", 0)
        return f"Fetched {len(chunks)} full chunks — {pool_used}/{pool_limit} pool slots used"

    return str(result)[:200]
