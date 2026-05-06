import json
import logging

from openai import AsyncOpenAI
import chromadb

logger = logging.getLogger(__name__)


def _entity_from_chroma(entity_id: str, meta: dict) -> dict:
    result = {"entity_id": entity_id, **meta}
    for field in ("source_chunks", "source_files", "entity_aliases"):
        if isinstance(result.get(field), str):
            result[field] = json.loads(result[field] or "[]")
    return result


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


class Retrievers:
    def __init__(
        self,
        openai_client: AsyncOpenAI,
        emb_deployment: str,
        chunk_col: chromadb.Collection,
        entity_col: chromadb.Collection,
        entity_top_k: int = 5,
        chunk_top_k: int = 15,
        max_entities_per_hop: int = 10,
    ):
        self.openai = openai_client
        self.emb_deployment = emb_deployment
        self.chunk_col = chunk_col
        self.entity_col = entity_col
        self.entity_top_k = entity_top_k
        self.chunk_top_k = chunk_top_k
        self.max_entities_per_hop = max_entities_per_hop

    async def embed(self, text: str) -> list[float]:
        resp = await self.openai.embeddings.create(
            model=self.emb_deployment,
            input=text,
        )
        return resp.data[0].embedding

    async def search_entities_by_phrase(self, seeds: list[dict]) -> list[dict]:
        results = []
        count = self.entity_col.count()
        if count == 0:
            return results

        for seed in seeds:
            phrase = seed.get("search_phrase", seed.get("name", ""))
            entity_type = seed.get("entity_type")

            embedding = await self.embed(phrase)
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

    async def fetch_entities_by_ids(self, entity_ids: list[str]) -> list[dict]:
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

    async def fetch_entity_name_map(self, entity_ids: list[str]) -> dict[str, str]:
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

    async def fetch_chunks(self, chunk_ids: list[str]) -> list[dict]:
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

    async def keyword_fallback(self, query: str) -> list[dict]:
        count = self.chunk_col.count()
        if count == 0:
            return []

        embedding = await self.embed(query)
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
