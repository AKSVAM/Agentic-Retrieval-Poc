import json
import logging
from datetime import datetime, timezone
import chromadb

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def _to_chroma_meta(entity: dict) -> dict:
    meta = {k: v for k, v in entity.items() if k not in ("entity_id", "embedding")}
    for field in ("source_chunks", "source_files", "entity_aliases", "allowedUsers", "allowedGroups"):
        if isinstance(meta.get(field), list):
            meta[field] = json.dumps(meta[field])
    return meta


def _parse_existing(entity_id: str, meta: dict) -> dict:
    result = {"entity_id": entity_id, **meta}
    for field in ("source_chunks", "source_files", "entity_aliases"):
        if isinstance(result.get(field), str):
            result[field] = json.loads(result[field] or "[]")
    return result


class EntitySearchManager:
    def __init__(self, chroma_client: chromadb.ClientAPI, entity_col: chromadb.Collection):
        self.chroma_client = chroma_client
        self.entity_col = entity_col

    async def upsert_entities(self, entities: list[dict]) -> None:
        if not entities:
            return

        merged = await self._merge_with_existing(entities)

        for i in range(0, len(merged), BATCH_SIZE):
            batch = merged[i : i + BATCH_SIZE]
            self.entity_col.upsert(
                ids=[e["entity_id"] for e in batch],
                embeddings=[e["embedding"] for e in batch],
                documents=[e.get("entity_value", "") for e in batch],
                metadatas=[_to_chroma_meta(e) for e in batch],
            )
            logger.info("Upserted %d entities (batch %d).", len(batch), i // BATCH_SIZE + 1)

    async def _merge_with_existing(self, new_entities: list[dict]) -> list[dict]:
        ids = [e["entity_id"] for e in new_entities]
        existing: dict[str, dict] = {}

        try:
            results = self.entity_col.get(
                ids=ids,
                include=["metadatas"],
            )
            for eid, meta in zip(results["ids"], results["metadatas"]):
                existing[eid] = _parse_existing(eid, meta)
        except Exception as e:
            logger.warning("Could not fetch existing entities for merge: %s", e)

        merged = []
        now = datetime.now(timezone.utc).isoformat()
        for e in new_entities:
            eid = e["entity_id"]
            if eid in existing:
                old = existing[eid]
                e["source_chunks"] = list(set((old.get("source_chunks") or []) + e.get("source_chunks", [])))
                e["source_files"] = list(set((old.get("source_files") or []) + e.get("source_files", [])))
                e["entity_aliases"] = list(set((old.get("entity_aliases") or []) + e.get("entity_aliases", [])))
                e["mention_count"] = (old.get("mention_count") or 0) + 1
                old_related = json.loads(old.get("related_entities") or "[]")
                new_related = json.loads(e.get("related_entities") or "[]")
                e["related_entities"] = json.dumps(_merge_related(old_related, new_related))
            e["last_seen"] = now
            merged.append(e)

        return merged


def _merge_related(old: list[dict], new: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {f"{r['entity_id']}:{r['relationship_type']}": r for r in old}
    for r in new:
        key = f"{r['entity_id']}:{r['relationship_type']}"
        if key not in seen:
            seen[key] = r
    return list(seen.values())
