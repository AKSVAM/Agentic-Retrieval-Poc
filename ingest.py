"""
Ingestion CLI for GraphRAG.POC.

Usage:
    python ingest.py --setup-indexes          # Create both ChromaDB collections (run once)
    python ingest.py --data-dir sample_data/  # Ingest all JSON + text files
    python ingest.py --file sample_data/json/PO-2024-001.json
"""

import argparse
import asyncio
import json
import logging
import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chromadb

from prepdocslib.entityextractor import EntityExtractor
from prepdocslib.entitysearchmanager import EntitySearchManager

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config from .env
# ---------------------------------------------------------------------------
CHUNK_INDEX = os.environ.get("AZURE_SEARCH_CHUNK_INDEX", "procurement-chunks")
ENTITY_INDEX = os.environ.get("AZURE_SEARCH_ENTITY_INDEX", "procurement-entities")
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "./chromadb_data")

CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
EMB_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")
EMB_DIMENSIONS = int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1024"))
USE_ENTITY_EXTRACTION = os.environ.get("USE_GRAPHRAG_ENTITY_EXTRACTION", "true").lower() == "true"


def _make_chunk_id(filename: str, chunk_index: int) -> str:
    safe = hashlib.md5(filename.encode()).hexdigest()[:12]
    return f"{safe}-chunk-{chunk_index}"


def _chunk_text(text: str, max_tokens: int = 800, overlap: int = 100) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunks.append(" ".join(words[start:end]))
        start = end - overlap if end < len(words) else end
    return [c for c in chunks if c.strip()]


def _document_to_text(file_path: Path) -> tuple[str, dict]:
    if file_path.suffix == ".json":
        data = json.loads(file_path.read_text(encoding="utf-8"))
        text = json.dumps(data, indent=2)
        meta = {
            "document_type": data.get("document_type", "unknown"),
            "vendor": data.get("vendor", ""),
            "project": data.get("project", ""),
        }
    else:
        text = file_path.read_text(encoding="utf-8")
        meta = _extract_meta_from_text(text)
    return text, meta


def _extract_meta_from_text(text: str) -> dict:
    meta = {"document_type": "unknown", "vendor": "", "project": ""}
    lower = text.lower()
    if "purchase order" in lower:
        meta["document_type"] = "purchase_order"
    elif "invoice" in lower:
        meta["document_type"] = "invoice"
    elif "goods receipt" in lower:
        meta["document_type"] = "goods_receipt_note"

    for line in text.splitlines():
        if line.strip().startswith("VENDOR") or "Vendor" in line:
            continue
        if "Acme Technologies" in line:
            meta["vendor"] = "Acme Technologies Ltd"
        elif "Global Supplies" in line:
            meta["vendor"] = "Global Supplies Co"
        elif "FastTrack Logistics" in line:
            meta["vendor"] = "FastTrack Logistics"

        if "Bangalore" in line and ("Data Center" in line or "DC" in line):
            meta["project"] = "Bangalore Data Center Upgrade"
        elif "Mumbai" in line and "Office" in line:
            meta["project"] = "Mumbai Office Renovation"
        elif "Delhi" in line and ("Sales" in line or "Office" in line):
            meta["project"] = "Delhi Sales Office Setup"

    return meta


async def embed_text(openai_client: AsyncOpenAI, text: str) -> list[float]:
    resp = await openai_client.embeddings.create(
        model=EMB_DEPLOYMENT, input=text[:8000]
    )
    return resp.data[0].embedding


async def ingest_file(
    file_path: Path,
    chunk_col: chromadb.Collection,
    entity_extractor: EntityExtractor | None,
    entity_manager: EntitySearchManager | None,
    openai_client: AsyncOpenAI,
) -> None:
    logger.info("Ingesting: %s", file_path.name)
    text, meta = _document_to_text(file_path)
    chunks = _chunk_text(text)

    docs = []
    all_entities = []

    for i, chunk_text in enumerate(chunks):
        chunk_id = _make_chunk_id(file_path.name, i)
        embedding = await embed_text(openai_client, chunk_text)

        doc = {
            "id": chunk_id,
            "content": chunk_text,
            "sourcefile": file_path.name,
            "sourcepage": f"{file_path.name}#chunk={i}",
            "document_type": meta["document_type"],
            "vendor": meta["vendor"],
            "project": meta["project"],
            "contentVector": embedding,
        }
        docs.append(doc)

        if entity_extractor:
            entities = await entity_extractor.extract_entities_from_chunk(
                chunk_id=chunk_id,
                chunk_text=chunk_text,
                source_file=file_path.name,
                allowed_users=[],
                allowed_groups=[],
            )
            all_entities.extend(entities)

    chunk_col.upsert(
        ids=[d["id"] for d in docs],
        embeddings=[d["contentVector"] for d in docs],
        documents=[d["content"] for d in docs],
        metadatas=[{k: v for k, v in d.items() if k not in ("id", "content", "contentVector")} for d in docs],
    )
    logger.info("  Indexed %d chunks.", len(docs))

    if entity_manager and all_entities:
        for e in all_entities:
            phrase = f"{e['entity_type']}: {e['entity_value']}"
            e["embedding"] = await embed_text(openai_client, phrase)
        await entity_manager.upsert_entities(all_entities)
        logger.info("  Upserted %d entities.", len(all_entities))


async def main():
    parser = argparse.ArgumentParser(description="GraphRAG.POC ingestion CLI")
    parser.add_argument("--setup-indexes", action="store_true", help="Create both ChromaDB collections and exit")
    parser.add_argument("--data-dir", type=Path, help="Directory of files to ingest")
    parser.add_argument("--file", type=Path, help="Single file to ingest")
    args = parser.parse_args()

    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    if args.setup_indexes:
        logger.info("Creating ChromaDB collections at '%s'...", CHROMADB_PATH)
        for col_name in [CHUNK_INDEX, ENTITY_INDEX]:
            try:
                chroma_client.delete_collection(col_name)
            except Exception:
                pass
            chroma_client.create_collection(name=col_name, metadata={"hnsw:space": "cosine"})
            logger.info("Collection '%s' created.", col_name)
        return

    chunk_col = chroma_client.get_or_create_collection(name=CHUNK_INDEX, metadata={"hnsw:space": "cosine"})
    entity_col = chroma_client.get_or_create_collection(name=ENTITY_INDEX, metadata={"hnsw:space": "cosine"})
    entity_manager = EntitySearchManager(chroma_client, entity_col) if USE_ENTITY_EXTRACTION else None
    entity_extractor = EntityExtractor(openai_client, CHAT_DEPLOYMENT) if USE_ENTITY_EXTRACTION else None

    files: list[Path] = []
    if args.data_dir:
        files = list(args.data_dir.rglob("*.json")) + list(args.data_dir.rglob("*.txt"))
    elif args.file:
        files = [args.file]
    else:
        parser.print_help()
        return

    for f in sorted(files):
        try:
            await ingest_file(f, chunk_col, entity_extractor, entity_manager, openai_client)
        except Exception as e:
            logger.error("Failed to ingest %s: %s", f, e)

    logger.info("Ingestion complete. %d files processed.", len(files))


if __name__ == "__main__":
    asyncio.run(main())
