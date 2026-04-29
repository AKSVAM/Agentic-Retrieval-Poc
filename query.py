"""
Interactive GraphRAG query CLI.

Usage:
    python query.py                   # Interactive loop
    python query.py --query "..."     # Single query and exit
    python query.py --no-graphrag     # Force keyword-only mode
"""

import argparse
import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chromadb

from approaches.graphragapproach import GraphRAGApproach
from approaches.queryrouter import QueryRouter

load_dotenv()
logging.basicConfig(level=logging.WARNING)

CHUNK_INDEX = os.environ.get("AZURE_SEARCH_CHUNK_INDEX", "procurement-chunks")
ENTITY_INDEX = os.environ.get("AZURE_SEARCH_ENTITY_INDEX", "procurement-entities")
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "./chromadb_data")

CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
EMB_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")
EMB_DIMENSIONS = int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1024"))
USE_GRAPHRAG = os.environ.get("USE_GRAPHRAG", "true").lower() == "true"
MAX_HOPS = int(os.environ.get("GRAPHRAG_MAX_HOPS", "3"))
ENTITY_TOP_K = int(os.environ.get("GRAPHRAG_ENTITY_TOP_K", "5"))
CHUNK_TOP_K = int(os.environ.get("GRAPHRAG_CHUNK_TOP_K", "15"))


def _print_result(result: dict, verbose: bool = False) -> None:
    print("\n" + "=" * 70)
    print(f"Query type : {result.get('query_type', 'unknown')}")
    print("=" * 70)
    print(result.get("answer", "(no answer)"))

    citations = result.get("citations", [])
    if citations:
        print(f"\nCitations  : {', '.join(citations)}")

    if verbose:
        print("\nThought steps:")
        for step in result.get("thought_steps", []):
            print(f"  [{step.get('title')}] {step.get('description')}")
    print("=" * 70 + "\n")


async def run_query(approach: GraphRAGApproach, router: QueryRouter, query: str, force_graphrag: bool = False) -> dict:
    use_graph = force_graphrag or (USE_GRAPHRAG and router.should_use_graphrag(query))
    if use_graph:
        print(f"\n[GraphRAG] Running agentic multi-hop retrieval...")
        result = await approach.run(query)
    else:
        print(f"\n[Standard] Running keyword/vector search...")
        result = await approach.run(query)
        result["query_type"] = "standard_vector"
    return result


async def main():
    parser = argparse.ArgumentParser(description="GraphRAG.POC query CLI")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    parser.add_argument("--no-graphrag", action="store_true", help="Force keyword-only mode")
    parser.add_argument("--verbose", action="store_true", help="Show thought steps")
    args = parser.parse_args()

    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    chunk_col = chroma_client.get_or_create_collection(name=CHUNK_INDEX, metadata={"hnsw:space": "cosine"})
    entity_col = chroma_client.get_or_create_collection(name=ENTITY_INDEX, metadata={"hnsw:space": "cosine"})
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    approach = GraphRAGApproach(
        openai_client=openai_client,
        chat_deployment=CHAT_DEPLOYMENT,
        emb_deployment=EMB_DEPLOYMENT,
        emb_dimensions=EMB_DIMENSIONS,
        chunk_col=chunk_col,
        entity_col=entity_col,
        max_hops=MAX_HOPS,
        entity_top_k=ENTITY_TOP_K,
        chunk_top_k=CHUNK_TOP_K,
    )
    router = QueryRouter()

    if args.query:
        result = await run_query(approach, router, args.query, force_graphrag=not args.no_graphrag)
        _print_result(result, verbose=args.verbose)
        return

    print("\nGraphRAG.POC — Procurement Query Interface")
    print("Type your question, or 'quit' to exit.")
    print("Tip: Try 'Show me all transactions with Acme Technologies'\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        result = await run_query(approach, router, query)
        _print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
