"""
Interactive GraphRAG query CLI.

Usage:
    python query.py                        # Interactive loop (agent mode)
    python query.py --query "..."          # Single query and exit
    python query.py --mode graphrag        # Force legacy multi-hop mode
    python query.py --mode vector          # Force vector-only mode
    python query.py --verbose              # Show thought steps
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
import chromadb

from approaches.graphragapproach import GraphRAGApproach
from approaches.agentrouter import AgentRouter
from approaches.retrievers import Retrievers

load_dotenv()
logging.basicConfig(level=logging.WARNING)

CHUNK_INDEX = os.environ.get("AZURE_SEARCH_CHUNK_INDEX", "procurement-chunks")
ENTITY_INDEX = os.environ.get("AZURE_SEARCH_ENTITY_INDEX", "procurement-entities")
CHROMADB_PATH = os.environ.get("CHROMADB_PATH", "./chromadb_data")

CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
EMB_DEPLOYMENT = os.environ.get("AZURE_OPENAI_EMB_DEPLOYMENT", "text-embedding-3-large")
EMB_DIMENSIONS = int(os.environ.get("AZURE_OPENAI_EMB_DIMENSIONS", "1024"))
MAX_HOPS = int(os.environ.get("GRAPHRAG_MAX_HOPS", "3"))
MAX_ENTITIES_PER_HOP = int(os.environ.get("GRAPHRAG_MAX_ENTITIES_PER_HOP", "10"))
ENTITY_TOP_K = int(os.environ.get("GRAPHRAG_ENTITY_TOP_K", "5"))
CHUNK_TOP_K = int(os.environ.get("GRAPHRAG_CHUNK_TOP_K", "15"))
AGENT_MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "8"))
AGENT_MAX_CHUNKS = int(os.environ.get("AGENT_MAX_CHUNKS", "20"))


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


async def run_query(
    approach: GraphRAGApproach,
    agent_router: AgentRouter,
    retrievers: Retrievers,
    query: str,
    mode: str = "auto",
) -> dict:
    if mode in ("auto", "agent"):
        print(f"\n[Agent] Running LLM tool-calling agent...")
        result = await agent_router.run(query)
    elif mode == "graphrag":
        print(f"\n[GraphRAG] Running agentic multi-hop retrieval...")
        result = await approach.run(query)
    elif mode == "vector":
        print(f"\n[Vector] Running keyword/vector search...")
        chunks = await retrievers.keyword_fallback(query)
        result = await approach._generate_answer(query, chunks, [], query_type="vector")
    else:
        print(f"\n[Agent] Running LLM tool-calling agent...")
        result = await agent_router.run(query)
    return result


async def main():
    parser = argparse.ArgumentParser(description="GraphRAG.POC query CLI")
    parser.add_argument("--query", type=str, help="Single query (non-interactive)")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["auto", "agent", "graphrag", "vector"],
        default="auto",
        help="Retrieval mode (default: auto → agent)",
    )
    parser.add_argument("--no-graphrag", action="store_true", help="(Deprecated) Use --mode vector instead")
    parser.add_argument("--verbose", action="store_true", help="Show thought steps")
    args = parser.parse_args()

    if args.no_graphrag:
        print("Warning: --no-graphrag is deprecated. Use --mode vector instead.", file=sys.stderr)
        args.mode = "vector"

    chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    chunk_col = chroma_client.get_or_create_collection(name=CHUNK_INDEX, metadata={"hnsw:space": "cosine"})
    entity_col = chroma_client.get_or_create_collection(name=ENTITY_INDEX, metadata={"hnsw:space": "cosine"})
    openai_client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )

    retrievers = Retrievers(
        openai_client=openai_client,
        emb_deployment=EMB_DEPLOYMENT,
        chunk_col=chunk_col,
        entity_col=entity_col,
        entity_top_k=ENTITY_TOP_K,
        chunk_top_k=CHUNK_TOP_K,
        max_entities_per_hop=MAX_ENTITIES_PER_HOP,
    )

    approach = GraphRAGApproach(
        openai_client=openai_client,
        chat_deployment=CHAT_DEPLOYMENT,
        emb_deployment=EMB_DEPLOYMENT,
        emb_dimensions=EMB_DIMENSIONS,
        chunk_col=chunk_col,
        entity_col=entity_col,
        max_hops=MAX_HOPS,
        max_entities_per_hop=MAX_ENTITIES_PER_HOP,
        entity_top_k=ENTITY_TOP_K,
        chunk_top_k=CHUNK_TOP_K,
        retrievers=retrievers,
    )

    agent_router = AgentRouter(
        openai_client=openai_client,
        chat_deployment=CHAT_DEPLOYMENT,
        retrievers=retrievers,
        max_turns=AGENT_MAX_TURNS,
        max_chunks=AGENT_MAX_CHUNKS,
    )

    if args.query:
        result = await run_query(approach, agent_router, retrievers, args.query, mode=args.mode)
        _print_result(result, verbose=args.verbose)
        return

    print("\nGraphRAG.POC — Procurement Query Interface")
    print(f"Mode: {args.mode} | Type your question, or 'quit' to exit.")
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

        result = await run_query(approach, agent_router, retrievers, query, mode=args.mode)
        _print_result(result, verbose=args.verbose)


if __name__ == "__main__":
    asyncio.run(main())
