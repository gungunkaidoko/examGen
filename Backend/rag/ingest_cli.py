"""
Ingest CLI
----------
Run this once (or whenever the knowledge base changes) to populate Pinecone.

Usage:
    python -m rag.ingest_cli                        # ingest all content types
    python -m rag.ingest_cli --force                # wipe namespace first
    python -m rag.ingest_cli --types content mcq    # only prose + MCQ chunks
    python -m rag.ingest_cli --jsonl path/to/file.jsonl
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_cli")

from config import RAG_CHUNKS_JSONL, RAG_MARKDOWN_FILE
from rag.ingestion import ingest
from rag.pinecone_store import get_store


def main():
    parser = argparse.ArgumentParser(description="Ingest CCC book into Pinecone")
    parser.add_argument(
        "--jsonl",
        default=RAG_CHUNKS_JSONL,
        help="Path to ccc_book_chunks.jsonl (default: config.RAG_CHUNKS_JSONL)",
    )
    parser.add_argument(
        "--md",
        default=RAG_MARKDOWN_FILE,
        help="Path to ccc_book.md fallback (default: config.RAG_MARKDOWN_FILE)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing namespace before ingesting",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        default=None,
        metavar="TYPE",
        help="content types to ingest: content mcq truefalse glossary practice_set",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print index stats after ingestion",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CCC RAG INGESTION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"  JSONL source : {args.jsonl}")
    logger.info(f"  MD fallback  : {args.md}")
    logger.info(f"  Force wipe   : {args.force}")
    logger.info(f"  Type filter  : {args.types or 'all'}")

    total = ingest(
        jsonl_path=args.jsonl,
        md_path=args.md,
        force_reingest=args.force,
        content_type_filter=args.types,
    )

    logger.info(f"✓ Done — {total} vectors in Pinecone")

    if args.stats:
        stats = get_store().describe()
        logger.info(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
