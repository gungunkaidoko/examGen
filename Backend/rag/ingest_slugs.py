"""
ingest_slugs.py
===============
Reads enriched_chunks.jsonl and upserts to Pinecone with the full metadata schema:

  chunk_id, chapter_id, chapter_name, topic_slug, chunk_type,
  bloom_levels, difficulty, keywords, used_in_sets, q_count, text

Run after build_slug_chunks.py:
    python3 -m rag.ingest_slugs               # upsert new chunks only
    python3 -m rag.ingest_slugs --force       # wipe namespace first
    python3 -m rag.ingest_slugs --dry-run     # count chunks, no Pinecone calls
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ingest_slugs")

ENRICHED_JSONL = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "Knowledge_base", "book", "enriched_chunks.jsonl",
)
NAMESPACE = "ccc-book-v2"   # new namespace — keeps old vectors intact


def _build_pinecone_metadata(chunk: dict) -> dict:
    """
    Build Pinecone metadata from a chunk dict.
    Pinecone metadata values must be str / int / float / bool / list[str].

    concept_id is a stable, book-agnostic identifier: "{chapter_id}:{topic_slug}".
    It stays the same even if you re-ingest from a different CCC textbook,
    as long as the canonical slug list in build_slug_chunks.py is consistent.

    book_id identifies the source book so multi-book ingestion is possible.
    """
    # concept_id = chapter:slug — book-agnostic, stable across re-ingestion
    concept_id = f"{chunk['chapter_id']}:{chunk['topic_slug']}"

    return {
        # ── Core identification ───────────────────────────────────────────
        "chunk_id":       chunk["chunk_id"],
        "chapter_id":     chunk["chapter_id"],
        "chapter_name":   chunk["chapter_name"],
        "topic_slug":     chunk["topic_slug"],
        "chunk_type":     chunk["chunk_type"],
        # ── Book-agnostic concept identifier (future-proof) ───────────────
        "concept_id":     concept_id,           # e.g. "ch8:upi_payment"
        "book_id":        chunk.get("book_id", "ccc_arihant_v1"),
        # ── Tagging for retrieval filters ─────────────────────────────────
        "bloom_levels":   chunk.get("bloom_levels", ["remember"]),
        "difficulty":     chunk.get("difficulty", ["easy"]),
        "keywords":       chunk.get("keywords", []),
        # ── Tracking (start empty, updated at generation time) ────────────
        "used_in_sets":   chunk.get("used_in_sets", []),
        "q_count":        chunk.get("q_count", 0),
        # ── Cross-reference (shared topics across chapters) ───────────────
        "home_chapter":   chunk.get("home_chapter", chunk["chapter_id"]),
        "cross_ref":      chunk.get("cross_ref", []),
        # ── Full text for prompt injection ────────────────────────────────
        "text":           chunk["text"][:2000],
        # Section context
        "section_heading": chunk.get("section_heading", ""),
    }


def ingest_slugs(force: bool = False, dry_run: bool = False) -> int:
    if not os.path.exists(ENRICHED_JSONL):
        logger.error(f"enriched_chunks.jsonl not found at {ENRICHED_JSONL}")
        logger.error("Run first: python3 -m rag.build_slug_chunks")
        sys.exit(1)

    chunks = []
    with open(ENRICHED_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    logger.info(f"Loaded {len(chunks)} chunks from {ENRICHED_JSONL}")

    if dry_run:
        by_chapter: dict[str, int] = {}
        by_slug: dict[str, int] = {}
        for c in chunks:
            by_chapter[c["chapter_id"]] = by_chapter.get(c["chapter_id"], 0) + 1
            by_slug[c["topic_slug"]]    = by_slug.get(c["topic_slug"], 0) + 1
        logger.info("DRY RUN — no Pinecone calls")
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Unique slugs: {len(by_slug)}")
        for ch, cnt in sorted(by_chapter.items()):
            logger.info(f"  {ch}: {cnt} chunks")
        return len(chunks)

    from rag.embedder import get_embedder
    from rag.pinecone_store import get_store

    store   = get_embedder.__module__ and get_store()
    store   = get_store()
    embedder = get_embedder()

    if force:
        logger.info(f"--force: deleting namespace '{NAMESPACE}'…")
        try:
            store._index.delete(delete_all=True, namespace=NAMESPACE)
            logger.info("Namespace cleared.")
        except Exception as e:
            logger.warning(f"Could not clear namespace: {e}")

    # Embed all texts
    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks via Amazon Titan V2…")
    embeddings = embedder.embed_texts(texts)
    logger.info("Embedding complete.")

    # Build vector tuples (id, embedding, metadata)
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vec_id   = chunk["chunk_id"]          # already unique slug-based id
        metadata = _build_pinecone_metadata(chunk)
        vectors.append((vec_id, embedding, metadata))

    # Upsert in batches
    from config import RAG_BATCH_SIZE
    total = 0
    for i in range(0, len(vectors), RAG_BATCH_SIZE):
        batch = vectors[i: i + RAG_BATCH_SIZE]
        records = [{"id": v[0], "values": v[1], "metadata": v[2]} for v in batch]
        store._index.upsert(vectors=records, namespace=NAMESPACE)
        total += len(batch)
        logger.info(f"  Upserted {total}/{len(vectors)} vectors")

    stats = store.describe()
    logger.info(f"✓ Done — {total} vectors in namespace '{NAMESPACE}'")
    logger.info(f"Index stats: {stats.get('namespaces', {})}")
    return total


def main():
    parser = argparse.ArgumentParser(description="Ingest enriched slug chunks into Pinecone")
    parser.add_argument("--force",   action="store_true", help="Wipe namespace before upserting")
    parser.add_argument("--dry-run", action="store_true", help="Count chunks, no API calls")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CCC SLUG-BASED RAG INGESTION")
    logger.info("=" * 60)
    logger.info(f"  Source   : {ENRICHED_JSONL}")
    logger.info(f"  Namespace: {NAMESPACE}")
    logger.info(f"  Force    : {args.force}")
    logger.info(f"  Dry-run  : {args.dry_run}")

    total = ingest_slugs(force=args.force, dry_run=args.dry_run)
    logger.info(f"✓ Ingestion complete — {total} vectors")


if __name__ == "__main__":
    main()
