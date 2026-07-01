"""
Pinecone Store
--------------
Manages the Pinecone index lifecycle and vector upserts.

Responsibilities:
  - Create the index if it doesn't exist (cosine metric, 1024 dims).
  - Upsert vectors in configurable batches with rate-limit back-off.
  - Expose query() for similarity search with optional metadata filter.
  - Expose delete_namespace() for re-ingestion without full index teardown.

Index layout
------------
  namespace  : "ccc-book"  (all CCC content lives here)
  vector id  : chunk["id"] from the JSONL  (e.g. "ccc-0042")
  metadata   : book, chapter, section, subsection, content_type,
               question_type, source_file, char_count, chunk_index
"""

import logging
import time
from typing import Optional

from pinecone import Pinecone, ServerlessSpec

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_ENVIRONMENT,
    PINECONE_HOST,
    EMBEDDING_DIMENSIONS,
    RAG_BATCH_SIZE,
)

logger = logging.getLogger(__name__)

NAMESPACE = "ccc-book"
METRIC = "cosine"


class PineconeStore:
    """
    Thin wrapper around the Pinecone Python SDK v3+.

    Usage:
        store = PineconeStore()
        store.upsert_vectors(vectors)          # [(id, embedding, metadata), ...]
        results = store.query(vector, top_k=5, filter={...})
    """

    def __init__(self):
        self._pc = Pinecone(api_key=PINECONE_API_KEY)
        self._index = self._ensure_index()
        logger.info(f"PineconeStore ready — index={PINECONE_INDEX_NAME}, ns={NAMESPACE}")

    # ── Index lifecycle ───────────────────────────────────────────────────────

    def _ensure_index(self):
        """Create the index if absent; return the Index object."""
        existing = [idx.name for idx in self._pc.list_indexes()]
        if PINECONE_INDEX_NAME not in existing:
            logger.info(f"Creating Pinecone index '{PINECONE_INDEX_NAME}'…")
            self._pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=EMBEDDING_DIMENSIONS,
                metric=METRIC,
                spec=ServerlessSpec(cloud="aws", region=PINECONE_ENVIRONMENT),
            )
            # Wait until the index is ready
            for _ in range(30):
                status = self._pc.describe_index(PINECONE_INDEX_NAME).status
                if status.get("ready"):
                    break
                time.sleep(2)
            logger.info(f"Index '{PINECONE_INDEX_NAME}' created and ready.")
        else:
            logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")

        # Connect via host if provided (avoids extra DNS lookup)
        if PINECONE_HOST:
            return self._pc.Index(host=PINECONE_HOST)
        return self._pc.Index(PINECONE_INDEX_NAME)

    def describe(self) -> dict:
        """Return index stats (vector count, namespaces, etc.)."""
        return self._index.describe_index_stats()

    # ── Write ─────────────────────────────────────────────────────────────────

    def upsert_vectors(
        self,
        vectors: list[tuple[str, list[float], dict]],
        namespace: str = NAMESPACE,
    ) -> int:
        """
        Upsert a list of (id, embedding, metadata) tuples.
        Splits into batches of RAG_BATCH_SIZE with back-off on errors.
        Returns total vectors upserted.
        """
        total = 0
        for i in range(0, len(vectors), RAG_BATCH_SIZE):
            batch = vectors[i: i + RAG_BATCH_SIZE]
            records = [
                {"id": v[0], "values": v[1], "metadata": v[2]}
                for v in batch
            ]
            self._upsert_with_backoff(records, namespace)
            total += len(batch)
            logger.info(
                f"  Upserted {total}/{len(vectors)} vectors "
                f"(batch {i // RAG_BATCH_SIZE + 1})"
            )
        return total

    def _upsert_with_backoff(self, records: list[dict], namespace: str, retries: int = 5):
        for attempt in range(retries):
            try:
                self._index.upsert(vectors=records, namespace=namespace)
                return
            except Exception as e:
                wait = 2 ** attempt
                logger.warning(
                    f"Pinecone upsert error (attempt {attempt + 1}/{retries}): {e}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)
        raise RuntimeError(f"Pinecone upsert failed after {retries} retries")

    # ── Read ──────────────────────────────────────────────────────────────────

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        filter: Optional[dict] = None,
        namespace: str = NAMESPACE,
        include_metadata: bool = True,
    ) -> list[dict]:
        """
        Run a similarity query. Returns a list of match dicts:
            [{"id": ..., "score": ..., "metadata": {...}}, ...]
        """
        kwargs = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": include_metadata,
            "namespace": namespace,
        }
        if filter:
            kwargs["filter"] = filter

        response = self._index.query(**kwargs)
        return [
            {
                "id": m["id"],
                "score": m["score"],
                "metadata": m.get("metadata", {}),
            }
            for m in response.get("matches", [])
        ]

    def delete_namespace(self, namespace: str = NAMESPACE) -> None:
        """Delete all vectors in a namespace (for full re-ingestion)."""
        self._index.delete(delete_all=True, namespace=namespace)
        logger.info(f"Deleted all vectors in namespace '{namespace}'")


# Module-level singleton
_store: Optional[PineconeStore] = None


def get_store() -> PineconeStore:
    """Return the module-level PineconeStore singleton."""
    global _store
    if _store is None:
        _store = PineconeStore()
    return _store
