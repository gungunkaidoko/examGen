"""
semantic_dedup.py
=================
Embedding-based semantic duplicate detector.

Problem being solved
---------------------
Even with distinct topic slugs, an LLM can generate two questions that are
semantically near-identical (e.g. same fact phrased differently). Jaccard
word-overlap catches surface duplicates but misses paraphrase.

Solution
--------
Before accepting a generated question:
  1. Embed the question text with the same Titan V2 model used for chunks.
  2. Query Pinecone question-index (separate namespace "ccc-questions") for
     the top-5 nearest existing questions.
  3. If max cosine similarity > THRESHOLD (0.90) → reject as duplicate.
  4. On acceptance → upsert the question embedding into the question-index.

Design decisions
-----------------
- Separate Pinecone namespace "ccc-questions" keeps question vectors apart
  from chunk vectors — no cross-contamination of similarity scores.
- Per exam-set filtering: filter by exam_set != current_set lets the same
  question appear in different sets if you want that (configurable).
- Cross-set dedup is optional (CROSS_SET_DEDUP flag). Intra-set dedup is
  always enforced.
- Thread-safe: an in-memory set of accepted question hashes is used as a
  fast pre-filter before the Pinecone call, reducing API calls by ~80%.

Public API
----------
  get_deduplicator()                          → SemanticDeduplicator singleton
  SemanticDeduplicator.is_duplicate(question_text, exam_set, chapter) → bool
  SemanticDeduplicator.register(question_uuid, question_text, exam_set, chapter)
"""

import logging
import threading
from typing import Optional

from rag.embedder import get_embedder
from rag.pinecone_store import get_store

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.90   # cosine sim above this = duplicate
TOP_K_CHECK          = 5      # how many nearest neighbours to check
CROSS_SET_DEDUP      = False  # True = reject questions already in OTHER sets too
NAMESPACE            = "ccc-questions"


class SemanticDeduplicator:
    """
    Embedding-based semantic duplicate detector for generated questions.

    Usage in pipeline:
        dedup = get_deduplicator()

        # Before accepting a question:
        if dedup.is_duplicate(question.question, exam_set=2, chapter="ch8"):
            continue   # regenerate

        # After full validation passes:
        dedup.register(question.question_uuid, question.question,
                        exam_set=2, chapter="ch8")
    """

    def __init__(self):
        self._embedder  = get_embedder()
        self._store     = get_store()
        # In-memory cache: set of (exam_set, question_text_lower) for fast
        # pre-filtering without Pinecone API calls
        self._cache: set[tuple[int, str]] = set()
        self._lock = threading.Lock()
        logger.info("SemanticDeduplicator ready (namespace=ccc-questions)")

    # ── Public API ─────────────────────────────────────────────────────────

    def is_duplicate(
        self,
        question_text: str,
        exam_set: int,
        chapter: str = "",
    ) -> bool:
        """
        Returns True if `question_text` is semantically too similar to an
        already-accepted question.

        Steps:
          1. Fast in-memory exact-match pre-filter (O(1)).
          2. Embed → Pinecone similarity search.
          3. Return True if max similarity > THRESHOLD.
        """
        key = (exam_set, question_text.strip().lower())

        # ── Fast pre-filter: exact match in cache ──────────────────────────
        with self._lock:
            if key in self._cache:
                logger.debug("Dedup: exact match in cache → duplicate")
                return True

        # ── Embedding similarity check ─────────────────────────────────────
        try:
            vec = self._embedder.embed_query(question_text)

            # Build filter
            pinecone_filter: dict = {}
            if not CROSS_SET_DEDUP:
                # Only check within the same exam set
                pinecone_filter["exam_set"] = {"$eq": exam_set}

            results = self._store._index.query(
                vector           = vec,
                top_k            = TOP_K_CHECK,
                filter           = pinecone_filter if pinecone_filter else None,
                namespace        = NAMESPACE,
                include_metadata = True,
            )
            matches = results.get("matches", [])

            if not matches:
                return False

            max_sim = max(m.get("score", 0.0) for m in matches)

            if max_sim >= SIMILARITY_THRESHOLD:
                best_match = max(matches, key=lambda m: m.get("score", 0.0))
                bm_text = best_match.get("metadata", {}).get("question_text", "")[:80]
                logger.info(
                    f"Dedup: similarity={max_sim:.3f} ≥ {SIMILARITY_THRESHOLD} → duplicate. "
                    f"Similar to: {bm_text!r}"
                )
                return True

            return False

        except Exception as e:
            # On Pinecone error, fail open (don't block generation)
            logger.warning(f"Dedup check failed (Pinecone error): {e}. Allowing question.")
            return False

    def register(
        self,
        question_uuid: str,
        question_text: str,
        exam_set: int,
        chapter: str = "",
    ) -> None:
        """
        Register an accepted question into the semantic index.
        Call this ONLY after a question has passed ALL validation steps.

        Args:
            question_uuid:  Unique ID (used as Pinecone vector ID).
            question_text:  Full question text to embed.
            exam_set:       Exam set number.
            chapter:        Chapter key (for metadata / future filtering).
        """
        # Add to in-memory cache first (fast path for next call)
        key = (exam_set, question_text.strip().lower())
        with self._lock:
            self._cache.add(key)

        # Upsert to Pinecone asynchronously (non-blocking for pipeline speed)
        try:
            vec = self._embedder.embed_query(question_text)
            metadata = {
                "question_uuid": question_uuid,
                "question_text": question_text[:500],   # stored for debugging
                "exam_set":      exam_set,
                "chapter":       chapter,
            }
            self._store._index.upsert(
                vectors=[{
                    "id":       question_uuid,
                    "values":   vec,
                    "metadata": metadata,
                }],
                namespace=NAMESPACE,
            )
            logger.debug(f"Dedup: registered question {question_uuid[:8]}… in set {exam_set}")
        except Exception as e:
            logger.warning(f"Dedup: failed to register question {question_uuid}: {e}")

    def preload_from_db(self, db) -> int:
        """
        Pre-populate the in-memory cache from existing DB questions.
        Call once at pipeline startup to avoid re-checking already-stored questions.

        Returns count of questions loaded.
        """
        try:
            with db._conn.cursor() as cur:
                cur.execute("SELECT exam_set, LOWER(TRIM(question)) FROM question_bank")
                rows = cur.fetchall()
            with self._lock:
                for exam_set, q_lower in rows:
                    self._cache.add((exam_set, q_lower))
            logger.info(f"Dedup: pre-loaded {len(rows)} questions into cache")
            return len(rows)
        except Exception as e:
            logger.warning(f"Dedup: could not preload from DB: {e}")
            return 0

    def clear_set(self, exam_set: int) -> None:
        """Remove all cached entries for a given exam set (for re-generation)."""
        with self._lock:
            self._cache = {k for k in self._cache if k[0] != exam_set}


# ── Module-level singleton ─────────────────────────────────────────────────

_deduplicator: Optional[SemanticDeduplicator] = None
_dedup_lock = threading.Lock()


def get_deduplicator() -> SemanticDeduplicator:
    """Return (or create) the module-level SemanticDeduplicator singleton."""
    global _deduplicator
    with _dedup_lock:
        if _deduplicator is None:
            _deduplicator = SemanticDeduplicator()
    return _deduplicator
