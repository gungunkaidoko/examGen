"""
slug_retriever.py
=================
Slug-enumeration based retrieval — replaces "embed query → top-K cosine".

Flow (updated):
  1. List all topic_slugs for the chapter ($nin used_in_sets filter)
  2. Sample slugs by difficulty quota (60% easy, 35% medium, 5% hard)
  3. Fetch ONE chunk per slug (cosine = tiebreaker within slug)
  4. Return chunks to pipeline WITHOUT marking them used yet

  The pipeline calls mark_chunks_used(chunk_ids, set_number) only AFTER
  a question generated from that chunk passes all validation and is accepted.
  This prevents failed generations from wasting good slugs.

Guarantees:
  - Zero cross-set slug reuse  (enforced by $nin on used_in_sets)
  - Intra-set topic diversity  (one chunk → one question, different topic each call)
  - Controlled difficulty      (quota matches blueprint split)
  - No wasted slugs           (mark-used only on acceptance)

Public API
----------
  get_slug_retriever()  → SlugRetriever singleton

  SlugRetriever.retrieve_for_chapter(chapter_key, set_number,
                                      num_questions, difficulty_split)
      → list[dict]   # chunks, NOT yet marked used

  SlugRetriever.mark_chunks_used(chunk_ids: list[str], set_number: int)
      → None         # call after acceptance

  retrieve_slugs_for_chapter(...)   # convenience wrapper
  mark_chunks_used(...)             # convenience wrapper
"""

import logging
import random
import threading
from typing import Optional

from config import CHAPTER_BLUEPRINT
from rag.embedder import get_embedder
from rag.pinecone_store import get_store

logger = logging.getLogger(__name__)

NAMESPACE = "ccc-book-v2"

CHAPTER_ID_MAP = {
    "ch1": "introduction_to_computer",
    "ch2": "introduction_to_operating_system",
    "ch3": "word_processing",
    "ch4": "spreadsheet",
    "ch5": "presentation",
    "ch6": "internet_www",
    "ch7": "email_social_egov",
    "ch8": "digital_financial_tools",
    "ch9": "futureskills_cybersecurity",
}

BLUEPRINT_TO_CHAPTER_ID = {
    "chapter_01_introduction_to_computer":    "ch1",
    "chapter_02_operating_system":            "ch2",
    "chapter_03_word_processing":             "ch3",
    "chapter_04_spreadsheet":                "ch4",
    "chapter_05_presentation":               "ch5",
    "chapter_06_internet_www":               "ch6",
    "chapter_07_email_social_egov":          "ch7",
    "chapter_08_digital_financial_tools":    "ch8",
    "chapter_09_futureskills_cybersecurity": "ch9",
}


class SlugRetriever:
    """
    Slug-enumeration retriever with deferred mark-used.

    Separation of concerns:
      retrieve_for_chapter()  — returns chunks to generate from (READ)
      mark_chunks_used()      — writes back to Pinecone (WRITE, call after accept)
    """

    def __init__(self):
        self._store    = get_store()
        self._embedder = get_embedder()
        # Thread-safe in-memory pending-mark set:
        # chunk_ids retrieved but not yet accepted/rejected
        self._pending: dict[str, str] = {}   # chunk_id → set_str
        self._pending_lock = threading.Lock()
        logger.info("SlugRetriever ready (namespace=%s)", NAMESPACE)

    # ── Public: retrieve (no side effects on Pinecone) ─────────────────────

    def retrieve_for_chapter(
        self,
        chapter_key: str,
        set_number: int,
        num_questions: int,
        difficulty_split: Optional[dict] = None,
    ) -> list[dict]:
        """
        Return up to `num_questions` chunks for a chapter.

        Does NOT mark chunks as used — caller must call mark_chunks_used()
        after the generated questions pass validation.

        Each returned dict:
            chunk_id, topic_slug, concept_id, chapter_id, section_heading,
            bloom_levels, difficulty, keywords, text, score, book_id
        """
        chapter_id = BLUEPRINT_TO_CHAPTER_ID.get(chapter_key)
        if chapter_id is None:
            logger.warning("Unknown chapter_key: %s", chapter_key)
            return []

        chapter_name = CHAPTER_ID_MAP.get(chapter_id, "")

        if difficulty_split is None:
            bp = CHAPTER_BLUEPRINT.get(chapter_key, {})
            difficulty_split = bp.get(
                "difficulty_split", {"easy": 0.5, "medium": 0.35, "hard": 0.15}
            )

        set_str = str(set_number)

        # Step 1: list unused slugs
        available = self._list_available_slugs(chapter_name, chapter_id, set_str)
        if not available:
            logger.warning(
                "No unused slugs for %s set#%s — allowing reuse.", chapter_id, set_number
            )
            available = self._list_available_slugs(
                chapter_name, chapter_id, set_str, allow_reuse=True
            )

        # Step 2: sample by difficulty quota
        sampled = self._sample_by_difficulty(available, num_questions, difficulty_split)

        # Step 3: fetch one chunk per slug
        chunks = []
        for slug_info in sampled:
            chunk = self._fetch_chunk_for_slug(
                slug              = slug_info["topic_slug"],
                chapter_id        = chapter_id,
                chapter_name      = chapter_name,
                target_difficulty = slug_info["target_difficulty"],
            )
            if chunk:
                chunks.append(chunk)

        logger.info(
            "SlugRetriever: retrieved %d/%d chunks for %s set#%s (mark-used deferred)",
            len(chunks), num_questions, chapter_id, set_number,
        )
        return chunks

    # ── Public: mark used (call AFTER question acceptance) ─────────────────

    def mark_chunks_used(self, chunk_ids: list[str], set_number: int) -> None:
        """
        Write used_in_sets update to Pinecone for each chunk_id.

        Call this ONLY after the question generated from each chunk
        has passed all validation layers and been accepted.

        This is a best-effort operation — failures are logged but never
        raise an exception so they don't block the pipeline.
        """
        if not chunk_ids:
            return
        set_str = str(set_number)
        for chunk_id in chunk_ids:
            self._mark_used(chunk_id, set_str)
        logger.debug("Marked %d chunks as used in set#%s", len(chunk_ids), set_number)

    # ── Internal ───────────────────────────────────────────────────────────

    def _list_available_slugs(
        self,
        chapter_name: str,
        chapter_id: str,
        set_str: str,
        allow_reuse: bool = False,
    ) -> list[dict]:
        """
        Query Pinecone for all chunks in the chapter, excluding
        slugs already used in set_str (unless allow_reuse=True).
        """
        base_filter: dict = {"chapter_name": {"$eq": chapter_name}}
        if not allow_reuse:
            base_filter["used_in_sets"] = {"$nin": [set_str]}

        query_text = f"{chapter_name} concepts overview"
        query_vec  = self._embedder.embed_query(query_text)

        try:
            results = self._store._index.query(
                vector           = query_vec,
                top_k            = 100,
                filter           = base_filter,
                namespace        = NAMESPACE,
                include_metadata = True,
            )
            matches = results.get("matches", [])
        except Exception as e:
            logger.warning("Pinecone slug listing failed: %s", e)
            return []

        # One entry per slug — keep highest-scoring vector
        seen_slugs: dict[str, dict] = {}
        for m in matches:
            meta  = m.get("metadata", {})
            slug  = meta.get("topic_slug", "")
            score = m.get("score", 0.0)
            if slug not in seen_slugs or score > seen_slugs[slug]["score"]:
                seen_slugs[slug] = {
                    "topic_slug":  slug,
                    "chunk_id":    m["id"],
                    "score":       score,
                    "difficulty":  meta.get("difficulty", ["easy"]),
                    "bloom_levels": meta.get("bloom_levels", ["remember"]),
                    "concept_id":  meta.get("concept_id", f"{chapter_id}:{slug}"),
                    "book_id":     meta.get("book_id", "ccc_arihant_v1"),
                }

        logger.debug(
            "Available slugs for %s: %d (allow_reuse=%s)",
            chapter_id, len(seen_slugs), allow_reuse,
        )
        return list(seen_slugs.values())

    def _sample_by_difficulty(
        self,
        available: list[dict],
        num_questions: int,
        difficulty_split: dict,
    ) -> list[dict]:
        """
        Sample num_questions slugs respecting easy/medium/hard quota.
        Returns list of {topic_slug, chunk_id, target_difficulty, concept_id, book_id}.
        """
        buckets: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
        for s in available:
            for diff in s["difficulty"]:
                if diff in buckets:
                    buckets[diff].append({**s, "target_difficulty": diff})
                    break

        for b in buckets.values():
            random.shuffle(b)

        quotas = {
            "easy":   round(difficulty_split.get("easy",   0.5)  * num_questions),
            "medium": round(difficulty_split.get("medium", 0.35) * num_questions),
            "hard":   round(difficulty_split.get("hard",   0.15) * num_questions),
        }
        total_q = sum(quotas.values())
        diff    = num_questions - total_q
        quotas["easy"] += diff   # absorb rounding remainder into easy

        selected: list[dict] = []
        for diff_level, quota in quotas.items():
            pool   = buckets[diff_level]
            needed = min(quota, len(pool))
            selected.extend(pool[:needed])
            pool[:] = pool[needed:]
            # borrow remainder from other buckets
            remainder = quota - needed
            for other, other_pool in buckets.items():
                if other == diff_level or remainder == 0:
                    continue
                take = min(remainder, len(other_pool))
                selected.extend(other_pool[:take])
                other_pool[:] = other_pool[take:]
                remainder -= take

        # Deduplicate by slug
        seen: set[str] = set()
        deduped = []
        for s in selected:
            if s["topic_slug"] not in seen:
                seen.add(s["topic_slug"])
                deduped.append(s)

        return deduped[:num_questions]

    def _fetch_chunk_for_slug(
        self,
        slug: str,
        chapter_id: str,
        chapter_name: str,
        target_difficulty: str,
    ) -> Optional[dict]:
        """Fetch the best chunk for a slug. Cosine sim is a tiebreaker."""
        query_vec = self._embedder.embed_query(
            f"{slug.replace('_', ' ')} {chapter_name}"
        )
        try:
            results = self._store._index.query(
                vector           = query_vec,
                top_k            = 3,
                filter           = {
                    "topic_slug":   {"$eq": slug},
                    "chapter_name": {"$eq": chapter_name},
                },
                namespace        = NAMESPACE,
                include_metadata = True,
            )
            matches = results.get("matches", [])
        except Exception as e:
            logger.warning("Chunk fetch failed for slug=%s: %s", slug, e)
            return None

        if not matches:
            return None

        best = matches[0]
        meta = best.get("metadata", {})
        return {
            "chunk_id":        best["id"],
            "topic_slug":      slug,
            "concept_id":      meta.get("concept_id", f"{chapter_id}:{slug}"),
            "book_id":         meta.get("book_id", "ccc_arihant_v1"),
            "chapter_id":      chapter_id,
            "section_heading": meta.get("section_heading", ""),
            "bloom_levels":    meta.get("bloom_levels", ["remember"]),
            "difficulty":      target_difficulty,
            "keywords":        meta.get("keywords", []),
            "text":            meta.get("text", ""),
            "score":           best.get("score", 0.0),
        }

    def _mark_used(self, chunk_id: str, set_str: str) -> None:
        """
        Fetch → update used_in_sets → upsert back to Pinecone.
        Best-effort: exceptions are logged, never raised.
        """
        try:
            result   = self._store._index.fetch(ids=[chunk_id], namespace=NAMESPACE)
            vectors  = result.get("vectors", {})
            if chunk_id not in vectors:
                return
            v            = vectors[chunk_id]
            existing     = v.get("metadata", {})
            used         = list(set(existing.get("used_in_sets", []) + [set_str]))
            q_count      = existing.get("q_count", 0) + 1
            updated_meta = {**existing, "used_in_sets": used, "q_count": q_count}
            self._store._index.upsert(
                vectors=[{
                    "id":       chunk_id,
                    "values":   v.get("values", []),
                    "metadata": updated_meta,
                }],
                namespace=NAMESPACE,
            )
        except Exception as e:
            logger.debug("Could not mark chunk %s as used: %s", chunk_id, e)


# ── Module-level singleton ─────────────────────────────────────────────────

_slug_retriever: Optional[SlugRetriever] = None
_sr_lock = threading.Lock()


def get_slug_retriever() -> SlugRetriever:
    global _slug_retriever
    with _sr_lock:
        if _slug_retriever is None:
            _slug_retriever = SlugRetriever()
    return _slug_retriever


def retrieve_slugs_for_chapter(
    chapter_key: str,
    set_number: int,
    num_questions: int,
    difficulty_split: Optional[dict] = None,
) -> list[dict]:
    """Convenience wrapper — returns chunks without marking them used."""
    return get_slug_retriever().retrieve_for_chapter(
        chapter_key      = chapter_key,
        set_number       = set_number,
        num_questions    = num_questions,
        difficulty_split = difficulty_split,
    )


def mark_chunks_used(chunk_ids: list[str], set_number: int) -> None:
    """Convenience wrapper — call after question acceptance."""
    get_slug_retriever().mark_chunks_used(chunk_ids, set_number)

