"""
Retriever
---------
Four-stage retrieval pipeline:

  Stage 1 — Pinecone similarity search + metadata filter (Top 20)
      Embed the query with Amazon Titan V2.
      Pass chapter + content_type as a Pinecone metadata filter so the
      index itself scopes the search — only relevant vectors are ranked.
      Fetch the top-20 most similar vectors within that filtered space.

  Stage 2 — MMR Re-ranking
      Greedy Maximal Marginal Relevance over the 20 candidates.
      Balances relevance (cosine sim to query) vs. redundancy (bigram
      Jaccard overlap between already-selected chunks).
      λ = 0.6  →  60 % relevance, 40 % diversity.

  Stage 3 — Return Top 5–8 chunks
      Default is 8; callers can pass top_k=5 for tighter prompts.

Public API
----------
  get_retriever()                 → RAGRetriever singleton
  retrieve_for_chapter(...)       → convenience one-liner for the pipeline
"""

import logging
import math
from typing import Optional

from rag.embedder import get_embedder
from rag.pinecone_store import get_store, NAMESPACE

logger = logging.getLogger(__name__)

# ── Pipeline constants ────────────────────────────────────────────────────────

SIMILARITY_CANDIDATES = 20   # Stage 1: always fetch exactly 20 from Pinecone
DEFAULT_TOP_K = 8            # Stage 4: default final chunks returned (5–8)
MMR_LAMBDA = 0.6             # λ: 0 = pure diversity, 1 = pure relevance


# ── Helpers ───────────────────────────────────────────────────────────────────

def _text_overlap_score(text: str, others: list[str]) -> float:
    """
    Bigram Jaccard overlap — lightweight redundancy proxy for MMR.
    Returns the MAX overlap between `text` and any doc already selected.
    No extra embedding calls needed.
    """
    def bigrams(t: str) -> set:
        words = t.lower().split()
        return {(words[i], words[i + 1]) for i in range(len(words) - 1)} if len(words) > 1 else set()

    bg = bigrams(text)
    if not bg:
        return 0.0

    max_j = 0.0
    for other in others:
        other_bg = bigrams(other)
        if not other_bg:
            continue
        union = bg | other_bg
        jaccard = len(bg & other_bg) / len(union) if union else 0.0
        max_j = max(max_j, jaccard)
    return max_j


# ── Stage 1: Build Pinecone metadata filter ───────────────────────────────────

def _build_pinecone_filter(
    chapter: Optional[str],
    content_types: Optional[list[str]],
) -> Optional[dict]:
    """
    Build a Pinecone metadata filter dict from chapter + content_type args.
    Returns None when no filter is needed (unscoped search).
    """
    conditions: dict = {}

    if chapter:
        conditions["chapter"] = {"$eq": chapter.strip()}

    if content_types:
        if len(content_types) == 1:
            conditions["content_type"] = {"$eq": content_types[0]}
        else:
            conditions["content_type"] = {"$in": content_types}

    return conditions if conditions else None


# ── Stage 3: MMR re-ranker ────────────────────────────────────────────────────

def _mmr_rerank(
    candidates: list[dict],
    top_k: int,
    lambda_: float = MMR_LAMBDA,
) -> list[dict]:
    """
    Stage 3 — Greedy MMR selection.

    Iteratively picks the candidate that maximises:
        MMR(d) = λ · sim(query, d)  −  (1 − λ) · max_{r ∈ selected} overlap(r, d)

    `sim(query, d)` comes directly from Pinecone's cosine score (Stage 1).
    `overlap(r, d)` is bigram Jaccard — zero extra API calls.
    """
    if len(candidates) <= top_k:
        logger.debug(f"MMR: pool size {len(candidates)} ≤ top_k {top_k}, returning as-is")
        return candidates

    selected: list[dict] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        best_idx = -1
        best_mmr = -math.inf

        selected_texts = [s["metadata"].get("text", "") for s in selected]

        for i, cand in enumerate(remaining):
            relevance = cand["score"]   # cosine similarity from Pinecone
            redundancy = (
                _text_overlap_score(cand["metadata"].get("text", ""), selected_texts)
                if selected else 0.0
            )
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = i

        chosen = remaining.pop(best_idx)
        chosen["mmr_score"] = round(best_mmr, 6)   # attach for debugging
        selected.append(chosen)

    return selected


# ── Retriever class ───────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Three-stage retriever:
      Pinecone similarity search + metadata filter (Top 20)
          → MMR Re-ranking
          → Return Top 5–8

    Args:
        top_k:      Final chunks to return (default 8, recommended 5–8).
        mmr_lambda: Relevance/diversity trade-off (default 0.6).
    """

    def __init__(
        self,
        top_k: int = DEFAULT_TOP_K,
        mmr_lambda: float = MMR_LAMBDA,
    ):
        self._embedder = get_embedder()
        self._store = get_store()
        self.top_k = top_k
        self.mmr_lambda = mmr_lambda
        logger.info(
            f"RAGRetriever ready — "
            f"candidates={SIMILARITY_CANDIDATES}, top_k={top_k}, λ={mmr_lambda}"
        )

    # ── Main entry point ──────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        chapter: Optional[str] = None,
        content_types: Optional[list[str]] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        """
        Run the full four-stage pipeline.

        Args:
            query:         Query string (topic, chapter name, or question stem).
            chapter:       Filter to this chapter after similarity search.
            content_types: Filter to these content_type values after search.
                           Defaults to ["content"] (prose only).
            top_k:         Final chunks to return. Overrides instance default.

        Returns:
            List of result dicts (sorted by MMR score), each with:
                id, score, mmr_score, metadata (incl. full text)
        """
        final_k = top_k if top_k is not None else self.top_k
        if content_types is None:
            content_types = ["content"]

        # ── Stage 1: Pinecone similarity search + metadata filter (Top 20) ─────
        query_vec = self._embedder.embed_query(query)
        pinecone_filter = _build_pinecone_filter(chapter, content_types)
        candidates = self._store.query(
            vector=query_vec,
            top_k=SIMILARITY_CANDIDATES,
            filter=pinecone_filter,   # scoped at index level
            namespace=NAMESPACE,
        )
        logger.debug(
            f"Stage1: {len(candidates)} candidates from Pinecone "
            f"(filter={pinecone_filter})"
        )

        if not candidates:
            logger.warning(
                f"Pinecone returned 0 results. "
                f"Query={query[:80]!r} | filter={pinecone_filter}"
            )
            return []

        # ── Stage 2: MMR Re-ranking ───────────────────────────────────────────
        reranked = _mmr_rerank(candidates, final_k, self.mmr_lambda)
        logger.debug(f"Stage2: MMR selected {len(reranked)} from {len(candidates)} candidates")

        # ── Stage 3: Return Top 5–8 ──────────────────────────────────────────
        logger.info(
            f"Retrieved {len(reranked)} chunks for query={query[:60]!r} | "
            f"chapter={chapter!r} | scores: "
            + ", ".join(f"{r['score']:.3f}" for r in reranked)
        )
        return reranked

    # ── Chapter-scoped convenience method ────────────────────────────────────

    def retrieve_for_chapter(
        self,
        chapter_name: str,
        query: Optional[str] = None,
        top_k: Optional[int] = None,
        content_types: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Retrieve chunks scoped to a chapter (used by question-generation pipeline).

        Args:
            chapter_name:  Exact chapter string as stored in Pinecone metadata.
            query:         Optional query. Defaults to chapter_name.
            top_k:         Final chunks (5–8). Defaults to instance top_k.
            content_types: Defaults to ["content"].

        Returns:
            MMR-ranked list of chunk dicts.
        """
        return self.retrieve(
            query=query or chapter_name,
            chapter=chapter_name,
            content_types=content_types,
            top_k=top_k,
        )


# ── Module-level singleton ────────────────────────────────────────────────────

_retriever: Optional[RAGRetriever] = None


def get_retriever(
    top_k: int = DEFAULT_TOP_K,
    mmr_lambda: float = MMR_LAMBDA,
) -> RAGRetriever:
    """Return (or create) the module-level RAGRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever(top_k=top_k, mmr_lambda=mmr_lambda)
    return _retriever


def retrieve_for_chapter(
    chapter_name: str,
    query: Optional[str] = None,
    top_k: int = DEFAULT_TOP_K,
    content_types: Optional[list[str]] = None,
) -> list[dict]:
    """
    Module-level one-liner for the question-generation pipeline.

    Each returned dict's metadata["text"] is the passage to inject
    into the generation prompt.

    Example:
        chunks = retrieve_for_chapter(
            chapter_name="Introduction to Computer",
            top_k=6,
        )
        for c in chunks:
            print(c["mmr_score"], c["metadata"]["section"])
            print(c["metadata"]["text"][:200])
    """
    return get_retriever(top_k=top_k).retrieve_for_chapter(
        chapter_name=chapter_name,
        query=query,
        top_k=top_k,
        content_types=content_types,
    )
