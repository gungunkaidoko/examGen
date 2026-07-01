"""
Duplicate Validator
-------------------
Three-tier deduplication in order of cost (cheapest first):

  Tier 1 — SHA-256 hash exact match  (O(1), zero string ops)
    - Within current set: set of hashes from accepted questions
    - Across all sets:    global_seen_hashes: set[str]

  Tier 2 — Lowercase text exact match  (O(n), kept for backward compat)
    - global_seen: set[str] of lowercased question texts

  Tier 3 — Fuzzy match  (SequenceMatcher, configurable threshold)
    - Within current set
    - Across all sets: global_seen_list: list[str]

  (Tier 4 — Embedding-based semantic dedup is a future extension;
   add it here without touching any other module.)
"""

import hashlib
from difflib import SequenceMatcher
from models import Question

FUZZY_THRESHOLD = 0.85


def compute_hash(question_text: str) -> str:
    """SHA-256 of the normalised (lowercased, stripped) question text."""
    return hashlib.sha256(question_text.strip().lower().encode("utf-8")).hexdigest()


def is_near_duplicate(
    q_text: str,
    existing_texts: list[str],
    threshold: float = FUZZY_THRESHOLD,
) -> tuple[bool, str]:
    """
    Fuzzy string similarity check using SequenceMatcher.

    Returns:
        (is_duplicate, matched_snippet)  — snippet is '' when not a duplicate.
    """
    q_norm = q_text.strip().lower()
    for existing in existing_texts:
        ratio = SequenceMatcher(None, q_norm, existing.strip().lower()).ratio()
        if ratio >= threshold:
            return True, existing[:80]
    return False, ""


def check_duplicates(
    question: Question,
    existing_questions: list[Question],
    global_seen: set[str] | None = None,
    global_seen_list: list[str] | None = None,
    global_seen_hashes: set[str] | None = None,
) -> tuple[bool, str]:
    """
    Run all deduplication tiers against a candidate question.

    Args:
        question:             Candidate question being validated.
        existing_questions:   Questions already accepted in the current set.
        global_seen:          Lowercased question texts from all sets (exact text lookup).
        global_seen_list:     Lowercased question texts from all sets (fuzzy lookup).
        global_seen_hashes:   SHA-256 hashes from all sets (fastest exact lookup).

    Returns:
        (passed, reason)
    """
    q_hash = question.question_hash or compute_hash(question.question)
    q_lower = question.question.strip().lower()

    # ── Tier 1: Hash exact match (fastest) ───────────────────────────────────

    # 1a — hash within current set
    for existing in existing_questions:
        existing_hash = existing.question_hash or compute_hash(existing.question)
        if existing_hash == q_hash:
            return False, "Exact duplicate (hash match) within current exam set"

    # 1b — hash across all sets
    if global_seen_hashes is not None and q_hash in global_seen_hashes:
        return False, "Exact duplicate (hash match) across exam sets"

    # ── Tier 2: Lowercase text exact match (backward compat) ─────────────────
    if global_seen is not None and q_lower in global_seen:
        return False, "Exact duplicate (text match) across exam sets"

    # ── Tier 3: Fuzzy near-duplicate ─────────────────────────────────────────

    # 3a — fuzzy within current set
    within_texts = [q.question for q in existing_questions]
    fuzzy_hit, matched = is_near_duplicate(question.question, within_texts)
    if fuzzy_hit:
        return False, f"Near-duplicate within set (≥{FUZZY_THRESHOLD:.0%} similar): '{matched}'"

    # 3b — fuzzy across all sets
    if global_seen_list:
        fuzzy_hit_global, matched_global = is_near_duplicate(question.question, global_seen_list)
        if fuzzy_hit_global:
            return False, f"Near-duplicate across sets (≥{FUZZY_THRESHOLD:.0%} similar): '{matched_global}'"

    return True, "No duplicates detected"
