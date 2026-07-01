"""
Rule Engine
-----------
Combines all fast deterministic checks into a single pipeline call:

  1. Structural checks    (option count, answer-in-options, distinct options)
  2. Formatting checks    (capitalisation, '?', whitespace, option length)
  3. Duplicate detection  (exact + fuzzy, within-set + cross-set)

No LLM calls — everything here is pure Python.
"""

import logging
from models import Question
from validators.formatting_validator import validate_formatting
from validators.duplicate_validator import check_duplicates

logger = logging.getLogger(__name__)


def run_rule_engine(
    question: Question,
    existing_questions: list[Question],
    global_seen: set[str] | None = None,
    global_seen_list: list[str] | None = None,
    global_seen_hashes: set[str] | None = None,
) -> tuple[bool, str]:
    """
    Run all deterministic rule checks in order. Returns on first failure.

    Args:
        question:             Candidate question.
        existing_questions:   Already-accepted questions in the current set.
        global_seen:          Set of lowercased question texts (exact cross-set dedup).
        global_seen_list:     List of lowercased question texts (fuzzy cross-set dedup).
        global_seen_hashes:   Set of SHA-256 hashes (fastest exact cross-set dedup).

    Returns:
        (passed, reason)
    """
    # ── 1. Structural checks ──────────────────────────────────────────────────
    if len(question.options) != 4:
        return False, f"Expected 4 options, got {len(question.options)}"

    if question.correct_answer not in question.options:
        return False, f"correct_answer '{question.correct_answer}' is not in options"

    if len(set(o.strip().lower() for o in question.options)) != 4:
        return False, "Duplicate options detected"

    if len(question.question.strip()) < 10:
        return False, "Question text is too short (< 10 chars)"

    # ── 2. Formatting checks ──────────────────────────────────────────────────
    fmt_ok, fmt_reason = validate_formatting(question)
    if not fmt_ok:
        return False, f"Formatting: {fmt_reason}"

    # ── 3. Duplicate detection (hash → exact text → fuzzy) ───────────────────
    dup_ok, dup_reason = check_duplicates(
        question, existing_questions, global_seen, global_seen_list, global_seen_hashes
    )
    if not dup_ok:
        return False, dup_reason

    return True, "All rule checks passed"
