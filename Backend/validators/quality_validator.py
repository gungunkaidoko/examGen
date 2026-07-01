"""
Quality Validator
-----------------
Independent post-generation validation layer that checks a batch of questions
for issues that individual per-question nodes cannot catch holistically:

  1. Semantic near-duplicates  — Bedrock Titan embeddings + cosine similarity.
  2. Distractor quality        — embedding semantic scoring + LLM plausibility audit.
  3. Grammar / punctuation     — LLM-based batched audit (Claude, temp=0).
  4. Option length consistency — deterministic.
  5. Match-the-Following format — auto-normalise then validate layout.
  6. Balanced answer distribution — ENFORCED: skewed questions flagged + regenerated.
  7. Assertion–Reason A-bias   — correct answer must not be repeatedly option A.

Any question that fails is flagged with a rejection reason.  The caller
(pipeline.py) is responsible for regenerating flagged questions.

Public API
----------
  validate_quality_batch(questions, chapter_key, exam_set)
      → list[tuple[bool, str]]   — (passed, reason) per question
  normalize_match_questions(questions) → list[Question]
      → returns questions with match_following stems normalised in-place
"""

import json
import logging
import math
import re
from typing import Optional

from models import Question
from validators.formatting_validator import (
    check_answer_distribution,
    _check_option_length_balance,
    _check_statement_format,
    _check_match_format,
    _ends_with_valid_punct,
)
from validators.distractor_validator import validate_distractors
from validators.match_formatter import normalize_match_question, validate_match_format
from validators.answer_leak_validator import validate_answer_leaks

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
QUALITY_BATCH_SIZE    = 20     # questions per LLM quality-check call
BEDROCK_SIM_THRESHOLD = 0.88   # cosine sim above this → near-duplicate (within batch)
MAX_POSITION_PCT      = 0.45   # soft limit: warn if any position exceeds 45%
SKEW_THRESHOLD        = 0.55   # hard limit: regenerate if any position exceeds 55%
MAX_AR_A_PCT          = 0.40   # max fraction of A/R questions where option A is correct


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bedrock embedding-based semantic dedup
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python dot product (vectors are already L2-normalised by Titan)."""
    return sum(x * y for x, y in zip(a, b))


def _bedrock_semantic_duplicates(
    questions: list[Question],
) -> list[tuple[bool, str]]:
    """
    Detect near-duplicate pairs within `questions` using Bedrock Titan embeddings.

    Steps:
      1. Embed all question texts in one batch (one Bedrock call per question,
         using the existing TitanEmbedder which handles throttle retries).
      2. Compute pairwise cosine similarity in-memory.
      3. First occurrence of any near-duplicate pair always passes; the later
         occurrence is flagged for regeneration.

    Falls back to (True, "") for all questions if Bedrock is unavailable.
    """
    n = len(questions)
    results: list[tuple[bool, str]] = [(True, "")] * n
    if n < 2:
        return results

    # ── Embed ────────────────────────────────────────────────────────────────
    try:
        from rag.embedder import get_embedder
        embedder = get_embedder()
        texts = [q.question.strip() for q in questions]
        vectors = embedder.embed_texts(texts)
        logger.debug(f"    Embedded {n} questions for semantic dedup")
    except Exception as e:
        logger.warning(f"  Bedrock embedding unavailable for semantic dedup: {e}. Skipping.")
        return results

    # ── Pairwise cosine similarity ────────────────────────────────────────────
    for i in range(n):
        if not results[i][0]:
            continue  # already flagged
        for j in range(i + 1, n):
            if not results[j][0]:
                continue  # already flagged
            sim = _cosine_similarity(vectors[i], vectors[j])
            if sim >= BEDROCK_SIM_THRESHOLD:
                reason = (
                    f"Near-duplicate of Q{i+1} "
                    f"(Bedrock cosine similarity {sim:.3f} ≥ {BEDROCK_SIM_THRESHOLD})"
                )
                results[j] = (False, reason)
                logger.info(
                    f"    ✗ Semantic-Dedup: Q{j+1} ~ Q{i+1} "
                    f"(sim={sim:.3f}) → flagged for regeneration"
                )

    dupes = sum(1 for ok, _ in results if not ok)
    if dupes:
        logger.info(f"    Bedrock dedup: {dupes}/{n} questions flagged as near-duplicates")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2. LLM-based quality audit (grammar, punctuation, distractor quality)
# ─────────────────────────────────────────────────────────────────────────────

QUALITY_SYSTEM_PROMPT = """You are a senior exam quality auditor for the NIELIT CCC exam paper.
Review each MCQ below and flag any that fail one or more of these quality criteria:

CRITERIA:
1. GRAMMAR — Question and all options must be grammatically correct English.
2. PUNCTUATION — Question must end with '?', '.', or ':'. Full stops on sentence options.
3. DISTRACTOR QUALITY — Wrong options must be plausible to a student who partially knows the topic.
   Reject if any distractor is obviously absurd, unrelated, or trivially eliminatable.
4. OPTION LENGTH — All 4 options should be roughly similar in length (within 3×).
   Do not allow the correct answer to be the uniquely long/detailed option.
5. STATEMENT FORMAT — For statement_based: each numbered statement must be on its own line.
6. MATCH FORMAT — For match_following: Column A and Column B must be clearly separated.
7. ASSERTION-REASON — For assertion_reason: the correct answer must not always be
   "Both A and R are true, and R is the correct explanation of A" (option A).
   Vary which of the four fixed options is correct.

For each question respond with PASS or FAIL and a brief reason (empty string if PASS).
Return ONLY a JSON array of exactly {num_questions} objects:
[
  {{"q_index": 1, "passed": true, "reason": ""}},
  {{"q_index": 2, "passed": false, "reason": "Distractor 'Red colour' is unrelated to networking"}},
  ...
]
Return ONLY the JSON array — no preamble, no markdown."""

QUALITY_HUMAN_PROMPT = """Review these {num_questions} MCQs:

{questions_text}

Return a JSON array of {num_questions} audit results."""


def _build_quality_questions_text(questions: list[Question]) -> str:
    lines = []
    for idx, q in enumerate(questions, 1):
        opts = "\n".join(
            f"   {chr(64+i)}) {o}" for i, o in enumerate(q.options, 1)
        )
        lines.append(
            f"[Q{idx}] Type: {q.question_type}\n"
            f"Question: {q.question}\n"
            f"Options:\n{opts}\n"
            f"Correct: {q.correct_answer}"
        )
    return "\n\n".join(lines)


def _llm_quality_check_batch(questions: list[Question]) -> list[tuple[bool, str]]:
    """
    Send a batch to Claude for grammar / punctuation / distractor audit.
    Returns (passed, reason) per question.  Fails open on error.
    """
    fallback = [(True, "LLM quality check skipped (fallback)")] * len(questions)
    try:
        from llm_client import validator_llm
        from langchain_core.prompts import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(QUALITY_SYSTEM_PROMPT),
            HumanMessagePromptTemplate.from_template(QUALITY_HUMAN_PROMPT),
        ])
        messages = prompt.format_messages(
            num_questions=len(questions),
            questions_text=_build_quality_questions_text(questions),
        )
        response = validator_llm.invoke(messages)
        raw = re.sub(r"```(?:json)?", "", response.content, flags=re.IGNORECASE).strip().strip("`")

        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            logger.warning("Quality LLM: no JSON array; failing open")
            return fallback

        parsed = json.loads(raw[start: end + 1])
        if not isinstance(parsed, list) or len(parsed) != len(questions):
            logger.warning(
                f"Quality LLM: expected {len(questions)} results, "
                f"got {len(parsed) if parsed else 'None'}; failing open"
            )
            return fallback

        return [
            (bool(item.get("passed", True)), item.get("reason", ""))
            for item in parsed
        ]

    except ImportError:
        logger.debug("llm_client not available for quality check")
        return fallback
    except Exception as e:
        logger.warning(f"Quality LLM check failed: {e}; failing open")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# 3. Deterministic per-question checks
# ─────────────────────────────────────────────────────────────────────────────

def _deterministic_checks(question: Question) -> tuple[bool, str]:
    """
    Fast Python checks: punctuation, option balance, statement/match format.
    Match_following questions are validated via match_formatter (stricter).
    """
    q_text = question.question.strip()

    # Terminal punctuation (match/sequence exempt — validated by format check)
    if question.question_type not in ("match_following", "sequence_order"):
        if not _ends_with_valid_punct(q_text):
            return False, "Missing terminal punctuation (must end with '?', '.', or ':')."

    # Option length balance
    ok, reason = _check_option_length_balance(question.options)
    if not ok:
        return False, reason

    # Statement format
    ok, reason = _check_statement_format(question)
    if not ok:
        return False, reason

    # Match format — use dedicated validator (more thorough than _check_match_format)
    ok, reason = validate_match_format(question)
    if not ok:
        return False, reason

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# 4. Balanced answer-distribution enforcement
# ─────────────────────────────────────────────────────────────────────────────

def _get_correct_position(question: Question) -> Optional[int]:
    """Return 0-indexed position (0=A, 1=B, 2=C, 3=D) of the correct answer, or None."""
    try:
        return question.options.index(question.correct_answer)
    except ValueError:
        return None


def enforce_answer_distribution(
    questions: list[Question],
) -> list[tuple[bool, str]]:
    """
    Enforce balanced correct-answer distribution across the batch.

    Strategy:
    - Walk through questions in order, tracking a running count per position.
    - When any position would exceed SKEW_THRESHOLD of questions seen so far
      AND that position is already overrepresented, flag the question.
    - This only flags questions that are *surplus* contributors to skew,
      so early-accepted questions are never retroactively penalised.

    Returns (passed, reason) per question.
    """
    n = len(questions)
    results: list[tuple[bool, str]] = [(True, "")] * n
    if n < 8:
        # Too few questions to meaningfully enforce distribution
        return results

    position_counts = [0, 0, 0, 0]
    labels = ["A", "B", "C", "D"]

    for i, q in enumerate(questions):
        pos = _get_correct_position(q)
        if pos is None:
            continue

        seen_so_far = i + 1
        # Would accepting this question push position `pos` over the hard limit?
        projected_count = position_counts[pos] + 1
        projected_pct   = projected_count / seen_so_far

        if projected_pct > SKEW_THRESHOLD and position_counts[pos] >= math.ceil(n * MAX_POSITION_PCT):
            reason = (
                f"Answer distribution skew: option {labels[pos]} would be correct "
                f"{projected_count}/{seen_so_far} times ({projected_pct:.0%} > {SKEW_THRESHOLD:.0%}). "
                f"Regenerate with a different correct answer position."
            )
            results[i] = (False, reason)
            logger.info(f"    ✗ Distribution-Enforce Q{i+1}: {reason}")
            # Don't increment counts — this question will be replaced
        else:
            position_counts[pos] += 1

    # Final summary log
    dist_ok, dist_reason = check_answer_distribution(
        [q for q, (ok, _) in zip(questions, results) if ok]
    )
    if not dist_ok:
        logger.warning(f"  ⚠ Post-enforcement distribution: {dist_reason}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 5. Assertion–Reason option-A bias check
# ─────────────────────────────────────────────────────────────────────────────

# The fixed option A text for assertion_reason questions (normalised)
_AR_OPTION_A_NORM = "both a and r are true and r is the correct explanation of a"


def _is_ar_option_a(question: Question) -> bool:
    """Return True if this A/R question has option A as the correct answer."""
    if question.question_type != "assertion_reason":
        return False
    correct_norm = question.correct_answer.strip().lower()
    # Handle minor variations: "and R is the correct explanation of A"
    return (
        correct_norm.startswith("both a and r are true")
        and "correct explanation of a" in correct_norm
    )


def check_assertion_reason_bias(
    questions: list[Question],
) -> list[tuple[bool, str]]:
    """
    Detect Assertion–Reason questions that are biased toward option A.

    Walk through A/R questions in order.  Once accepting a question would
    push option-A's share above MAX_AR_A_PCT (given ar_accepted >= 3),
    flag it for regeneration with a different correct option.

    Returns (passed, reason) for EVERY question (non-A/R questions always pass).
    """
    results: list[tuple[bool, str]] = [(True, "")] * len(questions)

    ar_indices = [i for i, q in enumerate(questions) if q.question_type == "assertion_reason"]
    if len(ar_indices) < 3:
        # Too few A/R questions to enforce bias rule
        return results

    ar_a_count = 0   # accepted A/R questions where option A was correct
    ar_accepted = 0  # total accepted A/R questions so far

    for i in ar_indices:
        q = questions[i]
        is_a = _is_ar_option_a(q)

        # Only enforce once we have at least 3 accepted A/R questions
        if is_a and ar_accepted >= 3:
            new_total     = ar_accepted + 1
            new_a_count   = ar_a_count + 1
            projected_pct = new_a_count / new_total
            if projected_pct > MAX_AR_A_PCT:
                reason = (
                    f"Assertion-Reason A-bias: option A (correct explanation) would be "
                    f"the correct answer {new_a_count}/{new_total} times "
                    f"({projected_pct:.0%} > {MAX_AR_A_PCT:.0%}). "
                    f"Regenerate with a different correct option (B, C, or D)."
                )
                results[i] = (False, reason)
                logger.info(f"    ✗ AR-Bias Q{i+1}: {reason}")
                # Do NOT count this question; it will be regenerated
                continue

        # Accept this question into the running counts
        ar_accepted += 1
        if is_a:
            ar_a_count += 1

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_match_questions(questions: list[Question]) -> list[Question]:
    """
    Apply match_formatter normalisation to all match_following questions in the list.
    Returns a new list (other question types are unchanged).
    Called BEFORE validate_quality_batch so the formatter runs first.
    """
    return [normalize_match_question(q) for q in questions]


def validate_quality_batch(
    questions: list[Question],
    chapter_key: str = "",
    exam_set: int = 0,
    run_llm_check: bool = True,
) -> list[tuple[bool, str]]:
    """
    Run the full independent quality validation on a list of questions.

    Pipeline (in order):
      Step 0 — Auto-normalise match_following stems (in-place, free)
      Step 1 — Deterministic per-question checks (fast, free)
      Step 2 — Bedrock embedding semantic dedup within the batch
      Step 3 — Distractor quality: embedding scoring + LLM audit
      Step 4 — LLM grammar / punctuation audit (batched)
      Step 5 — Balanced answer-distribution enforcement (hard limit)
      Step 6 — Assertion–Reason option-A bias check
      Step 7 — Cross-question answer leak detection

    Returns:
        list of (passed: bool, reason: str) — one per input question.
        passed=False  → caller must regenerate the question.
    """
    n = len(questions)
    if n == 0:
        return []

    # ── Step 0: Normalise match_following stems ──────────────────────────────
    questions = normalize_match_questions(questions)

    results: list[tuple[bool, str]] = [(True, "")] * n

    # ── Step 1: Deterministic ────────────────────────────────────────────────
    for i, q in enumerate(questions):
        ok, reason = _deterministic_checks(q)
        if not ok:
            results[i] = (False, f"Quality-Deterministic: {reason}")
            logger.info(f"    ✗ Quality-Det Q{i+1}: {reason}")

    # ── Step 2: Bedrock semantic dedup ───────────────────────────────────────
    dedup_results = _bedrock_semantic_duplicates(questions)
    for i, (ok, reason) in enumerate(dedup_results):
        if not ok and results[i][0]:
            results[i] = (False, f"Quality-SemanticDedup: {reason}")

    # ── Step 3: Distractor quality (embedding + LLM) ─────────────────────────
    live_for_distractor = [i for i in range(n) if results[i][0]]
    if live_for_distractor:
        dist_qs = [questions[i] for i in live_for_distractor]
        dist_results = validate_distractors(dist_qs, run_llm=run_llm_check)
        for j, (ok, reason) in enumerate(dist_results):
            orig_i = live_for_distractor[j]
            if not ok:
                results[orig_i] = (False, f"Quality-Distractor: {reason}")
                logger.info(f"    ✗ Quality-Distractor Q{orig_i+1}: {reason}")

    # ── Step 4: LLM grammar/punctuation audit (only on still-passing Qs) ─────
    if run_llm_check:
        live_idx = [i for i in range(n) if results[i][0]]
        live_qs  = [questions[i] for i in live_idx]

        for batch_start in range(0, len(live_qs), QUALITY_BATCH_SIZE):
            batch = live_qs[batch_start: batch_start + QUALITY_BATCH_SIZE]
            batch_results = _llm_quality_check_batch(batch)
            for j, (ok, reason) in enumerate(batch_results):
                orig_i = live_idx[batch_start + j]
                if not ok:
                    results[orig_i] = (False, f"Quality-LLM: {reason}")
                    logger.info(f"    ✗ Quality-LLM Q{orig_i+1}: {reason}")

    # ── Step 5: Answer distribution enforcement ──────────────────────────────
    dist_results = enforce_answer_distribution(questions)
    for i, (ok, reason) in enumerate(dist_results):
        if not ok and results[i][0]:
            results[i] = (False, f"Quality-Distribution: {reason}")

    # ── Step 6: Assertion–Reason A-bias ──────────────────────────────────────
    ar_results = check_assertion_reason_bias(questions)
    for i, (ok, reason) in enumerate(ar_results):
        if not ok and results[i][0]:
            results[i] = (False, f"Quality-AR-Bias: {reason}")

    # ── Step 7: Cross-question answer leak detection ──────────────────────────
    leak_results = validate_answer_leaks(questions)
    for i, (ok, reason) in enumerate(leak_results):
        if not ok and results[i][0]:
            results[i] = (False, f"Quality-AnswerLeak: {reason}")
            logger.info(f"    ✗ Quality-AnswerLeak Q{i+1}: {reason[:100]}")

    pass_count = sum(1 for ok, _ in results if ok)
    logger.info(
        f"  Quality validation complete: {pass_count}/{n} passed, "
        f"{n - pass_count} flagged for regeneration"
    )
    return results
