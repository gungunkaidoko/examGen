"""
Distractor Validator
--------------------
Validates the quality of MCQ distractors (wrong answer options) using two layers:

Layer 1 — Semantic Scoring (Bedrock Titan Embeddings)
  Computes cosine similarity between each distractor and the correct answer.
  Good distractors are semantically CLOSE (same domain) but not identical.
  - Too far (sim < DISTRACTOR_MIN_SIM): distractor is from a different domain → reject
  - Too close (sim > DISTRACTOR_MAX_SIM): distractor is nearly the correct answer → reject
  Falls back gracefully if Bedrock is unavailable.

Layer 2 — LLM Plausibility Audit (Claude, batched)
  Checks each question for:
  - Distractors that are obviously absurd or unrelated
  - Distractors that differ in grammatical form from the correct answer
  - One distractor being trivially eliminatable
  - Correct answer being uniquely long/detailed compared to distractors
  Returns pass/fail per question with a reason.

Public API
----------
  validate_distractors(questions) → list[tuple[bool, str]]
    Returns (passed, reason) per question. passed=False → regenerate.

  score_distractors_embedding(question) → dict
    Returns per-distractor similarity scores for debugging/logging.
"""

import json
import logging
import re
from typing import Optional

from models import Question

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────
DISTRACTOR_MIN_SIM   = 0.30   # below this → distractor is from a different domain
DISTRACTOR_MAX_SIM   = 0.97   # above this → distractor is almost the correct answer
DISTRACTOR_BATCH_SIZE = 15    # questions per LLM distractor audit call

# Question types with fixed options — skip distractor scoring for these
_FIXED_OPTION_TYPES = {"assertion_reason", "sequence_order"}


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — Embedding-based semantic scoring
# ─────────────────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    """Dot product — valid for L2-normalised Titan vectors."""
    return sum(x * y for x, y in zip(a, b))


def score_distractors_embedding(question: Question) -> dict:
    """
    Embed the correct answer and each distractor, return similarity scores.

    Returns:
        {
          "correct": <correct answer text>,
          "scores": [{"distractor": <text>, "sim": <float>}, ...]
        }
    Raises RuntimeError if embedding fails (caller should catch).
    """
    from rag.embedder import get_embedder
    embedder = get_embedder()

    distractors = [o for o in question.options if o != question.correct_answer]
    texts = [question.correct_answer] + distractors
    vectors = embedder.embed_texts(texts)

    correct_vec = vectors[0]
    scores = []
    for i, d in enumerate(distractors):
        sim = _cosine(correct_vec, vectors[i + 1])
        scores.append({"distractor": d, "sim": round(sim, 4)})

    return {"correct": question.correct_answer, "scores": scores}


def _embedding_distractor_check(question: Question) -> tuple[bool, str]:
    """
    Check a single question's distractors using Titan embeddings.

    Returns (passed, reason). Fails open if Bedrock unavailable.
    """
    if question.question_type in _FIXED_OPTION_TYPES:
        return True, ""

    try:
        result = score_distractors_embedding(question)
        for item in result["scores"]:
            sim = item["sim"]
            d   = item["distractor"]
            if sim < DISTRACTOR_MIN_SIM:
                return False, (
                    f"Distractor '{d[:60]}' is semantically distant from the correct answer "
                    f"(sim={sim:.3f} < {DISTRACTOR_MIN_SIM}). Use a same-domain term."
                )
            if sim > DISTRACTOR_MAX_SIM:
                return False, (
                    f"Distractor '{d[:60]}' is almost identical to the correct answer "
                    f"(sim={sim:.3f} > {DISTRACTOR_MAX_SIM}). Use a more distinct wrong option."
                )
    except Exception as e:
        logger.debug(f"Embedding distractor check skipped: {e}")
        return True, ""  # fail open

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — LLM plausibility audit
# ─────────────────────────────────────────────────────────────────────────────

_DISTRACTOR_AUDIT_SYSTEM = """You are an expert MCQ quality auditor for the NIELIT CCC exam.
For each question, evaluate ONLY the distractor (wrong option) quality.

Check for:
1. DOMAIN — Are all 3 wrong options from the same conceptual domain as the correct answer?
2. PLAUSIBILITY — Could a partially-informed student genuinely confuse each wrong option
   with the correct answer? (Not just any wrong thing — a believable mistake.)
3. GRAMMAR PARITY — Do all 4 options use the same grammatical form?
   (e.g. all noun phrases, all action verbs, all complete sentences)
4. NO GIVEAWAY — Is any one option trivially eliminatable because it obviously
   does not belong? (e.g. "The Sun" in a question about file systems)
5. NO LENGTH TRICK — Is the correct answer noticeably longer or more detailed than
   all the distractors? (This telegraphs the answer.)

For PASS: all 5 checks must pass.
For FAIL: state which check failed and why.

Return ONLY a JSON array of exactly {num_questions} objects:
[
  {{"q_index": 1, "passed": true, "reason": ""}},
  {{"q_index": 2, "passed": false, "reason": "Option 'The Sun' (domain mismatch) is trivially eliminatable"}}
]
Return ONLY the JSON array."""

_DISTRACTOR_AUDIT_HUMAN = """Audit the distractor quality of these {num_questions} questions:

{questions_text}

Return a JSON array of {num_questions} results."""


def _build_distractor_audit_text(questions: list[Question]) -> str:
    lines = []
    for idx, q in enumerate(questions, 1):
        distractors = [o for o in q.options if o != q.correct_answer]
        dist_text = "\n".join(f"   - {d}" for d in distractors)
        lines.append(
            f"[Q{idx}] Type: {q.question_type}\n"
            f"Question: {q.question}\n"
            f"Correct answer: {q.correct_answer}\n"
            f"Distractors (wrong options):\n{dist_text}"
        )
    return "\n\n".join(lines)


def _llm_distractor_audit(questions: list[Question]) -> list[tuple[bool, str]]:
    """
    LLM-based plausibility audit for a batch of questions.
    Fails open on any error.
    """
    fallback = [(True, "LLM distractor audit skipped")] * len(questions)

    # Skip question types with fixed options (no distractor design involved)
    effective_qs = [q for q in questions if q.question_type not in _FIXED_OPTION_TYPES]
    if not effective_qs:
        return fallback

    try:
        from llm_client import validator_llm
        from langchain_core.prompts import (
            ChatPromptTemplate,
            SystemMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )

        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(_DISTRACTOR_AUDIT_SYSTEM),
            HumanMessagePromptTemplate.from_template(_DISTRACTOR_AUDIT_HUMAN),
        ])

        # Build per-batch audit (only effective_qs)
        messages = prompt.format_messages(
            num_questions=len(effective_qs),
            questions_text=_build_distractor_audit_text(effective_qs),
        )
        response = validator_llm.invoke(messages)
        raw = re.sub(r"```(?:json)?", "", response.content, flags=re.IGNORECASE).strip().strip("`")

        start, end = raw.find("["), raw.rfind("]")
        if start == -1 or end == -1:
            logger.warning("Distractor audit: no JSON array; failing open")
            return fallback

        parsed = json.loads(raw[start: end + 1])
        if not isinstance(parsed, list) or len(parsed) != len(effective_qs):
            logger.warning(
                f"Distractor audit: expected {len(effective_qs)} results, "
                f"got {len(parsed) if parsed else 'None'}; failing open"
            )
            return fallback

        # Map back to original questions list (fixed-option types get pass)
        eff_iter = iter(parsed)
        results = []
        for q in questions:
            if q.question_type in _FIXED_OPTION_TYPES:
                results.append((True, ""))
            else:
                item = next(eff_iter)
                results.append((bool(item.get("passed", True)), item.get("reason", "")))
        return results

    except ImportError:
        logger.debug("llm_client unavailable for distractor audit")
        return fallback
    except Exception as e:
        logger.warning(f"Distractor audit LLM failed: {e}; failing open")
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def validate_distractors(
    questions: list[Question],
    run_llm: bool = True,
) -> list[tuple[bool, str]]:
    """
    Full distractor validation: embedding scoring + LLM plausibility audit.

    Args:
        questions:  List of questions to validate.
        run_llm:    Whether to run the LLM plausibility audit (set False in tests).

    Returns:
        list of (passed: bool, reason: str) — one per input question.
        passed=False → question should be regenerated with better distractors.
    """
    n = len(questions)
    if n == 0:
        return []

    results: list[tuple[bool, str]] = [(True, "")] * n

    # ── Layer 1: Embedding semantic scoring ──────────────────────────────────
    for i, q in enumerate(questions):
        ok, reason = _embedding_distractor_check(q)
        if not ok:
            results[i] = (False, f"Distractor-Embedding: {reason}")
            logger.info(f"    ✗ Distractor-Emb Q{i+1}: {reason}")

    # ── Layer 2: LLM plausibility audit (batched, only on passing Qs) ────────
    if run_llm:
        live_idx = [i for i in range(n) if results[i][0]]
        live_qs  = [questions[i] for i in live_idx]

        for batch_start in range(0, len(live_qs), DISTRACTOR_BATCH_SIZE):
            batch = live_qs[batch_start: batch_start + DISTRACTOR_BATCH_SIZE]
            batch_results = _llm_distractor_audit(batch)
            for j, (ok, reason) in enumerate(batch_results):
                orig_i = live_idx[batch_start + j]
                if not ok:
                    results[orig_i] = (False, f"Distractor-LLM: {reason}")
                    logger.info(f"    ✗ Distractor-LLM Q{orig_i+1}: {reason}")

    pass_count = sum(1 for ok, _ in results if ok)
    fail_count = n - pass_count
    if fail_count:
        logger.info(f"  Distractor validation: {pass_count}/{n} passed, {fail_count} flagged")
    return results
