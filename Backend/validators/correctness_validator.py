"""
Correctness Validator  (optimised)
------------------------------------
Optimisation 1 — Normalised answer comparison
  Strip "A) B) C) D)" prefixes, punctuation, extra spaces, lowercase
  before comparing. Eliminates the majority of false failures.

Optimisation 2 — Batch validation
  validate_correctness_batch() sends N questions in ONE Bedrock call
  instead of one call per question. Use this from the pipeline.

Optimisation 3 — Single-LLM combined check
  The batch prompt returns both judge verdict AND independent answer
  in one shot, halving the number of calls vs the original two-step approach.
"""

import json
import re
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from langchain_core.messages import AIMessage

from models import Question, ValidationResult, IndependentSolveResult
from prompt_builder import validation_prompt, independent_solve_prompt, batch_validation_prompt
from llm_client import validator_llm

logger = logging.getLogger(__name__)

# ── Answer normalisation ──────────────────────────────────────────────────────

_OPTION_PREFIX = re.compile(r"^[A-Da-d][).\]]\s*")
_PUNCTUATION   = re.compile(r"[^\w\s]")
_SPACES        = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """
    Canonical form for answer comparison:
      "B) Keyboard."  →  "keyboard"
      "A)  Central Processing Unit"  →  "central processing unit"
    """
    t = text.strip()
    t = _OPTION_PREFIX.sub("", t)          # remove A) B) C) D) prefixes
    t = _PUNCTUATION.sub(" ", t)           # remove punctuation
    t = _SPACES.sub(" ", t).strip()        # collapse spaces
    return t.lower()


def _answers_match(a: str, b: str) -> bool:
    return _normalise(a) == _normalise(b)


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _extract_json_object(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`")
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None


def _extract_json_array(text: str) -> Optional[list]:
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`")
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return None
    try:
        return json.loads(text[start: end + 1])
    except json.JSONDecodeError:
        return None


# ── Single-question validation (kept for regeneration retries) ────────────────

def _judge_correctness(question: Question) -> tuple[bool, str, Optional[str]]:
    messages = validation_prompt.format_messages(
        question=question.question,
        option_a=question.options[0],
        option_b=question.options[1],
        option_c=question.options[2],
        option_d=question.options[3],
        correct_answer=question.correct_answer,
        chapter_name=question.chapter,
    )
    try:
        response: AIMessage = validator_llm.invoke(messages)
        raw = _extract_json_object(response.content)
        if raw is None:
            return False, "Judge returned non-JSON; treated as failed.", None
        result = ValidationResult(**raw)
        suggested = result.suggested_answer if not result.is_correct else None
        # Normalise suggested answer
        if suggested and not any(_answers_match(suggested, o) for o in question.options):
            suggested = None
        return result.is_correct, result.reasoning, suggested
    except Exception as e:
        logger.warning(f"Judge error: {e}")
        return False, f"Judge exception: {e}", None


def _solve_independently(question: Question) -> tuple[bool, str]:
    messages = independent_solve_prompt.format_messages(
        question=question.question,
        option_a=question.options[0],
        option_b=question.options[1],
        option_c=question.options[2],
        option_d=question.options[3],
    )
    try:
        response: AIMessage = validator_llm.invoke(messages)
        raw = _extract_json_object(response.content)
        if raw is None:
            return False, "Independent solve returned non-JSON; treated as failed."
        result = IndependentSolveResult(**raw)
        # ← NORMALISED comparison (Optimisation 1)
        if _answers_match(result.answer, question.correct_answer):
            return True, "Independent solve agrees."
        # Check if it matches any option after normalisation
        for opt in question.options:
            if _answers_match(result.answer, opt) and _answers_match(opt, question.correct_answer):
                return True, "Independent solve agrees (after normalisation)."
        return (
            False,
            f"Independent solve got '{result.answer}', marked answer is '{question.correct_answer}'",
        )
    except Exception as e:
        logger.warning(f"Solve error: {e}")
        return False, f"Independent solve exception: {e}"


def validate_correctness(question: Question) -> tuple[bool, str, Optional[str]]:
    """Single-question correctness check. Used for regeneration retries."""
    judge_ok, judge_reason, suggested = _judge_correctness(question)
    if not judge_ok:
        # Try normalised match before rejecting
        if suggested and any(_answers_match(suggested, o) for o in question.options):
            matched = next(o for o in question.options if _answers_match(suggested, o))
            return False, f"Layer1a (judge): {judge_reason}", matched
        return False, f"Layer1a (judge): {judge_reason}", suggested

    solve_ok, solve_reason = _solve_independently(question)
    if not solve_ok:
        return False, f"Layer1b (solve): {solve_reason}", None

    return True, "Correctness confirmed.", None


# ── Batch validation (Optimisation 2 + 3) ────────────────────────────────────

def validate_correctness_batch(
    questions: list[Question],
    max_batch: int = 15,
) -> list[tuple[bool, str, Optional[str]]]:
    """
    Validate up to `max_batch` questions in ONE Bedrock call.

    Returns a parallel list of (passed, reason, suggested_answer) tuples,
    one per input question. Falls back to per-question validation if the
    batch call fails or returns mismatched results.

    Args:
        questions:   List of Question objects to validate.
        max_batch:   Max questions per Bedrock call (keep ≤ 15 for reliability).
    """
    if not questions:
        return []

    results: list[tuple[bool, str, Optional[str]]] = []

    # Process in sub-batches
    for i in range(0, len(questions), max_batch):
        batch = questions[i: i + max_batch]
        batch_results = _validate_batch_chunk(batch)
        results.extend(batch_results)

    return results


def _validate_batch_chunk(
    questions: list[Question],
) -> list[tuple[bool, str, Optional[str]]]:
    """Send one batch to Bedrock and parse the response array."""
    # Build numbered question list for the prompt
    q_lines = []
    for idx, q in enumerate(questions, 1):
        q_lines.append(
            f'{idx}. Q: {q.question}\n'
            f'   Options: A){q.options[0]} B){q.options[1]} C){q.options[2]} D){q.options[3]}\n'
            f'   Marked answer: {q.correct_answer}'
        )
    questions_text = "\n\n".join(q_lines)

    messages = batch_validation_prompt.format_messages(
        questions_text=questions_text,
        num_questions=len(questions),
    )

    try:
        response: AIMessage = validator_llm.invoke(messages)
        parsed = _extract_json_array(response.content)

        if parsed is None or len(parsed) != len(questions):
            logger.warning(
                f"Batch validation returned {len(parsed) if parsed else 'None'} results "
                f"for {len(questions)} questions. Falling back to individual calls."
            )
            return _validate_individually(questions)

        out = []
        for q, item in zip(questions, parsed):
            is_correct = bool(item.get("is_correct", False))
            reason = item.get("reasoning", "")
            suggested = item.get("suggested_answer", "") or None

            # Normalise suggested answer against actual options
            if suggested:
                matched = next(
                    (o for o in q.options if _answers_match(suggested, o)), None
                )
                suggested = matched  # None if no option matches

            # Independent solve check using normalised comparison
            solved = item.get("independent_answer", "") or ""
            if is_correct and solved:
                if not _answers_match(solved, q.correct_answer):
                    # Check if it matches any option that also matches correct_answer
                    solve_ok = any(
                        _answers_match(solved, o) and _answers_match(o, q.correct_answer)
                        for o in q.options
                    )
                    if not solve_ok:
                        is_correct = False
                        reason = (
                            f"Judge passed but independent solve got '{solved}' "
                            f"vs marked '{q.correct_answer}'"
                        )

            out.append((is_correct, reason, suggested if not is_correct else None))
        return out

    except Exception as e:
        logger.warning(f"Batch validation failed ({e}). Falling back to individual calls.")
        return _validate_individually(questions)


def _validate_individually(
    questions: list[Question],
) -> list[tuple[bool, str, Optional[str]]]:
    """Fallback: validate each question individually (original behaviour)."""
    return [validate_correctness(q) for q in questions]
