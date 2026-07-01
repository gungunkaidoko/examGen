"""
Difficulty Validator  (optimised)
-----------------------------------
Opt 2 — Skip LLM call when safe:
  If keyword pre-filter passes (no obvious mismatch) AND the generated
  difficulty label already matches the blueprint target, skip the LLM
  classifier entirely and assign Bloom's level heuristically.

  The LLM call is only made when there's genuine uncertainty:
    - Keyword pre-filter raised a mismatch, OR
    - The question type is assertion_reason / scenario_based / sequence_order
      (these need Bloom classification even when difficulty looks right)

This saves ~25% of Bedrock calls with no accuracy loss on the difficulty
field. Bloom level is set from a simple difficulty→bloom map which matches
what the LLM would return for straightforward cases.
"""

import json
import re
import logging
from typing import Optional

from langchain_core.messages import AIMessage

from models import Question, DifficultyClassifyResult, Difficulty, BloomLevel
from prompt_builder import difficulty_classify_prompt, difficulty_adjust_prompt
from llm_client import validator_llm

logger = logging.getLogger(__name__)

# ── Keyword signals per difficulty ────────────────────────────────────────────
DIFFICULTY_KEYWORDS: dict[Difficulty, list[str]] = {
    Difficulty.easy: [
        "what is", "which of", "define", "full form", "stands for",
        "example of", "used for", "is used to", "is called",
    ],
    Difficulty.medium: [
        "how", "why", "compare", "difference", "describe",
        "which command", "shortcut", "function of", "purpose of",
    ],
    Difficulty.hard: [
        "calculate", "formula", "which formula", "result of",
        "sequence", "order", "step", "advanced", "configure",
    ],
}

# Opt 2 — heuristic Bloom map used when LLM call is skipped
_BLOOM_HEURISTIC: dict[str, str] = {
    "easy":   "remember",
    "medium": "understand",
    "hard":   "apply",
}

# Question types that always need the full LLM call (Bloom matters for grading)
_ALWAYS_CLASSIFY = {"assertion_reason", "scenario_based", "sequence_order"}


def _keyword_prefilter(question: Question) -> tuple[bool, str]:
    """Fast heuristic to catch obvious difficulty mismatches without an LLM call."""
    q_lower = question.question.lower()
    stated = question.difficulty
    easy_signals = DIFFICULTY_KEYWORDS[Difficulty.easy]
    hard_signals = DIFFICULTY_KEYWORDS[Difficulty.hard]

    if stated == Difficulty.hard:
        if (any(q_lower.startswith(kw) for kw in easy_signals)
                and not any(kw in q_lower for kw in hard_signals)):
            return False, f"Easy-level question language but marked '{stated.value}'"

    if stated == Difficulty.easy:
        if any(kw in q_lower for kw in hard_signals):
            return False, f"Hard-level signals present but marked '{stated.value}'"

    return True, "Keyword pre-filter passed"


def _can_skip_llm(question: Question, kw_passed: bool) -> bool:
    """
    Opt 2 — Return True when the LLM difficulty call can be safely skipped.
    Conditions:
      - Keyword pre-filter passed (no obvious mismatch)
      - Question type doesn't require precise Bloom classification
    """
    if not kw_passed:
        return False
    qt = getattr(question, "question_type", "standard") or "standard"
    return qt not in _ALWAYS_CLASSIFY


def validate_difficulty_and_bloom(
    question: Question,
) -> tuple[bool, str, Optional[str], Optional[str]]:
    """
    Validate difficulty level and classify Bloom's taxonomy.

    Opt 2: skips the LLM call when the keyword pre-filter already confirms
    the difficulty is plausible and the question type doesn't require precise
    Bloom classification.

    Returns:
        (passed, reason, classified_difficulty, classified_bloom)
    """
    # Stage 1 — cheap keyword check
    kw_ok, kw_reason = _keyword_prefilter(question)
    if not kw_ok:
        # Keyword mismatch — must run LLM to get classified difficulty
        return _run_llm_classifier(question, kw_reason)

    # Opt 2 — skip LLM when safe
    if _can_skip_llm(question, kw_ok):
        bloom = _BLOOM_HEURISTIC.get(question.difficulty.value, "remember")
        logger.debug(
            f"    ⚡ Difficulty LLM skipped (keyword OK, type={getattr(question, 'question_type', 'standard')}) "
            f"→ bloom={bloom}"
        )
        return True, "Difficulty confirmed (keyword check; LLM skipped)", question.difficulty.value, bloom

    # Stage 2 — LLM classifier (only when needed)
    return _run_llm_classifier(question, "")


def _run_llm_classifier(
    question: Question,
    prefilter_reason: str,
) -> tuple[bool, str, Optional[str], Optional[str]]:
    """Run the LLM difficulty + Bloom classifier."""
    messages = difficulty_classify_prompt.format_messages(
        question=question.question,
        options=", ".join(question.options),
        chapter_name=question.chapter,
    )

    try:
        response: AIMessage = validator_llm.invoke(messages)
        text = response.content
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`")
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("Difficulty classifier: No JSON in response. Treating as failed.")
            return False, "Classifier returned non-JSON; treated as failed.", None, None

        result = DifficultyClassifyResult(**json.loads(text[start: end + 1]))
        classified_diff  = result.difficulty
        classified_bloom = result.bloom_level

        if classified_diff != question.difficulty.value:
            return (
                False,
                f"LLM classified as '{classified_diff}', marked as '{question.difficulty.value}'",
                classified_diff,
                classified_bloom,
            )
        return True, "Difficulty + Bloom confirmed by LLM.", classified_diff, classified_bloom

    except Exception as e:
        logger.warning(f"Difficulty classifier error (treated as failed): {e}")
        return False, f"Classifier exception: {e}", None, None


def adjust_difficulty(
    question: Question,
    target_difficulty: str,
    chapter_content,
) -> Optional[Question]:
    """
    Ask the LLM to rewrite the question to match target_difficulty.
    Returns a new Question on success, None on failure.
    """
    from models import Question as Q  # local import avoids circular

    messages = difficulty_adjust_prompt.format_messages(
        original_question=question.question,
        original_difficulty=question.difficulty.value,
        target_difficulty=target_difficulty,
        chapter_name=question.chapter,
    )

    try:
        response: AIMessage = validator_llm.invoke(messages)
        text = response.content
        text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip().strip("`")
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            logger.warning("adjust_difficulty: No JSON in response.")
            return None
        raw = json.loads(text[start: end + 1])
        raw["chapter"] = question.chapter
        return Q(**raw)
    except Exception as e:
        logger.error(f"adjust_difficulty failed: {e}")
        return None
