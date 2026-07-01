"""
Question Generator Node
-----------------------
Calls the LLM with a RAG-augmented generation prompt and parses the response
into validated Pydantic Question objects.

RAG integration
---------------
Each call to generate_questions_for_chapter() receives rag_passages — a list
of chunk dicts retrieved from Pinecone by the pipeline. These are formatted
into the prompt as "Book Passages" so the LLM grounds answers in the actual
CCC textbook content rather than relying solely on parametric knowledge.

Cross-set diversity
-------------------
The retrieval query is seeded with the exam set number so different sets
receive slightly different Pinecone candidates, driving topical variation
even within the same chapter.
"""

import json
import re
import random
import logging
from typing import Optional

from langchain_core.messages import AIMessage

from config import CHAPTER_BLUEPRINT, QUESTION_FORMAT_DISTRIBUTION
from content_extractor import ChapterContent
from models import Question, Difficulty
from prompt_builder import generation_prompt, regeneration_prompt
from llm_client import generator_llm

logger = logging.getLogger(__name__)

REQUIRED_KEYS = {"question", "options", "correct_answer", "difficulty", "chapter"}
_NO_RAG_CONTEXT = "(No additional book passages available — use topic notes only.)"

# Valid question type values
VALID_QUESTION_TYPES = {
    "standard", "assertion_reason", "statement_based",
    "scenario_based", "fill_blank", "match_following", "sequence_order",
}


def _validate_schema(raw_q: dict) -> tuple[bool, str]:
    """Pre-validate raw LLM dict before handing to Pydantic."""
    missing = REQUIRED_KEYS - raw_q.keys()
    if missing:
        return False, f"Missing keys: {missing}"
    if not isinstance(raw_q.get("options"), list) or len(raw_q["options"]) != 4:
        return False, f"options must be a list of exactly 4 items, got: {raw_q.get('options')}"
    if raw_q.get("difficulty") not in ("easy", "medium", "hard"):
        return False, f"Invalid difficulty: '{raw_q.get('difficulty')}'"
    if not isinstance(raw_q.get("question"), str) or len(raw_q["question"].strip()) < 10:
        return False, "question must be a non-empty string (min 10 chars)"
    if not isinstance(raw_q.get("correct_answer"), str) or not raw_q["correct_answer"].strip():
        return False, "correct_answer must be a non-empty string"
    # Normalise question_type — default to "standard" if missing or invalid
    qt = raw_q.get("question_type", "standard")
    if qt not in VALID_QUESTION_TYPES:
        raw_q["question_type"] = "standard"
    return True, ""


def _compute_format_distribution(num_questions: int) -> dict[str, int]:
    """
    Convert QUESTION_FORMAT_DISTRIBUTION percentages into exact counts
    that sum to num_questions.
    """
    dist = QUESTION_FORMAT_DISTRIBUTION
    counts: dict[str, int] = {}
    total = 0
    for fmt, pct in dist.items():
        counts[fmt] = max(0, round(pct * num_questions))
        total += counts[fmt]

    # Fix rounding drift — add/remove from "standard" (largest bucket)
    drift = num_questions - total
    counts["standard"] = max(0, counts["standard"] + drift)
    return counts


def _format_distribution_string(num_questions: int) -> str:
    """Human-readable format distribution for prompt injection."""
    counts = _compute_format_distribution(num_questions)
    parts = [f"{cnt} {fmt.replace('_', '-')}" for fmt, cnt in counts.items() if cnt > 0]
    return ", ".join(parts)


def _compute_difficulty_distribution(chapter_key: str, num_questions: int) -> dict[str, int]:
    """Convert percentage splits into exact question counts that sum to num_questions."""
    splits = CHAPTER_BLUEPRINT[chapter_key]["difficulty_split"]
    counts: dict[str, int] = {}
    total = 0
    for diff, pct in splits.items():
        counts[diff] = round(pct * num_questions)
        total += counts[diff]
    diff = num_questions - total
    if diff != 0:
        counts["medium"] += diff
    return counts


def _difficulty_distribution_string(chapter_key: str, num_questions: int) -> str:
    """Human-readable string for prompt injection."""
    counts = _compute_difficulty_distribution(chapter_key, num_questions)
    return f"{counts['easy']} easy, {counts['medium']} medium, {counts['hard']} hard"


def _extract_json_from_response(text: str) -> list[dict]:
    """Robustly extract a JSON array from LLM output."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON array found in LLM response. Response snippet: {text[:300]}")
    return json.loads(text[start: end + 1])


def _extract_single_json(text: str) -> dict:
    """Extract a single JSON object from LLM output."""
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = text.strip("`").strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found. Snippet: {text[:300]}")
    return json.loads(text[start: end + 1])


def _format_rag_passages(rag_passages: list[dict]) -> str:
    """
    Format Pinecone retrieval results into a numbered list of book passages
    for injection into the generation prompt.

    Each passage includes section/subsection breadcrumbs so the LLM knows
    exactly where in the chapter the content comes from.
    """
    if not rag_passages:
        return _NO_RAG_CONTEXT

    lines = []
    for i, chunk in enumerate(rag_passages, 1):
        meta = chunk.get("metadata", {})
        section = meta.get("section", "")
        subsection = meta.get("subsection", "")
        text = meta.get("text", "").strip()

        breadcrumb = " › ".join(filter(None, [section, subsection]))
        header = f"[Passage {i}]" + (f" ({breadcrumb})" if breadcrumb else "")
        lines.append(f"{header}\n{text}")

    return "\n\n".join(lines)


def generate_questions_for_chapter(
    chapter_key: str,
    chapter_content: ChapterContent,
    num_questions: int,
    rag_passages: Optional[list[dict]] = None,
) -> list[Question]:
    """
    Generate `num_questions` MCQs for a chapter using Claude via Bedrock.
    Injects RAG passages + question format distribution into the prompt.
    """
    diff_str   = _difficulty_distribution_string(chapter_key, num_questions)
    fmt_str    = _format_distribution_string(num_questions)
    rag_context = _format_rag_passages(rag_passages or [])

    messages = generation_prompt.format_messages(
        chapter_name=chapter_content.chapter_name,
        chapter_summary=chapter_content.summary,
        topics_content=chapter_content.format_topics_for_prompt(),
        rag_context=rag_context,
        num_questions=num_questions,
        difficulty_distribution=diff_str,
        format_distribution=fmt_str,
    )

    logger.info(
        f"Generating {num_questions} questions for: {chapter_content.chapter_name} "
        f"| RAG: {len(rag_passages or [])} passages | formats: {fmt_str}"
    )

    response: AIMessage = generator_llm.invoke(messages)
    raw_text: str = response.content
    raw_questions = _extract_json_from_response(raw_text)

    questions: list[Question] = []
    for raw_q in raw_questions:
        try:
            raw_q["chapter"] = chapter_content.chapter_name
            # Default question_type to "standard" if LLM omitted it
            if "question_type" not in raw_q:
                raw_q["question_type"] = "standard"
            schema_ok, schema_reason = _validate_schema(raw_q)
            if not schema_ok:
                logger.warning(f"Schema validation failed: {schema_reason} | Raw: {raw_q}")
                continue
            q = Question(**raw_q)
            questions.append(q)
        except Exception as e:
            logger.warning(f"Skipping malformed question: {e} | Raw: {raw_q}")

    logger.info(f"  ✓ Parsed {len(questions)}/{num_questions} questions")
    return questions


def regenerate_single_question(
    chapter_key: str,
    chapter_content: ChapterContent,
    difficulty: str,
    rejection_reason: str,
    topic_override: Optional[dict] = None,
    rag_passages: Optional[list[dict]] = None,
    question_type: str = "standard",
) -> Optional[Question]:
    """
    Regenerate a single replacement question after a validation failure.
    Picks a random topic from the chapter if no override is given.
    Uses a RAG-grounded passage as topic_notes when available.
    Preserves the original question_type for format consistency.
    """
    topic = topic_override or random.choice(chapter_content.topics)

    topic_notes = topic["notes"]
    if rag_passages:
        best = rag_passages[0]
        rag_text = best.get("metadata", {}).get("text", "").strip()
        if rag_text:
            topic_notes = f"{topic['notes']}\n\nBook passage:\n{rag_text[:600]}"

    # Validate question_type
    qt = question_type if question_type in VALID_QUESTION_TYPES else "standard"

    messages = regeneration_prompt.format_messages(
        chapter_name=chapter_content.chapter_name,
        topic_name=topic["topic"],
        topic_notes=topic_notes,
        difficulty=difficulty,
        question_type=qt,
        rejection_reason=rejection_reason,
    )

    try:
        response: AIMessage = generator_llm.invoke(messages)
        raw = _extract_single_json(response.content)
        raw["chapter"] = chapter_content.chapter_name
        raw["validation_status"] = "approved"
        if "question_type" not in raw or raw["question_type"] not in VALID_QUESTION_TYPES:
            raw["question_type"] = qt
        return Question(**raw)
    except Exception as e:
        logger.error(f"Regeneration failed: {e}")
        return None
