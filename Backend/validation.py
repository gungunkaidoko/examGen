"""
validation.py — Façade
----------------------
Re-exports everything from the validators/ package so existing callers
(pipeline.py, main.py, etc.) need no changes.

All logic lives in:
    validators/schema_validator.py
    validators/correctness_validator.py
    validators/formatting_validator.py
    validators/duplicate_validator.py
    validators/difficulty_validator.py
    validators/bloom_validator.py
    validators/rule_engine.py
"""

from validators.schema_validator import validate_schema
from validators.correctness_validator import validate_correctness, validate_correctness_batch
from validators.formatting_validator import validate_formatting, check_answer_distribution
from validators.duplicate_validator import check_duplicates, is_near_duplicate
from validators.difficulty_validator import validate_difficulty_and_bloom, adjust_difficulty
from validators.bloom_validator import validate_bloom_level
from validators.rule_engine import run_rule_engine
from validators.quality_validator import validate_quality_batch, normalize_match_questions
from validators.distractor_validator import validate_distractors
from validators.match_formatter import normalize_match_question, validate_match_format
from validators.answer_leak_validator import validate_answer_leaks

# Chapter/course alignment checks are thin enough to live here directly
# (they don't warrant their own module).
from models import Question


def validate_chapter_alignment(question: Question, expected_chapter_name: str) -> tuple[bool, str]:
    """Verify the question's chapter field matches the expected chapter."""
    if question.chapter.strip().lower() != expected_chapter_name.strip().lower():
        return (
            False,
            f"Chapter mismatch: got '{question.chapter}', expected '{expected_chapter_name}'",
        )
    return True, "Chapter alignment passed"


def validate_course_alignment(question: Question) -> tuple[bool, str]:
    """Reject questions that reference out-of-scope advanced topics."""
    out_of_scope = [
        "java programming", "c++", "data structures", "compiler design",
        "operating system kernel", "assembly language",
    ]
    q_lower = question.question.lower()
    for term in out_of_scope:
        if term in q_lower:
            return False, f"Out-of-scope topic referenced: '{term}'"
    return True, "Course alignment passed"


__all__ = [
    "validate_schema",
    "validate_correctness",
    "validate_correctness_batch",
    "validate_formatting",
    "check_answer_distribution",
    "check_duplicates",
    "is_near_duplicate",
    "validate_difficulty_and_bloom",
    "adjust_difficulty",
    "validate_bloom_level",
    "run_rule_engine",
    "validate_quality_batch",
    "normalize_match_questions",
    "validate_distractors",
    "normalize_match_question",
    "validate_match_format",
    "validate_chapter_alignment",
    "validate_course_alignment",
]
