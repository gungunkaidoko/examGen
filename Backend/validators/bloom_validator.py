"""
Bloom Validator
---------------
Validates and enriches the Bloom's taxonomy level on a Question.

The difficulty_validator already classifies Bloom's level as a side-effect
of its LLM call. This module provides:

  1. validate_bloom_level()   — checks that the bloom_level on the question
                                is consistent with its difficulty bracket.
  2. bloom_for_difficulty()   — helper that returns expected BloomLevel values
                                for a given difficulty.

No additional LLM calls — purely deterministic cross-check.
"""

from models import Question, Difficulty, BloomLevel

# Expected Bloom's levels for each difficulty
DIFFICULTY_BLOOM_MAP: dict[Difficulty, set[BloomLevel]] = {
    Difficulty.easy:   {BloomLevel.remember, BloomLevel.understand},
    Difficulty.medium: {BloomLevel.understand, BloomLevel.apply},
    Difficulty.hard:   {BloomLevel.apply, BloomLevel.analyze},
}


def bloom_for_difficulty(difficulty: Difficulty) -> set[BloomLevel]:
    """Return the expected BloomLevel values for a given difficulty."""
    return DIFFICULTY_BLOOM_MAP[difficulty]


def validate_bloom_level(question: Question) -> tuple[bool, str]:
    """
    Check that the question's bloom_level is consistent with its difficulty.

    For example, a 'hard' question should not have bloom_level='remember'.

    Returns:
        (passed, reason)
    """
    expected = DIFFICULTY_BLOOM_MAP.get(question.difficulty, set())
    if expected and question.bloom_level not in expected:
        return (
            False,
            (
                f"bloom_level '{question.bloom_level.value}' is inconsistent with "
                f"difficulty '{question.difficulty.value}'. "
                f"Expected one of: {[b.value for b in expected]}"
            ),
        )
    return True, f"Bloom level '{question.bloom_level.value}' is consistent with '{question.difficulty.value}'"
