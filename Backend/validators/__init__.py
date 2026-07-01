"""
validators/
-----------
Modular validation package. Each module owns one concern:

  schema_validator.py      — raw dict pre-validation (before Pydantic)
  correctness_validator.py — LLM judge + independent solve (Layer 1)
  formatting_validator.py  — deterministic text/option formatting checks
  duplicate_validator.py   — exact + fuzzy deduplication (within & across sets)
  difficulty_validator.py  — keyword pre-filter + LLM difficulty/Bloom classifier
  bloom_validator.py       — Bloom taxonomy consistency cross-check
  rule_engine.py           — orchestrates structural + formatting + dedup checks
  quality_validator.py     — post-generation holistic batch quality checks
  distractor_validator.py  — embedding + LLM distractor plausibility validation
  match_formatter.py       — auto-normalise match_following Column A/B layout
"""

from validators.schema_validator import validate_schema
from validators.correctness_validator import validate_correctness
from validators.formatting_validator import validate_formatting, check_answer_distribution
from validators.duplicate_validator import check_duplicates, is_near_duplicate
from validators.difficulty_validator import validate_difficulty_and_bloom, adjust_difficulty
from validators.bloom_validator import validate_bloom_level
from validators.rule_engine import run_rule_engine
from validators.quality_validator import validate_quality_batch, normalize_match_questions
from validators.distractor_validator import validate_distractors, score_distractors_embedding
from validators.match_formatter import normalize_match_question, validate_match_format
from validators.answer_leak_validator import validate_answer_leaks

__all__ = [
    "validate_schema",
    "validate_correctness",
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
    "score_distractors_embedding",
    "normalize_match_question",
    "validate_match_format",
]
