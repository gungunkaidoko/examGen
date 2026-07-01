"""
Pydantic models for structured output and database records.
"""

import uuid
import hashlib
from enum import Enum
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


def _compute_hash(question_text: str) -> str:
    """SHA-256 of the normalised (lowercased, stripped) question text."""
    normalised = question_text.strip().lower()
    return hashlib.sha256(normalised.encode("utf-8")).hexdigest()


class Difficulty(str, Enum):
    easy = "easy"
    medium = "medium"
    hard = "hard"


class BloomLevel(str, Enum):
    remember = "remember"
    understand = "understand"
    apply = "apply"
    analyze = "analyze"


class ValidationStatus(str, Enum):
    approved = "approved"
    rejected = "rejected"
    pending = "pending"


class Question(BaseModel):
    """
    Core question model with full metadata including Bloom's taxonomy,
    UUID, tags, and weightage for enriched exam bank storage.
    """

    question: str = Field(..., min_length=10, description="The MCQ question text")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 options")
    correct_answer: str = Field(..., description="The correct option text (must match one of the options)")
    difficulty: Difficulty = Field(..., description="Difficulty level: easy | medium | hard")
    chapter: str = Field(..., description="Chapter name this question belongs to")
    validation_status: ValidationStatus = Field(
        default=ValidationStatus.approved,
        description="Validation result"
    )
    bloom_level: BloomLevel = Field(
        default=BloomLevel.remember,
        description="Bloom's taxonomy cognitive level"
    )
    question_uuid: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for deduplication and tracking"
    )
    question_hash: str = Field(
        default="",
        description="SHA-256 of normalised question text for fast exact dedup"
    )
    question_type: str = Field(
        default="standard",
        description=(
            "MCQ format type: standard | assertion_reason | statement_based | "
            "scenario_based | fill_blank | match_following | sequence_order"
        ),
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Topic keywords derived from chapter content"
    )
    weightage: float = Field(
        default=1.0,
        description="Exam weight based on chapter blueprint"
    )

    # ── Generation metadata ──────────────────────────────────────────────────
    model_name: str = Field(
        default="",
        description="LLM model identifier used to generate this question",
    )
    prompt_version: str = Field(
        default="v1",
        description="Version tag of the prompt template used for generation",
    )
    generation_time: float = Field(
        default=0.0,
        description="Wall-clock seconds taken to generate this question",
    )
    retry_count: int = Field(
        default=0,
        description="Number of validation retries before this question was accepted",
    )

    @field_validator("options")
    @classmethod
    def options_must_be_four(cls, v: List[str]) -> List[str]:
        if len(v) != 4:
            raise ValueError("Exactly 4 options are required")
        return v

    @field_validator("correct_answer")
    @classmethod
    def correct_answer_must_be_in_options(cls, v: str, info) -> str:
        options = info.data.get("options", [])
        if options and v not in options:
            raise ValueError(f"correct_answer '{v}' must be one of the options: {options}")
        return v

    def model_post_init(self, __context) -> None:
        """Auto-compute question_hash after model initialisation if not set."""
        if not self.question_hash and self.question:
            object.__setattr__(self, "question_hash", _compute_hash(self.question))

    def to_db_dict(self) -> dict:
        """Serialize for PostgreSQL insertion."""
        vs = self.validation_status
        vs_val = vs.value if isinstance(vs, ValidationStatus) else str(vs)
        d = self.difficulty
        d_val = d.value if isinstance(d, Difficulty) else str(d)
        bl = self.bloom_level
        bl_val = bl.value if isinstance(bl, BloomLevel) else str(bl)
        return {
            "question_uuid": self.question_uuid,
            "question_hash": self.question_hash or _compute_hash(self.question),
            "question": self.question,
            "option_a": self.options[0],
            "option_b": self.options[1],
            "option_c": self.options[2],
            "option_d": self.options[3],
            "correct_answer": self.correct_answer,
            "difficulty": d_val,
            "bloom_level": bl_val,
            "chapter": self.chapter,
            "validation_status": vs_val,
            "tags": self.tags,
            "weightage": self.weightage,
            "question_type": self.question_type,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "generation_time": self.generation_time,
            "retry_count": self.retry_count,
        }

    def to_output_dict(self) -> dict:
        """Output format for JSON export."""
        vs = self.validation_status
        vs_val = vs.value if isinstance(vs, ValidationStatus) else str(vs)
        d = self.difficulty
        d_val = d.value if isinstance(d, Difficulty) else str(d)
        bl = self.bloom_level
        bl_val = bl.value if isinstance(bl, BloomLevel) else str(bl)
        return {
            "question_uuid": self.question_uuid,
            "question_hash": self.question_hash or _compute_hash(self.question),
            "question": self.question,
            "options": self.options,
            "correct_answer": self.correct_answer,
            "difficulty": d_val,
            "bloom_level": bl_val,
            "chapter": self.chapter,
            "validation_status": vs_val,
            "tags": self.tags,
            "weightage": self.weightage,
            "question_type": self.question_type,
            "model_name": self.model_name,
            "prompt_version": self.prompt_version,
            "generation_time": self.generation_time,
            "retry_count": self.retry_count,
        }


class GeneratedQuestionBatch(BaseModel):
    """LLM response wrapper — a batch of questions."""
    questions: List[Question]


class ValidationResult(BaseModel):
    """Result from the LLM correctness validator."""
    is_correct: bool = Field(..., description="Whether the marked answer is factually correct")
    reasoning: str = Field(..., description="Brief explanation of the verdict")
    suggested_answer: str = Field(default="", description="Suggested correct answer if is_correct is False")


class IndependentSolveResult(BaseModel):
    """Result from the independent-solve correctness check."""
    answer: str = Field(..., description="The answer the LLM independently determined is correct")


class DifficultyClassifyResult(BaseModel):
    """Result from the LLM difficulty + Bloom classifier."""
    difficulty: str = Field(..., description="Classified difficulty: easy | medium | hard")
    bloom_level: str = Field(..., description="Bloom's taxonomy level")


class ExamSet(BaseModel):
    """One complete exam set of 100 questions."""
    set_number: int
    questions: List[Question]

    @property
    def total(self) -> int:
        return len(self.questions)
