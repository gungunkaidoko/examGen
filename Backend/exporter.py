"""
Exporter
--------
Saves generated exam sets to:
  - JSON files  (one per set: exam_set_01.json … exam_set_10.json)
  - A combined  all_exam_sets.json  with all sets
  - A clean answer-key file per set
"""

import json
import os
import logging
from datetime import datetime

from models import ExamSet

logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def export_exam_set_json(exam_set: ExamSet) -> str:
    """Export one exam set to a JSON file. Returns the file path."""
    out_dir = _ensure_output_dir()
    filename = f"exam_set_{exam_set.set_number:02d}.json"
    filepath = os.path.join(out_dir, filename)

    data = {
        "exam_set": exam_set.set_number,
        "course": "CCC",
        "total_questions": exam_set.total,
        "generated_at": datetime.now().isoformat(),
        "questions": [q.to_output_dict() for q in exam_set.questions],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✓ Exported: {filepath}")
    return filepath


def export_answer_key(exam_set: ExamSet) -> str:
    """Export answer key as a compact JSON. Returns file path."""
    out_dir = _ensure_output_dir()
    filename = f"answer_key_{exam_set.set_number:02d}.json"
    filepath = os.path.join(out_dir, filename)

    key = {
        "exam_set": exam_set.set_number,
        "answer_key": [
            {
                "q_no": i + 1,
                "question": q.question,
                "correct_answer": q.correct_answer,
                "difficulty": q.difficulty.value,
                "chapter": q.chapter,
            }
            for i, q in enumerate(exam_set.questions)
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(key, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✓ Answer key: {filepath}")
    return filepath


def export_combined(exam_sets: list[ExamSet]) -> str:
    """Export all sets into one file for easy review. Returns file path."""
    out_dir = _ensure_output_dir()
    filepath = os.path.join(out_dir, "all_exam_sets.json")

    combined = {
        "course": "CCC",
        "total_sets": len(exam_sets),
        "questions_per_set": exam_sets[0].total if exam_sets else 0,
        "generated_at": datetime.now().isoformat(),
        "exam_sets": [
            {
                "set_number": es.set_number,
                "questions": [q.to_output_dict() for q in es.questions],
            }
            for es in exam_sets
        ],
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)

    logger.info(f"  ✓ Combined export: {filepath}")
    return filepath


def print_set_stats(exam_set: ExamSet) -> None:
    """Print chapter and difficulty breakdown for a set."""
    from collections import Counter

    chapters = Counter(q.chapter for q in exam_set.questions)
    difficulties = Counter(q.difficulty.value for q in exam_set.questions)

    print(f"\n  Exam Set {exam_set.set_number} — Stats:")
    print(f"  Total questions: {exam_set.total}")
    print(f"  Difficulty: {dict(difficulties)}")
    print("  By chapter:")
    for ch, cnt in sorted(chapters.items()):
        print(f"    {ch}: {cnt}")
