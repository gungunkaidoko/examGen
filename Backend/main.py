"""
CCC Exam Generation Platform — Main Entry Point
================================================
Generates 10 exam sets × 100 questions each using the Gemini 2.5 LLM.

Usage:
    python main.py                    # Generate all 10 sets
    python main.py --sets 1 2 3       # Generate only sets 1, 2, 3
    python main.py --no-db            # Skip PostgreSQL, output JSON only
    python main.py --dry-run          # Print allocation plan without generating
"""

import argparse
import logging
import sys
import os
from typing import Optional

# ── Setup path so local modules are importable ──────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import NUM_EXAM_SETS, QUESTIONS_PER_EXAM, DB_CONFIG
from content_extractor import ContentExtractor
from blueprint_allocator import allocate_questions, allocation_summary
from pipeline import generate_exam_set
from exporter import export_exam_set_json, export_answer_key, export_combined, print_set_stats

# ── Logging configuration ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("generation.log", mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CCC National Exam Generation Platform"
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        type=int,
        default=list(range(1, NUM_EXAM_SETS + 1)),
        metavar="N",
        help=f"Set numbers to generate (default: 1–{NUM_EXAM_SETS})",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Skip PostgreSQL storage, output JSON files only",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the question allocation plan per set and exit",
    )
    return parser.parse_args()


def dry_run_report(set_numbers: list[int]) -> None:
    print("\n" + "=" * 60)
    print("  CCC EXAM PLATFORM — DRY RUN ALLOCATION REPORT")
    print("=" * 60)
    for s in set_numbers:
        print(allocation_summary(s))
        print()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        dry_run_report(args.sets)
        return

    # ── Load all chapter content once ───────────────────────────────────────
    logger.info("Loading chapter knowledge base...")
    extractor = ContentExtractor()
    chapter_contents = extractor.load_all()
    logger.info(f"✓ Loaded {len(chapter_contents)} chapters")

    # ── Optional: connect to PostgreSQL ─────────────────────────────────────
    db = None
    if not args.no_db:
        try:
            from database import QuestionBankDB
            db = QuestionBankDB()
            db.connect()
        except Exception as e:
            logger.warning(
                f"Could not connect to PostgreSQL: {e}\n"
                "Continuing with JSON-only output. "
                "Use --no-db to suppress this warning."
            )
            db = None

    # ── Global dedup registry — shared across all sets this run ─────────────
    global_seen_questions: set[str] = set()       # lowercased text — exact lookup
    global_seen_list: list[str] = []              # lowercased text — fuzzy lookup
    global_seen_hashes: set[str] = set()          # SHA-256 hashes  — fastest exact lookup
    if db is not None:
        try:
            global_seen_questions = db.load_all_questions_lowercase()
            global_seen_list = list(global_seen_questions)
            global_seen_hashes = db.load_all_question_hashes()
            logger.info(
                f"✓ Pre-loaded {len(global_seen_questions)} existing questions "
                f"and {len(global_seen_hashes)} hashes into dedup registry"
            )
        except Exception as e:
            logger.warning(f"Could not pre-load dedup registry from DB: {e}")

    # ── Generate exam sets ───────────────────────────────────────────────────
    all_exam_sets = []

    print("\n" + "=" * 60)
    print(f"  CCC EXAM GENERATION PLATFORM")
    print(f"  Generating {len(args.sets)} exam set(s) × {QUESTIONS_PER_EXAM} questions")
    print("=" * 60)

    for set_number in args.sets:
        logger.info(f"\n{'='*60}")
        logger.info(f"EXAM SET {set_number}/{max(args.sets)}")
        logger.info(f"{'='*60}")

        allocation = allocate_questions(set_number)

        try:
            exam_set = generate_exam_set(
                set_number=set_number,
                chapter_contents=chapter_contents,
                allocation=allocation,
                global_seen=global_seen_questions,
                global_seen_list=global_seen_list,
                global_seen_hashes=global_seen_hashes,
                db=db,
            )
        except Exception as e:
            logger.error(f"Failed to generate exam set {set_number}: {e}", exc_info=True)
            continue

        # Print stats
        print_set_stats(exam_set)

        # Export to JSON
        export_exam_set_json(exam_set)
        export_answer_key(exam_set)

        # Save to PostgreSQL if connected
        if db is not None:
            try:
                db.save_exam_set(exam_set)
            except Exception as e:
                logger.error(f"DB save failed for set {set_number}: {e}")

        all_exam_sets.append(exam_set)
        logger.info(f"✓ Exam set {set_number} complete — {exam_set.total} questions")

    # ── Final combined export ────────────────────────────────────────────────
    if all_exam_sets:
        export_combined(all_exam_sets)

    if db is not None:
        db.close()

    print("\n" + "=" * 60)
    print(f"  GENERATION COMPLETE")
    print(f"  Sets generated : {len(all_exam_sets)}")
    print(f"  Output folder  : {os.path.join(os.path.dirname(__file__), 'output')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
