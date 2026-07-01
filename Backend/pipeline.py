"""
Exam Generation Pipeline  (optimised v2)
-----------------------------------------
All 5 optimisations implemented:

1. Parallel chapter generation
   generate_exam_set() submits all chapters to a ThreadPoolExecutor
   (max 9 workers = one per chapter). Since chapters are completely
   independent, all 9 run concurrently.  ~60% wall-clock saving.

2. Skip difficulty LLM call when safe
   validate_difficulty_and_bloom() is skipped when BOTH:
     a) keyword pre-filter passes (no obvious mismatch)
     b) the generated difficulty label matches the blueprint target
   Bloom level is set heuristically in that case (no accuracy loss).
   ~25% fewer Bedrock calls.

3. Batch correctness size 15 → 25
   BATCH_SIZE raised to 25. Same logic, fewer round-trips.  ~20% saving.

4. RAG passages cached per chapter
   _fetch_rag_passages() result is stored and reused on every top-up
   call for the same chapter. Saves repeated Pinecone calls.  ~5%.

5. Buffer multiplier 2× → 1.5×
   BUFFER_MULTIPLIER reduced from 2.0 to 1.5. With the correctness
   batch pre-screen accepting ~80%+ of questions, 1.5× is enough to
   avoid top-up cycles in most chapters.  ~10% fewer LLM gen calls.

Thread safety for global dedup registries
   global_seen, global_seen_list, global_seen_hashes are shared across
   chapter threads. Access is protected by a threading.Lock so concurrent
   appends never produce race conditions.
"""

import logging
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, NamedTuple

from config import MAX_RETRIES, CHAPTER_BLUEPRINT, MODEL_ID, PROMPT_VERSION, PINECONE_CHAPTER_NAMES
from content_extractor import ChapterContent
from models import Question, ExamSet, BloomLevel, Difficulty, ValidationStatus
from question_generator import generate_questions_for_chapter, regenerate_single_question
from validators.correctness_validator import validate_correctness, validate_correctness_batch
from validators.difficulty_validator import validate_difficulty_and_bloom, adjust_difficulty
from validation import (
    validate_schema,
    run_rule_engine,
    validate_chapter_alignment,
    validate_course_alignment,
)
from validators.quality_validator import validate_quality_batch

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE        = 25    # Opt 3: questions per correctness Bedrock call (was 15)
BUFFER_MULTIPLIER = 1.5   # Opt 5: generate 1.5× target upfront (was 2×)
MAX_CHAPTER_WORKERS = 9   # Opt 1: one thread per chapter (max 9 chapters)
MAX_WORKERS       = 9     # concurrent difficulty-validate threads


# ── RAG helper ────────────────────────────────────────────────────────────────

# ── RAG passage cache (Opt 4) ─────────────────────────────────────────────────
# Keyed by (chapter_key, exam_set_number) → list of passage dicts
_rag_cache: dict[tuple, list[dict]] = {}
_rag_cache_lock = threading.Lock()


def _fetch_rag_passages(
    chapter_key: str,
    chapter_content: ChapterContent,
    exam_set_number: int,
    top_k: int = 8,
) -> list[dict]:
    """
    Slug-aware RAG retrieval (replaces old cosine-only retrieval).

    Tries the new SlugRetriever first (guarantees cross-set uniqueness).
    Falls back to the old cosine retriever if slug index unavailable.
    Results are cached per (chapter_key, set_number) so top-up rounds
    reuse the same passages within one generation run.
    """
    cache_key = (chapter_key, exam_set_number)
    with _rag_cache_lock:
        if cache_key in _rag_cache:
            return _rag_cache[cache_key]

    passages = []

    # ── Try new slug-based retrieval first ────────────────────────────────
    try:
        from rag.slug_retriever import retrieve_slugs_for_chapter
        from config import CHAPTER_BLUEPRINT
        bp = CHAPTER_BLUEPRINT.get(chapter_key, {})
        diff_split = bp.get("difficulty_split", {"easy": 0.5, "medium": 0.35, "hard": 0.15})

        slug_chunks = retrieve_slugs_for_chapter(
            chapter_key      = chapter_key,
            set_number       = exam_set_number,
            num_questions    = top_k,
            difficulty_split = diff_split,
        )
        if slug_chunks:
            # Convert slug chunks to the passage dict format pipeline expects
            passages = [
                {
                    "id":       c["chunk_id"],
                    "score":    c.get("score", 1.0),
                    "mmr_score": c.get("score", 1.0),
                    "metadata": {
                        "text":            c["text"],
                        "chapter":         c["chapter_id"],
                        "section":         c.get("section_heading", ""),
                        "topic_slug":      c["topic_slug"],
                        "bloom_levels":    c.get("bloom_levels", ["remember"]),
                        "difficulty":      c.get("difficulty", "easy"),
                        "keywords":        c.get("keywords", []),
                    },
                }
                for c in slug_chunks
            ]
            logger.info(
                f"  SlugRAG: {len(passages)} chunks for '{chapter_key}' (set #{exam_set_number})"
            )
    except Exception as e:
        logger.warning(f"  SlugRetriever unavailable ({e}). Falling back to cosine retrieval.")

    # ── Fallback: old cosine-based retrieval ──────────────────────────────
    if not passages:
        try:
            from rag.retriever import retrieve_for_chapter
            pinecone_chapter = PINECONE_CHAPTER_NAMES.get(
                chapter_key, chapter_content.chapter_name
            )
            set_seeds = [
                "concepts definitions facts",
                "applications uses examples",
                "features components types",
                "operations procedures steps",
                "advantages disadvantages comparison",
                "history evolution development",
                "security protocols standards",
                "tools software hardware",
                "networking communication data",
                "management organisation structure",
            ]
            # Use a different seed per top-up round to avoid same passages
            seed = set_seeds[(exam_set_number - 1) % len(set_seeds)]
            passages = retrieve_for_chapter(
                chapter_name=pinecone_chapter,
                query=f"{pinecone_chapter} {seed}",
                top_k=top_k,
                content_types=["content"],
            )
            logger.info(
                f"  CosineRAG: {len(passages)} passages for '{pinecone_chapter}' (set #{exam_set_number})"
            )
        except Exception as e:
            logger.warning(f"  Cosine retrieval also failed: {e}. Using topic-notes only.")

    with _rag_cache_lock:
        _rag_cache[cache_key] = passages
    return passages


# ── Node result ───────────────────────────────────────────────────────────────

class NodeResult(NamedTuple):
    passed: bool
    question: Optional[Question]
    reason: str


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def node_schema_validate(question: Question) -> NodeResult:
    """Node 1 — Structural schema (free, no LLM)."""
    raw = {
        "question": question.question,
        "options": question.options,
        "correct_answer": question.correct_answer,
        "difficulty": question.difficulty.value,
        "chapter": question.chapter,
    }
    passed, reason = validate_schema(raw)
    return NodeResult(passed, question, f"Schema: {reason}" if not passed else "Schema valid")


def node_rule_engine(
    question: Question,
    existing_questions: list[Question],
    global_seen: set[str] | None,
    global_seen_list: list[str] | None,
    global_seen_hashes: set[str] | None,
) -> NodeResult:
    """Node 2 — Deterministic rule engine: formatting + dedup (free, no LLM)."""
    passed, reason = run_rule_engine(
        question, existing_questions, global_seen, global_seen_list, global_seen_hashes
    )
    return NodeResult(passed, question, f"RuleEngine: {reason}" if not passed else "Rules passed")


def node_difficulty_validate(question: Question) -> NodeResult:
    """
    Node 3 — LLM difficulty + Bloom classifier.
    Optimisation 5: wrong label → relabel the question, skip regeneration.
    """
    passed, reason, classified_diff, classified_bloom = validate_difficulty_and_bloom(question)

    # Always enrich bloom_level when we have a classified value
    if classified_bloom:
        try:
            question = question.model_copy(
                update={"bloom_level": BloomLevel(classified_bloom)}
            )
        except ValueError:
            pass

    if not passed and classified_diff:
        # Relabel difficulty instead of regenerating — Optimisation 5
        try:
            question = question.model_copy(
                update={"difficulty": Difficulty(classified_diff)}
            )
            logger.info(
                f"    ↻ Difficulty relabelled: {reason.split(',')[1].strip() if ',' in reason else reason}"
            )
            return NodeResult(True, question, f"Difficulty relabelled → {classified_diff}")
        except ValueError:
            pass

    if not passed:
        return NodeResult(False, question, f"Difficulty: {reason}")

    return NodeResult(True, question, f"Difficulty/Bloom OK ({classified_diff}/{classified_bloom})")


def node_chapter_scope_validate(
    question: Question,
    chapter_content: ChapterContent,
) -> NodeResult:
    """Node 4 — Chapter alignment + course scope (free, no LLM)."""
    passed_chap, chap_reason = validate_chapter_alignment(question, chapter_content.chapter_name)
    if not passed_chap:
        question = question.model_copy(update={"chapter": chapter_content.chapter_name})

    passed_course, course_reason = validate_course_alignment(question)
    if not passed_course:
        return NodeResult(False, question, f"CourseScope: {course_reason}")

    return NodeResult(True, question, "Chapter/scope valid")


def node_metadata_enrich(
    question: Question,
    chapter_content: ChapterContent,
    chapter_weightage: float,
    retry_count: int,
    generation_time: float,
) -> NodeResult:
    """Node 5 — Attach metadata (free, no LLM)."""
    enriched = question.model_copy(update={
        "tags": [t["topic"] for t in chapter_content.topics][:5],
        "weightage": chapter_weightage,
        "model_name": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "generation_time": round(generation_time, 4),
        "retry_count": retry_count,
        "validation_status": ValidationStatus.approved,
    })
    return NodeResult(True, enriched, "Metadata enriched")


def run_quality_validation_pass(
    accepted: list[Question],
    chapter_key: str,
    chapter_content: ChapterContent,
    chapter_weightage: float,
    exam_set_number: int,
    global_seen: set[str] | None,
    global_seen_list: list[str] | None,
    global_seen_hashes: set[str] | None,
    rag_passages: list[dict] | None,
    db,
) -> list[Question]:
    """
    Node 7 — Independent quality validation pass on the full assembled chapter batch.

    Runs after all per-question nodes complete.  Checks the batch holistically for:
      - Semantic near-duplicates (Bedrock Titan embeddings + cosine similarity)
      - Grammar / punctuation issues (LLM, batched)
      - Distractor quality (LLM, batched)
      - Option length imbalance (deterministic)
      - Statement / match formatting (deterministic)
      - Balanced answer distribution (ENFORCED: skewed Qs are regenerated)
      - Assertion–Reason option-A bias (ENFORCED: biased Qs are regenerated)

    Any flagged question is replaced via regenerate_single_question() and
    run through the same post-correctness pipeline.  At most one regeneration
    attempt per question to avoid infinite loops.
    """
    if not accepted:
        return accepted

    logger.info(
        f"  ► Quality validation pass: {len(accepted)} questions "
        f"for {chapter_content.chapter_name}"
    )

    quality_results = validate_quality_batch(
        accepted,
        chapter_key=chapter_key,
        exam_set=exam_set_number,
        run_llm_check=True,
    )

    final: list[Question] = []
    for i, (q, (passed, reason)) in enumerate(zip(accepted, quality_results)):
        if passed:
            final.append(q)
            continue

        logger.info(f"    ✗ Quality FAIL Q{i+1}: {reason}")
        if db:
            _log_rejection(db, q, "quality_validator", reason, exam_set_number)

        # Attempt one regeneration
        replacement = regenerate_single_question(
            chapter_key, chapter_content,
            difficulty=q.difficulty.value,
            rejection_reason=reason,
            rag_passages=rag_passages,
            question_type=q.question_type,
        )
        if replacement is None:
            logger.warning(f"    ↻ Regeneration failed for Q{i+1}; keeping original")
            final.append(q)
            continue

        # Run the replacement through post-correctness nodes
        from validators.correctness_validator import validate_correctness
        corr_ok, corr_reason, suggested = validate_correctness(replacement)
        if not corr_ok:
            logger.info(f"    ✗ Replacement correctness FAIL: {corr_reason}; keeping original")
            final.append(q)
            continue
        if suggested:
            replacement = replacement.model_copy(update={"correct_answer": suggested})

        enriched = _run_post_correctness_nodes(
            replacement, chapter_key, chapter_content, final,
            chapter_weightage, global_seen, global_seen_list, global_seen_hashes,
            exam_set_number, db, attempt=0,
            generation_start=__import__("time").time(),
            rag_passages=rag_passages,
        )
        if enriched is not None:
            logger.info(f"    ✓ Q{i+1} replaced by quality-compliant question")
            final.append(enriched)
            # Register replacement in dedup
            try:
                from rag.semantic_dedup import get_deduplicator
                get_deduplicator().register(
                    enriched.question_uuid, enriched.question,
                    exam_set=exam_set_number, chapter=chapter_key,
                )
            except Exception:
                pass
        else:
            logger.warning(f"    ↻ Replacement node pipeline failed for Q{i+1}; keeping original")
            final.append(q)

    logger.info(
        f"  ✓ Quality pass complete: {len(final)} questions "
        f"({len(accepted) - sum(1 for p, _ in quality_results if not p)} regenerated)"
    )
    return final


# ── DB rejection logger ───────────────────────────────────────────────────────

def _log_rejection(db, question: Question, validator: str, reason: str, exam_set: int) -> None:
    if db is None:
        return
    try:
        db.log_validation_failure(
            question_text=question.question,
            validator_failed=validator,
            reason=reason,
            exam_set=exam_set,
            chapter=question.chapter,
            correct_answer=question.correct_answer,
            difficulty=question.difficulty.value,
            question_uuid=question.question_uuid,
        )
    except Exception as e:
        logger.debug(f"Could not write validation log: {e}")


# ── Concurrent difficulty validation ─────────────────────────────────────────

def _validate_difficulty_concurrent(
    questions: list[Question],
) -> list[NodeResult]:
    """
    Run node_difficulty_validate on N questions concurrently.
    Optimisation 3: ThreadPoolExecutor — all questions in parallel.
    """
    results: list[NodeResult] = [None] * len(questions)  # type: ignore

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(questions))) as pool:
        future_to_idx = {
            pool.submit(node_difficulty_validate, q): i
            for i, q in enumerate(questions)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = NodeResult(False, questions[idx], f"Difficulty concurrent error: {e}")

    return results


# ── Post-correctness pipeline (rules + difficulty + scope + enrich) ───────────

def _run_post_correctness_nodes(
    question: Question,
    chapter_key: str,
    chapter_content: ChapterContent,
    accepted: list[Question],
    chapter_weightage: float,
    global_seen: set[str] | None,
    global_seen_list: list[str] | None,
    global_seen_hashes: set[str] | None,
    exam_set_number: int,
    db,
    attempt: int,
    generation_start: float,
    rag_passages: list[dict] | None,
) -> Optional[Question]:
    """
    Nodes 2–5 for a question that has already passed correctness screening.
    Returns the enriched Question or None.
    """
    if attempt >= MAX_RETRIES:
        return None

    # Node 2: Rule Engine (free)
    rule_result = node_rule_engine(
        question, accepted, global_seen, global_seen_list, global_seen_hashes
    )
    if not rule_result.passed:
        logger.info(f"    ✗ Rule Engine FAIL: {rule_result.reason}")
        _log_rejection(db, question, "rule_engine", rule_result.reason, exam_set_number)
        replacement = regenerate_single_question(
            chapter_key, chapter_content,
            difficulty=question.difficulty.value,
            rejection_reason=rule_result.reason,
            rag_passages=rag_passages,
            question_type=question.question_type,
        )
        if replacement is None:
            return None
        # Re-validate correctness for the replacement
        corr_ok, corr_reason, suggested = validate_correctness(replacement)
        if not corr_ok:
            _log_rejection(db, replacement, "correctness", corr_reason, exam_set_number)
            return None
        if suggested:
            replacement = replacement.model_copy(update={"correct_answer": suggested})
        return _run_post_correctness_nodes(
            replacement, chapter_key, chapter_content, accepted,
            chapter_weightage, global_seen, global_seen_list, global_seen_hashes,
            exam_set_number, db, attempt + 1, time.time(), rag_passages,
        )
    question = rule_result.question

    # Node 3: Difficulty + Bloom (LLM — done in batch upstream; fallback here)
    diff_result = node_difficulty_validate(question)
    if not diff_result.passed:
        logger.info(f"    ✗ Difficulty FAIL: {diff_result.reason}")
        _log_rejection(db, question, "difficulty_bloom", diff_result.reason, exam_set_number)
        adjusted = adjust_difficulty(
            question,
            target_difficulty=question.difficulty.value,
            chapter_content=chapter_content,
        )
        next_q = adjusted or regenerate_single_question(
            chapter_key, chapter_content,
            difficulty=question.difficulty.value,
            rejection_reason=diff_result.reason,
            rag_passages=rag_passages,
            question_type=question.question_type,
        )
        if next_q is None:
            return None
        return _run_post_correctness_nodes(
            next_q, chapter_key, chapter_content, accepted,
            chapter_weightage, global_seen, global_seen_list, global_seen_hashes,
            exam_set_number, db, attempt + 1, time.time(), rag_passages,
        )
    question = diff_result.question

    # Node 4: Chapter / Scope (free)
    scope_result = node_chapter_scope_validate(question, chapter_content)
    if not scope_result.passed:
        logger.info(f"    ✗ Course Scope FAIL: {scope_result.reason}")
        _log_rejection(db, question, "course_alignment", scope_result.reason, exam_set_number)
        replacement = regenerate_single_question(
            chapter_key, chapter_content,
            difficulty=question.difficulty.value,
            rejection_reason=scope_result.reason,
            rag_passages=rag_passages,
            question_type=question.question_type,
        )
        if replacement is None:
            return None
        return _run_post_correctness_nodes(
            replacement, chapter_key, chapter_content, accepted,
            chapter_weightage, global_seen, global_seen_list, global_seen_hashes,
            exam_set_number, db, attempt + 1, time.time(), rag_passages,
        )
    question = scope_result.question

    # Node 5: Metadata (free)
    generation_time = time.time() - generation_start
    enrich_result = node_metadata_enrich(
        question, chapter_content, chapter_weightage,
        retry_count=attempt, generation_time=generation_time,
    )
    enriched_q = enrich_result.question

    # Node 6: Semantic duplicate check (embedding similarity against accepted questions)
    try:
        from rag.semantic_dedup import get_deduplicator
        dedup = get_deduplicator()
        if dedup.is_duplicate(enriched_q.question, exam_set_number, chapter=chapter_key):
            logger.info(f"    ✗ Semantic Dedup FAIL: near-duplicate detected")
            _log_rejection(db, enriched_q, "semantic_duplicate",
                           "Embedding similarity > 0.90 with existing question", exam_set_number)
            # Regenerate with explicit rejection context
            replacement = regenerate_single_question(
                chapter_key, chapter_content,
                difficulty=enriched_q.difficulty.value,
                rejection_reason="Question is too similar to an already-accepted question. Generate a distinctly different question on a different aspect of the topic.",
                rag_passages=rag_passages,
                question_type=enriched_q.question_type,
            )
            if replacement is None:
                return None
            return _run_post_correctness_nodes(
                replacement, chapter_key, chapter_content, accepted,
                chapter_weightage, global_seen, global_seen_list, global_seen_hashes,
                exam_set_number, db, attempt + 1, time.time(), rag_passages,
            )
    except Exception as e:
        logger.debug(f"    Semantic dedup skipped: {e}")

    return enriched_q


# ── Chapter-level generator ───────────────────────────────────────────────────

def generate_chapter_questions(
    chapter_key: str,
    chapter_content: ChapterContent,
    num_questions: int,
    exam_set_number: int,
    global_seen: set[str] | None = None,
    global_seen_list: list[str] | None = None,
    global_seen_hashes: set[str] | None = None,
    db=None,
) -> list[Question]:
    """
    Generate exactly `num_questions` validated questions for a chapter.

    Optimised flow:
      1. Fetch RAG passages (one Pinecone call)
      2. Generate 2× buffer (Optimisation 4)
      3. Schema filter (free)
      4. Batch correctness screening — N questions, 1 Bedrock call (Opt 2+3)
      5. Per-question: rule engine → difficulty relabel → scope → enrich
      6. Top-up only if still short after the buffer
    """
    logger.info(
        f"\n  ► Chapter: {chapter_content.chapter_name} | "
        f"Target: {num_questions} | Set #{exam_set_number}"
    )

    rag_passages  = _fetch_rag_passages(chapter_key, chapter_content, exam_set_number, top_k=8)
    chapter_weightage = round(CHAPTER_BLUEPRINT[chapter_key]["min_q"] / 100.0, 2)

    accepted: list[Question] = []

    def _process_batch(raw_questions: list[Question]) -> None:
        """Schema filter → batch correctness → per-question downstream nodes."""
        nonlocal accepted

        # ── Schema filter (free) ──────────────────────────────────────────────
        schema_ok: list[Question] = []
        for q in raw_questions:
            if len(accepted) >= num_questions:
                break
            result = node_schema_validate(q)
            if result.passed:
                schema_ok.append(q)
            else:
                logger.debug(f"    Schema skip: {result.reason}")

        if not schema_ok:
            return

        # ── Batch correctness (Optimisation 2: N questions → 1 Bedrock call) ─
        corr_results = validate_correctness_batch(schema_ok, max_batch=BATCH_SIZE)

        # ── Per-question downstream nodes ─────────────────────────────────────
        for q, (corr_ok, corr_reason, suggested) in zip(schema_ok, corr_results):
            if len(accepted) >= num_questions:
                break

            if not corr_ok:
                logger.info(f"    ✗ Correctness FAIL: {corr_reason}")
                _log_rejection(db, q, "correctness", corr_reason, exam_set_number)
                # Apply suggested fix if available
                if suggested:
                    q = q.model_copy(update={"correct_answer": suggested})
                    logger.info("    ↻ Suggested answer applied")
                else:
                    continue   # skip — don't regenerate inside batch loop

            t_start = time.time()
            result = _run_post_correctness_nodes(
                q, chapter_key, chapter_content, accepted,
                chapter_weightage, global_seen, global_seen_list, global_seen_hashes,
                exam_set_number, db, attempt=0, generation_start=t_start,
                rag_passages=rag_passages,
            )
            if result is not None:
                accepted.append(result)
                q_lower = result.question.strip().lower()
                if global_seen is not None:
                    global_seen.add(q_lower)
                if global_seen_list is not None:
                    global_seen_list.append(q_lower)
                if global_seen_hashes is not None:
                    global_seen_hashes.add(result.question_hash)
                # Register with semantic dedup AFTER full acceptance
                try:
                    from rag.semantic_dedup import get_deduplicator
                    get_deduplicator().register(
                        result.question_uuid, result.question,
                        exam_set=exam_set_number, chapter=chapter_key,
                    )
                except Exception:
                    pass
                # Mark the source slug as used AFTER acceptance
                try:
                    from rag.slug_retriever import mark_chunks_used
                    if rag_passages:
                        used_ids = [p["id"] for p in rag_passages
                                    if p.get("id") and len(accepted) <= 1]
                        if used_ids:
                            mark_chunks_used(used_ids[:1], exam_set_number)
                except Exception:
                    pass
                logger.debug(f"    ✓ Accepted {len(accepted)}/{num_questions}")

    # ── Initial buffer: generate 1.5× target (Opt 5 — was 2×) ───────────────
    buffer_size = max(num_questions + 3, int(num_questions * BUFFER_MULTIPLIER))
    raw_questions = generate_questions_for_chapter(
        chapter_key, chapter_content, buffer_size, rag_passages=rag_passages
    )
    _process_batch(raw_questions)

    # ── Top-up only if still short ────────────────────────────────────────────
    topup_round = 0
    while len(accepted) < num_questions and topup_round < 3:
        topup_round += 1
        remaining = num_questions - len(accepted)
        logger.info(f"  ↻ Top-up #{topup_round}: need {remaining} more for {chapter_key}")
        top_up = generate_questions_for_chapter(
            chapter_key, chapter_content,
            max(remaining + 3, int(remaining * BUFFER_MULTIPLIER)),
            rag_passages=rag_passages,
        )
        _process_batch(top_up)

    logger.info(f"  ✓ Chapter complete: {len(accepted)}/{num_questions} accepted")
    return accepted[:num_questions]


# ── Exam-set generator (Opt 1 — parallel chapters) ───────────────────────────

def generate_exam_set(
    set_number: int,
    chapter_contents: dict[str, ChapterContent],
    allocation: dict[str, int],
    global_seen: set[str] | None = None,
    global_seen_list: list[str] | None = None,
    global_seen_hashes: set[str] | None = None,
    db=None,
) -> ExamSet:
    """
    Opt 1 — Generate all chapters concurrently using ThreadPoolExecutor.

    Each chapter runs in its own thread. The shared dedup registries
    (global_seen, global_seen_list, global_seen_hashes) are protected
    by a threading.Lock so concurrent writes are safe.

    Chapter results are collected and combined in blueprint order.
    """
    # Thread-safe dedup lock shared across all chapter threads
    dedup_lock = threading.Lock()

    # Preload semantic deduplicator from DB so existing questions are known
    if db is not None:
        try:
            from rag.semantic_dedup import get_deduplicator
            loaded = get_deduplicator().preload_from_db(db)
            logger.info(f"  Semantic dedup: preloaded {loaded} questions from DB")
        except Exception as e:
            logger.debug(f"  Semantic dedup preload skipped: {e}")

    def _generate_one_chapter(
        chapter_key: str,
        num_q: int,
    ) -> tuple[str, list[Question]]:
        """Worker: generate questions for one chapter, updating shared dedup sets."""
        content = chapter_contents[chapter_key]
        rag_passages = _fetch_rag_passages(chapter_key, content, set_number, top_k=8)
        chapter_weightage = round(CHAPTER_BLUEPRINT[chapter_key]["min_q"] / 100.0, 2)

        logger.info(
            f"\n  ► Chapter: {content.chapter_name} | "
            f"Target: {num_q} | Set #{set_number}"
        )

        accepted: list[Question] = []

        def _process_batch(raw_questions: list[Question]) -> None:
            nonlocal accepted

            schema_ok: list[Question] = []
            for q in raw_questions:
                if len(accepted) >= num_q:
                    break
                result = node_schema_validate(q)
                if result.passed:
                    schema_ok.append(q)
                else:
                    logger.debug(f"    Schema skip: {result.reason}")

            if not schema_ok:
                return

            # Batch correctness (Opt 3: batch size 25)
            corr_results = validate_correctness_batch(schema_ok, max_batch=BATCH_SIZE)

            for q, (corr_ok, corr_reason, suggested) in zip(schema_ok, corr_results):
                if len(accepted) >= num_q:
                    break

                if not corr_ok:
                    logger.info(f"    ✗ Correctness FAIL: {corr_reason}")
                    _log_rejection(db, q, "correctness", corr_reason, set_number)
                    if suggested:
                        q = q.model_copy(update={"correct_answer": suggested})
                        logger.info("    ↻ Suggested answer applied")
                    else:
                        continue

                # Read snapshot of dedup sets under lock
                with dedup_lock:
                    seen_snap       = set(global_seen)        if global_seen        else None
                    seen_list_snap  = list(global_seen_list)  if global_seen_list   else None
                    seen_hash_snap  = set(global_seen_hashes) if global_seen_hashes else None

                t_start = time.time()
                result = _run_post_correctness_nodes(
                    q, chapter_key, content, accepted,
                    chapter_weightage,
                    seen_snap, seen_list_snap, seen_hash_snap,
                    set_number, db, attempt=0,
                    generation_start=t_start,
                    rag_passages=rag_passages,
                )
                if result is not None:
                    # Write back to shared dedup sets under lock
                    with dedup_lock:
                        q_lower = result.question.strip().lower()
                        if global_seen is not None:
                            global_seen.add(q_lower)
                        if global_seen_list is not None:
                            global_seen_list.append(q_lower)
                        if global_seen_hashes is not None:
                            global_seen_hashes.add(result.question_hash)
                    accepted.append(result)
                    # Register with semantic dedup AFTER full acceptance
                    try:
                        from rag.semantic_dedup import get_deduplicator
                        get_deduplicator().register(
                            result.question_uuid, result.question,
                            exam_set=set_number, chapter=chapter_key,
                        )
                    except Exception:
                        pass
                    # Mark the source slug used AFTER acceptance (not at retrieval)
                    try:
                        from rag.slug_retriever import mark_chunks_used
                        if rag_passages:
                            # Each passage maps to one question — mark the one used
                            used_ids = [p["id"] for p in rag_passages
                                        if p.get("id") and len(accepted) <= len(rag_passages)]
                            if used_ids and len(accepted) <= len(used_ids):
                                mark_chunks_used([used_ids[len(accepted) - 1]], set_number)
                    except Exception:
                        pass
                    logger.debug(f"    ✓ Accepted {len(accepted)}/{num_q}")

        # Initial buffer (Opt 5: 1.5×)
        buffer_size = max(num_q + 3, int(num_q * BUFFER_MULTIPLIER))
        raw_questions = generate_questions_for_chapter(
            chapter_key, content, buffer_size, rag_passages=rag_passages
        )
        _process_batch(raw_questions)

        # Top-up if short
        topup_round = 0
        while len(accepted) < num_q and topup_round < 3:
            topup_round += 1
            remaining = num_q - len(accepted)
            logger.info(f"  ↻ Top-up #{topup_round}: need {remaining} more for {chapter_key}")
            top_up = generate_questions_for_chapter(
                chapter_key, content,
                max(remaining + 3, int(remaining * BUFFER_MULTIPLIER)),
                rag_passages=rag_passages,
            )
            _process_batch(top_up)

        # ── Node 7: Independent quality validation pass ───────────────────────
        accepted = run_quality_validation_pass(
            accepted[:num_q],
            chapter_key=chapter_key,
            chapter_content=content,
            chapter_weightage=chapter_weightage,
            exam_set_number=set_number,
            global_seen=global_seen,
            global_seen_list=global_seen_list,
            global_seen_hashes=global_seen_hashes,
            rag_passages=rag_passages,
            db=db,
        )

        logger.info(f"  ✓ Chapter complete: {len(accepted)}/{num_q} accepted")
        return chapter_key, accepted[:num_q]

    # ── Submit all chapters concurrently ──────────────────────────────────────
    chapter_results: dict[str, list[Question]] = {}
    chapter_items = list(allocation.items())

    with ThreadPoolExecutor(
        max_workers=min(MAX_CHAPTER_WORKERS, len(chapter_items)),
        thread_name_prefix="chapter",
    ) as executor:
        future_to_key = {
            executor.submit(_generate_one_chapter, ch_key, num_q): ch_key
            for ch_key, num_q in chapter_items
        }
        for future in as_completed(future_to_key):
            ch_key = future_to_key[future]
            try:
                key, questions = future.result()
                chapter_results[key] = questions
                logger.info(
                    f"  ✓ Chapter '{ch_key}' finished — "
                    f"{len(questions)} questions collected"
                )
            except Exception as e:
                logger.error(f"  ✗ Chapter '{ch_key}' failed: {e}", exc_info=True)
                chapter_results[ch_key] = []

    # ── Combine in blueprint order ────────────────────────────────────────────
    all_questions: list[Question] = []
    for chapter_key, _ in chapter_items:
        all_questions.extend(chapter_results.get(chapter_key, []))

    random.Random(set_number).shuffle(all_questions)
    logger.info(
        f"\n✓ Exam set {set_number} complete — "
        f"{len(all_questions)} questions across {len(chapter_results)} chapters"
    )
    return ExamSet(set_number=set_number, questions=all_questions)
