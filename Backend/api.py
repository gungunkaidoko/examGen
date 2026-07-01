"""
FastAPI layer for the CCC Exam Generation Platform.

Endpoints:
  GET  /api/exam-sets                      — list all generated sets
  GET  /api/exam-sets/{set_number}         — get all 100 questions in a set
  GET  /api/exam-sets/{set_number}/answer-key — get compact answer key
  GET  /api/exam-sets/{set_number}/pdf     — download set as PDF
  GET  /api/questions                      — query questions with filters
  GET  /api/stats                          — aggregate stats
  POST /api/generate                       — trigger pipeline generation
  GET  /api/generate/status                — poll generation job status

The API reads from PostgreSQL (question_bank table) OR falls back to
the JSON files in the output/ directory if the DB is unavailable.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Add project root to path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

# ── Resolve the venv Python that has all pipeline dependencies ────────────────
def _find_venv_python() -> str:
    """
    Find the Python interpreter that has langchain_aws, boto3, etc. installed.
    Probes sibling and local venv paths before falling back to sys.executable.
    """
    project_root = os.path.dirname(__file__)
    candidates = [
        # Sibling Backend/venv (most common layout)
        os.path.join(project_root, "venv", "bin", "python3"),
        os.path.join(project_root, "venv", "bin", "python"),
        # One level up — ndu_exam_project/Backend/venv
        os.path.join(os.path.dirname(project_root), "Backend", "venv", "bin", "python3"),
        os.path.join(os.path.dirname(project_root), "Backend", "venv", "bin", "python"),
    ]
    for c in candidates:
        if os.path.isfile(c) and os.access(c, os.X_OK):
            try:
                result = subprocess.run(
                    [c, "-c", "import langchain_aws"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    return c
            except Exception:
                pass
    return sys.executable  # last-resort: same interpreter that runs the API

PIPELINE_PYTHON = _find_venv_python()
logger.info(f"Pipeline interpreter: {PIPELINE_PYTHON}")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="CCC Exam Generation Platform",
    description="AI-powered MCQ bank for NIELIT CCC national exam",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── DB connection helper ──────────────────────────────────────────────────────
def _get_db():
    """Return a connected QuestionBankDB instance, or None on failure."""
    try:
        from database import QuestionBankDB
        db = QuestionBankDB()
        db.connect()
        return db
    except Exception as e:
        logger.warning(f"DB unavailable, falling back to JSON files: {e}")
        return None


# ── JSON fallback helpers ─────────────────────────────────────────────────────
def _list_json_sets() -> list[dict]:
    """Scan output/ for exam_set_NN.json files."""
    sets = []
    if not os.path.isdir(OUTPUT_DIR):
        return sets
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        if fname.startswith("exam_set_") and fname.endswith(".json"):
            path = os.path.join(OUTPUT_DIR, fname)
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                sets.append({
                    "set_number": data.get("exam_set"),
                    "total_questions": data.get("total_questions", 0),
                    "generated_at": data.get("generated_at"),
                    "source": "json",
                })
            except Exception:
                pass
    return sets


def _load_json_set(set_number: int) -> Optional[dict]:
    fname = f"exam_set_{set_number:02d}.json"
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_json_answer_key(set_number: int) -> Optional[dict]:
    fname = f"answer_key_{set_number:02d}.json"
    path = os.path.join(OUTPUT_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Generation job tracker ────────────────────────────────────────────────────
_generation_jobs: dict[str, dict] = {}


class GenerateRequest(BaseModel):
    sets: list[int] = list(range(1, 11))
    no_db: bool = False


# ── API Routes ────────────────────────────────────────────────────────────────

@app.get("/api/exam-sets")
def list_exam_sets():
    """Return metadata for all available exam sets."""
    db = _get_db()
    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute("""
                    SELECT exam_set, COUNT(*) as total, MAX(created_at) as generated_at
                    FROM question_bank
                    GROUP BY exam_set
                    ORDER BY exam_set
                """)
                rows = cur.fetchall()
            db.close()
            return {
                "exam_sets": [
                    {
                        "set_number": r[0],
                        "total_questions": r[1],
                        "generated_at": r[2].isoformat() if r[2] else None,
                        "source": "database",
                    }
                    for r in rows
                ]
            }
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    # Fallback to JSON files
    return {"exam_sets": _list_json_sets()}


@app.get("/api/exam-sets/{set_number}")
def get_exam_set(
    set_number: int,
    chapter: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    bloom_level: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
):
    """Return all questions for a set, with optional filters."""
    db = _get_db()
    questions = []

    if db:
        try:
            conditions = ["exam_set = %s"]
            params: list = [set_number]
            if chapter:
                conditions.append("chapter = %s")
                params.append(chapter)
            if difficulty:
                conditions.append("difficulty = %s")
                params.append(difficulty)
            if bloom_level:
                conditions.append("bloom_level = %s")
                params.append(bloom_level)
            if search:
                conditions.append("LOWER(question) LIKE %s")
                params.append(f"%{search.lower()}%")

            where = " AND ".join(conditions)
            with db._conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT question_uuid, question, option_a, option_b, option_c, option_d,
                           correct_answer, difficulty, bloom_level, chapter,
                           validation_status, tags, weightage, question_type,
                           model_name, prompt_version, generation_time, retry_count
                    FROM question_bank
                    WHERE {where}
                    ORDER BY id
                    """,
                    params,
                )
                rows = cur.fetchall()
            db.close()
            questions = [
                {
                    "question_uuid": str(r[0]),
                    "question": r[1],
                    "options": [r[2], r[3], r[4], r[5]],
                    "correct_answer": r[6],
                    "difficulty": r[7],
                    "bloom_level": r[8],
                    "chapter": r[9],
                    "validation_status": r[10],
                    "tags": r[11] or [],
                    "weightage": float(r[12]) if r[12] is not None else 1.0,
                    "question_type": r[13],
                    "model_name": r[14],
                    "prompt_version": r[15],
                    "generation_time": float(r[16]) if r[16] is not None else 0.0,
                    "retry_count": r[17],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            try:
                db.close()
            except Exception:
                pass
            questions = []

    if not questions:
        # Fallback to JSON
        data = _load_json_set(set_number)
        if not data:
            raise HTTPException(status_code=404, detail=f"Exam set {set_number} not found")
        questions = data.get("questions", [])
        # Apply filters on JSON data
        if chapter:
            questions = [q for q in questions if q.get("chapter") == chapter]
        if difficulty:
            questions = [q for q in questions if q.get("difficulty") == difficulty]
        if bloom_level:
            questions = [q for q in questions if q.get("bloom_level") == bloom_level]
        if search:
            s = search.lower()
            questions = [q for q in questions if s in q.get("question", "").lower()]

    return {
        "set_number": set_number,
        "total_questions": len(questions),
        "questions": questions,
    }


@app.get("/api/exam-sets/{set_number}/answer-key")
def get_answer_key(set_number: int):
    """Return compact answer key for a set."""
    db = _get_db()
    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT ROW_NUMBER() OVER (ORDER BY id) as q_no,
                           question, correct_answer, difficulty, chapter
                    FROM question_bank
                    WHERE exam_set = %s
                    ORDER BY id
                    """,
                    (set_number,),
                )
                rows = cur.fetchall()
            db.close()
            if rows:
                return {
                    "exam_set": set_number,
                    "answer_key": [
                        {
                            "q_no": r[0],
                            "question": r[1],
                            "correct_answer": r[2],
                            "difficulty": r[3],
                            "chapter": r[4],
                        }
                        for r in rows
                    ],
                }
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    # Fallback to JSON file
    data = _load_json_answer_key(set_number)
    if not data:
        raise HTTPException(status_code=404, detail=f"Answer key for set {set_number} not found")
    return data


@app.get("/api/exam-sets/{set_number}/pdf")
def download_pdf(set_number: int):
    """Generate and stream a PDF of the exam set."""
    # Fetch questions
    db = _get_db()
    questions = []
    generated_at = ""

    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT question, option_a, option_b, option_c, option_d,
                           correct_answer, difficulty, bloom_level, chapter, MAX(created_at)
                    FROM question_bank
                    WHERE exam_set = %s
                    GROUP BY id, question, option_a, option_b, option_c, option_d,
                             correct_answer, difficulty, bloom_level, chapter
                    ORDER BY id
                    """,
                    (set_number,),
                )
                rows = cur.fetchall()
                if rows:
                    generated_at = str(rows[-1][-1])
            db.close()
            questions = [
                {
                    "question": r[0],
                    "options": [r[1], r[2], r[3], r[4]],
                    "correct_answer": r[5],
                    "difficulty": r[6],
                    "bloom_level": r[7],
                    "chapter": r[8],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    if not questions:
        data = _load_json_set(set_number)
        if not data:
            raise HTTPException(status_code=404, detail=f"Exam set {set_number} not found")
        questions = data.get("questions", [])
        generated_at = data.get("generated_at", "")

    # Build PDF with ReportLab
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm,
            leftMargin=2 * cm,
            topMargin=2 * cm,
            bottomMargin=2 * cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontSize=16,
            spaceAfter=6,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#1e3a5f"),
        )
        subtitle_style = ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#6b7280"),
        )
        q_style = ParagraphStyle(
            "Question",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            spaceAfter=4,
            textColor=colors.HexColor("#111827"),
        )
        opt_style = ParagraphStyle(
            "Option",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            leftIndent=20,
            textColor=colors.HexColor("#374151"),
        )
        meta_style = ParagraphStyle(
            "Meta",
            parent=styles["Normal"],
            fontSize=8,
            textColor=colors.HexColor("#6b7280"),
            spaceAfter=12,
            leftIndent=20,
        )

        story = [
            Paragraph("NIELIT — CCC National Exam", title_style),
            Paragraph(
                f"Exam Set {set_number:02d} &nbsp;|&nbsp; 100 Questions &nbsp;|&nbsp; Generated: {generated_at[:10]}",
                subtitle_style,
            ),
        ]

        option_labels = ["A", "B", "C", "D"]
        for idx, q in enumerate(questions, 1):
            story.append(
                Paragraph(
                    f"<b>Q{idx}.</b> {q['question']}",
                    q_style,
                )
            )
            for label, opt in zip(option_labels, q.get("options", [])):
                story.append(Paragraph(f"({label}) {opt}", opt_style))
            story.append(
                Paragraph(
                    f"Chapter: {q.get('chapter', '')} &nbsp;|&nbsp; "
                    f"Difficulty: {q.get('difficulty', '')} &nbsp;|&nbsp; "
                    f"Bloom: {q.get('bloom_level', '')}",
                    meta_style,
                )
            )

        doc.build(story)
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="exam_set_{set_number:02d}.pdf"'
            },
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")


@app.get("/api/questions")
def list_questions(
    chapter: Optional[str] = Query(None),
    difficulty: Optional[str] = Query(None),
    bloom_level: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    exam_set: Optional[int] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """Query questions across all sets with optional filters and pagination."""
    db = _get_db()
    if db:
        try:
            conditions = []
            params: list = []
            if exam_set:
                conditions.append("exam_set = %s")
                params.append(exam_set)
            if chapter:
                conditions.append("chapter = %s")
                params.append(chapter)
            if difficulty:
                conditions.append("difficulty = %s")
                params.append(difficulty)
            if bloom_level:
                conditions.append("bloom_level = %s")
                params.append(bloom_level)
            if search:
                conditions.append("LOWER(question) LIKE %s")
                params.append(f"%{search.lower()}%")

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            offset = (page - 1) * page_size

            with db._conn.cursor() as cur:
                cur.execute(
                    f"SELECT COUNT(*) FROM question_bank {where}", params
                )
                total = cur.fetchone()[0]
                cur.execute(
                    f"""
                    SELECT question_uuid, exam_set, question, option_a, option_b, option_c, option_d,
                           correct_answer, difficulty, bloom_level, chapter, tags, weightage
                    FROM question_bank {where}
                    ORDER BY exam_set, id
                    LIMIT %s OFFSET %s
                    """,
                    params + [page_size, offset],
                )
                rows = cur.fetchall()
            db.close()

            return {
                "total": total,
                "page": page,
                "page_size": page_size,
                "questions": [
                    {
                        "question_uuid": str(r[0]),
                        "exam_set": r[1],
                        "question": r[2],
                        "options": [r[3], r[4], r[5], r[6]],
                        "correct_answer": r[7],
                        "difficulty": r[8],
                        "bloom_level": r[9],
                        "chapter": r[10],
                        "tags": r[11] or [],
                        "weightage": float(r[12]) if r[12] is not None else 1.0,
                    }
                    for r in rows
                ],
            }
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    raise HTTPException(status_code=503, detail="Database unavailable and no fallback for full question listing")


@app.get("/api/stats")
def get_stats():
    """Return aggregate statistics across all exam sets."""
    db = _get_db()
    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM question_bank")
                total = cur.fetchone()[0]
                cur.execute("SELECT COUNT(DISTINCT exam_set) FROM question_bank")
                total_sets = cur.fetchone()[0]
                cur.execute(
                    "SELECT difficulty, COUNT(*) FROM question_bank GROUP BY difficulty ORDER BY difficulty"
                )
                by_difficulty = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute(
                    "SELECT bloom_level, COUNT(*) FROM question_bank GROUP BY bloom_level ORDER BY bloom_level"
                )
                by_bloom = {r[0]: r[1] for r in cur.fetchall()}
                cur.execute(
                    "SELECT chapter, COUNT(*) FROM question_bank GROUP BY chapter ORDER BY chapter"
                )
                by_chapter = {r[0]: r[1] for r in cur.fetchall()}
            db.close()
            return {
                "total_questions": total,
                "total_sets": total_sets,
                "by_difficulty": by_difficulty,
                "by_bloom_level": by_bloom,
                "by_chapter": by_chapter,
            }
        except Exception as e:
            logger.warning(f"DB stats failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    # Fallback: aggregate from JSON files
    total = 0
    total_sets = 0
    by_difficulty: dict = {}
    by_bloom: dict = {}
    by_chapter: dict = {}
    for fname in sorted(os.listdir(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else []):
        if fname.startswith("exam_set_") and fname.endswith(".json"):
            try:
                with open(os.path.join(OUTPUT_DIR, fname), encoding="utf-8") as f:
                    data = json.load(f)
                total_sets += 1
                for q in data.get("questions", []):
                    total += 1
                    d = q.get("difficulty", "unknown")
                    by_difficulty[d] = by_difficulty.get(d, 0) + 1
                    b = q.get("bloom_level", "unknown")
                    by_bloom[b] = by_bloom.get(b, 0) + 1
                    c = q.get("chapter", "unknown")
                    by_chapter[c] = by_chapter.get(c, 0) + 1
            except Exception:
                pass
    return {
        "total_questions": total,
        "total_sets": total_sets,
        "by_difficulty": by_difficulty,
        "by_bloom_level": by_bloom,
        "by_chapter": by_chapter,
    }


@app.post("/api/generate")
def trigger_generation(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Trigger the Python pipeline to generate exam sets.
    Returns a job_id to poll status via GET /api/generate/status?job_id=...
    """
    job_id = f"job_{int(time.time())}"
    _generation_jobs[job_id] = {
        "status": "queued",
        "sets": request.sets,
        "started_at": time.time(),
        "message": f"Queued generation of sets {request.sets}",
    }

    def run_pipeline():
        _generation_jobs[job_id]["status"] = "running"
        _generation_jobs[job_id]["message"] = "Pipeline started..."
        _generation_jobs[job_id]["log_lines"] = []
        try:
            cmd = [PIPELINE_PYTHON, os.path.join(os.path.dirname(__file__), "main.py")]
            cmd += ["--sets"] + [str(s) for s in request.sets]
            if request.no_db:
                cmd.append("--no-db")

            process = subprocess.Popen(
                cmd,
                cwd=os.path.dirname(__file__),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                print(line, flush=True)   # also goes to terminal
                _generation_jobs[job_id]["message"] = line
                # Keep last 120 lines so frontend can render a live log
                _generation_jobs[job_id]["log_lines"].append(line)
                if len(_generation_jobs[job_id]["log_lines"]) > 120:
                    _generation_jobs[job_id]["log_lines"].pop(0)

            process.wait()
            if process.returncode == 0:
                _generation_jobs[job_id]["status"] = "completed"
                _generation_jobs[job_id]["message"] = "Generation completed successfully"
            else:
                _generation_jobs[job_id]["status"] = "failed"
                _generation_jobs[job_id]["message"] = f"Pipeline exited with code {process.returncode}"
        except Exception as e:
            _generation_jobs[job_id]["status"] = "failed"
            _generation_jobs[job_id]["message"] = str(e)

    background_tasks.add_task(run_pipeline)
    return {"job_id": job_id, "status": "queued", "sets": request.sets}


@app.get("/api/generate/status")
def generation_status(job_id: str = Query(...)):
    """Poll the status of a generation job."""
    job = _generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/filters")
def get_filter_options():
    """Return unique values for all filter dropdowns."""
    db = _get_db()
    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute("SELECT DISTINCT chapter FROM question_bank ORDER BY chapter")
                chapters = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT DISTINCT difficulty FROM question_bank ORDER BY difficulty")
                difficulties = [r[0] for r in cur.fetchall()]
                cur.execute("SELECT DISTINCT bloom_level FROM question_bank ORDER BY bloom_level")
                blooms = [r[0] for r in cur.fetchall()]
            db.close()
            return {"chapters": chapters, "difficulties": difficulties, "bloom_levels": blooms}
        except Exception as e:
            logger.warning(f"DB query failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    # Fallback from JSON
    chapters_set: set = set()
    difficulties_set: set = set()
    blooms_set: set = set()
    for fname in sorted(os.listdir(OUTPUT_DIR) if os.path.isdir(OUTPUT_DIR) else []):
        if fname.startswith("exam_set_") and fname.endswith(".json"):
            try:
                with open(os.path.join(OUTPUT_DIR, fname), encoding="utf-8") as f:
                    data = json.load(f)
                for q in data.get("questions", []):
                    if q.get("chapter"):
                        chapters_set.add(q["chapter"])
                    if q.get("difficulty"):
                        difficulties_set.add(q["difficulty"])
                    if q.get("bloom_level"):
                        blooms_set.add(q["bloom_level"])
            except Exception:
                pass
    return {
        "chapters": sorted(chapters_set),
        "difficulties": sorted(difficulties_set),
        "bloom_levels": sorted(blooms_set),
    }


@app.get("/api/exam-sets/{set_number}/metrics")
def get_metrics(set_number: int):
    """
    Compute quality metrics for one exam set entirely from stored data.
    No LLM calls — runs in < 1 second.

    Metrics returned:
      total_questions      int
      question_accuracy    float  — % approved (validation_status == approved)
      answer_accuracy      float  — % where correct_answer is in options
      rejection_rate       float  — retried questions / total generated attempts
      semantic_uniqueness  float  — % question pairs with Jaccard similarity < threshold
      concept_overlap      float  — % question pairs sharing ≥80% tags
      difficulty_balance   float  — mean alignment vs blueprint targets per chapter
      bloom_accuracy       float  — % questions with bloom consistent with difficulty
      chapter_coverage     dict   — actual vs target question counts per chapter
      difficulty_dist      dict   — easy / medium / hard counts
      bloom_dist           dict   — distribution across bloom levels
      avg_generation_time  float  — mean seconds per question
      avg_retry_count      float  — mean retries per question
    """
    import math
    from collections import Counter, defaultdict
    from itertools import combinations

    # ── 1. Load questions ────────────────────────────────────────────────────
    questions = []
    db = _get_db()
    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT question, option_a, option_b, option_c, option_d,
                           correct_answer, difficulty, bloom_level, chapter,
                           validation_status, tags, retry_count, generation_time
                    FROM question_bank
                    WHERE exam_set = %s
                    ORDER BY id
                    """,
                    (set_number,),
                )
                rows = cur.fetchall()
            db.close()
            questions = [
                {
                    "question":          r[0],
                    "options":           [r[1], r[2], r[3], r[4]],
                    "correct_answer":    r[5],
                    "difficulty":        r[6],
                    "bloom_level":       r[7],
                    "chapter":           r[8],
                    "validation_status": r[9],
                    "tags":              r[10] or [],
                    "retry_count":       r[11] or 0,
                    "generation_time":   float(r[12]) if r[12] else 0.0,
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"DB metrics query failed: {e}")
            try:
                db.close()
            except Exception:
                pass

    if not questions:
        data = _load_json_set(set_number)
        if not data:
            raise HTTPException(status_code=404, detail=f"Exam set {set_number} not found")
        for q in data.get("questions", []):
            questions.append({
                "question":          q.get("question", ""),
                "options":           q.get("options", []),
                "correct_answer":    q.get("correct_answer", ""),
                "difficulty":        q.get("difficulty", ""),
                "bloom_level":       q.get("bloom_level", ""),
                "chapter":           q.get("chapter", ""),
                "validation_status": q.get("validation_status", "approved"),
                "tags":              q.get("tags", []),
                "retry_count":       q.get("retry_count", 0),
                "generation_time":   float(q.get("generation_time", 0)),
            })

    n = len(questions)
    if n == 0:
        raise HTTPException(status_code=404, detail=f"No questions found for set {set_number}")

    # ── 2. Question accuracy — % with approved status ────────────────────────
    approved = sum(1 for q in questions if q["validation_status"] == "approved")
    question_accuracy = round(approved / n * 100, 1)

    # ── 3. Answer accuracy — correct_answer present in options ───────────────
    answer_ok = sum(1 for q in questions if q["correct_answer"] in q["options"])
    answer_accuracy = round(answer_ok / n * 100, 1)

    # ── 4. Rejection / retry rate ────────────────────────────────────────────
    total_retries = sum(q["retry_count"] for q in questions)
    total_attempts = n + total_retries
    rejection_rate = round(total_retries / total_attempts * 100, 1) if total_attempts > 0 else 0.0

    # ── 5. Semantic uniqueness — Jaccard similarity on question word-sets ─────
    JACCARD_THRESHOLD = 0.55   # pairs with > 55 % word overlap are near-duplicates
    near_dup_pairs = 0
    total_pairs = n * (n - 1) // 2

    def _jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    # O(n²) — fine for n=100 (4950 pairs, < 5 ms)
    texts = [q["question"] for q in questions]
    for i in range(n):
        for j in range(i + 1, n):
            if _jaccard(texts[i], texts[j]) > JACCARD_THRESHOLD:
                near_dup_pairs += 1

    semantic_uniqueness = round((1 - near_dup_pairs / total_pairs) * 100, 1) if total_pairs > 0 else 100.0

    # ── 6. Concept overlap — shared tags ─────────────────────────────────────
    TAG_OVERLAP_THRESHOLD = 0.8
    concept_overlap_pairs = 0

    def _tag_overlap(t1: list, t2: list) -> float:
        s1, s2 = set(t1), set(t2)
        if not s1 or not s2:
            return 0.0
        return len(s1 & s2) / min(len(s1), len(s2))

    for i in range(n):
        for j in range(i + 1, n):
            if _tag_overlap(questions[i]["tags"], questions[j]["tags"]) >= TAG_OVERLAP_THRESHOLD:
                concept_overlap_pairs += 1

    concept_overlap = round(concept_overlap_pairs / total_pairs * 100, 1) if total_pairs > 0 else 0.0

    # ── 7. Difficulty balance — actual vs blueprint target per chapter ─────────
    # Map display chapter names back to blueprint keys
    CHAPTER_DISPLAY_MAP = {
        "Introduction to Computer":                         "chapter_01_introduction_to_computer",
        "Introduction to Computer System":                  "chapter_01_introduction_to_computer",
        "Operating System":                                 "chapter_02_operating_system",
        "Introduction to Operating System":                 "chapter_02_operating_system",
        "Word Processing":                                  "chapter_03_word_processing",
        "Spreadsheet":                                      "chapter_04_spreadsheet",
        "Presentation":                                     "chapter_05_presentation",
        "Introduction to Internet and WWW":                 "chapter_06_internet_www",
        "Email, Social Networking and e-Governance Services": "chapter_07_email_social_egov",
        "Digital Financial Tools":                          "chapter_08_digital_financial_tools",
        "Digital Financial Tools and Applications":         "chapter_08_digital_financial_tools",
        # All known variants of Ch9 name stored by the pipeline
        "Overview of Futureskills and Cyber Security":      "chapter_09_futureskills_cybersecurity",
        "Overview of Future Skills and Cyber Security":     "chapter_09_futureskills_cybersecurity",
        "FutureSkills and Cybersecurity":                   "chapter_09_futureskills_cybersecurity",
        "Overview of FutureSkills and Cyber Security":      "chapter_09_futureskills_cybersecurity",
        "Overview of Futureskills & Cyber Security":        "chapter_09_futureskills_cybersecurity",
    }

    CHAPTER_BLUEPRINT_LOCAL = {
        "chapter_01_introduction_to_computer":      {"min_q": 8,  "max_q": 10, "split": {"easy": 0.50, "medium": 0.35, "hard": 0.15}},
        "chapter_02_operating_system":              {"min_q": 6,  "max_q": 8,  "split": {"easy": 0.50, "medium": 0.35, "hard": 0.15}},
        "chapter_03_word_processing":               {"min_q": 15, "max_q": 18, "split": {"easy": 0.40, "medium": 0.40, "hard": 0.20}},
        "chapter_04_spreadsheet":                   {"min_q": 12, "max_q": 15, "split": {"easy": 0.40, "medium": 0.40, "hard": 0.20}},
        "chapter_05_presentation":                  {"min_q": 5,  "max_q": 7,  "split": {"easy": 0.50, "medium": 0.35, "hard": 0.15}},
        "chapter_06_internet_www":                  {"min_q": 10, "max_q": 12, "split": {"easy": 0.40, "medium": 0.40, "hard": 0.20}},
        "chapter_07_email_social_egov":             {"min_q": 10, "max_q": 12, "split": {"easy": 0.40, "medium": 0.40, "hard": 0.20}},
        "chapter_08_digital_financial_tools":       {"min_q": 10, "max_q": 13, "split": {"easy": 0.35, "medium": 0.45, "hard": 0.20}},
        "chapter_09_futureskills_cybersecurity":    {"min_q": 12, "max_q": 15, "split": {"easy": 0.35, "medium": 0.45, "hard": 0.20}},
    }

    chapter_qs: dict = defaultdict(list)
    for q in questions:
        ch = q["chapter"]
        key = CHAPTER_DISPLAY_MAP.get(ch)
        if not key:
            # Fuzzy match: normalise both sides (lowercase, strip spaces/punctuation)
            import re as _re
            def _normalise(s):
                return _re.sub(r"[^a-z0-9]", "", s.lower())
            ch_norm = _normalise(ch)
            for display, bkey in CHAPTER_DISPLAY_MAP.items():
                if _normalise(display) == ch_norm:
                    key = bkey
                    break
            if not key:
                # substring fallback
                ch_lower = ch.lower()
                for display, bkey in CHAPTER_DISPLAY_MAP.items():
                    dl = display.lower()
                    if dl in ch_lower or ch_lower in dl:
                        key = bkey
                        break
            if not key:
                key = ch  # unmapped — keep as-is so it still appears in coverage
            if not key:
                key = ch  # unmapped — keep as-is
        chapter_qs[key].append(q["difficulty"])

    balance_scores = []
    chapter_coverage = {}
    for bkey, bp in CHAPTER_BLUEPRINT_LOCAL.items():
        qs = chapter_qs.get(bkey, [])
        actual_count = len(qs)
        in_range = bp["min_q"] <= actual_count <= bp["max_q"]
        chapter_coverage[bkey] = {
            "actual": actual_count,
            "min": bp["min_q"],
            "max": bp["max_q"],
            "in_range": in_range,
        }
        if actual_count == 0:
            balance_scores.append(0.0)
            continue
        actual_dist = Counter(qs)
        for level, target_pct in bp["split"].items():
            actual_pct = actual_dist.get(level, 0) / actual_count
            balance_scores.append(1.0 - abs(actual_pct - target_pct))

    difficulty_balance = round(sum(balance_scores) / len(balance_scores) * 100, 1) if balance_scores else 0.0

    # ── 8. Bloom accuracy — bloom level consistent with difficulty ────────────
    VALID_BLOOM = {
        "easy":   {"remember", "understand"},
        "medium": {"understand", "apply"},
        "hard":   {"apply", "analyze"},
    }
    bloom_ok = sum(
        1 for q in questions
        if q["bloom_level"] in VALID_BLOOM.get(q["difficulty"], set())
    )
    bloom_accuracy = round(bloom_ok / n * 100, 1)

    # ── 9. Distributions ─────────────────────────────────────────────────────
    diff_dist  = dict(Counter(q["difficulty"]  for q in questions))
    bloom_dist = dict(Counter(q["bloom_level"] for q in questions))

    # ── 10. Generation performance ───────────────────────────────────────────
    gen_times = [q["generation_time"] for q in questions if q["generation_time"] > 0]
    avg_gen_time   = round(sum(gen_times) / len(gen_times), 3) if gen_times else 0.0
    avg_retry      = round(sum(q["retry_count"] for q in questions) / n, 2)

    return {
        "set_number":          set_number,
        "total_questions":     n,
        "question_accuracy":   question_accuracy,
        "answer_accuracy":     answer_accuracy,
        "rejection_rate":      rejection_rate,
        "semantic_uniqueness": semantic_uniqueness,
        "concept_overlap":     concept_overlap,
        "difficulty_balance":  difficulty_balance,
        "bloom_accuracy":      bloom_accuracy,
        "chapter_coverage":    chapter_coverage,
        "difficulty_dist":     diff_dist,
        "bloom_dist":          bloom_dist,
        "avg_generation_time": avg_gen_time,
        "avg_retry_count":     avg_retry,
    }


@app.get("/api/health")
def health_check():
    """Quick health probe."""
    db = _get_db()
    db_ok = False
    if db:
        try:
            with db._conn.cursor() as cur:
                cur.execute("SELECT 1")
            db_ok = True
            db.close()
        except Exception:
            pass
    return {"status": "ok", "database": "connected" if db_ok else "unavailable"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
