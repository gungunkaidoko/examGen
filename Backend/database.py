"""
PostgreSQL Question Bank
------------------------
Manages schema creation and question insertion.
Each approved question is stored with full metadata including UUID,
Bloom's level, tags, and weightage.
"""

import logging
from typing import Optional
import psycopg2
from psycopg2.extras import execute_values

from config import DB_CONFIG
from models import Question, ExamSet

logger = logging.getLogger(__name__)


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS question_bank (
    id                SERIAL PRIMARY KEY,
    question_uuid     UUID NOT NULL,
    question_hash     CHAR(64) NOT NULL,
    exam_set          INTEGER NOT NULL,
    question          TEXT NOT NULL,
    option_a          TEXT NOT NULL,
    option_b          TEXT NOT NULL,
    option_c          TEXT NOT NULL,
    option_d          TEXT NOT NULL,
    correct_answer    TEXT NOT NULL,
    difficulty        VARCHAR(10) NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    bloom_level       VARCHAR(15) NOT NULL DEFAULT 'remember'
                          CHECK (bloom_level IN ('remember', 'understand', 'apply', 'analyze')),
    chapter           TEXT NOT NULL,
    validation_status VARCHAR(20) NOT NULL DEFAULT 'approved',
    tags              TEXT[] DEFAULT '{}',
    weightage         NUMERIC(4,2) DEFAULT 1.0,
    question_type     TEXT NOT NULL DEFAULT 'standard',
    model_name        TEXT NOT NULL DEFAULT '',
    prompt_version    TEXT NOT NULL DEFAULT 'v1',
    generation_time   NUMERIC(10,4) DEFAULT 0.0,
    retry_count       SMALLINT DEFAULT 0,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_qb_exam_set    ON question_bank (exam_set);
CREATE INDEX IF NOT EXISTS idx_qb_chapter     ON question_bank (chapter);
CREATE INDEX IF NOT EXISTS idx_qb_difficulty  ON question_bank (difficulty);
CREATE INDEX IF NOT EXISTS idx_qb_bloom       ON question_bank (bloom_level);
CREATE UNIQUE INDEX IF NOT EXISTS idx_qb_uuid ON question_bank (question_uuid);
CREATE UNIQUE INDEX IF NOT EXISTS idx_qb_hash ON question_bank (question_hash);

-- Validation logs: every rejection is stored for prompt analysis
CREATE TABLE IF NOT EXISTS validation_logs (
    id                SERIAL PRIMARY KEY,
    question_uuid     UUID,
    exam_set          INTEGER,
    chapter           TEXT,
    question_text     TEXT NOT NULL,
    correct_answer    TEXT,
    difficulty        VARCHAR(10),
    validator_failed  VARCHAR(60) NOT NULL,
    reason            TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vl_validator  ON validation_logs (validator_failed);
CREATE INDEX IF NOT EXISTS idx_vl_chapter    ON validation_logs (chapter);
CREATE INDEX IF NOT EXISTS idx_vl_exam_set   ON validation_logs (exam_set);
CREATE INDEX IF NOT EXISTS idx_vl_created_at ON validation_logs (created_at);
"""

MIGRATE_TABLE_SQL = """
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='question_uuid') THEN
        ALTER TABLE question_bank ADD COLUMN question_uuid UUID;
        UPDATE question_bank SET question_uuid = gen_random_uuid() WHERE question_uuid IS NULL;
        ALTER TABLE question_bank ALTER COLUMN question_uuid SET NOT NULL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='question_hash') THEN
        ALTER TABLE question_bank ADD COLUMN question_hash CHAR(64);
        UPDATE question_bank
           SET question_hash = encode(sha256(lower(trim(question))::bytea), 'hex')
         WHERE question_hash IS NULL;
        ALTER TABLE question_bank ALTER COLUMN question_hash SET NOT NULL;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='bloom_level') THEN
        ALTER TABLE question_bank ADD COLUMN bloom_level VARCHAR(15) NOT NULL DEFAULT 'remember'
            CHECK (bloom_level IN ('remember', 'understand', 'apply', 'analyze'));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='tags') THEN
        ALTER TABLE question_bank ADD COLUMN tags TEXT[] DEFAULT '{}';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='weightage') THEN
        ALTER TABLE question_bank ADD COLUMN weightage NUMERIC(4,2) DEFAULT 1.0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='question_type') THEN
        ALTER TABLE question_bank ADD COLUMN question_type TEXT NOT NULL DEFAULT 'standard';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='model_name') THEN
        ALTER TABLE question_bank ADD COLUMN model_name TEXT NOT NULL DEFAULT '';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='prompt_version') THEN
        ALTER TABLE question_bank ADD COLUMN prompt_version TEXT NOT NULL DEFAULT 'v1';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='generation_time') THEN
        ALTER TABLE question_bank ADD COLUMN generation_time NUMERIC(10,4) DEFAULT 0.0;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns
                   WHERE table_name='question_bank' AND column_name='retry_count') THEN
        ALTER TABLE question_bank ADD COLUMN retry_count SMALLINT DEFAULT 0;
    END IF;
END$$;
"""


class QuestionBankDB:
    """PostgreSQL-backed question bank."""

    def __init__(self):
        self._conn: Optional[psycopg2.extensions.connection] = None

    def connect(self) -> None:
        """Establish database connection and ensure schema exists."""
        self._conn = psycopg2.connect(**DB_CONFIG)
        self._conn.autocommit = False
        self._ensure_schema()
        logger.info("✓ Connected to PostgreSQL question bank")

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()
            logger.info("Database connection closed")

    def _ensure_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(CREATE_TABLE_SQL)
            cur.execute(MIGRATE_TABLE_SQL)
        self._conn.commit()

    # ── Write ────────────────────────────────────────────────────────────────

    def save_exam_set(self, exam_set: ExamSet) -> int:
        """Bulk-insert all questions for an exam set. Returns row count."""
        rows = []
        for q in exam_set.questions:
            d = q.to_db_dict()
            rows.append((
                d["question_uuid"],
                d["question_hash"],
                exam_set.set_number,
                d["question"],
                d["option_a"],
                d["option_b"],
                d["option_c"],
                d["option_d"],
                d["correct_answer"],
                d["difficulty"],
                d["bloom_level"],
                d["chapter"],
                d["validation_status"],
                d["tags"],
                d["weightage"],
                d["question_type"],
                d["model_name"],
                d["prompt_version"],
                d["generation_time"],
                d["retry_count"],
            ))

        sql = """
            INSERT INTO question_bank
                (question_uuid, question_hash, exam_set, question,
                 option_a, option_b, option_c, option_d,
                 correct_answer, difficulty, bloom_level, chapter,
                 validation_status, tags, weightage,
                 question_type, model_name, prompt_version, generation_time, retry_count)
            VALUES %s
            ON CONFLICT (question_uuid) DO NOTHING
        """
        with self._conn.cursor() as cur:
            execute_values(cur, sql, rows)
        self._conn.commit()
        logger.info(f"  ✓ Saved {len(rows)} questions for exam set {exam_set.set_number}")
        return len(rows)

    # ── Read ─────────────────────────────────────────────────────────────────

    def get_exam_set(self, set_number: int) -> list[dict]:
        """Retrieve all questions for a given exam set number."""
        sql = """
            SELECT question, option_a, option_b, option_c, option_d,
                   correct_answer, difficulty, bloom_level, chapter,
                   validation_status, tags, weightage, question_uuid
            FROM question_bank
            WHERE exam_set = %s
            ORDER BY id
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (set_number,))
            rows = cur.fetchall()

        return [
            {
                "question": r[0],
                "options": [r[1], r[2], r[3], r[4]],
                "correct_answer": r[5],
                "difficulty": r[6],
                "bloom_level": r[7],
                "chapter": r[8],
                "validation_status": r[9],
                "tags": r[10] or [],
                "weightage": float(r[11]),
                "question_uuid": str(r[12]),
            }
            for r in rows
        ]

    # ── Dedup helpers ────────────────────────────────────────────────────────

    def question_exists(self, question_text: str, exam_set: int) -> bool:
        """Check for duplicate question within a set."""
        sql = "SELECT 1 FROM question_bank WHERE LOWER(question) = LOWER(%s) AND exam_set = %s LIMIT 1"
        with self._conn.cursor() as cur:
            cur.execute(sql, (question_text, exam_set))
            return cur.fetchone() is not None

    def question_exists_global(self, question_text: str) -> bool:
        """Check for duplicate question across ALL exam sets."""
        sql = "SELECT 1 FROM question_bank WHERE LOWER(question) = LOWER(%s) LIMIT 1"
        with self._conn.cursor() as cur:
            cur.execute(sql, (question_text,))
            return cur.fetchone() is not None

    def hash_exists_global(self, question_hash: str) -> bool:
        """Fast O(1) exact duplicate check using SHA-256 hash."""
        sql = "SELECT 1 FROM question_bank WHERE question_hash = %s LIMIT 1"
        with self._conn.cursor() as cur:
            cur.execute(sql, (question_hash,))
            return cur.fetchone() is not None

    def load_all_questions_lowercase(self) -> set[str]:
        """Load all existing question texts (lowercased) for in-memory dedup registry."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT LOWER(question) FROM question_bank")
            return {row[0] for row in cur.fetchall()}

    def load_all_question_hashes(self) -> set[str]:
        """Load all existing question hashes for fast in-memory exact dedup."""
        with self._conn.cursor() as cur:
            cur.execute("SELECT question_hash FROM question_bank")
            return {row[0] for row in cur.fetchall()}

    # ── Validation Logs ──────────────────────────────────────────────────────

    def log_validation_failure(
        self,
        question_text: str,
        validator_failed: str,
        reason: str,
        exam_set: int | None = None,
        chapter: str | None = None,
        correct_answer: str | None = None,
        difficulty: str | None = None,
        question_uuid: str | None = None,
    ) -> None:
        """
        Persist a rejected question to validation_logs so failures
        can be analysed and prompts improved later.

        Args:
            question_text:    The raw question text that failed.
            validator_failed: Short name of the failing validator
                              (e.g. 'correctness', 'duplicate', 'formatting').
            reason:           Human-readable failure reason.
            exam_set:         Exam set number being generated (optional).
            chapter:          Chapter name (optional).
            correct_answer:   Marked correct answer (optional).
            difficulty:       Difficulty level (optional).
            question_uuid:    UUID if already assigned (optional).
        """
        sql = """
            INSERT INTO validation_logs
                (question_uuid, exam_set, chapter, question_text,
                 correct_answer, difficulty, validator_failed, reason)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, (
                    question_uuid,
                    exam_set,
                    chapter,
                    question_text,
                    correct_answer,
                    difficulty,
                    validator_failed[:60],   # column is VARCHAR(60)
                    reason,
                ))
            self._conn.commit()
        except Exception as e:
            logger.warning(f"Could not write validation log: {e}")
            self._conn.rollback()

    def get_validation_logs(
        self,
        validator_failed: str | None = None,
        chapter: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """
        Retrieve validation failure logs, optionally filtered.

        Args:
            validator_failed: Filter by validator name (e.g. 'correctness').
            chapter:          Filter by chapter name.
            limit:            Max rows to return.

        Returns:
            List of dicts with all log fields.
        """
        conditions = []
        params: list = []

        if validator_failed:
            conditions.append("validator_failed = %s")
            params.append(validator_failed)
        if chapter:
            conditions.append("chapter = %s")
            params.append(chapter)

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        params.append(limit)

        sql = f"""
            SELECT id, question_uuid, exam_set, chapter, question_text,
                   correct_answer, difficulty, validator_failed, reason, created_at
            FROM validation_logs
            {where}
            ORDER BY created_at DESC
            LIMIT %s
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            {
                "id": r[0],
                "question_uuid": str(r[1]) if r[1] else None,
                "exam_set": r[2],
                "chapter": r[3],
                "question_text": r[4],
                "correct_answer": r[5],
                "difficulty": r[6],
                "validator_failed": r[7],
                "reason": r[8],
                "created_at": r[9].isoformat() if r[9] else None,
            }
            for r in rows
        ]

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()
