"""
test_db_connection.py
---------------------
Quick sanity check — inserts one dummy question into question_bank
and immediately deletes it.

Run:
    python3 test_db_connection.py

Expected output:
    ✓ Connected to PostgreSQL
    ✓ Schema ready (question_bank table exists)
    ✓ Inserted test question  uuid=<uuid>
    ✓ Verified row exists in DB
    ✓ Cleaned up test row
    ✅ PostgreSQL is connected and working correctly
"""

import sys
import os
import uuid

sys.path.insert(0, os.path.dirname(__file__))

from database import QuestionBankDB

TEST_UUID = str(uuid.uuid4())
TEST_HASH = "test" + "0" * 60   # 64-char dummy hash

def main():
    db = QuestionBankDB()

    # ── 1. Connect ────────────────────────────────────────────────────────────
    try:
        db.connect()
        print("✓ Connected to PostgreSQL")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        sys.exit(1)

    # ── 2. Check table exists ─────────────────────────────────────────────────
    try:
        with db._conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.tables
                    WHERE table_name = 'question_bank'
                )
            """)
            exists = cur.fetchone()[0]
        if exists:
            print("✓ Schema ready (question_bank table exists)")
        else:
            print("✗ question_bank table not found — run schema creation first")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Schema check failed: {e}")
        sys.exit(1)

    # ── 3. Insert one dummy question ──────────────────────────────────────────
    try:
        with db._conn.cursor() as cur:
            cur.execute("""
                INSERT INTO question_bank (
                    question_uuid, question_hash, exam_set,
                    question, option_a, option_b, option_c, option_d,
                    correct_answer, difficulty, bloom_level, chapter,
                    validation_status, tags, weightage,
                    model_name, prompt_version, generation_time, retry_count
                ) VALUES (
                    %s, %s, 0,
                    'TEST: What is 1 + 1?',
                    '2', '3', '4', '5',
                    '2', 'easy', 'remember', 'TEST_CHAPTER',
                    'approved', '{}', 1.0,
                    'test-model', 'v0', 0.0, 0
                )
            """, (TEST_UUID, TEST_HASH))
        db._conn.commit()
        print(f"✓ Inserted test question  uuid={TEST_UUID}")
    except Exception as e:
        print(f"✗ Insert failed: {e}")
        db._conn.rollback()
        db.close()
        sys.exit(1)

    # ── 4. Verify + display the inserted row ─────────────────────────────────
    try:
        with db._conn.cursor() as cur:
            cur.execute("""
                SELECT id, question_uuid, question, correct_answer,
                       difficulty, chapter, created_at
                FROM question_bank
                WHERE question_uuid = %s
            """, (TEST_UUID,))
            row = cur.fetchone()
        if row:
            print(f"✓ Verified row exists in DB")
            print("\n── Inserted Row ─────────────────────────────────────────")
            print(f"  id            : {row[0]}")
            print(f"  question_uuid : {row[1]}")
            print(f"  question      : {row[2]}")
            print(f"  correct_answer: {row[3]}")
            print(f"  difficulty    : {row[4]}")
            print(f"  chapter       : {row[5]}")
            print(f"  created_at    : {row[6]}")
            print("─────────────────────────────────────────────────────────")
        else:
            print("✗ Row not found after insert — something is wrong")
            db.close()
            sys.exit(1)
    except Exception as e:
        print(f"✗ Verify failed: {e}")
        db.close()
        sys.exit(1)

    # ── 5. Clean up ───────────────────────────────────────────────────────────
    try:
        with db._conn.cursor() as cur:
            cur.execute(
                "DELETE FROM question_bank WHERE question_uuid = %s",
                (TEST_UUID,)
            )
        db._conn.commit()
        print("✓ Cleaned up test row")
    except Exception as e:
        print(f"✗ Cleanup failed: {e}")
        db.close()
        sys.exit(1)

    db.close()
    print("\n✅ PostgreSQL is connected and working correctly")


if __name__ == "__main__":
    main()
