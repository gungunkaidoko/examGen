"""
Central configuration for CCC Exam Generation Platform.
All blueprint weights, chapter mappings, and generation settings live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# API / DB configuration
# ─────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")

# AWS Bedrock / Claude configuration
AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
AWS_DEFAULT_REGION: str = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
BEDROCK_INFERENCE_PROFILE_ARN: str = os.getenv("BEDROCK_INFERENCE_PROFILE_ARN", "")
MODEL_ID: str = os.getenv("MODEL_ID", "anthropic.claude-sonnet-4-5-20250929-v1:0")

DB_CONFIG: dict = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
    "dbname": os.getenv("DB_NAME", "ccc_question_bank"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
}

# ─────────────────────────────────────────────
# Generation settings
# ─────────────────────────────────────────────
NUM_EXAM_SETS: int = int(os.getenv("NUM_EXAM_SETS", 10))
QUESTIONS_PER_EXAM: int = int(os.getenv("QUESTIONS_PER_EXAM", 100))
MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 3))

# Gemini model
GEMINI_MODEL: str = "gemini-1.5-flash"  # Switched from 2.5-flash due to availability

# ─────────────────────────────────────────────
# Generation metadata constants
# ─────────────────────────────────────────────
PROMPT_VERSION: str = "v2"   # Bump when any prompt template changes

# ─────────────────────────────────────────────
# Question format distribution
# Proportions across a chapter's question batch
# Keys match question_type field values
# ─────────────────────────────────────────────
QUESTION_FORMAT_DISTRIBUTION: dict = {
    "standard":        0.35,   # plain single-correct MCQ
    "assertion_reason": 0.10,  # Assertion–Reason
    "statement_based":  0.15,  # Statement I / II / III — which are correct?
    "scenario_based":   0.15,  # short case/scenario stem
    "fill_blank":       0.10,  # fill-in-the-blank with 4 options
    "match_following":  0.08,  # match column A to column B (option-based)
    "sequence_order":   0.07,  # arrange steps/items in correct order
}

# ─────────────────────────────────────────────
# RAG / Pinecone / Embedding configuration
# ─────────────────────────────────────────────
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "question-bank")
PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_HOST: str = os.getenv("PINECONE_HOST", "")

EMBEDDING_MODEL_ID: str = os.getenv("EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBEDDING_DIMENSIONS: int = int(os.getenv("EMBEDDING_DIMENSIONS", 1024))

# Source files produced by parsing.py (paths relative to project root)
RAG_CHUNKS_JSONL: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Knowledge_base", "book", "ccc_book_chunks.jsonl"
)
RAG_MARKDOWN_FILE: str = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Knowledge_base", "book", "ccc_book.md"
)

# Chunking settings
RAG_CHUNK_SIZE: int = 800        # chars — fits well within Titan's token limit
RAG_CHUNK_OVERLAP: int = 150     # ~19 % overlap keeps context across boundaries
RAG_BATCH_SIZE: int = 100        # vectors per Pinecone upsert batch

# ─────────────────────────────────────────────
# RAG chapter name mapping
# Maps blueprint chapter keys → exact chapter strings stored in Pinecone
# (these come from parsing.py's CHAPTER_TITLES list)
# ─────────────────────────────────────────────
PINECONE_CHAPTER_NAMES: dict = {
    "chapter_01_introduction_to_computer":      "1. INTRODUCTION TO COMPUTER",
    "chapter_02_operating_system":              "2. INTRODUCTION TO OPERATING SYSTEM",
    "chapter_03_word_processing":               "3. WORD PROCESSING",
    "chapter_04_spreadsheet":                   "4. SPREAD SHEET",
    "chapter_05_presentation":                  "5. PRESENTATION",
    "chapter_06_internet_www":                  "6. INTRODUCTION TO INTERNET AND WWW",
    "chapter_07_email_social_egov":             "7. E-MAIL, SOCIAL NETWORKING AND",
    "chapter_08_digital_financial_tools":       "8. DIGITAL FINANCIAL TOOLS",
    "chapter_09_futureskills_cybersecurity":    "9. OVERVIEW OF FUTURESKILLS",
}
# Based on official CCC weightage table
# ─────────────────────────────────────────────
CHAPTER_BLUEPRINT: dict = {
    "chapter_01_introduction_to_computer": {
        "file": "chapter_01_introduction_to_computer.json",
        "min_q": 8,
        "max_q": 10,
        "difficulty_split": {"easy": 0.5, "medium": 0.35, "hard": 0.15},
    },
    "chapter_02_operating_system": {
        "file": "chapter_02_operating_system.json",
        "min_q": 6,
        "max_q": 8,
        "difficulty_split": {"easy": 0.5, "medium": 0.35, "hard": 0.15},
    },
    "chapter_03_word_processing": {
        "file": "chapter_03_word_processing.json",
        "min_q": 15,
        "max_q": 18,
        "difficulty_split": {"easy": 0.4, "medium": 0.4, "hard": 0.20},
    },
    "chapter_04_spreadsheet": {
        "file": "chapter_04_spreadsheet.json",
        "min_q": 12,
        "max_q": 15,
        "difficulty_split": {"easy": 0.4, "medium": 0.4, "hard": 0.20},
    },
    "chapter_05_presentation": {
        "file": "chapter_05_presentation.json",
        "min_q": 5,
        "max_q": 7,
        "difficulty_split": {"easy": 0.5, "medium": 0.35, "hard": 0.15},
    },
    "chapter_06_internet_www": {
        "file": "chapter_06_internet_www.json",
        "min_q": 10,
        "max_q": 12,
        "difficulty_split": {"easy": 0.4, "medium": 0.4, "hard": 0.20},
    },
    "chapter_07_email_social_egov": {
        "file": "chapter_07_email_social_egov.json",
        "min_q": 10,
        "max_q": 12,
        "difficulty_split": {"easy": 0.4, "medium": 0.4, "hard": 0.20},
    },
    "chapter_08_digital_financial_tools": {
        "file": "chapter_08_digital_financial_tools.json",
        "min_q": 10,
        "max_q": 13,
        "difficulty_split": {"easy": 0.35, "medium": 0.45, "hard": 0.20},
    },
    "chapter_09_futureskills_cybersecurity": {
        "file": "chapter_09_futureskills_cybersecurity.json",
        "min_q": 12,
        "max_q": 15,
        "difficulty_split": {"easy": 0.35, "medium": 0.45, "hard": 0.20},
    },
}

# Directory containing chapter JSON files
KNOWLEDGE_BASE_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Knowledge_base", "ccc")
