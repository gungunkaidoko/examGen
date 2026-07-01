# CCC National Exam Generation Platform

Generates **10 exam sets × 100 questions** for the NIELIT CCC (Course on Computer Concepts) examination using **AWS Bedrock Claude Sonnet** and a multi-layer validation pipeline.

A full-stack web application (React + FastAPI) is included for browsing, filtering, and downloading exam sets.

---

## Quick Start — Full-Stack App

### 1. Start the FastAPI backend

```bash
cd NDU-project
python3 api.py
# Server runs at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

### 2. Start the React frontend

```bash
cd ../frontend
npm install        # first time only
npm run dev
# App runs at http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/exam-sets` | List all generated sets |
| GET | `/api/exam-sets/{n}` | Get all 100 questions (filterable) |
| GET | `/api/exam-sets/{n}/answer-key` | Compact answer key |
| GET | `/api/exam-sets/{n}/pdf` | Download set as PDF |
| GET | `/api/filters` | Available chapter / difficulty / bloom values |
| GET | `/api/stats` | Aggregate stats across all sets |
| POST | `/api/generate` | Trigger pipeline generation |
| GET | `/api/generate/status?job_id=...` | Poll generation status |
| GET | `/api/health` | Health check |

Query parameters for `/api/exam-sets/{n}`:
- `chapter`, `difficulty`, `bloom_level`, `search`

---

## Pipeline Usage (CLI)

```bash
# Generate all 10 exam sets (with PostgreSQL)
python main.py

# Generate without PostgreSQL (JSON output only)
python main.py --no-db

# Generate only specific sets
python main.py --sets 1 2 3 --no-db

# Preview allocation plan without generating
python main.py --dry-run
```

---

## Architecture

```
Chapter JSON Files
       │
       ▼
ContentExtractor  ──────────────────────────────────────────┐
       │                                                     │
       ▼                                                     │
BlueprintAllocator  (100 Qs / set, per-chapter weightage)   │
       │                                                     │
       ▼                                                     │
PromptBuilder  (LangChain templates)                        │
       │                                                     │
       ▼                                                     │
QuestionGenerator  ──(Claude Sonnet 4.5 via Bedrock)        │
       │                                                     │
       ▼                                                     │
Validation Layer 1  (LLM correctness check, temp=0)         │
  FAIL ──► RegenerateNode ──────────────────────────────────┘
       │
       ▼
Rule Engine  (structural / duplicate / option checks)
  FAIL ──► RegenerateNode
       │
       ▼
Validation Layer 2  (difficulty + chapter + course alignment)
  FAIL ──► DifficultyAdjustNode ──► QuestionGenerator
       │
       ▼
PostgreSQL Question Bank  +  JSON output files
       │
       ▼
FastAPI REST API  ──►  React Frontend
```

---

## Blueprint (CCC Weightage)

| Chapter | Min Q | Max Q |
|---------|-------|-------|
| 01 Introduction to Computer | 8 | 10 |
| 02 Operating System | 6 | 8 |
| 03 Word Processing | 15 | 18 |
| 04 Spreadsheet | 12 | 15 |
| 05 Presentation | 5 | 7 |
| 06 Internet & WWW | 10 | 12 |
| 07 Email, Social & eGov | 10 | 12 |
| 08 Digital Financial Tools | 10 | 13 |
| 09 FutureSkills & Cybersecurity | 12 | 15 |
| **TOTAL** | **88** | **100** |

---

## Output

```
output/
├── exam_set_01.json        # Set 1 — 100 questions
├── answer_key_01.json      # Compact answer key for set 1
├── exam_set_02.json
├── answer_key_02.json
...
└── all_exam_sets.json      # All sets combined
```

---

## File Structure

```
NDU-project/
├── api.py                  # ★ FastAPI REST layer (NEW)
├── main.py                 # CLI pipeline entry point
├── config.py               # Blueprint, DB config, env vars
├── models.py               # Pydantic Question, ExamSet models
├── database.py             # PostgreSQL question bank
├── pipeline.py             # End-to-end single-set pipeline
├── question_generator.py   # LLM call + JSON parsing
├── prompt_builder.py       # LangChain prompt templates
├── blueprint_allocator.py  # Per-set question count allocation
├── content_extractor.py    # JSON parser for chapter files
├── llm_client.py           # AWS Bedrock LLM instances
├── exporter.py             # JSON file exports
├── validation.py           # Validation layers + rule engine
├── validators/             # Individual validator modules
├── rag/                    # Pinecone RAG ingestion + retrieval
├── Knowledge_base/         # Chapter JSON + textbook chunks
├── output/                 # Generated exam set JSON files
└── requirements.txt

frontend/
├── src/
│   ├── pages/
│   │   ├── LandingPage.jsx     # Hero + stats + feature overview
│   │   ├── QuestionBankPage.jsx# Grid of all exam sets
│   │   └── ExamSetPage.jsx     # 100 questions with filters
│   ├── components/
│   │   ├── Navbar.jsx
│   │   ├── QuestionCard.jsx    # Single MCQ with show/hide answer
│   │   ├── FilterBar.jsx       # Search + chapter/difficulty/bloom
│   │   ├── GenerateModal.jsx   # Trigger generation + live log
│   │   ├── StatsRow.jsx
│   │   ├── Badge.jsx
│   │   ├── Spinner.jsx
│   │   └── ErrorMessage.jsx
│   └── lib/
│       ├── api.js              # Axios API client
│       └── utils.js            # Formatters + colour helpers
└── vite.config.js              # Proxy /api → localhost:8000
```
