# ============================================================
# CELL 1: Install Dependencies
# ============================================================
!pip install -q -U docling pymupdf

# ============================================================
# CELL 2: Upload PDF
# ============================================================
from google.colab import files
uploaded = files.upload()
pdf_file = list(uploaded.keys())[0]
print(f"Uploaded PDF: {pdf_file}")

# ============================================================
# CELL 3: Configure & Run Docling
# ============================================================
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = False              # digital PDF, confirmed from inspection
pipeline_options.do_table_structure = True

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)

try:
    result = converter.convert(pdf_file)
    doc = result.document
    print("✅ PDF Converted Successfully")
except Exception as e:
    print(f"❌ Conversion failed: {e}")
    raise

full_text = doc.export_to_text()
print(f"Characters extracted: {len(full_text):,}")

# ============================================================
# CELL 4: Strip front matter (publisher info, TOC, preface)
# ------------------------------------------------------------
# This book's real content starts at Chapter 1. Everything before
# that (cover, copyright, TOC) is noise for RAG and will only
# pollute retrieval.
# ============================================================
import re

chapter1_match = re.search(r"1\.\s*INTRODUCTION TO COMPUTER", full_text)
if chapter1_match:
    body_text = full_text[chapter1_match.start():]
    print(f"✅ Front matter stripped — body starts at char {chapter1_match.start()}")
else:
    body_text = full_text
    print("⚠️ Could not locate Chapter 1 start — using full text. Check manually.")

# ============================================================
# CELL 5: Split into top-level chapters + back-matter sections
# ------------------------------------------------------------
# This book has 9 numbered chapters, then Abbreviations, Glossary,
# and Practice Sets. Each becomes its own zone with its own
# chunking rules.
# ============================================================
CHAPTER_TITLES = [
    "1. INTRODUCTION TO COMPUTER",
    "2. INTRODUCTION TO OPERATING SYSTEM",
    "3. WORD PROCESSING",
    "4. SPREAD SHEET",
    "5. PRESENTATION",
    "6. INTRODUCTION TO INTERNET AND WWW",
    "7. E-MAIL, SOCIAL NETWORKING AND",
    "8. DIGITAL FINANCIAL TOOLS",
    "9. OVERVIEW OF FUTURESKILLS",
]
BACK_MATTER_TITLES = [
    "Computer Abbreviations",
    "Computer Glossary",
    "Practice Set 1",
    "Practice Set 2",
    "Practice Set 3",
    "Practice Set 4",
    "Practice Set 5",
]

ALL_SECTION_TITLES = CHAPTER_TITLES + BACK_MATTER_TITLES

def find_section_boundaries(text: str, titles: list[str]) -> list[dict]:
    """Find start offsets for each known section title, in document order."""
    found = []
    for title in titles:
        m = re.search(re.escape(title), text)
        if m:
            found.append({"title": title, "start": m.start()})
    found.sort(key=lambda x: x["start"])
    return found

boundaries = find_section_boundaries(body_text, ALL_SECTION_TITLES)

sections = []
for i, b in enumerate(boundaries):
    end = boundaries[i + 1]["start"] if i + 1 < len(boundaries) else len(body_text)
    sections.append({
        "title": b["title"],
        "is_chapter": b["title"] in CHAPTER_TITLES,
        "content": body_text[b["start"]:end].strip(),
    })

print(f"✅ Found {len(sections)} sections")
for s in sections:
    print(f"  [{'CHAPTER' if s['is_chapter'] else 'BACK-MATTER'}] {s['title']}  ({len(s['content'])} chars)")

# ============================================================
# CELL 6: Split each CHAPTER into content / MCQ / True-False zones
# ------------------------------------------------------------
# Within a chapter, structure is always:
#   <prose with numbered subsections> → "Multiple Choice Questions" → "True/False" (optional)
# ============================================================
def split_chapter_zones(chapter_text: str) -> dict:
    mcq_match = re.search(r"Multiple Choice Questions", chapter_text)
    tf_match = re.search(r"True/False", chapter_text)

    if mcq_match:
        content = chapter_text[:mcq_match.start()].strip()
        if tf_match and tf_match.start() > mcq_match.start():
            mcq = chapter_text[mcq_match.end():tf_match.start()].strip()
            truefalse = chapter_text[tf_match.end():].strip()
        else:
            mcq = chapter_text[mcq_match.end():].strip()
            truefalse = ""
    else:
        content, mcq, truefalse = chapter_text.strip(), "", ""

    return {"content": content, "mcq": mcq, "truefalse": truefalse}

# ============================================================
# CELL 7: Type-specific chunkers
# ------------------------------------------------------------
# 1) Prose -> split by numbered subsections (1.2, 1.2.1 etc), then
#    by paragraph if a subsection is still too long.
# 2) MCQ   -> split per question number, options stay attached.
# 3) True/False -> split per statement number.
# 4) Glossary/Abbreviations -> split per term entry.
# ============================================================
SUBSECTION_PATTERN = re.compile(r"(?m)^(\d+(?:\.\d+){1,3})\s+([A-Z][^\n]{2,80})$")
QUESTION_PATTERN = re.compile(r"(?m)^(\d{1,3})\.\s")
GLOSSARY_TERM_PATTERN = re.compile(r"(?m)^([A-Z][A-Za-z0-9/\(\) ]{1,40})(?=\s[A-Z][a-z])")

MAX_PROSE_CHUNK_CHARS = 1800

def chunk_prose(text: str, chapter_title: str) -> list[dict]:
    """Split chapter prose by numbered subsections; fall back to paragraph chunks."""
    matches = list(SUBSECTION_PATTERN.finditer(text))
    chunks = []

    if not matches:
        # No subsection markers found — chunk by paragraph
        paras = [p.strip() for p in text.split("\n\n") if len(p.strip()) > 30]
        for p in paras:
          chunks.append({
              "section": None,
              "subsection": None,
              "text": p
          })
        return _attach_metadata(chunks, chapter_title, "content")

    for i, m in enumerate(matches):
        subsection_no = m.group(1)
        subsection_title = m.group(2).strip()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[m.end():end].strip()

        # Long subsections still get split further, but tagged with the
        # SAME subsection metadata so retrieval context isn't lost.
        if len(body) > MAX_PROSE_CHUNK_CHARS:
            sub_paras = [p.strip() for p in body.split("\n\n") if len(p.strip()) > 20]
            buf = ""
            for p in sub_paras:
                if len(buf) + len(p) > MAX_PROSE_CHUNK_CHARS and buf:
                    chunks.append({
                        "section": subsection_no,
                        "subsection": subsection_title,
                        "text": buf.strip()
                    })
                    buf = ""
                buf += p + "\n"
            if buf.strip():
                chunks.append({
                    "section": subsection_no,
                    "subsection": subsection_title,
                    "text": buf.strip()
                })
        else:
            chunks.append({
            "section": subsection_no,
            "subsection": subsection_title,
            "text": body,
        })

    return _attach_metadata(chunks, chapter_title, "content")


def chunk_questions(text: str, chapter_title: str, content_type: str) -> list[dict]:
    """Split MCQ or True/False blocks — one chunk per question/statement."""
    matches = list(QUESTION_PATTERN.finditer(text))
    chunks = []
    for i, m in enumerate(matches):
        q_no = m.group(1)
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        q_text = text[start:end].strip()
        if len(q_text) < 8:
            continue
        chunks.append({"question_no": q_no, "question_type": content_type, "text": q_text})
    return _attach_metadata(chunks, chapter_title, content_type)


def chunk_glossary(text: str, section_title: str) -> list[dict]:
    """
    Split glossary/abbreviation text into term entries.
    Heuristic: a new entry starts where a short Title-Case run is
    immediately followed by a capitalized sentence start.
    Falls back to fixed-size chunking if the pattern is too noisy.
    """
    matches = list(GLOSSARY_TERM_PATTERN.finditer(text))
    chunks = []
    if len(matches) < 5:
        # Pattern too unreliable on this text — fall back to paragraph chunks
        paras = [p.strip() for p in text.split("\n") if len(p.strip()) > 15]
        buf = ""
        for p in paras:
            buf += p + " "
            if len(buf) > 500:
                chunks.append({"term": None, "text": buf.strip()})
                buf = ""
        if buf.strip():
            chunks.append({"term": None, "text": buf.strip()})
        return _attach_metadata(chunks, section_title, "glossary")

    for i, m in enumerate(matches):
        term = m.group(1).strip()
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        entry = text[start:end].strip()
        if len(entry) < 10:
            continue
        chunks.append({"term": term, "text": entry})
    return _attach_metadata(chunks, section_title, "glossary")


def _attach_metadata(chunks: list[dict], chapter_title: str, content_type: str) -> list[dict]:
    for c in chunks:
        c["book"] = "CCC Arihant"
        c["chapter"] = chapter_title
        c["content_type"] = content_type  # content | mcq | truefalse | glossary
    return chunks

# ============================================================
# CELL 8: Run the full pipeline across all sections
# ============================================================
all_chunks = []

for sec in sections:
    if sec["is_chapter"]:
        zones = split_chapter_zones(sec["content"])
        all_chunks += chunk_prose(zones["content"], sec["title"])
        if zones["mcq"]:
            all_chunks += chunk_questions(zones["mcq"], sec["title"], "mcq")
        if zones["truefalse"]:
            all_chunks += chunk_questions(zones["truefalse"], sec["title"], "truefalse")
    else:
        # Back matter: glossary/abbreviations use glossary chunker,
        # Practice Sets use the question chunker (they're pure MCQ/T-F banks)
        if "Glossary" in sec["title"] or "Abbreviations" in sec["title"]:
            all_chunks += chunk_glossary(sec["content"], sec["title"])
        else:
            all_chunks += chunk_questions(sec["content"], sec["title"], "practice_set")

# Assign stable IDs
for idx, c in enumerate(all_chunks):
    c["id"] = f"ccc-{idx:04d}"

print(f"✅ Total chunks created: {len(all_chunks)}")
from collections import Counter
print(Counter(c["content_type"] for c in all_chunks))

# ============================================================
# CELL 9: Quality check — peek at a few chunks per type
# ============================================================
import json

for ctype in ["content", "mcq", "truefalse", "glossary", "practice_set"]:
    sample = next((c for c in all_chunks if c["content_type"] == ctype), None)
    if sample:
        print(f"\n--- Sample [{ctype}] ---")
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:600])

# ============================================================
# CELL 10: Save chunks as JSONL (one JSON object per line)
# ------------------------------------------------------------
# JSONL is the standard format for feeding a chunking pipeline
# into an embedding/Pinecone step later — each line is one
# self-contained record ready to embed.
# ============================================================
import os

base_name = os.path.splitext(pdf_file)[0]
chunks_file = base_name + "_chunks.jsonl"

with open(chunks_file, "w", encoding="utf-8") as f:
    for c in all_chunks:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")

print(f"✅ Chunks saved as: {chunks_file}")

# ============================================================
# CELL 11: Also save full raw text + markdown (for backup/debug)
# ============================================================
markdown = doc.export_to_markdown()

txt_file = base_name + ".txt"
md_file = base_name + ".md"

with open(txt_file, "w", encoding="utf-8") as f:
    f.write(full_text)
with open(md_file, "w", encoding="utf-8") as f:
    f.write(markdown)

print(f"✅ Raw text saved as: {txt_file}")
print(f"✅ Markdown saved as: {md_file}")

# ============================================================
# CELL 12: Download files
# ============================================================
files.download(chunks_file)
files.download(txt_file)
files.download(md_file)
print("\n✅ All files downloaded successfully.")