"""
Ingestion Pipeline
------------------
Loads → chunks → embeds → upserts to Pinecone.

Source priority
~~~~~~~~~~~~~~~
1. ``ccc_book_chunks.jsonl``  (PRIMARY)
   Produced by parsing.py. Already split by section/subsection with rich
   metadata (book, chapter, section, subsection, content_type, question_type).
   Prose chunks that are still too long are re-split with
   RecursiveCharacterTextSplitter to keep each vector focused.

2. ``ccc_book.md``  (FALLBACK / SUPPLEMENT)
   Used only when the JSONL is unavailable. Parsed by header hierarchy
   (##, ###) so chapter/section metadata is preserved.

Chunking strategy
~~~~~~~~~~~~~~~~~
  content chunks  → RecursiveCharacterTextSplitter(800 chars, 150 overlap)
                    Separators: ["\\n\\n", "\\n", ". ", " "]
                    Keeps paragraphs together; falls back to sentence, then word.
  mcq / truefalse / practice_set / glossary chunks
                  → kept as-is (already atomic; splitting would break Q+options)

Each upserted vector carries this metadata:
  book, chapter, section, subsection, content_type, question_type,
  source_file, char_count, chunk_index, text (first 512 chars for debugging)
"""

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    RAG_CHUNKS_JSONL,
    RAG_MARKDOWN_FILE,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)
from rag.embedder import get_embedder
from rag.pinecone_store import get_store

logger = logging.getLogger(__name__)

# Content types that must NOT be further split
_ATOMIC_TYPES = {"mcq", "truefalse", "practice_set", "glossary"}

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)


# ── Loaders ───────────────────────────────────────────────────────────────────

def _load_jsonl(path: str) -> Iterator[dict]:
    """Yield raw chunk dicts from the JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"JSONL line {line_no} skipped (bad JSON): {e}")


def _load_markdown(path: str) -> list[dict]:
    """
    Parse a Markdown file by ## / ### headers into pseudo-chunks.
    Returns dicts compatible with the JSONL schema so the rest of the
    pipeline can treat them identically.
    """
    chunks = []
    current_chapter = ""
    current_section = ""
    buf = []
    chunk_id = 0

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip()
            if line.startswith("## "):
                if buf:
                    chunks.append(_md_chunk(buf, current_chapter, current_section, chunk_id))
                    chunk_id += 1
                    buf = []
                current_chapter = line[3:].strip()
                current_section = ""
            elif line.startswith("### "):
                if buf:
                    chunks.append(_md_chunk(buf, current_chapter, current_section, chunk_id))
                    chunk_id += 1
                    buf = []
                current_section = line[4:].strip()
            else:
                buf.append(line)

    if buf:
        chunks.append(_md_chunk(buf, current_chapter, current_section, chunk_id))

    return chunks


def _md_chunk(lines: list[str], chapter: str, section: str, idx: int) -> dict:
    return {
        "id": f"md-{idx:05d}",
        "text": "\n".join(lines).strip(),
        "book": "CCC Arihant",
        "chapter": chapter,
        "section": section,
        "subsection": None,
        "content_type": "content",
        "question_type": None,
    }


# ── Chunker ───────────────────────────────────────────────────────────────────

def _expand_chunk(raw: dict, source_file: str) -> list[dict]:
    """
    Return one or more final chunk dicts for a raw JSONL/markdown record.

    Atomic types (MCQ, glossary, etc.) pass through unchanged.
    Prose content longer than RAG_CHUNK_SIZE is split recursively;
    each sub-chunk inherits the parent's metadata.
    """
    text = (raw.get("text") or "").strip()
    if not text:
        return []

    content_type = raw.get("content_type", "content")
    source_id = raw.get("id", "")           # JSONL chunk id e.g. "ccc-0042"
    base_meta = {
        "book": raw.get("book", "CCC Arihant"),
        "chapter": raw.get("chapter", ""),
        "section": raw.get("section") or "",
        "subsection": raw.get("subsection") or "",
        "content_type": content_type,
        "question_type": raw.get("question_type") or "",
        "source_file": source_file,
        "source_id": source_id,
    }

    # Atomic chunks — preserve as-is
    if content_type in _ATOMIC_TYPES or len(text) <= RAG_CHUNK_SIZE:
        return [{**base_meta, "text": text, "chunk_index": 0, "source_id": source_id}]

    # Prose content — recursively split
    sub_texts = _splitter.split_text(text)
    return [
        {**base_meta, "text": sub, "chunk_index": i, "source_id": source_id}
        for i, sub in enumerate(sub_texts)
        if sub.strip()
    ]


# ── ID + metadata helpers ─────────────────────────────────────────────────────

def _stable_id(text: str, chapter: str, chunk_index: int, source_id: str = "") -> str:
    """
    Deterministic vector ID: SHA-1 of (source_id + chapter + chunk_index + first 200 chars).
    Including source_id (the JSONL chunk id) eliminates hash collisions between
    different chunks that share the same chapter and chunk_index.
    Stable across re-runs so re-ingestion doesn't duplicate vectors.
    """
    key = f"{source_id}|{chapter}|{chunk_index}|{text[:200]}"
    return "ccc-" + hashlib.sha1(key.encode()).hexdigest()[:20]


def _build_metadata(chunk: dict) -> dict:
    """
    Build the Pinecone metadata dict.
    Pinecone metadata values must be str / int / float / bool / list[str].
    Truncate text to 512 chars (metadata field limit is ~40 KB total, but
    keeping it short improves query latency).
    """
    return {
        "book": chunk["book"],
        "chapter": chunk["chapter"],
        "section": chunk["section"],
        "subsection": chunk["subsection"],
        "content_type": chunk["content_type"],
        "question_type": chunk["question_type"],
        "source_file": chunk["source_file"],
        "char_count": len(chunk["text"]),
        "chunk_index": chunk["chunk_index"],
        "text": chunk["text"][:512],   # stored for reference; full text in embed
    }


# ── Public entry point ────────────────────────────────────────────────────────

def ingest(
    jsonl_path: str = RAG_CHUNKS_JSONL,
    md_path: str = RAG_MARKDOWN_FILE,
    force_reingest: bool = False,
    content_type_filter: list[str] | None = None,
) -> int:
    """
    Full ingestion pipeline: load → chunk → embed → upsert.

    Args:
        jsonl_path:           Path to ccc_book_chunks.jsonl (primary source).
        md_path:              Path to ccc_book.md (fallback if JSONL missing).
        force_reingest:       Delete existing namespace before upserting.
        content_type_filter:  Only ingest chunks of these content_types.
                              None means ingest everything.

    Returns:
        Total number of vectors upserted.
    """
    store = get_store()
    embedder = get_embedder()

    if force_reingest:
        logger.info("force_reingest=True — deleting existing namespace…")
        store.delete_namespace()

    # ── 1. Load raw chunks ────────────────────────────────────────────────────
    if os.path.exists(jsonl_path):
        logger.info(f"Loading JSONL: {jsonl_path}")
        raw_chunks = list(_load_jsonl(jsonl_path))
        source_file = Path(jsonl_path).name
    elif os.path.exists(md_path):
        logger.warning(
            f"JSONL not found at {jsonl_path}. "
            f"Falling back to Markdown: {md_path}"
        )
        raw_chunks = _load_markdown(md_path)
        source_file = Path(md_path).name
    else:
        raise FileNotFoundError(
            f"No source files found.\n"
            f"  JSONL expected: {jsonl_path}\n"
            f"  MD expected   : {md_path}"
        )

    logger.info(f"Loaded {len(raw_chunks)} raw chunks from {source_file}")

    # ── 2. Expand / re-chunk ──────────────────────────────────────────────────
    final_chunks: list[dict] = []
    for raw in raw_chunks:
        if content_type_filter and raw.get("content_type") not in content_type_filter:
            continue
        final_chunks.extend(_expand_chunk(raw, source_file))

    logger.info(f"Expanded to {len(final_chunks)} final chunks after splitting")

    # ── 3. Embed + build vector tuples ────────────────────────────────────────
    vectors: list[tuple[str, list[float], dict]] = []
    texts = [c["text"] for c in final_chunks]

    logger.info(f"Embedding {len(texts)} texts via Titan V2…")
    embeddings = embedder.embed_texts(texts)

    for chunk, embedding in zip(final_chunks, embeddings):
        vec_id = _stable_id(chunk["text"], chunk["chapter"], chunk["chunk_index"], chunk.get("source_id", ""))
        metadata = _build_metadata(chunk)
        vectors.append((vec_id, embedding, metadata))

    # ── 4. Upsert to Pinecone ─────────────────────────────────────────────────
    logger.info(f"Upserting {len(vectors)} vectors to Pinecone…")
    total = store.upsert_vectors(vectors)

    stats = store.describe()
    logger.info(
        f"✓ Ingestion complete — {total} vectors upserted. "
        f"Index stats: {stats.get('namespaces', {})}"
    )
    return total
