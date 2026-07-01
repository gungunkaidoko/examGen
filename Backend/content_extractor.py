"""
Content Extraction Node
-----------------------
Loads chapter JSON files and returns structured content
ready to be consumed by the Prompt Builder.
"""

import json
import os
from typing import Any

from config import CHAPTER_BLUEPRINT, KNOWLEDGE_BASE_DIR


class ChapterContent:
    """Holds extracted content for a single chapter."""

    def __init__(self, chapter_key: str, raw: dict):
        self.chapter_key = chapter_key
        self.course: str = raw.get("course", "CCC")
        self.chapter_name: str = raw.get("chapter", chapter_key)
        self.summary: str = raw.get("summary", "")
        self.topics: list[dict] = raw.get("topics", [])

    def format_topics_for_prompt(self) -> str:
        """Returns a readable topic list for LLM prompt injection."""
        lines = []
        for t in self.topics:
            lines.append(f"• {t['topic']}: {t['notes']}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<ChapterContent: {self.chapter_name} ({len(self.topics)} topics)>"


class ContentExtractor:
    """
    Parses all chapter JSON files defined in the blueprint
    and caches them for repeated use across exam set generation.
    """

    def __init__(self):
        self._cache: dict[str, ChapterContent] = {}

    def load_all(self) -> dict[str, ChapterContent]:
        """Load every chapter defined in the blueprint."""
        for chapter_key, meta in CHAPTER_BLUEPRINT.items():
            if chapter_key not in self._cache:
                self._cache[chapter_key] = self._load_chapter(chapter_key, meta["file"])
        return self._cache

    def get(self, chapter_key: str) -> ChapterContent:
        """Return a single chapter's content, loading it if not cached."""
        if chapter_key not in self._cache:
            meta = CHAPTER_BLUEPRINT[chapter_key]
            self._cache[chapter_key] = self._load_chapter(chapter_key, meta["file"])
        return self._cache[chapter_key]

    def _load_chapter(self, chapter_key: str, filename: str) -> ChapterContent:
        filepath = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Chapter file not found: {filepath}\n"
                f"Expected at: {KNOWLEDGE_BASE_DIR}/{filename}"
            )
        with open(filepath, "r", encoding="utf-8") as f:
            raw: dict[str, Any] = json.load(f)
        return ChapterContent(chapter_key, raw)
