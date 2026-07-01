"""
Match-the-Following Formatter
------------------------------
Automatically normalises match_following question stems into a clean,
vertically-aligned Column A / Column B layout regardless of how the
LLM originally formatted them.

The canonical output format is:

  Match the items in Column A with Column B:

  Column A:
  1. <item one>
  2. <item two>
  3. <item three>

  Column B:
  a. <match one>
  b. <match two>
  c. <match three>

Public API
----------
  normalize_match_question(question) → Question
    Returns a new Question with the stem re-formatted (or the original
    if it cannot be parsed / is not a match_following question).

  validate_match_format(question) → tuple[bool, str]
    Returns (passed, reason). Fails if the stem cannot be parsed into
    a valid Column A / Column B layout after normalisation.
"""

import re
import logging
from models import Question

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

# Matches numbered items like "1. ...", "1) ...", "(1) ..."
_NUM_ITEM_RE  = re.compile(r"^\s*[\(\[]?(\d+)[\)\].\-–]?\s+(.+)$")

# Matches lettered items like "a. ...", "a) ...", "(a) ..."
_LTR_ITEM_RE  = re.compile(r"^\s*[\(\[]?([a-dA-D])[\)\].\-–]?\s+(.+)$")

# Section header patterns
_COL_A_RE = re.compile(r"column\s*a|list\s*[i1]|col\.?\s*a|part\s*a|group\s*a", re.IGNORECASE)
_COL_B_RE = re.compile(r"column\s*b|list\s*[ii2]|col\.?\s*b|part\s*b|group\s*b", re.IGNORECASE)


def _parse_column_sections(text: str) -> tuple[list[str], list[str]]:
    """
    Try to split the stem text into Column A items and Column B items.

    Returns (col_a_items, col_b_items).  Both lists may be empty if parsing fails.
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    col_a: list[str] = []
    col_b: list[str] = []
    current: list[str] | None = None

    for line in lines:
        # Detect section headers
        if _COL_A_RE.search(line):
            current = col_a
            continue
        if _COL_B_RE.search(line):
            current = col_b
            continue

        # Skip the leading instruction line ("Match the items...")
        if re.search(r"match\s+the|match\s+each|match\s+column", line, re.IGNORECASE):
            continue

        # Numbered item → goes into current section (or col_a if no section yet)
        m_num = _NUM_ITEM_RE.match(line)
        if m_num:
            item_text = m_num.group(2).strip()
            target = current if current is not None else col_a
            target.append(item_text)
            continue

        # Lettered item → goes into col_b if no section header found yet
        m_ltr = _LTR_ITEM_RE.match(line)
        if m_ltr:
            item_text = m_ltr.group(2).strip()
            target = current if current is not None else col_b
            target.append(item_text)
            continue

    return col_a, col_b


def _build_canonical_stem(col_a: list[str], col_b: list[str]) -> str:
    """Render Column A and Column B as the canonical vertically-aligned format."""
    col_a_lines = "\n".join(f"{i+1}. {item}" for i, item in enumerate(col_a))
    col_b_lines = "\n".join(f"{chr(97+i)}. {item}" for i, item in enumerate(col_b))
    return (
        "Match the items in Column A with Column B:\n\n"
        f"Column A:\n{col_a_lines}\n\n"
        f"Column B:\n{col_b_lines}"
    )


def _is_already_canonical(text: str) -> bool:
    """Return True if the stem already uses the canonical format."""
    return (
        "Column A:" in text
        and "Column B:" in text
        and bool(re.search(r"\n\d+\.", text))   # has numbered items on own lines
        and bool(re.search(r"\n[a-d]\.", text))  # has lettered items on own lines
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def normalize_match_question(question: Question) -> Question:
    """
    Re-format the stem of a match_following question into the canonical layout.

    If the stem is already canonical or cannot be parsed, returns the original
    question unchanged.
    """
    if question.question_type != "match_following":
        return question

    stem = question.question

    if _is_already_canonical(stem):
        return question  # nothing to do

    col_a, col_b = _parse_column_sections(stem)

    if len(col_a) < 2 or len(col_b) < 2:
        logger.debug(
            f"match_formatter: could not parse columns "
            f"(col_a={col_a}, col_b={col_b}) — leaving stem unchanged"
        )
        return question

    new_stem = _build_canonical_stem(col_a, col_b)
    logger.debug(f"match_formatter: normalised stem for question '{stem[:60]}'")
    return question.model_copy(update={"question": new_stem})


def validate_match_format(question: Question) -> tuple[bool, str]:
    """
    Validate that a match_following question has a parseable Column A / Column B layout.

    First attempts to normalise; if normalisation still yields < 2 items per column,
    returns (False, reason).

    Returns (True, "") for non-match_following questions.
    """
    if question.question_type != "match_following":
        return True, ""

    normalised = normalize_match_question(question)
    stem = normalised.question

    if not _is_already_canonical(stem):
        return False, (
            "match_following question could not be parsed into a valid "
            "Column A / Column B layout. Ensure the stem lists numbered items "
            "under 'Column A:' and lettered items under 'Column B:', "
            "each on their own line."
        )

    col_a, col_b = _parse_column_sections(stem)
    if len(col_a) < 2:
        return False, f"Column A has only {len(col_a)} item(s); need at least 2."
    if len(col_b) < 2:
        return False, f"Column B has only {len(col_b)} item(s); need at least 2."
    if len(col_a) != len(col_b):
        return False, (
            f"Column A has {len(col_a)} items but Column B has {len(col_b)} items. "
            "They must match."
        )

    return True, ""
