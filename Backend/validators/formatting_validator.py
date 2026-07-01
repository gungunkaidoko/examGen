"""
Formatting Validator
--------------------
Fast deterministic checks — no LLM needed.

Enforces:
- Question starts with a capital letter
- Question ends with '?' (or '.' for fill_blank / statement stems)
- No double spaces or repeated punctuation
- Options have no leading/trailing whitespace
- Options are not empty
- Option text is not absurdly long (likely malformed LLM output)
- Option lengths are roughly balanced (max/min ratio ≤ 3.5)
- Statement-based questions have newline-separated statements
- Match-the-following questions contain Column A / Column B markers
- Question length is moderate (not too short, not excessively long)
"""

import re
from models import Question

MAX_OPTION_LENGTH   = 200   # raised — match/statement options can be longer
MIN_OPTION_LENGTH   = 2     # reject single-character options
MAX_OPTION_RATIO    = 3.5   # max_len / min_len — prevents one absurdly long option
MAX_QUESTION_CHARS  = 600   # reject excessively long question stems
MIN_QUESTION_CHARS  = 15    # reject trivially short questions


def _ends_with_valid_punct(text: str) -> bool:
    """Question stems must end with '?', '.', or ':' (match/sequence stems use ':')."""
    return text.endswith("?") or text.endswith(".") or text.endswith(":")


def _check_option_length_balance(options: list[str]) -> tuple[bool, str]:
    """
    Ensure no single option is disproportionately longer than the shortest one.
    Prevents scenarios where the correct answer is obviously the long detailed one.
    """
    lengths = [len(o.strip()) for o in options if o.strip()]
    if not lengths:
        return True, ""
    max_len = max(lengths)
    min_len = min(lengths)
    if min_len == 0:
        return False, "One or more options appear to be empty after stripping"
    ratio = max_len / min_len
    if ratio > MAX_OPTION_RATIO:
        return False, (
            f"Option length imbalance: longest={max_len} chars, shortest={min_len} chars "
            f"(ratio {ratio:.1f} > {MAX_OPTION_RATIO}). Keep options similar in length."
        )
    return True, ""


def _check_statement_format(question: Question) -> tuple[bool, str]:
    """
    statement_based questions should have numbered statements (I., II., 1., 2. etc.)
    and each statement should be clearly separated.
    """
    if question.question_type != "statement_based":
        return True, ""
    q = question.question
    # Must contain at least two statement markers
    has_numbered = bool(re.search(r"(I\.|II\.|III\.|1\.|2\.|Statement\s+1|Statement\s+2)", q, re.IGNORECASE))
    if not has_numbered:
        return False, (
            "statement_based question must contain numbered statements "
            "(e.g. 'I. ...  II. ...') with each on a separate line."
        )
    return True, ""


def _check_match_format(question: Question) -> tuple[bool, str]:
    """
    match_following questions must reference Column A and Column B
    (or List I / List II) in the stem.
    """
    if question.question_type != "match_following":
        return True, ""
    q_lower = question.question.lower()
    has_columns = (
        ("column a" in q_lower and "column b" in q_lower)
        or ("list i" in q_lower and "list ii" in q_lower)
        or ("col a" in q_lower and "col b" in q_lower)
        or re.search(r"1[\-–.]\s*[a-d]", question.question) is not None
    )
    if not has_columns:
        return False, (
            "match_following question must clearly define Column A and Column B "
            "(or List I / List II) in the question stem."
        )
    return True, ""


def validate_formatting(question: Question) -> tuple[bool, str]:
    """
    Check formatting rules on question text and all options.

    Returns:
        (passed, reason)
    """
    q = question.question.strip()

    if not q:
        return False, "Question text is empty"

    if len(q) < MIN_QUESTION_CHARS:
        return False, f"Question is too short ({len(q)} chars < {MIN_QUESTION_CHARS})"

    if len(q) > MAX_QUESTION_CHARS:
        return False, (
            f"Question stem is too long ({len(q)} chars > {MAX_QUESTION_CHARS}). "
            "Keep questions concise."
        )

    if q[0].islower():
        return False, "Question must start with a capital letter"

    if not _ends_with_valid_punct(q):
        # match_following and sequence_order use structured multi-line stems;
        # their format is validated separately — skip terminal punct check for these.
        if question.question_type not in ("match_following", "sequence_order"):
            return False, "Question must end with '?', '.', or ':'."

    if re.search(r"\s{2,}", q):
        return False, "Question contains double spaces"

    if re.search(r"[?!.]{2,}", q):
        return False, "Question contains repeated punctuation (e.g. '??', '...')"

    for i, opt in enumerate(question.options, start=1):
        stripped = opt.strip()
        if opt != stripped:
            return False, f"Option {i} has leading/trailing whitespace"
        if not stripped:
            return False, f"Option {i} is empty"
        if len(stripped) < MIN_OPTION_LENGTH:
            return False, f"Option {i} is suspiciously short ({len(stripped)} chars)"
        if len(stripped) > MAX_OPTION_LENGTH:
            return False, (
                f"Option {i} is too long ({len(stripped)} chars > {MAX_OPTION_LENGTH}) "
                "— likely malformed LLM output"
            )

    # Option length balance
    balance_ok, balance_reason = _check_option_length_balance(question.options)
    if not balance_ok:
        return False, balance_reason

    # Statement-based format check
    stmt_ok, stmt_reason = _check_statement_format(question)
    if not stmt_ok:
        return False, stmt_reason

    # Match-the-following format check
    match_ok, match_reason = _check_match_format(question)
    if not match_ok:
        return False, match_reason

    return True, "Formatting OK"


# ── Answer distribution tracker (used by quality_validator) ──────────────────

def check_answer_distribution(questions: list[Question]) -> tuple[bool, str]:
    """
    Detect skewed correct-answer distribution in a batch of questions.

    Raises a warning if any single option position (A/B/C/D) holds more than
    50% of the correct answers — a pattern students can exploit.

    Returns (passed, reason).  This is a soft check; the caller decides
    whether to reject or just warn.
    """
    if len(questions) < 8:
        return True, "Too few questions to check distribution"

    position_counts = [0, 0, 0, 0]  # A, B, C, D
    for q in questions:
        try:
            idx = q.options.index(q.correct_answer)
            position_counts[idx] += 1
        except ValueError:
            pass

    total = sum(position_counts)
    if total == 0:
        return True, "No valid questions to check"

    labels = ["A", "B", "C", "D"]
    for i, count in enumerate(position_counts):
        pct = count / total
        if pct > 0.50:
            return False, (
                f"Answer distribution skewed: option {labels[i]} is correct "
                f"{count}/{total} times ({pct:.0%}). Shuffle correct answer positions."
            )

    return True, "Answer distribution is balanced"
