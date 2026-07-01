"""
Schema Validator
----------------
Pre-validates raw LLM dicts before they reach Pydantic.
Catches structural problems early with specific diagnostics so
malformed LLM output never silently becomes "0 questions parsed".
"""

REQUIRED_KEYS = {"question", "options", "correct_answer", "difficulty", "chapter"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


def validate_schema(raw_q: dict) -> tuple[bool, str]:
    """
    Validate the raw dict produced by the LLM before constructing a Question.

    Returns:
        (passed: bool, reason: str)
    """
    # 1. All required keys present
    missing = REQUIRED_KEYS - raw_q.keys()
    if missing:
        return False, f"Missing required keys: {missing}"

    # 2. options is a list of exactly 4 strings
    options = raw_q.get("options")
    if not isinstance(options, list):
        return False, f"'options' must be a list, got {type(options).__name__}"
    if len(options) != 4:
        return False, f"'options' must have exactly 4 items, got {len(options)}"
    for i, opt in enumerate(options):
        if not isinstance(opt, str) or not opt.strip():
            return False, f"Option {i + 1} must be a non-empty string"

    # 3. difficulty is valid
    difficulty = raw_q.get("difficulty")
    if difficulty not in VALID_DIFFICULTIES:
        return False, f"Invalid difficulty '{difficulty}'. Must be one of {VALID_DIFFICULTIES}"

    # 4. question is a non-trivially-short string
    question = raw_q.get("question")
    if not isinstance(question, str) or len(question.strip()) < 10:
        return False, "question must be a non-empty string (min 10 chars)"

    # 5. correct_answer is a non-empty string
    correct_answer = raw_q.get("correct_answer")
    if not isinstance(correct_answer, str) or not correct_answer.strip():
        return False, "correct_answer must be a non-empty string"

    # 6. correct_answer must be one of the options
    if correct_answer not in options:
        return False, f"correct_answer '{correct_answer}' is not in options {options}"

    return True, ""
