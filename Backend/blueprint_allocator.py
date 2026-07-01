"""
Blueprint Allocator
-------------------
For each exam set, determines how many questions to generate per chapter
such that:
  - Total across all chapters == QUESTIONS_PER_EXAM (100)
  - Each chapter's count is within [min_q, max_q] from the blueprint
  - Different sets get slightly different distributions (within allowed range)
    to ensure variety across sets
"""

import random
from config import CHAPTER_BLUEPRINT, QUESTIONS_PER_EXAM


def allocate_questions(set_number: int) -> dict[str, int]:
    """
    Returns a mapping of chapter_key → num_questions for one exam set.

    Strategy:
    1. Start with each chapter's min_q.
    2. Distribute the remaining budget randomly across chapters
       without exceeding their max_q.
    3. Seed random with (set_number) so allocations are reproducible
       but different across sets.
    """
    rng = random.Random(set_number * 42)  # deterministic per set

    chapters = list(CHAPTER_BLUEPRINT.keys())
    allocation: dict[str, int] = {ch: CHAPTER_BLUEPRINT[ch]["min_q"] for ch in chapters}

    budget = QUESTIONS_PER_EXAM - sum(allocation.values())

    # Shuffle to distribute budget fairly across chapters
    shuffled = chapters[:]
    rng.shuffle(shuffled)

    for ch in shuffled:
        if budget <= 0:
            break
        headroom = CHAPTER_BLUEPRINT[ch]["max_q"] - allocation[ch]
        if headroom > 0:
            add = rng.randint(0, min(headroom, budget))
            allocation[ch] += add
            budget -= add

    # If there's still budget (rng left some), force-fill from beginning
    if budget > 0:
        for ch in chapters:
            if budget <= 0:
                break
            headroom = CHAPTER_BLUEPRINT[ch]["max_q"] - allocation[ch]
            if headroom > 0:
                give = min(headroom, budget)
                allocation[ch] += give
                budget -= give

    total = sum(allocation.values())
    assert total == QUESTIONS_PER_EXAM, (
        f"Blueprint allocation error: total={total}, expected={QUESTIONS_PER_EXAM}. "
        f"Allocation: {allocation}"
    )

    return allocation


def allocation_summary(set_number: int) -> str:
    """Human-readable summary of the allocation for a set."""
    alloc = allocate_questions(set_number)
    lines = [f"Exam Set {set_number} — Question Allocation:"]
    total = 0
    for ch, count in alloc.items():
        info = CHAPTER_BLUEPRINT[ch]
        lines.append(f"  {ch}: {count}  (blueprint: {info['min_q']}–{info['max_q']})")
        total += count
    lines.append(f"  TOTAL: {total}")
    return "\n".join(lines)
