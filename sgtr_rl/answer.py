"""Answer extraction for SGTR binary classification."""

import re


def extract_answer(text: str) -> str | None:
    """Extract the answer token from a completion.

    Tries several strategies in order:
    1. Explicit "Answer: <digit>" pattern (case-insensitive)
    2. Last occurrence of a standalone "1" or "2" in the text
    3. If the entire (stripped) text is just "1" or "2"

    Returns:
        "1" or "2" if found, None otherwise.
    """
    text = text.strip()

    # Strategy 1: explicit answer pattern
    match = re.search(r"(?i)answer\s*[:=]\s*([12])", text)
    if match:
        return match.group(1)

    # Strategy 2: last standalone "1" or "2" (word boundary)
    matches = re.findall(r"\b([12])\b", text)
    if matches:
        return matches[-1]

    # Strategy 3: bare single-character response
    if text in ("1", "2"):
        return text

    return None
