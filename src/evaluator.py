"""Answer-correctness matching.

Verbalized free-text answers need lenient comparison: case, punctuation, and
minor surface variation should not count as wrong. The strategy, in order:

1. Normalise (lowercase, strip punctuation/whitespace, collapse spaces).
2. Exact match against the canonical answer or any alias.
3. Containment either direction (prediction inside an alias or vice versa),
   guarded by a minimum length so trivial substrings don't false-positive.
4. Fuzzy ratio (rapidfuzz) against the best alias, threshold 85.

If rapidfuzz is unavailable the function degrades gracefully to steps 1-3.
"""

from __future__ import annotations

import re
import string

try:
    from rapidfuzz import fuzz

    _HAVE_RAPIDFUZZ = True
except ImportError:  # pragma: no cover - exercised only on minimal installs
    _HAVE_RAPIDFUZZ = False

_FUZZY_THRESHOLD = 85
# Below this normalised length, containment is too weak to trust.
_MIN_CONTAINMENT_LEN = 4

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace.

    Args:
        text: Raw answer text.

    Returns:
        str: Normalised comparison key (possibly empty).
    """
    text = text.strip().lower().translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_correct(predicted: str, true_answer: str, aliases: list[str]) -> bool:
    """Decide whether a predicted answer matches the ground truth.

    Args:
        predicted: The model's extracted answer string.
        true_answer: The canonical correct answer.
        aliases: Additional acceptable surface forms.

    Returns:
        bool: ``True`` if the prediction is judged correct.
    """
    pred = normalize(predicted)
    if not pred:
        return False

    candidates = {normalize(true_answer)}
    candidates.update(normalize(a) for a in aliases)
    candidates.discard("")
    if not candidates:
        return False

    # 2. Exact match.
    if pred in candidates:
        return True

    # 3. Containment in either direction, length-guarded.
    for cand in candidates:
        if len(pred) >= _MIN_CONTAINMENT_LEN and pred in cand:
            return True
        if len(cand) >= _MIN_CONTAINMENT_LEN and cand in pred:
            return True

    # 4. Fuzzy fallback.
    if _HAVE_RAPIDFUZZ:
        best = max(fuzz.ratio(pred, cand) for cand in candidates)
        if best >= _FUZZY_THRESHOLD:
            return True

    return False
