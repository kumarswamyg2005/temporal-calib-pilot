"""Load and validate the curated pilot question set.

The on-disk format is JSON with a ``meta`` block and a ``questions`` array (a
bare top-level array is also accepted for forward compatibility with larger
scaled-up sets). Validation is strict: a malformed dataset should fail loudly
here rather than silently corrupt the study.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .config import get_questions_path

# Bucket -> expected temporal distance in months from the training cutoff.
BUCKET_DISTANCE: dict[str, int] = {"B1": 24, "B2": 12, "B3": 6, "B4": 3, "B5": 1}
ALLOWED_CATEGORIES: set[str] = {
    "sports",
    "politics",
    "tech",
    "science",
    "entertainment",
}
# DeepSeek-R1-Distill-Qwen-7B reported knowledge cutoff.
TRAINING_CUTOFF: date = date(2024, 7, 1)
EXPECTED_PER_BUCKET: int = 10


@dataclass(frozen=True)
class Question:
    """A single curated factual question.

    Attributes:
        id: Stable identifier, e.g. ``"Q001"``.
        bucket: One of ``B1``..``B5`` (temporal-distance bucket).
        distance_months: Nominal months between the event and the cutoff.
        event_date: ISO date the event occurred (always before the cutoff).
        category: One of :data:`ALLOWED_CATEGORIES`.
        question: The natural-language question text.
        answer: The canonical correct answer.
        answer_aliases: Acceptable answer surface forms for flexible matching.
        source: Provenance string for auditability.
    """

    id: str
    bucket: str
    distance_months: int
    event_date: str
    category: str
    question: str
    answer: str
    answer_aliases: list[str] = field(default_factory=list)
    source: str = ""


def _validate(questions: list[Question]) -> None:
    """Run strict structural and temporal checks on the question set.

    Args:
        questions: Parsed question objects.

    Raises:
        ValueError: On any duplicate id, unknown bucket/category, distance
            mismatch, post-cutoff event date, empty answer, or wrong count.
    """
    if not questions:
        raise ValueError("Question set is empty.")

    ids = [q.id for q in questions]
    dupes = [i for i, c in Counter(ids).items() if c > 1]
    if dupes:
        raise ValueError(f"Duplicate question ids: {sorted(dupes)}")

    for q in questions:
        if q.bucket not in BUCKET_DISTANCE:
            raise ValueError(f"{q.id}: unknown bucket {q.bucket!r}")
        if q.distance_months != BUCKET_DISTANCE[q.bucket]:
            raise ValueError(
                f"{q.id}: distance_months={q.distance_months} does not match "
                f"bucket {q.bucket} (expected {BUCKET_DISTANCE[q.bucket]})"
            )
        if q.category not in ALLOWED_CATEGORIES:
            raise ValueError(f"{q.id}: unknown category {q.category!r}")
        if not q.question.strip():
            raise ValueError(f"{q.id}: empty question text")
        if not q.answer.strip():
            raise ValueError(f"{q.id}: empty answer")
        try:
            ev = date.fromisoformat(q.event_date)
        except ValueError as exc:
            raise ValueError(f"{q.id}: bad event_date {q.event_date!r}") from exc
        if ev >= TRAINING_CUTOFF:
            raise ValueError(
                f"{q.id}: event_date {q.event_date} is on/after the training "
                f"cutoff {TRAINING_CUTOFF.isoformat()} — unsafe ground truth"
            )

    per_bucket = Counter(q.bucket for q in questions)
    for bucket in BUCKET_DISTANCE:
        n = per_bucket.get(bucket, 0)
        if n != EXPECTED_PER_BUCKET:
            raise ValueError(
                f"Bucket {bucket} has {n} questions; expected "
                f"{EXPECTED_PER_BUCKET}."
            )


def load_questions(path: Path | None = None) -> list[Question]:
    """Load, parse, and validate the pilot question set.

    Args:
        path: Optional explicit path. Defaults to the location resolved by
            :func:`config.get_questions_path` (local checkout or Kaggle input).

    Returns:
        list[Question]: Validated questions in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is malformed or validation fails.
    """
    qpath = path or get_questions_path()
    try:
        raw = json.loads(qpath.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"{qpath} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Could not read {qpath}: {exc}") from exc

    records = raw["questions"] if isinstance(raw, dict) else raw
    if not isinstance(records, list):
        raise ValueError(
            f"{qpath}: expected a list of questions (or a dict with a "
            "'questions' list)."
        )

    questions: list[Question] = []
    for i, rec in enumerate(records):
        try:
            questions.append(
                Question(
                    id=rec["id"],
                    bucket=rec["bucket"],
                    distance_months=int(rec["distance_months"]),
                    event_date=rec["event_date"],
                    category=rec["category"],
                    question=rec["question"],
                    answer=rec["answer"],
                    answer_aliases=list(rec.get("answer_aliases", [])),
                    source=rec.get("source", ""),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Record #{i} is malformed: {exc}") from exc

    _validate(questions)
    return questions


def sanity_summary(questions: list[Question]) -> str:
    """Build a human-readable per-bucket / per-category count summary.

    Args:
        questions: Validated questions.

    Returns:
        str: A multi-line summary suitable for printing in the notebook.
    """
    per_bucket = Counter(q.bucket for q in questions)
    lines = [f"Loaded {len(questions)} questions."]
    for bucket in sorted(BUCKET_DISTANCE, key=lambda b: BUCKET_DISTANCE[b]):
        cats = Counter(
            q.category for q in questions if q.bucket == bucket
        )
        cat_str = ", ".join(f"{k}:{v}" for k, v in sorted(cats.items()))
        lines.append(
            f"  {bucket} ({BUCKET_DISTANCE[bucket]:>2}mo): "
            f"{per_bucket[bucket]:>2} | {cat_str}"
        )
    return "\n".join(lines)
