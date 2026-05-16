"""Path resolution that adapts to Kaggle, Colab, or a local checkout.

The pilot writes its CSV and PNG outputs to a directory that *persists* as
notebook output. On Kaggle that must be ``/kaggle/working``; locally it is the
repository's ``outputs/`` folder.

The question set is read-only input. On Kaggle it may live either inside the
uploaded notebook's working copy of the repo or in an attached dataset under
``/kaggle/input``; :func:`get_questions_path` searches both.

All functions return :class:`pathlib.Path` objects, never strings.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path


def log(message: str) -> None:
    """Print a timestamped progress line to stdout.

    Format matches the spec, e.g. ``[12:34:56] Running question 23/50``.

    Args:
        message: The message to log.
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

# Repository root = parent of the directory containing this file (src/).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


def on_kaggle() -> bool:
    """Return ``True`` when running inside a Kaggle notebook kernel.

    Returns:
        bool: ``True`` if Kaggle environment markers are present.
    """
    return (
        "KAGGLE_KERNEL_RUN_TYPE" in os.environ
        or "KAGGLE_URL_BASE" in os.environ
        or Path("/kaggle/working").is_dir()
    )


def on_colab() -> bool:
    """Return ``True`` when running inside a Google Colab runtime.

    Returns:
        bool: ``True`` if the ``google.colab`` module is importable.
    """
    try:
        import google.colab  # noqa: F401  (import is the probe)

        return True
    except ImportError:
        return False


def get_outputs_dir() -> Path:
    """Return the directory for persisted outputs (CSV, PNG), creating it.

    On Kaggle this is ``/kaggle/working/outputs`` so artifacts survive as
    committed notebook output. Locally it is ``<repo>/outputs``.

    Returns:
        Path: An existing, writable output directory.
    """
    base = Path("/kaggle/working") if on_kaggle() else PROJECT_ROOT
    out = base / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def slugify_model(model_name: str) -> str:
    """Turn an HF repo id into a filename-safe slug.

    Args:
        model_name: e.g. ``"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"``.

    Returns:
        str: e.g. ``"deepseek-ai_DeepSeek-R1-Distill-Qwen-14B"``.
    """
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in model_name)


def get_results_path(model_name: str | None = None) -> Path:
    """Return the path for the per-call results CSV.

    Kept under the persisted outputs directory so partial progress is not
    lost if a Kaggle kernel is interrupted. When ``model_name`` is given the
    file is namespaced per model, so switching models (e.g. 7B -> 14B) does
    NOT make the resume logic skip a fresh run against the stale file.

    Args:
        model_name: Optional HF repo id to namespace the file by.

    Returns:
        Path: Destination path for the results CSV.
    """
    if model_name:
        return get_outputs_dir() / f"results__{slugify_model(model_name)}.csv"
    return get_outputs_dir() / "results.csv"


def get_questions_path() -> Path:
    """Locate the read-only ``pilot_questions.json`` dataset.

    Search order:
        1. ``<repo>/data/pilot_questions.json`` (local checkout / uploaded repo)
        2. Any ``pilot_questions.json`` under ``/kaggle/input`` (attached dataset)

    Returns:
        Path: Path to the question set.

    Raises:
        FileNotFoundError: If the question set cannot be located anywhere.
    """
    local = PROJECT_ROOT / "data" / "pilot_questions.json"
    if local.is_file():
        return local

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        matches = sorted(kaggle_input.rglob("pilot_questions.json"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "Could not find 'pilot_questions.json'. Expected it at "
        f"{local} or somewhere under /kaggle/input. If running on Kaggle, "
        "make sure the repo files are present in the working directory or "
        "attached as a dataset."
    )
