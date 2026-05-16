"""Stub for paid-API model backends (GPT-4o, Claude, ...).

Deliberately NOT implemented for the pilot — the pilot is API-cost-free and
DeepSeek-only. This module exists so that adding hosted models later is a
drop-in: implement :class:`APIModel` against the same :class:`BaseModel`
interface used by :mod:`model_runner` and route calls through it.

Design intent:
    * Read keys from environment (python-dotenv is already a dependency).
    * Mirror ``run_inference``'s return contract:
      ``{"answer", "confidence", "full_response", "tokens_used"}``.
    * Keep rate-limiting / retry / cost-accounting here, not in the runner.
"""

from __future__ import annotations

from .model_runner import BaseModel


class APIModel(BaseModel):
    """Placeholder hosted-API model. Not available in the pilot."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def load(self) -> None:  # noqa: D102 - see BaseModel
        raise NotImplementedError(
            "Paid-API backends are out of scope for the pilot. Implement this "
            "module after the pilot validates the hypothesis."
        )

    def run_inference(self, question: str, mode: str) -> dict:  # noqa: D102
        raise NotImplementedError(
            "Paid-API backends are out of scope for the pilot."
        )
