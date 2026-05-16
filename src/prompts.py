"""Instruction prompts for the two experimental conditions.

The pilot contrasts a *thinking ON* condition (the model reasons step by step
before answering) against a *thinking OFF* condition (the model answers
directly). Both conditions must emit a parseable ``ANSWER`` / ``CONFIDENCE``
footer so the evaluator and calibration metrics can consume them.

DeepSeek's own guidance for the R1 distilled models is to avoid a separate
system prompt and place all instructions in the user turn. These templates are
therefore prepended to the question inside the user message by
:mod:`model_runner`, not sent as a system role.
"""

from __future__ import annotations

from typing import Literal

Mode = Literal["thinking_on", "thinking_off"]

THINKING_ON_PROMPT: str = (
    "You are answering a factual question. Think through what you know step by "
    "step, then give your final answer. End your response with exactly this "
    "format on the final line:\n\n"
    "ANSWER: <your answer>\n"
    "CONFIDENCE: <0-100>"
)

THINKING_OFF_PROMPT: str = (
    "Answer the following factual question directly without reasoning or "
    "explanation. Respond using exactly this format:\n\n"
    "ANSWER: <your answer>\n"
    "CONFIDENCE: <0-100>"
)

# Shared clarification appended to both so "confidence" is unambiguous.
_CONFIDENCE_GLOSS: str = (
    "\n\nCONFIDENCE is how certain you are that your answer is correct, where "
    "0 means a pure guess and 100 means absolutely certain."
)

# DeepSeek-R1 distilled models do not reliably obey a plain "do not reason"
# instruction — they emit a <think> chain regardless, which would invalidate
# the thinking-OFF condition. The runner suppresses reasoning for the OFF
# condition by prefilling an empty think block (see model_runner). This flag
# documents and controls that behaviour.
FORCE_EMPTY_THINK_FOR_OFF: bool = True

# Prefill that, appended after the generation prompt, makes R1 skip reasoning.
EMPTY_THINK_PREFILL: str = "<think>\n\n</think>\n\n"


def get_instruction(mode: Mode) -> str:
    """Return the instruction block for a given experimental condition.

    Args:
        mode: ``"thinking_on"`` or ``"thinking_off"``.

    Returns:
        str: The instruction text (without the question appended).

    Raises:
        ValueError: If ``mode`` is not a recognised condition.
    """
    if mode == "thinking_on":
        return THINKING_ON_PROMPT + _CONFIDENCE_GLOSS
    if mode == "thinking_off":
        return THINKING_OFF_PROMPT + _CONFIDENCE_GLOSS
    raise ValueError(f"Unknown mode {mode!r}; expected 'thinking_on' or 'thinking_off'.")


def build_user_message(question: str, mode: Mode) -> str:
    """Compose the full user-turn text: instruction followed by the question.

    Args:
        question: The factual question to ask.
        mode: The experimental condition.

    Returns:
        str: Text to place in the user role of the chat template.
    """
    return f"{get_instruction(mode)}\n\nQuestion: {question.strip()}"
