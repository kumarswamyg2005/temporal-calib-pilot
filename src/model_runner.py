"""DeepSeek-R1 inference behind a swappable model interface.

The pilot only ships :class:`HFReasoningModel` (DeepSeek-R1-Distill-Qwen-7B on
a single 16 GB GPU). The :class:`BaseModel` ABC and the module-level
``load_model`` / ``run_inference`` helpers keep the call sites stable so that
extra HF models (Qwen3, Llama 3.1) or a hosted-API backend
(:mod:`api_runner`) can be added later without touching the notebook.

Reproducibility: :func:`set_seed` fixes Python/NumPy/torch RNG. The seed is set
once at model load; a full top-to-bottom notebook run is therefore
deterministic for a fixed model + dataset.
"""

from __future__ import annotations

import gc
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Literal

from .config import log
from .prompts import (
    EMPTY_THINK_PREFILL,
    FORCE_EMPTY_THINK_FOR_OFF,
    Mode,
    build_user_message,
)

# Model selection. The 7B distill is too weak factually for the temporal
# hypothesis (uniform ~8-22% accuracy floor). Default is the 14B distill in
# fp16 sharded across 2 GPUs — same R1 family so ON-vs-OFF stays a within-model
# contrast, much better factual recall.
#
# Why fp16 (not 4-bit) by default: Kaggle's torch is bleeding-edge (cu128) and
# no bitsandbytes wheel reliably loads against it, so 4-bit kept failing
# transformers' is_bitsandbytes_available() check. fp16 needs no bitsandbytes
# at all; 14B fp16 (~28 GB) fits across the GPU **T4 x2** accelerator
# (2 x ~15 GB) via device_map="auto". 4-bit is still available where it works.
#
# Overridable via environment so a re-run needs no code edit:
#   TCP_MODEL_NAME       e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
#   TCP_LOAD_IN_4BIT     "1" to opt INTO 4-bit (only if bitsandbytes works)
#   TCP_MAX_NEW_TOKENS_ON  reasoning-budget override (default 1024; lower =
#                          faster. 14B at 2048 risks blowing the 60-min target)
DEFAULT_MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODEL_NAME = os.environ.get("TCP_MODEL_NAME", DEFAULT_MODEL_NAME)
LOAD_IN_4BIT = os.environ.get("TCP_LOAD_IN_4BIT", "0") == "1"
SEED = 42

# DeepSeek's recommended sampling settings for the R1 family.
TEMPERATURE = 0.6
TOP_P = 0.95
# ON budget is a *cap*, not a target: an early-stop halts generation the moment
# the model finishes its ANSWER/CONFIDENCE footer (see _FooterStop), so most
# answers stop well before this. 768 keeps the cap high enough that genuine
# reasoning is not truncated (7B run median ON length ~543 tokens) while
# bounding the slow CPU-offloaded worst case. Override via TCP_MAX_NEW_TOKENS_ON
# (do NOT go below ~512: that truncates reasoning before ANSWER: and corrupts
# the ON arm's parse_ok).
_ON_BUDGET = int(os.environ.get("TCP_MAX_NEW_TOKENS_ON", "768"))
MAX_NEW_TOKENS: dict[str, int] = {"thinking_on": _ON_BUDGET, "thinking_off": 256}

# Parsing patterns (case-insensitive, last occurrence wins).
_ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+?)\s*(?:\n|$)", re.IGNORECASE)
_CONF_RE = re.compile(r"CONFIDENCE\s*:\s*(\d{1,3})", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

# Small models often skip the CONFIDENCE: label and instead jam the number
# into the answer, e.g. "Spain <95>", "Real Madrid (90%)", "Macron 80/100".
# These patterns recover that confidence and let us strip it from the answer.
_INLINE_CONF_PATTERNS = [
    re.compile(r"<\s*(\d{1,3})\s*>"),
    re.compile(r"\(\s*(\d{1,3})\s*%?\s*\)"),
    re.compile(r"\b(\d{1,3})\s*/\s*100\b"),
    re.compile(r"(\d{1,3})\s*%"),
]

# Module-level model cache: load the 7B weights once, reuse for all 100 calls.
_MODEL_CACHE: dict[str, object] = {}


def set_seed(seed: int = SEED) -> None:
    """Fix RNG state across Python, NumPy, and torch for reproducibility.

    Args:
        seed: The seed value.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except ImportError:
        pass


def _total_vram_gb() -> float:
    """Return summed total VRAM across all visible CUDA devices, in GB."""
    import torch

    if not torch.cuda.is_available():
        return 0.0
    return sum(
        torch.cuda.get_device_properties(i).total_memory
        for i in range(torch.cuda.device_count())
    ) / 1e9


def cuda_memory_summary() -> str:
    """Return a short human-readable per-GPU memory usage string.

    Reports every visible GPU (the default 14B-fp16 setup shards across two),
    plus the summed total.

    Returns:
        str: e.g. ``"2x GPU | Tesla T4: 13.9/15.6 GB; Tesla T4: 13.7/15.6 GB
        | total 31.2 GB"`` or a CPU notice.
    """
    import torch

    if not torch.cuda.is_available():
        return "CUDA not available (running on CPU — expect this to be slow)."
    parts, total = [], 0.0
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        alloc = torch.cuda.memory_allocated(i) / 1e9
        cap = torch.cuda.get_device_properties(i).total_memory / 1e9
        total += cap
        parts.append(f"{name}: {alloc:.1f}/{cap:.1f} GB")
    return (
        f"{torch.cuda.device_count()}x GPU | "
        + "; ".join(parts)
        + f" | total {total:.1f} GB"
    )


def _estimate_vram_need_gb(model_name: str, load_in_4bit: bool) -> float:
    """Rough VRAM needed to load + run a model, in GB.

    fp16 ≈ 2 bytes/param plus headroom; 4-bit ≈ ~1/3.5 of that. Size is read
    from the parameter count in the repo id (``...-7B``, ``-14B``, ``-32B``).

    Args:
        model_name: HF repo id.
        load_in_4bit: Whether 4-bit quantization is used.

    Returns:
        float: Estimated VRAM requirement in GB.
    """
    import re as _re

    m = _re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", model_name)
    params_b = float(m.group(1)) if m else 8.0
    # Weights (~2 B/param) + modest headroom. device_map="auto" spills overflow
    # to CPU rather than OOM-ing, so keep the bar low enough that a genuine
    # T4 x2 (~31 GB) clears 14B fp16 (~30 GB) but a single T4 (~16 GB) fails.
    fp16_gb = params_b * 2.0 + 2.0
    return fp16_gb / 3.5 if load_in_4bit else fp16_gb


def _free_cuda() -> None:
    """Release cached CUDA blocks (used in the OOM retry path)."""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class BaseModel(ABC):
    """Minimal interface every model backend must implement.

    Concrete backends: :class:`HFReasoningModel` (this module) and the future
    :class:`api_runner.APIModel`.
    """

    @abstractmethod
    def load(self) -> None:
        """Load weights/clients and make the backend ready for inference."""

    @abstractmethod
    def run_inference(self, question: str, mode: str) -> dict:
        """Answer one question.

        Returns:
            dict: ``{"answer", "confidence", "full_response", "tokens_used"}``
            plus optional diagnostic keys.
        """


def load_model(model_name: str | None = None, load_in_4bit: bool | None = None):
    """Load (and cache) the HF model and tokenizer.

    Loaded once and reused for every inference call. Sets the reproducibility
    seed as a side effect. 4-bit (nf4) quantization via bitsandbytes lets the
    14B/32B distills fit a 16 GB GPU; pass ``load_in_4bit=False`` for a small
    model you want in fp16.

    Args:
        model_name: HF repo id. Defaults to :data:`MODEL_NAME`
            (env ``TCP_MODEL_NAME``, else DeepSeek-R1-Distill-Qwen-14B).
        load_in_4bit: Whether to 4-bit quantize. Defaults to
            :data:`LOAD_IN_4BIT` (env ``TCP_LOAD_IN_4BIT``).

    Returns:
        tuple: ``(model, tokenizer)``.

    Raises:
        RuntimeError: If weights cannot be loaded.
    """
    model_name = model_name or MODEL_NAME
    load_in_4bit = LOAD_IN_4BIT if load_in_4bit is None else load_in_4bit
    cache_key = (model_name, load_in_4bit)
    if _MODEL_CACHE.get("key") == cache_key:
        return _MODEL_CACHE["model"], _MODEL_CACHE["tokenizer"]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(SEED)

    # --- Preflight: fail fast with an actionable message BEFORE the multi-GB
    # download, instead of OOM-ing or hitting a confusing bitsandbytes error.
    have_gb = _total_vram_gb()
    need_gb = _estimate_vram_need_gb(model_name, load_in_4bit)
    log(
        f"Preflight: need ~{need_gb:.0f} GB "
        f"({'4-bit' if load_in_4bit else 'fp16'}), have {have_gb:.0f} GB "
        f"across visible GPUs."
    )
    if have_gb == 0.0:
        raise RuntimeError(
            "No CUDA GPU visible. In Kaggle Settings pick the GPU "
            "accelerator (use 'GPU T4 x2' for the 14B fp16 default)."
        )
    if have_gb + 1.0 < need_gb:
        raise RuntimeError(
            f"{model_name} ({'4-bit' if load_in_4bit else 'fp16'}) needs "
            f"~{need_gb:.0f} GB but only {have_gb:.0f} GB is visible. "
            "Fix one of: (a) set the accelerator to 'GPU T4 x2' (2x16 GB, "
            "fits 14B fp16); (b) use a smaller model via TCP_MODEL_NAME "
            "(e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B); or (c) if "
            "bitsandbytes works on this image, set TCP_LOAD_IN_4BIT=1."
        )

    # Best-effort: reduce allocator fragmentation before any CUDA alloc.
    # Must be set before torch.cuda is initialised to take full effect; in a
    # Kaggle notebook that's usually the GPU-check cell, so this fires too late
    # on a warm kernel — but it is a no-op rather than an error in that case.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    kwargs: dict = {"device_map": "auto", "low_cpu_mem_usage": True}
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
            from transformers.utils import is_bitsandbytes_available

            if not is_bitsandbytes_available():
                raise ImportError("transformers cannot see a usable bitsandbytes")
        except ImportError as exc:
            raise RuntimeError(
                "4-bit requested (TCP_LOAD_IN_4BIT=1) but bitsandbytes is not "
                f"usable on this image ({exc}). Kaggle's torch (cu128) often "
                "has no compatible bitsandbytes wheel. Use the fp16 default "
                "(unset TCP_LOAD_IN_4BIT) with the 'GPU T4 x2' accelerator."
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        log(f"Loading {model_name} (4-bit nf4, device_map=auto)...")
    else:
        kwargs["torch_dtype"] = torch.float16
        # Cap per-GPU weight allocation so accelerate leaves headroom for the
        # lm_head dispatch during generation.  Without a cap, device_map="auto"
        # fills GPU 0 to ~96% (12.85/14.56 GB on T4); the 1.5 GB lm_head
        # tensor then cannot be moved there and OOMs on the first forward pass.
        # Setting GPU 0 to 12 GiB frees ~2.5 GB for activation / lm_head
        # dispatch; excess weights (~2 GB) spill to CPU, which is tolerable.
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_gpus >= 2:
            kwargs["max_memory"] = {0: "12GiB", 1: "14GiB", "cpu": "20GiB"}
            log("fp16 multi-GPU: capping GPU 0 at 12 GiB to leave lm_head headroom.")
        elif n_gpus == 1:
            kwargs["max_memory"] = {0: "14GiB", "cpu": "20GiB"}
        log(f"Loading {model_name} (fp16, device_map=auto)...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except Exception as exc:  # broad: surface any HF/torch load failure clearly
        raise RuntimeError(
            f"Failed to load {model_name}. On Kaggle ensure Settings -> "
            f"'Internet' is on and a GPU accelerator is selected. Note the "
            f"full fp16 weights download before 4-bit quantization "
            f"(14B ~28 GB, 32B ~65 GB). ({exc})"
        ) from exc

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    _MODEL_CACHE.update(
        {"model": model, "tokenizer": tokenizer, "name": model_name, "key": cache_key}
    )
    log(f"Model loaded. {cuda_memory_summary()}")
    return model, tokenizer


def _split_reasoning(text: str) -> tuple[str, str]:
    """Split a raw generation into (reasoning, answer-region).

    Args:
        text: Decoded generated text.

    Returns:
        tuple[str, str]: ``(reasoning_text, post_think_text)``. If no
        ``</think>`` marker is present, reasoning is empty and the whole text
        is treated as the answer region.
    """
    m = _THINK_CLOSE_RE.search(text)
    if not m:
        return "", text
    return text[: m.start()], text[m.end():]


def _parse(text: str) -> tuple[str, int, bool]:
    """Extract ``(answer, confidence, parse_ok)`` from the answer region.

    Args:
        text: The post-reasoning answer region.

    Returns:
        tuple[str, int, bool]: Parsed answer, confidence clamped to 0-100,
        and whether the expected ANSWER line was found.
    """
    ans_matches = list(_ANSWER_RE.finditer(text))
    conf_matches = list(_CONF_RE.finditer(text))
    has_answer_label = bool(ans_matches)

    if ans_matches:
        answer = ans_matches[-1].group(1).strip()
    else:
        # Fallback: last non-empty line that isn't the confidence line.
        lines = [
            ln.strip()
            for ln in text.strip().splitlines()
            if ln.strip() and not _CONF_RE.search(ln)
        ]
        answer = lines[-1] if lines else ""

    # Confidence priority: explicit CONFIDENCE: line, else an inline marker
    # recovered (and stripped) from the answer text, else a neutral prior.
    explicit_conf = (
        max(0, min(100, int(conf_matches[-1].group(1))))
        if conf_matches
        else None
    )
    answer, inline_conf = _strip_inline_confidence(answer)

    if explicit_conf is not None:
        confidence = explicit_conf
    elif inline_conf is not None:
        confidence = inline_conf
    else:
        confidence = 50  # neutral prior when the model gives no number

    # parse_ok = the model produced a structured answer we can trust: an
    # ANSWER label AND some confidence value. (This stays an honest measure
    # of format compliance — refusals/rambles still count as not-ok.)
    parse_ok = has_answer_label and (
        explicit_conf is not None or inline_conf is not None
    )

    # Strip stray markdown/quotes the model sometimes wraps the answer in.
    answer = answer.strip().strip("*").strip('"').strip("'").strip()
    return answer, confidence, parse_ok


def _strip_inline_confidence(answer: str) -> tuple[str, int | None]:
    """Recover and remove an inline confidence marker from an answer string.

    Handles forms like ``"Spain <95>"``, ``"Real Madrid (90%)"``,
    ``"Macron 80/100"``, ``"Microsoft 75%"``.

    Args:
        answer: The raw extracted answer text.

    Returns:
        tuple[str, int | None]: The answer with the marker removed, and the
        recovered confidence in ``0..100`` (or ``None`` if no marker found).
    """
    for pat in _INLINE_CONF_PATTERNS:
        m = pat.search(answer)
        if not m:
            continue
        val = int(m.group(1))
        if 0 <= val <= 100:
            cleaned = (answer[: m.start()] + answer[m.end():]).strip()
            return cleaned, val
    return answer, None


def _build_footer_stop(tokenizer, prompt_len: int):
    """Build a StoppingCriteria that halts once the ANSWER/CONFIDENCE footer
    is complete, so a finished answer does not run to the token cap.

    Stops only when BOTH ``ANSWER:`` and ``CONFIDENCE: <digits>`` appear in the
    generated tail — both together at the end is the required footer and is
    very unlikely to occur inside the reasoning, so this does not truncate
    genuine answers.

    Args:
        tokenizer: The model tokenizer (to decode the tail).
        prompt_len: Number of prompt tokens to skip when decoding.

    Returns:
        transformers.StoppingCriteria: The configured criterion.
    """
    from transformers import StoppingCriteria

    footer_re = re.compile(
        r"answer\s*:\s*\S.*?confidence\s*:\s*\d{1,3}", re.IGNORECASE | re.DOTALL
    )

    class _FooterStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs) -> bool:
            gen = input_ids[0][prompt_len:]
            if gen.shape[0] < 8:
                return False
            tail = tokenizer.decode(gen[-80:], skip_special_tokens=True)
            return bool(footer_re.search(tail))

    return _FooterStop()


def _generate(model, tokenizer, prompt_text: str, max_new_tokens: int) -> tuple[str, int]:
    """Run a single sampling pass and return (new_text, num_new_tokens).

    Inputs go on ``cuda:0`` when a GPU is present: with ``device_map="auto"``
    plus CPU offload the first parameter can be on the ``meta`` device, so
    keying off ``next(model.parameters()).device`` would break — accelerate's
    hooks dispatch across devices from cuda:0.
    """
    import torch
    from transformers import StoppingCriteriaList

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [_build_footer_stop(tokenizer, prompt_len)]
            ),
        )
    new_ids = out[0][prompt_len:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return text, int(new_ids.shape[0])


def run_inference(
    question: str,
    mode: Literal["thinking_on", "thinking_off"],
    model,
    tokenizer,
) -> dict:
    """Answer one question in one experimental condition.

    Builds a chat prompt (instruction + question in the user turn, per
    DeepSeek's guidance). For the thinking-OFF condition an empty ``<think>``
    block is prefilled so the distilled model actually skips reasoning instead
    of ignoring the instruction. Retries once on CUDA OOM with a reduced token
    budget.

    Args:
        question: The factual question.
        mode: ``"thinking_on"`` or ``"thinking_off"``.
        model: A loaded HF causal LM.
        tokenizer: Its tokenizer.

    Returns:
        dict: ``{"answer": str, "confidence": int, "full_response": str,
        "tokens_used": int}`` plus ``"parse_ok"`` and ``"reasoning_chars"``
        diagnostics.

    Raises:
        ValueError: If ``mode`` is unrecognised.
    """
    if mode not in MAX_NEW_TOKENS:
        raise ValueError(f"Unknown mode {mode!r}")

    user_msg = build_user_message(question, mode)  # type: ignore[arg-type]
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_msg}],
        tokenize=False,
        add_generation_prompt=True,
    )
    if mode == "thinking_off" and FORCE_EMPTY_THINK_FOR_OFF:
        prompt_text += EMPTY_THINK_PREFILL

    budget = MAX_NEW_TOKENS[mode]
    try:
        text, n_tokens = _generate(model, tokenizer, prompt_text, budget)
    except (RuntimeError, MemoryError) as exc:
        if "out of memory" not in str(exc).lower():
            raise
        log("CUDA OOM — clearing cache and retrying once with half budget.")
        _free_cuda()
        text, n_tokens = _generate(
            model, tokenizer, prompt_text, max(64, budget // 2)
        )

    reasoning, answer_region = _split_reasoning(text)
    answer, confidence, parse_ok = _parse(answer_region)

    return {
        "answer": answer,
        "confidence": confidence,
        "full_response": text,
        "tokens_used": n_tokens,
        "parse_ok": parse_ok,
        "reasoning_chars": len(reasoning),
    }


class HFReasoningModel(BaseModel):
    """:class:`BaseModel` adapter over the cached HF DeepSeek-R1 backend."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        """Load and cache the weights (idempotent)."""
        self._model, self._tokenizer = load_model(self.model_name)

    def run_inference(self, question: str, mode: str) -> dict:
        """Answer one question; loads the model lazily if needed."""
        if self._model is None:
            self.load()
        return run_inference(
            question, mode, self._model, self._tokenizer  # type: ignore[arg-type]
        )


def get_logprob_confidence(*args, **kwargs) -> float:
    """Stub: token-logprob-derived confidence (more reliable than verbalized).

    Planned post-pilot: return ``exp(mean log-prob)`` of the answer span (or a
    calibrated transform of it) instead of the model's self-reported number.
    Requires ``output_scores=True`` / ``return_dict_in_generate=True`` in
    :func:`_generate` and answer-span token alignment.

    Raises:
        NotImplementedError: Always — not part of the pilot.
    """
    raise NotImplementedError(
        "Log-probability confidence is a post-pilot enhancement (see docstring)."
    )
