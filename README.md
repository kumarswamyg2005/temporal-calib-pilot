# Temporal Calibration Pilot

A research pilot testing whether a reasoning-mode LLM becomes **overconfident**
on factual questions as the questioned events get closer to its training
cutoff. It runs 50 hand-curated questions through a DeepSeek-R1 distill with
thinking **ON** vs **OFF**, then plots the overconfidence gap against temporal
distance from the July 2024 cutoff.

**Model:** the 7B distill was tried first and proved too weak factually
(uniform ~8–22% accuracy, no temporal gradient — a floor effect, see "Run
history" below). The default is now **DeepSeek-R1-Distill-Qwen-14B in 4-bit**
(same R1 family → ON-vs-OFF is still a within-model contrast; ~10 GB, fits one
16 GB GPU). Change model with **no code edit** via environment variables:

| env var | default | purpose |
| --- | --- | --- |
| `TCP_MODEL_NAME` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | any HF R1-distill repo |
| `TCP_LOAD_IN_4BIT` | `1` | `0` = fp16 (only for small models) |
| `TCP_MAX_NEW_TOKENS_ON` | `1024` | reasoning budget; lower = faster |

Set them in the first notebook cell, e.g. `import os; os.environ["TCP_MODEL_NAME"]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"` (32B needs the T4×2 accelerator).

## How to run it on Kaggle

1. Create a new Kaggle Notebook.
2. **Settings → Accelerator**: choose **GPU T4 x2** or **GPU P100** (16 GB).
3. **Settings → Internet**: turn **On** (needed to download model weights).
4. Get the project files into the kernel (either is fine):
   - Upload this repo folder as a **Kaggle Dataset** and attach it, **or**
   - Add a first cell that `git clone`s / copies the repo into the working dir.
5. Open `notebooks/pilot_study.ipynb` and click **Run All**.
6. When it finishes, find the artifacts in **`/kaggle/working/outputs/`**:
   `results.csv`, `summary_table.csv`, `pilot_chart.png`. These persist as
   notebook output (Save Version → they are downloadable).

No Google Drive mount is required. The notebook auto-detects Kaggle and writes
everything to `/kaggle/working/` so it survives as output.

> Local / Colab also work: the code auto-detects the platform and falls back to
> the repo's `outputs/` directory.

## What the output means

Files are namespaced per model (e.g. `__DeepSeek-R1-Distill-Qwen-14B`) so
multiple model runs coexist without clobbering or false-resuming each other.

- **`results__<model>.csv`** — 100 rows (50 questions × 2 modes). Per-call
  answer, parsed confidence, correctness, `parse_ok`, token count, `model`.
- **`summary_table__<model>.csv`** — per `(bucket, mode)`: accuracy, mean
  confidence, **overconfidence gap**, and ECE.
- **`pilot_chart__<model>.png`** — the headline figure. Two lines (Thinking
  ON / OFF) of overconfidence gap vs months-to-cutoff (log x-axis, auto-scaled
  y so a very overconfident model does not clip off-chart).

**Overconfidence gap = mean_confidence − accuracy × 100.**
Positive = overconfident · 0 = perfectly calibrated · negative = underconfident.

The hypothesis is supported if the **Thinking ON** line rises (more
overconfident) as distance shrinks toward the cutoff, more so than Thinking
OFF.

## How "thinking OFF" works here

DeepSeek-R1 distilled models emit a `<think>` chain even when told not to. To
make the OFF condition real, the runner prefills an empty `<think></think>`
block so the model skips reasoning. This is controlled by
`FORCE_EMPTY_THINK_FOR_OFF` in `src/prompts.py` and documented there.

**Caveat:** R1-distills are RL-trained to always reason, so the suppressed-OFF
condition has poor format compliance (the 7B run followed the
`ANSWER:/CONFIDENCE:` format only 16/50 times in OFF). The parser now also
recovers inline confidences like `"Spain <95>"` / `"Real Madrid (90%)"`, and
`parse_ok` in the CSV records true format compliance — **check the OFF
`parse_ok` rate before trusting the OFF arm**. A cleaner non-reasoning baseline
(a plain instruct model) is the right follow-up if OFF compliance stays low.

## Note on the question set

The build spec's worked example asked for the 2024 UEFA Euro final — but that
final was played **2024-07-14**, on/after the stated July 2024 cutoff, so its
ground truth is unsafe for a calibration study. It was replaced with the
**2024 UEFA Champions League final (2024-06-01)**, which is unambiguously
pre-cutoff. Every answer in `data/pilot_questions.json` is verified to predate
the cutoff (the loader enforces this).

## Run history

- **7B distill (first run) — negative.** DeepSeek-R1-Distill-Qwen-7B:
  accuracy ≈ 22% (ON) / 8% (OFF), confidently hallucinating (e.g. "Apple M4
  chip at WWDC 2022", "Sony acquired Activision"). Overconfidence gap was
  35–79 pp at *every* bucket with **no rise toward the cutoff** — a floor
  effect, not a temporal signal. Verdict: *pivot to a stronger model.*
  Archived artifacts: `notebooks/run_7B/`.
- **14B distill (4-bit) — current default.** Re-run to test whether real
  factual headroom exposes a temporal calibration gradient.

## Known limitations of the pilot

- **Floor effect risk** — if model accuracy is near zero everywhere, the
  overconfidence "gap" is uninformative (this killed the 7B run). The notebook
  now prints a WARNING when accuracy < 30% in both modes.
- **Single model family per run.** No cross-model claims; the ON/OFF contrast
  is within one R1-distill.
- **N = 50** questions, 10 per bucket — too small for significance; this is a
  signal-detection pilot, not a confirmatory study.
- **OFF-condition validity** — see the caveat above; suppressed reasoning on an
  R1-distill is methodologically awkward.
- **Verbalized confidence is noisy** — self-reported numbers are not
  well-calibrated probabilities. A token-logprob confidence is stubbed for a
  follow-up (`get_logprob_confidence`).
- **Cutoff is approximate** — "July 2024" is the reported, not exactly known,
  cutoff; near-boundary buckets are inherently fuzzy.
- **Single sampled generation** per item (temperature 0.6). Run-level
  reproducible via fixed seed, but not multi-sample averaged.

## Project layout

```
data/pilot_questions.json   50 curated questions (read-only input)
src/                        config, prompts, loader, runner, evaluator,
                            metrics, visualizer, api_runner (stub)
notebooks/pilot_study.ipynb one-click end-to-end entry point
outputs/                    results.csv, summary_table.csv, pilot_chart.png
                            (on Kaggle: /kaggle/working/outputs/)
```

## Future-proofing (designed for, not built)

- More HF models: implement another `BaseModel` in `model_runner.py`.
- Paid APIs (GPT-4o, Claude): fill in `src/api_runner.py` (`APIModel`).
- Scale to ~1,200 questions: dataset format already supports it.
- Logprob confidence: implement `get_logprob_confidence`.
