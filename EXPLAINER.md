# Project Explainer — Temporal Calibration Pilot

## What is this project about?

When an AI model is trained, it learns from data up to a certain date — called the **training cutoff**. After that date, the model has no knowledge of what happened.

This project asks a simple question:

> **Does an AI model become overconfident about facts that happened just before its training cutoff?**

The intuition is: the model has seen a lot of data about old events (say, 2022), but data about very recent events (say, June 2024) is sparse — there just wasn't enough time for the internet to write extensively about them. So the model might "know" recent events but not realise how shaky that knowledge is, and express high confidence when it shouldn't.

We also test whether **reasoning mode** (chain-of-thought thinking) makes this worse. The idea: when the model reasons through a recent-event question, it might convince itself it knows the answer more than it actually does.

---

## The Hypothesis

> A reasoning-mode LLM (Thinking ON) will become **more overconfident** on factual questions as those questions get closer to its training cutoff — more so than the same model without reasoning (Thinking OFF).

---

## How We Tested It

### The Model
We used **DeepSeek-R1-Distill-Qwen-14B** — a 14 billion parameter open-source model with a training cutoff of approximately **July 2024**. It has two modes:
- **Thinking ON**: the model writes out its reasoning before answering (chain-of-thought)
- **Thinking OFF**: the model skips reasoning and answers directly

### The Questions
We wrote **50 factual questions** about real events, split into 5 buckets based on how close the event was to the July 2024 cutoff:

| Bucket | Distance from cutoff | Example topic |
|--------|----------------------|---------------|
| B1 | 24 months away | Events from July 2022 |
| B2 | 12 months away | Events from July 2023 |
| B3 | 6 months away  | Events from January 2024 |
| B4 | 3 months away  | Events from April 2024 |
| B5 | 1 month away   | Events from June 2024 |

10 questions per bucket, across categories: sports, politics, science, business, entertainment.

Every question has a verified correct answer that predates the cutoff (so the model *should* know it).

### The Measurement
After each answer the model also states its **confidence (0–100)**.

We compute the **overconfidence gap**:
```
Overconfidence gap = mean confidence − accuracy × 100
```
- **Positive** → model is overconfident (says 80% sure, only right 50% of the time)
- **Zero** → perfectly calibrated
- **Negative** → model is underconfident (says 20% sure, actually right 50% of the time)

We then plot this gap against temporal distance for both Thinking ON and OFF.

### The Platform
Run entirely for free on **Kaggle** using 2× NVIDIA T4 GPUs (~31 GB total VRAM). No API costs.

---

## What We Found

### Accuracy
Both modes achieved **50% accuracy** on the 50 questions. This is reasonable for open-ended factual questions (not multiple choice) — well above the 30% floor-effect threshold we set.

### The Surprise: Underconfidence, Not Overconfidence
The model expressed **very low confidence** — median of 9% for Thinking ON, 1% for Thinking OFF — while being 50% accurate. The overconfidence gap is negative everywhere. The model is systematically *underconfident*.

| Bucket | Months | Thinking ON gap | Thinking OFF gap |
|--------|--------|-----------------|------------------|
| B1 | 24 | −55.0 pp | −58.2 pp |
| B2 | 12 | −33.9 pp | −64.1 pp |
| B3 | 6  | −34.4 pp | −16.0 pp |
| B4 | 3  | **−9.7 pp** | −33.4 pp |
| B5 | 1  | −30.5 pp | −44.1 pp |

### The Signal
Even though overall confidence is low, **Thinking ON is consistently less underconfident than OFF**, and the ON gap rises toward the cutoff (B1: −55 → B4: −9.7). This is the hypothesis direction — just inverted: instead of *overconfidence* near the cutoff, there is *less underconfidence*.

**Verdict: Mild signal — scale up to confirm.**

---

## Why the Unexpected Direction?

Three possible explanations:

1. **Prompt calibration**: The prompt asks for a 0–100 confidence number. R1-distill models may be RL-trained to express epistemic humility — they say "5%" even when mostly sure.

2. **Thinking OFF suppression**: The empty `<think></think>` trick forces the model to skip reasoning, which also kills its confidence — it doesn't know why it's answering, so it says 1%.

3. **Verbalized confidence ≠ true calibration**: Self-reported numbers are noisy proxies. Token log-probability confidence would be more reliable.

---

## Limitations

- **N = 50 questions** — too small for statistical significance. This is a signal-detection pilot.
- **Single model family** — can't make cross-model claims.
- **Verbalized confidence is noisy** — the model's stated numbers may not reflect true calibration.
- **OFF condition is methodologically awkward** — R1-distills are RL-trained to reason; suppressing thinking is a workaround, not a clean baseline.
- **Cutoff is approximate** — "July 2024" is the reported cutoff, exact boundary unknown.

---

## What Would a Full Paper Need?

| Dimension | Pilot (done) | Full paper |
|-----------|-------------|------------|
| Questions | 50 | 500–1200 |
| Models | 1 (14B distill) | 3–5 models across families |
| Confidence measure | Verbalized (noisy) | Token log-probs (reliable) |
| Statistics | None (pilot) | Bootstrap CIs, permutation tests |
| Baseline | Thinking OFF via prefill hack | Clean non-reasoning instruct model |

---

## Project Structure

```
data/pilot_questions.json       50 hand-curated Q&A pairs
src/                            Pipeline code (config, prompts, runner, evaluator, metrics, visualizer)
notebooks/pilot_study.ipynb     Full end-to-end run (Kaggle, GPU T4 x2)
notebooks/show_results.ipynb    Results viewer — no GPU, runs in 30 seconds
notebooks/*.csv / *.png         Pre-computed outputs from the 14B run
```

## How to Show the Results

Open `notebooks/show_results.ipynb` on Kaggle (no GPU, Internet ON, Run All — 30 seconds).  
Or open `notebooks/pilot_chart__deepseek-ai_DeepSeek-R1-Distill-Qwen-14B.png` directly.
