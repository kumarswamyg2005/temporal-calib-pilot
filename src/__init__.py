"""Temporal Calibration Pilot — research package.

Pilot study testing whether reasoning-mode LLMs become miscalibrated
(overconfident) on factual questions about events near their training cutoff.

Modules
-------
config           : Path resolution (Kaggle / Colab / local aware).
question_loader  : Load and validate the curated question set.
prompts          : System/instruction prompts for thinking ON vs OFF.
model_runner     : DeepSeek-R1 inference behind a swappable BaseModel interface.
evaluator        : Answer-correctness matching.
metrics          : Accuracy, ECE, and overconfidence-gap computation.
visualizer       : The headline overconfidence-vs-distance chart.
api_runner       : Stub for future paid-API models (not implemented in pilot).
"""

__version__ = "0.1.0"
