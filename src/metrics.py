"""Calibration metrics: accuracy, mean confidence, overconfidence gap, ECE.

The headline quantity is the **overconfidence gap**:

    gap = mean_confidence - accuracy * 100

Positive => the model is more confident than it is accurate (overconfident).
Zero => perfectly calibrated. Negative => underconfident.

Expected Calibration Error (ECE) is reported alongside as a standard,
binning-based calibration measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Columns required in the per-call results frame consumed here.
REQUIRED_COLUMNS = {
    "bucket",
    "distance_months",
    "mode",
    "confidence",
    "is_correct",
}

SUMMARY_COLUMNS = [
    "bucket",
    "distance_months",
    "mode",
    "n",
    "accuracy",
    "mean_confidence",
    "overconfidence_gap",
    "ece",
]


def compute_ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Compute the Expected Calibration Error.

    Args:
        confidences: Confidence values in ``[0, 100]``.
        correct: Boolean/0-1 array of correctness, same length.
        n_bins: Number of equal-width probability bins.

    Returns:
        float: ECE in ``[0, 1]``. ``nan`` if there are no samples.
    """
    confidences = np.asarray(confidences, dtype=float) / 100.0
    correct = np.asarray(correct, dtype=float)
    if confidences.size == 0:
        return float("nan")

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = confidences.size
    for lo, hi in zip(bins[:-1], bins[1:]):
        # Last bin is closed on the right so confidence == 1.0 is counted.
        in_bin = (confidences > lo) & (confidences <= hi)
        if lo == 0.0:
            in_bin |= confidences == 0.0
        count = int(in_bin.sum())
        if count == 0:
            continue
        acc = correct[in_bin].mean()
        conf = confidences[in_bin].mean()
        ece += (count / n) * abs(acc - conf)
    return float(ece)


def _validate_frame(df: pd.DataFrame) -> None:
    """Raise if the results frame is missing required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Results frame is missing required columns: {sorted(missing)}"
        )


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-call results into per-(bucket, mode) calibration stats.

    Args:
        df: Per-call results with at least :data:`REQUIRED_COLUMNS`.

    Returns:
        pandas.DataFrame: One row per (bucket, mode) with columns
        :data:`SUMMARY_COLUMNS`, sorted by descending temporal distance then
        mode (so B1=24mo first, matching the chart's reading order).

    Raises:
        ValueError: If required columns are absent.
    """
    _validate_frame(df)

    rows = []
    for (bucket, mode), grp in df.groupby(["bucket", "mode"], sort=False):
        correct = grp["is_correct"].astype(float).to_numpy()
        conf = grp["confidence"].astype(float).to_numpy()
        accuracy = float(correct.mean())
        mean_conf = float(conf.mean())
        rows.append(
            {
                "bucket": bucket,
                "distance_months": int(grp["distance_months"].iloc[0]),
                "mode": mode,
                "n": int(len(grp)),
                "accuracy": round(accuracy, 4),
                "mean_confidence": round(mean_conf, 2),
                "overconfidence_gap": round(mean_conf - accuracy * 100.0, 2),
                "ece": round(compute_ece(conf, correct), 4),
            }
        )

    summary = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return summary.sort_values(
        ["distance_months", "mode"], ascending=[False, True]
    ).reset_index(drop=True)


def overall_by_mode(df: pd.DataFrame) -> pd.DataFrame:
    """Compute headline accuracy/calibration per mode, pooled across buckets.

    Args:
        df: Per-call results with at least :data:`REQUIRED_COLUMNS`.

    Returns:
        pandas.DataFrame: One row per mode with ``n``, ``accuracy``,
        ``mean_confidence``, ``overconfidence_gap``, ``ece``.
    """
    _validate_frame(df)
    rows = []
    for mode, grp in df.groupby("mode", sort=False):
        correct = grp["is_correct"].astype(float).to_numpy()
        conf = grp["confidence"].astype(float).to_numpy()
        acc = float(correct.mean())
        mc = float(conf.mean())
        rows.append(
            {
                "mode": mode,
                "n": int(len(grp)),
                "accuracy": round(acc, 4),
                "mean_confidence": round(mc, 2),
                "overconfidence_gap": round(mc - acc * 100.0, 2),
                "ece": round(compute_ece(conf, correct), 4),
            }
        )
    return pd.DataFrame(rows)
