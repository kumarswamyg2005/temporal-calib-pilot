"""The headline figure: overconfidence gap vs temporal distance to cutoff."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe in headless kernels; notebook still displays inline
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

_STYLE = {
    "thinking_on": {
        "color": "#1f4fd8",
        "linestyle": "-",
        "marker": "o",
        "label": "Thinking ON",
    },
    "thinking_off": {
        "color": "#e8740c",
        "linestyle": "--",
        "marker": "s",
        "label": "Thinking OFF",
    },
}


def plot_overconfidence(
    summary: pd.DataFrame,
    out_path: Path,
    model_label: str = "DeepSeek-R1-Distill-Qwen-7B",
) -> Path:
    """Render and save the overconfidence-gap line chart.

    Args:
        summary: Output of :func:`metrics.compute_summary` (needs columns
            ``distance_months``, ``mode``, ``overconfidence_gap``).
        out_path: Destination ``.png`` path.

    Returns:
        Path: The path the figure was written to.

    Raises:
        ValueError: If expected columns are missing.
        OSError: If the figure cannot be written.
    """
    needed = {"distance_months", "mode", "overconfidence_gap"}
    missing = needed - set(summary.columns)
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    for mode, style in _STYLE.items():
        sub = (
            summary[summary["mode"] == mode]
            .sort_values("distance_months")
        )
        if sub.empty:
            continue
        ax.plot(
            sub["distance_months"],
            sub["overconfidence_gap"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=9,
            linewidth=2.2,
            label=style["label"],
        )

    ax.axhline(
        0.0,
        color="grey",
        linewidth=1.3,
        linestyle=":",
        zorder=0,
    )
    ax.text(
        ax.get_xlim()[1],
        0.0,
        "  Perfectly calibrated",
        color="grey",
        va="center",
        ha="left",
        fontsize=9,
    )

    ax.set_xscale("log")
    ticks = [1, 3, 6, 12, 24]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlim(0.8, 30)
    # Data-driven y-limits: keep the spec's -10..60 window as a *minimum* so
    # small, well-calibrated runs still look the same, but expand it whenever
    # the observed gaps fall outside that range (a weak, very overconfident
    # model produces gaps well above 60 and would otherwise clip off-chart).
    gaps = summary["overconfidence_gap"].astype(float)
    y_lo = min(-10.0, float(gaps.min()) - 5.0)
    y_hi = max(60.0, float(gaps.max()) + 10.0)
    ax.set_ylim(y_lo, y_hi)
    ax.invert_xaxis()  # closest-to-cutoff (1mo) on the right reads naturally

    ax.set_xlabel("Temporal distance from training cutoff (months, log scale)")
    ax.set_ylabel("Overconfidence gap (percentage points)")
    ax.set_title(
        "Overconfidence Gap vs Temporal Distance from Training Cutoff\n"
        f"({model_label}, n=10 per bucket)",
        fontsize=13,
    )
    ax.legend(loc="upper left", frameon=True)
    fig.text(
        0.5,
        -0.02,
        "Pilot study, N=50 questions. Higher = more overconfident.",
        ha="center",
        fontsize=9,
        color="#444444",
    )

    out_path = Path(out_path)
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    except OSError as exc:
        plt.close(fig)
        raise OSError(f"Failed to write chart to {out_path}: {exc}") from exc

    return out_path
