"""Plot the archive-controller accuracy trade-off and boundedness result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLOCKS = ("rho", "B", "N", "A", "C")
BLOCK_LABELS = (r"$\rho$", r"$B$", r"$N$", r"$A$", r"$C$")
RAW_DIVERGENCE_THRESHOLD = 1.0e4
RAW_DIVERGENCE_TIME = 141.4058846


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _block_errors(summary: dict[str, object], route: str) -> np.ndarray:
    baseline = summary["baseline"]
    if not isinstance(baseline, dict):
        raise ValueError("analysis summary lacks baseline results")
    route_result = baseline[route]
    if not isinstance(route_result, dict):
        raise ValueError(f"analysis summary lacks {route} results")
    block_errors = route_result["block_errors"]
    if not isinstance(block_errors, dict):
        raise ValueError(f"analysis summary lacks {route} block errors")
    return np.asarray(
        [
            float(block_errors[name]["rms_error_over_exact_dynamic_rms"])
            for name in BLOCKS
        ],
        dtype=float,
    )


def build_figure(
    analysis_summary: Path,
    long_horizon_summary: Path,
    output_stem: Path,
) -> None:
    analysis = _read_json(analysis_summary)
    long_horizon = _read_json(long_horizon_summary)
    raw = _block_errors(analysis, "raw")
    corrected = _block_errors(analysis, "corrected")

    stats = long_horizon["stats"]
    if not isinstance(stats, dict):
        raise ValueError("long-horizon summary lacks statistics")
    corrected_maximum = float(stats["maximum_absolute_state"])
    corrected_horizon = float(long_horizon["final_time"])

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
        }
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.75),
        gridspec_kw={"width_ratios": (1.42, 1.0)},
    )

    x = np.arange(len(BLOCKS), dtype=float)
    width = 0.36
    axes[0].bar(
        x - width / 2,
        raw,
        width,
        label="uncorrected",
        color="#657786",
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].bar(
        x + width / 2,
        corrected,
        width,
        label="physicality-corrected",
        color="#7b3f98",
        edgecolor="white",
        linewidth=0.6,
    )
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, BLOCK_LABELS)
    axes[0].set_ylabel("dynamic-normalized time-RMS error")
    axes[0].set_title(r"(a) Accuracy over $0\leq t\leq4$")
    axes[0].grid(axis="y", which="both", alpha=0.22)
    axes[0].legend(frameon=False, loc="upper left")

    ratios = corrected / raw
    for index, ratio in enumerate(ratios):
        higher = max(raw[index], corrected[index])
        change = 100.0 * (ratio - 1.0)
        color = "#b94a3a" if change > 0.0 else "#187d73"
        axes[0].text(
            x[index],
            higher * 1.14,
            f"{change:+.0f}%",
            ha="center",
            va="bottom",
            color=color,
            fontweight="bold",
            fontsize=7.5,
        )
    axes[0].set_ylim(7.0e-2, 1.35e1)

    outcomes = np.asarray(
        [RAW_DIVERGENCE_THRESHOLD, corrected_maximum], dtype=float
    )
    outcome_colors = ("#c4513c", "#7b3f98")
    bars = axes[1].bar(
        (0, 1),
        outcomes,
        width=0.58,
        color=outcome_colors,
        edgecolor="white",
        linewidth=0.7,
    )
    axes[1].set_yscale("log")
    axes[1].set_xticks((0, 1), ("uncorrected", "corrected"))
    axes[1].set_ylabel(r"largest coordinate magnitude $\max_j|x_j|$")
    axes[1].set_title("(b) Boundedness over tested horizons")
    axes[1].axhline(
        RAW_DIVERGENCE_THRESHOLD,
        color="#8f2d20",
        linestyle="--",
        linewidth=0.9,
    )
    axes[1].grid(axis="y", which="both", alpha=0.22)
    axes[1].set_ylim(1.0, 3.0e4)
    axes[1].bar_label(
        bars,
        labels=(
            rf"$10^4$ at $t={RAW_DIVERGENCE_TIME:.1f}$",
            rf"${corrected_maximum:.2f}$ through $t={corrected_horizon:.0f}$",
        ),
        padding=3,
        fontsize=7.5,
    )

    figure.tight_layout(w_pad=2.0)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_summary", type=Path)
    parser.add_argument("long_horizon_summary", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    build_figure(
        args.analysis_summary,
        args.long_horizon_summary,
        args.output_stem,
    )


if __name__ == "__main__":
    main()
