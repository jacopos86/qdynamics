"""Regenerate the PHYS 765 QSE/ED spectral-validation plots.

This script intentionally consumes the compact plotted-data JSON committed with
the final-project artifact bundle.  It is a reporting-side reproduction helper:
the heavy QSE/ED calculation is represented by the JSON provenance artifact,
and this script regenerates the figure files used in the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path(
            "artifacts/agent_runs/20260503_qse_spectra_plots_best_pool_percent_v1/"
            "plotted_spectra_best_pool_percent.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/agent_runs/20260503_qse_spectra_plots_best_pool_percent_v1"),
    )
    return parser


def _save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.input_json.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectra = payload["spectra"]
    ed = spectra["ED4"]
    qse_label = next(label for label in spectra if label != "ED4")
    qse = spectra[qse_label]

    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot(range(len(ed)), ed, "o-", label="ED nph=4")
    ax.plot(range(len(qse)), qse, "s--", label=qse_label)
    ax.set_xlabel("level index")
    ax.set_ylabel("energy")
    ax.set_title("HH QSE/ED absolute levels")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, args.output_dir, "hh_qse_best_pool_absolute_levels")

    ed0 = float(ed[0])
    qse0 = float(qse[0])
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    ax.plot(range(len(ed)), [float(x) - ed0 for x in ed], "o-", label="ED nph=4 gaps")
    ax.plot(range(len(qse)), [float(x) - qse0 for x in qse], "s--", label=f"{qse_label} gaps")
    ax.set_xlabel("level index")
    ax.set_ylabel("gap from ground")
    ax.set_title("HH QSE/ED gaps")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, args.output_dir, "hh_qse_best_pool_gaps")

    rows = payload["nearest_gap_percent_errors_ed4_levels_1_to_6"]
    xs = [int(row["ed4_index"]) for row in rows]
    ys = [float(row["percent_gap_error"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    bars = ax.bar([str(x) for x in xs], ys, label="QSE ADAPT+H nearest gap")
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{y:.3g}%", ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("ED nph=4 level index; ground omitted: zero gap")
    ax.set_ylabel("Percent gap error (%)")
    ax.set_title("Nearest-gap percent error vs ED nph=4")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    _save(fig, args.output_dir, "hh_qse_best_pool_nearest_gap_percent_errors_ed4_levels_1_to_6")

    print(f"wrote QSE spectral plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
