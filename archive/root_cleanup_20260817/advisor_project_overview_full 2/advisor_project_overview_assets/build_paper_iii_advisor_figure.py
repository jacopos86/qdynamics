"""Render the advisor-facing Paper III promoted-root diagnostic figure.

The source artifact is deliberately diagnostic rather than paper-facing.  The
validations below make that status explicit so the figure cannot silently be
regenerated from an artifact that has acquired a different evidence role.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "output/diagnostics/"
    "paper_iii_hh_promoted_adaptive_ap_20260802_a005_v2/"
    "promoted_ap_result.json"
)
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "paper_iii"
FINE_GRID_KEY = "dt_0.025"


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-json",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Promoted AP diagnostic JSON.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="Directory receiving the PDF and PNG figure assets.",
    )
    return parser.parse_args()


def _load_and_validate(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    gate = payload["science_gate"]
    if payload["paper_facing"] is not False:
        raise ValueError("Expected a non-paper-facing diagnostic artifact.")
    if gate["algorithm_stack_result"] is not False:
        raise ValueError("Expected the promoted-root science gate to remain false.")
    if gate["evidence_classification"] != "local_diagnostic_not_paper_facing":
        raise ValueError("Unexpected evidence classification.")
    if gate["checks"]["independent_reference_validation"] is not True:
        raise ValueError("Independent exact-reference validation did not pass.")

    fine = payload["ap_trajectories"][FINE_GRID_KEY]
    if not np.isclose(float(fine["dt"]), 0.025):
        raise ValueError("The selected trajectory is not the locked fine grid.")
    if int(fine["point_count"]) != len(fine["trajectory"]):
        raise ValueError("Fine-grid point count does not match its trajectory.")
    return payload


def _column(rows: list[dict[str, Any]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) for row in rows], dtype=float)


def _validate_matched_references(
    adaptive_rows: list[dict[str, Any]],
    fixed_rows: list[dict[str, Any]],
) -> None:
    fields = (
        "time",
        "staggered_density_exact",
        "staggered_phonon_displacement_exact",
    )
    for field in fields:
        if not np.allclose(
            _column(adaptive_rows, field),
            _column(fixed_rows, field),
            rtol=0.0,
            atol=2.0e-13,
        ):
            raise ValueError(f"Fixed and adaptive references differ for {field}.")


def _style_axis(axis: plt.Axes, panel_label: str) -> None:
    axis.grid(alpha=0.18, linewidth=0.55)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.text(
        0.015,
        0.96,
        panel_label,
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        fontweight="bold",
    )


def _build_figure(payload: dict[str, Any], output_directory: Path) -> None:
    fine = payload["ap_trajectories"][FINE_GRID_KEY]
    adaptive_rows = fine["trajectory"]
    fixed_rows = fine["fixed_support_baseline"]["trajectory"]
    _validate_matched_references(adaptive_rows, fixed_rows)

    time = _column(adaptive_rows, "time")
    adaptive_density = _column(adaptive_rows, "staggered_density_ap")
    exact_density = _column(adaptive_rows, "staggered_density_exact")
    fixed_density = _column(fixed_rows, "staggered_density_ap")
    adaptive_phonon = _column(
        adaptive_rows, "staggered_phonon_displacement_ap"
    )
    exact_phonon = _column(
        adaptive_rows, "staggered_phonon_displacement_exact"
    )
    fixed_phonon = _column(fixed_rows, "staggered_phonon_displacement_ap")
    adaptive_fidelity = _column(adaptive_rows, "ap_exact_state_fidelity")
    fixed_fidelity = _column(fixed_rows, "ap_exact_state_fidelity")

    colors = {
        "exact": "#1A1A1A",
        "adaptive": "#D55E00",
        "fixed": "#6F7782",
    }
    with plt.rc_context(
        {
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "lines.linewidth": 1.55,
        }
    ):
        figure = plt.figure(figsize=(7.15, 4.55), constrained_layout=True)
        grid = figure.add_gridspec(2, 2, height_ratios=(1.0, 0.88))
        density_axis = figure.add_subplot(grid[0, 0])
        phonon_axis = figure.add_subplot(grid[0, 1])
        fidelity_axis = figure.add_subplot(grid[1, :])

        exact_style = {
            "color": colors["exact"],
            "linewidth": 1.75,
            "label": "exact reference",
        }
        adaptive_style = {
            "color": colors["adaptive"],
            "linewidth": 1.65,
            "label": r"AP-McLachlan, $\Delta t=0.025$",
        }
        fixed_style = {
            "color": colors["fixed"],
            "linewidth": 1.35,
            "linestyle": "--",
            "label": r"fixed McLachlan, $\Delta t=0.025$",
        }

        density_axis.plot(time, exact_density, **exact_style)
        density_axis.plot(time, adaptive_density, **adaptive_style)
        density_axis.plot(time, fixed_density, **fixed_style)
        density_axis.set_ylabel(r"staggered density $n_0-n_1$")
        density_axis.set_xlabel(r"time ($T_{\mathrm{hop}}^{-1}$)")
        density_axis.set_xlim(time[0], time[-1])
        _style_axis(density_axis, "(a)")

        phonon_axis.plot(time, exact_phonon, **exact_style)
        phonon_axis.plot(time, adaptive_phonon, **adaptive_style)
        phonon_axis.plot(time, fixed_phonon, **fixed_style)
        phonon_axis.set_ylabel(r"staggered displacement $X_0-X_1$")
        phonon_axis.set_xlabel(r"time ($T_{\mathrm{hop}}^{-1}$)")
        phonon_axis.set_xlim(time[0], time[-1])
        _style_axis(phonon_axis, "(b)")

        fidelity_axis.plot(time, adaptive_fidelity, **adaptive_style)
        fidelity_axis.plot(time, fixed_fidelity, **fixed_style)
        fidelity_axis.set_ylabel("exact-state fidelity")
        fidelity_axis.set_xlabel(r"time ($T_{\mathrm{hop}}^{-1}$)")
        fidelity_axis.set_xlim(time[0], time[-1])
        fidelity_axis.set_ylim(0.915, 1.001)
        _style_axis(fidelity_axis, "(c)")

        handles, labels = density_axis.get_legend_handles_labels()
        figure.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=3,
            frameon=False,
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        stem = output_directory / "paper_iii_promoted_ap_diagnostic"
        metadata = {
            "Title": "Paper III promoted-root AP-McLachlan diagnostic",
            "Subject": (
                "Local diagnostic; source: "
                "output/diagnostics/paper_iii_hh_promoted_adaptive_ap_"
                "20260802_a005_v2/promoted_ap_result.json"
            ),
        }
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
        figure.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
        plt.close(figure)


def main() -> None:
    arguments = _parse_arguments()
    payload = _load_and_validate(arguments.source_json.resolve())
    _build_figure(payload, arguments.output_directory.resolve())


if __name__ == "__main__":
    main()
