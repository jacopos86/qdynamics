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
DEFAULT_SPECTRUM_SOURCE = (
    REPO_ROOT
    / "output/diagnostics/"
    "paper_iii_hh_advisor_demo_20260802_a005/"
    "advisor_demo_result.json"
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
    parser.add_argument(
        "--spectrum-source-json",
        type=Path,
        default=DEFAULT_SPECTRUM_SOURCE,
        help="Advisor diagnostic JSON containing matched QSE and exact gaps.",
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


def _load_and_validate_spectrum(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    if payload["diagnostic_only"] is not True:
        raise ValueError("Expected a diagnostic-only spectrum artifact.")
    if payload["paper_facing"] is not False:
        raise ValueError("Expected a non-paper-facing spectrum artifact.")
    if payload["run_class"] != "diagnostic":
        raise ValueError("Unexpected spectrum evidence class.")

    model = payload["model"]
    expected_model = {
        "L": 2,
        "n_ph_max": 3,
        "boson_encoding": "binary",
        "boundary": "open",
    }
    for key, expected in expected_model.items():
        if model[key] != expected:
            raise ValueError(f"Unexpected spectrum model field {key}.")

    rows = payload["static_excited_states"]
    if len(rows) < 8:
        raise ValueError("Expected at least eight matched excited-state roots.")
    rows = rows[:8]
    if [int(row["root_index"]) for row in rows] != list(range(8)):
        raise ValueError("The first eight QSE roots are not contiguous.")
    if [int(row["matched_exact_state_index"]) for row in rows] != list(
        range(1, 9)
    ):
        raise ValueError("The first eight exact excited states are not matched.")

    for row in rows:
        exact_gap = float(row["matched_exact_gap"])
        qse_gap = float(row["qse_gap_from_ra_reference"])
        recorded_error = float(row["gap_error"])
        if not np.isclose(
            abs(qse_gap - exact_gap),
            recorded_error,
            rtol=1.0e-8,
            atol=1.0e-13,
        ):
            raise ValueError("A recorded excitation-gap error is inconsistent.")

    selected = payload["selected_excited_state"]
    if int(selected["root_index"]) != 0:
        raise ValueError("The driven initial state is not the first QSE root.")
    if not np.isclose(
        float(selected["qse_gap_from_ra_reference"]),
        float(rows[0]["qse_gap_from_ra_reference"]),
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError("The selected driven root and gap spectrum disagree.")
    return payload


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
            "label": r"Paper-II adaptive dynamics, $\Delta t=0.025$",
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
            "Title": "Paper III promoted-root Paper-II dynamics diagnostic",
            "Subject": (
                "Local diagnostic; source: "
                "output/diagnostics/paper_iii_hh_promoted_adaptive_ap_"
                "20260802_a005_v2/promoted_ap_result.json"
            ),
        }
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
        figure.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
        plt.close(figure)


def _build_gap_spectrum_figure(
    payload: dict[str, Any], output_directory: Path
) -> None:
    rows = payload["static_excited_states"][:8]
    state_index = np.arange(1, 9, dtype=int)
    exact_gap = _column(rows, "matched_exact_gap")
    qse_gap = _column(rows, "qse_gap_from_ra_reference")
    gap_error = np.abs(qse_gap - exact_gap)

    colors = {
        "exact": "#1A1A1A",
        "qse": "#D55E00",
        "selected": "#F3D7C8",
    }
    with plt.rc_context(
        {
            "font.size": 7.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "lines.linewidth": 1.25,
        }
    ):
        figure = plt.figure(figsize=(7.15, 1.72), constrained_layout=True)
        grid = figure.add_gridspec(1, 2, width_ratios=(2.4, 1.0))
        spectrum_axis = figure.add_subplot(grid[0, 0])
        error_axis = figure.add_subplot(grid[0, 1])

        spectrum_axis.axvspan(
            0.55,
            1.45,
            color=colors["selected"],
            alpha=0.55,
            linewidth=0.0,
            label="driven root",
        )
        spectrum_axis.plot(
            state_index - 0.055,
            exact_gap,
            linestyle="none",
            marker="_",
            markersize=10,
            markeredgewidth=1.7,
            color=colors["exact"],
            label="exact",
            zorder=3,
        )
        spectrum_axis.plot(
            state_index + 0.055,
            qse_gap,
            linestyle="none",
            marker="o",
            markersize=4.4,
            markerfacecolor="white",
            markeredgewidth=1.15,
            color=colors["qse"],
            label="QSE",
            zorder=4,
        )
        spectrum_axis.set_xlim(0.5, 8.5)
        spectrum_axis.set_ylim(0.76, 2.78)
        spectrum_axis.set_xticks(state_index)
        spectrum_axis.set_xlabel("excited-state index")
        spectrum_axis.set_ylabel(r"gap $\Delta E/t_{\mathrm{hop}}$")
        spectrum_axis.grid(axis="y", alpha=0.2, linewidth=0.5)
        spectrum_axis.spines["top"].set_visible(False)
        spectrum_axis.spines["right"].set_visible(False)
        spectrum_axis.text(
            0.012,
            0.95,
            "(a)",
            transform=spectrum_axis.transAxes,
            va="top",
            fontweight="bold",
        )
        handles, labels = spectrum_axis.get_legend_handles_labels()
        order = [1, 2, 0]
        spectrum_axis.legend(
            [handles[index] for index in order],
            [labels[index] for index in order],
            loc="upper left",
            bbox_to_anchor=(0.055, 1.0),
            ncol=3,
            frameon=False,
            handletextpad=0.35,
            columnspacing=0.8,
        )

        error_axis.semilogy(
            state_index,
            gap_error,
            marker="o",
            markersize=3.8,
            color=colors["qse"],
        )
        error_axis.set_xlim(0.5, 8.5)
        error_axis.set_ylim(1.0e-14, 1.0e-1)
        error_axis.set_xticks((1, 4, 8))
        error_axis.set_xlabel("excited-state index")
        error_axis.set_ylabel(r"$|\Delta E_{\rm QSE}-\Delta E_{\rm exact}|$")
        error_axis.grid(axis="y", which="both", alpha=0.2, linewidth=0.5)
        error_axis.spines["top"].set_visible(False)
        error_axis.spines["right"].set_visible(False)
        error_axis.text(
            0.035,
            0.95,
            "(b)",
            transform=error_axis.transAxes,
            va="top",
            fontweight="bold",
        )

        output_directory.mkdir(parents=True, exist_ok=True)
        stem = output_directory / "paper_iii_qse_gap_spectrum"
        metadata = {
            "Title": "Paper III QSE excitation-gap spectrum diagnostic",
            "Subject": (
                "Local diagnostic; source: output/diagnostics/"
                "paper_iii_hh_advisor_demo_20260802_a005/"
                "advisor_demo_result.json"
            ),
        }
        figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
        figure.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=300)
        plt.close(figure)


def main() -> None:
    arguments = _parse_arguments()
    payload = _load_and_validate(arguments.source_json.resolve())
    _build_figure(payload, arguments.output_directory.resolve())
    spectrum_payload = _load_and_validate_spectrum(
        arguments.spectrum_source_json.resolve()
    )
    _build_gap_spectrum_figure(
        spectrum_payload, arguments.output_directory.resolve()
    )


if __name__ == "__main__":
    main()
