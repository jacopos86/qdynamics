"""Compare Euclidean and lifted-Frobenius physicality corrections through t=20."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from paper5.stability import (
    DimerParameters,
    closed_scalar_rhs,
    structured_electron_phonon_barrier_correction,
)
from paper5.stability.matrix_reference import closed_scalar_to_matrix_state


BLOCKS = ("rho", "B", "N", "A", "C")
BLOCK_LABELS = (r"$\rho$", r"$B$", r"$N$", r"$A$", r"$C$")
BLOCK_FIELDS = (
    "electron_density",
    "coherent_phonon",
    "phonon_density",
    "anomalous_phonon_density",
    "electron_phonon_correlation",
)
ENERGY_NAMES = (
    "electronic",
    "phonon",
    "electron_phonon",
    "internal_total",
    "drive",
    "instantaneous_total",
)
COLORS = {
    "exact": "#171717",
    "raw": "#C4513C",
    "euclidean": "#6F3C8F",
    "frobenius": "#16827A",
}
LABELS = {
    "exact": "exact cutoff-16",
    "raw": "archive EOM",
    "euclidean": "Euclidean correction",
    "frobenius": "Frobenius correction",
}
LINESTYLES = {
    "exact": "-",
    "raw": "--",
    "euclidean": "-",
    "frobenius": "-.",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_lane(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        result = {key: payload[key].copy() for key in payload.files}
    result["metadata"] = json.loads(str(result.pop("metadata_json").item()))
    return result


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _interpolate(
    target_times: np.ndarray,
    source_times: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return np.interp(target_times, source_times, array)
    return np.column_stack(
        [
            np.interp(target_times, source_times, array[:, index])
            for index in range(array.shape[1])
        ]
    )


def _block_error_norms(
    approximate: np.ndarray,
    exact: np.ndarray,
) -> np.ndarray:
    errors = np.empty((approximate.shape[0], len(BLOCKS)), dtype=float)
    for row_index, (approximate_row, exact_row) in enumerate(
        zip(approximate, exact, strict=True)
    ):
        approximate_state = closed_scalar_to_matrix_state(approximate_row)
        exact_state = closed_scalar_to_matrix_state(exact_row)
        for block_index, field in enumerate(BLOCK_FIELDS):
            errors[row_index, block_index] = np.linalg.norm(
                getattr(approximate_state, field)
                - getattr(exact_state, field)
            )
    return errors


def _time_rms(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    duration = float(times[-1] - times[0])
    if duration <= 0.0:
        raise ValueError("trajectory duration must be positive")
    return np.sqrt(np.trapezoid(values**2, times, axis=0) / duration)


def _window(
    lane: dict[str, Any],
    horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.asarray(lane["times"], dtype=float)
    mask = times <= horizon + 1e-12
    return times[mask], mask


def _dynamic_scales(
    exact: dict[str, Any],
    common_times: np.ndarray,
) -> np.ndarray:
    exact_coordinates = _interpolate(
        common_times,
        np.asarray(exact["times"], dtype=float),
        np.asarray(exact["coordinates"], dtype=float),
    )
    initial = np.repeat(exact_coordinates[:1], common_times.size, axis=0)
    scales = _time_rms(
        common_times,
        _block_error_norms(exact_coordinates, initial),
    )
    if np.any(scales <= 0.0):
        raise RuntimeError("an exact dynamic normalization scale is zero")
    return scales


def _lane_metrics(
    lane: dict[str, Any],
    exact: dict[str, Any],
    *,
    horizon: float,
    dynamic_scales: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    times, mask = _window(lane, horizon)
    coordinates = np.asarray(lane["coordinates"], dtype=float)[mask]
    exact_coordinates = _interpolate(
        times,
        np.asarray(exact["times"], dtype=float),
        np.asarray(exact["coordinates"], dtype=float),
    )
    block_errors = _block_error_norms(coordinates, exact_coordinates)
    block_rms = _time_rms(times, block_errors)
    normalized = block_errors / dynamic_scales
    coordinate_errors = np.linalg.norm(coordinates - exact_coordinates, axis=1)

    exact_occupation = _interpolate(
        times,
        np.asarray(exact["times"], dtype=float),
        np.asarray(exact["site_occupation"], dtype=float),
    )
    occupation_errors = np.abs(
        np.asarray(lane["site_occupation"], dtype=float)[mask]
        - exact_occupation
    )
    exact_energies = _interpolate(
        times,
        np.asarray(exact["times"], dtype=float),
        np.asarray(exact["energy_components"], dtype=float),
    )
    energy_errors = np.abs(
        np.asarray(lane["energy_components"], dtype=float)[mask]
        - exact_energies
    )

    diagnostic_names = [
        str(value) for value in np.asarray(lane["diagnostic_names"])
    ]
    diagnostics = np.asarray(lane["diagnostics"], dtype=float)[mask]
    diagnostic_index = {
        name: index for index, name in enumerate(diagnostic_names)
    }
    post_drive = times >= 10.0 - 1e-12
    internal_energy = np.asarray(lane["energy_components"], dtype=float)[
        mask, ENERGY_NAMES.index("internal_total")
    ]

    metrics = {
        "horizon": horizon,
        "block_time_rms_frobenius_error": {
            block: float(block_rms[index])
            for index, block in enumerate(BLOCKS)
        },
        "block_dynamic_normalized_time_rms_error": {
            block: float(value)
            for block, value in zip(
                BLOCKS,
                _time_rms(times, normalized),
                strict=True,
            )
        },
        "block_final_frobenius_error": {
            block: float(block_errors[-1, index])
            for index, block in enumerate(BLOCKS)
        },
        "block_maximum_frobenius_error": {
            block: float(np.max(block_errors[:, index]))
            for index, block in enumerate(BLOCKS)
        },
        "combined_block_time_rms_frobenius_error": float(
            _time_rms(times, np.linalg.norm(block_errors, axis=1))
        ),
        "combined_dynamic_normalized_time_rms_error": float(
            _time_rms(times, np.linalg.norm(normalized, axis=1))
        ),
        "coordinate_l2_time_rms_error": float(
            _time_rms(times, coordinate_errors)
        ),
        "coordinate_l2_maximum_error": float(np.max(coordinate_errors)),
        "occupation_time_rms_error": float(
            _time_rms(times, occupation_errors)
        ),
        "occupation_maximum_error": float(np.max(occupation_errors)),
        "energy_time_rms_absolute_error": {
            name: float(value)
            for name, value in zip(
                ENERGY_NAMES,
                _time_rms(times, energy_errors),
                strict=True,
            )
        },
        "energy_maximum_absolute_error": {
            name: float(np.max(energy_errors[:, index]))
            for index, name in enumerate(ENERGY_NAMES)
        },
        "physicality": {
            "minimum_electron_eigenvalue": float(
                np.min(
                    diagnostics[
                        :,
                        diagnostic_index[
                            "electron_minimum_eigenvalue"
                        ],
                    ]
                )
            ),
            "maximum_electron_eigenvalue": float(
                np.max(
                    diagnostics[
                        :,
                        diagnostic_index[
                            "electron_maximum_eigenvalue"
                        ],
                    ]
                )
            ),
            "minimum_boson_moment_eigenvalue": float(
                np.min(
                    diagnostics[
                        :,
                        diagnostic_index[
                            "boson_moment_minimum_eigenvalue"
                        ],
                    ]
                )
            ),
            "minimum_joint_moment_eigenvalue": float(
                np.min(
                    diagnostics[
                        :,
                        diagnostic_index[
                            "joint_moment_minimum_eigenvalue"
                        ],
                    ]
                )
            ),
            "maximum_correlation_trace_absolute_value": float(
                np.max(
                    diagnostics[
                        :,
                        diagnostic_index[
                            "correlation_trace_absolute_value"
                        ],
                    ]
                )
            ),
        },
        "post_drive_internal_energy_range": (
            float(np.ptp(internal_energy[post_drive]))
            if np.any(post_drive)
            else None
        ),
    }
    return metrics, normalized


def _correction_effort(lane: dict[str, Any]) -> dict[str, Any]:
    stats = lane["metadata"]["stats"]
    evaluations = int(stats["rhs_evaluations"])
    return {
        "rhs_evaluations": evaluations,
        "active_correction_count": int(stats["active_correction_count"]),
        "active_fraction": (
            float(stats["active_correction_count"]) / evaluations
        ),
        "rms_coordinate_euclidean_norm": float(
            np.sqrt(stats["sum_squared_correction_norm"] / evaluations)
        ),
        "rms_lifted_frobenius_norm": float(
            np.sqrt(
                stats["sum_squared_correction_frobenius_norm"]
                / evaluations
            )
        ),
        "maximum_coordinate_euclidean_norm": float(
            stats["maximum_correction_norm"]
        ),
        "maximum_lifted_frobenius_norm": float(
            stats["maximum_correction_frobenius_norm"]
        ),
        "rms_lifted_block_norms": {
            block: float(np.sqrt(value / evaluations))
            for block, value in stats[
                "correction_block_sum_squared_norms"
            ].items()
        },
        "maximum_lifted_block_norms": {
            block: float(value)
            for block, value in stats[
                "correction_block_maximum_norms"
            ].items()
        },
        "maximum_constraint_count": int(
            stats["maximum_constraint_count"]
        ),
        "nonconverged_correction_count": int(
            stats["nonconverged_correction_count"]
        ),
        "minimum_corrected_joint_barrier_eigenvalue": float(
            stats["minimum_corrected_joint_barrier_eigenvalue"]
        ),
        "maximum_absolute_correction_energy_flux": float(
            stats["maximum_absolute_correction_energy_flux"]
        ),
    }


def _common_state_optimality_audit(
    lane: dict[str, Any],
    parameters: DimerParameters,
) -> dict[str, Any]:
    times = np.asarray(lane["times"], dtype=float)
    states = np.asarray(lane["coordinates"], dtype=float)
    euclidean_coordinate_norms: list[float] = []
    frobenius_coordinate_norms: list[float] = []
    euclidean_lifted_norms: list[float] = []
    frobenius_lifted_norms: list[float] = []
    all_converged = True
    for time_value, state in zip(times, states, strict=True):
        derivative = closed_scalar_rhs(float(time_value), state, parameters)
        common = {
            "activation_margin": 1e-5,
            "barrier_rate": 5.0,
            "energy_neutral": True,
            "preserve_correlation_trace": True,
            "cone_tolerance": 1e-8,
            "maximum_constraints": 128,
        }
        euclidean_result = structured_electron_phonon_barrier_correction(
            state,
            derivative,
            parameters,
            correction_metric="euclidean",
            **common,
        )
        frobenius_result = structured_electron_phonon_barrier_correction(
            state,
            derivative,
            parameters,
            correction_metric="frobenius",
            **common,
        )
        all_converged = (
            all_converged
            and euclidean_result.converged
            and frobenius_result.converged
        )
        euclidean_coordinate_norms.append(euclidean_result.correction_norm)
        frobenius_coordinate_norms.append(frobenius_result.correction_norm)
        euclidean_lifted_norms.append(
            euclidean_result.lifted_frobenius_norm
        )
        frobenius_lifted_norms.append(
            frobenius_result.lifted_frobenius_norm
        )

    coordinate_euclidean = np.asarray(euclidean_coordinate_norms)
    coordinate_frobenius = np.asarray(frobenius_coordinate_norms)
    lifted_euclidean = np.asarray(euclidean_lifted_norms)
    lifted_frobenius = np.asarray(frobenius_lifted_norms)
    return {
        "sample_count": int(times.size),
        "all_corrections_converged": bool(all_converged),
        "maximum_euclidean_optimality_violation": float(
            np.max(coordinate_euclidean - coordinate_frobenius)
        ),
        "maximum_frobenius_optimality_violation": float(
            np.max(lifted_frobenius - lifted_euclidean)
        ),
        "mean_coordinate_norm_increase_percent_for_frobenius": float(
            100.0
            * np.mean(
                (coordinate_frobenius - coordinate_euclidean)
                / np.maximum(coordinate_euclidean, 1e-15)
            )
        ),
        "mean_lifted_norm_reduction_percent_for_frobenius": float(
            100.0
            * np.mean(
                (lifted_euclidean - lifted_frobenius)
                / np.maximum(lifted_euclidean, 1e-15)
            )
        ),
    }


def _step_refinement(
    coarse: dict[str, Any],
    fine: dict[str, Any],
) -> dict[str, Any]:
    times = np.asarray(coarse["times"], dtype=float)
    coarse_coordinates = np.asarray(coarse["coordinates"], dtype=float)
    fine_coordinates = _interpolate(
        times,
        np.asarray(fine["times"], dtype=float),
        np.asarray(fine["coordinates"], dtype=float),
    )
    difference = coarse_coordinates - fine_coordinates
    block_differences = _block_error_norms(
        coarse_coordinates,
        fine_coordinates,
    )
    return {
        "coarse_time_step": float(coarse["metadata"]["time_step"]),
        "fine_time_step": float(fine["metadata"]["time_step"]),
        "maximum_absolute_coordinate_difference": float(
            np.max(np.abs(difference))
        ),
        "maximum_coordinate_l2_difference": float(
            np.max(np.linalg.norm(difference, axis=1))
        ),
        "block_time_rms_frobenius_difference": {
            block: float(value)
            for block, value in zip(
                BLOCKS,
                _time_rms(times, block_differences),
                strict=True,
            )
        },
        "block_maximum_frobenius_difference": {
            block: float(np.max(block_differences[:, index]))
            for index, block in enumerate(BLOCKS)
        },
    }


def _percentage_change(
    candidate: dict[str, float],
    baseline: dict[str, float],
) -> dict[str, float]:
    return {
        key: 100.0 * (candidate[key] - baseline[key]) / baseline[key]
        for key in baseline
    }


def _set_padded_limits(axis: plt.Axes, values: list[np.ndarray]) -> None:
    finite = np.concatenate(
        [
            np.asarray(value, dtype=float)[np.isfinite(value)]
            for value in values
        ]
    )
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = upper - lower
    padding = 0.08 * span if span > 0.0 else 0.08 * max(1.0, abs(lower))
    axis.set_ylim(lower - padding, upper + padding)


def _plot_observables(
    lanes: dict[str, dict[str, Any]],
    output_stem: Path,
    *,
    horizon: float,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=True)
    panels = (
        ("occupation", None, "site occupation"),
        ("energy", "electronic", "electronic energy"),
        ("energy", "phonon", "phonon energy"),
        ("energy", "electron_phonon", "electron-phonon energy"),
        ("energy", "internal_total", "total internal energy"),
        ("energy_error", "internal_total", "internal-energy error"),
    )
    for panel_index, (kind, component, title) in enumerate(panels):
        axis = axes.flat[panel_index]
        plotted: list[np.ndarray] = []
        for lane_name, lane in lanes.items():
            if kind == "energy_error" and lane_name == "exact":
                continue
            times, mask = _window(lane, horizon)
            if kind == "occupation":
                values = np.asarray(lane["site_occupation"], dtype=float)[mask]
            else:
                component_index = ENERGY_NAMES.index(str(component))
                values = np.asarray(lane["energy_components"], dtype=float)[
                    mask, component_index
                ]
                if kind == "energy_error":
                    exact_values = _interpolate(
                        times,
                        np.asarray(lanes["exact"]["times"], dtype=float),
                        np.asarray(
                            lanes["exact"]["energy_components"], dtype=float
                        )[:, component_index],
                    )
                    values = np.abs(values - exact_values)
            plotted.append(values)
            axis.plot(
                times,
                np.maximum(values, 1e-12)
                if kind == "energy_error"
                else values,
                color=COLORS[lane_name],
                linestyle=LINESTYLES[lane_name],
                linewidth=0.9 if lane_name != "raw" else 0.75,
                label=LABELS[lane_name],
            )
        axis.set_title(f"({chr(97 + panel_index)}) {title}", loc="left")
        axis.set_xlim(0.0, horizon)
        axis.axvspan(0.0, 4.0, color="#B9D7E8", alpha=0.20)
        axis.grid(alpha=0.17, linewidth=0.5)
        if panel_index >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")
        if kind == "occupation":
            axis.set_ylabel(r"$\rho_{11}$")
            axis.set_ylim(-0.02, 1.02)
        elif kind == "energy_error":
            axis.set_ylabel(r"$|\Delta E|/t_{\rm hop}$")
            axis.set_yscale("log")
        else:
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
            _set_padded_limits(axis, plotted)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=0.9,
        handlelength=2.0,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.105,
        top=0.875,
        hspace=0.42,
        wspace=0.32,
    )
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_block_errors(
    lanes: dict[str, dict[str, Any]],
    normalized_errors: dict[str, np.ndarray],
    output_stem: Path,
    *,
    horizon: float,
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55), sharex=True)
    for block_index, block_label in enumerate(BLOCK_LABELS):
        axis = axes.flat[block_index]
        for lane_name in ("raw", "euclidean", "frobenius"):
            times, _ = _window(lanes[lane_name], horizon)
            values = normalized_errors[lane_name][:, block_index]
            axis.plot(
                times,
                np.maximum(values, 1e-10),
                color=COLORS[lane_name],
                linestyle=LINESTYLES[lane_name],
                linewidth=0.9,
                label=LABELS[lane_name],
            )
        axis.set_yscale("log")
        axis.set_title(
            f"({chr(97 + block_index)}) {block_label} block",
            loc="left",
        )
        axis.set_xlim(0.0, horizon)
        axis.axvspan(0.0, 4.0, color="#B9D7E8", alpha=0.20)
        axis.grid(alpha=0.17, linewidth=0.5)
        if block_index >= 3:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")

    axis = axes.flat[5]
    diagnostic_name = "joint_moment_minimum_eigenvalue"
    for lane_name, lane in lanes.items():
        times, mask = _window(lane, horizon)
        names = [str(value) for value in lane["diagnostic_names"]]
        index = names.index(diagnostic_name)
        values = np.asarray(lane["diagnostics"], dtype=float)[mask, index]
        axis.plot(
            times,
            values,
            color=COLORS[lane_name],
            linestyle=LINESTYLES[lane_name],
            linewidth=0.9,
            label=LABELS[lane_name],
        )
    axis.axhline(0.0, color="black", linewidth=0.65)
    axis.set_yscale("symlog", linthresh=1e-4, linscale=0.9)
    axis.set_title(r"(f) joint-Gram minimum eigenvalue", loc="left")
    axis.set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    axis.set_xlabel(r"$t\,t_{\rm hop}$")
    axis.set_xlim(0.0, horizon)
    axis.axvspan(0.0, 4.0, color="#B9D7E8", alpha=0.20)
    axis.grid(alpha=0.17, linewidth=0.5)

    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        columnspacing=1.0,
        handlelength=2.1,
    )
    figure.text(
        0.012,
        0.52,
        "instantaneous dynamic-normalized block error",
        rotation=90,
        va="center",
        ha="left",
    )
    figure.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.875,
        hspace=0.42,
        wspace=0.30,
    )
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_summary(
    metrics: dict[str, Any],
    output_stem: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.35))
    positions = np.arange(len(BLOCKS), dtype=float)
    width = 0.24
    horizon = metrics["horizons"]["20"]
    for offset, lane_name in enumerate(("raw", "euclidean", "frobenius")):
        values = [
            horizon[lane_name][
                "block_dynamic_normalized_time_rms_error"
            ][block]
            for block in BLOCKS
        ]
        axes[0].bar(
            positions + (offset - 1) * width,
            values,
            width,
            color=COLORS[lane_name],
            label=LABELS[lane_name],
        )
    axes[0].set_yscale("log")
    axes[0].set_xticks(positions, BLOCK_LABELS)
    axes[0].set_ylabel("time-RMS normalized error")
    axes[0].set_title("(a) trajectory accuracy", loc="left")
    axes[0].grid(axis="y", alpha=0.18)

    changes = metrics["frobenius_vs_euclidean"][
        "block_dynamic_normalized_time_rms_error_percent"
    ]
    change_values = [changes[block] for block in BLOCKS]
    energy_change = metrics["frobenius_vs_euclidean"][
        "internal_energy_time_rms_error_percent"
    ]
    axes[1].bar(
        np.arange(len(BLOCKS) + 1),
        [*change_values, energy_change],
        color=[
            "#2F7D55" if value < 0.0 else "#A33A2B"
            for value in [*change_values, energy_change]
        ],
    )
    axes[1].axhline(0.0, color="black", linewidth=0.65)
    axes[1].set_xticks(
        np.arange(len(BLOCKS) + 1),
        [*BLOCK_LABELS, r"$E$"],
    )
    axes[1].set_ylabel("Frobenius vs Euclidean (%)")
    axes[1].set_title("(b) metric-induced change", loc="left")
    axes[1].grid(axis="y", alpha=0.18)

    effort = metrics["correction_effort"]
    for offset, lane_name in enumerate(("euclidean", "frobenius")):
        values = [
            effort[lane_name]["rms_lifted_block_norms"][block]
            for block in BLOCKS
        ]
        axes[2].bar(
            positions + (offset - 0.5) * 0.34,
            values,
            0.34,
            color=COLORS[lane_name],
            label=LABELS[lane_name],
        )
    axes[2].set_xticks(positions, BLOCK_LABELS)
    axes[2].set_ylabel("RMS lifted correction")
    axes[2].set_title("(c) correction allocation", loc="left")
    axes[2].grid(axis="y", alpha=0.18)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.18,
        top=0.77,
        wspace=0.38,
    )
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(figure)


def analyze(
    source_dir: Path,
    euclidean_dir: Path,
    frobenius_dir: Path,
    euclidean_fine_dir: Path,
    frobenius_fine_dir: Path,
    cutoff_summary_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        "exact": source_dir / "exact_trajectory.npz",
        "raw": source_dir / "raw_trajectory.npz",
        "source_euclidean": source_dir / "corrected_trajectory.npz",
    }
    lane_paths = {
        "euclidean": euclidean_dir / "corrected_trajectory.npz",
        "frobenius": frobenius_dir / "corrected_trajectory.npz",
        "euclidean_fine": euclidean_fine_dir / "corrected_trajectory.npz",
        "frobenius_fine": frobenius_fine_dir / "corrected_trajectory.npz",
    }
    loaded = {
        name: _load_lane(path)
        for name, path in {**source_paths, **lane_paths}.items()
    }
    lanes = {
        "exact": loaded["exact"],
        "raw": loaded["raw"],
        "euclidean": loaded["euclidean"],
        "frobenius": loaded["frobenius"],
    }

    horizon_metrics: dict[str, Any] = {}
    normalized_errors_20: dict[str, np.ndarray] = {}
    for horizon in (4.0, 20.0):
        common_times, _ = _window(loaded["euclidean"], horizon)
        scales = _dynamic_scales(loaded["exact"], common_times)
        lane_metrics: dict[str, Any] = {
            "dynamic_scales": {
                block: float(scales[index])
                for index, block in enumerate(BLOCKS)
            }
        }
        for lane_name in ("raw", "euclidean", "frobenius"):
            measured, normalized = _lane_metrics(
                loaded[lane_name],
                loaded["exact"],
                horizon=horizon,
                dynamic_scales=scales,
            )
            lane_metrics[lane_name] = measured
            if horizon == 20.0:
                normalized_errors_20[lane_name] = normalized
        horizon_metrics[f"{horizon:g}"] = lane_metrics

    source_times, source_mask = _window(loaded["source_euclidean"], 20.0)
    source_coordinates = np.asarray(
        loaded["source_euclidean"]["coordinates"], dtype=float
    )[source_mask]
    anchor_coordinates = _interpolate(
        source_times,
        np.asarray(loaded["euclidean"]["times"], dtype=float),
        np.asarray(loaded["euclidean"]["coordinates"], dtype=float),
    )
    anchor_difference = anchor_coordinates - source_coordinates

    source_manifest_path = source_dir / "runtime_manifest.json"
    euclidean_manifest_path = euclidean_dir / "runtime_manifest.json"
    frobenius_manifest_path = frobenius_dir / "runtime_manifest.json"
    source_manifest = _read_json(source_manifest_path)
    euclidean_manifest = _read_json(euclidean_manifest_path)
    frobenius_manifest = _read_json(frobenius_manifest_path)
    source_parameters = source_manifest["parameters"]
    parameters = DimerParameters(
        hopping=float(source_parameters["hopping"]),
        gamma=float(source_parameters["gamma"]),
        lambda_ep=float(source_parameters["lambda_ep"]),
        drive_amplitude=float(source_parameters["drive_amplitude"]),
        pulse_width=float(source_parameters["pulse_width"]),
    )

    common_state = {
        "euclidean_trajectory_states": _common_state_optimality_audit(
            loaded["euclidean"],
            parameters,
        ),
        "frobenius_trajectory_states": _common_state_optimality_audit(
            loaded["frobenius"],
            parameters,
        ),
    }
    anchor_maximum = float(np.max(np.abs(anchor_difference)))
    anchor_l2 = float(
        np.max(np.linalg.norm(anchor_difference, axis=1))
    )
    same_parameters = (
        source_manifest["parameters"] == euclidean_manifest["parameters"]
        == frobenius_manifest["parameters"]
    )
    source_integration = source_manifest["integration"]
    shared_integration_fields = (
        "time_step",
        "sample_step",
        "checkpoint_interval",
        "drive_cutoff",
        "exact_chunk",
    )
    same_integration = all(
        source_integration[field]
        == euclidean_manifest["integration"][field]
        == frobenius_manifest["integration"][field]
        for field in shared_integration_fields
    )
    anchor_passed = bool(
        same_parameters
        and same_integration
        and anchor_l2 <= 1e-5
        and loaded["euclidean"]["metadata"]["stats"][
            "nonconverged_correction_count"
        ]
        == 0
    )

    audit = {
        "schema": "source_locked_sensitivity_audit_v1",
        "source": {
            "visible_artifact": "paper_v_archive_eom_divergence_advisor.pdf",
            "source_manifest": str(source_manifest_path),
            "source_manifest_sha256": _sha256(source_manifest_path),
            "source_euclidean_trajectory": str(
                source_paths["source_euclidean"]
            ),
            "source_euclidean_trajectory_sha256": _sha256(
                source_paths["source_euclidean"]
            ),
            "source_variable_value": "euclidean",
            "source_final_time": source_integration["final_time"],
        },
        "sweep": {
            "classification": "diagnostic",
            "variable": "correction_metric",
            "values": ["euclidean", "frobenius"],
            "analysis_horizon": 20.0,
            "analysis_horizon_interpretation": (
                "prefix truncation of the completed source trajectory"
            ),
            "non_metric_settings_match": (
                same_parameters and same_integration
            ),
            "explicit_non_scientific_changes": [
                "output directory",
                "final_time truncated from 1000 to 20",
                "metric-support implementation and telemetry",
            ],
            "euclidean_manifest": str(euclidean_manifest_path),
            "frobenius_manifest": str(frobenius_manifest_path),
            "euclidean_manifest_sha256": _sha256(euclidean_manifest_path),
            "frobenius_manifest_sha256": _sha256(frobenius_manifest_path),
        },
        "anchor": {
            "value": "euclidean",
            "maximum_absolute_coordinate_difference": anchor_maximum,
            "maximum_coordinate_l2_difference": anchor_l2,
            "tolerance": 1e-5,
            "anchor_reproduces_source_prefix": anchor_passed,
        },
        "status": "pass" if anchor_passed else "diagnostic_invalid",
    }
    _write_json(output_dir / "source_lock_audit.json", audit)

    cutoff_summary = _read_json(cutoff_summary_path)
    cutoff_16_vs_20 = cutoff_summary["cutoff_convergence"]["16_vs_20"]
    horizon_20 = horizon_metrics["20"]
    euclidean_block = horizon_20["euclidean"][
        "block_dynamic_normalized_time_rms_error"
    ]
    frobenius_block = horizon_20["frobenius"][
        "block_dynamic_normalized_time_rms_error"
    ]
    euclidean_energy = horizon_20["euclidean"][
        "energy_time_rms_absolute_error"
    ]["internal_total"]
    frobenius_energy = horizon_20["frobenius"][
        "energy_time_rms_absolute_error"
    ]["internal_total"]

    metrics = {
        "schema_version": 1,
        "classification": "diagnostic",
        "evidence_status": "exploratory_local_not_promoted",
        "source_lock_audit": str(output_dir / "source_lock_audit.json"),
        "horizons": horizon_metrics,
        "frobenius_vs_euclidean": {
            "block_dynamic_normalized_time_rms_error_percent": (
                _percentage_change(frobenius_block, euclidean_block)
            ),
            "internal_energy_time_rms_error_percent": float(
                100.0
                * (frobenius_energy - euclidean_energy)
                / euclidean_energy
            ),
            "combined_block_time_rms_frobenius_error_percent": float(
                100.0
                * (
                    horizon_20["frobenius"][
                        "combined_block_time_rms_frobenius_error"
                    ]
                    - horizon_20["euclidean"][
                        "combined_block_time_rms_frobenius_error"
                    ]
                )
                / horizon_20["euclidean"][
                    "combined_block_time_rms_frobenius_error"
                ]
            ),
            "combined_dynamic_normalized_time_rms_error_percent": float(
                100.0
                * (
                    horizon_20["frobenius"][
                        "combined_dynamic_normalized_time_rms_error"
                    ]
                    - horizon_20["euclidean"][
                        "combined_dynamic_normalized_time_rms_error"
                    ]
                )
                / horizon_20["euclidean"][
                    "combined_dynamic_normalized_time_rms_error"
                ]
            ),
        },
        "correction_effort": {
            "euclidean": _correction_effort(loaded["euclidean"]),
            "frobenius": _correction_effort(loaded["frobenius"]),
        },
        "common_state_metric_optimality": common_state,
        "time_step_refinement": {
            "euclidean": _step_refinement(
                loaded["euclidean"],
                loaded["euclidean_fine"],
            ),
            "frobenius": _step_refinement(
                loaded["frobenius"],
                loaded["frobenius_fine"],
            ),
        },
        "exact_reference_cutoff_16_vs_20": cutoff_16_vs_20,
        "bounded_conclusion": (
            "The lifted-Frobenius objective preserves the tested physicality "
            "constraints and changes the accuracy allocation. Through t=20 "
            "it lowers the time-RMS errors in rho, B, N, and C, raises the A "
            "and energy errors, and does not uniformly improve dynamical "
            "accuracy."
        ),
    }
    _write_json(output_dir / "summary.json", metrics)

    plt.rcParams.update(
        {
            "font.size": 6.8,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.9,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "lines.solid_capstyle": "round",
        }
    )
    _plot_observables(
        lanes,
        output_dir / "paper_v_correction_metric_observables_t20",
        horizon=20.0,
    )
    _plot_block_errors(
        lanes,
        normalized_errors_20,
        output_dir / "paper_v_correction_metric_block_errors_t20",
        horizon=20.0,
    )
    _plot_summary(
        metrics,
        output_dir / "paper_v_correction_metric_summary_t20",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("euclidean_dir", type=Path)
    parser.add_argument("frobenius_dir", type=Path)
    parser.add_argument("euclidean_fine_dir", type=Path)
    parser.add_argument("frobenius_fine_dir", type=Path)
    parser.add_argument("cutoff_summary", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyze(
        args.source_dir,
        args.euclidean_dir,
        args.frobenius_dir,
        args.euclidean_fine_dir,
        args.frobenius_fine_dir,
        args.cutoff_summary,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
