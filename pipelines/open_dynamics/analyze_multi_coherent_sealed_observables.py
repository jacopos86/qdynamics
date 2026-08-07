"""Audit and plot observables from one consumed multi-coherent score.

This command reads stored trajectories only.  It does not propagate a model,
open another reference, or alter the one-shot score.  The observable
decomposition matches archive Eq. (22), as implemented by the existing
long-horizon archive diagnostic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    closed_scalar_to_matrix_state,
)
from paper5.stability.multi_coherent_scores import (
    CLOSED_COORDINATE_BLOCKS,
    closed_coordinate_distance,
)
from pipelines.open_dynamics.run_archive_long_horizon_observables import (
    _energy_components,
)


MEMBER_NAMES = ("central", "plus", "minus")
OBSERVABLE_NAMES = (
    "site_occupation",
    "electronic_energy",
    "phonon_energy",
    "electron_phonon_energy",
    "internal_total_energy",
)
OBSERVABLE_LABELS = (
    r"site occupation $\rho_{11}$",
    "electronic energy",
    "phonon energy",
    "electron-phonon energy",
    "total internal energy",
)
LANE_STYLES = {
    "exact_dop853": ("exact DOP853", "#171717", "-", 1.05),
    "exact_midpoint": ("exact midpoint", "#777777", ":", 0.95),
    "model_coarse": ("multi-coherent coarse", "#d07a24", "--", 0.80),
    "model_fine": ("multi-coherent fine", "#6f3c8f", "-", 0.95),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _trace_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_density = closed_scalar_to_matrix_state(left).electron_density
    right_density = closed_scalar_to_matrix_state(right).electron_density
    left_density = left_density / np.trace(left_density)
    right_density = right_density / np.trace(right_density)
    return float(
        0.5
        * np.sum(
            np.linalg.svd(left_density - right_density, compute_uv=False)
        )
    )


def _observable_trajectories(
    times: np.ndarray,
    coordinates: np.ndarray,
    parameters: DimerParameters,
) -> np.ndarray:
    """Return the five Figure-3-style observables from 31 coordinates."""

    time_array = np.asarray(times, dtype=float)
    path = np.asarray(coordinates, dtype=float)
    if path.shape != (time_array.size, 31):
        raise ValueError("coordinates must have shape (times, 31)")
    result = np.empty((time_array.size, len(OBSERVABLE_NAMES)), dtype=float)
    for index, (time_value, row) in enumerate(
        zip(time_array, path, strict=True)
    ):
        state = closed_scalar_to_matrix_state(row)
        energies = _energy_components(state, parameters, float(time_value))
        result[index] = (
            float(state.electron_density[0, 0].real),
            float(energies[0]),
            float(energies[1]),
            float(energies[2]),
            float(energies[3]),
        )
    return result


def _reference_postmortem(
    times: np.ndarray,
    dop853: np.ndarray,
    midpoint: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    """Locate the maximum exact-exact 31-output disagreement."""

    time_array = np.asarray(times, dtype=float)
    left = np.asarray(dop853, dtype=float)
    right = np.asarray(midpoint, dtype=float)
    scale_array = np.asarray(scales, dtype=float)
    expected = (len(MEMBER_NAMES), time_array.size, 31)
    if left.shape != expected or right.shape != expected:
        raise ValueError(f"exact paths must have shape {expected}")
    if scale_array.shape != (31,) or np.any(scale_array <= 0.0):
        raise ValueError("scales must be positive with shape (31,)")

    maximum = (-np.inf, 0, 0)
    for member_index in range(len(MEMBER_NAMES)):
        for time_index in range(time_array.size):
            distance = closed_coordinate_distance(
                left[member_index, time_index],
                right[member_index, time_index],
                scale_array,
            )
            if distance > maximum[0]:
                maximum = (distance, member_index, time_index)
    distance, member_index, time_index = maximum
    difference = left[member_index, time_index] - right[
        member_index, time_index
    ]
    block_contributions: dict[str, float] = {
        "rho": _trace_distance(
            left[member_index, time_index],
            right[member_index, time_index],
        )
    }
    for name in ("B", "N", "A", "C"):
        block = CLOSED_COORDINATE_BLOCKS[name]
        normalized = difference[block] / scale_array[block]
        block_contributions[name] = float(
            np.sqrt(np.mean(normalized**2))
        )

    scaled = np.abs(left - right) / scale_array[None, None, :]
    global_index = np.unravel_index(np.argmax(scaled), scaled.shape)
    global_member, global_time, global_coordinate = (
        int(value) for value in global_index
    )
    return {
        "maximum_equal_block_distance": float(distance),
        "member": MEMBER_NAMES[member_index],
        "time": float(time_array[time_index]),
        "sample_index": int(time_index),
        "block_root_contributions": block_contributions,
        "dominant_block_at_maximum": max(
            block_contributions,
            key=block_contributions.__getitem__,
        ),
        "largest_scaled_coordinate_disagreement": {
            "value": float(scaled[global_index]),
            "member": MEMBER_NAMES[global_member],
            "time": float(time_array[global_time]),
            "coordinate_index": global_coordinate,
            "coordinate_name": CLOSED_SCALAR_STATE_NAMES[global_coordinate],
            "raw_difference": float(
                left[global_index] - right[global_index]
            ),
            "scale": float(scale_array[global_coordinate]),
        },
    }


def _time_rms(times: np.ndarray, values: np.ndarray) -> float:
    duration = float(times[-1] - times[0])
    return float(np.sqrt(np.trapezoid(values**2, times) / duration))


def _comparison_metrics(
    times: np.ndarray,
    lanes: dict[str, np.ndarray],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for model_name in ("model_coarse", "model_fine"):
        comparisons[model_name] = {}
        for exact_name in ("exact_dop853", "exact_midpoint"):
            difference = lanes[model_name] - lanes[exact_name]
            comparisons[model_name][exact_name] = {
                name: {
                    "time_rms": _time_rms(times, difference[:, index]),
                    "maximum_absolute": float(
                        np.max(np.abs(difference[:, index]))
                    ),
                }
                for index, name in enumerate(OBSERVABLE_NAMES)
            }
    exact_difference = lanes["exact_dop853"] - lanes["exact_midpoint"]
    comparisons["exact_dop853_vs_midpoint"] = {
        name: {
            "time_rms": _time_rms(times, exact_difference[:, index]),
            "maximum_absolute": float(
                np.max(np.abs(exact_difference[:, index]))
            ),
        }
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    coarse_fine = lanes["model_coarse"] - lanes["model_fine"]
    comparisons["model_coarse_vs_fine"] = {
        name: {
            "time_rms": _time_rms(times, coarse_fine[:, index]),
            "maximum_absolute": float(np.max(np.abs(coarse_fine[:, index]))),
        }
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    return comparisons


def _plot_observables(
    times: np.ndarray,
    lanes: dict[str, np.ndarray],
    output_stem: Path,
    *,
    pulse_delays: tuple[float, ...],
    pulse_display_width: float = 4.0,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "legend.fontsize": 6.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(7.15, 4.55))
    panel_specs = (
        (0, "site occupation", None),
        (0, "early response", 20.0),
        (1, "electronic energy", None),
        (2, "phonon energy", None),
        (3, "electron-phonon energy", None),
        (4, "total internal energy", None),
    )
    for panel, (observable_index, title, horizon) in enumerate(panel_specs):
        axis = axes.flat[panel]
        mask = times <= horizon + 1e-12 if horizon is not None else slice(None)
        for lane_name, values in lanes.items():
            label, color, linestyle, width = LANE_STYLES[lane_name]
            axis.plot(
                times[mask],
                values[mask, observable_index],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=width,
                alpha=0.95,
            )
        if horizon is not None:
            axis.set_xlim(float(times[0]), horizon)
            for delay in pulse_delays:
                axis.axvspan(
                    delay,
                    delay + pulse_display_width,
                    color="#b9d7e8",
                    alpha=0.18,
                    linewidth=0.0,
                )
        else:
            axis.set_xlim(float(times[0]), float(times[-1]))
        axis.set_title(f"({chr(97 + panel)}) {title}", loc="left")
        axis.grid(alpha=0.17, linewidth=0.5)
        if observable_index == 0:
            axis.set_ylabel(r"$\rho_{11}$")
        elif panel in (2, 3):
            axis.set_ylabel(r"energy / $t_{\rm hop}$")
        if panel >= 3 or horizon is not None:
            axis.set_xlabel(r"$t\,t_{\rm hop}$")

    handles, labels = axes.flat[0].get_legend_handles_labels()
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
        0.5,
        0.012,
        (
            "Exploratory consumed-score diagnostic: the frozen exact "
            "moment-consistency gate was not cleared."
        ),
        ha="center",
        va="bottom",
        fontsize=6.3,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.12,
        top=0.88,
        hspace=0.42,
        wspace=0.32,
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def build_diagnostic(
    score_directory: Path,
    prepared_directory: Path,
    batch_directory: Path,
    output_stem: Path,
) -> dict[str, Any]:
    arrays_path = score_directory / "score_arrays.npz"
    score_summary_path = score_directory / "score_summary.json"
    prepared_manifest_path = prepared_directory / "pre_model_manifest.json"
    model_summary_path = batch_directory / "fine_central" / "summary.json"
    score_summary = json.loads(score_summary_path.read_text(encoding="utf-8"))
    prepared = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    model_summary = json.loads(model_summary_path.read_text(encoding="utf-8"))
    parameters = DimerParameters(
        hopping=float(prepared["parameters"]["hopping"]),
        gamma=float(prepared["parameters"]["gamma"]),
        lambda_ep=float(prepared["parameters"]["lambda_ep"]),
        drive_amplitude=float(prepared["parameters"]["drive_amplitude"]),
        pulse_width=float(prepared["parameters"]["pulse_width"]),
    )
    pulse_delays = tuple(
        float(value)
        for value in model_summary["parameters"]["drive_protocol"]["delays"]
    )

    with np.load(arrays_path) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        scales = np.asarray(arrays["coordinate_scales"], dtype=float)
        exact_dop853 = np.asarray(arrays["exact_dop853_closed"], dtype=float)
        exact_midpoint = np.asarray(
            arrays["exact_midpoint_closed"], dtype=float
        )
        central_coordinates = {
            "model_coarse": np.asarray(
                arrays["model_coarse_closed"][0], dtype=float
            ),
            "model_fine": np.asarray(
                arrays["model_fine_closed"][0], dtype=float
            ),
            "exact_dop853": exact_dop853[0],
            "exact_midpoint": exact_midpoint[0],
        }

    lanes = {
        name: _observable_trajectories(times, coordinates, parameters)
        for name, coordinates in central_coordinates.items()
    }
    postmortem = _reference_postmortem(
        times,
        exact_dop853,
        exact_midpoint,
        scales,
    )
    metrics = {
        "schema": "paper5.multi_coherent.sealed_observable_postmortem.v1",
        "classification": "exploratory_consumed_score_not_promoted",
        "score_status": score_summary["status"],
        "scientific_interpretation_allowed": False,
        "reason": (
            "The frozen exact moment-consistency gate was not cleared; "
            "both exact solvers and both model resolutions are displayed."
        ),
        "parameters": prepared["parameters"],
        "pulse_delays": list(pulse_delays),
        "reference_postmortem": postmortem,
        "central_observable_comparisons": _comparison_metrics(times, lanes),
        "input_hashes": {
            str(arrays_path): _sha256(arrays_path),
            str(score_summary_path): _sha256(score_summary_path),
            str(prepared_manifest_path): _sha256(prepared_manifest_path),
            str(model_summary_path): _sha256(model_summary_path),
        },
    }
    _plot_observables(
        times,
        lanes,
        output_stem,
        pulse_delays=pulse_delays,
    )
    metrics_path = output_stem.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-directory", type=Path, required=True)
    parser.add_argument("--prepared-directory", type=Path, required=True)
    parser.add_argument("--batch-directory", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    result = build_diagnostic(
        args.score_directory,
        args.prepared_directory,
        args.batch_directory,
        args.output_stem,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
