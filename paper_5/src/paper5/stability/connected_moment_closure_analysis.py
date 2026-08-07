"""Offline gate for the autonomous conditional-Pauli approximation to ``K``."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

from .connected_moment_closure import (
    conditional_pauli_regression_mixed_moment,
    conditional_pauli_regression_velocity_correction,
)
from .exact_reference import (
    ExactCorrelationClosureTrajectory,
    exact_holstein_correlation_closure_trajectory,
)
from .hubbard_dimer import DimerParameters


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sample_times(final_time: float, sample_step: float) -> np.ndarray:
    if final_time <= 0.0 or sample_step <= 0.0:
        raise ValueError("final_time and sample_step must be positive")
    intervals = int(round(final_time / sample_step))
    if not np.isclose(intervals * sample_step, final_time, atol=1e-12):
        raise ValueError("sample_step must divide final_time")
    return np.linspace(0.0, final_time, intervals + 1)


def _correlation_coordinates(correlation: np.ndarray) -> np.ndarray:
    values = np.asarray(correlation, dtype=complex)
    if values.ndim != 4 or values.shape[1:] != (2, 2, 2):
        raise ValueError("correlation must have shape (samples, 2, 2, 2)")
    coordinates = np.empty((values.shape[0], 14), dtype=float)
    shared_trace = 0.5 * (
        np.trace(values[:, 0], axis1=1, axis2=2)
        + np.trace(values[:, 1], axis1=1, axis2=2)
    )
    coordinates[:, 0] = shared_trace.real
    coordinates[:, 1] = shared_trace.imag
    for q in range(2):
        offset = 2 + 6 * q
        diagonal_difference = values[:, q, 0, 0] - values[:, q, 1, 1]
        coordinates[:, offset : offset + 6] = np.column_stack(
            [
                diagonal_difference.real,
                diagonal_difference.imag,
                values[:, q, 0, 1].real,
                values[:, q, 0, 1].imag,
                values[:, q, 1, 0].real,
                values[:, q, 1, 0].imag,
            ]
        )
    return coordinates


def _residual_subtracted(correlation: np.ndarray) -> np.ndarray:
    coordinates = _correlation_coordinates(correlation)
    return coordinates - coordinates[:1]


def _coordinate_metrics(coordinates: np.ndarray) -> dict[str, float]:
    norms = np.linalg.norm(coordinates, axis=1)
    return {
        "component_rms": float(np.sqrt(np.mean(coordinates**2))),
        "time_rms_l2": float(np.sqrt(np.mean(norms**2))),
        "maximum_l2": float(np.max(norms)),
        "final_l2": float(norms[-1]),
    }


def _complex_series_metrics(
    error: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    error_norms = np.linalg.norm(error.reshape(error.shape[0], -1), axis=1)
    reference_norms = np.linalg.norm(
        reference.reshape(reference.shape[0], -1),
        axis=1,
    )
    error_rms = float(np.sqrt(np.mean(error_norms**2)))
    reference_rms = float(np.sqrt(np.mean(reference_norms**2)))
    return {
        "time_rms_frobenius": error_rms,
        "reference_time_rms_frobenius": reference_rms,
        "relative_time_rms_frobenius": (
            error_rms / max(reference_rms, np.finfo(float).tiny)
        ),
        "maximum_frobenius": float(np.max(error_norms)),
    }


def _alignment_metrics(
    candidate: np.ndarray,
    target: np.ndarray,
) -> dict[str, float | None]:
    candidate_coordinates = _residual_subtracted(candidate)
    target_coordinates = _residual_subtracted(target)
    candidate_norm = np.linalg.norm(candidate_coordinates, axis=1)
    target_norm = np.linalg.norm(target_coordinates, axis=1)
    active = (candidate_norm > 1e-12) & (target_norm > 1e-12)
    if not np.any(active):
        return {
            "mean_cosine": None,
            "minimum_cosine": None,
            "positive_alignment_fraction": None,
            "time_rms_norm_ratio": None,
        }
    cosines = np.sum(
        candidate_coordinates[active] * target_coordinates[active],
        axis=1,
    ) / (candidate_norm[active] * target_norm[active])
    return {
        "mean_cosine": float(np.mean(cosines)),
        "minimum_cosine": float(np.min(cosines)),
        "positive_alignment_fraction": float(np.mean(cosines > 0.0)),
        "time_rms_norm_ratio": float(
            np.sqrt(np.mean(candidate_norm[active] ** 2))
            / np.sqrt(np.mean(target_norm[active] ** 2))
        ),
    }


def _evaluate_audit(
    audit: ExactCorrelationClosureTrajectory,
    parameters: DimerParameters,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    predicted_mixed_moments = []
    predicted_velocity_corrections = []
    support_ranks = []
    electronic_minima = []
    normal_residuals = []
    for state in audit.exact_trajectory.matrix_states:
        regression = conditional_pauli_regression_mixed_moment(state)
        predicted_mixed_moments.append(regression.mixed_moment)
        predicted_velocity_corrections.append(
            conditional_pauli_regression_velocity_correction(
                state,
                parameters,
            )
        )
        support_ranks.append(regression.electronic_support_rank)
        electronic_minima.append(
            regression.electronic_gram_minimum_eigenvalue
        )
        normal_residuals.append(
            regression.maximum_normal_equation_relative_residual
        )
    predicted_mixed = np.asarray(predicted_mixed_moments)
    predicted_velocity = np.asarray(predicted_velocity_corrections)
    exact_derivative = np.asarray(
        [
            derivative.electron_phonon_correlation
            for derivative in audit.exact_trajectory.matrix_derivatives
        ]
    )
    archive_error = audit.archive_correlation_derivatives - exact_derivative
    variants = {
        "archive": archive_error,
        "archive_plus_conditional_k": archive_error + predicted_velocity,
        "archive_plus_conditional_k_plus_pauli": (
            archive_error
            + predicted_velocity
            + audit.same_spin_pauli_velocity_corrections
        ),
        "archive_plus_exact_k": (
            archive_error + audit.mixed_moment_velocity_corrections
        ),
        "archive_plus_exact_k_plus_pauli": (
            archive_error
            + audit.mixed_moment_velocity_corrections
            + audit.same_spin_pauli_velocity_corrections
        ),
        "archive_plus_all_exact_terms": (
            archive_error
            + audit.mixed_moment_velocity_corrections
            + audit.same_spin_pauli_velocity_corrections
            + audit.opposite_spin_velocity_corrections
        ),
    }
    variant_coordinates = {
        name: _residual_subtracted(error)
        for name, error in variants.items()
    }
    arrays = {
        "times": audit.exact_trajectory.times,
        "exact_mixed_moment": audit.exact_mixed_moment,
        "factorized_mixed_moment": audit.factorized_mixed_moment,
        "exact_connected_mixed_moment": audit.mixed_moment_remainder,
        "conditional_connected_mixed_moment": predicted_mixed,
        "exact_k_velocity_correction": audit.mixed_moment_velocity_corrections,
        "conditional_k_velocity_correction": predicted_velocity,
        "same_spin_pauli_velocity_correction": (
            audit.same_spin_pauli_velocity_corrections
        ),
        **{
            f"{name}_residual_subtracted_coordinates": coordinates
            for name, coordinates in variant_coordinates.items()
        },
    }
    metrics = {
        "variant_C_derivative_defects": {
            name: _coordinate_metrics(coordinates)
            for name, coordinates in variant_coordinates.items()
        },
        "conditional_K_moment_error": _complex_series_metrics(
            predicted_mixed - audit.mixed_moment_remainder,
            audit.mixed_moment_remainder,
        ),
        "conditional_K_velocity_error": _complex_series_metrics(
            predicted_velocity - audit.mixed_moment_velocity_corrections,
            audit.mixed_moment_velocity_corrections,
        ),
        "conditional_K_velocity_alignment": _alignment_metrics(
            predicted_velocity,
            audit.mixed_moment_velocity_corrections,
        ),
        "regression_diagnostics": {
            "support_ranks": sorted(set(support_ranks)),
            "minimum_electronic_gram_eigenvalue": float(
                np.min(electronic_minima)
            ),
            "maximum_normal_equation_relative_residual": float(
                np.max(normal_residuals)
            ),
            "maximum_predicted_K_diagonal_absolute_value": float(
                np.max(
                    np.abs(
                        np.diagonal(predicted_mixed, axis1=-2, axis2=-1)
                    )
                )
            ),
            "maximum_predicted_K_trace_absolute_value": float(
                np.max(
                    np.abs(np.trace(predicted_mixed, axis1=-2, axis2=-1))
                )
            ),
        },
    }
    return arrays, metrics


def _write_plot(
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    times = arrays["times"]
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True)
    variants = {
        "archive": "archive",
        "archive_plus_conditional_k": "+ conditional K",
        "archive_plus_conditional_k_plus_pauli": "+ conditional K + Pauli",
        "archive_plus_exact_k": "+ exact K ceiling",
        "archive_plus_all_exact_terms": "+ all exact terms",
    }
    for name, label in variants.items():
        coordinates = arrays[f"{name}_residual_subtracted_coordinates"]
        axes[0].plot(
            times,
            np.maximum(np.linalg.norm(coordinates, axis=1), 1e-14),
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"residual $\|\Delta\dot C\|_2$")
    axes[0].set_title("Electron--phonon derivative defect")
    axes[0].legend(frameon=False, fontsize=8, ncol=2)

    exact_velocity = arrays["exact_k_velocity_correction"]
    conditional_velocity = arrays["conditional_k_velocity_correction"]
    axes[1].plot(
        times,
        np.linalg.norm(exact_velocity.reshape(times.size, -1), axis=1),
        label="exact offline K source",
        color="#222222",
    )
    axes[1].plot(
        times,
        np.linalg.norm(
            conditional_velocity.reshape(times.size, -1),
            axis=1,
        ),
        label="conditional-regression K source",
        color="#2878b5",
    )
    axes[1].set_ylabel(r"$\|\Delta\dot C_K\|_{\rm F}$")
    axes[1].set_xlabel(r"time $t\,t_{\rm hop}$")
    axes[1].set_title("Connected-moment source magnitude")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.tight_layout()
    figure.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def run_analysis(
    run_directory: Path,
    *,
    parameters: DimerParameters,
    final_time: float = 4.0,
    sample_step: float = 0.01,
    phonon_cutoffs: tuple[int, ...] = (12, 16, 20),
    decision_cutoff: int = 16,
    relative_tolerance: float = 1e-10,
    absolute_tolerance: float = 1e-12,
    maximum_step: float = 0.01,
    material_improvement_ratio: float = 0.9,
    cutoff_ratio_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Score the autonomous ``K`` approximation before any propagation."""

    if decision_cutoff not in phonon_cutoffs:
        raise ValueError("phonon_cutoffs must include decision_cutoff")
    if len(set(phonon_cutoffs)) < 2:
        raise ValueError("at least two distinct phonon cutoffs are required")
    run_directory.mkdir(parents=True, exist_ok=False)
    times = _sample_times(final_time, sample_step)
    cutoff_metrics: dict[str, Any] = {}
    decision_arrays: dict[str, np.ndarray] | None = None
    for cutoff in phonon_cutoffs:
        print(
            json.dumps(
                {
                    "event": "cutoff_started",
                    "phonon_cutoff": cutoff,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        audit = exact_holstein_correlation_closure_trajectory(
            parameters,
            sample_times=times,
            phonon_cutoff=cutoff,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
            maximum_step=maximum_step,
        )
        arrays, metrics = _evaluate_audit(audit, parameters)
        cutoff_metrics[str(cutoff)] = metrics
        if cutoff == decision_cutoff:
            decision_arrays = arrays
        print(
            json.dumps(
                {
                    "event": "cutoff_completed",
                    "phonon_cutoff": cutoff,
                    "conditional_K_C_defect": metrics[
                        "variant_C_derivative_defects"
                    ]["archive_plus_conditional_k"]["time_rms_l2"],
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if decision_arrays is None:  # pragma: no cover - validated above
        raise RuntimeError("decision arrays were not retained")

    decision_metrics = cutoff_metrics[str(decision_cutoff)]
    variants = decision_metrics["variant_C_derivative_defects"]
    raw_defect = variants["archive"]["time_rms_l2"]
    conditional_defect = variants["archive_plus_conditional_k"][
        "time_rms_l2"
    ]
    combined_defect = variants[
        "archive_plus_conditional_k_plus_pauli"
    ]["time_rms_l2"]
    highest_cutoffs = sorted(phonon_cutoffs)[-2:]
    safe_raw_defect = max(raw_defect, np.finfo(float).tiny)
    cutoff_ratios = []
    for cutoff in highest_cutoffs:
        cutoff_variant = cutoff_metrics[str(cutoff)][
            "variant_C_derivative_defects"
        ]
        cutoff_ratios.append(
            cutoff_variant["archive_plus_conditional_k"]["time_rms_l2"]
            / max(
                cutoff_variant["archive"]["time_rms_l2"],
                np.finfo(float).tiny,
            )
        )
    cutoff_ratio_difference = abs(cutoff_ratios[1] - cutoff_ratios[0])
    diagnostics = decision_metrics["regression_diagnostics"]
    structure_passed = (
        diagnostics["maximum_predicted_K_diagonal_absolute_value"] < 1e-12
        and diagnostics["maximum_predicted_K_trace_absolute_value"] < 1e-12
        and diagnostics["maximum_normal_equation_relative_residual"] < 1e-10
    )
    material_improvement_passed = (
        conditional_defect / safe_raw_defect <= material_improvement_ratio
    )
    cutoff_convergence_passed = (
        cutoff_ratio_difference <= cutoff_ratio_tolerance
    )
    short_propagation_authorized = (
        structure_passed
        and material_improvement_passed
        and cutoff_convergence_passed
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_question": (
            "Can a zero-parameter state-weighted Pauli regression recover a "
            "material, cutoff-converged part of the connected electron--two-"
            "phonon source using only the retained 31-coordinate state?"
        ),
        "parameters": {
            **asdict(parameters),
            "coupling": parameters.coupling,
            "final_time": final_time,
            "sample_step": sample_step,
            "phonon_cutoffs": list(phonon_cutoffs),
            "decision_cutoff": decision_cutoff,
            "relative_tolerance": relative_tolerance,
            "absolute_tolerance": absolute_tolerance,
            "maximum_step": maximum_step,
        },
        "cutoff_metrics": cutoff_metrics,
        "gate": {
            "structure_passed": structure_passed,
            "material_improvement_passed": material_improvement_passed,
            "cutoff_convergence_passed": cutoff_convergence_passed,
            "conditional_K_defect_ratio_over_raw": (
                conditional_defect / safe_raw_defect
            ),
            "conditional_K_plus_Pauli_defect_ratio_over_raw": (
                combined_defect / safe_raw_defect
            ),
            "highest_cutoff_improvement_ratio_difference": (
                cutoff_ratio_difference
            ),
            "short_propagation_authorized": short_propagation_authorized,
            "decision": (
                "run a short matched propagation with the joint controller"
                if short_propagation_authorized
                else "defer propagation and reject this K approximation"
            ),
        },
        "exact_reference_usage": (
            "offline derivative and cutoff scoring only; never queried by the "
            "conditional regression map"
        ),
    }

    arrays_path = run_directory / "conditional_k_gate.npz"
    plot_path = run_directory / "conditional_k_gate.png"
    summary_path = run_directory / "summary.json"
    manifest_path = run_directory / "runtime_manifest.json"
    _write_npz_atomic(arrays_path, decision_arrays)
    _write_plot(plot_path, decision_arrays)
    _write_json_atomic(summary_path, summary)
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("connected_moment_closure.py"),
        Path(__file__).with_name("exact_reference.py"),
        Path(__file__).with_name("matrix_reference.py"),
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "source_hashes": {
            str(path.resolve()): _sha256(path) for path in source_paths
        },
        "artifact_hashes": {
            path.name: _sha256(path)
            for path in (arrays_path, plot_path, summary_path)
        },
        "exact_reference_usage": summary["exact_reference_usage"],
    }
    _write_json_atomic(manifest_path, manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument("--final-time", type=float, default=4.0)
    parser.add_argument("--sample-step", type=float, default=0.01)
    parser.add_argument("--maximum-step", type=float, default=0.01)
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = run_analysis(
        args.run_directory,
        parameters=DimerParameters(
            lambda_ep=args.lambda_ep,
            gamma=args.gamma,
            drive_amplitude=args.drive,
        ),
        final_time=args.final_time,
        sample_step=args.sample_step,
        maximum_step=args.maximum_step,
    )
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
