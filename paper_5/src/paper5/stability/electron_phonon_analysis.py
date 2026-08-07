"""Matched exact, raw-closure, and joint-barrier dimer analyses.

This module is a reporting-only diagnostic.  Exact truncated-Hamiltonian
trajectories are contracted onto the same 31 coordinates as the archive
closure, but exact information never enters the closure or its online
barrier controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from .cone_correction import (
    closed_state_correction_energy_gradient,
    structured_electron_phonon_barrier_correction,
    structured_electron_phonon_moment_velocity_lift,
    structured_electron_velocity_lift,
)
from .exact_reference import (
    ExactDrivenTrajectory,
    exact_holstein_driven_trajectory,
)
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import (
    CLOSED_SCALAR_STATE_NAMES,
    boson_moment_matrix,
    closed_scalar_rhs,
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
    pauli_repaired_closed_scalar_rhs,
)


BLOCK_SLICES: dict[str, slice] = {
    "rho": slice(0, 3),
    "B": slice(3, 7),
    "N": slice(7, 11),
    "A": slice(11, 17),
    "C": slice(17, 31),
}
BLOCK_FIELDS = {
    "rho": "electron_density",
    "B": "coherent_phonon",
    "N": "phonon_density",
    "A": "anomalous_phonon_density",
    "C": "electron_phonon_correlation",
}
BARRIER_SECTORS = ("rho_lower", "rho_upper", "joint_gram")
JOINT_MODE_LABELS = (
    "delta_b0",
    "delta_b1",
    "delta_b0_dagger",
    "delta_b1_dagger",
    "delta_sigma_x",
    "delta_sigma_y",
    "delta_sigma_z",
)


@dataclass(frozen=True)
class FixedStepTrajectory:
    """One accepted-step trajectory and sampled controller diagnostics."""

    times: FloatArray
    coordinates: FloatArray
    correction_coordinates: FloatArray
    equality_only_coordinates: FloatArray
    raw_velocity_norms: FloatArray
    raw_barrier_minima: FloatArray
    corrected_barrier_minima: FloatArray
    joint_mode_weights: FloatArray
    constraint_counts: FloatArray
    integration_rhs_evaluations: int
    controller_diagnostic_evaluations: int


@dataclass(frozen=True)
class MatchedCase:
    """Exact, raw, and corrected trajectories for one physical point."""

    parameters: DimerParameters
    phonon_cutoff: int
    time_step: float
    exact: ExactDrivenTrajectory
    exact_coordinates: FloatArray
    raw: FixedStepTrajectory
    corrected: FixedStepTrajectory
    metrics: dict[str, Any]


@dataclass(frozen=True)
class PauliRepairAblation:
    """Matched four-lane test of the autonomous same-spin Pauli repair."""

    parameters: DimerParameters
    phonon_cutoff: int
    time_step: float
    exact: ExactDrivenTrajectory
    exact_coordinates: FloatArray
    raw: FixedStepTrajectory
    pauli_repaired: FixedStepTrajectory
    controller: FixedStepTrajectory
    pauli_repaired_controller: FixedStepTrajectory
    metrics: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _append_progress(path: Path, payload: dict[str, Any]) -> None:
    record = {"recorded_at_utc": _utc_now(), **payload}
    encoded = json.dumps(record, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(encoded, flush=True)


def _sample_times(final_time: float, time_step: float) -> FloatArray:
    if final_time <= 0.0 or time_step <= 0.0:
        raise ValueError("final_time and time_step must be positive")
    steps = int(round(final_time / time_step))
    if not np.isclose(steps * time_step, final_time, atol=1e-12, rtol=0.0):
        raise ValueError("time_step must divide final_time")
    return np.linspace(0.0, final_time, steps + 1)


def _minimum_norm_equality_correction(
    state: FloatArray,
    derivative: FloatArray,
    parameters: DimerParameters,
    *,
    barrier_rate: float,
) -> FloatArray:
    """Satisfy energy and ``Tr C`` tangency without cone inequalities."""

    rows = np.zeros((3, len(CLOSED_SCALAR_STATE_NAMES)), dtype=float)
    rows[0] = closed_state_correction_energy_gradient(state, parameters)
    rows[1, 17] = 1.0
    rows[2, 18] = 1.0
    targets = np.array(
        [
            0.0,
            -derivative[17] - barrier_rate * state[17],
            -derivative[18] - barrier_rate * state[18],
        ],
        dtype=float,
    )
    correction, *_ = np.linalg.lstsq(rows, targets, rcond=None)
    return np.asarray(correction, dtype=float)


def _barrier_diagnostics(
    state: FloatArray,
    derivative: FloatArray,
    correction: FloatArray,
    *,
    activation_margin: float,
    barrier_rate: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return raw/corrected barrier minima and the weakest joint mode."""

    matrix_state = closed_scalar_to_matrix_state(state)
    electron = np.asarray(matrix_state.electron_density, dtype=complex)
    electron = 0.5 * (electron + electron.conjugate().T)
    electron_velocity = structured_electron_velocity_lift(derivative[:3])
    electron_correction = structured_electron_velocity_lift(correction[:3])
    electron_identity = np.eye(2, dtype=complex)

    joint = electron_phonon_moment_matrix(matrix_state)
    joint_velocity = structured_electron_phonon_moment_velocity_lift(
        state,
        derivative,
    )
    joint_correction = structured_electron_phonon_moment_velocity_lift(
        state,
        correction,
    )
    joint_identity = np.eye(7, dtype=complex)
    raw_matrices = (
        electron_velocity
        + barrier_rate
        * (electron - activation_margin * electron_identity),
        -electron_velocity
        + barrier_rate
        * (electron_identity - electron - activation_margin * electron_identity),
        joint_velocity
        + barrier_rate * (joint - activation_margin * joint_identity),
    )
    corrected_matrices = (
        raw_matrices[0] + electron_correction,
        raw_matrices[1] - electron_correction,
        raw_matrices[2] + joint_correction,
    )
    raw_minima = np.asarray(
        [np.linalg.eigvalsh(matrix)[0] for matrix in raw_matrices],
        dtype=float,
    )
    corrected_minima = np.asarray(
        [np.linalg.eigvalsh(matrix)[0] for matrix in corrected_matrices],
        dtype=float,
    )
    _, eigenvectors = np.linalg.eigh(raw_matrices[2])
    mode_weights = np.abs(eigenvectors[:, 0]) ** 2
    return raw_minima, corrected_minima, np.asarray(mode_weights, dtype=float)


def integrate_closed_rk4(
    parameters: DimerParameters,
    initial_state: FloatArray,
    *,
    final_time: float,
    time_step: float,
    corrected: bool,
    pauli_repair: bool = False,
    rhs_override: Callable[
        [float, FloatArray, DimerParameters], FloatArray
    ]
    | None = None,
    activation_margin: float = 1e-5,
    barrier_rate: float = 5.0,
    cone_tolerance: float = 1e-8,
    maximum_constraints: int = 128,
) -> FixedStepTrajectory:
    """Integrate one archive/Pauli and controller/no-controller RK4 lane."""

    times = _sample_times(final_time, time_step)
    state = np.asarray(initial_state, dtype=float).copy()
    expected = (len(CLOSED_SCALAR_STATE_NAMES),)
    if state.shape != expected:
        raise ValueError(f"expected initial state shape {expected}, got {state.shape}")

    sample_count = times.size
    coordinates = np.empty((sample_count, state.size), dtype=float)
    corrections = np.zeros_like(coordinates)
    equality_only = np.zeros_like(coordinates)
    raw_velocity_norms = np.empty(sample_count, dtype=float)
    raw_barrier_minima = np.empty((sample_count, 3), dtype=float)
    corrected_barrier_minima = np.empty((sample_count, 3), dtype=float)
    joint_mode_weights = np.empty((sample_count, 7), dtype=float)
    constraint_counts = np.zeros(sample_count, dtype=float)
    integration_rhs_evaluations = 0
    controller_diagnostic_evaluations = 0
    if rhs_override is not None and pauli_repair:
        raise ValueError("rhs_override and pauli_repair cannot both be set")
    closure_rhs = rhs_override or (
        pauli_repaired_closed_scalar_rhs if pauli_repair else closed_scalar_rhs
    )

    def evaluate(
        time_value: float,
        current_state: FloatArray,
    ) -> tuple[FloatArray, FloatArray, int]:
        nonlocal integration_rhs_evaluations
        integration_rhs_evaluations += 1
        raw = closure_rhs(time_value, current_state, parameters)
        if not corrected:
            return raw, np.zeros_like(raw), 0
        result = structured_electron_phonon_barrier_correction(
            current_state,
            raw,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=cone_tolerance,
            maximum_constraints=maximum_constraints,
        )
        if not result.converged:
            raise RuntimeError(
                "joint electron-phonon correction did not converge: "
                f"minimum={result.corrected_joint_barrier_minimum_eigenvalue}"
            )
        return (
            raw + result.correction_coordinates,
            np.asarray(result.correction_coordinates, dtype=float),
            result.constraint_count,
        )

    for index, time_value in enumerate(times[:-1]):
        coordinates[index] = state
        k1, correction, constraint_count = evaluate(time_value, state)
        raw = closure_rhs(time_value, state, parameters)
        corrections[index] = correction
        constraint_counts[index] = constraint_count
        raw_velocity_norms[index] = np.linalg.norm(raw)
        if corrected:
            equality_only[index] = _minimum_norm_equality_correction(
                state,
                raw,
                parameters,
                barrier_rate=barrier_rate,
            )
        (
            raw_barrier_minima[index],
            corrected_barrier_minima[index],
            joint_mode_weights[index],
        ) = _barrier_diagnostics(
            state,
            raw,
            correction,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
        )

        half_step = 0.5 * time_step
        k2, _, _ = evaluate(
            time_value + half_step,
            state + half_step * k1,
        )
        k3, _, _ = evaluate(
            time_value + half_step,
            state + half_step * k2,
        )
        k4, _, _ = evaluate(
            time_value + time_step,
            state + time_step * k3,
        )
        state = state + (time_step / 6.0) * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        )
        if not np.all(np.isfinite(state)):
            raise RuntimeError(
                f"non-finite state reached at t={time_value + time_step}"
            )

    coordinates[-1] = state
    final_time_value = float(times[-1])
    raw = closure_rhs(final_time_value, state, parameters)
    raw_velocity_norms[-1] = np.linalg.norm(raw)
    if corrected:
        result = structured_electron_phonon_barrier_correction(
            state,
            raw,
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=cone_tolerance,
            maximum_constraints=maximum_constraints,
        )
        controller_diagnostic_evaluations += 1
        if not result.converged:
            raise RuntimeError("final sampled controller diagnostic did not converge")
        corrections[-1] = result.correction_coordinates
        constraint_counts[-1] = result.constraint_count
        equality_only[-1] = _minimum_norm_equality_correction(
            state,
            raw,
            parameters,
            barrier_rate=barrier_rate,
        )
    (
        raw_barrier_minima[-1],
        corrected_barrier_minima[-1],
        joint_mode_weights[-1],
    ) = _barrier_diagnostics(
        state,
        raw,
        corrections[-1],
        activation_margin=activation_margin,
        barrier_rate=barrier_rate,
    )
    return FixedStepTrajectory(
        times=times,
        coordinates=coordinates,
        correction_coordinates=corrections,
        equality_only_coordinates=equality_only,
        raw_velocity_norms=raw_velocity_norms,
        raw_barrier_minima=raw_barrier_minima,
        corrected_barrier_minima=corrected_barrier_minima,
        joint_mode_weights=joint_mode_weights,
        constraint_counts=constraint_counts,
        integration_rhs_evaluations=integration_rhs_evaluations,
        controller_diagnostic_evaluations=controller_diagnostic_evaluations,
    )


def _matrix_block_norms(coordinates: FloatArray) -> FloatArray:
    result = np.empty((coordinates.shape[0], len(BLOCK_FIELDS)), dtype=float)
    for row_index, row in enumerate(coordinates):
        state = closed_scalar_to_matrix_state(row)
        for block_index, field in enumerate(BLOCK_FIELDS.values()):
            result[row_index, block_index] = np.linalg.norm(getattr(state, field))
    return result


def _matrix_block_errors(
    reference: FloatArray,
    candidate: FloatArray,
) -> FloatArray:
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate trajectories must match")
    result = np.empty((reference.shape[0], len(BLOCK_FIELDS)), dtype=float)
    for row_index, (reference_row, candidate_row) in enumerate(
        zip(reference, candidate, strict=True)
    ):
        reference_state = closed_scalar_to_matrix_state(reference_row)
        candidate_state = closed_scalar_to_matrix_state(candidate_row)
        for block_index, field in enumerate(BLOCK_FIELDS.values()):
            result[row_index, block_index] = np.linalg.norm(
                getattr(candidate_state, field) - getattr(reference_state, field)
            )
    return result


def _block_series_summary(
    errors: FloatArray,
    reference: FloatArray,
) -> dict[str, dict[str, float]]:
    reference_norms = _matrix_block_norms(reference)
    reference_change = _matrix_block_errors(
        np.repeat(reference[:1], reference.shape[0], axis=0),
        reference,
    )
    result: dict[str, dict[str, float]] = {}
    for index, name in enumerate(BLOCK_FIELDS):
        rms_error = float(np.sqrt(np.mean(errors[:, index] ** 2)))
        reference_rms = float(
            np.sqrt(np.mean(reference_norms[:, index] ** 2))
        )
        dynamic_rms = float(
            np.sqrt(np.mean(reference_change[:, index] ** 2))
        )
        result[name] = {
            "maximum_frobenius_error": float(np.max(errors[:, index])),
            "final_frobenius_error": float(errors[-1, index]),
            "rms_frobenius_error": rms_error,
            "reference_rms_norm": reference_rms,
            "exact_dynamic_rms_scale": dynamic_rms,
            "rms_error_over_reference_rms": rms_error / max(reference_rms, 1e-14),
            "rms_error_over_exact_dynamic_rms": rms_error / max(dynamic_rms, 1e-14),
        }
    return result


def _trajectory_certificates(
    coordinates: FloatArray,
    times: FloatArray,
) -> dict[str, float | None]:
    if coordinates.shape[0] != times.size:
        raise ValueError("certificate coordinates and times must match")
    electron_minima: list[float] = []
    electron_maxima: list[float] = []
    boson_minima: list[float] = []
    joint_minima: list[float] = []
    trace_values: list[float] = []
    for row in coordinates:
        state = closed_scalar_to_matrix_state(row)
        electron_eigenvalues = np.linalg.eigvalsh(state.electron_density)
        electron_minima.append(float(electron_eigenvalues[0]))
        electron_maxima.append(float(electron_eigenvalues[-1]))
        boson_minima.append(float(np.linalg.eigvalsh(boson_moment_matrix(state))[0]))
        joint_minima.append(
            float(np.linalg.eigvalsh(electron_phonon_moment_matrix(state))[0])
        )
        correlation_traces = np.trace(
            state.electron_phonon_correlation,
            axis1=1,
            axis2=2,
        )
        trace_values.append(float(np.max(np.abs(correlation_traces))))
    joint_values = np.asarray(joint_minima, dtype=float)
    crossing_indices = np.flatnonzero(joint_values < 0.0)
    crossing_time: float | None = None
    if crossing_indices.size:
        crossing_time = float(times[crossing_indices[0]])
    return {
        "minimum_rho_eigenvalue": float(np.min(electron_minima)),
        "maximum_rho_eigenvalue": float(np.max(electron_maxima)),
        "minimum_boson_moment_eigenvalue": float(np.min(boson_minima)),
        "minimum_joint_gram_eigenvalue": float(np.min(joint_values)),
        "maximum_correlation_trace_absolute_value": float(np.max(trace_values)),
        "maximum_absolute_coordinate": float(np.max(np.abs(coordinates))),
        "first_sample_time_with_negative_joint_gram": crossing_time,
    }


def _protocol_metrics(
    times: FloatArray,
    exact_coordinates: FloatArray,
    candidate: FixedStepTrajectory,
    parameters: DimerParameters,
) -> tuple[dict[str, Any], FloatArray]:
    errors = _matrix_block_errors(exact_coordinates, candidate.coordinates)
    coordinate_errors = candidate.coordinates - exact_coordinates
    coordinate_l2 = np.linalg.norm(coordinate_errors, axis=1)
    exact_energies = np.asarray(
        [
            matrix_total_energy(closed_scalar_to_matrix_state(row), parameters)
            for row in exact_coordinates
        ]
    )
    candidate_energies = np.asarray(
        [
            matrix_total_energy(closed_scalar_to_matrix_state(row), parameters)
            for row in candidate.coordinates
        ]
    )
    return (
        {
            "maximum_coordinate_l2_error": float(np.max(coordinate_l2)),
            "final_coordinate_l2_error": float(coordinate_l2[-1]),
            "maximum_absolute_coordinate_error": float(
                np.max(np.abs(coordinate_errors))
            ),
            "maximum_static_energy_error": float(
                np.max(np.abs(candidate_energies - exact_energies))
            ),
            "final_static_energy_error": float(
                candidate_energies[-1] - exact_energies[-1]
            ),
            "block_errors": _block_series_summary(errors, exact_coordinates),
            "certificates": _trajectory_certificates(
                candidate.coordinates,
                times,
            ),
            "integration_rhs_evaluations": candidate.integration_rhs_evaluations,
            "sample_count": int(times.size),
        },
        errors,
    )


def _correction_history_metrics(
    trajectory: FixedStepTrajectory,
) -> dict[str, Any]:
    correction = trajectory.correction_coordinates
    equality = trajectory.equality_only_coordinates
    cone_increment = correction - equality
    correction_norms = np.linalg.norm(correction, axis=1)
    equality_norms = np.linalg.norm(equality, axis=1)
    cone_increment_norms = np.linalg.norm(cone_increment, axis=1)
    relative_norms = correction_norms / np.maximum(
        trajectory.raw_velocity_norms,
        1e-14,
    )
    active = correction_norms > 1e-10
    equality_active = equality_norms > 1e-10
    cone_active = cone_increment_norms > 1e-10
    block_norms = {
        name: np.linalg.norm(correction[:, block], axis=1)
        for name, block in BLOCK_SLICES.items()
    }
    cone_block_norms = {
        name: np.linalg.norm(cone_increment[:, block], axis=1)
        for name, block in BLOCK_SLICES.items()
    }
    block_squared = np.column_stack(
        [block_norms[name] ** 2 for name in BLOCK_SLICES]
    )
    cone_block_squared = np.column_stack(
        [cone_block_norms[name] ** 2 for name in BLOCK_SLICES]
    )
    total_squared = np.sum(block_squared, axis=1)
    cone_total_squared = np.sum(cone_block_squared, axis=1)
    cone_weights = cone_increment_norms[cone_active]
    if np.any(cone_active):
        mode_weights = np.average(
            trajectory.joint_mode_weights[cone_active],
            axis=0,
            weights=cone_weights,
        )
        trigger_indices = np.argmin(
            trajectory.raw_barrier_minima[cone_active],
            axis=1,
        )
    else:
        mode_weights = np.zeros(7, dtype=float)
        trigger_indices = np.array([], dtype=int)
    trigger_counts = {
        name: int(np.count_nonzero(trigger_indices == index))
        for index, name in enumerate(BARRIER_SECTORS)
    }
    peak_index = int(np.argmax(correction_norms))
    return {
        "first_active_time": (
            float(trajectory.times[np.flatnonzero(active)[0]])
            if np.any(active)
            else None
        ),
        "active_sample_fraction": float(np.mean(active)),
        "equality_action_sample_fraction": float(np.mean(equality_active)),
        "additional_cone_action_sample_fraction": float(np.mean(cone_active)),
        "first_additional_cone_action_time": (
            float(trajectory.times[np.flatnonzero(cone_active)[0]])
            if np.any(cone_active)
            else None
        ),
        "maximum_correction_norm": float(np.max(correction_norms)),
        "rms_correction_norm": float(np.sqrt(np.mean(correction_norms**2))),
        "maximum_relative_correction_norm": float(np.max(relative_norms)),
        "peak_correction_time": float(trajectory.times[peak_index]),
        "maximum_equality_only_norm": float(np.max(equality_norms)),
        "maximum_additional_cone_norm": float(np.max(cone_increment_norms)),
        "maximum_constraint_count": int(np.max(trajectory.constraint_counts)),
        "minimum_raw_barrier_by_sector": {
            name: float(np.min(trajectory.raw_barrier_minima[:, index]))
            for index, name in enumerate(BARRIER_SECTORS)
        },
        "minimum_corrected_barrier_by_sector": {
            name: float(np.min(trajectory.corrected_barrier_minima[:, index]))
            for index, name in enumerate(BARRIER_SECTORS)
        },
        "trigger_sector_sample_counts": trigger_counts,
        "weighted_joint_mode_component_weights": {
            name: float(mode_weights[index])
            for index, name in enumerate(JOINT_MODE_LABELS)
        },
        "weighted_joint_mode_boson_fraction": float(np.sum(mode_weights[:4])),
        "weighted_joint_mode_electron_fraction": float(np.sum(mode_weights[4:])),
        "correction_block_metrics": {
            name: {
                "maximum_norm": float(np.max(values)),
                "rms_norm": float(np.sqrt(np.mean(values**2))),
                "integrated_squared_fraction": float(
                    np.sum(block_squared[:, index])
                    / max(float(np.sum(total_squared)), 1e-30)
                ),
            }
            for index, (name, values) in enumerate(block_norms.items())
        },
        "additional_cone_block_metrics": {
            name: {
                "maximum_norm": float(np.max(values)),
                "rms_norm": float(np.sqrt(np.mean(values**2))),
                "integrated_squared_fraction": float(
                    np.sum(cone_block_squared[:, index])
                    / max(float(np.sum(cone_total_squared)), 1e-30)
                ),
            }
            for index, (name, values) in enumerate(cone_block_norms.items())
        },
    }


def _exact_defect_metrics(
    parameters: DimerParameters,
    exact: ExactDrivenTrajectory,
    exact_coordinates: FloatArray,
    *,
    activation_margin: float,
    barrier_rate: float,
    cone_tolerance: float,
    maximum_constraints: int,
    controller_stride: int,
) -> dict[str, Any]:
    undriven = replace(parameters, drive_amplitude=0.0)
    initial_residual = closed_scalar_rhs(
        0.0,
        exact_coordinates[0],
        undriven,
    )
    exact_derivatives = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(derivative)
            for derivative in exact.matrix_derivatives
        ],
        dtype=float,
    )
    closure_derivatives = np.asarray(
        [
            closed_scalar_rhs(time_value, state, parameters)
            for time_value, state in zip(
                exact.times,
                exact_coordinates,
                strict=True,
            )
        ],
        dtype=float,
    )
    defects = closure_derivatives - initial_residual - exact_derivatives
    block_norms = {
        name: np.linalg.norm(defects[:, block], axis=1)
        for name, block in BLOCK_SLICES.items()
    }
    sampled_indices = np.arange(
        0,
        exact.times.size,
        controller_stride,
        dtype=int,
    )
    if sampled_indices[-1] != exact.times.size - 1:
        sampled_indices = np.append(sampled_indices, exact.times.size - 1)
    c_defect_norms = np.linalg.norm(defects[:, 17:], axis=1)
    alignment_threshold = max(
        1e-8,
        0.01 * float(np.max(c_defect_norms)),
    )
    sampled_indices = sampled_indices[
        c_defect_norms[sampled_indices] >= alignment_threshold
    ]
    total_cosines: list[float] = []
    total_norm_ratios: list[float] = []
    total_projection_coefficients: list[float] = []
    cone_cosines: list[float] = []
    cone_norm_ratios: list[float] = []
    cone_projection_coefficients: list[float] = []
    converged_count = 0

    def append_alignment(
        candidate: FloatArray,
        target: FloatArray,
        cosines: list[float],
        norm_ratios: list[float],
        projection_coefficients: list[float],
    ) -> None:
        candidate_norm = float(np.linalg.norm(candidate))
        target_norm = float(np.linalg.norm(target))
        if candidate_norm <= 1e-14 or target_norm <= 1e-14:
            return
        cosines.append(
            float(candidate @ target / (candidate_norm * target_norm))
        )
        norm_ratios.append(candidate_norm / target_norm)
        projection_coefficients.append(
            float(candidate @ target / (target_norm**2))
        )

    for index in sampled_indices:
        result = structured_electron_phonon_barrier_correction(
            exact_coordinates[index],
            closure_derivatives[index],
            parameters,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            energy_neutral=True,
            preserve_correlation_trace=True,
            cone_tolerance=cone_tolerance,
            maximum_constraints=maximum_constraints,
        )
        converged_count += int(result.converged)
        missing_c = -defects[index, 17:]
        equality = _minimum_norm_equality_correction(
            exact_coordinates[index],
            closure_derivatives[index],
            parameters,
            barrier_rate=barrier_rate,
        )
        total_correction_c = result.correction_coordinates[17:]
        cone_increment_c = total_correction_c - equality[17:]
        append_alignment(
            total_correction_c,
            missing_c,
            total_cosines,
            total_norm_ratios,
            total_projection_coefficients,
        )
        append_alignment(
            cone_increment_c,
            missing_c,
            cone_cosines,
            cone_norm_ratios,
            cone_projection_coefficients,
        )
    dominant_counts = {name: 0 for name in BLOCK_SLICES}
    stacked_block_norms = np.column_stack(list(block_norms.values()))
    for dominant in np.argmax(stacked_block_norms, axis=1):
        dominant_counts[tuple(BLOCK_SLICES)[int(dominant)]] += 1
    return {
        "initial_residual_norm": float(np.linalg.norm(initial_residual)),
        "block_defect_norms": {
            name: {
                "maximum": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
                "final": float(values[-1]),
            }
            for name, values in block_norms.items()
        },
        "dominant_defect_block_sample_counts": dominant_counts,
        "controller_on_exact_sample_count": int(sampled_indices.size),
        "controller_on_exact_converged_count": converged_count,
        "C_alignment_minimum_defect_norm": alignment_threshold,
        "C_total_correction_vs_negative_defect_mean_cosine": (
            float(np.mean(total_cosines)) if total_cosines else None
        ),
        "C_total_correction_vs_negative_defect_positive_alignment_fraction": (
            float(np.mean(np.asarray(total_cosines) > 0.0))
            if total_cosines
            else None
        ),
        "C_total_correction_to_defect_mean_norm_ratio": (
            float(np.mean(total_norm_ratios)) if total_norm_ratios else None
        ),
        "C_total_correction_on_missing_defect_mean_projection": (
            float(np.mean(total_projection_coefficients))
            if total_projection_coefficients
            else None
        ),
        "C_cone_increment_vs_negative_defect_mean_cosine": (
            float(np.mean(cone_cosines)) if cone_cosines else None
        ),
        "C_cone_increment_vs_negative_defect_positive_alignment_fraction": (
            float(np.mean(np.asarray(cone_cosines) > 0.0))
            if cone_cosines
            else None
        ),
        "C_cone_increment_to_defect_mean_norm_ratio": (
            float(np.mean(cone_norm_ratios)) if cone_norm_ratios else None
        ),
        "C_cone_increment_on_missing_defect_mean_projection": (
            float(np.mean(cone_projection_coefficients))
            if cone_projection_coefficients
            else None
        ),
    }


def analyze_matched_case(
    parameters: DimerParameters,
    *,
    final_time: float,
    time_step: float,
    phonon_cutoff: int,
    activation_margin: float = 1e-5,
    barrier_rate: float = 5.0,
    cone_tolerance: float = 1e-8,
    maximum_constraints: int = 128,
    exact_relative_tolerance: float = 1e-10,
    exact_absolute_tolerance: float = 1e-12,
    exact_maximum_step: float = 0.02,
    include_exact_defect: bool = False,
    controller_stride: int = 5,
) -> MatchedCase:
    """Run one exact/raw/corrected comparison on a common sample grid."""

    times = _sample_times(final_time, time_step)
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        relative_tolerance=exact_relative_tolerance,
        absolute_tolerance=exact_absolute_tolerance,
        maximum_step=min(exact_maximum_step, time_step),
    )
    exact_coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ],
        dtype=float,
    )
    initial = exact_coordinates[0].copy()
    raw = integrate_closed_rk4(
        parameters,
        initial,
        final_time=final_time,
        time_step=time_step,
        corrected=False,
        activation_margin=activation_margin,
        barrier_rate=barrier_rate,
        cone_tolerance=cone_tolerance,
        maximum_constraints=maximum_constraints,
    )
    corrected = integrate_closed_rk4(
        parameters,
        initial,
        final_time=final_time,
        time_step=time_step,
        corrected=True,
        activation_margin=activation_margin,
        barrier_rate=barrier_rate,
        cone_tolerance=cone_tolerance,
        maximum_constraints=maximum_constraints,
    )
    raw_metrics, raw_errors = _protocol_metrics(
        times,
        exact_coordinates,
        raw,
        parameters,
    )
    corrected_metrics, corrected_errors = _protocol_metrics(
        times,
        exact_coordinates,
        corrected,
        parameters,
    )
    metrics: dict[str, Any] = {
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
            "coupling": parameters.coupling,
        },
        "phonon_cutoff": phonon_cutoff,
        "time_step": time_step,
        "final_time": final_time,
        "exact": {
            "hilbert_space_dimension": 4 * (phonon_cutoff + 1) ** 2,
            "function_evaluations": exact.function_evaluations,
            "maximum_norm_defect": float(
                np.max(np.abs(exact.state_norms - 1.0))
            ),
            "certificates": _trajectory_certificates(
                exact_coordinates,
                times,
            ),
        },
        "raw": raw_metrics,
        "corrected": corrected_metrics,
        "correction_history": _correction_history_metrics(corrected),
    }
    if include_exact_defect:
        metrics["exact_derivative_defect"] = _exact_defect_metrics(
            parameters,
            exact,
            exact_coordinates,
            activation_margin=activation_margin,
            barrier_rate=barrier_rate,
            cone_tolerance=cone_tolerance,
            maximum_constraints=maximum_constraints,
            controller_stride=controller_stride,
        )
    metrics["series_extrema"] = {
        "raw_block_maxima": {
            name: float(np.max(raw_errors[:, index]))
            for index, name in enumerate(BLOCK_FIELDS)
        },
        "corrected_block_maxima": {
            name: float(np.max(corrected_errors[:, index]))
            for index, name in enumerate(BLOCK_FIELDS)
        },
    }
    return MatchedCase(
        parameters=parameters,
        phonon_cutoff=phonon_cutoff,
        time_step=time_step,
        exact=exact,
        exact_coordinates=exact_coordinates,
        raw=raw,
        corrected=corrected,
        metrics=metrics,
    )


def _pauli_ablation_derivative_metrics(
    parameters: DimerParameters,
    exact: ExactDrivenTrajectory,
    exact_coordinates: FloatArray,
) -> dict[str, Any]:
    """Compare raw and Pauli-repaired ``dC`` on exact-trajectory samples."""

    exact_derivatives = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(derivative)
            for derivative in exact.matrix_derivatives
        ],
        dtype=float,
    )
    variants = {
        "raw": closed_scalar_rhs,
        "pauli_repaired": pauli_repaired_closed_scalar_rhs,
    }
    metrics: dict[str, Any] = {}
    for name, rhs in variants.items():
        predicted = np.asarray(
            [
                rhs(time_value, state, parameters)
                for time_value, state in zip(
                    exact.times,
                    exact_coordinates,
                    strict=True,
                )
            ],
            dtype=float,
        )
        c_error = predicted[:, BLOCK_SLICES["C"]] - exact_derivatives[
            :, BLOCK_SLICES["C"]
        ]
        residual_subtracted = c_error - c_error[:1]
        absolute_norm = np.linalg.norm(c_error, axis=1)
        residual_subtracted_norm = np.linalg.norm(
            residual_subtracted,
            axis=1,
        )
        metrics[name] = {
            "absolute_time_rms_l2": float(
                np.sqrt(np.mean(absolute_norm**2))
            ),
            "absolute_maximum_l2": float(np.max(absolute_norm)),
            "residual_subtracted_time_rms_l2": float(
                np.sqrt(np.mean(residual_subtracted_norm**2))
            ),
            "residual_subtracted_maximum_l2": float(
                np.max(residual_subtracted_norm)
            ),
            "residual_subtracted_final_l2": float(
                residual_subtracted_norm[-1]
            ),
        }
    return metrics


def analyze_pauli_repair_ablation(
    parameters: DimerParameters,
    *,
    final_time: float,
    time_step: float,
    phonon_cutoff: int,
    activation_margin: float = 1e-5,
    barrier_rate: float = 5.0,
    cone_tolerance: float = 1e-8,
    maximum_constraints: int = 128,
    exact_relative_tolerance: float = 1e-10,
    exact_absolute_tolerance: float = 1e-12,
    exact_maximum_step: float = 0.02,
) -> PauliRepairAblation:
    """Run raw, Pauli, controller, and Pauli-plus-controller trajectories."""

    times = _sample_times(final_time, time_step)
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=times,
        phonon_cutoff=phonon_cutoff,
        relative_tolerance=exact_relative_tolerance,
        absolute_tolerance=exact_absolute_tolerance,
        maximum_step=min(exact_maximum_step, time_step),
    )
    exact_coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ],
        dtype=float,
    )
    initial = exact_coordinates[0].copy()
    common_arguments = {
        "final_time": final_time,
        "time_step": time_step,
        "activation_margin": activation_margin,
        "barrier_rate": barrier_rate,
        "cone_tolerance": cone_tolerance,
        "maximum_constraints": maximum_constraints,
    }
    raw = integrate_closed_rk4(
        parameters,
        initial,
        corrected=False,
        pauli_repair=False,
        **common_arguments,
    )
    pauli_repaired = integrate_closed_rk4(
        parameters,
        initial,
        corrected=False,
        pauli_repair=True,
        **common_arguments,
    )
    controller = integrate_closed_rk4(
        parameters,
        initial,
        corrected=True,
        pauli_repair=False,
        **common_arguments,
    )
    pauli_repaired_controller = integrate_closed_rk4(
        parameters,
        initial,
        corrected=True,
        pauli_repair=True,
        **common_arguments,
    )
    lanes = {
        "raw": raw,
        "pauli_repaired": pauli_repaired,
        "controller": controller,
        "pauli_repaired_controller": pauli_repaired_controller,
    }
    lane_metrics = {
        name: _protocol_metrics(
            times,
            exact_coordinates,
            trajectory,
            parameters,
        )[0]
        for name, trajectory in lanes.items()
    }
    metrics = {
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "drive_amplitude": parameters.drive_amplitude,
            "pulse_width": parameters.pulse_width,
            "coupling": parameters.coupling,
        },
        "phonon_cutoff": phonon_cutoff,
        "time_step": time_step,
        "final_time": final_time,
        "exact": {
            "hilbert_space_dimension": 4 * (phonon_cutoff + 1) ** 2,
            "function_evaluations": exact.function_evaluations,
            "maximum_norm_defect": float(
                np.max(np.abs(exact.state_norms - 1.0))
            ),
            "certificates": _trajectory_certificates(
                exact_coordinates,
                times,
            ),
        },
        "lanes": lane_metrics,
        "controller_history": _correction_history_metrics(controller),
        "pauli_repaired_controller_history": _correction_history_metrics(
            pauli_repaired_controller
        ),
        "exact_sample_C_derivative_defect": (
            _pauli_ablation_derivative_metrics(
                parameters,
                exact,
                exact_coordinates,
            )
        ),
        "exact_reference_usage": (
            "common initial state and post-run scoring only; never queried by "
            "an integration RHS or controller"
        ),
    }
    return PauliRepairAblation(
        parameters=parameters,
        phonon_cutoff=phonon_cutoff,
        time_step=time_step,
        exact=exact,
        exact_coordinates=exact_coordinates,
        raw=raw,
        pauli_repaired=pauli_repaired,
        controller=controller,
        pauli_repaired_controller=pauli_repaired_controller,
        metrics=metrics,
    )


def _common_time_indices(
    first_times: FloatArray,
    second_times: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
    common, first_indices, second_indices = np.intersect1d(
        np.round(first_times, 12),
        np.round(second_times, 12),
        return_indices=True,
    )
    if common.size < 2:
        raise ValueError("trajectories have no useful common sample grid")
    return first_indices, second_indices


def _trajectory_difference_metrics(
    first_times: FloatArray,
    first: FloatArray,
    second_times: FloatArray,
    second: FloatArray,
) -> dict[str, Any]:
    first_indices, second_indices = _common_time_indices(
        first_times,
        second_times,
    )
    first_common = first[first_indices]
    second_common = second[second_indices]
    coordinate_difference = first_common - second_common
    block_errors = _matrix_block_errors(second_common, first_common)
    return {
        "common_sample_count": int(first_indices.size),
        "maximum_coordinate_l2_difference": float(
            np.max(np.linalg.norm(coordinate_difference, axis=1))
        ),
        "final_coordinate_l2_difference": float(
            np.linalg.norm(coordinate_difference[-1])
        ),
        "maximum_block_frobenius_difference": {
            name: float(np.max(block_errors[:, index]))
            for index, name in enumerate(BLOCK_FIELDS)
        },
        "rms_block_frobenius_difference": {
            name: float(np.sqrt(np.mean(block_errors[:, index] ** 2)))
            for index, name in enumerate(BLOCK_FIELDS)
        },
    }


def _case_arrays(prefix: str, case: MatchedCase) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__times": case.raw.times,
        f"{prefix}__exact": case.exact_coordinates,
        f"{prefix}__raw": case.raw.coordinates,
        f"{prefix}__corrected": case.corrected.coordinates,
        f"{prefix}__correction": case.corrected.correction_coordinates,
        f"{prefix}__equality_only_correction": (
            case.corrected.equality_only_coordinates
        ),
        f"{prefix}__raw_barrier_minima": case.corrected.raw_barrier_minima,
        f"{prefix}__corrected_barrier_minima": (
            case.corrected.corrected_barrier_minima
        ),
        f"{prefix}__joint_mode_weights": case.corrected.joint_mode_weights,
    }


def _parameter_key(parameters: DimerParameters) -> str:
    return (
        f"lambda_{parameters.lambda_ep:g}__gamma_{parameters.gamma:g}"
        f"__drive_{parameters.drive_amplitude:g}"
    ).replace(".", "p")


def _grid_row(case: MatchedCase) -> dict[str, Any]:
    raw = case.metrics["raw"]
    corrected = case.metrics["corrected"]
    history = case.metrics["correction_history"]
    return {
        "lambda_ep": case.parameters.lambda_ep,
        "gamma": case.parameters.gamma,
        "drive_amplitude": case.parameters.drive_amplitude,
        "coupling": case.parameters.coupling,
        "raw_minimum_joint_gram_eigenvalue": raw["certificates"][
            "minimum_joint_gram_eigenvalue"
        ],
        "corrected_minimum_joint_gram_eigenvalue": corrected[
            "certificates"
        ]["minimum_joint_gram_eigenvalue"],
        "raw_rho_dynamic_normalized_rms_error": raw["block_errors"]["rho"][
            "rms_error_over_exact_dynamic_rms"
        ],
        "corrected_rho_dynamic_normalized_rms_error": corrected[
            "block_errors"
        ]["rho"]["rms_error_over_exact_dynamic_rms"],
        "raw_C_dynamic_normalized_rms_error": raw["block_errors"]["C"][
            "rms_error_over_exact_dynamic_rms"
        ],
        "corrected_C_dynamic_normalized_rms_error": corrected[
            "block_errors"
        ]["C"]["rms_error_over_exact_dynamic_rms"],
        "maximum_correction_norm": history["maximum_correction_norm"],
        "active_sample_fraction": history["active_sample_fraction"],
        "equality_action_sample_fraction": history[
            "equality_action_sample_fraction"
        ],
        "additional_cone_action_sample_fraction": history[
            "additional_cone_action_sample_fraction"
        ],
        "weighted_joint_mode_boson_fraction": history[
            "weighted_joint_mode_boson_fraction"
        ],
        "weighted_joint_mode_electron_fraction": history[
            "weighted_joint_mode_electron_fraction"
        ],
    }


def _model_extension_decision(
    baseline: MatchedCase,
    time_step_convergence: dict[str, Any],
    cutoff_convergence: dict[str, Any],
    *,
    dynamic_error_threshold: float,
    numerical_separation_factor: float,
) -> dict[str, Any]:
    corrected_c = baseline.metrics["corrected"]["block_errors"]["C"]
    model_error = corrected_c["maximum_frobenius_error"]
    dynamic_normalized_error = corrected_c[
        "rms_error_over_exact_dynamic_rms"
    ]
    time_step_floor = time_step_convergence["0.01_vs_0.005"][
        "maximum_block_frobenius_difference"
    ]["C"]
    cutoff_floor = cutoff_convergence["16_vs_20"]["exact"][
        "maximum_block_frobenius_difference"
    ]["C"]
    numerical_floor = max(time_step_floor, cutoff_floor, 1e-14)
    separation = model_error / numerical_floor
    warranted = bool(
        dynamic_normalized_error > dynamic_error_threshold
        and separation > numerical_separation_factor
    )
    return {
        "retain_K_extension_warranted": warranted,
        "decision_rule": {
            "C_rms_error_over_exact_dynamic_rms_must_exceed": (
                dynamic_error_threshold
            ),
            "model_error_over_numerical_floor_must_exceed": (
                numerical_separation_factor
            ),
        },
        "measured": {
            "corrected_C_maximum_frobenius_error": model_error,
            "corrected_C_rms_error_over_exact_dynamic_rms": (
                dynamic_normalized_error
            ),
            "time_step_C_numerical_floor": time_step_floor,
            "cutoff_C_numerical_floor": cutoff_floor,
            "model_error_over_numerical_floor": separation,
        },
        "interpretation": (
            "The joint barrier stabilizes representability, but the remaining "
            "converged C error warrants adding the discarded mixed moment K "
            "and then testing the smaller opposite-spin covariance."
            if warranted
            else
            "The converged C error does not yet justify enlarging the closure."
        ),
    }


def _write_baseline_figures(run_directory: Path, baseline: MatchedCase) -> None:
    times = baseline.raw.times
    raw_errors = _matrix_block_errors(
        baseline.exact_coordinates,
        baseline.raw.coordinates,
    )
    corrected_errors = _matrix_block_errors(
        baseline.exact_coordinates,
        baseline.corrected.coordinates,
    )
    figure, axes = plt.subplots(3, 2, figsize=(9.0, 9.5), sharex=True)
    for index, (name, axis) in enumerate(zip(BLOCK_FIELDS, axes.flat[:5])):
        axis.plot(times, raw_errors[:, index], color="#a52a2a", label="raw")
        axis.plot(
            times,
            corrected_errors[:, index],
            color="#174a7e",
            label="joint barrier",
        )
        axis.set_title(f"{name} block")
        axis.set_ylabel("Frobenius error")
        axis.grid(alpha=0.22)
    axes.flat[0].legend(frameon=False)
    joint_axis = axes.flat[5]
    for coordinates, label, color in (
        (baseline.exact_coordinates, "exact", "#28724f"),
        (baseline.raw.coordinates, "raw", "#a52a2a"),
        (baseline.corrected.coordinates, "joint barrier", "#174a7e"),
    ):
        minima = [
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(row)
                )
            )[0]
            for row in coordinates
        ]
        joint_axis.plot(times, minima, label=label, color=color)
    joint_axis.axhline(0.0, color="black", linewidth=0.8)
    joint_axis.set_title("joint Gram certificate")
    joint_axis.set_ylabel(r"$\lambda_{\min}(\mathcal{G})$")
    joint_axis.legend(frameon=False)
    joint_axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel(r"time $t\,t_{\rm hop}$")
    figure.suptitle("Exact versus 31-coordinate closure over the driven horizon")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            run_directory / f"baseline_exact_raw_corrected.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)

    correction = baseline.corrected.correction_coordinates
    equality = baseline.corrected.equality_only_coordinates
    correction_norm = np.linalg.norm(correction, axis=1)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.8), sharex=True)
    for name, block in BLOCK_SLICES.items():
        axes[0, 0].plot(
            times,
            np.linalg.norm(correction[:, block], axis=1),
            label=name,
        )
    axes[0, 0].set_title("correction by coordinate block")
    axes[0, 0].set_ylabel(r"$\|u_{\rm block}\|_2$")
    axes[0, 0].legend(ncol=3, frameon=False)
    axes[0, 1].plot(times, correction_norm, label="full correction")
    axes[0, 1].plot(
        times,
        np.linalg.norm(equality, axis=1),
        label="energy + trace only",
    )
    axes[0, 1].plot(
        times,
        np.linalg.norm(correction - equality, axis=1),
        label="additional cone action",
    )
    axes[0, 1].set_title("equality and cone contributions")
    axes[0, 1].set_ylabel(r"coordinate $\ell_2$ norm")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].stackplot(
        times,
        baseline.corrected.joint_mode_weights[:, :4].sum(axis=1),
        baseline.corrected.joint_mode_weights[:, 4:].sum(axis=1),
        labels=("bosonic entries", "electronic entries"),
        colors=("#d4a017", "#3f7cac"),
        alpha=0.8,
    )
    axes[1, 0].set_title("weakest joint-barrier eigenmode")
    axes[1, 0].set_ylabel("mode weight")
    axes[1, 0].legend(frameon=False)
    for index, name in enumerate(BARRIER_SECTORS):
        axes[1, 1].plot(
            times,
            baseline.corrected.raw_barrier_minima[:, index],
            label=name,
        )
    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_title("uncorrected barrier velocities")
    axes[1, 1].set_ylabel("minimum eigenvalue")
    axes[1, 1].legend(frameon=False)
    for axis in axes.flat:
        axis.grid(alpha=0.22)
    for axis in axes[-1]:
        axis.set_xlabel(r"time $t\,t_{\rm hop}$")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            run_directory / f"correction_history.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_convergence_figure(
    run_directory: Path,
    time_step_convergence: dict[str, Any],
    cutoff_convergence: dict[str, Any],
) -> None:
    names = tuple(BLOCK_FIELDS)
    dt_values = [
        time_step_convergence["0.01_vs_0.005"][
            "maximum_block_frobenius_difference"
        ][name]
        for name in names
    ]
    cutoff_values = [
        cutoff_convergence["16_vs_20"]["exact"][
            "maximum_block_frobenius_difference"
        ][name]
        for name in names
    ]
    positions = np.arange(len(names), dtype=float)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    width = 0.36
    axis.bar(
        positions - width / 2,
        dt_values,
        width,
        label=r"corrected $\Delta t$: .01 vs .005",
    )
    axis.bar(
        positions + width / 2,
        cutoff_values,
        width,
        label="exact cutoff: 16 vs 20",
    )
    axis.set_yscale("log")
    axis.set_xticks(positions, names)
    axis.set_ylabel("maximum block difference")
    axis.set_title("Numerical convergence scales")
    axis.grid(axis="y", which="both", alpha=0.22)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            run_directory / f"convergence_scales.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _write_grid_figure(run_directory: Path, rows: list[dict[str, Any]]) -> None:
    labels = [
        rf"$({row['lambda_ep']:g},{row['gamma']:g},{row['drive_amplitude']:g})$"
        for row in rows
    ]
    positions = np.arange(len(rows), dtype=float)
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    axes[0].plot(
        positions,
        [row["raw_C_dynamic_normalized_rms_error"] for row in rows],
        "o-",
        label="raw C error",
        color="#a52a2a",
    )
    axes[0].plot(
        positions,
        [row["corrected_C_dynamic_normalized_rms_error"] for row in rows],
        "o-",
        label="corrected C error",
        color="#174a7e",
    )
    axes[0].axhline(0.1, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("C RMS error / exact dynamic RMS")
    axes[0].legend(frameon=False)
    axes[1].plot(
        positions,
        [row["maximum_correction_norm"] for row in rows],
        "o-",
        color="#6a3d9a",
        label="maximum correction",
    )
    axes[1].set_ylabel(r"maximum $\|u\|_2$")
    axes[1].set_xticks(positions, labels, rotation=45, ha="right")
    axes[1].set_xlabel(r"$(\lambda,\gamma,V)$")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    figure.suptitle("Joint-barrier robustness grid")
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(
            run_directory / f"parameter_grid.{suffix}",
            dpi=220 if suffix == "png" else None,
            bbox_inches="tight",
        )
    plt.close(figure)


def _validate_run_contract(run_directory: Path) -> tuple[dict[str, Any], Path]:
    plan_path = run_directory / "plan.json"
    authorization_path = run_directory / "authorization.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if plan["execution_authorized"]:
        raise RuntimeError("the immutable plan must remain unauthorized")
    if plan["classification"] != "diagnostic":
        raise RuntimeError("this runner accepts diagnostic plans only")
    if plan["evidence_status"] != "exploratory_local_not_promoted":
        raise RuntimeError("diagnostic output must remain unpromoted")
    if not authorization["authorized"]:
        raise RuntimeError("current user authorization is required")
    if authorization["run_id"] != plan["run_id"]:
        raise RuntimeError("authorization and plan run IDs do not match")
    repository_root = Path(__file__).resolve().parents[4]
    for relative_path, expected_hash in plan["source_hashes"].items():
        actual_hash = _sha256(repository_root / relative_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"source hash mismatch for {relative_path}: "
                f"{actual_hash} != {expected_hash}"
            )
    return plan, repository_root


def run_analysis(run_directory: Path) -> dict[str, Any]:
    """Execute the five-part electron-phonon diagnostic plan."""

    run_directory = run_directory.resolve()
    plan, _repository_root = _validate_run_contract(run_directory)
    runtime_manifest_path = run_directory / "runtime_manifest.json"
    progress_path = run_directory / "progress.jsonl"
    partial_summary_path = run_directory / "summary.partial.json"
    summary_path = run_directory / "summary.json"
    if runtime_manifest_path.exists() or summary_path.exists():
        raise RuntimeError("this immutable run directory already has runtime output")

    started_clock = time.perf_counter()
    runtime_manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "classification": plan["classification"],
        "evidence_status": plan["evidence_status"],
        "status": "running",
        "started_at_utc": _utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "plan_sha256": _sha256(run_directory / "plan.json"),
        "authorization_sha256": _sha256(run_directory / "authorization.json"),
        "source_hashes": plan["source_hashes"],
        "exact_reference_usage": (
            "post-run comparison only; never queried by the controller"
        ),
    }
    _write_json_atomic(runtime_manifest_path, runtime_manifest)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": plan["run_id"],
        "status": "running",
        "scientific_question": plan["scientific_question"],
    }

    parameter_values = plan["baseline_parameters"]
    baseline_parameters = DimerParameters(**parameter_values)
    integration = plan["integration"]
    controller = plan["controller"]
    final_time = float(integration["final_time"])
    baseline_cutoff = int(integration["baseline_phonon_cutoff"])
    baseline_time_step = float(integration["baseline_time_step"])
    common_arguments = {
        "final_time": final_time,
        "activation_margin": float(controller["activation_margin"]),
        "barrier_rate": float(controller["barrier_rate"]),
        "cone_tolerance": float(controller["cone_tolerance"]),
        "maximum_constraints": int(controller["maximum_constraints"]),
        "exact_relative_tolerance": float(
            integration["exact_relative_tolerance"]
        ),
        "exact_absolute_tolerance": float(
            integration["exact_absolute_tolerance"]
        ),
        "exact_maximum_step": float(integration["exact_maximum_step"]),
    }
    case_cache: dict[tuple[float, float, float, int, float], MatchedCase] = {}

    def get_case(
        parameters: DimerParameters,
        cutoff: int,
        time_step: float,
        *,
        include_exact_defect: bool = False,
    ) -> MatchedCase:
        key = (
            parameters.lambda_ep,
            parameters.gamma,
            parameters.drive_amplitude,
            cutoff,
            time_step,
        )
        cached = case_cache.get(key)
        if cached is not None and (
            not include_exact_defect or "exact_derivative_defect" in cached.metrics
        ):
            return cached
        case = analyze_matched_case(
            parameters,
            phonon_cutoff=cutoff,
            time_step=time_step,
            include_exact_defect=include_exact_defect,
            controller_stride=int(integration["controller_analysis_stride"]),
            **common_arguments,
        )
        case_cache[key] = case
        return case

    try:
        _append_progress(progress_path, {"event": "baseline_started"})
        baseline = get_case(
            baseline_parameters,
            baseline_cutoff,
            baseline_time_step,
            include_exact_defect=True,
        )
        summary["baseline"] = baseline.metrics
        _write_npz_atomic(
            run_directory / "baseline_trajectories.npz",
            _case_arrays("baseline", baseline),
        )
        _write_baseline_figures(run_directory, baseline)
        _write_json_atomic(partial_summary_path, summary)
        _append_progress(
            progress_path,
            {
                "event": "baseline_completed",
                "raw_C_dynamic_normalized_rms_error": baseline.metrics["raw"][
                    "block_errors"
                ]["C"]["rms_error_over_exact_dynamic_rms"],
                "corrected_C_dynamic_normalized_rms_error": baseline.metrics[
                    "corrected"
                ]["block_errors"]["C"]["rms_error_over_exact_dynamic_rms"],
                "maximum_correction_norm": baseline.metrics[
                    "correction_history"
                ]["maximum_correction_norm"],
            },
        )

        time_step_cases: dict[str, MatchedCase] = {}
        for time_step_value in integration["time_steps"]:
            time_step = float(time_step_value)
            label = f"{time_step:g}"
            _append_progress(
                progress_path,
                {"event": "time_step_case_started", "time_step": time_step},
            )
            time_step_cases[label] = get_case(
                baseline_parameters,
                baseline_cutoff,
                time_step,
            )
            _append_progress(
                progress_path,
                {"event": "time_step_case_completed", "time_step": time_step},
            )
        time_step_convergence: dict[str, Any] = {}
        ordered_steps = tuple(float(value) for value in integration["time_steps"])
        for coarse, fine in zip(ordered_steps[:-1], ordered_steps[1:]):
            coarse_case = time_step_cases[f"{coarse:g}"]
            fine_case = time_step_cases[f"{fine:g}"]
            time_step_convergence[f"{coarse:g}_vs_{fine:g}"] = (
                _trajectory_difference_metrics(
                    coarse_case.corrected.times,
                    coarse_case.corrected.coordinates,
                    fine_case.corrected.times,
                    fine_case.corrected.coordinates,
                )
            )
        summary["time_step_convergence"] = time_step_convergence

        cutoff_cases: dict[int, MatchedCase] = {}
        for cutoff_value in integration["phonon_cutoffs"]:
            cutoff = int(cutoff_value)
            _append_progress(
                progress_path,
                {"event": "cutoff_case_started", "phonon_cutoff": cutoff},
            )
            cutoff_cases[cutoff] = get_case(
                baseline_parameters,
                cutoff,
                baseline_time_step,
            )
            _append_progress(
                progress_path,
                {"event": "cutoff_case_completed", "phonon_cutoff": cutoff},
            )
        cutoff_convergence: dict[str, Any] = {}
        ordered_cutoffs = tuple(int(value) for value in integration["phonon_cutoffs"])
        for lower, upper in zip(ordered_cutoffs[:-1], ordered_cutoffs[1:]):
            lower_case = cutoff_cases[lower]
            upper_case = cutoff_cases[upper]
            cutoff_convergence[f"{lower}_vs_{upper}"] = {
                "exact": _trajectory_difference_metrics(
                    lower_case.exact.times,
                    lower_case.exact_coordinates,
                    upper_case.exact.times,
                    upper_case.exact_coordinates,
                ),
                "raw": _trajectory_difference_metrics(
                    lower_case.raw.times,
                    lower_case.raw.coordinates,
                    upper_case.raw.times,
                    upper_case.raw.coordinates,
                ),
                "corrected": _trajectory_difference_metrics(
                    lower_case.corrected.times,
                    lower_case.corrected.coordinates,
                    upper_case.corrected.times,
                    upper_case.corrected.coordinates,
                ),
            }
        summary["cutoff_convergence"] = cutoff_convergence
        convergence_arrays: dict[str, np.ndarray] = {}
        for label, case in time_step_cases.items():
            convergence_arrays.update(_case_arrays(f"dt_{label}", case))
        for cutoff, case in cutoff_cases.items():
            convergence_arrays.update(_case_arrays(f"cutoff_{cutoff}", case))
        _write_npz_atomic(
            run_directory / "convergence_trajectories.npz",
            convergence_arrays,
        )
        _write_convergence_figure(
            run_directory,
            time_step_convergence,
            cutoff_convergence,
        )
        _write_json_atomic(partial_summary_path, summary)

        grid = plan["parameter_grid"]
        grid_rows: list[dict[str, Any]] = []
        grid_arrays: dict[str, np.ndarray] = {}
        for lambda_ep, gamma, drive in product(
            grid["lambda_ep"],
            grid["gamma"],
            grid["drive_amplitude"],
        ):
            parameters = DimerParameters(
                hopping=baseline_parameters.hopping,
                lambda_ep=float(lambda_ep),
                gamma=float(gamma),
                drive_amplitude=float(drive),
                pulse_width=baseline_parameters.pulse_width,
            )
            key = _parameter_key(parameters)
            _append_progress(
                progress_path,
                {
                    "event": "grid_case_started",
                    "key": key,
                    "lambda_ep": parameters.lambda_ep,
                    "gamma": parameters.gamma,
                    "drive_amplitude": parameters.drive_amplitude,
                },
            )
            case = get_case(
                parameters,
                int(grid["phonon_cutoff"]),
                float(grid["time_step"]),
            )
            row = _grid_row(case)
            grid_rows.append(row)
            grid_arrays.update(_case_arrays(key, case))
            _append_progress(
                progress_path,
                {
                    "event": "grid_case_completed",
                    "key": key,
                    "corrected_C_dynamic_normalized_rms_error": row[
                        "corrected_C_dynamic_normalized_rms_error"
                    ],
                    "maximum_correction_norm": row["maximum_correction_norm"],
                },
            )
            summary["parameter_grid"] = {
                "phonon_cutoff": int(grid["phonon_cutoff"]),
                "time_step": float(grid["time_step"]),
                "rows": grid_rows,
            }
            _write_json_atomic(partial_summary_path, summary)
        _write_npz_atomic(
            run_directory / "parameter_grid_trajectories.npz",
            grid_arrays,
        )
        _write_grid_figure(run_directory, grid_rows)

        decision = plan["model_extension_decision"]
        summary["model_extension_decision"] = _model_extension_decision(
            baseline,
            time_step_convergence,
            cutoff_convergence,
            dynamic_error_threshold=float(
                decision["C_dynamic_normalized_rms_threshold"]
            ),
            numerical_separation_factor=float(
                decision["numerical_separation_factor"]
            ),
        )
        summary["status"] = "complete"
        summary["wall_elapsed_seconds"] = float(
            time.perf_counter() - started_clock
        )
        _write_json_atomic(summary_path, summary)
        _write_json_atomic(partial_summary_path, summary)
        _append_progress(
            progress_path,
            {
                "event": "run_completed",
                "wall_elapsed_seconds": summary["wall_elapsed_seconds"],
                "retain_K_extension_warranted": summary[
                    "model_extension_decision"
                ]["retain_K_extension_warranted"],
            },
        )
        artifact_paths = tuple(
            path
            for path in run_directory.iterdir()
            if path.is_file()
            and path.name not in {"runtime_manifest.json", "authorization.json"}
        )
        runtime_manifest.update(
            {
                "status": "complete",
                "finished_at_utc": _utc_now(),
                "wall_elapsed_seconds": summary["wall_elapsed_seconds"],
                "artifact_hashes": {
                    path.name: _sha256(path) for path in artifact_paths
                },
            }
        )
        _write_json_atomic(runtime_manifest_path, runtime_manifest)
        return summary
    except BaseException as error:
        runtime_manifest.update(
            {
                "status": (
                    "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
                ),
                "finished_at_utc": _utc_now(),
                "wall_elapsed_seconds": float(time.perf_counter() - started_clock),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_json_atomic(runtime_manifest_path, runtime_manifest)
        _append_progress(
            progress_path,
            {
                "event": runtime_manifest["status"],
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-directory",
        type=Path,
        required=True,
        help="Directory containing immutable plan.json and authorization.json.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    run_analysis(args.run_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
