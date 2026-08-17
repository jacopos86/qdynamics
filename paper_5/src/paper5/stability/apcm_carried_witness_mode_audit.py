"""Audit how negative carried-Gram modes couple into the retained velocity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .apcm_carried_witness import CarriedWitnessModel
from .hubbard_dimer import DimerParameters, FloatArray
from .matrix_reference import (
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-step", type=float, default=0.05)
    parser.add_argument("--coupling-horizon", type=float, default=0.05)
    parser.add_argument("--negative-threshold", type=float, default=1e-8)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    return parser


def _directional_retained_derivative(
    model: CarriedWitnessModel,
    time: float,
    retained: FloatArray,
    completion: FloatArray,
    direction: FloatArray,
    *,
    finite_difference_step: float = 1e-6,
) -> FloatArray:
    plus = model.retained_velocity(
        time,
        retained,
        completion + finite_difference_step * direction,
    )
    minus = model.retained_velocity(
        time,
        retained,
        completion - finite_difference_step * direction,
    )
    return (plus - minus) / (2.0 * finite_difference_step)


def negative_mode_coupling(
    model: CarriedWitnessModel,
    time: float,
    state: FloatArray,
    *,
    negative_threshold: float = 1e-8,
    coupling_horizon: float = 0.05,
) -> dict[str, float | int]:
    """Return local coupling diagnostics for the negative extended-Gram modes.

    For an eigenpair ``(lambda, v)``, the vector with entries
    ``v^dagger (dG/dz_j) v`` is the eigenvalue gradient in the scaled hidden
    coordinates.  The minimum-Euclidean-norm linearized displacement that
    raises that eigenvalue to zero is used only as an offline sensitivity
    direction.  No trajectory state is repaired here.
    """

    retained, completion = model.geometry.unpack_state(state)
    matrix = model.geometry.scaled_unified_matrix(retained, completion)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    coefficients = model.geometry.scaled_completion_coefficients()
    retained_velocity = model.retained_velocity(time, retained, completion)
    completion_velocity = model.desired_completion_velocity(
        time, retained, completion
    )
    matrix_rate = (
        model.geometry.scaled_unified_matrix(
            retained + retained_velocity,
            completion + completion_velocity,
        )
        - matrix
    )
    retained_norm = max(float(np.linalg.norm(retained_velocity)), 1e-15)
    correlation_norm = max(
        float(np.linalg.norm(retained_velocity[17:29])), 1e-15
    )
    entrance_norm = max(
        float(np.linalg.norm(retained_velocity[29:])), 1e-15
    )
    negative_indices = np.flatnonzero(eigenvalues < -negative_threshold)

    weakest_index = 0
    weakest_vector = eigenvectors[:, weakest_index]
    weakest_velocity = float(
        np.real(weakest_vector.conjugate() @ matrix_rate @ weakest_vector)
    )
    weakest_repair_norm = 0.0
    weakest_retained_relative = 0.0
    weakest_correlation_relative = 0.0
    maximum_retained_relative = 0.0
    maximum_correlation_relative = 0.0
    maximum_entrance_relative = 0.0
    maximum_two_stage_c_relative = 0.0

    audited_indices = (
        negative_indices
        if negative_indices.size
        else np.asarray([weakest_index], dtype=int)
    )
    for index in audited_indices:
        vector = eigenvectors[:, int(index)]
        gradient = np.asarray(
            np.einsum(
                "a,jab,b->j",
                vector.conjugate(),
                coefficients,
                vector,
                optimize=True,
            ).real,
            dtype=float,
        )
        gradient_norm = float(np.linalg.norm(gradient))
        if gradient_norm <= 1e-15:
            continue
        direction = gradient / gradient_norm
        repair_norm = max(0.0, -float(eigenvalues[int(index)])) / gradient_norm
        derivative = _directional_retained_derivative(
            model, time, retained, completion, direction
        )
        predicted = repair_norm * derivative
        retained_relative = float(np.linalg.norm(predicted)) / retained_norm
        correlation_relative = (
            float(np.linalg.norm(predicted[17:29])) / correlation_norm
        )
        entrance_relative = (
            float(np.linalg.norm(predicted[29:])) / entrance_norm
        )
        two_stage_velocity = model.retained_velocity(
            time,
            retained + coupling_horizon * predicted,
            completion,
        )
        two_stage_c_relative = (
            float(
                np.linalg.norm(
                    two_stage_velocity[17:29]
                    - retained_velocity[17:29]
                )
            )
            / correlation_norm
        )
        maximum_retained_relative = max(
            maximum_retained_relative, retained_relative
        )
        maximum_correlation_relative = max(
            maximum_correlation_relative, correlation_relative
        )
        maximum_entrance_relative = max(
            maximum_entrance_relative, entrance_relative
        )
        maximum_two_stage_c_relative = max(
            maximum_two_stage_c_relative, two_stage_c_relative
        )
        if int(index) == weakest_index:
            weakest_repair_norm = repair_norm
            weakest_retained_relative = retained_relative
            weakest_correlation_relative = correlation_relative

    return {
        "minimum_eigenvalue": float(eigenvalues[0]),
        "negative_mode_count": int(negative_indices.size),
        "weakest_eigenvalue_velocity": weakest_velocity,
        "weakest_linearized_repair_coordinate_norm": weakest_repair_norm,
        "weakest_predicted_retained_velocity_relative_change": (
            weakest_retained_relative
        ),
        "weakest_predicted_c_velocity_relative_change": (
            weakest_correlation_relative
        ),
        "maximum_negative_mode_predicted_retained_velocity_relative_change": (
            maximum_retained_relative
        ),
        "maximum_negative_mode_predicted_c_velocity_relative_change": (
            maximum_correlation_relative
        ),
        "maximum_negative_mode_predicted_entrance_velocity_relative_change": (
            maximum_entrance_relative
        ),
        "maximum_negative_mode_two_stage_c_velocity_relative_change": (
            maximum_two_stage_c_relative
        ),
    }


def _first_crossing(
    times: FloatArray, values: FloatArray, threshold: float
) -> float | None:
    indices = np.flatnonzero(values < threshold)
    return None if indices.size == 0 else float(times[int(indices[0])])


def _first_exceedance(
    times: FloatArray, values: FloatArray, threshold: float
) -> float | None:
    indices = np.flatnonzero(values > threshold)
    return None if indices.size == 0 else float(times[int(indices[0])])


def main() -> int:
    args = _parser().parse_args()
    if (
        args.sample_step <= 0.0
        or args.coupling_horizon <= 0.0
        or args.negative_threshold <= 0.0
    ):
        raise ValueError(
            "sample step, coupling horizon, and negative threshold must be positive"
        )
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    with np.load(args.trajectory.resolve(), allow_pickle=False) as arrays:
        times = np.asarray(arrays["times"], dtype=float)
        states = np.asarray(arrays["carried_states"], dtype=float)
        approximate = np.asarray(
            arrays["approximate_archive_coordinates"], dtype=float
        )
        exact = np.asarray(arrays["exact_archive_coordinates"], dtype=float)
        extended_minima = np.asarray(
            arrays["minimum_unshifted_eigenvalues"], dtype=float
        )

    parameters = DimerParameters(
        hopping=1.0,
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
    )
    model = CarriedWitnessModel(parameters)
    model.prepare(phonon_cutoff=args.phonon_cutoff)
    model.geometry.unpack_state(states[0])

    target_times = np.arange(
        float(times[0]),
        float(times[-1]) + 0.5 * args.sample_step,
        args.sample_step,
    )
    sampled_indices = np.unique(
        np.asarray(
            [int(np.argmin(np.abs(times - value))) for value in target_times],
            dtype=int,
        )
    )
    sampled_times = times[sampled_indices]
    diagnostics = [
        negative_mode_coupling(
            model,
            float(times[index]),
            states[index],
            negative_threshold=args.negative_threshold,
            coupling_horizon=args.coupling_horizon,
        )
        for index in sampled_indices
    ]
    diagnostic_names = tuple(diagnostics[0])
    diagnostic_values = np.asarray(
        [[float(row[name]) for name in diagnostic_names] for row in diagnostics],
        dtype=float,
    )

    retained_minima = np.asarray(
        [
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    closed_scalar_to_matrix_state(row)
                )
            )[0]
            for row in approximate
        ],
        dtype=float,
    )
    dynamic_difference = (approximate - exact) - (approximate[0] - exact[0])
    instantaneous_dynamic_scalar_rms = np.sqrt(
        np.mean(dynamic_difference**2, axis=1)
    )
    maximum_mode_coupling = diagnostic_values[
        :,
        diagnostic_names.index(
            "maximum_negative_mode_predicted_retained_velocity_relative_change"
        ),
    ]
    maximum_two_stage_c_coupling = diagnostic_values[
        :,
        diagnostic_names.index(
            "maximum_negative_mode_two_stage_c_velocity_relative_change"
        ),
    ]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "offline_no_guard_mode_relevance_audit",
        "trajectory": str(args.trajectory.resolve()),
        "time_interval": [float(times[0]), float(times[-1])],
        "sample_step": args.sample_step,
        "coupling_horizon": args.coupling_horizon,
        "negative_threshold": args.negative_threshold,
        "first_crossings": {
            "extended_gram_below_minus_1e-8": _first_crossing(
                times, extended_minima, -1e-8
            ),
            "retained_joint_gram_below_minus_1e-8": _first_crossing(
                times, retained_minima, -1e-8
            ),
            "dynamic_coordinate_rms_above_1e-3": _first_exceedance(
                times, instantaneous_dynamic_scalar_rms, 1e-3
            ),
            "dynamic_coordinate_rms_above_1e-2": _first_exceedance(
                times, instantaneous_dynamic_scalar_rms, 1e-2
            ),
            "dynamic_coordinate_rms_above_1e-1": _first_exceedance(
                times, instantaneous_dynamic_scalar_rms, 1e-1
            ),
            "predicted_retained_velocity_change_above_1e-2": (
                _first_exceedance(sampled_times, maximum_mode_coupling, 1e-2)
            ),
            "predicted_retained_velocity_change_above_1e-1": (
                _first_exceedance(sampled_times, maximum_mode_coupling, 1e-1)
            ),
            "two_stage_c_velocity_change_above_1e-2": _first_exceedance(
                sampled_times, maximum_two_stage_c_coupling, 1e-2
            ),
            "two_stage_c_velocity_change_above_1e-1": _first_exceedance(
                sampled_times, maximum_two_stage_c_coupling, 1e-1
            ),
        },
        "extrema": {
            "minimum_extended_gram_eigenvalue": float(
                np.min(extended_minima)
            ),
            "minimum_retained_joint_gram_eigenvalue": float(
                np.min(retained_minima)
            ),
            "maximum_instantaneous_dynamic_coordinate_rms": float(
                np.max(instantaneous_dynamic_scalar_rms)
            ),
            "maximum_predicted_retained_velocity_relative_change": float(
                np.max(maximum_mode_coupling)
            ),
            "maximum_two_stage_c_velocity_relative_change": float(
                np.max(maximum_two_stage_c_coupling)
            ),
        },
        "diagnostic_definition": {
            "metric": (
                "minimum-Euclidean-norm linearized hidden-coordinate repair "
                "for each negative extended-Gram eigenmode, propagated "
                "through the directional derivative of the retained velocity"
            ),
            "two_stage_c_metric": (
                "advance the induced entrance-rate change for the declared "
                "coupling horizon, then reevaluate the C velocity"
            ),
            "online_use": False,
            "coordinate_metric": "scaled hidden-coordinate Euclidean metric",
        },
    }
    np.savez_compressed(
        output_directory / "mode_coupling_audit.npz",
        times=times,
        extended_gram_minimum_eigenvalues=extended_minima,
        retained_joint_gram_minimum_eigenvalues=retained_minima,
        instantaneous_dynamic_coordinate_scalar_rms=(
            instantaneous_dynamic_scalar_rms
        ),
        sampled_times=sampled_times,
        diagnostic_names=np.asarray(diagnostic_names),
        diagnostic_values=diagnostic_values,
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
