"""Replay and trajectory parity audit for the opt-in M4 block solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from pipelines.open_dynamics.plot_paper_v_results_progression import (
    OBSERVABLE_NAMES,
    _observables,
)

from .adaptive_positive_moment import (
    HIDDEN_RELATIVE_MOMENT_KEYS,
    raw_moment_coordinates_to_matrix_state,
    relative_moments_from_matrix_state,
)
from .adaptive_positive_moment_propagation import (
    APCMSettings,
    ArchiveBackedAPCM,
    prepare_apcm_initial_state,
    unpack_apcm_state,
)
from .hubbard_dimer import DimerParameters
from .moment_hierarchy import MomentKey, ZERO_CUMULANT_CLOSURE
from .positive_moment_completion import (
    PositiveFourthMomentCompletion,
    PositiveMomentCompletionSettings,
    _frontier_scale,
    pauli_weyl_moment_matrix,
)


VALUE_TOLERANCE = 1.0e-6
CONTROLLER_TOLERANCE = 3.0e-13
OBSERVABLE_TOLERANCE = 1.0e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-a-trajectory", type=Path)
    parser.add_argument("--serial-trajectory", type=Path)
    parser.add_argument("--candidate-trajectory", type=Path)
    parser.add_argument("--full-reference-trajectory", type=Path)
    parser.add_argument("--candidate-continuation-trajectory", type=Path)
    parser.add_argument("--serial-prefix-summary", type=Path)
    parser.add_argument("--continuation-start-index", type=int, default=400)
    parser.add_argument("--full-only", action="store_true")
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=11)
    return parser


def _completion(
    representation: str,
    threads: int,
) -> PositiveFourthMomentCompletion:
    return PositiveFourthMomentCompletion(
        PositiveMomentCompletionSettings(
            phonon_envelope=16.0,
            logdet_weight=1.0,
            logdet_shift=1e-5,
            solver_tolerance=1e-7,
            maximum_iterations=2_000,
            cone_representation=representation,
            clarabel_max_threads=threads,
        )
    )


def _difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    reference_values = np.asarray(reference)
    candidate_values = np.asarray(candidate)
    if reference_values.shape != candidate_values.shape:
        raise ValueError("parity arrays have different shapes")
    reference_finite = np.isfinite(reference_values)
    candidate_finite = np.isfinite(candidate_values)
    if not np.array_equal(reference_finite, candidate_finite):
        return {
            "maximum_absolute_difference": float("inf"),
            "rms_difference": float("inf"),
        }
    if not np.any(reference_finite):
        return {
            "maximum_absolute_difference": 0.0,
            "rms_difference": 0.0,
        }
    values = (
        candidate_values[reference_finite]
        - reference_values[reference_finite]
    )
    return {
        "maximum_absolute_difference": float(np.max(np.abs(values))),
        "rms_difference": float(np.sqrt(np.mean(np.abs(values) ** 2))),
    }


def _moments_from_state(state: np.ndarray) -> dict[MomentKey, float]:
    raw, hidden = unpack_apcm_state(state)
    matrix_state = raw_moment_coordinates_to_matrix_state(raw)
    _, moments = relative_moments_from_matrix_state(matrix_state, hidden)
    return moments


def _retraction_objective(
    target: Mapping[MomentKey, float],
    corrected_lower: Mapping[MomentKey, float],
    frontier: Mapping[MomentKey, float],
    completion: PositiveFourthMomentCompletion,
) -> float:
    lower_scales = np.asarray(
        [_frontier_scale(key, 16.0) for key in HIDDEN_RELATIVE_MOMENT_KEYS],
        dtype=float,
    )
    lower_delta = np.asarray(
        [
            corrected_lower[key] - target[key]
            for key in HIDDEN_RELATIVE_MOMENT_KEYS
        ],
        dtype=float,
    )
    prior = ZERO_CUMULANT_CLOSURE.prepare(target, 3)
    frontier_scales = np.asarray(
        [_frontier_scale(key, 16.0) for key in completion.frontier_keys],
        dtype=float,
    )
    frontier_delta = np.asarray(
        [frontier[key] - prior.moment(key) for key in completion.frontier_keys],
        dtype=float,
    )
    return float(
        0.5 * np.sum((lower_delta / lower_scales) ** 2)
        + 1.0e-8 * np.sum((frontier_delta / frontier_scales) ** 2)
    )


def _sample_indices(arrays: Mapping[str, np.ndarray], count: int) -> np.ndarray:
    if count < 2:
        raise ValueError("sample_count must be at least two")
    times = arrays["times"]
    selected = set(
        np.linspace(0, times.size - 1, count, dtype=int).tolist()
    )
    for key in (
        "hidden_retraction_norms",
        "completion_minimum_eigenvalues",
        "joint_gram_minimum_eigenvalues",
    ):
        values = np.asarray(arrays[key], dtype=float)
        selected.add(int(np.nanargmin(values)))
        selected.add(int(np.nanargmax(values)))
    retraction = np.asarray(arrays["hidden_retraction_norms"], dtype=float)
    selected.update(
        np.argsort(retraction)[-min(8, retraction.size) :].tolist()
    )
    return np.asarray(sorted(selected), dtype=int)


def _replay_run_a(
    trajectory_path: Path,
    output_directory: Path,
    sample_count: int,
) -> dict:
    with np.load(trajectory_path) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    indices = _sample_indices(arrays, sample_count)
    states = np.asarray(arrays["apcm_states"], dtype=float)
    times = np.asarray(arrays["times"], dtype=float)
    serial = _completion("full", 0)
    candidate = _completion("spin_exchange_blocks", 4)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    settings = APCMSettings(
        apply_physicality_controller=True,
        terminal_completion="positive",
    )
    serial_model = ArchiveBackedAPCM(
        parameters, settings=settings, completion=serial
    )
    candidate_model = ArchiveBackedAPCM(
        parameters, settings=settings, completion=candidate
    )
    prepared = prepare_apcm_initial_state(
        parameters,
        phonon_cutoff=16,
        completion=serial,
    )
    initialization_difference = _difference(states[0], prepared.state)
    np.savez_compressed(
        output_directory / "serial_run_a_initialization.npz",
        apcm_states=prepared.state[None, :],
        frontier=np.asarray(
            [
                prepared.frontier_moments[key]
                for key in serial.frontier_keys
            ],
            dtype=float,
        ),
    )

    complete_frontier_differences = []
    complete_matrix_differences = []
    complete_minimum_differences = []
    complete_objective_differences = []
    retraction_lower_differences = []
    retraction_frontier_differences = []
    retraction_matrix_differences = []
    retraction_minimum_differences = []
    retraction_norm_differences = []
    retraction_objective_differences = []
    controller_differences = []
    derivative_differences = []

    for ordinal, index in enumerate(indices, start=1):
        state = states[index]
        moments = _moments_from_state(state)
        serial_complete = serial.complete(moments)
        candidate_complete = candidate.complete(moments)
        if not serial_complete.success or not candidate_complete.success:
            raise RuntimeError(f"completion failed at stored index {index}")
        serial_frontier = np.asarray(
            [
                serial_complete.frontier_moments[key]
                for key in serial.frontier_keys
            ],
            dtype=float,
        )
        candidate_frontier = np.asarray(
            [
                candidate_complete.frontier_moments[key]
                for key in candidate.frontier_keys
            ],
            dtype=float,
        )
        complete_frontier_differences.append(candidate_frontier - serial_frontier)
        serial_matrix = pauli_weyl_moment_matrix(serial_complete.moments)
        candidate_matrix = pauli_weyl_moment_matrix(candidate_complete.moments)
        complete_matrix_differences.append(candidate_matrix - serial_matrix)
        complete_minimum_differences.append(
            candidate_complete.minimum_moment_matrix_eigenvalue
            - serial_complete.minimum_moment_matrix_eigenvalue
        )
        complete_objective_differences.append(
            candidate_complete.objective - serial_complete.objective
        )

        serial_retraction = serial.retract_lower_moments(
            moments,
            adjustable_keys=HIDDEN_RELATIVE_MOMENT_KEYS,
            warm_frontier=serial_complete.frontier_moments,
        )
        candidate_retraction = candidate.retract_lower_moments(
            moments,
            adjustable_keys=HIDDEN_RELATIVE_MOMENT_KEYS,
            warm_frontier=candidate_complete.frontier_moments,
        )
        if not serial_retraction.success or not candidate_retraction.success:
            raise RuntimeError(f"retraction failed at stored index {index}")
        serial_lower = np.asarray(
            [
                serial_retraction.lower_moments[key]
                for key in HIDDEN_RELATIVE_MOMENT_KEYS
            ],
            dtype=float,
        )
        candidate_lower = np.asarray(
            [
                candidate_retraction.lower_moments[key]
                for key in HIDDEN_RELATIVE_MOMENT_KEYS
            ],
            dtype=float,
        )
        serial_retracted_frontier = np.asarray(
            [
                serial_retraction.frontier_moments[key]
                for key in serial.frontier_keys
            ],
            dtype=float,
        )
        candidate_retracted_frontier = np.asarray(
            [
                candidate_retraction.frontier_moments[key]
                for key in candidate.frontier_keys
            ],
            dtype=float,
        )
        retraction_lower_differences.append(candidate_lower - serial_lower)
        retraction_frontier_differences.append(
            candidate_retracted_frontier - serial_retracted_frontier
        )
        serial_retracted_matrix = pauli_weyl_moment_matrix(
            {
                **serial_retraction.lower_moments,
                **serial_retraction.frontier_moments,
            }
        )
        candidate_retracted_matrix = pauli_weyl_moment_matrix(
            {
                **candidate_retraction.lower_moments,
                **candidate_retraction.frontier_moments,
            }
        )
        retraction_matrix_differences.append(
            candidate_retracted_matrix - serial_retracted_matrix
        )
        retraction_minimum_differences.append(
            candidate_retraction.minimum_moment_matrix_eigenvalue
            - serial_retraction.minimum_moment_matrix_eigenvalue
        )
        retraction_norm_differences.append(
            candidate_retraction.scaled_lower_correction_norm
            - serial_retraction.scaled_lower_correction_norm
        )
        retraction_objective_differences.append(
            _retraction_objective(
                moments,
                candidate_retraction.lower_moments,
                candidate_retraction.frontier_moments,
                candidate,
            )
            - _retraction_objective(
                moments,
                serial_retraction.lower_moments,
                serial_retraction.frontier_moments,
                serial,
            )
        )

        serial_evaluation = serial_model.evaluate(
            float(times[index]),
            state,
            warm_frontier=serial_complete.frontier_moments,
        )
        candidate_evaluation = candidate_model.evaluate(
            float(times[index]),
            state,
            warm_frontier=candidate_complete.frontier_moments,
        )
        if serial_evaluation.controller is None or candidate_evaluation.controller is None:
            raise RuntimeError("joint-Gram controller was unexpectedly disabled")
        controller_differences.append(
            candidate_evaluation.controller.correction_coordinates
            - serial_evaluation.controller.correction_coordinates
        )
        derivative_differences.append(
            candidate_evaluation.derivative - serial_evaluation.derivative
        )
        print(
            f"replay {ordinal}/{indices.size}: index={index} "
            f"t={times[index]:.4f}",
            flush=True,
        )

    difference_arrays = {
        "complete_frontier": np.asarray(complete_frontier_differences),
        "complete_matrix": np.asarray(complete_matrix_differences),
        "complete_minimum": np.asarray(complete_minimum_differences),
        "complete_objective": np.asarray(complete_objective_differences),
        "retraction_lower": np.asarray(retraction_lower_differences),
        "retraction_frontier": np.asarray(retraction_frontier_differences),
        "retraction_matrix": np.asarray(retraction_matrix_differences),
        "retraction_minimum": np.asarray(retraction_minimum_differences),
        "retraction_norm": np.asarray(retraction_norm_differences),
        "retraction_objective": np.asarray(retraction_objective_differences),
        "controller_correction": np.asarray(controller_differences),
        "rhs_derivative": np.asarray(derivative_differences),
    }
    np.savez_compressed(
        output_directory / "run_a_replay_differences.npz",
        sampled_indices=indices,
        sampled_times=times[indices],
        **difference_arrays,
    )
    metrics = {
        key: _difference(np.zeros_like(value), value)
        for key, value in difference_arrays.items()
    }
    passes = {
        "completion_frontier": (
            metrics["complete_frontier"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
        "retraction_lower": (
            metrics["retraction_lower"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
        "retraction_frontier": (
            metrics["retraction_frontier"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
        "m4_eigenvalue": (
            metrics["retraction_minimum"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
        "joint_gram_correction": (
            metrics["controller_correction"]["maximum_absolute_difference"]
            <= CONTROLLER_TOLERANCE
        ),
    }
    return {
        "source": str(trajectory_path.resolve()),
        "serial_initialization_state_difference": initialization_difference,
        "sampled_state_count": int(indices.size),
        "sampled_indices": indices.tolist(),
        "tolerances": {
            "solver_derived_values": VALUE_TOLERANCE,
            "joint_gram_correction": CONTROLLER_TOLERANCE,
        },
        "metrics": metrics,
        "passes": passes,
        "all_required_parity_passed": bool(all(passes.values())),
    }


def _compare_trajectories(
    serial_path: Path,
    candidate_path: Path,
    *,
    include_runtime: bool = True,
) -> dict:
    with np.load(serial_path) as serial_file:
        serial = {key: np.asarray(serial_file[key]) for key in serial_file.files}
    with np.load(candidate_path) as candidate_file:
        candidate = {
            key: np.asarray(candidate_file[key]) for key in candidate_file.files
        }
    np.testing.assert_allclose(serial["times"], candidate["times"], atol=0.0, rtol=0.0)
    parameters = DimerParameters(
        hopping=1.0,
        gamma=0.5,
        lambda_ep=1.5,
        drive_amplitude=1.0,
    )
    serial_observables = _observables(
        serial["times"], serial["approximate_archive_coordinates"], parameters
    )
    candidate_observables = _observables(
        candidate["times"],
        candidate["approximate_archive_coordinates"],
        parameters,
    )
    metrics = {
        "all_60_coordinate_states": _difference(
            serial["apcm_states"], candidate["apcm_states"]
        ),
        "all_31_archive_coordinates": _difference(
            serial["approximate_archive_coordinates"],
            candidate["approximate_archive_coordinates"],
        ),
        "extended_m4_eigenvalue_history": _difference(
            serial["completion_minimum_eigenvalues"],
            candidate["completion_minimum_eigenvalues"],
        ),
        "retained_joint_gram_eigenvalue_history": _difference(
            serial["joint_gram_minimum_eigenvalues"],
            candidate["joint_gram_minimum_eigenvalues"],
        ),
        "all_plotted_observables": _difference(
            serial_observables, candidate_observables
        ),
    }
    metrics["plotted_observables"] = {
        name: _difference(serial_observables[:, index], candidate_observables[:, index])
        for index, name in enumerate(OBSERVABLE_NAMES)
    }
    passes = {
        "all_60_coordinate_states": (
            metrics["all_60_coordinate_states"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
        "all_plotted_observables": (
            metrics["all_plotted_observables"]["maximum_absolute_difference"]
            <= OBSERVABLE_TOLERANCE
        ),
        "extended_m4_eigenvalue_history": (
            metrics["extended_m4_eigenvalue_history"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
        "retained_joint_gram_eigenvalue_history": (
            metrics["retained_joint_gram_eigenvalue_history"]["maximum_absolute_difference"]
            <= VALUE_TOLERANCE
        ),
    }
    result = {
        "serial_source": str(serial_path.resolve()),
        "candidate_source": str(candidate_path.resolve()),
        "metrics": metrics,
        "tolerances": {
            "state_and_cone": VALUE_TOLERANCE,
            "observables": OBSERVABLE_TOLERANCE,
        },
        "passes": passes,
        "all_required_parity_passed": bool(all(passes.values())),
    }
    if include_runtime:
        serial_summary = json.loads(
            (serial_path.parent / "summary.json").read_text(encoding="utf-8")
        )
        candidate_summary = json.loads(
            (candidate_path.parent / "summary.json").read_text(encoding="utf-8")
        )
        serial_runtime = float(
            serial_summary["integration"]["autonomous_wall_seconds"]
        )
        candidate_runtime = float(
            candidate_summary["integration"]["autonomous_wall_seconds"]
        )
        result["runtime"] = {
            "serial_seconds": serial_runtime,
            "candidate_seconds": candidate_runtime,
            "candidate_speedup": serial_runtime / candidate_runtime,
        }
    return result


def _compare_full_continuation(
    reference_path: Path,
    continuation_path: Path,
    *,
    start_index: int,
    serial_prefix_summary_path: Path,
    output_directory: Path,
) -> dict:
    with np.load(reference_path) as reference_file:
        reference = {
            key: np.asarray(reference_file[key]) for key in reference_file.files
        }
    with np.load(continuation_path) as continuation_file:
        continuation = {
            key: np.asarray(continuation_file[key])
            for key in continuation_file.files
        }
    if start_index <= 0 or start_index >= reference["times"].size:
        raise ValueError("continuation_start_index is out of range")
    np.testing.assert_allclose(
        reference["times"][start_index:],
        continuation["times"],
        atol=1.0e-14,
        rtol=0.0,
    )
    keys = (
        "apcm_states",
        "approximate_archive_coordinates",
        "exact_archive_coordinates",
        "completion_minimum_eigenvalues",
        "joint_gram_minimum_eigenvalues",
        "correction_norms",
        "hidden_retraction_norms",
    )
    composite = {"times": reference["times"].copy()}
    for key in keys:
        composite[key] = np.concatenate(
            [reference[key][:start_index], continuation[key]],
            axis=0,
        )
    composite_path = output_directory / "full_composite_trajectory.npz"
    np.savez_compressed(composite_path, **composite)
    comparison = _compare_trajectories(
        reference_path,
        composite_path,
        include_runtime=False,
    )
    reference_summary = json.loads(
        (reference_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    continuation_summary = json.loads(
        (continuation_path.parent / "summary.json").read_text(encoding="utf-8")
    )
    prefix_summary = json.loads(
        serial_prefix_summary_path.read_text(encoding="utf-8")
    )
    serial_full = float(
        reference_summary["integration"]["autonomous_wall_seconds"]
    )
    serial_prefix = float(
        prefix_summary["integration"]["autonomous_wall_seconds"]
    )
    candidate_suffix = float(
        continuation_summary["integration"]["autonomous_wall_seconds"]
    )
    candidate_composite = serial_prefix + candidate_suffix
    comparison["runtime"] = {
        "serial_full_seconds": serial_full,
        "serial_prefix_seconds": serial_prefix,
        "inferred_serial_suffix_seconds": serial_full - serial_prefix,
        "candidate_suffix_seconds": candidate_suffix,
        "candidate_composite_seconds": candidate_composite,
        "candidate_composite_speedup": serial_full / candidate_composite,
        "candidate_suffix_speedup": (
            (serial_full - serial_prefix) / candidate_suffix
        ),
    }
    comparison["continuation_handoff"] = {
        "time": float(reference["times"][start_index]),
        "state_difference": _difference(
            reference["apcm_states"][start_index],
            continuation["apcm_states"][0],
        ),
    }
    return comparison


def main() -> int:
    args = _parser().parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    if args.full_only:
        required = (
            args.full_reference_trajectory,
            args.candidate_continuation_trajectory,
            args.serial_prefix_summary,
        )
        if any(value is None for value in required):
            raise SystemExit(
                "--full-only requires the full reference, candidate "
                "continuation, and serial prefix summary"
            )
        full = _compare_full_continuation(
            args.full_reference_trajectory.resolve(),
            args.candidate_continuation_trajectory.resolve(),
            start_index=args.continuation_start_index,
            serial_prefix_summary_path=args.serial_prefix_summary.resolve(),
            output_directory=output_directory,
        )
        summary = {
            "schema_version": 1,
            "candidate": (
                "exact spin-exchange M4 blocks with four-thread faer solve"
            ),
            "full_run_a_parity": full,
        }
        (output_directory / "full_parity_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
        return 0
    required = (
        args.run_a_trajectory,
        args.serial_trajectory,
        args.candidate_trajectory,
    )
    if any(value is None for value in required):
        raise SystemExit(
            "replay mode requires Run A and both short trajectories"
        )
    replay = _replay_run_a(
        args.run_a_trajectory.resolve(),
        output_directory,
        args.sample_count,
    )
    short = _compare_trajectories(
        args.serial_trajectory.resolve(),
        args.candidate_trajectory.resolve(),
    )
    summary = {
        "schema_version": 1,
        "candidate": "exact spin-exchange M4 blocks with four-thread faer solve",
        "run_a_state_replay": replay,
        "short_trajectory": short,
        "candidate_can_replace_serial": bool(
            replay["all_required_parity_passed"]
            and short["all_required_parity_passed"]
        ),
    }
    (output_directory / "parity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
