"""Run and score one exploratory adaptive positive-moment rollout."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive_positive_moment_propagation import (
    APCMSettings,
    ArchiveBackedAPCM,
    integrate_apcm_ssprk3,
    prepare_apcm_initial_state,
)
from .exact_reference import exact_holstein_driven_trajectory
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
)
from .positive_moment_completion import (
    PositiveFourthMomentCompletion,
    PositiveMomentCompletionSettings,
)


_BLOCK_FIELDS = {
    "electron_density": "electron_density",
    "coherent_phonon": "coherent_phonon",
    "normal_phonon": "phonon_density",
    "anomalous_phonon": "anomalous_phonon_density",
    "electron_phonon_correlation": "electron_phonon_correlation",
}


def _matrix_states(coordinates: np.ndarray) -> tuple:
    return tuple(closed_scalar_to_matrix_state(row) for row in coordinates)


def _block_metrics(
    exact_coordinates: np.ndarray,
    approximate_coordinates: np.ndarray,
) -> dict[str, dict[str, float]]:
    exact_states = _matrix_states(exact_coordinates)
    approximate_states = _matrix_states(approximate_coordinates)
    result: dict[str, dict[str, float]] = {}
    for label, field in _BLOCK_FIELDS.items():
        errors = np.asarray(
            [
                np.linalg.norm(
                    getattr(approximate, field) - getattr(exact, field)
                )
                for exact, approximate in zip(
                    exact_states,
                    approximate_states,
                    strict=True,
                )
            ],
            dtype=float,
        )
        result[label] = {
            "time_rms_frobenius_error": float(
                np.sqrt(np.mean(errors**2))
            ),
            "maximum_frobenius_error": float(np.max(errors)),
            "final_frobenius_error": float(errors[-1]),
        }
    return result


def _trajectory_metrics(
    parameters: DimerParameters,
    exact_coordinates: np.ndarray,
    approximate_coordinates: np.ndarray,
) -> dict[str, Any]:
    difference = approximate_coordinates - exact_coordinates
    exact_states = _matrix_states(exact_coordinates)
    approximate_states = _matrix_states(approximate_coordinates)
    exact_energy = np.asarray(
        [matrix_total_energy(state, parameters) for state in exact_states],
        dtype=float,
    )
    approximate_energy = np.asarray(
        [
            matrix_total_energy(state, parameters)
            for state in approximate_states
        ],
        dtype=float,
    )
    occupation_error = np.asarray(
        [
            approximate.electron_density[0, 0].real
            - exact.electron_density[0, 0].real
            for exact, approximate in zip(
                exact_states,
                approximate_states,
                strict=True,
            )
        ],
        dtype=float,
    )
    return {
        "all_coordinate_scalar_rms_error": float(
            np.sqrt(np.mean(difference**2))
        ),
        "time_rms_coordinate_l2_error": float(
            np.sqrt(np.mean(np.sum(difference**2, axis=1)))
        ),
        "maximum_absolute_coordinate_error": float(
            np.max(np.abs(difference))
        ),
        "final_coordinate_l2_error": float(
            np.linalg.norm(difference[-1])
        ),
        "site_0_occupation_rms_error": float(
            np.sqrt(np.mean(occupation_error**2))
        ),
        "static_energy_rms_error": float(
            np.sqrt(np.mean((approximate_energy - exact_energy) ** 2))
        ),
        "maximum_static_energy_error": float(
            np.max(np.abs(approximate_energy - exact_energy))
        ),
        "block_errors": _block_metrics(
            exact_coordinates,
            approximate_coordinates,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initial-time", type=float, default=0.0)
    parser.add_argument("--final-time", type=float, default=0.25)
    parser.add_argument("--time-step", type=float, default=0.0025)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument(
        "--terminal-completion",
        choices=("positive", "zero_cumulant_prior"),
        default="positive",
    )
    parser.add_argument(
        "--no-controller",
        action="store_true",
        help="Disable the retained joint-Gram physicality controller.",
    )
    parser.add_argument(
        "--initial-state-file",
        type=Path,
        help="Use apcm_states[0] from an existing trajectory NPZ.",
    )
    parser.add_argument(
        "--initial-state-index",
        type=int,
        default=0,
        help="Select one apcm_states row from --initial-state-file.",
    )
    parser.add_argument(
        "--initial-frontier-file",
        type=Path,
        help="Use the frontier array from a matched initialization NPZ.",
    )
    parser.add_argument(
        "--completion-cone-representation",
        choices=(
            "full",
            "spin_exchange_blocks",
        ),
        default="full",
        help="Select the serial full cone or the exact symmetry-block candidate.",
    )
    parser.add_argument(
        "--completion-max-threads",
        type=int,
        default=0,
        help="Bound Clarabel's faer threads; zero preserves the serial backend.",
    )
    return parser


def _first_time(times: np.ndarray, condition: np.ndarray) -> float | None:
    indices = np.flatnonzero(condition)
    return None if indices.size == 0 else float(times[int(indices[0])])


def main() -> int:
    args = _parser().parse_args()
    if (
        args.initial_frontier_file is not None
        and args.initial_state_file is None
    ):
        raise SystemExit("--initial-frontier-file requires --initial-state-file")
    if args.initial_time > 0.0 and args.initial_state_file is None:
        raise SystemExit("positive --initial-time requires --initial-state-file")
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    progress_path = output_directory / "progress.jsonl"
    checkpoint_path = output_directory / "checkpoint.npz"

    def progress(message: str) -> None:
        print(message, flush=True)
        with progress_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"message": message}) + "\n")

    parameters = DimerParameters(
        hopping=1.0,
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
    )
    started = time.perf_counter()
    completion = PositiveFourthMomentCompletion(
        PositiveMomentCompletionSettings(
            phonon_envelope=16.0,
            logdet_weight=1.0,
            logdet_shift=1e-5,
            solver_tolerance=1e-7,
            maximum_iterations=2_000,
            cone_representation=args.completion_cone_representation,
            clarabel_max_threads=args.completion_max_threads,
        )
    )
    model = ArchiveBackedAPCM(
        parameters,
        settings=APCMSettings(
            apply_physicality_controller=not args.no_controller,
            terminal_completion=args.terminal_completion,
        ),
        completion=completion,
    )
    if args.initial_state_file is None:
        initial = prepare_apcm_initial_state(
            parameters,
            phonon_cutoff=args.phonon_cutoff,
            completion=model.completion,
        )
        initial_state = initial.state
        initial_frontier = initial.frontier_moments
        initialization_summary = {
            "source": "single exact cutoff ground-state contraction",
            "hidden_retraction_scaled_norm": (
                initial.hidden_retraction.scaled_lower_correction_norm
            ),
            "extended_moment_minimum_eigenvalue": (
                initial.hidden_retraction.minimum_moment_matrix_eigenvalue
            ),
        }
    else:
        initial_arrays = np.load(args.initial_state_file.resolve())
        initial_state = np.asarray(
            initial_arrays["apcm_states"][args.initial_state_index],
            dtype=float,
        )
        if args.initial_frontier_file is None:
            initial_frontier = None
        else:
            frontier_arrays = np.load(args.initial_frontier_file.resolve())
            frontier_values = np.asarray(
                frontier_arrays["frontier"], dtype=float
            ).reshape(-1)
            if frontier_values.shape != (len(model.completion.frontier_keys),):
                raise ValueError("initial frontier has the wrong dimension")
            initial_frontier = {
                key: float(value)
                for key, value in zip(
                    model.completion.frontier_keys,
                    frontier_values,
                    strict=True,
                )
            }
        initialization_summary = {
            "source": str(args.initial_state_file.resolve()),
            "frontier_source": (
                None
                if args.initial_frontier_file is None
                else str(args.initial_frontier_file.resolve())
            ),
            "hidden_retraction_scaled_norm": None,
            "extended_moment_minimum_eigenvalue": None,
        }
    interval_count = int(
        round((args.final_time - args.initial_time) / args.time_step)
    )
    checkpoint_stride = max(1, interval_count // 20)

    def checkpoint(
        step: int,
        current_time: float,
        state: np.ndarray,
        frontier: dict,
    ) -> None:
        if step % checkpoint_stride != 0 and step != interval_count:
            return
        temporary = checkpoint_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                step=np.asarray(step),
                time=np.asarray(current_time),
                state=state,
                frontier=np.asarray(
                    [frontier[key] for key in model.completion.frontier_keys],
                    dtype=float,
                ),
            )
        os.replace(temporary, checkpoint_path)

    trajectory = integrate_apcm_ssprk3(
        model,
        initial_state,
        initial_time=args.initial_time,
        final_time=args.final_time,
        time_step=args.time_step,
        initial_frontier=initial_frontier,
        progress=progress,
        checkpoint=checkpoint,
    )
    autonomous_elapsed = time.perf_counter() - started

    # Freeze the autonomous result before constructing the offline reference.
    approximate_coordinates = trajectory.archive_coordinates
    np.savez_compressed(
        output_directory / "autonomous_trajectory.npz",
        times=trajectory.times,
        apcm_states=trajectory.states,
        approximate_archive_coordinates=approximate_coordinates,
        completion_minimum_eigenvalues=(
            trajectory.completion_minimum_eigenvalues
        ),
        joint_gram_minimum_eigenvalues=(
            trajectory.joint_gram_minimum_eigenvalues
        ),
        correction_norms=trajectory.correction_norms,
        hidden_retraction_norms=trajectory.hidden_retraction_norms,
    )
    exact_started = time.perf_counter()
    if np.isclose(trajectory.times[0], 0.0):
        exact_sample_times = trajectory.times
        exact_offset = 0
    else:
        exact_sample_times = np.concatenate(
            [np.asarray([0.0]), trajectory.times]
        )
        exact_offset = 1
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=exact_sample_times,
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=min(0.05, args.time_step * 4.0),
    )
    exact_coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states[exact_offset:]
        ],
        dtype=float,
    )
    exact_elapsed = time.perf_counter() - exact_started
    metrics = _trajectory_metrics(
        parameters,
        exact_coordinates,
        approximate_coordinates,
    )
    exact_joint_minimum = float(
        min(
            np.linalg.eigvalsh(electron_phonon_moment_matrix(state))[0]
            for state in exact.matrix_states[exact_offset:]
        )
    )
    summary = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "model": "archive_backed_apcm_entrance_layer_prototype",
        "state_dimension": int(initial_state.size),
        "ablation": {
            "terminal_completion": args.terminal_completion,
            "retained_joint_gram_controller": not args.no_controller,
            "completion_cone_representation": (
                args.completion_cone_representation
            ),
            "completion_max_threads": args.completion_max_threads,
        },
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "coupling": parameters.coupling,
            "drive_amplitude": parameters.drive_amplitude,
            "phonon_cutoff": args.phonon_cutoff,
        },
        "integration": {
            "method": "SSPRK(3,3) with hidden-moment stage retraction",
            "final_time": args.final_time,
            "initial_time": args.initial_time,
            "time_step": args.time_step,
            "completed_steps": trajectory.completed_steps,
            "rhs_evaluations": trajectory.rhs_evaluations,
            "autonomous_wall_seconds": autonomous_elapsed,
            "exact_reference_wall_seconds": exact_elapsed,
        },
        "initialization": initialization_summary,
        "feasibility": {
            "minimum_retained_joint_gram_eigenvalue": float(
                np.min(trajectory.joint_gram_minimum_eigenvalues)
            ),
            "minimum_extended_moment_eigenvalue": float(
                np.nanmin(trajectory.completion_minimum_eigenvalues)
            ),
            "exact_minimum_joint_gram_eigenvalue": exact_joint_minimum,
            "maximum_controller_frobenius_norm": float(
                np.max(trajectory.correction_norms)
            ),
            "maximum_hidden_retraction_scaled_norm": float(
                np.max(trajectory.hidden_retraction_norms)
            ),
            "steps_with_hidden_retraction": int(
                np.count_nonzero(
                    trajectory.hidden_retraction_norms > 1e-12
                )
            ),
            "first_negative_retained_joint_gram_time": _first_time(
                trajectory.times,
                trajectory.joint_gram_minimum_eigenvalues < 0.0,
            ),
            "first_negative_extended_moment_time": _first_time(
                trajectory.times,
                trajectory.completion_minimum_eigenvalues < 0.0,
            ),
            "first_extended_violation_beyond_tolerance_time": _first_time(
                trajectory.times,
                trajectory.completion_minimum_eigenvalues < -1e-6,
            ),
            "first_amplitude_threshold_time": _first_time(
                trajectory.times,
                np.max(np.abs(trajectory.states), axis=1) >= 1e4,
            ),
        },
        "exact_reference": {
            "type": "cutoff Hamiltonian DOP853, offline only",
            "function_evaluations": exact.function_evaluations,
            "maximum_state_norm_defect": float(
                np.max(np.abs(exact.state_norms - 1.0))
            ),
        },
        "accuracy": metrics,
    }
    np.savez_compressed(
        output_directory / "trajectory.npz",
        times=trajectory.times,
        apcm_states=trajectory.states,
        approximate_archive_coordinates=approximate_coordinates,
        exact_archive_coordinates=exact_coordinates,
        completion_minimum_eigenvalues=(
            trajectory.completion_minimum_eigenvalues
        ),
        joint_gram_minimum_eigenvalues=(
            trajectory.joint_gram_minimum_eigenvalues
        ),
        correction_norms=trajectory.correction_norms,
        hidden_retraction_norms=trajectory.hidden_retraction_norms,
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
