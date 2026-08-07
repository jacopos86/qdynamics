"""Run and score the carried-witness radial moment-flow pilot."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive_positive_moment import raw_moment_coordinates_to_matrix_state
from .apcm_carried_witness import (
    CWRMFSettings,
    CarriedWitnessModel,
    integrate_cwrmf_ssprk2,
)
from .exact_reference import exact_holstein_driven_trajectory
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    closed_scalar_to_matrix_state,
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
    matrix_total_energy,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--initial-time", type=float)
    parser.add_argument("--initial-state-file", type=Path)
    parser.add_argument("--initial-state-index", type=int, default=-1)
    parser.add_argument("--final-time", type=float, default=0.05)
    parser.add_argument("--time-step", type=float, default=0.0025)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument(
        "--maximum-critical-modes",
        type=int,
        default=None,
        help=(
            "Optional reproducibility cap on the critical Schur dimension. "
            "By default the dimension is selected from the Gram spectrum "
            "without an artificial cap."
        ),
    )
    parser.add_argument("--compact-output", action="store_true")
    return parser


def _archive_coordinates(model: CarriedWitnessModel, states: np.ndarray) -> np.ndarray:
    coordinates: list[np.ndarray] = []
    for state in states:
        retained, _ = model.geometry.unpack_state(state)
        raw, _ = model.geometry.split_retained(retained)
        matrix_state = raw_moment_coordinates_to_matrix_state(raw)
        coordinates.append(matrix_state_to_closed_scalar_coordinates(matrix_state))
    return np.asarray(coordinates, dtype=float)


def _accuracy_metrics(
    parameters: DimerParameters,
    exact_coordinates: np.ndarray,
    approximate_coordinates: np.ndarray,
) -> dict[str, Any]:
    difference = approximate_coordinates - exact_coordinates
    initial_offset = difference[0].copy()
    dynamic_difference = difference - initial_offset
    block_slices = {
        "rho": slice(0, 3),
        "B": slice(3, 7),
        "N": slice(7, 11),
        "A": slice(11, 17),
        "C": slice(17, 31),
    }
    exact_matrix_states = tuple(
        closed_scalar_to_matrix_state(row) for row in exact_coordinates
    )
    approximate_matrix_states = tuple(
        closed_scalar_to_matrix_state(row) for row in approximate_coordinates
    )
    occupation_error = np.asarray(
        [
            approximate.electron_density[0, 0].real
            - exact.electron_density[0, 0].real
            for exact, approximate in zip(
                exact_matrix_states,
                approximate_matrix_states,
                strict=True,
            )
        ],
        dtype=float,
    )
    dynamic_occupation_error = occupation_error - occupation_error[0]
    exact_energy = np.asarray(
        [matrix_total_energy(state, parameters) for state in exact_matrix_states],
        dtype=float,
    )
    approximate_energy = np.asarray(
        [
            matrix_total_energy(state, parameters)
            for state in approximate_matrix_states
        ],
        dtype=float,
    )
    energy_error = approximate_energy - exact_energy
    dynamic_energy_error = energy_error - energy_error[0]
    blockwise = {}
    for name, block in block_slices.items():
        blockwise[name] = {
            "scalar_rms_error": float(
                np.sqrt(np.mean(difference[:, block] ** 2))
            ),
            "dynamic_scalar_rms_error": float(
                np.sqrt(np.mean(dynamic_difference[:, block] ** 2))
            ),
            "final_l2_error": float(np.linalg.norm(difference[-1, block])),
            "dynamic_final_l2_error": float(
                np.linalg.norm(dynamic_difference[-1, block])
            ),
        }
    return {
        "initial_coordinate_offset_l2": float(np.linalg.norm(initial_offset)),
        "initial_coordinate_offset_maximum_absolute": float(
            np.max(np.abs(initial_offset))
        ),
        "all_coordinate_scalar_rms_error": float(
            np.sqrt(np.mean(difference**2))
        ),
        "dynamic_all_coordinate_scalar_rms_error": float(
            np.sqrt(np.mean(dynamic_difference**2))
        ),
        "time_rms_coordinate_l2_error": float(
            np.sqrt(np.mean(np.sum(difference**2, axis=1)))
        ),
        "dynamic_time_rms_coordinate_l2_error": float(
            np.sqrt(np.mean(np.sum(dynamic_difference**2, axis=1)))
        ),
        "maximum_absolute_coordinate_error": float(np.max(np.abs(difference))),
        "final_coordinate_l2_error": float(np.linalg.norm(difference[-1])),
        "site_0_occupation_rms_error": float(
            np.sqrt(np.mean(occupation_error**2))
        ),
        "dynamic_site_0_occupation_rms_error": float(
            np.sqrt(np.mean(dynamic_occupation_error**2))
        ),
        "static_energy_rms_error": float(
            np.sqrt(np.mean(energy_error**2))
        ),
        "dynamic_static_energy_rms_error": float(
            np.sqrt(np.mean(dynamic_energy_error**2))
        ),
        "maximum_static_energy_error": float(
            np.max(np.abs(energy_error))
        ),
        "blockwise": blockwise,
    }


def main() -> int:
    args = _parser().parse_args()
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)
    progress_path = output_directory / "progress.jsonl"

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
    settings = CWRMFSettings(
        maximum_critical_modes=args.maximum_critical_modes
    )
    started = time.perf_counter()
    model = CarriedWitnessModel(parameters, settings=settings)
    preparation = model.prepare(phonon_cutoff=args.phonon_cutoff)
    initial_state = preparation.state
    initial_time = 0.0 if args.initial_time is None else args.initial_time
    initialization_source = "single exact cutoff ground-state contraction"
    if args.initial_state_file is not None:
        initial_arrays = np.load(args.initial_state_file.resolve())
        if "state" in initial_arrays:
            initial_state = np.asarray(initial_arrays["state"], dtype=float)
            stored_time = float(np.asarray(initial_arrays["time"]))
            if args.initial_time is None:
                initial_time = stored_time
            elif not np.isclose(initial_time, stored_time, atol=1e-12):
                raise ValueError("initial time disagrees with checkpoint")
        elif "carried_states" in initial_arrays:
            initial_state = np.asarray(
                initial_arrays["carried_states"][args.initial_state_index],
                dtype=float,
            )
            stored_time = float(
                initial_arrays["times"][args.initial_state_index]
            )
            if args.initial_time is None:
                initial_time = stored_time
            elif not np.isclose(initial_time, stored_time, atol=1e-12):
                raise ValueError("initial time disagrees with trajectory row")
        else:
            raise ValueError("initial state file has no carried state")
        if "bundle_rank_hint" in initial_arrays:
            model.restore_bundle_rank_hint(
                int(np.asarray(initial_arrays["bundle_rank_hint"]))
            )
        model.geometry.unpack_state(initial_state)
        initialization_source = str(args.initial_state_file.resolve())
    progress(
        "prepared "
        f"state_dim={model.geometry.state_count} "
        f"completion_dim={model.geometry.completion_count} "
        f"readable_rates={model.geometry.readable_completion_indices.size} "
        f"lambda_shifted_lo={preparation.minimum_shifted_lower_bound:.3e}"
    )
    checkpoint_path = output_directory / "checkpoint.npz"

    def checkpoint(step: int, current_time: float, state: np.ndarray) -> None:
        temporary = checkpoint_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                step=np.asarray(step),
                time=np.asarray(current_time),
                state=state,
                bundle_rank_hint=np.asarray(model.bundle_rank_hint),
            )
        os.replace(temporary, checkpoint_path)

    trajectory = integrate_cwrmf_ssprk2(
        model,
        initial_state,
        initial_time=initial_time,
        final_time=args.final_time,
        time_step=args.time_step,
        progress=progress,
        checkpoint=checkpoint,
    )
    autonomous_elapsed = time.perf_counter() - started
    approximate_coordinates = _archive_coordinates(model, trajectory.states)

    # Freeze the autonomous output before constructing the offline reference.
    np.savez_compressed(
        output_directory / "autonomous_trajectory.npz",
        times=trajectory.times,
        carried_states=trajectory.states,
        approximate_archive_coordinates=approximate_coordinates,
        minimum_unshifted_eigenvalues=(
            trajectory.minimum_unshifted_eigenvalues
        ),
        minimum_shifted_lower_bounds=trajectory.minimum_shifted_lower_bounds,
        maximum_atom_seconds=trajectory.maximum_atom_seconds,
        correction_iterations=trajectory.correction_iterations,
        readable_rate_residuals=trajectory.readable_rate_residuals,
        completion_correction_norms=trajectory.completion_correction_norms,
        velocity_margins=trajectory.velocity_margins,
        critical_modes=trajectory.critical_modes,
    )

    exact_started = time.perf_counter()
    if np.isclose(trajectory.times[0], 0.0):
        exact_sample_times = trajectory.times
        exact_offset = 0
    else:
        exact_sample_times = np.concatenate(
            (np.asarray([0.0]), trajectory.times)
        )
        exact_offset = 1
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=exact_sample_times,
        phonon_cutoff=args.phonon_cutoff,
        maximum_step=min(0.01, args.time_step * 4.0),
    )
    exact_coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states[exact_offset:]
        ],
        dtype=float,
    )
    exact_elapsed = time.perf_counter() - exact_started
    metrics = _accuracy_metrics(
        parameters, exact_coordinates, approximate_coordinates
    )
    approximate_joint_minimum = float(
        min(
            np.linalg.eigvalsh(electron_phonon_moment_matrix(
                raw_moment_coordinates_to_matrix_state(
                    model.geometry.split_retained(
                        model.geometry.unpack_state(state)[0]
                    )[0]
                )
            ))[0]
            for state in trajectory.states
        )
    )
    summary = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "model": "carried_witness_radial_moment_flow",
        "strict_eight_mode_contract": args.maximum_critical_modes == 8,
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "coupling": parameters.coupling,
            "drive_amplitude": parameters.drive_amplitude,
            "phonon_cutoff": args.phonon_cutoff,
        },
        "state": {
            "retained_dimension": model.geometry.retained_count,
            "completion_dimension": model.geometry.completion_count,
            "total_dimension": model.geometry.state_count,
            "readable_completion_rates": int(
                model.geometry.readable_completion_indices.size
            ),
            "literal_gram_dimension": 62,
            "hierarchy_degree": preparation.hierarchy_degree,
        },
        "integration": {
            "method": "SSPRK2 with finite-step carried-witness radial atoms",
            "initial_time": initial_time,
            "final_time_requested": args.final_time,
            "last_time": float(trajectory.times[-1]),
            "time_step": args.time_step,
            "completed_steps": trajectory.completed_steps,
            "atom_evaluations": trajectory.atom_evaluations,
            "success": trajectory.success,
            "message": trajectory.message,
            "maximum_critical_modes": args.maximum_critical_modes,
            "critical_mode_selection": (
                "adaptive_from_gram_spectrum"
                if args.maximum_critical_modes is None
                else "spectrum_selected_with_explicit_cap"
            ),
            "maximum_active_critical_modes": int(
                np.max(trajectory.critical_modes)
            ),
            "mean_active_critical_modes": float(
                np.mean(trajectory.critical_modes)
            ),
            "autonomous_wall_seconds": autonomous_elapsed,
            "exact_reference_wall_seconds": exact_elapsed,
            "peak_resident_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        },
        "preparation": {
            "source": initialization_source,
            "ground_energy": preparation.ground_energy,
            "minimum_unshifted_eigenvalue": (
                preparation.minimum_unshifted_eigenvalue
            ),
            "minimum_shifted_lower_bound": (
                preparation.minimum_shifted_lower_bound
            ),
            "restriction_residual": preparation.restriction_residual,
            "factorization_residual": preparation.factorization_residual,
        },
        "feasibility": {
            "minimum_carried_unshifted_eigenvalue": float(
                np.min(trajectory.minimum_unshifted_eigenvalues)
            ),
            "minimum_carried_shifted_lower_bound": float(
                np.min(trajectory.minimum_shifted_lower_bounds)
            ),
            "minimum_retained_joint_gram_eigenvalue": (
                approximate_joint_minimum
            ),
            "maximum_readable_rate_residual": float(
                np.max(trajectory.readable_rate_residuals)
            ),
            "maximum_completion_correction_norm": float(
                np.max(trajectory.completion_correction_norms)
            ),
            "minimum_velocity_margin": float(
                np.min(trajectory.velocity_margins)
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
        carried_states=trajectory.states,
        approximate_archive_coordinates=approximate_coordinates,
        exact_archive_coordinates=exact_coordinates,
        minimum_unshifted_eigenvalues=(
            trajectory.minimum_unshifted_eigenvalues
        ),
        minimum_shifted_lower_bounds=trajectory.minimum_shifted_lower_bounds,
        completion_correction_norms=trajectory.completion_correction_norms,
        critical_modes=trajectory.critical_modes,
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.compact_output:
        print(
            json.dumps(
                {
                    "success": trajectory.success,
                    "last_time": float(trajectory.times[-1]),
                    "message": trajectory.message,
                    "minimum_shifted_lower_bound": float(
                        np.min(trajectory.minimum_shifted_lower_bounds)
                    ),
                    "maximum_readable_rate_residual": float(
                        np.max(trajectory.readable_rate_residuals)
                    ),
                    "autonomous_wall_seconds": autonomous_elapsed,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if trajectory.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
