"""Run and score the carried-witness radial moment-flow pilot."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from .adaptive_positive_moment import raw_moment_coordinates_to_matrix_state
from .apcm_carried_witness import (
    CWRMF_BASIS_CARRYING_SEED_POLICIES,
    CWRMF_BUNDLE_SEED_POLICIES,
    CWRMF_SOLVER_SETTING_PROFILES,
    DEFAULT_BUNDLE_SEED_POLICY,
    DEFAULT_SOLVER_SETTING_PROFILE,
    CarriedWitnessModel,
    cwrmf_settings_for_profile,
    integrate_cwrmf_ssprk2,
)
from .cwrmf_solver_instances import ConicInstanceRecorder
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
        "--numerical-profile",
        choices=("strict", "balanced"),
        default="strict",
        help=(
            "strict preserves the reference contract; balanced permits an "
            "explicit -1e-8 unshifted Gram floor and relaxed conic solves."
        ),
    )
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
    guard_group = parser.add_mutually_exclusive_group()
    guard_group.add_argument(
        "--no-gram-guard",
        action="store_true",
        help=(
            "Propagate the carried higher-moment rates directly, without "
            "the 62-row PSD solve or endpoint rejection."
        ),
    )
    guard_group.add_argument(
        "--event-triggered-guard",
        action="store_true",
        help=(
            "Propagate the unconstrained carried rates while the endpoint "
            "62-row Gram stays above the declared event floor, evaluate the "
            "online negative-mode coupling measure below it, and run the "
            "declared retraction or guarded conic correction only on "
            "triggered atoms."
        ),
    )
    parser.add_argument(
        "--event-gram-hard-floor",
        type=float,
        default=None,
        help=(
            "Optional override of the event policy's hard trigger floor. "
            "Repairs fire unconditionally once the endpoint Gram minimum "
            "falls below its negation, bounding how deep the conic repair "
            "problem can become."
        ),
    )
    parser.add_argument(
        "--maximum-step-subdivisions",
        type=int,
        default=0,
        help=(
            "Declared bound on local step halving.  At the default zero a "
            "failed radial atom or an unresolved convex node ends the "
            "trajectory.  At depth d the failed interval is retried from the "
            "last accepted state as 2**k steps of time_step / 2**k for "
            "k = 1 .. d; the reported time grid and the guard policy are "
            "unchanged."
        ),
    )
    parser.add_argument(
        "--batched-mode-gradients",
        action="store_true",
        help=(
            "Named experiment, off by default: form the eigenvalue gradients "
            "of all working modes in one contraction instead of one per "
            "mode, in the completion retraction and the negative-mode "
            "coupling measure.  Different BLAS blocking, so it agrees with "
            "the default to roundoff only."
        ),
    )
    parser.add_argument(
        "--tier-timing",
        action="store_true",
        help=(
            "Accumulate wall time below the radial atom by named tier (the "
            "62-row eigendecomposition, coefficient assembly and svec "
            "packing, the Clarabel solve, the completion retraction, and the "
            "unguarded velocity evaluation) and report it next to "
            "maximum_atom_seconds.  Off by default and free when off."
        ),
    )
    parser.add_argument(
        "--solver-setting-profile",
        choices=tuple(CWRMF_SOLVER_SETTING_PROFILES),
        default=DEFAULT_SOLVER_SETTING_PROFILE,
        help=(
            "Named Clarabel setting set applied to the four guarded conic "
            "solves.  The default applies no override, so the guarded path "
            "is the pinned single-threaded contract the parity anchors were "
            "measured under; every other entry is a declared experiment "
            "measured on the recorded instances by cwrmf_solver_replay."
        ),
    )
    parser.add_argument(
        "--record-conic-instances",
        action="store_true",
        help=(
            "Named experiment, off by default: save every conic problem the "
            "guard hands Clarabel -- the objective matrix and vector, the "
            "cone matrix and right-hand side, the cone specification, the "
            "solver settings, the tier name and the radial atom index -- "
            "with the solution Clarabel returned, under "
            "conic_instances/ in the run directory.  The rollout is "
            "unchanged; the recorder only reads finished operands."
        ),
    )
    parser.add_argument(
        "--bundle-seed-policy",
        choices=tuple(CWRMF_BUNDLE_SEED_POLICIES),
        default=DEFAULT_BUNDLE_SEED_POLICY,
        help=(
            "Named starting-working-set rule for the adaptive spectral "
            "bundle tier.  The default monotone_hint is the recorded rule "
            "(the larger of the spectral entry count and the monotone rank "
            "hint, which starts at zero).  The others are declared "
            "experiments aimed at the enlargement iterations whose solution "
            "is discarded."
        ),
    )
    parser.add_argument(
        "--bundle-seed-spectral-threshold",
        type=float,
        default=None,
        help=(
            "Declared threshold for the spectral_threshold seed policy: the "
            "working set starts at the number of endpoint unshifted 62-row "
            "Gram eigenvalues below it.  Unused by every other policy."
        ),
    )
    parser.add_argument(
        "--bundle-seed-rank-margin",
        type=int,
        default=None,
        help=(
            "Declared margin added to the previous accepted bundle rank by "
            "the previous_accepted_rank_margin seed policy.  Unused by "
            "every other policy."
        ),
    )
    parser.add_argument(
        "--bundle-growth-factor",
        type=float,
        default=None,
        help=(
            "Named experiment, default 1.0 which is the recorded rule: when "
            "the full-Gram acceptance test fails with a normalized residual "
            "above --bundle-growth-trigger, pad the enlarged working set "
            "with further predictor eigenvectors until its rank reaches this "
            "factor times the current rank, capped at the critical mode "
            "limit."
        ),
    )
    parser.add_argument(
        "--bundle-growth-trigger",
        type=float,
        default=None,
        help=(
            "Normalized residual above which --bundle-growth-factor applies. "
            "The default 0.0 makes every rejection a bad one."
        ),
    )
    parser.add_argument(
        "--bundle-interior-overshoot",
        type=float,
        default=None,
        help=(
            "Named experiment, default 1.0 which is the recorded rule: "
            "multiply the compressed problem's own interior margin by this "
            "factor, so a solve aims further inside the cone than the "
            "acceptance test demands.  The acceptance test itself is "
            "unchanged."
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
    settings = cwrmf_settings_for_profile(
        args.numerical_profile,
        maximum_critical_modes=args.maximum_critical_modes
    )
    if args.no_gram_guard:
        settings = replace(settings, enforce_gram_guard=False)
    if args.event_triggered_guard:
        settings = replace(settings, guard_policy="event_triggered")
        if args.event_gram_hard_floor is not None:
            settings = replace(
                settings, event_gram_hard_floor=args.event_gram_hard_floor
            )
    elif args.event_gram_hard_floor is not None:
        raise ValueError(
            "--event-gram-hard-floor requires --event-triggered-guard"
        )
    if args.batched_mode_gradients:
        settings = replace(settings, batched_mode_gradients=True)
    if args.solver_setting_profile != DEFAULT_SOLVER_SETTING_PROFILE:
        settings = replace(
            settings, solver_setting_profile=args.solver_setting_profile
        )
    if args.bundle_seed_policy != DEFAULT_BUNDLE_SEED_POLICY:
        settings = replace(
            settings, bundle_seed_policy=args.bundle_seed_policy
        )
    if args.bundle_seed_spectral_threshold is not None:
        settings = replace(
            settings,
            bundle_seed_spectral_threshold=(
                args.bundle_seed_spectral_threshold
            ),
        )
    if args.bundle_seed_rank_margin is not None:
        settings = replace(
            settings, bundle_seed_rank_margin=args.bundle_seed_rank_margin
        )
    if args.bundle_growth_factor is not None:
        settings = replace(
            settings, bundle_growth_factor=args.bundle_growth_factor
        )
    if args.bundle_growth_trigger is not None:
        settings = replace(
            settings, bundle_growth_trigger=args.bundle_growth_trigger
        )
    if args.bundle_interior_overshoot is not None:
        settings = replace(
            settings,
            bundle_interior_overshoot=args.bundle_interior_overshoot,
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
        if "bundle_accepted_rank" in initial_arrays:
            model.restore_bundle_accepted_rank(
                int(np.asarray(initial_arrays["bundle_accepted_rank"]))
            )
        if "bundle_accepted_basis" in initial_arrays:
            model.restore_bundle_accepted_basis(
                np.asarray(
                    initial_arrays["bundle_accepted_basis"], dtype=complex
                ),
                rank=(
                    int(np.asarray(initial_arrays["bundle_accepted_rank"]))
                    if "bundle_accepted_rank" in initial_arrays
                    else None
                ),
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

    carries_basis = (
        settings.bundle_seed_policy in CWRMF_BASIS_CARRYING_SEED_POLICIES
    )

    def checkpoint(step: int, current_time: float, state: np.ndarray) -> None:
        # Continuation metadata beside the physical state: the monotone rank
        # hint and the accepted rank are scalars and always recorded; the
        # accepted working set is a 62-row complex block and is recorded only
        # when the active seed policy is one that reads it, so a default-path
        # checkpoint gains no array.
        continuation: dict[str, np.ndarray] = {
            "bundle_rank_hint": np.asarray(model.bundle_rank_hint),
            "bundle_accepted_rank": np.asarray(model.bundle_accepted_rank),
        }
        accepted_basis = model.bundle_accepted_basis
        if carries_basis and accepted_basis is not None:
            continuation["bundle_accepted_basis"] = accepted_basis
        temporary = checkpoint_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                step=np.asarray(step),
                time=np.asarray(current_time),
                state=state,
                **continuation,
            )
        os.replace(temporary, checkpoint_path)

    if args.tier_timing:
        # Enabled after the preparation so the accumulators account only for
        # the rollout, next to maximum_atom_seconds.
        model.enable_tier_timing()
    instance_recorder = None
    if args.record_conic_instances:
        instance_recorder = ConicInstanceRecorder(
            output_directory / "conic_instances"
        )
        instance_recorder.write_context(
            step_scaled_completion_coefficients=(
                args.time_step
                * model.geometry.scaled_completion_coefficients()
            ),
            readable_completion_indices=(
                model.geometry.readable_completion_indices
            ),
            completion_count=model.geometry.completion_count,
            numerical_profile=args.numerical_profile,
            schur_safety_margin=settings.schur_safety_margin,
            solver_tolerance=settings.solver_tolerance,
            psd_inflation=settings.psd_inflation,
        )
        model.enable_instance_recording(instance_recorder)
    rollout_started = time.perf_counter()
    trajectory = integrate_cwrmf_ssprk2(
        model,
        initial_state,
        initial_time=initial_time,
        final_time=args.final_time,
        time_step=args.time_step,
        maximum_step_subdivisions=args.maximum_step_subdivisions,
        progress=progress,
        checkpoint=checkpoint,
    )
    rollout_elapsed = time.perf_counter() - rollout_started
    autonomous_elapsed = time.perf_counter() - started
    tier_timing = model.tier_timing
    if tier_timing is not None:
        tier_timing["rollout_wall_seconds"] = float(rollout_elapsed)
        tier_timing["unaccounted_seconds"] = float(
            rollout_elapsed - sum(tier_timing["seconds"].values())
        )
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
    # The same zero test the exact reference itself applies to its sample
    # times.  ``np.isclose`` defaults to an absolute tolerance of 1e-8, which
    # calls a resume that starts at or below 1e-8 a rollout from zero and then
    # hands the reference a first sample time it rejects at its own 1e-15.
    if abs(float(trajectory.times[0])) <= 1e-15:
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
        "model": (
            "carried_higher_moment_flow_without_gram_guard"
            if args.no_gram_guard
            else (
                "carried_witness_event_triggered_guard_flow"
                if args.event_triggered_guard
                else "carried_witness_radial_moment_flow"
            )
        ),
        "numerical_profile": args.numerical_profile,
        "numerical_settings": asdict(settings),
        "gram_guard_enabled": settings.enforce_gram_guard,
        "guard_policy": settings.guard_policy,
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
            "method": (
                "SSPRK2 with unconstrained carried higher-moment rates"
                if args.no_gram_guard
                else (
                    "SSPRK2 with event-triggered carried-witness radial atoms"
                    if args.event_triggered_guard
                    else "SSPRK2 with finite-step carried-witness radial atoms"
                )
            ),
            "initial_time": initial_time,
            "final_time_requested": args.final_time,
            "last_time": float(trajectory.times[-1]),
            "time_step": args.time_step,
            "completed_steps": trajectory.completed_steps,
            "atom_evaluations": trajectory.atom_evaluations,
            "maximum_step_subdivisions": args.maximum_step_subdivisions,
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
            "maximum_atom_seconds": float(
                np.max(trajectory.maximum_atom_seconds)
            ),
            "tier_seconds": tier_timing,
            "recorded_conic_instances": (
                None
                if instance_recorder is None
                else int(instance_recorder.instance_count)
            ),
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
            "minimum_velocity_margin": (
                float(np.nanmin(trajectory.velocity_margins))
                if settings.enforce_gram_guard
                and bool(np.any(np.isfinite(trajectory.velocity_margins)))
                else None
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
