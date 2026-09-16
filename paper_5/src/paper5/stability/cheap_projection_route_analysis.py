"""Run the cheap physicality repair as a named experiment.

One rollout of the fixed-dictionary APCM state (44 coordinates, 62 carried
frontier coordinates, 24-row positive extension) propagated with the
unguarded velocity and repaired by one of the named cheap projections of
``cheap_projection_route``.  No conic solve runs after the preparation.  The
trajectory is scored against the exact cutoff-16 driven reference with the
same helpers the carried-witness driver uses, and every physicality number
reported is a measured minimum eigenvalue along the trajectory -- there is no
certificate on this route.

Reference lines the run is placed next to, all at strong coupling, phonon
cutoff 16, exact-correlated preparation:

``t=0`` to ``4``, ``h=0.0025``
    projective fixed-dictionary APCM with the event-triggered outer guard
    ``0.026049`` at ``3,229.8 s``; raw 31-coordinate archive EOM ``0.11090``;
    carried moments with no Gram guard ``0.48049`` at ``153.53 s``.

``t=0`` to ``0.05``, ``h=0.0025``
    carried-witness strict always-on guard ``3.236e-6``; the projective route
    with the declared completion retry ladder ``1.2277e-5``; the projective
    route with the stage retraction disabled ``1.9408e-6`` over ``t=0`` to
    ``0.015`` before its per-stage completion solve refuses.
"""

from __future__ import annotations

import argparse
import json
import resource
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    raw_moment_coordinates_to_matrix_state,
)
from .apcm_carried_witness_analysis import _accuracy_metrics
from .apcm_moment_projection import APCMProjectionSettings
from .apcm_positive_extension import APCMExtensionSettings
from .cheap_projection_route import (
    CHEAP_PROJECTION_CADENCES,
    CHEAP_PROJECTION_COORDINATES,
    CHEAP_PROJECTION_ENERGY_REFERENCES,
    CHEAP_PROJECTION_POLICIES,
    CHEAP_PROJECTION_TARGETS,
    CarriedWitnessAffineGram,
    CheapProjectionSettings,
    integrate_carried_witness_cheap_projection,
    integrate_cheap_projection_ssprk3,
    prepare_cheap_projection_route,
)
from .exact_reference import exact_holstein_driven_trajectory
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
)
from .projected_apcm import (
    ProjectedAPCMSettings,
    unpack_projected_apcm_state,
)

_RECORDED = {
    "projective_event_guard_all_coordinate_scalar_rms_t0_to_4": 0.026049,
    "raw_archive_eom_31_coordinate_rms_t0_to_4": 0.11090,
    "carried_moments_no_gram_guard_31_coordinate_rms_t0_to_4": 0.48049,
    "carried_witness_strict_all_coordinate_scalar_rms_t0_to_0p05": 3.236e-6,
    "projective_retry_ladder_all_coordinate_scalar_rms_t0_to_0p05": 1.2277e-5,
    "projective_no_retraction_all_coordinate_scalar_rms_t0_to_0p015": 1.9408e-6,
    "both_cones_all_coordinate_scalar_rms_t0_to_20": 0.128607,
    "plain_archive_eom_all_coordinate_scalar_rms_t0_to_20": 0.193205,
    "both_cones_internal_energy_rms_t0_to_20": 0.000939628,
    "plain_archive_eom_internal_energy_rms_t0_to_20": 0.0182361,
    "projective_event_guard_wall_seconds_t0_to_4": 3229.8,
    "cheap_route_dykstra30_both_no_reselection_rms_t0_to_4": 0.1313351146218195,
    "cheap_route_no_repair_rms_t0_to_4": 0.13304,
    "source": "worklog: 2026-09-14 and 2026-09-15 overnight rounds 1 to 3; "
    "the t=0 to 20 lines are worklog:2781-2792 and 2898-2903",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--state",
        choices=("fixed_dictionary", "carried_witness"),
        default="fixed_dictionary",
        help=(
            "Which moment state is propagated: the 44-coordinate fixed "
            "dictionary with its carried lift, or the 384-coordinate "
            "carried-witness state through its no-guard flow with the repair "
            "applied to the 62-row unified Gram after every step."
        ),
    )
    parser.add_argument("--final-time", type=float, default=0.05)
    parser.add_argument("--time-step", type=float, default=0.0025)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
    parser.add_argument(
        "--projection-policy",
        choices=CHEAP_PROJECTION_POLICIES,
        default="eigenvalue_clip",
    )
    parser.add_argument(
        "--projection-target",
        choices=CHEAP_PROJECTION_TARGETS,
        default="extension",
    )
    parser.add_argument(
        "--projection-cadence",
        choices=CHEAP_PROJECTION_CADENCES,
        default="stage",
    )
    parser.add_argument("--eigenvalue-floor", type=float, default=0.0)
    parser.add_argument(
        "--retained-eigenvalue-floor",
        type=float,
        default=None,
        help=(
            "Separate floor for the retained joint Gram block; the default "
            "None uses --eigenvalue-floor for both blocks."
        ),
    )
    parser.add_argument(
        "--repair-coordinates",
        choices=CHEAP_PROJECTION_COORDINATES,
        default="state_and_frontier",
    )
    parser.add_argument("--dykstra-iterations", type=int, default=5)
    parser.add_argument("--repair-tolerance", type=float, default=0.0)
    parser.add_argument("--state-motion-scale", type=float, default=1.0)
    parser.add_argument("--frontier-motion-scale", type=float, default=1.0)
    parser.add_argument(
        "--entrance-keys",
        type=int,
        default=None,
        help=(
            "Optional prefix of ENTRANCE_RELATIVE_MOMENT_KEYS for a small "
            "fixture; the declared dictionary is all 15 keys."
        ),
    )
    parser.add_argument(
        "--lift-reselection-interval",
        type=int,
        default=0,
        help=(
            "Re-solve the declared proximal log-det completion at the "
            "accepted state every N steps and carry the solved lift, "
            "repaired by the cheap projection, in between.  0 carries the "
            "preparation's lift for the whole rollout, which is the route's "
            "original form.  Fixed-dictionary state only."
        ),
    )
    parser.add_argument(
        "--lift-reselection-start",
        choices=("warm", "cold"),
        default="warm",
        help=(
            "Whether a re-selection hands the carried frontier to the "
            "completion as its warm frontier."
        ),
    )
    parser.add_argument(
        "--energy-reference-mode",
        choices=CHEAP_PROJECTION_ENERGY_REFERENCES,
        default="none",
        help=(
            "Whether the dykstra repair also holds the Eq. (22) internal "
            "energy on a declared reference.  post_pulse turns the "
            "constraint on at --energy-reference-time and holds the energy "
            "at the value this trajectory itself had there, which is the "
            "model's exact conservation law once the drive has ended; "
            "drive_power turns it on at the start of the rollout and "
            "advances the reference by the power the drive delivers.  "
            "Fixed-dictionary state only."
        ),
    )
    parser.add_argument(
        "--energy-reference-time",
        type=float,
        default=4.0,
        help=(
            "The declared time the drive ends, used by the post_pulse "
            "energy reference mode."
        ),
    )
    parser.add_argument(
        "--energy-tolerance",
        type=float,
        default=0.0,
        help=(
            "With an active energy reference, a repair is applied whenever "
            "the internal energy is further than this from the reference, "
            "even when both cones are already satisfied."
        ),
    )
    parser.add_argument(
        "--completion-retry-shifts",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Declared ladder of log-determinant shifts retried, in order, "
            "when a lift re-selection's declared completion is not "
            "certified; empty is this route's earlier form, one solve and "
            "the carried lift kept on a refusal."
        ),
    )
    parser.add_argument(
        "--completion-retry-weights",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Declared ladder of log-determinant weights retried after the "
            "shifts at a refused lift re-selection."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help=(
            "Continue a rollout that stopped on its wall budget: the run "
            "directory whose trajectory.npz supplies the last accepted "
            "state, the carried frontier and the time this segment starts "
            "at, and whose summary.json supplies the declared "
            "internal-energy reference the segment carries.  The exact "
            "reference of a resumed segment is propagated from t=0 and "
            "sampled on that segment's own times."
        ),
    )
    parser.add_argument(
        "--completion-conic-tolerance",
        type=float,
        default=None,
        help=(
            "Conic tolerance of the proximal log-det completion the lift "
            "re-selections solve.  None keeps the declared "
            "APCMExtensionSettings value (1e-9) for every solve; a value "
            "is a named sensitivity setting that changes the re-selection "
            "completion alone and is recorded in the summary's "
            "reselection_extension_settings block.  The preparation's "
            "completion at t=0 keeps the declared settings: at 1e-10 it "
            "is not certified (Clarabel AlmostSolved), so the tighter "
            "decade cannot be applied there."
        ),
    )
    parser.add_argument("--wall-budget-seconds", type=float, default=1200.0)
    parser.add_argument(
        "--record-stages",
        action="store_true",
        help="Save one JSON line per repair to repairs.jsonl.",
    )
    return parser


def _extension_settings(
    conic_tolerance: float | None,
) -> APCMExtensionSettings:
    """The declared completion settings, with only the conic tolerance movable.

    ``None`` is the earlier form of the driver, the declared
    ``APCMExtensionSettings`` untouched; a value replaces that one field and
    nothing else, so a sensitivity rerun differs from the recorded run in
    exactly one declared numerical setting.
    """

    if conic_tolerance is None:
        return APCMExtensionSettings()
    return replace(
        APCMExtensionSettings(), conic_tolerance=float(conic_tolerance)
    )


def _peak_resident_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _exact_archive_coordinates(
    parameters: DimerParameters,
    times: np.ndarray,
    *,
    phonon_cutoff: int,
    time_step: float,
) -> np.ndarray:
    requested = np.asarray(times, dtype=float)
    prepended = abs(float(requested[0])) > 1e-15
    sample_times = (
        np.concatenate(([0.0], requested)) if prepended else requested
    )
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=sample_times,
        phonon_cutoff=phonon_cutoff,
        maximum_step=min(0.01, time_step * 4.0),
    )
    coordinates = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(state)
            for state in exact.matrix_states
        ],
        dtype=float,
    )
    return coordinates[1:] if prepended else coordinates


def _archive_coordinates(
    states: np.ndarray, *, active_keys: tuple
) -> np.ndarray:
    rows = []
    for state in states:
        raw, _ = unpack_projected_apcm_state(state, active_keys=active_keys)
        rows.append(
            matrix_state_to_closed_scalar_coordinates(
                raw_moment_coordinates_to_matrix_state(raw)
            )
        )
    return np.asarray(rows, dtype=float)


def _resume_state(
    directory: Path,
    *,
    parameters: DimerParameters,
    phonon_cutoff: int,
    time_step: float,
    entrance_key_count: int,
    settings: CheapProjectionSettings,
) -> dict[str, Any]:
    """Read the last accepted state of a segment that stopped on its budget.

    A segment is continued from exactly three saved quantities -- the last
    accepted 44-coordinate state, the standardized frontier carried with it,
    and the time that state was accepted at -- plus the declared
    internal-energy reference the earlier segment selected, so the resumed
    repair holds the same hyperplane rather than re-declaring one at the
    resumed state.  The model, the step and the preparation are re-derived
    from the same flags; a segment whose step, parameters, cutoff or
    dictionary differ is refused, and a segment whose repair settings differ
    is allowed and recorded, because completing a stopped run under a
    changed option is a declared experiment and must be visible in the
    summary.
    """

    directory = directory.resolve()
    summary = json.loads(
        (directory / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(directory / "trajectory.npz") as archive:
        times = np.asarray(archive["times"], dtype=float)
        states = np.asarray(archive["states"], dtype=float)
        frontiers = np.asarray(archive["frontier_standardized"], dtype=float)
        references = (
            np.asarray(archive["internal_energy_reference"], dtype=float)
            if "internal_energy_reference" in archive
            else np.zeros(0, dtype=float)
        )
    saved_parameters = summary["parameters"]
    for name, value in (
        ("gamma", parameters.gamma),
        ("lambda_ep", parameters.lambda_ep),
        ("drive_amplitude", parameters.drive_amplitude),
    ):
        if not np.isclose(float(saved_parameters[name]), float(value)):
            raise ValueError(
                f"the resumed segment was run at {name}="
                f"{saved_parameters[name]}, not {value}"
            )
    if int(summary["phonon_cutoff"]) != int(phonon_cutoff):
        raise ValueError("the resumed segment used a different phonon cutoff")
    if int(summary["entrance_key_count"]) != int(entrance_key_count):
        raise ValueError(
            "the resumed segment used a different entrance dictionary"
        )
    if not np.isclose(
        float(summary["integration"]["time_step"]), float(time_step)
    ):
        raise ValueError("the resumed segment used a different time step")
    saved_settings = dict(summary["projection_settings"])
    current_settings = dict(asdict(settings))
    changed = {
        key: [saved_settings.get(key), current_settings.get(key)]
        for key in sorted(set(saved_settings) | set(current_settings))
        if saved_settings.get(key) != current_settings.get(key)
    }
    reference = summary.get("internal_energy", {}).get(
        "declared_reference_value", float("nan")
    )
    if reference is None or not np.isfinite(float(reference)):
        reference = (
            float(references[-1])
            if references.size and np.isfinite(float(references[-1]))
            else None
        )
    else:
        reference = float(reference)
    return {
        "directory": str(directory),
        "initial_time": float(times[-1]),
        "state": states[-1].copy(),
        "frontier_standardized": frontiers[-1].copy(),
        "energy_reference_value": reference,
        "source_last_time": float(times[-1]),
        "source_completed_steps": int(
            summary["integration"]["completed_steps"]
        ),
        "source_message": summary["integration"]["message"],
        "projection_settings_changed": changed,
    }


def _carried_witness_main(args: argparse.Namespace) -> int:
    """Task (c): the 384-coordinate no-guard flow with the same repair."""

    from .apcm_carried_witness import CWRMFSettings, CarriedWitnessModel

    if args.energy_reference_mode != "none":
        raise ValueError(
            "the internal-energy hyperplane is implemented for the "
            "fixed-dictionary state; the carried-witness lane repairs the "
            "62-row unified Gram only"
        )
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    def progress(message: str) -> None:
        print(message, flush=True)

    parameters = DimerParameters(
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
    )
    settings = CheapProjectionSettings(
        policy=args.projection_policy,
        target="extension",
        cadence="step",
        eigenvalue_floor=args.eigenvalue_floor,
        repair_coordinates=args.repair_coordinates,
        dykstra_iterations=args.dykstra_iterations,
        repair_tolerance=args.repair_tolerance,
        state_motion_scale=args.state_motion_scale,
        frontier_motion_scale=args.frontier_motion_scale,
    )
    cwrmf_settings = CWRMFSettings(enforce_gram_guard=False)
    build_started = perf_counter()
    model = CarriedWitnessModel(parameters, settings=cwrmf_settings)
    preparation = model.prepare(phonon_cutoff=args.phonon_cutoff)
    gram = CarriedWitnessAffineGram(model.geometry)
    gram.factorization(settings)
    build_elapsed = perf_counter() - build_started
    progress(
        f"prepared state_dim={model.geometry.state_count} "
        f"completion_dim={model.geometry.completion_count} "
        f"gram_rows={gram.rows} build_seconds={build_elapsed:.3f}"
    )

    started = perf_counter()
    trajectory = integrate_carried_witness_cheap_projection(
        model,
        gram,
        preparation.state,
        settings=settings,
        final_time=args.final_time,
        time_step=args.time_step,
        progress=progress,
        deadline=started + float(args.wall_budget_seconds),
    )
    autonomous_elapsed = perf_counter() - started

    approximate = np.asarray(
        [
            matrix_state_to_closed_scalar_coordinates(
                raw_moment_coordinates_to_matrix_state(
                    model.geometry.split_retained(
                        model.geometry.unpack_state(state)[0]
                    )[0]
                )
            )
            for state in trajectory.states
        ],
        dtype=float,
    )
    exact_started = perf_counter()
    if trajectory.times.size >= 2:
        exact = _exact_archive_coordinates(
            parameters,
            trajectory.times,
            phonon_cutoff=args.phonon_cutoff,
            time_step=args.time_step,
        )
        metrics: dict[str, Any] | None = _accuracy_metrics(
            parameters, exact, approximate
        )
    else:
        exact = np.zeros((0, approximate.shape[1]))
        metrics = None
    exact_elapsed = perf_counter() - exact_started
    joint_minimum = float(
        min(
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    raw_moment_coordinates_to_matrix_state(
                        model.geometry.split_retained(
                            model.geometry.unpack_state(state)[0]
                        )[0]
                    )
                )
            )[0]
            for state in trajectory.states
        )
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "model": "carried_witness_no_gram_guard_with_cheap_projection_repair",
        "route": (
            "the record's no-guard CWRMF flow (enforce_gram_guard=False) "
            "stepped one step at a time, with the declared cheap projection "
            "applied to the 62-row unified Gram after every accepted step; "
            "no conic solve, no certificate"
        ),
        "parameters": asdict(parameters),
        "phonon_cutoff": args.phonon_cutoff,
        "projection_settings": asdict(settings),
        "applied_eigenvalue_floor": settings.applied_floor,
        "cwrmf_settings": asdict(cwrmf_settings),
        "integration": {
            "method": "SSPRK2 no-guard CWRMF with a per-step cheap repair",
            "final_time_requested": args.final_time,
            "last_time": float(trajectory.times[-1]),
            "time_step": args.time_step,
            "completed_steps": trajectory.completed_steps,
            "success": trajectory.success,
            "message": trajectory.message,
            "build_wall_seconds": build_elapsed,
            "autonomous_wall_seconds": autonomous_elapsed,
            "propagation_wall_seconds": trajectory.rollout_seconds,
            "exact_reference_wall_seconds": exact_elapsed,
            "wall_budget_seconds": args.wall_budget_seconds,
            "peak_resident_bytes": _peak_resident_bytes(),
        },
        "repair": {
            "applied": trajectory.repairs_applied,
            "skipped": trajectory.repairs_skipped,
            "repair_wall_seconds": trajectory.repair_seconds,
            "maximum_retained_displacement_euclidean": (
                trajectory.maximum_retained_displacement
            ),
            "maximum_completion_displacement_euclidean": (
                trajectory.maximum_completion_displacement
            ),
            "maximum_dykstra_iterations": (
                trajectory.maximum_dykstra_iterations
            ),
            "mean_repair_milliseconds": (
                1e3
                * trajectory.repair_seconds
                / max(
                    1,
                    trajectory.repairs_applied + trajectory.repairs_skipped,
                )
            ),
        },
        "feasibility": {
            "minimum_unshifted_retained_joint_gram_eigenvalue": joint_minimum,
            "minimum_scaled_unified_62_row_gram_eigenvalue": float(
                np.min(trajectory.minimum_unshifted_eigenvalues)
            ),
            "minimum_scaled_unified_62_row_gram_eigenvalue_before_repair": (
                trajectory.minimum_before_repair
            ),
        },
        "accuracy": metrics,
        "recorded_reference": _RECORDED,
    }
    np.savez_compressed(
        output_directory / "trajectory.npz",
        times=trajectory.times,
        carried_states=trajectory.states,
        approximate_archive_coordinates=approximate,
        exact_archive_coordinates=exact,
        minimum_unshifted_eigenvalues=(
            trajectory.minimum_unshifted_eigenvalues
        ),
    )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if trajectory.success else 1


def main() -> int:
    args = _parser().parse_args()
    if args.state == "carried_witness":
        return _carried_witness_main(args)
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=False)

    def progress(message: str) -> None:
        print(message, flush=True)

    if args.entrance_keys is None:
        active_keys = ENTRANCE_RELATIVE_MOMENT_KEYS
    else:
        if not 1 <= args.entrance_keys <= len(ENTRANCE_RELATIVE_MOMENT_KEYS):
            raise ValueError("entrance key prefix is out of range")
        active_keys = ENTRANCE_RELATIVE_MOMENT_KEYS[: args.entrance_keys]

    parameters = DimerParameters(
        gamma=args.gamma,
        lambda_ep=args.lambda_ep,
        drive_amplitude=args.drive,
    )
    settings = CheapProjectionSettings(
        policy=args.projection_policy,
        target=args.projection_target,
        cadence=args.projection_cadence,
        eigenvalue_floor=args.eigenvalue_floor,
        retained_eigenvalue_floor=args.retained_eigenvalue_floor,
        repair_coordinates=args.repair_coordinates,
        dykstra_iterations=args.dykstra_iterations,
        repair_tolerance=args.repair_tolerance,
        state_motion_scale=args.state_motion_scale,
        frontier_motion_scale=args.frontier_motion_scale,
        energy_reference_mode=args.energy_reference_mode,
        energy_tolerance=args.energy_tolerance,
    )
    extension_settings = APCMExtensionSettings()
    reselection_extension_settings = (
        None
        if args.completion_conic_tolerance is None
        else _extension_settings(args.completion_conic_tolerance)
    )
    projection_settings = APCMProjectionSettings()
    model_settings = ProjectedAPCMSettings()

    build_started = perf_counter()
    route, initial_state, initial_frontier, ground_energy = (
        prepare_cheap_projection_route(
            parameters,
            phonon_cutoff=args.phonon_cutoff,
            active_keys=active_keys,
            settings=settings,
            model_settings=model_settings,
            projection_settings=projection_settings,
            extension_settings=extension_settings,
            completion_retry_shifts=tuple(args.completion_retry_shifts),
            completion_retry_weights=tuple(args.completion_retry_weights),
            reselection_extension_settings=reselection_extension_settings,
        )
    )
    route.coordinates.factorization(settings)
    build_elapsed = perf_counter() - build_started
    progress(
        f"prepared state_dim={initial_state.size} "
        f"frontier={initial_frontier.size} "
        f"gram_rows={route.coordinates.extension_rows} "
        f"build_seconds={build_elapsed:.3f}"
    )

    initial_time = 0.0
    energy_reference_value: float | None = None
    resume: dict[str, Any] | None = None
    if args.resume_from is not None:
        resume = _resume_state(
            args.resume_from,
            parameters=parameters,
            phonon_cutoff=args.phonon_cutoff,
            time_step=args.time_step,
            entrance_key_count=len(route.coordinates.active_keys),
            settings=settings,
        )
        initial_time = float(resume["initial_time"])
        initial_state = np.asarray(resume["state"], dtype=float)
        initial_frontier = np.asarray(
            resume["frontier_standardized"], dtype=float
        )
        energy_reference_value = resume["energy_reference_value"]
        progress(
            f"resumed from {resume['directory']} at t={initial_time:.6f} "
            f"reference={energy_reference_value}"
        )

    started = perf_counter()
    trajectory = integrate_cheap_projection_ssprk3(
        route,
        initial_state,
        initial_frontier,
        final_time=args.final_time,
        time_step=args.time_step,
        initial_time=initial_time,
        progress=progress,
        deadline=started + float(args.wall_budget_seconds),
        record_stages=bool(args.record_stages),
        lift_reselection_interval=int(args.lift_reselection_interval),
        lift_reselection_warm_start=args.lift_reselection_start == "warm",
        energy_reference_time=float(args.energy_reference_time),
        energy_reference_value=energy_reference_value,
    )
    autonomous_elapsed = perf_counter() - started

    approximate = _archive_coordinates(
        trajectory.states, active_keys=route.coordinates.active_keys
    )
    exact_started = perf_counter()
    if trajectory.times.size >= 2:
        exact = _exact_archive_coordinates(
            parameters,
            trajectory.times,
            phonon_cutoff=args.phonon_cutoff,
            time_step=args.time_step,
        )
        metrics: dict[str, Any] | None = _accuracy_metrics(
            parameters, exact, approximate
        )
    else:
        exact = np.zeros((0, approximate.shape[1]))
        metrics = None
    exact_elapsed = perf_counter() - exact_started

    joint_minimum = float(
        min(
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    raw_moment_coordinates_to_matrix_state(
                        unpack_projected_apcm_state(
                            state, active_keys=route.coordinates.active_keys
                        )[0]
                    )
                )
            )[0]
            for state in trajectory.states
        )
    )

    resume_block: dict[str, Any] | None = None
    if resume is not None:
        resume_block = {
            "source_directory": resume["directory"],
            "source_last_time": resume["source_last_time"],
            "source_completed_steps": resume["source_completed_steps"],
            "source_message": resume["source_message"],
            "initial_time": float(initial_time),
            "supplied_internal_energy_reference": (
                resume["energy_reference_value"]
            ),
            "projection_settings_changed": resume[
                "projection_settings_changed"
            ],
        }

    summary: dict[str, Any] = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "model": "fixed_dictionary_projected_apcm_cheap_projection_repair",
        "route": (
            "unguarded fixed-dictionary velocity (archive plus exact K/P/D "
            "entrance and exact active commutators) with the carried frontier "
            "lift; physicality repaired by a cheap projection with no conic "
            "solve and no certificate"
        ),
        "parameters": asdict(parameters),
        "phonon_cutoff": args.phonon_cutoff,
        "entrance_key_count": len(route.coordinates.active_keys),
        "exact_ground_energy": ground_energy,
        "projection_settings": asdict(settings),
        "applied_eigenvalue_floor": settings.applied_floor,
        "applied_retained_eigenvalue_floor": settings.applied_retained_floor,
        "model_settings": asdict(model_settings),
        "apcm_projection_settings": asdict(projection_settings),
        "extension_settings": asdict(extension_settings),
        "reselection_extension_settings": (
            None
            if reselection_extension_settings is None
            else asdict(reselection_extension_settings)
        ),
        "integration": {
            "method": (
                "SSPRK(3,3) on the unguarded velocity with the declared "
                "cheap repair"
            ),
            "final_time_requested": args.final_time,
            "initial_time": float(initial_time),
            "last_time": float(trajectory.times[-1]),
            "time_step": args.time_step,
            "completed_steps": trajectory.completed_steps,
            "velocity_evaluations": trajectory.velocity_evaluations,
            "success": trajectory.success,
            "message": trajectory.message,
            "build_wall_seconds": build_elapsed,
            "autonomous_wall_seconds": autonomous_elapsed,
            "exact_reference_wall_seconds": exact_elapsed,
            "wall_budget_seconds": args.wall_budget_seconds,
            "peak_resident_bytes": _peak_resident_bytes(),
        },
        "repair": {
            "applied": trajectory.repairs_applied,
            "skipped": trajectory.repairs_skipped,
            "repair_wall_seconds": trajectory.repair_seconds,
            "velocity_wall_seconds": trajectory.velocity_seconds,
            "maximum_state_displacement_euclidean": (
                trajectory.maximum_state_displacement
            ),
            "maximum_frontier_displacement_euclidean": (
                trajectory.maximum_frontier_displacement
            ),
            "maximum_dykstra_iterations": (
                trajectory.maximum_dykstra_iterations
            ),
            "mean_repair_milliseconds": (
                1e3
                * trajectory.repair_seconds
                / max(
                    1,
                    trajectory.repairs_applied + trajectory.repairs_skipped,
                )
            ),
        },
        "lift": {
            "reselection_interval_steps": int(args.lift_reselection_interval),
            "reselection_start": args.lift_reselection_start,
            "reselections": trajectory.lift_reselections,
            "reselection_failures": trajectory.lift_reselection_failures,
            "reselection_wall_seconds": trajectory.lift_reselection_seconds,
            "mean_reselection_seconds": (
                trajectory.lift_reselection_seconds
                / max(1, trajectory.lift_reselections)
            ),
            "maximum_reselection_frontier_displacement_euclidean": (
                trajectory.maximum_lift_reselection_displacement
            ),
            "minimum_reselected_scaled_extension_eigenvalue": (
                trajectory.minimum_reselected_scaled_extension_eigenvalue
            ),
            "completion_retry_shifts": [
                float(value) for value in args.completion_retry_shifts
            ],
            "completion_retry_weights": [
                float(value) for value in args.completion_retry_weights
            ],
            "completion_retry_ladder_length": len(route.completion_ladder),
            "retries_attempted": (
                trajectory.lift_reselection_retries_attempted
            ),
            "retries_accepted": trajectory.lift_reselection_retries_accepted,
            "retries_exhausted": (
                trajectory.lift_reselection_retries_exhausted
            ),
            "retry_solves": trajectory.lift_reselection_retry_solves,
            "retry_wall_seconds": trajectory.lift_reselection_retry_seconds,
        },
        "internal_energy": {
            "reference_mode": args.energy_reference_mode,
            "declared_reference_time": float(args.energy_reference_time),
            "energy_tolerance": float(args.energy_tolerance),
            "reference_supplied_by_resume": bool(
                trajectory.energy_reference_supplied
            ),
            "declared_reference_value": (
                trajectory.energy_reference_value
            ),
            "constrained_repairs": trajectory.energy_constrained_repairs,
            "hyperplane_refusals": trajectory.energy_hyperplane_refusals,
            "maximum_absolute_deviation_from_reference": (
                trajectory.maximum_absolute_energy_deviation
            ),
            "value_at_first_sample": (
                float(trajectory.internal_energy[0])
                if trajectory.internal_energy.size
                else float("nan")
            ),
            "value_at_last_sample": (
                float(trajectory.internal_energy[-1])
                if trajectory.internal_energy.size
                else float("nan")
            ),
        },
        "feasibility": {
            "minimum_unshifted_retained_joint_gram_eigenvalue": joint_minimum,
            "minimum_unshifted_affine_retained_gram_eigenvalue": float(
                np.min(trajectory.retained_minimum_eigenvalues)
            ),
            "minimum_scaled_positive_extension_eigenvalue": float(
                np.min(trajectory.scaled_extension_minimum_eigenvalues)
            ),
            "maximum_standardized_frontier_envelope_ratio": float(
                np.max(trajectory.maximum_frontier_envelope_ratio)
            ),
        },
        "accuracy": metrics,
        "resume": resume_block,
        "recorded_reference": _RECORDED,
    }

    np.savez_compressed(
        output_directory / "trajectory.npz",
        times=trajectory.times,
        states=trajectory.states,
        frontier_standardized=trajectory.frontier_standardized,
        approximate_archive_coordinates=approximate,
        exact_archive_coordinates=exact,
        retained_minimum_eigenvalues=trajectory.retained_minimum_eigenvalues,
        scaled_extension_minimum_eigenvalues=(
            trajectory.scaled_extension_minimum_eigenvalues
        ),
        maximum_frontier_envelope_ratio=(
            trajectory.maximum_frontier_envelope_ratio
        ),
        internal_energy=trajectory.internal_energy,
        internal_energy_reference=trajectory.internal_energy_reference,
    )
    if trajectory.records:
        (output_directory / "repairs.jsonl").write_text(
            "".join(
                json.dumps(record, sort_keys=True) + "\n"
                for record in trajectory.records
            ),
            encoding="utf-8",
        )
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if trajectory.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
