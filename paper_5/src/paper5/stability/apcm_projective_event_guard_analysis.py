"""Run the projective (URPG) outer guard as a named experiment.

Three declared measurements share one command, each of which is exploratory
and none of which changes a default:

``--mode gate-reorder``
    One outer-aware invariant target solve at the correlated preparation with
    and without the kept gate reorder (``accept_first_provisional``).  The
    record's figures are about 500 s then 171 s at an unchanged objective
    ``0.01126712315`` (worklog:3267-3269, 3282-3288).

``--mode selector-audit``
    The same outer-aware target stage solved once as a strictly convex
    problem: the invariant target image quadratic in the whitened metric plus
    a fixed positive-definite frontier term of declared weight, replacing the
    two-solve selector (target image, then the minimum-norm lift under
    ``R_D y = q*``) whose lift returned ``NumericalError``.  Reported against
    the outer-aware witness of the gate-reorder artifact: objective and
    target-image displacement, next to the record's 0.5193246 displacement of
    that witness from the current-only target.

``--mode rollout``
    Fixed-dictionary APCM, SSPRK(3,3), containment before every RHS call,
    with the event-triggered outer guard: the unconstrained fixed-dictionary
    predictor is formed, its endpoint Gram is eigendecomposed against the
    carried frontier witness, the shared negative-mode coupling measure is
    evaluated below the declared floor, and
    ``select_outer_feasible_projective_guard`` runs only on triggered stages.
    Scored offline against the exact cutoff Hamiltonian reference with the
    same helpers the carried-witness driver uses.
"""

from __future__ import annotations

import argparse
import json
import resource
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    matrix_state_to_raw_moment_coordinates,
    raw_moment_coordinates_to_matrix_state,
)
from .apcm_carried_witness_analysis import _accuracy_metrics
from .apcm_moment_projection import (
    APCMProjectionSettings,
    SymmetryReducedAPCMGeometry,
    state_lower_moments,
)
from .apcm_positive_extension import (
    APCMExtensionSettings,
    SymmetryReducedPositiveExtension,
)
from .apcm_projective_guard import (
    compile_invariant_target_readout,
    select_outer_feasible_projective_guard,
)
from .exact_reference import (
    exact_holstein_driven_trajectory,
    exact_holstein_joint_moment_initial_state,
)
from .hubbard_dimer import DimerParameters
from .matrix_reference import (
    electron_phonon_moment_matrix,
    matrix_state_to_closed_scalar_coordinates,
)
from .moment_hierarchy import THIRD_ORDER_HIERARCHY
from .projected_apcm import (
    FixedDictionaryProjectedAPCM,
    ProjectedAPCMSettings,
    ProjectedAPCMStagePolicy,
    ProjectedAPCMStageFailure,
    ProjectedAPCMTrajectory,
    ProjectiveOuterEventGuard,
    ProjectiveOuterEventGuardSettings,
    integrate_projected_apcm_ssprk3,
    prepare_projected_apcm_initial_state,
    projected_apcm_partial_trajectory,
    projected_apcm_stage_failure,
    unpack_projected_apcm_state,
)

_RECORDED_OUTER_TARGET_OBJECTIVE = 0.01126712315
_RECORDED_OUTER_TARGET_DISPLACEMENT_FROM_CURRENT_ONLY = 0.5193246
_RECORDED_CWRMF_STRICT_ALL_COORDINATE_SCALAR_RMS = 3.24e-6


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("gate-reorder", "selector-audit", "rollout"),
        required=True,
    )
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--phonon-cutoff", type=int, default=16)
    parser.add_argument("--lambda-ep", type=float, default=1.5)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--drive", type=float, default=1.0)
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
        "--frontier-weight",
        type=float,
        nargs="+",
        default=[1.0],
        help=(
            "Declared weight of the fixed positive-definite frontier term in "
            "the strictly convex selector (--mode selector-audit).  More "
            "than one value solves the same preparation once per weight."
        ),
    )
    parser.add_argument(
        "--rollout-frontier-weight",
        type=float,
        default=0.0,
        help=(
            "Frontier weight used inside every triggered outer solve of the "
            "rollout; 0.0 is the recorded lexicographic selector."
        ),
    )
    parser.add_argument(
        "--reference-npz",
        type=Path,
        default=None,
        help=(
            "Gate-reorder artifact whose outer-aware target image the "
            "strictly convex selector is displaced against."
        ),
    )
    parser.add_argument("--final-time", type=float, default=0.05)
    parser.add_argument("--time-step", type=float, default=0.0025)
    parser.add_argument(
        "--event-gram-floor",
        type=float,
        default=ProjectiveOuterEventGuardSettings.event_gram_floor,
    )
    parser.add_argument(
        "--event-gram-hard-floor",
        type=float,
        default=ProjectiveOuterEventGuardSettings.event_gram_hard_floor,
    )
    parser.add_argument(
        "--event-coupling-threshold",
        type=float,
        default=ProjectiveOuterEventGuardSettings.event_coupling_threshold,
    )
    parser.add_argument(
        "--accept-first-provisional",
        action="store_true",
        help="Use the kept gate reorder inside every triggered outer solve.",
    )
    parser.add_argument(
        "--wall-budget-seconds",
        type=float,
        default=1200.0,
        help=(
            "Stop the rollout after this much autonomous wall time and "
            "report how far it reached and its per-stage cost."
        ),
    )
    parser.add_argument("--maximum-subdivisions", type=int, default=8)
    parser.add_argument(
        "--frontier-metric",
        choices=("identity", "readout_diagonal"),
        default="identity",
        help=(
            "Named scaling of the frontier term in the strictly convex "
            "selector; 'identity' is the recorded w I."
        ),
    )
    parser.add_argument(
        "--warm-refinement-passes",
        type=int,
        default=0,
        help=(
            "Further Clarabel passes on the displacement variable, warm "
            "started from the accumulated solution; 0 is the recorded path."
        ),
    )
    parser.add_argument(
        "--exploratory-acceptance-tolerance",
        type=float,
        default=None,
        help=(
            "Named exploratory acceptance threshold for the outer target "
            "solve only; the default None keeps the recorded 1e-6."
        ),
    )
    parser.add_argument(
        "--disable-stage-retraction",
        action="store_true",
        help=(
            "Named ablation: keep the declared completion and the envelope "
            "validation but never apply the conic stage retraction."
        ),
    )
    parser.add_argument(
        "--disable-quiet-containment",
        action="store_true",
        help=(
            "Named ablation: skip containment entirely on a stage the outer "
            "guard did not trigger."
        ),
    )
    parser.add_argument(
        "--carried-completion",
        action="store_true",
        help=(
            "Named hybrid: carry the previous stage's completion "
            "coordinates as the lift whenever they stay feasible at the "
            "trial point, instead of re-solving the declared completion."
        ),
    )
    parser.add_argument(
        "--carried-completion-margin-multiple",
        type=float,
        default=10.0,
        help=(
            "Interior the carried lift must keep before it is accepted, in "
            "units of the declared conic tolerance; 0.0 accepts any "
            "positive semidefinite carried lift inside the envelope."
        ),
    )
    parser.add_argument(
        "--cold-completion",
        action="store_true",
        help=(
            "Named control: solve the declared completion from the origin "
            "instead of from the carried frontier moments."
        ),
    )
    parser.add_argument(
        "--tolerate-stage-failure",
        action="store_true",
        help=(
            "Named option for an ablation expected to fail: stop at the last "
            "accepted step and report how far it reached instead of raising."
        ),
    )
    parser.add_argument(
        "--completion-retry-shifts",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Declared ladder of log-determinant shifts retried, in order, "
            "when the declared completion is not certified; empty is the "
            "committed path."
        ),
    )
    parser.add_argument(
        "--completion-retry-weights",
        type=float,
        nargs="+",
        default=[],
        help=(
            "Declared ladder of log-determinant weights retried after the "
            "shifts; empty is the committed path."
        ),
    )
    parser.add_argument(
        "--face-policy",
        choices=("refuse", "continuation", "inflated"),
        default=APCMProjectionSettings.face_policy,
        help=(
            "What the velocity projection does when a numerical face is not "
            "the same under the declared null-tolerance refinement; "
            "'refuse' is the committed path."
        ),
    )
    parser.add_argument(
        "--face-continuation-tolerance",
        type=float,
        default=APCMProjectionSettings.face_continuation_tolerance,
        help=(
            "Relative consistency gate a carried face must pass before it "
            "is reused under --face-policy continuation."
        ),
    )
    parser.add_argument(
        "--face-inflation-multiple",
        type=float,
        default=APCMProjectionSettings.face_inflation_multiple,
        help=(
            "Interior, in conic tolerances, of the inflated cone whose face "
            "is identified under --face-policy inflated; 100 is the declared "
            "stage margin of the coupled stage cone."
        ),
    )
    parser.add_argument(
        "--completion-logdet-shift",
        type=float,
        default=APCMExtensionSettings.logdet_shift,
        help=(
            "Declared positive-semidefinite inflation of the completion "
            "cone: the shift inside the log-determinant barrier of the "
            "proximal completion."
        ),
    )
    parser.add_argument(
        "--completion-logdet-weight",
        type=float,
        default=APCMExtensionSettings.logdet_weight,
        help=(
            "Weight of the log-determinant barrier against the proximal term "
            "of the declared completion; a smaller value is a proximal step "
            "of larger declared weight."
        ),
    )
    parser.add_argument(
        "--initial-state-file",
        type=Path,
        default=None,
        help=(
            "Continue the rollout from the last state of a stored "
            "trajectory .npz (keys 'times' and 'states'), instead of the "
            "exact-correlated preparation at t=0."
        ),
    )
    parser.add_argument(
        "--initial-time",
        type=float,
        default=None,
        help=(
            "Declared start time of the rollout; the default is 0.0 for a "
            "prepared start and the stored final time for a continued one."
        ),
    )
    parser.add_argument(
        "--disable-kpd-correction",
        action="store_true",
        help=(
            "Named ablation: switch off the exact K/P/D entrance correction "
            "of the electron-phonon correlation velocity."
        ),
    )
    return parser


def _entrance_keys(count: int | None) -> tuple:
    if count is None:
        return ENTRANCE_RELATIVE_MOMENT_KEYS
    if not 1 <= count <= len(ENTRANCE_RELATIVE_MOMENT_KEYS):
        raise ValueError("entrance key prefix is out of range")
    return ENTRANCE_RELATIVE_MOMENT_KEYS[:count]


def _preparation(
    parameters: DimerParameters,
    *,
    phonon_cutoff: int,
    active_keys: tuple,
) -> tuple[SymmetryReducedPositiveExtension, np.ndarray, dict]:
    """Contract the correlated preparation into the selector's lower moments.

    This is the same canonical-embedding contraction
    ``prepare_projected_apcm_initial_state`` performs, without the proximal
    log-det completion: the selector conditions on the lower moments only.
    """

    extension = SymmetryReducedPositiveExtension(active_keys=active_keys)
    exact = exact_holstein_joint_moment_initial_state(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=phonon_cutoff,
        canonical_embedding=True,
    )
    raw = matrix_state_to_raw_moment_coordinates(exact.matrix_state)
    _, moments = THIRD_ORDER_HIERARCHY.unpack(exact.hierarchy_coordinates)
    hidden = np.asarray(
        [moments[key] for key in active_keys], dtype=float
    )
    lower = state_lower_moments(raw, hidden, active_keys)
    return extension, raw, dict(lower)


def _stage_record(result: Any) -> dict[str, Any]:
    return {
        "status": result.status,
        "success": bool(result.success),
        "certified": bool(result.certified),
        "provisional": bool(result.provisional),
        "acceptance": result.acceptance,
        "objective_invariant_target_image_quadratic": float(result.objective),
        "minimum_scaled_eigenvalue": float(result.minimum_scaled_eigenvalue),
        "primal_residual": float(result.primal_residual),
        "dual_residual": float(result.dual_residual),
        "iterations": int(result.iterations),
        "independent_feasibility_residual": float(
            result.independent_feasibility_residual
        ),
        "stationarity_residual": float(result.stationarity_residual),
        "complementarity_residual": float(result.complementarity_residual),
        "relative_duality_gap": float(result.relative_duality_gap),
    }


def _run_outer_selector(
    extension: SymmetryReducedPositiveExtension,
    lower: dict,
    readout: Any,
    *,
    accept_first_provisional: bool,
    frontier_weight: float,
    frontier_metric: str = "identity",
    warm_refinement_passes: int = 0,
    exploratory_acceptance_tolerance: float | None = None,
    progress: Any,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
    label = (
        f"accept_first_provisional={accept_first_provisional}"
        f" frontier_weight={frontier_weight}"
        f" frontier_metric={frontier_metric}"
        f" warm_refinement_passes={warm_refinement_passes}"
        f" exploratory_acceptance={exploratory_acceptance_tolerance}"
    )
    progress(f"outer target solve start {label}")
    started = perf_counter()
    failure = ""
    try:
        selection = select_outer_feasible_projective_guard(
            extension,
            lower,
            readout,
            boxed=True,
            canonicalize=False,
            accept_first_provisional=accept_first_provisional,
            frontier_weight=frontier_weight,
            frontier_metric=frontier_metric,
            warm_refinement_passes=warm_refinement_passes,
            exploratory_acceptance_tolerance=(
                exploratory_acceptance_tolerance
            ),
        )
    except RuntimeError as error:  # declared restriction disagreement
        elapsed = perf_counter() - started
        progress(f"outer target solve failed {label} after {elapsed:.1f} s")
        return (
            {
                "accept_first_provisional": accept_first_provisional,
                "frontier_weight": frontier_weight,
                "frontier_metric": frontier_metric,
                "warm_refinement_passes": warm_refinement_passes,
                "exploratory_acceptance_tolerance": (
                    exploratory_acceptance_tolerance
                ),
                "wall_seconds": elapsed,
                "failure": str(error),
            },
            None,
            None,
        )
    elapsed = perf_counter() - started
    progress(
        f"outer target solve done {label} in {elapsed:.1f} s "
        f"objective={selection.outer_target_stage.objective:.11g} "
        f"success={selection.success}"
    )
    frontier_term = float(
        0.5
        * frontier_weight
        * float(
            np.dot(
                selection.outer_target_stage.standardized_values,
                selection.outer_target_stage.standardized_values,
            )
        )
    )
    record = {
        "accept_first_provisional": accept_first_provisional,
        "frontier_weight": frontier_weight,
        "frontier_metric": frontier_metric,
        "warm_refinement_passes": warm_refinement_passes,
        "exploratory_acceptance_tolerance": (
            exploratory_acceptance_tolerance
        ),
        "wall_seconds": elapsed,
        "failure": failure,
        "success": bool(selection.success),
        "witness_source": selection.witness_source,
        "canonical_lift": bool(selection.canonical_lift),
        "target_image_euclidean_norm": float(
            np.linalg.norm(selection.target_image)
        ),
        "frontier_quadratic_term": frontier_term,
        "current_minimum_scaled_eigenvalue": float(
            selection.current_minimum_scaled_eigenvalue
        ),
        "outer_target_stage": _stage_record(selection.outer_target_stage),
    }
    return (
        record,
        np.asarray(selection.target_image, dtype=float),
        np.asarray(selection.current_standardized_values, dtype=float),
    )


def _exact_archive_coordinates(
    parameters: DimerParameters,
    times: np.ndarray,
    *,
    phonon_cutoff: int,
    time_step: float,
) -> np.ndarray:
    """Exact closed scalar coordinates at ``times``, which may start late.

    The exact propagation is defined from the prepared ground state at
    ``t = 0``, so a rollout continued from a stored state is scored by
    propagating the exact reference from zero through a grid that ends on
    the rollout's own sample times and keeping the tail.
    """

    sample = np.asarray(times, dtype=float)
    start = float(sample[0])
    if start > 1e-15:
        leading = np.arange(
            0.0, start - 0.5 * time_step, time_step, dtype=float
        )
        sample = np.concatenate((leading, sample))
    exact = exact_holstein_driven_trajectory(
        parameters,
        sample_times=sample,
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
    return coordinates[coordinates.shape[0] - len(times):]


def _stage_failure_record(
    failure: ProjectedAPCMStageFailure | None,
) -> dict[str, Any] | None:
    """JSON view of the stage a failed rollout stopped at."""

    if failure is None:
        return None
    return {
        "stage": failure.stage,
        "time": failure.time,
        "message": failure.message,
        "lower_moment_count": len(failure.lower_moments),
        "lower_moments": {
            repr(key): float(value)
            for key, value in failure.lower_moments.items()
        },
        "state_euclidean_norm": float(np.linalg.norm(failure.state)),
    }


def _write_failure_artifacts(
    output_directory: Path,
    failure: ProjectedAPCMStageFailure | None,
) -> None:
    """Save the failing stage's state and lower moments for a rerun."""

    if failure is None:
        return
    keys = tuple(failure.lower_moments)
    np.savez_compressed(
        output_directory / "stage_failure.npz",
        stage=np.asarray(failure.stage),
        time=np.asarray(failure.time, dtype=float),
        state=np.asarray(failure.state, dtype=float),
        lower_moment_names=np.asarray([repr(key) for key in keys]),
        lower_moment_values=np.asarray(
            [float(failure.lower_moments[key]) for key in keys], dtype=float
        ),
        warm_frontier_names=np.asarray(
            []
            if failure.warm_frontier_moments is None
            else [repr(key) for key in failure.warm_frontier_moments]
        ),
        warm_frontier_values=np.asarray(
            []
            if failure.warm_frontier_moments is None
            else [
                float(value)
                for value in failure.warm_frontier_moments.values()
            ],
            dtype=float,
        ),
        message=np.asarray(failure.message),
    )


def _write(output_directory: Path, summary: dict[str, Any]) -> None:
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _peak_resident_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _archive_coordinates(
    states: np.ndarray,
    *,
    active_keys: tuple,
) -> np.ndarray:
    coordinates: list[np.ndarray] = []
    for state in states:
        raw, _ = unpack_projected_apcm_state(state, active_keys=active_keys)
        matrix_state = raw_moment_coordinates_to_matrix_state(raw)
        coordinates.append(
            matrix_state_to_closed_scalar_coordinates(matrix_state)
        )
    return np.asarray(coordinates, dtype=float)


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
    active_keys = _entrance_keys(args.entrance_keys)
    common = {
        "schema_version": 1,
        "classification": "exploratory_local_not_promoted",
        "mode": args.mode,
        "decisions_assumed": {
            "D1": "outer cone enforced as an event-triggered guard",
            "D2": (
                "strictly convex selector allowed as a named option; the "
                "lexicographic two-solve selector remains the default"
            ),
        },
        "parameters": {
            "hopping": parameters.hopping,
            "gamma": parameters.gamma,
            "lambda_ep": parameters.lambda_ep,
            "coupling": parameters.coupling,
            "drive_amplitude": parameters.drive_amplitude,
            "phonon_cutoff": args.phonon_cutoff,
        },
        "dictionary": {
            "entrance_keys": len(active_keys),
            "declared_entrance_keys": len(ENTRANCE_RELATIVE_MOMENT_KEYS),
        },
    }

    if args.mode in {"gate-reorder", "selector-audit"}:
        started = perf_counter()
        extension, _, lower = _preparation(
            parameters,
            phonon_cutoff=args.phonon_cutoff,
            active_keys=active_keys,
        )
        readout = compile_invariant_target_readout(extension)
        preparation_seconds = perf_counter() - started
        progress(
            f"prepared gram_rows={extension.dimension} "
            f"frontier={len(extension.frontier_keys)} "
            f"readout_rank={readout.rank} "
            f"in {preparation_seconds:.1f} s"
        )
        common["preparation"] = {
            "wall_seconds": preparation_seconds,
            "current_gram_rows": int(extension.dimension),
            "current_frontier_moments": int(len(extension.frontier_keys)),
            "invariant_target_readout_rank": int(readout.rank),
        }

    if args.mode == "gate-reorder":
        records = []
        images = {}
        values = {}
        for accept in (False, True):
            record, image, standardized = _run_outer_selector(
                extension,
                lower,
                readout,
                accept_first_provisional=accept,
                frontier_weight=0.0,
                frontier_metric=args.frontier_metric,
                warm_refinement_passes=args.warm_refinement_passes,
                exploratory_acceptance_tolerance=(
                    args.exploratory_acceptance_tolerance
                ),
                progress=progress,
            )
            records.append(record)
            if image is not None:
                images[str(accept)] = image
                values[str(accept)] = standardized
        payload = {}
        for key, image in images.items():
            payload[f"target_image_accept_{key}"] = image
            payload[f"standardized_values_accept_{key}"] = values[key]
        if "False" in images and "True" in images:
            payload["target_image_displacement_true_minus_false"] = np.asarray(
                images["True"] - images["False"]
            )
        np.savez_compressed(
            output_directory / "gate_reorder.npz", **payload
        )
        summary = dict(common)
        summary["selector"] = (
            "outer-aware invariant target image on the one-shell outer cone, "
            "minimum-norm lift skipped (canonicalize=False)"
        )
        summary["solves"] = records
        if "False" in images and "True" in images:
            summary["gate_reorder"] = {
                "target_image_euclidean_displacement": float(
                    np.linalg.norm(images["True"] - images["False"])
                ),
                "speedup": (
                    records[0]["wall_seconds"] / records[1]["wall_seconds"]
                    if records[1].get("wall_seconds")
                    else None
                ),
            }
        summary["recorded_reference"] = {
            "objective": _RECORDED_OUTER_TARGET_OBJECTIVE,
            "wall_seconds_without_then_with_gate_reorder": [500.0, 171.0],
            "source": "worklog:3267-3269, 3282-3288",
        }
        summary["peak_resident_bytes"] = _peak_resident_bytes()
        _write(output_directory, summary)
        return 0

    if args.mode == "selector-audit":
        records = []
        images = {}
        standardized_values = {}
        for weight in args.frontier_weight:
            record, image, standardized = _run_outer_selector(
                extension,
                lower,
                readout,
                accept_first_provisional=args.accept_first_provisional,
                frontier_weight=float(weight),
                frontier_metric=args.frontier_metric,
                warm_refinement_passes=args.warm_refinement_passes,
                exploratory_acceptance_tolerance=(
                    args.exploratory_acceptance_tolerance
                ),
                progress=progress,
            )
            records.append(record)
            if image is not None:
                images[repr(float(weight))] = image
                standardized_values[repr(float(weight))] = standardized
        reference_image = None
        if args.reference_npz is not None:
            reference = np.load(args.reference_npz.resolve())
            for key in (
                "target_image_accept_True",
                "target_image_accept_False",
            ):
                if key in reference:
                    reference_image = np.asarray(
                        reference[key], dtype=float
                    )
                    break
        payload: dict[str, np.ndarray] = {}
        for key, image in images.items():
            payload[f"target_image_weight_{key}"] = image
            payload[f"standardized_values_weight_{key}"] = (
                standardized_values[key]
            )
        if reference_image is not None:
            payload["reference_target_image"] = reference_image
        if payload:
            np.savez_compressed(
                output_directory / "selector_audit.npz", **payload
            )
        summary = dict(common)
        summary["selector"] = (
            "single strictly convex solve: invariant target image quadratic "
            "in the whitened metric plus a fixed positive-definite frontier "
            "term of declared weight, on the one-shell outer cone"
        )
        summary["solves"] = records
        if reference_image is not None:
            summary["displacement_against_outer_aware_witness"] = {
                key: {
                    "target_image_euclidean_displacement": float(
                        np.linalg.norm(image - reference_image)
                    ),
                }
                for key, image in images.items()
            } | {
                "reference_target_image_euclidean_norm": float(
                    np.linalg.norm(reference_image)
                ),
                "reference_source": str(args.reference_npz.resolve()),
            }
        summary["recorded_reference"] = {
            "objective": _RECORDED_OUTER_TARGET_OBJECTIVE,
            "outer_aware_witness_displacement_from_current_only_target": (
                _RECORDED_OUTER_TARGET_DISPLACEMENT_FROM_CURRENT_ONLY
            ),
            "source": "worklog:3253-3288",
        }
        summary["peak_resident_bytes"] = _peak_resident_bytes()
        _write(output_directory, summary)
        return 0

    guard_settings = ProjectiveOuterEventGuardSettings(
        event_gram_floor=args.event_gram_floor,
        event_gram_hard_floor=args.event_gram_hard_floor,
        event_coupling_threshold=args.event_coupling_threshold,
        accept_first_provisional=args.accept_first_provisional,
        frontier_weight=args.rollout_frontier_weight,
        frontier_metric=args.frontier_metric,
        warm_refinement_passes=args.warm_refinement_passes,
        exploratory_acceptance_tolerance=(
            args.exploratory_acceptance_tolerance
        ),
    )
    stage_policy = ProjectedAPCMStagePolicy(
        stage_retraction=not args.disable_stage_retraction,
        quiet_containment=not args.disable_quiet_containment,
        completion_start="cold" if args.cold_completion else "warm",
        carried_completion=bool(args.carried_completion),
        carried_completion_margin_multiple=(
            args.carried_completion_margin_multiple
        ),
        completion_retry_shifts=tuple(args.completion_retry_shifts),
        completion_retry_weights=tuple(args.completion_retry_weights),
    )
    model_settings = ProjectedAPCMSettings(
        include_k=not args.disable_kpd_correction,
        include_pauli=not args.disable_kpd_correction,
        include_opposite_spin=not args.disable_kpd_correction,
    )
    extension_settings = APCMExtensionSettings(
        logdet_weight=args.completion_logdet_weight,
        logdet_shift=args.completion_logdet_shift,
    )
    projection_settings = APCMProjectionSettings(
        face_policy=args.face_policy,
        face_continuation_tolerance=args.face_continuation_tolerance,
        face_inflation_multiple=args.face_inflation_multiple,
    )
    started = perf_counter()
    extension = SymmetryReducedPositiveExtension(
        active_keys=active_keys, settings=extension_settings
    )
    model = FixedDictionaryProjectedAPCM(
        parameters,
        extension=extension,
        geometry=SymmetryReducedAPCMGeometry(extension, projection_settings),
        settings=model_settings,
        stage_policy=stage_policy,
    )
    if args.initial_state_file is None:
        initialization = prepare_projected_apcm_initial_state(
            parameters,
            phonon_cutoff=args.phonon_cutoff,
            extension=extension,
        )
        initial_state = initialization.state
        initial_completion = initialization.completion
        initial_time = 0.0 if args.initial_time is None else args.initial_time
        continuation = None
    else:
        stored_path = args.initial_state_file.resolve()
        stored = np.load(stored_path)
        initial_state = np.asarray(stored["states"][-1], dtype=float)
        stored_time = float(np.asarray(stored["times"])[-1])
        initial_time = (
            stored_time if args.initial_time is None else args.initial_time
        )
        initial_completion = None
        continuation = {
            "initial_state_file": str(stored_path),
            "stored_final_time": stored_time,
            "initial_time": float(initial_time),
        }
        progress(
            f"continued from {stored_path} at t={initial_time:.6f} "
            f"state_dim={initial_state.size}"
        )
    guard = ProjectiveOuterEventGuard(model, settings=guard_settings)
    progress(
        f"prepared state_dim={len(model.state_names)} "
        f"gram_rows={extension.dimension} "
        f"frontier={len(extension.frontier_keys)} "
        + (
            "lambda_ext="
            f"{initial_completion.scaled_minimum_eigenvalue:.3e}"
            if initial_completion is not None
            else "lambda_ext=selected_at_start"
        )
    )
    deadline = perf_counter() + float(args.wall_budget_seconds)
    stage_failure: ProjectedAPCMStageFailure | None = None
    failure_message: str | None = None
    trajectory: ProjectedAPCMTrajectory | None
    try:
        trajectory = integrate_projected_apcm_ssprk3(
            model,
            initial_state,
            initial_completion=initial_completion,
            final_time=args.final_time,
            time_step=args.time_step,
            initial_time=float(initial_time),
            maximum_subdivisions=args.maximum_subdivisions,
            progress=progress,
            outer_guard=guard,
            deadline=deadline,
            tolerate_stage_failure=bool(args.tolerate_stage_failure),
        )
    except Exception as error:  # noqa: BLE001 - every failure path is saved
        trajectory = projected_apcm_partial_trajectory(error)
        stage_failure = projected_apcm_stage_failure(error)
        failure_message = f"{type(error).__name__}: {error}"
        progress(f"rollout_failed {failure_message}")
    autonomous_elapsed = perf_counter() - started
    if trajectory is None:
        summary = dict(common)
        summary["model"] = (
            "fixed_dictionary_projected_apcm_event_triggered_outer_guard"
        )
        summary["guard_settings"] = asdict(guard_settings)
        summary["stage_policy"] = asdict(stage_policy)
        summary["model_settings"] = asdict(model_settings)
        summary["projection_settings"] = asdict(projection_settings)
        summary["extension_settings"] = asdict(extension_settings)
        summary["face_policy_counters"] = dict(
            model.geometry.face_policy_counters
        )
        summary["stage_policy_counters"] = dict(model.stage_policy_counters)
        summary["retraction_completion_ladder_counters"] = dict(
            model.geometry.completion_ladder_counters
        )
        summary["continuation"] = continuation
        summary["failure"] = {
            "message": failure_message,
            "accepted_steps": 0,
            "stage": _stage_failure_record(stage_failure),
        }
        summary["integration"] = {
            "autonomous_wall_seconds": autonomous_elapsed,
            "final_time_requested": args.final_time,
            "initial_time": float(initial_time),
            "time_step": args.time_step,
            "success": False,
            "peak_resident_bytes": _peak_resident_bytes(),
        }
        _write_failure_artifacts(output_directory, stage_failure)
        _write(output_directory, summary)
        return 1
    approximate_coordinates = _archive_coordinates(
        trajectory.states, active_keys=model.active_keys
    )
    np.savez_compressed(
        output_directory / "autonomous_trajectory.npz",
        times=trajectory.times,
        states=trajectory.states,
        approximate_archive_coordinates=approximate_coordinates,
        base_minimum_eigenvalues=trajectory.base_minimum_eigenvalues,
        extension_minimum_eigenvalues=trajectory.extension_minimum_eigenvalues,
        stage_retractions=trajectory.stage_retractions,
        accepted_substeps=trajectory.accepted_substeps,
        outer_guard_triggered_stages=(
            trajectory.outer_guard_triggered_stages
            if trajectory.outer_guard_triggered_stages is not None
            else np.zeros(0, dtype=np.int64)
        ),
    )
    guard_records = guard.records
    (output_directory / "outer_guard_stages.jsonl").write_text(
        "".join(
            json.dumps(record, sort_keys=True) + "\n"
            for record in guard_records
        ),
        encoding="utf-8",
    )

    exact_started = perf_counter()
    if trajectory.times.size >= 2:
        exact_coordinates = _exact_archive_coordinates(
            parameters,
            trajectory.times,
            phonon_cutoff=args.phonon_cutoff,
            time_step=args.time_step,
        )
        metrics: dict[str, Any] | None = _accuracy_metrics(
            parameters, exact_coordinates, approximate_coordinates
        )
    else:
        exact_coordinates = np.zeros((0, approximate_coordinates.shape[1]))
        metrics = None
    exact_elapsed = perf_counter() - exact_started
    joint_minimum = float(
        min(
            np.linalg.eigvalsh(
                electron_phonon_moment_matrix(
                    raw_moment_coordinates_to_matrix_state(
                        unpack_projected_apcm_state(
                            state, active_keys=model.active_keys
                        )[0]
                    )
                )
            )[0]
            for state in trajectory.states
        )
    )
    stage_seconds = [
        float(record["wall_seconds"]) for record in guard_records
    ]
    triggered = [
        record for record in guard_records if record.get("triggered")
    ]
    summary = dict(common)
    summary["model"] = (
        "fixed_dictionary_projected_apcm_event_triggered_outer_guard"
    )
    summary["guard_settings"] = asdict(guard_settings)
    summary["stage_policy"] = asdict(stage_policy)
    summary["stage_policy_counters"] = dict(model.stage_policy_counters)
    summary["model_settings"] = asdict(model_settings)
    summary["projection_settings"] = asdict(projection_settings)
    summary["extension_settings"] = asdict(extension_settings)
    summary["face_policy_counters"] = dict(
        model.geometry.face_policy_counters
    )
    summary["retraction_completion_ladder_counters"] = dict(
        model.geometry.completion_ladder_counters
    )
    summary["continuation"] = continuation
    if failure_message is None and not trajectory.success:
        failure_message = trajectory.message
        stage_failure = trajectory.stage_failure
    if failure_message is not None:
        summary["failure"] = {
            "message": failure_message,
            "accepted_steps": int(trajectory.completed_steps),
            "last_accepted_time": float(trajectory.times[-1]),
            "stage": _stage_failure_record(stage_failure),
        }
        _write_failure_artifacts(output_directory, stage_failure)
    summary["integration"] = {
        "method": (
            "SSPRK(3,3) with containment before every RHS call and the "
            "event-triggered one-shell outer guard"
        ),
        "final_time_requested": args.final_time,
        "initial_time": float(initial_time),
        "last_time": float(trajectory.times[-1]),
        "time_step": args.time_step,
        "completed_steps": trajectory.completed_steps,
        "rhs_evaluations": trajectory.rhs_evaluations,
        "success": trajectory.success,
        "message": trajectory.message,
        "autonomous_wall_seconds": autonomous_elapsed,
        "exact_reference_wall_seconds": exact_elapsed,
        "wall_budget_seconds": args.wall_budget_seconds,
        "peak_resident_bytes": _peak_resident_bytes(),
    }
    summary["outer_guard"] = {
        "evaluated_stages": len(guard_records),
        "triggered_stages": len(triggered),
        "selector_solves": sum(
            1 for record in triggered if record.get("selector_wall_seconds")
        ),
        "guard_wall_seconds_total": float(sum(stage_seconds)),
        "maximum_stage_wall_seconds": (
            float(max(stage_seconds)) if stage_seconds else 0.0
        ),
        "mean_stage_wall_seconds": (
            float(np.mean(stage_seconds)) if stage_seconds else 0.0
        ),
        "minimum_endpoint_unshifted_gram_eigenvalue": (
            float(
                min(
                    float(record["endpoint_unshifted_minimum"])
                    for record in guard_records
                )
            )
            if guard_records
            else None
        ),
        "maximum_negative_mode_coupling_measure": (
            float(
                max(
                    float(record.get("coupling_measure", 0.0))
                    for record in guard_records
                )
            )
            if guard_records
            else None
        ),
    }
    summary["feasibility"] = {
        "minimum_unshifted_retained_joint_gram_eigenvalue": joint_minimum,
        "minimum_base_eigenvalue": float(
            np.min(trajectory.base_minimum_eigenvalues)
        ),
        "minimum_extension_scaled_eigenvalue": float(
            np.min(trajectory.extension_minimum_eigenvalues)
        ),
        "total_stage_retractions": int(np.sum(trajectory.stage_retractions)),
    }
    summary["accuracy"] = metrics
    summary["recorded_reference"] = {
        "cwrmf_strict_all_coordinate_scalar_rms_error_t0_to_0p05": (
            _RECORDED_CWRMF_STRICT_ALL_COORDINATE_SCALAR_RMS
        ),
        "source": "worklog:3796-3816",
    }
    np.savez_compressed(
        output_directory / "trajectory.npz",
        times=trajectory.times,
        states=trajectory.states,
        approximate_archive_coordinates=approximate_coordinates,
        exact_archive_coordinates=exact_coordinates,
        base_minimum_eigenvalues=trajectory.base_minimum_eigenvalues,
        extension_minimum_eigenvalues=trajectory.extension_minimum_eigenvalues,
    )
    _write(output_directory, summary)
    return 0 if (trajectory.success and failure_message is None) else 1


if __name__ == "__main__":
    raise SystemExit(main())
