#!/usr/bin/env python3
"""Benchmark-local AVQDS(T) row for the HH L=2 t=8 anchor.

This module implements an exact-assisted target-tangent AVQDS diagnostic inside
the benchmark surface only.  It reuses the fixed-pVQD benchmark's anchor
loading and exact interval propagation, plus the adaptive-pVQD replay-pool and
scaffold-epoch cost accounting seams.  It does not touch controller decisions
or production realtime paths.  The row is explicitly diagnostic and not
QPU-faithful.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_adaptive_pvqd_benchmark as adaptive
from pipelines.time_dynamics.legacy.hh_benchmarks import hh_fixed_pvqd_benchmark as base
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor


fixed = base.fixed
overlay = base.overlay

SCHEMA_VERSION = "hh_avqds_t_benchmark_v1"
RUN_SCHEMA_VERSION = "hh_avqds_t_run_v1"
DEFAULT_CASE_ID = base.DEFAULT_CASE_ID
METHOD_ID = "hh_td_avqds_t_pareto_lean_l2_exactv1"
METHOD_KIND = "avqds_t"
DECISION_MODE = base.DECISION_MODE
SEED_FAMILY = base.SEED_FAMILY
STATE_SCOPE = adaptive.STATE_SCOPE
HORIZON_SCOPE = adaptive.HORIZON_SCOPE
EPOCH_SOURCE_SCOPE = adaptive.EPOCH_SOURCE_SCOPE
CONTROLLER_STATE_SCOPE = base.CONTROLLER_STATE_SCOPE
CONTROLLER_SOURCE_SCOPE = base.CONTROLLER_SOURCE_SCOPE


@dataclass(frozen=True)
class AVQDSTBenchmarkCase:
    case_id: str
    controller_json: Path
    source_artifact_json: Path
    spec_name: str
    loader_mode: str
    generator_family: str
    fallback_family: str
    append_pool_family: str
    finite_difference_epsilon: float = 1.0e-5
    regularization_lambda: float = 1.0e-8
    pinv_relative_cutoff: float = 1.0e-10
    append_overlap_threshold: float = 0.9999
    append_min_overlap_gain: float = 1.0e-7
    append_candidate_limit: int | None = None
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class AVQDSTTangentStepResult:
    theta_runtime: np.ndarray
    final_state: np.ndarray
    initial_projection_loss: float
    final_projection_loss: float
    initial_overlap: float
    final_overlap: float
    delta_norm: float
    linear_solve_status: str
    linear_solve_count: int
    regularization_lambda: float
    finite_difference_epsilon: float
    pinv_relative_cutoff: float
    retained_rank: int
    parameter_count: int
    tangent_condition_estimate: float | None
    state_prep_count: int
    success: bool
    message: str


@dataclass(frozen=True)
class AppendCandidateTangentFit:
    candidate_pool_index: int
    candidate_label: str
    fit: AVQDSTTangentStepResult
    theta_runtime: np.ndarray
    terms: tuple[Any, ...]
    layout: Any
    executor: CompiledAnsatzExecutor
    new_runtime_indices: tuple[int, ...]


@dataclass(frozen=True)
class AVQDSTSimulationResult:
    method: str
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]
    final_state: np.ndarray
    final_terms: tuple[Any, ...]
    final_layout: Any
    final_theta_runtime: np.ndarray
    avqdst_steps: list[dict[str, Any]]
    append_events: list[dict[str, Any]]
    append_candidate_evaluations: list[dict[str, Any]]
    scaffold_epochs: list[dict[str, Any]]
    scaffold_snapshots: list[dict[str, Any]]
    epoch_compile_inputs: dict[str, dict[str, Any]]
    exact_reference_summary: dict[str, Any]


@dataclass(frozen=True)
class AVQDSTBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    status: str
    decision_mode: str
    diagnostic_exact_assisted: bool
    qpu_faithful: bool
    seed_family: str
    controller_json: str | None
    source_artifact_json: str | None
    drive_enabled: bool | None
    t_final: float | None
    num_times: int | None
    final_energy_total: float | None
    final_energy_total_exact: float | None
    final_abs_energy_total_error: float | None
    mean_abs_energy_total_error: float | None
    max_abs_energy_total_error: float | None
    fidelity_min: float | None
    append_events_total: int | None
    append_candidate_evaluations_total: int | None
    unique_scaffold_count: int | None
    final_logical_block_count: int | None
    final_runtime_parameter_count: int | None
    avqdst_linear_solve_total: int | None
    avqdst_step_count: int | None
    avqdst_state_prep_total: int | None
    state_at_time_scope: str
    state_at_time_basis: str | None
    state_at_time_2q: int | None
    state_at_time_depth: int | None
    state_at_time_size: int | None
    full_horizon_scope: str
    full_horizon_basis: str | None
    full_horizon_intervals: int | None
    full_horizon_horizon_2q: int | None
    full_horizon_depth_serial: int | None
    controller_state_scope: str
    controller_state_basis: str | None
    controller_state_2q: int | None
    controller_state_depth: int | None
    controller_state_size: int | None
    backend_name: str | None
    seed_transpiler: int | None
    optimization_level: int | None
    preferred_fake_backends: tuple[str, ...]
    artifact_run_json: str | None
    artifact_manifest_json: str | None
    artifact_rows_json: str | None
    artifact_summary_json: str | None
    exact_assisted_contract: str = "diagnostic_exact_assisted_not_qpu_faithful"


@dataclass(frozen=True)
class _CaseRunRecord:
    case: AVQDSTBenchmarkCase
    spec: fixed.FixedManifoldRunSpec
    run_json: Path
    run_artifact: Mapping[str, Any]
    row: dict[str, Any]
    compile_defaults: Mapping[str, Any]


@dataclass(frozen=True)
class _ScaffoldCompileRecord:
    signature: str
    interval_count: int
    cost: overlay.CircuitCostRow
    raw_rows: list[dict[str, Any]]


"Built Math: AVQDS(T) row = exact-step target propagation plus one finite-difference target-tangent linear solve per interval; append tests use the same tangent update and horizon cost is a serial sum over active scaffold epochs."
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return base._jsonable(asdict(value))
    return base._jsonable(value)


def _write_json(path: Path, payload: Any) -> Path:
    return base._write_json(path, payload)


def _maybe_mapping(value: Any) -> Mapping[str, Any]:
    return base._maybe_mapping(value)


def _maybe_float(value: Any) -> float | None:
    return base._maybe_float(value)


def _maybe_int(value: Any) -> int | None:
    return base._maybe_int(value)


def _maybe_bool(value: Any) -> bool | None:
    return base._maybe_bool(value)


def _as_optional_str(value: Any) -> str | None:
    return base._as_optional_str(value)


def _required_finite_float(value: Any, *, field: str) -> float:
    return base._required_finite_float(value, field=field)


def _required_int(value: Any, *, field: str) -> int:
    return base._required_int(value, field=field)


def _compiled_executor_for_terms(terms: Sequence[Any], layout: Any) -> CompiledAnsatzExecutor:
    return adaptive._compiled_executor_for_terms(terms, layout)


def _prepare_scaffold_state(
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_runtime: np.ndarray | Sequence[float],
) -> np.ndarray:
    return adaptive._prepare_scaffold_state(executor, psi_ref, theta_runtime)


def _append_candidate_pool(replay_context: Any) -> tuple[list[Any], dict[str, Any]]:
    try:
        return adaptive._append_candidate_pool(replay_context)
    except ValueError as exc:
        raise ValueError(str(exc).replace("adaptive pVQD", "AVQDS(T)")) from exc


_extend_scaffold_append_end = adaptive._extend_scaffold_append_end
_candidate_label = adaptive._candidate_label
_candidate_pool_indices = adaptive._candidate_pool_indices
_layout_signature_payload = adaptive._layout_signature_payload
_scaffold_signature = adaptive._scaffold_signature
_scaffold_snapshot_payload = adaptive._scaffold_snapshot_payload
_remember_scaffold_compile_input = adaptive._remember_scaffold_compile_input
_summarize_epochs = adaptive._summarize_epochs


def default_cases() -> tuple[AVQDSTBenchmarkCase, ...]:
    return (
        AVQDSTBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_artifact_json=fixed.DEFAULT_PARETO_ARTIFACT,
            spec_name=SEED_FAMILY,
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
            append_pool_family="match_replay",
        ),
    )


def _case_by_id(case_id: str) -> AVQDSTBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown AVQDS(T) benchmark case_id={case_id!r}; known cases: {known}")


def _run_spec_for_case(case: AVQDSTBenchmarkCase) -> fixed.FixedManifoldRunSpec:
    return fixed.FixedManifoldRunSpec(
        name=str(case.spec_name),
        artifact_json=Path(case.source_artifact_json),
        loader_mode=str(case.loader_mode),
        generator_family=str(case.generator_family),
        fallback_family=str(case.fallback_family),
        append_pool_family=str(case.append_pool_family),
    )


def _compile_defaults_for_case(
    case: AVQDSTBenchmarkCase,
    source_payload: Mapping[str, Any],
) -> dict[str, Any]:
    defaults = dict(overlay._source_compile_defaults(source_payload))
    preferred = case.preferred_fake_backends or tuple(defaults.get("preferred_fake_backends", ()))
    return {
        "backend_name": str(case.backend_name if case.backend_name is not None else defaults.get("backend_name")),
        "seed_transpiler": int(
            case.seed_transpiler if case.seed_transpiler is not None else defaults.get("seed_transpiler")
        ),
        "optimization_level": int(
            case.optimization_level if case.optimization_level is not None else defaults.get("optimization_level")
        ),
        "preferred_fake_backends": tuple(str(x) for x in preferred),
    }


def _finite_difference_tangents(
    prepare_state: Callable[[np.ndarray], np.ndarray],
    theta_runtime: np.ndarray | Sequence[float],
    *,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build centered finite-difference horizontal tangent columns."""

    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError(f"finite_difference_epsilon must be positive and finite; got {epsilon!r}")
    theta = np.asarray(theta_runtime, dtype=float).reshape(-1)
    psi0 = base._normalize_state(prepare_state(theta))
    state_prep_count = 1
    dim = int(psi0.size)
    if int(theta.size) == 0:
        return psi0, np.zeros((dim, 0), dtype=complex), int(state_prep_count)

    columns: list[np.ndarray] = []
    for idx in range(int(theta.size)):
        shift = np.zeros_like(theta)
        shift[int(idx)] = eps
        psi_plus = base._normalize_state(prepare_state(theta + shift))
        psi_minus = base._normalize_state(prepare_state(theta - shift))
        state_prep_count += 2
        tangent = np.asarray((psi_plus - psi_minus) / (2.0 * eps), dtype=complex).reshape(-1)
        tangent = tangent - psi0 * np.vdot(psi0, tangent)
        columns.append(tangent)
    return psi0, np.column_stack(columns), int(state_prep_count)


def _phase_align_target(current_state: np.ndarray, target_state: np.ndarray) -> np.ndarray:
    current = base._normalize_state(current_state)
    target = base._normalize_state(target_state)
    inner = np.vdot(current, target)
    magnitude = float(abs(inner))
    if magnitude <= 0.0 or not math.isfinite(magnitude):
        return target
    return base._normalize_state(target * np.exp(-1.0j * float(np.angle(inner))))


def _solve_target_tangent_step(
    *,
    prepare_state: Callable[[np.ndarray], np.ndarray],
    theta_start: np.ndarray | Sequence[float],
    target_state: np.ndarray | Sequence[complex],
    finite_difference_epsilon: float,
    regularization_lambda: float,
    pinv_relative_cutoff: float,
) -> AVQDSTTangentStepResult:
    """Apply one deterministic target-tangent/McLachlan linear-solve update."""

    theta_init = np.asarray(theta_start, dtype=float).reshape(-1)
    reg = float(regularization_lambda)
    cutoff = float(pinv_relative_cutoff)
    if not math.isfinite(reg) or reg < 0.0:
        raise ValueError(f"regularization_lambda must be non-negative and finite; got {regularization_lambda!r}")
    if not math.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError(f"pinv_relative_cutoff must be non-negative and finite; got {pinv_relative_cutoff!r}")

    target = base._normalize_state(np.asarray(target_state, dtype=complex).reshape(-1))
    current, tangents, state_prep_count = _finite_difference_tangents(
        prepare_state,
        theta_init,
        epsilon=float(finite_difference_epsilon),
    )
    initial_loss, initial_overlap = base._projection_loss_for_state(current, target)
    parameter_count = int(theta_init.size)
    if parameter_count == 0:
        return AVQDSTTangentStepResult(
            theta_runtime=np.asarray(theta_init, dtype=float),
            final_state=np.asarray(current, dtype=complex),
            initial_projection_loss=float(initial_loss),
            final_projection_loss=float(initial_loss),
            initial_overlap=float(initial_overlap),
            final_overlap=float(initial_overlap),
            delta_norm=0.0,
            linear_solve_status="no_runtime_parameters",
            linear_solve_count=0,
            regularization_lambda=float(reg),
            finite_difference_epsilon=float(finite_difference_epsilon),
            pinv_relative_cutoff=float(cutoff),
            retained_rank=0,
            parameter_count=0,
            tangent_condition_estimate=None,
            state_prep_count=int(state_prep_count),
            success=True,
            message="scaffold has no runtime parameters; tangent update skipped",
        )

    aligned_target = _phase_align_target(current, target)
    displacement = np.asarray(aligned_target - current, dtype=complex).reshape(-1)
    displacement = displacement - current * np.vdot(current, displacement)
    gram = np.real(np.conjugate(tangents).T @ tangents)
    gram = 0.5 * (gram + gram.T)
    force = np.real(np.conjugate(tangents).T @ displacement)
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(force)):
        raise ValueError("non-finite finite-difference tangent system")

    evals, evecs = np.linalg.eigh(gram)
    max_abs_eval = float(np.max(np.abs(evals))) if evals.size else 0.0
    negative_tol = max(1.0e-12, 1.0e-9 * max_abs_eval)
    min_eval = float(np.min(evals)) if evals.size else 0.0
    if min_eval < -negative_tol:
        raise ValueError(f"finite-difference tangent metric has negative eigenvalue {min_eval:.6e}")
    evals = np.maximum(np.asarray(evals, dtype=float), 0.0)
    max_eval = float(np.max(evals)) if evals.size else 0.0
    if max_eval <= 0.0:
        delta = np.zeros_like(theta_init)
        retained_rank = 0
        condition_estimate = None
        status = "unsupported_zero_tangent"
    else:
        threshold = float(cutoff) * float(max_eval)
        retained = np.asarray(evals >= threshold, dtype=bool)
        if not bool(np.any(retained)):
            delta = np.zeros_like(theta_init)
            retained_rank = 0
            condition_estimate = None
            status = "unsupported_no_retained_tangent_modes"
        else:
            v = np.asarray(evecs[:, retained], dtype=float)
            s = np.asarray(evals[retained], dtype=float)
            projected_force = np.asarray(v.T @ force, dtype=float).reshape(-1)
            delta = np.asarray(v @ (projected_force / (s + float(reg))), dtype=float).reshape(-1)
            retained_rank = int(s.size)
            condition_estimate = float(max_eval / max(float(np.min(s)), 1.0e-300))
            status = "regularized_spectral_solve" if reg > 0.0 else "spectral_pseudoinverse_solve"

    if not np.all(np.isfinite(delta)):
        raise ValueError(f"AVQDS(T) tangent solve produced non-finite delta; status={status}")
    theta_next = np.asarray(theta_init + delta, dtype=float).reshape(-1)
    final_state = base._normalize_state(prepare_state(theta_next))
    state_prep_count += 1
    final_loss, final_overlap = base._projection_loss_for_state(final_state, target)
    success = bool(math.isfinite(final_overlap) and math.isfinite(final_loss))
    if not success:
        raise ValueError("AVQDS(T) tangent step produced non-finite projection diagnostics")

    return AVQDSTTangentStepResult(
        theta_runtime=np.asarray(theta_next, dtype=float),
        final_state=np.asarray(final_state, dtype=complex),
        initial_projection_loss=float(initial_loss),
        final_projection_loss=float(final_loss),
        initial_overlap=float(initial_overlap),
        final_overlap=float(final_overlap),
        delta_norm=float(np.linalg.norm(delta)),
        linear_solve_status=str(status),
        linear_solve_count=1,
        regularization_lambda=float(reg),
        finite_difference_epsilon=float(finite_difference_epsilon),
        pinv_relative_cutoff=float(cutoff),
        retained_rank=int(retained_rank),
        parameter_count=int(parameter_count),
        tangent_condition_estimate=condition_estimate,
        state_prep_count=int(state_prep_count),
        success=bool(success),
        message="one finite-difference target-tangent linear solve",
    )


def _evaluate_append_candidates_for_avqdst(
    *,
    current_terms: Sequence[Any],
    current_layout: Any,
    current_theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    target_state: np.ndarray,
    append_pool: Sequence[Any],
    candidate_indices: Sequence[int],
    finite_difference_epsilon: float,
    regularization_lambda: float,
    pinv_relative_cutoff: float,
) -> list[AppendCandidateTangentFit]:
    out: list[AppendCandidateTangentFit] = []
    for pool_index in candidate_indices:
        term = append_pool[int(pool_index)]
        terms, layout, executor, theta_seed, new_runtime_indices = _extend_scaffold_append_end(
            current_terms=current_terms,
            current_layout=current_layout,
            current_theta_runtime=current_theta_runtime,
            candidate_term=term,
        )

        def prepare(theta_vec: np.ndarray, *, _executor: CompiledAnsatzExecutor = executor) -> np.ndarray:
            return _prepare_scaffold_state(_executor, psi_ref, np.asarray(theta_vec, dtype=float).reshape(-1))

        fit = _solve_target_tangent_step(
            prepare_state=prepare,
            theta_start=np.asarray(theta_seed, dtype=float).reshape(-1),
            target_state=target_state,
            finite_difference_epsilon=float(finite_difference_epsilon),
            regularization_lambda=float(regularization_lambda),
            pinv_relative_cutoff=float(pinv_relative_cutoff),
        )
        out.append(
            AppendCandidateTangentFit(
                candidate_pool_index=int(pool_index),
                candidate_label=_candidate_label(term, int(pool_index)),
                fit=fit,
                theta_runtime=np.asarray(fit.theta_runtime, dtype=float).reshape(-1),
                terms=tuple(terms),
                layout=layout,
                executor=executor,
                new_runtime_indices=tuple(int(idx) for idx in new_runtime_indices),
            )
        )
    return out


def _select_append_candidate(
    *,
    base_fit: AVQDSTTangentStepResult,
    candidate_fits: Sequence[AppendCandidateTangentFit],
    min_overlap_gain: float,
) -> tuple[AppendCandidateTangentFit | None, float]:
    if not candidate_fits:
        return None, 0.0
    best = max(candidate_fits, key=lambda item: float(item.fit.final_overlap))
    gain = float(best.fit.final_overlap) - float(base_fit.final_overlap)
    if gain >= float(min_overlap_gain):
        return best, float(gain)
    return None, float(gain)


def _summarize_trajectory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty AVQDS(T) trajectory")
    energies = [_required_finite_float(row.get("energy_total"), field="energy_total") for row in rows]
    exact_values = [_maybe_float(row.get("energy_total_exact")) for row in rows]
    errors = [_maybe_float(row.get("abs_energy_total_error")) for row in rows]
    finite_errors = [float(x) for x in errors if x is not None]
    fidelities = [_maybe_float(row.get("fidelity_exact")) for row in rows]
    finite_fidelities = [float(x) for x in fidelities if x is not None]
    solve_total = sum(int(row.get("avqdst_linear_solve_count", 0) or 0) for row in rows)
    state_prep_total = sum(int(row.get("avqdst_state_prep_count", 0) or 0) for row in rows)
    step_count = sum(1 for row in rows if int(row.get("checkpoint_index", 0)) > 0)
    return {
        "row_count": int(len(rows)),
        "final_energy_total": float(energies[-1]),
        "final_energy_total_exact": exact_values[-1] if exact_values else None,
        "final_abs_energy_total_error": None if not finite_errors else errors[-1],
        "mean_abs_energy_total_error": None if not finite_errors else float(sum(finite_errors) / len(finite_errors)),
        "max_abs_energy_total_error": None if not finite_errors else float(max(finite_errors)),
        "fidelity_min": None if not finite_fidelities else float(min(finite_fidelities)),
        "fidelity_final": None if not finite_fidelities else fidelities[-1],
        "avqdst_linear_solve_total": int(solve_total),
        "avqdst_step_count": int(step_count),
        "avqdst_state_prep_total": int(state_prep_total),
    }


def _simulate_avqds_t(
    *,
    case: AVQDSTBenchmarkCase,
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    exact_energy_total: Sequence[float] | None,
    observation_physical_times: Sequence[float],
    exact_steps_multiplier: int,
) -> AVQDSTSimulationResult:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if int(times_arr.size) < 2:
        raise ValueError("AVQDS(T) requires at least two time points")
    obs_physical = np.asarray(observation_physical_times, dtype=float).reshape(-1)
    if int(obs_physical.size) != int(times_arr.size):
        raise ValueError("observation_physical_times must match source time grid")
    exact_arr = None if exact_energy_total is None else np.asarray(exact_energy_total, dtype=float).reshape(-1)
    if exact_arr is not None and int(exact_arr.size) != int(times_arr.size):
        raise ValueError("exact_energy_total must match source time grid")

    loaded = context.loaded
    replay_context = loaded.replay_context
    append_pool, append_pool_meta = _append_candidate_pool(replay_context)
    psi_ref = np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1)
    current_terms: tuple[Any, ...] = tuple(replay_context.replay_terms)
    current_layout = replay_context.base_layout
    current_executor = _compiled_executor_for_terms(current_terms, current_layout)
    theta = np.asarray(replay_context.adapt_theta_runtime, dtype=float).reshape(-1)
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))

    current_state = _prepare_scaffold_state(current_executor, psi_ref, theta)
    exact_states = base._build_exact_reference_states(
        psi_initial=np.asarray(context.psi_initial, dtype=complex),
        times=times_arr,
        hmat_static=np.asarray(context.hmat, dtype=complex),
        drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
        drive_t0=float(drive_t0),
        drive_time_sampling=str(drive_sampling),
        exact_steps_multiplier=int(exact_steps_multiplier),
        nq=int(context.nq),
    )
    if len(exact_states) != int(times_arr.size):
        raise ValueError("exact reference state count does not match source time grid")

    trajectory: list[dict[str, Any]] = []
    avqdst_steps: list[dict[str, Any]] = []
    append_events: list[dict[str, Any]] = []
    append_candidate_evaluations: list[dict[str, Any]] = []
    used_append_indices: set[int] = set()
    active_signatures: list[str] = []
    snapshot_layouts: dict[str, Any] = {}
    epoch_compile_inputs: dict[str, dict[str, Any]] = {}

    def _remember_snapshot(signature: str, terms: Sequence[Any], layout: Any, theta_runtime: np.ndarray) -> None:
        _remember_scaffold_compile_input(
            snapshot_layouts=snapshot_layouts,
            epoch_compile_inputs=epoch_compile_inputs,
            signature=str(signature),
            terms=terms,
            layout=layout,
            theta_runtime=theta_runtime,
        )

    initial_signature = _scaffold_signature(current_layout)
    _remember_snapshot(initial_signature, current_terms, current_layout, theta)

    def _append_row(
        idx: int,
        state: np.ndarray,
        *,
        fit: AVQDSTTangentStepResult | None,
        scaffold_signature: str,
        layout: Any,
        interval_linear_solve_count: int = 0,
        interval_state_prep_count: int = 0,
        append_accepted: bool | None = None,
        append_candidate_evaluations_count: int = 0,
    ) -> None:
        hmat_total = overlay._hmat_total_at_observation(
            hmat_static=np.asarray(context.hmat, dtype=complex),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            physical_time=float(obs_physical[int(idx)]),
            nq=int(context.nq),
        )
        energy = base._expectation_hamiltonian(state, np.asarray(hmat_total, dtype=complex))
        exact_state = base._normalize_state(np.asarray(exact_states[int(idx)], dtype=complex).reshape(-1))
        exact_energy = (
            float(exact_arr[int(idx)])
            if exact_arr is not None
            else base._expectation_hamiltonian(exact_state, np.asarray(hmat_total, dtype=complex))
        )
        fidelity = float(abs(np.vdot(exact_state, base._normalize_state(state))) ** 2)
        row: dict[str, Any] = {
            "checkpoint_index": int(idx),
            "time": float(times_arr[int(idx)]),
            "physical_time": float(obs_physical[int(idx)]),
            "method": METHOD_ID,
            "method_kind": METHOD_KIND,
            "energy_total": float(energy),
            "energy_total_exact": float(exact_energy),
            "abs_energy_total_error": float(abs(float(energy) - float(exact_energy))),
            "fidelity_exact": float(fidelity),
            "state_norm": float(np.linalg.norm(state)),
            "runtime_parameter_count": int(layout.runtime_parameter_count),
            "logical_block_count": int(layout.logical_parameter_count),
            "scaffold_signature": str(scaffold_signature),
            "append_accepted": append_accepted,
            "append_candidate_evaluations": int(append_candidate_evaluations_count),
            "avqdst_linear_solve_count": int(interval_linear_solve_count),
            "avqdst_state_prep_count": int(interval_state_prep_count),
        }
        if fit is None:
            row.update(
                {
                    "avqdst_step_index": None,
                    "projection_loss_initial": None,
                    "projection_loss_final": None,
                    "projection_overlap_initial": None,
                    "projection_overlap_final": None,
                    "linear_solve_status": None,
                    "linear_solve_success": None,
                    "tangent_condition_estimate": None,
                    "tangent_retained_rank": None,
                    "tangent_delta_norm": None,
                }
            )
        else:
            row.update(
                {
                    "avqdst_step_index": int(idx) - 1,
                    "projection_loss_initial": float(fit.initial_projection_loss),
                    "projection_loss_final": float(fit.final_projection_loss),
                    "projection_overlap_initial": float(fit.initial_overlap),
                    "projection_overlap_final": float(fit.final_overlap),
                    "linear_solve_status": str(fit.linear_solve_status),
                    "linear_solve_success": bool(fit.success),
                    "tangent_condition_estimate": fit.tangent_condition_estimate,
                    "tangent_retained_rank": int(fit.retained_rank),
                    "tangent_delta_norm": float(fit.delta_norm),
                }
            )
        trajectory.append(row)

    _append_row(
        0,
        current_state,
        fit=None,
        scaffold_signature=initial_signature,
        layout=current_layout,
        append_accepted=None,
    )

    for k in range(int(times_arr.size) - 1):
        theta_start = np.asarray(theta, dtype=float).reshape(-1)
        target = base._apply_exact_interval(
            current_state,
            times=times_arr,
            interval_index=int(k),
            hmat_static=np.asarray(context.hmat, dtype=complex),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_sampling),
            exact_steps_multiplier=int(exact_steps_multiplier),
            nq=int(context.nq),
        )

        def prepare_current(theta_vec: np.ndarray) -> np.ndarray:
            return _prepare_scaffold_state(
                current_executor,
                psi_ref,
                np.asarray(theta_vec, dtype=float).reshape(-1),
            )

        base_fit = _solve_target_tangent_step(
            prepare_state=prepare_current,
            theta_start=theta_start,
            target_state=target,
            finite_difference_epsilon=float(case.finite_difference_epsilon),
            regularization_lambda=float(case.regularization_lambda),
            pinv_relative_cutoff=float(case.pinv_relative_cutoff),
        )
        accepted_fit = base_fit
        accepted_terms = current_terms
        accepted_layout = current_layout
        accepted_executor = current_executor
        accepted_theta = np.asarray(base_fit.theta_runtime, dtype=float).reshape(-1)
        accepted_state = np.asarray(base_fit.final_state, dtype=complex).reshape(-1)
        accepted_append: AppendCandidateTangentFit | None = None
        append_gain = 0.0
        candidate_fits: list[AppendCandidateTangentFit] = []
        append_triggered = bool(float(base_fit.final_overlap) < float(case.append_overlap_threshold))

        if append_triggered:
            candidate_indices = _candidate_pool_indices(
                len(append_pool),
                used_indices=used_append_indices,
                candidate_limit=case.append_candidate_limit,
            )
            candidate_fits = _evaluate_append_candidates_for_avqdst(
                current_terms=current_terms,
                current_layout=current_layout,
                current_theta_runtime=theta_start,
                psi_ref=psi_ref,
                target_state=target,
                append_pool=append_pool,
                candidate_indices=candidate_indices,
                finite_difference_epsilon=float(case.finite_difference_epsilon),
                regularization_lambda=float(case.regularization_lambda),
                pinv_relative_cutoff=float(case.pinv_relative_cutoff),
            )
            accepted_append, append_gain = _select_append_candidate(
                base_fit=base_fit,
                candidate_fits=candidate_fits,
                min_overlap_gain=float(case.append_min_overlap_gain),
            )
            if accepted_append is not None:
                accepted_fit = accepted_append.fit
                accepted_terms = accepted_append.terms
                accepted_layout = accepted_append.layout
                accepted_executor = accepted_append.executor
                accepted_theta = np.asarray(accepted_append.theta_runtime, dtype=float).reshape(-1)
                accepted_state = np.asarray(accepted_append.fit.final_state, dtype=complex).reshape(-1)
                used_append_indices.add(int(accepted_append.candidate_pool_index))

        for item in candidate_fits:
            append_candidate_evaluations.append(
                {
                    "interval_index": int(k),
                    "candidate_pool_index": int(item.candidate_pool_index),
                    "candidate_label": str(item.candidate_label),
                    "projection_overlap_initial": float(item.fit.initial_overlap),
                    "projection_overlap_final": float(item.fit.final_overlap),
                    "overlap_gain_vs_base": float(item.fit.final_overlap) - float(base_fit.final_overlap),
                    "linear_solve_status": str(item.fit.linear_solve_status),
                    "regularization_lambda": float(item.fit.regularization_lambda),
                    "tangent_condition_estimate": item.fit.tangent_condition_estimate,
                    "retained_rank": int(item.fit.retained_rank),
                    "state_prep_count": int(item.fit.state_prep_count),
                    "logical_block_count": int(item.layout.logical_parameter_count),
                    "runtime_parameter_count": int(item.layout.runtime_parameter_count),
                }
            )

        interval_linear_solve_count = int(base_fit.linear_solve_count) + sum(
            int(item.fit.linear_solve_count) for item in candidate_fits
        )
        interval_state_prep_count = int(base_fit.state_prep_count) + sum(
            int(item.fit.state_prep_count) for item in candidate_fits
        )
        current_terms = tuple(accepted_terms)
        current_layout = accepted_layout
        current_executor = accepted_executor
        theta = np.asarray(accepted_theta, dtype=float).reshape(-1)
        current_state = base._normalize_state(accepted_state)
        active_signature = _scaffold_signature(current_layout)
        active_signatures.append(str(active_signature))
        _remember_snapshot(active_signature, current_terms, current_layout, theta)

        append_accepted = accepted_append is not None
        if append_accepted and accepted_append is not None:
            event = {
                "interval_index": int(k),
                "time_start": float(times_arr[int(k)]),
                "time_stop": float(times_arr[int(k) + 1]),
                "candidate_pool_index": int(accepted_append.candidate_pool_index),
                "candidate_label": str(accepted_append.candidate_label),
                "overlap_before_append": float(base_fit.final_overlap),
                "overlap_after_append": float(accepted_append.fit.final_overlap),
                "overlap_gain": float(append_gain),
                "base_linear_solve_status": str(base_fit.linear_solve_status),
                "candidate_linear_solve_status": str(accepted_append.fit.linear_solve_status),
                "candidate_evaluations": int(len(candidate_fits)),
                "logical_block_count": int(current_layout.logical_parameter_count),
                "runtime_parameter_count": int(current_layout.runtime_parameter_count),
                "new_runtime_indices": [int(idx) for idx in accepted_append.new_runtime_indices],
                "scaffold_signature": str(active_signature),
            }
            append_events.append(event)

        step_payload = {
            "interval_index": int(k),
            "time_start": float(times_arr[int(k)]),
            "time_stop": float(times_arr[int(k) + 1]),
            "dt": float(times_arr[int(k) + 1] - times_arr[int(k)]),
            "finite_difference_epsilon": float(case.finite_difference_epsilon),
            "regularization_lambda": float(case.regularization_lambda),
            "pinv_relative_cutoff": float(case.pinv_relative_cutoff),
            "append_overlap_threshold": float(case.append_overlap_threshold),
            "append_min_overlap_gain": float(case.append_min_overlap_gain),
            "append_triggered": bool(append_triggered),
            "append_accepted": bool(append_accepted),
            "append_candidate_evaluations": int(len(candidate_fits)),
            "append_best_overlap_gain": float(append_gain),
            "accepted_candidate_pool_index": (
                None if accepted_append is None else int(accepted_append.candidate_pool_index)
            ),
            "accepted_candidate_label": (
                None if accepted_append is None else str(accepted_append.candidate_label)
            ),
            "success": bool(accepted_fit.success),
            "linear_solve_status": str(accepted_fit.linear_solve_status),
            "message": str(accepted_fit.message),
            "linear_solve_count": int(interval_linear_solve_count),
            "base_linear_solve_count": int(base_fit.linear_solve_count),
            "state_prep_count": int(interval_state_prep_count),
            "base_state_prep_count": int(base_fit.state_prep_count),
            "projection_loss_initial": float(accepted_fit.initial_projection_loss),
            "projection_loss_final": float(accepted_fit.final_projection_loss),
            "projection_overlap_initial": float(accepted_fit.initial_overlap),
            "projection_overlap_final": float(accepted_fit.final_overlap),
            "base_projection_overlap_final": float(base_fit.final_overlap),
            "tangent_delta_norm": float(accepted_fit.delta_norm),
            "tangent_condition_estimate": accepted_fit.tangent_condition_estimate,
            "tangent_retained_rank": int(accepted_fit.retained_rank),
            "tangent_parameter_count": int(accepted_fit.parameter_count),
            "scaffold_signature": str(active_signature),
            "logical_block_count": int(current_layout.logical_parameter_count),
            "runtime_parameter_count": int(current_layout.runtime_parameter_count),
        }
        avqdst_steps.append(step_payload)
        _append_row(
            k + 1,
            current_state,
            fit=accepted_fit,
            scaffold_signature=str(active_signature),
            layout=current_layout,
            interval_linear_solve_count=int(interval_linear_solve_count),
            interval_state_prep_count=int(interval_state_prep_count),
            append_accepted=bool(append_accepted),
            append_candidate_evaluations_count=int(len(candidate_fits)),
        )

    summary = _summarize_trajectory(trajectory)
    scaffold_epochs = _summarize_epochs(
        active_signatures=active_signatures,
        snapshot_layouts=snapshot_layouts,
    )
    active_unique = {str(row["signature"]) for row in scaffold_epochs if int(row.get("interval_count", 0)) > 0}
    summary.update(
        {
            "append_events_total": int(len(append_events)),
            "append_candidate_evaluations_total": int(
                sum(int(step.get("append_candidate_evaluations", 0) or 0) for step in avqdst_steps)
            ),
            "unique_scaffold_count": int(len(active_unique)),
            "trajectory_unique_scaffold_count": int(len(snapshot_layouts)),
            "final_logical_block_count": int(current_layout.logical_parameter_count),
            "final_runtime_parameter_count": int(current_layout.runtime_parameter_count),
            "append_pool_size": int(len(append_pool)),
            "append_candidate_pool_complete": bool(append_pool_meta.get("candidate_pool_complete", False)),
            "append_pool_source": append_pool_meta.get("append_pool_source", None),
        }
    )
    snapshots_payload = [
        _scaffold_snapshot_payload(signature=sig, layout=layout)
        for sig, layout in snapshot_layouts.items()
    ]
    return AVQDSTSimulationResult(
        method=METHOD_ID,
        trajectory=trajectory,
        summary=summary,
        final_state=np.asarray(current_state, dtype=complex).reshape(-1),
        final_terms=tuple(current_terms),
        final_layout=current_layout,
        final_theta_runtime=np.asarray(theta, dtype=float).reshape(-1),
        avqdst_steps=avqdst_steps,
        append_events=append_events,
        append_candidate_evaluations=append_candidate_evaluations,
        scaffold_epochs=scaffold_epochs,
        scaffold_snapshots=snapshots_payload,
        epoch_compile_inputs=epoch_compile_inputs,
        exact_reference_summary={
            "state_count": int(len(exact_states)),
            "reference_policy": "benchmark-local exact interval propagation for fidelity diagnostics",
            "exact_steps_multiplier": int(exact_steps_multiplier),
        },
    )


def _compile_scaffold(
    *,
    context: overlay.RebuiltOverlayContext,
    terms: Sequence[Any],
    layout: Any,
    theta_runtime: np.ndarray | Sequence[float],
    compile_defaults: Mapping[str, Any],
    scope: str,
) -> tuple[overlay.CircuitCostRow, list[dict[str, Any]]]:
    del terms  # layout carries the compiled generator/rotation structure.
    scaffold_circuit = overlay.build_ansatz_circuit(
        layout,
        np.asarray(theta_runtime, dtype=float).reshape(-1),
        int(context.nq),
        ref_state=np.asarray(context.loaded.replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    cost, raw_rows = overlay._compile_one_circuit_cost(
        method=METHOD_KIND,
        order=None,
        scope=str(scope),
        trotter_steps=None,
        includes_seed_prep=True,
        circuit=scaffold_circuit,
        backend_name=str(compile_defaults["backend_name"]),
        preferred_fake_backends=tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        seed_transpiler=int(compile_defaults["seed_transpiler"]),
        optimization_level=int(compile_defaults["optimization_level"]),
    )
    base._require_finite_cost(cost, label=f"AVQDS(T) scaffold compile row scope={scope}")
    return cost, [dict(row) for row in raw_rows if isinstance(row, Mapping)]


def _compile_epoch_scaffolds(
    *,
    context: overlay.RebuiltOverlayContext,
    simulation: AVQDSTSimulationResult,
    compile_defaults: Mapping[str, Any],
) -> list[_ScaffoldCompileRecord]:
    records: list[_ScaffoldCompileRecord] = []
    for epoch in simulation.scaffold_epochs:
        signature = str(epoch["signature"])
        interval_count = int(epoch.get("interval_count", 0) or 0)
        if interval_count <= 0:
            continue
        inputs = simulation.epoch_compile_inputs[signature]
        cost, raw_rows = _compile_scaffold(
            context=context,
            terms=inputs["terms"],
            layout=inputs["layout"],
            theta_runtime=inputs["theta_runtime"],
            compile_defaults=compile_defaults,
            scope=EPOCH_SOURCE_SCOPE,
        )
        records.append(
            _ScaffoldCompileRecord(
                signature=signature,
                interval_count=int(interval_count),
                cost=cost,
                raw_rows=raw_rows,
            )
        )
    return records


def _hardware_report_rows(
    *,
    final_state_cost: overlay.CircuitCostRow,
    controller_cost: overlay.CircuitCostRow,
    epoch_records: Sequence[_ScaffoldCompileRecord],
    intervals: int,
) -> list[dict[str, Any]]:
    state_2q = _required_int(final_state_cost.compiled_count_2q, field="state_at_time_2q")
    state_depth = _required_int(final_state_cost.compiled_depth, field="state_at_time_depth")
    state_size = _required_int(final_state_cost.compiled_size, field="state_at_time_size")
    controller_2q = _required_int(controller_cost.compiled_count_2q, field="controller_state_2q")
    controller_depth = _required_int(controller_cost.compiled_depth, field="controller_state_depth")
    controller_size = _required_int(controller_cost.compiled_size, field="controller_state_size")
    horizon_2q = 0
    horizon_depth = 0
    horizon_size = 0
    epoch_payloads: list[dict[str, Any]] = []
    for record in epoch_records:
        count = int(record.interval_count)
        twoq = _required_int(record.cost.compiled_count_2q, field=f"epoch[{record.signature}].compiled_count_2q")
        depth = _required_int(record.cost.compiled_depth, field=f"epoch[{record.signature}].compiled_depth")
        size = _required_int(record.cost.compiled_size, field=f"epoch[{record.signature}].compiled_size")
        horizon_2q += int(twoq) * int(count)
        horizon_depth += int(depth) * int(count)
        horizon_size += int(size) * int(count)
        epoch_payloads.append(
            {
                "scaffold_signature": str(record.signature),
                "interval_count": int(count),
                "compiled_count_2q": int(twoq),
                "compiled_depth": int(depth),
                "compiled_size": int(size),
                "horizon_count_2q": int(twoq) * int(count),
                "horizon_depth_serial": int(depth) * int(count),
            }
        )
    return [
        {
            "method": "controller",
            "group": "state_at_time",
            "scope": CONTROLLER_STATE_SCOPE,
            "basis": "controller state-at-time compile reference",
            "compiled_count_2q": int(controller_2q),
            "compiled_depth": int(controller_depth),
            "compiled_size": int(controller_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": controller_cost.scope,
        },
        {
            "method": METHOD_KIND,
            "group": "state_at_time",
            "scope": STATE_SCOPE,
            "basis": "AVQDS(T) final scaffold",
            "compiled_count_2q": int(state_2q),
            "compiled_depth": int(state_depth),
            "compiled_size": int(state_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": final_state_cost.scope,
        },
        {
            "method": METHOD_KIND,
            "group": "horizon",
            "scope": HORIZON_SCOPE,
            "basis": f"serial sum of {len(epoch_records)} active AVQDS(T) scaffold epochs over {int(intervals)} intervals; each epoch uses latest-active theta for its scaffold signature",
            "compiled_count_2q": int(horizon_2q),
            "compiled_depth": int(horizon_depth),
            "compiled_size": int(horizon_size),
            "horizon_count_2q": int(horizon_2q),
            "horizon_depth_serial": int(horizon_depth),
            "source_scope": EPOCH_SOURCE_SCOPE,
            "intervals": int(intervals),
            "unique_scaffold_count": int(len(epoch_records)),
            "compile_representative_policy": "latest_active_theta_for_scaffold_signature",
            "epoch_costs": epoch_payloads,
        },
    ]


def _required_report_row(
    hardware_report_rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    scope: str,
) -> Mapping[str, Any]:
    matches = [
        row
        for row in hardware_report_rows
        if str(row.get("method", "")) == str(method) and str(row.get("scope", "")) == str(scope)
    ]
    if not matches:
        raise ValueError(f"required hardware report row missing: method={method!r} scope={scope!r}")
    if len(matches) > 1:
        raise ValueError(f"required hardware report row ambiguous: method={method!r} scope={scope!r}")
    return matches[0]


def _parameter_manifest(
    *,
    case: AVQDSTBenchmarkCase,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    compile_defaults: Mapping[str, Any],
    output_dir: Path,
    exact_steps_multiplier: int,
) -> dict[str, Any]:
    settings = _maybe_mapping(context.loaded.payload.get("settings", {}))
    drive_cfg = _maybe_mapping(source_payload.get("drive_config"))
    reference = _maybe_mapping(source_payload.get("reference"))
    return {
        "model_family_name": "Hubbard-Holstein",
        "problem": str(settings.get("problem", "hh")),
        "L": _maybe_int(settings.get("L", getattr(context.loaded.cfg, "L", None))),
        "boundary": _as_optional_str(settings.get("boundary", getattr(context.loaded.cfg, "boundary", None))),
        "ordering": _as_optional_str(settings.get("ordering", getattr(context.loaded.cfg, "ordering", None))),
        "boson_encoding": _as_optional_str(settings.get("boson_encoding", getattr(context.loaded.cfg, "boson_encoding", None))),
        "ansatz_types": "ADAPT seed prep; AVQDS(T) one-step target-tangent update with benchmark-local one-generator append tests",
        "t": _maybe_float(settings.get("t")),
        "U": _maybe_float(settings.get("u", settings.get("U"))),
        "dv": _maybe_float(settings.get("dv")),
        "omega0": _maybe_float(settings.get("omega0")),
        "g_ep": _maybe_float(settings.get("g_ep")),
        "n_ph_max": _maybe_int(settings.get("n_ph_max")),
        "drive_enabled": _maybe_bool(drive_cfg.get("enabled", False)),
        "drive_A": drive_cfg.get("drive_A"),
        "drive_omega": drive_cfg.get("drive_omega"),
        "drive_tbar": drive_cfg.get("drive_tbar"),
        "drive_phi": drive_cfg.get("drive_phi"),
        "drive_pattern": drive_cfg.get("drive_pattern"),
        "drive_time_sampling": drive_cfg.get("drive_time_sampling"),
        "drive_t0": drive_cfg.get("drive_t0"),
        "t_final": float(times[-1]),
        "num_times": int(times.size),
        "finite_difference_epsilon": float(case.finite_difference_epsilon),
        "regularization_lambda": float(case.regularization_lambda),
        "pinv_relative_cutoff": float(case.pinv_relative_cutoff),
        "append_overlap_threshold": float(case.append_overlap_threshold),
        "append_min_overlap_gain": float(case.append_min_overlap_gain),
        "append_candidate_limit": case.append_candidate_limit,
        "exact_reference_method": _as_optional_str(reference.get("reference_method")),
        "exact_steps_multiplier": int(exact_steps_multiplier),
        "decision_mode": DECISION_MODE,
        "diagnostic_exact_assisted": True,
        "qpu_faithful": False,
        "compile_backend": str(compile_defaults["backend_name"]),
        "compile_seed_transpiler": int(compile_defaults["seed_transpiler"]),
        "compile_optimization_level": int(compile_defaults["optimization_level"]),
        "compile_preferred_fake_backends": tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        "controller_json": str(case.controller_json),
        "seed_artifact_json": str(case.source_artifact_json),
        "output_dir": str(output_dir),
    }


def _row_from_run_artifact(
    payload: Mapping[str, Any],
    *,
    case: AVQDSTBenchmarkCase,
    artifact_run_json: Path | str | None,
    artifact_manifest_json: Path | str | None = None,
    artifact_rows_json: Path | str | None = None,
    artifact_summary_json: Path | str | None = None,
    preferred_fake_backends: Sequence[str] | None = None,
) -> dict[str, Any]:
    summary = _maybe_mapping(payload.get("summary"))
    manifest = _maybe_mapping(payload.get("parameter_manifest"))
    source = _maybe_mapping(payload.get("source"))
    hardware_rows = [dict(row) for row in payload.get("hardware_report_rows", []) if isinstance(row, Mapping)]
    state_cost = _required_report_row(hardware_rows, method=METHOD_KIND, scope=STATE_SCOPE)
    full_cost = _required_report_row(hardware_rows, method=METHOD_KIND, scope=HORIZON_SCOPE)
    controller_cost = _required_report_row(hardware_rows, method="controller", scope=CONTROLLER_STATE_SCOPE)
    intervals = _maybe_int(full_cost.get("intervals"))
    if intervals is None:
        num_times = _maybe_int(manifest.get("num_times"))
        intervals = None if num_times is None else max(int(num_times) - 1, 0)

    row = AVQDSTBenchmarkRow(
        case_id=str(case.case_id),
        method_id=METHOD_ID,
        method_kind=METHOD_KIND,
        status="ok",
        decision_mode=DECISION_MODE,
        diagnostic_exact_assisted=True,
        qpu_faithful=False,
        seed_family=SEED_FAMILY,
        controller_json=_as_optional_str(manifest.get("controller_json") or source.get("controller_json")),
        source_artifact_json=_as_optional_str(manifest.get("seed_artifact_json") or source.get("artifact_json")),
        drive_enabled=_maybe_bool(manifest.get("drive_enabled")),
        t_final=_maybe_float(manifest.get("t_final")),
        num_times=_maybe_int(manifest.get("num_times")),
        final_energy_total=_required_finite_float(summary.get("final_energy_total"), field="summary.final_energy_total"),
        final_energy_total_exact=_required_finite_float(summary.get("final_energy_total_exact"), field="summary.final_energy_total_exact"),
        final_abs_energy_total_error=_required_finite_float(summary.get("final_abs_energy_total_error"), field="summary.final_abs_energy_total_error"),
        mean_abs_energy_total_error=_required_finite_float(summary.get("mean_abs_energy_total_error"), field="summary.mean_abs_energy_total_error"),
        max_abs_energy_total_error=_required_finite_float(summary.get("max_abs_energy_total_error"), field="summary.max_abs_energy_total_error"),
        fidelity_min=_required_finite_float(summary.get("fidelity_min"), field="summary.fidelity_min"),
        append_events_total=_required_int(summary.get("append_events_total"), field="summary.append_events_total"),
        append_candidate_evaluations_total=_required_int(
            summary.get("append_candidate_evaluations_total"),
            field="summary.append_candidate_evaluations_total",
        ),
        unique_scaffold_count=_required_int(summary.get("unique_scaffold_count"), field="summary.unique_scaffold_count"),
        final_logical_block_count=_required_int(summary.get("final_logical_block_count"), field="summary.final_logical_block_count"),
        final_runtime_parameter_count=_required_int(summary.get("final_runtime_parameter_count"), field="summary.final_runtime_parameter_count"),
        avqdst_linear_solve_total=_required_int(summary.get("avqdst_linear_solve_total"), field="summary.avqdst_linear_solve_total"),
        avqdst_step_count=_required_int(summary.get("avqdst_step_count"), field="summary.avqdst_step_count"),
        avqdst_state_prep_total=_required_int(summary.get("avqdst_state_prep_total"), field="summary.avqdst_state_prep_total"),
        state_at_time_scope=str(state_cost.get("scope")),
        state_at_time_basis=_as_optional_str(state_cost.get("basis")),
        state_at_time_2q=_required_int(state_cost.get("compiled_count_2q"), field="state_at_time_2q"),
        state_at_time_depth=_required_int(state_cost.get("compiled_depth"), field="state_at_time_depth"),
        state_at_time_size=_required_int(state_cost.get("compiled_size"), field="state_at_time_size"),
        full_horizon_scope=str(full_cost.get("scope")),
        full_horizon_basis=_as_optional_str(full_cost.get("basis")),
        full_horizon_intervals=(None if intervals is None else int(intervals)),
        full_horizon_horizon_2q=_required_int(full_cost.get("horizon_count_2q"), field="full_horizon_horizon_2q"),
        full_horizon_depth_serial=_required_int(full_cost.get("horizon_depth_serial"), field="full_horizon_depth_serial"),
        controller_state_scope=str(controller_cost.get("scope")),
        controller_state_basis=_as_optional_str(controller_cost.get("basis")),
        controller_state_2q=_required_int(controller_cost.get("compiled_count_2q"), field="controller_state_2q"),
        controller_state_depth=_required_int(controller_cost.get("compiled_depth"), field="controller_state_depth"),
        controller_state_size=_required_int(controller_cost.get("compiled_size"), field="controller_state_size"),
        backend_name=_as_optional_str(manifest.get("compile_backend")),
        seed_transpiler=_maybe_int(manifest.get("compile_seed_transpiler")),
        optimization_level=_maybe_int(manifest.get("compile_optimization_level")),
        preferred_fake_backends=tuple(str(x) for x in (preferred_fake_backends or ())),
        artifact_run_json=_as_optional_str(artifact_run_json),
        artifact_manifest_json=_as_optional_str(artifact_manifest_json),
        artifact_rows_json=_as_optional_str(artifact_rows_json),
        artifact_summary_json=_as_optional_str(artifact_summary_json),
    )
    out = _jsonable(row)
    out["source_controller_run_tag"] = _as_optional_str(source.get("run_tag"))
    out["exact_steps_multiplier"] = _maybe_int(manifest.get("exact_steps_multiplier"))
    out["finite_difference_epsilon"] = _maybe_float(manifest.get("finite_difference_epsilon"))
    out["regularization_lambda"] = _maybe_float(manifest.get("regularization_lambda"))
    out["pinv_relative_cutoff"] = _maybe_float(manifest.get("pinv_relative_cutoff"))
    out["append_overlap_threshold"] = _maybe_float(manifest.get("append_overlap_threshold"))
    out["append_min_overlap_gain"] = _maybe_float(manifest.get("append_min_overlap_gain"))
    return out


def _build_run_artifact(
    *,
    case: AVQDSTBenchmarkCase,
    spec: fixed.FixedManifoldRunSpec,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    simulation: AVQDSTSimulationResult,
    final_state_cost: overlay.CircuitCostRow,
    controller_cost: overlay.CircuitCostRow,
    epoch_compile_records: Sequence[_ScaffoldCompileRecord],
    final_raw_compile_rows: Sequence[Mapping[str, Any]],
    compile_defaults: Mapping[str, Any],
    output_dir: Path,
    command: str,
    exact_steps_multiplier: int,
) -> dict[str, Any]:
    intervals = max(int(times.size) - 1, 0)
    summary = dict(simulation.summary)
    hardware_report_rows = _hardware_report_rows(
        final_state_cost=final_state_cost,
        controller_cost=controller_cost,
        epoch_records=epoch_compile_records,
        intervals=int(intervals),
    )
    parameter_manifest = _parameter_manifest(
        case=case,
        source_payload=source_payload,
        context=context,
        times=times,
        compile_defaults=compile_defaults,
        output_dir=output_dir,
        exact_steps_multiplier=int(exact_steps_multiplier),
    )
    return _jsonable(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "generated_utc": _now_utc(),
            "case_id": str(case.case_id),
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "command": command,
            "source": {
                "controller_json": str(case.controller_json),
                "run_tag": source_payload.get("run_tag"),
                "artifact_json": str(case.source_artifact_json),
                "controller_artifact_json": _as_optional_str(source_payload.get("artifact_json")),
            },
            "resolved_run_spec": _jsonable(spec),
            "loader": dict(context.loaded.loader_summary),
            "parameter_manifest": parameter_manifest,
            "config": {
                "finite_difference_epsilon": float(case.finite_difference_epsilon),
                "regularization_lambda": float(case.regularization_lambda),
                "pinv_relative_cutoff": float(case.pinv_relative_cutoff),
                "append_overlap_threshold": float(case.append_overlap_threshold),
                "append_min_overlap_gain": float(case.append_min_overlap_gain),
                "append_candidate_limit": case.append_candidate_limit,
                "target_policy": "exact_interval_evolution_of_current_avqds_t_state",
                "step_policy": "one finite-difference target-tangent linear solve; no iterative pVQD projection optimizer",
                "append_policy": "one replay-pool operator appended at end; run the same tangent update; accept only fixed minimum overlap gain",
                "compile_backend_name": str(compile_defaults["backend_name"]),
                "compile_seed_transpiler": int(compile_defaults["seed_transpiler"]),
                "compile_optimization_level": int(compile_defaults["optimization_level"]),
                "compile_preferred_fake_backends": tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
            },
            "contract": {
                "decision_mode": DECISION_MODE,
                "diagnostic_exact_assisted": True,
                "qpu_faithful": False,
                "controller_decisions_modified": False,
                "controller_paths_called": False,
                "avqds_t_target_depends_on_exact_interval_propagation": True,
                "append_candidate_source": "benchmark-local replay append pool",
                "compile_cost_policy": "compile final scaffold for state-at-time; full horizon is serial sum over active unique scaffold epochs using latest-active theta per scaffold signature",
                "controller_reference_policy": "fail_closed_required_source_compile_reference",
            },
            "trajectory": simulation.trajectory,
            "summary": summary,
            "avqdst_steps": simulation.avqdst_steps,
            "append_events": simulation.append_events,
            "append_candidate_evaluations": simulation.append_candidate_evaluations,
            "scaffold_epochs": simulation.scaffold_epochs,
            "scaffold_snapshots": simulation.scaffold_snapshots,
            "exact_reference_summary": simulation.exact_reference_summary,
            "hardware_report_rows": hardware_report_rows,
            "circuit_costs": [_jsonable(final_state_cost), _jsonable(controller_cost)],
            "raw_compile_rows": {
                "avqds_t_final_state_scaffold": list(final_raw_compile_rows),
                "avqds_t_epoch_scaffolds": [
                    {
                        "scaffold_signature": record.signature,
                        "interval_count": int(record.interval_count),
                        "selected": _jsonable(record.cost),
                        "raw_rows": list(record.raw_rows),
                    }
                    for record in epoch_compile_records
                ],
                "controller": [{"selected": _jsonable(controller_cost), "raw_rows": []}],
                "time_grid": {
                    "points": int(times.size),
                    "intervals": int(intervals),
                    "exact_steps_multiplier": int(exact_steps_multiplier),
                },
            },
        }
    )


def _run_case(
    case: AVQDSTBenchmarkCase,
    *,
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> _CaseRunRecord:
    source_payload = overlay._load_source_payload(Path(case.controller_json))
    source_rows = overlay._state_sample_rows(source_payload)
    times = overlay._source_times(source_payload, source_rows)
    physical_times = overlay._source_physical_times(
        source_rows,
        fallback_drive_t0=float(_maybe_mapping(source_payload.get("drive_config")).get("drive_t0", 0.0)),
    )
    exact_energy = [
        _required_finite_float(row.get("energy_total_exact"), field=f"source_rows[{idx}].energy_total_exact")
        for idx, row in enumerate(source_rows)
    ]
    compile_defaults = _compile_defaults_for_case(case, source_payload)
    controller_cost = base._required_controller_cost_row(source_payload)
    spec = _run_spec_for_case(case)
    context = base._build_fixed_context(spec=spec, case=case, source_payload=source_payload)  # type: ignore[arg-type]
    exact_steps_multiplier = base._exact_steps_multiplier_from_source(source_payload)
    simulation = _simulate_avqds_t(
        case=case,
        context=context,
        times=times,
        exact_energy_total=exact_energy,
        observation_physical_times=physical_times,
        exact_steps_multiplier=int(exact_steps_multiplier),
    )
    final_state_cost, final_raw_compile_rows = _compile_scaffold(
        context=context,
        terms=simulation.final_terms,
        layout=simulation.final_layout,
        theta_runtime=simulation.final_theta_runtime,
        compile_defaults=compile_defaults,
        scope=STATE_SCOPE,
    )
    epoch_compile_records = _compile_epoch_scaffolds(
        context=context,
        simulation=simulation,
        compile_defaults=compile_defaults,
    )
    run_artifact = _build_run_artifact(
        case=case,
        spec=spec,
        source_payload=source_payload,
        context=context,
        times=times,
        simulation=simulation,
        final_state_cost=final_state_cost,
        controller_cost=controller_cost,
        epoch_compile_records=epoch_compile_records,
        final_raw_compile_rows=final_raw_compile_rows,
        compile_defaults=compile_defaults,
        output_dir=output_dir,
        command=command,
        exact_steps_multiplier=int(exact_steps_multiplier),
    )
    run_json = Path(output_dir) / "runs" / f"{case.case_id}.json"
    _write_json(run_json, run_artifact)
    row = _row_from_run_artifact(
        run_artifact,
        case=case,
        artifact_run_json=run_json,
        artifact_manifest_json=manifest_json,
        artifact_rows_json=rows_json,
        artifact_summary_json=summary_json,
        preferred_fake_backends=compile_defaults["preferred_fake_backends"],
    )
    return _CaseRunRecord(
        case=case,
        spec=spec,
        run_json=run_json,
        run_artifact=run_artifact,
        row=row,
        compile_defaults=compile_defaults,
    )


def _manifest_payload(
    *,
    records: Sequence[_CaseRunRecord],
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_avqds_t_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "decision_mode": DECISION_MODE,
            "diagnostic_exact_assisted": True,
            "qpu_faithful": False,
            "default_case_id": DEFAULT_CASE_ID,
            "seed_family": SEED_FAMILY,
            "target_policy": "exact_interval_evolution_of_current_avqds_t_state",
            "step_policy": "single finite-difference target-tangent linear solve per interval",
            "append_policy": "one replay-pool operator append/tangent update under overlap threshold and fixed minimum gain",
            "compile_cost_policy": "final scaffold state-at-time; serial sum over active unique scaffold epochs using latest-active theta per scaffold signature for full horizon",
            "controller_reference_policy": "fail_closed_required_source_compile_reference",
        },
        "command": command,
        "output_dir": str(output_dir),
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
            "runs_dir": str(Path(output_dir) / "runs"),
        },
        "cases": [
            {
                "case": _jsonable(record.case),
                "resolved_run_spec": _jsonable(record.spec),
                "compile_defaults": _jsonable(record.compile_defaults),
                "artifact_run_json": str(record.run_json),
            }
            for record in records
        ],
    }


def _summary_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    manifest_json: Path,
    rows_json: Path,
    summary_json: Path,
    command: str,
) -> dict[str, Any]:
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_avqds_t_time_dynamics",
        "command": command,
        "output_dir": str(output_dir),
        "row_count": int(len(rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "case_ids": [str(row.get("case_id")) for row in rows],
        "method_ids": [str(row.get("method_id")) for row in rows],
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
        },
        "key_metrics": [
            {
                "case_id": row.get("case_id"),
                "method_id": row.get("method_id"),
                "final_abs_energy_total_error": row.get("final_abs_energy_total_error"),
                "mean_abs_energy_total_error": row.get("mean_abs_energy_total_error"),
                "max_abs_energy_total_error": row.get("max_abs_energy_total_error"),
                "fidelity_min": row.get("fidelity_min"),
                "append_events_total": row.get("append_events_total"),
                "append_candidate_evaluations_total": row.get("append_candidate_evaluations_total"),
                "unique_scaffold_count": row.get("unique_scaffold_count"),
                "final_logical_block_count": row.get("final_logical_block_count"),
                "final_runtime_parameter_count": row.get("final_runtime_parameter_count"),
                "avqdst_linear_solve_total": row.get("avqdst_linear_solve_total"),
                "avqdst_step_count": row.get("avqdst_step_count"),
                "state_at_time_2q": row.get("state_at_time_2q"),
                "state_at_time_depth": row.get("state_at_time_depth"),
                "full_horizon_horizon_2q": row.get("full_horizon_horizon_2q"),
                "full_horizon_depth_serial": row.get("full_horizon_depth_serial"),
                "controller_state_2q": row.get("controller_state_2q"),
                "controller_state_depth": row.get("controller_state_depth"),
            }
            for row in rows
        ],
    }


def run_benchmark(
    *,
    cases: Sequence[AVQDSTBenchmarkCase],
    output_dir: Path,
    command: str = "",
) -> dict[str, Any]:
    root = Path(output_dir)
    manifest_json = root / "manifest.json"
    rows_json = root / "rows.json"
    summary_json = root / "summary.json"
    root.mkdir(parents=True, exist_ok=True)

    records = [
        _run_case(
            case,
            output_dir=root,
            manifest_json=manifest_json,
            rows_json=rows_json,
            summary_json=summary_json,
            command=command,
        )
        for case in cases
    ]
    rows = [dict(record.row) for record in records]
    manifest = _manifest_payload(
        records=records,
        output_dir=root,
        manifest_json=manifest_json,
        rows_json=rows_json,
        summary_json=summary_json,
        command=command,
    )
    summary = _summary_payload(
        rows=rows,
        output_dir=root,
        manifest_json=manifest_json,
        rows_json=rows_json,
        summary_json=summary_json,
        command=command,
    )

    _write_json(manifest_json, manifest)
    _write_json(rows_json, rows)
    _write_json(summary_json, summary)
    return {
        "manifest": manifest,
        "rows": rows,
        "summary": summary,
        "paths": {
            "manifest_json": str(manifest_json),
            "rows_json": str(rows_json),
            "summary_json": str(summary_json),
        },
    }


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    return base._parse_string_tuple(raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-local HH L2 t=8 AVQDS(T) row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-artifact-json", type=Path, default=None)
    parser.add_argument("--finite-difference-epsilon", type=float, default=None)
    parser.add_argument("--regularization-lambda", type=float, default=None)
    parser.add_argument("--pinv-relative-cutoff", type=float, default=None)
    parser.add_argument("--append-overlap-threshold", type=float, default=None)
    parser.add_argument("--append-min-overlap-gain", type=float, default=None)
    parser.add_argument("--append-candidate-limit", type=int, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> AVQDSTBenchmarkCase:
    case = _case_by_id(str(args.case_id))
    preferred = _parse_string_tuple(args.compile_preferred_fake_backends)
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
        source_artifact_json=(
            Path(args.source_artifact_json)
            if args.source_artifact_json is not None
            else case.source_artifact_json
        ),
        finite_difference_epsilon=(
            float(args.finite_difference_epsilon)
            if args.finite_difference_epsilon is not None
            else case.finite_difference_epsilon
        ),
        regularization_lambda=(
            float(args.regularization_lambda)
            if args.regularization_lambda is not None
            else case.regularization_lambda
        ),
        pinv_relative_cutoff=(
            float(args.pinv_relative_cutoff)
            if args.pinv_relative_cutoff is not None
            else case.pinv_relative_cutoff
        ),
        append_overlap_threshold=(
            float(args.append_overlap_threshold)
            if args.append_overlap_threshold is not None
            else case.append_overlap_threshold
        ),
        append_min_overlap_gain=(
            float(args.append_min_overlap_gain)
            if args.append_min_overlap_gain is not None
            else case.append_min_overlap_gain
        ),
        append_candidate_limit=(
            int(args.append_candidate_limit)
            if args.append_candidate_limit is not None
            else case.append_candidate_limit
        ),
        backend_name=str(args.compile_backend_name) if args.compile_backend_name is not None else case.backend_name,
        seed_transpiler=(
            int(args.compile_seed_transpiler)
            if args.compile_seed_transpiler is not None
            else case.seed_transpiler
        ),
        optimization_level=(
            int(args.compile_optimization_level)
            if args.compile_optimization_level is not None
            else case.optimization_level
        ),
        preferred_fake_backends=preferred or case.preferred_fake_backends,
    )


def _command_from_argv(argv: Sequence[str] | None) -> str:
    if argv is None:
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_avqds_t_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_avqds_t_benchmark", *map(str, argv)])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    case = _case_from_args(args)
    command = _command_from_argv(argv)
    result = run_benchmark(cases=(case,), output_dir=Path(args.output_dir), command=command)
    row = result["rows"][0]
    print(f"manifest_json={result['paths']['manifest_json']}")
    print(f"rows_json={result['paths']['rows_json']}")
    print(f"summary_json={result['paths']['summary_json']}")
    print(f"artifact_run_json={row.get('artifact_run_json')}")
    print(f"final_abs_energy_total_error={row.get('final_abs_energy_total_error')}")
    print(f"mean_abs_energy_total_error={row.get('mean_abs_energy_total_error')}")
    print(f"max_abs_energy_total_error={row.get('max_abs_energy_total_error')}")
    print(f"fidelity_min={row.get('fidelity_min')}")
    print(f"append_events_total={row.get('append_events_total')}")
    print(f"append_candidate_evaluations_total={row.get('append_candidate_evaluations_total')}")
    print(f"unique_scaffold_count={row.get('unique_scaffold_count')}")
    print(f"final_logical_block_count={row.get('final_logical_block_count')}")
    print(f"final_runtime_parameter_count={row.get('final_runtime_parameter_count')}")
    print(f"avqdst_linear_solve_total={row.get('avqdst_linear_solve_total')}")
    print(f"avqdst_step_count={row.get('avqdst_step_count')}")
    print(f"state_at_time_2q={row.get('state_at_time_2q')}")
    print(f"state_at_time_depth={row.get('state_at_time_depth')}")
    print(f"full_horizon_horizon_2q={row.get('full_horizon_horizon_2q')}")
    print(f"full_horizon_depth_serial={row.get('full_horizon_depth_serial')}")
    print(f"controller_state_2q={row.get('controller_state_2q')}")
    print(f"controller_state_depth={row.get('controller_state_depth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
