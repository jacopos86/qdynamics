#!/usr/bin/env python3
"""Benchmark-local adaptive pVQD row for the HH L=2 t=8 anchor.

This module implements an exact-assisted adaptive pVQD diagnostic inside the
benchmark surface only.  It reuses the fixed-pVQD benchmark's anchor loading,
exact interval propagation, projection optimizer, and compile helpers, but it
keeps ansatz-growth policy local to this file and does not touch controller
or production realtime decision paths.  The row is explicitly diagnostic and
not QPU-faithful.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.legacy.hh_benchmarks import hh_fixed_pvqd_benchmark as base
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor


fixed = base.fixed
overlay = base.overlay

SCHEMA_VERSION = "hh_adaptive_pvqd_benchmark_v1"
RUN_SCHEMA_VERSION = "hh_adaptive_pvqd_run_v1"
DEFAULT_CASE_ID = base.DEFAULT_CASE_ID
METHOD_ID = "hh_td_adaptive_pvqd_pareto_lean_l2_exactv1"
METHOD_KIND = "adaptive_pvqd"
DECISION_MODE = base.DECISION_MODE
SEED_FAMILY = base.SEED_FAMILY
STATE_SCOPE = "final_state_scaffold_source"
HORIZON_SCOPE = "serial_sum_checkpoint_scaffold_budget"
EPOCH_SOURCE_SCOPE = "checkpoint_scaffold_epoch_source"
CONTROLLER_STATE_SCOPE = base.CONTROLLER_STATE_SCOPE
CONTROLLER_SOURCE_SCOPE = base.CONTROLLER_SOURCE_SCOPE


@dataclass(frozen=True)
class AdaptivePVQDBenchmarkCase:
    case_id: str
    controller_json: Path
    source_artifact_json: Path
    spec_name: str
    loader_mode: str
    generator_family: str
    fallback_family: str
    append_pool_family: str
    optimizer_method: str
    optimizer_maxiter: int
    overlap_tol: float
    optimizer_ftol: float = 1.0e-10
    append_overlap_threshold: float = 0.9999
    append_min_overlap_gain: float = 1.0e-7
    append_candidate_limit: int | None = None
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class AppendCandidateFit:
    candidate_pool_index: int
    candidate_label: str
    fit: base.PVQDFitResult
    theta_runtime: np.ndarray
    terms: tuple[Any, ...]
    layout: Any
    executor: CompiledAnsatzExecutor
    new_runtime_indices: tuple[int, ...]


@dataclass(frozen=True)
class AdaptivePVQDSimulationResult:
    method: str
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]
    final_state: np.ndarray
    final_terms: tuple[Any, ...]
    final_layout: Any
    final_theta_runtime: np.ndarray
    pvqd_steps: list[dict[str, Any]]
    append_events: list[dict[str, Any]]
    scaffold_epochs: list[dict[str, Any]]
    scaffold_snapshots: list[dict[str, Any]]
    epoch_compile_inputs: dict[str, dict[str, Any]]
    exact_reference_summary: dict[str, Any]


@dataclass(frozen=True)
class AdaptivePVQDBenchmarkRow:
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
    pvqd_nfev_total: int | None
    pvqd_step_count: int | None
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
    case: AdaptivePVQDBenchmarkCase
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


"Built Math: adaptive-pVQD row = fixed pVQD exact-step overlap projection plus benchmark-local one-generator append/refit when overlap falls below threshold; horizon cost is a serial sum over active scaffold epochs."
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
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


def default_cases() -> tuple[AdaptivePVQDBenchmarkCase, ...]:
    return (
        AdaptivePVQDBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_artifact_json=fixed.DEFAULT_PARETO_ARTIFACT,
            spec_name=SEED_FAMILY,
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
            append_pool_family="match_replay",
            optimizer_method="Powell",
            optimizer_maxiter=80,
            overlap_tol=1.0e-8,
            append_overlap_threshold=0.9999,
            append_min_overlap_gain=1.0e-7,
            append_candidate_limit=None,
        ),
    )


def _case_by_id(case_id: str) -> AdaptivePVQDBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown adaptive-pVQD benchmark case_id={case_id!r}; known cases: {known}")


def _run_spec_for_case(case: AdaptivePVQDBenchmarkCase) -> fixed.FixedManifoldRunSpec:
    return fixed.FixedManifoldRunSpec(
        name=str(case.spec_name),
        artifact_json=Path(case.source_artifact_json),
        loader_mode=str(case.loader_mode),
        generator_family=str(case.generator_family),
        fallback_family=str(case.fallback_family),
        append_pool_family=str(case.append_pool_family),
    )


def _compile_defaults_for_case(
    case: AdaptivePVQDBenchmarkCase,
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


def _compiled_executor_for_terms(terms: Sequence[Any], layout: Any) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        list(terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )


def _prepare_scaffold_state(
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_runtime: np.ndarray | Sequence[float],
) -> np.ndarray:
    return base._prepare_fixed_state(executor, psi_ref, theta_runtime)


def _append_candidate_pool(replay_context: Any) -> tuple[list[Any], dict[str, Any]]:
    raw_meta = getattr(replay_context, "append_pool_meta", None)
    meta = dict(getattr(replay_context, "pool_meta", {}) if raw_meta is None else raw_meta)
    if not bool(meta.get("candidate_pool_complete", False)):
        raise ValueError(
            "adaptive pVQD requires a complete append candidate pool; "
            f"source={meta.get('append_pool_source', 'unknown')} "
            f"reason={meta.get('incomplete_reason', 'unknown')}"
        )
    raw_pool = getattr(replay_context, "append_family_pool", None)
    pool = list(getattr(replay_context, "family_pool", ()) if raw_pool is None else raw_pool)
    if not pool:
        raise ValueError("adaptive pVQD append candidate pool is empty")
    return pool, meta


def _extend_scaffold_append_end(
    *,
    current_terms: Sequence[Any],
    current_layout: Any,
    current_theta_runtime: np.ndarray | Sequence[float],
    candidate_term: Any,
) -> tuple[tuple[Any, ...], Any, CompiledAnsatzExecutor, np.ndarray, tuple[int, ...]]:
    theta_arr = np.asarray(current_theta_runtime, dtype=float).reshape(-1)
    new_terms = tuple(current_terms) + (candidate_term,)
    new_layout = build_parameter_layout(
        list(new_terms),
        ignore_identity=bool(current_layout.ignore_identity),
        coefficient_tolerance=float(current_layout.coefficient_tolerance),
        sort_terms=(str(current_layout.term_order).strip().lower() == "sorted"),
    )
    old_blocks = tuple(current_layout.blocks)
    if tuple(new_layout.blocks)[: len(old_blocks)] != old_blocks:
        raise ValueError("Adaptive pVQD append changed the runtime-layout prefix; cannot preserve current theta.")
    if int(theta_arr.size) != int(current_layout.runtime_parameter_count):
        raise ValueError(
            f"current theta/runtime layout mismatch: {theta_arr.size} vs {current_layout.runtime_parameter_count}"
        )
    extra_width = int(new_layout.runtime_parameter_count) - int(current_layout.runtime_parameter_count)
    theta_aug = np.concatenate([theta_arr, np.zeros(max(0, int(extra_width)), dtype=float)])
    new_runtime_indices = tuple(
        range(int(current_layout.runtime_parameter_count), int(new_layout.runtime_parameter_count))
    )
    return (
        tuple(new_terms),
        new_layout,
        _compiled_executor_for_terms(new_terms, new_layout),
        np.asarray(theta_aug, dtype=float).reshape(-1),
        tuple(int(idx) for idx in new_runtime_indices),
    )


def _candidate_label(term: Any, fallback_index: int) -> str:
    return str(getattr(term, "label", f"append_candidate_{int(fallback_index)}"))


def _candidate_pool_indices(
    pool_size: int,
    *,
    used_indices: set[int],
    candidate_limit: int | None,
) -> list[int]:
    indices = [idx for idx in range(int(pool_size)) if int(idx) not in used_indices]
    if candidate_limit is not None:
        indices = indices[: max(0, int(candidate_limit))]
    return [int(idx) for idx in indices]


def _evaluate_append_candidates(
    *,
    current_terms: Sequence[Any],
    current_layout: Any,
    current_theta_runtime: np.ndarray,
    psi_ref: np.ndarray,
    target_state: np.ndarray,
    append_pool: Sequence[Any],
    candidate_indices: Sequence[int],
    method: str,
    maxiter: int,
    overlap_tol: float,
    ftol: float,
) -> list[AppendCandidateFit]:
    out: list[AppendCandidateFit] = []
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

        fit = base._fit_projection_step(
            prepare_state=prepare,
            theta_start=np.asarray(theta_seed, dtype=float).reshape(-1),
            target_state=target_state,
            method=str(method),
            maxiter=int(maxiter),
            overlap_tol=float(overlap_tol),
            ftol=float(ftol),
        )
        out.append(
            AppendCandidateFit(
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
    base_fit: base.PVQDFitResult,
    candidate_fits: Sequence[AppendCandidateFit],
    min_overlap_gain: float,
) -> tuple[AppendCandidateFit | None, float]:
    if not candidate_fits:
        return None, 0.0
    best = max(candidate_fits, key=lambda item: float(item.fit.final_overlap))
    gain = float(best.fit.final_overlap) - float(base_fit.final_overlap)
    if gain >= float(min_overlap_gain):
        return best, float(gain)
    return None, float(gain)


def _layout_signature_payload(layout: Any) -> list[dict[str, Any]]:
    blocks = []
    for block in tuple(layout.blocks):
        blocks.append(
            {
                "candidate_label": str(block.candidate_label),
                "logical_index": int(block.logical_index),
                "runtime_start": int(block.runtime_start),
                "terms": [
                    {
                        "pauli_exyz": str(spec.pauli_exyz),
                        "coeff_real": float(spec.coeff_real),
                        "nq": int(spec.nq),
                    }
                    for spec in tuple(block.terms)
                ],
            }
        )
    return blocks


def _scaffold_signature(layout: Any) -> str:
    payload = {
        "mode": str(layout.mode),
        "term_order": str(layout.term_order),
        "ignore_identity": bool(layout.ignore_identity),
        "coefficient_tolerance": float(layout.coefficient_tolerance),
        "blocks": _layout_signature_payload(layout),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:20]


def _scaffold_snapshot_payload(*, signature: str, layout: Any, interval_count: int | None = None) -> dict[str, Any]:
    payload = {
        "signature": str(signature),
        "logical_block_count": int(layout.logical_parameter_count),
        "runtime_parameter_count": int(layout.runtime_parameter_count),
        "term_labels": [str(block.candidate_label) for block in tuple(layout.blocks)],
    }
    if interval_count is not None:
        payload["interval_count"] = int(interval_count)
    return payload


def _remember_scaffold_compile_input(
    *,
    snapshot_layouts: dict[str, Any],
    epoch_compile_inputs: dict[str, dict[str, Any]],
    signature: str,
    terms: Sequence[Any],
    layout: Any,
    theta_runtime: np.ndarray | Sequence[float],
) -> None:
    """Track a deterministic representative compile input for a scaffold shape.

    A scaffold epoch is keyed by layout/scaffold signature, but Qiskit can still
    produce different compiled circuits if zero-valued parameters are omitted by
    circuit construction.  Use the latest active theta for each signature as the
    representative compile input.  Therefore, a no-append single-signature run
    compiles the same final theta for both state-at-time and horizon accounting.
    """

    key = str(signature)
    if key not in snapshot_layouts:
        snapshot_layouts[key] = layout
    epoch_compile_inputs[key] = {
        "terms": tuple(terms),
        "layout": layout,
        "theta_runtime": np.asarray(theta_runtime, dtype=float).reshape(-1),
        "compile_representative_policy": "latest_active_theta_for_scaffold_signature",
    }


def _summarize_epochs(
    *,
    active_signatures: Sequence[str],
    snapshot_layouts: Mapping[str, Any],
) -> list[dict[str, Any]]:
    counts = Counter(str(sig) for sig in active_signatures)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sig in active_signatures:
        key = str(sig)
        if key in seen:
            continue
        seen.add(key)
        layout = snapshot_layouts[key]
        ordered.append(_scaffold_snapshot_payload(signature=key, layout=layout, interval_count=int(counts[key])))
    return ordered


def _simulate_adaptive_pvqd(
    *,
    case: AdaptivePVQDBenchmarkCase,
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    exact_energy_total: Sequence[float] | None,
    observation_physical_times: Sequence[float],
    exact_steps_multiplier: int,
) -> AdaptivePVQDSimulationResult:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if int(times_arr.size) < 2:
        raise ValueError("adaptive-pVQD requires at least two time points")
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
    pvqd_steps: list[dict[str, Any]] = []
    append_events: list[dict[str, Any]] = []
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
        fit: base.PVQDFitResult | None,
        scaffold_signature: str,
        layout: Any,
        interval_nfev: int = 0,
        append_accepted: bool | None = None,
        append_candidate_evaluations: int = 0,
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
            "append_candidate_evaluations": int(append_candidate_evaluations),
        }
        if fit is None:
            row.update(
                {
                    "pvqd_step_index": None,
                    "pvqd_nfev": 0,
                    "projection_loss_initial": None,
                    "projection_loss_final": None,
                    "projection_overlap_initial": None,
                    "projection_overlap_final": None,
                    "optimizer_status": None,
                    "optimizer_success": None,
                }
            )
        else:
            row.update(
                {
                    "pvqd_step_index": int(idx) - 1,
                    "pvqd_nfev": int(interval_nfev),
                    "projection_loss_initial": float(fit.initial_projection_loss),
                    "projection_loss_final": float(fit.final_projection_loss),
                    "projection_overlap_initial": float(fit.initial_overlap),
                    "projection_overlap_final": float(fit.final_overlap),
                    "optimizer_status": str(fit.status),
                    "optimizer_success": bool(fit.success),
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

        base_fit = base._fit_projection_step(
            prepare_state=prepare_current,
            theta_start=theta,
            target_state=target,
            method=str(case.optimizer_method),
            maxiter=int(case.optimizer_maxiter),
            overlap_tol=float(case.overlap_tol),
            ftol=float(case.optimizer_ftol),
        )
        accepted_fit = base_fit
        accepted_terms = current_terms
        accepted_layout = current_layout
        accepted_executor = current_executor
        accepted_theta = np.asarray(base_fit.theta_runtime, dtype=float).reshape(-1)
        accepted_append: AppendCandidateFit | None = None
        append_gain = 0.0
        candidate_fits: list[AppendCandidateFit] = []
        append_triggered = bool(float(base_fit.final_overlap) < float(case.append_overlap_threshold))

        if append_triggered:
            candidate_indices = _candidate_pool_indices(
                len(append_pool),
                used_indices=used_append_indices,
                candidate_limit=case.append_candidate_limit,
            )
            candidate_fits = _evaluate_append_candidates(
                current_terms=current_terms,
                current_layout=current_layout,
                current_theta_runtime=accepted_theta,
                psi_ref=psi_ref,
                target_state=target,
                append_pool=append_pool,
                candidate_indices=candidate_indices,
                method=str(case.optimizer_method),
                maxiter=int(case.optimizer_maxiter),
                overlap_tol=float(case.overlap_tol),
                ftol=float(case.optimizer_ftol),
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
                used_append_indices.add(int(accepted_append.candidate_pool_index))

        interval_nfev = int(base_fit.nfev) + sum(int(item.fit.nfev) for item in candidate_fits)
        current_terms = tuple(accepted_terms)
        current_layout = accepted_layout
        current_executor = accepted_executor
        theta = np.asarray(accepted_theta, dtype=float).reshape(-1)
        current_state = _prepare_scaffold_state(current_executor, psi_ref, theta)
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
                "nfev_base_fit": int(base_fit.nfev),
                "nfev_candidate_fit": int(accepted_append.fit.nfev),
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
            "optimizer_method": str(case.optimizer_method),
            "optimizer_maxiter": int(case.optimizer_maxiter),
            "overlap_tol": float(case.overlap_tol),
            "append_overlap_threshold": float(case.append_overlap_threshold),
            "append_min_overlap_gain": float(case.append_min_overlap_gain),
            "append_triggered": bool(append_triggered),
            "append_accepted": bool(append_accepted),
            "append_candidate_evaluations": int(len(candidate_fits)),
            "append_candidate_nfev_total": int(sum(int(item.fit.nfev) for item in candidate_fits)),
            "append_best_overlap_gain": float(append_gain),
            "accepted_candidate_pool_index": (
                None if accepted_append is None else int(accepted_append.candidate_pool_index)
            ),
            "accepted_candidate_label": (
                None if accepted_append is None else str(accepted_append.candidate_label)
            ),
            "success": bool(accepted_fit.success),
            "status": str(accepted_fit.status),
            "message": str(accepted_fit.message),
            "nit": accepted_fit.nit,
            "nfev": int(interval_nfev),
            "base_nfev": int(base_fit.nfev),
            "projection_loss_initial": float(accepted_fit.initial_projection_loss),
            "projection_loss_final": float(accepted_fit.final_projection_loss),
            "projection_overlap_initial": float(accepted_fit.initial_overlap),
            "projection_overlap_final": float(accepted_fit.final_overlap),
            "base_projection_overlap_final": float(base_fit.final_overlap),
            "scaffold_signature": str(active_signature),
            "logical_block_count": int(current_layout.logical_parameter_count),
            "runtime_parameter_count": int(current_layout.runtime_parameter_count),
        }
        pvqd_steps.append(step_payload)
        _append_row(
            k + 1,
            current_state,
            fit=accepted_fit,
            scaffold_signature=str(active_signature),
            layout=current_layout,
            interval_nfev=int(interval_nfev),
            append_accepted=bool(append_accepted),
            append_candidate_evaluations=int(len(candidate_fits)),
        )

    summary = base._summarize_trajectory(trajectory)
    scaffold_epochs = _summarize_epochs(
        active_signatures=active_signatures,
        snapshot_layouts=snapshot_layouts,
    )
    active_unique = {str(row["signature"]) for row in scaffold_epochs if int(row.get("interval_count", 0)) > 0}
    summary.update(
        {
            "append_events_total": int(len(append_events)),
            "append_candidate_evaluations_total": int(
                sum(int(step.get("append_candidate_evaluations", 0) or 0) for step in pvqd_steps)
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
    return AdaptivePVQDSimulationResult(
        method=METHOD_ID,
        trajectory=trajectory,
        summary=summary,
        final_state=np.asarray(current_state, dtype=complex).reshape(-1),
        final_terms=tuple(current_terms),
        final_layout=current_layout,
        final_theta_runtime=np.asarray(theta, dtype=float).reshape(-1),
        pvqd_steps=pvqd_steps,
        append_events=append_events,
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
        method="adaptive_pvqd",
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
    base._require_finite_cost(cost, label=f"adaptive-pVQD scaffold compile row scope={scope}")
    return cost, [dict(row) for row in raw_rows if isinstance(row, Mapping)]


def _compile_epoch_scaffolds(
    *,
    context: overlay.RebuiltOverlayContext,
    simulation: AdaptivePVQDSimulationResult,
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
            "method": "adaptive_pvqd",
            "group": "state_at_time",
            "scope": STATE_SCOPE,
            "basis": "adaptive pVQD final scaffold",
            "compiled_count_2q": int(state_2q),
            "compiled_depth": int(state_depth),
            "compiled_size": int(state_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": final_state_cost.scope,
        },
        {
            "method": "adaptive_pvqd",
            "group": "horizon",
            "scope": HORIZON_SCOPE,
            "basis": f"serial sum of {len(epoch_records)} active adaptive pVQD scaffold epochs over {int(intervals)} intervals; each epoch uses latest-active theta for its scaffold signature",
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
    case: AdaptivePVQDBenchmarkCase,
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
        "ansatz_types": "ADAPT seed prep; adaptive pVQD projection with benchmark-local one-generator append/refit",
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
        "optimizer_method": str(case.optimizer_method),
        "optimizer_maxiter": int(case.optimizer_maxiter),
        "overlap_tol": float(case.overlap_tol),
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
        "controller_json": str(case.controller_json),
        "seed_artifact_json": str(case.source_artifact_json),
        "output_dir": str(output_dir),
    }


def _row_from_run_artifact(
    payload: Mapping[str, Any],
    *,
    case: AdaptivePVQDBenchmarkCase,
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
    state_cost = _required_report_row(hardware_rows, method="adaptive_pvqd", scope=STATE_SCOPE)
    full_cost = _required_report_row(hardware_rows, method="adaptive_pvqd", scope=HORIZON_SCOPE)
    controller_cost = _required_report_row(hardware_rows, method="controller", scope=CONTROLLER_STATE_SCOPE)
    intervals = _maybe_int(full_cost.get("intervals"))
    if intervals is None:
        num_times = _maybe_int(manifest.get("num_times"))
        intervals = None if num_times is None else max(int(num_times) - 1, 0)

    row = AdaptivePVQDBenchmarkRow(
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
        pvqd_nfev_total=_required_int(summary.get("pvqd_nfev_total"), field="summary.pvqd_nfev_total"),
        pvqd_step_count=_required_int(summary.get("pvqd_step_count"), field="summary.pvqd_step_count"),
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
    out["append_overlap_threshold"] = _maybe_float(manifest.get("append_overlap_threshold"))
    out["append_min_overlap_gain"] = _maybe_float(manifest.get("append_min_overlap_gain"))
    return out


def _build_run_artifact(
    *,
    case: AdaptivePVQDBenchmarkCase,
    spec: fixed.FixedManifoldRunSpec,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    simulation: AdaptivePVQDSimulationResult,
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
                "optimizer_method": str(case.optimizer_method),
                "optimizer_maxiter": int(case.optimizer_maxiter),
                "overlap_tol": float(case.overlap_tol),
                "optimizer_ftol": float(case.optimizer_ftol),
                "append_overlap_threshold": float(case.append_overlap_threshold),
                "append_min_overlap_gain": float(case.append_min_overlap_gain),
                "append_candidate_limit": case.append_candidate_limit,
                "target_policy": "exact_interval_evolution_of_current_adaptive_pvqd_state",
                "append_policy": "one replay-pool operator appended at end; warm-start extended theta; accept only fixed minimum overlap gain",
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
                "pVQD_target_depends_on_exact_interval_propagation": True,
                "append_candidate_source": "benchmark-local replay append pool",
                "compile_cost_policy": "compile final scaffold for state-at-time; full horizon is serial sum over active unique scaffold epochs using latest-active theta per scaffold signature",
                "controller_reference_policy": "fail_closed_required_source_compile_reference",
            },
            "trajectory": simulation.trajectory,
            "summary": summary,
            "pvqd_steps": simulation.pvqd_steps,
            "append_events": simulation.append_events,
            "scaffold_epochs": simulation.scaffold_epochs,
            "scaffold_snapshots": simulation.scaffold_snapshots,
            "exact_reference_summary": simulation.exact_reference_summary,
            "hardware_report_rows": hardware_report_rows,
            "circuit_costs": [_jsonable(final_state_cost), _jsonable(controller_cost)],
            "raw_compile_rows": {
                "adaptive_pvqd_final_state_scaffold": list(final_raw_compile_rows),
                "adaptive_pvqd_epoch_scaffolds": [
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
    case: AdaptivePVQDBenchmarkCase,
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
    simulation = _simulate_adaptive_pvqd(
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
        "benchmark": "hh_adaptive_pvqd_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "decision_mode": DECISION_MODE,
            "diagnostic_exact_assisted": True,
            "qpu_faithful": False,
            "default_case_id": DEFAULT_CASE_ID,
            "seed_family": SEED_FAMILY,
            "target_policy": "exact_interval_evolution_of_current_adaptive_pvqd_state",
            "append_policy": "one replay-pool operator append/refit under overlap threshold and fixed minimum gain",
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
        "benchmark": "hh_adaptive_pvqd_time_dynamics",
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
                "unique_scaffold_count": row.get("unique_scaffold_count"),
                "final_logical_block_count": row.get("final_logical_block_count"),
                "final_runtime_parameter_count": row.get("final_runtime_parameter_count"),
                "pvqd_nfev_total": row.get("pvqd_nfev_total"),
                "pvqd_step_count": row.get("pvqd_step_count"),
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
    cases: Sequence[AdaptivePVQDBenchmarkCase],
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
        description="Run the benchmark-local HH L2 t=8 adaptive-pVQD row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-artifact-json", type=Path, default=None)
    parser.add_argument("--optimizer-method", type=str, default=None)
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    parser.add_argument("--overlap-tol", type=float, default=None)
    parser.add_argument("--append-overlap-threshold", type=float, default=None)
    parser.add_argument("--append-min-overlap-gain", type=float, default=None)
    parser.add_argument("--append-candidate-limit", type=int, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> AdaptivePVQDBenchmarkCase:
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
        optimizer_method=str(args.optimizer_method) if args.optimizer_method is not None else case.optimizer_method,
        optimizer_maxiter=int(args.optimizer_maxiter) if args.optimizer_maxiter is not None else case.optimizer_maxiter,
        overlap_tol=float(args.overlap_tol) if args.overlap_tol is not None else case.overlap_tol,
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
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_adaptive_pvqd_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_adaptive_pvqd_benchmark", *map(str, argv)])


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
    print(f"unique_scaffold_count={row.get('unique_scaffold_count')}")
    print(f"final_logical_block_count={row.get('final_logical_block_count')}")
    print(f"final_runtime_parameter_count={row.get('final_runtime_parameter_count')}")
    print(f"pvqd_nfev_total={row.get('pvqd_nfev_total')}")
    print(f"pvqd_step_count={row.get('pvqd_step_count')}")
    print(f"state_at_time_2q={row.get('state_at_time_2q')}")
    print(f"state_at_time_depth={row.get('state_at_time_depth')}")
    print(f"full_horizon_horizon_2q={row.get('full_horizon_horizon_2q')}")
    print(f"full_horizon_depth_serial={row.get('full_horizon_depth_serial')}")
    print(f"controller_state_2q={row.get('controller_state_2q')}")
    print(f"controller_state_depth={row.get('controller_state_depth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
