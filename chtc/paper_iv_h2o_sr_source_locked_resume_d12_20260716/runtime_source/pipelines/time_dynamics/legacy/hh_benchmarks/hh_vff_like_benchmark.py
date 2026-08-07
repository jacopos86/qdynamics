#!/usr/bin/env python3
"""Benchmark-local VFF-like diagnostic row for the HH L=2 t=8 anchor.

This module is intentionally *not* a true VFF or LSVQC implementation.  It
builds exact-state supervised labels only at sparse training checkpoints, fits
the existing fixed ``pareto_lean_l2`` scaffold to those labels, then performs
full-grid inference by deterministic piecewise-linear interpolation of the
trained runtime parameters.  Exact reference states are allowed only as offline
training labels and reporting side-channel diagnostics; inference itself never
optimizes and never queries exact state labels.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from pipelines.time_dynamics.fixed_manifold import mclachlan as fixed
from pipelines.time_dynamics.legacy.hh_benchmarks import hh_fixed_pvqd_benchmark as fixed_pvqd
from pipelines.time_dynamics.legacy.analysis import hh_realtime_suzuki_overlay as overlay


SCHEMA_VERSION = "hh_vff_like_benchmark_v1"
RUN_SCHEMA_VERSION = "hh_vff_like_run_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_vff_like_pareto_lean_l2_exacttrain_v1"
METHOD_KIND = "vff_like_statefit"
DECISION_MODE = "exact_train_interp_v1"
SEED_FAMILY = "pareto_lean_l2"
TRAINING_SCOPE = "offline_exact_state_supervised_fit"
STATE_SCOPE = "trained_fixed_scaffold_source"
HORIZON_SCOPE = "repeated_fixed_scaffold_inference_budget"
CONTROLLER_STATE_SCOPE = fixed_pvqd.CONTROLLER_STATE_SCOPE


@dataclass(frozen=True)
class VFFLikeBenchmarkCase:
    case_id: str
    controller_json: Path
    source_artifact_json: Path
    spec_name: str
    loader_mode: str
    generator_family: str
    fallback_family: str
    append_pool_family: str
    train_knot_stride: int
    interpolation_mode: str
    optimizer_method: str
    optimizer_maxiter: int
    overlap_tol: float
    optimizer_ftol: float = 1.0e-10
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class VFFLikeTrainingResult:
    knot_indices: tuple[int, ...]
    knot_times: tuple[float, ...]
    knot_thetas: np.ndarray
    records: list[dict[str, Any]]
    nfev_total: int
    fit_success_count: int


@dataclass(frozen=True)
class VFFLikeSimulationResult:
    method: str
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]
    final_state: np.ndarray
    theta_trajectory: np.ndarray
    training: VFFLikeTrainingResult
    exact_reference_summary: dict[str, Any]


@dataclass(frozen=True)
class VFFLikeBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    status: str
    decision_mode: str
    diagnostic_exact_assisted: bool
    qpu_faithful: bool
    exact_training_labels: bool
    inference_uses_exact: bool
    exact_fields_reporting_only: bool
    seed_family: str
    controller_json: str | None
    source_artifact_json: str | None
    drive_enabled: bool | None
    t_final: float | None
    num_times: int | None
    training_scope: str
    training_checkpoint_count: int
    training_nfev_total: int
    training_fit_success_count: int
    train_knot_stride: int
    interpolation_mode: str
    final_energy_total: float | None
    final_energy_total_exact: float | None
    final_abs_energy_total_error: float | None
    mean_abs_energy_total_error: float | None
    max_abs_energy_total_error: float | None
    fidelity_min: float | None
    final_logical_block_count: int | None
    final_runtime_parameter_count: int | None
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
    exact_assisted_contract: str = "exact_training_labels_diagnostic_not_qpu_faithful"


@dataclass(frozen=True)
class _CaseRunRecord:
    case: VFFLikeBenchmarkCase
    spec: fixed.FixedManifoldRunSpec
    run_json: Path
    run_artifact: Mapping[str, Any]
    row: dict[str, Any]
    compile_defaults: Mapping[str, Any]


"Built Math: VFF-like diagnostic row = sparse exact-state-supervised fixed-scaffold parameter labels + deterministic linear parameter interpolation; horizon cost is one trained fixed scaffold repeated over num_times-1 intervals."
def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, complex):
        return {"re": _jsonable(float(np.real(value))), "im": _jsonable(float(np.imag(value)))}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output_path


def _maybe_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _maybe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


def _as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _required_finite_float(value: Any, *, field: str) -> float:
    out = _maybe_float(value)
    if out is None:
        raise ValueError(f"{field} must be finite; got {value!r}")
    return float(out)


def _required_int(value: Any, *, field: str) -> int:
    out = _maybe_int(value)
    if out is None:
        raise ValueError(f"{field} must be present; got {value!r}")
    return int(out)


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [str(x) for x in raw]
    return tuple(part.strip() for part in parts if part.strip())


def default_cases() -> tuple[VFFLikeBenchmarkCase, ...]:
    return (
        VFFLikeBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_artifact_json=fixed.DEFAULT_PARETO_ARTIFACT,
            spec_name=SEED_FAMILY,
            loader_mode="replay_family",
            generator_family="match_adapt",
            fallback_family="full_meta",
            append_pool_family="match_replay",
            train_knot_stride=8,
            interpolation_mode="piecewise_linear",
            optimizer_method="Powell",
            optimizer_maxiter=80,
            overlap_tol=1.0e-8,
        ),
    )


def _case_by_id(case_id: str) -> VFFLikeBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown VFF-like benchmark case_id={case_id!r}; known cases: {known}")


def _run_spec_for_case(case: VFFLikeBenchmarkCase) -> fixed.FixedManifoldRunSpec:
    return fixed.FixedManifoldRunSpec(
        name=str(case.spec_name),
        artifact_json=Path(case.source_artifact_json),
        loader_mode=str(case.loader_mode),
        generator_family=str(case.generator_family),
        fallback_family=str(case.fallback_family),
        append_pool_family=str(case.append_pool_family),
    )


def _training_knot_indices(num_times: int, train_knot_stride: int) -> tuple[int, ...]:
    n = int(num_times)
    stride = int(train_knot_stride)
    if n < 2:
        raise ValueError("VFF-like benchmark requires at least two time samples")
    if stride < 1:
        raise ValueError("train_knot_stride must be >= 1")
    indices = {0, n - 1}
    indices.update(range(0, n, stride))
    out = tuple(sorted(int(idx) for idx in indices))
    if len(out) >= n:
        raise ValueError(
            "sparse VFF-like training requires training_checkpoint_count < num_times; "
            f"got training_checkpoint_count={len(out)} num_times={n}"
        )
    return out


def _interpolate_theta_piecewise_linear(
    *,
    times: np.ndarray | Sequence[float],
    knot_indices: Sequence[int],
    knot_thetas: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if times_arr.size <= 0:
        raise ValueError("cannot interpolate over an empty time grid")
    if not np.all(np.isfinite(times_arr)):
        raise ValueError("time grid contains non-finite values")
    knot_idx = np.asarray(tuple(int(x) for x in knot_indices), dtype=int).reshape(-1)
    if knot_idx.size < 2:
        raise ValueError("piecewise-linear interpolation requires at least two training knots")
    if np.any(knot_idx < 0) or np.any(knot_idx >= times_arr.size):
        raise ValueError("training knot index out of bounds")
    knot_times = times_arr[knot_idx]
    if np.any(np.diff(knot_times) <= 0.0):
        raise ValueError("training knot times must be strictly increasing")
    theta_knots = np.asarray(knot_thetas, dtype=float)
    if theta_knots.ndim != 2:
        raise ValueError("knot_thetas must be a two-dimensional array")
    if theta_knots.shape[0] != knot_idx.size:
        raise ValueError("knot_thetas row count must match knot_indices")
    if not np.all(np.isfinite(theta_knots)):
        raise ValueError("knot_thetas contains non-finite values")
    out = np.empty((times_arr.size, theta_knots.shape[1]), dtype=float)
    for col in range(theta_knots.shape[1]):
        out[:, col] = np.interp(times_arr, knot_times, theta_knots[:, col])
    if not np.all(np.isfinite(out)):
        raise ValueError("interpolated theta trajectory contains non-finite values")
    return out


def _fit_state_target(
    *,
    prepare_state: Callable[[np.ndarray], np.ndarray],
    theta_start: np.ndarray | Sequence[float],
    target_state: np.ndarray | Sequence[complex],
    method: str,
    maxiter: int,
    overlap_tol: float,
    ftol: float,
) -> fixed_pvqd.PVQDFitResult:
    return fixed_pvqd._fit_projection_step(
        prepare_state=prepare_state,
        theta_start=theta_start,
        target_state=target_state,
        method=method,
        maxiter=int(maxiter),
        overlap_tol=float(overlap_tol),
        ftol=float(ftol),
    )


def _require_finite_fit(fit: fixed_pvqd.PVQDFitResult, *, knot_index: int) -> None:
    theta = np.asarray(fit.theta_runtime, dtype=float).reshape(-1)
    if theta.size <= 0 or not np.all(np.isfinite(theta)):
        raise ValueError(f"non-finite knot fit at training knot {int(knot_index)}: theta")
    for field in ("initial_projection_loss", "final_projection_loss", "initial_overlap", "final_overlap"):
        value = getattr(fit, field)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"non-finite knot fit at training knot {int(knot_index)}: {field}") from None
        if not math.isfinite(numeric):
            raise ValueError(f"non-finite knot fit at training knot {int(knot_index)}: {field}")
    if not math.isfinite(float(fit.nfev)):
        raise ValueError(f"non-finite knot fit at training knot {int(knot_index)}: nfev")


def _fit_training_knots(
    *,
    case: VFFLikeBenchmarkCase,
    prepare_state: Callable[[np.ndarray], np.ndarray],
    theta_start: np.ndarray | Sequence[float],
    exact_states: Sequence[np.ndarray | Sequence[complex]],
    times: np.ndarray | Sequence[float],
    knot_indices: Sequence[int],
) -> VFFLikeTrainingResult:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    theta = np.asarray(theta_start, dtype=float).reshape(-1)
    if theta.size <= 0 or not np.all(np.isfinite(theta)):
        raise ValueError("theta_start must be a finite non-empty vector")
    knot_idx = tuple(int(idx) for idx in knot_indices)
    records: list[dict[str, Any]] = []
    knot_thetas: list[np.ndarray] = []
    nfev_total = 0
    success_count = 0
    for fit_number, idx in enumerate(knot_idx):
        if idx < 0 or idx >= len(exact_states) or idx >= times_arr.size:
            raise ValueError(f"training knot index out of bounds: {idx}")
        fit = _fit_state_target(
            prepare_state=prepare_state,
            theta_start=theta,
            target_state=exact_states[idx],
            method=str(case.optimizer_method),
            maxiter=int(case.optimizer_maxiter),
            overlap_tol=float(case.overlap_tol),
            ftol=float(case.optimizer_ftol),
        )
        _require_finite_fit(fit, knot_index=int(idx))
        theta = np.asarray(fit.theta_runtime, dtype=float).reshape(-1)
        nfev_total += int(fit.nfev)
        success_count += 1 if bool(fit.success) else 0
        knot_thetas.append(np.asarray(theta, dtype=float).copy())
        records.append(
            {
                "fit_number": int(fit_number),
                "knot_index": int(idx),
                "time": float(times_arr[int(idx)]),
                "optimizer_method": str(case.optimizer_method),
                "optimizer_maxiter": int(case.optimizer_maxiter),
                "overlap_tol": float(case.overlap_tol),
                "success": bool(fit.success),
                "status": str(fit.status),
                "message": str(fit.message),
                "nit": fit.nit,
                "nfev": int(fit.nfev),
                "projection_loss_initial": float(fit.initial_projection_loss),
                "projection_loss_final": float(fit.final_projection_loss),
                "projection_overlap_initial": float(fit.initial_overlap),
                "projection_overlap_final": float(fit.final_overlap),
                "theta_runtime": np.asarray(theta, dtype=float).tolist(),
            }
        )
    theta_arr = np.vstack(knot_thetas)
    if not np.all(np.isfinite(theta_arr)):
        raise ValueError("non-finite knot fit in fitted theta table")
    return VFFLikeTrainingResult(
        knot_indices=knot_idx,
        knot_times=tuple(float(times_arr[idx]) for idx in knot_idx),
        knot_thetas=theta_arr,
        records=records,
        nfev_total=int(nfev_total),
        fit_success_count=int(success_count),
    )


def _summarize_trajectory(rows: Sequence[Mapping[str, Any]], *, training: VFFLikeTrainingResult) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty VFF-like trajectory")
    energies = [_required_finite_float(row.get("energy_total"), field="energy_total") for row in rows]
    exact_values = [_maybe_float(row.get("energy_total_exact")) for row in rows]
    errors = [_maybe_float(row.get("abs_energy_total_error")) for row in rows]
    finite_errors = [float(x) for x in errors if x is not None]
    fidelities = [_maybe_float(row.get("fidelity_exact")) for row in rows]
    finite_fidelities = [float(x) for x in fidelities if x is not None]
    return {
        "row_count": int(len(rows)),
        "final_energy_total": float(energies[-1]),
        "final_energy_total_exact": exact_values[-1] if exact_values else None,
        "final_abs_energy_total_error": None if not finite_errors else errors[-1],
        "mean_abs_energy_total_error": None if not finite_errors else float(sum(finite_errors) / len(finite_errors)),
        "max_abs_energy_total_error": None if not finite_errors else float(max(finite_errors)),
        "fidelity_min": None if not finite_fidelities else float(min(finite_fidelities)),
        "fidelity_final": None if not finite_fidelities else fidelities[-1],
        "training_checkpoint_count": int(len(training.knot_indices)),
        "training_nfev_total": int(training.nfev_total),
        "training_fit_success_count": int(training.fit_success_count),
    }


def _simulate_vff_like(
    *,
    case: VFFLikeBenchmarkCase,
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    exact_energy_total: Sequence[float] | None,
    observation_physical_times: Sequence[float],
    exact_steps_multiplier: int,
) -> VFFLikeSimulationResult:
    if str(case.interpolation_mode) != "piecewise_linear":
        raise ValueError(f"unsupported interpolation_mode={case.interpolation_mode!r}")
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if int(times_arr.size) < 2:
        raise ValueError("VFF-like benchmark requires at least two time points")
    obs_physical = np.asarray(observation_physical_times, dtype=float).reshape(-1)
    if int(obs_physical.size) != int(times_arr.size):
        raise ValueError("observation_physical_times must match source time grid")
    exact_arr = None if exact_energy_total is None else np.asarray(exact_energy_total, dtype=float).reshape(-1)
    if exact_arr is not None and int(exact_arr.size) != int(times_arr.size):
        raise ValueError("exact_energy_total must match source time grid")

    loaded = context.loaded
    executor = fixed_pvqd._compiled_executor(loaded)
    psi_ref = np.asarray(loaded.replay_context.psi_ref, dtype=complex).reshape(-1)
    theta_start = np.asarray(loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1)
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))

    exact_states = fixed_pvqd._build_exact_reference_states(
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

    def prepare(theta_vec: np.ndarray) -> np.ndarray:
        return fixed_pvqd._prepare_fixed_state(
            executor,
            psi_ref,
            np.asarray(theta_vec, dtype=float).reshape(-1),
        )

    knot_indices = _training_knot_indices(
        num_times=int(times_arr.size),
        train_knot_stride=int(case.train_knot_stride),
    )
    training = _fit_training_knots(
        case=case,
        prepare_state=prepare,
        theta_start=theta_start,
        exact_states=exact_states,
        times=times_arr,
        knot_indices=knot_indices,
    )
    theta_trajectory = _interpolate_theta_piecewise_linear(
        times=times_arr,
        knot_indices=training.knot_indices,
        knot_thetas=training.knot_thetas,
    )

    trajectory: list[dict[str, Any]] = []
    for idx, theta_vec in enumerate(theta_trajectory):
        state = prepare(np.asarray(theta_vec, dtype=float).reshape(-1))
        hmat_total = overlay._hmat_total_at_observation(
            hmat_static=np.asarray(context.hmat, dtype=complex),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            physical_time=float(obs_physical[int(idx)]),
            nq=int(context.nq),
        )
        energy = fixed_pvqd._expectation_hamiltonian(state, np.asarray(hmat_total, dtype=complex))
        exact_state = fixed_pvqd._normalize_state(np.asarray(exact_states[int(idx)], dtype=complex).reshape(-1))
        exact_energy = (
            float(exact_arr[int(idx)])
            if exact_arr is not None
            else fixed_pvqd._expectation_hamiltonian(exact_state, np.asarray(hmat_total, dtype=complex))
        )
        fidelity = float(abs(np.vdot(exact_state, fixed_pvqd._normalize_state(state))) ** 2)
        trajectory.append(
            {
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
                "runtime_parameter_count": int(theta_trajectory.shape[1]),
                "logical_block_count": int(loaded.replay_context.base_layout.logical_parameter_count),
                "inference_optimizer_called": False,
                "inference_uses_exact": False,
                "theta_runtime": np.asarray(theta_vec, dtype=float).tolist(),
            }
        )

    summary = _summarize_trajectory(trajectory, training=training)
    return VFFLikeSimulationResult(
        method=METHOD_ID,
        trajectory=trajectory,
        summary=summary,
        final_state=prepare(np.asarray(theta_trajectory[-1], dtype=float).reshape(-1)),
        theta_trajectory=np.asarray(theta_trajectory, dtype=float),
        training=training,
        exact_reference_summary={
            "state_count": int(len(exact_states)),
            "reference_policy": "benchmark-local exact states used for sparse offline training labels and reporting diagnostics",
            "exact_steps_multiplier": int(exact_steps_multiplier),
            "training_checkpoint_count": int(len(training.knot_indices)),
            "inference_uses_exact": False,
        },
    )


def _compile_trained_state_scaffold(
    *,
    context: overlay.RebuiltOverlayContext,
    theta_runtime: np.ndarray | Sequence[float],
    compile_defaults: Mapping[str, Any],
) -> tuple[overlay.CircuitCostRow, list[dict[str, Any]]]:
    scaffold_circuit = overlay.build_ansatz_circuit(
        context.loaded.replay_context.base_layout,
        np.asarray(theta_runtime, dtype=float).reshape(-1),
        int(context.nq),
        ref_state=np.asarray(context.loaded.replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    cost, raw_rows = overlay._compile_one_circuit_cost(
        method=METHOD_KIND,
        order=None,
        scope=STATE_SCOPE,
        trotter_steps=None,
        includes_seed_prep=True,
        circuit=scaffold_circuit,
        backend_name=str(compile_defaults["backend_name"]),
        preferred_fake_backends=tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        seed_transpiler=int(compile_defaults["seed_transpiler"]),
        optimization_level=int(compile_defaults["optimization_level"]),
    )
    fixed_pvqd._require_finite_cost(cost, label="VFF-like trained fixed scaffold compile row")
    return cost, [dict(row) for row in raw_rows if isinstance(row, Mapping)]


def _hardware_report_rows(
    *,
    state_cost: overlay.CircuitCostRow,
    controller_cost: overlay.CircuitCostRow,
    intervals: int,
) -> list[dict[str, Any]]:
    state_2q = _required_int(state_cost.compiled_count_2q, field="state_at_time_2q")
    state_depth = _required_int(state_cost.compiled_depth, field="state_at_time_depth")
    state_size = _required_int(state_cost.compiled_size, field="state_at_time_size")
    controller_2q = _required_int(controller_cost.compiled_count_2q, field="controller_state_2q")
    controller_depth = _required_int(controller_cost.compiled_depth, field="controller_state_depth")
    controller_size = _required_int(controller_cost.compiled_size, field="controller_state_size")
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
            "basis": "VFF-like trained fixed scaffold inference state",
            "compiled_count_2q": int(state_2q),
            "compiled_depth": int(state_depth),
            "compiled_size": int(state_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": state_cost.scope,
        },
        {
            "method": METHOD_KIND,
            "group": "horizon",
            "scope": HORIZON_SCOPE,
            "basis": f"{int(intervals)} repeated trained fixed-scaffold inference states",
            "compiled_count_2q": int(state_2q) * int(intervals),
            "compiled_depth": int(state_depth) * int(intervals),
            "compiled_size": int(state_size) * int(intervals),
            "horizon_count_2q": int(state_2q) * int(intervals),
            "horizon_depth_serial": int(state_depth) * int(intervals),
            "source_scope": state_cost.scope,
            "intervals": int(intervals),
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
    case: VFFLikeBenchmarkCase,
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
        "ansatz_types": "ADAPT seed prep; VFF-like sparse exact-statefit labels on frozen scaffold; linear parameter interpolation",
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
        "training_scope": TRAINING_SCOPE,
        "train_knot_stride": int(case.train_knot_stride),
        "interpolation_mode": str(case.interpolation_mode),
        "optimizer_method": str(case.optimizer_method),
        "optimizer_maxiter": int(case.optimizer_maxiter),
        "overlap_tol": float(case.overlap_tol),
        "exact_reference_method": _as_optional_str(reference.get("reference_method")),
        "exact_steps_multiplier": int(exact_steps_multiplier),
        "decision_mode": DECISION_MODE,
        "diagnostic_exact_assisted": True,
        "qpu_faithful": False,
        "exact_training_labels": True,
        "inference_uses_exact": False,
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
    case: VFFLikeBenchmarkCase,
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

    row = VFFLikeBenchmarkRow(
        case_id=str(case.case_id),
        method_id=METHOD_ID,
        method_kind=METHOD_KIND,
        status="ok",
        decision_mode=DECISION_MODE,
        diagnostic_exact_assisted=True,
        qpu_faithful=False,
        exact_training_labels=True,
        inference_uses_exact=False,
        exact_fields_reporting_only=False,
        seed_family=SEED_FAMILY,
        controller_json=_as_optional_str(manifest.get("controller_json") or source.get("controller_json")),
        source_artifact_json=_as_optional_str(manifest.get("seed_artifact_json") or source.get("artifact_json")),
        drive_enabled=_maybe_bool(manifest.get("drive_enabled")),
        t_final=_maybe_float(manifest.get("t_final")),
        num_times=_maybe_int(manifest.get("num_times")),
        training_scope=TRAINING_SCOPE,
        training_checkpoint_count=_required_int(summary.get("training_checkpoint_count"), field="summary.training_checkpoint_count"),
        training_nfev_total=_required_int(summary.get("training_nfev_total"), field="summary.training_nfev_total"),
        training_fit_success_count=_required_int(summary.get("training_fit_success_count"), field="summary.training_fit_success_count"),
        train_knot_stride=_required_int(manifest.get("train_knot_stride"), field="manifest.train_knot_stride"),
        interpolation_mode=str(manifest.get("interpolation_mode", case.interpolation_mode)),
        final_energy_total=_required_finite_float(summary.get("final_energy_total"), field="summary.final_energy_total"),
        final_energy_total_exact=_required_finite_float(summary.get("final_energy_total_exact"), field="summary.final_energy_total_exact"),
        final_abs_energy_total_error=_required_finite_float(summary.get("final_abs_energy_total_error"), field="summary.final_abs_energy_total_error"),
        mean_abs_energy_total_error=_required_finite_float(summary.get("mean_abs_energy_total_error"), field="summary.mean_abs_energy_total_error"),
        max_abs_energy_total_error=_required_finite_float(summary.get("max_abs_energy_total_error"), field="summary.max_abs_energy_total_error"),
        fidelity_min=_required_finite_float(summary.get("fidelity_min"), field="summary.fidelity_min"),
        final_logical_block_count=_maybe_int(summary.get("final_logical_block_count")),
        final_runtime_parameter_count=_maybe_int(summary.get("final_runtime_parameter_count")),
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
    return out


def _build_run_artifact(
    *,
    case: VFFLikeBenchmarkCase,
    spec: fixed.FixedManifoldRunSpec,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    simulation: VFFLikeSimulationResult,
    state_cost: overlay.CircuitCostRow,
    controller_cost: overlay.CircuitCostRow,
    raw_compile_rows: Sequence[Mapping[str, Any]],
    compile_defaults: Mapping[str, Any],
    output_dir: Path,
    command: str,
    exact_steps_multiplier: int,
) -> dict[str, Any]:
    intervals = max(int(times.size) - 1, 0)
    summary = dict(simulation.summary)
    summary["final_logical_block_count"] = int(context.loaded.replay_context.base_layout.logical_parameter_count)
    summary["final_runtime_parameter_count"] = int(context.loaded.replay_context.base_layout.runtime_parameter_count)
    hardware_report_rows = _hardware_report_rows(
        state_cost=state_cost,
        controller_cost=controller_cost,
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
                "training_scope": TRAINING_SCOPE,
                "train_knot_stride": int(case.train_knot_stride),
                "interpolation_mode": str(case.interpolation_mode),
                "optimizer_method": str(case.optimizer_method),
                "optimizer_maxiter": int(case.optimizer_maxiter),
                "overlap_tol": float(case.overlap_tol),
                "optimizer_ftol": float(case.optimizer_ftol),
                "compile_backend_name": str(compile_defaults["backend_name"]),
                "compile_seed_transpiler": int(compile_defaults["seed_transpiler"]),
                "compile_optimization_level": int(compile_defaults["optimization_level"]),
                "compile_preferred_fake_backends": tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
            },
            "contract": {
                "decision_mode": DECISION_MODE,
                "diagnostic_exact_assisted": True,
                "qpu_faithful": False,
                "exact_training_labels": True,
                "inference_uses_exact": False,
                "controller_decisions_modified": False,
                "controller_paths_called": False,
                "method_kind_honesty": "vff_like_statefit_not_true_vff_or_lsvqc",
                "training_policy": "exact states are used only as sparse offline training labels",
                "inference_policy": "piecewise-linear parameter interpolation only; no per-time exact-state optimization",
                "compile_cost_policy": "compile one representative trained fixed scaffold; repeated-horizon budget multiplies by num_times-1 intervals",
                "controller_reference_policy": "fail_closed_required_source_compile_reference",
            },
            "trajectory": simulation.trajectory,
            "summary": summary,
            "training_knots": {
                "knot_indices": list(simulation.training.knot_indices),
                "knot_times": list(simulation.training.knot_times),
                "knot_thetas": simulation.training.knot_thetas,
                "records": simulation.training.records,
                "training_checkpoint_count": int(len(simulation.training.knot_indices)),
                "training_nfev_total": int(simulation.training.nfev_total),
                "training_fit_success_count": int(simulation.training.fit_success_count),
            },
            "theta_trajectory": simulation.theta_trajectory,
            "exact_reference_summary": simulation.exact_reference_summary,
            "hardware_report_rows": hardware_report_rows,
            "circuit_costs": [_jsonable(state_cost), _jsonable(controller_cost)],
            "raw_compile_rows": {
                "vff_like_trained_fixed_scaffold": list(raw_compile_rows),
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
    case: VFFLikeBenchmarkCase,
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
    compile_defaults = fixed_pvqd._compile_defaults_for_case(case, source_payload)
    controller_cost = fixed_pvqd._required_controller_cost_row(source_payload)
    spec = _run_spec_for_case(case)
    context = fixed_pvqd._build_fixed_context(spec=spec, case=case, source_payload=source_payload)
    exact_steps_multiplier = fixed_pvqd._exact_steps_multiplier_from_source(source_payload)
    simulation = _simulate_vff_like(
        case=case,
        context=context,
        times=times,
        exact_energy_total=exact_energy,
        observation_physical_times=physical_times,
        exact_steps_multiplier=int(exact_steps_multiplier),
    )
    state_cost, raw_compile_rows = _compile_trained_state_scaffold(
        context=context,
        theta_runtime=np.asarray(simulation.theta_trajectory[-1], dtype=float).reshape(-1),
        compile_defaults=compile_defaults,
    )
    run_artifact = _build_run_artifact(
        case=case,
        spec=spec,
        source_payload=source_payload,
        context=context,
        times=times,
        simulation=simulation,
        state_cost=state_cost,
        controller_cost=controller_cost,
        raw_compile_rows=raw_compile_rows,
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
        "benchmark": "hh_vff_like_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "decision_mode": DECISION_MODE,
            "diagnostic_exact_assisted": True,
            "qpu_faithful": False,
            "exact_training_labels": True,
            "inference_uses_exact": False,
            "default_case_id": DEFAULT_CASE_ID,
            "seed_family": SEED_FAMILY,
            "training_scope": TRAINING_SCOPE,
            "interpolation_mode": "piecewise_linear",
            "method_kind_honesty": "VFF-like exact-statefit surrogate; not true VFF/LSVQC",
            "compile_cost_policy": (
                "compile one representative trained fixed scaffold; repeated-horizon budget is "
                "state cost multiplied by num_times-1 intervals"
            ),
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
        "benchmark": "hh_vff_like_time_dynamics",
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
                "training_checkpoint_count": row.get("training_checkpoint_count"),
                "training_nfev_total": row.get("training_nfev_total"),
                "training_fit_success_count": row.get("training_fit_success_count"),
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
    cases: Sequence[VFFLikeBenchmarkCase],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the benchmark-local HH L2 t=8 VFF-like exact-trained diagnostic row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-artifact-json", type=Path, default=None)
    parser.add_argument("--train-knot-stride", type=int, default=None)
    parser.add_argument("--interpolation-mode", type=str, default=None)
    parser.add_argument("--optimizer-method", type=str, default=None)
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    parser.add_argument("--overlap-tol", type=float, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> VFFLikeBenchmarkCase:
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
        train_knot_stride=(
            int(args.train_knot_stride) if args.train_knot_stride is not None else case.train_knot_stride
        ),
        interpolation_mode=(
            str(args.interpolation_mode) if args.interpolation_mode is not None else case.interpolation_mode
        ),
        optimizer_method=str(args.optimizer_method) if args.optimizer_method is not None else case.optimizer_method,
        optimizer_maxiter=int(args.optimizer_maxiter) if args.optimizer_maxiter is not None else case.optimizer_maxiter,
        overlap_tol=float(args.overlap_tol) if args.overlap_tol is not None else case.overlap_tol,
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
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_vff_like_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_vff_like_benchmark", *map(str, argv)])


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
    print(f"training_checkpoint_count={row.get('training_checkpoint_count')}")
    print(f"training_nfev_total={row.get('training_nfev_total')}")
    print(f"training_fit_success_count={row.get('training_fit_success_count')}")
    print(f"state_at_time_2q={row.get('state_at_time_2q')}")
    print(f"state_at_time_depth={row.get('state_at_time_depth')}")
    print(f"full_horizon_horizon_2q={row.get('full_horizon_horizon_2q')}")
    print(f"full_horizon_depth_serial={row.get('full_horizon_depth_serial')}")
    print(f"controller_state_2q={row.get('controller_state_2q')}")
    print(f"controller_state_depth={row.get('controller_state_depth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
