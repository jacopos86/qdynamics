#!/usr/bin/env python3
"""Benchmark-local fixed pVQD row for the HH L=2 t=8 anchor.

This module implements a fixed-scaffold pVQD diagnostic inside the benchmark
surface only.  It reuses the validated anchor time grid/scaffold reconstruction
and compile-audit helpers, but it does not touch controller decisions or
production realtime routes.  The row is exact-assisted and explicitly not
QPU-faithful.
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

from pipelines.hardcoded import hubbard_pipeline as hc_pipeline
from pipelines.time_dynamics.fixed_manifold import mclachlan as fixed
from pipelines.time_dynamics.legacy.analysis import hh_realtime_suzuki_overlay as overlay
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor

try:  # pragma: no cover - import guard only
    from scipy.linalg import expm as _dense_expm
    from scipy.optimize import minimize as scipy_minimize
    from scipy.sparse import csc_matrix as _csc_matrix
    from scipy.sparse.linalg import expm_multiply as _expm_multiply
except ImportError:  # pragma: no cover
    _dense_expm = None
    scipy_minimize = None
    _csc_matrix = None
    _expm_multiply = None


SCHEMA_VERSION = "hh_fixed_pvqd_benchmark_v1"
RUN_SCHEMA_VERSION = "hh_fixed_pvqd_run_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_fixed_pvqd_pareto_lean_l2_exactv1"
METHOD_KIND = "fixed_pvqd"
DECISION_MODE = "exact_v1"
SEED_FAMILY = "pareto_lean_l2"
STATE_SCOPE = "state_scaffold_source"
HORIZON_SCOPE = "repeated_state_scaffold_budget"
CONTROLLER_STATE_SCOPE = "controller_state_at_time"
CONTROLLER_SOURCE_SCOPE = "controller_final_scaffold_source"


@dataclass(frozen=True)
class FixedPVQDBenchmarkCase:
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
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class PVQDFitResult:
    theta_runtime: np.ndarray
    initial_projection_loss: float
    final_projection_loss: float
    initial_overlap: float
    final_overlap: float
    nfev: int
    nit: int | None
    success: bool
    status: str
    message: str


@dataclass(frozen=True)
class FixedPVQDSimulationResult:
    method: str
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]
    final_state: np.ndarray
    pvqd_steps: list[dict[str, Any]]
    exact_reference_summary: dict[str, Any]


@dataclass(frozen=True)
class FixedPVQDBenchmarkRow:
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
    pvqd_nfev_total: int | None
    pvqd_step_count: int | None
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
    exact_assisted_contract: str = "diagnostic_exact_assisted_not_qpu_faithful"


@dataclass(frozen=True)
class _CaseRunRecord:
    case: FixedPVQDBenchmarkCase
    spec: fixed.FixedManifoldRunSpec
    run_json: Path
    run_artifact: Mapping[str, Any]
    row: dict[str, Any]
    compile_defaults: Mapping[str, Any]


"Built Math: fixed-pVQD row = exact-step target propagation on a frozen scaffold + overlap projection fit; horizon cost is one fixed state scaffold repeated over num_times-1 intervals."
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


def _mul_optional_int(value: int | None, factor: int) -> int | None:
    return None if value is None else int(value) * int(factor)


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [str(x) for x in raw]
    return tuple(part.strip() for part in parts if part.strip())


def default_cases() -> tuple[FixedPVQDBenchmarkCase, ...]:
    return (
        FixedPVQDBenchmarkCase(
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
        ),
    )


def _case_by_id(case_id: str) -> FixedPVQDBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown fixed-pVQD benchmark case_id={case_id!r}; known cases: {known}")


def _run_spec_for_case(case: FixedPVQDBenchmarkCase) -> fixed.FixedManifoldRunSpec:
    return fixed.FixedManifoldRunSpec(
        name=str(case.spec_name),
        artifact_json=Path(case.source_artifact_json),
        loader_mode=str(case.loader_mode),
        generator_family=str(case.generator_family),
        fallback_family=str(case.fallback_family),
        append_pool_family=str(case.append_pool_family),
    )


def _compile_defaults_for_case(
    case: FixedPVQDBenchmarkCase,
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


def _required_controller_cost_row(source_payload: Mapping[str, Any]) -> overlay.CircuitCostRow:
    row = overlay._source_controller_cost_row(source_payload)
    if row is None:
        raise ValueError("source controller compile reference row is absent")
    _require_finite_cost(row, label="source controller compile reference")
    return row


def _require_finite_cost(row: overlay.CircuitCostRow, *, label: str) -> None:
    for attr in ("compiled_count_2q", "compiled_depth", "compiled_size"):
        value = getattr(row, attr)
        if value is None:
            raise ValueError(f"{label} is missing {attr}")
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} has non-finite {attr}: {value!r}") from None
        if not math.isfinite(numeric):
            raise ValueError(f"{label} has non-finite {attr}: {value!r}")
    if str(row.transpile_status) != "ok":
        raise ValueError(f"{label} compile status is not ok: {row.transpile_status!r} {row.error or ''}")


def _num_qubits_from_state(state: Any) -> int:
    size = int(np.asarray(state, dtype=complex).reshape(-1).size)
    if size <= 0:
        raise ValueError("cannot infer qubit count from an empty reference state")
    nq = int(round(math.log2(size)))
    if 2**nq != size:
        raise ValueError(f"reference state length {size} is not a power of two")
    return nq


def _normalize_state(state: np.ndarray | Sequence[complex]) -> np.ndarray:
    arr = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("cannot normalize zero/non-finite state")
    return np.asarray(arr / norm, dtype=complex).reshape(-1)


def _expectation_hamiltonian(state: np.ndarray, hmat: np.ndarray) -> float:
    psi = _normalize_state(state)
    h = np.asarray(hmat, dtype=complex)
    return float(np.real(np.vdot(psi, h @ psi)))


def _compiled_executor(loaded: fixed.LoadedRunContext) -> CompiledAnsatzExecutor:
    layout = loaded.replay_context.base_layout
    return CompiledAnsatzExecutor(
        list(loaded.replay_context.replay_terms),
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )


def _prepare_fixed_state(
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    theta_runtime: np.ndarray | Sequence[float],
) -> np.ndarray:
    return _normalize_state(
        np.asarray(
            executor.prepare_state(
                np.asarray(theta_runtime, dtype=float).reshape(-1),
                np.asarray(psi_ref, dtype=complex).reshape(-1),
            ),
            dtype=complex,
        ).reshape(-1)
    )


def _build_fixed_context(
    *,
    spec: fixed.FixedManifoldRunSpec,
    case: FixedPVQDBenchmarkCase,
    source_payload: Mapping[str, Any],
) -> overlay.RebuiltOverlayContext:
    loaded = fixed.load_run_context(
        spec,
        tag=f"{case.case_id}_fixed_pvqd_benchmark",
        lock_fixed_manifold=True,
    )
    native_order, coeff_map = hc_pipeline._collect_hardcoded_terms_exyz(loaded.replay_context.h_poly)
    settings = loaded.payload.get("settings", {}) if isinstance(loaded.payload, Mapping) else {}
    term_order = str(_maybe_mapping(settings).get("term_order", "sorted"))
    ordered_labels = list(native_order) if term_order == "native" else sorted(coeff_map)
    hmat = hc_pipeline._build_hamiltonian_matrix(coeff_map)
    psi_initial = _normalize_state(np.asarray(loaded.psi_initial, dtype=complex).reshape(-1))
    nq = _num_qubits_from_state(psi_initial)
    drive_provider, drive_meta, drive_profile = overlay._build_drive_provider(
        source_payload,
        loaded=loaded,
        nq=int(nq),
        ordered_labels_exyz=ordered_labels,
    )
    return overlay.RebuiltOverlayContext(
        loaded=loaded,
        hmat=np.asarray(hmat, dtype=complex),
        ordered_labels_exyz=list(ordered_labels),
        coeff_map_exyz=dict(coeff_map),
        psi_initial=np.asarray(psi_initial, dtype=complex),
        nq=int(nq),
        drive_coeff_provider_exyz=drive_provider,
        drive_profile=drive_profile,
        drive_meta=drive_meta,
    )


def _projection_loss_for_state(psi_trial: np.ndarray, target_state: np.ndarray) -> tuple[float, float]:
    trial = _normalize_state(psi_trial)
    target = _normalize_state(target_state)
    overlap = float(abs(np.vdot(target, trial)) ** 2)
    if not math.isfinite(overlap):
        return float(1.0e12), 0.0
    overlap = min(1.0, max(0.0, float(overlap)))
    return float(max(0.0, 1.0 - overlap)), float(overlap)


def _fit_projection_step(
    *,
    prepare_state: Callable[[np.ndarray], np.ndarray],
    theta_start: np.ndarray | Sequence[float],
    target_state: np.ndarray | Sequence[complex],
    method: str,
    maxiter: int,
    overlap_tol: float,
    ftol: float = 1.0e-10,
) -> PVQDFitResult:
    """Fit a fixed-scaffold parameter vector to a target state by projection loss."""

    theta_init = np.asarray(theta_start, dtype=float).reshape(-1)
    target = _normalize_state(np.asarray(target_state, dtype=complex).reshape(-1))
    eval_count = 0

    def objective(theta_vec: np.ndarray) -> float:
        nonlocal eval_count
        eval_count += 1
        try:
            psi_trial = prepare_state(np.asarray(theta_vec, dtype=float).reshape(-1))
            loss, _overlap = _projection_loss_for_state(psi_trial, target)
        except Exception:
            return float(1.0e12)
        if not math.isfinite(float(loss)):
            return float(1.0e12)
        return float(loss)

    initial_loss = float(objective(theta_init))
    initial_overlap = float(max(0.0, min(1.0, 1.0 - initial_loss)))
    if initial_loss <= float(overlap_tol):
        return PVQDFitResult(
            theta_runtime=np.asarray(theta_init, dtype=float),
            initial_projection_loss=float(initial_loss),
            final_projection_loss=float(initial_loss),
            initial_overlap=float(initial_overlap),
            final_overlap=float(initial_overlap),
            nfev=int(eval_count),
            nit=0,
            success=True,
            status="skipped_overlap_tol",
            message="warm start already satisfies overlap tolerance",
        )

    if scipy_minimize is None:
        theta_best, final_loss, status, nit = _coordinate_refine(
            objective,
            theta_init,
            initial_loss=float(initial_loss),
            maxiter=int(maxiter),
            overlap_tol=float(overlap_tol),
        )
        final_overlap = float(max(0.0, min(1.0, 1.0 - final_loss)))
        return PVQDFitResult(
            theta_runtime=np.asarray(theta_best, dtype=float),
            initial_projection_loss=float(initial_loss),
            final_projection_loss=float(final_loss),
            initial_overlap=float(initial_overlap),
            final_overlap=float(final_overlap),
            nfev=int(eval_count),
            nit=int(nit),
            success=bool(final_loss < initial_loss),
            status=str(status),
            message="SciPy unavailable; used coordinate-refine fallback",
        )

    options: dict[str, Any] = {"maxiter": int(maxiter)}
    lower_method = str(method).strip().lower()
    if lower_method in {"powell", "nelder-mead"}:
        options.update({"ftol": float(ftol), "xtol": float(ftol)})
    elif lower_method in {"bfgs", "l-bfgs-b", "cg"}:
        options.update({"gtol": float(max(ftol, 1.0e-12))})
    result = scipy_minimize(
        objective,
        theta_init,
        method=str(method),
        options=options,
    )
    theta_best = np.asarray(getattr(result, "x", theta_init), dtype=float).reshape(-1)
    final_loss = float(objective(theta_best))
    final_overlap = float(max(0.0, min(1.0, 1.0 - final_loss)))
    success = bool(getattr(result, "success", False)) or bool(final_loss < initial_loss)
    return PVQDFitResult(
        theta_runtime=np.asarray(theta_best, dtype=float),
        initial_projection_loss=float(initial_loss),
        final_projection_loss=float(final_loss),
        initial_overlap=float(initial_overlap),
        final_overlap=float(final_overlap),
        nfev=int(eval_count),
        nit=(None if getattr(result, "nit", None) is None else int(getattr(result, "nit"))),
        success=bool(success),
        status="ok" if bool(success) else "optimizer_not_converged",
        message=str(getattr(result, "message", "")),
    )


def _coordinate_refine(
    objective: Callable[[np.ndarray], float],
    theta_init: np.ndarray,
    *,
    initial_loss: float,
    maxiter: int,
    overlap_tol: float,
) -> tuple[np.ndarray, float, str, int]:
    theta_best = np.asarray(theta_init, dtype=float).reshape(-1).copy()
    best = float(initial_loss)
    step = 0.1
    sweeps = 0
    for sweep in range(max(1, int(maxiter))):
        sweeps = int(sweep) + 1
        improved = False
        for idx in range(int(theta_best.size)):
            for sign in (-1.0, 1.0):
                candidate = theta_best.copy()
                candidate[int(idx)] += float(sign) * float(step)
                loss = float(objective(candidate))
                if loss + 1.0e-15 < best:
                    theta_best = candidate
                    best = float(loss)
                    improved = True
        if best <= float(overlap_tol):
            return theta_best, best, "coordinate_refine_overlap_tol", sweeps
        if not improved:
            step *= 0.5
            if step < 1.0e-8:
                break
    return theta_best, best, "coordinate_refine_done", sweeps


def _apply_reference_step(psi: np.ndarray, hmat: np.ndarray, *, dt: float) -> np.ndarray:
    op = (-1.0j * float(dt)) * np.asarray(hmat, dtype=complex)
    state = np.asarray(psi, dtype=complex).reshape(-1)
    if _csc_matrix is not None and _expm_multiply is not None:
        return _normalize_state(np.asarray(_expm_multiply(_csc_matrix(op), state), dtype=complex).reshape(-1))
    if _dense_expm is not None:
        return _normalize_state(np.asarray(_dense_expm(op) @ state, dtype=complex).reshape(-1))
    evals, evecs = np.linalg.eigh(np.asarray(hmat, dtype=complex))
    coeffs = np.asarray(np.conjugate(evecs).T @ state, dtype=complex).reshape(-1)
    return _normalize_state(np.asarray(evecs @ (np.exp(-1.0j * np.asarray(evals, dtype=float) * float(dt)) * coeffs), dtype=complex))


def _micro_sample_time(
    *,
    t_start: float,
    dt_micro: float,
    micro_index: int,
    drive_t0: float,
    drive_time_sampling: str,
) -> float:
    sampling = str(drive_time_sampling).strip().lower()
    base = float(t_start) + float(micro_index) * float(dt_micro)
    if sampling == "midpoint":
        return float(drive_t0) + float(base) + 0.5 * float(dt_micro)
    if sampling == "left":
        return float(drive_t0) + float(base)
    if sampling == "right":
        return float(drive_t0) + float(base) + float(dt_micro)
    raise ValueError("drive_time_sampling must be one of midpoint, left, right")


def _apply_exact_interval(
    psi: np.ndarray,
    *,
    times: np.ndarray,
    interval_index: int,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    drive_time_sampling: str,
    exact_steps_multiplier: int,
    nq: int,
) -> np.ndarray:
    multiplier = int(exact_steps_multiplier)
    if multiplier < 1:
        raise ValueError("exact_steps_multiplier must be >= 1")
    k = int(interval_index)
    t_start = float(times[k])
    t_stop = float(times[k + 1])
    dt_micro = float(t_stop - t_start) / float(multiplier)
    out = _normalize_state(np.asarray(psi, dtype=complex).reshape(-1))
    for micro in range(multiplier):
        physical_time = _micro_sample_time(
            t_start=float(t_start),
            dt_micro=float(dt_micro),
            micro_index=int(micro),
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_time_sampling),
        )
        hmat_total = overlay._hmat_total_at_observation(
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            physical_time=float(physical_time),
            nq=int(nq),
        )
        out = _apply_reference_step(out, np.asarray(hmat_total, dtype=complex), dt=float(dt_micro))
    return _normalize_state(out)


def _build_exact_reference_states(
    *,
    psi_initial: np.ndarray,
    times: np.ndarray,
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    drive_time_sampling: str,
    exact_steps_multiplier: int,
    nq: int,
) -> list[np.ndarray]:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if times_arr.size <= 0:
        return []
    psi = _normalize_state(np.asarray(psi_initial, dtype=complex).reshape(-1))
    out = [np.array(psi, copy=True)]
    for k in range(int(times_arr.size) - 1):
        psi = _apply_exact_interval(
            psi,
            times=times_arr,
            interval_index=int(k),
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_time_sampling),
            exact_steps_multiplier=int(exact_steps_multiplier),
            nq=int(nq),
        )
        out.append(np.array(psi, copy=True))
    return out


def _summarize_trajectory(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty fixed-pVQD trajectory")
    energies = [_required_finite_float(row.get("energy_total"), field="energy_total") for row in rows]
    exact_values = [_maybe_float(row.get("energy_total_exact")) for row in rows]
    errors = [_maybe_float(row.get("abs_energy_total_error")) for row in rows]
    finite_errors = [float(x) for x in errors if x is not None]
    fidelities = [_maybe_float(row.get("fidelity_exact")) for row in rows]
    finite_fidelities = [float(x) for x in fidelities if x is not None]
    nfev_total = sum(int(row.get("pvqd_nfev", 0) or 0) for row in rows)
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
        "pvqd_nfev_total": int(nfev_total),
        "pvqd_step_count": int(step_count),
    }


def _exact_steps_multiplier_from_source(source_payload: Mapping[str, Any]) -> int:
    drive_cfg = _maybe_mapping(source_payload.get("drive_config", {}))
    reference = _maybe_mapping(source_payload.get("reference", {}))
    raw = drive_cfg.get("exact_steps_multiplier", reference.get("reference_steps_multiplier", 1))
    value = _maybe_int(raw)
    if value is None or int(value) < 1:
        raise ValueError(f"source exact_steps_multiplier/reference_steps_multiplier must be >=1; got {raw!r}")
    return int(value)


def _simulate_fixed_pvqd(
    *,
    case: FixedPVQDBenchmarkCase,
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    exact_energy_total: Sequence[float] | None,
    observation_physical_times: Sequence[float],
    exact_steps_multiplier: int,
) -> FixedPVQDSimulationResult:
    times_arr = np.asarray(times, dtype=float).reshape(-1)
    if int(times_arr.size) < 2:
        raise ValueError("fixed-pVQD requires at least two time points")
    obs_physical = np.asarray(observation_physical_times, dtype=float).reshape(-1)
    if int(obs_physical.size) != int(times_arr.size):
        raise ValueError("observation_physical_times must match source time grid")
    exact_arr = None if exact_energy_total is None else np.asarray(exact_energy_total, dtype=float).reshape(-1)
    if exact_arr is not None and int(exact_arr.size) != int(times_arr.size):
        raise ValueError("exact_energy_total must match source time grid")

    loaded = context.loaded
    executor = _compiled_executor(loaded)
    psi_ref = np.asarray(loaded.replay_context.psi_ref, dtype=complex).reshape(-1)
    theta = np.asarray(loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1)
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))

    def prepare(theta_vec: np.ndarray) -> np.ndarray:
        return _prepare_fixed_state(executor, psi_ref, np.asarray(theta_vec, dtype=float).reshape(-1))

    current_state = prepare(theta)
    exact_states = _build_exact_reference_states(
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

    def _append_row(idx: int, state: np.ndarray, *, fit: PVQDFitResult | None) -> None:
        hmat_total = overlay._hmat_total_at_observation(
            hmat_static=np.asarray(context.hmat, dtype=complex),
            drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
            physical_time=float(obs_physical[int(idx)]),
            nq=int(context.nq),
        )
        energy = _expectation_hamiltonian(state, np.asarray(hmat_total, dtype=complex))
        exact_state = _normalize_state(np.asarray(exact_states[int(idx)], dtype=complex).reshape(-1))
        exact_energy = (
            float(exact_arr[int(idx)])
            if exact_arr is not None
            else _expectation_hamiltonian(exact_state, np.asarray(hmat_total, dtype=complex))
        )
        fidelity = float(abs(np.vdot(exact_state, _normalize_state(state))) ** 2)
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
            "runtime_parameter_count": int(theta.size),
            "logical_block_count": int(loaded.replay_context.base_layout.logical_parameter_count),
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
                    "pvqd_nfev": int(fit.nfev),
                    "projection_loss_initial": float(fit.initial_projection_loss),
                    "projection_loss_final": float(fit.final_projection_loss),
                    "projection_overlap_initial": float(fit.initial_overlap),
                    "projection_overlap_final": float(fit.final_overlap),
                    "optimizer_status": str(fit.status),
                    "optimizer_success": bool(fit.success),
                }
            )
        trajectory.append(row)

    _append_row(0, current_state, fit=None)
    for k in range(int(times_arr.size) - 1):
        target = _apply_exact_interval(
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
        fit = _fit_projection_step(
            prepare_state=prepare,
            theta_start=theta,
            target_state=target,
            method=str(case.optimizer_method),
            maxiter=int(case.optimizer_maxiter),
            overlap_tol=float(case.overlap_tol),
            ftol=float(case.optimizer_ftol),
        )
        theta = np.asarray(fit.theta_runtime, dtype=float).reshape(-1)
        current_state = prepare(theta)
        step_payload = {
            "interval_index": int(k),
            "time_start": float(times_arr[int(k)]),
            "time_stop": float(times_arr[int(k) + 1]),
            "dt": float(times_arr[int(k) + 1] - times_arr[int(k)]),
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
        }
        pvqd_steps.append(step_payload)
        _append_row(k + 1, current_state, fit=fit)

    summary = _summarize_trajectory(trajectory)
    return FixedPVQDSimulationResult(
        method=METHOD_ID,
        trajectory=trajectory,
        summary=summary,
        final_state=np.asarray(current_state, dtype=complex).reshape(-1),
        pvqd_steps=pvqd_steps,
        exact_reference_summary={
            "state_count": int(len(exact_states)),
            "reference_policy": "benchmark-local exact interval propagation for fidelity diagnostics",
            "exact_steps_multiplier": int(exact_steps_multiplier),
        },
    )


def _compile_fixed_state_scaffold(
    *,
    context: overlay.RebuiltOverlayContext,
    compile_defaults: Mapping[str, Any],
) -> tuple[overlay.CircuitCostRow, list[dict[str, Any]]]:
    scaffold_circuit = overlay.build_ansatz_circuit(
        context.loaded.replay_context.base_layout,
        np.asarray(context.loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        int(context.nq),
        ref_state=np.asarray(context.loaded.replay_context.psi_ref, dtype=complex).reshape(-1),
    )
    cost, raw_rows = overlay._compile_one_circuit_cost(
        method="fixed_pvqd",
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
    _require_finite_cost(cost, label="fixed-pVQD state scaffold compile row")
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
            "method": "fixed_pvqd",
            "group": "state_at_time",
            "scope": STATE_SCOPE,
            "basis": "fixed pVQD state scaffold",
            "compiled_count_2q": int(state_2q),
            "compiled_depth": int(state_depth),
            "compiled_size": int(state_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": state_cost.scope,
        },
        {
            "method": "fixed_pvqd",
            "group": "horizon",
            "scope": HORIZON_SCOPE,
            "basis": f"{int(intervals)} repeated fixed pVQD state scaffolds",
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
    case: FixedPVQDBenchmarkCase,
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
        "ansatz_types": "ADAPT seed prep; fixed pVQD projection on frozen scaffold",
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
    case: FixedPVQDBenchmarkCase,
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
    state_cost = _required_report_row(hardware_rows, method="fixed_pvqd", scope=STATE_SCOPE)
    full_cost = _required_report_row(hardware_rows, method="fixed_pvqd", scope=HORIZON_SCOPE)
    controller_cost = _required_report_row(hardware_rows, method="controller", scope=CONTROLLER_STATE_SCOPE)
    intervals = _maybe_int(full_cost.get("intervals"))
    if intervals is None:
        num_times = _maybe_int(manifest.get("num_times"))
        intervals = None if num_times is None else max(int(num_times) - 1, 0)

    row = FixedPVQDBenchmarkRow(
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
        pvqd_nfev_total=_required_int(summary.get("pvqd_nfev_total"), field="summary.pvqd_nfev_total"),
        pvqd_step_count=_required_int(summary.get("pvqd_step_count"), field="summary.pvqd_step_count"),
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
    case: FixedPVQDBenchmarkCase,
    spec: fixed.FixedManifoldRunSpec,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    simulation: FixedPVQDSimulationResult,
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
                "optimizer_method": str(case.optimizer_method),
                "optimizer_maxiter": int(case.optimizer_maxiter),
                "overlap_tol": float(case.overlap_tol),
                "optimizer_ftol": float(case.optimizer_ftol),
                "target_policy": "exact_interval_evolution_of_current_fixed_pvqd_state",
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
                "compile_cost_policy": "compile one representative fixed scaffold; repeated-horizon budget multiplies by num_times-1 intervals",
                "controller_reference_policy": "fail_closed_required_source_compile_reference",
            },
            "trajectory": simulation.trajectory,
            "summary": summary,
            "pvqd_steps": simulation.pvqd_steps,
            "exact_reference_summary": simulation.exact_reference_summary,
            "hardware_report_rows": hardware_report_rows,
            "circuit_costs": [_jsonable(state_cost), _jsonable(controller_cost)],
            "raw_compile_rows": {
                "fixed_pvqd_state_scaffold": list(raw_compile_rows),
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
    case: FixedPVQDBenchmarkCase,
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
    controller_cost = _required_controller_cost_row(source_payload)
    spec = _run_spec_for_case(case)
    context = _build_fixed_context(spec=spec, case=case, source_payload=source_payload)
    exact_steps_multiplier = _exact_steps_multiplier_from_source(source_payload)
    simulation = _simulate_fixed_pvqd(
        case=case,
        context=context,
        times=times,
        exact_energy_total=exact_energy,
        observation_physical_times=physical_times,
        exact_steps_multiplier=int(exact_steps_multiplier),
    )
    state_cost, raw_compile_rows = _compile_fixed_state_scaffold(
        context=context,
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
        "benchmark": "hh_fixed_pvqd_time_dynamics",
        "method_contract": {
            "method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "decision_mode": DECISION_MODE,
            "diagnostic_exact_assisted": True,
            "qpu_faithful": False,
            "default_case_id": DEFAULT_CASE_ID,
            "seed_family": SEED_FAMILY,
            "target_policy": "exact_interval_evolution_of_current_fixed_pvqd_state",
            "compile_cost_policy": (
                "compile one representative fixed state scaffold; repeated-horizon budget is "
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
        "benchmark": "hh_fixed_pvqd_time_dynamics",
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
    cases: Sequence[FixedPVQDBenchmarkCase],
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
        description="Run the benchmark-local HH L2 t=8 fixed-pVQD row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-artifact-json", type=Path, default=None)
    parser.add_argument("--optimizer-method", type=str, default=None)
    parser.add_argument("--optimizer-maxiter", type=int, default=None)
    parser.add_argument("--overlap-tol", type=float, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> FixedPVQDBenchmarkCase:
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
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_fixed_pvqd_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_fixed_pvqd_benchmark", *map(str, argv)])


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
