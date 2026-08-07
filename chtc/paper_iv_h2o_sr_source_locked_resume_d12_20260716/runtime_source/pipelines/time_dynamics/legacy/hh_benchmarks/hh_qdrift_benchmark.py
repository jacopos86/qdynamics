#!/usr/bin/env python3
"""Benchmark-local qDRIFT dynamics row for the HH L=2 t=8 anchor.

This module is intentionally read-only relative to controller logic.  It rebuilds
validated HH source context through ``hh_realtime_suzuki_overlay`` helpers,
simulates a deterministic qDRIFT/randomized-product-formula trajectory locally,
and emits benchmark row artifacts.  Exact/reference fields are used only after
the qDRIFT trajectory energy is computed.
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
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.hardcoded import hubbard_pipeline as hc_pipeline
from pipelines.hardcoded.adapt_circuit_execution import (
    append_pauli_rotation_exyz,
    build_ansatz_circuit,
)
from pipelines.time_dynamics.legacy.analysis import hh_realtime_suzuki_overlay as overlay


SCHEMA_VERSION = "hh_qdrift_benchmark_v1"
RUN_SCHEMA_VERSION = "hh_qdrift_run_v1"
DEFAULT_CASE_ID = "hh_l2_t8_anchor_v1"
METHOD_ID = "hh_td_qdrift_s16_v1"
DEFAULT_SAMPLES_PER_INTERVAL = 16
DEFAULT_RNG_SEED = 7
METHOD_KIND = "randomized_product_formula"
RANDOMIZATION_FAMILY = "qdrift"
STATE_SCOPE = "seed_plus_one_step_additive"
INTERVAL_SCOPE = "representative_interval0_evolution_only"
FULL_HORIZON_SCOPE = "full_horizon_with_seed_prep"
CONTROLLER_STATE_SCOPE = "controller_state_at_time"
CONTROLLER_SOURCE_SCOPE = "controller_final_scaffold_source"


@dataclass(frozen=True)
class QDriftBenchmarkCase:
    case_id: str
    controller_json: Path
    source_pdf: Path
    trotter_steps: int
    samples_per_interval: int
    rng_seed: int
    backend_name: str | None = None
    seed_transpiler: int | None = None
    optimization_level: int | None = None
    preferred_fake_backends: tuple[str, ...] = ()


@dataclass(frozen=True)
class QDriftSimulationResult:
    method: str
    trajectory: list[dict[str, Any]]
    summary: dict[str, Any]
    final_state: np.ndarray
    intervals: list[dict[str, Any]]


@dataclass(frozen=True)
class QDriftBenchmarkRow:
    case_id: str
    method_id: str
    method_kind: str
    status: str
    randomization_family: str
    samples_per_interval: int
    rng_seed: int
    controller_json: str | None
    source_pdf: str | None
    seed_artifact_json: str | None
    drive_enabled: bool | None
    t_final: float | None
    num_times: int | None
    trotter_steps: int | None
    final_energy_total: float | None
    final_energy_total_exact: float | None
    final_abs_energy_total_error: float | None
    mean_abs_energy_total_error: float | None
    max_abs_energy_total_error: float | None
    state_at_time_scope: str
    state_at_time_basis: str | None
    state_at_time_2q: int | None
    state_at_time_depth: int | None
    state_at_time_size: int | None
    full_horizon_scope: str
    full_horizon_basis: str | None
    full_horizon_2q: int | None
    full_horizon_depth: int | None
    full_horizon_size: int | None
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
    exact_reference_method: str | None
    exact_steps_multiplier: Any
    artifact_run_json: str | None
    artifact_manifest_json: str | None
    artifact_rows_json: str | None
    artifact_summary_json: str | None
    exact_fields_reporting_only: bool = True
    controller_decisions_modified: bool = False


@dataclass(frozen=True)
class _CaseRunRecord:
    case: QDriftBenchmarkCase
    run_json: Path
    run_artifact: Mapping[str, Any]
    row: dict[str, Any]
    compile_defaults: Mapping[str, Any]


"Built Math: qDRIFT interval U_k≈Π_m exp(-i s_{k,m} λ_k Δt/N P_{j_{k,m}}), j sampled by |a_j(t_k)|/λ_k; exact fields are appended after trajectory energies only."
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


def _sum_required_int(*values: Any, field: str) -> int:
    return int(sum(_required_int(value, field=field) for value in values))


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = [str(x) for x in raw]
    return tuple(part.strip() for part in parts if part.strip())


def method_id_for_config(*, samples_per_interval: int, rng_seed: int) -> str:
    """Return a stable qDRIFT method id for one stochastic configuration."""

    samples = int(samples_per_interval)
    seed = int(rng_seed)
    if samples == int(DEFAULT_SAMPLES_PER_INTERVAL) and seed == int(DEFAULT_RNG_SEED):
        return METHOD_ID
    return f"hh_td_qdrift_s{samples}_seed{seed}_v1"


def default_cases() -> tuple[QDriftBenchmarkCase, ...]:
    return (
        QDriftBenchmarkCase(
            case_id=DEFAULT_CASE_ID,
            controller_json=overlay.DEFAULT_CONTROLLER_JSON,
            source_pdf=overlay.DEFAULT_SOURCE_PDF,
            trotter_steps=160,
            samples_per_interval=DEFAULT_SAMPLES_PER_INTERVAL,
            rng_seed=DEFAULT_RNG_SEED,
        ),
    )


def _case_by_id(case_id: str) -> QDriftBenchmarkCase:
    for case in default_cases():
        if case.case_id == case_id:
            return case
    known = ", ".join(case.case_id for case in default_cases())
    raise ValueError(f"unknown qDRIFT benchmark case_id={case_id!r}; known cases: {known}")


def _compile_defaults_for_case(
    case: QDriftBenchmarkCase,
    source_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    need_source_defaults = (
        case.backend_name is None
        or case.seed_transpiler is None
        or case.optimization_level is None
        or not case.preferred_fake_backends
    )
    if need_source_defaults:
        source = (
            overlay._load_source_payload(Path(case.controller_json))
            if source_payload is None
            else source_payload
        )
        defaults = dict(overlay._source_compile_defaults(source))
    else:
        defaults = {}
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


def _summarize_energy_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty qDRIFT trajectory")
    energies = [_required_finite_float(row.get("energy_total"), field="energy_total") for row in rows]
    exact_values = [row.get("energy_total_exact") for row in rows]
    exact_finite = [_maybe_float(value) for value in exact_values]
    errors = [_maybe_float(row.get("abs_energy_total_error")) for row in rows]
    finite_errors = [float(x) for x in errors if x is not None]
    return {
        "row_count": int(len(rows)),
        "final_energy_total": float(energies[-1]),
        "final_energy_total_exact": exact_finite[-1] if exact_finite else None,
        "final_abs_energy_total_error": None if not finite_errors else errors[-1],
        "mean_abs_energy_total_error": None if not finite_errors else float(sum(finite_errors) / len(finite_errors)),
        "max_abs_energy_total_error": None if not finite_errors else float(max(finite_errors)),
    }


def _probability_summary(
    *,
    labels: Sequence[str],
    coefficients: Sequence[float],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for label, coeff, prob in zip(labels, coefficients, probabilities):
        if abs(float(coeff)) <= 0.0 and abs(float(prob)) <= 0.0:
            continue
        out.append(
            {
                "label_exyz": str(label),
                "coefficient_real": float(coeff),
                "probability": float(prob),
            }
        )
    return out


def _simulate_qdrift(
    *,
    psi_initial: np.ndarray,
    times: np.ndarray,
    exact_energy_total: Sequence[float] | None,
    observation_physical_times: Sequence[float],
    ordered_labels_exyz: Sequence[str],
    coeff_map_exyz: Mapping[str, complex],
    hmat_static: np.ndarray,
    drive_coeff_provider_exyz: Any | None,
    drive_t0: float,
    drive_time_sampling: str,
    nq: int,
    samples_per_interval: int,
    rng_seed: int,
    method_id: str | None = None,
    coeff_imag_tol: float = 1.0e-12,
    lambda_tol: float = 1.0e-15,
) -> QDriftSimulationResult:
    """Simulate qDRIFT with one RNG consumed in interval order.

    Exact energies are accepted only for post-trajectory reporting rows; sampling,
    state updates, and circuit provenance depend solely on coefficients and the
    fixed RNG seed.
    """

    times_arr = np.asarray(times, dtype=float)
    trotter_steps = int(times_arr.size) - 1
    dt = overlay._uniform_dt(times_arr, trotter_steps)
    obs_physical = np.asarray(observation_physical_times, dtype=float)
    if obs_physical.size != int(times_arr.size):
        raise ValueError("observation_physical_times must match source time grid")
    exact_arr = None if exact_energy_total is None else np.asarray(exact_energy_total, dtype=float)
    if exact_arr is not None and exact_arr.size != int(times_arr.size):
        raise ValueError("exact_energy_total must match source time grid")
    if int(samples_per_interval) <= 0:
        raise ValueError("samples_per_interval must be positive")
    method_label = str(
        method_id
        or method_id_for_config(
            samples_per_interval=int(samples_per_interval),
            rng_seed=int(rng_seed),
        )
    )

    labels = [str(label) for label in ordered_labels_exyz]
    compiled = {label: hc_pipeline._compile_pauli_action(label, int(nq)) for label in labels}
    rng = np.random.default_rng(int(rng_seed))
    psi = hc_pipeline._normalize_state(np.asarray(psi_initial, dtype=complex).reshape(-1))

    trajectory: list[dict[str, Any]] = []
    interval_records: list[dict[str, Any]] = []

    def _append_row(idx: int, state: np.ndarray) -> None:
        hmat_total = overlay._hmat_total_at_observation(
            hmat_static=np.asarray(hmat_static, dtype=complex),
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            physical_time=float(obs_physical[int(idx)]),
            nq=int(nq),
        )
        energy = float(hc_pipeline._expectation_hamiltonian(state, hmat_total))
        exact_energy = None if exact_arr is None else float(exact_arr[int(idx)])
        err = None if exact_energy is None else float(abs(energy - exact_energy))
        trajectory.append(
            {
                "checkpoint_index": int(idx),
                "time": float(times_arr[int(idx)]),
                "physical_time": float(obs_physical[int(idx)]),
                "method": method_label,
                "randomization_family": RANDOMIZATION_FAMILY,
                "samples_per_interval": int(samples_per_interval),
                "rng_seed": int(rng_seed),
                "energy_total": float(energy),
                "energy_total_exact": exact_energy,
                "abs_energy_total_error": err,
                "state_norm": float(np.linalg.norm(state)),
            }
        )

    _append_row(0, psi)
    for k in range(trotter_steps):
        coeffs = overlay._coeff_at_step(
            k=int(k),
            dt=float(dt),
            coeff_map_exyz=coeff_map_exyz,
            ordered_labels_exyz=labels,
            drive_coeff_provider_exyz=drive_coeff_provider_exyz,
            drive_t0=float(drive_t0),
            drive_time_sampling=str(drive_time_sampling),
        )
        real_coeffs: list[float] = []
        for label in labels:
            coeff = complex(coeffs[str(label)])
            if abs(float(np.imag(coeff))) > float(coeff_imag_tol):
                raise ValueError(f"qDRIFT requires real coefficients; {label} has {coeff!r}")
            real_coeffs.append(float(np.real(coeff)))
        lambda_k = float(sum(abs(value) for value in real_coeffs))
        sampled_labels: list[str] = []
        sampled_signs: list[float] = []
        sampled_probabilities: list[float] = []
        tau = 0.0 if lambda_k <= float(lambda_tol) else float(lambda_k) * float(dt) / float(samples_per_interval)
        if lambda_k > float(lambda_tol):
            probabilities = np.asarray([abs(value) / lambda_k for value in real_coeffs], dtype=float)
            sampled_indices = rng.choice(len(labels), size=int(samples_per_interval), replace=True, p=probabilities)
            for raw_idx in sampled_indices:
                idx = int(raw_idx)
                label = labels[idx]
                coeff_real = float(real_coeffs[idx])
                sign = 1.0 if coeff_real >= 0.0 else -1.0
                sampled_labels.append(label)
                sampled_signs.append(float(sign))
                sampled_probabilities.append(float(probabilities[idx]))
                psi = hc_pipeline._apply_exp_term(psi, compiled[label], complex(sign), float(tau))
        else:
            probabilities = np.zeros(len(labels), dtype=float)
        psi = hc_pipeline._normalize_state(psi)
        counts = Counter(sampled_labels)
        interval_records.append(
            {
                "interval_index": int(k),
                "time_start": float(times_arr[int(k)]),
                "time_stop": float(times_arr[int(k) + 1]),
                "dt": float(dt),
                "lambda": float(lambda_k),
                "tau": float(tau),
                "samples_per_interval": int(samples_per_interval),
                "sampled_labels": sampled_labels,
                "sampled_signs": sampled_signs,
                "sampled_probabilities": sampled_probabilities,
                "sampled_counts": {str(key): int(value) for key, value in sorted(counts.items())},
                "probability_summary": _probability_summary(
                    labels=labels,
                    coefficients=real_coeffs,
                    probabilities=probabilities.tolist(),
                ),
            }
        )
        _append_row(k + 1, psi)

    return QDriftSimulationResult(
        method=method_label,
        trajectory=trajectory,
        summary=_summarize_energy_rows(trajectory),
        final_state=np.asarray(psi, dtype=complex).reshape(-1),
        intervals=interval_records,
    )


def _build_seed_circuit(context: overlay.RebuiltOverlayContext) -> Any:
    return build_ansatz_circuit(
        context.loaded.replay_context.base_layout,
        np.asarray(context.loaded.replay_context.adapt_theta_runtime, dtype=float).reshape(-1),
        int(context.nq),
        ref_state=np.asarray(context.loaded.replay_context.psi_ref, dtype=complex).reshape(-1),
    )


def _append_qdrift_sequence_to_circuit(qc: Any, interval_record: Mapping[str, Any]) -> None:
    tau = float(interval_record.get("tau", 0.0))
    labels = list(interval_record.get("sampled_labels", []))
    signs = list(interval_record.get("sampled_signs", []))
    if len(labels) != len(signs):
        raise ValueError("qDRIFT interval sampled_labels/sampled_signs length mismatch")
    for label, sign in zip(labels, signs):
        append_pauli_rotation_exyz(
            qc,
            label_exyz=str(label),
            angle=2.0 * float(sign) * float(tau),
        )


def _build_qdrift_interval_circuit(
    *,
    nq: int,
    interval_record: Mapping[str, Any],
    include_seed_prep: bool = False,
    seed_circuit: Any | None = None,
) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(int(nq))
    if bool(include_seed_prep):
        if seed_circuit is None:
            raise ValueError("include_seed_prep requires seed_circuit")
        qc.compose(seed_circuit, inplace=True)
    _append_qdrift_sequence_to_circuit(qc, interval_record)
    return qc


def _build_qdrift_full_horizon_circuit(
    *,
    nq: int,
    intervals: Sequence[Mapping[str, Any]],
    seed_circuit: Any,
) -> Any:
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(int(nq))
    qc.compose(seed_circuit, inplace=True)
    for interval in intervals:
        _append_qdrift_sequence_to_circuit(qc, interval)
    return qc


def _compile_qdrift_costs(
    *,
    case: QDriftBenchmarkCase,
    context: overlay.RebuiltOverlayContext,
    source_payload: Mapping[str, Any],
    simulation: QDriftSimulationResult,
    compile_defaults: Mapping[str, Any],
) -> tuple[list[overlay.CircuitCostRow], dict[str, Any]]:
    controller_cost = _required_controller_cost_row(source_payload)
    seed_circuit = _build_seed_circuit(context)
    raw_compile_rows: dict[str, Any] = {
        "controller": [
            {
                "method": "controller",
                "scope": controller_cost.scope,
                "selected": _jsonable(controller_cost),
                "raw_rows": [],
            }
        ],
        "seed": [],
        "qdrift": [],
    }

    seed_cost, seed_raw = overlay._compile_one_circuit_cost(
        method="seed",
        order=None,
        scope="seed_prep_only",
        trotter_steps=None,
        includes_seed_prep=True,
        circuit=seed_circuit,
        backend_name=str(compile_defaults["backend_name"]),
        preferred_fake_backends=tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        seed_transpiler=int(compile_defaults["seed_transpiler"]),
        optimization_level=int(compile_defaults["optimization_level"]),
    )
    _require_finite_cost(seed_cost, label="seed-prep compile row")
    raw_compile_rows["seed"].append(
        {"method": "seed", "scope": seed_cost.scope, "selected": _jsonable(seed_cost), "raw_rows": seed_raw}
    )

    if not simulation.intervals:
        raise ValueError("qDRIFT simulation produced no intervals for compile-cost audit")
    interval0 = simulation.intervals[0]
    interval_circuit = _build_qdrift_interval_circuit(
        nq=int(context.nq),
        interval_record=interval0,
        include_seed_prep=False,
    )
    interval_cost, interval_raw = overlay._compile_one_circuit_cost(
        method="qdrift",
        order=None,
        scope=INTERVAL_SCOPE,
        trotter_steps=1,
        includes_seed_prep=False,
        circuit=interval_circuit,
        backend_name=str(compile_defaults["backend_name"]),
        preferred_fake_backends=tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        seed_transpiler=int(compile_defaults["seed_transpiler"]),
        optimization_level=int(compile_defaults["optimization_level"]),
    )
    _require_finite_cost(interval_cost, label="representative interval-0 qDRIFT compile row")
    raw_compile_rows["qdrift"].append(
        {
            "method": "qdrift",
            "scope": interval_cost.scope,
            "interval_index": 0,
            "selected": _jsonable(interval_cost),
            "raw_rows": interval_raw,
        }
    )

    full_circuit = _build_qdrift_full_horizon_circuit(
        nq=int(context.nq),
        intervals=simulation.intervals,
        seed_circuit=seed_circuit,
    )
    full_cost, full_raw = overlay._compile_one_circuit_cost(
        method="qdrift",
        order=None,
        scope=FULL_HORIZON_SCOPE,
        trotter_steps=int(case.trotter_steps),
        includes_seed_prep=True,
        circuit=full_circuit,
        backend_name=str(compile_defaults["backend_name"]),
        preferred_fake_backends=tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
        seed_transpiler=int(compile_defaults["seed_transpiler"]),
        optimization_level=int(compile_defaults["optimization_level"]),
    )
    _require_finite_cost(full_cost, label="full-horizon qDRIFT compile row")
    raw_compile_rows["qdrift"].append(
        {
            "method": "qdrift",
            "scope": full_cost.scope,
            "selected": _jsonable(full_cost),
            "raw_rows": full_raw,
        }
    )

    raw_compile_rows["time_grid"] = {
        "trotter_steps": int(case.trotter_steps),
        "interval_count": int(len(simulation.intervals)),
        "samples_per_interval": int(case.samples_per_interval),
    }
    return [controller_cost, seed_cost, interval_cost, full_cost], raw_compile_rows


def _method_cost(
    cost_rows: Sequence[overlay.CircuitCostRow],
    method: str,
    scope: str,
) -> overlay.CircuitCostRow | None:
    for row in cost_rows:
        if str(row.method) == str(method) and str(row.scope) == str(scope):
            return row
    return None


def _required_method_cost(
    cost_rows: Sequence[overlay.CircuitCostRow],
    *,
    method: str,
    scope: str,
) -> overlay.CircuitCostRow:
    row = _method_cost(cost_rows, method, scope)
    if row is None:
        raise ValueError(f"required compile cost missing: method={method!r} scope={scope!r}")
    _require_finite_cost(row, label=f"compile cost method={method!r} scope={scope!r}")
    return row


def _hardware_cost_rows(cost_rows: Sequence[overlay.CircuitCostRow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in cost_rows:
        rows.append(
            {
                "method": str(row.method),
                "scope": str(row.scope),
                "trotter_steps": row.trotter_steps,
                "includes_seed_prep": bool(row.includes_seed_prep),
                "abstract_size": row.abstract_size,
                "abstract_depth": row.abstract_depth,
                "compiled_count_2q": row.compiled_count_2q,
                "compiled_depth": row.compiled_depth,
                "compiled_size": row.compiled_size,
                "compiled_num_qubits": row.compiled_num_qubits,
                "compiled_op_counts": dict(row.compiled_op_counts),
                "backend_name": row.backend_name,
                "seed_transpiler": row.seed_transpiler,
                "optimization_level": row.optimization_level,
                "transpile_status": row.transpile_status,
                "error": row.error,
            }
        )
    return rows


def _hardware_report_rows(cost_rows: Sequence[overlay.CircuitCostRow]) -> list[dict[str, Any]]:
    controller = _required_method_cost(cost_rows, method="controller", scope=CONTROLLER_SOURCE_SCOPE)
    seed = _required_method_cost(cost_rows, method="seed", scope="seed_prep_only")
    interval = _required_method_cost(cost_rows, method="qdrift", scope=INTERVAL_SCOPE)
    full = _required_method_cost(cost_rows, method="qdrift", scope=FULL_HORIZON_SCOPE)
    additive_2q = _sum_required_int(seed.compiled_count_2q, interval.compiled_count_2q, field="state_at_time_2q")
    additive_depth = _sum_required_int(seed.compiled_depth, interval.compiled_depth, field="state_at_time_depth")
    additive_size = _sum_required_int(seed.compiled_size, interval.compiled_size, field="state_at_time_size")
    return [
        {
            "method": "controller",
            "group": "state_at_time",
            "scope": CONTROLLER_STATE_SCOPE,
            "basis": "controller state-at-time compile reference",
            "compiled_count_2q": int(controller.compiled_count_2q),
            "compiled_depth": int(controller.compiled_depth),
            "compiled_size": int(controller.compiled_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": controller.scope,
        },
        {
            "method": "qdrift",
            "group": "state_at_time",
            "scope": "seed_prep_only",
            "basis": "seed prep only",
            "compiled_count_2q": int(seed.compiled_count_2q),
            "compiled_depth": int(seed.compiled_depth),
            "compiled_size": int(seed.compiled_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": seed.scope,
        },
        {
            "method": "qdrift",
            "group": "state_at_time",
            "scope": INTERVAL_SCOPE,
            "basis": "representative interval-0 qDRIFT evolution only",
            "compiled_count_2q": int(interval.compiled_count_2q),
            "compiled_depth": int(interval.compiled_depth),
            "compiled_size": int(interval.compiled_size),
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": interval.scope,
        },
        {
            "method": "qdrift",
            "group": "state_at_time",
            "scope": STATE_SCOPE,
            "basis": "seed prep + representative interval-0 qDRIFT additive",
            "compiled_count_2q": additive_2q,
            "compiled_depth": additive_depth,
            "compiled_size": additive_size,
            "horizon_count_2q": None,
            "horizon_depth_serial": None,
            "source_scope": "seed_prep_only + representative_interval0_evolution_only",
        },
        {
            "method": "qdrift",
            "group": "horizon",
            "scope": FULL_HORIZON_SCOPE,
            "basis": "compiled seed prep + all qDRIFT microsteps",
            "compiled_count_2q": int(full.compiled_count_2q),
            "compiled_depth": int(full.compiled_depth),
            "compiled_size": int(full.compiled_size),
            "horizon_count_2q": int(full.compiled_count_2q),
            "horizon_depth_serial": int(full.compiled_depth),
            "source_scope": full.scope,
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
    case: QDriftBenchmarkCase,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    compile_defaults: Mapping[str, Any],
    output_dir: Path,
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
        "ansatz_types": "ADAPT seed prep; qDRIFT randomized product formula",
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
        "trotter_steps": int(case.trotter_steps),
        "samples_per_interval": int(case.samples_per_interval),
        "rng_seed": int(case.rng_seed),
        "randomization_family": RANDOMIZATION_FAMILY,
        "exact_reference_method": _as_optional_str(reference.get("reference_method")),
        "exact_steps_multiplier": drive_cfg.get("exact_steps_multiplier", reference.get("reference_steps_multiplier")),
        "compile_backend": str(compile_defaults["backend_name"]),
        "compile_seed_transpiler": int(compile_defaults["seed_transpiler"]),
        "compile_optimization_level": int(compile_defaults["optimization_level"]),
        "controller_json": str(case.controller_json),
        "source_pdf": str(case.source_pdf),
        "seed_artifact_json": _as_optional_str(source_payload.get("artifact_json")),
        "output_dir": str(output_dir),
    }


def _row_from_run_artifact(
    payload: Mapping[str, Any],
    *,
    case: QDriftBenchmarkCase,
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
    state_cost = _required_report_row(hardware_rows, method="qdrift", scope=STATE_SCOPE)
    full_cost = _required_report_row(hardware_rows, method="qdrift", scope=FULL_HORIZON_SCOPE)
    controller_cost = _required_report_row(hardware_rows, method="controller", scope=CONTROLLER_STATE_SCOPE)

    row = QDriftBenchmarkRow(
        case_id=str(case.case_id),
        method_id=str(
            payload.get("method_id")
            or method_id_for_config(
                samples_per_interval=int(case.samples_per_interval),
                rng_seed=int(case.rng_seed),
            )
        ),
        method_kind=METHOD_KIND,
        status="ok",
        randomization_family=RANDOMIZATION_FAMILY,
        samples_per_interval=int(case.samples_per_interval),
        rng_seed=int(case.rng_seed),
        controller_json=_as_optional_str(manifest.get("controller_json") or source.get("controller_json")),
        source_pdf=_as_optional_str(manifest.get("source_pdf") or source.get("source_pdf")),
        seed_artifact_json=_as_optional_str(manifest.get("seed_artifact_json") or source.get("artifact_json")),
        drive_enabled=_maybe_bool(manifest.get("drive_enabled")),
        t_final=_maybe_float(manifest.get("t_final")),
        num_times=_maybe_int(manifest.get("num_times")),
        trotter_steps=_maybe_int(manifest.get("trotter_steps")),
        final_energy_total=_required_finite_float(summary.get("final_energy_total"), field="summary.final_energy_total"),
        final_energy_total_exact=_required_finite_float(summary.get("final_energy_total_exact"), field="summary.final_energy_total_exact"),
        final_abs_energy_total_error=_required_finite_float(summary.get("final_abs_energy_total_error"), field="summary.final_abs_energy_total_error"),
        mean_abs_energy_total_error=_required_finite_float(summary.get("mean_abs_energy_total_error"), field="summary.mean_abs_energy_total_error"),
        max_abs_energy_total_error=_required_finite_float(summary.get("max_abs_energy_total_error"), field="summary.max_abs_energy_total_error"),
        state_at_time_scope=str(state_cost.get("scope")),
        state_at_time_basis=_as_optional_str(state_cost.get("basis")),
        state_at_time_2q=_required_int(state_cost.get("compiled_count_2q"), field="state_at_time_2q"),
        state_at_time_depth=_required_int(state_cost.get("compiled_depth"), field="state_at_time_depth"),
        state_at_time_size=_required_int(state_cost.get("compiled_size"), field="state_at_time_size"),
        full_horizon_scope=str(full_cost.get("scope")),
        full_horizon_basis=_as_optional_str(full_cost.get("basis")),
        full_horizon_2q=_required_int(full_cost.get("compiled_count_2q"), field="full_horizon_2q"),
        full_horizon_depth=_required_int(full_cost.get("compiled_depth"), field="full_horizon_depth"),
        full_horizon_size=_required_int(full_cost.get("compiled_size"), field="full_horizon_size"),
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
        exact_reference_method=_as_optional_str(manifest.get("exact_reference_method")),
        exact_steps_multiplier=manifest.get("exact_steps_multiplier"),
        artifact_run_json=_as_optional_str(artifact_run_json),
        artifact_manifest_json=_as_optional_str(artifact_manifest_json),
        artifact_rows_json=_as_optional_str(artifact_rows_json),
        artifact_summary_json=_as_optional_str(artifact_summary_json),
    )
    return _jsonable(row)


def _build_run_artifact(
    *,
    case: QDriftBenchmarkCase,
    source_payload: Mapping[str, Any],
    context: overlay.RebuiltOverlayContext,
    times: np.ndarray,
    simulation: QDriftSimulationResult,
    cost_rows: Sequence[overlay.CircuitCostRow],
    raw_compile_rows: Mapping[str, Any],
    compile_defaults: Mapping[str, Any],
    output_dir: Path,
    command: str,
) -> dict[str, Any]:
    parameter_manifest = _parameter_manifest(
        case=case,
        source_payload=source_payload,
        context=context,
        times=times,
        compile_defaults=compile_defaults,
        output_dir=output_dir,
    )
    hardware_report_rows = _hardware_report_rows(cost_rows)
    return _jsonable(
        {
            "schema_version": RUN_SCHEMA_VERSION,
            "generated_utc": _now_utc(),
            "case_id": str(case.case_id),
            "method_id": str(simulation.method),
            "method_kind": METHOD_KIND,
            "command": command,
            "source": {
                "controller_json": str(case.controller_json),
                "source_pdf": str(case.source_pdf),
                "run_tag": source_payload.get("run_tag"),
                "artifact_json": _as_optional_str(source_payload.get("artifact_json")),
            },
            "parameter_manifest": parameter_manifest,
            "config": {
                "trotter_steps": int(case.trotter_steps),
                "samples_per_interval": int(case.samples_per_interval),
                "rng_seed": int(case.rng_seed),
                "randomization_family": RANDOMIZATION_FAMILY,
                "compile_backend_name": str(compile_defaults["backend_name"]),
                "compile_seed_transpiler": int(compile_defaults["seed_transpiler"]),
                "compile_optimization_level": int(compile_defaults["optimization_level"]),
                "compile_preferred_fake_backends": tuple(str(x) for x in compile_defaults["preferred_fake_backends"]),
            },
            "contract": {
                "exact_fields_reporting_only": True,
                "controller_decisions_modified": False,
                "controller_paths_called": False,
                "qdrift_sampling_depends_on_exact_fields": False,
                "compile_cost_policy": "seed prep + representative interval-0 additive; full-horizon compiled once; controller reference required",
                "controller_reference_policy": "fail_closed_required_source_compile_reference",
            },
            "trajectory": simulation.trajectory,
            "summary": simulation.summary,
            "qdrift_intervals": simulation.intervals,
            "hardware_cost_rows": _hardware_cost_rows(cost_rows),
            "hardware_report_rows": hardware_report_rows,
            "circuit_costs": [_jsonable(row) for row in cost_rows],
            "raw_compile_rows": raw_compile_rows,
        }
    )


def _run_case(
    case: QDriftBenchmarkCase,
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
    dt = overlay._uniform_dt(times, int(case.trotter_steps))
    context = overlay._rebuild_context(source_payload)
    drive_t0 = float((context.drive_profile or {}).get("t0", 0.0))
    drive_sampling = str((context.drive_profile or {}).get("time_sampling", "midpoint"))
    physical_times = overlay._source_physical_times(source_rows, fallback_drive_t0=float(drive_t0))
    exact_energy = [
        _required_finite_float(row.get("energy_total_exact"), field=f"source_rows[{idx}].energy_total_exact")
        for idx, row in enumerate(source_rows)
    ]
    compile_defaults = _compile_defaults_for_case(case, source_payload=source_payload)
    simulation = _simulate_qdrift(
        psi_initial=context.psi_initial,
        times=times,
        exact_energy_total=exact_energy,
        observation_physical_times=physical_times,
        ordered_labels_exyz=context.ordered_labels_exyz,
        coeff_map_exyz=context.coeff_map_exyz,
        hmat_static=context.hmat,
        drive_coeff_provider_exyz=context.drive_coeff_provider_exyz,
        drive_t0=float(drive_t0),
        drive_time_sampling=str(drive_sampling),
        nq=int(context.nq),
        samples_per_interval=int(case.samples_per_interval),
        rng_seed=int(case.rng_seed),
        method_id=method_id_for_config(
            samples_per_interval=int(case.samples_per_interval),
            rng_seed=int(case.rng_seed),
        ),
    )
    cost_rows, raw_compile_rows = _compile_qdrift_costs(
        case=case,
        context=context,
        source_payload=source_payload,
        simulation=simulation,
        compile_defaults=compile_defaults,
    )
    raw_compile_rows["time_grid"].update({"dt": float(dt), "points": int(times.size)})
    run_artifact = _build_run_artifact(
        case=case,
        source_payload=source_payload,
        context=context,
        times=times,
        simulation=simulation,
        cost_rows=cost_rows,
        raw_compile_rows=raw_compile_rows,
        compile_defaults=compile_defaults,
        output_dir=output_dir,
        command=command,
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
    actual_methods = [
        {
            "case_id": str(record.case.case_id),
            "method_id": method_id_for_config(
                samples_per_interval=int(record.case.samples_per_interval),
                rng_seed=int(record.case.rng_seed),
            ),
            "samples_per_interval": int(record.case.samples_per_interval),
            "rng_seed": int(record.case.rng_seed),
        }
        for record in records
    ]
    method_ids = sorted({str(method["method_id"]) for method in actual_methods})
    sample_values = sorted({int(method["samples_per_interval"]) for method in actual_methods})
    seed_values = sorted({int(method["rng_seed"]) for method in actual_methods})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "benchmark": "hh_qdrift_time_dynamics",
        "method_contract": {
            "method_id": method_ids[0] if len(method_ids) == 1 else "multiple",
            "method_ids": method_ids,
            "default_method_id": METHOD_ID,
            "method_kind": METHOD_KIND,
            "randomization_family": RANDOMIZATION_FAMILY,
            "default_case_id": DEFAULT_CASE_ID,
            "samples_per_interval": sample_values[0] if len(sample_values) == 1 else sample_values,
            "rng_seed": seed_values[0] if len(seed_values) == 1 else seed_values,
            "default_samples_per_interval": DEFAULT_SAMPLES_PER_INTERVAL,
            "default_rng_seed": DEFAULT_RNG_SEED,
            "actual_methods": actual_methods,
            "exact_reference_policy": "reporting_only_after_trajectory_energy",
            "controller_decisions_modified": False,
            "hardware_scope_policy": "fail_closed_required_compile_costs",
            "required_compile_costs": [
                {"method": "controller", "scope": CONTROLLER_SOURCE_SCOPE},
                {"method": "seed", "scope": "seed_prep_only"},
                {"method": "qdrift", "scope": INTERVAL_SCOPE},
                {"method": "qdrift", "scope": FULL_HORIZON_SCOPE},
            ],
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
        "benchmark": "hh_qdrift_time_dynamics",
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
                "randomization_family": row.get("randomization_family"),
                "samples_per_interval": row.get("samples_per_interval"),
                "rng_seed": row.get("rng_seed"),
                "final_abs_energy_total_error": row.get("final_abs_energy_total_error"),
                "mean_abs_energy_total_error": row.get("mean_abs_energy_total_error"),
                "max_abs_energy_total_error": row.get("max_abs_energy_total_error"),
                "state_at_time_2q": row.get("state_at_time_2q"),
                "state_at_time_depth": row.get("state_at_time_depth"),
                "full_horizon_2q": row.get("full_horizon_2q"),
                "full_horizon_depth": row.get("full_horizon_depth"),
                "controller_state_2q": row.get("controller_state_2q"),
                "controller_state_depth": row.get("controller_state_depth"),
            }
            for row in rows
        ],
    }


def run_benchmark(
    *,
    cases: Sequence[QDriftBenchmarkCase],
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
        description="Run the benchmark-local HH L2 t=8 qDRIFT dynamics row."
    )
    parser.add_argument("--case-id", type=str, default=DEFAULT_CASE_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--controller-json", type=Path, default=None)
    parser.add_argument("--source-pdf", type=Path, default=None)
    parser.add_argument("--trotter-steps", type=int, default=None)
    parser.add_argument("--samples-per-interval", type=int, default=None)
    parser.add_argument("--rng-seed", type=int, default=None)
    parser.add_argument("--compile-backend-name", type=str, default=None)
    parser.add_argument("--compile-seed-transpiler", type=int, default=None)
    parser.add_argument("--compile-optimization-level", type=int, default=None)
    parser.add_argument("--compile-preferred-fake-backends", type=str, default=None)
    return parser


def _case_from_args(args: argparse.Namespace) -> QDriftBenchmarkCase:
    case = _case_by_id(str(args.case_id))
    preferred = _parse_string_tuple(args.compile_preferred_fake_backends)
    return replace(
        case,
        controller_json=Path(args.controller_json) if args.controller_json is not None else case.controller_json,
        source_pdf=Path(args.source_pdf) if args.source_pdf is not None else case.source_pdf,
        trotter_steps=int(args.trotter_steps) if args.trotter_steps is not None else case.trotter_steps,
        samples_per_interval=(
            int(args.samples_per_interval)
            if args.samples_per_interval is not None
            else case.samples_per_interval
        ),
        rng_seed=int(args.rng_seed) if args.rng_seed is not None else case.rng_seed,
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
        return " ".join([sys.executable, "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_qdrift_benchmark", *sys.argv[1:]])
    return " ".join(["python", "-m", "pipelines.time_dynamics.legacy.hh_benchmarks.hh_qdrift_benchmark", *map(str, argv)])


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
    print(f"method_id={row.get('method_id')}")
    print(f"randomization_family={row.get('randomization_family')}")
    print(f"samples_per_interval={row.get('samples_per_interval')}")
    print(f"rng_seed={row.get('rng_seed')}")
    print(f"final_abs_energy_total_error={row.get('final_abs_energy_total_error')}")
    print(f"mean_abs_energy_total_error={row.get('mean_abs_energy_total_error')}")
    print(f"max_abs_energy_total_error={row.get('max_abs_energy_total_error')}")
    print(f"state_at_time_2q={row.get('state_at_time_2q')}")
    print(f"state_at_time_depth={row.get('state_at_time_depth')}")
    print(f"full_horizon_2q={row.get('full_horizon_2q')}")
    print(f"full_horizon_depth={row.get('full_horizon_depth')}")
    print(f"controller_state_2q={row.get('controller_state_2q')}")
    print(f"controller_state_depth={row.get('controller_state_depth')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
