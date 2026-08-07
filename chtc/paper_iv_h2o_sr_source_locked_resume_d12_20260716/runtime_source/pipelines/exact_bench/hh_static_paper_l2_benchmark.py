#!/usr/bin/env python3
"""Paper-facing L=2 HH static benchmark table builder.

This module is deliberately benchmark-local.  It does not change the canonical
ADAPT/VQE workflows; it consumes rows emitted by ``hh_static_ground_state_benchmark``
and enriches them with paper-role labels plus reproducible static circuit-cost
columns when the row exposes enough ansatz structure to reconstruct a benchmark
state-preparation circuit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.hh_static_ground_state_benchmark import (
    HHBenchmarkAlgorithmSpec,
    HHBenchmarkCase,
    _reference_state_vector,
    _resolve_compiled_operator_terms,
    canonical_hh_benchmark_cases,
    default_hh_benchmark_algorithms,
    run_hh_static_ground_state_benchmark,
)
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context

SCHEMA_VERSION = "hh_static_paper_l2_benchmark_v1"
DEFAULT_INPUT_ROWS = Path("artifacts/agent_runs/20260428_static_l2_audit_v2/hh_static_benchmark_rows.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/agent_runs/20260429_static_paper_l2_v1")
DEFAULT_BACKEND_NAME = "FakeMarrakesh"
DEFAULT_SEED_TRANSPILER = 7
DEFAULT_OPTIMIZATION_LEVEL = 2
DEFAULT_L2_CASE_IDS = ("hh_L2_strong_canonical", "hh_L2_weak_diagnostic")


@dataclass(frozen=True)
class StaticCompileConfig:
    backend_name: str = DEFAULT_BACKEND_NAME
    seed_transpiler: int = DEFAULT_SEED_TRANSPILER
    optimization_level: int = DEFAULT_OPTIMIZATION_LEVEL
    preferred_fake_backends: tuple[str, ...] = (DEFAULT_BACKEND_NAME,)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        try:
            return _json_ready(value.tolist())
        except Exception:
            pass
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _case_lookup() -> dict[str, HHBenchmarkCase]:
    return {case.case_id: case for case in canonical_hh_benchmark_cases()}


def _algorithm_lookup() -> dict[str, HHBenchmarkAlgorithmSpec]:
    return {algorithm.algorithm_id: algorithm for algorithm in default_hh_benchmark_algorithms()}


def _case_from_row(row: Mapping[str, Any]) -> HHBenchmarkCase:
    known = _case_lookup()
    case_id = str(row.get("case_id", ""))
    if case_id in known:
        return known[case_id]
    return HHBenchmarkCase(
        case_id=case_id,
        num_sites=int(row.get("L", row.get("num_sites", 0))),
        t=float(row.get("t", 1.0)),
        u=float(row.get("u", 0.0)),
        dv=float(row.get("dv", 0.0)),
        omega0=float(row.get("omega0", 1.0)),
        g_ep=float(row.get("g_ep", 0.0)),
        n_ph_max=int(row.get("n_ph_max", 1)),
        boson_encoding=str(row.get("boson_encoding", "binary")),
        ordering=str(row.get("ordering", "blocked")),
        boundary=str(row.get("boundary", "open")),
        include_zero_point=bool(row.get("include_zero_point", True)),
    )


def _algorithm_from_row(row: Mapping[str, Any]) -> HHBenchmarkAlgorithmSpec | None:
    return _algorithm_lookup().get(str(row.get("method_id", "")))


def _row_method_id(row: Mapping[str, Any]) -> str:
    return str(row.get("method_id", row.get("algorithm_id", "")))


def _load_artifact_payload(row: Mapping[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    raw_path = row.get("artifact_json", row.get("artifact_path", ""))
    if not raw_path:
        return {}
    path = Path(str(raw_path))
    if not path.is_absolute() and base_dir is not None:
        candidate = base_dir / path
        if candidate.exists():
            path = candidate
    if not path.exists():
        return {"artifact_load_error": f"missing artifact_json: {raw_path}"}
    try:
        payload = _read_json(path)
    except Exception as exc:  # pragma: no cover - defensive for corrupted artifacts.
        return {"artifact_load_error": f"{type(exc).__name__}: {exc}"}
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"artifact_load_error": f"artifact payload was {type(payload).__name__}, expected object"}


def classify_static_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Classify a normalized static benchmark row for paper-facing tables."""

    method_id = _row_method_id(row)
    method_kind = str(row.get("method_kind", ""))
    quality_status = str(row.get("quality_status", ""))
    delta_e = _finite_float_or_none(row.get("delta_E_abs", row.get("abs_delta_e")))
    flags = [str(flag) for flag in row.get("benchmark_audit_flags", []) or []]

    if quality_status == "ok_paper_candidate" and method_id.startswith("hh_adapt_"):
        role = "candidate_ours"
        include = True
    elif quality_status == "ok_paper_candidate":
        role = "fair_competitor"
        include = True
    elif quality_status == "ok_optimizer_suspect":
        role = "diagnostic_optimizer_suspect"
        include = False
    elif quality_status == "ok_large_error":
        role = "diagnostic_large_error"
        include = False
    elif quality_status == "ok_reference_not_improved":
        role = "excluded_reference_not_improved"
        include = False
    else:
        role = "excluded_incomplete"
        include = False

    if method_kind in {"compiled_operator_qsci", "compiled_operator_sqd", "qsci", "sqd"}:
        qpu_compatibility = "qpu_compatible_sampling_subspace_diagnostic"
    elif method_id == "hh_hea_qiskit_vqe":
        qpu_compatibility = "qpu_compatible_qiskit_validation_path"
    elif method_kind in {"conventional_vqe", "compiled_operator_vqe", "compiled_operator_avqite", "adapt_vqe"}:
        qpu_compatibility = "qpu_compatible_state_preparation_benchmark"
    else:
        qpu_compatibility = "unknown"

    if "optimizer_suspect" in flags and include:
        include = False
        role = "diagnostic_optimizer_suspect"

    if delta_e is not None and delta_e > 0.1 and include:
        include = False
        role = "diagnostic_large_error"

    return {
        "paper_role": role,
        "paper_include": bool(include),
        "paper_reason": _paper_reason(role=role, quality_status=quality_status, delta_e=delta_e),
        "qpu_compatibility": qpu_compatibility,
    }


def _paper_reason(*, role: str, quality_status: str, delta_e: float | None) -> str:
    delta_text = "unknown |ΔE|" if delta_e is None else f"|ΔE|={delta_e:.6g}"
    if role == "candidate_ours":
        return f"current HH ADAPT-family candidate; {delta_text}; quality_status={quality_status}"
    if role == "fair_competitor":
        return f"fixed/adaptive competitor row with paper-candidate audit status; {delta_text}"
    if role == "diagnostic_optimizer_suspect":
        return f"kept as diagnostic only because optimizer convergence is suspect; {delta_text}"
    if role == "diagnostic_large_error":
        return f"kept as diagnostic only because error is large; {delta_text}"
    if role == "excluded_reference_not_improved":
        return f"excluded from paper Pareto claims because it does not improve the reference state; {delta_text}"
    return f"excluded from paper Pareto claims; {delta_text}; quality_status={quality_status}"


def _num_qubits_from_reference(ref_state: Any) -> int:
    arr = np.asarray(ref_state, dtype=complex).reshape(-1)
    if arr.size <= 0 or int(arr.size) & (int(arr.size) - 1):
        raise ValueError("reference state length must be a positive power of two")
    return int(arr.size.bit_length() - 1)


def _selected_labels(payload: Mapping[str, Any]) -> set[str]:
    labels = payload.get("selected_operator_labels", None)
    if not labels:
        return set()
    return {str(label) for label in labels}


def _repeat_terms(terms: Sequence[Any], reps: int) -> list[Any]:
    out: list[Any] = []
    for _ in range(max(1, int(reps))):
        out.extend(list(terms))
    return out


def _resolved_conventional_reps(row: Mapping[str, Any], payload: Mapping[str, Any], algorithm: HHBenchmarkAlgorithmSpec | None) -> int:
    for source in (payload, row):
        reps = _int_or_none(source.get("vqe_reps_used", source.get("vqe_reps")))
        if reps is not None and reps > 0:
            return reps
    if algorithm is not None and algorithm.vqe_reps is not None:
        return int(algorithm.vqe_reps)
    return 2


def _build_adapt_parameterization_circuit(
    *,
    payload: Mapping[str, Any],
    ref_state: Any,
) -> tuple[Any, str, str, int | None]:
    from qiskit import QuantumCircuit

    from pipelines.hardcoded.adapt_circuit_execution import append_pauli_rotation_exyz, append_reference_state

    parameterization = payload.get("parameterization", {})
    if not isinstance(parameterization, Mapping):
        raise ValueError("ADAPT artifact has no parameterization object")
    blocks = parameterization.get("blocks", [])
    if not isinstance(blocks, Sequence):
        raise ValueError("ADAPT parameterization.blocks is not a sequence")

    nq = None
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        for term in block.get("runtime_terms_exyz", []) or []:
            if isinstance(term, Mapping) and term.get("nq") is not None:
                nq = int(term["nq"])
                break
        if nq is not None:
            break
    if nq is None:
        nq = _num_qubits_from_reference(ref_state)

    qc = QuantumCircuit(int(nq))
    append_reference_state(qc, ref_state)
    runtime_term_count = 0
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        for term in block.get("runtime_terms_exyz", []) or []:
            if not isinstance(term, Mapping):
                continue
            label = str(term.get("pauli_exyz", "")).strip().lower()
            if not label:
                continue
            coeff_re = float(term.get("coeff_re", 0.0))
            # A nonzero structural angle is enough for transpiled circuit burden;
            # numeric theta is intentionally not used because most ADAPT artifacts
            # do not persist optimized runtime theta.
            append_pauli_rotation_exyz(qc, label_exyz=label, angle=2.0 * coeff_re)
            runtime_term_count += 1
    return (
        qc,
        "static_selected_adapt_skeleton_with_reference",
        "ADAPT optimized theta is not persisted; compile cost is for selected ansatz skeleton with nonzero structural angles.",
        runtime_term_count,
    )


def _build_conventional_circuit(
    *,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec | None,
    ref_state: Any,
) -> tuple[Any, str, str, int | None]:
    method_id = _row_method_id(row)
    ansatz_kind = str(payload.get("ansatz_kind", row.get("ansatz_name", row.get("ansatz_kind", "")))).strip()
    if method_id == "hh_hea_qiskit_vqe" or ansatz_kind in {"qiskit_hea", "hh_hea_qiskit"}:
        return _build_qiskit_hea_circuit(row=row, payload=payload, ref_state=ref_state, algorithm=algorithm)

    from pipelines.exact_bench.hh_conventional_vqe import _build_hh_conventional_ansatz
    from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit

    normalized_kind = "layerwise" if "layer" in ansatz_kind else "termwise"
    reps = _resolved_conventional_reps(row, payload, algorithm)
    ansatz = _build_hh_conventional_ansatz(
        ansatz_kind=normalized_kind,  # type: ignore[arg-type]
        num_sites=int(case.num_sites),
        t=float(case.t),
        u=float(case.u),
        omega0=float(case.omega0),
        g_ep=float(case.g_ep),
        n_ph_max=int(case.n_ph_max),
        boson_encoding=str(case.boson_encoding),
        boundary=str(case.boundary),
        ordering=str(case.ordering),
        reps=int(reps),
        include_zero_point=bool(case.include_zero_point),
    )
    terms = _repeat_terms(getattr(ansatz, "base_terms"), reps)
    _layout, qc = build_structural_ansatz_circuit(
        terms,
        nq=_num_qubits_from_reference(ref_state),
        ref_state=ref_state,
        structure_theta_value=1.0,
    )
    return (
        qc,
        f"static_{normalized_kind}_hva_skeleton_with_reference",
        "Native HVA compile cost reconstructed from the benchmark ansatz base terms and repetition count.",
        len(terms),
    )


def _build_qiskit_hea_circuit(
    *,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    ref_state: Any,
    algorithm: HHBenchmarkAlgorithmSpec | None,
) -> tuple[Any, str, str, int | None]:
    from qiskit import QuantumCircuit

    from pipelines.exact_bench.hh_conventional_vqe import _build_qiskit_hea_ansatz
    from pipelines.hardcoded.adapt_circuit_execution import append_reference_state

    nq = _num_qubits_from_reference(ref_state)
    reps = _resolved_conventional_reps(row, payload, algorithm)
    adapter = _build_qiskit_hea_ansatz(num_qubits=int(nq), reps=int(reps))
    base_circuit = getattr(adapter, "_circuit")
    parameters = tuple(getattr(adapter, "_parameters"))
    raw_theta = payload.get("theta", None)
    if isinstance(raw_theta, Sequence) and not isinstance(raw_theta, (str, bytes)) and len(raw_theta) == len(parameters):
        assignments = {param: float(raw_theta[idx]) for idx, param in enumerate(parameters)}
        hea_circuit = base_circuit.assign_parameters(assignments, inplace=False)
        note = "Qiskit HEA compile cost uses the persisted optimized theta."
    else:
        assignments = {param: 1.0 for param in parameters}
        hea_circuit = base_circuit.assign_parameters(assignments, inplace=False)
        note = "Qiskit HEA compile cost uses nonzero structural angles because persisted theta is unavailable or mismatched."
    qc = QuantumCircuit(int(nq))
    append_reference_state(qc, ref_state)
    qc.compose(hea_circuit, inplace=True)
    return qc, "static_qiskit_hea_with_reference", note, int(len(parameters))


def _build_compiled_operator_circuit(
    *,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec,
    resolved_problem: Any,
    ref_state: Any,
) -> tuple[Any, str, str, int | None]:
    from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit

    terms = list(_resolve_compiled_operator_terms(case=case, algorithm=algorithm, resolved_problem=resolved_problem))
    selected = _selected_labels(payload)
    if selected:
        filtered = [term for term in terms if str(getattr(term, "label", "")) in selected]
        if filtered:
            terms = filtered
    method_kind = str(row.get("method_kind", ""))
    if method_kind in {"compiled_operator_qsci", "compiled_operator_sqd", "qsci", "sqd"}:
        scope = "static_subspace_probe_generator_skeleton_with_reference"
        note = "QSCI/SQD cost is a probe-generator circuit burden, not a single variational final-state circuit."
    else:
        scope = "static_compiled_operator_skeleton_with_reference"
        note = "Compiled-operator benchmark cost reconstructed from selected operator source terms with nonzero structural angles."
    _layout, qc = build_structural_ansatz_circuit(
        terms,
        nq=_num_qubits_from_reference(ref_state),
        ref_state=ref_state,
        structure_theta_value=1.0,
    )
    return qc, scope, note, len(terms)


def build_static_circuit_for_row(
    *,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec | None,
    resolved_problem: Any,
) -> tuple[Any, str, str, int | None]:
    ref_state = _reference_state_vector(resolved_problem)
    if ref_state is None:
        raise ValueError("resolved problem lacks reference_state.build_state()")

    method_kind = str(row.get("method_kind", ""))
    method_id = _row_method_id(row)
    if method_kind == "adapt_vqe" or method_id.startswith("hh_adapt_"):
        return _build_adapt_parameterization_circuit(payload=payload, ref_state=ref_state)
    if (
        method_kind.startswith("compiled_operator")
        or method_kind in {"avqite", "qsci", "sqd"}
        or (algorithm is not None and str(getattr(algorithm, "operator_source", "")).strip())
    ):
        if algorithm is None:
            raise ValueError(f"No benchmark algorithm spec available for {method_id!r}")
        return _build_compiled_operator_circuit(
            row=row,
            payload=payload,
            case=case,
            algorithm=algorithm,
            resolved_problem=resolved_problem,
            ref_state=ref_state,
        )
    if method_kind == "conventional_vqe":
        return _build_conventional_circuit(
            row=row,
            payload=payload,
            case=case,
            algorithm=algorithm,
            ref_state=ref_state,
        )
    raise ValueError(f"Unsupported method_kind for static compile reconstruction: {method_kind!r}")


def _compile_cost_payload(
    *,
    method_id: str,
    circuit: Any,
    scope: str,
    config: StaticCompileConfig,
) -> dict[str, Any]:
    from dataclasses import asdict as dataclass_asdict

    from pipelines.time_dynamics.legacy.analysis.hh_realtime_suzuki_overlay import _compile_one_circuit_cost

    cost, audit_rows = _compile_one_circuit_cost(
        method=str(method_id),
        order=None,
        scope=str(scope),
        trotter_steps=None,
        includes_seed_prep=True,
        circuit=circuit,
        backend_name=str(config.backend_name),
        preferred_fake_backends=tuple(config.preferred_fake_backends),
        seed_transpiler=int(config.seed_transpiler),
        optimization_level=int(config.optimization_level),
        export_circuit_dir=None,
        export_stem=None,
    )
    payload = dataclass_asdict(cost)
    payload["compile_audit_rows"] = audit_rows
    return _json_ready(payload)


def _empty_compile_fields(status: str, *, error: str | None = None, note: str | None = None) -> dict[str, Any]:
    return {
        "static_compile_status": status,
        "static_compile_scope": None,
        "static_compile_note": note,
        "static_compile_error": error,
        "static_compile_backend": None,
        "static_compile_seed_transpiler": None,
        "static_compile_optimization_level": None,
        "static_abstract_size": None,
        "static_abstract_depth": None,
        "static_compiled_2q": None,
        "static_compiled_depth": None,
        "static_compiled_size": None,
        "static_compiled_num_qubits": None,
        "static_compiled_op_counts": {},
        "static_logical_to_physical": [],
        "static_compiled_operator_count": None,
    }


def enrich_row_with_static_cost(
    *,
    row: Mapping[str, Any],
    payload: Mapping[str, Any],
    case: HHBenchmarkCase,
    algorithm: HHBenchmarkAlgorithmSpec | None,
    resolved_problem: Any,
    config: StaticCompileConfig,
    compile_enabled: bool,
) -> dict[str, Any]:
    if not compile_enabled:
        return _empty_compile_fields("skipped_by_policy")
    if payload.get("artifact_load_error"):
        return _empty_compile_fields("artifact_unavailable", error=str(payload.get("artifact_load_error")))
    try:
        circuit, scope, note, operator_count = build_static_circuit_for_row(
            row=row,
            payload=payload,
            case=case,
            algorithm=algorithm,
            resolved_problem=resolved_problem,
        )
        cost = _compile_cost_payload(method_id=_row_method_id(row), circuit=circuit, scope=scope, config=config)
        status = str(cost.get("transpile_status", "unknown"))
        return {
            "static_compile_status": status,
            "static_compile_scope": scope,
            "static_compile_note": note,
            "static_compile_error": cost.get("error"),
            "static_compile_backend": cost.get("backend_name"),
            "static_compile_seed_transpiler": cost.get("seed_transpiler"),
            "static_compile_optimization_level": cost.get("optimization_level"),
            "static_abstract_size": cost.get("abstract_size"),
            "static_abstract_depth": cost.get("abstract_depth"),
            "static_compiled_2q": cost.get("compiled_count_2q"),
            "static_compiled_depth": cost.get("compiled_depth"),
            "static_compiled_size": cost.get("compiled_size"),
            "static_compiled_num_qubits": cost.get("compiled_num_qubits"),
            "static_compiled_op_counts": cost.get("compiled_op_counts", {}),
            "static_logical_to_physical": cost.get("logical_to_physical", []),
            "static_compiled_operator_count": operator_count,
            "static_compile_audit_rows": cost.get("compile_audit_rows", []),
        }
    except Exception as exc:
        return _empty_compile_fields("error", error=f"{type(exc).__name__}: {exc}")


def _compile_policy_enabled(policy: str, classified_row: Mapping[str, Any]) -> bool:
    if policy == "none":
        return False
    if policy == "all":
        return True
    if policy == "paper":
        return bool(classified_row.get("paper_include")) or str(classified_row.get("paper_role", "")).startswith("diagnostic_")
    raise ValueError(f"unknown compile policy {policy!r}")


def _resolve_rows(input_rows: Path | None, output_dir: Path) -> tuple[list[dict[str, Any]], Path, str]:
    if input_rows is not None:
        rows_path = Path(input_rows)
        rows = _read_json(rows_path)
        if not isinstance(rows, list):
            raise ValueError(f"input rows must be a JSON list: {rows_path}")
        return [dict(row) for row in rows], rows_path, "input_rows"

    raw_dir = output_dir / "raw_static_l2"
    result = run_hh_static_ground_state_benchmark(
        output_dir=raw_dir,
        case_ids=DEFAULT_L2_CASE_IDS,
    )
    rows_path_raw = result.get("rows_path", result.get("rows_json"))
    if rows_path_raw is None:
        raise KeyError("static benchmark result did not expose rows_path or rows_json")
    rows_path = Path(rows_path_raw)
    rows = _read_json(rows_path)
    return [dict(row) for row in rows], rows_path, "regenerated_static_l2"


def _l2_rows(rows: Iterable[Mapping[str, Any]], case_ids: Sequence[str]) -> list[dict[str, Any]]:
    allowed = {str(case_id) for case_id in case_ids}
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("case_id", "")) in allowed:
            out.append(dict(row))
    return out


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    role_counts = Counter(str(row.get("paper_role", "")) for row in rows)
    quality_counts = Counter(str(row.get("quality_status", "")) for row in rows)
    compile_counts = Counter(str(row.get("static_compile_status", "")) for row in rows)
    include_rows = [row for row in rows if bool(row.get("paper_include"))]
    best_by_case: dict[str, dict[str, Any]] = {}
    pareto_by_case: dict[str, list[dict[str, Any]]] = {}
    for case_id in sorted({str(row.get("case_id", "")) for row in rows}):
        case_rows = [row for row in rows if str(row.get("case_id", "")) == case_id]
        candidates = [row for row in case_rows if _finite_float_or_none(row.get("delta_E_abs")) is not None]
        if candidates:
            best = min(candidates, key=lambda item: float(item.get("delta_E_abs")))
            best_by_case[case_id] = _compact_summary_row(best)
        pareto_by_case[case_id] = [_compact_summary_row(row) for row in _pareto_front(case_rows)]
    return {
        "schema": SCHEMA_VERSION,
        "row_count": len(rows),
        "paper_include_count": len(include_rows),
        "role_counts": dict(sorted(role_counts.items())),
        "quality_status_counts": dict(sorted(quality_counts.items())),
        "static_compile_status_counts": dict(sorted(compile_counts.items())),
        "best_by_case_delta_e": best_by_case,
        "static_pareto_by_case": pareto_by_case,
    }


def _compact_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "method_id": row.get("method_id"),
        "paper_role": row.get("paper_role"),
        "paper_include": row.get("paper_include"),
        "quality_status": row.get("quality_status"),
        "delta_E_abs": row.get("delta_E_abs"),
        "energy": row.get("energy"),
        "static_compiled_2q": row.get("static_compiled_2q"),
        "static_compiled_depth": row.get("static_compiled_depth"),
        "static_compile_scope": row.get("static_compile_scope"),
    }


def _pareto_front(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    eligible = []
    for row in rows:
        delta = _finite_float_or_none(row.get("delta_E_abs"))
        cost = _int_or_none(row.get("static_compiled_2q"))
        if delta is None or cost is None:
            continue
        eligible.append((delta, cost, row))
    front = []
    for delta, cost, row in eligible:
        dominated = False
        for other_delta, other_cost, other_row in eligible:
            if other_row is row:
                continue
            if other_delta <= delta and other_cost <= cost and (other_delta < delta or other_cost < cost):
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda item: (float(item.get("delta_E_abs")), int(item.get("static_compiled_2q"))))


def build_paper_l2_benchmark(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    input_rows: str | Path | None = DEFAULT_INPUT_ROWS,
    case_ids: Sequence[str] = DEFAULT_L2_CASE_IDS,
    compile_policy: str = "all",
    compile_config: StaticCompileConfig | None = None,
) -> dict[str, Any]:
    output_path = Path(output_dir)
    config = compile_config or StaticCompileConfig()
    rows, source_rows_path, row_source = _resolve_rows(Path(input_rows) if input_rows is not None else None, output_path)
    selected_rows = _l2_rows(rows, case_ids)
    if not selected_rows:
        raise ValueError(f"No rows matched requested case_ids={tuple(case_ids)!r}")

    manifest = {
        "schema": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "output_dir": str(output_path),
        "row_source": row_source,
        "source_rows_path": str(source_rows_path),
        "case_ids": list(case_ids),
        "compile_policy": str(compile_policy),
        "compile_config": asdict(config),
        "contract": {
            "scope": "HH static/no-drive L=2 paper-facing benchmark table",
            "main_workflow_impact": "read-only wrapper over benchmark artifacts; no production route defaults are changed",
            "qiskit_boundary": "Qiskit is used only for benchmark/reference circuit compilation, not production/core VQE paths",
            "phase3_canonical_status": "not frozen; current ADAPT rows are candidates/diagnostics, not a promoted final Phase 3 claim",
        },
    }
    _write_json(output_path / "paper_l2_manifest.json", manifest)

    resolved_by_case: dict[str, Any] = {}
    enriched: list[dict[str, Any]] = []
    base_dir = Path.cwd()
    for raw_row in selected_rows:
        row = dict(raw_row)
        classification = classify_static_row(row)
        row.update(classification)
        case = _case_from_row(row)
        algorithm = _algorithm_from_row(row)
        case_id = str(case.case_id)
        if case_id not in resolved_by_case:
            resolved_by_case[case_id] = resolve_problem_context(case.to_problem_request())
        payload = _load_artifact_payload(row, base_dir=base_dir)
        compile_enabled = _compile_policy_enabled(str(compile_policy), row)
        row.update(
            enrich_row_with_static_cost(
                row=row,
                payload=payload,
                case=case,
                algorithm=algorithm,
                resolved_problem=resolved_by_case[case_id],
                config=config,
                compile_enabled=compile_enabled,
            )
        )
        enriched.append(row)
        _write_json(output_path / "paper_l2_rows.json", enriched)

    summary = _summary(enriched)
    _write_json(output_path / "paper_l2_summary.json", summary)
    return {
        "manifest_path": str(output_path / "paper_l2_manifest.json"),
        "rows_path": str(output_path / "paper_l2_rows.json"),
        "summary_path": str(output_path / "paper_l2_summary.json"),
        "summary": summary,
    }


def _parse_case_ids(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(raw).split(",") if part.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-rows", type=Path, default=DEFAULT_INPUT_ROWS, help="Existing hh_static_benchmark_rows.json to enrich. Use --regenerate-static to ignore this.")
    parser.add_argument("--regenerate-static", action="store_true", help="Regenerate L=2 static rows into output_dir/raw_static_l2 before enrichment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-ids", default=",".join(DEFAULT_L2_CASE_IDS), help="Comma-separated case_ids to include.")
    parser.add_argument("--compile-policy", choices=("all", "paper", "none"), default="all")
    parser.add_argument("--backend-name", default=DEFAULT_BACKEND_NAME)
    parser.add_argument("--seed-transpiler", type=int, default=DEFAULT_SEED_TRANSPILER)
    parser.add_argument("--optimization-level", type=int, default=DEFAULT_OPTIMIZATION_LEVEL)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_rows = None if bool(args.regenerate_static) else Path(args.input_rows)
    result = build_paper_l2_benchmark(
        output_dir=Path(args.output_dir),
        input_rows=input_rows,
        case_ids=_parse_case_ids(args.case_ids),
        compile_policy=str(args.compile_policy),
        compile_config=StaticCompileConfig(
            backend_name=str(args.backend_name),
            seed_transpiler=int(args.seed_transpiler),
            optimization_level=int(args.optimization_level),
            preferred_fake_backends=(str(args.backend_name),),
        ),
    )
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
