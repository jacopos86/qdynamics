#!/usr/bin/env python3
"""Compute circuit cost metrics for a completed ADAPT-VQE scaffold.

Loads an ADAPT output JSON, reconstructs the ansatz circuit, transpiles to
an IBM fake backend, and reports gate counts, depth, and measurement cost.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from pipelines.hardcoded.adapt_pipeline import (
    _build_hh_full_meta_pool,
    _build_hh_pareto_lean_l3_pool,
    _build_hh_pareto_lean_l2_pool,
    _build_hh_pareto_lean_pool,
)
from pipelines.exact_bench.noise_oracle_runtime import (
    _load_fake_backend,
    compile_circuit_for_local_backend,
    list_local_fake_backend_names,
)
from pipelines.hardcoded.adapt_circuit_execution import (
    append_pauli_rotation_exyz as _append_pauli_rotation_exyz_shared,
    append_reference_state as _append_reference_state_shared,
    build_ansatz_circuit as _build_ansatz_circuit_shared,
)
from pipelines.qiskit_backend_tools import (
    compiled_gate_stats as _compiled_gate_stats_shared,
    rank_compile_rows as _rank_compile_rows_shared,
    safe_circuit_depth as _safe_depth_shared,
)
from pipelines.hardcoded.hh_continuation_generators import rebuild_polynomial_from_serialized_terms
from pipelines.hardcoded.imported_artifact_resolution import (
    ImportedArtifactResolution,
    resolve_imported_artifact_path,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    deserialize_layout,
    expand_legacy_logical_theta,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _load_adapt_result(json_path: Path) -> dict:
    with open(json_path) as f:
        return json.load(f)


def _extract_continuation_block(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    top = payload.get("continuation", None)
    if isinstance(top, Mapping):
        return dict(top)
    adapt_vqe = payload.get("adapt_vqe", None)
    if isinstance(adapt_vqe, Mapping):
        nested = adapt_vqe.get("continuation", None)
        if isinstance(nested, Mapping):
            return dict(nested)
    return {}


def _normalize_adapt_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    if isinstance(payload.get("adapt_vqe", None), Mapping):
        normalized = dict(payload)
        if "continuation" not in normalized:
            continuation = _extract_continuation_block(payload)
            if continuation:
                normalized["continuation"] = dict(continuation)
        return normalized
    adapt_keys = (
        "energy",
        "exact_gs_energy",
        "abs_delta_e",
        "relative_error_abs",
        "operators",
        "optimal_point",
        "logical_optimal_point",
        "logical_num_parameters",
        "parameterization",
        "ansatz_depth",
        "num_parameters",
        "pool_type",
        "continuation_mode",
        "stop_reason",
        "pre_prune_scaffold",
        "prune_summary",
    )
    adapt_vqe = {key: payload[key] for key in adapt_keys if key in payload}
    if not adapt_vqe:
        return dict(payload)
    normalized = dict(payload)
    normalized["adapt_vqe"] = dict(adapt_vqe)
    continuation = _extract_continuation_block(payload)
    if continuation and "continuation" not in normalized:
        normalized["continuation"] = dict(continuation)
    return normalized


def _resolve_total_qubits(settings: Mapping[str, Any], layout: AnsatzParameterLayout) -> int:
    for block in layout.blocks:
        if block.terms:
            return int(block.terms[0].nq)
    from src.quantum.hubbard_latex_python_pairs import boson_qubits_per_site

    L = int(settings.get("L", 3))
    n_ph_max = int(settings.get("n_ph_max", 1))
    boson_encoding = str(settings.get("boson_encoding", "binary"))
    ferm_nq = 2 * L
    boson_nq = int(L * int(boson_qubits_per_site(n_ph_max, boson_encoding)))
    return int(ferm_nq + boson_nq)


def _selected_generator_term_map(payload: Mapping[str, Any]) -> dict[str, AnsatzTerm]:
    out: dict[str, AnsatzTerm] = {}
    continuation = _extract_continuation_block(payload)
    selected_meta = continuation.get("selected_generator_metadata", []) if isinstance(continuation, Mapping) else []
    if not isinstance(selected_meta, Sequence):
        return out
    for raw_meta in selected_meta:
        if not isinstance(raw_meta, Mapping):
            continue
        label = str(raw_meta.get("candidate_label", "")).strip()
        if label == "" or label in out:
            continue
        compile_meta = raw_meta.get("compile_metadata", {})
        serialized_terms = compile_meta.get("serialized_terms_exyz", []) if isinstance(compile_meta, Mapping) else []
        if not isinstance(serialized_terms, Sequence):
            continue
        try:
            poly = rebuild_polynomial_from_serialized_terms(serialized_terms)
        except Exception:
            continue
        out[label] = AnsatzTerm(label=label, polynomial=poly)
    return out


def _resolve_scaffold_ops(payload: Mapping[str, Any], h_poly: Any) -> list[AnsatzTerm]:
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    settings = payload.get("settings", {}) if isinstance(payload, Mapping) else {}
    operator_labels = [str(x) for x in adapt_vqe.get("operators", [])]
    if not operator_labels:
        return []

    L = int(settings.get("L", 3))
    n_ph_max = int(settings.get("n_ph_max", 1))
    boson_encoding = str(settings.get("boson_encoding", "binary"))
    pool_key = str(settings.get("adapt_pool", adapt_vqe.get("pool_type", "pareto_lean")))

    from src.quantum.vqe_latex_python_pairs import half_filled_num_particles

    num_particles = tuple(half_filled_num_particles(L))
    if pool_key == "pareto_lean":
        builder = _build_hh_pareto_lean_pool
    elif pool_key == "pareto_lean_l3":
        builder = _build_hh_pareto_lean_l3_pool
    elif pool_key == "pareto_lean_l2":
        builder = _build_hh_pareto_lean_l2_pool
    else:
        builder = _build_hh_full_meta_pool

    pool, _pool_meta = builder(
        h_poly=h_poly,
        num_sites=L,
        t=float(settings.get("t", 1.0)),
        u=float(settings.get("u", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g_ep=float(settings.get("g_ep", 0.5)),
        dv=float(settings.get("dv", 0.0)),
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        ordering=str(settings.get("ordering", "blocked")),
        boundary=str(settings.get("boundary", "open")),
        paop_r=int(settings.get("paop_r", 1)),
        paop_split_paulis=bool(settings.get("paop_split_paulis", False)),
        paop_prune_eps=float(settings.get("paop_prune_eps", 0.0)),
        paop_normalization=str(settings.get("paop_normalization", "none")),
        num_particles=num_particles,
    )

    pool_by_label = {str(t.label): t for t in pool}
    payload_terms = _selected_generator_term_map(payload)
    scaffold_ops: list[AnsatzTerm] = []
    missing: list[str] = []
    approximated: list[str] = []
    for label in operator_labels:
        if label in pool_by_label:
            scaffold_ops.append(pool_by_label[label])
            continue
        if label in payload_terms:
            scaffold_ops.append(payload_terms[label])
            continue
        parent = label.split("::child_set")[0] if "::child_set" in label else label
        if parent in pool_by_label:
            scaffold_ops.append(AnsatzTerm(label=label, polynomial=pool_by_label[parent].polynomial))
            approximated.append(label)
            continue
        missing.append(label)

    if missing:
        miss_preview = ", ".join(missing[:8])
        raise ValueError(f"Could not reconstruct scaffold operators for labels: {miss_preview}")
    if approximated:
        print("WARNING: using parent-polynomial approximation for unresolved split labels:")
        for label in approximated:
            print(f"  - {label}")
    return scaffold_ops


def _amplitudes_qn_to_q0_to_statevector(
    amps_payload: Mapping[str, Any],
    *,
    nq: int,
) -> np.ndarray:
    state = np.zeros(int(1 << int(nq)), dtype=complex)
    for bitstr_raw, coeff_raw in amps_payload.items():
        bitstr = str(bitstr_raw).strip()
        if len(bitstr) != int(nq):
            raise ValueError(
                f"Amplitude bitstring {bitstr!r} length does not match nq={int(nq)}."
            )
        if not isinstance(coeff_raw, Mapping):
            raise ValueError("Amplitude payload entries must be {re, im} mappings.")
        amp = complex(float(coeff_raw.get("re", 0.0)), float(coeff_raw.get("im", 0.0)))
        state[int(bitstr, 2)] = amp
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError("Amplitude payload defines a zero-norm statevector.")
    return np.asarray(state / norm, dtype=complex).reshape(-1)


def _resolve_ansatz_input_state_from_payload(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray | None, dict[str, Any]]:
    meta: dict[str, Any] = {
        "available": False,
        "reason": "missing_ansatz_input_state_provenance",
        "error": None,
        "source": None,
        "handoff_state_kind": None,
        "nq_total": None,
    }
    if not isinstance(payload, Mapping):
        return None, meta
    state_block = payload.get("ansatz_input_state", {})
    if not isinstance(state_block, Mapping):
        return None, meta
    amps = state_block.get("amplitudes_qn_to_q0", None)
    if not isinstance(amps, Mapping) or not amps:
        meta["reason"] = "invalid_ansatz_input_state_provenance"
        meta["error"] = "ansatz_input_state.amplitudes_qn_to_q0 missing or empty"
        return None, meta
    try:
        nq = int(state_block.get("nq_total", len(next(iter(amps.keys()), ""))))
        if int(nq) <= 0:
            raise ValueError("ansatz_input_state.nq_total must be positive.")
        state = _amplitudes_qn_to_q0_to_statevector(amps, nq=int(nq))
    except Exception as exc:
        meta["reason"] = "invalid_ansatz_input_state_provenance"
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta
    meta.update(
        {
            "available": True,
            "reason": None,
            "source": (None if state_block.get("source", None) is None else str(state_block.get("source"))),
            "handoff_state_kind": (
                None
                if state_block.get("handoff_state_kind", None) is None
                else str(state_block.get("handoff_state_kind"))
            ),
            "nq_total": int(nq),
        }
    )
    return state, meta


def _build_hh_hamiltonian_from_settings(settings: Mapping[str, Any]) -> Any:
    from src.quantum.hubbard_latex_python_pairs import build_hubbard_holstein_hamiltonian

    L = int(settings.get("L", 3))
    n_ph_max = int(settings.get("n_ph_max", 1))
    boson_encoding = str(settings.get("boson_encoding", "binary"))
    return build_hubbard_holstein_hamiltonian(
        dims=L,
        J=float(settings.get("t", 1.0)),
        U=float(settings.get("u", 4.0)),
        omega0=float(settings.get("omega0", 1.0)),
        g=float(settings.get("g_ep", 0.5)),
        n_ph_max=n_ph_max,
        boson_encoding=boson_encoding,
        v_t=None,
        v0=float(settings.get("dv", 0.0)),
        t_eval=None,
        repr_mode="JW",
        indexing=str(settings.get("ordering", "blocked")),
        pbc=(str(settings.get("boundary", "open")) == "periodic"),
        include_zero_point=True,
    )


def reconstruct_imported_adapt_circuit(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    data = _normalize_adapt_payload(payload)
    adapt_vqe = data.get("adapt_vqe", {}) if isinstance(data, Mapping) else {}
    settings = data.get("settings", {}) if isinstance(data, Mapping) else {}
    if not isinstance(adapt_vqe, Mapping):
        raise ValueError("Imported payload is missing adapt_vqe.")
    if not isinstance(settings, Mapping):
        raise ValueError("Imported payload is missing settings.")

    h_poly = _build_hh_hamiltonian_from_settings(settings)
    scaffold_ops: list[AnsatzTerm] = []
    if not isinstance(adapt_vqe.get("parameterization", None), Mapping):
        scaffold_ops = _resolve_scaffold_ops(data, h_poly)
    layout, theta_runtime = _resolve_runtime_layout_and_theta(data, scaffold_ops)
    nq = _resolve_total_qubits(settings, layout)
    ansatz_input_state, ansatz_input_state_meta = _resolve_ansatz_input_state_from_payload(data)
    qc = _build_ansatz_circuit(layout, theta_runtime, int(nq), ref_state=ansatz_input_state)
    return {
        "payload": data,
        "settings": dict(settings),
        "adapt_vqe": dict(adapt_vqe),
        "h_poly": h_poly,
        "layout": layout,
        "theta_runtime": np.asarray(theta_runtime, dtype=float).reshape(-1),
        "num_qubits": int(nq),
        "ansatz_input_state": ansatz_input_state,
        "ansatz_input_state_meta": dict(ansatz_input_state_meta),
        "circuit": qc,
    }


def _resolve_runtime_layout_and_theta(
    payload: Mapping[str, Any],
    scaffold_ops: Sequence[AnsatzTerm],
) -> tuple[AnsatzParameterLayout, np.ndarray]:
    adapt_vqe = payload.get("adapt_vqe", {}) if isinstance(payload, Mapping) else {}
    optimal_point = np.asarray(adapt_vqe.get("optimal_point", []), dtype=float)
    parameterization = adapt_vqe.get("parameterization", None)
    if isinstance(parameterization, Mapping):
        layout = deserialize_layout(parameterization)
        if int(optimal_point.size) != int(layout.runtime_parameter_count):
            raise ValueError(
                f"Runtime theta length {optimal_point.size} does not match serialized layout runtime count {layout.runtime_parameter_count}."
            )
        return layout, optimal_point
    layout = build_parameter_layout(scaffold_ops, ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    theta_runtime = expand_legacy_logical_theta(optimal_point, layout)
    return layout, np.asarray(theta_runtime, dtype=float)


def _append_reference_state(qc: QuantumCircuit, ref_state: np.ndarray | None) -> None:
    _append_reference_state_shared(qc, ref_state)


def _append_pauli_rotation_exyz(qc: QuantumCircuit, *, label_exyz: str, angle: float) -> None:
    _append_pauli_rotation_exyz_shared(qc, label_exyz=label_exyz, angle=angle)


def _build_ansatz_circuit(
    layout: AnsatzParameterLayout,
    theta_runtime: np.ndarray,
    nq: int,
    ref_state: np.ndarray | None = None,
) -> QuantumCircuit:
    return _build_ansatz_circuit_shared(layout, theta_runtime, nq, ref_state=ref_state)


def _hamiltonian_measurement_groups(h_poly, nq: int) -> list[SparsePauliOp]:
    """Group Hamiltonian terms by QWC (qubit-wise commuting) sets."""
    terms = h_poly.return_polynomial()
    pauli_list = []
    for term in terms:
        coeff = complex(term.p_coeff)
        if abs(coeff) <= 1e-12:
            continue
        ps = str(term.pw2strng())
        qiskit_label = ps.upper().replace("E", "I")[::-1]
        if all(c == "I" for c in qiskit_label):
            continue
        pauli_list.append((qiskit_label, coeff))

    if not pauli_list:
        return []

    groups: list[list[tuple[str, complex]]] = []
    for label, coeff in pauli_list:
        placed = False
        for group in groups:
            compatible = True
            for glabel, _ in group:
                for a, b in zip(label, glabel):
                    if a != "I" and b != "I" and a != b:
                        compatible = False
                        break
                if not compatible:
                    break
            if compatible:
                group.append((label, coeff))
                placed = True
                break
        if not placed:
            groups.append([(label, coeff)])

    return [SparsePauliOp([lbl for lbl, _ in g], [c for _, c in g]) for g in groups]


def _estimate_shots_per_group(
    observable: SparsePauliOp,
    target_precision: float = 1.6e-3,
) -> int:
    coeffs = np.asarray(observable.coeffs, dtype=complex)
    variance_bound = float(np.sum(np.abs(coeffs) ** 2))
    shots = int(np.ceil(variance_bound / (target_precision ** 2)))
    return max(shots, 1)


@dataclass(frozen=True)
class CompileScoutConfig:
    source: ImportedArtifactResolution
    requested_backend_name: str | None
    candidate_backends: tuple[str, ...]
    sweep_backends: bool
    seed_transpiler: int
    optimization_level: int
    output_json: Path | None


def _parse_csv(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        items = [chunk.strip() for chunk in raw.split(",")]
    else:
        items = [str(chunk).strip() for chunk in raw]
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item == "":
            continue
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _safe_depth(circuit: QuantumCircuit) -> int:
    return _safe_depth_shared(circuit)


def _compiled_gate_stats(compiled: QuantumCircuit) -> dict[str, Any]:
    return dict(_compiled_gate_stats_shared(compiled))


def _select_best_backend(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    best = _rank_compile_rows_shared(rows)
    return None if best is None else dict(best)


def _resolve_saved_exact_energy(payload: Mapping[str, Any], adapt_vqe: Mapping[str, Any]) -> tuple[float | None, float | None]:
    saved_energy = adapt_vqe.get("energy", None)
    exact_energy = adapt_vqe.get("exact_gs_energy", None)
    if exact_energy is None:
        exact_block = payload.get("exact", {}) if isinstance(payload, Mapping) else {}
        if isinstance(exact_block, Mapping):
            exact_energy = exact_block.get("E_exact_sector", None)
    saved_energy_f = None if saved_energy is None else float(saved_energy)
    exact_energy_f = None if exact_energy is None else float(exact_energy)
    return saved_energy_f, exact_energy_f


def _default_output_json_for_source(source_json: Path) -> Path:
    return source_json.with_name(f"{source_json.stem}_compile_scout.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile an imported HH/ADAPT circuit to local IBM fake backends, or sweep local fake backends "
            "and select the cheapest compiled circuit by 2q count, then depth, then size."
        )
    )
    parser.add_argument(
        "artifact_json",
        nargs="?",
        type=Path,
        help="Imported ADAPT artifact to compile. Defaults to the latest lean pareto_lean_l2 artifact.",
    )
    parser.add_argument(
        "legacy_backend_name",
        nargs="?",
        default=None,
        help="Legacy positional backend name. Prefer --backend-name.",
    )
    parser.add_argument(
        "legacy_opt_level",
        nargs="?",
        type=int,
        default=None,
        help="Legacy positional optimization level. Prefer --optimization-level.",
    )
    parser.add_argument("--artifact-json", dest="artifact_json_flag", type=Path, default=None)
    parser.add_argument(
        "--backend-name",
        type=str,
        default=None,
        help="Local fake backend name to try first. No IBM Runtime lookup is performed.",
    )
    parser.add_argument("--optimization-level", type=int, default=None)
    parser.add_argument("--seed-transpiler", type=int, default=7)
    parser.add_argument(
        "--sweep-backends",
        action="store_true",
        help="Sweep local fake backends and select the cheapest compiled circuit.",
    )
    parser.add_argument(
        "--candidate-backends",
        type=str,
        default=None,
        help="Comma-separated local fake backends to sweep. Defaults to all available fake V2 backends.",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(list(argv) if argv is not None else None)


def resolve_compile_scout_config(args: argparse.Namespace) -> CompileScoutConfig:
    artifact_json = getattr(args, "artifact_json_flag", None)
    if artifact_json is None:
        artifact_json = getattr(args, "artifact_json", None)
    source = resolve_imported_artifact_path(
        requested_json=artifact_json,
        require_default_import_source=True,
    )
    if str(source.mode) != "imported_artifact" or source.resolved_json is None:
        raise FileNotFoundError(
            "No imported ADAPT artifact was supplied and no default lean pareto_lean_l2 artifact was found."
        )

    requested_backend_name = getattr(args, "backend_name", None)
    if requested_backend_name in {None, ""}:
        requested_backend_name = getattr(args, "legacy_backend_name", None)
    if requested_backend_name in {None, "", "none"}:
        requested_backend_name = None

    optimization_level = getattr(args, "optimization_level", None)
    if optimization_level is None:
        optimization_level = getattr(args, "legacy_opt_level", None)
    if optimization_level is None:
        optimization_level = 1

    candidate_backends = _parse_csv(getattr(args, "candidate_backends", None))
    sweep_backends = bool(getattr(args, "sweep_backends", False) or requested_backend_name is None)
    output_json = getattr(args, "output_json", None)
    if output_json is not None:
        output_json = Path(output_json)

    return CompileScoutConfig(
        source=source,
        requested_backend_name=(None if requested_backend_name is None else str(requested_backend_name)),
        candidate_backends=tuple(candidate_backends),
        sweep_backends=bool(sweep_backends),
        seed_transpiler=int(getattr(args, "seed_transpiler", 7)),
        optimization_level=int(optimization_level),
        output_json=output_json,
    )


def _compile_candidate_row(
    *,
    backend_request: str,
    qc: QuantumCircuit,
    seed_transpiler: int,
    optimization_level: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "requested_backend_name": str(backend_request),
        "transpile_backend": None,
        "backend_kind": "local_fake",
        "transpile_status": "backend_unavailable",
        "logical_qubits_required": int(qc.num_qubits),
        "error": None,
    }
    try:
        backend, resolved_name = _load_fake_backend(str(backend_request))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["transpile_backend"] = str(backend_request)
        return row

    row["transpile_backend"] = str(resolved_name)
    try:
        backend_num_qubits = int(getattr(backend, "num_qubits", getattr(backend.configuration(), "num_qubits", 0)))
    except Exception:
        backend_num_qubits = int(getattr(backend, "num_qubits", 0))
    row["backend_num_qubits"] = int(backend_num_qubits)
    if int(backend_num_qubits) < int(qc.num_qubits):
        row["transpile_status"] = "insufficient_qubits"
        row["error"] = (
            f"logical circuit requires {int(qc.num_qubits)} qubits but backend provides {int(backend_num_qubits)}"
        )
        return row

    try:
        compiled_info = compile_circuit_for_local_backend(
            qc,
            backend,
            seed_transpiler=int(seed_transpiler),
            optimization_level=int(optimization_level),
        )
    except Exception as exc:
        row["transpile_status"] = "transpile_failed"
        row["error"] = f"{type(exc).__name__}: {exc}"
        return row

    compiled = compiled_info["compiled"]
    stats = _compiled_gate_stats(compiled)
    row.update(
        {
            "transpile_status": "ok",
            "compiled_size": int(compiled.size()),
            "compiled_depth": _safe_depth(compiled),
            "compiled_num_qubits": int(compiled_info["compiled_num_qubits"]),
            "layout_physical_qubits": [int(x) for x in compiled_info["logical_to_physical"]],
            **stats,
        }
    )
    return row


def run_compile_scout(cfg: CompileScoutConfig) -> dict[str, Any]:
    if cfg.source.resolved_json is None:
        raise FileNotFoundError("Compile scout requires an imported ADAPT artifact JSON.")

    source_json = Path(cfg.source.resolved_json)
    bundle = reconstruct_imported_adapt_circuit(_load_adapt_result(source_json))
    payload = bundle["payload"]
    adapt_vqe = bundle["adapt_vqe"]
    settings = bundle["settings"]
    qc = bundle["circuit"]
    layout = bundle["layout"]
    h_poly = bundle["h_poly"]
    ansatz_input_state_meta = dict(bundle.get("ansatz_input_state_meta", {}))

    measurement_groups = _hamiltonian_measurement_groups(h_poly, int(bundle["num_qubits"]))
    shots_per_group = [_estimate_shots_per_group(group) for group in measurement_groups]
    total_shots = int(sum(shots_per_group))
    saved_energy, exact_energy = _resolve_saved_exact_energy(payload, adapt_vqe)

    warnings: list[str] = []
    if not bool(ansatz_input_state_meta.get("available", False)):
        warnings.append(
            "ansatz_input_state provenance unavailable; compile costs remain useful but ideal-energy comparability to the saved artifact does not."
        )

    sweep_candidates = list(cfg.candidate_backends)
    if not sweep_candidates:
        sweep_candidates = list(list_local_fake_backend_names())

    request_rows: list[dict[str, Any]] = []
    requested_backend_block: dict[str, Any] = {
        "name": cfg.requested_backend_name,
        "supported_locally": None,
        "resolved_name": None,
        "fallback_to_sweep": False,
        "error": None,
    }
    requested_success = False
    if cfg.requested_backend_name is not None:
        requested_row = _compile_candidate_row(
            backend_request=str(cfg.requested_backend_name),
            qc=qc,
            seed_transpiler=int(cfg.seed_transpiler),
            optimization_level=int(cfg.optimization_level),
        )
        request_rows.append(requested_row)
        requested_success = str(requested_row.get("transpile_status", "")) == "ok"
        requested_backend_block.update(
            {
                "supported_locally": bool(requested_success),
                "resolved_name": requested_row.get("transpile_backend"),
                "error": requested_row.get("error"),
            }
        )
        if not requested_success and not bool(cfg.sweep_backends):
            requested_backend_block["fallback_to_sweep"] = True
            warnings.append(
                f"Requested backend {cfg.requested_backend_name!r} is not usable locally; falling back to local fake-backend sweep."
            )

    rows: list[dict[str, Any]] = list(request_rows)
    run_sweep = bool(cfg.sweep_backends or cfg.requested_backend_name is None or not requested_success)
    if run_sweep:
        seen: set[str] = set()
        for row in rows:
            seen.add(str(row.get("requested_backend_name", "")).lower())
            if row.get("transpile_backend") is not None:
                seen.add(str(row.get("transpile_backend", "")).lower())
        for backend_name in sweep_candidates:
            keys = {str(backend_name).lower()}
            if any(key in seen for key in keys):
                continue
            row = _compile_candidate_row(
                backend_request=str(backend_name),
                qc=qc,
                seed_transpiler=int(cfg.seed_transpiler),
                optimization_level=int(cfg.optimization_level),
            )
            rows.append(row)
            seen.add(str(backend_name).lower())
            if row.get("transpile_backend") is not None:
                seen.add(str(row.get("transpile_backend", "")).lower())

    selected_backend = _select_best_backend(rows)
    output_json = cfg.output_json or _default_output_json_for_source(source_json)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    logical_circuit = {
        "num_qubits": int(bundle["num_qubits"]),
        "operator_count": int(len(adapt_vqe.get("operators", []))),
        "logical_parameter_count": int(layout.logical_parameter_count),
        "runtime_parameter_count": int(layout.runtime_parameter_count),
        "abstract_size": int(qc.size()),
        "abstract_depth": _safe_depth(qc),
        "saved_artifact_energy": saved_energy,
        "exact_energy": exact_energy,
        "reference_state_embedded": bool(ansatz_input_state_meta.get("available", False)),
        "ansatz_input_state_source": ansatz_input_state_meta.get("source"),
        "ansatz_input_state_kind": ansatz_input_state_meta.get("handoff_state_kind"),
    }
    measurement_cost = {
        "hamiltonian_terms": int(len(h_poly.return_polynomial())),
        "qwc_groups": int(len(measurement_groups)),
        "shots_per_group_mha": [int(x) for x in shots_per_group],
        "total_shots_mha": int(total_shots),
    }
    summary = {
        "successful_candidates": int(sum(1 for row in rows if str(row.get("transpile_status", "")) == "ok")),
        "candidate_count": int(len(rows)),
        "selected_backend": (None if selected_backend is None else selected_backend.get("transpile_backend")),
        "requested_backend_supported_locally": requested_backend_block.get("supported_locally"),
        "reference_state_embedded": bool(ansatz_input_state_meta.get("available", False)),
        "warnings": list(warnings),
    }
    scout_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "adapt_circuit_compile_scout",
        "workflow_contract": {
            "scope": "local_fake_backend_transpile_only",
            "requested_backend_policy": "local_fake_only_then_sweep",
            "selection_metric": [
                "compiled_count_2q",
                "compiled_depth",
                "compiled_size",
                "transpile_backend",
            ],
            "runtime_lookup": False,
        },
        "config": _jsonable(asdict(cfg)),
        "import_source": _jsonable(asdict(cfg.source)),
        "logical_circuit": logical_circuit,
        "measurement_cost": measurement_cost,
        "requested_backend": requested_backend_block,
        "rows": rows,
        "selected_backend": selected_backend,
        "artifacts": {"output_json": str(output_json)},
        "summary": summary,
        "success": bool(selected_backend is not None),
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(_jsonable(scout_payload), handle, indent=2, sort_keys=True)

    return scout_payload


def _print_human_summary(payload: Mapping[str, Any]) -> None:
    logical_circuit = payload.get("logical_circuit", {}) if isinstance(payload, Mapping) else {}
    measurement_cost = payload.get("measurement_cost", {}) if isinstance(payload, Mapping) else {}
    summary = payload.get("summary", {}) if isinstance(payload, Mapping) else {}
    selected = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
    requested = payload.get("requested_backend", {}) if isinstance(payload, Mapping) else {}

    print(f"{'=' * 60}")
    print("Imported ADAPT Compile Scout")
    print(f"{'=' * 60}")
    print(f"Source: {payload.get('import_source', {}).get('resolved_json')}")
    print(f"Logical qubits: {logical_circuit.get('num_qubits')}")
    print(
        f"Parameters: logical={logical_circuit.get('logical_parameter_count')} runtime={logical_circuit.get('runtime_parameter_count')}"
    )
    print(
        f"Abstract circuit: size={logical_circuit.get('abstract_size')} depth={logical_circuit.get('abstract_depth')} operators={logical_circuit.get('operator_count')}"
    )
    print(
        f"Measurement cost: QWC groups={measurement_cost.get('qwc_groups')} total shots @ mHa={measurement_cost.get('total_shots_mha')}"
    )
    if requested.get("name") is not None:
        print(
            f"Requested backend: {requested.get('name')} (supported_locally={requested.get('supported_locally')}, resolved_name={requested.get('resolved_name')})"
        )
    if summary.get("warnings"):
        print("Warnings:")
        for warning in summary.get("warnings", []):
            print(f"  - {warning}")
    if isinstance(selected, Mapping) and selected:
        print("Selected backend:")
        print(
            f"  {selected.get('transpile_backend')}: 2q={selected.get('compiled_count_2q')} depth={selected.get('compiled_depth')} size={selected.get('compiled_size')}"
        )
    else:
        print("Selected backend: none")


def main(argv: Sequence[str] | None = None) -> None:
    cfg = resolve_compile_scout_config(parse_args(argv))
    payload = run_compile_scout(cfg)
    _print_human_summary(payload)
    print(f"compile_scout_json={payload['artifacts']['output_json']}")
    selected_backend = payload.get("selected_backend", {}) if isinstance(payload, Mapping) else {}
    if isinstance(selected_backend, Mapping) and selected_backend.get("transpile_backend") is not None:
        print(f"selected_backend={selected_backend['transpile_backend']}")
    requested_backend = payload.get("requested_backend", {}) if isinstance(payload, Mapping) else {}
    if isinstance(requested_backend, Mapping) and requested_backend.get("name") is not None:
        print(
            "requested_backend_supported_locally="
            f"{bool(requested_backend.get('supported_locally', False))}"
        )


if __name__ == "__main__":
    main()
