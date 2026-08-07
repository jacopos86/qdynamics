#!/usr/bin/env python3
"""Shared Qiskit backend resolution and transpile helpers.

This module is backend-target oriented rather than execution/noise oriented.
It provides a small reusable layer for resolving IBM Runtime backend objects,
falling back to installed fake providers when requested, and compiling Qiskit
circuits into backend-native form for structural burden estimation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from qiskit import QuantumCircuit, transpile


@dataclass(frozen=True)
class ResolvedBackendTarget:
    requested_name: str
    resolved_name: str
    resolution_kind: str
    using_fake_backend: bool
    backend_obj: Any
    target_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BackendResolutionAuditRow:
    requested_name: str
    resolved_name: str | None
    success: bool
    resolution_kind: str
    using_fake_backend: bool
    runtime_lookup_attempted: bool
    runtime_error: str | None = None
    fake_exact_attempted: str | None = None
    fallback_reason: str | None = None
    error: str | None = None
    target_snapshot: dict[str, Any] = field(default_factory=dict)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _safe_stem(raw: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(raw).strip())
    cleaned = cleaned.strip("._")
    return cleaned or "compiled_circuit"


def _canonicalize_token(name: str | None) -> str:
    raw = "" if name is None else str(name).strip()
    if raw == "":
        return ""
    return raw


def _family_token(name: str | None) -> str:
    raw = _canonicalize_token(name)
    if raw == "":
        return ""
    lowered = raw.lower()
    if lowered.startswith("fake"):
        raw = raw[4:]
    lowered = raw.lower()
    if lowered.startswith("ibm_"):
        raw = raw[4:]
    raw = re.sub(r"v\d+$", "", raw, flags=re.IGNORECASE)
    parts = [part for part in re.split(r"[^0-9A-Za-z]+", raw) if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _candidate_fake_names(name: str | None) -> tuple[str, ...]:
    raw = _canonicalize_token(name)
    if raw == "":
        return ()
    if raw.lower().startswith("fake"):
        fam = _family_token(raw)
    else:
        fam = _family_token(raw)
    if fam == "":
        return ()
    return (f"Fake{fam}", f"Fake{fam}V2", f"Fake{fam}V3")


def _candidate_runtime_names(name: str | None) -> tuple[str, ...]:
    raw = _canonicalize_token(name)
    if raw == "":
        return ()
    candidates: list[str] = []
    seen: set[str] = set()
    for cand in (
        raw,
        raw.lower(),
        (raw if raw.lower().startswith("ibm_") else f"ibm_{_family_token(raw).lower()}"),
    ):
        token = str(cand).strip()
        if token == "":
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        candidates.append(token)
    return tuple(candidates)


def list_local_fake_backend_names() -> tuple[str, ...]:
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception:
        return ()
    names: list[str] = []
    for name in dir(fake_provider):
        if not str(name).startswith("Fake"):
            continue
        obj = getattr(fake_provider, name, None)
        if not callable(obj):
            continue
        names.append(str(name))
    return tuple(sorted(set(names)))


def load_local_fake_backend(name: str) -> tuple[Any, str]:
    try:
        from qiskit_ibm_runtime import fake_provider
    except Exception as exc:
        raise RuntimeError("qiskit_ibm_runtime.fake_provider is unavailable.") from exc
    for class_name in _candidate_fake_names(name):
        backend_cls = getattr(fake_provider, class_name, None)
        if backend_cls is not None:
            return backend_cls(), class_name
    raise ValueError(f"Unknown local fake backend {str(name).strip()!r}.")


class _StaticCouplingMap:
    def __init__(self, edges: Sequence[Sequence[int]]) -> None:
        self._edges = tuple((int(u), int(v)) for u, v in edges)

    def get_edges(self) -> list[tuple[int, int]]:
        return list(self._edges)


class StaticFakeGraphBackend:
    """Minimal fake-backend graph object for analytic graph-span scoring only."""

    def __init__(
        self,
        *,
        name: str,
        num_qubits: int,
        coupling_edges: Sequence[Sequence[int]],
        backend_version: str | None = None,
        operation_names: Sequence[str] = (),
    ) -> None:
        self.name = str(name)
        self.num_qubits = int(num_qubits)
        self.backend_version = None if backend_version is None else str(backend_version)
        self.operation_names = tuple(str(op) for op in operation_names)
        self.coupling_map = _StaticCouplingMap(coupling_edges)


def _qiskit_ibm_runtime_file(relative_path: str) -> Path:
    try:
        dist = importlib_metadata.distribution("qiskit-ibm-runtime")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError("qiskit-ibm-runtime is not installed.") from exc
    return Path(dist.locate_file(relative_path))


def load_static_fake_graph_backend(name: str) -> tuple[StaticFakeGraphBackend, str, dict[str, Any]]:
    """Load a lightweight fake-backend coupling graph without importing Runtime.

    This is intended for analytic graph estimators that need the historical
    backend coupling map but do not need a Qiskit BackendV2 transpilation target.
    """
    family = _family_token(name)
    if family != "Marrakesh":
        raise ValueError(f"No static fake graph backend is registered for {str(name).strip()!r}.")
    conf_path = _qiskit_ibm_runtime_file(
        "qiskit_ibm_runtime/fake_provider/backends/marrakesh/conf_marrakesh.json"
    )
    with conf_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    coupling_edges = payload.get("coupling_map")
    if not isinstance(coupling_edges, list) or not coupling_edges:
        for gate in payload.get("gates", []):
            if str(gate.get("name", "")).lower() == "cz":
                coupling_edges = gate.get("coupling_map")
                break
    if not isinstance(coupling_edges, list) or not coupling_edges:
        raise ValueError("static FakeMarrakesh config missing coupling_map.")
    num_qubits = int(payload.get("n_qubits", payload.get("num_qubits", 0)) or 0)
    if num_qubits <= 0:
        raise ValueError("static FakeMarrakesh config missing n_qubits.")
    backend = StaticFakeGraphBackend(
        name="FakeMarrakesh",
        num_qubits=int(num_qubits),
        coupling_edges=coupling_edges,
        backend_version=(None if payload.get("backend_version") is None else str(payload.get("backend_version"))),
        operation_names=payload.get("basis_gates", ()),
    )
    metadata = {
        "schema": "static_fake_graph_backend_v1",
        "source": "qiskit_ibm_runtime.fake_provider.backends.marrakesh.conf_marrakesh_json",
        "source_path": str(conf_path),
        "backend_name_in_config": str(payload.get("backend_name", "")),
        "resolved_name": "FakeMarrakesh",
        "num_qubits": int(num_qubits),
        "directed_coupling_edge_count": int(len(coupling_edges)),
    }
    return backend, "FakeMarrakesh", _jsonable(metadata)


def _logical_to_physical_qubits(compiled: QuantumCircuit, logical_qubits: int) -> tuple[int, ...]:
    layout = getattr(compiled, "layout", None)
    if layout is None or not hasattr(layout, "final_index_layout"):
        return tuple(range(int(logical_qubits)))
    try:
        mapped = list(layout.final_index_layout())
    except Exception:
        mapped = []
    if len(mapped) < int(logical_qubits):
        return tuple(range(int(logical_qubits)))
    return tuple(int(mapped[idx]) for idx in range(int(logical_qubits)))


def compile_circuit_for_backend(
    circuit: QuantumCircuit,
    backend: Any,
    *,
    seed_transpiler: int,
    optimization_level: int = 1,
    initial_layout: Sequence[int] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if initial_layout is not None:
        kwargs["initial_layout"] = [int(q) for q in initial_layout]
    compiled = transpile(
        circuit,
        backend=backend,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed_transpiler),
        **kwargs,
    )
    logical_to_physical = _logical_to_physical_qubits(compiled, int(circuit.num_qubits))
    return {
        "compiled": compiled,
        "logical_to_physical": logical_to_physical,
        "compiled_num_qubits": int(compiled.num_qubits),
    }


def safe_circuit_depth(circuit: QuantumCircuit) -> int:
    depth = circuit.depth()
    return 0 if depth is None else int(depth)


def safe_two_qubit_depth(circuit: QuantumCircuit) -> int:
    try:
        depth = circuit.depth(filter_function=lambda inst: len(getattr(inst, "qubits", ())) == 2)
    except TypeError:
        depth = circuit.depth(filter_function=lambda inst: len(inst[1]) == 2)
    return 0 if depth is None else int(depth)


def compiled_gate_stats(compiled: QuantumCircuit) -> dict[str, Any]:
    op_counts = {str(name): int(count) for name, count in compiled.count_ops().items()}
    compiled_count_1q = 0
    compiled_count_2q = 0
    compiled_cx_count = 0
    compiled_ecr_count = 0
    excluded_gate_count_noops = {"barrier", "delay", "id", "measure", "reset"}
    for inst in compiled.data:
        name = str(getattr(inst.operation, "name", "")).lower()
        if name in excluded_gate_count_noops:
            continue
        if len(inst.qubits) == 1:
            compiled_count_1q += 1
        if len(inst.qubits) != 2:
            continue
        compiled_count_2q += 1
        if name == "cx":
            compiled_cx_count += 1
        elif name == "ecr":
            compiled_ecr_count += 1
    return {
        "compiled_count_1q": int(compiled_count_1q),
        "compiled_count_1q_semantics": "post_transpile_one_qubit_quantum_ops_excluding_barrier_delay_id_measure_reset",
        "compiled_count_2q": int(compiled_count_2q),
        "compiled_depth_2q": int(safe_two_qubit_depth(compiled)),
        "compiled_cx_count": int(compiled_cx_count),
        "compiled_ecr_count": int(compiled_ecr_count),
        "compiled_op_counts": op_counts,
    }


def _instruction_payload(circuit: QuantumCircuit, index: int, inst: Any) -> dict[str, Any]:
    operation = inst.operation
    qubits = [int(circuit.find_bit(bit).index) for bit in inst.qubits]
    clbits = [int(circuit.find_bit(bit).index) for bit in inst.clbits]
    return {
        "index": int(index),
        "operation": str(getattr(operation, "name", "")),
        "label": None if getattr(operation, "label", None) is None else str(operation.label),
        "num_qubits": int(getattr(operation, "num_qubits", len(qubits))),
        "num_clbits": int(getattr(operation, "num_clbits", len(clbits))),
        "qubits": qubits,
        "clbits": clbits,
        "params": _jsonable(list(getattr(operation, "params", []))),
    }


def _write_ops_jsonl(circuit: QuantumCircuit, path: Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for idx, inst in enumerate(circuit.data):
            handle.write(json.dumps(_instruction_payload(circuit, idx, inst), sort_keys=True))
            handle.write("\n")
            count += 1
    return int(count)


def _write_preview_text(
    circuit: QuantumCircuit,
    path: Path,
    *,
    draw_max_ops: int,
    preview_edge_ops: int,
) -> dict[str, Any]:
    size = int(circuit.size())
    preview_is_complete = size <= int(draw_max_ops)
    with Path(path).open("w", encoding="utf-8") as handle:
        if preview_is_complete:
            handle.write("# Complete Qiskit text drawing\n")
            handle.write(f"# size={size} depth={safe_circuit_depth(circuit)} qubits={circuit.num_qubits}\n\n")
            handle.write(str(circuit.draw(output="text")))
            handle.write("\n")
        else:
            edge = max(1, int(preview_edge_ops))
            handle.write("# Preview only: full circuit is in QPY and ops JSONL artifacts\n")
            handle.write(f"# size={size} depth={safe_circuit_depth(circuit)} qubits={circuit.num_qubits}\n")
            handle.write(f"# showing first {edge} and last {edge} instructions\n\n")
            payloads = [_instruction_payload(circuit, idx, inst) for idx, inst in enumerate(circuit.data)]
            for row in payloads[:edge]:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
            omitted = max(0, int(size) - 2 * edge)
            handle.write(f"... omitted {omitted} instructions ...\n")
            for row in payloads[-edge:]:
                handle.write(json.dumps(row, sort_keys=True))
                handle.write("\n")
    return {
        "preview_is_complete": bool(preview_is_complete),
        "preview_kind": "qiskit_text_drawer" if preview_is_complete else "ops_head_tail",
        "draw_max_ops": int(draw_max_ops),
        "preview_edge_ops": int(preview_edge_ops),
    }


def export_compiled_circuit_artifacts(
    compiled: QuantumCircuit,
    *,
    output_dir: str | Path,
    stem: str,
    metadata: Mapping[str, Any] | None = None,
    draw_max_ops: int = 400,
    preview_edge_ops: int = 80,
) -> dict[str, Any]:
    """Write a compiled Qiskit circuit as QPY, full ops JSONL, and readable preview."""
    from qiskit import qpy

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_stem(stem)
    qpy_path = out_dir / f"{safe}.qpy"
    ops_path = out_dir / f"{safe}.ops.jsonl"
    preview_path = out_dir / f"{safe}.preview.txt"
    with qpy_path.open("wb") as handle:
        qpy.dump([compiled], handle)
    ops_rows = _write_ops_jsonl(compiled, ops_path)
    preview = _write_preview_text(
        compiled,
        preview_path,
        draw_max_ops=int(draw_max_ops),
        preview_edge_ops=int(preview_edge_ops),
    )
    stats = compiled_gate_stats(compiled)
    payload = {
        "stem": safe,
        "qpy_path": str(qpy_path),
        "ops_jsonl_path": str(ops_path),
        "preview_text_path": str(preview_path),
        "ops_jsonl_rows": int(ops_rows),
        "compiled_size": int(compiled.size()),
        "compiled_depth": int(safe_circuit_depth(compiled)),
        "compiled_num_qubits": int(compiled.num_qubits),
        **stats,
        **preview,
    }
    if metadata is not None:
        payload["metadata"] = _jsonable(dict(metadata))
    return _jsonable(payload)


def rank_compile_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    status_key: str = "transpile_status",
    field_order: Sequence[str] = ("compiled_count_2q", "compiled_depth", "compiled_size", "transpile_backend"),
) -> dict[str, Any] | None:
    successful = [dict(row) for row in rows if str(row.get(status_key, "")) == "ok"]
    if not successful:
        return None

    def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        out: list[Any] = []
        for field in field_order:
            value = row.get(field, None)
            if isinstance(value, (int, float)):
                out.append(float(value))
            else:
                out.append(str(value))
        return tuple(out)

    return min(successful, key=_row_key)


def undirected_coupling_edges(backend: Any) -> tuple[tuple[int, int], ...]:
    """Return symmetrized, deduplicated coupling-map edges for a backend."""
    coupling_map = getattr(backend, "coupling_map", None)
    if coupling_map is None or not hasattr(coupling_map, "get_edges"):
        raise ValueError("backend_missing_coupling_map_get_edges")
    edges: set[tuple[int, int]] = set()
    for raw_u, raw_v in coupling_map.get_edges():
        u = int(raw_u)
        v = int(raw_v)
        if u == v:
            continue
        edges.add((min(u, v), max(u, v)))
    return tuple(sorted(edges))


def backend_coupling_graph_snapshot(backend: Any) -> dict[str, Any]:
    """Return a deterministic symmetrized coupling graph snapshot."""
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        try:
            backend_name = str(name_attr())
        except Exception:
            backend_name = None
    else:
        backend_name = None if name_attr is None else str(name_attr)
    edges = undirected_coupling_edges(backend)
    return _jsonable(
        {
            "backend_name": backend_name,
            "num_qubits": int(getattr(backend, "num_qubits", 0) or 0),
            "coupling_edge_count": int(len(edges)),
            "coupling_edges": [list(edge) for edge in edges],
            "graph_directed_source": "backend.coupling_map.get_edges",
            "graph_symmetrized": True,
        }
    )


def snapshot_backend_target(backend: Any) -> dict[str, Any]:
    name_attr = getattr(backend, "name", None)
    if callable(name_attr):
        try:
            backend_name = str(name_attr())
        except Exception:
            backend_name = None
    else:
        backend_name = None if name_attr is None else str(name_attr)
    backend_version = None
    try:
        backend_version = getattr(backend, "backend_version", None)
    except Exception:
        backend_version = None
    target = getattr(backend, "target", None)
    coupling_map = getattr(backend, "coupling_map", None)
    edge_count = None
    if coupling_map is not None:
        try:
            edge_count = int(len(coupling_map.get_edges()))
        except Exception:
            edge_count = None
    operation_names = []
    try:
        operation_names = [str(x) for x in list(getattr(backend, "operation_names", []))]
    except Exception:
        operation_names = []
    dt_val = None
    try:
        dt_raw = getattr(backend, "dt", None)
        dt_val = None if dt_raw is None else float(dt_raw)
    except Exception:
        dt_val = None
    instruction_durations_present = False
    try:
        instruction_durations_present = getattr(backend, "instruction_durations", None) is not None
    except Exception:
        instruction_durations_present = False
    snapshot = {
        "backend_name": backend_name,
        "backend_version": (None if backend_version is None else str(backend_version)),
        "num_qubits": int(getattr(backend, "num_qubits", 0) or 0),
        "operation_names": list(operation_names),
        "coupling_edge_count": edge_count,
        "dt": dt_val,
        "instruction_durations_present": bool(instruction_durations_present),
        "target_present": bool(target is not None),
    }
    return _jsonable(snapshot)


def resolve_backend_targets(
    *,
    requested_names: Sequence[str],
    preferred_fake_backends: Sequence[str] = ("FakeNighthawk", "FakeFez", "FakeMarrakesh"),
    allow_preferred_fallback: bool = True,
    fallback_mode: str = "single",
    allow_runtime_lookup: bool = True,
) -> tuple[tuple[ResolvedBackendTarget, ...], list[dict[str, Any]]]:
    requested_unique: list[str] = []
    seen_requested: set[str] = set()
    for name in requested_names:
        token = _canonicalize_token(name)
        if token == "":
            continue
        key = token.lower()
        if key in seen_requested:
            continue
        seen_requested.add(key)
        requested_unique.append(token)

    available_fake = set(list_local_fake_backend_names())
    runtime_service: Any | None = None
    runtime_error: str | None = None
    non_fake_requested = any(not str(name).lower().startswith("fake") for name in requested_unique)
    runtime_lookup_needed = bool(allow_runtime_lookup) and bool(non_fake_requested)
    if runtime_lookup_needed:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService

            runtime_service = QiskitRuntimeService()
        except Exception as exc:
            runtime_error = f"{type(exc).__name__}: {exc}"

    targets: list[ResolvedBackendTarget] = []
    audit: list[BackendResolutionAuditRow] = []
    seen_resolved: set[str] = set()

    def _append_target(*, requested: str, resolved: str, resolution_kind: str, using_fake: bool, backend_obj: Any, fallback_reason: str | None = None, fake_exact_attempted: str | None = None, local_runtime_error: str | None = None, runtime_attempted: bool = False) -> None:
        resolved_key = str(resolved).lower()
        snapshot = snapshot_backend_target(backend_obj)
        if resolved_key not in seen_resolved:
            seen_resolved.add(resolved_key)
            targets.append(
                ResolvedBackendTarget(
                    requested_name=str(requested),
                    resolved_name=str(resolved),
                    resolution_kind=str(resolution_kind),
                    using_fake_backend=bool(using_fake),
                    backend_obj=backend_obj,
                    target_snapshot=dict(snapshot),
                )
            )
        audit.append(
            BackendResolutionAuditRow(
                requested_name=str(requested),
                resolved_name=str(resolved),
                success=True,
                resolution_kind=str(resolution_kind),
                using_fake_backend=bool(using_fake),
                runtime_lookup_attempted=bool(runtime_attempted),
                runtime_error=(None if local_runtime_error is None else str(local_runtime_error)),
                fake_exact_attempted=(None if fake_exact_attempted is None else str(fake_exact_attempted)),
                fallback_reason=(None if fallback_reason is None else str(fallback_reason)),
                error=None,
                target_snapshot=dict(snapshot),
            )
        )

    for requested in requested_unique:
        requested_lower = str(requested).lower()
        attempted_fake_name: str | None = None
        local_runtime_error: str | None = None
        runtime_attempted = False
        if requested_lower.startswith("fake"):
            for cand in _candidate_fake_names(requested):
                attempted_fake_name = cand
                try:
                    backend_obj, resolved_name = load_local_fake_backend(cand)
                except Exception as exc:
                    if str(cand) in available_fake:
                        audit.append(
                            BackendResolutionAuditRow(
                                requested_name=str(requested),
                                resolved_name=None,
                                success=False,
                                resolution_kind="unavailable",
                                using_fake_backend=True,
                                runtime_lookup_attempted=False,
                                fake_exact_attempted=str(cand),
                                fallback_reason=None,
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        break
                    continue
                _append_target(
                    requested=str(requested),
                    resolved=str(resolved_name),
                    resolution_kind="fake_exact",
                    using_fake=True,
                    backend_obj=backend_obj,
                    fake_exact_attempted=str(cand),
                    runtime_attempted=False,
                )
                break
            else:
                audit.append(
                    BackendResolutionAuditRow(
                        requested_name=str(requested),
                        resolved_name=None,
                        success=False,
                        resolution_kind="unavailable",
                        using_fake_backend=True,
                        runtime_lookup_attempted=False,
                        fake_exact_attempted=attempted_fake_name,
                        fallback_reason=None,
                        error=f"No installed local fake backend matched {requested!r}.",
                    )
                )
            continue

        if runtime_service is not None:
            runtime_attempted = True
            for runtime_name in _candidate_runtime_names(requested):
                try:
                    backend_obj = runtime_service.backend(str(runtime_name))
                    _append_target(
                        requested=str(requested),
                        resolved=str(runtime_name),
                        resolution_kind="runtime",
                        using_fake=False,
                        backend_obj=backend_obj,
                        runtime_attempted=True,
                    )
                    local_runtime_error = None
                    break
                except Exception as exc:
                    local_runtime_error = f"{type(exc).__name__}: {exc}"
            if local_runtime_error is None:
                continue
        elif runtime_lookup_needed:
            runtime_attempted = True
            local_runtime_error = runtime_error
        elif not bool(allow_runtime_lookup) and not requested_lower.startswith("fake"):
            runtime_attempted = False
            local_runtime_error = "runtime_lookup_disabled"

        resolved_exact = False
        for cand in _candidate_fake_names(requested):
            attempted_fake_name = cand
            try:
                backend_obj, resolved_name = load_local_fake_backend(cand)
            except Exception as exc:
                if str(cand) in available_fake:
                    local_runtime_error = local_runtime_error or runtime_error
                    audit.append(
                        BackendResolutionAuditRow(
                            requested_name=str(requested),
                            resolved_name=None,
                            success=False,
                            resolution_kind="unavailable",
                            using_fake_backend=True,
                            runtime_lookup_attempted=bool(runtime_attempted),
                            runtime_error=(None if local_runtime_error is None else str(local_runtime_error)),
                            fake_exact_attempted=str(cand),
                            fallback_reason=None,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    resolved_exact = True
                    break
                continue
            _append_target(
                requested=str(requested),
                resolved=str(resolved_name),
                resolution_kind="fake_exact",
                using_fake=True,
                backend_obj=backend_obj,
                fake_exact_attempted=str(cand),
                local_runtime_error=local_runtime_error,
                runtime_attempted=bool(runtime_attempted),
            )
            resolved_exact = True
            break
        if resolved_exact:
            continue

        audit.append(
            BackendResolutionAuditRow(
                requested_name=str(requested),
                resolved_name=None,
                success=False,
                resolution_kind="unavailable",
                using_fake_backend=False,
                runtime_lookup_attempted=bool(runtime_attempted),
                runtime_error=(None if local_runtime_error is None else str(local_runtime_error)),
                fake_exact_attempted=attempted_fake_name,
                fallback_reason=None,
                error=f"Unable to resolve backend target {requested!r}.",
            )
            )

    if not targets and allow_preferred_fallback:
        preferred_available: list[tuple[str, Any, str]] = []
        for preferred in preferred_fake_backends:
            try:
                backend_obj, resolved_name = load_local_fake_backend(str(preferred))
            except Exception:
                continue
            preferred_available.append((str(preferred), backend_obj, str(resolved_name)))
        if fallback_mode == "single" and preferred_available:
            pick, backend_obj, resolved_name = preferred_available[0]
            _append_target(
                requested=(requested_unique[0] if requested_unique else pick),
                resolved=str(resolved_name),
                resolution_kind="fake_preferred_fallback",
                using_fake=True,
                backend_obj=backend_obj,
                fallback_reason="runtime_or_exact_backend_unavailable",
                local_runtime_error=runtime_error,
                runtime_attempted=bool(runtime_lookup_needed),
            )
        elif fallback_mode == "shortlist" and preferred_available:
            for pick, backend_obj, resolved_name in preferred_available:
                _append_target(
                    requested=str(pick),
                    resolved=str(resolved_name),
                    resolution_kind="fake_preferred_fallback",
                    using_fake=True,
                    backend_obj=backend_obj,
                    fallback_reason="runtime_or_exact_backend_unavailable_for_requested_shortlist",
                    local_runtime_error=runtime_error,
                    runtime_attempted=bool(runtime_lookup_needed),
                )

    return tuple(targets), [asdict(row) if isinstance(row, BackendResolutionAuditRow) else row for row in audit]
