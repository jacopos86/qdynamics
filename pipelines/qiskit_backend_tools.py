#!/usr/bin/env python3
"""Shared Qiskit backend resolution and transpile helpers.

This module is backend-target oriented rather than execution/noise oriented.
It provides a small reusable layer for resolving IBM Runtime backend objects,
falling back to installed fake providers when requested, and compiling Qiskit
circuits into backend-native form for structural burden estimation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
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
) -> dict[str, Any]:
    compiled = transpile(
        circuit,
        backend=backend,
        optimization_level=int(optimization_level),
        seed_transpiler=int(seed_transpiler),
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


def compiled_gate_stats(compiled: QuantumCircuit) -> dict[str, Any]:
    op_counts = {str(name): int(count) for name, count in compiled.count_ops().items()}
    compiled_count_2q = 0
    compiled_cx_count = 0
    compiled_ecr_count = 0
    for inst in compiled.data:
        if len(inst.qubits) != 2:
            continue
        compiled_count_2q += 1
        name = str(getattr(inst.operation, "name", "")).lower()
        if name == "cx":
            compiled_cx_count += 1
        elif name == "ecr":
            compiled_ecr_count += 1
    return {
        "compiled_count_2q": int(compiled_count_2q),
        "compiled_cx_count": int(compiled_cx_count),
        "compiled_ecr_count": int(compiled_ecr_count),
        "compiled_op_counts": op_counts,
    }


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
    runtime_lookup_needed = any(not str(name).lower().startswith("fake") for name in requested_unique)
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
