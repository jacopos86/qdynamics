#!/usr/bin/env python3
"""Opt-in local fake-backend compile audit for HH Math17A realtime outputs.

The audit is intentionally compile-only and local-fake-only.  It never performs
IBM Runtime service lookup; requested non-Fake backend names are resolved only
through installed local fake-provider classes or the configured fake fallback
shortlist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


_DEFAULT_PREFERRED_FAKES = ("FakeMarrakesh", "FakeNighthawk", "FakeFez")


@dataclass(frozen=True)
class RealtimeCompileAuditConfig:
    mode: str = "off"
    backend_name: str = "FakeMarrakesh"
    seed_transpiler: int = 7
    optimization_level: int = 2
    preferred_fake_backends: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_PREFERRED_FAKES)
    export_circuit_dir: Path | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _parse_string_tuple(raw: str | Sequence[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    chunks = raw.split(",") if isinstance(raw, str) else list(raw)
    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        token = str(chunk).strip()
        if token == "":
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(token)
    return tuple(out)


def normalize_compile_audit_mode(raw: str | None) -> str:
    text = "off" if raw is None else str(raw).strip().lower()
    aliases = {
        "none": "off",
        "disabled": "off",
        "final": "final_scaffold",
        "final-scaffold": "final_scaffold",
    }
    normalized = aliases.get(text, text)
    if normalized not in {"off", "final_scaffold"}:
        raise ValueError(f"Unsupported realtime compile audit mode {raw!r}.")
    return str(normalized)


def accepted_prune_event_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return accepted prune rows from a controller trajectory or ledger.

    A paper-facing prune-cost audit must be keyed to an actual accepted removal,
    not to a proposed prune.  The stable cross-version signal is a negative
    runtime-parameter delta; action_kind is retained as an additional guard.
    """
    out: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            continue
        raw_delta = row.get("runtime_parameter_count_delta", None)
        if raw_delta is None:
            before = row.get("runtime_parameter_count_before", None)
            after = row.get("runtime_parameter_count_after", None)
            try:
                raw_delta = int(after) - int(before)
            except Exception:
                raw_delta = None
        try:
            delta = int(raw_delta)
        except Exception:
            continue
        if delta >= 0:
            continue
        action_kind = str(row.get("action_kind", ""))
        if action_kind and action_kind != "prune_coordinate":
            continue
        record = dict(row)
        record["row_index"] = int(row_index)
        record["runtime_parameter_count_delta"] = int(delta)
        out.append(record)
    return out


def build_compile_audit_config_from_args(args: Any) -> RealtimeCompileAuditConfig:
    preferred = _parse_string_tuple(getattr(args, "compile_audit_preferred_fake_backends", None))
    if not preferred:
        preferred = _DEFAULT_PREFERRED_FAKES
    backend_name = getattr(args, "compile_audit_backend_name", None)
    if backend_name in {None, ""}:
        backend_name = "FakeMarrakesh"
    return RealtimeCompileAuditConfig(
        mode=normalize_compile_audit_mode(getattr(args, "compile_audit_mode", "off")),
        backend_name=str(backend_name),
        seed_transpiler=int(getattr(args, "compile_audit_seed_transpiler", 7)),
        optimization_level=int(getattr(args, "compile_audit_optimization_level", 2)),
        preferred_fake_backends=tuple(str(x) for x in preferred),
        export_circuit_dir=(
            None
            if getattr(args, "compile_audit_export_circuit_dir", None) in {None, ""}
            else Path(str(getattr(args, "compile_audit_export_circuit_dir")))
        ),
    )


def _infer_num_qubits(*, controller: Any) -> int:
    raw_num_qubits = getattr(controller, "_num_qubits", None)
    if raw_num_qubits is not None:
        return int(raw_num_qubits)
    ref_state = getattr(getattr(controller, "replay_context", None), "psi_ref", None)
    if ref_state is not None:
        size = int(np.asarray(ref_state, dtype=complex).reshape(-1).size)
        if size > 0:
            return int(round(np.log2(size)))
    raise ValueError("Cannot infer controller qubit count for compile audit.")


def _percent_reduction(before: Any, after: Any) -> float | None:
    try:
        b = float(before)
        a = float(after)
    except Exception:
        return None
    if not np.isfinite(b) or not np.isfinite(a) or b == 0.0:
        return None
    return float(100.0 * (b - a) / b)


def _compile_request_payload(config: RealtimeCompileAuditConfig) -> dict[str, Any]:
    return {
        "mode": str(config.mode),
        "backend_name": str(config.backend_name),
        "requested_backend_name": str(config.backend_name),
        "seed_transpiler": int(config.seed_transpiler),
        "transpile_seed": int(config.seed_transpiler),
        "optimization_level": int(config.optimization_level),
        "transpile_optimization_level": int(config.optimization_level),
        "preferred_fake_backends": [str(x) for x in config.preferred_fake_backends],
        "export_circuit_dir": None if config.export_circuit_dir is None else str(config.export_circuit_dir),
        "local_fake_only": True,
        "runtime_lookup_enabled": False,
        "allow_runtime_lookup": False,
    }


def _scaffold_labels_from_snapshot(snapshot: Mapping[str, Any]) -> list[str]:
    labels = snapshot.get("labels", [])
    if isinstance(labels, Sequence) and not isinstance(labels, (str, bytes)):
        return [str(x) for x in labels]
    return []


def _compile_scaffold_snapshot(
    *,
    snapshot: Mapping[str, Any],
    replay_context: Any,
    num_qubits: int,
    config: RealtimeCompileAuditConfig,
    scope: str,
    export_stem: str | None = None,
) -> dict[str, Any]:
    """Compile one in-memory controller scaffold snapshot to local fake targets."""
    # Lazy imports keep default realtime execution free of Qiskit compile overhead.
    from pipelines.hardcoded.adapt_circuit_execution import build_ansatz_circuit
    from pipelines.qiskit_backend_tools import (
        compile_circuit_for_backend,
        compiled_gate_stats,
        export_compiled_circuit_artifacts,
        rank_compile_rows,
        resolve_backend_targets,
        safe_circuit_depth,
    )

    layout = snapshot["layout"]
    theta_runtime = np.asarray(snapshot["theta_runtime"], dtype=float).reshape(-1)
    ref_state = np.asarray(getattr(replay_context, "psi_ref"), dtype=complex).reshape(-1)
    qc = build_ansatz_circuit(
        layout,
        theta_runtime,
        int(num_qubits),
        ref_state=ref_state,
    )
    labels = _scaffold_labels_from_snapshot(snapshot)
    logical = {
        "num_qubits": int(num_qubits),
        "logical_block_count": int(getattr(layout, "logical_parameter_count")),
        "runtime_parameter_count": int(getattr(layout, "runtime_parameter_count")),
        "abstract_size": int(qc.size()),
        "abstract_depth": int(safe_circuit_depth(qc)),
        "labels": labels,
    }

    targets, resolution_audit = resolve_backend_targets(
        requested_names=(str(config.backend_name),),
        preferred_fake_backends=tuple(str(x) for x in config.preferred_fake_backends),
        allow_preferred_fallback=True,
        fallback_mode="single",
        allow_runtime_lookup=False,
    )

    rows: list[dict[str, Any]] = []
    compiled_by_backend: dict[str, Any] = {}
    for target in targets:
        row: dict[str, Any] = {
            "scope": str(scope),
            "requested_backend": str(target.requested_name),
            "requested_backend_name": str(target.requested_name),
            "transpile_backend": str(target.resolved_name),
            "backend_name": str(target.resolved_name),
            "resolution_kind": str(target.resolution_kind),
            "using_fake_backend": bool(target.using_fake_backend),
            "target_snapshot": dict(getattr(target, "target_snapshot", {}) or {}),
            "seed_transpiler": int(config.seed_transpiler),
            "transpile_seed": int(config.seed_transpiler),
            "optimization_level": int(config.optimization_level),
            "transpile_optimization_level": int(config.optimization_level),
            "runtime_lookup_enabled": False,
            "transpile_status": "not_run",
            "error": None,
            "abstract_size": int(qc.size()),
            "abstract_depth": int(safe_circuit_depth(qc)),
        }
        try:
            compiled_info = compile_circuit_for_backend(
                qc,
                target.backend_obj,
                seed_transpiler=int(config.seed_transpiler),
                optimization_level=int(config.optimization_level),
            )
            compiled = compiled_info["compiled"]
            compiled_by_backend[str(target.resolved_name)] = compiled
            row.update(
                {
                    "transpile_status": "ok",
                    "compiled_depth": int(safe_circuit_depth(compiled)),
                    "compiled_size": int(compiled.size()),
                    "compiled_num_qubits": int(compiled_info.get("compiled_num_qubits", compiled.num_qubits)),
                    "logical_to_physical": [int(x) for x in compiled_info.get("logical_to_physical", ())],
                }
            )
            row.update(dict(compiled_gate_stats(compiled)))
        except Exception as exc:
            row.update(
                {
                    "transpile_status": "error",
                    "compiled_depth": None,
                    "compiled_size": None,
                    "compiled_num_qubits": None,
                    "logical_to_physical": [],
                    "compiled_count_2q": None,
                    "compiled_cx_count": None,
                    "compiled_ecr_count": None,
                    "compiled_op_counts": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows.append(row)

    selected = rank_compile_rows(rows)
    if selected is None:
        return {
            "status": "unavailable",
            "success": False,
            "error": "no_compile_target_succeeded",
            "logical_circuit": _jsonable(logical),
            "rows": _jsonable(rows),
            "resolution_audit": _jsonable(resolution_audit),
            "selected_backend": None,
            "observation": None,
            "backend_snapshot": None,
            "compiled_circuit_artifacts": [],
        }

    selected_backend = str(selected.get("transpile_backend", selected.get("backend_name", "")))
    compiled_circuit_artifacts: list[dict[str, Any]] = []
    if config.export_circuit_dir is not None:
        compiled_selected = compiled_by_backend.get(selected_backend)
        if compiled_selected is not None:
            artifact = export_compiled_circuit_artifacts(
                compiled_selected,
                output_dir=Path(config.export_circuit_dir),
                stem=str(export_stem or scope),
                metadata={
                    "method": "controller",
                    "scope": str(scope),
                    "backend_name": selected_backend,
                    "seed_transpiler": int(config.seed_transpiler),
                    "optimization_level": int(config.optimization_level),
                },
            )
            compiled_circuit_artifacts.append(dict(artifact))
            selected["compiled_circuit_artifacts"] = [dict(artifact)]

    backend_snapshot = dict(selected.get("target_snapshot", {}) or {})
    if backend_snapshot.get("backend_name") in {None, ""}:
        backend_snapshot["backend_name"] = selected_backend
    observation = {
        "compiled_count_2q": selected.get("compiled_count_2q"),
        "compiled_depth": selected.get("compiled_depth"),
        "compiled_size": selected.get("compiled_size"),
        "compiled_num_qubits": selected.get("compiled_num_qubits"),
        "backend_name": selected_backend,
        "compile_backend": selected_backend,
        "transpile_backend": selected_backend,
        "seed_transpiler": int(config.seed_transpiler),
        "transpile_seed": int(config.seed_transpiler),
        "optimization_level": int(config.optimization_level),
        "transpile_optimization_level": int(config.optimization_level),
        "resolution_kind": selected.get("resolution_kind"),
        "using_fake_backend": selected.get("using_fake_backend"),
        "logical_to_physical": selected.get("logical_to_physical", []),
        "compiled_op_counts": dict(selected.get("compiled_op_counts", {}) or {}),
        "compiled_cx_count": selected.get("compiled_cx_count"),
        "compiled_ecr_count": selected.get("compiled_ecr_count"),
    }
    if compiled_circuit_artifacts:
        observation["compiled_circuit_artifacts"] = list(compiled_circuit_artifacts)
    return {
        "status": "ok",
        "success": True,
        "error": None,
        "logical_circuit": _jsonable(logical),
        "rows": _jsonable(rows),
        "resolution_audit": _jsonable(resolution_audit),
        "selected_backend": _jsonable(selected),
        "observation": _jsonable(observation),
        "backend_snapshot": _jsonable(backend_snapshot),
        "compiled_circuit_artifacts": _jsonable(compiled_circuit_artifacts),
    }


"Built Math: C_final = transpile(U_final(theta_final)|psi_ref>, FakeBackend); audit = argmin_backend(2Q, depth, size)."
def run_final_scaffold_compile_audit(
    *,
    controller: Any,
    config: RealtimeCompileAuditConfig,
) -> dict[str, Any]:
    """Compile the final realtime scaffold to local fake backend targets only."""
    request = _compile_request_payload(config)
    payload: dict[str, Any] = {
        "schema_version": "hh_realtime_final_scaffold_compile_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(config.mode),
        "scope": "final_scaffold",
        "workflow_contract": {
            "compile_only": True,
            "local_fake_only": True,
            "runtime_lookup": False,
            "selection_metric": ["compiled_count_2q", "compiled_depth", "compiled_size", "transpile_backend"],
        },
        "request": dict(request),
        "resolution_audit": [],
        "rows": [],
        "selected_backend": None,
        "observation": None,
        "backend_snapshot": None,
        "logical_circuit": {},
        "compiled_circuit_artifacts": [],
        "success": False,
        "status": "not_run",
        "error": None,
    }
    if str(config.mode) == "off":
        payload["status"] = "off"
        return payload

    try:
        layout = getattr(controller, "current_layout")
        theta_runtime = np.asarray(getattr(controller, "current_theta"), dtype=float).reshape(-1)
        replay_context = getattr(controller, "replay_context")
        num_qubits = _infer_num_qubits(controller=controller)
        labels = []
        if hasattr(controller, "_current_scaffold_labels"):
            try:
                labels = [str(x) for x in controller._current_scaffold_labels()]
            except Exception:
                labels = []
        compiled = _compile_scaffold_snapshot(
            snapshot={
                "layout": layout,
                "theta_runtime": theta_runtime,
                "labels": labels,
            },
            replay_context=replay_context,
            num_qubits=int(num_qubits),
            config=config,
            scope="final_scaffold",
            export_stem="controller_final_scaffold_source",
        )
        payload.update(
            {
                "resolution_audit": compiled.get("resolution_audit", []),
                "rows": compiled.get("rows", []),
                "selected_backend": compiled.get("selected_backend"),
                "observation": compiled.get("observation"),
                "backend_snapshot": compiled.get("backend_snapshot"),
                "logical_circuit": compiled.get("logical_circuit", {}),
                "compiled_circuit_artifacts": compiled.get("compiled_circuit_artifacts", []),
                "success": bool(compiled.get("success", False)),
                "status": str(compiled.get("status", "error")),
                "error": compiled.get("error"),
            }
        )
        if not bool(payload["success"]):
            return payload
        payload["success"] = True
        payload["status"] = "ok"
        return payload
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return _jsonable(payload)


def run_prune_event_compile_audit(
    *,
    controller: Any,
    config: RealtimeCompileAuditConfig,
) -> dict[str, Any]:
    """Compile accepted prune before/after scaffolds captured during a run."""
    request = _compile_request_payload(config)
    payload: dict[str, Any] = {
        "schema_version": "hh_realtime_prune_event_compile_audit_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": str(config.mode),
        "scope": "accepted_prune_events",
        "workflow_contract": {
            "compile_only": True,
            "local_fake_only": True,
            "runtime_lookup": False,
            "accepted_prune_signal": "runtime_parameter_count_delta < 0",
            "selection_metric": ["compiled_count_2q", "compiled_depth", "compiled_size", "transpile_backend"],
            "runtime_parameter_count_is_secondary_diagnostic": True,
        },
        "request": dict(request),
        "accepted_prune_event_count": 0,
        "captured_prune_event_count": 0,
        "compiled_prune_event_count": 0,
        "events": [],
        "success": False,
        "status": "not_run",
        "error": None,
    }
    if str(config.mode) == "off":
        payload["status"] = "off"
        return payload

    ledger_rows = [
        dict(row)
        for row in getattr(controller, "_ledger", [])
        if isinstance(row, Mapping)
    ]
    accepted_rows = accepted_prune_event_rows(ledger_rows)
    captured_events = [
        dict(event)
        for event in getattr(controller, "_compile_audit_prune_events", [])
        if isinstance(event, Mapping)
    ]
    payload["accepted_prune_event_count"] = int(len(accepted_rows))
    payload["captured_prune_event_count"] = int(len(captured_events))
    if not accepted_rows and not captured_events:
        payload["success"] = True
        payload["status"] = "no_prune_events"
        return payload
    if len(captured_events) < len(accepted_rows):
        payload["status"] = "unavailable"
        payload["error"] = (
            "accepted_prune_events_missing_compile_snapshots: "
            f"accepted={len(accepted_rows)} captured={len(captured_events)}"
        )
        payload["events"] = _jsonable(accepted_rows)
        return payload

    try:
        replay_context = getattr(controller, "replay_context")
        num_qubits = _infer_num_qubits(controller=controller)
        compiled_events: list[dict[str, Any]] = []
        for event_index, event in enumerate(captured_events):
            before_snapshot = event.get("before")
            after_snapshot = event.get("after")
            if not isinstance(before_snapshot, Mapping) or not isinstance(after_snapshot, Mapping):
                compiled_events.append(
                    {
                        "event_index": int(event_index),
                        "status": "unavailable",
                        "success": False,
                        "error": "missing_before_or_after_snapshot",
                        "event": _jsonable({k: v for k, v in event.items() if k not in {"before", "after"}}),
                    }
                )
                continue
            event_meta = {
                k: v
                for k, v in event.items()
                if k not in {"before", "after"}
            }
            before = _compile_scaffold_snapshot(
                snapshot=before_snapshot,
                replay_context=replay_context,
                num_qubits=int(num_qubits),
                config=config,
                scope="prune_before",
                export_stem=f"prune_event_{event_index:04d}_before",
            )
            after = _compile_scaffold_snapshot(
                snapshot=after_snapshot,
                replay_context=replay_context,
                num_qubits=int(num_qubits),
                config=config,
                scope="prune_after",
                export_stem=f"prune_event_{event_index:04d}_after",
            )
            before_obs = before.get("observation") if isinstance(before.get("observation"), Mapping) else {}
            after_obs = after.get("observation") if isinstance(after.get("observation"), Mapping) else {}
            reductions = {
                "compiled_count_2q_percent_reduction": _percent_reduction(
                    before_obs.get("compiled_count_2q"),
                    after_obs.get("compiled_count_2q"),
                ),
                "compiled_depth_percent_reduction": _percent_reduction(
                    before_obs.get("compiled_depth"),
                    after_obs.get("compiled_depth"),
                ),
                "compiled_size_percent_reduction": _percent_reduction(
                    before_obs.get("compiled_size"),
                    after_obs.get("compiled_size"),
                ),
                "runtime_parameter_count_percent_reduction": _percent_reduction(
                    event_meta.get("runtime_parameter_count_before"),
                    event_meta.get("runtime_parameter_count_after"),
                ),
            }
            compiled_events.append(
                {
                    "event_index": int(event_index),
                    "status": "ok" if bool(before.get("success")) and bool(after.get("success")) else "error",
                    "success": bool(before.get("success")) and bool(after.get("success")),
                    "event": _jsonable(event_meta),
                    "before": _jsonable(before),
                    "after": _jsonable(after),
                    "percent_reductions": _jsonable(reductions),
                }
            )
        payload["events"] = _jsonable(compiled_events)
        payload["compiled_prune_event_count"] = int(
            sum(1 for event in compiled_events if bool(event.get("success", False)))
        )
        if int(payload["compiled_prune_event_count"]) != int(len(captured_events)):
            payload["status"] = "error"
            payload["error"] = "one_or_more_prune_events_failed_to_compile"
            return payload
        payload["success"] = True
        payload["status"] = "ok"
        return payload
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        return _jsonable(payload)


def compile_audit_summary_mirrors(audit_payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit_payload, Mapping):
        return {}
    observation = audit_payload.get("observation")
    request = audit_payload.get("request")
    backend_snapshot = audit_payload.get("backend_snapshot")
    if not isinstance(observation, Mapping):
        return {}
    mirrors = {
        "oracle_compile_observation": dict(observation),
        "oracle_backend_snapshot": dict(backend_snapshot) if isinstance(backend_snapshot, Mapping) else {},
        "oracle_compile_request": dict(request) if isinstance(request, Mapping) else {},
    }
    prune_audit = audit_payload.get("prune_event_audit")
    if isinstance(prune_audit, Mapping):
        mirrors.update(
            {
                "prune_compile_audit_status": prune_audit.get("status"),
                "prune_compile_audit_success": bool(prune_audit.get("success", False)),
                "prune_compile_audit_event_count": int(prune_audit.get("accepted_prune_event_count", 0) or 0),
                "prune_compile_audit_compiled_event_count": int(prune_audit.get("compiled_prune_event_count", 0) or 0),
            }
        )
    return mirrors


__all__ = [
    "RealtimeCompileAuditConfig",
    "accepted_prune_event_rows",
    "build_compile_audit_config_from_args",
    "compile_audit_summary_mirrors",
    "normalize_compile_audit_mode",
    "run_final_scaffold_compile_audit",
    "run_prune_event_compile_audit",
]
