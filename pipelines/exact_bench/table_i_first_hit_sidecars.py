#!/usr/bin/env python3
"""Generate Paper-I Table-I first-hit Qiskit resource sidecars.

This pass is intentionally conservative.  It may compile a sidecar only when a
SNAKE/Phase3 artifact exposes enough first-crossing history, ansatz structure,
source hash, and reference-state information to reconstruct the first accepted
hit.  Otherwise it emits an inventory row explaining why a rerun or richer
artifact is needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.generic_static_metric_enrichment import (
    SNAKE_FIRST_CROSSING_COST_SCHEMA,
    SNAKE_TABLE_I_ALGORITHM_ID,
    _sha256_json_without_snake_sidecars,
    table_i_threshold_cost_from_row,
)
from pipelines.exact_bench.snake_table_i_measurement_work import (
    snake_algorithmic_work_from_payload,
    snake_deterministic_shot_proxy_from_payload,
)
from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    TableICompileUnavailable,
    compile_table_i_pauli_label_groups,
)
from pipelines.exact_bench.static_benchmark_runtime import (
    _paper_i_history_row_acceptance_status,
)

INVENTORY_SCHEMA = "paper_i_first_hit_cost_inventory_v1"
SNAKE_SIDECAR_SOURCE_KIND = "snake_qiskit_compiled_first_hit_ansatz_circuit"
SIDECAR_KEY = "paper_i_first_crossing_compiled_cost"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_json_mapping(path: str | Path | None) -> dict[str, Any] | None:
    if path is None or path == "":
        return None
    candidate = Path(str(path))
    if not candidate.exists() or not candidate.is_file():
        return None
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _resolve_existing_path(raw: Any, *, base_path: str | Path | None = None) -> Path | None:
    if raw is None or raw == "":
        return None
    candidate = Path(str(raw))
    if candidate.exists():
        return candidate
    if base_path is not None and base_path != "":
        base = Path(str(base_path))
        rel = base.parent / candidate
        if rel.exists():
            return rel
    return None


def _adapt_payload(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    nested = payload.get("adapt_vqe")
    return nested if isinstance(nested, Mapping) else payload


def _mapping_at(payload: Mapping[str, Any] | None, key: str) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get(key)
    return value if isinstance(value, Mapping) else None


def _first_crossing(*payloads: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    for payload in payloads:
        direct = _mapping_at(payload, "paper_i_first_crossing")
        if direct is not None:
            return direct
        adapt = _adapt_payload(payload)
        nested = _mapping_at(adapt, "paper_i_first_crossing")
        if nested is not None:
            return nested
        result = _mapping_at(payload, "result")
        nested = _mapping_at(result, "paper_i_first_crossing")
        if nested is not None:
            return nested
    return None


def _history_rows(payload: Mapping[str, Any] | None) -> tuple[Mapping[str, Any], ...]:
    adapt = _adapt_payload(payload)
    rows = adapt.get("history") if isinstance(adapt, Mapping) else None
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return ()
    return tuple(row for row in rows if isinstance(row, Mapping))


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return float(out)


def _int_or_none(value: Any) -> int | None:
    parsed = _float_or_none(value)
    if parsed is None or not float(parsed).is_integer():
        return None
    return int(parsed)


def _list_of_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [str(value)] if str(value).strip() else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if str(item).strip()]
    return []


def _history_row_selected_labels(row: Mapping[str, Any]) -> list[str]:
    labels = _list_of_text(row.get("selected_ops"))
    if labels:
        return labels
    labels = _list_of_text(row.get("selected_logical_ops"))
    if labels:
        return labels
    label = str(row.get("selected_logical_op") or row.get("selected_op") or "").strip()
    return [label] if label else []


def _history_row_positions(row: Mapping[str, Any], *, count: int, current_len: int) -> list[int]:
    raw = row.get("selected_positions")
    positions: list[int] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            parsed = _int_or_none(item)
            if parsed is not None:
                positions.append(int(parsed))
    elif row.get("selected_position") is not None and row.get("selected_position") != "":
        parsed = _int_or_none(row.get("selected_position"))
        if parsed is not None:
            positions.append(int(parsed))
    if len(positions) == int(count):
        return positions
    if not positions:
        return [int(current_len + idx) for idx in range(int(count))]
    return []


def _replay_first_hit_labels(
    history: Sequence[Mapping[str, Any]],
    *,
    history_position_tau: int,
    k_tau: int | None = None,
    operator_count_at_crossing: int | None = None,
) -> tuple[list[str] | None, dict[str, Any]]:
    selected: list[str] = []
    committed_operator_count: int | None = None
    initial_operator_count: int | None = None
    committed_depth: int | None = None
    accepted_count = 0
    missing: list[str] = []
    acceptance_reason_counts: Counter[str] = Counter()
    for idx, row in enumerate(history, start=1):
        if idx > int(history_position_tau):
            break
        accepted, reason = _paper_i_history_row_acceptance_status(
            row,
            committed_operator_count=committed_operator_count,
            initial_operator_count=initial_operator_count,
            committed_depth=committed_depth,
        )
        acceptance_reason_counts[str(reason)] += 1
        row_operator_count = _int_or_none(row.get("logical_num_parameters_after_opt"))
        if not accepted:
            if reason == "preexisting_initial_operator_count" and row_operator_count is not None and initial_operator_count is None:
                initial_operator_count = int(row_operator_count)
            continue
        labels = _history_row_selected_labels(row)
        if not labels:
            missing.append(f"history[{idx}].selected_ops")
            continue
        positions = _history_row_positions(row, count=len(labels), current_len=len(selected))
        if len(positions) != len(labels):
            missing.append(f"history[{idx}].selected_positions")
            continue
        for label, pos in zip(labels, positions):
            insert_at = max(0, min(int(pos), len(selected)))
            selected.insert(insert_at, str(label))
        accepted_count += 1
        if row_operator_count is not None:
            committed_operator_count = int(row_operator_count)
        row_depth = _int_or_none(row.get("depth"))
        if row_depth is not None:
            committed_depth = int(row_depth)
    if missing:
        return None, {
            "status": "missing_history_replay_fields",
            "missing_fields": missing,
            "accepted_replayed_count": int(accepted_count),
            "acceptance_reason_counts": dict(sorted(acceptance_reason_counts.items())),
        }
    if k_tau is not None and int(k_tau) != int(accepted_count):
        return None, {
            "status": "k_tau_replay_mismatch",
            "expected_k_tau": int(k_tau),
            "accepted_replayed_count": int(accepted_count),
            "acceptance_reason_counts": dict(sorted(acceptance_reason_counts.items())),
        }
    if operator_count_at_crossing is not None and int(operator_count_at_crossing) != len(selected):
        return None, {
            "status": "operator_count_replay_mismatch",
            "operator_count_at_crossing": int(operator_count_at_crossing),
            "replayed_operator_count": int(len(selected)),
            "acceptance_reason_counts": dict(sorted(acceptance_reason_counts.items())),
        }
    return selected, {
        "status": "ok",
        "accepted_replayed_count": int(accepted_count),
        "replayed_operator_count": int(len(selected)),
        "acceptance_reason_counts": dict(sorted(acceptance_reason_counts.items())),
    }


def _layout_blocks(payload: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    adapt = _adapt_payload(payload)
    parameterization = adapt.get("parameterization") if isinstance(adapt, Mapping) else None
    if not isinstance(parameterization, Mapping):
        parameterization = payload.get("parameterization") if isinstance(payload, Mapping) else None
    blocks = parameterization.get("blocks") if isinstance(parameterization, Mapping) else None
    if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
        return []
    return [block for block in blocks if isinstance(block, Mapping)]


def _groups_from_layout_blocks(
    *,
    labels: Sequence[str],
    payload: Mapping[str, Any] | None,
) -> tuple[list[list[str]] | None, dict[str, Any]]:
    blocks = _layout_blocks(payload)
    if not blocks:
        return None, {"status": "missing_parameterization_blocks", "missing_fields": ["parameterization.blocks"]}
    used: set[int] = set()
    groups: list[list[str]] = []
    missing_labels: list[str] = []
    for label in labels:
        match_idx: int | None = None
        for idx, block in enumerate(blocks):
            if idx in used:
                continue
            if str(block.get("candidate_label") or "") == str(label):
                match_idx = idx
                break
        if match_idx is None:
            missing_labels.append(str(label))
            continue
        used.add(int(match_idx))
        terms_raw = blocks[match_idx].get("runtime_terms_exyz")
        if not isinstance(terms_raw, Sequence) or isinstance(terms_raw, (str, bytes)):
            return None, {"status": "bad_parameterization_runtime_terms", "label": str(label)}
        group: list[str] = []
        for raw in terms_raw:
            if not isinstance(raw, Mapping):
                continue
            pauli = str(raw.get("pauli_exyz") or "").strip().lower()
            if pauli:
                group.append(pauli)
        groups.append(group)
    if missing_labels:
        return None, {
            "status": "selected_label_missing_from_parameterization",
            "missing_labels": missing_labels,
            "available_label_count": int(len(blocks)),
        }
    return groups, {"status": "ok", "source": "parameterization.blocks"}


def _num_qubits_from_groups(groups: Sequence[Sequence[str]], payload: Mapping[str, Any] | None) -> int | None:
    for group in groups:
        for label in group:
            if str(label):
                return len(str(label))
    adapt = _adapt_payload(payload)
    for key in ("num_qubits", "nq", "total_qubits"):
        parsed = _int_or_none(adapt.get(key) if isinstance(adapt, Mapping) else None)
        if parsed is not None and parsed > 0:
            return int(parsed)
        parsed = _int_or_none(payload.get(key) if isinstance(payload, Mapping) else None)
        if parsed is not None and parsed > 0:
            return int(parsed)
    return None


def _reference_state_from_payload(payload: Mapping[str, Any] | None, *, num_qubits: int) -> tuple[np.ndarray | None, str]:
    adapt = _adapt_payload(payload)
    for key in ("hf_bitstring_qn_to_q0", "reference_bitstring_qn_to_q0", "hf_bitstring"):
        raw = adapt.get(key) if isinstance(adapt, Mapping) else None
        if (raw is None or raw == "") and isinstance(payload, Mapping):
            raw = payload.get(key)
        text = str(raw or "").strip()
        if text and set(text) <= {"0", "1"} and len(text) == int(num_qubits):
            vec = np.zeros(1 << int(num_qubits), dtype=complex)
            vec[int(text, 2)] = 1.0 + 0.0j
            return vec, f"{key}_basis_state"
    return None, "missing_reference_state_bitstring"


def _sidecar_work_fields(
    crossing_row: Mapping[str, Any] | None,
    *,
    source_payload: Mapping[str, Any] | None = None,
    history_position: int | None = None,
    source_label: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(crossing_row, Mapping):
        return (
            {"S_alg": None, "S_norm": None, "S_alg_missing_reason": "first_hit_history_row_missing"},
            {"status": "missing", "reason": "first_hit_history_row_missing"},
        )
    legacy_crossing_work = {
        key: float(value)
        for key in ("S_alg_at_crossing", "first_hit_S_alg", "S_alg")
        if (value := _float_or_none(crossing_row.get(key))) is not None and value >= 0.0
    }
    if isinstance(source_payload, Mapping):
        scoped_work, scoped_audit = snake_algorithmic_work_from_payload(
            source_payload,
            scope="display_prefix",
            history_position=history_position,
            source_label=source_label,
        )
        if scoped_work.get("S_alg_status") == "ok" and scoped_work.get("S_alg") is not None:
            fields = {
                "S_alg": float(scoped_work["S_alg"]),
                "S_norm": float(scoped_work["S_alg"]),
                "S_alg_missing_reason": None,
                "S_alg_status": "ok",
                "algorithmic_measurement_work": scoped_work.get("algorithmic_measurement_work"),
                "table_i_measurement_event_ledger": scoped_work.get("table_i_measurement_event_ledger"),
            }
            if legacy_crossing_work:
                fields["legacy_work_proxies"] = legacy_crossing_work
            for key in (
                "S_alg_N_H_outer_eval",
                "S_alg_N_grad_probe",
                "S_alg_N_metric_probe",
                "S_alg_N_H_refit_eval",
                "S_alg_N_other_quantum",
            ):
                if key in scoped_work:
                    fields[key] = scoped_work[key]
            return fields, {"status": "ok", "source_key": "snake_algorithmic_work_from_payload", "audit": scoped_audit}
        reason = str(scoped_work.get("S_alg_missing_reason") or scoped_work.get("S_alg_status") or "first_hit_algorithmic_work_ledger_missing")
        return (
            {
                "S_alg": None,
                "S_norm": None,
                "S_alg_status": str(scoped_work.get("S_alg_status") or reason),
                "S_alg_missing_reason": reason,
                "algorithmic_measurement_work": scoped_work.get("algorithmic_measurement_work"),
                "legacy_work_proxies": legacy_crossing_work or None,
            },
            {"status": "missing", "reason": reason, "audit": scoped_audit},
        )
    return (
        {
            "S_alg": None,
            "S_norm": None,
            "S_alg_status": "missing",
            "S_alg_missing_reason": "first_hit_algorithmic_work_ledger_missing",
            "legacy_work_proxies": legacy_crossing_work or None,
        },
        {"status": "missing", "reason": "first_hit_algorithmic_work_ledger_missing"},
    )


def _source_result_path_from_payload(payload: Mapping[str, Any] | None, *, payload_path: str | Path | None = None) -> Path | None:
    if isinstance(payload, Mapping):
        result = _mapping_at(payload, "result")
        for source in (payload, result):
            if not isinstance(source, Mapping):
                continue
            resolved = _resolve_existing_path(source.get("result_json"), base_path=payload_path)
            if resolved is not None:
                return resolved
    return _resolve_existing_path(payload_path)


def build_snake_first_hit_sidecar_for_payload(
    *,
    payload: Mapping[str, Any],
    payload_path: str | Path | None = None,
    threshold: float = 2e-4,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Return ``(sidecar, inventory_row)`` for one SNAKE/Phase3 payload."""

    wrapper_result = _mapping_at(payload, "result") or payload
    source_result_path = _source_result_path_from_payload(wrapper_result, payload_path=payload_path)
    if source_result_path is None:
        return None, {
            "status": "rerun_needed",
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["source_result_path"],
            "rerun_needed_reason": "source_result_path_missing_or_unreadable",
        }
    source_payload = _load_json_mapping(source_result_path)
    if source_payload is None:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["source_result_payload"],
            "rerun_needed_reason": "source_result_payload_unreadable",
        }
    crossing = _first_crossing(wrapper_result, source_payload)
    if not isinstance(crossing, Mapping):
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["paper_i_first_crossing"],
            "rerun_needed_reason": "paper_i_first_crossing_missing",
        }
    tau = _float_or_none(crossing.get("tau_phys") if crossing.get("tau_phys") is not None else crossing.get("threshold"))
    primary_error = _float_or_none(crossing.get("primary_error_at_crossing"))
    history_position = _int_or_none(crossing.get("history_position_tau"))
    k_tau = _int_or_none(crossing.get("k_tau"))
    operator_count = _int_or_none(crossing.get("operator_count_at_crossing"))
    reached = crossing.get("reached") is True or str(crossing.get("status") or "").lower() == "reached"
    missing_crossing = []
    if tau is None:
        missing_crossing.append("paper_i_first_crossing.tau_phys")
    if primary_error is None:
        missing_crossing.append("paper_i_first_crossing.primary_error_at_crossing")
    if history_position is None:
        missing_crossing.append("paper_i_first_crossing.history_position_tau")
    if missing_crossing:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": missing_crossing,
            "rerun_needed_reason": "paper_i_first_crossing_required_fields_missing",
        }
    if not math.isclose(float(tau), float(threshold), rel_tol=0.0, abs_tol=1e-12):
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["paper_i_first_crossing.tau_phys_matches_requested_threshold"],
            "rerun_needed_reason": "paper_i_first_crossing_tau_mismatch",
            "tau_phys": tau,
        }
    if not reached or float(primary_error) > float(threshold):
        return None, {
            "status": "not_reached",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": [],
            "rerun_needed_reason": None,
            "history_position_tau": history_position,
            "primary_error_at_crossing": primary_error,
        }
    history = _history_rows(source_payload)
    if not history:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["adapt_vqe.history"],
            "rerun_needed_reason": "adapt_history_missing",
        }
    if int(history_position) < 1 or int(history_position) > len(history):
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["history[history_position_tau]"],
            "rerun_needed_reason": "history_position_tau_out_of_range",
            "history_position_tau": history_position,
            "history_row_count": len(history),
        }
    labels, replay_meta = _replay_first_hit_labels(
        history,
        history_position_tau=int(history_position),
        k_tau=k_tau,
        operator_count_at_crossing=operator_count,
    )
    if labels is None:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": list(replay_meta.get("missing_fields", [])) or [str(replay_meta.get("status"))],
            "rerun_needed_reason": str(replay_meta.get("status")),
            "history_position_tau": history_position,
            "replay": replay_meta,
        }
    groups, groups_meta = _groups_from_layout_blocks(labels=labels, payload=source_payload)
    if groups is None:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": list(groups_meta.get("missing_fields", [])) or [str(groups_meta.get("status"))],
            "rerun_needed_reason": str(groups_meta.get("status")),
            "history_position_tau": history_position,
            "replay": replay_meta,
        }
    num_qubits = _num_qubits_from_groups(groups, source_payload)
    if num_qubits is None:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["num_qubits"],
            "rerun_needed_reason": "num_qubits_missing",
            "history_position_tau": history_position,
            "replay": replay_meta,
        }
    reference_state, reference_state_status = _reference_state_from_payload(source_payload, num_qubits=int(num_qubits))
    if reference_state is None:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": "missing",
            "missing_fields": ["reference_state"],
            "rerun_needed_reason": reference_state_status,
            "history_position_tau": history_position,
            "replay": replay_meta,
        }
    try:
        compiled = compile_table_i_pauli_label_groups(
            pauli_label_groups=groups,
            num_qubits=int(num_qubits),
            reference_state=reference_state,
            source_kind=SNAKE_SIDECAR_SOURCE_KIND,
        )
    except TableICompileUnavailable as exc:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "failed",
            "work_resource_status": "missing",
            "missing_fields": ["qiskit_compile"],
            "rerun_needed_reason": exc.status,
            "compile_error": exc.reason,
            "history_position_tau": history_position,
            "replay": replay_meta,
        }
    crossing_row = history[int(history_position) - 1]
    work_fields, work_meta = _sidecar_work_fields(
        crossing_row,
        source_payload=source_payload,
        history_position=int(history_position),
        source_label=str(source_result_path),
    )
    deterministic_fields, deterministic_audit = snake_deterministic_shot_proxy_from_payload(
        source_payload,
        scope="display_prefix",
        history_position=int(history_position),
        source_label=str(source_result_path),
    )
    work_fields = dict(work_fields)
    work_fields["snake_deterministic_shot_proxy"] = deterministic_audit
    if deterministic_audit.get("status") == "ok":
        work_fields.update(deterministic_fields)
    source_hash = _sha256_json_without_snake_sidecars(source_result_path)
    if not source_hash:
        return None, {
            "status": "rerun_needed",
            "source_result_path": str(source_result_path),
            "paper_i_first_crossing": dict(crossing),
            "compiled_resource_status": "missing",
            "work_resource_status": work_meta.get("status"),
            "missing_fields": ["source_result_sha256"],
            "rerun_needed_reason": "source_result_sha256_unavailable",
            "history_position_tau": history_position,
            "replay": replay_meta,
        }
    sidecar = {
        "schema": SNAKE_FIRST_CROSSING_COST_SCHEMA,
        "source_kind": SNAKE_SIDECAR_SOURCE_KIND,
        "first_hit_cost_source_kind": SNAKE_SIDECAR_SOURCE_KIND,
        "benchmark_id": str(
            crossing.get("benchmark_id")
            or wrapper_result.get("benchmark_id")
            or wrapper_result.get("case_id")
            or ""
        ),
        "family": wrapper_result.get("family"),
        "case_id": wrapper_result.get("case_id"),
        "algorithm_id": SNAKE_TABLE_I_ALGORITHM_ID,
        "tau_phys": float(threshold),
        "current_target_threshold": float(threshold),
        "history_position_tau": int(history_position),
        "k_tau": None if k_tau is None else int(k_tau),
        "primary_error_at_crossing": float(primary_error),
        "source_result_path": str(source_result_path),
        "source_result_sha256": str(source_hash),
        "source_result_hash_convention": "canonical_json_without_snake_sidecars_v1",
        "reconstructability_status": "ok" if work_meta.get("status") == "ok" else "compiled_ok_work_missing",
        "selected_operator_labels": [str(x) for x in labels],
        "selected_operator_pauli_labels_exyz": [[str(x) for x in group] for group in groups],
        "reference_state_status": reference_state_status,
        "first_hit_theta_status": "not_required_for_structural_gate_count_compile",
        **compiled,
        **work_fields,
    }
    status = "sidecar_generated" if work_meta.get("status") == "ok" else "sidecar_generated_work_missing"
    return sidecar, {
        "status": status,
        "source_result_path": str(source_result_path),
        "source_result_sha256": str(source_hash),
        "sidecar_key": SIDECAR_KEY,
        "compiled_resource_status": "ok",
        "work_resource_status": str(work_meta.get("status")),
        "S_alg_missing_reason": sidecar.get("S_alg_missing_reason"),
        "missing_fields": [] if work_meta.get("status") == "ok" else ["first_hit_algorithmic_work_ledger"],
        "rerun_needed_reason": None if work_meta.get("status") == "ok" else str(work_meta.get("reason") or "first_hit_algorithmic_work_ledger_missing_for_S_alg"),
        "history_position_tau": int(history_position),
        "k_tau": None if k_tau is None else int(k_tau),
        "primary_error_at_crossing": float(primary_error),
        "benchmark_id": sidecar.get("benchmark_id"),
        "family": sidecar.get("family"),
        "case_id": sidecar.get("case_id"),
        "reconstructability_status": sidecar.get("reconstructability_status"),
        "compiled_count_2q_total": sidecar.get("compiled_count_2q_total"),
        "compiled_depth_2q_total": sidecar.get("compiled_depth_2q_total"),
        "compiled_depth_total": sidecar.get("compiled_depth_total"),
        "replay": replay_meta,
        "parameterization_reconstruction": groups_meta,
    }


def _is_snake_summary_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("algorithm_id") or "") == SNAKE_TABLE_I_ALGORITHM_ID or str(row.get("method") or "").strip().lower() == "snake"


def _payload_for_summary_row(row: Mapping[str, Any]) -> tuple[dict[str, Any] | None, Path | None]:
    for key in ("payload_path", "source_payload_path", "result_json"):
        path = _resolve_existing_path(row.get(key))
        if path is not None:
            payload = _load_json_mapping(path)
            if payload is not None:
                return payload, path
    return None, None


def _numeric_equal(a: Any, b: Any, *, atol: float = 1e-12) -> bool:
    try:
        af = float(a)
        bf = float(b)
    except Exception:
        return a == b
    if not math.isfinite(af) or not math.isfinite(bf):
        return a == b
    return abs(af - bf) <= float(atol)


def _sidecar_conflict_fields(existing: Mapping[str, Any], generated: Mapping[str, Any]) -> list[str]:
    keys = (
        "source_result_sha256",
        "history_position_tau",
        "k_tau",
        "primary_error_at_crossing",
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
        "S_alg",
        "S_norm",
    )
    conflicts: list[str] = []
    for key in keys:
        if key not in existing and key not in generated:
            continue
        if not _numeric_equal(existing.get(key), generated.get(key)):
            conflicts.append(key)
    return conflicts


def _validated_existing_sidecar_cost(
    row: Mapping[str, Any],
    *,
    threshold: float,
    payload_path: Path | None,
) -> tuple[bool, Mapping[str, Any] | None]:
    if not isinstance(row.get(SIDECAR_KEY), Mapping):
        return False, None
    cost = table_i_threshold_cost_from_row(
        algorithm_id=SNAKE_TABLE_I_ALGORITHM_ID,
        row=row,
        threshold=float(threshold),
        record={
            "record_id": row.get("record_id"),
            "case_id": row.get("case_id") or row.get("benchmark_id"),
            "algorithm_id": SNAKE_TABLE_I_ALGORITHM_ID,
        },
        result_path=payload_path,
    )
    valid = (
        str(cost.get("threshold_status") or "") == "ok_native_first_hit"
        and cost.get("resource_display_allowed") is True
        and str(cost.get("compiled_resource_validation_status") or "") == "ok"
        and str(cost.get("sidecar_validation_status") or "") == "ok"
        and cost.get("sidecar_hash_verified") is True
    )
    return bool(valid), cost


def _promote_sidecar_into_summary_row(
    row: Mapping[str, Any],
    *,
    sidecar: Mapping[str, Any],
    threshold: float,
    payload_path: Path | None,
) -> dict[str, Any]:
    candidate = dict(row)
    candidate[SIDECAR_KEY] = dict(sidecar)
    if not isinstance(candidate.get("paper_i_first_crossing"), Mapping):
        candidate["paper_i_first_crossing"] = {
            "schema": "paper_i_first_crossing_v1",
            "status": "reached",
            "reached": True,
            "tau_phys": float(threshold),
            "benchmark_id": sidecar.get("benchmark_id"),
            "history_position_tau": sidecar.get("history_position_tau"),
            "k_tau": sidecar.get("k_tau"),
            "primary_error_at_crossing": sidecar.get("primary_error_at_crossing"),
        }
    cost = table_i_threshold_cost_from_row(
        algorithm_id=SNAKE_TABLE_I_ALGORITHM_ID,
        row=candidate,
        threshold=float(threshold),
        record={
            "record_id": row.get("record_id"),
            "case_id": row.get("case_id") or sidecar.get("case_id") or sidecar.get("benchmark_id"),
            "algorithm_id": SNAKE_TABLE_I_ALGORITHM_ID,
        },
        result_path=payload_path,
    )
    threshold_status = str(cost.get("threshold_status") or "unknown")
    cost_included = (
        threshold_status == "ok_native_first_hit"
        and cost.get("resource_display_allowed") is True
        and str(cost.get("compiled_resource_validation_status") or "") == "ok"
    )
    out = dict(row)
    out[SIDECAR_KEY] = dict(sidecar)
    out.update(
        {
            "algorithm_id": SNAKE_TABLE_I_ALGORITHM_ID,
            "threshold_status": threshold_status,
            "cost_included": bool(cost_included),
            "abs_delta_e": cost.get("abs_delta_e"),
            "S_alg": cost.get("S_alg") if cost_included else None,
            "S_norm": cost.get("S_norm") if cost_included else None,
            "count_2q": cost.get("count_2q") if cost_included else None,
            "depth_2q": cost.get("depth_2q") if cost_included else None,
            "circuit_depth": cost.get("circuit_depth") if cost_included else None,
            "cost_source": cost.get("cost_source"),
            "source": cost.get("source"),
            "first_hit_semantics": cost.get("first_hit_semantics"),
            "method_cost_semantics": cost.get("method_cost_semantics"),
            "resource_display_allowed": bool(cost.get("resource_display_allowed") is True and cost_included),
            "compiled_resource_validation_status": cost.get("compiled_resource_validation_status"),
            "compiled_resource_validation_reason": cost.get("compiled_resource_validation_reason"),
            "first_hit_cost_source_kind": cost.get("first_hit_cost_source_kind"),
            "source_resource_fields_present": cost.get("source_resource_fields_present"),
            "sidecar_validation_status": cost.get("sidecar_validation_status"),
            "sidecar_validation_reason": cost.get("sidecar_validation_reason"),
            "sidecar_hash_verified": cost.get("sidecar_hash_verified"),
            "sidecar_source_kind": cost.get("sidecar_source_kind"),
            "snake_first_crossing_cost_sidecar_key": cost.get("snake_first_crossing_cost_sidecar_key"),
            "snake_first_crossing_history_position_tau": cost.get("snake_first_crossing_history_position_tau"),
            "source_result_path": sidecar.get("source_result_path"),
            "source_result_sha256": cost.get("source_result_sha256"),
            "S_alg_missing_reason": cost.get("S_alg_missing_reason"),
            "reconstructability_status": sidecar.get("reconstructability_status"),
        }
    )
    return out


def inventory_summary_payload(
    summary: Mapping[str, Any],
    *,
    threshold: float = 2e-4,
    write_sidecars_dir: str | Path | None = None,
    update_summary_rows: bool = False,
    replace_existing_sidecars: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows_raw = summary.get("row_results")
    rows = list(rows_raw) if isinstance(rows_raw, Sequence) and not isinstance(rows_raw, (str, bytes)) else []
    inventory_rows: list[dict[str, Any]] = []
    output_rows: list[Any] = []
    sidecar_dir = Path(write_sidecars_dir) if write_sidecars_dir is not None and write_sidecars_dir != "" else None
    for idx, row_raw in enumerate(rows):
        if not isinstance(row_raw, Mapping):
            output_rows.append(row_raw)
            continue
        row = dict(row_raw)
        if not _is_snake_summary_row(row):
            output_rows.append(row)
            continue
        payload, payload_path = _payload_for_summary_row(row)
        if payload is None:
            inv = {
                "row_index": int(idx),
                "record_id": row.get("record_id"),
                "family": row.get("family"),
                "case_id": row.get("case_id"),
                "algorithm_id": SNAKE_TABLE_I_ALGORITHM_ID,
                "status": "rerun_needed",
                "compiled_resource_status": "missing",
                "work_resource_status": "missing",
                "missing_fields": ["payload_path"],
                "rerun_needed_reason": "payload_path_missing_or_unreadable",
            }
            inventory_rows.append(inv)
            output_rows.append(row)
            continue
        sidecar, inv = build_snake_first_hit_sidecar_for_payload(
            payload=payload,
            payload_path=payload_path,
            threshold=float(threshold),
        )
        inv.update(
            {
                "row_index": int(idx),
                "record_id": row.get("record_id"),
                "family": inv.get("family") or row.get("family"),
                "case_id": inv.get("case_id") or row.get("case_id"),
                "algorithm_id": SNAKE_TABLE_I_ALGORITHM_ID,
                "source_payload_path": str(payload_path) if payload_path is not None else None,
            }
        )
        if sidecar is not None and sidecar_dir is not None:
            stem = str(row.get("record_id") or row.get("case_id") or f"snake_row_{idx}").replace("/", "__")
            sidecar_path = sidecar_dir / f"{stem}.paper_i_first_crossing_compiled_cost.json"
            _write_json(sidecar_path, sidecar)
            inv["sidecar_artifact_path"] = str(sidecar_path)
        inventory_rows.append(inv)
        if sidecar is not None and update_summary_rows:
            existing_valid, existing_cost = _validated_existing_sidecar_cost(
                row,
                threshold=float(threshold),
                payload_path=payload_path,
            )
            existing_sidecar = row.get(SIDECAR_KEY) if isinstance(row.get(SIDECAR_KEY), Mapping) else None
            if existing_valid and existing_sidecar is not None and not bool(replace_existing_sidecars):
                conflicts = _sidecar_conflict_fields(existing_sidecar, sidecar)
                inv["promotion_mode"] = (
                    "existing_valid_sidecar_conflict_preserved"
                    if conflicts
                    else "existing_valid_sidecar_preserved"
                )
                inv["existing_valid_sidecar_preserved"] = True
                inv["replace_existing_sidecars"] = False
                inv["sidecar_conflict_fields"] = conflicts
                inv["existing_threshold_status"] = None if existing_cost is None else existing_cost.get("threshold_status")
                inv["existing_cost_source"] = None if existing_cost is None else existing_cost.get("cost_source")
                output_rows.append(row)
            else:
                inv["promotion_mode"] = (
                    "generated_sidecar_replaced_existing"
                    if existing_sidecar is not None and bool(replace_existing_sidecars)
                    else "generated_sidecar_promoted"
                )
                inv["replace_existing_sidecars"] = bool(replace_existing_sidecars)
                output_rows.append(
                    _promote_sidecar_into_summary_row(
                        row,
                        sidecar=sidecar,
                        threshold=float(threshold),
                        payload_path=payload_path,
                    )
                )
        else:
            output_rows.append(row)
    status_counts = Counter(str(row.get("status") or "unknown") for row in inventory_rows)
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "threshold": float(threshold),
        "row_count": int(len(inventory_rows)),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": inventory_rows,
    }
    output_summary = copy.deepcopy(dict(summary))
    if update_summary_rows:
        output_summary["row_results"] = output_rows
    output_summary["paper_i_first_hit_cost_inventory"] = inventory
    return inventory, output_summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-json", type=Path, default=None, help="Calibrated fixed-accuracy summary JSON to inventory/update.")
    parser.add_argument("--input-json", type=Path, default=None, help="Single generic_static_single/result JSON payload to inventory.")
    parser.add_argument("--threshold", type=float, default=2e-4)
    parser.add_argument("--output-json", type=Path, default=None, help="Inventory JSON path.")
    parser.add_argument("--output-summary-json", type=Path, default=None, help="Optional summary copy with generated sidecars promoted into row_results.")
    parser.add_argument("--sidecar-output-dir", type=Path, default=None, help="Optional directory for generated sidecar JSON files.")
    parser.add_argument(
        "--replace-existing-sidecars",
        action="store_true",
        help="Allow generated sidecars to replace an existing valid SNAKE first-hit sidecar in the derived summary copy.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.summary_json is None and args.input_json is None:
        raise SystemExit("provide --summary-json or --input-json")
    if args.summary_json is not None:
        summary = _load_json_mapping(args.summary_json)
        if summary is None:
            raise SystemExit(f"Could not read summary JSON: {args.summary_json}")
        inventory, output_summary = inventory_summary_payload(
            summary,
            threshold=float(args.threshold),
            write_sidecars_dir=args.sidecar_output_dir,
            update_summary_rows=args.output_summary_json is not None,
            replace_existing_sidecars=bool(args.replace_existing_sidecars),
        )
        if args.output_summary_json is not None:
            _write_json(args.output_summary_json, output_summary)
    else:
        payload = _load_json_mapping(args.input_json)
        if payload is None:
            raise SystemExit(f"Could not read input JSON: {args.input_json}")
        sidecar, row = build_snake_first_hit_sidecar_for_payload(
            payload=payload,
            payload_path=args.input_json,
            threshold=float(args.threshold),
        )
        if sidecar is not None and args.sidecar_output_dir is not None:
            stem = args.input_json.stem
            path = args.sidecar_output_dir / f"{stem}.paper_i_first_crossing_compiled_cost.json"
            _write_json(path, sidecar)
            row["sidecar_artifact_path"] = str(path)
        inventory = {
            "schema": INVENTORY_SCHEMA,
            "threshold": float(args.threshold),
            "row_count": 1,
            "status_counts": {str(row.get("status") or "unknown"): 1},
            "rows": [row],
        }
    if args.output_json is not None:
        _write_json(args.output_json, inventory)
    print(json.dumps(inventory, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
