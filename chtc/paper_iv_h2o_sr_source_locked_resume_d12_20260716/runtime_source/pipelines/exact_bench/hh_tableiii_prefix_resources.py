#!/usr/bin/env python3
"""Export exact per-prefix Qiskit resources for replayable Paper-I HH Table-III rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.exact_bench.table_i_qiskit_resource_compile import (
    TABLE_I_QISKIT_COMPILE_CONVENTION,
    TableICompileUnavailable,
    compile_table_i_pauli_label_groups,
)
from pipelines.reporting.audit_paper_i_hh_prefix_replayability import (
    DEFAULT_SOURCE_MAP,
    build_audit,
)

DEFAULT_AUDIT_JSON = Path("output/pdf/paper_i_hh_tableiii_prefix_replayability_audit_20260613.json")
DEFAULT_OUTPUT_JSON = Path("output/pdf/paper_i_hh_tableiii_prefix_resources_20260613.json")
READY_CLASSES = {"exact-prefix-replay-ready", "exact-prefix-compiled-ready"}
ENERGY_KEYS = ("energy_after", "energy_after_opt", "energy", "primary_energy_metric_after")
ERROR_KEYS = (
    "abs_delta_e_same_cutoff_after",
    "abs_delta_e_after",
    "benchmark_target_abs_delta_current",
    "delta_abs_current",
    "delta_E_abs_after",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _first_present(row: Mapping[str, Any], keys: Sequence[str]) -> tuple[str | None, Any]:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return key, value
    return None, None


def _history(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result = payload.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("adapt_history"), list):
        return [row for row in result["adapt_history"] if isinstance(row, Mapping)]
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping) and isinstance(adapt.get("history"), list):
        return [row for row in adapt["history"] if isinstance(row, Mapping)]
    if isinstance(payload.get("adapt_history"), list):
        return [row for row in payload["adapt_history"] if isinstance(row, Mapping)]
    if isinstance(payload.get("history"), list):
        return [row for row in payload["history"] if isinstance(row, Mapping)]
    return []


def _payload_result_row(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    rows = payload.get("rows")
    if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
        return rows[0]
    result = payload.get("result")
    if isinstance(result, Mapping):
        rows = result.get("rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], Mapping):
            return rows[0]
    return None


def _adapt_vqe_block(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    adapt = payload.get("adapt_vqe")
    if isinstance(adapt, Mapping):
        return adapt
    result = payload.get("result")
    if isinstance(result, Mapping) and isinstance(result.get("adapt_vqe"), Mapping):
        return result["adapt_vqe"]
    row = _payload_result_row(payload)
    if isinstance(row, Mapping) and isinstance(row.get("adapt_vqe"), Mapping):
        return row["adapt_vqe"]
    return None


def _payload_case_id(payload: Mapping[str, Any]) -> str | None:
    row = _payload_result_row(payload)
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
    for obj in (payload, row, result):
        if isinstance(obj, Mapping):
            value = obj.get("case_id") or obj.get("benchmark_id")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _payload_family(payload: Mapping[str, Any]) -> str | None:
    row = _payload_result_row(payload)
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
    for obj in (payload, row, result):
        if isinstance(obj, Mapping):
            value = obj.get("family")
            if isinstance(value, str) and value.strip():
                return value.strip()
    case_id = _payload_case_id(payload)
    if isinstance(case_id, str) and case_id.startswith("hh_"):
        return "hh"
    return None


def _selected_labels(row: Mapping[str, Any]) -> list[str]:
    value = row.get("selected_batch_labels")
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    value = row.get("selected_ops")
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    for key in ("selected_op", "selected_logical_op", "selected_label"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return [value]
    return []


def _pauli_map(row: Mapping[str, Any]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in ("top_candidates", "admitted_records", "selected_feature_rows"):
        records = row.get(key)
        if not isinstance(records, list):
            continue
        for rec in records:
            if not isinstance(rec, Mapping):
                continue
            label = rec.get("label") or rec.get("candidate_label") or rec.get("selected_label")
            paulis = rec.get("pauli_labels_exyz") or rec.get("pauli_labels") or rec.get("pauli_strings")
            if isinstance(label, str) and isinstance(paulis, list):
                clean = [str(pauli).strip().lower() for pauli in paulis if str(pauli).strip()]
                if clean:
                    out[label] = clean
    return out


def _resolve_runtime_seed_path(source_path: Path, payload: Mapping[str, Any]) -> Path | None:
    raw = payload.get("runtime_seed_json")
    if not isinstance(raw, str) or not raw:
        return None
    candidates = [Path(raw), source_path.parent / Path(raw).name]
    return next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)


def _state_vector_from_amplitudes(state_payload: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(state_payload, Mapping):
        return None
    nq = state_payload.get("nq_total")
    try:
        nq_int = int(nq)
    except Exception:
        return None
    amplitudes = state_payload.get("amplitudes_qn_to_q0")
    if not isinstance(amplitudes, Mapping):
        return None
    vec = np.zeros(1 << nq_int, dtype=complex)
    for bitstring, coeff in amplitudes.items():
        text = str(bitstring).strip()
        if not text or set(text) - {"0", "1"}:
            continue
        if len(text) != nq_int:
            continue
        if isinstance(coeff, Mapping):
            re = float(coeff.get("re", 0.0) or 0.0)
            im = float(coeff.get("im", 0.0) or 0.0)
        else:
            re = float(coeff)
            im = 0.0
        vec[int(text, 2)] = complex(re, im)
    if np.linalg.norm(vec) <= 0.0:
        return None
    return vec


def _reference_state_from_source(source_path: Path, payload: Mapping[str, Any]) -> tuple[np.ndarray | None, dict[str, Any]]:
    # Prefer an explicit ansatz input/reference-state envelope when present.
    state_sources: list[tuple[str, Mapping[str, Any]]] = [("top_level", payload)]
    row = _payload_result_row(payload)
    if isinstance(row, Mapping):
        state_sources.append(("rows[0]", row))
    adapt = _adapt_vqe_block(payload)
    if isinstance(adapt, Mapping):
        state_sources.append(("adapt_vqe", adapt))
    for source_label, obj in state_sources:
        state = _state_vector_from_amplitudes(obj.get("ansatz_input_state") if isinstance(obj.get("ansatz_input_state"), Mapping) else None)
        if state is not None:
            return state, {
                "status": "ok",
                "state_source": source_label,
                "state_key": "ansatz_input_state",
                "num_qubits": int(np.log2(state.size)),
            }

    seed_path = _resolve_runtime_seed_path(source_path, payload)
    if seed_path is not None:
        seed = _read_json(seed_path)
        if not isinstance(seed, Mapping):
            return None, {"status": "runtime_seed_not_object", "runtime_seed_json": str(seed_path)}
        for key in ("ansatz_input_state", "initial_state"):
            state = _state_vector_from_amplitudes(seed.get(key) if isinstance(seed.get(key), Mapping) else None)
            if state is not None:
                return state, {
                    "status": "ok",
                    "runtime_seed_json": str(seed_path),
                    "runtime_seed_sha256": _sha256(seed_path),
                    "state_key": key,
                    "num_qubits": int(np.log2(state.size)),
                }

    # Generic static comparator artifacts sometimes omit runtime_seed.json even
    # though the case id fully determines the reference state.  Derive it from
    # the same canonical case resolver used by the runner and record that fact.
    case_id = _payload_case_id(payload)
    if case_id:
        try:
            from pipelines.exact_bench.table_i_canonical_cases import (
                TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
                table_i_canonical_spec_by_case_id,
            )
            from pipelines.exact_bench.generic_static_adapt_variants import _resolve_context_from_spec

            family = _payload_family(payload)
            if not family:
                raise ValueError(f"cannot infer family for case_id={case_id!r}")
            profile = TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE if "three_model_sym" in str(case_id) else None
            spec = table_i_canonical_spec_by_case_id(str(family), str(case_id), profile=profile)
            context = _resolve_context_from_spec(spec)
            state = np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1)
            norm = float(np.linalg.norm(state))
            if norm > 0.0:
                state = state / norm
                return state, {
                    "status": "ok",
                    "state_source": "derived_from_table_i_canonical_case",
                    "case_id": str(case_id),
                    "num_qubits": int(np.log2(state.size)),
                }
        except Exception as exc:
            return None, {"status": "case_reference_state_derivation_failed", "case_id": str(case_id), "error": str(exc)}

    return None, {"status": "missing_reference_state", "runtime_seed_json": payload.get("runtime_seed_json")}

def _num_qubits_from_groups(groups: Sequence[Sequence[str]], reference_state: np.ndarray | None) -> int:
    for group in groups:
        for label in group:
            if str(label):
                return len(str(label))
    if reference_state is not None:
        return int(np.log2(reference_state.size))
    raise ValueError("cannot infer num_qubits from empty prefix")


def _clean_pauli_group(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, Mapping):
            label = item.get("pauli_exyz") or item.get("pauli") or item.get("label")
        else:
            label = item
        text = str(label).strip().lower()
        if text:
            out.append(text)
    return out


def _final_selected_step_groups(payload: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> list[list[list[str]]] | None:
    row = _payload_result_row(payload)
    sources = [row, _adapt_vqe_block(payload), payload]
    for obj in sources:
        if not isinstance(obj, Mapping):
            continue
        labels = obj.get("selected_operators") or obj.get("operators")
        paulis = obj.get("selected_operator_pauli_labels_exyz")
        batches = obj.get("selected_operator_batches")
        if not (isinstance(labels, list) and isinstance(paulis, list) and len(labels) == len(paulis)):
            continue
        if isinstance(batches, list) and batches:
            out: list[list[list[str]]] = []
            cursor = 0
            for batch in batches:
                if not isinstance(batch, list):
                    return None
                step: list[list[str]] = []
                for _label in batch:
                    if cursor >= len(paulis):
                        return None
                    group = _clean_pauli_group(paulis[cursor])
                    if not group:
                        return None
                    step.append(group)
                    cursor += 1
                out.append(step)
            return out if len(out) >= len(history) else None
        if len(paulis) >= len(history):
            out = []
            for idx in range(len(history)):
                group = _clean_pauli_group(paulis[idx])
                if not group:
                    return None
                out.append([group])
            return out
    return None


def _parameterization_step_groups(payload: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> list[list[list[str]]] | None:
    adapt = _adapt_vqe_block(payload)
    if not isinstance(adapt, Mapping):
        return None
    parameterization = adapt.get("parameterization")
    if not isinstance(parameterization, Mapping):
        return None
    blocks = parameterization.get("blocks")
    if not isinstance(blocks, list):
        return None
    by_label: dict[str, list[list[str]]] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        label = block.get("candidate_label")
        terms = block.get("runtime_terms_exyz")
        group = _clean_pauli_group(terms)
        if isinstance(label, str) and group:
            by_label.setdefault(label, []).append(group)
    if not by_label:
        return None
    used: dict[str, int] = {}
    out: list[list[list[str]]] = []
    for row in history:
        step: list[list[str]] = []
        for label in _selected_labels(row):
            groups = by_label.get(label)
            if not groups:
                return None
            idx = used.get(label, 0)
            group = groups[idx] if idx < len(groups) else groups[-1]
            used[label] = idx + 1
            step.append(group)
        if not step:
            return None
        out.append(step)
    return out


def _fallback_step_groups(payload: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> tuple[list[list[list[str]]] | None, str | None]:
    for source, builder in (
        ("final_selected_operator_pauli_labels_exyz", _final_selected_step_groups),
        ("adapt_vqe_parameterization_runtime_terms_exyz", _parameterization_step_groups),
    ):
        groups = builder(payload, history)
        if groups is not None:
            return groups, source
    return None, None


def _selected_positions(row: Mapping[str, Any], count: int) -> list[int | None]:
    raw = row.get("selected_positions")
    if isinstance(raw, list) and len(raw) >= count:
        out: list[int | None] = []
        for value in raw[:count]:
            try:
                out.append(int(value))
            except Exception:
                out.append(None)
        return out
    value = row.get("selected_position")
    if value is not None and count == 1:
        try:
            return [int(value)]
        except Exception:
            return [None]
    return [None for _ in range(count)]


def _compile_prefix_rows(
    *,
    regime: str,
    method: str,
    source_path: Path,
    source_sha256: str | None,
    visible_cells: Mapping[str, Any],
    max_prefixes: int | None = None,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = _read_json(source_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"source is not a JSON object: {source_path}")
    history = _history(payload)
    reference_state, reference_meta = _reference_state_from_source(source_path, payload)
    if reference_state is None:
        raise ValueError(f"reference state unavailable for {source_path}: {reference_meta}")

    fallback_groups, fallback_source = _fallback_step_groups(payload, history)
    rows: list[dict[str, Any]] = []
    pauli_groups: list[list[str]] = []
    if progress:
        print(f"export_source regime={regime} method={method} history_len={len(history)} path={source_path}", flush=True)
    for index, hist_row in enumerate(history, start=1):
        if max_prefixes is not None and index > int(max_prefixes):
            break
        labels = _selected_labels(hist_row)
        pauli_lookup = _pauli_map(hist_row)
        step_groups: list[list[str]] = []
        if labels and all(pauli_lookup.get(label) for label in labels):
            step_groups = [list(pauli_lookup[label]) for label in labels]
            step_pauli_source = "history_selected_candidate_pauli_labels"
        elif fallback_groups is not None and index <= len(fallback_groups):
            step_groups = [list(group) for group in fallback_groups[index - 1]]
            step_pauli_source = str(fallback_source)
        else:
            missing = [label for label in labels if label not in pauli_lookup]
            raise ValueError(f"missing Pauli group for {regime}/{method} prefix {index}: {missing or labels}")
        positions = _selected_positions(hist_row, len(step_groups))
        for group, position in zip(step_groups, positions):
            if position is None or position < 0 or position > len(pauli_groups):
                pauli_groups.append(group)
            else:
                pauli_groups.insert(int(position), group)
        if not pauli_groups:
            continue
        num_qubits = _num_qubits_from_groups(pauli_groups, reference_state)
        if progress:
            print(
                f"compile_prefix regime={regime} method={method} prefix={index} logical_ops={len(pauli_groups)}",
                flush=True,
            )
        try:
            compiled = compile_table_i_pauli_label_groups(
                pauli_label_groups=tuple(tuple(group) for group in pauli_groups),
                num_qubits=num_qubits,
                reference_state=reference_state,
                source_kind="paper_i_hh_tableiii_exact_prefix_resource",
            )
            compile_status = "ok"
            compile_error = None
        except TableICompileUnavailable as exc:
            compiled = {}
            compile_status = exc.status
            compile_error = exc.reason
        energy_key, energy = _first_present(hist_row, ENERGY_KEYS)
        error_key, error = _first_present(hist_row, ERROR_KEYS)
        rows.append(
            {
                "schema": "paper_i_hh_tableiii_exact_prefix_resource_row_v1",
                "regime": regime,
                "method": method,
                "source_json": str(source_path),
                "source_sha256": source_sha256,
                "prefix_k": int(index),
                "prefix_k_semantics": "adapt_history_row_index_1based",
                "adapt_iteration": hist_row.get("iteration", index),
                "logical_operator_prefix_len": int(len(pauli_groups)),
                "selected_batch_size": int(len(labels)),
                "selected_labels": labels,
                "selected_pauli_source": step_pauli_source,
                "prefix_order_semantics": "history_selected_positions_when_serialized_else_append_order",
                "energy": None if energy is None else float(energy),
                "energy_field": energy_key,
                "abs_delta_e": None if error is None else float(error),
                "abs_delta_e_field": error_key,
                "compile_status": compile_status,
                "compile_error": compile_error,
                "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
                "N1q": compiled.get("compiled_count_1q_total"),
                "N2q": compiled.get("compiled_count_2q_total"),
                "D_circ": compiled.get("compiled_depth_total"),
                "D2q": compiled.get("compiled_depth_2q_total"),
                "compiled_count_1q_semantics": compiled.get("compiled_count_1q_semantics"),
                "compiled_op_counts": compiled.get("compiled_op_counts"),
                "num_qubits": compiled.get("num_qubits", num_qubits),
                "runtime_rotation_count": compiled.get("runtime_rotation_count"),
                "reference_state_status": reference_meta.get("status"),
                "visible_cells_terminal": dict(visible_cells),
            }
        )
    return rows, reference_meta


def export_prefix_resources(
    *,
    source_map_path: Path = DEFAULT_SOURCE_MAP,
    audit_json_path: Path = DEFAULT_AUDIT_JSON,
    output_json_path: Path = DEFAULT_OUTPUT_JSON,
    rebuild_audit_if_missing: bool = True,
    include_regimes: set[str] | None = None,
    include_methods: set[str] | None = None,
    max_prefixes_per_source: int | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    if audit_json_path.exists():
        audit = _read_json(audit_json_path)
    elif rebuild_audit_if_missing:
        audit = build_audit(source_map_path)
    else:
        raise FileNotFoundError(audit_json_path)
    if not isinstance(audit, Mapping):
        raise ValueError(f"audit is not a JSON object: {audit_json_path}")

    rows: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    reference_meta_by_source: dict[str, Any] = {}
    for audit_row in audit.get("rows", []):
        if not isinstance(audit_row, Mapping):
            continue
        classification = str(audit_row.get("classification") or "")
        regime = str(audit_row.get("regime") or "")
        method = str(audit_row.get("method") or "")
        if include_regimes is not None and regime not in include_regimes:
            continue
        if include_methods is not None and method not in include_methods:
            continue
        primary = audit_row.get("primary_source") if isinstance(audit_row.get("primary_source"), Mapping) else {}
        path_text = str(primary.get("path") or "")
        if classification not in READY_CLASSES:
            blocked.append(
                {
                    "regime": regime,
                    "method": method,
                    "classification": classification,
                    "blockers": list(audit_row.get("blockers") or []),
                    "primary_source": dict(primary),
                }
            )
            continue
        source_path = Path(path_text)
        if not source_path.exists():
            blocked.append({"regime": regime, "method": method, "classification": "source-missing-at-export", "primary_source": dict(primary)})
            continue
        visible_cells = audit_row.get("visible_cells") if isinstance(audit_row.get("visible_cells"), Mapping) else {}
        compiled_rows, reference_meta = _compile_prefix_rows(
            regime=regime,
            method=method,
            source_path=source_path,
            source_sha256=_sha256(source_path),
            visible_cells=visible_cells,
            max_prefixes=max_prefixes_per_source,
            progress=progress,
        )
        rows.extend(compiled_rows)
        reference_meta_by_source[str(source_path)] = reference_meta

    manifest = {
        "schema": "paper_i_hh_tableiii_exact_prefix_resources_v1",
        "source_map": str(source_map_path),
        "source_map_sha256": _sha256(source_map_path),
        "audit_json": str(audit_json_path) if audit_json_path.exists() else None,
        "audit_schema": audit.get("schema"),
        "compile_convention": TABLE_I_QISKIT_COMPILE_CONVENTION,
        "ready_classes": sorted(READY_CLASSES),
        "row_count": len(rows),
        "blocked_row_count": len(blocked),
        "filters": {
            "include_regimes": sorted(include_regimes) if include_regimes is not None else None,
            "include_methods": sorted(include_methods) if include_methods is not None else None,
            "max_prefixes_per_source": max_prefixes_per_source,
        },
        "compiled_ok_count": sum(1 for row in rows if row.get("compile_status") == "ok"),
        "reference_meta_by_source": reference_meta_by_source,
        "blocked_rows": blocked,
        "rows": rows,
    }
    _write_json(output_json_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--no-rebuild-audit", action="store_true")
    parser.add_argument("--regime", action="append", help="Limit export to one regime; may be repeated.")
    parser.add_argument("--method", action="append", help="Limit export to one method; may be repeated.")
    parser.add_argument("--max-prefixes-per-source", type=int, default=None, help="Smoke/debug limit for compiled prefixes per ready source.")
    parser.add_argument("--progress", action="store_true", help="Print per-source/per-prefix progress.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_prefix_resources(
        source_map_path=args.source_map,
        audit_json_path=args.audit_json,
        output_json_path=args.output_json,
        rebuild_audit_if_missing=not args.no_rebuild_audit,
        include_regimes=set(args.regime) if args.regime else None,
        include_methods=set(args.method) if args.method else None,
        max_prefixes_per_source=args.max_prefixes_per_source,
        progress=bool(args.progress),
    )
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "row_count": manifest["row_count"],
                "compiled_ok_count": manifest["compiled_ok_count"],
                "blocked_row_count": manifest["blocked_row_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
