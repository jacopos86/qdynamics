#!/usr/bin/env python3
"""Audit Paper-I HH Table-III sources for exact per-prefix circuit-cost replay.

This audit is intentionally fail-closed.  It does not compile circuits and it
does not infer exact backend-transpiled prefix costs from controller/proxy cost
fields.  Its job is to classify the current visible convergence sources before a
separate exporter tries to build Qiskit circuits for every prefix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_SOURCE_MAP = Path("MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json")
DEFAULT_OUTPUT_JSON = Path("output/pdf/paper_i_hh_tableiii_prefix_replayability_audit_20260613.json")
REGIME_ORDER = ("weak_weak", "strong_weak", "weak_strong", "strong_strong")
METHOD_ORDER = ("Append-ADAPT", "TETRIS-ADAPT", "SNAKE", "Geo-ADAPT")

HISTORY_PATHS: tuple[tuple[str, ...], ...] = (
    ("result", "adapt_history"),
    ("adapt_vqe", "history"),
    ("history",),
    ("adapt_history",),
)
ENERGY_KEYS = ("energy_after", "energy_after_opt", "energy", "primary_energy_metric_after")
ERROR_KEYS = (
    "abs_delta_e_same_cutoff_after",
    "abs_delta_e_after",
    "benchmark_target_abs_delta_current",
    "delta_abs_current",
    "delta_E_abs_after",
)
SELECTED_LABEL_KEYS = (
    "selected_batch_labels",
    "selected_ops",
    "selected_op",
    "selected_logical_op",
    "selected_label",
)
POSITION_KEYS = ("selected_insertion_position", "selected_position", "selected_positions", "position_id", "append_position")
EXACT_COMPILED_MARKERS = (
    "compiled_count_1q",
    "compiled_count_2q",
    "compiled_count_2q_total",
    "compiled_depth",
    "compiled_depth_2q",
    "compiled_position_cost_backend",
    "delta_compiled_count_2q",
    "trial_compiled_count_2q",
)


def _json_default(value: Any) -> Any:
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


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _get_path(payload: Mapping[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = payload
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _find_history(payload: Mapping[str, Any]) -> tuple[str | None, list[Mapping[str, Any]]]:
    for keys in HISTORY_PATHS:
        value = _get_path(payload, keys)
        if isinstance(value, list):
            rows = [row for row in value if isinstance(row, Mapping)]
            return ".".join(keys), rows
    return None, []


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


def _has_state_envelope_or_derivable_case(payload: Mapping[str, Any]) -> bool:
    row = _payload_result_row(payload)
    adapt = _adapt_vqe_block(payload)
    for obj in (payload, row, adapt):
        if isinstance(obj, Mapping) and isinstance(obj.get("ansatz_input_state"), Mapping):
            return True
    return _payload_case_id(payload) is not None


def _has_final_selected_pauli_source(payload: Mapping[str, Any], history_len: int) -> bool:
    row = _payload_result_row(payload)
    adapt = _adapt_vqe_block(payload)
    for obj in (row, adapt, payload):
        if not isinstance(obj, Mapping):
            continue
        labels = obj.get("selected_operators") or obj.get("operators")
        paulis = obj.get("selected_operator_pauli_labels_exyz")
        batches = obj.get("selected_operator_batches")
        if not (isinstance(labels, list) and isinstance(paulis, list) and len(labels) == len(paulis)):
            continue
        if isinstance(batches, list) and len(batches) >= history_len:
            return True
        if len(paulis) >= history_len:
            return True
    return False


def _has_parameterization_pauli_source(payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bool:
    adapt = _adapt_vqe_block(payload)
    if not isinstance(adapt, Mapping):
        return False
    parameterization = adapt.get("parameterization")
    if not isinstance(parameterization, Mapping):
        return False
    blocks = parameterization.get("blocks")
    if not isinstance(blocks, list):
        return False
    labels_with_terms = set()
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        label = block.get("candidate_label")
        terms = block.get("runtime_terms_exyz")
        if isinstance(label, str) and isinstance(terms, list) and terms:
            labels_with_terms.add(label)
    if not labels_with_terms:
        return False
    for row in rows:
        labels = _selected_labels(row)
        if not labels or any(label not in labels_with_terms for label in labels):
            return False
    return True


def _flatten(value: Any) -> Iterable[Any]:
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def _selected_labels(row: Mapping[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in SELECTED_LABEL_KEYS:
        value = row.get(key)
        if value is None:
            continue
        for item in _flatten(value):
            if isinstance(item, str) and item and item not in labels:
                labels.append(item)
    records = []
    for key in ("admitted_records", "selected_feature_rows"):
        value = row.get(key)
        if isinstance(value, list):
            records.extend(v for v in value if isinstance(v, Mapping))
    for rec in records:
        for key in ("candidate_label", "selected_label", "label"):
            value = rec.get(key)
            if isinstance(value, str) and value and value not in labels:
                labels.append(value)
    return labels


def _candidate_pauli_map(row: Mapping[str, Any]) -> dict[str, list[str]]:
    records: list[Mapping[str, Any]] = []
    for key in ("top_candidates", "admitted_records", "selected_feature_rows", "shortlisted_records", "scored_surface_records"):
        value = row.get(key)
        if isinstance(value, list):
            records.extend(v for v in value if isinstance(v, Mapping))
    out: dict[str, list[str]] = {}
    for rec in records:
        label = rec.get("label") or rec.get("candidate_label") or rec.get("selected_label")
        paulis = rec.get("pauli_labels_exyz") or rec.get("pauli_labels") or rec.get("pauli_strings")
        if isinstance(label, str) and isinstance(paulis, list) and all(isinstance(p, str) for p in paulis):
            out[label] = [str(p) for p in paulis]
    return out


def _has_any_key_fragment(value: Any, fragments: Sequence[str], depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_lower = str(key).lower()
            if any(fragment in key_lower for fragment in fragments):
                return True
            if _has_any_key_fragment(nested, fragments, depth + 1):
                return True
    elif isinstance(value, list):
        for nested in value[:20]:
            if _has_any_key_fragment(nested, fragments, depth + 1):
                return True
    return False


def _has_nonempty_key_fragment(value: Any, fragments: Sequence[str], depth: int = 0) -> bool:
    if depth > 5:
        return False
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_lower = str(key).lower()
            if any(fragment in key_lower for fragment in fragments):
                if nested is None:
                    continue
                if isinstance(nested, (Mapping, list, tuple, str)) and len(nested) == 0:
                    continue
                return True
            if _has_nonempty_key_fragment(nested, fragments, depth + 1):
                return True
    elif isinstance(value, list):
        for nested in value[:20]:
            if _has_nonempty_key_fragment(nested, fragments, depth + 1):
                return True
    return False


def _row_has_exact_compiled_fields(row: Mapping[str, Any]) -> bool:
    return _has_nonempty_key_fragment(row, EXACT_COMPILED_MARKERS)


def _row_has_position(row: Mapping[str, Any]) -> bool:
    if any(key in row and row.get(key) is not None for key in POSITION_KEYS):
        return True
    for key in ("admitted_records", "selected_feature_rows"):
        value = row.get(key)
        if isinstance(value, list):
            for rec in value:
                if isinstance(rec, Mapping) and any(k in rec and rec.get(k) is not None for k in POSITION_KEYS):
                    return True
    return False


def _history_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_label_rows = 0
    selected_label_total = 0
    pauli_rows_all_selected = 0
    energy_rows = 0
    error_rows = 0
    position_rows = 0
    exact_compiled_rows = 0
    missing_pauli_examples: list[dict[str, Any]] = []
    batch_size_histogram: Counter[str] = Counter()

    for idx, row in enumerate(rows, start=1):
        labels = _selected_labels(row)
        selected_label_total += len(labels)
        if labels:
            selected_label_rows += 1
        pauli_map = _candidate_pauli_map(row)
        missing = [label for label in labels if label not in pauli_map]
        if labels and not missing:
            pauli_rows_all_selected += 1
        elif missing and len(missing_pauli_examples) < 5:
            missing_pauli_examples.append({"row_index_1based": idx, "missing_labels": missing[:5]})
        if any(row.get(key) is not None for key in ENERGY_KEYS):
            energy_rows += 1
        if any(row.get(key) is not None for key in ERROR_KEYS):
            error_rows += 1
        if _row_has_position(row):
            position_rows += 1
        if _row_has_exact_compiled_fields(row):
            exact_compiled_rows += 1
        batch_size = row.get("batch_size")
        if batch_size is not None:
            batch_size_histogram[str(batch_size)] += 1

    history_len = len(rows)
    return {
        "history_len": history_len,
        "selected_label_rows": selected_label_rows,
        "selected_label_total": selected_label_total,
        "pauli_rows_all_selected": pauli_rows_all_selected,
        "energy_rows": energy_rows,
        "error_rows": error_rows,
        "position_rows": position_rows,
        "exact_compiled_rows": exact_compiled_rows,
        "batch_size_histogram": dict(sorted(batch_size_histogram.items())),
        "missing_pauli_examples": missing_pauli_examples,
        "all_rows_have_selected_labels": bool(history_len and selected_label_rows == history_len),
        "all_rows_have_selected_paulis": bool(history_len and pauli_rows_all_selected == history_len),
        "all_rows_have_energy_and_error": bool(history_len and energy_rows == history_len and error_rows == history_len),
        "all_rows_have_positions": bool(history_len and position_rows == history_len),
        "all_rows_have_exact_compiled_fields": bool(history_len and exact_compiled_rows == history_len),
    }


def _truthy_stdout(*values: Any) -> bool:
    text = " ".join(str(v).lower() for v in values if v is not None)
    return "stdout" in text or "ai_log" in text or "no_replayable_current_json" in text


def _add_candidate(candidates: list[dict[str, Any]], *, key: str, path: Any, expected_sha256: Any = None, primary: bool = False) -> None:
    if not isinstance(path, str) or not path:
        return
    if any(candidate["path"] == path and candidate["key"] == key for candidate in candidates):
        return
    candidates.append({"key": key, "path": path, "expected_sha256": expected_sha256, "primary_visible_source": primary})


def _method_candidates(method: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    primary_key = "history_source_json" if isinstance(method.get("history_source_json"), str) else "source_json"
    for key in (
        "history_source_json",
        "source_json",
        "strict_replay_json",
        "compiled_cost_json",
        "base_source_json_before_continuation",
        "previous_source_json",
        "previous_source_json_before_stdout_continuation_20260601",
        "previous_json_backed_source_json",
        "current_best_json",
        "summary_json",
        "trial_summary_json",
    ):
        sha_key = key.replace("_json", "_sha256")
        _add_candidate(candidates, key=key, path=method.get(key), expected_sha256=method.get(sha_key), primary=(key == primary_key))
    diag = method.get("shot_proxy_diagnostic")
    if isinstance(diag, Mapping):
        _add_candidate(
            candidates,
            key="shot_proxy_diagnostic.source_json",
            path=diag.get("source_json"),
            expected_sha256=diag.get("source_sha256"),
            primary=False,
        )
    return candidates


def _resolve_runtime_seed(source_path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    raw = payload.get("runtime_seed_json")
    if not isinstance(raw, str) or not raw:
        return {"runtime_seed_json": raw, "exists": False, "resolution": "absent"}
    raw_path = Path(raw)
    candidates = [raw_path]
    candidates.append(source_path.parent / raw_path.name)
    resolved = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)
    if resolved is None:
        return {"runtime_seed_json": raw, "exists": False, "resolution": "missing", "candidate_paths": [str(c) for c in candidates]}
    seed_payload: Any = None
    try:
        seed_payload = _read_json(resolved)
    except Exception:
        seed_payload = None
    has_ansatz = isinstance(seed_payload, Mapping) and isinstance(seed_payload.get("ansatz_input_state"), Mapping)
    has_initial = isinstance(seed_payload, Mapping) and isinstance(seed_payload.get("initial_state"), Mapping)
    return {
        "runtime_seed_json": raw,
        "exists": True,
        "resolved_path": str(resolved),
        "sha256": _sha256(resolved),
        "has_ansatz_input_state": has_ansatz,
        "has_initial_state": has_initial,
    }


def _candidate_audit(candidate: Mapping[str, Any], method_meta: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(candidate["path"]))
    base: dict[str, Any] = {
        "key": candidate.get("key"),
        "path": str(candidate.get("path")),
        "primary_visible_source": bool(candidate.get("primary_visible_source")),
        "expected_sha256": candidate.get("expected_sha256"),
        "exists_locally": path.exists() and path.is_file(),
    }
    if not base["exists_locally"]:
        base.update(classification="source-missing", blockers=["source_not_available_locally"])
        return base
    actual_sha = _sha256(path)
    base["actual_sha256"] = actual_sha
    if candidate.get("expected_sha256") and candidate.get("expected_sha256") != actual_sha:
        base["sha256_status"] = "mismatch"
    elif candidate.get("expected_sha256"):
        base["sha256_status"] = "match"
    else:
        base["sha256_status"] = "not_declared"

    try:
        payload = _read_json(path)
    except Exception as exc:
        base.update(classification="source-unreadable", blockers=["source_unreadable"], exception_type=type(exc).__name__, exception=str(exc))
        return base
    if not isinstance(payload, Mapping):
        base.update(classification="source-not-object", blockers=["source_not_json_object"])
        return base

    primary_candidate = bool(candidate.get("primary_visible_source"))
    source_kind = payload.get("source_kind") or payload.get("evidence_status")
    if source_kind is None and primary_candidate:
        source_kind = method_meta.get("source_kind")
    stdout_markers = [source_kind, path]
    if primary_candidate:
        stdout_markers.extend([method_meta.get("evidence_status"), method_meta.get("strict_replay_validation")])
    stdout_only = _truthy_stdout(*stdout_markers)
    history_path, rows = _find_history(payload)
    history = _history_audit(rows)
    runtime_seed = _resolve_runtime_seed(path, payload)
    has_strict_resume_state = isinstance(payload.get("ansatz_input_state"), Mapping) or isinstance(payload.get("initial_state"), Mapping)
    has_runtime_seed_state = bool(runtime_seed.get("has_ansatz_input_state") and runtime_seed.get("has_initial_state"))
    has_replay_state = has_strict_resume_state or has_runtime_seed_state

    blockers: list[str] = []
    classification = "exact-prefix-replay-ready"
    if base.get("sha256_status") == "mismatch":
        blockers.append("source_sha256_mismatch")
    if stdout_only:
        classification = "stdout-only-blocked"
        blockers.append("visible_source_is_stdout_or_ai_log_derived")
    elif not rows:
        if _has_any_key_fragment(payload, EXACT_COMPILED_MARKERS):
            classification = "terminal-only"
            blockers.append("no_prefix_history_for_per_prefix_export")
        else:
            classification = "needs-richer-history"
            blockers.append("no_adapt_history_found")
    else:
        if not history["all_rows_have_selected_labels"]:
            blockers.append("missing_selected_labels_in_history")
        if not history["all_rows_have_energy_and_error"]:
            blockers.append("missing_energy_or_same_cutoff_error_history")
        has_recoverable_paulis = bool(
            history["all_rows_have_selected_paulis"]
            or _has_final_selected_pauli_source(payload, history["history_len"])
            or _has_parameterization_pauli_source(payload, rows)
        )
        has_recoverable_state = bool(has_replay_state or _has_state_envelope_or_derivable_case(payload))
        if not has_recoverable_paulis:
            blockers.append("missing_selected_pauli_expansions")
        if not has_recoverable_state:
            blockers.append("missing_runtime_seed_or_strict_resume_state")
        if blockers:
            classification = "needs-richer-history"
        elif history["all_rows_have_exact_compiled_fields"]:
            classification = "exact-prefix-compiled-ready"

    base.update(
        classification=classification,
        blockers=blockers,
        source_kind=source_kind,
        history_path=history_path,
        history=history,
        runtime_seed=runtime_seed,
        has_strict_resume_state=has_strict_resume_state,
        recoverable_pauli_source={
            "history_selected_paulis": bool(history["all_rows_have_selected_paulis"]),
            "final_selected_operator_pauli_labels_exyz": bool(_has_final_selected_pauli_source(payload, history["history_len"])),
            "adapt_vqe_parameterization_runtime_terms_exyz": bool(_has_parameterization_pauli_source(payload, rows)),
        },
        recoverable_reference_state=bool(has_replay_state or _has_state_envelope_or_derivable_case(payload)),
        top_level_keys=sorted(str(k) for k in payload.keys())[:80],
    )
    return base


def _primary_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    for candidate in candidates:
        if candidate.get("primary_visible_source"):
            return candidate
    return candidates[0] if candidates else None


def _classification_rank(classification: str) -> int:
    order = {
        "exact-prefix-compiled-ready": 0,
        "exact-prefix-replay-ready": 1,
        "needs-richer-history": 2,
        "terminal-only": 3,
        "stdout-only-blocked": 4,
        "source-missing": 5,
        "source-unreadable": 6,
        "source-not-object": 7,
    }
    return order.get(classification, 99)


def _audit_method(regime: str, method_name: str, method: Mapping[str, Any]) -> dict[str, Any]:
    method_meta = dict(method)
    method_meta["method"] = method_name
    candidates = _method_candidates(method)
    candidate_audits = [_candidate_audit(candidate, method_meta) for candidate in candidates]
    primary = next((audit for audit in candidate_audits if audit.get("primary_visible_source")), None)
    best = min(candidate_audits, key=lambda row: _classification_rank(str(row.get("classification")))) if candidate_audits else None
    classification = str(primary.get("classification")) if primary else "source-missing"
    blockers = list(primary.get("blockers", [])) if primary else ["no_primary_source_candidate"]
    return {
        "regime": regime,
        "method": method_name,
        "classification": classification,
        "blockers": blockers,
        "primary_source": {
            "key": primary.get("key") if primary else None,
            "path": primary.get("path") if primary else None,
            "classification": primary.get("classification") if primary else None,
            "history_len": (primary.get("history") or {}).get("history_len") if primary else None,
        },
        "best_available_candidate": {
            "key": best.get("key") if best else None,
            "path": best.get("path") if best else None,
            "classification": best.get("classification") if best else None,
            "history_len": (best.get("history") or {}).get("history_len") if best else None,
        },
        "visible_cells": method.get("visible_cells", {}),
        "last_history_len_from_source_map": method.get("last_history_len"),
        "candidate_count": len(candidate_audits),
        "candidates": candidate_audits,
    }


def build_audit(source_map_path: Path = DEFAULT_SOURCE_MAP) -> dict[str, Any]:
    source_map = _read_json(source_map_path)
    if not isinstance(source_map, Mapping):
        raise ValueError(f"source map is not a JSON object: {source_map_path}")
    regimes = _as_mapping(source_map.get("regimes")) or {}
    rows: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        regime_block = _as_mapping(regimes.get(regime)) or {}
        methods = _as_mapping(regime_block.get("methods")) or {}
        for method_name in METHOD_ORDER:
            method = _as_mapping(methods.get(method_name))
            if method is None:
                rows.append({"regime": regime, "method": method_name, "classification": "source-missing", "blockers": ["method_block_missing"]})
                continue
            rows.append(_audit_method(regime, method_name, method))
    status_counts = Counter(str(row.get("classification")) for row in rows)
    return {
        "schema": "paper_i_hh_tableiii_prefix_replayability_audit_v1",
        "source_map": str(source_map_path),
        "source_map_sha256_at_audit_time": _sha256(source_map_path),
        "policy": {
            "exact_prefix_cost_requirement": "per-prefix Qiskit-compiled circuit resources require replayable prefix structure; controller/proxy costs are not accepted as exact compiled costs",
            "ready_classes": ["exact-prefix-replay-ready", "exact-prefix-compiled-ready"],
            "blocked_classes": ["needs-richer-history", "stdout-only-blocked", "terminal-only", "source-missing"],
            "prefix_k_semantics": "ADAPT iteration row; TETRIS rows may add selected_batch_labels>1 and therefore logical_operator_prefix_len must be tracked separately by the exporter",
        },
        "row_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    args = parser.parse_args(argv)
    audit = build_audit(args.source_map)
    _write_json(args.output_json, audit)
    print(json.dumps({"output_json": str(args.output_json), "status_counts": audit["status_counts"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
