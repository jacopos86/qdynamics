#!/usr/bin/env python3
"""Build Paper-I three-model support maps for table costs and fidelities.

The fixed-accuracy manuscript tables are intentionally small, but their support
data should not be hand-patched.  This script merges comparator support maps
with completed Route-A/SNAKE Optuna summaries and computes SNAKE fidelity
diagnostics from immutable result JSONs.  Exact references are used only as a
post-hoc reporting side channel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipelines.exact_bench.generic_static_metric_enrichment import (
    _normalize_state,
    _prepare_selected_state,
    _reference_cutoff_metrics,
    _resolve_context,
    build_full_meta_candidate_pool,
)
from pipelines.exact_bench.snake_table_i_measurement_work import snake_controller_shot_proxy_from_payload
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id

SCHEMA = "paper_i_three_model_support_maps_v1"
SNAKE_ALGORITHM_ID = "static_family_native_adapt_phase3"
SNAKE_METHOD = "SNAKE"


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _family_from_case_id(case_id: str) -> str:
    text = str(case_id)
    if text.startswith("spin_boson_"):
        return "spin_boson"
    if text.startswith("hubbard_"):
        return "hubbard"
    if text.startswith("hh_"):
        return "hh"
    raise ValueError(f"cannot infer Paper-I three-model family from benchmark id {case_id!r}")


def _snake_summary_paths(roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.name.startswith("routeA_paper_i_three_model_"):
            paths.update(root.glob("run/*/summary.json"))
        paths.update(root.glob("routeA_paper_i_three_model_*_selected_logical_v*/run/*/summary.json"))
    return sorted(paths)


def _best_result_path(summary_path: Path, summary: Mapping[str, Any]) -> Path | None:
    raw_trial = summary.get("best_trial_number")
    try:
        trial_number = int(raw_trial)
    except (TypeError, ValueError):
        return None
    benchmark_id = summary_path.parent.name
    path = summary_path.parent / f"trial_{trial_number:04d}" / benchmark_id / "json" / "result.json"
    return path if path.exists() else None


def _selected_state_from_snake_result(*, family: str, case_id: str, suite_profile: str, result: Mapping[str, Any]) -> np.ndarray:
    spec = table_i_canonical_spec_by_case_id(family, case_id, profile=suite_profile)
    context = _resolve_context(spec)
    pool = build_full_meta_candidate_pool(context)
    by_label = {str(getattr(candidate, "label")): candidate for candidate in pool}
    row = result.get("adapt_vqe")
    if not isinstance(row, Mapping):
        raise ValueError("missing adapt_vqe payload")
    raw_ops = row.get("operators")
    if not isinstance(raw_ops, list) or not raw_ops:
        raise ValueError("missing SNAKE selected operators")
    selected = []
    for raw_label in raw_ops:
        label = str(raw_label)
        candidates = [label]
        if "::" in label:
            candidates.append(label.split("::", 1)[1])
        match = next((candidate for candidate in candidates if candidate in by_label), None)
        if match is None:
            raise ValueError(f"SNAKE selected operator {label!r} not found in rebuilt full-meta pool")
        selected.append(by_label[match])
    theta_raw = row.get("logical_optimal_point")
    if not isinstance(theta_raw, list) or len(theta_raw) != len(selected):
        theta_raw = row.get("optimal_point")
    if not isinstance(theta_raw, list) or len(theta_raw) != len(selected):
        raise ValueError("SNAKE logical parameter count does not match selected operators")
    theta = np.asarray(theta_raw, dtype=float).reshape(-1)
    psi_ref = _normalize_state(np.asarray(context.reference_state.build_state(), dtype=complex).reshape(-1))
    return _prepare_selected_state(selected=selected, theta=theta, psi_ref=psi_ref, pauli_action_cache={})


def _snake_fidelity_fields(
    *,
    family: str,
    case_id: str,
    suite_profile: str,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    row = result.get("adapt_vqe")
    if not isinstance(row, Mapping):
        return {
            "fidelity": None,
            "infidelity": None,
            "one_minus_fidelity": None,
            "infidelity_source_key": None,
            "infidelity_status": "missing_adapt_vqe_payload",
            "infidelity_statuses": {},
        }

    same_fidelity = _num(row.get("exact_state_fidelity"))
    same_infidelity = None if same_fidelity is None else float(max(0.0, min(1.0, 1.0 - same_fidelity)))
    statuses: dict[str, str] = {}
    if same_infidelity is not None:
        statuses["infidelity_same"] = "ok"

    try:
        spec = table_i_canonical_spec_by_case_id(family, case_id, profile=suite_profile)
        context = _resolve_context(spec)
        psi = _selected_state_from_snake_result(
            family=family,
            case_id=case_id,
            suite_profile=suite_profile,
            result=result,
        )
        _ref_metric, ref_updates, ref_statuses = _reference_cutoff_metrics(spec, context, psi)
        statuses.update(ref_statuses)
        ref_infidelity = _num(ref_updates.get("infidelity_4"))
        if str(ref_statuses.get("infidelity_4") or "") == "ok" and ref_infidelity is not None:
            infidelity = float(max(0.0, min(1.0, ref_infidelity)))
            return {
                "fidelity": float(max(0.0, min(1.0, 1.0 - infidelity))),
                "infidelity": infidelity,
                "one_minus_fidelity": infidelity,
                "infidelity_reference": infidelity,
                "infidelity_source_key": "infidelity_4",
                "infidelity_status": "ok",
                "infidelity_statuses": statuses,
                "fidelity_replay_source": "snake_selected_logical_reference_cutoff_replay",
            }
    except Exception as exc:
        statuses["infidelity_reference_replay"] = f"failed:{type(exc).__name__}"

    if same_infidelity is None:
        return {
            "fidelity": None,
            "infidelity": None,
            "one_minus_fidelity": None,
            "infidelity_source_key": None,
            "infidelity_status": statuses.get("infidelity_reference_replay") or "missing_exact_state_fidelity",
            "infidelity_statuses": statuses,
        }
    return {
        "fidelity": float(max(0.0, min(1.0, 1.0 - same_infidelity))),
        "infidelity": same_infidelity,
        "one_minus_fidelity": same_infidelity,
        "infidelity_same": same_infidelity,
        "infidelity_source_key": "infidelity_same",
        "infidelity_status": "ok",
        "infidelity_statuses": statuses,
        "fidelity_replay_source": str(row.get("exact_state_fidelity_source") or "adapt_vqe.exact_state_fidelity"),
    }


def _snake_rows_from_summary(
    *,
    summary_path: Path,
    threshold: float,
    suite_profile: str,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    summary = _load_json(summary_path)
    result_path = _best_result_path(summary_path, summary)
    if result_path is None:
        return None
    result = _load_json(result_path)
    case_id = str(summary_path.parent.name)
    family = _family_from_case_id(case_id)
    best_trial_number = int(summary.get("best_trial_number"))
    row = result.get("adapt_vqe") if isinstance(result.get("adapt_vqe"), Mapping) else {}
    crossing = result.get("paper_i_first_crossing") if isinstance(result.get("paper_i_first_crossing"), Mapping) else {}
    best_attrs = summary.get("best_user_attrs") if isinstance(summary.get("best_user_attrs"), Mapping) else {}
    objective = best_attrs.get("objective_score_components") if isinstance(best_attrs.get("objective_score_components"), Mapping) else {}
    resources = objective.get("terminal_compiled_resources") if isinstance(objective.get("terminal_compiled_resources"), Mapping) else {}
    controller_proxy = snake_controller_shot_proxy_from_payload(result, source_label=str(result_path))
    controller_proxy_status = str(controller_proxy.get("status") or "unknown")
    corrected_shot_proxy = (
        _num(controller_proxy.get("controller_shot_proxy"))
        if controller_proxy_status == "ok"
        else None
    )
    legacy_shot_proxy = _num(resources.get("shot_cost_proxy"))
    abs_delta_e = _num(crossing.get("primary_error_at_crossing")) or _num(row.get("abs_delta_e"))
    reached = bool(crossing.get("reached"))
    source_kind = (
        "snake_qiskit_compiled_first_hit_ansatz_circuit"
        if reached and bool(objective.get("terminal_compile_matches_first_hit"))
        else "snake_qiskit_compiled_final_ansatz_circuit"
    )
    common = {
        "family": family,
        "case_id": case_id,
        "algorithm_id": SNAKE_ALGORITHM_ID,
        "method": SNAKE_METHOD,
        "record_id": f"routeA__{case_id}__snake",
        "source_json": str(result_path),
        "source_rank": "snake_best_optuna_trial",
        "status": "completed" if bool(row.get("success") is True or result_path.exists()) else "unknown",
        "best_trial_number": best_trial_number,
        "suite_profile": suite_profile,
        "abs_delta_e": abs_delta_e,
        "delta_e_excess_display": None if abs_delta_e is None else max(0.0, float(abs_delta_e) - float(threshold)),
        "paper_i_first_crossing": dict(crossing),
        "first_crossing_reached": reached,
        "first_crossing_history_position_tau": crossing.get("history_position_tau"),
    }
    cost_row = {
        **common,
        "compiled_count_2q_total": _num(resources.get("count_2q")),
        "compiled_depth_2q_total": _num(resources.get("depth_2q")),
        "compiled_depth_total": _num(resources.get("circuit_depth")),
        "compiled_circuit_stats_status": "ok" if resources else "missing_terminal_compiled_resources",
        "compiled_resource_source_kind": source_kind,
        "first_hit_cost_source_kind": source_kind,
        "qiskit_first_hit_cost_status": objective.get("qiskit_first_hit_cost_status"),
        "paper_facing_resource_display_reason": objective.get("paper_facing_resource_display_reason"),
        "Snorm": corrected_shot_proxy,
        "Snorm_source": (
            "snake_runtime_controller_phase_reconstruction_v1"
            if corrected_shot_proxy is not None
            else None
        ),
        "Snorm_status": controller_proxy_status,
        "legacy_terminal_compiled_resources_shot_cost_proxy": legacy_shot_proxy,
        "legacy_terminal_compiled_resources_shot_cost_proxy_source": (
            "objective_score_components.terminal_compiled_resources.shot_cost_proxy"
            if legacy_shot_proxy is not None
            else None
        ),
        "snake_controller_shot_proxy_reconstruction": controller_proxy,
        "selected_logical_phase0_accounting": controller_proxy.get("selected_logical_phase0_accounting"),
    }
    fidelity_fields = _snake_fidelity_fields(
        family=family,
        case_id=case_id,
        suite_profile=suite_profile,
        result=result,
    )
    fidelity_row = {
        **common,
        **fidelity_fields,
        "enrichment_json": None,
        "enrichment_status": "snake_result_replay",
    }
    return cost_row, fidelity_row


def _merge_rows(base: Sequence[Mapping[str, Any]], additions: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {
        (str(row.get("family")), str(row.get("case_id")), str(row.get("algorithm_id"))): dict(row)
        for row in base
    }
    for row in additions:
        merged[(str(row.get("family")), str(row.get("case_id")), str(row.get("algorithm_id")))] = dict(row)
    return [merged[key] for key in sorted(merged)]


def build_support_maps(
    *,
    comparator_cost_source: Path | None,
    comparator_fidelity_source: Path | None,
    snake_roots: Sequence[Path],
    output_cost_source: Path,
    output_fidelity_source: Path,
    threshold: float,
    suite_profile: str,
) -> dict[str, Any]:
    cost_source = _load_json(comparator_cost_source)
    fidelity_source = _load_json(comparator_fidelity_source)
    cost_rows = list(cost_source.get("rows") or [])
    fidelity_rows = list(fidelity_source.get("rows") or [])

    snake_cost_rows: list[dict[str, Any]] = []
    snake_fidelity_rows: list[dict[str, Any]] = []
    for summary_path in _snake_summary_paths(snake_roots):
        pair = _snake_rows_from_summary(
            summary_path=summary_path,
            threshold=float(threshold),
            suite_profile=suite_profile,
        )
        if pair is None:
            continue
        cost_row, fidelity_row = pair
        snake_cost_rows.append(cost_row)
        snake_fidelity_rows.append(fidelity_row)

    generated_utc = datetime.now(timezone.utc).isoformat()
    cost_payload = {
        "schema": SCHEMA,
        "source_kind": "paper_i_three_model_cost_sources_with_snake",
        "generated_utc": generated_utc,
        "threshold": float(threshold),
        "comparator_source_json": None if comparator_cost_source is None else str(comparator_cost_source),
        "snake_roots": [str(path) for path in snake_roots],
        "rows": _merge_rows(cost_rows, snake_cost_rows),
    }
    fidelity_payload = {
        "schema": SCHEMA,
        "source_kind": "paper_i_three_model_fidelity_sources_with_snake",
        "generated_utc": generated_utc,
        "threshold": float(threshold),
        "fidelity_semantics": {
            "default": "post_hoc overlap diagnostic for displayed ansatz",
            "phonon_rows": "reference-cutoff overlap when the ansatz can be replayed into the ED cutoff; otherwise same-cutoff overlap is labeled by infidelity_source_key",
            "decision_usage": "post_hoc_reporting_only",
        },
        "comparator_source_json": None if comparator_fidelity_source is None else str(comparator_fidelity_source),
        "snake_roots": [str(path) for path in snake_roots],
        "rows": _merge_rows(fidelity_rows, snake_fidelity_rows),
    }
    output_cost_source.parent.mkdir(parents=True, exist_ok=True)
    output_fidelity_source.parent.mkdir(parents=True, exist_ok=True)
    output_cost_source.write_text(json.dumps(cost_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_fidelity_source.write_text(json.dumps(fidelity_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "schema": SCHEMA,
        "cost_output": str(output_cost_source),
        "fidelity_output": str(output_fidelity_source),
        "snake_row_count": len(snake_cost_rows),
        "cost_row_count": len(cost_payload["rows"]),
        "fidelity_row_count": len(fidelity_payload["rows"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparator-cost-source", type=Path, default=None)
    parser.add_argument("--comparator-fidelity-source", type=Path, default=None)
    parser.add_argument("--snake-root", action="append", type=Path, default=[])
    parser.add_argument("--output-cost-source", type=Path, required=True)
    parser.add_argument("--output-fidelity-source", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=2e-4)
    parser.add_argument("--suite-profile", default="paper_i_three_model_main_20260525_v1")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_support_maps(
        comparator_cost_source=args.comparator_cost_source,
        comparator_fidelity_source=args.comparator_fidelity_source,
        snake_roots=tuple(args.snake_root),
        output_cost_source=args.output_cost_source,
        output_fidelity_source=args.output_fidelity_source,
        threshold=float(args.threshold),
        suite_profile=str(args.suite_profile),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
