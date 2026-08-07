"""Ordered batch-admission helpers for static ADAPT."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import AnsatzParameterLayout, runtime_insert_position
from src.quantum.vqe_latex_python_pairs import AnsatzTerm
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.scaffold.hh_continuation_scoring import (
    _batch_record_generator_identity,
)

__all__ = [
    "BatchOrderingConfig",
    "BatchOrderingRuntime",
    "_batch_admission_record_key",
    "_batch_record_term",
    "_batch_order_trial_state",
    "_finite_step_order_rescue_records",
    "_order_batch_records_for_admission",
    "_record_candidate_label",
    "_schur_batch_context_from_summary",
]


@dataclass(frozen=True)
class BatchOrderingConfig:
    mode: str
    max_permutations: int
    rho: float
    batch_target_size: int
    batch_size_cap: int
    batch_near_degenerate_ratio: float


@dataclass(frozen=True)
class BatchOrderingRuntime:
    pool: Sequence[AnsatzTerm]
    adapt_state_backend_key: str
    build_selected_layout: Callable[[list[AnsatzTerm]], AnsatzParameterLayout]
    build_compiled_executor: Callable[[list[AnsatzTerm]], Any]
    splice_candidate_at_position: Callable[..., tuple[list[AnsatzTerm], np.ndarray]]
    evaluate_selected_energy_objective: Callable[..., float]


def _record_candidate_label(record: Mapping[str, Any]) -> str:
    value = record.get("candidate_label")
    if value not in {None, ""}:
        return str(value)
    feat = record.get("feature")
    if isinstance(feat, CandidateFeatures):
        feat_label = getattr(feat, "candidate_label", None)
        if feat_label not in {None, ""}:
            return str(feat_label)
    candidate_term = record.get("candidate_term")
    if isinstance(candidate_term, AnsatzTerm):
        return str(candidate_term.label)
    return ""


def _batch_admission_record_key(record: Mapping[str, Any]) -> tuple[int, int, str]:
    feat = record.get("feature")
    if isinstance(feat, CandidateFeatures):
        return (
            int(feat.candidate_pool_index),
            int(feat.position_id),
            str(feat.candidate_label),
        )
    return (
        int(record.get("candidate_pool_index", -1)),
        int(record.get("position_id", -1)),
        _record_candidate_label(record),
    )


def _batch_record_term(
    record: Mapping[str, Any],
    *,
    pool: Sequence[AnsatzTerm],
) -> tuple[int, AnsatzTerm] | None:
    feat = record.get("feature")
    if not isinstance(feat, CandidateFeatures):
        return None
    idx_sel = int(feat.candidate_pool_index)
    term = record.get("candidate_term")
    if not isinstance(term, AnsatzTerm):
        if idx_sel < 0 or idx_sel >= len(pool):
            return None
        term = pool[int(idx_sel)]
    return int(idx_sel), term


def _batch_order_trial_state(
    *,
    runtime: BatchOrderingRuntime,
    base_ops: Sequence[AnsatzTerm],
    base_theta: np.ndarray,
    base_layout: AnsatzParameterLayout,
    ordered_records: Sequence[Mapping[str, Any]],
) -> tuple[list[AnsatzTerm], np.ndarray, AnsatzParameterLayout, list[tuple[int, int]], str | None]:
    trial_ops = list(base_ops)
    trial_theta = np.asarray(base_theta, dtype=float).reshape(-1).copy()
    trial_layout = base_layout
    inserted_runtime_slices: list[tuple[int, int]] = []
    original_positions_seen: list[int] = []
    for record in ordered_records:
        feat = record.get("feature")
        term_payload = _batch_record_term(record, pool=runtime.pool)
        if not isinstance(feat, CandidateFeatures) or term_payload is None:
            return trial_ops, trial_theta, trial_layout, inserted_runtime_slices, "invalid_record"
        _idx_sel, admitted_term = term_payload
        pos_orig = int(feat.position_id)
        pos_eff = int(pos_orig + sum(1 for prev in original_positions_seen if int(prev) <= int(pos_orig)))
        admitted_layout = runtime.build_selected_layout([admitted_term])
        runtime_insert_pos = int(runtime_insert_position(trial_layout, int(pos_eff)))
        trial_ops, trial_theta = runtime.splice_candidate_at_position(
            ops=trial_ops,
            theta=np.asarray(trial_theta, dtype=float),
            op=admitted_term,
            position_id=int(pos_eff),
            init_theta=0.0,
        )
        inserted_runtime_slices.append(
            (
                int(runtime_insert_pos),
                int(runtime_insert_pos) + int(admitted_layout.runtime_parameter_count),
            )
        )
        trial_layout = runtime.build_selected_layout(trial_ops)
        original_positions_seen.append(int(pos_orig))
    return trial_ops, trial_theta, trial_layout, inserted_runtime_slices, None


def _order_batch_records_for_admission(
    *,
    records: Sequence[Mapping[str, Any]],
    base_ops: Sequence[AnsatzTerm],
    base_theta: np.ndarray,
    base_layout: AnsatzParameterLayout,
    depth_one_based: int,
    config: BatchOrderingConfig,
    runtime: BatchOrderingRuntime,
    score_singleton: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records_list = [dict(record) for record in records]
    original_keys = [_batch_admission_record_key(record) for record in records_list]
    original_labels = [str(key[2]) for key in original_keys]
    mode = str(config.mode)
    summary_base: dict[str, Any] = {
        "schema": "phase3_ordered_batch_admission_v1",
        "mode": str(mode),
        "selected": False,
        "reordered": False,
        "reason": "not_applicable",
        "depth": int(depth_one_based),
        "input_count": int(len(records_list)),
        "original_labels": [str(x) for x in original_labels],
        "selected_labels": [str(x) for x in original_labels],
    }
    if len(records_list) <= 1 and not bool(score_singleton):
        summary_base["reason"] = "singleton_batch"
        return records_list, summary_base
    if mode == "score_sorted":
        summary_base["reason"] = "score_sorted_legacy"
        return records_list, summary_base
    if mode != "finite_step_v1":
        summary_base["reason"] = "unsupported_mode_fallback"
        return records_list, summary_base

    max_perms = int(max(1, config.max_permutations))
    step0 = float(min(0.35, max(1e-4, float(config.rho))))
    best_records = records_list
    best_energy = float("inf")
    best_eval_count = 0
    best_passes = 0
    scored_orders: list[dict[str, Any]] = []
    permutation_count = 0
    truncated = False
    for permutation in itertools.permutations(records_list):
        if permutation_count >= max_perms:
            truncated = True
            break
        permutation_count += 1
        perm_records = [dict(record) for record in permutation]
        trial_ops, trial_theta, trial_layout, inserted_slices, invalid_reason = _batch_order_trial_state(
            runtime=runtime,
            base_ops=list(base_ops),
            base_theta=np.asarray(base_theta, dtype=float),
            base_layout=base_layout,
            ordered_records=perm_records,
        )
        labels_now = [str(_batch_admission_record_key(record)[2]) for record in perm_records]
        if invalid_reason is not None or not inserted_slices:
            scored_orders.append(
                {
                    "labels": labels_now,
                    "valid": False,
                    "reason": str(invalid_reason or "no_inserted_slices"),
                }
            )
            continue
        trial_executor = (
            runtime.build_compiled_executor(trial_ops)
            if str(runtime.adapt_state_backend_key) == "compiled"
            else None
        )

        def _proxy_energy(theta_probe: np.ndarray) -> float:
            return float(
                runtime.evaluate_selected_energy_objective(
                    ops_now=list(trial_ops),
                    theta_now=np.asarray(theta_probe, dtype=float),
                    executor_now=trial_executor,
                    parameter_layout_now=trial_layout,
                    objective_stage="batch_order_finite_step_proxy",
                    depth_marker=int(depth_one_based),
                )
            )

        theta_best = np.asarray(trial_theta, dtype=float).reshape(-1).copy()
        eval_count = 0
        energy_best = _proxy_energy(theta_best)
        eval_count += 1
        step = float(step0)
        passes = 0
        for _pass_idx in range(2):
            passes += 1
            any_improved = False
            for start, stop in inserted_slices:
                start_i = int(start)
                stop_i = int(stop)
                if stop_i <= start_i:
                    continue
                current_value = float(np.mean(theta_best[start_i:stop_i]))
                candidates = (current_value + step, current_value - step, current_value)
                local_best_value = current_value
                local_best_energy = float(energy_best)
                for candidate_value in candidates:
                    theta_probe = np.asarray(theta_best, dtype=float).copy()
                    theta_probe[start_i:stop_i] = float(candidate_value)
                    energy_probe = _proxy_energy(theta_probe)
                    eval_count += 1
                    if float(energy_probe) < float(local_best_energy) - 1e-12:
                        local_best_energy = float(energy_probe)
                        local_best_value = float(candidate_value)
                if float(local_best_energy) < float(energy_best) - 1e-12:
                    theta_best[start_i:stop_i] = float(local_best_value)
                    energy_best = float(local_best_energy)
                    any_improved = True
            step *= 0.5
            if not any_improved:
                break
        scored_orders.append(
            {
                "labels": labels_now,
                "valid": True,
                "energy_proxy": float(energy_best),
                "eval_count": int(eval_count),
                "passes": int(passes),
            }
        )
        if float(energy_best) < float(best_energy) - 1e-12:
            best_energy = float(energy_best)
            best_records = perm_records
            best_eval_count = int(eval_count)
            best_passes = int(passes)

    selected_keys = [_batch_admission_record_key(record) for record in best_records]
    reordered = [tuple(x) for x in selected_keys] != [tuple(x) for x in original_keys]
    summary = {
        **summary_base,
        "selected": bool(scored_orders),
        "reordered": bool(reordered),
        "reason": "finite_step_proxy_scored" if scored_orders else "no_valid_orderings",
        "selected_labels": [str(key[2]) for key in selected_keys],
        "permutation_count": int(permutation_count),
        "max_permutations": int(max_perms),
        "truncated": bool(truncated),
        "finite_step": float(step0),
        "best_energy_proxy": (float(best_energy) if math.isfinite(float(best_energy)) else None),
        "best_eval_count": int(best_eval_count),
        "best_passes": int(best_passes),
        "orders_scored_sample": [dict(row) for row in scored_orders[: min(8, len(scored_orders))]],
    }
    annotated: list[dict[str, Any]] = []
    for rank, record in enumerate(best_records):
        updated = dict(record)
        updated["batch_order_proxy"] = {
            "schema": "phase3_ordered_batch_admission_v1",
            "mode": str(mode),
            "rank": int(rank),
            "reordered": bool(reordered),
            "selected_labels": [str(key[2]) for key in selected_keys],
            "best_energy_proxy": summary["best_energy_proxy"],
        }
        annotated.append(updated)
    return annotated, summary


def _int_list(value: Any) -> list[int]:
    if value is None or isinstance(value, (str, bytes, bytearray)):
        return []
    if not isinstance(value, Sequence):
        return []
    out: list[int] = []
    for raw in value:
        try:
            out.append(int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _schur_batch_context_from_summary(
    *,
    records: Sequence[Mapping[str, Any]],
    batch_summary: Mapping[str, Any] | None,
    int_list: Callable[[Any], list[int]] = _int_list,
) -> dict[str, Any]:
    if not isinstance(batch_summary, Mapping) or len(records) <= 1:
        return {}
    alpha_raw = batch_summary.get("alpha")
    G_raw = batch_summary.get("G")
    common_window_raw = batch_summary.get("common_window_indices")
    solves_raw = batch_summary.get("schur_window_solves")
    if not isinstance(alpha_raw, Sequence) or isinstance(alpha_raw, (str, bytes, bytearray)):
        return {}
    if len(alpha_raw) != len(records):
        return {}
    if not isinstance(solves_raw, Sequence) or isinstance(solves_raw, (str, bytes, bytearray)):
        return {}
    if len(solves_raw) != len(records):
        return {}
    try:
        keys = [_batch_admission_record_key(record) for record in records]
        alpha = [float(x) for x in alpha_raw]
        common_window = [int(x) for x in int_list(common_window_raw)]
        solves = [[float(x) for x in solve] for solve in solves_raw]
    except Exception:
        return {}
    if len(set(keys)) != len(keys):
        return {}
    if any(len(solve) != len(common_window) for solve in solves):
        return {}
    payload: dict[str, Any] = {
        "schema": "static_adapt_batch_schur_context_v1",
        "source": "reduced_plane_batch_select",
        "batch_model": "linear_window_superposition_diag_candidate_curvature_v1",
        "h_alphaalpha_offdiag_available": False,
        "candidate_curvature_model": "diagonal_h_eff_only",
        "joint_alpha_source": "reduced_plane_batch_select.alpha",
        "record_keys": [list(key) for key in keys],
        "G_key_order": [list(key) for key in keys],
        "alpha_abs_by_key": {json.dumps(list(key), sort_keys=True): float(alpha[idx]) for idx, key in enumerate(keys)},
        "schur_window_solve_by_key": {
            json.dumps(list(key), sort_keys=True): [float(x) for x in solves[idx]]
            for idx, key in enumerate(keys)
        },
        "common_window_indices": [int(x) for x in common_window],
    }
    if isinstance(G_raw, Sequence) and not isinstance(G_raw, (str, bytes, bytearray)):
        try:
            payload["G"] = [[float(x) for x in row] for row in G_raw]
        except Exception:
            pass
    for name in (
        "joint_gain",
        "contextual_single_total",
        "additivity_defect",
        "lambda_min",
        "rank_floor",
        "mu_tan",
    ):
        try:
            value = float(batch_summary.get(name))
        except Exception:
            continue
        if math.isfinite(value):
            payload[name] = float(value)
    return payload


def _order_rescue_record_score(record: Mapping[str, Any]) -> float:
    for key in ("full_v2_score", "phase2_raw_score", "simple_score"):
        try:
            value = float(record.get(key, float("-inf")))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return float(value)
    return float("-inf")


def _order_rescue_dormant_sort_key(
    record: Mapping[str, Any],
    *,
    record_sort_key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
) -> tuple[int, tuple[Any, ...]]:
    key_base = record_sort_key(record)
    label = str(_batch_admission_record_key(record)[2])
    hamiltonian_fallback = int(label == "ham_full" or label.startswith("ham_term("))
    return (hamiltonian_fallback, tuple(key_base))


def _finite_step_order_rescue_records(
    *,
    source_records: Sequence[Mapping[str, Any]],
    selected_records: Sequence[Mapping[str, Any]],
    depth_one_based: int,
    config: BatchOrderingConfig,
    record_sort_key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [dict(record) for record in selected_records]
    target_count = int(
        max(
            1,
            min(
                int(max(1, config.batch_target_size)),
                int(max(1, config.batch_size_cap)),
            ),
        )
    )
    mode = str(config.mode)
    summary: dict[str, Any] = {
        "schema": "phase3_ordered_batch_admission_rescue_v1",
        "mode": str(mode),
        "depth": int(depth_one_based),
        "target_count": int(target_count),
        "selected_input_count": int(len(selected)),
        "candidate_source_count": int(len(source_records)),
        "used": False,
        "reason": "not_attempted",
    }
    if mode != "finite_step_v1":
        summary["reason"] = "order_mode_not_finite_step"
        return selected, summary
    if target_count <= int(len(selected)):
        summary["reason"] = "target_already_satisfied"
        return selected, summary

    ranked = sorted(
        [dict(record) for record in source_records],
        key=record_sort_key,
    )
    positive = [
        dict(record)
        for record in ranked
        if float(_order_rescue_record_score(record)) > 0.0
    ]
    if not positive:
        summary["reason"] = "no_positive_candidate_scores"
        return selected, summary

    top_score = float(_order_rescue_record_score(positive[0]))
    ratio = float(max(0.0, min(1.0, config.batch_near_degenerate_ratio)))
    chosen: list[dict[str, Any]] = [dict(record) for record in selected]
    seen_keys = {_batch_admission_record_key(record) for record in chosen}
    seen_labels = {str(key[2]) for key in seen_keys}
    seen_generator_identities = {
        _batch_record_generator_identity(record) for record in chosen
    }

    def _try_append(record: Mapping[str, Any], *, source: str) -> None:
        if len(chosen) >= target_count:
            return
        key = _batch_admission_record_key(record)
        label = str(key[2])
        generator_identity = _batch_record_generator_identity(record)
        if (
            key in seen_keys
            or label in seen_labels
            or generator_identity in seen_generator_identities
        ):
            return
        score = float(_order_rescue_record_score(record))
        if not math.isfinite(score) or score <= 0.0:
            return
        updated = dict(record)
        updated["batch_order_rescue"] = {
            "schema": "phase3_ordered_batch_admission_rescue_v1",
            "source": str(source),
            "score": float(score),
            "top_score": float(top_score),
            "near_degenerate_ratio": float(ratio),
        }
        chosen.append(updated)
        seen_keys.add(key)
        seen_labels.add(label)
        seen_generator_identities.add(generator_identity)

    def _try_append_dormant(record: Mapping[str, Any], *, source: str) -> None:
        if len(chosen) >= target_count:
            return
        key = _batch_admission_record_key(record)
        label = str(key[2])
        generator_identity = _batch_record_generator_identity(record)
        if (
            key in seen_keys
            or label in seen_labels
            or generator_identity in seen_generator_identities
        ):
            return
        score = float(_order_rescue_record_score(record))
        if not math.isfinite(score) or score > 0.0:
            return
        updated = dict(record)
        updated["batch_order_rescue"] = {
            "schema": "phase3_ordered_batch_admission_rescue_v1",
            "source": str(source),
            "score": float(score),
            "top_score": float(top_score),
            "near_degenerate_ratio": float(ratio),
            "dormant": True,
        }
        chosen.append(updated)
        seen_keys.add(key)
        seen_labels.add(label)
        seen_generator_identities.add(generator_identity)

    for record in positive:
        score = float(_order_rescue_record_score(record))
        if score >= float(ratio) * float(top_score):
            _try_append(record, source="near_degenerate_shell")
        if len(chosen) >= target_count:
            break
    for record in positive:
        _try_append(record, source="positive_score_fill")
        if len(chosen) >= target_count:
            break
    dormant = [
        dict(record)
        for record in sorted(
            ranked,
            key=lambda row: _order_rescue_dormant_sort_key(
                row,
                record_sort_key=record_sort_key,
            ),
        )
        if math.isfinite(float(_order_rescue_record_score(record)))
        and float(_order_rescue_record_score(record)) <= 0.0
    ]
    for record in dormant:
        _try_append_dormant(record, source="dormant_finite_step_fill")
        if len(chosen) >= target_count:
            break

    summary.update(
        used=bool(len(chosen) > len(selected)),
        reason=(
            "finite_step_positive_dormant_shell"
            if len(chosen) > len(selected)
            else "no_distinct_rescue_fill_records"
        ),
        selected_input_labels=[
            str(_batch_admission_record_key(record)[2]) for record in selected
        ],
        rescue_candidate_labels=[
            str(_batch_admission_record_key(record)[2]) for record in chosen
        ],
        top_score=float(top_score),
        positive_candidate_count=int(len(positive)),
        dormant_candidate_count=int(len(dormant)),
        near_degenerate_ratio=float(ratio),
        dormant_fill_policy="finite_nonpositive_non_hamiltonian_before_hamiltonian_fallback",
    )
    return chosen, summary
