#!/usr/bin/env python3
"""Checkpoint-local ADAPT-style scaffold growth diagnostic for HH fidelity recovery."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.scaffold.hh_continuation_rescue import rank_candidates_by_gain
from pipelines.scaffold.hh_continuation_generators import _polynomial_signature
from pipelines.static_adapt.builders.hh_pool_presets import build_hh_pool_by_key
from pipelines.time_dynamics.fixed_manifold.exact_fit import (
    FrozenScaffoldExactFitConfig,
    capture_checkpoint_snapshot_from_args,
    fit_checkpoint_snapshot,
)
from pipelines.time_dynamics.legacy.checkpoint_controller import (
    RuntimeTermCarrier,
    _build_candidate_carrier,
    _carrier_to_term,
    _insert_theta_block,
    _layout_from_carriers,
    _site_resolved_number_observables,
)
from pipelines.time_dynamics.runners.hh_from_adapt_artifact import (
    _to_jsonable,
    build_controller_bundle_from_args,
    build_parser as build_realtime_parser,
)
from src.quantum.ansatz_parameterization import AnsatzParameterLayout, build_parameter_layout, runtime_insert_position
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


@dataclass(frozen=True)
class CheckpointLocalAdaptConfig:
    strategy: str = "gradient_local_v1"
    objective: str = "fidelity_first"
    pool_mode: str = "family_pool"
    target_fidelity: float = 0.99
    max_steps: int = 8
    gradient_threshold: float = 1.0e-6
    probe_scale: float = 0.15
    min_fidelity_gain: float = 1.0e-4
    plateau_patience: int = 2
    candidate_rank_limit: int = 8
    joint_site_weight: float = 1.0
    joint_energy_weight: float = 1.0
    joint_energy_norm_floor: float = 1.0e-8
    joint_min_gain: float = 1.0e-6
    joint_opt_mode: str = "fidelity_fit_joint_rank"


@dataclass(frozen=True)
class CheckpointLocalAdaptRuntimeState:
    terms: tuple[Any, ...]
    layout: AnsatzParameterLayout
    executor: CompiledAnsatzExecutor
    theta_runtime: np.ndarray
    metrics: dict[str, Any]
    scaffold_labels: tuple[str, ...]
    source_labels: tuple[str, ...]


@dataclass(frozen=True)
class CheckpointLocalAdaptRuntimeResult:
    payload: dict[str, Any]
    state: CheckpointLocalAdaptRuntimeState


def _parse_int_tuple(raw: str | None) -> tuple[int, ...]:
    if raw is None:
        return ()
    text = str(raw).strip()
    if not text:
        return ()
    return tuple(int(chunk.strip()) for chunk in text.split(",") if chunk.strip())


def _to_ansatz_term(term_like: Any) -> AnsatzTerm:
    if isinstance(term_like, AnsatzTerm):
        return term_like
    if isinstance(term_like, RuntimeTermCarrier):
        return _carrier_to_term(term_like)
    if hasattr(term_like, "polynomial") and hasattr(term_like, "label"):
        return AnsatzTerm(label=str(term_like.label), polynomial=term_like.polynomial)
    raise TypeError(f"Cannot convert {type(term_like)!r} into AnsatzTerm.")


def _term_signature(term_like: Any) -> tuple[tuple[str, float], ...]:
    return _polynomial_signature(_to_ansatz_term(term_like).polynomial)


def _current_source_labels(terms: Sequence[Any]) -> set[str]:
    out: set[str] = set()
    for term in terms:
        if isinstance(term, RuntimeTermCarrier):
            out.add(str(term.source_label))
        else:
            out.add(str(_to_ansatz_term(term).label))
    return out


def _current_scaffold_labels(terms: Sequence[Any]) -> list[str]:
    return [str(term.label) for term in terms]


def _build_executor(terms: Sequence[Any], layout: AnsatzParameterLayout) -> CompiledAnsatzExecutor:
    return CompiledAnsatzExecutor(
        [_to_ansatz_term(term) for term in terms],
        coefficient_tolerance=float(layout.coefficient_tolerance),
        ignore_identity=bool(layout.ignore_identity),
        sort_terms=(str(layout.term_order).strip().lower() == "sorted"),
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )


def _append_candidate_to_scaffold(
    current_terms: Sequence[Any],
    *,
    current_layout: AnsatzParameterLayout,
    current_theta: np.ndarray | Sequence[float],
    candidate_term: AnsatzTerm,
    candidate_pool_index: int,
    position_id: int | None = None,
) -> dict[str, Any]:
    theta_arr = np.asarray(current_theta, dtype=float).reshape(-1)
    insert_at = int(len(current_terms) if position_id is None else position_id)
    if all(isinstance(term, RuntimeTermCarrier) for term in current_terms):
        unique_label = f"{candidate_term.label}__pool{int(candidate_pool_index)}__p{int(insert_at)}"
        candidate_carrier = _build_candidate_carrier(
            candidate_term,
            logical_index=int(insert_at),
            unique_label=str(unique_label),
            template_layout=current_layout,
            candidate_pool_index=int(candidate_pool_index),
        )
        aug_terms = list(current_terms)
        aug_terms.insert(int(insert_at), candidate_carrier)
        aug_layout = _layout_from_carriers(aug_terms, template=current_layout)
        runtime_pos = int(runtime_insert_position(current_layout, int(insert_at)))
        theta_aug = _insert_theta_block(
            theta_arr,
            runtime_position=int(runtime_pos),
            width=int(len(candidate_carrier.runtime_specs)),
        )
        new_runtime_indices = tuple(
            range(int(runtime_pos), int(runtime_pos + len(candidate_carrier.runtime_specs)))
        )
    else:
        aug_terms = list(current_terms)
        aug_terms.insert(int(insert_at), _to_ansatz_term(candidate_term))
        aug_layout = build_parameter_layout(
            [_to_ansatz_term(term) for term in aug_terms],
            ignore_identity=bool(current_layout.ignore_identity),
            coefficient_tolerance=float(current_layout.coefficient_tolerance),
            sort_terms=(str(current_layout.term_order).strip().lower() == "sorted"),
        )
        old_blocks = tuple(current_layout.blocks)
        if tuple(aug_layout.blocks)[: len(old_blocks)] != old_blocks:
            raise ValueError("Augmented scaffold changed the runtime-layout prefix; cannot preserve current theta.")
        theta_aug = np.concatenate(
            [
                theta_arr,
                np.zeros(
                    int(aug_layout.runtime_parameter_count) - int(current_layout.runtime_parameter_count),
                    dtype=float,
                ),
            ]
        )
        new_runtime_indices = tuple(
            range(int(current_layout.runtime_parameter_count), int(aug_layout.runtime_parameter_count))
        )
    aug_executor = _build_executor(aug_terms, aug_layout)
    return {
        "aug_terms": list(aug_terms),
        "aug_layout": aug_layout,
        "aug_executor": aug_executor,
        "theta_aug": np.asarray(theta_aug, dtype=float).reshape(-1),
        "new_runtime_indices": tuple(int(idx) for idx in new_runtime_indices),
    }


def _observable_payload(
    psi_state: np.ndarray,
    *,
    num_sites: int,
    ordering: str,
) -> dict[str, Any]:
    raw = _site_resolved_number_observables(
        np.asarray(psi_state, dtype=complex).reshape(-1),
        num_sites=int(num_sites),
        ordering=str(ordering),
    )
    return {
        "site_occupations": [float(x) for x in np.asarray(raw.n_site, dtype=float).tolist()],
        "doublon": float(raw.doublon),
        "staggered": float(raw.staggered),
    }


def _build_snapshot_with_scaffold(
    base_snapshot: Mapping[str, Any],
    *,
    terms: Sequence[Any],
    layout: AnsatzParameterLayout,
    executor: CompiledAnsatzExecutor,
    theta_runtime: np.ndarray | Sequence[float],
) -> dict[str, Any]:
    theta_arr = np.asarray(theta_runtime, dtype=float).reshape(-1)
    psi_ref = np.asarray(base_snapshot["psi_ref"], dtype=complex).reshape(-1)
    psi_current = np.asarray(executor.prepare_state(theta_arr, psi_ref), dtype=complex).reshape(-1)
    current_observables = _observable_payload(
        psi_current,
        num_sites=int(base_snapshot["num_sites"]),
        ordering=str(base_snapshot["ordering"]),
    )
    energy_current = float(
        np.real(np.vdot(psi_current, np.asarray(base_snapshot["hmat_step"], dtype=complex) @ psi_current))
    )
    out = dict(base_snapshot)
    out.update(
        {
            "terms": list(terms),
            "layout": layout,
            "executor": executor,
            "theta_runtime": theta_arr.copy(),
            "psi_current": psi_current,
            "current_observables": dict(current_observables),
            "energy_current": float(energy_current),
            "abs_energy_total_error": float(abs(float(base_snapshot["energy_exact"]) - energy_current)),
            "scaffold_labels": _current_scaffold_labels(terms),
            "logical_block_count": int(layout.logical_parameter_count),
            "runtime_parameter_count": int(layout.runtime_parameter_count),
        }
    )
    return out


def _objective_payload(fit_payload: Mapping[str, Any], *, objective: str) -> dict[str, Any]:
    for row in fit_payload.get("objectives", []):
        if str(row.get("objective")) == str(objective):
            return dict(row)
    objectives = list(fit_payload.get("objectives", []))
    if not objectives:
        raise ValueError("fit_payload has no objective rows.")
    return dict(objectives[0])


def resolve_candidate_pool_terms(
    bundle: Mapping[str, Any],
    *,
    pool_mode: str,
) -> tuple[list[AnsatzTerm], dict[str, Any]]:
    replay_context = bundle["loaded"].replay_context
    mode_key = str(pool_mode).strip().lower()
    if mode_key == "family_pool":
        return list(replay_context.family_pool), {
            "pool_mode": "family_pool",
            "candidate_pool_complete": bool(replay_context.pool_meta.get("candidate_pool_complete", False)),
            "raw_pool_size": int(len(replay_context.family_pool)),
            "resolved_family": str(replay_context.family_info.get("resolved", "unknown")),
        }
    if mode_key == "full_meta":
        cfg = replay_context.cfg
        pool_terms, method_name, class_meta, label_meta = build_hh_pool_by_key(
            pool_key_hh="full_meta",
            h_poly=replay_context.h_poly,
            num_sites=int(cfg.L),
            t=float(cfg.t),
            u=float(cfg.u),
            omega0=float(cfg.omega0),
            g_ep=float(cfg.g_ep),
            dv=float(cfg.dv),
            n_ph_max=int(cfg.n_ph_max),
            boson_encoding=str(cfg.boson_encoding),
            ordering=str(cfg.ordering),
            boundary=str(cfg.boundary),
            paop_r=int(cfg.paop_r),
            paop_split_paulis=bool(cfg.paop_split_paulis),
            paop_prune_eps=float(cfg.paop_prune_eps),
            paop_normalization=str(cfg.paop_normalization),
            num_particles=(int(cfg.sector_n_up), int(cfg.sector_n_dn)),
            full_meta_class_filter_spec=None,
            full_meta_label_filter_spec=None,
            ai_log=None,
        )
        return list(pool_terms), {
            "pool_mode": "full_meta",
            "pool_method": str(method_name),
            "raw_pool_size": int(len(pool_terms)),
            "class_filter_meta": class_meta,
            "label_filter_meta": label_meta,
        }
    raise ValueError(f"Unknown checkpoint-local ADAPT pool mode {pool_mode!r}.")


def available_candidate_terms(
    current_terms: Sequence[Any],
    pool_terms: Sequence[AnsatzTerm],
) -> list[tuple[int, AnsatzTerm]]:
    selected_signatures = {_term_signature(term) for term in current_terms}
    selected_source_labels = _current_source_labels(current_terms)
    dedup_seen: set[tuple[tuple[str, float], ...]] = set()
    out: list[tuple[int, AnsatzTerm]] = []
    for pool_index, term in enumerate(pool_terms):
        signature = _term_signature(term)
        if signature in selected_signatures or signature in dedup_seen:
            continue
        if str(term.label) in selected_source_labels:
            continue
        dedup_seen.add(signature)
        out.append((int(pool_index), term))
    return out


"""
dF/dθ_i = 2 Re(conj(<ψ*|ψ>) <ψ*|∂_i ψ>) on the appended runtime block at θ_new = 0
"""
def _fidelity_gradient_components(
    *,
    aug_executor: CompiledAnsatzExecutor,
    theta_aug: np.ndarray | Sequence[float],
    psi_ref: np.ndarray,
    psi_exact: np.ndarray,
    runtime_indices: Sequence[int],
) -> list[float]:
    theta_arr = np.asarray(theta_aug, dtype=float).reshape(-1)
    psi_trial, tangents = aug_executor.prepare_state_with_runtime_tangents(
        theta_arr,
        np.asarray(psi_ref, dtype=complex).reshape(-1),
        runtime_indices=[int(idx) for idx in runtime_indices],
    )
    overlap = complex(np.vdot(np.asarray(psi_exact, dtype=complex).reshape(-1), np.asarray(psi_trial, dtype=complex).reshape(-1)))
    grads: list[float] = []
    for idx in runtime_indices:
        tangent = np.asarray(tangents[int(idx)], dtype=complex).reshape(-1)
        grad_val = 2.0 * float(
            np.real(np.conjugate(overlap) * np.vdot(np.asarray(psi_exact, dtype=complex).reshape(-1), tangent))
        )
        grads.append(float(grad_val))
    return grads


def _score_candidates(
    *,
    base_snapshot: Mapping[str, Any],
    current_terms: Sequence[Any],
    current_layout: AnsatzParameterLayout,
    current_theta: np.ndarray | Sequence[float],
    available_terms: Sequence[tuple[int, AnsatzTerm]],
    rank_limit: int,
    position_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    psi_ref = np.asarray(base_snapshot["psi_ref"], dtype=complex).reshape(-1)
    psi_exact = np.asarray(base_snapshot["psi_exact"], dtype=complex).reshape(-1)
    if position_ids is None:
        positions = (int(len(current_terms)),)
    else:
        positions = tuple(int(x) for x in position_ids)
    rows: list[dict[str, Any]] = []
    for pool_index, candidate_term in available_terms:
        for position_id in positions:
            aug_payload = _append_candidate_to_scaffold(
                current_terms,
                current_layout=current_layout,
                current_theta=current_theta,
                candidate_term=candidate_term,
                candidate_pool_index=int(pool_index),
                position_id=int(position_id),
            )
            gradients = _fidelity_gradient_components(
                aug_executor=aug_payload["aug_executor"],
                theta_aug=aug_payload["theta_aug"],
                psi_ref=psi_ref,
                psi_exact=psi_exact,
                runtime_indices=aug_payload["new_runtime_indices"],
            )
            grad_arr = np.asarray(gradients, dtype=float).reshape(-1)
            rows.append(
                {
                    "candidate_pool_index": int(pool_index),
                    "position_id": int(position_id),
                    "label": str(candidate_term.label),
                    "gradient_components": [float(x) for x in grad_arr.tolist()],
                    "gradient_l2": float(np.linalg.norm(grad_arr)),
                    "gradient_max_abs": float(np.max(np.abs(grad_arr))) if grad_arr.size > 0 else 0.0,
                    "simple_score": float(np.linalg.norm(grad_arr)),
                    "selector_score": float(np.linalg.norm(grad_arr)),
                    "candidate_term": candidate_term,
                    "aug_payload": aug_payload,
                }
            )
    rows.sort(key=lambda row: (float(row["gradient_l2"]), float(row["gradient_max_abs"])), reverse=True)
    rank_cap = max(1, int(rank_limit))
    return rows[:rank_cap] + rows[rank_cap:]


def _strategy_fit_cfg(
    *,
    adapt_cfg: CheckpointLocalAdaptConfig,
    fit_cfg: FrozenScaffoldExactFitConfig,
    base_snapshot: Mapping[str, Any] | None = None,
) -> FrozenScaffoldExactFitConfig:
    if str(adapt_cfg.strategy).strip().lower() != "phase3_joint_rescue_v1":
        return fit_cfg
    if str(adapt_cfg.joint_opt_mode).strip().lower() != "joint_fit_joint_rank":
        return fit_cfg
    energy_weight = float(adapt_cfg.joint_energy_weight)
    if base_snapshot is not None:
        energy_weight = float(
            float(adapt_cfg.joint_energy_weight)
            / float(_joint_energy_denominator(base_snapshot=base_snapshot, adapt_cfg=adapt_cfg))
        )
    return replace(
        fit_cfg,
        objectives=("balanced",),
        balanced_energy_weight=float(energy_weight),
        balanced_site_weight=float(adapt_cfg.joint_site_weight),
    )


def _strategy_fit_objective(*, adapt_cfg: CheckpointLocalAdaptConfig) -> str:
    if (
        str(adapt_cfg.strategy).strip().lower() == "phase3_joint_rescue_v1"
        and str(adapt_cfg.joint_opt_mode).strip().lower() == "joint_fit_joint_rank"
    ):
        return "balanced"
    return str(adapt_cfg.objective)


def _joint_energy_denominator(
    *,
    base_snapshot: Mapping[str, Any],
    adapt_cfg: CheckpointLocalAdaptConfig,
) -> float:
    raw = float(base_snapshot.get("reference_energy_total_span_full_run", 0.0))
    if not np.isfinite(raw):
        raw = 0.0
    return float(max(abs(raw), float(adapt_cfg.joint_energy_norm_floor)))


def _joint_gain_payload(
    *,
    current_metrics: Mapping[str, Any],
    next_metrics: Mapping[str, Any],
    energy_denominator: float,
    adapt_cfg: CheckpointLocalAdaptConfig,
) -> dict[str, Any]:
    fidelity_gain = float(next_metrics["fidelity_exact"] - current_metrics["fidelity_exact"])
    site_error_gain = float(
        current_metrics["site_occupations_abs_error_max"] - next_metrics["site_occupations_abs_error_max"]
    )
    abs_energy_gain = float(
        current_metrics["abs_energy_total_error"] - next_metrics["abs_energy_total_error"]
    )
    normalized_energy_gain = float(abs_energy_gain / float(energy_denominator))
    joint_metric_gain = float(
        fidelity_gain
        + float(adapt_cfg.joint_site_weight) * site_error_gain
        + float(adapt_cfg.joint_energy_weight) * normalized_energy_gain
    )
    return {
        "fidelity_exact": float(fidelity_gain),
        "site_occupations_abs_error_max": float(site_error_gain),
        "abs_energy_total_error": float(abs_energy_gain),
        "normalized_energy_gain": float(normalized_energy_gain),
        "joint_metric_gain": float(joint_metric_gain),
    }


def _candidate_ranking_summary(rows: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cap = max(1, int(limit))
    return [
        {
            "label": str(row["label"]),
            "candidate_pool_index": int(row["candidate_pool_index"]),
            "position_id": int(row.get("position_id", -1)),
            "gradient_l2": float(row["gradient_l2"]),
            "gradient_max_abs": float(row["gradient_max_abs"]),
        }
        for row in list(rows)[:cap]
    ]


def _joint_ranking_summary(rows: Sequence[Mapping[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    cap = max(1, int(limit))
    return [
        {
            "label": str(row["label"]),
            "candidate_pool_index": int(row["candidate_pool_index"]),
            "position_id": int(row.get("position_id", -1)),
            "joint_metric_gain": float(row["joint_metric_gain"]),
            "fidelity_gain": float(row["joint_gain_components"]["fidelity_exact"]),
            "site_gain": float(row["joint_gain_components"]["site_occupations_abs_error_max"]),
            "normalized_energy_gain": float(row["joint_gain_components"]["normalized_energy_gain"]),
            "fit_objective": str(row["fit_objective"]),
        }
        for row in list(rows)[:cap]
    ]


def _phase3_joint_candidate_records(
    *,
    base_snapshot: Mapping[str, Any],
    current_metrics: Mapping[str, Any],
    ranked_candidates: Sequence[Mapping[str, Any]],
    strategy_fit_cfg: FrozenScaffoldExactFitConfig,
    fit_objective: str,
    adapt_cfg: CheckpointLocalAdaptConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    energy_denominator = _joint_energy_denominator(base_snapshot=base_snapshot, adapt_cfg=adapt_cfg)
    records: list[dict[str, Any]] = []
    rejected_nonpositive_fidelity = 0
    for selected in list(ranked_candidates)[: max(1, int(adapt_cfg.candidate_rank_limit))]:
        aug_payload = dict(selected["aug_payload"])
        theta_start = np.asarray(aug_payload["theta_aug"], dtype=float).reshape(-1)
        grad_arr = np.asarray(selected["gradient_components"], dtype=float).reshape(-1)
        grad_norm = float(np.linalg.norm(grad_arr))
        if grad_arr.size > 0 and grad_norm > 0.0 and float(adapt_cfg.probe_scale) > 0.0:
            theta_start[np.asarray(aug_payload["new_runtime_indices"], dtype=int)] = (
                float(adapt_cfg.probe_scale) * (grad_arr / grad_norm)
            )
        aug_snapshot = _build_snapshot_with_scaffold(
            base_snapshot,
            terms=aug_payload["aug_terms"],
            layout=aug_payload["aug_layout"],
            executor=aug_payload["aug_executor"],
            theta_runtime=theta_start,
        )
        fit_payload = fit_checkpoint_snapshot(aug_snapshot, cfg=strategy_fit_cfg)
        fit_row = _objective_payload(fit_payload, objective=str(fit_objective))
        next_metrics = dict(fit_row["best_metrics"])
        joint_gain_components = _joint_gain_payload(
            current_metrics=current_metrics,
            next_metrics=next_metrics,
            energy_denominator=float(energy_denominator),
            adapt_cfg=adapt_cfg,
        )
        if float(joint_gain_components["fidelity_exact"]) <= 0.0:
            rejected_nonpositive_fidelity += 1
            continue
        records.append(
            {
                **dict(selected),
                "fit_objective": str(fit_objective),
                "fit_payload": fit_payload,
                "next_metrics": dict(next_metrics),
                "joint_gain_components": dict(joint_gain_components),
                "joint_metric_gain": float(joint_gain_components["joint_metric_gain"]),
            }
        )
    return records, {
        "energy_denominator_mode": "full_run_exact_span",
        "energy_denominator_value": float(energy_denominator),
        "smoothness_applied": False,
        "rejected_nonpositive_fidelity": int(rejected_nonpositive_fidelity),
    }


def adapt_checkpoint_snapshot_with_state(
    base_snapshot: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    adapt_cfg: CheckpointLocalAdaptConfig,
    fit_cfg: FrozenScaffoldExactFitConfig,
) -> CheckpointLocalAdaptRuntimeResult:
    strategy_key = str(adapt_cfg.strategy).strip().lower()
    if strategy_key not in {"gradient_local_v1", "phase3_joint_rescue_v1"}:
        raise ValueError(f"Unknown checkpoint-local ADAPT strategy {adapt_cfg.strategy!r}.")
    current_terms = list(base_snapshot["terms"])
    current_layout = base_snapshot["layout"]
    current_executor = base_snapshot["executor"]
    current_theta = np.asarray(base_snapshot["theta_runtime"], dtype=float).reshape(-1)
    strategy_fit_cfg = _strategy_fit_cfg(adapt_cfg=adapt_cfg, fit_cfg=fit_cfg, base_snapshot=base_snapshot)
    fit_objective = _strategy_fit_objective(adapt_cfg=adapt_cfg)
    initial_live_snapshot = _build_snapshot_with_scaffold(
        base_snapshot,
        terms=current_terms,
        layout=current_layout,
        executor=current_executor,
        theta_runtime=current_theta,
    )
    initial_live_fit = fit_checkpoint_snapshot(initial_live_snapshot, cfg=strategy_fit_cfg)
    objective_row = _objective_payload(initial_live_fit, objective=str(fit_objective))
    current_theta = np.asarray(objective_row["best_metrics"]["theta_runtime"], dtype=float).reshape(-1)
    working_snapshot = _build_snapshot_with_scaffold(
        base_snapshot,
        terms=current_terms,
        layout=current_layout,
        executor=current_executor,
        theta_runtime=current_theta,
    )
    current_metrics = dict(objective_row["best_metrics"])
    history: list[dict[str, Any]] = []
    stop_reason = "max_steps_reached"
    plateau_hits = 0
    pool_terms, pool_meta = resolve_candidate_pool_terms(bundle, pool_mode=str(adapt_cfg.pool_mode))
    available_initial = available_candidate_terms(current_terms, pool_terms)
    joint_rescue_meta: dict[str, Any] = {
        "enabled": bool(strategy_key == "phase3_joint_rescue_v1"),
        "joint_opt_mode": str(adapt_cfg.joint_opt_mode),
        "energy_denominator_mode": "full_run_exact_span",
        "energy_denominator_value": float(_joint_energy_denominator(base_snapshot=base_snapshot, adapt_cfg=adapt_cfg)),
        "smoothness_applied": False,
        "positions_considered": ("all_insertions" if strategy_key == "phase3_joint_rescue_v1" else "append_only"),
    }

    if float(current_metrics["fidelity_exact"]) >= float(adapt_cfg.target_fidelity):
        stop_reason = "target_fidelity_reached_initial_refit"
    else:
        for step_index in range(1, int(adapt_cfg.max_steps) + 1):
            available_terms = available_candidate_terms(current_terms, pool_terms)
            if not available_terms:
                stop_reason = "candidate_pool_exhausted"
                break
            ranked = _score_candidates(
                base_snapshot=base_snapshot,
                current_terms=current_terms,
                current_layout=current_layout,
                current_theta=current_theta,
                available_terms=available_terms,
                rank_limit=int(adapt_cfg.candidate_rank_limit),
                position_ids=(
                    tuple(range(int(len(current_terms)) + 1))
                    if strategy_key == "phase3_joint_rescue_v1"
                    else None
                ),
            )
            if not ranked:
                stop_reason = "candidate_pool_exhausted"
                break
            if float(ranked[0]["gradient_l2"]) < float(adapt_cfg.gradient_threshold):
                stop_reason = "gradient_below_threshold"
                break
            if strategy_key == "phase3_joint_rescue_v1":
                evaluated_rows, evaluated_meta = _phase3_joint_candidate_records(
                    base_snapshot=base_snapshot,
                    current_metrics=current_metrics,
                    ranked_candidates=ranked,
                    strategy_fit_cfg=strategy_fit_cfg,
                    fit_objective=str(fit_objective),
                    adapt_cfg=adapt_cfg,
                )
                selected, ranking_meta = rank_candidates_by_gain(
                    records=evaluated_rows,
                    gain_key="joint_metric_gain",
                    max_candidates=int(adapt_cfg.candidate_rank_limit),
                    min_gain=float(adapt_cfg.joint_min_gain),
                )
                joint_rescue_meta.update(dict(evaluated_meta))
                if selected is None:
                    stop_reason = "joint_gain_below_threshold"
                    history.append(
                        {
                            "step_index": int(step_index),
                            "candidate_ranking_top": _candidate_ranking_summary(
                                ranked, limit=int(adapt_cfg.candidate_rank_limit)
                            ),
                            "joint_ranking_top": _joint_ranking_summary(
                                ranking_meta.get("ranked", ()),
                                limit=int(adapt_cfg.candidate_rank_limit),
                            ),
                            "ranking_reason": str(ranking_meta.get("reason", "insufficient_gain")),
                            "pre_metrics": dict(current_metrics),
                        }
                    )
                    break
                aug_payload = dict(selected["aug_payload"])
                next_metrics = dict(selected["next_metrics"])
                history.append(
                    {
                        "step_index": int(step_index),
                        "selected_label": str(selected["label"]),
                        "selected_pool_index": int(selected["candidate_pool_index"]),
                        "selected_position_id": int(selected.get("position_id", -1)),
                        "selected_gradient_l2": float(selected["gradient_l2"]),
                        "selected_gradient_max_abs": float(selected["gradient_max_abs"]),
                        "selected_gradient_components": [float(x) for x in selected["gradient_components"]],
                        "selected_fit_objective": str(selected["fit_objective"]),
                        "selected_joint_metric_gain": float(selected["joint_metric_gain"]),
                        "selected_joint_gain_components": dict(selected["joint_gain_components"]),
                        "candidate_ranking_top": _candidate_ranking_summary(
                            ranked, limit=int(adapt_cfg.candidate_rank_limit)
                        ),
                        "joint_ranking_top": _joint_ranking_summary(
                            ranking_meta.get("ranked", ()),
                            limit=int(adapt_cfg.candidate_rank_limit),
                        ),
                        "logical_block_count_before": int(current_layout.logical_parameter_count),
                        "logical_block_count_after": int(aug_payload["aug_layout"].logical_parameter_count),
                        "runtime_parameter_count_before": int(current_layout.runtime_parameter_count),
                        "runtime_parameter_count_after": int(aug_payload["aug_layout"].runtime_parameter_count),
                        "pre_metrics": dict(current_metrics),
                        "post_metrics": dict(next_metrics),
                        "delta_vs_previous": {
                            "fidelity_exact": float(
                                selected["joint_gain_components"]["fidelity_exact"]
                            ),
                            "abs_energy_total_error": float(
                                selected["joint_gain_components"]["abs_energy_total_error"]
                            ),
                            "site_occupations_abs_error_max": float(
                                selected["joint_gain_components"]["site_occupations_abs_error_max"]
                            ),
                            "normalized_energy_gain": float(
                                selected["joint_gain_components"]["normalized_energy_gain"]
                            ),
                            "joint_metric_gain": float(
                                selected["joint_gain_components"]["joint_metric_gain"]
                            ),
                        },
                    }
                )
                current_terms = list(aug_payload["aug_terms"])
                current_layout = aug_payload["aug_layout"]
                current_executor = aug_payload["aug_executor"]
                current_theta = np.asarray(next_metrics["theta_runtime"], dtype=float).reshape(-1)
                working_snapshot = _build_snapshot_with_scaffold(
                    base_snapshot,
                    terms=current_terms,
                    layout=current_layout,
                    executor=current_executor,
                    theta_runtime=current_theta,
                )
                current_metrics = dict(next_metrics)
                if float(current_metrics["fidelity_exact"]) >= float(adapt_cfg.target_fidelity):
                    stop_reason = "target_fidelity_reached"
                    break
                continue

            selected = ranked[0]
            aug_payload = dict(selected["aug_payload"])
            theta_start = np.asarray(aug_payload["theta_aug"], dtype=float).reshape(-1)
            grad_arr = np.asarray(selected["gradient_components"], dtype=float).reshape(-1)
            grad_norm = float(np.linalg.norm(grad_arr))
            if grad_arr.size > 0 and grad_norm > 0.0 and float(adapt_cfg.probe_scale) > 0.0:
                theta_start[np.asarray(aug_payload["new_runtime_indices"], dtype=int)] = (
                    float(adapt_cfg.probe_scale) * (grad_arr / grad_norm)
                )
            aug_snapshot = _build_snapshot_with_scaffold(
                base_snapshot,
                terms=aug_payload["aug_terms"],
                layout=aug_payload["aug_layout"],
                executor=aug_payload["aug_executor"],
                theta_runtime=theta_start,
            )
            fit_payload = fit_checkpoint_snapshot(aug_snapshot, cfg=strategy_fit_cfg)
            fit_row = _objective_payload(fit_payload, objective=str(fit_objective))
            next_metrics = dict(fit_row["best_metrics"])
            fidelity_gain = float(next_metrics["fidelity_exact"] - current_metrics["fidelity_exact"])
            history.append(
                {
                    "step_index": int(step_index),
                    "selected_label": str(selected["label"]),
                    "selected_pool_index": int(selected["candidate_pool_index"]),
                    "selected_position_id": int(selected.get("position_id", -1)),
                    "selected_gradient_l2": float(selected["gradient_l2"]),
                    "selected_gradient_max_abs": float(selected["gradient_max_abs"]),
                    "selected_gradient_components": [float(x) for x in selected["gradient_components"]],
                    "candidate_ranking_top": [
                        {
                            "label": str(row["label"]),
                            "candidate_pool_index": int(row["candidate_pool_index"]),
                            "position_id": int(row.get("position_id", -1)),
                            "gradient_l2": float(row["gradient_l2"]),
                            "gradient_max_abs": float(row["gradient_max_abs"]),
                        }
                        for row in ranked[: max(1, int(adapt_cfg.candidate_rank_limit))]
                    ],
                    "logical_block_count_before": int(current_layout.logical_parameter_count),
                    "logical_block_count_after": int(aug_payload["aug_layout"].logical_parameter_count),
                    "runtime_parameter_count_before": int(current_layout.runtime_parameter_count),
                    "runtime_parameter_count_after": int(aug_payload["aug_layout"].runtime_parameter_count),
                    "pre_metrics": dict(current_metrics),
                    "post_metrics": dict(next_metrics),
                    "delta_vs_previous": {
                        "fidelity_exact": float(fidelity_gain),
                        "abs_energy_total_error": float(
                            current_metrics["abs_energy_total_error"] - next_metrics["abs_energy_total_error"]
                        ),
                        "site_occupations_abs_error_max": float(
                            current_metrics["site_occupations_abs_error_max"]
                            - next_metrics["site_occupations_abs_error_max"]
                        ),
                    },
                }
            )
            if fidelity_gain <= 1.0e-12:
                stop_reason = "no_fidelity_gain_after_append"
                break
            current_terms = list(aug_payload["aug_terms"])
            current_layout = aug_payload["aug_layout"]
            current_executor = aug_payload["aug_executor"]
            current_theta = np.asarray(next_metrics["theta_runtime"], dtype=float).reshape(-1)
            working_snapshot = _build_snapshot_with_scaffold(
                base_snapshot,
                terms=current_terms,
                layout=current_layout,
                executor=current_executor,
                theta_runtime=current_theta,
            )
            current_metrics = dict(next_metrics)
            if float(current_metrics["fidelity_exact"]) >= float(adapt_cfg.target_fidelity):
                stop_reason = "target_fidelity_reached"
                break
            if fidelity_gain < float(adapt_cfg.min_fidelity_gain):
                plateau_hits += 1
            else:
                plateau_hits = 0
            if plateau_hits >= int(adapt_cfg.plateau_patience):
                stop_reason = "fidelity_plateau"
                break

    payload = {
        "checkpoint_index": int(base_snapshot["checkpoint_index"]),
        "time": float(base_snapshot["time"]),
        "physical_time": float(base_snapshot["physical_time"]),
        "pool_meta": dict(pool_meta),
        "initial_available_candidate_count": int(len(available_initial)),
        "initial_live_metrics": dict(initial_live_fit["current_metrics"]),
        "initial_refit_metrics": dict(objective_row["best_metrics"]),
        "strategy": str(adapt_cfg.strategy),
        "fit_objective_used": str(fit_objective),
        "effective_fit_config": _to_jsonable(asdict(strategy_fit_cfg)),
        "joint_rescue": _to_jsonable(joint_rescue_meta),
        "initial_logical_block_count": int(base_snapshot["logical_block_count"]),
        "initial_runtime_parameter_count": int(base_snapshot["runtime_parameter_count"]),
        "stop_reason": str(stop_reason),
        "history": history,
        "operators_added": int(max(0, len(current_terms) - len(base_snapshot["terms"]))),
        "final_metrics": dict(current_metrics),
        "final_scaffold_labels": _current_scaffold_labels(current_terms),
        "final_source_labels": sorted(_current_source_labels(current_terms)),
        "final_logical_block_count": int(current_layout.logical_parameter_count),
        "final_runtime_parameter_count": int(current_layout.runtime_parameter_count),
        "final_theta_runtime": [float(x) for x in np.asarray(current_theta, dtype=float).tolist()],
        "requested_target_fidelity": float(adapt_cfg.target_fidelity),
        "objective": str(adapt_cfg.objective),
    }
    state = CheckpointLocalAdaptRuntimeState(
        terms=tuple(current_terms),
        layout=current_layout,
        executor=current_executor,
        theta_runtime=np.asarray(current_theta, dtype=float).reshape(-1).copy(),
        metrics=dict(current_metrics),
        scaffold_labels=tuple(_current_scaffold_labels(current_terms)),
        source_labels=tuple(sorted(_current_source_labels(current_terms))),
    )
    return CheckpointLocalAdaptRuntimeResult(payload=payload, state=state)


def adapt_checkpoint_snapshot(
    base_snapshot: Mapping[str, Any],
    *,
    bundle: Mapping[str, Any],
    adapt_cfg: CheckpointLocalAdaptConfig,
    fit_cfg: FrozenScaffoldExactFitConfig,
) -> dict[str, Any]:
    return adapt_checkpoint_snapshot_with_state(
        base_snapshot,
        bundle=bundle,
        adapt_cfg=adapt_cfg,
        fit_cfg=fit_cfg,
    ).payload


def checkpoint_local_adapt_config_from_args(args: argparse.Namespace) -> CheckpointLocalAdaptConfig:
    return CheckpointLocalAdaptConfig(
        strategy=str(getattr(args, "checkpoint_adapt_strategy", "gradient_local_v1")),
        objective=str(getattr(args, "checkpoint_adapt_objective", "fidelity_first")),
        pool_mode=str(getattr(args, "checkpoint_adapt_pool_mode", "family_pool")),
        target_fidelity=float(getattr(args, "checkpoint_adapt_target_fidelity", 0.99)),
        max_steps=int(getattr(args, "checkpoint_adapt_max_steps", 8)),
        gradient_threshold=float(getattr(args, "checkpoint_adapt_gradient_threshold", 1.0e-6)),
        probe_scale=float(getattr(args, "checkpoint_adapt_probe_scale", 0.15)),
        min_fidelity_gain=float(getattr(args, "checkpoint_adapt_min_fidelity_gain", 1.0e-4)),
        plateau_patience=int(getattr(args, "checkpoint_adapt_plateau_patience", 2)),
        candidate_rank_limit=int(getattr(args, "checkpoint_adapt_candidate_rank_limit", 8)),
        joint_site_weight=float(getattr(args, "checkpoint_adapt_joint_site_weight", 1.0)),
        joint_energy_weight=float(getattr(args, "checkpoint_adapt_joint_energy_weight", 1.0)),
        joint_energy_norm_floor=float(getattr(args, "checkpoint_adapt_joint_energy_norm_floor", 1.0e-8)),
        joint_min_gain=float(getattr(args, "checkpoint_adapt_joint_min_gain", 1.0e-6)),
        joint_opt_mode=str(getattr(args, "checkpoint_adapt_joint_opt_mode", "fidelity_fit_joint_rank")),
    )


def frozen_fit_config_from_checkpoint_adapt_args(
    args: argparse.Namespace,
    *,
    adapt_cfg: CheckpointLocalAdaptConfig | None = None,
) -> FrozenScaffoldExactFitConfig:
    resolved_adapt_cfg = adapt_cfg or checkpoint_local_adapt_config_from_args(args)
    return FrozenScaffoldExactFitConfig(
        objectives=(str(resolved_adapt_cfg.objective),),
        method=str(getattr(args, "fit_method", "Powell")),
        maxiter=int(getattr(args, "fit_maxiter", 400)),
        restarts=int(getattr(args, "fit_restarts", 4)),
        seed=int(getattr(args, "fit_seed", 7)),
        initial_sigma=float(getattr(args, "fit_initial_sigma", 0.15)),
        balanced_energy_weight=float(getattr(args, "fit_balanced_energy_weight", 1.0)),
        balanced_site_weight=float(getattr(args, "fit_balanced_site_weight", 1.0)),
    )


def add_checkpoint_local_adapt_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--checkpoint-adapt-strategy",
        choices=["gradient_local_v1", "phase3_joint_rescue_v1"],
        default="gradient_local_v1",
        help="gradient_local_v1 keeps the current append/refit path; phase3_joint_rescue_v1 widens to insertion-aware shortlist + joint ranking.",
    )
    parser.add_argument(
        "--checkpoint-adapt-objective",
        default="fidelity_first",
        help="Checkpoint-local reoptimization objective: fidelity_first, energy_only, site_only, balanced.",
    )
    parser.add_argument(
        "--checkpoint-adapt-pool-mode",
        choices=["family_pool", "full_meta"],
        default="family_pool",
        help="Stage A uses the replay/controller family pool; Stage B widens to the HH full_meta analogue.",
    )
    parser.add_argument("--checkpoint-adapt-target-fidelity", type=float, default=0.99)
    parser.add_argument("--checkpoint-adapt-max-steps", type=int, default=8)
    parser.add_argument("--checkpoint-adapt-gradient-threshold", type=float, default=1.0e-6)
    parser.add_argument("--checkpoint-adapt-probe-scale", type=float, default=0.15)
    parser.add_argument("--checkpoint-adapt-min-fidelity-gain", type=float, default=1.0e-4)
    parser.add_argument("--checkpoint-adapt-plateau-patience", type=int, default=2)
    parser.add_argument("--checkpoint-adapt-candidate-rank-limit", type=int, default=8)
    parser.add_argument("--checkpoint-adapt-joint-site-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-adapt-joint-energy-weight", type=float, default=1.0)
    parser.add_argument("--checkpoint-adapt-joint-energy-norm-floor", type=float, default=1.0e-8)
    parser.add_argument("--checkpoint-adapt-joint-min-gain", type=float, default=1.0e-6)
    parser.add_argument(
        "--checkpoint-adapt-joint-opt-mode",
        choices=["fidelity_fit_joint_rank", "joint_fit_joint_rank"],
        default="fidelity_fit_joint_rank",
        help="Use fidelity/objective-local reopt then joint ranking, or balanced joint reopt then joint ranking.",
    )
    parser.add_argument("--fit-method", type=str, default="Powell")
    parser.add_argument("--fit-maxiter", type=int, default=400)
    parser.add_argument("--fit-restarts", type=int, default=4)
    parser.add_argument("--fit-seed", type=int, default=7)
    parser.add_argument("--fit-initial-sigma", type=float, default=0.15)
    parser.add_argument("--fit-balanced-energy-weight", type=float, default=1.0)
    parser.add_argument("--fit-balanced-site-weight", type=float, default=1.0)


def run_checkpoint_local_adapt_from_args(args: argparse.Namespace) -> dict[str, Any]:
    checkpoints = sorted(
        {int(x) for x in _parse_int_tuple(getattr(args, "checkpoint_adapt_checkpoints", None))}
    )
    if not checkpoints:
        raise ValueError("--checkpoint-adapt-checkpoints must be non-empty")
    force_stay_checkpoints = tuple(
        int(x) for x in _parse_int_tuple(getattr(args, "force_stay_checkpoints", None))
    )
    adapt_cfg = checkpoint_local_adapt_config_from_args(args)
    fit_cfg = frozen_fit_config_from_checkpoint_adapt_args(args, adapt_cfg=adapt_cfg)
    exact_reference_cache: dict[str, object] = {}
    bootstrap_bundle = build_controller_bundle_from_args(
        args,
        exact_reference_cache=exact_reference_cache,
    )
    results: list[dict[str, Any]] = []
    for checkpoint_index in checkpoints:
        snapshot, bundle = capture_checkpoint_snapshot_from_args(
            args,
            checkpoint_index=int(checkpoint_index),
            force_stay_checkpoints=force_stay_checkpoints,
            exact_reference_cache=exact_reference_cache,
        )
        results.append(
            adapt_checkpoint_snapshot(
                snapshot,
                bundle=bundle,
                adapt_cfg=adapt_cfg,
                fit_cfg=fit_cfg,
            )
        )
    output_json = Path(args.output_json).expanduser().resolve()
    payload = {
        "pipeline": "hh_checkpoint_local_adapt_v1",
        "run_tag": str(args.run_tag),
        "artifact_json": str(Path(args.artifact_json).expanduser().resolve()),
        "output_json": str(output_json),
        "loader_mode": str(args.loader_mode),
        "checkpoint_adapt_checkpoints": [int(x) for x in checkpoints],
        "requested_force_stay_checkpoints": [int(x) for x in force_stay_checkpoints],
        "checkpoint_adapt_config": _to_jsonable(asdict(adapt_cfg)),
        "fit_config": _to_jsonable(asdict(fit_cfg)),
        "strategy_fit_config_template": _to_jsonable(asdict(_strategy_fit_cfg(adapt_cfg=adapt_cfg, fit_cfg=fit_cfg))),
        "effective_strategy_fit_objective": str(_strategy_fit_objective(adapt_cfg=adapt_cfg)),
        "controller_config": _to_jsonable(bootstrap_bundle["cfg"]),
        "drive_config": _to_jsonable(bootstrap_bundle["drive_config"]),
        "oracle_config": _to_jsonable(bootstrap_bundle["oracle_config"]),
        "results": _to_jsonable(results),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = build_realtime_parser()
    parser.description = (
        "Checkpoint-local ADAPT-style scaffold growth against the exact HH target state."
    )
    parser.add_argument("--checkpoint-adapt-checkpoints", required=True)
    add_checkpoint_local_adapt_config_args(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_checkpoint_local_adapt_from_args(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
