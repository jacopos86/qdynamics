"""Payload helpers for static-ADAPT prune/Schur diagnostics.

This module does not own prune acceptance. Schur surrogate values here are
nomination and telemetry surfaces only; remove-refit energy safety remains the
deletion authority in the pruning engine.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Sequence

from pipelines.scaffold.hh_continuation_pruning import (
    PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    PruneConfig,
)

__all__ = [
    "_compact_prune_schur_rows",
    "_inactive_prune_schur_nomination_payload",
    "_prune_authority_telemetry",
    "_prune_nomination_sources",
    "_prune_schur_nomination_gate_threshold",
    "_update_prune_schur_gate_payload",
]


def _prune_authority_telemetry(
    *,
    prune_cfg: PruneConfig,
    algebraic_lane_policy_active: bool,
    amplitude_effective_required: bool,
    typed_compensator_active: bool | None = None,
) -> dict[str, Any]:
    recoverability_policy = bool(
        str(prune_cfg.policy) == PRUNE_POLICY_RECOVERABILITY_LADDER_V1
    )
    typed_active = (
        bool(algebraic_lane_policy_active and recoverability_policy)
        if typed_compensator_active is None
        else bool(typed_compensator_active)
    )
    amplitude_effective_required = False
    amplitude_authority = "diagnostic_only"
    return {
        "screen_authority": "nomination_only",
        "deletion_authority": "remove_refit_energy_safety",
        "surrogate_used_for_acceptance": False,
        "amplitude_witness_config_required": bool(prune_cfg.amplitude_witness_required),
        "amplitude_witness_effective_required": False,
        "amplitude_witness_authority": str(amplitude_authority),
        "nomination_lanes": {
            "amplitude_activity": {
                "active": True,
                "authority": "nomination_only",
            },
            "recoverability_broad_rank": {
                "active": recoverability_policy,
                "authority": "nomination_only",
            },
            "schur_surrogate": {
                "active": bool(prune_cfg.surrogate_enabled and recoverability_policy),
                "authority": "nomination_only",
                "reason": (
                    "exact_current_hessian_schur_nomination"
                    if bool(prune_cfg.surrogate_enabled and recoverability_policy)
                    else "requires_recoverability_ladder_policy_and_surrogate_enabled"
                ),
            },
            "cached_tangent_fubini": {
                "active": False,
                "authority": "nomination_only",
                "reason": "no_valid_tangent_fubini_cache_wired_in_this_slice",
            },
        },
        "compensator_window_authority": {
            "typed_compensator_window": {
                "active": bool(typed_active),
                "authority": "window_selection_and_escalation_only",
                "reason": (
                    "typed compensator windows define remove-refit rung breadth, "
                    "not deletion-target nomination lanes"
                ),
            }
        },
    }


def _prune_nomination_sources(
    *,
    prune_cfg: PruneConfig,
    algebraic_lane_policy_active: bool,
    indices: Sequence[int],
    labels_now: Sequence[str],
    amplitude_effective_required: bool,
    typed_compensator_active: bool | None = None,
    lane_membership: Mapping[str, Sequence[int] | set[int]] | None = None,
) -> list[dict[str, Any]]:
    authority = _prune_authority_telemetry(
        prune_cfg=prune_cfg,
        algebraic_lane_policy_active=bool(algebraic_lane_policy_active),
        amplitude_effective_required=bool(amplitude_effective_required),
        typed_compensator_active=typed_compensator_active,
    )
    active_lanes = {
        str(key): payload
        for key, payload in authority.get("nomination_lanes", {}).items()
        if isinstance(payload, Mapping)
        and bool(payload.get("active", False))
        and str(key) != "typed_compensator_window"
    }
    membership = {
        str(key): {int(x) for x in vals}
        for key, vals in (lane_membership or {}).items()
    }
    out: list[dict[str, Any]] = []
    for idx in indices:
        idx_i = int(idx)
        lanes = [
            str(key)
            for key in active_lanes
            if idx_i in membership.get(str(key), set())
        ]
        out.append(
            {
                "index": int(idx),
                "label": str(labels_now[int(idx)])
                if 0 <= int(idx) < len(labels_now)
                else "",
                "lanes": [str(x) for x in lanes],
                "authority": "nomination_only",
            }
        )
    return out


def _prune_schur_nomination_gate_threshold(
    *,
    prune_cfg: PruneConfig,
    max_regression_effective: float,
) -> float | None:
    if not bool(prune_cfg.surrogate_nomination_gate_enabled):
        return None
    if str(prune_cfg.policy) != PRUNE_POLICY_RECOVERABILITY_LADDER_V1:
        return None
    factor = float(prune_cfg.surrogate_nomination_gate_factor)
    if not math.isfinite(factor) or factor < 0.0:
        return None
    threshold = float(max(0.0, float(max_regression_effective)) * factor)
    return float(threshold)


def _update_prune_schur_gate_payload(
    summary: MutableMapping[str, Any],
    *,
    prune_cfg: PruneConfig,
    threshold: float | None,
    pre_gate_candidate_count: int,
    post_gate_candidate_count: int,
) -> None:
    payload = summary.get("schur_surrogate_nomination")
    if not isinstance(payload, dict):
        return
    payload["nomination_gate_enabled"] = bool(
        prune_cfg.surrogate_nomination_gate_enabled
    )
    payload["nomination_gate_factor"] = float(
        prune_cfg.surrogate_nomination_gate_factor
    )
    payload["nomination_gate_threshold"] = (
        None if threshold is None else float(threshold)
    )
    payload["exact_trial_cap"] = int(prune_cfg.surrogate_exact_trial_cap)
    payload["pre_gate_candidate_count"] = int(pre_gate_candidate_count)
    payload["post_gate_candidate_count"] = int(post_gate_candidate_count)
    payload["used_for_acceptance"] = False
    summary["schur_surrogate_nomination"] = dict(payload)


def _inactive_prune_schur_nomination_payload(
    *,
    prune_cfg: PruneConfig,
    selected_parameterization_mode: str,
    reason: str,
    logical_parameter_count: int = 0,
    runtime_parameter_count: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": "static_prune_schur_nomination_v1",
        "enabled": bool(prune_cfg.surrogate_enabled),
        "active": False,
        "reason": str(reason),
        "authority": "rank_window_diag_only",
        "used_for_nomination": False,
        "used_for_acceptance": False,
        "score_count": 0,
        "rows": [],
        "nomination_gate_enabled": bool(prune_cfg.surrogate_nomination_gate_enabled),
        "nomination_gate_factor": float(prune_cfg.surrogate_nomination_gate_factor),
        "nomination_gate_threshold": None,
        "exact_trial_cap": int(prune_cfg.surrogate_exact_trial_cap),
        "post_gate_candidate_count": 0,
        "selected_parameterization_mode": str(selected_parameterization_mode),
        "logical_parameter_count": int(logical_parameter_count),
        "runtime_parameter_count": (
            None if runtime_parameter_count is None else int(runtime_parameter_count)
        ),
        "local_window_size": int(prune_cfg.local_window_size),
        "recovery_trust_radius": float(prune_cfg.surrogate_recovery_trust_radius),
        "ridge": float(prune_cfg.surrogate_ridge),
        "monotonicity_tol": float(prune_cfg.surrogate_monotonicity_tol),
        "schur_nomination_route": str(prune_cfg.schur_nomination_route),
        "metric_schur_mu": float(prune_cfg.metric_schur_mu),
        "metric_schur_solve_mode": str(prune_cfg.metric_schur_solve_mode),
        "metric_schur_cost_weighting": str(prune_cfg.metric_schur_cost_weighting),
        "hessian_shape": [],
    }


def _compact_prune_schur_rows(
    surrogate_scores: Mapping[int, Mapping[str, Any]],
    *,
    max_rows: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in sorted(
        surrogate_scores.items(),
        key=lambda item: (
            float(item[1].get("score", item[1].get("schur_min", float("inf")))),
            int(item[0]),
        ),
    )[: max(0, int(max_rows))]:
        ladder_rows = [
            dict(x) for x in row.get("schur_rows", []) if isinstance(x, Mapping)
        ]
        rows.append(
            {
                "index": int(idx),
                "label": str(row.get("label", "")),
                "score": float(row.get("score", row.get("schur_min", float("inf")))),
                "schur_min": float(row.get("schur_min", row.get("score", float("inf")))),
                "bounded_score": float(
                    row.get("bounded_score", row.get("score", float("inf")))
                ),
                "unweighted_score": float(
                    row.get("unweighted_score", row.get("score", float("inf")))
                ),
                "entry_cost_denominator": float(
                    row.get("entry_cost_denominator", 1.0)
                ),
                "schur_model": str(row.get("schur_model", "hessian_coupling_v1")),
                "metric_mu": float(row.get("metric_mu", 0.0)),
                "metric_schur_solve_mode": str(
                    row.get("metric_schur_solve_mode", "stationary_gw_zero_v1")
                ),
                "bounded_recovery_active": bool(
                    row.get("bounded_recovery_active", False)
                ),
                "recovery_trust_radius": float(row.get("recovery_trust_radius", 0.0)),
                "schur_health": str(row.get("schur_health", "unavailable")),
                "schur_monotone": bool(row.get("schur_monotone", False)),
                "rung_values": [
                    float(x.get("schur_value", float("inf")))
                    for x in ladder_rows
                    if x.get("schur_value") is not None
                ],
                "bounded_rung_values": [
                    float(x.get("bounded_value", x.get("schur_value", float("inf"))))
                    for x in ladder_rows
                    if x.get("bounded_value", x.get("schur_value")) is not None
                ],
                "compensation_norms": [
                    float(x.get("compensation_norm", 0.0)) for x in ladder_rows
                ],
                "window_sizes": [
                    int(len(x.get("window_indices", []))) for x in ladder_rows
                ],
                "surrogate_authority": str(
                    row.get("surrogate_authority", "rank_window_diag_only")
                ),
                "used_for_acceptance": bool(row.get("used_for_acceptance", False)),
            }
        )
    return rows
