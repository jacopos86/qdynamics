"""Runtime helpers for quarantined Route-C plateau acquisition.

This module owns Route-C runtime helpers that are not part of canonical
Route-A SNAKE.  Pure Route-C config/state helpers stay in
``plateau_acquisition.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

from pipelines.static_adapt.engine_support import _make_reduced_parameter_expander
from pipelines.static_adapt.nested_windows import (
    NestedWindowError,
    predict_active_dormant_nested_window,
    serialize_active_dormant_nested_window,
)
from pipelines.static_adapt.plateau_acquisition import (
    PlateauAcquisitionState,
    PlateauCandidateKey,
    candidate_key_from_record,
    duplicate_status,
    failed_family_backoff_status,
    plateau_score_formula,
)
from pipelines.scaffold.hh_continuation_scoring import (
    attach_route_c_plateau_acquisition_payload,
    phase3_plateau_novelty_cost_score_components,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from src.quantum.ansatz_parameterization import runtime_indices_for_logical_indices
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    energy_via_one_apply,
)

ROUTE_C_PLATEAU_EVENT_COMPACT_KEYS = (
    "schema",
    "event",
    "depth",
    "entry_source",
    "mode",
    "acquisition_score",
    "score_formula",
    "selection_depth",
    "selection_mode",
    "candidate_key",
    "candidate_label",
    "generator_id",
    "position_id",
    "duplicate_status",
    "duplicate_blocked",
    "block_reason",
    "eligible",
    "phase3_plateau_acquisition_score",
    "phase3_plateau_acquisition_score_kind",
    "phase3_plateau_score_numerator",
    "route_c_plateau_acquisition_score",
    "phase3_plateau_novelty",
    "phase3_plateau_novelty_source",
    "phase3_plateau_novelty_mode",
    "phase3_plateau_novelty_error",
    "phase3_plateau_novelty_context_indices",
    "phase3_plateau_novelty_dormant_indices",
    "phase3_plateau_novelty_context_count",
    "phase3_plateau_novelty_dormant_count",
    "phase3_plateau_novelty_F_raw",
    "phase3_plateau_novelty_F_red",
    "phase3_plateau_novelty_h_eff",
    "phase3_plateau_novelty_ridge_used",
    "N3_plat",
    "phase3_plateau_geometry_source",
    "phase3_plateau_geometry_solve_mode",
    "phase3_plateau_context_dimension",
    "phase3_plateau_rank_context",
    "phase3_plateau_rank_augmented",
    "phase3_plateau_rank_delta",
    "phase3_plateau_F_raw",
    "phase3_plateau_F_safe",
    "phase3_plateau_sigma_perp",
    "phase3_plateau_sigma_perp_lcb",
    "phase3_plateau_sigma_perp_lambda",
    "phase3_plateau_sigma_perp_lambda_lcb",
    "phase3_plateau_fractional_residual",
    "phase3_plateau_fractional_residual_lcb",
    "phase3_plateau_log_volume_gain",
    "phase3_plateau_log_volume_gain_lcb",
    "phase3_plateau_lambda_vol",
    "phase3_plateau_lambda_vol_used",
    "phase3_plateau_sigma_min",
    "phase3_plateau_nu_min",
    "phase3_plateau_volume_min",
    "phase3_plateau_failed_family_patience",
    "failed_family_backoff_status",
    "failed_family_backoff_blocked",
    "failed_family_backoff_count",
    "phase3_plateau_burden_total",
    "denominator_1_plus_K3",
    "hardware_cost_denominator",
    "active_old_pre_indices",
    "dormant_old_pre_indices",
    "context_old_pre_indices",
    "optimizer_active_post_indices",
    "trainable_logical_indices",
    "trainable_runtime_indices",
    "seed_probe",
    "seed_probe_mode",
    "seed_probe_enabled",
    "seed_probe_nfev",
    "seed_probe_base_energy",
    "seed_probe_best_energy",
    "seed_probe_best_drop",
    "seed_probe_improved",
    "seed_probe_best_index",
    "seed_probe_radius",
    "seed_probe_probe_logical_indices",
    "seed_probe_probe_runtime_indices",
    "trial_optimizer_requested",
    "trial_optimizer_effective",
    "trial_optimizer_info",
    "dormant_indices_before",
    "dormant_indices_after",
    "trial_energy",
    "trial_drop",
    "unlock_margin",
    "unlock_success",
    "unlock_status",
    "reportable_energy_before",
    "reportable_energy_after",
    "selected_abs_theta",
    "logical_index_space",
    "logical_parameter_count_after_trial",
    "promoted_logical_indices",
    "promoted_logical_abs_theta",
    "remaining_dormant_indices",
    "remaining_dormant_abs_theta",
    "window_error",
)

__all__ = [
    "ROUTE_C_PLATEAU_EVENT_COMPACT_KEYS",
    "RouteCPlateauScoringContext",
    "route_c_plateau_compact_event_payload",
    "route_c_plateau_compact_state_payload",
    "route_c_plateau_active_old_pre_indices",
    "route_c_plateau_active_dormant_novelty_payload",
    "route_c_plateau_candidate_term",
    "route_c_plateau_feature_identity_payload",
    "route_c_plateau_payload_for_record",
    "route_c_plateau_runtime_state_payload",
    "route_c_plateau_score_record",
    "route_c_plateau_sort_key",
    "route_c_zero_logical_indices",
    "run_route_c_sp_qngd_trial_optimizer",
]


def route_c_plateau_compact_state_payload(
    state: PlateauAcquisitionState,
) -> dict[str, Any]:
    """Serialize plateau state without recursively embedding dormant payloads."""

    return {
        "schema": str(state.schema),
        "active_episode": bool(state.active_episode),
        "dormant_count": int(len(state.dormant_records)),
        "dormant_logical_indices": [int(x) for x in state.dormant_logical_indices()],
        "dormant_records": [
            {
                "candidate_key": record.candidate_key.as_dict(),
                "logical_index": int(record.logical_index),
                "candidate_label": record.candidate_label,
                "generator_id": record.generator_id,
                "position_id": None if record.position_id is None else int(record.position_id),
                "admission_step": (
                    None if record.admission_step is None else int(record.admission_step)
                ),
            }
            for record in state.dormant_records
        ],
        "acquired_candidate_keys": [
            key.as_dict() for key in state.acquired_candidate_keys
        ],
        "failed_unlock_count": int(state.failed_unlock_count),
        "unlock_count": int(state.unlock_count),
        "last_event": route_c_plateau_compact_event_payload(state.last_event),
    }


def route_c_plateau_compact_event_payload(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Keep Route-C checkpoint/history telemetry bounded and non-recursive."""

    if not isinstance(payload, Mapping):
        return None
    compact: dict[str, Any] = {}
    for key in ROUTE_C_PLATEAU_EVENT_COMPACT_KEYS:
        if key in payload:
            compact[key] = payload[key]
    if isinstance(payload.get("active_dormant_nested_window"), Mapping):
        window = payload.get("active_dormant_nested_window", {})
        compact["active_dormant_nested_window"] = {
            str(k): v
            for k, v in dict(window).items()
            if k not in {"debug_records", "candidate_records", "payloads"}
        }
    if isinstance(payload.get("selected_record"), Mapping):
        compact["selected_record"] = dict(
            route_c_plateau_compact_event_payload(payload.get("selected_record")) or {}
        )
    if isinstance(payload.get("state_before"), PlateauAcquisitionState):
        compact["state_before"] = route_c_plateau_compact_state_payload(
            payload["state_before"]
        )
    elif isinstance(payload.get("state_before"), Mapping):
        raw_before = payload.get("state_before", {})
        compact["state_before"] = {
            "schema": raw_before.get("schema"),
            "active_episode": bool(raw_before.get("active_episode", False)),
            "dormant_count": raw_before.get("dormant_count"),
            "dormant_logical_indices": raw_before.get("dormant_logical_indices", []),
            "failed_unlock_count": raw_before.get("failed_unlock_count"),
            "unlock_count": raw_before.get("unlock_count"),
        }
    if isinstance(payload.get("state_after"), PlateauAcquisitionState):
        compact["state_after"] = route_c_plateau_compact_state_payload(
            payload["state_after"]
        )
    elif isinstance(payload.get("state_after"), Mapping):
        raw_after = payload.get("state_after", {})
        compact["state_after"] = {
            "schema": raw_after.get("schema"),
            "active_episode": bool(raw_after.get("active_episode", False)),
            "dormant_count": raw_after.get("dormant_count"),
            "dormant_logical_indices": raw_after.get("dormant_logical_indices", []),
            "failed_unlock_count": raw_after.get("failed_unlock_count"),
            "unlock_count": raw_after.get("unlock_count"),
        }
    return compact


def route_c_plateau_active_old_pre_indices(
    state: PlateauAcquisitionState,
    pre_parameter_count: int,
) -> list[int]:
    pre_n = int(max(0, pre_parameter_count))
    dormant = {
        int(x)
        for x in state.dormant_logical_indices()
        if 0 <= int(x) < pre_n
    }
    return [int(i) for i in range(pre_n) if int(i) not in dormant]


def route_c_plateau_runtime_state_payload(
    *,
    state: PlateauAcquisitionState,
    config: Any,
    events: Sequence[Mapping[str, Any]],
    theta_zero: float,
    seed_probe_mode: str,
    seed_probe_count: int,
    seed_probe_radius: float,
    seed_probe_enabled: bool,
    seed_probe_seed: int | None,
    reportable_energy: float | None = None,
    events_tail: int = 50,
) -> dict[str, Any]:
    compact_state = route_c_plateau_compact_state_payload(state)
    plateau_config_payload = config.as_dict()
    plateau_config_payload.update(
        {
            "seed_probe_mode": str(seed_probe_mode),
            "seed_probe_count": int(seed_probe_count),
            "seed_probe_radius": float(seed_probe_radius),
            "seed_probe_enabled": bool(seed_probe_enabled),
            "seed_probe_seed": seed_probe_seed,
        }
    )
    payload: dict[str, Any] = {
        "schema": str(compact_state.get("schema", "route_c_plateau_acquisition_v1")),
        "enabled": bool(config.enabled),
        "config": plateau_config_payload,
        "state": dict(compact_state),
        "active_episode": bool(state.active_episode),
        "dormant_count": int(len(state.dormant_records)),
        "dormant_logical_indices": [int(x) for x in state.dormant_logical_indices()],
        "failed_unlock_count": int(state.failed_unlock_count),
        "unlock_count": int(state.unlock_count),
        "events_tail": [
            dict(x)
            for x in events[-int(max(0, events_tail)):]
            if isinstance(x, Mapping)
        ],
        "theta_zero": float(theta_zero),
    }
    if reportable_energy is not None:
        payload["reportable_energy"] = float(reportable_energy)
    return payload


def route_c_plateau_feature_identity_payload(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    feat = record.get("feature") if isinstance(record, Mapping) else None
    label = None
    generator_id = None
    position_id = None
    if isinstance(feat, CandidateFeatures):
        label = feat.candidate_label
        generator_id = feat.generator_id
        position_id = feat.position_id
    if label in {None, ""}:
        label = record.get("candidate_label") or getattr(
            record.get("candidate_term"),
            "label",
            None,
        )
    if generator_id in {None, ""}:
        generator_id = record.get("generator_id")
    if position_id is None:
        position_id = record.get("position_id")
    return {
        "candidate_label": None if label in {None, ""} else str(label),
        "generator_id": None if generator_id in {None, ""} else str(generator_id),
        "position_id": None if position_id is None else int(position_id),
    }


def route_c_plateau_candidate_term(
    record: Mapping[str, Any],
    *,
    pool: Sequence[Any],
) -> Any | None:
    term = record.get("candidate_term") if isinstance(record, Mapping) else None
    if term is not None:
        return term
    feat = record.get("feature") if isinstance(record, Mapping) else None
    if not isinstance(feat, CandidateFeatures):
        return None
    try:
        pool_idx = int(feat.candidate_pool_index)
    except (TypeError, ValueError):
        return None
    if 0 <= int(pool_idx) < len(pool):
        return pool[int(pool_idx)]
    return None


@dataclass(frozen=True)
class RouteCPlateauScoringContext:
    state: PlateauAcquisitionState
    plateau_config: Any
    events: Sequence[Mapping[str, Any]]
    entry_source: str
    pool: Sequence[Any]
    scaffold_context_cache: dict[tuple[int, ...], Any]
    novelty_oracle: Any
    curvature_oracle: Any
    score_config: Any
    selected_ops: Sequence[Any]
    theta_logical_current: np.ndarray
    psi_ref: np.ndarray
    psi_current: np.ndarray
    h_compiled: CompiledPolynomialAction
    hpsi_current: np.ndarray
    pauli_action_cache: Any
    compiled_term_cache: Any


def _finite_float_or_none(value: Any) -> float | None:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    return float(value_f) if math.isfinite(value_f) else None


def route_c_plateau_active_dormant_novelty_payload(
    record: Mapping[str, Any],
    *,
    context_old_pre_indices: Sequence[int],
    dormant_old_pre_indices: Sequence[int],
    context: RouteCPlateauScoringContext,
) -> tuple[float | None, dict[str, Any], dict[str, Any]]:
    """Evaluate legacy Route-C novelty against active and dormant coordinates."""

    feat = record.get("feature") if isinstance(record, Mapping) else None
    context_indices = [int(i) for i in context_old_pre_indices]
    dormant_indices = [int(i) for i in dormant_old_pre_indices]
    base_payload: dict[str, Any] = {
        "phase3_plateau_novelty_source": "active_dormant_schur_v1",
        "phase3_plateau_novelty_context_indices": list(context_indices),
        "phase3_plateau_novelty_dormant_indices": list(dormant_indices),
        "phase3_plateau_novelty_context_count": int(len(context_indices)),
        "phase3_plateau_novelty_dormant_count": int(len(dormant_indices)),
    }
    if not isinstance(feat, CandidateFeatures):
        return None, {
            **base_payload,
            "phase3_plateau_novelty_source": "active_dormant_schur_v1_error",
            "phase3_plateau_novelty_error": "missing_candidate_feature",
        }, {}
    candidate_term = route_c_plateau_candidate_term(record, pool=context.pool)
    if candidate_term is None:
        return None, {
            **base_payload,
            "phase3_plateau_novelty_source": "active_dormant_schur_v1_error",
            "phase3_plateau_novelty_error": "missing_candidate_term",
        }, {}
    try:
        context_key = tuple(context_indices)
        scaffold_context = context.scaffold_context_cache.get(context_key)
        if scaffold_context is None:
            scaffold_context = context.novelty_oracle.prepare_scaffold_context(
                selected_ops=list(context.selected_ops),
                theta=np.asarray(context.theta_logical_current, dtype=float),
                psi_ref=np.asarray(context.psi_ref, dtype=complex),
                psi_state=np.asarray(context.psi_current, dtype=complex),
                h_compiled=context.h_compiled,
                hpsi_state=np.asarray(context.hpsi_current, dtype=complex),
                refit_window_indices=list(context_key),
                pauli_action_cache=context.pauli_action_cache,
            )
            context.scaffold_context_cache[context_key] = scaffold_context
        novelty_info = context.novelty_oracle.estimate(
            scaffold_context=scaffold_context,
            candidate_label=str(feat.candidate_label),
            candidate_term=candidate_term,
            compiled_cache=context.compiled_term_cache,
            pauli_action_cache=context.pauli_action_cache,
            novelty_eps=float(context.score_config.novelty_eps),
        )
        curvature_info = context.curvature_oracle.estimate(
            base_feature=feat,
            novelty_info=novelty_info,
            scaffold_context=scaffold_context,
            h_compiled=context.h_compiled,
            cfg=context.score_config,
            optimizer_memory=None,
        )
        novelty = _finite_float_or_none(curvature_info.get("novelty"))
        if novelty is None:
            return None, {
                **base_payload,
                "phase3_plateau_novelty_source": "active_dormant_schur_v1_error",
                "phase3_plateau_novelty_error": "nonfinite_active_dormant_novelty",
            }, {}
        scalar_payload = {
            "phase3_plateau_novelty_mode": curvature_info.get("curvature_mode"),
            "phase3_plateau_novelty_F_raw": _finite_float_or_none(
                novelty_info.get("F_raw")
            ),
            "phase3_plateau_novelty_F_red": _finite_float_or_none(
                curvature_info.get("F_red")
            ),
            "phase3_plateau_novelty_h_eff": _finite_float_or_none(
                curvature_info.get("h_eff")
            ),
            "phase3_plateau_novelty_ridge_used": _finite_float_or_none(
                curvature_info.get("ridge_used")
            ),
        }
        geometry_inputs = {
            "F_raw": novelty_info.get("F_raw"),
            "Q_window": novelty_info.get("Q_window"),
            "q_window": novelty_info.get("q_window"),
        }
        return float(min(1.0, max(0.0, novelty))), {
            **base_payload,
            **scalar_payload,
            "phase3_plateau_novelty_error": None,
        }, geometry_inputs
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return None, {
            **base_payload,
            "phase3_plateau_novelty_source": "active_dormant_schur_v1_error",
            "phase3_plateau_novelty_error": str(message[:240]),
        }, {}


def route_c_plateau_payload_for_record(
    record: Mapping[str, Any],
    *,
    pre_parameter_count: int,
    context: RouteCPlateauScoringContext,
) -> tuple[dict[str, Any], PlateauCandidateKey] | None:
    if not isinstance(record, Mapping):
        return None
    feat = record.get("feature")
    if not isinstance(feat, CandidateFeatures):
        return None
    key = candidate_key_from_record(record)
    identity_payload = route_c_plateau_feature_identity_payload(record)
    dormant_old = [
        int(i)
        for i in context.state.dormant_logical_indices()
        if 0 <= int(i) < int(pre_parameter_count)
    ]
    active_old = route_c_plateau_active_old_pre_indices(
        context.state,
        int(pre_parameter_count),
    )
    duplicate_payload = duplicate_status(
        context.state,
        key,
        duplicate_policy=str(context.plateau_config.duplicate_policy),
    )
    failed_family_payload = failed_family_backoff_status(
        context.events,
        key,
        patience=int(context.plateau_config.failed_family_patience),
    )
    window_payload: dict[str, Any] | None = None
    window_error: str | None = None
    try:
        active_dormant_window = predict_active_dormant_nested_window(
            pre_parameter_count=int(pre_parameter_count),
            position_id=int(feat.position_id),
            active_old_pre_indices=list(active_old),
            dormant_old_pre_indices=list(dormant_old),
            origin="route_c_plateau_acquisition_v1",
        )
        window_payload = serialize_active_dormant_nested_window(active_dormant_window)
    except NestedWindowError as exc:
        window_error = str(exc)
    context_old_pre_indices = (
        [int(i) for i in window_payload.get("context_old_pre_indices", [])]
        if isinstance(window_payload, Mapping)
        else [int(i) for i in list(active_old) + list(dormant_old)]
    )
    plateau_novelty, plateau_novelty_payload, plateau_geometry_inputs = (
        route_c_plateau_active_dormant_novelty_payload(
            record,
            context_old_pre_indices=list(context_old_pre_indices),
            dormant_old_pre_indices=list(dormant_old),
            context=context,
        )
    )
    score_payload = phase3_plateau_novelty_cost_score_components(
        feat,
        context.score_config,
        plateau_novelty=plateau_novelty,
        acquisition_score=context.plateau_config.acquisition_score,
        F_raw=plateau_geometry_inputs.get("F_raw"),
        Q_window=plateau_geometry_inputs.get("Q_window"),
        q_window=plateau_geometry_inputs.get("q_window"),
        lambda_vol=float(context.plateau_config.lambda_vol),
        sigma_min=float(context.plateau_config.sigma_min),
        nu_min=float(context.plateau_config.nu_min),
        volume_min=float(context.plateau_config.volume_min),
        duplicate_blocked=bool(duplicate_payload.get("blocked", False)),
        duplicate_key=key.as_dict(),
        context_indices=list(context_old_pre_indices),
        dormant_indices=list(dormant_old),
    )
    if (
        plateau_novelty is None
        and window_error is None
        and not bool(duplicate_payload.get("blocked", False))
        and not bool(failed_family_payload.get("blocked", False))
    ):
        score_payload = {
            **dict(score_payload),
            "eligible": False,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": (
                "active_dormant_novelty_error:"
                f"{plateau_novelty_payload.get('phase3_plateau_novelty_error')}"
            ),
        }
    if bool(failed_family_payload.get("blocked", False)):
        score_payload = {
            **dict(score_payload),
            "eligible": False,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": "failed_family_backoff",
        }
    if window_error is not None:
        score_payload = {
            **dict(score_payload),
            "eligible": False,
            "phase3_plateau_acquisition_score": float("-inf"),
            "route_c_plateau_acquisition_score": float("-inf"),
            "block_reason": f"active_dormant_window_error:{window_error}",
        }
    payload = {
        "schema": "route_c_plateau_acquisition_v1",
        "mode": str(context.plateau_config.mode),
        "acquisition_score": str(context.plateau_config.acquisition_score),
        "score_formula": str(
            plateau_score_formula(context.plateau_config.acquisition_score)
        ),
        "entry_source": str(context.entry_source),
        "candidate_key": key.as_dict(),
        **dict(identity_payload),
        "duplicate_status": dict(duplicate_payload),
        "duplicate_blocked": bool(duplicate_payload.get("blocked", False)),
        "failed_family_backoff_status": dict(failed_family_payload),
        "failed_family_backoff_blocked": bool(
            failed_family_payload.get("blocked", False)
        ),
        "failed_family_backoff_count": int(
            failed_family_payload.get("failed_family_count", 0)
        ),
        "phase3_plateau_failed_family_patience": int(
            context.plateau_config.failed_family_patience
        ),
        "active_old_pre_indices": [int(i) for i in active_old],
        "dormant_old_pre_indices": [int(i) for i in dormant_old],
        "context_old_pre_indices": [int(i) for i in context_old_pre_indices],
        "active_dormant_nested_window": (
            dict(window_payload) if isinstance(window_payload, Mapping) else None
        ),
        "window_error": window_error,
        "state_before": route_c_plateau_compact_state_payload(context.state),
        "dormant_indices_before": [int(i) for i in dormant_old],
        **dict(plateau_novelty_payload),
        **dict(score_payload),
    }
    return payload, key


def route_c_plateau_score_record(
    record: Mapping[str, Any],
    *,
    pre_parameter_count: int,
    context: RouteCPlateauScoringContext,
) -> dict[str, Any] | None:
    payload_key = route_c_plateau_payload_for_record(
        record,
        pre_parameter_count=int(pre_parameter_count),
        context=context,
    )
    if payload_key is None:
        return None
    payload, _key = payload_key
    feat = record.get("feature")
    if not isinstance(feat, CandidateFeatures):
        return None
    feat_scored = attach_route_c_plateau_acquisition_payload(feat, payload)
    scored = dict(record)
    scored["feature"] = feat_scored
    scored["route_c_plateau_acquisition"] = dict(payload)
    scored["phase3_plateau_acquisition_score"] = float(
        payload.get("phase3_plateau_acquisition_score", float("-inf"))
    )
    scored["route_c_plateau_acquisition_score"] = float(
        payload.get("route_c_plateau_acquisition_score", float("-inf"))
    )
    scored["plateau_acquisition_eligible"] = bool(payload.get("eligible", False))
    return scored


def route_c_zero_logical_indices(
    theta_runtime: np.ndarray,
    layout: Any,
    logical_indices: Sequence[int],
    *,
    canonicalize_runtime_theta: Callable[[np.ndarray, Any], np.ndarray],
) -> np.ndarray:
    theta_out = np.asarray(theta_runtime, dtype=float).copy()
    for logical_idx in sorted({int(i) for i in logical_indices if int(i) >= 0}):
        if int(logical_idx) >= int(getattr(layout, "logical_parameter_count", 0)):
            continue
        for runtime_idx in runtime_indices_for_logical_indices(layout, [int(logical_idx)]):
            if 0 <= int(runtime_idx) < int(theta_out.size):
                theta_out[int(runtime_idx)] = 0.0
    return np.asarray(
        canonicalize_runtime_theta(theta_out, layout),
        dtype=float,
    )


def route_c_plateau_sort_key(
    record: Mapping[str, Any],
) -> tuple[float, int, int, str]:
    payload = (
        record.get("route_c_plateau_acquisition", {})
        if isinstance(record, Mapping)
        else {}
    )
    try:
        score = float(
            record.get(
                "phase3_plateau_acquisition_score",
                payload.get("phase3_plateau_acquisition_score", float("-inf")),
            )
        )
    except (TypeError, ValueError):
        score = float("-inf")
    if not math.isfinite(score):
        score = float("-inf")
    feat = record.get("feature") if isinstance(record, Mapping) else None
    pool_index = int(
        getattr(feat, "candidate_pool_index", record.get("candidate_pool_index", -1))
    )
    position = int(getattr(feat, "position_id", record.get("position_id", -1)))
    label = str(getattr(feat, "candidate_label", record.get("candidate_label", "")))
    return (-float(score), int(pool_index), int(position), str(label))


def run_route_c_sp_qngd_trial_optimizer(
    *,
    x0: np.ndarray,
    theta_template: np.ndarray,
    active_runtime_indices: Sequence[int],
    maxiter_value: int,
    trial_qngd_maxiter: int,
    metric_floor: float,
    h_compiled: CompiledPolynomialAction,
    prepare_state_for_theta: Callable[[np.ndarray], np.ndarray],
    canonicalize_runtime_theta: Callable[[np.ndarray], np.ndarray],
    oracle_inner_objective_enabled: bool,
    analytic_noise_enabled: bool,
) -> SimpleNamespace:
    """Run the exact-state Route-C SP-QNGD plateau trial optimizer.

    The optimizer differentiates the same reduced coordinate surface used by
    the ordinary refit objective.  It is intentionally exact-state only; noisy
    or oracle inner objective surfaces are rejected by the compatibility guard.
    """

    if bool(oracle_inner_objective_enabled) or bool(analytic_noise_enabled):
        raise ValueError(
            "phase3_plateau_trial_optimizer=sp_qngd requires the exact prepared-state "
            "inner objective; noisy/oracle objective surfaces are not supported."
        )

    x = np.asarray(x0, dtype=float).reshape(-1).copy()
    theta_template_arr = np.asarray(theta_template, dtype=float).reshape(-1).copy()
    active_indices = [int(i) for i in active_runtime_indices]
    expand_active, _ = _make_reduced_parameter_expander(theta_template_arr, active_indices)
    maxiter = max(0, min(int(maxiter_value), int(trial_qngd_maxiter)))
    max_abs_step = 0.25
    max_backtracks = 10
    accept_tol = 1e-12
    tangent_eps = 1e-5
    metric_floor_value = max(1e-12, float(metric_floor))
    nfev = 0
    nit = 0
    accepted_step_count = 0
    state_probe_count = 0
    metric_eval_count = 0
    backtracks_total = 0
    energy_decrease_total = 0.0
    last_rank = 0
    last_condition: float | None = None
    last_fs_norm = 0.0
    last_l2_norm = 0.0
    last_max_abs_step = 0.0
    message = "sp_qngd_maxiter_reached"
    success = False

    def _theta_for_x(x_vec: np.ndarray) -> np.ndarray:
        theta_full = expand_active(np.asarray(x_vec, dtype=float).reshape(-1))
        return np.asarray(canonicalize_runtime_theta(theta_full), dtype=float).reshape(-1)

    def _state_for_x(x_vec: np.ndarray) -> np.ndarray:
        theta_full = _theta_for_x(np.asarray(x_vec, dtype=float))
        return np.asarray(
            prepare_state_for_theta(theta_full),
            dtype=complex,
        ).reshape(-1)

    def _exact_energy_hpsi_for_x(x_vec: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        nonlocal nfev
        psi_val = _state_for_x(np.asarray(x_vec, dtype=float))
        energy_val, hpsi_val = energy_via_one_apply(psi_val, h_compiled)
        nfev += 1
        return float(energy_val), psi_val, np.asarray(hpsi_val, dtype=complex).reshape(-1)

    def _horizontal_tangents(x_vec: np.ndarray, psi_center: np.ndarray) -> list[np.ndarray]:
        nonlocal state_probe_count
        if int(x_vec.size) == 0:
            return []
        tangents: list[np.ndarray] = []
        for k in range(int(x_vec.size)):
            step_vec = np.zeros_like(x_vec, dtype=float)
            step_vec[int(k)] = float(tangent_eps)
            psi_plus = _state_for_x(np.asarray(x_vec, dtype=float) + step_vec)
            psi_minus = _state_for_x(np.asarray(x_vec, dtype=float) - step_vec)
            state_probe_count += 2
            tangent = (psi_plus - psi_minus) / (2.0 * float(tangent_eps))
            tangent = tangent - psi_center * complex(np.vdot(psi_center, tangent))
            tangents.append(np.asarray(tangent, dtype=complex).reshape(-1))
        return tangents

    current_energy, psi_current, hpsi_current = _exact_energy_hpsi_for_x(x)
    if int(x.size) == 0:
        return SimpleNamespace(
            x=np.asarray(x, dtype=float),
            fun=float(current_energy),
            nfev=int(nfev),
            nit=0,
            success=True,
            message="sp_qngd_empty_parameter_vector",
            qngd_info={
                "optimizer": "route_c_sp_qngd_v1",
                "qngd_metric_rank_last": 0,
                "qngd_metric_condition_last": None,
                "qngd_state_probe_count": 0,
                "qngd_metric_eval_count": 0,
                "qngd_accepted_step_count": 0,
                "qngd_line_search_backtracks_total": 0,
                "qngd_energy_decrease_total": 0.0,
                "qngd_maxiter_effective": int(maxiter),
            },
        )
    if int(maxiter) <= 0:
        return SimpleNamespace(
            x=np.asarray(x, dtype=float),
            fun=float(current_energy),
            nfev=int(nfev),
            nit=0,
            success=True,
            message="sp_qngd_maxiter_zero",
            qngd_info={
                "optimizer": "route_c_sp_qngd_v1",
                "qngd_metric_rank_last": 0,
                "qngd_metric_condition_last": None,
                "qngd_state_probe_count": int(state_probe_count),
                "qngd_metric_eval_count": 0,
                "qngd_accepted_step_count": 0,
                "qngd_line_search_backtracks_total": 0,
                "qngd_energy_decrease_total": 0.0,
                "qngd_maxiter_effective": int(maxiter),
            },
        )

    for iteration in range(int(maxiter)):
        tangents = _horizontal_tangents(x, psi_current)
        metric_eval_count += 1
        residual = np.asarray(hpsi_current, dtype=complex).reshape(-1) - float(current_energy) * psi_current
        n = int(len(tangents))
        metric = np.zeros((n, n), dtype=float)
        for i in range(n):
            ti = tangents[int(i)]
            for j in range(i, n):
                val = float(np.real(np.vdot(ti, tangents[int(j)])))
                metric[int(i), int(j)] = val
                metric[int(j), int(i)] = val
        eigvals = np.linalg.eigvalsh(0.5 * (metric + metric.T)) if n else np.zeros(0, dtype=float)
        max_eig = float(np.max(eigvals)) if eigvals.size else 0.0
        rank_tol = max(float(metric_floor_value), 1e-12) * max(1.0, max_eig)
        resolved = eigvals[eigvals > rank_tol] if eigvals.size else np.zeros(0, dtype=float)
        last_rank = int(resolved.size)
        last_condition = (
            float(max_eig / float(np.min(resolved)))
            if resolved.size and max_eig > 0.0
            else None
        )
        regularization = max(float(metric_floor_value), 1e-12 * max(1.0, max_eig))
        metric_reg = metric + float(regularization) * np.eye(n, dtype=float)
        force = np.asarray(
            [-2.0 * float(np.real(np.vdot(tangent, residual))) for tangent in tangents],
            dtype=float,
        )
        step = np.linalg.pinv(metric_reg, rcond=max(1e-10, float(metric_floor_value))) @ force
        step = np.asarray(step, dtype=float).reshape(-1)
        last_l2_norm = float(np.linalg.norm(step)) if step.size else 0.0
        last_fs_norm = (
            float(math.sqrt(max(0.0, float(step @ metric_reg @ step))))
            if step.size
            else 0.0
        )
        last_max_abs_step = float(np.max(np.abs(step))) if step.size else 0.0
        if last_fs_norm < max(1e-10, float(metric_floor_value)):
            message = "sp_qngd_natural_step_threshold"
            success = True
            break
        if last_max_abs_step > float(max_abs_step):
            step *= float(max_abs_step) / float(last_max_abs_step)
            last_max_abs_step = float(max_abs_step)
            last_l2_norm = float(np.linalg.norm(step))
            last_fs_norm = float(math.sqrt(max(0.0, float(step @ metric_reg @ step))))

        accepted = False
        trial_x = x
        trial_energy = float(current_energy)
        trial_psi = psi_current
        trial_hpsi = hpsi_current
        alpha = 1.0
        for backtrack in range(int(max_backtracks) + 1):
            candidate_x = np.asarray(x, dtype=float) + float(alpha) * step
            candidate_energy, candidate_psi, candidate_hpsi = _exact_energy_hpsi_for_x(candidate_x)
            if float(candidate_energy) <= float(current_energy) - float(accept_tol):
                trial_x = np.asarray(candidate_x, dtype=float).copy()
                trial_energy = float(candidate_energy)
                trial_psi = np.asarray(candidate_psi, dtype=complex).reshape(-1)
                trial_hpsi = np.asarray(candidate_hpsi, dtype=complex).reshape(-1)
                backtracks_total += int(backtrack)
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            backtracks_total += int(max_backtracks) + 1
            message = (
                "sp_qngd_stationary_line_search_exhausted"
                if int(accepted_step_count) > 0
                else "sp_qngd_line_search_failed"
            )
            success = bool(int(accepted_step_count) > 0)
            break
        step_drop = max(0.0, float(current_energy) - float(trial_energy))
        energy_decrease_total += float(step_drop)
        x = np.asarray(trial_x, dtype=float).reshape(-1)
        current_energy = float(trial_energy)
        psi_current = np.asarray(trial_psi, dtype=complex).reshape(-1)
        hpsi_current = np.asarray(trial_hpsi, dtype=complex).reshape(-1)
        nit += 1
        accepted_step_count += 1
        if float(step_drop) < 1e-12:
            message = "sp_qngd_energy_decrease_threshold"
            success = True
            break
        if (iteration + 1) >= int(maxiter):
            message = "sp_qngd_maxiter_reached"
            success = False

    return SimpleNamespace(
        x=np.asarray(x, dtype=float),
        fun=float(current_energy),
        nfev=int(nfev),
        nit=int(nit),
        success=bool(success),
        message=str(message),
        qngd_info={
            "optimizer": "route_c_sp_qngd_v1",
            "qngd_metric_rank_last": int(last_rank),
            "qngd_metric_condition_last": last_condition,
            "qngd_step_fs_norm_last": float(last_fs_norm),
            "qngd_step_l2_norm_last": float(last_l2_norm),
            "qngd_max_abs_step_last": float(last_max_abs_step),
            "qngd_state_probe_count": int(state_probe_count),
            "qngd_metric_eval_count": int(metric_eval_count),
            "qngd_accepted_step_count": int(accepted_step_count),
            "qngd_line_search_backtracks_total": int(backtracks_total),
            "qngd_energy_decrease_total": float(energy_decrease_total),
            "qngd_maxiter_effective": int(maxiter),
            "qngd_tangent_eps": float(tangent_eps),
            "qngd_metric_floor": float(metric_floor_value),
        },
    )
