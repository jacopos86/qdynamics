#!/usr/bin/env python3
"""Standalone ADAPT engine helpers extracted from adapt_pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.builders.problem_registry import (
    default_continuation_mode_for_problem,
    supported_continuation_modes_for_problem,
)
from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    expand_legacy_logical_theta,
    project_runtime_theta_block_mean,
    runtime_indices_for_logical_indices,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    energy_via_one_apply,
)
from src.quantum.coordinate_descent_optimizer import rotosolve_coordinate_descent
from src.quantum.pauli_actions import CompiledPauliAction
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    apply_exp_pauli_polynomial,
    apply_exp_pauli_polynomial_termwise,
    apply_pauli_string,
    expval_pauli_polynomial,
)
from pipelines.scaffold.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.scaffold.hh_continuation_stage_control import StageController
from pipelines.scaffold.hh_continuation_types import ScaffoldCoordinateMetadata
from pipelines.static_adapt.selector_measurement_proxy import ControllerMeasurementWorkAccumulator
from pipelines.static_adapt.route_a_trust_region import (
    RouteATrustRegionState,
    exact_fubini_study_distance,
)
from pipelines.static_adapt.joint_step_warm_start import (
    guard_atomic_joint_step_candidates,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_POWELL_COORDINATE_CHART_POLICY_CHOICES,
)

STATIC_ADAPT_ALLOW_UNSTENCILED_ROTOSOLVE_ENV = "STATIC_ADAPT_ALLOW_UNSTENCILED_ROTOSOLVE"


@dataclass
class AdaptVQEResult:
    """Result container for the hardcoded ADAPT-VQE run."""

    energy: float
    theta: np.ndarray
    selected_ops: list[AnsatzTerm]
    history: list[dict[str, Any]]
    stop_reason: str
    nfev_total: int


OptimizerCoordinateMode = Literal["logical_shared", "runtime"]


@dataclass(frozen=True)
class SelectedOptimizerChart:
    """Typed optimizer view over the expanded runtime parameter vector.

    Static-ADAPT artifacts retain one runtime entry per executable Pauli
    factor.  Powell may instead optimize one shared coordinate per logical
    generator.  This value object records both coordinate systems explicitly
    so callers do not have to infer whether a reduced position denotes a
    logical generator or an individual runtime factor.
    """

    objective: Callable[[np.ndarray], float]
    x0: np.ndarray
    lift_to_runtime: Callable[[np.ndarray], np.ndarray]
    coordinate_mode: OptimizerCoordinateMode
    active_logical_indices: tuple[int, ...]
    active_runtime_indices: tuple[int, ...]
    active_optimizer_indices: tuple[int, ...]
    reduced_positions_by_logical: Mapping[int, tuple[int, ...]]


@dataclass(frozen=True)
class _ADAPTLogicalCandidate:
    """Private ADAPT selection unit for multi-parameter logical pool elements."""

    logical_label: str
    pool_indices: tuple[int, ...]
    parameterization: str
    family_id: str


@dataclass
class _BeamBranchState:
    branch_id: int
    parent_branch_id: int | None
    depth_local: int
    terminated: bool
    stop_reason: str | None
    selected_ops: list[AnsatzTerm]
    theta: np.ndarray
    energy_current: float
    available_indices: set[int]
    selection_counts: np.ndarray
    history: list[dict[str, Any]]
    phase1_stage: StageController
    phase1_residual_opened: bool
    phase1_last_probe_reason: str
    phase1_last_positions_considered: list[int]
    phase1_last_trough_detected: bool
    phase1_last_trough_probe_triggered: bool
    phase1_last_selected_score: float | None
    phase1_features_history: list[dict[str, Any]]
    phase1_stage_events: list[dict[str, Any]]
    phase1_measure_cache: MeasurementCacheAudit
    controller_measurement_work: ControllerMeasurementWorkAccumulator
    phase1_last_retained_records: list[dict[str, Any]]
    phase2_optimizer_memory: dict[str, Any]
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_geometric_shortlist_records: list[dict[str, Any]]
    phase2_last_retained_shortlist_records: list[dict[str, Any]]
    phase2_last_admitted_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
    phase2_last_batch_schur_context: dict[str, Any]
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    drop_prev_delta_abs: float
    drop_plateau_hits: int
    eps_energy_low_streak: int
    phase3_split_events: list[dict[str, Any]]
    phase3_runtime_split_summary: dict[str, Any]
    phase3_motif_usage: dict[str, Any]
    phase3_rescue_history: list[dict[str, Any]]
    phase1_prune_metadata: list[ScaffoldCoordinateMetadata]
    phase1_prune_first_seen_steps: dict[str, int]
    phase1_last_prune_summary: dict[str, Any]
    last_transition_kind: str
    last_admission_record_count: int
    cumulative_selector_score: float
    cumulative_selector_burden: float
    nfev_total_local: int
    route_a_trust_region_state: RouteATrustRegionState | None = None
    cumulative_beam_cost: float = 0.0
    source_lock_admitted_candidate_labels: list[str] = field(default_factory=list)
    formal_manifold_runtime: Any | None = None

    def clone_for_child(self, *, branch_id: int) -> "_BeamBranchState":
        return _BeamBranchState(
            branch_id=int(branch_id),
            parent_branch_id=int(self.branch_id),
            depth_local=int(self.depth_local),
            terminated=bool(self.terminated),
            stop_reason=(None if self.stop_reason is None else str(self.stop_reason)),
            selected_ops=list(self.selected_ops),
            theta=np.asarray(self.theta, dtype=float).copy(),
            energy_current=float(self.energy_current),
            available_indices=set(int(x) for x in self.available_indices),
            selection_counts=np.asarray(self.selection_counts, dtype=np.int64).copy(),
            history=copy.deepcopy(self.history),
            phase1_stage=self.phase1_stage.clone(),
            phase1_residual_opened=bool(self.phase1_residual_opened),
            phase1_last_probe_reason=str(self.phase1_last_probe_reason),
            phase1_last_positions_considered=[int(x) for x in self.phase1_last_positions_considered],
            phase1_last_trough_detected=bool(self.phase1_last_trough_detected),
            phase1_last_trough_probe_triggered=bool(self.phase1_last_trough_probe_triggered),
            phase1_last_selected_score=(
                None if self.phase1_last_selected_score is None else float(self.phase1_last_selected_score)
            ),
            phase1_features_history=copy.deepcopy(self.phase1_features_history),
            phase1_stage_events=copy.deepcopy(self.phase1_stage_events),
            phase1_measure_cache=self.phase1_measure_cache.clone(),
            controller_measurement_work=self.controller_measurement_work.clone(),
            phase1_last_retained_records=copy.deepcopy(self.phase1_last_retained_records),
            phase2_optimizer_memory=copy.deepcopy(self.phase2_optimizer_memory),
            phase2_last_shortlist_records=copy.deepcopy(self.phase2_last_shortlist_records),
            phase2_last_geometric_shortlist_records=copy.deepcopy(self.phase2_last_geometric_shortlist_records),
            phase2_last_retained_shortlist_records=copy.deepcopy(self.phase2_last_retained_shortlist_records),
            phase2_last_admitted_records=copy.deepcopy(self.phase2_last_admitted_records),
            phase2_last_batch_selected=bool(self.phase2_last_batch_selected),
            phase2_last_batch_penalty_total=float(self.phase2_last_batch_penalty_total),
            phase2_last_batch_schur_context=copy.deepcopy(self.phase2_last_batch_schur_context),
            phase2_last_optimizer_memory_reused=bool(self.phase2_last_optimizer_memory_reused),
            phase2_last_optimizer_memory_source=str(self.phase2_last_optimizer_memory_source),
            phase2_last_shortlist_eval_records=copy.deepcopy(self.phase2_last_shortlist_eval_records),
            drop_prev_delta_abs=float(self.drop_prev_delta_abs),
            drop_plateau_hits=int(self.drop_plateau_hits),
            eps_energy_low_streak=int(self.eps_energy_low_streak),
            phase3_split_events=copy.deepcopy(self.phase3_split_events),
            phase3_runtime_split_summary=copy.deepcopy(self.phase3_runtime_split_summary),
            phase3_motif_usage=copy.deepcopy(self.phase3_motif_usage),
            phase3_rescue_history=copy.deepcopy(self.phase3_rescue_history),
            phase1_prune_metadata=[
                ScaffoldCoordinateMetadata(**dict(x.__dict__)) for x in self.phase1_prune_metadata
            ],
            phase1_prune_first_seen_steps={
                str(k): int(v) for k, v in self.phase1_prune_first_seen_steps.items()
            },
            phase1_last_prune_summary=copy.deepcopy(self.phase1_last_prune_summary),
            last_transition_kind=str(self.last_transition_kind),
            last_admission_record_count=int(self.last_admission_record_count),
            cumulative_selector_score=float(self.cumulative_selector_score),
            cumulative_selector_burden=float(self.cumulative_selector_burden),
            nfev_total_local=int(self.nfev_total_local),
            route_a_trust_region_state=(
                None
                if self.route_a_trust_region_state is None
                else self.route_a_trust_region_state.clone()
            ),
            cumulative_beam_cost=float(getattr(self, "cumulative_beam_cost", 0.0)),
            source_lock_admitted_candidate_labels=[
                str(x) for x in getattr(self, "source_lock_admitted_candidate_labels", [])
            ],
            formal_manifold_runtime=(
                None
                if getattr(self, "formal_manifold_runtime", None) is None
                else self.formal_manifold_runtime.fork(
                    branch_id=f"beam_branch:{int(branch_id)}"
                )
            ),
        )


@dataclass(frozen=True)
class _BranchExpansionPlan:
    candidate_pool_index: int
    position_id: int
    selection_mode: str
    candidate_label: str
    candidate_term: AnsatzTerm
    feature_row: dict[str, Any] | None
    init_theta: float = 0.0
    batch_records: tuple[dict[str, Any], ...] = ()
    batch_summary: dict[str, Any] | None = None
    batch_score: float | None = None
    batch_delta_e3: float | None = None
    batch_k3: float | None = None
    batch_denominator_1_plus_k3: float | None = None


@dataclass
class _BranchStepScratch:
    energy_current: float
    psi_current: np.ndarray
    hpsi_current: np.ndarray
    gradients: np.ndarray
    grad_magnitudes: np.ndarray
    max_grad: float
    gradient_eval_elapsed_s: float
    append_position: int
    best_idx: int
    selected_position: int
    selection_mode: str
    stage_name: str
    phase1_feature_selected: dict[str, Any] | None
    phase1_stage_transition_reason: str
    phase1_stage_now: str
    phase1_stage_after_transition: StageController
    phase1_last_probe_reason: str
    phase1_last_positions_considered: list[int]
    phase1_last_trough_detected: bool
    phase1_last_trough_probe_triggered: bool
    phase1_last_selected_score: float | None
    phase1_last_retained_records: list[dict[str, Any]]
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_geometric_shortlist_records: list[dict[str, Any]]
    phase2_last_retained_shortlist_records: list[dict[str, Any]]
    phase2_last_admitted_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
    phase2_last_batch_schur_context: dict[str, Any]
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    phase1_residual_opened: bool
    available_indices_after_transition: set[int]
    phase1_stage_events_after_transition: list[dict[str, Any]]
    controller_measurement_work_after_eval: ControllerMeasurementWorkAccumulator
    controller_measurement_work_step_proxy: dict[str, Any]
    phase3_runtime_split_summary_after_eval: dict[str, Any]
    proposals: list[_BranchExpansionPlan]
    stop_reason: str | None
    fallback_scan_size: int
    fallback_best_probe_delta_e: float | None
    fallback_best_probe_theta: float | None
    phase1_raw_record_count: int = 0
    phase2_raw_record_count: int = 0
    phase1_shortlist_size: int = 0
    phase2_shortlist_size: int = 0
    phase3_shortlist_size: int = 0
    gradient_parallel_telemetry: dict[str, Any] | None = None
    nfev_delta: int = 0
    sr_active_only_correction: dict[str, Any] | None = None


@dataclass
class _BeamParentScratchResult:
    parent_ordinal: int
    parent_branch_id: int
    scratch: _BranchStepScratch
    log_events: list[dict[str, Any]]
    elapsed_s: float


def _beam_label_signature(ops: Sequence[AnsatzTerm]) -> tuple[str, ...]:
    return tuple(str(op.label) for op in ops)


def _beam_round10_theta(theta_now: np.ndarray) -> tuple[float, ...]:
    theta_vec = np.asarray(theta_now, dtype=float).reshape(-1)
    return tuple(round(float(x), 10) for x in theta_vec.tolist())


def _branch_state_fingerprint(branch: _BeamBranchState) -> str:
    payload = {
        "depth_local": int(branch.depth_local),
        "labels": list(_beam_label_signature(branch.selected_ops)),
        "theta_round10": list(_beam_round10_theta(branch.theta)),
    }
    trust_state = getattr(branch, "route_a_trust_region_state", None)
    if trust_state is not None and not math.isclose(
        float(trust_state.radius),
        float(trust_state.reference_radius),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        payload["route_a_trust_region_radius"] = round(
            float(trust_state.radius),
            12,
        )
    formal_runtime = getattr(branch, "formal_manifold_runtime", None)
    if formal_runtime is not None:
        behavior = getattr(
            formal_runtime, "behavioral_fingerprint_payload", None
        )
        if not callable(behavior):
            raise TypeError(
                "formal_manifold_runtime must expose "
                "behavioral_fingerprint_payload()."
            )
        payload["formal_manifold_behavior"] = behavior()
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _proposal_fingerprint(
    *,
    parent: _BeamBranchState,
    plan: _BranchExpansionPlan,
) -> str:
    if not getattr(plan, "batch_records", ()):
        payload = {
            "parent": _branch_state_fingerprint(parent),
            "candidate_pool_index": int(plan.candidate_pool_index),
            "position_id": int(plan.position_id),
            "selection_mode": str(plan.selection_mode),
            "candidate_label": str(plan.candidate_label),
            "init_theta": round(float(plan.init_theta), 12),
        }
    else:
        batch_payload = []
        for rec in plan.batch_records:
            term = rec.get("candidate_term") if isinstance(rec, Mapping) else None
            label = (
                str(rec.get("candidate_label"))
                if isinstance(rec, Mapping) and rec.get("candidate_label") is not None
                else (str(getattr(term, "label")) if getattr(term, "label", None) is not None else "")
            )
            batch_payload.append(
                {
                    "candidate_pool_index": int(rec.get("candidate_pool_index", -1)),
                    "position_id": int(rec.get("position_id", -1)),
                    "candidate_label": str(label),
                    "init_theta": round(float(rec.get("init_theta", 0.0)), 12),
                }
            )
        payload = {
            "parent": _branch_state_fingerprint(parent),
            "selection_mode": str(plan.selection_mode),
            "batch": batch_payload,
            "batch_score": (
                None if plan.batch_score is None else round(float(plan.batch_score), 12)
            ),
            "batch_delta_e3": (
                None if plan.batch_delta_e3 is None else round(float(plan.batch_delta_e3), 12)
            ),
            "batch_k3": None if plan.batch_k3 is None else round(float(plan.batch_k3), 12),
        }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _branch_optimizer_seed(
    *,
    base_seed: int,
    stage_tag: str,
    depth_local: int,
    parent_state_fingerprint: str,
    proposal_fingerprint: str | None,
) -> int:
    payload = {
        "base_seed": int(base_seed),
        "stage_tag": str(stage_tag),
        "depth_local": int(depth_local),
        "parent_state_fingerprint": str(parent_state_fingerprint),
        "proposal_fingerprint": (None if proposal_fingerprint is None else str(proposal_fingerprint)),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _resolve_beam_local_reopt_seed_inputs(
    *,
    proposal_count: int,
    proposal_fingerprint: str | None,
) -> tuple[str, str | None]:
    if int(proposal_count) > 1:
        return "parent_shared", None
    return "proposal_conditioned", (
        None if proposal_fingerprint is None else str(proposal_fingerprint)
    )


def _beam_prune_key_payload(branch: _BeamBranchState) -> dict[str, Any]:
    return {
        "energy": float(branch.energy_current),
        "cumulative_selector_score": float(branch.cumulative_selector_score),
        "cumulative_selector_burden": float(branch.cumulative_selector_burden),
        "ansatz_depth": int(len(branch.selected_ops)),
        "labels": list(_beam_label_signature(branch.selected_ops)),
        "theta_round10": [float(x) for x in _beam_round10_theta(branch.theta)],
        "theta_round10_digits": 10,
        "branch_id": int(branch.branch_id),
    }


def _beam_prune_key(branch: _BeamBranchState) -> tuple[Any, ...]:
    payload = _beam_prune_key_payload(branch)
    return (
        float(payload["energy"]),
        -float(payload["cumulative_selector_score"]),
        float(payload["cumulative_selector_burden"]),
        int(payload["ansatz_depth"]),
        tuple(str(x) for x in payload["labels"]),
        tuple(float(x) for x in payload["theta_round10"]),
        int(payload["branch_id"]),
    )


def _beam_dedup(branches: Sequence[_BeamBranchState]) -> list[_BeamBranchState]:
    keep: dict[str, _BeamBranchState] = {}
    for branch in branches:
        fingerprint = _branch_state_fingerprint(branch)
        incumbent = keep.get(fingerprint)
        if incumbent is None or _beam_prune_key(branch) < _beam_prune_key(incumbent):
            keep[fingerprint] = branch
    return list(keep.values())


def _beam_prune(
    branches: Sequence[_BeamBranchState],
    *,
    cap: int,
) -> list[_BeamBranchState]:
    deduped = _beam_dedup(branches)
    return sorted(deduped, key=_beam_prune_key)[: int(max(0, cap))]


def _beam_branch_cost(branch: _BeamBranchState) -> float:
    raw_cost = getattr(branch, "cumulative_beam_cost", None)
    if raw_cost is not None:
        try:
            cost = float(raw_cost)
            if math.isfinite(cost):
                return float(max(0.0, cost))
        except (TypeError, ValueError):
            pass
    return float(max(0.0, float(getattr(branch, "cumulative_selector_burden", 1.0)) - 1.0))


def _beam_energy_cost_scalar(branch: _BeamBranchState, *, lambda_beam: float) -> float:
    return float(float(branch.energy_current) + float(lambda_beam) * _beam_branch_cost(branch))


def _beam_energy_cost_prune_key_payload(
    branch: _BeamBranchState,
    *,
    lambda_beam: float,
) -> dict[str, Any]:
    cost = _beam_branch_cost(branch)
    return {
        "schema": "beam_energy_cost_pareto_lambda_v1",
        "energy": float(branch.energy_current),
        "cumulative_beam_cost": float(cost),
        "lambda_beam": float(lambda_beam),
        "energy_cost_scalar": float(_beam_energy_cost_scalar(branch, lambda_beam=lambda_beam)),
        "cumulative_selector_score": float(branch.cumulative_selector_score),
        "cumulative_selector_burden": float(branch.cumulative_selector_burden),
        "ansatz_depth": int(len(branch.selected_ops)),
        "labels": list(_beam_label_signature(branch.selected_ops)),
        "theta_round10": [float(x) for x in _beam_round10_theta(branch.theta)],
        "theta_round10_digits": 10,
        "branch_id": int(branch.branch_id),
        "pareto_survival_policy": "pairwise_energy_cost_lambda_v1",
    }


def _beam_energy_cost_sort_key(
    branch: _BeamBranchState,
    *,
    lambda_beam: float,
) -> tuple[Any, ...]:
    payload = _beam_energy_cost_prune_key_payload(branch, lambda_beam=lambda_beam)
    return (
        float(payload["energy_cost_scalar"]),
        float(payload["energy"]),
        float(payload["cumulative_beam_cost"]),
        -float(payload["cumulative_selector_score"]),
        float(payload["cumulative_selector_burden"]),
        int(payload["ansatz_depth"]),
        tuple(str(x) for x in payload["labels"]),
        tuple(float(x) for x in payload["theta_round10"]),
        int(payload["branch_id"]),
    )


def _beam_energy_cost_dominance_case(
    lhs: _BeamBranchState,
    rhs: _BeamBranchState,
    *,
    lambda_beam: float,
    atol: float = 1e-12,
) -> dict[str, Any]:
    e_l = float(lhs.energy_current)
    e_r = float(rhs.energy_current)
    k_l = _beam_branch_cost(lhs)
    k_r = _beam_branch_cost(rhs)
    de = float(e_l - e_r)
    dk = float(k_l - k_r)
    scalar_gap = float(de + float(lambda_beam) * dk)
    lhs_no_worse = bool(e_l <= e_r + float(atol) and k_l <= k_r + float(atol))
    rhs_no_worse = bool(e_r <= e_l + float(atol) and k_r <= k_l + float(atol))
    lhs_strict = bool(e_l < e_r - float(atol) or k_l < k_r - float(atol))
    rhs_strict = bool(e_r < e_l - float(atol) or k_r < k_l - float(atol))
    if lhs_no_worse and lhs_strict:
        case = "lhs_strict_pareto"
        dominates = True
    elif rhs_no_worse and rhs_strict:
        case = "rhs_strict_pareto"
        dominates = False
    elif abs(de) <= float(atol) and abs(dk) <= float(atol):
        case = "equal_energy_cost"
        dominates = False
    elif e_l < e_r and k_l > k_r:
        case = "tradeoff_lhs_lower_energy_higher_cost"
        dominates = bool(scalar_gap < -float(atol))
    elif e_l > e_r and k_l < k_r:
        case = "tradeoff_lhs_higher_energy_lower_cost"
        dominates = bool(scalar_gap < -float(atol))
    elif e_l < e_r:
        case = "lhs_lower_energy"
        dominates = bool(scalar_gap < -float(atol))
    elif k_l < k_r:
        case = "lhs_lower_cost"
        dominates = bool(scalar_gap < -float(atol))
    else:
        case = "rhs_preferred"
        dominates = False
    return {
        "case": str(case),
        "lhs_dominates_rhs": bool(dominates),
        "energy_lhs": float(e_l),
        "energy_rhs": float(e_r),
        "cost_lhs": float(k_l),
        "cost_rhs": float(k_r),
        "delta_energy_lhs_minus_rhs": float(de),
        "delta_cost_lhs_minus_rhs": float(dk),
        "lambda_beam": float(lambda_beam),
        "scalar_gap_lhs_minus_rhs": float(scalar_gap),
    }


def _beam_energy_cost_dominates(
    lhs: _BeamBranchState,
    rhs: _BeamBranchState,
    *,
    lambda_beam: float,
    atol: float = 1e-12,
) -> bool:
    return bool(
        _beam_energy_cost_dominance_case(
            lhs,
            rhs,
            lambda_beam=lambda_beam,
            atol=atol,
        ).get("lhs_dominates_rhs", False)
    )


def _beam_dedup_energy_cost(
    branches: Sequence[_BeamBranchState],
    *,
    lambda_beam: float,
) -> list[_BeamBranchState]:
    keep: dict[str, _BeamBranchState] = {}
    for branch in branches:
        fingerprint = _branch_state_fingerprint(branch)
        incumbent = keep.get(fingerprint)
        if incumbent is None or _beam_energy_cost_sort_key(branch, lambda_beam=lambda_beam) < _beam_energy_cost_sort_key(incumbent, lambda_beam=lambda_beam):
            keep[fingerprint] = branch
    return list(keep.values())


def _beam_prune_energy_cost_pareto_with_audit(
    branches: Sequence[_BeamBranchState],
    *,
    cap: int,
    lambda_beam: float,
) -> tuple[list[_BeamBranchState], dict[str, Any]]:
    deduped = _beam_dedup_energy_cost(branches, lambda_beam=lambda_beam)
    dominated_ids: set[int] = set()
    dominance_events: list[dict[str, Any]] = []
    for i, candidate in enumerate(deduped):
        for j, challenger in enumerate(deduped):
            if i == j:
                continue
            event = _beam_energy_cost_dominance_case(
                challenger,
                candidate,
                lambda_beam=float(lambda_beam),
            )
            if bool(event.get("lhs_dominates_rhs", False)):
                dominated_ids.add(int(candidate.branch_id))
                dominance_events.append(
                    {
                        **dict(event),
                        "dominating_branch_id": int(challenger.branch_id),
                        "dominated_branch_id": int(candidate.branch_id),
                    }
                )
                break
    nondominated = [branch for branch in deduped if int(branch.branch_id) not in dominated_ids]
    ordered = sorted(nondominated, key=lambda branch: _beam_energy_cost_sort_key(branch, lambda_beam=lambda_beam))
    kept = ordered[: int(max(0, cap))]
    audit = {
        "schema": "beam_energy_cost_pareto_prune_audit_v1",
        "lambda_beam": float(lambda_beam),
        "input_count": int(len(branches)),
        "deduped_count": int(len(deduped)),
        "dominated_count": int(len(dominated_ids)),
        "nondominated_count": int(len(nondominated)),
        "kept_count": int(len(kept)),
        "cap": int(max(0, cap)),
        "dominance_events": dominance_events,
        "kept_branch_ids": [int(branch.branch_id) for branch in kept],
    }
    return kept, audit


def _beam_gain_per_added_cost_prune_key_payload(
    branch: _BeamBranchState,
    *,
    energy_root: float,
    legacy_lambda_beam: float | None = None,
) -> dict[str, Any]:
    """Canonical Route-A beam key from realized energy gain and raw added cost."""

    root = float(energy_root)
    energy = float(branch.energy_current)
    if not math.isfinite(root) or not math.isfinite(energy):
        raise ValueError("Canonical beam survival requires finite root and branch energies.")
    added_burden = _beam_branch_cost(branch)
    denominator = float(1.0 + added_burden)
    realized_gain = float(root - energy)
    score = float(realized_gain / denominator)
    return {
        "schema": "beam_realized_gain_per_added_cost_v1",
        "energy_root": float(root),
        "energy": float(energy),
        "realized_energy_gain": float(realized_gain),
        "cumulative_added_burden": float(added_burden),
        "cumulative_beam_cost": float(added_burden),
        "denominator_1_plus_added_burden": float(denominator),
        "beam_survival_score": float(score),
        "legacy_lambda_beam": (
            None if legacy_lambda_beam is None else float(legacy_lambda_beam)
        ),
        "legacy_lambda_beam_effect": "ignored",
        "cumulative_selector_score": float(branch.cumulative_selector_score),
        "cumulative_selector_burden": float(branch.cumulative_selector_burden),
        "ansatz_depth": int(len(branch.selected_ops)),
        "labels": list(_beam_label_signature(branch.selected_ops)),
        "theta_round10": [float(x) for x in _beam_round10_theta(branch.theta)],
        "theta_round10_digits": 10,
        "branch_id": int(branch.branch_id),
        "pareto_survival_policy": "realized_energy_added_burden_v1",
        "energy_state": "post_refit_post_prune",
    }


def _beam_gain_per_added_cost_sort_key(
    branch: _BeamBranchState,
    *,
    energy_root: float,
    legacy_lambda_beam: float | None = None,
) -> tuple[Any, ...]:
    payload = _beam_gain_per_added_cost_prune_key_payload(
        branch,
        energy_root=float(energy_root),
        legacy_lambda_beam=legacy_lambda_beam,
    )
    return (
        -float(payload["beam_survival_score"]),
        float(payload["energy"]),
        float(payload["cumulative_added_burden"]),
        -float(payload["realized_energy_gain"]),
        int(payload["ansatz_depth"]),
        tuple(str(x) for x in payload["labels"]),
        tuple(float(x) for x in payload["theta_round10"]),
        int(payload["branch_id"]),
    )


def _beam_realized_energy_added_cost_dominance_case(
    lhs: _BeamBranchState,
    rhs: _BeamBranchState,
    *,
    atol: float = 1e-12,
) -> dict[str, Any]:
    """Return strict Pareto dominance in realized energy and added burden."""

    e_l = float(lhs.energy_current)
    e_r = float(rhs.energy_current)
    k_l = _beam_branch_cost(lhs)
    k_r = _beam_branch_cost(rhs)
    if not all(math.isfinite(value) for value in (e_l, e_r, k_l, k_r)):
        raise ValueError("Canonical beam Pareto comparison requires finite energy and cost.")
    lhs_no_worse = bool(e_l <= e_r + float(atol) and k_l <= k_r + float(atol))
    rhs_no_worse = bool(e_r <= e_l + float(atol) and k_r <= k_l + float(atol))
    lhs_strict = bool(e_l < e_r - float(atol) or k_l < k_r - float(atol))
    rhs_strict = bool(e_r < e_l - float(atol) or k_r < k_l - float(atol))
    if lhs_no_worse and lhs_strict:
        case = "lhs_strict_pareto"
        dominates = True
    elif rhs_no_worse and rhs_strict:
        case = "rhs_strict_pareto"
        dominates = False
    elif abs(e_l - e_r) <= float(atol) and abs(k_l - k_r) <= float(atol):
        case = "equal_energy_added_burden"
        dominates = False
    else:
        case = "nondominated_tradeoff"
        dominates = False
    return {
        "case": str(case),
        "lhs_dominates_rhs": bool(dominates),
        "energy_lhs": float(e_l),
        "energy_rhs": float(e_r),
        "cumulative_added_burden_lhs": float(k_l),
        "cumulative_added_burden_rhs": float(k_r),
        "delta_energy_lhs_minus_rhs": float(e_l - e_r),
        "delta_added_burden_lhs_minus_rhs": float(k_l - k_r),
    }


def _beam_dedup_gain_per_added_cost(
    branches: Sequence[_BeamBranchState],
    *,
    energy_root: float,
    legacy_lambda_beam: float | None = None,
) -> list[_BeamBranchState]:
    keep: dict[str, _BeamBranchState] = {}
    for branch in branches:
        fingerprint = _branch_state_fingerprint(branch)
        incumbent = keep.get(fingerprint)
        if incumbent is None or _beam_gain_per_added_cost_sort_key(
            branch,
            energy_root=float(energy_root),
            legacy_lambda_beam=legacy_lambda_beam,
        ) < _beam_gain_per_added_cost_sort_key(
            incumbent,
            energy_root=float(energy_root),
            legacy_lambda_beam=legacy_lambda_beam,
        ):
            keep[fingerprint] = branch
    return list(keep.values())


def _beam_prune_gain_per_added_cost_pareto_with_audit(
    branches: Sequence[_BeamBranchState],
    *,
    cap: int,
    energy_root: float,
    legacy_lambda_beam: float | None = None,
    cost_contract: Mapping[str, Any] | None = None,
) -> tuple[list[_BeamBranchState], dict[str, Any]]:
    """Apply strict Pareto filtering, then maximize canonical beam score."""

    root = float(energy_root)
    if not math.isfinite(root):
        raise ValueError("Canonical beam survival requires a finite shared energy_root.")
    deduped = _beam_dedup_gain_per_added_cost(
        branches,
        energy_root=root,
        legacy_lambda_beam=legacy_lambda_beam,
    )
    dominated_ids: set[int] = set()
    dominance_events: list[dict[str, Any]] = []
    for i, candidate in enumerate(deduped):
        for j, challenger in enumerate(deduped):
            if i == j:
                continue
            event = _beam_realized_energy_added_cost_dominance_case(challenger, candidate)
            if bool(event.get("lhs_dominates_rhs", False)):
                dominated_ids.add(int(candidate.branch_id))
                dominance_events.append(
                    {
                        **dict(event),
                        "dominating_branch_id": int(challenger.branch_id),
                        "dominated_branch_id": int(candidate.branch_id),
                    }
                )
                break
    nondominated = [branch for branch in deduped if int(branch.branch_id) not in dominated_ids]
    ordered = sorted(
        nondominated,
        key=lambda branch: _beam_gain_per_added_cost_sort_key(
            branch,
            energy_root=root,
            legacy_lambda_beam=legacy_lambda_beam,
        ),
    )
    kept = ordered[: int(max(0, cap))]
    audit = {
        "schema": "beam_realized_gain_per_added_cost_prune_audit_v1",
        "energy_root": float(root),
        "legacy_lambda_beam": (
            None if legacy_lambda_beam is None else float(legacy_lambda_beam)
        ),
        "legacy_lambda_beam_effect": "ignored",
        "cost_contract": dict(cost_contract or {}),
        "cost_accumulation": "sum_step_denominator_minus_one",
        "phase_local_normalized_scores_accumulated": False,
        "input_count": int(len(branches)),
        "deduped_count": int(len(deduped)),
        "dominated_count": int(len(dominated_ids)),
        "nondominated_count": int(len(nondominated)),
        "kept_count": int(len(kept)),
        "cap": int(max(0, cap)),
        "dominance_events": dominance_events,
        "input_branch_keys": [
            _beam_gain_per_added_cost_prune_key_payload(
                branch,
                energy_root=root,
                legacy_lambda_beam=legacy_lambda_beam,
            )
            for branch in deduped
        ],
        "kept_branch_ids": [int(branch.branch_id) for branch in kept],
    }
    return kept, audit


def _parse_seq2p_step_label(label: str) -> tuple[str, str] | None:
    raw = str(label).strip()
    for step_name in ("ferm", "motif"):
        suffix = f"::step={step_name}"
        if raw.endswith(suffix):
            return raw[: -len(suffix)], step_name
    return None


def _build_seq2p_logical_candidates(
    pool: Sequence[AnsatzTerm],
    *,
    family_id: str,
) -> list[_ADAPTLogicalCandidate]:
    candidates: list[_ADAPTLogicalCandidate] = []
    idx = 0
    while idx < len(pool):
        if idx + 1 >= len(pool):
            raise ValueError("Malformed seq2p pool: trailing unpaired flat term.")
        lhs = _parse_seq2p_step_label(str(pool[idx].label))
        rhs = _parse_seq2p_step_label(str(pool[idx + 1].label))
        if lhs is None or rhs is None:
            raise ValueError("Malformed seq2p pool: expected paired ::step=ferm/::step=motif labels.")
        lhs_base, lhs_step = lhs
        rhs_base, rhs_step = rhs
        if lhs_step != "ferm" or rhs_step != "motif" or lhs_base != rhs_base:
            raise ValueError("Malformed seq2p pool: expected adjacent ferm/motif terms for each logical pair.")
        candidates.append(
            _ADAPTLogicalCandidate(
                logical_label=str(lhs_base),
                pool_indices=(int(idx), int(idx + 1)),
                parameterization="double_sequential",
                family_id=str(family_id),
            )
        )
        idx += 2
    return candidates


def _logical_candidate_gradient_summary(
    candidate: _ADAPTLogicalCandidate,
    gradients: np.ndarray,
) -> tuple[float, list[float], list[float]]:
    signed_components = [float(gradients[int(idx)]) for idx in candidate.pool_indices]
    abs_components = [abs(float(val)) for val in signed_components]
    score = math.sqrt(sum(float(val) * float(val) for val in abs_components))
    return float(score), signed_components, abs_components


def _apply_pauli_polynomial_uncached(state: np.ndarray, poly: Any) -> np.ndarray:
    r"""Compute G|psi> where G is a PauliPolynomial (sum of weighted Pauli strings).

    G = \sum_j c_j P_j   =>   G|psi> = \sum_j c_j P_j|psi>
    """
    terms = poly.return_polynomial()
    if not terms:
        return np.zeros_like(state)
    nq = int(terms[0].nqubit())
    id_str = "e" * nq
    result = np.zeros_like(state)
    for term in terms:
        ps = term.pw2strng()
        coeff = complex(term.p_coeff)
        if abs(coeff) < 1e-15:
            continue
        if ps == id_str:
            result += coeff * state
        else:
            result += coeff * apply_pauli_string(state, ps)
    return result


def _apply_compiled_polynomial(state: np.ndarray, compiled_poly: CompiledPolynomialAction) -> np.ndarray:
    """Apply a compiled PauliPolynomial action to a statevector."""
    if int(getattr(compiled_poly, "nq", 0)) == 0 and len(compiled_poly.terms) == 0:
        return np.zeros_like(state)
    return _apply_compiled_polynomial_shared(state, compiled_poly)


def _apply_pauli_polynomial(
    state: np.ndarray,
    poly: Any,
    *,
    compiled: CompiledPolynomialAction | None = None,
) -> np.ndarray:
    if compiled is not None:
        return _apply_compiled_polynomial(state, compiled)
    return _apply_pauli_polynomial_uncached(state, poly)


def _commutator_gradient(
    h_poly: Any,
    pool_op: AnsatzTerm,
    psi_current: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    pool_compiled: CompiledPolynomialAction | None = None,
    hpsi_precomputed: np.ndarray | None = None,
) -> float:
    r"""Compute dE/dtheta at theta=0 for appending pool_op to the current state.

    E(theta) = <psi|exp(+i theta G) H exp(-i theta G)|psi>

    The analytic gradient at theta=0 is:
        dE/dtheta|_0 = i <psi|[H, G]|psi> = 2 Im(<psi|H G|psi>)

    Since H is Hermitian: <psi|H G|psi> = <H psi | G psi>.

    This is exact and works for multi-term PauliPolynomial generators
    (unlike the parameter-shift rule which requires single-Pauli generators).
    """
    g_psi = _apply_pauli_polynomial(psi_current, pool_op.polynomial, compiled=pool_compiled)
    h_psi = (
        np.asarray(hpsi_precomputed, dtype=complex)
        if hpsi_precomputed is not None
        else _apply_pauli_polynomial(psi_current, h_poly, compiled=h_compiled)
    )
    return adapt_commutator_grad_from_hpsi(h_psi, g_psi)


def evaluate_exact_gradient_surface(
    *,
    psi_current: np.ndarray,
    hpsi_current: np.ndarray,
    pool_compiled: Sequence[CompiledPolynomialAction],
    available_indices: Sequence[int] | set[int] | frozenset[int],
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Evaluate exact/noiseless ADAPT commutator gradients for pool indices.

    The helper is intentionally narrow: workers only read immutable state and
    return ``(index, gradient)`` records; all writes into full-length gradient
    arrays happen on the caller thread in ascending index order.
    """
    worker_count_requested = int(workers)
    if worker_count_requested < 1:
        raise ValueError("workers must be >= 1")

    pool_size = int(len(pool_compiled))
    sorted_indices = sorted({int(idx) for idx in available_indices})
    for idx in sorted_indices:
        if int(idx) < 0 or int(idx) >= pool_size:
            raise ValueError(
                f"available index out of range: idx={int(idx)}, pool_size={int(pool_size)}"
            )

    gradients = np.zeros(pool_size, dtype=float)
    grad_magnitudes = np.zeros(pool_size, dtype=float)
    hpsi_vec = np.asarray(hpsi_current, dtype=complex)
    psi_vec = np.asarray(psi_current, dtype=complex)
    effective_workers = (
        int(worker_count_requested)
        if int(worker_count_requested) > 1 and len(sorted_indices) >= 2
        else 1
    )

    def _evaluate_one(index: int) -> tuple[int, float]:
        apsi = _apply_compiled_polynomial(psi_vec, pool_compiled[int(index)])
        gradient = adapt_commutator_grad_from_hpsi(hpsi_vec, apsi)
        return (int(index), float(gradient))

    if int(effective_workers) > 1:
        with ThreadPoolExecutor(max_workers=int(effective_workers)) as executor:
            worker_results = list(executor.map(_evaluate_one, sorted_indices))
        backend = "ThreadPoolExecutor"
        mode = "parallel_threadpool"
    else:
        worker_results = [_evaluate_one(int(idx)) for idx in sorted_indices]
        backend = "serial"
        mode = "serial"

    gradient_by_index = {int(idx): float(gradient) for idx, gradient in worker_results}
    for idx in sorted_indices:
        gradient = float(gradient_by_index[int(idx)])
        gradients[int(idx)] = gradient
        grad_magnitudes[int(idx)] = abs(gradient)

    telemetry = {
        "surface": "exact_noiseless_commutator_v1",
        "requested_workers": int(worker_count_requested),
        "effective_workers": int(effective_workers),
        "parallel_enabled": bool(int(effective_workers) > 1),
        "backend": str(backend),
        "mode": str(mode),
        "pool_size": int(pool_size),
        "evaluated_count": int(len(sorted_indices)),
        "input_count": int(len(list(available_indices)) if not isinstance(available_indices, set) else len(available_indices)),
        "deduped_count": int(len(sorted_indices)),
        "merge_order": "ascending_index",
        "evaluated_indices": [int(idx) for idx in sorted_indices],
    }
    return gradients, grad_magnitudes, telemetry


def _prepare_adapt_state(
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    parameter_layout: AnsatzParameterLayout | None = None,
) -> np.ndarray:
    """Apply the current ADAPT ansatz.

    Supports both legacy logical-shared theta and per-Pauli runtime theta.
    """
    psi = np.array(psi_ref, copy=True)
    if not selected_ops:
        return psi
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    layout = (
        parameter_layout
        if parameter_layout is not None
        else build_parameter_layout(selected_ops, ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    )
    if any(
        str(getattr(op, "execution_mode", "termwise_product") or "termwise_product").strip().lower()
        == "grouped_exact"
        for op in selected_ops
    ):
        if int(theta_arr.size) == int(layout.logical_parameter_count):
            theta_logical = np.asarray(theta_arr, dtype=float)
        elif int(theta_arr.size) == int(layout.runtime_parameter_count):
            theta_uniform_tol = max(1e-10, 100.0 * float(layout.coefficient_tolerance))
            for block, op in zip(layout.blocks, selected_ops):
                execution_mode = str(
                    getattr(op, "execution_mode", "termwise_product") or "termwise_product"
                ).strip().lower()
                if execution_mode != "grouped_exact" or int(block.runtime_count) <= 1:
                    continue
                block_theta = theta_arr[int(block.runtime_start):int(block.runtime_stop)]
                block_mean = float(np.mean(block_theta))
                max_deviation = float(np.max(np.abs(block_theta - block_mean)))
                if max_deviation > float(theta_uniform_tol):
                    raise ValueError(
                        "non-uniform grouped_exact theta block: "
                        f"label={block.candidate_label!r}, max_deviation={max_deviation:.3e}, "
                        f"tolerance={theta_uniform_tol:.3e}"
                    )
            theta_logical = np.asarray(project_runtime_theta_block_mean(theta_arr, layout), dtype=float)
        else:
            raise ValueError(
                f"ADAPT theta length mismatch for grouped_exact reconstruction: got {theta_arr.size}, "
                f"expected {layout.runtime_parameter_count} (runtime) or {layout.logical_parameter_count} (logical)."
            )
        executor = CompiledAnsatzExecutor(
            selected_ops,
            coefficient_tolerance=float(layout.coefficient_tolerance),
            ignore_identity=bool(layout.ignore_identity),
            sort_terms=(str(layout.term_order) == "sorted"),
            parameterization_mode="logical_shared",
            parameterization_layout=layout,
        )
        return executor.prepare_state(theta_logical, psi)
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        for block, op in zip(layout.blocks, selected_ops):
            if int(block.runtime_count) <= 0:
                continue
            psi = apply_exp_pauli_polynomial_termwise(
                psi,
                op.polynomial,
                theta_arr[block.runtime_start:block.runtime_stop],
                ignore_identity=bool(layout.ignore_identity),
                coefficient_tolerance=float(layout.coefficient_tolerance),
                sort_terms=(str(layout.term_order) == "sorted"),
            )
        return psi
    if int(theta_arr.size) == int(len(selected_ops)):
        for k, op in enumerate(selected_ops):
            psi = apply_exp_pauli_polynomial(psi, op.polynomial, float(theta_arr[k]))
        return psi
    raise ValueError(
        f"ADAPT theta length mismatch: got {theta_arr.size}, expected {layout.runtime_parameter_count} (runtime) or {len(selected_ops)} (logical)."
    )


def _logical_theta_alias(theta: np.ndarray, layout: AnsatzParameterLayout) -> np.ndarray:
    theta_arr = np.asarray(theta, dtype=float).reshape(-1)
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        return np.asarray(project_runtime_theta_block_mean(theta_arr, layout), dtype=float)
    if int(theta_arr.size) == int(layout.logical_parameter_count):
        return np.asarray(theta_arr, dtype=float)
    raise ValueError(
        f"Cannot project theta of length {theta_arr.size} onto logical blocks of size {layout.logical_parameter_count}."
    )


def _adapt_energy_fn(
    h_poly: Any,
    psi_ref: np.ndarray,
    selected_ops: list[AnsatzTerm],
    theta: np.ndarray,
    *,
    h_compiled: CompiledPolynomialAction | None = None,
    parameter_layout: AnsatzParameterLayout | None = None,
) -> float:
    """Energy of the current ADAPT ansatz at parameters theta."""
    psi = _prepare_adapt_state(psi_ref, selected_ops, theta, parameter_layout=parameter_layout)
    if h_compiled is not None:
        energy, _hpsi = energy_via_one_apply(psi, h_compiled)
        return float(energy)
    return float(expval_pauli_polynomial(psi, h_poly))


_VALID_REOPT_POLICIES = {"append_only", "full", "windowed"}
_VALID_ADAPT_INNER_OPTIMIZERS = frozenset({"BFGS", "COBYLA", "POWELL", "ROTOSOLVE", "SPSA", "QNSPSA"})


def _scipy_adapt_heartbeat_event(method_key: str) -> str:
    method = str(method_key).strip().upper()
    if method == "COBYLA":
        return "hardcoded_adapt_cobyla_heartbeat"
    return "hardcoded_adapt_scipy_heartbeat"


def _scipy_adapt_optimizer_options(
    *,
    method_key: str,
    maxiter: int,
    maxfev: int | None = None,
) -> dict[str, Any]:
    method = str(method_key).strip().upper()
    options: dict[str, Any] = {"maxiter": int(maxiter)}
    if method == "BFGS":
        options["gtol"] = 1e-8
    if method == "COBYLA":
        options["rhobeg"] = 0.3
    if method == "POWELL":
        options["xtol"] = 1e-4
        options["ftol"] = 1e-8
        if maxfev is not None and int(maxfev) > 0:
            options["maxfev"] = int(maxfev)
    return options


def _run_scipy_adapt_optimizer(
    *,
    method_key: str,
    objective: Any,
    x0: np.ndarray,
    maxiter: int,
    maxfev: int | None = None,
    context_label: str,
    scipy_minimize_fn: Any,
) -> Any:
    if scipy_minimize_fn is None:
        raise RuntimeError(f"SciPy minimize is unavailable for {method_key} {context_label}.")
    return scipy_minimize_fn(
        objective,
        x0,
        method=str(method_key),
        options=_scipy_adapt_optimizer_options(
            method_key=str(method_key),
            maxiter=int(maxiter),
            maxfev=maxfev,
        ),
    )


def _run_rotosolve_adapt_optimizer(
    *,
    objective: Any,
    x0: np.ndarray,
    maxiter: int,
    context_label: str,
    callback: Any | None = None,
    period: Any | None = None,
    shift: Any | None = None,
) -> Any:
    del context_label
    if (period is None) ^ (shift is None):
        raise ValueError("ROTOSOLVE period and shift must be provided together.")
    has_explicit_stencil = period is not None and shift is not None
    if not has_explicit_stencil:
        allow_unstenciled = os.environ.get(
            STATIC_ADAPT_ALLOW_UNSTENCILED_ROTOSOLVE_ENV,
            "",
        ).strip().lower() in {"1", "true", "yes", "on"}
        if not allow_unstenciled:
            raise ValueError(
                "SNAKE ROTOSOLVE requires explicit coefficient-aware period/shift "
                "stencils for the selected coordinates. The native ADAPT wrapper no "
                "longer falls back silently to the generic [2*pi, pi/2] stencil. Set "
                f"{STATIC_ADAPT_ALLOW_UNSTENCILED_ROTOSOLVE_ENV}=1 only for explicitly "
                "labelled diagnostic fixed-stencil coordinate-descent runs."
            )
        period = 2.0 * math.pi
        shift = 0.5 * math.pi
    return rotosolve_coordinate_descent(
        objective,
        np.asarray(x0, dtype=float).reshape(-1),
        maxiter=int(maxiter),
        callback=callback,
        period=period,
        shift=shift,
    )


def _resolve_reopt_active_indices(
    *,
    policy: str,
    n: int,
    theta: np.ndarray,
    window_size: int = 3,
    window_topk: int = 0,
    periodic_full_refit_triggered: bool = False,
) -> tuple[list[int], str]:
    """Return (sorted_active_indices, effective_policy_name).

    Active-index selection contract (windowed):
      1. w_eff = min(window_size, n)
      2. newest = [n - w_eff, ..., n - 1]
      3. older  = [0, ..., n - w_eff - 1]
      4. If window_topk > 0, rank older by descending |theta[i]|,
         break ties by ascending index i.
      5. k_eff = min(window_topk, len(older))
      6. active = union(newest, top-k older)
      7. return sorted ascending

    For append_only: active = [n - 1]
    For full or periodic full-refit override: active = [0 .. n-1]
    """
    policy_key = str(policy).strip().lower()
    if n <= 0:
        return [], policy_key

    if policy_key == "append_only":
        return [n - 1], "append_only"

    if policy_key == "full":
        return list(range(n)), "full"

    if policy_key != "windowed":
        raise ValueError(f"Unknown reopt policy '{policy_key}'.")

    if periodic_full_refit_triggered:
        return list(range(n)), "windowed_periodic_full"

    w_eff = min(int(window_size), n)
    newest = list(range(n - w_eff, n))

    older_start = n - w_eff
    if older_start <= 0 or int(window_topk) <= 0:
        return sorted(newest), "windowed"

    older_candidates = list(range(0, older_start))
    older_ranked = sorted(
        older_candidates,
        key=lambda i: (-abs(float(theta[i])), i),
    )
    k_eff = min(int(window_topk), len(older_ranked))
    selected_older = older_ranked[:k_eff]
    active = sorted(set(newest) | set(selected_older))
    return active, "windowed"


def _make_reduced_parameter_expander(
    full_theta: np.ndarray,
    active_indices: list[int],
) -> tuple[Any, np.ndarray]:
    """Build a reduced-to-full theta expander and reduced initial point."""
    frozen_theta = np.array(full_theta, dtype=float, copy=True).reshape(-1)
    active_idx = [int(i) for i in active_indices]
    n_full = int(frozen_theta.size)
    if len(set(active_idx)) != len(active_idx):
        raise ValueError("active_indices must be unique")
    for idx in active_idx:
        if int(idx) < 0 or int(idx) >= n_full:
            raise ValueError("active_indices entries must be in range for full_theta")
    full_identity_active = bool(active_idx == list(range(n_full)))
    x0 = np.array([float(frozen_theta[i]) for i in active_idx], dtype=float)

    def _expand(x_active: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x_active, dtype=float).ravel()
        if full_identity_active:
            if int(x_arr.size) != n_full:
                raise ValueError(
                    f"reduced theta length mismatch: got {int(x_arr.size)}, expected {n_full}"
                )
            return np.asarray(x_arr, dtype=float).copy()
        if int(x_arr.size) != len(active_idx):
            raise ValueError(
                f"reduced theta length mismatch: got {int(x_arr.size)}, expected {len(active_idx)}"
            )
        full = np.array(frozen_theta, copy=True)
        for k, idx in enumerate(active_idx):
            full[int(idx)] = float(x_arr[int(k)])
        return full

    return _expand, x0


def _make_logical_shared_reduced_parameter_expander(
    full_theta: np.ndarray,
    layout: AnsatzParameterLayout,
    active_logical_indices: list[int],
) -> tuple[Any, np.ndarray]:
    """Build a reduced logical-coordinate expander for shared generators.

    Static ADAPT keeps an expanded runtime vector in artifacts so every Pauli
    factor remains explicitly replayable.  That storage layout must not become
    the optimizer chart: otherwise Powell sees one redundant flat coordinate
    per Pauli factor even though execution projects the whole block to one
    mean angle.  This helper optimizes exactly one coordinate per selected
    logical generator and expands the result to uniform runtime blocks only at
    the objective/execution boundary.
    """

    theta_arr = np.asarray(full_theta, dtype=float).reshape(-1)
    if int(theta_arr.size) == int(layout.runtime_parameter_count):
        frozen_logical = np.asarray(
            project_runtime_theta_block_mean(theta_arr, layout),
            dtype=float,
        )
    elif int(theta_arr.size) == int(layout.logical_parameter_count):
        frozen_logical = np.asarray(theta_arr, dtype=float).copy()
    else:
        raise ValueError(
            "logical-shared theta length mismatch: got "
            f"{theta_arr.size}, expected {layout.runtime_parameter_count} "
            f"(runtime) or {layout.logical_parameter_count} (logical)."
        )
    expand_logical, x0 = _make_reduced_parameter_expander(
        frozen_logical,
        [int(index) for index in active_logical_indices],
    )

    def _expand(x_active: np.ndarray) -> np.ndarray:
        logical_theta = np.asarray(
            expand_logical(np.asarray(x_active, dtype=float)),
            dtype=float,
        )
        return np.asarray(
            expand_legacy_logical_theta(logical_theta, layout),
            dtype=float,
        )

    return _expand, np.asarray(x0, dtype=float)


def _make_logical_shared_reduced_objective(
    full_theta: np.ndarray,
    layout: AnsatzParameterLayout,
    active_logical_indices: list[int],
    obj_fn: Any,
) -> tuple[Any, np.ndarray, Any]:
    """Return a logical reduced objective, initial point, and runtime lift."""

    expand_active, x0 = _make_logical_shared_reduced_parameter_expander(
        full_theta,
        layout,
        active_logical_indices,
    )

    def _reduced(x_active: np.ndarray) -> float:
        return float(obj_fn(expand_active(np.asarray(x_active, dtype=float))))

    return _reduced, np.asarray(x0, dtype=float), expand_active


def make_selected_optimizer_chart(
    *,
    full_theta: np.ndarray,
    layout: AnsatzParameterLayout,
    active_logical_indices: Sequence[int],
    objective: Callable[[np.ndarray], float],
    parameterization_mode: str,
    effective_optimizer_key: str,
    powell_coordinate_chart_policy: str,
    logical_alias_atol: float = 1.0e-10,
) -> SelectedOptimizerChart:
    """Build the optimizer chart selected by execution mode and optimizer.

    The expanded runtime vector remains the storage/execution boundary.  For
    ``logical_shared`` plus effective ``POWELL``, the explicit policy selects
    either the historical expanded runtime chart with block-mean projection at
    every objective/lift boundary, or the newer one-coordinate-per-logical-
    generator chart.  Every non-Powell combination retains the existing
    runtime chart.

    The reduced chart accepts a logical-shared runtime vector only when every
    generator block is already a uniform alias.  The expanded historical chart
    intentionally preserves its source behavior: Powell sees every active
    runtime coordinate and execution sees the blockwise mean projection.
    """

    active_logical = tuple(int(index) for index in active_logical_indices)
    if len(set(active_logical)) != len(active_logical):
        raise ValueError("active_logical_indices must be unique")
    logical_count = int(layout.logical_parameter_count)
    for index in active_logical:
        if index < 0 or index >= logical_count:
            raise ValueError(
                "active_logical_indices entries must be in range for layout: "
                f"got {index}, expected 0 <= index < {logical_count}"
            )

    alias_atol = float(logical_alias_atol)
    if not math.isfinite(alias_atol) or alias_atol < 0.0:
        raise ValueError("logical_alias_atol must be finite and non-negative")

    mode_key = str(parameterization_mode).strip().lower()
    optimizer_key = str(effective_optimizer_key).strip().upper()
    powell_chart_policy_key = str(powell_coordinate_chart_policy).strip().lower()
    if powell_chart_policy_key not in SR_POWELL_COORDINATE_CHART_POLICY_CHOICES:
        raise ValueError(
            "powell_coordinate_chart_policy must be one of "
            f"{list(SR_POWELL_COORDINATE_CHART_POLICY_CHOICES)}."
        )
    expanded_runtime_powell_active = bool(
        mode_key == "logical_shared"
        and optimizer_key == "POWELL"
        and powell_chart_policy_key
        == SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    reduced_logical_powell_active = bool(
        mode_key == "logical_shared"
        and optimizer_key == "POWELL"
        and powell_chart_policy_key
        == SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    theta_input = np.asarray(full_theta, dtype=float).reshape(-1)
    runtime_count = int(layout.runtime_parameter_count)

    logical_theta: np.ndarray | None = None
    if mode_key == "logical_shared":
        if int(theta_input.size) == runtime_count:
            theta_runtime = np.asarray(theta_input, dtype=float).copy()
            if not expanded_runtime_powell_active:
                logical_theta = np.zeros(logical_count, dtype=float)
                for block in layout.blocks:
                    start = int(block.runtime_start)
                    stop = int(block.runtime_stop)
                    values = np.asarray(theta_input[start:stop], dtype=float)
                    if int(values.size) == 0:
                        logical_theta[int(block.logical_index)] = 0.0
                        continue
                    anchor = float(values[0])
                    max_deviation = float(np.max(np.abs(values - anchor)))
                    if not math.isfinite(max_deviation) or max_deviation > alias_atol:
                        raise ValueError(
                            "logical_shared optimizer chart requires uniform runtime "
                            "blocks; refusing to average nonuniform block "
                            f"logical_index={int(block.logical_index)}, "
                            f"label={block.candidate_label!r}, "
                            f"max_deviation={max_deviation:.3e}, "
                            f"tolerance={alias_atol:.3e}"
                        )
                    logical_theta[int(block.logical_index)] = anchor
        elif int(theta_input.size) == logical_count:
            logical_theta = np.asarray(theta_input, dtype=float).copy()
            theta_runtime = np.asarray(
                expand_legacy_logical_theta(logical_theta, layout),
                dtype=float,
            )
        else:
            raise ValueError(
                "logical_shared optimizer chart theta length mismatch: got "
                f"{int(theta_input.size)}, expected {runtime_count} (runtime) "
                f"or {logical_count} (logical)"
            )
    else:
        if int(theta_input.size) != runtime_count:
            raise ValueError(
                "runtime optimizer chart theta length mismatch: got "
                f"{int(theta_input.size)}, expected {runtime_count}"
            )
        theta_runtime = np.asarray(theta_input, dtype=float).copy()

    active_runtime = tuple(
        int(index)
        for index in runtime_indices_for_logical_indices(layout, active_logical)
    )

    if reduced_logical_powell_active:
        assert logical_theta is not None
        for logical_index in active_logical:
            block = layout.blocks[int(logical_index)]
            if int(block.runtime_count) <= 0:
                raise ValueError(
                    "logical_shared Powell chart cannot optimize an empty "
                    f"runtime block at logical_index={logical_index}, "
                    f"label={block.candidate_label!r}"
                )
        expand_logical, x0 = _make_reduced_parameter_expander(
            logical_theta,
            list(active_logical),
        )

        def _lift_logical(x_active: np.ndarray) -> np.ndarray:
            full_logical = np.asarray(
                expand_logical(np.asarray(x_active, dtype=float)),
                dtype=float,
            )
            return np.asarray(
                expand_legacy_logical_theta(full_logical, layout),
                dtype=float,
            )

        def _logical_objective(x_active: np.ndarray) -> float:
            return float(objective(_lift_logical(np.asarray(x_active, dtype=float))))

        return SelectedOptimizerChart(
            objective=_logical_objective,
            x0=np.asarray(x0, dtype=float),
            lift_to_runtime=_lift_logical,
            coordinate_mode="logical_shared",
            active_logical_indices=active_logical,
            active_runtime_indices=active_runtime,
            active_optimizer_indices=active_logical,
            reduced_positions_by_logical={
                int(logical_index): (int(position),)
                for position, logical_index in enumerate(active_logical)
            },
        )

    lift_runtime_raw, x0 = _make_reduced_parameter_expander(
        theta_runtime,
        list(active_runtime),
    )

    def _lift_runtime(x_active: np.ndarray) -> np.ndarray:
        runtime_theta = np.asarray(
            lift_runtime_raw(np.asarray(x_active, dtype=float)),
            dtype=float,
        )
        if mode_key != "logical_shared":
            return runtime_theta
        logical_theta_now = np.asarray(
            project_runtime_theta_block_mean(runtime_theta, layout),
            dtype=float,
        )
        return np.asarray(
            expand_legacy_logical_theta(logical_theta_now, layout),
            dtype=float,
        )

    def _runtime_objective(x_active: np.ndarray) -> float:
        return float(objective(_lift_runtime(np.asarray(x_active, dtype=float))))

    reduced_positions_by_logical: dict[int, tuple[int, ...]] = {}
    reduced_position = 0
    for logical_index in active_logical:
        runtime_size = int(layout.blocks[int(logical_index)].runtime_count)
        reduced_positions_by_logical[int(logical_index)] = tuple(
            range(reduced_position, reduced_position + runtime_size)
        )
        reduced_position += runtime_size

    return SelectedOptimizerChart(
        objective=_runtime_objective,
        x0=np.asarray(x0, dtype=float),
        lift_to_runtime=_lift_runtime,
        coordinate_mode="runtime",
        active_logical_indices=active_logical,
        active_runtime_indices=active_runtime,
        active_optimizer_indices=active_runtime,
        reduced_positions_by_logical=reduced_positions_by_logical,
    )


def _guard_sr_active_only_step(
    *,
    chart: SelectedOptimizerChart,
    active_parameter_step: Sequence[float],
    retained_joint_step_candidates: Sequence[Sequence[float]] | None = None,
    retained_candidate_predicted_reductions: Sequence[float | None] | None = None,
    retained_candidate_roles: Sequence[str] | None = None,
    candidate_coordinate_count: int = 0,
    candidate_block_zero_tolerance: float = 1.0e-12,
    guard_abs_tol: float = 1.0e-12,
    guard_rel_tol: float = 1.0e-12,
    nonlinear_backtracking_enabled: bool = False,
    state_at_optimizer_x: Callable[[np.ndarray], np.ndarray] | None = None,
    incumbent_state: np.ndarray | None = None,
    max_exact_fubini_study_distance: float | None = None,
    quadratic_model_gradient: Sequence[float] | None = None,
    quadratic_model_hessian: Sequence[Sequence[float]] | None = None,
    backtracking_max_halvings: int | None = None,
) -> tuple[np.ndarray, dict[str, Any], int]:
    """Atomically map and guard retained SR active-restriction candidates.

    ``active_parameter_step`` is ordered exactly like
    ``chart.active_logical_indices``.  A logical step is copied to every
    runtime factor in a per-Pauli chart and applied once in a shared-logical
    chart.  A hard-case or uncertainty-reflection certificate may retain more
    than one full joint candidate.  All such candidates are mapped before any
    exact evaluation and all exact evaluations must be finite before a
    downhill candidate can be selected.  The optional nonlinear guard retries
    the complete retained set at dyadic fractions, largest first, and accepts
    only an exactly downhill endpoint inside the branch's exact physical
    Fubini--Study radius.
    """

    active_step = np.asarray(active_parameter_step, dtype=float).reshape(-1)
    active_logical = tuple(int(value) for value in chart.active_logical_indices)
    if int(active_step.size) != int(len(active_logical)):
        raise ValueError(
            "SR active-only step length must match the optimizer chart's "
            "active logical coordinates."
        )
    if not np.all(np.isfinite(active_step)):
        raise ValueError("SR active-only step must be finite.")
    abs_tol = float(guard_abs_tol)
    rel_tol = float(guard_rel_tol)
    if not math.isfinite(abs_tol) or abs_tol < 0.0:
        raise ValueError("guard_abs_tol must be finite and nonnegative.")
    if not math.isfinite(rel_tol) or rel_tol < 0.0:
        raise ValueError("guard_rel_tol must be finite and nonnegative.")
    candidate_count = int(candidate_coordinate_count)
    if candidate_count < 0:
        raise ValueError("candidate_coordinate_count must be nonnegative.")
    batch_zero_tolerance = float(candidate_block_zero_tolerance)
    if not math.isfinite(batch_zero_tolerance) or batch_zero_tolerance < 0.0:
        raise ValueError(
            "candidate_block_zero_tolerance must be finite and nonnegative."
        )

    incumbent = np.asarray(chart.x0, dtype=float).reshape(-1).copy()
    mapped_groups: list[list[int]] = []
    for logical_index in active_logical:
        group = tuple(
            int(position)
            for position in chart.reduced_positions_by_logical.get(
                int(logical_index), ()
            )
        )
        if not group:
            raise ValueError(
                "SR active-only optimizer chart is missing an active logical "
                f"coordinate: {logical_index}."
            )
        if any(position < 0 or position >= int(incumbent.size) for position in group):
            raise ValueError(
                "SR active-only optimizer chart contains an out-of-range "
                "reduced position."
            )
        mapped_groups.append([int(position) for position in group])

    fallback_joint_step = [
        *[float(value) for value in active_step.tolist()],
        *(0.0 for _ in range(candidate_count)),
    ]
    retained = (
        [fallback_joint_step]
        if retained_joint_step_candidates is None
        else [list(value) for value in retained_joint_step_candidates]
    )

    def _map_joint_step(
        joint_step: Sequence[float],
    ) -> tuple[np.ndarray | None, Mapping[str, Any]]:
        step = np.asarray(joint_step, dtype=float).reshape(-1)
        expected = int(len(active_logical) + candidate_count)
        if int(step.size) != expected or not np.all(np.isfinite(step)):
            return None, {
                "status": "unavailable",
                "reason": "active_only_joint_coordinate_count_mismatch",
                "expected_joint_coordinate_count": int(expected),
                "received_joint_coordinate_count": int(step.size),
            }
        batch_step = np.asarray(step[len(active_logical) :], dtype=float)
        batch_norm = float(np.linalg.norm(batch_step))
        if batch_norm > batch_zero_tolerance:
            return None, {
                "status": "unavailable",
                "reason": "active_only_candidate_block_not_zero",
                "candidate_block_norm": float(batch_norm),
                "candidate_block_zero_tolerance": float(batch_zero_tolerance),
            }
        proposal = np.asarray(incumbent, dtype=float).copy()
        for delta, group in zip(step[: len(active_logical)], mapped_groups):
            for position in group:
                proposal[int(position)] += float(delta)
        if not np.all(np.isfinite(proposal)):
            return None, {
                "status": "unavailable",
                "reason": "nonfinite_mapped_active_only_step",
            }
        return proposal, {
            "status": "mapped",
            "reason": "active_only_joint_step_mapped",
            "active_reduced_position_groups": copy.deepcopy(mapped_groups),
            "active_parameter_step": [
                float(value) for value in step[: len(active_logical)].tolist()
            ],
            "candidate_coordinate_step": [
                float(value) for value in batch_step.tolist()
            ],
            "candidate_block_norm": float(batch_norm),
            "candidate_block_zero_tolerance": float(batch_zero_tolerance),
        }

    backtracking_enabled = bool(nonlinear_backtracking_enabled)
    maximum_halvings = (
        int(np.finfo(float).nmant)
        if backtracking_max_halvings is None
        else int(backtracking_max_halvings)
    )
    if maximum_halvings < 0 or maximum_halvings > int(np.finfo(float).nmant):
        raise ValueError(
            "backtracking_max_halvings must lie between zero and the float64 "
            "significand width."
        )
    endpoint_distance_budget: float | None = None
    endpoint_distance_service: Callable[[np.ndarray], float] | None = None
    model_gradient: np.ndarray | None = None
    model_hessian: np.ndarray | None = None
    expected_joint_coordinate_count = int(len(active_logical) + candidate_count)
    retained_arrays: list[np.ndarray] = []
    for value in retained:
        try:
            candidate = np.asarray(value, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            candidate = np.zeros(0, dtype=float)
        retained_arrays.append(candidate)
    if backtracking_enabled:
        if state_at_optimizer_x is None or incumbent_state is None:
            raise ValueError(
                "SR active-only nonlinear backtracking requires an exact "
                "endpoint-state service and the incumbent state."
            )
        endpoint_distance_budget = float(
            max_exact_fubini_study_distance
            if max_exact_fubini_study_distance is not None
            else float("nan")
        )
        if (
            not math.isfinite(endpoint_distance_budget)
            or endpoint_distance_budget <= 0.0
        ):
            raise ValueError(
                "SR active-only nonlinear backtracking requires a finite "
                "positive exact Fubini--Study radius."
            )
        incumbent_state_value = np.asarray(incumbent_state, dtype=complex).copy()

        def _endpoint_distance_service(point: np.ndarray) -> float:
            endpoint_state = np.asarray(
                state_at_optimizer_x(np.asarray(point, dtype=float)),
                dtype=complex,
            )
            return float(
                exact_fubini_study_distance(
                    incumbent_state_value,
                    endpoint_state,
                )
            )

        endpoint_distance_service = _endpoint_distance_service
        model_gradient = np.asarray(
            quadratic_model_gradient, dtype=float
        ).reshape(-1)
        model_hessian = np.asarray(
            quadratic_model_hessian, dtype=float
        )
        if (
            model_gradient.size != expected_joint_coordinate_count
            or model_hessian.shape
            != (
                expected_joint_coordinate_count,
                expected_joint_coordinate_count,
            )
            or not np.all(np.isfinite(model_gradient))
            or not np.all(np.isfinite(model_hessian))
            or any(
                candidate.size != expected_joint_coordinate_count
                or not np.all(np.isfinite(candidate))
                for candidate in retained_arrays
            )
        ):
            raise ValueError(
                "SR active-only nonlinear backtracking requires a finite "
                "full-joint quadratic model matching every retained candidate."
            )
        model_hessian = 0.5 * (model_hessian + model_hessian.T)

    backtracking_attempts: list[dict[str, Any]] = []
    total_guard_nfev = 0
    total_endpoint_state_evals = 0
    cached_incumbent_energy: float | None = None
    selected = np.asarray(incumbent, dtype=float).copy()
    atomic_payload: dict[str, Any] = {
        "status": "unavailable",
        "reason": "active_only_guard_not_attempted",
    }
    coordinate_resolution_exhausted = False
    exponents = range(maximum_halvings + 1) if backtracking_enabled else range(1)
    for exponent in exponents:
        scale = float(math.ldexp(1.0, -int(exponent)))
        if backtracking_enabled:
            scaled_candidates = [
                [float(item) for item in (scale * candidate).tolist()]
                for candidate in retained_arrays
            ]
            scaled_predictions = [
                float(
                    model_gradient.T @ (scale * candidate)
                    - 0.5
                    * (scale * candidate).T
                    @ model_hessian
                    @ (scale * candidate)
                )
                for candidate in retained_arrays
            ]
        else:
            scaled_candidates = retained
            scaled_predictions = retained_candidate_predicted_reductions

        selected_now, payload_now, nfev_now = guard_atomic_joint_step_candidates(
            objective=chart.objective,
            canonical_x0=incumbent,
            joint_steps=scaled_candidates,
            map_joint_step=_map_joint_step,
            predicted_reductions=scaled_predictions,
            candidate_roles=retained_candidate_roles,
            guard_abs_tol=float(abs_tol),
            guard_rel_tol=float(rel_tol),
            incumbent_energy=cached_incumbent_energy,
            endpoint_distance_from_incumbent=endpoint_distance_service,
            max_endpoint_distance=endpoint_distance_budget,
        )
        total_guard_nfev += int(nfev_now)
        total_endpoint_state_evals += int(
            payload_now.get("guard_endpoint_state_evals", 0)
        )
        incumbent_energy_now = payload_now.get("mapped_seed_incumbent_energy")
        if (
            cached_incumbent_energy is None
            and incumbent_energy_now is not None
            and math.isfinite(float(incumbent_energy_now))
        ):
            cached_incumbent_energy = float(incumbent_energy_now)
        scaled_evaluations: list[dict[str, Any]] = []
        evaluations_now = payload_now.get("candidate_evaluations", [])
        if isinstance(evaluations_now, Sequence) and not isinstance(
            evaluations_now, (str, bytes, bytearray)
        ):
            for record in evaluations_now:
                if not isinstance(record, Mapping):
                    continue
                scaled_evaluations.append(
                    {
                        **dict(record),
                        "backtracking_halvings": int(exponent),
                        "joint_step_scale": float(scale),
                    }
                )
        payload_now = {
            **dict(payload_now),
            "candidate_evaluations": scaled_evaluations,
            "backtracking_halvings": int(exponent),
            "joint_step_scale": float(scale),
        }
        backtracking_attempts.append(
            {
                "backtracking_halvings": int(exponent),
                "joint_step_scale": float(scale),
                "status": str(payload_now.get("status", "unavailable")),
                "reason": str(payload_now.get("reason", "unavailable")),
                "transaction_failure_kind": payload_now.get(
                    "transaction_failure_kind"
                ),
                "candidate_evaluations": copy.deepcopy(scaled_evaluations),
            }
        )
        atomic_payload = dict(payload_now)
        selected = np.asarray(selected_now, dtype=float).copy()
        if str(atomic_payload.get("status", "")) == "accepted":
            break
        if atomic_payload.get("transaction_failure_kind") not in {None, ""}:
            break
        if not backtracking_enabled:
            break
        mapped_points = [
            np.asarray(record.get("mapped_optimizer_x", []), dtype=float)
            for record in scaled_evaluations
            if record.get("mapped_optimizer_x") is not None
        ]
        if mapped_points and all(
            point.shape == incumbent.shape and np.array_equal(point, incumbent)
            for point in mapped_points
        ):
            coordinate_resolution_exhausted = True
            break

    guard_accepted = bool(str(atomic_payload.get("status", "")) == "accepted")
    empty_retained_candidate_contract = bool(
        backtracking_enabled and not retained
    )
    if empty_retained_candidate_contract:
        atomic_payload = {
            **dict(atomic_payload),
            "status": "unavailable",
            "reason": "active_only_backtracking_missing_retained_candidates",
            "transaction_failure_kind": "mapping_contract",
            "trust_action": "hold",
            "sr_transaction_outcome": "no_state_refinement_trust_hold",
            "fallback_to_incumbent": True,
            "no_state_transition": True,
        }
    finite_backtracking_exhausted = bool(
        backtracking_enabled
        and not empty_retained_candidate_contract
        and not guard_accepted
        and atomic_payload.get("transaction_failure_kind") in {None, ""}
    )
    if finite_backtracking_exhausted:
        atomic_payload = {
            **dict(atomic_payload),
            "status": "rejected",
            "reason": "active_only_nonlinear_backtracking_exhausted",
            "transaction_failure_kind": "finite_nonlinear_model_disagreement",
            "trust_action": "contract_branch_radius",
            "sr_transaction_outcome": (
                "branch_local_radius_refinement_required"
            ),
            "fallback_to_incumbent": True,
            "selected_candidate_index": None,
            "selected_candidate_role": None,
            "mapped_seed_exact_gain": None,
            "mapped_seed_predicted_reduction": None,
            "no_state_transition": True,
            "nonlinear_backtracking_exhausted": True,
            "all_backtracking_candidates_finite": True,
            "optimizer_coordinate_resolution_exhausted": bool(
                coordinate_resolution_exhausted
            ),
        }
        selected = np.asarray(incumbent, dtype=float).copy()
    else:
        atomic_payload = {
            **dict(atomic_payload),
            "nonlinear_backtracking_exhausted": False,
            "all_backtracking_candidates_finite": bool(
                guard_accepted
                and atomic_payload.get("transaction_failure_kind") in {None, ""}
            ),
            "optimizer_coordinate_resolution_exhausted": False,
        }
    atomic_payload.update(
        {
            "nonlinear_backtracking_enabled": bool(backtracking_enabled),
            "backtracking_policy": (
                "largest_first_dyadic_exact_energy_and_endpoint_fs_v1"
                if backtracking_enabled
                else "disabled"
            ),
            "backtracking_max_halvings": int(maximum_halvings),
            "backtracking_attempt_count": int(len(backtracking_attempts)),
            "backtracking_applied": bool(
                guard_accepted
                and int(atomic_payload.get("backtracking_halvings", 0)) > 0
            ),
            "applied_joint_step_scale": (
                float(atomic_payload.get("joint_step_scale", 1.0))
                if guard_accepted
                else None
            ),
            "quadratic_prediction_authority": (
                "scaled_full_joint_quadratic_model_v1"
                if backtracking_enabled
                else "caller_retained_candidate_prediction"
            ),
            "backtracking_attempts": copy.deepcopy(backtracking_attempts),
            "guard_objective_evals": int(total_guard_nfev),
            "guard_endpoint_state_evals": int(total_endpoint_state_evals),
        }
    )
    nfev = int(total_guard_nfev)
    selected_index = atomic_payload.get("selected_candidate_index")
    selected_record = None
    if isinstance(selected_index, int):
        evaluations = atomic_payload.get("candidate_evaluations", [])
        if (
            isinstance(evaluations, Sequence)
            and not isinstance(evaluations, (str, bytes, bytearray))
            and 0 <= int(selected_index) < len(evaluations)
            and isinstance(evaluations[int(selected_index)], Mapping)
        ):
            selected_record = dict(evaluations[int(selected_index)])
    incumbent_energy = atomic_payload.get("mapped_seed_incumbent_energy")
    selected_energy = (
        None if selected_record is None else selected_record.get("energy")
    )
    if selected_energy is None:
        selected_energy = incumbent_energy
    mapped_optimizer_x = (
        None if selected_record is None else selected_record.get("mapped_optimizer_x")
    )
    applied_joint_step = (
        None if selected_record is None else selected_record.get("joint_step")
    )
    applied_active_parameter_step = (
        None
        if not isinstance(applied_joint_step, Sequence)
        or isinstance(applied_joint_step, (str, bytes, bytearray))
        else [
            float(value)
            for value in list(applied_joint_step)[: len(active_logical)]
        ]
    )
    payload = {
        **dict(atomic_payload),
        "schema": "sr_active_only_exact_seed_guard_v3",
        "optimizer_coordinate_mode": str(chart.coordinate_mode),
        "active_logical_indices": [int(value) for value in active_logical],
        "active_reduced_position_groups": copy.deepcopy(mapped_groups),
        "active_parameter_step": [float(value) for value in active_step.tolist()],
        "proposed_active_parameter_step": [
            float(value) for value in active_step.tolist()
        ],
        "applied_active_parameter_step": applied_active_parameter_step,
        "applied_joint_step": (
            None
            if applied_joint_step is None
            else [float(value) for value in applied_joint_step]
        ),
        "candidate_coordinate_count": int(candidate_count),
        "incumbent_optimizer_x": [float(value) for value in incumbent.tolist()],
        "mapped_optimizer_x": mapped_optimizer_x,
        "selected_optimizer_x": [float(value) for value in selected.tolist()],
        "incumbent_energy": incumbent_energy,
        "mapped_seed_energy": selected_energy,
        "selected_seed_energy": selected_energy,
        "mapped_seed_materially_downhill": bool(
            atomic_payload.get("status") == "accepted"
        ),
    }
    return np.asarray(selected, dtype=float).copy(), payload, int(nfev)


def _retain_sr_active_only_refit_outcome(
    *,
    incumbent_x: np.ndarray,
    incumbent_energy: float,
    guarded_seed_x: np.ndarray,
    guarded_seed_energy: float,
    optimizer_x: np.ndarray,
    optimizer_energy: float,
    guard_abs_tol: float = 1.0e-12,
    guard_rel_tol: float = 1.0e-12,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    """Select a finite exact active-only outcome without worsening incumbent."""

    incumbent = np.asarray(incumbent_x, dtype=float).reshape(-1).copy()
    seed = np.asarray(guarded_seed_x, dtype=float).reshape(-1).copy()
    optimized = np.asarray(optimizer_x, dtype=float).reshape(-1).copy()
    incumbent_e = float(incumbent_energy)
    seed_e = float(guarded_seed_energy)
    optimizer_e = float(optimizer_energy)
    if not math.isfinite(incumbent_e) or not np.all(np.isfinite(incumbent)):
        raise ValueError("SR active-only incumbent transaction input is nonfinite.")
    if seed.shape != incumbent.shape or not np.all(np.isfinite(seed)):
        seed_e = float("inf")
    if not math.isfinite(seed_e):
        seed_e = float("inf")
    if optimized.shape != incumbent.shape or not np.all(np.isfinite(optimized)):
        optimizer_e = float("inf")
    if not math.isfinite(optimizer_e):
        optimizer_e = float("inf")

    scale = max(
        1.0,
        abs(float(incumbent_e)),
        abs(float(seed_e)) if math.isfinite(seed_e) else 0.0,
        abs(float(optimizer_e)) if math.isfinite(optimizer_e) else 0.0,
    )
    tolerance = float(guard_abs_tol) + float(guard_rel_tol) * float(scale)
    selected_x = incumbent
    selected_energy = float(incumbent_e)
    selected_source = "incumbent"
    if math.isfinite(seed_e) and seed_e < selected_energy - tolerance:
        selected_x = seed
        selected_energy = float(seed_e)
        selected_source = "mapped_active_restriction_seed"
    if math.isfinite(optimizer_e) and optimizer_e < selected_energy - tolerance:
        selected_x = optimized
        selected_energy = float(optimizer_e)
        selected_source = "optimizer_result"
    if selected_energy > incumbent_e + tolerance:
        raise RuntimeError("SR active-only safe transaction worsened the incumbent.")

    payload = {
        "schema": "sr_active_only_safe_refit_outcome_v1",
        "incumbent_energy": float(incumbent_e),
        "guarded_seed_energy": (
            None if not math.isfinite(seed_e) else float(seed_e)
        ),
        "optimizer_energy": (
            None if not math.isfinite(optimizer_e) else float(optimizer_e)
        ),
        "selected_energy": float(selected_energy),
        "selected_source": str(selected_source),
        "comparison_tolerance": float(tolerance),
        "incumbent_retained": bool(selected_source == "incumbent"),
        "mapped_seed_retained": bool(
            selected_source == "mapped_active_restriction_seed"
        ),
        "optimizer_result_retained": bool(selected_source == "optimizer_result"),
        "optimizer_result_discarded": bool(selected_source != "optimizer_result"),
        "nonworsening_certified": bool(
            selected_energy <= incumbent_e + tolerance
        ),
        "state_selection_rule": (
            "lowest_materially_downhill_exact_state_among_incumbent_seed_refit_v1"
        ),
    }
    return (
        np.asarray(selected_x, dtype=float).copy(),
        float(selected_energy),
        payload,
    )


def _make_reduced_objective(
    full_theta: np.ndarray,
    active_indices: list[int],
    obj_fn: Any,
) -> tuple[Any, np.ndarray]:
    """Build a reduced-variable objective and its initial point.

    Returns (reduced_obj, x0_reduced) where:
      - reduced_obj(x_active) reconstructs a full theta from frozen+active
        and calls obj_fn(full_theta)
      - x0_reduced = full_theta[active_indices]
    """
    expand_active, x0 = _make_reduced_parameter_expander(full_theta, active_indices)
    frozen_theta = np.array(full_theta, dtype=float, copy=True).reshape(-1)
    active_idx = [int(i) for i in active_indices]

    if active_idx == list(range(int(frozen_theta.size))):
        return obj_fn, np.array(frozen_theta, copy=True)

    def _reduced(x_active: np.ndarray) -> float:
        return float(obj_fn(expand_active(x_active)))

    return _reduced, x0


def _resolve_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    problem_key = str(problem).strip().lower()
    try:
        default_mode = str(default_continuation_mode_for_problem(problem_key))
        allowed_modes = set(supported_continuation_modes_for_problem(problem_key))
    except ValueError:
        default_mode = "phase3_v1" if problem_key == "hh" else "legacy"
        allowed_modes = {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}
    if requested_mode is None:
        return str(default_mode)
    mode_raw = str(requested_mode).strip().lower()
    if mode_raw == "":
        return str(default_mode)
    if mode_raw not in allowed_modes:
        raise ValueError("adapt_continuation_mode must be one of {'legacy','phase1_v1','phase2_v1','phase3_v1'}.")
    return str(mode_raw)


def _resolve_cli_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    return _resolve_adapt_continuation_mode(problem=problem, requested_mode=requested_mode)
