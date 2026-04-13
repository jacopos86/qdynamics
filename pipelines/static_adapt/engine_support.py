#!/usr/bin/env python3
"""Standalone ADAPT engine helpers extracted from adapt_pipeline."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from src.quantum.ansatz_parameterization import (
    AnsatzParameterLayout,
    build_parameter_layout,
    project_runtime_theta_block_mean,
)
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    adapt_commutator_grad_from_hpsi,
    apply_compiled_polynomial as _apply_compiled_polynomial_shared,
    energy_via_one_apply,
)
from src.quantum.pauli_actions import CompiledPauliAction
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    apply_exp_pauli_polynomial,
    apply_exp_pauli_polynomial_termwise,
    apply_pauli_string,
    expval_pauli_polynomial,
)
from pipelines.hardcoded.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.hardcoded.hh_continuation_stage_control import StageController
from pipelines.hardcoded.hh_continuation_types import ScaffoldCoordinateMetadata


@dataclass
class AdaptVQEResult:
    """Result container for the hardcoded ADAPT-VQE run."""

    energy: float
    theta: np.ndarray
    selected_ops: list[AnsatzTerm]
    history: list[dict[str, Any]]
    stop_reason: str
    nfev_total: int


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
    phase1_last_retained_records: list[dict[str, Any]]
    phase2_optimizer_memory: dict[str, Any]
    phase2_last_shortlist_records: list[dict[str, Any]]
    phase2_last_geometric_shortlist_records: list[dict[str, Any]]
    phase2_last_retained_shortlist_records: list[dict[str, Any]]
    phase2_last_admitted_records: list[dict[str, Any]]
    phase2_last_batch_selected: bool
    phase2_last_batch_penalty_total: float
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
            phase1_last_retained_records=copy.deepcopy(self.phase1_last_retained_records),
            phase2_optimizer_memory=copy.deepcopy(self.phase2_optimizer_memory),
            phase2_last_shortlist_records=copy.deepcopy(self.phase2_last_shortlist_records),
            phase2_last_geometric_shortlist_records=copy.deepcopy(self.phase2_last_geometric_shortlist_records),
            phase2_last_retained_shortlist_records=copy.deepcopy(self.phase2_last_retained_shortlist_records),
            phase2_last_admitted_records=copy.deepcopy(self.phase2_last_admitted_records),
            phase2_last_batch_selected=bool(self.phase2_last_batch_selected),
            phase2_last_batch_penalty_total=float(self.phase2_last_batch_penalty_total),
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
    phase2_last_optimizer_memory_reused: bool
    phase2_last_optimizer_memory_source: str
    phase2_last_shortlist_eval_records: list[dict[str, Any]]
    phase1_residual_opened: bool
    available_indices_after_transition: set[int]
    phase1_stage_events_after_transition: list[dict[str, Any]]
    phase3_runtime_split_summary_after_eval: dict[str, Any]
    proposals: list[_BranchExpansionPlan]
    stop_reason: str | None
    fallback_scan_size: int
    fallback_best_probe_delta_e: float | None
    fallback_best_probe_theta: float | None


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
_VALID_ADAPT_INNER_OPTIMIZERS = frozenset({"COBYLA", "POWELL", "SPSA"})


def _scipy_adapt_heartbeat_event(method_key: str) -> str:
    method = str(method_key).strip().upper()
    if method == "COBYLA":
        return "hardcoded_adapt_cobyla_heartbeat"
    return "hardcoded_adapt_scipy_heartbeat"


def _scipy_adapt_optimizer_options(*, method_key: str, maxiter: int) -> dict[str, Any]:
    method = str(method_key).strip().upper()
    options: dict[str, Any] = {"maxiter": int(maxiter)}
    if method == "COBYLA":
        options["rhobeg"] = 0.3
    return options


def _run_scipy_adapt_optimizer(
    *,
    method_key: str,
    objective: Any,
    x0: np.ndarray,
    maxiter: int,
    context_label: str,
    scipy_minimize_fn: Any,
) -> Any:
    if scipy_minimize_fn is None:
        raise RuntimeError(f"SciPy minimize is unavailable for {method_key} {context_label}.")
    return scipy_minimize_fn(
        objective,
        x0,
        method=str(method_key),
        options=_scipy_adapt_optimizer_options(method_key=str(method_key), maxiter=int(maxiter)),
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
    frozen_theta = np.array(full_theta, copy=True)
    active_idx = list(active_indices)
    n_active = len(active_idx)
    x0 = np.array([float(frozen_theta[i]) for i in active_idx], dtype=float)

    if n_active == len(frozen_theta):
        return obj_fn, np.array(frozen_theta, copy=True)

    def _reduced(x_active: np.ndarray) -> float:
        full = np.array(frozen_theta, copy=True)
        x_arr = np.asarray(x_active, dtype=float).ravel()
        for k, idx in enumerate(active_idx):
            full[idx] = float(x_arr[k])
        return float(obj_fn(full))

    return _reduced, x0


def _resolve_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    problem_key = str(problem).strip().lower()
    if requested_mode is None:
        return "phase3_v1" if problem_key == "hh" else "legacy"
    mode_raw = str(requested_mode).strip().lower()
    if mode_raw == "":
        return "phase3_v1" if problem_key == "hh" else "legacy"
    if mode_raw not in {"legacy", "phase1_v1", "phase2_v1", "phase3_v1"}:
        raise ValueError("adapt_continuation_mode must be one of {'legacy','phase1_v1','phase2_v1','phase3_v1'}.")
    return str(mode_raw)


def _resolve_cli_adapt_continuation_mode(*, problem: str, requested_mode: str | None) -> str:
    return _resolve_adapt_continuation_mode(problem=problem, requested_mode=requested_mode)
