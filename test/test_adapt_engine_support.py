from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pipelines.static_adapt.adapt_pipeline as hardcoded_adapt
import pipelines.static_adapt.adapt_pipeline as static_adapt
import pipelines.static_adapt.beam_search as beam_search
import pipelines.static_adapt.cli_config as cli_config
import pipelines.static_adapt.engine_support as engine_support
from pipelines.static_adapt.extensions import BEAM_RUNTIME_KEYS
from pipelines.scaffold.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.scaffold.hh_continuation_stage_control import StageController, StageControllerConfig
from pipelines.static_adapt.selector_measurement_proxy import ControllerMeasurementWorkAccumulator
from pipelines.static_adapt.route_a_trust_region import RouteATrustRegionState



def _make_branch(*, branch_id: int, labels: tuple[str, ...], theta: tuple[float, ...]):
    return engine_support._BeamBranchState(
        branch_id=int(branch_id),
        parent_branch_id=None,
        depth_local=2,
        terminated=False,
        stop_reason=None,
        selected_ops=[SimpleNamespace(label=label) for label in labels],
        theta=np.asarray(theta, dtype=float),
        energy_current=-1.25,
        available_indices={0, 1},
        selection_counts=np.asarray([1, 0], dtype=np.int64),
        history=[],
        phase1_stage=StageController(StageControllerConfig()),
        phase1_residual_opened=False,
        phase1_last_probe_reason="none",
        phase1_last_positions_considered=[0],
        phase1_last_trough_detected=False,
        phase1_last_trough_probe_triggered=False,
        phase1_last_selected_score=None,
        phase1_features_history=[],
        phase1_stage_events=[],
        phase1_measure_cache=MeasurementCacheAudit(),
        controller_measurement_work=ControllerMeasurementWorkAccumulator(),
        phase1_last_retained_records=[],
        phase2_optimizer_memory={},
        phase2_last_shortlist_records=[],
        phase2_last_geometric_shortlist_records=[],
        phase2_last_retained_shortlist_records=[],
        phase2_last_admitted_records=[],
        phase2_last_batch_selected=False,
        phase2_last_batch_penalty_total=0.0,
        phase2_last_batch_schur_context={},
        phase2_last_optimizer_memory_reused=False,
        phase2_last_optimizer_memory_source="none",
        phase2_last_shortlist_eval_records=[],
        drop_prev_delta_abs=0.0,
        drop_plateau_hits=0,
        eps_energy_low_streak=0,
        phase3_split_events=[],
        phase3_runtime_split_summary={},
        phase3_motif_usage={},
        phase3_rescue_history=[],
        phase1_prune_metadata=[],
        phase1_prune_first_seen_steps={},
        phase1_last_prune_summary={},
        last_transition_kind="none",
        last_admission_record_count=0,
        cumulative_selector_score=2.5,
        cumulative_selector_burden=0.25,
        nfev_total_local=3,
    )




def _make_beam_scratch(**overrides: Any) -> engine_support._BranchStepScratch:
    values: dict[str, Any] = {
        "energy_current": -1.0,
        "psi_current": np.asarray([1.0], dtype=complex),
        "hpsi_current": np.asarray([0.0], dtype=complex),
        "gradients": np.asarray([], dtype=float),
        "grad_magnitudes": np.asarray([], dtype=float),
        "max_grad": 0.0,
        "gradient_eval_elapsed_s": 0.0,
        "append_position": 0,
        "best_idx": 0,
        "selected_position": 0,
        "selection_mode": "beam",
        "stage_name": "phase3",
        "phase1_feature_selected": None,
        "phase1_stage_transition_reason": "none",
        "phase1_stage_now": "phase3",
        "phase1_stage_after_transition": StageController(StageControllerConfig()),
        "phase1_last_probe_reason": "none",
        "phase1_last_positions_considered": [],
        "phase1_last_trough_detected": False,
        "phase1_last_trough_probe_triggered": False,
        "phase1_last_selected_score": None,
        "phase1_last_retained_records": [],
        "phase2_last_shortlist_records": [],
        "phase2_last_geometric_shortlist_records": [],
        "phase2_last_retained_shortlist_records": [],
        "phase2_last_admitted_records": [],
        "phase2_last_batch_selected": False,
        "phase2_last_batch_penalty_total": 0.0,
        "phase2_last_batch_schur_context": {},
        "phase2_last_optimizer_memory_reused": False,
        "phase2_last_optimizer_memory_source": "none",
        "phase2_last_shortlist_eval_records": [],
        "phase1_residual_opened": False,
        "available_indices_after_transition": set(),
        "phase1_stage_events_after_transition": [],
        "controller_measurement_work_after_eval": ControllerMeasurementWorkAccumulator(),
        "controller_measurement_work_step_proxy": {},
        "phase3_runtime_split_summary_after_eval": {},
        "proposals": [],
        "stop_reason": None,
        "fallback_scan_size": 0,
        "fallback_best_probe_delta_e": None,
        "fallback_best_probe_theta": None,
    }
    values.update(overrides)
    return engine_support._BranchStepScratch(**values)


def _empty_beam_round_diagnostic() -> dict[str, Any]:
    return {
        "raw_candidate_record_count": 0,
        "phase2_raw_candidate_record_count": 0,
        "phase1_shortlist_size": 0,
        "phase2_shortlist_size": 0,
        "phase3_shortlist_size": 0,
        "best_available_gradient": None,
        "best_available_simple_score": None,
        "best_available_phase2_raw_score": None,
        "best_available_full_v2_score": None,
        "best_available_gain": None,
        "parent_stop_reason_counts": {},
    }


def test_adapt_pipeline_uses_engine_support_dataclass_identities() -> None:
    expected_names = [
        "AdaptVQEResult",
        "_BeamBranchState",
        "_BranchExpansionPlan",
        "_BranchStepScratch",
    ]
    for name in expected_names:
        assert getattr(static_adapt, name) is getattr(engine_support, name)
        assert getattr(hardcoded_adapt, name) is getattr(engine_support, name)


def test_snake_rotosolve_wrapper_requires_explicit_stencil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(engine_support.STATIC_ADAPT_ALLOW_UNSTENCILED_ROTOSOLVE_ENV, raising=False)

    with pytest.raises(ValueError, match="requires explicit coefficient-aware period/shift"):
        engine_support._run_rotosolve_adapt_optimizer(
            objective=lambda theta: float(np.sum(np.asarray(theta, dtype=float) ** 2)),
            x0=np.asarray([0.1], dtype=float),
            maxiter=2,
            context_label="test",
        )


def test_snake_rotosolve_wrapper_rejects_partial_stencil(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(engine_support.STATIC_ADAPT_ALLOW_UNSTENCILED_ROTOSOLVE_ENV, "1")

    with pytest.raises(ValueError, match="period and shift must be provided together"):
        engine_support._run_rotosolve_adapt_optimizer(
            objective=lambda theta: float(np.sum(np.asarray(theta, dtype=float) ** 2)),
            x0=np.asarray([0.1], dtype=float),
            maxiter=2,
            context_label="test",
            period=math.pi,
        )


def test_snake_rotosolve_wrapper_accepts_explicit_stencil() -> None:
    def objective(theta: np.ndarray) -> float:
        (x,) = np.asarray(theta, dtype=float).reshape(-1)
        return float(1.0 + math.cos(2.0 * x))

    result = engine_support._run_rotosolve_adapt_optimizer(
        objective=objective,
        x0=np.asarray([0.125 * math.pi], dtype=float),
        maxiter=3,
        context_label="test",
        period=math.pi,
        shift=0.25 * math.pi,
    )

    assert result.success is True
    assert result.fun == pytest.approx(0.0, abs=1e-12)


def test_scipy_powell_options_accept_explicit_maxfev() -> None:
    options = engine_support._scipy_adapt_optimizer_options(
        method_key="POWELL",
        maxiter=300,
        maxfev=1234,
    )

    assert options["maxiter"] == 300
    assert options["maxfev"] == 1234
    assert "maxfev" not in engine_support._scipy_adapt_optimizer_options(
        method_key="BFGS",
        maxiter=300,
        maxfev=1234,
    )




def test_beam_energy_cost_dominance_uses_pareto_and_lambda_tradeoffs() -> None:
    lhs = _make_branch(branch_id=1, labels=("x",), theta=(0.1,))
    rhs = _make_branch(branch_id=2, labels=("y",), theta=(0.2,))

    lhs.energy_current = -2.0
    lhs.cumulative_beam_cost = 3.0
    rhs.energy_current = -1.0
    rhs.cumulative_beam_cost = 4.0
    pareto = engine_support._beam_energy_cost_dominance_case(lhs, rhs, lambda_beam=10.0)
    assert pareto["lhs_dominates_rhs"] is True
    assert pareto["case"] == "lhs_strict_pareto"

    lhs.energy_current = -1.0
    lhs.cumulative_beam_cost = 5.0
    rhs.energy_current = -2.0
    rhs.cumulative_beam_cost = 4.0
    worse_both = engine_support._beam_energy_cost_dominance_case(lhs, rhs, lambda_beam=0.0)
    assert worse_both["lhs_dominates_rhs"] is False
    assert worse_both["case"] == "rhs_strict_pareto"

    lhs.energy_current = -2.0
    lhs.cumulative_beam_cost = 10.0
    rhs.energy_current = -1.8
    rhs.cumulative_beam_cost = 1.0
    low_lambda = engine_support._beam_energy_cost_dominance_case(lhs, rhs, lambda_beam=0.01)
    high_lambda = engine_support._beam_energy_cost_dominance_case(lhs, rhs, lambda_beam=0.10)
    assert low_lambda["case"] == "tradeoff_lhs_lower_energy_higher_cost"
    assert low_lambda["lhs_dominates_rhs"] is True
    assert high_lambda["lhs_dominates_rhs"] is False

    lhs.energy_current = -1.8
    lhs.cumulative_beam_cost = 1.0
    rhs.energy_current = -2.0
    rhs.cumulative_beam_cost = 10.0
    saved_cost = engine_support._beam_energy_cost_dominance_case(lhs, rhs, lambda_beam=0.10)
    assert saved_cost["case"] == "tradeoff_lhs_higher_energy_lower_cost"
    assert saved_cost["lhs_dominates_rhs"] is True


def test_beam_search_policy_helpers_preserve_legacy_ranking_contract() -> None:
    first = _make_branch(branch_id=1, labels=("g0",), theta=(0.1,))
    second = _make_branch(branch_id=2, labels=("g1",), theta=(0.2,))
    second.energy_current = -1.75

    assert beam_search._beam_prune_key_payload_for_policy(
        first,
        ordered_batch_beam_mode=False,
        lambda_beam=0.25,
    ) == engine_support._beam_prune_key_payload(first)
    assert beam_search._beam_sort_key_for_policy(
        first,
        ordered_batch_beam_mode=False,
        lambda_beam=0.25,
    ) == engine_support._beam_prune_key(first)

    kept, audit = beam_search._beam_prune_for_policy(
        [first, second],
        cap=1,
        ordered_batch_beam_mode=False,
        lambda_beam=0.25,
        source="legacy_round",
    )
    expected = engine_support._beam_prune([first, second], cap=1)
    assert [branch.branch_id for branch in kept] == [branch.branch_id for branch in expected]
    assert audit is None

    duplicate = _make_branch(branch_id=9, labels=("g0",), theta=(0.1,))
    duplicate.energy_current = first.energy_current
    deduped, dedup_audit = beam_search._beam_dedup_for_policy(
        [first, duplicate],
        ordered_batch_beam_mode=False,
        lambda_beam=0.25,
        source="legacy_dedup",
    )
    assert [branch.branch_id for branch in deduped] == [
        branch.branch_id for branch in engine_support._beam_dedup([first, duplicate])
    ]
    assert dedup_audit is None


def test_beam_search_policy_helpers_preserve_ordered_batch_audit_contract() -> None:
    low_energy_high_cost = _make_branch(branch_id=1, labels=("g0",), theta=(0.1,))
    high_energy_low_cost = _make_branch(branch_id=2, labels=("g1",), theta=(0.2,))
    low_energy_high_cost.energy_current = -2.0
    low_energy_high_cost.cumulative_beam_cost = 10.0
    high_energy_low_cost.energy_current = -1.8
    high_energy_low_cost.cumulative_beam_cost = 1.0

    assert beam_search._beam_prune_key_payload_for_policy(
        low_energy_high_cost,
        ordered_batch_beam_mode=True,
        lambda_beam=0.10,
    ) == engine_support._beam_energy_cost_prune_key_payload(
        low_energy_high_cost,
        lambda_beam=0.10,
    )
    assert beam_search._beam_sort_key_for_policy(
        low_energy_high_cost,
        ordered_batch_beam_mode=True,
        lambda_beam=0.10,
    ) == engine_support._beam_energy_cost_sort_key(
        low_energy_high_cost,
        lambda_beam=0.10,
    )

    kept, audit = beam_search._beam_prune_for_policy(
        [low_energy_high_cost, high_energy_low_cost],
        cap=1,
        ordered_batch_beam_mode=True,
        lambda_beam=0.10,
        source="round_frontier",
    )
    expected_kept, expected_audit = engine_support._beam_prune_energy_cost_pareto_with_audit(
        [low_energy_high_cost, high_energy_low_cost],
        cap=1,
        lambda_beam=0.10,
    )
    expected_audit["source"] = "round_frontier"
    assert [branch.branch_id for branch in kept] == [branch.branch_id for branch in expected_kept]
    assert audit == expected_audit


def test_canonical_beam_survival_uses_realized_gain_per_added_cost_after_pareto() -> None:
    energy_root = 0.0
    old_rule_winner = _make_branch(branch_id=1, labels=("deep",), theta=(0.1,))
    ratio_winner = _make_branch(branch_id=2, labels=("efficient",), theta=(0.2,))
    old_rule_winner.energy_current = -8.0
    old_rule_winner.cumulative_beam_cost = 9.0
    old_rule_winner.cumulative_selector_score = 1e9
    ratio_winner.energy_current = -5.0
    ratio_winner.cumulative_beam_cost = 1.0
    ratio_winner.cumulative_selector_score = -1e9

    legacy_kept, _legacy_audit = engine_support._beam_prune_energy_cost_pareto_with_audit(
        [old_rule_winner, ratio_winner],
        cap=1,
        lambda_beam=0.1,
    )
    canonical_kept, audit = (
        engine_support._beam_prune_gain_per_added_cost_pareto_with_audit(
            [old_rule_winner, ratio_winner],
            cap=1,
            energy_root=energy_root,
            legacy_lambda_beam=0.1,
            cost_contract={"lambdas": {"2q": 0.2}},
        )
    )

    assert [branch.branch_id for branch in legacy_kept] == [1]
    assert [branch.branch_id for branch in canonical_kept] == [2]
    assert audit["dominated_count"] == 0
    assert audit["phase_local_normalized_scores_accumulated"] is False
    assert audit["legacy_lambda_beam_effect"] == "ignored"
    by_id = {row["branch_id"]: row for row in audit["input_branch_keys"]}
    assert by_id[1]["beam_survival_score"] == pytest.approx(0.8)
    assert by_id[2]["beam_survival_score"] == pytest.approx(2.5)


def test_canonical_beam_survival_applies_strict_realized_energy_cost_pareto() -> None:
    dominant = _make_branch(branch_id=1, labels=("dominant",), theta=(0.1,))
    dominated = _make_branch(branch_id=2, labels=("dominated",), theta=(0.2,))
    dominant.energy_current = -3.0
    dominant.cumulative_beam_cost = 1.0
    dominated.energy_current = -2.0
    dominated.cumulative_beam_cost = 2.0

    kept, audit = engine_support._beam_prune_gain_per_added_cost_pareto_with_audit(
        [dominated, dominant],
        cap=2,
        energy_root=0.0,
    )

    assert [branch.branch_id for branch in kept] == [1]
    assert audit["dominated_count"] == 1
    assert audit["dominance_events"][0]["dominating_branch_id"] == 1
    assert audit["dominance_events"][0]["dominated_branch_id"] == 2


def test_canonical_beam_survival_ignores_legacy_lambda_for_all_decisions() -> None:
    branches = [
        _make_branch(branch_id=1, labels=("a",), theta=(0.1,)),
        _make_branch(branch_id=2, labels=("b",), theta=(0.2,)),
    ]
    branches[0].energy_current = -4.0
    branches[0].cumulative_beam_cost = 5.0
    branches[1].energy_current = -3.0
    branches[1].cumulative_beam_cost = 1.0

    low_lambda, low_audit = beam_search._beam_prune_for_policy(
        branches,
        cap=1,
        ordered_batch_beam_mode=True,
        lambda_beam=0.0,
        canonical_beam_survival=True,
        energy_root=0.0,
    )
    high_lambda, high_audit = beam_search._beam_prune_for_policy(
        branches,
        cap=1,
        ordered_batch_beam_mode=False,
        lambda_beam=1e6,
        canonical_beam_survival=True,
        energy_root=0.0,
    )

    assert [branch.branch_id for branch in low_lambda] == [2]
    assert [branch.branch_id for branch in high_lambda] == [2]
    assert low_audit is not None and high_audit is not None
    assert low_audit["kept_branch_ids"] == high_audit["kept_branch_ids"]
    assert [row["beam_survival_score"] for row in low_audit["input_branch_keys"]] == pytest.approx(
        [row["beam_survival_score"] for row in high_audit["input_branch_keys"]]
    )


def test_canonical_beam_policy_requires_one_explicit_shared_root_energy() -> None:
    branch = _make_branch(branch_id=1, labels=("a",), theta=(0.1,))
    with pytest.raises(ValueError, match="energy_root"):
        beam_search._beam_prune_for_policy(
            [branch],
            cap=1,
            ordered_batch_beam_mode=False,
            lambda_beam=0.0,
            canonical_beam_survival=True,
        )


def test_beam_parent_round_policy_enables_parallel_with_stable_worker_budget() -> None:
    policy = beam_search._resolve_beam_parent_round_policy(
        frontier_input_count=4,
        requested_parent_workers=3,
        adapt_parallel_gradient_workers=8,
        finite_angle_fallback=False,
        cap_worker_limit_for_items=lambda requested, items: min(requested, items),
    )

    assert policy == beam_search._BeamParentRoundPolicy(
        parent_workers_requested=3,
        parent_workers_effective=3,
        parent_parallel_enabled=True,
        parent_parallel_disabled_reason=None,
        branch_worker_budget=2,
    )


@pytest.mark.parametrize(
    (
        "frontier_input_count",
        "requested_parent_workers",
        "finite_angle_fallback",
        "disabled_reason",
    ),
    [
        (3, 1, False, "workers_leq_one"),
        (1, 4, False, "single_parent_frontier"),
        (3, 4, True, "finite_angle_fallback_mutates_nfev"),
    ],
)
def test_beam_parent_round_policy_records_disabled_reason(
    frontier_input_count: int,
    requested_parent_workers: int,
    finite_angle_fallback: bool,
    disabled_reason: str,
) -> None:
    policy = beam_search._resolve_beam_parent_round_policy(
        frontier_input_count=frontier_input_count,
        requested_parent_workers=requested_parent_workers,
        adapt_parallel_gradient_workers=8,
        finite_angle_fallback=finite_angle_fallback,
        cap_worker_limit_for_items=lambda requested, items: min(requested, items),
    )

    assert policy.parent_workers_effective == min(max(1, requested_parent_workers), frontier_input_count)
    assert policy.parent_parallel_enabled is False
    assert policy.parent_parallel_disabled_reason == disabled_reason
    assert policy.branch_worker_budget == 8


def test_beam_round_frontier_diagnostic_accumulates_counts_and_bests() -> None:
    diagnostic = _empty_beam_round_diagnostic()
    scratch = _make_beam_scratch(
        max_grad=0.5,
        phase1_raw_record_count=5,
        phase2_raw_record_count=6,
        phase1_shortlist_size=2,
        phase2_shortlist_size=3,
        phase3_shortlist_size=4,
        phase1_last_retained_records=[{"simple_score": 1.25}],
        phase2_last_shortlist_eval_records=[{"phase2_raw_score": 2.5}],
        phase2_last_geometric_shortlist_records=[{"full_v2_score": -0.75}],
        phase2_last_retained_shortlist_records=[{"phase2_raw_trust_gain": 3.5}],
        phase2_last_admitted_records=[{"simple_score": 0.25}],
        proposals=[object()],
    )

    beam_search._accumulate_beam_round_frontier_diagnostic(diagnostic, scratch)

    assert diagnostic["raw_candidate_record_count"] == 5
    assert diagnostic["phase2_raw_candidate_record_count"] == 6
    assert diagnostic["phase1_shortlist_size"] == 2
    assert diagnostic["phase2_shortlist_size"] == 3
    assert diagnostic["phase3_shortlist_size"] == 4
    assert diagnostic["best_available_gradient"] == pytest.approx(0.5)
    assert diagnostic["best_available_simple_score"] == pytest.approx(1.25)
    assert diagnostic["best_available_phase2_raw_score"] == pytest.approx(2.5)
    assert diagnostic["best_available_full_v2_score"] == pytest.approx(-0.75)
    assert diagnostic["best_available_gain"] == pytest.approx(3.5)
    assert diagnostic["parent_stop_reason_counts"] == {"expanded": 1}

    lower_scoring_terminal = _make_beam_scratch(
        max_grad=0.1,
        phase1_raw_record_count=1,
        phase2_raw_record_count=1,
        phase1_shortlist_size=1,
        phase2_shortlist_size=1,
        phase3_shortlist_size=1,
        phase1_last_retained_records=[{"simple_score": -10.0}],
        stop_reason="eps_grad",
    )

    beam_search._accumulate_beam_round_frontier_diagnostic(
        diagnostic,
        lower_scoring_terminal,
    )

    assert diagnostic["raw_candidate_record_count"] == 6
    assert diagnostic["phase2_raw_candidate_record_count"] == 7
    assert diagnostic["phase1_shortlist_size"] == 3
    assert diagnostic["phase2_shortlist_size"] == 4
    assert diagnostic["phase3_shortlist_size"] == 5
    assert diagnostic["best_available_gradient"] == pytest.approx(0.5)
    assert diagnostic["best_available_simple_score"] == pytest.approx(1.25)
    assert diagnostic["parent_stop_reason_counts"] == {"expanded": 1, "eps_grad": 1}


def test_beam_base_branch_from_parent_scratch_copies_scratch_state_without_aliasing() -> None:
    parent = _make_branch(branch_id=4, labels=("g0",), theta=(0.1,))
    parent_available = set(parent.available_indices)
    scratch_stage = StageController(StageControllerConfig())
    scratch_work = ControllerMeasurementWorkAccumulator()
    scratch = _make_beam_scratch(
        energy_current=-2.5,
        available_indices_after_transition={8, 9},
        phase1_stage_after_transition=scratch_stage,
        phase1_residual_opened=True,
        phase1_stage_events_after_transition=[{"event": "stage"}],
        phase1_last_probe_reason="probe",
        phase1_last_positions_considered=[3, 4],
        phase1_last_trough_detected=True,
        phase1_last_trough_probe_triggered=True,
        phase1_last_selected_score=1.5,
        phase1_last_retained_records=[{"simple_score": 1.0}],
        phase2_last_shortlist_records=[{"candidate_label": "a"}],
        phase2_last_geometric_shortlist_records=[{"candidate_label": "b"}],
        phase2_last_retained_shortlist_records=[{"candidate_label": "c"}],
        phase2_last_admitted_records=[{"candidate_label": "d"}],
        phase2_last_batch_selected=True,
        phase2_last_batch_penalty_total=0.125,
        phase2_last_batch_schur_context={"nested": {"value": 7}},
        phase2_last_optimizer_memory_reused=True,
        phase2_last_optimizer_memory_source="scratch",
        phase2_last_shortlist_eval_records=[{"candidate_label": "eval"}],
        phase3_runtime_split_summary_after_eval={"split": {"count": 2}},
        controller_measurement_work_after_eval=scratch_work,
    )

    base = beam_search._beam_base_branch_from_parent_scratch(
        parent,
        scratch,
        branch_id=11,
        parent_branch_id=3,
    )

    assert parent.branch_id == 4
    assert parent.parent_branch_id is None
    assert parent.available_indices == parent_available

    assert base.branch_id == 11
    assert base.parent_branch_id == 3
    assert base.energy_current == pytest.approx(-2.5)
    assert base.available_indices == {8, 9}
    assert base.phase1_stage is not scratch_stage
    assert base.phase1_residual_opened is True
    assert base.phase1_last_probe_reason == "probe"
    assert base.phase1_last_positions_considered == [3, 4]
    assert base.phase1_last_trough_detected is True
    assert base.phase1_last_trough_probe_triggered is True
    assert base.phase1_last_selected_score == pytest.approx(1.5)
    assert base.phase2_last_batch_selected is True
    assert base.phase2_last_batch_penalty_total == pytest.approx(0.125)
    assert base.phase2_last_optimizer_memory_reused is True
    assert base.phase2_last_optimizer_memory_source == "scratch"
    assert base.controller_measurement_work is not scratch_work

    scratch.phase1_last_retained_records[0]["simple_score"] = 99.0
    scratch.phase2_last_batch_schur_context["nested"]["value"] = 99
    scratch.phase3_runtime_split_summary_after_eval["split"]["count"] = 99

    assert base.phase1_last_retained_records == [{"simple_score": 1.0}]
    assert base.phase2_last_batch_schur_context == {"nested": {"value": 7}}
    assert base.phase3_runtime_split_summary == {"split": {"count": 2}}


@pytest.mark.parametrize(
    ("scratch_stop_reason", "proposals", "expected_stop_reason"),
    [
        ("eps_grad", [object()], "eps_grad"),
        (None, [object()], "stop"),
        (None, [], "empty"),
    ],
)
def test_beam_terminal_child_from_scratch_preserves_stop_reason_contract(
    scratch_stop_reason: str | None,
    proposals: list[object],
    expected_stop_reason: str,
) -> None:
    base = _make_branch(branch_id=12, labels=("g0",), theta=(0.1,))
    base.parent_branch_id = 7
    base.terminated = False
    base.stop_reason = None
    scratch = _make_beam_scratch(
        proposals=proposals,
        stop_reason=scratch_stop_reason,
    )

    terminal = beam_search._beam_terminal_child_from_scratch(base, scratch)

    assert terminal is not base
    assert terminal.branch_id == 12
    assert terminal.parent_branch_id == 7
    assert terminal.last_transition_kind == "stop_child"
    assert terminal.last_admission_record_count == 0
    assert terminal.terminated is True
    assert terminal.stop_reason == expected_stop_reason
    assert base.terminated is False
    assert base.stop_reason is None


def test_beam_round_prune_audit_summary_counts_payload_fields() -> None:
    accepted = _make_branch(branch_id=1, labels=("g0",), theta=(0.1,))
    skipped = _make_branch(branch_id=2, labels=("g1",), theta=(0.2,))
    blocked = _make_branch(branch_id=3, labels=("g2",), theta=(0.3,))
    accepted.phase1_last_prune_summary = {
        "permission_reason": "eligible",
        "permission_open": True,
        "executed": True,
        "accepted_count": 2,
    }
    skipped.phase1_last_prune_summary = {
        "permission_reason": "eligible",
        "permission_open": True,
        "executed": False,
        "accepted_count": 0,
    }
    blocked.phase1_last_prune_summary = {
        "permission_reason": "cooldown",
        "permission_open": False,
        "executed": False,
        "accepted_count": 0,
    }

    summary = beam_search._beam_round_prune_audit_summary(
        [accepted, skipped, blocked],
        compact_prune_audit=lambda raw: dict(raw or {}),
    )

    assert summary.child_count == 3
    assert summary.permission_open_count == 2
    assert summary.executed_count == 1
    assert summary.accepted_count == 2
    assert summary.permission_reason_counts == {"eligible": 2, "cooldown": 1}
    assert summary.audits == [
        accepted.phase1_last_prune_summary,
        skipped.phase1_last_prune_summary,
        blocked.phase1_last_prune_summary,
    ]

    accepted.phase1_last_prune_summary["accepted_count"] = 99
    assert summary.audits[0]["accepted_count"] == 2


@pytest.mark.parametrize(
    (
        "frontier_input_count",
        "frontier_kept_count",
        "proposal_family_count",
        "parent_stop_reason_counts",
        "expected",
    ),
    [
        (2, 1, 0, {"eps_grad": 2}, None),
        (2, 0, 1, {"expanded": 2}, None),
        (0, 0, 0, {}, None),
        (2, 0, 0, {"eps_grad": 2}, "eps_grad"),
        (2, 0, 0, {"eps_grad": 1, "pool_exhausted": 1}, "mixed"),
        (2, 0, 0, {}, "empty"),
    ],
)
def test_beam_round_stop_reason_matches_runtime_payload_contract(
    frontier_input_count: int,
    frontier_kept_count: int,
    proposal_family_count: int,
    parent_stop_reason_counts: dict[str, int],
    expected: str | None,
) -> None:
    assert beam_search._beam_round_stop_reason(
        frontier_input_count=frontier_input_count,
        frontier_kept_count=frontier_kept_count,
        proposal_family_count=proposal_family_count,
        parent_stop_reason_counts=parent_stop_reason_counts,
    ) == expected


def test_beam_round_diagnostics_payload_preserves_runtime_fields() -> None:
    audit = {
        "permission_reason": "eligible",
        "permission_open": True,
        "executed": True,
        "accepted_count": 2,
    }
    summary = beam_search._BeamRoundPruneAuditSummary(
        audits=[audit],
        permission_reason_counts={"eligible": 1},
        child_count=3,
        permission_open_count=2,
        executed_count=1,
        accepted_count=2,
    )
    frontier_diagnostic = {
        "raw_candidate_record_count": 7,
        "parent_stop_reason_counts": {"expanded": 2},
        "best_available_gain": 0.25,
    }

    payload = beam_search._beam_round_diagnostics_payload(
        depth=4,
        frontier_input_count=5,
        parents_expanded_count=4,
        proposals_selected_count=3,
        proposal_family_count=2,
        stop_children_count=1,
        child_frontier_count=6,
        round_terminal_count=2,
        active_children_unique_count=5,
        frontier_kept_count=4,
        round_live_cap=0,
        terminal_pool_candidate_count=8,
        terminal_pool_unique_count=6,
        terminal_kept_count=3,
        round_stop_reason="mixed",
        beam_parent_workers_requested=9,
        beam_parent_workers_effective=4,
        beam_parent_parallel_enabled=True,
        beam_parent_parallel_disabled_reason=None,
        beam_parent_result_elapsed_s=[1, 2.5],
        round_frontier_diagnostic=frontier_diagnostic,
        round_prune_audit_summary=summary,
    )

    assert payload["depth"] == 5
    assert payload["children_materialized_count"] == 8
    assert payload["active_children_raw_count"] == 6
    assert payload["active_children_unique_count"] == 5
    assert payload["frontier_kept_count"] == 4
    assert payload["frontier_cap_effective"] == 1
    assert payload["round_terminals_raw_count"] == 2
    assert payload["terminal_pool_candidate_count"] == 8
    assert payload["terminal_pool_unique_count"] == 6
    assert payload["terminal_kept_count"] == 3
    assert payload["stop_reason"] == "mixed"
    assert payload["beam_parent_workers_requested"] == 9
    assert payload["beam_parent_workers_effective"] == 4
    assert payload["beam_parent_parallel_enabled"] is True
    assert payload["beam_parent_parallel_merge_order"] == "frontier_order"
    assert payload["beam_parent_result_elapsed_s"] == [1.0, 2.5]
    assert payload["raw_candidate_record_count"] == 7
    assert payload["parent_stop_reason_counts"] == {"expanded": 2}
    assert payload["best_available_gain"] == 0.25
    assert payload["prune_child_count"] == 3
    assert payload["prune_permission_open_count"] == 2
    assert payload["prune_executed_count"] == 1
    assert payload["prune_accepted_count"] == 2
    assert payload["prune_permission_reason_counts"] == {"eligible": 1}
    assert payload["prune_audits"] == [audit]

    audit["accepted_count"] = 99
    assert payload["prune_audits"][0]["accepted_count"] == 2


def test_beam_round_done_log_payload_preserves_runtime_fields() -> None:
    payload = beam_search._beam_round_done_log_payload(
        depth=2,
        frontier_input_count=5,
        parents_expanded_count=4,
        proposals_selected_count=3,
        proposal_family_count=2,
        stop_children_count=1,
        frontier_kept_count=6,
        round_live_cap=0,
        terminal_kept_count=7,
        round_stop_reason="mixed",
        beam_parent_workers_requested=8,
        beam_parent_workers_effective=4,
        beam_parent_parallel_enabled=True,
        beam_parent_parallel_disabled_reason=None,
        round_frontier_diagnostic={
            "frontier_kept_count": 99,
            "parent_stop_reason_counts": {"expanded": 2},
            "best_available_gain": 0.25,
        },
    )

    assert payload["depth"] == 3
    assert payload["frontier_input_count"] == 5
    assert payload["parents_expanded_count"] == 4
    assert payload["proposals_selected_count"] == 3
    assert payload["proposal_family_count"] == 2
    assert payload["stop_children_count"] == 1
    assert payload["frontier_kept_count"] == 99
    assert payload["frontier_cap_effective"] == 1
    assert payload["terminal_kept_count"] == 7
    assert payload["stop_reason"] == "mixed"
    assert payload["beam_parent_workers_requested"] == 8
    assert payload["beam_parent_workers_effective"] == 4
    assert payload["beam_parent_parallel_enabled"] is True
    assert payload["beam_parent_parallel_disabled_reason"] is None
    assert payload["beam_parent_parallel_merge_order"] == "frontier_order"
    assert payload["parent_stop_reason_counts"] == {"expanded": 2}
    assert payload["best_available_gain"] == 0.25


def test_beam_branch_replay_summary_payload_preserves_checkpoint_contract() -> None:
    branch = _make_branch(branch_id=7, labels=("op_a", "op_b"), theta=(0.1, 0.2))
    branch.parent_branch_id = 3
    branch.depth_local = 5
    branch.terminated = True
    branch.stop_reason = "benchmark_abs_delta_e_target"
    branch.energy_current = -1.01
    branch.history = [
        {"depth": 1, "selected_op": "old", "energy_after_opt": -1.2},
        {
            "depth": 5,
            "branch_id": 7,
            "parent_branch_id": 3,
            "selected_ops": ["op_b"],
            "selected_positions": [2],
            "selected_pool_indices": [8],
            "selected_feature_rows": [
                {
                    "candidate_label": "gen_b",
                    "generator_id": "g:b",
                    "runtime_split_child_generator_ids": ["child:b"],
                }
            ],
            "energy_before_opt": -1.05,
            "energy_after_opt": -1.01,
            "delta_energy": -0.04,
        },
    ]

    prune_key = {"policy": "unit", "branch_id": 7}
    payload = beam_search._beam_branch_replay_summary_payload(
        branch,
        keep_history_tail=1,
        benchmark_stop_reference_energy=-1.0,
        benchmark_target_abs_delta_e=0.02,
        beam_prune_key_payload=lambda branch_now: {
            **prune_key,
            "observed_branch_id": int(branch_now.branch_id),
        },
    )

    assert payload["branch_id"] == 7
    assert payload["parent_branch_id"] == 3
    assert payload["status"] == "terminal"
    assert payload["terminated"] is True
    assert payload["stop_reason"] == "benchmark_abs_delta_e_target"
    assert payload["depth_local"] == 5
    assert payload["ansatz_depth"] == 2
    assert payload["energy"] == pytest.approx(-1.01)
    assert payload["operator_labels"] == ["op_a", "op_b"]
    assert payload["history_count"] == 2
    assert payload["history_tail_count"] == 1
    assert payload["history_tail"][0]["depth"] == 5
    assert payload["last_selected_records"] == [
        {
            "operator_label": "op_b",
            "generator_label": "gen_b",
            "generator_id": "g:b",
            "parent_generator_id": None,
            "template_id": None,
            "position_id": 2,
            "candidate_pool_index": 8,
            "selection_mode": "",
            "runtime_split_mode": "off",
                "runtime_split_chosen_representation": None,
                "runtime_split_child_generator_ids": ["child:b"],
                "route_a_child_identity": None,
                "route_a_global_pauli_identity": None,
                "route_a_child_parent_labels": [],
                "route_a_child_parent_count": None,
                "route_a_child_direction_normalization": None,
            }
        ]
    assert payload["benchmark_target_abs_delta_current"] == pytest.approx(0.01)
    assert payload["benchmark_target_abs_delta_e"] == 0.02
    assert payload["benchmark_target_error_within_threshold"] is True
    assert payload["benchmark_target_hit"] is True
    assert payload["benchmark_target_classification"]["source"] == "beam_branch_replay_summary"
    assert payload["benchmark_target_classification"]["target_hit_success"] is True
    assert payload["benchmark_target_classification"]["required_stop_reason"] == (
        "benchmark_abs_delta_e_target"
    )
    assert payload["frontier_prune_key"] == {
        "policy": "unit",
        "branch_id": 7,
        "observed_branch_id": 7,
    }
    assert "formal_manifold_runtime_checkpoint" not in payload
    assert "formal_manifold_behavioral_fingerprint" not in payload


def test_beam_branch_summary_payload_preserves_final_diagnostics_contract() -> None:
    branch = _make_branch(branch_id=8, labels=("op_a", "op_b"), theta=(0.1, 0.2))
    branch.parent_branch_id = 4
    branch.depth_local = 6
    branch.terminated = True
    branch.stop_reason = "benchmark_abs_delta_e_target"
    branch.energy_current = -1.005
    branch.last_transition_kind = "non_stop_child"
    branch.last_admission_record_count = 2
    branch.cumulative_selector_score = 3.5
    branch.cumulative_selector_burden = 0.75
    branch.phase1_residual_opened = True
    branch.phase1_last_probe_reason = "phase3"
    branch.phase1_stage_events = [{"event": "stage"}]
    branch.phase1_last_prune_summary = {
        "permission_reason": "eligible",
        "permission_open": True,
        "executed": True,
        "accepted_count": 1,
        "candidate_count": 2,
    }
    branch.history = [
        {"depth": 1, "post_admission_prune": {"accepted_count": 0}},
        {
            "depth": 2,
            "post_admission_prune": {
                "permission_reason": "eligible",
                "permission_open": True,
                "executed": True,
                "accepted_count": 1,
            },
        },
    ]
    branch.phase2_last_shortlist_records = [
        {"candidate_label": "op_a", "generator_id": "g:a", "position_id": 0},
        {"candidate_label": "op_b", "generator_id": "g:b", "position_id": 1},
    ]
    branch.phase2_last_retained_shortlist_records = [
        {"candidate_label": "op_b", "generator_id": "g:b", "position_id": 1}
    ]
    branch.phase2_last_admitted_records = [
        {"candidate_label": "op_b", "generator_id": "g:b", "position_id": 1}
    ]
    branch.phase2_optimizer_memory = {
        "available": True,
        "optimizer": "POWELL",
        "parameter_count": 2,
        "source": "unit",
        "remap_events": [{"op": "insert"}],
    }
    branch.phase2_last_optimizer_memory_source = "unit_source"
    branch.phase2_last_optimizer_memory_reused = True

    payload = beam_search._beam_branch_summary_payload(
        branch,
        benchmark_stop_reference_energy=-1.0,
        benchmark_target_abs_delta_e=0.01,
        generator_ids=["g:a", "g:b"],
        beam_prune_key_payload=lambda branch_now: {
            "policy": "unit",
            "branch_id": int(branch_now.branch_id),
        },
    )

    assert payload["branch_id"] == 8
    assert payload["parent_branch_id"] == 4
    assert payload["depth_local"] == 6
    assert payload["status"] == "terminal"
    assert payload["termination_label"] == "benchmark_abs_delta_e_target"
    assert payload["last_transition_kind"] == "non_stop_child"
    assert payload["last_admission_record_count"] == 2
    assert payload["energy"] == pytest.approx(-1.005)
    assert payload["benchmark_target_abs_delta_current"] == pytest.approx(0.005)
    assert payload["benchmark_target_abs_delta_e"] == 0.01
    assert payload["benchmark_target_hit"] is True
    assert payload["benchmark_target_classification"]["source"] == "beam_branch_summary"
    assert payload["cumulative_selector_score"] == pytest.approx(3.5)
    assert payload["cumulative_selector_burden"] == pytest.approx(0.75)
    assert payload["scored_surface_count"] == 2
    assert payload["retained_shortlist_count"] == 1
    assert payload["admitted_count"] == 1
    assert "formal_manifold_runtime_checkpoint" not in payload
    assert "formal_manifold_behavioral_fingerprint" not in payload
    assert payload["phase3_surface_summary"]["scored_surface"]["count"] == 2
    assert payload["phase3_surface_summary"]["admitted_set"]["operator_labels"] == ["op_b"]
    assert payload["prune_key"] == {"policy": "unit", "branch_id": 8}
    assert payload["last_prune"]["accepted_count"] == 1
    assert len(payload["prune_history"]) == 2
    assert payload["prune_history"][1]["accepted_count"] == 1
    assert payload["branch_state_summary"]["branch_id"] == 8
    assert payload["branch_state_summary"]["status"] == "terminal"
    assert payload["branch_state_summary"]["controller_telemetry"]["residual_opened"] is True

    memory_contract = payload["optimizer_memory_contract_summary"]
    assert memory_contract["branch_id"] == 8
    assert memory_contract["memory_available"] is True
    assert memory_contract["memory_optimizer"] == "POWELL"
    assert memory_contract["memory_parameter_count"] == 2
    assert memory_contract["last_active_subset_source"] == "unit_source"
    assert memory_contract["last_active_subset_reused"] is True
    assert memory_contract["structural_transport_detected"] is True
    assert memory_contract["scaffold_fingerprint"]["selected_operator_labels"] == [
        "op_a",
        "op_b",
    ]
    assert memory_contract["scaffold_fingerprint"]["selected_generator_ids"] == [
        "g:a",
        "g:b",
    ]


def test_beam_final_diagnostics_payload_preserves_winner_relationship_contract() -> None:
    winner = _make_branch(branch_id=8, labels=("winner",), theta=(0.8,))
    winner.parent_branch_id = 4
    winner.depth_local = 2
    winner.terminated = True
    winner.stop_reason = "max_depth"
    winner.energy_current = -3.0
    winner.phase1_last_prune_summary = {"accepted_count": 1}

    frontier_late = _make_branch(branch_id=6, labels=("late",), theta=(0.6,))
    frontier_late.parent_branch_id = 1
    frontier_late.depth_local = 3
    frontier_late.energy_current = -1.0

    frontier_best = _make_branch(branch_id=3, labels=("best",), theta=(0.3,))
    frontier_best.parent_branch_id = 2
    frontier_best.depth_local = 5
    frontier_best.energy_current = -2.0

    terminal_other = _make_branch(branch_id=9, labels=("other",), theta=(0.9,))
    terminal_other.terminated = True
    terminal_other.stop_reason = "empty_frontier"
    terminal_other.energy_current = -0.5

    target_classification = {
        "target_hit_success": False,
        "non_hit_reason": "stop_reason",
        "source": "beam_final_winner",
    }

    def sort_key(branch: engine_support._BeamBranchState) -> tuple[float, int]:
        return (float(branch.energy_current), int(branch.branch_id))

    def prune_key(branch: engine_support._BeamBranchState) -> dict[str, Any]:
        return {"policy": "unit", "branch_id": int(branch.branch_id)}

    def replay_summary(branch: engine_support._BeamBranchState) -> dict[str, Any]:
        return {
            "branch_id": int(branch.branch_id),
            "status": "terminal" if branch.terminated else "frontier",
        }

    def branch_summary(branch: engine_support._BeamBranchState) -> dict[str, Any]:
        return {
            "branch_id": int(branch.branch_id),
            "branch_state_summary": {
                "branch_id": int(branch.branch_id),
                "status": "terminal" if branch.terminated else "frontier",
            },
            "optimizer_memory_contract_summary": {
                "branch_id": int(branch.branch_id),
                "memory_available": branch.branch_id == winner.branch_id,
            },
        }

    survival_audit = {"round": 1, "kept_branch_ids": [8, 3]}
    payload = beam_search._beam_final_diagnostics_payload(
        frontier=[frontier_late, frontier_best],
        terminals=[winner, terminal_other],
        finalists=[terminal_other, frontier_best, winner],
        winner_branch=winner,
        winner_target_classification=target_classification,
        beam_sort_key=sort_key,
        branch_state_fingerprint=lambda branch: f"fp:{branch.branch_id}",
        beam_prune_key_payload=prune_key,
        beam_branch_replay_summary=replay_summary,
        beam_branch_summary=branch_summary,
        beam_survival_audits=[survival_audit],
    )

    assert payload["frontier_final_count"] == 2
    assert payload["terminal_final_count"] == 2
    assert payload["finalist_count"] == 3
    assert payload["winner_branch_id"] == 8
    assert payload["winner_parent_branch_id"] == 4
    assert payload["winner_stop_reason"] == "max_depth"
    assert payload["winner_target_hit_success"] is False
    assert payload["winner_target_non_hit_reason"] == "stop_reason"
    assert payload["winner_target_hit_classification"] == target_classification
    assert payload["winner_fingerprint"] == "fp:8"
    assert payload["winner_prune_key"] == {"policy": "unit", "branch_id": 8}
    assert payload["winner_survival_key"] == payload["winner_prune_key"]
    assert payload["winner_branch_summary"]["branch_id"] == 8
    assert payload["winner_branch_state_summary"] == {
        "branch_id": 8,
        "status": "terminal",
    }
    assert payload["winner_optimizer_memory_contract"] == {
        "branch_id": 8,
        "memory_available": True,
    }
    assert [row["branch_id"] for row in payload["finalist_summaries"]] == [8, 3, 9]
    assert payload["survival_audits"] == [survival_audit]
    assert payload["survival_audits"][0] is not survival_audit

    relationship = payload["final_checkpoint_relationship"]
    assert relationship["schema_version"] == (
        "static_adapt_beam_final_checkpoint_relationship_v1"
    )
    assert relationship["relationship_present"] is True
    assert relationship["reason"] == (
        "non_target_terminal_selected_with_recoverable_frontier"
    )
    assert relationship["diagnostic_terminal_branch_id"] == 8
    assert relationship["diagnostic_terminal_stop_reason"] == "max_depth"
    assert relationship["diagnostic_terminal_target_hit_classification"] == (
        target_classification
    )
    assert relationship["recoverable_frontier_branch_id"] == 3
    assert relationship["recoverable_frontier_parent_branch_id"] == 2
    assert relationship["recoverable_frontier_deeper_than_terminal"] is True
    assert relationship["checkpoint_branch_policy"] == "best_frontier_branch"
    assert relationship["diagnostic_terminal_branch"] == {
        "branch_id": 8,
        "status": "terminal",
    }
    assert relationship["recoverable_frontier_branch"] == {
        "branch_id": 3,
        "status": "frontier",
    }

    relationship["diagnostic_terminal_target_hit_classification"][
        "target_hit_success"
    ] = True
    assert payload["winner_target_hit_classification"]["target_hit_success"] is False


def test_beam_replay_round_payload_preserves_runtime_shape_and_ordering() -> None:
    frontier_late = _make_branch(branch_id=3, labels=("g3",), theta=(0.3,))
    frontier_early = _make_branch(branch_id=1, labels=("g1",), theta=(0.1,))
    terminal_done = _make_branch(branch_id=2, labels=("g2",), theta=(0.2,))
    terminal_empty = _make_branch(branch_id=4, labels=("g4",), theta=(0.4,))
    terminal_done.terminated = True
    terminal_empty.terminated = True
    terminal_done.stop_reason = "done"
    terminal_empty.stop_reason = "empty"
    frontier_diagnostic = {
        "parent_stop_reason_counts": {"expanded": 2},
        "best_available_gradient": "1.25",
        "best_available_simple_score": "nan",
        "best_available_phase2_raw_score": 0.5,
        "best_available_full_v2_score": None,
        "best_available_gain": 0.75,
    }

    def _str_or_none(value: Any) -> str | None:
        if value in {None, ""}:
            return None
        return str(value)

    def _float_or_none(value: Any) -> float | None:
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            return None
        return float(value_f) if math.isfinite(value_f) else None

    payload = beam_search._beam_replay_round_payload(
        depth=6,
        frontier_input_count=5,
        parents_expanded_count=4,
        proposals_selected_count=3,
        proposal_family_count=2,
        stop_children_count=1,
        round_live_cap=0,
        round_stop_reason="",
        round_frontier_diagnostic=frontier_diagnostic,
        frontier_branches=[frontier_late, frontier_early],
        terminal_branches=[terminal_empty, terminal_done],
        round_terminal_branches=[terminal_empty],
        beam_sort_key=lambda branch: (int(branch.branch_id),),
        beam_branch_replay_summary=lambda branch: {
            "branch_id": int(branch.branch_id),
            "terminated": bool(branch.terminated),
        },
        current_str_or_none=_str_or_none,
        current_float=_float_or_none,
    )

    assert payload["schema_version"] == "static_adapt_beam_replay_round_v1"
    assert payload["depth"] == 6
    assert payload["stop_reason"] is None
    assert payload["frontier_input_count"] == 5
    assert payload["parents_expanded_count"] == 4
    assert payload["proposals_selected_count"] == 3
    assert payload["proposal_family_count"] == 2
    assert payload["stop_children_count"] == 1
    assert payload["parent_stop_reason_counts"] == {"expanded": 2}
    assert payload["best_available"] == {
        "gradient": 1.25,
        "simple_score": None,
        "phase2_raw_score": 0.5,
        "full_v2_score": None,
        "gain": 0.75,
    }
    assert payload["frontier_summary"] == {
        "kept_count": 2,
        "cap_effective": 1,
        "branch_ids": [1, 3],
    }
    assert payload["terminal_summary"] == {
        "kept_count": 2,
        "round_terminal_count": 1,
        "branch_ids": [2, 4],
        "round_branch_ids": [4],
        "stop_reason_counts": {"done": 1, "empty": 1},
    }
    assert payload["frontier"]["branches"] == [
        {"branch_id": 1, "terminated": False},
        {"branch_id": 3, "terminated": False},
    ]
    assert payload["terminal"]["branches"] == [
        {"branch_id": 2, "terminated": True},
        {"branch_id": 4, "terminated": True},
    ]
    assert payload["terminal"]["round_branches"] == [
        {"branch_id": 4, "terminated": True},
    ]

    frontier_diagnostic["parent_stop_reason_counts"]["expanded"] = 99
    assert payload["parent_stop_reason_counts"] == {"expanded": 2}


def test_beam_replay_telemetry_payload_preserves_tail_policy_and_copies() -> None:
    leading = _make_branch(branch_id=3, labels=("lead",), theta=(0.3,))
    checkpoint = _make_branch(branch_id=1, labels=("checkpoint",), theta=(0.1,))
    round_payload = {"round": 3, "nested": {"value": 1}}
    replay_rounds = [
        beam_search._compact_beam_replay_round_payload(
            {
                "round": round_id,
                "nested": {"value": round_id - 1},
                "frontier": {
                    "branches": [
                        {
                            "branch_id": round_id,
                            "history_tail": [{"round": round_id}],
                        }
                    ]
                },
                "terminal": {"branches": [], "round_branches": []},
            }
        )
        for round_id in (1, 2, 3)
    ]

    payload = beam_search._beam_replay_telemetry_payload(
        depth=4,
        round_replay_payload=round_payload,
        beam_replay_rounds=replay_rounds,
        replay_tail_count=2,
        leading_branch=leading,
        checkpoint_branch=checkpoint,
        has_checkpoint_frontier_candidates=True,
        beam_branch_replay_summary=lambda branch: {
            "branch_id": int(branch.branch_id),
        },
    )

    assert payload["schema_version"] == "static_adapt_beam_replay_telemetry_v1"
    assert payload["depth"] == 5
    assert payload["current_round"] == round_payload
    assert payload["rounds_storage_policy"] == "historical_round_compact_v1"
    assert payload["rounds"] == replay_rounds[-2:]
    assert payload["leading_branch"] == {"branch_id": 3}
    assert payload["checkpoint_branch"] == {"branch_id": 1}
    assert payload["checkpoint_branch_policy"] == "best_frontier_branch"

    round_payload["nested"]["value"] = 99
    replay_rounds[-1]["nested"]["value"] = 99
    assert payload["current_round"]["nested"]["value"] == 1
    assert payload["rounds"][-1]["nested"]["value"] == 2
    historical_branch = payload["rounds"][-1]["frontier"]["branches"][0]
    assert "history_tail" not in historical_branch
    assert len(historical_branch["history_tail_sha256"]) == 64
    assert len(historical_branch["full_payload_sha256"]) == 64

    fallback_payload = beam_search._beam_replay_telemetry_payload(
        depth=4,
        round_replay_payload={},
        beam_replay_rounds=[],
        replay_tail_count=1,
        leading_branch=leading,
        checkpoint_branch=checkpoint,
        has_checkpoint_frontier_candidates=False,
        beam_branch_replay_summary=lambda branch: {
            "branch_id": int(branch.branch_id),
        },
    )
    assert fallback_payload["checkpoint_branch_policy"] == (
        "best_terminal_or_root_branch"
    )


def test_beam_fingerprint_and_seed_helpers_match_contract() -> None:
    branch = _make_branch(branch_id=7, labels=("g0", "g1"), theta=(0.125, -0.25))
    plan = engine_support._BranchExpansionPlan(
        candidate_pool_index=4,
        position_id=1,
        selection_mode="beam",
        candidate_label="cand::x",
        candidate_term=SimpleNamespace(label="cand::x"),
        feature_row=None,
        init_theta=0.375,
    )

    expected_proposal_payload = {
        "parent": engine_support._branch_state_fingerprint(branch),
        "candidate_pool_index": 4,
        "position_id": 1,
        "selection_mode": "beam",
        "candidate_label": "cand::x",
        "init_theta": round(0.375, 12),
    }
    expected_proposal = hashlib.sha256(
        json.dumps(expected_proposal_payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert engine_support._proposal_fingerprint(parent=branch, plan=plan) == expected_proposal

    expected_seed_payload = {
        "base_seed": 17,
        "stage_tag": "beam-child",
        "depth_local": 2,
        "parent_state_fingerprint": engine_support._branch_state_fingerprint(branch),
        "proposal_fingerprint": expected_proposal,
    }
    expected_seed = int.from_bytes(
        hashlib.sha256(json.dumps(expected_seed_payload, sort_keys=True).encode("utf-8")).digest()[:8],
        "big",
    ) % (2**31 - 1)
    assert engine_support._branch_optimizer_seed(
        base_seed=17,
        stage_tag="beam-child",
        depth_local=2,
        parent_state_fingerprint=engine_support._branch_state_fingerprint(branch),
        proposal_fingerprint=expected_proposal,
    ) == expected_seed


def test_beam_child_clones_trust_state_without_aliasing() -> None:
    branch = _make_branch(branch_id=7, labels=("g0",), theta=(0.125,))
    branch.route_a_trust_region_state = RouteATrustRegionState(radius=0.25)

    first = branch.clone_for_child(branch_id=8)
    second = branch.clone_for_child(branch_id=9)
    assert first.route_a_trust_region_state is not branch.route_a_trust_region_state
    assert second.route_a_trust_region_state is not first.route_a_trust_region_state

    first.route_a_trust_region_state.radius = 0.5
    assert branch.route_a_trust_region_state.radius == pytest.approx(0.25)
    assert second.route_a_trust_region_state.radius == pytest.approx(0.25)


def test_beam_fingerprint_distinguishes_future_trust_radius() -> None:
    first = _make_branch(branch_id=7, labels=("g0",), theta=(0.125,))
    second = _make_branch(branch_id=8, labels=("g0",), theta=(0.125,))
    first.route_a_trust_region_state = RouteATrustRegionState(radius=0.25)
    second.route_a_trust_region_state = RouteATrustRegionState(
        radius=0.5,
        reference_radius=0.25,
    )

    assert engine_support._branch_state_fingerprint(first) != (
        engine_support._branch_state_fingerprint(second)
    )










def test_unchanged_reference_radius_preserves_legacy_beam_fingerprint() -> None:
    legacy = _make_branch(branch_id=7, labels=("g0",), theta=(0.125,))
    adaptive = _make_branch(branch_id=8, labels=("g0",), theta=(0.125,))
    adaptive.route_a_trust_region_state = RouteATrustRegionState(radius=0.25)

    assert engine_support._branch_state_fingerprint(adaptive) == (
        engine_support._branch_state_fingerprint(legacy)
    )


def test_resolve_beam_local_reopt_seed_inputs_shares_seed_across_siblings() -> None:
    assert engine_support._resolve_beam_local_reopt_seed_inputs(
        proposal_count=2,
        proposal_fingerprint="fp-123",
    ) == ("parent_shared", None)
    assert engine_support._resolve_beam_local_reopt_seed_inputs(
        proposal_count=1,
        proposal_fingerprint="fp-123",
    ) == ("proposal_conditioned", "fp-123")


def test_beam_dedup_keeps_incumbent_on_exact_prune_key_tie() -> None:
    first = _make_branch(branch_id=9, labels=("g0",), theta=(0.5,))
    second = _make_branch(branch_id=9, labels=("g0",), theta=(0.5,))
    deduped = engine_support._beam_dedup([first, second])
    assert len(deduped) == 1
    assert deduped[0] is first
