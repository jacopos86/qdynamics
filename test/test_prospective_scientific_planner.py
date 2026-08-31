"""Transition and provenance tests for the prospective HH/Hubbard planner."""

from __future__ import annotations

from dataclasses import replace

import pytest

from pipelines.time_dynamics.parallel_benchmark_matrix import ParallelBenchmarkMatrix
from pipelines.time_dynamics.prospective_campaign_bootstrap import (
    build_identity_registry,
)
from pipelines.time_dynamics.prospective_scientific_planner import (
    ARM_APM,
    ARM_AVQDS,
    FAILURE_MEMORY,
    FAILURE_OTHER,
    HH_HUBBARD_ACTIVATION_CUT_LADDER,
    HORIZON_FULL,
    HORIZON_PREFIX,
    KIND_RECORD_SAFEGUARD,
    KIND_RUN,
    KIND_SCORE,
    LANE_HH,
    LANE_HUBBARD,
    ProspectiveEvidence,
    ProspectivePlannerError,
    ProspectiveRunEvidence,
    ProspectiveSafeguardEvidence,
    ProspectiveScientificBindings,
    SAFEGUARD_OPERATIONALLY_INFEASIBLE,
    SCORE_COMPLETED,
    SCORE_NOT_STARTED,
    STATE_COMPLETED,
    STATE_FAILED,
    plan_prospective_cell,
    policy_sha256,
    prospective_run_action_id,
    prospective_safeguard_id,
)
from pipelines.time_dynamics.staged_campaign import AccuracyMetrics, AccuracyPolicy


def _sha(character: str) -> str:
    return character * 64


POLICY = AccuracyPolicy.response_normalized_matched_pair(
    response_scales={
        "weakslow": {
            horizon: {
                "energy_rms_excursion": 0.02,
                "doublon_rms_excursion": 0.01,
                "max_site_occupation_rms_excursion": 0.01,
            }
            for horizon in (HORIZON_PREFIX, HORIZON_FULL)
        }
    },
    response_relative_target=0.05,
    response_relative_guard=0.15,
    matched_accuracy_factor=2.0,
    matched_component_deadband=0.10,
    infidelity_ceiling=1.0e-4,
    absolute_energy_error_ceiling=1.0e-2,
    absolute_doublon_error_ceiling=5.0e-3,
    absolute_site_occupation_error_ceiling=5.0e-3,
)


def _bindings(
    *,
    cell_id: str = "hh__nph3__crossover__weakslow",
    **updates: object,
) -> ProspectiveScientificBindings:
    matrix = ParallelBenchmarkMatrix.read()
    identities = build_identity_registry(matrix)
    reference = next(row for row in identities.references if row.cell_id == cell_id)
    seed = next(row for row in identities.seeds if row.seed_id == reference.seed_id)
    values: dict[str, object] = {
        "lane_id": reference.lane_id,
        "cell_id": reference.cell_id,
        "policy_regime": str(reference.physical.get("regime") or "hubbard_l4_u4"),
        "drive": str(reference.drive["drive_id"]),
        "controller": "state_motion_5e-5",
        "cell_identity_sha256": reference.identity_sha256,
        "seed_id": seed.seed_id,
        "seed_identity_sha256": seed.identity_sha256,
        "seed_artifact_sha256": _sha("2"),
        "seed_receipt_sha256": _sha("3"),
        "reference_id": reference.reference_id,
        "reference_identity_sha256": reference.identity_sha256,
        "reference_manifest_sha256": _sha("4"),
        "reference_data_sha256": _sha("5"),
        "reference_receipt_sha256": _sha("6"),
        "source_epoch_identity_sha256": _sha("a"),
        "source_revision": "7" * 40,
        "source_tree_sha256": _sha("8"),
        "matrix_id": matrix.matrix_id,
        "matrix_raw_sha256": matrix.raw_sha256,
        "matrix_payload_sha256": matrix.payload_sha256,
        "matrix_expanded_semantic_sha256": matrix.expanded_semantic_sha256,
        "matrix_accuracy_policy_id": "response_normalized_matched_pair_v2",
        "matrix_accuracy_policy_sha256": matrix.accuracy_policy_spec_sha256,
        "accuracy_policy_mode": POLICY.mode,
        "accuracy_policy": POLICY.to_json_dict(),
        "accuracy_policy_sha256": policy_sha256(POLICY),
        "qiskit_compilation": False,
    }
    values.update(updates)
    return ProspectiveScientificBindings(**values)


def test_scientific_binding_policy_is_deeply_immutable() -> None:
    bindings = _bindings()

    with pytest.raises(TypeError):
        bindings.accuracy_policy["response_scales"]["weakslow"][HORIZON_PREFIX][
            "energy_rms_excursion"
        ] = 999.0


def _metrics(
    q: float,
    *,
    components: tuple[float, float, float, float] | None = None,
) -> AccuracyMetrics:
    q_energy, q_doublon, q_site, q_fidelity = components or (q, q, q, q)
    return AccuracyMetrics(
        mean_abs_energy_error=q_energy * 0.02 * 0.05,
        mean_abs_doublon_error=q_doublon * 0.01 * 0.05,
        mean_max_abs_site_occupation_error=q_site * 0.01 * 0.05,
        mean_infidelity=q_fidelity * 1.0e-4,
    )


def _completed(
    bindings: ProspectiveScientificBindings,
    *,
    arm: str,
    horizon: str,
    index: int,
    q: float | None,
    sequence: int,
    attempt: int = 0,
    memory: int = 8,
) -> ProspectiveRunEvidence:
    cut = HH_HUBBARD_ACTIVATION_CUT_LADDER[index]
    resume = (
        None
        if horizon == HORIZON_PREFIX
        else prospective_run_action_id(
            bindings,
            arm=arm,
            horizon=HORIZON_PREFIX,
            activation_cut=cut,
        )
    )
    return ProspectiveRunEvidence(
        action_id=prospective_run_action_id(
            bindings, arm=arm, horizon=horizon, activation_cut=cut
        ),
        bindings_sha256=bindings.digest,
        arm=arm,
        horizon=horizon,
        activation_cut=cut,
        attempt=attempt,
        planner_sequence=sequence,
        requested_memory_gib=memory,
        state=STATE_COMPLETED,
        resume_from_action_id=resume,
        resume_run_receipt_sha256=_sha("a") if resume is not None else None,
        resume_run_sha256=_sha("b") if resume is not None else None,
        run_receipt_sha256=_sha("a"),
        run_sha256=_sha("b"),
        score_state=SCORE_COMPLETED if q is not None else SCORE_NOT_STARTED,
        score_receipt_sha256=_sha("c") if q is not None else None,
        score_sha256=_sha("d") if q is not None else None,
        metrics=None if q is None else _metrics(q),
    )


def _failed_memory(
    bindings: ProspectiveScientificBindings,
    *,
    action,
    attempt: int,
    memory: int,
    peak: float,
    sequence: int,
) -> ProspectiveRunEvidence:
    return ProspectiveRunEvidence(
        action_id=action.action_id,
        bindings_sha256=bindings.digest,
        arm=str(action.arm),
        horizon=str(action.horizon),
        activation_cut=float(action.activation_cut),
        attempt=attempt,
        planner_sequence=sequence,
        requested_memory_gib=memory,
        state=STATE_FAILED,
        resume_from_action_id=action.inputs.get("resume_from_action_id"),
        resume_run_receipt_sha256=action.inputs.get("resume_run_receipt_sha256"),
        resume_run_sha256=action.inputs.get("resume_run_sha256"),
        failure_kind=FAILURE_MEMORY,
        failure_receipt_sha256=_sha("e"),
        failure_reason="verified cgroup OOM",
        observed_peak_gib=peak,
    )


def _plan(bindings: ProspectiveScientificBindings, rows=(), safeguards=()):
    return plan_prospective_cell(
        bindings,
        POLICY,
        evidence=ProspectiveEvidence(tuple(rows), tuple(safeguards)),
    )


def test_frozen_ladder_is_exact_complete_and_never_requests_qiskit() -> None:
    assert len(HH_HUBBARD_ACTIVATION_CUT_LADDER) == 21
    assert HH_HUBBARD_ACTIVATION_CUT_LADDER[:5] == (
        1.0e-2,
        3.0e-3,
        1.0e-3,
        3.0e-4,
        1.0e-4,
    )
    assert HH_HUBBARD_ACTIVATION_CUT_LADDER[-2:] == (3.0e-12, 1.0e-12)

    bindings = _bindings()
    plan = _plan(bindings)
    assert plan.phase == "avqds_prefix_scout"
    assert len(plan.next_actions) == 1
    action = plan.next_actions[0]
    assert action.kind == KIND_RUN
    assert action.arm == ARM_AVQDS
    assert action.horizon == HORIZON_PREFIX
    assert action.activation_cut == 1.0e-2
    assert action.qiskit_compilation_requested is False
    assert action.inputs["qiskit_compilation"] is False
    assert action.inputs["bindings_sha256"] == bindings.digest
    assert action.inputs["seed_receipt_sha256"] == bindings.seed_receipt_sha256
    assert (
        action.inputs["reference_receipt_sha256"]
        == bindings.reference_receipt_sha256
    )
    assert action.inputs["accuracy_policy_sha256"] == policy_sha256(POLICY)


def test_same_planner_owns_the_frozen_hubbard_l4_lane() -> None:
    bindings = _bindings(
        cell_id="hubbard__l4__u4__weakslow",
    )
    plan = _plan(bindings)
    assert plan.next_actions[0].arm == ARM_AVQDS
    assert plan.next_actions[0].activation_cut == 1.0e-2
    assert plan.next_actions[0].inputs["lane_id"] == LANE_HUBBARD


def test_completed_raw_run_is_scored_before_the_ladder_advances() -> None:
    bindings = _bindings()
    raw = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=None,
        sequence=0,
    )
    plan = _plan(bindings, (raw,))
    assert len(plan.next_actions) == 1
    score = plan.next_actions[0]
    assert score.kind == KIND_SCORE
    assert score.inputs["arm"] == score.arm
    assert score.action_id == f"score::{raw.action_id}"
    assert score.inputs["run_receipt_sha256"] == raw.run_receipt_sha256
    assert score.inputs["run_sha256"] == raw.run_sha256
    assert (
        score.inputs["reference_manifest_sha256"]
        == bindings.reference_manifest_sha256
    )
    assert score.exact_reference_blind is False


def test_avqds_scout_advances_one_registered_rung_then_seeds_apm_at_its_cut() -> None:
    bindings = _bindings()
    outside = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=4.0,
        sequence=0,
    )
    plan = _plan(bindings, (outside,))
    assert plan.next_actions[0].arm == ARM_AVQDS
    assert plan.next_actions[0].ladder_index == 1

    scout = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=1,
        q=2.0,
        sequence=1,
    )
    plan = _plan(bindings, (outside, scout))
    first_apm = plan.next_actions[0]
    assert plan.phase == "apm_prefix_calibration"
    assert first_apm.arm == ARM_APM
    assert first_apm.activation_cut == scout.activation_cut
    assert first_apm.inputs["seeded_from_scout_action_id"] == scout.action_id


def test_apm_calibrates_independently_then_freezes_while_avqds_moves_tighter() -> None:
    bindings = _bindings()
    scout = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=2.8,
        sequence=0,
    )
    apm_outside = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=0,
        q=4.0,
        sequence=1,
    )
    plan = _plan(bindings, (scout, apm_outside))
    assert plan.phase == "apm_prefix_calibration"
    assert plan.next_actions[0].arm == ARM_APM
    assert plan.next_actions[0].ladder_index == 1

    anchor = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=1,
        q=1.0,
        sequence=2,
    )
    plan = _plan(bindings, (scout, apm_outside, anchor))
    assert plan.phase == "avqds_prefix_alignment"
    assert len(plan.next_actions) == 1
    aligned = plan.next_actions[0]
    assert aligned.arm == ARM_AVQDS
    assert aligned.ladder_index == 1
    assert plan.selected_prefix_action_ids == {ARM_APM: anchor.action_id}
    assert "tighter" in aligned.reason


def test_equal_aggregate_with_complementary_components_does_not_match() -> None:
    bindings = _bindings()
    avqds = replace(
        _completed(
            bindings,
            arm=ARM_AVQDS,
            horizon=HORIZON_PREFIX,
            index=0,
            q=1.0,
            sequence=0,
        ),
        metrics=_metrics(1.0, components=(0.2, 2.0, 0.2, 0.2)),
    )
    apm = replace(
        _completed(
            bindings,
            arm=ARM_APM,
            horizon=HORIZON_PREFIX,
            index=0,
            q=1.0,
            sequence=1,
        ),
        metrics=_metrics(1.0, components=(2.0, 0.2, 0.2, 0.2)),
    )
    plan = _plan(bindings, (avqds, apm))
    assert plan.phase == "avqds_prefix_alignment"
    assert plan.next_actions[0].arm == ARM_AVQDS
    assert plan.next_actions[0].ladder_index == 1


def test_deadbanded_negligible_component_difference_still_allows_pair() -> None:
    bindings = _bindings()
    avqds = replace(
        _completed(
            bindings,
            arm=ARM_AVQDS,
            horizon=HORIZON_PREFIX,
            index=0,
            q=1.0,
            sequence=0,
        ),
        metrics=_metrics(1.0, components=(0.04, 1.0, 1.0, 1.0)),
    )
    apm = replace(
        _completed(
            bindings,
            arm=ARM_APM,
            horizon=HORIZON_PREFIX,
            index=0,
            q=1.0,
            sequence=1,
        ),
        metrics=_metrics(1.0, components=(0.09, 1.0, 1.0, 1.0)),
    )
    plan = _plan(bindings, (avqds, apm))
    assert plan.phase == "full_horizon_continuation_or_score"
    assert {row.arm for row in plan.next_actions} == {ARM_APM, ARM_AVQDS}


def test_alignment_cycle_explores_unmeasured_frontier_without_monotonic_inference(
) -> None:
    bindings = _bindings()
    coarse_outside = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=4.0,
        sequence=0,
    )
    overaccurate_scout = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=1,
        q=0.4,
        sequence=1,
    )
    anchor = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=1,
        q=1.0,
        sequence=2,
    )
    plan = _plan(bindings, (coarse_outside, overaccurate_scout, anchor))
    assert plan.phase == "avqds_prefix_alignment"
    assert plan.next_actions[0].kind == KIND_RUN
    assert plan.next_actions[0].ladder_index == 2
    assert "tighter" in plan.next_actions[0].reason


def test_alignment_at_coarse_boundary_explores_other_unmeasured_direction() -> None:
    bindings = _bindings()
    overaccurate_scout = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=0.4,
        sequence=0,
    )
    anchor = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=1,
    )
    plan = _plan(bindings, (overaccurate_scout, anchor))
    assert plan.phase == "avqds_prefix_alignment"
    assert plan.next_actions[0].kind == KIND_RUN
    assert plan.next_actions[0].ladder_index == 1


def test_matched_prefix_emits_two_continuations_and_full_pair_completes() -> None:
    bindings = _bindings()
    avqds = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.2,
        sequence=0,
    )
    apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=1,
    )
    plan = _plan(bindings, (avqds, apm))
    assert plan.phase == "full_horizon_continuation_or_score"
    assert {row.arm for row in plan.next_actions} == {ARM_APM, ARM_AVQDS}
    assert all(row.horizon == HORIZON_FULL for row in plan.next_actions)
    for action in plan.next_actions:
        prefix = apm if action.arm == ARM_APM else avqds
        assert action.inputs["resume_from_action_id"] == prefix.action_id
        assert action.inputs["resume_run_receipt_sha256"] == prefix.run_receipt_sha256
        assert action.inputs["resume_run_sha256"] == prefix.run_sha256

    full_apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_FULL,
        index=0,
        q=1.0,
        sequence=2,
    )
    full_avqds = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_FULL,
        index=0,
        q=1.2,
        sequence=3,
    )
    complete = _plan(bindings, (avqds, apm, full_apm, full_avqds))
    assert complete.complete
    assert complete.cost_comparison_admissible
    assert not complete.unresolved
    assert complete.next_actions == ()
    assert complete.selected_full_action_ids == {
        ARM_APM: full_apm.action_id,
        ARM_AVQDS: full_avqds.action_id,
    }


def test_t20_apm_guard_miss_reopens_apm_at_one_adjacent_tighter_cut() -> None:
    bindings = _bindings()
    prefix_avqds = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=0,
    )
    prefix_apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=1,
    )
    full_apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_FULL,
        index=0,
        q=4.0,
        sequence=2,
    )
    full_avqds = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_FULL,
        index=0,
        q=1.0,
        sequence=3,
    )
    plan = _plan(
        bindings, (prefix_avqds, prefix_apm, full_apm, full_avqds)
    )
    assert plan.phase == "apm_prefix_calibration"
    assert plan.next_actions[0].arm == ARM_APM
    assert plan.next_actions[0].horizon == HORIZON_PREFIX
    assert plan.next_actions[0].ladder_index == 1


def test_t20_pair_miss_reopens_bidirectional_avqds_alignment() -> None:
    bindings = _bindings()
    prefix_avqds = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=0,
    )
    prefix_apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=1,
    )
    full_apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_FULL,
        index=0,
        q=1.0,
        sequence=2,
    )
    full_avqds = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_FULL,
        index=0,
        q=2.8,
        sequence=3,
    )
    plan = _plan(
        bindings, (prefix_avqds, prefix_apm, full_apm, full_avqds)
    )
    assert plan.phase == "avqds_prefix_alignment"
    assert plan.next_actions[0].arm == ARM_AVQDS
    assert plan.next_actions[0].ladder_index == 1
    assert "tighter" in plan.next_actions[0].reason


def test_avqds_floor_runs_one_apm_diagnostic_then_records_unresolved() -> None:
    bindings = _bindings()
    avqds = tuple(
        _completed(
            bindings,
            arm=ARM_AVQDS,
            horizon=HORIZON_PREFIX,
            index=index,
            q=4.0,
            sequence=index,
        )
        for index in range(len(HH_HUBBARD_ACTIVATION_CUT_LADDER))
    )
    plan = _plan(bindings, avqds)
    assert plan.phase == "avqds_exhausted_apm_diagnostic"
    diagnostic = plan.next_actions[0]
    assert diagnostic.arm == ARM_APM
    assert diagnostic.ladder_index == len(HH_HUBBARD_ACTIVATION_CUT_LADDER) - 1
    assert "diagnostic" in diagnostic.reason

    apm = _completed(
        bindings,
        arm=ARM_APM,
        horizon=HORIZON_PREFIX,
        index=len(HH_HUBBARD_ACTIVATION_CUT_LADDER) - 1,
        q=1.0,
        sequence=100,
    )
    safeguard_plan = _plan(bindings, (*avqds, apm))
    assert safeguard_plan.next_actions[0].kind == KIND_RECORD_SAFEGUARD
    assert safeguard_plan.phase == "record_ladder_unresolved"
    assert not safeguard_plan.cost_comparison_admissible


def test_one_memory_retry_uses_next_four_gib_tier_then_becomes_infeasible() -> None:
    bindings = _bindings()
    initial = _plan(bindings).next_actions[0]
    failed0 = _failed_memory(
        bindings,
        action=initial,
        attempt=0,
        memory=8,
        peak=9.0,
        sequence=0,
    )
    retry_plan = _plan(bindings, (failed0,))
    retry = retry_plan.next_actions[0]
    assert retry.kind == KIND_RUN
    assert retry.action_id == initial.action_id
    assert retry.attempt == 1
    assert retry.requested_memory_gib == 16
    assert retry.inputs["resource_only_retry"] is True

    failed1 = _failed_memory(
        bindings,
        action=retry,
        attempt=1,
        memory=16,
        peak=14.0,
        sequence=1,
    )
    terminal_plan = _plan(bindings, (failed0, failed1))
    safeguard = terminal_plan.next_actions[0]
    assert safeguard.kind == KIND_RECORD_SAFEGUARD
    assert safeguard.inputs["safeguard_reason"] == SAFEGUARD_OPERATIONALLY_INFEASIBLE

    receipt = ProspectiveSafeguardEvidence(
        safeguard_id=safeguard.action_id,
        bindings_sha256=bindings.digest,
        reason=SAFEGUARD_OPERATIONALLY_INFEASIBLE,
        subject_action_id=initial.action_id,
        receipt_sha256=_sha("f"),
        planner_sequence=2,
    )
    complete = _plan(bindings, (failed0, failed1), (receipt,))
    assert complete.complete
    assert complete.unresolved
    assert not complete.cost_comparison_admissible


def test_nonmemory_failure_blocks_without_mutating_the_scientific_action() -> None:
    bindings = _bindings()
    action = _plan(bindings).next_actions[0]
    failed = ProspectiveRunEvidence(
        action_id=action.action_id,
        bindings_sha256=bindings.digest,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        activation_cut=1.0e-2,
        attempt=0,
        planner_sequence=0,
        requested_memory_gib=8,
        state=STATE_FAILED,
        failure_kind=FAILURE_OTHER,
        failure_receipt_sha256=_sha("e"),
        failure_reason="source package validation failed",
    )
    plan = _plan(bindings, (failed,))
    assert plan.blocked
    assert plan.next_actions == ()
    assert plan.dispositions[0].disposition == "blocked"


def test_evidence_and_policy_drift_fail_closed() -> None:
    bindings = _bindings()
    with pytest.raises(ProspectivePlannerError, match="policy digest"):
        plan_prospective_cell(
            replace(bindings, accuracy_policy_sha256=_sha("0")), POLICY
        )

    row = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=1.0,
        sequence=0,
    )
    with pytest.raises(ProspectivePlannerError, match="different physical"):
        _plan(bindings, (replace(row, bindings_sha256=_sha("9")),))

    with pytest.raises(ProspectivePlannerError, match="Qiskit"):
        _bindings(qiskit_compilation=True)


def test_full_horizon_evidence_must_bind_an_existing_prefix_receipt() -> None:
    bindings = _bindings()
    orphan = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_FULL,
        index=0,
        q=1.0,
        sequence=0,
    )
    with pytest.raises(ProspectivePlannerError, match="no completed verified prefix"):
        _plan(bindings, (orphan,))


def test_prefix_history_cannot_skip_an_unmeasured_registered_rung() -> None:
    bindings = _bindings()
    first = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=0,
        q=4.0,
        sequence=0,
    )
    skipped = _completed(
        bindings,
        arm=ARM_AVQDS,
        horizon=HORIZON_PREFIX,
        index=2,
        q=2.0,
        sequence=1,
    )
    with pytest.raises(ProspectivePlannerError, match="skips"):
        _plan(bindings, (first, skipped))
