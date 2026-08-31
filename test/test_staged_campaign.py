"""Replay and anti-drift tests for the staged Paper-II campaign policy."""

from __future__ import annotations

import json
import math

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import _build_parser
from pipelines.time_dynamics.staged_campaign import (
    ACTION_RECORD_SAFEGUARD,
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellKey,
    METHOD_APM,
    METHOD_AVQDS,
    RungRecord,
    STAGE_ACTIVE_FRONTIER,
    STAGE_COMPLETE,
    STAGE_PROMOTE_AVQDS,
    STAGE_SCREEN_APM,
    STAGE_SCREEN_AVQDS,
    STATUS_ACCURACY_QUALIFIED,
    STATUS_REQUIRES_THRESHOLD_COARSENING,
    STATUS_REQUIRES_THRESHOLD_EXTENSION,
    StagedCampaign,
    ThresholdSchedule,
    factorial_cells,
)
from pipelines.time_dynamics.staged_campaign_artifacts import (
    record_safeguard_action,
)


CONTROLLERS = ("delta_theta_5e-5", "state_motion_5e-5")
DRIVES = ("fastweak", "weakslow")
CELLS = factorial_cells(
    regime="hh_snake_nph1",
    drives=DRIVES,
    controllers=CONTROLLERS,
)
POLICY = AccuracyPolicy()
WINDOW_POLICY = AccuracyPolicy.mean_infidelity_window(
    floor=2.0e-6,
    ceiling=8.0e-6,
)
MATCHED_POLICY = AccuracyPolicy.response_normalized_matched_pair(
    response_scales={
        drive: {
            horizon: {
                "energy_rms_excursion": 0.1,
                "doublon_rms_excursion": 0.1,
                "max_site_occupation_rms_excursion": 0.1,
            }
            for horizon in ("t5", "t20")
        }
        for drive in DRIVES
    },
    response_relative_target=0.05,
    response_relative_guard=0.15,
    matched_accuracy_factor=2.0,
    infidelity_ceiling=1.0e-4,
)


def _campaign(*, maximum_rungs: int = 8) -> StagedCampaign:
    return StagedCampaign(
        cells=CELLS,
        accuracy_policy=POLICY,
        threshold_schedule=ThresholdSchedule(
            {key: (1.0e-3, 3.0e-4) for key in CELLS},
            maximum_rungs_per_horizon=maximum_rungs,
        ),
    )


def _matched_campaign_one(*, maximum_rungs: int = 8) -> StagedCampaign:
    cells = tuple(
        CellKey("hh_snake_nph1", "weakslow", method, CONTROLLERS[0])
        for method in (METHOD_AVQDS, METHOD_APM)
    )
    return StagedCampaign(
        cells=cells,
        accuracy_policy=MATCHED_POLICY,
        threshold_schedule=ThresholdSchedule(
            {key: (1.0e-3,) for key in cells},
            maximum_rungs_per_horizon=maximum_rungs,
        ),
        comparator_first=True,
    )


def _append_matched(
    ledger: CampaignLedger,
    action,
    metrics: AccuracyMetrics,
) -> CampaignLedger:
    return ledger.append(
        RungRecord.completed(
            key=action.key,
            horizon=action.horizon,
            threshold=action.threshold,
            run_path=f"output/{action.action_id}/run.json",
            metrics=metrics,
            policy=MATCHED_POLICY,
            resume_from_run_path=action.resume_from_run_path,
        )
    )


def _metrics(value: float) -> AccuracyMetrics:
    return AccuracyMetrics(value, value / 2.0, value / 3.0, value / 4.0)


def _record(
    ledger: CampaignLedger,
    action,
    value: float,
) -> CampaignLedger:
    return ledger.append(
        RungRecord.completed(
            key=action.key,
            horizon=action.horizon,
            threshold=action.threshold,
            run_path=f"output/{action.action_id}/run.json",
            metrics=_metrics(value),
            policy=POLICY,
            resume_from_run_path=action.resume_from_run_path,
        )
    )


def test_componentwise_policy_qualifies_better_than_preferred_band() -> None:
    assert POLICY.classify(_metrics(5.0e-5)) == STATUS_ACCURACY_QUALIFIED
    very_accurate = _metrics(1.0e-7)
    assert POLICY.qualifies(very_accurate)
    assert POLICY.band(very_accurate) == "better_than_preferred_band"
    failed = AccuracyMetrics(2.0e-4, 1.0e-8, 1.0e-8, 1.0e-8)
    assert POLICY.classify(failed) == STATUS_REQUIRES_THRESHOLD_EXTENSION


def test_infidelity_is_required_for_new_campaign_records() -> None:
    with pytest.raises(ValueError, match="requires statevector-derived infidelity"):
        POLICY.qualifies(AccuracyMetrics(1.0e-5, 1.0e-5, 1.0e-5, None))


def test_infidelity_window_demotes_observables_to_diagnostics() -> None:
    inside = AccuracyMetrics(9.0e-2, 8.0e-2, 7.0e-2, 5.0e-6)
    assert WINDOW_POLICY.classify(inside) == STATUS_ACCURACY_QUALIFIED
    assert WINDOW_POLICY.band(inside) == "infidelity_window"

    over_resolved = AccuracyMetrics(9.0e-2, 8.0e-2, 7.0e-2, 1.0e-6)
    assert (
        WINDOW_POLICY.classify(over_resolved)
        == STATUS_REQUIRES_THRESHOLD_COARSENING
    )
    assert WINDOW_POLICY.band(over_resolved) == "below_infidelity_window"

    too_inaccurate = AccuracyMetrics(1.0e-12, 1.0e-12, 1.0e-12, 9.0e-6)
    assert (
        WINDOW_POLICY.classify(too_inaccurate)
        == STATUS_REQUIRES_THRESHOLD_EXTENSION
    )
    assert WINDOW_POLICY.band(too_inaccurate) == "above_infidelity_window"


def test_response_policy_separates_target_guard_and_pair_matching() -> None:
    avqds_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_AVQDS, CONTROLLERS[0]
    )
    apm_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_APM, CONTROLLERS[0]
    )
    target = AccuracyMetrics(0.004, 0.003, 0.002, 5.0e-5)
    broad = AccuracyMetrics(0.008, 0.004, 0.003, 6.0e-5)
    outside = AccuracyMetrics(0.016, 0.001, 0.001, 1.0e-6)
    absolute_guard_failure = AccuracyMetrics(0.004, 0.006, 0.002, 5.0e-5)

    assert MATCHED_POLICY.qualifies(target, key=avqds_key, horizon="t20")
    assert not MATCHED_POLICY.qualifies(broad, key=apm_key, horizon="t20")
    assert MATCHED_POLICY.comparison_eligible(
        broad, key=apm_key, horizon="t20"
    )
    assert not MATCHED_POLICY.comparison_eligible(
        outside, key=apm_key, horizon="t20"
    )
    assert not MATCHED_POLICY.comparison_eligible(
        absolute_guard_failure, key=apm_key, horizon="t20"
    )
    assert MATCHED_POLICY.pair_matches(
        broad,
        left_key=apm_key,
        right_metrics=target,
        right_key=avqds_key,
        horizon="t20",
    )


def test_response_policy_applies_the_broad_guard_to_fidelity() -> None:
    key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_AVQDS, CONTROLLERS[0]
    )
    matched_guard_only = AccuracyMetrics(0.004, 0.003, 0.002, 2.0e-4)
    outside_broad_guard = AccuracyMetrics(0.004, 0.003, 0.002, 4.0e-4)

    assert not MATCHED_POLICY.qualifies(
        matched_guard_only,
        key=key,
        horizon="t20",
    )
    assert MATCHED_POLICY.comparison_eligible(
        matched_guard_only,
        key=key,
        horizon="t20",
    )
    assert not MATCHED_POLICY.comparison_eligible(
        outside_broad_guard,
        key=key,
        horizon="t20",
    )


def test_response_policy_uses_q_components_and_rms_burden_without_mutating_raw_metrics() -> None:
    key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_AVQDS, CONTROLLERS[0]
    )
    metrics = AccuracyMetrics(0.005, 0.0025, 0.0005, 2.5e-5)
    raw_metrics = metrics.to_json_dict()

    components = MATCHED_POLICY.response_components(
        metrics,
        key=key,
        horizon="t20",
    )

    assert components == pytest.approx(
        {
            "mean_abs_energy_error": 1.0,
            "mean_abs_doublon_error": 0.5,
            "mean_max_abs_site_occupation_error": 0.1,
            "mean_infidelity": 0.25,
        }
    )
    assert MATCHED_POLICY.accuracy_burden(
        metrics,
        key=key,
        horizon="t20",
    ) == pytest.approx(
        math.sqrt((1.0**2 + 0.5**2 + 0.1**2 + 0.25**2) / 4.0)
    )
    assert metrics.to_json_dict() == raw_metrics
    assert (
        AccuracyPolicy.from_json_dict(MATCHED_POLICY.to_json_dict())
        == MATCHED_POLICY
    )
    serialized = MATCHED_POLICY.to_json_dict()
    assert serialized["matched_policy_version"] == 2
    assert serialized["component_normalization"] == "target_normalized_q"
    assert serialized["aggregate_burden"] == "rms_q_components"
    assert serialized["comparison_guard_components"] == (
        "all_target_normalized_q_including_fidelity"
    )
    assert serialized["absolute_safety_components"] == [
        "mean_abs_energy_error",
        "mean_abs_doublon_error",
        "mean_max_abs_site_occupation_error",
    ]
    assert serialized["infidelity_role"] == (
        "target_normalized_component_with_broad_guard_not_absolute_safety"
    )
    assert serialized["matched_component_deadband"] == pytest.approx(
        0.10
    )


def test_pair_matching_rejects_complementary_components_with_equal_q_burdens(
) -> None:
    avqds_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_AVQDS, CONTROLLERS[0]
    )
    apm_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_APM, CONTROLLERS[0]
    )
    energy_heavy = AccuracyMetrics(0.005, 0.0005, 0.0005, 1.0e-5)
    doublon_heavy = AccuracyMetrics(0.0005, 0.005, 0.0005, 1.0e-5)

    assert MATCHED_POLICY.comparison_eligible(
        energy_heavy, key=apm_key, horizon="t20"
    )
    assert MATCHED_POLICY.comparison_eligible(
        doublon_heavy, key=avqds_key, horizon="t20"
    )
    assert MATCHED_POLICY.accuracy_burden(
        energy_heavy, key=apm_key, horizon="t20"
    ) == pytest.approx(
        MATCHED_POLICY.accuracy_burden(
            doublon_heavy, key=avqds_key, horizon="t20"
        )
    )
    assert not MATCHED_POLICY.pair_matches(
        energy_heavy,
        left_key=apm_key,
        right_metrics=doublon_heavy,
        right_key=avqds_key,
        horizon="t20",
    )

    low_fidelity_burden = AccuracyMetrics(0.0025, 0.0025, 0.0025, 1.0e-5)
    high_fidelity_burden = AccuracyMetrics(0.0025, 0.0025, 0.0025, 1.0e-4)
    assert not MATCHED_POLICY.pair_matches(
        low_fidelity_burden,
        left_key=apm_key,
        right_metrics=high_fidelity_burden,
        right_key=avqds_key,
        horizon="t20",
    )


def test_pair_matching_applies_component_deadband_below_point_one_q() -> None:
    avqds_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_AVQDS, CONTROLLERS[0]
    )
    apm_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_APM, CONTROLLERS[0]
    )
    left = AccuracyMetrics(0.0002, 0.0025, 0.0025, 5.0e-5)
    right = AccuracyMetrics(0.00045, 0.0025, 0.0025, 5.0e-5)

    left_q = MATCHED_POLICY.response_components(
        left, key=apm_key, horizon="t20"
    )
    right_q = MATCHED_POLICY.response_components(
        right, key=avqds_key, horizon="t20"
    )
    assert (
        right_q["mean_abs_energy_error"] / left_q["mean_abs_energy_error"]
        > 2.0
    )
    assert MATCHED_POLICY.pair_matches(
        left,
        left_key=apm_key,
        right_metrics=right,
        right_key=avqds_key,
        horizon="t20",
    )


def test_pair_reclassification_admits_latest_matched_t20_records() -> None:
    avqds_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_AVQDS, CONTROLLERS[0]
    )
    apm_key = CellKey(
        "hh_snake_nph1", "weakslow", METHOD_APM, CONTROLLERS[0]
    )
    metrics = AccuracyMetrics(0.008, 0.004, 0.003, 6.0e-5)
    ledger = CampaignLedger(
        (
            RungRecord.completed(
                key=avqds_key,
                horizon="t20",
                threshold=1.0e-6,
                run_path="output/avqds/run.json",
                metrics=metrics,
                policy=MATCHED_POLICY,
                resume_from_run_path="output/avqds/t5/run.json",
            ),
            RungRecord.completed(
                key=apm_key,
                horizon="t20",
                threshold=1.0e-6,
                run_path="output/apm/run.json",
                metrics=metrics,
                policy=MATCHED_POLICY,
                resume_from_run_path="output/apm/t5/run.json",
            ),
        ),
        qualification_policy=MATCHED_POLICY.to_json_dict(),
    ).reclassified(MATCHED_POLICY)

    assert {record.status for record in ledger.records} == {
        STATUS_ACCURACY_QUALIFIED
    }
    assert all("matched within factor" in record.reason for record in ledger.records)


def test_scout_opens_apm_at_same_cut_then_freezes_anchor_and_tightens_avqds() -> None:
    campaign = _matched_campaign_one()
    ledger = CampaignLedger(qualification_policy=MATCHED_POLICY.to_json_dict())
    first = campaign.next_actions(ledger)
    assert len(first) == 1 and first[0].key.method == METHOD_AVQDS
    broad_scout = AccuracyMetrics(0.010, 0.004, 0.003, 5.0e-5)
    ledger = _append_matched(ledger, first[0], broad_scout)

    seeded = campaign.next_actions(ledger)
    assert len(seeded) == 1 and seeded[0].key.method == METHOD_APM
    assert seeded[0].threshold == pytest.approx(first[0].threshold)
    target_anchor = AccuracyMetrics(0.004, 0.002, 0.002, 5.0e-5)
    ledger = _append_matched(ledger, seeded[0], target_anchor)

    aligned = campaign.next_actions(ledger)
    assert len(aligned) == 1 and aligned[0].key.method == METHOD_AVQDS
    assert aligned[0].threshold == pytest.approx(first[0].threshold / 3.0)
    assert not any(action.key.method == METHOD_APM for action in aligned)


def test_overaccurate_avqds_moves_coarser_relative_to_apm_anchor() -> None:
    campaign = _matched_campaign_one()
    avqds, apm = campaign.cells
    overaccurate = AccuracyMetrics(2.0e-4, 2.0e-4, 2.0e-4, 1.0e-6)
    anchor = AccuracyMetrics(0.008, 0.004, 0.003, 6.0e-5)
    ledger = CampaignLedger(
        (
            RungRecord.completed(
                key=avqds,
                horizon="t5",
                threshold=1.0e-3,
                run_path="output/avqds/run.json",
                metrics=overaccurate,
                policy=MATCHED_POLICY,
            ),
            RungRecord.completed(
                key=apm,
                horizon="t5",
                threshold=1.0e-3,
                run_path="output/apm/run.json",
                metrics=anchor,
                policy=MATCHED_POLICY,
            ),
        ),
        qualification_policy=MATCHED_POLICY.to_json_dict(),
    )
    actions = campaign.next_actions(ledger)
    assert len(actions) == 1 and actions[0].key.method == METHOD_AVQDS
    assert actions[0].threshold == pytest.approx(3.0e-3)


def test_matched_t5_pair_promotes_avqds_first_and_keeps_t5_nonfinal() -> None:
    campaign = _matched_campaign_one()
    avqds, apm = campaign.cells
    broad = AccuracyMetrics(0.008, 0.004, 0.003, 6.0e-5)
    ledger = CampaignLedger(
        (
            RungRecord.completed(
                key=avqds,
                horizon="t5",
                threshold=1.0e-3,
                run_path="output/avqds/t5/run.json",
                metrics=broad,
                policy=MATCHED_POLICY,
            ),
            RungRecord.completed(
                key=apm,
                horizon="t5",
                threshold=1.0e-3,
                run_path="output/apm/t5/run.json",
                metrics=broad,
                policy=MATCHED_POLICY,
            ),
        ),
        qualification_policy=MATCHED_POLICY.to_json_dict(),
    )
    actions = campaign.next_actions(ledger)
    assert len(actions) == 1
    assert actions[0].key.method == METHOD_AVQDS
    assert actions[0].horizon == "t20"
    assert actions[0].resume_from_run_path == "output/avqds/t5/run.json"


def test_avqds_exhaustion_opens_exactly_one_tightest_apm_diagnostic() -> None:
    campaign = _matched_campaign_one(maximum_rungs=1)
    ledger = CampaignLedger(qualification_policy=MATCHED_POLICY.to_json_dict())
    scout = campaign.next_actions(ledger)[0]
    outside = AccuracyMetrics(0.02, 0.004, 0.003, 5.0e-5)
    ledger = _append_matched(ledger, scout, outside)
    safeguard = campaign.next_actions(ledger)[0]
    assert safeguard.action == ACTION_RECORD_SAFEGUARD
    ledger = record_safeguard_action(ledger=ledger, action=safeguard)
    diagnostic = campaign.next_actions(ledger)
    assert len(diagnostic) == 1
    assert diagnostic[0].key.method == METHOD_APM
    assert diagnostic[0].threshold == pytest.approx(scout.threshold)
    assert "diagnostic" in diagnostic[0].reason


def test_ledger_reclassification_preserves_artifacts_and_binds_policy() -> None:
    campaign = _campaign()
    action = campaign.next_actions(CampaignLedger())[0]
    original = _record(CampaignLedger(), action, 5.0e-5)
    replayed = original.reclassified(WINDOW_POLICY)

    assert original.records[0].status == STATUS_ACCURACY_QUALIFIED
    assert replayed.records[0].status == STATUS_REQUIRES_THRESHOLD_EXTENSION
    assert replayed.records[0].run_path == original.records[0].run_path
    assert replayed.records[0].metrics == original.records[0].metrics
    assert replayed.qualification_policy == WINDOW_POLICY.to_json_dict()


def test_threshold_schedule_enumerates_prior_then_extends_without_error_assumption() -> None:
    key = CELLS[0]
    schedule = _campaign().threshold_schedule
    assert schedule.next(key, ()).threshold == pytest.approx(1.0e-3)
    assert schedule.next(key, (1.0e-3,)).threshold == pytest.approx(3.0e-4)
    assert schedule.next(key, (1.0e-3, 3.0e-4)).threshold == pytest.approx(1.0e-4)


def test_threshold_schedule_coarsens_for_an_over_resolved_window_rung() -> None:
    key = CELLS[0]
    schedule = _campaign().threshold_schedule
    decision = schedule.next(key, (1.0e-3,), direction="coarsen")
    assert decision.status == STATUS_REQUIRES_THRESHOLD_COARSENING
    assert decision.threshold == pytest.approx(3.0e-3)


def test_coarsening_skips_roundoff_equivalent_factor_three_rungs() -> None:
    key = CELLS[0]
    schedule = ThresholdSchedule(
        {key: (3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7)},
        maximum_rungs_per_horizon=24,
    )
    decision = schedule.next(
        key,
        (3.0e-6, 1.0e-6, 3.0e-7, 1.0e-7, 3.333333333333e-8),
        direction="coarsen",
    )
    assert decision.threshold == pytest.approx(9.0e-6)


def test_qualified_cell_promotes_while_other_t5_cells_keep_screening() -> None:
    campaign = _campaign()
    ledger = CampaignLedger()
    actions = campaign.next_actions(ledger)
    assert actions
    assert {action.stage for action in actions} == {
        STAGE_SCREEN_AVQDS,
        STAGE_SCREEN_APM,
    }
    assert {action.key.method for action in actions} == {METHOD_AVQDS, METHOD_APM}
    assert {action.horizon for action in actions} == {"t5"}

    qualified = actions[0]
    ledger = _record(ledger, qualified, 5.0e-5)
    mixed = campaign.next_actions(ledger)
    screening = tuple(action for action in mixed if action.horizon == "t5")
    promotion = tuple(action for action in mixed if action.horizon == "t20")

    assert len(screening) == len(actions) - 1
    assert len(promotion) == 1
    assert promotion[0].key == qualified.key
    assert promotion[0].resume_from_run_path
    assert campaign.stage(ledger) == STAGE_ACTIVE_FRONTIER


def test_all_qualified_t5_cells_emit_the_complete_t20_frontier() -> None:
    campaign = _campaign()
    ledger = CampaignLedger()
    for action in campaign.next_actions(ledger):
        ledger = _record(ledger, action, 5.0e-5)
    promotion = campaign.next_actions(ledger)
    assert {action.key.method for action in promotion} == {METHOD_AVQDS, METHOD_APM}
    assert {action.horizon for action in promotion} == {"t20"}
    assert all(action.resume_from_run_path for action in promotion)


def test_failed_rung_extends_only_that_cell_and_replay_is_idempotent() -> None:
    campaign = _campaign()
    ledger = CampaignLedger()
    first = campaign.next_actions(ledger)
    failed = first[0]
    ledger = _record(ledger, failed, 2.0e-4)
    for action in first[1:]:
        ledger = _record(ledger, action, 5.0e-5)
    replay_one = campaign.next_actions(ledger)
    replay_two = campaign.next_actions(ledger)
    assert replay_one == replay_two
    screening = tuple(action for action in replay_one if action.horizon == "t5")
    promotion = tuple(action for action in replay_one if action.horizon == "t20")
    assert len(screening) == 1
    assert screening[0].key == failed.key
    assert screening[0].threshold == pytest.approx(3.0e-4)
    assert len(promotion) == len(first) - 1


def test_safeguard_is_explicit_instead_of_silently_stopping() -> None:
    campaign = _campaign(maximum_rungs=1)
    ledger = CampaignLedger()
    actions = campaign.next_actions(ledger)
    failed = actions[0]
    ledger = _record(ledger, failed, 2.0e-4)
    for action in actions[1:]:
        ledger = _record(ledger, action, 5.0e-5)
    next_actions = campaign.next_actions(ledger)
    safeguards = tuple(
        action for action in next_actions if action.action == ACTION_RECORD_SAFEGUARD
    )
    assert len(safeguards) == 1
    assert safeguards[0].key == failed.key
    assert safeguards[0].threshold != failed.threshold
    recorded = record_safeguard_action(ledger=ledger, action=safeguards[0])
    assert recorded.records[-1].status == "safeguard_tripped"


def test_ledger_round_trip_is_append_only_and_conflicts_fail(tmp_path) -> None:
    campaign = _campaign()
    action = campaign.next_actions(CampaignLedger())[0]
    ledger = _record(CampaignLedger(), action, 5.0e-5)
    path = tmp_path / "ledger.json"
    ledger.write(path)
    assert CampaignLedger.read(path) == ledger
    assert json.loads(path.read_text())["records"]
    same = ledger.append(ledger.records[0])
    assert same is ledger
    with pytest.raises(ValueError, match="different record"):
        ledger.append(
            RungRecord.completed(
                key=action.key,
                horizon=action.horizon,
                threshold=action.threshold,
                run_path="different/run.json",
                metrics=_metrics(5.0e-5),
                policy=POLICY,
            )
        )


def test_run_command_uses_method_specific_cut_statevectors_and_global_resume_grid() -> None:
    campaign = _campaign()
    parser = _build_parser()
    ledger = CampaignLedger()
    for action in campaign.next_actions(ledger):
        ledger = _record(ledger, action, 5.0e-5)
    action = campaign.next_actions(ledger)[0]
    command = campaign.build_run_command(
        action,
        seed_path="seed.json",
        output_json="out/run.json",
        numerics=runs.EULER_RIDGE1E6_NUMERICS.numerics_id,
        gate=runs.MCLACHLAN_L2_GATE.gate_id,
        search_profile=runs.DIAGNOSTIC_FAST_SEARCH.profile_id,
    )
    parsed = parser.parse_args(command.argv())
    assert parsed.avqds_l2_cut == pytest.approx(action.threshold)
    assert parsed.record_statevector is True
    assert parsed.resume_from_run_json == action.resume_from_run_path
    assert parsed.t_initial is None
    assert parsed.t_final == pytest.approx(20.0)
    assert parsed.num_times == 376


def test_complete_campaign_has_no_actions() -> None:
    campaign = _campaign()
    ledger = CampaignLedger()
    while True:
        actions = campaign.next_actions(ledger)
        if not actions:
            break
        for action in actions:
            ledger = _record(ledger, action, 5.0e-5)
    assert campaign.stage(ledger) == STAGE_COMPLETE
    assert campaign.next_actions(ledger) == ()
