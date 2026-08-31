"""Executor-facing contract for the recurring Paper-II staged campaign."""

from __future__ import annotations

import json
from pathlib import Path

from pipelines.time_dynamics.campaigns import paper_ii_staged_accuracy_v1 as staged
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellAction,
    RungRecord,
)


def test_declared_campaign_is_two_methods_by_three_controllers_by_six_drives() -> None:
    campaign = staged.build_campaign(prior_root=None)
    assert len(campaign.cells) == 2 * 3 * 6
    assert len({key.drive for key in campaign.cells}) == 6
    assert len({key.controller for key in campaign.cells}) == 3
    assert {key.method for key in campaign.cells} == {"avqds", "exchange"}
    policy = campaign.accuracy_policy
    assert policy.mode == "response_normalized_matched_pair"
    assert policy.response_relative_target == 0.05
    assert policy.response_relative_guard == 0.15
    assert policy.matched_accuracy_factor == 2.0
    assert policy.infidelity_ceiling == 1.0e-4
    assert policy.response_scales["schema"] == "paper_ii_response_scale_index_v2"
    assert set(
        policy.response_scales["regimes"]["hh_snake_nph1"]["controllers"][
            "state_motion_5e-5"
        ]["drives"]
    ) == {
        "weakslow",
        "fastweak",
        "midres",
        "slowstrong",
        "strongfast",
        "fastfast",
    }


def test_legacy_submission_policy_is_available_only_for_evidence_replay() -> None:
    campaign = staged.build_campaign(
        prior_root=None,
        submission_policy_replay="componentwise_1e-4",
    )
    policy = campaign.accuracy_policy
    assert policy.mode == "componentwise_ceiling"
    assert policy.ceiling == 1.0e-4
    assert policy.require_infidelity is True


def test_run_skill_and_overnight_contract_match_live_policy_and_refresh_rule() -> None:
    root = Path(__file__).resolve().parents[1]
    run_skill = (root / "agent_guidance/skills/paper-ii-run/SKILL.md").read_text()
    overnight = (
        root / "agent_guidance/time-dynamics/paper-ii-overnight-campaign.md"
    ).read_text()

    assert "response-normalized" in run_skill
    assert "factor of two" in run_skill
    assert "infidelity-window study is preserved historical evidence only" in run_skill
    assert "Snapshot-to-report transaction" in overnight
    assert "Completed but awaiting ingestion" in overnight
    assert "`prior_root` is tri-state provenance" in overnight


def test_first_worklist_establishes_avqds_comparators_and_records_statevectors(
    tmp_path,
) -> None:
    campaign = staged.build_campaign(prior_root=None)
    payload = staged._next_payload(
        campaign=campaign,
        ledger=CampaignLedger(),
        output_root=tmp_path,
        search_profile="diagnostic_fast_v1",
    )
    assert payload["stage"] == "screen_t5_frontier"
    assert payload["action_count"] == 3 * 6
    assert {row["key"]["method"] for row in payload["actions"]} == {"avqds"}
    for row in payload["actions"]:
        assert row["horizon"] == "t5"
        assert "--record-statevector" in row["runner_argv"]


def test_action_files_round_trip_the_scientific_action(tmp_path) -> None:
    campaign = staged.build_campaign(prior_root=None)
    payload = staged._next_payload(
        campaign=campaign,
        ledger=CampaignLedger(),
        output_root=tmp_path / "output",
        search_profile="diagnostic_fast_v1",
    )
    staged._write_action_files(payload, tmp_path / "actions")
    first = sorted((tmp_path / "actions").glob("*.json"))[0]
    action = CellAction.from_json_dict(json.loads(first.read_text()))
    assert action.action_id == json.loads(first.read_text())["action_id"]


def test_next_command_prints_nonempty_machine_readable_worklist(tmp_path, capsys) -> None:
    worklist = tmp_path / "nested" / "worklist.json"
    status = staged.main(
        [
            "next",
            "--ledger",
            str(tmp_path / "ledger.json"),
            "--output-root",
            str(tmp_path / "output"),
            "--search-profile",
            "diagnostic_fast_v1",
            "--worklist-json",
            str(worklist),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert payload["action_count"] == 18
    assert payload["actions"][0]["runner_shell"].strip()
    assert json.loads(worklist.read_text()) == payload


def test_apm_is_seeded_at_t5_then_reopens_after_matched_avqds_t20_scout() -> None:
    campaign = staged.build_campaign(prior_root=None)
    ledger = CampaignLedger(
        qualification_policy=campaign.accuracy_policy.to_json_dict()
    )
    first = campaign.next_actions(ledger)[0]
    metrics = AccuracyMetrics(5.0e-5, 5.0e-5, 5.0e-5, 5.0e-5)
    screen_path = f"output/{first.action_id}/run.json"
    ledger = ledger.append(
        RungRecord.completed(
            key=first.key,
            horizon="t5",
            threshold=first.threshold,
            run_path=screen_path,
            metrics=metrics,
            policy=campaign.accuracy_policy,
        )
    )
    apm_screen = next(
        action
        for action in campaign.next_actions(ledger)
        if action.key.drive == first.key.drive
        and action.key.controller == first.key.controller
        and action.key.method == "exchange"
        and action.horizon == "t5"
    )
    assert apm_screen.threshold == first.threshold
    ledger = ledger.append(
        RungRecord.completed(
            key=apm_screen.key,
            horizon="t5",
            threshold=apm_screen.threshold,
            run_path=f"output/{apm_screen.action_id}/run.json",
            metrics=metrics,
            policy=campaign.accuracy_policy,
        )
    )
    promotion = next(
        action
        for action in campaign.next_actions(ledger)
        if action.key == first.key and action.horizon == "t20"
    )
    ledger = ledger.append(
        RungRecord.completed(
            key=promotion.key,
            horizon="t20",
            threshold=promotion.threshold,
            run_path=f"output/{promotion.action_id}/run.json",
            metrics=metrics,
            policy=campaign.accuracy_policy,
            resume_from_run_path=promotion.resume_from_run_path,
        )
    )

    apm = [
        action
        for action in campaign.next_actions(ledger)
        if action.key.drive == first.key.drive
        and action.key.controller == first.key.controller
        and action.key.method == "exchange"
    ]
    assert len(apm) == 1
    assert apm[0].horizon == "t20"
