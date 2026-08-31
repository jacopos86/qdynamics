"""CHTC packaging for any current staged-campaign frontier."""

from __future__ import annotations

import json

from pipelines.time_dynamics.campaigns import paper_ii_staged_accuracy_v1 as staged
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    CampaignLedger,
    RungRecord,
)
from pipelines.time_dynamics.staged_campaign_chtc import (
    prepare_staged_chtc_campaign,
)


REVISION = "a" * 40


def _prepare(tmp_path, campaign, ledger, seed, *, in_flight_action_ids=()):
    return prepare_staged_chtc_campaign(
        campaign_id=staged.CAMPAIGN_ID,
        campaign=campaign,
        ledger=ledger,
        package_dir=tmp_path / "package",
        seed_path=seed,
        numerics=staged.NUMERICS_ID,
        gate="mclachlan_l2_1e-3",
        search_profile="diagnostic_fast_v1",
        repo_root=tmp_path,
        code_revision=REVISION,
        in_flight_action_ids=in_flight_action_ids,
    )


def test_t5_package_starts_with_avqds_comparators_without_terminal_compile(
    tmp_path,
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text('{"seed": true}\n', encoding="utf-8")
    campaign = staged.build_campaign(prior_root=None)
    prepared = _prepare(tmp_path, campaign, CampaignLedger(), seed)
    rows = [
        json.loads(line)
        for line in prepared.actions_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 18
    assert {row["stage"] for row in rows} == {"screen_avqds_t5"}
    assert {row["action"]["key"]["method"] for row in rows} == {"avqds"}
    assert all(row["compile_terminal_cost"] is False for row in rows)
    assert all("--record-statevector" in row["runner_argv"] for row in rows)
    assert prepared.submit.is_file()
    assert "request_memory = 10GB" in prepared.submit.read_text(encoding="utf-8")
    wrapper = prepared.wrapper.read_text(encoding="utf-8")
    assert "mkdir -p raw_outputs" in wrapper
    assert wrapper.index("mkdir -p raw_outputs") < wrapper.index("apptainer")


def test_t5_package_excludes_in_flight_action_and_records_receipt(tmp_path) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text('{"seed": true}\n', encoding="utf-8")
    campaign = staged.build_campaign(prior_root=None)
    current = campaign.next_actions(CampaignLedger())
    in_flight = current[0].action_id

    prepared = _prepare(
        tmp_path,
        campaign,
        CampaignLedger(),
        seed,
        in_flight_action_ids=(in_flight,),
    )
    manifest = json.loads(prepared.manifest.read_text(encoding="utf-8"))
    emitted = {
        json.loads(line)["action_id"]
        for line in prepared.actions_jsonl.read_text(encoding="utf-8").splitlines()
    }

    assert in_flight not in emitted
    assert manifest["current_action_count"] == 18
    assert manifest["action_count"] == 17
    assert manifest["in_flight_action_ids"] == [in_flight]


def test_t20_package_copies_each_resume_source_and_preserves_global_grid(
    tmp_path,
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text('{"seed": true}\n', encoding="utf-8")
    campaign = staged.build_campaign(prior_root=None)
    ledger = CampaignLedger()
    metrics = AccuracyMetrics(5.0e-5, 4.0e-5, 3.0e-5, 5.0e-6)
    for action in campaign.next_actions(ledger):
        screen = tmp_path / "screens" / action.action_id / "run.json"
        screen.parent.mkdir(parents=True, exist_ok=True)
        screen.write_text('{"resume_state": {}}\n', encoding="utf-8")
        ledger = ledger.append(
            RungRecord.completed(
                key=action.key,
                horizon=action.horizon,
                threshold=action.threshold,
                run_path=str(screen),
                metrics=metrics,
                policy=campaign.accuracy_policy,
            )
        )
    prepared = _prepare(tmp_path, campaign, ledger, seed)
    rows = [
        json.loads(line)
        for line in prepared.actions_jsonl.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 18
    assert {row["stage"] for row in rows} == {"promote_avqds_t20"}
    assert all(row["compile_terminal_cost"] is True for row in rows)
    assert all(row["resume_receipt"]["sha256"] for row in rows)
    for row in rows:
        argv = row["runner_argv"]
        point_counts = [
            argv[index + 1]
            for index, value in enumerate(argv)
            if value == "--num-times"
        ]
        # The registered t20 horizon declares 501 points; the continuation
        # override is deliberately later in argv and therefore authoritative.
        assert point_counts[-1] == "376"
        packaged_resume = argv[argv.index("--resume-from-run-json") + 1]
        assert "input/resume" in packaged_resume


def test_package_can_exclude_in_flight_t5_actions_and_emit_ready_promotions(
    tmp_path,
) -> None:
    seed = tmp_path / "seed.json"
    seed.write_text('{"seed": true}\n', encoding="utf-8")
    campaign = staged.build_campaign(prior_root=None)
    ledger = CampaignLedger()
    screening = campaign.next_actions(ledger)
    qualified = screening[:2]
    metrics = AccuracyMetrics(5.0e-5, 4.0e-5, 3.0e-5, 5.0e-6)
    for action in qualified:
        screen = tmp_path / "screens" / action.action_id / "run.json"
        screen.parent.mkdir(parents=True, exist_ok=True)
        screen.write_text('{"resume_state": {}}\n', encoding="utf-8")
        ledger = ledger.append(
            RungRecord.completed(
                key=action.key,
                horizon=action.horizon,
                threshold=action.threshold,
                run_path=str(screen),
                metrics=metrics,
                policy=campaign.accuracy_policy,
            )
        )

    current = campaign.next_actions(ledger)
    in_flight = tuple(
        action.action_id for action in current if action.horizon == "t5"
    )
    prepared = _prepare(
        tmp_path,
        campaign,
        ledger,
        seed,
        in_flight_action_ids=in_flight,
    )
    rows = [
        json.loads(line)
        for line in prepared.actions_jsonl.read_text(encoding="utf-8").splitlines()
    ]

    assert len(rows) == 2
    assert all(row["action"]["horizon"] == "t20" for row in rows)
    assert all(row["compile_terminal_cost"] is True for row in rows)
