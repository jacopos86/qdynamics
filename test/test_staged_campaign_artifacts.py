"""Run-receipt checks at the staged-campaign artifact boundary."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics import paper_ii_runs as runs
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellKey,
    StagedCampaign,
    ThresholdSchedule,
)
from pipelines.time_dynamics.staged_campaign_artifacts import (
    record_completed_action,
    record_prepared_action_outputs,
    record_run_outcome,
    verify_run_receipt,
)
from pipelines.time_dynamics.campaigns.paper_ii_staged_accuracy_v1 import (
    _selected_action_rows,
)


def test_record_package_selects_exact_increment_action_ids(tmp_path: Path) -> None:
    actions = tmp_path / "actions.jsonl"
    actions.write_text(
        "".join(
            json.dumps({"action_id": action_id}) + "\n"
            for action_id in ("a", "b", "c")
        ),
        encoding="utf-8",
    )

    selected = _selected_action_rows(actions, ("c", "a"))

    assert [row["action_id"] for row in selected] == ["c", "a"]
    with pytest.raises(ValueError, match="absent"):
        _selected_action_rows(actions, ("missing",))


KEY = CellKey("hh_snake_nph1", "fastweak", "avqds", "delta_theta_5e-5")
CAMPAIGN = StagedCampaign(
    cells=(KEY,),
    accuracy_policy=AccuracyPolicy(),
    threshold_schedule=ThresholdSchedule({KEY: (1.0e-3,)}),
)


def _command(action):
    return CAMPAIGN.build_run_command(
        action,
        seed_path="seed.json",
        output_json="out/run.json",
        numerics=runs.EULER_RIDGE1E6_NUMERICS.numerics_id,
        gate=runs.MCLACHLAN_L2_GATE.gate_id,
        search_profile=runs.DIAGNOSTIC_FAST_SEARCH.profile_id,
    )


def _run_payload(action, command):
    control = command.step_control
    profile = command.search_profile
    plot_rows = [
        {
            "time": 5.0 * index / 125,
            "psi_real": [1.0, 0.0],
            "psi_imag": [0.0, 0.0],
        }
        for index in range(126)
    ]
    return {
        "trajectory": {
            "integrator_method": command.numerics.integrator,
            "support_patch_config": {
                "dynamics_policy": "avqds",
                "avqds_l2_cut": action.threshold,
                "max_structural_pool_size": profile.pool_size,
                "max_joint_patch_evaluations": profile.joint_patch_evaluations,
                "max_certification_attempts_per_level": (
                    profile.certification_attempts_per_level
                ),
                "max_certification_attempts_per_deletion_branch": (
                    profile.certification_attempts_per_deletion_branch
                ),
                "max_insertion_rounds_per_checkpoint": profile.rounds_per_checkpoint,
                "max_insertion_batch_size": profile.insertion_cardinality,
            },
        },
        "run_lock": {
            "physics": {
                "drive_profile": {
                    "A": command.drive.amplitude,
                    "omega": command.drive.omega,
                },
                "time_grid": {
                    "t_initial": 0.0,
                    "t_final": 5.0,
                    "point_count": 126,
                },
            },
            "numerics": {
                "inverse_policy": {
                    "pinv_rcond": command.numerics.pinv_rcond,
                    "ridge_lambda": command.numerics.ridge_lambda,
                    "solve_damping": command.numerics.solve_damping,
                },
                "solve_repair": {
                    "enabled": control.solve_repair_enabled,
                    "ridge_ladder": [command.numerics.ridge_lambda],
                    "pinv_rcond_ladder": [command.numerics.pinv_rcond],
                    "solve_damping_ladder": [command.numerics.solve_damping],
                },
            },
            "shared_protocol": {
                "time_step_controller": {
                    "parameter_displacement_max": (
                        None
                        if control.solve_repair_enabled
                        else control.parameter_step_max
                    ),
                    "composed_parameter_step_max": (
                        control.parameter_step_max
                        if control.solve_repair_enabled
                        else None
                    ),
                    "state_motion_l2_step_max": control.state_motion_l2_step_max,
                    "max_local_subdivisions": control.max_local_subdivisions,
                }
            },
        },
        "plot_rows": plot_rows,
        "resume_state": {"controller_state": {"schema": "controller"}},
        "summary": {"uncured_local_subdivision_count": 0},
    }


def _resumed_payload(tmp_path):
    screen_action = CAMPAIGN.next_actions(CampaignLedger())[0]
    source = tmp_path / "screen" / "run.json"
    source.parent.mkdir(parents=True, exist_ok=True)
    labels = ("saved::r0::xe", "saved::r1::ye")
    source.write_text(
        json.dumps(
            {
                "resume_state": {
                    "coordinates": [{"label": label} for label in labels],
                    "controller_state": {"schema": "controller"},
                }
            }
        ),
        encoding="utf-8",
    )
    action = replace(
        screen_action,
        stage="promote_avqds_t20",
        horizon="t20",
        resume_from_run_path=str(source),
    )
    command = _command(action)
    payload = _run_payload(action, command)
    payload["plot_rows"] = [
        {
            "time": 5.0 + 15.0 * index / 375,
            "psi_real": [1.0, 0.0],
            "psi_imag": [0.0, 0.0],
        }
        for index in range(376)
    ]
    payload["run_lock"]["physics"]["time_grid"] = {
        "t_initial": 5.0,
        "t_final": 20.0,
        "point_count": 376,
    }
    payload["initial_state"] = {
        "runtime_coordinate_labels": list(labels),
        "selected_term_labels": list(labels),
        "runtime_parameter_count": len(labels),
        "logical_parameter_count": len(labels),
    }
    payload["drive_aligned_ansatz"] = {
        "applied": False,
        "reason": "resume_terminal_support_already_contains_drive_alignment",
    }
    return action, command, payload


def test_receipt_accepts_the_requested_live_configuration() -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    achieved = verify_run_receipt(
        action=action,
        command=command,
        run=_run_payload(action, command),
    )
    assert achieved["method"] == "avqds"
    assert achieved["activation_cut"] == pytest.approx(action.threshold)


def test_receipt_requires_exact_saved_support_at_resume_boundary(tmp_path) -> None:
    action, command, payload = _resumed_payload(tmp_path)

    achieved = verify_run_receipt(action=action, command=command, run=payload)

    assert achieved["resume_support_boundary_verified"] is True
    assert achieved["resume_runtime_coordinate_count"] == 2


def test_receipt_rejects_drive_reaugmentation_even_if_run_completed(tmp_path) -> None:
    action, command, payload = _resumed_payload(tmp_path)
    payload["initial_state"]["runtime_coordinate_labels"].append("drive::r0::xe")
    payload["initial_state"]["selected_term_labels"].append("drive::r0::xe")
    payload["initial_state"]["runtime_parameter_count"] = 3
    payload["initial_state"]["logical_parameter_count"] = 3
    payload["drive_aligned_ansatz"] = {
        "applied": True,
        "reason": "applied",
    }

    with pytest.raises(ValueError, match="preserve its saved support boundary"):
        verify_run_receipt(action=action, command=command, run=payload)


def test_receipt_ignores_serialized_defaults_for_inactive_state_controller() -> None:
    """The runner records raw solve-repair defaults even when it is disabled."""

    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    payload = _run_payload(action, command)
    controller = payload["run_lock"]["shared_protocol"]["time_step_controller"]
    controller["state_motion_l2_step_max"] = 1.0e-2
    controller["max_local_subdivisions"] = 10

    achieved = verify_run_receipt(
        action=action,
        command=command,
        run=payload,
    )

    assert achieved["solve_repair_enabled"] is False
    assert achieved["state_motion_l2_step_max"] is None
    assert achieved["max_local_subdivisions"] is None


def test_receipt_rejects_wrong_solve_repair_activation() -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    payload = _run_payload(action, command)
    payload["run_lock"]["numerics"]["solve_repair"]["enabled"] = True

    with pytest.raises(ValueError, match="solve_repair_enabled"):
        verify_run_receipt(action=action, command=command, run=payload)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("ridge_ladder", [1.0e-7], "repair_ridge_ladder"),
        ("pinv_rcond_ladder", [1.0e-11], "repair_pinv_rcond_ladder"),
        ("solve_damping_ladder", [1.0e-8], "repair_solve_damping_ladder"),
    ),
)
def test_receipt_rejects_noncanonical_inverse_repair_ladder(
    field, value, message
) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    payload = _run_payload(action, command)
    payload["run_lock"]["numerics"]["solve_repair"][field] = value

    with pytest.raises(ValueError, match=message):
        verify_run_receipt(action=action, command=command, run=payload)


def test_receipt_accepts_composed_parameter_cap_from_solve_repair() -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = replace(
        _command(action),
        step_control=runs.COMPOSED_CONTROL_CALIBRATED,
    )

    achieved = verify_run_receipt(
        action=action,
        command=command,
        run=_run_payload(action, command),
    )

    assert achieved["solve_repair_enabled"] is True
    assert achieved["parameter_displacement_max"] == pytest.approx(5.0e-5)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("dynamics_policy", "exchange", "method"),
        ("avqds_l2_cut", 9.0e-2, "activation_cut"),
    ),
)
def test_receipt_rejects_silent_arm_or_cut_drift(field, value, message) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    payload = _run_payload(action, command)
    payload["trajectory"]["support_patch_config"][field] = value
    with pytest.raises(ValueError, match=message):
        verify_run_receipt(action=action, command=command, run=payload)


def test_record_completed_action_appends_metrics_and_provenance(
    tmp_path, monkeypatch
) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_run_payload(action, command)), encoding="utf-8")
    expected_metrics = AccuracyMetrics(5.0e-5, 4.0e-5, 3.0e-5, 2.0e-5)
    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign_artifacts.physical_errors_from_artifacts",
        lambda **_kwargs: expected_metrics,
    )
    ledger = record_completed_action(
        campaign=CAMPAIGN,
        ledger=CampaignLedger(),
        action=action,
        command=command,
        run_path=path,
        seed_path="seed.json",
    )
    assert ledger.records[0].metrics == expected_metrics
    assert ledger.records[0].status == "accuracy_qualified"
    assert ledger.records[0].achieved_configuration["cut_field"] == "avqds_l2_cut"


def test_run_outcome_records_uncured_subdivision_as_safeguard(
    tmp_path, monkeypatch
) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    payload = _run_payload(action, command)
    payload["summary"]["uncured_local_subdivision_count"] = 2
    path = tmp_path / "run.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign_artifacts.physical_errors_from_artifacts",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("unsafe trajectory must not be scored")
        ),
    )

    ledger = record_run_outcome(
        campaign=CAMPAIGN,
        ledger=CampaignLedger(),
        action=action,
        command=command,
        run_path=path,
        seed_path="seed.json",
    )

    record = ledger.records[0]
    assert record.status == "safeguard_tripped"
    assert record.run_path == str(path)
    assert "2 uncured" in record.reason
    assert record.metrics is None


def test_empty_run_artifact_fails_loudly(tmp_path, monkeypatch) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    path = tmp_path / "run.json"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="missing or empty"):
        record_completed_action(
            campaign=CAMPAIGN,
            ledger=CampaignLedger(),
            action=action,
            command=_command(action),
            run_path=path,
            seed_path="seed.json",
        )


def test_prepared_package_ingestion_is_validated_and_idempotent(
    tmp_path, monkeypatch
) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    command = _command(action)
    relative = f"raw_outputs/campaign/{action.action_id}/run.json"
    root = tmp_path / "verified_extract"
    run_path = root / relative
    run_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(_run_payload(action, command)), encoding="utf-8")
    metrics = AccuracyMetrics(5.0e-5, 4.0e-5, 3.0e-5, 2.0e-5)
    monkeypatch.setattr(
        "pipelines.time_dynamics.staged_campaign_artifacts.physical_errors_from_artifacts",
        lambda **_kwargs: metrics,
    )
    row = {
        "action_id": action.action_id,
        "action": action.to_json_dict(),
        "trajectory_json": relative,
    }

    ledger, receipts = record_prepared_action_outputs(
        campaign=CAMPAIGN,
        ledger=CampaignLedger(),
        action_rows=(row,),
        artifact_root=root,
        seed_path="seed.json",
        numerics=runs.EULER_RIDGE1E6_NUMERICS.numerics_id,
        gate=runs.MCLACHLAN_L2_GATE.gate_id,
        search_profile=runs.DIAGNOSTIC_FAST_SEARCH.profile_id,
    )
    replayed, replay_receipts = record_prepared_action_outputs(
        campaign=CAMPAIGN,
        ledger=ledger,
        action_rows=(row,),
        artifact_root=root,
        seed_path="seed.json",
        numerics=runs.EULER_RIDGE1E6_NUMERICS.numerics_id,
        gate=runs.MCLACHLAN_L2_GATE.gate_id,
        search_profile=runs.DIAGNOSTIC_FAST_SEARCH.profile_id,
    )

    assert len(ledger.records) == 1
    assert replayed == ledger
    assert receipts[0]["status"] == "recorded"
    assert replay_receipts[0]["status"] == "already_recorded"


def test_prepared_package_ingestion_rejects_path_escape(tmp_path) -> None:
    action = CAMPAIGN.next_actions(CampaignLedger())[0]
    root = tmp_path / "verified_extract"
    root.mkdir()
    row = {
        "action_id": action.action_id,
        "action": action.to_json_dict(),
        "trajectory_json": "../another_attempt/run.json",
    }

    with pytest.raises(ValueError, match="stay under the extraction root"):
        record_prepared_action_outputs(
            campaign=CAMPAIGN,
            ledger=CampaignLedger(),
            action_rows=(row,),
            artifact_root=root,
            seed_path="seed.json",
            numerics=runs.EULER_RIDGE1E6_NUMERICS.numerics_id,
            gate=runs.MCLACHLAN_L2_GATE.gate_id,
            search_profile=runs.DIAGNOSTIC_FAST_SEARCH.profile_id,
        )
