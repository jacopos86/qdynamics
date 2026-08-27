"""Focused tests for static-QSE artifact verification and exact-subset ingestion."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipelines.qse_spectra.static_campaign import (
    DEFAULT_TARGET_ROOTS,
    GroundStateArtifactRef,
    METHOD_RECORD_FIRST,
    STAGE_GROUND_STATE_IMPORT,
    STATUS_CANDIDATE,
    StaticCampaignError,
    StaticQSEAction,
    StaticQSECampaign,
    StaticQSECellKey,
    StaticQSELedger,
    hh_physics_key,
)
from pipelines.qse_spectra.static_campaign_artifacts import (
    RESULT_ENVELOPE_FILENAME,
    StaticCampaignArtifactError,
    action_output_dir,
    build_result_envelope,
    discover_completed_action_ids,
    record_action_outputs,
    sha256_file,
    verify_achieved_action,
    write_result_envelope,
)


def _cell(method: str = METHOD_RECORD_FIRST) -> StaticQSECellKey:
    artifact = GroundStateArtifactRef(
        provider_id="provider::weak_weak_nph1",
        manifest_path="states/weak_weak_nph1.json",
        manifest_sha256="ab" * 32,
        state_key="initial_state",
        state_payload_sha256="cd" * 32,
        num_qubits=8,
        sector="half_filled_11",
        energy=-1.25,
        source_revision="e6b2c2ca",
        achieved_settings_digest="ef" * 32,
    )
    return StaticQSECellKey(
        physics=hh_physics_key("weak_weak", 0.25, 0.353553390593, 1),
        ground_state_artifact_id=artifact.artifact_id,
        method_id=method,
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest="pool" * 6,
        spectral_request_digest="spec" * 6,
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
    )


def _action() -> StaticQSEAction:
    return StaticQSEAction(
        cell=_cell(), stage=STAGE_GROUND_STATE_IMPORT, dependency_action_ids=()
    )


def _produce(action: StaticQSEAction, root: Path, *, method_label: str | None = None,
             qualification=None) -> Path:
    out_dir = action_output_dir(root, action.action_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = out_dir / "ground_state_import.json"
    payload.write_text(json.dumps({"norm": 1.0}), encoding="utf-8")
    envelope = build_result_envelope(
        action=action,
        achieved_settings={"state_key": "initial_state"},
        source_revision="deadbeef",
        input_digests={"manifest": "aa" * 32},
        payload_files={"ground_state_import.json": payload},
        disposition="completed",
        telemetry={"elapsed_s": 0.1},
    )
    if method_label is not None:
        envelope["achieved"]["method_id"] = method_label
    if qualification is not None:
        envelope["qualification"] = qualification
    write_result_envelope(envelope, out_dir)
    return out_dir


def test_round_trip_verification(tmp_path: Path) -> None:
    action = _action()
    _produce(action, tmp_path)
    attempt = verify_achieved_action(action, tmp_path)
    assert attempt.action_id == action.action_id
    assert attempt.disposition == "completed"
    assert dict(attempt.output_digests)["ground_state_import.json"]
    assert dict(attempt.achieved_config)["cell_digest"] == action.cell.digest()


def test_alias_method_in_achieved_config_normalizes(tmp_path: Path) -> None:
    action = _action()
    _produce(action, tmp_path, method_label="record_first_adaptive_qse_v2")
    attempt = verify_achieved_action(action, tmp_path)
    assert dict(attempt.achieved_config)["method_id"] == METHOD_RECORD_FIRST


def test_output_digest_drift_fails_closed(tmp_path: Path) -> None:
    action = _action()
    out_dir = _produce(action, tmp_path)
    (out_dir / "ground_state_import.json").write_text(
        json.dumps({"norm": 0.5}), encoding="utf-8"
    )
    with pytest.raises(StaticCampaignArtifactError, match="digest drift"):
        verify_achieved_action(action, tmp_path)


def test_missing_payload_file_fails_closed(tmp_path: Path) -> None:
    action = _action()
    out_dir = _produce(action, tmp_path)
    (out_dir / "ground_state_import.json").unlink()
    with pytest.raises(StaticCampaignArtifactError, match="missing"):
        verify_achieved_action(action, tmp_path)


def test_achieved_stage_mismatch_fails_closed(tmp_path: Path) -> None:
    action = _action()
    out_dir = _produce(action, tmp_path)
    envelope = json.loads((out_dir / RESULT_ENVELOPE_FILENAME).read_text())
    envelope["achieved"]["stage"] = "static_qse_trace"
    (out_dir / RESULT_ENVELOPE_FILENAME).write_text(json.dumps(envelope))
    with pytest.raises(StaticCampaignArtifactError, match="stage"):
        verify_achieved_action(action, tmp_path)


def test_missing_envelope_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(StaticCampaignArtifactError, match="missing result envelope"):
        verify_achieved_action(_action(), tmp_path)


def test_failed_gate_is_admitted_as_candidate_with_values(tmp_path: Path) -> None:
    action = _action()
    _produce(
        action,
        tmp_path,
        qualification={"status": STATUS_CANDIDATE, "gates": ["sum_rule_gate"]},
    )
    attempt = verify_achieved_action(action, tmp_path)
    assert attempt.qualification_status == STATUS_CANDIDATE
    assert dict(attempt.output_digests)  # numerical values remain referenced

    campaign = StaticQSECampaign(campaign_id="c", cells=(action.cell,))
    ledger = StaticQSELedger()
    ledger.append(attempt)
    statuses = campaign.cell_stage_statuses(ledger, action.cell)
    assert statuses[STAGE_GROUND_STATE_IMPORT] == STATUS_CANDIDATE


def test_record_action_outputs_admits_exact_subsets(tmp_path: Path) -> None:
    action = _action()
    _produce(action, tmp_path)
    package = SimpleNamespace(
        actions_by_id={action.action_id: action}, artifact_root=tmp_path
    )
    ledger = StaticQSELedger()
    summary = record_action_outputs(package, [action.action_id], ledger)
    assert summary.admitted == (action.action_id,)

    replay = record_action_outputs(package, [action.action_id], ledger)
    assert replay.admitted == ()
    assert replay.idempotent == (action.action_id,)


def test_record_action_outputs_rejects_unknown_and_empty(tmp_path: Path) -> None:
    action = _action()
    package = SimpleNamespace(
        actions_by_id={action.action_id: action}, artifact_root=tmp_path
    )
    ledger = StaticQSELedger()
    with pytest.raises(StaticCampaignArtifactError, match="nonempty"):
        record_action_outputs(package, [], ledger)
    with pytest.raises(StaticCampaignArtifactError, match="unknown action ids"):
        record_action_outputs(package, ["f" * 64], ledger)


def test_conflicting_reingestion_fails(tmp_path: Path) -> None:
    action = _action()
    out_dir = _produce(action, tmp_path)
    package = SimpleNamespace(
        actions_by_id={action.action_id: action}, artifact_root=tmp_path
    )
    ledger = StaticQSELedger()
    record_action_outputs(package, [action.action_id], ledger)

    # Regenerate the artifact with different numerical content and a matching
    # (self-consistent) envelope: same action id, different output digests.
    payload = out_dir / "ground_state_import.json"
    payload.write_text(json.dumps({"norm": 0.999}), encoding="utf-8")
    envelope = build_result_envelope(
        action=action,
        achieved_settings={"state_key": "initial_state"},
        source_revision="deadbeef",
        input_digests={"manifest": "aa" * 32},
        payload_files={"ground_state_import.json": payload},
        disposition="completed",
    )
    write_result_envelope(envelope, out_dir)
    with pytest.raises(StaticCampaignError, match="conflicting replay"):
        record_action_outputs(package, [action.action_id], ledger)


def test_discovery_lists_only_enveloped_directories(tmp_path: Path) -> None:
    action = _action()
    _produce(action, tmp_path)
    (tmp_path / "not_an_action").mkdir()
    assert discover_completed_action_ids(tmp_path) == (action.action_id,)
    assert discover_completed_action_ids(tmp_path / "absent") == ()


def test_sha256_file_matches_python_hashlib(tmp_path: Path) -> None:
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"paper-iii")
    assert sha256_file(path) == hashlib.sha256(b"paper-iii").hexdigest()
