"""Focused tests for the verified static-QSE results contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.qse_spectra.static_campaign import (
    DEFAULT_TARGET_ROOTS,
    GroundStateArtifactRef,
    METHOD_DIRECT_DAVIDSON,
    METHOD_RECORD_FIRST,
    NotPlannedCell,
    STAGE_GROUND_STATE_IMPORT,
    STAGE_ORDER,
    STAGE_QUALIFICATION,
    STAGE_STATIC_QSE_TRACE,
    STATUS_CANDIDATE,
    STATUS_GATE_PENDING,
    STATUS_NOT_PLANNED,
    STATUS_PENDING,
    STATUS_QUALIFIED,
    StaticCampaignError,
    StaticQSEAttempt,
    StaticQSECampaign,
    StaticQSECellKey,
    StaticQSELedger,
    hh_physics_key,
)
from pipelines.reporting.paper_iii_static_results_contract import (
    STATIC_RESULTS_CONTRACT_SCHEMA_VERSION,
    StaticResultsContract,
    build_static_results_contract,
    load_static_results_contract,
    manuscript_rows_from_static_results_contract,
    validate_static_results_contract,
    write_static_results_contract,
)


def _artifact(slot: str) -> GroundStateArtifactRef:
    return GroundStateArtifactRef(
        provider_id=f"provider::{slot}",
        manifest_path=f"states/{slot}.json",
        manifest_sha256="ab" * 32,
        state_key="initial_state",
        state_payload_sha256="cd" * 32,
        num_qubits=8,
        sector="half_filled_11",
        energy=-1.25,
        source_revision="e6b2c2ca",
        achieved_settings_digest="ef" * 32,
    )


def _cell(method: str, regime: str = "weak_weak") -> StaticQSECellKey:
    return StaticQSECellKey(
        physics=hh_physics_key(regime, 0.25, 0.353553390593, 1),
        ground_state_artifact_id=_artifact(f"{regime}_nph1").artifact_id,
        method_id=method,
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest="pool" * 6,
        spectral_request_digest="spec" * 6,
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
    )


def _campaign() -> StaticQSECampaign:
    return StaticQSECampaign(
        campaign_id="contract_test",
        cells=(_cell(METHOD_RECORD_FIRST), _cell(METHOD_DIRECT_DAVIDSON)),
        not_planned=(
            NotPlannedCell(
                physics=hh_physics_key("weak_weak", 0.25, 0.353553390593, 1),
                method_id="fixed_linear_response_qse",
                reason="declared unplanned for this test",
            ),
        ),
    )


def _attempt(action_id: str, *, qualification=None) -> StaticQSEAttempt:
    return StaticQSEAttempt.from_mappings(
        action_id=action_id,
        achieved_config={"stage": "s"},
        source_revision="deadbeef",
        input_digests={},
        output_digests={"payload.json": "11" * 32},
        payload_paths=[f"{action_id[:8]}/payload.json"],
        disposition="completed",
        qualification=qualification,
    )


def test_every_declared_cell_and_slot_appears_exactly_once() -> None:
    campaign = _campaign()
    contract = build_static_results_contract(campaign, StaticQSELedger())
    validate_static_results_contract(contract)
    assert len(contract.rows) == 3
    kinds = [row["row_kind"] for row in contract.rows]
    assert kinds.count("planned_cell") == 2
    assert kinds.count("not_planned_slot") == 1
    planned = [r for r in contract.rows if r["row_kind"] == "planned_cell"]
    for row in planned:
        assert row["cell_status"] == STATUS_PENDING
        assert set(row["stage_statuses"]) == set(STAGE_ORDER)


def test_one_verified_action_is_visible_without_siblings() -> None:
    campaign = _campaign()
    cell = campaign.cells[0]
    actions = campaign.cell_actions(cell)
    ledger = StaticQSELedger()
    ledger.append(_attempt(actions[STAGE_GROUND_STATE_IMPORT].action_id))

    contract = build_static_results_contract(campaign, ledger)
    validate_static_results_contract(contract)
    row = next(
        r for r in contract.rows
        if r.get("row_kind") == "planned_cell" and r["cell_digest"] == cell.digest()
    )
    assert row["stage_statuses"][STAGE_GROUND_STATE_IMPORT] == STATUS_GATE_PENDING
    assert row["stage_statuses"][STAGE_STATIC_QSE_TRACE] == STATUS_PENDING
    assert row["cell_status"] == STATUS_GATE_PENDING
    other = next(
        r for r in contract.rows
        if r.get("row_kind") == "planned_cell" and r["cell_digest"] != cell.digest()
    )
    assert other["cell_status"] == STATUS_PENDING


def test_failed_gates_stay_candidate_with_payloads_visible() -> None:
    campaign = _campaign()
    cell = campaign.cells[0]
    actions = campaign.cell_actions(cell)
    ledger = StaticQSELedger()
    for stage in STAGE_ORDER:
        qualification = None
        if stage == STAGE_QUALIFICATION:
            qualification = {
                "status": STATUS_CANDIDATE,
                "failed_gates": ["sum_rule_deficit_gate"],
            }
        ledger.append(_attempt(actions[stage].action_id, qualification=qualification))

    contract = build_static_results_contract(campaign, ledger)
    validate_static_results_contract(contract)
    row = next(
        r for r in contract.rows
        if r.get("row_kind") == "planned_cell" and r["cell_digest"] == cell.digest()
    )
    assert row["cell_status"] == STATUS_CANDIDATE
    assert row["failed_gates"] == ["sum_rule_deficit_gate"]
    assert row["payload_paths"]  # numerical values stay referenced


def test_qualified_cell_rolls_up_qualified() -> None:
    campaign = _campaign()
    cell = campaign.cells[0]
    actions = campaign.cell_actions(cell)
    ledger = StaticQSELedger()
    for stage in STAGE_ORDER:
        qualification = (
            {"status": STATUS_QUALIFIED, "failed_gates": []}
            if stage == STAGE_QUALIFICATION
            else None
        )
        ledger.append(_attempt(actions[stage].action_id, qualification=qualification))
    contract = build_static_results_contract(campaign, ledger)
    row = next(
        r for r in contract.rows
        if r.get("row_kind") == "planned_cell" and r["cell_digest"] == cell.digest()
    )
    assert row["cell_status"] == STATUS_QUALIFIED


def test_contract_round_trips_and_pins_both_digests(tmp_path: Path) -> None:
    campaign = _campaign()
    ledger = StaticQSELedger()
    contract = build_static_results_contract(campaign, ledger)
    assert contract.campaign_digest == campaign.digest()
    assert contract.ledger_digest == ledger.digest()
    path = tmp_path / "contract.json"
    write_static_results_contract(contract, path)
    loaded = load_static_results_contract(path)
    assert loaded.digest() == contract.digest()


def test_validation_fails_on_tampered_rows(tmp_path: Path) -> None:
    campaign = _campaign()
    contract = build_static_results_contract(campaign, StaticQSELedger())
    tampered = StaticResultsContract(
        payload={**contract.payload, "schema_version": "bogus"}
    )
    with pytest.raises(StaticCampaignError, match="schema mismatch"):
        validate_static_results_contract(tampered)

    rows = [dict(r) for r in contract.payload["rows"]]
    rows[0]["cell_status"] = "victorious"
    with pytest.raises(StaticCampaignError, match="unknown cell status"):
        validate_static_results_contract(
            StaticResultsContract(payload={**contract.payload, "rows": rows})
        )


def test_manuscript_rows_come_only_from_the_verified_accessor() -> None:
    campaign = _campaign()
    contract = build_static_results_contract(campaign, StaticQSELedger())
    rows = manuscript_rows_from_static_results_contract(contract)
    assert len(rows) == 3
    statuses = {row.cell_status for row in rows}
    assert statuses == {STATUS_PENDING, STATUS_NOT_PLANNED}
    not_planned = [r for r in rows if r.cell_status == STATUS_NOT_PLANNED]
    assert not_planned[0].not_planned_reason == "declared unplanned for this test"

    broken = StaticResultsContract(
        payload={**contract.payload, "schema_version": "bogus"}
    )
    with pytest.raises(StaticCampaignError):
        manuscript_rows_from_static_results_contract(broken)
