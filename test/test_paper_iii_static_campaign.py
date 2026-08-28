"""Focused tests for the typed Paper III static-QSE campaign and ledger."""

from __future__ import annotations

import json

import pytest

from pipelines.qse_spectra.static_campaign import (
    CONSTRUCTION_DIRECT_DAVIDSON,
    CONSTRUCTION_SELECTED_QSE_FIXED_POOL,
    DEFAULT_TARGET_ROOTS,
    GroundStateArtifactRef,
    HH_REGIMES,
    METHOD_DIRECT_DAVIDSON,
    METHOD_FIXED_CLASS,
    METHOD_KRYLOV,
    METHOD_RECORD_FIRST,
    PUBLIC_METHODS,
    STAGE_GREEN_FUNCTION,
    STAGE_GROUND_STATE_IMPORT,
    STAGE_ORDER,
    STAGE_QUALIFICATION,
    STAGE_SPECTRAL_POSTPROCESSING,
    STAGE_STATIC_QSE_TRACE,
    STATUS_CANDIDATE,
    STATUS_GATE_PENDING,
    STATUS_PENDING,
    STATUS_QUALIFIED,
    StaticCampaignError,
    StaticQSEAttempt,
    StaticQSECampaign,
    action_from_payload,
    StaticQSECellKey,
    StaticQSELedger,
    cell_key_from_payload,
    declare_hh_matrix,
    declare_siam_matrix,
    hh_physics_key,
    normalize_method_id,
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


def _cell(method: str = METHOD_RECORD_FIRST, regime: str = "weak_weak") -> StaticQSECellKey:
    u, g_ep = next((u, g) for name, u, g in HH_REGIMES if name == regime)
    return StaticQSECellKey(
        physics=hh_physics_key(regime, u, g_ep, 1),
        ground_state_artifact_id=_artifact(f"{regime}_nph1").artifact_id,
        method_id=method,
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest="pool" * 6,
        spectral_request_digest="spec" * 6,
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
    )


def _completed_attempt(action_id: str, *, qualification=None, output_sha="11" * 32):
    return StaticQSEAttempt.from_mappings(
        action_id=action_id,
        achieved_config={"cell_digest": "x", "stage": "y"},
        source_revision="deadbeef",
        input_digests={"ground_state": "aa" * 32},
        output_digests={"trace.json": output_sha},
        payload_paths=["trace.json"],
        disposition="completed",
        qualification=qualification,
    )


# --- identities -------------------------------------------------------------


def test_canonical_ids_are_stable_across_reconstruction() -> None:
    first = _cell()
    second = _cell()
    assert first.digest() == second.digest()
    campaign_a = StaticQSECampaign(campaign_id="c", cells=(first,))
    campaign_b = StaticQSECampaign(campaign_id="c", cells=(second,))
    ids_a = [a.action_id for a in campaign_a.all_actions()]
    ids_b = [a.action_id for a in campaign_b.all_actions()]
    assert ids_a == ids_b


def test_action_ids_differ_by_stage_and_cell() -> None:
    campaign = StaticQSECampaign(
        campaign_id="c",
        cells=(_cell(METHOD_RECORD_FIRST), _cell(METHOD_DIRECT_DAVIDSON)),
    )
    ids = [a.action_id for a in campaign.all_actions()]
    assert len(set(ids)) == len(ids) == 2 * len(STAGE_ORDER)


def test_old_hybrid_id_cannot_normalize_to_fixed_pool_selected_qse() -> None:
    with pytest.raises(StaticCampaignError, match="quarantined"):
        normalize_method_id("record_first_adaptive_qse_v2")


def test_construction_policy_is_explicit_identity_and_method_locked() -> None:
    selected = _cell(METHOD_RECORD_FIRST)
    direct = _cell(METHOD_DIRECT_DAVIDSON)

    assert selected.construction_policy_id == CONSTRUCTION_SELECTED_QSE_FIXED_POOL
    assert direct.construction_policy_id == CONSTRUCTION_DIRECT_DAVIDSON
    assert selected.digest() != direct.digest()
    tampered = direct.payload()
    tampered["construction_policy_id"] = CONSTRUCTION_SELECTED_QSE_FIXED_POOL
    with pytest.raises(StaticCampaignError, match="construction_policy_id"):
        cell_key_from_payload(tampered)


def test_unknown_method_fails_closed() -> None:
    with pytest.raises(StaticCampaignError, match="unknown method"):
        normalize_method_id("record_first_adaptive_qse_v3")


def test_v1_cell_action_and_malformed_method_policy_fail_closed() -> None:
    cell_payload = _cell().payload()
    cell_payload["schema"] = "static_qse_cell_key_v1"
    with pytest.raises(StaticCampaignError, match="cell-key schema"):
        cell_key_from_payload(cell_payload)

    action_payload = StaticQSECampaign(
        campaign_id="c", cells=(_cell(),)
    ).all_actions()[0].payload()
    action_payload["schema"] = "static_qse_action_v1"
    with pytest.raises(StaticCampaignError, match="action schema"):
        action_from_payload(action_payload)

    missing_policy = _cell().payload()
    missing_policy["method_policy"] = {}
    with pytest.raises((StaticCampaignError, ValueError), match="switch-policy schema"):
        cell_key_from_payload(missing_policy)


def test_ledger_v2_rejects_nested_v1_attempt(tmp_path) -> None:
    action = StaticQSECampaign(campaign_id="c", cells=(_cell(),)).all_actions()[0]
    attempt = _completed_attempt(action.action_id).payload()
    attempt["schema"] = "paper_iii_static_qse_attempt_v1"
    path = tmp_path / "ledger.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "paper_iii_static_qse_ledger_v2",
                "attempt_count": 1,
                "attempts": [attempt],
            }
        )
    )
    with pytest.raises(StaticCampaignError, match="attempt.*schema"):
        StaticQSELedger.load(path)


def test_accuracy_targets_do_not_multiply_actions() -> None:
    # Post-run eps_E ladders live outside the identity: the cell payload has
    # no accuracy-target field, so one trace serves every target.
    payload = _cell().payload()
    rendered = str(sorted(payload)).lower()
    assert "eps" not in rendered
    assert "accuracy" not in rendered
    campaign = StaticQSECampaign(campaign_id="c", cells=(_cell(),))
    assert len(campaign.all_actions()) == len(STAGE_ORDER)


def test_planner_order_is_independent_of_declaration_order() -> None:
    cells = (_cell(METHOD_RECORD_FIRST), _cell(METHOD_KRYLOV))
    forward = StaticQSECampaign(campaign_id="c", cells=cells)
    backward = StaticQSECampaign(campaign_id="c", cells=tuple(reversed(cells)))
    assert [a.action_id for a in forward.all_actions()] == [
        a.action_id for a in backward.all_actions()
    ]


# --- planning ---------------------------------------------------------------


def test_only_ready_dependencies_are_planned() -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    ledger = StaticQSELedger()
    ready = campaign.next_actions(ledger)
    assert [a.stage for a in ready] == [STAGE_GROUND_STATE_IMPORT]

    actions = campaign.cell_actions(cell)
    ledger.append(_completed_attempt(actions[STAGE_GROUND_STATE_IMPORT].action_id))
    ready = campaign.next_actions(ledger)
    assert [a.stage for a in ready] == [STAGE_STATIC_QSE_TRACE]

    ledger.append(_completed_attempt(actions[STAGE_STATIC_QSE_TRACE].action_id))
    ready_stages = {a.stage for a in campaign.next_actions(ledger)}
    assert ready_stages == {"transitions", "green_function", "resource_evaluation"}
    assert STAGE_SPECTRAL_POSTPROCESSING not in ready_stages
    assert STAGE_QUALIFICATION not in ready_stages


def test_candidate_dependency_still_unblocks_downstream() -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    actions = campaign.cell_actions(cell)
    ledger = StaticQSELedger()
    ledger.append(
        _completed_attempt(
            actions[STAGE_GROUND_STATE_IMPORT].action_id,
            qualification={"status": STATUS_CANDIDATE, "gates": ["norm_gate"]},
        )
    )
    assert [a.stage for a in campaign.next_actions(ledger)] == [STAGE_STATIC_QSE_TRACE]


def test_failed_process_does_not_satisfy_dependencies() -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    actions = campaign.cell_actions(cell)
    ledger = StaticQSELedger()
    ledger.append(
        StaticQSEAttempt.from_mappings(
            action_id=actions[STAGE_GROUND_STATE_IMPORT].action_id,
            achieved_config={},
            source_revision="deadbeef",
            input_digests={},
            output_digests={},
            payload_paths=[],
            disposition="failed_process",
        )
    )
    # The failed stage itself is still the ready action; nothing downstream is.
    assert [a.stage for a in campaign.next_actions(ledger)] == [STAGE_GROUND_STATE_IMPORT]


# --- ledger semantics -------------------------------------------------------


def test_exact_replay_is_an_idempotent_noop() -> None:
    ledger = StaticQSELedger()
    attempt = _completed_attempt("a" * 64)
    first = ledger.append(attempt)
    second = ledger.append(attempt)
    assert first.admitted and not first.idempotent
    assert (not second.admitted) and second.idempotent
    assert len(ledger) == 1


def test_conflicting_replay_fails() -> None:
    ledger = StaticQSELedger()
    ledger.append(_completed_attempt("a" * 64, output_sha="11" * 32))
    with pytest.raises(StaticCampaignError, match="conflicting replay"):
        ledger.append(_completed_attempt("a" * 64, output_sha="22" * 32))


def test_requalification_appends_without_conflict() -> None:
    ledger = StaticQSELedger()
    ledger.append(_completed_attempt("a" * 64))
    result = ledger.append(
        _completed_attempt(
            "a" * 64, qualification={"status": STATUS_QUALIFIED, "gates": []}
        )
    )
    assert result.admitted
    assert ledger.completed("a" * 64).qualification_status == STATUS_QUALIFIED


def test_ledger_round_trips_through_disk(tmp_path) -> None:
    ledger = StaticQSELedger()
    ledger.append(
        _completed_attempt("a" * 64, qualification={"status": STATUS_CANDIDATE})
    )
    path = tmp_path / "ledger.json"
    ledger.save(path)
    loaded = StaticQSELedger.load(path)
    assert loaded.digest() == ledger.digest()


# --- statuses ---------------------------------------------------------------


def test_absent_cells_remain_pending_and_gates_stay_candidate() -> None:
    cell = _cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    actions = campaign.cell_actions(cell)
    ledger = StaticQSELedger()
    statuses = campaign.cell_stage_statuses(ledger, cell)
    assert set(statuses.values()) == {STATUS_PENDING}

    ledger.append(_completed_attempt(actions[STAGE_GROUND_STATE_IMPORT].action_id))
    ledger.append(
        _completed_attempt(
            actions[STAGE_STATIC_QSE_TRACE].action_id,
            qualification={
                "status": STATUS_CANDIDATE,
                "gates": ["window_error_gate"],
            },
        )
    )
    statuses = campaign.cell_stage_statuses(ledger, cell)
    assert statuses[STAGE_GROUND_STATE_IMPORT] == STATUS_GATE_PENDING
    assert statuses[STAGE_STATIC_QSE_TRACE] == STATUS_CANDIDATE
    assert statuses[STAGE_GREEN_FUNCTION] == STATUS_PENDING


# --- declared matrices ------------------------------------------------------


def _hh_declarations():
    artifacts = {}
    pools = {}
    for n in (1, 3):
        for regime, _u, _g in HH_REGIMES:
            slot = f"{regime}_nph{n}"
            artifacts[slot] = _artifact(slot)
            pools[slot] = f"digest_{slot}"
    return artifacts, pools


def test_hh_matrix_is_the_declared_six_by_two_by_four() -> None:
    artifacts, pools = _hh_declarations()
    cells = declare_hh_matrix(
        ground_state_artifacts=artifacts,
        pool_policy_digests=pools,
        spectral_request_digest="spec" * 6,
    )
    assert len(cells) == 6 * 2 * 4
    methods = {cell.method_id for cell in cells}
    assert methods == set(PUBLIC_METHODS)
    caps = {cell.method_id: cell.dimension_cap for cell in cells}
    assert caps[METHOD_RECORD_FIRST] == 64
    assert caps[METHOD_KRYLOV] == 40


def test_hh_matrix_missing_artifact_is_a_declaration_error() -> None:
    artifacts, pools = _hh_declarations()
    del artifacts["weak_weak_nph1"]
    with pytest.raises(StaticCampaignError, match="missing ground-state artifact"):
        declare_hh_matrix(
            ground_state_artifacts=artifacts,
            pool_policy_digests=pools,
            spectral_request_digest="spec" * 6,
        )


def test_siam_fixed_class_is_not_planned_not_pending() -> None:
    artifacts = {
        "siam_nb2_fixture": _artifact("siam_nb2_fixture"),
        "siam_nb4_pilot": _artifact("siam_nb4_pilot"),
    }
    pools = {k: f"digest_{k}" for k in artifacts}
    cells, not_planned = declare_siam_matrix(
        ground_state_artifacts=artifacts,
        pool_policy_digests=pools,
        spectral_request_digest="spec" * 6,
    )
    planned_methods = {cell.method_id for cell in cells}
    assert planned_methods == {METHOD_RECORD_FIRST, METHOD_DIRECT_DAVIDSON}
    unplanned = {(slot.physics.case_id, slot.method_id) for slot in not_planned}
    assert ("siam_nb2_fixture", METHOD_FIXED_CLASS) in unplanned
    assert ("siam_nb4_pilot", METHOD_FIXED_CLASS) in unplanned
    # Krylov excluded by default until the generic adapter is faithful.
    assert ("siam_nb2_fixture", METHOD_KRYLOV) in unplanned

    campaign = StaticQSECampaign(
        campaign_id="siam", cells=cells, not_planned=not_planned
    )
    assert campaign.digest() == StaticQSECampaign(
        campaign_id="siam", cells=cells, not_planned=not_planned
    ).digest()
