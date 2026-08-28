"""Focused tests for the one-action static-QSE producer (SIAM N_b=2 fixture)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.exact_bench.paper_iii_static_qse_cell import (
    StaticQSECellError,
    run_action,
)
from pipelines.qse_spectra.siam_static_adapter import (
    build_siam_pool_elements,
    exact_sector_ground_fixture,
    ordered_pool_digest,
    siam_physics_key,
    state_payload_sha256,
)
from pipelines.qse_spectra.static_campaign import (
    DEFAULT_TARGET_ROOTS,
    GroundStateArtifactRef,
    METHOD_RECORD_FIRST,
    STAGE_GREEN_FUNCTION,
    STAGE_GROUND_STATE_IMPORT,
    STAGE_ORDER,
    STAGE_QUALIFICATION,
    STAGE_RESOURCE_EVALUATION,
    STAGE_SPECTRAL_POSTPROCESSING,
    STAGE_STATIC_QSE_TRACE,
    STAGE_TRANSITIONS,
    StaticQSECampaign,
    StaticQSECellKey,
    StaticQSELedger,
    canonical_digest,
    siam_physics_key_typed,
)
from pipelines.qse_spectra.static_campaign_artifacts import (
    record_action_outputs,
    verify_achieved_action,
)
from pipelines.qse_spectra.static_campaign_package import build_local_package

_SPECTRAL_REQUEST = {
    "transition_observables": [
        {"name": "impurity_occupation", "builder": "siam_impurity_occupation"},
        {"name": "impurity_double_occupancy", "builder": "siam_impurity_double_occupancy"},
    ],
    "green_function_modes": ["d_up=0", "d_down=3"],
    "grid": {"omega_min": -6.0, "omega_max": 6.0, "num_points": 121},
    "kernel": {"kernel": "lorentzian", "eta": 0.1},
    "windows": [],
}


def _diagnostic_artifact_ref(case_id: str) -> GroundStateArtifactRef:
    key = siam_physics_key(case_id)
    ground, provenance = exact_sector_ground_fixture(key)
    return GroundStateArtifactRef(
        provider_id="exact_sector_diagnostic_fixture",
        manifest_path="(diagnostic-fixture: no manifest)",
        manifest_sha256="0" * 64,
        state_key="exact_sector_ground",
        state_payload_sha256=state_payload_sha256(ground),
        num_qubits=key.n_qubits,
        sector=f"n_up_{key.n_up}_n_down_{key.n_down}",
        energy=float(provenance["ground_energy"]),
        source_revision="in_process",
        achieved_settings_digest=canonical_digest({"fixture": "exact_sector"}),
    )


def _siam_cell(case_id: str = "siam_nb2_fixture") -> StaticQSECellKey:
    key = siam_physics_key(case_id)
    return StaticQSECellKey(
        physics=siam_physics_key_typed(case_id),
        ground_state_artifact_id=_diagnostic_artifact_ref(case_id).artifact_id,
        method_id=METHOD_RECORD_FIRST,
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest=ordered_pool_digest(build_siam_pool_elements(key)),
        spectral_request_digest=canonical_digest(_SPECTRAL_REQUEST),
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
    )


def _context() -> dict:
    return {
        "problem": {"kind": "siam", "case_id": "siam_nb2_fixture"},
        "ground_state": {"kind": "exact_diagnostic"},
        "trace": {"stride": 2},
        "spectral_request": dict(_SPECTRAL_REQUEST),
        "qualification_policy": {},
        "resources": {"include_qiskit_compile": False},
    }


@pytest.fixture(scope="module")
def executed_chain(tmp_path_factory) -> dict:
    """Package and execute the full seven-stage chain for the N_b=2 fixture."""

    root = tmp_path_factory.mktemp("siam_nb2_chain")
    cell = _siam_cell()
    campaign = StaticQSECampaign(campaign_id="siam_nb2_smoke", cells=(cell,))
    package = build_local_package(
        campaign,
        campaign.all_actions(),
        package_root=root,
        cell_contexts={cell.digest(): _context()},
    )
    ledger = StaticQSELedger()
    envelopes: dict[str, dict] = {}
    executed_order: list[str] = []
    while True:
        ready = campaign.next_actions(ledger)
        if not ready:
            break
        for action in ready:
            action_dir = package.action_dir(action)
            envelope = run_action(action_dir / "ACTION.json", action_dir)
            envelopes[action.stage] = envelope
            executed_order.append(action.stage)
            record_action_outputs(package, [action.action_id], ledger)
    return {
        "root": root,
        "cell": cell,
        "campaign": campaign,
        "package": package,
        "ledger": ledger,
        "envelopes": envelopes,
        "executed_order": executed_order,
    }


def test_chain_completes_every_stage(executed_chain: dict) -> None:
    assert set(executed_chain["executed_order"]) == set(STAGE_ORDER)
    assert executed_chain["executed_order"][0] == STAGE_GROUND_STATE_IMPORT
    assert executed_chain["executed_order"][-1] == STAGE_QUALIFICATION
    assert executed_chain["campaign"].next_actions(executed_chain["ledger"]) == ()


def test_ground_state_import_is_labeled_diagnostic(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    action = campaign.cell_actions(cell)[STAGE_GROUND_STATE_IMPORT]
    payload = json.loads(
        (package.action_dir(action) / "ground_state_import.json").read_text()
    )
    assert payload["ground_source"] == "exact_sector_diagnostic_fixture"
    assert payload["diagnostic_only"] is True
    assert payload["num_qubits"] == 6


def test_trace_respects_the_online_policy_boundaries(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    action = campaign.cell_actions(cell)[STAGE_STATIC_QSE_TRACE]
    payload = json.loads(
        (package.action_dir(action) / "static_qse_trace.json").read_text()
    )
    assert payload["method_id"] == METHOD_RECORD_FIRST
    assert payload["pool_digest"] == cell.pool_policy_digest
    assert payload["rows"]
    # Exact data is absent from online-selection inputs: the selector receipt
    # carries only the acquisition order and the record-stop reason.
    receipt = payload["selection_receipt"]
    assert set(receipt) == {"acquisition_order", "record_stop", "residual_marker"}
    trigger = payload["trigger"]
    # Davidson may start only on genuine pool exhaustion.
    if trigger and trigger.get("fired"):
        assert trigger["reason"] == "record_pool_exhausted"
        assert trigger["trigger_source"] == "record_selector_terminal_receipt"
    else:
        assert trigger["reason"] in {
            "record_residual_converged",
            "record_acquisition_not_terminal",
            "record_stop_beyond_dimension_cap",
        }
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False


def test_exchange_telemetry_records_whole_supports(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    action = campaign.cell_actions(cell)[STAGE_STATIC_QSE_TRACE]
    payload = json.loads(
        (package.action_dir(action) / "static_qse_trace.json").read_text()
    )
    record_rows = [r for r in payload["rows"] if r.get("support_indices") is not None]
    assert record_rows
    for row in record_rows:
        assert len(row["support_indices"]) == row["support_size"]
    assert payload["terminal_support_indices"] == record_rows[-1]["support_indices"]


def test_transitions_and_green_products_are_complete(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    actions = campaign.cell_actions(cell)
    transitions = json.loads(
        (package.action_dir(actions[STAGE_TRANSITIONS]) / "transitions.json").read_text()
    )
    assert [rec["name"] for rec in transitions["transition_observables"]] == [
        "impurity_occupation",
        "impurity_double_occupancy",
    ]
    assert len(transitions["root_energies"]) == transitions["retained_rank"]

    green = json.loads(
        (package.action_dir(actions[STAGE_GREEN_FUNCTION]) / "green_function.json").read_text()
    )
    assert green["schema_version"] == "qse_green_function_v1"
    assert [m["label"] for m in green["modes"]] == ["d_up", "d_down"]
    for mode in green["modes"]:
        deficit = mode["diagonal_sum_rule_diagnostics"]["source_norm_canonical_deficit_abs"]
        assert deficit < 1.0e-8


def test_resources_report_all_four_axes(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    action = campaign.cell_actions(cell)[STAGE_RESOURCE_EVALUATION]
    payload = json.loads(
        (package.action_dir(action) / "resource_evaluation.json").read_text()
    )
    axes = payload["axes"]
    assert set(axes) == {"proxy", "graph_span", "qiskit_compiled", "estimator"}
    assert set(axes["graph_span"]) == {"n2q", "d2q", "dc"}
    assert axes["estimator"]["convention"] == "record_pencil_qwc_basis_cover"
    assert axes["estimator"]["queries"] >= 1
    assert axes["qiskit_compiled"] == {"status": "not_requested"}


def test_qualification_is_typed_and_classifies(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    ledger = executed_chain["ledger"]
    action = campaign.cell_actions(cell)[STAGE_QUALIFICATION]
    payload = json.loads(
        (package.action_dir(action) / "qualification.json").read_text()
    )
    assert payload["status"] in {"qualified", "candidate"}
    assert set(payload["gates"]) >= {
        "window_error_gate",
        "generalized_residual_gate",
        "sum_rule_deficit_gate",
        "spectral_negativity_gate",
    }
    assert payload["selection"]["pool_digest"] == cell.pool_policy_digest
    assert payload["resources"] is not None
    attempt = ledger.completed(action.action_id)
    assert attempt is not None
    assert attempt.qualification_status == payload["status"]


def test_verified_attempts_round_trip_from_disk(executed_chain: dict) -> None:
    package = executed_chain["package"]
    for action in package.actions:
        attempt = verify_achieved_action(action, package.artifact_root)
        assert attempt.disposition == "completed"


def test_spectral_postprocessing_is_nonnegative(executed_chain: dict) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    action = campaign.cell_actions(cell)[STAGE_SPECTRAL_POSTPROCESSING]
    payload = json.loads(
        (package.action_dir(action) / "spectral_postprocessing.json").read_text()
    )
    for observable in payload["spectral"]["observables"]:
        assert min(observable["values"]) >= 0.0


def test_wrong_spectral_digest_fails_closed(tmp_path: Path) -> None:
    cell = _siam_cell()
    campaign = StaticQSECampaign(campaign_id="c", cells=(cell,))
    package = build_local_package(
        campaign,
        campaign.all_actions(),
        package_root=tmp_path,
        cell_contexts={cell.digest(): _context()},
    )
    action = campaign.cell_actions(cell)[STAGE_GROUND_STATE_IMPORT]
    action_dir = package.action_dir(action)
    action_file = json.loads((action_dir / "ACTION.json").read_text())
    action_file["context"]["spectral_request"]["kernel"]["eta"] = 0.5
    (action_dir / "ACTION.json").write_text(json.dumps(action_file))
    with pytest.raises(StaticQSECellError, match="spectral request digest"):
        run_action(action_dir / "ACTION.json", action_dir)


# --- declared switch policy -------------------------------------------------


def _early_switch_cell(policy) -> StaticQSECellKey:
    key = siam_physics_key("siam_nb2_fixture")
    return StaticQSECellKey(
        physics=siam_physics_key_typed("siam_nb2_fixture"),
        ground_state_artifact_id=_diagnostic_artifact_ref("siam_nb2_fixture").artifact_id,
        method_id="early_switch",
        target_roots=DEFAULT_TARGET_ROOTS,
        dimension_cap=64,
        pool_policy_digest=ordered_pool_digest(build_siam_pool_elements(key)),
        spectral_request_digest=canonical_digest(_SPECTRAL_REQUEST),
        resource_policy_id="graph_span_two_qubit_only_v1_plus_pencil_estimator_v1",
        method_policy=policy,
    )


def test_early_switch_is_a_declared_cell_not_a_code_edit(
    tmp_path: Path, executed_chain: dict
) -> None:
    """A threshold trigger runs as its own cell and switches earlier."""

    from pipelines.qse_spectra.switch_policy import window_residual_stall_policy

    policy = window_residual_stall_policy(stall_rounds=2, improvement_ratio=0.5)
    cell = _early_switch_cell(policy)
    campaign = StaticQSECampaign(campaign_id="early_switch_probe", cells=(cell,))
    package = build_local_package(
        campaign,
        campaign.all_actions(),
        package_root=tmp_path,
        cell_contexts={cell.digest(): _context()},
    )
    actions = campaign.cell_actions(cell)
    for stage in (STAGE_GROUND_STATE_IMPORT, STAGE_STATIC_QSE_TRACE):
        action_dir = package.action_dir(actions[stage])
        run_action(action_dir / "ACTION.json", action_dir)

    payload = json.loads(
        (package.action_dir(actions[STAGE_STATIC_QSE_TRACE]) / "static_qse_trace.json").read_text()
    )
    decision = payload["trigger"]["switch_decision"]
    assert decision["switch_trigger"] == "window_residual_stall"
    assert decision["fired"] is True
    assert decision["policy_digest"] == policy.digest()
    assert decision["inputs_are_online_only"] is True
    assert decision["uses_exact_reference_for_decision"] is False
    # The record phase was truncated at the declared trigger point, and the
    # substituted stop reason is labelled rather than disguised.
    assert len(payload["selection_receipt"]["acquisition_order"]) == decision["handoff_k"]
    assert payload["trigger"]["record_stop"]["synthesized_by_switch_policy"] is True
    assert payload["trigger"]["record_stop"]["underlying_stop_reason"] == "residual_converged"

    # The canonical twin from the module fixture ran the same physics and did
    # not switch: the ablation is a genuinely different declared run.
    canonical_cell = executed_chain["cell"]
    canonical_payload = json.loads(
        (
            executed_chain["package"].action_dir(
                executed_chain["campaign"].cell_actions(canonical_cell)[STAGE_STATIC_QSE_TRACE]
            )
            / "static_qse_trace.json"
        ).read_text()
    )
    assert canonical_payload["trigger"]["switch_decision"]["fired"] is False
    assert canonical_cell.digest() != cell.digest()
    assert len(canonical_payload["selection_receipt"]["acquisition_order"]) > decision["handoff_k"]


def test_canonical_cell_switch_decision_is_recorded_but_inert(
    executed_chain: dict,
) -> None:
    package = executed_chain["package"]
    campaign = executed_chain["campaign"]
    cell = executed_chain["cell"]
    payload = json.loads(
        (
            package.action_dir(campaign.cell_actions(cell)[STAGE_STATIC_QSE_TRACE])
            / "static_qse_trace.json"
        ).read_text()
    )
    decision = payload["trigger"]["switch_decision"]
    assert decision["switch_trigger"] == "pool_exhausted_only"
    assert decision["handoff_k"] is None
    assert cell.method_policy.is_canonical is True
