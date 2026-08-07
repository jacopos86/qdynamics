from __future__ import annotations

from dataclasses import replace
import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.scaffold.hh_continuation_scoring as scoring
from pipelines.scaffold.hh_continuation_scoring import (
    EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1,
    FullScoreConfig,
    MeasurementCacheAudit,
    Phase1CompileCostOracle,
    SimpleScoreConfig,
    build_candidate_features,
    evaluate_historical_singleton_material_window_coordinate_models,
    measurement_group_keys_for_term,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)
from pipelines.static_adapt.phase3_material_window import (
    Phase3MaterialWindowPolicy,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import compile_polynomial_action
from src.quantum.pauli_polynomial_class import PauliPolynomial, PauliTerm


def _term(label: str) -> object:
    return type(
        "_DummyAnsatzTerm",
        (),
        {
            "label": str(label),
            "polynomial": PauliPolynomial(
                "JW",
                [PauliTerm(len(str(label)), ps=str(label), pc=1.0)],
            ),
        },
    )()


def _hamiltonian() -> object:
    polynomial = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="zz", pc=1.0),
            PauliTerm(2, ps="xe", pc=0.3),
        ],
    )
    return compile_polynomial_action(polynomial, pauli_action_cache={})


def _record(
    *,
    candidate_label: str,
    candidate_pool_index: int,
    position_id: int,
) -> dict[str, object]:
    candidate = _term(candidate_label)
    feature = build_candidate_features(
        stage_name="core",
        candidate_label=str(candidate_label),
        candidate_family="core",
        candidate_pool_index=int(candidate_pool_index),
        position_id=int(position_id),
        append_position=int(position_id),
        positions_considered=[int(position_id)],
        gradient_signed=0.5,
        metric_proxy=1.0,
        sigma_hat=0.0,
        refit_window_indices=[],
        compile_cost=Phase1CompileCostOracle().estimate(
            candidate_term_count=1,
            position_id=int(position_id),
            append_position=int(position_id),
            refit_active_count=0,
            candidate_term=candidate,
        ),
        measurement_stats=MeasurementCacheAudit().estimate(
            measurement_group_keys_for_term(candidate)
        ),
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        cfg=SimpleScoreConfig(z_alpha=0.0),
        cheap_score_cfg=FullScoreConfig(z_alpha=0.0),
    )
    return {
        "feature": feature,
        "candidate_pool_index": int(candidate_pool_index),
        "position_id": int(position_id),
        "candidate_term": candidate,
        "phase2_raw_score": 1.0,
    }


@pytest.fixture
def geometry_case() -> dict[str, object]:
    selected = [_term("xe"), _term("yz"), _term("zx")]
    theta = np.asarray([0.23, -0.31, 0.17], dtype=float)
    rng = np.random.default_rng(9)
    psi_ref = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    psi_ref = np.asarray(psi_ref / np.linalg.norm(psi_ref), dtype=complex)
    psi_state = CompiledAnsatzExecutor(selected).prepare_state(theta, psi_ref)
    return {
        "selected": selected,
        "theta": theta,
        "psi_ref": psi_ref,
        "psi_state": psi_state,
        "h_compiled": _hamiltonian(),
        "record": _record(
            candidate_label="xy",
            candidate_pool_index=7,
            position_id=1,
        ),
    }


def _partial_policy() -> Phase3MaterialWindowPolicy:
    return Phase3MaterialWindowPolicy(
        gram_entry_threshold=0.3,
        hessian_entry_threshold=2.0,
        gram_omitted_l2_tolerance=1.0,
        hessian_omitted_l2_tolerance=1.0,
        # Most tests in this file isolate retained-principal-block and support
        # behavior.  The independent W-by-O closure gate has its own focused
        # test below, so keep it permissive in this shared fixture.
        gram_cross_block_tolerance=1.0e6,
        hessian_cross_block_tolerance=1.0e6,
    )


def _evaluate(case: dict[str, object], **kwargs: object):
    return evaluate_historical_singleton_material_window_coordinate_models(
        [case["record"]],
        cfg=FullScoreConfig(z_alpha=0.0, rho=0.5),
        selected_ops=case["selected"],
        theta=case["theta"],
        psi_ref=case["psi_ref"],
        psi_state=case["psi_state"],
        h_compiled=case["h_compiled"],
        material_window_policy=_partial_policy(),
        pauli_action_cache={},
        **kwargs,
    )


def test_candidate_coupling_screen_never_serializes_placeholder_old_old_blocks(
    geometry_case: dict[str, object],
) -> None:
    context = scoring._selector_scaffold_context(
        selected_ops=geometry_case["selected"],
        theta=geometry_case["theta"],
        psi_ref=geometry_case["psi_ref"],
        psi_state=geometry_case["psi_state"],
        active_indices=(0, 1, 2),
        h_compiled=geometry_case["h_compiled"],
        measure_old_old_geometry=False,
    )
    payload = scoring._exact_insertion_joint_geometry_payload(
        scaffold_context=context,
        candidate_term=geometry_case["record"]["candidate_term"],
        position_id=1,
        h_compiled=geometry_case["h_compiled"],
        pauli_action_cache={},
        state_consistency_tolerance=1e-8,
        acquisition_mode=(
            EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1
        ),
    )

    assert payload["schema"] == "phase3_candidate_coupling_screen_v1"
    assert set(("G_AA", "H_AA", "g_A")).isdisjoint(payload)
    assert payload["placeholder_old_old_blocks_serialized"] is False
    assert payload["G_AA_element_count"] == 0
    assert payload["H_AA_element_count"] == 0
    assert len(payload["G_A_diagonal"]) == 3
    assert len(payload["G_AB"]) == len(payload["H_AB"]) == 3

    measured_context = scoring._selector_scaffold_context(
        selected_ops=geometry_case["selected"],
        theta=geometry_case["theta"],
        psi_ref=geometry_case["psi_ref"],
        psi_state=geometry_case["psi_state"],
        active_indices=(0, 1, 2),
        h_compiled=geometry_case["h_compiled"],
        measure_old_old_geometry=True,
    )
    with pytest.raises(ValueError, match="old_old_geometry_measured=False"):
        scoring._exact_insertion_joint_geometry_payload(
            scaffold_context=measured_context,
            candidate_term=geometry_case["record"]["candidate_term"],
            position_id=1,
            h_compiled=geometry_case["h_compiled"],
            pauli_action_cache={},
            state_consistency_tolerance=1e-8,
            acquisition_mode=(
                EXACT_INSERTION_GEOMETRY_CANDIDATE_COUPLING_SCREEN_V1
            ),
        )


def test_retained_workspace_is_exact_principal_workspace_and_emits_plan(
    geometry_case: dict[str, object],
) -> None:
    evaluation = _evaluate(geometry_case)
    receipt = evaluation.records[0]["material_window_receipt"]
    plan = evaluation.records[0]["estimator_acquisition_plan"]
    assert receipt["retained_indices"] == [2]
    assert receipt["omitted_indices"] == [0, 1]
    assert plan["retained_retained_pairs"] == [[2, 2]]
    assert sorted(plan["retained_omitted_closure_pairs"]) == [[0, 2], [1, 2]]
    assert plan["omitted_omitted_refresh_pairs"] == []
    assert plan["full_geometry_refresh_performed"] is False
    assert plan["screen_gram_diagonal_indices"] == [0, 1, 2]
    assert plan["active_gradient_indices_acquired"] == [2]
    json.dumps(plan, allow_nan=False)

    full_context = scoring._selector_scaffold_context(
        selected_ops=geometry_case["selected"],
        theta=geometry_case["theta"],
        psi_ref=geometry_case["psi_ref"],
        psi_state=geometry_case["psi_state"],
        active_indices=(0, 1, 2),
        h_compiled=geometry_case["h_compiled"],
        measure_old_old_geometry=True,
    )
    full_payload = scoring._exact_insertion_joint_geometry_payload(
        scaffold_context=full_context,
        candidate_term=geometry_case["record"]["candidate_term"],
        position_id=1,
        h_compiled=geometry_case["h_compiled"],
        pauli_action_cache={},
        state_consistency_tolerance=1e-8,
    )
    workspace = evaluation.workspace
    assert workspace.active_indices == (2,)
    assert workspace.G_AA == pytest.approx(
        np.asarray(full_payload["G_AA"])[np.ix_([2], [2])]
    )
    assert workspace.H_AA == pytest.approx(
        np.asarray(full_payload["H_AA"])[np.ix_([2], [2])]
    )
    assert workspace.g_A == pytest.approx(np.asarray(full_payload["g_A"])[[2]])
    assert workspace.G_AB[:, 0] == pytest.approx(
        np.asarray(full_payload["G_AB"])[[2]]
    )
    assert workspace.H_AB[:, 0] == pytest.approx(
        np.asarray(full_payload["H_AB"])[[2]]
    )
    feature = evaluation.records[0]["feature"]
    assert feature.phase3_response_coordinate_indices == [1, 3]
    assert feature.phase3_response_pre_support_count == 2
    assert evaluation.telemetry["joint_linear_solve_policy_requested"] == (
        JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
    )


def test_all_retained_material_model_matches_existing_full_exact_model(
    geometry_case: dict[str, object],
) -> None:
    cfg = FullScoreConfig(
        z_alpha=0.0,
        rho=0.5,
        batch_joint_linear_solve_policy=(
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        batch_joint_context_mode="full_ansatz_v1",
    )
    policy = Phase3MaterialWindowPolicy(
        gram_entry_threshold=0.0,
        hessian_entry_threshold=0.0,
        gram_omitted_l2_tolerance=1.0,
        hessian_omitted_l2_tolerance=1.0,
    )
    material = evaluate_historical_singleton_material_window_coordinate_models(
        [geometry_case["record"]],
        cfg=cfg,
        selected_ops=geometry_case["selected"],
        theta=geometry_case["theta"],
        psi_ref=geometry_case["psi_ref"],
        psi_state=geometry_case["psi_state"],
        h_compiled=geometry_case["h_compiled"],
        material_window_policy=policy,
        pauli_action_cache={},
    )
    full = scoring.evaluate_historical_singleton_coordinate_models(
        [geometry_case["record"]],
        cfg=cfg,
        selected_ops=geometry_case["selected"],
        theta=geometry_case["theta"],
        psi_ref=geometry_case["psi_ref"],
        psi_state=geometry_case["psi_state"],
        h_compiled=geometry_case["h_compiled"],
        pauli_action_cache={},
    )

    for name in ("G_AA", "H_AA", "G_AB", "H_AB", "G_BB", "H_BB", "g_A", "g_B"):
        assert getattr(material.workspace, name) == pytest.approx(
            getattr(full.workspace, name), abs=1e-12
        )
    material_summary = material.records[0]["feature"].phase2_joint_geometry_reuse
    full_summary = full.records[0]["feature"].phase2_joint_geometry_reuse
    assert material_summary["feasible"] is full_summary["feasible"]
    assert material_summary["joint_metric_support_rank"] == (
        full_summary["joint_metric_support_rank"]
    )
    assert material_summary["joint_gain"] == pytest.approx(
        full_summary["joint_gain"], abs=1e-12
    )


def test_closure_failure_refreshes_only_omitted_block_once(
    geometry_case: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_builder = scoring.build_phase3_material_window

    def _forced_closure_failure(**kwargs: object):
        receipt = original_builder(**kwargs)
        return replace(
            receipt,
            closure_satisfied=False,
            closure_reason="forced_test_closure_failure",
            receipt_sha256="",
        )

    monkeypatch.setattr(
        scoring,
        "build_phase3_material_window",
        _forced_closure_failure,
    )
    evaluation = _evaluate(geometry_case)
    row = evaluation.records[0]
    plan = row["estimator_acquisition_plan"]
    refresh = row["feature"].phase2_joint_geometry_reuse[
        "material_window_refresh"
    ]

    assert evaluation.telemetry["full_geometry_refresh_count"] == 1
    assert refresh["performed"] is True
    assert refresh["count"] == 1
    assert "closure_failed" in refresh["reasons"]
    assert plan["omitted_omitted_refresh_pairs"] == [
        [0, 0],
        [0, 1],
        [1, 1],
    ]
    assert plan["full_geometry_refresh_count"] == 1
    assert len(plan["old_old_metric_pairs_acquired"]) == 6
    assert len({tuple(pair) for pair in plan["old_old_metric_pairs_acquired"]}) == 6
    assert evaluation.workspace.active_indices == (0, 1, 2)


def test_same_retained_window_nullity_anchor_does_not_force_refresh(
    geometry_case: dict[str, object],
) -> None:
    first = _evaluate(geometry_case)
    receipt = first.records[0]["material_window_receipt"]
    identity = scoring._batch_record_identity_key(geometry_case["record"])
    second = _evaluate(
        geometry_case,
        prior_retained_support_nullities={
            identity: (
                receipt["measured_active_nullity"],
                receipt["measured_joint_nullity"],
            )
        },
    )
    summary = second.records[0]["feature"].phase2_joint_geometry_reuse
    assert summary["prior_nullity_comparison_scope"] == (
        "same_retained_W_and_W_plus_candidate_v1"
    )
    assert summary["material_window_refresh"]["performed"] is False


def test_same_window_support_nullity_drift_requests_one_full_refresh(
    geometry_case: dict[str, object],
) -> None:
    first = _evaluate(geometry_case)
    receipt = first.records[0]["material_window_receipt"]
    measured_active_nullity = int(receipt["measured_active_nullity"])
    different_active_nullity = 1 - measured_active_nullity
    identity = scoring._batch_record_identity_key(geometry_case["record"])
    drifted = _evaluate(
        geometry_case,
        prior_retained_support_nullities={
            identity: (
                different_active_nullity,
                receipt["measured_joint_nullity"],
            )
        },
    )
    refresh = drifted.records[0]["feature"].phase2_joint_geometry_reuse[
        "material_window_refresh"
    ]
    assert refresh["performed"] is True
    assert refresh["count"] == 1
    assert "active_support_nullity_drift" in refresh["reasons"]
    assert drifted.telemetry["full_geometry_refresh_count"] == 1


def test_candidate_only_empty_active_and_membership_order() -> None:
    psi_ref = np.asarray([1.0, 0.0], dtype=complex)
    h_compiled = compile_polynomial_action(
        PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
        pauli_action_cache={},
    )
    records = [
        _record(candidate_label="x", candidate_pool_index=11, position_id=0),
        _record(candidate_label="y", candidate_pool_index=4, position_id=0),
    ]
    policy = Phase3MaterialWindowPolicy(
        gram_entry_threshold=2.0,
        hessian_entry_threshold=2.0,
        gram_omitted_l2_tolerance=1.0,
        hessian_omitted_l2_tolerance=1.0,
    )
    evaluation = evaluate_historical_singleton_material_window_coordinate_models(
        records,
        cfg=FullScoreConfig(z_alpha=0.0, rho=0.5),
        selected_ops=[],
        theta=np.zeros(0, dtype=float),
        psi_ref=psi_ref,
        psi_state=psi_ref,
        h_compiled=h_compiled,
        material_window_policy=policy,
        pauli_action_cache={},
    )

    assert [row["candidate_pool_index"] for row in evaluation.records] == [11, 4]
    assert evaluation.telemetry["membership_preserved"] is True
    assert evaluation.telemetry["order_preserved"] is True
    assert evaluation.workspace.active_indices == ()
    for row in evaluation.records:
        receipt = row["material_window_receipt"]
        plan = row["estimator_acquisition_plan"]
        feature = row["feature"]
        assert receipt["closure_reason"] == "candidate_only"
        assert receipt["retained_indices"] == []
        assert plan["old_old_metric_pair_count"] == 0
        assert plan["active_gradient_indices_acquired"] == []
        assert feature.phase3_response_coordinate_indices == [0]
        assert feature.phase3_response_pre_support_count == 1
