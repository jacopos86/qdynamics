from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import numpy as np
import pytest

from pipelines.static_adapt.adapt_pipeline import (
    _candidate_physical_tangent_operand_identity,
    _physical_tangent_operand_identity,
    _projective_state_overlap_estimator_call_key,
    _restored_estimator_prefix_checkpoint_state,
)
from pipelines.static_adapt.estimator_call_ledger import (
    CALL_KEY_SCHEMA,
    CALL_KEY_SCHEMA_V2,
    EstimatorCallKey,
    EstimatorCallLedger,
    PhysicalTangentOperandIdentity,
    PRIMITIVE_SET_SUMMARY_SCHEMA,
    build_formal_manifold_query_closure_from_estimator_ledger,
    is_optimizer_energy_scope,
    is_optimizer_or_guard_energy_scope,
    optimizer_nfev_from_occurrence_summary,
    projective_state_fingerprint,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _key(
    *,
    state: str = "projective-state-1",
    kind: str = "energy",
    formula: str = "hamiltonian_expectation",
    pair: tuple[str, str] | None = None,
) -> EstimatorCallKey:
    return EstimatorCallKey(
        projective_state_fingerprint=state,
        hamiltonian_fingerprint="hh-hamiltonian-1",
        backend_fingerprint="exact-statevector-backend-v1",
        precision_contract="float64-exact",
        primitive_kind=kind,
        observable_or_formula_identity=formula,
        symmetric_pair=pair,
    )


def _physical_tangent(
    *,
    derivative_circuit: str = "ordered-derivative-circuit-1",
    generator: str = "generator-x0-y1",
    insertion_position: int = 3,
    tie_map: str = "raw-parameter-tie-map",
) -> PhysicalTangentOperandIdentity:
    return PhysicalTangentOperandIdentity(
        derivative_circuit_fingerprint=derivative_circuit,
        generator_fingerprint=generator,
        insertion_position=insertion_position,
        parameterization_tie_map_fingerprint=tie_map,
    )


def _v2_key(
    *,
    state: str = "projective-state-1",
    kind: str = "coordinate_gradient",
    formula: str = "commutator_gradient_v2",
    operand: PhysicalTangentOperandIdentity | str | None = None,
    pair: tuple[
        PhysicalTangentOperandIdentity | str,
        PhysicalTangentOperandIdentity | str,
    ]
    | None = None,
) -> EstimatorCallKey:
    return EstimatorCallKey(
        projective_state_fingerprint=state,
        hamiltonian_fingerprint="hh-hamiltonian-1",
        backend_fingerprint="exact-statevector-backend-v1",
        precision_contract="float64-exact",
        primitive_kind=kind,
        observable_or_formula_identity=formula,
        symmetric_pair=pair,
        schema=CALL_KEY_SCHEMA_V2,
        operand_identity=operand,
    )


def _single_pauli_ansatz_term(label: str, pauli_exyz: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=str(label),
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(len(pauli_exyz), ps=pauli_exyz, pc=1.0)],
        ),
    )


def test_v1_call_key_literal_parsing_and_serialization_remain_unchanged():
    payload = {
        "schema": "estimator_call_key_v1",
        "projective_state_fingerprint": "projective-state-1",
        "hamiltonian_fingerprint": "hh-hamiltonian-1",
        "backend_fingerprint": "exact-statevector-backend-v1",
        "precision_contract": "float64-exact",
        "primitive_kind": "metric_element",
        "observable_or_formula_identity": "fubini_study_metric",
        "symmetric_pair": ["coordinate-1", "coordinate-2"],
    }
    restored = EstimatorCallKey.from_dict(payload)

    assert CALL_KEY_SCHEMA == "estimator_call_key_v1"
    assert restored.schema == CALL_KEY_SCHEMA
    assert restored.operand_identity is None
    assert restored.as_dict() == payload
    assert restored.primitive_id == EstimatorCallKey(
        projective_state_fingerprint="projective-state-1",
        hamiltonian_fingerprint="hh-hamiltonian-1",
        backend_fingerprint="exact-statevector-backend-v1",
        precision_contract="float64-exact",
        primitive_kind="metric_element",
        observable_or_formula_identity="fubini_study_metric",
        symmetric_pair=("coordinate-2", "coordinate-1"),
    ).primitive_id


def test_v2_candidate_and_accepted_aliases_share_one_physical_tangent_charge():
    candidate_tangent = _physical_tangent()
    accepted_tangent = PhysicalTangentOperandIdentity.from_dict(
        candidate_tangent.as_dict()
    )
    candidate_key = _v2_key(operand=candidate_tangent)
    accepted_key = _v2_key(operand=accepted_tangent)

    assert candidate_tangent.operand_id == accepted_tangent.operand_id
    assert candidate_key.primitive_id == accepted_key.primitive_id
    for representational_label in (
        "candidate_label",
        "accepted_coordinate_label",
        "route",
        "branch_id",
        "frame_id",
    ):
        assert representational_label not in candidate_tangent.as_dict()
        assert representational_label not in candidate_key.as_dict()

    ledger = EstimatorCallLedger()
    candidate_receipt = ledger.record_call(
        candidate_key,
        component="N_grad",
        consumer_scope="sr_selector:candidate-label",
        branch_id="selector-branch",
    )
    accepted_receipt = ledger.record_call(
        accepted_key,
        component="N_grad",
        consumer_scope="fm_growth:active-coordinate-label",
        branch_id="accepted-branch",
    )

    assert candidate_receipt.charged is True
    assert accepted_receipt.charged is False
    assert ledger.summary()["S_unique"] == 1
    serialized = json.loads(json.dumps(ledger.to_payload()))
    restored = EstimatorCallLedger.from_payload(serialized)
    assert restored.to_payload() == serialized
    assert restored.summary()["S_unique"] == 1


@pytest.mark.parametrize("insertion_position", [0, 1])
def test_adapt_zero_amplitude_admission_reuses_gradient_metric_and_hessian_ids(
    insertion_position: int,
):
    old_term = _single_pauli_ansatz_term("old-label", "xe")
    candidate_term = _single_pauli_ansatz_term("candidate-label", "ey")
    accepted_alias = _single_pauli_ansatz_term("accepted-label", "ey")
    pre_admission_ops = [old_term]
    accepted_ops = [old_term]
    accepted_ops.insert(int(insertion_position), accepted_alias)
    accepted_old_index = 1 if int(insertion_position) == 0 else 0
    pre_admission_theta = np.asarray([0.37], dtype=float)
    accepted_theta = np.insert(
        pre_admission_theta,
        int(insertion_position),
        0.0,
    )

    candidate_operand = _candidate_physical_tangent_operand_identity(
        pre_admission_ops,
        pre_admission_theta,
        candidate_term,
        insertion_position=int(insertion_position),
        parameterization_mode="logical_shared",
    )
    accepted_operand = _physical_tangent_operand_identity(
        accepted_ops,
        int(insertion_position),
        logical_theta_now=accepted_theta,
        parameterization_mode="logical_shared",
        zero_amplitude_indices=(int(insertion_position),),
    )
    candidate_old_operand = _physical_tangent_operand_identity(
        (
            [candidate_term, old_term]
            if int(insertion_position) == 0
            else [old_term, candidate_term]
        ),
        int(accepted_old_index),
        logical_theta_now=accepted_theta,
        parameterization_mode="logical_shared",
        zero_amplitude_indices=(int(insertion_position),),
    )
    accepted_old_operand = _physical_tangent_operand_identity(
        accepted_ops,
        int(accepted_old_index),
        logical_theta_now=accepted_theta,
        parameterization_mode="logical_shared",
        zero_amplitude_indices=(int(insertion_position),),
    )
    pre_admission_old_operand = _physical_tangent_operand_identity(
        pre_admission_ops,
        0,
        logical_theta_now=pre_admission_theta,
        parameterization_mode="logical_shared",
    )

    assert candidate_operand.operand_id == accepted_operand.operand_id
    assert candidate_old_operand.operand_id == accepted_old_operand.operand_id
    assert candidate_old_operand.operand_id == pre_admission_old_operand.operand_id

    shifted_point_operand = _physical_tangent_operand_identity(
        pre_admission_ops,
        0,
        logical_theta_now=np.asarray([0.38], dtype=float),
        parameterization_mode="logical_shared",
    )
    assert shifted_point_operand.operand_id != pre_admission_old_operand.operand_id

    ledger = EstimatorCallLedger()
    for primitive_kind, formula, candidate_pair, accepted_pair in (
        (
            "metric_element",
            "fubini_study_metric_v2",
            (candidate_old_operand, candidate_operand),
            (accepted_old_operand, accepted_operand),
        ),
        (
            "hessian_element",
            "energy_hessian_v2",
            (candidate_old_operand, candidate_operand),
            (accepted_old_operand, accepted_operand),
        ),
    ):
        candidate_key = _v2_key(
            kind=primitive_kind,
            formula=formula,
            pair=candidate_pair,
        )
        accepted_key = _v2_key(
            kind=primitive_kind,
            formula=formula,
            pair=accepted_pair,
        )
        assert candidate_key.primitive_id == accepted_key.primitive_id
        first = ledger.record_call(
            candidate_key,
            component="N_metric",
            consumer_scope="selector_candidate_geometry",
            branch_id="candidate-branch",
        )
        second = ledger.record_call(
            accepted_key,
            component="N_metric",
            consumer_scope="same_ray_accepted_geometry",
            branch_id="accepted-branch",
        )
        assert first.charged is True
        assert second.charged is False

    selector_gradient = _v2_key(
        formula="coordinate_energy_gradient_v2",
        operand=candidate_operand,
    )
    accepted_gradient = _v2_key(
        formula="coordinate_energy_gradient_v2",
        operand=accepted_operand,
    )
    assert selector_gradient.primitive_id == accepted_gradient.primitive_id
    selector_receipt = ledger.record_call(
        selector_gradient,
        component="N_grad",
        consumer_scope="gradient_surface",
        branch_id="candidate-branch",
    )
    growth_receipt = ledger.record_call(
        accepted_gradient,
        component="N_grad",
        consumer_scope="same_ray_growth",
        branch_id="accepted-branch",
    )
    assert selector_receipt.charged is True
    assert growth_receipt.charged is False
    assert ledger.summary()["S_unique"] == 3


def test_phase1_phase2_phase3_charge_scalar_union_not_phase_sum():
    candidate = _physical_tangent(
        derivative_circuit="candidate-circuit",
        generator="candidate-generator",
        insertion_position=2,
    )
    active = _physical_tangent(
        derivative_circuit="active-circuit",
        generator="active-generator",
        insertion_position=0,
    )
    gradient = _v2_key(
        kind="coordinate_gradient",
        formula="coordinate_energy_gradient_v2",
        operand=candidate,
    )
    self_metric = _v2_key(
        kind="metric_element",
        formula="fubini_study_metric_v2",
        pair=(candidate, candidate),
    )
    self_hessian = _v2_key(
        kind="hessian_element",
        formula="energy_hessian_v2",
        pair=(candidate, candidate),
    )
    cross_metric = _v2_key(
        kind="metric_element",
        formula="fubini_study_metric_v2",
        pair=(active, candidate),
    )
    cross_hessian = _v2_key(
        kind="hessian_element",
        formula="energy_hessian_v2",
        pair=(active, candidate),
    )

    ledger = EstimatorCallLedger()
    assert ledger.record_call(
        gradient,
        component="N_grad",
        consumer_scope="child_phase1",
    ).charged
    phase2 = [
        ledger.record_call(
            gradient,
            component="N_grad",
            consumer_scope="child_phase2",
        ),
        ledger.record_call(
            self_metric,
            component="N_metric",
            consumer_scope="child_phase2",
        ),
        ledger.record_call(
            self_hessian,
            component="N_metric",
            consumer_scope="child_phase2",
        ),
    ]
    assert [receipt.charged for receipt in phase2] == [False, True, True]
    phase3 = [
        ledger.record_call(
            key,
            component=("N_grad" if key is gradient else "N_metric"),
            consumer_scope="child_phase3",
        )
        for key in (
            gradient,
            self_metric,
            self_hessian,
            cross_metric,
            cross_hessian,
        )
    ]
    assert [receipt.charged for receipt in phase3] == [
        False,
        False,
        False,
        True,
        True,
    ]
    summary = ledger.summary()
    assert summary["S_unique"] == 5
    assert summary["deduplicated_reuse_occurrence_count"] == 4


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("derivative_circuit_fingerprint", "ordered-derivative-circuit-2"),
        ("generator_fingerprint", "generator-z0-x1"),
        ("insertion_position", 4),
        ("parameterization_tie_map_fingerprint", "tied-parameter-map"),
        ("tangent_convention", "raw_parameter_derivative_v1"),
    ],
)
def test_v2_physically_distinct_tangents_never_collide(field, replacement):
    baseline = _physical_tangent()
    distinct = replace(baseline, **{field: replacement})

    assert distinct.operand_id != baseline.operand_id
    assert _v2_key(operand=distinct).primitive_id != _v2_key(
        operand=baseline
    ).primitive_id


def test_v2_gram_operand_pair_reversal_is_one_primitive():
    left = _physical_tangent(generator="generator-left", insertion_position=1)
    right = _physical_tangent(generator="generator-right", insertion_position=2)
    forward = _v2_key(
        kind="metric_element",
        formula="fubini_study_metric_v2",
        pair=(left, right),
    )
    reverse = _v2_key(
        kind="metric_element",
        formula="fubini_study_metric_v2",
        pair=(right, left),
    )

    assert forward.symmetric_pair == tuple(sorted((left.operand_id, right.operand_id)))
    assert reverse.symmetric_pair == forward.symmetric_pair
    assert reverse.primitive_id == forward.primitive_id

    ledger = EstimatorCallLedger()
    first = ledger.record_call(
        forward,
        component="N_metric",
        consumer_scope="phase3_joint_gram",
        branch_id="candidate-branch",
    )
    second = ledger.record_call(
        reverse,
        component="N_metric",
        consumer_scope="same_ray_growth_gram",
        branch_id="accepted-branch",
    )
    assert first.charged is True
    assert second.charged is False
    assert ledger.summary()["N_metric"] == 1


def test_projective_endpoint_overlap_is_phase_and_direction_invariant():
    before = np.asarray([1.0, 1.0j], dtype=complex) / np.sqrt(2.0)
    after = np.asarray([1.0, -1.0j], dtype=complex) / np.sqrt(2.0)
    kwargs = {
        "hamiltonian_fingerprint": "hh-hamiltonian-1",
        "backend_fingerprint": "exact-statevector-backend-v1",
        "precision_contract": "float64-exact",
    }
    forward = _projective_state_overlap_estimator_call_key(
        before,
        after,
        **kwargs,
    )
    reverse_rephased = _projective_state_overlap_estimator_call_key(
        np.exp(0.37j) * after,
        np.exp(-0.29j) * before,
        **kwargs,
    )
    distinct = _projective_state_overlap_estimator_call_key(
        before,
        np.asarray([1.0, 0.0], dtype=complex),
        **kwargs,
    )

    assert forward.primitive_kind == "state_overlap"
    assert forward.schema == CALL_KEY_SCHEMA_V2
    assert reverse_rephased.primitive_id == forward.primitive_id
    assert distinct.primitive_id != forward.primitive_id

    ledger = EstimatorCallLedger()
    first = ledger.record_call(
        forward,
        component="N_metric",
        consumer_scope="adaptive_trust_endpoint_overlap",
    )
    second = ledger.record_call(
        reverse_rephased,
        component="N_metric",
        consumer_scope="adaptive_trust_endpoint_overlap",
    )
    assert first.charged is True
    assert second.charged is False
    closure = build_formal_manifold_query_closure_from_estimator_ledger(
        ledger,
        winning_branch_ids=None,
        stored_nfev_total=0,
    )
    assert closure["all_executed"]["counts"]["N_cross"] == 1


def test_restored_prefix_receipt_compacts_authenticated_ledger_without_reset():
    ledger = EstimatorCallLedger()
    ledger.record_call(
        _key(kind="hamiltonian_expectation"),
        component="N_H_refit",
        consumer_scope="optimizer:powell",
    )
    ledger.record_call(
        _v2_key(operand=_physical_tangent()),
        component="N_grad",
        consumer_scope="phase3_active_gradient",
    )

    cursor, receipts = _restored_estimator_prefix_checkpoint_state(
        ledger,
        prior_receipts=None,
        source_outer_iteration=21,
    )

    assert cursor["checkpoint_sequence"] == 1
    assert cursor["raw_occurrence_count"] == 2
    assert cursor["unique_primitive_count"] == 2
    assert cursor["unique_components"]["N_H_refit"] == 1
    assert cursor["unique_components"]["N_grad"] == 1
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt["restored_prefix_compaction"] is True
    assert receipt["outer_iteration"] == 21
    assert receipt["raw_occurrence_delta"]["total"] == 2
    assert receipt["unique_primitive_delta"]["S_unique"] == 2


def test_v2_physical_tangent_key_uses_projective_global_phase_identity():
    state = np.array([0.5 + 0.25j, -0.125j, 0.75 - 0.5j])
    rephased = 2.3 * np.exp(-0.71j) * state
    tangent = _physical_tangent()
    first_key = _v2_key(
        state=projective_state_fingerprint(state),
        operand=tangent,
    )
    rephased_key = _v2_key(
        state=projective_state_fingerprint(rephased),
        operand=tangent,
    )

    assert first_key.projective_state_fingerprint == (
        rephased_key.projective_state_fingerprint
    )
    assert first_key.primitive_id == rephased_key.primitive_id

    ledger = EstimatorCallLedger()
    assert ledger.record_call(
        first_key,
        component="N_grad",
        consumer_scope="selector_gradient",
        branch_id="candidate",
    ).charged
    assert not ledger.record_call(
        rephased_key,
        component="N_grad",
        consumer_scope="accepted_growth_gradient",
        branch_id="accepted",
    ).charged


def test_projective_state_fingerprint_ignores_norm_and_global_phase():
    state = np.array([1.0 + 2.0j, -0.25 + 0.5j, 0.75 - 0.1j])
    globally_rephased = 3.7 * np.exp(0.37j) * state
    assert projective_state_fingerprint(state) == projective_state_fingerprint(
        globally_rephased
    )
    assert projective_state_fingerprint(state) != projective_state_fingerprint(
        state + np.array([0.0, 0.0, 0.01])
    )
    equal_magnitudes = np.array([1.0, 1.0j, -1.0, -1.0j])
    assert projective_state_fingerprint(
        equal_magnitudes
    ) == projective_state_fingerprint(np.exp(1.13j) * equal_magnitudes)
    with pytest.raises(ValueError, match="norm"):
        projective_state_fingerprint(np.zeros(2, dtype=complex))


def test_branch_and_scope_are_consumers_not_primitive_identity():
    identity = _key(kind="coordinate_gradient", formula="commutator:candidate-a")
    ledger = EstimatorCallLedger()
    first = ledger.record_call(
        identity,
        component="N_grad",
        consumer_scope="phase0",
        branch_id="branch-a",
    )
    reused = ledger.record_call(
        identity,
        component="N_grad",
        consumer_scope="phase1",
        branch_id="branch-b",
    )

    assert first.charged is True
    assert reused.charged is False
    summary = ledger.summary()
    assert summary["N_grad"] == summary["S_unique"] == 1
    assert summary["selected_call_occurrence_count"] == 2
    assert summary["deduplicated_reuse_occurrence_count"] == 1
    assert summary["unique_primitive_count_by_consumer_branch"] == {
        "branch-a": 1,
        "branch-b": 1,
    }
    assert summary["unique_primitive_count_by_consumer_scope"] == {
        "phase0": 1,
        "phase1": 1,
    }
    assert "branch" not in identity.as_dict()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("projective_state_fingerprint", "projective-state-2"),
        ("hamiltonian_fingerprint", "hh-hamiltonian-2"),
        ("backend_fingerprint", "shot-backend-v1"),
        ("precision_contract", "shots-4096"),
        ("primitive_kind", "coordinate_gradient"),
        ("observable_or_formula_identity", "different-formula"),
    ],
)
def test_every_required_identity_field_changes_the_primitive_id(field, replacement):
    baseline = _key()
    assert replace(baseline, **{field: replacement}).primitive_id != baseline.primitive_id


def test_different_physical_state_charges_and_symmetric_pair_deduplicates():
    ledger = EstimatorCallLedger()
    forward = _key(
        kind="metric_element",
        formula="fubini_study_metric",
        pair=("coordinate-2", "coordinate-1"),
    )
    reversed_pair = _key(
        kind="metric_element",
        formula="fubini_study_metric",
        pair=("coordinate-1", "coordinate-2"),
    )
    different_state = _key(
        state="projective-state-2",
        kind="metric_element",
        formula="fubini_study_metric",
        pair=("coordinate-1", "coordinate-2"),
    )
    assert forward.symmetric_pair == ("coordinate-1", "coordinate-2")
    assert forward.primitive_id == reversed_pair.primitive_id

    ledger.record_call(
        forward,
        component="N_metric",
        consumer_scope="metric_build",
        branch_id="branch-a",
    )
    ledger.record_call(
        reversed_pair,
        component="N_metric",
        consumer_scope="metric_reuse",
        branch_id="branch-b",
    )
    ledger.record_call(
        different_state,
        component="N_metric",
        consumer_scope="metric_build",
        branch_id="branch-a",
    )
    assert ledger.summary()["N_metric"] == 2


def test_winner_lineage_summary_separates_discarded_branch_work():
    ledger = EstimatorCallLedger()
    outer = _key(state="state-initial", formula="initial-energy")
    gradient = _key(
        state="state-shared",
        kind="coordinate_gradient",
        formula="gradient:candidate-a",
    )
    loser_refit = _key(state="state-loser", formula="powell-objective")
    winner_metric = _key(
        state="state-winner",
        kind="metric_element",
        formula="metric",
        pair=("0", "1"),
    )

    ledger.record_call(
        outer, component="N_H_outer", consumer_scope="initial", branch_id=None
    )
    ledger.record_call(
        gradient,
        component="N_grad",
        consumer_scope="phase0",
        branch_id="discarded",
    )
    ledger.record_call(
        gradient,
        component="N_grad",
        consumer_scope="phase1",
        branch_id="winner",
    )
    ledger.record_call(
        loser_refit,
        component="N_H_refit",
        consumer_scope="boundary_refit",
        branch_id="discarded",
    )
    ledger.record_call(
        winner_metric,
        component="N_metric",
        consumer_scope="phase2",
        branch_id="winner",
    )

    all_branches = ledger.summary()
    assert all_branches["components"] == {
        "N_H_outer": 1,
        "N_H_refit": 1,
        "N_grad": 1,
        "N_metric": 1,
    }
    assert all_branches["S_unique"] == 4

    winner = ledger.summary(branch_ids=("winner",), include_unbranched=True)
    assert winner["components"] == {
        "N_H_outer": 1,
        "N_H_refit": 0,
        "N_grad": 1,
        "N_metric": 1,
    }
    assert winner["S_unique"] == 3
    assert ledger.summary(
        branch_ids=("winner",), include_unbranched=False
    )["S_unique"] == 2
    assert ledger.summary(include_unbranched=False)["S_unique"] == 3


def test_cross_component_reuse_is_charged_to_first_selected_consumer():
    ledger = EstimatorCallLedger()
    energy = _key(state="shared-energy-state", formula="energy")
    ledger.record_call(
        energy,
        component="N_H_outer",
        consumer_scope="outer_guard",
        branch_id=None,
    )
    ledger.record_call(
        energy,
        component="N_H_refit",
        consumer_scope="final_refit",
        branch_id="winner",
    )

    all_branches = ledger.summary()
    assert all_branches["N_H_outer"] == 1
    assert all_branches["N_H_refit"] == 0
    assert all_branches["S_unique"] == 1
    assert all_branches["cross_component_reuse_primitive_ids"] == [
        energy.primitive_id
    ]

    winner_only = ledger.summary(
        branch_ids=("winner",), include_unbranched=False
    )
    assert winner_only["N_H_outer"] == 0
    assert winner_only["N_H_refit"] == winner_only["S_unique"] == 1


def test_primitive_set_summary_uses_global_charged_components_and_stable_digest():
    ledger = EstimatorCallLedger()
    shared_energy = _key(state="shared-energy", formula="shared-energy")
    refit_energy = _key(state="refit-energy", formula="refit-energy")
    gradient = _key(
        state="gradient-state",
        kind="coordinate_gradient",
        formula="gradient",
    )
    metric = _key(
        state="metric-state",
        kind="metric_element",
        formula="metric",
        pair=("0", "1"),
    )

    ledger.record_call(
        shared_energy,
        component="N_H_outer",
        consumer_scope="source_energy",
        branch_id="source",
    )
    ledger.record_call(
        shared_energy,
        component="N_H_refit",
        consumer_scope="prune_refit",
        branch_id="prune",
    )
    ledger.record_call(
        refit_energy,
        component="N_H_refit",
        consumer_scope="prune_refit",
        branch_id="prune",
    )
    ledger.record_call(
        gradient,
        component="N_grad",
        consumer_scope="source_gradient",
        branch_id="source",
    )
    ledger.record_call(
        metric,
        component="N_metric",
        consumer_scope="source_metric",
        branch_id="source",
    )

    requested_ids = [
        metric.primitive_id,
        shared_energy.primitive_id,
        gradient.primitive_id,
        refit_energy.primitive_id,
        shared_energy.primitive_id,
    ]
    summary = ledger.summary_for_primitive_ids(requested_ids)
    reordered = ledger.summary_for_primitive_ids(reversed(requested_ids))

    assert summary["schema"] == PRIMITIVE_SET_SUMMARY_SCHEMA
    assert summary["component_assignment"] == (
        "ledger_global_charged_component_v1"
    )
    assert summary["component_contract"] == [
        "N_H_outer",
        "N_H_refit",
        "N_grad",
        "N_metric",
    ]
    assert summary["components"] == {
        "N_H_outer": 1,
        "N_H_refit": 1,
        "N_grad": 1,
        "N_metric": 1,
    }
    assert summary["N_H_outer"] == 1
    assert summary["N_H_refit"] == 1
    assert summary["N_grad"] == 1
    assert summary["N_metric"] == 1
    assert summary["S_unique"] == summary["unique_primitive_count"] == 4
    assert summary["primitive_ids"] == sorted(set(requested_ids))
    assert summary["component_by_primitive_id"][shared_energy.primitive_id] == (
        "N_H_outer"
    )
    assert len(summary["primitive_set_sha256"]) == 64
    assert reordered == summary
    assert ledger.summary_for_primitive_ids(
        requested_ids[:-2]
    )["primitive_set_sha256"] != summary["primitive_set_sha256"]


def test_primitive_set_summary_empty_input_zero_fills_all_components():
    summary = EstimatorCallLedger().summary_for_primitive_ids(())

    assert summary["components"] == {
        "N_H_outer": 0,
        "N_H_refit": 0,
        "N_grad": 0,
        "N_metric": 0,
    }
    assert summary["S_unique"] == summary["unique_primitive_count"] == 0
    assert summary["primitive_ids"] == []
    assert summary["component_by_primitive_id"] == {}
    assert len(summary["primitive_set_sha256"]) == 64


def test_primitive_set_summary_fails_closed_on_unknown_id():
    ledger = EstimatorCallLedger()
    known = _key(state="known", formula="known")
    ledger.record_call(
        known,
        component="N_H_outer",
        consumer_scope="source_energy",
    )

    with pytest.raises(ValueError, match="absent from the ledger"):
        ledger.summary_for_primitive_ids(
            [known.primitive_id, "unknown-primitive-id"]
        )


def test_full_payload_roundtrip_is_json_serializable_and_validated():
    ledger = EstimatorCallLedger()
    identity = _key(kind="coordinate_gradient", formula="gradient:candidate-a")
    ledger.record_call(
        identity,
        component="N_grad",
        consumer_scope="phase0",
        branch_id="branch-a",
    )
    ledger.record_call(
        identity,
        component="N_grad",
        consumer_scope="phase1",
        branch_id="branch-a",
    )
    payload = json.loads(json.dumps(ledger.to_payload()))
    restored = EstimatorCallLedger.from_payload(payload)
    assert restored.to_payload() == payload

    payload["ledger_fingerprint"] = "tampered"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        EstimatorCallLedger.from_payload(payload)


@pytest.mark.parametrize(
    ("corruption", "expected_error"),
    [
        ("occurrence_scope", "occurrence consumers do not reconcile"),
        ("occurrence_branch", "occurrence consumers do not reconcile"),
        ("consumer_count", "consumer occurrence aggregates"),
        ("consumer_first_last", "consumer occurrence aggregates"),
        ("charged_occurrence", "charged occurrence does not reconcile"),
    ],
)
def test_full_payload_rejects_occurrences_inconsistent_with_aggregates(
    corruption,
    expected_error,
):
    ledger = EstimatorCallLedger()
    identity = _key(kind="coordinate_gradient", formula="gradient:candidate-a")
    for _ in range(2):
        ledger.record_call(
            identity,
            component="N_grad",
            consumer_scope="phase0",
            branch_id="branch-a",
        )
    ledger.record_call(
        identity,
        component="N_grad",
        consumer_scope="phase1",
        branch_id="branch-b",
    )
    payload = json.loads(json.dumps(ledger.to_payload()))
    consumers = {
        row["scope"]: row for row in payload["entries"][0]["consumers"]
    }

    if corruption == "occurrence_scope":
        payload["occurrences"][1]["consumer_scope"] = "phase1"
    elif corruption == "occurrence_branch":
        payload["occurrences"][1]["branch_id"] = "branch-b"
    elif corruption == "consumer_count":
        # Preserve the record-level total so only exact occurrence-to-consumer
        # reconciliation can detect the redistributed counts.
        consumers["phase0"]["occurrence_count"] = 1
        consumers["phase1"]["occurrence_count"] = 2
    elif corruption == "consumer_first_last":
        # Preserve the record-level extrema while corrupting each consumer's
        # exact sequence range.
        consumers["phase0"]["last_seen_sequence"] = 3
        consumers["phase1"]["first_seen_sequence"] = 2
    elif corruption == "charged_occurrence":
        payload["occurrences"][0]["charged"] = False
        payload["occurrences"][1]["charged"] = True
    else:  # pragma: no cover - guarded by the parameterization above.
        raise AssertionError(f"unsupported test corruption: {corruption}")

    with pytest.raises(ValueError, match=expected_error):
        EstimatorCallLedger.from_payload(payload)


def test_concurrent_duplicate_recording_charges_once_and_retains_consumers():
    ledger = EstimatorCallLedger()
    identity = _key(kind="coordinate_gradient", formula="gradient:candidate-a")

    def record_branch(branch_index: int) -> int:
        charged = 0
        for occurrence in range(25):
            receipt = ledger.record_call(
                identity,
                component="N_grad",
                consumer_scope=f"phase-{occurrence % 3}",
                branch_id=f"branch-{branch_index}",
            )
            charged += int(receipt.charged)
        return charged

    with ThreadPoolExecutor(max_workers=8) as executor:
        charged_counts = list(executor.map(record_branch, range(8)))

    assert sum(charged_counts) == 1
    summary = ledger.summary()
    assert summary["N_grad"] == summary["S_unique"] == 1
    assert summary["selected_call_occurrence_count"] == 200
    assert summary["deduplicated_reuse_occurrence_count"] == 199
    assert summary["unique_primitive_count_by_consumer_branch"] == {
        f"branch-{index}": 1 for index in range(8)
    }


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        ("energy:powell_objective", True),
        ("energy:terminal_prune_refit", True),
        ("finite_angle_objective_guard", True),
        ("outer_state_refresh", False),
        ("prune_surrogate_anchor", False),
        ("final_verification", False),
        ("finite_angle_objective_guard_extra", False),
    ],
)
def test_optimizer_or_guard_energy_scope_contract(scope, expected):
    assert is_optimizer_or_guard_energy_scope(scope) is expected


def test_formal_manifold_estimator_closure_partitions_unique_and_executed_work():
    ledger = EstimatorCallLedger()

    initial_energy = _key(
        state="state-initial",
        kind="hamiltonian_expectation",
        formula="initial-energy",
    )
    refresh_energy = _key(
        state="state-refresh",
        kind="hamiltonian_expectation",
        formula="outer-state-refresh",
    )
    winner_objective = _key(
        state="state-winner-objective",
        kind="hamiltonian_expectation",
        formula="powell-objective",
    )
    winner_guard = _key(
        state="state-winner-guard",
        kind="hamiltonian_expectation",
        formula="finite-angle-guard",
    )
    loser_objective = _key(
        state="state-loser-objective",
        kind="hamiltonian_expectation",
        formula="powell-objective",
    )
    loser_guard = _key(
        state="state-loser-guard",
        kind="hamiltonian_expectation",
        formula="finite-angle-guard",
    )
    shared_gradient = _key(
        state="state-shared-gradient",
        kind="coordinate_gradient",
        formula="gradient:candidate-a",
    )
    winner_metric = _key(
        state="state-winner-metric",
        kind="metric_element",
        formula="fubini-study-metric",
        pair=("0", "1"),
    )
    loser_hessian = _key(
        state="state-loser-hessian",
        kind="coordinate_second_derivative",
        formula="energy-hessian",
        pair=("0", "1"),
    )
    winner_hessian_vector = _key(
        state="state-winner-hessian-vector",
        kind="hessian_vector",
        formula="hessian-vector:0",
    )
    loser_cross_state = _key(
        state="state-loser-cross",
        kind="cross_state_tangent",
        formula="cross-state-tangent",
        pair=("old-0", "new-0"),
    )

    # Shared setup work is part of the winning lineage.  The repeated initial
    # objective is a real optimizer occurrence but only one unique oracle call.
    for _ in range(2):
        ledger.record_call(
            initial_energy,
            component="N_H_outer",
            consumer_scope="energy:initial",
            branch_id=None,
        )
    ledger.record_call(
        refresh_energy,
        component="N_H_outer",
        consumer_scope="outer_state_refresh",
        branch_id=None,
    )

    for _ in range(2):
        ledger.record_call(
            winner_objective,
            component="N_H_refit",
            consumer_scope="energy:powell_objective",
            branch_id="winner",
        )
    ledger.record_call(
        winner_guard,
        component="N_H_refit",
        consumer_scope="finite_angle_objective_guard",
        branch_id="winner",
    )
    ledger.record_call(
        loser_objective,
        component="N_H_refit",
        consumer_scope="energy:powell_objective",
        branch_id="discarded",
    )
    ledger.record_call(
        loser_guard,
        component="N_H_refit",
        consumer_scope="finite_angle_objective_guard",
        branch_id="discarded",
    )

    # The loser consumes this gradient first, but the winner consumes the same
    # physical primitive later.  It must be retained in winning/shared unique
    # work and excluded from discarded-only unique overhead.
    ledger.record_call(
        shared_gradient,
        component="N_grad",
        consumer_scope="phase1_gradient",
        branch_id="discarded",
    )
    ledger.record_call(
        shared_gradient,
        component="N_grad",
        consumer_scope="phase2_gradient",
        branch_id="winner",
    )
    for _ in range(2):
        ledger.record_call(
            winner_metric,
            component="N_metric",
            consumer_scope="metric_build",
            branch_id="winner",
        )
    ledger.record_call(
        loser_hessian,
        component="N_metric",
        consumer_scope="hessian_build",
        branch_id="discarded",
    )
    ledger.record_call(
        winner_hessian_vector,
        component="N_metric",
        consumer_scope="hessian_vector_probe",
        branch_id="winner",
    )
    ledger.record_call(
        loser_cross_state,
        component="N_metric",
        consumer_scope="cross_state_registration",
        branch_id="discarded",
    )

    closure = build_formal_manifold_query_closure_from_estimator_ledger(
        ledger,
        winning_branch_ids=("winner",),
        stored_nfev_total=6,
    )

    assert closure["schema"] == "formal_manifold_estimator_ledger_query_closure_v1"
    assert closure["primitive_kind_to_query_category"] == {
        "coordinate_gradient": "N_grad",
        "coordinate_second_derivative": "N_Q",
        "cross_state_tangent": "N_cross",
        "directional_metric_bilinear": "N_G",
        "energy": "N_E",
        "hamiltonian_expectation": "N_E",
        "hessian_element": "N_Q",
        "hessian_vector": "N_Hv",
        "metric_element": "N_G",
        "state_overlap": "N_cross",
        "tangent_or_metric": "N_G",
    }

    winning = closure["winning_branch"]
    assert winning["counts"] == {
        "N_E": 4,
        "N_grad": 1,
        "N_G": 1,
        "N_Q": 0,
        "N_Hv": 1,
        "N_cross": 0,
    }
    assert winning["S_alg"] == winning["unique_primitive_count"] == 7
    assert shared_gradient.primitive_id in winning["primitive_ids"]

    discarded = closure["discarded_branch_operational_overhead"]
    assert discarded["counts"] == {
        "N_E": 2,
        "N_grad": 0,
        "N_G": 0,
        "N_Q": 1,
        "N_Hv": 0,
        "N_cross": 1,
    }
    assert discarded["S_alg"] == discarded["unique_primitive_count"] == 4
    assert shared_gradient.primitive_id not in discarded["primitive_ids"]

    all_executed = closure["all_executed"]
    assert all_executed["counts"] == {
        "N_E": 6,
        "N_grad": 1,
        "N_G": 1,
        "N_Q": 1,
        "N_Hv": 1,
        "N_cross": 1,
    }
    assert all_executed["S_alg"] == all_executed["unique_primitive_count"] == 11

    assert closure["primitive_set_reconciliation"] == {
        "winning_discarded_disjoint": True,
        "union_equals_all_executed": True,
        "winning_count": 7,
        "discarded_count": 4,
        "all_executed_count": 11,
    }

    occurrences = closure["executed_occurrence_accounting"]
    assert occurrences["all_execution"]["total_call_occurrences"] == 15
    assert occurrences["all_execution"]["same_identity_reuse_occurrence_count"] == 4
    assert occurrences["winning_plus_shared_execution"][
        "total_call_occurrences"
    ] == 10
    assert occurrences["winning_plus_shared_execution"][
        "same_identity_reuse_occurrence_count"
    ] == 3
    assert occurrences["discarded_branch_execution"][
        "total_call_occurrences"
    ] == 5

    # N_E includes non-optimizer refresh/verification measurements, whereas
    # nfev is exactly the occurrences whose scopes are optimizer objectives or
    # the finite-angle objective guard.
    assert closure["stored_nfev_total"] == 6
    assert closure["raw_optimizer_nfev_all_execution"] == 5
    assert closure["raw_optimizer_nfev_winning_lineage"] == 4
    assert closure[
        "raw_optimizer_nfev_discarded_operational_overhead"
    ] == 1
    assert closure["stored_nfev_matches_winning_raw_optimizer"] is False
    assert closure["corrected_nfev_total"] == 7
    assert closure["nfev_correction"] == 1
    assert closure["nfev_winning_lineage"] == 5
    assert closure["nfev_discarded_operational_overhead"] == 2
    assert closure["nfev_reconciled"] is True


def test_raw_optimizer_nfev_excludes_guards_and_discarded_beam_work():
    ledger = EstimatorCallLedger()
    shared = _key(state="shared", formula="shared-objective")
    winner = _key(state="winner", formula="winner-objective")
    winner_guard = _key(state="winner-guard", formula="winner-guard")
    discarded = _key(state="discarded", formula="discarded-objective")
    discarded_guard = _key(
        state="discarded-guard", formula="discarded-guard"
    )

    for _ in range(2):
        ledger.record_call(
            shared,
            component="N_H_outer",
            consumer_scope="energy:initial_state",
            branch_id=None,
        )
    for _ in range(3):
        ledger.record_call(
            winner,
            component="N_H_refit",
            consumer_scope="energy:beam_local_reopt",
            branch_id="winner",
        )
    ledger.record_call(
        winner_guard,
        component="N_H_refit",
        consumer_scope="finite_angle_objective_guard",
        branch_id="winner",
    )
    for _ in range(5):
        ledger.record_call(
            discarded,
            component="N_H_refit",
            consumer_scope="energy:beam_local_reopt",
            branch_id="discarded",
        )
    ledger.record_call(
        discarded_guard,
        component="N_H_refit",
        consumer_scope="finite_angle_objective_guard",
        branch_id="discarded",
    )

    closure = build_formal_manifold_query_closure_from_estimator_ledger(
        ledger,
        winning_branch_ids=("winner",),
        stored_nfev_total=5,
    )
    assert is_optimizer_energy_scope("energy:beam_local_reopt") is True
    assert is_optimizer_energy_scope("finite_angle_objective_guard") is False
    assert is_optimizer_or_guard_energy_scope(
        "finite_angle_objective_guard"
    ) is True
    winning_occurrences = closure["executed_occurrence_accounting"][
        "winning_plus_shared_hamiltonian_execution"
    ]
    assert optimizer_nfev_from_occurrence_summary(winning_occurrences) == 5
    assert closure["raw_optimizer_nfev_winning_lineage"] == 5
    assert closure["raw_optimizer_nfev_discarded_operational_overhead"] == 5
    assert closure["raw_optimizer_nfev_all_execution"] == 10
    assert closure["stored_nfev_matches_winning_raw_optimizer"] is True
    assert closure["nfev_winning_lineage"] == 6
    assert closure["nfev_discarded_operational_overhead"] == 6
    assert closure["corrected_nfev_total"] == 12
    assert closure["nfev_reconciled"] is True


def test_formal_manifold_estimator_closure_fails_closed_on_unknown_primitive_kind():
    ledger = EstimatorCallLedger()
    ledger.record_call(
        _key(
            state="state-unknown",
            kind="unclassified_quantum_oracle",
            formula="mystery-probe",
        ),
        component="N_metric",
        consumer_scope="mystery_probe",
        branch_id="winner",
    )

    with pytest.raises(RuntimeError, match="primitive kind|primitive_kind|unsupported"):
        build_formal_manifold_query_closure_from_estimator_ledger(
            ledger,
            winning_branch_ids=("winner",),
        )


def test_formal_manifold_estimator_closure_rejects_kind_component_mismatch():
    ledger = EstimatorCallLedger()
    ledger.record_call(
        _key(
            state="state-gradient",
            kind="coordinate_gradient",
            formula="gradient:candidate-a",
        ),
        component="N_H_refit",
        consumer_scope="energy:misclassified-gradient",
        branch_id="winner",
    )

    with pytest.raises(
        RuntimeError,
        match="primitive-kind/legacy-component compatibility",
    ):
        build_formal_manifold_query_closure_from_estimator_ledger(
            ledger,
            winning_branch_ids=("winner",),
        )


def test_formal_manifold_nfev_filters_to_energy_primitive_occurrences():
    ledger = EstimatorCallLedger()
    ledger.record_call(
        _key(
            state="state-metric",
            kind="metric_element",
            formula="metric:0,0",
        ),
        component="N_metric",
        consumer_scope="energy:misleading-scope-name",
        branch_id="winner",
    )
    ledger.record_call(
        _key(
            state="state-energy",
            kind="hamiltonian_expectation",
            formula="optimizer-energy",
        ),
        component="N_H_refit",
        consumer_scope="energy:powell_objective",
        branch_id="winner",
    )

    closure = build_formal_manifold_query_closure_from_estimator_ledger(
        ledger,
        winning_branch_ids=("winner",),
        stored_nfev_total=1,
    )
    assert closure["all_executed"]["counts"] == {
        "N_E": 1,
        "N_grad": 0,
        "N_G": 1,
        "N_Q": 0,
        "N_Hv": 0,
        "N_cross": 0,
    }
    assert closure["corrected_nfev_total"] == 1
    assert closure["nfev_correction"] == 0
    assert closure["raw_energy_evaluation_occurrence_count"] == 1
