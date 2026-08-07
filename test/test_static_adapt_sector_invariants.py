from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.problem import (
    FixedCountConstraint,
    ProblemRequest,
    RegisterBlockSpec,
    RegisterLayoutSpec,
    SectorSelection,
)
from pipelines.static_adapt.adapt_pipeline import (
    _DefaultNoPruneStateService,
    _resolve_selected_parameterization_mode,
    _splice_candidate_at_position,
)
from pipelines.static_adapt.builders.primitive_pools import (
    _build_hh_fermionic_reusable_pool,
)
from pipelines.static_adapt.builders.problem_registry import resolve_problem_context
from pipelines.static_adapt.sector_invariants import (
    FixedCountQubitGroup,
    FixedCountSectorStateAuditor,
    audit_candidate_pool_sector_contract,
    audit_generator_sector_contract,
    audit_strict_state_replay,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.ansatz_parameterization import (
    build_parameter_layout,
    project_runtime_theta_block_mean,
)


def _hh_l4_nph1_context():
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=4,
            t=1.0,
            u=8.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.3535533905932738,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def _bond_charge_current_up_term():
    pool = _build_hh_fermionic_reusable_pool(
        num_sites=4,
        t=1.0,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    return next(
        term
        for term in pool
        if term.label
        == "hh_fermionic_reusable::bond_charge_current_nn_up(0,1)"
    )


def _fixed_total_count_context(*, quantity: str = "n_f"):
    layout = RegisterLayoutSpec(
        total_qubits=3,
        fermion_qubits=3,
        boson_qubits=0,
        ordering="blocked",
        boson_encoding=None,
        blocks=(
            RegisterBlockSpec(
                name="fermion",
                kind="fermion",
                start_qubit=0,
                stop_qubit=3,
            ),
        ),
    )
    sector = SectorSelection(
        label="one_particle",
        comparison_space_label="one_particle",
        constraints=(FixedCountConstraint(quantity=quantity, value=1),),
        num_particles=None,
    )
    request = ProblemRequest(
        problem_key="test",
        num_sites=3,
        t=0.0,
        u=0.0,
        dv=0.0,
        omega0=0.0,
        g_ep=0.0,
        n_ph_max=0,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    return SimpleNamespace(layout=layout, sector=sector, request=request)


def _test_term(
    label: str,
    components: list[tuple[str, complex]],
    *,
    execution_mode: str = "termwise_product",
):
    nq = len(components[0][0])
    return SimpleNamespace(
        label=label,
        execution_mode=execution_mode,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(nq, ps=word, pc=coefficient) for word, coefficient in components],
        ),
    )


def test_sector_contract_forces_shared_coordinate_independent_of_route_alias():
    context = _hh_l4_nph1_context()
    term = _bond_charge_current_up_term()
    audit = audit_candidate_pool_sector_contract([term], resolved_problem=context)

    assert audit["passed"] is True
    assert audit["execution_passed"] is True
    assert audit["execution_violation_count"] == 0
    assert audit["requires_logical_shared_parameterization"] is True
    assert audit["logical_shared_required_count"] == 1
    assert (
        _resolve_selected_parameterization_mode(
            problem_key_value="hh",
            pool_key_value="phase3_v1",
            candidate_terms=[term],
            sector_contract_requires_logical_shared=audit[
                "requires_logical_shared_parameterization"
            ],
        )
        == "logical_shared"
    )


def test_archived_independent_bond_current_angles_are_rejected_by_state_contract():
    context = _hh_l4_nph1_context()
    term = _bond_charge_current_up_term()
    reference = context.reference_state.build_state()
    auditor = FixedCountSectorStateAuditor(context)
    executor = CompiledAnsatzExecutor([term], parameterization_mode="per_pauli_term")
    archived_runtime_angles = np.asarray(
        [
            1.5725637596055289,
            0.0017674366138219686,
            -2.7872493146981677e-7,
            -2.7424657265302626e-7,
            -2.18974882007118e-7,
            -2.1719878215548292e-7,
        ],
        dtype=float,
    )

    leaked = executor.prepare_state(archived_runtime_angles, reference)
    audit = auditor.audit(leaked, source="archived_independent_angles")

    assert audit["state_norm"] == pytest.approx(1.0, abs=1e-12)
    assert audit["joint_target_sector_probability"] < 1e-12
    with pytest.raises(RuntimeError, match="problem-sector contract"):
        auditor.assert_valid(leaked, source="archived_independent_angles")


def test_shared_bond_current_angle_preserves_sector_and_strict_replay():
    context = _hh_l4_nph1_context()
    term = _bond_charge_current_up_term()
    reference = context.reference_state.build_state()
    auditor = FixedCountSectorStateAuditor(context)
    executor = CompiledAnsatzExecutor([term], parameterization_mode="logical_shared")

    state = executor.prepare_state(np.asarray([0.26238836784569713]), reference)
    state_audit = auditor.assert_valid(state, source="shared_angle")
    replay = executor.prepare_state(np.asarray([0.26238836784569713]), reference)
    replay_audit = audit_strict_state_replay(
        state,
        replay,
        source="shared_angle",
    )

    assert state_audit["joint_target_sector_probability"] == pytest.approx(
        1.0, abs=1e-12
    )
    for row in state_audit["fixed_count_constraints"]:
        assert row["variance"] == pytest.approx(0.0, abs=1e-12)
    assert replay_audit["passed"] is True
    assert replay_audit["fidelity"] == pytest.approx(1.0, abs=1e-12)


def test_zero_angle_splice_preserves_parent_state_and_parameter_mapping():
    context = _hh_l4_nph1_context()
    term = _bond_charge_current_up_term()
    reference = context.reference_state.build_state()
    parent_ops = [term]
    parent_layout = build_parameter_layout(parent_ops)
    parent_runtime_theta = np.full(
        parent_layout.runtime_parameter_count,
        0.17,
        dtype=float,
    )
    parent_executor = CompiledAnsatzExecutor(
        parent_ops,
        parameterization_mode="logical_shared",
        parameterization_layout=parent_layout,
    )
    parent_state = parent_executor.prepare_state(
        project_runtime_theta_block_mean(parent_runtime_theta, parent_layout),
        reference,
    )

    child_ops, child_runtime_theta = _splice_candidate_at_position(
        ops=parent_ops,
        theta=parent_runtime_theta,
        op=term,
        position_id=0,
        init_theta=0.0,
    )
    child_layout = build_parameter_layout(child_ops)
    child_executor = CompiledAnsatzExecutor(
        child_ops,
        parameterization_mode="logical_shared",
        parameterization_layout=child_layout,
    )
    child_state = child_executor.prepare_state(
        project_runtime_theta_block_mean(child_runtime_theta, child_layout),
        reference,
    )

    replay_audit = audit_strict_state_replay(
        parent_state,
        child_state,
        source="zero_angle_front_splice",
    )
    assert replay_audit["passed"] is True
    assert replay_audit["phase_aligned_l2"] == pytest.approx(0.0, abs=1e-12)


def test_commutator_accumulator_does_not_drop_violation_below_legacy_reducer_cutoff():
    term = _test_term("tiny_x", [("x", 1.0e-8)])
    audit = audit_generator_sector_contract(
        term,
        groups=(
            FixedCountQubitGroup(
                quantity="n_f",
                target=0,
                qubits=(0,),
                scope="full_register",
            ),
        ),
        total_qubits=1,
        tolerance=1.0e-10,
    )

    assert audit["grouped_commutator_l1"]["n_f"] == pytest.approx(1.0e-8)
    assert audit["grouped_preserves_fixed_counts"] is False
    assert audit["execution_preserves_fixed_counts"] is False


def test_termwise_execution_accepts_commuting_grouped_components_but_not_trotter_leakage():
    group = FixedCountQubitGroup(
        quantity="n_f",
        target=1,
        qubits=(0, 1, 2),
        scope="full_register",
    )
    one_bond = _test_term(
        "one_bond_hopping",
        [("exx", 1.0), ("eyy", 1.0)],
    )
    one_bond_audit = audit_generator_sector_contract(
        one_bond,
        groups=(group,),
        total_qubits=3,
    )

    assert one_bond_audit["grouped_preserves_fixed_counts"] is True
    assert one_bond_audit["components_individually_preserve_fixed_counts"] is False
    assert one_bond_audit["all_nonzero_components_mutually_commute"] is True
    assert one_bond_audit["execution_preserves_fixed_counts"] is True
    assert one_bond_audit["requires_logical_shared_parameterization"] is True

    two_bonds = _test_term(
        "two_bond_hopping",
        [
            ("exx", 1.0),
            ("eyy", 1.0),
            ("xxe", 1.0),
            ("yye", 1.0),
        ],
    )
    two_bond_audit = audit_generator_sector_contract(
        two_bonds,
        groups=(group,),
        total_qubits=3,
    )

    assert two_bond_audit["grouped_preserves_fixed_counts"] is True
    assert two_bond_audit["all_nonzero_components_mutually_commute"] is False
    assert two_bond_audit["noncommuting_component_pair_count"] > 0
    assert two_bond_audit["execution_preserves_fixed_counts"] is False

    grouped_exact = _test_term(
        "two_bond_hopping_exact",
        [
            ("exx", 1.0),
            ("eyy", 1.0),
            ("xxe", 1.0),
            ("yye", 1.0),
        ],
        execution_mode="grouped_exact",
    )
    grouped_exact_audit = audit_generator_sector_contract(
        grouped_exact,
        groups=(group,),
        total_qubits=3,
    )

    assert grouped_exact_audit["all_nonzero_components_mutually_commute"] is False
    assert grouped_exact_audit["execution_preserves_fixed_counts"] is True


def test_pool_audit_exposes_execution_violation_indices_and_counts():
    context = _fixed_total_count_context()
    safe = _test_term(
        "safe_one_bond",
        [("exx", 1.0), ("eyy", 1.0)],
    )
    unsafe = _test_term(
        "unsafe_two_bonds",
        [
            ("exx", 1.0),
            ("eyy", 1.0),
            ("xxe", 1.0),
            ("yye", 1.0),
        ],
    )

    audit = audit_candidate_pool_sector_contract(
        [safe, unsafe],
        resolved_problem=context,
    )

    assert audit["passed"] is True
    assert audit["execution_passed"] is False
    assert audit["grouped_violation_count"] == 0
    assert audit["execution_violation_count"] == 1
    assert audit["execution_violation_indices"] == [1]
    assert audit["execution_violation_labels"] == ["unsafe_two_bonds"]


def test_fast_state_assertion_uses_joint_mask_and_detailed_audit_remains_available():
    context = _fixed_total_count_context()
    auditor = FixedCountSectorStateAuditor(context)
    valid_state = np.zeros(8, dtype=complex)
    valid_state[1] = 1.0

    assert auditor.assert_valid_fast(valid_state, source="fast_valid") is None
    detailed = auditor.audit(valid_state, source="detailed_valid")
    assert detailed["checked"] is True
    assert detailed["passed"] is True
    assert detailed["joint_target_sector_probability"] == pytest.approx(1.0)

    invalid_state = np.zeros(8, dtype=complex)
    invalid_state[0] = 1.0
    with pytest.raises(RuntimeError, match="problem-sector contract"):
        auditor.assert_valid_fast(invalid_state, source="fast_invalid")


def test_default_state_service_passes_fixed_sector_basis_to_compiled_executor():
    context = _fixed_total_count_context()
    auditor = FixedCountSectorStateAuditor(context)
    reference = np.zeros(8, dtype=complex)
    reference[1] = 1.0
    service = _DefaultNoPruneStateService(
        reference_state=reference,
        state_backend="compiled",
        parameterization_mode="logical_shared",
        pauli_action_cache={},
        fixed_count_auditor=auditor,
    )
    term = _test_term(
        "number_preserving_grouped",
        [("exx", 0.5), ("eyy", 0.5)],
        execution_mode="grouped_exact",
    )

    executor = service.build_executor([term])

    np.testing.assert_array_equal(
        executor.invariant_basis_indices,
        np.asarray([1, 2, 4], dtype=np.int64),
    )
    assert executor.invariant_basis_indices is not None
    assert executor.invariant_basis_indices.flags.writeable is False


def test_unsupported_fixed_count_quantity_fails_closed_in_pool_and_state_audits():
    context = _fixed_total_count_context(quantity="unknown_conserved_charge")
    term = _test_term("identity", [("eee", 1.0)])
    pool_audit = audit_candidate_pool_sector_contract(
        [term],
        resolved_problem=context,
    )

    assert pool_audit["checked"] is False
    assert pool_audit["passed"] is False
    assert pool_audit["fixed_count_support_complete"] is False
    assert pool_audit["skip_reason"] == "unsupported_fixed_count_quantities"
    assert pool_audit["unsupported_fixed_count_quantities"] == [
        "unknown_conserved_charge"
    ]

    auditor = FixedCountSectorStateAuditor(context)
    state = np.zeros(8, dtype=complex)
    state[0] = 1.0
    state_audit = auditor.audit(state, source="unsupported")
    assert state_audit["checked"] is False
    assert state_audit["passed"] is False
    assert state_audit["fixed_count_support_complete"] is False
    with pytest.raises(RuntimeError, match="fixed-count quantities are unsupported"):
        auditor.assert_valid_fast(state, source="unsupported")
