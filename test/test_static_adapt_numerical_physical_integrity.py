from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pipelines.contracts.problem import (
    ExactTargetSpec,
    FixedCountConstraint,
    HamiltonianFamilyCapabilities,
    ProblemRequest,
    ReferenceStateSpec,
    RegisterBlockSpec,
    RegisterLayoutSpec,
    ResolvedProblemContext,
    SectorSelection,
    TruncationConstraint,
)
from pipelines.static_adapt.numerical_physical_integrity import (
    APPEND_NONWORSENING_ABSOLUTE_TOLERANCE,
    NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA,
    build_append_numerical_physical_integrity,
    numerical_physical_integrity_from_mapping,
    sector_probability,
)


def _problem() -> ResolvedProblemContext:
    request = ProblemRequest(
        problem_key="hh",
        num_sites=1,
        t=1.0,
        u=2.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=2,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
    )
    reference = np.zeros(8, dtype=complex)
    reference[1] = 1.0
    return ResolvedProblemContext(
        family_key="hh",
        request=request,
        layout=RegisterLayoutSpec(
            total_qubits=3,
            fermion_qubits=1,
            boson_qubits=2,
            ordering="blocked",
            boson_encoding="binary",
            blocks=(
                RegisterBlockSpec(
                    name="fermion",
                    kind="fermion",
                    start_qubit=0,
                    stop_qubit=1,
                ),
                RegisterBlockSpec(
                    name="boson",
                    kind="boson",
                    start_qubit=1,
                    stop_qubit=3,
                ),
            ),
        ),
        hamiltonian=object(),
        sector=SectorSelection(
            label="n_f=1, n_b<=2",
            comparison_space_label="test",
            constraints=(
                FixedCountConstraint(
                    quantity="n_f",
                    value=1,
                    scope="fermion_register",
                ),
                TruncationConstraint(
                    quantity="n_b",
                    max_local_occupancy=2,
                    scope="boson_register",
                ),
            ),
            num_particles=(1, 0),
        ),
        reference_state=ReferenceStateSpec(
            kind="test",
            source_label="test",
            state_kind="statevector",
            build_state=lambda: reference.copy(),
        ),
        exact_target=ExactTargetSpec(
            kind="test",
            comparison_space_label="test",
            resolve_energy=lambda **_kwargs: -1.0,
            exact_state_policy="test",
            build_fallback_anchor_state=lambda: reference.copy(),
            fallback_policy="test",
        ),
        default_controller_profile="test",
        default_continuation_mode="test",
        admissible_pool_keys=("hva",),
        default_pool_key="hva",
        default_pool_resolution_scope="test",
        default_sector_label="test",
        default_reference_label="test",
        exact_target_label="test",
        exact_comparison_space_label="test",
        default_num_particles=(1, 0),
        capabilities=HamiltonianFamilyCapabilities(),
    )


def _basis_state(index: int) -> np.ndarray:
    state = np.zeros(8, dtype=complex)
    state[index] = 1.0
    return state


def _history(*, energy_after: float = -1.25) -> list[dict[str, object]]:
    return [
        {
            "controller_round": 1,
            "energy_before": -1.0,
            "energy_after": energy_after,
            "accepted_refit": {
                "origin_logical_theta": [0.0],
                "origin_runtime_theta": [0.0],
                "final_logical_theta": [0.25],
                "final_runtime_theta": [0.25],
            },
        }
    ]


def test_shared_sector_diagnostic_distinguishes_fixed_and_boson_leaks() -> None:
    problem = _problem()
    good = sector_probability(problem, _basis_state(1))
    fixed_leak = sector_probability(problem, _basis_state(0))
    boson_leak = sector_probability(problem, _basis_state(7))

    assert good["sector_leak_flag"] is False
    assert good["boson_truncation_leak_flag"] is False
    assert good["fixed_count_sector_probability"] == pytest.approx(1.0)
    assert good["boson_legal_probability_min"] == pytest.approx(1.0)

    assert fixed_leak["sector_leak_flag"] is True
    assert fixed_leak["fixed_count_sector_probability"] == pytest.approx(0.0)
    assert fixed_leak["boson_truncation_leak_flag"] is False

    assert boson_leak["sector_leak_flag"] is True
    assert boson_leak["fixed_count_sector_probability"] == pytest.approx(1.0)
    assert boson_leak["boson_truncation_leak_flag"] is True
    assert boson_leak["boson_illegal_probability_max"] == pytest.approx(1.0)


def test_append_integrity_is_typed_reporting_only_and_passes_good_state() -> None:
    receipt = build_append_numerical_physical_integrity(
        problem=_problem(),
        final_state=_basis_state(1),
        history=_history(),
        logical_parameters=(0.25,),
        runtime_parameters=(0.25,),
        final_energy=-1.25,
    )

    assert receipt.schema == NUMERICAL_PHYSICAL_INTEGRITY_SCHEMA
    assert receipt.method == "append_adapt"
    assert receipt.reporting_only is True
    assert receipt.controller_decision_influence is False
    assert receipt.finite_values_passed is True
    assert receipt.sector_leak_flag is False
    assert receipt.boson_truncation_leak_flag is False
    assert receipt.accepted_energy_integrity_passed is True
    assert receipt.integrity_passed is True
    assert len(receipt.accepted_energy_transitions) == 1
    transition = receipt.accepted_energy_transitions[0]
    assert transition.absolute_tolerance == (
        APPEND_NONWORSENING_ABSOLUTE_TOLERANCE
    )
    assert transition.nonincrease_passed is True
    assert transition.typed_rollback_receipt is None
    assert transition.gate_passed is True
    assert numerical_physical_integrity_from_mapping(
        receipt.to_dict()
    ) == receipt


def test_append_integrity_reports_worsening_or_leak_without_controller_input() -> None:
    worsening = build_append_numerical_physical_integrity(
        problem=_problem(),
        final_state=_basis_state(1),
        history=_history(energy_after=-0.5),
        logical_parameters=(0.25,),
        runtime_parameters=(0.25,),
        final_energy=-0.5,
    )
    leaking = build_append_numerical_physical_integrity(
        problem=_problem(),
        final_state=_basis_state(7),
        history=_history(),
        logical_parameters=(0.25,),
        runtime_parameters=(0.25,),
        final_energy=-1.25,
    )

    assert worsening.accepted_energy_integrity_passed is False
    assert worsening.integrity_passed is False
    assert worsening.accepted_energy_transitions[0].gate_passed is False
    assert leaking.sector_leak_flag is True
    assert leaking.boson_truncation_leak_flag is True
    assert leaking.integrity_passed is False


def test_integrity_contract_rejects_a_contradictory_overall_flag() -> None:
    receipt = build_append_numerical_physical_integrity(
        problem=_problem(),
        final_state=_basis_state(1),
        history=_history(),
        logical_parameters=(0.25,),
        runtime_parameters=(0.25,),
        final_energy=-1.25,
    )

    with pytest.raises(ValueError, match="does not close"):
        replace(receipt, integrity_passed=False)
