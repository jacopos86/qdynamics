from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import (
    TimeDependentHamiltonian,
    time_dependent_hamiltonian_from_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.drive_aligned import (
    augment_state_with_drive_aligned_generator,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
    runtime_indices_by_block_label,
    state_with_appended_terms,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(label: str, coeff: float, *, nq: int = 1) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _multi_poly(labels: tuple[tuple[str, float], ...], *, nq: int = 1) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for label, coeff in labels:
        poly.add_term(PauliTerm(int(nq), ps=str(label), pc=float(coeff)))
    poly._reduce()
    return poly


def _runtime_input() -> ScaffoldRuntimeInput:
    selected = (AnsatzTerm(label="seed_x", polynomial=_poly("x", 1.0)),)
    layout = build_parameter_layout(selected)
    return ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("z", 2.0)),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([np.cos(0.1), -1j * np.sin(0.1)], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.1], dtype=float),
        theta_logical=np.array([0.1], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(AnsatzTerm(label="candidate_y", polynomial=_poly("y", 1.0)),),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json", "seed_method": "snake"},
    )


class _DriveModel:
    family_key = "toy"
    operator_label = "toy_x_drive"
    drive_poly = _poly("x", 3.0)
    drive_term_count = 1
    profile_payload = {"shape": "linear"}

    def coefficient_at(self, physical_time: float) -> float:
        return 1.0 + float(physical_time)


class _ZeroDriveModel:
    family_key = "toy"
    operator_label = "toy_x_drive"
    drive_poly = _poly("x", 3.0)
    drive_term_count = 1
    drive_A = 0.0
    profile_payload = {"shape": "zero"}

    def coefficient_at(self, physical_time: float) -> float:
        return 0.0


def test_state_from_scaffold_runtime_input_preserves_neutral_seed_contract() -> None:
    state = state_from_scaffold_runtime_input(_runtime_input())

    assert state.logical_parameter_count == 1
    assert state.runtime_parameter_count == 1
    assert state.runtime_coordinate_labels == ("seed_x::r0::x",)
    assert state.can_structural_edit is True
    assert runtime_indices_by_block_label(state.layout) == {"seed_x": (0,)}
    np.testing.assert_allclose(state.prepare_state(), state.psi_initial)
    payload = state.to_json_dict()
    assert payload["selected_term_labels"] == ["seed_x"]
    assert payload["candidate_pool_complete"] is True
    assert payload["provenance"]["seed_method"] == "snake"


def test_state_from_scaffold_runtime_input_disambiguates_repeated_seed_labels() -> None:
    selected = (
        AnsatzTerm(label="repeat_x", polynomial=_poly("x", 1.0)),
        AnsatzTerm(label="repeat_x", polynomial=_poly("x", 1.0)),
    )
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("z", 1.0)),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([np.cos(0.3), -1j * np.sin(0.3)], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.1, 0.2], dtype=float),
        theta_logical=np.array([0.1, 0.2], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=(
            AnsatzTerm(label="candidate_y", polynomial=_poly("y", 1.0)),
        ),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "repeated.json", "seed_method": "snake"},
    )

    state = state_from_scaffold_runtime_input(runtime_input)
    appended = state_with_appended_terms(
        state,
        (AnsatzTerm(label="candidate_y", polynomial=_poly("y", 1.0)),),
    )

    assert state.runtime_coordinate_labels == (
        "repeat_x::r0::x",
        "repeat_x::ap_occurrence[2]::r0::x",
    )
    assert len(set(appended.runtime_coordinate_labels)) == 3
    np.testing.assert_allclose(state.prepare_state(), state.psi_initial)
    relabeling = state.extensions["selected_term_label_disambiguation"]["relabeling"]
    assert relabeling == [
        {
            "logical_index": 1,
            "source_label": "repeat_x",
            "resolved_label": "repeat_x::ap_occurrence[2]",
            "occurrence": 2,
        }
    ]


def test_state_from_scaffold_runtime_input_supports_logical_shared_mode() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_multi_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    layout = build_parameter_layout(selected)
    theta_logical = np.array([0.1], dtype=float)
    psi_ref = np.array([1.0, 0.0], dtype=complex)
    executor_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x", 1.0)),
        psi_ref=psi_ref,
        psi_initial=np.array(
            [
                np.exp(-0.05j) * np.cos(0.1),
                -1j * np.exp(0.05j) * np.sin(0.1),
            ],
            dtype=complex,
        ),
        base_layout=layout,
        theta_runtime=np.array([0.1, 0.1], dtype=float),
        theta_logical=theta_logical,
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "toy.json", "seed_method": "snake"},
    )

    state = state_from_scaffold_runtime_input(
        executor_input,
        parameterization_mode="logical_shared",
    )

    assert state.parameterization_mode == "logical_shared"
    assert state.parameterization_label == "per logical / macro generator"
    assert state.logical_parameter_count == 1
    assert state.runtime_parameter_count == 1
    assert state.runtime_pauli_parameter_count == 2
    assert state.runtime_coordinate_labels == ("seed_xz::logical::generator",)
    assert state.runtime_pauli_coordinate_labels == ("seed_xz::r0::x", "seed_xz::r1::z")
    np.testing.assert_allclose(state.prepare_state(), state.psi_initial)
    payload = state.to_json_dict()
    assert payload["parameterization_mode"] == "logical_shared"
    assert payload["parameterization_label"] == "per logical / macro generator"
    assert payload["active_parameter_count"] == 1
    assert payload["runtime_pauli_parameter_count"] == 2


def test_per_pauli_ap_adapts_grouped_exact_seed_execution_when_state_parity_holds() -> None:
    selected = (
        AnsatzTerm(
            label="commuting_group",
            polynomial=_multi_poly((("xe", 1.0), ("ex", 0.5)), nq=2),
            execution_mode="grouped_exact",
        ),
    )
    layout = build_parameter_layout(selected)
    psi_ref = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    grouped_executor = CompiledAnsatzExecutor(
        selected,
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    )
    psi_initial = grouped_executor.prepare_state(np.array([0.1]), psi_ref)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy",
            hamiltonian=_poly("ze", 1.0, nq=2),
        ),
        psi_ref=psi_ref,
        psi_initial=psi_initial,
        base_layout=layout,
        theta_runtime=np.array([0.1, 0.1], dtype=float),
        theta_logical=np.array([0.1], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "grouped.json", "seed_method": "snake"},
    )

    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode="per_pauli_term",
    )

    assert state.terms[0].execution_mode == "termwise_product"
    assert state.layout.blocks[0].execution_mode == "termwise_product"
    np.testing.assert_allclose(state.prepare_state(), psi_initial, atol=1.0e-12)
    adaptation = state.extensions["selected_term_execution_mode_adaptation"]
    assert adaptation["adaptations"] == [
        {
            "logical_index": 0,
            "label": "commuting_group",
            "source_execution_mode": "grouped_exact",
            "ap_execution_mode": "termwise_product",
        }
    ]


def test_per_pauli_ap_rejects_grouped_exact_seed_when_state_parity_fails() -> None:
    selected = (
        AnsatzTerm(
            label="noncommuting_group",
            polynomial=_multi_poly((("x", 1.0), ("z", 0.5))),
            execution_mode="grouped_exact",
        ),
    )
    layout = build_parameter_layout(selected)
    psi_ref = np.array([1.0, 0.0], dtype=complex)
    grouped_executor = CompiledAnsatzExecutor(
        selected,
        parameterization_mode="logical_shared",
        parameterization_layout=layout,
    )
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("z", 1.0)),
        psi_ref=psi_ref,
        psi_initial=grouped_executor.prepare_state(np.array([0.2]), psi_ref),
        base_layout=layout,
        theta_runtime=np.array([0.2, 0.2], dtype=float),
        theta_logical=np.array([0.2], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "grouped.json", "seed_method": "snake"},
    )

    with pytest.raises(ValueError, match="Prepared-state parity check failed"):
        state_from_scaffold_runtime_input(
            runtime_input,
            parameterization_mode="per_pauli_term",
        )


def test_logical_shared_append_adds_one_macro_generator_coordinate() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_multi_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x", 1.0)),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.0, 0.0], dtype=float),
        theta_logical=np.array([0.0], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "toy.json", "seed_method": "snake"},
    )
    state = state_from_scaffold_runtime_input(
        runtime_input,
        parameterization_mode="logical_shared",
    )
    appended = state_with_appended_terms(
        state,
        (
            AnsatzTerm(
                label="candidate_yz",
                polynomial=_multi_poly((("y", 1.0), ("z", -0.25))),
            ),
        ),
    )

    assert appended.parameterization_mode == "logical_shared"
    assert appended.runtime_parameter_count == 2
    assert appended.logical_parameter_count == 2
    assert appended.runtime_pauli_parameter_count == 4
    assert appended.runtime_coordinate_labels == (
        "seed_xz::logical::generator",
        "candidate_yz::logical::generator",
    )
    assert appended.theta_runtime.tolist() == [0.0, 0.0]


def test_logical_shared_mode_rejects_incompatible_handoff_state() -> None:
    selected = (
        AnsatzTerm(
            label="seed_xz",
            polynomial=_multi_poly((("x", 1.0), ("z", 0.5))),
        ),
    )
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("x", 1.0)),
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.array([0.1, -0.2], dtype=float),
        theta_logical=np.array([0.1], dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        provenance={"artifact_json": "toy.json", "seed_method": "snake"},
    )

    with pytest.raises(ValueError, match="Prepared-state parity check failed"):
        state_from_scaffold_runtime_input(
            runtime_input,
            parameterization_mode="logical_shared",
        )


def test_state_rejects_theta_layout_mismatch() -> None:
    runtime_input = _runtime_input()
    bad = ScaffoldRuntimeInput(
        resolved_problem=runtime_input.resolved_problem,
        psi_ref=runtime_input.psi_ref,
        psi_initial=runtime_input.psi_initial,
        base_layout=runtime_input.base_layout,
        theta_runtime=np.array([0.1, 0.2]),
        theta_logical=runtime_input.theta_logical,
        structure_locked=runtime_input.structure_locked,
        exact_energy=runtime_input.exact_energy,
        selected_terms=runtime_input.selected_terms,
        candidate_pool_terms=runtime_input.candidate_pool_terms,
        candidate_pool_source=runtime_input.candidate_pool_source,
        provenance=runtime_input.provenance,
    )

    with pytest.raises(ValueError, match="theta_runtime length mismatch"):
        state_from_scaffold_runtime_input(bad)


def test_time_dependent_hamiltonian_combines_static_and_drive_neutrally() -> None:
    provider = TimeDependentHamiltonian(static_poly=_poly("z", 2.0), drive_model=_DriveModel())

    mat = provider.matrix_at(1.0)

    expected_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    expected_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    np.testing.assert_allclose(mat, 2.0 * expected_z + 6.0 * expected_x)
    assert provider.drive_coefficient_at(1.0) == pytest.approx(2.0)
    assert provider.to_json_dict()["drive_operator_label"] == "toy_x_drive"


def test_provider_from_runtime_input_accepts_provided_drive_model_without_hh_builder() -> None:
    runtime_input = _runtime_input()
    provider = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_model=_DriveModel(),
    )

    assert provider.drive_enabled is True
    assert provider.to_json_dict()["drive_family_key"] == "toy"
    assert provider.to_json_dict()["metadata"]["drive_source"] == "provided_drive_model"
    assert provider.to_json_dict()["metadata"]["static_hamiltonian_parity"]["passed"] is True


def test_drive_aligned_augmentation_adds_zero_angle_tangent_without_changing_state() -> None:
    runtime_input = _runtime_input()
    state = state_from_scaffold_runtime_input(runtime_input)
    provider = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_model=_DriveModel(),
    )

    result = augment_state_with_drive_aligned_generator(state, hamiltonian=provider)

    assert result.applied is True
    assert result.logical_delta == 1
    assert result.runtime_delta == 1
    assert result.label == "drive_aligned_operator(operator=toy_x_drive,pattern=none)"
    assert result.state.logical_parameter_count == state.logical_parameter_count + 1
    assert result.state.runtime_parameter_count == state.runtime_parameter_count + 1
    np.testing.assert_allclose(result.state.prepare_state(), state.prepare_state())


def test_static_hamiltonian_parity_gate_rejects_mismatched_seed_hamiltonian() -> None:
    runtime_input = SimpleNamespace(
        h_poly=_poly("x", 1.0),
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly("z", 1.0)),
        provenance={},
    )

    with pytest.raises(ValueError, match="Static Hamiltonian parity check failed"):
        time_dependent_hamiltonian_from_runtime_input(runtime_input)


def test_zero_drive_parity_records_static_reduction() -> None:
    runtime_input = _runtime_input()
    provider = time_dependent_hamiltonian_from_runtime_input(
        runtime_input,
        drive_model=_ZeroDriveModel(),
    )

    payload = provider.to_json_dict()
    assert payload["drive_enabled"] is True
    assert payload["metadata"]["zero_drive_static_parity"]["passed"] is True


# ---------------------------------------------------------------------------
# Dense-operator cache parity (runtime acceleration pass)
# ---------------------------------------------------------------------------

#: matrix_at() recombines two cached dense operators instead of rebuilding the
#: combined Pauli polynomial, so agreement is to floating point, not bitwise.
DENSE_CACHE_ATOL = 1.0e-11
DENSE_CACHE_RTOL = 1.0e-12


def _uncached_matrix_at(provider, time):
    """The pre-acceleration construction, kept as the parity reference."""

    from src.quantum.vqe_latex_python_pairs import hamiltonian_matrix

    return np.asarray(hamiltonian_matrix(provider.polynomial_at(float(time))), dtype=complex)


def _drive_provider():
    """Driven provider whose static and drive operators share a Pauli label.

    The shared ``x`` label is the case where recombination order matters: the
    rebuild adds the coefficients before the dense mapping, the cache adds two
    dense matrices.
    """

    return TimeDependentHamiltonian(
        static_poly=_multi_poly((("z", 2.0), ("x", 0.5))),
        drive_model=_DriveModel(),
    )


def test_dense_operator_cache_matches_polynomial_rebuild_across_times():
    provider = _drive_provider()
    for time in (0.0, 0.017, 0.5, 1.25, 2.0, 3.7, 12.5, 50.0):
        cached = provider.matrix_at(time)
        reference = _uncached_matrix_at(provider, time)
        assert cached.shape == reference.shape
        assert np.allclose(
            cached, reference, atol=DENSE_CACHE_ATOL, rtol=DENSE_CACHE_RTOL
        ), f"dense cache diverged at t={time}: max={np.max(np.abs(cached - reference)):.3e}"


def test_dense_operator_cache_is_hermitian_and_repeatable():
    provider = _drive_provider()
    first = provider.matrix_at(0.9)
    second = provider.matrix_at(0.9)
    assert np.array_equal(first, second)
    assert np.allclose(first, first.conj().T, atol=DENSE_CACHE_ATOL)


def test_matrix_at_returns_a_fresh_writable_array_each_call():
    """Callers historically owned the returned array; mutation must not leak."""

    provider = _drive_provider()
    first = provider.matrix_at(0.4)
    assert first.flags.writeable
    first[0, 0] += 12.5
    assert not np.allclose(provider.matrix_at(0.4), first)


def test_zero_drive_coefficient_returns_the_static_matrix_exactly():
    """The A=0 parity requirement must stay exact, not merely close."""

    provider = TimeDependentHamiltonian(
        static_poly=_multi_poly((("z", 2.0), ("x", 0.5))),
        drive_model=_ZeroDriveModel(),
    )
    static_reference = np.asarray(
        _uncached_matrix_at(
            TimeDependentHamiltonian(static_poly=_multi_poly((("z", 2.0), ("x", 0.5)))),
            0.0,
        ),
        dtype=complex,
    )
    for time in (0.0, 1.0, 7.5):
        assert np.array_equal(provider.matrix_at(time), static_reference)


def test_undriven_provider_matches_the_static_rebuild_exactly():
    provider = TimeDependentHamiltonian(static_poly=_multi_poly((("z", 2.0), ("x", 0.5))))
    for time in (0.0, 2.5, 40.0):
        assert np.array_equal(provider.matrix_at(time), _uncached_matrix_at(provider, time))
