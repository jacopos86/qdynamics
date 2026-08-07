from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pipelines.scaffold.runtime_contract import CandidatePoolSource
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
)
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    PRUNE_TARGET_APPENDED_ONLY,
    PRUNE_TARGET_REDUNDANT_APPENDED_ONLY,
    SupportPatchControllerConfig,
    _PruneControllerRuntimeState,
    _active_prune_atoms,
    _transport_exact_adjacent_duplicate_angles,
)
from pipelines.time_dynamics.ap_mclachlan.support_atoms import (
    active_support_atoms,
    appended_origin_atom_labels,
    state_without_active_atoms,
)
from pipelines.time_dynamics.normalized_pauli_pool import (
    NORMALIZED_POOL_HAMILTONIAN_DRIVE,
    NormalizedPauliPoolAtom,
    NormalizedPauliPoolContract,
)
from pipelines.time_dynamics.redundancy_stress import (
    REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES,
    RedundancyStressConfig,
    inject_zero_angle_redundancy_layers,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(len(pauli), ps=pauli, pc=1.0)],
        ),
    )


def _state() -> object:
    selected = (_term("seed_z", "ze"),)
    layout = build_parameter_layout(selected)
    psi_ref = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex)
    return state_from_scaffold_runtime_input(
        SimpleNamespace(
            selected_terms=selected,
            base_layout=layout,
            theta_runtime=np.asarray([0.0], dtype=float),
            theta_logical=np.asarray([0.0], dtype=float),
            psi_ref=psi_ref,
            psi_initial=psi_ref.copy(),
            h_poly=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ze", pc=1.0)],
            ),
            resolved_problem=SimpleNamespace(),
            exact_energy=None,
            candidate_pool_terms=(),
            candidate_pool_source=CandidatePoolSource(
                source_kind="resolved_pool",
                pool_key="unit",
                completeness="complete",
            ),
            provenance={},
            extensions={},
        )
    )


def _contract() -> NormalizedPauliPoolContract:
    atoms = (
        NormalizedPauliPoolAtom(
            pauli_exyz="ex",
            nq=2,
            repr_mode="JW",
            source_labels=("unit",),
        ),
        NormalizedPauliPoolAtom(
            pauli_exyz="xe",
            nq=2,
            repr_mode="JW",
            source_labels=("unit",),
        ),
    )
    return NormalizedPauliPoolContract(
        profile=NORMALIZED_POOL_HAMILTONIAN_DRIVE,
        atoms=atoms,
        source_occurrence_count=2,
        source_parent_count=1,
        untruncated_atom_count=2,
    )


def test_repeated_zero_angle_layers_preserve_state_and_duplicate_tangents() -> None:
    state = _state()
    result = inject_zero_angle_redundancy_layers(
        state,
        pool_contract=_contract(),
        config=RedundancyStressConfig(layer_count=2),
    )

    assert result.receipt["prepared_state_parity_passed"] is True
    assert result.receipt["appended_coordinate_count"] == 4
    assert result.state.runtime_parameter_count == state.runtime_parameter_count + 4
    np.testing.assert_allclose(result.state.prepare_state(), state.prepare_state())
    assert result.state.runtime_coordinate_labels[-4:] == (
        "normalized_pool::hamiltonian_drive_pauli::p0000::ex::r0::ex",
        "normalized_pool::hamiltonian_drive_pauli::p0001::xe::r0::xe",
        "normalized_pool::hamiltonian_drive_pauli::p0000::ex::ap_occ2::r0::ex",
        "normalized_pool::hamiltonian_drive_pauli::p0001::xe::ap_occ2::r0::xe",
    )
    assert appended_origin_atom_labels(result.state) == frozenset(
        result.state.runtime_coordinate_labels[-4:]
    )

    _, tangents = result.state.executor.prepare_state_with_runtime_tangents(
        result.state.theta_runtime,
        result.state.psi_ref,
    )
    offset = state.runtime_parameter_count
    np.testing.assert_allclose(tangents[offset], tangents[offset + 2], atol=1.0e-12)
    np.testing.assert_allclose(
        tangents[offset + 1],
        tangents[offset + 3],
        atol=1.0e-12,
    )


def test_redundancy_fixture_is_deterministic_across_comparison_arms() -> None:
    config = RedundancyStressConfig(layer_count=2)
    left = inject_zero_angle_redundancy_layers(
        _state(),
        pool_contract=_contract(),
        config=config,
    )
    right = inject_zero_angle_redundancy_layers(
        _state(),
        pool_contract=_contract(),
        config=config,
    )

    assert left.receipt["layers"] == right.receipt["layers"]
    assert (
        left.receipt["pool_ordered_atom_contract_sha256"]
        == right.receipt["pool_ordered_atom_contract_sha256"]
    )


def test_adjacent_duplicate_layout_groups_equal_pauli_coordinates() -> None:
    state = _state()
    result = inject_zero_angle_redundancy_layers(
        state,
        pool_contract=_contract(),
        config=RedundancyStressConfig(
            layer_count=2,
            layout_mode=REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES,
        ),
    )

    assert result.receipt["layout_mode"] == "adjacent_duplicates"
    assert result.state.runtime_coordinate_labels[-4:] == (
        "normalized_pool::hamiltonian_drive_pauli::p0000::ex::r0::ex",
        "normalized_pool::hamiltonian_drive_pauli::p0000::ex::ap_occ2::r0::ex",
        "normalized_pool::hamiltonian_drive_pauli::p0001::xe::r0::xe",
        "normalized_pool::hamiltonian_drive_pauli::p0001::xe::ap_occ2::r0::xe",
    )
    theta = np.asarray(result.state.theta_runtime, dtype=float).copy()
    theta[-4:] = np.asarray([0.11, -0.04, 0.07, 0.02], dtype=float)
    _, tangents = result.state.executor.prepare_state_with_runtime_tangents(
        theta,
        result.state.psi_ref,
    )
    offset = int(theta.size - 4)
    np.testing.assert_allclose(tangents[offset], tangents[offset + 1], atol=1.0e-12)
    np.testing.assert_allclose(
        tangents[offset + 2],
        tangents[offset + 3],
        atol=1.0e-12,
    )


def test_adjacent_duplicate_angle_transport_preserves_the_state_ray() -> None:
    result = inject_zero_angle_redundancy_layers(
        _state(),
        pool_contract=_contract(),
        config=RedundancyStressConfig(
            layer_count=2,
            layout_mode=REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES,
        ),
    )
    theta = np.asarray(result.state.theta_runtime, dtype=float).copy()
    theta[-4:] = np.asarray([0.11, -0.04, 0.07, 0.02], dtype=float)
    removed_index = int(theta.size - 4)
    removed_atom = next(
        atom
        for atom in active_support_atoms(result.state, theta)
        if tuple(atom.runtime_indices) == (removed_index,)
    )
    pruned_state, theta_zero = state_without_active_atoms(
        result.state,
        (removed_atom,),
        theta_runtime=theta,
    )
    theta_refit, metadata = _transport_exact_adjacent_duplicate_angles(
        result.state,
        theta_runtime=theta,
        removed_runtime_indices=(removed_index,),
        theta_patched=theta_zero,
    )

    assert metadata["prune_patch_refit_mode"] == "exact_adjacent_duplicate_transport"
    assert metadata["prune_patch_exact_duplicate_transfer_count"] == 1
    np.testing.assert_allclose(
        pruned_state.prepare_state(theta_refit),
        result.state.prepare_state(theta),
        atol=1.0e-12,
    )


def test_appended_only_prune_policy_protects_the_serialized_seed() -> None:
    result = inject_zero_angle_redundancy_layers(
        _state(),
        pool_contract=_contract(),
        config=RedundancyStressConfig(
            layer_count=2,
            layout_mode=REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES,
        ),
    )
    atoms = _active_prune_atoms(
        result.state,
        theta_runtime=result.state.theta_runtime,
        support_config=SupportPatchControllerConfig(
            prune_appended_origin_target_policy=PRUNE_TARGET_APPENDED_ONLY,
        ),
        runtime_state=_PruneControllerRuntimeState(),
        time_index=0,
    )

    assert {str(atom.atom_label) for atom in atoms} == set(
        result.state.runtime_coordinate_labels[-4:]
    )


def test_redundant_appended_policy_protects_last_family_representative() -> None:
    result = inject_zero_angle_redundancy_layers(
        _state(),
        pool_contract=_contract(),
        config=RedundancyStressConfig(
            layer_count=2,
            layout_mode=REDUNDANCY_STRESS_LAYOUT_ADJACENT_DUPLICATES,
        ),
    )
    active = active_support_atoms(result.state, result.state.theta_runtime)
    first_duplicate = next(
        atom
        for atom in active
        if tuple(atom.runtime_indices) == (result.state.runtime_parameter_count - 4,)
    )
    pruned_state, theta_pruned = state_without_active_atoms(
        result.state,
        (first_duplicate,),
        theta_runtime=result.state.theta_runtime,
    )
    atoms = _active_prune_atoms(
        pruned_state,
        theta_runtime=theta_pruned,
        support_config=SupportPatchControllerConfig(
            prune_appended_origin_target_policy=(
                PRUNE_TARGET_REDUNDANT_APPENDED_ONLY
            ),
        ),
        runtime_state=_PruneControllerRuntimeState(),
        time_index=0,
    )

    assert {str(atom.atom_label) for atom in atoms} == set(
        result.state.runtime_coordinate_labels[-2:]
    )


def test_disabled_redundancy_fixture_is_a_noop() -> None:
    state = _state()
    result = inject_zero_angle_redundancy_layers(
        state,
        pool_contract=None,
        config=RedundancyStressConfig(layer_count=0),
    )

    assert result.state is state
    assert result.receipt["applied"] is False
    assert result.receipt["appended_coordinate_count"] == 0
