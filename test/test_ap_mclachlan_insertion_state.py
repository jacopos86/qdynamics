"""Insertion-at-cut state materialization for the exchange selector.

The load-bearing identities: inserting zero-angle coordinates at any cut
leaves the prepared state unchanged, tail-cut insertion reproduces the append
result exactly, and multi-child parent blocks split into ordered children
without changing the implemented unitary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.state import (
    APMcLachlanStateParityError,
    state_from_scaffold_runtime_input,
    state_with_appended_runtime_coordinates,
    state_with_inserted_runtime_coordinates,
    state_without_runtime_indices,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


def _state(selected: tuple[AnsatzTerm, ...], theta: np.ndarray):
    layout = build_parameter_layout(selected)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(family_key="toy", hamiltonian=_poly(("ez", 2.0))),
        psi_ref=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        psi_initial=np.array([1.0, 0.0, 0.0, 0.0], dtype=complex),
        base_layout=layout,
        theta_runtime=np.zeros(0, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=selected,
        candidate_pool_terms=selected,
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json"},
    )
    # Recompute psi fields consistently for the requested theta.
    state = state_from_scaffold_runtime_input(
        _replace_runtime_theta(runtime_input, layout, theta)
    )
    return state


def _replace_runtime_theta(runtime_input, layout, theta):
    from dataclasses import replace as dc_replace

    from src.quantum.compiled_ansatz import CompiledAnsatzExecutor

    executor = CompiledAnsatzExecutor(
        tuple(runtime_input.selected_terms),
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_initial = executor.prepare_state(
        np.asarray(theta, dtype=float),
        np.asarray(runtime_input.psi_ref, dtype=complex).reshape(-1),
    )
    return dc_replace(
        runtime_input,
        theta_runtime=np.asarray(theta, dtype=float),
        psi_initial=np.asarray(psi_initial, dtype=complex),
    )


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
Y1 = AnsatzTerm(label="sy1", polynomial=_poly(("ye", 1.0)))
CAND = AnsatzTerm(label="cand", polynomial=_poly(("xe", 0.7)))
CAND2 = AnsatzTerm(label="cand2", polynomial=_poly(("ze", 0.4)))
MACRO = AnsatzTerm(label="macro", polynomial=_poly(("ex", 0.5), ("ye", 0.5)))


def test_insertion_at_internal_cut_orders_labels_and_preserves_state() -> None:
    state = _state((X0, Z0, Y1), np.array([0.3, -0.2, 0.5]))
    psi_before = state.prepare_state(state.theta_runtime)

    next_state, theta = state_with_inserted_runtime_coordinates(
        state,
        insertions=((1, CAND, "cand::r0::xe"),),
    )
    assert next_state.runtime_coordinate_labels == (
        "sx0::r0::ex",
        "cand::r0::xe",
        "sz0::r0::ez",
        "sy1::r0::ye",
    )
    assert theta.tolist() == [0.3, 0.0, -0.2, 0.5]
    psi_after = next_state.prepare_state(theta)
    assert np.allclose(psi_after, psi_before, atol=1.0e-12)


def test_tail_cut_insertion_equals_append() -> None:
    state = _state((X0, Z0), np.array([0.4, 0.1]))
    inserted, theta_ins = state_with_inserted_runtime_coordinates(
        state,
        insertions=((2, CAND, "cand::r0::xe"),),
    )
    appended, theta_app = state_with_appended_runtime_coordinates(
        state,
        (CAND,),
        coordinate_labels=("cand::r0::xe",),
    )
    assert inserted.runtime_coordinate_labels == appended.runtime_coordinate_labels
    assert theta_ins.tolist() == theta_app.tolist()
    assert np.allclose(
        inserted.prepare_state(theta_ins),
        appended.prepare_state(theta_app),
        atol=1.0e-12,
    )


def test_multiple_insertions_keep_given_order_within_and_across_cuts() -> None:
    state = _state((X0, Z0), np.array([0.2, -0.4]))
    next_state, theta = state_with_inserted_runtime_coordinates(
        state,
        insertions=(
            (0, CAND, "cand::r0::xe"),
            (0, CAND2, "cand2::r0::ze"),
            (2, AnsatzTerm(label="tail", polynomial=_poly(("xe", 1.0))), "tail::r0::xe"),
        ),
    )
    assert next_state.runtime_coordinate_labels == (
        "cand::r0::xe",
        "cand2::r0::ze",
        "sx0::r0::ex",
        "sz0::r0::ez",
        "tail::r0::xe",
    )
    assert theta.tolist() == [0.0, 0.0, 0.2, -0.4, 0.0]
    assert np.allclose(
        next_state.prepare_state(theta),
        state.prepare_state(state.theta_runtime),
        atol=1.0e-12,
    )


def test_insertion_inside_multi_child_parent_splits_it_exactly() -> None:
    state = _state((MACRO, Z0), np.array([0.3, 0.7, -0.1]))
    assert state.runtime_parameter_count == 3  # macro has two children
    psi_before = state.prepare_state(state.theta_runtime)

    # Cut 1 lands strictly inside the macro parent block.
    next_state, theta = state_with_inserted_runtime_coordinates(
        state,
        insertions=((1, CAND, "cand::r0::xe"),),
    )
    labels = next_state.runtime_coordinate_labels
    assert labels[0].startswith("macro::")
    assert labels[1] == "cand::r0::xe"
    assert labels[2].startswith("macro::")
    assert labels[3] == "sz0::r0::ez"
    assert theta.tolist() == [0.3, 0.0, 0.7, -0.1]
    assert np.allclose(next_state.prepare_state(theta), psi_before, atol=1.0e-12)


def test_delete_then_insert_composes_into_reduced_word_layout() -> None:
    state = _state((X0, Z0, Y1), np.array([0.3, -0.2, 0.5]))
    pruned, theta_pruned = state_without_runtime_indices(state, (1,))
    next_state, theta = state_with_inserted_runtime_coordinates(
        pruned,
        insertions=((1, CAND, "cand::r0::xe"),),
        theta_runtime=theta_pruned,
    )
    assert next_state.runtime_coordinate_labels == (
        "sx0::r0::ex",
        "cand::r0::xe",
        "sy1::r0::ye",
    )
    assert theta.tolist() == [0.3, 0.0, 0.5]
    assert np.allclose(
        next_state.prepare_state(theta),
        pruned.prepare_state(theta_pruned),
        atol=1.0e-12,
    )


def test_empty_insertions_is_a_validated_noop() -> None:
    state = _state((X0,), np.array([0.9]))
    same_state, theta = state_with_inserted_runtime_coordinates(state, insertions=())
    assert same_state is state
    assert theta.tolist() == [0.9]


def test_out_of_range_cut_is_rejected() -> None:
    state = _state((X0, Z0), np.array([0.0, 0.0]))
    with pytest.raises(ValueError, match=r"out of range \[0, 2\]"):
        state_with_inserted_runtime_coordinates(
            state, insertions=((3, CAND, "cand::r0::xe"),)
        )


def test_multi_child_inserted_term_is_rejected() -> None:
    state = _state((X0,), np.array([0.0]))
    with pytest.raises(ValueError, match="one Pauli child per inserted term"):
        state_with_inserted_runtime_coordinates(
            state, insertions=((0, MACRO, "macro::r0::ex"),)
        )


def test_label_collision_is_rejected() -> None:
    state = _state((X0,), np.array([0.0]))
    with pytest.raises(ValueError, match="collide"):
        state_with_inserted_runtime_coordinates(
            state, insertions=((0, CAND, "sx0::r0::ex"),)
        )
