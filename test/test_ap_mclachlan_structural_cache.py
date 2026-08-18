"""Structural cache parity: cached frozen-ray geometry vs direct materialization.

The authoritative check: every positioned tangent, assembled Gram block, and
candidate solve from the cache must equal what direct zero-angle insertion
materialization plus the ordinary geometry evaluator produce on the same ray.
Deletion branches must equal row selection of the directly evaluated
augmented geometry, which is exactly the frozen-ray column-removal semantics.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
    state_with_inserted_runtime_coordinates,
)
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    assemble_candidate_geometry,
    build_structural_insertion_cache,
    structural_candidate_solve,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*components: tuple[str, float], nq: int = 2) -> PauliPolynomial:
    poly = PauliPolynomial("JW")
    for word, coeff in components:
        poly.add_term(PauliTerm(int(nq), ps=str(word), pc=float(coeff)))
    poly._reduce()
    return poly


def _state(selected, theta):
    layout = build_parameter_layout(selected)
    executor = CompiledAnsatzExecutor(
        tuple(selected),
        parameterization_layout=layout,
        parameterization_mode="per_pauli_term",
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    psi_initial = executor.prepare_state(np.asarray(theta, dtype=float), psi_ref)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy", hamiltonian=_poly(("ez", 2.0), ("xx", 0.6))
        ),
        psi_ref=psi_ref,
        psi_initial=np.asarray(psi_initial, dtype=complex),
        base_layout=layout,
        theta_runtime=np.asarray(theta, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=tuple(selected),
        candidate_pool_terms=tuple(selected),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool",
            pool_key="toy_pool",
            completeness="complete",
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
Y1 = AnsatzTerm(label="sy1", polynomial=_poly(("ye", 1.0)))
CA = AnsatzTerm(label="ca", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="cb", polynomial=_poly(("zz", 0.4)))

HAM = TimeDependentHamiltonian(static_poly=_poly(("ez", 2.0), ("xx", 0.6)))

POLICY = McLachlanInversePolicy(
    pinv_rcond=1.0e-8, ridge_lambda=1.0e-6, solve_damping=1.0e-4
)


def _setup(theta=(0.3, -0.2, 0.5)):
    state = _state((X0, Z0, Y1), np.array(theta))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    atoms = {
        "ca": SimpleNamespace(atom_id="ca", term=CA),
        "cb": SimpleNamespace(atom_id="cb", term=CB),
    }
    cuts = {"ca": (0, 1, 3), "cb": (0, 2)}
    cache = build_structural_insertion_cache(
        state=state,
        evaluation=evaluation,
        cuts_by_atom=cuts,
        atoms_by_id=atoms,
        checkpoint_key=("toy", 0),
    )
    return state, evaluation, atoms, cuts, cache


def _direct_augmented(state, insertions):
    """Materialize insertions and evaluate full geometry on the same ray."""

    next_state, theta = state_with_inserted_runtime_coordinates(
        state, insertions=insertions
    )
    evaluation = evaluate_mclachlan_geometry(
        state=next_state,
        hamiltonian=HAM,
        theta_runtime=theta,
        time=0.0,
        include_tangent_matrix=True,
    )
    return next_state, evaluation


def test_every_positioned_tangent_matches_direct_materialization() -> None:
    state, _evaluation, atoms, cuts, cache = _setup()
    for atom_id, atom_cuts in cuts.items():
        term = atoms[atom_id].term
        for cut in atom_cuts:
            pauli = {"ca": "xe", "cb": "zz"}[atom_id]
            label = f"{atom_id}at{cut}::r0::{pauli}"
            next_state, direct_eval = _direct_augmented(
                state, ((cut, term, label),)
            )
            position = next_state.runtime_coordinate_labels.index(label)
            direct_column = np.asarray(
                direct_eval.tangent_matrix[:, position], dtype=complex
            )
            cached_column = cache.tangent_matrix[
                :, cache.column_index[(atom_id, cut)]
            ]
            assert np.allclose(cached_column, direct_column, atol=1.0e-12), (
                f"tangent mismatch for ({atom_id}, {cut}): "
                f"max={np.max(np.abs(cached_column - direct_column)):.3e}"
            )


def test_columns_are_horizontal_at_the_frozen_ray() -> None:
    _state_, evaluation, _atoms, _cuts, cache = _setup()
    psi = np.asarray(evaluation.psi, dtype=complex).reshape(-1)
    overlaps = np.abs(psi.conj() @ cache.tangent_matrix)
    assert float(np.max(overlaps)) < 1.0e-12


def test_assembled_geometry_matches_direct_augmented_subblocks() -> None:
    state, evaluation, atoms, _cuts, cache = _setup()
    plan = (("ca", 1), ("cb", 2))
    pauli = {"ca": "xe", "cb": "zz"}
    labels = {key: f"{key[0]}at{key[1]}::r0::{pauli[key[0]]}" for key in plan}
    insertions = tuple(
        (cut, atoms[atom_id].term, labels[(atom_id, cut)]) for atom_id, cut in plan
    )
    next_state, direct_eval = _direct_augmented(state, insertions)

    direct_labels = next_state.runtime_coordinate_labels
    survivor_labels = state.runtime_coordinate_labels
    perm = [direct_labels.index(lab) for lab in survivor_labels] + [
        direct_labels.index(labels[key]) for key in plan
    ]
    K_direct = np.asarray(direct_eval.geometry.K, dtype=float)
    f_direct = np.asarray(direct_eval.geometry.f, dtype=float).reshape(-1)

    G, f = assemble_candidate_geometry(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        keep_indices=range(state.runtime_parameter_count),
        inserted_selection=plan,
    )
    assert np.allclose(G, K_direct[np.ix_(perm, perm)], atol=1.0e-12)
    assert np.allclose(f, f_direct[perm], atol=1.0e-12)


def test_deletion_branch_equals_row_selection_of_augmented_geometry() -> None:
    state, evaluation, atoms, _cuts, cache = _setup()
    plan = (("ca", 1),)
    pauli = {"ca": "xe", "cb": "zz"}
    labels = {key: f"{key[0]}at{key[1]}::r0::{pauli[key[0]]}" for key in plan}
    insertions = tuple(
        (cut, atoms[atom_id].term, labels[(atom_id, cut)]) for atom_id, cut in plan
    )
    next_state, direct_eval = _direct_augmented(state, insertions)
    direct_labels = next_state.runtime_coordinate_labels
    survivor_labels = state.runtime_coordinate_labels

    keep = (0, 2)  # delete survivor index 1
    perm = [direct_labels.index(survivor_labels[i]) for i in keep] + [
        direct_labels.index(labels[key]) for key in plan
    ]
    K_direct = np.asarray(direct_eval.geometry.K, dtype=float)
    f_direct = np.asarray(direct_eval.geometry.f, dtype=float).reshape(-1)

    G, f = assemble_candidate_geometry(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        keep_indices=keep,
        inserted_selection=plan,
    )
    assert np.allclose(G, K_direct[np.ix_(perm, perm)], atol=1.0e-12)
    assert np.allclose(f, f_direct[perm], atol=1.0e-12)


def test_candidate_solve_matches_direct_solve_under_active_policy() -> None:
    state, evaluation, _atoms, _cuts, cache = _setup()
    plan = (("ca", 1), ("cb", 0))
    keep = (0, 2)
    G, f = assemble_candidate_geometry(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        keep_indices=keep,
        inserted_selection=plan,
    )
    direct = solve_theta_dot(G, f, policy=POLICY)
    norm_b_sq = float(evaluation.geometry.norm_b_sq)
    Q, q = structural_candidate_solve(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=norm_b_sq,
        keep_indices=keep,
        inserted_selection=plan,
        inverse_policy=POLICY,
        epsilon_norm=1.0e-12,
    )
    assert Q == pytest.approx(float(direct.captured_drift), rel=1.0e-12)
    assert q == pytest.approx(Q / (norm_b_sq + 1.0e-12), rel=1.0e-12)


def test_solve_memo_is_keyed_and_reused() -> None:
    state, evaluation, _atoms, _cuts, cache = _setup()
    kwargs = dict(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        keep_indices=(0, 1, 2),
        inserted_selection=(("ca", 1),),
        inverse_policy=POLICY,
        epsilon_norm=1.0e-12,
    )
    key = (("ca", 1), (0, 1, 2))
    first = structural_candidate_solve(memo_key=key, **kwargs)
    assert cache.solve_memo[key] == first
    cache.solve_memo[key] = (123.0, 456.0)  # poison to prove the memo is read
    assert structural_candidate_solve(memo_key=key, **kwargs) == (123.0, 456.0)


def test_empty_candidate_geometry_scores_zero() -> None:
    state, evaluation, _atoms, _cuts, cache = _setup()
    Q, q = structural_candidate_solve(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        keep_indices=(),
        inserted_selection=(),
        inverse_policy=POLICY,
        epsilon_norm=1.0e-12,
    )
    assert (Q, q) == (0.0, 0.0)


def test_bad_cut_and_multi_child_atom_are_rejected() -> None:
    state, evaluation, atoms, _cuts, _cache = _setup()
    with pytest.raises(ValueError, match=r"out of range"):
        build_structural_insertion_cache(
            state=state,
            evaluation=evaluation,
            cuts_by_atom={"ca": (7,)},
            atoms_by_id=atoms,
        )
    macro = SimpleNamespace(
        atom_id="macro",
        term=AnsatzTerm(label="macro", polynomial=_poly(("ex", 0.5), ("ye", 0.5))),
    )
    with pytest.raises(ValueError, match="one runtime Pauli child"):
        build_structural_insertion_cache(
            state=state,
            evaluation=evaluation,
            cuts_by_atom={"macro": (0,)},
            atoms_by_id={"macro": macro},
        )
