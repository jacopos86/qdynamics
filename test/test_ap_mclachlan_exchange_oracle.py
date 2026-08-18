"""Small-system brute-force oracle for the exchange selector.

The oracle rebuilds the complete structural search from first principles —
every hard-feasible deletion subset via itertools, every singleton insertion
cut via direct unitary-equivalence checks on materialized states, and every
candidate's captured drift via a dense augmented solve on a directly
materialized zero-angle patched state — and requires the selector stack to
reproduce the candidate set, the scores, and the final selection exactly.
"""

from __future__ import annotations

from itertools import combinations
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralScoreWeights,
    enumerate_structural_candidates,
)
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    tokens_commute_from_terms,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import (
    McLachlanInversePolicy,
    solve_theta_dot,
)
from pipelines.time_dynamics.ap_mclachlan.state import (
    state_from_scaffold_runtime_input,
    state_with_inserted_runtime_coordinates,
)
from pipelines.time_dynamics.ap_mclachlan.structural_cache import (
    build_structural_insertion_cache,
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


X0 = AnsatzTerm(label="sx0", polynomial=_poly(("ex", 1.0)))
Z0 = AnsatzTerm(label="sz0", polynomial=_poly(("ez", 1.0)))
Y1 = AnsatzTerm(label="sy1", polynomial=_poly(("ye", 1.0)))
CA = AnsatzTerm(label="ca", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="cb", polynomial=_poly(("zz", 0.4)))
BLOCKS = (X0, Z0, Y1)
POOL = {"ca": CA, "cb": CB}
PAULI = {"ca": "xe", "cb": "zz"}
HAM = TimeDependentHamiltonian(static_poly=_poly(("ez", 2.0), ("xx", 0.6)))
POLICY = McLachlanInversePolicy(
    pinv_rcond=1.0e-8, ridge_lambda=1.0e-6, solve_damping=1.0e-4
)
WEIGHTS = StructuralScoreWeights()
THETA = (0.3, -0.2, 0.5)
DELETABLE = (0, 1)
MIN_SURVIVORS = 1
EPS = WEIGHTS.epsilon_norm


def _state():
    layout = build_parameter_layout(BLOCKS)
    executor = CompiledAnsatzExecutor(
        BLOCKS, parameterization_layout=layout, parameterization_mode="per_pauli_term"
    )
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[0] = 1.0
    psi_initial = executor.prepare_state(np.asarray(THETA, dtype=float), psi_ref)
    runtime_input = ScaffoldRuntimeInput(
        resolved_problem=SimpleNamespace(
            family_key="toy", hamiltonian=_poly(("ez", 2.0), ("xx", 0.6))
        ),
        psi_ref=psi_ref,
        psi_initial=np.asarray(psi_initial, dtype=complex),
        base_layout=layout,
        theta_runtime=np.asarray(THETA, dtype=float),
        theta_logical=np.zeros(int(layout.logical_parameter_count), dtype=float),
        structure_locked=False,
        exact_energy=None,
        selected_terms=BLOCKS,
        candidate_pool_terms=BLOCKS,
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete"
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


def _oracle_cuts(state, atom_id):
    """Retained cuts by direct materialized-unitary equivalence.

    Cut p is redundant when inserting the candidate at p and at the class
    representative below it produce identical prepared states for EVERY
    survivor angle assignment probed — checked here on a random probe set,
    which is exact for these operators because equivalence is angle-uniform.
    """

    term = POOL[atom_id]
    rng = np.random.default_rng(11)
    probes = [np.asarray(THETA, dtype=float)] + [
        rng.uniform(-1.0, 1.0, size=3) for _ in range(4)
    ]
    retained = [0]
    for cut in range(1, len(BLOCKS) + 1):
        equivalent_to_previous_class = True
        for theta in probes:
            prev_state, prev_theta = state_with_inserted_runtime_coordinates(
                state,
                insertions=((retained[-1], term, f"probe::r0::{PAULI[atom_id]}"),),
                theta_runtime=theta,
            )
            cut_state, cut_theta = state_with_inserted_runtime_coordinates(
                state,
                insertions=((cut, term, f"probe::r0::{PAULI[atom_id]}"),),
                theta_runtime=theta,
            )
            # Compare implemented unitaries via action on a probe vector with a
            # NONZERO inserted angle (zero angle is trivially equal).
            probe_prev = prev_theta.copy()
            probe_cut = cut_theta.copy()
            idx_prev = list(prev_state.runtime_coordinate_labels).index(
                f"probe::r0::{PAULI[atom_id]}"
            )
            idx_cut = list(cut_state.runtime_coordinate_labels).index(
                f"probe::r0::{PAULI[atom_id]}"
            )
            probe_prev[idx_prev] = 0.37
            probe_cut[idx_cut] = 0.37
            psi_prev = prev_state.prepare_state(probe_prev)
            psi_cut = cut_state.prepare_state(probe_cut)
            if not np.allclose(psi_prev, psi_cut, atol=1.0e-12):
                equivalent_to_previous_class = False
                break
        if not equivalent_to_previous_class:
            retained.append(cut)
    return tuple(retained)


def _oracle_q(state, evaluation, removed, selection):
    """Captured drift from a directly materialized zero-angle patched state."""

    if not selection and len(removed) == len(BLOCKS):
        return 0.0
    insertions = tuple(
        (cut, POOL[a], f"{a}o{i}::r0::{PAULI[a]}")
        for i, (a, cut) in enumerate(selection)
    )
    aug_state, aug_theta = state_with_inserted_runtime_coordinates(
        state, insertions=insertions
    )
    aug_eval = evaluate_mclachlan_geometry(
        state=aug_state,
        hamiltonian=HAM,
        theta_runtime=aug_theta,
        time=0.0,
        include_tangent_matrix=True,
    )
    labels = list(aug_state.runtime_coordinate_labels)
    survivor_labels = [
        lab for i, lab in enumerate(state.runtime_coordinate_labels) if i not in set(removed)
    ]
    keep = [labels.index(lab) for lab in survivor_labels] + [
        labels.index(f"{a}o{i}::r0::{PAULI[a]}")
        for i, (a, _cut) in enumerate(selection)
    ]
    K = np.asarray(aug_eval.geometry.K, dtype=float)[np.ix_(keep, keep)]
    f = np.asarray(aug_eval.geometry.f, dtype=float)[keep]
    if f.size == 0:
        return 0.0
    solve = solve_theta_dot(K, f, policy=POLICY)
    return float(solve.captured_drift) / (
        float(aug_eval.geometry.norm_b_sq) + EPS
    )


def test_selector_stack_matches_full_brute_force_oracle() -> None:
    state = _state()
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )

    # --- oracle cut sets vs the commutation module ------------------------
    oracle_cuts = {a: _oracle_cuts(state, a) for a in POOL}
    module_cuts = {a: singleton_insertion_cuts(POOL[a], BLOCKS) for a in POOL}
    assert oracle_cuts == module_cuts

    # --- oracle candidate family ------------------------------------------
    insertion_cost = lambda atom_ids: 1.0 + 0.5 * len(atom_ids)
    deletion_cost = lambda removed: 1.0 + 0.25 * len(removed)

    oracle = {}
    deletions = [()] + [
        tuple(sorted(c))
        for d in range(1, len(DELETABLE) + 1)
        for c in combinations(DELETABLE, d)
        if len(BLOCKS) - d >= MIN_SURVIVORS
    ]
    q_cache = {}

    def q_of(removed, selection):
        key = (removed, selection)
        if key not in q_cache:
            q_cache[key] = _oracle_q(state, evaluation, removed, selection)
        return q_cache[key]

    q_base = q_of((), ())
    for removed in deletions:
        for selection in [()] + [
            ((a, c),) for a in sorted(POOL) for c in module_cuts[a]
        ]:
            if removed == () and selection == ():
                continue
            q = q_of(removed, selection)
            delta = q - q_base
            gain = max(0.0, q - q_of(removed, ())) if selection else 0.0
            loss = max(0.0, q_of((), selection) - q) if removed else 0.0
            u_ins = (
                gain / insertion_cost(tuple(a for a, _ in selection))
                if selection
                else 0.0
            )
            u_del = (
                deletion_cost(removed) / (loss + WEIGHTS.epsilon_L)
                if removed
                else 0.0
            )
            oracle[(removed, selection)] = u_ins + u_del + delta

    # --- selector stack ----------------------------------------------------
    atoms = {a: SimpleNamespace(atom_id=a, term=POOL[a]) for a in POOL}
    cache = build_structural_insertion_cache(
        state=state,
        evaluation=evaluation,
        cuts_by_atom=module_cuts,
        atoms_by_id=atoms,
    )
    terms_by_key = dict(
        zip(state.runtime_coordinate_labels, BLOCKS), **POOL
    )
    result = enumerate_structural_candidates(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        inverse_policy=POLICY,
        weights=WEIGHTS,
        deletable_indices=DELETABLE,
        min_surviving_support=MIN_SURVIVORS,
        cuts_by_atom=module_cuts,
        candidate_pool_for_deletion=lambda removed: tuple(sorted(POOL)),
        insertion_cost=insertion_cost,
        deletion_cost=deletion_cost,
        tokens_commute=tokens_commute_from_terms(terms_by_key),
        max_insertion_batch_size=1,
    )
    stack = {
        (c.removed_runtime_indices, c.inserted_selection): c.score
        for c in result.candidates
        if c.kind != "stay"
    }

    # Identical candidate sets and, per candidate, identical scores.
    assert set(stack) == set(oracle)
    for key in sorted(oracle):
        assert stack[key] == pytest.approx(oracle[key], rel=1.0e-9, abs=1.0e-12), key

    # Identical final selection under the frozen ordering.
    oracle_best = min(
        sorted(oracle),
        key=lambda k: (-oracle[k], k),
    )
    stack_best = result.ranked()[0]
    assert (
        stack_best.removed_runtime_indices,
        stack_best.inserted_selection,
    ) == oracle_best
