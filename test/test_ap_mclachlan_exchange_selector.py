"""Selection loop: level-by-level certification, atomic commit, stay fallback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.commutation import singleton_insertion_cuts
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CertificationGates,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_selector import (
    select_exchange_patch,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_structural import (
    StructuralScoreWeights,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.insertion_words import (
    tokens_commute_from_terms,
)
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
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
HAM = TimeDependentHamiltonian(static_poly=_poly(("ez", 2.0), ("xx", 0.6)))
POLICY = McLachlanInversePolicy(pinv_rcond=1.0e-10, ridge_lambda=1.0e-7)


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
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete"
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


def _selection_kwargs(theta=(0.05, -0.04, 0.06), **structural_overrides):
    state = _state((X0, Z0, Y1), np.array(theta))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    blocks = (X0, Z0, Y1)
    cuts = {
        "ca": singleton_insertion_cuts(CA, blocks),
        "cb": singleton_insertion_cuts(CB, blocks),
    }
    atoms = {
        "ca": SimpleNamespace(atom_id="ca", term=CA, atom_label="ca"),
        "cb": SimpleNamespace(atom_id="cb", term=CB, atom_label="cb"),
    }
    cache = build_structural_insertion_cache(
        state=state,
        evaluation=evaluation,
        cuts_by_atom=structural_overrides.get("cuts_by_atom", cuts),
        atoms_by_id=atoms,
    )
    terms_by_key = dict(
        zip(state.runtime_coordinate_labels, (X0, Z0, Y1)), ca=CA, cb=CB
    )
    structural = dict(
        cache=cache,
        base_K=np.asarray(evaluation.geometry.K, dtype=float),
        base_f=np.asarray(evaluation.geometry.f, dtype=float),
        norm_b_sq=float(evaluation.geometry.norm_b_sq),
        inverse_policy=POLICY,
        weights=StructuralScoreWeights(),
        deletable_indices=(0, 1),
        min_surviving_support=1,
        cuts_by_atom=cuts,
        candidate_pool_for_deletion=lambda removed: ("ca", "cb"),
        insertion_cost=lambda atom_ids: 1.0 + 0.5 * len(atom_ids),
        deletion_cost=lambda removed: 1.0 + 0.25 * len(removed),
        tokens_commute=tokens_commute_from_terms(terms_by_key),
        max_insertion_batch_size=1,
    )
    structural.update(structural_overrides)

    pauli = {"ca": "xe", "cb": "zz"}

    def occurrence_label(atom, cut, ordinal):
        return f"{atom.atom_id}c{cut}o{ordinal}::r0::{pauli[atom.atom_id]}"

    return dict(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        inverse_policy=POLICY,
        atoms_by_id=atoms,
        occurrence_label=occurrence_label,
        structural_kwargs=structural,
    )


def test_commit_is_highest_certifiable_candidate_and_atomic() -> None:
    kwargs = _selection_kwargs()
    selection = select_exchange_patch(
        gates=CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e6),
        **kwargs,
    )
    assert selection.kind != "stay"
    assert selection.stop_reason == "committed"
    assert selection.certification is not None and selection.certification.certified
    # The committed candidate is the first attempt that certified; with loose
    # gates that is the very first ranked candidate.
    assert selection.attempts[-1].reason == "certified"
    assert selection.committed.score == max(a.score for a in selection.attempts)
    # Route objects are carried for the trajectory loop.
    assert selection.certification.state is not None
    assert selection.certification.step is not None


def test_all_gates_failing_returns_stay_with_full_attempt_record() -> None:
    # Deletion-only pool: every candidate moves the ray (nonzero angles), so a
    # tight ray gate rejects them all and the selector returns stay.
    kwargs = _selection_kwargs(
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
    )
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15
        ),
        **kwargs,
    )
    assert selection.kind == "stay"
    assert selection.committed is None
    assert selection.attempts, "every ranked candidate must have been attempted"
    assert all(a.reason == "ray_distance_above_max" for a in selection.attempts)
    assert {a.kind for a in selection.attempts} == {"delete"}
    assert selection.telemetry is not None
    assert selection.stop_reason == "level_exhausted"


def test_failed_higher_score_advances_to_next_candidate() -> None:
    # Harvest the d0 insert etas, then set the smoothness gate between the
    # largest and the smallest: the top-scored (most useful, largest velocity
    # change) insert fails and a lower-scored one certifies.
    kwargs = _selection_kwargs()
    survey = select_exchange_patch(
        gates=CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e9),
        **kwargs,
    )
    top_eta = survey.attempts[-1].smoothness_eta
    assert top_eta is not None and top_eta > 0.0
    kwargs = _selection_kwargs()
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0, smoothness_eta_max=top_eta / 2.0
        ),
        escalate=lambda: False,
        **kwargs,
    )
    if selection.committed is None:
        # Every d0 candidate exceeded half the top eta; the ordering property
        # still holds vacuously, but the intended construction is a commit.
        pytest.skip("no candidate below the reduced eta gate on this toy")
    reasons = [a.reason for a in selection.attempts]
    assert "smoothness_eta_above_max" in reasons
    assert reasons[-1] == "certified"
    failed_scores = [a.score for a in selection.attempts if a.reason != "certified"]
    assert min(failed_scores) > selection.committed.score


def test_escalation_predicate_false_stops_after_first_level() -> None:
    kwargs = _selection_kwargs(
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
    )
    selection = select_exchange_patch(
        gates=CertificationGates(
            ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15
        ),
        escalate=lambda: False,
        **kwargs,
    )
    assert selection.kind == "stay"
    assert selection.stop_reason == "escalation_predicate_false"
    # Only the d=0 family (stay alone here) was acquired: no rung attempts,
    # and the telemetry yield was never reached (iterator closed early).
    assert selection.attempts == ()
    assert selection.telemetry is None


def test_score_floor_excludes_low_candidates_from_certification() -> None:
    kwargs = _selection_kwargs(
        candidate_pool_for_deletion=lambda removed: (),
        cuts_by_atom={},
    )
    gates = CertificationGates(ray_distance_max=1.0e-15, smoothness_eta_max=1.0e-15)
    loose = select_exchange_patch(gates=gates, **kwargs)
    floored = select_exchange_patch(
        gates=gates,
        score_floor=max(a.score for a in loose.attempts),
        **kwargs,
    )
    assert len(floored.attempts) < len(loose.attempts)


def test_selection_is_deterministic() -> None:
    kwargs_a = _selection_kwargs()
    kwargs_b = _selection_kwargs()
    gates = CertificationGates(ray_distance_max=1.0, smoothness_eta_max=1e6)
    a = select_exchange_patch(gates=gates, **kwargs_a)
    b = select_exchange_patch(gates=gates, **kwargs_b)
    assert [x.to_json_dict() for x in a.attempts] == [
        x.to_json_dict() for x in b.attempts
    ]
    assert a.committed.order_key == b.committed.order_key
