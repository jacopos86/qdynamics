"""AVQDS adaptive-append comparator on shared checkpoint geometry."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
    SupportPatchControllerConfig,
    _active_prune_atoms,
    _PruneControllerRuntimeState,
)
from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
    select_deletion_conditioned_patch,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
from types import SimpleNamespace

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
CA = AnsatzTerm(label="candidate_ca", polynomial=_poly(("xe", 0.7)))
CB = AnsatzTerm(label="candidate_cb", polynomial=_poly(("ye", 0.4)))
HAM = TimeDependentHamiltonian(static_poly=_poly(("ez", 2.0), ("xx", 0.6)))
POLICY = McLachlanInversePolicy(pinv_rcond=1.0e-10, ridge_lambda=1.0e-7)


def _state(theta=(0.05, -0.04)):
    selected = (X0, Z0)
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
        selected_terms=selected,
        candidate_pool_terms=(CA, CB),
        candidate_pool_source=CandidatePoolSource(
            source_kind="resolved_pool", pool_key="toy_pool", completeness="complete"
        ),
        provenance={"artifact_json": "toy.json"},
    )
    return state_from_scaffold_runtime_input(runtime_input)


def _run(config: SupportPatchControllerConfig, theta=(0.05, -0.04)):
    state = _state(theta)
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    return select_deletion_conditioned_patch(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        inverse_policy=POLICY,
        support_config=config,
        runtime_state=_PruneControllerRuntimeState(),
        time_index=3,
        active_prune_atoms=_active_prune_atoms,
    )


def _config(**overrides):
    base = dict(
        append_ladder_mode="combinatorial",
        residual_ratio_threshold=0.0,
        prune_ray_distance_tol=1.0,
        prune_patch_smoothness_eta_max=1.0e6,
        min_runtime_parameter_count=1,
        max_append_batch_size=1,
    )
    base.update(overrides)
    return SupportPatchControllerConfig(**base)


def _avqds_setup(theta=(0.05, -0.04)):
    state = _state(theta)
    evaluation = evaluate_mclachlan_geometry(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime,
        time=0.0, include_tangent_matrix=True,
    )
    atoms = {"ca": SimpleNamespace(atom_id="ca", atom_label="ca", term=CA),
             "cb": SimpleNamespace(atom_id="cb", atom_label="cb", term=CB)}
    label = lambda atom, cut, ordinal: f"{atom.atom_label}::avqds{cut}o{ordinal}::r0::" + (
        "xe" if atom.atom_id == "ca" else "ye")
    return state, evaluation, atoms, label


def test_l2_matches_norm_b_minus_captured_drift() -> None:
    from pipelines.time_dynamics.ap_mclachlan.avqds import mclachlan_distance_squared
    from pipelines.time_dynamics.ap_mclachlan.inverse import solve_theta_dot

    _s, ev, _a, _l = _avqds_setup()
    solve = solve_theta_dot(
        np.asarray(ev.geometry.K, dtype=float),
        np.asarray(ev.geometry.f, dtype=float).reshape(-1),
        policy=POLICY,
    )
    expected = float(ev.geometry.norm_b_sq) - float(solve.captured_drift)
    assert mclachlan_distance_squared(ev, inverse_policy=POLICY) == pytest.approx(
        max(0.0, expected), rel=1e-12
    )


def test_avqds_appends_nothing_when_threshold_is_loose() -> None:
    from pipelines.time_dynamics.ap_mclachlan.avqds import select_avqds_appends

    state, ev, atoms, label = _avqds_setup()
    decision = select_avqds_appends(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime, time=0.0,
        evaluation=ev, atoms_by_id=atoms, occurrence_label=label,
        inverse_policy=POLICY, l2_cut=1.0e9,
    )
    assert not decision.accepted
    assert decision.stop_reason == "below_threshold"
    assert decision.candidates_scored == 0


def test_avqds_appends_until_threshold_and_picks_max_reduction() -> None:
    from pipelines.time_dynamics.ap_mclachlan.avqds import (
        mclachlan_distance_squared,
        select_avqds_appends,
    )
    from pipelines.time_dynamics.ap_mclachlan.state import (
        state_with_inserted_runtime_coordinates,
    )

    state, ev, atoms, label = _avqds_setup()
    decision = select_avqds_appends(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime, time=0.0,
        evaluation=ev, atoms_by_id=atoms, occurrence_label=label,
        inverse_policy=POLICY, l2_cut=0.0, max_appends_per_checkpoint=1,
    )
    assert decision.accepted and len(decision.appended_atom_ids) == 1
    assert decision.l2_after < decision.l2_before
    # The appended atom is the one whose zero-angle addition minimizes L^2.
    cut = int(state.runtime_parameter_count)
    scores = {}
    for atom_id, atom in atoms.items():
        cand_state, cand_theta = state_with_inserted_runtime_coordinates(
            state, insertions=((cut, atom.term, label(atom, cut, 0)),),
            theta_runtime=state.theta_runtime,
        )
        cand_eval = evaluate_mclachlan_geometry(
            state=cand_state, hamiltonian=HAM, theta_runtime=cand_theta,
            time=0.0, include_tangent_matrix=True,
        )
        scores[atom_id] = mclachlan_distance_squared(cand_eval, inverse_policy=POLICY)
    assert decision.appended_atom_ids[0] == min(scores, key=scores.get)
    assert decision.l2_after == pytest.approx(min(scores.values()), rel=1e-12)


def test_avqds_respects_append_budget() -> None:
    from pipelines.time_dynamics.ap_mclachlan.avqds import select_avqds_appends

    state, ev, atoms, label = _avqds_setup()
    decision = select_avqds_appends(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime, time=0.0,
        evaluation=ev, atoms_by_id=atoms, occurrence_label=label,
        inverse_policy=POLICY, l2_cut=0.0, max_appends_per_checkpoint=2,
    )
    assert len(decision.appended_atom_ids) <= 2
    assert decision.state is not None
    assert decision.state.runtime_parameter_count == 2 + len(decision.appended_atom_ids)
