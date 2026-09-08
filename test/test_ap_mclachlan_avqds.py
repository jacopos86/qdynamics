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
    select_generalized_exchange_patch,
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
    return select_generalized_exchange_patch(
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
    )


def _config(**overrides):
    base = dict(
        residual_ratio_threshold=0.0,
        prune_ray_distance_tol=1.0,
        prune_patch_smoothness_eta_max=1.0e6,
        min_runtime_parameter_count=1,
        max_insertion_batch_size=1,
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
    # Yao et al. Eq. (8): L^2 = 2 var[H] - V M^-1 V, so the factor of two is
    # part of the published convention and a threshold quoted for AVQDS is in
    # these units.
    expected = 2.0 * (float(ev.geometry.norm_b_sq) - float(solve.captured_drift))
    assert mclachlan_distance_squared(ev, inverse_policy=POLICY) == pytest.approx(
        max(0.0, expected), rel=1e-12
    )


def test_avqds_decision_preserves_existing_positional_field_order() -> None:
    from pipelines.time_dynamics.ap_mclachlan.avqds import AVQDSDecision

    state_marker = object()
    decision = AVQDSDecision((), 0.0, 0.0, 0, "test", state_marker, None, None)

    assert decision.state is state_marker
    assert decision.candidate_geometry_rounds == ()


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
    assert decision.candidate_geometry_rounds == ()
    work = decision.to_json_dict()["estimator_work"]
    assert work == {
        "schema": "paper_ii_avqds_estimator_work_v1",
        "selector_round_count": 0,
        "full_geometry_evaluation_attempt_count": 0,
        "full_geometry_evaluation_count": 0,
        "logical_candidate_geometry_entries": 0,
        "complete": True,
    }


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
    assert len(decision.candidate_geometry_rounds) == 1
    round_work = decision.candidate_geometry_rounds[0]
    assert round_work["base_runtime_parameter_count"] == 2
    assert round_work["candidate_runtime_parameter_count"] == 3
    assert round_work["candidate_pool_count"] == 2
    assert round_work["geometry_evaluation_attempt_count"] == 2
    assert round_work["geometry_evaluation_count"] == 2
    assert round_work["logical_geometry_entry_count"] == 22
    assert round_work["committed"] is True
    assert round_work["complete"] is True
    work = decision.to_json_dict()["estimator_work"]
    assert work["full_geometry_evaluation_count"] == decision.candidates_scored == 2
    assert work["logical_candidate_geometry_entries"] == 22
    assert work["complete"] is True


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
    assert sum(
        int(row["geometry_evaluation_count"])
        for row in decision.candidate_geometry_rounds
    ) == decision.candidates_scored
    assert [
        int(row["candidate_runtime_parameter_count"])
        for row in decision.candidate_geometry_rounds
    ] == [
        3 + index for index in range(len(decision.candidate_geometry_rounds))
    ]


def test_avqds_records_failed_candidate_geometry_attempt(monkeypatch) -> None:
    from pipelines.time_dynamics.ap_mclachlan import avqds as avqds_module

    state, ev, atoms, label = _avqds_setup()
    real_evaluator = avqds_module.evaluate_mclachlan_geometry
    calls = 0

    def fail_first_candidate(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic candidate failure")
        return real_evaluator(**kwargs)

    monkeypatch.setattr(
        avqds_module, "evaluate_mclachlan_geometry", fail_first_candidate
    )
    decision = avqds_module.select_avqds_appends(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        evaluation=ev,
        atoms_by_id=atoms,
        occurrence_label=label,
        inverse_policy=POLICY,
        l2_cut=0.0,
        max_appends_per_checkpoint=1,
    )

    round_work = decision.candidate_geometry_rounds[0]
    assert round_work["geometry_evaluation_attempt_count"] == 2
    assert round_work["geometry_evaluation_count"] == 1
    assert round_work["logical_geometry_entry_count"] == 11
    assert round_work["complete"] is False
    assert decision.candidates_scored == 1
    assert decision.to_json_dict()["estimator_work"]["complete"] is False


def test_avqds_distinguishes_materialization_failure_from_estimator_work(
    monkeypatch,
) -> None:
    from pipelines.time_dynamics.ap_mclachlan import avqds as avqds_module

    state, ev, atoms, label = _avqds_setup()
    real_materializer = avqds_module.state_with_inserted_runtime_coordinates
    calls = 0

    def fail_first_materialization(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("synthetic materialization failure")
        return real_materializer(*args, **kwargs)

    monkeypatch.setattr(
        avqds_module,
        "state_with_inserted_runtime_coordinates",
        fail_first_materialization,
    )
    decision = avqds_module.select_avqds_appends(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        evaluation=ev,
        atoms_by_id=atoms,
        occurrence_label=label,
        inverse_policy=POLICY,
        l2_cut=0.0,
        max_appends_per_checkpoint=1,
    )

    round_work = decision.candidate_geometry_rounds[0]
    assert round_work["candidate_pool_count"] == 2
    assert round_work["materialization_failure_count"] == 1
    assert round_work["geometry_evaluation_attempt_count"] == 1
    assert round_work["geometry_evaluation_count"] == 1
    assert round_work["logical_geometry_entry_count"] == 11
    assert round_work["complete"] is True


def test_comparator_and_exchange_share_the_complete_pool() -> None:
    """Both structural policies retain every deduplicated candidate direction."""
    from dataclasses import replace
    from itertools import product
    from pipelines.time_dynamics.ap_mclachlan.exchange_integration import (
        build_candidate_pool,
        build_selector_inputs,
    )
    from pipelines.time_dynamics.ap_mclachlan.adaptive_trajectory import (
        _active_prune_atoms,
        _PruneControllerRuntimeState,
    )

    words = tuple(
        "".join(chars) for chars in product("exyz", repeat=2)
        if chars != ("e", "e")
    )
    terms = tuple(
        AnsatzTerm(label=f"candidate_{index}", polynomial=_poly((word, 1.0)))
        for index, word in enumerate(words)
    )
    duplicate = AnsatzTerm(label="duplicate", polynomial=_poly((words[0], 2.0)))
    state = replace(_state(), candidate_pool_terms=(*terms, duplicate))
    evaluation = evaluate_mclachlan_geometry(
        state=state, hamiltonian=HAM, theta_runtime=state.theta_runtime,
        time=0.0, include_tangent_matrix=True,
    )
    config = _config()

    pool_atoms, _pauli, telemetry = build_candidate_pool(
        state, support_config=config, insertions_enabled=True
    )
    selector_inputs = build_selector_inputs(
        state=state, evaluation=evaluation, support_config=config,
        runtime_state=_PruneControllerRuntimeState(),
        theta_runtime=state.theta_runtime, time_index=0,
        deletable_atoms=_active_prune_atoms(
            state,
            theta_runtime=state.theta_runtime,
            support_config=config,
            runtime_state=_PruneControllerRuntimeState(),
            time_index=0,
        ),
    )

    shared_ids = [str(a.atom_id) for a in pool_atoms]
    selector_ids = list(selector_inputs["atoms_by_id"])
    assert shared_ids == selector_ids, "selector must use the shared pool order"
    assert telemetry["candidate_pool_order"] == "frozen_pool_order"
    assert len(shared_ids) == len(words) > 8
    assert list(_pauli.values()) == list(words)
    assert telemetry["candidate_pool_raw"] == len(words) + 1
    assert telemetry["candidate_pool_deduplicated"] == len(words)
