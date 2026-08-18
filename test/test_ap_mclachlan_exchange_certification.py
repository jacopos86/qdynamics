"""Finalist certification: materialization, remapping, and hard gates."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.contracts.scaffold import CandidatePoolSource, ScaffoldRuntimeInput
from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
    CertificationGates,
    certify_finalist,
    remap_cuts_after_deletion,
)
from pipelines.time_dynamics.ap_mclachlan.fixed_step import solve_fixed_mclachlan_step
from pipelines.time_dynamics.ap_mclachlan.geometry_eval import (
    evaluate_mclachlan_geometry,
)
from pipelines.time_dynamics.ap_mclachlan.hamiltonian import TimeDependentHamiltonian
from pipelines.time_dynamics.ap_mclachlan.inverse import McLachlanInversePolicy
from pipelines.time_dynamics.ap_mclachlan.state import state_from_scaffold_runtime_input
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
CAND = AnsatzTerm(label="cand", polynomial=_poly(("xe", 0.7)))
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


def _setup(theta=(0.05, -0.04, 0.06)):
    state = _state((X0, Z0, Y1), np.array(theta))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    return state, evaluation, step


def test_remap_cuts_after_deletion() -> None:
    assert remap_cuts_after_deletion((0, 1, 2, 3), (1,)) == (0, 1, 1, 2)
    assert remap_cuts_after_deletion((3,), (0, 2)) == (1,)
    assert remap_cuts_after_deletion((2,), ()) == (2,)
    assert remap_cuts_after_deletion((), (0,)) == ()


def test_exchange_finalist_certifies_and_carries_patched_route_objects() -> None:
    state, evaluation, step = _setup()
    result = certify_finalist(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(1,),
        insertions=((2, CAND, "cand::r0::xe"),),
        inverse_policy=POLICY,
        gates=CertificationGates(ray_distance_max=0.5, smoothness_eta_max=10.0),
    )
    assert result.certified, result.reason
    # Deleted sz0 (small angle) and inserted cand at original cut 2 -> after
    # remapping, position 1 of the survivor word (sx0, sy1).
    assert result.state.runtime_coordinate_labels == (
        "sx0::r0::ex",
        "cand::r0::xe",
        "sy1::r0::ye",
    )
    assert result.theta.tolist() == [0.05, 0.0, 0.06]
    assert result.evaluation is not None and result.step is not None
    assert result.ray_distance is not None and result.smoothness_eta is not None
    # Ray displacement comes only from the deleted angle (insertion is zero
    # angle), so it is small but nonzero for theta_1 != 0.
    assert 0.0 < result.ray_distance < 0.1


def test_ray_gate_rejects_large_deletions_and_preserves_atomicity() -> None:
    state, evaluation, step = _setup(theta=(0.05, 1.2, 0.06))
    result = certify_finalist(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(1,),  # deleting a large angle moves the ray
        insertions=((2, CAND, "cand::r0::xe"),),
        inverse_policy=POLICY,
        gates=CertificationGates(ray_distance_max=1.0e-3, smoothness_eta_max=10.0),
    )
    assert not result.certified
    assert result.reason == "ray_distance_above_max"
    assert result.state is None and result.step is None  # nothing committed


def test_smoothness_gate_is_hard_and_reports_eta() -> None:
    state, evaluation, step = _setup(theta=(0.05, 0.8, 0.06))
    result = certify_finalist(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(1,),
        insertions=(),
        inverse_policy=POLICY,
        gates=CertificationGates(
            ray_distance_max=1.0, smoothness_eta_max=1.0e-12
        ),
    )
    assert not result.certified
    assert result.reason == "smoothness_eta_above_max"
    assert result.smoothness_eta is not None and result.smoothness_eta > 1.0e-12


def test_pure_zero_angle_insertion_passes_trivially() -> None:
    state, evaluation, step = _setup()
    result = certify_finalist(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(),
        insertions=((0, CAND, "cand::r0::xe"),),
        inverse_policy=POLICY,
        gates=CertificationGates(
            ray_distance_max=1.0e-9, smoothness_eta_max=100.0
        ),
    )
    # Zero-angle insertion leaves the ray untouched, so even a 1e-9 ray gate
    # passes; the velocity DOES change (the inserted direction captures more
    # drift), which is why the smoothness gate is a separate authority.
    assert result.certified, result.reason
    assert result.ray_distance == pytest.approx(0.0, abs=1.0e-7)
    assert result.smoothness_eta is not None and result.smoothness_eta > 0.0


def test_refit_hook_is_applied_before_gating() -> None:
    state, evaluation, step = _setup()
    calls = []

    def refit(patched_state, patched_theta):
        calls.append(True)
        return patched_state, patched_theta

    result = certify_finalist(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(),
        insertions=((1, CAND, "cand::r0::xe"),),
        inverse_policy=POLICY,
        gates=CertificationGates(ray_distance_max=0.5, smoothness_eta_max=10.0),
        refit=refit,
    )
    assert calls == [True]
    assert result.certified


def test_materialization_failure_is_a_reported_rejection() -> None:
    state, evaluation, step = _setup()
    result = certify_finalist(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(),
        insertions=((99, CAND, "cand::r0::xe"),),  # bad cut
        inverse_policy=POLICY,
        gates=CertificationGates(),
    )
    assert not result.certified
    assert result.reason.startswith("materialization_failed:")


def test_local_ray_refit_recovers_deleted_angle_via_duplicate_generator() -> None:
    """Hook #3: a survivor with the same generator absorbs the deleted angle."""

    from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
        build_local_ray_refit,
    )
    from pipelines.time_dynamics.ap_mclachlan.state import (
        state_without_runtime_indices,
    )

    xa = AnsatzTerm(label="sx0a", polynomial=_poly(("ex", 1.0)))
    xb = AnsatzTerm(label="sx0b", polynomial=_poly(("ex", 1.0)))
    state = _state((xa, xb, Z0), np.array([0.4, 0.3, -0.2]))
    target = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
    ).psi

    pruned_state, pruned_theta = state_without_runtime_indices(
        state, (0,), theta_runtime=state.theta_runtime
    )
    refit = build_local_ray_refit(
        target_psi=target, trust_radius=0.6, max_iterations=50
    )
    refit_state, refit_theta = refit(pruned_state, pruned_theta)
    assert refit_state is pruned_state
    psi = evaluate_mclachlan_geometry(
        state=pruned_state,
        hamiltonian=HAM,
        theta_runtime=refit_theta,
        time=0.0,
    ).psi
    fidelity = abs(complex(np.vdot(target, psi))) ** 2
    assert fidelity == pytest.approx(1.0, abs=1.0e-10)
    # The surviving same-generator coordinate absorbed the deleted 0.4.
    assert refit_theta[0] == pytest.approx(0.7, abs=1.0e-5)
    assert refit_theta[1] == pytest.approx(-0.2, abs=1.0e-5)


def test_local_ray_refit_respects_trust_region_and_monotonicity() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
        build_local_ray_refit,
    )
    from pipelines.time_dynamics.ap_mclachlan.state import (
        state_without_runtime_indices,
    )

    xa = AnsatzTerm(label="sx0a", polynomial=_poly(("ex", 1.0)))
    xb = AnsatzTerm(label="sx0b", polynomial=_poly(("ex", 1.0)))
    state = _state((xa, xb, Z0), np.array([0.4, 0.3, -0.2]))
    target = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
    ).psi
    pruned_state, pruned_theta = state_without_runtime_indices(
        state, (0,), theta_runtime=state.theta_runtime
    )

    # Radius 0.1 cannot reach the exact compensation (+0.4): the refit stays
    # inside the box and still strictly improves the overlap.
    refit = build_local_ray_refit(
        target_psi=target, trust_radius=0.1, max_iterations=50
    )
    _s, refit_theta = refit(pruned_state, pruned_theta)
    assert np.all(np.abs(refit_theta - pruned_theta) <= 0.1 + 1.0e-12)

    def infidelity(theta):
        psi = evaluate_mclachlan_geometry(
            state=pruned_state, hamiltonian=HAM, theta_runtime=theta, time=0.0
        ).psi
        return 1.0 - abs(complex(np.vdot(target, psi))) ** 2

    assert infidelity(refit_theta) < infidelity(pruned_theta)

    # A state already on the ray is returned untouched (pure-insert skip).
    on_ray = build_local_ray_refit(target_psi=target, trust_radius=0.5)
    same_state, same_theta = on_ray(state, state.theta_runtime)
    assert same_state is state
    assert np.array_equal(same_theta, np.asarray(state.theta_runtime, dtype=float))


def test_refit_certifies_deletion_that_fails_without_it() -> None:
    from pipelines.time_dynamics.ap_mclachlan.exchange_certification import (
        build_local_ray_refit,
    )

    xa = AnsatzTerm(label="sx0a", polynomial=_poly(("ex", 1.0)))
    xb = AnsatzTerm(label="sx0b", polynomial=_poly(("ex", 1.0)))
    state = _state((xa, xb, Z0), np.array([0.4, 0.3, -0.2]))
    evaluation = evaluate_mclachlan_geometry(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        include_tangent_matrix=True,
    )
    step = solve_fixed_mclachlan_step(evaluation.geometry, inverse_policy=POLICY)
    gates = CertificationGates(ray_distance_max=1.0e-4, smoothness_eta_max=1.0e6)
    common = dict(
        state=state,
        hamiltonian=HAM,
        theta_runtime=state.theta_runtime,
        time=0.0,
        base_evaluation=evaluation,
        base_step=step,
        removed_runtime_indices=(0,),
        insertions=(),
        inverse_policy=POLICY,
        gates=gates,
    )

    rejected = certify_finalist(**common)
    assert not rejected.certified
    assert rejected.reason == "ray_distance_above_max"

    refit = build_local_ray_refit(
        target_psi=evaluation.psi, trust_radius=0.6, max_iterations=50
    )
    certified = certify_finalist(**common, refit=refit)
    assert certified.certified
    assert certified.ray_distance <= 1.0e-4
    assert certified.theta[0] == pytest.approx(0.7, abs=1.0e-5)
