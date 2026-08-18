"""Route-object integration of the exchange selector."""

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


def test_selector_commits_from_real_route_objects_and_reports_payload() -> None:
    selection, payload = _run(_config())
    assert selection.kind in {"insert", "delete", "exchange"}
    assert payload["selection_policy"].startswith("paper_ii_deletion_conditioned")
    assert payload["kind"] == selection.kind
    assert payload["attempt_count"] == len(selection.attempts)
    assert payload["cost_normalization"] == "per_candidate_raw_v1"
    committed = payload["committed"]
    assert committed["score"] == pytest.approx(selection.committed.score)
    # Certified route objects exist and the patched state differs structurally.
    assert selection.certification.state is not None
    if selection.kind == "insert":
        assert (
            selection.certification.state.runtime_parameter_count
            > 2
        )
    # Inserted coordinate labels are the occurrence labels (time index tagged).
    if selection.committed.inserted_selection:
        labels = selection.certification.state.runtime_coordinate_labels
        assert any("::insr3c" in lab for lab in labels)


def test_work_guard_and_frontier_telemetry_flow_through_payload() -> None:
    selection, payload = _run(
        _config(
            prune_ray_distance_tol=1.0e-15,
            prune_patch_smoothness_eta_max=1.0e-15,
            structural_score_floor=1.0e30,
        )
    )
    assert selection.kind == "stay"
    assert payload["kind"] == "stay"
    assert "work_guard" in payload
    assert payload["work_guard"]["rejected_family"] is None
    assert payload["attempt_count"] == 0


def test_escalation_respects_residual_threshold_predicate() -> None:
    # A score floor removes every d0 candidate; with the residual threshold
    # far above the toy residual, escalation stops before any rung opens.
    selection, payload = _run(
        _config(
            residual_ratio_threshold=1.0e9,
            structural_score_floor=1.0e30,
        )
    )
    assert selection.kind == "stay"
    assert selection.stop_reason == "escalation_predicate_false"
    assert selection.attempts == ()


def test_min_surviving_support_bounds_deletions() -> None:
    selection, payload = _run(
        _config(
            min_runtime_parameter_count=2,
            prune_ray_distance_tol=1.0e-15,
            prune_patch_smoothness_eta_max=1.0e-15,
        )
    )
    # With 2 active coordinates and min survivors 2, no deletion is feasible:
    # every attempt is insert-kind only.
    kinds = {a.kind for a in selection.attempts}
    assert "delete" not in kinds and "exchange" not in kinds
