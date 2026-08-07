from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from pipelines.static_adapt.selector_exact_query_geometry import (
    build_exact_query_closed_candidate_geometry,
    candidate_generator_fingerprint,
    compiled_hamiltonian_fingerprint,
)
from pipelines.static_adapt.selector_query_closure import (
    CAPABILITY_COMMON_TANGENT_CONTRACTION,
    CAPABILITY_LIVE_TANGENT,
    EstimatorPrimitiveIdentity,
    QueryReceipt,
    evaluate_phase1_query_closed_score,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    apply_compiled_polynomial,
    compile_polynomial_action,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(label: str, pauli: str, coefficient: float = 1.0) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(len(pauli), ps=pauli, pc=coefficient)],
        ),
    )


def _normalized_random_state(*, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=4) + 1.0j * rng.normal(size=4)
    state = np.asarray(state, dtype=complex)
    return state / np.linalg.norm(state)


def _candidate_receipt(
    *,
    state_fingerprint: str,
    candidate_fingerprint: str,
    insertion_position: int,
    hamiltonian_fingerprint: str,
) -> QueryReceipt:
    primitive = EstimatorPrimitiveIdentity(
        primitive_kind="coordinate_gradient",
        physical_state_fingerprint=state_fingerprint,
        branch_id="branch-exact-adapter",
        ordered_scaffold_fingerprint="scaffold-receipt",
        theta_fingerprint="theta-receipt",
        coordinate_registry_fingerprint="registry-exact-adapter",
        candidate_generator_fingerprint=candidate_fingerprint,
        candidate_insertion_position=insertion_position,
        parameterization_tie_map_fingerprint="tie-map-exact-adapter",
        hamiltonian_fingerprint=hamiltonian_fingerprint,
        provider_backend_id="compiled-exact-state-v1",
        estimator_precision_contract="float64-exact",
        formula_primitive_identity="candidate-coordinate-differential-and-tangent",
    )
    return QueryReceipt.from_primitives(
        requested=(primitive,),
        returned_fields=("b_B", "tangent_handle"),
        closure_capabilities=(CAPABILITY_COMMON_TANGENT_CONTRACTION,),
        provenance_by_field={
            "b_B": "exact_state_coordinate_differential",
            "tangent_handle": "exact_state_live_tangent",
        },
        provider_kind="exact_state",
        statevector_shortcut_used=True,
    )


def _problem(*, supplied_phase: complex = 1.0 + 0.0j) -> dict[str, Any]:
    selected = (
        _term("selected-xe", "xe", 0.83),
        _term("selected-ey", "ey", -0.71),
    )
    candidate = _term("candidate-zx", "zx", 0.64)
    theta = np.asarray([0.19, -0.27], dtype=float)
    psi_ref = _normalized_random_state(seed=20260712)
    psi_state_unphased = CompiledAnsatzExecutor(selected).prepare_state(
        theta,
        psi_ref,
    )
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="ze", pc=0.73),
            PauliTerm(2, ps="ex", pc=-0.41),
            PauliTerm(2, ps="yy", pc=0.29),
        ],
    )
    h_compiled = compile_polynomial_action(hamiltonian, tol=1.0e-14)
    psi_state = complex(supplied_phase) * psi_state_unphased
    hpsi_state = complex(supplied_phase) * apply_compiled_polynomial(
        psi_state_unphased,
        h_compiled,
    )
    candidate_fingerprint = candidate_generator_fingerprint(candidate)
    receipt = _candidate_receipt(
        state_fingerprint="physical-state-receipt",
        candidate_fingerprint=candidate_fingerprint,
        insertion_position=1,
        hamiltonian_fingerprint=compiled_hamiltonian_fingerprint(h_compiled),
    )
    geometry = build_exact_query_closed_candidate_geometry(
        selected_ops=selected,
        theta=theta,
        psi_ref=psi_ref,
        psi_state=psi_state,
        hpsi_state=hpsi_state,
        candidate_term=candidate,
        insertion_position=1,
        active_coordinate_indices=(0, 1),
        branch_id="branch-exact-adapter",
        manifold_id="manifold-exact-adapter",
        coordinate_registry_fingerprint="registry-exact-adapter",
        parameterization_tie_map_fingerprint="tie-map-exact-adapter",
        h_compiled=h_compiled,
        candidate_query_receipt=receipt,
        pauli_action_cache={},
    )
    return {
        "selected": selected,
        "candidate": candidate,
        "theta": theta,
        "psi_ref": psi_ref,
        "psi_state": psi_state,
        "h_compiled": h_compiled,
        "receipt": receipt,
        "geometry": geometry,
    }


def _chart_energy(problem: dict[str, Any], chart_delta: np.ndarray) -> float:
    selected = problem["selected"]
    candidate = problem["candidate"]
    combined = (selected[0], candidate, selected[1])
    theta = problem["theta"]
    combined_theta = np.asarray(
        [
            theta[0] + chart_delta[0],
            chart_delta[1],
            theta[1] + chart_delta[2],
        ],
        dtype=float,
    )
    state = CompiledAnsatzExecutor(combined).prepare_state(
        combined_theta,
        problem["psi_ref"],
    )
    return float(
        np.real(
            np.vdot(
                state,
                apply_compiled_polynomial(state, problem["h_compiled"]),
            )
        )
    )


def test_exact_adapter_builds_anchor_and_candidate_in_insertion_chart() -> None:
    problem = _problem()
    geometry = problem["geometry"]
    anchor = geometry.anchor
    candidate = geometry.candidate_record

    assert geometry.state_reconstruction_delta_norm < 1.0e-13
    assert geometry.active_combined_indices == (0, 2)
    assert geometry.candidate_combined_index == 1
    assert anchor.active_coordinate_indices == (0, 1)
    assert anchor.parameterization_mode == "logical_shared"
    assert candidate.insertion_position == 1
    assert candidate.candidate_fingerprint == candidate_generator_fingerprint(
        problem["candidate"]
    )
    assert CAPABILITY_LIVE_TANGENT in candidate.closure_capabilities

    tangents = (*anchor.active_tangent_handles, candidate.tangent_handle)
    for tangent in tangents:
        assert abs(np.vdot(problem["psi_state"], tangent)) < 2.0e-14
    direct_gram = np.asarray(
        [
            [float(np.real(np.vdot(left, right))) for right in tangents]
            for left in tangents
        ],
        dtype=float,
    )
    np.testing.assert_allclose(anchor.G_AA, direct_gram[np.ix_((0, 1), (0, 1))])
    np.testing.assert_allclose(candidate.G_AB, direct_gram[(0, 1), 2])
    assert candidate.G_BB == pytest.approx(direct_gram[2, 2])

    epsilon = 1.0e-7
    finite_difference = np.zeros(3, dtype=float)
    for index in range(3):
        plus = np.zeros(3, dtype=float)
        minus = np.zeros(3, dtype=float)
        plus[index] = epsilon
        minus[index] = -epsilon
        finite_difference[index] = (
            _chart_energy(problem, plus) - _chart_energy(problem, minus)
        ) / (2.0 * epsilon)
    np.testing.assert_allclose(anchor.b_A, finite_difference[(0, 2),], atol=3.0e-9)
    assert candidate.b_B == pytest.approx(finite_difference[1], abs=3.0e-9)


def test_exact_adapter_is_global_phase_invariant() -> None:
    baseline = _problem()["geometry"]
    phase = np.exp(1.0j * 0.731)
    phased = _problem(supplied_phase=phase)["geometry"]

    # The primitive identity denotes a physical state, not a representative ray.
    assert phased.anchor.state_fingerprint == baseline.anchor.state_fingerprint
    np.testing.assert_allclose(phased.anchor.G_AA, baseline.anchor.G_AA, atol=2.0e-14)
    np.testing.assert_allclose(phased.anchor.b_A, baseline.anchor.b_A, atol=2.0e-14)
    np.testing.assert_allclose(
        phased.candidate_record.G_AB,
        baseline.candidate_record.G_AB,
        atol=2.0e-14,
    )
    assert phased.candidate_record.G_BB == pytest.approx(
        baseline.candidate_record.G_BB,
        abs=2.0e-14,
    )
    assert phased.candidate_record.b_B == pytest.approx(
        baseline.candidate_record.b_B,
        abs=2.0e-14,
    )
    for unphased_tangent, phased_tangent in zip(
        (*baseline.anchor.active_tangent_handles, baseline.candidate_record.tangent_handle),
        (*phased.anchor.active_tangent_handles, phased.candidate_record.tangent_handle),
    ):
        np.testing.assert_allclose(phased_tangent, phase * unphased_tangent, atol=2.0e-14)

    baseline_score = evaluate_phase1_query_closed_score(
        anchor=baseline.anchor,
        candidate=baseline.candidate_record,
        trust_radius=0.2,
        baseline_primitive_ids=baseline.candidate_record.source_primitive_ids,
    )
    phased_score = evaluate_phase1_query_closed_score(
        anchor=phased.anchor,
        candidate=phased.candidate_record,
        trust_radius=0.2,
        baseline_primitive_ids=phased.candidate_record.source_primitive_ids,
    )
    assert phased_score.feasible == baseline_score.feasible
    assert phased_score.score == pytest.approx(baseline_score.score, abs=2.0e-14)


def test_exact_adapter_portable_payload_excludes_dense_live_objects() -> None:
    geometry = _problem()["geometry"]
    payload = geometry.portable_payload()
    encoded = json.dumps(payload, sort_keys=True)

    assert payload["live_tangent_handles_serialized"] is False
    assert payload["dense_statevector_serialized"] is False
    assert "tangent_handle" not in payload["anchor"]
    assert "active_tangent_handles" not in payload["anchor"]
    assert "tangent_handle" not in payload["candidate_record"]
    assert "statevector" not in payload["anchor"]
    assert "statevector" not in payload["candidate_record"]
    assert "complex" not in encoded


def test_exact_adapter_output_is_compatible_with_query_closed_phase1_score() -> None:
    geometry = _problem()["geometry"]
    source_ids = (
        geometry.anchor.source_primitive_ids
        | geometry.candidate_record.source_primitive_ids
    )
    score = evaluate_phase1_query_closed_score(
        anchor=geometry.anchor,
        candidate=geometry.candidate_record,
        trust_radius=0.25,
        resource_burden=0.1,
        rank_relative_tolerance=1.0e-8,
        metric_regularization=1.0e-10,
        baseline_primitive_ids=source_ids,
    )

    assert score.feasible
    assert score.rank_gain == 1
    assert score.schur_metric > score.support_threshold
    assert score.primitive_set_reconciled is True
    assert score.incremental_query_charge == 0
    assert score.score > 0.0
    assert score.candidate_key == geometry.candidate_record.candidate_key


def test_candidate_fingerprint_tracks_generator_content_not_only_label() -> None:
    baseline = _term("same-label", "zx", 0.64)
    changed_word = _term("same-label", "zy", 0.64)
    changed_coefficient = _term("same-label", "zx", -0.64)

    assert candidate_generator_fingerprint(baseline) != candidate_generator_fingerprint(
        changed_word
    )
    assert candidate_generator_fingerprint(
        baseline
    ) != candidate_generator_fingerprint(changed_coefficient)


def test_exact_adapter_reuses_one_retained_anchor_across_candidates() -> None:
    problem = _problem()
    baseline = problem["geometry"]
    hpsi = apply_compiled_polynomial(
        problem["psi_state"], problem["h_compiled"]
    )
    reused = build_exact_query_closed_candidate_geometry(
        selected_ops=problem["selected"],
        theta=problem["theta"],
        psi_ref=problem["psi_ref"],
        psi_state=problem["psi_state"],
        hpsi_state=hpsi,
        candidate_term=problem["candidate"],
        insertion_position=1,
        active_coordinate_indices=(0, 1),
        branch_id="branch-exact-adapter",
        manifold_id="manifold-exact-adapter",
        coordinate_registry_fingerprint="registry-exact-adapter",
        parameterization_tie_map_fingerprint="tie-map-exact-adapter",
        h_compiled=problem["h_compiled"],
        candidate_query_receipt=problem["receipt"],
        retained_anchor=baseline.anchor,
        pauli_action_cache={},
    )
    assert reused.retained_anchor_reused is True
    assert reused.anchor is baseline.anchor
    np.testing.assert_allclose(
        reused.candidate_record.G_AB, baseline.candidate_record.G_AB
    )
    assert reused.candidate_record.G_BB == pytest.approx(
        baseline.candidate_record.G_BB
    )


def test_candidate_fingerprint_tracks_execution_parameterization_mode() -> None:
    polynomial = _term("same-label", "zx", 0.64).polynomial
    termwise = AnsatzTerm(
        label="same-label",
        polynomial=polynomial,
        execution_mode="termwise_product",
    )
    grouped = AnsatzTerm(
        label="same-label",
        polynomial=polynomial,
        execution_mode="grouped_exact",
    )
    assert candidate_generator_fingerprint(termwise) != (
        candidate_generator_fingerprint(grouped)
    )
