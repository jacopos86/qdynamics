from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest

from pipelines.static_adapt.exact_geometry_backend import (
    CompiledExactManifoldAdapter,
    EXACT_STATE_PROVENANCE,
    build_coordinate_registry_override,
    build_compiled_exact_manifold_adapter,
)
from pipelines.static_adapt.geometry_fingerprints import (
    candidate_generator_fingerprint,
)
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.compiled_polynomial import (
    CompiledPolynomialAction,
    compile_polynomial_action,
    energy_via_one_apply,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _normalized_random_state(nq: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    state = rng.normal(size=1 << nq) + 1.0j * rng.normal(size=1 << nq)
    state = np.asarray(state, dtype=complex)
    return state / np.linalg.norm(state)


def _problem(
    parameterization_mode: str,
) -> tuple[CompiledAnsatzExecutor, np.ndarray, CompiledPolynomialAction]:
    terms = [
        AnsatzTerm(
            label="multi",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xx", pc=0.75),
                    PauliTerm(2, ps="zz", pc=-0.4),
                ],
            ),
        ),
        AnsatzTerm(
            label="single",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ey", pc=0.6)],
            ),
        ),
    ]
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode=parameterization_mode,  # type: ignore[arg-type]
    )
    psi_ref = _normalized_random_state(2, seed=7121)
    hamiltonian = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="ze", pc=0.7),
            PauliTerm(2, ps="ex", pc=-0.25),
            PauliTerm(2, ps="yy", pc=0.31),
        ],
    )
    h_compiled = compile_polynomial_action(hamiltonian, tol=1.0e-14)
    return executor, psi_ref, h_compiled


def _assert_exact_evaluation_matches_finite_difference(
    *,
    adapter: CompiledExactManifoldAdapter,
    executor: CompiledAnsatzExecutor,
    psi_ref: np.ndarray,
    h_compiled: CompiledPolynomialAction,
) -> None:
    x = adapter.x0
    assert adapter.backend.supports_sparse_endpoint_primitives is True
    energy_only = adapter.backend.evaluate_energy(x)
    gradient_only = adapter.backend.evaluate_gradient(x)
    assert energy_only.metadata["typed_estimator_primitive"] == "energy"
    assert gradient_only.metadata["typed_estimator_primitive"] == (
        "coordinate_gradient"
    )
    assert gradient_only.metadata["full_metric_formed"] is False
    evaluation = adapter.backend.evaluate(x)
    assert evaluation.statevector.ndim == 1
    assert evaluation.tangents.shape == (evaluation.statevector.size, x.size)
    assert evaluation.gradient.shape == x.shape
    assert evaluation.metadata["gradient_provenance"] == EXACT_STATE_PROVENANCE
    assert evaluation.metadata["finite_differences_used"] is False
    assert evaluation.metadata["hamiltonian_apply_count"] == 1
    assert energy_only.energy == pytest.approx(evaluation.energy, abs=1.0e-13)
    np.testing.assert_allclose(
        gradient_only.gradient, evaluation.gradient, atol=1.0e-13
    )

    direct_energy, hpsi = energy_via_one_apply(evaluation.statevector, h_compiled)
    assert evaluation.energy == pytest.approx(direct_energy, abs=1.0e-13)
    direct_gradient = 2.0 * np.real(
        np.conjugate(evaluation.tangents).T @ hpsi
    )
    np.testing.assert_allclose(evaluation.gradient, direct_gradient, atol=1.0e-13)

    epsilon = 1.0e-7
    for coordinate_index in range(x.size):
        plus = x.copy()
        minus = x.copy()
        plus[coordinate_index] += epsilon
        minus[coordinate_index] -= epsilon
        state_plus = executor.prepare_state(plus, psi_ref)
        state_minus = executor.prepare_state(minus, psi_ref)
        tangent_fd_raw = (state_plus - state_minus) / (2.0 * epsilon)
        tangent_fd = tangent_fd_raw - evaluation.statevector * np.vdot(
            evaluation.statevector,
            tangent_fd_raw,
        )
        np.testing.assert_allclose(
            evaluation.tangents[:, coordinate_index],
            tangent_fd,
            atol=3.0e-8,
            rtol=3.0e-8,
        )
        energy_plus, _ = energy_via_one_apply(state_plus, h_compiled)
        energy_minus, _ = energy_via_one_apply(state_minus, h_compiled)
        gradient_fd = (energy_plus - energy_minus) / (2.0 * epsilon)
        assert evaluation.gradient[coordinate_index] == pytest.approx(
            gradient_fd,
            abs=3.0e-8,
            rel=3.0e-8,
        )


def test_logical_shared_adapter_uses_logical_executor_coordinates_and_lifts() -> None:
    executor, psi_ref, h_compiled = _problem("logical_shared")
    theta_runtime = np.asarray([0.24, -0.08, 0.11], dtype=float)
    adapter = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id="unit_logical_shared",
    )

    np.testing.assert_allclose(adapter.x0, [0.08, 0.11], atol=0.0)
    np.testing.assert_allclose(
        adapter.lift_to_runtime(np.asarray([0.3, -0.2])),
        [0.3, 0.3, -0.2],
        atol=0.0,
    )
    assert len(adapter.coordinate_registry) == executor.logical_parameter_count
    assert len(set(adapter.coordinate_registry)) == len(adapter.coordinate_registry)
    assert all(value.startswith("logical:") for value in adapter.coordinate_registry)

    summary = adapter.summary
    assert summary["parameterization_mode"] == "logical_shared"
    assert summary["finite_differences_used"] is False
    assert len(summary["layout_sha256"]) == 64
    assert summary["coordinate_records"][0]["runtime_indices"] == [0, 1]
    json.dumps(summary, allow_nan=False, sort_keys=True)

    rebuilt = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id="unit_logical_shared",
    )
    assert rebuilt.summary["layout_sha256"] == summary["layout_sha256"]
    assert rebuilt.coordinate_registry == adapter.coordinate_registry

    _assert_exact_evaluation_matches_finite_difference(
        adapter=adapter,
        executor=executor,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
    )


def test_per_pauli_adapter_uses_runtime_executor_coordinates_and_identity_lift() -> None:
    executor, psi_ref, h_compiled = _problem("per_pauli_term")
    theta_runtime = np.asarray([0.24, -0.08, 0.11], dtype=float)
    adapter = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=executor.layout,
        theta_runtime=theta_runtime,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        manifold_id="unit_per_pauli",
    )

    np.testing.assert_array_equal(adapter.x0, theta_runtime)
    lifted = adapter.lift_to_runtime(np.asarray([-0.4, 0.2, 0.05]))
    np.testing.assert_array_equal(lifted, [-0.4, 0.2, 0.05])
    assert len(adapter.coordinate_registry) == executor.runtime_parameter_count
    assert len(set(adapter.coordinate_registry)) == len(adapter.coordinate_registry)
    assert all(value.startswith("runtime:") for value in adapter.coordinate_registry)
    summary = adapter.summary
    assert summary["parameterization_mode"] == "per_pauli_term"
    assert [
        record["coordinate_index"] for record in summary["coordinate_records"]
    ] == [0, 1, 2]
    json.dumps(summary, allow_nan=False, sort_keys=True)

    _assert_exact_evaluation_matches_finite_difference(
        adapter=adapter,
        executor=executor,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
    )


@pytest.mark.parametrize("parameterization_mode", ["logical_shared", "per_pauli_term"])
def test_coordinate_ids_survive_unrelated_block_insertion(
    parameterization_mode: str,
) -> None:
    executor, psi_ref, h_compiled = _problem(parameterization_mode)
    inserted = AnsatzTerm(
        label="unrelated_inserted_block",
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(2, ps="xy", pc=0.2)],
        ),
    )
    grown_executor = CompiledAnsatzExecutor(
        [inserted, *executor.terms],
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode=parameterization_mode,  # type: ignore[arg-type]
    )
    base = build_compiled_exact_manifold_adapter(
        executor=executor,
        layout=executor.layout,
        theta_runtime=np.zeros(executor.runtime_parameter_count),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
    )
    grown = build_compiled_exact_manifold_adapter(
        executor=grown_executor,
        layout=grown_executor.layout,
        theta_runtime=np.zeros(grown_executor.runtime_parameter_count),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
    )

    def ids_by_label(summary: dict[str, Any]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for record in summary["coordinate_records"]:
            out.setdefault(str(record["candidate_label"]), []).append(
                str(record["coordinate_id"])
            )
        return out

    base_ids = ids_by_label(base.summary)
    grown_ids = ids_by_label(grown.summary)
    assert grown_ids["multi"] == base_ids["multi"]
    assert grown_ids["single"] == base_ids["single"]


@pytest.mark.parametrize(
    "parameterization_mode", ["logical_shared", "per_pauli_term"]
)
def test_repeat_insertion_preserves_inherited_coordinate_instances(
    parameterization_mode: str,
) -> None:
    _executor, psi_ref, h_compiled = _problem(parameterization_mode)
    generator_a = AnsatzTerm(
        label="A",
        polynomial=PauliPolynomial(
            "JW", [PauliTerm(2, ps="xe", pc=0.5)]
        ),
    )
    generator_b = AnsatzTerm(
        label="B",
        polynomial=PauliPolynomial(
            "JW", [PauliTerm(2, ps="ze", pc=-0.7)]
        ),
    )
    base_executor = CompiledAnsatzExecutor(
        [generator_a, generator_b, generator_a],
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode=parameterization_mode,  # type: ignore[arg-type]
    )
    base_generators = tuple(
        candidate_generator_fingerprint(term)
        for term in base_executor.terms
    )
    base_override = build_coordinate_registry_override(
        base_executor.layout,
        parameterization_mode=parameterization_mode,
        current_generator_fingerprints=base_generators,
        admission_context="repeat_base",
    )
    theta_base = np.asarray([0.17, -0.23, 0.31], dtype=float)
    base = build_compiled_exact_manifold_adapter(
        executor=base_executor,
        layout=base_executor.layout,
        theta_runtime=theta_base,
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        coordinate_registry_override=base_override,
    )

    grown_executor = CompiledAnsatzExecutor(
        [generator_a, generator_a, generator_b, generator_a],
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode=parameterization_mode,  # type: ignore[arg-type]
    )
    grown_generators = tuple(
        candidate_generator_fingerprint(term)
        for term in grown_executor.terms
    )
    grown_override = build_coordinate_registry_override(
        grown_executor.layout,
        parameterization_mode=parameterization_mode,
        inherited_coordinate_ids=base.coordinate_registry,
        old_to_new_registry_mapping=(1, 2, 3),
        parent_generator_fingerprints=base_generators,
        current_generator_fingerprints=grown_generators,
        old_to_new_generator_mapping=(1, 2, 3),
        admission_context="repeat_insert_before_first_a",
    )
    grown = build_compiled_exact_manifold_adapter(
        executor=grown_executor,
        layout=grown_executor.layout,
        theta_runtime=np.asarray([0.0, *theta_base], dtype=float),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        coordinate_registry_override=grown_override,
    )

    assert grown_override.admitted_coordinate_positions == (0,)
    assert grown_override.old_to_new_registry_mapping == (1, 2, 3)
    assert tuple(
        grown.coordinate_registry[index] for index in (1, 2, 3)
    ) == base.coordinate_registry
    assert grown.coordinate_registry[0] not in set(base.coordinate_registry)
    np.testing.assert_allclose(
        grown.backend.evaluate(grown.x0).statevector,
        base.backend.evaluate(base.x0).statevector,
        atol=2.0e-13,
        rtol=0.0,
    )

def test_registry_override_rejects_reordered_inherited_generators() -> None:
    executor, _psi_ref, _h_compiled = _problem("logical_shared")
    generators = tuple(
        candidate_generator_fingerprint(term) for term in executor.terms
    )
    with pytest.raises(ValueError, match="preserve inherited gate order"):
        build_coordinate_registry_override(
            executor.layout,
            parameterization_mode="logical_shared",
            inherited_coordinate_ids=("old:0", "old:1"),
            old_to_new_registry_mapping=(1, 0),
            parent_generator_fingerprints=generators,
            current_generator_fingerprints=generators,
            old_to_new_generator_mapping=(1, 0),
            admission_context="invalid_reorder",
        )


def test_registry_override_prune_keeps_only_surviving_coordinate_instances() -> None:
    _executor, psi_ref, h_compiled = _problem("logical_shared")
    generator_a = AnsatzTerm(
        label="A",
        polynomial=PauliPolynomial(
            "JW", [PauliTerm(2, ps="xe", pc=0.5)]
        ),
    )
    generator_b = AnsatzTerm(
        label="B",
        polynomial=PauliPolynomial(
            "JW", [PauliTerm(2, ps="ze", pc=-0.7)]
        ),
    )
    parent_executor = CompiledAnsatzExecutor(
        [generator_a, generator_b, generator_a],
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="logical_shared",
    )
    parent_generators = tuple(
        candidate_generator_fingerprint(term)
        for term in parent_executor.terms
    )
    parent_override = build_coordinate_registry_override(
        parent_executor.layout,
        parameterization_mode="logical_shared",
        current_generator_fingerprints=parent_generators,
        admission_context="prune_parent",
    )
    parent = build_compiled_exact_manifold_adapter(
        executor=parent_executor,
        layout=parent_executor.layout,
        theta_runtime=np.asarray([0.17, 0.0, -0.31], dtype=float),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        coordinate_registry_override=parent_override,
    )

    pruned_executor = CompiledAnsatzExecutor(
        [generator_a, generator_a],
        coefficient_tolerance=1.0e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="logical_shared",
    )
    surviving_parent_indices = (0, 2)
    surviving_generators = tuple(
        parent_generators[index] for index in surviving_parent_indices
    )
    pruned_override = build_coordinate_registry_override(
        pruned_executor.layout,
        parameterization_mode="logical_shared",
        inherited_coordinate_ids=tuple(
            parent.coordinate_registry[index]
            for index in surviving_parent_indices
        ),
        old_to_new_registry_mapping=(0, 1),
        parent_generator_fingerprints=surviving_generators,
        current_generator_fingerprints=surviving_generators,
        old_to_new_generator_mapping=(0, 1),
        admission_context="accepted_variable_prune",
    )
    pruned = build_compiled_exact_manifold_adapter(
        executor=pruned_executor,
        layout=pruned_executor.layout,
        theta_runtime=np.asarray([0.17, -0.31], dtype=float),
        psi_ref=psi_ref,
        h_compiled=h_compiled,
        coordinate_registry_override=pruned_override,
    )

    assert pruned.coordinate_registry == (
        parent.coordinate_registry[0],
        parent.coordinate_registry[2],
    )
    assert pruned_override.admitted_coordinate_positions == ()
    assert pruned_override.allocation_records == ()
    np.testing.assert_allclose(
        pruned.backend.evaluate(pruned.x0).statevector,
        parent.backend.evaluate(parent.x0).statevector,
        atol=2.0e-13,
        rtol=0.0,
    )
