from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on path (same pattern as other integration tests).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    HubbardHolsteinTermwiseAnsatz,
    HubbardTermwiseAnsatz,
    PauliPolynomial,
    PauliTerm,
    apply_exp_pauli_polynomial_termwise,
)


def _random_state(nq: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    psi = rng.normal(size=1 << int(nq)) + 1j * rng.normal(size=1 << int(nq))
    psi = np.asarray(psi, dtype=complex)
    return psi / np.linalg.norm(psi)


def test_compiled_ansatz_parity_hubbard_termwise():
    ansatz = HubbardTermwiseAnsatz(
        dims=2,
        t=1.0,
        U=4.0,
        v=0.2,
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
        include_potential_terms=True,
    )
    rng = np.random.default_rng(111)
    theta = rng.normal(scale=0.3, size=ansatz.num_parameters)
    psi_ref = _random_state(ansatz.nq, seed=112)

    psi_slow = ansatz.prepare_state(
        theta,
        psi_ref,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        ansatz.base_terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
    )
    psi_fast = executor.prepare_state(theta, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10


def test_compiled_ansatz_parity_hh_termwise():
    ansatz = HubbardHolsteinTermwiseAnsatz(
        dims=2,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        v=[0.1, -0.1],
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
        include_zero_point=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    assert int(ansatz.nq) <= 10

    rng = np.random.default_rng(211)
    theta = rng.normal(scale=0.2, size=ansatz.num_parameters)
    psi_ref = _random_state(ansatz.nq, seed=212)

    psi_slow = ansatz.prepare_state(
        theta,
        psi_ref,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        ansatz.base_terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
    )
    psi_fast = executor.prepare_state(theta, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10


def test_compiled_ansatz_parity_per_pauli_term_mode():
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="xx", pc=1.0),
            PauliTerm(2, ps="zz", pc=0.5),
        ],
    )
    term = AnsatzTerm(label="multi", polynomial=poly)
    theta_runtime = np.array([0.2, -0.15], dtype=float)
    psi_ref = _random_state(2, seed=313)

    psi_slow = apply_exp_pauli_polynomial_termwise(
        psi_ref,
        poly,
        theta_runtime,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
    )
    psi_fast = executor.prepare_state(theta_runtime, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10


def test_grouped_exact_large_dimension_path_matches_dense() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(3, ps="xxe", pc=1.0),
            PauliTerm(3, ps="zze", pc=0.5),
        ],
    )
    term = AnsatzTerm(label="grouped", polynomial=poly, execution_mode="grouped_exact")
    theta = np.array([0.37], dtype=float)
    psi_ref = _random_state(3, seed=350)

    dense_executor = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="logical_shared",
    )
    psi_dense = dense_executor.prepare_state(theta, psi_ref)

    old_limit = CompiledAnsatzExecutor.GROUPED_EXACT_MAX_DIM
    try:
        CompiledAnsatzExecutor.GROUPED_EXACT_MAX_DIM = 1
        sparse_executor = CompiledAnsatzExecutor(
            [term],
            coefficient_tolerance=1e-12,
            ignore_identity=True,
            sort_terms=True,
            parameterization_mode="logical_shared",
        )
    finally:
        CompiledAnsatzExecutor.GROUPED_EXACT_MAX_DIM = old_limit

    assert sparse_executor._plans[0].exact_sparse_matrix is None
    psi_sparse = sparse_executor.prepare_state(theta, psi_ref)
    assert sparse_executor._plans[0].exact_sparse_matrix is not None

    assert np.linalg.norm(psi_dense - psi_sparse) < 1e-10


def test_grouped_exact_zero_angle_defers_sparse_plan_construction() -> None:
    term = AnsatzTerm(
        label="deferred_grouped",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(3, ps="xxe", pc=0.5),
                PauliTerm(3, ps="yye", pc=0.5),
            ],
        ),
        execution_mode="grouped_exact",
    )
    old_limit = CompiledAnsatzExecutor.GROUPED_EXACT_MAX_DIM
    try:
        CompiledAnsatzExecutor.GROUPED_EXACT_MAX_DIM = 1
        executor = CompiledAnsatzExecutor([term])
        assert executor._plans[0].exact_sparse_matrix is None

        psi_ref = np.zeros(8, dtype=complex)
        psi_ref[1] = 1.0
        np.testing.assert_array_equal(
            executor.prepare_state(np.asarray([0.0]), psi_ref),
            psi_ref,
        )
        assert executor._plans[0].exact_sparse_matrix is None

        executor.prepare_state(np.asarray([0.2]), psi_ref)
        assert executor._plans[0].exact_sparse_matrix is not None
    finally:
        CompiledAnsatzExecutor.GROUPED_EXACT_MAX_DIM = old_limit


def test_grouped_exact_invariant_basis_matches_full_space() -> None:
    term = AnsatzTerm(
        label="number_preserving_grouped",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(2, ps="xx", pc=0.5),
                PauliTerm(2, ps="yy", pc=0.5),
            ],
        ),
        execution_mode="grouped_exact",
    )
    theta = np.asarray([0.37], dtype=float)
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[1] = np.sqrt(0.4)
    psi_ref[2] = 1.0j * np.sqrt(0.6)

    full = CompiledAnsatzExecutor([term])
    reduced = CompiledAnsatzExecutor(
        [term],
        invariant_basis_indices=np.asarray([1, 2], dtype=np.int64),
    )
    expected = full.prepare_state(theta, psi_ref)
    observed = reduced.prepare_state(theta, psi_ref)

    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1.0e-13)
    plan = reduced._plans[0]
    reduced_matrix = (
        plan.exact_sparse_matrix
        if plan.exact_sparse_matrix is not None
        else plan.exact_eigvecs
    )
    assert reduced_matrix is not None
    assert reduced_matrix.shape == (2, 2)


def test_grouped_exact_eigendecomposition_reuses_shared_pauli_cache() -> None:
    term = AnsatzTerm(
        label="repeated_grouped_parent",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(3, ps="xxx", pc=0.5),
                PauliTerm(3, ps="zxx", pc=-0.25),
            ],
        ),
        execution_mode="grouped_exact",
    )
    shared_cache: dict[object, object] = {}
    first = CompiledAnsatzExecutor([term], pauli_action_cache=shared_cache)  # type: ignore[arg-type]
    second = CompiledAnsatzExecutor([term, term], pauli_action_cache=shared_cache)  # type: ignore[arg-type]

    first_plan = first._plans[0]
    second_plan = second._plans[0]
    repeated_plan = second._plans[1]
    assert first_plan.exact_eigvals is second_plan.exact_eigvals
    assert first_plan.exact_eigvals is repeated_plan.exact_eigvals
    assert first_plan.exact_eigvecs is second_plan.exact_eigvecs
    assert first_plan.exact_eigvecs is repeated_plan.exact_eigvecs
    assert first_plan.exact_eigvecs_dagger is second_plan.exact_eigvecs_dagger
    assert first_plan.exact_eigvecs_dagger is repeated_plan.exact_eigvecs_dagger

    psi_ref = _random_state(3, seed=351)
    expected = first.prepare_state(np.asarray([0.37]), psi_ref)
    actual = second.prepare_state(np.asarray([0.37, 0.0]), psi_ref)
    assert np.linalg.norm(expected - actual) < 1e-12


def test_logical_prefix_state_cache_is_bitwise_identical_and_reuses_unchanged_prefix() -> None:
    terms = [
        AnsatzTerm(
            label=f"term_{index}",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(3, ps=left, pc=0.5),
                    PauliTerm(3, ps=right, pc=-0.25),
                ],
            ),
            execution_mode="grouped_exact",
        )
        for index, (left, right) in enumerate(
            (("xxx", "zxx"), ("exx", "ezx"), ("xex", "zez"))
        )
    ]
    shared_cache: dict[object, object] = {}
    cached = CompiledAnsatzExecutor(
        terms,
        pauli_action_cache=shared_cache,  # type: ignore[arg-type]
        enable_prefix_state_cache=True,
    )
    uncached = CompiledAnsatzExecutor(
        terms,
        pauli_action_cache=shared_cache,  # type: ignore[arg-type]
        enable_prefix_state_cache=False,
    )
    psi_ref = _random_state(3, seed=352)
    theta_sequence = (
        np.asarray([0.1, 0.2, 0.3]),
        np.asarray([0.1, 0.2, 0.35]),
        np.asarray([0.1, 0.25, 0.35]),
        np.asarray([0.1, 0.25, 0.35]),
    )
    for theta in theta_sequence:
        expected = uncached.prepare_state(theta, psi_ref)
        actual = cached.prepare_state(theta, psi_ref)
        np.testing.assert_array_equal(actual, expected)

    assert cached.prefix_state_cache_evaluation_count == 4
    assert cached.prefix_state_cache_hit_count == 3
    assert cached.prefix_state_cache_reused_operator_count == 6


def test_compiled_ansatz_honors_provided_native_layout_order() -> None:
    poly = PauliPolynomial(
        "JW",
        [
            PauliTerm(2, ps="zz", pc=0.5),
            PauliTerm(2, ps="xx", pc=1.0),
        ],
    )
    term = AnsatzTerm(label="multi", polynomial=poly)
    layout = build_parameter_layout(
        [term],
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=False,
    )
    theta_runtime = np.array([0.1, -0.2], dtype=float)
    psi_ref = _random_state(2, seed=401)

    psi_slow = apply_exp_pauli_polynomial_termwise(
        psi_ref,
        poly,
        theta_runtime,
        ignore_identity=True,
        coefficient_tolerance=1e-12,
        sort_terms=False,
    )
    executor = CompiledAnsatzExecutor(
        [term],
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    psi_fast = executor.prepare_state(theta_runtime, psi_ref)

    assert np.linalg.norm(psi_slow - psi_fast) < 1e-10


def test_compiled_ansatz_runtime_tangents_match_finite_difference() -> None:
    terms = [
        AnsatzTerm(
            label="multi",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xx", pc=1.0),
                    PauliTerm(2, ps="zz", pc=0.5),
                ],
            ),
        ),
        AnsatzTerm(
            label="single",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="yy", pc=0.75)]),
        ),
    ]
    theta_runtime = np.array([0.2, 0.0, -0.1], dtype=float)
    psi_ref = _random_state(2, seed=402)
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="per_pauli_term",
    )

    psi_fast, tangents = executor.prepare_state_with_runtime_tangents(theta_runtime, psi_ref)
    psi_check = executor.prepare_state(theta_runtime, psi_ref)
    assert np.linalg.norm(psi_fast - psi_check) < 1e-12

    eps = 1e-7
    for runtime_idx in range(int(executor.runtime_parameter_count)):
        psi_single, tangent_single = executor.prepare_state_with_runtime_tangents(
            theta_runtime,
            psi_ref,
            runtime_indices=(runtime_idx,),
        )
        assert np.linalg.norm(psi_fast - psi_single) < 1e-12
        assert np.linalg.norm(tangents[runtime_idx] - tangent_single[runtime_idx]) < 1e-12

        theta_plus = np.array(theta_runtime, copy=True)
        theta_minus = np.array(theta_runtime, copy=True)
        theta_plus[runtime_idx] += eps
        theta_minus[runtime_idx] -= eps
        fd = (
            executor.prepare_state(theta_plus, psi_ref)
            - executor.prepare_state(theta_minus, psi_ref)
        ) / (2.0 * eps)
        assert np.linalg.norm(fd - tangents[runtime_idx]) < 5e-6


def test_compiled_ansatz_logical_shared_tangents_match_finite_difference() -> None:
    terms = [
        AnsatzTerm(
            label="multi",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xx", pc=1.0),
                    PauliTerm(2, ps="zz", pc=0.5),
                ],
            ),
        ),
        AnsatzTerm(
            label="single",
            polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="yy", pc=0.75)]),
        ),
    ]
    theta_logical = np.array([0.2, -0.1], dtype=float)
    psi_ref = _random_state(2, seed=403)
    executor = CompiledAnsatzExecutor(
        terms,
        coefficient_tolerance=1e-12,
        ignore_identity=True,
        sort_terms=True,
        parameterization_mode="logical_shared",
    )

    psi_fast, tangents = executor.prepare_state_with_parameter_tangents(
        theta_logical,
        psi_ref,
    )
    psi_check = executor.prepare_state(theta_logical, psi_ref)
    assert np.linalg.norm(psi_fast - psi_check) < 1e-12

    eps = 1e-7
    for logical_idx in range(int(executor.logical_parameter_count)):
        theta_plus = np.array(theta_logical, copy=True)
        theta_minus = np.array(theta_logical, copy=True)
        theta_plus[logical_idx] += eps
        theta_minus[logical_idx] -= eps
        fd = (
            executor.prepare_state(theta_plus, psi_ref)
            - executor.prepare_state(theta_minus, psi_ref)
        ) / (2.0 * eps)
        assert np.linalg.norm(fd - tangents[logical_idx]) < 5e-6
