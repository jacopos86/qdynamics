from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

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


# ---------------------------------------------------------------------------
# Permutation/sign table cache (runtime acceleration pass)
# ---------------------------------------------------------------------------


def _reference_apply(psi, action):
    """Pre-cache construction, kept as the bitwise parity reference."""

    from src.quantum.pauli_actions import _basis_indices, _phase_signs

    psi_vec = np.asarray(psi, dtype=complex).reshape(-1)
    indices = _basis_indices(int(action.nq))
    source = (
        indices
        if int(action.flip_mask) == 0
        else np.bitwise_xor(indices, np.int64(action.flip_mask))
    )
    out = np.asarray(psi_vec[source], dtype=complex)
    signs = _phase_signs(source, int(action.phase_mask))
    if signs is not None:
        out = out * signs
    prefactor = (1.0 + 0.0j, 0.0 + 1.0j, -1.0 + 0.0j, 0.0 - 1.0j)[
        int(action.y_count_mod4)
    ]
    if prefactor != 1.0 + 0.0j:
        out = out * prefactor
    return out


@pytest.mark.parametrize(
    "word",
    ["exyz", "xxxx", "zzzz", "eeee", "xyex", "yyyy", "zexy", "eeyx"],
)
def test_cached_pauli_action_is_bitwise_identical_to_rebuild(word: str) -> None:
    from src.quantum.pauli_actions import apply_compiled_pauli, compile_pauli_action_exyz

    action = compile_pauli_action_exyz(word, 4)
    rng = np.random.default_rng(7)
    psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    for _ in range(3):  # repeat so the second call hits the cache
        assert np.array_equal(apply_compiled_pauli(psi, action), _reference_apply(psi, action))


def test_cached_pauli_columns_match_single_vector_application() -> None:
    from src.quantum.pauli_actions import (
        apply_compiled_pauli,
        apply_compiled_pauli_to_columns,
        compile_pauli_action_exyz,
    )

    action = compile_pauli_action_exyz("xyzе".replace("е", "e"), 4)
    rng = np.random.default_rng(11)
    columns = rng.standard_normal((16, 5)) + 1j * rng.standard_normal((16, 5))
    applied = apply_compiled_pauli_to_columns(columns, action)
    for index in range(columns.shape[1]):
        assert np.array_equal(applied[:, index], apply_compiled_pauli(columns[:, index], action))


def test_cached_tables_are_read_only_and_not_mutated_by_callers() -> None:
    """A caller scaling its result must not corrupt the shared table."""

    from src.quantum.pauli_actions import (
        _permutation_and_signs,
        apply_compiled_pauli,
        compile_pauli_action_exyz,
    )

    action = compile_pauli_action_exyz("xyzx", 4)
    source, signs = _permutation_and_signs(action)
    assert not source.flags.writeable
    if signs is not None:
        assert not signs.flags.writeable
    before = (source.copy(), None if signs is None else signs.copy())
    rng = np.random.default_rng(3)
    psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
    out = apply_compiled_pauli(psi, action)
    out *= 5.0  # caller mutates its own result
    after_source, after_signs = _permutation_and_signs(action)
    assert np.array_equal(before[0], after_source)
    if signs is not None:
        assert np.array_equal(before[1], after_signs)


def test_large_systems_bypass_the_table_cache() -> None:
    """Above the ceiling the tables must not be retained."""

    from src.quantum.pauli_actions import (
        _PERMUTATION_TABLE_CACHE_MAX_NQ,
        _cached_permutation_and_signs,
        _permutation_and_signs,
        compile_pauli_action_exyz,
    )

    big_nq = int(_PERMUTATION_TABLE_CACHE_MAX_NQ) + 2
    action = compile_pauli_action_exyz("x" + "e" * (big_nq - 1), big_nq)
    _cached_permutation_and_signs.cache_clear()
    first, _ = _permutation_and_signs(action)
    second, _ = _permutation_and_signs(action)
    assert _cached_permutation_and_signs.cache_info().currsize == 0
    assert first is not second  # rebuilt, not retained
    assert np.array_equal(first, second)


def test_small_systems_reuse_the_same_table_object() -> None:
    from src.quantum.pauli_actions import _permutation_and_signs, compile_pauli_action_exyz

    action = compile_pauli_action_exyz("xyzx", 4)
    first, _ = _permutation_and_signs(action)
    second, _ = _permutation_and_signs(action)
    assert first is second
