"""The offline replay bench reads the recorded operands the way Clarabel does."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from scipy import sparse

from paper5.stability.apcm_carried_witness import cwrmf_settings_for_profile
from paper5.stability.apcm_positive_extension import _clarabel_svec_upper
from paper5.stability.cwrmf_solver_instances import ConicInstance
from paper5.stability.cwrmf_solver_replay import (
    BALANCED_INFLATION_SHIFT,
    PROFILE_ACCEPTANCE,
    PROFILE_PSD_INFLATION,
    ClarabelCandidate,
    ReducedBundleCandidate,
    WarmStartedScsCandidate,
    _scs_cone_and_permutation,
    cone_blocks,
    maximum_cone_violation,
    objective_value,
    pack_svec_upper,
    unpack_svec_upper,
)
from paper5.stability.cwrmf_solver_replay_bench import declared_candidates


def _diagonal_instance(size: int, floor: float) -> ConicInstance:
    """One variable per diagonal entry of a ``size`` x ``size`` PSD block.

    ``A x + s = b`` with ``b = floor * svec(I)`` and the ``j``-th column of
    ``A`` equal to ``-svec(E_jj)``, so ``s`` is the diagonal matrix with
    entries ``floor + x_j`` and the cone constraint is ``x_j >= -floor``.
    """

    columns = []
    for index in range(size):
        basis = np.zeros((size, size))
        basis[index, index] = 1.0
        columns.append(-_clarabel_svec_upper(basis))
    cone_matrix = sparse.csc_matrix(np.column_stack(columns))
    cone_rhs = floor * _clarabel_svec_upper(np.eye(size))
    return ConicInstance(
        index=0,
        tier="bundle_cone",
        atom_index=0,
        atom_time=0.0,
        iteration=1,
        step_size=1.0,
        objective_matrix=sparse.eye(size, format="csc"),
        objective_vector=np.zeros(size),
        cone_matrix=cone_matrix,
        cone_rhs=cone_rhs,
        cone_specification=(("PSDTriangleConeT", size),),
        solver_settings={
            "verbose": False,
            "max_threads": 1,
            "max_iter": 200,
            "tol_gap_abs": 1e-10,
            "tol_gap_rel": 1e-10,
            "tol_feas": 1e-10,
        },
        base_velocity=np.zeros(size),
        correction_indices=np.arange(size, dtype=np.int64),
        gram_predictor=np.eye(size, dtype=complex),
        desired_velocity=np.zeros(size),
        completion_radius_squared=100.0,
        retained_scaled_norm=0.0,
        velocity_ceiling=10.0,
        recorded_solution=np.zeros(size),
        recorded_status="Solved",
        recorded_objective=0.0,
        recorded_primal_residual=0.0,
        recorded_dual_residual=0.0,
        recorded_iterations=1,
        recorded_clarabel_solve_time=0.0,
    )


@pytest.mark.parametrize("size", [1, 2, 5, 12])
def test_svec_round_trip_matches_the_call_sites_packing(size: int) -> None:
    generator = np.random.default_rng(size)
    matrix = generator.standard_normal((size, size))
    matrix = 0.5 * (matrix + matrix.T)

    packed = pack_svec_upper(matrix)

    np.testing.assert_array_equal(packed, _clarabel_svec_upper(matrix))
    np.testing.assert_allclose(
        unpack_svec_upper(packed, size), matrix, rtol=0.0, atol=1e-15
    )


def test_cone_blocks_cover_every_recorded_row() -> None:
    instance = _diagonal_instance(4, 1.0)

    blocks = cone_blocks(instance)

    assert blocks == [("PSDTriangleConeT", 4, slice(0, 10))]
    assert instance.constraint_count == 10


def test_maximum_cone_violation_signs_feasible_and_infeasible_points() -> None:
    instance = _diagonal_instance(3, 1.0)

    interior = maximum_cone_violation(instance, np.zeros(3))
    boundary = maximum_cone_violation(instance, -np.ones(3))
    outside = maximum_cone_violation(instance, -2.0 * np.ones(3))

    assert interior == pytest.approx(-1.0)
    assert boundary == pytest.approx(0.0, abs=1e-15)
    assert outside == pytest.approx(1.0)


def test_objective_value_is_the_conic_objective() -> None:
    instance = _diagonal_instance(3, 1.0)
    point = np.asarray([1.0, 2.0, 3.0])

    assert objective_value(instance, point) == pytest.approx(7.0)


def test_scs_permutation_maps_upper_column_major_to_lower_column_major(
) -> None:
    size = 4
    instance = _diagonal_instance(size, 1.0)
    generator = np.random.default_rng(7)
    matrix = generator.standard_normal((size, size))
    matrix = 0.5 * (matrix + matrix.T)
    packed = pack_svec_upper(matrix)

    cone, permutation = _scs_cone_and_permutation(instance)
    reordered = packed[permutation]

    assert cone == {"s": [size]}
    expected = []
    for column in range(size):
        for row in range(column, size):
            value = matrix[row, column]
            expected.append(
                value if row == column else np.sqrt(2.0) * value
            )
    np.testing.assert_allclose(
        reordered, np.asarray(expected), rtol=0.0, atol=1e-15
    )


def test_clarabel_candidate_reproduces_a_feasible_projection() -> None:
    instance = _diagonal_instance(3, -1.0)

    answer = ClarabelCandidate("recorded", "recorded settings", {}).solve(
        instance
    )

    assert answer.status == "Solved"
    np.testing.assert_allclose(
        answer.solution, np.ones(3), rtol=0.0, atol=1e-8
    )
    assert maximum_cone_violation(instance, answer.solution) <= 1e-8


def test_reduced_bundle_candidate_shrinks_the_positive_semidefinite_block(
) -> None:
    size = 6
    instance = _diagonal_instance(size, 1.0)

    answer = ReducedBundleCandidate(
        "reduced", "declared threshold", 1e-3
    ).solve(instance)

    # Every eigenvalue of the base is 1.0, so a 1e-3 relative threshold keeps
    # no direction and the candidate falls back to the single lowest one.
    assert answer.status.endswith(f"bundle=1/{size}")


def test_profile_acceptance_constants_are_the_named_profiles() -> None:
    for name, (safety, tolerance) in PROFILE_ACCEPTANCE.items():
        settings = cwrmf_settings_for_profile(name)
        assert settings.schur_safety_margin == safety
        assert settings.solver_tolerance == tolerance
        assert settings.psd_inflation == PROFILE_PSD_INFLATION[name]
    assert BALANCED_INFLATION_SHIFT == pytest.approx(1e-8 - 1e-11)


def _zero_gram_instance(size: int) -> ConicInstance:
    """A feasible instance whose reconstructed Gram is exactly zero.

    The reconstructed Gram is ``gram_predictor`` plus the completion
    coefficients contracted with the correction, so a zero predictor and zero
    coefficients put the minimum eigenvalue at ``0.0``: below the balanced
    profile's ``schur_safety_margin`` of ``1e-10`` without the substituted
    ``psd_inflation``, above it with the substitution.
    """

    return dataclasses.replace(
        _diagonal_instance(size, -1.0),
        gram_predictor=np.zeros((size, size), dtype=complex),
    )


def test_polish_gate_uses_the_declared_inflation_shift() -> None:
    size = 3
    instance = _zero_gram_instance(size)
    coefficients = np.zeros((size, size, size), dtype=complex)

    unshifted = WarmStartedScsCandidate(
        "scs_polish_unshifted",
        "balanced polish gate at the default inflation shift",
        1e-9,
        polish_profile="balanced",
        coefficients=coefficients,
    ).solve(instance)
    shifted = WarmStartedScsCandidate(
        "scs_polish_shifted",
        "balanced polish gate with the substituted psd_inflation",
        1e-9,
        polish_profile="balanced",
        coefficients=coefficients,
        polish_inflation_shift=BALANCED_INFLATION_SHIFT,
    ).solve(instance)

    # Without the substitution the balanced test rejects a Gram whose minimum
    # eigenvalue is 0.0, so the polish fires; with it the same answer is
    # accepted and no Clarabel solve follows.
    assert "->clarabel:" in unshifted.status
    assert "->clarabel:" not in shifted.status


def test_declared_balanced_polish_gate_carries_the_substituted_inflation(
) -> None:
    coefficients = np.zeros((3, 3, 3), dtype=complex)

    catalogue = declared_candidates(coefficients)

    balanced = catalogue["scs_warm_1e_9_polish_balanced"]
    strict = catalogue["scs_warm_1e_9_polish_strict"]
    assert balanced.polish_profile == "balanced"
    assert balanced.polish_inflation_shift == BALANCED_INFLATION_SHIFT
    # The strict acceptance test is evaluated on the strict-profile operands
    # the instances carry, so it needs no substitution.
    assert strict.polish_profile == "strict"
    assert strict.polish_inflation_shift == 0.0
    assert WarmStartedScsCandidate(
        "default", "declared default", 1e-9
    ).polish_inflation_shift == 0.0
