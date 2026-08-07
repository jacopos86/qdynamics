from __future__ import annotations

import numpy as np

from paper5.stability.adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    HIDDEN_RELATIVE_MOMENT_KEYS,
    kpd_correlation_velocity_correction,
    matrix_state_to_raw_moment_coordinates,
    uncentered_joint_moment_matrix,
)
from paper5.stability.apcm_moment_projection import state_lower_moments
from paper5.stability.apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
)
from paper5.stability.apcm_projective_guard import (
    FrozenOuterFaceSelector,
    ProjectiveGuardCuttingPlaneSelector,
    canonical_psd_center_cross,
    compile_entrance_source_audit,
    compile_invariant_target_readout,
    center_core_null_directions,
    prefix_restriction,
    prefix_union,
    projective_guard_outer_extension,
    relative_core_moment_matrix,
    relative_core_restriction,
    retained_prefix_restriction,
    select_outer_feasible_projective_guard,
    select_projective_guard,
    unified_glued_moment_matrix,
    unified_guard_dimension,
    unified_core_moment_matrix,
    unified_to_relative_restriction,
)
from paper5.stability.exact_reference import (
    exact_holstein_joint_moment_initial_state,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.matrix_reference import MatrixDimerState
from paper5.stability.moment_hierarchy import THIRD_ORDER_HIERARCHY


def _prepared_state() -> tuple[np.ndarray, dict]:
    exact = exact_holstein_joint_moment_initial_state(
        DimerParameters(lambda_ep=1.5),
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=6,
        canonical_embedding=True,
    )
    raw = matrix_state_to_raw_moment_coordinates(exact.matrix_state)
    _, moments = THIRD_ORDER_HIERARCHY.unpack(exact.hierarchy_coordinates)
    return raw, moments


def test_entrance_source_audit_is_the_independent_fifteen_key_chart() -> None:
    audit = compile_entrance_source_audit()
    assert audit.entrance_keys == ENTRANCE_RELATIVE_MOMENT_KEYS
    assert audit.entrance_rank == 15
    assert set(audit.entrance_keys).isdisjoint(audit.omitted_hidden_keys)
    assert set(audit.entrance_keys).union(audit.omitted_hidden_keys) == set(
        HIDDEN_RELATIVE_MOMENT_KEYS
    )
    assert len(audit.registry_hash) == 64


def test_kpd_source_is_independent_of_every_omitted_hidden_key() -> None:
    parameters = DimerParameters(lambda_ep=1.5)
    exact = exact_holstein_joint_moment_initial_state(
        parameters,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=6,
        canonical_embedding=True,
    )
    _, moments = THIRD_ORDER_HIERARCHY.unpack(exact.hierarchy_coordinates)
    changed = dict(moments)
    rng = np.random.default_rng(20260806)
    entrance = set(ENTRANCE_RELATIVE_MOMENT_KEYS)
    for key in HIDDEN_RELATIVE_MOMENT_KEYS:
        if key not in entrance:
            changed[key] = float(rng.normal())
    np.testing.assert_allclose(
        kpd_correlation_velocity_correction(
            exact.matrix_state, parameters, moments
        ),
        kpd_correlation_velocity_correction(
            exact.matrix_state, parameters, changed
        ),
        atol=0.0,
        rtol=0.0,
    )


def test_invariant_target_readout_has_exact_kernel_and_is_structural() -> None:
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    readout = compile_invariant_target_readout(extension)
    assert readout.rank > 0
    assert readout.rank < len(extension.frontier_keys)
    assert len(readout.nullspace) == len(extension.frontier_keys) - readout.rank
    assert len(readout.registry_hash) == 64
    for vector in readout.nullspace:
        values = np.asarray([float(value) for value in vector])
        np.testing.assert_allclose(readout.matrix @ values, 0.0, atol=0.0, rtol=0.0)


def test_unified_core_has_literal_retained_prefix_and_relative_congruence() -> None:
    raw, moments = _prepared_state()
    core = unified_core_moment_matrix(raw, moments)
    retained = retained_prefix_restriction()
    relative = relative_core_restriction()
    np.testing.assert_allclose(
        retained @ core @ retained.conjugate().T,
        uncentered_joint_moment_matrix(raw),
        atol=2e-13,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        relative.conjugate() @ core @ relative.T,
        relative_core_moment_matrix(moments),
        atol=3e-12,
        rtol=0.0,
    )


def test_prefix_restrictions_compose_exactly() -> None:
    old = ENTRANCE_RELATIVE_MOMENT_KEYS[:4]
    middle = prefix_union(old, ENTRANCE_RELATIVE_MOMENT_KEYS[4:9])
    outer = prefix_union(middle, ENTRANCE_RELATIVE_MOMENT_KEYS[9:])
    old_from_middle = prefix_restriction(old, middle)
    middle_from_outer = prefix_restriction(middle, outer)
    old_from_outer = prefix_restriction(old, outer)
    np.testing.assert_array_equal(
        old_from_outer, old_from_middle @ middle_from_outer
    )


def test_outer_guard_reopens_current_rhs_values_instead_of_promoting_them() -> None:
    current = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    outer = projective_guard_outer_extension(current)
    assert current.active_keys == outer.active_keys
    assert set(current.rhs_frontier_keys).issubset(outer.frontier_keys)
    assert set(current.rhs_frontier_keys).isdisjoint(outer.lower_keys)
    assert outer.dimension == 60
    # Reopening the 15 old RHS-facing values adds them to the formerly
    # coordinate-frozen 223-variable outer frontier.
    assert len(outer.frontier_keys) == 238
    assert unified_guard_dimension(current) == 26
    assert unified_guard_dimension(outer) == 62


def test_glued_unified_gram_restricts_exactly_to_relative_extension() -> None:
    raw, moments = _prepared_state()
    entrance = np.asarray(
        [moments[key] for key in ENTRANCE_RELATIVE_MOMENT_KEYS], dtype=float
    )
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    lower = state_lower_moments(
        raw, entrance, ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    completion = extension.complete(lower)
    assert completion.success, completion.message
    center_cross = np.random.default_rng(81).normal(
        size=(2, extension.dimension - 9)
    ) + 1j * np.random.default_rng(82).normal(
        size=(2, extension.dimension - 9)
    )
    unified = unified_glued_moment_matrix(
        unified_core_moment_matrix(raw, moments),
        completion.moment_matrix,
        center_cross=center_cross,
    )
    restriction = unified_to_relative_restriction(extension.dimension)
    np.testing.assert_allclose(
        restriction.conjugate() @ unified @ restriction.T,
        completion.moment_matrix,
        atol=3e-11,
        rtol=0.0,
    )


def test_canonical_center_cross_preserves_positive_clique_completion() -> None:
    raw, moments = _prepared_state()
    entrance = np.asarray(
        [moments[key] for key in ENTRANCE_RELATIVE_MOMENT_KEYS], dtype=float
    )
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    lower = state_lower_moments(
        raw, entrance, ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    completion = extension.complete(lower)
    assert completion.success, completion.message
    core = unified_core_moment_matrix(raw, moments)
    center_cross = canonical_psd_center_cross(core, completion.moment_matrix)
    unified = unified_glued_moment_matrix(
        core, completion.moment_matrix, center_cross=center_cross
    )
    assert np.linalg.eigvalsh(unified)[0] >= -2e-9
    restriction = unified_to_relative_restriction(extension.dimension)
    np.testing.assert_allclose(
        restriction.conjugate() @ unified @ restriction.T,
        completion.moment_matrix,
        atol=3e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        center_core_null_directions().conjugate().T
        @ center_core_null_directions(),
        np.eye(2),
        atol=2e-15,
        rtol=0.0,
    )


def test_current_projective_selector_accepts_verified_almost_solved_preparation_provisionally() -> None:
    raw, moments = _prepared_state()
    entrance = np.asarray(
        [moments[key] for key in ENTRANCE_RELATIVE_MOMENT_KEYS], dtype=float
    )
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    lower = state_lower_moments(
        raw, entrance, ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    readout = compile_invariant_target_readout(extension)
    selection = select_projective_guard(extension, lower, readout, boxed=True)
    assert selection.target_stage.success
    assert selection.target_stage.provisional
    assert not selection.target_stage.certified
    assert selection.target_stage.acceptance == "provisional_independent_kkt"
    assert "AlmostSolved" in selection.target_stage.status
    assert np.all(np.isfinite(selection.target_image))
    assert max(
        selection.target_stage.independent_feasibility_residual,
        selection.target_stage.stationarity_residual,
        selection.target_stage.complementarity_residual,
        selection.target_stage.relative_duality_gap,
    ) <= 1e-6
    scaled_readout = readout.matrix * extension.frontier_scales[None, :]
    frozen = FrozenOuterFaceSelector.from_witness(
        extension,
        scaled_readout,
        selection.target_metric,
        selection.target_stage,
    )
    repeated = frozen.solve(lower)
    assert repeated.success, repeated.status
    assert frozen.face_rank > 0
    assert repeated.objective <= selection.target_stage.objective + 2e-8
    assert repeated.independent_feasibility_residual <= 1e-8
    scaled_witness = extension.scaled_matrix(
        selection.target_stage.moment_matrix
    )
    _, witness_vectors = np.linalg.eigh(scaled_witness)
    cutting_plane = ProjectiveGuardCuttingPlaneSelector(
        extension=extension,
        scaled_readout=scaled_readout,
        target_metric=selection.target_metric,
        maximum_cuts=32,
    )
    cut_result = cutting_plane.solve(
        lower, seed_eigenvectors=witness_vectors[:, :1]
    )
    assert not cut_result.success
    assert cut_result.cut_count == 32
    assert cut_result.guard.minimum_scaled_eigenvalue < -1e-8


def test_strict_fixture_selector_preserves_only_the_invariant_image() -> None:
    state = MatrixDimerState(
        electron_density=0.5 * np.eye(2, dtype=complex),
        coherent_phonon=np.zeros(2, dtype=complex),
        phonon_density=np.eye(2, dtype=complex),
        anomalous_phonon_density=np.zeros((2, 2), dtype=complex),
        electron_phonon_correlation=np.zeros((2, 2, 2), dtype=complex),
    )
    raw = matrix_state_to_raw_moment_coordinates(state)
    entrance = np.zeros(len(ENTRANCE_RELATIVE_MOMENT_KEYS))
    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    lower = state_lower_moments(
        raw, entrance, ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    readout = compile_invariant_target_readout(extension)
    selection = select_projective_guard(extension, lower, readout, boxed=True)
    assert selection.target_stage.success, selection.target_stage.status
    assert selection.lift_stage.success, selection.lift_stage.status
    assert selection.target_stage.certified
    assert not selection.target_stage.provisional
    scaled = readout.matrix * extension.frontier_scales[None, :]
    cutting_plane = ProjectiveGuardCuttingPlaneSelector(
        extension=extension,
        scaled_readout=scaled,
        target_metric=selection.target_metric,
        maximum_cuts=128,
    )
    cut_result = cutting_plane.solve(lower)
    assert cut_result.success, cut_result.guard.status
    assert cut_result.guard.minimum_scaled_eigenvalue >= -1e-8
    np.testing.assert_allclose(
        scaled @ selection.lift_stage.standardized_values,
        selection.target_image,
        atol=2e-8,
        rtol=0.0,
    )


def test_outer_feasible_selector_restricts_one_common_witness() -> None:
    raw, moments = _prepared_state()
    active = ENTRANCE_RELATIVE_MOMENT_KEYS[:1]
    entrance = np.asarray([moments[key] for key in active], dtype=float)
    extension = SymmetryReducedPositiveExtension(active_keys=active)
    lower = state_lower_moments(raw, entrance, active)
    readout = compile_invariant_target_readout(extension)
    selection = select_outer_feasible_projective_guard(
        extension, lower, readout, boxed=True
    )
    assert selection.success, (
        selection.outer_target_stage.status,
        selection.outer_lift_stage.status,
    )
    assert selection.current_minimum_scaled_eigenvalue >= -1e-7
    scaled_readout = readout.matrix * extension.frontier_scales[None, :]
    np.testing.assert_allclose(
        scaled_readout @ selection.current_standardized_values,
        selection.target_image,
        atol=1e-7,
        rtol=0.0,
    )
