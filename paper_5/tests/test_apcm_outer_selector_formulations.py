"""Alternative conic formulations of the one-shell outer target stage.

Every test here is on a small or structurally simple fixture; the cutoff-16
correlated preparation and the forced-trigger stage operands are measured in
`agent_guidance/paper-v/overnight/outer-selector-formulation-20260915.md` and
are far too slow for the suite.
"""

from __future__ import annotations

import numpy as np

from paper5.stability.adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    matrix_state_to_raw_moment_coordinates,
)
from paper5.stability.apcm_moment_projection import state_lower_moments
from paper5.stability.apcm_outer_selector_formulations import (
    NAMED_FORMULATIONS,
    OuterTargetFormulation,
    build_outer_target_operands,
    outer_target_operands_from_program,
    psd_affine_admm,
    recorded_program,
    ruiz_congruence,
    save_outer_target_operands,
    solve_named_formulation,
    solve_outer_target_dual,
    solve_outer_target_first_order_polish,
    solve_recorded_outer_target,
    solve_outer_target_with_tightened_retry,
)
from paper5.stability.apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
)
from paper5.stability.apcm_projective_guard import (
    GuardConicResult,
    _solve_explicit_target_qp,
    compile_invariant_target_readout,
    select_projective_guard,
)
from paper5.stability.matrix_reference import MatrixDimerState


def _strict_fixture_state() -> MatrixDimerState:
    """The fixture ``test_apcm_projective_guard`` calls the strict one."""

    return MatrixDimerState(
        electron_density=0.5 * np.eye(2, dtype=complex),
        coherent_phonon=np.zeros(2, dtype=complex),
        phonon_density=np.eye(2, dtype=complex),
        anomalous_phonon_density=np.zeros((2, 2), dtype=complex),
        electron_phonon_correlation=np.zeros((2, 2, 2), dtype=complex),
    )


def _strict_fixture(keys: int):
    raw = matrix_state_to_raw_moment_coordinates(_strict_fixture_state())
    active = ENTRANCE_RELATIVE_MOMENT_KEYS[:keys]
    extension = SymmetryReducedPositiveExtension(active_keys=active)
    lower = state_lower_moments(raw, np.zeros(len(active)), active)
    readout = compile_invariant_target_readout(extension)
    return extension, lower, readout


def _current_stage_operands():
    """The strict fixture's own current-cone target stage, 15 entrance keys.

    These are the operands ``select_projective_guard`` hands
    ``_solve_explicit_target_qp`` at that fixture.
    """

    extension, lower, readout = _strict_fixture(
        len(ENTRANCE_RELATIVE_MOMENT_KEYS)
    )
    scaled_readout = readout.matrix * extension.frontier_scales[None, :]
    target_metric = np.linalg.inv(scaled_readout @ scaled_readout.T)
    return extension, lower, readout, scaled_readout, target_metric


def test_default_formulation_states_the_recorded_program() -> None:
    assert OuterTargetFormulation().as_record() == {
        "eliminate_image_equality": False,
        "absorb_whitening": False,
        "gram_congruence": "none",
        "frontier_column_scaling": "none",
        "psd_inflation": 0.0,
        "homogeneous_frontier_scale": 1.0,
        "frontier_weight": 0.0,
        "retry_profile": 0,
        "equilibration_sweeps": 20,
        "time_limit_seconds": None,
    }
    assert NAMED_FORMULATIONS["recorded_kernel"] == OuterTargetFormulation()


def test_recorded_kernel_reproduces_the_recorded_target_stage_bitwise() -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    recorded = solve_recorded_outer_target(operands)
    kernel = solve_named_formulation(operands, "recorded_kernel")
    assert recorded.result.status == kernel.result.status
    np.testing.assert_array_equal(
        recorded.result.standardized_values,
        kernel.result.standardized_values,
    )
    assert recorded.result.objective == kernel.result.objective
    assert recorded.result.acceptance == kernel.result.acceptance
    assert (
        recorded.result.minimum_scaled_eigenvalue
        == kernel.result.minimum_scaled_eigenvalue
    )


def test_recorded_program_matches_the_recorded_variable_and_row_order() -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    quadratic, linear, matrix, rhs, cones = recorded_program(operands)
    count = operands.frontier_count
    target_count = operands.target_count
    assert quadratic.shape == (count + target_count, count + target_count)
    assert linear.shape == (count + target_count,)
    psd_rows = operands.coefficient_vectors.shape[0]
    assert matrix.shape == (psd_rows + 2 * count + target_count, count + target_count)
    np.testing.assert_array_equal(rhs[:psd_rows], operands.base_vector)
    np.testing.assert_array_equal(
        rhs[psd_rows : psd_rows + 2 * count], np.ones(2 * count)
    )
    np.testing.assert_array_equal(rhs[psd_rows + 2 * count :], np.zeros(target_count))
    assert len(cones) == 3


def test_target_image_quadratic_is_an_orthogonal_projector() -> None:
    """``R^T M R`` is the projector onto the row space of the readout.

    The target metric is the inverse of the readout covariance ``R R^T``, so
    the invariant target image quadratic is exactly the orthogonal projector
    onto the row space of ``R``, and the whitened readout ``U R`` has
    orthonormal rows.  This is the structural fact the whitened formulation
    uses, and it holds on both the current and the one-shell outer cone.
    """

    from scipy import linalg

    _, _, readout, scaled_readout, target_metric = _current_stage_operands()
    projector = scaled_readout.T @ target_metric @ scaled_readout
    eigenvalues = np.linalg.eigvalsh(projector)
    assert eigenvalues.shape[0] == scaled_readout.shape[1]
    np.testing.assert_allclose(
        eigenvalues[-readout.rank :], np.ones(readout.rank), atol=1e-10
    )
    np.testing.assert_allclose(
        eigenvalues[: -readout.rank],
        np.zeros(eigenvalues.shape[0] - readout.rank),
        atol=1e-10,
    )
    whitened = linalg.cholesky(target_metric, lower=False) @ scaled_readout
    np.testing.assert_allclose(
        whitened @ whitened.T, np.eye(readout.rank), atol=1e-10
    )


def test_tightened_retry_target_stage_on_the_strict_fixture() -> None:
    """The winning formulation, on the fixture the guard's tests certify.

    ``select_projective_guard`` certifies this fixture, so the comparison is
    against a certified witness: the tightened-retry stage must be accepted
    under the unchanged exploratory acceptance rule, must select the same
    invariant target image, and must return the ``GuardConicResult`` shape.
    """

    extension, lower, readout, scaled_readout, target_metric = (
        _current_stage_operands()
    )
    selection = select_projective_guard(extension, lower, readout, boxed=True)
    assert selection.target_stage.certified

    result = solve_outer_target_with_tightened_retry(
        extension, lower, scaled_readout, target_metric, boxed=True
    )
    assert isinstance(result, GuardConicResult)
    assert result.success, result.status
    assert result.acceptance in ("certified", "provisional_independent_kkt")
    assert set(result.frontier_moments) == set(extension.frontier_keys)
    assert result.standardized_values.shape == (
        len(extension.frontier_keys),
    )
    assert result.moment_matrix.shape == (
        extension.dimension,
        extension.dimension,
    )
    assert result.minimum_scaled_eigenvalue >= -1e-8
    np.testing.assert_allclose(
        scaled_readout @ result.standardized_values,
        selection.target_image,
        atol=1e-7,
        rtol=0.0,
    )
    assert abs(result.objective - selection.target_stage.objective) <= 1e-9


def test_tightened_retry_target_stage_reaches_the_one_shell_outer_cone() -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    recorded = solve_recorded_outer_target(operands)
    for ladder in (True, False):
        for whiten in (False, True):
            result = solve_outer_target_with_tightened_retry(
                operands.extension,
                operands.lower_moments,
                operands.scaled_readout,
                operands.target_metric,
                boxed=True,
                ladder=ladder,
                whiten_image=whiten,
            )
            assert result.success, result.status
            assert result.minimum_scaled_eigenvalue >= -1e-8
            assert abs(result.objective - recorded.result.objective) <= 1e-8


def test_the_recorded_retry_ladder_does_not_fire_on_numerical_error() -> None:
    """The one line that keeps the tightened profile from running.

    ``_solve_explicit_target_qp`` retries at its tightened profile only for
    the three statuses below, so a first pass that ends in ``NumericalError``
    is returned as it stands.  This is the measured reason the recorded outer
    target stage fails at the cutoff-16 correlated preparation while the same
    program at ``retry_profile=2`` is accepted.
    """

    import inspect

    source = inspect.getsource(_solve_explicit_target_qp)
    assert (
        'retry_statuses = {"AlmostSolved", "InsufficientProgress", '
        '"MaxIterations"}' in source
    )
    assert "NumericalError" not in source


def test_psd_inflation_acceptance_is_read_on_the_uninflated_gram() -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    outcome = solve_named_formulation(operands, "psd_inflation_1em8")
    frontier = np.asarray(
        [
            outcome.result.frontier_moments[key]
            for key in operands.extension.frontier_keys
        ],
        dtype=float,
    )
    matrix = operands.extension.matrix(
        operands.lower_moments,
        dict(zip(operands.extension.frontier_keys, frontier, strict=True)),
    )
    minimum = float(
        np.linalg.eigvalsh(operands.extension.scaled_matrix(matrix))[0]
    )
    assert outcome.result.minimum_scaled_eigenvalue == minimum
    assert outcome.options["psd_inflation"] == 1e-8


def test_homogeneous_box_rescaling_returns_the_same_witness() -> None:
    """The standardized box has unit half-widths in every coordinate.

    The recorded program's box is ``|z_i| <= 1``, so rescaling each frontier
    variable by its own box half-width is the identity; the only content of a
    homogeneous rescaling is one common factor, which is what the named
    option applies.
    """

    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    _, _, _, rhs, _ = recorded_program(operands)
    psd_rows = operands.coefficient_vectors.shape[0]
    box = rhs[psd_rows : psd_rows + 2 * operands.frontier_count]
    np.testing.assert_array_equal(box, np.ones_like(box))

    recorded = solve_named_formulation(operands, "recorded_kernel")
    rescaled = solve_named_formulation(operands, "box_rescaled_1em1")
    assert abs(recorded.result.objective - rescaled.result.objective) <= 1e-8


def test_ruiz_congruence_equilibrates_a_symmetric_magnitude() -> None:
    magnitude = np.array([[1.0, 1e-3], [1e-3, 1e6]])
    factors = ruiz_congruence(magnitude, sweeps=50)
    scaled = factors[:, None] * magnitude * factors[None, :]
    np.testing.assert_allclose(np.max(scaled, axis=1), np.ones(2), atol=1e-8)


def test_first_order_pass_respects_the_box_and_the_cone() -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    first = psd_affine_admm(operands, iterations=200)
    assert first.iterations >= 1
    assert float(np.max(np.abs(first.standardized_values))) <= 1.0 + 1e-6
    assert first.minimum_scaled_eigenvalue >= -1e-6


def test_first_order_polish_and_dual_agree_with_the_recorded_objective() -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(extension, lower, readout)
    recorded = solve_recorded_outer_target(operands)
    polish = solve_outer_target_first_order_polish(operands)
    dual = solve_outer_target_dual(operands)
    assert abs(polish.result.objective - recorded.result.objective) <= 1e-8
    assert abs(dual.result.objective - recorded.result.objective) <= 1e-8
    assert dual.notes["frontier_point_taken_from"] in (
        "zero_cone_multiplier",
        "negated_zero_cone_multiplier",
        "dual_program_primal_block",
    )


def test_operands_round_trip_to_a_saved_run_directory(tmp_path) -> None:
    extension, lower, readout = _strict_fixture(1)
    operands = build_outer_target_operands(
        extension, lower, readout, name="fixture"
    )
    save_outer_target_operands(tmp_path, operands)
    stored = np.load(tmp_path / "operands.npz")
    np.testing.assert_array_equal(
        stored["scaled_readout"], operands.scaled_readout
    )
    np.testing.assert_array_equal(
        stored["target_metric"], operands.target_metric
    )
    np.testing.assert_array_equal(stored["real_base"], operands.real_base)
    assert (tmp_path / "operands.json").exists()


def test_operands_from_program_take_the_recorded_four_operands() -> None:
    extension, lower, readout = _strict_fixture(1)
    built = build_outer_target_operands(extension, lower, readout)
    direct = outer_target_operands_from_program(
        built.extension,
        built.lower_moments,
        built.scaled_readout,
        built.target_metric,
    )
    np.testing.assert_array_equal(direct.real_base, built.real_base)
    np.testing.assert_array_equal(
        direct.coefficient_vectors, built.coefficient_vectors
    )
    recorded = _solve_explicit_target_qp(
        built.extension,
        built.lower_moments,
        built.scaled_readout,
        built.target_metric,
        boxed=True,
    )
    kernel = solve_named_formulation(direct, "recorded_kernel")
    np.testing.assert_array_equal(
        recorded.standardized_values, kernel.result.standardized_values
    )
