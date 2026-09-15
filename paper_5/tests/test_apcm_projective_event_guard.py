"""Focused tests for the named projective event guard and selector option.

Every test here exercises a named option whose default is the recorded path.
The two bitwise gates are the point of the file: the new
``frontier_weight`` keyword at its default reproduces the declared
lexicographic target stage exactly, and an untriggered event guard reproduces
the per-stage default containment exactly.
"""

from __future__ import annotations

import json
from time import perf_counter

import numpy as np
import pytest

from paper5.stability.adaptive_positive_moment import (
    ENTRANCE_RELATIVE_MOMENT_KEYS,
    matrix_state_to_raw_moment_coordinates,
)
from paper5.stability.apcm_moment_projection import state_lower_moments
from paper5.stability.apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
)
from paper5.stability.apcm_projective_guard import (
    GuardConicResult,
    OuterFeasibleProjectiveGuardSelection,
    compile_invariant_target_readout,
    select_outer_feasible_projective_guard,
)
from paper5.stability.exact_reference import (
    exact_holstein_joint_moment_initial_state,
)
from paper5.stability.hubbard_dimer import DimerParameters
from paper5.stability.moment_hierarchy import THIRD_ORDER_HIERARCHY
from paper5.stability.projected_apcm import (
    FixedDictionaryProjectedAPCM,
    ProjectiveOuterEventGuard,
    ProjectiveOuterEventGuardSettings,
    integrate_projected_apcm_ssprk3,
    prepare_projected_apcm_initial_state,
)

_PARAMETERS = DimerParameters(lambda_ep=1.5)
_SMALL_KEYS = ENTRANCE_RELATIVE_MOMENT_KEYS[:2]


def _small_selector_fixture() -> tuple:
    """A two-key entrance dictionary the outer solve finishes in under a second."""

    exact = exact_holstein_joint_moment_initial_state(
        _PARAMETERS,
        hierarchy=THIRD_ORDER_HIERARCHY,
        phonon_cutoff=6,
        canonical_embedding=True,
    )
    raw = matrix_state_to_raw_moment_coordinates(exact.matrix_state)
    _, moments = THIRD_ORDER_HIERARCHY.unpack(exact.hierarchy_coordinates)
    hidden = np.asarray(
        [moments[key] for key in _SMALL_KEYS], dtype=float
    )
    extension = SymmetryReducedPositiveExtension(active_keys=_SMALL_KEYS)
    lower = state_lower_moments(raw, hidden, _SMALL_KEYS)
    readout = compile_invariant_target_readout(extension)
    return extension, lower, readout


def _entrance_model() -> tuple:
    """The declared fifteen-key entrance dictionary at a small phonon cutoff."""

    model = FixedDictionaryProjectedAPCM(_PARAMETERS)
    initialization = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6
    )
    return model, initialization


def test_frontier_weight_default_reproduces_the_declared_selector_bitwise() -> None:
    extension, lower, readout = _small_selector_fixture()
    declared = select_outer_feasible_projective_guard(
        extension, lower, readout, boxed=True, canonicalize=False
    )
    explicit = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        accept_first_provisional=False,
        frontier_weight=0.0,
    )
    assert declared.witness_source == explicit.witness_source
    np.testing.assert_array_equal(
        declared.target_image, explicit.target_image
    )
    np.testing.assert_array_equal(
        declared.current_standardized_values,
        explicit.current_standardized_values,
    )
    assert (
        declared.outer_target_stage.objective
        == explicit.outer_target_stage.objective
    )


def test_strictly_convex_selector_is_one_solve_that_needs_no_lift() -> None:
    extension, lower, readout = _small_selector_fixture()
    selection = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=True,
        frontier_weight=1.0,
    )
    assert selection.success, selection.outer_target_stage.status
    assert selection.witness_source == "strictly_convex_outer_target_stage"
    assert not selection.canonical_lift
    assert selection.outer_lift_stage is selection.outer_target_stage
    assert selection.current_minimum_scaled_eigenvalue >= -1e-7


def test_strictly_convex_selector_raises_the_target_image_quadratic() -> None:
    extension, lower, readout = _small_selector_fixture()
    lexicographic = select_outer_feasible_projective_guard(
        extension, lower, readout, boxed=True, canonicalize=False
    )
    strict = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        frontier_weight=1.0,
    )
    assert lexicographic.success and strict.success
    assert (
        strict.outer_target_stage.objective
        >= lexicographic.outer_target_stage.objective - 1e-12
    )
    assert float(
        np.linalg.norm(strict.outer_target_stage.standardized_values)
    ) < float(
        np.linalg.norm(lexicographic.outer_target_stage.standardized_values)
    )


def test_coupling_adapter_reproduces_the_unprojected_velocity() -> None:
    model, initialization = _entrance_model()
    guard = ProjectiveOuterEventGuard(model)
    adapter = guard._coupling
    frontier = initialization.completion.frontier_moments
    standardized = np.asarray(
        [
            float(frontier[key]) / scale
            for key, scale in zip(
                model.extension.frontier_keys,
                model.extension.frontier_scales,
                strict=True,
            )
        ],
        dtype=float,
    )
    expected = model.unprojected_velocity_with_frontier(
        0.0, initialization.state, frontier
    )
    observed = adapter.retained_velocity(0.0, initialization.state, standardized)
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-14)
    matrix = adapter.geometry.scaled_unified_matrix(
        initialization.state, standardized
    )
    assert matrix.shape == (model.extension.dimension,) * 2
    assert float(np.linalg.eigvalsh(matrix)[0]) >= -1e-9


def test_quiet_event_guard_reproduces_the_per_stage_default_bitwise() -> None:
    model, initialization = _entrance_model()
    plain = integrate_projected_apcm_ssprk3(
        model,
        initialization.state,
        initial_completion=initialization.completion,
        final_time=0.0025,
        time_step=0.0025,
    )
    guard = ProjectiveOuterEventGuard(model)
    guarded = integrate_projected_apcm_ssprk3(
        model,
        initialization.state,
        initial_completion=initialization.completion,
        final_time=0.0025,
        time_step=0.0025,
        outer_guard=guard,
    )
    assert plain.outer_guard_triggered_stages is None
    assert guarded.outer_guard_triggered_stages.tolist() == [0, 0]
    assert all(not record["triggered"] for record in guard.records)
    np.testing.assert_array_equal(plain.states, guarded.states)
    np.testing.assert_array_equal(
        plain.extension_minimum_eigenvalues,
        guarded.extension_minimum_eigenvalues,
    )


def test_event_guard_tolerates_a_roundoff_depth_without_the_selector() -> None:
    model, initialization = _entrance_model()

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the selector must not run on a quiet stage")

    guard = ProjectiveOuterEventGuard(
        model,
        settings=ProjectiveOuterEventGuardSettings(
            event_gram_floor=1e-18, event_gram_hard_floor=1e-3
        ),
        selector=refuse,
    )
    evaluation = model.evaluate(
        0.0, initialization.state, completion=initialization.completion
    )
    trial = initialization.state + 0.0025 * evaluation.derivative
    guard.contain_stage(0.0, initialization.state, evaluation, trial, 0.0025)
    record = guard.records[0]
    assert record["reason"] == "tolerated"
    assert not record["triggered"]
    assert record["endpoint_unshifted_minimum"] < -1e-18
    assert 0.0 < record["coupling_measure"] < 1e-2


def test_event_guard_fires_below_the_hard_floor_and_carries_the_witness() -> None:
    model, initialization = _entrance_model()
    calls: list[dict] = []

    def stub(extension, lower, readout, **kwargs):  # noqa: ANN001, ANN202
        calls.append(dict(kwargs))
        completion = extension.complete(lower)
        values = np.asarray(
            [
                completion.frontier_moments[key]
                for key in extension.frontier_keys
            ],
            dtype=float,
        ) / extension.frontier_scales
        stage = GuardConicResult(
            success=True,
            status="stub:Solved",
            boxed=True,
            conditioned=False,
            standardized_values=values,
            frontier_moments=completion.frontier_moments,
            moment_matrix=completion.moment_matrix,
            minimum_scaled_eigenvalue=completion.scaled_minimum_eigenvalue,
            primal_residual=0.0,
            dual_residual=0.0,
            objective=0.0,
            iterations=1,
            certified=True,
            provisional=False,
            acceptance="certified",
            independent_feasibility_residual=0.0,
        )
        return OuterFeasibleProjectiveGuardSelection(
            target_image=np.zeros(readout.rank),
            target_metric=np.eye(readout.rank),
            outer_extension=extension,
            outer_target_stage=stage,
            outer_lift_stage=stage,
            current_standardized_values=values,
            current_frontier_moments=completion.frontier_moments,
            current_moment_matrix=completion.moment_matrix,
            current_minimum_scaled_eigenvalue=(
                completion.scaled_minimum_eigenvalue
            ),
            witness_source="stub_outer_witness",
        )

    guard = ProjectiveOuterEventGuard(
        model,
        settings=ProjectiveOuterEventGuardSettings(
            event_gram_floor=1e-18, event_gram_hard_floor=1e-17
        ),
        selector=stub,
    )
    evaluation = model.evaluate(
        0.0, initialization.state, completion=initialization.completion
    )
    trial = initialization.state + 0.0025 * evaluation.derivative
    retraction = guard.contain_stage(
        0.0, initialization.state, evaluation, trial, 0.0025
    )
    record = guard.records[0]
    assert record["triggered"]
    assert record["reason"] == "hard_floor"
    assert record["selector_outcome"] == "outer_feasible_witness"
    assert record["selector_witness_source"] == "stub_outer_witness"
    assert len(calls) == 1
    assert calls[0]["frontier_weight"] == 0.0
    assert calls[0]["accept_first_provisional"] is False
    assert not retraction.applied
    assert retraction.completion.message.startswith(
        "projective_outer_event_guard:"
    )


def test_guard_settings_reject_unordered_floors() -> None:
    with pytest.raises(ValueError):
        ProjectiveOuterEventGuardSettings(
            event_gram_floor=1e-3, event_gram_hard_floor=1e-8
        )
    with pytest.raises(ValueError):
        ProjectiveOuterEventGuardSettings(event_coupling_threshold=0.0)
    with pytest.raises(ValueError):
        ProjectiveOuterEventGuardSettings(frontier_weight=-1.0)


def test_deadline_stops_the_rollout_at_the_steps_it_completed() -> None:
    model, initialization = _entrance_model()
    trajectory = integrate_projected_apcm_ssprk3(
        model,
        initialization.state,
        initial_completion=initialization.completion,
        final_time=0.0075,
        time_step=0.0025,
        deadline=perf_counter() - 1.0,
    )
    assert not trajectory.success
    assert trajectory.completed_steps == 0
    assert trajectory.message.startswith("wall_budget_exceeded")
    assert trajectory.times.size == 1
    assert trajectory.states.shape[0] == 1


def test_driver_selector_audit_writes_a_named_option_summary(tmp_path) -> None:  # noqa: ANN001
    from paper5.stability import apcm_projective_event_guard_analysis as driver

    output = tmp_path / "selector_audit"
    argv = [
        "apcm_projective_event_guard_analysis",
        "--mode",
        "selector-audit",
        "--output-directory",
        str(output),
        "--entrance-keys",
        "2",
        "--phonon-cutoff",
        "6",
        "--frontier-weight",
        "0.0",
        "1.0",
    ]
    import sys

    saved = sys.argv
    sys.argv = argv
    try:
        assert driver.main() == 0
    finally:
        sys.argv = saved
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "selector-audit"
    assert len(summary["solves"]) == 2
    weights = [record["frontier_weight"] for record in summary["solves"]]
    assert weights == [0.0, 1.0]
    assert summary["solves"][0]["witness_source"] == "outer_target_stage"
    assert (
        summary["solves"][1]["witness_source"]
        == "strictly_convex_outer_target_stage"
    )
    assert (output / "selector_audit.npz").exists()
