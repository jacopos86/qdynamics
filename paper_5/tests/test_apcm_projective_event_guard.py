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
from paper5.stability.apcm_moment_projection import (
    SymmetryReducedAPCMGeometry,
    state_lower_moments,
)
from paper5.stability.apcm_positive_extension import (
    SymmetryReducedPositiveExtension,
)
from paper5.stability.apcm_projective_guard import (
    GuardConicResult,
    OuterFeasibleProjectiveGuardSelection,
    _solve_explicit_target_qp,
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
    ProjectedAPCMStagePolicy,
    ProjectiveOuterEventGuard,
    ProjectiveOuterEventGuardSettings,
    integrate_projected_apcm_ssprk3,
    prepare_projected_apcm_initial_state,
    resolve_outer_target_selector,
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


def test_stage_policy_default_reproduces_the_recorded_rollout_bitwise() -> None:
    """The default stage policy is the recorded path, array for array."""

    model, initialization = _entrance_model()
    recorded = integrate_projected_apcm_ssprk3(
        model,
        initialization.state,
        initial_completion=initialization.completion,
        final_time=0.005,
        time_step=0.0025,
    )
    explicit = FixedDictionaryProjectedAPCM(
        _PARAMETERS, stage_policy=ProjectedAPCMStagePolicy()
    )
    spelled = integrate_projected_apcm_ssprk3(
        explicit,
        initialization.state,
        initial_completion=initialization.completion,
        final_time=0.005,
        time_step=0.0025,
    )
    np.testing.assert_array_equal(recorded.states, spelled.states)
    np.testing.assert_array_equal(
        recorded.extension_minimum_eigenvalues,
        spelled.extension_minimum_eigenvalues,
    )
    np.testing.assert_array_equal(
        recorded.stage_retractions, spelled.stage_retractions
    )
    assert explicit.stage_policy_counters == {
        "carried_completion_accepted": 0,
        "carried_completion_rejected": 0,
        "retraction_suppressed": 0,
        "quiet_containment_skipped": 0,
        "completion_retry_accepted": 0,
        "completion_retry_exhausted": 0,
    }
    assert explicit.completion_ladder == ()


def test_disabled_stage_retraction_keeps_the_trial_stage() -> None:
    model = FixedDictionaryProjectedAPCM(
        _PARAMETERS,
        stage_policy=ProjectedAPCMStagePolicy(stage_retraction=False),
    )
    initialization = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6
    )
    evaluation = model.evaluate(
        0.0, initialization.state, completion=initialization.completion
    )
    trial = initialization.state + 0.0025 * evaluation.derivative
    retraction = model.contain_stage(
        trial,
        warm_frontier=evaluation.targets.completion.frontier_moments,
        retained_metric=evaluation.projection.retained_metric,
        auxiliary_metric=evaluation.projection.auxiliary_metric,
    )
    assert not retraction.applied
    assert retraction.status == "stage_retraction_disabled"
    assert model.stage_policy_counters["retraction_suppressed"] == 1
    np.testing.assert_array_equal(
        np.concatenate(
            (retraction.raw_coordinates, retraction.hidden_values)
        ),
        trial,
    )


def test_disabled_quiet_containment_skips_the_feasibility_test() -> None:
    model = FixedDictionaryProjectedAPCM(
        _PARAMETERS,
        stage_policy=ProjectedAPCMStagePolicy(quiet_containment=False),
    )
    initialization = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6
    )
    evaluation = model.evaluate(
        0.0, initialization.state, completion=initialization.completion
    )
    trial = initialization.state + 0.0025 * evaluation.derivative
    quiet = model.contain_stage(
        trial,
        warm_frontier=evaluation.targets.completion.frontier_moments,
        retained_metric=evaluation.projection.retained_metric,
        auxiliary_metric=evaluation.projection.auxiliary_metric,
    )
    assert quiet.status == "quiet_containment_disabled"
    assert model.stage_policy_counters["quiet_containment_skipped"] == 1
    triggered = model.contain_stage(
        trial,
        warm_frontier=evaluation.targets.completion.frontier_moments,
        retained_metric=evaluation.projection.retained_metric,
        auxiliary_metric=evaluation.projection.auxiliary_metric,
        stage_triggered=True,
    )
    assert triggered.status == "already_feasible"
    assert model.stage_policy_counters["quiet_containment_skipped"] == 1


def test_carried_completion_reuses_the_previous_frontier_when_feasible() -> None:
    model = FixedDictionaryProjectedAPCM(
        _PARAMETERS,
        stage_policy=ProjectedAPCMStagePolicy(carried_completion=True),
    )
    initialization = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6
    )
    evaluation = model.evaluate(
        0.0, initialization.state, completion=initialization.completion
    )
    trial = initialization.state + 0.0025 * evaluation.derivative
    carried = evaluation.targets.completion.frontier_moments
    retraction = model.contain_stage(
        trial,
        warm_frontier=carried,
        retained_metric=evaluation.projection.retained_metric,
        auxiliary_metric=evaluation.projection.auxiliary_metric,
    )
    counters = model.stage_policy_counters
    assert (
        counters["carried_completion_accepted"]
        + counters["carried_completion_rejected"]
        == 1
    )
    if counters["carried_completion_accepted"]:
        assert retraction.completion.message == "carried_completion"
        assert retraction.completion.iterations == 0
        for key, value in carried.items():
            assert retraction.completion.frontier_moments[key] == value
    else:
        assert retraction.completion.message != "carried_completion"


def test_stage_policy_rejects_an_unknown_completion_start() -> None:
    with pytest.raises(ValueError):
        ProjectedAPCMStagePolicy(completion_start="hot")


def test_cold_completion_reaches_the_same_lift_as_the_warm_start() -> None:
    """The declared completion is determined by its lower moments.

    The recorded path warm starts every stage completion from the previous
    stage's frontier moments.  Solving the same lower moments from the origin
    must reach the same selected frontier to the declared conic tolerance, or
    the warm start is doing scientific work rather than saving iterations.
    """

    warm_model = FixedDictionaryProjectedAPCM(_PARAMETERS)
    cold_model = FixedDictionaryProjectedAPCM(
        _PARAMETERS,
        stage_policy=ProjectedAPCMStagePolicy(completion_start="cold"),
    )
    initialization = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6
    )
    evaluation = warm_model.evaluate(
        0.0, initialization.state, completion=initialization.completion
    )
    trial = initialization.state + 0.0025 * evaluation.derivative
    arguments = dict(
        warm_frontier=evaluation.targets.completion.frontier_moments,
        retained_metric=evaluation.projection.retained_metric,
        auxiliary_metric=evaluation.projection.auxiliary_metric,
    )
    warm = warm_model.contain_stage(trial, **arguments)
    cold = cold_model.contain_stage(trial, **arguments)
    keys = tuple(warm_model.extension.frontier_keys)
    warm_values = np.asarray(
        [warm.completion.frontier_moments[key] for key in keys], dtype=float
    )
    cold_values = np.asarray(
        [cold.completion.frontier_moments[key] for key in keys], dtype=float
    )
    np.testing.assert_allclose(cold_values, warm_values, rtol=0.0, atol=1e-7)


def test_frontier_metric_default_reproduces_the_identity_term_bitwise() -> None:
    extension, lower, readout = _small_selector_fixture()
    declared = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        frontier_weight=1.0,
    )
    spelled = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        frontier_weight=1.0,
        frontier_metric="identity",
        warm_refinement_passes=0,
        exploratory_acceptance_tolerance=None,
    )
    np.testing.assert_array_equal(declared.target_image, spelled.target_image)
    np.testing.assert_array_equal(
        declared.current_standardized_values,
        spelled.current_standardized_values,
    )
    assert (
        declared.outer_target_stage.objective
        == spelled.outer_target_stage.objective
    )


def test_readout_diagonal_frontier_metric_stays_on_the_outer_cone() -> None:
    extension, lower, readout = _small_selector_fixture()
    scaled = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        frontier_weight=1.0,
        frontier_metric="readout_diagonal",
    )
    assert scaled.witness_source in {
        "strictly_convex_outer_target_stage",
        "none",
    }
    if scaled.success:
        assert scaled.current_minimum_scaled_eigenvalue >= -1e-7


def test_warm_refinement_pass_reports_the_accumulated_point() -> None:
    extension, lower, readout = _small_selector_fixture()
    single = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        frontier_weight=1.0,
    )
    refined = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        frontier_weight=1.0,
        warm_refinement_passes=1,
    )
    assert "+warm" in refined.outer_target_stage.status
    assert np.isfinite(refined.outer_target_stage.relative_duality_gap)
    assert isinstance(refined.outer_target_stage.dual_residual, float)
    assert isinstance(refined.outer_target_stage.primal_residual, float)
    assert np.isfinite(refined.outer_target_stage.dual_residual)
    np.testing.assert_allclose(
        refined.outer_target_stage.standardized_values,
        single.outer_target_stage.standardized_values,
        rtol=0.0,
        atol=1e-6,
    )


def test_exploratory_acceptance_tolerance_only_widens_acceptance() -> None:
    from paper5.stability.apcm_projective_guard import (
        _IndependentKKTDiagnostics,
        _guard_acceptance,
    )

    diagnostics = _IndependentKKTDiagnostics(
        feasibility=1e-10,
        stationarity=1e-10,
        complementarity=1e-10,
        relative_gap=1.119e-6,
    )
    default = _guard_acceptance(
        "AlmostSolved", strict_success=False, diagnostics=diagnostics
    )
    assert default == (False, False, False, "rejected")
    widened = _guard_acceptance(
        "AlmostSolved",
        strict_success=False,
        diagnostics=diagnostics,
        exploratory_acceptance_tolerance=2e-6,
    )
    assert widened == (True, False, True, "provisional_independent_kkt")
    with pytest.raises(ValueError):
        _guard_acceptance(
            "AlmostSolved",
            strict_success=False,
            diagnostics=diagnostics,
            exploratory_acceptance_tolerance=0.0,
        )


def test_driver_saves_the_horizon_a_failed_rollout_reached(  # noqa: ANN001
    tmp_path,
    monkeypatch,
) -> None:
    """A forced stage failure leaves a scoreable partial trajectory behind.

    The failure is forced in the fourth containment, which is the first
    stage of the second step, so exactly one step is accepted.  The driver
    must then write the partial trajectory, the last accepted state, the
    failing stage's lower moments and the failure message, and report a
    nonzero exit status.
    """

    from paper5.stability import apcm_projective_event_guard_analysis as driver
    from paper5.stability.projected_apcm import ProjectedAPCMFailure

    calls = {"retract": 0}
    recorded = SymmetryReducedAPCMGeometry.retract_stage

    def forced(self, *args, **kwargs):  # noqa: ANN001, ANN202
        calls["retract"] += 1
        if calls["retract"] == 4:
            raise ProjectedAPCMFailure("forced test stage failure")
        return recorded(self, *args, **kwargs)

    monkeypatch.setattr(SymmetryReducedAPCMGeometry, "retract_stage", forced)

    output = tmp_path / "forced_failure"
    argv = [
        "apcm_projective_event_guard_analysis",
        "--mode",
        "rollout",
        "--output-directory",
        str(output),
        "--phonon-cutoff",
        "6",
        "--final-time",
        "0.0075",
        "--time-step",
        "0.0025",
        "--maximum-subdivisions",
        "0",
    ]
    import sys

    saved = sys.argv
    sys.argv = argv
    try:
        assert driver.main() == 1
    finally:
        sys.argv = saved

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert "forced test stage failure" in summary["failure"]["message"]
    assert summary["failure"]["accepted_steps"] == 1
    assert summary["failure"]["last_accepted_time"] == pytest.approx(0.0025)
    assert summary["failure"]["stage"]["stage"] == "contain"
    assert summary["failure"]["stage"]["lower_moment_count"] > 0
    assert summary["integration"]["success"] is False
    assert summary["accuracy"]["all_coordinate_scalar_rms_error"] >= 0.0

    trajectory = np.load(output / "trajectory.npz")
    assert trajectory["times"].tolist() == [0.0, 0.0025]
    assert trajectory["exact_archive_coordinates"].shape[0] == 2
    autonomous = np.load(output / "autonomous_trajectory.npz")
    assert autonomous["states"].shape[0] == 2

    stage_failure = np.load(output / "stage_failure.npz", allow_pickle=False)
    assert str(stage_failure["stage"]) == "contain"
    assert stage_failure["lower_moment_values"].size == (
        summary["failure"]["stage"]["lower_moment_count"]
    )
    assert stage_failure["state"].size == autonomous["states"].shape[1]


def test_driver_continues_a_rollout_from_a_stored_state(tmp_path) -> None:  # noqa: ANN001
    """``--initial-state-file`` starts the declared model at the stored time."""

    from paper5.stability import apcm_projective_event_guard_analysis as driver

    stored = tmp_path / "stored.npz"
    _, initialization = _entrance_model()
    np.savez_compressed(
        stored,
        times=np.asarray([0.0]),
        states=np.asarray([initialization.state]),
    )

    output = tmp_path / "continued"
    argv = [
        "apcm_projective_event_guard_analysis",
        "--mode",
        "rollout",
        "--output-directory",
        str(output),
        "--phonon-cutoff",
        "6",
        "--initial-state-file",
        str(stored),
        "--initial-time",
        "0.05",
        "--final-time",
        "0.055",
        "--time-step",
        "0.0025",
    ]
    import sys

    saved = sys.argv
    sys.argv = argv
    try:
        assert driver.main() == 0
    finally:
        sys.argv = saved
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["continuation"]["stored_final_time"] == pytest.approx(0.0)
    assert summary["integration"]["initial_time"] == pytest.approx(0.05)
    assert summary["integration"]["last_time"] == pytest.approx(0.055)
    times = np.load(output / "trajectory.npz")["times"]
    assert times.tolist() == pytest.approx([0.05, 0.0525, 0.055])
    exact = np.load(output / "trajectory.npz")["exact_archive_coordinates"]
    assert exact.shape[0] == 3


def test_driver_declared_settings_reach_the_model_that_runs(tmp_path) -> None:  # noqa: ANN001
    """The named completion and face flags are the ones the rollout uses."""

    from paper5.stability import apcm_projective_event_guard_analysis as driver

    output = tmp_path / "declared_settings"
    argv = [
        "apcm_projective_event_guard_analysis",
        "--mode",
        "rollout",
        "--output-directory",
        str(output),
        "--phonon-cutoff",
        "6",
        "--final-time",
        "0.0025",
        "--time-step",
        "0.0025",
        "--face-policy",
        "inflated",
        "--face-inflation-multiple",
        "250.0",
        "--completion-logdet-shift",
        "1e-9",
    ]
    import sys

    saved = sys.argv
    sys.argv = argv
    try:
        assert driver.main() == 0
    finally:
        sys.argv = saved
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["projection_settings"]["face_policy"] == "inflated"
    assert summary["projection_settings"]["face_inflation_multiple"] == 250.0
    assert summary["extension_settings"]["logdet_shift"] == 1e-9
    assert summary["extension_settings"]["logdet_weight"] == 1.0
    assert summary["face_policy_counters"]["identified"] > 0
    assert summary["face_policy_counters"]["refused"] == 0


def test_driver_records_a_failure_that_carries_no_accepted_step(  # noqa: ANN001
    tmp_path,
    monkeypatch,
) -> None:
    """An exception with no partial trajectory still leaves the message."""

    from paper5.stability import apcm_projective_event_guard_analysis as driver

    def raise_unexpected(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise RuntimeError("forced test failure before the first step")

    monkeypatch.setattr(
        driver, "integrate_projected_apcm_ssprk3", raise_unexpected
    )
    output = tmp_path / "no_step"
    argv = [
        "apcm_projective_event_guard_analysis",
        "--mode",
        "rollout",
        "--output-directory",
        str(output),
        "--phonon-cutoff",
        "6",
        "--final-time",
        "0.0025",
        "--time-step",
        "0.0025",
    ]
    import sys

    saved = sys.argv
    sys.argv = argv
    try:
        assert driver.main() == 1
    finally:
        sys.argv = saved
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["failure"]["accepted_steps"] == 0
    assert "forced test failure" in summary["failure"]["message"]
    assert summary["failure"]["stage"] is None
    assert summary["integration"]["success"] is False
    assert not (output / "trajectory.npz").exists()


def test_completion_retry_ladder_takes_the_first_certified_result() -> None:
    """A declared retry answers a completion the declared settings refuse."""

    from paper5.stability.apcm_positive_extension import APCMExtensionResult
    from paper5.stability.projected_apcm import build_completion_ladder

    model = FixedDictionaryProjectedAPCM(
        _PARAMETERS,
        stage_policy=ProjectedAPCMStagePolicy(
            completion_retry_shifts=(1e-8,),
            completion_retry_weights=(1e-1,),
        ),
    )
    assert [
        extension.settings.logdet_shift for extension in model.completion_ladder
    ] == [1e-8, 1e-10]
    assert [
        extension.settings.logdet_weight
        for extension in model.completion_ladder
    ] == [1.0, 1e-1]
    assert model.geometry.completion_ladder == model.completion_ladder

    initialization = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6
    )
    raw, hidden = (
        initialization.state[: -len(model.active_keys)],
        initialization.state[-len(model.active_keys) :],
    )
    lower = state_lower_moments(raw, hidden, model.active_keys)
    declared = initialization.completion

    def refuse(moments, *, warm_frontier=None):  # noqa: ANN001, ANN202
        return APCMExtensionResult(
            lower_moments=declared.lower_moments,
            frontier_moments=declared.frontier_moments,
            moment_matrix=declared.moment_matrix,
            scaled_minimum_eigenvalue=declared.scaled_minimum_eigenvalue,
            minimum_eigenvalue=declared.minimum_eigenvalue,
            objective=declared.objective,
            kkt_backward_error=float("inf"),
            iterations=0,
            success=False,
            message="forced test refusal",
        )

    model.extension.complete = refuse
    result = model._select_stage_completion(lower, declared.frontier_moments)
    assert result.success
    assert model.stage_policy_counters["completion_retry_accepted"] == 1
    assert model.stage_policy_counters["completion_retry_exhausted"] == 0


def test_the_default_stage_policy_declares_no_completion_retry() -> None:
    model = FixedDictionaryProjectedAPCM(_PARAMETERS)
    assert model.completion_ladder == ()
    assert model.geometry.completion_ladder == ()
    assert model.stage_policy.is_default
    assert model.stage_policy_counters["completion_retry_accepted"] == 0

def test_the_default_event_guard_selector_is_the_recorded_selector() -> None:
    model, _ = _entrance_model()
    assert ProjectiveOuterEventGuardSettings().outer_target_solver == "recorded"
    assert (
        ProjectiveOuterEventGuard(model).selector
        is select_outer_feasible_projective_guard
    )
    assert (
        resolve_outer_target_selector(ProjectiveOuterEventGuardSettings())
        is select_outer_feasible_projective_guard
    )


def test_guard_settings_reject_an_unknown_outer_target_solver() -> None:
    with pytest.raises(ValueError):
        ProjectiveOuterEventGuardSettings(outer_target_solver="whatever")
    with pytest.raises(ValueError):
        ProjectiveOuterEventGuardSettings(
            outer_target_time_limit_seconds=0.0
        )


def test_an_injected_target_solver_replaces_only_the_target_stage() -> None:
    extension, lower, readout = _small_selector_fixture()
    declared = select_outer_feasible_projective_guard(
        extension, lower, readout, boxed=True, canonicalize=False
    )
    operands: list[tuple] = []

    def relay(outer, outer_lower, scaled_readout, target_metric, **kwargs):  # noqa: ANN001, ANN202
        operands.append(
            (outer, outer_lower, scaled_readout, target_metric, dict(kwargs))
        )
        return _solve_explicit_target_qp(
            outer,
            outer_lower,
            scaled_readout,
            target_metric,
            **kwargs,
        )

    relayed = select_outer_feasible_projective_guard(
        extension,
        lower,
        readout,
        boxed=True,
        canonicalize=False,
        target_solver=relay,
    )
    assert len(operands) == 1
    outer, outer_lower, scaled_readout, target_metric, kwargs = operands[0]
    assert len(outer.frontier_keys) > len(extension.frontier_keys)
    assert outer_lower is lower
    assert scaled_readout.shape == (readout.rank, len(outer.frontier_keys))
    assert target_metric.shape == (readout.rank, readout.rank)
    assert kwargs == {"boxed": True, "frontier_weight": 0.0}
    np.testing.assert_array_equal(declared.target_image, relayed.target_image)
    np.testing.assert_array_equal(
        declared.current_standardized_values,
        relayed.current_standardized_values,
    )
    assert (
        declared.outer_target_stage.objective
        == relayed.outer_target_stage.objective
    )
    assert declared.witness_source == relayed.witness_source


def test_a_supplied_target_solver_refuses_the_recorded_named_options() -> None:
    extension, lower, readout = _small_selector_fixture()

    def unused(*args: object, **kwargs: object) -> None:
        raise AssertionError("the refused call must not reach the solver")

    for named in (
        {"accept_first_provisional": True},
        {"frontier_metric": "readout_diagonal"},
        {"warm_refinement_passes": 1},
        {"exploratory_acceptance_tolerance": 2e-6},
    ):
        with pytest.raises(ValueError):
            select_outer_feasible_projective_guard(
                extension,
                lower,
                readout,
                boxed=True,
                canonicalize=False,
                target_solver=unused,
                **named,
            )


def test_the_named_tightened_retry_selector_accepts_the_same_image() -> None:
    extension, lower, readout = _small_selector_fixture()
    declared = select_outer_feasible_projective_guard(
        extension, lower, readout, boxed=True, canonicalize=False
    )
    selector = resolve_outer_target_selector(
        ProjectiveOuterEventGuardSettings(
            outer_target_solver="tightened_retry"
        )
    )
    assert selector is not select_outer_feasible_projective_guard
    tightened = selector(
        extension, lower, readout, boxed=True, canonicalize=False
    )
    assert tightened.success
    assert tightened.witness_source == declared.witness_source
    np.testing.assert_allclose(
        tightened.target_image, declared.target_image, rtol=0.0, atol=1e-9
    )
    assert (
        tightened.outer_target_stage.minimum_scaled_eigenvalue
        >= declared.outer_target_stage.minimum_scaled_eigenvalue - 1e-9
    )


def test_a_forced_trigger_stage_carries_the_witness_through_the_seam() -> None:
    model, initialization = _entrance_model()
    seen: list[dict] = []

    def stub_target_solver(  # noqa: ANN202
        outer, outer_lower, scaled_readout, target_metric, **kwargs
    ):  # noqa: ANN001
        seen.append(
            {
                "outer_frontier": len(outer.frontier_keys),
                "outer_dimension": outer.dimension,
                "readout_rank": scaled_readout.shape[0],
                **kwargs,
            }
        )
        completion = model.extension.complete(outer_lower)
        index = {
            key: position
            for position, key in enumerate(outer.frontier_keys)
        }
        values = np.zeros(len(outer.frontier_keys))
        for key, scale in zip(
            model.extension.frontier_keys,
            model.extension.frontier_scales,
            strict=True,
        ):
            values[index[key]] = completion.frontier_moments[key] / scale
        return GuardConicResult(
            success=True,
            status="stub_target:Solved",
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

    def selector(current, lower, readout=None, **keywords):  # noqa: ANN001, ANN202
        return select_outer_feasible_projective_guard(
            current,
            lower,
            readout,
            target_solver=stub_target_solver,
            **keywords,
        )

    guard = ProjectiveOuterEventGuard(
        model,
        settings=ProjectiveOuterEventGuardSettings(
            event_gram_floor=1e-18,
            event_gram_hard_floor=1e-17,
            outer_target_solver="tightened_retry",
        ),
        selector=selector,
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
    assert record["selector_witness_source"] == "outer_target_stage"
    assert len(seen) == 1
    assert seen[0]["outer_frontier"] > len(model.extension.frontier_keys)
    assert seen[0]["boxed"] is True
    assert seen[0]["frontier_weight"] == 0.0
    assert not retraction.applied


def test_the_exact_start_contracts_the_state_the_preparation_agrees_with() -> None:
    """The named exact start at a small time is the prepared state."""

    from paper5.stability.apcm_projective_event_guard_analysis import (
        _exact_contracted_initialization,
    )
    from paper5.stability.projected_apcm import build_completion_ladder

    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    prepared = prepare_projected_apcm_initial_state(
        _PARAMETERS, phonon_cutoff=6, extension=extension
    )
    state, completion, record = _exact_contracted_initialization(
        _PARAMETERS,
        1e-3,
        phonon_cutoff=6,
        extension=extension,
        completion_ladder=build_completion_ladder(
            extension, shifts=(1e-8, 1e-7), weights=(1e-1,)
        ),
    )
    assert record["initial_exact_time"] == 1e-3
    assert record["exact_propagation_success"]
    assert abs(record["exact_state_norm"] - 1.0) < 1e-9
    assert float(np.max(np.abs(state - prepared.state))) < 1e-5
    assert completion.success
    assert (
        record["initial_declared_completion_success"]
        or record["initial_completion_retry_index"] >= 0
    )
    assert record["initial_completion_envelope_ratio"] < 1.0


def test_the_exact_start_refuses_a_nonpositive_time() -> None:
    from paper5.stability.apcm_projective_event_guard_analysis import (
        _exact_contracted_initialization,
    )

    extension = SymmetryReducedPositiveExtension(
        active_keys=ENTRANCE_RELATIVE_MOMENT_KEYS
    )
    with pytest.raises(ValueError):
        _exact_contracted_initialization(
            _PARAMETERS, 0.0, phonon_cutoff=6, extension=extension
        )
