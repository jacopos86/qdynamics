from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.accepted_refit import (
    ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1,
    ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1,
    AcceptedRefitConfig,
    SupportedFSPowellChart,
)
from pipelines.static_adapt.joint_linear_solve import JointLinearSolveConfig
from pipelines.static_adapt.sr_snake_route_profile import (
    CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS,
    PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1,
    canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract,
)


def _runtime_source() -> str:
    return inspect.getsource(adapt_pipeline._run_hardcoded_adapt_vqe)


def _runtime_tree() -> ast.AST:
    return ast.parse(_runtime_source())


def _assignments_to(name: str) -> list[ast.Assign | ast.AnnAssign]:
    matches: list[ast.Assign | ast.AnnAssign] = []
    for node in ast.walk(_runtime_tree()):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            ):
                matches.append(node)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            matches.append(node)
    return matches


def test_material_window_runtime_uses_authoritative_phase3_population() -> None:
    """The independent window must screen every authoritative Phase-III row.

    A Phase-II/Phase-III shortlist is an admission decision, not permission to
    omit candidates from the material-window response surface.  This source-
    level guard deliberately targets the executable orchestration function so
    a route-profile declaration alone cannot satisfy the test.
    """

    assignments = _assignments_to("coordinate_overlay_input_records")
    assert assignments, "runtime never resolves the coordinate-overlay population"
    assignment_sources = [ast.unparse(node) for node in assignments]
    authoritative_assignments = [
        source
        for source in assignment_sources
        if "phase3_measurement_input_records" in source
    ]
    assert authoritative_assignments, (
        "coordinate-overlay runtime never selects the authoritative Phase-III "
        "measurement population"
    )
    assert any(
        "PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1"
        in source
        for source in authoritative_assignments
    ), (
        "the material-window scope is still routed through the shortlisted "
        "admission population"
    )

    material_calls = []
    for node in ast.walk(_runtime_tree()):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if (
            isinstance(function, ast.Name)
            and function.id
            == "evaluate_historical_singleton_material_window_coordinate_models"
        ):
            material_calls.append(node)
    assert material_calls, "runtime never invokes the material-window evaluator"
    assert any(
        call.args
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == "retained_coordinate_records"
        for call in material_calls
    ), (
        "the runtime material-window evaluator is not fed the resolved "
        "authoritative coordinate-overlay records"
    )


def test_material_window_runtime_consumes_exact_estimator_acquisition_plan() -> None:
    """Guard against reverting to full-matrix cardinality proxy accounting."""

    source = _runtime_source()
    required_plan_fields = {
        "screen_gram_diagonal_indices",
        "candidate_cross_gram_active_indices",
        "candidate_cross_hessian_active_indices",
        "candidate_self_gram_acquired",
        "candidate_self_hessian_acquired",
        "candidate_gradient_acquired",
        "old_old_metric_pairs_acquired",
        "old_old_hessian_pairs_acquired",
        "active_gradient_indices_acquired",
        "full_geometry_refresh_performed",
    }
    missing = sorted(field for field in required_plan_fields if field not in source)
    assert not missing, (
        "runtime does not consume the exact material-window estimator "
        f"acquisition plan fields: {missing!r}"
    )
    assert "estimator_acquisition_plan" in source
    assert "phase3_material_window_estimator_acquisition_plan_v1" in source
    assert "material_window_estimator" in source, (
        "runtime has no material-window-specific estimator-ledger consumer; "
        "full scaffold proxy accounting could silently replace the sparse plan"
    )


def test_material_window_does_not_change_full_supported_fs_powell_refit() -> None:
    """The Phase-III W set is independent of the accepted Powell chart."""

    settings = dict(
        CANONICAL_SR_SNAKE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_MATERIAL_WINDOW_V1_EXECUTION_SETTINGS
    )
    assert settings["phase3_response_coordinate_scope"] == (
        PHASE3_RESPONSE_COORDINATE_SCOPE_CANDIDATE_MATERIAL_COUPLING_WINDOW_V1
    )
    assert settings["adapt_accepted_refit_scope"] == (
        ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1
    )
    assert settings["adapt_accepted_refit_coordinate_chart"] == (
        ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1
    )
    contract = (
        canonical_sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_material_window_v1_contract()
    )
    invariants = contract["semantic_invariants"]
    assert invariants[
        "phase3_material_window_independent_from_powell_refit_window"
    ] is True
    assert invariants["phase3_supported_whitening_active"] is False
    assert invariants["accepted_refit_scope"] == ACCEPTED_REFIT_SCOPE_FULL_ANSATZ_V1
    assert invariants["accepted_refit_coordinate_chart"] == (
        ACCEPTED_REFIT_CHART_SUPPORTED_FS_WHITENED_FIXED_V1
    )

    config = AcceptedRefitConfig(
        scope=settings["adapt_accepted_refit_scope"],
        coordinate_chart=settings["adapt_accepted_refit_coordinate_chart"],
        base_chart_policy=settings["adapt_accepted_refit_base_chart_policy"],
        supported_metric=JointLinearSolveConfig(),
    )
    selector_material_window = (1, 4)
    assert config.resolve_logical_indices(
        selector_active_indices=selector_material_window,
        logical_parameter_count=6,
    ) == (0, 1, 2, 3, 4, 5)
    assert selector_material_window == (1, 4)

    runtime_source = _runtime_source()
    helper_start = runtime_source.index(
        "def _make_accepted_refit_optimizer_chart("
    )
    helper_stop = runtime_source.index(
        "\n    def ", helper_start + len("def _make_accepted_refit_optimizer_chart(")
    )
    helper_source = runtime_source[helper_start:helper_stop]
    assert "accepted_refit_config.resolve_logical_indices" in helper_source
    assert "build_supported_fs_powell_chart" in helper_source
    assert "selector_inputs_mutated\": False" in helper_source
    assert "material_window" not in helper_source


def test_material_window_trust_reuses_full_accepted_refit_source_metric() -> None:
    chart = SupportedFSPowellChart(
        objective=lambda value: float(np.sum(np.asarray(value) ** 2)),
        x0=np.zeros(3, dtype=float),
        lift_to_runtime=lambda value: np.asarray(value, dtype=float),
        coordinate_mode="supported_fs_whitened:logical_shared_reduced_v1",
        active_logical_indices=(0, 1, 2),
        active_runtime_indices=(0, 1, 2),
        active_optimizer_indices=(0, 1, 2),
        reduced_positions_by_logical={0: (0,), 1: (1,), 2: (2,)},
        origin_state=np.asarray([1.0, 0.0], dtype=complex),
        origin_logical_theta=np.zeros(3, dtype=float),
        origin_runtime_theta=np.zeros(3, dtype=float),
        whitened_to_logical_map=np.diag([1.0, 0.5, 1.0 / 3.0]),
        logical_to_whitened_map=np.diag([1.0, 2.0, 3.0]),
        coordinate_registry=("old0", "candidate", "old1"),
        base_telemetry={
            "base_coordinate_kind": "logical_shared_reduced",
            "raw_logical_fs_metric": np.diag([1.0, 4.0, 9.0]).tolist(),
            "supported_metric_whitening_provenance_id": "accepted-refit-proof",
        },
    )
    selector_summary = {
        "joint_linear_solve_policy_effective": (
            "supported_metric_projected_generalized_trust_v1"
        ),
        "joint_batch_context_mode": "active_window_v1",
        "geometry_workspace": {"active_indices": [1]},
        # W(old pre-index 1), then the inserted singleton at pre-position 1.
        "joint_step": [0.2, 0.3],
        "joint_fubini_study_displacement_sq": 0.72,
        "material_window_receipt": {
            "retained_indices": [1],
            "omitted_indices": [0],
        },
    }
    transaction = (
        adapt_pipeline._material_window_full_source_metric_trust_transaction(
            selector_summary=selector_summary,
            pre_parameter_count=2,
            positions_in_commit_order=[1],
            accepted_refit_chart=chart,
            pre_refit_logical_theta=np.zeros(3, dtype=float),
            post_refit_logical_theta=np.asarray([0.1, 0.4, 0.2]),
            radius_before=0.5,
            metric_query_accounting={
                "incremental_quantum_query_charge": 6,
                "metric_element_count_acquired": 6,
            },
        )
    )

    # Original old order is [old0, old1], while the post chart is
    # [old0, candidate, old1]. Omitted old0 is zero-filled.
    assert transaction["predicted_full_post_logical_step"] == pytest.approx(
        [0.0, 0.3, 0.2]
    )
    assert transaction["predicted_source_metric_displacement_sq"] == pytest.approx(
        0.72
    )
    assert transaction["realized_source_metric_displacement_sq"] == pytest.approx(
        1.01
    )
    assert transaction["source_metric_reused_from_accepted_refit"] is True
    assert transaction["incremental_quantum_query_charge"] == 0
    assert transaction["accepted_refit_metric_query_accounting"][
        "metric_element_count_acquired"
    ] == 6
    assert transaction["phase3_prediction_coordinate_scope"] == (
        "candidate_material_W_plus_singleton_v1"
    )
    assert transaction["trust_calibration_metric_scope"] == (
        "full_accepted_refit_source_gram_v1"
    )
