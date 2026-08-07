from types import SimpleNamespace

import numpy as np

from pipelines.static_adapt.adapt_pipeline import (
    _apply_rotosolve_parameterization_policy,
    _optimizer_coordinate_chart_payload,
    _resolve_selected_parameterization_mode,
    _resolve_sr_powell_coordinate_chart_runtime_policy,
)
from pipelines.static_adapt.engine_support import (
    _make_logical_shared_reduced_objective,
)
from pipelines.static_adapt.sr_snake_escape_controller import (
    SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1,
    SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
    SR_ESCAPE_DISABLED,
    SR_POWELL_COORDINATE_CHART_AUTO,
    SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1,
    SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1,
    SR_ROUTE_PROFILE_DISABLED,
    SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED,
    SR_ROUTE_PROFILE_REDUCED_POWELL,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def test_canonical_sr_v1_auto_resolves_historical_expanded_powell_chart():
    receipt = _resolve_sr_powell_coordinate_chart_runtime_policy(
        historical_singleton_overlay_active=True,
        sr_escape_mode=SR_ESCAPE_DISABLED,
        coordinate_solve_scope=SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
        requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
        source_locked_replay=False,
    )

    assert receipt["runtime_policy"] == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    assert receipt["route_profile"] == SR_ROUTE_PROFILE_DISABLED
    assert receipt["resolution_source"] == "canonical_sr_profile_auto"


def test_reduced_powell_chart_has_distinct_sr_route_profile():
    receipt = _resolve_sr_powell_coordinate_chart_runtime_policy(
        historical_singleton_overlay_active=True,
        sr_escape_mode=SR_ESCAPE_DISABLED,
        coordinate_solve_scope=SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
        requested_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
        source_locked_replay=False,
    )

    assert receipt["runtime_policy"] == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    assert receipt["route_profile"] == SR_ROUTE_PROFILE_REDUCED_POWELL
    assert receipt["resolution_source"] == "explicit_request"


def test_phase2_whitening_auto_keeps_reduced_powell_profile():
    receipt = _resolve_sr_powell_coordinate_chart_runtime_policy(
        historical_singleton_overlay_active=True,
        sr_escape_mode=SR_ESCAPE_DISABLED,
        coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
        ),
        requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
        source_locked_replay=False,
    )

    assert receipt["runtime_policy"] == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    assert receipt["route_profile"] == SR_ROUTE_PROFILE_PHASE2_PHASE3_WHITENED


def test_source_locked_sr_replay_rejects_auto_powell_chart():
    import pytest

    with pytest.raises(ValueError, match="Source-locked SR-SNAKE replay"):
        _resolve_sr_powell_coordinate_chart_runtime_policy(
            historical_singleton_overlay_active=True,
            sr_escape_mode=SR_ESCAPE_DISABLED,
            coordinate_solve_scope=SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
            requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
            source_locked_replay=True,
        )


def test_fm_profile_validates_its_own_phase3_only_scope() -> None:
    receipt = _resolve_sr_powell_coordinate_chart_runtime_policy(
        historical_singleton_overlay_active=True,
        sr_escape_mode=SR_ESCAPE_DISABLED,
        coordinate_solve_scope=SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
        requested_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
        source_locked_replay=False,
        formal_manifold_selector_profile=(
            "supported_whitened_adaptive_trust_v1"
        ),
        formal_manifold_required_coordinate_solve_scope=(
            SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1
        ),
    )

    assert receipt["runtime_policy"] == (
        SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
    )
    assert receipt["formal_manifold_required_coordinate_solve_scope"] == (
        SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1
    )


def test_fm_profile_rejects_coordinate_scope_drift() -> None:
    import pytest

    with pytest.raises(ValueError, match="coordinate scope"):
        _resolve_sr_powell_coordinate_chart_runtime_policy(
            historical_singleton_overlay_active=True,
            sr_escape_mode=SR_ESCAPE_DISABLED,
            coordinate_solve_scope=(
                SR_COORDINATE_SOLVE_SCOPE_PHASE2_AND_PHASE3_V1
            ),
            requested_policy=(
                SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
            ),
            source_locked_replay=False,
            formal_manifold_selector_profile=(
                "supported_whitened_adaptive_trust_v1"
            ),
            formal_manifold_required_coordinate_solve_scope=(
                SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1
            ),
        )


def test_non_sr_runtime_preserves_reduced_powell_compatibility_default():
    receipt = _resolve_sr_powell_coordinate_chart_runtime_policy(
        historical_singleton_overlay_active=False,
        sr_escape_mode=SR_ESCAPE_DISABLED,
        coordinate_solve_scope=SR_COORDINATE_SOLVE_SCOPE_PHASE3_ONLY_V1,
        requested_policy=SR_POWELL_COORDINATE_CHART_AUTO,
        source_locked_replay=False,
    )

    assert receipt["runtime_policy"] == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    assert receipt["resolution_source"] == "non_sr_compatibility_default"


def test_hh_full_meta_resolves_logical_shared_parameterization():
    assert (
        _resolve_selected_parameterization_mode(
            problem_key_value="hh",
            pool_key_value="full_meta",
            candidate_terms=[],
        )
        == "logical_shared"
    )


def test_molecular_vibronic_h2o_linear_fd_full_meta_resolves_logical_shared_parameterization():
    assert (
        _resolve_selected_parameterization_mode(
            problem_key_value="molecular_vibronic_h2o_linear_fd",
            pool_key_value="full_meta",
            candidate_terms=[],
        )
        == "logical_shared"
    )


def test_grouped_exact_candidate_forces_logical_shared_parameterization():
    term = SimpleNamespace(execution_mode="grouped_exact")

    assert (
        _resolve_selected_parameterization_mode(
            problem_key_value="hh",
            pool_key_value="paop",
            candidate_terms=[term],
        )
        == "logical_shared"
    )


def test_rotosolve_preserves_required_logical_shared_parameterization():
    mode, source = _apply_rotosolve_parameterization_policy(
        selected_parameterization_mode="logical_shared",
        selected_parameterization_mode_source="route_default",
        adapt_inner_optimizer_key="ROTOSOLVE",
    )

    assert mode == "logical_shared"
    assert source == "rotosolve_preserve_logical_shared"


def test_rotosolve_uses_per_pauli_when_route_does_not_require_grouping():
    mode, source = _apply_rotosolve_parameterization_policy(
        selected_parameterization_mode="per_pauli_term",
        selected_parameterization_mode_source="route_default",
        adapt_inner_optimizer_key="ROTOSOLVE",
    )

    assert mode == "per_pauli_term"
    assert source == "rotosolve_runtime_per_pauli"


def test_non_rotosolve_leaves_parameterization_policy_unchanged():
    mode, source = _apply_rotosolve_parameterization_policy(
        selected_parameterization_mode="logical_shared",
        selected_parameterization_mode_source="route_default",
        adapt_inner_optimizer_key="POWELL",
    )

    assert mode == "logical_shared"
    assert source == "route_default"


def test_logical_shared_optimizer_uses_one_coordinate_per_generator():
    terms = [
        AnsatzTerm(
            label="two_factor",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xy", pc=0.5),
                    PauliTerm(2, ps="yx", pc=-0.5),
                ],
            ),
        ),
        AnsatzTerm(
            label="one_factor",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ze", pc=1.0)],
            ),
        ),
    ]
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    runtime_theta = np.asarray([0.2, 0.2, -0.4], dtype=float)
    seen: list[np.ndarray] = []

    def objective(theta: np.ndarray) -> float:
        seen.append(np.asarray(theta, dtype=float).copy())
        return float(np.dot(theta, theta))

    reduced, x0, lift = _make_logical_shared_reduced_objective(
        runtime_theta,
        layout,
        [0, 1],
        objective,
    )

    assert x0.tolist() == [0.2, -0.4]
    trial = np.asarray([0.7, -0.1], dtype=float)
    assert reduced(trial) == objective(lift(trial))
    assert lift(trial).tolist() == [0.7, 0.7, -0.1]
    assert seen[-1].tolist() == [0.7, 0.7, -0.1]


def test_optimizer_coordinate_chart_reports_powell_logical_reduction():
    terms = [
        AnsatzTerm(
            label="two_factor",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xy", pc=0.5),
                    PauliTerm(2, ps="yx", pc=-0.5),
                ],
            ),
        ),
        AnsatzTerm(
            label="one_factor",
            polynomial=PauliPolynomial(
                "JW",
                [PauliTerm(2, ps="ze", pc=1.0)],
            ),
        ),
    ]
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )

    payload = _optimizer_coordinate_chart_payload(
        layout=layout,
        parameterization_execution_mode="logical_shared",
        optimizer_key="powell",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    assert payload["optimizer"] == "POWELL"
    assert payload["coordinate_mode"] == "logical_shared"
    assert payload["optimizer_dimension"] == 2
    assert payload["logical_dimension"] == 2
    assert payload["runtime_dimension"] == 3
    assert payload["one_coordinate_per_logical_generator"] is True
    assert payload["powell_coordinate_chart_policy"] == (
        SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
    )
    assert payload["expanded_runtime_projected_logical"] is False
    assert payload["runtime_vector_remains_expanded"] is True


def test_optimizer_coordinate_chart_reports_historical_expanded_powell_chart():
    terms = [
        AnsatzTerm(
            label="two_factor",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xy", pc=0.5),
                    PauliTerm(2, ps="yx", pc=-0.5),
                ],
            ),
        )
    ]
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )

    payload = _optimizer_coordinate_chart_payload(
        layout=layout,
        parameterization_execution_mode="logical_shared",
        optimizer_key="POWELL",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_EXPANDED_RUNTIME_PROJECTED_LOGICAL_V1
        ),
    )

    assert payload["coordinate_mode"] == "runtime"
    assert payload["optimizer_dimension"] == 2
    assert payload["logical_dimension"] == 1
    assert payload["runtime_dimension"] == 2
    assert payload["one_coordinate_per_logical_generator"] is False
    assert payload["expanded_runtime_projected_logical"] is True
    assert payload["logical_projection_boundary"] == (
        "block_mean_each_objective_and_lift"
    )


def test_optimizer_coordinate_chart_reports_runtime_coordinates_for_non_powell():
    terms = [
        AnsatzTerm(
            label="two_factor",
            polynomial=PauliPolynomial(
                "JW",
                [
                    PauliTerm(2, ps="xy", pc=0.5),
                    PauliTerm(2, ps="yx", pc=-0.5),
                ],
            ),
        )
    ]
    layout = build_parameter_layout(
        terms,
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )

    payload = _optimizer_coordinate_chart_payload(
        layout=layout,
        parameterization_execution_mode="logical_shared",
        optimizer_key="SPSA",
        powell_coordinate_chart_policy=(
            SR_POWELL_COORDINATE_CHART_LOGICAL_SHARED_REDUCED_V1
        ),
    )

    assert payload["coordinate_mode"] == "runtime"
    assert payload["optimizer_dimension"] == 2
    assert payload["logical_dimension"] == 1
    assert payload["runtime_dimension"] == 2
    assert payload["one_coordinate_per_logical_generator"] is False
