from __future__ import annotations

import ast
import inspect
import math
import textwrap

from pipelines.static_adapt import adapt_pipeline as hardcoded_adapt
from pipelines.scaffold.hh_continuation_pruning import (
    PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
    PruneConfig,
)
from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt import prune_schur_payloads


def _recoverability_cfg(**overrides) -> PruneConfig:
    values = {
        "policy": PRUNE_POLICY_RECOVERABILITY_LADDER_V1,
        "surrogate_enabled": True,
        "surrogate_nomination_gate_enabled": True,
        "surrogate_nomination_gate_factor": 0.5,
        "surrogate_exact_trial_cap": 3,
        "local_window_size": 2,
        "surrogate_recovery_trust_radius": 0.25,
        "surrogate_ridge": 1e-5,
        "surrogate_monotonicity_tol": 1e-9,
    }
    values.update(overrides)
    return PruneConfig(**values)


def test_prune_schur_payload_helpers_remain_available_through_wrappers() -> None:
    for name in [
        "_compact_prune_schur_rows",
        "_inactive_prune_schur_nomination_payload",
        "_prune_authority_telemetry",
        "_prune_nomination_sources",
        "_prune_schur_nomination_gate_threshold",
        "_update_prune_schur_gate_payload",
    ]:
        assert getattr(adapt_pipeline, name) is getattr(prune_schur_payloads, name)
        assert getattr(hardcoded_adapt, name) is getattr(prune_schur_payloads, name)


def test_both_prune_surrogate_coordinate_modes_record_the_anchor_energy() -> None:
    """Every direct Hamiltonian application must enter the estimator ledger.

    The logical-shared and runtime-per-Pauli branches build the same prune
    surrogate from different coordinate charts.  This source-level regression
    keeps their direct ``energy_via_one_apply`` anchors accounting-symmetric
    without running a scientific optimization.
    """

    source = textwrap.dedent(
        inspect.getsource(adapt_pipeline._run_hardcoded_adapt_vqe)
    )
    tree = ast.parse(source)
    surrogate_builder = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_build_prune_schur_nomination_scores"
    )
    direct_energy_calls = [
        node
        for node in ast.walk(surrogate_builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "energy_via_one_apply"
    ]
    recorded_anchor_calls = [
        node
        for node in ast.walk(surrogate_builder)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_record_estimator_primitive"
        and any(
            keyword.arg == "consumer_scope"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value == "prune_surrogate_anchor_energy"
            for keyword in node.keywords
        )
    ]

    assert len(direct_energy_calls) == 2
    assert len(recorded_anchor_calls) == len(direct_energy_calls)


def test_prune_authority_payload_keeps_surrogate_nomination_only() -> None:
    payload = prune_schur_payloads._prune_authority_telemetry(
        prune_cfg=_recoverability_cfg(),
    )

    assert payload["screen_authority"] == "nomination_only"
    assert payload["deletion_authority"] == "remove_refit_energy_safety"
    assert payload["surrogate_used_for_acceptance"] is False
    assert not any("amplitude" in key for key in payload)
    assert payload["nomination_lanes"]["schur_surrogate"]["active"] is True
    assert (
        payload["compensator_window_authority"]["typed_compensator_window"]["active"]
        is True
    )


def test_nomination_sources_exclude_typed_compensator_window_lane() -> None:
    rows = prune_schur_payloads._prune_nomination_sources(
        prune_cfg=_recoverability_cfg(),
        indices=[2, 0, 7],
        labels_now=["a", "b", "c"],
        lane_membership={
            "schur_surrogate": {2},
            "typed_compensator_window": {0, 2, 7},
        },
    )

    assert [row["index"] for row in rows] == [2, 0, 7]
    assert rows[0]["label"] == "c"
    assert rows[0]["lanes"] == ["schur_surrogate"]
    assert rows[1]["lanes"] == []
    assert rows[2]["label"] == ""
    assert "typed_compensator_window" not in rows[0]["lanes"]
    assert all(row["authority"] == "nomination_only" for row in rows)


def test_prune_schur_gate_threshold_and_payload_update() -> None:
    cfg = _recoverability_cfg()
    assert prune_schur_payloads._prune_schur_nomination_gate_threshold(
        prune_cfg=cfg,
        max_regression_effective=8.0,
    ) == 4.0
    assert prune_schur_payloads._prune_schur_nomination_gate_threshold(
        prune_cfg=_recoverability_cfg(surrogate_nomination_gate_factor=math.inf),
        max_regression_effective=8.0,
    ) is None
    assert prune_schur_payloads._prune_schur_nomination_gate_threshold(
        prune_cfg=PruneConfig(surrogate_nomination_gate_enabled=True),
        max_regression_effective=8.0,
    ) == 8.0

    summary = {"schur_surrogate_nomination": {"used_for_acceptance": True}}
    prune_schur_payloads._update_prune_schur_gate_payload(
        summary,
        prune_cfg=cfg,
        threshold=4.0,
        pre_gate_candidate_count=5,
        post_gate_candidate_count=3,
    )
    payload = summary["schur_surrogate_nomination"]
    assert payload["nomination_gate_enabled"] is True
    assert payload["nomination_gate_factor"] == 0.5
    assert payload["nomination_gate_threshold"] == 4.0
    assert payload["exact_trial_cap"] == 3
    assert payload["pre_gate_candidate_count"] == 5
    assert payload["post_gate_candidate_count"] == 3
    assert payload["used_for_acceptance"] is False

    ignored = {"schur_surrogate_nomination": object()}
    prune_schur_payloads._update_prune_schur_gate_payload(
        ignored,
        prune_cfg=cfg,
        threshold=1.0,
        pre_gate_candidate_count=1,
        post_gate_candidate_count=1,
    )
    assert "nomination_gate_threshold" not in ignored


def test_inactive_prune_schur_payload_shape() -> None:
    cfg = _recoverability_cfg()
    payload = prune_schur_payloads._inactive_prune_schur_nomination_payload(
        prune_cfg=cfg,
        selected_parameterization_mode="per_pauli_term",
        reason="not_built",
        logical_parameter_count=4,
        runtime_parameter_count=9,
    )

    assert payload["schema"] == "static_prune_schur_nomination_v1"
    assert payload["enabled"] is True
    assert payload["active"] is False
    assert payload["reason"] == "not_built"
    assert payload["authority"] == "rank_window_diag_only"
    assert payload["used_for_nomination"] is False
    assert payload["used_for_acceptance"] is False
    assert payload["selected_parameterization_mode"] == "per_pauli_term"
    assert payload["logical_parameter_count"] == 4
    assert payload["runtime_parameter_count"] == 9
    assert payload["local_window_size"] == 2
    assert payload["recovery_trust_radius"] == 0.25
    assert payload["ridge"] == 1e-5
    assert payload["monotonicity_tol"] == 1e-9
    assert payload["hessian_shape"] == []


def test_compact_prune_schur_rows_preserves_ranked_diagnostics() -> None:
    rows = prune_schur_payloads._compact_prune_schur_rows(
        {
            5: {
                "label": "late",
                "score": 3.0,
                "schur_rows": [{"schur_value": 3.0, "window_indices": [0, 1]}],
            },
            2: {
                "label": "early",
                "score": 1.0,
                "unweighted_score": 2.0,
                "schur_min": 2.0,
                "bounded_score": 0.75,
                "entry_cost_denominator": 2.0,
                "schur_model": "metric_regularized_v1",
                "metric_mu": 1e-6,
                "metric_schur_solve_mode": "stationary_gw_zero_v1",
                "bounded_recovery_active": True,
                "recovery_trust_radius": 0.2,
                "schur_health": "ok",
                "schur_monotone": True,
                "schur_rows": [
                    {
                        "schur_value": 2.0,
                        "bounded_value": 0.75,
                        "compensation_norm": 0.4,
                        "window_indices": [1],
                    }
                ],
                "surrogate_authority": "rank_window_diag_only",
                "used_for_acceptance": True,
            },
        },
        max_rows=1,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["index"] == 2
    assert row["label"] == "early"
    assert row["score"] == 1.0
    assert row["unweighted_score"] == 2.0
    assert row["schur_min"] == 2.0
    assert row["bounded_score"] == 0.75
    assert row["entry_cost_denominator"] == 2.0
    assert row["schur_model"] == "metric_regularized_v1"
    assert row["metric_mu"] == 1e-6
    assert row["metric_schur_solve_mode"] == "stationary_gw_zero_v1"
    assert row["bounded_recovery_active"] is True
    assert row["rung_values"] == [2.0]
    assert row["bounded_rung_values"] == [0.75]
    assert row["compensation_norms"] == [0.4]
    assert row["window_sizes"] == [1]
    assert row["surrogate_authority"] == "rank_window_diag_only"
    assert row["used_for_acceptance"] is True
