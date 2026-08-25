from __future__ import annotations

import inspect
from typing import Any

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
)
from pipelines.scaffold.hh_continuation_pruning import (
    AffineDeletionFSTrustState,
)
from src.quantum.hubbard_latex_python_pairs import (
    build_hubbard_holstein_hamiltonian,
)


def _estimator_key(identity: str) -> EstimatorCallKey:
    return EstimatorCallKey(
        projective_state_fingerprint=f"state:{identity}",
        hamiltonian_fingerprint="hamiltonian:test",
        backend_fingerprint="backend:test",
        precision_contract="precision:test",
        primitive_kind="hamiltonian_expectation",
        observable_or_formula_identity="hamiltonian_expectation_v1",
    )


def test_v4_prune_trial_id_is_parent_branch_scoped_under_beam() -> None:
    shared = {
        "selector_step": 5,
        "candidate_index": 2,
        "candidate_label": "macro:test",
    }
    unbranched = adapt_pipeline._sr_v4_prune_trial_branch_id(**shared)
    parent_a = adapt_pipeline._sr_v4_prune_trial_branch_id(
        **shared,
        parent_branch_id="beam:parent:a",
    )
    parent_b = adapt_pipeline._sr_v4_prune_trial_branch_id(
        **shared,
        parent_branch_id="beam:parent:b",
    )

    assert len({unbranched, parent_a, parent_b}) == 3
    assert unbranched == adapt_pipeline._sr_v4_prune_trial_branch_id(**shared)


def test_v4_disabled_finite_angle_switch_skips_flat_gradient_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The switch selected by v4 must bypass the executable probe guard."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=0.5,
        omega0=1.0,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )

    payload, _psi = adapt_pipeline._run_hardcoded_adapt_vqe(
        h_poly=h_poly,
        num_sites=2,
        ordering="blocked",
        problem="hh",
        adapt_pool="full_hamiltonian",
        t=1.0,
        u=0.5,
        dv=0.0,
        boundary="open",
        omega0=1.0,
        g_ep=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        max_depth=1,
        eps_grad=1.0e9,
        eps_energy=0.0,
        maxiter=2,
        seed=7,
        allow_repeats=True,
        finite_angle_fallback=False,
        finite_angle=0.1,
        finite_angle_min_improvement=1.0e-12,
        adapt_continuation_mode="legacy",
        phase0_pilot_enabled=False,
        phase1_prune_enabled=False,
        phase3_enable_rescue=False,
        adapt_estimator_call_ledger_enabled=True,
    )

    assert payload["finite_angle_fallback"] is False
    assert payload["stop_reason"] == "eps_grad"
    assert payload["history"] == []
    accounting = payload["estimator_call_accounting"]
    scopes = accounting["executed_occurrence_accounting"]["all_execution"][
        "occurrence_count_by_consumer_scope"
    ]
    assert "finite_angle_objective_guard" not in scopes
    assert all(
        occurrence["consumer_scope"] != "finite_angle_objective_guard"
        for occurrence in accounting["full_ledger"]["occurrences"]
    )


def _prune_history_row(
    *,
    depth: int,
    branch_id: str,
    accepted: bool,
    nfev: int = 1,
) -> dict[str, object]:
    return {
        "depth": int(depth),
        "post_admission_prune": {
            "accepted_count": int(bool(accepted)),
            "phase1_prune_exact_refit_work_accounting": {
                "schema": "sr_v4_prune_exact_refit_work_accounting_v1",
                "classification": (
                    "committed_prune" if accepted else "discarded_prune"
                ),
                "estimator_trial_branch_id": str(branch_id),
                "nfev": int(nfev),
            },
        },
    }


def test_v4_prune_estimator_partition_commits_only_accepted_trial_work() -> None:
    accepted_branch = adapt_pipeline._sr_v4_prune_trial_branch_id(
        selector_step=4,
        candidate_index=1,
        candidate_label="accepted",
    )
    rejected_branch = adapt_pipeline._sr_v4_prune_trial_branch_id(
        selector_step=5,
        candidate_index=2,
        candidate_label="rejected",
    )
    history_rows = [
        _prune_history_row(
            depth=4,
            branch_id=accepted_branch,
            accepted=True,
        ),
        _prune_history_row(
            depth=5,
            branch_id=rejected_branch,
            accepted=False,
        ),
    ]
    ledger = EstimatorCallLedger()
    shared_key = _estimator_key("shared")
    accepted_key = _estimator_key("accepted")
    rejected_key = _estimator_key("rejected")
    ledger.record_call(
        shared_key,
        component="N_H_outer",
        consumer_scope="energy:outer",
    )
    ledger.record_call(
        accepted_key,
        component="N_H_refit",
        consumer_scope="energy:prune_refit_live",
        branch_id=accepted_branch,
    )
    ledger.record_call(
        rejected_key,
        component="N_H_refit",
        consumer_scope="energy:prune_refit_live",
        branch_id=rejected_branch,
    )
    # A rejected trial may reuse a primitive already needed by the winner.  It
    # remains visible in raw discarded occurrences but is not double charged.
    ledger.record_call(
        shared_key,
        component="N_H_refit",
        consumer_scope="energy:prune_refit_live",
        branch_id=rejected_branch,
    )
    for row in history_rows:
        row["post_admission_prune"][
            "phase1_prune_source_geometry_reuse"
        ] = {
            "schema": "sr_material_window_prune_source_geometry_reuse_v1",
            "primitive_ids": [shared_key.primitive_id],
            "incremental_quantum_query_charge": 0,
            "duplicate_measurement_performed": False,
        }

    views = adapt_pipeline._sr_v4_prune_estimator_accounting_views(
        ledger=ledger,
        history_rows=history_rows,
    )

    assert views["schema"] == "sr_v4_prune_estimator_accounting_views_v1"
    assert views["accepted_trial_branch_ids"] == [accepted_branch]
    assert views["rejected_trial_branch_ids"] == [rejected_branch]
    assert set(views["winning_lineage"]["primitive_ids"]) == {
        shared_key.primitive_id,
        accepted_key.primitive_id,
    }
    assert views["discarded_prune_only_by_unique_set_difference"][
        "primitive_ids"
    ] == [rejected_key.primitive_id]
    assert views["rejected_prune_execution"]["total_call_occurrences"] == 2
    assert views["rejected_prune_shared_with_winner_ids"] == [
        shared_key.primitive_id
    ]
    assert views["all_work"]["S_unique"] == 3
    assert views["winning_lineage"]["S_unique"] == 2
    assert views["rejected_prune_consumer_unique"]["S_unique"] == 2
    assert views["shared_source_state"] == {
        "schema": "estimator_call_ledger_unique_primitive_set_summary_v2",
        "component_contract": [
            "N_H_outer",
            "N_H_refit",
            "N_grad",
            "N_metric",
        ],
        "component_assignment": "ledger_global_charged_component_v1",
        "components": {
            "N_H_outer": 1,
            "N_H_refit": 0,
            "N_grad": 0,
            "N_metric": 0,
        },
        "N_H_outer": 1,
        "N_H_refit": 0,
        "N_grad": 0,
        "N_metric": 0,
        "S_unique": 1,
        "unique_primitive_count": 1,
        "primitive_ids": [shared_key.primitive_id],
        "primitive_set_sha256": views["shared_source_state"][
            "primitive_set_sha256"
        ],
        "component_by_primitive_id": {
            shared_key.primitive_id: "N_H_outer",
        },
    }
    assert views["discarded_prune_only_by_unique_set_difference"][
        "S_unique"
    ] == 1
    assert views["winning_lineage_excluding_shared_source"]["S_unique"] == 1
    assert views["all_work"]["S_unique"] == (
        views["shared_source_state"]["S_unique"]
        + views["winning_lineage_excluding_shared_source"]["S_unique"]
        + views["discarded_prune_only_by_unique_set_difference"]["S_unique"]
    )
    assert rejected_key.primitive_id in views["all_work"]["primitive_ids"]
    assert rejected_key.primitive_id not in views["winning_lineage"][
        "primitive_ids"
    ]
    assert views["primitive_set_reconciliation"] == {
        "partition": (
            "shared_source_plus_winning_excluding_shared_plus_"
            "discarded_only_v1"
        ),
        "pairwise_disjoint": True,
        "union_equals_all_work": True,
        "all_work_S_unique": 3,
        "partition_S_unique": 3,
        "shared_source_count": 1,
        "winning_excluding_shared_count": 1,
        "discarded_count": 1,
        "all_work_count": 3,
    }


def test_minimal_keep_prune_runtime_guards_immutable_keep_and_zero_query_rollback() -> None:
    """Lock the execution guard, not merely the route-profile declaration."""

    source = inspect.getsource(adapt_pipeline._run_hardcoded_adapt_vqe)
    snapshot_pos = source.index("keep_branch_theta_snapshot =")
    measured_sibling_pos = source.index(
        ") = recoverability_prune_ladder(",
        snapshot_pos,
    )
    intact_check_pos = source.index(
        "keep_branch_intact = bool(",
        measured_sibling_pos,
    )
    fail_closed_pos = source.index(
        "Keep-versus-prune verification mutated the surviving ",
        intact_check_pos,
    )
    receipt_pos = source.index(
        'summary["minimal_keep_prune_verification_beam"]',
        fail_closed_pos,
    )

    assert snapshot_pos < measured_sibling_pos < intact_check_pos
    assert intact_check_pos < fail_closed_pos < receipt_pos
    assert '"intact_before_decision": True' in source[receipt_pos:]
    assert '"destructively_mutated_then_restored": False' in source[receipt_pos:]
    assert '"rollback_classical_query_charge": 0' in source[receipt_pos:]
    assert 'summary["rollback_snapshot_restored"] = False' in source[receipt_pos:]


def test_v4_prune_estimator_partition_fails_closed_on_acceptance_mismatch() -> None:
    branch_id = adapt_pipeline._sr_v4_prune_trial_branch_id(
        selector_step=4,
        candidate_index=1,
        candidate_label="mismatch",
    )
    row = _prune_history_row(
        depth=4,
        branch_id=branch_id,
        accepted=False,
    )
    row["post_admission_prune"]["accepted_count"] = 1  # type: ignore[index]

    with pytest.raises(RuntimeError, match="disagrees with acceptance"):
        adapt_pipeline._sr_v4_prune_trial_branch_partition([row])


def test_v4_all_infeasible_affine_prune_holds_state_without_fallback() -> None:
    state = AffineDeletionFSTrustState(
        radius=0.125,
        metric_damping=1.0e-6,
        update_count=7,
    )
    receipt = adapt_pipeline._sr_v4_all_infeasible_prune_hold_receipt(
        route_active=True,
        nomination_payload={
            "affine_deletion_model_count": 2,
            "affine_deletion_feasible_count": 0,
            "affine_deletion_models": [
                {
                    "feasible": False,
                    "reason": "deletion_displacement_exceeds_trust_radius",
                },
                {
                    "feasible": False,
                    "reason": "affine_deletion_certificate_failed",
                },
            ],
        },
        trust_state=state,
    )

    assert receipt is not None
    assert receipt["status"] == "skipped_no_feasible_affine_deletion_models"
    assert receipt["legacy_nomination_fallback_used"] is False
    assert receipt["exact_delete_refit_trial_count"] == 0
    assert receipt["trust_state_before"] == receipt["trust_state_after"]
    assert receipt["trust_update"]["radius_before"] == pytest.approx(0.125)
    assert receipt["trust_update"]["radius_after"] == pytest.approx(0.125)
    assert receipt["trust_update"]["metric_damping_before"] == pytest.approx(
        1.0e-6
    )
    assert receipt["trust_update"]["metric_damping_after"] == pytest.approx(
        1.0e-6
    )
    assert receipt["trust_update"]["update_count_before"] == 7
    assert receipt["trust_update"]["update_count_after"] == 7
    assert receipt["classical_quantum_query_charge"] == 0


def test_v4_affine_prune_hold_receipt_is_inactive_for_legacy_routes() -> None:
    assert (
        adapt_pipeline._sr_v4_all_infeasible_prune_hold_receipt(
            route_active=False,
            nomination_payload={},
            trust_state=None,
        )
        is None
    )


@pytest.mark.parametrize(
    "normalization_mode",
    [
        "family_robust_v1",
        "family_robust_symmetric_arctan_v1",
    ],
)
def test_phase1_runtime_receives_resolved_hardware_cost_policy(
    normalization_mode: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the Phase-I bridge aligned with the resolved route policy."""

    class _StaticPayloadCaptured(RuntimeError):
        pass

    captured: dict[str, Any] = {}

    def _capture_static_payload(payload: dict[str, Any]) -> str:
        captured["payload"] = payload
        raise _StaticPayloadCaptured

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        adapt_pipeline,
        "_candidate_record_payload_digest",
        _capture_static_payload,
    )
    h_poly = build_hubbard_holstein_hamiltonian(
        dims=2,
        J=1.0,
        U=0.5,
        omega0=1.0,
        g=0.2,
        n_ph_max=1,
        boson_encoding="binary",
        repr_mode="JW",
        indexing="blocked",
        pbc=False,
        include_zero_point=True,
    )

    with pytest.raises(_StaticPayloadCaptured):
        adapt_pipeline._run_hardcoded_adapt_vqe(
            h_poly=h_poly,
            num_sites=2,
            ordering="blocked",
            problem="hh",
            adapt_pool="full_meta",
            t=1.0,
            u=0.5,
            dv=0.0,
            boundary="open",
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            max_depth=0,
            eps_grad=1.0e-12,
            eps_energy=0.0,
            maxiter=2,
            seed=7,
            allow_repeats=True,
            finite_angle_fallback=False,
            finite_angle=0.1,
            finite_angle_min_improvement=1.0e-12,
            adapt_continuation_mode="phase3_v1",
            phase0_pilot_enabled=False,
            phase1_prune_enabled=False,
            phase3_hardware_cost_normalization_mode=normalization_mode,
        )

    payload = captured["payload"]
    phase1_cfg = payload["phase1_score_cfg"]
    phase2_cfg = payload["phase2_score_cfg"]
    assert phase1_cfg["class"] == "SimpleScoreConfig"
    assert (
        phase1_cfg["fields"]["hardware_cost_normalization_mode"]
        == normalization_mode
    )
    assert phase2_cfg["class"] == "FullScoreConfig"
    assert (
        phase2_cfg["fields"]["hardware_cost_normalization_mode"]
        == normalization_mode
    )
