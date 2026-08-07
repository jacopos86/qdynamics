from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.scaffold.hh_continuation_pruning import (
    AffineDeletionFSTrustState,
    solve_full_logical_affine_deletion_fs_trust,
)
from pipelines.static_adapt.adapt_pipeline import (
    _normalize_sr_material_window_prune_source_geometry,
    _sr_material_window_prune_deletion_coordinates,
    _sr_v4_no_eligible_material_window_prune_hold_receipt,
    _sr_v4_prune_estimator_accounting_views,
    _sr_v4_prune_trial_branch_id,
)
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallKey,
    EstimatorCallLedger,
)
from pipelines.static_adapt.joint_linear_solve import (
    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1,
)


def _source_workspace() -> dict[str, object]:
    plan = {
        "schema": "phase3_material_window_estimator_acquisition_plan_v1",
        "state_fingerprint": "state-1",
        "ordered_scaffold_fingerprint": "scaffold-1",
        "theta_fingerprint": "theta-1",
        "hamiltonian_fingerprint": "hamiltonian-1",
        "candidate_coordinate_fingerprint": "candidate-coordinate-1",
        "candidate_pool_index": 17,
        "candidate_label": "candidate",
        "candidate_position_id": 1,
        "active_indices": [0, 1, 2],
        "screen_gram_diagonal_indices": [0, 1, 2],
        "candidate_cross_gram_active_indices": [0, 1, 2],
        "candidate_cross_hessian_active_indices": [0, 1, 2],
        "candidate_self_gram_acquired": True,
        "candidate_self_hessian_acquired": True,
        "candidate_gradient_acquired": True,
        "old_old_metric_pairs_acquired": [[0, 0], [0, 2], [2, 2]],
        "old_old_hessian_pairs_acquired": [[0, 0], [0, 2], [2, 2]],
        "active_gradient_indices_acquired": [0, 2],
    }
    return {
        "schema": "historical_singleton_coordinate_model_v1",
        "joint_batch_context_mode": "active_window_v1",
        "joint_linear_solve_policy_effective": (
            JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_PROJECTED_GENERALIZED_TRUST_V1
        ),
        "feasible": True,
        "candidate_pool_index": 17,
        "candidate_label": "candidate",
        "position_id": 1,
        "active_coordinate_identities": ["old-0", "old-2"],
        "batch_coordinate_identities": [
            {
                "candidate_pool_index": 17,
                "candidate_label": "candidate",
                "position_id": 1,
                "global_child_identity": "child-17",
            }
        ],
        "G_AA_raw": [[1.0, 0.1], [0.1, 1.5]],
        "G_AB_raw": [[0.05], [0.02]],
        "G_BB_raw": [[0.8]],
        "H_AA_raw": [[2.0, 0.2], [0.2, 1.7]],
        "H_AB_raw": [[0.03], [0.04]],
        "H_BB_raw": [[1.2]],
        "g_A": [0.4, -0.2],
        "g_B": [0.1],
        "material_window_receipt": {
            "retained_indices": [0, 2],
            "omitted_indices": [1],
        },
        "material_window_refresh": {
            "performed": False,
            "final_active_indices": [0, 2],
        },
        "estimator_acquisition_plan": copy.deepcopy(plan),
        "material_window_estimator_accounting": {
            "schema": "phase3_material_window_estimator_accounting_v1",
            "candidate_pool_index": 17,
            "candidate_label": "candidate",
            "candidate_position_id": 1,
            "source_plan": copy.deepcopy(plan),
            "unique_primitive_id_count": 2,
            "primitive_ids": ["primitive-a", "primitive-b"],
        },
    }


def test_material_window_prune_source_reuse_maps_W_plus_candidate_without_queries():
    model = _normalize_sr_material_window_prune_source_geometry(
        selector_summary=_source_workspace(),
        post_admission_labels=["old-0", "candidate", "old-1", "old-2"],
        post_admission_theta=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )

    assert model["model_post_indices"] == [0, 3, 1]
    assert np.allclose(model["theta"], [0.1, 0.4, 0.2])
    # The scorer records descent gradients; the prune solver requires dE/dtheta.
    assert np.allclose(model["gradient"], [-0.4, 0.2, -0.1])
    assert np.asarray(model["metric"]).shape == (3, 3)
    assert np.asarray(model["hessian"]).shape == (3, 3)
    receipt = model["receipt"]
    assert receipt["incremental_quantum_query_charge"] == 0
    assert receipt["duplicate_measurement_performed"] is False
    assert receipt["unsupported_logical_coordinates_nominated"] is False
    assert receipt["active_post_indices"] == [0, 3]

    deletion_coordinates = _sr_material_window_prune_deletion_coordinates(
        model_post_indices=model["model_post_indices"],
        source_geometry_reuse_receipt=receipt,
    )
    assert deletion_coordinates == [(0, 0), (1, 3)]
    assert all(post_index != 1 for _, post_index in deletion_coordinates)
    solves = [
        solve_full_logical_affine_deletion_fs_trust(
            theta=np.asarray(model["theta"], dtype=float),
            gradient=np.asarray(model["gradient"], dtype=float),
            hessian=np.asarray(model["hessian"], dtype=float),
            metric=np.asarray(model["metric"], dtype=float),
            deletion_index=model_index,
            trust_radius=1.0,
            metric_damping=0.0,
        )
        for model_index, _post_index in deletion_coordinates
    ]
    assert all(result.feasible for result in solves)


def test_material_window_prune_source_reuse_binds_only_blank_candidate_label():
    payload = _source_workspace()
    payload["batch_coordinate_identities"][0]["candidate_label"] = ""

    model = _normalize_sr_material_window_prune_source_geometry(
        selector_summary=payload,
        post_admission_labels=["old-0", "candidate", "old-1", "old-2"],
        post_admission_theta=np.asarray([0.1, 0.2, 0.3, 0.4]),
    )

    receipt = model["receipt"]
    assert receipt["candidate_identity"]["candidate_label"] == "candidate"
    binding = receipt["candidate_identity_binding"]
    assert binding["placeholder_filled"] is True
    assert binding["candidate_label_before_binding"] == ""
    assert binding["candidate_label_after_binding"] == "candidate"
    assert binding["pool_index_and_position_crosschecked"] is True
    assert binding["numeric_geometry_modified"] is False
    assert binding["incremental_quantum_query_charge"] == 0


def test_material_window_prune_source_reuse_rejects_conflicting_candidate_label():
    payload = _source_workspace()
    payload["batch_coordinate_identities"][0]["candidate_label"] = "other"

    with pytest.raises(RuntimeError, match="candidate identity drifted"):
        _normalize_sr_material_window_prune_source_geometry(
            selector_summary=payload,
            post_admission_labels=["old-0", "candidate", "old-1", "old-2"],
            post_admission_theta=np.asarray([0.1, 0.2, 0.3, 0.4]),
        )


def test_material_window_prune_source_reuse_accepts_candidate_only_window():
    payload = _source_workspace()
    plan = payload["estimator_acquisition_plan"]
    accounting = payload["material_window_estimator_accounting"]
    plan.update(
        {
            "active_indices": [0],
            "screen_gram_diagonal_indices": [0],
            "candidate_cross_gram_active_indices": [],
            "candidate_cross_hessian_active_indices": [],
            "old_old_metric_pairs_acquired": [],
            "old_old_hessian_pairs_acquired": [],
            "active_gradient_indices_acquired": [],
        }
    )
    accounting["source_plan"] = copy.deepcopy(plan)
    payload.update(
        {
            "active_coordinate_identities": [],
            "G_AA_raw": np.empty((0, 0), dtype=float),
            "G_AB_raw": np.empty((0, 1), dtype=float),
            "H_AA_raw": np.empty((0, 0), dtype=float),
            "H_AB_raw": np.empty((0, 1), dtype=float),
            "g_A": np.empty((0,), dtype=float),
            "material_window_receipt": {
                "retained_indices": [],
                "omitted_indices": [0],
            },
            "material_window_refresh": {
                "performed": False,
                "final_active_indices": [],
            },
        }
    )

    model = _normalize_sr_material_window_prune_source_geometry(
        selector_summary=payload,
        post_admission_labels=["old-0", "candidate"],
        post_admission_theta=np.asarray([0.1, 0.2]),
    )

    assert model["model_post_indices"] == [1]
    assert np.allclose(model["theta"], [0.2])
    assert np.allclose(model["gradient"], [-0.1])
    assert np.asarray(model["metric"]).shape == (1, 1)
    assert np.asarray(model["hessian"]).shape == (1, 1)
    receipt = model["receipt"]
    assert receipt["active_pre_indices"] == []
    assert receipt["active_post_indices"] == []
    assert receipt["candidate_post_index"] == 1
    assert receipt["incremental_quantum_query_charge"] == 0
    assert receipt["duplicate_measurement_performed"] is False

    deletion_coordinates = _sr_material_window_prune_deletion_coordinates(
        model_post_indices=model["model_post_indices"],
        source_geometry_reuse_receipt=receipt,
    )
    assert deletion_coordinates == []
    nomination_payload = {
        "reason": "no_eligible_old_coordinates_in_material_window",
        "affine_deletion_models": [],
        "affine_deletion_model_count": 0,
        "affine_deletion_feasible_count": 0,
        "score_count": 0,
    }
    hold = _sr_v4_no_eligible_material_window_prune_hold_receipt(
        route_active=True,
        nomination_payload=nomination_payload,
        trust_state=AffineDeletionFSTrustState(
            radius=0.25,
            metric_damping=1.0e-8,
            update_count=3,
        ),
    )
    assert hold is not None
    assert hold["status"] == "skipped_no_eligible_old_coordinates"
    assert hold["admitted_singleton_nominated"] is False
    assert hold["exact_delete_refit_trial_count"] == 0
    assert hold["classical_quantum_query_charge"] == 0
    assert hold["trust_state_before"] == hold["trust_state_after"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.pop("H_AB_raw"),
        lambda payload: payload["material_window_estimator_accounting"].update(
            {"primitive_ids": []}
        ),
        lambda payload: payload["estimator_acquisition_plan"].update(
            {"active_gradient_indices_acquired": [0]}
        ),
    ],
)
def test_material_window_prune_source_reuse_fails_closed_without_remeasurement(
    mutation,
):
    payload = _source_workspace()
    mutation(payload)
    with pytest.raises(RuntimeError):
        _normalize_sr_material_window_prune_source_geometry(
            selector_summary=payload,
            post_admission_labels=[
                "old-0",
                "candidate",
                "old-1",
                "old-2",
            ],
            post_admission_theta=np.asarray([0.1, 0.2, 0.3, 0.4]),
        )


def _ledger_key(identity: str) -> EstimatorCallKey:
    return EstimatorCallKey(
        projective_state_fingerprint=f"state-{identity}",
        hamiltonian_fingerprint="hamiltonian-1",
        backend_fingerprint="exact-statevector-backend-v1",
        precision_contract="float64-exact",
        primitive_kind="hamiltonian_expectation",
        observable_or_formula_identity="hamiltonian_expectation_v1",
    )


def test_prune_accounting_partitions_explicit_shared_source_and_both_outcomes():
    ledger = EstimatorCallLedger()
    source_receipt = ledger.record_call(
        _ledger_key("source"),
        component="N_metric",
        consumer_scope="historical_phase3_material_window",
        branch_id=None,
    )
    accepted_branch = _sr_v4_prune_trial_branch_id(
        selector_step=7,
        candidate_index=0,
        candidate_label="accepted-delete",
    )
    rejected_branch = _sr_v4_prune_trial_branch_id(
        selector_step=8,
        candidate_index=1,
        candidate_label="rejected-delete",
    )
    ledger.record_call(
        _ledger_key("accepted"),
        component="N_H_refit",
        consumer_scope="prune_refit",
        branch_id=accepted_branch,
    )
    ledger.record_call(
        _ledger_key("rejected"),
        component="N_H_refit",
        consumer_scope="prune_refit",
        branch_id=rejected_branch,
    )

    def _history_row(
        *, branch_id: str, accepted: bool, depth: int
    ) -> dict[str, object]:
        return {
            "depth": depth,
            "post_admission_prune": {
                "accepted_count": int(accepted),
                "phase1_prune_exact_refit_work_accounting": {
                    "schema": "sr_v4_prune_exact_refit_work_accounting_v1",
                    "classification": (
                        "committed_prune" if accepted else "discarded_prune"
                    ),
                    "estimator_trial_branch_id": branch_id,
                    "nfev": 1,
                },
                "phase1_prune_source_geometry_reuse": {
                    "schema": "sr_material_window_prune_source_geometry_reuse_v1",
                    "primitive_ids": [source_receipt.primitive_id],
                    "incremental_quantum_query_charge": 0,
                    "duplicate_measurement_performed": False,
                },
            },
        }

    views = _sr_v4_prune_estimator_accounting_views(
        ledger=ledger,
        history_rows=[
            _history_row(branch_id=accepted_branch, accepted=True, depth=7),
            _history_row(branch_id=rejected_branch, accepted=False, depth=8),
        ],
    )

    assert views["all_work"]["S_unique"] == 3
    assert views["winning_lineage"]["S_unique"] == 2
    assert views["shared_source_state"]["S_unique"] == 1
    assert views["winning_lineage_excluding_shared_source"]["S_unique"] == 1
    assert views["discarded_prune_only_by_unique_set_difference"]["S_unique"] == 1
    reconciliation = views["primitive_set_reconciliation"]
    assert reconciliation["pairwise_disjoint"] is True
    assert reconciliation["union_equals_all_work"] is True
    assert reconciliation["partition_S_unique"] == reconciliation["all_work_S_unique"]
