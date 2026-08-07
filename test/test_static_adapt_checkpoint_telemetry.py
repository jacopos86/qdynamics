from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import pipelines.static_adapt.adapt_pipeline as hardcoded_adapt
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt import checkpoint_telemetry


def test_checkpoint_telemetry_helpers_remain_available_through_wrappers() -> None:
    helper_names = (
        "_phase3_surface_audit_payload",
        "_active_hh_pool_summary_payload",
        "_scaffold_fingerprint_payload",
        "_optimizer_memory_contract_summary_payload",
        "_controller_runtime_boundary_summary_payload",
    )
    for name in helper_names:
        assert getattr(adapt_pipeline, name) is getattr(checkpoint_telemetry, name)
        assert getattr(hardcoded_adapt, name) is getattr(checkpoint_telemetry, name)


def test_current_jsonable_normalizes_numpy_and_nonfinite_values() -> None:
    payload = checkpoint_telemetry._current_jsonable(
        {
            "int": np.int64(3),
            "float": np.float64(1.25),
            "nan": float("nan"),
            "arr": np.asarray([1, 2], dtype=np.int64),
            "bool": np.bool_(True),
            "object": object(),
        }
    )

    assert payload["int"] == 3
    assert payload["float"] == 1.25
    assert payload["nan"] is None
    assert payload["arr"] == [1, 2]
    assert payload["bool"] is True
    assert isinstance(payload["object"], str)


def test_compact_prune_audit_preserves_checkpoint_contract_defaults() -> None:
    audit = checkpoint_telemetry._compact_prune_audit(
        {
            "enabled": True,
            "permission_open": True,
            "permission_reason": "unit",
            "accepted_count": 2,
            "candidate_count": 5,
            "prune_prefilter_blocked_indices": [1, np.int64(2)],
                "prune_prefilter_blocked_labels": ["a", 3],
                "probe_indices": [4],
                "max_regression": 0.1,
            "prune_tolerance": {
                "effective_tolerance": 0.2,
                "used_component": "delta_num_plus_shot",
            },
            "nfev_formal_manifold_prune_reanchor": 1,
            "formal_manifold_post_prune_reanchor": {
                "schema": "formal_manifold_post_prune_exact_reanchor_v1",
                "state_action": "reset_then_exact_reanchor_committed",
                "curvature_action": "isotropic_reset_on_pruned_frame",
                "qbroyd_action": "reset_on_pruned_frame",
                "nfev": 1,
                "whitening_id": "white-id",
                "frame_id": "frame-id",
                "logical_range_id": "range-id",
                "query_receipt": {
                    "primitive_ids_requested": ["grad-id", "metric-id"],
                    "primitive_ids_reused": [],
                },
            },
        }
    )

    assert audit["enabled"] is True
    assert audit["permission_open"] is True
    assert audit["permission_reason"] == "unit"
    assert audit["accepted_count"] == 2
    assert audit["candidate_count"] == 5
    assert audit["prune_prefilter_blocked_indices"] == [1, 2]
    assert audit["prune_prefilter_blocked_labels"] == ["a", "3"]
    assert audit["probe_indices"] == [4]
    assert "small_angle_pool_indices" not in audit
    assert audit["prune_tolerance_effective"] == 0.2
    assert audit["prune_tolerance_used_component"] == "delta_num_plus_shot"
    assert audit["nfev_formal_manifold_prune_reanchor"] == 1
    reanchor = audit["formal_manifold_post_prune_reanchor"]
    assert reanchor["whitening_id"] == "white-id"
    assert reanchor["frame_id"] == "frame-id"
    assert reanchor["query_primitive_ids_requested"] == [
        "grad-id",
        "metric-id",
    ]

    defaults = checkpoint_telemetry._compact_prune_audit(None)
    assert defaults["enabled"] is False
    assert defaults["permission_reason"] == "unknown"
    assert defaults["accepted_count"] == 0
    assert defaults["formal_manifold_post_prune_reanchor"] is None


def test_history_tail_for_checkpoint_compacts_selected_records_and_route_c_payload() -> None:
    rows = [
        {"selected_op": "old", "energy_after_opt": -1.0},
        {
            "depth": 7,
            "selected_ops": ["op_a", "op_b"],
            "selected_positions": [0, 2],
            "selected_pool_indices": [5, 6],
            "selected_feature_rows": [
                {
                    "candidate_label": "gen_a",
                    "generator_id": "g:a",
                    "runtime_split_child_generator_ids": ["c1"],
                    "route_a_child_identity": "pauli:a",
                    "route_a_global_pauli_identity": "pauli:a",
                    "route_a_child_parent_labels": ["parent-a", "parent-b"],
                    "route_a_child_parent_count": 2,
                    "route_a_child_direction_normalization": {
                        "status": "normalized",
                        "canonical_coefficient_l2_norm": 1.0,
                    },
                },
                {
                    "candidate_label": "gen_b",
                    "parent_generator_id": "parent",
                    "template_id": "tmpl",
                    "runtime_split_mode": "child_set",
                },
            ],
            "energy_before_opt": np.float64(-0.5),
            "energy_after_opt": float("nan"),
            "batch_size": 2,
            "nfev_opt": 17,
            "nfev_seed_probe": 0,
            "initial_energy_nfev": 0,
            "nfev_schur_warm_start_guard": 3,
            "schur_warm_start": {
                "schema": "route_a_joint_step_warm_start_v1",
                "enabled": True,
                "mode": "exact_applied_joint_step_guarded_v1",
                "attempted": True,
                "status": "accepted",
                "chosen_source": "route_a_exact_applied_joint_step_v1",
                "guard_objective_evals": 2,
                "selected_labels": ["op_a", "op_b"],
                "active_parameter_relaxation": [0.1, -0.2],
                "batch_coordinate_step": [0.3, 0.4],
                "selector_applied_predicted_reduction": 0.5,
                "mapped_seed_incumbent_energy": -0.5,
                "mapped_seed_proposal_energy": -0.6,
                "mapped_seed_exact_gain": 0.1,
                "prediction_to_exact_seed_ratio": 5.0,
                "guard": {
                    "proposal_count": 1,
                    "incumbent_energy": -0.5,
                    "chosen_energy": -0.6,
                    "guard_objective_evals": 2,
                    "evaluations": [
                        {"name": "incumbent", "status": "ok", "energy": -0.5},
                        {
                            "name": "route_a_exact_applied_joint_step_v1",
                            "status": "ok",
                            "energy": -0.6,
                        },
                    ],
                },
                "full_selector_summary": {"large": [1] * 1000},
            },
            "outer_nfev": 3,
            "nfev_total_before_step": 5,
            "nfev_total_after_step": 25,
            "nfev_step_total_delta": 20,
            "nit_opt": 2,
            "opt_success": True,
            "opt_message": "ok",
            "controller_measurement_work_proxy": {
                "schema": "controller_measurement_work_proxy_v1",
                "total_shots_new": 3,
                "by_phase": {"phase1": {"actual_operator_probe_count": 3}},
                "events": [{"large": "drop"}],
            },
            "route_c_plateau_acquisition": {
                "event": "successful_unlock",
                "selected_record": {"candidate_label": "nested", "debug_records": ["drop"]},
                "state_after": {
                    "schema": "route_c_plateau_acquisition_v1",
                    "active_episode": False,
                    "dormant_count": 0,
                    "dormant_logical_indices": [],
                    "failed_unlock_count": 0,
                    "unlock_count": 1,
                    "payloads": ["drop"],
                },
            },
            "route_a_trust_region_update": {
                "schema": "route_a_trust_region_update_v1",
                "radius_before": np.float64(0.25),
                "radius_after": np.float64(0.5),
                "update_reason": "binding_radius_realized_displacement_larger",
            },
            "post_admission_prune": {"accepted_count": 1},
        },
    ]

    tail = checkpoint_telemetry._history_tail_for_checkpoint(rows, keep_history_tail=1)

    assert len(tail) == 1
    row = tail[0]
    assert row["depth"] == 7
    assert row["energy_before_opt"] == -0.5
    assert row["energy_after_opt"] is None
    assert row["post_admission_prune"]["accepted_count"] == 1
    assert row["route_a_trust_region_update"] == {
        "schema": "route_a_trust_region_update_v1",
        "radius_before": 0.25,
        "radius_after": 0.5,
        "update_reason": "binding_radius_realized_displacement_larger",
    }
    assert row["selected_records"] == [
        {
            "operator_label": "op_a",
            "generator_label": "gen_a",
            "generator_id": "g:a",
            "parent_generator_id": None,
            "template_id": None,
            "position_id": 0,
            "candidate_pool_index": 5,
            "selection_mode": "",
            "runtime_split_mode": "off",
            "runtime_split_chosen_representation": None,
            "runtime_split_child_generator_ids": ["c1"],
            "route_a_child_identity": "pauli:a",
            "route_a_global_pauli_identity": "pauli:a",
            "route_a_child_parent_labels": ["parent-a", "parent-b"],
            "route_a_child_parent_count": 2,
            "route_a_child_direction_normalization": {
                "status": "normalized",
                "canonical_coefficient_l2_norm": 1.0,
            },
        },
        {
            "operator_label": "op_b",
            "generator_label": "gen_b",
            "generator_id": None,
            "parent_generator_id": "parent",
            "template_id": "tmpl",
            "position_id": 2,
            "candidate_pool_index": 6,
            "selection_mode": "",
            "runtime_split_mode": "child_set",
            "runtime_split_chosen_representation": None,
            "runtime_split_child_generator_ids": [],
            "route_a_child_identity": None,
            "route_a_global_pauli_identity": None,
            "route_a_child_parent_labels": [],
            "route_a_child_parent_count": None,
            "route_a_child_direction_normalization": None,
        },
    ]
    assert row["batch_size"] == 2
    assert row["nfev_opt"] == 17
    assert row["nfev_schur_warm_start_guard"] == 3
    warm_start = row["schur_warm_start"]
    assert warm_start["status"] == "accepted"
    assert warm_start["mapped_seed_exact_gain"] == 0.1
    assert warm_start["prediction_to_exact_seed_ratio"] == 5.0
    assert warm_start["guard"]["guard_objective_evals"] == 2
    assert len(warm_start["guard"]["evaluations"]) == 2
    assert "full_selector_summary" not in warm_start
    assert row["nfev_total_after_step"] == 25
    assert row["nit_opt"] == 2
    assert row["opt_success"] is True
    assert row["opt_message"] == "ok"
    controller_work = row["controller_measurement_work_proxy"]
    assert "events" not in controller_work
    assert controller_work["by_phase"]["phase1"][
        "actual_operator_probe_count"
    ] == 3
    route_c = row["route_c_plateau_acquisition"]
    assert route_c["event"] == "successful_unlock"
    assert route_c["selected_record"]["candidate_label"] == "nested"
    assert route_c["state_after"]["unlock_count"] == 1

    assert checkpoint_telemetry._history_tail_for_checkpoint(rows, keep_history_tail=0) == []


def test_history_checkpoint_retains_compact_exact_outer_transport_certificate() -> None:
    control = {
        "schema": "formal_manifold_exact_outer_control_v1",
        "mode": "shadow_each_outer_v1",
        "available": True,
        "prediction_available": False,
        "controller_state_mutated": False,
        "shadow_does_not_affect_controller": True,
        "diagnostic": {
            "schema": "formal_manifold_exact_shadow_transport_diagnostic_v1",
            "predicted_frame_id": "predicted-frame",
            "exact_frame_id": "exact-frame",
            "exact_source_frame_id": "exact-source-frame",
            "predicted_support_rank": 3,
            "exact_support_rank": 3,
            "exact_source_support_rank": 3,
            "exact_source_spectrum": {
                "raw_metric_sha256": "source-metric-hash",
                "retained_condition_number": 4.0,
                "support_threshold": 1.0e-6,
            },
            "exact_endpoint_spectrum": {
                "raw_metric_sha256": "endpoint-metric-hash",
                "retained_condition_number": 5.0,
                "support_threshold": 2.0e-6,
            },
            "support_rank_match": True,
            "support_projector_defect": 0.02,
            "raw_metric_relative_error_fro": 0.03,
            "predicted_whitener_exact_metric_identity_residual": 0.04,
            "endpoint_frame_registration": {
                "sigma_min": 0.9,
                "singular_values": [1.0, 0.95, 0.9],
            },
            "orientation": {
                "available": True,
                "reason": "exact_cross_state_procrustes_compared",
                "transport_fully_compared": True,
                "transport_error_fro": 0.05,
                "transport_error_spectral": 0.04,
                "exact_registration": [[1.0, 0.0], [0.0, 1.0]],
                "exact_registration_telemetry": {
                    "sigma_min": 0.88,
                    "singular_values": [0.99, 0.88],
                },
            },
            "gradient": {
                "available": True,
                "exact_available": True,
                "comparison_available": False,
                "relative_error": 0.06,
                "cosine": 0.97,
            },
            "curvature": {
                "available": False,
                "reason": "exact_or_predicted_direct_curvature_not_supplied",
            },
        },
        "query_accounting": {
            "source_metric_elements_charged": 6,
            "endpoint_metric_elements_charged": 6,
            "endpoint_gradient_elements_charged": 3,
            "cross_state_tangent_elements_charged": 9,
            "endpoint_primitive_ids": ["endpoint-b", "endpoint-a"],
            "source_primitive_ids": ["source-b", "source-a"],
            "cross_frame_primitive_ids": ["cross-b", "cross-a"],
        },
    }
    compact = checkpoint_telemetry._compact_history_row_for_checkpoint(
        {
            "depth": 4,
            "formal_manifold_warm_start": {
                "exact_outer_control": control,
            },
        },
        fallback_depth=4,
    )["formal_outer_exact_control"]

    assert compact["mode"] == "shadow_each_outer_v1"
    assert compact["prediction_available"] is False
    assert compact["controller_state_mutated"] is False
    assert compact["diagnostic"]["raw_metric_relative_error_fro"] == 0.03
    assert compact["diagnostic"]["exact_source_support_rank"] == 3
    assert compact["diagnostic"]["exact_endpoint_spectrum"][
        "raw_metric_sha256"
    ] == "endpoint-metric-hash"
    assert compact["diagnostic"]["gradient"]["exact_available"] is True
    assert compact["diagnostic"]["orientation"]["transport_error_fro"] == 0.05
    assert "exact_registration" not in compact["diagnostic"]["orientation"]
    assert compact["query_accounting"]["endpoint_primitive_id_inventory"][
        "count"
    ] == 2
    assert compact["query_accounting"]["source_primitive_id_inventory"][
        "count"
    ] == 2
    assert compact["query_accounting"]["cross_frame_primitive_id_inventory"][
        "count"
    ] == 2


def test_surface_rows_summary_deduplicates_relevant_identifiers() -> None:
    summary = checkpoint_telemetry._surface_rows_summary(
        [
            {"candidate_label": "a", "generator_id": "g1", "position_id": 0, "runtime_split_mode": "off"},
            {"candidate_label": "a", "generator_id": "g2", "position_id": 1, "runtime_split_mode": "child"},
            {"candidate_label": "", "generator_id": "g2", "position_id": 1, "runtime_split_mode": "child"},
        ]
    )

    assert summary == {
        "count": 3,
        "operator_labels": ["a"],
        "generator_ids": ["g1", "g2"],
        "position_ids": [0, 1],
        "runtime_split_modes": ["off", "child"],
    }


def test_phase3_surface_audit_payload_preserves_main_and_beam_notation() -> None:
    scored_rows = [
        {"candidate_label": "a", "generator_id": "g1", "position_id": 0, "runtime_split_mode": "off"},
        {"candidate_label": "b", "generator_id": "g2", "position_id": 1, "runtime_split_mode": "child"},
    ]
    retained_rows = [scored_rows[1]]
    admitted_rows = [{"candidate_label": "b", "generator_id": "g2", "position_id": 1}]

    main = checkpoint_telemetry._phase3_surface_audit_payload(
        scored_rows=scored_rows,
        retained_rows=retained_rows,
        admitted_rows=admitted_rows,
        beam_enabled=False,
    )
    beam = checkpoint_telemetry._phase3_surface_audit_payload(
        scored_rows=scored_rows,
        retained_rows=retained_rows,
        admitted_rows=admitted_rows,
        beam_enabled=True,
    )

    assert main["scored_surface_notation"] == "R_3(t)"
    assert main["retained_shortlist_notation"] == "S_3(t)"
    assert main["admitted_set_notation"] == "B_t^*"
    assert main["admitted_set_semantics"] == "reduced_plane_admitted_set"
    assert beam["scored_surface_notation"] == "R_3(b)"
    assert beam["retained_shortlist_notation"] == "S_3(b)"
    assert beam["admitted_set_notation"] == "A_b"
    assert beam["admitted_set_semantics"] == "branch_local_retained_admission_set"
    assert main["scored_surface"] == checkpoint_telemetry._surface_rows_summary(scored_rows)
    assert beam["retained_shortlist"] == checkpoint_telemetry._surface_rows_summary(retained_rows)


def test_active_hh_pool_summary_closes_runtime_split_generator_images() -> None:
    payload = checkpoint_telemetry._active_hh_pool_summary_payload(
        phase1_rows=[{"candidate_label": "parent", "generator_id": "g_parent"}],
        phase2_rows=[
            {
                "candidate_label": "child",
                "runtime_split_parent_label": "parent",
                "generator_id": "g_child",
            },
            {"candidate_label": "independent", "generator_id": "g_independent"},
        ],
        phase3_rows=[
            {
                "candidate_label": "grandchild",
                "runtime_split_parent_label": "child",
                "generator_id": "g_grandchild",
            }
        ],
    )

    assert payload["summary_label"] == "Omega_HH_active"
    assert payload["omega_chain"] == ["Omega_HH^(1)", "Omega_HH^(2)", "Omega_HH^(3)"]
    phase1 = payload["phases"]["phase1"]
    phase2 = payload["phases"]["phase2"]
    phase3 = payload["phases"]["phase3"]
    assert phase1["generator_image_labels_effective"] == ["child", "grandchild", "parent"]
    assert phase1["generator_image_count_effective"] == 3
    assert phase2["generator_image_labels_effective"] == ["child", "grandchild", "independent"]
    assert phase3["generator_image_labels_effective"] == ["grandchild"]
    assert payload["nested_generator_image_inclusion"] == {
        "phase2_in_phase1": False,
        "phase3_in_phase2": True,
        "phase3_in_phase1": True,
    }


def test_scaffold_fingerprint_and_optimizer_memory_contract_payloads() -> None:
    fingerprint = checkpoint_telemetry._scaffold_fingerprint_payload(
        operator_labels=["op_a", "op_b"],
        generator_ids=["g_a", ""],
        num_parameters=2,
    )
    fingerprint_again = checkpoint_telemetry._scaffold_fingerprint_payload(
        operator_labels=["op_a", "op_b"],
        generator_ids=["g_a", ""],
        num_parameters=2,
    )

    assert fingerprint["fingerprint_notation"] == "fp(O_*)"
    assert fingerprint["selected_generator_ids"] == ["g_a"]
    assert fingerprint["fingerprint_sha256"] == fingerprint_again["fingerprint_sha256"]

    memory_state = {
        "available": True,
        "optimizer": "SPSA",
        "parameter_count": 2,
        "source": "memory_cache",
        "remap_events": [{"op": "noop", "i": 0}, *[{"op": "insert", "i": i} for i in range(9)]],
    }
    contract = checkpoint_telemetry._optimizer_memory_contract_summary_payload(
        beam_enabled=True,
        branch_id=4,
        memory_state=memory_state,
        operator_labels=["op_a", "op_b"],
        generator_ids=["g_a"],
        num_parameters=2,
        last_active_subset_source="active_subset",
        last_active_subset_reused=True,
    )
    unavailable = checkpoint_telemetry._optimizer_memory_contract_summary_payload(
        beam_enabled=False,
        branch_id=None,
        memory_state=None,
        operator_labels=["op_a"],
        generator_ids=[],
        num_parameters=1,
        last_active_subset_source=None,
        last_active_subset_reused=False,
    )

    assert contract["beam_enabled"] is True
    assert contract["branch_id"] == 4
    assert contract["memory_available"] is True
    assert contract["memory_optimizer"] == "SPSA"
    assert contract["memory_source"] == "memory_cache"
    assert contract["structural_transport_detected"] is True
    assert contract["observed_transport_mode"] == "canonical_embedding_or_index_remap"
    assert contract["remap_event_count"] == 10
    assert len(contract["remap_event_tail"]) == 8
    assert contract["scaffold_fingerprint"]["fingerprint_sha256"] == fingerprint["fingerprint_sha256"]
    assert unavailable["observed_transport_mode"] == "unavailable"
    assert unavailable["memory_source"] == "unavailable"


def test_controller_runtime_boundary_summary_reads_config_and_payloads() -> None:
    cfg = SimpleNamespace(
        tau_phase1_min=0.1,
        tau_phase1_max=0.2,
        tau_phase2_min=0.3,
        tau_phase2_max=0.4,
        tau_phase3_min=0.5,
        tau_phase3_max=0.6,
        cap_phase1_min=1,
        cap_phase1_max=2,
        cap_phase2_min=3,
        cap_phase2_max=4,
        cap_phase3_min=5,
        cap_phase3_max=6,
        shot_min=7,
        shot_max=8,
        shot_cap_phase1=9,
        shot_cap_phase2=10,
        shot_cap_phase3=11,
    )
    stage_payload = {"controller": "payload"}
    snapshot_payload = {"snapshot": "payload"}

    payload = checkpoint_telemetry._controller_runtime_boundary_summary_payload(
        phase_enabled=True,
        cfg=cfg,
        stage_controller_payload=stage_payload,
        current_snapshot_payload=snapshot_payload,
        beam_enabled=True,
        branch_id=3,
    )

    assert payload["summary_label"] == "appendix_a_runtime_boundary"
    assert payload["beam_enabled"] is True
    assert payload["branch_id"] == 3
    assert payload["phase_enabled"] is True
    assert "selected_scaffold_summary" in payload["symbolic_result_keys"]
    assert "stage_controller" in payload["runtime_controller_keys"]
    assert payload["runtime_law_notation"]["thresholds"] == "tau_k(t)"
    assert payload["runtime_dependencies"] == [
        "available_depth",
        "wall_clock",
        "sampling_budget",
        "device_noise",
    ]
    assert payload["calibration_status"] == "runtime_calibrated_not_symbolic"
    assert payload["configured_bounds"]["tau_phase3_max"] == 0.6
    assert payload["configured_bounds"]["cap_phase3_max"] == 6
    assert payload["configured_bounds"]["shot_cap_phase2"] == 10
    assert "phase_live_hysteresis_enabled" not in payload["configured_bounds"]
    assert payload["stage_controller_payload"] == stage_payload
    assert payload["stage_controller_payload"] is not stage_payload
    assert payload["current_controller_snapshot"] == snapshot_payload
    assert payload["current_controller_snapshot"] is not snapshot_payload
