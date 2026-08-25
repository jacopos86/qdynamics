from __future__ import annotations

from pipelines.contracts.static_provenance import (
    classify_static_physical_operator_lane,
    summarize_static_physical_operator_pool_labels,
)
from pipelines.static_adapt import adapt_pipeline, run_control
from pipelines.static_adapt.lane_routes import (
    physical_lane_route_variant_id_for_problem,
    resolve_static_shortlist_lane_spec,
)
from pipelines.static_adapt.historical_route_identity import (
    ROUTE_ID_A,
    read_historical_route_identity,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
)


def test_h2o_paper_i_profile_is_registered_for_complete_runtime_handoff() -> None:
    assert (
        SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3
        in adapt_pipeline._REGISTERED_COMPLETE_SR_ROUTE_PROFILES
    )


def test_h2o_derivative_pool_satisfies_preserved_controller_provenance() -> None:
    observed = {
        "continuation_mode": "phase3_v1",
        "phase2_novelty_mode": "collective_span_v1",
        "phase3_selector_policy": "algebraic_nested_v1",
        "phase3_selector_geometry_mode": "reduced",
        "algebraic_shortlisting_enabled": True,
        "hardware_resolution_schema": "gradient_resolution_v1",
        "hardware_resolution_mode": "ideal",
        "phase2_raw_score_formula": "DeltaE_TR_raw * N2 / (1 + K2)",
        "canonical_score_formula": "DeltaE_TR * N3 / (1 + K3)",
        "primary_selector_score_key": "full_v2_score",
        "auxiliary_terms_primary_mode": "tie_break_only",
        "phase3_novelty_ablation_mode": "off",
        "phase3_window_relaxation_mode": "reduced",
        "phase3_enable_batching": True,
        "phase3_batch_selection_mode": "reduced_plane",
        "phase3_batch_prefilter_mode": "off",
        "phase3_batch_order_selection_mode": "finite_step_v1",
        "phase3_nested_window_application": "composed_batch_window_v1",
        "phase1_prune_enabled": True,
        "phase1_prune_policy": "recoverability_ladder_v1",
        "phase1_prune_mode": "both",
        "base_pool_key": "full_meta_derivative_resolved_v2",
        "adapt_pool": "full_meta_derivative_resolved_v2",
        "route_variant_id": (
            "route_a_h2o_linear_fd_physical_operator_lanes_v2_derivative_resolved"
        ),
    }
    payload = read_historical_route_identity(
        observed,
        declared_route_id=ROUTE_ID_A,
    )

    assert payload["valid"] is True
    assert (
        payload["required_components"]["base_pool_key"]
        == "full_meta_derivative_resolved_v2"
    )
    validation = adapt_pipeline._validate_resolved_static_route_identity(
        payload,
        declared_route_id=ROUTE_ID_A,
        historical_singleton_overlay_active=True,
        sr_route_profile=SR_ROUTE_PROFILE_H2O_DERIVATIVE_RESOLVED_PAPER_I_V3,
    )
    assert validation["valid"] is True


def test_h2o_linear_fd_physical_operator_lanes_cover_production_pool() -> None:
    labels = [
        "el::uccsd_sing(alpha:0->1)",
        "el::uccsd_dbl(ab:0,2->1,3)",
        "boson::bend::p",
        "coupled::bend::dH_dQ_times_p",
        "coupled::bend::dH_dQ_one_body_factor[0]_times_p",
        "coupled::bend::dH_dQ_two_body_factor[0]_times_p",
        "conditional::bend::q_times_uccsd_sing(alpha:0->1)",
        "conditional::bend::q_times_uccsd_dbl(ab:0,2->1,3)",
    ]
    expected = [
        "electronic_single",
        "electronic_double",
        "vibrational_momentum",
        "vibronic_derivative_momentum",
        "vibronic_one_body_response",
        "vibronic_two_body_response",
        "vibronic_conditional_single",
        "vibronic_conditional_double",
    ]
    assert [
        classify_static_physical_operator_lane(
            f"{label}::split[0]::x",
            problem="molecular_vibronic_h2o_linear_fd",
        )["physical_operator_lane"]
        for label in labels
    ] == expected

    audit = summarize_static_physical_operator_pool_labels(
        labels,
        problem="molecular_vibronic_h2o_linear_fd",
    )
    assert audit["classified_count"] == 8
    assert audit["other_count"] == 0
    assert audit["require_no_other_pass"] is True


def test_h2o_linear_fd_physical_lane_route_contract() -> None:
    spec = resolve_static_shortlist_lane_spec(
        "physical_operator_type",
        problem="molecular_vibronic_h2o_linear_fd",
    )
    assert spec.lanes == (
        "electronic_single",
        "electronic_double",
        "vibrational_momentum",
        "vibronic_derivative_momentum",
        "vibronic_one_body_response",
        "vibronic_two_body_response",
        "vibronic_conditional_single",
        "vibronic_conditional_double",
        "other",
    )
    assert physical_lane_route_variant_id_for_problem(
        "molecular_vibronic_h2o_linear_fd"
    ) == (
        "route_a_h2o_linear_fd_physical_operator_lanes_v2_derivative_resolved"
    )


def test_h2o_electronic_control_physical_lane_route_contract() -> None:
    labels = [
        "uccsd_sing(alpha:0->4)",
        "uccsd_dbl(ab:0,6->4,10)",
    ]
    audit = summarize_static_physical_operator_pool_labels(
        labels,
        problem="molecular_restricted_closed_shell",
    )
    assert audit["lane_counts"] == {
        "electronic_single": 1,
        "electronic_double": 1,
        "other": 0,
    }
    assert audit["require_no_other_pass"] is True
    spec = resolve_static_shortlist_lane_spec(
        "physical_operator_type",
        problem="molecular_restricted_closed_shell",
    )
    assert spec.lanes == ("electronic_single", "electronic_double", "other")
    assert physical_lane_route_variant_id_for_problem(
        "molecular_restricted_closed_shell"
    ) == "route_a_molecular_restricted_physical_operator_lanes_v1"


def test_h2o_nph1_satisfies_singleton_padding_without_projection() -> None:
    contract = adapt_pipeline._historical_singleton_child_padding_contract(
        problem_key="molecular_vibronic_h2o_linear_fd",
        boson_encoding="binary",
        n_ph_max=1,
        projection_active=False,
    )
    assert contract == {
        "satisfied": True,
        "source": "h2o_linear_fd_nph1_full_binary_code_space_v1",
        "projection_active": False,
    }
    assert (
        adapt_pipeline._historical_singleton_child_padding_contract(
            problem_key="molecular_vibronic_h2o_linear_fd",
            boson_encoding="binary",
            n_ph_max=2,
            projection_active=False,
        )["satisfied"]
        is False
    )


def test_h2o_runtime_split_uses_fixture_particle_sector() -> None:
    assert adapt_pipeline._runtime_split_fixed_num_particles(
        problem_key="molecular_vibronic_h2o_linear_fd",
        num_particles=(4, 4),
    ) == (4, 4)
    assert adapt_pipeline._runtime_split_fixed_num_particles(
        problem_key="molecular_restricted_closed_shell",
        num_particles=(4, 4),
    ) == (4, 4)
    assert (
        adapt_pipeline._runtime_split_fixed_num_particles(
            problem_key="spin_boson",
            num_particles=(0, 0),
        )
        is None
    )


def test_h2o_electronic_control_activates_whitened_singleton_route() -> None:
    padding = adapt_pipeline._historical_singleton_child_padding_contract(
        problem_key="molecular_restricted_closed_shell",
        boson_encoding="binary",
        n_ph_max=0,
        projection_active=False,
    )
    assert padding == {
        "satisfied": True,
        "source": "fermion_only_no_binary_padding_v1",
        "projection_active": False,
    }
    assert adapt_pipeline._historical_paper_i_route_compatibility_active(
        problem_key="molecular_restricted_closed_shell",
        static_route_id_key="route_a",
        static_meta_feature_profile="paper_i_production_v1",
        static_lane_route_key="physical_operator_type",
        route_a_funnel_active=False,
        adapt_pool="uccsd",
        adapt_continuation_mode="phase3_v1",
        phase3_runtime_split_mode="shortlist_pauli_children_v1",
        phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
        phase3_runtime_split_max_subset_size=1,
        phase3_runtime_split_subset_sizes=(1,),
        physical_lane_shortlist_factor=3,
        phase1_shortlist_size_base=24,
        phase2_shortlist_size_base=12,
        phase2_shortlist_fraction_base=0.25,
    )


def test_h2o_derivative_resolved_pool_activates_whitened_singleton_route() -> None:
    assert adapt_pipeline._historical_paper_i_route_compatibility_active(
        problem_key="molecular_vibronic_h2o_linear_fd",
        static_route_id_key="route_a",
        static_meta_feature_profile="paper_i_production_v1",
        static_lane_route_key="physical_operator_type",
        route_a_funnel_active=False,
        adapt_pool="full_meta_derivative_resolved_v2",
        adapt_continuation_mode="phase3_v1",
        phase3_runtime_split_mode="shortlist_pauli_children_v1",
        phase3_runtime_split_selection_mode="archival_child_set_forward_v1",
        phase3_runtime_split_max_subset_size=1,
        phase3_runtime_split_subset_sizes=(1,),
        physical_lane_shortlist_factor=3,
        phase1_shortlist_size_base=24,
        phase2_shortlist_size_base=12,
        phase2_shortlist_fraction_base=0.25,
    )


def test_skip_trajectory_avoids_unneeded_dense_hamiltonian_matrix() -> None:
    assert not adapt_pipeline._dense_hamiltonian_matrix_required(
        hilbert_dim=4096,
        dense_eigh_max_dim=8192,
        skip_trajectory=True,
    )
    assert adapt_pipeline._dense_hamiltonian_matrix_required(
        hilbert_dim=4096,
        dense_eigh_max_dim=8192,
        skip_trajectory=False,
    )
    assert not adapt_pipeline._dense_hamiltonian_matrix_required(
        hilbert_dim=16384,
        dense_eigh_max_dim=8192,
        skip_trajectory=False,
    )


def test_h2o_runtime_split_admission_guard_accepts_fixture_sector() -> None:
    gate = {
        "checked": True,
        "passed": True,
        "gate_scope": "fixed_count_sector_invariance_v1",
        "skipped_reason": None,
        "fixed_count_sector": {
            "fixed_num_particles": {"n_up": 4, "n_dn": 4},
        },
    }

    assert adapt_pipeline._runtime_split_admission_guard_contract_satisfied(
        symmetry_gate=gate,
        expected_fixed_num_particles=(4, 4),
    )
    assert not adapt_pipeline._runtime_split_admission_guard_contract_satisfied(
        symmetry_gate=gate,
        expected_fixed_num_particles=(1, 1),
    )


def test_beam_segment_admission_count_survives_metric_prune_depth_drop() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id="h2o",
        target_depth=None,
        max_new_admissions=15,
        wallclock_cap_s=None,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=8,
        current_runtime_parameter_count=8,
        requested_max_depth=15,
        start_time_s=0.0,
    )
    history = [
        {"selected_ops": ["resume_0"]},
        {"selected_ops": ["resume_1"]},
        {"selected_ops": ["new_0"]},
        {"selected_ops": ["new_1"]},
        {"selected_ops": ["new_2"]},
    ]
    run_control._sync_adapt_segment_new_admissions_from_depth(
        state,
        final_depth=9,
        history_rows=history,
        start_history_length=2,
    )
    assert state.new_admissions_count == 3
