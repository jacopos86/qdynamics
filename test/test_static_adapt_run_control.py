from __future__ import annotations

import pytest

import pipelines.static_adapt.run_control as run_control


def test_adapt_segment_controls_normalization_and_validation_contract() -> None:
    controls = run_control._resolve_adapt_segment_controls(
        adapt_segment_id="",
        adapt_segment_target_depth="3",
        adapt_segment_max_new_admissions=2.9,
        adapt_segment_wallclock_cap_s="4.5",
        adapt_segment_target_controller_round="45",
    )

    assert controls == run_control._AdaptSegmentControls(
        segment_id=None,
        target_depth=3,
        max_new_admissions=2,
        wallclock_cap_s=4.5,
        target_controller_round=45,
    )

    with pytest.raises(ValueError, match="adapt_segment_target_depth must be >= 0 when provided."):
        run_control._resolve_adapt_segment_controls(
            adapt_segment_id=None,
            adapt_segment_target_depth=-1,
            adapt_segment_max_new_admissions=None,
            adapt_segment_wallclock_cap_s=None,
        )
    with pytest.raises(ValueError, match="adapt_segment_max_new_admissions must be >= 0 when provided."):
        run_control._resolve_adapt_segment_controls(
            adapt_segment_id=None,
            adapt_segment_target_depth=None,
            adapt_segment_max_new_admissions=-1,
            adapt_segment_wallclock_cap_s=None,
        )
    with pytest.raises(ValueError, match="adapt_segment_wallclock_cap_s must be finite and >= 0 when provided."):
        run_control._resolve_adapt_segment_controls(
            adapt_segment_id=None,
            adapt_segment_target_depth=None,
            adapt_segment_max_new_admissions=None,
            adapt_segment_wallclock_cap_s=float("nan"),
        )
    with pytest.raises(
        ValueError,
        match="adapt_segment_target_controller_round must be >= 0 when provided.",
    ):
        run_control._resolve_adapt_segment_controls(
            adapt_segment_id=None,
            adapt_segment_target_depth=None,
            adapt_segment_max_new_admissions=None,
            adapt_segment_wallclock_cap_s=None,
            adapt_segment_target_controller_round=-1,
        )


def test_adapt_segment_initial_clipping_preserves_stop_reason_priority() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id="seg",
        target_depth=4,
        max_new_admissions=0,
        wallclock_cap_s=0.0,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=4,
        current_runtime_parameter_count=7,
        requested_max_depth=10,
        start_time_s=100.0,
    )

    assert state.max_depth_effective == 0
    assert state.initial_stop_reason == "segment_target_depth"
    assert run_control._adapt_segment_controls_resolved_log_payload(
        state=state,
        requested_max_depth=10,
    ) == {
        "segment_id": "seg",
        "start_depth": 4,
        "requested_max_depth": 10,
        "effective_max_depth": 0,
        "target_depth": 4,
        "source_controller_round": 0,
        "target_controller_round": None,
        "max_new_admissions": 0,
        "wallclock_cap_s": 0.0,
        "initial_stop_reason": "segment_target_depth",
    }

    controls = run_control._AdaptSegmentControls(
        segment_id=None,
        target_depth=8,
        max_new_admissions=2,
        wallclock_cap_s=None,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=3,
        current_runtime_parameter_count=3,
        requested_max_depth=10,
        start_time_s=0.0,
    )

    assert state.max_depth_effective == 2
    assert state.initial_stop_reason is None


def test_controller_round_target_caps_round21_resume_independent_of_active_depth() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id="round21_to_round45",
        target_depth=None,
        max_new_admissions=30,
        wallclock_cap_s=None,
        target_controller_round=45,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=12,
        current_runtime_parameter_count=45,
        requested_max_depth=45,
        start_time_s=0.0,
        source_controller_round=21,
    )

    # Active ansatz depth may be below the source round after pruning; the
    # controller-round horizon still permits exactly rounds 22 through 45.
    assert state.max_depth_effective == 24
    assert state.source_controller_round == 21
    assert run_control._adapt_segment_controller_round(
        state,
        segment_round_index=0,
    ) == 22
    assert run_control._adapt_segment_controller_round(
        state,
        segment_round_index=23,
    ) == 45
    assert (
        run_control._adapt_segment_loop_stop_reason(
            state,
            current_depth=12,
            current_controller_round=44,
            now_s=1.0,
        )
        is None
    )
    assert (
        run_control._adapt_segment_loop_stop_reason(
            state,
            current_depth=12,
            current_controller_round=45,
            now_s=1.0,
        )
        == "segment_target_controller_round"
    )
    with pytest.raises(ValueError, match="current_controller_round is required"):
        run_control._adapt_segment_loop_stop_reason(
            state,
            current_depth=12,
            now_s=1.0,
        )

    active_depth_cap = run_control._initialize_adapt_segment_run(
        controls=run_control._AdaptSegmentControls(
            segment_id=None,
            target_depth=14,
            max_new_admissions=30,
            wallclock_cap_s=None,
            target_controller_round=45,
        ),
        current_depth=12,
        current_runtime_parameter_count=45,
        requested_max_depth=45,
        start_time_s=0.0,
        source_controller_round=21,
    )
    assert active_depth_cap.max_depth_effective == 2

    completed = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=12,
        current_runtime_parameter_count=45,
        requested_max_depth=45,
        start_time_s=0.0,
        source_controller_round=45,
    )
    assert completed.max_depth_effective == 0
    assert completed.initial_stop_reason == "segment_target_controller_round"


def test_adapt_segment_loop_stop_reason_priority() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id=None,
        target_depth=5,
        max_new_admissions=2,
        wallclock_cap_s=1.0,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=3,
        current_runtime_parameter_count=3,
        requested_max_depth=10,
        start_time_s=10.0,
    )
    state.new_admissions_count = 2

    assert (
        run_control._adapt_segment_loop_stop_reason(
            state,
            current_depth=5,
            now_s=12.0,
        )
        == "segment_target_depth"
    )
    assert (
        run_control._adapt_segment_loop_stop_reason(
            state,
            current_depth=4,
            now_s=12.0,
        )
        == "segment_max_new_admissions"
    )
    state.new_admissions_count = 1
    assert (
        run_control._adapt_segment_loop_stop_reason(
            state,
            current_depth=4,
            now_s=12.0,
        )
        == "segment_wallclock_cap"
    )


def test_adapt_segment_batch_decision_truncates_with_dict_copies() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id="seg",
        target_depth=None,
        max_new_admissions=2,
        wallclock_cap_s=1.0,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=1,
        current_runtime_parameter_count=1,
        requested_max_depth=5,
        start_time_s=0.0,
    )
    state.new_admissions_count = 1
    records = [{"label": "a"}, {"label": "b"}]

    decision = run_control._resolve_adapt_segment_batch_decision(
        state,
        records=records,
        current_depth=1,
    )

    assert decision.admit_count == 1
    assert decision.remaining_slots == 1
    assert decision.stop_reason is None
    assert decision.truncated_records == [{"label": "a"}]
    assert decision.truncated_records is not None
    assert decision.truncated_records[0] is not records[0]

    state.new_admissions_count = 2
    stop_decision = run_control._resolve_adapt_segment_batch_decision(
        state,
        records=records,
        current_depth=1,
    )

    assert stop_decision.admit_count == 0
    assert stop_decision.stop_reason == "segment_max_new_admissions"


def test_adapt_segment_history_fields_preserve_rollback_omissions() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id="seg",
        target_depth=None,
        max_new_admissions=None,
        wallclock_cap_s=None,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=0,
        current_runtime_parameter_count=0,
        requested_max_depth=5,
        start_time_s=0.0,
    )
    state.cap_truncated_batch = True

    fields = run_control._adapt_segment_history_fields(
        state,
        selected_batch_label_count=3,
    )
    assert fields == {
        "segment_new_admissions_committed": 3,
        "segment_new_admissions_total": 3,
        "segment_id": "seg",
        "segment_cap_truncated_batch": True,
    }

    next_fields = run_control._adapt_segment_history_fields(
        state,
        selected_batch_label_count=2,
    )
    assert next_fields == {
        "segment_new_admissions_committed": 2,
        "segment_new_admissions_total": 5,
        "segment_id": "seg",
        "segment_cap_truncated_batch": True,
    }


def test_adapt_segment_payload_schema_contract() -> None:
    controls = run_control._AdaptSegmentControls(
        segment_id="seg",
        target_depth=4,
        max_new_admissions=2,
        wallclock_cap_s=7.5,
        target_controller_round=4,
    )
    state = run_control._initialize_adapt_segment_run(
        controls=controls,
        current_depth=1,
        current_runtime_parameter_count=2,
        requested_max_depth=10,
        start_time_s=0.0,
        source_controller_round=1,
    )
    state.cap_truncated_batch = True

    payload = run_control._build_adapt_segment_payload(
        state=state,
        final_depth=3,
        final_runtime_parameter_count=4,
        new_admission_records=2,
        stop_reason="segment_max_new_admissions",
        resume_enabled=True,
        resume_mode="scaffold_v1",
        boundary_refit_executed=False,
        compile_smoke_result={"ok": True},
        final_controller_round=3,
    )

    assert payload == {
        "schema_version": "static_hh_adapt_segment_v1",
        "segment_id": "seg",
        "resume_enabled": True,
        "resume_mode": "scaffold_v1",
        "base_depth": 1,
        "base_runtime_parameter_count": 2,
        "final_depth": 3,
        "final_runtime_parameter_count": 4,
        "target_depth": 4,
        "source_controller_round": 1,
        "target_controller_round": 4,
        "final_controller_round": 3,
        "max_new_admissions": 2,
        "wallclock_cap_s": 7.5,
        "new_admission_records": 2,
        "segment_cap_truncated_batch": True,
        "stop_reason": "segment_max_new_admissions",
        "boundary_refit_executed": False,
        "compile_smoke": {"ok": True},
        "no_credentials_serialized": True,
    }


def test_benchmark_target_error_from_energy_accepts_only_finite_values() -> None:
    assert (
        run_control._benchmark_target_error_from_energy(
            energy_value="-1.25",
            reference_energy=-1.0,
        )
        == pytest.approx(0.25)
    )
    assert (
        run_control._benchmark_target_error_from_energy(
            energy_value=None,
            reference_energy=-1.0,
        )
        is None
    )
    assert (
        run_control._benchmark_target_error_from_energy(
            energy_value=float("nan"),
            reference_energy=-1.0,
        )
        is None
    )


def test_benchmark_target_hit_classification_unconfigured_payload_contract() -> None:
    payload = run_control._benchmark_target_hit_classification_payload(
        stop_reason_snapshot="benchmark_abs_delta_e_target",
        target_error=0.0,
        target_threshold=None,
        source="unit",
    )

    assert payload == {
        "schema_version": "static_adapt_target_hit_classification_v1",
        "source": "unit",
        "target_hit_success": False,
        "status": "target_not_requested",
        "non_hit_reason": "benchmark_target_abs_delta_e_not_configured",
        "terminal_stop_reason": "benchmark_abs_delta_e_target",
        "required_stop_reason": "benchmark_abs_delta_e_target",
        "target_configured": False,
        "target_error": 0.0,
        "target_threshold": None,
        "target_error_within_threshold": False,
        "target_error_within_threshold_without_target_stop": False,
    }


def test_benchmark_target_hit_classification_success_payload_contract() -> None:
    payload = run_control._benchmark_target_hit_classification_payload(
        stop_reason_snapshot="benchmark_abs_delta_e_target",
        target_error=0.01,
        target_threshold=0.02,
        source="unit",
    )

    assert payload == {
        "schema_version": "static_adapt_target_hit_classification_v1",
        "source": "unit",
        "target_hit_success": True,
        "status": "target_hit_success",
        "non_hit_reason": None,
        "terminal_stop_reason": "benchmark_abs_delta_e_target",
        "required_stop_reason": "benchmark_abs_delta_e_target",
        "target_configured": True,
        "target_error": 0.01,
        "target_threshold": 0.02,
        "target_error_within_threshold": True,
        "target_error_within_threshold_without_target_stop": False,
    }


def test_benchmark_target_hit_classification_non_hit_statuses() -> None:
    inconsistent = run_control._benchmark_target_hit_classification_payload(
        stop_reason_snapshot="benchmark_abs_delta_e_target",
        target_error=0.03,
        target_threshold=0.02,
        source="unit",
    )
    assert inconsistent["target_hit_success"] is False
    assert inconsistent["status"] == "inconsistent_target_stop_non_hit"
    assert inconsistent["non_hit_reason"] == "benchmark_target_stop_without_in_threshold_error"

    active_frontier = run_control._benchmark_target_hit_classification_payload(
        stop_reason_snapshot=None,
        target_error=0.01,
        target_threshold=0.02,
        source="unit",
    )
    assert active_frontier["status"] == "active_frontier_non_hit"
    assert (
        active_frontier["non_hit_reason"]
        == "active_or_recoverable_frontier_without_benchmark_target_stop"
    )
    assert active_frontier["target_error_within_threshold_without_target_stop"] is True

    terminal_non_hit = run_control._benchmark_target_hit_classification_payload(
        stop_reason_snapshot="max_depth",
        target_error=0.01,
        target_threshold=0.02,
        source="unit",
    )
    assert terminal_non_hit["status"] == "non_hit_diagnostic"
    assert terminal_non_hit["non_hit_reason"] == "terminal_stop_reason_not_target_hit:max_depth"
    assert terminal_non_hit["target_error_within_threshold_without_target_stop"] is True
