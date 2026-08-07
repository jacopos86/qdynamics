from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from pipelines.scaffold.hh_continuation_generators import serialize_polynomial_terms_exyz
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.selector_measurement_proxy import (
    ControllerMeasurementWorkAccumulator,
    ControllerMeasurementWorkRecordRuntime,
    _common_exposure_probe_payload_for_records,
    _controller_work_group_keys_from_records,
    _phase3_batch_summary_supplement_event,
    _logical_operator_probe_count_for_records,
    _record_controller_work_for_records,
    record_joint_selector_workspace_work,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _poly(*labels: str) -> PauliPolynomial:
    return PauliPolynomial(
        "JW",
        [PauliTerm(len(label), ps=label, pc=1.0) for label in labels],
    )


def _term(label: str, *paulis: str) -> AnsatzTerm:
    return AnsatzTerm(label=label, polynomial=_poly(*paulis))


def _feature(**overrides: object) -> CandidateFeatures:
    values: dict[str, object] = dict(
        stage_name="phase3",
        candidate_label="a",
        candidate_family="core",
        candidate_pool_index=0,
        position_id=0,
        append_position=0,
        positions_considered=[0],
        g_signed=0.25,
        g_abs=0.25,
        g_lcb=0.25,
        sigma_hat=0.0,
        F_metric=1.0,
        metric_proxy=1.0,
        novelty=0.8,
        curvature_mode="current_curv",
        novelty_mode="current_novelty",
        refit_window_indices=[0],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=0.25,
        score_version="test_v1",
        phase2_raw_score=0.25,
        full_v2_score=0.25,
        selector_score=0.25,
        phase_score_components={"existing": 1.0},
        actual_fallback_mode="exact_reduced",
    )
    values.update(overrides)
    return CandidateFeatures(**values)


def _runtime(
    pool: Sequence[AnsatzTerm],
    *,
    registry: Mapping[str, Any] | None = None,
    problem_key: str = "generic",
    split_mode: str = "off",
    selection_mode: str = "proxy_child_set_preselection",
    max_subset_size: int = 3,
) -> ControllerMeasurementWorkRecordRuntime:
    return ControllerMeasurementWorkRecordRuntime(
        pool=list(pool),
        pool_generator_registry=dict(registry or {}),
        phase3_enabled=True,
        pool_symmetry_specs=[None] * len(pool),
        problem_key=str(problem_key),
        num_sites=1,
        ordering="block",
        qpb=2,
        phase3_runtime_split_mode_key=str(split_mode),
        phase3_runtime_split_selection_mode_key=str(selection_mode),
        phase3_runtime_split_child_set_symmetry_policy_key="parent",
        phase3_runtime_split_max_subset_size_value=int(max_subset_size),
    )


def test_group_key_resolution_uses_direct_term_and_pool_fallback() -> None:
    x_term = _term("x0", "x")
    z_term = _term("z0", "z")
    runtime = _runtime([x_term, z_term])
    records = [
        {"candidate_term": x_term},
        {"candidate_pool_index": 1},
        {"candidate_pool_index": "not-an-int"},
    ]

    group_keys, records_evaluated, records_with_group_keys = _controller_work_group_keys_from_records(
        records,
        runtime=runtime,
    )

    assert records_evaluated == 3
    assert records_with_group_keys == 2
    assert sorted(group_keys) == ["x", "z"]
    assert _logical_operator_probe_count_for_records(records, runtime=runtime) == 2


def test_common_exposure_identity_mode_counts_parent_records_and_stable_digest() -> None:
    records = [{"candidate_term": _term("x0", "x")}, {"candidate_term": _term("z0", "z")}]
    runtime = _runtime([record["candidate_term"] for record in records])

    payload = _common_exposure_probe_payload_for_records(
        records,
        runtime=runtime,
        expand_runtime_split=False,
    )
    payload_again = _common_exposure_probe_payload_for_records(
        records,
        runtime=runtime,
        expand_runtime_split=False,
    )

    assert payload["common_parent_candidate_count"] == 2
    assert payload["common_expanded_candidate_count"] == 2
    assert payload["common_exposure_operator_probe_count"] == 2
    assert payload["common_universe_manifest_digest"] == payload_again["common_universe_manifest_digest"]


def test_common_exposure_runtime_split_expands_macro_child_sets() -> None:
    macro_term = _term("macro", "xe", "ez")
    serialized_terms = serialize_polynomial_terms_exyz(macro_term.polynomial)
    generator_metadata = {
        "generator_id": "macro-generator",
        "is_macro_generator": True,
        "compile_metadata": {"serialized_terms_exyz": serialized_terms},
    }
    runtime = _runtime(
        [macro_term],
        registry={"macro": generator_metadata},
        split_mode="shortlist_pauli_children_v1",
    )
    records = [
        {
            "candidate_term": macro_term,
            "candidate_pool_index": 0,
            "feature": _feature(
                candidate_label="macro",
                candidate_family="macro_family",
                generator_metadata=generator_metadata,
            ),
        }
    ]

    unsplit = _common_exposure_probe_payload_for_records(
        records,
        runtime=runtime,
        expand_runtime_split=False,
    )
    split = _common_exposure_probe_payload_for_records(
        records,
        runtime=runtime,
        expand_runtime_split=True,
    )

    assert unsplit["common_parent_candidate_count"] == 1
    assert unsplit["common_expanded_candidate_count"] == 1
    assert split["common_parent_candidate_count"] == 1
    assert split["common_expanded_candidate_count"] > split["common_parent_candidate_count"]
    assert split["common_exposure_operator_probe_count"] == split["common_expanded_candidate_count"]
    assert split["common_universe_manifest_digest"] != unsplit["common_universe_manifest_digest"]


def test_record_controller_work_ignores_passive_historical_liveness_and_records_event_fields() -> None:
    x_term = _term("x0", "x")
    runtime = _runtime([x_term])
    records = [{"candidate_term": x_term}]
    accumulator = ControllerMeasurementWorkAccumulator()

    event = _record_controller_work_for_records(
        accumulator,
        runtime=runtime,
        snapshot={"phase_live": {"phase1": False}, "phase_shots": {"phase1": 2}},
        phase="phase1",
        event_kind="phase1_append_probe",
        records=records,
        depth_value=3,
        candidate_count=1,
        evaluated_count=1,
        shortlist_size=1,
        retained_count=1,
        probe_role="gradient",
        actual_operator_probe_count=1,
        common_parent_candidate_count=1,
        common_expanded_candidate_count=1,
        common_exposure_operator_probe_count=1,
        common_universe_manifest_digest="passive-old-liveness",
    )

    assert accumulator.event_count() == 1
    assert event is not None
    assert event["phase"] == "phase1"
    assert event["shot_phase"] == "phase1"
    assert event["event_kind"] == "phase1_append_probe"
    assert event["depth"] == 3
    assert event["nominal_shots_per_group"] == 2
    assert event["records_evaluated"] == 1
    assert event["records_with_group_keys"] == 1
    assert event["actual_operator_probe_count"] == 1
    assert event["common_exposure_operator_probe_count"] == 1
    assert event["common_universe_manifest_digest"] == "passive-old-liveness"
    assert event["method_shortlist_candidate_count"] == 1


def test_record_reuse_event_preserves_exact_reuse_metadata_and_zero_new_work() -> None:
    accumulator = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=7)

    event = accumulator.record_reuse_event(
        phase="phase3",
        event_kind="route_a_child_phase3_metric",
        group_keys=["x", "z"],
        reused_record_count=2,
        records_with_group_keys=2,
        reuse_key="unit-route-a-child-full-feature-key",
        reuse_source_event_kind="route_a_child_phase2_metric",
        source_record_keys=("source-a", "source-b", "source-c"),
        reused_record_keys=("source-a", "source-c"),
        depth=5,
        shortlist_size=2,
        retained_count=1,
    )
    summary = accumulator.summary()

    assert event["actual_operator_probe_count"] == 0
    assert event["actual_evaluated_candidate_count"] == 0
    assert event["shots_new"] == 0.0
    assert event["total_shots_new"] == 0.0
    assert event["shots_reused"] == 14.0
    assert event["reused_operator_probe_count"] == 2
    assert event["measurement_reuse_key"] == "unit-route-a-child-full-feature-key"
    assert event["measurement_reuse_policy"] == "exact_full_feature_record_v1"
    assert event["measurement_reuse_validation_status"] == "exact_match"
    assert event["reuse_source_event_kind"] == "route_a_child_phase2_metric"
    assert event["measurement_reuse_source_record_key_count"] == 3
    assert event["measurement_reuse_record_key_count"] == 2
    assert summary["actual_operator_probe_count"] == 0
    assert summary["reused_operator_probe_count_total"] == 2
    assert summary["shots_new"] == 0.0
    assert summary["shots_reused"] == 14.0
    phase3 = summary["by_phase"]["phase3"]
    assert phase3["actual_operator_probe_count"] == 0
    assert phase3["reused_operator_probe_count_total"] == 2
    assert phase3["measurement_reuse_policy"] == "exact_full_feature_record_v1"
    assert phase3["measurement_reuse_validation_status"] == "exact_match"


def test_record_reuse_event_rejects_keys_not_recorded_by_source_event() -> None:
    accumulator = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=7)

    with pytest.raises(ValueError, match="exact subset"):
        accumulator.record_reuse_event(
            phase="phase3",
            event_kind="route_a_child_phase3_metric",
            group_keys=["x"],
            reused_record_count=1,
            records_with_group_keys=1,
            reuse_key="unit-route-a-child-full-feature-key",
            reuse_source_event_kind="route_a_child_phase2_metric",
            source_record_keys=("source-a",),
            reused_record_keys=("different-branch-key",),
            depth=5,
            shortlist_size=1,
            retained_count=1,
        )


def test_joint_selector_query_supplement_charges_unique_matrix_elements_once() -> None:
    row = {
        "phase3_batch_summary": {
            "schema": "route_a_joint_schur_selector_v1",
            "canonical_selection_stage": "post_child_phase2_joint_selector",
            "candidate_batch_eval_count": 7,
            "reused_child_phase2_singleton_subset_count": 3,
            "geometry_workspace": {
                "query_chargeable_unique_geometry_element_count": 6,
            },
        }
    }
    existing = {
        "events": [
            {
                "phase": "phase2",
                "event_kind": "batch_union_scoring",
                "actual_operator_probe_count": 2,
            }
        ],
        "by_scope": {
            "static_adapt|phase=phase2|event=batch_union_scoring|depth=1": {
                "phase": "phase2",
                "event_kind": "batch_union_scoring",
                "actual_operator_probe_count_total": 2,
            }
        },
    }

    supplement = _phase3_batch_summary_supplement_event(row, existing)

    assert supplement is not None
    assert supplement["phase"] == "phase2"
    assert supplement["probe_role"] == "metric"
    assert supplement["actual_operator_probe_count"] == 4
    assert supplement["candidate_count"] == 4
    detail = supplement["phase3_batch_summary_supplement"]
    assert detail["candidate_batch_eval_count"] == 7
    assert detail["query_chargeable_batch_subset_count"] == 6
    assert detail["existing_batch_union_actual_operator_probe_count"] == 2
    assert detail["supplement_actual_operator_probe_count"] == 4
    assert detail["canonical_joint_selector"] is True
    assert detail["query_charge_basis"] == "unique_full_geometry_elements"


def test_joint_selector_workspace_work_is_recorded_without_an_admission() -> None:
    accumulator = ControllerMeasurementWorkAccumulator(nominal_shots_per_group=7)
    selector_summary = {
        "schema": "route_a_joint_schur_selector_v1",
        "canonical_selection_stage": "post_child_phase2_joint_selector",
        "selected_cardinality": 0,
        "geometry_workspace": {
            "search_population_count": 3,
            "workspace_fingerprint": "workspace-a",
            "workspace_build_mode": "phase2_reuse_plus_required_candidate_pairs_v1",
            "query_chargeable_unique_geometry_element_count": 6,
            "query_chargeable_gradient_repair_count": 2,
            "total_mathematically_required_element_count": 18,
            "reused_phase2_element_count": 12,
            "newly_measured_element_count": 6,
            "required_element_counts": {"G_CC_off_diagonal": 3},
            "reused_element_counts": {"G_CC_diagonal": 3},
            "newly_measured_element_counts": {"G_CC_off_diagonal": 3},
            "matrix_cache_hit_element_count": 12,
            "matrix_cache_miss_element_count": 6,
            "matrix_cache_invalidation_reason_counts": {},
            "required_candidate_pair_count": 3,
            "constructed_candidate_pair_count": 2,
            "reused_cached_candidate_pair_count": 1,
            "joint_pair_cache_hit_count": 1,
            "joint_pair_cache_miss_count": 2,
            "joint_pair_workers_effective": 2,
            "phase2_joint_geometry_reuse_validation": {
                "full_unique_geometry_element_count": 12,
            },
        },
    }

    event = record_joint_selector_workspace_work(
        accumulator,
        snapshot=None,
        selector_summary=selector_summary,
        depth=4,
    )

    assert event is not None
    assert event["phase"] == "phase2"
    assert event["event_kind"] == "batch_union_scoring"
    assert event["actual_operator_probe_count"] == 6
    assert event["retained_count"] == 0
    assert event["candidate_count"] == 3
    assert event["groups_total"] == 0
    assert event["total_mathematically_required_element_count"] == 18
    assert event["reused_phase2_element_count"] == 12
    assert event["newly_measured_element_count"] == 6
    assert event["required_element_counts"] == {"G_CC_off_diagonal": 3}
    assert event["required_candidate_pair_count"] == 3
    assert event["constructed_candidate_pair_count"] == 2
    assert event["reused_cached_candidate_pair_count"] == 1
    assert event["joint_pair_cache_hit_count"] == 1
    assert event["joint_pair_cache_miss_count"] == 2
    assert event["joint_pair_workers_effective"] == 2
    assert event["workspace_build_mode"] == (
        "phase2_reuse_plus_required_candidate_pairs_v1"
    )
    summary = accumulator.summary()
    repair_event = next(
        row
        for row in summary["events"]
        if row["event_kind"] == "joint_selector_gradient_repair"
    )
    assert repair_event["probe_role"] == "gradient"
    assert repair_event["actual_operator_probe_count"] == 2

    supplement = _phase3_batch_summary_supplement_event(
        {"phase3_batch_summary": selector_summary},
        summary,
    )
    assert supplement is None


def test_joint_selector_cached_pair_work_is_not_double_charged() -> None:
    accumulator = ControllerMeasurementWorkAccumulator()
    selector_summary = {
        "schema": "route_a_joint_schur_selector_v1",
        "canonical_selection_stage": "post_child_phase2_joint_selector",
        "geometry_workspace": {
            "search_population_count": 2,
            "workspace_fingerprint": "workspace-pair-cache-hit",
            "query_chargeable_unique_geometry_element_count": 0,
            "query_chargeable_gradient_repair_count": 0,
            "required_candidate_pair_count": 1,
            "constructed_candidate_pair_count": 0,
            "reused_cached_candidate_pair_count": 1,
            "joint_pair_cache_hit_count": 1,
            "joint_pair_cache_miss_count": 0,
        },
    }

    event = record_joint_selector_workspace_work(
        accumulator,
        snapshot=None,
        selector_summary=selector_summary,
        depth=4,
    )

    assert event is None
    assert accumulator.summary().get("actual_operator_probe_count_total", 0) == 0


def test_joint_selector_workspace_work_is_branch_local_and_additive() -> None:
    accumulator = ControllerMeasurementWorkAccumulator()
    selector_summary = {
        "schema": "route_a_joint_schur_selector_v1",
        "canonical_selection_stage": "post_child_phase2_joint_selector",
        "geometry_workspace": {
            "search_population_count": 2,
            "workspace_fingerprint": "workspace-beam",
            "query_chargeable_unique_geometry_element_count": 5,
        },
    }

    for branch_id in (1, 2):
        record_joint_selector_workspace_work(
            accumulator,
            snapshot=None,
            selector_summary=selector_summary,
            depth=3,
            scope_qualifiers={"branch_id": branch_id},
        )

    summary = accumulator.summary()
    assert summary["actual_operator_probe_count_total"] == 10
    scopes = [
        event["work_scope"]
        for event in summary["events"]
        if event["event_kind"] == "batch_union_scoring"
    ]
    assert len(scopes) == 2
    assert len(set(scopes)) == 2
