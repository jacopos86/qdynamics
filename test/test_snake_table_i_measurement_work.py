#!/usr/bin/env python3
"""Tests for SNAKE Table-I measurement-work normalization metadata."""

from __future__ import annotations

import json

import pytest

from pipelines.exact_bench.snake_table_i_measurement_work import (
    _load_source_payload_map,
    enrich_snake_support_payload,
    normalize_snake_measurement_work_row,
    snake_algorithmic_work_from_payload,
    snake_controller_shot_proxy_from_payload,
    snake_deterministic_shot_proxy_from_payload,
    snake_fair_expanded_work_from_payload,
    snake_mechanism_resolved_work_from_payload,
)
from pipelines.exact_bench.table_i_first_hit_sidecars import (
    build_snake_first_hit_sidecar_for_payload,
    inventory_summary_payload,
)
from pipelines.reporting.build_paper_i_hh_child_fairness_pdf import (
    SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
    _terminal_snake_cost_cache_is_current,
)


def _candidate_ledger_fields(candidate_count: int, *, event_count: int = 1) -> dict:
    return {
        "candidate_work_ledger_schema": "controller_candidate_work_ledger_v1",
        "candidate_work_ledger_status": "explicit_candidate_work_ledger_v1",
        "candidate_work_event_count": int(event_count),
        "candidate_work_missing_event_count": 0,
        "candidate_count_total": int(candidate_count),
        "evaluated_count_total": int(candidate_count),
        "pre_shortlist_count_total": int(candidate_count),
        "shortlist_size_total": int(candidate_count),
        "retained_count_total": int(candidate_count),
        "rejected_count_total": 0,
        "candidate_work_ledger_scope": "event_records_measured_v1",
        "candidate_work_ledger_scopes": {"event_records_measured_v1": int(event_count)},
    }


def _common_exposure_fields(count: int) -> dict:
    return {
        "common_exposure_operator_probe_count": int(count),
        "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
        "common_exposure_stage": "post_common_eligibility_post_expansion_pre_method_filter",
        "common_exposure_policy_id": "trajectory_conditioned_full_child_common_exposure_v1",
        "expansion_policy_id": "unit_child_expansion_v1",
        "eligibility_policy_id": "unit_common_eligibility_v1",
        "deduplication_policy_id": "unit_dedup_v1",
        "probe_enumerator_id": "unit_operator_probe_enumerator_v1",
    }


def _actual_probe_fields(count: int, *, role: str = "gradient") -> dict:
    return {
        "actual_operator_probe_count": int(count),
        "operator_probe_event_schema": "paper_i_operator_probe_event_v2",
        "work_contract_id": "paper_i_hh_operator_probe_contract_v2",
        "operator_probe_charge_basis": "logical_estimator_request_pre_grouping_v1",
        "probe_role": str(role),
    }


def test_terminal_snake_cost_cache_requires_work_semantics_version() -> None:
    old_cache = {"source_sha256": "abc"}
    current_cache = {
        "source_sha256": "abc",
        "work_semantics_version": SNAKE_TERMINAL_WORK_SEMANTICS_VERSION,
    }

    assert _terminal_snake_cost_cache_is_current(old_cache, "abc") is False
    assert _terminal_snake_cost_cache_is_current(current_cache, "abc") is True
    assert _terminal_snake_cost_cache_is_current(current_cache, "def") is False


def _minimal_first_hit_source_payload() -> dict:
    return {
        "adapt_vqe": {
            "hf_bitstring_qn_to_q0": "00",
            "paper_i_first_crossing": {
                "schema": "paper_i_first_crossing_v1",
                "status": "reached",
                "reached": True,
                "tau_phys": 2e-4,
                "benchmark_id": "hh_L2_clean_weak",
                "history_position_tau": 1,
                "k_tau": 1,
                "operator_count_at_crossing": 1,
                "primary_error_at_crossing": 1e-4,
            },
            "history": [
                {
                    "accepted": True,
                    "selected_ops": ["opA"],
                    "selected_positions": [0],
                    "logical_parameters_added_this_step": 1,
                    "logical_num_parameters_after_opt": 1,
                    "energy_after_opt": -1.0,
                    "S_alg_at_crossing": 7,
                    "nfev_opt": 4,
                    "nfev_seed_probe": 1,
                    "controller_measurement_work_proxy": {
                        "schema": "controller_measurement_work_proxy_v1",
                        "source": "native_controller_work",
                        "source_kind": "native_controller_work",
                        "legacy_fallback_used": False,
                        "records_evaluated": 8,
                        "records_with_group_keys": 8,
                        "shots_new": 8,
                        "total_shots_new": 8,
                        **_candidate_ledger_fields(8),
                        "by_phase": {
                            "phase1": {
                                "records_with_group_keys": 6,
                                "groups_total": 6,
                                **_actual_probe_fields(6, role="gradient"),
                            },
                            "phase2": {
                                "records_with_group_keys": 2,
                                "groups_total": 2,
                                **_actual_probe_fields(2, role="metric"),
                            },
                        },
                    },
                }
            ],
            "parameterization": {
                "mode": "logical_runtime_parameterization",
                "logical_operator_count": 1,
                "runtime_parameter_count": 1,
                "blocks": [
                    {
                        "candidate_label": "opA",
                        "logical_index": 0,
                        "runtime_start": 0,
                        "runtime_count": 1,
                        "runtime_terms_exyz": [
                            {"pauli_exyz": "xz", "coeff_re": 1.0, "coeff_im": 0.0, "nq": 2}
                        ],
                    }
                ],
            },
        }
    }


def _native_runtime_payload() -> dict:
    return {
        "adapt_vqe": {
            "controller_measurement_work_summary": {
                "schema": "controller_measurement_work_proxy_v1",
                "source": "native_controller_live_decision_work_v1",
                "source_kind": "native_controller_work",
                "legacy_fallback_used": False,
                **_candidate_ledger_fields(11, event_count=3),
                "by_phase": {
                    "phase1": {
                        "records_with_group_keys": 6,
                        "groups_total": 6,
                        **_actual_probe_fields(6, role="gradient"),
                    },
                    "phase2": {
                        "records_with_group_keys": 2,
                        "groups_total": 2,
                        **_actual_probe_fields(2, role="metric"),
                    },
                    "phase3": {
                        "records_with_group_keys": 3,
                        "groups_total": 3,
                        **_actual_probe_fields(3, role="metric"),
                    },
                },
            },
            "history": [{"nfev_opt": 4}, {"nfev_opt": 5}],
            "resume_boundary_refit": {"nfev": 1},
            "final_full_refit": {"nfev": 2},
            "nfev_total": 20,
        }
    }


def _selected_logical_runtime_payload() -> dict:
    return {
        "adapt_vqe": {
            "adapt_selected_logical_filter": {
                "schema": "adapt_selected_logical_pool_filter_v1",
                "applied": True,
                "fallback_to_full_pool": False,
                "pool_size_before": 46,
                "pool_size_after": 2,
            },
            "controller_measurement_work_summary": {
                "schema": "controller_measurement_work_proxy_v1",
                "source": "native_controller_live_decision_work_v1",
                "source_kind": "native_controller_work",
                "legacy_fallback_used": False,
                **_candidate_ledger_fields(54, event_count=4),
                "by_phase": {
                    "phase1": {
                        "records_with_group_keys": 2,
                        "groups_total": 2,
                        **_actual_probe_fields(2, role="gradient"),
                    },
                    "phase2": {
                        "records_with_group_keys": 3,
                        "groups_total": 3,
                        **_actual_probe_fields(3, role="metric"),
                    },
                    "phase3": {
                        "records_with_group_keys": 3,
                        "groups_total": 3,
                        **_actual_probe_fields(3, role="metric"),
                    },
                },
            },
            "history": [],
            "resume_boundary_refit": {"executed": False},
            "final_full_refit": {"executed": False},
            "nfev_total": 0,
        }
    }


def _prefix_runtime_payload() -> dict:
    return {
        "adapt_vqe": {
            "history": [
                {
                    "nfev_opt": 4,
                    "nfev_seed_probe": 1,
                    "controller_measurement_work_proxy": {
                        "schema": "controller_measurement_work_proxy_v1",
                        "source": "native_controller_work",
                        "source_kind": "native_controller_work",
                        "legacy_fallback_used": False,
                        "records_evaluated": 8,
                        "records_with_group_keys": 8,
                        "shots_new": 8,
                        "total_shots_new": 8,
                        **_candidate_ledger_fields(8),
                        "by_phase": {
                            "phase1": {
                                "records_with_group_keys": 6,
                                "groups_total": 6,
                                **_actual_probe_fields(6, role="gradient"),
                            },
                            "phase2": {
                                "records_with_group_keys": 2,
                                "groups_total": 2,
                                **_actual_probe_fields(2, role="metric"),
                            },
                        },
                    },
                },
                {
                    "nfev_opt": 5,
                    "nfev_seed_probe": 2,
                    "controller_measurement_work_proxy": {
                        "schema": "controller_measurement_work_proxy_v1",
                        "source": "native_controller_work",
                        "source_kind": "native_controller_work",
                        "legacy_fallback_used": False,
                        "records_evaluated": 3,
                        "records_with_group_keys": 3,
                        "shots_new": 3,
                        "total_shots_new": 3,
                        **_candidate_ledger_fields(3),
                        "by_phase": {
                            "phase3": {
                                "records_with_group_keys": 3,
                                "groups_total": 3,
                                **_actual_probe_fields(3, role="metric"),
                            },
                        },
                    },
                },
            ],
            "resume_boundary_refit": {"nfev": 99},
            "final_full_refit": {"nfev": 101},
            "nfev_total": 999,
        }
    }


def _beam_aggregate_runtime_payload() -> dict:
    payload = _prefix_runtime_payload()
    adapt = payload["adapt_vqe"]
    adapt["resume_boundary_refit"] = {"nfev": 1}
    adapt["final_full_refit"] = {"nfev": 2}
    adapt["nfev_total"] = 1000
    adapt["controller_measurement_work_summary"] = {
        "schema": "controller_measurement_work_proxy_v1",
        "source": "native_controller_live_decision_work_v1",
        "source_kind": "native_controller_work",
        "legacy_fallback_used": False,
        "beam_run_scope": "all_expanded_scored_branches",
        "winner_history_scope": "winner_lineage_only",
        **_candidate_ledger_fields(200, event_count=5),
        "by_phase": {
            "phase1": {
                "records_with_group_keys": 100,
                "groups_total": 100,
                **_actual_probe_fields(100, role="gradient"),
            },
            "phase2": {
                "records_with_group_keys": 20,
                "groups_total": 20,
                **_actual_probe_fields(20, role="metric"),
            },
            "phase3": {
                "records_with_group_keys": 30,
                "groups_total": 30,
                **_actual_probe_fields(30, role="metric"),
            },
        },
    }
    return payload


def _add_by_scope(summary: dict, scopes: list[tuple[str, int, str]]) -> None:
    summary["by_scope"] = {
        f"static_adapt|phase={phase}|event={event}|depth={idx}": {
            "schema": "controller_measurement_work_proxy_v1",
            "source": "native_controller_work",
            "source_kind": "native_controller_work",
            "legacy_fallback_used": False,
            "records_evaluated": int(count),
            "records_with_group_keys": int(count),
            "shots_new": int(count),
            "total_shots_new": int(count),
            **_candidate_ledger_fields(int(count)),
            **_actual_probe_fields(int(count), role="gradient" if phase in {"phase0", "phase1"} else "metric"),
        }
        for idx, (phase, count, event) in enumerate(scopes, start=1)
    }


def _route_a_child_reuse_runtime_payload() -> dict:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    summary.update(_candidate_ledger_fields(10, event_count=3))
    summary["by_phase"] = {
        "phase1": {
            "records_with_group_keys": 4,
            "groups_total": 4,
            **_actual_probe_fields(4, role="gradient"),
        },
        "phase2": {
            "records_with_group_keys": 3,
            "groups_total": 3,
            **_actual_probe_fields(3, role="metric"),
        },
        "phase3": {
            "records_evaluated": 3,
            "records_with_group_keys": 3,
            "groups_total": 3,
            "groups_reused": 3,
            "shots_total": 300,
            "shots_reused": 300,
            "shots_new": 0,
            "total_shots_new": 0,
            **_actual_probe_fields(0, role="metric"),
        },
    }
    _add_by_scope(
        summary,
        [
            ("phase1", 4, "route_a_child_phase1_gradient"),
            ("phase2", 3, "route_a_child_phase2_metric"),
            ("phase3", 0, "route_a_child_phase3_metric"),
        ],
    )
    reuse_scope = next(key for key in summary["by_scope"] if "route_a_child_phase3_metric" in key)
    summary["by_scope"][reuse_scope].update(
        {
            "records_evaluated": 3,
            "records_with_group_keys": 3,
            "groups_total": 3,
            "groups_reused": 3,
            "shots_total": 300,
            "shots_reused": 300,
            "shots_new": 0,
            "total_shots_new": 0,
            "measurement_reuse_key": "unit-route-a-child-full-feature-key",
            "measurement_reuse_policy": "exact_full_feature_record_v1",
            "measurement_reuse_validation_status": "exact_match",
            "reuse_source_event_kind": "route_a_child_phase2_metric",
            "reused_operator_probe_count_total": 3,
            **_candidate_ledger_fields(3),
            "evaluated_count_total": 0,
            **_actual_probe_fields(0, role="metric"),
        }
    )
    return payload


def test_snake_algorithmic_work_terminal_scope_uses_canonical_runtime_components() -> None:
    work, audit = snake_algorithmic_work_from_payload(
        _native_runtime_payload(),
        scope="terminal",
        source_label="unit.terminal",
    )

    assert audit["status"] == "ok"
    assert audit["scope"] == "terminal"
    assert work["S_alg_status"] == "ok"
    assert work["S_alg"] == 31.0
    assert work["S_alg_N_H_outer_eval"] == 8.0
    assert work["S_alg_N_grad_probe"] == 6.0
    assert work["S_alg_N_metric_probe"] == 5.0
    assert work["S_alg_N_H_refit_eval"] == 12.0
    assert work["table_i_measurement_event_ledger"]["component_totals"]["N_H_refit_eval"] == 12.0
    assert work["S_actual"] == 31.0
    assert work["S_actual_status"] == "ok"
    assert work["table_i_measurement_event_ledger"]["component_source_kind"] == "actual_operator_probe_components"


def test_route_a_child_phase3_exact_reuse_adds_no_metric_or_s_alg_work() -> None:
    payload = _route_a_child_reuse_runtime_payload()

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.route_a_child_reuse",
    )
    mechanism, mechanism_audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.route_a_child_reuse",
    )

    assert audit["status"] == "ok"
    assert work["S_alg_N_grad_probe"] == 4.0
    assert work["S_alg_N_metric_probe"] == 3.0
    assert work["S_alg"] == 27.0
    assert mechanism_audit["status"] == "ok"
    assert mechanism["mechanism_resolution_status"] == "exact"
    assert mechanism["measurement_work"]["gradient"]["route_a_child_phase1_gradient"] == 4.0
    assert mechanism["measurement_work"]["metric"]["route_a_child_phase2_metric"] == 3.0
    assert mechanism["measurement_work"]["metric"]["route_a_child_phase3_metric"] == 0.0
    assert mechanism["mechanism_algorithmic_work"]["S_alg_N_metric_probe"] == 3.0
    assert mechanism["mechanism_algorithmic_work"]["S_alg"] == 27.0
    phase3_operand = next(
        event
        for event in mechanism["formula_operands"]["event_records"]
        if event["event_kind"] == "route_a_child_phase3_metric"
    )
    assert phase3_operand["actual_operator_probe_count"] == 0.0
    assert phase3_operand["count_status"] == "ok"
    assert phase3_operand["evaluated_count_total"] == 0


def test_snake_mechanism_work_reconciles_by_scope_without_promoting_exposure_counts() -> None:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    summary["candidate_count_total"] = 9999
    _add_by_scope(
        summary,
        [
            ("phase1", 6, "phase1_append_probe"),
            ("phase2", 2, "phase2_rerank_records"),
            ("phase3", 3, "phase3_reduced_geometry_rerank"),
        ],
    )

    work, audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.mechanism.by_scope",
    )

    assert audit["status"] == "ok"
    assert work["status"] == "ok"
    assert work["S_alg"] == 31.0
    assert work["candidate_exposure"]["candidate_count_total"] == 9999
    assert work["measurement_work"]["gradient"]["phase1_append_probe"] == 6.0
    assert work["measurement_work"]["metric"]["phase2_rerank_unclassified"] == 2.0
    assert work["measurement_work"]["metric"]["phase3_reduced_geometry_scoring"] == 3.0
    assert work["measurement_work"]["H"]["H_total"] == 20.0
    assert work["reconciliation"]["status"] == "ok"
    assert work["mechanism_resolution_status"] == "partial"
    assert work["partial_mechanism_reconstruction"] is True
    assert work["requires_formula_reconstruction"] is True
    assert work["mechanism_resolution_detail"] == "phase2_metric_subsplit_requires_formula_reconstruction"
    mechanism_alg = work["mechanism_algorithmic_work"]
    assert mechanism_alg["publishable"] is False
    assert mechanism_alg["status"] == "requires_phase2_formula_reconstruction"
    assert mechanism_alg["S_alg"] is None
    assert mechanism_alg["coarse_S_alg"] == 31.0


def test_snake_mechanism_work_publishable_when_metric_bins_are_exact() -> None:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    _add_by_scope(
        summary,
        [
            ("phase1", 6, "phase1_append_probe"),
            ("phase3", 5, "phase3_reduced_geometry_rerank"),
        ],
    )

    work, audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.mechanism.exact",
    )

    assert audit["status"] == "ok"
    assert work["status"] == "ok"
    assert work["mechanism_resolution_status"] == "exact"
    assert work["requires_formula_reconstruction"] is False
    mechanism_alg = work["mechanism_algorithmic_work"]
    assert mechanism_alg["publishable"] is True
    assert mechanism_alg["status"] == "ok"
    assert mechanism_alg["S_alg"] == 31.0
    assert mechanism_alg["S_alg_N_metric_probe"] == 5.0


def test_snake_mechanism_work_by_phase_only_stays_unclassified() -> None:
    payload = _native_runtime_payload()
    payload["adapt_vqe"]["controller_measurement_work_summary"]["by_scope"] = {}

    work, audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.mechanism.by_phase",
    )

    assert audit["status"] == "ok"
    assert work["status"] == "ok"
    assert work["S_alg"] == 31.0
    assert work["mechanism_event_source"]["source"] == "by_phase"
    assert work["mechanism_resolution_status"] == "partial"
    assert work["measurement_work"]["gradient"]["unclassified_gradient"] == 6.0
    assert work["measurement_work"]["metric"]["unclassified_metric"] == 5.0


def test_snake_mechanism_work_raw_events_keep_singular_exposure_aliases_diagnostic() -> None:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    summary["events"] = [
        {
            "phase": "phase1",
            "event_kind": "phase1_append_probe",
            "candidate_count": 600,
            "evaluated_count": 6,
            "shortlist_size": 4,
            "retained_count": 2,
            **_actual_probe_fields(6, role="gradient"),
        },
        {
            "phase": "phase2",
            "event_kind": "phase2_rerank_records",
            "candidate_count": 200,
            "evaluated_count": 2,
            "shortlist_size": 2,
            "retained_count": 2,
            **_actual_probe_fields(2, role="metric"),
        },
        {
            "phase": "phase3",
            "event_kind": "phase3_reduced_geometry_rerank",
            "candidate_count": 300,
            "evaluated_count": 3,
            "shortlist_size": 3,
            "retained_count": 1,
            **_actual_probe_fields(3, role="metric"),
        },
    ]

    work, audit = snake_mechanism_resolved_work_from_payload(payload, scope="terminal")

    assert audit["status"] == "ok"
    assert work["S_alg"] == 31.0
    assert work["mechanism_event_source"]["source"] == "events"
    first_event = work["formula_operands"]["event_records"][0]
    assert first_event["candidate_count_total"] == 600
    assert first_event["evaluated_count_total"] == 6
    assert first_event["shortlist_size_total"] == 4
    assert first_event["retained_count_total"] == 2
    assert work["measurement_work"]["gradient"]["phase1_append_probe"] == 6.0


def test_snake_mechanism_work_blocks_by_scope_missing_typed_probe_count() -> None:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    summary["by_scope"] = {
        "static_adapt|phase=phase1|event=phase1_append_probe|depth=1": {
            "records_evaluated": 999,
            "records_with_group_keys": 999,
            **_candidate_ledger_fields(999),
        }
    }

    work, audit = snake_mechanism_resolved_work_from_payload(payload, scope="terminal")

    assert audit["status"] == "invalid_reconciliation"
    assert work["status"] == "invalid_reconciliation"
    assert work["mechanism_resolution_status"] == "blocked"
    assert work["S_alg"] == 31.0
    assert work["invalid_event_records"][0]["detail"]["status"] == "missing_actual_operator_probe_count"
    assert work["reconciliation"]["mismatches"]["S_alg_N_grad_probe"]["expected"] == 6.0


def test_snake_mechanism_work_blocks_charge_basis_mismatch() -> None:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    _add_by_scope(
        summary,
        [
            ("phase1", 6, "phase1_append_probe"),
            ("phase2", 2, "phase2_rerank_records"),
            ("phase3", 3, "phase3_reduced_geometry_rerank"),
        ],
    )
    first_scope = next(iter(summary["by_scope"].values()))
    first_scope["operator_probe_charge_basis"] = "grouped_measurement_basis_not_s_alg"

    work, audit = snake_mechanism_resolved_work_from_payload(payload, scope="terminal")

    assert audit["status"] == "invalid_reconciliation"
    assert work["mechanism_resolution_status"] == "blocked"
    assert work["invalid_event_records"][0]["detail"]["status"] == "policy_mismatch"


def test_snake_algorithmic_work_final_prefix_matches_terminal_scope() -> None:
    payload = _native_runtime_payload()

    terminal_work, terminal_audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.terminal.parity",
    )
    prefix_work, prefix_audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=len(payload["adapt_vqe"]["history"]),
        source_label="unit.terminal.parity",
    )

    assert terminal_audit["status"] == "ok"
    assert prefix_audit["status"] == "ok"
    assert prefix_audit["scope"] == "display_prefix"
    assert prefix_audit["terminal_scope_equivalence"] is True
    assert prefix_work["S_alg"] == terminal_work["S_alg"]
    assert prefix_work["S_alg_N_H_outer_eval"] == terminal_work["S_alg_N_H_outer_eval"]
    assert prefix_work["S_alg_N_grad_probe"] == terminal_work["S_alg_N_grad_probe"]
    assert prefix_work["S_alg_N_metric_probe"] == terminal_work["S_alg_N_metric_probe"]
    assert prefix_work["S_alg_N_H_refit_eval"] == terminal_work["S_alg_N_H_refit_eval"]


def test_snake_algorithmic_work_can_preserve_exact_final_history_prefix() -> None:
    payload = _prefix_runtime_payload()

    prefix_work, prefix_audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=len(payload["adapt_vqe"]["history"]),
        source_label="unit.final.accepted.history.prefix",
        allow_terminal_scope_equivalence=False,
    )

    assert prefix_audit["status"] == "ok"
    assert prefix_audit["scope"] == "display_prefix"
    assert "terminal_scope_equivalence" not in prefix_audit
    assert prefix_work["S_alg"] is not None
    assert prefix_audit["history_position"] == len(payload["adapt_vqe"]["history"])


def test_snake_algorithmic_work_prefix_counts_batch_candidate_evaluations() -> None:
    baseline_payload = _prefix_runtime_payload()
    supplemented_payload = json.loads(json.dumps(baseline_payload))
    _add_by_scope(
        baseline_payload["adapt_vqe"]["history"][1]["controller_measurement_work_proxy"],
        [("phase3", 3, "batch_union_scoring")],
    )
    _add_by_scope(
        supplemented_payload["adapt_vqe"]["history"][1]["controller_measurement_work_proxy"],
        [("phase3", 3, "batch_union_scoring")],
    )
    supplemented_payload["adapt_vqe"]["history"][1]["phase3_batch_summary"] = {
        "selection_mode": "combinatorial_reduced_plane",
        "candidate_batch_eval_count": 8,
        "selected_count": 1,
        "selected": False,
    }

    baseline_work, baseline_audit = snake_algorithmic_work_from_payload(
        baseline_payload,
        scope="display_prefix",
        history_position=2,
        source_label="unit.prefix.batch_eval.baseline",
    )
    supplemented_work, supplemented_audit = snake_algorithmic_work_from_payload(
        supplemented_payload,
        scope="display_prefix",
        history_position=2,
        source_label="unit.prefix.batch_eval.supplemented",
    )

    assert baseline_audit["status"] == "ok"
    assert supplemented_audit["status"] == "ok"
    assert baseline_work["S_alg_N_metric_probe"] == 5.0
    assert supplemented_work["S_alg_N_metric_probe"] == 10.0
    assert supplemented_work["S_alg"] == baseline_work["S_alg"] + 5.0


def test_snake_algorithmic_work_prefix_zero_batch_candidate_evaluations_unchanged() -> None:
    baseline_payload = _prefix_runtime_payload()
    zero_payload = json.loads(json.dumps(baseline_payload))
    zero_payload["adapt_vqe"]["history"][1]["phase3_batch_summary"] = {
        "selection_mode": "greedy_reduced_plane",
        "candidate_batch_eval_count": 0,
    }

    baseline_work, baseline_audit = snake_algorithmic_work_from_payload(
        baseline_payload,
        scope="display_prefix",
        history_position=2,
        source_label="unit.prefix.batch_eval.zero_baseline",
    )
    zero_work, zero_audit = snake_algorithmic_work_from_payload(
        zero_payload,
        scope="display_prefix",
        history_position=2,
        source_label="unit.prefix.batch_eval.zero",
    )

    assert baseline_audit["status"] == "ok"
    assert zero_audit["status"] == "ok"
    assert zero_work["S_alg_N_metric_probe"] == baseline_work["S_alg_N_metric_probe"]
    assert zero_work["S_alg"] == baseline_work["S_alg"]


def test_snake_algorithmic_work_beam_terminal_uses_winner_history_not_aggregate() -> None:
    payload = _beam_aggregate_runtime_payload()

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.beam.winner_terminal",
    )

    assert audit["status"] == "ok"
    assert audit["scope"] == "terminal"
    assert audit["S_alg_work_scope"] == "winner_lineage_terminal"
    assert audit["S_alg_row_policy"] == "beam_terminal_winner_history_v1"
    assert work["work_semantics_version"] == SNAKE_TERMINAL_WORK_SEMANTICS_VERSION
    assert work["S_alg_status"] == "ok"
    assert work["S_alg_work_scope"] == "winner_lineage_terminal"
    assert work["S_alg_N_H_outer_eval"] == 3.0
    assert work["S_alg_N_grad_probe"] == 6.0
    assert work["S_alg_N_metric_probe"] == 5.0
    assert work["S_alg_N_H_refit_eval"] == 12.0
    assert work["S_alg"] == 26.0
    assert work["S_beam_search_total"] == 1150.0
    assert work["S_beam_search_total_status"] == "ok"
    assert audit["beam_search_total_reconstruction"]["promoted_to_row_s_alg"] is False


def test_snake_algorithmic_work_beam_terminal_counts_rejected_candidates_on_winner_prefix() -> None:
    payload = _beam_aggregate_runtime_payload()
    first, second = payload["adapt_vqe"]["history"]

    # Phase counts are deliberately larger than the retained counts so this
    # catches regressions that charge only the admitted operator instead of all
    # actual candidate probes on the winning branch prefix.
    first_proxy = first["controller_measurement_work_proxy"]
    first_proxy.update(
        candidate_count_total=11,
        evaluated_count_total=11,
        pre_shortlist_count_total=11,
        shortlist_size_total=4,
        retained_count_total=2,
        rejected_count_total=9,
    )
    first_proxy["by_phase"]["phase1"].update(
        records_with_group_keys=90,
        groups_total=90,
        actual_operator_probe_count=9,
        candidate_count_total=9,
        evaluated_count_total=9,
        retained_count_total=1,
        rejected_count_total=8,
    )
    first_proxy["by_phase"]["phase2"].update(
        records_with_group_keys=20,
        groups_total=20,
        actual_operator_probe_count=2,
        candidate_count_total=2,
        evaluated_count_total=2,
        retained_count_total=1,
        rejected_count_total=1,
    )

    second_proxy = second["controller_measurement_work_proxy"]
    second_proxy.update(
        candidate_count_total=4,
        evaluated_count_total=4,
        pre_shortlist_count_total=4,
        shortlist_size_total=2,
        retained_count_total=1,
        rejected_count_total=3,
    )
    second_proxy["by_phase"]["phase3"].update(
        records_with_group_keys=40,
        groups_total=40,
        actual_operator_probe_count=4,
        candidate_count_total=4,
        evaluated_count_total=4,
        retained_count_total=1,
        rejected_count_total=3,
    )

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.beam.rejected_candidates_winner_prefix",
    )

    assert audit["status"] == "ok"
    assert work["S_alg_status"] == "ok"
    assert audit["S_alg_row_policy"] == "beam_terminal_winner_history_v1"
    assert audit["beam_search_total_reconstruction"]["promoted_to_row_s_alg"] is False
    assert work["S_alg_work_scope"] == "winner_lineage_terminal"
    assert work["S_alg_N_H_outer_eval"] == 3.0
    assert work["S_alg_N_grad_probe"] == 9.0
    assert work["S_alg_N_metric_probe"] == 6.0
    assert work["S_alg_N_H_refit_eval"] == 12.0
    assert work["S_alg"] == 30.0
    assert work["S_beam_search_total"] == 1150.0
    assert work["S_alg"] != work["S_beam_search_total"]
    ledger = work["table_i_measurement_event_ledger"]["candidate_work_ledger"]
    assert ledger["candidate_count_total"] == 15
    assert ledger["evaluated_count_total"] == 15
    assert ledger["retained_count_total"] == 3
    assert ledger["rejected_count_total"] == 12


def test_snake_mechanism_work_beam_terminal_uses_winner_not_aggregate_scope() -> None:
    payload = _beam_aggregate_runtime_payload()
    aggregate = payload["adapt_vqe"]["controller_measurement_work_summary"]
    _add_by_scope(
        aggregate,
        [
            ("phase1", 100, "phase1_append_probe"),
            ("phase2", 20, "phase2_rerank_records"),
            ("phase3", 30, "phase3_reduced_geometry_rerank"),
        ],
    )
    first, second = payload["adapt_vqe"]["history"]
    _add_by_scope(
        first["controller_measurement_work_proxy"],
        [
            ("phase1", 6, "phase1_append_probe"),
            ("phase2", 2, "phase2_rerank_records"),
        ],
    )
    _add_by_scope(
        second["controller_measurement_work_proxy"],
        [("phase3", 3, "phase3_reduced_geometry_rerank")],
    )

    work, audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.mechanism.beam",
    )

    assert audit["status"] == "ok"
    assert work["S_alg"] == 26.0
    assert work["candidate_exposure"]["candidate_count_total"] == 11
    assert work["beam_search_total_provenance"]["S_beam_search_total"] == 1150.0
    assert work["mechanism_scope_summary"]["beam_aggregate_summary_excluded"] is True
    assert work["measurement_work"]["gradient"]["phase1_append_probe"] == 6.0
    assert work["measurement_work"]["metric"]["phase2_rerank_unclassified"] == 2.0
    assert work["measurement_work"]["metric"]["phase3_reduced_geometry_scoring"] == 3.0


def test_snake_algorithmic_work_beam_terminal_blocks_missing_winner_outer_nfev() -> None:
    payload = _beam_aggregate_runtime_payload()
    for row in payload["adapt_vqe"]["history"]:
        row.pop("nfev_seed_probe", None)

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.beam.missing_winner_outer",
    )

    assert audit["status"] == "missing_prefix_outer_nfev"
    assert audit["S_beam_search_total"] == 1150.0
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "missing_prefix_outer_nfev"
    assert work["S_beam_search_total"] == 1150.0
    assert work["S_beam_search_total_status"] == "ok"


def test_snake_mechanism_work_display_prefix_uses_prefix_history_rows() -> None:
    payload = _prefix_runtime_payload()
    first, second = payload["adapt_vqe"]["history"]
    _add_by_scope(
        first["controller_measurement_work_proxy"],
        [
            ("phase1", 6, "phase1_append_probe"),
            ("phase2", 2, "phase2_rerank_records"),
        ],
    )
    _add_by_scope(
        second["controller_measurement_work_proxy"],
        [("phase3", 3, "phase3_reduced_geometry_rerank")],
    )

    work, audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=1,
        source_label="unit.mechanism.prefix",
    )

    assert audit["status"] == "ok"
    assert work["S_alg"] == 13.0
    assert work["mechanism_scope_summary"]["summary_source"] == "controller_proxy_from_history_rows(prefix)"
    assert work["measurement_work"]["gradient"]["phase1_append_probe"] == 6.0
    assert work["measurement_work"]["metric"]["phase2_rerank_unclassified"] == 2.0
    assert work["measurement_work"]["metric"]["phase3_reduced_geometry_scoring"] == 0.0
    assert work["measurement_work"]["H"]["H_total"] == 5.0


def test_snake_mechanism_work_reconstructs_phase2_window_formula_components() -> None:
    def _payload(*, gamma: float, gain_mode: str, novelty_ablation: str = "off") -> dict:
        payload = _prefix_runtime_payload()
        payload["settings"] = {
            "phase2_gamma_N": gamma,
            "phase2_selector_gain_mode": gain_mode,
        }
        payload["adapt_vqe"]["phase3_novelty_ablation_mode"] = novelty_ablation
        payload["adapt_vqe"]["history"][0]["scored_surface_records"] = [
            {"phase2_geometry_window_indices": [0, 1], "schur_window_indices": [0, 1]},
            {"phase2_geometry_window_indices": [0, 1], "schur_window_indices": [0, 1]},
        ]
        return payload

    novelty_only, _audit = snake_mechanism_resolved_work_from_payload(
        _payload(gamma=1.0, gain_mode="unit_gain_v1"),
        scope="display_prefix",
        history_position=1,
    )
    second_order_only, _audit = snake_mechanism_resolved_work_from_payload(
        _payload(gamma=0.0, gain_mode="trust_region_v1"),
        scope="display_prefix",
        history_position=1,
    )
    combined, _audit = snake_mechanism_resolved_work_from_payload(
        _payload(gamma=1.0, gain_mode="trust_region_v1"),
        scope="display_prefix",
        history_position=1,
    )

    assert novelty_only["mechanism_algorithmic_work"]["status"] == "ok_phase2_window_formula_v1"
    assert second_order_only["mechanism_algorithmic_work"]["status"] == "ok_phase2_window_formula_v1"
    assert combined["mechanism_algorithmic_work"]["status"] == "ok_phase2_window_formula_v1"
    assert novelty_only["mechanism_algorithmic_work"]["S_alg_N_metric_probe"] == 9.0
    assert second_order_only["mechanism_algorithmic_work"]["S_alg_N_metric_probe"] == 11.0
    assert combined["mechanism_algorithmic_work"]["S_alg_N_metric_probe"] == 18.0
    assert novelty_only["mechanism_algorithmic_work"]["S_alg"] == 20.0
    assert second_order_only["mechanism_algorithmic_work"]["S_alg"] == 22.0
    assert combined["mechanism_algorithmic_work"]["S_alg"] == 29.0


def test_snake_mechanism_formula_publishes_when_raw_windows_exist() -> None:
    payload = _prefix_runtime_payload()
    payload["settings"] = {
        "phase2_gamma_N": 1.0,
        "phase2_selector_gain_mode": "trust_region_v1",
    }
    payload["adapt_vqe"]["history"][0]["scored_surface_records"] = [
        {"phase2_geometry_window_indices": [0], "schur_window_indices": [0]},
        {"phase2_geometry_window_indices": [0], "schur_window_indices": [0]},
    ]
    work, _audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=1,
    )

    mechanism_alg = work["mechanism_algorithmic_work"]
    assert mechanism_alg["publishable"] is True
    assert mechanism_alg["status"] == "ok_phase2_window_formula_v1"
    assert mechanism_alg["S_alg_N_metric_probe"] == 10.0
    assert mechanism_alg["S_alg"] == 21.0


def test_snake_mechanism_formula_preserves_non_phase2_metric_work() -> None:
    baseline = _prefix_runtime_payload()
    supplemented = json.loads(json.dumps(baseline))
    for payload in (baseline, supplemented):
        payload["settings"] = {
            "phase2_gamma_N": 1.0,
            "phase2_selector_gain_mode": "trust_region_v1",
        }
        _add_by_scope(
            payload["adapt_vqe"]["history"][0]["controller_measurement_work_proxy"],
            [
                ("phase1", 6, "phase1_append_probe"),
                ("phase2", 2, "phase2_rerank_records"),
            ],
        )
        payload["adapt_vqe"]["history"][0]["scored_surface_records"] = [
            {"phase2_geometry_window_indices": [0], "schur_window_indices": [0]},
            {"phase2_geometry_window_indices": [0], "schur_window_indices": [0]},
        ]
    supplemented["adapt_vqe"]["history"][0]["phase3_batch_summary"] = {
        "selection_mode": "greedy_reduced_plane",
        "candidate_batch_eval_count": 8,
        "selected_count": 1,
    }

    baseline_work, _audit = snake_mechanism_resolved_work_from_payload(
        baseline,
        scope="display_prefix",
        history_position=1,
    )
    supplemented_work, _audit = snake_mechanism_resolved_work_from_payload(
        supplemented,
        scope="display_prefix",
        history_position=1,
    )

    baseline_alg = baseline_work["mechanism_algorithmic_work"]
    supplemented_alg = supplemented_work["mechanism_algorithmic_work"]
    assert baseline_alg["S_alg_N_metric_probe"] == 10.0
    assert supplemented_alg["S_alg_N_metric_probe"] == 18.0
    assert supplemented_alg["S_alg"] == baseline_alg["S_alg"] + 8.0
    components = supplemented_alg["phase2_formula_reconstruction"]["components"]
    assert components["phase2_replaced_coarse_metric"] == 2.0
    assert components["non_phase2_metric_preserved"] == 8.0


def test_snake_mechanism_work_final_prefix_ignores_terminal_by_scope_for_mechanism_bins() -> None:
    payload = _prefix_runtime_payload()
    first, second = payload["adapt_vqe"]["history"]
    _add_by_scope(
        first["controller_measurement_work_proxy"],
        [
            ("phase1", 6, "phase1_append_probe"),
            ("phase2", 2, "phase2_rerank_records"),
        ],
    )
    _add_by_scope(
        second["controller_measurement_work_proxy"],
        [("phase3", 3, "phase3_reduced_geometry_rerank")],
    )
    payload["adapt_vqe"]["resume_boundary_refit"] = {"executed": False}
    payload["adapt_vqe"]["final_full_refit"] = {"executed": False}
    payload["adapt_vqe"]["nfev_total"] = 12
    terminal_summary = {
        "schema": "controller_measurement_work_proxy_v1",
        "source": "native_controller_live_decision_work_v1",
        "source_kind": "native_controller_work",
        "legacy_fallback_used": False,
        **_candidate_ledger_fields(11, event_count=3),
        "by_phase": {
            "phase1": {"records_with_group_keys": 6, "groups_total": 6, **_actual_probe_fields(6, role="gradient")},
            "phase2": {"records_with_group_keys": 2, "groups_total": 2, **_actual_probe_fields(2, role="metric")},
            "phase3": {"records_with_group_keys": 3, "groups_total": 3, **_actual_probe_fields(3, role="metric")},
        },
    }
    _add_by_scope(
        terminal_summary,
        [
            ("phase1", 600, "phase1_append_probe"),
            ("phase2", 200, "phase2_rerank_records"),
            ("phase3", 300, "phase3_reduced_geometry_rerank"),
        ],
    )
    payload["adapt_vqe"]["controller_measurement_work_summary"] = terminal_summary

    work, audit = snake_mechanism_resolved_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=2,
        source_label="unit.mechanism.final_prefix",
    )

    assert audit["status"] == "ok"
    assert work["S_alg"] == 23.0
    assert work["mechanism_scope_summary"]["summary_source"] == "controller_proxy_from_history_rows(prefix)"
    assert work["measurement_work"]["gradient"]["phase1_append_probe"] == 6.0
    assert work["measurement_work"]["metric"]["phase2_rerank_unclassified"] == 2.0
    assert work["measurement_work"]["metric"]["phase3_reduced_geometry_scoring"] == 3.0


def test_snake_fair_work_rejects_group_fields_as_common_exposure() -> None:
    payload = _native_runtime_payload()

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.group_fields_not_common",
    )
    fair_work, fair_audit = snake_fair_expanded_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.group_fields_not_common",
    )

    assert audit["status"] == "ok"
    assert work["S_actual"] == 31.0
    assert work["S_alg"] == 31.0
    assert fair_audit["status"] == "missing_common_exposure_ledger"
    assert fair_work["S_actual"] == 31
    assert fair_work["S_fair"] is None
    assert fair_work["S_fair_status"] == "missing_common_exposure_ledger"
    assert fair_work["S_common_exposure"] is None
    assert fair_work["S_common_exposure_status"] == "missing_common_exposure_ledger"
    details = fair_work["table_i_measurement_event_ledger"]["runtime_reconstruction"][
        "controller_phase_common_exposure_operator_probe_counts"
    ]
    assert details["phase1"]["forbidden_operator_probe_aliases_present"] == ["groups_total"]


def test_snake_fair_work_uses_explicit_common_exposure_operator_probe_ledger() -> None:
    payload = _native_runtime_payload()
    by_phase = payload["adapt_vqe"]["controller_measurement_work_summary"]["by_phase"]
    by_phase["phase1"].update(_common_exposure_fields(60))
    by_phase["phase2"].update(_common_exposure_fields(20))
    by_phase["phase3"].update(_common_exposure_fields(30))

    fair_work, fair_audit = snake_fair_expanded_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.common_exposure",
    )

    assert fair_audit["status"] == "ok"
    assert fair_work["S_actual"] == 31
    assert fair_work["S_common_exposure"] == 130
    assert fair_work["S_fair"] == 130
    assert fair_work["S_fair_status"] == "ok"
    assert fair_work["S_fair_source"] == "S_common_exposure"
    assert fair_work["S_common_exposure_components"] == {
        "N_H_outer_eval": 8,
        "N_grad_probe_common_exposure": 60,
        "N_metric_probe_common_exposure": 50,
        "N_H_refit_eval": 12,
    }


def test_snake_fair_work_blocks_bare_pre_shortlist_common_substitute() -> None:
    payload = _native_runtime_payload()
    by_phase = payload["adapt_vqe"]["controller_measurement_work_summary"]["by_phase"]
    by_phase["phase1"]["pre_shortlist_count_total"] = 600
    by_phase["phase2"]["pre_shortlist_count_total"] = 200
    by_phase["phase3"]["pre_shortlist_count_total"] = 300

    fair_work, fair_audit = snake_fair_expanded_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.bare_pre_shortlist",
    )

    assert fair_audit["status"] == "missing_common_exposure_ledger"
    assert fair_work["S_fair"] is None
    details = fair_work["table_i_measurement_event_ledger"]["runtime_reconstruction"][
        "controller_phase_common_exposure_operator_probe_counts"
    ]
    assert "pre_shortlist_count_total" in details["phase1"]["forbidden_operator_probe_aliases_present"]


def test_snake_weak_weak_migration_reconstructs_actual_and_rejects_group_common_value() -> None:
    payload = {
        "adapt_vqe": {
            "controller_measurement_work_summary": {
                "schema": "controller_measurement_work_proxy_v1",
                "source": "native_controller_live_decision_work_v1",
                "source_kind": "native_controller_work",
                "legacy_fallback_used": False,
                **_candidate_ledger_fields(12_434, event_count=4),
                "by_phase": {
                    "phase0": {
                        "records_with_group_keys": 5_578,
                        "groups_total": 8_645,
                        **_actual_probe_fields(5_578, role="gradient"),
                    },
                    "phase1": {
                        "records_with_group_keys": 5_577,
                        "groups_total": 8_645,
                        **_actual_probe_fields(5_577, role="gradient"),
                    },
                    "phase2": {
                        "records_with_group_keys": 780,
                        "groups_total": 905,
                        **_actual_probe_fields(780, role="metric"),
                    },
                    "phase3": {
                        "records_with_group_keys": 499,
                        "groups_total": 378,
                        **_actual_probe_fields(499, role="metric"),
                    },
                },
            },
            "history": [{"nfev_opt": 12_431}],
            "resume_boundary_refit": {"executed": False},
            "final_full_refit": {"executed": False},
            "nfev_total": 20_853 + 12_431,
        }
    }

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.weak_weak_migration",
    )
    fair_work, fair_audit = snake_fair_expanded_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.weak_weak_migration",
    )

    assert audit["status"] == "ok"
    assert work["S_alg"] == 45_718.0
    assert work["S_actual"] == 45_718.0
    assert work["S_alg_N_grad_probe"] == 11_155.0
    assert work["S_alg_N_metric_probe"] == 1_279.0
    assert work["S_alg"] != 51_857
    assert fair_audit["status"] == "missing_common_exposure_ledger"
    assert fair_work["S_actual"] == 45_718
    assert fair_work["S_fair"] is None
    assert fair_work["S_common_exposure"] is None
    assert fair_work["S_fair_status"] == "missing_common_exposure_ledger"


def test_snake_algorithmic_work_blocks_old_runtime_payload_without_candidate_ledger() -> None:
    payload = _native_runtime_payload()
    summary = payload["adapt_vqe"]["controller_measurement_work_summary"]
    for key in list(_candidate_ledger_fields(0)):
        summary.pop(key, None)

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.old_runtime",
    )

    assert audit["status"] == "missing_explicit_candidate_work_ledger"
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "missing_explicit_candidate_work_ledger"
    assert work["S_alg_lower_bound"] is None
    assert work["algorithmic_measurement_work"]["S_alg"] is None
    assert "S_alg_lower_bound" not in work["algorithmic_measurement_work"]


def test_snake_algorithmic_work_blocks_records_with_group_keys_only_payload() -> None:
    payload = _native_runtime_payload()
    by_phase = payload["adapt_vqe"]["controller_measurement_work_summary"]["by_phase"]
    for phase in ("phase1", "phase2", "phase3"):
        by_phase[phase].pop("actual_operator_probe_count", None)
        by_phase[phase].pop("operator_probe_event_schema", None)
        by_phase[phase].pop("work_contract_id", None)
        by_phase[phase].pop("operator_probe_charge_basis", None)
        by_phase[phase].pop("probe_role", None)

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="terminal",
        source_label="unit.group_keys_only",
    )

    assert audit["status"] == "invalid_controller_phase_counts"
    assert work["S_alg"] is None
    assert work["S_norm"] is None
    assert work["S_alg_status"] == "invalid_controller_phase_counts"
    assert audit["phases"]["phase1"]["status"] == "missing_actual_operator_probe_count"
    assert audit["phases"]["phase1"]["records_with_group_keys_diagnostic"] == 6.0


def test_snake_algorithmic_work_prefix_scope_uses_history_rows_not_terminal_summary() -> None:
    work, audit = snake_algorithmic_work_from_payload(
        _prefix_runtime_payload(),
        scope="display_prefix",
        history_position=1,
        source_label="unit.prefix",
    )

    assert audit["status"] == "ok"
    assert audit["scope"] == "display_prefix"
    assert audit["history_position"] == 1
    assert audit["prefix_refit_sources"]["history"]["nfev"] == 4.0
    assert work["S_alg_status"] == "ok"
    assert work["S_alg"] == 13.0
    assert work["S_alg_N_H_outer_eval"] == 1.0
    assert work["S_alg_N_grad_probe"] == 6.0
    assert work["S_alg_N_metric_probe"] == 2.0
    assert work["S_alg_N_H_refit_eval"] == 4.0


def test_snake_algorithmic_work_prefix_scope_blocks_missing_merged_phase_count() -> None:
    payload = _prefix_runtime_payload()
    del payload["adapt_vqe"]["history"][0]["controller_measurement_work_proxy"]["by_phase"]["phase1"][
        "actual_operator_probe_count"
    ]

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=1,
        source_label="unit.prefix.missing_phase_count",
    )

    assert audit["status"] == "missing_controller_numeric_fields"
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "missing_controller_numeric_fields"
    missing = audit["controller_numeric_validation"]["missing_required_fields"]
    assert any(
        item["field"] == "actual_operator_probe_count"
        and item["phase"] == "phase1"
        and item["paper_i_blocking"] is True
        and item["reason"] == "missing_required_typed_operator_probe_count"
        for item in missing
    )


def test_snake_algorithmic_work_prefix_scope_blocks_nonfinite_merged_phase_count() -> None:
    payload = _prefix_runtime_payload()
    payload["adapt_vqe"]["history"][0]["controller_measurement_work_proxy"]["by_phase"]["phase1"][
        "actual_operator_probe_count"
    ] = float("nan")

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=1,
        source_label="unit.prefix.nonfinite_phase_count",
    )

    assert audit["status"] == "invalid_controller_numeric_fields"
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "invalid_controller_numeric_fields"
    invalid = audit["controller_numeric_validation"]["invalid_fields"]
    assert any(
        item["field"] == "actual_operator_probe_count"
        and item["phase"] == "phase1"
        and item["paper_i_blocking"] is True
        and item["reason"] == "nonfinite_controller_integer_field"
        for item in invalid
    )


def test_snake_deterministic_shot_proxy_terminal_complete_inputs_use_comparator_formula() -> None:
    payload = _native_runtime_payload()
    payload["hamiltonian_pauli_term_count"] = 11
    payload["shots_per_pauli_term_proxy"] = 10

    fields, audit = snake_deterministic_shot_proxy_from_payload(
        payload,
        scope="terminal",
        source_label="unit.terminal.shots",
    )

    assert audit["status"] == "ok"
    assert audit["display_policy"] == "comparable_deterministic_total_shot_proxy"
    assert audit["component_counts"]["energy_eval_count_proxy"] == 20
    assert audit["component_counts"]["gradient_operator_probe_count_proxy"] == 6
    assert audit["component_counts"]["metric_operator_probe_count_proxy"] == 5
    assert fields["shots_total"] == 10 * 11 * (20 + 6 + 5)
    assert fields["static_shot_estimate_status"] == "deterministic_proxy_not_physical_shots"
    assert fields["shot_proxy_formula"] == (
        "shots_total = shots_per_pauli_term_proxy * hamiltonian_pauli_term_count * "
        "(energy_eval_count_proxy + gradient_operator_probe_count_proxy + metric_operator_probe_count_proxy)"
    )


def test_snake_deterministic_shot_proxy_blocks_missing_shots_per_term_but_keeps_s_alg_valid() -> None:
    payload = _native_runtime_payload()
    payload["hamiltonian_pauli_term_count"] = 11

    fields, audit = snake_deterministic_shot_proxy_from_payload(
        payload,
        scope="terminal",
        source_label="unit.terminal.missing_shots_per_term",
    )

    assert fields == {}
    assert audit["status"] == "missing_shots_per_pauli_term_proxy"
    assert audit["S_alg_status"] == "ok"
    assert audit["S_alg"] == 31.0


def test_snake_deterministic_shot_proxy_never_uses_controller_proxy_as_shots_total() -> None:
    payload = _native_runtime_payload()
    payload["measurement_shots_proxy"] = 999999
    payload["shot_cost_proxy"] = 888888
    payload["controller_shot_proxy"] = 777777

    fields, audit = snake_deterministic_shot_proxy_from_payload(
        payload,
        scope="terminal",
        source_label="unit.terminal.legacy_only",
    )

    assert fields == {}
    assert audit["status"] == "missing_hamiltonian_pauli_term_count"
    assert audit["S_alg_status"] == "ok"
    assert audit["legacy_work_proxies"]["root.measurement_shots_proxy"] == 999999
    assert "shots_total" not in audit


def test_snake_algorithmic_work_prefix_scope_requires_history_position() -> None:
    work, audit = snake_algorithmic_work_from_payload(_prefix_runtime_payload(), scope="display_prefix")

    assert audit["status"] == "history_position_required"
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "history_position_required"


def test_snake_algorithmic_work_prefix_scope_blocks_missing_outer_eval_breakdown() -> None:
    payload = _prefix_runtime_payload()
    for row in payload["adapt_vqe"]["history"]:
        row.pop("nfev_seed_probe", None)

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=1,
        source_label="unit.prefix.missing_outer",
    )

    assert audit["status"] == "missing_prefix_outer_nfev"
    assert audit["outer_nfev"]["missing_policy"] == "block_without_explicit_prefix_outer_fields_per_history_row"
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "missing_prefix_outer_nfev"


def test_snake_algorithmic_work_prefix_scope_blocks_any_missing_outer_eval_row() -> None:
    payload = _prefix_runtime_payload()
    payload["adapt_vqe"]["history"][1].pop("nfev_seed_probe", None)

    work, audit = snake_algorithmic_work_from_payload(
        payload,
        scope="display_prefix",
        history_position=2,
        source_label="unit.prefix.partial_missing_outer",
    )

    assert audit["status"] == "missing_prefix_outer_nfev"
    assert audit["outer_nfev"]["missing_history_indices"] == [2]
    assert audit["outer_nfev"]["sources"] == [{"history_index": 1, "fields": ["nfev_seed_probe"], "nfev": 1.0}]
    assert work["S_alg"] is None
    assert work["S_alg_status"] == "missing_prefix_outer_nfev"


def test_first_hit_sidecar_generator_builds_qiskit_snake_sidecar(tmp_path) -> None:
    pytest.importorskip("qiskit")
    source = tmp_path / "result.json"
    source_payload = _minimal_first_hit_source_payload()
    source.write_text(json.dumps(source_payload, sort_keys=True) + "\n", encoding="utf-8")
    wrapper = {
        "result": {
            "result_json": str(source),
            "family": "hh",
            "case_id": "hh_L2_clean_weak",
            "benchmark_id": "hh_L2_clean_weak",
            "paper_i_first_crossing": source_payload["adapt_vqe"]["paper_i_first_crossing"],
        }
    }

    sidecar, inventory = build_snake_first_hit_sidecar_for_payload(
        payload=wrapper,
        payload_path=None,
        threshold=2e-4,
    )

    assert inventory["status"] == "sidecar_generated"
    assert inventory["compiled_resource_status"] == "ok"
    assert inventory["work_resource_status"] == "ok"
    assert sidecar is not None
    assert sidecar["schema"] == "snake_first_crossing_compiled_cost_v1"
    assert sidecar["source_kind"] == "snake_qiskit_compiled_first_hit_ansatz_circuit"
    assert sidecar["first_hit_cost_source_kind"] == "snake_qiskit_compiled_first_hit_ansatz_circuit"
    assert sidecar["compiled_circuit_stats_status"] == "ok"
    assert sidecar["compiled_resource_qiskit_validated"] is True
    assert sidecar["compiled_count_2q_total"] is not None
    assert sidecar["compiled_depth_2q_total"] is not None
    assert sidecar["compiled_depth_total"] >= sidecar["compiled_depth_2q_total"]
    assert sidecar["S_alg"] == 13.0
    assert sidecar["legacy_work_proxies"] == {"S_alg_at_crossing": 7.0}
    assert sidecar["source_result_path"] == str(source)
    assert sidecar["source_result_sha256"]
    assert sidecar["reconstructability_status"] == "ok"


def test_first_hit_sidecar_generator_rejects_scalar_crossing_s_alg_without_runtime_ledger(tmp_path) -> None:
    pytest.importorskip("qiskit")
    source = tmp_path / "result.json"
    source_payload = _minimal_first_hit_source_payload()
    row = source_payload["adapt_vqe"]["history"][0]
    row.pop("nfev_opt", None)
    row.pop("nfev_seed_probe", None)
    row.pop("controller_measurement_work_proxy", None)
    source.write_text(json.dumps(source_payload, sort_keys=True) + "\n", encoding="utf-8")
    wrapper = {
        "result": {
            "result_json": str(source),
            "family": "hh",
            "case_id": "hh_L2_clean_weak",
            "benchmark_id": "hh_L2_clean_weak",
            "paper_i_first_crossing": source_payload["adapt_vqe"]["paper_i_first_crossing"],
        }
    }

    sidecar, inventory = build_snake_first_hit_sidecar_for_payload(
        payload=wrapper,
        payload_path=None,
        threshold=2e-4,
    )

    assert inventory["status"] == "sidecar_generated_work_missing"
    assert inventory["work_resource_status"] == "missing"
    assert inventory["rerun_needed_reason"] == "missing_prefix_outer_nfev"
    assert sidecar is not None
    assert sidecar["S_alg"] is None
    assert sidecar["S_alg_status"] == "missing_prefix_outer_nfev"
    assert sidecar["legacy_work_proxies"] == {"S_alg_at_crossing": 7.0}


def test_first_hit_sidecar_generator_derives_s_alg_from_prefix_runtime_payload(tmp_path) -> None:
    pytest.importorskip("qiskit")
    source = tmp_path / "result.json"
    source_payload = _minimal_first_hit_source_payload()
    row = source_payload["adapt_vqe"]["history"][0]
    row.pop("S_alg_at_crossing")
    row.update(_prefix_runtime_payload()["adapt_vqe"]["history"][0])
    source.write_text(json.dumps(source_payload, sort_keys=True) + "\n", encoding="utf-8")
    wrapper = {
        "result": {
            "result_json": str(source),
            "family": "hh",
            "case_id": "hh_L2_clean_weak",
            "benchmark_id": "hh_L2_clean_weak",
            "paper_i_first_crossing": source_payload["adapt_vqe"]["paper_i_first_crossing"],
        }
    }

    sidecar, inventory = build_snake_first_hit_sidecar_for_payload(
        payload=wrapper,
        payload_path=None,
        threshold=2e-4,
    )

    assert inventory["status"] == "sidecar_generated"
    assert inventory["work_resource_status"] == "ok"
    assert sidecar is not None
    assert sidecar["S_alg"] == 13.0
    assert sidecar["S_alg_status"] == "ok"
    assert sidecar["S_alg_N_H_outer_eval"] == 1.0
    assert sidecar["S_alg_N_grad_probe"] == 6.0
    assert sidecar["S_alg_N_metric_probe"] == 2.0
    assert sidecar["S_alg_N_H_refit_eval"] == 4.0
    assert sidecar["table_i_measurement_event_ledger"]["component_totals"]["N_H_refit_eval"] == 4.0


def test_first_hit_sidecar_inventory_reports_rerun_needed_for_missing_parameterization(tmp_path) -> None:
    source = tmp_path / "result.json"
    source_payload = _minimal_first_hit_source_payload()
    del source_payload["adapt_vqe"]["parameterization"]
    source.write_text(json.dumps(source_payload, sort_keys=True) + "\n", encoding="utf-8")
    wrapper_path = tmp_path / "generic_static_single.json"
    wrapper_path.write_text(
        json.dumps({"result": {"result_json": str(source), "paper_i_first_crossing": source_payload["adapt_vqe"]["paper_i_first_crossing"]}}) + "\n",
        encoding="utf-8",
    )
    summary = {
        "row_results": [
            {
                "record_id": "static_table__hh__hh_L2_clean_weak__static_family_native_adapt_phase3",
                "family": "hh",
                "case_id": "hh_L2_clean_weak",
                "algorithm_id": "static_family_native_adapt_phase3",
                "payload_path": str(wrapper_path),
            }
        ]
    }

    inventory, output_summary = inventory_summary_payload(summary, threshold=2e-4, update_summary_rows=True)

    row = inventory["rows"][0]
    assert inventory["status_counts"] == {"rerun_needed": 1}
    assert row["rerun_needed_reason"] == "missing_parameterization_blocks"
    assert "parameterization.blocks" in row["missing_fields"]
    assert output_summary["row_results"][0]["payload_path"] == str(wrapper_path)


def test_raw_only_snake_row_keeps_raw_proxy_as_audit_only() -> None:
    row = {"measurement_shots_proxy": 50.5, "shot_cost_proxy": 999.0}

    enriched = normalize_snake_measurement_work_row(row)

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "missing_component_breakdown"
    assert enriched["S_alg"] is None
    assert enriched["S_alg_status"] == "legacy_proxy_not_event_ledger"
    assert enriched["measurement_work_proxy"] is None
    assert enriched["measurement_work_proxy_source"] is None
    assert enriched["measurement_work_proxy_status"] == "legacy_proxy_not_event_ledger"
    assert enriched["legacy_measurement_work_proxy"] is None
    assert enriched["legacy_measurement_work_proxy_source"] is None
    assert enriched["legacy_measurement_work_proxy_status"] == "unavailable:missing_component_breakdown"
    assert enriched["raw_shot_proxy_fallback_forbidden"] is True
    assert enriched["raw_shot_proxy_fallback_audit"] == {
        "value": 50.5,
        "source": "measurement_shots_proxy",
        "S_norm_status": "missing_component_breakdown",
    }
    assert enriched["measurement_work"]["legacy_raw_proxy"]["shot_cost_proxy"] == 999.0
    assert row["measurement_shots_proxy"] == 50.5


def test_explicit_components_promote_to_s_norm_and_preserve_legacy_proxy() -> None:
    row = {
        "N_H_eval": 2,
        "N_grad": 3,
        "N_metric": 5,
        "N_refit_eval": 7,
        "shot_proxy": 101,
    }

    enriched = normalize_snake_measurement_work_row(row)

    assert enriched["S_norm_status"] == "ok"
    assert enriched["S_norm"] == 17.0
    assert enriched["S_norm_N_H_outer_eval"] == 2.0
    assert enriched["S_norm_N_H_eval"] == 2.0
    assert enriched["S_norm_N_grad"] == 3.0
    assert enriched["S_norm_N_metric"] == 5.0
    assert enriched["S_norm_N_H_refit_eval"] == 7.0
    assert enriched["S_norm_N_refit_eval"] == 7.0
    assert enriched["S_grp_status"] == "missing_grouped_measurement_breakdown"
    assert enriched["S_grp_total"] is None
    assert enriched["S_alg"] is None
    assert enriched["S_alg_status"] == "legacy_proxy_not_event_ledger"
    assert enriched["measurement_work_proxy"] is None
    assert enriched["measurement_work_proxy_source"] is None
    assert enriched["measurement_work_proxy_status"] == "legacy_proxy_not_event_ledger"
    assert enriched["legacy_measurement_work_proxy"] == 17.0
    assert enriched["legacy_measurement_work_proxy_source"] == "S_norm"
    assert enriched["legacy_measurement_work_proxy_status"] == "legacy_normalized"


def test_explicit_s_alg_components_are_paper_facing_measurement_work() -> None:
    row = {
        "S_alg_N_H_outer_eval": 2,
        "S_alg_N_grad_probe": 3,
        "S_alg_N_metric_probe": 5,
        "S_alg_N_H_refit_eval": 7,
        "S_alg_N_other_quantum": 0,
        "shot_proxy": 101,
    }

    enriched = normalize_snake_measurement_work_row(row)

    assert enriched["S_alg_status"] == "ok"
    assert enriched["S_alg"] == 17.0
    assert enriched["measurement_work_proxy"] == 17.0
    assert enriched["measurement_work_proxy_source"] == "S_alg"
    assert enriched["measurement_work_proxy_status"] == "ok"
    assert enriched["S_norm"] is None
    assert enriched["legacy_measurement_work_proxy"] is None
    assert enriched["legacy_measurement_work_proxy_source"] is None
    assert enriched["legacy_measurement_work_proxy_status"] == "unavailable:missing_component_breakdown"
    assert enriched["raw_shot_proxy_fallback_forbidden"] is True
    assert enriched["raw_shot_proxy_fallback_audit"]["source"] == "shot_proxy"


def test_non_ok_event_ledger_status_does_not_promote_s_alg() -> None:
    row = {
        "table_i_measurement_event_ledger": {
            "schema": "table_i_measurement_event_ledger_v1",
            "status": "failed",
            "component_totals": {
                "N_H_outer_eval": 2,
                "N_grad_probe": 3,
                "N_metric_probe": 5,
                "N_H_refit_eval": 7,
                "N_other_quantum": 0,
            },
        },
        "measurement_shots_proxy": 99,
    }

    enriched = normalize_snake_measurement_work_row(row)

    assert enriched["S_alg"] is None
    assert enriched["S_alg_status"] == "invalid_event_ledger"
    assert enriched["measurement_work_proxy"] is None


def test_explicit_component_aliases_and_zero_promote_but_nonfinite_does_not() -> None:
    enriched = normalize_snake_measurement_work_row(
        {
            "measurement_work_N_H_eval": 0,
            "measurement_work_N_grad": 2,
            "measurement_work_N_metric": 3,
            "measurement_work_N_refit_eval": 4,
            "S_grp_H_outer": 11,
            "S_grp_grad": 13,
            "S_grp_metric": 17,
            "S_grp_H_refit": 19,
            "shot_cost_proxy": 10,
        }
    )

    assert enriched["S_norm_status"] == "ok"
    assert enriched["S_norm"] == 9.0
    assert enriched["measurement_work"]["component_sources"]["N_H_outer_eval"] == "measurement_work_N_H_eval"
    assert enriched["S_grp_status"] == "ok"
    assert enriched["S_grp_total"] == 60.0

    nonfinite = normalize_snake_measurement_work_row(
        {
            "S_norm_N_H_eval": 1,
            "S_norm_N_grad": float("inf"),
            "S_norm_N_metric": 3,
            "S_norm_N_refit_eval": 4,
            "shot_cost_proxy": 10,
        }
    )

    assert nonfinite["S_norm"] is None
    assert nonfinite["S_norm_status"] == "invalid_component_value"
    assert nonfinite["measurement_work_proxy"] is None
    assert nonfinite["legacy_measurement_work_proxy"] is None
    assert nonfinite["raw_shot_proxy_fallback_forbidden"] is True


def test_invalid_explicit_s_norm_alias_blocks_runtime_s_alg_promotion() -> None:
    enriched = normalize_snake_measurement_work_row(
        {
            "S_norm_N_H_eval": 1,
            "S_norm_N_grad": float("inf"),
            "S_norm_N_metric": 3,
            "S_norm_N_refit_eval": 4,
            "measurement_shots_proxy": 9,
        },
        source_payload=_native_runtime_payload(),
    )

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "invalid_component_value"
    assert "runtime_reconstruction" not in enriched["measurement_work"]
    assert enriched["S_alg"] is None
    assert enriched["S_alg_status"] == "legacy_proxy_not_event_ledger"


def test_invalid_explicit_s_norm_alias_blocks_embedded_runtime_s_alg_promotion() -> None:
    fresh = normalize_snake_measurement_work_row(
        {"measurement_shots_proxy": 9},
        source_payload=_native_runtime_payload(),
    )
    old_embedded_runtime_row = {
        key: value
        for key, value in fresh.items()
        if not key.startswith("S_alg")
        and key
        not in {
            "algorithmic_measurement_work",
            "measurement_work_proxy",
            "measurement_work_proxy_source",
            "measurement_work_proxy_status",
            "table_i_measurement_event_ledger",
        }
    }
    old_embedded_runtime_row["S_norm_N_grad"] = float("inf")

    enriched = normalize_snake_measurement_work_row(old_embedded_runtime_row)

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "invalid_component_value"
    assert enriched["S_alg"] is None
    assert enriched["S_alg_status"] == "legacy_proxy_not_event_ledger"
    assert "table_i_measurement_event_ledger" not in enriched


def test_partial_components_do_not_promote_partial_sum() -> None:
    enriched = normalize_snake_measurement_work_row(
        {"N_H_eval": 2, "N_grad": 3, "measurement_shots_proxy": 9}
    )

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "missing_component_breakdown"
    assert enriched["measurement_work_proxy"] is None
    assert enriched["measurement_work_proxy_source"] is None
    assert enriched["legacy_measurement_work_proxy"] is None
    assert enriched["legacy_measurement_work_proxy_source"] is None
    assert enriched["legacy_measurement_work_proxy_status"] == "unavailable:missing_component_breakdown"
    assert enriched["raw_shot_proxy_fallback_audit"]["source"] == "measurement_shots_proxy"
    assert enriched["measurement_work"]["components"] is None


def test_negative_component_is_invalid_and_not_clamped() -> None:
    enriched = normalize_snake_measurement_work_row(
        {
            "N_H_eval": 2,
            "N_grad": -3,
            "N_metric": 5,
            "N_refit_eval": 7,
            "shot_cost_proxy": 11,
        }
    )

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "invalid_component_value"
    assert enriched["measurement_work_proxy"] is None
    assert enriched["measurement_work_proxy_status"] == "legacy_proxy_not_event_ledger"
    assert enriched["legacy_measurement_work_proxy"] is None
    assert enriched["legacy_measurement_work_proxy_status"] == "unavailable:invalid_component_value"
    assert enriched["raw_shot_proxy_fallback_audit"]["source"] == "shot_cost_proxy"


def test_runtime_payload_reconstruction_promotes_s_norm_and_s_alg_without_promoting_s_grp() -> None:
    row = {"measurement_shots_proxy": 99}

    enriched = normalize_snake_measurement_work_row(
        row,
        source_payload=_native_runtime_payload(),
        source_label="unit.source",
    )

    assert enriched["S_norm_status"] == "ok"
    assert enriched["S_norm"] == 31.0
    assert enriched["S_norm_N_H_outer_eval"] == 8.0
    assert enriched["S_norm_N_grad"] == 6.0
    assert enriched["S_norm_N_metric"] == 5.0
    assert enriched["S_norm_N_H_refit_eval"] == 12.0
    assert enriched["S_alg_status"] == "ok"
    assert enriched["S_alg"] == 31.0
    assert enriched["S_alg_N_H_outer_eval"] == 8.0
    assert enriched["S_alg_N_grad_probe"] == 6.0
    assert enriched["S_alg_N_metric_probe"] == 5.0
    assert enriched["S_alg_N_H_refit_eval"] == 12.0
    assert enriched["measurement_work_proxy"] == 31.0
    assert enriched["measurement_work_proxy_source"] == "S_alg"
    assert enriched["measurement_work_proxy_status"] == "ok"
    assert enriched["legacy_measurement_work_proxy"] == 31.0
    assert enriched["legacy_measurement_work_proxy_source"] == "S_norm"
    assert enriched["measurement_work"]["runtime_reconstruction"]["status"] == "ok"
    assert enriched["measurement_work"]["runtime_reconstruction"]["source_label"] == "unit.source"
    assert enriched["table_i_measurement_event_ledger"]["schema"] == "table_i_measurement_event_ledger_v1"
    assert enriched["table_i_measurement_event_ledger"]["component_totals"] == {
        "N_H_outer_eval": 8.0,
        "N_grad_probe": 6.0,
        "N_metric_probe": 5.0,
        "N_H_refit_eval": 12.0,
        "N_other_quantum": 0.0,
    }
    assert enriched["S_grp_status"] == "missing_grouped_measurement_breakdown"
    assert enriched["S_grp_total"] is None


def test_powell_nfev_delta_changes_displayed_enriched_query_work_axis() -> None:
    baseline_payload = _native_runtime_payload()
    baseline_payload["adapt_vqe"]["adapt_inner_optimizer"] = "POWELL"
    increased_payload = json.loads(json.dumps(baseline_payload))
    delta_nfev = 17
    increased_payload["adapt_vqe"]["history"][0]["nfev_opt"] += delta_nfev
    increased_payload["adapt_vqe"]["nfev_total"] += delta_nfev

    baseline = normalize_snake_measurement_work_row(
        {"optimizer": "POWELL"},
        source_payload=baseline_payload,
        source_label="unit.powell.baseline",
    )
    increased = normalize_snake_measurement_work_row(
        {"optimizer": "POWELL"},
        source_payload=increased_payload,
        source_label="unit.powell.increased",
    )

    assert baseline["measurement_work_proxy_source"] == "S_alg"
    assert increased["measurement_work_proxy_source"] == "S_alg"
    assert increased["S_alg_N_H_outer_eval"] == baseline["S_alg_N_H_outer_eval"]
    assert increased["S_alg_N_H_refit_eval"] == baseline["S_alg_N_H_refit_eval"] + delta_nfev
    assert increased["S_alg"] == baseline["S_alg"] + delta_nfev
    assert increased["measurement_work_proxy"] == baseline["measurement_work_proxy"] + delta_nfev


def test_selected_logical_runtime_payload_infers_phase0_gradient_work() -> None:
    payload = _selected_logical_runtime_payload()

    enriched = normalize_snake_measurement_work_row(
        {"measurement_shots_proxy": 8},
        source_payload=payload,
        source_label="unit.selected",
    )
    controller_proxy = snake_controller_shot_proxy_from_payload(payload, source_label="unit.selected")

    assert enriched["S_norm_status"] == "ok"
    assert enriched["S_norm"] == 54.0
    assert enriched["S_norm_N_grad"] == 48.0
    assert enriched["S_norm_N_metric"] == 6.0
    assert enriched["S_alg_status"] == "ok"
    assert enriched["S_alg"] == 54.0
    assert enriched["S_actual"] == 54.0
    assert enriched["S_actual_status"] == "ok"
    accounting = enriched["measurement_work"]["runtime_reconstruction"]["selected_logical_phase0_accounting"]
    assert accounting["status"] == "inferred_from_selected_logical_filter"
    assert accounting["gradient_probe_count"] == 46.0
    assert enriched["table_i_measurement_event_ledger"]["component_totals"]["N_grad_probe"] == 48.0
    assert (
        enriched["table_i_measurement_event_ledger"]["common_algorithmic_component_status"]
        == "missing_common_exposure_ledger"
    )
    assert controller_proxy["status"] == "ok"
    assert controller_proxy["controller_shot_proxy"] == 54.0
    assert controller_proxy["selected_logical_phase0_accounting"]["status"] == "inferred_from_selected_logical_filter"


def test_selected_logical_explicit_phase0_is_not_double_counted() -> None:
    payload = _selected_logical_runtime_payload()
    payload["adapt_vqe"]["controller_measurement_work_summary"]["by_phase"]["phase0"] = {
        "records_with_group_keys": 46,
        **_actual_probe_fields(46, role="gradient"),
    }

    enriched = normalize_snake_measurement_work_row({"measurement_shots_proxy": 8}, source_payload=payload)
    controller_proxy = snake_controller_shot_proxy_from_payload(payload)

    assert enriched["S_norm"] == 54.0
    assert enriched["S_norm_N_grad"] == 48.0
    accounting = enriched["measurement_work"]["runtime_reconstruction"]["selected_logical_phase0_accounting"]
    assert accounting["status"] == "explicit_controller_phase0"
    assert accounting["metadata_not_added_due_to_explicit_phase0"] is True
    assert controller_proxy["controller_shot_proxy"] == 54.0


def test_selected_logical_fallback_to_full_pool_does_not_infer_phase0() -> None:
    payload = _selected_logical_runtime_payload()
    payload["adapt_vqe"]["adapt_selected_logical_filter"]["fallback_to_full_pool"] = True

    enriched = normalize_snake_measurement_work_row({"measurement_shots_proxy": 8}, source_payload=payload)
    controller_proxy = snake_controller_shot_proxy_from_payload(payload)

    assert enriched["S_norm"] == 8.0
    assert enriched["S_norm_N_grad"] == 2.0
    accounting = enriched["measurement_work"]["runtime_reconstruction"]["selected_logical_phase0_accounting"]
    assert accounting["status"] == "fallback_to_full_pool_no_inference"
    assert controller_proxy["controller_shot_proxy"] == 8.0


def test_selected_logical_invalid_phase0_metadata_fails_closed() -> None:
    payload = _selected_logical_runtime_payload()
    del payload["adapt_vqe"]["adapt_selected_logical_filter"]["pool_size_before"]

    enriched = normalize_snake_measurement_work_row({"measurement_shots_proxy": 8}, source_payload=payload)
    controller_proxy = snake_controller_shot_proxy_from_payload(payload)

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "missing_component_breakdown"
    assert enriched["measurement_work"]["runtime_reconstruction"]["status"] == "invalid_selected_logical_phase0_accounting"
    assert enriched["measurement_work_proxy"] is None
    assert controller_proxy["status"] == "invalid_selected_logical_phase0_accounting"


def test_runtime_payload_still_rejects_unknown_positive_controller_phase() -> None:
    payload = _selected_logical_runtime_payload()
    payload["adapt_vqe"]["controller_measurement_work_summary"]["by_phase"]["phaseX"] = {
        "records_with_group_keys": 1
    }

    enriched = normalize_snake_measurement_work_row({"measurement_shots_proxy": 8}, source_payload=payload)
    controller_proxy = snake_controller_shot_proxy_from_payload(payload)

    assert enriched["S_norm"] is None
    assert enriched["measurement_work"]["runtime_reconstruction"]["status"] == "unassigned_controller_phase_work"
    assert controller_proxy["status"] == "unassigned_controller_phase_work"


def test_runtime_reconstruction_ignores_partial_stale_component_aliases() -> None:
    enriched = normalize_snake_measurement_work_row(
        {"S_norm_N_H_eval": 999, "measurement_shots_proxy": 99},
        source_payload=_native_runtime_payload(),
    )

    assert enriched["S_norm_status"] == "ok"
    assert enriched["S_norm_N_H_outer_eval"] == 8.0
    assert enriched["S_norm"] == 31.0
    assert enriched["measurement_work"]["component_sources"]["N_H_outer_eval"] == "N_H_outer_eval"
    assert enriched["S_alg_status"] == "ok"
    assert enriched["S_alg_N_H_outer_eval"] == 8.0


def test_runtime_payload_can_upgrade_previously_enriched_raw_row() -> None:
    raw_enriched = normalize_snake_measurement_work_row({"measurement_shots_proxy": 99})
    assert raw_enriched["S_norm_status"] == "missing_component_breakdown"

    upgraded = normalize_snake_measurement_work_row(
        raw_enriched,
        source_payload=_native_runtime_payload(),
    )

    assert upgraded["S_norm_status"] == "ok"
    assert upgraded["S_norm"] == 31.0
    assert upgraded["S_alg_status"] == "ok"
    assert upgraded["S_alg"] == 31.0
    assert upgraded["measurement_work_proxy_source"] == "S_alg"
    assert upgraded["legacy_measurement_work_proxy_source"] == "S_norm"


def test_runtime_payload_does_not_override_invalid_explicit_components() -> None:
    enriched = normalize_snake_measurement_work_row(
        {"N_H_eval": 1, "N_grad": -2, "N_metric": 0, "N_refit_eval": 3, "measurement_shots_proxy": 9},
        source_payload=_native_runtime_payload(),
    )

    assert enriched["S_norm"] is None
    assert enriched["S_norm_status"] == "invalid_component_value"
    assert "runtime_reconstruction" not in enriched["measurement_work"]
    assert enriched["measurement_work_proxy"] is None
    assert enriched["legacy_measurement_work_proxy"] is None


def test_ok_s_norm_row_recomputes_when_explicit_s_grp_components_are_added() -> None:
    enriched = normalize_snake_measurement_work_row(
        {"N_H_eval": 1, "N_grad": 2, "N_metric": 3, "N_refit_eval": 4}
    )
    assert enriched["S_norm_status"] == "ok"
    assert enriched["S_grp_status"] == "missing_grouped_measurement_breakdown"

    with_grouped = dict(enriched)
    with_grouped.update({"S_grp_H_outer": 10, "S_grp_grad": 20, "S_grp_metric": 30, "S_grp_H_refit": 40})
    recomputed = normalize_snake_measurement_work_row(with_grouped)

    assert recomputed["S_norm_status"] == "ok"
    assert recomputed["S_norm"] == 10.0
    assert recomputed["S_grp_status"] == "ok"
    assert recomputed["S_grp_total"] == 100.0


def test_stale_enriched_row_recomputes_when_explicit_components_are_added() -> None:
    raw_enriched = normalize_snake_measurement_work_row({"measurement_shots_proxy": 99})
    stale_plus_explicit = dict(raw_enriched)
    stale_plus_explicit.update({"N_H_eval": 1, "N_grad": 2, "N_metric": 3, "N_refit_eval": 4})

    recomputed = normalize_snake_measurement_work_row(stale_plus_explicit)

    assert recomputed["S_norm_status"] == "ok"
    assert recomputed["S_norm"] == 10.0
    assert recomputed["measurement_work_proxy_source"] is None
    assert recomputed["legacy_measurement_work_proxy_source"] == "S_norm"


def test_runtime_payload_rejects_legacy_or_inconsistent_sources() -> None:
    legacy = _native_runtime_payload()
    legacy["adapt_vqe"]["controller_measurement_work_summary"]["legacy_fallback_used"] = True

    legacy_enriched = normalize_snake_measurement_work_row(
        {"measurement_shots_proxy": 99},
        source_payload=legacy,
    )

    assert legacy_enriched["S_norm"] is None
    assert legacy_enriched["measurement_work_proxy"] is None
    assert legacy_enriched["legacy_measurement_work_proxy"] is None
    assert legacy_enriched["measurement_work"]["runtime_reconstruction"]["status"] == "legacy_controller_summary_not_promotable"

    inconsistent = _native_runtime_payload()
    inconsistent["adapt_vqe"]["nfev_total"] = 2

    inconsistent_enriched = normalize_snake_measurement_work_row(
        {"measurement_shots_proxy": 88},
        source_payload=inconsistent,
    )

    assert inconsistent_enriched["S_norm"] is None
    assert inconsistent_enriched["measurement_work_proxy"] is None
    assert inconsistent_enriched["legacy_measurement_work_proxy"] is None
    assert inconsistent_enriched["measurement_work"]["runtime_reconstruction"]["status"] == "inconsistent_nfev_partition"


def test_source_payload_map_enriches_targeted_support_path_only() -> None:
    payload = {
        "bosonic_snake_current_table_support": {
            "measurement_shots_proxy": 3,
            "per_benchmark": {
                "bose_hubbard_L2": {"measurement_shots_proxy": 4},
                "harmonic_kerr_L2": {"measurement_shots_proxy": 5},
            },
        },
    }

    enriched = enrich_snake_support_payload(
        payload,
        source_payloads={
            "bosonic_snake_current_table_support": _native_runtime_payload(),
            "bosonic_snake_current_table_support.per_benchmark.bose_hubbard_L2": _native_runtime_payload(),
        },
    )

    assert enriched["bosonic_snake_current_table_support"]["S_alg_status"] == "legacy_proxy_not_event_ledger"
    assert enriched["bosonic_snake_current_table_support"]["S_alg"] is None
    per = enriched["bosonic_snake_current_table_support"]["per_benchmark"]
    assert per["bose_hubbard_L2"]["S_norm_status"] == "ok"
    assert per["bose_hubbard_L2"]["S_norm"] == 31.0
    assert per["bose_hubbard_L2"]["S_alg_status"] == "ok"
    assert per["bose_hubbard_L2"]["S_alg"] == 31.0
    assert per["harmonic_kerr_L2"]["S_norm_status"] == "missing_component_breakdown"
    summary = enriched["snake_measurement_work_normalization"]
    assert summary["s_norm_available_count"] == 1
    assert summary["s_alg_available_count"] == 1
    assert summary["raw_fallback_forbidden_count"] == 2
    assert summary["source_payload_count"] == 2
    assert summary["source_payload_consumed_count"] == 2
    assert summary["runtime_reconstruction_status_counts"] == {"ok": 1}


def test_class_aggregate_s_alg_is_mean_of_child_event_ledgers_only_when_complete() -> None:
    payload = {
        "bosonic_snake_current_table_support": {
            "measurement_shots_proxy": 3,
            "per_benchmark": {
                "a": {"measurement_shots_proxy": 4},
                "b": {"measurement_shots_proxy": 5},
            },
        },
    }

    enriched = enrich_snake_support_payload(
        payload,
        source_payloads={
            "bosonic_snake_current_table_support.per_benchmark.a": _native_runtime_payload(),
            "bosonic_snake_current_table_support.per_benchmark.b": _native_runtime_payload(),
        },
    )

    aggregate = enriched["bosonic_snake_current_table_support"]
    assert aggregate["S_alg_status"] == "ok"
    assert aggregate["S_alg"] == 31.0
    assert aggregate["algorithmic_measurement_work"]["source_kind"] == "event_ledger"
    assert aggregate["table_i_measurement_event_ledger"]["source_kind"] == "aggregate_mean_of_child_event_ledgers_v1"
    assert aggregate["table_i_measurement_event_ledger"]["child_row_count"] == 2
    summary = enriched["snake_measurement_work_normalization"]
    assert summary["s_alg_available_count"] == 3


def test_source_payload_map_accepts_map_relative_and_cwd_relative_paths(tmp_path, monkeypatch) -> None:
    map_dir = tmp_path / "paper_facing"
    map_dir.mkdir()
    map_relative_payload = map_dir / "local-result.json"
    cwd_payload = tmp_path / "repo-root-result.json"
    map_relative_payload.write_text('{"kind": "map-relative"}', encoding="utf-8")
    cwd_payload.write_text('{"kind": "cwd-relative"}', encoding="utf-8")
    source_map = map_dir / "source-map.json"
    source_map.write_text(
        (
            '{\n'
            '  "schema": "snake_table_i_source_payload_map_v1",\n'
            '  "sources": [\n'
            '    {"support_path": "a", "result_json": "local-result.json"},\n'
            '    {"support_path": "b", "result_json": "repo-root-result.json"}\n'
            '  ]\n'
            '}\n'
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    loaded = _load_source_payload_map(source_map)

    assert loaded["a"] == {"kind": "map-relative"}
    assert loaded["b"] == {"kind": "cwd-relative"}


def test_support_payload_traversal_is_explicit_and_idempotent() -> None:
    payload = {
        "all_averaged_snake_current_table_support": {"shot_cost_proxy": 1},
        "fermionic_snake_current_table_support": {"measurement_shots_proxy": 2},
        "bosonic_snake_current_table_support": {
            "measurement_shots_proxy": 3,
            "per_benchmark": {
                "bose_hubbard_L2": {"measurement_shots_proxy": 4},
                "untouched_non_row": "not-a-row",
            },
        },
        "fermion_boson_snake_current_table_support": {
            "aggregate": {"shot_cost_proxy": 5},
            "inputs": [
                {"shot_cost_proxy": 6},
                {"N_H_eval": 1, "N_grad": 2, "N_metric": 3, "N_refit_eval": 4, "shot_proxy": 7},
            ],
        },
        "route_a_definition_20260513": {"shot_cost_proxy": 999},
    }

    enriched = enrich_snake_support_payload(payload)
    enriched_twice = enrich_snake_support_payload(enriched)

    summary = enriched["snake_measurement_work_normalization"]
    assert summary["enriched_path_count"] == 7
    assert summary["s_norm_available_count"] == 1
    assert summary["s_grp_available_count"] == 0
    assert summary["raw_fallback_forbidden_count"] == 6
    assert summary["status_counts"] == {"missing_component_breakdown": 6, "ok": 1}
    assert summary["s_grp_status_counts"] == {"missing_grouped_measurement_breakdown": 7}
    assert "measurement_work" not in enriched["route_a_definition_20260513"]
    assert enriched_twice == enriched
