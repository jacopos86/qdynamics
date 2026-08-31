#!/usr/bin/env python3
"""Build and validate the Paper-II Table-I SNAKE recovery manifest.

This is a non-run recovery contract: it reads the fixed v2 seed-track case
manifest and seed ledger, selects only SNAKE static-seed cases, freezes the
current 2026-05-19 Table-I inputs as diagnostic-only, and writes a compact
machine-readable manifest for later parity, calibration, family-repair, and
rerun planning.  It does not submit CHTC jobs or mutate fetched/raw outputs.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.time_dynamics.tables.table_lock_contract import validate_class_settings_lock_manifest
INPUT = Path("chtc/generic_time_dynamics_table/input")
SOURCE_CASE_MANIFEST = INPUT / "paper_ii_seed_tracks_cases_v2.json"
SOURCE_SEED_LEDGER = INPUT / "paper_ii_seed_tracks_seed_ledger_v2.json"
RECOVERY_MANIFEST = INPUT / "paper_ii_table_i_snake_recovery_manifest_v1.json"
CLASS_SETTINGS_LOCK = INPUT / "class_settings" / "paper_ii_class_settings_lock_v1.json"
ALL_ALGORITHM_CANDIDATE_LOCK = (
    INPUT / "class_settings" / "paper_ii_all_algorithm_class_settings_candidate_lock_v1.json"
)
CLASS_CALIBRATION_INPUT_MANIFEST = (
    Path("chtc/time_dynamics_optuna/input")
    / "paper_ii_all_algorithm_class_calibration_v1_manifest.json"
)

SCHEMA = "paper_ii_table_i_snake_recovery_manifest_v1"
CREATED_AT = "2026-05-19T00:00:00+00:00"
EXPECTED_SNAKE_CASE_COUNT = 20
EXPECTED_FAMILIES: tuple[str, ...] = (
    "hubbard",
    "ionic_hubbard",
    "extended_hubbard",
    "ttprime_hubbard",
    "spinless_tv",
    "bose_hubbard",
    "harmonic_kerr_chain",
    "spin_boson",
    "hh",
    "molecular_vibronic_h2",
)
EXPECTED_DRIVE_AMPLITUDES: tuple[float, ...] = (0.2, 0.6)
FORBIDDEN_SEED_SOURCE_TOKENS: tuple[str, ...] = ("staged", "fallback", "pending", "recovery")

TABLE_I_ALGORITHMS: tuple[dict[str, str], ...] = (
    {
        "algorithm_id": "dyn_controller_full",
        "settings_kind": "controller",
        "table_role": "primary_checkpoint_mclachlan_method",
    },
    {
        "algorithm_id": "dyn_fixed_mclachlan",
        "settings_kind": "mclachlan",
        "table_role": "fixed_mclachlan_comparator",
    },
    {
        "algorithm_id": "dyn_product_formula_envelope",
        "settings_kind": "comparator",
        "table_role": "product_formula_comparator",
    },
    {
        "algorithm_id": "dyn_qdrift",
        "settings_kind": "comparator",
        "table_role": "qdrift_comparator",
    },
    {
        "algorithm_id": "dyn_fixed_pvqd",
        "settings_kind": "comparator",
        "table_role": "fixed_pvqd_comparator",
    },
    {
        "algorithm_id": "dyn_adaptive_pvqd",
        "settings_kind": "comparator",
        "table_role": "adaptive_pvqd_comparator",
    },
    {
        "algorithm_id": "dyn_avqds",
        "settings_kind": "comparator",
        "table_role": "avqds_comparator",
    },
    {
        "algorithm_id": "dyn_avqds_t",
        "settings_kind": "comparator",
        "table_role": "avqds_t_comparator",
    },
)

PARITY_CORRECTNESS_MATRIX: tuple[dict[str, str], ...] = (
    {
        "algorithm_id": "dyn_controller_full",
        "required_check_type": "strict_decision_contract_and_nonfrozen_trajectory",
        "sidecar_name": "controller_row_telemetry_no_separate_sidecar",
        "timing": "post_run",
        "enforcement_point": "controller_artifact_validation_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "additive_correctness_provenance_not_physical_error",
        "support_scope": "strict_exact_free_decision_contract_and_nonfrozen_diagnostic_trajectory",
    },
    {
        "algorithm_id": "dyn_fixed_mclachlan",
        "required_check_type": "qiskit_fixed_scaffold_state_observable_parity",
        "sidecar_name": "qiskit_parity.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "qiskit_parity_delta_is_not_physical_accuracy",
        "support_scope": "fixed_scaffold_state_and_observable_parity_only",
    },
    {
        "algorithm_id": "dyn_fixed_mclachlan",
        "required_check_type": "mclachlan_metric_rhs_solve_integrator_correctness",
        "sidecar_name": "mclachlan_correctness.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "additive_correctness_provenance_not_physical_error",
        "support_scope": "metric_rhs_regularized_solve_integrator_a0_and_nonfrozen_invariants",
    },
    {
        "algorithm_id": "dyn_product_formula_envelope",
        "required_check_type": "qiskit_product_formula_sequence_parity",
        "sidecar_name": "qiskit_parity.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "qiskit_parity_delta_is_not_physical_accuracy",
        "support_scope": "selected_product_formula_order_and_resource_metadata",
    },
    {
        "algorithm_id": "dyn_qdrift",
        "required_check_type": "qiskit_qdrift_realized_sample_plan_parity",
        "sidecar_name": "qiskit_parity.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "qiskit_parity_delta_is_not_physical_accuracy",
        "support_scope": "realized_sample_plan_and_deterministic_rng_provenance",
    },
    {
        "algorithm_id": "dyn_fixed_pvqd",
        "required_check_type": "qiskit_pvqd_component_parity",
        "sidecar_name": "qiskit_parity.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "qiskit_parity_delta_is_not_physical_accuracy",
        "support_scope": "fixed_pvqd_scaffold_target_projection_component_parity",
    },
    {
        "algorithm_id": "dyn_adaptive_pvqd",
        "required_check_type": "qiskit_adaptive_pvqd_component_parity",
        "sidecar_name": "qiskit_parity.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "qiskit_parity_delta_is_not_physical_accuracy",
        "support_scope": "adaptive_pvqd_component_parity_plus_append_candidate_telemetry",
    },
    {
        "algorithm_id": "dyn_avqds",
        "required_check_type": "dense_rhs_tangent_and_invariant_correctness",
        "sidecar_name": "avqds_correctness.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "additive_correctness_provenance_not_physical_error",
        "support_scope": "rhs_tangent_solve_grouped_exact_exclusion_append_and_norm_invariants",
    },
    {
        "algorithm_id": "dyn_avqds_t",
        "required_check_type": "dense_target_tangent_and_invariant_correctness",
        "sidecar_name": "avqds_t_correctness.json",
        "timing": "post_run",
        "enforcement_point": "row_normalization_and_table_audit",
        "required_status": "passed",
        "physical_error_policy": "additive_correctness_provenance_not_physical_error",
        "support_scope": "product_formula_target_tangent_append_ranking_and_norm_invariants",
    },
)

CLASS_CALIBRATION_SEARCH_PROFILES: tuple[dict[str, Any], ...] = (
    {
        "algorithm_id": "dyn_controller_full",
        "settings_kind": "controller",
        "search_profile_id": "checkpoint_controller_current_lock_seed_v1",
        "class_level_hooks": [
            "checkpoint_controller_mode",
            "checkpoint_controller_exact_input_mode",
            "integrator_policy",
            "append_enabled",
            "prune_mode",
        ],
    },
    {
        "algorithm_id": "dyn_fixed_mclachlan",
        "settings_kind": "mclachlan",
        "search_profile_id": "fixed_mclachlan_integrator_policy_v1",
        "class_level_hooks": ["fixed_mclachlan_integrator_policy", "regularization_lambda", "pinv_rcond"],
    },
    {
        "algorithm_id": "dyn_product_formula_envelope",
        "settings_kind": "comparator",
        "search_profile_id": "product_formula_order_v1",
        "class_level_hooks": ["product_formula_order"],
    },
    {
        "algorithm_id": "dyn_qdrift",
        "settings_kind": "comparator",
        "search_profile_id": "qdrift_samples_rng_v1",
        "class_level_hooks": ["qdrift_samples_per_interval", "qdrift_rng_seed"],
    },
    {
        "algorithm_id": "dyn_fixed_pvqd",
        "settings_kind": "comparator",
        "search_profile_id": "fixed_pvqd_projection_v1",
        "class_level_hooks": ["pvqd_optimizer_maxiter", "pvqd_overlap_tol", "pvqd_target_product_formula_order"],
    },
    {
        "algorithm_id": "dyn_adaptive_pvqd",
        "settings_kind": "comparator",
        "search_profile_id": "adaptive_pvqd_append_projection_v1",
        "class_level_hooks": [
            "adaptive_pvqd_optimizer_maxiter",
            "adaptive_pvqd_overlap_tol",
            "adaptive_pvqd_target_product_formula_order",
            "adaptive_pvqd_append_loss_threshold",
            "adaptive_pvqd_append_min_loss_improvement",
            "adaptive_pvqd_append_candidate_limit",
        ],
    },
    {
        "algorithm_id": "dyn_avqds",
        "settings_kind": "comparator",
        "search_profile_id": "avqds_rhs_tangent_v1",
        "class_level_hooks": [
            "avqds_regularization_lambda",
            "avqds_pinv_relative_cutoff",
            "avqds_append_rhs_residual_ratio_threshold",
            "avqds_append_min_residual_ratio_gain",
            "avqds_append_candidate_limit",
        ],
    },
    {
        "algorithm_id": "dyn_avqds_t",
        "settings_kind": "comparator",
        "search_profile_id": "avqds_t_target_tangent_v1",
        "class_level_hooks": [
            "avqds_t_regularization_lambda",
            "avqds_t_pinv_relative_cutoff",
            "avqds_t_target_product_formula_order",
            "avqds_t_append_target_tangent_residual_ratio_threshold",
            "avqds_t_append_min_residual_ratio_gain",
            "avqds_t_append_candidate_limit",
        ],
    },
)

CALIBRATION_SCORING_SELECTION_RULES: dict[str, Any] = {
    "schema": "paper_ii_class_calibration_scoring_selection_rules_v1",
    "strict_online_controller_feedback": "exact_free_measurement_compatible_observables_only",
    "comparator_runtime_decisions": "exact_free_prepared_state_or_product_formula_target_observables_only",
    "offline_class_calibration_review": "exact_diagnostic_physical_errors_allowed_for_reporting_only",
    "discard_before_scoring": [
        "failed_same_seed_validation",
        "missing_or_failed_required_parity_correctness",
        "exact_decision_leakage",
        "latest_phase3_source_artifact_missing_locally",
        "staged_fallback_pending_or_recovery_seed_source",
        "pathological_incomplete_or_frozen_trajectory",
    ],
    "primary_rank": "median_physical_error_across_all_snake_cases_and_drives_in_tuning_class",
    "secondary_objective": "resource_cost_pareto_front",
    "tie_breakers": ["lower_resource_cost", "simpler_settings", "deterministic_record_order"],
    "case_specific_winner_policy": "forbidden_unless_selected_as_class_level_setting_for_entire_tuning_class",
    "promotion_policy": "explicit_user_approval_and_separate_promotion_step_required",
}

RECOVERY_VALIDATION_GATES: tuple[dict[str, Any], ...] = (
    {"gate_id": "snake_only_recovery_cases", "severity": "ERROR"},
    {"gate_id": "same_seed_hash_all_methods_per_group", "severity": "ERROR"},
    {"gate_id": "no_staged_fallback_pending_recovery_seed_source", "severity": "ERROR"},
    {"gate_id": "latest_phase3_source_artifact_present_locally", "severity": "ERROR"},
    {"gate_id": "class_level_settings_for_checkpoint_fixed_and_all_comparators", "severity": "ERROR"},
    {"gate_id": "required_parity_correctness_sidecars_passing", "severity": "ERROR"},
    {"gate_id": "strict_exact_free_controller_decision_contract", "severity": "ERROR"},
    {"gate_id": "exact_reference_fields_diagnostic_report_only", "severity": "ERROR"},
    {"gate_id": "full_horizon_or_accepted_nonpathological_completion", "severity": "ERROR"},
    {"gate_id": "non_frozen_mclachlan_controller_when_exact_diagnostics_move", "severity": "ERROR"},
    {"gate_id": "missing_epsilon_spec_shots_or_fidelity_marked_missing_evidence", "severity": "WARN"},
)

FAMILY_REPAIR_ORDER: tuple[str, ...] = (
    "observable_and_primary_density_semantics",
    "drive_and_operator_semantics",
    "frozen_no_dynamics_telemetry",
    "frozen_no_dynamics_tranche",
    "wrong_observable_tranche",
    "hh_damping_undershoot_tranche",
)

FINAL_CLOSEOUT_REGRESSION_REQUIREMENTS: tuple[str, ...] = (
    "snake_only_recovery_manifest_selection",
    "controller_mclachlan_comparator_class_settings_lock_validation",
    "comparator_settings_consumption",
    "qiskit_parity_required_behavior",
    "avqds_and_avqds_t_correctness_sidecars",
    "fixed_mclachlan_algorithmic_correctness",
    "same_seed_validation",
    "exact_reference_decision_boundary_violations",
    "missing_evidence_reporting",
)

DIAGNOSTIC_BASELINE_SOURCES: tuple[dict[str, str], ...] = (
    {
        "source_id": "visible_manuscript_table_i_cells_20260519",
        "path": "MATH/paper_details/time_dynamics_paper_II.tex",
        "status": "diagnostic_frozen_not_paper_facing",
        "mutation_policy": "do_not_edit_in_recovery_manifest_wave",
    },
    {
        "source_id": "paper_ii_tentative_table_appendix_20260519",
        "path": "MATH/paper_facing/paper_II_dynamics/tentative_table_artifact_appendix.md",
        "status": "diagnostic_frozen_not_paper_facing",
        "mutation_policy": "do_not_edit_in_recovery_manifest_wave",
    },
    {
        "source_id": "seedtracks_v2_retained_snake_comparator_rows",
        "path": "raw_outputs/chtc_seedtracks_v2_complete_20260518/",
        "status": "diagnostic_baseline_input_only",
        "mutation_policy": "do_not_overwrite_or_reaggregate",
    },
    {
        "source_id": "repaired_checkpoint_diagnostic_best_rows",
        "path": "raw_outputs/chtc_paper_ii_repaired_checkpoint_fullsearch_snake_v1_20260519/",
        "status": "diagnostic_baseline_input_only",
        "mutation_policy": "do_not_overwrite_or_reaggregate",
    },
    {
        "source_id": "checkpoint_cost_recovery_rows",
        "path": "raw_outputs/table_i_checkpoint_cost_recovery_20260519/",
        "status": "diagnostic_baseline_input_only",
        "mutation_policy": "do_not_overwrite_or_reaggregate",
    },
    {
        "source_id": "fixed_mclachlan_rk4_refresh_rows",
        "path": "raw_outputs/chtc_fixed_rk4_v1_20260519/aggregate_summary.*",
        "status": "diagnostic_baseline_input_only",
        "mutation_policy": "do_not_overwrite_or_reaggregate",
    },
)


class RecoveryManifestValidationError(ValueError):
    """Raised when the SNAKE recovery manifest fails a blocking gate."""


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _read_json(path: str | Path) -> Any:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = _repo_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(_repo_path(path).read_bytes()).hexdigest()


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RecoveryManifestValidationError(f"{label} must be a JSON object")
    return dict(value)


def _source_text_for_seed_lock(case: Mapping[str, Any], seed_lock: Mapping[str, Any]) -> str:
    keys = (
        "artifact_json",
        "normalized_seed_artifact_json",
        "seed_selection_policy",
        "selected_static_seed_source",
        "source_artifact_json",
    )
    values = [str(case.get("artifact_json", ""))]
    values.extend(str(seed_lock.get(key, "")) for key in keys)
    return "\n".join(values).lower()


def _seed_source_has_forbidden_token(case: Mapping[str, Any], seed_lock: Mapping[str, Any]) -> str | None:
    text = _source_text_for_seed_lock(case, seed_lock)
    for token in FORBIDDEN_SEED_SOURCE_TOKENS:
        if token in text:
            return token
    return None


def _case_drive(case: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    drive = metadata.get("drive", {})
    if not isinstance(drive, Mapping):
        raise RecoveryManifestValidationError(f"case {case.get('case_id')}: metadata.drive must be an object")
    return dict(drive)


def _selected_case_record(case: Mapping[str, Any]) -> dict[str, Any]:
    metadata = _as_mapping(case.get("metadata", {}), label=f"case {case.get('case_id')} metadata")
    seed_lock = _as_mapping(metadata.get("seed_lock", {}), label=f"case {case.get('case_id')} seed_lock")
    drive = _case_drive(case, metadata)
    return {
        "case_id": str(case.get("case_id", "")),
        "family": str(case.get("family", "")),
        "tuning_class": str(case.get("tuning_class", "")),
        "table_class": str(case.get("table_class", "")),
        "artifact_json": str(case.get("artifact_json", "")),
        "seed_track": str(seed_lock.get("seed_track", "")),
        "seed_selection_policy": str(seed_lock.get("seed_selection_policy", "")),
        "seed_artifact_sha256": str(seed_lock.get("seed_artifact_sha256", "")),
        "source_artifact_sha256": str(seed_lock.get("source_artifact_sha256", "")),
        "source_artifact_json": str(seed_lock.get("source_artifact_json", "")),
        "normalized_seed_artifact_json": str(seed_lock.get("normalized_seed_artifact_json", case.get("artifact_json", ""))),
        "same_seed_comparator_group_id": str(seed_lock.get("same_seed_comparator_group_id", "")),
        "latest_phase3_source_artifact_missing_locally": bool(
            metadata.get("latest_phase3_source_artifact_missing_locally", False)
        ),
        "drive_metadata": drive,
        "t_final": float(case.get("t_final")),
        "num_times": int(case.get("num_times")),
        "observable_policy": {
            "policy_id": "paper_ii_seed_tracks_v2_family_runtime_observables",
            "status": "preserve_from_v2_case_and_runtime_family_adapter",
            "qpu_faithful_controller_data_contract": str(
                metadata.get("qpu_faithful_controller_data_contract", "")
            ),
            "time_dependence": str(metadata.get("time_dependence", "")),
        },
        "diagnostic_exact_reference_policy": str(metadata.get("diagnostic_exact_reference_mode", "")),
        "compile_target": {
            "status": "preserve_existing_generic_time_dynamics_table_runtime_target",
            "submit_wrapper": "chtc/generic_time_dynamics_table/run_task_apptainer.sh",
            "submit_requirements": "TARGET.HasSIF",
            "class_settings_manifest": str(CLASS_SETTINGS_LOCK),
            "qiskit_parity_mode_policy": "run_task.sh defaults supported comparator algorithms to parity_required",
        },
    }


def _excluded_case_record(case: Mapping[str, Any], seed_track: str) -> dict[str, Any]:
    return {
        "case_id": str(case.get("case_id", "")),
        "family": str(case.get("family", "")),
        "seed_track": seed_track,
        "reason": "non_snake_seed_track_out_of_scope_for_recovery",
    }


def _case_records_from_source(source_cases_manifest: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = _as_mapping(_read_json(source_cases_manifest), label="source case manifest")
    cases = source.get("cases", [])
    if not isinstance(cases, list):
        raise RecoveryManifestValidationError("source case manifest cases must be a list")

    selected_cases: list[dict[str, Any]] = []
    excluded_cases: list[dict[str, Any]] = []
    for case_raw in cases:
        case = _as_mapping(case_raw, label="source case")
        metadata = _as_mapping(case.get("metadata", {}), label=f"case {case.get('case_id')} metadata")
        seed_lock = _as_mapping(metadata.get("seed_lock", {}), label=f"case {case.get('case_id')} seed_lock")
        seed_track = str(seed_lock.get("seed_track", "")).strip().lower()
        if seed_track == "snake":
            selected_cases.append(_selected_case_record(case))
        else:
            excluded_cases.append(_excluded_case_record(case, seed_track or "missing"))
    return selected_cases, excluded_cases


def _expected_algorithms() -> list[dict[str, str]]:
    return [
        {
            **dict(algorithm),
            "same_seed_scope": "all_selected_snake_cases",
            "parity_correctness_status": "required_matrix_specified_sidecar_enforced_work_items_4_6",
            "class_calibration_status": "planned_required_not_validated_work_items_7_11",
        }
        for algorithm in TABLE_I_ALGORITHMS
    ]


def _parity_correctness_matrix() -> list[dict[str, str]]:
    return [dict(item) for item in PARITY_CORRECTNESS_MATRIX]


def _family_repair_rerun_eligibility_gate() -> dict[str, Any]:
    return {
        "schema": "paper_ii_family_repair_rerun_eligibility_gate_v1",
        "status": "blocked_until_recovery_parity_calibration_gates_pass_and_user_approves",
        "family_repair_can_resume": False,
        "rerun_eligibility": "blocked",
        "prior_family_bugfix_plan": "docs/plans/paper-ii-time-dynamics-family-bugfix-2026-05-19.md",
        "prior_family_bugfix_plan_role": "later_stage_dependency_not_immediate_next_step",
        "dependencies_before_resume": [
            "snake_recovery_manifest_validation_passed",
            "all_comparator_parity_correctness_contracts_implemented_and_passing_targeted_tests_or_smokes",
            "class_calibration_contract_exists_for_checkpoint_fixed_mclachlan_and_every_comparator",
            "candidate_class_settings_reviewed_without_accidental_promotion",
            "explicit_user_approval_to_resume_family_repair",
        ],
        "repair_order": list(FAMILY_REPAIR_ORDER),
        "tranche_review_policy": {
            "local_diagnostic_overlay_pdf_required_per_tranche": True,
            "user_visual_approval_required_before_any_rerun_escalation": True,
            "chtc_submission_allowed_by_this_gate": False,
        },
        "rerun_preservation_policy": {
            "preserve_fixed_snake_seed_artifacts_and_hashes": True,
            "preserve_drive_time_grid_observable_policy_diagnostic_reference_compile_target": True,
            "preserve_same_seed_comparator_groups": True,
            "rerun_checkpoint_fixed_and_every_comparator_for_affected_same_seed_group": True,
        },
        "full_table_rerun_policy": "blocked_until_narrow_repaired_calibrated_reruns_and_audits_pass_with_user_approval",
    }


def _final_aggregation_promotion_closeout_gate() -> dict[str, Any]:
    return {
        "schema": "paper_ii_table_i_final_aggregation_promotion_closeout_gate_v1",
        "status": "blocked_pending_user_approved_repaired_reruns_post_fetch_audit_and_separate_promotion_workflow",
        "final_aggregation_allowed": False,
        "table_promotion_allowed": False,
        "post_fetch_audit_no_blocking_errors_required": True,
        "explicit_user_approval_required_for_final_aggregation": True,
        "separate_user_approved_workflow_required_for_table_promotion": True,
        "required_before_final_aggregation": [
            "all_required_parity_correctness_sidecars_passing",
            "approved_class_level_settings_for_all_methods",
            "same_seed_validation_passed",
            "family_repair_tranche_pdfs_user_visual_approved",
            "post_fetch_audit_no_blocking_errors",
            "missing_evidence_marked_unavailable_not_invented",
        ],
        "aggregate_separation_policy": {
            "physical_errors": "exact diagnostic/reference error fields",
            "qiskit_parity_deltas": "implementation parity only and never physical accuracy",
            "dense_component_correctness": "native correctness sidecar diagnostics",
            "resource_cost_fields": "compiled/resources/shot proxy fields",
            "missing_evidence": "epsilon_spec, shots_total, and fidelity are explicit unavailable values when absent",
        },
        "regression_coverage_required": list(FINAL_CLOSEOUT_REGRESSION_REQUIREMENTS),
        "closeout_records_required": [
            "frozen_diagnostic_baseline_artifacts",
            "recovery_manifest_path_and_hash",
            "parity_correctness_status_by_algorithm",
            "candidate_class_calibration_reports",
            "approved_or_rejected_family_repair_tranches",
            "remaining_blockers_before_table_i_promotion",
        ],
        "forbidden_without_separate_explicit_workflow": [
            "manuscript_tex_edits",
            "visible_table_i_edits",
            "settings_promotion",
            "table_value_promotion",
        ],
    }


def build_recovery_manifest(
    *,
    source_cases_manifest: str | Path = SOURCE_CASE_MANIFEST,
    source_seed_ledger: str | Path = SOURCE_SEED_LEDGER,
    created_at: str = CREATED_AT,
) -> dict[str, Any]:
    """Build the deterministic SNAKE-only recovery manifest payload."""

    selected_cases, excluded_cases = _case_records_from_source(source_cases_manifest)
    algorithms = _expected_algorithms()

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "manifest_id": SCHEMA,
        "created_at": created_at,
        "manifest_role": "diagnostic_freeze_and_snake_recovery_contract_only_not_paper_evidence",
        "source_cases_manifest": str(source_cases_manifest),
        "source_seed_ledger": str(source_seed_ledger),
        "source_hashes": {
            "source_cases_manifest_sha256": _sha256_file(source_cases_manifest),
            "source_seed_ledger_sha256": _sha256_file(source_seed_ledger),
        },
        "selection_policy": {
            "selected_seed_track": "snake",
            "expected_case_count": EXPECTED_SNAKE_CASE_COUNT,
            "expected_families": list(EXPECTED_FAMILIES),
            "expected_drive_amplitudes": list(EXPECTED_DRIVE_AMPLITUDES),
            "excluded_seed_tracks": ["posgeo"],
            "posgeo_scope": "historical_diagnostic_out_of_scope_for_this_recovery",
            "seed_regeneration_policy": "forbidden_preserve_v2_seed_artifact_hashes_exactly",
            "seed_substitution_policy": "forbidden_no_staged_fallback_pending_or_recovery_seed_sources",
        },
        "selected_case_count": len(selected_cases),
        "excluded_case_count": len(excluded_cases),
        "selected_cases": selected_cases,
        "excluded_cases": excluded_cases,
        "algorithms": algorithms,
        "parity_correctness_requirements": {
            "schema": "paper_ii_table_i_parity_correctness_matrix_v1",
            "status": "specified_work_items_4_6_sidecar_contracts_implemented_no_recovery_rerun_evidence_yet",
            "policy": "every Table-I algorithm must have a passing required parity or correctness check before paper-facing use",
            "matrix": _parity_correctness_matrix(),
            "required_algorithm_ids": [algorithm["algorithm_id"] for algorithm in TABLE_I_ALGORITHMS],
            "qiskit_parity_sidecar_name": "qiskit_parity.json",
            "native_correctness_sidecars": [
                "mclachlan_correctness.json",
                "avqds_correctness.json",
                "avqds_t_correctness.json",
            ],
            "controller_check_sidecar_policy": "controller strict-decision and non-frozen checks are enforced from row/artifact telemetry rather than Qiskit-as-primary dynamics",
            "missing_failed_or_not_applicable_required_check_blocks_table_i_use": True,
            "missing_or_failed_required_check_blocks_table_i_use": True,
            "physical_error_policy": "parity and correctness outcomes are additive provenance fields; physical error columns remain exact-diagnostic reporting fields",
        },
        "class_calibration_requirements": {
            "schema": "paper_ii_all_algorithm_class_calibration_requirements_v1",
            "status": "candidate_contract_implemented_not_promoted_no_recovery_rerun_evidence_yet",
            "policy": "class-level settings are required for checkpoint, fixed McLachlan, and every comparator before promotion",
            "candidate_only_not_promoted": True,
            "case_specific_calibration_forbidden": True,
            "tuning_granularity": "coarse_hamiltonian_class",
            "required_tuning_classes": ["fermionic", "bosonic", "hybrid"],
            "required_algorithm_settings": [
                {
                    "algorithm_id": algorithm["algorithm_id"],
                    "settings_kind": algorithm["settings_kind"],
                }
                for algorithm in TABLE_I_ALGORITHMS
            ],
            "search_profiles": [dict(item) for item in CLASS_CALIBRATION_SEARCH_PROFILES],
            "candidate_class_settings_lock": str(ALL_ALGORITHM_CANDIDATE_LOCK),
            "candidate_input_manifest": str(CLASS_CALIBRATION_INPUT_MANIFEST),
        },
        "class_settings_sources": {
            "current_controller_class_settings_lock": str(CLASS_SETTINGS_LOCK),
            "all_algorithm_candidate_lock": str(ALL_ALGORITHM_CANDIDATE_LOCK),
            "all_algorithm_candidate_lock_status": "candidate_not_promoted_not_paper_evidence",
            "candidate_input_manifest": str(CLASS_CALIBRATION_INPUT_MANIFEST),
            "promotion_policy": "explicit_user_approval_and_separate_promotion_step_required",
            "status": "candidate all-algorithm class settings lock exists but is not promoted",
        },
        "calibration_scoring_selection_rules": dict(CALIBRATION_SCORING_SELECTION_RULES),
        "recovery_validation_gates": [dict(item) for item in RECOVERY_VALIDATION_GATES],
        "family_repair_rerun_eligibility_gate": _family_repair_rerun_eligibility_gate(),
        "final_aggregation_promotion_closeout_gate": _final_aggregation_promotion_closeout_gate(),
        "missing_evidence_fields": [
            {
                "field": "all_comparator_parity_correctness_sidecars",
                "status": "requirements_specified_sidecar_contracts_implemented_no_recovery_rerun_evidence_yet",
                "blocks_paper_facing_use": True,
            },
            {
                "field": "all_algorithm_class_calibration_lock",
                "status": "candidate_lock_and_input_generation_available_not_promoted_no_recovery_rerun_evidence_yet",
                "blocks_paper_facing_use": True,
            },
            {
                "field": "family_repair_tranche_pdfs_user_approved",
                "status": "not_started_user_visual_approval_required_before_rerun_escalation",
                "blocks_paper_facing_use": True,
            },
            {
                "field": "post_fetch_audit_no_blocking_errors",
                "status": "not_available_no_recovery_rerun_or_fetch_performed",
                "blocks_paper_facing_use": True,
            },
            {
                "field": "final_aggregation_user_approval",
                "status": "explicit_user_approval_required_separate_workflow",
                "blocks_paper_facing_use": True,
            },
            {
                "field": "table_promotion_user_approval",
                "status": "explicit_user_approval_required_separate_workflow",
                "blocks_paper_facing_use": True,
            },
            {
                "field": "epsilon_spec",
                "status": "not_asserted_by_recovery_manifest",
                "blocks_paper_facing_use": False,
            },
            {
                "field": "shots_total",
                "status": "not_asserted_by_recovery_manifest",
                "blocks_paper_facing_use": False,
            },
            {
                "field": "fidelity",
                "status": "not_asserted_by_recovery_manifest",
                "blocks_paper_facing_use": False,
            },
        ],
        "diagnostic_baseline_sources": [dict(item) for item in DIAGNOSTIC_BASELINE_SOURCES],
        "validation_contract": {
            "fail_closed_on_non_snake_selected_case": True,
            "fail_closed_on_unexpected_selected_case_count": True,
            "fail_closed_on_missing_same_seed_group_or_hash": True,
            "fail_closed_on_staged_fallback_pending_or_recovery_seed_source": True,
            "fail_closed_on_latest_phase3_source_artifact_missing_locally": True,
            "fail_closed_on_missing_required_parity_correctness_sidecar": True,
            "fail_closed_on_failed_required_parity_correctness_sidecar": True,
            "fail_closed_on_qiskit_parity_claim_for_unsupported_algorithm": True,
            "fail_closed_on_missing_all_algorithm_class_settings": True,
            "fail_closed_on_same_seed_group_hash_mismatch": True,
            "fail_closed_on_strict_controller_exact_leakage": True,
            "fail_closed_on_frozen_mclachlan_or_controller_when_exact_diagnostics_move": True,
            "fail_closed_on_family_repair_without_recovery_parity_calibration_gates": True,
            "fail_closed_on_final_aggregation_without_user_approval": True,
            "fail_closed_on_table_promotion_without_separate_user_workflow": True,
            "warn_on_missing_epsilon_spec_shots_total_or_fidelity": True,
            "future_evidence_root_policy": "create_new_root_never_overwrite_diagnostic_baseline_sources",
        },
    }
    validate_recovery_manifest(manifest)
    return manifest


def _source_hashes_for_manifest(manifest: Mapping[str, Any]) -> dict[str, str]:
    source_cases = str(manifest.get("source_cases_manifest", "")).strip()
    source_ledger = str(manifest.get("source_seed_ledger", "")).strip()
    if not source_cases:
        raise RecoveryManifestValidationError("source_cases_manifest is required")
    if not source_ledger:
        raise RecoveryManifestValidationError("source_seed_ledger is required")
    try:
        return {
            "source_cases_manifest_sha256": _sha256_file(source_cases),
            "source_seed_ledger_sha256": _sha256_file(source_ledger),
        }
    except OSError as exc:
        raise RecoveryManifestValidationError(f"declared source manifest path is not readable: {exc}") from exc


def _validate_work_item_scope_fields(manifest: Mapping[str, Any]) -> None:
    if manifest.get("manifest_role") != "diagnostic_freeze_and_snake_recovery_contract_only_not_paper_evidence":
        raise RecoveryManifestValidationError("manifest_role must remain diagnostic-only for the SNAKE recovery contract")
    if manifest.get("algorithms") != _expected_algorithms():
        raise RecoveryManifestValidationError("algorithms must match the Work Items 4-6 parity/correctness contract")

    parity = _as_mapping(manifest.get("parity_correctness_requirements", {}), label="parity_correctness_requirements")
    if parity.get("schema") != "paper_ii_table_i_parity_correctness_matrix_v1":
        raise RecoveryManifestValidationError("parity/correctness requirements schema must be matrix v1")
    if parity.get("status") != "specified_work_items_4_6_sidecar_contracts_implemented_no_recovery_rerun_evidence_yet":
        raise RecoveryManifestValidationError("parity/correctness requirements must remain the Work Items 4-6 matrix contract")
    if parity.get("matrix") != _parity_correctness_matrix():
        raise RecoveryManifestValidationError("parity/correctness matrix does not match required Work Items 4-6 contract")
    if parity.get("required_algorithm_ids") != [algorithm["algorithm_id"] for algorithm in TABLE_I_ALGORITHMS]:
        raise RecoveryManifestValidationError("parity/correctness required_algorithm_ids changed")
    if parity.get("missing_or_failed_required_check_blocks_table_i_use") is not True:
        raise RecoveryManifestValidationError("missing/failed parity checks must block future Table-I use")
    if parity.get("missing_failed_or_not_applicable_required_check_blocks_table_i_use") is not True:
        raise RecoveryManifestValidationError("missing/failed/not-applicable correctness checks must block future Table-I use")

    matrix = [_as_mapping(item, label="parity correctness matrix item") for item in parity.get("matrix", [])]
    algorithms_in_matrix = {str(item.get("algorithm_id", "")) for item in matrix}
    if algorithms_in_matrix != {item["algorithm_id"] for item in TABLE_I_ALGORITHMS}:
        raise RecoveryManifestValidationError("parity/correctness matrix must cover every Table-I algorithm")
    if not any(
        item.get("algorithm_id") == "dyn_fixed_mclachlan" and item.get("sidecar_name") == "mclachlan_correctness.json"
        for item in matrix
    ):
        raise RecoveryManifestValidationError("fixed McLachlan requires mclachlan_correctness.json beyond scaffold parity")
    for algorithm_id in ("dyn_avqds", "dyn_avqds_t"):
        if any(item.get("algorithm_id") == algorithm_id and item.get("sidecar_name") == "qiskit_parity.json" for item in matrix):
            raise RecoveryManifestValidationError(f"{algorithm_id} must not claim Qiskit parity as its required correctness sidecar")

    calibration = _as_mapping(manifest.get("class_calibration_requirements", {}), label="class_calibration_requirements")
    if calibration.get("schema") != "paper_ii_all_algorithm_class_calibration_requirements_v1":
        raise RecoveryManifestValidationError("class calibration requirements schema must be all-algorithm v1")
    if calibration.get("status") != "candidate_contract_implemented_not_promoted_no_recovery_rerun_evidence_yet":
        raise RecoveryManifestValidationError("class calibration requirements must remain candidate/not-promoted")
    if calibration.get("candidate_only_not_promoted") is not True:
        raise RecoveryManifestValidationError("class calibration requirements must be candidate-only/not-promoted")
    if calibration.get("case_specific_calibration_forbidden") is not True:
        raise RecoveryManifestValidationError("case-specific calibration must be forbidden")
    required_algorithm_settings = [
        {
            "algorithm_id": algorithm["algorithm_id"],
            "settings_kind": algorithm["settings_kind"],
        }
        for algorithm in TABLE_I_ALGORITHMS
    ]
    if calibration.get("required_algorithm_settings") != required_algorithm_settings:
        raise RecoveryManifestValidationError("class calibration requirements must cover every Table-I algorithm")
    if calibration.get("search_profiles") != [dict(item) for item in CLASS_CALIBRATION_SEARCH_PROFILES]:
        raise RecoveryManifestValidationError("class calibration search profiles changed")
    if calibration.get("candidate_class_settings_lock") != str(ALL_ALGORITHM_CANDIDATE_LOCK):
        raise RecoveryManifestValidationError("candidate all-algorithm class-settings lock path changed")

    class_sources = _as_mapping(manifest.get("class_settings_sources", {}), label="class_settings_sources")
    if class_sources.get("current_controller_class_settings_lock") != str(CLASS_SETTINGS_LOCK):
        raise RecoveryManifestValidationError("current controller class-settings lock path changed")
    if class_sources.get("all_algorithm_candidate_lock") != str(ALL_ALGORITHM_CANDIDATE_LOCK):
        raise RecoveryManifestValidationError("all_algorithm_candidate_lock path must match Work Items 7-11 candidate lock")
    if class_sources.get("all_algorithm_candidate_lock_status") != "candidate_not_promoted_not_paper_evidence":
        raise RecoveryManifestValidationError("all_algorithm_candidate_lock must remain candidate/not-promoted")
    candidate_lock_payload = _as_mapping(_read_json(ALL_ALGORITHM_CANDIDATE_LOCK), label="candidate class settings lock")
    candidate_lock_validation = validate_class_settings_lock_manifest(
        candidate_lock_payload,
        require_all_table_i_algorithm_classes=True,
    )
    if int(candidate_lock_validation.get("required_algorithm_class_entry_count", 0)) != len(TABLE_I_ALGORITHMS) * 3:
        raise RecoveryManifestValidationError("candidate class settings lock must cover every algorithm and tuning class")

    scoring = _as_mapping(manifest.get("calibration_scoring_selection_rules", {}), label="calibration_scoring_selection_rules")
    if scoring != CALIBRATION_SCORING_SELECTION_RULES:
        raise RecoveryManifestValidationError("calibration scoring/selection rules changed")
    gates = manifest.get("recovery_validation_gates", [])
    if gates != [dict(item) for item in RECOVERY_VALIDATION_GATES]:
        raise RecoveryManifestValidationError("recovery validation gates changed")

    family_gate = _as_mapping(
        manifest.get("family_repair_rerun_eligibility_gate", {}),
        label="family_repair_rerun_eligibility_gate",
    )
    if family_gate != _family_repair_rerun_eligibility_gate():
        raise RecoveryManifestValidationError("family repair/rerun eligibility gate changed")
    if family_gate.get("family_repair_can_resume") is not False or family_gate.get("rerun_eligibility") != "blocked":
        raise RecoveryManifestValidationError("family repair/rerun eligibility must remain blocked")

    closeout_gate = _as_mapping(
        manifest.get("final_aggregation_promotion_closeout_gate", {}),
        label="final_aggregation_promotion_closeout_gate",
    )
    if closeout_gate != _final_aggregation_promotion_closeout_gate():
        raise RecoveryManifestValidationError("final aggregation/promotion closeout gate changed")
    if closeout_gate.get("final_aggregation_allowed") is not False or closeout_gate.get("table_promotion_allowed") is not False:
        raise RecoveryManifestValidationError("final aggregation and table promotion must remain blocked")

    validation_contract = _as_mapping(manifest.get("validation_contract", {}), label="validation_contract")
    for key in (
        "fail_closed_on_family_repair_without_recovery_parity_calibration_gates",
        "fail_closed_on_final_aggregation_without_user_approval",
        "fail_closed_on_table_promotion_without_separate_user_workflow",
    ):
        if validation_contract.get(key) is not True:
            raise RecoveryManifestValidationError(f"validation contract {key} must be true")

    missing = manifest.get("missing_evidence_fields", [])
    if not isinstance(missing, Sequence) or isinstance(missing, (str, bytes)):
        raise RecoveryManifestValidationError("missing_evidence_fields must be a list")
    missing_by_field = {
        str(_as_mapping(item, label="missing evidence field").get("field", "")): _as_mapping(
            item, label="missing evidence field"
        )
        for item in missing
    }
    for field in (
        "all_comparator_parity_correctness_sidecars",
        "all_algorithm_class_calibration_lock",
        "family_repair_tranche_pdfs_user_approved",
        "post_fetch_audit_no_blocking_errors",
        "final_aggregation_user_approval",
        "table_promotion_user_approval",
    ):
        entry = missing_by_field.get(field)
        if entry is None or entry.get("blocks_paper_facing_use") is not True:
            raise RecoveryManifestValidationError(f"missing evidence field {field!r} must block paper-facing use")


def _validate_source_anchored_records(
    manifest: Mapping[str, Any],
    selected_cases: Sequence[Mapping[str, Any]],
    excluded_cases: Sequence[Mapping[str, Any]],
) -> None:
    expected_hashes = _source_hashes_for_manifest(manifest)
    if manifest.get("source_hashes") != expected_hashes:
        raise RecoveryManifestValidationError("source_hashes do not match declared source manifests")

    source_cases_path = str(manifest.get("source_cases_manifest"))
    expected_selected, expected_excluded = _case_records_from_source(source_cases_path)
    actual_selected_by_id = {str(case.get("case_id", "")): dict(case) for case in selected_cases}
    expected_selected_by_id = {case["case_id"]: case for case in expected_selected}
    if actual_selected_by_id != expected_selected_by_id:
        raise RecoveryManifestValidationError("selected_cases do not exactly match SNAKE cases from source manifest")

    actual_excluded_by_id = {str(case.get("case_id", "")): dict(case) for case in excluded_cases}
    expected_excluded_by_id = {case["case_id"]: case for case in expected_excluded}
    if actual_excluded_by_id != expected_excluded_by_id:
        raise RecoveryManifestValidationError("excluded_cases do not exactly match non-SNAKE cases from source manifest")

    ledger = _as_mapping(_read_json(str(manifest.get("source_seed_ledger"))), label="source seed ledger")
    ledger_entries = [_as_mapping(entry, label="source seed ledger entry") for entry in ledger.values()]
    for case in selected_cases:
        family = str(case.get("family", ""))
        seed_hash = str(case.get("seed_artifact_sha256", ""))
        matches = [
            entry
            for entry in ledger_entries
            if entry.get("seed_track") == "snake"
            and entry.get("problem") == family
            and entry.get("seed_artifact_sha256") == seed_hash
        ]
        if len(matches) != 1:
            raise RecoveryManifestValidationError(
                f"selected case {case.get('case_id')!r} must match exactly one SNAKE seed-ledger entry"
            )
        entry = matches[0]
        for key in ("seed_selection_policy", "source_artifact_json", "source_artifact_sha256"):
            if str(case.get(key, "")) != str(entry.get(key, "")):
                raise RecoveryManifestValidationError(
                    f"selected case {case.get('case_id')!r} {key} does not match seed ledger"
                )
        if Path(str(case.get("normalized_seed_artifact_json", ""))).name != Path(
            str(entry.get("normalized_seed_artifact_json", ""))
        ).name:
            raise RecoveryManifestValidationError(
                f"selected case {case.get('case_id')!r} normalized seed artifact does not match seed ledger"
            )


def validate_recovery_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail-closed validation for the SNAKE recovery contract."""

    manifest = _as_mapping(payload, label="recovery manifest")
    if manifest.get("schema") != SCHEMA:
        raise RecoveryManifestValidationError(f"unsupported recovery manifest schema {manifest.get('schema')!r}")
    _validate_work_item_scope_fields(manifest)

    selected = manifest.get("selected_cases", [])
    if not isinstance(selected, Sequence) or isinstance(selected, (str, bytes)):
        raise RecoveryManifestValidationError("selected_cases must be a list")
    selected_cases = [_as_mapping(case, label="selected case") for case in selected]
    if len(selected_cases) != EXPECTED_SNAKE_CASE_COUNT:
        raise RecoveryManifestValidationError(
            f"expected {EXPECTED_SNAKE_CASE_COUNT} SNAKE recovery cases, got {len(selected_cases)}"
        )
    if int(manifest.get("selected_case_count", -1)) != len(selected_cases):
        raise RecoveryManifestValidationError("selected_case_count does not match selected_cases length")

    families = {str(case.get("family", "")) for case in selected_cases}
    expected_families = set(EXPECTED_FAMILIES)
    if families != expected_families:
        raise RecoveryManifestValidationError(
            f"selected SNAKE families mismatch: expected {sorted(expected_families)}, got {sorted(families)}"
        )

    for family in EXPECTED_FAMILIES:
        drive_values = sorted(float(case.get("drive_metadata", {}).get("A")) for case in selected_cases if case.get("family") == family)
        if drive_values != list(EXPECTED_DRIVE_AMPLITUDES):
            raise RecoveryManifestValidationError(
                f"family {family!r} must have drives {list(EXPECTED_DRIVE_AMPLITUDES)}, got {drive_values}"
            )

    seen_case_ids: set[str] = set()
    seen_groups: set[str] = set()
    for case in selected_cases:
        case_id = str(case.get("case_id", ""))
        if not case_id:
            raise RecoveryManifestValidationError("selected case missing case_id")
        if case_id in seen_case_ids:
            raise RecoveryManifestValidationError(f"duplicate selected case_id {case_id!r}")
        seen_case_ids.add(case_id)
        if str(case.get("seed_track", "")).strip().lower() != "snake":
            raise RecoveryManifestValidationError(f"selected case {case_id!r} is not SNAKE")
        if "posgeo" in case_id.lower() or "posgeo" in str(case.get("artifact_json", "")).lower():
            raise RecoveryManifestValidationError(f"selected case {case_id!r} contains POS-GEO provenance")
        seed_hash = str(case.get("seed_artifact_sha256", "")).strip()
        if not seed_hash:
            raise RecoveryManifestValidationError(f"selected case {case_id!r} lacks seed_artifact_sha256")
        group_id = str(case.get("same_seed_comparator_group_id", "")).strip()
        if not group_id:
            raise RecoveryManifestValidationError(f"selected case {case_id!r} lacks same_seed_comparator_group_id")
        if group_id in seen_groups:
            raise RecoveryManifestValidationError(f"duplicate same_seed_comparator_group_id {group_id!r}")
        seen_groups.add(group_id)
        if bool(case.get("latest_phase3_source_artifact_missing_locally", False)):
            raise RecoveryManifestValidationError(
                f"selected case {case_id!r} reports latest_phase3_source_artifact_missing_locally=true"
            )
        forbidden = _seed_source_has_forbidden_token(case, case)
        if forbidden:
            raise RecoveryManifestValidationError(
                f"selected case {case_id!r} has forbidden seed source token {forbidden!r}"
            )

    excluded = manifest.get("excluded_cases", [])
    if not isinstance(excluded, Sequence) or isinstance(excluded, (str, bytes)):
        raise RecoveryManifestValidationError("excluded_cases must be a list")
    excluded_cases = [_as_mapping(case, label="excluded case") for case in excluded]
    if int(manifest.get("excluded_case_count", -1)) != len(excluded_cases):
        raise RecoveryManifestValidationError("excluded_case_count does not match excluded_cases length")
    if any(str(case.get("seed_track", "")).strip().lower() == "snake" for case in excluded_cases):
        raise RecoveryManifestValidationError("excluded_cases contains a SNAKE case")

    baseline_sources = manifest.get("diagnostic_baseline_sources", [])
    if not baseline_sources:
        raise RecoveryManifestValidationError("diagnostic_baseline_sources must be non-empty")
    for source in baseline_sources:
        item = _as_mapping(source, label="diagnostic baseline source")
        if "diagnostic" not in str(item.get("status", "")).lower():
            raise RecoveryManifestValidationError(
                f"diagnostic baseline source {item.get('source_id')!r} is not marked diagnostic"
            )
        if str(item.get("mutation_policy", "")) not in {
            "do_not_edit_in_recovery_manifest_wave",
            "do_not_overwrite_or_reaggregate",
        }:
            raise RecoveryManifestValidationError(
                f"diagnostic baseline source {item.get('source_id')!r} lacks a freeze mutation policy"
            )

    _validate_source_anchored_records(manifest, selected_cases, excluded_cases)

    return {
        "status": "passed",
        "selected_case_count": len(selected_cases),
        "excluded_case_count": len(excluded_cases),
        "families": sorted(families),
        "same_seed_group_count": len(seen_groups),
    }


def write_recovery_manifest(output_path: str | Path = RECOVERY_MANIFEST) -> dict[str, Any]:
    manifest = build_recovery_manifest()
    _write_json(output_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(RECOVERY_MANIFEST), help="manifest JSON path to write")
    parser.add_argument("--validate-only", action="store_true", help="validate an existing output manifest without rewriting it")
    args = parser.parse_args()

    if args.validate_only:
        payload = _read_json(args.output)
        result = validate_recovery_manifest(payload)
    else:
        payload = write_recovery_manifest(args.output)
        result = validate_recovery_manifest(payload)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
