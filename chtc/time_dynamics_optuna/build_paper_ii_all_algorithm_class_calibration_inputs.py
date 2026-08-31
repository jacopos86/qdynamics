#!/usr/bin/env python3
"""Build SNAKE-only Paper-II all-algorithm class-calibration inputs.

This is a candidate-definition builder only.  It consumes the SNAKE recovery
manifest, preserves the v2 static seed artifacts/hashes and same-seed groups,
and emits namespaced record/case manifests for later calibration smokes or
candidate batches.  It does not submit CHTC jobs, copy seed artifacts, promote
settings, or mutate fetched/raw outputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (
    DYNAMICS_ALL_ALGORITHM_CLASS_SETTINGS_CANDIDATE_SOURCE,
    DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND,
    json_safe,
)
from pipelines.time_dynamics.tables.table_lock_contract import validate_class_settings_lock_manifest
from chtc.generic_time_dynamics_table.build_paper_ii_snake_recovery_manifest import (
    RECOVERY_MANIFEST as DEFAULT_RECOVERY_MANIFEST_REL,
    validate_recovery_manifest,
)

DEFAULT_RECOVERY_MANIFEST = REPO_ROOT / DEFAULT_RECOVERY_MANIFEST_REL
DEFAULT_CANDIDATE_LOCK = (
    REPO_ROOT
    / "chtc/generic_time_dynamics_table/input/class_settings/"
    / "paper_ii_all_algorithm_class_settings_candidate_lock_v1.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "chtc/time_dynamics_optuna/input"
OUTPUT_STEM = "paper_ii_all_algorithm_class_calibration_v1"
SCHEMA = "paper_ii_all_algorithm_class_calibration_inputs_v1"
CREATED_AT = "2026-05-19T00:00:00+00:00"

ALGORITHM_ORDER: tuple[str, ...] = tuple(DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND)

CALIBRATION_SEARCH_PROFILES: dict[str, dict[str, Any]] = {
    "dyn_controller_full": {
        "profile_id": "checkpoint_controller_current_lock_seed_v1",
        "settings_kind": "controller",
        "search_space": {
            "controller_policy_source": ["current_promoted_lock_seed"],
            "append_prune_guardrail_refinement": ["class_level_only"],
        },
        "runtime_decision_policy": "strict_exact_free_online_feedback",
    },
    "dyn_fixed_mclachlan": {
        "profile_id": "fixed_mclachlan_integrator_policy_v1",
        "settings_kind": "mclachlan",
        "search_space": {
            "fixed_mclachlan_integrator_policy": ["euler", "rk4", "auto_euler_rk4"],
            "regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "pinv_rcond": [1.0e-12, 1.0e-10, 1.0e-8],
        },
        "runtime_decision_policy": "fixed_scaffold_exact_free_mclachlan_geometry",
    },
    "dyn_product_formula_envelope": {
        "profile_id": "product_formula_order_v1",
        "settings_kind": "comparator",
        "search_space": {"product_formula_order": [1, 2]},
        "runtime_decision_policy": "fixed_order_no_exact_runtime_selection",
    },
    "dyn_qdrift": {
        "profile_id": "qdrift_samples_rng_v1",
        "settings_kind": "comparator",
        "search_space": {
            "qdrift_samples_per_interval": [8, 16, 32],
            "qdrift_rng_seed_policy": "deterministic_class_level_from_candidate_manifest",
        },
        "runtime_decision_policy": "deterministic_manifest_rng_no_exact_runtime_selection",
    },
    "dyn_fixed_pvqd": {
        "profile_id": "fixed_pvqd_projection_v1",
        "settings_kind": "comparator",
        "search_space": {
            "pvqd_optimizer_maxiter": [12, 24, 48],
            "pvqd_overlap_tol": [1.0e-6, 1.0e-8],
            "pvqd_target_product_formula_order": [1, 2],
        },
        "runtime_decision_policy": "ideal_overlap_estimator_for_prepared_product_formula_target",
    },
    "dyn_adaptive_pvqd": {
        "profile_id": "adaptive_pvqd_append_projection_v1",
        "settings_kind": "comparator",
        "search_space": {
            "adaptive_pvqd_optimizer_maxiter": [12, 24, 48],
            "adaptive_pvqd_overlap_tol": [1.0e-6, 1.0e-8],
            "adaptive_pvqd_target_product_formula_order": [1, 2],
            "adaptive_pvqd_append_loss_threshold": [1.0e-2, 1.0e-3, 1.0e-4],
            "adaptive_pvqd_append_min_loss_improvement": [1.0e-4, 1.0e-5],
            "adaptive_pvqd_append_candidate_limit": [2, 4, 8],
        },
        "runtime_decision_policy": "append_scoring_from_prepared_target_overlap_not_exact_reference",
    },
    "dyn_avqds": {
        "profile_id": "avqds_rhs_tangent_v1",
        "settings_kind": "comparator",
        "search_space": {
            "avqds_regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "avqds_pinv_relative_cutoff": [1.0e-12, 1.0e-10, 1.0e-8],
            "avqds_append_rhs_residual_ratio_threshold": [5.0e-3, 1.0e-2, 5.0e-2],
            "avqds_append_min_residual_ratio_gain": [1.0e-5, 1.0e-4],
            "avqds_append_candidate_limit": [2, 4, 8],
        },
        "runtime_decision_policy": "rhs_tangent_residual_from_current_prepared_state_not_exact_target",
    },
    "dyn_avqds_t": {
        "profile_id": "avqds_t_target_tangent_v1",
        "settings_kind": "comparator",
        "search_space": {
            "avqds_t_regularization_lambda": [1.0e-10, 1.0e-8, 1.0e-6],
            "avqds_t_pinv_relative_cutoff": [1.0e-12, 1.0e-10, 1.0e-8],
            "avqds_t_target_product_formula_order": [1, 2],
            "avqds_t_append_target_tangent_residual_ratio_threshold": [5.0e-3, 1.0e-2, 5.0e-2],
            "avqds_t_append_min_residual_ratio_gain": [1.0e-5, 1.0e-4],
            "avqds_t_append_candidate_limit": [2, 4, 8],
        },
        "runtime_decision_policy": "product_formula_target_tangent_not_exact_target_forecast",
    },
}

CALIBRATION_SCORING_RULES: dict[str, Any] = {
    "schema": "paper_ii_class_calibration_scoring_rules_v1",
    "strict_online_controller_feedback": "exact_free_measurement_compatible_observables_only",
    "comparator_runtime_decisions": "exact_free_prepared_state_or_product_formula_target_observables_only",
    "offline_class_calibration_review": "may_use_exact_diagnostics_for_physical_error_reporting_only",
    "discard_gates": [
        "failed_same_seed_validation",
        "missing_or_failed_required_parity_correctness",
        "exact_decision_leakage",
        "latest_phase3_source_artifact_missing_locally",
        "staged_fallback_pending_or_recovery_seed_source",
        "pathological_incomplete_or_frozen_trajectory",
    ],
    "primary_rank": "median_physical_error_over_all_snake_cases_and_drives_in_tuning_class",
    "pareto_secondary": "resource_cost_frontier_compiled_two_qubit_count_depth_shot_proxy",
    "tie_breakers": ["lower_resource_cost", "simpler_settings", "deterministic_record_order"],
    "case_specific_winner_policy": "cannot_select_case_specific_winner_unless_promoted_for_entire_tuning_class",
    "promotion_policy": "no_setting_promoted_without_explicit_user_approval_and_separate_promotion_step",
}

RECORD_HEADER: tuple[str, ...] = (
    "record_id",
    "queue",
    "kind",
    "family",
    "tuning_class",
    "case_id",
    "algorithm_id",
    "settings_kind",
    "candidate_search_profile_id",
    "case_manifest",
    "class_settings_manifest",
    "candidate_only_not_promoted",
    "require_algorithm_class_settings",
    "require_parity_correctness_sidecars",
    "parity_correctness_sidecar_name",
    "seed_track",
    "same_seed_comparator_group_id",
    "seed_artifact_sha256",
    "source_artifact_json",
    "artifact_json",
    "t_final",
    "num_times",
    "drive_A",
    "candidate_search_profile_json",
    "calibration_scoring_policy",
)

PARITY_SIDECAR_BY_ALGORITHM: dict[str, str] = {
    "dyn_controller_full": "controller_strict_decision_contract_telemetry",
    "dyn_fixed_mclachlan": "qiskit_parity.json+mclachlan_correctness.json",
    "dyn_product_formula_envelope": "qiskit_parity.json",
    "dyn_qdrift": "qiskit_parity.json",
    "dyn_fixed_pvqd": "qiskit_parity.json",
    "dyn_adaptive_pvqd": "qiskit_parity.json",
    "dyn_avqds": "avqds_correctness.json",
    "dyn_avqds_t": "avqds_t_correctness.json",
}


def _rel(path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p.relative_to(REPO_ROOT))
    return str(p)


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_HEADER, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_ids(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.write_text("\n".join(str(row["record_id"]) for row in rows) + "\n", encoding="utf-8")


def _slug(value: str) -> str:
    return str(value).replace("dyn_", "").replace("_", "-")


def _case_manifest_artifact_path(case: Mapping[str, Any]) -> str:
    return str(Path("../../generic_time_dynamics_table/input") / str(case["artifact_json"]))


def _runtime_case_record(case: Mapping[str, Any], *, candidate_lock_rel: str) -> dict[str, Any]:
    drive = case.get("drive_metadata", {}) if isinstance(case.get("drive_metadata"), Mapping) else {}
    return {
        "case_id": str(case["case_id"]),
        "family": str(case["family"]),
        "table_class": str(case.get("table_class", case.get("tuning_class", ""))),
        "tuning_class": str(case["tuning_class"]),
        "artifact_json": _case_manifest_artifact_path(case),
        "description": "Paper-II SNAKE-only all-algorithm class-calibration candidate case",
        "t_final": float(case["t_final"]),
        "num_times": int(case["num_times"]),
        "loader_mode": "replay_family",
        "generator_family": "match_adapt",
        "fallback_family": "full_meta",
        "append_pool_family": "match_replay",
        "metadata": {
            "seed_lock": {
                "seed_track": "snake",
                "seed_selection_policy": str(case.get("seed_selection_policy", "")),
                "seed_artifact_sha256": str(case["seed_artifact_sha256"]),
                "source_artifact_sha256": str(case.get("source_artifact_sha256", "")),
                "source_artifact_json": str(case.get("source_artifact_json", "")),
                "same_seed_comparator_group_id": str(case["same_seed_comparator_group_id"]),
            },
            "same_seed_comparator_group_id": str(case["same_seed_comparator_group_id"]),
            "controller_settings_scope": "coarse_hamiltonian_class",
            "static_scaffold_scope": "benchmark_point",
            "canonical_case_manifest_id": "paper_ii_seed_tracks_cases_v2",
            "paper_ii_table_lock": True,
            "latest_phase3_source_artifact_missing_locally": bool(
                case.get("latest_phase3_source_artifact_missing_locally", False)
            ),
            "drive": dict(drive),
            "enable_drive": bool(drive.get("enable_drive", True)),
            "diagnostic_exact_reference_mode": str(case.get("diagnostic_exact_reference_policy", "")),
            "qpu_faithful_controller_data_contract": str(
                case.get("observable_policy", {}).get("qpu_faithful_controller_data_contract", "")
                if isinstance(case.get("observable_policy", {}), Mapping)
                else ""
            ),
            "class_settings_lock_manifest": candidate_lock_rel,
            "require_algorithm_class_settings": True,
            "require_parity_correctness_sidecars": True,
            "candidate_only_not_promoted": True,
        },
    }


def _record_from_case_algorithm(
    case: Mapping[str, Any],
    *,
    algorithm_id: str,
    case_manifest_rel: str,
    candidate_lock_rel: str,
    queue: str,
    prefix: str,
) -> dict[str, str]:
    profile = CALIBRATION_SEARCH_PROFILES[algorithm_id]
    drive = case.get("drive_metadata", {}) if isinstance(case.get("drive_metadata"), Mapping) else {}
    record_id = f"{prefix}__{_slug(algorithm_id)}__{case['case_id']}"
    return {
        "record_id": record_id,
        "queue": queue,
        "kind": "benchmark",
        "family": str(case["family"]),
        "tuning_class": str(case["tuning_class"]),
        "case_id": str(case["case_id"]),
        "algorithm_id": str(algorithm_id),
        "settings_kind": str(profile["settings_kind"]),
        "candidate_search_profile_id": str(profile["profile_id"]),
        "case_manifest": case_manifest_rel,
        "class_settings_manifest": candidate_lock_rel,
        "candidate_only_not_promoted": "1",
        "require_algorithm_class_settings": "1",
        "require_parity_correctness_sidecars": "1",
        "parity_correctness_sidecar_name": PARITY_SIDECAR_BY_ALGORITHM[str(algorithm_id)],
        "seed_track": "snake",
        "same_seed_comparator_group_id": str(case["same_seed_comparator_group_id"]),
        "seed_artifact_sha256": str(case["seed_artifact_sha256"]),
        "source_artifact_json": str(case.get("source_artifact_json", "")),
        "artifact_json": str(case["artifact_json"]),
        "t_final": str(float(case["t_final"])),
        "num_times": str(int(case["num_times"])),
        "drive_A": str(float(drive.get("A", 0.0))),
        "candidate_search_profile_json": json.dumps(json_safe(profile), sort_keys=True, separators=(",", ":")),
        "calibration_scoring_policy": CALIBRATION_SCORING_RULES["primary_rank"],
    }


def _smoke_cases(selected_cases: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_class: dict[str, Mapping[str, Any]] = {}
    for case in sorted(selected_cases, key=lambda item: (str(item.get("tuning_class", "")), str(item.get("case_id", "")))):
        by_class.setdefault(str(case.get("tuning_class", "")), case)
    return [by_class[key] for key in sorted(by_class)]


def build_inputs(
    *,
    recovery_manifest: Path = DEFAULT_RECOVERY_MANIFEST,
    candidate_lock: Path = DEFAULT_CANDIDATE_LOCK,
    out_dir: Path = DEFAULT_OUT_DIR,
    output_stem: str = OUTPUT_STEM,
) -> dict[str, Any]:
    recovery_payload = _read_json(recovery_manifest)
    validate_recovery_manifest(recovery_payload)
    candidate_payload = _read_json(candidate_lock)
    lock_validation = validate_class_settings_lock_manifest(
        candidate_payload,
        require_all_table_i_algorithm_classes=True,
    )

    selected_cases = list(recovery_payload["selected_cases"])
    candidate_lock_rel = _rel(candidate_lock)
    case_manifest = out_dir / f"{output_stem}_cases.json"
    case_manifest_rel = _rel(case_manifest)
    runtime_cases = [
        _runtime_case_record(case, candidate_lock_rel=candidate_lock_rel)
        for case in selected_cases
    ]
    case_manifest_payload = {
        "schema": "paper_ii_all_algorithm_class_calibration_case_manifest_v1",
        "created_at": CREATED_AT,
        "source_recovery_manifest": _rel(recovery_manifest),
        "candidate_class_settings_lock": candidate_lock_rel,
        "case_count": len(runtime_cases),
        "candidate_only_not_promoted": True,
        "cases": runtime_cases,
    }
    _write_json(case_manifest, case_manifest_payload)

    candidate_rows = [
        _record_from_case_algorithm(
            case,
            algorithm_id=algorithm_id,
            case_manifest_rel=case_manifest_rel,
            candidate_lock_rel=candidate_lock_rel,
            queue=f"{output_stem}_candidate",
            prefix=output_stem,
        )
        for case in selected_cases
        for algorithm_id in ALGORITHM_ORDER
    ]
    smoke_rows = [
        _record_from_case_algorithm(
            case,
            algorithm_id=algorithm_id,
            case_manifest_rel=case_manifest_rel,
            candidate_lock_rel=candidate_lock_rel,
            queue=f"{output_stem}_smoke",
            prefix=f"{output_stem}_smoke",
        )
        for case in _smoke_cases(selected_cases)
        for algorithm_id in ALGORITHM_ORDER
    ]

    candidate_records = out_dir / f"{output_stem}_records.tsv"
    smoke_records = out_dir / f"{output_stem}_smoke_records.tsv"
    candidate_ids = out_dir / f"{output_stem}_record_ids.txt"
    smoke_ids = out_dir / f"{output_stem}_smoke_record_ids.txt"
    _write_tsv(candidate_records, candidate_rows)
    _write_tsv(smoke_records, smoke_rows)
    _write_ids(candidate_ids, candidate_rows)
    _write_ids(smoke_ids, smoke_rows)

    manifest = {
        "schema": SCHEMA,
        "manifest_id": output_stem,
        "created_at": CREATED_AT,
        "manifest_role": "candidate_only_snake_class_calibration_input_definition_not_submitted_not_promoted",
        "source_recovery_manifest": _rel(recovery_manifest),
        "source_recovery_manifest_sha256": _sha256(recovery_manifest),
        "candidate_class_settings_lock": candidate_lock_rel,
        "candidate_class_settings_lock_sha256": _sha256(candidate_lock),
        "candidate_settings_source": DYNAMICS_ALL_ALGORITHM_CLASS_SETTINGS_CANDIDATE_SOURCE,
        "candidate_class_settings_lock_validation": lock_validation,
        "case_manifest": case_manifest_rel,
        "case_count": int(len(selected_cases)),
        "algorithm_ids": list(ALGORITHM_ORDER),
        "algorithm_count": int(len(ALGORITHM_ORDER)),
        "candidate_record_count": int(len(candidate_rows)),
        "smoke_record_count": int(len(smoke_rows)),
        "candidate_records": _rel(candidate_records),
        "smoke_records": _rel(smoke_records),
        "candidate_record_ids": _rel(candidate_ids),
        "smoke_record_ids": _rel(smoke_ids),
        "seed_track_filter": ["snake"],
        "same_seed_policy": "preserve_v2_same_seed_comparator_group_id_and_seed_artifact_sha256",
        "seed_artifact_policy": "do_not_copy_stage_regenerate_or_substitute_seed_artifacts",
        "smoke_and_candidate_sets_are_separate": True,
        "candidate_only_not_promoted": True,
        "no_chtc_submission_performed": True,
        "class_level_selection_only": True,
        "search_profiles": CALIBRATION_SEARCH_PROFILES,
        "calibration_scoring_rules": CALIBRATION_SCORING_RULES,
    }
    _write_json(out_dir / f"{output_stem}_manifest.json", manifest)
    validate_inputs_manifest(manifest)
    return json_safe(manifest)


def validate_inputs_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise ValueError(f"unsupported calibration input manifest schema {payload.get('schema')!r}")
    if payload.get("candidate_only_not_promoted") is not True:
        raise ValueError("calibration inputs must be candidate-only/not-promoted")
    if payload.get("no_chtc_submission_performed") is not True:
        raise ValueError("calibration input builder must not submit CHTC jobs")
    if payload.get("seed_track_filter") != ["snake"]:
        raise ValueError("calibration input manifest must be SNAKE-only")
    if int(payload.get("case_count", 0)) != 20:
        raise ValueError("calibration input manifest must preserve the 20-case SNAKE recovery surface")
    if int(payload.get("algorithm_count", 0)) != len(ALGORITHM_ORDER):
        raise ValueError("calibration input manifest must cover every Table-I algorithm")
    expected_candidate = int(payload["case_count"]) * int(payload["algorithm_count"])
    if int(payload.get("candidate_record_count", -1)) != expected_candidate:
        raise ValueError("candidate record count must be case_count * algorithm_count")
    if int(payload.get("smoke_record_count", 0)) <= 0:
        raise ValueError("smoke record set must be separate and non-empty")
    return {
        "status": "passed",
        "candidate_record_count": int(payload["candidate_record_count"]),
        "smoke_record_count": int(payload["smoke_record_count"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-manifest", type=Path, default=DEFAULT_RECOVERY_MANIFEST)
    parser.add_argument("--candidate-lock", type=Path, default=DEFAULT_CANDIDATE_LOCK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--output-stem", default=OUTPUT_STEM)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    manifest_path = args.out_dir / f"{args.output_stem}_manifest.json"
    if args.validate_only:
        payload = _read_json(manifest_path)
        result = validate_inputs_manifest(payload)
    else:
        payload = build_inputs(
            recovery_manifest=args.recovery_manifest.resolve(),
            candidate_lock=args.candidate_lock.resolve(),
            out_dir=args.out_dir.resolve(),
            output_stem=str(args.output_stem),
        )
        result = validate_inputs_manifest(payload)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
