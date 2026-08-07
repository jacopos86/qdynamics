#!/usr/bin/env python3
"""Generate fixed-policy strong-Hubbard SNAKE ablation records.

This is not an Optuna recalibration matrix. It replays the visible Paper-I
Table-I strong-Hubbard SNAKE source policy, then changes one mechanism per row.
The baseline smoke row should reproduce the visible source before the ablation
rows are treated as evidence.
"""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

BATCH_ID = "paper_i_hubbard_strong_ablation_selectedlog_v4_t0_fixed_20260601_v1"
SOURCE_DIR = SCRIPT_DIR / "input" / "routeA_paper_i_three_model_selected_logical_20260525_v4"
SOURCE_RECORDS_TSV = SOURCE_DIR / "paper_i_three_model_routeA_records.tsv"
SOURCE_RECORD_ID = "routeA_paper_i_three_model_hubbard_l2_three_model_strong_selected_logical_v4"
SOURCE_SUMMARY_JSON = (
    REPO_ROOT
    / "tmp/chtc_partial_paper_i_three_model_v3_v4_20260525/extracted/raw_outputs"
    / SOURCE_RECORD_ID
    / "summary.json"
)
SOURCE_TRIAL_DIR = (
    REPO_ROOT
    / "tmp/chtc_partial_paper_i_three_model_v3_v4_20260525/extracted/raw_outputs"
    / SOURCE_RECORD_ID
    / "run/hubbard_L2_three_model_strong/trial_0000"
)
SOURCE_EFFECTIVE_MANIFEST_JSON = SOURCE_TRIAL_DIR / "effective_trial_manifest.json"
SOURCE_RESULT_JSON = SOURCE_TRIAL_DIR / "hubbard_L2_three_model_strong/json/result.json"

INPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID
RECORDS_TSV = INPUT_DIR / "paper_i_hubbard_strong_selectedlog_v4_t0_fixed_v1_ablation_records.tsv"
ALL_RECORD_IDS = INPUT_DIR / "paper_i_hubbard_strong_selectedlog_v4_t0_fixed_v1_ablation_record_ids.txt"
SMOKE_RECORD_IDS = INPUT_DIR / "paper_i_hubbard_strong_selectedlog_v4_t0_fixed_v1_smoke_record_ids.txt"
ABLATION_ONLY_RECORD_IDS = (
    INPUT_DIR / "paper_i_hubbard_strong_selectedlog_v4_t0_fixed_v1_ablation_only_record_ids.txt"
)
SUMMARY_JSON = INPUT_DIR / "source_hubbard_strong_selectedlog_v4_trial0_replay_summary.json"
MANIFEST_JSON = INPUT_DIR / "ablation_hubbard_strong_selectedlog_v4_t0_fixed_v1_manifest.json"
SUBMIT_ALL = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"
SUBMIT_SMOKE = SCRIPT_DIR / f"submit_{BATCH_ID}_smoke.sub"
SUBMIT_ABLATION_ONLY = SCRIPT_DIR / f"submit_{BATCH_ID}_ablation_only.sub"

SOURCE_EXPECTED = {
    "benchmark_ids": "hubbard_L2_three_model_strong",
    "families": "hubbard",
    "suite_profile": "paper_i_three_model_main_20260525_v1",
    "static_route_id": "route_a",
    "meta_feature_profile": "paper_i_production_v1",
    "route_base_pool_key": "full_meta",
    "selected_logical_route": "historical_selected",
    "selected_logical_transfer_mode": "exact_match_v1",
    "phase2_novelty_mode": "collective_span_v1",
    "phase3_selector_policy": "algebraic_nested_v1",
    "phase3_selector_geometry_mode": "reduced",
    "phase3_novelty_ablation_mode": "off",
    "phase3_window_relaxation_mode": "reduced",
    "phase2_enable_batching": "true",
    "phase3_enable_batching": "true",
    "phase3_batch_selection_mode": "reduced_plane",
    "phase3_batch_prefilter_mode": "off",
    "phase3_nested_window_application": "composed_batch_window_v1",
    "phase1_prune_enabled": "true",
    "phase1_prune_policy": "recoverability_ladder_v1",
    "phase1_prune_mode": "both",
    "phase1_prune_amplitude_witness_required": "true",
    "primary_selector_score_key": "full_v2_score",
    "auxiliary_terms_primary_mode": "tie_break_only",
    "continuation_mode": "phase3_v1",
    "algebraic_shortlisting_enabled": "true",
    "hardware_resolution_schema": "gradient_resolution_v1",
    "hardware_resolution_mode": "ideal",
}

BASE_RECORD_OVERLAY = {
    "mode": "oracle-grid",
    "benchmark_ids": "hubbard_L2_three_model_strong",
    "families": "hubbard",
    "oracle_summary_root": f"chtc/phase3_optuna/input/{BATCH_ID}/{SUMMARY_JSON.name}",
    "oracle_enqueue_limit": "1",
    "oracle_required_static_route_id": "route_a",
    "oracle_required_suite_profile": "paper_i_three_model_main_20260525_v1",
    "oracle_require_phase0_aware": "true",
    "oracle_require_compatible_warm_starts": "true",
    "enqueue_default": "false",
    "enqueue_historical": "false",
    "n_trials": "1",
    "n_jobs": "1",
    "benchmarks_per_trial_jobs": "1",
    "seed": "95118",
    "trial_timeout_sec": "21600",
    "compile_timeout_sec": "1200",
    "robustness_gate": "off",
    "phase3_oracle_gradient_mode": "off",
    "phase3_oracle_inner_objective_mode": "exact",
    "phase3_oracle_value_noise_model": "off",
    "phase3_oracle_value_noise_std": "0.0",
    "route_evidence_role": "fixed_policy_ablation_candidate",
    "paper_i_recovery_intent": "strong_hubbard_fixed_policy_ablation",
}

# Current code exposes additional hardware-resolution lambdas that were not in
# the selected-logical v4 source trial. Pin them to zero for source-compatible
# replay; the original source already used compile/measure/phase-2 cost terms.
SOURCE_COMMAND_COMPAT_OVERRIDES = {
    "lambda_1q": 0.0,
    "lambda_2q": 0.0,
    "lambda_d": 0.0,
    "lambda_shot": 0.0,
    "lambda_theta": 0.0,
}

NO_HARDWARE_COST_OVERRIDES = {
    "lambda_compile": 0.0,
    "lambda_measure": 0.0,
    "lambda_leak": 0.0,
    "lambda_1q": 0.0,
    "lambda_2q": 0.0,
    "lambda_d": 0.0,
    "lambda_shot": 0.0,
    "lambda_theta": 0.0,
    "compile_cx_weight": 0.0,
    "compile_sq_weight": 0.0,
    "compile_rotation_step_weight": 0.0,
    "compile_position_shift_weight": 0.0,
    "compile_refit_active_weight": 0.0,
    "measure_groups_weight": 0.0,
    "measure_shots_weight": 0.0,
    "measure_reuse_weight": 0.0,
    "opt_dim_cost_scale": 0.0,
    "phase2_w_depth": 0.0,
    "phase2_w_group": 0.0,
    "phase2_w_shot": 0.0,
    "phase2_w_optdim": 0.0,
    "phase2_w_reuse": 0.0,
    "phase2_w_lifetime": 0.0,
    "phase3_backend_cost_mode": "proxy",
}

VARIANTS: tuple[dict[str, Any], ...] = (
    {
        "name": "full_snake_anchor",
        "record_suffix": "full_snake_anchor",
        "static_route_id": "route_a",
        "note": "Baseline replay of visible strong-Hubbard SNAKE trial 0; no mechanism disabled.",
        "row_updates": {},
        "override_updates": {},
    },
    {
        "name": "no_hardware_cost",
        "record_suffix": "no_hardware_cost",
        "static_route_id": "unspecified",
        "note": "Disable hardware/compile/measurement cost terms inside the fixed selector policy only.",
        "row_updates": {},
        "override_updates": NO_HARDWARE_COST_OVERRIDES,
    },
    {
        "name": "no_tangent_novelty",
        "record_suffix": "no_tangent_novelty",
        "static_route_id": "unspecified",
        "note": "Disable Phase-III tangent/collective novelty only.",
        "row_updates": {"phase3_novelty_ablation_mode": "all"},
        "override_updates": {"phase3_novelty_ablation_mode": "all"},
    },
    {
        "name": "no_phase3_schur_rerank",
        "record_suffix": "no_phase3_schur_rerank",
        "static_route_id": "unspecified",
        "note": "Disable reduced Schur/rerank geometry only by using raw exact geometry.",
        "row_updates": {"phase3_selector_geometry_mode": "raw_exact"},
        "override_updates": {"phase3_selector_geometry_mode": "raw_exact"},
    },
    {
        "name": "no_active_local_window_refit",
        "record_suffix": "no_active_local_window_refit",
        "static_route_id": "unspecified",
        "note": "Disable active local window relaxation only.",
        "row_updates": {"phase3_window_relaxation_mode": "no_relaxation"},
        "override_updates": {"phase3_window_relaxation_mode": "no_relaxation"},
    },
    {
        "name": "no_generator_ablation",
        "record_suffix": "no_generator_ablation",
        "static_route_id": "unspecified",
        "note": "Disable recoverability generator pruning/ablation only.",
        "row_updates": {
            "phase1_prune_enabled": "false",
            "phase1_prune_amplitude_witness_required": "false",
        },
        "override_updates": {
            "phase1_prune_enabled": False,
            "phase1_prune_amplitude_witness_required": False,
        },
    },
    {
        "name": "no_batching",
        "record_suffix": "no_batching",
        "static_route_id": "unspecified",
        "note": "Disable Phase-II/III batching only.",
        "row_updates": {
            "phase2_enable_batching": "false",
            "phase3_enable_batching": "false",
        },
        "override_updates": {"phase2_enable_batching": False},
    },
    {
        "name": "no_beam_continuation",
        "record_suffix": "no_beam_continuation",
        "static_route_id": "unspecified",
        "note": "Disable beam continuation only by forcing single live/child/kept branch.",
        "row_updates": {},
        "override_updates": {
            "adapt_beam_live_branches": 1,
            "adapt_beam_children_per_parent": 1,
            "adapt_beam_terminated_keep": 1,
            "phase3_tie_beam_max_branches": 1,
        },
    },
    {
        "name": "append_only_adapt_limit",
        "record_suffix": "append_only_adapt_limit",
        "static_route_id": "unspecified",
        "note": "Disable adaptive insertion/refit/rollback only; keep novelty, batching, pruning, and beam unchanged.",
        "row_updates": {},
        "override_updates": {
            "adapt_reopt_policy": "append_only",
            "adapt_insertion_mode": "append_only",
            "adapt_full_refit_every": 0,
            "adapt_final_full_refit": False,
            "adapt_window_size": 1,
            "adapt_window_topk": 0,
        },
    },
)


def _repo_rel(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve()))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_source_row() -> tuple[dict[str, str], list[str]]:
    with SOURCE_RECORDS_TSV.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or ())
    matches = [row for row in rows if row.get("record_id") == SOURCE_RECORD_ID]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one source row {SOURCE_RECORD_ID!r}; found {len(matches)}")
    row = matches[0]
    mismatches = {
        key: {"expected": expected, "actual": row.get(key, "")}
        for key, expected in SOURCE_EXPECTED.items()
        if str(row.get(key, "")) != str(expected)
    }
    if mismatches:
        raise ValueError(f"source row no longer matches strong-Hubbard contract: {mismatches}")
    return row, fieldnames


def _load_source_trial_overrides() -> dict[str, Any]:
    payload = json.loads(SOURCE_EFFECTIVE_MANIFEST_JSON.read_text(encoding="utf-8"))
    sampled = payload.get("sampled_params")
    if not isinstance(sampled, Mapping):
        raise ValueError(f"source effective manifest has no sampled_params: {SOURCE_EFFECTIVE_MANIFEST_JSON}")
    overrides = dict(SOURCE_COMMAND_COMPAT_OVERRIDES)
    # Static trial overrides intentionally carry only compatibility pins. The
    # source sampled Optuna parameters are replayed via the copied summary JSON.
    return overrides


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_override_json(name: str, note: str, source_overrides: Mapping[str, Any], updates: Mapping[str, Any]) -> str:
    merged = dict(source_overrides)
    merged.update(dict(updates))
    path = INPUT_DIR / f"{name}_trial_param_overrides.json"
    _write_json(
        path,
        {
            "schema": "phase3_trial_param_overrides_v1",
            "purpose": "Strong-Hubbard SNAKE source-policy ablation anchored to selected-logical v4 trial 0; no recalibration.",
            "ablation_label": name,
            "ablation_note": note,
            "source_effective_manifest_json": _repo_rel(SOURCE_EFFECTIVE_MANIFEST_JSON),
            "source_result_json": _repo_rel(SOURCE_RESULT_JSON),
            "trial_param_overrides": merged,
        },
    )
    return _repo_rel(path)


def _copy_replay_summary() -> dict[str, Any]:
    payload = json.loads(SOURCE_SUMMARY_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"source summary is not a JSON object: {SOURCE_SUMMARY_JSON}")
    summary = deepcopy(payload)
    hh = summary.get("summaries", {}).get("hubbard_L2_three_model_strong", {})
    if not isinstance(hh, dict):
        raise ValueError("source summary missing hubbard_L2_three_model_strong entry")
    if hh.get("best_trial_number") != 0:
        raise ValueError(f"expected visible source best_trial_number=0, got {hh.get('best_trial_number')!r}")
    summary["schema"] = "phase3_oracle_summary_for_fixed_strong_hubbard_ablation_v1"
    summary["generated_utc"] = datetime.now(timezone.utc).isoformat()
    summary["purpose"] = "Fixed-policy replay seed for strong-Hubbard SNAKE ablations; source trial 0 only."
    summary["ablation_batch_id"] = BATCH_ID
    summary["source"] = {
        "record_id": SOURCE_RECORD_ID,
        "summary_json": _repo_rel(SOURCE_SUMMARY_JSON),
        "summary_sha256": _sha256(SOURCE_SUMMARY_JSON),
        "effective_manifest_json": _repo_rel(SOURCE_EFFECTIVE_MANIFEST_JSON),
        "effective_manifest_sha256": _sha256(SOURCE_EFFECTIVE_MANIFEST_JSON),
        "result_json": _repo_rel(SOURCE_RESULT_JSON),
        "result_sha256": _sha256(SOURCE_RESULT_JSON),
        "best_trial_number": hh.get("best_trial_number"),
        "best_value": hh.get("best_value"),
    }
    _write_json(SUMMARY_JSON, summary)
    return summary


def _record_for_variant(source_row: Mapping[str, str], variant: Mapping[str, Any], override_path: str) -> dict[str, str]:
    row = {key: str(value or "") for key, value in source_row.items()}
    row.update(BASE_RECORD_OVERLAY)
    row.update({key: str(value) for key, value in SOURCE_EXPECTED.items()})
    row["record_id"] = f"{BATCH_ID}_{variant['record_suffix']}"
    row["static_route_id"] = str(variant["static_route_id"])
    row["trial_param_overrides_json"] = override_path
    row["algorithm_variant"] = f"paper_i_hubbard_strong_fixed_policy_ablation_{variant['record_suffix']}"
    for key, value in dict(variant.get("row_updates", {})).items():
        row[str(key)] = str(value).lower() if isinstance(value, bool) else str(value)
    return row


def _validate_records(rows: list[Mapping[str, str]], summary: Mapping[str, Any]) -> dict[str, Any]:
    record_ids = [row["record_id"] for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate record_id in generated ablation records")
    best_params = (
        summary.get("summaries", {})
        .get("hubbard_L2_three_model_strong", {})
        .get("best_params", {})
    )
    if not isinstance(best_params, Mapping):
        raise ValueError("replay summary missing best_params mapping")
    full = rows[0]
    for key, expected in SOURCE_EXPECTED.items():
        if full.get(key) != expected:
            raise ValueError(f"full anchor row drifted on {key}: {full.get(key)!r} != {expected!r}")
    append = next(row for row in rows if row["record_id"].endswith("append_only_adapt_limit"))
    for key in (
        "phase3_novelty_ablation_mode",
        "phase3_selector_geometry_mode",
        "phase3_window_relaxation_mode",
        "phase1_prune_enabled",
        "phase1_prune_amplitude_witness_required",
    ):
        if append.get(key) != SOURCE_EXPECTED[key]:
            raise ValueError(f"append-only row changed non-append mechanism {key}: {append.get(key)}")
    return {
        "record_count": len(rows),
        "smoke_record_id": rows[0]["record_id"],
        "ablation_record_ids": record_ids[1:],
        "source_best_params_key_count": len(best_params),
        "source_trial_notes": {
            "baseline_feature_phase3_batching_enabled": best_params.get("feature_phase3_batching_enabled"),
            "baseline_adapt_beam_live_branches": best_params.get("adapt_beam_live_branches"),
            "baseline_adapt_beam_children_per_parent": best_params.get("adapt_beam_children_per_parent"),
        },
    }


def _write_submit(path: Path, *, record_ids_path: Path, job_batch_name: str, request_cpus: int = 10) -> None:
    content = f"""universe = vanilla
executable = chtc/phase3_optuna/run_task_apptainer.sh
arguments = $(record_id)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna
transfer_output_files = raw_outputs, logs
log = logs/{job_batch_name}.$(Cluster).$(Process).log
output = logs/{job_batch_name}.$(Cluster).$(Process).out
error = logs/{job_batch_name}.$(Cluster).$(Process).err
stream_output = False
stream_error = False
requirements = TARGET.HasSIF
request_cpus = {int(request_cpus)}
request_memory = 32GB
request_disk = 122880MB
+MaxRuntime = 172800
+JobBatchName = "holstein-{job_batch_name}"
environment = "PHASE3_RECORDS_PATH={_repo_rel(RECORDS_TSV)} PHASE3_TERMINATE_ON_STALE_PROGRESS=1 PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC=3600 PHASE3_PROGRESS_STALE_AFTER_SEC=3600 PHASE3_HEARTBEAT_INTERVAL_SEC=60 PHASE3_SHELL_HEARTBEAT_SEC=60"
queue record_id from {_repo_rel(record_ids_path)}
"""
    path.write_text(content, encoding="utf-8")


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_row, fieldnames = _load_source_row()
    source_overrides = _load_source_trial_overrides()
    summary = _copy_replay_summary()

    rows: list[dict[str, str]] = []
    override_paths: dict[str, str] = {}
    for variant in VARIANTS:
        override_path = _write_override_json(
            str(variant["name"]),
            str(variant["note"]),
            source_overrides,
            dict(variant.get("override_updates", {})),
        )
        override_paths[str(variant["name"])] = override_path
        rows.append(_record_for_variant(source_row, variant, override_path))

    for extra in sorted({key for row in rows for key in row} - set(fieldnames)):
        fieldnames.append(extra)
    with RECORDS_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    ALL_RECORD_IDS.write_text("".join(f"{row['record_id']}\n" for row in rows), encoding="utf-8")
    SMOKE_RECORD_IDS.write_text(f"{rows[0]['record_id']}\n", encoding="utf-8")
    ABLATION_ONLY_RECORD_IDS.write_text("".join(f"{row['record_id']}\n" for row in rows[1:]), encoding="utf-8")
    _write_submit(SUBMIT_ALL, record_ids_path=ALL_RECORD_IDS, job_batch_name=BATCH_ID)
    _write_submit(SUBMIT_SMOKE, record_ids_path=SMOKE_RECORD_IDS, job_batch_name=f"{BATCH_ID}_smoke")
    _write_submit(
        SUBMIT_ABLATION_ONLY,
        record_ids_path=ABLATION_ONLY_RECORD_IDS,
        job_batch_name=f"{BATCH_ID}_ablation_only",
    )

    validation = _validate_records(rows, summary)
    manifest = {
        "schema": "paper_i_hubbard_strong_fixed_ablation_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "table_label": "tab:fixed_accuracy_claims",
        "regime_or_case": "pure Hubbard strong, U/t=1.5",
        "method": "SNAKE",
        "visible_value": {
            "delta_e": 0.0,
            "compiled_two_qubit_count": 56,
            "compiled_two_qubit_depth": 52,
            "compiled_depth": 219,
            "shot_proxy": 350,
        },
        "records_tsv": _repo_rel(RECORDS_TSV),
        "all_record_ids": _repo_rel(ALL_RECORD_IDS),
        "smoke_record_ids": _repo_rel(SMOKE_RECORD_IDS),
        "ablation_only_record_ids": _repo_rel(ABLATION_ONLY_RECORD_IDS),
        "submit_all": _repo_rel(SUBMIT_ALL),
        "submit_smoke": _repo_rel(SUBMIT_SMOKE),
        "submit_ablation_only": _repo_rel(SUBMIT_ABLATION_ONLY),
        "source_record_id": SOURCE_RECORD_ID,
        "source_records_tsv": _repo_rel(SOURCE_RECORDS_TSV),
        "source_summary_json": _repo_rel(SOURCE_SUMMARY_JSON),
        "source_effective_manifest_json": _repo_rel(SOURCE_EFFECTIVE_MANIFEST_JSON),
        "source_result_json": _repo_rel(SOURCE_RESULT_JSON),
        "source_result_sha256": _sha256(SOURCE_RESULT_JSON),
        "replay_summary_json": _repo_rel(SUMMARY_JSON),
        "validation": validation,
        "override_paths": override_paths,
        "settings_reused": {
            "source_best_trial_number": 0,
            "suite_profile": SOURCE_EXPECTED["suite_profile"],
            "benchmark_ids": SOURCE_EXPECTED["benchmark_ids"],
            "selected_logical_route": SOURCE_EXPECTED["selected_logical_route"],
            "meta_feature_profile": SOURCE_EXPECTED["meta_feature_profile"],
            "phase3_selector_policy": SOURCE_EXPECTED["phase3_selector_policy"],
            "phase3_selector_geometry_mode": SOURCE_EXPECTED["phase3_selector_geometry_mode"],
            "phase3_window_relaxation_mode": SOURCE_EXPECTED["phase3_window_relaxation_mode"],
            "phase1_prune_enabled": SOURCE_EXPECTED["phase1_prune_enabled"],
            "primary_selector_score_key": SOURCE_EXPECTED["primary_selector_score_key"],
            "n_trials": 1,
            "no_optuna_recalibration": True,
        },
        "settings_changed": {
            row["record_id"]: next(v["note"] for v in VARIANTS if row["record_id"].endswith(str(v["record_suffix"])))
            for row in rows
        },
        "run_gate": {
            "submit_order": ["baseline_smoke_only", "ablation_rows_after_baseline_reproduces_source"],
            "baseline_expected_energy": -1.3860009363293697,
            "baseline_expected_primary_error": 1.2878587085651816e-14,
            "baseline_expected_operator_count": 3,
            "no_matrix_before_smoke": True,
        },
    }
    _write_json(MANIFEST_JSON, manifest)
    print(json.dumps({"ok": True, **manifest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
