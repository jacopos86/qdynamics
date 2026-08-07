#!/usr/bin/env python3
"""Generate the current canonical Paper-I HH SNAKE iteration-50 batch.

The six fresh candidate rows start from the current visible physical-operator-
lane source commands, then apply the approved forward SNAKE controls.  This is
not a continuation and not a one-variable sensitivity sweep.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_BATCH_ID = "paper_i_hh_fullmeta_singleton_symmetry_current_snake_iter50_20260710_v1"
SOURCE_RECORDS = (
    ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_all_regime_snake_mechanism_ablation_20260709_v1"
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)
DEFAULT_OUTPUT_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / DEFAULT_BATCH_ID

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
IMPLEMENTATION_FILES = (
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/cli_config.py",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_task_apptainer.sh",
    "chtc/phase3_optuna/preflight_submit.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _git_head() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _arg_value(args: Sequence[str], flag: str) -> str | None:
    try:
        index = list(args).index(flag)
    except ValueError:
        return None
    return str(args[index + 1]) if index + 1 < len(args) else None


def _load_templates() -> tuple[list[str], dict[str, dict[str, str]]]:
    with SOURCE_RECORDS.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        rows = [
            {str(key): "" if value is None else str(value) for key, value in row.items()}
            for row in reader
            if str(row.get("hh_mechanism_ablation_variant") or "") == "no_batching_reference"
        ]
    by_regime = {row["display_regime"]: row for row in rows}
    if tuple(regime for regime in REGIME_ORDER if regime in by_regime) != REGIME_ORDER or len(by_regime) != 6:
        raise ValueError(f"Expected one no-batching source row for each regime; got {sorted(by_regime)}")
    return fieldnames, by_regime


def _validate_source(row: Mapping[str, str], regime: str) -> None:
    expected = {
        "display_regime": regime,
        "method_key": "snake",
        "algorithm_id": "static_family_native_adapt_phase3",
        "adapt_optimizer_kind": "powell",
        "budget": "200",
        "pool_contract": "full_meta_unfiltered",
        "hh_adaptive_pool_profile": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
        "snake_phase3_runtime_split_mode": "shortlist_pauli_children_v1",
        "snake_phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
        "snake_phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "snake_phase3_runtime_split_max_subset_size": "1",
        "shared_pauli_pool_mode": "off",
    }
    mismatches = [
        f"{key}={row.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if str(row.get(key) or "") != value
    ]
    source_json = ROOT / str(row.get("source_json") or "")
    if not source_json.is_file():
        mismatches.append(f"source_json missing: {source_json}")
    elif _sha256(source_json) != str(row.get("source_json_sha256") or ""):
        mismatches.append(f"source_json SHA mismatch: {source_json}")
    source_args = [str(value) for value in json.loads(str(row.get("source_command_args_json") or "[]"))]
    if _arg_value(source_args, "--static-route-id") != "route_a":
        mismatches.append("source command is not route_a")
    if _arg_value(source_args, "--static-lane-route") != "physical_operator_type":
        mismatches.append("source command is not physical_operator_type")
    if mismatches:
        raise ValueError(f"Source contract mismatch for {regime}: {'; '.join(mismatches)}")


def _overrides(record_id: str) -> dict[str, Any]:
    return {
        "set_flags": {
            "--adapt-segment-id": record_id,
            "--adapt-reopt-policy": "full",
            "--adapt-window-size": "99",
            "--adapt-window-topk": "0",
            "--adapt-full-refit-every": "1",
            "--adapt-final-full-refit": "true",
            "--adapt-insertion-mode": "full_commutation_reduced",
            "--adapt-beam-live-branches": "3",
            "--adapt-beam-children-per-parent": "2",
            "--adapt-beam-lambda": "0.005",
            "--phase1-probe-max-positions": "999999",
            "--phase1-trough-margin-ratio": "1.0",
            "--phase1-lambda-theta": "0.001",
            "--phase1-prune-schur-nomination-route": "metric_regularized_v1",
            "--phase1-prune-metric-schur-mu": "0.01",
            "--phase1-prune-metric-schur-cost-weighting": "ansatz_entry_denominator_v1",
            "--phase3-geometry-window-size": "99",
            "--phase3-novelty-ablation-mode": "off",
            "--phase3-runtime-split-max-subset-size": "1",
            "--phase1-maturity-cap-min": "999999",
            "--phase1-maturity-cap-max": "999999",
            "--phase2-maturity-cap-min": "999999",
            "--phase2-maturity-cap-max": "999999",
            "--phase3-maturity-cap-min": "999999",
            "--phase3-maturity-cap-max": "999999",
            "--phase-maturity-shot-min": "1",
            "--phase-maturity-shot-max": "1",
            "--phase1-maturity-shot-cap": "1",
            "--phase2-maturity-shot-cap": "1",
            "--phase3-maturity-shot-cap": "1",
            "--static-route-id": "route_a",
            "--static-lane-route": "physical_operator_type",
            "--physical-lane-shortlist-aggressiveness": "3",
        },
        "enable_flags": [
            "--phase-live-hysteresis-disabled",
            "--phase2-no-batching",
            "--phase3-no-batching",
        ],
        "remove_bool_flags": [
            "--phase-live-hysteresis-enabled",
            "--phase2-enable-batching",
            "--phase3-enable-batching",
        ],
        "remove_value_flags": [
            "--phase3-source-lock-preferred-sequence",
            "--phase2-batch-selection-mode",
            "--phase2-batch-target-size",
            "--phase2-batch-size-cap",
            "--phase3-batch-selection-mode",
            "--phase3-batch-target-size",
            "--phase3-batch-size-cap",
            "--phase2-null-nrem-high-threshold",
            "--phase2-live-nrem-low-threshold",
            "--phase3-null-nrem-high-threshold",
            "--phase3-live-nrem-low-threshold",
            "--phase2-hysteresis-steps",
            "--phase3-hysteresis-steps",
        ],
    }


def _effective_contract(effective: Sequence[str], record_id: str) -> None:
    expected_values = {
        "--adapt-max-depth": "50",
        "--adapt-segment-target-depth": "50",
        "--adapt-segment-max-new-admissions": "50",
        "--adapt-maxiter": "200",
        "--adapt-final-refit-maxiter": "200",
        "--adapt-inner-optimizer": "POWELL",
        "--adapt-reopt-policy": "full",
        "--adapt-insertion-mode": "full_commutation_reduced",
        "--adapt-window-size": "99",
        "--adapt-window-topk": "0",
        "--phase3-geometry-window-size": "99",
        "--phase3-runtime-split-mode": "shortlist_pauli_children_v1",
        "--phase3-runtime-split-selection-mode": "archival_child_set_forward_v1",
        "--phase3-runtime-split-child-set-symmetry-policy": "hard_guard",
        "--phase3-runtime-split-max-subset-size": "1",
        "--static-route-id": "route_a",
    }
    mismatches = [
        f"{flag}={_arg_value(effective, flag)!r}, expected {value!r}"
        for flag, value in expected_values.items()
        if _arg_value(effective, flag) != value
    ]
    required_bool = {
        "--allow-archival-phase3-runtime-split",
        "--phase-live-hysteresis-disabled",
        "--phase2-no-batching",
        "--phase3-no-batching",
    }
    missing = sorted(required_bool.difference(effective))
    forbidden = sorted(
        flag
        for flag in (
            "--phase3-source-lock-preferred-sequence",
            "--phase2-enable-batching",
            "--phase3-enable-batching",
            "--phase-live-hysteresis-enabled",
        )
        if flag in effective
    )
    if mismatches or missing or forbidden:
        raise ValueError(
            f"Effective command contract failed for {record_id}: "
            f"mismatches={mismatches}, missing={missing}, forbidden={forbidden}"
        )


def generate(batch_id: str = DEFAULT_BATCH_ID, output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    from chtc.phase3_optuna import run_paper_i_hh_spsa_budget_ladder_cell as runner

    fieldnames, templates = _load_templates()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    command_audits: list[dict[str, Any]] = []
    for regime in REGIME_ORDER:
        template = templates[regime]
        _validate_source(template, regime)
        internal = str(template["internal_regime"])
        record_id = f"{batch_id}__{internal}__snake__current_forward__powell200__iter50"
        record_output_dir = f"raw_outputs/{batch_id}/{record_id}"
        overrides = _overrides(record_id)
        row = dict(template)
        row.update(
            {
                "record_id": record_id,
                "batch_id": batch_id,
                "run_class": "candidate",
                "runnable": "true",
                "blocker": "",
                "blocked_reason": "",
                "budget": "200",
                "max_depth": "50",
                "adapt_optimizer_kind": "powell",
                "optimizer": "POWELL",
                "optimizer_profile": "powell_maxiter200_final_refit200",
                "optimizer_overlay_id": "powell",
                "optimizer_contract_id": "powell_maxiter200_final_refit200_depth50_v1",
                "engine_label": "current canonical Paper-I HH SNAKE Powell200 iteration-50 candidate",
                "matrix_label": "A_native_staged_singleton_hard_guard",
                "matrix_role": "current_forward_canonical_snake_iter50",
                "child_policy": "native_phase3_singleton",
                "symmetry_policy": "hard_guard",
                "pool_contract": "full_meta_unfiltered",
                "hh_adaptive_pool_profile": "full_meta_unfiltered",
                "adapt_pool_class_filter_json": "off",
                "adapt_schur_warm_start_mode": "append-prune",
                "snake_phase3_runtime_split_mode": "shortlist_pauli_children_v1",
                "snake_phase3_runtime_split_selection_mode": "archival_child_set_forward_v1",
                "snake_phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
                "snake_phase3_runtime_split_max_subset_size": "1",
                "shared_pauli_pool_mode": "off",
                "shared_pauli_pool_symmetry_policy": "",
                "shared_pauli_pool_max_subset_size": "",
                "child_subset_size": "1",
                "static_route_id": "route_a",
                "static_lane_route": "physical_operator_type",
                "physical_lane_shortlist_aggressiveness": "3",
                "generic_adapt_stop_policy": "",
                "changed_fields_vs_anchor": (
                    "forward_canonical_controls;max_depth:30->50;output_identity:new;"
                    "disable_drop_stop;disable_benchmark_target_stop;suppress_drop_plateau_terminal_stop"
                ),
                "source_settings_status": "visible_row_plus_approved_forward_canonical_controls",
                "schedule_source_policy": "visible_physical_lane_source_plus_forward_canonical_controls",
                "schedule_source_note": (
                    "Fresh current SNAKE run to iteration 50; no operators, theta, history, or query ledger are resumed. "
                    "The visible source command supplies the regime physics and pool provenance."
                ),
                "source_contract_note": (
                    "Fresh six-regime canonical SNAKE candidate batch. Unfiltered full_meta with HVA included; "
                    "native archival Pauli-child cap 1 with hard guard; physical operator lanes; no batching; "
                    "Powell maxiter/final-refit maxiter 200; fixed iteration-50 horizon."
                ),
                "provenance_layer": "visible_row_plus_forward_canonical_controls",
                "record_output_dir": record_output_dir,
                "result_json_rel": f"{record_output_dir}/json/result.json",
                "current_json_rel": f"{record_output_dir}/current.json",
                "snake_algorithmic_work_rel": f"{record_output_dir}/snake_algorithmic_work.json",
                "source_lock_command_audit_rel": f"{record_output_dir}/source_lock_command_audit.json",
                "stdout_rel": f"{record_output_dir}/stdout.log",
                "stderr_rel": f"{record_output_dir}/stderr.log",
                "cell_manifest_rel": f"{record_output_dir}/cell_manifest.json",
                "request_memory_mb": "32768",
                "request_disk_mb": "61440",
                "resource_tier": "standard",
                "ordered_batch_beam_enabled": "false",
                "ordered_batch_beam_label": "current_forward_no_batching",
                "ordered_batch_beam_run_role": "canonical_no_batching",
                "phase2_batch_selection_mode": "",
                "phase2_batch_target_size": "",
                "phase2_batch_size_cap": "",
                "phase3_batch_selection_mode": "",
                "phase3_batch_target_size": "",
                "phase3_batch_size_cap": "",
                "route_variant": "current_forward_no_batching",
                "batch_variant_gate": "canonical_candidate_batch",
                "snake_cli_overrides_json": _json(overrides),
                "settings_reused_json": _json(
                    {
                        "source_json": template["source_json"],
                        "source_json_sha256": template["source_json_sha256"],
                        "physics_and_cutoffs": "source_command",
                        "pool": "full_meta_unfiltered_hva_included",
                        "optimizer": "POWELL",
                    }
                ),
                "settings_changed_json": _json(overrides),
                "settings_change_reason": "approved_current_forward_canonical_snake_controls_and_iteration50_horizon",
                "hh_mechanism_ablation_variant": "",
                "hh_mechanism_ablation_feature": "",
                "hh_mechanism_ablation_role": "",
                "hh_mechanism_ablation_submit_group": "",
                "hh_mechanism_ablation_expected_status": "",
                "hh_mechanism_ablation_overrides_json": "",
                "hh_mechanism_ablation_plan_md": "",
            }
        )
        source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(
            row,
            ROOT / record_output_dir,
        )
        del source_cmd
        if audit.get("status") != "pass":
            raise ValueError(
                f"Source-lock audit failed for {record_id}: {audit.get('non_allowed_flag_changes')}"
            )
        _effective_contract(effective_cmd, record_id)
        command_audits.append(
            {
                "record_id": record_id,
                "status": audit["status"],
                "changed_flags": audit.get("changed_flags"),
                "effective_command": effective_cmd,
            }
        )
        rows.append(row)

    if len(rows) != 6:
        raise ValueError(f"Expected six rows, generated {len(rows)}")

    records_path = output_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    extras = sorted({key for row in rows for key in row}.difference(fieldnames))
    write_fields = [*fieldnames, *extras]
    with records_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=write_fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    ids_path = output_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    ids_path.write_text("\n".join(row["record_id"] for row in rows) + "\n", encoding="utf-8")
    queue_path = output_dir / "paper_i_hh_spsa_budget_ladder_record_queue.tsv"
    queue_path.write_text(
        "".join(
            f"{row['record_id']}\t{row['request_memory_mb']}\t{row['request_disk_mb']}\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    implementation_lock = [
        {
            "path": rel,
            "sha256": _sha256(ROOT / rel),
            "size_bytes": (ROOT / rel).stat().st_size,
        }
        for rel in IMPLEMENTATION_FILES
    ]
    generated_utc = datetime.now(timezone.utc).isoformat()
    audit_payload = {
        "schema": "paper_i_hh_current_snake_iter50_submission_audit_v1",
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "run_class": "candidate",
        "paper_target": "Paper_I.pdf page 13; fig:hh_main_results_composite; SNAKE overlay",
        "source_locked_sensitivity_status": "not_applicable_canonical_candidate_batch",
        "source_locked_sensitivity_note": (
            "This batch applies approved forward canonical controls in addition to the iteration-50 horizon; "
            "it is not represented as a one-variable depth sensitivity."
        ),
        "source_records": _repo_relative(SOURCE_RECORDS),
        "source_records_sha256": _sha256(SOURCE_RECORDS),
        "source_results": [
            {
                "regime": row["display_regime"],
                "path": row["source_json"],
                "sha256": row["source_json_sha256"],
            }
            for row in rows
        ],
        "implementation_lock": implementation_lock,
        "command_audits": command_audits,
        "git_head": _git_head(),
        "git_worktree_note": "Exact implementation SHA256 values identify the submitted worktree.",
        "preserved_contract": {
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "final_refit_maxiter": 200,
            "pool": "full_meta_unfiltered",
            "hva_included": True,
            "runtime_split": "shortlist_pauli_children_v1",
            "runtime_split_selection": "archival_child_set_forward_v1",
            "runtime_split_cap": 1,
            "runtime_split_symmetry": "hard_guard",
            "shared_pauli_pool": "off",
            "static_route": "route_a",
            "static_lane": "physical_operator_type",
            "phase2_batching": "off",
            "phase3_batching": "off",
        },
        "horizon": {
            "max_depth": 50,
            "segment_target_depth": 50,
            "max_new_admissions": 50,
            "early_energy_and_drop_stops": "disabled",
        },
        "durability": {
            "condor_transfer_policy": "ON_EXIT_OR_EVICT",
            "current_json": "current.json",
            "restartable_from_iteration_checkpoint": False,
            "eviction_behavior": "partial output transfers; retry starts fresh",
        },
        "status": "pass",
    }
    audit_path = output_dir / "submission_audit.json"
    audit_path.write_text(json.dumps(audit_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {
        "schema": "paper_i_hh_current_snake_iter50_manifest_v1",
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "record_count": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "regimes": list(REGIME_ORDER),
        "optimizer": "POWELL",
        "max_depth": 50,
        "records_path": _repo_relative(records_path),
        "record_ids_path": _repo_relative(ids_path),
        "record_queue_path": _repo_relative(queue_path),
        "submission_audit": _repo_relative(audit_path),
        "status": "ready_for_preflight",
    }
    manifest_path = output_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    print(json.dumps(generate(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
