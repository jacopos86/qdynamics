#!/usr/bin/env python3
"""Generate the corrected Paper-I HH parent-comparator CHTC matrix.

This is a fresh correction batch, not a continuation and not a one-variable
sensitivity sweep.  It starts from the 12 currently visible C-macro Powell
rows, vendors their result/cell-manifest provenance, and records the approved
implementation/accounting corrections plus the selector-horizon change from
30 to 50 iterations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
REFERENCE_ROOT = Path.home() / "Documents" / "Holstein_implementation" / "Holstein_test_fullclone_3"
DEFAULT_BATCH_ID = "paper_i_hh_fullmeta_singleton_symmetry_corrected_parent_iter50_20260710_v1"
DEFAULT_SOURCE_MAP = (
    ROOT
    / "raw_outputs"
    / "paper_i_hh_six_regime_corrected_parent_comparators_powell200_depth30_local_20260710_v1"
    / "visible_source_map.json"
)
DEFAULT_SOURCE_LOCK_DIR = DEFAULT_SOURCE_MAP.parent / "source_locks"
DEFAULT_OUTPUT_DIR = ROOT / "chtc" / "phase3_optuna" / "input" / DEFAULT_BATCH_ID

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
METHODS = (
    ("Geo-ADAPT", "geo", "static_geo_adapt_vqe"),
    ("Append-ADAPT", "append", "static_full_meta_append_adapt_vqe"),
)
IMPLEMENTATION_FILES = (
    "src/quantum/compiled_ansatz.py",
    "pipelines/exact_bench/generic_static_adapt_variants.py",
    "pipelines/exact_bench/generic_static_benchmark.py",
    "pipelines/exact_bench/table_i_qiskit_resource_compile.py",
    "pipelines/exact_bench/generic_static_metric_enrichment.py",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_cell.py",
    "chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_task_apptainer.sh",
    "chtc/phase3_optuna/preflight_submit.py",
)
CORRECTED_ANCHORS = {
    "geo": (
        ROOT
        / "raw_outputs"
        / "paper_i_hh_weak_weak_parent_only_comparator_fix_powell200_depth30_local_20260710_v1"
        / "geo"
        / "result"
        / "generic_static_single.json"
    ),
    "append": (
        ROOT
        / "raw_outputs"
        / "paper_i_hh_weak_weak_parent_only_comparator_fix_powell200_depth30_local_20260710_v1"
        / "append"
        / "result"
        / "generic_static_single.json"
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def _git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def _source_cell_manifest(source_json: Path) -> Path:
    candidate = source_json.parent.parent / "cell_manifest.json"
    if candidate.is_file():
        return candidate
    try:
        repo_relative = source_json.resolve().relative_to(ROOT.resolve())
    except ValueError:
        repo_relative = None
    if repo_relative is not None:
        reference_candidate = (REFERENCE_ROOT / repo_relative).parent.parent / "cell_manifest.json"
        if reference_candidate.is_file():
            return reference_candidate
    raise FileNotFoundError(f"Visible source cell manifest is missing: {candidate}")


def _validate_template_row(
    row: Mapping[str, Any],
    *,
    regime: str,
    method_key: str,
    algorithm_id: str,
) -> None:
    expected = {
        "display_regime": regime,
        "method_key": method_key,
        "algorithm_id": algorithm_id,
        "run_class": "candidate",
        "adapt_optimizer_kind": "powell",
        "budget": "200",
        "max_depth": "30",
        "matrix_label": "C_macro_only",
        "child_policy": "macro_only",
        "pool_contract": "full_meta_unfiltered",
        "hh_adaptive_pool_profile": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
        "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
    }
    problems = [
        f"{key}={row.get(key)!r}, expected {value!r}"
        for key, value in expected.items()
        if str(row.get(key) or "") != value
    ]
    if str(row.get("shared_pauli_pool_mode") or "off") != "off":
        problems.append(f"shared_pauli_pool_mode={row.get('shared_pauli_pool_mode')!r}, expected 'off'")
    if str(row.get("generic_adapt_runtime_split_mode") or "off") != "off":
        problems.append(
            f"generic_adapt_runtime_split_mode={row.get('generic_adapt_runtime_split_mode')!r}, expected 'off'"
        )
    if problems:
        raise ValueError(f"Visible row contract mismatch for {regime}/{method_key}: {'; '.join(problems)}")


def _anchor_summary(method_key: str, source: Path, vendored: Path) -> dict[str, Any]:
    payload = _read_json(source)
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"Corrected anchor has no result object: {source}")
    if str(payload.get("status") or "") != "completed" or str(result.get("status") or "") != "ok":
        raise ValueError(f"Corrected anchor is not complete/ok: {source}")
    return {
        "method_key": method_key,
        "source_json": str(source),
        "source_sha256": _sha256(source),
        "vendored_json": _repo_relative(vendored),
        "vendored_sha256": _sha256(vendored),
        "adapt_max_iterations": result.get("adapt_max_iterations"),
        "adapt_num_iterations": result.get("adapt_num_iterations"),
        "adapt_depth_reached": result.get("adapt_depth_reached"),
        "adapt_stop_reason": result.get("adapt_stop_reason"),
        "abs_delta_e_same_cutoff": result.get("abs_delta_e_same_cutoff"),
        "S_alg": result.get("S_alg"),
        "geo_selection_with_replacement": result.get("geo_selection_with_replacement"),
        "geo_immediate_repeat_blocked": result.get("geo_immediate_repeat_blocked"),
        "selected_operator_count": result.get("selected_operator_count"),
        "selected_unique_operator_count": result.get("selected_unique_operator_count"),
    }


def generate(
    *,
    batch_id: str,
    source_map_path: Path,
    source_lock_dir: Path,
    output_dir: Path,
    max_iterations: int,
) -> dict[str, Any]:
    if max_iterations != 50:
        raise ValueError("This locked batch generator requires max_iterations=50")
    source_map = _read_json(source_map_path)
    if source_map.get("table_label") != "fig:hh_main_results_composite":
        raise ValueError(f"Unexpected visible target in {source_map_path}: {source_map.get('table_label')!r}")

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    baseline_dir = output_dir / "visible_baselines"
    lock_dir = output_dir / "visible_source_locks"
    corrected_anchor_dir = output_dir / "corrected_anchors"
    baseline_dir.mkdir()
    lock_dir.mkdir()
    corrected_anchor_dir.mkdir()

    vendored_source_map = output_dir / "visible_source_map.json"
    shutil.copy2(source_map_path, vendored_source_map)

    rows: list[dict[str, str]] = []
    baseline_records: list[dict[str, Any]] = []
    regimes_payload = source_map.get("regimes")
    if not isinstance(regimes_payload, Mapping):
        raise ValueError(f"Visible source map has no regimes object: {source_map_path}")

    for regime in REGIME_ORDER:
        regime_payload = regimes_payload.get(regime)
        if not isinstance(regime_payload, Mapping):
            raise ValueError(f"Visible source map is missing regime {regime}")
        methods_payload = regime_payload.get("methods")
        if not isinstance(methods_payload, Mapping):
            raise ValueError(f"Visible source map is missing methods for {regime}")
        for method_label, method_key, algorithm_id in METHODS:
            source_entry = methods_payload.get(method_label)
            if not isinstance(source_entry, Mapping):
                raise ValueError(f"Visible source map is missing {regime}/{method_label}")
            source_json = Path(str(source_entry.get("source_json") or "")).expanduser().resolve()
            expected_sha = str(source_entry.get("source_sha256") or "")
            if not source_json.is_file():
                raise FileNotFoundError(f"Visible source JSON is missing: {source_json}")
            actual_sha = _sha256(source_json)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Visible source SHA mismatch for {regime}/{method_key}: expected {expected_sha}, got {actual_sha}"
                )
            source_cell = _source_cell_manifest(source_json)
            source_cell_payload = _read_json(source_cell)
            template = source_cell_payload.get("row")
            if not isinstance(template, Mapping):
                raise ValueError(f"Visible source cell manifest has no row object: {source_cell}")
            _validate_template_row(template, regime=regime, method_key=method_key, algorithm_id=algorithm_id)

            slug = f"{regime.replace('-', '_')}__{method_key}"
            vendored_result = baseline_dir / f"{slug}__generic_static_single.json"
            vendored_cell = baseline_dir / f"{slug}__cell_manifest.json"
            shutil.copy2(source_json, vendored_result)
            shutil.copy2(source_cell, vendored_cell)

            lock_name = f"{regime}_{method_key}_adapt.json"
            lock_source = source_lock_dir / lock_name
            if not lock_source.is_file():
                raise FileNotFoundError(f"Visible resolver lock is missing: {lock_source}")
            lock_payload = _read_json(lock_source)
            if lock_payload.get("status") != "ok" or lock_payload.get("source_sha256_match") is not True:
                raise ValueError(f"Visible resolver lock did not pass: {lock_source}")
            lock_target = lock_dir / lock_name
            shutil.copy2(lock_source, lock_target)

            record_id = (
                f"{batch_id}__{regime.replace('-', '_')}__{method_key}__"
                "C_macro_only__powell200__iter50__fullmeta_parent"
            )
            record_output_dir = f"raw_outputs/{batch_id}/{record_id}"
            row = {str(key): "" if value is None else str(value) for key, value in template.items()}
            row.update(
                {
                    "record_id": record_id,
                    "batch_id": batch_id,
                    "run_class": "candidate",
                    "runnable": "true",
                    "blocker": "",
                    "blocked_reason": "",
                    "max_depth": "50",
                    "adapt_optimizer_kind": "powell",
                    "budget": "200",
                    "optimizer": "POWELL",
                    "optimizer_overlay_id": "powell200_iter50_v1",
                    "optimizer_contract_id": "powell_maxiter200_iter50_v1",
                    "engine_label": "corrected parent-generator Powell200 fixed 50-iteration candidate",
                    "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
                    "adapt_pool_class_filter_json": "off",
                    "pool_contract": "full_meta_unfiltered",
                    "hh_adaptive_pool_profile": "full_meta_unfiltered",
                    "matrix_label": "C_macro_only",
                    "matrix_role": "Parent macro-generator corrected rerun",
                    "child_policy": "macro_only",
                    "symmetry_policy": "not_applicable",
                    "generic_adapt_runtime_split_mode": "off",
                    "generic_adapt_runtime_split_symmetry_policy": "off",
                    "generic_adapt_runtime_split_max_subset_size": "3",
                    "shared_pauli_pool_mode": "off",
                    "shared_pauli_pool_symmetry_policy": "off",
                    "shared_pauli_pool_max_subset_size": "3",
                    "source_json": _repo_relative(vendored_result),
                    "source_json_sha256": actual_sha,
                    "anchor_source_json": _repo_relative(vendored_result),
                    "anchor_source_sha256": actual_sha,
                    "anchor_cell_manifest_rel": _repo_relative(vendored_cell),
                    "schedule_source_json": _repo_relative(vendored_result),
                    "schedule_source_policy": "visible_parent_powell_correction_rerun_iter50_v1",
                    "schedule_source_regime": regime,
                    "schedule_source_method": method_key,
                    "schedule_source_note": (
                        "Fresh corrected parent-generator run from the current visible Powell row; "
                        "approved changes are comparator implementation semantics, estimator-query accounting, "
                        "selector horizon 30-to-50, and output identity."
                    ),
                    "source_settings_status": "ok_visible_parent_source_plus_corrected_code_lock",
                    "source_contract_note": (
                        "Fresh run; no prior operators, theta, history, or query ledger are loaded. "
                        "Parent full_meta generators with HVA included; runtime split and shared Pauli pool off."
                    ),
                    "changed_fields_vs_anchor": (
                        "implementation_semantics:corrected_append_replacement_and_geo_immediate_repeat;"
                        "estimator_query_accounting:corrected;max_adapt_iterations:30->50;output_identity:new"
                    ),
                    "record_output_dir": record_output_dir,
                    "result_json_rel": f"{record_output_dir}/result/generic_static_single.json",
                    "current_json_rel": f"{record_output_dir}/adapt_iteration_progress.jsonl",
                    "stdout_rel": f"{record_output_dir}/stdout.log",
                    "stderr_rel": f"{record_output_dir}/stderr.log",
                    "cell_manifest_rel": f"{record_output_dir}/cell_manifest.json",
                    "source_lock_command_audit_rel": "",
                    "visible_source_original_path": str(source_json),
                    "visible_source_map_rel": _repo_relative(vendored_source_map),
                    "visible_source_lock_rel": _repo_relative(lock_target),
                    "correction_contract_id": "paper_i_hh_parent_comparator_corrections_20260710_v1",
                    "selector_horizon_semantics": "outer_selector_iterations_not_guaranteed_ansatz_depth",
                }
            )
            rows.append(row)
            baseline_records.append(
                {
                    "regime": regime,
                    "method": method_label,
                    "method_key": method_key,
                    "algorithm_id": algorithm_id,
                    "visible_value": source_entry.get("visible_value"),
                    "source_json_original": str(source_json),
                    "source_sha256": actual_sha,
                    "source_cell_manifest_original": str(source_cell),
                    "source_cell_manifest_sha256": _sha256(source_cell),
                    "vendored_source_json": _repo_relative(vendored_result),
                    "vendored_cell_manifest": _repo_relative(vendored_cell),
                    "vendored_source_lock": _repo_relative(lock_target),
                    "record_id": record_id,
                }
            )

    if len(rows) != 12:
        raise ValueError(f"Expected 12 rows, generated {len(rows)}")

    records_path = output_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    fieldnames = sorted({key for row in rows for key in row})
    with records_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    record_ids_path = output_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    record_ids_path.write_text("\n".join(row["record_id"] for row in rows) + "\n", encoding="utf-8")
    queue_path = output_dir / "paper_i_hh_spsa_budget_ladder_record_queue.tsv"
    queue_path.write_text(
        "".join(
            f"{row['record_id']}\t{row.get('request_memory_mb') or '16384'}\t{row.get('request_disk_mb') or '32768'}\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    corrected_anchor_summaries: list[dict[str, Any]] = []
    for method_key, source in CORRECTED_ANCHORS.items():
        if not source.is_file():
            raise FileNotFoundError(f"Corrected weak-weak anchor is missing: {source}")
        target = corrected_anchor_dir / f"weak_weak__{method_key}__generic_static_single.json"
        shutil.copy2(source, target)
        corrected_anchor_summaries.append(_anchor_summary(method_key, source, target))

    implementation_lock: list[dict[str, Any]] = []
    for rel in IMPLEMENTATION_FILES:
        path = ROOT / rel
        if not path.is_file():
            raise FileNotFoundError(f"Implementation-lock file is missing: {path}")
        implementation_lock.append({"path": rel, "sha256": _sha256(path), "size_bytes": path.stat().st_size})

    generated_utc = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema": "paper_i_hh_corrected_parent_comparator_submission_audit_v1",
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "run_class": "candidate",
        "paper_target": "Paper_I.pdf page 13; fig:hh_main_results_composite",
        "source_locked_sensitivity_status": "not_applicable_correction_batch",
        "source_locked_sensitivity_note": (
            "This batch intentionally changes implementation semantics and estimator-query accounting in addition "
            "to the selector horizon, so it is not represented as a one-variable sensitivity sweep."
        ),
        "visible_source_map": _repo_relative(vendored_source_map),
        "visible_source_map_sha256": _sha256(vendored_source_map),
        "visible_baseline_count": len(baseline_records),
        "visible_baselines": baseline_records,
        "corrected_depth30_behavior_anchors": corrected_anchor_summaries,
        "implementation_lock": implementation_lock,
        "git_head": _git_head(),
        "git_worktree_note": "Exact file SHA256 values, not git HEAD alone, identify the corrected dirty worktree.",
        "approved_changes_vs_visible_rows": [
            "append scores the full parent pool with replacement",
            "geo scores the full parent pool and suppresses only an immediate repeat append",
            "grouped-exact generator execution mode is preserved",
            "estimator-query component accounting is corrected",
            "outer selector horizon changes from 30 to 50 iterations",
            "fresh output and batch identity",
        ],
        "preserved_contract": {
            "methods": [algorithm_id for _label, _key, algorithm_id in METHODS],
            "optimizer": "POWELL",
            "optimizer_maxiter": 200,
            "seed": 42,
            "pool": "full_meta_unfiltered",
            "hva_included": True,
            "pool_exposure": "parent_macro_generators_only",
            "generic_runtime_split_mode": "off",
            "shared_pauli_pool_mode": "off",
            "stop_policy": "fixed_horizon_no_target_v1",
            "primary_metric": "same_cutoff_abs_delta_e",
            "exact_reference_usage": "reporting_only_after_optimization",
        },
        "selector_horizon": {
            "max_iterations": 50,
            "semantics": "outer selector iterations; Geo immediate-repeat skips may yield ansatz depth below 50",
        },
        "durability": {
            "condor_transfer_policy": "ON_EXIT_OR_EVICT",
            "progress_ledger": "adapt_iteration_progress.jsonl",
            "restartable_from_iteration_checkpoint": False,
            "eviction_behavior": "partial output transfers, then retry starts from iteration zero",
        },
        "status": "pass",
    }
    audit_path = output_dir / "submission_audit.json"
    _write_json(audit_path, audit)

    manifest = {
        "schema": "paper_i_hh_corrected_parent_comparator_iter50_manifest_v1",
        "generated_utc": generated_utc,
        "batch_id": batch_id,
        "run_class": "candidate",
        "paper_target": "Paper_I.pdf page 13; fig:hh_main_results_composite",
        "record_count": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "regimes": list(REGIME_ORDER),
        "methods": [algorithm_id for _label, _key, algorithm_id in METHODS],
        "max_selector_iterations": max_iterations,
        "records_path": _repo_relative(records_path),
        "record_ids_path": _repo_relative(record_ids_path),
        "record_queue_path": _repo_relative(queue_path),
        "submission_audit": _repo_relative(audit_path),
        "source_map": _repo_relative(vendored_source_map),
        "status": "ready_for_preflight",
    }
    manifest_path = output_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--source-map", type=Path, default=DEFAULT_SOURCE_MAP)
    parser.add_argument("--source-lock-dir", type=Path, default=DEFAULT_SOURCE_LOCK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-iterations", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate(
        batch_id=str(args.batch_id),
        source_map_path=Path(args.source_map).expanduser().resolve(),
        source_lock_dir=Path(args.source_lock_dir).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        max_iterations=int(args.max_iterations),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
