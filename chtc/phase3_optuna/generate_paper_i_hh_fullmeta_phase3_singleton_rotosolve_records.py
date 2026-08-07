#!/usr/bin/env python3
"""Generate Paper-I HH full-meta Phase-III singleton matrix records.

This is a narrow follow-up to the shared-pool reset diagnostics.  It keeps the
full-meta/HVA-inclusive pool, runs an explicit inner-optimizer overlay with
maxiter=200, and compares:

- SNAKE with singleton Pauli children opened in the Phase-III archival split;
- Geo-ADAPT with singleton Pauli children opened through the generic comparator
  runtime-split route.

The generator is intentionally separate from the older shared-pool generator so
the full-meta Phase-III singleton contract cannot be confused with the
full-meta-minus-HVA/shared-pool contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_BATCH_ID = "paper_i_hh_fullmeta_phase3_singleton_rotosolve_20260629_v1"
DEFAULT_BUDGET = 200
DEFAULT_REQUEST_CPUS = 4
DEFAULT_REQUEST_MEMORY_MB = 24576
DEFAULT_REQUEST_DISK_MB = 40960
DEFAULT_HIGH_MEMORY_MB = 49152
DEFAULT_HIGH_MEMORY_DISK_MB = 61440
DEFAULT_MAX_RUNTIME_S = 172800
SRC_SANITIZED_TARBALL_NAME = "src_sanitized.tar.gz"
# Keep this generator self-contained.  It used to import optional transfer
# inputs from the native200 generator, but that import can block unrelated
# recovery/candidate record generation.  Required inputs are listed explicitly
# in _write_matrix_submit_file; these extras are optional only.
EXTERNAL_TRANSFER_INPUTS: tuple[Path, ...] = ()

SOURCE_BATCH_ID = "paper_i_hh_native_forced_child_matrix_depth30_20260623_v1"
SOURCE_RECORDS_TSV = (
    ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / SOURCE_BATCH_ID
    / "paper_i_hh_spsa_budget_ladder_records.tsv"
)

FIELDNAMES = (
    "record_id",
    "batch_id",
    "run_class",
    "runnable",
    "blocker",
    "method_key",
    "method_label",
    "algorithm_id",
    "engine_key",
    "engine_label",
    "spsa_refit_engine",
    "budget",
    "display_regime",
    "internal_regime",
    "source_map_regime",
    "suite_profile",
    "case_id",
    "family",
    "n_ph_work",
    "n_ph_ref",
    "same_cutoff_exact_gs_energy",
    "same_cutoff_energy_key_hash",
    "exact_reference_energy",
    "exact_reference_energy_key_hash",
    "exact_reference_n_ph_max",
    "primary_energy_metric",
    "same_cutoff_error_role",
    "target_abs_delta_e",
    "max_depth",
    "adapt_optimizer_kind",
    "adapt_spsa_a",
    "adapt_spsa_c",
    "adapt_spsa_alpha",
    "adapt_spsa_gamma",
    "adapt_spsa_big_a",
    "adapt_spsa_seed",
    "adapt_spsa_maxiter",
    "optimizer_profile",
    "generic_adapt_runtime_split_mode",
    "generic_adapt_runtime_split_symmetry_policy",
    "generic_adapt_runtime_split_max_subset_size",
    "generic_adapt_stop_policy",
    "shared_pauli_pool_mode",
    "shared_pauli_pool_symmetry_policy",
    "shared_pauli_pool_max_subset_size",
    "adapt_pool_class_filter_json",
    "resource_qubit_cap",
    "resource_pool_term_cap",
    "adapt_schur_warm_start_mode",
    "snake_phase3_runtime_split_mode",
    "snake_phase3_runtime_split_selection_mode",
    "snake_phase3_runtime_split_child_set_symmetry_policy",
    "snake_phase3_runtime_split_max_subset_size",
    "source_json",
    "source_json_sha256",
    "source_command_sh",
    "source_command_sha256",
    "source_command_args_json",
    "source_rank",
    "source_trial",
    "source_settings_status",
    "schedule_source_policy",
    "schedule_source_regime",
    "schedule_source_method",
    "schedule_source_json",
    "schedule_source_note",
    "record_output_dir",
    "result_json_rel",
    "current_json_rel",
    "snake_algorithmic_work_rel",
    "source_lock_command_audit_rel",
    "stdout_rel",
    "stderr_rel",
    "cell_manifest_rel",
    "anchor_source_json",
    "anchor_source_sha256",
    "anchor_cell_manifest_rel",
    "changed_fields_vs_anchor",
    "source_contract_note",
)
SCHEDULE_FIELDS = (
    "adapt_spsa_a",
    "adapt_spsa_c",
    "adapt_spsa_alpha",
    "adapt_spsa_gamma",
    "adapt_spsa_big_a",
    "adapt_spsa_seed",
    "adapt_spsa_maxiter",
)
_CURRENT_BATCH_ID = DEFAULT_BATCH_ID


def configure_batch(batch_id: str) -> None:
    global _CURRENT_BATCH_ID
    _CURRENT_BATCH_ID = str(batch_id)


def rel_or_abs(path: str | Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def output_paths(record_id: str, method_key: str) -> dict[str, str]:
    del method_key
    root = Path("raw_outputs") / _CURRENT_BATCH_ID / record_id
    return {
        "record_output_dir": str(root),
        "result_json_rel": str(root / "json" / "result.json"),
        "current_json_rel": str(root / "current.json"),
        "snake_algorithmic_work_rel": str(root / "snake_algorithmic_work.json"),
        "source_lock_command_audit_rel": str(root / "source_lock_command_audit.json"),
        "stdout_rel": str(root / "stdout.log"),
        "stderr_rel": str(root / "stderr.log"),
        "cell_manifest_rel": str(root / "cell_manifest.json"),
    }

REGIME_ORDER = (
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
)
METHOD_ORDER = ("snake", "geo", "append")
ALL_METHODS = ("snake", "geo", "append")

PAULI_CHILD_MODE = "shortlist_pauli_children_v1"
SNAKE_RUNTIME_SELECTION_MODE = "archival_child_set_forward_v1"
SINGLETON_SUBSET_SIZE = "1"
SNAKE_SCHUR_WARM_START_MODE = "append-prune"
SHARED_PAULI_POOL_MODE = "shared_pauli_child_sets_v1"
SOURCE_ROW_SPSA_SCHEDULE_POLICY = "source_row_regime_specific_user_approved"
SNAKE_SPSA_SCHEDULE_POLICY = "paper_i_hh_snake_source_command_args_v1"
NON_SPSA_SCHEDULE_POLICY = "cleared_non_spsa_inner_optimizer"
SOURCE_ROW_SPSA_ALLOWED_NON_SNAKE_LABELS = frozenset(
    {
        "A_native_staged_singleton_hard_guard",
        "A_native_staged_singleton_no_guard",
        "B_common_phase0_singleton_hard_guard",
        "B_common_phase0_singleton_no_guard",
        "C_macro_only",
    }
)
PHASE3_TRUE_NO_GUARD_LABEL = "A_native_staged_singleton_true_no_guard"

MATRIX_FIELDS = (
    "matrix_label",
    "matrix_role",
    "static_route_id",
    "pool_contract",
    "hh_adaptive_pool_profile",
    "child_policy",
    "symmetry_policy",
    "optimizer",
    "optimizer_overlay_id",
    "optimizer_contract_id",
    "child_subset_size",
    "resource_tier",
    "request_memory_mb",
    "request_disk_mb",
    "spsa_schedule_policy",
    "blocked_reason",
    "snake_cli_overrides_json",
    "resume_scaffold_repair_json",
    "resume_scaffold_repair_status",
    "ordered_batch_beam_label",
    "ordered_batch_beam_enabled",
    "ordered_batch_beam_run_role",
    "phase2_batch_selection_mode",
    "phase2_batch_target_size",
    "phase2_batch_size_cap",
    "adapt_beam_live_branches",
    "adapt_beam_children_per_parent",
    "adapt_beam_lambda",
    "ordered_batch_beam_expected_diagnostics_json",
    "provenance_layer",
    "visible_support_csv",
    "visible_anchor_result_json",
    "visible_effective_command_json",
    "settings_reused_json",
    "settings_changed_json",
    "settings_change_reason",
    "route_variant",
    "anchor_gate_status",
    "batch_variant_gate",
    "work_semantics_expected_json",
    "latex_report_stem",
    "latex_report_output_dir",
    "report_update_policy",
    "regime_wave_index",
    "regime_wave_label",
    "optimizer_stage_order",
)
OUTPUT_FIELDNAMES = tuple(dict.fromkeys((*FIELDNAMES, *MATRIX_FIELDS)))

RESUME_SCAFFOLD_FLAG = "--adapt-resume-scaffold-json"
RESUME_MODE_FLAG = "--adapt-resume-mode"
RESUME_COMPILE_SMOKE_FLAG = "--adapt-resume-compile-smoke"
SEGMENT_ID_FLAG = "--adapt-segment-id"
STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR = "source_resume_repair"
STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO = "depth_zero"
STRONG_STRONG_SNAKE_START_MODES = (
    STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR,
    STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO,
)
STRONG_STRONG_RESUME_REPAIR_JSON = (
    "chtc/phase3_optuna/input/paper_i_hh_resume_scaffold_repairs/"
    "strong_strong_u8_trial_0001_current_strict_reconstructed_initial_state.json"
)


@dataclass(frozen=True)
class OptimizerOverlay:
    overlay_id: str
    optimizer_label: str
    adapt_optimizer_kind: str
    contract_prefix: str
    schedule_source_policy: str
    clears_spsa_schedule: bool = True
    spsa_schedule: Mapping[str, str] | None = None


OPTIMIZER_OVERLAYS: Mapping[str, OptimizerOverlay] = {
    "rotosolve": OptimizerOverlay(
        overlay_id="rotosolve",
        optimizer_label="ROTOSOLVE",
        adapt_optimizer_kind="rotosolve",
        contract_prefix="rotosolve",
        schedule_source_policy="candidate_rotosolve_fullmeta_singleton_symmetry_matrix",
    ),
    "powell": OptimizerOverlay(
        overlay_id="powell",
        optimizer_label="POWELL",
        adapt_optimizer_kind="powell",
        contract_prefix="powell",
        schedule_source_policy="candidate_powell_fullmeta_singleton_symmetry_matrix",
    ),
    "spsa_paper_i_hh": OptimizerOverlay(
        overlay_id="spsa_paper_i_hh",
        optimizer_label="SPSA",
        adapt_optimizer_kind="spsa",
        contract_prefix="spsa_paper_i_hh",
        schedule_source_policy="paper_i_hh_snake_source_command_args_v1",
        clears_spsa_schedule=False,
        spsa_schedule={
            "adapt_spsa_a": "0.1",
            "adapt_spsa_c": "0.02",
            "adapt_spsa_alpha": "0.602",
            "adapt_spsa_gamma": "0.101",
            "adapt_spsa_big_a": "5.0",
            "adapt_spsa_seed": "7",
            "adapt_spsa_maxiter": "200",
            "optimizer_profile": "paper_i_main_tables_spsa_v1",
        },
    ),
}


@dataclass(frozen=True)
class MatrixPolicy:
    label: str
    role: str
    child_policy: str
    symmetry_policy: str
    runnable: bool = True
    blocked_reason: str = ""


DEFAULT_MATRIX_POLICIES: tuple[MatrixPolicy, ...] = (
    MatrixPolicy(
        "A_native_staged_singleton_hard_guard",
        "Main strongest disclosed SNAKE route",
        "native_phase3_singleton",
        "hard_guard",
    ),
    MatrixPolicy(
        "A_native_staged_singleton_no_guard",
        "Symmetry ablation for main route",
        "native_phase3_singleton",
        "off",
    ),
    MatrixPolicy(
        "B_common_phase0_singleton_hard_guard",
        "Strict common-exposure fairness control",
        "common_phase0_singleton",
        "hard_guard",
    ),
    MatrixPolicy(
        "B_common_phase0_singleton_no_guard",
        "No-guard strict fairness control",
        "common_phase0_singleton",
        "off",
    ),
    MatrixPolicy(
        "C_macro_only",
        "Macro-generator control",
        "macro_only",
        "not_applicable",
    ),
)
EXTRA_MATRIX_POLICIES: tuple[MatrixPolicy, ...] = (
    MatrixPolicy(
        PHASE3_TRUE_NO_GUARD_LABEL,
        "SNAKE-only Phase-III true no-guard repair",
        "native_phase3_singleton",
        "off",
    ),
)
MATRIX_POLICIES: tuple[MatrixPolicy, ...] = (*DEFAULT_MATRIX_POLICIES, *EXTRA_MATRIX_POLICIES)

SNAKE_CANONICAL_CLI_OVERRIDES = {
    "set_flags": {
        "--phase1-lambda-theta": "0.001",
        "--phase2-rho": "0.5",
        "--phase2-w-shot": "0.05",
        "--phase3-backend-w-depth": "0.15",
        "--adapt-window-size": "50",
        "--adapt-window-topk": "50",
        "--phase3-geometry-window-size": "10",
        "--phase1-prune-fraction": "0.4",
        "--phase2-batch-near-degenerate-ratio": "0.98",
        "--phase3-batch-near-degenerate-ratio": "0.98",
        "--phase2-batch-rank-rel-tol": "0.25",
        "--phase3-batch-rank-rel-tol": "0.25",
        "--phase2-batch-additivity-tol": "0.25",
        "--phase3-batch-additivity-tol": "0.25",
        "--phase1-maturity-cap-min": "10",
        "--phase1-maturity-cap-max": "25",
        "--phase2-maturity-cap-min": "8",
        "--phase2-maturity-cap-max": "25",
        "--phase3-maturity-cap-min": "4",
        "--phase3-maturity-cap-max": "10",
        "--phase-maturity-shot-min": "1",
        "--phase-maturity-shot-max": "1",
        "--phase1-maturity-shot-cap": "1",
        "--phase2-maturity-shot-cap": "1",
        "--phase3-maturity-shot-cap": "1",
    },
    "enable_flags": ["--phase-live-hysteresis-disabled"],
    "remove_bool_flags": ["--phase-live-hysteresis-enabled"],
    "remove_value_flags": [
        "--phase1-prune-collapse-peak-abs-min",
        "--phase1-prune-collapse-current-abs-max",
        "--phase1-prune-collapse-ratio",
        "--phase1-prune-collapse-min-abs-drop",
        "--phase1-prune-collapse-min-observations",
        "--phase2-null-nrem-high-threshold",
        "--phase2-live-nrem-low-threshold",
        "--phase3-null-nrem-high-threshold",
        "--phase3-live-nrem-low-threshold",
        "--phase2-hysteresis-steps",
        "--phase3-hysteresis-steps",
    ],
}


def read_records(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            {str(k): "" if v is None else str(v) for k, v in row.items()}
            for row in csv.DictReader(fh, delimiter="\t")
        ]


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def _matches_any(value: str, selected: Sequence[str] | None) -> bool:
    return bool(selected) and value in {str(item) for item in selected}


def _is_high_memory_row(
    row: Mapping[str, str],
    *,
    high_memory_record_ids: Sequence[str] | None = None,
    high_memory_regimes: Sequence[str] | None = None,
    high_memory_methods: Sequence[str] | None = None,
    high_memory_matrix_labels: Sequence[str] | None = None,
) -> bool:
    """Return whether a row should use the explicit high-memory repair tier.

    Keep this opt-in.  The normal full matrix should not silently increase quota;
    memory-cap repairs should name the affected record ids or row selectors.
    """

    record_id = str(row.get("record_id") or "")
    if _matches_any(record_id, high_memory_record_ids):
        return True
    selectors = (
        (str(row.get("display_regime") or ""), high_memory_regimes),
        (str(row.get("method_key") or ""), high_memory_methods),
        (str(row.get("matrix_label") or ""), high_memory_matrix_labels),
    )
    active = [(value, selected) for value, selected in selectors if selected]
    return bool(active) and all(_matches_any(value, selected) for value, selected in active)


def _apply_resource_tiers(
    records: Sequence[dict[str, str]],
    *,
    request_memory_mb: int,
    request_disk_mb: int,
    high_memory_mb: int,
    high_memory_disk_mb: int,
    high_memory_record_ids: Sequence[str] | None = None,
    high_memory_regimes: Sequence[str] | None = None,
    high_memory_methods: Sequence[str] | None = None,
    high_memory_matrix_labels: Sequence[str] | None = None,
) -> tuple[list[dict[str, str]], list[str]]:
    out: list[dict[str, str]] = []
    high_memory_ids: list[str] = []
    for row in records:
        new_row = dict(row)
        high = _is_high_memory_row(
            row,
            high_memory_record_ids=high_memory_record_ids,
            high_memory_regimes=high_memory_regimes,
            high_memory_methods=high_memory_methods,
            high_memory_matrix_labels=high_memory_matrix_labels,
        )
        new_row["resource_tier"] = "high_memory" if high else "standard"
        new_row["request_memory_mb"] = str(int(high_memory_mb if high else request_memory_mb))
        new_row["request_disk_mb"] = str(int(high_memory_disk_mb if high else request_disk_mb))
        if high and str(new_row.get("runnable") or "").lower() == "true":
            high_memory_ids.append(str(new_row["record_id"]))
        out.append(new_row)
    return out, high_memory_ids


def _write_matrix_submit_file(
    *,
    batch_id: str,
    submit_path: Path,
    records_tsv: Path,
    record_queue: Path,
    request_cpus: int,
    max_runtime_s: int,
    src_sanitized_tarball: Path,
) -> None:
    """Write a matrix submit file with per-row memory/disk queue variables."""

    job_batch = "holstein-" + batch_id.replace("_", "-")
    output_root = f"raw_outputs/{batch_id}"
    logs_root = f"logs/{batch_id}"
    transfer_inputs = [
        "pipelines",
        "docs",
        "test_support",
        "chtc/phase3_optuna",
        "MATH/paper_facing/paper_I_static_scaffold/hh_ed_reference_manifest_20260614.json",
    ]
    for path in EXTERNAL_TRANSFER_INPUTS:
        if path.exists():
            transfer_inputs.append(rel_or_abs(path))
    transfer_inputs.append(rel_or_abs(src_sanitized_tarball))
    lines = [
        "universe = vanilla",
        "executable = chtc/phase3_optuna/run_paper_i_hh_spsa_budget_ladder_task_apptainer_srcpkg.sh",
        f"arguments = $(record_id) {rel_or_abs(records_tsv)} {output_root}/$(record_id)",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "transfer_executable = True",
        "preserve_relative_paths = True",
        "transfer_input_files = " + ", ".join(transfer_inputs),
        f"transfer_output_files = {output_root}, {logs_root}",
        "stream_output = False",
        "stream_error = False",
        f"log = logs/{batch_id}.$(Cluster).$(Process).log",
        f"output = logs/{batch_id}.$(Cluster).$(Process).out",
        f"error = logs/{batch_id}.$(Cluster).$(Process).err",
        "requirements = TARGET.HasSIF",
        f"request_cpus = {int(request_cpus)}",
        "request_memory = $(memory_mb)MB",
        "request_disk = $(disk_mb)MB",
        f"+MaxRuntime = {int(max_runtime_s)}",
        f'+JobBatchName = "{job_batch}"',
        f"queue record_id, memory_mb, disk_mb from {rel_or_abs(record_queue)}",
    ]
    submit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sanitized_src_tarball(path: Path) -> None:
    """Write a CHTC-safe source bundle excluding local chemistry environments."""

    if path.exists() and path.stat().st_size > 0:
        return
    for existing in sorted(
        (ROOT / "chtc" / "phase3_optuna" / "input").glob(
            "paper_i_hh_recovery_candidate_20260705_*_nobatch_wave*/src_sanitized.tar.gz"
        )
    ):
        if existing == path or not existing.exists() or existing.stat().st_size <= 0:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        try:
            os.link(existing, path)
        except OSError:
            shutil.copy2(existing, path)
        return

    src_root = ROOT / "src"
    excluded_prefix = src_root / "quantum" / "chemistry" / "conda-env"
    excluded_venv = src_root / "quantum" / "chemistry" / ".venv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as tf:
        for dirpath, dirnames, filenames in os.walk(src_root):
            current = Path(dirpath)
            pruned: list[str] = []
            for dirname in dirnames:
                child = current / dirname
                if dirname == "__pycache__":
                    continue
                try:
                    child.relative_to(excluded_prefix)
                    continue
                except ValueError:
                    pass
                try:
                    child.relative_to(excluded_venv)
                    continue
                except ValueError:
                    pass
                pruned.append(dirname)
            dirnames[:] = pruned
            if current != src_root:
                tf.add(current, arcname=current.relative_to(ROOT), recursive=False)
            for filename in sorted(filenames):
                item = current / filename
                if item.suffix == ".pyc":
                    continue
                if item.name.endswith(".generated.json") and "chemistry" in item.parts:
                    continue
                tf.add(item, arcname=item.relative_to(ROOT), recursive=False)


def _source_child_row(row: Mapping[str, str]) -> bool:
    record_id = str(row.get("record_id") or "")
    method = str(row.get("method_key") or "")
    if not record_id.endswith("__polychildren"):
        return False
    if method == "snake":
        return str(row.get("snake_phase3_runtime_split_mode") or "") == PAULI_CHILD_MODE
    return str(row.get("generic_adapt_runtime_split_mode") or "") == PAULI_CHILD_MODE


def source_rows(path: Path = SOURCE_RECORDS_TSV) -> dict[tuple[str, str], dict[str, str]]:
    out: dict[tuple[str, str], dict[str, str]] = {}
    for row in read_records(path):
        method = str(row.get("method_key") or "")
        regime = str(row.get("display_regime") or "")
        if method not in set(ALL_METHODS) or regime not in set(REGIME_ORDER):
            continue
        if str(row.get("engine_key") or "") != "native_forced":
            continue
        if str(row.get("budget") or "") != str(DEFAULT_BUDGET):
            continue
        if not _source_child_row(row):
            continue
        key = (regime, method)
        if key in out:
            raise ValueError(f"Duplicate source child row for {key}")
        out[key] = row
    return out


def _record_id(
    batch_id: str,
    regime: str,
    method_key: str,
    policy: MatrixPolicy,
    budget: int,
    overlay: OptimizerOverlay,
    max_depth: int = 30,
) -> str:
    return (
        f"{batch_id}__{regime.replace('-', '_')}__{method_key}"
        f"__{policy.label}"
        f"__native_forced__{overlay.overlay_id}{int(budget)}__depth{int(max_depth)}_noearlystop"
        "__fullmeta_singleton_symmetry"
    )


def _source_command_args(row: Mapping[str, str]) -> list[str]:
    raw = str(row.get("source_command_args_json") or "").strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes)):
        return []
    return [str(item) for item in payload]


def _needs_strong_strong_resume_repair(source: Mapping[str, str]) -> bool:
    if str(source.get("method_key") or "") != "snake":
        return False
    if str(source.get("display_regime") or "") != "strong-strong":
        return False
    args = _source_command_args(source)
    return RESUME_SCAFFOLD_FLAG in args


def _snake_cli_overrides_for_source(
    source: Mapping[str, str],
    *,
    strong_strong_snake_start_mode: str = STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR,
) -> dict[str, Any]:
    overrides = json.loads(json.dumps(SNAKE_CANONICAL_CLI_OVERRIDES))
    if _needs_strong_strong_resume_repair(source):
        if strong_strong_snake_start_mode == STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO:
            remove_value_flags = list(overrides.get("remove_value_flags", []))
            remove_value_flags.extend(
                [
                    RESUME_SCAFFOLD_FLAG,
                    RESUME_MODE_FLAG,
                    RESUME_COMPILE_SMOKE_FLAG,
                    SEGMENT_ID_FLAG,
                ]
            )
            overrides["remove_value_flags"] = list(dict.fromkeys(remove_value_flags))
        else:
            set_flags = dict(overrides.get("set_flags", {}))
            set_flags[RESUME_SCAFFOLD_FLAG] = STRONG_STRONG_RESUME_REPAIR_JSON
            overrides["set_flags"] = set_flags
    return overrides


def _snake_runtime_policy(policy: MatrixPolicy) -> dict[str, str]:
    if policy.child_policy == "native_phase3_singleton":
        if policy.label == PHASE3_TRUE_NO_GUARD_LABEL:
            child_set_symmetry_policy = "off"
        elif policy.symmetry_policy == "hard_guard":
            child_set_symmetry_policy = "hard_guard"
        else:
            child_set_symmetry_policy = "parent"
        return {
            "snake_phase3_runtime_split_mode": PAULI_CHILD_MODE,
            "snake_phase3_runtime_split_selection_mode": SNAKE_RUNTIME_SELECTION_MODE,
            "snake_phase3_runtime_split_child_set_symmetry_policy": child_set_symmetry_policy,
            "snake_phase3_runtime_split_max_subset_size": SINGLETON_SUBSET_SIZE,
            "snake_adapt_child_pool_expansion_mode": "off",
            "snake_adapt_child_pool_expansion_symmetry_policy": "",
            "snake_adapt_child_pool_expansion_max_subset_size": "",
            "shared_pauli_pool_mode": "off",
            "shared_pauli_pool_symmetry_policy": "",
            "shared_pauli_pool_max_subset_size": "",
        }
    if policy.child_policy == "common_phase0_singleton":
        return {
            "snake_phase3_runtime_split_mode": "off",
            "snake_phase3_runtime_split_selection_mode": "",
            "snake_phase3_runtime_split_child_set_symmetry_policy": "",
            "snake_phase3_runtime_split_max_subset_size": "",
            "snake_adapt_child_pool_expansion_mode": "off",
            "snake_adapt_child_pool_expansion_symmetry_policy": "",
            "snake_adapt_child_pool_expansion_max_subset_size": "",
            "shared_pauli_pool_mode": SHARED_PAULI_POOL_MODE,
            "shared_pauli_pool_symmetry_policy": policy.symmetry_policy,
            "shared_pauli_pool_max_subset_size": SINGLETON_SUBSET_SIZE,
        }
    return {
        "snake_phase3_runtime_split_mode": "off",
        "snake_phase3_runtime_split_selection_mode": "",
        "snake_phase3_runtime_split_child_set_symmetry_policy": "",
        "snake_phase3_runtime_split_max_subset_size": "",
        "snake_adapt_child_pool_expansion_mode": "off",
        "snake_adapt_child_pool_expansion_symmetry_policy": "",
        "snake_adapt_child_pool_expansion_max_subset_size": "",
        "shared_pauli_pool_mode": "off",
        "shared_pauli_pool_symmetry_policy": "",
        "shared_pauli_pool_max_subset_size": "",
    }


def _generic_runtime_policy(policy: MatrixPolicy) -> dict[str, str]:
    if policy.child_policy == "native_phase3_singleton":
        return {
            "generic_adapt_runtime_split_mode": PAULI_CHILD_MODE,
            "generic_adapt_runtime_split_symmetry_policy": (
                "hard_guard" if policy.symmetry_policy == "hard_guard" else "off"
            ),
            "generic_adapt_runtime_split_max_subset_size": SINGLETON_SUBSET_SIZE,
            "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
            "shared_pauli_pool_mode": "off",
            "shared_pauli_pool_symmetry_policy": "",
            "shared_pauli_pool_max_subset_size": "",
        }
    if policy.child_policy == "common_phase0_singleton":
        return {
            "generic_adapt_runtime_split_mode": "",
            "generic_adapt_runtime_split_symmetry_policy": "",
            "generic_adapt_runtime_split_max_subset_size": "",
            "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
            "shared_pauli_pool_mode": SHARED_PAULI_POOL_MODE,
            "shared_pauli_pool_symmetry_policy": policy.symmetry_policy,
            "shared_pauli_pool_max_subset_size": SINGLETON_SUBSET_SIZE,
        }
    return {
        "generic_adapt_runtime_split_mode": "",
        "generic_adapt_runtime_split_symmetry_policy": "",
        "generic_adapt_runtime_split_max_subset_size": "",
        "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
        "shared_pauli_pool_mode": "off",
        "shared_pauli_pool_symmetry_policy": "",
        "shared_pauli_pool_max_subset_size": "",
    }


def _schedule_source_note(
    overlay: OptimizerOverlay,
    policy: MatrixPolicy,
    regime: str,
    method_key: str,
    budget: int,
) -> str:
    if overlay.overlay_id == "spsa_paper_i_hh" and method_key in {"geo", "append"}:
        return (
            f"Candidate {overlay.optimizer_label} matrix row {policy.label} from the native-forced "
            f"depth30/no-target Pauli-child {regime} source row. SPSA schedule/budget fields are copied "
            f"from the matching {method_key} source row under user-approved regime-specific schedule policy; "
            "pool, HVA/class-filter, child-policy, output-path, and no-guard settings remain the current "
            f"{policy.label} full-meta singleton contract with maxiter={int(budget)}."
        )
    if overlay.spsa_schedule is not None:
        return (
            f"Candidate {overlay.optimizer_label} matrix row {policy.label} from the native-forced "
            f"depth30/no-target Pauli-child {regime} source row. SPSA schedule fields are copied "
            "from the Paper-I HH SNAKE source_command_args_json "
            "(a=0.1, c=0.02, alpha=0.602, gamma=0.101, A=5.0, seed=7) while "
            f"the current matrix budget stays maxiter={int(budget)}."
        )
    return (
        f"Candidate {overlay.optimizer_label} matrix row {policy.label} from the native-forced "
        f"depth30/no-target Pauli-child {regime} source row. SPSA schedule fields are cleared "
        f"because the inner optimizer is {overlay.optimizer_label} with maxiter={int(budget)}."
    )


def _spsa_schedule_policy(overlay: OptimizerOverlay, method_key: str) -> str:
    if overlay.clears_spsa_schedule:
        return NON_SPSA_SCHEDULE_POLICY
    if overlay.overlay_id == "spsa_paper_i_hh" and method_key in {"geo", "append"}:
        return SOURCE_ROW_SPSA_SCHEDULE_POLICY
    return SNAKE_SPSA_SCHEDULE_POLICY


def make_row(
    batch_id: str,
    source: Mapping[str, str],
    *,
    policy: MatrixPolicy,
    budget: int,
    overlay: OptimizerOverlay,
    max_depth: int = 30,
    strong_strong_snake_start_mode: str = STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR,
) -> dict[str, str]:
    method_key = str(source["method_key"])
    regime = str(source["display_regime"])
    record_id = _record_id(batch_id, regime, method_key, policy, budget, overlay, max_depth)
    changed = [
        field
        for field in str(source.get("changed_fields_vs_anchor") or "").split(",")
        if field
    ]
    changed.extend(["adapt_optimizer_kind", "budget", "full_meta_contract", "matrix_label"])
    row_runnable = bool(policy.runnable)
    row_blocked_reason = "" if row_runnable else str(policy.blocked_reason)
    if policy.label == PHASE3_TRUE_NO_GUARD_LABEL and method_key != "snake":
        row_runnable = False
        row_blocked_reason = "phase3_true_no_guard_is_snake_only"
    row = dict(source)
    row.update(
        {
            "record_id": record_id,
            "batch_id": batch_id,
            "run_class": "candidate",
            "runnable": "true" if row_runnable else "false",
            "blocker": "" if row_runnable else row_blocked_reason,
            "matrix_label": policy.label,
            "matrix_role": policy.role,
            "pool_contract": "full_meta_unfiltered",
            "hh_adaptive_pool_profile": "full_meta_unfiltered",
            "child_policy": policy.child_policy,
            "symmetry_policy": policy.symmetry_policy,
            "optimizer": overlay.optimizer_label,
            "optimizer_overlay_id": overlay.overlay_id,
            "optimizer_contract_id": (
                f"{overlay.contract_prefix}_maxiter{int(budget)}_depth{int(max_depth)}_v1"
            ),
            "child_subset_size": SINGLETON_SUBSET_SIZE if policy.child_policy != "macro_only" else "",
            "blocked_reason": "" if row_runnable else row_blocked_reason,
            "spsa_schedule_policy": _spsa_schedule_policy(overlay, method_key),
            "engine_key": f"native_forced_{overlay.overlay_id}",
            "engine_label": (
                f"native forced {overlay.optimizer_label} full-meta singleton symmetry matrix candidate"
            ),
            "spsa_refit_engine": "",
            "budget": str(int(budget)),
            "max_depth": str(int(max_depth)),
            "adapt_optimizer_kind": overlay.adapt_optimizer_kind,
            "optimizer_profile": "",
            "source_settings_status": (
                "ok_candidate_fullmeta_singleton_symmetry_matrix"
                if row_runnable
                else "blocked_candidate_fullmeta_singleton_symmetry_matrix"
            ),
            "schedule_source_policy": overlay.schedule_source_policy,
            "schedule_source_regime": regime,
            "schedule_source_method": method_key,
            "schedule_source_note": _schedule_source_note(overlay, policy, regime, method_key, budget),
            "source_contract_note": (
                "Current user-selected full_meta contract: no full_meta_minus_hva class filter. "
                f"Matrix label {policy.label}: child_policy={policy.child_policy}; "
                f"symmetry_policy={policy.symmetry_policy}; singleton cap=1. This row is not the "
                "older shared-pool or minus-HVA diagnostic surface."
            ),
        }
    )
    if overlay.clears_spsa_schedule:
        for field in SCHEDULE_FIELDS:
            row[field] = ""
        row["spsa_refit_engine"] = ""
    elif method_key in {"geo", "append"}:
        for field in SCHEDULE_FIELDS:
            row[field] = str(source.get(field) or "")
        row["spsa_refit_engine"] = str(source.get("spsa_refit_engine") or "")
        changed.extend(["paper_i_hh_source_row_spsa_schedule", *SCHEDULE_FIELDS, "spsa_refit_engine"])
    elif overlay.spsa_schedule is not None:
        for field, value in overlay.spsa_schedule.items():
            row[field] = str(value)
        changed.extend(["paper_i_hh_spsa_schedule", *SCHEDULE_FIELDS])

    row["adapt_pool_class_filter_json"] = "off"

    if method_key == "snake":
        changed.extend(
            [
                "adapt_inner_optimizer",
                "adapt_schur_warm_start_mode",
                "snake_phase3_runtime_split_mode",
                "snake_phase3_runtime_split_selection_mode",
                "snake_phase3_runtime_split_child_set_symmetry_policy",
                "snake_phase3_runtime_split_max_subset_size",
                "snake_adapt_child_pool_expansion_mode",
                "shared_pauli_pool_mode",
                "shared_pauli_pool_symmetry_policy",
                "shared_pauli_pool_max_subset_size",
                "snake_cli_overrides_json",
            ]
        )
        if _needs_strong_strong_resume_repair(source):
            if strong_strong_snake_start_mode == STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO:
                changed.extend(
                    [
                        "adapt_resume_scaffold_json_removed",
                        "adapt_resume_mode_removed",
                        "adapt_resume_compile_smoke_removed",
                        "adapt_segment_id_removed",
                        "strong_strong_snake_start_mode_depth_zero",
                    ]
                )
            else:
                changed.extend(["adapt_resume_scaffold_json", "resume_scaffold_repair_json"])
        row["adapt_schur_warm_start_mode"] = SNAKE_SCHUR_WARM_START_MODE
        row.update(_snake_runtime_policy(policy))
        row["generic_adapt_runtime_split_mode"] = ""
        row["generic_adapt_runtime_split_symmetry_policy"] = ""
        row["generic_adapt_runtime_split_max_subset_size"] = ""
        row["generic_adapt_stop_policy"] = ""
        row["resource_pool_term_cap"] = ""
        row["snake_cli_overrides_json"] = json.dumps(
            _snake_cli_overrides_for_source(
                source,
                strong_strong_snake_start_mode=strong_strong_snake_start_mode,
            ),
            sort_keys=True,
            separators=(",", ":"),
        )
        if _needs_strong_strong_resume_repair(source):
            if strong_strong_snake_start_mode == STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO:
                row["resume_scaffold_repair_json"] = ""
                row["resume_scaffold_repair_status"] = "removed_for_depth_zero_fair_contract"
            else:
                row["resume_scaffold_repair_json"] = STRONG_STRONG_RESUME_REPAIR_JSON
                row["resume_scaffold_repair_status"] = "strict_reconstructed_initial_state"
    else:
        changed.extend(
            [
                "generic_adapt_runtime_split_mode",
                "generic_adapt_runtime_split_symmetry_policy",
                "generic_adapt_runtime_split_max_subset_size",
                "generic_adapt_stop_policy",
                "shared_pauli_pool_mode",
                "shared_pauli_pool_symmetry_policy",
                "shared_pauli_pool_max_subset_size",
            ]
        )
        row["adapt_schur_warm_start_mode"] = ""
        row["snake_phase3_runtime_split_mode"] = ""
        row["snake_phase3_runtime_split_selection_mode"] = ""
        row["snake_phase3_runtime_split_child_set_symmetry_policy"] = ""
        row["snake_phase3_runtime_split_max_subset_size"] = ""
        row["snake_adapt_child_pool_expansion_mode"] = ""
        row["snake_adapt_child_pool_expansion_symmetry_policy"] = ""
        row["snake_adapt_child_pool_expansion_max_subset_size"] = ""
        row.update(_generic_runtime_policy(policy))
        row["resource_pool_term_cap"] = "0"
        row["snake_cli_overrides_json"] = ""
        row["resume_scaffold_repair_json"] = ""
        row["resume_scaffold_repair_status"] = ""

    row["changed_fields_vs_anchor"] = ",".join(dict.fromkeys(changed))
    row.update(output_paths(record_id, method_key))
    for field in OUTPUT_FIELDNAMES:
        row.setdefault(field, "")
    return row


def build_records_for_regimes(
    batch_id: str,
    *,
    regimes: Sequence[str] = REGIME_ORDER,
    methods: Sequence[str] = METHOD_ORDER,
    matrix_labels: Sequence[str] | None = None,
    budget: int = DEFAULT_BUDGET,
    max_depth: int = 30,
    optimizer_overlay_id: str = "rotosolve",
    strong_strong_snake_start_mode: str = STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR,
) -> list[dict[str, str]]:
    if strong_strong_snake_start_mode not in STRONG_STRONG_SNAKE_START_MODES:
        raise ValueError(
            f"Unknown strong_strong_snake_start_mode={strong_strong_snake_start_mode!r}; "
            f"expected one of {STRONG_STRONG_SNAKE_START_MODES}"
        )
    try:
        overlay = OPTIMIZER_OVERLAYS[str(optimizer_overlay_id)]
    except KeyError as exc:
        raise ValueError(
            f"Unknown optimizer overlay {optimizer_overlay_id!r}; expected "
            f"{sorted(OPTIMIZER_OVERLAYS.keys())}"
        ) from exc
    selected_regimes = tuple(str(regime) for regime in regimes)
    unknown_regimes = [regime for regime in selected_regimes if regime not in set(REGIME_ORDER)]
    if unknown_regimes:
        raise ValueError(f"Unknown regimes: {unknown_regimes}; expected subset of {REGIME_ORDER}")
    selected_methods = tuple(str(method) for method in methods)
    unknown_methods = [method for method in selected_methods if method not in set(ALL_METHODS)]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}; expected subset of {ALL_METHODS}")
    selected_label_set = None if not matrix_labels else {str(label) for label in matrix_labels}
    if (
        overlay.overlay_id == "spsa_paper_i_hh"
        and any(method != "snake" for method in selected_methods)
        and (
            not selected_label_set
            or not selected_label_set.issubset(SOURCE_ROW_SPSA_ALLOWED_NON_SNAKE_LABELS)
        )
    ):
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked_spsa_overlay_non_snake_requires_source_row_repair",
                    "reason": (
                        "Non-SNAKE SPSA rows are currently allowed only for the user-approved "
                        "source-row schedule repair labels, where Geo/append copy "
                        "regime-specific SPSA schedule fields from matching Paper-I source rows."
                    ),
                    "allowed_matrix_labels": sorted(SOURCE_ROW_SPSA_ALLOWED_NON_SNAKE_LABELS),
                    "requested_methods": list(selected_methods),
                    "requested_matrix_labels": list(matrix_labels or ()),
                    "source_records_tsv": rel_or_abs(SOURCE_RECORDS_TSV),
                },
                indent=2,
                sort_keys=True,
            )
        )
    policy_source = DEFAULT_MATRIX_POLICIES if selected_label_set is None else MATRIX_POLICIES
    selected_policies = tuple(
        policy
        for policy in policy_source
        if selected_label_set is None or policy.label in selected_label_set
    )
    unknown_labels = [
        str(label)
        for label in (matrix_labels or ())
        if str(label) not in {policy.label for policy in MATRIX_POLICIES}
    ]
    if unknown_labels:
        raise ValueError(f"Unknown matrix labels: {unknown_labels}; expected {[p.label for p in MATRIX_POLICIES]}")
    configure_batch(batch_id)
    sources = source_rows()
    missing = [
        {"regime": regime, "method_key": method}
        for regime in selected_regimes
        for method in selected_methods
        if (regime, method) not in sources
    ]
    if missing:
        raise SystemExit(
            json.dumps(
                {
                    "status": "blocked_missing_source_rows",
                    "source_records_tsv": rel_or_abs(SOURCE_RECORDS_TSV),
                    "missing": missing,
                },
                indent=2,
                sort_keys=True,
            )
        )
    return [
        make_row(
            batch_id,
            sources[(regime, method)],
            policy=policy,
            budget=int(budget),
            overlay=overlay,
            max_depth=int(max_depth),
            strong_strong_snake_start_mode=strong_strong_snake_start_mode,
        )
        for regime in selected_regimes
        for policy in selected_policies
        for method in selected_methods
    ]


def write_records(
    batch_id: str,
    records: Sequence[dict[str, str]],
    *,
    budget: int = DEFAULT_BUDGET,
    max_depth: int = 30,
    request_cpus: int = DEFAULT_REQUEST_CPUS,
    request_memory_mb: int = DEFAULT_REQUEST_MEMORY_MB,
    request_disk_mb: int = DEFAULT_REQUEST_DISK_MB,
    high_memory_mb: int = DEFAULT_HIGH_MEMORY_MB,
    high_memory_disk_mb: int = DEFAULT_HIGH_MEMORY_DISK_MB,
    high_memory_record_ids: Sequence[str] | None = None,
    high_memory_regimes: Sequence[str] | None = None,
    high_memory_methods: Sequence[str] | None = None,
    high_memory_matrix_labels: Sequence[str] | None = None,
    max_runtime_s: int = DEFAULT_MAX_RUNTIME_S,
    strong_strong_snake_start_mode: str = STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR,
) -> dict[str, Any]:
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records, high_memory_ids = _apply_resource_tiers(
        records,
        request_memory_mb=int(request_memory_mb),
        request_disk_mb=int(request_disk_mb),
        high_memory_mb=int(high_memory_mb),
        high_memory_disk_mb=int(high_memory_disk_mb),
        high_memory_record_ids=high_memory_record_ids,
        high_memory_regimes=high_memory_regimes,
        high_memory_methods=high_memory_methods,
        high_memory_matrix_labels=high_memory_matrix_labels,
    )
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    record_ids = input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    record_queue = input_dir / "paper_i_hh_spsa_budget_ladder_record_queue.tsv"
    smoke_ids = input_dir / "paper_i_hh_spsa_budget_ladder_smoke_record_ids.txt"
    all_record_ids = input_dir / "paper_i_hh_fullmeta_singleton_symmetry_all_record_ids.txt"
    blocked_record_ids = input_dir / "paper_i_hh_fullmeta_singleton_symmetry_blocked_record_ids.txt"
    high_memory_record_ids_path = input_dir / "paper_i_hh_fullmeta_singleton_symmetry_high_memory_record_ids.txt"
    src_sanitized_tarball = input_dir / SRC_SANITIZED_TARBALL_NAME
    manifest_json = input_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"
    runnable_records = [row for row in records if str(row.get("runnable") or "").lower() == "true"]
    blocked_records = [row for row in records if str(row.get("runnable") or "").lower() != "true"]
    if not runnable_records:
        raise ValueError("No runnable records were generated.")

    input_dir.mkdir(parents=True, exist_ok=True)
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(OUTPUT_FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)
    write_lines(record_ids, (row["record_id"] for row in runnable_records))
    write_lines(
        record_queue,
        (
            f"{row['record_id']}\t{int(row['request_memory_mb'])}\t{int(row['request_disk_mb'])}"
            for row in runnable_records
        ),
    )
    write_lines(all_record_ids, (row["record_id"] for row in records))
    write_lines(blocked_record_ids, (row["record_id"] for row in blocked_records))
    write_lines(high_memory_record_ids_path, high_memory_ids)
    write_lines(smoke_ids, (runnable_records[0]["record_id"],))
    _write_sanitized_src_tarball(src_sanitized_tarball)

    by_method: dict[str, list[str]] = defaultdict(list)
    by_regime: dict[str, list[str]] = defaultdict(list)
    by_matrix_label: dict[str, list[str]] = defaultdict(list)
    for row in records:
        by_method[str(row["method_key"])].append(row["record_id"])
        by_regime[str(row["display_regime"])].append(row["record_id"])
        by_matrix_label[str(row["matrix_label"])].append(row["record_id"])
    overlay_ids = sorted({str(row.get("optimizer_overlay_id") or "") for row in records if row.get("optimizer_overlay_id")})
    overlay_id = overlay_ids[0] if len(overlay_ids) == 1 else "mixed_optimizer"
    optimizer_labels = sorted({str(row.get("optimizer") or "") for row in records if row.get("optimizer")})
    optimizer_label = optimizer_labels[0] if len(optimizer_labels) == 1 else "MIXED"
    for method_key, ids in sorted(by_method.items()):
        write_lines(input_dir / f"paper_i_hh_fullmeta_phase3_singleton_{overlay_id}_{method_key}_record_ids.txt", ids)
    for regime, ids in sorted(by_regime.items()):
        write_lines(
            input_dir / f"paper_i_hh_fullmeta_phase3_singleton_{overlay_id}_{regime.replace('-', '_')}_record_ids.txt",
            ids,
        )
    for matrix_label, ids in sorted(by_matrix_label.items()):
        write_lines(
            input_dir / f"paper_i_hh_fullmeta_singleton_symmetry_{matrix_label}_record_ids.txt",
            ids,
        )

    manifest = {
        "schema": "paper_i_hh_fullmeta_phase3_singleton_crossoptimizer_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "run_class": "candidate",
        "source_contract": {
            "source_batch_id": SOURCE_BATCH_ID,
            "source_records_tsv": rel_or_abs(SOURCE_RECORDS_TSV),
            "regimes": [regime for regime in REGIME_ORDER if regime in by_regime],
            "methods": sorted(by_method),
            "optimizer": optimizer_label,
            "optimizer_overlay_id": overlay_id,
            "maxiter": int(budget),
            "max_depth": int(max_depth),
            "pool": {
                "adapt_pool": "full_meta",
                "adapt_pool_class_filter_json": "off",
                "hh_adaptive_pool_profile": "full_meta_unfiltered",
                "hva_policy": "included_unfiltered_full_meta",
            },
            "snake_child_policy": {
                "phase3_runtime_split_mode": PAULI_CHILD_MODE,
                "allow_archival_phase3_runtime_split": True,
                "phase3_runtime_split_selection_mode": SNAKE_RUNTIME_SELECTION_MODE,
                "phase3_runtime_split_child_set_symmetry_policy": "matrix_label_dependent",
                "phase3_runtime_split_max_subset_size": int(SINGLETON_SUBSET_SIZE),
                "adapt_child_pool_expansion_mode": "off",
                "shared_pauli_pool_mode": "matrix_label_dependent",
                "shared_pauli_pool_symmetry_policy": "hard_guard_or_off_by_matrix_label",
                "shared_pauli_pool_max_subset_size": int(SINGLETON_SUBSET_SIZE),
                "schur_warm_start_mode": SNAKE_SCHUR_WARM_START_MODE,
            },
            "generic_child_policy": {
                "generic_adapt_runtime_split_mode": "matrix_label_dependent",
                "generic_adapt_runtime_split_symmetry_policy": "hard_guard_or_off_by_matrix_label",
                "generic_adapt_runtime_split_max_subset_size": int(SINGLETON_SUBSET_SIZE),
                "generic_adapt_stop_policy": "fixed_horizon_no_target_v1",
                "shared_pauli_pool_mode": "matrix_label_dependent",
                "shared_pauli_pool_symmetry_policy": "hard_guard_or_off_by_matrix_label",
                "shared_pauli_pool_max_subset_size": int(SINGLETON_SUBSET_SIZE),
            },
            "paper_facing_status": "candidate_pending_completed_run_evidence",
            "matrix_policies": [
                {
                    "matrix_label": policy.label,
                    "matrix_role": policy.role,
                    "child_policy": policy.child_policy,
                    "symmetry_policy": policy.symmetry_policy,
                    "runnable": bool(policy.runnable),
                    "blocked_reason": policy.blocked_reason,
                    "shared_pauli_pool_mode": (
                        SHARED_PAULI_POOL_MODE if policy.child_policy == "common_phase0_singleton" else "off"
                    ),
                    "shared_pauli_pool_symmetry_policy": (
                        policy.symmetry_policy if policy.child_policy == "common_phase0_singleton" else ""
                    ),
                    "shared_pauli_pool_max_subset_size": (
                        int(SINGLETON_SUBSET_SIZE) if policy.child_policy == "common_phase0_singleton" else None
                    ),
                }
                for policy in MATRIX_POLICIES
                if policy.label in by_matrix_label
            ],
            "snake_canonical_cli_overrides": SNAKE_CANONICAL_CLI_OVERRIDES,
            "strong_strong_snake_start_mode": strong_strong_snake_start_mode,
            "depth_zero_fair_contract": (
                "Rows generated with strong_strong_snake_start_mode=depth_zero remove "
                "--adapt-resume-scaffold-json, --adapt-resume-mode, "
                "--adapt-resume-compile-smoke, and --adapt-segment-id from strong-strong "
                "SNAKE source commands so all methods start from depth zero."
            ),
            "note": (
                "Built from the existing native-forced polychildren source rows but changes the "
                f"pool contract to unfiltered full_meta, uses {optimizer_label} maxiter={int(budget)}, and emits "
                "the A/B/C singleton symmetry matrix from the 2026-06-29 agent brief. This "
                "intentionally differs from the older full_meta_minus_hva shared-pool reset batch."
            ),
        },
        "resources": {
            "request_cpus": int(request_cpus),
            "standard_request_memory_mb": int(request_memory_mb),
            "standard_request_disk_mb": int(request_disk_mb),
            "high_memory_request_memory_mb": int(high_memory_mb),
            "high_memory_request_disk_mb": int(high_memory_disk_mb),
            "high_memory_record_count": len(high_memory_ids),
            "high_memory_selectors": {
                "record_ids": list(high_memory_record_ids or ()),
                "regimes": list(high_memory_regimes or ()),
                "methods": list(high_memory_methods or ()),
                "matrix_labels": list(high_memory_matrix_labels or ()),
            },
            "max_runtime_s": int(max_runtime_s),
        },
        "paths": {
            "records_tsv": rel_or_abs(records_tsv),
            "record_ids_txt": rel_or_abs(record_ids),
            "record_queue_tsv": rel_or_abs(record_queue),
            "all_record_ids_txt": rel_or_abs(all_record_ids),
            "blocked_record_ids_txt": rel_or_abs(blocked_record_ids),
            "high_memory_record_ids_txt": rel_or_abs(high_memory_record_ids_path),
            "smoke_record_ids_txt": rel_or_abs(smoke_ids),
            "src_sanitized_tarball": rel_or_abs(src_sanitized_tarball),
            "submit_file": rel_or_abs(submit_path),
        },
        "record_count": len(records),
        "runnable_record_count": len(runnable_records),
        "blocked_record_count": len(blocked_records),
        "records": [
            {
                "record_id": row["record_id"],
                "runnable": row["runnable"],
                "blocker": row["blocker"],
                "matrix_label": row["matrix_label"],
                "matrix_role": row["matrix_role"],
                "pool_contract": row["pool_contract"],
                "child_policy": row["child_policy"],
                "symmetry_policy": row["symmetry_policy"],
                "resource_tier": row["resource_tier"],
                "request_memory_mb": row["request_memory_mb"],
                "request_disk_mb": row["request_disk_mb"],
                "spsa_schedule_policy": row.get("spsa_schedule_policy"),
                "optimizer": row["optimizer"],
                "optimizer_overlay_id": row["optimizer_overlay_id"],
                "optimizer_contract_id": row["optimizer_contract_id"],
                "child_subset_size": row["child_subset_size"],
                "blocked_reason": row["blocked_reason"],
                "method_key": row["method_key"],
                "display_regime": row["display_regime"],
                "adapt_optimizer_kind": row["adapt_optimizer_kind"],
                "budget": row["budget"],
                "max_depth": row["max_depth"],
                "adapt_pool_class_filter_json": row.get("adapt_pool_class_filter_json"),
                "snake_phase3_runtime_split_mode": row.get("snake_phase3_runtime_split_mode"),
                "snake_phase3_runtime_split_selection_mode": row.get("snake_phase3_runtime_split_selection_mode"),
                "snake_phase3_runtime_split_child_set_symmetry_policy": row.get("snake_phase3_runtime_split_child_set_symmetry_policy"),
                "snake_phase3_runtime_split_max_subset_size": row.get("snake_phase3_runtime_split_max_subset_size"),
                "shared_pauli_pool_mode": row.get("shared_pauli_pool_mode"),
                "shared_pauli_pool_symmetry_policy": row.get("shared_pauli_pool_symmetry_policy"),
                "shared_pauli_pool_max_subset_size": row.get("shared_pauli_pool_max_subset_size"),
                "generic_adapt_runtime_split_mode": row.get("generic_adapt_runtime_split_mode"),
                "generic_adapt_runtime_split_symmetry_policy": row.get("generic_adapt_runtime_split_symmetry_policy"),
                "generic_adapt_runtime_split_max_subset_size": row.get("generic_adapt_runtime_split_max_subset_size"),
                "source_json": row.get("source_json"),
                "source_command_sh": row.get("source_command_sh"),
                "ordered_batch_beam_label": row.get("ordered_batch_beam_label"),
                "ordered_batch_beam_enabled": row.get("ordered_batch_beam_enabled"),
                "ordered_batch_beam_run_role": row.get("ordered_batch_beam_run_role"),
                "phase2_batch_selection_mode": row.get("phase2_batch_selection_mode"),
                "phase2_batch_target_size": row.get("phase2_batch_target_size"),
                "phase2_batch_size_cap": row.get("phase2_batch_size_cap"),
                "adapt_beam_live_branches": row.get("adapt_beam_live_branches"),
                "adapt_beam_children_per_parent": row.get("adapt_beam_children_per_parent"),
                "adapt_beam_lambda": row.get("adapt_beam_lambda"),
            }
            for row in records
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_matrix_submit_file(
        batch_id=batch_id,
        submit_path=submit_path,
        records_tsv=records_tsv,
        record_queue=record_queue,
        request_cpus=int(request_cpus),
        max_runtime_s=int(max_runtime_s),
        src_sanitized_tarball=src_sanitized_tarball,
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=30,
        help="ADAPT logical depth cap for generated rows. Defaults to production depth 30.",
    )
    parser.add_argument(
        "--optimizer-overlay",
        default="rotosolve",
        choices=tuple(OPTIMIZER_OVERLAYS.keys()),
        help=(
            "Inner-optimizer overlay. ROTOSOLVE and POWELL clear SPSA fields; "
            "spsa_paper_i_hh uses fixed SNAKE schedule fields and may use "
            "user-approved regime-specific source-row SPSA schedules for targeted Geo/append repair rows."
        ),
    )
    parser.add_argument(
        "--regime",
        action="append",
        choices=REGIME_ORDER,
        help="Regime to include. May be repeated. Defaults to all six regimes.",
    )
    parser.add_argument(
        "--method",
        action="append",
        choices=ALL_METHODS,
        help="Method to include. May be repeated. Defaults to SNAKE, Geo-ADAPT, and append-only ADAPT.",
    )
    parser.add_argument(
        "--matrix-label",
        action="append",
        choices=[policy.label for policy in MATRIX_POLICIES],
        help="Matrix row label to include. May be repeated. Defaults to all A/B/C rows.",
    )
    parser.add_argument("--request-cpus", type=int, default=DEFAULT_REQUEST_CPUS)
    parser.add_argument("--request-memory-mb", type=int, default=DEFAULT_REQUEST_MEMORY_MB)
    parser.add_argument("--request-disk-mb", type=int, default=DEFAULT_REQUEST_DISK_MB)
    parser.add_argument("--high-memory-mb", type=int, default=DEFAULT_HIGH_MEMORY_MB)
    parser.add_argument("--high-memory-disk-mb", type=int, default=DEFAULT_HIGH_MEMORY_DISK_MB)
    parser.add_argument(
        "--high-memory-record-id",
        action="append",
        default=[],
        help="Exact record id to run in the high-memory tier. May be repeated.",
    )
    parser.add_argument(
        "--high-memory-regime",
        action="append",
        choices=REGIME_ORDER,
        default=[],
        help="Regime selector for high-memory repair rows. May be repeated.",
    )
    parser.add_argument(
        "--high-memory-method",
        action="append",
        choices=ALL_METHODS,
        default=[],
        help="Method selector for high-memory repair rows. May be repeated.",
    )
    parser.add_argument(
        "--high-memory-matrix-label",
        action="append",
        choices=[policy.label for policy in MATRIX_POLICIES],
        default=[],
        help="Matrix-label selector for high-memory repair rows. May be repeated.",
    )
    parser.add_argument("--max-runtime-s", type=int, default=DEFAULT_MAX_RUNTIME_S)
    parser.add_argument(
        "--strong-strong-snake-start-mode",
        choices=STRONG_STRONG_SNAKE_START_MODES,
        default=STRONG_STRONG_SNAKE_START_MODE_SOURCE_REPAIR,
        help=(
            "How to handle historical strong-strong SNAKE source rows that contain "
            "resume scaffolds. Use depth_zero for fair repair batches where all methods "
            "must start from depth zero."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    budget = int(args.budget)
    if budget < 1:
        raise ValueError(f"--budget must be positive; got {budget}")
    max_depth = int(args.max_depth)
    if max_depth < 1:
        raise ValueError(f"--max-depth must be positive; got {max_depth}")
    records = build_records_for_regimes(
        batch_id,
        regimes=tuple(args.regime or REGIME_ORDER),
        methods=tuple(args.method or METHOD_ORDER),
        matrix_labels=tuple(args.matrix_label) if args.matrix_label else None,
        budget=budget,
        max_depth=max_depth,
        optimizer_overlay_id=str(args.optimizer_overlay),
        strong_strong_snake_start_mode=str(args.strong_strong_snake_start_mode),
    )
    manifest = write_records(
        batch_id,
        records,
        budget=budget,
        max_depth=max_depth,
        request_cpus=int(args.request_cpus),
        request_memory_mb=int(args.request_memory_mb),
        request_disk_mb=int(args.request_disk_mb),
        high_memory_mb=int(args.high_memory_mb),
        high_memory_disk_mb=int(args.high_memory_disk_mb),
        high_memory_record_ids=tuple(args.high_memory_record_id or ()),
        high_memory_regimes=tuple(args.high_memory_regime or ()),
        high_memory_methods=tuple(args.high_memory_method or ()),
        high_memory_matrix_labels=tuple(args.high_memory_matrix_label or ()),
        max_runtime_s=int(args.max_runtime_s),
        strong_strong_snake_start_mode=str(args.strong_strong_snake_start_mode),
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
