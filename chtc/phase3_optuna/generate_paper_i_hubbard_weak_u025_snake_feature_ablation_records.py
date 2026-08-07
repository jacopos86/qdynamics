#!/usr/bin/env python3
"""Generate depth-10 U/t=0.25 Hubbard-weak SNAKE feature-ablation records."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna.generate_paper_i_hh_spsa_budget_ladder_records import (
    FIELDNAMES as BASE_FIELDNAMES,
    configure_batch,
    output_paths,
    rel_or_abs,
    sha256_path,
)


DEFAULT_BATCH_ID = "paper_i_hubbard_weak_u025_snake_feature_ablation_depth10_20260620_v1"
TEMPLATE_COMMAND = (
    ROOT
    / "tmp"
    / "paper_i_hubbard_table_i_emitted_cli_exact_selected_source_20260604_v1"
    / "weak"
    / "trust_region_v1"
    / "command.sh"
)
REGIME = "hubbard-weak-u025"
METHOD_KEY = "snake"
METHOD_LABEL = "SNAKE"
MAX_DEPTH = 10
BUDGET = 200

BASE_CHANGED_FLAGS = (
    "--adapt-current-json",
    "--adapt-drop-floor",
    "--adapt-drop-min-depth",
    "--adapt-drop-patience",
    "--adapt-final-refit-maxiter",
    "--adapt-grad-floor",
    "--adapt-max-depth",
    "--adapt-maxiter",
    "--adapt-segment-id",
    "--adapt-segment-max-new-admissions",
    "--adapt-segment-target-depth",
    "--adapt-benchmark-target-abs-delta-e",
    "--adapt-benchmark-target-reference-energy",
    "--output-json",
    "--u",
    "--adapt-selected-logical-source-json",
    "--phase2-enable-batching",
    "--phase2-no-batching",
    "--phase3-enable-batching",
    "--phase3-no-batching",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    record_suffix: str
    feature: str
    submit_group: str
    note: str
    overrides: Mapping[str, Any]
    allowed_flags: Sequence[str]


def _zero_flags(*flags: str) -> dict[str, str]:
    return {flag: "0.0" for flag in flags}


DISABLED_MECHANISM_ROUTE_FLAGS = {"--static-route-id": "unspecified"}

NO_COST_SET_FLAGS: dict[str, str] = {
    **_zero_flags(
        "--phase1-lambda-compile",
        "--phase1-lambda-measure",
        "--phase1-lambda-leak",
        "--phase1-lambda-2q",
        "--phase1-lambda-d",
        "--phase1-lambda-1q",
        "--phase1-lambda-theta",
        "--phase1-lambda-shot",
        "--phase1-compile-cx-proxy-weight",
        "--phase1-compile-sq-proxy-weight",
        "--phase1-compile-rotation-step-weight",
        "--phase1-compile-position-shift-weight",
        "--phase1-compile-refit-active-weight",
        "--phase1-measure-groups-weight",
        "--phase1-measure-shots-weight",
        "--phase1-measure-reuse-weight",
        "--phase1-opt-dim-cost-scale",
        "--phase1-family-repeat-cost-scale",
        "--phase2-lambda-2q",
        "--phase2-lambda-d",
        "--phase2-lambda-1q",
        "--phase2-lambda-theta",
        "--phase2-lambda-shot",
        "--phase2-compile-cx-proxy-weight",
        "--phase2-compile-sq-proxy-weight",
        "--phase2-compile-rotation-step-weight",
        "--phase2-compile-position-shift-weight",
        "--phase2-compile-refit-active-weight",
        "--phase2-measure-groups-weight",
        "--phase2-measure-shots-weight",
        "--phase2-measure-reuse-weight",
        "--phase2-opt-dim-cost-scale",
        "--phase2-family-repeat-cost-scale",
        "--phase2-w-depth",
        "--phase2-w-group",
        "--phase2-w-shot",
        "--phase2-w-optdim",
        "--phase2-w-reuse",
        "--phase2-w-lifetime",
        "--phase3-backend-w-2q",
        "--phase3-backend-w-depth",
        "--phase3-backend-w-size",
    ),
    "--phase3-lifetime-cost-mode": "off",
    "--phase3-backend-cost-mode": "proxy",
}


VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="full_snake_anchor",
        record_suffix="full_snake_anchor",
        feature="none",
        submit_group="anchor",
        note="U/t=0.25 depth-10 source-command anchor with batching enabled and early stops disabled.",
        overrides={"set_flags": {}, "remove_flags": [], "boolean_pairs": []},
        allowed_flags=(),
    ),
    VariantSpec(
        name="no_batching",
        record_suffix="no_batching",
        feature="phase2_phase3_batching",
        submit_group="normal",
        note="Disable Phase-II/Phase-III batching only.",
        overrides={
            "set_flags": DISABLED_MECHANISM_ROUTE_FLAGS,
            "remove_flags": [],
            "boolean_pairs": [
                {"enable": "--phase2-enable-batching", "disable": "--phase2-no-batching", "enabled": False},
                {"enable": "--phase3-enable-batching", "disable": "--phase3-no-batching", "enabled": False},
            ],
        },
        allowed_flags=("--static-route-id", "--phase2-enable-batching", "--phase2-no-batching", "--phase3-enable-batching", "--phase3-no-batching"),
    ),
    VariantSpec(
        name="no_prune",
        record_suffix="no_prune",
        feature="recoverability_prune",
        submit_group="normal",
        note="Disable Phase-I prune/recoverability deletion only.",
        overrides={
            "set_flags": DISABLED_MECHANISM_ROUTE_FLAGS,
            "remove_flags": [],
            "boolean_pairs": [
                {"enable": "--phase1-prune-enabled", "disable": "--phase1-no-prune", "enabled": False},
                {
                    "enable": "--phase1-prune-amplitude-witness-required",
                    "disable": "--phase1-prune-amplitude-witness-optional",
                    "enabled": False,
                },
            ],
        },
        allowed_flags=(
            "--phase1-no-prune",
            "--static-route-id",
            "--phase1-prune-enabled",
            "--phase1-prune-amplitude-witness-optional",
            "--phase1-prune-amplitude-witness-required",
        ),
    ),
    VariantSpec(
        name="no_cost_term",
        record_suffix="no_cost_term",
        feature="resource_cost_term",
        submit_group="normal",
        note="Zero selector cost/resource weights while preserving telemetry.",
        overrides={"set_flags": {**DISABLED_MECHANISM_ROUTE_FLAGS, **NO_COST_SET_FLAGS}, "remove_flags": [], "boolean_pairs": []},
        allowed_flags=tuple(sorted((*NO_COST_SET_FLAGS, "--static-route-id"))),
    ),
    VariantSpec(
        name="no_novelty",
        record_suffix="no_novelty",
        feature="phase2_phase3_novelty",
        submit_group="normal",
        note="Disable Phase-II novelty and Phase-III novelty multiplier.",
        overrides={
            "set_flags": {
                **DISABLED_MECHANISM_ROUTE_FLAGS,
                "--phase2-gamma-N": "0.0",
                "--phase2-gamma-N-schedule-mode": "fixed",
                "--phase3-novelty-ablation-mode": "all",
            },
            "remove_flags": ["--phase2-gamma-N-schedule-start", "--phase2-gamma-N-schedule-end"],
            "boolean_pairs": [],
        },
        allowed_flags=(
            "--phase2-gamma-N",
            "--phase2-gamma-N-schedule-mode",
            "--phase2-gamma-N-schedule-start",
            "--phase2-gamma-N-schedule-end",
            "--phase3-novelty-ablation-mode",
            "--static-route-id",
        ),
    ),
    VariantSpec(
        name="phase2_novelty_only_no_second_order",
        record_suffix="phase2_novelty_only_no_second_order",
        feature="phase2_second_order_energy",
        submit_group="normal",
        note="Disable Phase-II trust-region second-order gain; Phase-II raw score is novelty divided by cost.",
        overrides={
            "set_flags": {**DISABLED_MECHANISM_ROUTE_FLAGS, "--phase2-selector-gain-mode": "unit_gain_v1"},
            "remove_flags": [],
            "boolean_pairs": [],
        },
        allowed_flags=("--phase2-selector-gain-mode", "--static-route-id"),
    ),
    VariantSpec(
        name="phase2_second_order_only_no_novelty",
        record_suffix="phase2_second_order_only_no_novelty",
        feature="phase2_novelty",
        submit_group="normal",
        note="Disable Phase-II novelty contribution while preserving trust-region second-order gain.",
        overrides={
            "set_flags": {
                **DISABLED_MECHANISM_ROUTE_FLAGS,
                "--phase2-selector-gain-mode": "trust_region_v1",
                "--phase2-gamma-N": "0.0",
                "--phase2-gamma-N-schedule-mode": "fixed",
                "--phase3-novelty-ablation-mode": "no_phase2",
            },
            "remove_flags": ["--phase2-gamma-N-schedule-start", "--phase2-gamma-N-schedule-end"],
            "boolean_pairs": [],
        },
        allowed_flags=(
            "--phase2-selector-gain-mode",
            "--phase2-gamma-N",
            "--phase2-gamma-N-schedule-mode",
            "--phase2-gamma-N-schedule-start",
            "--phase2-gamma-N-schedule-end",
            "--phase3-novelty-ablation-mode",
            "--static-route-id",
        ),
    ),
    VariantSpec(
        name="no_phase3",
        record_suffix="no_phase3",
        feature="phase3",
        submit_group="normal",
        note="Disable Phase III by running Phase I+II continuation only.",
        overrides={
            "set_flags": {**DISABLED_MECHANISM_ROUTE_FLAGS, "--adapt-continuation-mode": "phase2_v1", "--phase3-backend-cost-mode": "proxy"},
            "remove_flags": [],
            "boolean_pairs": [],
        },
        allowed_flags=("--adapt-continuation-mode", "--phase3-backend-cost-mode", "--static-route-id"),
    ),
    VariantSpec(
        name="phase1_only_no_phase2_phase3",
        record_suffix="phase1_only_no_phase2_phase3",
        feature="phase2_phase3",
        submit_group="normal",
        note="Disable Phase II and Phase III by running Phase I continuation only.",
        overrides={
            "set_flags": {**DISABLED_MECHANISM_ROUTE_FLAGS, "--adapt-continuation-mode": "phase1_v1", "--phase3-backend-cost-mode": "proxy"},
            "remove_flags": [],
            "boolean_pairs": [],
        },
        allowed_flags=("--adapt-continuation-mode", "--phase3-backend-cost-mode", "--static-route-id"),
    ),
    VariantSpec(
        name="full_geometry_window",
        record_suffix="full_geometry_window",
        feature="phase3_geometry_window",
        submit_group="normal",
        note="Use raw exact/full geometry selector instead of the reduced selector.",
        overrides={
            "set_flags": {**DISABLED_MECHANISM_ROUTE_FLAGS, "--phase3-selector-geometry-mode": "raw_exact"},
            "remove_flags": [],
            "boolean_pairs": [],
        },
        allowed_flags=("--phase3-selector-geometry-mode", "--static-route-id"),
    ),
    VariantSpec(
        name="no_shortlisting",
        record_suffix="no_shortlisting",
        feature="shortlisting",
        submit_group="no_shortlisting",
        note="Force full-pool shortlist/frontier behavior; expected to cost more.",
        overrides={
            "set_flags": {
                **DISABLED_MECHANISM_ROUTE_FLAGS,
                "--phase0-pilot-max-records": "0",
                "--phase1-shortlist-size": "100000",
                "--phase2-shortlist-fraction": "1.0",
                "--phase2-shortlist-size": "100000",
                "--phase2-frontier-ratio": "1.0",
                "--phase3-frontier-ratio": "1.0",
                "--algebraic-phase2-lane-rel-threshold": "0.0",
                "--algebraic-phase1-lane-quota-pressure": "1.0",
                "--algebraic-phase2-lane-quota-pressure": "1.0",
            },
            "remove_flags": [],
            "boolean_pairs": [],
        },
        allowed_flags=(
            "--phase0-pilot-max-records",
            "--phase1-shortlist-size",
            "--phase2-shortlist-fraction",
            "--phase2-shortlist-size",
            "--phase2-frontier-ratio",
            "--phase3-frontier-ratio",
            "--algebraic-phase2-lane-rel-threshold",
            "--algebraic-phase1-lane-quota-pressure",
            "--algebraic-phase2-lane-quota-pressure",
            "--static-route-id",
        ),
    ),
)

EXTRA_FIELDNAMES = (
    "hh_feature_ablation_variant",
    "hh_feature_ablation_feature",
    "hh_feature_ablation_submit_group",
    "hh_feature_ablation_note",
    "hh_feature_ablation_overrides_json",
    "hh_feature_ablation_allowed_flags_json",
    "hh_feature_ablation_plateau_source_json",
    "hh_feature_ablation_plateau_source_sha256",
    "hh_feature_ablation_plateau_k",
    "hh_feature_ablation_plateau_abs_delta_e",
    "hh_feature_ablation_plateau_s_alg",
    "hh_feature_ablation_fanout_gate",
)
FIELDNAMES = tuple(dict.fromkeys((*BASE_FIELDNAMES, *EXTRA_FIELDNAMES)))


def command_args_from_script(path: Path) -> list[str]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("set "):
            continue
        if "pipelines.static_adapt.adapt_pipeline" in line:
            return shlex.split(line)
    raise ValueError(f"No adapt_pipeline command found in {path}")


def flag_value(args: Sequence[str], flag: str) -> str | None:
    tokens = list(args)
    if flag not in tokens:
        return None
    idx = tokens.index(flag)
    if idx >= len(tokens) - 1:
        return None
    value = str(tokens[idx + 1])
    return None if value.startswith("--") else value


def remove_flag(args: list[str], flag: str) -> None:
    while flag in args:
        idx = args.index(flag)
        del args[idx : min(idx + 2, len(args))]


def set_flag(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        idx = args.index(flag)
        if idx == len(args) - 1:
            raise ValueError(f"{flag} has no value in source command")
        args[idx + 1] = str(value)
    else:
        args.extend([flag, str(value)])


def remove_toggle_pair(args: list[str], enable: str, disable: str) -> None:
    for flag in (enable, disable):
        while flag in args:
            args.pop(args.index(flag))


def apply_boolean_pair(args: list[str], *, enable: str, disable: str, enabled: bool) -> None:
    remove_toggle_pair(args, enable, disable)
    args.append(enable if enabled else disable)


def stage_selected_logical(batch_id: str, source_args: Sequence[str]) -> tuple[str, str, str]:
    raw = flag_value(source_args, "--adapt-selected-logical-source-json")
    if not raw:
        return "", "", ""
    source = Path(raw)
    if not source.is_absolute():
        source = ROOT / source
    if not source.exists():
        raise FileNotFoundError(f"selected-logical source is missing: {source}")
    staged = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id / "sources" / "hubbard_weak_u025" / source.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    return raw, rel_or_abs(staged), sha256_path(staged)


def normalize_source_args(batch_id: str, source_args: Sequence[str]) -> tuple[list[str], dict[str, Any]]:
    args = list(source_args)
    old_selected, staged_selected, staged_selected_sha = stage_selected_logical(batch_id, args)
    set_flag(args, "--u", "0.25")
    set_flag(args, "--adapt-selected-logical-source-json", staged_selected)
    set_flag(args, "--adapt-max-depth", str(MAX_DEPTH))
    set_flag(args, "--adapt-maxiter", str(BUDGET))
    set_flag(args, "--adapt-final-refit-maxiter", str(BUDGET))
    set_flag(args, "--adapt-segment-target-depth", str(MAX_DEPTH))
    set_flag(args, "--adapt-segment-max-new-admissions", str(MAX_DEPTH))
    set_flag(args, "--adapt-segment-wallclock-cap-s", "21600")
    set_flag(args, "--adapt-drop-floor", "-1")
    set_flag(args, "--adapt-drop-patience", "0")
    set_flag(args, "--adapt-drop-min-depth", "0")
    set_flag(args, "--adapt-grad-floor", "-1")
    set_flag(args, "--phase2-selector-gain-mode", "trust_region_v1")
    remove_flag(args, "--adapt-benchmark-target-reference-energy")
    remove_flag(args, "--adapt-benchmark-target-abs-delta-e")
    apply_boolean_pair(args, enable="--phase2-enable-batching", disable="--phase2-no-batching", enabled=True)
    apply_boolean_pair(args, enable="--phase3-enable-batching", disable="--phase3-no-batching", enabled=True)
    provenance = {
        "template_command": rel_or_abs(TEMPLATE_COMMAND),
        "template_command_sha256": sha256_path(TEMPLATE_COMMAND),
        "old_selected_logical_source_json": old_selected,
        "staged_selected_logical_source_json": staged_selected,
        "staged_selected_logical_sha256": staged_selected_sha,
        "baseline_command_changes": {
            "u": "0.25",
            "max_depth": MAX_DEPTH,
            "budget": BUDGET,
            "early_stops": "disabled",
            "batching": "enabled",
            "removed_reference_energy_stop": True,
        },
    }
    return args, provenance


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def make_record(
    *,
    batch_id: str,
    source_args: Sequence[str],
    source_provenance: Mapping[str, Any],
    variant: VariantSpec,
) -> dict[str, str]:
    record_id = f"{batch_id}__hubbard_weak_u025__snake__native200__{variant.record_suffix}"
    overrides = json.loads(json.dumps(variant.overrides))
    set_flags = dict(overrides.get("set_flags") or {})
    set_flags["--adapt-segment-id"] = record_id
    overrides["set_flags"] = set_flags
    row: dict[str, str] = {
        "record_id": record_id,
        "batch_id": batch_id,
        "run_class": "candidate",
        "runnable": "true",
        "blocker": "",
        "method_key": METHOD_KEY,
        "method_label": METHOD_LABEL,
        "algorithm_id": "static_family_native_adapt_phase3",
        "engine_key": "native_forced",
        "engine_label": "native SPSA",
        "spsa_refit_engine": "src.quantum.spsa_optimizer:spsa_minimize",
        "budget": str(BUDGET),
        "display_regime": REGIME,
        "internal_regime": "hubbard_weak_u025",
        "source_map_regime": "",
        "suite_profile": "paper_i_hubbard_u025_feature_ablation_depth10",
        "case_id": "hubbard_L2_u025_weak_diagnostic",
        "family": "hubbard",
        "n_ph_work": "0",
        "n_ph_ref": "0",
        "same_cutoff_exact_gs_energy": "",
        "same_cutoff_energy_key_hash": "",
        "exact_reference_energy": "",
        "exact_reference_energy_key_hash": "",
        "exact_reference_n_ph_max": "",
        "primary_energy_metric": "same_cutoff_abs_delta_e",
        "same_cutoff_error_role": "primary",
        "target_abs_delta_e": "",
        "max_depth": str(MAX_DEPTH),
        "adapt_optimizer_kind": "spsa",
        "source_json": source_provenance.get("staged_selected_logical_source_json", ""),
        "source_json_sha256": source_provenance.get("staged_selected_logical_sha256", ""),
        "source_command_sh": str(source_provenance.get("template_command", "")),
        "source_command_sha256": str(source_provenance.get("template_command_sha256", "")),
        "source_command_args_json": json.dumps(list(source_args), separators=(",", ":")),
        "source_rank": "",
        "source_trial": "",
        "source_settings_status": "template_command_u025_depth10_retargeted",
        "schedule_source_policy": "paper_i_hubbard_weak_template_command_retargeted_u025_depth10_feature_ablation",
        "schedule_source_regime": "hubbard-weak-template",
        "schedule_source_method": METHOD_LABEL,
        "schedule_source_json": str(source_provenance.get("staged_selected_logical_source_json", "")),
        "schedule_source_note": (
            "Diagnostic U/t=0.25 Hubbard-weak feature ablation. The selected-logical source "
            "comes from the visible Hubbard-weak command template; physics U, maxiter, depth, "
            "early-stop behavior, batching, output paths, and listed variant flags are the only intended changes."
        ),
        "anchor_source_json": str(source_provenance.get("staged_selected_logical_source_json", "")),
        "anchor_source_sha256": str(source_provenance.get("staged_selected_logical_sha256", "")),
        "anchor_cell_manifest_rel": "",
        "changed_fields_vs_anchor": ",".join(sorted(set((*BASE_CHANGED_FLAGS, *variant.allowed_flags)))),
        "source_contract_note": (
            "forced_depth30_no_early_stop marker reused intentionally for a forced depth-10 run; "
            "anchor must pass command audit before fan-out."
        ),
        "hh_feature_ablation_variant": variant.name,
        "hh_feature_ablation_feature": variant.feature,
        "hh_feature_ablation_submit_group": variant.submit_group,
        "hh_feature_ablation_note": variant.note,
        "hh_feature_ablation_overrides_json": json.dumps(overrides, sort_keys=True, separators=(",", ":")),
        "hh_feature_ablation_allowed_flags_json": json.dumps(
            sorted(set((*BASE_CHANGED_FLAGS, *variant.allowed_flags))),
            separators=(",", ":"),
        ),
        "hh_feature_ablation_plateau_source_json": "",
        "hh_feature_ablation_plateau_source_sha256": "",
        "hh_feature_ablation_plateau_k": "",
        "hh_feature_ablation_plateau_abs_delta_e": "",
        "hh_feature_ablation_plateau_s_alg": "",
        "hh_feature_ablation_fanout_gate": "source_value_anchor" if variant.submit_group == "anchor" else "anchor_must_pass_before_submit",
    }
    row.update(output_paths(record_id, METHOD_KEY))
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return row


def write_submit_file(*, batch_id: str, submit_path: Path, records_tsv: Path, record_ids: Path, submit_group: str) -> None:
    job_batch = "holstein-" + f"{batch_id}-{submit_group}".replace("_", "-")
    output_root = f"raw_outputs/{batch_id}"
    logs_root = f"logs/{batch_id}"
    transfer_inputs = ["pipelines", "src", "docs", "test_support", "chtc/phase3_optuna"]
    lines = [
        "universe = vanilla",
        "executable = chtc/phase3_optuna/run_paper_i_hh_native200_snake_feature_ablation_task_apptainer.sh",
        f"arguments = $(record_id) {rel_or_abs(records_tsv)} {output_root}/$(record_id)",
        "should_transfer_files = YES",
        "when_to_transfer_output = ON_EXIT_OR_EVICT",
        "transfer_executable = True",
        "preserve_relative_paths = True",
        "transfer_input_files = " + ", ".join(transfer_inputs),
        f"transfer_output_files = {output_root}, {logs_root}",
        "stream_output = False",
        "stream_error = False",
        f"log = logs/{batch_id}_{submit_group}.$(Cluster).$(Process).log",
        f"output = logs/{batch_id}_{submit_group}.$(Cluster).$(Process).out",
        f"error = logs/{batch_id}_{submit_group}.$(Cluster).$(Process).err",
        "requirements = TARGET.HasSIF",
        "request_cpus = 10",
        "request_memory = 49152MB",
        "request_disk = 40960MB",
        "+MaxRuntime = 86400",
        f'+JobBatchName = "{job_batch}"',
        f"queue record_id from {rel_or_abs(record_ids)}",
    ]
    submit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_records(batch_id: str, records: Sequence[dict[str, str]], source_provenance: Mapping[str, Any]) -> dict[str, Any]:
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hubbard_weak_u025_snake_feature_ablation_records.tsv"
    all_ids = input_dir / "paper_i_hubbard_weak_u025_snake_feature_ablation_record_ids.txt"
    anchor_ids = input_dir / "paper_i_hubbard_weak_u025_snake_feature_ablation_anchor_record_ids.txt"
    normal_ids = input_dir / "paper_i_hubbard_weak_u025_snake_feature_ablation_normal_record_ids.txt"
    no_shortlisting_ids = input_dir / "paper_i_hubbard_weak_u025_snake_feature_ablation_no_shortlisting_record_ids.txt"
    manifest_json = input_dir / "paper_i_hubbard_weak_u025_snake_feature_ablation_manifest.json"
    input_dir.mkdir(parents=True, exist_ok=True)
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow(row)

    write_lines(all_ids, (row["record_id"] for row in records))
    write_lines(anchor_ids, (row["record_id"] for row in records if row["hh_feature_ablation_submit_group"] == "anchor"))
    write_lines(normal_ids, (row["record_id"] for row in records if row["hh_feature_ablation_submit_group"] == "normal"))
    write_lines(no_shortlisting_ids, (row["record_id"] for row in records if row["hh_feature_ablation_submit_group"] == "no_shortlisting"))

    submit_anchor = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}_anchor.sub"
    submit_normal = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}_normal_after_anchor.sub"
    submit_no_shortlisting = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}_no_shortlisting_after_anchor.sub"
    write_submit_file(batch_id=batch_id, submit_path=submit_anchor, records_tsv=records_tsv, record_ids=anchor_ids, submit_group="anchor")
    write_submit_file(batch_id=batch_id, submit_path=submit_normal, records_tsv=records_tsv, record_ids=normal_ids, submit_group="normal")
    write_submit_file(batch_id=batch_id, submit_path=submit_no_shortlisting, records_tsv=records_tsv, record_ids=no_shortlisting_ids, submit_group="no_shortlisting")

    manifest = {
        "schema": "paper_i_hubbard_weak_u025_snake_feature_ablation_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": rel_or_abs(Path(__file__).resolve()),
        "batch_id": batch_id,
        "regime": REGIME,
        "method": METHOD_LABEL,
        "diagnostic_scope": {
            "family": "hubbard",
            "L": 2,
            "t": 1.0,
            "u": 0.25,
            "boundary": "periodic",
            "max_depth": MAX_DEPTH,
            "maxiter": BUDGET,
            "early_stops_disabled": True,
            "skip_trajectory": True,
            "anchor_batching": "enabled",
        },
        "source_provenance": source_provenance,
        "paths": {
            "records_tsv": rel_or_abs(records_tsv),
            "all_record_ids": rel_or_abs(all_ids),
            "anchor_record_ids": rel_or_abs(anchor_ids),
            "normal_record_ids": rel_or_abs(normal_ids),
            "no_shortlisting_record_ids": rel_or_abs(no_shortlisting_ids),
            "submit_anchor": rel_or_abs(submit_anchor),
            "submit_normal_after_anchor": rel_or_abs(submit_normal),
            "submit_no_shortlisting_after_anchor": rel_or_abs(submit_no_shortlisting),
        },
        "record_count": len(records),
        "records": [
            {
                "record_id": row["record_id"],
                "variant": row["hh_feature_ablation_variant"],
                "feature": row["hh_feature_ablation_feature"],
                "submit_group": row["hh_feature_ablation_submit_group"],
                "allowed_flags": json.loads(row["hh_feature_ablation_allowed_flags_json"]),
            }
            for row in records
        ],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_records(batch_id: str) -> tuple[list[dict[str, str]], dict[str, Any]]:
    configure_batch(batch_id)
    template_args = command_args_from_script(TEMPLATE_COMMAND)
    source_args, source_provenance = normalize_source_args(batch_id, template_args)
    records = [
        make_record(batch_id=batch_id, source_args=source_args, source_provenance=source_provenance, variant=variant)
        for variant in VARIANTS
    ]
    return records, source_provenance


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    batch_id = str(args.batch_id)
    records, source_provenance = build_records(batch_id)
    manifest = write_records(batch_id, records, source_provenance)
    print(json.dumps({"batch_id": batch_id, "record_count": len(records), "manifest": manifest["paths"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
