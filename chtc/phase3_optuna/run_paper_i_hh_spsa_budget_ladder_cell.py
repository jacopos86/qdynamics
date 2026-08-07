#!/usr/bin/env python3
"""Run one Paper-I HH SPSA budget/engine diagnostic ladder cell."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipelines.exact_bench.snake_table_i_measurement_work import (
    snake_algorithmic_work_from_payload,
    snake_fair_expanded_work_from_payload,
)


LOCAL_REPO_PREFIX = "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3"
WORK_REPO_PREFIX = "/work"
LEGACY_ENGINE = "exact_bench_spsa:energy_only_descent"
NATIVE_ENGINE = "src.quantum.spsa_optimizer:spsa_minimize"
SCHUR_WARM_START_FLAG = "--adapt-schur-warm-start-mode"
ADAPT_INNER_OPTIMIZER_FLAG = "--adapt-inner-optimizer"
SNAKE_RUNTIME_SPLIT_FLAG = "--phase3-runtime-split-mode"
SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG = "--allow-archival-phase3-runtime-split"
SNAKE_RUNTIME_SPLIT_SELECTION_FLAG = "--phase3-runtime-split-selection-mode"
SNAKE_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_FLAG = "--phase3-runtime-split-child-set-symmetry-policy"
SNAKE_RUNTIME_SPLIT_MAX_SUBSET_FLAG = "--phase3-runtime-split-max-subset-size"
SNAKE_CHILD_POOL_EXPANSION_FLAG = "--adapt-child-pool-expansion-mode"
SNAKE_CHILD_POOL_EXPANSION_SYMMETRY_FLAG = "--adapt-child-pool-expansion-symmetry-policy"
SNAKE_CHILD_POOL_EXPANSION_MAX_SUBSET_FLAG = "--adapt-child-pool-expansion-max-subset-size"
SHARED_PAULI_POOL_FLAG = "--shared-pauli-pool-mode"
SHARED_PAULI_POOL_SYMMETRY_FLAG = "--shared-pauli-pool-symmetry-policy"
SHARED_PAULI_POOL_MAX_SUBSET_FLAG = "--shared-pauli-pool-max-subset-size"
ADAPT_POOL_CLASS_FILTER_FLAG = "--adapt-pool-class-filter-json"
SNAKE_CLI_OVERRIDES_FIELD = "snake_cli_overrides_json"
RUNTIME_WORKER_OVERRIDE_FLAGS = (
    "--adapt-parallel-gradient-workers",
    "--adapt-beam-parent-workers",
    "--adapt-spsa-parallel-evaluations",
)
SOURCE_LOCK_ALLOWED_FLAG_CHANGES = {
    "--adapt-max-depth",
    "--adapt-maxiter",
    "--adapt-final-refit-maxiter",
    "--adapt-segment-target-depth",
    "--adapt-segment-max-new-admissions",
    "--adapt-drop-floor",
    "--adapt-drop-patience",
    "--adapt-drop-min-depth",
    "--adapt-grad-floor",
    "--adapt-benchmark-target-abs-delta-e",
    "--adapt-benchmark-target-reference-energy",
    "--output-json",
    "--adapt-current-json",
}
SOURCE_LOCK_PRESERVED_FLAGS = {
    "--adapt-ref-json",
    "--adapt-resume-scaffold-json",
    "--adapt-resume-mode",
    "--adapt-segment-id",
    "--adapt-segment-target-depth",
    "--adapt-segment-max-new-admissions",
    "--adapt-resume-compile-smoke",
    "--phase3-backend-cost-mode",
    "--adapt-pool",
    "--adapt-pool-class-filter-json",
    "--static-route-id",
    "--static-meta-feature-profile",
    "--adapt-continuation-mode",
    "--phase3-selector-policy",
    "--phase3-selector-geometry-mode",
    "--phase2-enable-batching",
    "--phase3-enable-batching",
}
RESUME_SCAFFOLD_FLAG = "--adapt-resume-scaffold-json"
RESUME_SCAFFOLD_REPAIR_BASENAME = "strict_resume_scaffold_repaired_initial_state.json"
RESUME_SCAFFOLD_REPAIR_SCRATCH_BASENAME = "_strict_resume_scaffold_without_initial_state.json"


def load_record(record_id: str, records_path: Path) -> dict[str, str]:
    with records_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("record_id") == record_id:
                return {str(k): "" if v is None else str(v) for k, v in row.items()}
    raise KeyError(f"record_id={record_id!r} not found in {records_path}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def set_flag(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        idx = args.index(flag)
        if idx == len(args) - 1:
            raise ValueError(f"Flag {flag} has no value in source command.")
        args[idx + 1] = str(value)
    else:
        args.extend([flag, str(value)])


def remove_flag(args: list[str], flag: str) -> None:
    while flag in args:
        idx = args.index(flag)
        del args[idx : min(idx + 2, len(args))]


def remove_bool_flag(args: list[str], flag: str) -> None:
    while flag in args:
        del args[args.index(flag)]


def flag_value(args: Sequence[str], flag: str) -> str | None:
    tokens = list(args)
    if flag not in tokens:
        return None
    idx = tokens.index(flag)
    if idx == len(tokens) - 1:
        raise ValueError(f"Flag {flag} has no value in source command.")
    return str(tokens[idx + 1])


def source_scaffold_depth(source_json: Path) -> int:
    payload = json.loads(source_json.read_text(encoding="utf-8"))
    adapt_vqe = payload.get("adapt_vqe")
    if not isinstance(adapt_vqe, Mapping):
        raise ValueError(f"SNAKE source JSON has no adapt_vqe object: {source_json}")
    raw_depth = adapt_vqe.get("ansatz_depth")
    try:
        depth = int(raw_depth)
    except (TypeError, ValueError):
        operators = adapt_vqe.get("operators")
        if not isinstance(operators, list):
            raise ValueError(f"SNAKE source JSON has no usable scaffold depth: {source_json}")
        depth = int(len(operators))
    if depth <= 0:
        raise ValueError(f"SNAKE source scaffold depth must be positive: {source_json}")
    return depth


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _strict_reconstructed_resume_scaffold(
    source_path: Path,
    *,
    output_root: Path,
    original_error: str,
) -> Path:
    """Repair legacy resume artifacts whose serialized prepared state is stale.

    The repaired artifact keeps the source settings, selected operators,
    parameterization, and theta values.  Only ``initial_state`` is replaced by
    the deterministic state reconstructed from the scaffold itself, after which
    the normal strict resume loader must accept the repaired artifact.
    """

    from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
    from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input
    from pipelines.static_adapt.resume_scaffold import load_static_resume_source

    source_path = Path(source_path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Resume scaffold payload is not a JSON object: {source_path}")

    output_root.mkdir(parents=True, exist_ok=True)
    scratch_path = output_root / RESUME_SCAFFOLD_REPAIR_SCRATCH_BASENAME
    scratch_payload = dict(payload)
    stale_initial_state = scratch_payload.pop("initial_state", None)
    write_json(scratch_path, scratch_payload)

    runtime_input = load_scaffold_runtime_input(
        scratch_path,
        loader_mode="replay_family",
        generator_family="match_adapt",
        fallback_family="full_meta",
    )
    repaired_payload = dict(payload)
    repaired_payload["initial_state"] = build_statevector_manifest(
        psi_state=runtime_input.psi_initial,
        source="runtime_loader.reconstructed_from_scaffold",
        handoff_state_kind="prepared_state",
        amplitude_cutoff=1.0e-14,
    )
    repaired_payload["resume_scaffold_repair"] = {
        "schema": "paper_i_hh_resume_scaffold_repair_v1",
        "repair": "strict_reconstructed_initial_state",
        "source_artifact_json": str(source_path),
        "source_artifact_sha256": _file_sha256(source_path),
        "original_loader_error": str(original_error),
        "stale_initial_state_present": isinstance(stale_initial_state, Mapping),
        "selected_term_count": int(len(runtime_input.selected_terms)),
        "runtime_parameter_count": int(runtime_input.base_layout.runtime_parameter_count),
        "logical_parameter_count": int(runtime_input.base_layout.logical_parameter_count),
        "no_credentials_serialized": True,
    }
    repaired_path = output_root / RESUME_SCAFFOLD_REPAIR_BASENAME
    write_json(repaired_path, repaired_payload)

    # Prove the repair through the same strict loader used by adapt_pipeline.
    load_static_resume_source(repaired_path, loader_mode="replay_family")
    return repaired_path


def repair_legacy_resume_scaffold_if_needed(args: list[str], *, output_root: Path) -> Path | None:
    raw_path = flag_value(args, RESUME_SCAFFOLD_FLAG)
    if not raw_path:
        return None
    source_path = Path(raw_path)
    if not source_path.exists():
        return None
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    if isinstance(payload, Mapping):
        repair = payload.get("resume_scaffold_repair")
        if (
            isinstance(repair, Mapping)
            and str(repair.get("schema") or "") == "paper_i_hh_resume_scaffold_repair_v1"
            and str(repair.get("repair") or "") == "strict_reconstructed_initial_state"
        ):
            return None
    from pipelines.static_adapt.resume_scaffold import load_static_resume_source

    try:
        load_static_resume_source(source_path, loader_mode="replay_family")
        return None
    except Exception as exc:
        message = str(exc)
        if "Prepared-state parity check failed" not in message:
            raise
        repaired_path = _strict_reconstructed_resume_scaffold(
            source_path,
            output_root=output_root,
            original_error=message,
        )
        set_flag(args, RESUME_SCAFFOLD_FLAG, str(repaired_path))
        return repaired_path


def replace_repo_paths(args: list[str]) -> list[str]:
    out: list[str] = []
    prefix = LOCAL_REPO_PREFIX + "/"
    work_prefix = WORK_REPO_PREFIX + "/"
    for token in args:
        text = str(token)
        if text == LOCAL_REPO_PREFIX:
            out.append(str(ROOT))
        elif text.startswith(prefix):
            out.append(str(ROOT / text[len(prefix) :]))
        elif text == WORK_REPO_PREFIX:
            out.append(str(ROOT))
        elif text.startswith(work_prefix):
            out.append(str(ROOT / text[len(work_prefix) :]))
        else:
            out.append(text)
    return out


def resolve_row_path(raw: str, fallback: Path) -> Path:
    text = str(raw or "").strip()
    if not text:
        return fallback
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def local_result_fallback(row: Mapping[str, str], output_root: Path) -> Path:
    if str(row.get("method_key") or "") == "snake":
        return output_root / "json" / "result.json"
    return output_root / "result" / "generic_static_single.json"


def resolve_result_path(row: Mapping[str, str], output_root: Path) -> Path:
    fallback = local_result_fallback(row, output_root)
    if fallback.exists():
        return fallback
    return resolve_row_path(str(row.get("result_json_rel") or ""), fallback)


def force_depth30_no_early_stop(row: Mapping[str, str]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "batch_id",
            "changed_fields_vs_anchor",
            "schedule_source_note",
            "source_contract_note",
        )
    )
    return (
        "forced_depth30_no_early_stop" in text
        or "disable_benchmark_target_stop" in text
        or "disable_drop_stop" in text
    )


def suppress_drop_plateau_terminal_stop(row: Mapping[str, str]) -> bool:
    text = " ".join(
        str(row.get(key) or "")
        for key in (
            "batch_id",
            "changed_fields_vs_anchor",
            "schedule_source_note",
            "source_contract_note",
        )
    )
    return "suppress_drop_plateau_terminal_stop" in text


def flag_values(args: Sequence[str]) -> dict[str, list[str | bool]]:
    values: dict[str, list[str | bool]] = {}
    idx = 0
    tokens = list(args)
    while idx < len(tokens):
        token = str(tokens[idx])
        if token.startswith("--"):
            if idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
                values.setdefault(token, []).append(str(tokens[idx + 1]))
                idx += 2
                continue
            values.setdefault(token, []).append(True)
        idx += 1
    return values


def positional_tokens(args: Sequence[str]) -> list[str]:
    out: list[str] = []
    idx = 0
    tokens = list(args)
    while idx < len(tokens):
        token = str(tokens[idx])
        if token.startswith("--"):
            if idx + 1 < len(tokens) and not str(tokens[idx + 1]).startswith("--"):
                idx += 2
            else:
                idx += 1
            continue
        out.append(token)
        idx += 1
    return out


def changed_flag_values(source_cmd: Sequence[str], effective_cmd: Sequence[str]) -> list[dict[str, Any]]:
    source_flags = flag_values(source_cmd)
    effective_flags = flag_values(effective_cmd)
    changes: list[dict[str, Any]] = []
    for flag in sorted(set(source_flags) | set(effective_flags)):
        before = source_flags.get(flag, [])
        after = effective_flags.get(flag, [])
        if before != after:
            changes.append({"flag": flag, "source": before, "effective": after})
    return changes


def _source_lock_allowed_flag_changes(
    *,
    schur_warm_start_mode: str = "off",
    inner_optimizer: str = "SPSA",
    runtime_worker_overrides: Mapping[str, int] | None = None,
    child_pool_expansion_mode: str = "source",
    shared_pauli_pool_mode: str = "source",
    adapt_pool_class_filter_json: str = "source",
    snake_cli_overrides: Mapping[str, Any] | None = None,
) -> set[str]:
    allowed = set(SOURCE_LOCK_ALLOWED_FLAG_CHANGES)
    if schur_warm_start_mode != "off":
        allowed.add(SCHUR_WARM_START_FLAG)
    allowed.add(ADAPT_INNER_OPTIMIZER_FLAG)
    if runtime_worker_overrides:
        allowed.update(runtime_worker_overrides.keys())
    if child_pool_expansion_mode != "source":
        allowed.update(
            {
                SNAKE_CHILD_POOL_EXPANSION_FLAG,
                SNAKE_CHILD_POOL_EXPANSION_SYMMETRY_FLAG,
                SNAKE_CHILD_POOL_EXPANSION_MAX_SUBSET_FLAG,
                SNAKE_RUNTIME_SPLIT_FLAG,
                SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG,
            }
        )
    if shared_pauli_pool_mode != "source":
        allowed.update(
            {
                SHARED_PAULI_POOL_FLAG,
                SHARED_PAULI_POOL_SYMMETRY_FLAG,
                SHARED_PAULI_POOL_MAX_SUBSET_FLAG,
                SNAKE_CHILD_POOL_EXPANSION_FLAG,
                SNAKE_CHILD_POOL_EXPANSION_SYMMETRY_FLAG,
                SNAKE_CHILD_POOL_EXPANSION_MAX_SUBSET_FLAG,
                SNAKE_RUNTIME_SPLIT_FLAG,
                SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG,
            }
        )
    if adapt_pool_class_filter_json != "source":
        allowed.add(ADAPT_POOL_CLASS_FILTER_FLAG)
    if snake_cli_overrides:
        set_flags = snake_cli_overrides.get("set_flags")
        if isinstance(set_flags, Mapping):
            allowed.update(str(flag) for flag in set_flags)
        for key in ("enable_flags", "remove_value_flags", "remove_bool_flags"):
            values = snake_cli_overrides.get(key)
            if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                allowed.update(str(flag) for flag in values)
    return allowed


def row_inner_optimizer(row: Mapping[str, str]) -> str:
    raw = str(row.get("adapt_optimizer_kind") or "powell").strip().lower()
    if raw in {"", "source", "preserve", "powell"}:
        return "POWELL"
    if raw == "spsa":
        return "SPSA"
    if raw == "qnspsa":
        return "QNSPSA"
    if raw == "rotosolve":
        return "ROTOSOLVE"
    if raw == "bfgs":
        return "BFGS"
    if raw == "cobyla":
        return "COBYLA"
    raise ValueError(f"Unsupported row adapt_optimizer_kind for SNAKE source command: {raw!r}")


def row_schur_warm_start_mode(row: Mapping[str, str], cli_mode: str) -> str:
    row_mode = str(row.get("adapt_schur_warm_start_mode") or "").strip()
    if cli_mode != "off" or not row_mode:
        return str(cli_mode)
    return row_mode


def row_snake_runtime_split_mode(row: Mapping[str, str]) -> str:
    raw = str(row.get("snake_phase3_runtime_split_mode") or "").strip()
    if raw in {"", "source", "preserve"}:
        return "source"
    if raw not in {"off", "shortlist_pauli_children_v1"}:
        raise ValueError(
            "snake_phase3_runtime_split_mode must be blank/source, off, or "
            f"shortlist_pauli_children_v1; got {raw!r}"
        )
    return raw


def row_snake_runtime_split_selection_mode(row: Mapping[str, str]) -> str:
    raw = str(
        row.get("snake_phase3_runtime_split_selection_mode")
        or row.get("phase3_runtime_split_selection_mode")
        or ""
    ).strip()
    if raw in {"", "source", "preserve"}:
        return "source"
    if raw not in {
        "proxy_child_set_preselection",
        "full_child_set_scoring",
        "archival_child_set_forward_v1",
    }:
        raise ValueError(
            "snake_phase3_runtime_split_selection_mode must be blank/source, "
            "proxy_child_set_preselection, full_child_set_scoring, or "
            f"archival_child_set_forward_v1; got {raw!r}"
        )
    return raw


def row_snake_runtime_split_child_set_symmetry_policy(row: Mapping[str, str]) -> str:
    raw = str(
        row.get("snake_phase3_runtime_split_child_set_symmetry_policy")
        or row.get("phase3_runtime_split_child_set_symmetry_policy")
        or ""
    ).strip()
    if raw in {"", "source", "preserve"}:
        return "source"
    if raw not in {"off", "parent", "hard_guard"}:
        raise ValueError(
            "snake_phase3_runtime_split_child_set_symmetry_policy must be blank/source, "
            f"off, parent, or hard_guard; got {raw!r}"
        )
    return raw


def row_snake_runtime_split_max_subset_size(row: Mapping[str, str]) -> str:
    raw = str(
        row.get("snake_phase3_runtime_split_max_subset_size")
        or row.get("phase3_runtime_split_max_subset_size")
        or ""
    ).strip()
    if raw in {"", "source", "preserve"}:
        return "source"
    if int(raw) <= 0:
        raise ValueError("snake_phase3_runtime_split_max_subset_size must be >= 1.")
    return str(int(raw))


def row_snake_child_pool_expansion(row: Mapping[str, str]) -> tuple[str, str, str]:
    mode = str(
        row.get("snake_adapt_child_pool_expansion_mode")
        or row.get("adapt_child_pool_expansion_mode")
        or ""
    ).strip()
    if mode in {"", "source", "preserve"}:
        mode = "source"
    if mode not in {"source", "off", "global_pauli_child_sets_v1", "pauli_child_sets_v1"}:
        raise ValueError(
            "snake_adapt_child_pool_expansion_mode must be blank/source, off, "
            f"global_pauli_child_sets_v1, or pauli_child_sets_v1; got {mode!r}"
        )
    policy = str(
        row.get("snake_adapt_child_pool_expansion_symmetry_policy")
        or row.get("adapt_child_pool_expansion_symmetry_policy")
        or ""
    ).strip()
    if policy in {"", "source", "preserve"}:
        policy = "source"
    if policy not in {"source", "off", "hard_guard"}:
        raise ValueError(
            "snake_adapt_child_pool_expansion_symmetry_policy must be blank/source, off, "
            f"or hard_guard; got {policy!r}"
        )
    max_subset_size = str(
        row.get("snake_adapt_child_pool_expansion_max_subset_size")
        or row.get("adapt_child_pool_expansion_max_subset_size")
        or ""
    ).strip()
    if max_subset_size in {"", "source", "preserve"}:
        max_subset_size = "source"
    elif int(max_subset_size) <= 0:
        raise ValueError("snake_adapt_child_pool_expansion_max_subset_size must be >= 1.")
    return mode, policy, max_subset_size


def row_shared_pauli_pool(row: Mapping[str, str]) -> tuple[str, str, str]:
    mode = str(row.get("shared_pauli_pool_mode") or "").strip()
    if mode in {"", "source", "preserve"}:
        mode = "source"
    mode_key = mode.lower().replace("-", "_")
    if mode_key in {"pauli_child_sets_v1", "global_pauli_child_sets_v1"}:
        mode_key = "shared_pauli_child_sets_v1"
    if mode_key not in {
        "source",
        "off",
        "shared_pauli_child_sets_v1",
        "projected_singleton_children_only_v1",
    }:
        raise ValueError(
            "shared_pauli_pool_mode must be blank/source, off, "
            "shared_pauli_child_sets_v1, or projected_singleton_children_only_v1; "
            f"got {mode!r}"
        )
    policy = str(row.get("shared_pauli_pool_symmetry_policy") or "").strip()
    if policy in {"", "source", "preserve"}:
        policy = "source"
    policy_key = policy.lower().replace("-", "_")
    if policy_key not in {"source", "off", "hard_guard"}:
        raise ValueError(
            "shared_pauli_pool_symmetry_policy must be blank/source, off, "
            f"or hard_guard; got {policy!r}"
        )
    max_subset_size = str(row.get("shared_pauli_pool_max_subset_size") or "").strip()
    if max_subset_size in {"", "source", "preserve"}:
        max_subset_size = "source"
    elif int(max_subset_size) <= 0:
        raise ValueError("shared_pauli_pool_max_subset_size must be >= 1.")
    if mode_key == "projected_singleton_children_only_v1":
        if policy_key != "hard_guard":
            raise ValueError(
                "projected_singleton_children_only_v1 requires "
                "shared_pauli_pool_symmetry_policy=hard_guard."
            )
        if max_subset_size != "1":
            raise ValueError(
                "projected_singleton_children_only_v1 requires "
                "shared_pauli_pool_max_subset_size=1."
            )
    return mode_key, policy_key, max_subset_size


def row_adapt_pool_class_filter_json(row: Mapping[str, str]) -> str:
    raw = str(
        row.get("adapt_pool_class_filter_json")
        or row.get("snake_adapt_pool_class_filter_json")
        or ""
    ).strip()
    if raw in {"", "source", "preserve"}:
        return "source"
    if raw.lower().replace("-", "_") in {"off", "none", "unfiltered"}:
        return "off"
    path = Path(raw)
    check_path = path if path.is_absolute() else ROOT / path
    if not check_path.exists():
        raise FileNotFoundError(f"adapt_pool_class_filter_json does not exist: {raw}")
    return raw


def row_snake_cli_overrides(row: Mapping[str, str]) -> dict[str, Any]:
    raw = str(row.get(SNAKE_CLI_OVERRIDES_FIELD) or "").strip()
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{SNAKE_CLI_OVERRIDES_FIELD} must be a JSON object.")
    out: dict[str, Any] = {}
    set_flags = payload.get("set_flags", {})
    if set_flags:
        if not isinstance(set_flags, Mapping):
            raise ValueError(f"{SNAKE_CLI_OVERRIDES_FIELD}.set_flags must be a JSON object.")
        out["set_flags"] = {str(flag): str(value) for flag, value in set_flags.items()}
    for key in ("enable_flags", "remove_value_flags", "remove_bool_flags"):
        values = payload.get(key, [])
        if values:
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise ValueError(f"{SNAKE_CLI_OVERRIDES_FIELD}.{key} must be a JSON array.")
            out[key] = [str(flag) for flag in values]
    return out


def apply_snake_cli_overrides(args: list[str], overrides: Mapping[str, Any]) -> None:
    for flag in overrides.get("remove_value_flags", []) or []:
        remove_flag(args, str(flag))
    for flag in overrides.get("remove_bool_flags", []) or []:
        remove_bool_flag(args, str(flag))
    set_flags = overrides.get("set_flags") or {}
    if isinstance(set_flags, Mapping):
        for flag, value in set_flags.items():
            set_flag(args, str(flag), str(value))
    for flag in overrides.get("enable_flags", []) or []:
        flag = str(flag)
        remove_bool_flag(args, flag)
        args.append(flag)


def apply_snake_runtime_split_override(
    args: list[str],
    mode: str,
    *,
    selection_mode: str = "source",
    child_set_symmetry_policy: str = "source",
    max_subset_size: str = "source",
) -> None:
    if mode == "source":
        if selection_mode != "source" or child_set_symmetry_policy != "source" or max_subset_size != "source":
            raise ValueError("runtime-split child-set overrides require a runtime split mode override.")
        return
    if mode == "off" and selection_mode != "source":
        raise ValueError("runtime-split selection-mode override requires runtime split mode to be active.")
    set_flag(args, SNAKE_RUNTIME_SPLIT_FLAG, mode)
    remove_bool_flag(args, SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG)
    if mode != "off":
        args.append(SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG)
    if selection_mode != "source":
        set_flag(args, SNAKE_RUNTIME_SPLIT_SELECTION_FLAG, selection_mode)
    if child_set_symmetry_policy != "source":
        set_flag(args, SNAKE_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_FLAG, child_set_symmetry_policy)
    if max_subset_size != "source":
        set_flag(args, SNAKE_RUNTIME_SPLIT_MAX_SUBSET_FLAG, max_subset_size)


def apply_snake_child_pool_expansion_override(
    args: list[str],
    *,
    mode: str,
    symmetry_policy: str,
    max_subset_size: str,
) -> None:
    if mode == "source":
        if symmetry_policy != "source" or max_subset_size != "source":
            raise ValueError("child pool symmetry/max-subset overrides require a child pool mode override.")
        return
    set_flag(args, SNAKE_CHILD_POOL_EXPANSION_FLAG, mode)
    if mode not in {"", "off", "none", "false", "0", "disabled"}:
        set_flag(args, SNAKE_RUNTIME_SPLIT_FLAG, "off")
        remove_bool_flag(args, SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG)
    if symmetry_policy != "source":
        set_flag(args, SNAKE_CHILD_POOL_EXPANSION_SYMMETRY_FLAG, symmetry_policy)
    if max_subset_size != "source":
        set_flag(args, SNAKE_CHILD_POOL_EXPANSION_MAX_SUBSET_FLAG, max_subset_size)


def apply_shared_pauli_pool_override(
    args: list[str],
    *,
    mode: str,
    symmetry_policy: str,
    max_subset_size: str,
) -> None:
    if mode == "source":
        if symmetry_policy != "source" or max_subset_size != "source":
            raise ValueError("shared pool symmetry/max-subset overrides require a shared pool mode override.")
        return
    set_flag(args, SHARED_PAULI_POOL_FLAG, mode)
    if symmetry_policy != "source":
        set_flag(args, SHARED_PAULI_POOL_SYMMETRY_FLAG, symmetry_policy)
    if max_subset_size != "source":
        set_flag(args, SHARED_PAULI_POOL_MAX_SUBSET_FLAG, max_subset_size)
    if mode not in {"", "off", "none", "false", "0", "disabled"}:
        set_flag(args, SNAKE_RUNTIME_SPLIT_FLAG, "off")
        remove_bool_flag(args, SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG)
        set_flag(args, SNAKE_CHILD_POOL_EXPANSION_FLAG, "off")


def build_snake_source_locked_command(
    row: Mapping[str, str],
    output_root: Path,
    *,
    schur_warm_start_mode: str | None = None,
    inner_optimizer: str | None = None,
    runtime_worker_overrides: Mapping[str, int] | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    raw = str(row.get("source_command_args_json") or "")
    if not raw:
        raise ValueError("SNAKE row has no source_command_args_json")
    source_json_raw = str(row.get("source_json") or "").strip()
    if not source_json_raw:
        raise ValueError("SNAKE row has no source_json")
    source_json = resolve_row_path(source_json_raw, Path(source_json_raw))
    if not source_json.exists():
        raise FileNotFoundError(f"SNAKE source_json does not exist: {source_json}")
    source_cmd = [str(x) for x in json.loads(raw)]
    if not source_cmd:
        raise ValueError("SNAKE source command args are empty")
    source_cmd[0] = sys.executable
    source_cmd = replace_repo_paths(source_cmd)
    effective_cmd = list(source_cmd)
    effective_schur_warm_start_mode = row_schur_warm_start_mode(
        row,
        "off" if schur_warm_start_mode is None else str(schur_warm_start_mode),
    )
    effective_inner_optimizer = row_inner_optimizer(row) if inner_optimizer is None else str(inner_optimizer)
    snake_runtime_split_mode = row_snake_runtime_split_mode(row)
    snake_runtime_split_selection_mode = row_snake_runtime_split_selection_mode(row)
    snake_runtime_split_child_set_symmetry_policy = row_snake_runtime_split_child_set_symmetry_policy(row)
    snake_runtime_split_max_subset_size = row_snake_runtime_split_max_subset_size(row)
    (
        snake_child_pool_expansion_mode,
        snake_child_pool_expansion_symmetry_policy,
        snake_child_pool_expansion_max_subset_size,
    ) = row_snake_child_pool_expansion(row)
    (
        shared_pauli_pool_mode,
        shared_pauli_pool_symmetry_policy,
        shared_pauli_pool_max_subset_size,
    ) = row_shared_pauli_pool(row)
    adapt_pool_class_filter_json = row_adapt_pool_class_filter_json(row)
    snake_cli_overrides = row_snake_cli_overrides(row)
    apply_snake_source_locked_overrides(
        effective_cmd,
        row=row,
        output_root=output_root,
        schur_warm_start_mode=effective_schur_warm_start_mode,
        inner_optimizer=effective_inner_optimizer,
        snake_runtime_split_mode=snake_runtime_split_mode,
        snake_runtime_split_selection_mode=snake_runtime_split_selection_mode,
        snake_runtime_split_child_set_symmetry_policy=snake_runtime_split_child_set_symmetry_policy,
        snake_runtime_split_max_subset_size=snake_runtime_split_max_subset_size,
        snake_child_pool_expansion_mode=snake_child_pool_expansion_mode,
        snake_child_pool_expansion_symmetry_policy=snake_child_pool_expansion_symmetry_policy,
        snake_child_pool_expansion_max_subset_size=snake_child_pool_expansion_max_subset_size,
        shared_pauli_pool_mode=shared_pauli_pool_mode,
        shared_pauli_pool_symmetry_policy=shared_pauli_pool_symmetry_policy,
        shared_pauli_pool_max_subset_size=shared_pauli_pool_max_subset_size,
        adapt_pool_class_filter_json=adapt_pool_class_filter_json,
        runtime_worker_overrides=runtime_worker_overrides,
        snake_cli_overrides=snake_cli_overrides,
    )
    audit = audit_snake_source_locked_command(
        source_cmd=source_cmd,
        effective_cmd=effective_cmd,
        row=row,
        output_root=output_root,
        source_json=source_json,
        schur_warm_start_mode=effective_schur_warm_start_mode,
        inner_optimizer=effective_inner_optimizer,
        snake_runtime_split_mode=snake_runtime_split_mode,
        snake_runtime_split_selection_mode=snake_runtime_split_selection_mode,
        snake_runtime_split_child_set_symmetry_policy=snake_runtime_split_child_set_symmetry_policy,
        snake_runtime_split_max_subset_size=snake_runtime_split_max_subset_size,
        snake_child_pool_expansion_mode=snake_child_pool_expansion_mode,
        snake_child_pool_expansion_symmetry_policy=snake_child_pool_expansion_symmetry_policy,
        snake_child_pool_expansion_max_subset_size=snake_child_pool_expansion_max_subset_size,
        shared_pauli_pool_mode=shared_pauli_pool_mode,
        shared_pauli_pool_symmetry_policy=shared_pauli_pool_symmetry_policy,
        shared_pauli_pool_max_subset_size=shared_pauli_pool_max_subset_size,
        adapt_pool_class_filter_json=adapt_pool_class_filter_json,
        runtime_worker_overrides=runtime_worker_overrides,
        snake_cli_overrides=snake_cli_overrides,
    )
    return source_cmd, effective_cmd, audit


def apply_snake_source_locked_overrides(
    args: list[str],
    *,
    row: Mapping[str, str],
    output_root: Path,
    schur_warm_start_mode: str = "off",
    inner_optimizer: str = "SPSA",
    snake_runtime_split_mode: str = "source",
    snake_runtime_split_selection_mode: str = "source",
    snake_runtime_split_child_set_symmetry_policy: str = "source",
    snake_runtime_split_max_subset_size: str = "source",
    snake_child_pool_expansion_mode: str = "source",
    snake_child_pool_expansion_symmetry_policy: str = "source",
    snake_child_pool_expansion_max_subset_size: str = "source",
    shared_pauli_pool_mode: str = "source",
    shared_pauli_pool_symmetry_policy: str = "source",
    shared_pauli_pool_max_subset_size: str = "source",
    adapt_pool_class_filter_json: str = "source",
    runtime_worker_overrides: Mapping[str, int] | None = None,
    snake_cli_overrides: Mapping[str, Any] | None = None,
) -> None:
    max_depth = str(row.get("max_depth") or "30")
    set_flag(args, "--adapt-max-depth", max_depth)
    set_flag(args, "--adapt-maxiter", str(row["budget"]))
    set_flag(args, "--adapt-final-refit-maxiter", str(row["budget"]))
    set_flag(args, ADAPT_INNER_OPTIMIZER_FLAG, str(inner_optimizer or "SPSA").strip().upper())
    if force_depth30_no_early_stop(row):
        set_flag(args, "--adapt-segment-target-depth", max_depth)
        set_flag(args, "--adapt-segment-max-new-admissions", max_depth)
        set_flag(args, "--adapt-drop-floor", "-1")
        set_flag(args, "--adapt-drop-patience", "0")
        set_flag(args, "--adapt-drop-min-depth", "0")
        set_flag(args, "--adapt-grad-floor", "-1")
        remove_flag(args, "--adapt-benchmark-target-abs-delta-e")
        remove_flag(args, "--adapt-benchmark-target-reference-energy")
    set_flag(args, "--output-json", str(output_root / "json" / "result.json"))
    set_flag(args, "--adapt-current-json", str(output_root / "current.json"))
    if schur_warm_start_mode != "off":
        set_flag(args, SCHUR_WARM_START_FLAG, schur_warm_start_mode)
    if runtime_worker_overrides:
        for flag, value in runtime_worker_overrides.items():
            if flag not in RUNTIME_WORKER_OVERRIDE_FLAGS:
                raise ValueError(f"Unsupported runtime worker override flag: {flag}")
            set_flag(args, flag, str(int(value)))
    apply_snake_runtime_split_override(
        args,
        snake_runtime_split_mode,
        selection_mode=snake_runtime_split_selection_mode,
        child_set_symmetry_policy=snake_runtime_split_child_set_symmetry_policy,
        max_subset_size=snake_runtime_split_max_subset_size,
    )
    apply_snake_child_pool_expansion_override(
        args,
        mode=snake_child_pool_expansion_mode,
        symmetry_policy=snake_child_pool_expansion_symmetry_policy,
        max_subset_size=snake_child_pool_expansion_max_subset_size,
    )
    apply_shared_pauli_pool_override(
        args,
        mode=shared_pauli_pool_mode,
        symmetry_policy=shared_pauli_pool_symmetry_policy,
        max_subset_size=shared_pauli_pool_max_subset_size,
    )
    if adapt_pool_class_filter_json == "off":
        remove_flag(args, ADAPT_POOL_CLASS_FILTER_FLAG)
    elif adapt_pool_class_filter_json != "source":
        set_flag(args, ADAPT_POOL_CLASS_FILTER_FLAG, adapt_pool_class_filter_json)
    if snake_cli_overrides:
        apply_snake_cli_overrides(args, snake_cli_overrides)
    repair_legacy_resume_scaffold_if_needed(args, output_root=output_root)


def resume_scaffold_repair_flag_change_allowed(
    *,
    changed_flags: Sequence[Mapping[str, Any]],
    output_root: Path,
) -> bool:
    for item in changed_flags:
        if str(item.get("flag") or "") != RESUME_SCAFFOLD_FLAG:
            continue
        effective = item.get("effective")
        if not isinstance(effective, Sequence) or isinstance(effective, (str, bytes)):
            return False
        if len(effective) != 1:
            return False
        repaired_path = Path(str(effective[0]))
        try:
            repaired_path.relative_to(output_root)
        except ValueError:
            return False
        if repaired_path.name != RESUME_SCAFFOLD_REPAIR_BASENAME or not repaired_path.exists():
            return False
        try:
            payload = json.loads(repaired_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        repair = payload.get("resume_scaffold_repair")
        if not isinstance(repair, Mapping):
            return False
        if str(repair.get("schema") or "") != "paper_i_hh_resume_scaffold_repair_v1":
            return False
        if str(repair.get("repair") or "") != "strict_reconstructed_initial_state":
            return False
        return True
    return False


def audit_snake_source_locked_command(
    *,
    source_cmd: Sequence[str],
    effective_cmd: Sequence[str],
    row: Mapping[str, str],
    output_root: Path,
    source_json: Path,
    schur_warm_start_mode: str = "off",
    inner_optimizer: str = "SPSA",
    snake_runtime_split_mode: str = "source",
    snake_runtime_split_selection_mode: str = "source",
    snake_runtime_split_child_set_symmetry_policy: str = "source",
    snake_runtime_split_max_subset_size: str = "source",
    snake_child_pool_expansion_mode: str = "source",
    snake_child_pool_expansion_symmetry_policy: str = "source",
    snake_child_pool_expansion_max_subset_size: str = "source",
    shared_pauli_pool_mode: str = "source",
    shared_pauli_pool_symmetry_policy: str = "source",
    shared_pauli_pool_max_subset_size: str = "source",
    adapt_pool_class_filter_json: str = "source",
    runtime_worker_overrides: Mapping[str, int] | None = None,
    snake_cli_overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    expected_cmd = list(source_cmd)
    apply_snake_source_locked_overrides(
        expected_cmd,
        row=row,
        output_root=output_root,
        schur_warm_start_mode=schur_warm_start_mode,
        inner_optimizer=inner_optimizer,
        snake_runtime_split_mode=snake_runtime_split_mode,
        snake_runtime_split_selection_mode=snake_runtime_split_selection_mode,
        snake_runtime_split_child_set_symmetry_policy=snake_runtime_split_child_set_symmetry_policy,
        snake_runtime_split_max_subset_size=snake_runtime_split_max_subset_size,
        snake_child_pool_expansion_mode=snake_child_pool_expansion_mode,
        snake_child_pool_expansion_symmetry_policy=snake_child_pool_expansion_symmetry_policy,
        snake_child_pool_expansion_max_subset_size=snake_child_pool_expansion_max_subset_size,
        shared_pauli_pool_mode=shared_pauli_pool_mode,
        shared_pauli_pool_symmetry_policy=shared_pauli_pool_symmetry_policy,
        shared_pauli_pool_max_subset_size=shared_pauli_pool_max_subset_size,
        adapt_pool_class_filter_json=adapt_pool_class_filter_json,
        runtime_worker_overrides=runtime_worker_overrides,
        snake_cli_overrides=snake_cli_overrides,
    )

    allowed_flag_changes = _source_lock_allowed_flag_changes(
        schur_warm_start_mode=schur_warm_start_mode,
        inner_optimizer=inner_optimizer,
        runtime_worker_overrides=runtime_worker_overrides,
        child_pool_expansion_mode=snake_child_pool_expansion_mode,
        shared_pauli_pool_mode=shared_pauli_pool_mode,
        adapt_pool_class_filter_json=adapt_pool_class_filter_json,
        snake_cli_overrides=snake_cli_overrides,
    )
    if snake_runtime_split_mode != "source":
        allowed_flag_changes.update(
            {
                SNAKE_RUNTIME_SPLIT_FLAG,
                SNAKE_RUNTIME_SPLIT_ARCHIVAL_FLAG,
                SNAKE_RUNTIME_SPLIT_SELECTION_FLAG,
                SNAKE_RUNTIME_SPLIT_CHILD_SET_SYMMETRY_FLAG,
                SNAKE_RUNTIME_SPLIT_MAX_SUBSET_FLAG,
            }
        )
    changed_flags = changed_flag_values(source_cmd, effective_cmd)
    resume_repair_allowed = resume_scaffold_repair_flag_change_allowed(
        changed_flags=changed_flags,
        output_root=output_root,
    )
    if resume_repair_allowed:
        allowed_flag_changes.add(RESUME_SCAFFOLD_FLAG)
    non_allowed = [item for item in changed_flags if item["flag"] not in allowed_flag_changes]
    source_flags = flag_values(source_cmd)
    effective_flags = flag_values(effective_cmd)
    preserved_flags = {
        flag: {
            "source": source_flags.get(flag, []),
            "effective": effective_flags.get(flag, []),
            "preserved": source_flags.get(flag, []) == effective_flags.get(flag, []),
        }
        for flag in sorted(SOURCE_LOCK_PRESERVED_FLAGS)
        if flag in source_flags or flag in effective_flags
    }
    source_positionals = positional_tokens(source_cmd)
    effective_positionals = positional_tokens(effective_cmd)
    positional_match = source_positionals == effective_positionals
    exact_expected_match = list(effective_cmd) == expected_cmd
    status = "pass" if exact_expected_match and not non_allowed and positional_match else "blocked"
    return {
        "schema": "paper_i_hh_spsa_budget_ladder_snake_source_lock_command_audit_v1",
        "status": status,
        "record_id": row.get("record_id"),
        "display_regime": row.get("display_regime"),
        "method_key": row.get("method_key"),
        "engine_key": row.get("engine_key"),
        "budget": row.get("budget"),
        "source_json": str(source_json),
        "allowed_flag_changes": sorted(allowed_flag_changes),
        "diagnostic_schur_warm_start_mode": schur_warm_start_mode,
        "diagnostic_inner_optimizer": str(inner_optimizer or "SPSA").strip().upper(),
        "diagnostic_snake_phase3_runtime_split_mode": snake_runtime_split_mode,
        "diagnostic_snake_phase3_runtime_split_selection_mode": snake_runtime_split_selection_mode,
        "diagnostic_snake_phase3_runtime_split_child_set_symmetry_policy": (
            snake_runtime_split_child_set_symmetry_policy
        ),
        "diagnostic_snake_phase3_runtime_split_max_subset_size": snake_runtime_split_max_subset_size,
        "diagnostic_snake_adapt_child_pool_expansion_mode": snake_child_pool_expansion_mode,
        "diagnostic_snake_adapt_child_pool_expansion_symmetry_policy": snake_child_pool_expansion_symmetry_policy,
        "diagnostic_snake_adapt_child_pool_expansion_max_subset_size": snake_child_pool_expansion_max_subset_size,
        "diagnostic_shared_pauli_pool_mode": shared_pauli_pool_mode,
        "diagnostic_shared_pauli_pool_symmetry_policy": shared_pauli_pool_symmetry_policy,
        "diagnostic_shared_pauli_pool_max_subset_size": shared_pauli_pool_max_subset_size,
        "diagnostic_adapt_pool_class_filter_json": adapt_pool_class_filter_json,
        "diagnostic_runtime_worker_overrides": dict(runtime_worker_overrides or {}),
        "diagnostic_snake_cli_overrides": dict(snake_cli_overrides or {}),
        "changed_flags": changed_flags,
        "non_allowed_flag_changes": non_allowed,
        "resume_scaffold_repair_allowed": bool(resume_repair_allowed),
        "preserved_flags": preserved_flags,
        "source_positionals": source_positionals,
        "effective_positionals": effective_positionals,
        "positional_match": positional_match,
        "exact_expected_match": exact_expected_match,
        "source_command": list(source_cmd),
        "expected_command": expected_cmd,
        "effective_command": list(effective_cmd),
        "spsa_engine_env_change": {
            "env": "ADAPT_SPSA_REFIT_ENGINE",
            "effective": row.get("spsa_refit_engine"),
            "allowed": True,
        },
        "forced_depth30_no_early_stop": bool(force_depth30_no_early_stop(row)),
        "suppress_drop_plateau_terminal_stop": bool(suppress_drop_plateau_terminal_stop(row)),
    }


def common_env(row: Mapping[str, str], output_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    optimizer_kind = str(row.get("adapt_optimizer_kind") or "powell").strip().lower()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "STATIC_ADAPT_HH_POOL_CACHE": "disk",
            "STATIC_ADAPT_HH_POOL_CACHE_SCOPE": "paper_i_holstein_sector",
            "STATIC_ADAPT_CANDIDATE_RECORD_CACHE": "disk",
            "HOLSTEIN_SKIP_MATPLOTLIB_IMPORT": "1",
        }
    )
    if optimizer_kind == "spsa":
        env["ADAPT_SPSA_REFIT_ENGINE"] = str(row["spsa_refit_engine"])
        env["GENERIC_STATIC_TABLE_ADAPT_SPSA_REFIT_ENGINE"] = str(row["spsa_refit_engine"])
    env.setdefault("STATIC_ADAPT_HH_POOL_CACHE_DIR", str(ROOT / "raw_outputs" / "cache" / "hh_pool_cache_v1"))
    env.setdefault(
        "STATIC_ADAPT_CANDIDATE_RECORD_CACHE_DIR",
        str(ROOT / "raw_outputs" / "cache" / "adapt_candidate_record_cache_v1"),
    )
    env["PAPER_I_HH_SPSA_BUDGET_CELL_OUTPUT_ROOT"] = str(output_root)
    if suppress_drop_plateau_terminal_stop(row):
        env["STATIC_ADAPT_SUPPRESS_DROP_PLATEAU_TERMINAL_STOP"] = "1"
    for field, env_name in (
        ("resource_qubit_cap", "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"),
        ("resource_pool_term_cap", "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"),
    ):
        value = str(row.get(field) or "").strip()
        if value:
            env[env_name] = value
    return env


def append_geo_env(row: Mapping[str, str], output_root: Path) -> dict[str, str]:
    env = common_env(row, output_root)
    generic_adapt_stop_policy = str(row.get("generic_adapt_stop_policy") or "").strip()
    optimizer_kind = str(row.get("adapt_optimizer_kind") or "powell").strip().lower()
    hh_pool_profile = str(row.get("hh_adaptive_pool_profile") or "").strip()
    if not hh_pool_profile and str(row.get("pool_contract") or "").strip() == "full_meta_unfiltered":
        hh_pool_profile = "full_meta_unfiltered"
    hh_class_filter = str(row.get("adapt_pool_class_filter_json") or "").strip()
    env.update(
        {
            "TABLE_I_STATIC_SUITE_PROFILE": str(row["suite_profile"]),
            "GENERIC_STATIC_TABLE_ADAPT_OPTIMIZER_KIND": optimizer_kind,
            "GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH": str(row.get("max_depth") or "13"),
            # Both Paper-I comparators score with replacement.  Geo applies
            # its adjacent-repeat rule only after the full-pool selector solve.
            "GENERIC_STATIC_TABLE_PHASE3_ADAPT_ALLOW_REPEATS": "true",
            "GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAXITER": str(row["budget"]),
            "GENERIC_STATIC_TABLE_PHASE3_REFIT_MAXITER": str(row["budget"]),
            "GENERIC_STATIC_TABLE_PHASE3_FINAL_MAXITER": str(row["budget"]),
            "GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY": str(row["same_cutoff_exact_gs_energy"]),
            "GENERIC_STATIC_TABLE_EXACT_REFERENCE_ENERGY": str(row["exact_reference_energy"]),
            "GENERIC_STATIC_TABLE_EXACT_REFERENCE_N_PH_MAX": str(row["exact_reference_n_ph_max"]),
            "GENERIC_STATIC_TABLE_PRIMARY_ENERGY_METRIC": "same_cutoff_abs_delta_e",
            "GENERIC_STATIC_TABLE_SAME_CUTOFF_ERROR_ROLE": "primary",
            "GENERIC_STATIC_TABLE_PROGRESS_JSONL_PATH": str(output_root / "adapt_iteration_progress.jsonl"),
            "GENERIC_STATIC_TABLE_PROGRESS_STDOUT": "1",
            "GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS": "0.0002,0.001,0.01",
        }
    )
    if hh_pool_profile:
        env["GENERIC_STATIC_TABLE_HH_ADAPTIVE_POOL_PROFILE"] = hh_pool_profile
    if hh_class_filter and hh_class_filter not in {"source", "preserve"}:
        env["GENERIC_STATIC_TABLE_HH_FULL_META_CLASS_FILTER_JSON"] = hh_class_filter
    exact_fidelity_max_qubits = str(row.get("exact_fidelity_max_qubits") or "").strip()
    if exact_fidelity_max_qubits:
        env["GENERIC_STATIC_TABLE_EXACT_FIDELITY_MAX_QUBITS"] = exact_fidelity_max_qubits
    if force_depth30_no_early_stop(row) or generic_adapt_stop_policy == "fixed_horizon_no_target_v1":
        env["GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"] = ""
        env["PAPER_I_HH_FORCED_DEPTH30_NO_EARLY_STOP"] = "1"
    else:
        env["GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"] = "0.0002"
    if optimizer_kind == "spsa":
        env["GENERIC_STATIC_TABLE_ADAPT_SPSA_MAXITER"] = str(row["budget"])
        env["GENERIC_STATIC_TABLE_ADAPT_SPSA_SEED"] = str(row.get("adapt_spsa_seed") or "42")
    if optimizer_kind == "spsa" and str(row["suite_profile"]) == "paper_i_main_tables_spsa":
        env["GENERIC_STATIC_TABLE_OPTIMIZER_PROFILE"] = "paper_i_main_tables_spsa_v1"
    explicit_seed_raw = row.get("seed")
    legacy_seed_raw = row.get("adapt_seed")
    if explicit_seed_raw not in {None, ""} and legacy_seed_raw not in {None, ""}:
        if int(explicit_seed_raw) != int(legacy_seed_raw):
            raise ValueError(
                "Append/Geo comparator seed and adapt_seed disagree; "
                "refusing ambiguous stochastic identity."
            )
    comparator_seed_raw = explicit_seed_raw
    if comparator_seed_raw in {None, ""}:
        comparator_seed_raw = legacy_seed_raw
    comparator_seed = str(
        "" if comparator_seed_raw in {None, ""} else comparator_seed_raw
    ).strip()
    if comparator_seed:
        if int(comparator_seed) < 0:
            raise ValueError("Append/Geo comparator seed must be >= 0.")
        env["GENERIC_STATIC_TABLE_ADAPT_SEED"] = str(int(comparator_seed))
    if generic_adapt_stop_policy:
        env["GENERIC_STATIC_TABLE_GENERIC_ADAPT_STOP_POLICY"] = generic_adapt_stop_policy
    if optimizer_kind == "spsa":
        for field, env_name in (
            ("adapt_spsa_a", "GENERIC_STATIC_TABLE_ADAPT_SPSA_A"),
            ("adapt_spsa_c", "GENERIC_STATIC_TABLE_ADAPT_SPSA_C"),
            ("adapt_spsa_alpha", "GENERIC_STATIC_TABLE_ADAPT_SPSA_ALPHA"),
            ("adapt_spsa_gamma", "GENERIC_STATIC_TABLE_ADAPT_SPSA_GAMMA"),
            ("adapt_spsa_big_a", "GENERIC_STATIC_TABLE_ADAPT_SPSA_BIG_A"),
            ("adapt_spsa_a", "GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_A"),
            ("adapt_spsa_c", "GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_C"),
            ("adapt_spsa_alpha", "GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_ALPHA"),
            ("adapt_spsa_gamma", "GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_GAMMA"),
            ("adapt_spsa_big_a", "GENERIC_STATIC_TABLE_PHASE3_ADAPT_SPSA_BIG_A"),
        ):
            value = str(row.get(field) or "").strip()
            if value:
                env[env_name] = value
    for field, env_name in (
        ("generic_adapt_runtime_split_mode", "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MODE"),
        (
            "generic_adapt_runtime_split_symmetry_policy",
            "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_SYMMETRY_POLICY",
        ),
        (
            "generic_adapt_runtime_split_max_subset_size",
            "GENERIC_STATIC_TABLE_GENERIC_ADAPT_RUNTIME_SPLIT_MAX_SUBSET_SIZE",
        ),
        ("shared_pauli_pool_mode", "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MODE"),
        (
            "shared_pauli_pool_symmetry_policy",
            "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_SYMMETRY_POLICY",
        ),
        (
            "shared_pauli_pool_max_subset_size",
            "GENERIC_STATIC_TABLE_SHARED_PAULI_POOL_MAX_SUBSET_SIZE",
        ),
    ):
        value = str(row.get(field) or "").strip()
        if value:
            env[env_name] = value
    return env


def run_append_geo(row: Mapping[str, str], output_root: Path) -> tuple[list[str], dict[str, str], int]:
    continuation_text = " ".join(
        str(row.get(key) or "")
        for key in (
            "batch_id",
            "schedule_source_note",
            "source_contract_note",
        )
    ).lower()
    if "continuation" in continuation_text:
        raise RuntimeError(
            "Append/Geo comparator continuation is blocked: this runner does not load the declared "
            "source operators, theta, history, and cumulative query ledger. Start a fresh corrected "
            "fixed-horizon run or implement a hash-validated continuation contract first."
        )
    result_dir = output_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "pipelines.exact_bench.generic_static_benchmark",
        "--run-single",
        "--family",
        "hh",
        "--case-id",
        str(row["case_id"]),
        "--algorithm-id",
        str(row["algorithm_id"]),
        "--output-dir",
        str(result_dir),
    ]
    env = append_geo_env(row, output_root)
    cmd_out, env_overlay, returncode = run_subprocess(cmd, env, output_root)
    if returncode == 0:
        summary = result_summary_from_artifacts(row, output_root)
        if summary.get("status") != "ok":
            returncode = 3
    return cmd_out, env_overlay, returncode


def run_snake(
    row: Mapping[str, str],
    output_root: Path,
    *,
    schur_warm_start_mode: str = "off",
    runtime_worker_overrides: Mapping[str, int] | None = None,
) -> tuple[list[str], dict[str, str], int]:
    inner_optimizer = row_inner_optimizer(row)
    _source_cmd, cmd, audit = build_snake_source_locked_command(
        row,
        output_root,
        schur_warm_start_mode=schur_warm_start_mode,
        inner_optimizer=inner_optimizer,
        runtime_worker_overrides=runtime_worker_overrides,
    )
    audit_path = resolve_row_path(str(row.get("source_lock_command_audit_rel") or ""), output_root / "source_lock_command_audit.json")
    write_json(audit_path, audit)
    if audit.get("status") != "pass":
        raise RuntimeError(f"SNAKE source-lock command audit failed for {row.get('record_id')}: {audit_path}")
    env = common_env(row, output_root)
    cmd_out, env_overlay, returncode = run_subprocess(cmd, env, output_root)
    if returncode == 0:
        sidecar = write_snake_algorithmic_work_sidecar(row, output_root)
        fair_sidecar = write_snake_fair_shot_work_sidecar(row, output_root)
        if sidecar.get("S_alg_status") != "ok" or fair_sidecar.get("S_fair_status") != "ok":
            returncode = 3
    return cmd_out, env_overlay, returncode


def _finite_nonnegative(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0.0:
        return None
    return float(parsed)


def _phase_map(summary: Mapping[str, Any]) -> Mapping[str, Any] | None:
    by_phase = summary.get("by_phase")
    if isinstance(by_phase, Mapping):
        return by_phase
    per_phase = summary.get("per_phase")
    if isinstance(per_phase, Mapping):
        return per_phase
    return None


def _phase_records_with_group_keys(phase_map: Mapping[str, Any], phase: str) -> tuple[float | None, dict[str, Any]]:
    entry = phase_map.get(phase)
    if entry is None:
        return 0.0, {"phase": phase, "status": "absent_zero"}
    if not isinstance(entry, Mapping):
        return None, {"phase": phase, "status": "invalid_phase_payload"}
    value = _finite_nonnegative(entry.get("records_with_group_keys"))
    if value is None:
        return None, {"phase": phase, "status": "invalid_records_with_group_keys"}
    return float(value), {"phase": phase, "status": "ok", "records_with_group_keys": float(value)}


def _explicit_phase0_count(phase_map: Mapping[str, Any]) -> tuple[float | None, dict[str, Any]]:
    present: list[tuple[str, float]] = []
    details: dict[str, Any] = {}
    for phase in ("phase0", "phase_0"):
        value, detail = _phase_records_with_group_keys(phase_map, phase)
        details[phase] = detail
        if value is None:
            return None, {"status": "invalid_explicit_controller_phase0", "phases": details}
        if value > 0.0:
            present.append((phase, float(value)))
    if len(present) > 1:
        return None, {
            "status": "ambiguous_duplicate_explicit_controller_phase0",
            "positive_phase0_aliases": {phase: value for phase, value in present},
            "phases": details,
        }
    if present:
        phase, value = present[0]
        return float(value), {
            "status": "explicit_controller_phase0",
            "phase": phase,
            "records_with_group_keys": float(value),
            "phases": details,
        }
    return 0.0, {"status": "absent_zero", "phases": details}


def _positive_unassigned_phase_work(phase_map: Mapping[str, Any]) -> dict[str, float]:
    allowed = {"phase0", "phase_0", "phase1", "phase2", "phase3"}
    out: dict[str, float] = {}
    for phase, entry in phase_map.items():
        if str(phase) in allowed or not isinstance(entry, Mapping):
            continue
        value = _finite_nonnegative(entry.get("records_with_group_keys"))
        if value is not None and value > 0.0:
            out[str(phase)] = float(value)
    return out


def _history_refit_nfev(history: Any) -> tuple[float | None, dict[str, Any]]:
    if not isinstance(history, list):
        return None, {"status": "missing_history"}
    total = 0.0
    missing: list[int] = []
    for idx, row in enumerate(history):
        if not isinstance(row, Mapping):
            missing.append(int(idx))
            continue
        value = _finite_nonnegative(row.get("nfev_opt"))
        if value is None:
            value = _finite_nonnegative(row.get("optimizer_nfev"))
        if value is None:
            missing.append(int(idx))
            continue
        total += float(value)
    if missing:
        return None, {"status": "missing_history_nfev", "indices": missing[:20]}
    return float(total), {"status": "ok", "history_count": int(len(history)), "nfev": float(total)}


def _optional_refit_nfev(adapt_payload: Mapping[str, Any], key: str) -> tuple[float | None, dict[str, Any]]:
    info = adapt_payload.get(key)
    if not isinstance(info, Mapping):
        return None, {"status": "missing_refit_payload", "field": key}
    value = _finite_nonnegative(info.get("nfev"))
    if value is not None:
        return float(value), {"status": "ok", "field": key, "nfev": float(value)}
    if info.get("executed") is False or info.get("attempted") is False:
        return 0.0, {"status": "explicit_not_executed_zero", "field": key}
    return None, {"status": "missing_refit_nfev", "field": key}


def _candidate_work_ledger_audit(summary: Mapping[str, Any]) -> dict[str, Any]:
    status = str(summary.get("candidate_work_ledger_status") or "")
    schema = str(summary.get("candidate_work_ledger_schema") or "")
    missing = _finite_nonnegative(summary.get("candidate_work_missing_event_count"))
    events = _finite_nonnegative(summary.get("candidate_work_event_count"))
    ok = status == "explicit_candidate_work_ledger_v1" and schema == "controller_candidate_work_ledger_v1" and missing in {None, 0.0}
    return {
        "schema": "snake_candidate_work_ledger_audit_v1",
        "status": "ok" if ok else "legacy_lower_bound_missing_candidate_ledger",
        "candidate_work_ledger_status": status or None,
        "candidate_work_ledger_schema": schema or None,
        "candidate_work_event_count": None if events is None else int(events),
        "candidate_work_missing_event_count": None if missing is None else int(missing),
        "candidate_count_total": summary.get("candidate_count_total"),
        "evaluated_count_total": summary.get("evaluated_count_total"),
        "pre_shortlist_count_total": summary.get("pre_shortlist_count_total"),
        "shortlist_size_total": summary.get("shortlist_size_total"),
        "retained_count_total": summary.get("retained_count_total"),
        "rejected_count_total": summary.get("rejected_count_total"),
        "candidate_work_ledger_scope": summary.get("candidate_work_ledger_scope"),
        "candidate_work_ledger_scopes": summary.get("candidate_work_ledger_scopes"),
    }


def _snake_algorithmic_work_from_payload_fast(
    source_payload: Mapping[str, Any],
    *,
    source_label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    return snake_algorithmic_work_from_payload(
        source_payload,
        scope="terminal",
        source_label=source_label,
    )


def write_snake_algorithmic_work_sidecar(row: Mapping[str, str], output_root: Path) -> dict[str, Any]:
    result_path = resolve_result_path(row, output_root)
    sidecar_path = resolve_row_path(
        str(row.get("snake_algorithmic_work_rel") or ""),
        output_root / "snake_algorithmic_work.json",
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    work, audit = _snake_algorithmic_work_from_payload_fast(payload, source_label=str(result_path))
    alg_work = work.get("algorithmic_measurement_work") if isinstance(work, Mapping) else None
    if not isinstance(alg_work, Mapping):
        alg_work = {}
    status = str(work.get("S_alg_status") or alg_work.get("status") or audit.get("status") or "unknown")
    sidecar = {
        "schema": "paper_i_hh_spsa_budget_ladder_snake_algorithmic_work_sidecar_v1",
        "record_id": row.get("record_id"),
        "display_regime": row.get("display_regime"),
        "engine_key": row.get("engine_key"),
        "budget": row.get("budget"),
        "result_json": str(result_path),
        "S_alg": work.get("S_alg"),
        "S_alg_status": status,
        "S_alg_missing_reason": work.get("S_alg_missing_reason"),
        "component_counts": alg_work.get("components"),
        "component_sources": alg_work.get("component_sources"),
        "algorithmic_measurement_work": alg_work,
        "table_i_measurement_event_ledger": work.get("table_i_measurement_event_ledger"),
        "reconstruction_audit": audit,
        "legacy_scalar_proxy_policy": "forbidden_for_report_S_alg",
    }
    write_json(sidecar_path, sidecar)
    return sidecar


def write_snake_fair_shot_work_sidecar(row: Mapping[str, str], output_root: Path) -> dict[str, Any]:
    result_path = resolve_result_path(row, output_root)
    sidecar_path = output_root / "snake_fair_shot_work.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    fair_work, audit = snake_fair_expanded_work_from_payload(payload, source_label=str(result_path))
    sidecar = {
        "schema": "paper_i_hh_spsa_budget_ladder_snake_fair_shot_work_sidecar_v1",
        "record_id": row.get("record_id"),
        "display_regime": row.get("display_regime"),
        "engine_key": row.get("engine_key"),
        "budget": row.get("budget"),
        "result_json": str(result_path),
        "S_actual": fair_work.get("S_actual"),
        "S_actual_status": fair_work.get("S_actual_status"),
        "S_actual_reason": fair_work.get("S_actual_reason"),
        "S_actual_policy": fair_work.get("S_actual_policy"),
        "S_common_exposure": fair_work.get("S_common_exposure"),
        "S_common_exposure_status": fair_work.get("S_common_exposure_status"),
        "S_common_exposure_reason": fair_work.get("S_common_exposure_reason"),
        "S_common_exposure_policy": fair_work.get("S_common_exposure_policy"),
        "S_fair": fair_work.get("S_fair"),
        "S_fair_status": fair_work.get("S_fair_status"),
        "S_fair_missing_reason": fair_work.get("S_fair_missing_reason"),
        "S_fair_reason": fair_work.get("S_fair_reason"),
        "S_fair_policy": fair_work.get("S_fair_policy"),
        "S_fair_source": fair_work.get("S_fair_source"),
        "S_fair_source_kind": fair_work.get("S_fair_source_kind"),
        "fair_work_currency": fair_work.get("fair_work_currency"),
        "component_counts": fair_work.get("component_counts"),
        "component_sources": fair_work.get("component_sources"),
        "component_source_kind": fair_work.get("component_source_kind"),
        "operator_probe_charge_basis": fair_work.get("operator_probe_charge_basis"),
        "work_contract_id": fair_work.get("work_contract_id"),
        "common_algorithmic_component_status": fair_work.get("common_algorithmic_component_status"),
        "candidate_work_ledger_status": fair_work.get("candidate_work_ledger_status"),
        "algorithmic_measurement_work": fair_work.get("algorithmic_measurement_work"),
        "table_i_measurement_event_ledger": fair_work.get("table_i_measurement_event_ledger"),
        "reconstruction_audit": audit,
        "legacy_grouped_sidecar_policy": (
            "snake_algorithmic_work.json is diagnostic for legacy/mixed rows; "
            "report-visible S_fair comes from this expanded/common sidecar"
        ),
    }
    write_json(sidecar_path, sidecar)
    return sidecar


def run_subprocess(cmd: Sequence[str], env: Mapping[str, str], output_root: Path) -> tuple[list[str], dict[str, str], int]:
    output_root.mkdir(parents=True, exist_ok=True)
    stdout_path = output_root / "stdout.log"
    stderr_path = output_root / "stderr.log"
    stream_to_parent = str(env.get("GENERIC_STATIC_TABLE_PROGRESS_STDOUT") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w", encoding="utf-8") as err:
        if not stream_to_parent:
            proc = subprocess.run(
                list(cmd),
                cwd=str(ROOT),
                env=dict(env),
                stdout=out,
                stderr=err,
                text=True,
                check=False,
            )
            returncode = int(proc.returncode)
        else:
            proc = subprocess.Popen(
                list(cmd),
                cwd=str(ROOT),
                env=dict(env),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )

            def relay(pipe: Any, log_handle: Any, console: Any) -> None:
                try:
                    for line in iter(pipe.readline, ""):
                        log_handle.write(line)
                        log_handle.flush()
                        console.write(line)
                        console.flush()
                finally:
                    pipe.close()

            stdout_thread = threading.Thread(
                target=relay,
                args=(proc.stdout, out, sys.stdout),
                name="paper-i-hh-cell-stdout-relay",
                daemon=True,
            )
            stderr_thread = threading.Thread(
                target=relay,
                args=(proc.stderr, err, sys.stderr),
                name="paper-i-hh-cell-stderr-relay",
                daemon=True,
            )
            stdout_thread.start()
            stderr_thread.start()
            returncode = int(proc.wait())
            stdout_thread.join()
            stderr_thread.join()
    overlay = {
        key: env[key]
        for key in sorted(env)
        if key.startswith("GENERIC_STATIC_TABLE_")
        or key.startswith("ADAPT_SPSA")
        or key.startswith("STATIC_ADAPT_")
        or key in {"TABLE_I_STATIC_SUITE_PROFILE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"}
    }
    return list(cmd), overlay, returncode


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_lock_audit_manifest_summary(
    row: Mapping[str, str],
    output_root: Path,
    env_overlay: Mapping[str, str],
) -> dict[str, Any]:
    audit_path = resolve_row_path(
        str(row.get("source_lock_command_audit_rel") or ""),
        output_root / "source_lock_command_audit.json",
    )
    if not audit_path.exists():
        return {"path": str(audit_path), "status": "missing"}
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"path": str(audit_path), "status": "unreadable", "exception": repr(exc)}
    return {
        "path": str(audit_path),
        "sha256": sha256_file(audit_path),
        "schema": audit.get("schema"),
        "status": audit.get("status"),
        "non_allowed_flag_changes": audit.get("non_allowed_flag_changes"),
        "changed_flags": [item.get("flag") for item in audit.get("changed_flags", [])],
        "forced_depth30_no_early_stop": audit.get("forced_depth30_no_early_stop"),
        "suppress_drop_plateau_terminal_stop": env_overlay.get("STATIC_ADAPT_SUPPRESS_DROP_PLATEAU_TERMINAL_STOP") == "1",
        "diagnostic_schur_warm_start_mode": audit.get("diagnostic_schur_warm_start_mode"),
        "diagnostic_inner_optimizer": audit.get("diagnostic_inner_optimizer"),
        "diagnostic_snake_phase3_runtime_split_mode": audit.get("diagnostic_snake_phase3_runtime_split_mode"),
        "diagnostic_snake_phase3_runtime_split_selection_mode": audit.get(
            "diagnostic_snake_phase3_runtime_split_selection_mode"
        ),
        "preserved_flags_all_true": all(
            bool(item.get("preserved")) for item in (audit.get("preserved_flags") or {}).values()
        ),
    }


def _ai_log_events(stdout_path: Path) -> list[dict[str, Any]]:
    if not stdout_path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in stdout_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("AI_LOG "):
            continue
        try:
            payload = json.loads(line[len("AI_LOG ") :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def result_summary_from_artifacts(row: Mapping[str, str], output_root: Path) -> dict[str, Any]:
    result_path = resolve_result_path(row, output_root)
    if not result_path.exists():
        return {"result_json": str(result_path), "status": "missing_result_json"}
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"result_json": str(result_path), "status": "unreadable_result_json", "exception": repr(exc)}
    payload_status = str(payload.get("status") or "") if isinstance(payload, Mapping) else ""
    adapt = payload.get("adapt_vqe") if isinstance(payload, Mapping) else None
    if not isinstance(adapt, Mapping):
        nested_result = payload.get("result") if isinstance(payload, Mapping) else None
        adapt = nested_result if isinstance(nested_result, Mapping) else payload if isinstance(payload, Mapping) else {}
    result_status = str(adapt.get("status") or "")
    failure_statuses = {
        "failed",
        "quality_nonpassing",
        "completed_quality_nonpassing",
        "resource_guard",
        "skipped_optional_dependency",
        "blocked",
    }
    summary_status = (
        "failed_payload_status"
        if payload_status.lower() in failure_statuses or result_status.lower() in failure_statuses
        else "ok"
    )
    summary: dict[str, Any] = {
        "result_json": str(result_path),
        "status": summary_status,
        "payload_status": payload_status or None,
        "result_status": result_status or None,
        "success": adapt.get("success"),
        "stop_reason": adapt.get("stop_reason", adapt.get("adapt_stop_reason")),
        "energy": adapt.get("energy"),
        "exact_gs_energy": adapt.get("exact_gs_energy", adapt.get("exact_energy")),
        "abs_delta_e": adapt.get("abs_delta_e"),
        "ansatz_depth": adapt.get("ansatz_depth", adapt.get("adapt_depth_reached")),
        "terminal_accepted_ansatz_depth": adapt.get(
            "ansatz_depth", adapt.get("adapt_depth_reached")
        ),
        "benchmark_target_error_within_threshold": adapt.get("benchmark_target_error_within_threshold"),
        "benchmark_target_hit_success": adapt.get("benchmark_target_hit_success"),
        "benchmark_target_non_hit_reason": adapt.get("benchmark_target_non_hit_reason"),
    }
    history = adapt.get("history", adapt.get("adapt_history"))
    if isinstance(history, list):
        summary["winner_branch_history_step_count"] = len(history)
        last_history = history[-1] if history and isinstance(history[-1], Mapping) else None
        if last_history:
            summary["winner_branch_depth_local"] = last_history.get("depth")

    round_count = 0
    checkpoint_count = 0
    max_round_depth: int | None = None
    last_round: Mapping[str, Any] | None = None
    last_checkpoint: Mapping[str, Any] | None = None
    for event in _ai_log_events(output_root / "stdout.log"):
        if event.get("event") == "hardcoded_adapt_beam_round_done":
            round_count += 1
            last_round = event
            depth = event.get("depth")
            if isinstance(depth, int):
                max_round_depth = depth if max_round_depth is None else max(max_round_depth, depth)
        elif event.get("event") == "hardcoded_adapt_current_checkpoint_written":
            checkpoint_count += 1
            last_checkpoint = event
    if round_count:
        summary["logged_beam_round_count"] = round_count
        summary["max_logged_beam_round_depth"] = max_round_depth
    if last_round is not None:
        summary["last_beam_round_depth"] = last_round.get("depth")
        summary["last_beam_round_parent_count"] = last_round.get("parent_count")
        summary["last_beam_round_child_count"] = last_round.get("child_count")
        summary["last_beam_round_live_count"] = last_round.get("live_count")
        summary["last_beam_round_terminal_count"] = last_round.get("terminal_count")
    if checkpoint_count:
        summary["checkpoint_event_count"] = checkpoint_count
    if last_checkpoint is not None:
        summary["last_checkpoint_depth"] = last_checkpoint.get("depth")
        summary["last_checkpoint_ansatz_depth"] = last_checkpoint.get("ansatz_depth")
        summary["last_checkpoint_abs_delta_e"] = last_checkpoint.get("benchmark_target_abs_delta_e_current")
        summary["last_checkpoint_energy"] = last_checkpoint.get("energy_current")
        summary["last_checkpoint_branch_id"] = last_checkpoint.get("branch_id")
    summary["depth_semantics"] = (
        "max_logged_beam_round_depth/logged_beam_round_count describe executed adaptive selection rounds from stdout telemetry; "
        "winner_branch_depth_local/winner_branch_history_step_count describe the selected best finalist branch in result.json; "
        "ansatz_depth/terminal_accepted_ansatz_depth describe the accepted operator count after pruning."
    )
    return summary


def run_cell(
    record_id: str,
    records_path: Path,
    output_root: Path,
    *,
    schur_warm_start_mode: str = "off",
    runtime_worker_overrides: Mapping[str, int] | None = None,
) -> int:
    row = load_record(record_id, records_path)
    schur_warm_start_mode = row_schur_warm_start_mode(row, schur_warm_start_mode)
    snake_runtime_split_mode = row_snake_runtime_split_mode(row) if row.get("method_key") == "snake" else "source"
    snake_runtime_split_selection_mode = (
        row_snake_runtime_split_selection_mode(row) if row.get("method_key") == "snake" else "source"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    if row.get("runnable") != "true":
        manifest = {
            "schema": "paper_i_hh_spsa_budget_ladder_cell_manifest_v1",
            "record_id": record_id,
            "status": "blocked",
            "blocker": row.get("blocker"),
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "row": dict(row),
        }
        write_json(output_root / "cell_manifest.json", manifest)
        return 2

    try:
        if schur_warm_start_mode != "off" and row["method_key"] != "snake":
            raise ValueError("--adapt-schur-warm-start-mode diagnostics are only supported for snake rows")
        if row["method_key"] == "snake":
            cmd, env_overlay, returncode = run_snake(
                row,
                output_root,
                schur_warm_start_mode=schur_warm_start_mode,
                runtime_worker_overrides=runtime_worker_overrides,
            )
        else:
            cmd, env_overlay, returncode = run_append_geo(row, output_root)
        status = "ok" if returncode == 0 else "failed"
        snake_work: dict[str, Any] = {}
        if row.get("method_key") == "snake":
            sidecar_path = resolve_row_path(
                str(row.get("snake_algorithmic_work_rel") or ""),
                output_root / "snake_algorithmic_work.json",
            )
            if sidecar_path.exists():
                try:
                    sidecar_payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    snake_work = {
                        "snake_algorithmic_work_sidecar": str(sidecar_path),
                        "S_alg_status": "unreadable_sidecar",
                        "S_alg_sidecar_exception": repr(exc),
                    }
                else:
                    snake_work = {
                        "snake_algorithmic_work_sidecar": str(sidecar_path),
                        "S_alg": sidecar_payload.get("S_alg"),
                        "S_alg_status": sidecar_payload.get("S_alg_status"),
                    }
            else:
                snake_work = {
                    "snake_algorithmic_work_sidecar": str(sidecar_path),
                    "S_alg_status": "missing_sidecar",
                }
            fair_sidecar_path = output_root / "snake_fair_shot_work.json"
            if fair_sidecar_path.exists():
                try:
                    fair_sidecar_payload = json.loads(fair_sidecar_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    snake_work.update(
                        {
                            "snake_fair_shot_work_sidecar": str(fair_sidecar_path),
                            "S_fair_status": "unreadable_sidecar",
                            "S_fair_sidecar_exception": repr(exc),
                        }
                    )
                else:
                    snake_work.update(
                        {
                            "snake_fair_shot_work_sidecar": str(fair_sidecar_path),
                            "S_fair": fair_sidecar_payload.get("S_fair"),
                            "S_fair_status": fair_sidecar_payload.get("S_fair_status"),
                            "S_fair_source_kind": fair_sidecar_payload.get("S_fair_source_kind"),
                            "fair_work_currency": fair_sidecar_payload.get("fair_work_currency"),
                        }
                    )
            else:
                snake_work.update(
                    {
                        "snake_fair_shot_work_sidecar": str(fair_sidecar_path),
                        "S_fair_status": "missing_sidecar",
                    }
                )
            snake_work["source_lock_command_audit"] = source_lock_audit_manifest_summary(
                row,
                output_root,
                env_overlay,
            )
        manifest = {
            "schema": "paper_i_hh_spsa_budget_ladder_cell_manifest_v1",
            "record_id": record_id,
            "status": status,
            "returncode": returncode,
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "command": cmd,
            "env_overlay": env_overlay,
            "diagnostic_schur_warm_start_mode": schur_warm_start_mode,
            "diagnostic_inner_optimizer": (
                row_inner_optimizer(row) if row.get("method_key") == "snake" else str(row.get("adapt_optimizer_kind") or "powell")
            ),
            "diagnostic_snake_phase3_runtime_split_mode": snake_runtime_split_mode,
            "diagnostic_snake_phase3_runtime_split_selection_mode": snake_runtime_split_selection_mode,
            "diagnostic_runtime_worker_overrides": dict(runtime_worker_overrides or {}),
            **snake_work,
            "result_summary": result_summary_from_artifacts(row, output_root),
            "row": dict(row),
        }
        write_json(output_root / "effective_command.json", {"command": cmd})
        write_json(output_root / "effective_env_overlay.json", env_overlay)
        write_json(output_root / "cell_manifest.json", manifest)
        return int(returncode)
    except Exception as exc:
        manifest = {
            "schema": "paper_i_hh_spsa_budget_ladder_cell_manifest_v1",
            "record_id": record_id,
            "status": "runner_exception",
            "exception": repr(exc),
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "row": dict(row),
        }
        write_json(output_root / "cell_manifest.json", manifest)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record_id")
    parser.add_argument("records_path", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--adapt-schur-warm-start-mode",
        choices=["off", "append", "prune", "append-prune"],
        default="off",
        help="Diagnostic source-locked variable to pass through to SNAKE adapt_pipeline rows.",
    )
    parser.add_argument("--adapt-parallel-gradient-workers", type=int, default=None)
    parser.add_argument("--adapt-beam-parent-workers", type=int, default=None)
    parser.add_argument("--adapt-spsa-parallel-evaluations", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runtime_worker_overrides = {
        flag: int(value)
        for flag, value in (
            ("--adapt-parallel-gradient-workers", args.adapt_parallel_gradient_workers),
            ("--adapt-beam-parent-workers", args.adapt_beam_parent_workers),
            ("--adapt-spsa-parallel-evaluations", args.adapt_spsa_parallel_evaluations),
        )
        if value is not None
    }
    return run_cell(
        str(args.record_id),
        Path(args.records_path),
        Path(args.output_root),
        schur_warm_start_mode=str(args.adapt_schur_warm_start_mode),
        runtime_worker_overrides=runtime_worker_overrides or None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
