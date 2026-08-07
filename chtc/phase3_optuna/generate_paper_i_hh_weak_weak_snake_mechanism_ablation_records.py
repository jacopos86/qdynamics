#!/usr/bin/env python3
"""Generate Paper-I HH weak-weak SNAKE mechanism-ablation CHTC records.

This generator is intentionally source-locked to existing weak-weak POWELL
SNAKE anchors.  It emits one doubled mechanism-ablation matrix:

* the existing combinatorial ordered-batch cap-3 anchor family;
* the existing physical-operator-lane anchor family.

The Pauli-child subset cap remains 1.  Batch cap/target 3 is a separate route
control and is never encoded as a child-subset cap.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_hh_fullmeta_phase3_singleton_rotosolve_records as base


DEFAULT_BATCH_ID = "paper_i_hh_weak_weak_snake_mechanism_ablation_20260708_v1"
DEFAULT_REQUEST_CPUS = 4
DEFAULT_REQUEST_MEMORY_MB = 32768
DEFAULT_REQUEST_DISK_MB = 61440
DEFAULT_MAX_RUNTIME_S = 172800

RUN_PLAN_MD = (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_powell_weak_weak_snake_mechanism_ablation_chtc_plan_20260708.md"
)
REGIME = "weak-weak"
INTERNAL_REGIME = "weak_weak"
SUITE_PROFILE = "paper_i_three_model_hh_symmetric_20260527_v1"
CASE_ID = "hh_L2_nph2_three_model_sym_weak_weak"
N_PH_WORK = "2"
N_PH_REF = "5"
EXACT_GS_ENERGY = "-0.9183531194992405"
SOURCE_PDF = (
    "output/pdf/paper_i_hh_physical_operator_lane_comparison_20260708/"
    "paper_i_hh_physical_operator_lane_comparison_20260708.pdf"
)

PAULI_CHILD_MODE = "shortlist_pauli_children_v1"
RUNTIME_SPLIT_SELECTION = "archival_child_set_forward_v1"
CHILD_SUBSET_CAP = "1"
BEAM_LAMBDA = "0.005"
BEAM_LIVE_BRANCHES = "3"
BEAM_CHILDREN_PER_PARENT = "2"
METRIC_PRUNE_ROUTE = "metric_regularized_v1"

COMBO_SOURCE_RESULT = (
    "raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/"
    "weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3/"
    "json/result.json"
)
COMBO_SOURCE_COMMAND = (
    "raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/"
    "weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__combinatorial_cap3/"
    "run_command.json"
)
COMBO_NOBATCH_REFERENCE = (
    "raw_outputs/paper_i_hh_powell_visible_batchroute_nobatch_20260706/"
    "weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__nobatch_fullv2/"
    "json/result.json"
)
COMBO_GREEDY_REFERENCE = (
    "raw_outputs/paper_i_hh_powell_weak_weak_snake_batchcap3_ablation_20260707/"
    "weak_weak__snake__powell200__phase3_archival_subset1__beam0p005_metricprune__greedy_cap3/"
    "json/result.json"
)

PHYSICAL_SOURCE_RESULT = "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/weak_weak/json/result.json"
PHYSICAL_COMMANDS_JSON = "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json"
PHYSICAL_SOURCE_LOCK_MANIFEST = "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/source_lock_manifest.json"


EXTRA_FIELDNAMES = (
    "source_anchor_family",
    "source_anchor_role",
    "source_anchor_reference_status",
    "source_anchor_result_json",
    "source_anchor_result_sha256",
    "source_anchor_command_json",
    "source_anchor_command_sha256",
    "source_anchor_support_pdf",
    "source_anchor_support_pdf_sha256",
    "source_anchor_lock_manifest",
    "source_anchor_lock_manifest_sha256",
    "hh_mechanism_ablation_variant",
    "hh_mechanism_ablation_feature",
    "hh_mechanism_ablation_role",
    "hh_mechanism_ablation_submit_group",
    "hh_mechanism_ablation_expected_status",
    "hh_mechanism_ablation_overrides_json",
    "hh_mechanism_ablation_blocked_reason",
    "hh_mechanism_ablation_reference_json",
    "hh_mechanism_ablation_reference_sha256",
    "hh_mechanism_ablation_plan_md",
)
FIELDNAMES = tuple(dict.fromkeys((*base.OUTPUT_FIELDNAMES, *EXTRA_FIELDNAMES)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _rel(path: str | Path) -> str:
    return base.rel_or_abs(_repo_path(path))


def _read_json(path: str | Path) -> Any:
    return json.loads(_repo_path(path).read_text(encoding="utf-8"))


def _json(payload: Mapping[str, Any] | Sequence[Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _zero_flags(*flags: str) -> dict[str, str]:
    return {flag: "0.0" for flag in flags}


NO_COST_FLAGS: dict[str, str] = {
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


@dataclass(frozen=True)
class SourceAnchor:
    family: str
    role: str
    source_result: str
    source_command_json: str
    command_args: tuple[str, ...]
    source_lock_manifest: str = ""
    support_pdf: str = SOURCE_PDF

    @property
    def result_sha256(self) -> str:
        return sha256_file(_repo_path(self.source_result))

    @property
    def command_sha256(self) -> str:
        return sha256_file(_repo_path(self.source_command_json))

    @property
    def source_lock_manifest_sha256(self) -> str:
        if not self.source_lock_manifest:
            return ""
        return sha256_file(_repo_path(self.source_lock_manifest))

    @property
    def support_pdf_sha256(self) -> str:
        if not self.support_pdf or not _repo_path(self.support_pdf).exists():
            return ""
        return sha256_file(_repo_path(self.support_pdf))


@dataclass(frozen=True)
class Variant:
    name: str
    feature: str
    role: str
    submit_group: str
    matrix_label: str
    child_policy: str
    symmetry_policy: str
    runnable_by_family: Mapping[str, bool]
    reference_by_family: Mapping[str, str]
    blocker: str = ""
    set_flags: Mapping[str, str] | None = None
    enable_flags: Sequence[str] = ()
    remove_bool_flags: Sequence[str] = ()
    remove_value_flags: Sequence[str] = ()
    batch_mode: str = ""


STATIC_UNSPECIFIED = {"--static-route-id": "unspecified"}


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "full_anchor_reference",
        "none",
        "reference",
        "reference",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": False, "physical_operator_lane": False},
        {"batch_cap3_combinatorial": COMBO_SOURCE_RESULT, "physical_operator_lane": PHYSICAL_SOURCE_RESULT},
        blocker="reference_existing_source_anchor_not_submitted",
    ),
    Variant(
        "no_batching_reference",
        "phase2_phase3_batching",
        "reference",
        "reference",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": False, "physical_operator_lane": False},
        {"batch_cap3_combinatorial": COMBO_NOBATCH_REFERENCE, "physical_operator_lane": PHYSICAL_SOURCE_RESULT},
        blocker="reference_existing_no_batch_row_not_submitted",
        set_flags=STATIC_UNSPECIFIED,
        enable_flags=("--phase2-no-batching", "--phase3-no-batching"),
        remove_bool_flags=("--phase2-enable-batching", "--phase3-enable-batching"),
        remove_value_flags=(
            "--phase2-batch-selection-mode",
            "--phase3-batch-selection-mode",
            "--phase2-batch-target-size",
            "--phase2-batch-size-cap",
            "--phase3-batch-target-size",
            "--phase3-batch-size-cap",
        ),
    ),
    Variant(
        "greedy_cap3",
        "phase2_phase3_batching",
        "batch_addition_or_reference",
        "batch",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": False, "physical_operator_lane": True},
        {"batch_cap3_combinatorial": COMBO_GREEDY_REFERENCE},
        blocker="reference_existing_greedy_cap3_row_not_submitted",
        set_flags={
            **STATIC_UNSPECIFIED,
            "--phase2-batch-selection-mode": "greedy_reduced_plane",
            "--phase3-batch-selection-mode": "greedy_reduced_plane",
            "--phase2-batch-target-size": "3",
            "--phase2-batch-size-cap": "3",
            "--phase3-batch-target-size": "3",
            "--phase3-batch-size-cap": "3",
        },
        enable_flags=("--phase2-enable-batching", "--phase3-enable-batching"),
        remove_bool_flags=("--phase2-no-batching", "--phase3-no-batching"),
        batch_mode="greedy_reduced_plane",
    ),
    Variant(
        "combinatorial_cap3",
        "phase2_phase3_batching",
        "batch_anchor_or_addition",
        "batch",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": False, "physical_operator_lane": True},
        {"batch_cap3_combinatorial": COMBO_SOURCE_RESULT},
        blocker="reference_existing_combinatorial_cap3_anchor_not_submitted",
        set_flags={
            **STATIC_UNSPECIFIED,
            "--phase2-batch-selection-mode": "combinatorial_reduced_plane",
            "--phase3-batch-selection-mode": "combinatorial_reduced_plane",
            "--phase2-batch-target-size": "3",
            "--phase2-batch-size-cap": "3",
            "--phase3-batch-target-size": "3",
            "--phase3-batch-size-cap": "3",
        },
        enable_flags=("--phase2-enable-batching", "--phase3-enable-batching"),
        remove_bool_flags=("--phase2-no-batching", "--phase3-no-batching"),
        batch_mode="combinatorial_reduced_plane",
    ),
    Variant(
        "no_prune",
        "recoverability_prune",
        "disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags=STATIC_UNSPECIFIED,
        enable_flags=("--phase1-no-prune",),
        remove_bool_flags=(
            "--phase1-prune-enabled",
            "--phase1-prune-amplitude-witness-optional",
            "--phase1-prune-amplitude-witness-required",
        ),
        remove_value_flags=("--phase1-prune-policy", "--phase1-prune-mode", "--phase1-prune-schur-nomination-route"),
    ),
    Variant(
        "no_cost_term",
        "resource_cost_term",
        "disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={**STATIC_UNSPECIFIED, **NO_COST_FLAGS},
    ),
    Variant(
        "no_novelty",
        "phase2_phase3_novelty",
        "disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={
            **STATIC_UNSPECIFIED,
            "--phase2-gamma-N": "0.0",
            "--phase2-gamma-N-schedule-mode": "fixed",
            "--phase3-novelty-ablation-mode": "all",
        },
        remove_value_flags=("--phase2-gamma-N-schedule-start", "--phase2-gamma-N-schedule-end"),
    ),
    Variant(
        "phase2_novelty_only_no_second_order",
        "phase2_second_order_energy",
        "phase2_only_disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={
            **STATIC_UNSPECIFIED,
            "--adapt-continuation-mode": "phase2_v1",
            "--phase3-backend-cost-mode": "proxy",
            "--phase2-selector-gain-mode": "unit_gain_v1",
        },
    ),
    Variant(
        "phase2_second_order_only_no_novelty",
        "phase2_novelty",
        "phase2_only_disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={
            **STATIC_UNSPECIFIED,
            "--adapt-continuation-mode": "phase2_v1",
            "--phase3-backend-cost-mode": "proxy",
            "--phase2-selector-gain-mode": "trust_region_v1",
            "--phase2-gamma-N": "0.0",
            "--phase2-gamma-N-schedule-mode": "fixed",
            "--phase3-novelty-ablation-mode": "no_phase2",
        },
        remove_value_flags=("--phase2-gamma-N-schedule-start", "--phase2-gamma-N-schedule-end"),
    ),
    Variant(
        "no_phase3",
        "phase3",
        "disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={**STATIC_UNSPECIFIED, "--adapt-continuation-mode": "phase2_v1", "--phase3-backend-cost-mode": "proxy"},
    ),
    Variant(
        "phase1_only_macro_pool",
        "phase2_phase3_and_child_policy",
        "phase1_only",
        "ablation",
        "C_macro_only",
        "macro_only",
        "not_applicable",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={**STATIC_UNSPECIFIED, "--adapt-continuation-mode": "phase1_v1", "--phase3-backend-cost-mode": "proxy"},
    ),
    Variant(
        "phase1_only_singleton_pool",
        "phase2_phase3_and_child_policy",
        "phase1_only",
        "ablation",
        "B_common_phase0_singleton_hard_guard",
        "common_phase0_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={**STATIC_UNSPECIFIED, "--adapt-continuation-mode": "phase1_v1", "--phase3-backend-cost-mode": "proxy"},
    ),
    Variant(
        "full_geometry_window",
        "phase3_geometry",
        "disabled_minus_full",
        "ablation",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": True, "physical_operator_lane": True},
        {},
        set_flags={**STATIC_UNSPECIFIED, "--phase3-selector-geometry-mode": "raw_exact"},
    ),
    Variant(
        "no_shortlisting",
        "shortlisting",
        "blocked",
        "blocked",
        "A_native_staged_singleton_hard_guard",
        "native_phase3_singleton",
        "hard_guard",
        {"batch_cap3_combinatorial": False, "physical_operator_lane": False},
        {},
        blocker="blocked_until_audited_route_opens_phase_shortlists_controller_caps_and_lane_gates",
    ),
)


def _physical_weak_weak_command() -> tuple[str, ...]:
    payload = _read_json(PHYSICAL_COMMANDS_JSON)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {PHYSICAL_COMMANDS_JSON}")
    for row in payload:
        if isinstance(row, Mapping) and str(row.get("regime") or "") == REGIME:
            argv = row.get("argv")
            if isinstance(argv, list) and argv:
                return tuple(str(item) for item in argv)
    raise ValueError(f"No weak-weak command in {PHYSICAL_COMMANDS_JSON}")


def load_anchors() -> dict[str, SourceAnchor]:
    combo_args = tuple(str(item) for item in _read_json(COMBO_SOURCE_COMMAND))
    physical_args = _physical_weak_weak_command()
    anchors = {
        "batch_cap3_combinatorial": SourceAnchor(
            family="batch_cap3_combinatorial",
            role="existing_combinatorial_batch_cap3_anchor",
            source_result=COMBO_SOURCE_RESULT,
            source_command_json=COMBO_SOURCE_COMMAND,
            command_args=combo_args,
            source_lock_manifest="",
        ),
        "physical_operator_lane": SourceAnchor(
            family="physical_operator_lane",
            role="existing_physical_operator_lane_nobatch_parent_for_batch_source_rebuild",
            source_result=PHYSICAL_SOURCE_RESULT,
            source_command_json=PHYSICAL_COMMANDS_JSON,
            command_args=physical_args,
            source_lock_manifest=PHYSICAL_SOURCE_LOCK_MANIFEST,
        ),
    }
    for anchor in anchors.values():
        for path in (anchor.source_result, anchor.source_command_json):
            if not _repo_path(path).exists():
                raise FileNotFoundError(path)
        if anchor.source_lock_manifest and not _repo_path(anchor.source_lock_manifest).exists():
            raise FileNotFoundError(anchor.source_lock_manifest)
    return anchors


def _reference_sha(variant: Variant, family: str) -> str:
    path = variant.reference_by_family.get(family, "")
    if not path:
        return ""
    p = _repo_path(path)
    return sha256_file(p) if p.exists() else ""


def _base_overrides(record_id: str, variant: Variant) -> dict[str, Any]:
    set_flags = {
        "--adapt-segment-id": record_id,
        "--phase3-runtime-split-max-subset-size": CHILD_SUBSET_CAP,
        "--adapt-beam-live-branches": BEAM_LIVE_BRANCHES,
        "--adapt-beam-children-per-parent": BEAM_CHILDREN_PER_PARENT,
        "--adapt-beam-lambda": BEAM_LAMBDA,
        "--phase1-prune-schur-nomination-route": METRIC_PRUNE_ROUTE,
    }
    set_flags.update(dict(variant.set_flags or {}))
    payload: dict[str, Any] = {
        "set_flags": set_flags,
        "enable_flags": list(dict.fromkeys(str(flag) for flag in variant.enable_flags)),
        "remove_bool_flags": list(dict.fromkeys(str(flag) for flag in variant.remove_bool_flags)),
        "remove_value_flags": list(
            dict.fromkeys(
                [
                    "--phase3-source-lock-preferred-sequence",
                    *(str(flag) for flag in variant.remove_value_flags),
                ]
            )
        ),
    }
    return {key: value for key, value in payload.items() if value}


def _runtime_fields(variant: Variant) -> dict[str, str]:
    if variant.child_policy == "macro_only":
        return {
            "snake_phase3_runtime_split_mode": "off",
            "snake_phase3_runtime_split_selection_mode": "",
            "snake_phase3_runtime_split_child_set_symmetry_policy": "",
            "snake_phase3_runtime_split_max_subset_size": "",
            "shared_pauli_pool_mode": "off",
            "shared_pauli_pool_symmetry_policy": "",
            "shared_pauli_pool_max_subset_size": "",
            "child_subset_size": "",
        }
    if variant.child_policy == "common_phase0_singleton":
        return {
            "snake_phase3_runtime_split_mode": "off",
            "snake_phase3_runtime_split_selection_mode": "",
            "snake_phase3_runtime_split_child_set_symmetry_policy": "",
            "snake_phase3_runtime_split_max_subset_size": "",
            "shared_pauli_pool_mode": base.SHARED_PAULI_POOL_MODE,
            "shared_pauli_pool_symmetry_policy": "hard_guard",
            "shared_pauli_pool_max_subset_size": CHILD_SUBSET_CAP,
            "child_subset_size": CHILD_SUBSET_CAP,
        }
    return {
        "snake_phase3_runtime_split_mode": PAULI_CHILD_MODE,
        "snake_phase3_runtime_split_selection_mode": RUNTIME_SPLIT_SELECTION,
        "snake_phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "snake_phase3_runtime_split_max_subset_size": CHILD_SUBSET_CAP,
        "shared_pauli_pool_mode": "off",
        "shared_pauli_pool_symmetry_policy": "",
        "shared_pauli_pool_max_subset_size": "",
        "child_subset_size": CHILD_SUBSET_CAP,
    }


def _record_id(batch_id: str, family: str, variant: str) -> str:
    return f"{batch_id}__weak_weak__snake__{family}__{variant}"


def make_row(batch_id: str, anchor: SourceAnchor, variant: Variant) -> dict[str, str]:
    record_id = _record_id(batch_id, anchor.family, variant.name)
    runnable = bool(variant.runnable_by_family.get(anchor.family, False))
    reference = variant.reference_by_family.get(anchor.family, "")
    blocker = "" if runnable else (variant.blocker or "reference_existing_row_not_submitted")
    row: dict[str, str] = {
        "record_id": record_id,
        "batch_id": batch_id,
        "run_class": "candidate",
        "runnable": "true" if runnable else "false",
        "blocker": "" if runnable else blocker,
        "method_key": "snake",
        "method_label": "SNAKE",
        "algorithm_id": "static_family_native_adapt_phase3",
        "engine_key": "source_locked_powell_weak_weak_snake_mechanism_ablation",
        "engine_label": "Paper-I HH weak-weak source-locked SNAKE mechanism ablation",
        "spsa_refit_engine": "",
        "budget": "200",
        "display_regime": REGIME,
        "internal_regime": INTERNAL_REGIME,
        "source_map_regime": REGIME,
        "suite_profile": SUITE_PROFILE,
        "case_id": CASE_ID,
        "family": "hh",
        "n_ph_work": N_PH_WORK,
        "n_ph_ref": N_PH_REF,
        "same_cutoff_exact_gs_energy": EXACT_GS_ENERGY,
        "same_cutoff_energy_key_hash": "",
        "exact_reference_energy": "",
        "exact_reference_energy_key_hash": "",
        "exact_reference_n_ph_max": N_PH_REF,
        "primary_energy_metric": "same_cutoff_abs_delta_e",
        "same_cutoff_error_role": "primary",
        "target_abs_delta_e": "",
        "max_depth": "30",
        "adapt_optimizer_kind": "powell",
        "optimizer_profile": "powell_maxiter200_final_refit200",
        "generic_adapt_runtime_split_mode": "",
        "generic_adapt_runtime_split_symmetry_policy": "",
        "generic_adapt_runtime_split_max_subset_size": "",
        "generic_adapt_stop_policy": "",
        "adapt_pool_class_filter_json": "off",
        "resource_qubit_cap": "",
        "resource_pool_term_cap": "",
        "adapt_schur_warm_start_mode": base.SNAKE_SCHUR_WARM_START_MODE,
        "source_json": anchor.source_result,
        "source_json_sha256": anchor.result_sha256,
        "source_command_sh": anchor.source_command_json,
        "source_command_sha256": anchor.command_sha256,
        "source_command_args_json": json.dumps(list(anchor.command_args), separators=(",", ":")),
        "source_settings_status": "source_locked_anchor_command",
        "schedule_source_policy": "powell_source_locked_no_spsa_schedule",
        "schedule_source_regime": REGIME,
        "schedule_source_method": "SNAKE",
        "schedule_source_json": anchor.source_result,
        "schedule_source_note": "POWELL weak-weak HH mechanism ablation; SPSA schedule fields intentionally empty.",
        "anchor_source_json": anchor.source_result,
        "anchor_source_sha256": anchor.result_sha256,
        "changed_fields_vs_anchor": ",".join(
            [
                "hh_mechanism_ablation_variant",
                "snake_cli_overrides_json",
                *(sorted((variant.set_flags or {}).keys())),
                *(variant.enable_flags),
                *(variant.remove_bool_flags),
                *(variant.remove_value_flags),
            ]
        ),
        "source_contract_note": (
            "Weak-weak Paper-I HH SNAKE mechanism ablation. Preserves full_meta/HVA, POWELL budget 200, "
            "depth cap 30, and Pauli-child subset cap 1. Variant-specific changes are explicit in "
            "snake_cli_overrides_json."
        ),
        "matrix_label": variant.matrix_label,
        "matrix_role": "weak_weak_mechanism_ablation",
        "static_route_id": "unspecified" if runnable else "source",
        "pool_contract": "full_meta_unfiltered",
        "hh_adaptive_pool_profile": "full_meta_unfiltered",
        "child_policy": variant.child_policy,
        "symmetry_policy": variant.symmetry_policy,
        "optimizer": "POWELL",
        "optimizer_overlay_id": "powell",
        "optimizer_contract_id": "powell_maxiter200_final_refit200_depth30_v1",
        "resource_tier": "standard",
        "request_memory_mb": str(DEFAULT_REQUEST_MEMORY_MB),
        "request_disk_mb": str(DEFAULT_REQUEST_DISK_MB),
        "spsa_schedule_policy": "not_applicable_powell",
        "blocked_reason": "" if runnable else blocker,
        "snake_cli_overrides_json": _json(_base_overrides(record_id, variant)),
        "ordered_batch_beam_label": variant.name,
        "ordered_batch_beam_enabled": "true" if variant.batch_mode else "false",
        "ordered_batch_beam_run_role": variant.role,
        "phase2_batch_selection_mode": variant.batch_mode,
        "phase2_batch_target_size": "3" if variant.batch_mode else "",
        "phase2_batch_size_cap": "3" if variant.batch_mode else "",
        "adapt_beam_live_branches": BEAM_LIVE_BRANCHES,
        "adapt_beam_children_per_parent": BEAM_CHILDREN_PER_PARENT,
        "adapt_beam_lambda": BEAM_LAMBDA,
        "ordered_batch_beam_expected_diagnostics_json": _json(
            {
                "lambda_beam": BEAM_LAMBDA,
                "child_subset_cap": CHILD_SUBSET_CAP,
                "batch_target_size": "3" if variant.batch_mode else "",
                "batch_size_cap": "3" if variant.batch_mode else "",
                "batch_selection_mode": variant.batch_mode,
            }
        ),
        "provenance_layer": "source_anchor_family",
        "visible_support_csv": (
            "output/pdf/paper_i_hh_physical_operator_lane_comparison_20260708/"
            "paper_i_hh_physical_operator_lane_comparison_20260708_provenance.csv"
        ),
        "visible_anchor_result_json": anchor.source_result,
        "visible_effective_command_json": anchor.source_command_json,
        "settings_reused_json": _json(
            {
                "source_anchor_family": anchor.family,
                "optimizer": "POWELL",
                "maxiter": 200,
                "final_refit_maxiter": 200,
                "max_depth": 30,
                "pool_contract": "full_meta_unfiltered",
                "hva_policy": "included",
                "runtime_split_mode": PAULI_CHILD_MODE,
                "runtime_split_selection": RUNTIME_SPLIT_SELECTION,
                "child_subset_cap": 1,
                "beam_lambda": BEAM_LAMBDA,
            }
        ),
        "settings_changed_json": _json(
            {
                "variant": variant.name,
                "feature": variant.feature,
                "set_flags": dict(variant.set_flags or {}),
                "enable_flags": list(variant.enable_flags),
                "remove_bool_flags": list(variant.remove_bool_flags),
                "remove_value_flags": list(variant.remove_value_flags),
            }
        ),
        "settings_change_reason": variant.role,
        "route_variant": variant.name,
        "anchor_gate_status": "source_anchor_existing",
        "batch_variant_gate": "source_anchor_family_local_existing",
        "work_semantics_expected_json": _json(
            {
                "S_alg_work_scope": "winner_lineage_display_prefix_and_terminal",
                "S_beam_search_total_scope": "all_expanded_scored_branches_when_available",
            }
        ),
        "latex_report_stem": "paper_i_hh_weak_weak_snake_mechanism_ablation_20260708",
        "latex_report_output_dir": "output/pdf/paper_i_hh_weak_weak_snake_mechanism_ablation_20260708",
        "report_update_policy": "after_fetch_build_latex_pdf_json_csv_sidecars",
        "source_anchor_family": anchor.family,
        "source_anchor_role": anchor.role,
        "source_anchor_reference_status": "existing_reference" if reference else "",
        "source_anchor_result_json": anchor.source_result,
        "source_anchor_result_sha256": anchor.result_sha256,
        "source_anchor_command_json": anchor.source_command_json,
        "source_anchor_command_sha256": anchor.command_sha256,
        "source_anchor_support_pdf": anchor.support_pdf,
        "source_anchor_support_pdf_sha256": anchor.support_pdf_sha256,
        "source_anchor_lock_manifest": anchor.source_lock_manifest,
        "source_anchor_lock_manifest_sha256": anchor.source_lock_manifest_sha256,
        "hh_mechanism_ablation_variant": variant.name,
        "hh_mechanism_ablation_feature": variant.feature,
        "hh_mechanism_ablation_role": variant.role,
        "hh_mechanism_ablation_submit_group": variant.submit_group,
        "hh_mechanism_ablation_expected_status": "queued" if runnable else ("blocked" if variant.submit_group == "blocked" else "reference"),
        "hh_mechanism_ablation_overrides_json": _json(_base_overrides(record_id, variant)),
        "hh_mechanism_ablation_blocked_reason": "" if runnable else blocker,
        "hh_mechanism_ablation_reference_json": reference,
        "hh_mechanism_ablation_reference_sha256": _reference_sha(variant, anchor.family),
        "hh_mechanism_ablation_plan_md": RUN_PLAN_MD,
    }
    row.update(_runtime_fields(variant))
    if anchor.family == "physical_operator_lane" and variant.name == "combinatorial_cap3":
        row.update(
            {
                "source_anchor_role": "physical_operator_lane_combinatorial_cap3_source_anchor_rebuild",
                "source_anchor_reference_status": "queued_source_anchor_rebuild_from_existing_nobatch_parent",
                "hh_mechanism_ablation_role": "physical_operator_lane_source_anchor_rebuild",
                "hh_mechanism_ablation_expected_status": "queued_source_anchor",
                "anchor_gate_status": "queued_physical_operator_lane_batch_cap3_source_anchor",
                "settings_change_reason": "physical_operator_lane_source_anchor_rebuild_with_combinatorial_batch_cap3",
            }
        )
    row.update(base.output_paths(record_id, "snake"))
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return row


def build_records(batch_id: str = DEFAULT_BATCH_ID) -> list[dict[str, str]]:
    base.configure_batch(batch_id)
    anchors = load_anchors()
    rows: list[dict[str, str]] = []
    for anchor_key in ("batch_cap3_combinatorial", "physical_operator_lane"):
        anchor = anchors[anchor_key]
        for variant in VARIANTS:
            rows.append(make_row(batch_id, anchor, variant))
    return rows


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def _source_transfer_inputs(anchors: Mapping[str, SourceAnchor], records: Sequence[Mapping[str, str]]) -> list[str]:
    values: list[str] = []
    for anchor in anchors.values():
        values.extend([anchor.source_result, anchor.source_command_json])
        if anchor.source_lock_manifest:
            values.append(anchor.source_lock_manifest)
        if anchor.support_pdf:
            values.append(anchor.support_pdf)
    for row in records:
        for field in (
            "hh_mechanism_ablation_reference_json",
            "source_anchor_result_json",
            "source_anchor_command_json",
            "source_anchor_lock_manifest",
        ):
            value = str(row.get(field) or "").strip()
            if value:
                values.append(value)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value or value in seen:
            continue
        path = _repo_path(value)
        if not path.exists():
            continue
        seen.add(value)
        out.append(_rel(value))
    return out


def _patch_submit_transfer_inputs(submit_path: Path, extra_inputs: Sequence[str]) -> None:
    if not extra_inputs:
        return
    lines = submit_path.read_text(encoding="utf-8").splitlines()
    patched: list[str] = []
    done = False
    for line in lines:
        if line.startswith("transfer_input_files = ") and not done:
            existing = [item.strip() for item in line.split("=", 1)[1].split(",") if item.strip()]
            merged = list(dict.fromkeys([*existing, *extra_inputs]))
            patched.append("transfer_input_files = " + ", ".join(merged))
            done = True
        else:
            patched.append(line)
    if not done:
        raise ValueError(f"submit file has no transfer_input_files line: {submit_path}")
    submit_path.write_text("\n".join(patched) + "\n", encoding="utf-8")


def write_records(
    batch_id: str,
    records: Sequence[Mapping[str, str]],
    *,
    request_cpus: int = DEFAULT_REQUEST_CPUS,
    request_memory_mb: int = DEFAULT_REQUEST_MEMORY_MB,
    request_disk_mb: int = DEFAULT_REQUEST_DISK_MB,
    max_runtime_s: int = DEFAULT_MAX_RUNTIME_S,
) -> dict[str, Any]:
    base.configure_batch(batch_id)
    input_dir = ROOT / "chtc" / "phase3_optuna" / "input" / batch_id
    records_tsv = input_dir / "paper_i_hh_spsa_budget_ladder_records.tsv"
    record_ids = input_dir / "paper_i_hh_spsa_budget_ladder_record_ids.txt"
    record_queue = input_dir / "paper_i_hh_spsa_budget_ladder_record_queue.tsv"
    smoke_ids = input_dir / "paper_i_hh_spsa_budget_ladder_smoke_record_ids.txt"
    all_record_ids = input_dir / "paper_i_hh_mechanism_ablation_all_record_ids.txt"
    blocked_record_ids = input_dir / "paper_i_hh_mechanism_ablation_blocked_or_reference_record_ids.txt"
    manifest_json = input_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    src_sanitized_tarball = input_dir / base.SRC_SANITIZED_TARBALL_NAME
    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"

    input_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in records]
    anchors = load_anchors()
    runnable_rows = [row for row in rows if str(row.get("runnable") or "").lower() == "true"]
    nonrunnable_rows = [row for row in rows if str(row.get("runnable") or "").lower() != "true"]
    if not runnable_rows:
        raise ValueError("No runnable mechanism-ablation records were generated.")

    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_lines(record_ids, (str(row["record_id"]) for row in runnable_rows))
    write_lines(
        record_queue,
        (
            f"{row['record_id']}\t{int(row.get('request_memory_mb') or request_memory_mb)}\t{int(row.get('request_disk_mb') or request_disk_mb)}"
            for row in runnable_rows
        ),
    )
    write_lines(all_record_ids, (str(row["record_id"]) for row in rows))
    write_lines(blocked_record_ids, (str(row["record_id"]) for row in nonrunnable_rows))
    write_lines(smoke_ids, (str(runnable_rows[0]["record_id"]),))
    base._write_sanitized_src_tarball(src_sanitized_tarball)
    base._write_matrix_submit_file(
        batch_id=batch_id,
        submit_path=submit_path,
        records_tsv=records_tsv,
        record_queue=record_queue,
        request_cpus=int(request_cpus),
        max_runtime_s=int(max_runtime_s),
        src_sanitized_tarball=src_sanitized_tarball,
    )
    source_transfer_inputs = _source_transfer_inputs(anchors, rows)
    _patch_submit_transfer_inputs(submit_path, source_transfer_inputs)

    manifest: dict[str, Any] = {
        "schema": "paper_i_hh_weak_weak_snake_mechanism_ablation_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "run_plan_md": RUN_PLAN_MD,
        "records_tsv": base.rel_or_abs(records_tsv),
        "record_id_file": base.rel_or_abs(record_ids),
        "record_queue": base.rel_or_abs(record_queue),
        "submit_path": base.rel_or_abs(submit_path),
        "record_count": len(rows),
        "runnable_record_count": len(runnable_rows),
        "reference_or_blocked_record_count": len(nonrunnable_rows),
        "request_cpus": int(request_cpus),
        "request_memory_mb": int(request_memory_mb),
        "request_disk_mb": int(request_disk_mb),
        "max_runtime_s": int(max_runtime_s),
        "source_anchor_families": {
            family: {
                "source_result": anchor.source_result,
                "source_result_sha256": anchor.result_sha256,
                "source_command_json": anchor.source_command_json,
                "source_command_sha256": anchor.command_sha256,
                "source_lock_manifest": anchor.source_lock_manifest,
                "source_lock_manifest_sha256": anchor.source_lock_manifest_sha256,
            }
            for family, anchor in anchors.items()
        },
        "source_transfer_input_count": len(source_transfer_inputs),
        "source_transfer_inputs": list(source_transfer_inputs),
        "expected_runnable_rows": 20,
        "runnable_record_ids": [row["record_id"] for row in runnable_rows],
        "reference_or_blocked_record_ids": [row["record_id"] for row in nonrunnable_rows],
    }
    manifest_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--request-cpus", type=int, default=DEFAULT_REQUEST_CPUS)
    parser.add_argument("--request-memory-mb", type=int, default=DEFAULT_REQUEST_MEMORY_MB)
    parser.add_argument("--request-disk-mb", type=int, default=DEFAULT_REQUEST_DISK_MB)
    parser.add_argument("--max-runtime-s", type=int, default=DEFAULT_MAX_RUNTIME_S)
    args = parser.parse_args(argv)

    rows = build_records(args.batch_id)
    manifest = write_records(
        args.batch_id,
        rows,
        request_cpus=args.request_cpus,
        request_memory_mb=args.request_memory_mb,
        request_disk_mb=args.request_disk_mb,
        max_runtime_s=args.max_runtime_s,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
