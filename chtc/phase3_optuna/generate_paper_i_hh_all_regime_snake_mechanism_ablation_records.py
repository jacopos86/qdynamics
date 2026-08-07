#!/usr/bin/env python3
"""Generate Paper-I HH all-regime SNAKE mechanism-ablation CHTC records.

This is the all-six-regime follow-up to the weak-weak mechanism-ablation
support batch.  It intentionally uses the current Paper-I HH SNAKE canonical
contract, with one approved exception: the anchor enables Phase-III ordered
batching with target/cap 3.  The Pauli-child subset cap remains 1.
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
from chtc.phase3_optuna.generate_paper_i_hh_weak_weak_snake_mechanism_ablation_records import (
    NO_COST_FLAGS,
    STATIC_UNSPECIFIED,
)


DEFAULT_BATCH_ID = "paper_i_hh_all_regime_snake_mechanism_ablation_20260709_v1"
DEFAULT_REQUEST_CPUS = 4
DEFAULT_REQUEST_MEMORY_MB = 32768
DEFAULT_REQUEST_DISK_MB = 61440
DEFAULT_MAX_RUNTIME_S = 172800

RUN_PLAN_MD = (
    "MATH/paper_facing/paper_I_static_scaffold/"
    "paper_i_hh_all_regime_snake_mechanism_ablation_20260709.md"
)
SOURCE_COMMANDS_JSON = "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/commands.json"
SOURCE_LOCK_MANIFEST = "raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/source_lock_manifest.json"
SOURCE_SUPPORT_PDF = (
    "output/pdf/paper_i_hh_physical_operator_lane_comparison_20260708/"
    "paper_i_hh_physical_operator_lane_comparison_20260708.pdf"
)

SUITE_PROFILE = "paper_i_three_model_hh_symmetric_20260527_v1"
PAULI_CHILD_MODE = "shortlist_pauli_children_v1"
RUNTIME_SPLIT_SELECTION = "archival_child_set_forward_v1"
CHILD_SUBSET_CAP = "1"
BEAM_LAMBDA = "0.005"
BEAM_LIVE_BRANCHES = "3"
BEAM_CHILDREN_PER_PARENT = "2"
METRIC_PRUNE_ROUTE = "metric_regularized_v1"
FULL_REOPT_WINDOW = "99"


@dataclass(frozen=True)
class RegimeSpec:
    display: str
    internal: str
    case_id: str
    n_ph_work: str
    n_ph_ref: str


REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec("weak-weak", "weak_weak", "hh_L2_nph2_three_model_sym_weak_weak", "2", "5"),
    # The executable case registry retains the historical strong_weak /
    # strong_strong names for the intermediate U/t=1.25 points.
    RegimeSpec("intermediate-weak", "intermediate_weak", "hh_L2_nph2_three_model_sym_strong_weak", "2", "5"),
    RegimeSpec("strong-weak", "strong_weak", "hh_L2_nph2_three_model_sym_strong_weak", "2", "5"),
    RegimeSpec("weak-strong", "weak_strong", "hh_L2_nph4_three_model_sym_weak_strong", "4", "7"),
    RegimeSpec("intermediate-strong", "intermediate_strong", "hh_L2_nph4_three_model_sym_strong_strong", "4", "7"),
    RegimeSpec("strong-strong", "strong_strong", "hh_L2_nph4_three_model_sym_strong_strong", "4", "7"),
)


EXTRA_FIELDNAMES = (
    "source_anchor_family",
    "source_anchor_role",
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
    "hh_mechanism_ablation_plan_md",
    "phase3_batch_selection_mode",
    "phase3_batch_target_size",
    "phase3_batch_size_cap",
    "static_lane_route",
    "physical_lane_shortlist_aggressiveness",
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


def _source_result_path(internal_regime: str) -> str:
    return f"raw_outputs/paper_i_hh_physical_operator_lanes_nobatch_factor3_20260708/{internal_regime}/json/result.json"


@dataclass(frozen=True)
class SourceAnchor:
    regime: RegimeSpec
    command_args: tuple[str, ...]
    source_result: str
    source_command_json: str = SOURCE_COMMANDS_JSON
    source_lock_manifest: str = SOURCE_LOCK_MANIFEST
    support_pdf: str = SOURCE_SUPPORT_PDF

    @property
    def result_sha256(self) -> str:
        return sha256_file(_repo_path(self.source_result))

    @property
    def command_sha256(self) -> str:
        return sha256_file(_repo_path(self.source_command_json))

    @property
    def source_lock_manifest_sha256(self) -> str:
        return sha256_file(_repo_path(self.source_lock_manifest))

    @property
    def support_pdf_sha256(self) -> str:
        path = _repo_path(self.support_pdf)
        return sha256_file(path) if path.exists() else ""

    @property
    def exact_energy(self) -> str:
        payload = _read_json(self.source_result)
        value = (payload.get("ground_state") or {}).get("exact_energy")
        return "" if value is None else str(value)


@dataclass(frozen=True)
class Variant:
    name: str
    feature: str
    role: str
    matrix_label: str = "A_native_staged_singleton_hard_guard"
    child_policy: str = "native_phase3_singleton"
    symmetry_policy: str = "hard_guard"
    submit_group: str = "ablation"
    set_flags: Mapping[str, str] | None = None
    enable_flags: Sequence[str] = ()
    remove_bool_flags: Sequence[str] = ()
    remove_value_flags: Sequence[str] = ()
    phase3_batch_mode: str = ""
    static_lane_route: str = "physical_operator_type"


def _phase3_batch_flags(mode: str) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        {
            "--phase3-batch-selection-mode": mode,
            "--phase3-batch-target-size": "3",
            "--phase3-batch-size-cap": "3",
        },
        ("--phase3-enable-batching", "--phase2-no-batching"),
        ("--phase3-no-batching", "--phase2-enable-batching"),
        ("--phase2-batch-selection-mode", "--phase2-batch-target-size", "--phase2-batch-size-cap"),
    )


COMBO_SET, COMBO_ENABLE, COMBO_REMOVE_BOOL, COMBO_REMOVE_VALUE = _phase3_batch_flags("combinatorial_reduced_plane")
GREEDY_SET, GREEDY_ENABLE, GREEDY_REMOVE_BOOL, GREEDY_REMOVE_VALUE = _phase3_batch_flags("greedy_reduced_plane")


VARIANTS: tuple[Variant, ...] = (
    Variant(
        "combinatorial_cap3_anchor",
        "phase3_batching",
        "source_anchor",
        submit_group="anchor",
        set_flags={**STATIC_UNSPECIFIED, **COMBO_SET},
        enable_flags=COMBO_ENABLE,
        remove_bool_flags=COMBO_REMOVE_BOOL,
        remove_value_flags=COMBO_REMOVE_VALUE,
        phase3_batch_mode="combinatorial_reduced_plane",
    ),
    Variant(
        "greedy_cap3",
        "phase3_batching",
        "batch_comparator",
        submit_group="batch",
        set_flags={**STATIC_UNSPECIFIED, **GREEDY_SET},
        enable_flags=GREEDY_ENABLE,
        remove_bool_flags=GREEDY_REMOVE_BOOL,
        remove_value_flags=GREEDY_REMOVE_VALUE,
        phase3_batch_mode="greedy_reduced_plane",
    ),
    Variant(
        "no_batching_reference",
        "phase3_batching",
        "disabled_minus_full",
        set_flags=STATIC_UNSPECIFIED,
        enable_flags=("--phase3-no-batching", "--phase2-no-batching"),
        remove_bool_flags=("--phase3-enable-batching", "--phase2-enable-batching"),
        remove_value_flags=(
            "--phase3-batch-selection-mode",
            "--phase3-batch-target-size",
            "--phase3-batch-size-cap",
            "--phase2-batch-selection-mode",
            "--phase2-batch-target-size",
            "--phase2-batch-size-cap",
        ),
    ),
    Variant(
        "no_prune",
        "recoverability_prune",
        "disabled_minus_full",
        set_flags=COMBO_SET | STATIC_UNSPECIFIED,
        enable_flags=(*COMBO_ENABLE, "--phase1-no-prune"),
        remove_bool_flags=(*COMBO_REMOVE_BOOL, "--phase1-prune-enabled"),
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase1-prune-policy", "--phase1-prune-mode", "--phase1-prune-schur-nomination-route"),
        phase3_batch_mode="combinatorial_reduced_plane",
    ),
    Variant(
        "no_cost_term",
        "resource_cost_term",
        "disabled_minus_full",
        set_flags={**STATIC_UNSPECIFIED, **COMBO_SET, **NO_COST_FLAGS},
        enable_flags=COMBO_ENABLE,
        remove_bool_flags=COMBO_REMOVE_BOOL,
        remove_value_flags=COMBO_REMOVE_VALUE,
        phase3_batch_mode="combinatorial_reduced_plane",
    ),
    Variant(
        "no_novelty",
        "phase2_phase3_novelty",
        "disabled_minus_full",
        set_flags={
            **STATIC_UNSPECIFIED,
            **COMBO_SET,
            "--phase2-gamma-N": "0.0",
            "--phase2-gamma-N-schedule-mode": "fixed",
            "--phase3-novelty-ablation-mode": "all",
        },
        enable_flags=COMBO_ENABLE,
        remove_bool_flags=COMBO_REMOVE_BOOL,
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase2-gamma-N-schedule-start", "--phase2-gamma-N-schedule-end"),
        phase3_batch_mode="combinatorial_reduced_plane",
    ),
    Variant(
        "phase2_novelty_only_no_second_order",
        "phase2_second_order_energy",
        "phase2_only_disabled_minus_full",
        set_flags={
            **STATIC_UNSPECIFIED,
            "--adapt-continuation-mode": "phase2_v1",
            "--phase3-backend-cost-mode": "proxy",
            "--phase2-selector-gain-mode": "unit_gain_v1",
        },
        enable_flags=("--phase3-no-batching", "--phase2-no-batching"),
        remove_bool_flags=("--phase3-enable-batching", "--phase2-enable-batching"),
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase3-batch-selection-mode", "--phase3-batch-target-size", "--phase3-batch-size-cap"),
    ),
    Variant(
        "phase2_second_order_only_no_novelty",
        "phase2_novelty",
        "phase2_only_disabled_minus_full",
        set_flags={
            **STATIC_UNSPECIFIED,
            "--adapt-continuation-mode": "phase2_v1",
            "--phase3-backend-cost-mode": "proxy",
            "--phase2-selector-gain-mode": "trust_region_v1",
            "--phase2-gamma-N": "0.0",
            "--phase2-gamma-N-schedule-mode": "fixed",
            "--phase3-novelty-ablation-mode": "no_phase2",
        },
        enable_flags=("--phase3-no-batching", "--phase2-no-batching"),
        remove_bool_flags=("--phase3-enable-batching", "--phase2-enable-batching"),
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase3-batch-selection-mode", "--phase3-batch-target-size", "--phase3-batch-size-cap", "--phase2-gamma-N-schedule-start", "--phase2-gamma-N-schedule-end"),
    ),
    Variant(
        "no_phase3",
        "phase3",
        "disabled_minus_full",
        set_flags={**STATIC_UNSPECIFIED, "--adapt-continuation-mode": "phase2_v1", "--phase3-backend-cost-mode": "proxy"},
        enable_flags=("--phase3-no-batching", "--phase2-no-batching"),
        remove_bool_flags=("--phase3-enable-batching", "--phase2-enable-batching"),
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase3-batch-selection-mode", "--phase3-batch-target-size", "--phase3-batch-size-cap"),
    ),
    Variant(
        "phase1_only_macro_pool",
        "phase2_phase3_and_child_policy",
        "phase1_only",
        matrix_label="C_macro_only",
        child_policy="macro_only",
        symmetry_policy="not_applicable",
        set_flags={**STATIC_UNSPECIFIED, "--adapt-continuation-mode": "phase1_v1", "--phase3-backend-cost-mode": "proxy"},
        enable_flags=("--phase3-no-batching", "--phase2-no-batching"),
        remove_bool_flags=("--phase3-enable-batching", "--phase2-enable-batching"),
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase3-batch-selection-mode", "--phase3-batch-target-size", "--phase3-batch-size-cap"),
    ),
    Variant(
        "phase1_only_singleton_pool",
        "phase2_phase3_and_child_policy",
        "phase1_only",
        matrix_label="B_common_phase0_singleton_hard_guard",
        child_policy="common_phase0_singleton",
        symmetry_policy="hard_guard",
        set_flags={**STATIC_UNSPECIFIED, "--adapt-continuation-mode": "phase1_v1", "--phase3-backend-cost-mode": "proxy"},
        enable_flags=("--phase3-no-batching", "--phase2-no-batching"),
        remove_bool_flags=("--phase3-enable-batching", "--phase2-enable-batching"),
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--phase3-batch-selection-mode", "--phase3-batch-target-size", "--phase3-batch-size-cap"),
    ),
    Variant(
        "no_beam",
        "beam_search",
        "disabled_minus_full",
        set_flags={**STATIC_UNSPECIFIED, **COMBO_SET, "--adapt-beam-live-branches": "1", "--adapt-beam-children-per-parent": "1"},
        enable_flags=COMBO_ENABLE,
        remove_bool_flags=COMBO_REMOVE_BOOL,
        remove_value_flags=COMBO_REMOVE_VALUE,
        phase3_batch_mode="combinatorial_reduced_plane",
    ),
    Variant(
        "no_lane_global_pool",
        "physical_operator_lane",
        "disabled_minus_full",
        set_flags={**STATIC_UNSPECIFIED, **COMBO_SET, "--static-lane-route": "algebraic"},
        enable_flags=COMBO_ENABLE,
        remove_bool_flags=COMBO_REMOVE_BOOL,
        remove_value_flags=(*COMBO_REMOVE_VALUE, "--physical-lane-shortlist-aggressiveness"),
        phase3_batch_mode="combinatorial_reduced_plane",
        static_lane_route="algebraic",
    ),
)


def _command_rows() -> dict[str, tuple[str, ...]]:
    payload = _read_json(SOURCE_COMMANDS_JSON)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {SOURCE_COMMANDS_JSON}")
    out: dict[str, tuple[str, ...]] = {}
    for row in payload:
        if not isinstance(row, Mapping):
            continue
        regime = str(row.get("regime") or "")
        argv = row.get("argv")
        if regime and isinstance(argv, list) and argv:
            out[regime] = tuple(str(item) for item in argv)
    return out


def load_anchors() -> dict[str, SourceAnchor]:
    command_rows = _command_rows()
    anchors: dict[str, SourceAnchor] = {}
    for spec in REGIMES:
        if spec.display not in command_rows:
            raise ValueError(f"Missing source command for {spec.display} in {SOURCE_COMMANDS_JSON}")
        source_result = _source_result_path(spec.internal)
        for path in (source_result, SOURCE_COMMANDS_JSON, SOURCE_LOCK_MANIFEST):
            if not _repo_path(path).exists():
                raise FileNotFoundError(path)
        anchors[spec.display] = SourceAnchor(
            regime=spec,
            command_args=command_rows[spec.display],
            source_result=source_result,
        )
    return anchors


def _record_id(batch_id: str, regime: RegimeSpec, variant: Variant) -> str:
    return f"{batch_id}__{regime.internal}__snake__physical_operator_lane__{variant.name}"


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


def _base_overrides(record_id: str, variant: Variant) -> dict[str, Any]:
    set_flags: dict[str, str] = {
        "--adapt-segment-id": record_id,
        "--adapt-reopt-policy": "full",
        "--adapt-window-size": FULL_REOPT_WINDOW,
        "--adapt-full-refit-every": "1",
        "--adapt-final-full-refit": "true",
        "--phase3-geometry-window-size": FULL_REOPT_WINDOW,
        "--phase3-runtime-split-max-subset-size": CHILD_SUBSET_CAP,
        "--adapt-beam-live-branches": BEAM_LIVE_BRANCHES,
        "--adapt-beam-children-per-parent": BEAM_CHILDREN_PER_PARENT,
        "--adapt-beam-lambda": BEAM_LAMBDA,
        "--phase1-prune-schur-nomination-route": METRIC_PRUNE_ROUTE,
        "--static-lane-route": variant.static_lane_route,
    }
    if variant.static_lane_route == "physical_operator_type":
        set_flags["--physical-lane-shortlist-aggressiveness"] = "3"
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


def make_row(batch_id: str, anchor: SourceAnchor, variant: Variant) -> dict[str, str]:
    regime = anchor.regime
    record_id = _record_id(batch_id, regime, variant)
    overrides = _base_overrides(record_id, variant)
    row: dict[str, str] = {
        "record_id": record_id,
        "batch_id": batch_id,
        "run_class": "candidate",
        "runnable": "true",
        "blocker": "",
        "method_key": "snake",
        "method_label": "SNAKE",
        "algorithm_id": "static_family_native_adapt_phase3",
        "engine_key": "source_locked_powell_all_regime_snake_mechanism_ablation",
        "engine_label": "Paper-I HH all-regime source-command SNAKE mechanism ablation",
        "spsa_refit_engine": "",
        "budget": "200",
        "display_regime": regime.display,
        "internal_regime": regime.internal,
        "source_map_regime": regime.display,
        "suite_profile": SUITE_PROFILE,
        "case_id": regime.case_id,
        "family": "hh",
        "n_ph_work": regime.n_ph_work,
        "n_ph_ref": regime.n_ph_ref,
        "same_cutoff_exact_gs_energy": anchor.exact_energy,
        "same_cutoff_energy_key_hash": "",
        "exact_reference_energy": "",
        "exact_reference_energy_key_hash": "",
        "exact_reference_n_ph_max": regime.n_ph_ref,
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
        "source_settings_status": "source_command_parent_with_explicit_canonical_overrides",
        "schedule_source_policy": "powell_source_command_no_spsa_schedule",
        "schedule_source_regime": regime.display,
        "schedule_source_method": "SNAKE",
        "schedule_source_json": anchor.source_result,
        "schedule_source_note": "POWELL HH all-regime SNAKE mechanism ablation; SPSA schedule fields intentionally empty.",
        "anchor_source_json": anchor.source_result,
        "anchor_source_sha256": anchor.result_sha256,
        "changed_fields_vs_anchor": ",".join(
            [
                "hh_mechanism_ablation_variant",
                "snake_cli_overrides_json",
                *(sorted(overrides.get("set_flags", {}).keys())),
                *(overrides.get("enable_flags", [])),
                *(overrides.get("remove_bool_flags", [])),
                *(overrides.get("remove_value_flags", [])),
            ]
        ),
        "source_contract_note": (
            "All-regime Paper-I HH SNAKE mechanism ablation. Source commands come from the "
            "physical-operator-lane no-batch parent artifact and are explicitly overridden to "
            "canonical full-refit/full-geometry settings. Phase-III batching cap 3 is the "
            "approved anchor exception; Pauli-child subset cap remains 1."
        ),
        "matrix_label": variant.matrix_label,
        "matrix_role": "all_regime_mechanism_ablation",
        "static_route_id": "unspecified",
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
        "blocked_reason": "",
        "snake_cli_overrides_json": _json(overrides),
        "ordered_batch_beam_label": variant.name,
        "ordered_batch_beam_enabled": "true" if variant.phase3_batch_mode else "false",
        "ordered_batch_beam_run_role": variant.role,
        "phase2_batch_selection_mode": "",
        "phase2_batch_target_size": "",
        "phase2_batch_size_cap": "",
        "phase3_batch_selection_mode": variant.phase3_batch_mode,
        "phase3_batch_target_size": "3" if variant.phase3_batch_mode else "",
        "phase3_batch_size_cap": "3" if variant.phase3_batch_mode else "",
        "adapt_beam_live_branches": str(overrides["set_flags"].get("--adapt-beam-live-branches", BEAM_LIVE_BRANCHES)),
        "adapt_beam_children_per_parent": str(overrides["set_flags"].get("--adapt-beam-children-per-parent", BEAM_CHILDREN_PER_PARENT)),
        "adapt_beam_lambda": BEAM_LAMBDA,
        "ordered_batch_beam_expected_diagnostics_json": _json(
            {
                "phase2_batching": "off",
                "phase3_batch_selection_mode": variant.phase3_batch_mode,
                "phase3_batch_target_size": "3" if variant.phase3_batch_mode else "",
                "phase3_batch_size_cap": "3" if variant.phase3_batch_mode else "",
                "child_subset_cap": CHILD_SUBSET_CAP,
            }
        ),
        "provenance_layer": "physical_operator_lane_source_command_parent",
        "visible_support_csv": "",
        "visible_anchor_result_json": anchor.source_result,
        "visible_effective_command_json": anchor.source_command_json,
        "settings_reused_json": _json(
            {
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
                "static_lane_route": "physical_operator_type",
            }
        ),
        "settings_changed_json": _json(
            {
                "variant": variant.name,
                "feature": variant.feature,
                "set_flags": dict(overrides.get("set_flags", {})),
                "enable_flags": list(overrides.get("enable_flags", [])),
                "remove_bool_flags": list(overrides.get("remove_bool_flags", [])),
                "remove_value_flags": list(overrides.get("remove_value_flags", [])),
            }
        ),
        "settings_change_reason": variant.role,
        "route_variant": variant.name,
        "anchor_gate_status": "queued_phase3_combinatorial_cap3_anchor" if variant.name == "combinatorial_cap3_anchor" else "matched_ablation_row",
        "batch_variant_gate": "phase3_only_batching_cap3" if variant.phase3_batch_mode else "phase3_batching_disabled",
        "work_semantics_expected_json": _json(
            {
                "S_alg_work_scope": "winner_lineage_display_prefix_and_terminal",
                "S_beam_search_total_scope": "all_expanded_scored_branches_when_available",
            }
        ),
        "latex_report_stem": "paper_i_hh_all_regime_snake_mechanism_ablation_20260709",
        "latex_report_output_dir": "output/pdf/paper_i_hh_all_regime_snake_mechanism_ablation_20260709",
        "report_update_policy": "after_fetch_build_latex_pdf_json_csv_sidecars",
        "static_lane_route": variant.static_lane_route,
        "physical_lane_shortlist_aggressiveness": "3" if variant.static_lane_route == "physical_operator_type" else "",
        "source_anchor_family": "physical_operator_lane",
        "source_anchor_role": "all_regime_physical_operator_lane_source_command_parent",
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
        "hh_mechanism_ablation_expected_status": "queued",
        "hh_mechanism_ablation_overrides_json": _json(overrides),
        "hh_mechanism_ablation_plan_md": RUN_PLAN_MD,
    }
    row.update(_runtime_fields(variant))
    row.update(base.output_paths(record_id, "snake"))
    for field in FIELDNAMES:
        row.setdefault(field, "")
    return row


def build_records(batch_id: str = DEFAULT_BATCH_ID) -> list[dict[str, str]]:
    base.configure_batch(batch_id)
    anchors = load_anchors()
    rows: list[dict[str, str]] = []
    for spec in REGIMES:
        anchor = anchors[spec.display]
        for variant in VARIANTS:
            rows.append(make_row(batch_id, anchor, variant))
    return rows


def write_lines(path: Path, rows: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")


def _source_transfer_inputs(anchors: Mapping[str, SourceAnchor]) -> list[str]:
    values: list[str] = [SOURCE_COMMANDS_JSON, SOURCE_LOCK_MANIFEST, SOURCE_SUPPORT_PDF]
    for anchor in anchors.values():
        values.append(anchor.source_result)
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        path = _repo_path(value)
        if path.exists():
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
    manifest_json = input_dir / "paper_i_hh_spsa_budget_ladder_manifest.json"
    src_sanitized_tarball = input_dir / base.SRC_SANITIZED_TARBALL_NAME
    submit_path = ROOT / "chtc" / "phase3_optuna" / f"submit_{batch_id}.sub"

    input_dir.mkdir(parents=True, exist_ok=True)
    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("No all-regime mechanism-ablation records were generated.")
    anchors = load_anchors()
    with records_tsv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDNAMES), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    write_lines(record_ids, (str(row["record_id"]) for row in rows))
    write_lines(
        record_queue,
        (
            f"{row['record_id']}\t{int(row.get('request_memory_mb') or request_memory_mb)}\t{int(row.get('request_disk_mb') or request_disk_mb)}"
            for row in rows
        ),
    )
    write_lines(smoke_ids, (str(rows[0]["record_id"]),))
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
    source_transfer_inputs = _source_transfer_inputs(anchors)
    _patch_submit_transfer_inputs(submit_path, source_transfer_inputs)

    manifest: dict[str, Any] = {
        "schema": "paper_i_hh_all_regime_snake_mechanism_ablation_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
        "run_plan_md": RUN_PLAN_MD,
        "records_tsv": base.rel_or_abs(records_tsv),
        "record_id_file": base.rel_or_abs(record_ids),
        "record_queue": base.rel_or_abs(record_queue),
        "submit_path": base.rel_or_abs(submit_path),
        "record_count": len(rows),
        "runnable_record_count": len(rows),
        "regimes": [spec.display for spec in REGIMES],
        "variants": [variant.name for variant in VARIANTS],
        "expected_runnable_rows": len(REGIMES) * len(VARIANTS),
        "request_cpus": int(request_cpus),
        "request_memory_mb": int(request_memory_mb),
        "request_disk_mb": int(request_disk_mb),
        "max_runtime_s": int(max_runtime_s),
        "source_anchor_family": "physical_operator_lane",
        "source_transfer_input_count": len(source_transfer_inputs),
        "source_transfer_inputs": list(source_transfer_inputs),
        "runnable_record_ids": [row["record_id"] for row in rows],
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
