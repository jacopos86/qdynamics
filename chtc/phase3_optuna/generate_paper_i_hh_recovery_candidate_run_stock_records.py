#!/usr/bin/env python3
"""Generate Paper-I HH recovery/candidate run-stock records.

This generator is deliberately separate from the earlier full-meta singleton
symmetry and ordered-batch diagnostic generators.  The current run stock uses
the visible-row recovery route, but with user-approved cap-3 Phase-III
archival child sets, metric-prune, and nonzero beam cost.  No-batch anchors are
generated separately from gated ordered-batch variants.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chtc.phase3_optuna import generate_paper_i_hh_fullmeta_phase3_singleton_rotosolve_records as base


DEFAULT_SPEC_PATH = (
    ROOT
    / "chtc"
    / "phase3_optuna"
    / "config"
    / "paper_i_hh_recovery_candidate_run_stock_20260705_v1.json"
)
DEFAULT_BATCH_ID = "paper_i_hh_recovery_candidate_20260705_powell_nobatch_wave0"

MATRIX_LABEL = "A_native_staged_singleton_hard_guard"
CHILD_POLICY = "native_phase3_singleton"
SYMMETRY_POLICY = "hard_guard"
SUBSET_SIZE = "3"
METRIC_PRUNE_ROUTE = "metric_regularized_v1"
ADAPT_BEAM_LAMBDA = "0.005"
ADAPT_PARALLEL_GRADIENT_WORKERS = "4"
ADAPT_BEAM_PARENT_WORKERS = "3"
WORK_SEMANTICS_EXPECTED = {
    "work_semantics_version": "snake_terminal_s_alg_winner_lineage_v1",
    "S_alg_work_scope": "winner_lineage_terminal",
    "S_alg_row_policy": "beam_terminal_winner_history_v1",
    "S_beam_search_scope": "all_expanded_scored_branches",
}
NO_BATCH_ROUTE_VARIANT = "nobatch_anchor_cap3_metricprune_beam0p005"
BATCH_VARIANTS = {
    "greedy_batch_cap3": {
        "phase2_batch_selection_mode": "greedy_reduced_plane",
        "phase3_batch_selection_mode": "greedy_reduced_plane",
    },
    "combinatorial_batch_cap3": {
        "phase2_batch_selection_mode": "combinatorial_reduced_plane",
        "phase3_batch_selection_mode": "combinatorial_reduced_plane",
    },
}


@dataclass(frozen=True)
class Stage:
    name: str
    optimizer_overlay: str
    optimizer_order: int
    variant: str


STAGES: dict[str, Stage] = {
    "powell_nobatch_anchor": Stage("powell_nobatch_anchor", "powell", 0, "nobatch_anchor"),
    "spsa_nobatch_anchor": Stage("spsa_nobatch_anchor", "spsa_paper_i_hh", 1, "nobatch_anchor"),
    "rotosolve_nobatch_anchor": Stage("rotosolve_nobatch_anchor", "rotosolve", 2, "nobatch_anchor"),
    "rotosolve_historical_comparators": Stage(
        "rotosolve_historical_comparators",
        "rotosolve",
        3,
        "historical_pool_comparator",
    ),
    "powell_batch_gated": Stage("powell_batch_gated", "powell", 10, "batch_gated"),
    "spsa_batch_gated": Stage("spsa_batch_gated", "spsa_paper_i_hh", 11, "batch_gated"),
    "rotosolve_batch_gated": Stage("rotosolve_batch_gated", "rotosolve", 12, "batch_gated"),
}

COMPARATOR_POLICIES = {
    "geo": base.MatrixPolicy(
        "C_macro_only",
        "Paper-I historical Geo-ADAPT macro-generator comparator pool",
        "macro_only",
        "not_applicable",
    ),
    "append": base.MatrixPolicy(
        "B_common_phase0_singleton_hard_guard",
        "Paper-I historical append filtered macro plus Pauli-child comparator pool",
        "common_phase0_singleton",
        "hard_guard",
    ),
}


def load_spec(path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "paper_i_hh_recovery_candidate_run_stock_v1":
        raise ValueError(f"Unexpected run-stock schema in {path}: {payload.get('schema')!r}")
    return payload


def _regime_wave(spec: Mapping[str, Any], wave_index: int) -> tuple[str, tuple[str, ...]]:
    waves = spec.get("regime_waves")
    if not isinstance(waves, Sequence) or isinstance(waves, (str, bytes)):
        raise ValueError("Spec has no regime_waves sequence.")
    for item in waves:
        if not isinstance(item, Mapping):
            continue
        if int(item.get("index", -1)) != int(wave_index):
            continue
        regimes = tuple(str(regime) for regime in item.get("regimes", ()))
        if not (1 <= len(regimes) <= 2):
            raise ValueError(f"Wave {wave_index} must contain one or two regimes, got {regimes}.")
        unknown = [regime for regime in regimes if regime not in set(base.REGIME_ORDER)]
        if unknown:
            raise ValueError(f"Wave {wave_index} has unknown regimes: {unknown}")
        return str(item.get("label") or f"wave{wave_index}"), regimes
    raise ValueError(f"No regime wave index {wave_index} in spec.")


def _record_id(batch_id: str, *, regime: str, stage: Stage, route_variant: str) -> str:
    return (
        f"{batch_id}__{regime.replace('-', '_')}__snake__{stage.name}"
        f"__{route_variant}__{stage.optimizer_overlay}200__depth30__fullmeta_hva_phase3_childset_cap3"
    )


def _json_dump(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _merge_base_overrides(row: Mapping[str, str], *, route_variant: str) -> dict[str, Any]:
    payload = json.loads(str(row["snake_cli_overrides_json"]))
    set_flags = dict(payload.get("set_flags") or {})
    set_flags.update(
        {
            "--phase3-runtime-split-max-subset-size": SUBSET_SIZE,
            "--phase1-prune-schur-nomination-route": METRIC_PRUNE_ROUTE,
            "--adapt-beam-lambda": ADAPT_BEAM_LAMBDA,
            "--adapt-parallel-gradient-workers": ADAPT_PARALLEL_GRADIENT_WORKERS,
            "--adapt-beam-parent-workers": ADAPT_BEAM_PARENT_WORKERS,
        }
    )
    enable_flags = list(payload.get("enable_flags") or [])
    remove_bool_flags = list(payload.get("remove_bool_flags") or [])
    remove_value_flags = list(payload.get("remove_value_flags") or [])

    if route_variant == NO_BATCH_ROUTE_VARIANT:
        for flag in ("--phase2-no-batching", "--phase3-no-batching"):
            if flag not in enable_flags:
                enable_flags.append(flag)
        for flag in ("--phase2-enable-batching", "--phase3-enable-batching"):
            if flag not in remove_bool_flags:
                remove_bool_flags.append(flag)
        for flag in (
            "--phase2-batch-selection-mode",
            "--phase3-batch-selection-mode",
            "--phase2-batch-target-size",
            "--phase2-batch-size-cap",
            "--phase3-batch-size-cap",
            "--phase3-source-lock-preferred-sequence",
        ):
            if flag not in remove_value_flags:
                remove_value_flags.append(flag)
    else:
        variant = BATCH_VARIANTS[route_variant]
        set_flags.update(
            {
                "--static-route-id": "unspecified",
                "--phase2-batch-selection-mode": variant["phase2_batch_selection_mode"],
                "--phase3-batch-selection-mode": variant["phase3_batch_selection_mode"],
                "--phase2-batch-target-size": "3",
                "--phase2-batch-size-cap": "3",
                "--adapt-beam-live-branches": "3",
                "--adapt-beam-children-per-parent": "3",
            }
        )
        for flag in ("--phase2-enable-batching", "--phase3-enable-batching"):
            if flag not in enable_flags:
                enable_flags.append(flag)
        for flag in ("--phase2-no-batching", "--phase3-no-batching"):
            if flag not in remove_bool_flags:
                remove_bool_flags.append(flag)
        if "--phase3-source-lock-preferred-sequence" not in remove_value_flags:
            remove_value_flags.append("--phase3-source-lock-preferred-sequence")

    payload["set_flags"] = set_flags
    payload["enable_flags"] = list(dict.fromkeys(enable_flags))
    payload["remove_bool_flags"] = list(dict.fromkeys(remove_bool_flags))
    payload["remove_value_flags"] = list(dict.fromkeys(remove_value_flags))
    return payload


def _settings_reused(stage: Stage) -> dict[str, Any]:
    return {
        "provenance_layer": "visible_row",
        "pool_contract": "full_meta_unfiltered",
        "hva_policy": "included_unfiltered_full_meta",
        "optimizer_overlay": stage.optimizer_overlay,
        "phase3_runtime_split_mode": base.PAULI_CHILD_MODE,
        "phase3_runtime_split_selection_mode": base.SNAKE_RUNTIME_SELECTION_MODE,
        "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "adapt_child_pool_expansion_mode": "off",
        "shared_pauli_pool_mode": "off",
        "maxiter": base.DEFAULT_BUDGET,
        "max_depth": 30,
    }


def _settings_changed(route_variant: str) -> dict[str, Any]:
    if route_variant == "rotosolve_historical_comparator":
        return {
            "optimizer_overlay": "rotosolve",
            "geo_pool_policy": "paper_i_historical_macro_only",
            "append_pool_policy": "paper_i_historical_macro_filtered_plus_pauli_child_filtered",
            "methods": "geo_append_only",
            "snake_settings": "not_applicable",
        }
    changed: dict[str, Any] = {
        "--phase3-runtime-split-max-subset-size": SUBSET_SIZE,
        "--phase1-prune-schur-nomination-route": METRIC_PRUNE_ROUTE,
        "--adapt-beam-lambda": ADAPT_BEAM_LAMBDA,
        "--adapt-parallel-gradient-workers": ADAPT_PARALLEL_GRADIENT_WORKERS,
        "--adapt-beam-parent-workers": ADAPT_BEAM_PARENT_WORKERS,
    }
    if route_variant == NO_BATCH_ROUTE_VARIANT:
        changed["batching"] = "disabled"
    else:
        changed.update(BATCH_VARIANTS[route_variant])
        changed["--phase2-batch-target-size"] = "3"
        changed["--phase2-batch-size-cap"] = "3"
        changed["--adapt-beam-live-branches"] = "3"
        changed["--adapt-beam-children-per-parent"] = "3"
    return changed


def _build_comparator_row(
    *,
    batch_id: str,
    source: Mapping[str, str],
    spec: Mapping[str, Any],
    stage: Stage,
    wave_index: int,
    wave_label: str,
) -> dict[str, str]:
    method = str(source["method_key"])
    policy = COMPARATOR_POLICIES[method]
    overlay = base.OPTIMIZER_OVERLAYS[stage.optimizer_overlay]
    row = base.make_row(
        batch_id,
        source,
        policy=policy,
        budget=base.DEFAULT_BUDGET,
        max_depth=30,
        overlay=overlay,
        strong_strong_snake_start_mode=base.STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO,
    )
    route_variant = "rotosolve_historical_comparator"
    record_id = (
        f"{batch_id}__{str(source['display_regime']).replace('-', '_')}__{method}"
        f"__{stage.name}__{policy.label}__rotosolve200__depth30__paper_i_historical_pool"
    )
    row["record_id"] = record_id
    row.update(base.output_paths(record_id, method))
    row["run_class"] = "candidate"
    row["matrix_label"] = policy.label
    row["matrix_role"] = policy.role
    row["pool_contract"] = "full_meta_unfiltered"
    row["hh_adaptive_pool_profile"] = "full_meta_unfiltered"
    row["adapt_pool_class_filter_json"] = "off"
    row["provenance_layer"] = "visible_row"
    row["visible_support_csv"] = str(spec["visible_support_csv"])
    row["visible_anchor_result_json"] = str(source.get("source_json") or "")
    row["visible_effective_command_json"] = str(source.get("source_command_sh") or "")
    row["settings_reused_json"] = _json_dump(
        {
            "provenance_layer": "visible_row",
            "optimizer_overlay": stage.optimizer_overlay,
            "pool_contract": "full_meta_unfiltered",
            "hva_policy": "included_unfiltered_full_meta",
            "method": method,
            "historical_pool_policy": (
                "paper_i_historical_macro_only"
                if method == "geo"
                else "paper_i_historical_macro_filtered_plus_pauli_child_filtered"
            ),
            "maxiter": base.DEFAULT_BUDGET,
            "max_depth": 30,
        }
    )
    row["settings_changed_json"] = _json_dump(_settings_changed(route_variant))
    row["settings_change_reason"] = (
        "User-approved ROTOSOLVE-only comparator extension using Paper-I historical "
        "comparator pools: Geo macro-only and append filtered macro plus Pauli-child pool."
    )
    row["route_variant"] = route_variant
    row["anchor_gate_status"] = "rotosolve_comparator_not_snake_anchor"
    row["batch_variant_gate"] = "not_batch_variant"
    row["ordered_batch_beam_enabled"] = "false"
    row["ordered_batch_beam_run_role"] = "not_applicable_comparator"
    row["ordered_batch_beam_label"] = route_variant
    row["adapt_beam_lambda"] = ""
    row["work_semantics_expected_json"] = _json_dump(
        {
            "work_semantics_version": "generic_static_comparator_components_v1",
            "S_alg_work_scope": "terminal_generic_adapt_history",
            "S_alg_row_policy": "explicit_generic_static_single_components",
        }
    )
    report = spec["canonical_report"]
    row["latex_report_stem"] = str(report["latex_report_stem"])
    row["latex_report_output_dir"] = str(report["latex_report_output_dir"])
    row["report_update_policy"] = str(report["report_update_policy"])
    row["regime_wave_index"] = str(int(wave_index))
    row["regime_wave_label"] = wave_label
    row["optimizer_stage_order"] = str(stage.optimizer_order)
    changed = [field for field in str(row.get("changed_fields_vs_anchor") or "").split(",") if field]
    changed.extend(
        [
            "provenance_layer",
            "rotosolve_historical_comparator_pool",
            "latex_report_contract",
        ]
    )
    row["changed_fields_vs_anchor"] = ",".join(dict.fromkeys(changed))
    for field in base.OUTPUT_FIELDNAMES:
        row.setdefault(field, "")
    return row


def _build_snake_row(
    *,
    batch_id: str,
    source: Mapping[str, str],
    spec: Mapping[str, Any],
    stage: Stage,
    wave_index: int,
    wave_label: str,
    route_variant: str,
) -> dict[str, str]:
    overlay = base.OPTIMIZER_OVERLAYS[stage.optimizer_overlay]
    policy = base.MatrixPolicy(
        MATRIX_LABEL,
        "Visible-row archival Phase-III child-set cap-3 recovery/candidate route",
        CHILD_POLICY,
        SYMMETRY_POLICY,
    )
    row = base.make_row(
        batch_id,
        source,
        policy=policy,
        budget=base.DEFAULT_BUDGET,
        max_depth=30,
        overlay=overlay,
        strong_strong_snake_start_mode=base.STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO,
    )
    record_id = _record_id(batch_id, regime=str(source["display_regime"]), stage=stage, route_variant=route_variant)
    row["record_id"] = record_id
    row.update(base.output_paths(record_id, "snake"))
    row["run_class"] = "candidate"
    row["matrix_label"] = MATRIX_LABEL
    row["matrix_role"] = "visible-row recovery/candidate cap-3 route"
    row["child_policy"] = CHILD_POLICY
    row["symmetry_policy"] = SYMMETRY_POLICY
    row["child_subset_size"] = SUBSET_SIZE
    row["snake_phase3_runtime_split_max_subset_size"] = SUBSET_SIZE
    row["pool_contract"] = "full_meta_unfiltered"
    row["hh_adaptive_pool_profile"] = "full_meta_unfiltered"
    row["adapt_pool_class_filter_json"] = "off"
    row["snake_cli_overrides_json"] = _json_dump(_merge_base_overrides(row, route_variant=route_variant))
    row["ordered_batch_beam_enabled"] = "false" if route_variant == NO_BATCH_ROUTE_VARIANT else "true"
    row["ordered_batch_beam_run_role"] = "nobatch_anchor" if route_variant == NO_BATCH_ROUTE_VARIANT else "gated_batch_variant"
    row["ordered_batch_beam_label"] = route_variant
    if route_variant in BATCH_VARIANTS:
        row["static_route_id"] = "unspecified"
        row["phase2_batch_selection_mode"] = BATCH_VARIANTS[route_variant]["phase2_batch_selection_mode"]
        row["phase2_batch_target_size"] = "3"
        row["phase2_batch_size_cap"] = "3"
        row["adapt_beam_live_branches"] = "3"
        row["adapt_beam_children_per_parent"] = "3"
    else:
        row["phase2_batch_selection_mode"] = ""
        row["phase2_batch_target_size"] = ""
        row["phase2_batch_size_cap"] = ""
        row["adapt_beam_live_branches"] = ""
        row["adapt_beam_children_per_parent"] = ""
    row["adapt_beam_lambda"] = ADAPT_BEAM_LAMBDA
    row["provenance_layer"] = "visible_row"
    row["visible_support_csv"] = str(spec["visible_support_csv"])
    row["visible_anchor_result_json"] = str(source.get("source_json") or "")
    row["visible_effective_command_json"] = str(source.get("source_command_sh") or "")
    row["settings_reused_json"] = _json_dump(_settings_reused(stage))
    row["settings_changed_json"] = _json_dump(_settings_changed(route_variant))
    row["settings_change_reason"] = (
        "User-approved Paper-I HH recovery/candidate perturbation: cap-3 archival "
        "Phase-III child sets, metric-prune route, adapt_beam_lambda=0.005, and "
        "worker parallelism; batching only for gated batch variants."
    )
    row["route_variant"] = route_variant
    row["anchor_gate_status"] = "anchor_row" if route_variant == NO_BATCH_ROUTE_VARIANT else "requires_matching_nobatch_anchor_pass"
    row["batch_variant_gate"] = "not_batch_variant" if route_variant == NO_BATCH_ROUTE_VARIANT else "gated_after_anchor"
    row["work_semantics_expected_json"] = _json_dump(WORK_SEMANTICS_EXPECTED)
    report = spec["canonical_report"]
    row["latex_report_stem"] = str(report["latex_report_stem"])
    row["latex_report_output_dir"] = str(report["latex_report_output_dir"])
    row["report_update_policy"] = str(report["report_update_policy"])
    row["regime_wave_index"] = str(int(wave_index))
    row["regime_wave_label"] = wave_label
    row["optimizer_stage_order"] = str(stage.optimizer_order)
    changed = [field for field in str(row.get("changed_fields_vs_anchor") or "").split(",") if field]
    changed.extend(
        [
            "provenance_layer",
            "phase3_runtime_split_max_subset_size_3",
            "phase1_prune_schur_nomination_route",
            "adapt_beam_lambda",
            "runtime_worker_parallelism",
            "route_variant",
            "latex_report_contract",
        ]
    )
    if route_variant != NO_BATCH_ROUTE_VARIANT:
        changed.extend(["ordered_batch_variant", "batch_selection_mode", "batch_cap3"])
    row["changed_fields_vs_anchor"] = ",".join(dict.fromkeys(changed))
    for field in base.OUTPUT_FIELDNAMES:
        row.setdefault(field, "")
    return row


def build_records(
    batch_id: str,
    *,
    spec_path: Path = DEFAULT_SPEC_PATH,
    stage_name: str = "powell_nobatch_anchor",
    wave_index: int = 0,
    include_batch_variants: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    spec = load_spec(spec_path)
    if stage_name not in STAGES:
        raise ValueError(f"Unknown stage {stage_name!r}; expected {sorted(STAGES)}")
    stage = STAGES[stage_name]
    wave_label, regimes = _regime_wave(spec, wave_index)
    base.configure_batch(batch_id)
    sources = base.source_rows()
    route_variants = (NO_BATCH_ROUTE_VARIANT,)
    if stage.variant == "batch_gated":
        requested = tuple(include_batch_variants or tuple(BATCH_VARIANTS))
        unknown = [variant for variant in requested if variant not in BATCH_VARIANTS]
        if unknown:
            raise ValueError(f"Unknown batch variants {unknown}; expected {sorted(BATCH_VARIANTS)}")
        route_variants = requested
    if stage.variant == "historical_pool_comparator":
        rows = []
        sources = base.source_rows()
        for regime in regimes:
            for method in ("geo", "append"):
                key = (regime, method)
                if key not in sources:
                    raise ValueError(f"Missing {method} source row for {regime!r}.")
                rows.append(
                    _build_comparator_row(
                        batch_id=batch_id,
                        source=sources[key],
                        spec=spec,
                        stage=stage,
                        wave_index=wave_index,
                        wave_label=wave_label,
                    )
                )
        preflight_records(rows, stage=stage)
        return rows
    rows: list[dict[str, str]] = []
    for regime in regimes:
        key = (regime, "snake")
        if key not in sources:
            raise ValueError(f"Missing SNAKE source row for {regime!r}.")
        for route_variant in route_variants:
            rows.append(
                _build_snake_row(
                    batch_id=batch_id,
                    source=sources[key],
                    spec=spec,
                    stage=stage,
                    wave_index=wave_index,
                    wave_label=wave_label,
                    route_variant=route_variant,
                )
            )
    preflight_records(rows, stage=stage)
    return rows


def _arg_value(args: Sequence[str], flag: str) -> str | None:
    tokens = list(args)
    if flag not in tokens:
        return None
    idx = tokens.index(flag)
    if idx >= len(tokens) - 1:
        raise ValueError(f"Flag {flag} has no value.")
    return str(tokens[idx + 1])


def preflight_records(records: Sequence[Mapping[str, str]], *, stage: Stage) -> None:
    if not records:
        raise ValueError("No records generated.")
    regimes = {str(row.get("display_regime") or "") for row in records}
    if len(regimes) > 2:
        raise ValueError(f"Generated records span more than two regimes: {sorted(regimes)}")
    for row in records:
        route_variant = str(row.get("route_variant") or "")
        if stage.variant != "historical_pool_comparator" and str(row.get("method_key") or "") != "snake":
            raise ValueError(f"Recovery run stock currently supports SNAKE rows only: {row.get('record_id')}")
        if stage.variant == "historical_pool_comparator":
            method = str(row.get("method_key") or "")
            if stage.optimizer_overlay != "rotosolve" or method not in {"geo", "append"}:
                raise ValueError(f"Historical comparator stage drift: {row.get('record_id')}")
            if route_variant != "rotosolve_historical_comparator":
                raise ValueError(f"Historical comparator route drift: {row.get('record_id')}")
            if str(row.get("pool_contract") or "") != "full_meta_unfiltered":
                raise ValueError(f"Comparator pool contract drift: {row.get('record_id')}")
            if str(row.get("adapt_pool_class_filter_json") or "") != "off":
                raise ValueError(f"Comparator class-filter drift: {row.get('record_id')}")
            if "hh_full_meta_minus_hva_class_filter.json" in json.dumps(dict(row)):
                raise ValueError(f"Minus-HVA filter leaked into comparator row: {row.get('record_id')}")
            if method == "geo" and row.get("matrix_label") != "C_macro_only":
                raise ValueError(f"Geo comparator must use C_macro_only: {row.get('record_id')}")
            if method == "append" and row.get("matrix_label") != "B_common_phase0_singleton_hard_guard":
                raise ValueError(f"Append comparator must use B_common_phase0_singleton_hard_guard: {row.get('record_id')}")
            if str(row.get("adapt_optimizer_kind") or "") != "rotosolve":
                raise ValueError(f"Comparator optimizer drift: {row.get('record_id')}")
            continue
        if str(row.get("pool_contract") or "") != "full_meta_unfiltered":
            raise ValueError(f"Pool contract drift: {row.get('record_id')}")
        if str(row.get("adapt_pool_class_filter_json") or "") != "off":
            raise ValueError(f"Class-filter drift: {row.get('record_id')}")
        if "hh_full_meta_minus_hva_class_filter.json" in json.dumps(dict(row)):
            raise ValueError(f"Minus-HVA filter leaked into row: {row.get('record_id')}")
        if str(row.get("snake_phase3_runtime_split_max_subset_size") or "") != SUBSET_SIZE:
            raise ValueError(f"Phase-III split cap drift: {row.get('record_id')}")
        overrides = json.loads(str(row["snake_cli_overrides_json"]))
        set_flags = overrides.get("set_flags") or {}
        remove_values = set(overrides.get("remove_value_flags") or [])
        if set_flags.get("--phase1-prune-schur-nomination-route") != METRIC_PRUNE_ROUTE:
            raise ValueError(f"Metric-prune route missing: {row.get('record_id')}")
        if set_flags.get("--adapt-beam-lambda") != ADAPT_BEAM_LAMBDA:
            raise ValueError(f"Beam lambda drift: {row.get('record_id')}")
        if "--phase3-source-lock-preferred-sequence" not in remove_values:
            raise ValueError(f"Preferred-sequence source lock not removed: {row.get('record_id')}")
        if route_variant == NO_BATCH_ROUTE_VARIANT:
            enable_flags = set(overrides.get("enable_flags") or [])
            remove_bool = set(overrides.get("remove_bool_flags") or [])
            if not {"--phase2-no-batching", "--phase3-no-batching"}.issubset(enable_flags):
                raise ValueError(f"No-batch flags missing: {row.get('record_id')}")
            if not {"--phase2-enable-batching", "--phase3-enable-batching"}.issubset(remove_bool):
                raise ValueError(f"Batching enable flags not removed: {row.get('record_id')}")
            if str(row.get("ordered_batch_beam_enabled") or "") != "false":
                raise ValueError(f"No-batch anchor marked ordered-batch enabled: {row.get('record_id')}")
        else:
            if route_variant not in BATCH_VARIANTS:
                raise ValueError(f"Unknown route variant {route_variant!r}: {row.get('record_id')}")
            if str(row.get("ordered_batch_beam_enabled") or "") != "true":
                raise ValueError(f"Batch variant not marked ordered-batch enabled: {row.get('record_id')}")
            if row.get("phase2_batch_selection_mode") != BATCH_VARIANTS[route_variant]["phase2_batch_selection_mode"]:
                raise ValueError(f"Batch-selection drift: {row.get('record_id')}")

        if os.environ.get("PAPER_I_HH_RECOVERY_RUNNER_AUDIT") == "1":
            from chtc.phase3_optuna import run_paper_i_hh_spsa_budget_ladder_cell as runner

            source_cmd, effective_cmd, audit = runner.build_snake_source_locked_command(
                row,
                ROOT / "tmp" / "paper_i_hh_recovery_candidate_preflight" / str(row["record_id"]),
            )
            del source_cmd
            if audit.get("status") != "pass":
                raise ValueError(
                    "Source-lock command audit failed for "
                    f"{row.get('record_id')}: {audit.get('non_allowed_flag_changes')}"
                )
            if _arg_value(effective_cmd, "--phase3-runtime-split-max-subset-size") != SUBSET_SIZE:
                raise ValueError(f"Effective command missing cap 3: {row.get('record_id')}")
            if _arg_value(effective_cmd, "--phase1-prune-schur-nomination-route") != METRIC_PRUNE_ROUTE:
                raise ValueError(f"Effective command missing metric-prune route: {row.get('record_id')}")
            if _arg_value(effective_cmd, "--adapt-beam-lambda") != ADAPT_BEAM_LAMBDA:
                raise ValueError(f"Effective command missing beam lambda: {row.get('record_id')}")
            if "--phase3-source-lock-preferred-sequence" in effective_cmd:
                raise ValueError(f"Effective command contains preferred-sequence source lock: {row.get('record_id')}")
            if route_variant == NO_BATCH_ROUTE_VARIANT:
                if "--phase2-enable-batching" in effective_cmd or "--phase3-enable-batching" in effective_cmd:
                    raise ValueError(f"Effective no-batch command still enables batching: {row.get('record_id')}")
                if "--phase2-no-batching" not in effective_cmd or "--phase3-no-batching" not in effective_cmd:
                    raise ValueError(f"Effective no-batch command does not disable batching: {row.get('record_id')}")
    if stage.variant == "nobatch_anchor" and any(
        str(row.get("route_variant") or "") != NO_BATCH_ROUTE_VARIANT for row in records
    ):
        raise ValueError("No-batch stage generated non-anchor route variants.")


def write_records(
    batch_id: str,
    records: Sequence[dict[str, str]],
    *,
    request_cpus: int,
    request_memory_mb: int,
    request_disk_mb: int,
    max_runtime_s: int,
    spec_path: Path = DEFAULT_SPEC_PATH,
    stage_name: str,
    wave_index: int,
) -> dict[str, Any]:
    manifest = base.write_records(
        batch_id,
        records,
        budget=base.DEFAULT_BUDGET,
        max_depth=30,
        request_cpus=int(request_cpus),
        request_memory_mb=int(request_memory_mb),
        request_disk_mb=int(request_disk_mb),
        max_runtime_s=int(max_runtime_s),
        strong_strong_snake_start_mode=base.STRONG_STRONG_SNAKE_START_MODE_DEPTH_ZERO,
    )
    spec = load_spec(spec_path)
    comparator_stage = stage_name == "rotosolve_historical_comparators"
    source_contract = manifest.setdefault("source_contract", {})
    if isinstance(source_contract, dict):
        if comparator_stage:
            source_contract["note"] = (
                "Paper-I HH recovery/candidate ROTOSOLVE comparator extension. "
                "Uses user-approved historical comparator pools: Geo macro-only "
                "and append filtered macro plus Pauli-child pool. SNAKE cap-3, "
                "metric-prune, and beam-lambda settings are not applied to these "
                "non-SNAKE comparator rows."
            )
        else:
            source_contract["note"] = (
                "Paper-I HH recovery/candidate run stock. Starts from the visible-row route "
                "and applies only user-approved cap-3 archival Phase-III child sets, "
                "metric-prune, adapt_beam_lambda=0.005, worker parallelism, and any "
                "explicitly requested gated batching variant."
            )
        source_contract["visible_provenance_doc"] = str(spec["visible_provenance_doc"])
        source_contract["visible_support_csv"] = str(spec["visible_support_csv"])
        source_contract["provenance_layer"] = "visible_row"
        snake_policy = source_contract.get("snake_child_policy")
        if isinstance(snake_policy, dict) and not comparator_stage:
            snake_policy["phase3_runtime_split_max_subset_size"] = int(SUBSET_SIZE)
            snake_policy["phase3_runtime_split_contract_note"] = (
                "This recovery/candidate stock intentionally uses cap 3. Older "
                "singleton wording in compatibility filenames must not be read as "
                "the executable subset-size contract."
            )
        for matrix_policy in source_contract.get("matrix_policies", []) or []:
            if isinstance(matrix_policy, dict) and matrix_policy.get("matrix_label") == MATRIX_LABEL:
                matrix_policy["matrix_role"] = "visible-row recovery/candidate cap-3 route"
                matrix_policy["child_subset_size"] = int(SUBSET_SIZE)
                matrix_policy["child_policy_note"] = (
                    "Effective Phase-III runtime split max subset size is 3."
                )
    manifest["schema"] = "paper_i_hh_recovery_candidate_run_stock_manifest_v1"
    manifest["run_stock"] = {
        "schema": "paper_i_hh_recovery_candidate_run_stock_manifest_extension_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "spec_path": base.rel_or_abs(spec_path),
        "stage": stage_name,
        "wave_index": int(wave_index),
        "provenance_layer": "visible_row",
        "visible_support_csv": spec["visible_support_csv"],
        "latex_report_stem": spec["canonical_report"]["latex_report_stem"],
        "latex_report_output_dir": spec["canonical_report"]["latex_report_output_dir"],
        "report_update_policy": spec["canonical_report"]["report_update_policy"],
        "settings_changed": _settings_changed(str(records[0]["route_variant"])),
        "work_semantics_expected": (
            {
                "work_semantics_version": "generic_static_comparator_components_v1",
                "S_alg_work_scope": "terminal_generic_adapt_history",
                "S_alg_row_policy": "explicit_generic_static_single_components",
            }
            if comparator_stage
            else WORK_SEMANTICS_EXPECTED
        ),
        "paper_facing_status": "candidate_pending_completed_run_evidence",
        "promotion_status": "not_promoted_user_decides",
    }
    manifest_path = (
        ROOT
        / "chtc"
        / "phase3_optuna"
        / "input"
        / batch_id
        / "paper_i_hh_spsa_budget_ladder_manifest.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    parser.add_argument("--stage", choices=tuple(STAGES), default="powell_nobatch_anchor")
    parser.add_argument("--wave-index", type=int, default=0)
    parser.add_argument("--batch-variant", choices=tuple(BATCH_VARIANTS), action="append", default=[])
    parser.add_argument("--request-cpus", type=int, default=1)
    parser.add_argument("--request-memory-mb", type=int, default=32768)
    parser.add_argument("--request-disk-mb", type=int, default=61440)
    parser.add_argument("--max-runtime-s", type=int, default=172800)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    records = build_records(
        str(args.batch_id),
        spec_path=args.spec,
        stage_name=str(args.stage),
        wave_index=int(args.wave_index),
        include_batch_variants=tuple(args.batch_variant or ()),
    )
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_pass",
                    "batch_id": str(args.batch_id),
                    "stage": str(args.stage),
                    "wave_index": int(args.wave_index),
                    "record_count": len(records),
                    "record_ids": [row["record_id"] for row in records],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    manifest = write_records(
        str(args.batch_id),
        records,
        request_cpus=int(args.request_cpus),
        request_memory_mb=int(args.request_memory_mb),
        request_disk_mb=int(args.request_disk_mb),
        max_runtime_s=int(args.max_runtime_s),
        spec_path=args.spec,
        stage_name=str(args.stage),
        wave_index=int(args.wave_index),
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
