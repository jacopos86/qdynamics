#!/usr/bin/env python3
"""Generate bounded fixed-policy HH strong-weak SNAKE ablation records.

This is not an Optuna recalibration matrix.  It freezes the bounded local
HH strong-weak policy and emits one full row plus one-disabled-mechanism rows.
The comparison metric is terminal disabled-minus-full behavior at a fixed
depth cap, not target crossing.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import generate_routeA_phase0_nph1_oracle_records as nph1_oracle  # noqa: E402
from chtc.phase3_optuna.paper_i_clean_ladder_contract import (  # noqa: E402
    PAPER_I_CLEAN_TAU_PHYS,
    PAPER_I_CLEAN_TAU_TIGHT,
    PHONON_CUTOFF_TSV_FIELDS,
)
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
    table_i_executable_specs,
)
from pipelines.static_adapt.route_identity import (  # noqa: E402
    ROUTE_ID_A,
    ROUTE_ID_UNSPECIFIED,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_ID = "paper_i_hh_strong_weak_nph2_bounded_feature_ablation_20260605_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID

ABLATION_RECORDS = "paper_i_hh_strong_weak_bounded_feature_ablation_records.tsv"
ABLATION_IDS = "paper_i_hh_strong_weak_bounded_feature_ablation_record_ids.txt"
MANIFEST = "paper_i_hh_strong_weak_bounded_feature_ablation_manifest.json"
REFERENCE_CACHE = "paper_i_hh_strong_weak_bounded_feature_ablation_reference_energy_cache.json"
SUBMIT_ABLATION = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"

HH_BENCHMARK_ID = "hh_L2_nph2_three_model_sym_strong_weak"
HH_FAMILY = "hh"
N_PH_WORK = 2
N_PH_REF = 5
ABLATION_OBJECTIVE_MODE = "terminal_proxy"
ABLATION_REQUIRED_TARGET_PROFILE = "none"
ABLATION_RUN_CLASS = "candidate_hh_strong_weak_bounded_feature_ablation"
ABLATION_POLICY_PROFILE = "paper_i_hh_strong_weak_terminal_proxy_depth16_fixed_policy_v1"
ABLATION_TRIAL_TIMEOUT_SEC = "21600"
ABLATION_COMPILE_TIMEOUT_SEC = "1200"
ABLATION_PARALLEL_GRADIENT_WORKERS = 4
ABLATION_BEAM_PARENT_WORKERS = 1
ABLATION_N_TRIALS = 1
ABLATION_SEED = 97208

_EXTRA_FIELDNAMES = (
    "route_base_pool_display_name",
    "discovery_objective_mode",
    "phase3_adapt_beam_parent_workers",
    "trial_param_overrides_json",
    "hh_ablation_role",
    "hh_ablation_policy_profile",
    "hh_ablation_feature",
    "hh_ablation_disabled_mechanism",
    "hh_ablation_variant_note",
    "hh_ablation_full_record_id",
    "hh_ablation_no_per_row_recalibration",
    "hh_ablation_terminal_metric",
    "hh_ablation_table_target_status",
    *PHONON_CUTOFF_TSV_FIELDS,
)
FIELDNAMES = tuple(dict.fromkeys((*nph1_oracle.FIELDNAMES, *_EXTRA_FIELDNAMES)))


@dataclass(frozen=True)
class VariantSpec:
    name: str
    record_suffix: str
    disabled_mechanism: str
    note: str
    static_route_id: str
    row_updates: Mapping[str, str]
    override_updates: Mapping[str, Any]


BASELINE_TRIAL_PARAM_OVERRIDES: dict[str, Any] = {
    "adapt_beam_children_per_parent": 2,
    "adapt_beam_live_branches": 2,
    "adapt_beam_terminated_keep": 2,
    "adapt_drop_floor": 1e-14,
    "adapt_drop_min_depth": 8,
    "adapt_drop_patience": 16,
    "adapt_final_full_refit": True,
    "adapt_full_refit_every": 1,
    "adapt_insertion_mode": "adaptive",
    "adapt_max_depth": 16,
    "adapt_maxiter": 1000,
    "adapt_reopt_policy": "windowed",
    "adapt_window_size": 128,
    "adapt_window_topk": 64,
    "compile_cx_weight": 1.595975599765995,
    "compile_position_shift_weight": 0.0,
    "compile_refit_active_weight": 0.9680081334113348,
    "compile_rotation_step_weight": 0.9680081334113348,
    "compile_sq_weight": 0.9680081334113348,
    "lambda_1q": 0.13999999999999999,
    "lambda_2q": 0.13999999999999999,
    "lambda_compile": 0.051174100504100345,
    "lambda_d": 0.13999999999999999,
    "lambda_leak": 0.0,
    "lambda_measure": 0.018825899495899658,
    "lambda_shot": 0.13999999999999999,
    "lambda_theta": 0.13999999999999999,
    "measure_groups_weight": 1.0,
    "measure_reuse_weight": 1.0,
    "measure_shots_weight": 1.0,
    "opt_dim_cost_scale": 1.0,
    "phase0_algebraic_lane_mode": "weak",
    "phase0_pilot_enabled": True,
    "phase1_maturity_shot_cap": 0,
    "phase1_probe_max_positions": 6,
    "phase1_prune_amplitude_witness_required": True,
    "phase1_prune_cooldown_steps": 1,
    "phase1_prune_enabled": True,
    "phase1_prune_protect_steps": 1,
    "phase1_prune_stale_age": 2,
    "phase2_batch_size_cap": 16,
    "phase2_batch_target_size": 8,
    "phase2_enable_batching": True,
    "phase2_frontier_ratio": 1.0,
    "phase2_gamma_N": 1.0,
    "phase2_gamma_N_schedule_end": None,
    "phase2_gamma_N_schedule_mode": "fixed",
    "phase2_gamma_N_schedule_start": None,
    "phase2_maturity_shot_cap": 0,
    "phase2_motif_bonus_weight": 0.0,
    "phase2_shortlist_fraction": 0.25,
    "phase2_w_depth": 0.13999999999999999,
    "phase2_w_group": 0.13999999999999999,
    "phase2_w_lifetime": 0.05,
    "phase2_w_optdim": 0.13999999999999999,
    "phase2_w_reuse": 0.13999999999999999,
    "phase2_w_shot": 0.13999999999999999,
    "phase3_backend_cost_mode": "auto",
    "phase3_batch_prefilter_mode": "off",
    "phase3_batch_selection_mode": "reduced_plane",
    "phase3_frontier_ratio": 1.0,
    "phase3_maturity_shot_cap": 0,
    "phase3_novelty_ablation_mode": "off",
    "phase3_selector_geometry_mode": "reduced",
    "phase3_selector_policy": "algebraic_nested_v1",
    "phase3_tie_beam_max_branches": 2,
    "phase3_window_relaxation_mode": "reduced",
    "phase_maturity_shot_max": 1,
}

NO_HARDWARE_COST_OVERRIDES: dict[str, Any] = {
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

VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec(
        name="full_snake",
        record_suffix="full_snake",
        disabled_mechanism="none",
        note="Full bounded SNAKE policy; no mechanism disabled.",
        static_route_id=ROUTE_ID_A,
        row_updates={},
        override_updates={},
    ),
    VariantSpec(
        name="no_hardware_cost",
        record_suffix="no_hardware_cost",
        disabled_mechanism="hardware_cost",
        note="Disable hardware, compile, measurement, and Phase-II resource costs only.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={},
        override_updates=NO_HARDWARE_COST_OVERRIDES,
    ),
    VariantSpec(
        name="no_tangent_novelty",
        record_suffix="no_tangent_novelty",
        disabled_mechanism="phase3_tangent_novelty",
        note="Disable Phase-III tangent/collective novelty only.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={"phase3_novelty_ablation_mode": "all"},
        override_updates={"phase3_novelty_ablation_mode": "all"},
    ),
    VariantSpec(
        name="no_phase3_schur_rerank",
        record_suffix="no_phase3_schur_rerank",
        disabled_mechanism="phase3_schur_rerank",
        note="Disable reduced Schur/rerank geometry only by using raw exact geometry.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={"phase3_selector_geometry_mode": "raw_exact"},
        override_updates={"phase3_selector_geometry_mode": "raw_exact"},
    ),
    VariantSpec(
        name="no_active_local_window_refit",
        record_suffix="no_active_local_window_refit",
        disabled_mechanism="active_local_window_refit",
        note="Disable active local window relaxation only.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={"phase3_window_relaxation_mode": "no_relaxation"},
        override_updates={"phase3_window_relaxation_mode": "no_relaxation"},
    ),
    VariantSpec(
        name="no_generator_ablation",
        record_suffix="no_generator_ablation",
        disabled_mechanism="recoverability_generator_pruning",
        note="Disable recoverability generator pruning/ablation only.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={
            "phase1_prune_enabled": "false",
            "phase1_prune_amplitude_witness_required": "false",
        },
        override_updates={
            "phase1_prune_enabled": False,
            "phase1_prune_amplitude_witness_required": False,
        },
    ),
    VariantSpec(
        name="no_batching",
        record_suffix="no_batching",
        disabled_mechanism="phase2_phase3_batching",
        note="Disable Phase-II/Phase-III batching only.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={
            "phase2_enable_batching": "false",
            "phase3_enable_batching": "false",
        },
        override_updates={"phase2_enable_batching": False},
    ),
    VariantSpec(
        name="no_beam_continuation",
        record_suffix="no_beam_continuation",
        disabled_mechanism="beam_continuation",
        note="Disable beam continuation only by forcing one live/child/kept branch.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={},
        override_updates={
            "adapt_beam_live_branches": 1,
            "adapt_beam_children_per_parent": 1,
            "adapt_beam_terminated_keep": 1,
            "phase3_tie_beam_max_branches": 1,
        },
    ),
    VariantSpec(
        name="append_only_adapt_limit",
        record_suffix="append_only_adapt_limit",
        disabled_mechanism="adaptive_insertion_refit_rollback",
        note="Disable adaptive insertion/refit/rollback only; keep other mechanisms unchanged.",
        static_route_id=ROUTE_ID_UNSPECIFIED,
        row_updates={},
        override_updates={
            "adapt_reopt_policy": "append_only",
            "adapt_insertion_mode": "append_only",
            "adapt_full_refit_every": 0,
            "adapt_final_full_refit": False,
            "adapt_window_size": 1,
            "adapt_window_topk": 0,
        },
    ),
)


def _record_token(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_")


def _repo_relative(path: str | Path) -> str:
    path = Path(path)
    if not path.is_absolute():
        return str(path).replace("\\", "/")
    try:
        return str(path.resolve(strict=False).relative_to(REPO_ROOT.resolve(strict=False))).replace("\\", "/")
    except ValueError:
        return str(path)


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _spec() -> Any:
    matches = [
        spec
        for spec in table_i_executable_specs(TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE)
        if str(spec.benchmark_id) == HH_BENCHMARK_ID
    ]
    if len(matches) != 1:
        available = sorted(str(spec.benchmark_id) for spec in table_i_executable_specs(TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE))
        raise ValueError(f"{HH_BENCHMARK_ID}: expected exactly one executable spec; available={available}")
    return matches[0]


def _reference_cache_payload() -> dict[str, Any]:
    records: dict[str, Any] = {}
    spec = _spec()
    for label, nph in (("same_cutoff", N_PH_WORK), ("reference_cutoff", N_PH_REF)):
        energy, key_hash, key = exact_energy_for_spec(spec, n_ph_max=int(nph))
        records[key_hash] = {
            "schema": "static_reference_energy_record_v1",
            "label": label,
            "key_hash": key_hash,
            "key": key,
            "exact_energy": float(energy),
            "source": "pipelines.exact_bench.static_reference_metrics",
            "status": "ok",
        }
    return {
        "schema": "static_reference_energy_cache_v1",
        "purpose": "Reference energies for bounded HH strong-weak SNAKE feature ablations.",
        "record_count": len(records),
        "records": records,
    }


def _reference_fields(*, cache_rel: str) -> dict[str, str]:
    spec = _spec()
    same_energy, same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=N_PH_WORK)
    ref_energy, ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=N_PH_REF)
    return {
        "n_ph_work": str(N_PH_WORK),
        "n_ph_ref": str(N_PH_REF),
        "primary_energy_metric": "higher_cutoff_reference_abs_delta_e",
        "same_cutoff_error_role": "diagnostic_only",
        "paper_i_cutoff_ladder_stage": "",
        "paper_i_ladder_acceptance_threshold": "",
        "paper_i_ladder_requires_prior_failure": "false",
        "paper_i_ladder_escalation_reason": "bounded_feature_ablation",
        "paper_i_ladder_allow_ref5": "true",
        "paper_i_ladder_snake_policy": "snake_only",
        "reference_energy_cache_json": cache_rel,
        "same_cutoff_reference_energy_key": same_key,
        "reference_cutoff_energy_key": ref_key,
        "same_cutoff_exact_gs_energy": repr(float(same_energy)),
        "exact_reference_energy": repr(float(ref_energy)),
        "exact_reference_n_ph_max": str(N_PH_REF),
        "reference_energy_status": "ok",
        "tau_phys": str(float(PAPER_I_CLEAN_TAU_PHYS)),
        "tau_tight": str(float(PAPER_I_CLEAN_TAU_TIGHT)),
    }


def _override_payload(variant: VariantSpec) -> dict[str, Any]:
    merged = dict(BASELINE_TRIAL_PARAM_OVERRIDES)
    merged.update(dict(variant.override_updates))
    return {
        "schema": "phase3_trial_param_overrides_v1",
        "purpose": "Bounded HH strong-weak fixed-policy feature ablation; no per-row recalibration.",
        "ablation_label": variant.name,
        "disabled_mechanism": variant.disabled_mechanism,
        "ablation_note": variant.note,
        "baseline_policy_profile": ABLATION_POLICY_PROFILE,
        "source_local_probe": {
            "record": "tmp/local_feature_probe/hh_tableiii_strong_weak",
            "benchmark_id": HH_BENCHMARK_ID,
            "terminal_primary_abs_delta_e": 0.0777799,
            "terminal_same_cutoff_abs_delta_e": 0.07777,
            "role": "policy_freeze_source_only_not_chct_input",
        },
        "trial_param_overrides": merged,
    }


def _variant_override_path(output_dir: Path, variant: VariantSpec) -> Path:
    return output_dir / f"{variant.record_suffix}_trial_param_overrides.json"


def _base_row_for_variant(output_dir: Path, variant: VariantSpec) -> dict[str, str]:
    cache_rel = _repo_relative(output_dir / REFERENCE_CACHE)
    override_rel = _repo_relative(_variant_override_path(output_dir, variant))
    record_id = f"{BATCH_ID}_{variant.record_suffix}"
    full_record_id = f"{BATCH_ID}_{VARIANTS[0].record_suffix}"
    row = nph1_oracle._base_row(
        record_id=record_id,
        mode="oracle-grid",
        families=(HH_FAMILY,),
        benchmark_ids=(HH_BENCHMARK_ID,),
        n_trials=ABLATION_N_TRIALS,
        seed=ABLATION_SEED,
        policy_search_profile="default",
        meta_feature_profile=STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
        oracle_required_static_route_id="",
        oracle_required_suite_profile="",
        oracle_require_phase0_aware=False,
        oracle_require_compatible_warm_starts=False,
        algorithm_variant="paper_i_hh_strong_weak_bounded_feature_ablation",
    )
    row.update(
        {
            "suite_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
            "static_route_id": variant.static_route_id,
            "route_base_pool_key": "full_meta",
            "route_base_pool_display_name": "full_meta",
            "boson_cutoff": str(N_PH_WORK),
            "boson_cutoffs": "",
            "exact_reference_boson_cutoff": str(N_PH_REF),
            "physics_grid_profile": "paper_i_clean",
            "historical_ledger": "",
            "selected_logical_transfer_mode": "exact_match_v1",
            "selected_logical_source_json": "",
            "selected_logical_route": "standard",
            "oracle_summary_root": "",
            "oracle_enqueue_limit": "",
            "objective_profile": "balanced",
            "target_abs_delta_e": "",
            "objective_energy_weight": "32.0",
            "objective_2q_weight": "0.0",
            "objective_2q_depth_weight": "0.0",
            "objective_depth_weight": "0.0",
            "objective_parameter_weight": "0.0",
            "objective_shot_weight": "0.0",
            "objective_weight_preset": "uniform",
            "required_target_profile": ABLATION_REQUIRED_TARGET_PROFILE,
            "required_target_benchmark_ids": "",
            "required_target_abs_delta_e": "",
            "required_target_penalty": "0.0",
            "discovery_objective_mode": ABLATION_OBJECTIVE_MODE,
            "trial_timeout_sec": ABLATION_TRIAL_TIMEOUT_SEC,
            "compile_timeout_sec": ABLATION_COMPILE_TIMEOUT_SEC,
            "enqueue_default": "true",
            "enqueue_historical": "false",
            "fixed_inner_optimizer": "SPSA",
            "route_evidence_role": "candidate_bounded_feature_ablation_not_main_table_evidence",
            "phase3_adapt_parallel_gradient_workers": str(ABLATION_PARALLEL_GRADIENT_WORKERS),
            "phase3_adapt_beam_parent_workers": str(ABLATION_BEAM_PARENT_WORKERS),
            "phase3_oracle_gradient_mode": "off",
            "phase3_oracle_inner_objective_mode": "exact",
            "phase3_oracle_value_noise_model": "off",
            "phase3_oracle_value_noise_std": "0.0",
            "phase3_adapt_allow_repeats": "true",
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
            "phase2_raw_score_formula": "DeltaE_TR_raw * N2 / (1 + K2)",
            "canonical_score_formula": "DeltaE_TR * N3 / (1 + K3)",
            "primary_selector_score_key": "full_v2_score",
            "auxiliary_terms_primary_mode": "tie_break_only",
            "continuation_mode": "phase3_v1",
            "algebraic_shortlisting_enabled": "true",
            "hardware_resolution_schema": "gradient_resolution_v1",
            "hardware_resolution_mode": "ideal",
            "trial_param_overrides_json": override_rel,
            "hh_ablation_role": ABLATION_RUN_CLASS,
            "hh_ablation_policy_profile": ABLATION_POLICY_PROFILE,
            "hh_ablation_feature": variant.name,
            "hh_ablation_disabled_mechanism": variant.disabled_mechanism,
            "hh_ablation_variant_note": variant.note,
            "hh_ablation_full_record_id": full_record_id,
            "hh_ablation_no_per_row_recalibration": "true",
            "hh_ablation_terminal_metric": "terminal_disabled_minus_full_energy_and_resource_delta",
            "hh_ablation_table_target_status": "diagnostic_same_physics_point_not_table_evidence",
        }
    )
    row.update({key: str(value) for key, value in variant.row_updates.items()})
    row.update(_reference_fields(cache_rel=cache_rel))
    return {key: str(row.get(key, "")) for key in FIELDNAMES}


def build_rows(output_dir: Path | None = None) -> list[dict[str, str]]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    rows = [_base_row_for_variant(output_dir, variant) for variant in VARIANTS]
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != len(VARIANTS):
        raise ValueError(f"expected {len(VARIANTS)} bounded HH ablation rows, got {len(rows)}")
    record_ids = [str(row.get("record_id") or "") for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate record_id in HH bounded ablation rows")
    full = rows[0]
    if full.get("static_route_id") != ROUTE_ID_A:
        raise ValueError("full bounded HH row must declare route_a")
    for row, variant in zip(rows, VARIANTS, strict=True):
        rid = str(row.get("record_id") or "")
        if str(row.get("benchmark_ids") or "") != HH_BENCHMARK_ID:
            raise ValueError(f"{rid}: wrong benchmark_id")
        if str(row.get("suite_profile") or "") != TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE:
            raise ValueError(f"{rid}: wrong suite_profile")
        if str(row.get("physics_grid_profile") or "") != "paper_i_clean":
            raise ValueError(f"{rid}: runner physics_grid_profile must be paper_i_clean")
        if str(row.get("families") or "") != HH_FAMILY:
            raise ValueError(f"{rid}: wrong family")
        if str(row.get("boson_cutoff") or "") != str(N_PH_WORK):
            raise ValueError(f"{rid}: boson_cutoff must be n_ph_work={N_PH_WORK}")
        if str(row.get("exact_reference_boson_cutoff") or "") != str(N_PH_REF):
            raise ValueError(f"{rid}: exact_reference_boson_cutoff must be n_ph_ref={N_PH_REF}")
        if str(row.get("n_ph_work") or "") != str(N_PH_WORK) or str(row.get("n_ph_ref") or "") != str(N_PH_REF):
            raise ValueError(f"{rid}: phonon cutoff fields must be {N_PH_WORK}->{N_PH_REF}")
        if str(row.get("required_target_profile") or "") != ABLATION_REQUIRED_TARGET_PROFILE:
            raise ValueError(f"{rid}: ablation must not use a target-required profile")
        if str(row.get("discovery_objective_mode") or "") != ABLATION_OBJECTIVE_MODE:
            raise ValueError(f"{rid}: ablation must use terminal_proxy objective")
        if str(row.get("n_trials") or "") != str(ABLATION_N_TRIALS):
            raise ValueError(f"{rid}: ablation must use n_trials=1")
        if str(row.get("seed") or "") != str(ABLATION_SEED):
            raise ValueError(f"{rid}: ablation rows must use matched seed")
        if str(row.get("trial_param_overrides_json") or "") == "":
            raise ValueError(f"{rid}: missing trial_param_overrides_json")
        if str(row.get("hh_ablation_no_per_row_recalibration") or "") != "true":
            raise ValueError(f"{rid}: no-per-row-recalibration marker missing")
        if variant is not VARIANTS[0] and str(row.get("static_route_id") or "") != ROUTE_ID_UNSPECIFIED:
            raise ValueError(f"{rid}: disabled mechanism rows must declare static_route_id=unspecified")

    append = next(row for row in rows if row["hh_ablation_feature"] == "append_only_adapt_limit")
    for key in (
        "phase3_novelty_ablation_mode",
        "phase3_selector_geometry_mode",
        "phase3_window_relaxation_mode",
        "phase2_enable_batching",
        "phase3_enable_batching",
        "phase1_prune_enabled",
        "phase1_prune_amplitude_witness_required",
    ):
        if append.get(key) != full.get(key):
            raise ValueError(f"append-only row changed non-append mechanism {key}: {append.get(key)}")


def _tsv_text(rows: Sequence[Mapping[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(FIELDNAMES), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    return buf.getvalue()


def _ids_text(rows: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(row["record_id"]) for row in rows) + "\n"


def _submit_text(*, records_path: Path, record_ids_path: Path, job_batch_name: str) -> str:
    return f"""universe = vanilla
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
request_cpus = {ABLATION_PARALLEL_GRADIENT_WORKERS}
request_memory = 32GB
request_disk = 122880MB
+MaxRuntime = 172800
+JobBatchName = "holstein-{job_batch_name}"
environment = "PHASE3_RECORDS_PATH={_repo_relative(records_path)} PHASE3_TERMINATE_ON_STALE_PROGRESS=1 PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC=3600 PHASE3_PROGRESS_STALE_AFTER_SEC=3600 PHASE3_HEARTBEAT_INTERVAL_SEC=60 PHASE3_SHELL_HEARTBEAT_SEC=60"
queue record_id from {_repo_relative(record_ids_path)}
"""


def _manifest_payload(*, rows: Sequence[Mapping[str, str]], output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "paper_i_hh_strong_weak_bounded_feature_ablation_manifest_v1",
        "batch_id": BATCH_ID,
        "generated_by": "chtc/phase3_optuna/generate_paper_i_hh_strong_weak_bounded_feature_ablation_records.py",
        "run_class": ABLATION_RUN_CLASS,
        "records_tsv": _repo_relative(output_dir / ABLATION_RECORDS),
        "record_ids": [row["record_id"] for row in rows],
        "record_id_file": _repo_relative(output_dir / ABLATION_IDS),
        "submit_ablation": _repo_relative(SUBMIT_ABLATION),
        "reference_energy_cache_json": _repo_relative(output_dir / REFERENCE_CACHE),
        "benchmark_id": HH_BENCHMARK_ID,
        "suite_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_PROFILE,
        "n_ph_work": N_PH_WORK,
        "n_ph_ref": N_PH_REF,
        "n_trials": ABLATION_N_TRIALS,
        "seed": ABLATION_SEED,
        "feature_count_including_full": len(VARIANTS),
        "variant_order": [asdict(variant) for variant in VARIANTS],
        "baseline_trial_param_overrides": dict(BASELINE_TRIAL_PARAM_OVERRIDES),
        "scientific_framing": {
            "not_main_table_evidence": True,
            "target_hit_required": False,
            "comparison_convention": "disabled_minus_full_terminal_metric",
            "no_per_ablation_optuna": True,
            "fixed_depth_cap": int(BASELINE_TRIAL_PARAM_OVERRIDES["adapt_max_depth"]),
            "purpose": "show feature utility on HH strong-weak nph2 by holding the policy fixed and disabling one mechanism at a time",
        },
        "row_contract": {
            "full_static_route_id": ROUTE_ID_A,
            "disabled_static_route_id": ROUTE_ID_UNSPECIFIED,
            "meta_feature_profile": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
            "selected_logical_route": "standard",
            "route_base_pool_key": "full_meta",
            "required_target_profile": ABLATION_REQUIRED_TARGET_PROFILE,
            "discovery_objective_mode": ABLATION_OBJECTIVE_MODE,
            "same_seed_all_rows": True,
            "same_trial_overrides_except_disabled_mechanism": True,
        },
        "progress_contract": {
            "submit_watchdog_env": {
                "PHASE3_TERMINATE_ON_STALE_PROGRESS": "1",
                "PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC": "3600",
                "PHASE3_PROGRESS_STALE_AFTER_SEC": "3600",
                "PHASE3_HEARTBEAT_INTERVAL_SEC": "60",
            },
            "run_task_must_supply_progress_dir": True,
            "current_best_required": True,
        },
    }


def render_artifacts(output_dir: Path | None = None) -> dict[Path, str]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    rows = build_rows(output_dir)
    artifacts: dict[Path, str] = {
        output_dir / ABLATION_RECORDS: _tsv_text(rows),
        output_dir / ABLATION_IDS: _ids_text(rows),
        output_dir / MANIFEST: json.dumps(_manifest_payload(rows=rows, output_dir=output_dir), indent=2, sort_keys=True) + "\n",
        output_dir / REFERENCE_CACHE: json.dumps(_reference_cache_payload(), indent=2, sort_keys=True) + "\n",
        SUBMIT_ABLATION: _submit_text(
            records_path=output_dir / ABLATION_RECORDS,
            record_ids_path=output_dir / ABLATION_IDS,
            job_batch_name=BATCH_ID,
        ),
    }
    for variant in VARIANTS:
        artifacts[_variant_override_path(output_dir, variant)] = (
            json.dumps(_override_payload(variant), indent=2, sort_keys=True) + "\n"
        )
    return artifacts


def write_artifacts(output_dir: Path | None = None) -> dict[str, str]:
    artifacts = render_artifacts(output_dir)
    for path, text in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {path.name: str(path) for path in artifacts}


def check_artifacts(output_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    for path, expected in render_artifacts(output_dir).items():
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"missing generated artifact: {path}")
            continue
        if actual != expected:
            errors.append(f"generated artifact is stale: {path}")
    return errors


def generate_records(output_dir: Path | None = None, *, write: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    artifacts = write_artifacts(output_dir) if write else {path.name: str(path) for path in render_artifacts(output_dir)}
    rows = build_rows(output_dir)
    return {
        "schema": "paper_i_hh_strong_weak_bounded_feature_ablation_generation_summary_v1",
        "batch_id": BATCH_ID,
        "output_dir": str(output_dir),
        "wrote_files": bool(write),
        "record_count": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "submit_ablation": str(SUBMIT_ABLATION),
        "paths": artifacts,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        errors = check_artifacts(args.output_dir)
        if errors:
            for error in errors:
                print(error)
            return 1
        print("Paper-I HH strong-weak bounded feature-ablation artifacts are current")
        return 0
    summary = generate_records(args.output_dir, write=bool(args.write))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
