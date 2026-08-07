#!/usr/bin/env python3
"""Generate source-locked Paper-I SNAKE rho-sweep records.

This is a one-variable sensitivity batch.  It reuses the visible Paper-I SNAKE
source settings for each case and varies only the row-level trust-region rho
(`phase2_rho`, the current global single-record trust-region scalar).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent

BATCH_ID = "routeA_paper_i_fixed_settings_rho_sweep_20260607_v1"
INPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID
OVERRIDE_DIR = INPUT_DIR / "trial_param_overrides"
SUMMARY_DIR = INPUT_DIR / "source_locked_summaries"
RECORDS_TSV = INPUT_DIR / "paper_i_fixed_settings_rho_sweep_records.tsv"
RECORD_IDS = INPUT_DIR / "paper_i_fixed_settings_rho_sweep_record_ids.txt"
MANIFEST_JSON = INPUT_DIR / "paper_i_fixed_settings_rho_sweep_manifest.json"
SUBMIT_FILE = SCRIPT_DIR / f"submit_{BATCH_ID}_full.sub"

RHO_GRID = (0.05, 0.1, 0.25, 0.5, 1.0)


@dataclass(frozen=True)
class CaseSpec:
    key: str
    table_label: str
    regime_label: str
    family: str
    benchmark_id: str
    base_records_tsv: str
    base_record_id: str
    source_result_json: str
    source_summary_json: str | None = None
    source_effective_manifest_json: str | None = None
    source_override_json: str | None = None
    source_trial_number: int | None = None
    n_ph_work: int | None = None
    n_ph_ref: int | None = None
    row_updates: Mapping[str, str] = field(default_factory=dict)


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        key="hubbard_weak",
        table_label="Table I",
        regime_label="Hubbard weak, U/t=0.5",
        family="hubbard",
        benchmark_id="hubbard_L2_three_model_weak",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_three_model_full_meta_20260526_v1/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_hubbard_l2_three_model_weak_full_meta_v1",
        source_result_json="tmp/paper_i_hubbard_weak_fullmeta_routeA_batching_optuna_20260604_v1/run/trial_0001/hubbard_L2_three_model_weak/json/result.json",
        source_summary_json="tmp/paper_i_hubbard_weak_fullmeta_routeA_batching_optuna_20260604_v1/run/summary.json",
        source_effective_manifest_json="tmp/paper_i_hubbard_weak_fullmeta_routeA_batching_optuna_20260604_v1/run/trial_0001/effective_trial_manifest.json",
        source_trial_number=1,
        row_updates={
            "suite_profile": "paper_i_main_tables_spsa_v1",
            "oracle_required_suite_profile": "paper_i_main_tables_spsa_v1",
            "phase3_adapt_parallel_gradient_workers": "1",
            "phase3_adapt_beam_parent_workers": "1",
            "seed": "7",
            "trial_timeout_sec": "21600",
            "compile_timeout_sec": "1200",
        },
    ),
    CaseSpec(
        key="hubbard_strong",
        table_label="Table I",
        regime_label="Hubbard strong, U/t=1.5",
        family="hubbard",
        benchmark_id="hubbard_L2_three_model_strong",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_three_model_selected_logical_20260525_v4/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_hubbard_l2_three_model_strong_selected_logical_v4",
        source_result_json=(
            "tmp/chtc_partial_paper_i_three_model_v3_v4_20260525/extracted/raw_outputs/"
            "routeA_paper_i_three_model_hubbard_l2_three_model_strong_selected_logical_v4/"
            "run/hubbard_L2_three_model_strong/trial_0000/hubbard_L2_three_model_strong/json/result.json"
        ),
        source_summary_json=(
            "tmp/chtc_partial_paper_i_three_model_v3_v4_20260525/extracted/raw_outputs/"
            "routeA_paper_i_three_model_hubbard_l2_three_model_strong_selected_logical_v4/summary.json"
        ),
        source_trial_number=0,
    ),
    CaseSpec(
        key="spin_boson_weak",
        table_label="Table II",
        regime_label="Spin-boson weak, g/omega0=0.05, n_ph_work=1 -> n_ph_ref=5",
        family="spin_boson",
        benchmark_id="spin_boson_L2_nph1_three_model_weak",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_three_model_selected_logical_20260525_v4/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_spin_boson_l2_nph1_three_model_weak_selected_logical_v4",
        source_result_json=(
            "tmp/chtc_partial_paper_i_three_model_v3_v4_20260525/extracted/raw_outputs/"
            "routeA_paper_i_three_model_spin_boson_l2_nph1_three_model_weak_selected_logical_v4/"
            "run/spin_boson_L2_nph1_three_model_weak/trial_0002/spin_boson_L2_nph1_three_model_weak/json/result.json"
        ),
        source_summary_json=(
            "tmp/chtc_partial_paper_i_three_model_v3_v4_20260525/extracted/raw_outputs/"
            "routeA_paper_i_three_model_spin_boson_l2_nph1_three_model_weak_selected_logical_v4/summary.json"
        ),
        source_trial_number=2,
        n_ph_work=1,
        n_ph_ref=5,
        row_updates={
            "boson_cutoff": "1",
            "exact_reference_boson_cutoff": "5",
            "n_ph_work": "1",
            "n_ph_ref": "5",
            "exact_reference_n_ph_max": "5",
            "primary_energy_metric": "higher_cutoff_reference_abs_delta_e",
            "same_cutoff_error_role": "diagnostic_only",
        },
    ),
    CaseSpec(
        key="spin_boson_strong",
        table_label="Table II",
        regime_label="Spin-boson strong, g/omega0=0.1, n_ph_work=2 -> n_ph_ref=6",
        family="spin_boson",
        benchmark_id="spin_boson_L2_nph2_three_model_strong",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_spin_boson_g0p05_ref5_g0p1_ref6_full_meta_20260526_v1/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_spin_boson_l2_nph2_three_model_strong_full_meta_v1",
        source_result_json=(
            "artifacts/agent_runs/spin_boson_snake_pauli_children_no_shots_local_optuna_20260527_v1/"
            "strong/run/spin_boson_L2_nph2_three_model_strong/trial_0737/"
            "spin_boson_L2_nph2_three_model_strong/json/result.json"
        ),
        source_effective_manifest_json=(
            "artifacts/agent_runs/spin_boson_snake_pauli_children_no_shots_local_optuna_20260527_v1/"
            "strong/run/spin_boson_L2_nph2_three_model_strong/trial_0737/effective_trial_manifest.json"
        ),
        source_trial_number=737,
        n_ph_work=2,
        n_ph_ref=6,
    ),
    CaseSpec(
        key="hh_weak_weak",
        table_label="Table III",
        regime_label="HH weak-weak, (U/t,lambda)=(0.25,0.25), n_ph_work=2 -> n_ph_ref=5",
        family="hh",
        benchmark_id="hh_L2_nph2_three_model_sym_weak_weak",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_hh_weak_weak_snake_flatnovelty_nocost_bounded_20260530_v3/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_weak_weak_full_meta_flatnovelty_nocost_bounded_v3",
        source_result_json="raw_outputs/chtc_fetches/hh_snake_weak_weak_current_full_20260531/hh_ww_trial0000_current.json",
        source_override_json="chtc/phase3_optuna/input/routeA_paper_i_hh_weak_weak_snake_flatnovelty_nocost_bounded_20260530_v3/flatnovelty_nocost_bounded_trial_param_overrides.json",
        source_trial_number=0,
        n_ph_work=2,
        n_ph_ref=5,
        row_updates={"trial_timeout_sec": "21600", "compile_timeout_sec": "1200"},
    ),
    CaseSpec(
        key="hh_strong_weak",
        table_label="Table III",
        regime_label="HH strong-weak, (U/t,lambda)=(1.25,0.25), n_ph_work=2 -> n_ph_ref=5",
        family="hh",
        benchmark_id="hh_L2_nph2_three_model_sym_strong_weak",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_strong_weak_full_meta_energygeom_nocost_routefix_v6",
        source_result_json=(
            "raw_outputs/chtc_fetches/hh_snake_strong_weak_trial0011_20260530_113725/raw_outputs/"
            "routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_strong_weak_full_meta_energygeom_nocost_routefix_v6/"
            "run/hh_L2_nph2_three_model_sym_strong_weak/trial_0011/"
            "hh_L2_nph2_three_model_sym_strong_weak/json/result.json"
        ),
        source_override_json="chtc/phase3_optuna/input/routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6/energy_geom_nocost_trial_param_overrides.json",
        source_trial_number=11,
        n_ph_work=2,
        n_ph_ref=5,
    ),
    CaseSpec(
        key="hh_weak_strong",
        table_label="Table III",
        regime_label="HH weak-strong, (U/t,lambda)=(0.25,1.25), n_ph_work=4 -> n_ph_ref=7",
        family="hh",
        benchmark_id="hh_L2_nph4_three_model_sym_weak_strong",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_hh_l2_nph4_three_model_sym_weak_strong_new_full_meta_flatnovelty_nocost_bounded_longtrial_v4",
        source_result_json="raw_outputs/chtc_fetches/hh_snake_all_time_best_ws_ss_20260531/hh_ws_trial0004_result.json",
        source_override_json="chtc/phase3_optuna/input/routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/flatnovelty_nocost_bounded_strong_holstein_trial_param_overrides.json",
        source_trial_number=4,
        n_ph_work=4,
        n_ph_ref=7,
    ),
    CaseSpec(
        key="hh_strong_strong",
        table_label="Table III",
        regime_label="HH strong-strong, (U/t,lambda)=(1.25,1.25), n_ph_work=4 -> n_ph_ref=7",
        family="hh",
        benchmark_id="hh_L2_nph4_three_model_sym_strong_strong",
        base_records_tsv="chtc/phase3_optuna/input/routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/paper_i_three_model_routeA_records.tsv",
        base_record_id="routeA_paper_i_three_model_hh_l2_nph4_three_model_sym_strong_strong_new_full_meta_flatnovelty_nocost_bounded_longtrial_v4",
        source_result_json="raw_outputs/chtc_fetches/hh_snake_all_time_best_ws_ss_20260531/hh_ss_trial0001_result.json",
        source_override_json="chtc/phase3_optuna/input/routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/flatnovelty_nocost_bounded_strong_holstein_trial_param_overrides.json",
        source_trial_number=1,
        n_ph_work=4,
        n_ph_ref=7,
    ),
)

TRIAL_PARAM_KEYS = {
    "adapt_allow_repeats",
    "adapt_beam_children_per_parent",
    "adapt_beam_live_branches",
    "adapt_beam_terminated_keep",
    "adapt_drop_floor",
    "adapt_drop_min_depth",
    "adapt_drop_patience",
    "adapt_final_full_refit",
    "adapt_full_refit_every",
    "adapt_insertion_mode",
    "adapt_max_depth",
    "adapt_maxiter",
    "adapt_reopt_policy",
    "adapt_window_size",
    "adapt_window_topk",
    "algebraic_phase1_lane_quota_pressure",
    "algebraic_phase2_lane_quota_pressure",
    "algebraic_phase2_lane_rel_threshold",
    "phase0_algebraic_lane_mode",
    "phase0_lane_quota_pressure",
    "phase0_pilot_alpha",
    "phase0_pilot_enabled",
    "phase0_pilot_max_records",
    "phase0_pilot_threshold",
    "phase1_max_count",
    "phase1_min_count",
    "phase1_maturity_cap_max_fraction",
    "phase1_maturity_cap_min_fraction",
    "phase1_maturity_shot_cap",
    "phase1_pool_fraction",
    "phase1_probe_max_positions",
    "phase1_prune_amplitude_witness_required",
    "phase1_prune_checkpoint_period",
    "phase1_prune_collapse_current_abs_max",
    "phase1_prune_collapse_min_abs_drop",
    "phase1_prune_collapse_min_observations",
    "phase1_prune_collapse_peak_abs_min",
    "phase1_prune_collapse_ratio",
    "phase1_prune_cooldown_steps",
    "phase1_prune_enabled",
    "phase1_prune_fraction",
    "phase1_prune_local_window_size",
    "phase1_prune_maturity_threshold",
    "phase1_prune_max_candidates",
    "phase1_prune_max_regression",
    "phase1_prune_min_candidates",
    "phase1_prune_mode",
    "phase1_prune_old_fraction",
    "phase1_prune_policy",
    "phase1_prune_protect_steps",
    "phase1_prune_retained_gain_ratio",
    "phase1_prune_small_theta_abs",
    "phase1_prune_small_theta_relative",
    "phase1_prune_snr_threshold",
    "phase1_prune_stagnation_threshold",
    "phase1_prune_stale_age",
    "phase1_prune_tolerance_chem",
    "phase1_prune_tolerance_mode",
    "phase1_prune_tolerance_rel_coeff",
    "phase1_prune_tolerance_screen_coeff",
    "phase1_prune_tolerance_shot_coeff",
    "phase1_qubit_slope",
    "phase1_score_mode",
    "phase1_score_z_alpha",
    "phase2_batch_additivity_tol",
    "phase2_batch_near_degenerate_ratio",
    "phase2_batch_rank_rel_tol",
    "phase2_batch_size_cap",
    "phase2_batch_target_size",
    "phase2_enable_batching",
    "phase2_frontier_ratio",
    "phase2_gamma_N",
    "phase2_gamma_N_schedule_end",
    "phase2_gamma_N_schedule_mode",
    "phase2_gamma_N_schedule_start",
    "phase2_hysteresis_steps",
    "phase2_leakage_cap",
    "phase2_live_nrem_low_threshold",
    "phase2_maturity_cap_max_fraction",
    "phase2_maturity_cap_min_fraction",
    "phase2_maturity_shot_cap",
    "phase2_max_count",
    "phase2_min_count",
    "phase2_motif_bonus_weight",
    "phase2_novelty_mode",
    "phase2_null_nrem_high_threshold",
    "phase2_pool_fraction",
    "phase2_qubit_slope",
    "phase2_score_z_alpha",
    "phase2_shortlist_fraction",
    "phase2_w_depth",
    "phase2_w_group",
    "phase2_w_lifetime",
    "phase2_w_optdim",
    "phase2_w_reuse",
    "phase2_w_shot",
    "phase3_backend_cost_mode",
    "phase3_batch_prefilter_mode",
    "phase3_batch_selection_mode",
    "phase3_frontier_ratio",
    "phase3_hysteresis_steps",
    "phase3_live_nrem_low_threshold",
    "phase3_maturity_cap_max_fraction",
    "phase3_maturity_cap_min_fraction",
    "phase3_maturity_shot_cap",
    "phase3_null_nrem_high_threshold",
    "phase3_runtime_split_max_subset_size",
    "phase3_runtime_split_mode",
    "phase3_runtime_split_selection_mode",
    "phase3_selector_geometry_mode",
    "phase3_selector_policy",
    "phase3_tie_beam_abs_tol",
    "phase3_tie_beam_max_branches",
    "phase3_tie_beam_max_late_coordinate",
    "phase3_tie_beam_min_depth_left",
    "phase3_tie_beam_score_ratio",
    "phase3_window_relaxation_mode",
    "phase_live_hysteresis_enabled",
    "phase_maturity_shot_max",
    "phase_maturity_shot_min",
    "spsa_A",
    "spsa_a",
    "spsa_alpha",
    "spsa_c",
    "spsa_gamma",
    "static_meta_feature_profile",
    "static_route_id",
}

OVERRIDE_KEYS = TRIAL_PARAM_KEYS | {
    "allow_archival_phase3_runtime_split",
    "compile_cx_weight",
    "compile_position_shift_weight",
    "compile_refit_active_weight",
    "compile_rotation_step_weight",
    "compile_sq_weight",
    "lambda_1q",
    "lambda_2q",
    "lambda_compile",
    "lambda_d",
    "lambda_leak",
    "lambda_measure",
    "lambda_shot",
    "lambda_theta",
    "measure_groups_weight",
    "measure_reuse_weight",
    "measure_shots_weight",
    "opt_dim_cost_scale",
    "phase1_prune_max_candidates",
    "phase1_prune_min_candidates",
    "suppress_explicit_hardware_lambdas",
}

ALIASES = {
    "phase1_compile_cx_proxy_weight": "compile_cx_weight",
    "phase1_compile_position_shift_weight": "compile_position_shift_weight",
    "phase1_compile_refit_active_weight": "compile_refit_active_weight",
    "phase1_compile_rotation_step_weight": "compile_rotation_step_weight",
    "phase1_compile_sq_proxy_weight": "compile_sq_weight",
    "phase1_lambda_1q": "lambda_1q",
    "phase1_lambda_2q": "lambda_2q",
    "phase1_lambda_compile": "lambda_compile",
    "phase1_lambda_d": "lambda_d",
    "phase1_lambda_leak": "lambda_leak",
    "phase1_lambda_measure": "lambda_measure",
    "phase1_lambda_shot": "lambda_shot",
    "phase1_lambda_theta": "lambda_theta",
    "phase1_measure_groups_weight": "measure_groups_weight",
    "phase1_measure_reuse_weight": "measure_reuse_weight",
    "phase1_measure_shots_weight": "measure_shots_weight",
    "phase1_opt_dim_cost_scale": "opt_dim_cost_scale",
    "phase2_compile_cx_proxy_weight": "compile_cx_weight",
    "phase2_compile_position_shift_weight": "compile_position_shift_weight",
    "phase2_compile_refit_active_weight": "compile_refit_active_weight",
    "phase2_compile_rotation_step_weight": "compile_rotation_step_weight",
    "phase2_compile_sq_proxy_weight": "compile_sq_weight",
    "phase2_measure_groups_weight": "measure_groups_weight",
    "phase2_measure_reuse_weight": "measure_reuse_weight",
    "phase2_measure_shots_weight": "measure_shots_weight",
    "phase2_opt_dim_cost_scale": "opt_dim_cost_scale",
}


def _path(value: str | Path | None) -> Path | None:
    if value in {None, ""}:
        return None
    path = Path(str(value))
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_rel(path: str | Path) -> str:
    resolved = _path(path)
    if resolved is None:
        raise ValueError("empty path")
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    resolved = _path(path)
    if resolved is None:
        raise ValueError("cannot hash empty path")
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: str | Path | None) -> dict[str, Any]:
    resolved = _path(path)
    if resolved is None:
        return {}
    return json.loads(resolved.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _load_row(path: str, record_id: str) -> tuple[dict[str, str], list[str]]:
    resolved = _path(path)
    if resolved is None:
        raise ValueError("empty records path")
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fieldnames = list(reader.fieldnames or ())
        rows = [dict(row) for row in reader]
    matches = [row for row in rows if row.get("record_id") == record_id]
    if len(matches) != 1:
        raise ValueError(f"expected one source row {record_id!r} in {path}; found {len(matches)}")
    return matches[0], fieldnames


def _rho_token(rho: float) -> str:
    text = f"{rho:g}".replace(".", "p")
    return text.replace("-", "m")


def _clean_params(params: Mapping[str, Any], *, allowed: set[str] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in dict(params).items():
        key = str(key).strip()
        if not key or key == "phase2_rho":
            continue
        if allowed is not None and key not in allowed:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if value is None:
            continue
        out[key] = value
    return out


def _summary_best_params(payload: Mapping[str, Any], benchmark_id: str) -> dict[str, Any]:
    if isinstance(payload.get("best_params"), Mapping):
        return dict(payload.get("best_params") or {})
    summaries = payload.get("summaries")
    if isinstance(summaries, Mapping):
        item = summaries.get(benchmark_id)
        if isinstance(item, Mapping) and isinstance(item.get("best_params"), Mapping):
            return dict(item.get("best_params") or {})
    return {}


def _summary_best_value(payload: Mapping[str, Any], benchmark_id: str) -> Any:
    if "best_value" in payload:
        return payload.get("best_value")
    summaries = payload.get("summaries")
    if isinstance(summaries, Mapping) and isinstance(summaries.get(benchmark_id), Mapping):
        return summaries[benchmark_id].get("best_value")
    return None


def _effective_sampled_params(payload: Mapping[str, Any]) -> dict[str, Any]:
    sampled = payload.get("sampled_params")
    return dict(sampled) if isinstance(sampled, Mapping) else {}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _add_if_present(out: dict[str, Any], source: Mapping[str, Any], key: str, target: str | None = None) -> None:
    if key in source and source.get(key) is not None:
        out[target or key] = source.get(key)


def _extract_result_params(result: Mapping[str, Any]) -> dict[str, Any]:
    settings = result.get("settings") if isinstance(result.get("settings"), Mapping) else {}
    adapt = result.get("adapt_vqe") if isinstance(result.get("adapt_vqe"), Mapping) else {}
    continuation = adapt.get("continuation") if isinstance(adapt.get("continuation"), Mapping) else {}
    phase2 = continuation.get("phase2") if isinstance(continuation.get("phase2"), Mapping) else {}

    out: dict[str, Any] = {}
    for source in (settings, adapt):
        for key in sorted((set(TRIAL_PARAM_KEYS) | set(OVERRIDE_KEYS)) - {"phase2_rho"}):
            _add_if_present(out, source, key)
        for key, target in ALIASES.items():
            _add_if_present(out, source, key, target)

    if "allow_repeats" in adapt:
        out["adapt_allow_repeats"] = adapt.get("allow_repeats")

    for source in (settings.get("adapt_spsa"), adapt.get("adapt_spsa")):
        if isinstance(source, Mapping):
            for src, dst in (("A", "spsa_A"), ("a", "spsa_a"), ("alpha", "spsa_alpha"), ("c", "spsa_c"), ("gamma", "spsa_gamma")):
                if source.get(src) is not None:
                    out[dst] = source.get(src)

    for src, dst in (
        ("batch_additivity_tol", "phase2_batch_additivity_tol"),
        ("batch_near_degenerate_ratio", "phase2_batch_near_degenerate_ratio"),
        ("batch_rank_rel_tol", "phase2_batch_rank_rel_tol"),
        ("batch_size_cap", "phase2_batch_size_cap"),
        ("batch_target_size", "phase2_batch_target_size"),
        ("batching_enabled", "phase2_enable_batching"),
        ("frontier_ratio", "phase2_frontier_ratio"),
        ("gamma_N", "phase2_gamma_N"),
        ("gamma_N_schedule_end", "phase2_gamma_N_schedule_end"),
        ("gamma_N_schedule_mode", "phase2_gamma_N_schedule_mode"),
        ("gamma_N_schedule_start", "phase2_gamma_N_schedule_start"),
        ("phase3_batch_prefilter_mode", "phase3_batch_prefilter_mode"),
        ("phase3_batch_selection_mode", "phase3_batch_selection_mode"),
        ("phase3_frontier_ratio", "phase3_frontier_ratio"),
        ("phase2_novelty_mode", "phase2_novelty_mode"),
        ("score_z_alpha", "phase2_score_z_alpha"),
    ):
        _add_if_present(out, phase2, src, dst)

    # When only emitted shortlist sizes are available, force the equivalent
    # budget cap so a partial replay does not resample the shortlist breadth.
    for source_key, min_key, max_key in (
        ("phase1_shortlist_size", "phase1_min_count", "phase1_max_count"),
        ("phase2_shortlist_size", "phase2_min_count", "phase2_max_count"),
    ):
        raw = settings.get(source_key)
        if raw not in {None, ""}:
            try:
                value = int(float(str(raw)))
            except Exception:
                continue
            out[min_key] = value
            out[max_key] = value

    out.setdefault("phase1_score_mode", "trust_region_v1")
    out.setdefault("static_route_id", "route_a")
    out.setdefault("static_meta_feature_profile", "paper_i_production_v1")
    return _clean_params(out)


def _load_source_override(path: str | None) -> dict[str, Any]:
    if not path:
        return {}
    payload = _read_json(path)
    overrides = payload.get("trial_param_overrides") if isinstance(payload.get("trial_param_overrides"), Mapping) else payload
    return dict(overrides) if isinstance(overrides, Mapping) else {}


def _source_params(case: CaseSpec) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    result = _read_json(case.source_result_json)
    source_summary = _read_json(case.source_summary_json)
    effective = _read_json(case.source_effective_manifest_json)
    summary_params = _summary_best_params(source_summary, case.benchmark_id)
    effective_params = _effective_sampled_params(effective)
    extracted = _extract_result_params(result)
    source_override = _load_source_override(case.source_override_json)

    best_params: dict[str, Any] = {}
    best_params.update(_clean_params(summary_params))
    best_params.update(_clean_params(effective_params))
    # Only enqueue known trial-parameter names from result reconstruction.
    best_params.update(_clean_params(extracted, allowed=TRIAL_PARAM_KEYS))

    overrides: dict[str, Any] = {}
    overrides.update(_clean_params(source_override, allowed=OVERRIDE_KEYS))
    overrides.update(_clean_params(extracted, allowed=OVERRIDE_KEYS))
    overrides["phase1_score_mode"] = "trust_region_v1"

    return best_params, overrides, result, source_summary


def _benchmark_payload(case: CaseSpec, row: Mapping[str, str]) -> dict[str, Any]:
    base_args: list[str] = []
    if case.n_ph_work is not None:
        base_args.extend(["--n-ph-max", str(case.n_ph_work)])
    return {
        "benchmark_id": case.benchmark_id,
        "family": case.family,
        "features": {"L": 2},
        "base_pipeline_args": base_args,
        "exact_reference_n_ph_max": case.n_ph_ref,
        "selected_logical_route": row.get("selected_logical_route") or "standard",
        "selected_logical_transfer_mode": row.get("selected_logical_transfer_mode") or "exact_match_v1",
    }


def _write_source_summary(
    case: CaseSpec,
    *,
    row: Mapping[str, str],
    best_params: Mapping[str, Any],
    source_summary: Mapping[str, Any],
) -> str:
    value = _summary_best_value(source_summary, case.benchmark_id)
    if value is None:
        value = 0.0
    summary_path = SUMMARY_DIR / f"{case.key}_source_locked_summary.json"
    payload = {
        "schema": "paper_i_fixed_settings_rho_sweep_source_summary_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Fixed-settings rho sweep warm start; one source parameter set, no settings search.",
        "batch_id": BATCH_ID,
        "static_route_id": row.get("static_route_id") or "route_a",
        "suite_profile": row.get("suite_profile") or "",
        "phase0_aware": True,
        "summaries": {
            case.benchmark_id: {
                "benchmark_id": case.benchmark_id,
                "family": case.family,
                "static_route_id": row.get("static_route_id") or "route_a",
                "suite_profile": row.get("suite_profile") or "",
                "phase0_aware": True,
                "best_trial_number": case.source_trial_number,
                "best_value": value,
                "best_params": dict(best_params),
                "benchmarks": [_benchmark_payload(case, row)],
            }
        },
    }
    _write_json(summary_path, payload)
    return _repo_rel(summary_path)


def _write_override(case: CaseSpec, *, rho: float, overrides: Mapping[str, Any]) -> str:
    path = OVERRIDE_DIR / f"{case.key}_rho{_rho_token(rho)}_trial_param_overrides.json"
    cleaned = _clean_params(overrides, allowed=OVERRIDE_KEYS)
    if "phase2_rho" in cleaned:
        raise ValueError(f"{case.key}: phase2_rho leaked into override payload")
    payload = {
        "schema": "phase3_trial_param_overrides_v1",
        "purpose": "Paper-I fixed-settings rho sweep; no Optuna/settings search.",
        "batch_id": BATCH_ID,
        "case_key": case.key,
        "rho": rho,
        "source_result_json": _repo_rel(case.source_result_json),
        "source_result_sha256": _sha256(case.source_result_json),
        "source_summary_json": _repo_rel(case.source_summary_json) if case.source_summary_json else None,
        "source_effective_manifest_json": _repo_rel(case.source_effective_manifest_json) if case.source_effective_manifest_json else None,
        "source_override_json": _repo_rel(case.source_override_json) if case.source_override_json else None,
        "omitted_sweep_fields": ["phase2_rho"],
        "trial_param_overrides": cleaned,
    }
    _write_json(path, payload)
    return _repo_rel(path)


def _record_for(case: CaseSpec, source_row: Mapping[str, str], *, rho: float, summary_path: str, override_path: str) -> dict[str, str]:
    row = {key: str(value or "") for key, value in source_row.items()}
    row.update({key: str(value) for key, value in case.row_updates.items()})
    row.update(
        {
            "mode": "oracle-grid",
            "record_id": f"{BATCH_ID}_{case.key}_rho{_rho_token(rho)}",
            "benchmark_ids": case.benchmark_id,
            "families": case.family,
            "oracle_summary_root": summary_path,
            "oracle_enqueue_limit": "1",
            "oracle_required_static_route_id": "route_a",
            "oracle_required_suite_profile": row.get("oracle_required_suite_profile") or row.get("suite_profile") or "",
            "oracle_require_phase0_aware": "true",
            "oracle_require_compatible_warm_starts": "true",
            "enqueue_default": "false",
            "enqueue_historical": "false",
            "n_trials": "1",
            "n_jobs": "1",
            "benchmarks_per_trial_jobs": "1",
            "phase2_rho": f"{rho:g}",
            "trial_param_overrides_json": override_path,
            "route_evidence_role": "rho_sensitivity_candidate_not_promoted",
            "paper_i_recovery_intent": "fixed_settings_rho_sweep_no_optuna_search",
            "algorithm_variant": f"paper_i_fixed_settings_rho_sweep_{case.key}",
            "fixed_inner_optimizer": "SPSA",
            "meta_feature_profile": row.get("meta_feature_profile") or "paper_i_production_v1",
            "static_route_id": row.get("static_route_id") or "route_a",
            "phase2_novelty_mode": row.get("phase2_novelty_mode") or "collective_span_v1",
            "robustness_gate": "off",
            "phase3_oracle_gradient_mode": row.get("phase3_oracle_gradient_mode") or "off",
            "phase3_oracle_inner_objective_mode": row.get("phase3_oracle_inner_objective_mode") or "exact",
            "phase3_oracle_value_noise_model": row.get("phase3_oracle_value_noise_model") or "off",
            "phase3_oracle_value_noise_std": row.get("phase3_oracle_value_noise_std") or "0.0",
        }
    )
    if case.n_ph_work is not None:
        row["n_ph_work"] = str(case.n_ph_work)
        row["boson_cutoff"] = str(case.n_ph_work)
    if case.n_ph_ref is not None:
        row["n_ph_ref"] = str(case.n_ph_ref)
        row["exact_reference_boson_cutoff"] = str(case.n_ph_ref)
        row["exact_reference_n_ph_max"] = str(case.n_ph_ref)
        row["primary_energy_metric"] = "higher_cutoff_reference_abs_delta_e"
        row["same_cutoff_error_role"] = "diagnostic_only"
    if case.family == "hubbard":
        row["exact_reference_boson_cutoff"] = "0"
    return row


def _write_submit() -> None:
    job_batch = BATCH_ID + "_full"
    content = f"""universe = vanilla
executable = chtc/phase3_optuna/run_task_apptainer.sh
arguments = $(record_id)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna
transfer_output_files = raw_outputs, logs
log = logs/{job_batch}.$(Cluster).$(Process).log
output = logs/{job_batch}.$(Cluster).$(Process).out
error = logs/{job_batch}.$(Cluster).$(Process).err
stream_output = False
stream_error = False
requirements = TARGET.HasSIF
request_cpus = 10
request_memory = 32GB
request_disk = 122880MB
+MaxRuntime = 172800
+JobBatchName = "holstein-{job_batch}"
environment = "PHASE3_RECORDS_PATH={_repo_rel(RECORDS_TSV)} PHASE3_TERMINATE_ON_STALE_PROGRESS=1 PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC=3600 PHASE3_PROGRESS_STALE_AFTER_SEC=3600 PHASE3_HEARTBEAT_INTERVAL_SEC=60 PHASE3_SHELL_HEARTBEAT_SEC=60"
queue record_id from {_repo_rel(RECORD_IDS)}
"""
    SUBMIT_FILE.write_text(content, encoding="utf-8")


def _validate(rows: list[Mapping[str, str]], source_details: Mapping[str, Any]) -> dict[str, Any]:
    ids = [row["record_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate record_id generated")
    if len(rows) != len(CASES) * len(RHO_GRID):
        raise ValueError(f"expected {len(CASES) * len(RHO_GRID)} rows; got {len(rows)}")
    by_case: dict[str, list[float]] = {}
    for row in rows:
        suffix = row["record_id"].removeprefix(f"{BATCH_ID}_")
        case_key = suffix.rsplit("_rho", 1)[0]
        by_case.setdefault(case_key, []).append(float(row["phase2_rho"]))
        if row.get("n_trials") != "1":
            raise ValueError(f"{row['record_id']}: n_trials is not 1")
        override_path = _path(row.get("trial_param_overrides_json"))
        if override_path is None or not override_path.exists():
            raise ValueError(f"{row['record_id']}: missing override json")
        override_payload = json.loads(override_path.read_text(encoding="utf-8"))
        override_fields = dict(override_payload.get("trial_param_overrides") or {})
        if "phase2_rho" in override_fields:
            raise ValueError(f"{row['record_id']}: phase2_rho leaked into override json")
        if row.get("enqueue_historical") != "false" or row.get("enqueue_default") != "false":
            raise ValueError(f"{row['record_id']}: enqueue controls are not locked off")
    for case in CASES:
        got = sorted(by_case.get(case.key, ()))
        if got != list(RHO_GRID):
            raise ValueError(f"{case.key}: rho grid mismatch {got}")
    return {
        "record_count": len(rows),
        "case_count": len(CASES),
        "rho_grid": list(RHO_GRID),
        "records_per_case": {case.key: len(by_case.get(case.key, ())) for case in CASES},
        "no_optuna_search": True,
        "all_rows_n_trials": 1,
        "enqueue_default": "false",
        "enqueue_historical": "false",
        "phase2_rho_only_in_row": True,
        "source_details": source_details,
    }


def main() -> int:
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERRIDE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    source_details: dict[str, Any] = {}

    for case in CASES:
        source_row, source_fields = _load_row(case.base_records_tsv, case.base_record_id)
        for field_name in source_fields:
            if field_name not in fieldnames:
                fieldnames.append(field_name)

        best_params, overrides, result, source_summary = _source_params(case)
        summary_stub_row = {**source_row, **case.row_updates}
        summary_path = _write_source_summary(case, row=summary_stub_row, best_params=best_params, source_summary=source_summary)
        source_details[case.key] = {
            "table_label": case.table_label,
            "regime_label": case.regime_label,
            "benchmark_id": case.benchmark_id,
            "base_records_tsv": case.base_records_tsv,
            "base_record_id": case.base_record_id,
            "source_result_json": case.source_result_json,
            "source_result_sha256": _sha256(case.source_result_json),
            "source_summary_json": case.source_summary_json,
            "source_effective_manifest_json": case.source_effective_manifest_json,
            "source_override_json": case.source_override_json,
            "source_trial_number": case.source_trial_number,
            "best_param_count": len(best_params),
            "override_count": len(overrides),
            "source_ansatz_depth": _nested(result, "adapt_vqe", "ansatz_depth") or result.get("ansatz_depth"),
            "source_abs_delta_e": _nested(result, "adapt_vqe", "abs_delta_e") or result.get("abs_delta_e"),
            "settings_reused": "visible_source_settings_locked; trust-region rho intentionally varied by generated row only",
            "settings_changed": ["phase2_rho"],
        }

        for rho in RHO_GRID:
            override_path = _write_override(case, rho=rho, overrides=overrides)
            rows.append(_record_for(case, source_row, rho=rho, summary_path=summary_path, override_path=override_path))

    for extra in sorted({key for row in rows for key in row} - set(fieldnames)):
        fieldnames.append(extra)

    with RECORDS_TSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    RECORD_IDS.write_text("".join(f"{row['record_id']}\n" for row in rows), encoding="utf-8")
    _write_submit()
    validation = _validate(rows, source_details)
    manifest = {
        "schema": "paper_i_fixed_settings_rho_sweep_manifest_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "purpose": "Paper-I one-variable trust-region rho sensitivity sweep.",
        "records_tsv": _repo_rel(RECORDS_TSV),
        "record_ids": _repo_rel(RECORD_IDS),
        "submit_file": _repo_rel(SUBMIT_FILE),
        "rho_grid": list(RHO_GRID),
        "case_keys": [case.key for case in CASES],
        "validation": validation,
    }
    _write_json(MANIFEST_JSON, manifest)
    print(json.dumps({"ok": True, "batch_id": BATCH_ID, "records": len(rows), "submit_file": _repo_rel(SUBMIT_FILE)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
