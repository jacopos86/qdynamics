#!/usr/bin/env python3
"""Generate Paper-I U/t=8 HH strong-strong SNAKE Pareto Cartesian Optuna records.

This diagnostic batch runs the four-arm Cartesian product requested on
2026-06-14 for the HH U/t=8 strong-strong row only:
flat/non-flat novelty crossed with cost/no-cost selector penalties.

The Optuna objective is the same-cutoff Pareto vector.  The Geo-ADAPT
strong-strong incumbent from paper_i_true_strong_replacement_20260613 is carried
as a dominance-filter anchor, not as a scalar residual target or fixed prefix.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import generate_paper_i_hh_tableiii_snake_novelty_optuna_records as tableiii  # noqa: E402
from chtc.phase3_optuna import generate_paper_i_hh_u8_snake_flatnovelty_nocost_records as nocost  # noqa: E402
from chtc.phase3_optuna import generate_routeA_phase0_nph1_oracle_records as nph1_oracle  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE  # noqa: E402
from pipelines.static_adapt.route_identity import (  # noqa: E402
    ROUTE_ID_UNSPECIFIED,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_ID = "paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna_20260614_v3_live"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID

RECORDS = "paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna_records.tsv"
RECORD_IDS = "paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna_record_ids.txt"
MANIFEST = "paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna_manifest.json"
SUBMIT_FILE = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"
OVERRIDE_DIRNAME = "trial_param_overrides"

N_TRIALS = 20
TRIAL_TIMEOUT_SEC = "21600"
COMPILE_TIMEOUT_SEC = "1200"
PARALLEL_GRADIENT_WORKERS = 4
BEAM_PARENT_WORKERS = 2
HH_HARD_MAX_DEPTH = 64
ADAPT_DROP_PATIENCE = 12
ADAPT_DROP_FLOOR = 1e-8

NOVELTY_ARMS = ("flat_novelty", "nonflat_novelty")
COST_ARMS = ("cost", "no_cost")
POLICY_PROFILE_BY_NOVELTY = {
    "flat_novelty": "snake_u8_flat_novelty_v1",
    "nonflat_novelty": "snake_u8_exponent_novelty_v1",
}

EXTRA_FIELDNAMES = (
    "trial_param_overrides_json",
    "run_class",
    "table_label",
    "strong_sector_target",
    "case_group",
    "hh_tableiii_regime",
    "n_ph_work",
    "n_ph_ref",
    "force_same_cutoff_objective",
    "primary_energy_metric",
    "same_cutoff_error_role",
    "route_base_pool_display_name",
    "phase3_adapt_beam_parent_workers",
    "paper_i_recovery_intent",
    "novelty_arm",
    "cost_arm",
    "incumbent_reference_method",
    "incumbent_reference_label",
    "incumbent_reference_error",
    "incumbent_reference_metric",
    "incumbent_reference_k",
    "incumbent_reference_n2q",
    "incumbent_reference_d2q",
    "incumbent_reference_dc",
    "incumbent_reference_s_work",
    "incumbent_reference_source_json",
    "incumbent_reference_source_sha256",
    "incumbent_reference_role",
    "source_plateau_iteration",
    "soft_plateau_depth",
    "hard_max_depth",
    "current_snake_source_json",
    "current_snake_source_sha256",
    "diagnostic_wrapper_approved",
    "promotion_status",
)
FIELDNAMES = tuple(dict.fromkeys((*nph1_oracle.FIELDNAMES, *EXTRA_FIELDNAMES)))


@dataclass(frozen=True)
class TargetSpec:
    key: str
    case_group: str
    family: str
    benchmark_id: str
    suite_profile: str
    n_ph_work: int
    incumbent_depth: int
    incumbent_method: str
    incumbent_label: str
    incumbent_error: float
    incumbent_n2q: float
    incumbent_d2q: float
    incumbent_dc: float
    incumbent_s_work: float
    incumbent_source_json: str
    incumbent_source_sha256: str
    hard_max_depth: int
    seed_base: int
    current_snake_source_json: str
    current_snake_source_sha256: str = ""


TARGETS: tuple[TargetSpec, ...] = (
    TargetSpec(
        key="hh_u8_strong_strong",
        case_group="hubbard_holstein",
        family="hh",
        benchmark_id="hh_L2_nph4_three_model_sym_u8_strong_strong",
        suite_profile=TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
        n_ph_work=4,
        incumbent_depth=15,
        incumbent_method="Geo-ADAPT",
        incumbent_label="Geo-ADAPT U/t=8 strong-strong",
        incumbent_error=0.00011327391134974274,
        incumbent_n2q=840.0,
        incumbent_d2q=711.0,
        incumbent_dc=3940.0,
        incumbent_s_work=180704.0,
        incumbent_source_json=(
            "raw_outputs/chtc_fetches/paper_i_hh_u8_strong_strong_completed_20260613/raw_outputs/"
            "paper_i_hh_u8_comparator_spsa_v1/records/"
            "paper_i_hh_u8_comp_spsa__full__static_geo_adapt_vqe__hh_u8_strong_strong/"
            "trial_0015/cases/hh_L2_nph4_three_model_sym_u8_strong_strong/result.json"
        ),
        incumbent_source_sha256="438f159bf008457d2f02582a42f1546ac6e6adafa98120ee7709cc93ee0bfd4f",
        hard_max_depth=HH_HARD_MAX_DEPTH,
        seed_base=98300,
        current_snake_source_json=(
            "raw_outputs/chtc_fetches/live_snapshots/"
            "paper_i_u8_strong_strong_live_20260613T133909/raw_outputs/"
            "paper_i_hh_snake_novelty_surface_optuna_20260611_v2_u8_strong_strong/"
            "run/hh_L2_nph4_three_model_sym_u8_strong_strong/trial_0009/"
            "hh_L2_nph4_three_model_sym_u8_strong_strong/json/result.json"
        ),
    ),
)


def _repo_relative(path: str | Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        return str(p)
    return str(p.relative_to(REPO_ROOT))


def _override_path(output_dir: Path, target: TargetSpec, novelty_arm: str, cost_arm: str) -> Path:
    return output_dir / OVERRIDE_DIRNAME / f"{target.key}_{novelty_arm}_{cost_arm}_trial_param_overrides.json"


def _novelty_overrides(novelty_arm: str) -> dict[str, Any]:
    if novelty_arm in {"flat_novelty", "nonflat_novelty"}:
        return {"phase3_novelty_ablation_mode": "off"}
    raise ValueError(f"unknown novelty arm: {novelty_arm}")


def _cost_overrides(cost_arm: str) -> dict[str, Any]:
    if cost_arm == "cost":
        return {}
    if cost_arm == "no_cost":
        return dict(nocost.NO_COST_STATIC_UPDATES)
    raise ValueError(f"unknown cost arm: {cost_arm}")


def _trial_param_overrides(target: TargetSpec, novelty_arm: str, cost_arm: str) -> dict[str, Any]:
    return {
        "adapt_max_depth": int(target.hard_max_depth),
        "adapt_drop_min_depth": int(target.incumbent_depth) + 1,
        "adapt_drop_patience": ADAPT_DROP_PATIENCE,
        "adapt_drop_floor": ADAPT_DROP_FLOOR,
        "pool_key": "full_meta",
        "family_repeat_penalty": 0.0,
        "novelty_bonus": 0.0,
        "phase2_motif_bonus_weight": 0.0,
        "compile_position_shift_weight": 0.0,
        "phase1_prune_amplitude_witness_required": False,
        "phase2_enable_batching": True,
        "phase3_batch_selection_mode": "reduced_plane",
        "phase3_batch_prefilter_mode": "off",
        "phase_live_hysteresis_enabled": False,
        **_novelty_overrides(novelty_arm),
        **_cost_overrides(cost_arm),
    }


def _incumbent_payload(target: TargetSpec) -> dict[str, Any]:
    return {
        "method": target.incumbent_method,
        "label": target.incumbent_label,
        "metric": "same_cutoff_abs_delta_e",
        "error": float(target.incumbent_error),
        "k": int(target.incumbent_depth),
        "N2q": float(target.incumbent_n2q),
        "D2q": float(target.incumbent_d2q),
        "Dc": float(target.incumbent_dc),
        "S_work": float(target.incumbent_s_work),
        "source_json": target.incumbent_source_json,
        "source_sha256": target.incumbent_source_sha256,
        "role": "pareto_anchor_metadata_only",
        "used_as_target": False,
        "used_as_prune_threshold": False,
        "used_as_fixed_prefix": False,
    }


def _override_payload(output_dir: Path, target: TargetSpec, novelty_arm: str, cost_arm: str) -> dict[str, Any]:
    overrides = _trial_param_overrides(target, novelty_arm, cost_arm)
    sampled = [
        "phase1/phase2 shortlist budgets",
        "insertion/refit/window controls",
        "batch caps/gates",
        "beam caps",
        "prune tolerances with amplitude-history witness disabled",
    ]
    if novelty_arm == "flat_novelty":
        sampled.append("phase2_gamma_N fixed-schedule weight")
    if novelty_arm == "nonflat_novelty":
        sampled.extend(["phase2_gamma_N_schedule_start", "phase2_gamma_N_schedule_end"])
    if cost_arm == "cost":
        sampled.append("compiled/measurement/hardware cost weights")
    return {
        "schema": "phase3_trial_param_overrides_v1",
        "purpose": "U/t=8 HH strong-strong SNAKE Pareto Cartesian Optuna search with motif/path priors disabled.",
        "batch_id": BATCH_ID,
        "target_key": target.key,
        "benchmark_id": target.benchmark_id,
        "novelty_arm": novelty_arm,
        "cost_arm": cost_arm,
        "trial_param_overrides_path": _repo_relative(_override_path(output_dir, target, novelty_arm, cost_arm)),
        "trial_param_overrides": overrides,
        "sampled_by_optuna": sampled,
        "fixed_spsa_contract": {
            "fixed_inner_optimizer": "SPSA",
            "spsa_schedule_sampled_by_optuna": False,
            "purpose": "compare selector policy effects without letting SPSA schedule explain differences",
        },
        "fixed_locks": {
            "pool_key": "full_meta",
            "phase2_motif_bonus_weight": 0.0,
            "novelty_bonus": 0.0,
            "compile_position_shift_weight": 0.0,
            "phase1_prune_amplitude_witness_required": False,
            "phase2_enable_batching": True,
            "phase3_batch_prefilter_mode": "off",
        },
        "no_target_contract": {
            "target_abs_delta_e": None,
            "required_target_profile": "none",
            "required_target_abs_delta_e": None,
            "robustness_gate": "off",
            "incumbent_errors_are_thresholds": False,
            "trial_prune_gate_enabled": False,
            "fixed_prefix_from_incumbent_enabled": False,
        },
        "incumbent_pareto_anchor": _incumbent_payload(target),
        "dominance_filter": {
            "same_cutoff_abs_delta_e_lt": float(target.incumbent_error),
            "count_2q_lt": float(target.incumbent_n2q),
            "depth_2q_lt": float(target.incumbent_d2q),
            "circuit_depth_lt": float(target.incumbent_dc),
            "source": "paper_i_true_strong_replacement_20260613 Geo-ADAPT strong-strong incumbent",
        },
    }


def _base_row(target: TargetSpec, novelty_arm: str, cost_arm: str, *, output_dir: Path, seed: int) -> dict[str, str]:
    record_id = f"{BATCH_ID}_{target.key}_{novelty_arm}_{cost_arm}"
    row = {key: "" for key in FIELDNAMES}
    n_ph = str(int(target.n_ph_work))
    row.update(
        {
            "record_id": record_id,
            "mode": "oracle-grid",
            "suite_profile": target.suite_profile,
            "benchmark_ids": target.benchmark_id,
            "calibration_profile": "off",
            "families": target.family,
            "sizes": "",
            "boson_cutoff": n_ph,
            "boson_cutoffs": "",
            "exact_reference_boson_cutoff": "0",
            "force_same_cutoff_objective": "true",
            "physics_grid_profile": "",
            "molecular_problem_json": "",
            "historical_ledger": "",
            "selected_logical_transfer_mode": "exact_match_v1",
            "selected_logical_source_json": "",
            "selected_logical_route": "standard",
            "oracle_summary_root": "",
            "oracle_enqueue_limit": "",
            "oracle_required_static_route_id": "",
            "oracle_required_suite_profile": "",
            "oracle_require_phase0_aware": "",
            "oracle_require_compatible_warm_starts": "",
            "objective_profile": "same_cutoff_pareto",
            "target_abs_delta_e": "",
            "objective_energy_weight": "",
            "objective_2q_weight": "",
            "objective_2q_depth_weight": "",
            "objective_depth_weight": "",
            "objective_parameter_weight": "",
            "objective_shot_weight": "",
            "objective_weight_preset": "uniform",
            "objective_family_weights": "",
            "objective_benchmark_weights": "",
            "n_trials": str(N_TRIALS),
            "n_jobs": "1",
            "benchmarks_per_trial_jobs": "1",
            "seed": str(int(seed)),
            "trial_timeout_sec": TRIAL_TIMEOUT_SEC,
            "compile_timeout_sec": COMPILE_TIMEOUT_SEC,
            "enqueue_default": "true",
            "enqueue_historical": "false",
            "fixed_inner_optimizer": "SPSA",
            "canonical_lane": "",
            "canonical_lane_stage": "",
            "policy_search_profile": POLICY_PROFILE_BY_NOVELTY[novelty_arm],
            "meta_feature_profile": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
            "required_target_profile": "none",
            "required_target_benchmark_ids": "",
            "required_target_abs_delta_e": "",
            "required_target_penalty": "1000.0",
            "robustness_gate": "off",
            "robustness_gate_lanes": "",
            "robustness_gate_target_abs_delta_e": "",
            "phase0_pilot_enabled": "",
            "phase0_pilot_alpha": "",
            "phase0_pilot_threshold": "",
            "phase0_pilot_max_records": "",
            "phase0_lane_quota_pressure": "",
            "phase0_algebraic_lane_mode": "",
            "algorithm_variant": "paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna",
            "static_route_id": ROUTE_ID_UNSPECIFIED,
            "route_base_pool_key": "full_meta",
            "route_base_pool_display_name": "full_meta",
            "canonical_snake_eligible_expected": "false",
            "route_evidence_role": "candidate_u8_hh_strong_strong_snake_pareto_cartesian_optuna",
            "phase2_novelty_mode": "collective_span_v1",
            "phase3_selector_policy": "algebraic_nested_v1",
            "phase3_selector_geometry_mode": "reduced",
            "phase3_novelty_ablation_mode": "off",
            "phase3_window_relaxation_mode": "reduced",
            "phase2_enable_batching": "true",
            "phase3_enable_batching": "true",
            "phase3_batch_selection_mode": "reduced_plane",
            "phase3_batch_prefilter_mode": "off",
            "phase1_prune_enabled": "true",
            "phase1_prune_policy": "recoverability_ladder_v1",
            "phase1_prune_mode": "both",
            "phase1_prune_amplitude_witness_required": "false",
            "continuation_mode": "phase3_v1",
            "algebraic_shortlisting_enabled": "true",
            "hardware_resolution_schema": "gradient_resolution_v1",
            "hardware_resolution_mode": "ideal",
            "phase3_adapt_parallel_gradient_workers": str(PARALLEL_GRADIENT_WORKERS),
            "phase3_adapt_beam_parent_workers": str(BEAM_PARENT_WORKERS),
            "phase3_oracle_gradient_mode": "off",
            "phase3_oracle_backend_name": "",
            "phase3_oracle_use_fake_backend": "",
            "phase3_oracle_inner_objective_mode": "exact",
            "phase3_oracle_value_noise_model": "off",
            "phase3_oracle_value_noise_std": "0.0",
            "phase3_adapt_allow_repeats": "true",
            "trial_param_overrides_json": _repo_relative(_override_path(output_dir, target, novelty_arm, cost_arm)),
            "run_class": "candidate_settings_search",
            "table_label": "diagnostic:u8_hh_strong_strong_snake_pareto_cartesian_optuna",
            "strong_sector_target": "U/t=8",
            "case_group": target.case_group,
            "hh_tableiii_regime": target.key,
            "n_ph_work": n_ph,
            "n_ph_ref": n_ph,
            "primary_energy_metric": "same_cutoff_abs_delta_e",
            "same_cutoff_error_role": "primary_metric_for_no_target_pareto_calibration",
            "paper_i_recovery_intent": "recoverable_per_trial_per_depth_current_json_and_logs",
            "novelty_arm": novelty_arm,
            "cost_arm": cost_arm,
            "incumbent_reference_method": target.incumbent_method,
            "incumbent_reference_label": target.incumbent_label,
            "incumbent_reference_error": f"{float(target.incumbent_error):.17g}",
            "incumbent_reference_metric": "same_cutoff_abs_delta_e",
            "incumbent_reference_k": str(int(target.incumbent_depth)),
            "incumbent_reference_n2q": f"{float(target.incumbent_n2q):.17g}",
            "incumbent_reference_d2q": f"{float(target.incumbent_d2q):.17g}",
            "incumbent_reference_dc": f"{float(target.incumbent_dc):.17g}",
            "incumbent_reference_s_work": f"{float(target.incumbent_s_work):.17g}",
            "incumbent_reference_source_json": target.incumbent_source_json,
            "incumbent_reference_source_sha256": target.incumbent_source_sha256,
            "incumbent_reference_role": "pareto_anchor_metadata_only_not_target",
            "source_plateau_iteration": "",
            "soft_plateau_depth": str(int(target.incumbent_depth) + 1),
            "hard_max_depth": str(int(target.hard_max_depth)),
            "current_snake_source_json": target.current_snake_source_json,
            "current_snake_source_sha256": target.current_snake_source_sha256,
            "diagnostic_wrapper_approved": "true",
            "promotion_status": "candidate_not_table_update",
        }
    )
    return {key: str(row.get(key, "")) for key in FIELDNAMES}


def build_rows(output_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    rows: list[dict[str, str]] = []
    details: list[dict[str, Any]] = []
    for target in TARGETS:
        for novelty_index, novelty_arm in enumerate(NOVELTY_ARMS):
            for cost_index, cost_arm in enumerate(COST_ARMS):
                seed = int(target.seed_base + 10 * novelty_index + cost_index)
                rows.append(_base_row(target, novelty_arm, cost_arm, output_dir=output_dir, seed=seed))
                details.append(
                    {
                        "target_key": target.key,
                        "benchmark_id": target.benchmark_id,
                        "family": target.family,
                        "suite_profile": target.suite_profile,
                        "n_ph_work": target.n_ph_work,
                        "n_ph_ref": target.n_ph_work,
                        "novelty_arm": novelty_arm,
                        "cost_arm": cost_arm,
                        "policy_search_profile": POLICY_PROFILE_BY_NOVELTY[novelty_arm],
                        "incumbent_pareto_anchor": _incumbent_payload(target),
                        "soft_plateau_depth": target.incumbent_depth + 1,
                        "hard_max_depth": target.hard_max_depth,
                        "seed": seed,
                    }
                )
    return rows, details


def _tsv_text(rows: Sequence[Mapping[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(FIELDNAMES), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    return buf.getvalue()


def _ids_text(rows: Sequence[Mapping[str, str]]) -> str:
    return "\n".join(str(row["record_id"]) for row in rows) + "\n"


def _manifest_payload(rows: Sequence[Mapping[str, str]], details: Sequence[Mapping[str, Any]], output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna_manifest_v1",
        "batch_id": BATCH_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "chtc/phase3_optuna/generate_paper_i_u8_hh_strong_strong_snake_pareto_cartesian_optuna_records.py",
        "run_class": "candidate_settings_search",
        "records_tsv": _repo_relative(output_dir / RECORDS),
        "record_id_file": _repo_relative(output_dir / RECORD_IDS),
        "submit_file": _repo_relative(SUBMIT_FILE),
        "record_count": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "targets": [target.key for target in TARGETS],
        "novelty_arms": list(NOVELTY_ARMS),
        "cost_arms": list(COST_ARMS),
        "n_trials_per_job": N_TRIALS,
        "primary_energy_metric": "same_cutoff_abs_delta_e",
        "same_cutoff_objective_forced": True,
        "multi_objective_mode": "same_cutoff_pareto",
        "objective_vector_names": [
            "same_cutoff_abs_delta_e",
            "count_2q",
            "depth_2q",
            "circuit_depth",
            "parameter_count",
            "shot_cost_proxy",
        ],
        "target_details": list(details),
        "no_target_contract": {
            "target_abs_delta_e": None,
            "required_target_profile": "none",
            "required_target_abs_delta_e": None,
            "robustness_gate": "off",
            "incumbent_errors_are_thresholds": False,
            "trial_prune_gate_enabled": False,
            "fixed_prefix_from_incumbent_enabled": False,
            "objective": "multi-objective same-cutoff Pareto vector; filter terminal Pareto points against incumbent residual/resource anchors after the run",
            "optuna_runner_currently_scalarized": False,
            "true_pareto_dominance_requires_runner_multi_objective_patch": False,
        },
        "dominance_filters": [
            _incumbent_payload(target)
            | {
                "target_key": target.key,
                "same_cutoff_abs_delta_e_lt": float(target.incumbent_error),
                "count_2q_lt": float(target.incumbent_n2q),
                "depth_2q_lt": float(target.incumbent_d2q),
                "circuit_depth_lt": float(target.incumbent_dc),
            }
            for target in TARGETS
        ],
        "incumbent_pareto_anchors": [_incumbent_payload(target) | {"target_key": target.key} for target in TARGETS],
        "fixed_spsa_contract": {
            "fixed_inner_optimizer": "SPSA",
            "spsa_schedule_sampled_by_optuna": False,
            "purpose": "compare selector policy effects without letting SPSA schedule explain differences",
        },
        "fixed_locks": {
            "pool_key": "full_meta",
            "operator_pool_contract": "problem-local full_meta; no selected-logical reduced/winning pool",
            "phase2_motif_bonus_weight": 0.0,
            "novelty_bonus": 0.0,
            "compile_position_shift_weight": 0.0,
            "phase1_prune_amplitude_witness_required": False,
            "phase2_enable_batching": True,
            "phase3_batch_prefilter_mode": "off",
        },
        "progress_contract": {
            "adapt_current_json_every_depth": 1,
            "adapt_current_json_keep_history_tail": 100,
            "transfer_policy": "ON_EXIT_OR_EVICT_plus_condor_tail_authorized_live_status",
            "transfer_output_files": [
                "raw_outputs",
                "logs",
                "raw_outputs/$(record_id)/progress/live_status.json",
                "raw_outputs/$(record_id)/progress/live_status.jsonl",
            ],
            "live_retrieval": {
                "tool": "condor_tail",
                "json_path_template": "raw_outputs/<record_id>/progress/live_status.json",
                "jsonl_path_template": "raw_outputs/<record_id>/progress/live_status.jsonl",
                "example": "condor_tail -maxbytes 200000 <cluster.proc> raw_outputs/<record_id>/progress/live_status.json",
            },
        },
        "promotion_status": "candidate_not_table_update",
    }


def _submit_text(*, records_path: Path, record_ids_path: Path, job_batch_name: str) -> str:
    live_status_json = "raw_outputs/$(record_id)/progress/live_status.json"
    live_status_jsonl = "raw_outputs/$(record_id)/progress/live_status.jsonl"
    env = (
        f"PHASE3_RECORDS_PATH={_repo_relative(records_path)} "
        "PHASE3_TERMINATE_ON_STALE_PROGRESS=1 "
        "PHASE3_REQUIRE_FIRST_PROGRESS_WITHIN_SEC=3600 "
        "PHASE3_PROGRESS_STALE_AFTER_SEC=3600 "
        "PHASE3_HEARTBEAT_INTERVAL_SEC=30 "
        "PHASE3_SHELL_HEARTBEAT_SEC=30"
    )
    return f"""universe = vanilla
executable = chtc/phase3_optuna/run_task_apptainer.sh
arguments = $(record_id)
should_transfer_files = YES
when_to_transfer_output = ON_EXIT_OR_EVICT
transfer_executable = True
preserve_relative_paths = True
transfer_input_files = pipelines, src, docs, test_support, chtc/phase3_optuna
transfer_output_files = raw_outputs, logs, {live_status_json}, {live_status_jsonl}
log = logs/{job_batch_name}.$(Cluster).$(Process).log
output = logs/{job_batch_name}.$(Cluster).$(Process).out
error = logs/{job_batch_name}.$(Cluster).$(Process).err
stream_output = False
stream_error = False
requirements = TARGET.HasSIF
request_cpus = 4
request_memory = 32GB
request_disk = 122880MB
+MaxRuntime = 172800
+JobBatchName = "holstein-{job_batch_name}"
environment = "{env}"
queue record_id from {_repo_relative(record_ids_path)}
"""


def render_artifacts(output_dir: Path | None = None) -> dict[Path, str]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    rows, details = build_rows(output_dir)
    artifacts: dict[Path, str] = {
        output_dir / RECORDS: _tsv_text(rows),
        output_dir / RECORD_IDS: _ids_text(rows),
        output_dir / MANIFEST: json.dumps(
            _manifest_payload(rows, details, output_dir),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        SUBMIT_FILE: _submit_text(
            records_path=output_dir / RECORDS,
            record_ids_path=output_dir / RECORD_IDS,
            job_batch_name=BATCH_ID,
        ),
    }
    for target in TARGETS:
        for novelty_arm in NOVELTY_ARMS:
            for cost_arm in COST_ARMS:
                artifacts[_override_path(output_dir, target, novelty_arm, cost_arm)] = (
                    json.dumps(_override_payload(output_dir, target, novelty_arm, cost_arm), indent=2, sort_keys=True)
                    + "\n"
                )
    return artifacts


def write_artifacts(output_dir: Path | None = None) -> dict[str, str]:
    artifacts = render_artifacts(output_dir)
    for path, text in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return {path.name: str(path) for path in artifacts}


def _normalized_for_check(path: Path, text: str) -> str:
    if path.name != MANIFEST:
        return text
    payload = json.loads(text)
    if isinstance(payload, dict):
        payload.pop("generated_utc", None)
    return json.dumps(payload, sort_keys=True)


def check_artifacts(output_dir: Path | None = None) -> list[str]:
    errors: list[str] = []
    for path, expected in render_artifacts(output_dir).items():
        try:
            actual = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            errors.append(f"missing generated artifact: {path}")
            continue
        if _normalized_for_check(path, actual) != _normalized_for_check(path, expected):
            errors.append(f"generated artifact is stale: {path}")
    return errors


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
                print(error, file=sys.stderr)
            return 1
        rows, _details = build_rows(Path(args.output_dir or DEFAULT_OUTPUT_DIR))
        print(json.dumps({"status": "ok", "record_count": len(rows)}, indent=2))
        return 0
    if args.write:
        paths = write_artifacts(args.output_dir)
        rows, _details = build_rows(Path(args.output_dir or DEFAULT_OUTPUT_DIR))
        print(
            json.dumps(
                {
                    "status": "written",
                    "batch_id": BATCH_ID,
                    "record_count": len(rows),
                    "record_ids": [row["record_id"] for row in rows],
                    "paths": paths,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    rows, details = build_rows(Path(args.output_dir or DEFAULT_OUTPUT_DIR))
    print(json.dumps(_manifest_payload(rows, details, Path(args.output_dir or DEFAULT_OUTPUT_DIR)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
