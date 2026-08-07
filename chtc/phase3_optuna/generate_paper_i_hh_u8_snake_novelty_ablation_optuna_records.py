#!/usr/bin/env python3
"""Generate Paper-I HH U/t=8 SNAKE novelty/no-novelty Optuna records."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import generate_paper_i_hh_tableiii_snake_novelty_optuna_records as base  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import (  # noqa: E402
    TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
)
from pipelines.static_adapt.route_identity import (  # noqa: E402
    ROUTE_ID_A,
    ROUTE_ID_UNSPECIFIED,
    STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BATCH_ID = "paper_i_hh_u8_snake_novelty_ablation_optuna_20260611_v1"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "input" / BATCH_ID

SOURCE_MAP = Path("MATH/paper_facing/paper_I_static_scaffold/hh_tableiii_convergence_sources.json")
RECORDS = "paper_i_hh_u8_snake_novelty_ablation_optuna_records.tsv"
RECORD_IDS = "paper_i_hh_u8_snake_novelty_ablation_optuna_record_ids.txt"
MANIFEST = "paper_i_hh_u8_snake_novelty_ablation_optuna_manifest.json"
SUBMIT_FILE = SCRIPT_DIR / f"submit_{BATCH_ID}.sub"
OVERRIDE_DIRNAME = "trial_param_overrides"

N_TRIALS = 20
HARD_MAX_DEPTH = 64
ADAPT_DROP_PATIENCE = 12
ADAPT_DROP_FLOOR = 1e-8
TRIAL_TIMEOUT_SEC = "21600"
COMPILE_TIMEOUT_SEC = "1200"
PARALLEL_GRADIENT_WORKERS = 4
BEAM_PARENT_WORKERS = 2

EXTRA_FIELDNAMES = (
    "u_over_t",
    "lambda_ep",
    "g_ep",
    "physics_profile",
    "source_baseline_profile",
)
FIELDNAMES = tuple(dict.fromkeys((*base.FIELDNAMES, *EXTRA_FIELDNAMES)))


@dataclass(frozen=True)
class RegimeSpec:
    key: str
    benchmark_id: str
    base_records_tsv: str
    base_record_id: str
    gamma_source_json: str
    n_ph_work: int
    n_ph_ref: int
    seed: int
    lambda_ep: str
    g_ep: str


REGIMES: tuple[RegimeSpec, ...] = (
    RegimeSpec(
        key="strong_weak",
        benchmark_id="hh_L2_nph2_three_model_sym_u8_strong_weak",
        base_records_tsv=(
            "chtc/phase3_optuna/input/"
            "routeA_paper_i_hh_symmetric_snake_energy_geom_nocost_routefix_20260530_v6/"
            "paper_i_three_model_routeA_records.tsv"
        ),
        base_record_id="routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_strong_weak_full_meta_energygeom_nocost_routefix_v6",
        gamma_source_json=(
            "raw_outputs/chtc_fetches/hh_snake_strong_weak_trial0011_20260530_113725/raw_outputs/"
            "routeA_paper_i_three_model_hh_l2_nph2_three_model_sym_strong_weak_full_meta_energygeom_nocost_routefix_v6/"
            "run/hh_L2_nph2_three_model_sym_strong_weak/trial_0011/"
            "hh_L2_nph2_three_model_sym_strong_weak/json/result.json"
        ),
        n_ph_work=2,
        n_ph_ref=5,
        seed=96118,
        lambda_ep="0.25",
        g_ep="0.3535533905932738",
    ),
    RegimeSpec(
        key="strong_strong",
        benchmark_id="hh_L2_nph4_three_model_sym_u8_strong_strong",
        base_records_tsv=(
            "chtc/phase3_optuna/input/"
            "routeA_paper_i_hh_strong_holstein_snake_flatnovelty_nocost_bounded_longtrial_20260530_v4/"
            "paper_i_three_model_routeA_records.tsv"
        ),
        base_record_id="routeA_paper_i_three_model_hh_l2_nph4_three_model_sym_strong_strong_new_full_meta_flatnovelty_nocost_bounded_longtrial_v4",
        gamma_source_json="raw_outputs/chtc_fetches/hh_snake_all_time_best_ws_ss_20260531/hh_ss_trial0001_result.json",
        n_ph_work=4,
        n_ph_ref=7,
        seed=96120,
        lambda_ep="1.25",
        g_ep="0.7905694150420949",
    ),
)


def _override_path(output_dir: Path, regime: RegimeSpec, arm: str) -> Path:
    return output_dir / OVERRIDE_DIRNAME / f"{regime.key}_{arm}_trial_param_overrides.json"


def _arm_settings(regime: RegimeSpec, arm: str) -> dict[str, Any]:
    if arm == "novelty":
        return {
            **base._extract_source_gamma_settings(regime.gamma_source_json),
            "phase3_novelty_ablation_mode": "off",
        }
    if arm == "no_novelty":
        return {
            "phase2_gamma_N": 0.0,
            "phase2_gamma_N_schedule_mode": "fixed",
            "phase2_gamma_N_schedule_start": None,
            "phase2_gamma_N_schedule_end": None,
            "phase2_motif_bonus_weight": 0.0,
            "novelty_bonus": 0.0,
            "phase3_novelty_ablation_mode": "all",
        }
    raise ValueError(f"unknown arm: {arm}")


def _override_payload(
    *,
    output_dir: Path,
    regime: RegimeSpec,
    arm: str,
    source_entry: Mapping[str, Any],
    source_plateau_iteration: int,
) -> dict[str, Any]:
    soft_depth = int(source_plateau_iteration) + 3
    trial_param_overrides = {
        "adapt_max_depth": HARD_MAX_DEPTH,
        "adapt_drop_min_depth": soft_depth,
        "adapt_drop_patience": ADAPT_DROP_PATIENCE,
        "adapt_drop_floor": ADAPT_DROP_FLOOR,
        **{key: base._clean_json_value(value) for key, value in _arm_settings(regime, arm).items()},
    }
    return {
        "schema": "phase3_trial_param_overrides_v1",
        "purpose": "Diagnostic HH U/t=8 SNAKE Optuna novelty/no-novelty comparison.",
        "batch_id": BATCH_ID,
        "run_class": "diagnostic_hh_u8_snake_novelty_ablation_optuna",
        "regime": regime.key,
        "benchmark_id": regime.benchmark_id,
        "novelty_arm": arm,
        "u_over_t": "8.0",
        "lambda_ep": regime.lambda_ep,
        "g_ep": regime.g_ep,
        "suite_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
        "source_baseline_profile": "paper_i_three_model_hh_symmetric_20260527_v1",
        "source_map": base._repo_relative(SOURCE_MAP),
        "source_json": source_entry.get("source_json"),
        "source_sha256": source_entry.get("source_sha256"),
        "gamma_source_json": base._repo_relative(regime.gamma_source_json),
        "gamma_source_sha256": base._sha256(regime.gamma_source_json),
        "source_plateau_iteration": int(source_plateau_iteration),
        "soft_plateau_depth": soft_depth,
        "hard_max_depth": HARD_MAX_DEPTH,
        "trial_param_overrides_path": base._repo_relative(_override_path(output_dir, regime, arm)),
        "trial_param_overrides": trial_param_overrides,
        "wrapper_used": True,
        "wrapper_kind": "phase3_policy_optuna",
        "diagnostic_wrapper_approved": True,
    }


def _load_row(path: str, record_id: str) -> dict[str, str]:
    row, _fields = base._load_row(path, record_id)
    return row


def _row_for(
    *,
    output_dir: Path,
    source_row: Mapping[str, str],
    regime: RegimeSpec,
    arm: str,
    source_entry: Mapping[str, Any],
    source_plateau_iteration: int,
) -> dict[str, str]:
    row = {key: str(value or "") for key, value in source_row.items()}
    record_id = f"{BATCH_ID}_{regime.key}_{arm}"
    soft_depth = int(source_plateau_iteration) + 3
    route_id = ROUTE_ID_A if arm == "novelty" else ROUTE_ID_UNSPECIFIED
    novelty_ablation = "off" if arm == "novelty" else "all"
    row.update(
        {
            "record_id": record_id,
            "mode": "oracle-grid",
            "suite_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
            "benchmark_ids": regime.benchmark_id,
            "families": "hh",
            "boson_cutoff": str(regime.n_ph_work),
            "boson_cutoffs": "",
            "exact_reference_boson_cutoff": str(regime.n_ph_ref),
            "physics_grid_profile": "",
            "historical_ledger": "",
            "selected_logical_transfer_mode": "exact_match_v1",
            "selected_logical_source_json": "",
            "selected_logical_route": "standard",
            "oracle_summary_root": "",
            "oracle_enqueue_limit": "",
            "objective_profile": "balanced",
            "target_abs_delta_e": "",
            "objective_weight_preset": "uniform",
            "n_trials": str(N_TRIALS),
            "n_jobs": "1",
            "benchmarks_per_trial_jobs": "1",
            "seed": str(regime.seed),
            "trial_timeout_sec": TRIAL_TIMEOUT_SEC,
            "compile_timeout_sec": COMPILE_TIMEOUT_SEC,
            "enqueue_default": "true",
            "enqueue_historical": "false",
            "fixed_inner_optimizer": "SPSA",
            "policy_search_profile": "default",
            "meta_feature_profile": STATIC_META_FEATURE_PROFILE_PAPER_I_PRODUCTION_V1,
            "required_target_profile": "paper_i_phys_v1",
            "required_target_benchmark_ids": "",
            "required_target_abs_delta_e": "",
            "required_target_penalty": "1000.0",
            "robustness_gate": "off",
            "algorithm_variant": "paper_i_hh_u8_snake_novelty_ablation_optuna",
            "static_route_id": route_id,
            "route_base_pool_key": "full_meta",
            "route_base_pool_display_name": "full_meta",
            "canonical_snake_eligible_expected": "true",
            "route_evidence_role": "diagnostic_u8_novelty_ablation_optuna_not_table_update",
            "phase2_novelty_mode": "collective_span_v1",
            "phase3_selector_policy": "algebraic_nested_v1",
            "phase3_selector_geometry_mode": "reduced",
            "phase3_novelty_ablation_mode": novelty_ablation,
            "phase3_window_relaxation_mode": "reduced",
            "phase2_enable_batching": row.get("phase2_enable_batching") or "true",
            "phase3_enable_batching": row.get("phase3_enable_batching") or "true",
            "phase3_batch_selection_mode": "reduced_plane",
            "phase3_batch_prefilter_mode": "off",
            "phase3_nested_window_application": "composed_batch_window_v1",
            "phase1_prune_enabled": "true",
            "phase1_prune_policy": "recoverability_ladder_v1",
            "phase1_prune_mode": "both",
            "phase1_prune_amplitude_witness_required": "true",
            "continuation_mode": "phase3_v1",
            "algebraic_shortlisting_enabled": "true",
            "hardware_resolution_schema": "gradient_resolution_v1",
            "hardware_resolution_mode": "ideal",
            "phase3_adapt_parallel_gradient_workers": str(PARALLEL_GRADIENT_WORKERS),
            "phase3_adapt_beam_parent_workers": str(BEAM_PARENT_WORKERS),
            "phase3_oracle_gradient_mode": "off",
            "phase3_oracle_inner_objective_mode": "exact",
            "phase3_oracle_value_noise_model": "off",
            "phase3_oracle_value_noise_std": "0.0",
            "phase3_adapt_allow_repeats": row.get("phase3_adapt_allow_repeats") or "true",
            "trial_param_overrides_json": base._repo_relative(_override_path(output_dir, regime, arm)),
            "run_class": "diagnostic",
            "table_label": "diagnostic:hh_u8_strong_hubbard_novelty_ablation",
            "hh_tableiii_regime": regime.key,
            "n_ph_work": str(regime.n_ph_work),
            "n_ph_ref": str(regime.n_ph_ref),
            "exact_reference_n_ph_max": str(regime.n_ph_ref),
            "primary_energy_metric": "same_cutoff_abs_delta_e_with_higher_cutoff_diagnostic",
            "same_cutoff_error_role": "diagnostic_u8_metric",
            "paper_i_recovery_intent": "recoverable_per_trial_per_depth_current_json_and_logs",
            "novelty_arm": arm,
            "source_plateau_iteration": str(int(source_plateau_iteration)),
            "soft_plateau_depth": str(soft_depth),
            "hard_max_depth": str(HARD_MAX_DEPTH),
            "source_map_json": base._repo_relative(SOURCE_MAP),
            "source_json": str(source_entry.get("source_json") or ""),
            "source_sha256": str(source_entry.get("source_sha256") or ""),
            "gamma_source_json": base._repo_relative(regime.gamma_source_json),
            "gamma_source_sha256": base._sha256(regime.gamma_source_json),
            "diagnostic_wrapper_approved": "true",
            "promotion_status": "diagnostic_not_table_update",
            "u_over_t": "8.0",
            "lambda_ep": regime.lambda_ep,
            "g_ep": regime.g_ep,
            "physics_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
            "source_baseline_profile": "paper_i_three_model_hh_symmetric_20260527_v1",
        }
    )
    return {key: str(row.get(key, "")) for key in FIELDNAMES}


def build_rows(output_dir: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    source_map = base._load_source_map()
    rows: list[dict[str, str]] = []
    source_details: dict[str, Any] = {}
    for regime in REGIMES:
        source_row = _load_row(regime.base_records_tsv, regime.base_record_id)
        regime_payload = base._nested(source_map, "regimes", regime.key, "methods", "SNAKE")
        plateau_payload = base._nested(source_map, "plateau_markers", regime.key, "SNAKE")
        if not isinstance(regime_payload, Mapping):
            raise ValueError(f"missing SNAKE source-map entry for {regime.key}")
        if not isinstance(plateau_payload, Mapping) or plateau_payload.get("iteration") is None:
            raise ValueError(f"missing SNAKE plateau marker for {regime.key}")
        source_plateau_iteration = int(plateau_payload["iteration"])
        for arm in ("novelty", "no_novelty"):
            rows.append(
                _row_for(
                    output_dir=output_dir,
                    source_row=source_row,
                    regime=regime,
                    arm=arm,
                    source_entry=regime_payload,
                    source_plateau_iteration=source_plateau_iteration,
                )
            )
        source_details[regime.key] = {
            "benchmark_id": regime.benchmark_id,
            "u_over_t": "8.0",
            "lambda_ep": regime.lambda_ep,
            "g_ep": regime.g_ep,
            "n_ph_work": regime.n_ph_work,
            "n_ph_ref": regime.n_ph_ref,
            "base_records_tsv": regime.base_records_tsv,
            "base_record_id": regime.base_record_id,
            "source_baseline_profile": "paper_i_three_model_hh_symmetric_20260527_v1",
            "source_json": regime_payload.get("source_json"),
            "source_sha256": regime_payload.get("source_sha256"),
            "gamma_source_json": base._repo_relative(regime.gamma_source_json),
            "gamma_source_sha256": base._sha256(regime.gamma_source_json),
            "source_plateau_iteration": source_plateau_iteration,
            "soft_plateau_depth": source_plateau_iteration + 3,
            "hard_max_depth": HARD_MAX_DEPTH,
            "novelty_gamma_settings": base._extract_source_gamma_settings(regime.gamma_source_json),
            "no_novelty_settings": _arm_settings(regime, "no_novelty"),
        }
    return rows, source_details


def _tsv_text(rows: Sequence[Mapping[str, str]]) -> str:
    import io

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(FIELDNAMES), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})
    return buf.getvalue()


def _manifest_payload(*, rows: Sequence[Mapping[str, str]], source_details: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    return {
        "schema": "paper_i_hh_u8_snake_novelty_ablation_optuna_manifest_v1",
        "batch_id": BATCH_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "chtc/phase3_optuna/generate_paper_i_hh_u8_snake_novelty_ablation_optuna_records.py",
        "run_class": "diagnostic",
        "table_label": "diagnostic:hh_u8_strong_hubbard_novelty_ablation",
        "suite_profile": TABLE_I_THREE_MODEL_HH_SYMMETRIC_U8_PROFILE,
        "records_tsv": base._repo_relative(output_dir / RECORDS),
        "record_id_file": base._repo_relative(output_dir / RECORD_IDS),
        "submit_file": base._repo_relative(SUBMIT_FILE),
        "record_ids": [row["record_id"] for row in rows],
        "record_count": len(rows),
        "regime_order": [regime.key for regime in REGIMES],
        "novelty_arms": ["novelty", "no_novelty"],
        "n_trials_per_job": N_TRIALS,
        "hard_max_depth": HARD_MAX_DEPTH,
        "adapt_drop_patience": ADAPT_DROP_PATIENCE,
        "adapt_drop_floor": ADAPT_DROP_FLOOR,
        "progress_contract": {
            "adapt_current_json_every_depth": 1,
            "adapt_current_json_keep_history_tail": 100,
            "hard_cap_fits_current_history_tail": HARD_MAX_DEPTH <= 100,
            "transfer_policy": "ON_EXIT_OR_EVICT",
            "transfer_output_files": ["raw_outputs", "logs"],
        },
        "source_map": base._repo_relative(SOURCE_MAP),
        "source_details": source_details,
        "promotion_status": "diagnostic_not_table_update",
    }


def render_artifacts(output_dir: Path | None = None) -> dict[Path, str]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    rows, source_details = build_rows(output_dir)
    artifacts: dict[Path, str] = {
        output_dir / RECORDS: _tsv_text(rows),
        output_dir / RECORD_IDS: base._ids_text(rows),
        output_dir / MANIFEST: json.dumps(
            _manifest_payload(rows=rows, source_details=source_details, output_dir=output_dir),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        SUBMIT_FILE: base._submit_text(
            records_path=output_dir / RECORDS,
            record_ids_path=output_dir / RECORD_IDS,
            job_batch_name=BATCH_ID,
        ),
    }
    source_map = base._load_source_map()
    for regime in REGIMES:
        regime_payload = base._nested(source_map, "regimes", regime.key, "methods", "SNAKE")
        plateau_payload = base._nested(source_map, "plateau_markers", regime.key, "SNAKE")
        if not isinstance(regime_payload, Mapping) or not isinstance(plateau_payload, Mapping):
            raise ValueError(f"missing source-map payload for {regime.key}")
        source_plateau_iteration = int(plateau_payload["iteration"])
        for arm in ("novelty", "no_novelty"):
            artifacts[_override_path(output_dir, regime, arm)] = (
                json.dumps(
                    _override_payload(
                        output_dir=output_dir,
                        regime=regime,
                        arm=arm,
                        source_entry=regime_payload,
                        source_plateau_iteration=source_plateau_iteration,
                    ),
                    indent=2,
                    sort_keys=True,
                )
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


def generate_records(output_dir: Path | None = None, *, write: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
    artifacts = write_artifacts(output_dir) if write else {path.name: str(path) for path in render_artifacts(output_dir)}
    rows, source_details = build_rows(output_dir)
    return {
        "schema": "paper_i_hh_u8_snake_novelty_ablation_optuna_generation_summary_v1",
        "batch_id": BATCH_ID,
        "output_dir": str(output_dir),
        "wrote_files": bool(write),
        "record_count": len(rows),
        "record_ids": [row["record_id"] for row in rows],
        "submit_file": str(SUBMIT_FILE),
        "source_details": source_details,
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
        print("Paper-I HH U/t=8 SNAKE novelty/no-novelty Optuna artifacts are current")
        return 0
    summary = generate_records(args.output_dir, write=bool(args.write))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
