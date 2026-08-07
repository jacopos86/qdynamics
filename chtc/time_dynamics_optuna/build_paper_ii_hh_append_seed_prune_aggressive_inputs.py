#!/usr/bin/env python3
"""Build Paper-II HH append-seed aggressive prune Optuna inputs.

The emitted records are diagnostic class-settings candidates only.  They use
the weak-weak and strong-weak HH append seeds from the Paper-II seed-track v1
manifest, shorten the dynamics mesh, and force the strict exact-free
append/prune-aggressive profile.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASE_MANIFEST = (
    REPO_ROOT / "chtc/generic_time_dynamics_table/input/paper_ii_hh_seed_tracks_cases_v1.json"
)
DEFAULT_OUT_DIR = REPO_ROOT / "chtc/time_dynamics_optuna/input"
OUTPUT_STEM = "paper_ii_hh_append_seed_prune_aggressive_20260601_v1"
QUEUE_STEM = "paper_ii_hh_append_seed_prune_aggressive_20260601_v1"
RECORD_PREFIX = "paper_ii_hh_append_prune_aggressive_v1"
PROFILE = "strict_qpu_faithful_append_prune_aggressive_v1"
TARGET_CASE_IDS = (
    "table1_hh_weak_weak_append_A0p2_t8_dt321_seedtracks_v1",
    "table1_hh_weak_weak_append_A0p6_t8_dt321_seedtracks_v1",
    "table1_hh_strong_weak_append_A0p2_t8_dt321_seedtracks_v1",
    "table1_hh_strong_weak_append_A0p6_t8_dt321_seedtracks_v1",
)

HEADER = [
    "record_id",
    "queue",
    "validation_profile",
    "family",
    "tuning_class",
    "source_artifact_json",
    "artifact_json",
    "study_profile",
    "route_label",
    "loader_mode",
    "generator_family",
    "fallback_family",
    "append_pool_family",
    "lock_fixed_manifold",
    "allow_repeats",
    "t_final",
    "num_times",
    "exact_steps_multiplier",
    "enable_drive",
    "disable_drive",
    "drive_A",
    "drive_omega",
    "drive_tbar",
    "drive_phi",
    "drive_pattern",
    "drive_custom_weights",
    "drive_include_identity",
    "drive_time_sampling",
    "drive_t0",
    "n_trials",
    "n_startup_trials",
    "sampler_seed",
    "n_jobs",
    "no_baseline_trial",
    "pair",
    "objective_window_start",
    "objective_window_end",
    "spectra_detrend",
    "spectra_window",
    "max_peaks",
    "max_harmonic",
    "skip_spectra_pdf",
    "min_completed_trials",
    "require_full_horizon",
    "expected_drive_enabled",
    "objective_weight_pair_mae_over_span",
    "objective_weight_epsilon_osc",
    "objective_weight_peak_omega",
    "objective_weight_pair_corr_defect",
    "objective_weight_site_mae",
    "objective_weight_energy_mae",
    "objective_weight_energy_bias_abs",
    "objective_weight_energy_max_abs",
    "objective_weight_energy_under_response_mean",
    "objective_weight_energy_under_response_max",
    "objective_weight_fidelity_defect",
    "objective_weight_runtime_count",
    "objective_weight_append_count",
    "objective_weight_prune_count",
    "objective_weight_rk4_count",
    "invalid_max_mean_energy_mae",
    "invalid_max_final_energy_mae",
    "invalid_min_fidelity",
    "invalid_max_mean_total_occupation_mae",
    "invalid_max_primary_observable_mae_over_span",
    "class_settings_source",
    "class_settings_lock_json",
    "case_id",
    "seed_track",
    "same_seed_comparator_group_id",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: str | Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p.relative_to(REPO_ROOT))
    return str(p)


def _read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, Mapping) else None
    if not isinstance(cases, list):
        raise ValueError(f"case manifest lacks cases list: {path}")
    return [dict(case) for case in cases if isinstance(case, Mapping)]


def _selected_cases(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(case.get("case_id", "")): dict(case) for case in cases}
    missing = [case_id for case_id in TARGET_CASE_IDS if case_id not in by_id]
    if missing:
        raise ValueError("missing target cases: " + ", ".join(missing))
    return [by_id[case_id] for case_id in TARGET_CASE_IDS]


def _record_from_case(
    case: Mapping[str, Any],
    *,
    t_final: float,
    num_times: int,
    n_trials: int,
    n_startup_trials: int,
    sampler_seed_base: int,
) -> dict[str, str]:
    case_id = str(case["case_id"])
    record_id = f"{RECORD_PREFIX}__{case_id}"
    metadata = case.get("metadata", {}) if isinstance(case.get("metadata"), Mapping) else {}
    drive = metadata.get("drive", {}) if isinstance(metadata.get("drive"), Mapping) else {}
    seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock"), Mapping) else {}
    source_artifact = Path("chtc/generic_time_dynamics_table/input") / str(case["artifact_json"])
    staged_artifact = Path("chtc/time_dynamics_optuna/input/seed_artifacts") / f"{record_id}.json"
    index_hash = sum(ord(ch) for ch in record_id) % 100000
    row = {key: "" for key in HEADER}
    row.update(
        {
            "record_id": record_id,
            "queue": QUEUE_STEM,
            "validation_profile": "strict_qpu_faithful",
            "family": str(case["family"]),
            "tuning_class": str(case["tuning_class"]),
            "source_artifact_json": _rel(source_artifact),
            "artifact_json": _rel(staged_artifact),
            "study_profile": PROFILE,
            "route_label": f"Paper-II diagnostic HH append-seed aggressive prune probe {case_id}",
            "loader_mode": str(case.get("loader_mode", "replay_family")),
            "generator_family": str(case.get("generator_family", "full_meta")),
            "fallback_family": str(case.get("fallback_family", "full_meta")),
            "append_pool_family": str(case.get("append_pool_family", "full_meta")),
            "lock_fixed_manifold": "0",
            "allow_repeats": "0",
            "t_final": str(float(t_final)),
            "num_times": str(int(num_times)),
            "exact_steps_multiplier": "2",
            "enable_drive": "1" if bool(drive.get("enable_drive", metadata.get("enable_drive", True))) else "0",
            "disable_drive": "1" if bool(metadata.get("disable_drive", False)) else "0",
            "drive_A": str(float(drive.get("A", 0.0))),
            "drive_omega": str(float(drive.get("omega", 1.0))),
            "drive_tbar": str(float(drive.get("tbar", 1.0))),
            "drive_phi": str(float(drive.get("phi", 0.0))),
            "drive_pattern": str(drive.get("pattern", "staggered")),
            "drive_custom_weights": str(drive.get("custom_weights", "")),
            "drive_include_identity": "1" if bool(drive.get("include_identity", False)) else "0",
            "drive_time_sampling": str(drive.get("time_sampling", "midpoint")),
            "drive_t0": str(float(drive.get("t0", 0.0))),
            "n_trials": str(int(n_trials)),
            "n_startup_trials": str(int(n_startup_trials)),
            "sampler_seed": str(int(sampler_seed_base + index_hash)),
            "n_jobs": "1",
            "no_baseline_trial": "0",
            "pair": "auto",
            "spectra_detrend": "constant",
            "spectra_window": "hann",
            "max_peaks": "5",
            "max_harmonic": "12",
            "skip_spectra_pdf": "1",
            "min_completed_trials": str(max(1, min(4, int(n_trials)))),
            "require_full_horizon": "1",
            "expected_drive_enabled": "1" if bool(drive.get("enable_drive", metadata.get("enable_drive", True))) else "0",
            "objective_weight_pair_mae_over_span": "0",
            "objective_weight_epsilon_osc": "0",
            "objective_weight_peak_omega": "0",
            "objective_weight_pair_corr_defect": "0",
            "objective_weight_site_mae": "0",
            "objective_weight_energy_mae": "0",
            "objective_weight_energy_bias_abs": "0",
            "objective_weight_energy_max_abs": "0",
            "objective_weight_energy_under_response_mean": "0",
            "objective_weight_energy_under_response_max": "0",
            "objective_weight_fidelity_defect": "0",
            "objective_weight_runtime_count": "0.002",
            "objective_weight_append_count": "0.03",
            "objective_weight_prune_count": "1.0",
            "objective_weight_rk4_count": "0",
            "class_settings_source": "diagnostic_hh_append_seed_prune_probe_not_promoted",
            "case_id": case_id,
            "seed_track": str(seed_lock.get("seed_track", "")),
            "same_seed_comparator_group_id": str(seed_lock.get("same_seed_comparator_group_id", "")),
        }
    )
    return row


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _stage_seed_artifacts(rows: Sequence[Mapping[str, str]]) -> list[str]:
    staged: list[str] = []
    for row in rows:
        source = REPO_ROOT / row["source_artifact_json"]
        target = REPO_ROOT / row["artifact_json"]
        if not source.exists():
            raise FileNotFoundError(f"missing source seed artifact for {row['record_id']}: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        staged.append(_rel(target))
    return staged


def build_inputs(
    *,
    case_manifest: Path,
    out_dir: Path,
    t_final: float,
    num_times: int,
    n_trials: int,
    n_startup_trials: int,
    sampler_seed_base: int,
) -> dict[str, Any]:
    rows = [
        _record_from_case(
            case,
            t_final=t_final,
            num_times=num_times,
            n_trials=n_trials,
            n_startup_trials=n_startup_trials,
            sampler_seed_base=sampler_seed_base,
        )
        for case in _selected_cases(_read_cases(case_manifest))
    ]
    records_path = out_dir / f"{OUTPUT_STEM}_records.tsv"
    record_ids_path = out_dir / f"{OUTPUT_STEM}_record_ids.txt"
    manifest_path = out_dir / f"{OUTPUT_STEM}_manifest.json"
    staged_seed_artifacts = _stage_seed_artifacts(rows)
    _write_tsv(records_path, rows)
    record_ids_path.write_text(
        "\n".join(str(row["record_id"]) for row in rows) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema": "paper_ii_hh_append_seed_prune_aggressive_inputs_v1",
        "generated_utc": _now(),
        "case_manifest": _rel(case_manifest),
        "case_ids": list(TARGET_CASE_IDS),
        "record_count": len(rows),
        "study_profile": PROFILE,
        "validation_profile": "strict_qpu_faithful",
        "records": _rel(records_path),
        "record_ids": _rel(record_ids_path),
        "t_final": float(t_final),
        "num_times": int(num_times),
        "n_trials": int(n_trials),
        "n_startup_trials": int(n_startup_trials),
        "objective_contract": {
            "exact_reference_objective_weights": 0,
            "final_runtime_parameter_count": 0.002,
            "append_count": 0.03,
            "prune_count": 1.0,
            "rk4_count": 0.0,
        },
        "diagnostic_only_not_paper_evidence": True,
        "training_role": "diagnostic_append_seed_prune_probe",
        "controller_settings_scope": "diagnostic_hh_append_seed_probe_not_promoted",
        "promotion_level": "diagnostic_only_not_paper_evidence",
        "class_tuned_result_locked": False,
        "staged_seed_artifact_count": len(staged_seed_artifacts),
        "staged_seed_artifacts": staged_seed_artifacts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-manifest", type=Path, default=DEFAULT_CASE_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--t-final", type=float, default=2.0)
    parser.add_argument("--num-times", type=int, default=81)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--n-startup-trials", type=int, default=8)
    parser.add_argument("--sampler-seed-base", type=int, default=93000)
    args = parser.parse_args()
    manifest = build_inputs(
        case_manifest=args.case_manifest.resolve(),
        out_dir=args.out_dir.resolve(),
        t_final=float(args.t_final),
        num_times=int(args.num_times),
        n_trials=int(args.n_trials),
        n_startup_trials=int(args.n_startup_trials),
        sampler_seed_base=int(args.sampler_seed_base),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
