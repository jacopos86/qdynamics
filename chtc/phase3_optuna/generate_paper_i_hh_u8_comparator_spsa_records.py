#!/usr/bin/env python3
"""Generate Paper-I HH U/t=8 comparator SPSA Optuna records."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import generate_paper_i_comparator_spsa_calibration_records as base  # noqa: E402
from pipelines.exact_bench.paper_i_hh_u8_comparator_spsa_optuna import (  # noqa: E402
    PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS,
    PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_VERSION,
    PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH,
    PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID,
    PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
    PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_ABS_DELTA_E,
    PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS,
    PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_LABEL,
    config_sha256_for_path,
    full_method_target_records,
    load_and_validate_config,
    target_by_id,
    validate_full_method_target_records,
    validate_method_id,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (  # noqa: E402
    PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS,
)
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id  # noqa: E402
from pipelines.exact_bench.table_i_static_benchmark import table_i_method_label  # noqa: E402

DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "chtc" / "phase3_optuna" / "config" / "paper_i_hh_u8_comparator_spsa_v1_smoke.json"
)
DEFAULT_FULL_CONFIG_PATH = (
    REPO_ROOT / "chtc" / "phase3_optuna" / "config" / "paper_i_hh_u8_comparator_spsa_v1_full_approved.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "chtc" / "phase3_optuna" / "input" / "paper_i_hh_u8_comparator_spsa_v1_smoke"
DEFAULT_QUEUE_OUTPUT_ROOT = Path("raw_outputs/paper_i_hh_u8_comparator_spsa_v1")

RECORDS_TSV = "paper_i_hh_u8_comparator_spsa_records.tsv"
RECORD_IDS_TXT = "paper_i_hh_u8_comparator_spsa_record_ids.txt"
SMOKE_RECORDS_TSV = "paper_i_hh_u8_comparator_spsa_smoke_records.tsv"
SMOKE_RECORD_IDS_TXT = "paper_i_hh_u8_comparator_spsa_smoke_record_ids.txt"
MANIFEST_JSON = "paper_i_hh_u8_comparator_spsa_records_manifest.json"
SMOKE_TARGET_IDS = (PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS[0],)
EXTRA_FIELDNAMES = (
    "u_over_t",
    "lambda_ep",
    "g_ep",
    "physics_profile",
    "hh_u8_spsa_scope",
)
FIELDNAMES = tuple(dict.fromkeys((*base.FIELDNAMES, *EXTRA_FIELDNAMES)))


def _repo_relative(path: str | Path) -> str:
    candidate = Path(path)
    resolved = candidate.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(candidate)


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _blank_optimizer_fields() -> dict[str, str]:
    return {field: "" for field in PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS}


def _optimizer_fields_for_method(method_id: str, *, maxiter_budget: int) -> dict[str, str]:
    method = validate_method_id(method_id)
    fields = _blank_optimizer_fields()
    fields["optimizer_profile"] = ""
    if method == "static_hea_qiskit_vqe":
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["hea"]
        fields["hea_optimizer"] = str(defaults["optimizer"])
        fields["hea_spsa_maxiter"] = str(int(maxiter_budget))
        fields["hea_spsa_seed"] = str(int(defaults["spsa_seed"]))
    elif method == "static_family_informed_vqe":
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["family_informed"]
        fields["family_informed_optimizer"] = str(defaults["optimizer"])
        fields["family_informed_spsa_maxiter"] = str(int(maxiter_budget))
        fields["family_informed_spsa_seed"] = str(int(defaults["spsa_seed"]))
    else:
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["adapt"]
        fields["adapt_optimizer_kind"] = str(defaults["optimizer_kind"])
        fields["adapt_spsa_maxiter"] = str(int(maxiter_budget))
        fields["adapt_spsa_seed"] = str(int(defaults["spsa_seed"]))
    return fields


def _stable_record_id(*, method_id: str, target_id: str, mode: str) -> str:
    return f"paper_i_hh_u8_comp_spsa__{mode}__{method_id}__{target_id}"


def _record_paths(record_id: str) -> dict[str, str]:
    root = Path("records") / record_id
    return {
        "record_output_dir": str(root),
        "progress_dir": str(root / "progress"),
        "summary_json": str(root / "summary.json"),
        "best_schedule_json": str(root / "best_schedule.json"),
        "heartbeat_json": str(root / "heartbeat.json"),
        "trial_output_template": str(root / "trial_{trial_number:04d}" / "cases" / "{case_id}"),
    }


def _reference_fields_for_target(target_id: str) -> dict[str, str]:
    target = target_by_id(target_id)
    spec = table_i_canonical_spec_by_case_id(target.family, target.case_ids[0], PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE)
    same_energy, same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=int(target.n_ph_work))
    ref_energy, ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=int(target.n_ph_ref))
    return {
        "table_label": "diagnostic:hh_u8_strong_hubbard_comparator_spsa",
        "hh_tableiii_regime": target.hh_regime,
        "n_ph_work": str(int(target.n_ph_work)),
        "n_ph_ref": str(int(target.n_ph_ref)),
        "same_cutoff_exact_gs_energy": repr(float(same_energy)),
        "exact_reference_energy": repr(float(ref_energy)),
        "exact_reference_n_ph_max": str(int(target.n_ph_ref)),
        "same_cutoff_reference_energy_key": same_key,
        "reference_cutoff_energy_key": ref_key,
        "reference_energy_status": "ok",
        "primary_energy_metric": "higher_cutoff_reference_abs_delta_e",
        "same_cutoff_error_role": "diagnostic_only",
        "u_over_t": repr(float(target.u_over_t)),
        "lambda_ep": repr(float(target.lambda_ep)),
        "g_ep": repr(float(target.g_ep)),
        "physics_profile": PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
        "hh_u8_spsa_scope": "new_strong_hubbard_sector_v1",
    }


def _rows_for_method_targets(
    method_target_records: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    config_sha256: str,
    generation_mode: str,
    queue_output_root: str | Path,
) -> list[dict[str, str]]:
    budgets = config.get("method_maxiter_budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError("validated U8 comparator SPSA config missing method_maxiter_budgets mapping")
    clipping = config.get("clipping_log10_error_ratio")
    if not isinstance(clipping, Sequence) or isinstance(clipping, (str, bytes)) or len(clipping) != 2:
        raise ValueError("validated U8 comparator SPSA config has malformed clipping_log10_error_ratio")
    rows: list[dict[str, str]] = []
    for stub in method_target_records:
        method_id = validate_method_id(str(stub["method_id"]))
        target = target_by_id(str(stub["target_id"]))
        maxiter_budget = int(budgets[method_id])
        record_id = _stable_record_id(method_id=method_id, target_id=target.target_id, mode=generation_mode)
        row: dict[str, str] = {
            "record_id": record_id,
            "profile_id": PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID,
            "record_schema": "paper_i_hh_u8_comparator_spsa_record_v1",
            "run_class": "smoke" if generation_mode == "smoke" else "calibration_candidate_not_table_evidence",
            "method_id": method_id,
            "algorithm_id": method_id,
            "method_label": table_i_method_label(method_id),
            "target_id": target.target_id,
            "target_family": target.family,
            "family": target.family,
            "case_ids_json": _json_compact(list(target.case_ids)),
            "case_count": str(len(target.case_ids)),
            "suite_profile": PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
            "optimizer_profile": "",
            "config_path": _repo_relative(config_path),
            "config_sha256": str(config_sha256),
            "config_version": PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_VERSION,
            "config_mode": str(config.get("mode")),
            "approved_for_full_generation": "true" if bool(config.get("approved_for_full_generation")) else "false",
            "approved_by": "" if config.get("approved_by") is None else str(config.get("approved_by")),
            "approved_at": "" if config.get("approved_at") is None else str(config.get("approved_at")),
            "plan_path": PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH,
            "n_trials": str(int(config["n_trials"])),
            "sampler_seed": str(int(config["sampler_seed"])),
            "n_jobs": "1",
            "method_maxiter_budget": str(maxiter_budget),
            "failure_penalty": str(float(config["failure_penalty"])),
            "objective_mode": "mean_clipped_log10_abs_delta_e_plus_resource_tiebreak_v1",
            "target_abs_delta_e": str(float(PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_ABS_DELTA_E)),
            "target_label": PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_LABEL,
            "clip_log10_error_ratio_min": str(float(clipping[0])),
            "clip_log10_error_ratio_max": str(float(clipping[1])),
            "resource_tiebreak_weight": str(float(config["resource_tiebreak_weight"])),
            "resource_metric_precedence_json": _json_compact(list(base.RESOURCE_METRIC_PRECEDENCE)),
            "search_space_fields_json": _json_compact(
                sorted(str(name) for name in config["per_method_search_spaces"][method_id])
            ),
            "queue_output_root": str(queue_output_root),
            "repair_scope": "",
            "calibration_stage": str(generation_mode),
            "calibration_usable_status_policy": "",
            "quality_nonpassing_penalty": "",
            "resource_qubit_cap": "0",
            "resource_pool_term_cap": "0",
            **_record_paths(record_id),
            **_reference_fields_for_target(target.target_id),
            **_optimizer_fields_for_method(method_id, maxiter_budget=maxiter_budget),
        }
        for field in PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS:
            row.setdefault(field, "")
        rows.append({field: str(row.get(field, "")) for field in FIELDNAMES})
    rows.sort(key=lambda item: (item["method_id"], item["target_id"], item["record_id"]))
    return rows


def _select_smoke_stubs() -> tuple[dict[str, object], ...]:
    return full_method_target_records(
        method_ids=PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS,
        target_ids=SMOKE_TARGET_IDS,
    )


def build_rows(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    generation_mode: str | None = None,
    queue_output_root: str | Path = DEFAULT_QUEUE_OUTPUT_ROOT,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    config = load_and_validate_config(config_path)
    mode = str(generation_mode or config["mode"]).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError(f"generation_mode must be smoke or full; got {generation_mode!r}")
    if mode == "full" and str(config.get("mode")) != "full":
        raise ValueError("full generation requires a config with mode='full' and approved full-generation metadata")
    config_hash = str(config.get("config_sha256") or config_sha256_for_path(config_path))
    if mode == "full":
        full_stubs = full_method_target_records()
        validate_full_method_target_records(full_stubs)
        records = _rows_for_method_targets(
            full_stubs,
            config=config,
            config_path=config_path,
            config_sha256=config_hash,
            generation_mode="full",
            queue_output_root=queue_output_root,
        )
        smoke_config = load_and_validate_config(DEFAULT_CONFIG_PATH)
        smoke_records = _rows_for_method_targets(
            _select_smoke_stubs(),
            config=smoke_config,
            config_path=DEFAULT_CONFIG_PATH,
            config_sha256=str(smoke_config["config_sha256"]),
            generation_mode="smoke",
            queue_output_root=queue_output_root,
        )
    else:
        records = _rows_for_method_targets(
            _select_smoke_stubs(),
            config=config,
            config_path=config_path,
            config_sha256=config_hash,
            generation_mode="smoke",
            queue_output_root=queue_output_root,
        )
        smoke_records = list(records)
    return records, smoke_records, config


def _write_tsv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_ids(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row['record_id']}\n" for row in rows), encoding="utf-8")


def write_outputs(
    output_dir: str | Path,
    records: Sequence[Mapping[str, str]],
    smoke_records: Sequence[Mapping[str, str]],
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    generation_mode: str,
    queue_output_root: str | Path = DEFAULT_QUEUE_OUTPUT_ROOT,
) -> dict[str, Any]:
    output = Path(output_dir)
    records_path = output / RECORDS_TSV
    ids_path = output / RECORD_IDS_TXT
    smoke_path = output / SMOKE_RECORDS_TSV
    smoke_ids_path = output / SMOKE_RECORD_IDS_TXT
    manifest_path = output / MANIFEST_JSON
    _write_tsv(records_path, records)
    _write_ids(ids_path, records)
    _write_tsv(smoke_path, smoke_records)
    _write_ids(smoke_ids_path, smoke_records)
    method_targets = sorted({(row["method_id"], row["target_id"]) for row in records})
    manifest = {
        "schema": "paper_i_hh_u8_comparator_spsa_records_manifest_v1",
        "profile_id": PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID,
        "generation_mode": str(generation_mode),
        "run_class": "smoke" if generation_mode == "smoke" else "calibration_candidate_not_table_evidence",
        "evidence_role": "calibration_only_not_manuscript_table_evidence",
        "table_evidence_status": "not_table_evidence",
        "suite_profile": PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
        "optimizer_profile": "explicit_spsa_env_overlay_no_named_profile",
        "config_path": _repo_relative(config_path),
        "config_sha256": str(config.get("config_sha256") or config_sha256_for_path(config_path)),
        "config_mode": str(config.get("mode")),
        "approved_for_full_generation": bool(config.get("approved_for_full_generation")),
        "approved_by": config.get("approved_by"),
        "approved_at": config.get("approved_at"),
        "plan_path": PAPER_I_HH_U8_COMPARATOR_SPSA_PLAN_PATH,
        "record_count": len(records),
        "smoke_record_count": len(smoke_records),
        "expected_full_record_count": 12,
        "method_ids": list(PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_METHOD_IDS),
        "target_ids": list(PAPER_I_HH_U8_COMPARATOR_SPSA_TARGET_IDS),
        "method_target_pairs": [list(pair) for pair in method_targets],
        "smoke_target_ids": list(SMOKE_TARGET_IDS),
        "n_trials": int(config["n_trials"]),
        "sampler_seed": int(config["sampler_seed"]),
        "n_jobs": 1,
        "method_maxiter_budgets": dict(config["method_maxiter_budgets"]),
        "failure_penalty": float(config["failure_penalty"]),
        "clipping_log10_error_ratio": list(config["clipping_log10_error_ratio"]),
        "resource_tiebreak_weight": float(config["resource_tiebreak_weight"]),
        "resource_metric_precedence": list(base.RESOURCE_METRIC_PRECEDENCE),
        "allowed_schedule_fields_by_method": {
            method: list(fields) for method, fields in PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.items()
        },
        "queue_output_root": str(queue_output_root),
        "smoke_config_path": _repo_relative(DEFAULT_CONFIG_PATH),
        "smoke_config_sha256": config_sha256_for_path(DEFAULT_CONFIG_PATH),
        "paths": {
            "records_tsv": _repo_relative(records_path),
            "record_ids_txt": _repo_relative(ids_path),
            "smoke_records_tsv": _repo_relative(smoke_path),
            "smoke_record_ids_txt": _repo_relative(smoke_ids_path),
            "manifest_json": _repo_relative(manifest_path),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def generate_records(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    generation_mode: str | None = None,
    queue_output_root: str | Path = DEFAULT_QUEUE_OUTPUT_ROOT,
) -> dict[str, Any]:
    records, smoke_records, config = build_rows(
        config_path=config_path,
        generation_mode=generation_mode,
        queue_output_root=queue_output_root,
    )
    mode = str(generation_mode or config["mode"]).strip().lower()
    return write_outputs(
        output_dir,
        records,
        smoke_records,
        config=config,
        config_path=config_path,
        generation_mode=mode,
        queue_output_root=queue_output_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Paper-I HH U/t=8 comparator SPSA Optuna record TSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("smoke", "full"), default=None)
    parser.add_argument("--queue-output-root", type=Path, default=DEFAULT_QUEUE_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_records(
        output_dir=args.output_dir,
        config_path=args.config_path,
        generation_mode=args.mode,
        queue_output_root=args.queue_output_root,
    )
    print(json.dumps(manifest["paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
