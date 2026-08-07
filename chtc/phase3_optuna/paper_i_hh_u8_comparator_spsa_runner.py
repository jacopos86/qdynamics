#!/usr/bin/env python3
"""Run one Paper-I HH U/t=8 comparator SPSA Optuna record."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import paper_i_comparator_spsa_calibration_runner as base_runner  # noqa: E402
from chtc.phase3_optuna.generate_paper_i_hh_u8_comparator_spsa_records import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_QUEUE_OUTPUT_ROOT,
    RECORDS_TSV,
)
from pipelines.exact_bench.paper_i_hh_u8_comparator_spsa_optuna import (  # noqa: E402
    PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID,
    PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
    config_sha256_for_path,
    load_and_validate_config,
    target_by_id,
    validate_method_id,
)

DEFAULT_RECORDS_PATH = (
    REPO_ROOT
    / "chtc"
    / "phase3_optuna"
    / "input"
    / "paper_i_hh_u8_comparator_spsa_v1_smoke"
    / RECORDS_TSV
)
DEFAULT_OUTPUT_ROOT = DEFAULT_QUEUE_OUTPUT_ROOT


def _case_ids_from_row(row: Mapping[str, str]) -> tuple[str, ...]:
    raw = str(row.get("case_ids_json", "")).strip()
    if not raw:
        raise ValueError("U8 comparator SPSA record must include non-empty case_ids_json")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError("U8 comparator SPSA record has malformed case_ids_json") from exc
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError("U8 comparator SPSA record case_ids_json must be a non-empty JSON string array")
    return tuple(str(item).strip() for item in payload)


def _finite_float(value: object, *, field: str) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except Exception as exc:
        raise ValueError(f"{field} must be finite numeric; got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite numeric; got {value!r}")
    return float(parsed)


def validate_record(row: Mapping[str, str], *, config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    if str(row.get("profile_id", "")) != PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID:
        raise ValueError(
            f"record profile_id must be {PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID!r}; got {row.get('profile_id')!r}"
        )
    method_id = validate_method_id(str(row.get("method_id") or row.get("algorithm_id") or ""))
    if str(row.get("algorithm_id") or "").strip() != method_id:
        raise ValueError(f"U8 comparator SPSA algorithm_id={row.get('algorithm_id')!r} must match method_id={method_id!r}")
    target = target_by_id(str(row.get("target_id") or ""))
    case_ids = _case_ids_from_row(row)
    if case_ids != target.case_ids:
        raise ValueError(f"record target_id={target.target_id!r} case_ids={case_ids!r}; expected {target.case_ids!r}")
    family = str(row.get("family") or row.get("target_family") or "").strip()
    if family != target.family:
        raise ValueError(f"record family={family!r}; expected target family {target.family!r}")
    if str(row.get("suite_profile", "")) != PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE:
        raise ValueError(f"record suite_profile must be {PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE!r}")
    optimizer_profile = str(row.get("optimizer_profile", "") or "").strip()
    if optimizer_profile not in {"", "off", "none"}:
        raise ValueError(
            "U8 comparator SPSA records must leave optimizer_profile blank and use explicit SPSA env fields; "
            f"got {optimizer_profile!r}."
        )
    expected_hash = config_sha256_for_path(config_path)
    if str(row.get("config_sha256", "")).strip() != expected_hash:
        raise ValueError(
            f"record config_sha256={row.get('config_sha256')!r} does not match config path hash {expected_hash!r}"
        )
    if int(row.get("n_jobs") or "1") != 1:
        raise ValueError("U8 comparator SPSA runner requires n_jobs=1 because generic_static_benchmark.run_single reads env")
    if int(row.get("n_trials") or 0) != int(config["n_trials"]):
        raise ValueError(f"record n_trials={row.get('n_trials')!r} does not match config n_trials={config['n_trials']!r}")
    if int(row.get("method_maxiter_budget") or 0) != int(config["method_maxiter_budgets"][method_id]):
        raise ValueError("record method_maxiter_budget does not match config method_maxiter_budgets")
    if str(row.get("primary_energy_metric") or "") != "higher_cutoff_reference_abs_delta_e":
        raise ValueError("U8 comparator SPSA records must use higher_cutoff_reference_abs_delta_e as primary metric")
    if str(row.get("same_cutoff_error_role") or "") != "diagnostic_only":
        raise ValueError("U8 comparator SPSA same_cutoff_error_role must be diagnostic_only")
    for field in ("same_cutoff_exact_gs_energy", "exact_reference_energy"):
        _finite_float(row.get(field), field=field)
    if int(float(str(row.get("n_ph_work") or "0"))) != int(target.n_ph_work):
        raise ValueError(f"n_ph_work mismatch for {target.target_id}: {row.get('n_ph_work')} != {target.n_ph_work}")
    if int(float(str(row.get("n_ph_ref") or "0"))) != int(target.n_ph_ref):
        raise ValueError(f"n_ph_ref mismatch for {target.target_id}: {row.get('n_ph_ref')} != {target.n_ph_ref}")
    if int(float(str(row.get("exact_reference_n_ph_max") or "0"))) != int(target.n_ph_ref):
        raise ValueError("exact_reference_n_ph_max must match n_ph_ref")
    return {"method_id": method_id, "target": target, "case_ids": case_ids, "family": family}


def _install_u8_contract() -> None:
    base_runner.PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID = PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID
    base_runner.PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD = (
        PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD
    )
    base_runner.validate_record = validate_record  # type: ignore[assignment]
    base_runner.target_by_id = target_by_id  # type: ignore[assignment]
    base_runner.validate_method_id = validate_method_id  # type: ignore[assignment]
    base_runner.config_sha256_for_path = config_sha256_for_path  # type: ignore[assignment]
    base_runner.load_and_validate_config = load_and_validate_config  # type: ignore[assignment]


def run_calibration_record(
    *,
    record: Mapping[str, str],
    config: Mapping[str, Any],
    config_path: str | Path,
    out_root: str | Path,
    study_factory: Callable[..., Any] = base_runner._create_study,
    storage: str | None = None,
) -> dict[str, Any]:
    _install_u8_contract()
    return base_runner.run_calibration_record(
        record=record,
        config=config,
        config_path=config_path,
        out_root=out_root,
        study_factory=study_factory,
        storage=storage,
    )


def run_record(
    *,
    record_id: str,
    records_path: str | Path,
    config_path: str | Path,
    out_root: str | Path,
    storage: str | None = None,
    study_factory: Callable[..., Any] = base_runner._create_study,
) -> dict[str, Any]:
    config = load_and_validate_config(config_path)
    row = base_runner.load_record(records_path, record_id)
    return run_calibration_record(
        record=row,
        config=config,
        config_path=config_path,
        out_root=out_root,
        storage=storage,
        study_factory=study_factory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Paper-I HH U/t=8 comparator SPSA Optuna method-target record.")
    parser.add_argument("record_id")
    parser.add_argument("records_path", nargs="?", default=None)
    parser.add_argument("out_root", nargs="?", default=None)
    parser.add_argument("--config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--storage", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records_path = Path(args.records_path or os.environ.get("PAPER_I_HH_U8_COMPARATOR_SPSA_RECORDS_PATH", "") or DEFAULT_RECORDS_PATH)
    row = base_runner.load_record(records_path, str(args.record_id))
    default_record_out = Path(str(row.get("queue_output_root") or DEFAULT_OUTPUT_ROOT)) / str(
        row.get("record_output_dir") or row["record_id"]
    )
    out_root = Path(args.out_root or default_record_out)
    summary = run_calibration_record(
        record=row,
        config=load_and_validate_config(args.config_path),
        config_path=args.config_path,
        out_root=out_root,
        storage=args.storage,
    )
    print(base_runner._compact_json({"summary_json": summary["summary_json"], "ok": summary["ok"], "status": summary["status"]}))
    return 0 if bool(summary.get("ok")) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
