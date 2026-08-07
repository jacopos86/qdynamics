#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
import tarfile
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna import run_task  # noqa: E402
try:
    from chtc.phase3_optuna.paper_i_clean_ladder_contract import (  # noqa: E402
        PAPER_I_CLEAN_TAU_PHYS,
        PAPER_I_CLEAN_TAU_TIGHT,
        PAPER_I_LADDER_STAGE_CONFIGS,
        validate_clean_ladder_source_metadata,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in stripped local checkouts
    PAPER_I_CLEAN_TAU_PHYS = float("nan")
    PAPER_I_CLEAN_TAU_TIGHT = float("nan")
    PAPER_I_LADDER_STAGE_CONFIGS: dict[str, Any] = {}

    def validate_clean_ladder_source_metadata(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["paper_i_clean_ladder_contract_module_missing"]


try:
    from chtc.phase3_optuna.paper_i_table_i_audit_escalation import (  # noqa: E402
        validate_candidate_row_authorization,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised in stripped local checkouts

    def validate_candidate_row_authorization(*_args: Any, **_kwargs: Any) -> list[str]:
        return ["paper_i_table_i_audit_escalation_module_missing"]
from pipelines.exact_bench.paper_i_hh_shared_spsa_calibration import (  # noqa: E402
    PAPER_I_HH_SHARED_SPSA_CALIBRATION_PROFILE_ID,
    config_sha256_for_path as shared_spsa_config_sha256_for_path,
    load_and_validate_config as load_and_validate_shared_spsa_config,
    method_by_key_or_id as shared_spsa_method_by_key_or_id,
    normalize_spsa_refit_engine_key as normalize_shared_spsa_refit_engine_key,
)

PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID = "paper_i_comparator_spsa_optuna_calibration_v1"
PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID = "paper_i_hh_u8_comparator_spsa_optuna_v1"
HH_GEO_QEB_TABLEIII_REPAIR_SCOPE = "hh_geo_qeb_tableiii_v1"


def _resolve(path_value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else repo_root / path


def _read_record_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"record id file not found: {path}")
    record_ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        record_ids.append(stripped.split()[0])
    return record_ids


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _positive_int_from_row(row: Mapping[str, str], field: str, blockers: list[str], *, min_value: int = 1) -> int | None:
    raw = str(row.get(field) or "").strip()
    try:
        value = int(raw)
    except Exception:
        blockers.append(f"calibration_{field}_invalid:{raw!r}")
        return None
    if value < int(min_value):
        blockers.append(f"calibration_{field}_below_minimum:{value}:min={min_value}")
        return None
    return value


def _finite_float_from_row(
    row: Mapping[str, str],
    field: str,
    blockers: list[str],
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float | None:
    raw = str(row.get(field) or "").strip()
    try:
        value = float(raw)
    except Exception:
        blockers.append(f"calibration_{field}_invalid:{raw!r}")
        return None
    if not math.isfinite(value):
        blockers.append(f"calibration_{field}_nonfinite:{raw!r}")
        return None
    if positive and value <= 0.0:
        blockers.append(f"calibration_{field}_nonpositive:{value}")
        return None
    if nonnegative and value < 0.0:
        blockers.append(f"calibration_{field}_negative:{value}")
        return None
    return value


def _submit_argument_tokens(contract: Mapping[str, Any]) -> list[str]:
    try:
        return shlex.split(str(contract.get("arguments") or ""))
    except ValueError:
        return []


def _calibration_submit_config_path(contract: Mapping[str, Any], row: Mapping[str, str]) -> str:
    env = contract.get("environment")
    if isinstance(env, Mapping):
        value = str(env.get("PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_PATH") or "").strip()
        if value:
            return value
    tokens = _submit_argument_tokens(contract)
    if "--config" in tokens:
        idx = tokens.index("--config")
        if idx + 1 < len(tokens):
            return tokens[idx + 1]
    if len(tokens) >= 4 and tokens[0] == "$(record_id)":
        return tokens[3]
    return str(row.get("config_path") or "").strip()


def _calibration_dependency_checks(
    *,
    contract: Mapping[str, Any],
    transfer_inputs: Sequence[str],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    requirements_path = repo_root / "chtc" / "phase3_optuna" / "requirements-chtc.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8") if requirements_path.exists() else ""
    for package in ("optuna", "qiskit-algorithms"):
        declared = any(line.strip().split("==", 1)[0] == package for line in requirements_text.splitlines())
        checks.append(
            {
                "check": "chtc_requirements_declares_python_package",
                "package": package,
                "requirements_path": run_task._safe_relative(requirements_path, repo_root=repo_root),
                "ok": bool(declared),
            }
        )
        if not declared:
            blockers.append(f"dependency_not_declared_in_requirements:{package}")
    executable = str(contract.get("executable") or "").strip()
    executable_path = run_task._resolve_under_repo(executable, repo_root=repo_root) if executable else repo_root
    executable_exists = bool(executable and executable_path.exists())
    executable_visible = bool(
        executable and run_task._is_sandbox_visible(executable, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    checks.append(
        {
            "check": "submit_executable_visible",
            "executable": executable,
            "exists": executable_exists,
            "sandbox_visible": executable_visible,
            "ok": executable_exists and executable_visible,
        }
    )
    if not executable_exists:
        blockers.append(f"submit_executable_missing:{executable}")
    if executable and not executable_visible:
        blockers.append(f"submit_executable_not_transferred:{executable}")
    shell_wrapper = repo_root / "chtc" / "phase3_optuna" / "run_paper_i_comparator_spsa_calibration_task.sh"
    apptainer_wrapper = repo_root / "chtc" / "phase3_optuna" / "run_paper_i_comparator_spsa_calibration_task_apptainer.sh"
    shell_text = shell_wrapper.read_text(encoding="utf-8") if shell_wrapper.exists() else ""
    apptainer_text = apptainer_wrapper.read_text(encoding="utf-8") if apptainer_wrapper.exists() else ""
    wrapper_runtime_ok = all(token in shell_text for token in ("optuna", "qiskit_algorithms", "paper_i_comparator_spsa_calibration_runner"))
    checks.append(
        {
            "check": "wrapper_runtime_dependency_imports_present",
            "wrapper": run_task._safe_relative(shell_wrapper, repo_root=repo_root),
            "ok": bool(wrapper_runtime_ok),
        }
    )
    if not wrapper_runtime_ok:
        blockers.append("wrapper_dependency_check_missing:optuna_or_qiskit_algorithms")
    if executable.endswith("_apptainer.sh"):
        apptainer_ok = "run_paper_i_comparator_spsa_calibration_task.sh" in apptainer_text
        checks.append(
            {
                "check": "apptainer_wrapper_invokes_shell_wrapper",
                "wrapper": run_task._safe_relative(apptainer_wrapper, repo_root=repo_root),
                "ok": bool(apptainer_ok),
            }
        )
        if not apptainer_ok:
            blockers.append("apptainer_wrapper_does_not_invoke_calibration_shell_wrapper")
    return checks, blockers


def _calibration_output_expectations(row: Mapping[str, str], blockers: list[str]) -> dict[str, str]:
    record_output_dir = str(row.get("record_output_dir") or "").strip()
    progress_dir = str(row.get("progress_dir") or "").strip()
    summary_json = str(row.get("summary_json") or "").strip()
    best_schedule_json = str(row.get("best_schedule_json") or "").strip()
    heartbeat_json = str(row.get("heartbeat_json") or "").strip()
    expected = {
        "record_output_dir": record_output_dir,
        "progress_dir": f"{record_output_dir}/progress" if record_output_dir else "",
        "current_best_json": f"{record_output_dir}/progress/current_best.json" if record_output_dir else "",
        "trial_events_jsonl": f"{record_output_dir}/progress/trial_events.jsonl" if record_output_dir else "",
        "summary_json": f"{record_output_dir}/summary.json" if record_output_dir else "",
        "best_schedule_json": f"{record_output_dir}/best_schedule.json" if record_output_dir else "",
        "heartbeat_json": f"{record_output_dir}/heartbeat.json" if record_output_dir else "",
    }
    for field, value in (
        ("record_output_dir", record_output_dir),
        ("progress_dir", progress_dir),
        ("summary_json", summary_json),
        ("best_schedule_json", best_schedule_json),
        ("heartbeat_json", heartbeat_json),
    ):
        if not value:
            blockers.append(f"calibration_output_field_missing:{field}")
    for field, actual in (
        ("progress_dir", progress_dir),
        ("summary_json", summary_json),
        ("best_schedule_json", best_schedule_json),
        ("heartbeat_json", heartbeat_json),
    ):
        if actual and expected[field] and actual != expected[field]:
            blockers.append(f"calibration_output_expectation_mismatch:{field}:actual={actual}:expected={expected[field]}")
    return expected


def _calibration_warm_start_checks(
    row: Mapping[str, str],
    *,
    method_id: str,
    target_id: str,
    transfer_inputs: Sequence[str],
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    lock_value = str(row.get("warm_start_schedule_lock_json") or "").strip()
    key_value = str(row.get("warm_start_schedule_key") or "").strip()
    detail: dict[str, Any] = {
        "warm_start_schedule_lock_json": lock_value or None,
        "warm_start_schedule_key": key_value or None,
        "warm_start_schedule_exists": False,
        "warm_start_schedule_sandbox_visible": False,
        "warm_start_schedule_fields": [],
    }
    if not lock_value and not key_value:
        return detail
    expected_key = f"{method_id}::{target_id}"
    if not lock_value:
        blockers.append("warm_start_schedule_key_set_without_lock_json")
        return detail
    if key_value != expected_key:
        blockers.append(f"warm_start_schedule_key_mismatch:{key_value}:{expected_key}")
    lock_path = run_task._resolve_under_repo(lock_value, repo_root=repo_root)
    lock_exists = lock_path.exists()
    lock_visible = run_task._is_sandbox_visible(lock_value, transfer_input_files=transfer_inputs, repo_root=repo_root)
    detail["warm_start_schedule_exists"] = bool(lock_exists)
    detail["warm_start_schedule_sandbox_visible"] = bool(lock_visible)
    if not lock_exists:
        blockers.append(f"warm_start_schedule_lock_missing:{lock_value}")
        return detail
    if not lock_visible:
        blockers.append(f"warm_start_schedule_lock_not_transferred:{lock_value}")
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        entries = payload.get("method_target_schedules") if isinstance(payload, Mapping) else None
        if not isinstance(entries, Mapping):
            blockers.append("warm_start_schedule_lock_missing_method_target_schedules")
            return detail
        entry = entries.get(key_value)
        if not isinstance(entry, Mapping):
            blockers.append(f"warm_start_schedule_lock_missing_key:{key_value}")
            return detail
        schedule = entry.get("schedule")
        if not isinstance(schedule, Mapping):
            blockers.append(f"warm_start_schedule_lock_missing_schedule:{key_value}")
            return detail
        expected_fields = set(json.loads(str(row.get("search_space_fields_json") or "[]")))
        keys = {str(name) for name in schedule}
        detail["warm_start_schedule_fields"] = sorted(keys)
        if keys != expected_fields:
            blockers.append(f"warm_start_schedule_fields_mismatch:got={sorted(keys)}:expected={sorted(expected_fields)}")
        allowed = set(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.get(method_id, ()))
        extra = sorted(keys - allowed)
        if extra:
            blockers.append(f"warm_start_schedule_fields_not_allowed:{extra}")
        for name, value in schedule.items():
            parsed = _finite_float_from_row({str(name): str(value)}, str(name), blockers, positive=True)
            if str(name) in {"family_informed_spsa_eval_repeats", "family_informed_spsa_avg_last"} and parsed is not None:
                if not float(parsed).is_integer():
                    blockers.append(f"warm_start_schedule_integer_field_not_integral:{name}={value!r}")
    except Exception as exc:
        blockers.append(f"warm_start_schedule_lock_invalid:{type(exc).__name__}:{exc}")
    return detail


def _calibration_repair_scope_checks(row: Mapping[str, str], blockers: list[str]) -> dict[str, Any]:
    scope = str(row.get("repair_scope") or "").strip()
    if not scope:
        return {"repair_scope": None}
    detail: dict[str, Any] = {
        "repair_scope": scope,
        "table_label": row.get("table_label"),
        "hh_tableiii_regime": row.get("hh_tableiii_regime"),
        "n_ph_work": row.get("n_ph_work"),
        "n_ph_ref": row.get("n_ph_ref"),
        "phase3_adapt_max_depth": row.get("phase3_adapt_max_depth"),
        "primary_energy_metric": row.get("primary_energy_metric"),
        "same_cutoff_error_role": row.get("same_cutoff_error_role"),
        "calibration_usable_status_policy": row.get("calibration_usable_status_policy"),
        "quality_nonpassing_penalty": row.get("quality_nonpassing_penalty"),
    }
    if scope != HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
        blockers.append(f"calibration_repair_scope_unknown:{scope}")
        return detail
    method_id = str(row.get("method_id") or row.get("algorithm_id") or "").strip()
    target_id = str(row.get("target_id") or "").strip()
    if method_id not in HH_TABLEIII_REPAIR_METHOD_IDS:
        blockers.append(f"hh_tableiii_repair_method_not_allowed:{method_id}")
    if target_id not in HH_TABLEIII_REPAIR_TARGET_IDS:
        blockers.append(f"hh_tableiii_repair_target_not_allowed:{target_id}")
        return detail
    repair_target = hh_tableiii_repair_target_by_id(target_id)
    if str(row.get("table_label") or "").strip() != HH_TABLEIII_REPAIR_TABLE_LABEL:
        blockers.append(f"hh_tableiii_repair_table_label_mismatch:{row.get('table_label')}")
    if str(row.get("hh_tableiii_regime") or "").strip() != repair_target.hh_tableiii_regime:
        blockers.append(f"hh_tableiii_repair_regime_mismatch:{row.get('hh_tableiii_regime')}:{repair_target.hh_tableiii_regime}")
    n_ph_work = _positive_int_from_row(row, "n_ph_work", blockers, min_value=1)
    n_ph_ref = _positive_int_from_row(row, "n_ph_ref", blockers, min_value=1)
    depth_cap = _positive_int_from_row(row, "phase3_adapt_max_depth", blockers, min_value=1)
    if n_ph_work is not None and int(n_ph_work) != int(repair_target.n_ph_work):
        blockers.append(f"hh_tableiii_repair_n_ph_work_mismatch:{n_ph_work}:{repair_target.n_ph_work}")
    if n_ph_ref is not None and int(n_ph_ref) != int(repair_target.n_ph_ref):
        blockers.append(f"hh_tableiii_repair_n_ph_ref_mismatch:{n_ph_ref}:{repair_target.n_ph_ref}")
    if depth_cap is not None and str(row.get("run_class") or "") != "smoke" and int(depth_cap) != int(repair_target.adapt_max_depth):
        blockers.append(f"hh_tableiii_repair_depth_cap_mismatch:{depth_cap}:{repair_target.adapt_max_depth}")
    for field in ("same_cutoff_exact_gs_energy", "exact_reference_energy"):
        _finite_float_from_row(row, field, blockers)
    exact_ref_nph = _positive_int_from_row(row, "exact_reference_n_ph_max", blockers, min_value=1)
    if exact_ref_nph is not None and n_ph_ref is not None and int(exact_ref_nph) != int(n_ph_ref):
        blockers.append(f"hh_tableiii_repair_exact_reference_nph_mismatch:{exact_ref_nph}:{n_ph_ref}")
    for field in ("same_cutoff_reference_energy_key", "reference_cutoff_energy_key", "reference_energy_status"):
        if not str(row.get(field) or "").strip():
            blockers.append(f"hh_tableiii_repair_reference_field_missing:{field}")
    if str(row.get("primary_energy_metric") or "").strip() != HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC:
        blockers.append(f"hh_tableiii_repair_primary_metric_mismatch:{row.get('primary_energy_metric')}")
    if str(row.get("same_cutoff_error_role") or "").strip() != HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE:
        blockers.append(f"hh_tableiii_repair_same_cutoff_role_mismatch:{row.get('same_cutoff_error_role')}")
    if str(row.get("calibration_usable_status_policy") or "").strip() != HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY:
        blockers.append(f"hh_tableiii_repair_status_policy_mismatch:{row.get('calibration_usable_status_policy')}")
    penalty = _finite_float_from_row(row, "quality_nonpassing_penalty", blockers, nonnegative=True)
    if penalty is not None and abs(float(penalty) - float(HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY)) > 1e-12:
        blockers.append(f"hh_tableiii_repair_quality_penalty_mismatch:{penalty}:{HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY}")
    return detail


def build_paper_i_comparator_spsa_calibration_preflight_manifest(
    row: dict[str, str],
    *,
    record_id: str,
    records_path: Path,
    submit_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (  # noqa: WPS433
        HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC,
        HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY,
        HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE,
        HH_TABLEIII_REPAIR_TABLE_LABEL,
        HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY,
        PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
        config_sha256_for_path,
        hh_tableiii_repair_target_by_id,
        load_and_validate_config,
        target_by_id,
        validate_method_id,
    )

    contract = run_task.parse_submit_contract(submit_path)
    transfer_inputs = list(contract.get("transfer_input_files") or [])
    blockers: list[str] = []
    records_rel = run_task._safe_relative(run_task._resolve_under_repo(records_path, repo_root=repo_root), repo_root=repo_root)
    submit_records = str(contract.get("records_path") or contract.get("argument_records_path") or "").strip()
    if submit_records and submit_records != records_rel:
        blockers.append(f"records_path_mismatch:submit={submit_records}:preflight={records_rel}")
    if submit_records and not run_task._is_sandbox_visible(submit_records, transfer_input_files=transfer_inputs, repo_root=repo_root):
        blockers.append(f"submit_records_not_transferred:{submit_records}")
    queue_file = str(contract.get("queue_record_id_file") or "").strip()
    if queue_file:
        queue_path = run_task._resolve_under_repo(queue_file, repo_root=repo_root)
        if not queue_path.exists():
            blockers.append(f"queue_record_id_file_missing:{queue_file}")
        elif not run_task._is_sandbox_visible(queue_file, transfer_input_files=transfer_inputs, repo_root=repo_root):
            blockers.append(f"queue_record_id_file_not_transferred:{queue_file}")
    dependency_checks, dependency_blockers = _calibration_dependency_checks(
        contract=contract,
        transfer_inputs=transfer_inputs,
        repo_root=repo_root,
    )
    blockers.extend(dependency_blockers)

    config_value = _calibration_submit_config_path(contract, row)
    row_config_value = str(row.get("config_path") or "").strip()
    if row_config_value and config_value and config_value != row_config_value:
        blockers.append(f"config_path_mismatch:submit={config_value}:record={row_config_value}")
    config_path = run_task._resolve_under_repo(config_value or row_config_value, repo_root=repo_root)
    config_exists = config_path.exists()
    config_visible = bool(
        (config_value or row_config_value)
        and run_task._is_sandbox_visible(config_value or row_config_value, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    config_payload: dict[str, Any] | None = None
    if not config_exists:
        blockers.append(f"config_missing:{config_value or row_config_value}")
    if (config_value or row_config_value) and not config_visible:
        blockers.append(f"config_not_transferred:{config_value or row_config_value}")
    if config_exists:
        try:
            config_payload = load_and_validate_config(config_path)
        except Exception as exc:
            blockers.append(f"config_invalid:{type(exc).__name__}:{exc}")
    if config_exists and str(row.get("config_sha256") or "").strip():
        actual_hash = config_sha256_for_path(config_path)
        if str(row.get("config_sha256") or "").strip() != actual_hash:
            blockers.append(f"config_sha256_mismatch:record={row.get('config_sha256')}:actual={actual_hash}")

    method_id = str(row.get("method_id") or row.get("algorithm_id") or "").strip()
    target_id = str(row.get("target_id") or "").strip()
    case_ids: list[str] = []
    validation_detail: dict[str, Any] = {}
    try:
        method_id = validate_method_id(method_id)
        target = target_by_id(target_id)
        case_ids = list(target.case_ids)
        if config_payload is not None:
            from chtc.phase3_optuna.paper_i_comparator_spsa_calibration_runner import (  # noqa: WPS433
                validate_record as validate_comparator_spsa_calibration_record,
            )

            raw_validation = validate_comparator_spsa_calibration_record(
                row,
                config=config_payload,
                config_path=config_path,
                repo_root=repo_root,
            )
            validation_target = raw_validation.get("target")
            validation_detail = {
                "method_id": raw_validation.get("method_id"),
                "target_id": getattr(validation_target, "target_id", target_id),
                "case_ids": list(raw_validation.get("case_ids") or ()),
                "family": raw_validation.get("family"),
            }
    except Exception as exc:
        blockers.append(f"calibration_record_invalid:{type(exc).__name__}:{exc}")
    if str(row.get("record_id") or "").strip() != str(record_id):
        blockers.append(f"record_id_mismatch:row={row.get('record_id')}:queue={record_id}")
    if str(row.get("profile_id") or "").strip() != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
        blockers.append(f"profile_id_mismatch:{row.get('profile_id')}")
    if str(row.get("algorithm_id") or "").strip() != method_id:
        blockers.append(f"algorithm_method_mismatch:{row.get('algorithm_id')}:{method_id}")

    run_class = str(row.get("run_class") or "").strip()
    config_mode = str(row.get("config_mode") or (config_payload or {}).get("mode") or "").strip().lower()
    submit_name = submit_path.name.lower()
    full_queue = "full" in submit_name or run_class == "calibration_candidate_not_table_evidence" or config_mode == "full"
    if full_queue:
        approved = bool((config_payload or {}).get("approved_for_full_generation"))
        if config_mode != "full" or not approved:
            blockers.append("full_queue_requires_approved_full_config")
        if not str((config_payload or {}).get("approved_by") or "").strip() or not str((config_payload or {}).get("approved_at") or "").strip():
            blockers.append("full_queue_requires_approval_metadata")
    else:
        if run_class != "smoke":
            blockers.append(f"smoke_queue_run_class_mismatch:{run_class}")
    _positive_int_from_row(row, "n_jobs", blockers, min_value=1)
    if str(row.get("n_jobs") or "").strip() != "1":
        blockers.append(f"calibration_n_jobs_must_be_1:{row.get('n_jobs')}")
    _positive_int_from_row(row, "n_trials", blockers, min_value=1)
    _positive_int_from_row(row, "method_maxiter_budget", blockers, min_value=1)
    _finite_float_from_row(row, "failure_penalty", blockers, positive=True)
    _finite_float_from_row(row, "target_abs_delta_e", blockers, positive=True)
    _finite_float_from_row(row, "resource_tiebreak_weight", blockers, nonnegative=True)
    repair_detail = _calibration_repair_scope_checks(row, blockers)
    warm_start_detail = _calibration_warm_start_checks(
        row,
        method_id=method_id,
        target_id=target_id,
        transfer_inputs=transfer_inputs,
        repo_root=repo_root,
        blockers=blockers,
    )

    plan_path_value = str(row.get("plan_path") or "").strip()
    plan_path = run_task._resolve_under_repo(plan_path_value, repo_root=repo_root) if plan_path_value else repo_root
    plan_visible = bool(
        plan_path_value and run_task._is_sandbox_visible(plan_path_value, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    if not plan_path_value or not plan_path.exists():
        blockers.append(f"plan_path_missing:{plan_path_value}")
    if plan_path_value and not plan_visible:
        blockers.append(f"plan_path_not_transferred:{plan_path_value}")
    output_expectations = _calibration_output_expectations(row, blockers)
    tokens = _submit_argument_tokens(contract)
    submit_out_root = ""
    expected_out_root = ""
    if tokens:
        if len(tokens) < 3 or tokens[0] != "$(record_id)":
            blockers.append(f"calibration_submit_arguments_invalid:{contract.get('arguments')}")
        else:
            submit_out_root = tokens[2].replace("$(record_id)", str(record_id))
            queue_root = str(row.get("queue_output_root") or "").strip()
            expected_out_root = f"{queue_root.rstrip('/')}/{str(row.get('record_output_dir') or '').strip()}" if queue_root else ""
            if not expected_out_root:
                blockers.append("calibration_queue_output_root_missing")
            elif submit_out_root != expected_out_root:
                blockers.append(f"calibration_output_root_mismatch:submit={submit_out_root}:expected={expected_out_root}")

    return {
        "schema": "paper_i_comparator_spsa_calibration_chtc_preflight_manifest_v1",
        "status": "fail" if blockers else "pass",
        "ok": not blockers,
        "record_id": record_id,
        "profile_id": row.get("profile_id"),
        "evidence_role": "calibration_only_not_manuscript_table_evidence",
        "table_evidence_status": "not_table_evidence",
        "records_path": records_rel,
        "submit_contract": contract,
        "transfer_input_files": transfer_inputs,
        "blocking_reasons": blockers,
        "dependency_checks": dependency_checks,
        "source_artifacts": [
            {
                "field": "config_path",
                "value": config_value or row_config_value,
                "resolved_path": str(config_path),
                "exists": bool(config_exists),
                "sandbox_visible": bool(config_visible),
                "sha256": config_sha256_for_path(config_path) if config_exists and config_path.is_file() else None,
            },
            {
                "field": "plan_path",
                "value": plan_path_value,
                "resolved_path": str(plan_path),
                "exists": bool(plan_path.exists()) if plan_path_value else False,
                "sandbox_visible": bool(plan_visible),
            },
            {
                "field": "warm_start_schedule_lock_json",
                "value": warm_start_detail.get("warm_start_schedule_lock_json"),
                "resolved_path": None
                if not warm_start_detail.get("warm_start_schedule_lock_json")
                else str(run_task._resolve_under_repo(str(warm_start_detail["warm_start_schedule_lock_json"]), repo_root=repo_root)),
                "exists": bool(warm_start_detail.get("warm_start_schedule_exists")),
                "sandbox_visible": bool(warm_start_detail.get("warm_start_schedule_sandbox_visible")),
            },
        ],
        "calibration_record": {
            "record_schema": row.get("record_schema"),
            "run_class": run_class,
            "method_id": method_id,
            "algorithm_id": row.get("algorithm_id"),
            "target_id": target_id,
            "case_ids": case_ids,
            "config_mode": config_mode,
            "n_trials": row.get("n_trials"),
            "n_jobs": row.get("n_jobs"),
            "method_maxiter_budget": row.get("method_maxiter_budget"),
            "allowed_schedule_fields": list(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.get(method_id, ())),
            "validation_detail": validation_detail,
            **repair_detail,
            **warm_start_detail,
        },
        "submit_output_root": {
            "argument_value": submit_out_root,
            "expected_value": expected_out_root,
        },
        "current_best_expectation": {
            "progress_current_best_json": output_expectations["current_best_json"],
            "progress_trial_events_jsonl": output_expectations["trial_events_jsonl"],
            "summary_json": output_expectations["summary_json"],
            "best_schedule_json": output_expectations["best_schedule_json"],
            "heartbeat_json": output_expectations["heartbeat_json"],
        },
    }


def _hh_u8_comparator_spsa_submit_config_path(contract: Mapping[str, Any], row: Mapping[str, str]) -> str:
    env = contract.get("environment")
    if isinstance(env, Mapping):
        value = str(env.get("PAPER_I_HH_U8_COMPARATOR_SPSA_CONFIG_PATH") or "").strip()
        if value:
            return value
    tokens = _submit_argument_tokens(contract)
    if "--config" in tokens:
        idx = tokens.index("--config")
        if idx + 1 < len(tokens):
            return tokens[idx + 1]
    if len(tokens) >= 4 and tokens[0] == "$(record_id)":
        return tokens[3]
    return str(row.get("config_path") or "").strip()


def _hh_u8_comparator_spsa_dependency_checks(
    *,
    contract: Mapping[str, Any],
    transfer_inputs: Sequence[str],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    requirements_path = repo_root / "chtc" / "phase3_optuna" / "requirements-chtc.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8") if requirements_path.exists() else ""
    for package in ("optuna", "qiskit-algorithms"):
        declared = any(line.strip().split("==", 1)[0] == package for line in requirements_text.splitlines())
        checks.append(
            {
                "check": "chtc_requirements_declares_python_package",
                "package": package,
                "requirements_path": run_task._safe_relative(requirements_path, repo_root=repo_root),
                "ok": bool(declared),
            }
        )
        if not declared:
            blockers.append(f"dependency_not_declared_in_requirements:{package}")
    executable = str(contract.get("executable") or "").strip()
    executable_path = run_task._resolve_under_repo(executable, repo_root=repo_root) if executable else repo_root
    executable_exists = bool(executable and executable_path.exists())
    executable_visible = bool(
        executable and run_task._is_sandbox_visible(executable, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    checks.append(
        {
            "check": "submit_executable_visible",
            "executable": executable,
            "exists": executable_exists,
            "sandbox_visible": executable_visible,
            "ok": executable_exists and executable_visible,
        }
    )
    if not executable_exists:
        blockers.append(f"submit_executable_missing:{executable}")
    if executable and not executable_visible:
        blockers.append(f"submit_executable_not_transferred:{executable}")
    shell_wrapper = repo_root / "chtc" / "phase3_optuna" / "run_paper_i_hh_u8_comparator_spsa_task.sh"
    apptainer_wrapper = repo_root / "chtc" / "phase3_optuna" / "run_paper_i_hh_u8_comparator_spsa_task_apptainer.sh"
    shell_text = shell_wrapper.read_text(encoding="utf-8") if shell_wrapper.exists() else ""
    apptainer_text = apptainer_wrapper.read_text(encoding="utf-8") if apptainer_wrapper.exists() else ""
    wrapper_runtime_ok = all(token in shell_text for token in ("optuna", "qiskit_algorithms", "paper_i_hh_u8_comparator_spsa_runner"))
    checks.append(
        {
            "check": "wrapper_runtime_dependency_imports_present",
            "wrapper": run_task._safe_relative(shell_wrapper, repo_root=repo_root),
            "ok": bool(wrapper_runtime_ok),
        }
    )
    if not wrapper_runtime_ok:
        blockers.append("wrapper_dependency_check_missing:optuna_or_qiskit_algorithms_or_u8_runner")
    if executable.endswith("_apptainer.sh"):
        apptainer_ok = "run_paper_i_hh_u8_comparator_spsa_task.sh" in apptainer_text
        checks.append(
            {
                "check": "apptainer_wrapper_invokes_shell_wrapper",
                "wrapper": run_task._safe_relative(apptainer_wrapper, repo_root=repo_root),
                "ok": bool(apptainer_ok),
            }
        )
        if not apptainer_ok:
            blockers.append("apptainer_wrapper_does_not_invoke_hh_u8_comparator_spsa_shell_wrapper")
    return checks, blockers


def build_paper_i_hh_u8_comparator_spsa_preflight_manifest(
    row: dict[str, str],
    *,
    record_id: str,
    records_path: Path,
    submit_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    from pipelines.exact_bench.paper_i_hh_u8_comparator_spsa_optuna import (  # noqa: WPS433
        PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
        PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE,
        config_sha256_for_path as u8_config_sha256_for_path,
        load_and_validate_config as load_and_validate_u8_config,
        target_by_id as hh_u8_target_by_id,
        validate_method_id as validate_hh_u8_method_id,
    )

    contract = run_task.parse_submit_contract(submit_path)
    transfer_inputs = list(contract.get("transfer_input_files") or [])
    blockers: list[str] = []
    records_rel = run_task._safe_relative(run_task._resolve_under_repo(records_path, repo_root=repo_root), repo_root=repo_root)
    submit_records = str(contract.get("records_path") or contract.get("argument_records_path") or "").strip()
    if submit_records and submit_records != records_rel:
        blockers.append(f"records_path_mismatch:submit={submit_records}:preflight={records_rel}")
    if submit_records and not run_task._is_sandbox_visible(submit_records, transfer_input_files=transfer_inputs, repo_root=repo_root):
        blockers.append(f"submit_records_not_transferred:{submit_records}")
    queue_file = str(contract.get("queue_record_id_file") or "").strip()
    if queue_file:
        queue_path = run_task._resolve_under_repo(queue_file, repo_root=repo_root)
        if not queue_path.exists():
            blockers.append(f"queue_record_id_file_missing:{queue_file}")
        elif not run_task._is_sandbox_visible(queue_file, transfer_input_files=transfer_inputs, repo_root=repo_root):
            blockers.append(f"queue_record_id_file_not_transferred:{queue_file}")

    dependency_checks, dependency_blockers = _hh_u8_comparator_spsa_dependency_checks(
        contract=contract,
        transfer_inputs=transfer_inputs,
        repo_root=repo_root,
    )
    blockers.extend(dependency_blockers)

    config_value = _hh_u8_comparator_spsa_submit_config_path(contract, row)
    row_config_value = str(row.get("config_path") or "").strip()
    if row_config_value and config_value and config_value != row_config_value:
        blockers.append(f"config_path_mismatch:submit={config_value}:record={row_config_value}")
    config_path = run_task._resolve_under_repo(config_value or row_config_value, repo_root=repo_root)
    config_exists = config_path.exists()
    config_visible = bool(
        (config_value or row_config_value)
        and run_task._is_sandbox_visible(config_value or row_config_value, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    config_payload: dict[str, Any] | None = None
    if not config_exists:
        blockers.append(f"config_missing:{config_value or row_config_value}")
    if (config_value or row_config_value) and not config_visible:
        blockers.append(f"config_not_transferred:{config_value or row_config_value}")
    if config_exists:
        try:
            config_payload = load_and_validate_u8_config(config_path)
        except Exception as exc:
            blockers.append(f"config_invalid:{type(exc).__name__}:{exc}")
    if config_exists and str(row.get("config_sha256") or "").strip():
        actual_hash = u8_config_sha256_for_path(config_path)
        if str(row.get("config_sha256") or "").strip() != actual_hash:
            blockers.append(f"config_sha256_mismatch:record={row.get('config_sha256')}:actual={actual_hash}")

    method_id = str(row.get("method_id") or row.get("algorithm_id") or "").strip()
    target_id = str(row.get("target_id") or "").strip()
    case_ids: list[str] = []
    validation_detail: dict[str, Any] = {}
    try:
        method_id = validate_hh_u8_method_id(method_id)
        target = hh_u8_target_by_id(target_id)
        case_ids = list(target.case_ids)
        if config_payload is not None:
            from chtc.phase3_optuna.paper_i_hh_u8_comparator_spsa_runner import (  # noqa: WPS433
                validate_record as validate_hh_u8_comparator_spsa_record,
            )

            raw_validation = validate_hh_u8_comparator_spsa_record(row, config=config_payload, config_path=config_path)
            validation_target = raw_validation.get("target")
            validation_detail = {
                "method_id": raw_validation.get("method_id"),
                "target_id": getattr(validation_target, "target_id", target_id),
                "case_ids": list(raw_validation.get("case_ids") or ()),
                "family": raw_validation.get("family"),
            }
    except Exception as exc:
        blockers.append(f"hh_u8_comparator_spsa_record_invalid:{type(exc).__name__}:{exc}")
    if str(row.get("record_id") or "").strip() != str(record_id):
        blockers.append(f"record_id_mismatch:row={row.get('record_id')}:queue={record_id}")
    if str(row.get("profile_id") or "").strip() != PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID:
        blockers.append(f"profile_id_mismatch:{row.get('profile_id')}")
    if str(row.get("algorithm_id") or "").strip() != method_id:
        blockers.append(f"algorithm_method_mismatch:{row.get('algorithm_id')}:{method_id}")
    if str(row.get("suite_profile") or "").strip() != PAPER_I_HH_U8_COMPARATOR_SPSA_SUITE_PROFILE:
        blockers.append(f"hh_u8_suite_profile_mismatch:{row.get('suite_profile')}")
    if str(row.get("optimizer_profile") or "").strip():
        blockers.append(f"hh_u8_optimizer_profile_must_be_blank:{row.get('optimizer_profile')}")

    run_class = str(row.get("run_class") or "").strip()
    config_mode = str(row.get("config_mode") or (config_payload or {}).get("mode") or "").strip().lower()
    submit_name = submit_path.name.lower()
    full_queue = "full" in submit_name or run_class == "calibration_candidate_not_table_evidence" or config_mode == "full"
    if full_queue:
        approved = bool((config_payload or {}).get("approved_for_full_generation"))
        if config_mode != "full" or not approved:
            blockers.append("full_queue_requires_approved_full_config")
        if not str((config_payload or {}).get("approved_by") or "").strip() or not str((config_payload or {}).get("approved_at") or "").strip():
            blockers.append("full_queue_requires_approval_metadata")
    else:
        if run_class != "smoke":
            blockers.append(f"smoke_queue_run_class_mismatch:{run_class}")
    _positive_int_from_row(row, "n_jobs", blockers, min_value=1)
    if str(row.get("n_jobs") or "").strip() != "1":
        blockers.append(f"calibration_n_jobs_must_be_1:{row.get('n_jobs')}")
    _positive_int_from_row(row, "n_trials", blockers, min_value=1)
    _positive_int_from_row(row, "method_maxiter_budget", blockers, min_value=1)
    _positive_int_from_row(row, "n_ph_work", blockers, min_value=1)
    _positive_int_from_row(row, "n_ph_ref", blockers, min_value=1)
    _positive_int_from_row(row, "exact_reference_n_ph_max", blockers, min_value=1)
    _finite_float_from_row(row, "failure_penalty", blockers, positive=True)
    _finite_float_from_row(row, "target_abs_delta_e", blockers, positive=True)
    _finite_float_from_row(row, "resource_tiebreak_weight", blockers, nonnegative=True)
    _finite_float_from_row(row, "same_cutoff_exact_gs_energy", blockers)
    _finite_float_from_row(row, "exact_reference_energy", blockers)
    if str(row.get("primary_energy_metric") or "").strip() != "higher_cutoff_reference_abs_delta_e":
        blockers.append("hh_u8_primary_metric_mismatch")
    if str(row.get("same_cutoff_error_role") or "").strip() != "diagnostic_only":
        blockers.append("hh_u8_same_cutoff_role_mismatch")

    plan_path_value = str(row.get("plan_path") or "").strip()
    plan_path = run_task._resolve_under_repo(plan_path_value, repo_root=repo_root) if plan_path_value else repo_root
    plan_visible = bool(
        plan_path_value and run_task._is_sandbox_visible(plan_path_value, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    if not plan_path_value or not plan_path.exists():
        blockers.append(f"plan_path_missing:{plan_path_value}")
    if plan_path_value and not plan_visible:
        blockers.append(f"plan_path_not_transferred:{plan_path_value}")
    output_expectations = _calibration_output_expectations(row, blockers)
    tokens = _submit_argument_tokens(contract)
    submit_out_root = ""
    expected_out_root = ""
    if tokens:
        if len(tokens) < 3 or tokens[0] != "$(record_id)":
            blockers.append(f"calibration_submit_arguments_invalid:{contract.get('arguments')}")
        else:
            submit_out_root = tokens[2].replace("$(record_id)", str(record_id))
            queue_root = str(row.get("queue_output_root") or "").strip()
            expected_out_root = f"{queue_root.rstrip('/')}/{str(row.get('record_output_dir') or '').strip()}" if queue_root else ""
            if not expected_out_root:
                blockers.append("calibration_queue_output_root_missing")
            elif submit_out_root != expected_out_root:
                blockers.append(f"calibration_output_root_mismatch:submit={submit_out_root}:expected={expected_out_root}")

    return {
        "schema": "paper_i_hh_u8_comparator_spsa_chtc_preflight_manifest_v1",
        "status": "fail" if blockers else "pass",
        "ok": not blockers,
        "record_id": record_id,
        "profile_id": row.get("profile_id"),
        "evidence_role": "calibration_only_not_manuscript_table_evidence",
        "table_evidence_status": "not_table_evidence",
        "records_path": records_rel,
        "submit_contract": contract,
        "transfer_input_files": transfer_inputs,
        "blocking_reasons": blockers,
        "dependency_checks": dependency_checks,
        "source_artifacts": [
            {
                "field": "config_path",
                "value": config_value or row_config_value,
                "resolved_path": str(config_path),
                "exists": bool(config_exists),
                "sandbox_visible": bool(config_visible),
                "sha256": u8_config_sha256_for_path(config_path) if config_exists and config_path.is_file() else None,
            },
            {
                "field": "plan_path",
                "value": plan_path_value,
                "resolved_path": str(plan_path),
                "exists": bool(plan_path.exists()) if plan_path_value else False,
                "sandbox_visible": bool(plan_visible),
            },
        ],
        "calibration_record": {
            "record_schema": row.get("record_schema"),
            "run_class": run_class,
            "method_id": method_id,
            "algorithm_id": row.get("algorithm_id"),
            "target_id": target_id,
            "case_ids": case_ids,
            "config_mode": config_mode,
            "n_trials": row.get("n_trials"),
            "n_jobs": row.get("n_jobs"),
            "method_maxiter_budget": row.get("method_maxiter_budget"),
            "suite_profile": row.get("suite_profile"),
            "optimizer_profile": row.get("optimizer_profile"),
            "n_ph_work": row.get("n_ph_work"),
            "n_ph_ref": row.get("n_ph_ref"),
            "primary_energy_metric": row.get("primary_energy_metric"),
            "same_cutoff_error_role": row.get("same_cutoff_error_role"),
            "allowed_schedule_fields": list(PAPER_I_HH_U8_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.get(method_id, ())),
            "validation_detail": validation_detail,
        },
        "submit_output_root": {
            "argument_value": submit_out_root,
            "expected_value": expected_out_root,
        },
        "current_best_expectation": {
            "progress_current_best_json": output_expectations["current_best_json"],
            "progress_trial_events_jsonl": output_expectations["trial_events_jsonl"],
            "summary_json": output_expectations["summary_json"],
            "best_schedule_json": output_expectations["best_schedule_json"],
            "heartbeat_json": output_expectations["heartbeat_json"],
        },
    }


def _shared_spsa_dependency_checks(
    *,
    contract: Mapping[str, Any],
    transfer_inputs: Sequence[str],
    repo_root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    blockers: list[str] = []
    checks: list[dict[str, Any]] = []
    requirements_path = repo_root / "chtc" / "phase3_optuna" / "requirements-chtc.txt"
    requirements_text = requirements_path.read_text(encoding="utf-8") if requirements_path.exists() else ""
    for package in ("optuna", "qiskit-algorithms"):
        declared = any(line.strip().split("==", 1)[0] == package for line in requirements_text.splitlines())
        checks.append(
            {
                "check": "chtc_requirements_declares_python_package",
                "package": package,
                "requirements_path": run_task._safe_relative(requirements_path, repo_root=repo_root),
                "ok": bool(declared),
            }
        )
        if not declared:
            blockers.append(f"dependency_not_declared_in_requirements:{package}")
    executable = str(contract.get("executable") or "").strip()
    executable_path = run_task._resolve_under_repo(executable, repo_root=repo_root) if executable else repo_root
    executable_exists = bool(executable and executable_path.exists())
    executable_visible = bool(
        executable and run_task._is_sandbox_visible(executable, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    checks.append(
        {
            "check": "submit_executable_visible",
            "executable": executable,
            "exists": executable_exists,
            "sandbox_visible": executable_visible,
            "ok": executable_exists and executable_visible,
        }
    )
    if not executable_exists:
        blockers.append(f"submit_executable_missing:{executable}")
    if executable and not executable_visible:
        blockers.append(f"submit_executable_not_transferred:{executable}")
    shell_wrapper = repo_root / "chtc" / "phase3_optuna" / "run_paper_i_hh_shared_spsa_calibration_task.sh"
    apptainer_wrapper = repo_root / "chtc" / "phase3_optuna" / "run_paper_i_hh_shared_spsa_calibration_task_apptainer.sh"
    shell_text = shell_wrapper.read_text(encoding="utf-8") if shell_wrapper.exists() else ""
    apptainer_text = apptainer_wrapper.read_text(encoding="utf-8") if apptainer_wrapper.exists() else ""
    wrapper_runtime_ok = all(token in shell_text for token in ("optuna", "qiskit_algorithms", "paper_i_hh_shared_spsa_calibration_runner"))
    checks.append(
        {
            "check": "wrapper_runtime_dependency_imports_present",
            "wrapper": run_task._safe_relative(shell_wrapper, repo_root=repo_root),
            "ok": bool(wrapper_runtime_ok),
        }
    )
    if not wrapper_runtime_ok:
        blockers.append("shared_spsa_wrapper_dependency_check_missing:optuna_or_qiskit_algorithms")
    if executable.endswith("_apptainer.sh"):
        apptainer_ok = "run_paper_i_hh_shared_spsa_calibration_task.sh" in apptainer_text
        checks.append(
            {
                "check": "apptainer_wrapper_invokes_shell_wrapper",
                "wrapper": run_task._safe_relative(apptainer_wrapper, repo_root=repo_root),
                "ok": bool(apptainer_ok),
            }
        )
        if not apptainer_ok:
            blockers.append("shared_spsa_apptainer_wrapper_does_not_invoke_shell_wrapper")
    return checks, blockers


def _json_field(row: Mapping[str, str], field: str, blockers: list[str]) -> Any:
    raw = str(row.get(field) or "").strip()
    if not raw:
        blockers.append(f"shared_spsa_json_field_missing:{field}")
        return None
    try:
        return json.loads(raw)
    except Exception as exc:
        blockers.append(f"shared_spsa_json_field_invalid:{field}:{type(exc).__name__}:{exc}")
        return None


def _shared_spsa_source_artifact(
    row: Mapping[str, str],
    field: str,
    *,
    transfer_inputs: Sequence[str],
    repo_root: Path,
    blockers: list[str],
) -> dict[str, Any]:
    value = str(row.get(field) or "").strip()
    resolved = run_task._resolve_under_repo(value, repo_root=repo_root) if value else repo_root
    exists = bool(value and resolved.exists())
    visible = bool(value and run_task._is_sandbox_visible(value, transfer_input_files=transfer_inputs, repo_root=repo_root))
    if not value or not exists:
        blockers.append(f"shared_spsa_source_records_missing:{field}:{value}")
    if value and not visible:
        blockers.append(f"shared_spsa_source_records_not_transferred:{field}:{value}")
    return {
        "field": field,
        "value": value,
        "resolved_path": str(resolved),
        "exists": exists,
        "sandbox_visible": visible,
    }


def build_paper_i_hh_shared_spsa_preflight_manifest(
    row: dict[str, str],
    *,
    record_id: str,
    records_path: Path,
    submit_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    contract = run_task.parse_submit_contract(submit_path)
    transfer_inputs = list(contract.get("transfer_input_files") or [])
    blockers: list[str] = []
    records_rel = run_task._safe_relative(run_task._resolve_under_repo(records_path, repo_root=repo_root), repo_root=repo_root)
    submit_records = str(contract.get("records_path") or contract.get("argument_records_path") or "").strip()
    if submit_records and submit_records != records_rel:
        blockers.append(f"records_path_mismatch:submit={submit_records}:preflight={records_rel}")
    if submit_records and not run_task._is_sandbox_visible(submit_records, transfer_input_files=transfer_inputs, repo_root=repo_root):
        blockers.append(f"submit_records_not_transferred:{submit_records}")
    queue_file = str(contract.get("queue_record_id_file") or "").strip()
    if queue_file:
        queue_path = run_task._resolve_under_repo(queue_file, repo_root=repo_root)
        if not queue_path.exists():
            blockers.append(f"queue_record_id_file_missing:{queue_file}")
        elif not run_task._is_sandbox_visible(queue_file, transfer_input_files=transfer_inputs, repo_root=repo_root):
            blockers.append(f"queue_record_id_file_not_transferred:{queue_file}")

    dependency_checks, dependency_blockers = _shared_spsa_dependency_checks(
        contract=contract,
        transfer_inputs=transfer_inputs,
        repo_root=repo_root,
    )
    blockers.extend(dependency_blockers)

    config_value = _calibration_submit_config_path(contract, row)
    row_config_value = str(row.get("config_path") or "").strip()
    if row_config_value and config_value and config_value != row_config_value:
        blockers.append(f"config_path_mismatch:submit={config_value}:record={row_config_value}")
    config_path = run_task._resolve_under_repo(config_value or row_config_value, repo_root=repo_root)
    config_exists = config_path.exists()
    config_visible = bool(
        (config_value or row_config_value)
        and run_task._is_sandbox_visible(config_value or row_config_value, transfer_input_files=transfer_inputs, repo_root=repo_root)
    )
    config_payload: dict[str, Any] | None = None
    if not config_exists:
        blockers.append(f"config_missing:{config_value or row_config_value}")
    if (config_value or row_config_value) and not config_visible:
        blockers.append(f"config_not_transferred:{config_value or row_config_value}")
    if config_exists:
        try:
            config_payload = load_and_validate_shared_spsa_config(config_path)
        except Exception as exc:
            blockers.append(f"config_invalid:{type(exc).__name__}:{exc}")
    if config_exists and str(row.get("config_sha256") or "").strip():
        actual_hash = shared_spsa_config_sha256_for_path(config_path)
        if str(row.get("config_sha256") or "").strip() != actual_hash:
            blockers.append(f"config_sha256_mismatch:record={row.get('config_sha256')}:actual={actual_hash}")

    if str(row.get("record_id") or "").strip() != str(record_id):
        blockers.append(f"record_id_mismatch:row={row.get('record_id')}:queue={record_id}")
    if str(row.get("profile_id") or "").strip() != PAPER_I_HH_SHARED_SPSA_CALIBRATION_PROFILE_ID:
        blockers.append(f"profile_id_mismatch:{row.get('profile_id')}")
    if str(row.get("run_class") or "").strip() != "diagnostic":
        blockers.append(f"shared_spsa_run_class_must_be_diagnostic:{row.get('run_class')}")

    method_key = str(row.get("method_key") or "").strip()
    engine_key = str(row.get("engine_key") or "native_forced").strip()
    try:
        method = shared_spsa_method_by_key_or_id(method_key)
        if str(row.get("algorithm_id") or "").strip() != method.algorithm_id:
            blockers.append(f"algorithm_method_mismatch:{row.get('algorithm_id')}:{method.algorithm_id}")
        if config_payload is not None and method.method_key not in set(str(item) for item in config_payload.get("methods", ())):
            blockers.append(f"shared_spsa_method_not_enabled_by_config:{method.method_key}")
    except Exception as exc:
        blockers.append(f"shared_spsa_method_invalid:{type(exc).__name__}:{exc}")
    try:
        normalized_engine_key = normalize_shared_spsa_refit_engine_key(engine_key)
        if config_payload is not None and normalized_engine_key not in set(str(item) for item in config_payload.get("spsa_refit_engines", ())):
            blockers.append(f"shared_spsa_engine_not_enabled_by_config:{normalized_engine_key}")
    except Exception as exc:
        normalized_engine_key = engine_key
        blockers.append(f"shared_spsa_engine_invalid:{type(exc).__name__}:{exc}")

    for field in ("n_trials", "maxiter", "max_depth", "case_parallelism", "per_case_cpus", "spsa_seed", "spsa_eval_repeats"):
        observed = _positive_int_from_row(row, field, blockers, min_value=0 if field == "spsa_seed" else 1)
        if config_payload is not None and observed is not None and int(observed) != int(config_payload[field]):
            blockers.append(f"shared_spsa_config_row_mismatch:{field}:row={observed}:config={config_payload[field]}")
    _positive_int_from_row(row, "spsa_avg_last", blockers, min_value=0)
    _finite_float_from_row(row, "target_abs_delta_e", blockers, positive=True)
    _finite_float_from_row(row, "resource_tiebreak_weight", blockers, nonnegative=True)
    if config_payload is not None:
        regimes = _json_field(row, "regimes_json", blockers)
        if regimes is not None and list(regimes) != list(config_payload["regimes"]):
            blockers.append(f"shared_spsa_regimes_mismatch:row={regimes}:config={config_payload['regimes']}")
        worker_overrides = _json_field(row, "snake_runtime_worker_overrides_json", blockers)
        if worker_overrides is not None and dict(worker_overrides) != dict(config_payload.get("snake_runtime_worker_overrides") or {}):
            blockers.append(
                "shared_spsa_worker_overrides_mismatch:"
                f"row={worker_overrides}:config={config_payload.get('snake_runtime_worker_overrides')}"
            )

    source_artifacts = [
        _shared_spsa_source_artifact(
            row,
            "append_geo_source_records",
            transfer_inputs=transfer_inputs,
            repo_root=repo_root,
            blockers=blockers,
        ),
        _shared_spsa_source_artifact(
            row,
            "snake_source_records",
            transfer_inputs=transfer_inputs,
            repo_root=repo_root,
            blockers=blockers,
        ),
    ]

    record_output_dir = str(row.get("record_output_dir") or "").strip()
    expected_outputs = {
        "progress_dir": f"{record_output_dir}/progress" if record_output_dir else "",
        "summary_json": f"{record_output_dir}/summary.json" if record_output_dir else "",
        "best_schedule_json": f"{record_output_dir}/best_schedule.json" if record_output_dir else "",
        "current_best_json": f"{record_output_dir}/progress/current_best.json" if record_output_dir else "",
        "heartbeat_json": f"{record_output_dir}/progress/heartbeat.json" if record_output_dir else "",
    }
    for field, expected in expected_outputs.items():
        actual = str(row.get(field) or "").strip()
        if not actual:
            blockers.append(f"shared_spsa_output_field_missing:{field}")
        elif actual != expected:
            blockers.append(f"shared_spsa_output_expectation_mismatch:{field}:actual={actual}:expected={expected}")

    tokens = _submit_argument_tokens(contract)
    submit_out_root = ""
    if tokens:
        if len(tokens) < 4 or tokens[0] != "$(record_id)":
            blockers.append(f"shared_spsa_submit_arguments_invalid:{contract.get('arguments')}")
        else:
            submit_out_root = tokens[2].replace("$(record_id)", str(record_id))
            if submit_out_root != record_output_dir:
                blockers.append(f"shared_spsa_output_root_mismatch:submit={submit_out_root}:expected={record_output_dir}")

    return {
        "schema": "paper_i_hh_shared_spsa_calibration_chtc_preflight_manifest_v1",
        "status": "fail" if blockers else "pass",
        "ok": not blockers,
        "record_id": record_id,
        "profile_id": row.get("profile_id"),
        "evidence_role": "diagnostic_calibration_only_not_manuscript_table_evidence",
        "table_evidence_status": "not_table_evidence",
        "records_path": records_rel,
        "submit_contract": contract,
        "transfer_input_files": transfer_inputs,
        "blocking_reasons": blockers,
        "dependency_checks": dependency_checks,
        "source_artifacts": [
            {
                "field": "config_path",
                "value": config_value or row_config_value,
                "resolved_path": str(config_path),
                "exists": bool(config_exists),
                "sandbox_visible": bool(config_visible),
                "sha256": shared_spsa_config_sha256_for_path(config_path) if config_exists and config_path.is_file() else None,
            },
            *source_artifacts,
        ],
        "calibration_record": {
            "record_schema": row.get("record_schema"),
            "run_class": row.get("run_class"),
            "method_key": method_key,
            "algorithm_id": row.get("algorithm_id"),
            "engine_key": normalized_engine_key,
            "engine_label": row.get("engine_label"),
            "spsa_refit_engine": row.get("spsa_refit_engine"),
            "config_mode": row.get("config_mode"),
            "n_trials": row.get("n_trials"),
            "maxiter": row.get("maxiter"),
            "max_depth": row.get("max_depth"),
            "case_parallelism": row.get("case_parallelism"),
            "per_case_cpus": row.get("per_case_cpus"),
            "regimes_json": row.get("regimes_json"),
            "snake_runtime_worker_overrides_json": row.get("snake_runtime_worker_overrides_json"),
        },
        "submit_output_root": {
            "argument_value": submit_out_root,
            "expected_value": record_output_dir,
        },
        "current_best_expectation": expected_outputs,
    }


_GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS = {
    "static_full_meta_append_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_adapt_vqe",
    "static_pos_geo_adapt_vqe",
}
_PHASE3_STATIC_ADAPT_ALGORITHM_IDS = {"static_family_native_adapt_phase3", "static_append_only_adapt_phase3"}
_HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_RUN_CLASS = "paper_i_hh_tableiii_snake_shot_proxy_repair"
_HH_TABLEIII_SNAKE_EXACT_PREFIX_RECOVERY_RUN_CLASS = "paper_i_hh_tableiii_snake_exact_prefix_recovery"
_HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_TABLE_LABEL = "tab:hh_first_plateau_prefix_costs"
_HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_PRIMARY_ENERGY_METRIC = (
    "same_cutoff_plateau_prefix_abs_delta_e_with_higher_cutoff_diagnostic"
)
_HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_SAME_CUTOFF_ERROR_ROLE = "primary_tableiii_metric"
_HH_U8_SNAKE_SOURCE_LOCKED_REPLAY_RUN_CLASS = "paper_i_hh_u8_snake_source_locked_replay"
_HH_U8_SNAKE_SOURCE_LOCKED_REPLAY_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_U8_SNAKE_SOURCE_LOCKED_REPLAY_SAME_CUTOFF_ERROR_ROLE = "primary_same_phonon_cutoff_metric"
_HH_SPSA_BUDGET_LADDER_BATCH_PREFIX = "paper_i_hh_spsa_budget_ladder_"
_HH_SPSA_BUDGET_LADDER_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_SPSA_BUDGET_LADDER_SAME_CUTOFF_ERROR_ROLE = "primary"
_HH_SNAKE_PAULI_CHILD_REPAIR_BATCH_PREFIX = "paper_i_hh_snake_pauli_child_repair_"
_HH_SNAKE_PAULI_CHILD_REPAIR_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_SNAKE_PAULI_CHILD_REPAIR_SAME_CUTOFF_ERROR_ROLE = "primary"
_HH_ROTOSOLVE_MACRO_BATCH_PREFIX = "paper_i_hh_rotosolve_macro_"
_HH_ROTOSOLVE_MACRO_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_ROTOSOLVE_MACRO_SAME_CUTOFF_ERROR_ROLE = "primary"
_HH_SHARED_POOL_OPTIMIZER_BATCH_PREFIX = "paper_i_hh_shared_pool_"
_HH_SHARED_POOL_OPTIMIZER_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_SHARED_POOL_OPTIMIZER_SAME_CUTOFF_ERROR_ROLE = "primary"
_HH_FULLMETA_PHASE3_SINGLETON_BATCH_PREFIX = "paper_i_hh_fullmeta_phase3_singleton_"
_HH_FULLMETA_SINGLETON_SYMMETRY_BATCH_PREFIX = "paper_i_hh_fullmeta_singleton_symmetry_"
_HH_RECOVERY_CANDIDATE_BATCH_PREFIX = "paper_i_hh_recovery_candidate_"
_HH_FULLMETA_SINGLETON_ORDERED_BATCH_BEAM_PREFIX = (
    "paper_i_hh_fullmeta_singleton_symmetry_ordered_batch_beam_"
)
_HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_BATCH_PREFIX = (
    "paper_i_hh_weak_weak_snake_mechanism_ablation_"
)
_HH_ALL_REGIME_SNAKE_MECHANISM_ABLATION_BATCH_PREFIX = (
    "paper_i_hh_all_regime_snake_mechanism_ablation_"
)
_HH_FULLMETA_SINGLETON_ORDERED_BATCH_BEAM_REGIMES = {
    "weak-weak",
    "intermediate-weak",
    "strong-weak",
    "weak-strong",
    "intermediate-strong",
    "strong-strong",
}
_HH_FULLMETA_PHASE3_SINGLETON_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_FULLMETA_PHASE3_SINGLETON_SAME_CUTOFF_ERROR_ROLE = "primary"
_HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_PRIMARY_ENERGY_METRIC = "same_cutoff_abs_delta_e"
_HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_SAME_CUTOFF_ERROR_ROLE = "primary"
_PAPER_I_SCALING_MATRIX_BATCH_PREFIX = "paper_i_scaling_matrix_"
_PAPER_I_SCALING_MATRIX_PROFILE = "paper_i_scaling_matrix_20260710_v1"
_PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_BATCH_PREFIX = "paper_i_scaling_matrix_snake_overlay_repair_"
_PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_SCOPE = "snake_only_all_34_physical_cases_overlay_plumbing_v1"
_PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SCOPE = (
    "single_append_hubbard_L4_strong_powell_cap_finite_nonincreasing_v1"
)
_PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SOURCE_HASHES = {
    "source_result_json_sha256": "00dca6c25128958ee7e7b5a9c85714098aaddb7b80dbd9fc011d78a4b6babdce",
    "source_cell_manifest_sha256": "340d419416ea173a868136d7bf1fff85ccdaa3d4cfc3dc29dc7d4ffa2f4d2297",
    "source_code_bundle_sha256": "d0982f4696eaecdde533aaf2194af0900fb5fa7c4adde301056cbab80354c3fb",
    "source_implementation_lock_sha256": "737fc83467002beaa820f1e411563c9877b5eaafb3dc203027c1e0d56fddff05",
}
_PAPER_I_SCALING_MATRIX_ALGORITHMS = {
    "snake": "static_family_native_adapt_phase3",
    "geo": "static_geo_adapt_vqe",
    "append": "static_full_meta_append_adapt_vqe",
}

def _is_hh_tableiii_snake_shot_proxy_repair(row: Mapping[str, str]) -> bool:
    return (
        str(row.get("run_class") or "").strip() == _HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_RUN_CLASS
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
    )


def _is_hh_tableiii_snake_exact_prefix_recovery(row: Mapping[str, str]) -> bool:
    return (
        str(row.get("run_class") or "").strip() == _HH_TABLEIII_SNAKE_EXACT_PREFIX_RECOVERY_RUN_CLASS
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
    )


def _is_hh_u8_snake_source_locked_replay(row: Mapping[str, str]) -> bool:
    return (
        str(row.get("run_class") or "").strip() == _HH_U8_SNAKE_SOURCE_LOCKED_REPLAY_RUN_CLASS
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
    )


def _is_hh_spsa_budget_ladder(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    algorithm_id = str(row.get("algorithm_id") or "").strip()
    return (
        batch_id.startswith(_HH_SPSA_BUDGET_LADDER_BATCH_PREFIX)
        and str(row.get("family") or "").strip() == "hh"
        and algorithm_id
        in {
            "static_family_native_adapt_phase3",
            "static_full_meta_append_adapt_vqe",
            "static_geo_adapt_vqe",
        }
    )


def _is_hh_snake_pauli_child_repair(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    return (
        batch_id.startswith(_HH_SNAKE_PAULI_CHILD_REPAIR_BATCH_PREFIX)
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
    )


def _is_hh_rotosolve_macro(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    algorithm_id = str(row.get("algorithm_id") or "").strip()
    return (
        batch_id.startswith(_HH_ROTOSOLVE_MACRO_BATCH_PREFIX)
        and str(row.get("family") or "").strip() == "hh"
        and algorithm_id
        in {
            "static_family_native_adapt_phase3",
            "static_full_meta_append_adapt_vqe",
            "static_geo_adapt_vqe",
        }
    )


def _is_hh_shared_pool_optimizer(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    algorithm_id = str(row.get("algorithm_id") or "").strip()
    return (
        batch_id.startswith(_HH_SHARED_POOL_OPTIMIZER_BATCH_PREFIX)
        and str(row.get("run_class") or "").strip() == "diagnostic"
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("shared_pauli_pool_mode") or "").strip() == "shared_pauli_child_sets_v1"
        and algorithm_id
        in {
            "static_family_native_adapt_phase3",
            "static_full_meta_append_adapt_vqe",
            "static_geo_adapt_vqe",
        }
    )


def _is_hh_fullmeta_phase3_singleton(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    algorithm_id = str(row.get("algorithm_id") or "").strip()
    fullmeta_singleton_symmetry = batch_id.startswith(_HH_FULLMETA_SINGLETON_SYMMETRY_BATCH_PREFIX)
    recovery_candidate = batch_id.startswith(_HH_RECOVERY_CANDIDATE_BATCH_PREFIX)
    ordered_batch_beam = (
        batch_id.startswith(_HH_FULLMETA_SINGLETON_ORDERED_BATCH_BEAM_PREFIX)
        or str(row.get("ordered_batch_beam_enabled") or "").strip().lower() == "true"
    )
    run_class = str(row.get("run_class") or "").strip()
    if not (
        (
            batch_id.startswith(_HH_FULLMETA_PHASE3_SINGLETON_BATCH_PREFIX)
            or fullmeta_singleton_symmetry
            or recovery_candidate
        )
        and (run_class == "candidate" or (ordered_batch_beam and run_class == "diagnostic"))
        and str(row.get("family") or "").strip() == "hh"
        and algorithm_id
        in {
            "static_family_native_adapt_phase3",
            "static_full_meta_append_adapt_vqe",
            "static_geo_adapt_vqe",
        }
    ):
        return False
    if fullmeta_singleton_symmetry or recovery_candidate:
        return (
            str(row.get("pool_contract") or "").strip() == "full_meta_unfiltered"
            and str(row.get("hh_adaptive_pool_profile") or "").strip() == "full_meta_unfiltered"
            and str(row.get("matrix_label") or "").strip()
            in {
                "A_native_staged_singleton_hard_guard",
                "A_native_staged_singleton_no_guard",
                "A_native_staged_singleton_true_no_guard",
                "B_common_phase0_singleton_hard_guard",
                "B_common_phase0_singleton_no_guard",
                "C_macro_only",
            }
        )
    if str(row.get("method_key") or "").strip() == "snake":
        return (
            str(row.get("snake_phase3_runtime_split_mode") or "").strip() == "shortlist_pauli_children_v1"
            and str(row.get("snake_phase3_runtime_split_selection_mode") or "").strip()
            == "archival_child_set_forward_v1"
            and str(row.get("snake_phase3_runtime_split_max_subset_size") or "").strip() == "1"
        )
    return (
        str(row.get("generic_adapt_runtime_split_mode") or "").strip() == "shortlist_pauli_children_v1"
        and str(row.get("generic_adapt_runtime_split_symmetry_policy") or "").strip() == "hard_guard"
        and str(row.get("generic_adapt_runtime_split_max_subset_size") or "").strip() == "1"
    )


def _is_hh_weak_weak_snake_mechanism_ablation(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    return (
        batch_id.startswith(_HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_BATCH_PREFIX)
        and str(row.get("run_class") or "").strip() == "candidate"
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("display_regime") or "").strip() == "weak-weak"
        and str(row.get("method_key") or "").strip() == "snake"
        and str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
    )


def _is_hh_all_regime_snake_mechanism_ablation(row: Mapping[str, str]) -> bool:
    batch_id = str(row.get("batch_id") or "").strip()
    return (
        batch_id.startswith(_HH_ALL_REGIME_SNAKE_MECHANISM_ABLATION_BATCH_PREFIX)
        and str(row.get("run_class") or "").strip() == "candidate"
        and str(row.get("family") or "").strip() == "hh"
        and str(row.get("method_key") or "").strip() == "snake"
        and str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
    )


def _is_paper_i_scaling_matrix(row: Mapping[str, str]) -> bool:
    return str(row.get("batch_id") or "").strip().startswith(_PAPER_I_SCALING_MATRIX_BATCH_PREFIX)


def _paper_i_scaling_matrix_contract_blockers(row: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    method = str(row.get("method_key") or "").strip()
    batch_id = str(row.get("batch_id") or "").strip()
    algorithm = str(row.get("algorithm_id") or "").strip()
    expected_algorithm = _PAPER_I_SCALING_MATRIX_ALGORITHMS.get(method)
    if expected_algorithm is None or algorithm != expected_algorithm:
        blockers.append(f"paper_i_scaling_method_algorithm_mismatch:{method}:{algorithm}:{expected_algorithm}")
    expected = {
        "suite_profile": _PAPER_I_SCALING_MATRIX_PROFILE,
        "run_class": "candidate",
        "runnable": "true",
        "optimizer": "POWELL",
        "adapt_optimizer_kind": "powell",
        "budget": "200",
        "phase3_adapt_maxiter": "200",
        "phase3_refit_maxiter": "200",
        "phase3_final_maxiter": "200",
        "pool_contract": "full_meta_unfiltered",
        "child_policy": "macro_only",
        "matrix_label": "paper_i_scaling_matrix_parent_only",
        "generic_adapt_runtime_split_mode": "off",
        "snake_phase3_runtime_split_mode": "off",
        "shared_pauli_pool_mode": "off",
        "phase2_batching": "off",
        "phase3_batching": "off",
        "one_accepted_parent_per_outer_iteration": "true",
        "primary_energy_metric": "same_cutoff_abs_delta_e",
        "same_cutoff_error_role": "primary",
        "exact_fidelity_max_qubits": "10",
        "resource_qubit_cap": "16",
        "resource_pool_term_cap": "1024",
    }
    for field, expected_value in expected.items():
        actual = str(row.get(field) or "").strip()
        if actual != expected_value:
            blockers.append(f"paper_i_scaling_{field}_mismatch:{actual}:expected:{expected_value}")
    try:
        horizon = int(str(row.get("expected_horizon") or ""))
    except Exception:
        horizon = 0
        blockers.append(f"paper_i_scaling_expected_horizon_invalid:{row.get('expected_horizon')}")
    if horizon < 1:
        blockers.append(f"paper_i_scaling_expected_horizon_nonpositive:{horizon}")
    for field in ("max_depth", "phase3_adapt_max_depth"):
        if str(row.get(field) or "").strip() != str(horizon):
            blockers.append(f"paper_i_scaling_{field}_mismatch:{row.get(field)}:expected:{horizon}")
    family = str(row.get("family") or "").strip()
    try:
        L = int(str(row.get("L") or ""))
    except Exception:
        L = 0
        blockers.append(f"paper_i_scaling_L_invalid:{row.get('L')}")
    expected_horizon = 50 if family == "hh" else 20 if family == "hubbard" and L == 2 else 30 if family == "hubbard" else None
    if expected_horizon is not None and horizon != expected_horizon:
        blockers.append(
            f"paper_i_scaling_locked_horizon_mismatch:{family}:L{L}:{horizon}:expected:{expected_horizon}"
        )
    if method == "snake":
        if str(row.get("adapt_allow_repeats") or "").strip() != "true":
            blockers.append("paper_i_scaling_snake_repeat_policy_mismatch")
        if str(row.get("generic_adapt_stop_policy") or "").strip():
            blockers.append("paper_i_scaling_snake_generic_stop_policy_must_be_blank")
        if str(row.get("snake_fixed_horizon_no_target") or "").strip() != "true":
            blockers.append("paper_i_scaling_snake_fixed_horizon_flag_missing")
        if not str(row.get("phase3_policy_json") or "").strip():
            blockers.append("paper_i_scaling_snake_policy_json_missing")
        if str(row.get("request_cpus") or "").strip() != "4":
            blockers.append(f"paper_i_scaling_snake_request_cpus_mismatch:{row.get('request_cpus')}:expected:4")
        if str(row.get("adapt_parallel_gradient_workers") or "").strip() != "2":
            blockers.append("paper_i_scaling_snake_parallel_gradient_workers_mismatch")
        if str(row.get("adapt_beam_parent_workers") or "").strip() != "2":
            blockers.append("paper_i_scaling_snake_beam_parent_workers_mismatch")
        if str(row.get("phase3_adapt_parallel_gradient_workers") or "").strip() != "2":
            blockers.append("paper_i_scaling_snake_phase3_parallel_gradient_workers_mismatch")
        if str(row.get("phase3_adapt_beam_parent_workers") or "").strip() != "2":
            blockers.append("paper_i_scaling_snake_phase3_beam_parent_workers_mismatch")
    else:
        if str(row.get("adapt_allow_repeats") or "").strip() != "true":
            blockers.append("paper_i_scaling_comparator_replacement_policy_mismatch")
        if str(row.get("generic_adapt_stop_policy") or "").strip() != "fixed_horizon_no_target_v1":
            blockers.append("paper_i_scaling_comparator_fixed_horizon_policy_missing")
        if str(row.get("request_cpus") or "").strip() != "1":
            blockers.append(
                f"paper_i_scaling_comparator_request_cpus_mismatch:{row.get('request_cpus')}:expected:1"
            )
        if str(row.get("adapt_parallel_gradient_workers") or "").strip() != "not_applicable":
            blockers.append("paper_i_scaling_comparator_parallel_gradient_workers_must_be_not_applicable")
        if str(row.get("adapt_beam_parent_workers") or "").strip() != "not_applicable":
            blockers.append("paper_i_scaling_comparator_beam_parent_workers_must_be_not_applicable")
    if batch_id.startswith(_PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_BATCH_PREFIX):
        if method != "snake":
            blockers.append("paper_i_scaling_snake_overlay_repair_non_snake_row")
        if str(row.get("repair_scope") or "").strip() != _PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_SCOPE:
            blockers.append("paper_i_scaling_snake_overlay_repair_scope_mismatch")
        if not str(row.get("repair_source_batch_id") or "").strip():
            blockers.append("paper_i_scaling_snake_overlay_repair_source_batch_missing")
        if not str(row.get("repair_source_record_id") or "").strip():
            blockers.append("paper_i_scaling_snake_overlay_repair_source_record_missing")
    repair_scope = str(row.get("repair_scope") or "").strip()
    if repair_scope == _PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SCOPE:
        if method != "append" or algorithm != "static_full_meta_append_adapt_vqe":
            blockers.append("paper_i_scaling_append_powell_cap_repair_method_mismatch")
        if family != "hubbard" or str(row.get("case_id") or "").strip() != (
            "hubbard_L4_scaling_strong"
        ):
            blockers.append("paper_i_scaling_append_powell_cap_repair_case_mismatch")
        if str(row.get("powell_maxiter_cap_policy") or "").strip() != (
            "accept_finite_nonincreasing_v1"
        ):
            blockers.append("paper_i_scaling_append_powell_cap_repair_policy_mismatch")
        if str(row.get("repair_source_batch_id") or "").strip() != (
            "paper_i_scaling_matrix_parent_powell200_20260710_v1"
        ):
            blockers.append("paper_i_scaling_append_powell_cap_repair_source_batch_mismatch")
        if not str(row.get("repair_source_record_id") or "").strip():
            blockers.append("paper_i_scaling_append_powell_cap_repair_source_record_missing")
        for field, expected_hash in _PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SOURCE_HASHES.items():
            actual_hash = str(row.get(field) or "").strip()
            if actual_hash != expected_hash:
                blockers.append(
                    f"paper_i_scaling_append_powell_cap_repair_{field}_mismatch:"
                    f"{actual_hash}:expected:{expected_hash}"
                )
        if not str(row.get("implementation_repair_audit") or "").strip():
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_missing")
        if not str(row.get("settings_diff_json") or "").strip():
            blockers.append("paper_i_scaling_append_powell_cap_repair_settings_diff_missing")
    elif str(row.get("powell_maxiter_cap_policy") or "").strip():
        blockers.append("paper_i_scaling_powell_cap_policy_outside_approved_repair_scope")
    if method == "geo" and str(row.get("geo_immediate_repeat_policy") or "").strip() != (
        "block_only_adjacent_repeat_after_full_pool_selection"
    ):
        blockers.append("paper_i_scaling_geo_immediate_repeat_contract_mismatch")
    if method == "append" and str(row.get("append_selection_policy") or "").strip() != (
        "append_only_with_replacement"
    ):
        blockers.append("paper_i_scaling_append_contract_mismatch")
    if family == "hh":
        if str(row.get("hh_adaptive_pool_profile") or "").strip() != "full_meta_unfiltered":
            blockers.append("paper_i_scaling_hh_pool_profile_mismatch")
        if str(row.get("hh_pool_cache_mode") or "").strip() != "disk":
            blockers.append("paper_i_scaling_hh_cache_mode_mismatch")
        if str(row.get("hh_pool_cache_scope") or "").strip() != "exact":
            blockers.append("paper_i_scaling_hh_cache_scope_mismatch")
        if str(row.get("hh_pool_cache_required") or "").strip() != "true":
            blockers.append("paper_i_scaling_hh_cache_required_flag_missing")
        if str(row.get("hh_generator_registry_cache_mode") or "").strip() != "disk":
            blockers.append("paper_i_scaling_hh_generator_registry_cache_mode_mismatch")
        if str(row.get("hh_generator_registry_cache_required") or "").strip() != "true":
            blockers.append("paper_i_scaling_hh_generator_registry_cache_required_flag_missing")
    return blockers


def _paper_i_scaling_transfer_blockers(
    transfer_inputs: Sequence[str],
    *,
    records_rel: str,
) -> list[str]:
    blockers: list[str] = []
    normalized_transfers = {str(value).strip().rstrip("/") for value in transfer_inputs}
    if "chtc/phase3_optuna" in normalized_transfers:
        blockers.append("paper_i_scaling_broad_phase3_optuna_transfer_forbidden")
    expected_input_dir = str(Path(records_rel).parent)
    if expected_input_dir not in normalized_transfers:
        blockers.append(f"paper_i_scaling_specific_input_dir_not_transferred:{expected_input_dir}")
    for value in normalized_transfers:
        if not value.startswith("chtc/phase3_optuna/input/"):
            continue
        if value != expected_input_dir and not value.startswith(expected_input_dir + "/"):
            blockers.append(f"paper_i_scaling_old_input_tree_transfer_forbidden:{value}")
    required_transfers = {
        "chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh",
        "chtc/phase3_optuna/image.sif",
    }
    for required in sorted(required_transfers - normalized_transfers):
        blockers.append(f"paper_i_scaling_required_transfer_missing:{required}")
    forbidden_direct_transfers = {
        "pipelines",
        "src",
        "docs",
        "test_support",
        "chtc/__init__.py",
        "chtc/phase3_optuna/__init__.py",
        "chtc/phase3_optuna/run_paper_i_scaling_matrix_cell.py",
    }
    for forbidden in sorted(forbidden_direct_transfers & normalized_transfers):
        blockers.append(f"paper_i_scaling_direct_source_transfer_forbidden_use_bundle:{forbidden}")
    return blockers


def _paper_i_scaling_snake_policy_blockers(
    row: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    if str(row.get("method_key") or "").strip() != "snake":
        return []
    blockers: list[str] = []
    value = str(row.get("phase3_policy_json") or "").strip()
    expected_sha = str(row.get("phase3_policy_json_sha256") or "").strip()
    try:
        path = run_task._resolve_under_repo(value, repo_root=repo_root)
        actual_sha = run_task.sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"paper_i_scaling_snake_policy_sha_mismatch:{actual_sha}:{expected_sha}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        pool = payload.get("pool")
        static = payload.get("static")
        inner = payload.get("inner_optimizer")
        if not isinstance(pool, Mapping) or not isinstance(static, Mapping) or not isinstance(inner, Mapping):
            blockers.append("paper_i_scaling_snake_policy_sections_invalid")
            return blockers
        expected_pool = {"pool_key": "full_meta"}
        expected_static: dict[str, Any] = {
            "static_meta_feature_profile": "paper_i_production_v1",
            "static_route_id": "route_a",
            "adapt_max_depth": 50,
            "adapt_maxiter": 200,
            "adapt_drop_floor": -1.0,
            "adapt_drop_patience": 0,
            "adapt_drop_min_depth": 0,
            "adapt_eps_grad": 0.0,
            "adapt_eps_energy": 0.0,
            "adapt_reopt_policy": "full",
            "adapt_insertion_mode": "full_commutation_reduced",
            "adapt_full_refit_every": 1,
            "adapt_final_full_refit": True,
            "adapt_final_refit_maxiter": 200,
            "adapt_allow_repeats": True,
            "adapt_parallel_gradient_workers": 2,
            "adapt_beam_parent_workers": 2,
            "phase2_enable_batching": False,
            "phase3_enable_batching": False,
            "phase3_runtime_split_mode": "off",
            "shared_pauli_pool_mode": "off",
            "shared_pauli_pool_symmetry_policy": "off",
        }
        expected_inner: dict[str, Any] = {
            "inner_optimizer": "POWELL",
            "final_optimizer_type": "POWELL",
            "refit_maxiter": 200,
            "final_maxiter": 200,
        }
        for section_name, section, expected in (
            ("pool", pool, expected_pool),
            ("static", static, expected_static),
            ("inner_optimizer", inner, expected_inner),
        ):
            for field, expected_value in expected.items():
                if section.get(field) != expected_value:
                    blockers.append(
                        f"paper_i_scaling_snake_policy_{section_name}_{field}_mismatch:"
                        f"{section.get(field)!r}:expected:{expected_value!r}"
                    )
    except Exception as exc:
        blockers.append(f"paper_i_scaling_snake_policy_invalid:{type(exc).__name__}:{exc}")
    return blockers


def _paper_i_scaling_matrix_bundle_blockers(rows: Sequence[Mapping[str, str]]) -> list[str]:
    from collections import Counter, defaultdict

    from pipelines.exact_bench.table_i_canonical_cases import table_i_executable_case_ids_by_family

    blockers: list[str] = []
    scaling_rows = [row for row in rows if _is_paper_i_scaling_matrix(row)]
    if not scaling_rows:
        return blockers
    repair_scopes = {str(row.get("repair_scope") or "").strip() for row in scaling_rows}
    snake_overlay_repair = repair_scopes == {_PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_SCOPE}
    append_powell_cap_repair = repair_scopes == {
        _PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SCOPE
    }
    if len(repair_scopes) != 1:
        blockers.append(f"paper_i_scaling_matrix_mixed_repair_scopes:{sorted(repair_scopes)}")
    expected_row_count = 1 if append_powell_cap_repair else 34 if snake_overlay_repair else 102
    if len(scaling_rows) != expected_row_count:
        blockers.append(
            f"paper_i_scaling_matrix_row_count_mismatch:{len(scaling_rows)}:expected:{expected_row_count}"
        )
    record_ids = [str(row.get("record_id") or "") for row in scaling_rows]
    if len(set(record_ids)) != len(record_ids) or any(not value for value in record_ids):
        blockers.append("paper_i_scaling_matrix_record_ids_not_unique_nonblank")

    if append_powell_cap_repair:
        expected_case_methods = {
            (
                "hubbard",
                "hubbard_L4_scaling_strong",
                "append",
                "static_full_meta_append_adapt_vqe",
            )
        }
        actual_case_methods = {
            (
                str(row.get("family") or ""),
                str(row.get("case_id") or ""),
                str(row.get("method_key") or ""),
                str(row.get("algorithm_id") or ""),
            )
            for row in scaling_rows
        }
        if actual_case_methods != expected_case_methods:
            blockers.append(
                "paper_i_scaling_append_powell_cap_repair_case_method_mismatch:"
                f"{sorted(actual_case_methods)}:expected:{sorted(expected_case_methods)}"
            )
        if {str(row.get("request_cpus") or "") for row in scaling_rows} != {"1"}:
            blockers.append("paper_i_scaling_append_powell_cap_repair_cpu_mismatch")
        return blockers

    canonical = table_i_executable_case_ids_by_family(_PAPER_I_SCALING_MATRIX_PROFILE)
    expected_cases = {(str(family), str(case_id)) for family, case_ids in canonical.items() for case_id in case_ids}
    actual_cases = {
        (str(row.get("family") or ""), str(row.get("case_id") or ""))
        for row in scaling_rows
    }
    if actual_cases != expected_cases:
        blockers.append(
            "paper_i_scaling_matrix_case_set_mismatch:"
            f"missing={sorted(expected_cases - actual_cases)}:extra={sorted(actual_cases - expected_cases)}"
        )

    by_case: dict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for row in scaling_rows:
        by_case[(str(row.get("family") or ""), str(row.get("case_id") or ""))].append(row)
    expected_method_pairs = (
        {("snake", _PAPER_I_SCALING_MATRIX_ALGORITHMS["snake"])}
        if snake_overlay_repair
        else set(_PAPER_I_SCALING_MATRIX_ALGORITHMS.items())
    )
    expected_methods_per_case = 1 if snake_overlay_repair else 3
    for case_key, case_rows in sorted(by_case.items()):
        actual_method_pairs = {
            (str(row.get("method_key") or ""), str(row.get("algorithm_id") or ""))
            for row in case_rows
        }
        if len(case_rows) != expected_methods_per_case or actual_method_pairs != expected_method_pairs:
            blockers.append(
                f"paper_i_scaling_matrix_case_method_set_mismatch:{case_key}:"
                f"count={len(case_rows)}:methods={sorted(actual_method_pairs)}"
            )

    expected_family_row_counts = (
        {"hh": 12, "hubbard": 6, "spin_boson": 8, "bose_hubbard": 8}
        if snake_overlay_repair
        else {"hh": 36, "hubbard": 18, "spin_boson": 24, "bose_hubbard": 24}
    )
    actual_family_row_counts = Counter(str(row.get("family") or "") for row in scaling_rows)
    if dict(actual_family_row_counts) != expected_family_row_counts:
        blockers.append(
            f"paper_i_scaling_matrix_family_row_counts_mismatch:{dict(actual_family_row_counts)}:"
            f"expected:{expected_family_row_counts}"
        )

    physical_rows = [case_rows[0] for case_rows in by_case.values() if case_rows]
    pair_counts: dict[str, Counter[tuple[int, int | None]]] = defaultdict(Counter)
    for row in physical_rows:
        try:
            L = int(str(row.get("L") or ""))
            n_ph = None if str(row.get("n_ph_work") or "").strip() == "" else int(str(row.get("n_ph_work")))
        except Exception:
            blockers.append(f"paper_i_scaling_matrix_pair_invalid:{row.get('case_id')}")
            continue
        pair_counts[str(row.get("family") or "")][(L, n_ph)] += 1
    expected_pair_counts = {
        "hh": Counter({(3, 2): 6, (4, 1): 6}),
        "hubbard": Counter({(2, None): 2, (3, None): 2, (4, None): 2}),
        "spin_boson": Counter({(2, 4): 2, (3, 3): 2, (3, 2): 2, (4, 1): 2}),
        "bose_hubbard": Counter({(2, 3): 2, (3, 3): 2, (3, 2): 2, (4, 1): 2}),
    }
    for family, expected in expected_pair_counts.items():
        if pair_counts.get(family, Counter()) != expected:
            blockers.append(
                f"paper_i_scaling_matrix_pair_counts_mismatch:{family}:{dict(pair_counts.get(family, Counter()))}:"
                f"expected:{dict(expected)}"
            )
    for family in ("spin_boson", "bose_hubbard"):
        horizons = {
            str(row.get("expected_horizon") or "")
            for row in scaling_rows
            if str(row.get("family") or "") == family
        }
        if len(horizons) != 1:
            blockers.append(f"paper_i_scaling_matrix_mixed_family_horizons:{family}:{sorted(horizons)}")
    cpu_counts = Counter(str(row.get("request_cpus") or "") for row in scaling_rows)
    expected_cpu_counts = Counter({"4": 34}) if snake_overlay_repair else Counter({"4": 34, "1": 68})
    if cpu_counts != expected_cpu_counts:
        blockers.append(
            f"paper_i_scaling_matrix_cpu_counts_mismatch:{dict(cpu_counts)}:expected:{dict(expected_cpu_counts)}"
        )
    return blockers


def _paper_i_scaling_submit_contract_blockers(
    contract: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    *,
    records_rel: str,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    blockers: list[str] = []
    scaling_rows = [row for row in rows if _is_paper_i_scaling_matrix(row)]
    if not scaling_rows:
        return blockers
    batch_ids = {str(row.get("batch_id") or "").strip() for row in scaling_rows}
    if len(batch_ids) != 1:
        blockers.append(f"paper_i_scaling_submit_mixed_batch_ids:{sorted(batch_ids)}")
        return blockers
    batch_id = next(iter(batch_ids))
    repair_scopes = {str(row.get("repair_scope") or "").strip() for row in scaling_rows}
    snake_overlay_repair = repair_scopes == {_PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_SCOPE}
    append_powell_cap_repair = repair_scopes == {
        _PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SCOPE
    }
    expected_arguments = f"$(record_id) {records_rel} raw_outputs/{batch_id}/$(record_id)"
    expected_output = [f"raw_outputs/{batch_id}/$(record_id)"]
    expected_scalars = {
        "executable": "chtc/phase3_optuna/run_paper_i_scaling_matrix_task_apptainer.sh",
        "arguments": expected_arguments,
        "should_transfer_files": "YES",
        "when_to_transfer_output": "ON_EXIT_OR_EVICT",
        "transfer_executable": "True",
        "preserve_relative_paths": "True",
        "request_cpus": "$(cpus)",
        "request_memory": "$(memory_mb)MB",
        "request_disk": "$(disk_mb)MB",
        "max_runtime": "259200",
        "job_batch_name": (
            "paper-i-scaling-snake-overlay-repair"
            if snake_overlay_repair
            else "paper-i-append-powell-cap-repair"
            if append_powell_cap_repair
            else "paper-i-scaling-parent-powell200"
        ),
    }
    for field, expected in expected_scalars.items():
        actual = str(contract.get(field) or "").strip()
        if actual != expected:
            blockers.append(f"paper_i_scaling_submit_{field}_mismatch:{actual}:expected:{expected}")
    actual_outputs = list(contract.get("transfer_output_files") or [])
    if actual_outputs != expected_output:
        blockers.append(f"paper_i_scaling_submit_output_route_mismatch:{actual_outputs}:expected:{expected_output}")
    if "TARGET.HasSIF" not in str(contract.get("requirements") or ""):
        blockers.append("paper_i_scaling_submit_has_sif_requirement_missing")

    queue_value = str(contract.get("queue_record_id_file") or "").strip()
    try:
        queue_path = run_task._resolve_under_repo(queue_value, repo_root=repo_root)
        queue_rows: list[tuple[str, str, str, str]] = []
        for line_number, raw in enumerate(queue_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not raw.strip():
                continue
            parts = raw.split("\t")
            if len(parts) != 4:
                blockers.append(f"paper_i_scaling_queue_column_count:{line_number}:{len(parts)}")
                continue
            queue_rows.append((parts[0], parts[1], parts[2], parts[3]))
        expected_ids = [str(row.get("record_id") or "") for row in scaling_rows]
        queue_ids = [record_id for record_id, _cpus, _memory, _disk in queue_rows]
        if queue_ids != expected_ids:
            blockers.append("paper_i_scaling_queue_record_ids_or_order_mismatch")
        if len(set(queue_ids)) != len(queue_ids):
            blockers.append("paper_i_scaling_queue_record_ids_not_unique")
        expected_resources = {
            str(row.get("record_id") or ""): (
                str(row.get("request_cpus") or ""),
                str(row.get("request_memory_mb") or ""),
                str(row.get("request_disk_mb") or ""),
            )
            for row in scaling_rows
        }
        for record_id, cpus, memory_mb, disk_mb in queue_rows:
            if expected_resources.get(record_id) != (cpus, memory_mb, disk_mb):
                blockers.append(
                    f"paper_i_scaling_queue_resource_mismatch:{record_id}:{cpus}:{memory_mb}:{disk_mb}:"
                    f"expected:{expected_resources.get(record_id)}"
                )
    except Exception as exc:
        blockers.append(f"paper_i_scaling_queue_invalid:{type(exc).__name__}:{exc}")
    return blockers


def _append_unique_blocker(blockers: list[str], blocker: str) -> None:
    if blocker not in blockers:
        blockers.append(blocker)


def _parse_json_mapping_field(row: Mapping[str, str], field: str, blockers: list[str]) -> Mapping[str, Any]:
    try:
        payload = json.loads(str(row.get(field) or "{}"))
    except json.JSONDecodeError:
        blockers.append(f"paper_i_hh_weak_weak_mechanism_{field}_invalid_json")
        return {}
    if not isinstance(payload, Mapping):
        blockers.append(f"paper_i_hh_weak_weak_mechanism_{field}_not_mapping")
        return {}
    return payload


def _override_parts(row: Mapping[str, str], blockers: list[str]) -> tuple[Mapping[str, Any], set[str], set[str], set[str]]:
    overrides = _parse_json_mapping_field(row, "snake_cli_overrides_json", blockers)
    set_flags = overrides.get("set_flags") if isinstance(overrides, Mapping) else None
    enable_flags = overrides.get("enable_flags") if isinstance(overrides, Mapping) else None
    remove_bool_flags = overrides.get("remove_bool_flags") if isinstance(overrides, Mapping) else None
    remove_value_flags = overrides.get("remove_value_flags") if isinstance(overrides, Mapping) else None
    if not isinstance(set_flags, Mapping):
        blockers.append("paper_i_hh_weak_weak_mechanism_set_flags_missing")
        set_flags = {}
    enable_set = {str(flag) for flag in enable_flags} if isinstance(enable_flags, Sequence) and not isinstance(enable_flags, (str, bytes)) else set()
    remove_bool_set = (
        {str(flag) for flag in remove_bool_flags}
        if isinstance(remove_bool_flags, Sequence) and not isinstance(remove_bool_flags, (str, bytes))
        else set()
    )
    remove_value_set = (
        {str(flag) for flag in remove_value_flags}
        if isinstance(remove_value_flags, Sequence) and not isinstance(remove_value_flags, (str, bytes))
        else set()
    )
    return set_flags, enable_set, remove_bool_set, remove_value_set


def _require_override(set_flags: Mapping[str, Any], blockers: list[str], flag: str, expected: str) -> None:
    actual = str(set_flags.get(flag) or "").strip()
    if actual != expected:
        blockers.append(f"paper_i_hh_weak_weak_mechanism_override_mismatch:{flag}:{actual}:expected:{expected}")


def _hh_weak_weak_snake_mechanism_ablation_contract_blockers(row: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    source_family = str(row.get("source_anchor_family") or "").strip()
    variant = str(row.get("hh_mechanism_ablation_variant") or row.get("route_variant") or "").strip()
    child_policy = str(row.get("child_policy") or "").strip()
    set_flags, enable_flags, remove_bool_flags, remove_value_flags = _override_parts(row, blockers)

    expected_static = {
        "display_regime": "weak-weak",
        "internal_regime": "weak_weak",
        "case_id": "hh_L2_nph2_three_model_sym_weak_weak",
        "suite_profile": "paper_i_three_model_hh_symmetric_20260527_v1",
        "optimizer": "POWELL",
        "adapt_optimizer_kind": "powell",
        "budget": "200",
        "max_depth": "30",
        "pool_contract": "full_meta_unfiltered",
        "hh_adaptive_pool_profile": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
        "adapt_beam_lambda": "0.005",
        "adapt_beam_live_branches": "3",
        "adapt_beam_children_per_parent": "2",
    }
    for field, expected in expected_static.items():
        actual = str(row.get(field) or "").strip()
        if actual != expected:
            blockers.append(f"paper_i_hh_weak_weak_mechanism_{field}_mismatch:{actual}:expected:{expected}")
    if source_family not in {"batch_cap3_combinatorial", "physical_operator_lane"}:
        blockers.append(f"paper_i_hh_weak_weak_mechanism_unknown_source_anchor_family:{source_family}")
    if "hh_full_meta_minus_hva_class_filter.json" in json.dumps(dict(row), sort_keys=True):
        blockers.append("paper_i_hh_weak_weak_mechanism_minus_hva_filter_present")

    _require_override(set_flags, blockers, "--phase3-runtime-split-max-subset-size", "1")
    _require_override(set_flags, blockers, "--adapt-beam-live-branches", "3")
    _require_override(set_flags, blockers, "--adapt-beam-children-per-parent", "2")
    _require_override(set_flags, blockers, "--adapt-beam-lambda", "0.005")
    _require_override(set_flags, blockers, "--phase1-prune-schur-nomination-route", "metric_regularized_v1")
    if str(set_flags.get("--static-route-id") or "").strip() != "unspecified" and str(row.get("runnable") or "").strip().lower() == "true":
        blockers.append("paper_i_hh_weak_weak_mechanism_static_route_id_not_unspecified")
    if "--phase3-source-lock-preferred-sequence" not in remove_value_flags:
        blockers.append("paper_i_hh_weak_weak_mechanism_source_lock_preferred_sequence_not_removed")

    if child_policy == "native_phase3_singleton":
        if str(row.get("snake_phase3_runtime_split_mode") or "").strip() != "shortlist_pauli_children_v1":
            blockers.append("paper_i_hh_weak_weak_mechanism_runtime_split_mode_mismatch")
        if str(row.get("snake_phase3_runtime_split_selection_mode") or "").strip() != "archival_child_set_forward_v1":
            blockers.append("paper_i_hh_weak_weak_mechanism_runtime_split_selection_mismatch")
        if str(row.get("snake_phase3_runtime_split_child_set_symmetry_policy") or "").strip() != "hard_guard":
            blockers.append("paper_i_hh_weak_weak_mechanism_runtime_split_symmetry_mismatch")
        if str(row.get("snake_phase3_runtime_split_max_subset_size") or "").strip() != "1":
            blockers.append("paper_i_hh_weak_weak_mechanism_runtime_split_cap_mismatch")
        if str(row.get("shared_pauli_pool_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_weak_weak_mechanism_shared_pool_unexpected_for_native")
    elif child_policy == "macro_only":
        if str(row.get("snake_phase3_runtime_split_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_weak_weak_mechanism_macro_split_not_off")
        if str(row.get("shared_pauli_pool_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_weak_weak_mechanism_macro_shared_pool_not_off")
    elif child_policy == "common_phase0_singleton":
        if str(row.get("snake_phase3_runtime_split_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_weak_weak_mechanism_phase0_split_not_off")
        if str(row.get("shared_pauli_pool_mode") or "").strip() != "shared_pauli_child_sets_v1":
            blockers.append("paper_i_hh_weak_weak_mechanism_phase0_shared_pool_mode_mismatch")
        if str(row.get("shared_pauli_pool_symmetry_policy") or "").strip() != "hard_guard":
            blockers.append("paper_i_hh_weak_weak_mechanism_phase0_shared_pool_policy_mismatch")
        if str(row.get("shared_pauli_pool_max_subset_size") or "").strip() != "1":
            blockers.append("paper_i_hh_weak_weak_mechanism_phase0_shared_pool_cap_mismatch")
    else:
        blockers.append(f"paper_i_hh_weak_weak_mechanism_unknown_child_policy:{child_policy}")

    if variant in {"greedy_cap3", "combinatorial_cap3"}:
        expected_mode = "greedy_reduced_plane" if variant == "greedy_cap3" else "combinatorial_reduced_plane"
        if source_family == "batch_cap3_combinatorial" and str(row.get("runnable") or "").strip().lower() == "true":
            blockers.append("paper_i_hh_weak_weak_mechanism_existing_combo_batch_variant_should_not_queue")
        if str(row.get("ordered_batch_beam_enabled") or "").strip() != "true":
            blockers.append("paper_i_hh_weak_weak_mechanism_batch_variant_not_enabled")
        if str(row.get("phase2_batch_selection_mode") or "").strip() != expected_mode:
            blockers.append("paper_i_hh_weak_weak_mechanism_batch_mode_mismatch")
        if str(row.get("phase2_batch_target_size") or "").strip() != "3":
            blockers.append("paper_i_hh_weak_weak_mechanism_batch_target_size_mismatch")
        if str(row.get("phase2_batch_size_cap") or "").strip() != "3":
            blockers.append("paper_i_hh_weak_weak_mechanism_batch_size_cap_mismatch")
        for flag in ("--phase2-enable-batching", "--phase3-enable-batching"):
            if flag not in enable_flags:
                blockers.append(f"paper_i_hh_weak_weak_mechanism_batch_enable_missing:{flag}")
        for flag in (
            "--phase2-batch-selection-mode",
            "--phase3-batch-selection-mode",
        ):
            _require_override(set_flags, blockers, flag, expected_mode)
        for flag in (
            "--phase2-batch-target-size",
            "--phase2-batch-size-cap",
            "--phase3-batch-target-size",
            "--phase3-batch-size-cap",
        ):
            _require_override(set_flags, blockers, flag, "3")
    else:
        if str(row.get("ordered_batch_beam_enabled") or "").strip() == "true":
            blockers.append("paper_i_hh_weak_weak_mechanism_nonbatch_marked_batch_enabled")

    if variant == "no_batching_reference":
        for flag in ("--phase2-no-batching", "--phase3-no-batching"):
            if flag not in enable_flags:
                blockers.append(f"paper_i_hh_weak_weak_mechanism_nobatch_flag_missing:{flag}")
        for flag in ("--phase2-enable-batching", "--phase3-enable-batching"):
            if flag not in remove_bool_flags:
                blockers.append(f"paper_i_hh_weak_weak_mechanism_nobatch_remove_missing:{flag}")

    if variant == "no_prune" and "--phase1-no-prune" not in enable_flags:
        blockers.append("paper_i_hh_weak_weak_mechanism_no_prune_flag_missing")
    if variant == "no_cost_term":
        for flag in (
            "--phase2-w-shot",
            "--phase2-w-depth",
            "--phase3-backend-w-depth",
            "--phase3-backend-cost-mode",
        ):
            expected = "proxy" if flag == "--phase3-backend-cost-mode" else "0.0"
            _require_override(set_flags, blockers, flag, expected)
        if str(set_flags.get("--adapt-beam-lambda") or "").strip() != "0.005":
            blockers.append("paper_i_hh_weak_weak_mechanism_no_cost_changed_beam_lambda")
    if variant == "no_novelty":
        _require_override(set_flags, blockers, "--phase2-gamma-N", "0.0")
        _require_override(set_flags, blockers, "--phase3-novelty-ablation-mode", "all")
    if variant in {"phase2_novelty_only_no_second_order", "phase2_second_order_only_no_novelty", "no_phase3"}:
        _require_override(set_flags, blockers, "--adapt-continuation-mode", "phase2_v1")
        _require_override(set_flags, blockers, "--phase3-backend-cost-mode", "proxy")
    if variant == "phase2_novelty_only_no_second_order":
        _require_override(set_flags, blockers, "--phase2-selector-gain-mode", "unit_gain_v1")
    if variant == "phase2_second_order_only_no_novelty":
        _require_override(set_flags, blockers, "--phase2-selector-gain-mode", "trust_region_v1")
        _require_override(set_flags, blockers, "--phase2-gamma-N", "0.0")
    if variant in {"phase1_only_macro_pool", "phase1_only_singleton_pool"}:
        _require_override(set_flags, blockers, "--adapt-continuation-mode", "phase1_v1")
        _require_override(set_flags, blockers, "--phase3-backend-cost-mode", "proxy")
    if variant == "full_geometry_window":
        _require_override(set_flags, blockers, "--phase3-selector-geometry-mode", "raw_exact")
    if variant == "no_shortlisting" and str(row.get("runnable") or "").strip().lower() == "true":
        blockers.append("paper_i_hh_weak_weak_mechanism_no_shortlisting_should_remain_blocked")

    source_args_raw = str(row.get("source_command_args_json") or "[]")
    try:
        source_args = [str(item) for item in json.loads(source_args_raw)]
    except Exception:
        blockers.append("paper_i_hh_weak_weak_mechanism_source_command_args_invalid")
        source_args = []
    if source_family == "physical_operator_lane":
        if "--static-lane-route" not in source_args:
            blockers.append("paper_i_hh_weak_weak_mechanism_physical_source_missing_lane_flag")
        else:
            idx = source_args.index("--static-lane-route")
            actual = source_args[idx + 1] if idx + 1 < len(source_args) else ""
            if actual != "physical_operator_type":
                blockers.append(f"paper_i_hh_weak_weak_mechanism_physical_source_lane_mismatch:{actual}")
        if "--physical-lane-shortlist-aggressiveness" not in source_args:
            blockers.append("paper_i_hh_weak_weak_mechanism_physical_source_missing_aggressiveness")

    return blockers


def _hh_all_regime_snake_mechanism_ablation_contract_blockers(row: Mapping[str, str]) -> list[str]:
    blockers: list[str] = []
    variant = str(row.get("hh_mechanism_ablation_variant") or row.get("route_variant") or "").strip()
    child_policy = str(row.get("child_policy") or "").strip()
    display_regime = str(row.get("display_regime") or "").strip()
    set_flags, enable_flags, remove_bool_flags, remove_value_flags = _override_parts(row, blockers)

    allowed_regimes = {
        "weak-weak",
        "intermediate-weak",
        "strong-weak",
        "weak-strong",
        "intermediate-strong",
        "strong-strong",
    }
    allowed_variants = {
        "combinatorial_cap3_anchor",
        "greedy_cap3",
        "no_batching_reference",
        "no_prune",
        "no_cost_term",
        "no_novelty",
        "phase2_novelty_only_no_second_order",
        "phase2_second_order_only_no_novelty",
        "no_phase3",
        "phase1_only_macro_pool",
        "phase1_only_singleton_pool",
        "no_beam",
        "no_lane_global_pool",
    }
    if display_regime not in allowed_regimes:
        blockers.append(f"paper_i_hh_all_regime_mechanism_regime_mismatch:{display_regime}")
    if variant not in allowed_variants:
        blockers.append(f"paper_i_hh_all_regime_mechanism_unknown_variant:{variant}")

    expected_case_ids = {
        "weak-weak": "hh_L2_nph2_three_model_sym_weak_weak",
        "intermediate-weak": "hh_L2_nph2_three_model_sym_strong_weak",
        "strong-weak": "hh_L2_nph2_three_model_sym_strong_weak",
        "weak-strong": "hh_L2_nph4_three_model_sym_weak_strong",
        "intermediate-strong": "hh_L2_nph4_three_model_sym_strong_strong",
        "strong-strong": "hh_L2_nph4_three_model_sym_strong_strong",
    }
    expected_cutoffs = {
        "weak-weak": ("2", "5"),
        "intermediate-weak": ("2", "5"),
        "strong-weak": ("2", "5"),
        "weak-strong": ("4", "7"),
        "intermediate-strong": ("4", "7"),
        "strong-strong": ("4", "7"),
    }
    if display_regime in expected_case_ids and str(row.get("case_id") or "").strip() != expected_case_ids[display_regime]:
        blockers.append(
            f"paper_i_hh_all_regime_mechanism_case_id_mismatch:{row.get('case_id')}:expected:{expected_case_ids[display_regime]}"
        )
    if display_regime in expected_cutoffs:
        n_ph_work, n_ph_ref = expected_cutoffs[display_regime]
        if str(row.get("n_ph_work") or "").strip() != n_ph_work:
            blockers.append(f"paper_i_hh_all_regime_mechanism_n_ph_work_mismatch:{row.get('n_ph_work')}:expected:{n_ph_work}")
        if str(row.get("n_ph_ref") or "").strip() != n_ph_ref:
            blockers.append(f"paper_i_hh_all_regime_mechanism_n_ph_ref_mismatch:{row.get('n_ph_ref')}:expected:{n_ph_ref}")

    expected_static = {
        "source_anchor_family": "physical_operator_lane",
        "optimizer": "POWELL",
        "adapt_optimizer_kind": "powell",
        "budget": "200",
        "max_depth": "30",
        "pool_contract": "full_meta_unfiltered",
        "hh_adaptive_pool_profile": "full_meta_unfiltered",
        "adapt_pool_class_filter_json": "off",
    }
    for field, expected in expected_static.items():
        actual = str(row.get(field) or "").strip()
        if actual != expected:
            blockers.append(f"paper_i_hh_all_regime_mechanism_{field}_mismatch:{actual}:expected:{expected}")
    if "hh_full_meta_minus_hva_class_filter.json" in json.dumps(dict(row), sort_keys=True):
        blockers.append("paper_i_hh_all_regime_mechanism_minus_hva_filter_present")
    if str(row.get("runnable") or "").strip().lower() != "true":
        blockers.append("paper_i_hh_all_regime_mechanism_row_not_runnable")

    for flag, expected in {
        "--static-route-id": "unspecified",
        "--adapt-reopt-policy": "full",
        "--adapt-window-size": "99",
        "--adapt-full-refit-every": "1",
        "--adapt-final-full-refit": "true",
        "--phase3-geometry-window-size": "99",
        "--phase3-runtime-split-max-subset-size": "1",
        "--phase1-prune-schur-nomination-route": "metric_regularized_v1",
    }.items():
        _require_override(set_flags, blockers, flag, expected)
    if "--phase3-source-lock-preferred-sequence" not in remove_value_flags:
        blockers.append("paper_i_hh_all_regime_mechanism_source_lock_preferred_sequence_not_removed")
    if "--phase2-enable-batching" in enable_flags:
        blockers.append("paper_i_hh_all_regime_mechanism_phase2_batching_enabled")
    if "--phase2-no-batching" not in enable_flags:
        blockers.append("paper_i_hh_all_regime_mechanism_phase2_no_batching_missing")
    if "--phase2-enable-batching" not in remove_bool_flags:
        blockers.append("paper_i_hh_all_regime_mechanism_phase2_enable_not_removed")

    expected_beam_live = "1" if variant == "no_beam" else "3"
    expected_beam_children = "1" if variant == "no_beam" else "2"
    _require_override(set_flags, blockers, "--adapt-beam-live-branches", expected_beam_live)
    _require_override(set_flags, blockers, "--adapt-beam-children-per-parent", expected_beam_children)
    _require_override(set_flags, blockers, "--adapt-beam-lambda", "0.005")
    if str(row.get("adapt_beam_live_branches") or "").strip() != expected_beam_live:
        blockers.append("paper_i_hh_all_regime_mechanism_live_branches_row_mismatch")
    if str(row.get("adapt_beam_children_per_parent") or "").strip() != expected_beam_children:
        blockers.append("paper_i_hh_all_regime_mechanism_children_per_parent_row_mismatch")

    expected_lane = "algebraic" if variant == "no_lane_global_pool" else "physical_operator_type"
    _require_override(set_flags, blockers, "--static-lane-route", expected_lane)
    if str(row.get("static_lane_route") or "").strip() != expected_lane:
        blockers.append(f"paper_i_hh_all_regime_mechanism_lane_row_mismatch:{row.get('static_lane_route')}:expected:{expected_lane}")
    if variant == "no_lane_global_pool":
        if "--physical-lane-shortlist-aggressiveness" not in remove_value_flags:
            blockers.append("paper_i_hh_all_regime_mechanism_no_lane_physical_aggressiveness_not_removed")
    else:
        _require_override(set_flags, blockers, "--physical-lane-shortlist-aggressiveness", "3")

    phase3_batch_variants = {
        "combinatorial_cap3_anchor": "combinatorial_reduced_plane",
        "greedy_cap3": "greedy_reduced_plane",
        "no_prune": "combinatorial_reduced_plane",
        "no_cost_term": "combinatorial_reduced_plane",
        "no_novelty": "combinatorial_reduced_plane",
        "no_beam": "combinatorial_reduced_plane",
        "no_lane_global_pool": "combinatorial_reduced_plane",
    }
    if variant in phase3_batch_variants:
        expected_mode = phase3_batch_variants[variant]
        if str(row.get("ordered_batch_beam_enabled") or "").strip() != "true":
            blockers.append("paper_i_hh_all_regime_mechanism_batch_variant_not_enabled")
        if str(row.get("phase3_batch_selection_mode") or "").strip() != expected_mode:
            blockers.append("paper_i_hh_all_regime_mechanism_phase3_batch_mode_row_mismatch")
        _require_override(set_flags, blockers, "--phase3-batch-selection-mode", expected_mode)
        _require_override(set_flags, blockers, "--phase3-batch-target-size", "3")
        _require_override(set_flags, blockers, "--phase3-batch-size-cap", "3")
        if "--phase3-enable-batching" not in enable_flags:
            blockers.append("paper_i_hh_all_regime_mechanism_phase3_enable_missing")
        if "--phase3-no-batching" not in remove_bool_flags:
            blockers.append("paper_i_hh_all_regime_mechanism_phase3_no_batching_not_removed")
        for flag in ("--phase2-batch-selection-mode", "--phase2-batch-target-size", "--phase2-batch-size-cap"):
            if flag not in remove_value_flags:
                blockers.append(f"paper_i_hh_all_regime_mechanism_phase2_batch_value_not_removed:{flag}")
    else:
        if str(row.get("ordered_batch_beam_enabled") or "").strip() == "true":
            blockers.append("paper_i_hh_all_regime_mechanism_nonbatch_marked_batch_enabled")
        if "--phase3-no-batching" not in enable_flags:
            blockers.append("paper_i_hh_all_regime_mechanism_phase3_no_batching_missing")
        if "--phase3-enable-batching" not in remove_bool_flags:
            blockers.append("paper_i_hh_all_regime_mechanism_phase3_enable_not_removed")

    if child_policy == "native_phase3_singleton":
        if str(row.get("snake_phase3_runtime_split_mode") or "").strip() != "shortlist_pauli_children_v1":
            blockers.append("paper_i_hh_all_regime_mechanism_runtime_split_mode_mismatch")
        if str(row.get("snake_phase3_runtime_split_selection_mode") or "").strip() != "archival_child_set_forward_v1":
            blockers.append("paper_i_hh_all_regime_mechanism_runtime_split_selection_mismatch")
        if str(row.get("snake_phase3_runtime_split_child_set_symmetry_policy") or "").strip() != "hard_guard":
            blockers.append("paper_i_hh_all_regime_mechanism_runtime_split_symmetry_mismatch")
        if str(row.get("snake_phase3_runtime_split_max_subset_size") or "").strip() != "1":
            blockers.append("paper_i_hh_all_regime_mechanism_runtime_split_cap_mismatch")
        if str(row.get("shared_pauli_pool_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_all_regime_mechanism_shared_pool_unexpected_for_native")
    elif child_policy == "macro_only":
        if str(row.get("snake_phase3_runtime_split_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_all_regime_mechanism_macro_split_not_off")
        if str(row.get("shared_pauli_pool_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_all_regime_mechanism_macro_shared_pool_not_off")
    elif child_policy == "common_phase0_singleton":
        if str(row.get("snake_phase3_runtime_split_mode") or "").strip() != "off":
            blockers.append("paper_i_hh_all_regime_mechanism_phase0_split_not_off")
        if str(row.get("shared_pauli_pool_mode") or "").strip() != "shared_pauli_child_sets_v1":
            blockers.append("paper_i_hh_all_regime_mechanism_phase0_shared_pool_mode_mismatch")
        if str(row.get("shared_pauli_pool_symmetry_policy") or "").strip() != "hard_guard":
            blockers.append("paper_i_hh_all_regime_mechanism_phase0_shared_pool_policy_mismatch")
        if str(row.get("shared_pauli_pool_max_subset_size") or "").strip() != "1":
            blockers.append("paper_i_hh_all_regime_mechanism_phase0_shared_pool_cap_mismatch")
    else:
        blockers.append(f"paper_i_hh_all_regime_mechanism_unknown_child_policy:{child_policy}")

    if variant == "no_prune" and "--phase1-no-prune" not in enable_flags:
        blockers.append("paper_i_hh_all_regime_mechanism_no_prune_flag_missing")
    if variant == "no_cost_term":
        for flag in ("--phase2-w-shot", "--phase2-w-depth", "--phase3-backend-w-depth"):
            _require_override(set_flags, blockers, flag, "0.0")
        _require_override(set_flags, blockers, "--phase3-backend-cost-mode", "proxy")
    if variant == "no_novelty":
        _require_override(set_flags, blockers, "--phase2-gamma-N", "0.0")
        _require_override(set_flags, blockers, "--phase3-novelty-ablation-mode", "all")
    if variant in {"phase2_novelty_only_no_second_order", "phase2_second_order_only_no_novelty", "no_phase3"}:
        _require_override(set_flags, blockers, "--adapt-continuation-mode", "phase2_v1")
        _require_override(set_flags, blockers, "--phase3-backend-cost-mode", "proxy")
    if variant == "phase2_novelty_only_no_second_order":
        _require_override(set_flags, blockers, "--phase2-selector-gain-mode", "unit_gain_v1")
    if variant == "phase2_second_order_only_no_novelty":
        _require_override(set_flags, blockers, "--phase2-selector-gain-mode", "trust_region_v1")
        _require_override(set_flags, blockers, "--phase2-gamma-N", "0.0")
    if variant in {"phase1_only_macro_pool", "phase1_only_singleton_pool"}:
        _require_override(set_flags, blockers, "--adapt-continuation-mode", "phase1_v1")
        _require_override(set_flags, blockers, "--phase3-backend-cost-mode", "proxy")

    source_args_raw = str(row.get("source_command_args_json") or "[]")
    try:
        source_args = [str(item) for item in json.loads(source_args_raw)]
    except Exception:
        blockers.append("paper_i_hh_all_regime_mechanism_source_command_args_invalid")
        source_args = []
    if "--static-lane-route" not in source_args:
        blockers.append("paper_i_hh_all_regime_mechanism_source_missing_lane_flag")
    else:
        idx = source_args.index("--static-lane-route")
        actual = source_args[idx + 1] if idx + 1 < len(source_args) else ""
        if actual != "physical_operator_type":
            blockers.append(f"paper_i_hh_all_regime_mechanism_source_lane_mismatch:{actual}")
    return blockers


def _hh_fullmeta_phase3_singleton_contract_blockers(row: Mapping[str, str]) -> list[str]:
    label = str(row.get("matrix_label") or "").strip()
    batch_id = str(row.get("batch_id") or "").strip()
    recovery_candidate = batch_id.startswith(_HH_RECOVERY_CANDIDATE_BATCH_PREFIX)
    ordered_batch_beam = (
        batch_id.startswith(_HH_FULLMETA_SINGLETON_ORDERED_BATCH_BEAM_PREFIX)
        or str(row.get("ordered_batch_beam_enabled") or "").strip().lower() == "true"
    )
    blockers: list[str] = []
    expected_symmetry = {
        "A_native_staged_singleton_hard_guard": "hard_guard",
        "A_native_staged_singleton_no_guard": "off",
        "A_native_staged_singleton_true_no_guard": "off",
        "B_common_phase0_singleton_hard_guard": "hard_guard",
        "B_common_phase0_singleton_no_guard": "off",
        "C_macro_only": "not_applicable",
    }.get(label)
    if expected_symmetry is None:
        blockers.append(f"paper_i_hh_fullmeta_singleton_unknown_matrix_label:{label}")
        return blockers
    actual_symmetry = str(row.get("symmetry_policy") or "").strip()
    if actual_symmetry != expected_symmetry:
        blockers.append(
            f"paper_i_hh_fullmeta_singleton_symmetry_policy_mismatch:{label}:{actual_symmetry}:expected:{expected_symmetry}"
        )
    if str(row.get("pool_contract") or "").strip() != "full_meta_unfiltered":
        blockers.append("paper_i_hh_fullmeta_singleton_pool_contract_mismatch")
    if str(row.get("hh_adaptive_pool_profile") or "").strip() != "full_meta_unfiltered":
        blockers.append("paper_i_hh_fullmeta_singleton_hh_profile_mismatch")
    class_filter = str(row.get("adapt_pool_class_filter_json") or "").strip()
    if class_filter not in {"", "off"}:
        blockers.append(f"paper_i_hh_fullmeta_singleton_unexpected_class_filter:{class_filter}")
    if label.startswith("A_native_staged_singleton_"):
        method = str(row.get("method_key") or "").strip()
        if str(row.get("child_policy") or "").strip() != "native_phase3_singleton":
            blockers.append(f"paper_i_hh_fullmeta_singleton_child_policy_mismatch:{label}")
        if method == "snake":
            mode = str(row.get("snake_phase3_runtime_split_mode") or "").strip()
            selection_mode = str(row.get("snake_phase3_runtime_split_selection_mode") or "").strip()
            policy = str(row.get("snake_phase3_runtime_split_child_set_symmetry_policy") or "").strip()
            cap = str(row.get("snake_phase3_runtime_split_max_subset_size") or "").strip()
            expected_child_policy = {
                "A_native_staged_singleton_hard_guard": "hard_guard",
                "A_native_staged_singleton_no_guard": "parent",
                "A_native_staged_singleton_true_no_guard": "off",
            }.get(label)
            if mode != "shortlist_pauli_children_v1":
                blockers.append(f"paper_i_hh_fullmeta_singleton_snake_phase3_mode_mismatch:{label}:{mode}")
            if selection_mode != "archival_child_set_forward_v1":
                blockers.append(
                    f"paper_i_hh_fullmeta_singleton_snake_phase3_selection_mode_mismatch:{label}:{selection_mode}"
                )
            if policy != expected_child_policy:
                blockers.append(
                    f"paper_i_hh_fullmeta_singleton_snake_phase3_child_symmetry_policy_mismatch:{label}:{policy}:expected:{expected_child_policy}"
                )
            expected_cap = "3" if (ordered_batch_beam or recovery_candidate) else "1"
            if cap != expected_cap:
                blockers.append(
                    f"paper_i_hh_fullmeta_singleton_snake_phase3_cap_mismatch:{label}:{cap}:expected:{expected_cap}"
                )
        elif method in {"geo", "append"}:
            mode = str(row.get("generic_adapt_runtime_split_mode") or "").strip()
            policy = str(row.get("generic_adapt_runtime_split_symmetry_policy") or "").strip()
            cap = str(row.get("generic_adapt_runtime_split_max_subset_size") or "").strip()
            expected_generic_policy = "hard_guard" if label == "A_native_staged_singleton_hard_guard" else "off"
            if mode != "shortlist_pauli_children_v1":
                blockers.append(f"paper_i_hh_fullmeta_singleton_generic_split_mode_mismatch:{label}:{mode}")
            if policy != expected_generic_policy:
                blockers.append(
                    f"paper_i_hh_fullmeta_singleton_generic_split_symmetry_policy_mismatch:{label}:{policy}:expected:{expected_generic_policy}"
                )
            if cap != "1":
                blockers.append(f"paper_i_hh_fullmeta_singleton_generic_split_cap_mismatch:{label}:{cap}:expected:1")
    if label.startswith("B_common_phase0_singleton_"):
        mode = str(row.get("shared_pauli_pool_mode") or "").strip()
        policy = str(row.get("shared_pauli_pool_symmetry_policy") or "").strip()
        cap = str(row.get("shared_pauli_pool_max_subset_size") or "").strip()
        if str(row.get("child_policy") or "").strip() != "common_phase0_singleton":
            blockers.append(f"paper_i_hh_fullmeta_singleton_child_policy_mismatch:{label}")
        if mode != "shared_pauli_child_sets_v1":
            blockers.append(f"paper_i_hh_fullmeta_singleton_shared_pool_mode_mismatch:{label}:{mode}")
        if policy != expected_symmetry:
            blockers.append(
                f"paper_i_hh_fullmeta_singleton_shared_pool_symmetry_policy_mismatch:{label}:{policy}:expected:{expected_symmetry}"
            )
        if cap != "1":
            blockers.append(f"paper_i_hh_fullmeta_singleton_shared_pool_cap_mismatch:{label}:{cap}:expected:1")
    if ordered_batch_beam:
        expected_run_class = "candidate" if recovery_candidate else "diagnostic"
        expected_target_size = "3" if recovery_candidate else "5"
        expected_size_cap = "3" if recovery_candidate else "5"
        allowed_optimizers = {"powell", "rotosolve", "spsa"} if recovery_candidate else {"powell"}
        allowed_regimes = set(_HH_FULLMETA_SINGLETON_ORDERED_BATCH_BEAM_REGIMES)
        if str(row.get("run_class") or "").strip() != expected_run_class:
            blockers.append("paper_i_hh_ordered_batch_beam_run_class_mismatch")
        if str(row.get("method_key") or "").strip() != "snake":
            blockers.append("paper_i_hh_ordered_batch_beam_method_mismatch")
        if str(row.get("display_regime") or "").strip() not in allowed_regimes:
            blockers.append("paper_i_hh_ordered_batch_beam_regime_mismatch")
        if str(row.get("adapt_optimizer_kind") or "").strip() not in allowed_optimizers:
            blockers.append("paper_i_hh_ordered_batch_beam_optimizer_mismatch")
        if str(row.get("static_route_id") or "").strip() != "unspecified":
            blockers.append("paper_i_hh_ordered_batch_beam_static_route_id_mismatch")
        if str(row.get("ordered_batch_beam_enabled") or "").strip() != "true":
            blockers.append("paper_i_hh_ordered_batch_beam_not_enabled")
        if str(row.get("phase2_batch_selection_mode") or "").strip() not in {
            "greedy_reduced_plane",
            "combinatorial_reduced_plane",
        }:
            blockers.append("paper_i_hh_ordered_batch_beam_selection_mode_mismatch")
        if str(row.get("phase2_batch_target_size") or "").strip() != expected_target_size:
            blockers.append("paper_i_hh_ordered_batch_beam_target_size_mismatch")
        if str(row.get("phase2_batch_size_cap") or "").strip() != expected_size_cap:
            blockers.append("paper_i_hh_ordered_batch_beam_size_cap_mismatch")
        if str(row.get("adapt_beam_live_branches") or "").strip() != "3":
            blockers.append("paper_i_hh_ordered_batch_beam_live_branches_mismatch")
        if str(row.get("adapt_beam_children_per_parent") or "").strip() != "3":
            blockers.append("paper_i_hh_ordered_batch_beam_children_per_parent_mismatch")
        try:
            lambda_beam = float(str(row.get("adapt_beam_lambda") or ""))
        except ValueError:
            blockers.append("paper_i_hh_ordered_batch_beam_lambda_missing")
        else:
            if lambda_beam < 0.0:
                blockers.append("paper_i_hh_ordered_batch_beam_lambda_negative")
        try:
            overrides = json.loads(str(row.get("snake_cli_overrides_json") or "{}"))
        except json.JSONDecodeError:
            blockers.append("paper_i_hh_ordered_batch_beam_overrides_invalid_json")
        else:
            set_flags = overrides.get("set_flags") if isinstance(overrides, Mapping) else None
            enable_flags = overrides.get("enable_flags") if isinstance(overrides, Mapping) else None
            if not isinstance(set_flags, Mapping):
                blockers.append("paper_i_hh_ordered_batch_beam_set_flags_missing")
            else:
                expected_pairs = {
                    "--static-route-id": "unspecified",
                    "--phase2-batch-selection-mode": str(row.get("phase2_batch_selection_mode") or "").strip(),
                    "--phase3-batch-selection-mode": str(row.get("phase2_batch_selection_mode") or "").strip(),
                    "--phase2-batch-target-size": expected_target_size,
                    "--phase2-batch-size-cap": expected_size_cap,
                    "--adapt-beam-live-branches": "3",
                    "--adapt-beam-children-per-parent": "3",
                    "--adapt-beam-lambda": str(row.get("adapt_beam_lambda") or "").strip(),
                }
                for flag, expected in expected_pairs.items():
                    if str(set_flags.get(flag) or "").strip() != expected:
                        blockers.append(f"paper_i_hh_ordered_batch_beam_override_mismatch:{flag}")
            if not isinstance(enable_flags, Sequence) or isinstance(enable_flags, (str, bytes)):
                blockers.append("paper_i_hh_ordered_batch_beam_enable_flags_missing")
            elif "--phase2-enable-batching" not in {str(flag) for flag in enable_flags}:
                blockers.append("paper_i_hh_ordered_batch_beam_phase2_enable_missing")
    return blockers


_SCALING_HH_CACHE_VALIDATION_MEMO: dict[tuple[Any, ...], tuple[str, ...]] = {}


def _paper_i_scaling_hh_cache_blockers(
    row: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    manifest_value = str(row.get("hh_pool_cache_manifest") or "").strip()
    pool_dir_value = str(row.get("hh_pool_cache_dir") or "").strip()
    registry_dir_value = str(row.get("hh_generator_registry_cache_dir") or "").strip()
    try:
        manifest_path = run_task._resolve_under_repo(manifest_value, repo_root=repo_root)
        pool_dir = run_task._resolve_under_repo(pool_dir_value, repo_root=repo_root)
        registry_dir = run_task._resolve_under_repo(registry_dir_value, repo_root=repo_root)
        file_signature = tuple(
            (
                str(path),
                int(path.stat().st_size),
                int(path.stat().st_mtime_ns),
            )
            for directory in (pool_dir, registry_dir)
            for path in sorted(directory.iterdir())
            if path.is_file() and path.suffix == ".pickle"
        )
        memo_key = (
            str(manifest_path),
            run_task.sha256_file(manifest_path),
            str(pool_dir),
            str(registry_dir),
            file_signature,
        )
    except Exception as exc:
        return [f"paper_i_scaling_hh_cache_manifest_invalid:{type(exc).__name__}:{exc}"]
    cached = _SCALING_HH_CACHE_VALIDATION_MEMO.get(memo_key)
    if cached is not None:
        return list(cached)

    blockers: list[str] = []
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(payload.get("schema") or "") != "paper_i_scaling_matrix_hh_dual_cache_prewarm_v1":
            blockers.append(f"paper_i_scaling_hh_cache_schema_mismatch:{payload.get('schema')}")
        if str(payload.get("status") or "").strip() != "pass":
            blockers.append(f"paper_i_scaling_hh_cache_prewarm_not_pass:{payload.get('status')}")
        if int(payload.get("case_count") or 0) != 12:
            blockers.append(f"paper_i_scaling_hh_cache_case_count_mismatch:{payload.get('case_count')}")

        sections = (
            ("pool_cache", pool_dir, True),
            ("generator_registry_cache", registry_dir, False),
        )
        computed_total_size = 0
        listed_path_sets: dict[str, set[Path]] = {}
        for section_name, expected_dir, require_exact_scope in sections:
            section = payload.get(section_name)
            if not isinstance(section, Mapping):
                blockers.append(f"paper_i_scaling_hh_{section_name}_section_missing")
                continue
            if str(section.get("mode") or "").strip() != "disk":
                blockers.append(f"paper_i_scaling_hh_{section_name}_mode_mismatch:{section.get('mode')}")
            if require_exact_scope and str(section.get("scope") or "").strip() != "exact":
                blockers.append("paper_i_scaling_hh_pool_cache_manifest_scope_mismatch")
            manifest_dir = run_task._resolve_under_repo(
                str(section.get("cache_dir") or ""),
                repo_root=repo_root,
            ).resolve()
            if manifest_dir != expected_dir.resolve():
                blockers.append(
                    f"paper_i_scaling_hh_{section_name}_dir_mismatch:{manifest_dir}:expected:{expected_dir.resolve()}"
                )
            files = section.get("files")
            if not isinstance(files, list) or len(files) != 12:
                blockers.append(
                    f"paper_i_scaling_hh_{section_name}_manifest_file_count_mismatch:"
                    f"{len(files) if isinstance(files, list) else 'invalid'}:expected:12"
                )
                files = []
            if int(section.get("file_count") or 0) != 12:
                blockers.append(
                    f"paper_i_scaling_hh_{section_name}_declared_file_count_mismatch:{section.get('file_count')}"
                )
            listed_paths: set[Path] = set()
            for item in files:
                if not isinstance(item, Mapping):
                    blockers.append(f"paper_i_scaling_hh_{section_name}_file_entry_invalid")
                    continue
                path = run_task._resolve_under_repo(str(item.get("path") or ""), repo_root=repo_root).resolve()
                listed_paths.add(path)
                if path.parent != expected_dir.resolve() or path.suffix != ".pickle":
                    blockers.append(f"paper_i_scaling_hh_{section_name}_file_path_invalid:{path}")
                    continue
                if not path.is_file():
                    blockers.append(f"paper_i_scaling_hh_{section_name}_file_missing:{path}")
                    continue
                actual_size = int(path.stat().st_size)
                computed_total_size += actual_size
                if actual_size != int(item.get("size_bytes") or -1):
                    blockers.append(f"paper_i_scaling_hh_{section_name}_file_size_mismatch:{path.name}")
                actual_sha = run_task.sha256_file(path)
                if actual_sha != str(item.get("sha256") or ""):
                    blockers.append(f"paper_i_scaling_hh_{section_name}_file_sha_mismatch:{path.name}")
            actual_paths = {
                path.resolve()
                for path in expected_dir.iterdir()
                if path.is_file() and path.suffix == ".pickle"
            }
            if actual_paths != listed_paths:
                blockers.append(
                    f"paper_i_scaling_hh_{section_name}_directory_set_mismatch:"
                    f"actual={len(actual_paths)}:manifest={len(listed_paths)}"
                )
            listed_path_sets[section_name] = listed_paths

        if int(payload.get("total_size_bytes") or -1) != computed_total_size:
            blockers.append(
                f"paper_i_scaling_hh_cache_total_size_mismatch:{payload.get('total_size_bytes')}:"
                f"expected:{computed_total_size}"
            )
        verified = payload.get("disk_hit_verification")
        if not isinstance(verified, list) or len(verified) != 12:
            blockers.append("paper_i_scaling_hh_cache_disk_hit_verification_incomplete")
        else:
            case_ids: set[str] = set()
            for item in verified:
                if not isinstance(item, Mapping):
                    blockers.append("paper_i_scaling_hh_cache_disk_hit_entry_invalid")
                    continue
                case_ids.add(str(item.get("case_id") or ""))
                if item.get("pool_cache_disk_hit_verified") is not True:
                    blockers.append(f"paper_i_scaling_hh_pool_cache_disk_hit_missing:{item.get('case_id')}")
                if item.get("generator_registry_cache_disk_hit_verified") is not True:
                    blockers.append(
                        f"paper_i_scaling_hh_generator_registry_cache_disk_hit_missing:{item.get('case_id')}"
                    )
                for field, section_name in (
                    ("pool_cache_path", "pool_cache"),
                    ("generator_registry_cache_path", "generator_registry_cache"),
                ):
                    event_path = run_task._resolve_under_repo(
                        str(item.get(field) or ""),
                        repo_root=repo_root,
                    ).resolve()
                    if event_path not in listed_path_sets.get(section_name, set()):
                        blockers.append(
                            f"paper_i_scaling_hh_cache_verified_path_not_manifested:{field}:{event_path.name}"
                        )
            if len(case_ids) != 12 or "" in case_ids:
                blockers.append(f"paper_i_scaling_hh_cache_verified_case_set_invalid:{len(case_ids)}")
    except Exception as exc:
        blockers.append(f"paper_i_scaling_hh_cache_manifest_invalid:{type(exc).__name__}:{exc}")

    _SCALING_HH_CACHE_VALIDATION_MEMO[memo_key] = tuple(blockers)
    return blockers


def _paper_i_scaling_implementation_lock_blockers(
    row: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    blockers: list[str] = []
    lock_value = str(row.get("implementation_lock") or "").strip()
    lock_sha = str(row.get("implementation_lock_sha256") or "").strip()
    bundle_value = str(row.get("code_bundle") or "").strip()
    bundle_sha = str(row.get("code_bundle_sha256") or "").strip()
    required_critical = {
        "pipelines/exact_bench/table_i_canonical_cases.py",
        "pipelines/exact_bench/static_reference_metrics.py",
        "pipelines/exact_bench/generic_static_benchmark.py",
        "pipelines/exact_bench/generic_static_adapt_variants.py",
        "pipelines/static_adapt/optimization/phase3_policy_optuna.py",
        "pipelines/static_adapt/adapt_pipeline.py",
        "pipelines/static_adapt/cli_config.py",
        "pipelines/static_adapt/engine_support.py",
        "pipelines/static_adapt/joint_step_warm_start.py",
        "pipelines/static_adapt/paper_i_runner.py",
        "pipelines/static_adapt/resume_scaffold.py",
        "pipelines/static_adapt/sector_invariants.py",
        "src/quantum/compiled_ansatz.py",
        "chtc/phase3_optuna/run_paper_i_scaling_matrix_cell.py",
    }
    try:
        lock_path = run_task._resolve_under_repo(lock_value, repo_root=repo_root)
        actual_lock_sha = run_task.sha256_file(lock_path)
        if actual_lock_sha != lock_sha:
            blockers.append(f"paper_i_scaling_implementation_lock_sha_mismatch:{actual_lock_sha}:{lock_sha}")
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        if str(payload.get("schema") or "") != "paper_i_scaling_matrix_implementation_lock_v1":
            blockers.append(f"paper_i_scaling_implementation_lock_schema_mismatch:{payload.get('schema')}")
        if str(payload.get("status") or "") != "pass":
            blockers.append(f"paper_i_scaling_implementation_lock_not_pass:{payload.get('status')}")
        bundle_contract = payload.get("code_bundle")
        if not isinstance(bundle_contract, Mapping):
            blockers.append("paper_i_scaling_implementation_lock_bundle_contract_missing")
        else:
            if str(bundle_contract.get("path") or "") != bundle_value:
                blockers.append("paper_i_scaling_implementation_lock_bundle_path_mismatch")
            if str(bundle_contract.get("sha256") or "") != bundle_sha:
                blockers.append("paper_i_scaling_implementation_lock_bundle_sha_mismatch")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            blockers.append("paper_i_scaling_implementation_lock_entries_missing")
            entries = []
        entry_by_path = {
            str(item.get("path") or ""): item
            for item in entries
            if isinstance(item, Mapping) and str(item.get("path") or "")
        }
        critical_paths = {
            path for path, item in entry_by_path.items() if item.get("critical_bundle_member") is True
        }
        if critical_paths != required_critical:
            blockers.append(
                f"paper_i_scaling_implementation_lock_critical_set_mismatch:"
                f"actual={sorted(critical_paths)}:expected={sorted(required_critical)}"
            )
        if "pipelines/exact_bench/static_reference_metrics.py" not in entry_by_path:
            blockers.append("paper_i_scaling_exact_resolver_missing_from_implementation_lock")
        for rel, item in entry_by_path.items():
            local_path = run_task._resolve_under_repo(rel, repo_root=repo_root)
            if not local_path.is_file():
                blockers.append(f"paper_i_scaling_implementation_lock_local_file_missing:{rel}")
                continue
            actual_local_sha = run_task.sha256_file(local_path)
            if actual_local_sha != str(item.get("sha256") or ""):
                blockers.append(f"paper_i_scaling_implementation_lock_local_sha_mismatch:{rel}")

        bundle_path = run_task._resolve_under_repo(bundle_value, repo_root=repo_root)
        with tarfile.open(bundle_path, "r:gz") as archive:
            archive_names = set(archive.getnames())
            for rel in sorted(required_critical):
                item = entry_by_path.get(rel)
                if not isinstance(item, Mapping):
                    blockers.append(f"paper_i_scaling_implementation_lock_critical_entry_missing:{rel}")
                    continue
                if rel not in archive_names:
                    blockers.append(f"paper_i_scaling_code_bundle_critical_member_missing:{rel}")
                    continue
                extracted = archive.extractfile(rel)
                if extracted is None:
                    blockers.append(f"paper_i_scaling_code_bundle_critical_member_unreadable:{rel}")
                    continue
                actual_member_sha = hashlib.sha256(extracted.read()).hexdigest()
                if actual_member_sha != str(item.get("sha256") or ""):
                    blockers.append(f"paper_i_scaling_code_bundle_critical_member_sha_mismatch:{rel}")
                if actual_member_sha != str(item.get("bundle_member_sha256") or ""):
                    blockers.append(f"paper_i_scaling_implementation_lock_bundle_member_sha_mismatch:{rel}")
    except Exception as exc:
        blockers.append(f"paper_i_scaling_implementation_lock_invalid:{type(exc).__name__}:{exc}")
    return blockers


def _generic_static_source_artifacts(row: dict[str, str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for field in (
        "phase3_policy_json",
        "hardware_resolution_profile_json",
        "reference_energy_cache_json",
        "trial_param_overrides_json",
        "enqueue_trial_params_json",
        "selected_logical_source_json",
        "paper_i_ladder_candidate_manifest_json",
        "paper_i_ladder_source_audit_json",
        "hh_pool_cache_dir",
        "hh_pool_cache_manifest",
        "hh_generator_registry_cache_dir",
        "code_bundle",
        "implementation_lock",
        "exact_energy_manifest",
        "implementation_repair_audit",
        "settings_diff_json",
        "source_result_json_local",
        "source_cell_manifest_local",
    ):
        value = str(row.get(field) or "").strip()
        if value and value.lower() != "none":
            out.append((field, value))
    return out


def _paper_i_scaling_append_powell_cap_repair_audit_blockers(
    row: Mapping[str, str],
    *,
    repo_root: Path = REPO_ROOT,
) -> list[str]:
    if str(row.get("repair_scope") or "").strip() != (
        _PAPER_I_SCALING_APPEND_POWELL_CAP_REPAIR_SCOPE
    ):
        return []
    blockers: list[str] = []
    audit_value = str(row.get("implementation_repair_audit") or "").strip()
    diff_value = str(row.get("settings_diff_json") or "").strip()
    try:
        audit_path = run_task._resolve_under_repo(audit_value, repo_root=repo_root)
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if str(audit.get("schema") or "") != (
            "paper_i_user_approved_implementation_repair_audit_v1"
        ):
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_schema_mismatch")
        if str(audit.get("classification") or "") != (
            "implementation_repair_not_sensitivity_sweep"
        ):
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_classification_mismatch")
        if audit.get("source_locked_sensitivity_claim") is not False:
            blockers.append("paper_i_scaling_append_powell_cap_repair_false_source_lock_claim")
        if str(audit.get("status") or "") != "approved_implementation_repair_prepared":
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_status_mismatch")
        source = audit.get("source")
        if not isinstance(source, Mapping):
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_source_invalid")
        else:
            source_hash_fields = {
                "source_result_json_sha256": "source_sha256",
                "source_cell_manifest_sha256": "source_cell_manifest_sha256",
            }
            for row_field, audit_field in source_hash_fields.items():
                if str(source.get(audit_field) or "") != str(row.get(row_field) or ""):
                    blockers.append(
                        f"paper_i_scaling_append_powell_cap_repair_audit_{audit_field}_mismatch"
                    )
            for row_field, audit_field in (
                ("source_result_json_local", "source_json_local"),
                ("source_cell_manifest_local", "source_cell_manifest_local"),
            ):
                if str(source.get(audit_field) or "") != str(row.get(row_field) or ""):
                    blockers.append(
                        f"paper_i_scaling_append_powell_cap_repair_audit_{audit_field}_mismatch"
                    )
        prepared_rows = audit.get("prepared_rows")
        prepared = prepared_rows[0] if isinstance(prepared_rows, list) and len(prepared_rows) == 1 else None
        if not isinstance(prepared, Mapping):
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_prepared_row_invalid")
        else:
            if list(prepared.get("changed_fields_vs_source") or []) != [
                "powell_maxiter_cap_policy"
            ]:
                blockers.append("paper_i_scaling_append_powell_cap_repair_audit_changed_fields_mismatch")
            if list(prepared.get("declared_record_non_changed_fields_diff") or []) != []:
                blockers.append("paper_i_scaling_append_powell_cap_repair_audit_declared_diff_nonempty")
        anchor = audit.get("source_value_anchor")
        if not isinstance(anchor, Mapping) or str(anchor.get("status") or "") != "not_claimed":
            blockers.append("paper_i_scaling_append_powell_cap_repair_audit_anchor_claim_mismatch")
    except Exception as exc:
        blockers.append(
            "paper_i_scaling_append_powell_cap_repair_audit_invalid:"
            f"{type(exc).__name__}:{exc}"
        )

    for path_field, hash_field in (
        ("source_result_json_local", "source_result_json_sha256"),
        ("source_cell_manifest_local", "source_cell_manifest_sha256"),
    ):
        try:
            source_path = run_task._resolve_under_repo(
                str(row.get(path_field) or ""),
                repo_root=repo_root,
            )
            actual_sha = run_task.sha256_file(source_path)
            expected_sha = str(row.get(hash_field) or "")
            if actual_sha != expected_sha:
                blockers.append(
                    f"paper_i_scaling_append_powell_cap_repair_{path_field}_sha_mismatch:"
                    f"{actual_sha}:expected:{expected_sha}"
                )
        except Exception as exc:
            blockers.append(
                f"paper_i_scaling_append_powell_cap_repair_{path_field}_invalid:"
                f"{type(exc).__name__}:{exc}"
            )

    try:
        diff_path = run_task._resolve_under_repo(diff_value, repo_root=repo_root)
        settings_diff = json.loads(diff_path.read_text(encoding="utf-8"))
        expected_science_diff = {
            "powell_maxiter_cap_policy": {
                "source": "strict_failure_v1",
                "repair": "accept_finite_nonincreasing_v1",
            }
        }
        if str(settings_diff.get("schema") or "") != (
            "paper_i_scaling_matrix_single_row_settings_diff_v1"
        ):
            blockers.append("paper_i_scaling_append_powell_cap_repair_settings_diff_schema_mismatch")
        if settings_diff.get("declared_record_science_settings_diff") != expected_science_diff:
            blockers.append("paper_i_scaling_append_powell_cap_repair_settings_diff_content_mismatch")
        if str(settings_diff.get("source_lock_status") or "") != "not_claimed_not_evaluated":
            blockers.append("paper_i_scaling_append_powell_cap_repair_settings_diff_source_lock_claim")
        if str(settings_diff.get("status") or "") != "pass_declared_record_diff_not_source_lock":
            blockers.append("paper_i_scaling_append_powell_cap_repair_settings_diff_status_mismatch")
    except Exception as exc:
        blockers.append(
            "paper_i_scaling_append_powell_cap_repair_settings_diff_invalid:"
            f"{type(exc).__name__}:{exc}"
        )
    return blockers


def build_generic_static_table_preflight_manifest(
    row: dict[str, str],
    *,
    record_id: str,
    records_path: Path,
    submit_path: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id
    from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec

    contract = run_task.parse_submit_contract(submit_path)
    transfer_inputs = list(contract.get("transfer_input_files") or [])
    blockers: list[str] = []
    paper_i_scaling_matrix = _is_paper_i_scaling_matrix(row)
    records_rel = run_task._safe_relative(run_task._resolve_under_repo(records_path, repo_root=repo_root), repo_root=repo_root)
    submit_records = str(contract.get("records_path") or contract.get("argument_records_path") or "").strip()
    if submit_records and submit_records != records_rel:
        blockers.append(f"records_path_mismatch:submit={submit_records}:preflight={records_rel}")
    if submit_records and not run_task._is_sandbox_visible(submit_records, transfer_input_files=transfer_inputs, repo_root=repo_root):
        blockers.append(f"submit_records_not_transferred:{submit_records}")
    queue_file = str(contract.get("queue_record_id_file") or "").strip()
    if queue_file:
        queue_path = run_task._resolve_under_repo(queue_file, repo_root=repo_root)
        if not queue_path.exists():
            blockers.append(f"queue_record_id_file_missing:{queue_file}")
        elif not run_task._is_sandbox_visible(queue_file, transfer_input_files=transfer_inputs, repo_root=repo_root):
            blockers.append(f"queue_record_id_file_not_transferred:{queue_file}")
    if paper_i_scaling_matrix:
        blockers.extend(_paper_i_scaling_transfer_blockers(transfer_inputs, records_rel=records_rel))
        blockers.extend(_paper_i_scaling_matrix_contract_blockers(row))

    source_artifacts: list[dict[str, Any]] = []
    for field, value in _generic_static_source_artifacts(row):
        resolved = run_task._resolve_under_repo(value, repo_root=repo_root)
        exists = resolved.exists()
        sandbox_visible = run_task._is_sandbox_visible(value, transfer_input_files=transfer_inputs, repo_root=repo_root)
        source_artifacts.append(
            {
                "field": field,
                "value": value,
                "resolved_path": str(resolved),
                "repo_relative_path": run_task._safe_relative(resolved, repo_root=repo_root),
                "exists": bool(exists),
                "is_dir": bool(resolved.is_dir()) if exists else False,
                "sandbox_visible": bool(sandbox_visible),
                "sha256": run_task.sha256_file(resolved) if exists and resolved.is_file() else None,
            }
        )
        if not exists:
            blockers.append(f"source_artifact_missing:{field}:{value}")
        if not sandbox_visible:
            blockers.append(f"source_artifact_not_transferred:{field}:{value}")

    if paper_i_scaling_matrix:
        blockers.extend(_paper_i_scaling_snake_policy_blockers(row, repo_root=repo_root))
        bundle_value = str(row.get("code_bundle") or "").strip()
        bundle_sha = str(row.get("code_bundle_sha256") or "").strip()
        try:
            bundle_path = run_task._resolve_under_repo(bundle_value, repo_root=repo_root)
            actual_bundle_sha = run_task.sha256_file(bundle_path)
            if actual_bundle_sha != bundle_sha:
                blockers.append(f"paper_i_scaling_code_bundle_sha_mismatch:{actual_bundle_sha}:{bundle_sha}")
        except Exception as exc:
            blockers.append(f"paper_i_scaling_code_bundle_invalid:{type(exc).__name__}:{exc}")
        blockers.extend(_paper_i_scaling_implementation_lock_blockers(row, repo_root=repo_root))

        exact_manifest_value = str(row.get("exact_energy_manifest") or "").strip()
        exact_manifest_sha = str(row.get("exact_energy_manifest_sha256") or "").strip()
        try:
            exact_manifest_path = run_task._resolve_under_repo(exact_manifest_value, repo_root=repo_root)
            actual_exact_manifest_sha = run_task.sha256_file(exact_manifest_path)
            if actual_exact_manifest_sha != exact_manifest_sha:
                blockers.append(
                    f"paper_i_scaling_exact_manifest_sha_mismatch:{actual_exact_manifest_sha}:{exact_manifest_sha}"
                )
            exact_payload = json.loads(exact_manifest_path.read_text(encoding="utf-8"))
            exact_records = exact_payload.get("records")
            exact_row = exact_records.get(str(row.get("case_id") or "")) if isinstance(exact_records, Mapping) else None
            if not isinstance(exact_row, Mapping):
                blockers.append(f"paper_i_scaling_exact_manifest_case_missing:{row.get('case_id')}")
            else:
                expected_energy = float(exact_row.get("exact_energy"))
                for field in ("same_cutoff_exact_gs_energy", "exact_reference_energy"):
                    actual_energy = float(str(row.get(field) or "nan"))
                    if not math.isfinite(actual_energy) or not math.isclose(
                        actual_energy,
                        expected_energy,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    ):
                        blockers.append(
                            f"paper_i_scaling_exact_energy_mismatch:{field}:{actual_energy}:{expected_energy}"
                        )
                if str(row.get("exact_energy_key") or "").strip() != str(exact_row.get("key_hash") or "").strip():
                    blockers.append("paper_i_scaling_exact_energy_key_mismatch")
        except Exception as exc:
            blockers.append(f"paper_i_scaling_exact_manifest_invalid:{type(exc).__name__}:{exc}")
        blockers.extend(
            _paper_i_scaling_append_powell_cap_repair_audit_blockers(
                row,
                repo_root=repo_root,
            )
        )

    if paper_i_scaling_matrix and str(row.get("family") or "").strip() == "hh":
        blockers.extend(_paper_i_scaling_hh_cache_blockers(row, repo_root=repo_root))

    spec_summary: dict[str, Any] = {}
    cutoff_floor_diagnostics: list[dict[str, Any]] = []
    spec: Any | None = None
    try:
        spec = table_i_canonical_spec_by_case_id(
            str(row.get("family") or ""),
            str(row.get("case_id") or ""),
            str(row.get("suite_profile") or "standard"),
        )
        spec_summary = {
            "benchmark_id": spec.benchmark_id,
            "family": spec.family,
            "suite_profile": str(row.get("suite_profile") or "standard"),
            "exact_reference_n_ph_max": spec.exact_reference_n_ph_max,
        }
    except Exception as exc:
        blockers.append(f"non_executable_table_i_case:{row.get('family')}:{row.get('case_id')}:{type(exc).__name__}:{exc}")

    n_ph_work = str(row.get("n_ph_work") or "").strip()
    n_ph_ref = str(row.get("n_ph_ref") or "").strip()
    hh_tableiii_shot_proxy_repair = _is_hh_tableiii_snake_shot_proxy_repair(row)
    hh_tableiii_exact_prefix_recovery = _is_hh_tableiii_snake_exact_prefix_recovery(row)
    hh_u8_source_locked_replay = _is_hh_u8_snake_source_locked_replay(row)
    hh_spsa_budget_ladder = _is_hh_spsa_budget_ladder(row)
    hh_snake_pauli_child_repair = _is_hh_snake_pauli_child_repair(row)
    hh_rotosolve_macro = _is_hh_rotosolve_macro(row)
    hh_shared_pool_optimizer = _is_hh_shared_pool_optimizer(row)
    hh_fullmeta_phase3_singleton = _is_hh_fullmeta_phase3_singleton(row)
    hh_weak_weak_snake_mechanism_ablation = _is_hh_weak_weak_snake_mechanism_ablation(row)
    hh_all_regime_snake_mechanism_ablation = _is_hh_all_regime_snake_mechanism_ablation(row)
    if hh_fullmeta_phase3_singleton:
        for blocker in _hh_fullmeta_phase3_singleton_contract_blockers(row):
            _append_unique_blocker(blockers, blocker)
    if hh_weak_weak_snake_mechanism_ablation:
        for blocker in _hh_weak_weak_snake_mechanism_ablation_contract_blockers(row):
            _append_unique_blocker(blockers, blocker)
    if hh_all_regime_snake_mechanism_ablation:
        for blocker in _hh_all_regime_snake_mechanism_ablation_contract_blockers(row):
            _append_unique_blocker(blockers, blocker)
    if spec is not None and n_ph_work and n_ph_ref:
        if hh_tableiii_shot_proxy_repair or hh_tableiii_exact_prefix_recovery:
            if str(row.get("table_label") or "").strip() != _HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_TABLE_LABEL:
                blockers.append(f"paper_i_hh_tableiii_snake_table_label_mismatch:{row.get('table_label')}")
            if str(row.get("requires_deterministic_shot_proxy") or "").strip().lower() != "true":
                blockers.append("paper_i_hh_tableiii_snake_requires_shot_proxy_flag_missing")
            if hh_tableiii_exact_prefix_recovery:
                for field in ("requires_s_alg", "requires_first_hit_resource_sidecar", "requires_strict_replay_json", "requires_per_prefix_resource_export"):
                    if str(row.get(field) or "").strip().lower() != "true":
                        blockers.append(f"paper_i_hh_tableiii_exact_prefix_requirement_missing:{field}")
            _positive_int_from_row(row, "shots_per_pauli_term_proxy", blockers, min_value=1)
            if str(row.get("primary_energy_metric") or "").strip() != _HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_PRIMARY_ENERGY_METRIC:
                blockers.append("paper_i_hh_tableiii_snake_primary_metric_mismatch")
            if str(row.get("same_cutoff_error_role") or "").strip() != _HH_TABLEIII_SNAKE_SHOT_PROXY_REPAIR_SAME_CUTOFF_ERROR_ROLE:
                blockers.append("paper_i_hh_tableiii_snake_same_cutoff_role_mismatch")
        elif hh_u8_source_locked_replay:
            if str(row.get("primary_energy_metric") or "").strip() != _HH_U8_SNAKE_SOURCE_LOCKED_REPLAY_PRIMARY_ENERGY_METRIC:
                blockers.append("paper_i_hh_u8_snake_source_locked_primary_metric_mismatch")
            if str(row.get("same_cutoff_error_role") or "").strip() != _HH_U8_SNAKE_SOURCE_LOCKED_REPLAY_SAME_CUTOFF_ERROR_ROLE:
                blockers.append("paper_i_hh_u8_snake_source_locked_same_cutoff_role_mismatch")
        elif hh_spsa_budget_ladder:
            if str(row.get("primary_energy_metric") or "").strip() != _HH_SPSA_BUDGET_LADDER_PRIMARY_ENERGY_METRIC:
                blockers.append("paper_i_hh_spsa_budget_ladder_primary_metric_mismatch")
            if str(row.get("same_cutoff_error_role") or "").strip() != _HH_SPSA_BUDGET_LADDER_SAME_CUTOFF_ERROR_ROLE:
                blockers.append("paper_i_hh_spsa_budget_ladder_same_cutoff_role_mismatch")
        elif hh_snake_pauli_child_repair:
            if (
                str(row.get("primary_energy_metric") or "").strip()
                != _HH_SNAKE_PAULI_CHILD_REPAIR_PRIMARY_ENERGY_METRIC
            ):
                blockers.append("paper_i_hh_snake_pauli_child_repair_primary_metric_mismatch")
            if (
                str(row.get("same_cutoff_error_role") or "").strip()
                != _HH_SNAKE_PAULI_CHILD_REPAIR_SAME_CUTOFF_ERROR_ROLE
            ):
                blockers.append("paper_i_hh_snake_pauli_child_repair_same_cutoff_role_mismatch")
        elif hh_rotosolve_macro:
            if (
                str(row.get("primary_energy_metric") or "").strip()
                != _HH_ROTOSOLVE_MACRO_PRIMARY_ENERGY_METRIC
            ):
                blockers.append("paper_i_hh_rotosolve_macro_primary_metric_mismatch")
            if (
                str(row.get("same_cutoff_error_role") or "").strip()
                != _HH_ROTOSOLVE_MACRO_SAME_CUTOFF_ERROR_ROLE
            ):
                blockers.append("paper_i_hh_rotosolve_macro_same_cutoff_role_mismatch")
        elif hh_shared_pool_optimizer:
            if (
                str(row.get("primary_energy_metric") or "").strip()
                != _HH_SHARED_POOL_OPTIMIZER_PRIMARY_ENERGY_METRIC
            ):
                blockers.append("paper_i_hh_shared_pool_optimizer_primary_metric_mismatch")
            if (
                str(row.get("same_cutoff_error_role") or "").strip()
                != _HH_SHARED_POOL_OPTIMIZER_SAME_CUTOFF_ERROR_ROLE
            ):
                blockers.append("paper_i_hh_shared_pool_optimizer_same_cutoff_role_mismatch")
        elif hh_fullmeta_phase3_singleton:
            if (
                str(row.get("primary_energy_metric") or "").strip()
                != _HH_FULLMETA_PHASE3_SINGLETON_PRIMARY_ENERGY_METRIC
            ):
                blockers.append("paper_i_hh_fullmeta_phase3_singleton_primary_metric_mismatch")
            if (
                str(row.get("same_cutoff_error_role") or "").strip()
                != _HH_FULLMETA_PHASE3_SINGLETON_SAME_CUTOFF_ERROR_ROLE
            ):
                blockers.append("paper_i_hh_fullmeta_phase3_singleton_same_cutoff_role_mismatch")
        elif hh_weak_weak_snake_mechanism_ablation or hh_all_regime_snake_mechanism_ablation:
            if (
                str(row.get("primary_energy_metric") or "").strip()
                != _HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_PRIMARY_ENERGY_METRIC
            ):
                blockers.append("paper_i_hh_mechanism_primary_metric_mismatch")
            if (
                str(row.get("same_cutoff_error_role") or "").strip()
                != _HH_WEAK_WEAK_SNAKE_MECHANISM_ABLATION_SAME_CUTOFF_ERROR_ROLE
            ):
                blockers.append("paper_i_hh_mechanism_same_cutoff_role_mismatch")
        elif paper_i_scaling_matrix:
            if str(row.get("primary_energy_metric") or "").strip() != "same_cutoff_abs_delta_e":
                blockers.append("paper_i_scaling_primary_metric_mismatch")
            if str(row.get("same_cutoff_error_role") or "").strip() != "primary":
                blockers.append("paper_i_scaling_same_cutoff_role_mismatch")
        else:
            if str(row.get("primary_energy_metric") or "").strip() != "higher_cutoff_reference_abs_delta_e":
                blockers.append("paper_i_phonon_primary_metric_mismatch")
            if str(row.get("same_cutoff_error_role") or "").strip() != "diagnostic_only":
                blockers.append("paper_i_phonon_same_cutoff_role_mismatch")
        spec_ref = getattr(spec, "exact_reference_n_ph_max", None)
        if spec_ref is not None and int(spec_ref) != int(n_ph_ref) and not hh_u8_source_locked_replay:
            blockers.append(f"paper_i_phonon_spec_ref_mismatch:{spec_ref}:{n_ph_ref}")
        if (
            hh_u8_source_locked_replay
            or hh_spsa_budget_ladder
            or hh_snake_pauli_child_repair
            or hh_rotosolve_macro
            or hh_shared_pool_optimizer
            or hh_fullmeta_phase3_singleton
            or hh_weak_weak_snake_mechanism_ablation
            or hh_all_regime_snake_mechanism_ablation
            or paper_i_scaling_matrix
        ):
            cutoff_floor_diagnostics.append(
                {
                    "benchmark_id": str(getattr(spec, "benchmark_id", row.get("case_id"))),
                    "family": str(getattr(spec, "family", row.get("family"))),
                    "n_ph_work": int(n_ph_work),
                    "n_ph_ref": int(n_ph_ref),
                    "status": "not_applicable_same_cutoff_primary_metric",
                }
            )
        else:
            try:
                tau_raw = str(row.get("tau_phys") or row.get("energy_stop_target") or PAPER_I_CLEAN_TAU_PHYS)
                tau_phys = float(tau_raw)
                same_energy, same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=int(n_ph_work))
                ref_energy, ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=int(n_ph_ref))
                floor = abs(float(same_energy) - float(ref_energy))
                status = "pass" if floor <= tau_phys else "fail"
                cutoff_floor_diagnostics.append(
                    {
                        "benchmark_id": str(getattr(spec, "benchmark_id", row.get("case_id"))),
                        "family": str(getattr(spec, "family", row.get("family"))),
                        "n_ph_work": int(n_ph_work),
                        "n_ph_ref": int(n_ph_ref),
                        "same_cutoff_energy": float(same_energy),
                        "reference_cutoff_energy": float(ref_energy),
                        "same_cutoff_reference_energy_key": same_key,
                        "reference_cutoff_energy_key": ref_key,
                        "exact_cutoff_floor_abs_delta_e": float(floor),
                        "tau_phys": float(tau_phys),
                        "status": status,
                    }
                )
                if status == "fail" and not (hh_tableiii_shot_proxy_repair or hh_tableiii_exact_prefix_recovery):
                    blockers.append(
                        "paper_i_cutoff_floor_exceeds_tau:"
                        f"{getattr(spec, 'benchmark_id', row.get('case_id'))}:"
                        f"n_ph_work={int(n_ph_work)}:n_ph_ref={int(n_ph_ref)}:"
                        f"floor={floor:.16g}:tau={tau_phys:.16g}"
                    )
            except Exception as exc:
                blockers.append(f"paper_i_cutoff_floor_check_failed:{type(exc).__name__}:{exc}")

    generic_adapt_stop_policy = str(row.get("generic_adapt_stop_policy") or "").strip()
    algorithm_id_value = str(row.get("algorithm_id") or "").strip()
    if generic_adapt_stop_policy:
        if algorithm_id_value in _PHASE3_STATIC_ADAPT_ALGORITHM_IDS:
            blockers.append(
                "generic_adapt_stop_policy_invalid_for_phase3_static_adapt:"
                f"{algorithm_id_value}:{generic_adapt_stop_policy}"
            )
        elif algorithm_id_value not in _GENERIC_STATIC_ADAPT_VARIANT_ALGORITHM_IDS:
            blockers.append(
                "generic_adapt_stop_policy_invalid_for_non_generic_static_adapt_variants:"
                f"{algorithm_id_value}:{generic_adapt_stop_policy}"
            )

    selected_route = str(row.get("selected_logical_route") or "").strip().lower().replace("-", "_")
    selected_source = str(row.get("selected_logical_source_json") or "").strip()
    selected_transfer = str(row.get("selected_logical_transfer_mode") or "exact_match_v1").strip().lower()
    if selected_route:
        if selected_route not in {"standard", "historical_selected"}:
            blockers.append(f"selected_logical_route_invalid:{selected_route}")
        if selected_transfer not in {"exact_match_v1", "boundary_v1"}:
            blockers.append(f"selected_logical_transfer_mode_invalid:{selected_transfer}")
        if selected_route == "historical_selected" and not selected_source:
            blockers.append("selected_logical_source_missing_for_historical_selected")
        selected_algorithm_id = str(row.get("algorithm_id") or "").strip()
        selected_logical_supported_algorithms = {
            "static_full_meta_append_adapt_vqe",
            "static_tetris_qubit_adapt_vqe",
            "static_geo_adapt_vqe",
            "static_pos_geo_adapt_vqe",
            "static_family_native_adapt_phase3",
        }
        if selected_route == "historical_selected" and selected_algorithm_id not in selected_logical_supported_algorithms:
            blockers.append(
                f"selected_logical_overlay_unsupported_algorithm:{row.get('algorithm_id')}"
            )

    ladder_stage = str(row.get("paper_i_cutoff_ladder_stage") or "").strip()
    if ladder_stage:
        if ladder_stage not in PAPER_I_LADDER_STAGE_CONFIGS:
            blockers.append(f"paper_i_ladder_unknown_stage:{ladder_stage}")
        else:
            stage_config = PAPER_I_LADDER_STAGE_CONFIGS[ladder_stage]
            expected_profile = stage_config.suite_profile
            expected_work = stage_config.n_ph_work
            expected_ref = stage_config.n_ph_ref
            requires_prior_failure = bool(stage_config.requires_prior_failure)
            requires_ref5 = bool(stage_config.requires_ref5_allowance)
            if str(row.get("suite_profile") or "").strip() != expected_profile:
                blockers.append(
                    f"paper_i_ladder_suite_profile_mismatch:{row.get('suite_profile')}:{expected_profile}"
                )
            if str(row.get("family") or "").strip() not in {"bose_hubbard", "harmonic_kerr_chain", "spin_boson", "hh"}:
                blockers.append(f"paper_i_ladder_non_phonon_family:{row.get('family')}")
            snake_policy = str(row.get("paper_i_ladder_snake_policy") or "").strip()
            if snake_policy not in {"included", "snake_only", "benchmarks_only_explicit"}:
                blockers.append(f"paper_i_ladder_snake_policy_missing_or_invalid:{snake_policy}")
            if (
                str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3"
                and snake_policy == "benchmarks_only_explicit"
            ):
                blockers.append("paper_i_ladder_benchmarks_only_contains_snake_phase3")
            if str(row.get("n_ph_work") or "").strip() != str(expected_work):
                blockers.append(f"paper_i_ladder_n_ph_work_mismatch:{row.get('n_ph_work')}:{expected_work}")
            if str(row.get("n_ph_ref") or "").strip() != str(expected_ref):
                blockers.append(f"paper_i_ladder_n_ph_ref_mismatch:{row.get('n_ph_ref')}:{expected_ref}")
            spec_ref = spec_summary.get("exact_reference_n_ph_max")
            if spec_ref is not None and int(spec_ref) != int(expected_ref):
                blockers.append(f"paper_i_ladder_spec_ref_mismatch:{spec_ref}:{expected_ref}")
            if str(row.get("primary_energy_metric") or "").strip() != "higher_cutoff_reference_abs_delta_e":
                blockers.append("paper_i_ladder_primary_metric_mismatch")
            if str(row.get("same_cutoff_error_role") or "").strip() != "diagnostic_only":
                blockers.append("paper_i_ladder_same_cutoff_role_mismatch")
            if not math.isclose(float(str(row.get("paper_i_ladder_acceptance_threshold") or "nan")), float(PAPER_I_CLEAN_TAU_TIGHT), rel_tol=0.0, abs_tol=1e-15):
                blockers.append("paper_i_ladder_acceptance_threshold_mismatch")
            if not math.isclose(float(str(row.get("tau_phys") or "nan")), float(PAPER_I_CLEAN_TAU_PHYS), rel_tol=0.0, abs_tol=1e-15):
                blockers.append("paper_i_ladder_tau_phys_mismatch")
            if not math.isclose(float(str(row.get("tau_tight") or "nan")), float(PAPER_I_CLEAN_TAU_TIGHT), rel_tol=0.0, abs_tol=1e-15):
                blockers.append("paper_i_ladder_tau_tight_mismatch")
            for numeric_field in ("same_cutoff_exact_gs_energy", "exact_reference_energy"):
                try:
                    value = float(str(row.get(numeric_field) or ""))
                    if not math.isfinite(value):
                        raise ValueError("not_finite")
                except Exception:
                    blockers.append(f"paper_i_ladder_{numeric_field}_missing_or_invalid")
            cache_value = str(row.get("reference_energy_cache_json") or "").strip()
            cache_key = str(row.get("reference_cutoff_energy_key") or "").strip()
            if not cache_value:
                blockers.append("paper_i_ladder_reference_energy_cache_missing")
            if not cache_key:
                blockers.append("paper_i_ladder_reference_cutoff_energy_key_missing")
            if cache_value and cache_key:
                cache_path = run_task._resolve_under_repo(cache_value, repo_root=repo_root)
                try:
                    cache_payload = json.loads(cache_path.read_text(encoding="utf-8"))
                    if cache_key not in dict(cache_payload.get("records") or {}):
                        blockers.append(f"paper_i_ladder_reference_cache_key_missing:{cache_key}")
                except Exception as exc:
                    blockers.append(f"paper_i_ladder_reference_cache_unreadable:{type(exc).__name__}:{exc}")
            if requires_prior_failure and not str(row.get("paper_i_ladder_escalation_reason") or "").strip():
                blockers.append("paper_i_ladder_missing_escalation_reason")
            if requires_ref5 and str(row.get("paper_i_ladder_allow_ref5") or "").strip().lower() != "true":
                blockers.append("paper_i_ladder_missing_ref5_allowance")
            blockers.extend(
                validate_clean_ladder_source_metadata(
                    row,
                    stage=ladder_stage,
                    target_case_id=str(row.get("case_id") or ""),
                )
            )
            if requires_prior_failure:
                lane = "snake" if str(row.get("algorithm_id") or "").strip() == "static_family_native_adapt_phase3" else "comparator"
                blockers.extend(
                    validate_candidate_row_authorization(
                        row,
                        target_case_id=str(row.get("case_id") or ""),
                        lane=lane,
                        algorithm_id=str(row.get("algorithm_id") or ""),
                        repo_root=repo_root,
                    )
                )

    return {
        "schema": "generic_static_table_chtc_preflight_manifest_v1",
        "status": "fail" if blockers else "pass",
        "ok": not blockers,
        "record_id": record_id,
        "records_path": records_rel,
        "submit_contract": contract,
        "transfer_input_files": transfer_inputs,
        "blocking_reasons": blockers,
        "source_artifacts": source_artifacts,
        "paper_i_cutoff_floor": cutoff_floor_diagnostics,
        "table_i_case": {
            "family": row.get("family"),
            "case_id": row.get("case_id"),
            "algorithm_id": row.get("algorithm_id"),
            "suite_profile": row.get("suite_profile"),
            "paper_i_cutoff_ladder_stage": row.get("paper_i_cutoff_ladder_stage"),
            "paper_i_ladder_snake_policy": row.get("paper_i_ladder_snake_policy"),
            "run_class": row.get("run_class"),
            "requires_deterministic_shot_proxy": row.get("requires_deterministic_shot_proxy"),
            "shots_per_pauli_term_proxy": row.get("shots_per_pauli_term_proxy"),
            "hh_tableiii_regime": row.get("hh_tableiii_regime"),
            "table_label": row.get("table_label"),
            "is_hh_tableiii_snake_shot_proxy_repair": hh_tableiii_shot_proxy_repair,
            "is_hh_tableiii_snake_exact_prefix_recovery": hh_tableiii_exact_prefix_recovery,
            "is_paper_i_scaling_matrix": paper_i_scaling_matrix,
            "spec": spec_summary,
        },
    }


def build_preflight_bundle(
    *,
    submit_path: Path,
    records_path: Path | None = None,
    record_ids: Sequence[str] | None = None,
    record_id_file: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    submit_path = _resolve(submit_path, repo_root=repo_root)
    contract = run_task.parse_submit_contract(submit_path)
    resolved_records = _resolve(records_path or contract.get("records_path") or contract.get("argument_records_path"), repo_root=repo_root)
    if record_ids is None or not tuple(record_ids):
        id_file_value = record_id_file or contract["queue_record_id_file"]
        if not id_file_value:
            raise ValueError("record ids were not provided and submit file has no `queue record_id from ...` line")
        record_ids = _read_record_ids(_resolve(id_file_value, repo_root=repo_root))
    manifests: list[dict[str, Any]] = []
    calibration_repair_rows: list[dict[str, str]] = []
    selected_rows: list[dict[str, str]] = []
    for record_id in record_ids:
        row = run_task.load_record(resolved_records, str(record_id))
        selected_rows.append(dict(row))
        if str(row.get("profile_id") or "").strip() == PAPER_I_HH_SHARED_SPSA_CALIBRATION_PROFILE_ID:
            manifests.append(
                build_paper_i_hh_shared_spsa_preflight_manifest(
                    row,
                    record_id=str(record_id),
                    records_path=resolved_records,
                    submit_path=submit_path,
                    repo_root=repo_root,
                )
            )
        elif str(row.get("profile_id") or "").strip() == PAPER_I_HH_U8_COMPARATOR_SPSA_PROFILE_ID:
            manifests.append(
                build_paper_i_hh_u8_comparator_spsa_preflight_manifest(
                    row,
                    record_id=str(record_id),
                    records_path=resolved_records,
                    submit_path=submit_path,
                    repo_root=repo_root,
                )
            )
        elif str(row.get("profile_id") or "").strip() == PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
            if str(row.get("repair_scope") or "").strip() == HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
                calibration_repair_rows.append(dict(row))
            manifests.append(
                build_paper_i_comparator_spsa_calibration_preflight_manifest(
                    row,
                    record_id=str(record_id),
                    records_path=resolved_records,
                    submit_path=submit_path,
                    repo_root=repo_root,
                )
            )
        elif "phase3_policy_json" in row or "case_id" in row:
            manifests.append(
                build_generic_static_table_preflight_manifest(
                    row,
                    record_id=str(record_id),
                    records_path=resolved_records,
                    submit_path=submit_path,
                    repo_root=repo_root,
                )
            )
        else:
            manifests.append(
                run_task.build_phase3_preflight_manifest(
                    row,
                    record_id=str(record_id),
                    records_path=resolved_records,
                    submit_path=submit_path,
                    repo_root=repo_root,
                )
            )
    bundle_blockers: list[str] = []
    if any(_is_paper_i_scaling_matrix(row) for row in selected_rows):
        queue_value = str(contract.get("queue_record_id_file") or "").strip()
        try:
            queue_ids = _read_record_ids(_resolve(queue_value, repo_root=repo_root))
            queue_rows = [run_task.load_record(resolved_records, record_id) for record_id in queue_ids]
            bundle_blockers.extend(_paper_i_scaling_matrix_bundle_blockers(queue_rows))
            records_rel = run_task._safe_relative(
                run_task._resolve_under_repo(resolved_records, repo_root=repo_root),
                repo_root=repo_root,
            )
            bundle_blockers.extend(
                _paper_i_scaling_submit_contract_blockers(
                    contract,
                    queue_rows,
                    records_rel=records_rel,
                    repo_root=repo_root,
                )
            )
        except Exception as exc:
            bundle_blockers.append(f"paper_i_scaling_bundle_queue_validation_failed:{type(exc).__name__}:{exc}")
    ladder_manifests = [
        item
        for item in manifests
        if str(((item.get("table_i_case") or {}).get("paper_i_cutoff_ladder_stage")) or "").strip()
        or str(((item.get("paper_i_cutoff_ladder") or {}).get("stage")) or "").strip()
    ]
    if ladder_manifests:
        policies = {
            str(
                ((item.get("table_i_case") or {}).get("paper_i_ladder_snake_policy"))
                or ((item.get("paper_i_cutoff_ladder") or {}).get("snake_policy"))
                or ""
            ).strip()
            for item in ladder_manifests
        }
        if len(policies) != 1:
            bundle_blockers.append(f"paper_i_ladder_mixed_snake_policy:{sorted(policies)}")
        policy = next(iter(policies)) if len(policies) == 1 else ""
        has_snake = any(
            str(((item.get("table_i_case") or {}).get("algorithm_id")) or "").strip() == "static_family_native_adapt_phase3"
            or str(((item.get("paper_i_cutoff_ladder") or {}).get("snake_policy")) or "").strip() == "snake_only"
            for item in ladder_manifests
        )
        if policy in {"included", "snake_only"} and not has_snake:
            bundle_blockers.append(f"paper_i_ladder_missing_snake_rows_for_policy:{policy}")
    if calibration_repair_rows:
        from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (  # noqa: WPS433
            HH_TABLEIII_REPAIR_METHOD_IDS,
            HH_TABLEIII_REPAIR_TARGET_IDS,
            validate_hh_tableiii_repair_records,
        )

        try:
            validate_hh_tableiii_repair_records(calibration_repair_rows)
        except Exception as exc:
            bundle_blockers.append(f"hh_tableiii_repair_matrix_invalid:{type(exc).__name__}:{exc}")
        repair_pairs = {
            (str(row.get("method_id") or row.get("algorithm_id") or ""), str(row.get("target_id") or ""))
            for row in calibration_repair_rows
        }
        expected_pairs = {(method_id, target_id) for method_id in HH_TABLEIII_REPAIR_METHOD_IDS for target_id in HH_TABLEIII_REPAIR_TARGET_IDS}
        if repair_pairs != expected_pairs:
            bundle_blockers.append(
                "hh_tableiii_repair_matrix_incomplete:"
                f"missing={sorted(expected_pairs - repair_pairs)}:extra={sorted(repair_pairs - expected_pairs)}"
            )
    failed = [item for item in manifests if not bool(item.get("ok", False))]
    ok = (not failed) and (not bundle_blockers)
    return {
        "schema": "phase3_chtc_submit_preflight_bundle_v1",
        "submit_path": str(submit_path),
        "records_path": str(resolved_records),
        "record_ids": [str(record_id) for record_id in record_ids],
        "status": "fail" if not ok else "pass",
        "ok": bool(ok),
        "record_count": len(manifests),
        "failed_record_count": len(failed),
        "blocking_reasons": bundle_blockers + [
            f"{manifest.get('record_id')}:{reason}"
            for manifest in failed
            for reason in manifest.get("blocking_reasons", ())
        ],
        "records": manifests,
    }


def _mark_complete_scaling_preflight_pass(
    payload: Mapping[str, Any],
    *,
    output_json: Path,
    repo_root: Path,
) -> None:
    """Record a successful full-matrix preflight without touching Condor state."""

    record_ids = [str(value) for value in payload.get("record_ids", ())]
    if (
        payload.get("ok") is not True
        or int(payload.get("record_count") or 0) != 102
        or len(record_ids) != 102
        or not all(value.startswith("paper_i_scaling_matrix_") for value in record_ids)
    ):
        raise ValueError("Explicit scaling status update requires a successful complete 102-row scaling preflight")
    records_path = Path(str(payload.get("records_path") or "")).expanduser().resolve()
    batch_dir = records_path.parent
    preflight_path = Path(output_json).expanduser().resolve()
    expected_preflight_path = (batch_dir / "preflight.json").resolve()
    if preflight_path != expected_preflight_path:
        raise ValueError(
            "Explicit scaling status update requires the staged batch preflight path: "
            f"{expected_preflight_path}; got {preflight_path}"
        )
    try:
        preflight_rel = str(preflight_path.relative_to(Path(repo_root).expanduser().resolve()))
    except ValueError as exc:
        raise ValueError(
            f"Explicit scaling status update requires preflight.json under repo_root={Path(repo_root).resolve()}"
        ) from exc
    preflight_record = {
        "status": "pass",
        "path": preflight_rel,
        "sha256": run_task.sha256_file(preflight_path),
        "record_count": 102,
        "failed_record_count": 0,
    }
    for name in ("paper_i_scaling_matrix_manifest.json", "submission_audit.json"):
        path = batch_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Scaling status artifact missing after preflight: {path}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict):
            raise TypeError(f"Scaling status artifact must be an object: {path}")
        if int(artifact.get("record_count") or 0) != 102:
            raise ValueError(f"Scaling status artifact record count mismatch: {path}")
        artifact["preflight"] = dict(preflight_record)
        artifact["status"] = "preflight_pass"
        _write_json(path, artifact)


def _mark_snake_overlay_repair_preflight_pass(
    payload: Mapping[str, Any],
    *,
    output_json: Path,
    repo_root: Path,
) -> None:
    """Record a passing complete 34-row SNAKE repair preflight."""

    record_ids = [str(value) for value in payload.get("record_ids", ())]
    if (
        payload.get("ok") is not True
        or int(payload.get("record_count") or 0) != 34
        or len(record_ids) != 34
        or not all(value.startswith(_PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_BATCH_PREFIX) for value in record_ids)
    ):
        raise ValueError("Explicit SNAKE overlay-repair status update requires a successful complete 34-row preflight")
    repo_root = Path(repo_root).expanduser().resolve()
    records_path = Path(str(payload.get("records_path") or "")).expanduser().resolve()
    rows = [run_task.load_record(records_path, record_id) for record_id in record_ids]
    if {str(row.get("repair_scope") or "").strip() for row in rows} != {
        _PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_SCOPE
    }:
        raise ValueError("Explicit SNAKE overlay-repair status update requires the dedicated repair scope")
    blockers = _paper_i_scaling_matrix_bundle_blockers(rows)
    if blockers:
        raise ValueError(f"SNAKE overlay-repair matrix contract failed during status update: {blockers}")
    batch_dir = records_path.parent
    preflight_path = Path(output_json).expanduser().resolve()
    expected_preflight_path = (batch_dir / "preflight.json").resolve()
    if preflight_path != expected_preflight_path:
        raise ValueError(
            "Explicit SNAKE overlay-repair status update requires the staged batch preflight path: "
            f"{expected_preflight_path}; got {preflight_path}"
        )
    try:
        preflight_rel = str(preflight_path.relative_to(repo_root))
    except ValueError as exc:
        raise ValueError(f"SNAKE overlay-repair preflight must be under repo_root={repo_root}") from exc
    preflight_record = {
        "status": "pass",
        "path": preflight_rel,
        "sha256": run_task.sha256_file(preflight_path),
        "record_count": 34,
        "failed_record_count": 0,
    }
    for name in ("paper_i_scaling_matrix_manifest.json", "submission_audit.json"):
        path = batch_dir / name
        artifact = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(artifact, dict) or int(artifact.get("record_count") or 0) != 34:
            raise ValueError(f"SNAKE overlay-repair status artifact record count mismatch: {path}")
        if str(artifact.get("repair_scope") or "") != _PAPER_I_SCALING_SNAKE_OVERLAY_REPAIR_SCOPE:
            raise ValueError(f"SNAKE overlay-repair status artifact scope mismatch: {path}")
        if int(artifact.get("repair_source_cluster_id") or 0) != 8772847:
            raise ValueError(f"SNAKE overlay-repair source cluster provenance missing: {path}")
        if artifact.get("repair_source_held_procs") != [30, 33]:
            raise ValueError(f"SNAKE overlay-repair held-proc coverage mismatch: {path}")
        artifact["preflight"] = dict(preflight_record)
        artifact["status"] = "preflight_pass"
        _write_json(path, artifact)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fail-closed preflight for Phase3 Optuna CHTC submit files.")
    ap.add_argument("--submit", required=True, type=Path)
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--record-id", action="append", default=None)
    ap.add_argument("--record-id-file", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    ap.add_argument(
        "--update-scaling-status",
        action="store_true",
        default=False,
        help=(
            "Explicitly update the complete scaling batch manifest/audit after a passing 102-row preflight. "
            "Requires --output-json to be that batch's staged preflight.json."
        ),
    )
    ap.add_argument(
        "--update-scaling-repair-status",
        action="store_true",
        default=False,
        help=(
            "Explicitly update a complete 34-row SNAKE overlay-repair manifest/audit after a passing "
            "preflight. Requires --output-json to be that batch's staged preflight.json."
        ),
    )
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = ap.parse_args(argv)
    if args.update_scaling_status and args.output_json is None:
        ap.error("--update-scaling-status requires --output-json")
    if args.update_scaling_repair_status and args.output_json is None:
        ap.error("--update-scaling-repair-status requires --output-json")
    if args.update_scaling_status and args.update_scaling_repair_status:
        ap.error("choose only one scaling status-update mode")

    try:
        payload = build_preflight_bundle(
            submit_path=args.submit,
            records_path=args.records,
            record_ids=args.record_id,
            record_id_file=args.record_id_file,
            repo_root=args.repo_root,
        )
    except Exception as exc:
        payload = {
            "schema": "phase3_chtc_submit_preflight_bundle_v1",
            "submit_path": str(args.submit),
            "status": "fail",
            "ok": False,
            "blocking_reasons": [f"preflight_exception:{type(exc).__name__}:{exc}"],
            "records": [],
        }
    if args.output_json is not None:
        output_json = Path(args.output_json)
        _write_json(output_json, payload)
        if args.update_scaling_status:
            try:
                _mark_complete_scaling_preflight_pass(
                    payload,
                    output_json=output_json,
                    repo_root=Path(args.repo_root),
                )
            except Exception as exc:
                payload = dict(payload)
                payload["status"] = "fail"
                payload["ok"] = False
                payload["blocking_reasons"] = list(payload.get("blocking_reasons", ())) + [
                    f"scaling_preflight_status_update_failed:{type(exc).__name__}:{exc}"
                ]
                _write_json(output_json, payload)
        elif args.update_scaling_repair_status:
            try:
                _mark_snake_overlay_repair_preflight_pass(
                    payload,
                    output_json=output_json,
                    repo_root=Path(args.repo_root),
                )
            except Exception as exc:
                payload = dict(payload)
                payload["status"] = "fail"
                payload["ok"] = False
                payload["blocking_reasons"] = list(payload.get("blocking_reasons", ())) + [
                    f"scaling_repair_preflight_status_update_failed:{type(exc).__name__}:{exc}"
                ]
                _write_json(output_json, payload)
    print(
        f"phase3 CHTC preflight {payload['status']}: "
        f"{payload.get('failed_record_count', 1 if not payload.get('ok') else 0)}/"
        f"{payload.get('record_count', 0)} failed",
        flush=True,
    )
    for reason in payload.get("blocking_reasons", ()):
        print(f"BLOCKER {reason}", file=sys.stderr, flush=True)
    return 0 if bool(payload.get("ok", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
