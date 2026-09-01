#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# RESTORED 2026-09-01 -- incomplete-graft gap #1 from the 8ec6fd92 report
# ("chtc/phase3_optuna/paper_i_comparator_spsa_calibration_runner does not
# exist"). Source: origin/codex/wip-local-active-source-snapshot-20260706-120000
# @ 517eca4c, blob 466d71fcb5b7bf4ad06c5a9173b0528875615b5a, 2026-07-08.
# Content below this header is VERBATIM from that blob; nothing was
# forward-ported. Loss traced to docs/repository_snapshots/2026-07-06/
# SNAPSHOT_DATALESS_SKIPPED.txt:472, which lists this module as deliberately
# skipped by the dataless snapshot. Two of its neighbours have drifted since --
# pipelines/exact_bench/paper_i_comparator_spsa_calibration.py and
# chtc/phase3_optuna/preflight_submit.py -- so treat era mismatches here as
# expected rather than as regressions.
# ---------------------------------------------------------------------------
"""Run one Paper-I comparator SPSA calibration record in-process.

The runner consumes one generated method-target record and evaluates every case
in the target under strict environment save/restore.  It writes durable progress
and current-best artifacts while the study is running.  Real execution requires
Optuna; tests may monkeypatch the study factory and ``run_single`` to avoid heavy
benchmark work.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.phase3_optuna.generate_paper_i_comparator_spsa_calibration_records import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    DEFAULT_QUEUE_OUTPUT_ROOT,
    RESOURCE_METRIC_PRECEDENCE,
)
from pipelines.exact_bench import generic_static_benchmark  # noqa: E402
from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (  # noqa: E402
    HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY,
    PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
    PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE,
    PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS,
    PAPER_I_NATIVE_SPSA_REFIT_ENGINE,
    config_sha256_for_path,
    load_and_validate_config,
    target_by_id,
    validate_method_id,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (  # noqa: E402
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_TARGET,
)

DEFAULT_RECORDS_PATH = REPO_ROOT / "chtc" / "phase3_optuna" / "input" / "paper_i_comparator_spsa_calibration_v1_smoke" / "paper_i_comparator_spsa_calibration_records.tsv"
DEFAULT_OUTPUT_ROOT = DEFAULT_QUEUE_OUTPUT_ROOT
_TABLE_I_STATIC_SUITE_PROFILE_ENV = "TABLE_I_STATIC_SUITE_PROFILE"
_ENERGY_STOP_TARGET_ENV = "GENERIC_STATIC_TABLE_ENERGY_STOP_TARGET"
_FIRST_HIT_THRESHOLDS_ENV = "GENERIC_STATIC_TABLE_FIRST_HIT_THRESHOLDS"
_RESOURCE_QUBIT_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_QUBIT_CAP"
_RESOURCE_POOL_TERM_CAP_ENV = "GENERIC_STATIC_TABLE_RESOURCE_POOL_TERM_CAP"
_GENERIC_STATIC_ADAPT_PROGRESS_JSONL_ENV = "GENERIC_STATIC_TABLE_PROGRESS_JSONL_PATH"
_EXACT_REFERENCE_ENERGY_ENV = "GENERIC_STATIC_TABLE_EXACT_REFERENCE_ENERGY"
_EXACT_REFERENCE_N_PH_MAX_ENV = "GENERIC_STATIC_TABLE_EXACT_REFERENCE_N_PH_MAX"
_SAME_CUTOFF_EXACT_GS_ENERGY_ENV = "GENERIC_STATIC_TABLE_SAME_CUTOFF_EXACT_GS_ENERGY"
_PRIMARY_ENERGY_METRIC_ENV = "GENERIC_STATIC_TABLE_PRIMARY_ENERGY_METRIC"
_SAME_CUTOFF_ERROR_ROLE_ENV = "GENERIC_STATIC_TABLE_SAME_CUTOFF_ERROR_ROLE"
_PHASE3_ADAPT_MAX_DEPTH_ENV = "GENERIC_STATIC_TABLE_PHASE3_ADAPT_MAX_DEPTH"
_HH_POOL_CACHE_ENV = "STATIC_ADAPT_HH_POOL_CACHE"
_HH_POOL_CACHE_SCOPE_ENV = "STATIC_ADAPT_HH_POOL_CACHE_SCOPE"
_HH_POOL_CACHE_DIR_ENV = "STATIC_ADAPT_HH_POOL_CACHE_DIR"
_HH_FULL_META_MINUS_HVA_CACHE_METHOD_IDS = frozenset(
    {
        "static_full_meta_append_adapt_vqe",
        "static_geo_adapt_vqe",
    }
)
_MANAGED_ENV_NAMES = frozenset(
    {
        _TABLE_I_STATIC_SUITE_PROFILE_ENV,
        _ENERGY_STOP_TARGET_ENV,
        _FIRST_HIT_THRESHOLDS_ENV,
        _RESOURCE_QUBIT_CAP_ENV,
        _RESOURCE_POOL_TERM_CAP_ENV,
        _GENERIC_STATIC_ADAPT_PROGRESS_JSONL_ENV,
        _EXACT_REFERENCE_ENERGY_ENV,
        _EXACT_REFERENCE_N_PH_MAX_ENV,
        _SAME_CUTOFF_EXACT_GS_ENERGY_ENV,
        _PRIMARY_ENERGY_METRIC_ENV,
        _SAME_CUTOFF_ERROR_ROLE_ENV,
        _PHASE3_ADAPT_MAX_DEPTH_ENV,
        _HH_POOL_CACHE_ENV,
        _HH_POOL_CACHE_SCOPE_ENV,
        *PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES.values(),
    }
)
_INT_SCHEDULE_FIELDS = {"family_informed_spsa_eval_repeats", "family_informed_spsa_avg_last"}
_STRICT_STATUS_USABLE_POLICY = "strict_status_completed_v1"


def _repo_path(path_value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else Path(repo_root) / path
_QUALITY_NONPASSING_STATUSES = {"quality_nonpassing", "completed_quality_nonpassing"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_write(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl_append(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _compact_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_record(records_path: str | Path, record_id: str) -> dict[str, str]:
    with Path(records_path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("record_id", "")) == str(record_id):
                return {str(key): "" if value is None else str(value) for key, value in row.items()}
    raise SystemExit(f"record_id {record_id!r} not found in {records_path}")


def _case_ids_from_row(row: Mapping[str, str]) -> tuple[str, ...]:
    raw = str(row.get("case_ids_json", "")).strip()
    if not raw:
        raise ValueError("calibration record must include non-empty case_ids_json")
    try:
        payload = json.loads(raw)
    except Exception as exc:
        raise ValueError("calibration record has malformed case_ids_json") from exc
    if not isinstance(payload, list) or not payload or not all(isinstance(item, str) and item.strip() for item in payload):
        raise ValueError("calibration record case_ids_json must be a non-empty JSON string array")
    return tuple(str(item).strip() for item in payload)


def validate_record(
    row: Mapping[str, str],
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    if str(row.get("profile_id", "")) != PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID:
        raise ValueError(
            f"record profile_id must be {PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID!r}; got {row.get('profile_id')!r}"
        )
    method_id = validate_method_id(str(row.get("method_id") or row.get("algorithm_id") or ""))
    algorithm_id = str(row.get("algorithm_id") or "").strip()
    if algorithm_id != method_id:
        raise ValueError(f"calibration record algorithm_id={algorithm_id!r} must match method_id={method_id!r}")
    target = target_by_id(str(row.get("target_id") or ""))
    case_ids = _case_ids_from_row(row)
    if case_ids != target.case_ids:
        raise ValueError(f"record target_id={target.target_id!r} case_ids={case_ids!r}; expected {target.case_ids!r}")
    family = str(row.get("family") or row.get("target_family") or "").strip()
    if family != target.family:
        raise ValueError(f"record family={family!r}; expected target family {target.family!r}")
    if str(row.get("suite_profile", "")) != PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID:
        raise ValueError(f"record suite_profile must be {PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID!r}")
    if str(row.get("optimizer_profile", "")) != PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID:
        raise ValueError(f"record optimizer_profile must be {PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID!r}")
    expected_hash = config_sha256_for_path(config_path)
    if str(row.get("config_sha256", "")).strip() != expected_hash:
        raise ValueError(
            f"record config_sha256={row.get('config_sha256')!r} does not match config path hash {expected_hash!r}"
        )
    if int(row.get("n_jobs") or "1") != 1:
        raise ValueError("calibration runner requires n_jobs=1 because generic_static_benchmark.run_single reads process-global env")
    if int(row.get("n_trials") or 0) != int(config["n_trials"]):
        raise ValueError(f"record n_trials={row.get('n_trials')!r} does not match config n_trials={config['n_trials']!r}")
    if int(row.get("method_maxiter_budget") or 0) != int(config["method_maxiter_budgets"][method_id]):
        raise ValueError("record method_maxiter_budget does not match config method_maxiter_budgets")
    rerun_scope = str(row.get("rerun_scope") or "").strip()
    adaptive_refit_engine = str(row.get("adaptive_refit_engine") or "").strip()
    if rerun_scope:
        if rerun_scope != PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
            raise ValueError(
                f"Unsupported rerun_scope={rerun_scope!r}; expected {PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE!r}."
            )
        if method_id not in PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS:
            raise ValueError(f"native SPSA adaptive rerun record method_id={method_id!r} is not in the affected adaptive scope")
        if adaptive_refit_engine != PAPER_I_NATIVE_SPSA_REFIT_ENGINE:
            raise ValueError(
                f"native SPSA adaptive rerun record adaptive_refit_engine={adaptive_refit_engine!r}; "
                f"expected {PAPER_I_NATIVE_SPSA_REFIT_ENGINE!r}"
            )
    elif adaptive_refit_engine:
        raise ValueError("adaptive_refit_engine may only be set on an explicit native SPSA adaptive rerun record")
    warm_start_schedule_lock_json = str(row.get("warm_start_schedule_lock_json") or "").strip()
    warm_start_schedule_key = str(row.get("warm_start_schedule_key") or "").strip()
    if warm_start_schedule_lock_json and not warm_start_schedule_key:
        raise ValueError("warm_start_schedule_key is required when warm_start_schedule_lock_json is set")
    if warm_start_schedule_key and warm_start_schedule_key != f"{method_id}::{target.target_id}":
        raise ValueError(
            f"warm_start_schedule_key={warm_start_schedule_key!r}; expected {method_id}::{target.target_id}"
        )
    if warm_start_schedule_lock_json and not _repo_path(warm_start_schedule_lock_json, repo_root=repo_root).exists():
        raise ValueError(f"warm_start_schedule_lock_json not found: {warm_start_schedule_lock_json}")
    return {
        "method_id": method_id,
        "target": target,
        "case_ids": case_ids,
        "family": family,
        "rerun_scope": rerun_scope,
        "adaptive_refit_engine": adaptive_refit_engine,
        "warm_start_schedule_lock_json": warm_start_schedule_lock_json,
        "warm_start_schedule_key": warm_start_schedule_key,
    }


def _create_study(*, study_name: str, sampler_seed: int, storage: str | None = None) -> Any:
    try:
        import optuna  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "Optuna is required for real Paper-I comparator SPSA calibration runs. "
            "Install optuna in the execution environment or run the focused unit tests with a monkeypatched study factory."
        ) from exc
    sampler = optuna.samplers.TPESampler(seed=int(sampler_seed))
    return optuna.create_study(
        direction="minimize",
        sampler=sampler,
        study_name=str(study_name),
        storage=storage,
        load_if_exists=False,
    )


def _suggest_value(trial: Any, name: str, spec: Mapping[str, Any]) -> int | float:
    kind = str(spec.get("type", spec.get("kind", ""))).strip().lower().replace("-", "_")
    if kind == "choice":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        low_key = "low" if "low" in spec else "min"
        high_key = "high" if "high" in spec else "max"
        return int(trial.suggest_int(name, int(spec[low_key]), int(spec[high_key])))
    low_key = "low" if "low" in spec else "min"
    high_key = "high" if "high" in spec else "max"
    return float(trial.suggest_float(name, float(spec[low_key]), float(spec[high_key]), log=(kind == "log_uniform")))


def suggest_schedule(trial: Any, *, method_id: str, config: Mapping[str, Any]) -> dict[str, int | float]:
    method_space = config["per_method_search_spaces"][method_id]
    allowed = set(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD[method_id])
    schedule: dict[str, int | float] = {}
    for name in sorted(method_space):
        if name not in allowed:
            raise ValueError(f"search-space field {name!r} is not allowed for method {method_id!r}")
        value = _suggest_value(trial, name, method_space[name])
        schedule[name] = int(value) if name in _INT_SCHEDULE_FIELDS else float(value)
    return schedule


def _validate_schedule_values(
    schedule: Mapping[str, Any],
    *,
    method_id: str,
    expected_fields: set[str],
    label: str,
) -> dict[str, int | float]:
    allowed = set(PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD[method_id])
    keys = {str(key) for key in schedule}
    extra = sorted(keys - allowed)
    missing = sorted(expected_fields - keys)
    unexpected = sorted(keys - expected_fields)
    if extra or missing or unexpected:
        raise ValueError(
            f"{label} schedule fields mismatch: extra={extra}, missing={missing}, unexpected={unexpected}"
        )
    out: dict[str, int | float] = {}
    for key in sorted(expected_fields):
        value = schedule[key]
        try:
            numeric = float(value)
        except Exception as exc:
            raise ValueError(f"{label} schedule value for {key!r} is not numeric: {value!r}") from exc
        if not math.isfinite(numeric) or numeric <= 0.0:
            raise ValueError(f"{label} schedule value for {key!r} must be positive finite: {value!r}")
        if key in _INT_SCHEDULE_FIELDS:
            if not float(numeric).is_integer():
                raise ValueError(f"{label} schedule integer value for {key!r} is not integral: {value!r}")
            out[key] = int(numeric)
        else:
            out[key] = float(numeric)
    return out


def _warm_start_schedule_from_record(
    record: Mapping[str, str],
    *,
    method_id: str,
    target_id: str,
    expected_fields: set[str],
) -> dict[str, int | float] | None:
    lock_value = str(record.get("warm_start_schedule_lock_json") or "").strip()
    if not lock_value:
        return None
    key = str(record.get("warm_start_schedule_key") or "").strip() or f"{method_id}::{target_id}"
    if key != f"{method_id}::{target_id}":
        raise ValueError(f"warm_start_schedule_key={key!r}; expected {method_id}::{target_id}")
    lock_path = _repo_path(lock_value)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"warm-start schedule lock is not an object: {lock_value}")
    entries = payload.get("method_target_schedules")
    if not isinstance(entries, Mapping) or key not in entries:
        raise ValueError(f"warm-start schedule lock {lock_value} missing key {key!r}")
    entry = entries[key]
    if not isinstance(entry, Mapping):
        raise ValueError(f"warm-start schedule lock entry {key!r} is not an object")
    schedule = entry.get("schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError(f"warm-start schedule lock entry {key!r} missing schedule object")
    return _validate_schedule_values(schedule, method_id=method_id, expected_fields=expected_fields, label="warm_start")


def _overlay_from_row_and_schedule(row: Mapping[str, str], schedule: Mapping[str, int | float]) -> dict[str, str]:
    overlay: dict[str, str] = {
        _TABLE_I_STATIC_SUITE_PROFILE_ENV: str(row["suite_profile"]),
        _ENERGY_STOP_TARGET_ENV: str(row.get("target_abs_delta_e") or PAPER_I_MAIN_TABLES_SPSA_TARGET),
        _FIRST_HIT_THRESHOLDS_ENV: str(row.get("target_abs_delta_e") or PAPER_I_MAIN_TABLES_SPSA_TARGET),
    }
    row_env_fields = {
        "resource_qubit_cap": _RESOURCE_QUBIT_CAP_ENV,
        "resource_pool_term_cap": _RESOURCE_POOL_TERM_CAP_ENV,
        "same_cutoff_exact_gs_energy": _SAME_CUTOFF_EXACT_GS_ENERGY_ENV,
        "exact_reference_energy": _EXACT_REFERENCE_ENERGY_ENV,
        "exact_reference_n_ph_max": _EXACT_REFERENCE_N_PH_MAX_ENV,
        "primary_energy_metric": _PRIMARY_ENERGY_METRIC_ENV,
        "same_cutoff_error_role": _SAME_CUTOFF_ERROR_ROLE_ENV,
        "phase3_adapt_max_depth": _PHASE3_ADAPT_MAX_DEPTH_ENV,
    }
    for field, env_name in row_env_fields.items():
        value = str(row.get(field, "") or "").strip()
        if value:
            overlay[env_name] = value
    for field in PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS:
        value = str(row.get(field, "") or "").strip()
        if value:
            overlay[PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES[field]] = value
    for field, value in schedule.items():
        overlay[PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_ENV_NAMES[field]] = str(value)
    if _uses_hh_full_meta_minus_hva_cache(row):
        overlay[_HH_POOL_CACHE_ENV] = "disk"
        overlay[_HH_POOL_CACHE_SCOPE_ENV] = "paper_i_holstein_sector"
    return overlay


def _uses_hh_full_meta_minus_hva_cache(row: Mapping[str, str]) -> bool:
    method_id = str(row.get("method_id") or row.get("algorithm_id") or "").strip()
    family = str(row.get("family") or row.get("target_family") or "").strip().lower()
    return family == "hh" and method_id in _HH_FULL_META_MINUS_HVA_CACHE_METHOD_IDS


@contextmanager
def _temporary_env(overlay: Mapping[str, str]) -> Iterator[None]:
    managed = set(_MANAGED_ENV_NAMES) | set(overlay)
    old = {name: os.environ.get(name) for name in managed}
    try:
        for name in managed:
            os.environ.pop(name, None)
        for name, value in overlay.items():
            os.environ[str(name)] = str(value)
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _finite_float_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return float(parsed)


def _candidate_result_rows(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    out: list[Mapping[str, Any]] = []
    result = payload.get("result")
    if isinstance(result, Mapping):
        out.append(result)
    rows = payload.get("rows")
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        for row in rows:
            if isinstance(row, Mapping):
                out.append(row)
    out.append(payload)
    return out


def _extract_abs_delta_e(payload: Mapping[str, Any]) -> float | None:
    for row in _candidate_result_rows(payload):
        for field in ("abs_delta_e", "delta_E_abs"):
            value = _finite_float_or_none(row.get(field))
            if value is not None:
                return value
    return None


def _extract_resource_metric(payload: Mapping[str, Any], precedence: Sequence[str]) -> tuple[float, str | None]:
    for row in _candidate_result_rows(payload):
        for field in precedence:
            value = _finite_float_or_none(row.get(field))
            if value is not None and value >= 0.0:
                return float(value), str(field)
    return 0.0, None


def _status_from_payload(payload: Mapping[str, Any]) -> str:
    for row in _candidate_result_rows(payload):
        status = row.get("status")
        if status not in {None, ""}:
            return str(status)
    return "unknown"


def _status_is_usable(status: str) -> bool:
    return str(status).strip().lower() in {"completed", "ok", "success", "done"}


def _status_usable_policy(row: Mapping[str, str]) -> str:
    raw = str(row.get("calibration_usable_status_policy") or "").strip()
    if raw in {"", "none", "strict"}:
        return _STRICT_STATUS_USABLE_POLICY
    if raw != HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY:
        raise ValueError(
            f"Unknown calibration_usable_status_policy={raw!r}; expected {_STRICT_STATUS_USABLE_POLICY!r} "
            f"or {HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY!r}."
        )
    return raw


def _quality_nonpassing_penalty(row: Mapping[str, str]) -> float:
    raw = str(row.get("quality_nonpassing_penalty") or "").strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except Exception as exc:
        raise ValueError(f"quality_nonpassing_penalty must be finite numeric; got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"quality_nonpassing_penalty must be finite and nonnegative; got {raw!r}")
    return float(value)


def _case_usability(
    *,
    status: str,
    abs_delta_e: float | None,
    error: str | None,
    status_policy: str,
) -> tuple[bool, bool, bool]:
    strict_usable = abs_delta_e is not None and error is None and _status_is_usable(status)
    quality_nonpassing_usable = (
        status_policy == HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY
        and abs_delta_e is not None
        and error is None
        and str(status).strip().lower() in _QUALITY_NONPASSING_STATUSES
    )
    calibration_usable = strict_usable or quality_nonpassing_usable
    return strict_usable, calibration_usable, quality_nonpassing_usable


def _case_loss(abs_delta_e: float | None, *, target: float, clipping: Sequence[float], failure_penalty: float) -> float:
    if abs_delta_e is None:
        return float(failure_penalty)
    ratio = max(float(abs_delta_e), 1e-12) / float(target)
    raw = math.log10(ratio)
    return float(min(float(clipping[1]), max(float(clipping[0]), raw)))


def _trial_set_user_attr(trial: Any, key: str, value: Any) -> None:
    setter = getattr(trial, "set_user_attr", None)
    if callable(setter):
        setter(str(key), value)


def _trial_number(trial: Any) -> int:
    return int(getattr(trial, "number", 0))


def _case_output_dir(out_root: Path, trial_number: int, case_id: str) -> Path:
    return out_root / f"trial_{int(trial_number):04d}" / "cases" / str(case_id)


def run_calibration_record(
    *,
    record: Mapping[str, str],
    config: Mapping[str, Any],
    config_path: str | Path,
    out_root: str | Path,
    study_factory: Callable[..., Any] = _create_study,
    storage: str | None = None,
) -> dict[str, Any]:
    validation = validate_record(record, config=config, config_path=config_path)
    method_id = str(validation["method_id"])
    target = validation["target"]
    case_ids = tuple(validation["case_ids"])
    family = str(validation["family"])
    rerun_scope = str(validation.get("rerun_scope") or "")
    adaptive_refit_engine = str(validation.get("adaptive_refit_engine") or "")
    warm_start_schedule_lock_json = str(validation.get("warm_start_schedule_lock_json") or "")
    warm_start_schedule_key = str(validation.get("warm_start_schedule_key") or "")
    record_id = str(record["record_id"])
    out = Path(out_root)
    progress = out / "progress"
    trial_events_path = progress / "trial_events.jsonl"
    current_best_path = progress / "current_best.json"
    summary_path = out / "summary.json"
    best_schedule_path = out / "best_schedule.json"
    heartbeat_path = out / "heartbeat.json"
    out.mkdir(parents=True, exist_ok=True)
    progress.mkdir(parents=True, exist_ok=True)
    if trial_events_path.exists():
        trial_events_path.unlink()

    config_hash = str(config.get("config_sha256") or config_sha256_for_path(config_path))
    n_trials = int(config["n_trials"])
    sampler_seed = int(config["sampler_seed"])
    failure_penalty = float(config["failure_penalty"])
    clipping = [float(config["clipping_log10_error_ratio"][0]), float(config["clipping_log10_error_ratio"][1])]
    target_abs = float(record.get("target_abs_delta_e") or PAPER_I_MAIN_TABLES_SPSA_TARGET)
    resource_weight = float(config["resource_tiebreak_weight"])
    resource_precedence = list(RESOURCE_METRIC_PRECEDENCE)
    status_policy = _status_usable_policy(record)
    quality_nonpassing_penalty = _quality_nonpassing_penalty(record)
    started = _utc_now()
    study_name = f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID}::{record_id}"
    events: list[dict[str, Any]] = []
    best_event: dict[str, Any] | None = None
    best_usable_event: dict[str, Any] | None = None
    best_calibration_usable_event: dict[str, Any] | None = None
    expected_schedule_fields = set(str(name) for name in config["per_method_search_spaces"][method_id])
    warm_start_schedule = _warm_start_schedule_from_record(
        record,
        method_id=method_id,
        target_id=target.target_id,
        expected_fields=expected_schedule_fields,
    )
    warm_start_metadata = {
        "warm_start_schedule_lock_json": warm_start_schedule_lock_json or None,
        "warm_start_schedule_key": warm_start_schedule_key or None,
        "warm_start_enqueued_trial_count": 1 if warm_start_schedule is not None else 0,
        "warm_start_schedule": None if warm_start_schedule is None else dict(warm_start_schedule),
    }
    uses_hh_pool_cache = _uses_hh_full_meta_minus_hva_cache(record)
    hh_pool_cache_metadata = {
        "hh_full_meta_minus_hva_cache_overlay": bool(uses_hh_pool_cache),
        "hh_pool_cache_mode": "disk" if uses_hh_pool_cache else None,
        "hh_pool_cache_scope": "paper_i_holstein_sector" if uses_hh_pool_cache else None,
        "hh_pool_cache_dir_source": "inherited_env" if os.environ.get(_HH_POOL_CACHE_DIR_ENV) else None,
    }
    record_scope_metadata = {
        "rerun_scope": rerun_scope or None,
        "adaptive_refit_engine": adaptive_refit_engine or None,
        **warm_start_metadata,
        **hh_pool_cache_metadata,
    }

    def _write_heartbeat(status: str, **extra: Any) -> None:
        payload = {
            "schema": "paper_i_comparator_spsa_calibration_heartbeat_v1",
            "status": status,
            "updated_utc": _utc_now(),
            "started_utc": started,
            "record_id": record_id,
            "method_id": method_id,
            "target_id": target.target_id,
            "case_ids": list(case_ids),
            **record_scope_metadata,
            "completed_trial_count": len(events),
            "usable_trial_count": sum(1 for event in events if bool(event.get("all_cases_usable"))),
            "strict_usable_trial_count": sum(1 for event in events if bool(event.get("all_cases_usable"))),
            "calibration_usable_trial_count": sum(1 for event in events if bool(event.get("all_cases_calibration_usable"))),
            "status_usable_policy": status_policy,
            **extra,
        }
        _json_write(heartbeat_path, payload)

    def _write_current_best(event: Mapping[str, Any] | None) -> None:
        payload = {
            "schema": "paper_i_comparator_spsa_current_best_v1",
            "study_name": study_name,
            "record_id": record_id,
            "method_id": method_id,
            "target_id": target.target_id,
            "case_ids": list(case_ids),
            **record_scope_metadata,
            "updated_utc": _utc_now(),
            "source_schema_version": "paper_i_comparator_spsa_calibration_runner_v1",
            "best": None if event is None else dict(event),
            "completed_trial_count": len(events),
            "usable_trial_count": sum(1 for item in events if bool(item.get("all_cases_usable"))),
            "strict_usable_trial_count": sum(1 for item in events if bool(item.get("all_cases_usable"))),
            "calibration_usable_trial_count": sum(1 for item in events if bool(item.get("all_cases_calibration_usable"))),
            "status_usable_policy": status_policy,
            "progress_events_jsonl": str(trial_events_path),
        }
        _json_write(current_best_path, payload)

    def _objective(trial: Any) -> float:
        nonlocal best_event, best_usable_event, best_calibration_usable_event
        number = _trial_number(trial)
        schedule = suggest_schedule(trial, method_id=method_id, config=config)
        overlay = _overlay_from_row_and_schedule(record, schedule)
        case_summaries: list[dict[str, Any]] = []
        case_losses: list[float] = []
        resources: list[float] = []
        trial_start = time.perf_counter()
        _write_heartbeat("running", current_trial_number=number, current_schedule=dict(schedule))
        for case_id in case_ids:
            case_dir = _case_output_dir(out, number, case_id)
            case_dir.mkdir(parents=True, exist_ok=True)
            case_progress_jsonl = case_dir / "progress" / "generic_static_adapt_progress.jsonl"
            case_overlay = dict(overlay)
            case_overlay[_GENERIC_STATIC_ADAPT_PROGRESS_JSONL_ENV] = str(case_progress_jsonl)
            payload: Mapping[str, Any] = {}
            error: str | None = None
            with _temporary_env(case_overlay):
                try:
                    payload = generic_static_benchmark.run_single(
                        family=family,
                        case_id=str(case_id),
                        algorithm_id=method_id,
                        output_dir=case_dir,
                    )
                except Exception as exc:  # keep the trial alive; failed case gets penalty.
                    error = f"{type(exc).__name__}: {exc}"
                    payload = {
                        "schema": "paper_i_comparator_spsa_case_exception_v1",
                        "status": "failed_exception",
                        "error": error,
                    }
                    _json_write(case_dir / "case_exception.json", payload)
            abs_delta_e = _extract_abs_delta_e(payload)
            resource_metric, resource_field = _extract_resource_metric(payload, resource_precedence)
            status = _status_from_payload(payload)
            strict_usable, calibration_usable, quality_nonpassing_usable = _case_usability(
                status=status,
                abs_delta_e=abs_delta_e,
                error=error,
                status_policy=status_policy,
            )
            penalty_applied = quality_nonpassing_penalty if quality_nonpassing_usable else 0.0
            loss = _case_loss(
                abs_delta_e if calibration_usable else None,
                target=target_abs,
                clipping=clipping,
                failure_penalty=failure_penalty,
            ) + float(penalty_applied)
            case_summary = {
                "schema": "paper_i_comparator_spsa_case_summary_v1",
                "record_id": record_id,
                "method_id": method_id,
                "target_id": target.target_id,
                "case_id": str(case_id),
                "trial_number": number,
                **record_scope_metadata,
                "status": status,
                "usable": bool(strict_usable),
                "strict_status_usable": bool(strict_usable),
                "calibration_usable": bool(calibration_usable),
                "status_usable_policy": status_policy,
                "quality_nonpassing_usable": bool(quality_nonpassing_usable),
                "quality_nonpassing_penalty_applied": float(penalty_applied),
                "error": error,
                "abs_delta_e": abs_delta_e,
                "case_loss": float(loss),
                "resource_metric": float(resource_metric),
                "resource_metric_field": resource_field,
                "output_dir": str(case_dir),
                "generic_static_single_json": str(case_dir / "generic_static_single.json"),
                "generic_static_adapt_progress_jsonl": str(case_progress_jsonl),
            }
            _json_write(case_dir / "case_summary.json", case_summary)
            case_summaries.append(case_summary)
            case_losses.append(float(loss))
            resources.append(float(resource_metric))
        all_usable = all(bool(case["usable"]) for case in case_summaries) and len(case_summaries) == len(case_ids)
        all_calibration_usable = all(bool(case["calibration_usable"]) for case in case_summaries) and len(case_summaries) == len(case_ids)
        mean_loss = float(sum(case_losses) / len(case_losses)) if case_losses else float(failure_penalty)
        mean_resource = float(sum(math.log1p(max(0.0, value)) for value in resources) / len(resources)) if resources else 0.0
        objective = float(mean_loss + resource_weight * mean_resource)
        elapsed_s = float(time.perf_counter() - trial_start)
        event = {
            "schema": "paper_i_comparator_spsa_trial_event_v1",
            "study_name": study_name,
            "record_id": record_id,
            "method_id": method_id,
            "target_id": target.target_id,
            "case_ids": list(case_ids),
            **record_scope_metadata,
            "trial_number": number,
            "trial_state": "COMPLETE",
            "objective": objective,
            "mean_case_loss": mean_loss,
            "mean_log1p_resource_metric": mean_resource,
            "resource_tiebreak_weight": resource_weight,
            "all_cases_usable": bool(all_usable),
            "all_cases_calibration_usable": bool(all_calibration_usable),
            "status_usable_policy": status_policy,
            "strict_usable_case_count": sum(1 for case in case_summaries if bool(case.get("strict_status_usable"))),
            "calibration_usable_case_count": sum(1 for case in case_summaries if bool(case.get("calibration_usable"))),
            "quality_nonpassing_penalty": quality_nonpassing_penalty,
            "schedule": dict(schedule),
            "case_results": case_summaries,
            "case_abs_delta_e": {case["case_id"]: case["abs_delta_e"] for case in case_summaries},
            "case_output_dirs": {case["case_id"]: case["output_dir"] for case in case_summaries},
            "log_paths": {"stdout": None, "stderr": None},
            "warm_start_trial": bool(warm_start_schedule is not None and number == 0),
            "elapsed_s": elapsed_s,
            "updated_utc": _utc_now(),
        }
        _json_write(out / f"trial_{number:04d}" / "trial_summary.json", event)
        events.append(event)
        _jsonl_append(trial_events_path, event)
        if best_event is None or float(event["objective"]) < float(best_event["objective"]):
            best_event = event
        if all_usable and (best_usable_event is None or float(event["objective"]) < float(best_usable_event["objective"])):
            best_usable_event = event
        if all_calibration_usable and (
            best_calibration_usable_event is None
            or float(event["objective"]) < float(best_calibration_usable_event["objective"])
        ):
            best_calibration_usable_event = event
        _write_current_best(best_event)
        _write_heartbeat(
            "running",
            current_trial_number=number,
            best_trial_number=None if best_event is None else best_event["trial_number"],
            best_objective=None if best_event is None else best_event["objective"],
        )
        _trial_set_user_attr(trial, "schedule", dict(schedule))
        _trial_set_user_attr(trial, "case_results", case_summaries)
        _trial_set_user_attr(trial, "all_cases_usable", bool(all_usable))
        _trial_set_user_attr(trial, "all_cases_calibration_usable", bool(all_calibration_usable))
        return objective

    _json_write(out / "record.json", dict(record))
    _json_write(out / "config_snapshot.json", dict(config))
    _write_heartbeat("starting")
    _write_current_best(None)
    study = study_factory(study_name=study_name, sampler_seed=sampler_seed, storage=storage)
    if warm_start_schedule is not None:
        enqueue_trial = getattr(study, "enqueue_trial", None)
        if not callable(enqueue_trial):
            raise RuntimeError("study_factory returned a study without enqueue_trial; cannot honor warm-start schedule")
        enqueue_trial(dict(warm_start_schedule))
        _write_heartbeat("starting", warm_start_enqueued=True)
    study.optimize(_objective, n_trials=n_trials, n_jobs=1)

    usable_trial_count = sum(1 for event in events if bool(event.get("all_cases_usable")))
    calibration_usable_trial_count = sum(1 for event in events if bool(event.get("all_cases_calibration_usable")))
    best_for_schedule = best_usable_event or best_calibration_usable_event or best_event
    best_schedule_payload = {
        "schema": "paper_i_comparator_spsa_best_schedule_v1",
        "study_name": study_name,
        "record_id": record_id,
        "method_id": method_id,
        "target_id": target.target_id,
        "case_ids": list(case_ids),
        **record_scope_metadata,
        "updated_utc": _utc_now(),
        "usable": bool(best_usable_event is not None),
        "strict_status_usable": bool(best_usable_event is not None),
        "calibration_usable": bool(best_calibration_usable_event is not None or best_usable_event is not None),
        "status_usable_policy": status_policy,
        "trial_number": None if best_for_schedule is None else best_for_schedule["trial_number"],
        "objective": None if best_for_schedule is None else best_for_schedule["objective"],
        "schedule": {} if best_for_schedule is None else dict(best_for_schedule["schedule"]),
        "case_abs_delta_e": {} if best_for_schedule is None else dict(best_for_schedule["case_abs_delta_e"]),
        "source_trial_summary_json": None
        if best_for_schedule is None
        else str(out / f"trial_{int(best_for_schedule['trial_number']):04d}" / "trial_summary.json"),
    }
    _json_write(best_schedule_path, best_schedule_payload)
    _write_current_best(best_event)
    summary = {
        "schema": "paper_i_comparator_spsa_calibration_summary_v1",
        "ok": bool(calibration_usable_trial_count > 0),
        "status": "completed" if calibration_usable_trial_count > 0 else "completed_no_usable_trials",
        "evidence_role": "calibration_only_not_manuscript_table_evidence",
        "study_name": study_name,
        "record_id": record_id,
        "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
        "method_id": method_id,
        "target_id": target.target_id,
        **record_scope_metadata,
        "family": family,
        "case_ids": list(case_ids),
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "n_trials_requested": n_trials,
        "completed_trial_count": len(events),
        "usable_trial_count": usable_trial_count,
        "strict_usable_trial_count": usable_trial_count,
        "calibration_usable_trial_count": calibration_usable_trial_count,
        "failed_or_incomplete_trial_count": len(events) - calibration_usable_trial_count,
        "strict_failed_or_incomplete_trial_count": len(events) - usable_trial_count,
        "best_trial_number": best_schedule_payload["trial_number"],
        "best_objective": best_schedule_payload["objective"],
        "best_schedule": best_schedule_payload["schedule"],
        "summary_json": str(summary_path),
        "best_schedule_json": str(best_schedule_path),
        "progress_trial_events_jsonl": str(trial_events_path),
        "progress_current_best_json": str(current_best_path),
        "heartbeat_json": str(heartbeat_path),
        "resource_metric_precedence": list(resource_precedence),
        "objective_mode": "mean_clipped_log10_abs_delta_e_plus_resource_tiebreak_v1",
        "target_abs_delta_e": target_abs,
        "failure_penalty": failure_penalty,
        "clipping_log10_error_ratio": clipping,
        "resource_tiebreak_weight": resource_weight,
        "status_usable_policy": status_policy,
        "quality_nonpassing_penalty": quality_nonpassing_penalty,
        "started_utc": started,
        "finished_utc": _utc_now(),
    }
    _json_write(summary_path, summary)
    _write_heartbeat(
        str(summary["status"]),
        best_trial_number=summary["best_trial_number"],
        best_objective=summary["best_objective"],
        summary_json=str(summary_path),
        best_schedule_json=str(best_schedule_path),
    )
    return summary


def run_record(
    *,
    record_id: str,
    records_path: str | Path,
    config_path: str | Path,
    out_root: str | Path,
    storage: str | None = None,
    study_factory: Callable[..., Any] = _create_study,
) -> dict[str, Any]:
    config = load_and_validate_config(config_path)
    row = load_record(records_path, record_id)
    return run_calibration_record(
        record=row,
        config=config,
        config_path=config_path,
        out_root=out_root,
        storage=storage,
        study_factory=study_factory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one Paper-I comparator SPSA calibration method-target record.")
    parser.add_argument("record_id")
    parser.add_argument("records_path", nargs="?", default=None)
    parser.add_argument("out_root", nargs="?", default=None)
    parser.add_argument("--config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--storage", default=None, help="Optional Optuna storage URL; defaults to in-memory deterministic rerun.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records_path = Path(args.records_path or os.environ.get("PAPER_I_COMPARATOR_SPSA_CALIBRATION_RECORDS_PATH", "") or DEFAULT_RECORDS_PATH)
    row = load_record(records_path, str(args.record_id))
    default_record_out = Path(str(row.get("queue_output_root") or DEFAULT_OUTPUT_ROOT)) / str(row.get("record_output_dir") or row["record_id"])
    out_root = Path(args.out_root or default_record_out)
    summary = run_calibration_record(
        record=row,
        config=load_and_validate_config(args.config_path),
        config_path=args.config_path,
        out_root=out_root,
        storage=args.storage,
    )
    print(_compact_json({"summary_json": summary["summary_json"], "ok": summary["ok"], "status": summary["status"]}))
    return 0 if bool(summary.get("ok")) else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
