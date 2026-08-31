#!/usr/bin/env python3
"""Generate isolated Paper-I comparator SPSA calibration records.

This lane is deliberately separate from the Phase3/Route-A/SNAKE Optuna record
surfaces.  It emits one record per visible non-SNAKE comparator method and
calibration target for full approved configs, or a tiny deterministic smoke slice
from the tracked smoke config.  The generated rows are calibration inputs only;
they are not manuscript/table evidence.
"""
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

from pipelines.exact_bench.paper_i_comparator_spsa_calibration import (  # noqa: E402
    HH_GEO_QEB_TABLEIII_REPAIR_RECORD_MODE,
    HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,
    HH_GEO_QEB_TABLEIII_REPAIR_SMOKE_RECORD_MODE,
    HH_TABLEIII_REPAIR_METHOD_IDS,
    HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC,
    HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY,
    HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE,
    HH_TABLEIII_REPAIR_TABLE_LABEL,
    HH_TABLEIII_REPAIR_TARGET_IDS,
    HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY,
    PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_VERSION,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
    PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS,
    PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE,
    PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS,
    PAPER_I_NATIVE_SPSA_REFIT_ENGINE,
    config_sha256_for_path,
    full_method_target_records,
    hh_tableiii_repair_method_target_records,
    hh_tableiii_repair_target_by_id,
    load_and_validate_config,
    native_spsa_adaptive_method_target_records,
    target_by_id,
    validate_full_method_target_records,
    validate_hh_tableiii_repair_records,
    validate_method_id,
    validate_native_spsa_adaptive_records,
)
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import (  # noqa: E402
    PAPER_I_MAIN_TABLES_SPSA_ADAPT_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS,
    PAPER_I_MAIN_TABLES_SPSA_FAMILY_INFORMED_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_HEA_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
    PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS,
    PAPER_I_MAIN_TABLES_SPSA_TARGET,
    PAPER_I_MAIN_TABLES_SPSA_TARGET_LABEL,
)
from pipelines.exact_bench.static_reference_metrics import exact_energy_for_spec  # noqa: E402
from pipelines.exact_bench.table_i_canonical_cases import table_i_canonical_spec_by_case_id  # noqa: E402
from pipelines.exact_bench.table_i_static_benchmark import table_i_method_label  # noqa: E402

DEFAULT_CONFIG_PATH = REPO_ROOT / "chtc" / "phase3_optuna" / "config" / "paper_i_comparator_spsa_calibration_v1_smoke.json"
DEFAULT_HH_REPAIR_SMOKE_CONFIG_PATH = (
    REPO_ROOT / "chtc" / "phase3_optuna" / "config" / "paper_i_hh_geo_qeb_spsa_repair_v1_smoke.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "chtc" / "phase3_optuna" / "input" / "paper_i_comparator_spsa_calibration_v1_smoke"
DEFAULT_QUEUE_OUTPUT_ROOT = Path("raw_outputs/paper_i_comparator_spsa_calibration_v1")

RECORDS_TSV = "paper_i_comparator_spsa_calibration_records.tsv"
RECORD_IDS_TXT = "paper_i_comparator_spsa_calibration_record_ids.txt"
SMOKE_RECORDS_TSV = "paper_i_comparator_spsa_calibration_smoke_records.tsv"
SMOKE_RECORD_IDS_TXT = "paper_i_comparator_spsa_calibration_smoke_record_ids.txt"
MANIFEST_JSON = "paper_i_comparator_spsa_calibration_records_manifest.json"

RESOURCE_METRIC_PRECEDENCE = (
    "compiled_count_2q_total",
    "compiled_depth_2q_total",
    "compiled_depth_total",
    "first_hit_compiled_count_2q_total",
    "first_hit_compiled_depth_2q_total",
    "qiskit_compiled_count_2q_total",
    "qiskit_compiled_depth_2q_total",
    "two_qubit_count",
    "depth_2q",
    "circuit_depth",
    "num_parameters",
    "selected_operator_count",
)

ADAPT_METHOD_IDS = {
    "static_full_meta_append_adapt_vqe",
    "static_qubit_qeb_adapt_vqe",
    "static_tetris_qubit_adapt_vqe",
    "static_geo_adapt_vqe",
}
RESOURCE_GUARD_TSV_FIELDS = ("resource_qubit_cap", "resource_pool_term_cap")
GENERIC_ADAPT_REPAIR_TSV_FIELDS: tuple[str, ...] = ()
HH_TABLEIII_REPAIR_TSV_FIELDS = (
    "repair_scope",
    "calibration_stage",
    "table_label",
    "hh_tableiii_regime",
    "n_ph_work",
    "n_ph_ref",
    "phase3_adapt_max_depth",
    "same_cutoff_exact_gs_energy",
    "exact_reference_energy",
    "exact_reference_n_ph_max",
    "same_cutoff_reference_energy_key",
    "reference_cutoff_energy_key",
    "reference_energy_status",
    "primary_energy_metric",
    "same_cutoff_error_role",
    "calibration_usable_status_policy",
    "quality_nonpassing_penalty",
)
SMOKE_TARGET_IDS = (PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS[0],)
FIELDNAMES = (
    "record_id",
    "profile_id",
    "record_schema",
    "rerun_scope",
    "adaptive_refit_engine",
    "warm_start_schedule_lock_json",
    "warm_start_schedule_key",
    "run_class",
    "method_id",
    "algorithm_id",
    "method_label",
    "target_id",
    "target_family",
    "family",
    "case_ids_json",
    "case_count",
    "suite_profile",
    "optimizer_profile",
    "config_path",
    "config_sha256",
    "config_version",
    "config_mode",
    "approved_for_full_generation",
    "approved_by",
    "approved_at",
    "plan_path",
    "n_trials",
    "sampler_seed",
    "n_jobs",
    "method_maxiter_budget",
    "failure_penalty",
    "objective_mode",
    "target_abs_delta_e",
    "target_label",
    "clip_log10_error_ratio_min",
    "clip_log10_error_ratio_max",
    "resource_tiebreak_weight",
    "resource_metric_precedence_json",
    "search_space_fields_json",
    "queue_output_root",
    "record_output_dir",
    "progress_dir",
    "summary_json",
    "best_schedule_json",
    "heartbeat_json",
    "trial_output_template",
    *HH_TABLEIII_REPAIR_TSV_FIELDS,
    *RESOURCE_GUARD_TSV_FIELDS,
    *GENERIC_ADAPT_REPAIR_TSV_FIELDS,
    *PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS,
)


def _repo_relative(path: str | Path) -> str:
    candidate = Path(path)
    resolved = candidate.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(candidate)


def _json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load_warm_start_schedule_lock(path: str | Path | None) -> dict[str, Mapping[str, Any]]:
    if path is None or str(path).strip() == "":
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"warm-start schedule lock is not a JSON object: {path}")
    raw = payload.get("method_target_schedules")
    if not isinstance(raw, Mapping):
        raise ValueError(f"warm-start schedule lock missing method_target_schedules: {path}")
    out: dict[str, Mapping[str, Any]] = {}
    for key, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"warm-start schedule lock entry {key!r} is not an object")
        schedule = value.get("schedule")
        if not isinstance(schedule, Mapping) or not schedule:
            raise ValueError(f"warm-start schedule lock entry {key!r} missing non-empty schedule")
        out[str(key)] = value
    return out


def _blank_optimizer_fields() -> dict[str, str]:
    return {field: "" for field in PAPER_I_MAIN_TABLES_SPSA_OPTIMIZER_TSV_FIELDS}


def _repair_fields_for_method_target(*, method_id: str, target_id: str, generation_mode: str) -> dict[str, str]:
    target = target_by_id(target_id)
    fields = {
        **{field: "" for field in HH_TABLEIII_REPAIR_TSV_FIELDS},
        "resource_qubit_cap": "",
        "resource_pool_term_cap": "",
    }
    if target.family == "hh" and method_id in ADAPT_METHOD_IDS:
        fields["resource_qubit_cap"] = "0"
        fields["resource_pool_term_cap"] = "0"
    if target.family == "hh" and method_id in HH_TABLEIII_REPAIR_METHOD_IDS and target_id in HH_TABLEIII_REPAIR_TARGET_IDS:
        repair_target = hh_tableiii_repair_target_by_id(target_id)
        if target.case_ids != repair_target.case_ids:
            raise ValueError(
                f"HH Table III repair target drifted for {target_id}: {target.case_ids!r} != {repair_target.case_ids!r}"
            )
        spec = table_i_canonical_spec_by_case_id(target.family, repair_target.case_ids[0], PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID)
        same_energy, same_key, _same_payload = exact_energy_for_spec(spec, n_ph_max=int(repair_target.n_ph_work))
        ref_energy, ref_key, _ref_payload = exact_energy_for_spec(spec, n_ph_max=int(repair_target.n_ph_ref))
        depth_cap = 1 if str(generation_mode).strip().lower() == "smoke" else int(repair_target.adapt_max_depth)
        fields.update(
            {
                "repair_scope": HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,
                "calibration_stage": str(generation_mode).strip().lower(),
                "table_label": HH_TABLEIII_REPAIR_TABLE_LABEL,
                "hh_tableiii_regime": repair_target.hh_tableiii_regime,
                "n_ph_work": str(int(repair_target.n_ph_work)),
                "n_ph_ref": str(int(repair_target.n_ph_ref)),
                "phase3_adapt_max_depth": str(int(depth_cap)),
                "same_cutoff_exact_gs_energy": repr(float(same_energy)),
                "exact_reference_energy": repr(float(ref_energy)),
                "exact_reference_n_ph_max": str(int(repair_target.n_ph_ref)),
                "same_cutoff_reference_energy_key": same_key,
                "reference_cutoff_energy_key": ref_key,
                "reference_energy_status": "ok",
                "primary_energy_metric": HH_TABLEIII_REPAIR_PRIMARY_ENERGY_METRIC,
                "same_cutoff_error_role": HH_TABLEIII_REPAIR_SAME_CUTOFF_ERROR_ROLE,
                "calibration_usable_status_policy": HH_TABLEIII_REPAIR_USABLE_STATUS_POLICY,
                "quality_nonpassing_penalty": repr(float(HH_TABLEIII_REPAIR_QUALITY_NONPASSING_PENALTY)),
            }
        )
    return fields


def _optimizer_fields_for_method(method_id: str, *, maxiter_budget: int) -> dict[str, str]:
    method = validate_method_id(method_id)
    fields = _blank_optimizer_fields()
    fields["optimizer_profile"] = PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
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
    elif method in ADAPT_METHOD_IDS:
        defaults = PAPER_I_MAIN_TABLES_SPSA_BUDGET_DEFAULTS["adapt"]
        fields["adapt_optimizer_kind"] = str(defaults["optimizer_kind"])
        fields["adapt_spsa_maxiter"] = str(int(maxiter_budget))
        fields["adapt_spsa_seed"] = str(int(defaults["spsa_seed"]))
    else:  # pragma: no cover - validate_method_id should make this unreachable.
        raise ValueError(f"Unsupported comparator calibration method_id={method!r}")
    return fields


def _stable_record_id(*, method_id: str, target_id: str, mode: str) -> str:
    return f"paper_i_comp_spsa_cal__{mode}__{method_id}__{target_id}"


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


def _reject_unsupported_filters(
    *,
    calibration_profile: str | None = None,
    family_filter: Sequence[str] | None = None,
    algorithm_filter: Sequence[str] | None = None,
) -> None:
    profile = str(calibration_profile or "off").strip().lower()
    if profile not in {"", "off", "none"}:
        raise ValueError(
            f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID} is isolated from Phase3/Route-A calibration_profile={calibration_profile!r}."
        )
    families = tuple(str(item).strip() for item in (family_filter or ()) if str(item).strip())
    if families:
        raise ValueError(
            f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID} must generate the closed visible target matrix; family filters are not allowed: {families!r}."
        )
    algorithms = tuple(str(item).strip() for item in (algorithm_filter or ()) if str(item).strip())
    if algorithms:
        for method_id in algorithms:
            validate_method_id(method_id)
        if set(algorithms) != set(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS):
            raise ValueError(
                f"{PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID} must generate exactly the six visible non-SNAKE comparators; got {algorithms!r}."
            )


def _rows_for_method_targets(
    method_target_records: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, Any],
    config_path: str | Path,
    config_sha256: str,
    generation_mode: str,
    record_id_mode: str | None = None,
    queue_output_root: str | Path,
    rerun_scope: str | None = None,
    warm_start_schedule_lock_path: str | Path | None = None,
    warm_start_schedule_lock: Mapping[str, Mapping[str, Any]] | None = None,
    allow_partial_warm_start: bool = False,
) -> list[dict[str, str]]:
    budgets = config.get("method_maxiter_budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError("validated calibration config missing method_maxiter_budgets mapping")
    clipping = config.get("clipping_log10_error_ratio")
    if not isinstance(clipping, Sequence) or isinstance(clipping, (str, bytes)) or len(clipping) != 2:
        raise ValueError("validated calibration config has malformed clipping_log10_error_ratio")
    rows: list[dict[str, str]] = []
    for stub in method_target_records:
        method_id = validate_method_id(str(stub["method_id"]))
        target = target_by_id(str(stub["target_id"]))
        maxiter_budget = int(budgets[method_id])
        record_id = _stable_record_id(
            method_id=method_id,
            target_id=target.target_id,
            mode=str(record_id_mode or generation_mode),
        )
        paths = _record_paths(record_id)
        active_rerun_scope = str(rerun_scope or stub.get("rerun_scope") or "").strip()
        warm_start_key = f"{method_id}::{target.target_id}"
        warm_start_path = ""
        if warm_start_schedule_lock_path is not None and str(warm_start_schedule_lock_path).strip():
            lock = dict(warm_start_schedule_lock or {})
            if warm_start_key not in lock:
                if not allow_partial_warm_start:
                    raise ValueError(f"warm-start schedule lock {warm_start_schedule_lock_path} missing key {warm_start_key!r}")
            else:
                warm_start_path = _repo_relative(warm_start_schedule_lock_path)
        row: dict[str, str] = {
            "record_id": record_id,
            "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
            "record_schema": "paper_i_comparator_spsa_calibration_record_v1",
            "rerun_scope": active_rerun_scope,
            "adaptive_refit_engine": PAPER_I_NATIVE_SPSA_REFIT_ENGINE
            if active_rerun_scope == PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE
            else str(stub.get("adaptive_refit_engine") or ""),
            "warm_start_schedule_lock_json": warm_start_path,
            "warm_start_schedule_key": warm_start_key if warm_start_path else "",
            "run_class": "smoke" if generation_mode == "smoke" else "calibration_candidate_not_table_evidence",
            "method_id": method_id,
            "algorithm_id": method_id,
            "method_label": table_i_method_label(method_id),
            "target_id": target.target_id,
            "target_family": target.family,
            "family": target.family,
            "case_ids_json": _json_compact(list(target.case_ids)),
            "case_count": str(len(target.case_ids)),
            "suite_profile": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
            "optimizer_profile": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
            "config_path": _repo_relative(config_path),
            "config_sha256": str(config_sha256),
            "config_version": PAPER_I_COMPARATOR_SPSA_CALIBRATION_CONFIG_VERSION,
            "config_mode": str(config.get("mode")),
            "approved_for_full_generation": "true" if bool(config.get("approved_for_full_generation")) else "false",
            "approved_by": "" if config.get("approved_by") is None else str(config.get("approved_by")),
            "approved_at": "" if config.get("approved_at") is None else str(config.get("approved_at")),
            "plan_path": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH,
            "n_trials": str(int(config["n_trials"])),
            "sampler_seed": str(int(config["sampler_seed"])),
            "n_jobs": "1",
            "method_maxiter_budget": str(maxiter_budget),
            "failure_penalty": str(float(config["failure_penalty"])),
            "objective_mode": "mean_clipped_log10_abs_delta_e_plus_resource_tiebreak_v1",
            "target_abs_delta_e": str(float(PAPER_I_MAIN_TABLES_SPSA_TARGET)),
            "target_label": PAPER_I_MAIN_TABLES_SPSA_TARGET_LABEL,
            "clip_log10_error_ratio_min": str(float(clipping[0])),
            "clip_log10_error_ratio_max": str(float(clipping[1])),
            "resource_tiebreak_weight": str(float(config["resource_tiebreak_weight"])),
            "resource_metric_precedence_json": _json_compact(list(RESOURCE_METRIC_PRECEDENCE)),
            "search_space_fields_json": _json_compact(
                sorted(str(name) for name in config["per_method_search_spaces"][method_id])
            ),
            "queue_output_root": str(queue_output_root),
            **paths,
        }
        row.update(
            _repair_fields_for_method_target(
                method_id=method_id,
                target_id=target.target_id,
                generation_mode=generation_mode,
            )
        )
        row.update(_optimizer_fields_for_method(method_id, maxiter_budget=maxiter_budget))
        # Keep method-specific search schedule fields blank in records.  The
        # runner writes trial suggestions into env overlays under strict
        # save/restore, so rows never bake a stale schedule.
        for field in PAPER_I_MAIN_TABLES_SPSA_SCHEDULE_TSV_FIELDS:
            row.setdefault(field, "")
        rows.append(row)
    rows.sort(key=lambda item: (item["method_id"], item["target_id"], item["record_id"]))
    return rows


def _select_smoke_stubs() -> tuple[dict[str, object], ...]:
    return full_method_target_records(
        method_ids=PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS,
        target_ids=SMOKE_TARGET_IDS,
    )


def build_rows(
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    generation_mode: str | None = None,
    queue_output_root: str | Path = DEFAULT_QUEUE_OUTPUT_ROOT,
    repair_scope: str | None = None,
    rerun_scope: str | None = None,
    warm_start_schedule_lock: str | Path | None = None,
    allow_partial_warm_start: bool = False,
    calibration_profile: str | None = None,
    family_filter: Sequence[str] | None = None,
    algorithm_filter: Sequence[str] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Return generated rows, smoke rows, and the validated config.

    ``generation_mode='full'`` fails unless the config itself is an approved full
    config.  The tracked smoke config can only produce the deterministic smoke
    slice.
    """

    _reject_unsupported_filters(
        calibration_profile=calibration_profile,
        family_filter=family_filter,
        algorithm_filter=algorithm_filter,
    )
    config = load_and_validate_config(config_path)
    mode = str(generation_mode or config["mode"]).strip().lower()
    if mode not in {"smoke", "full"}:
        raise ValueError(f"generation_mode must be smoke or full; got {generation_mode!r}")
    scope = str(repair_scope or "").strip()
    active_rerun_scope = str(rerun_scope or "").strip()
    if scope and active_rerun_scope:
        raise ValueError("repair_scope and rerun_scope are mutually exclusive.")
    if scope and scope != HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
        raise ValueError(
            f"Unsupported repair_scope={repair_scope!r}; known: {HH_GEO_QEB_TABLEIII_REPAIR_SCOPE!r}."
        )
    if active_rerun_scope and active_rerun_scope != PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
        raise ValueError(
            f"Unsupported rerun_scope={rerun_scope!r}; known: {PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE!r}."
        )
    warm_lock_path = Path(warm_start_schedule_lock) if warm_start_schedule_lock is not None and str(warm_start_schedule_lock).strip() else None
    warm_lock = _load_warm_start_schedule_lock(warm_lock_path) if warm_lock_path is not None else {}
    if warm_lock_path is not None and active_rerun_scope != PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
        raise ValueError("warm_start_schedule_lock is currently supported only for the native SPSA adaptive rerun scope")
    if mode == "full" and str(config.get("mode")) != "full":
        raise ValueError("full generation requires a config with mode='full' and approved full-generation metadata")
    config_hash = str(config.get("config_sha256") or config_sha256_for_path(config_path))
    if mode == "full":
        if scope == HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
            full_stubs = hh_tableiii_repair_method_target_records()
            validate_hh_tableiii_repair_records(full_stubs)
            expected_count = len(HH_TABLEIII_REPAIR_METHOD_IDS) * len(HH_TABLEIII_REPAIR_TARGET_IDS)
            record_id_mode = HH_GEO_QEB_TABLEIII_REPAIR_RECORD_MODE
        elif active_rerun_scope == PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
            full_stubs = native_spsa_adaptive_method_target_records()
            validate_native_spsa_adaptive_records(full_stubs)
            expected_count = len(PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS) * len(PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS)
            record_id_mode = PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE
        else:
            full_stubs = full_method_target_records()
            validate_full_method_target_records(full_stubs)
            expected_count = 36
            record_id_mode = None
        records = _rows_for_method_targets(
            full_stubs,
            config=config,
            config_path=config_path,
            config_sha256=config_hash,
            generation_mode="full",
            record_id_mode=record_id_mode,
            queue_output_root=queue_output_root,
            rerun_scope=active_rerun_scope,
            warm_start_schedule_lock_path=warm_lock_path,
            warm_start_schedule_lock=warm_lock,
            allow_partial_warm_start=allow_partial_warm_start,
        )
        if len(records) != expected_count:
            raise ValueError(f"full generation expected {expected_count} rows; got {len(records)}")
        # Keep smoke artifacts cheap even when a full approved config is used.
        # They are generated from the tracked smoke config, not from the full
        # search budget, so the smoke TSV cannot silently become expensive while
        # retaining smoke IDs/classes.
        if scope == HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
            smoke_stubs = hh_tableiii_repair_method_target_records()
            smoke_record_id_mode = HH_GEO_QEB_TABLEIII_REPAIR_SMOKE_RECORD_MODE
            smoke_config_path = DEFAULT_HH_REPAIR_SMOKE_CONFIG_PATH
        elif active_rerun_scope == PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
            smoke_stubs = native_spsa_adaptive_method_target_records(target_ids=SMOKE_TARGET_IDS)
            smoke_record_id_mode = f"{PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE}_smoke"
            smoke_config_path = DEFAULT_CONFIG_PATH
        else:
            smoke_stubs = _select_smoke_stubs()
            smoke_record_id_mode = None
            smoke_config_path = DEFAULT_CONFIG_PATH
        smoke_config = load_and_validate_config(smoke_config_path)
        smoke_records = _rows_for_method_targets(
            smoke_stubs,
            config=smoke_config,
            config_path=smoke_config_path,
            config_sha256=str(smoke_config["config_sha256"]),
            generation_mode="smoke",
            record_id_mode=smoke_record_id_mode,
            queue_output_root=queue_output_root,
            rerun_scope=active_rerun_scope,
        )
    else:
        if scope == HH_GEO_QEB_TABLEIII_REPAIR_SCOPE:
            stubs = hh_tableiii_repair_method_target_records()
            record_id_mode = HH_GEO_QEB_TABLEIII_REPAIR_SMOKE_RECORD_MODE
        elif active_rerun_scope == PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE:
            stubs = native_spsa_adaptive_method_target_records(target_ids=SMOKE_TARGET_IDS)
            record_id_mode = f"{PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE}_smoke"
        else:
            stubs = _select_smoke_stubs()
            record_id_mode = None
        records = _rows_for_method_targets(
            stubs,
            config=config,
            config_path=config_path,
            config_sha256=config_hash,
            generation_mode="smoke",
            record_id_mode=record_id_mode,
            queue_output_root=queue_output_root,
            rerun_scope=active_rerun_scope,
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
    repair_scope: str | None = None,
    rerun_scope: str | None = None,
    warm_start_schedule_lock: str | Path | None = None,
    allow_partial_warm_start: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    records_path = output / RECORDS_TSV
    ids_path = output / RECORD_IDS_TXT
    smoke_path = output / SMOKE_RECORDS_TSV
    smoke_ids_path = output / SMOKE_RECORD_IDS_TXT
    manifest_path = output / MANIFEST_JSON
    is_repair = str(repair_scope or "").strip() == HH_GEO_QEB_TABLEIII_REPAIR_SCOPE
    is_native_adaptive = str(rerun_scope or "").strip() == PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE
    warm_lock_path = Path(warm_start_schedule_lock) if warm_start_schedule_lock is not None and str(warm_start_schedule_lock).strip() else None
    smoke_config_path = DEFAULT_HH_REPAIR_SMOKE_CONFIG_PATH if is_repair else DEFAULT_CONFIG_PATH
    _write_tsv(records_path, records)
    _write_ids(ids_path, records)
    _write_tsv(smoke_path, smoke_records)
    _write_ids(smoke_ids_path, smoke_records)
    method_targets = sorted({(row["method_id"], row["target_id"]) for row in records})
    warm_started_rows = [row for row in records if str(row.get("warm_start_schedule_key") or "").strip()]
    cold_start_rows = [
        row
        for row in records
        if is_native_adaptive
        and warm_lock_path is not None
        and not str(row.get("warm_start_schedule_key") or "").strip()
    ]
    missing_warm_start_keys = sorted(f"{row['method_id']}::{row['target_id']}" for row in cold_start_rows)
    manifest = {
        "schema": "paper_i_comparator_spsa_calibration_records_manifest_v1",
        "profile_id": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PROFILE_ID,
        "generation_mode": str(generation_mode),
        "run_class": "smoke" if generation_mode == "smoke" else "calibration_candidate_not_table_evidence",
        "evidence_role": "calibration_only_not_manuscript_table_evidence",
        "suite_profile": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        "optimizer_profile": PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        "config_path": _repo_relative(config_path),
        "config_sha256": str(config.get("config_sha256") or config_sha256_for_path(config_path)),
        "config_mode": str(config.get("mode")),
        "repair_scope": None if str(repair_scope or "").strip() == "" else str(repair_scope),
        "rerun_scope": PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE if is_native_adaptive else None,
        "adaptive_refit_engine": PAPER_I_NATIVE_SPSA_REFIT_ENGINE if is_native_adaptive else None,
        "warm_start_schedule_lock_json": None if warm_lock_path is None else _repo_relative(warm_lock_path),
        "warm_start_schedule_lock_sha256": None if warm_lock_path is None else config_sha256_for_path(warm_lock_path),
        "warm_start_schedule_allow_partial": bool(allow_partial_warm_start),
        "warm_start_record_count": len(warm_started_rows),
        "cold_start_record_count": len(cold_start_rows),
        "missing_warm_start_keys": missing_warm_start_keys,
        "approved_for_full_generation": bool(config.get("approved_for_full_generation")),
        "approved_by": config.get("approved_by"),
        "approved_at": config.get("approved_at"),
        "plan_path": PAPER_I_COMPARATOR_SPSA_CALIBRATION_PLAN_PATH,
        "record_count": len(records),
        "smoke_record_count": len(smoke_records),
        "expected_full_record_count": 8 if is_repair else 24 if is_native_adaptive else 36,
        "method_ids": list(HH_TABLEIII_REPAIR_METHOD_IDS)
        if is_repair
        else list(PAPER_I_NATIVE_SPSA_AFFECTED_ADAPTIVE_METHOD_IDS)
        if is_native_adaptive
        else list(PAPER_I_COMPARATOR_SPSA_CALIBRATION_ALLOWED_METHOD_IDS),
        "target_ids": list(HH_TABLEIII_REPAIR_TARGET_IDS) if is_repair else list(PAPER_I_COMPARATOR_SPSA_CALIBRATION_TARGET_IDS),
        "method_target_pairs": [list(pair) for pair in method_targets],
        "smoke_target_ids": list(HH_TABLEIII_REPAIR_TARGET_IDS) if is_repair else list(SMOKE_TARGET_IDS),
        "n_trials": int(config["n_trials"]),
        "sampler_seed": int(config["sampler_seed"]),
        "n_jobs": 1,
        "method_maxiter_budgets": dict(config["method_maxiter_budgets"]),
        "failure_penalty": float(config["failure_penalty"]),
        "clipping_log10_error_ratio": list(config["clipping_log10_error_ratio"]),
        "resource_tiebreak_weight": float(config["resource_tiebreak_weight"]),
        "resource_metric_precedence": list(RESOURCE_METRIC_PRECEDENCE),
        "allowed_schedule_fields_by_method": {
            method: list(fields) for method, fields in PAPER_I_COMPARATOR_SPSA_ALLOWED_SCHEDULE_FIELDS_BY_METHOD.items()
        },
        "queue_output_root": str(queue_output_root),
        "smoke_config_path": _repo_relative(smoke_config_path),
        "smoke_config_sha256": config_sha256_for_path(smoke_config_path),
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
    repair_scope: str | None = None,
    rerun_scope: str | None = None,
    warm_start_schedule_lock: str | Path | None = None,
    allow_partial_warm_start: bool = False,
    calibration_profile: str | None = None,
    family_filter: Sequence[str] | None = None,
    algorithm_filter: Sequence[str] | None = None,
) -> dict[str, Any]:
    records, smoke_records, config = build_rows(
        config_path=config_path,
        generation_mode=generation_mode,
        queue_output_root=queue_output_root,
        repair_scope=repair_scope,
        rerun_scope=rerun_scope,
        warm_start_schedule_lock=warm_start_schedule_lock,
        allow_partial_warm_start=allow_partial_warm_start,
        calibration_profile=calibration_profile,
        family_filter=family_filter,
        algorithm_filter=algorithm_filter,
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
        repair_scope=repair_scope,
        rerun_scope=rerun_scope,
        warm_start_schedule_lock=warm_start_schedule_lock,
        allow_partial_warm_start=allow_partial_warm_start,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Paper-I comparator SPSA calibration record TSVs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", dest="config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--mode", choices=("smoke", "full"), default=None, help="Generation mode; defaults to config mode.")
    parser.add_argument("--queue-output-root", type=Path, default=DEFAULT_QUEUE_OUTPUT_ROOT)
    parser.add_argument(
        "--repair-scope",
        default=None,
        choices=(HH_GEO_QEB_TABLEIII_REPAIR_SCOPE,),
        help="Optional explicit repair subset; currently HH Table III Geo/Qubit-QEB only.",
    )
    parser.add_argument(
        "--rerun-scope",
        default=None,
        choices=(PAPER_I_NATIVE_SPSA_ADAPTIVE_RERUN_SCOPE,),
        help="Optional native-SPSA rerun subset for affected adaptive exact-bench comparators.",
    )
    parser.add_argument(
        "--warm-start-schedule-lock",
        type=Path,
        default=None,
        help="Optional schedule-lock candidate JSON used to enqueue one warm-start trial per native adaptive record.",
    )
    parser.add_argument(
        "--allow-partial-warm-start",
        action="store_true",
        help="Allow a native adaptive warm-start lock to omit specific method-target keys; omitted rows cold-start explicitly.",
    )
    parser.add_argument("--calibration-profile", default="off", help="Rejected unless off; prevents Phase3/Route-A mixing.")
    parser.add_argument("--family", action="append", dest="family_filter", default=None, help="Rejected: this lane is a closed target matrix.")
    parser.add_argument("--algorithm-id", action="append", dest="algorithm_filter", default=None, help="Rejected unless it names exactly the six visible comparators.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_records(
        output_dir=args.output_dir,
        config_path=args.config_path,
        generation_mode=args.mode,
        queue_output_root=args.queue_output_root,
        repair_scope=args.repair_scope,
        rerun_scope=args.rerun_scope,
        warm_start_schedule_lock=args.warm_start_schedule_lock,
        allow_partial_warm_start=args.allow_partial_warm_start,
        calibration_profile=args.calibration_profile,
        family_filter=args.family_filter,
        algorithm_filter=args.algorithm_filter,
    )
    print(json.dumps(manifest["paths"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
