#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chtc.paper_iii_excited_dynamics.preflight_inputs import (  # noqa: E402
    MODE_COMPATIBILITY_AUDIT_ONLY,
    MODE_REPORT_ONLY_EXISTING_OUTPUT,
    MODE_STRICT_HH,
    load_record_ids,
    parse_bool,
)
from pipelines.excited_dynamics.paper_iii_local_science_pilot import (  # noqa: E402
    PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE,
    PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION,
    validate_strict_hh_payload,
)

DEFAULT_ROOT = Path("raw_outputs") / "paper_iii_excited_dynamics"
VALIDATION_SCHEMA_VERSION = "paper_iii_excited_dynamics_validation_report_v1"
COMPATIBILITY_MATRIX_VALIDATION_SCHEMA_VERSION = "paper_iii_compatibility_matrix_validation_report_v1"
COMPATIBILITY_MATRIX_MANIFEST_SCHEMA_VERSION = "paper_iii_compatibility_matrix_manifest_v1"
COMPATIBILITY_MATRIX_PROFILE = "paper_iii_nph1_compatibility_v1"
COMPATIBILITY_MATRIX_REQUIRED_FILES = (
    "manifest.json",
    "records.tsv",
    "full_record_ids.txt",
    "smoke_record_ids.txt",
    "blockers.tsv",
    "blockers.json",
    "compatibility_audit_summary.md",
    "seed_repair_inventory.tsv",
    "seed_repair_inventory.json",
    "seed_repair_summary.md",
)
SEED_REPAIR_INVENTORY_SCHEMA_VERSION = "paper_iii_seed_repair_inventory_v1"
SEED_REPAIR_ALLOWED_STATUSES = {
    "repairable_local",
    "missing_local",
    "staged_or_recovery_only",
    "path_escapes_repo",
    "manual_decision_required",
}
SEED_REPAIR_EXPECTED_UNIQUE_SLOTS = 36
COMPATIBILITY_MATRIX_DECLARED_CASES_BY_FAMILY: dict[str, tuple[str, ...]] = {
    "hubbard": ("hubbard_L2", "hubbard_L2_u6"),
    "ionic_hubbard": ("ionic_hubbard_L2", "ionic_hubbard_L2_u6"),
    "extended_hubbard": ("extended_hubbard_L2", "extended_hubbard_L2_u6"),
    "ttprime_hubbard": ("ttprime_hubbard_L2", "ttprime_hubbard_L2_u6"),
    "spinless_tv": ("spinless_tv_L2", "spinless_tv_L2_v1p5"),
    "bose_hubbard": ("bose_hubbard_L2", "bose_hubbard_L2_u2"),
    "harmonic_kerr_chain": ("harmonic_kerr_chain_L2", "harmonic_kerr_chain_L2_w0p75"),
    "spin_boson": ("spin_boson_L1", "spin_boson_L1_g0p7"),
    "hh": ("hh_L2",),
    "molecular_vibronic_h2": ("molecular_vibronic_h2_L2",),
}
COMPATIBILITY_MATRIX_COMPARATOR_IDS = (
    "geo_selected_qse",
    "fixed_alphabet_qse",
    "family_restricted_qse",
    "full_meta_largest_alphabet_qse",
    "hamiltonian_derived_qse",
    "krylov_hamiltonian_power_qse",
    "qeom_qsc_eom",
    "qse_coefficient_propagation",
)
COMPATIBILITY_MATRIX_DRIVE_AMPLITUDES = ("0.2", "0.6")
COMPATIBILITY_MATRIX_EXPECTED_CASE_COUNT = sum(len(v) for v in COMPATIBILITY_MATRIX_DECLARED_CASES_BY_FAMILY.values())
COMPATIBILITY_MATRIX_EXPECTED_COMPARATOR_ROWS = (
    COMPATIBILITY_MATRIX_EXPECTED_CASE_COUNT
    * len(COMPATIBILITY_MATRIX_COMPARATOR_IDS)
    * len(COMPATIBILITY_MATRIX_DRIVE_AMPLITUDES)
)
COMPATIBILITY_MATRIX_EXPECTED_TOTAL_RECORDS = COMPATIBILITY_MATRIX_EXPECTED_COMPARATOR_ROWS + 1
LEAK_TRUE_KEYS = {
    "feeds_controller_decisions",
    "uses_reference_for_decision",
    "uses_future_exact_forecast_for_decision",
}
RAW_MARKER_DENYLIST = (
    "amplitudes_qn_to_q0",
    "raw_physical_state",
    "basis_matrix_vectors",
    "exact_target_trajectories",
    "exact_step_forecast",
    "state_at(",
)


@dataclass
class ValidationResult:
    record_id: str
    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    mode: str | None = None
    strict_pass: bool | None = None
    leakage_failure: bool = False
    blocker_count: int = 0
    report_json: str | None = None
    run_manifest_json: str | None = None

    def error(self, message: str, *, leakage: bool = False) -> None:
        self.ok = False
        self.errors.append(message)
        if leakage:
            self.leakage_failure = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "mode": self.mode,
            "strict_pass": self.strict_pass,
            "leakage_failure": self.leakage_failure,
            "blocker_count": self.blocker_count,
            "report_json": self.report_json,
            "run_manifest_json": self.run_manifest_json,
        }


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _compatibility_profile_for_n_ph(n_ph_max: int) -> str:
    return f"paper_iii_nph{int(n_ph_max)}_compatibility_v1"


def _relative_for_error(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def resolve_fetched_path(path_value: Any, record_dir: str | Path) -> Path | None:
    if path_value in {None, ""}:
        return None
    record_root = Path(record_dir)
    record_id = record_root.name
    raw = Path(str(path_value))
    if raw.exists():
        return raw
    candidates: list[Path] = []
    parts = raw.parts
    for idx, part in enumerate(parts[:-1]):
        if part == record_id:
            candidates.append(record_root.joinpath(*parts[idx + 1 :]))
    for idx, part in enumerate(parts[:-1]):
        if part == "paper_iii_excited_dynamics" and idx + 1 < len(parts) and parts[idx + 1] == record_id:
            candidates.append(record_root.joinpath(*parts[idx + 2 :]))
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)
        candidates.append(record_root.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else raw


def _find_key_values(value: Any, *, key: str, prefix: str = "") -> list[tuple[str, Any]]:
    hits: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for raw_key, raw_value in value.items():
            path = f"{prefix}.{raw_key}" if prefix else str(raw_key)
            if raw_key == key:
                hits.append((path, raw_value))
            hits.extend(_find_key_values(raw_value, key=key, prefix=path))
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            path = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_key_values(item, key=key, prefix=path))
    return hits


def _scan_for_leakage(value: Any, *, context: str) -> list[str]:
    violations: list[str] = []
    for key in LEAK_TRUE_KEYS:
        for path, raw_value in _find_key_values(value, key=key):
            if raw_value is True:
                violations.append(f"{context}:{path}=true")
    for path, raw_value in _find_key_values(value, key="decision_backend"):
        if raw_value == "exact":
            violations.append(f"{context}:{path}=exact")
    for path, raw_value in _find_key_values(value, key="controller_exact_input_mode"):
        if raw_value != "off":
            violations.append(f"{context}:{path} must be off")
    for path, raw_value in _find_key_values(value, key="exact_decision_checkpoints"):
        if _as_int(raw_value, 0) > 0:
            violations.append(f"{context}:{path}={raw_value}")
    return violations


def _raw_marker_hits(value: Any) -> list[str]:
    try:
        serialized = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        return ["nonfinite_or_nonserializable_json_value"]
    return [marker for marker in RAW_MARKER_DENYLIST if marker in serialized]


def _scan_loaded_artifact(value: Any, *, context: str, result: ValidationResult) -> None:
    for violation in _scan_for_leakage(value, context=context):
        result.error(violation, leakage=True)
    marker_hits = _raw_marker_hits(value)
    if marker_hits:
        result.error(f"{context} contains forbidden raw/reference marker(s): {marker_hits}", leakage=True)


def _require_file(path: Path, record_dir: Path, result: ValidationResult) -> None:
    if not path.exists():
        result.error(f"missing required file: {_relative_for_error(path, record_dir)}")


def _load_common(record_dir: Path, result: ValidationResult) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    record_path = record_dir / "record.json"
    status_path = record_dir / "chtc_status.json"
    task_result_path = record_dir / "task_result.json"
    command_path = record_dir / "command.sh"
    for path in (record_path, status_path, task_result_path, command_path):
        _require_file(path, record_dir, result)
    if not result.ok:
        return None
    try:
        record = _load_json(record_path)
        status = _load_json(status_path)
        task_result = _load_json(task_result_path)
    except Exception as exc:
        result.error(f"failed to load common JSON: {type(exc).__name__}: {exc}")
        return None

    result.mode = _clean(record.get("mode"))
    for label, payload in (("record.json", record), ("chtc_status.json", status), ("task_result.json", task_result)):
        payload_record_id = payload.get("record_id")
        if payload_record_id not in {None, result.record_id}:
            result.error(f"{label} record_id {payload_record_id!r} != directory record_id {result.record_id!r}")
    try:
        status_rc = int(status.get("return_code"))
    except Exception:
        status_rc = 1
    try:
        task_rc = int(task_result.get("return_code"))
    except Exception:
        task_rc = None
    if status_rc != 0:
        result.error(f"chtc_status return_code is {status_rc}")
    if task_rc != status_rc:
        result.error(f"task_result return_code {task_rc} != chtc_status return_code {status_rc}")

    report_path = resolve_fetched_path(
        task_result.get("report_json") or record_dir / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json",
        record_dir,
    )
    manifest_path = resolve_fetched_path(
        task_result.get("run_manifest_json") or record_dir / "paper_iii_local_science_pilot" / "run_manifest.json",
        record_dir,
    )
    if report_path is None:
        report_path = record_dir / "paper_iii_local_science_pilot" / "paper_iii_local_science_pilot_report.json"
    if manifest_path is None:
        manifest_path = record_dir / "paper_iii_local_science_pilot" / "run_manifest.json"
    _require_file(report_path, record_dir, result)
    _require_file(manifest_path, record_dir, result)
    if not result.ok:
        return None
    try:
        report = _load_json(report_path)
        manifest = _load_json(manifest_path)
    except Exception as exc:
        result.error(f"failed to load report/manifest JSON: {type(exc).__name__}: {exc}")
        return None
    result.report_json = str(report_path)
    result.run_manifest_json = str(manifest_path)
    return record, status, task_result, report, manifest


def _mapping(value: Any, *, result: ValidationResult, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        result.error(f"{name} must be a JSON object")
        return {}
    return value


def _validate_strict_route(report: Mapping[str, Any], *, result: ValidationResult) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    runs = _mapping(report.get("runs"), result=result, name="report.runs")
    run = _mapping(runs.get("strict_hh_runtime_dynamics"), result=result, name="strict_hh_runtime_dynamics")
    metrics = _mapping(run.get("metrics"), result=result, name="strict_hh_runtime_dynamics.metrics")
    strict_route = _mapping(metrics.get("strict_route"), result=result, name="strict_route")
    strict_validation = _mapping(run.get("strict_validation"), result=result, name="strict_validation")
    result.strict_pass = bool(strict_validation.get("passed"))
    if strict_validation.get("passed") is not True:
        result.error(f"strict_validation.passed is not true: {strict_validation.get('violations', [])}", leakage=True)
    expected = {
        "controller_exact_input_mode": "off",
        "diagnostic_exact_reference_mode": "benchmark_exact",
        "uses_reference_for_decision": False,
        "uses_future_exact_forecast_for_decision": False,
        "qpu_faithful_decisions_passed": True,
        "strict_measurement_oracle_certified": True,
    }
    for key, expected_value in expected.items():
        if strict_route.get(key) != expected_value:
            result.error(f"strict_route.{key} expected {expected_value!r}, got {strict_route.get(key)!r}", leakage=True)
    if _as_int(strict_route.get("exact_decision_checkpoints"), -1) != 0:
        result.error("strict_route.exact_decision_checkpoints must be 0", leakage=True)
    return run, strict_route


def _validate_payload_source(
    *,
    run: Mapping[str, Any],
    task_result: Mapping[str, Any],
    record: Mapping[str, Any],
    record_dir: Path,
    result: ValidationResult,
) -> None:
    payload_values = [
        run.get("strict_payload_source_json"),
        run.get("output_json"),
        task_result.get("existing_strict_output_json"),
        task_result.get("strict_output_json"),
    ]
    payload_path: Path | None = None
    payload_value: Any = None
    for candidate_value in payload_values:
        candidate_path = resolve_fetched_path(candidate_value, record_dir)
        if candidate_path is not None and candidate_path.exists():
            payload_path = candidate_path
            payload_value = candidate_value
            break
    if payload_path is None:
        result.error(f"strict payload source JSON not found after remap: {payload_values!r}")
        return
    try:
        payload = _load_json(payload_path)
        _scan_loaded_artifact(payload, context=f"strict_payload:{payload_path.name}", result=result)
        validation = validate_strict_hh_payload(
            payload,
            expected_t_final=float(record.get("t_final") or 1.0),
            expected_num_times=int(record.get("num_times") or 9),
        )
    except Exception as exc:
        result.error(f"strict payload validation failed to load: {type(exc).__name__}: {exc}", leakage=True)
        return
    if validation.get("passed") is not True:
        result.error(f"strict payload validation failed for {payload_value!r}: {validation.get('violations', [])}", leakage=True)


def validate_record_dir(record_dir: str | Path) -> ValidationResult:
    root = Path(record_dir)
    result = ValidationResult(record_id=root.name)
    common = _load_common(root, result)
    if common is None:
        return result
    record, status, task_result, report, manifest = common

    if report.get("schema_version") != PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION:
        result.error(
            "report schema_version expected "
            f"{PAPER_III_LOCAL_SCIENCE_PILOT_REPORT_SCHEMA_VERSION!r}, got {report.get('schema_version')!r}"
        )
    if report.get("pipeline") != PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE:
        result.error(f"report.pipeline expected {PAPER_III_LOCAL_SCIENCE_PILOT_PIPELINE!r}, got {report.get('pipeline')!r}")
    blockers = report.get("blockers") or []
    if not isinstance(blockers, list):
        result.error("report.blockers must be a list")
        blockers = ["invalid_blockers_shape"]
    result.blocker_count = len(blockers)
    if blockers:
        result.error(f"report blockers are not empty: {blockers}")
    if parse_bool(record.get("require_science_benchmark"), default=True) and report.get("paper_iii_science_benchmark") is not True:
        result.error("paper_iii_science_benchmark is not true")

    run, _strict_route = _validate_strict_route(report, result=result)
    command_status = _clean(run.get("command_status"))
    mode = _clean(record.get("mode"))
    if mode == MODE_REPORT_ONLY_EXISTING_OUTPUT:
        if command_status != "existing_output":
            result.error(f"report-only record command_status expected 'existing_output', got {command_status!r}")
        existing_values = [
            run.get("existing_strict_output_json"),
            task_result.get("existing_strict_output_json"),
            task_result.get("source_existing_strict_output_json"),
        ]
        existing_path = None
        for existing_value in existing_values:
            candidate_path = resolve_fetched_path(existing_value, root)
            if candidate_path is not None and candidate_path.exists():
                existing_path = candidate_path
                break
        if existing_path is None:
            result.error(f"existing strict output JSON not found after remap: {existing_values!r}")
        elif _clean(run.get("report_only_existing_output")) not in {"True", "true", "1"} and run.get("report_only_existing_output") is not True:
            result.error("report-only record did not preserve report_only_existing_output=true")
    elif mode == MODE_STRICT_HH:
        if command_status != "completed":
            result.error(f"strict_hh record command_status expected 'completed', got {command_status!r}")
        for label, key, run_key in (
            ("progress", "progress_json", "progress_json_exists"),
            ("partial payload", "partial_payload_json", "partial_payload_json_exists"),
        ):
            value = task_result.get(key) or run.get(key)
            path = resolve_fetched_path(value, root)
            exists_flag = bool(task_result.get(key.replace("_json", "_exists")) or run.get(run_key))
            if not exists_flag:
                result.error(f"strict_hh record {label} exists flag is false")
            if path is None or not path.exists():
                result.error(f"strict_hh record {label} JSON missing after remap: {value!r}")
            else:
                try:
                    sidecar_payload = _load_json(path)
                    _scan_loaded_artifact(sidecar_payload, context=f"strict_sidecar:{label}", result=result)
                except Exception as exc:
                    result.error(f"strict_hh record {label} JSON failed to load: {type(exc).__name__}: {exc}", leakage=True)
    else:
        result.error(f"unsupported record mode {mode!r}")

    _validate_payload_source(run=run, task_result=task_result, record=record, record_dir=root, result=result)

    for context, payload in (
        ("record", record),
        ("chtc_status", status),
        ("task_result", task_result),
        ("report", report),
        ("run_manifest", manifest),
    ):
        _scan_loaded_artifact(payload, context=context, result=result)
    return result


def validate_outputs(
    root: str | Path = DEFAULT_ROOT,
    *,
    record_ids: Sequence[str] | None = None,
    record_list: str | Path | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    output_root = Path(root)
    if record_ids:
        expected_ids: list[str] | None = [str(item) for item in record_ids]
    elif record_list is not None:
        expected_ids = load_record_ids(record_list)
    else:
        expected_ids = None
    results: list[ValidationResult] = []
    if expected_ids is None:
        record_dirs = sorted(path for path in output_root.iterdir() if path.is_dir()) if output_root.exists() else []
        results = [validate_record_dir(path) for path in record_dirs if (path / "record.json").exists() or (path / "chtc_status.json").exists()]
    else:
        for record_id in expected_ids:
            record_dir = output_root / record_id
            if not record_dir.exists():
                missing = ValidationResult(record_id=record_id, ok=False)
                missing.error(f"missing expected record output directory: {record_id}")
                results.append(missing)
            else:
                results.append(validate_record_dir(record_dir))
    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "root": str(output_root),
        "record_list": None if record_list is None else str(record_list),
        "expected_records": expected_ids,
        "record_count": len(results),
        "ok": bool(results) and all(item.ok for item in results),
        "strict_leakage_failure_count": sum(1 for item in results if item.leakage_failure),
        "leakage_failures": [item.record_id for item in results if item.leakage_failure],
        "records": [item.as_dict() for item in results],
        "failed_records": [item.record_id for item in results if not item.ok],
    }
    if not results:
        report["ok"] = False
        report["errors"] = ["no record output directories found"]
    if write_report:
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / "failed_records.txt").write_text(
            "\n".join(report["failed_records"]) + ("\n" if report["failed_records"] else ""),
            encoding="utf-8",
        )
    return report


def _load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def _split_codes(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]


def _matrix_bool(value: Any, *, default: bool = False) -> bool:
    try:
        return parse_bool(value, default=default)
    except ValueError:
        return default


def _matrix_int(value: Any, *, errors: list[str], label: str, default: int = -1) -> int:
    try:
        return int(value)
    except Exception:
        errors.append(f"{label} must be an integer, got {value!r}")
        return default


def _pairs_from_blocker_rows(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    out: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        record_id = _clean(row.get("record_id"))
        code = _clean(row.get("code"))
        if record_id and code:
            out.add((record_id, code))
    return out


def validate_compatibility_matrix_dir(matrix_dir: str | Path) -> dict[str, Any]:
    root = Path(matrix_dir)
    errors: list[str] = []
    warnings: list[str] = []
    missing_files = [name for name in COMPATIBILITY_MATRIX_REQUIRED_FILES if not (root / name).exists()]
    if missing_files:
        errors.extend(f"missing required matrix artifact: {name}" for name in missing_files)
        return {
            "schema_version": COMPATIBILITY_MATRIX_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _now_utc(),
            "compatibility_matrix_dir": str(root),
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "missing_files": missing_files,
        }

    try:
        manifest = _load_json(root / "manifest.json")
        records = _load_tsv(root / "records.tsv")
        blocker_tsv_rows = _load_tsv(root / "blockers.tsv")
        blockers_payload = _load_json(root / "blockers.json")
        full_ids = load_record_ids(root / "full_record_ids.txt")
        smoke_ids = load_record_ids(root / "smoke_record_ids.txt")
        summary_text = (root / "compatibility_audit_summary.md").read_text(encoding="utf-8")
        seed_inventory_tsv = _load_tsv(root / "seed_repair_inventory.tsv")
        seed_inventory_payload = _load_json(root / "seed_repair_inventory.json")
        seed_summary_text = (root / "seed_repair_summary.md").read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"failed to load matrix artifacts: {type(exc).__name__}: {exc}")
        return {
            "schema_version": COMPATIBILITY_MATRIX_VALIDATION_SCHEMA_VERSION,
            "generated_utc": _now_utc(),
            "compatibility_matrix_dir": str(root),
            "ok": False,
            "errors": errors,
            "warnings": warnings,
        }

    if manifest.get("schema_version") != COMPATIBILITY_MATRIX_MANIFEST_SCHEMA_VERSION:
        errors.append(f"manifest schema_version expected {COMPATIBILITY_MATRIX_MANIFEST_SCHEMA_VERSION!r}, got {manifest.get('schema_version')!r}")
    n_ph_max = _matrix_int(manifest.get("n_ph_max"), errors=errors, label="manifest.n_ph_max")
    if n_ph_max < 1:
        errors.append(f"manifest n_ph_max must be >= 1, got {n_ph_max}")
        n_ph_max = 1
    expected_profile = _compatibility_profile_for_n_ph(n_ph_max)
    if manifest.get("profile") != expected_profile:
        errors.append(f"manifest profile expected {expected_profile!r}, got {manifest.get('profile')!r}")
    if manifest.get("no_submit") is not True or manifest.get("no_submit_assertion") is not True:
        errors.append("manifest must assert no_submit=true and no_submit_assertion=true")
    if manifest.get("submits_chtc") is not False or manifest.get("runs_realtime_dynamics") is not False:
        errors.append("manifest must assert submits_chtc=false and runs_realtime_dynamics=false")
    blockers_json = blockers_payload.get("blockers")
    if not isinstance(blockers_json, list):
        errors.append("blockers.json must contain a blockers list")
        blockers_json = []
    elif any(not isinstance(row, Mapping) for row in blockers_json):
        errors.append("every blockers.json entry must be an object")
    if blockers_payload.get("no_submit") is not True:
        errors.append("blockers.json must assert no_submit=true")
    if "no submit" not in summary_text.lower() and "no-submit" not in summary_text.lower():
        errors.append("summary markdown must contain an explicit no-submit policy")

    record_ids = [row.get("record_id", "") for row in records]
    if len(record_ids) != len(set(record_ids)):
        errors.append("records.tsv contains duplicate record_id values")
    if full_ids != record_ids:
        errors.append("full_record_ids.txt must exactly match records.tsv order")
    smoke_set = set(smoke_ids)
    if not smoke_set <= set(record_ids):
        errors.append("smoke_record_ids.txt contains ids absent from records.tsv")

    expected_counts = manifest.get("expected_counts") if isinstance(manifest.get("expected_counts"), Mapping) else {}
    actual_counts = manifest.get("actual_counts") if isinstance(manifest.get("actual_counts"), Mapping) else {}
    comparator_rows = [row for row in records if _clean(row.get("row_kind")) == "comparator_row"]
    scope_rows = [row for row in records if _clean(row.get("row_kind")) == "scope_decision"]
    families = {_clean(row.get("family")) for row in comparator_rows}
    cases = {(_clean(row.get("family")), _clean(row.get("paper_i_case_id"))) for row in comparator_rows}
    comparators = {_clean(row.get("comparator_id")) for row in comparator_rows}
    drive_amplitudes = {_clean(row.get("drive_amplitude")) for row in comparator_rows}
    expected_cases = {
        (family, case_id)
        for family, case_ids in COMPATIBILITY_MATRIX_DECLARED_CASES_BY_FAMILY.items()
        for case_id in case_ids
    }
    expected_cross_product = {
        (family, case_id, comparator_id, amplitude)
        for family, case_id in expected_cases
        for comparator_id in COMPATIBILITY_MATRIX_COMPARATOR_IDS
        for amplitude in COMPATIBILITY_MATRIX_DRIVE_AMPLITUDES
    }
    actual_cross_product = {
        (
            _clean(row.get("family")),
            _clean(row.get("paper_i_case_id")),
            _clean(row.get("comparator_id")),
            _clean(row.get("drive_amplitude")),
        )
        for row in comparator_rows
    }
    manifest_expectations = {
        "table_i_declared_families": len(COMPATIBILITY_MATRIX_DECLARED_CASES_BY_FAMILY),
        "table_i_declared_cases": COMPATIBILITY_MATRIX_EXPECTED_CASE_COUNT,
        "paper_ii_source_cases": 20,
        "dynamics_classes": 3,
        "comparators": len(COMPATIBILITY_MATRIX_COMPARATOR_IDS),
        "comparator_rows": COMPATIBILITY_MATRIX_EXPECTED_COMPARATOR_ROWS,
        "records_total": COMPATIBILITY_MATRIX_EXPECTED_TOTAL_RECORDS,
    }
    for key, expected_value in manifest_expectations.items():
        actual_value = _matrix_int(expected_counts.get(key), errors=errors, label=f"manifest.expected_counts.{key}")
        if actual_value != expected_value:
            errors.append(f"manifest expected_counts.{key} expected {expected_value}, got {actual_value}")
    if len(comparator_rows) != COMPATIBILITY_MATRIX_EXPECTED_COMPARATOR_ROWS:
        errors.append(f"comparator row count expected {COMPATIBILITY_MATRIX_EXPECTED_COMPARATOR_ROWS}, got {len(comparator_rows)}")
    if len(records) != COMPATIBILITY_MATRIX_EXPECTED_TOTAL_RECORDS:
        errors.append(f"record count expected {COMPATIBILITY_MATRIX_EXPECTED_TOTAL_RECORDS}, got {len(records)}")
    if _matrix_int(actual_counts.get("record_count"), errors=errors, label="manifest.actual_counts.record_count") != len(records):
        errors.append("manifest actual record_count does not match records.tsv")
    if families != set(COMPATIBILITY_MATRIX_DECLARED_CASES_BY_FAMILY):
        errors.append("comparator row family set does not match the Paper-III n_ph=1 profile contract")
    if cases != expected_cases:
        missing_cases = sorted(expected_cases - cases)[:5]
        extra_cases = sorted(cases - expected_cases)[:5]
        errors.append(f"case coverage mismatch; missing={missing_cases}, extra={extra_cases}")
    if comparators != set(COMPATIBILITY_MATRIX_COMPARATOR_IDS):
        errors.append("comparator id set does not match the Paper-III n_ph=1 profile contract")
    if drive_amplitudes != set(COMPATIBILITY_MATRIX_DRIVE_AMPLITUDES):
        errors.append(f"drive amplitudes expected {COMPATIBILITY_MATRIX_DRIVE_AMPLITUDES}, got {sorted(drive_amplitudes)}")
    if actual_cross_product != expected_cross_product:
        missing_rows = sorted(expected_cross_product - actual_cross_product)[:5]
        extra_rows = sorted(actual_cross_product - expected_cross_product)[:5]
        errors.append(f"cross-product coverage mismatch; missing={missing_rows}, extra={extra_rows}")
    expected_smoke_ids = {
        row["record_id"]
        for row in comparator_rows
        if row.get("comparator_id") == "geo_selected_qse" and row.get("drive_amplitude") == "0.2"
    }
    if set(smoke_ids) != expected_smoke_ids or len(smoke_ids) != 18:
        errors.append("smoke_record_ids.txt must contain exactly the 18 geo_selected_qse A=0.2 audit rows")
    if len(scope_rows) != 1 or not any(row.get("family") == "molecular_restricted_closed_shell" for row in scope_rows):
        errors.append("exactly one registry-only molecular_restricted_closed_shell scope decision row is required")

    blocker_pairs_from_records: set[tuple[str, str]] = set()
    for row in records:
        record_id = _clean(row.get("record_id"))
        if _clean(row.get("mode")) != MODE_COMPATIBILITY_AUDIT_ONLY:
            errors.append(f"{record_id}: mode must be {MODE_COMPATIBILITY_AUDIT_ONLY}")
        if not _matrix_bool(row.get("no_submit"), default=False):
            errors.append(f"{record_id}: no_submit must be true")
        if _clean(row.get("profile")) != expected_profile:
            errors.append(f"{record_id}: profile mismatch")
        if not _clean(row.get("expected_status")):
            errors.append(f"{record_id}: expected_status is blank")
        codes = _split_codes(row.get("blocker_codes"))
        for code in codes:
            blocker_pairs_from_records.add((record_id, code))
        if _matrix_int(row.get("blocker_count"), errors=errors, label=f"{record_id}.blocker_count") != len(codes):
            errors.append(f"{record_id}: blocker_count does not match blocker_codes")
        if row.get("expected_status") == "blocked" and not codes:
            errors.append(f"{record_id}: blocked row has no blocker codes")
        if _clean(row.get("row_kind")) == "comparator_row":
            strict_expected = {
                "controller_exact_input_mode": "off",
                "diagnostic_exact_reference_mode": "benchmark_exact",
                "uses_reference_for_decision": "false",
                "uses_future_exact_forecast_for_decision": "false",
                "strict_measurement_oracle_certified": "true",
                "qpu_faithful_decisions_passed": "true",
                "strict_policy_status": "pass",
            }
            for key, expected in strict_expected.items():
                if _clean(row.get(key)).lower() != expected:
                    errors.append(f"{record_id}: {key} expected {expected!r}, got {_clean(row.get(key))!r}")
            if _clean(row.get("exact_decision_checkpoints")) != "0":
                errors.append(f"{record_id}: exact_decision_checkpoints must be 0")
            if _clean(row.get("target_excited_roots")) != "6":
                errors.append(f"{record_id}: target_excited_roots must be 6")
            if _clean(row.get("n_ph_max")) != str(n_ph_max):
                errors.append(f"{record_id}: n_ph_max must be {n_ph_max}")

    blocker_pairs_tsv = _pairs_from_blocker_rows(blocker_tsv_rows)
    blocker_pairs_json = _pairs_from_blocker_rows(blockers_json)
    if blocker_pairs_tsv != blocker_pairs_json:
        errors.append("blockers.tsv and blockers.json disagree on record/code pairs")
    if blocker_pairs_from_records != blocker_pairs_json:
        errors.append("record blocker_codes and blockers.json disagree on record/code pairs")
    if len(blocker_pairs_json) != _matrix_int(blockers_payload.get("blocker_count"), errors=errors, label="blockers.json.blocker_count"):
        errors.append("blockers.json blocker_count does not match blocker rows")
    if any(not _matrix_bool(row.get("no_submit"), default=False) for row in blocker_tsv_rows):
        errors.append("blockers.tsv must mark every blocker row no_submit=true")

    def _mapping_counter(value: Any, *, label: str) -> dict[str, int]:
        if not isinstance(value, Mapping):
            errors.append(f"{label} must be an object of counts")
            return {}
        out: dict[str, int] = {}
        for key, raw_count in value.items():
            try:
                out[str(key)] = int(raw_count)
            except Exception:
                errors.append(f"{label}.{key} must be an integer, got {raw_count!r}")
        return out

    seed_slots_raw = seed_inventory_payload.get("slots")
    if not isinstance(seed_slots_raw, list):
        errors.append("seed_repair_inventory.json must contain a slots list")
        seed_slots_raw = []
    seed_slots: list[Mapping[str, Any]] = []
    for idx, slot in enumerate(seed_slots_raw):
        if not isinstance(slot, Mapping):
            errors.append(f"seed_repair_inventory.json slots[{idx}] must be an object")
        else:
            seed_slots.append(slot)
    seed_scope_rows = seed_inventory_payload.get("scope_decision_rows")
    if not isinstance(seed_scope_rows, list):
        errors.append("seed_repair_inventory.json must contain a scope_decision_rows list")
        seed_scope_rows = []
    seed_scope_mappings = [row for row in seed_scope_rows if isinstance(row, Mapping)]

    if seed_inventory_payload.get("schema_version") != SEED_REPAIR_INVENTORY_SCHEMA_VERSION:
        errors.append(
            "seed_repair_inventory.json schema_version expected "
            f"{SEED_REPAIR_INVENTORY_SCHEMA_VERSION!r}, got {seed_inventory_payload.get('schema_version')!r}"
        )
    if seed_inventory_payload.get("profile") != expected_profile:
        errors.append("seed_repair_inventory.json profile mismatch")
    if seed_inventory_payload.get("matrix_batch_id") != manifest.get("matrix_batch_id"):
        errors.append("seed_repair_inventory.json matrix_batch_id must match manifest")
    if seed_inventory_payload.get("no_submit") is not True or seed_inventory_payload.get("local_only") is not True:
        errors.append("seed inventory must assert no_submit=true and local_only=true")
    for key in ("submits_chtc", "runs_realtime_dynamics", "repairs_seed_artifacts", "changes_seed_paths"):
        if seed_inventory_payload.get(key) is not False:
            errors.append(f"seed inventory must assert {key}=false")
    seed_summary_lower = seed_summary_text.lower()
    if "no submit" not in seed_summary_lower and "no-submit" not in seed_summary_lower:
        errors.append("seed repair summary must contain an explicit no-submit assertion")
    if "no seed repair" not in seed_summary_lower:
        errors.append("seed repair summary must state no seed repair was performed")

    if len(seed_slots) != SEED_REPAIR_EXPECTED_UNIQUE_SLOTS:
        errors.append(f"seed inventory slot count expected {SEED_REPAIR_EXPECTED_UNIQUE_SLOTS}, got {len(seed_slots)}")
    if len(seed_inventory_tsv) != SEED_REPAIR_EXPECTED_UNIQUE_SLOTS:
        errors.append(f"seed repair TSV row count expected {SEED_REPAIR_EXPECTED_UNIQUE_SLOTS}, got {len(seed_inventory_tsv)}")
    if _matrix_int(seed_inventory_payload.get("unique_seed_slot_count"), errors=errors, label="seed_inventory.unique_seed_slot_count") != SEED_REPAIR_EXPECTED_UNIQUE_SLOTS:
        errors.append("seed inventory unique_seed_slot_count must be 36")
    if _matrix_int(seed_inventory_payload.get("expected_unique_seed_slot_count"), errors=errors, label="seed_inventory.expected_unique_seed_slot_count") != SEED_REPAIR_EXPECTED_UNIQUE_SLOTS:
        errors.append("seed inventory expected_unique_seed_slot_count must be 36")
    if _matrix_int(manifest.get("seed_repair_unique_slot_count"), errors=errors, label="manifest.seed_repair_unique_slot_count") != SEED_REPAIR_EXPECTED_UNIQUE_SLOTS:
        errors.append("manifest seed_repair_unique_slot_count must be 36")
    if _matrix_int(manifest.get("seed_repair_expected_unique_slot_count"), errors=errors, label="manifest.seed_repair_expected_unique_slot_count") != SEED_REPAIR_EXPECTED_UNIQUE_SLOTS:
        errors.append("manifest seed_repair_expected_unique_slot_count must be 36")
    if _matrix_int(seed_inventory_payload.get("scope_decision_row_count"), errors=errors, label="seed_inventory.scope_decision_row_count") != 1:
        errors.append("seed inventory scope_decision_row_count must be 1")
    if _matrix_int(manifest.get("seed_repair_scope_decision_row_count"), errors=errors, label="manifest.seed_repair_scope_decision_row_count") != 1:
        errors.append("manifest seed_repair_scope_decision_row_count must be 1")
    if len(seed_scope_mappings) != 1 or not any(_clean(row.get("family")) == "molecular_restricted_closed_shell" for row in seed_scope_mappings):
        errors.append("seed inventory must preserve exactly one molecular_restricted_closed_shell scope row")
    if any(_clean(slot.get("family")) == "molecular_restricted_closed_shell" for slot in seed_slots):
        errors.append("registry-only molecular_restricted_closed_shell scope row must be excluded from seed slots")

    slot_ids = [_clean(slot.get("slot_id")) for slot in seed_slots]
    if len(slot_ids) != len(set(slot_ids)):
        errors.append("seed repair inventory JSON contains duplicate slot_id values")
    tsv_slot_ids = [_clean(row.get("slot_id")) for row in seed_inventory_tsv]
    if len(tsv_slot_ids) != len(set(tsv_slot_ids)):
        errors.append("seed repair inventory TSV contains duplicate slot_id values")
    if set(slot_ids) != set(tsv_slot_ids):
        errors.append("seed repair inventory TSV and JSON disagree on slot_id set")
    slots_by_id = {_clean(slot.get("slot_id")): slot for slot in seed_slots}
    tsv_by_id = {_clean(row.get("slot_id")): row for row in seed_inventory_tsv}
    for slot_id in sorted(set(slots_by_id) & set(tsv_by_id)):
        json_slot = slots_by_id[slot_id]
        tsv_slot = tsv_by_id[slot_id]
        for field in (
            "family",
            "paper_i_case_id",
            "paper_ii_case_id",
            "drive_amplitude",
            "row_count",
            "comparator_ids",
            "current_static_artifact_path",
            "current_static_artifact_status",
            "inherited_seed_blocker_codes",
            "repair_status",
        ):
            if _clean(json_slot.get(field)) != _clean(tsv_slot.get(field)):
                errors.append(f"seed inventory TSV/JSON mismatch for {slot_id}.{field}")

    expected_seed_keys = {
        (family, case_id, amplitude)
        for family, case_ids in COMPATIBILITY_MATRIX_DECLARED_CASES_BY_FAMILY.items()
        for case_id in case_ids
        for amplitude in COMPATIBILITY_MATRIX_DRIVE_AMPLITUDES
    }
    actual_seed_keys = {
        (_clean(slot.get("family")), _clean(slot.get("paper_i_case_id")), _clean(slot.get("drive_amplitude")))
        for slot in seed_slots
    }
    if actual_seed_keys != expected_seed_keys:
        missing_keys = sorted(expected_seed_keys - actual_seed_keys)[:5]
        extra_keys = sorted(actual_seed_keys - expected_seed_keys)[:5]
        errors.append(f"seed-slot coverage mismatch; missing={missing_keys}, extra={extra_keys}")
    if len(actual_seed_keys) != len(seed_slots):
        errors.append("seed repair inventory contains duplicate (family, paper_i_case_id, drive_amplitude) keys")

    expected_seed_blockers_by_key: dict[tuple[str, str, str], set[str]] = {}
    expected_row_counts_by_key: Counter[tuple[str, str, str]] = Counter()
    expected_comparators_by_key: dict[tuple[str, str, str], set[str]] = {}
    for row in comparator_rows:
        key = (_clean(row.get("family")), _clean(row.get("paper_i_case_id")), _clean(row.get("drive_amplitude")))
        expected_row_counts_by_key[key] += 1
        expected_comparators_by_key.setdefault(key, set()).add(_clean(row.get("comparator_id")))
        expected_seed_blockers_by_key.setdefault(key, set()).update(
            code for code in _split_codes(row.get("blocker_codes")) if code.startswith("seed.")
        )
    seed_blocker_counter: Counter[str] = Counter()
    repair_status_counter: Counter[str] = Counter()
    for slot in seed_slots:
        key = (_clean(slot.get("family")), _clean(slot.get("paper_i_case_id")), _clean(slot.get("drive_amplitude")))
        slot_id = _clean(slot.get("slot_id"))
        if _matrix_int(slot.get("row_count"), errors=errors, label=f"{slot_id}.row_count") != 8:
            errors.append(f"{slot_id}: row_count must be 8")
        if expected_row_counts_by_key.get(key) != 8:
            errors.append(f"{slot_id}: comparator matrix has {expected_row_counts_by_key.get(key, 0)} rows for this seed slot")
        comparator_ids = set(_split_codes(slot.get("comparator_ids")))
        if comparator_ids != set(COMPATIBILITY_MATRIX_COMPARATOR_IDS) or comparator_ids != expected_comparators_by_key.get(key, set()):
            errors.append(f"{slot_id}: comparator_ids do not match the 8 comparator matrix rows")
        status = _clean(slot.get("repair_status"))
        repair_status_counter[status] += 1
        if status not in SEED_REPAIR_ALLOWED_STATUSES:
            errors.append(f"{slot_id}: repair_status {status!r} is not allowed")
        inherited_codes = set(_split_codes(slot.get("inherited_seed_blocker_codes")))
        if any(not code.startswith("seed.") for code in inherited_codes):
            errors.append(f"{slot_id}: inherited_seed_blocker_codes must contain only seed.* blockers")
        expected_codes = expected_seed_blockers_by_key.get(key, set())
        if inherited_codes != expected_codes:
            errors.append(f"{slot_id}: inherited seed blockers mismatch; expected={sorted(expected_codes)}, got={sorted(inherited_codes)}")
        seed_blocker_counter.update(inherited_codes)

    repair_status_counts = dict(sorted(repair_status_counter.items()))
    if repair_status_counts != _mapping_counter(seed_inventory_payload.get("repair_status_counts"), label="seed_inventory.repair_status_counts"):
        errors.append("seed inventory repair_status_counts do not match slots")
    if repair_status_counts != _mapping_counter(manifest.get("seed_repair_status_counts"), label="manifest.seed_repair_status_counts"):
        errors.append("manifest seed_repair_status_counts do not match slots")
    seed_blocker_counts = dict(sorted(seed_blocker_counter.items()))
    if seed_blocker_counts != _mapping_counter(seed_inventory_payload.get("seed_blocker_code_counts"), label="seed_inventory.seed_blocker_code_counts"):
        errors.append("seed inventory seed_blocker_code_counts do not match slots")
    if seed_blocker_counts != _mapping_counter(manifest.get("seed_repair_seed_blocker_code_counts"), label="manifest.seed_repair_seed_blocker_code_counts"):
        errors.append("manifest seed_repair_seed_blocker_code_counts do not match slots")

    report = {
        "schema_version": COMPATIBILITY_MATRIX_VALIDATION_SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "compatibility_matrix_dir": str(root),
        "profile": manifest.get("profile"),
        "matrix_batch_id": manifest.get("matrix_batch_id"),
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "record_count": len(records),
        "comparator_row_count": len(comparator_rows),
        "scope_decision_row_count": len(scope_rows),
        "blocker_count": len(blocker_pairs_json),
        "family_count": len(families),
        "case_count": len(cases),
        "comparator_count": len(comparators),
        "full_record_count": len(full_ids),
        "smoke_record_count": len(smoke_ids),
        "seed_repair_unique_slot_count": len(seed_slots),
        "seed_repair_expected_unique_slot_count": SEED_REPAIR_EXPECTED_UNIQUE_SLOTS,
        "seed_repair_scope_decision_row_count": len(seed_scope_mappings),
        "seed_repair_status_counts": repair_status_counts,
        "no_submit": bool(manifest.get("no_submit")),
    }
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate local/fetched Paper III excited-dynamics CHTC outputs.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--record-list", type=Path, default=None)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument(
        "--compatibility-matrix-dir",
        type=Path,
        default=None,
        help="Validate a local-only Paper III compatibility matrix artifact directory instead of fetched CHTC outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.compatibility_matrix_dir is not None:
        report = validate_compatibility_matrix_dir(args.compatibility_matrix_dir)
    else:
        report = validate_outputs(args.root, record_ids=args.record_id, record_list=args.record_list, write_report=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


__all__ = [
    "COMPATIBILITY_MATRIX_PROFILE",
    "DEFAULT_ROOT",
    "resolve_fetched_path",
    "validate_compatibility_matrix_dir",
    "validate_outputs",
    "validate_record_dir",
]


if __name__ == "__main__":
    raise SystemExit(main())
