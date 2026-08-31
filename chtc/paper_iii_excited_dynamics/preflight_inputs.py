#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.excited_dynamics.paper_iii_local_science_pilot import (  # noqa: E402
    validate_strict_hh_payload,
)

DEFAULT_RECORDS = Path(__file__).resolve().parent / "input" / "records.tsv"
REPORT_NAME = "preflight_report.json"
PACKAGE_INPUT_REL = Path("chtc") / "paper_iii_excited_dynamics" / "input"
SEED_ARTIFACTS_REL = PACKAGE_INPUT_REL / "seed_artifacts"
EXISTING_STRICT_OUTPUTS_REL = PACKAGE_INPUT_REL / "existing_strict_outputs"
SCHEMA_VERSION = "paper_iii_excited_dynamics_preflight_report_v1"
MODE_REPORT_ONLY_EXISTING_OUTPUT = "report_only_existing_output"
MODE_STRICT_HH = "strict_hh"
MODE_COMPATIBILITY_AUDIT_ONLY = "compatibility_audit_only"
SUPPORTED_MODES = {MODE_REPORT_ONLY_EXISTING_OUTPUT, MODE_STRICT_HH, MODE_COMPATIBILITY_AUDIT_ONLY}
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def parse_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"cannot parse boolean value {value!r}")


def repo_path(path: str | Path, *, repo_root: Path) -> Path:
    raw = Path(str(path)).expanduser()
    return raw if raw.is_absolute() else repo_root / raw


def _path_is_under_repo(path: Path, *, repo_root: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root.resolve())
        return True
    except ValueError:
        return False


def load_records(path: str | Path = DEFAULT_RECORDS) -> list[dict[str, str]]:
    records_path = Path(path)
    with records_path.open(newline="", encoding="utf-8") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter="\t")]


def load_record_ids(path: str | Path) -> list[str]:
    return [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def find_record(rows: Sequence[Mapping[str, str]], record_id: str) -> dict[str, str]:
    matches = [dict(row) for row in rows if _clean(row.get("record_id")) == str(record_id)]
    if not matches:
        raise KeyError(f"record_id {record_id!r} not found")
    if len(matches) > 1:
        raise ValueError(f"record_id {record_id!r} appears {len(matches)} times")
    return matches[0]


def selected_records(
    rows: Sequence[Mapping[str, str]],
    *,
    record_ids: Sequence[str] | None = None,
    record_list: str | Path | None = None,
    all_records: bool = False,
) -> list[dict[str, str]]:
    if record_ids:
        ids = [str(item) for item in record_ids]
    elif record_list is not None:
        ids = load_record_ids(record_list)
    elif all_records:
        return [dict(row) for row in rows]
    else:
        return [dict(row) for row in rows]
    return [find_record(rows, record_id) for record_id in ids]


def _positive_float(row: Mapping[str, str], key: str, errors: list[str]) -> float | None:
    raw = _clean(row.get(key))
    if not raw:
        errors.append(f"missing {key}")
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{key} must be a positive finite float, got {raw!r}")
        return None
    if not value > 0.0 or value in {float("inf"), float("-inf")}:
        errors.append(f"{key} must be a positive finite float, got {raw!r}")
        return None
    return value


def _positive_int(row: Mapping[str, str], key: str, errors: list[str], *, minimum: int = 1) -> int | None:
    raw = _clean(row.get(key))
    if not raw:
        errors.append(f"missing {key}")
        return None
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{key} must be an integer >= {minimum}, got {raw!r}")
        return None
    if value < minimum:
        errors.append(f"{key} must be an integer >= {minimum}, got {raw!r}")
        return None
    return value


def _expected_seed_rel(record_id: str) -> Path:
    return SEED_ARTIFACTS_REL / f"{record_id}.json"


def _expected_existing_rel(record_id: str) -> Path:
    return EXISTING_STRICT_OUTPUTS_REL / f"{record_id}.json"


def _relative_to_repo(path: Path, *, repo_root: Path) -> Path | None:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None


def _validate_canonical_json_path(
    *,
    raw_value: str,
    expected_rel: Path,
    repo_root: Path,
    field: str,
    errors: list[str],
) -> Path | None:
    if not raw_value:
        errors.append(f"missing {field}")
        return None
    path = repo_path(raw_value, repo_root=repo_root)
    actual_rel = _relative_to_repo(path, repo_root=repo_root)
    if actual_rel != expected_rel:
        errors.append(f"{field} must equal {expected_rel.as_posix()!r}")
    return path


def _copy_if_requested(source: Path | None, target: Path | None, *, stage: bool, errors: list[str]) -> None:
    if source is None or target is None or errors or not stage:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def _json_size(path: Path | None) -> int | None:
    if path is None or not path.exists():
        return None
    try:
        return int(path.stat().st_size)
    except OSError:
        return None


def _split_blocker_codes(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]


def _expected_n_ph_from_profile(profile: str) -> int | None:
    text = _clean(profile)
    marker = "paper_iii_nph"
    if not text.startswith(marker):
        return None
    suffix = text[len(marker) :]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return None
    return int("".join(digits))


def _nonnegative_int(row: Mapping[str, str], key: str, errors: list[str]) -> int | None:
    raw = _clean(row.get(key))
    if not raw:
        errors.append(f"missing {key}")
        return None
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{key} must be a nonnegative integer, got {raw!r}")
        return None
    if value < 0:
        errors.append(f"{key} must be a nonnegative integer, got {raw!r}")
        return None
    return value


def _require_clean(row: Mapping[str, str], key: str, errors: list[str]) -> str:
    value = _clean(row.get(key))
    if not value:
        errors.append(f"missing {key}")
    return value


def _validate_exact_free_audit_fields(row: Mapping[str, str], errors: list[str]) -> None:
    expected = {
        "controller_exact_input_mode": "off",
        "diagnostic_exact_reference_mode": "benchmark_exact",
        "uses_reference_for_decision": "false",
        "uses_future_exact_forecast_for_decision": "false",
        "strict_measurement_oracle_certified": "true",
        "qpu_faithful_decisions_passed": "true",
    }
    for key, expected_value in expected.items():
        raw = _clean(row.get(key)).lower()
        if raw != expected_value:
            errors.append(f"{key} expected {expected_value!r}, got {_clean(row.get(key))!r}")
    if _clean(row.get("exact_decision_checkpoints")) != "0":
        errors.append("exact_decision_checkpoints must be 0")


def _validate_compatibility_audit_row(row: Mapping[str, str], *, repo_root: Path) -> dict[str, Any]:
    record_id = _clean(row.get("record_id"))
    errors: list[str] = []
    warnings: list[str] = []
    if not record_id:
        errors.append("missing record_id")
    elif any(char in record_id for char in "/\\"):
        errors.append("record_id must not contain path separators")
    if _clean(row.get("mode")) != MODE_COMPATIBILITY_AUDIT_ONLY:
        errors.append(f"mode must equal {MODE_COMPATIBILITY_AUDIT_ONLY!r}")

    for key in (
        "matrix_batch_id",
        "profile",
        "row_kind",
        "family",
        "case_id",
        "comparator_id",
        "expected_status",
        "blocker_count",
        "strict_policy_status",
        "no_submit",
    ):
        _require_clean(row, key, errors)
    try:
        if not parse_bool(row.get("no_submit"), default=False):
            errors.append("compatibility_audit_only records must set no_submit=true")
    except ValueError as exc:
        errors.append(str(exc))
    for key in ("require_progress_json", "require_partial_payload_json"):
        try:
            if parse_bool(row.get(key), default=False):
                errors.append(f"compatibility_audit_only records must set {key}=false")
        except ValueError as exc:
            errors.append(str(exc))
    try:
        require_science = parse_bool(row.get("require_science_benchmark"), default=False)
    except ValueError as exc:
        errors.append(str(exc))
        require_science = False
    if require_science:
        warnings.append("audit-only row requests science benchmark output, but no dynamics will be launched")

    blocker_count = _nonnegative_int(row, "blocker_count", errors)
    blocker_codes = _split_blocker_codes(row.get("blocker_codes"))
    if blocker_count is not None and blocker_count != len(blocker_codes):
        errors.append(f"blocker_count {blocker_count} != parsed blocker code count {len(blocker_codes)}")
    expected_status = _clean(row.get("expected_status"))
    if expected_status not in {"ready", "blocked", "not_applicable"}:
        errors.append(f"expected_status must be ready/blocked/not_applicable, got {expected_status!r}")
    if expected_status == "blocked" and blocker_count == 0:
        errors.append("blocked audit rows must list at least one blocker code")
    if expected_status == "blocked" and not blocker_codes:
        errors.append("blocked audit rows must provide blocker_codes")
    if expected_status == "ready" and blocker_count not in {0, None}:
        errors.append("ready audit rows must not list blocker codes")
    if expected_status == "ready" and blocker_codes:
        errors.append("ready audit rows must leave blocker_codes empty")

    row_kind = _clean(row.get("row_kind"))
    if row_kind == "comparator_row":
        _positive_float(row, "t_final", errors)
        _positive_int(row, "num_times", errors, minimum=2)
        _positive_int(row, "target_excited_roots", errors, minimum=1)
        n_ph = _positive_int(row, "n_ph_max", errors, minimum=1)
        expected_n_ph = _expected_n_ph_from_profile(row.get("profile", "")) or 1
        if n_ph is not None and n_ph != int(expected_n_ph):
            errors.append(f"compatibility matrix records must have n_ph_max={int(expected_n_ph)}")
        if _clean(row.get("target_excited_roots")) != "6":
            errors.append("compatibility matrix records must target 6 excited roots")
        _positive_float(row, "drive_amplitude", errors)
        if _clean(row.get("strict_policy_status")) != "pass":
            errors.append("comparator audit rows must carry strict_policy_status=pass")
        _validate_exact_free_audit_fields(row, errors)
    elif row_kind == "scope_decision":
        _validate_exact_free_audit_fields(row, errors)
    else:
        errors.append(f"unsupported row_kind {row_kind!r}")

    try:
        t_final_value = None if not _clean(row.get("t_final")) else float(_clean(row.get("t_final")))
    except ValueError:
        t_final_value = None
    try:
        num_times_value = None if not _clean(row.get("num_times")) else int(_clean(row.get("num_times")))
    except ValueError:
        num_times_value = None

    return {
        "record_id": record_id,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "mode": MODE_COMPATIBILITY_AUDIT_ONLY,
        "queue": _clean(row.get("queue")),
        "source_artifact_json": _clean(row.get("source_artifact_json")),
        "source_artifact_path": None,
        "artifact_json": _clean(row.get("artifact_json")),
        "staged_artifact_path": None,
        "staged_artifact_exists": False,
        "staged_artifact_size_bytes": None,
        "source_existing_strict_output_json": None,
        "source_existing_strict_output_path": None,
        "existing_strict_output_json": None,
        "staged_existing_strict_output_path": None,
        "staged_existing_strict_output_exists": False,
        "staged_existing_strict_output_size_bytes": None,
        "strict_existing_output_validation_passed": None,
        "strict_existing_output_validation": None,
        "t_final": t_final_value,
        "num_times": num_times_value,
        "timeout_seconds": None,
        "require_progress_json": False,
        "require_partial_payload_json": False,
        "require_science_benchmark": require_science,
        "matrix_batch_id": _clean(row.get("matrix_batch_id")),
        "profile": _clean(row.get("profile")),
        "row_kind": row_kind,
        "family": _clean(row.get("family")),
        "case_id": _clean(row.get("case_id")),
        "comparator_id": _clean(row.get("comparator_id")),
        "expected_status": expected_status,
        "blocker_count": blocker_count,
        "blocker_codes": blocker_codes,
        "no_submit": True,
    }


def _validate_existing_strict_payload(
    path: Path,
    *,
    expected_t_final: float,
    expected_num_times: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        return {"passed": False, "violations": ["existing strict output must be a JSON object"], "violation_count": 1}
    return validate_strict_hh_payload(
        payload,
        expected_t_final=float(expected_t_final),
        expected_num_times=int(expected_num_times),
    )


def validate_record_row(
    row: Mapping[str, str],
    *,
    repo_root: Path,
    stage: bool = True,
    validate_existing_strict_output: bool = True,
) -> dict[str, Any]:
    record_id = _clean(row.get("record_id"))
    mode = _clean(row.get("mode"))
    errors: list[str] = []
    warnings: list[str] = []
    if not record_id:
        errors.append("missing record_id")
    elif any(char in record_id for char in "/\\"):
        errors.append("record_id must not contain path separators")
    if mode not in SUPPORTED_MODES:
        errors.append(f"unsupported mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
    if mode == MODE_COMPATIBILITY_AUDIT_ONLY:
        return _validate_compatibility_audit_row(row, repo_root=repo_root)

    t_final = _positive_float(row, "t_final", errors)
    num_times = _positive_int(row, "num_times", errors, minimum=2)
    timeout_seconds = _positive_int(row, "timeout_seconds", errors, minimum=1)

    require_progress = False
    require_partial = False
    require_science = True
    for key, target in (
        ("require_progress_json", "progress"),
        ("require_partial_payload_json", "partial payload"),
        ("require_science_benchmark", "science benchmark"),
    ):
        try:
            parsed = parse_bool(row.get(key), default=(key == "require_science_benchmark"))
        except ValueError as exc:
            errors.append(str(exc))
            parsed = False
        if key == "require_progress_json":
            require_progress = parsed
        elif key == "require_partial_payload_json":
            require_partial = parsed
        else:
            require_science = parsed
        if target == "science benchmark" and not parsed:
            warnings.append("record does not require paper_iii_science_benchmark")

    if mode == MODE_STRICT_HH:
        if not require_progress:
            errors.append("strict_hh records must set require_progress_json=true")
        if not require_partial:
            errors.append("strict_hh records must set require_partial_payload_json=true")
    if mode == MODE_REPORT_ONLY_EXISTING_OUTPUT:
        if require_progress or require_partial:
            errors.append("report_only_existing_output records must not require progress/partial payload sidecars")

    source_raw = _clean(row.get("source_artifact_json"))
    artifact_raw = _clean(row.get("artifact_json"))
    source_path = repo_path(source_raw, repo_root=repo_root) if source_raw else None
    if source_path is None:
        errors.append("missing source_artifact_json")
    elif not _path_is_under_repo(source_path, repo_root=repo_root):
        errors.append(f"source_artifact_json must resolve under repo root: {source_raw}")
    elif not source_path.exists():
        errors.append(f"source_artifact_json does not exist: {source_raw}")
    artifact_path = _validate_canonical_json_path(
        raw_value=artifact_raw,
        expected_rel=_expected_seed_rel(record_id) if record_id else SEED_ARTIFACTS_REL / "<missing>.json",
        repo_root=repo_root,
        field="artifact_json",
        errors=errors,
    )

    source_existing_raw = _clean(row.get("source_existing_strict_output_json"))
    existing_raw = _clean(row.get("existing_strict_output_json"))
    source_existing_path: Path | None = None
    existing_path: Path | None = None
    existing_validation: dict[str, Any] | None = None
    if mode == MODE_REPORT_ONLY_EXISTING_OUTPUT:
        source_existing_path = repo_path(source_existing_raw, repo_root=repo_root) if source_existing_raw else None
        if source_existing_path is None:
            errors.append("report_only_existing_output records require source_existing_strict_output_json")
        elif not _path_is_under_repo(source_existing_path, repo_root=repo_root):
            errors.append(f"source_existing_strict_output_json must resolve under repo root: {source_existing_raw}")
        elif not source_existing_path.exists():
            errors.append(f"source_existing_strict_output_json does not exist: {source_existing_raw}")
        existing_path = _validate_canonical_json_path(
            raw_value=existing_raw,
            expected_rel=_expected_existing_rel(record_id) if record_id else EXISTING_STRICT_OUTPUTS_REL / "<missing>.json",
            repo_root=repo_root,
            field="existing_strict_output_json",
            errors=errors,
        )
    elif mode == MODE_STRICT_HH:
        if source_existing_raw:
            errors.append("strict_hh records must leave source_existing_strict_output_json blank")
        if existing_raw:
            errors.append("strict_hh records must leave existing_strict_output_json blank")

    _copy_if_requested(source_path, artifact_path, stage=stage, errors=errors)
    _copy_if_requested(source_existing_path, existing_path, stage=stage, errors=errors)

    if artifact_path is not None and artifact_path.exists() and _json_size(artifact_path) == 0:
        errors.append(f"staged artifact is empty: {artifact_path}")
    existing_validation_path = existing_path if existing_path is not None and existing_path.exists() else source_existing_path
    if (
        mode == MODE_REPORT_ONLY_EXISTING_OUTPUT
        and validate_existing_strict_output
        and existing_validation_path is not None
        and not errors
    ):
        try:
            existing_validation = _validate_existing_strict_payload(
                existing_validation_path,
                expected_t_final=float(t_final if t_final is not None else 0.0),
                expected_num_times=int(num_times if num_times is not None else 0),
            )
            if not existing_validation.get("passed"):
                errors.append(
                    "existing strict output validation failed: "
                    + "; ".join(str(item) for item in existing_validation.get("violations", []))
                )
        except Exception as exc:
            errors.append(f"existing strict output validation failed: {type(exc).__name__}: {exc}")

    return {
        "record_id": record_id,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "mode": mode,
        "queue": _clean(row.get("queue")),
        "source_artifact_json": source_raw,
        "source_artifact_path": None if source_path is None else str(source_path),
        "artifact_json": artifact_raw,
        "staged_artifact_path": None if artifact_path is None else str(artifact_path),
        "staged_artifact_exists": bool(artifact_path is not None and artifact_path.exists()),
        "staged_artifact_size_bytes": _json_size(artifact_path),
        "source_existing_strict_output_json": source_existing_raw or None,
        "source_existing_strict_output_path": None if source_existing_path is None else str(source_existing_path),
        "existing_strict_output_json": existing_raw or None,
        "staged_existing_strict_output_path": None if existing_path is None else str(existing_path),
        "staged_existing_strict_output_exists": bool(existing_path is not None and existing_path.exists()),
        "staged_existing_strict_output_size_bytes": _json_size(existing_path),
        "strict_existing_output_validation_passed": None
        if existing_validation is None
        else bool(existing_validation.get("passed")),
        "strict_existing_output_validation": existing_validation,
        "t_final": t_final,
        "num_times": num_times,
        "timeout_seconds": timeout_seconds,
        "require_progress_json": require_progress,
        "require_partial_payload_json": require_partial,
        "require_science_benchmark": require_science,
    }


def cleanup_stale_staged_artifacts(
    selected: Sequence[Mapping[str, str]],
    *,
    repo_root: Path,
) -> list[str]:
    keep_by_dir: dict[Path, set[str]] = {
        repo_root / SEED_ARTIFACTS_REL: set(),
        repo_root / EXISTING_STRICT_OUTPUTS_REL: set(),
    }
    for row in selected:
        record_id = _clean(row.get("record_id"))
        if not record_id:
            continue
        if _clean(row.get("artifact_json")):
            keep_by_dir[repo_root / SEED_ARTIFACTS_REL].add(f"{record_id}.json")
        if _clean(row.get("existing_strict_output_json")):
            keep_by_dir[repo_root / EXISTING_STRICT_OUTPUTS_REL].add(f"{record_id}.json")
    removed: list[str] = []
    for directory, keep in keep_by_dir.items():
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name in keep:
                continue
            path.unlink()
            rel = _relative_to_repo(path, repo_root=repo_root)
            removed.append(str(rel if rel is not None else path))
    return removed


def preflight_records(
    *,
    records_path: str | Path = DEFAULT_RECORDS,
    record_list: str | Path | None = None,
    record_ids: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
    stage: bool = True,
    all_records: bool = False,
    clean_staged: bool = True,
    validate_existing_strict_output: bool = True,
    write_report: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    records_file = Path(records_path)
    rows = load_records(records_file)
    selected = selected_records(rows, record_ids=record_ids, record_list=record_list, all_records=all_records)
    selected_modes = {_clean(row.get("mode")) for row in selected}
    audit_only_selection = bool(selected) and selected_modes <= {MODE_COMPATIBILITY_AUDIT_ONLY}
    removed_stale_artifacts = (
        cleanup_stale_staged_artifacts(selected, repo_root=repo)
        if stage and clean_staged and not audit_only_selection
        else []
    )
    record_results = [
        validate_record_row(
            row,
            repo_root=repo,
            stage=stage,
            validate_existing_strict_output=validate_existing_strict_output,
        )
        for row in selected
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_utc(),
        "records_path": str(records_file),
        "record_list": None if record_list is None else str(record_list),
        "record_count": len(record_results),
        "ok": bool(record_results) and all(item.get("ok") for item in record_results),
        "stage": bool(stage),
        "clean_staged": bool(stage and clean_staged),
        "removed_stale_artifacts": removed_stale_artifacts,
        "records": record_results,
        "failed_records": [item["record_id"] for item in record_results if not item.get("ok")],
    }
    if not record_results:
        report["ok"] = False
        report["errors"] = ["no records selected"]
    if write_report:
        report_path = records_file.parent / REPORT_NAME
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and stage Paper III excited-dynamics CHTC input records.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--record-list", type=Path, default=None)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--all", action="store_true", help="Validate every row in records.tsv.")
    parser.add_argument("--no-stage", action="store_true", help="Validate without copying staged JSON artifacts.")
    parser.add_argument(
        "--keep-stale-staged",
        action="store_true",
        help="Do not prune staged JSON artifacts outside the selected queue.",
    )
    parser.add_argument(
        "--no-validate-existing-strict-output",
        action="store_true",
        help="Skip strict HH payload validation for report-only existing-output records.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = preflight_records(
        records_path=args.records,
        record_list=args.record_list,
        record_ids=args.record_id,
        repo_root=args.repo_root,
        stage=not bool(args.no_stage),
        all_records=bool(args.all),
        clean_staged=not bool(args.keep_stale_staged),
        validate_existing_strict_output=not bool(args.no_validate_existing_strict_output),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


__all__ = [
    "DEFAULT_RECORDS",
    "EXISTING_STRICT_OUTPUTS_REL",
    "MODE_COMPATIBILITY_AUDIT_ONLY",
    "MODE_REPORT_ONLY_EXISTING_OUTPUT",
    "MODE_STRICT_HH",
    "SEED_ARTIFACTS_REL",
    "find_record",
    "load_record_ids",
    "load_records",
    "preflight_records",
    "repo_path",
    "selected_records",
    "validate_record_row",
]


if __name__ == "__main__":
    raise SystemExit(main())
