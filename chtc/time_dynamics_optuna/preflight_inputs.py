#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone

from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from pipelines.time_dynamics.tables.dynamics_benchmark_contract import (  # noqa: E402
    DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND,
    validate_dynamics_tuning_class,
)
from pipelines.time_dynamics.tables.table_lock_contract import (  # noqa: E402
    validate_class_settings_lock_manifest,
)
DEFAULT_RECORDS = Path(__file__).resolve().parent / "input" / "records.tsv"
REPORT_NAME = "preflight_report.json"
SEED_ARTIFACTS_REL = Path("chtc") / "time_dynamics_optuna" / "input" / "seed_artifacts"
_BOSON_DRIVEN_FAMILIES = {"spin_boson", "bose_hubbard", "harmonic_kerr_chain"}
_EXPECTED_VALIDATION_PROFILE = {
    "generic_l2_exact_v1": "generic_exact_v1",
    "append_prune_noharm_l2_v1": "generic_exact_v1",
    "append_prune_recoverability_l2_v1": "generic_exact_v1",
    "append_live_guard_l2_v1": "generic_exact_v1",
    "strict_qpu_faithful_recoverability_v1": "strict_qpu_faithful",
    "strict_qpu_faithful_append_prune_recoverability_v1": "strict_qpu_faithful",
    "strict_qpu_faithful_append_prune_guardrail_only_v1": "strict_qpu_faithful",
    "strict_qpu_faithful_append_prune_aggressive_v1": "strict_qpu_faithful",
    "strict_qpu_hh_recoverability_v1": "strict_qpu_hh",
}
_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n", ""}
_DRIVE_TIME_SAMPLING = {"", "left", "midpoint", "right"}
_ALL_ALGORITHM_CALIBRATION_STEM = "paper_ii_all_algorithm_class_calibration_v1"
_STRICT_APPEND_PRUNE_PROFILES = {
    "strict_qpu_faithful_append_prune_recoverability_v1",
    "strict_qpu_faithful_append_prune_guardrail_only_v1",
    "strict_qpu_faithful_append_prune_aggressive_v1",
}
_STRICT_EXACT_ONLY_OPTION_KEYS = {
    "invalid_max_mean_energy_mae",
    "invalid_max_final_energy_mae",
    "invalid_min_fidelity",
    "invalid_max_mean_total_occupation_mae",
    "invalid_max_primary_observable_mae_over_span",
    "objective_weight_site_mae",
    "objective_weight_energy_mae",
    "objective_weight_energy_bias_abs",
    "objective_weight_energy_max_abs",
    "objective_weight_energy_under_response_mean",
    "objective_weight_energy_under_response_max",
    "objective_weight_fidelity_defect",
    "objective_weight_pair_mae_over_span",
    "objective_weight_epsilon_osc",
    "objective_weight_peak_omega",
    "objective_weight_pair_corr_defect",
}


def _nonzero_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        return float(text) != 0.0
    except ValueError:
        return True


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _validate_optional_positive_float(row: Mapping[str, str], key: str, errors: list[str]) -> None:
    raw = str(row.get(key, "")).strip()
    if not raw:
        return
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{key} must be blank or a positive finite float, got {raw!r}")
        return
    if not math.isfinite(value) or value <= 0.0:
        errors.append(f"{key} must be blank or a positive finite float, got {raw!r}")


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
    matches = [dict(row) for row in rows if str(row.get("record_id", "")) == str(record_id)]
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
    ids: list[str]
    if record_ids:
        ids = [str(item) for item in record_ids]
    elif record_list is not None:
        ids = load_record_ids(record_list)
    elif all_records:
        return [dict(row) for row in rows]
    else:
        return [dict(row) for row in rows]
    return [find_record(rows, record_id) for record_id in ids]


def validate_profile(row: Mapping[str, str]) -> str | None:
    study_profile = str(row.get("study_profile", "")).strip()
    validation_profile = str(row.get("validation_profile", "")).strip()
    expected = _EXPECTED_VALIDATION_PROFILE.get(study_profile)
    if expected is None:
        return f"unsupported study_profile {study_profile!r}"
    if validation_profile != expected:
        return (
            f"validation_profile {validation_profile!r} does not match "
            f"study_profile {study_profile!r} expected {expected!r}"
        )
    return None


def _is_generic_benchmark_row(row: Mapping[str, str]) -> bool:
    return str(row.get("kind", "")).strip().lower() == "benchmark" and bool(str(row.get("algorithm_id", "")).strip())


def _resolve_row_path(value: Any, *, repo_root: Path, base_dir: Path | None = None) -> Path:
    raw = Path(str(value or "").strip()).expanduser()
    if raw.is_absolute():
        return raw
    if base_dir is not None:
        candidate = base_dir / raw
        if candidate.exists():
            return candidate
    return repo_root / raw


def _load_case_from_manifest(
    *,
    case_manifest: Path,
    case_id: str,
) -> Mapping[str, Any] | None:
    payload = json.loads(case_manifest.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases", []) if isinstance(payload, Mapping) else []
    if not isinstance(raw_cases, list):
        raise ValueError("case manifest must contain a cases list")
    for item in raw_cases:
        if isinstance(item, Mapping) and str(item.get("case_id", "")) == str(case_id):
            return item
    return None


def validate_generic_benchmark_record_row(
    row: Mapping[str, str],
    *,
    repo_root: Path,
    records_dir: Path | None = None,
) -> dict[str, Any]:
    record_id = str(row.get("record_id", "")).strip()
    errors: list[str] = []
    warnings: list[str] = []
    family = str(row.get("family", "")).strip()
    tuning_class = str(row.get("tuning_class", "")).strip()
    case_id = str(row.get("case_id", "")).strip()
    algorithm_id = str(row.get("algorithm_id", "")).strip()
    settings_kind = str(row.get("settings_kind", "")).strip()
    case_manifest_raw = str(row.get("case_manifest", "")).strip()
    class_settings_raw = str(row.get("class_settings_manifest", "")).strip()
    case_manifest_path: Path | None = None
    class_settings_path: Path | None = None
    runtime_artifact_path: Path | None = None
    if not record_id:
        errors.append("missing record_id")
    if not family or not case_id:
        errors.append("family and case_id are required for generic benchmark rows")
    expected_kind = DYNAMICS_TABLE_I_ALGORITHM_SETTINGS_KIND.get(algorithm_id)
    if expected_kind is None:
        errors.append(f"unsupported Table-I algorithm_id {algorithm_id!r}")
    elif settings_kind != expected_kind:
        errors.append(
            f"settings_kind {settings_kind!r} does not match algorithm_id {algorithm_id!r} expected {expected_kind!r}"
        )
    if tuning_class:
        try:
            validate_dynamics_tuning_class(family=family, tuning_class=tuning_class)
        except ValueError as exc:
            errors.append(str(exc))
    for key in ("candidate_only_not_promoted", "require_algorithm_class_settings", "require_parity_correctness_sidecars"):
        try:
            if parse_bool(row.get(key), default=False) is not True:
                errors.append(f"{key} must be true")
        except ValueError as exc:
            errors.append(str(exc))
    if str(row.get("seed_track", "")).strip() != "snake":
        errors.append("seed_track must be 'snake'")
    if not str(row.get("same_seed_comparator_group_id", "")).strip():
        errors.append("missing same_seed_comparator_group_id")
    if not str(row.get("seed_artifact_sha256", "")).strip():
        errors.append("missing seed_artifact_sha256")
    if not case_manifest_raw:
        errors.append("missing case_manifest")
    else:
        case_manifest_path = _resolve_row_path(case_manifest_raw, repo_root=repo_root, base_dir=records_dir)
        if not case_manifest_path.exists():
            errors.append(f"case_manifest does not exist: {case_manifest_raw}")
    if not class_settings_raw:
        errors.append("missing class_settings_manifest")
    else:
        class_settings_path = _resolve_row_path(class_settings_raw, repo_root=repo_root, base_dir=records_dir)
        if not class_settings_path.exists():
            errors.append(f"class_settings_manifest does not exist: {class_settings_raw}")
        else:
            try:
                validation = validate_class_settings_lock_manifest(
                    json.loads(class_settings_path.read_text(encoding="utf-8")),
                    require_all_table_i_algorithm_classes=True,
                )
                if _ALL_ALGORITHM_CALIBRATION_STEM in record_id:
                    payload = json.loads(class_settings_path.read_text(encoding="utf-8"))
                    if payload.get("lock_status") != "candidate_not_promoted":
                        errors.append("all-algorithm calibration smoke must use candidate_not_promoted settings")
                    if payload.get("candidate_only_not_promoted") is not True:
                        errors.append("all-algorithm calibration smoke settings must be candidate-only/not-promoted")
                    if validation.get("candidate_only_entry_count") != validation.get("required_algorithm_class_entry_count"):
                        errors.append("all-algorithm calibration smoke settings entries must all be candidate-only")
            except Exception as exc:
                errors.append(f"class_settings_manifest validation failed: {type(exc).__name__}: {exc}")
    if case_manifest_path is not None and case_manifest_path.exists() and case_id:
        try:
            case = _load_case_from_manifest(case_manifest=case_manifest_path, case_id=case_id)
            if case is None:
                errors.append(f"case_id {case_id!r} not found in case_manifest")
            else:
                if str(case.get("family", "")) != family:
                    errors.append(f"case_manifest family {case.get('family')!r} does not match row family {family!r}")
                if str(case.get("tuning_class", "")) != tuning_class:
                    errors.append(
                        f"case_manifest tuning_class {case.get('tuning_class')!r} does not match row tuning_class {tuning_class!r}"
                    )
                artifact_raw = str(case.get("artifact_json", "")).strip()
                if not artifact_raw:
                    errors.append("case manifest entry missing artifact_json")
                else:
                    runtime_artifact_path = _resolve_row_path(
                        artifact_raw,
                        repo_root=repo_root,
                        base_dir=case_manifest_path.parent,
                    )
                    if not runtime_artifact_path.exists():
                        errors.append(f"case runtime artifact does not exist: {artifact_raw}")
                metadata = case.get("metadata", {}) if isinstance(case.get("metadata", {}), Mapping) else {}
                seed_lock = metadata.get("seed_lock", {}) if isinstance(metadata.get("seed_lock", {}), Mapping) else {}
                if seed_lock.get("seed_artifact_sha256") != row.get("seed_artifact_sha256"):
                    errors.append("seed_artifact_sha256 does not match case manifest seed lock")
                if seed_lock.get("same_seed_comparator_group_id") != row.get("same_seed_comparator_group_id"):
                    errors.append("same_seed_comparator_group_id does not match case manifest seed lock")
                if metadata.get("candidate_only_not_promoted") is not True:
                    errors.append("case manifest metadata must be candidate-only/not-promoted")
                if metadata.get("controller_settings_scope") != "coarse_hamiltonian_class":
                    errors.append("case manifest metadata must set controller_settings_scope=coarse_hamiltonian_class")
                if metadata.get("static_scaffold_scope") != "benchmark_point":
                    errors.append("case manifest metadata must set static_scaffold_scope=benchmark_point")
                if metadata.get("require_algorithm_class_settings") is not True:
                    errors.append("case manifest metadata must require algorithm class settings")
                if metadata.get("require_parity_correctness_sidecars") is not True:
                    errors.append("case manifest metadata must require parity/correctness sidecars")
        except Exception as exc:
            errors.append(f"case_manifest validation failed: {type(exc).__name__}: {exc}")
    profile_raw = str(row.get("candidate_search_profile_json", "")).strip()
    if profile_raw:
        try:
            profile = json.loads(profile_raw)
            if str(profile.get("profile_id", "")) != str(row.get("candidate_search_profile_id", "")).strip():
                errors.append("candidate_search_profile_json profile_id does not match row")
            if str(profile.get("settings_kind", "")) != settings_kind:
                errors.append("candidate_search_profile_json settings_kind does not match row")
        except Exception as exc:
            errors.append(f"candidate_search_profile_json is invalid: {type(exc).__name__}: {exc}")
    return {
        "record_id": record_id,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_artifact_json": str(row.get("source_artifact_json", "")).strip(),
        "source_artifact_path": None,
        "artifact_json": str(row.get("artifact_json", "")).strip(),
        "staged_artifact_path": None,
        "staged_artifact_exists": False,
        "staged_artifact_size_bytes": None,
        "runtime_artifact_path": None if runtime_artifact_path is None else str(runtime_artifact_path),
        "validation_profile": str(row.get("validation_profile", "")),
        "study_profile": str(row.get("study_profile", "")),
        "kind": "benchmark",
        "algorithm_id": algorithm_id,
        "candidate_only_not_promoted": parse_bool(row.get("candidate_only_not_promoted"), default=False),
    }


def validate_record_row(
    row: Mapping[str, str],
    *,
    repo_root: Path,
    stage: bool = True,
    validate_load: bool = False,
    records_dir: Path | None = None,
) -> dict[str, Any]:
    if _is_generic_benchmark_row(row):
        return validate_generic_benchmark_record_row(row, repo_root=repo_root, records_dir=records_dir)
    record_id = str(row.get("record_id", "")).strip()
    errors: list[str] = []
    warnings: list[str] = []
    staged_path: str | None = None
    source_path: str | None = None
    if not record_id:
        errors.append("missing record_id")
    n_jobs = str(row.get("n_jobs", "")).strip()
    if n_jobs not in {"", "1"}:
        errors.append(f"n_jobs must be blank or 1 for sequential TD Optuna, got {n_jobs!r}")
    profile_error = validate_profile(row)
    if profile_error:
        errors.append(profile_error)
    _validate_optional_positive_float(
        row,
        "invalid_max_primary_observable_mae_over_span",
        errors,
    )
    is_strict_validation = str(row.get("validation_profile", "")).strip() in {"strict_qpu_faithful", "strict_qpu_hh"}
    if (
        is_strict_validation
        and str(row.get("invalid_max_primary_observable_mae_over_span", "")).strip()
    ):
        errors.append(
            "invalid_max_primary_observable_mae_over_span is diagnostic exact-v1 only; "
            "strict/QPU-faithful records must leave it blank"
        )
    if is_strict_validation:
        for key in sorted(_STRICT_EXACT_ONLY_OPTION_KEYS):
            if key == "invalid_max_primary_observable_mae_over_span":
                continue
            if _nonzero_text(row.get(key, "")):
                errors.append(
                    f"{key} is exact-reference objective/gate feedback; strict/QPU-faithful records must leave it blank or zero"
                )
    try:
        enable_drive = parse_bool(row.get("enable_drive"), default=False)
        disable_drive = parse_bool(row.get("disable_drive"), default=False)
        if enable_drive and disable_drive:
            errors.append("enable_drive and disable_drive cannot both be true")
    except ValueError as exc:
        enable_drive = False
        errors.append(str(exc))
    family = str(row.get("family", "")).strip()
    tuning_class = str(row.get("tuning_class", "")).strip()
    if str(row.get("study_profile", "")).strip() in _STRICT_APPEND_PRUNE_PROFILES and not tuning_class:
        errors.append("strict append/prune class-policy records must set tuning_class")
    if tuning_class:
        try:
            validate_dynamics_tuning_class(family=family, tuning_class=tuning_class)
        except ValueError as exc:
            errors.append(str(exc))
    if (
        str(row.get("study_profile", "")).strip() == "generic_l2_exact_v1"
        and family in _BOSON_DRIVEN_FAMILIES
        and not enable_drive
    ):
        errors.append(f"{family} generic_l2_exact_v1 records must enable drive")
    drive_time_sampling = str(row.get("drive_time_sampling", "")).strip()
    if drive_time_sampling not in _DRIVE_TIME_SAMPLING:
        errors.append(
            "drive_time_sampling must be blank or one of "
            f"{sorted(item for item in _DRIVE_TIME_SAMPLING if item)}, got {drive_time_sampling!r}"
        )
    source_raw = str(row.get("source_artifact_json", "")).strip()
    artifact_raw = str(row.get("artifact_json", "")).strip()
    if not artifact_raw:
        errors.append("missing artifact_json")
    if source_raw:
        source = repo_path(source_raw, repo_root=repo_root)
        source_path = str(source)
        if not source.exists():
            errors.append(f"source_artifact_json does not exist: {source_raw}")
    else:
        errors.append("missing source_artifact_json")
        source = None
    if artifact_raw:
        artifact = repo_path(artifact_raw, repo_root=repo_root)
        staged_path = str(artifact)
        expected_name = f"{record_id}.json"
        if artifact.name != expected_name:
            errors.append(f"artifact_json must end with {expected_name!r}")
        expected_rel = (
            SEED_ARTIFACTS_REL
            / expected_name
        )
        try:
            actual_rel = artifact.resolve().relative_to(repo_root.resolve())
        except ValueError:
            actual_rel = None
        if actual_rel != expected_rel:
            errors.append(f"artifact_json must equal {expected_rel.as_posix()!r}")
    else:
        artifact = None
    if not errors and stage and source is not None and artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != artifact.resolve():
            shutil.copy2(source, artifact)
    if not errors and validate_load and artifact is not None:
        try:
            from pipelines.scaffold.runtime_loader import load_scaffold_runtime_input

            load_scaffold_runtime_input(
                artifact,
                loader_mode=str(row.get("loader_mode", "") or "replay_family"),
                tag=record_id,
                generator_family=str(row.get("generator_family", "") or "match_adapt"),
                fallback_family=str(row.get("fallback_family", "") or "full_meta"),
            )
        except Exception as exc:  # pragma: no cover - optional heavy validation
            errors.append(f"load_scaffold_runtime_input failed: {type(exc).__name__}: {exc}")
    if artifact is not None and artifact.exists():
        try:
            size_bytes = artifact.stat().st_size
        except OSError:
            size_bytes = None
        if size_bytes is not None and size_bytes <= 0:
            errors.append(f"staged artifact is empty: {artifact}")
    else:
        size_bytes = None
    return {
        "record_id": record_id,
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "source_artifact_json": source_raw,
        "source_artifact_path": source_path,
        "artifact_json": artifact_raw,
        "staged_artifact_path": staged_path,
        "staged_artifact_exists": bool(artifact is not None and artifact.exists()),
        "staged_artifact_size_bytes": size_bytes,
        "validation_profile": str(row.get("validation_profile", "")),
        "study_profile": str(row.get("study_profile", "")),
    }


def cleanup_stale_staged_artifacts(
    selected: Sequence[Mapping[str, str]],
    *,
    repo_root: Path,
) -> list[str]:
    """Remove staged seed artifacts not needed by the selected queue.

    CHTC transfers the harness directory as one input bundle. Keeping stale
    artifacts from an earlier all-record preflight can turn a smoke job into a
    multi-GiB transfer. The canonical sources remain in ``source_artifact_json``;
    this only prunes the transient ``input/seed_artifacts`` staging cache.
    """
    keep: set[str] = set()
    for row in selected:
        record_id = str(row.get("record_id", "")).strip()
        artifact_raw = str(row.get("artifact_json", "")).strip()
        if record_id and artifact_raw:
            keep.add(f"{record_id}.json")
    seed_dir = repo_root / SEED_ARTIFACTS_REL
    if not seed_dir.exists():
        return []
    removed: list[str] = []
    for path in seed_dir.glob("*.json"):
        if path.name not in keep:
            path.unlink()
            try:
                removed.append(str(path.relative_to(repo_root)))
            except ValueError:
                removed.append(str(path))
    return removed


def preflight_records(
    *,
    records_path: str | Path = DEFAULT_RECORDS,
    record_list: str | Path | None = None,
    record_ids: Sequence[str] | None = None,
    repo_root: str | Path | None = None,
    stage: bool = True,
    validate_load: bool = False,
    write_report: bool = True,
    all_records: bool = False,
    clean_staged: bool = True,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    records_file = Path(records_path)
    rows = load_records(records_file)
    selected = selected_records(rows, record_ids=record_ids, record_list=record_list, all_records=all_records)
    removed_stale_artifacts = (
        cleanup_stale_staged_artifacts(selected, repo_root=repo)
        if stage and clean_staged
        else []
    )
    record_results = [
        validate_record_row(
            row,
            repo_root=repo,
            stage=stage,
            validate_load=validate_load,
            records_dir=records_file.parent,
        )
        for row in selected
    ]
    report = {
        "generated_utc": _now_utc(),
        "records_path": str(records_file),
        "record_list": None if record_list is None else str(record_list),
        "record_count": len(record_results),
        "ok": all(item.get("ok") for item in record_results),
        "clean_staged": bool(stage and clean_staged),
        "removed_stale_artifacts": removed_stale_artifacts,
        "records": record_results,
        "failed_records": [item["record_id"] for item in record_results if not item.get("ok")],
    }
    if write_report:
        report_path = records_file.parent / REPORT_NAME
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["report_path"] = str(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preflight and stage TD Optuna CHTC input records.")
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--record-list", type=Path, default=None)
    parser.add_argument("--record-id", action="append", default=[])
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--all", action="store_true", help="Validate every row in records.tsv.")
    parser.add_argument("--no-stage", action="store_true", help="Validate without copying source artifacts.")
    parser.add_argument(
        "--keep-stale-staged",
        action="store_true",
        help="Do not prune input/seed_artifacts/*.json outside the selected queue.",
    )
    parser.add_argument("--validate-load", action="store_true", help="Also load staged artifacts through runtime_loader.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = preflight_records(
        records_path=args.records,
        record_list=args.record_list,
        record_ids=args.record_id,
        repo_root=args.repo_root,
        stage=not bool(args.no_stage),
        validate_load=bool(args.validate_load),
        all_records=bool(args.all),
        clean_staged=not bool(args.keep_stale_staged),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
