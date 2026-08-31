#!/usr/bin/env python3
"""Generate the Paper III n_ph compatibility/audit matrix.

This module is metadata-only.  It mines already-wired Paper-I/Table-I,
Paper-II dynamics, registry, and QSE/comparator planning surfaces and emits a
local audit scaffold.  It intentionally does not submit CHTC jobs, run realtime
dynamics, or implement missing comparator physics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.exact_bench import table_i_canonical_cases as table_i_cases  # noqa: E402
from pipelines.static_adapt.builders import problem_registry  # noqa: E402

PROFILE = "paper_iii_nph1_compatibility_v1"
MATRIX_SCHEMA_VERSION = "paper_iii_compatibility_matrix_manifest_v1"
BLOCKERS_SCHEMA_VERSION = "paper_iii_compatibility_matrix_blockers_v1"
RECORDS_SCHEMA_VERSION = "paper_iii_compatibility_matrix_records_v1"
MATRIX_BATCH_ID = "paper_iii_nph1_compatibility_v1_local_audit"
MODE_COMPATIBILITY_AUDIT_ONLY = "compatibility_audit_only"
NO_SUBMIT_POLICY = (
    "local_audit_only_no_condor_submit_no_realtime_dynamics_no_route_default_changes"
)
PAPER_II_CASES_REL = Path("chtc/generic_time_dynamics_table/input/paper_ii_class_tuned_cases_v1.json")
PAPER_II_CLASS_SETTINGS_REL = Path(
    "chtc/generic_time_dynamics_table/input/class_settings/paper_ii_class_settings_lock_v1.json"
)
DEFAULT_OUTPUT_DIR = Path("chtc/paper_iii_excited_dynamics/input/compatibility_matrix_nph1")
TARGET_EXCITED_ROOTS = 6
N_PH_MAX = 1
SMOKE_COMPARATOR_ID = "geo_selected_qse"
REGISTRY_SCOPE_FAMILY = "molecular_restricted_closed_shell"

RECORD_FIELDNAMES = [
    "record_id",
    "queue",
    "mode",
    "matrix_batch_id",
    "profile",
    "row_kind",
    "family",
    "case_id",
    "paper_i_case_id",
    "paper_ii_case_id",
    "n_ph_max",
    "target_excited_roots",
    "static_artifact_path",
    "static_artifact_sha256",
    "static_artifact_status",
    "seed_source_ledger",
    "seed_is_local",
    "seed_is_staged_or_recovery",
    "latest_phase3_source_artifact_missing_locally",
    "paper_ii_seed_policy",
    "operator_support_source",
    "qse_basis_source",
    "comparator_id",
    "comparator_family",
    "comparator_status",
    "comparator_blockers",
    "qse_matrix_measurement_proxy_status",
    "ed_same_cutoff_status",
    "ed_higher_cutoff_status",
    "dynamics_class",
    "dynamics_settings_source",
    "t_final",
    "num_times",
    "time_grid_convention",
    "drive_enabled",
    "drive_amplitude",
    "drive_frequency_grid_id",
    "paper_ii_drive_pattern",
    "controller_exact_input_mode",
    "diagnostic_exact_reference_mode",
    "uses_reference_for_decision",
    "uses_future_exact_forecast_for_decision",
    "uses_statevector_as_ideal_observable_estimator",
    "exact_decision_checkpoints",
    "strict_measurement_oracle_certified",
    "qpu_faithful_decisions_passed",
    "strict_policy_status",
    "expected_status",
    "blocker_count",
    "blocker_codes",
    "no_submit",
    "no_submit_policy",
    "require_progress_json",
    "require_partial_payload_json",
    "require_science_benchmark",
    "notes",
]

BLOCKER_FIELDNAMES = [
    "record_id",
    "family",
    "case_id",
    "paper_i_case_id",
    "drive_amplitude",
    "comparator_id",
    "code",
    "source",
    "severity",
    "detail",
    "no_submit",
]

SEED_REPAIR_INVENTORY_SCHEMA_VERSION = "paper_iii_seed_repair_inventory_v1"
SEED_REPAIR_ALLOWED_STATUSES = (
    "repairable_local",
    "missing_local",
    "staged_or_recovery_only",
    "path_escapes_repo",
    "manual_decision_required",
)
SEED_STAGED_MARKERS = ("staged", "pending", "recovery", "fallback")
SEED_STAGED_PATH_FRAGMENTS = (
    "chtc/time_dynamics_optuna/input/",
    "seed_artifacts_table_lock_available_phase3/",
    "chtc/paper_iii_excited_dynamics/input/seed_artifacts/",
)
SEED_REPAIR_FIELDNAMES = [
    "slot_id",
    "family",
    "paper_i_case_id",
    "paper_ii_case_id",
    "drive_amplitude",
    "row_count",
    "comparator_ids",
    "current_static_artifact_path",
    "current_static_artifact_sha256",
    "current_static_artifact_status",
    "seed_is_local",
    "seed_is_staged_or_recovery",
    "latest_phase3_source_artifact_missing_locally",
    "paper_ii_seed_policy",
    "selected_static_seed_source",
    "same_seed_comparator_group_id",
    "source_artifact_json",
    "source_artifact_path_under_repo",
    "source_artifact_exists",
    "source_artifact_actual_sha256",
    "source_artifact_hash_matches",
    "inherited_seed_blocker_codes",
    "candidate_canonical_static_artifact_path",
    "candidate_canonical_static_artifact_sha256",
    "candidate_canonical_static_artifact_source",
    "repair_status",
    "notes",
]


@dataclass(frozen=True)
class ComparatorSpec:
    comparator_id: str
    comparator_family: str
    qse_basis_source: str
    default_status: str
    blocker_codes: tuple[str, ...]
    notes: str


COMPARATOR_SPECS: tuple[ComparatorSpec, ...] = (
    ComparatorSpec(
        comparator_id="geo_selected_qse",
        comparator_family="geometry_aware_qse",
        qse_basis_source="selected_adapt_blocks",
        default_status="implemented_partial",
        blocker_codes=("qse.selected_adapt_blocks_require_local_seed_artifact",),
        notes="Geometry-aware selected-block QSE path exists, but all-family six-root support must be audited from local static artifacts.",
    ),
    ComparatorSpec(
        comparator_id="fixed_alphabet_qse",
        comparator_family="fixed_alphabet_qse",
        qse_basis_source="fixed_alphabet_registry",
        default_status="blocked_missing_comparator",
        blocker_codes=("comparator.fixed_alphabet_registry_missing",),
        notes="No canonical Paper III fixed-alphabet registry is wired for all families.",
    ),
    ComparatorSpec(
        comparator_id="family_restricted_qse",
        comparator_family="family_restricted_qse",
        qse_basis_source="family_restricted_alphabet_registry",
        default_status="blocked_missing_comparator",
        blocker_codes=("comparator.family_restricted_basis_registry_missing",),
        notes="Family-restricted QSE alphabet choices are not yet canonicalized for this matrix.",
    ),
    ComparatorSpec(
        comparator_id="full_meta_largest_alphabet_qse",
        comparator_family="full_meta_qse",
        qse_basis_source="full_meta_or_full_meta_filtered",
        default_status="implemented_partial",
        blocker_codes=("qse.full_meta_hh_path_requires_six_root_audit",),
        notes="Current full_meta QSE support is HH-centered; non-HH rows are blocked explicitly.",
    ),
    ComparatorSpec(
        comparator_id="hamiltonian_derived_qse",
        comparator_family="hamiltonian_term_qse",
        qse_basis_source="hamiltonian_terms",
        default_status="implemented_partial",
        blocker_codes=("qse.hamiltonian_term_measurement_proxy_unvalidated",),
        notes="Hamiltonian-term basis inclusion is available as metadata, but measurement proxy and six-root readiness are not validated here.",
    ),
    ComparatorSpec(
        comparator_id="krylov_hamiltonian_power_qse",
        comparator_family="krylov_qse",
        qse_basis_source="hamiltonian_powers",
        default_status="blocked_missing_comparator",
        blocker_codes=("comparator.krylov_hamiltonian_power_qse_not_implemented",),
        notes="Krylov/Hamiltonian-power QSE is not wired as a Paper III comparator row.",
    ),
    ComparatorSpec(
        comparator_id="qeom_qsc_eom",
        comparator_family="eom_qse",
        qse_basis_source="qeom_qsc_eom",
        default_status="blocked_missing_comparator",
        blocker_codes=("comparator.qeom_qsc_eom_not_implemented",),
        notes="qEOM/q-sc-EOM is not wired as a Paper III comparator row.",
    ),
    ComparatorSpec(
        comparator_id="qse_coefficient_propagation",
        comparator_family="qse_time_propagation",
        qse_basis_source="frozen_or_adaptive_qse_coefficients",
        default_status="blocked_missing_comparator",
        blocker_codes=("comparator.qse_coefficient_propagation_not_matrix_wired",),
        notes="Frozen/adaptive QSE propagation prototypes are not yet all-family production comparator rows.",
    ),
)


def _now_utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_n_ph_max(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"n_ph_max must be an integer >= 1; got {value!r}")
    out = int(value)
    if out < 1:
        raise ValueError(f"n_ph_max must be >= 1; got {out}")
    return out


def _profile_for_n_ph(n_ph_max: int) -> str:
    return f"paper_iii_nph{_validate_n_ph_max(n_ph_max)}_compatibility_v1"


def _matrix_batch_id_for_n_ph(n_ph_max: int) -> str:
    return f"{_profile_for_n_ph(n_ph_max)}_local_audit"


def _default_output_dir_for_n_ph(n_ph_max: int) -> Path:
    return Path(f"chtc/paper_iii_excited_dynamics/input/compatibility_matrix_nph{_validate_n_ph_max(n_ph_max)}")


def _resolve_profile(*, profile: str | None, n_ph_max: int) -> str:
    expected = _profile_for_n_ph(n_ph_max)
    if profile is None or str(profile) == PROFILE:
        return expected
    if str(profile) != expected:
        raise ValueError(f"unsupported profile {profile!r}; expected {expected!r}")
    return str(profile)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _bool_text(value: bool) -> str:
    return "true" if bool(value) else "false"


def _safe_record_fragment(value: Any) -> str:
    text = _clean(value).lower().replace(".", "p")
    out = []
    for char in text:
        out.append(char if char.isalnum() else "_")
    return "_".join(part for part in "".join(out).split("_") if part)


def _amplitude_label(value: Any) -> str:
    return "A" + _safe_record_fragment(f"{float(value):g}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relative_or_absolute(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_under_repo(raw_path: str, *, repo_root: Path) -> tuple[Path | None, bool]:
    if not raw_path:
        return None, False
    raw = Path(raw_path).expanduser()
    candidate = raw if raw.is_absolute() else repo_root / raw
    try:
        candidate.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return None, True
    return candidate, False


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source_file_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    return {
        "path": _relative_or_absolute(path, repo_root=repo_root),
        "exists": path.exists(),
        "sha256": _sha256_file(path) if path.exists() else None,
    }


def _setting_by_tuning_class(settings_payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for row in settings_payload.get("settings") or []:
        if isinstance(row, Mapping):
            tuning_class = _clean(row.get("tuning_class"))
            if tuning_class:
                out[tuning_class] = row
    return out


def _paper_ii_cases_by_family_and_amplitude(cases_payload: Mapping[str, Any]) -> dict[tuple[str, float], Mapping[str, Any]]:
    out: dict[tuple[str, float], Mapping[str, Any]] = {}
    for row in cases_payload.get("cases") or []:
        if not isinstance(row, Mapping):
            continue
        family = _clean(row.get("family"))
        drive = row.get("metadata", {}).get("drive", {}) if isinstance(row.get("metadata"), Mapping) else {}
        try:
            amplitude = float(drive.get("A"))
        except Exception:
            continue
        out[(family, amplitude)] = row
    return out


def _drive_amplitudes(cases_payload: Mapping[str, Any]) -> tuple[float, ...]:
    amplitudes: set[float] = set()
    for row in cases_payload.get("cases") or []:
        if not isinstance(row, Mapping):
            continue
        metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
        drive = metadata.get("drive") if isinstance(metadata.get("drive"), Mapping) else {}
        try:
            amplitudes.add(float(drive.get("A")))
        except Exception:
            pass
    return tuple(sorted(amplitudes))


def _case_count(declared_by_family: Mapping[str, Sequence[str]]) -> int:
    return sum(len(tuple(case_ids)) for case_ids in declared_by_family.values())


def _class_setting_status(
    *,
    case: Mapping[str, Any] | None,
    settings_by_class: Mapping[str, Mapping[str, Any]],
    class_settings_payload: Mapping[str, Any],
) -> tuple[str, str, list[str]]:
    blockers: list[str] = []
    if case is None:
        return "blocked", "", ["dynamics.paper_ii_case_missing"]
    tuning_class = _clean(case.get("tuning_class"))
    metadata = case.get("metadata") if isinstance(case.get("metadata"), Mapping) else {}
    if metadata.get("controller_settings_scope") != "coarse_hamiltonian_class":
        blockers.append("strict.controller_settings_scope_not_coarse_class")
    if class_settings_payload.get("lock_status") != "locked":
        blockers.append("strict.class_settings_lock_status_not_locked")
    setting = settings_by_class.get(tuning_class)
    source = ""
    if not setting:
        blockers.append("strict.class_setting_missing_for_dynamics_class")
    else:
        source = f"{PAPER_II_CLASS_SETTINGS_REL.as_posix()}:{setting.get('settings_id')}"
        payload = setting.get("settings_payload") if isinstance(setting.get("settings_payload"), Mapping) else {}
        if setting.get("class_tuned_result_locked") is not True:
            blockers.append("strict.class_tuned_result_not_locked")
        if setting.get("strict_online_feedback_exact_free") is not True:
            blockers.append("strict.online_feedback_not_exact_free")
        if payload.get("checkpoint_controller_exact_input_mode") != "off":
            blockers.append("strict.checkpoint_controller_exact_input_mode_not_off")
    return ("pass" if not blockers else "blocked"), source, blockers


def _artifact_status_and_seed_blockers(
    *,
    case: Mapping[str, Any] | None,
    repo_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    if case is None:
        return (
            {
                "static_artifact_path": "",
                "static_artifact_sha256": "",
                "static_artifact_status": "missing_paper_ii_case",
                "seed_source_ledger": PAPER_II_CASES_REL.as_posix(),
                "seed_is_local": False,
                "seed_is_staged_or_recovery": False,
                "latest_phase3_source_artifact_missing_locally": False,
                "paper_ii_seed_policy": "",
            },
            ["seed.paper_ii_case_missing"],
        )
    metadata = case.get("metadata") if isinstance(case.get("metadata"), Mapping) else {}
    seed_lock = metadata.get("seed_lock") if isinstance(metadata.get("seed_lock"), Mapping) else {}
    source_rel = _clean(seed_lock.get("source_artifact_json"))
    expected_sha = _clean(seed_lock.get("seed_artifact_sha256"))
    seed_policy = _clean(seed_lock.get("seed_selection_policy"))
    selected_source = _clean(seed_lock.get("selected_static_seed_source"))
    local_path, path_escapes_repo = _resolve_under_repo(source_rel, repo_root=repo_root)
    local_exists = bool(local_path is not None and local_path.exists())
    actual_sha = _sha256_file(local_path) if local_exists and local_path is not None else ""
    blockers: list[str] = []
    latest_missing = bool(metadata.get("latest_phase3_source_artifact_missing_locally"))
    if latest_missing:
        blockers.append("seed.latest_phase3_source_artifact_missing_locally")
    lower_policy = f"{seed_policy} {selected_source}".lower()
    if any(marker in lower_policy for marker in ("staged", "pending", "recovery", "fallback")):
        blockers.append("seed.staged_or_recovery_policy")
    if source_rel.startswith("chtc/time_dynamics_optuna/input/"):
        blockers.append("seed.source_points_to_staged_dynamics_path")
    if not source_rel:
        blockers.append("seed.source_artifact_json_missing")
    elif path_escapes_repo:
        blockers.append("seed.source_artifact_path_escapes_repo")
    elif not local_exists:
        blockers.append("seed.local_artifact_missing")
    elif expected_sha and actual_sha != expected_sha:
        blockers.append("seed.hash_mismatch")
    staged_or_recovery = any(
        marker in f"{seed_policy} {selected_source} {source_rel}".lower()
        for marker in ("staged", "pending", "recovery", "fallback")
    )
    status = "path_escapes_repo" if path_escapes_repo else ("missing" if not local_exists else ("hash_match" if actual_sha == expected_sha else "hash_mismatch"))
    return (
        {
            "static_artifact_path": source_rel,
            "static_artifact_sha256": expected_sha,
            "static_artifact_status": status,
            "seed_source_ledger": PAPER_II_CASES_REL.as_posix(),
            "seed_is_local": local_exists,
            "seed_is_staged_or_recovery": staged_or_recovery,
            "latest_phase3_source_artifact_missing_locally": latest_missing,
            "paper_ii_seed_policy": seed_policy,
        },
        blockers,
    )


def _operator_support_source(family: str) -> str:
    try:
        spec = problem_registry.get_problem_family_spec(family)
    except Exception:
        return "problem_registry:missing"
    default_pool = spec.default_pool_key or "controller_resolved"
    return f"problem_registry:{family}:default_pool={default_pool}:admissible={','.join(spec.admissible_pool_keys)}"


def _comparator_status_and_blockers(family: str, spec: ComparatorSpec) -> tuple[str, list[str]]:
    blockers = list(spec.blocker_codes)
    status = spec.default_status
    if spec.comparator_id == "full_meta_largest_alphabet_qse" and family != "hh":
        status = "blocked_family_path"
        blockers = ["comparator.full_meta_basis_source_hh_centered_for_non_hh"]
    if spec.comparator_id in {"geo_selected_qse", "full_meta_largest_alphabet_qse", "hamiltonian_derived_qse"}:
        blockers.append("qse.target_six_excited_roots_not_validated_for_matrix")
    return status, blockers


def _blocker_source(code: str) -> str:
    if code.startswith("seed."):
        return "paper_ii_seed_provenance"
    if code.startswith("comparator.") or code.startswith("qse."):
        return "qse_comparator_metadata"
    if code.startswith("dynamics.") or code.startswith("strict."):
        return "paper_ii_dynamics_lock"
    if code.startswith("static."):
        return "table_i_static_contract"
    if code.startswith("scope."):
        return "problem_registry_scope"
    return "compatibility_matrix"


def _blocker_detail(code: str) -> str:
    details = {
        "seed.latest_phase3_source_artifact_missing_locally": "Paper-II manifest says latest Phase-III source artifact is missing locally; paper-facing use is blocked until repaired.",
        "seed.staged_or_recovery_policy": "Seed policy contains staged/pending/recovery/fallback provenance; keep as blocker, not as final evidence.",
        "seed.source_points_to_staged_dynamics_path": "Seed source points at staged dynamics input rather than a static ADAPT source artifact.",
        "seed.local_artifact_missing": "Static seed artifact path is not present locally.",
        "seed.hash_mismatch": "Local static seed artifact hash does not match seed_lock metadata.",
        "seed.source_artifact_path_escapes_repo": "Seed source artifact path resolves outside the repository root and was not hashed.",
        "qse.target_six_excited_roots_not_validated_for_matrix": "Target six-root QSE/excited-state support has not been validated for this row.",
        "qse.selected_adapt_blocks_require_local_seed_artifact": "Selected-block QSE requires inspecting the local static artifact for ADAPT block metadata.",
        "qse.full_meta_hh_path_requires_six_root_audit": "HH full_meta QSE support exists only as partial support until six-root behavior is audited.",
        "qse.hamiltonian_term_measurement_proxy_unvalidated": "Hamiltonian-term QSE measurement proxy/cost accounting is not validated for this matrix row.",
        "comparator.fixed_alphabet_registry_missing": "No canonical all-family fixed-alphabet comparator registry is wired.",
        "comparator.family_restricted_basis_registry_missing": "No canonical all-family family-restricted QSE comparator registry is wired.",
        "comparator.full_meta_basis_source_hh_centered_for_non_hh": "The current full_meta/full_meta_filtered QSE path is HH-centered for this audit slice.",
        "comparator.krylov_hamiltonian_power_qse_not_implemented": "Krylov/Hamiltonian-power QSE comparator is not implemented for Paper III production rows.",
        "comparator.qeom_qsc_eom_not_implemented": "qEOM/q-sc-EOM comparator is not implemented for Paper III production rows.",
        "comparator.qse_coefficient_propagation_not_matrix_wired": "QSE coefficient propagation is not wired as an all-family production comparator row.",
        "static.table_i_case_deferred_at_head": "Declared Table-I case is intentionally deferred at this HEAD.",
        "scope.registry_only_family_not_table_i_declared": "Registry-only family is wired but not part of the declared Table-I Paper-III family set.",
        "scope.requires_user_decision_for_paper_iii": "Paper III inclusion/exclusion needs an explicit future scope decision.",
    }
    return details.get(code, code.replace("_", " "))


def _dedupe_codes(codes: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for code in codes:
        clean = _clean(code)
        if clean and clean not in seen:
            seen.add(clean)
            out.append(clean)
    return out


def _split_codes(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()]


def _has_staged_seed_marker(*values: Any) -> bool:
    text = " ".join(_clean(value).lower() for value in values)
    if any(marker in text for marker in SEED_STAGED_MARKERS):
        return True
    return any(fragment in text for fragment in SEED_STAGED_PATH_FRAGMENTS)


def _seed_slot_id(*, family: str, paper_i_case_id: str, drive_amplitude: str, n_ph_max: int = N_PH_MAX) -> str:
    return (
        f"paper_iii_nph{_validate_n_ph_max(n_ph_max)}_seed_slot_"
        f"{_safe_record_fragment(family)}_"
        f"{_safe_record_fragment(paper_i_case_id)}_"
        f"{_amplitude_label(drive_amplitude)}"
    )


def _source_path_audit(
    raw_path: str,
    *,
    repo_root: Path,
    expected_sha256: str,
) -> dict[str, str]:
    local_path, escapes_repo = _resolve_under_repo(raw_path, repo_root=repo_root)
    exists = bool(local_path is not None and local_path.exists())
    actual_sha = _sha256_file(local_path) if exists and local_path is not None else ""
    if not raw_path:
        hash_matches = ""
    elif exists and expected_sha256:
        hash_matches = _bool_text(actual_sha == expected_sha256)
    else:
        hash_matches = ""
    return {
        "source_artifact_path_under_repo": _bool_text(bool(raw_path and not escapes_repo)),
        "source_artifact_exists": _bool_text(exists),
        "source_artifact_actual_sha256": actual_sha,
        "source_artifact_hash_matches": hash_matches,
    }


def _seed_repair_status(
    *,
    current_status: str,
    seed_blocker_codes: Sequence[str],
    source_exists: bool,
    source_path_under_repo: bool,
    staged_marker_present: bool,
    candidate_path: str,
) -> str:
    codes = set(seed_blocker_codes)
    if current_status == "path_escapes_repo" or "seed.source_artifact_path_escapes_repo" in codes or not source_path_under_repo:
        return "path_escapes_repo"
    if staged_marker_present and not source_exists and not candidate_path:
        return "missing_local"
    if staged_marker_present:
        return "staged_or_recovery_only"
    if candidate_path:
        return "repairable_local"
    if current_status in {"missing", "missing_paper_ii_case"} or "seed.local_artifact_missing" in codes:
        return "missing_local"
    return "manual_decision_required"


def _inventory_candidate_from_source(
    *,
    source_rel: str,
    source_exists: bool,
    source_path_under_repo: bool,
    source_sha256: str,
    staged_marker_present: bool,
) -> tuple[str, str, str]:
    if source_rel and source_exists and source_path_under_repo and not staged_marker_present:
        return source_rel, source_sha256, "paper_ii_seed_lock.source_artifact_json"
    return "", "", ""


def _seed_blockers_by_slot(blockers: Sequence[Mapping[str, str]]) -> dict[tuple[str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str], list[str]] = {}
    for blocker in blockers:
        code = _clean(blocker.get("code"))
        if not code.startswith("seed."):
            continue
        key = (
            _clean(blocker.get("family")),
            _clean(blocker.get("paper_i_case_id")),
            _clean(blocker.get("drive_amplitude")),
        )
        grouped.setdefault(key, [])
        if code not in grouped[key]:
            grouped[key].append(code)
    return {key: sorted(codes) for key, codes in grouped.items()}


def build_seed_repair_inventory(
    *,
    rows: Sequence[Mapping[str, str]],
    blockers: Sequence[Mapping[str, str]],
    paper_ii_cases_payload: Mapping[str, Any],
    repo_root: str | Path = REPO_ROOT,
    profile: str = PROFILE,
    matrix_batch_id: str = MATRIX_BATCH_ID,
    n_ph_max: int = N_PH_MAX,
) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    comparator_rows = [row for row in rows if _clean(row.get("row_kind")) == "comparator_row"]
    scope_rows = [row for row in rows if _clean(row.get("row_kind")) == "scope_decision"]
    groups: dict[tuple[str, str, str], list[Mapping[str, str]]] = {}
    for row in comparator_rows:
        key = (_clean(row.get("family")), _clean(row.get("paper_i_case_id")), _clean(row.get("drive_amplitude")))
        groups.setdefault(key, []).append(row)

    paper_ii_by_key = _paper_ii_cases_by_family_and_amplitude(paper_ii_cases_payload)
    declared_by_family = table_i_cases.table_i_declared_case_ids_by_family(table_i_cases.TABLE_I_STANDARD_PROFILE)
    drive_amplitudes = sorted({_clean(row.get("drive_amplitude")) for row in comparator_rows}, key=lambda value: float(value))
    expected_unique_seed_slots = _case_count(declared_by_family) * len(drive_amplitudes)
    seed_blockers = _seed_blockers_by_slot(blockers)
    comparator_order = {spec.comparator_id: idx for idx, spec in enumerate(COMPARATOR_SPECS)}

    slots: list[dict[str, str]] = []
    for family, paper_i_case_id, drive_amplitude in sorted(groups, key=lambda key: (key[0], key[1], float(key[2]))):
        group_rows = groups[(family, paper_i_case_id, drive_amplitude)]
        first = group_rows[0]
        try:
            amplitude_float = float(drive_amplitude)
        except Exception:
            amplitude_float = 0.0
        paper_ii_case = paper_ii_by_key.get((family, amplitude_float))
        metadata = paper_ii_case.get("metadata") if isinstance(paper_ii_case, Mapping) and isinstance(paper_ii_case.get("metadata"), Mapping) else {}
        seed_lock = metadata.get("seed_lock") if isinstance(metadata.get("seed_lock"), Mapping) else {}
        source_rel = _clean(seed_lock.get("source_artifact_json")) or _clean(first.get("static_artifact_path"))
        expected_sha = _clean(seed_lock.get("seed_artifact_sha256")) or _clean(first.get("static_artifact_sha256"))
        source_audit = _source_path_audit(source_rel, repo_root=repo, expected_sha256=expected_sha)
        source_exists = source_audit["source_artifact_exists"] == "true"
        source_under_repo = source_audit["source_artifact_path_under_repo"] == "true"
        seed_codes = seed_blockers.get((family, paper_i_case_id, drive_amplitude), [])
        staged_marker_present = _has_staged_seed_marker(
            first.get("seed_is_staged_or_recovery"),
            first.get("paper_ii_seed_policy"),
            seed_lock.get("selected_static_seed_source"),
            source_rel,
        ) or _clean(first.get("seed_is_staged_or_recovery")).lower() == "true"
        candidate_path, candidate_sha, candidate_source = _inventory_candidate_from_source(
            source_rel=source_rel,
            source_exists=source_exists,
            source_path_under_repo=source_under_repo,
            source_sha256=source_audit["source_artifact_actual_sha256"],
            staged_marker_present=staged_marker_present,
        )
        repair_status = _seed_repair_status(
            current_status=_clean(first.get("static_artifact_status")),
            seed_blocker_codes=seed_codes,
            source_exists=source_exists,
            source_path_under_repo=source_under_repo,
            staged_marker_present=staged_marker_present,
            candidate_path=candidate_path,
        )
        comparator_ids = sorted(
            {_clean(row.get("comparator_id")) for row in group_rows if _clean(row.get("comparator_id"))},
            key=lambda value: comparator_order.get(value, len(comparator_order)),
        )
        notes = [
            "inventory_only_no_seed_repair_performed",
            f"collapsed_from_{len(group_rows)}_comparator_rows",
        ]
        if seed_codes:
            notes.append("inherits_seed_blockers=" + ",".join(seed_codes))
        slot = {
            "slot_id": _seed_slot_id(
                family=family,
                paper_i_case_id=paper_i_case_id,
                drive_amplitude=drive_amplitude,
                n_ph_max=n_ph_max,
            ),
            "family": family,
            "paper_i_case_id": paper_i_case_id,
            "paper_ii_case_id": _clean(first.get("paper_ii_case_id")),
            "drive_amplitude": drive_amplitude,
            "row_count": str(len(group_rows)),
            "comparator_ids": ";".join(comparator_ids),
            "current_static_artifact_path": _clean(first.get("static_artifact_path")),
            "current_static_artifact_sha256": _clean(first.get("static_artifact_sha256")),
            "current_static_artifact_status": _clean(first.get("static_artifact_status")),
            "seed_is_local": _clean(first.get("seed_is_local")),
            "seed_is_staged_or_recovery": _clean(first.get("seed_is_staged_or_recovery")),
            "latest_phase3_source_artifact_missing_locally": _clean(first.get("latest_phase3_source_artifact_missing_locally")),
            "paper_ii_seed_policy": _clean(first.get("paper_ii_seed_policy")),
            "selected_static_seed_source": _clean(seed_lock.get("selected_static_seed_source")),
            "same_seed_comparator_group_id": _clean(seed_lock.get("same_seed_comparator_group_id")),
            "source_artifact_json": source_rel,
            **source_audit,
            "inherited_seed_blocker_codes": ";".join(seed_codes),
            "candidate_canonical_static_artifact_path": candidate_path,
            "candidate_canonical_static_artifact_sha256": candidate_sha,
            "candidate_canonical_static_artifact_source": candidate_source,
            "repair_status": repair_status,
            "notes": ";".join(notes),
        }
        slots.append(slot)

    status_counter = Counter(slot["repair_status"] for slot in slots)
    seed_blocker_counter = Counter(
        code for slot in slots for code in _split_codes(slot.get("inherited_seed_blocker_codes"))
    )
    scope_decision_rows = [
        {
            "record_id": _clean(row.get("record_id")),
            "family": _clean(row.get("family")),
            "case_id": _clean(row.get("case_id")),
            "paper_i_case_id": _clean(row.get("paper_i_case_id")),
            "blocker_codes": _clean(row.get("blocker_codes")),
            "excluded_from_unique_seed_slot_denominator": True,
        }
        for row in scope_rows
    ]
    return {
        "schema_version": SEED_REPAIR_INVENTORY_SCHEMA_VERSION,
        "profile": profile,
        "matrix_batch_id": matrix_batch_id,
        "generated_utc": _now_utc(),
        "no_submit": True,
        "no_submit_policy": NO_SUBMIT_POLICY,
        "local_only": True,
        "submits_chtc": False,
        "runs_realtime_dynamics": False,
        "repairs_seed_artifacts": False,
        "changes_seed_paths": False,
        "unique_seed_slot_count": len(slots),
        "expected_unique_seed_slot_count": expected_unique_seed_slots,
        "scope_decision_row_count": len(scope_decision_rows),
        "repair_status_counts": dict(sorted(status_counter.items())),
        "seed_blocker_code_counts": dict(sorted(seed_blocker_counter.items())),
        "slots": slots,
        "scope_decision_rows": scope_decision_rows,
    }


def _build_record_row(
    *,
    family: str,
    paper_i_case_id: str,
    drive_amplitude: float,
    paper_ii_case: Mapping[str, Any] | None,
    comparator: ComparatorSpec,
    repo_root: Path,
    settings_by_class: Mapping[str, Mapping[str, Any]],
    class_settings_payload: Mapping[str, Any],
    profile: str = PROFILE,
    matrix_batch_id: str = MATRIX_BATCH_ID,
    n_ph_max: int = N_PH_MAX,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    amp_label = _amplitude_label(drive_amplitude)
    n_ph = _validate_n_ph_max(n_ph_max)
    record_id = f"paper_iii_nph{n_ph}_{_safe_record_fragment(paper_i_case_id)}_{amp_label}_{comparator.comparator_id}"
    metadata = paper_ii_case.get("metadata") if isinstance(paper_ii_case, Mapping) and isinstance(paper_ii_case.get("metadata"), Mapping) else {}
    drive = metadata.get("drive") if isinstance(metadata.get("drive"), Mapping) else {}
    artifact_fields, seed_blockers = _artifact_status_and_seed_blockers(case=paper_ii_case, repo_root=repo_root)
    strict_status, settings_source, strict_blockers = _class_setting_status(
        case=paper_ii_case,
        settings_by_class=settings_by_class,
        class_settings_payload=class_settings_payload,
    )
    comparator_status, comparator_blockers = _comparator_status_and_blockers(family, comparator)
    deferred_reason = table_i_cases.table_i_deferred_case_reason(family, paper_i_case_id, table_i_cases.TABLE_I_STANDARD_PROFILE)
    static_blockers = ["static.table_i_case_deferred_at_head"] if deferred_reason else []
    blockers = _dedupe_codes([*seed_blockers, *strict_blockers, *comparator_blockers, *static_blockers])
    expected_status = "blocked" if blockers else "ready"
    measurement_proxy_status = (
        "blocked_unvalidated"
        if any(code.startswith("qse.") for code in blockers)
        else "planned_report_only"
    )
    comparator_blocker_codes = _dedupe_codes([*comparator_blockers, *static_blockers])
    drive_pattern = _clean(drive.get("pattern")) or "unknown"
    try:
        omega = float(drive.get("omega"))
    except Exception:
        omega = 1.0
    time_sampling = _clean(drive.get("time_sampling")) or "unknown"
    row: dict[str, str] = {
        "record_id": record_id,
        "queue": "full",
        "mode": MODE_COMPATIBILITY_AUDIT_ONLY,
        "matrix_batch_id": matrix_batch_id,
        "profile": profile,
        "row_kind": "comparator_row",
        "family": family,
        "case_id": paper_i_case_id,
        "paper_i_case_id": paper_i_case_id,
        "paper_ii_case_id": _clean(paper_ii_case.get("case_id")) if isinstance(paper_ii_case, Mapping) else "",
        "n_ph_max": str(n_ph),
        "target_excited_roots": str(TARGET_EXCITED_ROOTS),
        "static_artifact_path": _clean(artifact_fields["static_artifact_path"]),
        "static_artifact_sha256": _clean(artifact_fields["static_artifact_sha256"]),
        "static_artifact_status": _clean(artifact_fields["static_artifact_status"]),
        "seed_source_ledger": _clean(artifact_fields["seed_source_ledger"]),
        "seed_is_local": _bool_text(bool(artifact_fields["seed_is_local"])),
        "seed_is_staged_or_recovery": _bool_text(bool(artifact_fields["seed_is_staged_or_recovery"])),
        "latest_phase3_source_artifact_missing_locally": _bool_text(
            bool(artifact_fields["latest_phase3_source_artifact_missing_locally"])
        ),
        "paper_ii_seed_policy": _clean(artifact_fields["paper_ii_seed_policy"]),
        "operator_support_source": _operator_support_source(family),
        "qse_basis_source": comparator.qse_basis_source,
        "comparator_id": comparator.comparator_id,
        "comparator_family": comparator.comparator_family,
        "comparator_status": comparator_status,
        "comparator_blockers": ";".join(comparator_blocker_codes),
        "qse_matrix_measurement_proxy_status": measurement_proxy_status,
        "ed_same_cutoff_status": "diagnostic_report_only_planned",
        "ed_higher_cutoff_status": "diagnostic_report_only_deferred_nph1_first",
        "dynamics_class": _clean(paper_ii_case.get("tuning_class")) if isinstance(paper_ii_case, Mapping) else "",
        "dynamics_settings_source": settings_source,
        "t_final": _clean(paper_ii_case.get("t_final")) if isinstance(paper_ii_case, Mapping) else "",
        "num_times": _clean(paper_ii_case.get("num_times")) if isinstance(paper_ii_case, Mapping) else "",
        "time_grid_convention": "paper_ii_num_times_samples",
        "drive_enabled": _bool_text(bool(metadata.get("enable_drive", drive.get("enable_drive", True)))),
        "drive_amplitude": f"{float(drive_amplitude):g}",
        "drive_frequency_grid_id": f"paper_ii_{drive_pattern}_omega{_safe_record_fragment(f'{omega:g}')}_{time_sampling}",
        "paper_ii_drive_pattern": drive_pattern,
        "controller_exact_input_mode": "off",
        "diagnostic_exact_reference_mode": "benchmark_exact",
        "uses_reference_for_decision": "false",
        "uses_future_exact_forecast_for_decision": "false",
        "uses_statevector_as_ideal_observable_estimator": "true",
        "exact_decision_checkpoints": "0",
        "strict_measurement_oracle_certified": "true",
        "qpu_faithful_decisions_passed": "true",
        "strict_policy_status": strict_status,
        "expected_status": expected_status,
        "blocker_count": str(len(blockers)),
        "blocker_codes": ";".join(blockers),
        "no_submit": "true",
        "no_submit_policy": NO_SUBMIT_POLICY,
        "require_progress_json": "false",
        "require_partial_payload_json": "false",
        "require_science_benchmark": "false",
        "notes": comparator.notes if deferred_reason is None else f"{comparator.notes} Deferred Table-I case: {deferred_reason}",
    }
    blocker_rows = [
        {
            "record_id": record_id,
            "family": family,
            "case_id": paper_i_case_id,
            "paper_i_case_id": paper_i_case_id,
            "drive_amplitude": f"{float(drive_amplitude):g}",
            "comparator_id": comparator.comparator_id,
            "code": code,
            "source": _blocker_source(code),
            "severity": "blocker",
            "detail": _blocker_detail(code) if code != "static.table_i_case_deferred_at_head" else (deferred_reason or _blocker_detail(code)),
            "no_submit": "true",
        }
        for code in blockers
    ]
    return row, blocker_rows


def _build_scope_decision_row(
    *,
    profile: str = PROFILE,
    matrix_batch_id: str = MATRIX_BATCH_ID,
    n_ph_max: int = N_PH_MAX,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    n_ph = _validate_n_ph_max(n_ph_max)
    record_id = f"paper_iii_nph{n_ph}_molecular_restricted_closed_shell_scope_decision"
    blockers = ["scope.registry_only_family_not_table_i_declared", "scope.requires_user_decision_for_paper_iii"]
    row = {field: "" for field in RECORD_FIELDNAMES}
    row.update(
        {
            "record_id": record_id,
            "queue": "scope",
            "mode": MODE_COMPATIBILITY_AUDIT_ONLY,
            "matrix_batch_id": matrix_batch_id,
            "profile": profile,
            "row_kind": "scope_decision",
            "family": REGISTRY_SCOPE_FAMILY,
            "case_id": "registry_only_molecular_restricted_closed_shell",
            "paper_i_case_id": "not_table_i_declared",
            "n_ph_max": str(n_ph),
            "target_excited_roots": str(TARGET_EXCITED_ROOTS),
            "operator_support_source": _operator_support_source(REGISTRY_SCOPE_FAMILY),
            "qse_basis_source": "scope_decision_only",
            "comparator_id": "scope_decision",
            "comparator_family": "scope_decision",
            "comparator_status": "not_applicable_scope_decision",
            "qse_matrix_measurement_proxy_status": "not_applicable_scope_decision",
            "ed_same_cutoff_status": "not_applicable_scope_decision",
            "ed_higher_cutoff_status": "not_applicable_scope_decision",
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "uses_reference_for_decision": "false",
            "uses_future_exact_forecast_for_decision": "false",
            "uses_statevector_as_ideal_observable_estimator": "true",
            "exact_decision_checkpoints": "0",
            "strict_measurement_oracle_certified": "true",
            "qpu_faithful_decisions_passed": "true",
            "strict_policy_status": "not_applicable_scope_decision",
            "expected_status": "blocked",
            "blocker_count": str(len(blockers)),
            "blocker_codes": ";".join(blockers),
            "no_submit": "true",
            "no_submit_policy": NO_SUBMIT_POLICY,
            "require_progress_json": "false",
            "require_partial_payload_json": "false",
            "require_science_benchmark": "false",
            "notes": "Registry-only molecular closed-shell family is wired but not declared as the Paper-III Table-I molecular_vibronic_h2 row; keep explicit until scoped.",
        }
    )
    blocker_rows = [
        {
            "record_id": record_id,
            "family": REGISTRY_SCOPE_FAMILY,
            "case_id": row["case_id"],
            "paper_i_case_id": row["paper_i_case_id"],
            "drive_amplitude": "",
            "comparator_id": "scope_decision",
            "code": code,
            "source": _blocker_source(code),
            "severity": "blocker",
            "detail": _blocker_detail(code),
            "no_submit": "true",
        }
        for code in blockers
    ]
    return row, blocker_rows


def build_matrix(
    *,
    repo_root: str | Path = REPO_ROOT,
    profile: str = PROFILE,
    matrix_batch_id: str = MATRIX_BATCH_ID,
    n_ph_max: int = N_PH_MAX,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    repo = Path(repo_root).resolve()
    n_ph = _validate_n_ph_max(n_ph_max)
    paper_ii_cases_path = repo / PAPER_II_CASES_REL
    class_settings_path = repo / PAPER_II_CLASS_SETTINGS_REL
    paper_ii_cases_payload = _load_json(paper_ii_cases_path)
    class_settings_payload = _load_json(class_settings_path)
    declared_by_family = table_i_cases.table_i_declared_case_ids_by_family(table_i_cases.TABLE_I_STANDARD_PROFILE)
    registry_families = tuple(problem_registry.available_problem_keys())
    paper_ii_by_key = _paper_ii_cases_by_family_and_amplitude(paper_ii_cases_payload)
    drive_amplitudes = _drive_amplitudes(paper_ii_cases_payload)
    settings_by_class = _setting_by_tuning_class(class_settings_payload)

    rows: list[dict[str, str]] = []
    blockers: list[dict[str, str]] = []
    for family, case_ids in declared_by_family.items():
        for paper_i_case_id in case_ids:
            for amplitude in drive_amplitudes:
                paper_ii_case = paper_ii_by_key.get((family, amplitude))
                for comparator in COMPARATOR_SPECS:
                    row, row_blockers = _build_record_row(
                        family=str(family),
                        paper_i_case_id=str(paper_i_case_id),
                        drive_amplitude=float(amplitude),
                        paper_ii_case=paper_ii_case,
                        comparator=comparator,
                        repo_root=repo,
                        settings_by_class=settings_by_class,
                        class_settings_payload=class_settings_payload,
                        profile=profile,
                        matrix_batch_id=matrix_batch_id,
                        n_ph_max=n_ph,
                    )
                    rows.append(row)
                    blockers.extend(row_blockers)
    if REGISTRY_SCOPE_FAMILY in registry_families and REGISTRY_SCOPE_FAMILY not in declared_by_family:
        scope_row, scope_blockers = _build_scope_decision_row(
            profile=profile,
            matrix_batch_id=matrix_batch_id,
            n_ph_max=n_ph,
        )
        rows.append(scope_row)
        blockers.extend(scope_blockers)

    comparator_rows = [row for row in rows if row.get("row_kind") == "comparator_row"]
    families = sorted({row["family"] for row in comparator_rows})
    cases = sorted({(row["family"], row["paper_i_case_id"]) for row in comparator_rows})
    source_files = {
        "generator": _source_file_record(Path(__file__).resolve(), repo_root=repo),
        "table_i_canonical_cases": _source_file_record(Path(table_i_cases.__file__).resolve(), repo_root=repo),
        "problem_registry": _source_file_record(Path(problem_registry.__file__).resolve(), repo_root=repo),
        "paper_ii_cases": _source_file_record(paper_ii_cases_path, repo_root=repo),
        "paper_ii_class_settings": _source_file_record(class_settings_path, repo_root=repo),
    }
    blocker_counter = Counter(row["code"] for row in blockers)
    status_counter = Counter(row["expected_status"] for row in rows)
    comparator_status_counter = Counter(row["comparator_status"] for row in comparator_rows)
    seed_status_counter = Counter(row["static_artifact_status"] for row in comparator_rows)
    strict_status_counter = Counter(row["strict_policy_status"] for row in rows)
    manifest = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "records_schema_version": RECORDS_SCHEMA_VERSION,
        "profile": profile,
        "matrix_batch_id": matrix_batch_id,
        "generated_utc": _now_utc(),
        "no_submit": True,
        "no_submit_policy": NO_SUBMIT_POLICY,
        "no_submit_assertion": True,
        "local_only": True,
        "runs_realtime_dynamics": False,
        "submits_chtc": False,
        "changes_adapt_route_defaults": False,
        "changes_mclachlan_controller_defaults": False,
        "exact_ed_qse_diagnostics_role": "report_only_never_controller_decision_input",
        "strict_policy": {
            "controller_exact_input_mode": "off",
            "diagnostic_exact_reference_mode": "benchmark_exact",
            "uses_reference_for_decision": False,
            "uses_future_exact_forecast_for_decision": False,
            "exact_decision_checkpoints": 0,
            "strict_measurement_oracle_certified": True,
            "qpu_faithful_decisions_passed": True,
        },
        "target_excited_roots": TARGET_EXCITED_ROOTS,
        "n_ph_max": n_ph,
        "expected_counts": {
            "table_i_declared_families": len(declared_by_family),
            "table_i_declared_cases": _case_count(declared_by_family),
            "paper_ii_source_cases": int(paper_ii_cases_payload.get("case_count", len(paper_ii_cases_payload.get("cases") or []))),
            "paper_ii_drive_amplitudes_per_family": len(drive_amplitudes),
            "dynamics_classes": len(settings_by_class),
            "comparators": len(COMPARATOR_SPECS),
            "comparator_rows": _case_count(declared_by_family) * len(drive_amplitudes) * len(COMPARATOR_SPECS),
            "registry_scope_rows": 1 if any(row.get("row_kind") == "scope_decision" for row in rows) else 0,
            "records_total": len(rows),
        },
        "actual_counts": {
            "record_count": len(rows),
            "comparator_row_count": len(comparator_rows),
            "scope_decision_row_count": sum(1 for row in rows if row.get("row_kind") == "scope_decision"),
            "blocker_count": len(blockers),
            "rows_with_blockers": sum(1 for row in rows if int(row.get("blocker_count") or 0) > 0),
            "families_represented": len(families),
            "cases_represented": len(cases),
        },
        "declared_table_i_families": list(declared_by_family.keys()),
        "declared_table_i_cases_by_family": {str(k): list(v) for k, v in declared_by_family.items()},
        "registry_wired_families": list(registry_families),
        "registry_only_scope_families": [REGISTRY_SCOPE_FAMILY] if any(row.get("family") == REGISTRY_SCOPE_FAMILY for row in rows if row.get("row_kind") == "scope_decision") else [],
        "paper_ii_drive_amplitudes": [float(value) for value in drive_amplitudes],
        "paper_ii_dynamics_classes": sorted(settings_by_class),
        "comparator_specs": [_jsonable(comparator.__dict__) for comparator in COMPARATOR_SPECS],
        "source_files": source_files,
        "status_counts": dict(status_counter),
        "comparator_status_counts": dict(comparator_status_counter),
        "seed_artifact_status_counts": dict(seed_status_counter),
        "strict_policy_status_counts": dict(strict_status_counter),
        "blocker_code_counts": dict(sorted(blocker_counter.items())),
    }
    return rows, blockers, _jsonable(manifest)


def _write_tsv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _clean(row.get(field)) for field in fieldnames})


def _write_id_lists(output_dir: Path, rows: Sequence[Mapping[str, str]]) -> tuple[list[str], list[str]]:
    full_ids = [row["record_id"] for row in rows]
    smoke_ids = [
        row["record_id"]
        for row in rows
        if row.get("row_kind") == "comparator_row"
        and row.get("comparator_id") == SMOKE_COMPARATOR_ID
        and row.get("drive_amplitude") == "0.2"
    ]
    header = f"# no_submit=true; policy={NO_SUBMIT_POLICY}\n"
    (output_dir / "full_record_ids.txt").write_text(header + "\n".join(full_ids) + "\n", encoding="utf-8")
    (output_dir / "smoke_record_ids.txt").write_text(header + "\n".join(smoke_ids) + "\n", encoding="utf-8")
    return full_ids, smoke_ids


def _summary_markdown(manifest: Mapping[str, Any]) -> str:
    n_ph = manifest.get("n_ph_max", N_PH_MAX)
    expected = manifest.get("expected_counts", {}) if isinstance(manifest.get("expected_counts"), Mapping) else {}
    actual = manifest.get("actual_counts", {}) if isinstance(manifest.get("actual_counts"), Mapping) else {}
    blocker_counts = manifest.get("blocker_code_counts", {}) if isinstance(manifest.get("blocker_code_counts"), Mapping) else {}
    top_blockers = sorted(blocker_counts.items(), key=lambda item: (-int(item[1]), str(item[0])))[:12]
    top_blocker_lines = "\n".join(f"- `{code}`: {count}" for code, count in top_blockers) or "- None"
    return f"""# Paper III n_ph={n_ph} compatibility audit matrix

Profile: `{manifest['profile']}`  
Batch: `{manifest['matrix_batch_id']}`  
Generated UTC: `{manifest['generated_utc']}`

## No-submit policy

**No submit.** This artifact is local audit/planning metadata only: `{manifest['no_submit_policy']}`. It does not submit CHTC jobs, run realtime dynamics, change ADAPT route defaults, or change McLachlan/controller defaults. Exact ED/QSE diagnostics are report-only and never controller decision inputs.

## Coverage counts

- Table-I declared families: {expected.get('table_i_declared_families')}
- Table-I declared cases: {expected.get('table_i_declared_cases')}
- Paper-II source dynamics cases: {expected.get('paper_ii_source_cases')}
- Paper-II drive amplitudes per family: {expected.get('paper_ii_drive_amplitudes_per_family')}
- Comparator specs: {expected.get('comparators')}
- Comparator rows: {actual.get('comparator_row_count')} / expected {expected.get('comparator_rows')}
- Registry-only scope rows: {actual.get('scope_decision_row_count')}
- Total records: {actual.get('record_count')}
- Blocker entries: {actual.get('blocker_count')}
- Rows with blockers: {actual.get('rows_with_blockers')}

## Strict policy

Rows carry `controller_exact_input_mode=off`, `diagnostic_exact_reference_mode=benchmark_exact`, `uses_reference_for_decision=false`, `uses_future_exact_forecast_for_decision=false`, `exact_decision_checkpoints=0`, `strict_measurement_oracle_certified=true`, and `qpu_faithful_decisions_passed=true` unless the row is the explicit registry scope decision. This matrix is a QPU-faithfulness audit scaffold, not a controller decision backend.

## Top blockers

{top_blocker_lines}

## Decision

Stop after local generation, preflight, validation, focused tests, and scoreboard recording. Do not move to smoke/full CHTC submission until seed provenance is repaired, missing comparator support is implemented or deliberately scoped out, six-root QSE behavior is audited, and no-submit local validation remains green.
"""


def _seed_repair_summary_markdown(inventory: Mapping[str, Any]) -> str:
    status_counts = inventory.get("repair_status_counts", {}) if isinstance(inventory.get("repair_status_counts"), Mapping) else {}
    seed_blocker_counts = inventory.get("seed_blocker_code_counts", {}) if isinstance(inventory.get("seed_blocker_code_counts"), Mapping) else {}
    status_lines = "\n".join(f"- `{status}`: {count}" for status, count in sorted(status_counts.items())) or "- None"
    blocker_lines = "\n".join(f"- `{code}`: {count}" for code, count in sorted(seed_blocker_counts.items())) or "- None"
    return f"""# Paper III seed-provenance repair inventory

Profile: `{inventory['profile']}`
Batch: `{inventory['matrix_batch_id']}`
Generated UTC: `{inventory['generated_utc']}`

## No-submit / no-repair assertion

**No submit.** This is local-only audit metadata. It does not submit CHTC jobs, run realtime dynamics, implement QSE/comparators, copy artifacts, change seed paths, change ADAPT defaults, or change McLachlan/controller defaults. **No seed repair** was performed.

## Unique seed-slot denominator

- Unique comparator seed slots: {inventory.get('unique_seed_slot_count')} / expected {inventory.get('expected_unique_seed_slot_count')}
- Denominator key: `(family, paper_i_case_id, drive_amplitude)`
- Registry-only scope rows: {inventory.get('scope_decision_row_count')} visible and explicitly excluded from the 36-slot denominator

## Repair status counts

{status_lines}

## Inherited seed blocker counts

{blocker_lines}

## Stop policy

Use this inventory to plan a future seed-provenance repair slice only. Do not repair seeds, copy artifacts, submit CHTC, or launch dynamics from this compatibility-audit matrix.
"""


def _write_seed_repair_artifacts(*, output_dir: Path, inventory: Mapping[str, Any]) -> None:
    slots = inventory.get("slots") if isinstance(inventory.get("slots"), list) else []
    _write_tsv(output_dir / "seed_repair_inventory.tsv", slots, SEED_REPAIR_FIELDNAMES)
    (output_dir / "seed_repair_inventory.json").write_text(
        json.dumps(_jsonable(inventory), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "seed_repair_summary.md").write_text(_seed_repair_summary_markdown(inventory), encoding="utf-8")


def write_matrix_artifacts(
    *,
    rows: Sequence[Mapping[str, str]],
    blockers: Sequence[Mapping[str, str]],
    manifest: Mapping[str, Any],
    output_dir: str | Path,
    seed_repair_inventory: Mapping[str, Any] | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = _clean(manifest.get("profile")) or PROFILE
    matrix_batch_id = _clean(manifest.get("matrix_batch_id")) or MATRIX_BATCH_ID
    n_ph_max = _validate_n_ph_max(int(manifest.get("n_ph_max", N_PH_MAX)))
    _write_tsv(out / "records.tsv", rows, RECORD_FIELDNAMES)
    full_ids, smoke_ids = _write_id_lists(out, rows)
    _write_tsv(out / "blockers.tsv", blockers, BLOCKER_FIELDNAMES)
    if seed_repair_inventory is None:
        repo = Path(repo_root).resolve()
        seed_repair_inventory = build_seed_repair_inventory(
            rows=rows,
            blockers=blockers,
            paper_ii_cases_payload=_load_json(repo / PAPER_II_CASES_REL),
            repo_root=repo,
            profile=profile,
            matrix_batch_id=matrix_batch_id,
            n_ph_max=n_ph_max,
        )
    _write_seed_repair_artifacts(output_dir=out, inventory=seed_repair_inventory)
    blockers_payload = {
        "schema_version": BLOCKERS_SCHEMA_VERSION,
        "profile": profile,
        "matrix_batch_id": matrix_batch_id,
        "no_submit": True,
        "no_submit_policy": NO_SUBMIT_POLICY,
        "blocker_count": len(blockers),
        "blockers": list(blockers),
    }
    manifest_payload = dict(manifest)
    manifest_payload["artifact_files"] = {
        "manifest_json": "manifest.json",
        "records_tsv": "records.tsv",
        "full_record_ids": "full_record_ids.txt",
        "smoke_record_ids": "smoke_record_ids.txt",
        "blockers_tsv": "blockers.tsv",
        "blockers_json": "blockers.json",
        "compatibility_audit_summary_md": "compatibility_audit_summary.md",
        "seed_repair_inventory_tsv": "seed_repair_inventory.tsv",
        "seed_repair_inventory_json": "seed_repair_inventory.json",
        "seed_repair_summary_md": "seed_repair_summary.md",
    }
    manifest_payload["full_record_count"] = len(full_ids)
    manifest_payload["smoke_record_count"] = len(smoke_ids)
    manifest_payload["seed_repair_inventory_schema_version"] = SEED_REPAIR_INVENTORY_SCHEMA_VERSION
    manifest_payload["seed_repair_unique_slot_count"] = seed_repair_inventory.get("unique_seed_slot_count")
    manifest_payload["seed_repair_expected_unique_slot_count"] = seed_repair_inventory.get("expected_unique_seed_slot_count")
    manifest_payload["seed_repair_scope_decision_row_count"] = seed_repair_inventory.get("scope_decision_row_count")
    manifest_payload["seed_repair_status_counts"] = seed_repair_inventory.get("repair_status_counts", {})
    manifest_payload["seed_repair_seed_blocker_code_counts"] = seed_repair_inventory.get("seed_blocker_code_counts", {})
    (out / "manifest.json").write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "blockers.json").write_text(json.dumps(_jsonable(blockers_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "compatibility_audit_summary.md").write_text(_summary_markdown(manifest_payload), encoding="utf-8")
    return {
        "ok": True,
        "output_dir": str(out),
        "profile": profile,
        "matrix_batch_id": matrix_batch_id,
        "record_count": len(rows),
        "full_record_count": len(full_ids),
        "smoke_record_count": len(smoke_ids),
        "blocker_count": len(blockers),
        "seed_repair_unique_slot_count": seed_repair_inventory.get("unique_seed_slot_count"),
        "seed_repair_expected_unique_slot_count": seed_repair_inventory.get("expected_unique_seed_slot_count"),
        "seed_repair_scope_decision_row_count": seed_repair_inventory.get("scope_decision_row_count"),
        "seed_repair_status_counts": seed_repair_inventory.get("repair_status_counts", {}),
        "no_submit": True,
        "artifact_files": manifest_payload["artifact_files"],
    }


def generate_compatibility_matrix(
    *,
    output_dir: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    profile: str | None = PROFILE,
    no_submit: bool = True,
    n_ph_max: int = N_PH_MAX,
) -> dict[str, Any]:
    n_ph = _validate_n_ph_max(n_ph_max)
    resolved_profile = _resolve_profile(profile=profile, n_ph_max=n_ph)
    matrix_batch_id = _matrix_batch_id_for_n_ph(n_ph)
    if not no_submit:
        raise ValueError("this generator is local audit-only; pass --no-submit")
    rows, blockers, manifest = build_matrix(
        repo_root=repo_root,
        profile=resolved_profile,
        matrix_batch_id=matrix_batch_id,
        n_ph_max=n_ph,
    )
    return write_matrix_artifacts(
        rows=rows,
        blockers=blockers,
        manifest=manifest,
        output_dir=output_dir if output_dir is not None else _default_output_dir_for_n_ph(n_ph),
        repo_root=repo_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate local-only Paper III n_ph compatibility matrix artifacts.")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--n-ph-max", type=int, default=N_PH_MAX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Required explicit assertion that this command does not submit CHTC jobs or launch dynamics.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    result = generate_compatibility_matrix(
        output_dir=args.output_dir,
        repo_root=args.repo_root,
        profile=args.profile,
        no_submit=bool(args.no_submit),
        n_ph_max=int(args.n_ph_max),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


__all__ = [
    "BLOCKERS_SCHEMA_VERSION",
    "COMPARATOR_SPECS",
    "DEFAULT_OUTPUT_DIR",
    "MATRIX_BATCH_ID",
    "MATRIX_SCHEMA_VERSION",
    "MODE_COMPATIBILITY_AUDIT_ONLY",
    "NO_SUBMIT_POLICY",
    "PROFILE",
    "RECORD_FIELDNAMES",
    "SEED_REPAIR_ALLOWED_STATUSES",
    "SEED_REPAIR_FIELDNAMES",
    "SEED_REPAIR_INVENTORY_SCHEMA_VERSION",
    "build_matrix",
    "build_seed_repair_inventory",
    "generate_compatibility_matrix",
    "write_matrix_artifacts",
    "_default_output_dir_for_n_ph",
    "_matrix_batch_id_for_n_ph",
    "_profile_for_n_ph",
]


if __name__ == "__main__":
    raise SystemExit(main())
