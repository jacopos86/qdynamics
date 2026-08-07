#!/usr/bin/env python3
"""Closed contract for the three nph7 historical-mean global-singleton repairs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_"
    "r50_20260802_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_v2"
)
RUN_CLASS = "diagnostic"
TARGET_HORIZON = 50
SOURCE_HORIZON = 50
EXECUTION_MODE = "fresh_0_to_50"

BASE_BUNDLE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_global_singleton_insertion12_v1/"
    "ra_repair_global_singleton_stationary_all_phase_insertion_v1"
)
BASE_BUNDLE_MANIFEST_FILE_SHA256 = (
    "1ebc97c0892e8997100db16e9705437962e6514fbde1404d40746dae6566fa37"
)
BASE_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "cfe4b8500ca956718dd73c77fc0d389648259ed01e57b4a638da5056a48594f6"
)
BASE_SOURCE_LOCKS_FILE_SHA256 = (
    "297408fc4a47d337810414c5bf68d81ad07b11ddab99b2829bd317c7278fe28b"
)
BASE_SOURCE_LOCKS_CANONICAL_SHA256 = (
    "a50c816dfe1178b13a33eaf4ffab15bd89cfa1c525e9b8e608b065354187d28d"
)
SOURCE_BUNDLE_ID = "ra_repair_global_singleton_stationary_all_phase_insertion_v1"
SOURCE_BUNDLE_RELATIVE = BASE_BUNDLE_RELATIVE
SOURCE_BUNDLE_MANIFEST_FILE_SHA256 = BASE_BUNDLE_MANIFEST_FILE_SHA256
SOURCE_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    BASE_BUNDLE_MANIFEST_CANONICAL_SHA256
)
SOURCE_LOCKS_FILE_SHA256 = BASE_SOURCE_LOCKS_FILE_SHA256
SOURCE_LOCKS_CANONICAL_SHA256 = BASE_SOURCE_LOCKS_CANONICAL_SHA256

RUNTIME_REFERENCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260801_v4_chtc"
)
RUNTIME_REFERENCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "6c5666fb90d2840625b159f3fa45c06501093c0e5af44dbd14a3329d858f3454"
)
RUNTIME_REFERENCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "5bfa293ebcb467fb69b95b27dc675465887e4f846a9e94c821e4e56ae0906d96"
)
RUNTIME_SOURCE_ARCHIVE_FILE_SHA256 = (
    "f8b42ea0411e9f3f763d79bcddb7ab39b1873550451e5a5e73cd53b88b07ec26"
)
RUNTIME_SOURCE_ARCHIVE_MANIFEST_FILE_SHA256 = (
    "dc856a808bd97719ccaacc6d6bf44f73193d986b8bfa9a9af8ade2fab22693d2"
)
RUNTIME_SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256 = (
    "6c6e7421b02bead99676fbd3a991ff8de8a81b840f7772a68534a25e85960e9e"
)
RUNTIME_SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "24fb4e5e78231ca1bebf907449f8993d2e7cd6cf24b8473d5d2968af3fe34f76"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "1d2e923e68dbbec924fe80c5485b567dc97b59b3d9ed6c78573e9186642265f5"
)
SOURCE_PACKAGE_RELATIVE = RUNTIME_REFERENCE_PACKAGE_RELATIVE
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    RUNTIME_REFERENCE_PACKAGE_MANIFEST_FILE_SHA256
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    RUNTIME_REFERENCE_PACKAGE_MANIFEST_CANONICAL_SHA256
)

ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "all_phase_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
CANDIDATE_ADAPTER_ID = (
    "paper_i_ra_adapt_global_single_pauli_word_candidate_adapter_v1"
)
ALGORITHM_ID = "paper_i_ra_adapt_global_singleton_plateau_commutation_v1"
ROUTE_ID = "ra_global_singleton_plateau_commutation"
ROUTE_CONTRACT_SHA256 = (
    "69af64db5bbaf5b811685b8353b82b748dc13d16306e4c08ddfe5ffde07f301b"
)
PARENT_ROUTE_CONTRACT_SHA256 = (
    "aa669d7f0c3621d9ddf7f8595f96333c56b536c8fc79547607e76d8d91d4b6ff"
)
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__global_guarded_singleton_phase_i__"
    "identity_phase_ii__stationary_source_response_v1__"
    "all_phase_resource_weighting_v1"
)
PHASE_I_CANDIDATE_SUPPLY = "global_guarded_singleton_pool_v1"
PHASE_I_CANDIDATE_VISIBILITY = "all_executable_candidates_v1"
PHASE_II_CANDIDATE_EXPOSURE = "identity_on_retained_singletons_v1"
PHASE_I_SHORTLIST_SIZE = 24
PHASE_II_SHORTLIST_SIZE = 12
PHASE_III_ADMISSION_CARDINALITY = 1

PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD = 1.0e-4
PLATEAU_COMPARISON = "marginal_to_prior_mean_strictly_below_v2"
PLATEAU_TRIGGER = (
    "immediately_preceding_marginal_over_prior_mean_"
    "accepted_post_full_refit_energy_decrease_v2"
)
PLATEAU_CALIBRATION = "source_locked_counterfactual_trigger_replay_v2"

REGIME_ROWS: tuple[tuple[str, int], ...] = (
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)

# Preserve the prior global-singleton operational envelopes.  The large
# executable singleton inventories are materially heavier than ordinary
# singleton routes; any later resource reduction must be history-backed.
RESOURCE_ENVELOPES = {
    7: {
        "request_cpus": 4,
        "request_memory_mb": 49_152,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
    },
}

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "package_manifest_v2"
)
PROTOCOL_BUNDLE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "protocol_bundle_manifest_v2"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "source_archive_manifest_v2"
)
SOURCE_LOCK_AUDIT_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "source_lock_audit_v2"
)
JOB_SCHEMA = "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_job_v2"
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "execution_authorization_v2"
)
RUN_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_historical_mean_global_singleton_plateau3_nph7_r50_"
    "run_manifest_v2"
)

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
    "execute_authorized_job.sh",
    "submit.sub.in",
    "source_patch.diff",
)
GENERATED_PATHS = (
    "protocols",
    "jobs",
    "source",
    "source_locks_snapshot.json",
    "protocol_bundle_manifest.json",
    "source_lock_audit.json",
    "execution_plan.json",
    "queue.tsv",
    "package_manifest.json",
)

ALLOWED_SOURCE_TO_TARGET_DIFF_PATHS = frozenset(
    {
        ("bundle_id",),
        ("bundle_manifest_sha256",),
        ("bundle_materialization", "bundle_id"),
        ("bundle_materialization", "bundle_manifest_sha256"),
        ("bundle_materialization", "cell_id"),
        ("bundle_materialization", "sha256"),
        ("bundle_materialization", "source_lock_refs_sha256"),
        ("bundle_materialization", "source_locks_sha256"),
        ("lineage_authority", "parent_contract_sha256"),
        ("lineage_authority", "parent_route_profile"),
        ("request", "observation", "checkpoint", "path"),
        ("request", "observation", "estimator_ledger", "path"),
        ("route_contract", "execution_settings", "adapt_insertion_mode"),
        ("route_contract", "lineage_authority", "parent_contract_sha256"),
        ("route_contract", "lineage_authority", "parent_route_profile"),
        ("route_contract", "route_profile"),
        (
            "route_contract",
            "semantic_invariants",
            "experimental_insertion_policy",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "plateau_energy_decrease_threshold",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "plateau_prior_mean_decrease_ratio_threshold",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "plateau_threshold_calibration_status",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "plateau_threshold_comparison",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "plateau_trigger_source",
        ),
        ("route_contract", "sha256"),
        ("sha256",),
        ("source_locks", "implementation_source_inventory_sha256"),
        ("source_locks", "source_locks_manifest_sha256"),
    }
)
ROUTE_DIFFERENCE_PATHS = frozenset()
HORIZON_DIFFERENCE_PATHS = frozenset()
SOURCE_PATCH_BINDINGS = (
    (
        "pipelines/static_adapt/ra_adapt/engine.py",
        "b8801def8836fdc5de54bad9d9d058e8ceda950b7b841b3a367bf4fdb4decc57",
        "41f57e5432f2c008eb68b675f647d981218281184496aab3479eabd0027316c0",
    ),
    (
        "pipelines/reporting/paper_i_run_summary.py",
        "5d19112c465a6169e7109deeb300369383a465439866e90fecaacedf50ead99f",
        "d515ab2bd75dd5fb2d56121ddbe5cba71e108c09281e333b9c392dc335c6d8ae",
    ),
)


class PackageContractError(RuntimeError):
    """Fail-closed package or execution-contract violation."""


def source_cell_id(regime_id: str, nph: int) -> str:
    return (
        f"global_singleton__{regime_id}__nph{int(nph)}__"
        "ra_global_singleton_plateau_commutation"
    )


def execution_id(regime_id: str, nph: int) -> str:
    return (
        f"historical_mean_global_singleton_v2_nph7_r50__{regime_id}__"
        f"nph{int(nph)}__ra_global_singleton_plateau"
    )


def expected_source_cell_ids() -> tuple[str, ...]:
    return tuple(source_cell_id(*row) for row in REGIME_ROWS)


def expected_execution_ids() -> tuple[str, ...]:
    return tuple(execution_id(*row) for row in REGIME_ROWS)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def digested(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload.pop("sha256", None)
    payload["sha256"] = canonical_sha256(payload)
    return payload


def verify_self_digest(value: Mapping[str, Any], *, label: str) -> str:
    unsigned = {key: item for key, item in value.items() if key != "sha256"}
    observed = canonical_sha256(unsigned)
    if value.get("sha256") != observed:
        raise PackageContractError(f"{label} self-digest drifted.")
    return observed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PackageContractError(f"Cannot load {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PackageContractError(f"{label} must be a JSON object.")
    return value


def safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PackageContractError(f"{label} must be a nonempty path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise PackageContractError(f"{label} is unsafe: {value!r}.")
    return Path(*pure.parts)


def binding(path: Path, *, root: Path, canonical: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise PackageContractError(f"Missing or unsafe binding target: {path}")
    try:
        display = resolved.relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise PackageContractError(f"Binding target escaped package: {path}") from exc
    result: dict[str, Any] = {
        "path": display,
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if canonical:
        payload = load_json(resolved, label=display)
        result["canonical_sha256"] = verify_self_digest(payload, label=display)
    return result


def scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    result: list[tuple[tuple[str | int, ...], Any, Any]] = []
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        for key in sorted(set(before) | set(after)):
            if key not in before:
                result.append(((*path, str(key)), "<missing>", after[key]))
            elif key not in after:
                result.append(((*path, str(key)), before[key], "<missing>"))
            else:
                result.extend(
                    scalar_differences(before[key], after[key], path=(*path, str(key)))
                )
        return result
    if isinstance(before, Sequence) and not isinstance(before, (str, bytes)):
        if not isinstance(after, Sequence) or isinstance(after, (str, bytes)):
            return [(path, before, after)]
        if len(before) != len(after):
            return [(path, before, after)]
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(
                scalar_differences(left, right, path=(*path, index))
            )
        return result
    if before != after:
        result.append((path, before, after))
    return result


def repo_root_from_script(script: str | Path) -> Path:
    for parent in Path(script).resolve().parents:
        if (parent / "AGENTS.md").is_file() and (parent / "pipelines").is_dir():
            return parent
    raise PackageContractError("Cannot resolve the active repository root.")


__all__ = [name for name in globals() if name.isupper()] + [
    "PackageContractError",
    "binding",
    "canonical_json_bytes",
    "canonical_sha256",
    "digested",
    "execution_id",
    "expected_execution_ids",
    "expected_source_cell_ids",
    "load_json",
    "repo_root_from_script",
    "safe_relative_path",
    "scalar_differences",
    "sha256_file",
    "source_cell_id",
    "verify_self_digest",
]
