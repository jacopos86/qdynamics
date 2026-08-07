#!/usr/bin/env python3
"""Closed contract for the six-cell cumulative-relative RA r70 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_"
    "r70_fresh_20260731_v1_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_fresh_v1"
)
SOURCE_BUNDLE_ID = "ra_repair_stationary_late_core_v1"
SOURCE_BUNDLE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/bundles/materializations/"
    "ra_adapt_stationary_late_core_v13/ra_repair_stationary_late_core_v1"
)
SOURCE_BUNDLE_MANIFEST_FILE_SHA256 = (
    "d49f4a03d6b74d830c9f1aecbb4f191bfeb8afd4b67d25dab75aca727ef9936e"
)
SOURCE_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "7a4518c160ba5270c52c98f9885e839cdb1c9f22e5e4e8d525a57d4a8625304a"
)
SOURCE_LOCKS_FILE_SHA256 = (
    "4b65d21bde548345a4dc47995974271f110c23bb6ead2237b1bccea18e0ae8cd"
)
SOURCE_LOCKS_CANONICAL_SHA256 = (
    "0a5e58daa29f133d3e3aa1406672053cb8e80f6182baed9f89690b4aeec16c17"
)

SOURCE_HORIZON = 50
TARGET_HORIZON = 70
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
ALGORITHM_ID = "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
ROUTE_ID = "ra_singleton_plateau"
ROUTE_CONTRACT_SHA256 = (
    "7ae0d6294ccee8ff8d79d3ed308d50659b6a7c810423835b0c2cdfa849efa4fd"
)
PLATEAU_RATIO_THRESHOLD = 1.0e-4
PLATEAU_COMPARISON = "marginal_to_prior_cumulative_strictly_below_v1"
PLATEAU_TRIGGER = (
    "immediately_preceding_marginal_over_prior_cumulative_"
    "accepted_post_full_refit_energy_decrease_v1"
)
PLATEAU_CALIBRATION = "source_locked_completed_trajectory_replay_v1"

REGIME_ROWS: tuple[tuple[str, int], ...] = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)

RESOURCE_ENVELOPES = {
    3: {
        "request_cpus": 4,
        "request_memory_mb": 24_576,
        "request_disk_mb": 40_960,
        "max_runtime_seconds": 259_200,
    },
    7: {
        "request_cpus": 4,
        "request_memory_mb": 32_768,
        "request_disk_mb": 61_440,
        "max_runtime_seconds": 259_200,
    },
}

PACKAGE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_"
    "package_manifest_v1"
)
PROTOCOL_BUNDLE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_"
    "protocol_bundle_manifest_v1"
)
SOURCE_ARCHIVE_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_"
    "source_archive_manifest_v1"
)
SOURCE_LOCK_AUDIT_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_"
    "source_lock_audit_v1"
)
JOB_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_job_v1"
)
AUTHORIZATION_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_"
    "execution_authorization_v1"
)
RUN_MANIFEST_SCHEMA = (
    "paper_i_ra_adapt_cumulative_relative_singleton_plateau6_r70_"
    "run_manifest_v1"
)

CONTROL_FILES = (
    "package_contract.py",
    "build_package.py",
    "run_cell.py",
    "validate_package.py",
    "execute_authorized_job.sh",
    "submit.sub.in",
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

ROUTE_DIFFERENCE_PATHS = frozenset(
    {
        ("lineage_authority", "parent_contract_sha256"),
        (
            "semantic_invariants",
            "plateau_cumulative_decrease_ratio_threshold",
        ),
        ("semantic_invariants", "plateau_energy_decrease_threshold"),
        (
            "semantic_invariants",
            "plateau_threshold_calibration_status",
        ),
        ("semantic_invariants", "plateau_threshold_comparison"),
        ("semantic_invariants", "plateau_trigger_source"),
        ("sha256",),
    }
)
HORIZON_DIFFERENCE_PATHS = frozenset(
    {
        ("horizon",),
        ("request", "execution", "stop", "maximum_controller_rounds"),
        ("sha256",),
        ("stopping_rule", "maximum_controller_rounds"),
    }
)


class PackageContractError(RuntimeError):
    """Fail-closed package or execution-contract violation."""


def source_cell_id(regime_id: str, nph: int) -> str:
    return f"core__{regime_id}__nph{int(nph)}__ra_singleton_plateau"


def execution_id(regime_id: str, nph: int) -> str:
    return (
        f"cumulative_r70_fresh__{regime_id}__nph{int(nph)}__"
        "ra_singleton_plateau"
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
        raise PackageContractError(f"Missing unsafe binding target: {path}")
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
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
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
