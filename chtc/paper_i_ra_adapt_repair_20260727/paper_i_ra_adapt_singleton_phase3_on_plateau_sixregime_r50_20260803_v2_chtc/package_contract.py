#!/usr/bin/env python3
"""Closed contract for the six-regime Phase-III-on-plateau RA package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v2_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_v1"
)

# The immutable v1 package supplies the complete scientific package.  v2 is an
# operational repair only: it replaces the accepted-winner stabilization
# collector in engine.py and preserves every route/science setting.
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v1_chtc"
)
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "46a488ad7a68f1b9865515c01898a93302af8d4f1abb8c988b4f14b3f165c73e"
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "84cecf7d1ecf17e73e33d80eff5fa3b489ab2e46c34e20acf59491a4879edd37"
)
SOURCE_PROTOCOL_BUNDLE_MANIFEST_FILE_SHA256 = (
    "9424f354103eff74e37493cca1883fc12288499c4ca19e75faa334095e4317e1"
)
SOURCE_PROTOCOL_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "d34194a2b48dc6b632cc8e4d9346ca7625e89ce3084bd02d4e5ac4a7fd84107c"
)
SOURCE_LOCKS_SNAPSHOT_FILE_SHA256 = (
    "e55eca02a39dcb0ff1743383d2a72cf8be05788a6909a01e71045c67f169d684"
)
SOURCE_LOCKS_SNAPSHOT_CANONICAL_SHA256 = (
    "ea5635f1443b6d44749c151da5167aea32ce9e7fd269a7c1dbe17daf0bbf60e8"
)
SOURCE_ARCHIVE_FILE_SHA256 = (
    "e388f39093664ad7a342907f7e604e0a100673a0abb60d79923947b199af006b"
)
SOURCE_ARCHIVE_MANIFEST_FILE_SHA256 = (
    "7118eac73ee0625c2e375801bb0e29c349b9eba26bd0a31b5e5c31fb5cf4a186"
)
SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256 = (
    "9338b2f47cb4df12319f609f962009047aa726c92e57227b650a3546b280a77d"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "1abcefba4fe1f611fc98f0392d84f40d891b16425dbd8d8bd93b2d2578e823b4"
)
TARGET_SOURCE_LOCKS_FILE_SHA256 = (
    "059326ac222a2429b96105838215bad89070f5c4881c932eceaf494860ac0c99"
)
TARGET_SOURCE_LOCKS_CANONICAL_SHA256 = (
    "1b6de8b700ce7b68635eeb42d2b816a5ca6f9682aa722ee0a63378250c491409"
)
TARGET_IMPLEMENTATION_INVENTORY_SHA256 = (
    "b8a5b74c5e0f7d59cb0ff3caf269613e9839a93b2977bbc163080fe24a2dabe7"
)
TARGET_ENGINE_SHA256 = (
    "2c58e06896a0c033c649fb7817fba0563bcb66637fe701839673649d9f50180c"
)

SOURCE_HORIZON = 50
TARGET_HORIZON = 50
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
SOURCE_ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_phase3_population_on_insertion_plateau_v1"
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_phase3_population_on_insertion_plateau_v1"
)
ROUTE_ID = "ra_singleton_plateau"
ROUTE_CONTRACT_SHA256 = (
    "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
)
PARENT_ROUTE_CONTRACT_SHA256 = (
    "aa669d7f0c3621d9ddf7f8595f96333c56b536c8fc79547607e76d8d91d4b6ff"
)
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "late_resource_weighting_v1__phase3_population_on_insertion_plateau_v1"
)
PLATEAU_PRIOR_MEAN_RATIO_THRESHOLD = 1.0e-4
PLATEAU_COMPARISON = "marginal_to_prior_mean_strictly_below_v2"
PLATEAU_TRIGGER = (
    "immediately_preceding_marginal_over_prior_mean_"
    "accepted_post_full_refit_energy_decrease_v2"
)
PLATEAU_CALIBRATION = "source_locked_counterfactual_trigger_replay_v2"

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

SCHEMA_PREFIX = "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50"
PACKAGE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_package_manifest_v1"
PROTOCOL_BUNDLE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_protocol_bundle_manifest_v1"
SOURCE_ARCHIVE_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_source_archive_manifest_v1"
SOURCE_LOCK_AUDIT_SCHEMA = f"{SCHEMA_PREFIX}_source_lock_audit_v1"
JOB_SCHEMA = f"{SCHEMA_PREFIX}_job_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RUN_MANIFEST_SCHEMA = f"{SCHEMA_PREFIX}_run_manifest_v1"

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

SOURCE_TO_TARGET_DIFFERENCE_PATHS = frozenset(
    {
        ("bundle_id",),
        ("bundle_manifest_sha256",),
        ("bundle_materialization", "bundle_id"),
        ("bundle_materialization", "bundle_manifest_sha256"),
        ("bundle_materialization", "sha256"),
        ("bundle_materialization", "source_lock_refs_sha256"),
        ("bundle_materialization", "source_locks_sha256"),
        ("sha256",),
        ("source_locks", "implementation_source_inventory_sha256"),
        ("source_locks", "source_locks_manifest_sha256"),
    }
)

SOURCE_PATCH_BINDINGS = (
    (
        "pipelines/static_adapt/ra_adapt/engine.py",
        "55aafe2e8a163ea1c9d2649c9a3ae3181304e8cd64b4d2c968a812722723c4cb",
        TARGET_ENGINE_SHA256,
    ),
)


class PackageContractError(RuntimeError):
    """Fail-closed package or execution-contract violation."""


def source_cell_id(regime_id: str, nph: int) -> str:
    return (
        f"phase3_on_plateau_r50__{regime_id}__nph{int(nph)}__"
        "ra_singleton_plateau"
    )


def execution_id(regime_id: str, nph: int) -> str:
    return (
        f"phase3_on_plateau_r50__{regime_id}__nph{int(nph)}__"
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
        # The length equality above preserves strict-zip semantics while
        # remaining executable on the CHTC access-point Python (pre-3.10).
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
