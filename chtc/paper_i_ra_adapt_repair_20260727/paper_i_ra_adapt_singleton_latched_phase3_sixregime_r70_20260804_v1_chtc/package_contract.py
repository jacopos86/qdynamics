#!/usr/bin/env python3
"""Closed contract for the six-regime latched-Phase-III RA r70 package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_singleton_latched_phase3_sixregime_"
    "r70_20260804_v1_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_singleton_latched_phase3_sixregime_r70_v1"
)

# The immutable page-8 v3 package supplies the complete scientific package and
# worker-durability contract.  This derivative changes only the four declared
# implementation files, the route identity/semantics, and the fresh horizon.
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v3_chtc"
)
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "37092457bf337ce14bcb472fdcdb1d34227363ada5765434db09da2bff770ec0"
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "311b2327e2ad254156eac6fb7f5f2d6f91a391913fce93edeae99b2d52ba031b"
)
SOURCE_PROTOCOL_BUNDLE_MANIFEST_FILE_SHA256 = (
    "1b472a3d82fbeae0ff681476fb9b01ced61e02bbd7ac5d0268eb0665cc91c7c6"
)
SOURCE_PROTOCOL_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "e5cb1974760df4ab041a8e3a6451310f164a5b76af2ce336d753830a1023d231"
)
SOURCE_LOCKS_SNAPSHOT_FILE_SHA256 = (
    "26858f9752c2adeb814e9b5b266378fea57ea41a94db9d34cda2610857cf28c8"
)
SOURCE_LOCKS_SNAPSHOT_CANONICAL_SHA256 = (
    "2f2853bd0cc0cf9abfdceff86314556b0ad12f85eeff13e78ee9147508042a39"
)
SOURCE_ARCHIVE_FILE_SHA256 = (
    "bd94a87b632646e051bf99fe760639275de04f1e21cfb660fc5e8ef21f56d4bd"
)
SOURCE_ARCHIVE_MANIFEST_FILE_SHA256 = (
    "e1952eef464976c088a2aa84ca1aada1c6f0ec34cf5f089a2de7e33ce64c5ded"
)
SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256 = (
    "0363bc046caa8a5b8088fb09d04210590f1735716269e5d081cc3af7522f416c"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "20acd89a7b6747d3f93960fbf1c4a5e7c680679631fb09422026030ba2dc3be6"
)
SOURCE_ARCHIVE_MEMBER_COUNT = 165
TARGET_SOURCE_LOCKS_FILE_SHA256 = (
    "9e3c3fba93dc7fc9ee219ed1cc68eccf15ad98902c8bd426b1b19cb70c4c79b5"
)
TARGET_SOURCE_LOCKS_CANONICAL_SHA256 = (
    "df3c7cc479e6766331fa5c43807681a130cd7d4f8ee51145c9a4cec03e270e85"
)
TARGET_IMPLEMENTATION_INVENTORY_SHA256 = (
    "05a8eb3f30a3ff77ceb5024df311d53ef3354d7a1dcb00554bc4b038801081fb"
)
TARGET_ADAPT_PIPELINE_SHA256 = (
    "6b43463612fe8599bc08701a10bf75e019e4aefea12f2095b277e65ccebb978d"
)
TARGET_CONTRACTS_SHA256 = (
    "b408834aa0c540e6e56e45684075f11836ba65209adc8a0ef36bf0363c5f94f0"
)
TARGET_ENGINE_SHA256 = (
    "10610bcb2121a6508ad78b0d4ae9b3c06e648a96dcb151f28a983a8e17fb8d71"
)
TARGET_RUN_SUMMARY_SHA256 = (
    "98d0aa4f00164298778858ab1133a2fbcce9d10dc7856767093a9e72be1a02ca"
)

SOURCE_HORIZON = 50
TARGET_HORIZON = 70
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
SOURCE_ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_phase3_population_on_insertion_plateau_v1"
)
ALGORITHM_ID = (
    "paper_i_ra_adapt_singleton_latched_phase3_separate_plateau_insertion_v1"
)
ROUTE_ID = "ra_singleton_plateau"
ROUTE_CONTRACT_SHA256 = (
    "75388a99be96225951ade4f278677702051a41ffd723fc47c8b05e77e6a9e086"
)
PARENT_ROUTE_CONTRACT_SHA256 = (
    "ac868db4dab4f8446ff06e768c5ea77512ef70764efd5699621bd95ad341599d"
)
PARENT_ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "late_resource_weighting_v1__phase3_population_on_insertion_plateau_v1"
)
ROUTE_PROFILE = (
    "paper_i_ra_adapt__single_pauli_word_v1__"
    "insertion_commutation_plateau_v2__stationary_source_response_v1__"
    "late_resource_weighting_v1__"
    "phase3_population_latched_on_progress_plateau_v1__"
    "insertion_on_phase3_plateau_v1"
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

SCHEMA_PREFIX = "paper_i_ra_adapt_singleton_latched_phase3_sixregime_r70"
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
        ("algorithm_id",),
        ("bundle_id",),
        ("bundle_manifest_sha256",),
        ("bundle_materialization", "algorithm_id"),
        ("bundle_materialization", "bundle_id"),
        ("bundle_materialization", "bundle_manifest_sha256"),
        ("bundle_materialization", "cell_id"),
        ("bundle_materialization", "sha256"),
        ("bundle_materialization", "source_lock_refs_sha256"),
        ("bundle_materialization", "source_locks_sha256"),
        ("horizon",),
        ("lineage_authority", "parent_contract_sha256"),
        ("lineage_authority", "parent_route_profile"),
        ("request", "execution", "stop", "maximum_controller_rounds"),
        ("request", "observation", "checkpoint", "path"),
        ("request", "observation", "estimator_ledger", "path"),
        (
            "route_contract",
            "execution_settings",
            "ra_insertion_plateau_history_scope",
        ),
        (
            "route_contract",
            "execution_settings",
            "ra_phase3_population_activation_policy",
        ),
        (
            "route_contract",
            "lineage_authority",
            "only_intended_scientific_changes",
        ),
        (
            "route_contract",
            "lineage_authority",
            "parent_contract_sha256",
        ),
        (
            "route_contract",
            "lineage_authority",
            "parent_route_profile",
        ),
        (
            "route_contract",
            "lineage_authority",
            "supersession_reason",
        ),
        ("route_contract", "route_profile"),
        (
            "route_contract",
            "semantic_invariants",
            "insertion_activation_changes_phase3_latch",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "insertion_activation_requires_prior_phase3_latch",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "insertion_plateau_history_scope",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "phase3_activation_independent_latch",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "phase3_activation_source",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "phase3_competitive_population_activation",
        ),
        (
            "route_contract",
            "semantic_invariants",
            "phase3_latch_retirement_policy",
        ),
        ("route_contract", "sha256"),
        ("sha256",),
        ("source_locks", "implementation_source_inventory_sha256"),
        ("source_locks", "source_locks_manifest_sha256"),
        ("stopping_rule", "maximum_controller_rounds"),
    }
)

SOURCE_PATCH_BINDINGS = (
    (
        "pipelines/static_adapt/adapt_pipeline.py",
        "ae7df90b0184c3c2e923016bd461c1003846ef20bb97f044ad7c8d36bdd697e2",
        TARGET_ADAPT_PIPELINE_SHA256,
    ),
    (
        "pipelines/static_adapt/ra_adapt/contracts.py",
        "49d33a2f134a86e67c89f7be306577b96208f55023558f200bfb7942aabec08d",
        TARGET_CONTRACTS_SHA256,
    ),
    (
        "pipelines/static_adapt/ra_adapt/engine.py",
        "89dabdfdd316fffac31da9c9a1172fd0b0ef380acfcc0d486eb8f9cc88fd0248",
        TARGET_ENGINE_SHA256,
    ),
    (
        "pipelines/reporting/paper_i_run_summary.py",
        "561a8688c9d9bede50d0816cef392ae9d87d880d6900f9a26c6b96e32e338d69",
        TARGET_RUN_SUMMARY_SHA256,
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
        f"latched_phase3_r70__{regime_id}__nph{int(nph)}__"
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
