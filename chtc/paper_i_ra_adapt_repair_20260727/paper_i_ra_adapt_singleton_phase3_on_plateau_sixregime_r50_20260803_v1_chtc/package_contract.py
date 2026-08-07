#!/usr/bin/env python3
"""Closed contract for the six-regime Phase-III-on-plateau RA package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PACKAGE_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_"
    "r50_20260803_v1_chtc"
)
CAMPAIGN_ID = (
    "paper_i_ra_adapt_singleton_phase3_on_plateau_sixregime_r50_v1"
)

# The sealed v5 package supplies the six regime/cutoff physics, full-meta pool,
# and scheduler template.  Target runtime bytes and route identity are locked
# independently to the completed strong--weak source-value anchor.
SOURCE_PACKAGE_RELATIVE = Path(
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_historical_average_singleton_plateau6_"
    "r70_fresh_20260802_v5_chtc"
)
SOURCE_PACKAGE_MANIFEST_FILE_SHA256 = (
    "f20c8fbb18fe7630c21a704346359af6fb2327023fc842854f8f03ea19ebb9ea"
)
SOURCE_PACKAGE_MANIFEST_CANONICAL_SHA256 = (
    "f6e6e2b58220d3f94943e6d2a1e36e4ae1845ac894c980eddad289eaf7f8de20"
)
SOURCE_PROTOCOL_BUNDLE_MANIFEST_FILE_SHA256 = (
    "cd54dac3745b9d69411a2bda7955ed77ba3e892550bceba78652e89e668fdde0"
)
SOURCE_PROTOCOL_BUNDLE_MANIFEST_CANONICAL_SHA256 = (
    "eb49a7a58d5acda404eec78340ed50fc4575d74981339f91b5303780b6591286"
)
SOURCE_LOCKS_SNAPSHOT_FILE_SHA256 = (
    "c25000f3bcb52237a6ce07d17a3bb6c2644b43d8b9edea19a6d3c2f3b129e5a7"
)
SOURCE_LOCKS_SNAPSHOT_CANONICAL_SHA256 = (
    "7415ae292fdb34bc1ecc57a37904bcccaa645491185a640ae0210de53add03e7"
)
SOURCE_ARCHIVE_FILE_SHA256 = (
    "f8b42ea0411e9f3f763d79bcddb7ab39b1873550451e5a5e73cd53b88b07ec26"
)
SOURCE_ARCHIVE_MANIFEST_FILE_SHA256 = (
    "dc856a808bd97719ccaacc6d6bf44f73193d986b8bfa9a9af8ade2fab22693d2"
)
SOURCE_ARCHIVE_MANIFEST_CANONICAL_SHA256 = (
    "6c6e7421b02bead99676fbd3a991ff8de8a81b840f7772a68534a25e85960e9e"
)
SOURCE_IMPLEMENTATION_INVENTORY_SHA256 = (
    "24fb4e5e78231ca1bebf907449f8993d2e7cd6cf24b8473d5d2968af3fe34f76"
)
TARGET_SOURCE_LOCKS_RELATIVE = Path(
    "output/local_runs/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_"
    "r10_local_20260802_v2/materialization/source_locks_snapshot.json"
)
TARGET_SOURCE_LOCKS_FILE_SHA256 = (
    "e55eca02a39dcb0ff1743383d2a72cf8be05788a6909a01e71045c67f169d684"
)
TARGET_SOURCE_LOCKS_CANONICAL_SHA256 = (
    "ea5635f1443b6d44749c151da5167aea32ce9e7fd269a7c1dbe17daf0bbf60e8"
)
TARGET_IMPLEMENTATION_INVENTORY_SHA256 = (
    "1abcefba4fe1f611fc98f0392d84f40d891b16425dbd8d8bd93b2d2578e823b4"
)
TARGET_COMPLETED_RESULT_RELATIVE = Path(
    "output/local_runs/"
    "paper_i_ra_adapt_singleton_phase3_on_plateau_strong_weak_"
    "r50_storage_retry_local_20260802_v2/run_no_checkpoint/result.json"
)
TARGET_COMPLETED_RESULT_FILE_SHA256 = (
    "97b133a418c89f0695cb31b7100a8b0420950cdf6da4d08b12380d72d43db133"
)

SOURCE_HORIZON = 70
TARGET_HORIZON = 50
ACTIVE_GRADIENT_POLICY = "stationary_source_response_v1"
RESOURCE_WEIGHTING_SCOPE = "late_resource_weighting_v1"
CANDIDATE_REPRESENTATION = "single_pauli_word_v1"
SOURCE_ALGORITHM_ID = "paper_i_ra_adapt_singleton_plateau_insertion_repair_v1"
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
        ("request", "execution", "stop", "maximum_controller_rounds"),
        ("request", "observation", "checkpoint", "path"),
        ("request", "observation", "estimator_ledger", "path"),
        ("route_contract", "execution_settings", "ra_phase3_population_activation_policy"),
        ("route_contract", "execution_settings", "ra_phase3_preplateau_materialization_policy"),
        ("route_contract", "lineage_authority", "only_intended_scientific_changes"),
        ("route_contract", "lineage_authority", "supersession_reason"),
        ("route_contract", "route_profile"),
        ("route_contract", "semantic_invariants", "phase1_activation_scope"),
        ("route_contract", "semantic_invariants", "phase2_activation_scope"),
        ("route_contract", "semantic_invariants", "phase3_activation_hysteresis_active"),
        ("route_contract", "semantic_invariants", "phase3_activation_independent_latch"),
        ("route_contract", "semantic_invariants", "phase3_activation_source"),
        ("route_contract", "semantic_invariants", "phase3_competitive_population_activation"),
        ("route_contract", "semantic_invariants", "phase3_preplateau_admission_authority"),
        ("route_contract", "semantic_invariants", "phase3_preplateau_materialization_policy"),
        ("route_contract", "sha256"),
        ("sha256",),
        ("source_locks", "implementation_source_inventory_sha256"),
        ("source_locks", "source_locks_manifest_sha256"),
        ("stopping_rule", "maximum_controller_rounds"),
    }
)

SOURCE_PATCH_BINDINGS = (
    ("pipelines/reporting/paper_i_run_summary.py", "5d19112c465a6169e7109deeb300369383a465439866e90fecaacedf50ead99f", "561a8688c9d9bede50d0816cef392ae9d87d880d6900f9a26c6b96e32e338d69"),
    ("pipelines/static_adapt/adapt_pipeline.py", "eb5d578eaddf76cdc02e2293b09bce6b1dde24b2edb8aed1be95d273310cb243", "ae7df90b0184c3c2e923016bd461c1003846ef20bb97f044ad7c8d36bdd697e2"),
    ("pipelines/static_adapt/builders/legal_subspace_filter.py", "ef703de200a231f65abcf2e6656b2d431447b0292ed7f298391984cca87c1bb1", "1ed88ae132c7d074ca9dc716f10c7b8991ff3ba6b2c37ea9331ab72b4de0021e"),
    ("pipelines/static_adapt/builders/shared_pauli_pool_contract.py", "e0ba4fff2d8473988b3ea527eac3e497f07d5acab7416ca7316e64599f6a0a46", "3df1aef354b5b83efc21201df340ac959e1e7b0b9e2058d23ecc73d37d79e96c"),
    ("pipelines/static_adapt/ra_adapt/__init__.py", "492f681e7627742f6f8d80a25132714bfc923b0f6c3e6a65c32a8f38153ac792", "f62337cb298a721d6bc9c426f6d5571cc27b783558a79acc0e70220cfd9f655c"),
    ("pipelines/static_adapt/ra_adapt/adapters.py", "736f9eb64e13615aa5152e882ec57a12294d3c13427922eb064215e0b97b47db", "a62583c680cc2958785b76425f9fd57f254b3cd41293487b0d1d6b9d35b646d4"),
    ("pipelines/static_adapt/ra_adapt/contracts.py", "91287afedced746c1ce2edd04387212ce1d3f3f4f679be91411928a3f5ac3c65", "49d33a2f134a86e67c89f7be306577b96208f55023558f200bfb7942aabec08d"),
    ("pipelines/static_adapt/ra_adapt/engine.py", "b8801def8836fdc5de54bad9d9d058e8ceda950b7b841b3a367bf4fdb4decc57", "55aafe2e8a163ea1c9d2649c9a3ae3181304e8cd64b4d2c968a812722723c4cb"),
    ("pipelines/static_adapt/ra_adapt/h2o_application.py", "<absent>", "6554592dbbb01930690a4d71f071a8d7684a2e0155d33ccf222b63dec41369db"),
    ("pipelines/static_adapt/ra_adapt/pools.py", "cf8964a8acd1b7b5851d9ff27f2f2aa05d8d2848a7c1bde0f02e18751767e842", "a57405bc27b94788f8db5de34ef5b459c35ee89ae7b67eb6a0bc0529f018aa5d"),
    ("pipelines/static_adapt/route_a_child_padding.py", "b98605c1a5288cb46696821747b5bcae28baebb120fcad4ce08ed1fc787fab4b", "2e221209fc57eb85b291dd12da3a11a5a346c8712f01c7442ed57cd85f9ead15"),
    ("pipelines/static_adapt/sector_invariants.py", "b2aec6229dafc93846c2f61689ca90b7bd975be6294bad882e894e2287f4378d", "6fd5008979a7dd7bd1b002ffd01c1265494fd53235cc4664e76beb846d447d6f"),
    ("pipelines/static_adapt/sr_snake/_context.py", "7553884472a45b2e1b481c4c3e6fa4bb023717835a6fc7dcc594ccc51092fd12", "41a9ab852ee0fdeb4d259c1bc310153a22e6d1cde21a2c40eecfab14121ff929"),
    ("pipelines/static_adapt/sr_snake/_controller.py", "5fd617814a9167a9eea1ed5dcdb3789fbc39c80caa1f78fd8a643b0a05472430", "4b0171acdbf9b0c9bcf3d0d626da959ffae7f20129eaa2ee251d02f62c8c265a"),
    ("pipelines/static_adapt/sr_snake/contracts.py", "45e27fad002935f25ad649bd90a7b91affff9a0d0f53d24e335aa8dbbb843145", "ecb2c2f167245033969756c5635fee14d41444853cffe2901ad66db88dbea7d7"),
    ("src/quantum/compiled_ansatz.py", "890991abc92b1719508925b63b12f341ee633d472677cd7e2cdb81b50c3f7ebf", "a379caac82f9f3484002989770c464000e9f6475106c38739b082c8fe7c7e5b2"),
    ("src/quantum/pauli_actions.py", "de1685f785ecbb46dc681dff1246f4e9e6a68cbcfeb6bb7783f60a29fa2f46e1", "4570b652127a39615f71d5e1db851930f0d626ba936212634b80654bea5ead5d"),
)


class PackageContractError(RuntimeError):
    """Fail-closed package or execution-contract violation."""


def source_cell_id(regime_id: str, nph: int) -> str:
    return (
        f"historical_average_v5_r70_fresh__{regime_id}__nph{int(nph)}__"
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
