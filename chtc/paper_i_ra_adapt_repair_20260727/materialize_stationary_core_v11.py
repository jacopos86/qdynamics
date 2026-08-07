"""Atomically materialize the selected 48-cell stationary Paper-I core.

This append-only command has no scientific-execution or scheduler-submission
seam. It preserves v1-v10, reuses all 48 selected v10 scientific source locks
byte for byte, verifies every archive/member/global source exactly, refreshes
the implementation inventory from the repaired source tree, validates all 48
protocols, and publishes v11 with no-replace semantics. None of the CHTC
9381198 scientific results are adopted as Paper-I evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from chtc.paper_i_ra_adapt_repair_20260727 import (
    materialize_stationary_core_v10 as v10,
)
from pipelines.static_adapt.ra_adapt.bundles import (
    CLAIM_FACING_REGIME_CUTOFF_PAIRS,
    CORE_BUNDLE_ID,
    CORE_CAMPAIGN_ID,
    CORE_RUN_CLASS,
    CORE_SELECTION_AUTHORITY_PATH,
    CORE_SELECTION_AUTHORITY_SHA256,
    CORE_VISIBLE_TARGET_ID,
    FULL_HORIZON,
    GLOBAL_SOURCE_LOCKS,
    MACRO_ROUTE_IDS,
    ROUTE_APPEND_SINGLETON,
    ROUTE_RA_SINGLETON_ALWAYS,
    ROUTE_RA_SINGLETON_APPEND_ONLY,
    ROUTE_RA_SINGLETON_PLATEAU,
    SINGLETON_CORE_ROUTE_IDS,
    SOURCE_LOCK_SCHEMA,
    _implementation_source_inventory,
    build_core_cell_specs,
    load_validated_bundle_protocol,
    materialize_core_bundle,
    normalize_and_verify_source_locks,
    source_lock_id,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    canonical_json_bytes,
    canonical_sha256,
)


REPO_ROOT = v10.REPO_ROOT
MATERIALIZATIONS_ROOT = v10.MATERIALIZATIONS_ROOT
support = v10.support
V10_ROOT = v10.V10_ROOT
V11_ROOT = (
    MATERIALIZATIONS_ROOT / "ra_adapt_stationary_late_core_v11"
)
V10_SOURCE_LOCKS_INPUT = (
    V10_ROOT / "source_materialization" / "source_locks_input.json"
)
V10_PROBLEM_BASELINES = (
    V10_ROOT / "source_materialization" / "problem_baselines.json"
)
V10_FINAL_RECEIPT = V10_ROOT / "final_publication_receipt.json"
APPEND_REGISTRY = (
    REPO_ROOT
    / "agent_guidance/static-adapt/reporting/"
    "canonical-append-registry-v1.json"
)
ED_REFERENCE = REPO_ROOT / GLOBAL_SOURCE_LOCKS["ed_cutoff_reference"]["path"]
ICM_RECEIPT = (
    REPO_ROOT
    / "agent_guidance/static-adapt/icm/ra-adapt-repair-20260727/"
    "materialize-stationary-core-v11.json"
)
FINAL_RECEIPT_NAME = "final_publication_receipt.json"

EXPECTED_V10_FILE_COUNT = 111
EXPECTED_V10_TOTAL_SIZE_BYTES = 4_238_638
EXPECTED_V10_RELATIVE_TREE_SHA256 = (
    "19933f41db4255b710c61aa94bc760252618b69ec03e29790a38347aec9d112f"
)
EXPECTED_V10_FINAL_RECEIPT_FILE_SHA256 = (
    "8c396883ebd728150057eb9b223793621f774b0475fde59a80675d2de2ccd354"
)
EXPECTED_V10_FINAL_RECEIPT_CANONICAL_SHA256 = (
    "5924cf714ca3f1a36b3b766b4c5e30c5599d1606267ffd98fc08685afa1a9e80"
)
EXPECTED_V10_SOURCE_LOCKS_FILE_SHA256 = (
    "e31a6a0814cdee2d1b4bce7b7fefd18612e0f5d93fe5010f7e48a4e6679acccc"
)
EXPECTED_V10_PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)
EXPECTED_APPEND_REGISTRY_FILE_SHA256 = (
    "2b1effdb864c46c8edbd9c16a24909497ceffeeda57f3a960eb5432fc605ea95"
)
EXPECTED_CORE_VALIDATION_CHECK_IDS = {
    "bundle_schema_and_digest",
    "exact_core_cell_matrix",
    "source_locks_exact_bytes",
    "resolved_protocol_contracts",
    "macro_pool_hash_equality",
    "singleton_pool_exposure_contracts",
    "all_cells_direct_execution",
    "protocol_execution_separation",
    "paper_i_run_materialization_gate",
}
EXPECTED_ROUTE_IDS = (*MACRO_ROUTE_IDS, *SINGLETON_CORE_ROUTE_IDS)
EXPECTED_CELL_COUNT = 48
APPEND_REGISTRY_PATH = APPEND_REGISTRY.relative_to(REPO_ROOT).as_posix()
V10_SOURCE_LOCKS_PATH = V10_SOURCE_LOCKS_INPUT.relative_to(REPO_ROOT).as_posix()

_CORE_COMMON_DELTA_IDS = (
    "core_stationary_gradient_policy",
    "core_candidate_representation_axis",
    "core_fixed_horizon",
)
_ROUTE_INSERTION_KIND = {
    ROUTE_RA_SINGLETON_APPEND_ONLY: "append_only",
    ROUTE_RA_SINGLETON_PLATEAU: "plateau_commutation",
    ROUTE_RA_SINGLETON_ALWAYS: "full_commutation",
}
_REGIME_ED_NAME = {
    "weak_weak": "weak-weak",
    "intermediate_weak": "intermediate-weak",
    "strong_weak_u8": "strong-weak",
    "weak_strong": "weak-strong",
    "intermediate_strong": "intermediate-strong",
    "strong_strong_u8": "strong-strong",
}
_PROC_BY_REGIME = {
    "intermediate_strong": 0,
    "intermediate_weak": 1,
    "strong_strong_u8": 2,
    "strong_weak_u8": 3,
    "weak_strong": 4,
    "weak_weak": 5,
}
_TRANSFER_ARCHIVE_SHA256 = {
    0: "04cd2bbe9ddab23c05909cd2b4df6167221ddcfc3fbaa584623e98dd1e3fe02a",
    1: "5c3309a5fa0c3c4519617a8569bb6560e6cda37c80b3b02e6e389f1469c22f18",
    2: "227632955dcc6bf0ff7a176c1989caf7986691d5d32edae9497f13e088a7e847",
    3: "86a2f9a70bcfdd31c46d6a287dfee1d01b2ae2a326a175973b2dfa049809ba3a",
    4: "b7a8347697cc95c7689617c1c2dd94eaacf87aaa855ff0e719bf16d704d84364",
    5: "4f8640b11c61ba7ac7068181e6aac58e970554ed163c70949606d332a2a23d91",
}
_TRANSFER_ARCHIVE_NAME = {
    0: "9381198.0__intermediate_strong_transfer.tar.gz",
    1: "9381198.1__intermediate_weak_transfer.tar.gz",
    2: "9381198.2__strong_strong_u8_transfer.tar.gz",
    3: "9381198.3__strong_weak_u8_transfer.tar.gz",
    4: "9381198.4__weak_strong_transfer.tar.gz",
    5: "9381198.5__weak_weak_transfer.tar.gz",
}
_TRANSFER_MEMBER_SHA256 = {
    0: "d1d3d88e865d3803fff68dca2f1a2dc38e525e56f1c09bb608fbf70f0fcc4a6d",
    1: "d9f5cc4554e86f5439c7dc9453accaafcd98f447f0d3e15fd4709fd50c1077ae",
    2: "a675489f2b0fefd8f5fd221385f0a02133c0bbf660442e1532d6cd67bca5ae3c",
    3: "52b456b1e0e8b32610d5deafc3a36864ec385f35aa6e2cfa08fa5058334027d3",
    4: "977d6ae5b0f5b9e12d752044956bd4dece42096cc2278ed8d7463040de9c36fe",
    5: "40a1532fc3cc7abb03549248ef117047a023e227de28f8347f39e5f692cd1e0f",
}
_TRANSFER_JOB_MANIFEST_SHA256 = {
    0: "1c838b342bbc5aeddd77c56048ce215e4931c9a48fae7682209ae7116090725f",
    1: "9bced13c037f8b62b92d93f33c31bc60bd075beee4c949cd6dea8d4d1891d755",
    2: "e48d61fb25ea686281fca1016141660f8d99683e93ca0b17927cb8c35e2f1f43",
    3: "0ae1e50e0fc2ac436862b5ca1c50896885aaa464bd878c2900f1b624ed38bd58",
    4: "767004bcb71b7a0b5d002a166db04d523a141a285ecc75ac4521d227c342aeb8",
    5: "87f1963ec6c4a18c608960cca74ddf1cec268c6b309236c387535de62d3c1e85",
}
_TRANSFER_JOB_MANIFEST_PREFIX = (
    "chtc/phase3_optuna/input/"
    "paper_i_hh_sr_snake_singleton_insertion_commutation_plateau_all_six_"
    "r50_20260725_v3_chtc/jobs"
)
_TRANSFER_MEMBER_PREFIX = (
    "raw_outputs/"
    "paper_i_hh_sr_snake_singleton_insertion_commutation_plateau_all_six_"
    "r50_20260725_v3_chtc"
)
_TRANSFER_ROOT = (
    REPO_ROOT / "tmp/chtc_retrieval/paper_i_insertion_20260726"
)
_RA_REUSED_EXECUTION_FIELDS = (
    "adapt_final_full_refit",
    "adapt_finite_angle",
    "adapt_full_refit_every",
    "adapt_reopt_policy",
    "adapt_window_size",
    "adapt_window_topk",
    "phase1_prune_enabled",
    "phase2_enable_batching",
    "adapt_final_refit_maxiter",
    "adapt_inner_optimizer",
    "adapt_maxiter",
    "adapt_seed",
    "phase3_backend_transpile_seed",
)


MaterializationAuditError = v10.MaterializationAuditError


def _assert_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise MaterializationAuditError(
            f"{label} drifted: {actual!r} != {expected!r}."
        )


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    return v10._load_mapping(path, label=label)


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    return v10._load_digested(path, label=label)


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return v10._write_receipt(path, payload)


def _write_plain_json(path: Path, payload: Mapping[str, Any]) -> None:
    support._write_bytes_atomic_no_replace(
        path, canonical_json_bytes(payload) + b"\n"
    )


def _file_binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = _load_digested(path, label=f"digested file {path.name}")
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "canonical_sha256": payload["sha256"],
        "file_sha256": support._hash_file(path),
    }


def _historical_snapshots() -> dict[str, dict[str, Any]]:
    snapshots = v10._historical_snapshots()
    snapshots["v10"] = support._snapshot_roots((V10_ROOT,))
    return snapshots


def _assert_historical_anchors(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    v10._assert_historical_anchors(snapshots)
    observed = snapshots.get("v10")
    if not isinstance(observed, Mapping):
        raise MaterializationAuditError("Missing immutable v10 snapshot.")
    relative = support._snapshot_roots((V10_ROOT,), relative_to=V10_ROOT)
    _assert_equal(
        int(relative["file_count"]),
        EXPECTED_V10_FILE_COUNT,
        label="v10 immutable file count",
    )
    _assert_equal(
        int(relative["total_size_bytes"]),
        EXPECTED_V10_TOTAL_SIZE_BYTES,
        label="v10 immutable total bytes",
    )
    _assert_equal(
        relative["tree_sha256"],
        EXPECTED_V10_RELATIVE_TREE_SHA256,
        label="v10 immutable relative tree SHA-256",
    )
    _assert_equal(
        support._hash_file(V10_FINAL_RECEIPT),
        EXPECTED_V10_FINAL_RECEIPT_FILE_SHA256,
        label="v10 final receipt file SHA-256",
    )
    final = _load_digested(
        V10_FINAL_RECEIPT, label="v10 final materialization receipt"
    )
    _assert_equal(
        final["sha256"],
        EXPECTED_V10_FINAL_RECEIPT_CANONICAL_SHA256,
        label="v10 final receipt canonical SHA-256",
    )


def _assert_historical_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    label: str,
) -> None:
    if before != after:
        changed = sorted(
            revision
            for revision in set(before) | set(after)
            if before.get(revision) != after.get(revision)
        )
        raise MaterializationAuditError(
            f"{label} modified immutable materializations: {changed}."
        )


def _preservation_comparison(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        revision: {
            "file_count": int(before[revision]["file_count"]),
            "pre_tree_sha256": before[revision]["tree_sha256"],
            "post_tree_sha256": after[revision]["tree_sha256"],
            "unchanged": before[revision] == after[revision],
        }
        for revision in sorted(before)
    }


def _same_cutoff_ed_by_regime() -> dict[str, dict[str, Any]]:
    payload = _load_mapping(ED_REFERENCE, label="same-cutoff ED reference")
    rows = payload.get("regimes")
    if not isinstance(rows, list):
        raise MaterializationAuditError(
            "Same-cutoff ED authority has no regime list."
        )
    by_name = {
        str(row.get("name")): row
        for row in rows
        if isinstance(row, Mapping)
    }
    result: dict[str, dict[str, Any]] = {}
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        name = _REGIME_ED_NAME[regime_id]
        row = by_name.get(name)
        if not isinstance(row, Mapping):
            raise MaterializationAuditError(
                f"Missing ED regime {name!r}."
            )
        cells = row.get("cells")
        if not isinstance(cells, list):
            raise MaterializationAuditError(
                f"Missing ED cells for {name!r}."
            )
        cell = next(
            (
                item
                for item in cells
                if isinstance(item, Mapping)
                and int(item.get("M", -1)) == int(nph)
            ),
            None,
        )
        if not isinstance(cell, Mapping):
            raise MaterializationAuditError(
                f"Missing ED cutoff {nph} for {name!r}."
            )
        result[regime_id] = {
            "path": GLOBAL_SOURCE_LOCKS["ed_cutoff_reference"]["path"],
            "sha256": GLOBAL_SOURCE_LOCKS["ed_cutoff_reference"]["sha256"],
            "regime_name": name,
            "nph": int(nph),
            "E_ED": float(cell["E_ED"]),
            "required": True,
            "reference_role": "same_cutoff_reporting_reference",
        }
    return result


def _core_changes(
    *,
    route_id: str,
    source_insertion: str,
) -> list[dict[str, Any]]:
    changes = [
        {
            "id": "core_stationary_gradient_policy",
            "field": "active_gradient_policy",
            "to": "stationary_source_response_v1",
            "authority": {
                "path": CORE_SELECTION_AUTHORITY_PATH,
                "sha256": CORE_SELECTION_AUTHORITY_SHA256,
            },
        },
        {
            "id": "core_candidate_representation_axis",
            "field": "candidate_representation",
            "to": "single_pauli_word",
            "classification": "selected_48_cell_core_axis_v1",
        },
        {
            "id": "core_fixed_horizon",
            "field": "horizon",
            "to": FULL_HORIZON,
            "classification": "settled_full_50_round_core_v1",
        },
    ]
    if route_id == ROUTE_APPEND_SINGLETON:
        changes.append(
            {
                "id": "core_conventional_append_baseline",
                "field": "selector_and_geometry",
                "from": "canonical_append_registry_v1",
                "to": "conventional_unwhitened_append_v1",
            }
        )
    else:
        changes.append(
            {
                "id": "core_insertion_policy_variant",
                "field": "insertion_policy",
                "from": source_insertion,
                "to": _ROUTE_INSERTION_KIND[route_id],
                "route_id": route_id,
            }
        )
    return changes


def _core_source_anchor(
    *,
    regime_id: str,
    nph: int,
    route_id: str,
    source_insertion: str,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    if route_id == ROUTE_APPEND_SINGLETON:
        anchor_family = "canonical_append_registry_v1"
        source_route_id = ROUTE_APPEND_SINGLETON
        target_insertion = "conventional_unwhitened_append_v1"
        route_delta_id = "core_conventional_append_baseline"
    else:
        anchor_family = "chtc_9381198_singleton_plateau_v1"
        source_route_id = ROUTE_RA_SINGLETON_PLATEAU
        target_insertion = _ROUTE_INSERTION_KIND[route_id]
        route_delta_id = "core_insertion_policy_variant"
    return {
        "schema": "paper_i_ra_adapt_core_singleton_source_anchor_v1",
        "anchor_family": anchor_family,
        "regime_id": regime_id,
        "nph": int(nph),
        "scientific_result_anchor_claimed": False,
        "settings_authority": dict(authority),
        "route_derivation": {
            "source_route_id": source_route_id,
            "target_route_id": route_id,
            "source_insertion_policy": source_insertion,
            "target_insertion_policy": target_insertion,
            "declared_delta_ids": [
                *_CORE_COMMON_DELTA_IDS,
                route_delta_id,
            ],
        },
    }


def _append_singleton_locks(
    *,
    registry: Mapping[str, Any],
    ed_by_regime: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _assert_equal(
        registry.get("schema"),
        "paper_i_canonical_append_registry_v1",
        label="Append registry schema",
    )
    _assert_equal(
        registry.get("route_id"),
        "append_adapt_projected_singleton_nph3_7",
        label="Append registry route",
    )
    records = registry.get("records")
    if not isinstance(records, list) or len(records) != 6:
        raise MaterializationAuditError(
            "Canonical Append registry must contain six records."
        )
    by_regime = {
        str(record.get("regime")): record
        for record in records
        if isinstance(record, Mapping)
    }
    locks: dict[str, Any] = {}
    authority_rows: list[dict[str, Any]] = []
    cutoff_by_regime = dict(CLAIM_FACING_REGIME_CUTOFF_PAIRS)
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        record = by_regime.get(regime_id)
        if not isinstance(record, Mapping):
            raise MaterializationAuditError(
                f"Append registry is missing {regime_id!r}."
            )
        _assert_equal(
            int(cutoff_by_regime[regime_id]),
            int(nph),
            label=f"{regime_id} selected cutoff",
        )
        comparison = record.get("comparison_contract")
        source = record.get("source")
        if not isinstance(comparison, Mapping) or not isinstance(
            source, Mapping
        ):
            raise MaterializationAuditError(
                f"Append registry record {regime_id!r} is malformed."
            )
        _assert_equal(
            comparison.get("optimizer"),
            "POWELL",
            label=f"{regime_id} Append optimizer",
        )
        _assert_equal(
            int(comparison.get("optimizer_maxiter", -1)),
            200,
            label=f"{regime_id} Append optimizer cap",
        )
        _assert_equal(
            int(comparison.get("seed", -1)),
            7,
            label=f"{regime_id} Append seed",
        )
        archive_path = str(source["archive_path"])
        archive_sha = str(source["archive_sha256"])
        member_path = str(source["runtime_seed_member"])
        member_sha = str(source["runtime_seed_sha256"])
        lock_id = source_lock_id(
            regime_id, int(nph), ROUTE_APPEND_SINGLETON
        )
        authority = {
            "path": APPEND_REGISTRY_PATH,
            "file_sha256": EXPECTED_APPEND_REGISTRY_FILE_SHA256,
            "record_regime": regime_id,
            "member_role": "runtime_seed_settings_authority",
        }
        trace = {
            "source_map": APPEND_REGISTRY_PATH,
            "regime_or_case": regime_id,
            "method": lock_id,
            "source_json": f"{archive_path}#{member_path}",
            "source_sha256_expected": member_sha,
            "source_sha256_actual": member_sha,
            "source_sha256_match": True,
            "settings_reused": {
                "settings": {
                    "optimizer": comparison["optimizer"],
                    "optimizer_maxiter": int(
                        comparison["optimizer_maxiter"]
                    ),
                    "seed": int(comparison["seed"]),
                }
            },
            "settings_reused_sources": {
                "registry_record": (
                    f"{APPEND_REGISTRY_PATH}#/records/"
                    f"{records.index(record)}/comparison_contract"
                ),
                "runtime_seed_member": member_path,
            },
            "settings_changed": _core_changes(
                route_id=ROUTE_APPEND_SINGLETON,
                source_insertion="conventional_unwhitened_append_v1",
            ),
            "same_cutoff_ed_reference": dict(
                ed_by_regime[regime_id]
            ),
            "core_source_anchor": _core_source_anchor(
                regime_id=regime_id,
                nph=int(nph),
                route_id=ROUTE_APPEND_SINGLETON,
                source_insertion="conventional_unwhitened_append_v1",
                authority=authority,
            ),
            "source_archive_member_authority": {
                "archive": {
                    "path": archive_path,
                    "sha256": archive_sha,
                },
                "member": {
                    "path": member_path,
                    "sha256": member_sha,
                    "role": "runtime_seed_settings_authority",
                },
                "defining_authority": True,
                "scientific_result_anchor_claimed": False,
            },
            "status": "ok",
            "problems": [],
        }
        locks[lock_id] = {
            "regime_id": regime_id,
            "nph": int(nph),
            "route_id": ROUTE_APPEND_SINGLETON,
            "archive": {
                "path": archive_path,
                "sha256": archive_sha,
            },
            "member": {
                "path": member_path,
                "sha256": member_sha,
            },
            "resolver_trace": trace,
        }
        authority_rows.append(
            {
                "regime_id": regime_id,
                "nph": int(nph),
                "source_lock_id": lock_id,
                "archive_path": archive_path,
                "archive_sha256": archive_sha,
                "member_path": member_path,
                "member_sha256": member_sha,
                "archive_digest_authority": (
                    "canonical_append_registry_v1"
                ),
                "member_digest_authority": (
                    "canonical_append_registry_v1"
                ),
                "settings_authority": authority,
                "scientific_result_anchor_claimed": False,
            }
        )
    return locks, authority_rows


def _transfer_archive_path(proc_id: int) -> Path:
    return _TRANSFER_ROOT / _TRANSFER_ARCHIVE_NAME[proc_id]


def _transfer_member_path(regime_id: str) -> str:
    return f"{_TRANSFER_MEMBER_PREFIX}/{regime_id}/normalized_run_manifest.json"


def _transfer_job_manifest_path(regime_id: str) -> str:
    return f"{_TRANSFER_JOB_MANIFEST_PREFIX}/{regime_id}.json"


def _read_and_validate_transfer_manifests(
    *,
    baselines: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    regimes = baselines.get("regimes")
    if not isinstance(regimes, Mapping):
        raise MaterializationAuditError(
            "Problem baselines have no regime mapping."
        )
    settings_by_regime: dict[str, dict[str, Any]] = {}
    semantic_rows: list[dict[str, Any]] = []
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        proc_id = _PROC_BY_REGIME[regime_id]
        archive_path = _transfer_archive_path(proc_id)
        member_path = _transfer_member_path(regime_id)
        with tarfile.open(archive_path, "r:gz") as archive:
            member = archive.getmember(member_path)
            handle = archive.extractfile(member)
            if handle is None:
                raise MaterializationAuditError(
                    f"Unable to read {archive_path}:{member_path}."
                )
            raw = handle.read()
        member_sha = hashlib.sha256(raw).hexdigest()
        _assert_equal(
            member_sha,
            _TRANSFER_MEMBER_SHA256[proc_id],
            label=f"9381198.{proc_id} normalized manifest SHA-256",
        )
        payload = json.loads(raw)
        _assert_equal(
            payload.get("schema"),
            "paper_i_hh_sr_symcost_noprune_runtime_manifest_v1",
            label=f"9381198.{proc_id} manifest schema",
        )
        _assert_equal(
            payload.get("job_manifest"),
            _transfer_job_manifest_path(regime_id),
            label=f"9381198.{proc_id} job manifest path",
        )
        _assert_equal(
            payload.get("job_manifest_sha256"),
            _TRANSFER_JOB_MANIFEST_SHA256[proc_id],
            label=f"9381198.{proc_id} job manifest SHA-256",
        )
        physics = payload.get("physics")
        baseline = regimes.get(regime_id)
        baseline_physics = (
            baseline.get("physics")
            if isinstance(baseline, Mapping)
            else None
        )
        if not isinstance(physics, Mapping) or not isinstance(
            baseline_physics, Mapping
        ):
            raise MaterializationAuditError(
                f"Missing physics authority for {regime_id}."
            )
        expected_physics = {
            "L": int(baseline_physics["L"]),
            "t": float(baseline_physics["t"]),
            "u_over_t": float(baseline_physics["u"])
            / float(baseline_physics["t"]),
            "dv": float(baseline_physics["dv"]),
            "omega0": float(baseline_physics["omega0"]),
            "g_ep": float(baseline_physics["g_ep"]),
            "n_ph_work": int(nph),
            "ordering": str(baseline_physics["ordering"]),
            "boundary": str(baseline_physics["boundary"]),
            "same_cutoff_reference": True,
        }
        for field, expected in expected_physics.items():
            _assert_equal(
                physics.get(field),
                expected,
                label=f"9381198.{proc_id} physics.{field}",
            )
        route = payload.get("route_identity")
        profile = (
            route.get("profile_contract")
            if isinstance(route, Mapping)
            else None
        )
        execution = (
            profile.get("execution_settings")
            if isinstance(profile, Mapping)
            else None
        )
        lineage = (
            profile.get("lineage_authority")
            if isinstance(profile, Mapping)
            else None
        )
        if (
            not isinstance(route, Mapping)
            or not isinstance(execution, Mapping)
            or not isinstance(lineage, Mapping)
        ):
            raise MaterializationAuditError(
                f"9381198.{proc_id} has no typed route profile."
            )
        _assert_equal(
            route.get("profile_request"),
            "insertion_commutation_plateau_v1",
            label=f"9381198.{proc_id} source route request",
        )
        _assert_equal(
            execution.get("adapt_insertion_mode"),
            "insertion_commutation_plateau_v1",
            label=f"9381198.{proc_id} insertion mode",
        )
        _assert_equal(
            lineage.get("scientific_result_anchor_claimed"),
            False,
            label=f"9381198.{proc_id} result-anchor claim",
        )
        selected_settings = {
            field: execution[field]
            for field in _RA_REUSED_EXECUTION_FIELDS
        }
        expected_settings = {
            "adapt_final_full_refit": "false",
            "adapt_finite_angle": 0.1,
            "adapt_full_refit_every": 0,
            "adapt_reopt_policy": "windowed",
            "adapt_window_size": 3,
            "adapt_window_topk": 0,
            "phase1_prune_enabled": False,
            "phase2_enable_batching": False,
            "adapt_final_refit_maxiter": 200,
            "adapt_inner_optimizer": "POWELL",
            "adapt_maxiter": 200,
            "adapt_seed": 7,
            "phase3_backend_transpile_seed": 7,
        }
        _assert_equal(
            selected_settings,
            expected_settings,
            label=f"9381198.{proc_id} reused execution settings",
        )
        settings_by_regime[regime_id] = selected_settings
        semantic_rows.append(
            {
                "cluster_id": 9381198,
                "proc_id": proc_id,
                "regime_id": regime_id,
                "nph": int(nph),
                "archive_path": archive_path.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "archive_sha256": _TRANSFER_ARCHIVE_SHA256[proc_id],
                "member_path": member_path,
                "member_sha256": member_sha,
                "job_manifest": payload["job_manifest"],
                "job_manifest_sha256": payload[
                    "job_manifest_sha256"
                ],
                "profile_request": route["profile_request"],
                "profile_resolved": route["profile_resolved"],
                "profile_contract_sha256": route[
                    "profile_contract_sha256"
                ],
                "settings_reused": selected_settings,
                "scientific_result_anchor_claimed": False,
                "status": "passed",
            }
        )
    return settings_by_regime, semantic_rows


def _ra_singleton_locks(
    *,
    settings_by_regime: Mapping[str, Mapping[str, Any]],
    ed_by_regime: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    locks: dict[str, Any] = {}
    authority_rows: list[dict[str, Any]] = []
    for regime_id, nph in CLAIM_FACING_REGIME_CUTOFF_PAIRS:
        proc_id = _PROC_BY_REGIME[regime_id]
        archive_path = _transfer_archive_path(proc_id)
        archive_relative = archive_path.relative_to(REPO_ROOT).as_posix()
        member_path = _transfer_member_path(regime_id)
        member_sha = _TRANSFER_MEMBER_SHA256[proc_id]
        archive_authority = (
            {
                "kind": "predecessor_v10_archive_lock_v1",
                "predecessor_source_locks_path": V10_SOURCE_LOCKS_PATH,
                "predecessor_source_lock_id": source_lock_id(
                    regime_id, 3, "singleton_plateau"
                ),
            }
            if proc_id in {2, 3}
            else {
                "kind": (
                    "first_lock_acquisition_from_retrieved_transfer_bytes_v1"
                ),
                "acquisition_date": "2026-07-28",
                "acquisition_tool": "shasum -a 256",
                "independently_preexisting_expected_hash": False,
            }
        )
        settings_authority = {
            "cluster_id": 9381198,
            "proc_id": proc_id,
            "archive_path": archive_relative,
            "archive_sha256": _TRANSFER_ARCHIVE_SHA256[proc_id],
            "archive_digest_authority": archive_authority,
            "member_path": member_path,
            "member_sha256": member_sha,
            "member_digest_authority": {
                "kind": (
                    "first_lock_acquisition_from_exact_archive_member_bytes_v1"
                ),
                "independently_preexisting_expected_hash": False,
            },
            "job_manifest": _transfer_job_manifest_path(regime_id),
            "job_manifest_sha256": (
                _TRANSFER_JOB_MANIFEST_SHA256[proc_id]
            ),
        }
        for route_id in (
            ROUTE_RA_SINGLETON_APPEND_ONLY,
            ROUTE_RA_SINGLETON_PLATEAU,
            ROUTE_RA_SINGLETON_ALWAYS,
        ):
            lock_id = source_lock_id(regime_id, int(nph), route_id)
            trace = {
                "source_map": (
                    "chtc_cluster_9381198_singleton_plateau_"
                    "normalized_manifests_v1"
                ),
                "regime_or_case": regime_id,
                "method": lock_id,
                "source_json": f"{archive_relative}#{member_path}",
                "source_sha256_expected": member_sha,
                "source_sha256_actual": member_sha,
                "source_sha256_match": True,
                "settings_reused": {
                    "settings": dict(settings_by_regime[regime_id])
                },
                "settings_reused_sources": {
                    "archive_member": {
                        "path": member_path,
                        "sha256": member_sha,
                    },
                    "field_path": (
                        "route_identity.profile_contract."
                        "execution_settings"
                    ),
                },
                "settings_changed": _core_changes(
                    route_id=route_id,
                    source_insertion="plateau_commutation",
                ),
                "same_cutoff_ed_reference": dict(
                    ed_by_regime[regime_id]
                ),
                "core_source_anchor": _core_source_anchor(
                    regime_id=regime_id,
                    nph=int(nph),
                    route_id=route_id,
                    source_insertion="plateau_commutation",
                    authority=settings_authority,
                ),
                "source_archive_member_authority": {
                    **settings_authority,
                    "defining_authority": True,
                    "scientific_result_anchor_claimed": False,
                },
                "status": "ok",
                "problems": [],
            }
            locks[lock_id] = {
                "regime_id": regime_id,
                "nph": int(nph),
                "route_id": route_id,
                "archive": {
                    "path": archive_relative,
                    "sha256": _TRANSFER_ARCHIVE_SHA256[proc_id],
                },
                "member": {
                    "path": member_path,
                    "sha256": member_sha,
                },
                "resolver_trace": trace,
            }
        authority_rows.append(
            {
                **settings_authority,
                "regime_id": regime_id,
                "nph": int(nph),
                "derived_target_route_ids": [
                    ROUTE_RA_SINGLETON_APPEND_ONLY,
                    ROUTE_RA_SINGLETON_PLATEAU,
                    ROUTE_RA_SINGLETON_ALWAYS,
                ],
                "source_route_id": ROUTE_RA_SINGLETON_PLATEAU,
                "source_insertion_policy": "plateau_commutation",
                "scientific_result_anchor_claimed": False,
            }
        )
    return locks, authority_rows


def _build_source_locks(
    *,
    v10_source_locks: Mapping[str, Any],
    registry: Mapping[str, Any],
    baselines: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    _assert_equal(
        v10_source_locks.get("schema"),
        SOURCE_LOCK_SCHEMA,
        label="v10 source-lock schema",
    )
    raw_v10_cells = v10_source_locks.get("cell_locks")
    if not isinstance(raw_v10_cells, Mapping):
        raise MaterializationAuditError(
            "v10 source locks have no cell mapping."
        )
    required_ids = {
        cell.source_lock_id for cell in build_core_cell_specs()
    }
    _assert_equal(
        set(raw_v10_cells),
        required_ids,
        label="v10 stationary-core source-lock id set",
    )
    global_sources = v10_source_locks.get("global_sources")
    if not isinstance(global_sources, Mapping) or not global_sources:
        raise MaterializationAuditError(
            "v10 source locks have no global source mapping."
        )
    source_locks = copy.deepcopy(dict(v10_source_locks))
    macro_locks = {
        lock_id: copy.deepcopy(raw_v10_cells[lock_id])
        for lock_id in sorted(raw_v10_cells)
        if str(raw_v10_cells[lock_id]["route_id"]) in MACRO_ROUTE_IDS
    }
    singleton_locks = {
        lock_id: copy.deepcopy(raw_v10_cells[lock_id])
        for lock_id in sorted(raw_v10_cells)
        if str(raw_v10_cells[lock_id]["route_id"])
        in SINGLETON_CORE_ROUTE_IDS
    }
    _assert_equal(
        (len(macro_locks), len(singleton_locks)),
        (24, 24),
        label="v10 macro/singleton source-lock counts",
    )
    macro_receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_core_macro_lock_reuse_receipt_v1"
            ),
            "status": "passed",
            "predecessor_source_locks": {
                "path": V10_SOURCE_LOCKS_PATH,
                "file_sha256": (
                    EXPECTED_V10_SOURCE_LOCKS_FILE_SHA256
                ),
            },
            "selection_policy": (
                "settled_claim_facing_regime_cutoff_pairs_all_four_"
                "macro_routes_v1"
            ),
            "selected_source_lock_count": len(macro_locks),
            "selected_source_lock_ids": sorted(macro_locks),
            "selected_source_lock_objects_sha256": canonical_sha256(
                macro_locks
            ),
            "byte_semantics": (
                "canonical_json_objects_deep_copied_without_edit_v1"
            ),
            "objects_equal_to_predecessor": all(
                macro_locks[lock_id] == raw_v10_cells[lock_id]
                for lock_id in macro_locks
            ),
        }
    )
    singleton_receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_core_singleton_anchor_derivation_v1"
            ),
            "status": "passed",
            "derivation_mode": (
                "predecessor_v10_exact_source_lock_reuse_v1"
            ),
            "predecessor_source_locks": {
                "path": V10_SOURCE_LOCKS_PATH,
                "file_sha256": (
                    EXPECTED_V10_SOURCE_LOCKS_FILE_SHA256
                ),
            },
            "append_registry": {
                "path": APPEND_REGISTRY_PATH,
                "file_sha256": EXPECTED_APPEND_REGISTRY_FILE_SHA256,
                "verified_unchanged": True,
            },
            "problem_baselines": {
                "path": V10_PROBLEM_BASELINES.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "file_sha256": (
                    EXPECTED_V10_PROBLEM_BASELINES_FILE_SHA256
                ),
            },
            "append_source_lock_count": 6,
            "ra_source_lock_count": 18,
            "singleton_source_lock_count": len(singleton_locks),
            "singleton_source_lock_ids": sorted(singleton_locks),
            "singleton_source_lock_objects_sha256": canonical_sha256(
                singleton_locks
            ),
            "byte_semantics": (
                "canonical_json_objects_deep_copied_without_edit_v1"
            ),
            "objects_equal_to_predecessor": all(
                singleton_locks[lock_id] == raw_v10_cells[lock_id]
                for lock_id in singleton_locks
            ),
            "all_scientific_result_anchor_claims_false": True,
        }
    )
    ra_singleton_rows = [
        {
            "source_lock_id": lock_id,
            "regime_id": lock["regime_id"],
            "nph": lock["nph"],
            "route_id": lock["route_id"],
            "archive": copy.deepcopy(lock["archive"]),
            "member": copy.deepcopy(lock["member"]),
            "predecessor_v10_object_reused_without_edit": True,
            "scientific_result_anchor_claimed": False,
        }
        for lock_id, lock in sorted(singleton_locks.items())
        if str(lock["route_id"]) != ROUTE_APPEND_SINGLETON
    ]
    semantic_receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_core_9381198_member_semantics_v1"
            ),
            "status": "passed",
            "source_role": (
                "predecessor_v10_source_authority_reuse_no_result_"
                "adoption_v1"
            ),
            "cluster_id": 9381198,
            "row_count": len(ra_singleton_rows),
            "rows": ra_singleton_rows,
            "all_scientific_result_anchor_claims_false": True,
        }
    )
    return (
        source_locks,
        macro_receipt,
        singleton_receipt,
        semantic_receipt,
    )


def _run_loader_validation(
    load_root: Path,
    *,
    display_root: Path,
    phase: str,
) -> dict[str, Any]:
    bundle_root = load_root / CORE_BUNDLE_ID
    manifest = _load_digested(
        bundle_root / "bundle_manifest.json",
        label=f"{phase} core bundle manifest",
    )
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise MaterializationAuditError(
            f"{phase} core manifest has no cells."
        )
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise MaterializationAuditError(
                f"{phase} core manifest has a malformed cell."
            )
        cell_id = str(cell["cell_id"])
        load_path = bundle_root / "protocols" / f"{cell_id}.json"
        display_path = (
            display_root
            / CORE_BUNDLE_ID
            / "protocols"
            / f"{cell_id}.json"
        )
        protocol = _load_digested(
            load_path, label=f"{phase} core protocol"
        )
        try:
            loaded = load_validated_bundle_protocol(load_path)
            if loaded.sha256 != protocol["sha256"]:
                raise MaterializationAuditError(
                    f"Loaded protocol digest drifted for {cell_id}."
                )
            status = "passed"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "bundle_id": CORE_BUNDLE_ID,
                "cell_id": cell_id,
                "protocol_path": display_path.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "protocol_sha256": protocol["sha256"],
                "status": status,
                **({"error": error} if error is not None else {}),
            }
        )
    passed = sum(row["status"] == "passed" for row in rows)
    failed = len(rows) - passed
    _assert_equal(
        len(rows), EXPECTED_CELL_COUNT, label=f"{phase} loader total"
    )
    if failed:
        raise MaterializationAuditError(
            f"{phase} loader validation failed {failed}/{len(rows)}."
        )
    return _digested(
        {
            "schema": "ra_adapt_cross_file_loader_validation_v1",
            "status": "passed",
            "phase": phase,
            "validated_root_path": display_root.relative_to(
                REPO_ROOT
            ).as_posix(),
            "loader": (
                "pipelines.static_adapt.ra_adapt.bundles."
                "load_validated_bundle_protocol"
            ),
            "bundle_counts": {
                CORE_BUNDLE_ID: EXPECTED_CELL_COUNT
            },
            "total_count": EXPECTED_CELL_COUNT,
            "passed_count": passed,
            "failed_count": failed,
            "rows": rows,
        }
    )


def _validate_bundle_surface(
    bundle_root: Path,
    *,
    expected_implementation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = bundle_root / "bundle_manifest.json"
    source_path = bundle_root / "source_locks.json"
    expected_path = bundle_root / "expected_artifacts.json"
    validation_path = bundle_root / "validation_report.json"
    manifest = _load_digested(manifest_path, label="core manifest")
    source_locks = _load_digested(
        source_path, label="core source locks"
    )
    expected = _load_digested(
        expected_path, label="core expected artifacts"
    )
    validation = _load_digested(
        validation_path, label="core validation report"
    )
    _assert_equal(
        manifest.get("campaign_id"),
        CORE_CAMPAIGN_ID,
        label="core manifest campaign",
    )
    _assert_equal(
        manifest.get("bundle_id"),
        CORE_BUNDLE_ID,
        label="core manifest bundle",
    )
    _assert_equal(
        manifest.get("run_class"),
        CORE_RUN_CLASS,
        label="core manifest run class",
    )
    _assert_equal(
        manifest.get("visible_target", {}).get("target_id"),
        CORE_VISIBLE_TARGET_ID,
        label="core visible target",
    )
    _assert_equal(
        int(manifest.get("cell_count", -1)),
        EXPECTED_CELL_COUNT,
        label="core manifest cell count",
    )
    _assert_equal(
        int(manifest.get("core_cell_count", -1)),
        EXPECTED_CELL_COUNT,
        label="core manifest core cell count",
    )
    cells = manifest.get("cells")
    if not isinstance(cells, list):
        raise MaterializationAuditError("Core manifest has no cells.")
    _assert_equal(
        [row["cell_id"] for row in cells],
        [cell.cell_id for cell in build_core_cell_specs()],
        label="core ordered cell ids",
    )
    _assert_equal(
        {row["route_id"] for row in cells},
        set(EXPECTED_ROUTE_IDS),
        label="core semantic routes",
    )
    _assert_equal(
        {row["stage"] for row in cells},
        {"core"},
        label="core stages",
    )
    forbidden_manifest = {
        "study1_shared_execution_dedupe",
        "execution_progression_contract",
        "post_study_1_user_decision_required",
        "validation_cell_count",
        "full_cell_count",
    }
    if forbidden_manifest.intersection(manifest):
        raise MaterializationAuditError(
            "Core manifest contains obsolete Study-1 fields."
        )
    for payload, label in (
        (manifest, "manifest"),
        (validation, "validation"),
    ):
        _assert_equal(
            payload.get("execution_authorized"),
            False,
            label=f"core {label} execution authorization",
        )
        _assert_equal(
            payload.get("submission_state"),
            "not_submitted",
            label=f"core {label} submission state",
        )
        _assert_equal(
            payload.get("submitted"),
            False,
            label=f"core {label} submitted flag",
        )
    _assert_equal(
        validation.get("materialization_status"),
        "passed",
        label="core materialization status",
    )
    checks = validation.get("checks")
    if not isinstance(checks, list):
        raise MaterializationAuditError(
            "Core validation report has no checks."
        )
    checks_by_id = {
        str(check["id"]): check
        for check in checks
        if isinstance(check, Mapping)
    }
    _assert_equal(
        set(checks_by_id),
        EXPECTED_CORE_VALIDATION_CHECK_IDS,
        label="core validation check ids",
    )
    if any(
        check.get("status") != "passed"
        for check in checks_by_id.values()
    ):
        raise MaterializationAuditError(
            "Core materialization has a non-passed check."
        )
    binding = validation.get("core_validation_binding")
    if not isinstance(binding, Mapping):
        raise MaterializationAuditError(
            "Core validation binding is missing."
        )
    _assert_equal(
        binding.get("implementation_source_inventory_sha256"),
        expected_implementation_inventory["sha256"],
        label="core validation implementation inventory",
    )
    _assert_equal(
        binding.get("direct_execution_cell_count"),
        EXPECTED_CELL_COUNT,
        label="core direct execution count",
    )
    _assert_equal(
        tuple(binding.get("semantic_route_ids", ())),
        EXPECTED_ROUTE_IDS,
        label="core validation semantic routes",
    )
    _assert_equal(
        source_locks["implementation_sources"],
        expected_implementation_inventory,
        label="core source-lock implementation inventory",
    )
    _assert_equal(
        source_locks.get("required_cell_lock_count"),
        EXPECTED_CELL_COUNT,
        label="core required source lock count",
    )
    _assert_equal(
        source_locks.get("all_required_files_verified"),
        True,
        label="core source verification flag",
    )
    if any(
        lock.get("verification", {}).get(
            "verification_mode"
        )
        != "local_exact_bytes_v1"
        for lock in source_locks["cell_locks"].values()
    ):
        raise MaterializationAuditError(
            "Core contains a source lock not verified from exact bytes."
        )
    protocol_files = sorted(
        (bundle_root / "protocols").glob("*.json")
    )
    template_files = sorted(
        (bundle_root / "execution_templates").glob("*.json")
    )
    _assert_equal(
        len(protocol_files),
        EXPECTED_CELL_COUNT,
        label="core protocol file count",
    )
    _assert_equal(
        len(template_files),
        EXPECTED_CELL_COUNT,
        label="core execution template count",
    )
    direct_count = 0
    for path in template_files:
        template = _load_digested(
            path, label=f"core execution template {path.stem}"
        )
        fulfillment = template.get("execution_fulfillment")
        if (
            isinstance(fulfillment, Mapping)
            and fulfillment.get("fulfillment_kind")
            == "direct_execution_v1"
            and fulfillment.get("canonical_execution")
            == {
                "bundle_id": CORE_BUNDLE_ID,
                "cell_id": path.stem,
            }
        ):
            direct_count += 1
        _assert_equal(
            template.get("execution_authorized"),
            False,
            label=f"{path.stem} execution authorization",
        )
    _assert_equal(
        direct_count,
        EXPECTED_CELL_COUNT,
        label="core all-direct templates",
    )
    expected_cells = expected.get("cells")
    if not isinstance(expected_cells, Mapping):
        raise MaterializationAuditError(
            "Core expected-artifact index has no cells."
        )
    _assert_equal(
        set(expected_cells),
        {row["cell_id"] for row in cells},
        label="core expected-artifact cell ids",
    )
    _assert_equal(
        len(expected_cells),
        EXPECTED_CELL_COUNT,
        label="core expected-artifact cell count",
    )
    return {
        "bundle_id": CORE_BUNDLE_ID,
        "status": "passed",
        "cell_count": EXPECTED_CELL_COUNT,
        "direct_execution_cell_count": direct_count,
        "semantic_route_ids": list(EXPECTED_ROUTE_IDS),
        "manifest": _file_binding(
            manifest_path, relative_to=bundle_root.parent
        ),
        "source_locks": _file_binding(
            source_path, relative_to=bundle_root.parent
        ),
        "expected_artifacts": _file_binding(
            expected_path, relative_to=bundle_root.parent
        ),
        "validation": _file_binding(
            validation_path, relative_to=bundle_root.parent
        ),
        "core_validation_binding_sha256": binding["sha256"],
        "implementation_source_inventory_sha256": (
            expected_implementation_inventory["sha256"]
        ),
        "execution_authorized": False,
        "submission_authorized": False,
        "submission_state": "not_submitted",
        "submitted": False,
    }


def _source_verification_receipt(
    bundle_root: Path,
) -> dict[str, Any]:
    source_locks = _load_digested(
        bundle_root / "source_locks.json",
        label="verified core source locks",
    )
    rows: list[dict[str, Any]] = []
    for lock_id, lock in sorted(source_locks["cell_locks"].items()):
        verification = lock["verification"]
        rows.append(
            {
                "source_lock_id": lock_id,
                "regime_id": lock["regime_id"],
                "nph": lock["nph"],
                "route_id": lock["route_id"],
                "archive": lock["archive"],
                "member": lock["member"],
                "archive_sha256_verified": verification[
                    "archive_sha256_verified"
                ],
                "member_sha256_verified": verification[
                    "member_sha256_verified"
                ],
                "resolver_trace_compatible": verification[
                    "resolver_trace_compatible"
                ],
            }
        )
    _assert_equal(
        len(rows),
        EXPECTED_CELL_COUNT,
        label="source verification row count",
    )
    if not all(
        row["archive_sha256_verified"]
        and row["member_sha256_verified"]
        and row["resolver_trace_compatible"]
        for row in rows
    ):
        raise MaterializationAuditError(
            "A stationary-core source-lock verification did not pass."
        )
    return _digested(
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_source_byte_"
                "verification_v1"
            ),
            "status": "passed",
            "source_locks_sha256": source_locks["sha256"],
            "cell_lock_count": len(rows),
            "unique_archive_count": len(
                {
                    (row["archive"]["path"], row["archive"]["sha256"])
                    for row in rows
                }
            ),
            "unique_member_count": len(
                {
                    (
                        row["archive"]["path"],
                        row["member"]["path"],
                        row["member"]["sha256"],
                    )
                    for row in rows
                }
            ),
            "verification_mode": "local_exact_bytes_v1",
            "rows": rows,
            "all_required_files_verified": True,
        }
    )


def _summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "file_count": int(snapshot["file_count"]),
        "total_size_bytes": int(snapshot["total_size_bytes"]),
        "relative_tree_sha256": str(snapshot["tree_sha256"]),
    }


def _run_source_isolated_public_import_preflight(
    inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy only inventoried sources and import the public seams in isolation."""

    files = inventory.get("files")
    if not isinstance(files, list) or not files:
        raise MaterializationAuditError(
            "Implementation inventory has no source files."
        )
    with tempfile.TemporaryDirectory(
        prefix="ra-adapt-core-source-isolation."
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        source_root = temporary_root / "source"
        source_root.mkdir(parents=True, exist_ok=False)
        copied_rows: list[dict[str, Any]] = []
        for row in files:
            if not isinstance(row, Mapping):
                raise MaterializationAuditError(
                    "Implementation inventory contains a malformed file row."
                )
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise MaterializationAuditError(
                    f"Unsafe implementation inventory path: {relative}."
                )
            source = REPO_ROOT / relative
            destination = source_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            observed = support._hash_file(destination)
            _assert_equal(
                observed,
                row["sha256"],
                label=f"isolated source copy {relative}",
            )
            copied_rows.append(
                {
                    "path": relative.as_posix(),
                    "sha256": observed,
                }
            )

        program = r"""
import json
from pathlib import Path
import sys

source_root = Path(sys.argv[1]).resolve()
ambient_repo = Path(sys.argv[2]).resolve()

def within(path, root):
    try:
        path.resolve().relative_to(root)
        return True
    except (OSError, ValueError):
        return False

sys.path = [
    entry
    for entry in sys.path
    if not within(Path(entry or ".").resolve(), ambient_repo)
]
if str(source_root) not in sys.path:
    sys.path.insert(0, str(source_root))

import pipelines.static_adapt.ra_adapt as ra_adapt
import pipelines.static_adapt.ra_adapt.bundles as bundles
import pipelines.static_adapt.sr_snake as sr_snake
import pipelines.static_adapt.sr_snake.runner as sr_runner

public_callables = {
    "pipelines.static_adapt.ra_adapt.run_append_adapt": (
        callable(ra_adapt.run_append_adapt)
    ),
    "pipelines.static_adapt.ra_adapt.run_ra_adapt": (
        callable(ra_adapt.run_ra_adapt)
    ),
    "pipelines.static_adapt.sr_snake.run_sr_snake": (
        callable(sr_snake.run_sr_snake)
    ),
}
if not all(public_callables.values()):
    raise RuntimeError("A required public RA/SR-SNAKE seam is not callable.")
if len(bundles.build_core_cell_specs()) != 48:
    raise RuntimeError("The isolated core cell builder did not return 48 cells.")

required_modules = (
    "pipelines.static_adapt.ra_adapt",
    "pipelines.static_adapt.ra_adapt.bundles",
    "pipelines.static_adapt.sr_snake",
    "pipelines.static_adapt.sr_snake.runner",
)
module_files = {}
for name in required_modules:
    module = sys.modules[name]
    path = Path(module.__file__).resolve()
    if not within(path, source_root):
        raise RuntimeError(f"{name} escaped isolated source root: {path}")
    module_files[name] = path.relative_to(source_root).as_posix()

ambient_modules = {}
for name, module in sorted(sys.modules.items()):
    value = getattr(module, "__file__", None)
    if value is None:
        continue
    try:
        path = Path(value).resolve()
    except (OSError, TypeError):
        continue
    if within(path, ambient_repo):
        ambient_modules[name] = str(path)
if ambient_modules:
    raise RuntimeError(
        "Repo-local modules loaded from ambient checkout: "
        + json.dumps(ambient_modules, sort_keys=True)
    )

print(json.dumps({
    "status": "passed",
    "public_callables": public_callables,
    "core_cell_count": 48,
    "required_module_files": module_files,
    "ambient_repo_module_count": 0,
}, sort_keys=True))
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source_root)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
        environment["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
        result = subprocess.run(
            (
                sys.executable,
                "-B",
                "-c",
                program,
                str(source_root),
                str(REPO_ROOT),
            ),
            cwd=temporary_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise MaterializationAuditError(
                "Source-isolated public import preflight failed: "
                f"stdout={result.stdout!r}; stderr={result.stderr!r}."
            )
        try:
            observation = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise MaterializationAuditError(
                "Source-isolated public import preflight emitted malformed "
                f"JSON: {result.stdout!r}."
            ) from exc
        _assert_equal(
            observation.get("status"),
            "passed",
            label="source-isolated import status",
        )
        _assert_equal(
            observation.get("ambient_repo_module_count"),
            0,
            label="source-isolated ambient module count",
        )
        _assert_equal(
            observation.get("core_cell_count"),
            EXPECTED_CELL_COUNT,
            label="source-isolated core cell count",
        )
        return _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_source_isolated_public_import_"
                    "preflight_v1"
                ),
                "status": "passed",
                "isolation_policy": (
                    "inventory_only_source_root_no_ambient_repo_modules_v1"
                ),
                "implementation_source_inventory_sha256": (
                    inventory["sha256"]
                ),
                "copied_file_count": len(copied_rows),
                "copied_files_sha256": canonical_sha256(copied_rows),
                "python_flags": ["-B"],
                "cache_policy": {
                    "python_bytecode": "disabled",
                    "hh_pool_cache": "off",
                    "candidate_record_cache": "off",
                },
                "observation": observation,
                "returncode": result.returncode,
            }
        )


def _validate_final_receipt_tree(
    *,
    final_receipt: Mapping[str, Any],
    complete_tree: Mapping[str, Any],
) -> None:
    rows = [
        row
        for row in complete_tree["files"]
        if row["path"] != FINAL_RECEIPT_NAME
    ]
    observed = {
        "scope": (
            "materialization_tree_excluding_"
            "final_publication_receipt_v1"
        ),
        "excluded_paths": [FINAL_RECEIPT_NAME],
        "file_count": len(rows),
        "total_size_bytes": sum(
            int(row["size_bytes"]) for row in rows
        ),
        "relative_tree_sha256": canonical_sha256(rows),
    }
    _assert_equal(
        final_receipt.get("tree"),
        observed,
        label="final receipt excluded-tree binding",
    )


def _build_icm_payload(
    *,
    complete_tree: Mapping[str, Any],
    final_receipt: Mapping[str, Any],
    loader_receipt: Mapping[str, Any],
    implementation_source_inventory_sha256: str,
    final_receipt_file_sha256: str,
) -> dict[str, Any]:
    return _digested(
        {
            "schema": "ra_adapt_icm_stage_receipt_v1",
            "stage": "materialize-stationary-core",
            "state": "complete",
            "status": "passed",
            "campaign_id": CORE_CAMPAIGN_ID,
            "bundle_id": CORE_BUNDLE_ID,
            "selected_policy_authority": {
                "path": CORE_SELECTION_AUTHORITY_PATH,
                "sha256": CORE_SELECTION_AUTHORITY_SHA256,
            },
            "predecessor_v10": {
                "path": V10_ROOT.relative_to(REPO_ROOT).as_posix(),
                "file_count": EXPECTED_V10_FILE_COUNT,
                "total_size_bytes": (
                    EXPECTED_V10_TOTAL_SIZE_BYTES
                ),
                "relative_tree_sha256": (
                    EXPECTED_V10_RELATIVE_TREE_SHA256
                ),
                "final_receipt_file_sha256": (
                    EXPECTED_V10_FINAL_RECEIPT_FILE_SHA256
                ),
                "final_receipt_canonical_sha256": (
                    EXPECTED_V10_FINAL_RECEIPT_CANONICAL_SHA256
                ),
            },
            "materialization": {
                "path": V11_ROOT.relative_to(REPO_ROOT).as_posix(),
                "complete_tree": _summary(complete_tree),
                "final_receipt": {
                    "path": FINAL_RECEIPT_NAME,
                    "canonical_sha256": final_receipt["sha256"],
                    "file_sha256": final_receipt_file_sha256,
                },
            },
            "loader_validation": {
                "path": "cross_file_loader_validation.json",
                "canonical_sha256": loader_receipt["sha256"],
                "total_count": EXPECTED_CELL_COUNT,
                "passed_count": EXPECTED_CELL_COUNT,
                "failed_count": 0,
            },
            "implementation_source_inventory_sha256": (
                implementation_source_inventory_sha256
            ),
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        }
    )


def _complete_missing_icm_receipt() -> int:
    """Safely complete the sole receipt outside an already-published v11."""

    if not V11_ROOT.is_dir() or ICM_RECEIPT.exists():
        raise MaterializationAuditError(
            "The missing-ICM recovery path requires published v11 and no "
            "existing ICM receipt."
        )
    support._darwin_renameatx_np()
    historical = _historical_snapshots()
    _assert_historical_anchors(historical)
    final_path = V11_ROOT / FINAL_RECEIPT_NAME
    final_receipt = _load_digested(
        final_path,
        label="published stationary-core final receipt for ICM recovery",
    )
    _assert_equal(
        final_receipt.get("schema"),
        (
            "paper_i_ra_adapt_stationary_late_core_"
            "materialization_receipt_v1"
        ),
        label="ICM recovery final receipt schema",
    )
    _assert_equal(
        final_receipt.get("publication_status"),
        "passed",
        label="ICM recovery publication status",
    )
    _assert_equal(
        final_receipt.get("authorization"),
        {
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
            "explicit_future_user_authorization_required": True,
        },
        label="ICM recovery authorization",
    )
    source_locks = _load_digested(
        V11_ROOT / CORE_BUNDLE_ID / "source_locks.json",
        label="published core source locks for ICM recovery",
    )
    implementation_inventory = source_locks.get(
        "implementation_sources"
    )
    if not isinstance(implementation_inventory, Mapping):
        raise MaterializationAuditError(
            "Published core has no implementation inventory."
        )
    bundle_summary = _validate_bundle_surface(
        V11_ROOT / CORE_BUNDLE_ID,
        expected_implementation_inventory=implementation_inventory,
    )
    _assert_equal(
        final_receipt.get("bundle_receipt"),
        bundle_summary,
        label="ICM recovery final-to-bundle binding",
    )
    _assert_equal(
        final_receipt.get("implementation_source_inventory"),
        implementation_inventory,
        label="ICM recovery implementation inventory binding",
    )
    loader_receipt = _load_digested(
        V11_ROOT / "cross_file_loader_validation.json",
        label="published loader receipt for ICM recovery",
    )
    _assert_equal(
        (
            loader_receipt.get("total_count"),
            loader_receipt.get("passed_count"),
            loader_receipt.get("failed_count"),
        ),
        (EXPECTED_CELL_COUNT, EXPECTED_CELL_COUNT, 0),
        label="ICM recovery loader counts",
    )
    read_only_loader = _run_loader_validation(
        V11_ROOT,
        display_root=V11_ROOT,
        phase="canonical_post_publish_icm_recovery_read_only",
    )
    _assert_equal(
        read_only_loader["passed_count"],
        EXPECTED_CELL_COUNT,
        label="ICM recovery read-only loader count",
    )
    complete_tree = support._snapshot_roots(
        (V11_ROOT,), relative_to=V11_ROOT
    )
    _validate_final_receipt_tree(
        final_receipt=final_receipt,
        complete_tree=complete_tree,
    )
    icm_payload = _build_icm_payload(
        complete_tree=complete_tree,
        final_receipt=final_receipt,
        loader_receipt=loader_receipt,
        implementation_source_inventory_sha256=str(
            implementation_inventory["sha256"]
        ),
        final_receipt_file_sha256=support._hash_file(final_path),
    )
    support._write_bytes_atomic_no_replace(
        ICM_RECEIPT,
        canonical_json_bytes(icm_payload) + b"\n",
    )
    persisted = _load_digested(
        ICM_RECEIPT,
        label="recovered stationary-core v11 ICM receipt",
    )
    _assert_equal(
        persisted,
        icm_payload,
        label="recovered stationary-core v11 ICM receipt payload",
    )
    print(
        json.dumps(
            {
                "destination": V11_ROOT.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "status": "passed",
                "recovery_action": "published_missing_icm_receipt",
                "icm_receipt": ICM_RECEIPT.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main() -> int:
    if V11_ROOT.exists():
        if not ICM_RECEIPT.exists():
            return _complete_missing_icm_receipt()
        raise FileExistsError(
            f"Refusing to overwrite immutable v11: {V11_ROOT}"
        )
    if ICM_RECEIPT.exists():
        raise FileExistsError(
            f"Refusing to overwrite ICM receipt: {ICM_RECEIPT}"
        )
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise MaterializationAuditError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )
    support._darwin_renameatx_np()

    captured_utc = support._utc_now()
    repository_state = support._repository_state()
    historical_before = _historical_snapshots()
    _assert_historical_anchors(historical_before)
    _assert_equal(
        support._hash_file(V10_SOURCE_LOCKS_INPUT),
        EXPECTED_V10_SOURCE_LOCKS_FILE_SHA256,
        label="v10 source-lock input file SHA-256",
    )
    _assert_equal(
        support._hash_file(V10_PROBLEM_BASELINES),
        EXPECTED_V10_PROBLEM_BASELINES_FILE_SHA256,
        label="v10 problem-baselines file SHA-256",
    )
    _assert_equal(
        support._hash_file(APPEND_REGISTRY),
        EXPECTED_APPEND_REGISTRY_FILE_SHA256,
        label="canonical Append registry file SHA-256",
    )
    selection_authority = REPO_ROOT / CORE_SELECTION_AUTHORITY_PATH
    _assert_equal(
        support._hash_file(selection_authority),
        CORE_SELECTION_AUTHORITY_SHA256,
        label="stationary-core selection authority SHA-256",
    )
    implementation_preflight = _implementation_source_inventory(
        REPO_ROOT
    )
    isolated_import_preflight = (
        _run_source_isolated_public_import_preflight(
            implementation_preflight
        )
    )
    v10_source_locks = _load_mapping(
        V10_SOURCE_LOCKS_INPUT, label="v10 source-lock input"
    )
    baselines = _load_mapping(
        V10_PROBLEM_BASELINES, label="v10 problem baselines"
    )
    registry = _load_mapping(
        APPEND_REGISTRY, label="canonical Append registry"
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".ra_adapt_stationary_late_core_v11.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    source_root = staging_root / "source_materialization"
    source_root.mkdir(parents=True, exist_ok=False)
    support._write_bytes_atomic_no_replace(
        source_root / "problem_baselines.json",
        V10_PROBLEM_BASELINES.read_bytes(),
    )
    _assert_equal(
        support._hash_file(source_root / "problem_baselines.json"),
        EXPECTED_V10_PROBLEM_BASELINES_FILE_SHA256,
        label="copied v11 problem-baselines SHA-256",
    )

    (
        raw_source_locks,
        macro_reuse,
        singleton_derivation,
        transfer_semantics,
    ) = _build_source_locks(
        v10_source_locks=v10_source_locks,
        registry=registry,
        baselines=baselines,
    )
    _write_plain_json(
        source_root / "source_locks_input.json",
        raw_source_locks,
    )
    macro_reuse_receipt = _write_receipt(
        source_root / "macro_lock_reuse_receipt.json",
        {key: value for key, value in macro_reuse.items() if key != "sha256"},
    )
    singleton_derivation_receipt = _write_receipt(
        source_root / "singleton_anchor_derivation_receipt.json",
        {
            key: value
            for key, value in singleton_derivation.items()
            if key != "sha256"
        },
    )
    transfer_semantics_receipt = _write_receipt(
        source_root / "chtc_9381198_member_semantics_receipt.json",
        {
            key: value
            for key, value in transfer_semantics.items()
            if key != "sha256"
        },
    )
    isolated_import_receipt = _write_receipt(
        staging_root / "implementation_source_isolation_receipt.json",
        {
            key: value
            for key, value in isolated_import_preflight.items()
            if key != "sha256"
        },
    )
    preflight_receipt = _write_receipt(
        staging_root / "preflight_receipt.json",
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_v11_preflight_v1"
            ),
            "status": "passed",
            "captured_utc": captured_utc,
            "repository_state": repository_state,
            "campaign_id": CORE_CAMPAIGN_ID,
            "bundle_id": CORE_BUNDLE_ID,
            "stationarity_selection_authority": {
                "path": CORE_SELECTION_AUTHORITY_PATH,
                "sha256": CORE_SELECTION_AUTHORITY_SHA256,
                "verified": True,
            },
            "implementation_inventory": implementation_preflight,
            "implementation_source_isolation": {
                "path": (
                    "implementation_source_isolation_receipt.json"
                ),
                "canonical_sha256": isolated_import_receipt["sha256"],
                "status": "passed",
                "ambient_repo_module_count": 0,
            },
            "predecessor_v10": {
                "path": V10_ROOT.relative_to(REPO_ROOT).as_posix(),
                "relative_tree": {
                    "file_count": EXPECTED_V10_FILE_COUNT,
                    "total_size_bytes": (
                        EXPECTED_V10_TOTAL_SIZE_BYTES
                    ),
                    "relative_tree_sha256": (
                        EXPECTED_V10_RELATIVE_TREE_SHA256
                    ),
                },
                "final_receipt_file_sha256": (
                    EXPECTED_V10_FINAL_RECEIPT_FILE_SHA256
                ),
                "final_receipt_canonical_sha256": (
                    EXPECTED_V10_FINAL_RECEIPT_CANONICAL_SHA256
                ),
            },
            "source_inputs": {
                "v10_source_locks": {
                    "path": V10_SOURCE_LOCKS_PATH,
                    "file_sha256": (
                        EXPECTED_V10_SOURCE_LOCKS_FILE_SHA256
                    ),
                },
                "problem_baselines": {
                    "path": V10_PROBLEM_BASELINES.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "file_sha256": (
                        EXPECTED_V10_PROBLEM_BASELINES_FILE_SHA256
                    ),
                },
                "append_registry": {
                    "path": APPEND_REGISTRY_PATH,
                    "file_sha256": (
                        EXPECTED_APPEND_REGISTRY_FILE_SHA256
                    ),
                },
            },
            "source_derivation_receipts": {
                "macro_lock_reuse_sha256": (
                    macro_reuse_receipt["sha256"]
                ),
                "singleton_anchor_derivation_sha256": (
                    singleton_derivation_receipt["sha256"]
                ),
                "chtc_9381198_member_semantics_sha256": (
                    transfer_semantics_receipt["sha256"]
                ),
            },
            "cell_count": EXPECTED_CELL_COUNT,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )

    receipt = materialize_core_bundle(
        staging_root,
        problem_resolver=support._problem_resolver_from(baselines),
        source_locks=raw_source_locks,
        repository_state=repository_state,
        repo_root=REPO_ROOT,
        horizon=FULL_HORIZON,
        dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
        materialization_timestamp=support._utc_now(),
        verify_source_files=True,
    )
    _assert_equal(
        receipt.bundle_id,
        CORE_BUNDLE_ID,
        label="core materializer bundle id",
    )
    _assert_equal(
        receipt.cell_count,
        EXPECTED_CELL_COUNT,
        label="core materializer cell count",
    )
    _assert_equal(
        receipt.materialization_status,
        "passed",
        label="core materializer status",
    )
    bundle_summary = _validate_bundle_surface(
        staging_root / CORE_BUNDLE_ID,
        expected_implementation_inventory=implementation_preflight,
    )
    source_verification = _source_verification_receipt(
        staging_root / CORE_BUNDLE_ID
    )
    source_verification_receipt = _write_receipt(
        source_root / "source_byte_verification_receipt.json",
        {
            key: value
            for key, value in source_verification.items()
            if key != "sha256"
        },
    )
    staged_loader = _run_loader_validation(
        staging_root,
        display_root=V11_ROOT,
        phase="canonical_path_projected_pre_publish",
    )
    loader_receipt = _write_receipt(
        staging_root / "cross_file_loader_validation.json",
        {
            key: value
            for key, value in staged_loader.items()
            if key != "sha256"
        },
    )
    implementation_after_loader = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_after_loader,
        implementation_preflight,
        label="preflight-to-loader implementation inventory",
    )
    historical_pre_publish = _historical_snapshots()
    _assert_historical_unchanged(
        historical_before,
        historical_pre_publish,
        label="Staged stationary-core v11 materialization",
    )
    final_bundle_summary = _validate_bundle_surface(
        staging_root / CORE_BUNDLE_ID,
        expected_implementation_inventory=implementation_preflight,
    )
    _assert_equal(
        final_bundle_summary,
        bundle_summary,
        label="revalidated staged core bundle summary",
    )
    staged_receipt = _write_receipt(
        staging_root / "staged_materialization_receipt.json",
        {
            "schema": (
                "paper_i_ra_adapt_stationary_core_v11_staged_"
                "materialization_receipt_v1"
            ),
            "status": "passed",
            "campaign_id": CORE_CAMPAIGN_ID,
            "bundle_id": CORE_BUNDLE_ID,
            "cell_count": EXPECTED_CELL_COUNT,
            "preflight_receipt_sha256": preflight_receipt["sha256"],
            "source_byte_verification_receipt_sha256": (
                source_verification_receipt["sha256"]
            ),
            "implementation_source_isolation_receipt_sha256": (
                isolated_import_receipt["sha256"]
            ),
            "loader_validation_receipt_sha256": (
                loader_receipt["sha256"]
            ),
            "bundle_receipt": final_bundle_summary,
            "implementation_inventory": {
                "preflight_sha256": (
                    implementation_preflight["sha256"]
                ),
                "post_loader_sha256": (
                    implementation_after_loader["sha256"]
                ),
                "stable": True,
            },
            "older_materialization_preservation": (
                _preservation_comparison(
                    historical_before, historical_pre_publish
                )
            ),
            "atomic_publish_ready": True,
            "atomic_publish_method": (
                "darwin_renameatx_np_RENAME_EXCL_v1"
            ),
            "atomic_publish_no_replace": True,
            "all_final_files_staged_before_publish": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )
    tree_before_final = support._snapshot_roots(
        (staging_root,), relative_to=staging_root
    )
    final_receipt = _write_receipt(
        staging_root / FINAL_RECEIPT_NAME,
        {
            "schema": (
                "paper_i_ra_adapt_stationary_late_core_"
                "materialization_receipt_v1"
            ),
            "status": "passed",
            "publication_status": "passed",
            "materialization_id": "ra_adapt_stationary_late_core_v11",
            "campaign_id": CORE_CAMPAIGN_ID,
            "run_class": CORE_RUN_CLASS,
            "bundle_id": CORE_BUNDLE_ID,
            "finalized_utc": support._utc_now(),
            "stationarity_selection": {
                "winner_selected": True,
                "active_gradient_policy": (
                    "stationary_source_response_v1"
                ),
                "authority": {
                    "path": CORE_SELECTION_AUTHORITY_PATH,
                    "sha256": CORE_SELECTION_AUTHORITY_SHA256,
                },
            },
            "matrix": {
                "cell_count": EXPECTED_CELL_COUNT,
                "regime_cutoff_pairs": [
                    {"regime_id": regime_id, "nph": int(nph)}
                    for regime_id, nph
                    in CLAIM_FACING_REGIME_CUTOFF_PAIRS
                ],
                "semantic_route_ids": list(EXPECTED_ROUTE_IDS),
                "horizon": FULL_HORIZON,
                "direct_execution_cell_count": EXPECTED_CELL_COUNT,
            },
            "bundle_receipt": final_bundle_summary,
            "source_derivation": {
                "macro_lock_reuse_receipt": _file_binding(
                    source_root / "macro_lock_reuse_receipt.json",
                    relative_to=staging_root,
                ),
                "singleton_anchor_derivation_receipt": _file_binding(
                    source_root
                    / "singleton_anchor_derivation_receipt.json",
                    relative_to=staging_root,
                ),
                "chtc_9381198_member_semantics_receipt": _file_binding(
                    source_root
                    / "chtc_9381198_member_semantics_receipt.json",
                    relative_to=staging_root,
                ),
                "source_byte_verification_receipt": _file_binding(
                    source_root
                    / "source_byte_verification_receipt.json",
                    relative_to=staging_root,
                ),
                "macro_source_lock_count": 24,
                "append_singleton_source_lock_count": 6,
                "ra_singleton_source_lock_count": 18,
                "all_scientific_result_anchor_claims_false": True,
            },
            "validation": {
                "loader_receipt": _file_binding(
                    staging_root
                    / "cross_file_loader_validation.json",
                    relative_to=staging_root,
                ),
                "loader_total_count": EXPECTED_CELL_COUNT,
                "loader_passed_count": EXPECTED_CELL_COUNT,
                "loader_failed_count": 0,
                "core_validation_binding_sha256": (
                    final_bundle_summary[
                        "core_validation_binding_sha256"
                    ]
                ),
                "p3_execution_receipt_required": True,
            },
            "implementation_source_inventory": (
                implementation_preflight
            ),
            "implementation_source_inventory_binding": {
                "source_pointer": (
                    f"{CORE_BUNDLE_ID}/source_locks.json"
                    "#/implementation_sources"
                ),
                "stable": True,
            },
            "implementation_source_isolation": {
                "receipt": _file_binding(
                    staging_root
                    / "implementation_source_isolation_receipt.json",
                    relative_to=staging_root,
                ),
                "ambient_repo_module_count": 0,
                "status": "passed",
            },
            "inventory_snapshot": {
                "repository_state": repository_state,
                "preflight_implementation_sha256": (
                    implementation_preflight["sha256"]
                ),
                "post_loader_implementation_sha256": (
                    implementation_after_loader["sha256"]
                ),
                "stable": True,
            },
            "historical_immutability": {
                "predecessor_v10": {
                    "path": V10_ROOT.relative_to(REPO_ROOT).as_posix(),
                    "file_count": EXPECTED_V10_FILE_COUNT,
                    "total_size_bytes": (
                        EXPECTED_V10_TOTAL_SIZE_BYTES
                    ),
                    "relative_tree_sha256": (
                        EXPECTED_V10_RELATIVE_TREE_SHA256
                    ),
                    "final_receipt_file_sha256": (
                        EXPECTED_V10_FINAL_RECEIPT_FILE_SHA256
                    ),
                    "final_receipt_canonical_sha256": (
                        EXPECTED_V10_FINAL_RECEIPT_CANONICAL_SHA256
                    ),
                },
                "v1_through_v10_preservation": (
                    _preservation_comparison(
                        historical_before, historical_pre_publish
                    )
                ),
            },
            "atomic_publish": {
                "method": "darwin_renameatx_np_RENAME_EXCL_v1",
                "no_replace": True,
                "unsupported_platform_behavior": "fail_closed",
                "all_final_files_staged_before_publish": True,
                "staged_materialization_receipt_sha256": (
                    staged_receipt["sha256"]
                ),
                "post_publish_v11_write_count": 0,
            },
            "tree": {
                "scope": (
                    "materialization_tree_excluding_"
                    "final_publication_receipt_v1"
                ),
                "excluded_paths": [FINAL_RECEIPT_NAME],
                "file_count": int(tree_before_final["file_count"]),
                "total_size_bytes": int(
                    tree_before_final["total_size_bytes"]
                ),
                "relative_tree_sha256": (
                    tree_before_final["tree_sha256"]
                ),
            },
            "authorization": {
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_state": "not_submitted",
                "submitted": False,
                "explicit_future_user_authorization_required": True,
            },
            "execution_authorized": False,
            "submission_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )
    persisted_final = _load_digested(
        staging_root / FINAL_RECEIPT_NAME,
        label="staged final stationary-core receipt",
    )
    _assert_equal(
        persisted_final,
        final_receipt,
        label="staged final stationary-core receipt payload",
    )
    complete_staged_tree = support._snapshot_roots(
        (staging_root,), relative_to=staging_root
    )
    _validate_final_receipt_tree(
        final_receipt=final_receipt,
        complete_tree=complete_staged_tree,
    )
    historical_final_pre_publish = _historical_snapshots()
    _assert_historical_unchanged(
        historical_before,
        historical_final_pre_publish,
        label="Complete staged stationary-core v11 materialization",
    )
    implementation_final_pre_publish = (
        _implementation_source_inventory(REPO_ROOT)
    )
    _assert_equal(
        implementation_final_pre_publish,
        implementation_preflight,
        label="complete-staging implementation inventory",
    )

    icm_payload = _build_icm_payload(
        complete_tree=complete_staged_tree,
        final_receipt=final_receipt,
        loader_receipt=loader_receipt,
        implementation_source_inventory_sha256=(
            implementation_preflight["sha256"]
        ),
        final_receipt_file_sha256=support._hash_file(
            staging_root / FINAL_RECEIPT_NAME
        ),
    )

    support._atomic_rename_no_replace(staging_root, V11_ROOT)

    published_final = _load_digested(
        V11_ROOT / FINAL_RECEIPT_NAME,
        label="published final stationary-core receipt",
    )
    _assert_equal(
        published_final,
        final_receipt,
        label="staged-to-published final stationary-core receipt",
    )
    published_bundle_summary = _validate_bundle_surface(
        V11_ROOT / CORE_BUNDLE_ID,
        expected_implementation_inventory=implementation_preflight,
    )
    _assert_equal(
        published_bundle_summary,
        final_bundle_summary,
        label="staged-to-published core bundle summary",
    )
    post_publish_loader = _run_loader_validation(
        V11_ROOT,
        display_root=V11_ROOT,
        phase="canonical_post_publish_read_only",
    )
    _assert_equal(
        post_publish_loader["passed_count"],
        EXPECTED_CELL_COUNT,
        label="post-publish core loader count",
    )
    historical_after = _historical_snapshots()
    _assert_historical_unchanged(
        historical_before,
        historical_after,
        label="Published stationary-core v11 materialization",
    )
    implementation_post_publish = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_post_publish,
        implementation_preflight,
        label="post-publish implementation inventory",
    )
    published_tree = support._snapshot_roots(
        (V11_ROOT,), relative_to=V11_ROOT
    )
    _assert_equal(
        published_tree,
        complete_staged_tree,
        label="staged-to-published complete v11 tree",
    )

    support._write_bytes_atomic_no_replace(
        ICM_RECEIPT,
        canonical_json_bytes(icm_payload) + b"\n",
    )
    persisted_icm = _load_digested(
        ICM_RECEIPT,
        label="stationary-core v11 ICM stage receipt",
    )
    _assert_equal(
        persisted_icm,
        icm_payload,
        label="stationary-core v11 ICM receipt payload",
    )
    print(
        json.dumps(
            {
                "destination": V11_ROOT.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "status": "passed",
                "bundle_id": CORE_BUNDLE_ID,
                "cell_count": EXPECTED_CELL_COUNT,
                "loader_validation": "48/48",
                "implementation_inventory_sha256": (
                    implementation_preflight["sha256"]
                ),
                "core_validation_binding_sha256": (
                    final_bundle_summary[
                        "core_validation_binding_sha256"
                    ]
                ),
                "final_receipt_sha256": final_receipt["sha256"],
                "final_receipt_file_sha256": support._hash_file(
                    V11_ROOT / FINAL_RECEIPT_NAME
                ),
                "published_tree_sha256": (
                    published_tree["tree_sha256"]
                ),
                "published_file_count": published_tree["file_count"],
                "icm_receipt": ICM_RECEIPT.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "execution_authorized": False,
                "submission_authorized": False,
                "submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
