"""Materialize the repaired stationary Paper-I core as immutable v13.

This command has no scientific-execution or scheduler-submission seam.  It
derives from v12, preserves every archive/member/global source binding and the
problem baselines, and changes only the route-delta metadata of the twelve
``ra_*_always`` cells from the retired raw-full policy to the active
always-commutation-reduced policy.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
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

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    materialize_stationary_core_v12 as v12,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    CLAIM_FACING_REGIME_CUTOFF_PAIRS,
    CORE_BUNDLE_ID,
    CORE_CAMPAIGN_ID,
    CORE_RUN_CLASS,
    CORE_SELECTION_AUTHORITY_PATH,
    CORE_SELECTION_AUTHORITY_SHA256,
    FULL_HORIZON,
    MACRO_ROUTE_IDS,
    SINGLETON_CORE_ROUTE_IDS,
    _implementation_source_inventory,
    materialize_core_bundle,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)


REPO_ROOT = v12.REPO_ROOT
MATERIALIZATIONS_ROOT = v12.MATERIALIZATIONS_ROOT
support = v12.support
V12_ROOT = (
    MATERIALIZATIONS_ROOT / "ra_adapt_stationary_late_core_v12"
)
V13_ROOT = (
    MATERIALIZATIONS_ROOT / "ra_adapt_stationary_late_core_v13"
)
V12_SOURCE_LOCKS_INPUT = (
    V12_ROOT / "source_materialization" / "source_locks_input.json"
)
V12_PROBLEM_BASELINES = (
    V12_ROOT / "source_materialization" / "problem_baselines.json"
)
V12_FINAL_RECEIPT = V12_ROOT / "final_publication_receipt.json"
FINAL_RECEIPT_NAME = "final_publication_receipt.json"

EXPECTED_V12_FILE_COUNT = 111
EXPECTED_V12_TOTAL_SIZE_BYTES = 4_239_521
EXPECTED_V12_RELATIVE_TREE_SHA256 = (
    "2a251ab5e3d7de59c98b19e5e445cd44d776c8a4560b89979c68a6e2bd2153e4"
)
EXPECTED_V12_FINAL_FILE_SHA256 = (
    "9e3579b7bad6c2640f6f119a2f8a6a600b0c0ef00459206dc367b306e50f935b"
)
EXPECTED_V12_FINAL_CANONICAL_SHA256 = (
    "00783a83403c595abca94f5534160e95aba169260e37e0edf73eeeb09e518461"
)
EXPECTED_V12_SOURCE_LOCKS_FILE_SHA256 = (
    "e31a6a0814cdee2d1b4bce7b7fefd18612e0f5d93fe5010f7e48a4e6679acccc"
)
EXPECTED_V12_PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)
EXPECTED_CELL_COUNT = 48
EXPECTED_ALWAYS_CELL_COUNT = 12
EXPECTED_ALWAYS_ROUTE_DELTA_VALUE_COUNT = 18
ALWAYS_INSERTION_POLICY = "always_commutation_reduced"
EXPECTED_ROUTE_IDS = (*MACRO_ROUTE_IDS, *SINGLETON_CORE_ROUTE_IDS)


MaterializationAuditError = v12.MaterializationAuditError


def _assert_equal(actual: Any, expected: Any, *, label: str) -> None:
    if actual != expected:
        raise MaterializationAuditError(
            f"{label} drifted: {actual!r} != {expected!r}."
        )


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    return v12._load_mapping(path, label=label)


def _load_digested(path: Path, *, label: str) -> dict[str, Any]:
    return v12._load_digested(path, label=label)


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return v12._write_receipt(path, payload)


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


def _snapshot_v12() -> dict[str, Any]:
    return support._snapshot_roots((V12_ROOT,), relative_to=V12_ROOT)


def _assert_v12_anchor(snapshot: Mapping[str, Any]) -> None:
    _assert_equal(
        int(snapshot["file_count"]),
        EXPECTED_V12_FILE_COUNT,
        label="v12 immutable file count",
    )
    _assert_equal(
        int(snapshot["total_size_bytes"]),
        EXPECTED_V12_TOTAL_SIZE_BYTES,
        label="v12 immutable total bytes",
    )
    _assert_equal(
        snapshot["tree_sha256"],
        EXPECTED_V12_RELATIVE_TREE_SHA256,
        label="v12 immutable relative tree SHA-256",
    )
    _assert_equal(
        support._hash_file(V12_FINAL_RECEIPT),
        EXPECTED_V12_FINAL_FILE_SHA256,
        label="v12 final receipt file SHA-256",
    )
    _assert_equal(
        _load_digested(
            V12_FINAL_RECEIPT, label="v12 final publication receipt"
        )["sha256"],
        EXPECTED_V12_FINAL_CANONICAL_SHA256,
        label="v12 final receipt canonical SHA-256",
    )


def _scalar_differences(
    before: Any,
    after: Any,
    *,
    path: tuple[str | int, ...] = (),
) -> list[tuple[tuple[str | int, ...], Any, Any]]:
    if isinstance(before, Mapping) and isinstance(after, Mapping):
        if set(before) != set(after):
            return [(path, before, after)]
        result: list[tuple[tuple[str | int, ...], Any, Any]] = []
        for key in sorted(before):
            result.extend(
                _scalar_differences(
                    before[key], after[key], path=(*path, str(key))
                )
            )
        return result
    if isinstance(before, list) and isinstance(after, list):
        if len(before) != len(after):
            return [(path, before, after)]
        result = []
        for index, (left, right) in enumerate(zip(before, after)):
            result.extend(
                _scalar_differences(
                    left, right, path=(*path, index)
                )
            )
        return result
    return [] if before == after else [(path, before, after)]


def _repair_always_route_deltas(
    predecessor: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    repaired = copy.deepcopy(dict(predecessor))
    raw_cells = predecessor.get("cell_locks")
    cells = repaired.get("cell_locks")
    if not isinstance(raw_cells, Mapping) or not isinstance(cells, dict):
        raise MaterializationAuditError(
            "v12 source locks have no mutable cell-lock projection."
        )
    _assert_equal(
        len(cells), EXPECTED_CELL_COUNT, label="v13 source-lock count"
    )

    changed_rows: list[dict[str, Any]] = []
    for lock_id in sorted(cells):
        original = raw_cells[lock_id]
        lock = cells[lock_id]
        if not isinstance(original, Mapping) or not isinstance(lock, dict):
            raise MaterializationAuditError(
                f"Malformed source lock {lock_id!r}."
            )
        _assert_equal(
            lock.get("archive"),
            original.get("archive"),
            label=f"{lock_id} archive binding",
        )
        _assert_equal(
            lock.get("member"),
            original.get("member"),
            label=f"{lock_id} member binding",
        )
        route_id = str(lock.get("route_id"))
        if not route_id.endswith("_always"):
            _assert_equal(
                lock,
                original,
                label=f"{lock_id} non-always source lock",
            )
            continue

        trace = lock.get("resolver_trace")
        if not isinstance(trace, dict):
            raise MaterializationAuditError(
                f"Always source lock {lock_id!r} has no resolver trace."
            )
        changes = trace.get("settings_changed")
        if not isinstance(changes, list):
            raise MaterializationAuditError(
                f"Always source lock {lock_id!r} has no change list."
            )
        insertion_changes = [
            change
            for change in changes
            if isinstance(change, dict)
            and change.get("field") == "insertion_policy"
            and change.get("route_id") == route_id
        ]
        if len(insertion_changes) != 1:
            raise MaterializationAuditError(
                f"Always source lock {lock_id!r} has "
                f"{len(insertion_changes)} insertion route deltas."
            )
        previous_value = insertion_changes[0].get("to")
        insertion_changes[0]["to"] = ALWAYS_INSERTION_POLICY

        anchor_changed = False
        anchor = trace.get("core_source_anchor")
        if route_id == "ra_singleton_always":
            if not isinstance(anchor, dict):
                raise MaterializationAuditError(
                    f"Singleton always source lock {lock_id!r} has no anchor."
                )
            derivation = anchor.get("route_derivation")
            if not isinstance(derivation, dict):
                raise MaterializationAuditError(
                    f"Singleton always source lock {lock_id!r} has no "
                    "route derivation."
                )
            _assert_equal(
                derivation.get("target_insertion_policy"),
                "full_commutation",
                label=f"{lock_id} predecessor target insertion",
            )
            derivation["target_insertion_policy"] = ALWAYS_INSERTION_POLICY
            anchor_changed = True

        differences = _scalar_differences(original, lock)
        allowed = 2 if anchor_changed else 1
        _assert_equal(
            len(differences),
            allowed,
            label=f"{lock_id} route-delta-only scalar changes",
        )
        for path, before, after in differences:
            if after != ALWAYS_INSERTION_POLICY or not (
                (
                    len(path) >= 4
                    and path[0] == "resolver_trace"
                    and path[1] == "settings_changed"
                    and path[-1] == "to"
                )
                or path
                == (
                    "resolver_trace",
                    "core_source_anchor",
                    "route_derivation",
                    "target_insertion_policy",
                )
            ):
                raise MaterializationAuditError(
                    f"Unexpected v13 source-lock change at "
                    f"{lock_id}:{path}: {before!r} -> {after!r}."
                )
        changed_rows.append(
            {
                "source_lock_id": lock_id,
                "route_id": route_id,
                "previous_insertion_policy": previous_value,
                "target_insertion_policy": ALWAYS_INSERTION_POLICY,
                "changed_scalar_paths": [
                    list(path) for path, _, _ in differences
                ],
                "archive_binding_preserved": True,
                "member_binding_preserved": True,
            }
        )

    _assert_equal(
        len(changed_rows),
        EXPECTED_ALWAYS_CELL_COUNT,
        label="v13 repaired always source-lock count",
    )
    _assert_equal(
        sum(len(row["changed_scalar_paths"]) for row in changed_rows),
        EXPECTED_ALWAYS_ROUTE_DELTA_VALUE_COUNT,
        label="v13 repaired route-delta scalar count",
    )
    for key, value in predecessor.items():
        if key != "cell_locks":
            _assert_equal(
                repaired[key],
                value,
                label=f"v13 preserved source-lock root field {key}",
            )
    return repaired, changed_rows


def _tree_binding(
    snapshot: Mapping[str, Any],
    *,
    exclude: Sequence[str] = (),
) -> dict[str, Any]:
    excluded = set(exclude)
    rows = [
        row for row in snapshot["files"] if row["path"] not in excluded
    ]
    return {
        "scope": "materialization_tree_excluding_final_publication_receipt_v1",
        "excluded_paths": list(exclude),
        "file_count": len(rows),
        "total_size_bytes": sum(int(row["size_bytes"]) for row in rows),
        "relative_tree_sha256": canonical_sha256(rows),
    }


def main() -> int:
    if V13_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable v13: {V13_ROOT}"
        )
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise MaterializationAuditError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )

    v12_before = _snapshot_v12()
    _assert_v12_anchor(v12_before)
    _assert_equal(
        support._hash_file(V12_SOURCE_LOCKS_INPUT),
        EXPECTED_V12_SOURCE_LOCKS_FILE_SHA256,
        label="v12 source-lock input file SHA-256",
    )
    _assert_equal(
        support._hash_file(V12_PROBLEM_BASELINES),
        EXPECTED_V12_PROBLEM_BASELINES_FILE_SHA256,
        label="v12 problem-baselines file SHA-256",
    )

    captured_utc = support._utc_now()
    repository_state = support._repository_state()
    implementation = _implementation_source_inventory(REPO_ROOT)
    source_isolation = v12._run_source_isolated_public_import_preflight(
        implementation
    )
    predecessor_locks = _load_mapping(
        V12_SOURCE_LOCKS_INPUT, label="v12 source-lock input"
    )
    repaired_locks, changed_rows = _repair_always_route_deltas(
        predecessor_locks
    )
    baselines = _load_mapping(
        V12_PROBLEM_BASELINES, label="v12 problem baselines"
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".ra_adapt_stationary_late_core_v13.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    try:
        source_root = staging_root / "source_materialization"
        source_root.mkdir(parents=True, exist_ok=False)
        support._write_bytes_atomic_no_replace(
            source_root / "problem_baselines.json",
            V12_PROBLEM_BASELINES.read_bytes(),
        )
        _write_plain_json(
            source_root / "source_locks_input.json", repaired_locks
        )
        route_delta_receipt = _write_receipt(
            source_root / "always_route_delta_receipt.json",
            {
                "schema": (
                    "paper_i_ra_adapt_always_commutation_reduced_"
                    "route_delta_receipt_v1"
                ),
                "status": "passed",
                "predecessor_materialization": (
                    V12_ROOT.relative_to(REPO_ROOT).as_posix()
                ),
                "predecessor_source_locks_file_sha256": (
                    EXPECTED_V12_SOURCE_LOCKS_FILE_SHA256
                ),
                "repaired_source_lock_count": len(changed_rows),
                "changed_scalar_value_count": sum(
                    len(row["changed_scalar_paths"])
                    for row in changed_rows
                ),
                "target_insertion_policy": ALWAYS_INSERTION_POLICY,
                "rows": changed_rows,
                "all_archive_bindings_preserved": True,
                "all_member_bindings_preserved": True,
                "global_source_bindings_preserved": (
                    repaired_locks.get("global_sources")
                    == predecessor_locks.get("global_sources")
                ),
                "problem_baselines_file_sha256": (
                    EXPECTED_V12_PROBLEM_BASELINES_FILE_SHA256
                ),
                "scientific_result_anchor_claimed": False,
            },
        )
        isolation_receipt = _write_receipt(
            staging_root / "implementation_source_isolation_receipt.json",
            {
                key: value
                for key, value in source_isolation.items()
                if key != "sha256"
            },
        )
        preflight_receipt = _write_receipt(
            staging_root / "preflight_receipt.json",
            {
                "schema": (
                    "paper_i_ra_adapt_stationary_core_v13_preflight_v1"
                ),
                "status": "passed",
                "captured_utc": captured_utc,
                "repository_state": repository_state,
                "campaign_id": CORE_CAMPAIGN_ID,
                "bundle_id": CORE_BUNDLE_ID,
                "predecessor_v12": {
                    "path": V12_ROOT.relative_to(REPO_ROOT).as_posix(),
                    "file_count": EXPECTED_V12_FILE_COUNT,
                    "total_size_bytes": EXPECTED_V12_TOTAL_SIZE_BYTES,
                    "relative_tree_sha256": (
                        EXPECTED_V12_RELATIVE_TREE_SHA256
                    ),
                    "final_receipt_file_sha256": (
                        EXPECTED_V12_FINAL_FILE_SHA256
                    ),
                    "final_receipt_canonical_sha256": (
                        EXPECTED_V12_FINAL_CANONICAL_SHA256
                    ),
                },
                "always_route_delta_receipt_sha256": (
                    route_delta_receipt["sha256"]
                ),
                "implementation_source_inventory_sha256": (
                    implementation["sha256"]
                ),
                "implementation_source_isolation_receipt_sha256": (
                    isolation_receipt["sha256"]
                ),
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_state": "not_submitted",
                "submitted": False,
            },
        )

        receipt = materialize_core_bundle(
            staging_root,
            problem_resolver=support._problem_resolver_from(baselines),
            source_locks=repaired_locks,
            repository_state=repository_state,
            repo_root=REPO_ROOT,
            horizon=FULL_HORIZON,
            dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
            materialization_timestamp=support._utc_now(),
            verify_source_files=True,
        )
        _assert_equal(
            (receipt.bundle_id, receipt.cell_count, receipt.materialization_status),
            (CORE_BUNDLE_ID, EXPECTED_CELL_COUNT, "passed"),
            label="v13 core materialization result",
        )
        bundle_summary = v12._validate_bundle_surface(
            staging_root / CORE_BUNDLE_ID,
            expected_implementation_inventory=implementation,
        )
        source_verification = v12._source_verification_receipt(
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
        loader = v12._run_loader_validation(
            staging_root,
            display_root=V13_ROOT,
            phase="canonical_path_projected_pre_publish",
        )
        loader_receipt = _write_receipt(
            staging_root / "cross_file_loader_validation.json",
            {
                key: value
                for key, value in loader.items()
                if key != "sha256"
            },
        )
        _assert_equal(
            _implementation_source_inventory(REPO_ROOT),
            implementation,
            label="v13 stable implementation inventory",
        )
        _assert_equal(
            _snapshot_v12(),
            v12_before,
            label="v12 immutability before v13 publish",
        )

        staged_receipt = _write_receipt(
            staging_root / "staged_materialization_receipt.json",
            {
                "schema": (
                    "paper_i_ra_adapt_stationary_core_v13_staged_"
                    "materialization_receipt_v1"
                ),
                "status": "passed",
                "campaign_id": CORE_CAMPAIGN_ID,
                "bundle_id": CORE_BUNDLE_ID,
                "cell_count": EXPECTED_CELL_COUNT,
                "preflight_receipt_sha256": preflight_receipt["sha256"],
                "bundle_receipt": bundle_summary,
                "always_route_delta_receipt_sha256": (
                    route_delta_receipt["sha256"]
                ),
                "source_byte_verification_receipt_sha256": (
                    source_verification_receipt["sha256"]
                ),
                "loader_validation_receipt_sha256": (
                    loader_receipt["sha256"]
                ),
                "atomic_publish_ready": True,
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
                "materialization_id": (
                    "ra_adapt_stationary_late_core_v13"
                ),
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
                "bundle_receipt": bundle_summary,
                "source_derivation": {
                    "predecessor_v12_source_locks": {
                        "path": V12_SOURCE_LOCKS_INPUT.relative_to(
                            REPO_ROOT
                        ).as_posix(),
                        "file_sha256": (
                            EXPECTED_V12_SOURCE_LOCKS_FILE_SHA256
                        ),
                    },
                    "always_route_delta_receipt": _file_binding(
                        source_root / "always_route_delta_receipt.json",
                        relative_to=staging_root,
                    ),
                    "source_byte_verification_receipt": _file_binding(
                        source_root / "source_byte_verification_receipt.json",
                        relative_to=staging_root,
                    ),
                    "always_source_lock_count": (
                        EXPECTED_ALWAYS_CELL_COUNT
                    ),
                    "changed_scalar_value_count": (
                        EXPECTED_ALWAYS_ROUTE_DELTA_VALUE_COUNT
                    ),
                    "all_underlying_source_bindings_preserved": True,
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
                    "core_validation_binding_sha256": bundle_summary[
                        "core_validation_binding_sha256"
                    ],
                    "two_round_package_smoke_required": True,
                },
                "implementation_source_inventory": implementation,
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
                        implementation["sha256"]
                    ),
                    "post_loader_implementation_sha256": (
                        implementation["sha256"]
                    ),
                    "stable": True,
                },
                "historical_immutability": {
                    "predecessor_v12": {
                        "path": V12_ROOT.relative_to(REPO_ROOT).as_posix(),
                        "file_count": EXPECTED_V12_FILE_COUNT,
                        "total_size_bytes": EXPECTED_V12_TOTAL_SIZE_BYTES,
                        "relative_tree_sha256": (
                            EXPECTED_V12_RELATIVE_TREE_SHA256
                        ),
                        "final_receipt_file_sha256": (
                            EXPECTED_V12_FINAL_FILE_SHA256
                        ),
                        "final_receipt_canonical_sha256": (
                            EXPECTED_V12_FINAL_CANONICAL_SHA256
                        ),
                    },
                    "pre_publish_unchanged": True,
                },
                "atomic_publish": {
                    "method": "darwin_renameatx_np_RENAME_EXCL_v1",
                    "no_replace": True,
                    "unsupported_platform_behavior": "fail_closed",
                    "all_final_files_staged_before_publish": True,
                    "staged_materialization_receipt_sha256": (
                        staged_receipt["sha256"]
                    ),
                    "post_publish_v13_write_count": 0,
                },
                "tree": _tree_binding(
                    tree_before_final, exclude=(FINAL_RECEIPT_NAME,)
                ),
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
        complete_tree = support._snapshot_roots(
            (staging_root,), relative_to=staging_root
        )
        v12._validate_final_receipt_tree(
            final_receipt=final_receipt,
            complete_tree=complete_tree,
        )
        _assert_equal(
            _snapshot_v12(),
            v12_before,
            label="v12 immutability at v13 publication",
        )

        support._darwin_renameatx_np()
        support._atomic_rename_no_replace(staging_root, V13_ROOT)
        published = _load_digested(
            V13_ROOT / FINAL_RECEIPT_NAME,
            label="published v13 final publication receipt",
        )
        _assert_equal(
            published,
            final_receipt,
            label="staged-to-published v13 final receipt",
        )
        _assert_equal(
            v12._validate_bundle_surface(
                V13_ROOT / CORE_BUNDLE_ID,
                expected_implementation_inventory=implementation,
            ),
            bundle_summary,
            label="published v13 bundle surface",
        )
        _assert_equal(
            _snapshot_v12(),
            v12_before,
            label="v12 immutability after v13 publication",
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "materialization_root": V13_ROOT.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "final_receipt_sha256": final_receipt["sha256"],
                    "cell_count": EXPECTED_CELL_COUNT,
                    "always_source_lock_count": (
                        EXPECTED_ALWAYS_CELL_COUNT
                    ),
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submission_state": "not_submitted",
                    "submitted": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
