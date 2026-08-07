"""Atomically materialize and audit immutable Paper-I Study-1 v8 bundles.

This append-only command has no scientific execution or scheduler seam.  It:

* verifies the exact immutable v1-v7 materialization anchors;
* byte-copies v7's inherited source-materialization tree;
* inventories the current RA implementation import closure;
* materializes both 58-cell Study-1 bundles with the v2 objective gates;
* authenticates repeat-aware candidate-position ranking, the selected
  retained-parent identity/label, accepted-registry immutability, and signed
  hard-guard resume hydration;
* creates compact, self-digested G1/G2/G3/G8/G13 authority;
* validates every one of the 116 protocol loader bindings;
* atomically publishes v8 without replacing any existing path; and
* writes a final self-digested receipt that binds all of the above.

The command deliberately does not execute a cell, authorize execution, submit
a scheduler job, select a stationarity policy, remove failed staging trees, or
overwrite any prior materialization.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

# Direct path execution sets ``sys.path[0]`` to this campaign directory.
# Bootstrap the active repository before importing its namespace packages.
_BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_BOOTSTRAP_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_REPO_ROOT))

from chtc.paper_i_ra_adapt_repair_20260727 import (
    materialize_study1_v7 as v7,
)
from pipelines.static_adapt.ra_adapt.adapters import (
    SinglePauliWordCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt.bundles import (
    OBJECTIVE_EXECUTION_GATE_IDS,
    STUDY1_BUNDLE_POLICIES,
    VALIDATION_REGIMES,
    VALIDATION_ROUTE_IDS,
    _hash_archive_member,
    _implementation_source_inventory,
    load_validated_bundle_protocol,
    materialize_study1_bundles,
    preservation_execution_gate_contract,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.exact_reference_isolation import (
    STUDY1_TRUSTED_EXECUTION_SCHEMA,
    build_study1_trusted_execution_receipt,
    validate_study1_trusted_execution_receipt,
)


REPO_ROOT = v7.REPO_ROOT
CAMPAIGN_ROOT = v7.CAMPAIGN_ROOT
MATERIALIZATIONS_ROOT = v7.MATERIALIZATIONS_ROOT
support = v7.v6
V7_ROOT = v7.V7_ROOT
V8_ROOT = (
    MATERIALIZATIONS_ROOT / "ra_adapt_unification_post_refactor_v8"
)
V7_SOURCE_ROOT = V7_ROOT / "source_materialization"
SOURCE_LOCKS_INPUT = V7_SOURCE_ROOT / "source_locks_input.json"
PROBLEM_BASELINES = V7_SOURCE_ROOT / "problem_baselines.json"
V7_FINAL_RECEIPT = V7_ROOT / "final_materialization_receipt.json"

PACKAGE_ID = "paper_i_ra_adapt_study1_minimal_20260728_v3_chtc"
AUTHORITY_NAME = "study1_objective_gate_authority_receipt.json"
AUTHORITY_SCHEMA = (
    "paper_i_ra_adapt_study1_objective_gate_authority_v3"
)
SOURCE_LOCK_CELL_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_source_lock_cell_receipt_v2"
)
POOL_CONSTRUCTION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_pool_construction_receipt_v2"
)
TRUSTED_EXECUTION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_study1_trusted_execution_receipt_v2"
)
T13_CHARACTERIZATION_RECEIPT_SCHEMA = (
    "paper_i_ra_adapt_t13_characterization_receipt_v2"
)

EXPECTED_V7_TREE_SHA256 = (
    "a85daf852b9b2d934834d9e1a8ff48140304bfa1cf53381da568b653c52a2f0e"
)
EXPECTED_V7_RELATIVE_TREE_SHA256 = (
    "32397770a3fc903876cf467393f8d282d9fd814b84cc02b354dbb618c0bfcbae"
)
EXPECTED_V7_FILE_COUNT = 375
EXPECTED_V7_TOTAL_SIZE_BYTES = 13_592_188
EXPECTED_V7_FINAL_RECEIPT_FILE_SHA256 = (
    "891856824aa7f3e7874859b75ef5a902ffa304de2479c0f8ed29a60745de2190"
)
EXPECTED_V7_FINAL_RECEIPT_CANONICAL_SHA256 = (
    "12374acfc5989541a56194868fe938e330fe7ec371cb6a658bd53bc01a01be2d"
)
EXPECTED_V7_SOURCE_TREE_SHA256 = (
    "08ac8303870ba206493859e2037e53f0e32f758f235c06084681b149073e7c26"
)
EXPECTED_V7_SOURCE_FILE_COUNT = 126

EXPECTED_LOGICAL_AUTHORITY_CELL_COUNT = 20
EXPECTED_PROTOCOL_COUNT = 116
EXPECTED_MACRO_MEMBERSHIP = {
    "count": 102,
    "ordered_labels_sha256": (
        "a8831528590e870a09ce08492b6f61da4a4d377e63fa8983b30ca9698af5d3d9"
    ),
}
EXPECTED_SINGLETON_PARENT_MEMBERSHIP = {
    "count": 123,
    "ordered_labels_sha256": (
        "17cc97b744f8e6b50b686b24edd28426ca2c055bc2c31054fd353ddfa10efbe3"
    ),
}

T13_FIXTURE = REPO_ROOT / "test/fixtures/ra_adapt_singleton_trajectory_nph3.json"
T13_PROBLEM_REQUEST_SHA256 = (
    "b7299ce9e978abc1f5c2db8b11328dbe2df4f679be1abc4d6aa4da3fc0159c53"
)
T13_FIXTURE_FILE_SHA256 = (
    "722ff9e3503c46a577d18ef9206b5914b8ad7a5b965a6510e42e69ed645ac220"
)
T13_FIXTURE_CANONICAL_SHA256 = (
    "df5e73b94f0abea0e3d37b0b8c1c00eff15ecdd357dde670c72b9d4f2ca1bd67"
)

REQUIRED_IMPLEMENTATION_PATHS = {
    "pipelines/static_adapt/adapt_pipeline.py",
    "pipelines/static_adapt/numerical_physical_integrity.py",
    "pipelines/static_adapt/ra_adapt/__init__.py",
    "pipelines/static_adapt/ra_adapt/append.py",
    "pipelines/static_adapt/ra_adapt/bundles.py",
    "pipelines/static_adapt/ra_adapt/contracts.py",
    "pipelines/static_adapt/ra_adapt/engine.py",
    "pipelines/static_adapt/ra_adapt/exact_reference_isolation.py",
    "pipelines/static_adapt/ra_adapt/replay_evidence.py",
    "pipelines/static_adapt/ra_adapt/runtime.py",
    "pipelines/static_adapt/sr_snake/_context.py",
    "pipelines/static_adapt/sr_snake/_controller.py",
    "pipelines/static_adapt/sr_snake/_observation.py",
    "pipelines/static_adapt/sr_snake/_resume.py",
    "pipelines/static_adapt/sr_snake/_selection.py",
    "pipelines/static_adapt/sr_snake/_transition.py",
    "pipelines/static_adapt/sr_snake/contracts.py",
}

REPAIR_SOURCE_CONTRACTS = {
    "pipelines/static_adapt/adapt_pipeline.py": {
        "repeat_aware_pool_identity": (
            "def _default_no_prune_pool_entry_identity(",
            'return f"{str(generator_id)}::pool[{int(pool_index)}]"',
        ),
        "repeat_aware_scored_population": (
            '(row["domain_record_id"], row["generator_id"])',
            '"paper_i_scored_insertion_position_population_v1"',
        ),
        "actual_retained_owner_binding": (
            "def _authenticated_singleton_owner_parent_label(",
            "def _retained_parent_owner_receipt(",
            '"ra_adapt_retained_parent_owner_v1"',
            '"candidate_manifest_sha256"',
        ),
        "accepted_registry_immutability": (
            '"An accepted RA child lost its immutable "',
            '"retained-parent registry binding."',
            '"An accepted RA singleton child lost its "',
            '"retained-parent registry receipt."',
        ),
        "signed_hard_guard_hydration": (
            '"ra_retained_parent_owner": (',
            '"route_a_child_padding_lineage": (',
            'for key in (',
            '"runtime_split",',
            '"symmetry_gate",',
            '"route_a_child_padding_lineage",',
        ),
    },
    "pipelines/static_adapt/sr_snake/_resume.py": {
        "retained_owner_history_authentication": (
            '"ra_adapt_retained_parent_owner_v1"',
            '"parent_generator_identity"',
            '"candidate_manifest_sha256"',
            '"owner is not bound to its authenticated child, "',
        ),
        "retained_owner_signed_prefix_closure": (
            '"ra_retained_parent_owner"',
            "selected retained-parent ",
            '"owner does not close to its signed active prefix."',
        ),
    },
    "pipelines/static_adapt/sr_snake/_selection.py": {
        "candidate_position_rank_identity": (
            "return record.domain_record_id, record.generator_id",
            "shortlist rank changed its source pool index",
            "shortlist rank changed its insertion position",
        ),
        "macro_identity_position_rank": (
            "macro identities are not contiguous and one-based",
            "macro-identity position ranks are incomplete",
            "macro-identity positions are not score ranked",
            "macro representatives are not score ranked",
        ),
    },
}


MaterializationAuditError = v7.MaterializationAuditError


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    payload = v7._load_mapping(path, label=label)
    if not isinstance(payload, dict):
        raise MaterializationAuditError(f"{label} must be a JSON object.")
    return payload


def _load_canonical_digested(path: Path, *, label: str) -> dict[str, Any]:
    return v7._load_canonical_digested(path, label=label)


def _sha256_file(path: Path) -> str:
    return support._hash_file(path)


def _summary(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return support._snapshot_summary(snapshot)


def _assert_equal(actual: Any, expected: Any, *, label: str) -> None:
    v7._assert_equal(actual, expected, label=label)


def _historical_snapshots() -> dict[str, dict[str, Any]]:
    snapshots = v7._historical_snapshots()
    snapshots["v7"] = support._snapshot_roots((V7_ROOT,))
    return snapshots


def _assert_historical_anchors(
    snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    v7._assert_historical_anchors(snapshots)
    observed = snapshots.get("v7")
    if not isinstance(observed, Mapping):
        raise MaterializationAuditError("Missing immutable v7 snapshot.")
    expected = {
        "file_count": EXPECTED_V7_FILE_COUNT,
        "total_size_bytes": EXPECTED_V7_TOTAL_SIZE_BYTES,
        "tree_sha256": EXPECTED_V7_TREE_SHA256,
    }
    for key, value in expected.items():
        _assert_equal(observed.get(key), value, label=f"v7 immutable {key}")
    relative_snapshot = support._snapshot_roots(
        (V7_ROOT,),
        relative_to=V7_ROOT,
    )
    _assert_equal(
        relative_snapshot.get("tree_sha256"),
        EXPECTED_V7_RELATIVE_TREE_SHA256,
        label="v7 immutable relative-root tree SHA-256",
    )
    _assert_equal(
        _sha256_file(V7_FINAL_RECEIPT),
        EXPECTED_V7_FINAL_RECEIPT_FILE_SHA256,
        label="v7 final receipt file SHA-256",
    )
    final = _load_canonical_digested(
        V7_FINAL_RECEIPT, label="v7 final materialization receipt"
    )
    _assert_equal(
        final["sha256"],
        EXPECTED_V7_FINAL_RECEIPT_CANONICAL_SHA256,
        label="v7 final receipt canonical SHA-256",
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


def _copy_v7_source_materialization(
    staging_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_before = support._snapshot_roots(
        (V7_SOURCE_ROOT,), relative_to=V7_SOURCE_ROOT
    )
    _assert_equal(
        source_before["file_count"],
        EXPECTED_V7_SOURCE_FILE_COUNT,
        label="v7 source-materialization file count",
    )
    _assert_equal(
        source_before["tree_sha256"],
        EXPECTED_V7_SOURCE_TREE_SHA256,
        label="v7 source-materialization tree SHA-256",
    )
    destination = staging_root / "source_materialization"
    shutil.copytree(
        V7_SOURCE_ROOT,
        destination,
        copy_function=shutil.copy2,
    )
    source_after = support._snapshot_roots(
        (destination,), relative_to=destination
    )
    _assert_equal(
        source_after,
        source_before,
        label="v7-to-v8 byte-identical source inheritance",
    )
    return source_before, source_after


def _inventory_rows(
    inventory: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = inventory.get("files")
    if not isinstance(raw, list):
        raise MaterializationAuditError(
            "Implementation inventory has no ordered files."
        )
    rows: dict[str, Mapping[str, Any]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise MaterializationAuditError(
                "Implementation inventory contains a non-object row."
            )
        path = str(item.get("path", ""))
        if not path or path in rows:
            raise MaterializationAuditError(
                f"Implementation inventory path is missing or duplicate: {path!r}."
            )
        rows[path] = item
    _assert_equal(
        len(rows),
        int(inventory.get("file_count", -1)),
        label="implementation inventory file count",
    )
    missing = sorted(REQUIRED_IMPLEMENTATION_PATHS.difference(rows))
    if missing:
        raise MaterializationAuditError(
            f"Current implementation inventory misses required sources: {missing}."
        )
    return rows


def _validate_repair_source_contracts(
    implementation_inventory: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Bind the exact source files and fragments that distinguish v8."""

    inventory_by_path = _inventory_rows(implementation_inventory)
    rows: list[dict[str, Any]] = []
    for relative_path, contracts in sorted(
        REPAIR_SOURCE_CONTRACTS.items()
    ):
        source_path = _safe_repo_file(
            relative_path,
            label=f"v8 repair source {relative_path}",
        )
        inventory_row = inventory_by_path.get(relative_path)
        if not isinstance(inventory_row, Mapping):
            raise MaterializationAuditError(
                f"v8 repair source is outside the implementation inventory: "
                f"{relative_path}."
            )
        observed_file_sha256 = _sha256_file(source_path)
        _assert_equal(
            inventory_row.get("sha256"),
            observed_file_sha256,
            label=f"{relative_path} inventory SHA-256",
        )
        source_text = source_path.read_text(encoding="utf-8")
        contract_rows: list[dict[str, Any]] = []
        for contract_id, required_fragments in sorted(contracts.items()):
            missing = [
                fragment
                for fragment in required_fragments
                if fragment not in source_text
            ]
            if missing:
                raise MaterializationAuditError(
                    f"{relative_path} does not satisfy {contract_id}: "
                    f"missing fragments={missing!r}."
                )
            contract_rows.append(
                {
                    "contract_id": contract_id,
                    "required_fragments": list(required_fragments),
                    "fragment_count": len(required_fragments),
                    "status": "passed",
                }
            )
        rows.append(
            {
                "path": relative_path,
                "file_sha256": observed_file_sha256,
                "contract_count": len(contract_rows),
                "contracts": contract_rows,
                "status": "passed",
            }
        )
    return rows


def _pool_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    projection = {
        "count": int(value["count"]),
        "ordered_labels_sha256": str(value["ordered_labels_sha256"]),
        "ordered_pool_sha256": str(value["ordered_pool_sha256"]),
    }
    for field in ("ordered_labels_sha256", "ordered_pool_sha256"):
        if (
            len(projection[field]) != 64
            or any(char not in "0123456789abcdef" for char in projection[field])
        ):
            raise MaterializationAuditError(
                f"Pool projection has an invalid {field}."
            )
    return projection


def _protocol_path(
    root: Path,
    *,
    bundle_id: str,
    cell_id: str,
) -> Path:
    return root / bundle_id / "protocols" / f"{cell_id}.json"


def _validation_cell_id(regime_id: str, route_id: str) -> str:
    return f"validation__{regime_id}__nph3__{route_id}"


def _validate_v2_gate_semantics(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    expected_ids = list(OBJECTIVE_EXECUTION_GATE_IDS)
    for bundle_id, active_gradient_policy in STUDY1_BUNDLE_POLICIES:
        manifest = _load_canonical_digested(
            root / bundle_id / "bundle_manifest.json",
            label=f"{bundle_id} manifest",
        )
        validation = _load_canonical_digested(
            root / bundle_id / "validation_report.json",
            label=f"{bundle_id} validation report",
        )
        gates = validation.get("objective_execution_gates")
        if not isinstance(gates, list):
            raise MaterializationAuditError(
                f"{bundle_id} has no objective execution gates."
            )
        _assert_equal(
            [gate.get("id") for gate in gates],
            expected_ids,
            label=f"{bundle_id} v2 objective gate IDs",
        )
        if any(
            gate.get("status") != "not_run"
            or gate.get("blocks_full_matrix") is not True
            for gate in gates
        ):
            raise MaterializationAuditError(
                f"{bundle_id} objective gates are not fail-closed."
            )
        expected_preservation = preservation_execution_gate_contract(
            active_gradient_policy=active_gradient_policy
        )
        cells = manifest.get("cells")
        if not isinstance(cells, list):
            raise MaterializationAuditError(
                f"{bundle_id} manifest has no cell list."
            )
        singleton_rows = [
            row
            for row in cells
            if (
                isinstance(row, Mapping)
                and row.get("stage") == "validation"
                and row.get("route_id") == "singleton_plateau"
            )
        ]
        _assert_equal(
            len(singleton_rows),
            2,
            label=f"{bundle_id} singleton preservation row count",
        )
        for cell in singleton_rows:
            _assert_equal(
                cell.get("preservation_execution_gate"),
                expected_preservation,
                label=(
                    f"{bundle_id}::{cell.get('cell_id')} v2 preservation gate"
                ),
            )
        rows.append(
            {
                "bundle_id": bundle_id,
                "active_gradient_policy": active_gradient_policy,
                "objective_gate_ids": expected_ids,
                "preservation_gate_id": expected_preservation["gate_id"],
                "preservation_gate_sha256": expected_preservation["sha256"],
                "same_problem_deterministic_replay_required": True,
                "paired_policy_comparison_required": True,
                "trajectory_deviation_is_pass_condition": False,
                "generic_t13_study1_numerical_baseline": False,
                "singleton_validation_cell_count": len(singleton_rows),
            }
        )
    return _digested(
        {
            "schema": "ra_adapt_study1_v2_gate_semantics_validation_v1",
            "status": "passed",
            "objective_gate_ids": expected_ids,
            "correction_class": (
                "approved_v2_objective_gate_and_same_physics_preservation_v1"
            ),
            "rows": rows,
            "execution_authorized": False,
            "submitted": False,
        }
    )


def _validate_v7_v8_repair_semantics(
    root: Path,
    *,
    implementation_inventory: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate the source-level repair while preserving v2 gates."""

    gate_validation = _validate_v2_gate_semantics(root)
    source_contracts = _validate_repair_source_contracts(
        implementation_inventory
    )
    return _digested(
        {
            "schema": "ra_adapt_v7_v8_repair_semantics_receipt_v1",
            "status": "passed",
            "baseline_revision": "v7",
            "target_revision": "v8",
            "correction_class": (
                "repeat_aware_selection_owner_authenticated_resume_"
                "hydration_v1"
            ),
            "implementation_inventory_sha256": (
                implementation_inventory["sha256"]
            ),
            "required_implementation_paths": sorted(
                REQUIRED_IMPLEMENTATION_PATHS
            ),
            "source_contracts": source_contracts,
            "repair_contracts": {
                "repeat_aware_candidate_position_identity": True,
                "candidate_position_and_macro_rank_authentication": True,
                "signed_actual_retained_owner_identity_and_label": True,
                "coherent_alternate_ancestor_tamper_rejected": True,
                "hard_guard_metadata_hydrated_from_signed_prefix": True,
                "accepted_registry_binding_is_immutable": True,
            },
            "objective_gate_validation": gate_validation,
            "execution_authorized": False,
            "submitted": False,
        }
    )


def _safe_repo_file(relative: Any, *, label: str) -> Path:
    raw = str(relative)
    candidate = (REPO_ROOT / raw).resolve()
    try:
        candidate.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise MaterializationAuditError(
            f"{label} escapes the active repository: {raw!r}."
        ) from exc
    if not candidate.is_file() or candidate.is_symlink():
        raise MaterializationAuditError(
            f"{label} is not a regular in-repository file: {raw!r}."
        )
    return candidate


def _build_g1_g2_cell_receipts(
    root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    archive_hash_cache: dict[Path, str] = {}
    member_hash_cache: dict[tuple[Path, str], str] = {}

    def rehash_archive(path: Path) -> str:
        if path not in archive_hash_cache:
            archive_hash_cache[path] = _sha256_file(path)
        return archive_hash_cache[path]

    def rehash_member(path: Path, member_path: str) -> str:
        key = (path, member_path)
        if key not in member_hash_cache:
            member_hash_cache[key] = _hash_archive_member(
                path, member_path
            )
        return member_hash_cache[key]

    for bundle_id, _policy in STUDY1_BUNDLE_POLICIES:
        source_locks = _load_canonical_digested(
            root / bundle_id / "source_locks.json",
            label=f"{bundle_id} source locks",
        )
        cell_locks = source_locks.get("cell_locks")
        if not isinstance(cell_locks, Mapping):
            raise MaterializationAuditError(
                f"{bundle_id} source locks have no cell rows."
            )
        for regime_id in VALIDATION_REGIMES:
            for route_id in VALIDATION_ROUTE_IDS:
                cell_id = _validation_cell_id(regime_id, route_id)
                protocol = _load_canonical_digested(
                    _protocol_path(
                        root,
                        bundle_id=bundle_id,
                        cell_id=cell_id,
                    ),
                    label=f"{bundle_id}::{cell_id} protocol",
                )
                source_refs = protocol.get("source_locks")
                problem = protocol.get("problem")
                if not isinstance(source_refs, Mapping) or not isinstance(
                    problem, Mapping
                ):
                    raise MaterializationAuditError(
                        f"{bundle_id}::{cell_id} lacks problem/source-lock data."
                    )
                source_lock_id = str(
                    source_refs.get("cell_source_lock_id", "")
                )
                source_row = cell_locks.get(source_lock_id)
                if not isinstance(source_row, Mapping):
                    raise MaterializationAuditError(
                        f"{bundle_id}::{cell_id} source-lock row is missing."
                    )
                support._verify_self_digest(
                    source_row,
                    label=f"{bundle_id}::{cell_id} source-lock row",
                )
                archive = source_row.get("archive")
                member = source_row.get("member")
                trace = source_row.get("resolver_trace")
                if not all(
                    isinstance(value, Mapping)
                    for value in (archive, member, trace)
                ):
                    raise MaterializationAuditError(
                        f"{bundle_id}::{cell_id} source provenance is incomplete."
                    )
                archive_path = _safe_repo_file(
                    archive["path"],
                    label=f"{bundle_id}::{cell_id} source archive",
                )
                member_path = str(member["path"])
                archive_sha = rehash_archive(archive_path)
                member_sha = rehash_member(archive_path, member_path)
                _assert_equal(
                    archive_sha,
                    archive.get("sha256"),
                    label=f"{bundle_id}::{cell_id} source archive SHA-256",
                )
                _assert_equal(
                    member_sha,
                    member.get("sha256"),
                    label=f"{bundle_id}::{cell_id} source member SHA-256",
                )
                exact_reference = trace.get("same_cutoff_ed_reference")
                if not isinstance(exact_reference, Mapping):
                    raise MaterializationAuditError(
                        f"{bundle_id}::{cell_id} has no same-cutoff ED receipt."
                    )
                nph_work = int(problem["n_ph_max"])
                nph_reference = int(exact_reference["nph"])
                _assert_equal(
                    nph_work,
                    nph_reference,
                    label=f"{bundle_id}::{cell_id} same-cutoff identity",
                )
                receipt = _digested(
                    {
                        "schema": SOURCE_LOCK_CELL_RECEIPT_SCHEMA,
                        "logical_key": f"{bundle_id}::{cell_id}",
                        "bundle_id": bundle_id,
                        "cell_id": cell_id,
                        "protocol_sha256": protocol["sha256"],
                        "problem_request_sha256": problem[
                            "problem_request_sha256"
                        ],
                        "source_locks_manifest_sha256": source_locks["sha256"],
                        "cell_source_lock_id": source_lock_id,
                        "cell_source_lock_sha256": source_row["sha256"],
                        "source_archive_sha256": archive_sha,
                        "source_member_sha256": member_sha,
                        "source_archive_rehashed_at_materialization": True,
                        "source_member_rehashed_at_materialization": True,
                        "same_cutoff_reference": {
                            "n_ph_work": nph_work,
                            "n_ph_reference": nph_reference,
                            "exact_target_label": problem[
                                "exact_target_label"
                            ],
                            "ed_receipt_path": exact_reference["path"],
                            "ed_receipt_sha256": exact_reference["sha256"],
                            "reference_role": (
                                "same_cutoff_reporting_reference"
                            ),
                        },
                    }
                )
                receipts.append(receipt)
    _assert_equal(
        len(receipts),
        EXPECTED_LOGICAL_AUTHORITY_CELL_COUNT,
        label="G1/G2 logical authority receipt count",
    )
    _assert_equal(
        len({receipt["logical_key"] for receipt in receipts}),
        EXPECTED_LOGICAL_AUTHORITY_CELL_COUNT,
        label="G1/G2 unique logical authority receipt count",
    )
    return receipts, {
        "archives": {
            path.relative_to(REPO_ROOT).as_posix(): digest
            for path, digest in sorted(
                archive_hash_cache.items(),
                key=lambda row: row[0].as_posix(),
            )
        },
        "archive_members": [
            {
                "archive_path": archive.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "member_path": member_path,
                "sha256": digest,
            }
            for (archive, member_path), digest in sorted(
                member_hash_cache.items(),
                key=lambda row: (
                    row[0][0].as_posix(),
                    row[0][1],
                ),
            )
        ],
    }


def _build_g3_pool_receipt(
    root: Path,
    *,
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    resolver = support._problem_resolver_from(baselines)
    adapter = SinglePauliWordCandidateAdapter()
    groups: list[dict[str, Any]] = []
    for regime_id in VALIDATION_REGIMES:
        macro_protocols: list[Mapping[str, Any]] = []
        singleton_protocols: list[Mapping[str, Any]] = []
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES:
            for route_id in VALIDATION_ROUTE_IDS:
                protocol = _load_canonical_digested(
                    _protocol_path(
                        root,
                        bundle_id=bundle_id,
                        cell_id=_validation_cell_id(regime_id, route_id),
                    ),
                    label=f"{bundle_id}::{regime_id}::{route_id} protocol",
                )
                if route_id == "singleton_plateau":
                    singleton_protocols.append(protocol)
                else:
                    macro_protocols.append(protocol)
        macro = _pool_projection(macro_protocols[0]["executable_pool"])
        _assert_equal(
            {
                "count": macro["count"],
                "ordered_labels_sha256": macro[
                    "ordered_labels_sha256"
                ],
            },
            EXPECTED_MACRO_MEMBERSHIP,
            label=f"{regime_id} stable macro membership",
        )
        if any(
            _pool_projection(protocol["executable_pool"]) != macro
            for protocol in macro_protocols
        ):
            raise MaterializationAuditError(
                f"{regime_id} RA/Append macro pools are not identical."
            )
        parent = _pool_projection(singleton_protocols[0]["parent_inventory"])
        _assert_equal(
            {
                "count": parent["count"],
                "ordered_labels_sha256": parent[
                    "ordered_labels_sha256"
                ],
            },
            EXPECTED_SINGLETON_PARENT_MEMBERSHIP,
            label=f"{regime_id} stable singleton parent membership",
        )
        if any(
            _pool_projection(protocol["parent_inventory"]) != parent
            for protocol in singleton_protocols
        ):
            raise MaterializationAuditError(
                f"{regime_id} singleton parents differ across policies."
            )
        problem = resolver(regime_id, 3)
        runtime_parent = adapter.parent_inventory(problem).receipt.to_dict()
        runtime_child = adapter.global_executable_pool(
            problem
        ).receipt.to_dict()
        _assert_equal(
            _pool_projection(runtime_parent),
            parent,
            label=f"{regime_id} runtime singleton parent",
        )
        child = _pool_projection(runtime_child)
        problem_request_sha256 = singleton_protocols[0]["problem"][
            "problem_request_sha256"
        ]
        construction_equivalence = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_study1_singleton_"
                    "construction_equivalence_v1"
                ),
                "regime_id": regime_id,
                "problem_request_sha256": problem_request_sha256,
                "ra_parent": parent,
                "append_parent": dict(parent),
                "append_guarded_child": child,
                "ra_exposure_mode": "staged_child_exposure_v1",
                "append_exposure_mode": (
                    "global_guarded_child_pool_v1"
                ),
                "ra_staged_funnel_invoked": True,
                "append_ra_staged_funnel_invoked": False,
                "parent_identity_equal": True,
                "guarded_child_construction_passed": True,
                "guarded_child_inventory_receipt_sha256": runtime_child[
                    "sha256"
                ],
            }
        )
        groups.append(
            {
                "regime_id": regime_id,
                "nph": 3,
                "problem_request_sha256": problem_request_sha256,
                "macro": macro,
                "singleton_construction": {
                    "ra_parent": parent,
                    "append_parent": dict(parent),
                    "append_guarded_child": child,
                    "source_parent_ordered_labels_sha256": parent[
                        "ordered_labels_sha256"
                    ],
                    "construction_equivalence_receipt": (
                        construction_equivalence
                    ),
                    "construction_receipt_sha256": (
                        construction_equivalence["sha256"]
                    ),
                },
            }
        )
    return _digested(
        {
            "schema": POOL_CONSTRUCTION_RECEIPT_SCHEMA,
            "regime_groups": groups,
        }
    )


def _validate_g8_materialized_protocol_guard(root: Path) -> dict[str, Any]:
    """Prove the narrow Study-1 requests do not enable generic exact stops."""

    protocol_rows: list[dict[str, Any]] = []
    forbidden_stop_fields = {
        "exact_ed_target",
        "exact_energy",
        "exact_gs_override",
        "benchmark_target_reference_energy",
    }
    for bundle_id, _policy in STUDY1_BUNDLE_POLICIES:
        manifest = _load_canonical_digested(
            root / bundle_id / "bundle_manifest.json",
            label=f"{bundle_id} manifest for G8",
        )
        cells = manifest.get("cells")
        if not isinstance(cells, list):
            raise MaterializationAuditError(
                f"{bundle_id} manifest has no cells for G8."
            )
        for cell in cells:
            cell_id = str(cell.get("cell_id", ""))
            protocol = _load_canonical_digested(
                _protocol_path(
                    root,
                    bundle_id=bundle_id,
                    cell_id=cell_id,
                ),
                label=f"{bundle_id}::{cell_id} protocol for G8",
            )
            request = protocol.get("request")
            if not isinstance(request, Mapping):
                raise MaterializationAuditError(
                    f"{bundle_id}::{cell_id} request is unavailable for G8."
                )
            execution = request.get("execution")
            stop = (
                execution.get("stop")
                if isinstance(execution, Mapping)
                else None
            )
            if not isinstance(stop, Mapping):
                raise MaterializationAuditError(
                    f"{bundle_id}::{cell_id} has no typed stop request."
                )
            observed_forbidden = sorted(
                forbidden_stop_fields.intersection(stop)
            )
            if observed_forbidden:
                raise MaterializationAuditError(
                    f"{bundle_id}::{cell_id} exposes exact-reference controller "
                    f"inputs: {observed_forbidden}."
                )
            _assert_equal(
                set(stop),
                {"maximum_controller_rounds"},
                label=f"{bundle_id}::{cell_id} controller stop fields",
            )
            protocol_rows.append(
                {
                    "bundle_id": bundle_id,
                    "cell_id": cell_id,
                    "protocol_sha256": protocol["sha256"],
                    "controller_stop_fields": sorted(stop),
                    "controller_exact_reference_inputs": [],
                }
            )
    _assert_equal(
        len(protocol_rows),
        EXPECTED_PROTOCOL_COUNT,
        label="G8 protocol request scan count",
    )
    return _digested(
        {
            "schema": (
                "paper_i_ra_adapt_study1_protocol_exact_reference_guard_v1"
            ),
            "status": "passed",
            "scope": (
                "materialized_study1_protocol_request_controller_inputs_v1"
            ),
            "protocol_count": len(protocol_rows),
            "forbidden_exact_reference_stop_fields": sorted(
                forbidden_stop_fields
            ),
            "controller_exact_reference_inputs": [],
            "protocol_rows": protocol_rows,
            "claim_boundary": (
                "study1_materialized_requests_only_not_generic_sr_capability_v1"
            ),
        }
    )


def _build_g8_trusted_execution_receipt() -> dict[str, Any]:
    """Build and source-reverify the one runtime/materialization G8 authority."""

    _assert_equal(
        TRUSTED_EXECUTION_RECEIPT_SCHEMA,
        STUDY1_TRUSTED_EXECUTION_SCHEMA,
        label="G8 trusted-execution schema",
    )
    built = build_study1_trusted_execution_receipt(
        source_root=REPO_ROOT
    ).to_dict()
    validated = validate_study1_trusted_execution_receipt(
        built,
        source_root=REPO_ROOT,
        reverify_source=True,
    )
    _assert_equal(
        validated,
        built,
        label="G8 trusted-execution source reverification",
    )
    return validated


def _build_g13_t13_receipt() -> dict[str, Any]:
    fixture = _load_mapping(T13_FIXTURE, label="T13 fixture")
    _assert_equal(
        _sha256_file(T13_FIXTURE),
        T13_FIXTURE_FILE_SHA256,
        label="T13 fixture file SHA-256",
    )
    _assert_equal(
        canonical_sha256(fixture),
        T13_FIXTURE_CANONICAL_SHA256,
        label="T13 fixture canonical SHA-256",
    )
    problem = fixture.get("problem")
    route = fixture.get("route")
    if not isinstance(problem, Mapping) or not isinstance(route, Mapping):
        raise MaterializationAuditError(
            "T13 fixture lacks problem/route characterization."
        )
    expected_problem = {
        "u": 2.0,
        "g_ep": 1.0,
        "n_ph_max": 3,
        "problem_request_sha256": T13_PROBLEM_REQUEST_SHA256,
    }
    for key, expected in expected_problem.items():
        _assert_equal(
            problem.get(key),
            expected,
            label=f"T13 fixture problem.{key}",
        )
    route_sha = str(route.get("contract_sha256", ""))
    if len(route_sha) != 64:
        raise MaterializationAuditError(
            "T13 fixture route contract has no SHA-256."
        )
    return _digested(
        {
            "schema": T13_CHARACTERIZATION_RECEIPT_SCHEMA,
            "fixture_contract_id": (
                "historical_singleton_plateau_route_t13_v1"
            ),
            "problem_request_sha256": T13_PROBLEM_REQUEST_SHA256,
            "fixture_file_sha256": T13_FIXTURE_FILE_SHA256,
            "fixture_canonical_sha256": T13_FIXTURE_CANONICAL_SHA256,
            "route_contract_sha256": route_sha,
            "status": "passed",
            "study1_problem_comparison_performed": False,
        }
    )


def _build_objective_gate_authority(
    root: Path,
    *,
    baselines: Mapping[str, Any],
) -> dict[str, Any]:
    g1_g2, _source_rehashes = _build_g1_g2_cell_receipts(root)
    return _digested(
        {
            "schema": AUTHORITY_SCHEMA,
            "package_id": PACKAGE_ID,
            "materialization_revision": "v8",
            "g1_g2_cell_receipts": g1_g2,
            "g3_pool_construction_receipt": _build_g3_pool_receipt(
                root, baselines=baselines
            ),
            "g8_trusted_execution_receipt": (
                _build_g8_trusted_execution_receipt()
            ),
            "g13_t13_characterization_receipt": (
                _build_g13_t13_receipt()
            ),
        }
    )


def _write_receipt(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return v7._write_receipt(path, payload)


def _authority_binding(
    path: Path,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "path": AUTHORITY_NAME,
        "canonical_sha256": str(payload["sha256"]),
        "file_sha256": _sha256_file(path),
    }


def _validate_authority_binding(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    observed = _load_canonical_digested(
        path, label="Study-1 objective-gate authority"
    )
    _assert_equal(
        observed,
        expected,
        label="Study-1 objective-gate authority payload",
    )
    return _authority_binding(path, observed)


def _run_loader_validation(
    load_root: Path,
    *,
    display_root: Path,
    phase: str,
) -> dict[str, Any]:
    """Load every protocol while recording rename-stable display paths."""

    rows: list[dict[str, Any]] = []
    bundle_counts: dict[str, int] = {}
    for bundle_id, _policy in STUDY1_BUNDLE_POLICIES:
        bundle_root = load_root / bundle_id
        manifest = _load_canonical_digested(
            bundle_root / "bundle_manifest.json",
            label=f"{phase} bundle manifest",
        )
        cells = manifest.get("cells")
        if not isinstance(cells, list):
            raise MaterializationAuditError(
                f"{phase}/{bundle_id} has no ordered cells."
            )
        bundle_counts[bundle_id] = len(cells)
        for cell in cells:
            cell_id = str(cell.get("cell_id", ""))
            load_path = bundle_root / "protocols" / f"{cell_id}.json"
            display_path = (
                display_root
                / bundle_id
                / "protocols"
                / f"{cell_id}.json"
            )
            protocol = _load_canonical_digested(
                load_path, label=f"{phase} protocol"
            )
            try:
                loaded = load_validated_bundle_protocol(load_path)
                if str(loaded.sha256) != protocol["sha256"]:
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
                    "bundle_id": bundle_id,
                    "cell_id": cell_id,
                    "protocol_path": display_path.relative_to(
                        REPO_ROOT
                    ).as_posix(),
                    "protocol_sha256": protocol["sha256"],
                    "status": status,
                    **({"error": error} if error is not None else {}),
                }
            )
    total = sum(bundle_counts.values())
    passed = sum(row["status"] == "passed" for row in rows)
    failed = total - passed
    _assert_equal(
        total, EXPECTED_PROTOCOL_COUNT, label=f"{phase} loader total"
    )
    _assert_equal(
        len(rows), total, label=f"{phase} loader row count"
    )
    if failed:
        raise MaterializationAuditError(
            f"{phase} cross-loader validation failed {failed}/{total} rows."
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
            "bundle_counts": bundle_counts,
            "total_count": total,
            "passed_count": passed,
            "failed_count": failed,
            "rows": rows,
        }
    )


def main() -> int:
    if V8_ROOT.exists():
        raise FileExistsError(
            f"Refusing to overwrite immutable v8: {V8_ROOT}"
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
    implementation_preflight = _implementation_source_inventory(REPO_ROOT)
    _inventory_rows(implementation_preflight)
    source_locks = _load_mapping(
        SOURCE_LOCKS_INPUT, label="source-lock input"
    )
    baselines = _load_mapping(
        PROBLEM_BASELINES, label="problem baselines"
    )

    staging_root = Path(
        tempfile.mkdtemp(
            prefix=".ra_adapt_unification_post_refactor_v8.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    source_before, source_after = _copy_v7_source_materialization(
        staging_root
    )
    inheritance_receipt = _write_receipt(
        staging_root / "source_materialization_inheritance_receipt.json",
        {
            "schema": "ra_adapt_source_materialization_inheritance_v1",
            "status": "passed",
            "source_revision": "v7",
            "target_revision": "v8",
            "source_path": V7_SOURCE_ROOT.relative_to(
                REPO_ROOT
            ).as_posix(),
            "target_path": (
                V8_ROOT / "source_materialization"
            ).relative_to(REPO_ROOT).as_posix(),
            "copy_policy": "byte_identical_no_path_rebasing_v1",
            "historical_embedded_paths_preserved": True,
            "file_count": source_before["file_count"],
            "total_size_bytes": source_before["total_size_bytes"],
            "source_relative_tree_sha256": source_before["tree_sha256"],
            "copied_relative_tree_sha256": source_after["tree_sha256"],
            "files_equal": True,
        },
    )
    preflight_receipt = _write_receipt(
        staging_root / "preflight_receipt.json",
        {
            "schema": "ra_adapt_v8_preflight_receipt_v1",
            "status": "passed",
            "captured_utc": captured_utc,
            "repository_state": repository_state,
            "implementation_inventory": implementation_preflight,
            "required_implementation_paths": sorted(
                REQUIRED_IMPLEMENTATION_PATHS
            ),
            "visible_source_locks": {
                "path": SOURCE_LOCKS_INPUT.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "file_sha256": _sha256_file(SOURCE_LOCKS_INPUT),
                "schema": source_locks.get("schema"),
                "cell_lock_count": len(
                    source_locks.get("cell_locks", {})
                ),
                "global_source_count": len(
                    source_locks.get("global_sources", {})
                ),
            },
            "problem_baselines": {
                "path": PROBLEM_BASELINES.relative_to(
                    REPO_ROOT
                ).as_posix(),
                "file_sha256": _sha256_file(PROBLEM_BASELINES),
                "schema": baselines.get("schema"),
            },
            "older_materialization_preservation": {
                revision: _summary(snapshot)
                for revision, snapshot in historical_before.items()
            },
            "source_materialization_inheritance_receipt_sha256": (
                inheritance_receipt["sha256"]
            ),
            "execution_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )

    timestamp = support._utc_now()
    materialized_receipts = materialize_study1_bundles(
        staging_root,
        problem_resolver=support._problem_resolver_from(baselines),
        source_locks=source_locks,
        validation_horizon=23,
        repository_state=repository_state,
        repo_root=REPO_ROOT,
        full_horizon=50,
        dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
        materialization_timestamp=timestamp,
        verify_source_files=True,
    )
    bundle_summaries = [
        support._validate_bundle_surface(
            staging_root / bundle_id,
            expected_implementation_inventory=implementation_preflight,
        )
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    receipt_by_bundle = {
        receipt.bundle_id: receipt for receipt in materialized_receipts
    }
    for summary in bundle_summaries:
        receipt = receipt_by_bundle[summary["bundle_id"]]
        _assert_equal(
            receipt.cell_count,
            58,
            label=f"{receipt.bundle_id} materializer cell count",
        )
        _assert_equal(
            receipt.materialization_status,
            "passed",
            label=f"{receipt.bundle_id} materializer status",
        )
        _assert_equal(
            receipt.bundle_manifest_sha256,
            summary["bundle_manifest_sha256"],
            label=f"{receipt.bundle_id} manifest receipt binding",
        )

    repair_semantics = _validate_v7_v8_repair_semantics(
        staging_root,
        implementation_inventory=implementation_preflight,
    )
    repair_semantics_receipt = _write_receipt(
        staging_root / "v7_v8_repair_semantics_receipt.json",
        {
            key: value
            for key, value in repair_semantics.items()
            if key != "sha256"
        },
    )
    g8_protocol_guard = _validate_g8_materialized_protocol_guard(
        staging_root
    )
    g8_protocol_guard_receipt = _write_receipt(
        staging_root / "study1_g8_protocol_guard_receipt.json",
        {
            key: value
            for key, value in g8_protocol_guard.items()
            if key != "sha256"
        },
    )
    authority = _build_objective_gate_authority(
        staging_root,
        baselines=baselines,
    )
    authority_path = staging_root / AUTHORITY_NAME
    authority_receipt = _write_receipt(
        authority_path,
        {
            key: value
            for key, value in authority.items()
            if key != "sha256"
        },
    )
    staged_authority_binding = _validate_authority_binding(
        authority_path, expected=authority_receipt
    )

    staged_loader = _run_loader_validation(
        staging_root,
        display_root=V8_ROOT,
        phase="staged_pre_publish",
    )
    staged_loader_receipt = _write_receipt(
        staging_root / "cross_file_loader_validation_staged.json",
        {
            key: value
            for key, value in staged_loader.items()
            if key != "sha256"
        },
    )
    final_loader = _run_loader_validation(
        staging_root,
        display_root=V8_ROOT,
        phase="canonical_path_projected_pre_publish",
    )
    final_loader_receipt = _write_receipt(
        staging_root / "cross_file_loader_validation.json",
        {
            key: value
            for key, value in final_loader.items()
            if key != "sha256"
        },
    )
    implementation_post_staged_loader = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_post_staged_loader,
        implementation_preflight,
        label="preflight-to-staged-loader implementation inventory",
    )
    historical_pre_publish = _historical_snapshots()
    _assert_historical_unchanged(
        historical_before,
        historical_pre_publish,
        label="Staged v8 materialization",
    )
    source_staged = support._snapshot_roots(
        (staging_root / "source_materialization",),
        relative_to=staging_root / "source_materialization",
    )
    _assert_equal(
        source_staged,
        source_before,
        label="staged inherited source-materialization",
    )
    final_bundle_summaries = [
        support._validate_bundle_surface(
            staging_root / bundle_id,
            expected_implementation_inventory=implementation_preflight,
        )
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    _assert_equal(
        final_bundle_summaries,
        bundle_summaries,
        label="revalidated staged bundle summaries",
    )
    final_repair_semantics = _validate_v7_v8_repair_semantics(
        staging_root,
        implementation_inventory=implementation_preflight,
    )
    _assert_equal(
        final_repair_semantics,
        repair_semantics_receipt,
        label="revalidated staged v7-to-v8 repair semantics",
    )
    final_g8_protocol_guard = _validate_g8_materialized_protocol_guard(
        staging_root
    )
    _assert_equal(
        final_g8_protocol_guard,
        g8_protocol_guard_receipt,
        label="revalidated staged G8 protocol guard",
    )
    v7_final = _load_canonical_digested(
        V7_FINAL_RECEIPT, label="v7 final receipt for supersession"
    )
    supersession_chain = list(v7_final.get("supersession_chain", ()))
    if (
        not supersession_chain
        or supersession_chain[-1].get("revision") != "v7"
    ):
        raise MaterializationAuditError(
            "The v7 supersession chain has no v7 terminus."
        )
    supersession_chain.append(
        {
            "revision": "v8",
            "path": V8_ROOT.relative_to(REPO_ROOT).as_posix(),
            "status": "passed",
        }
    )
    implementation_post_final_loader = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_post_final_loader,
        implementation_preflight,
        label="preflight-to-final-projected-loader implementation inventory",
    )
    staged_receipt = _write_receipt(
        staging_root / "staged_materialization_receipt.json",
        {
            "schema": "ra_adapt_v8_staged_materialization_receipt_v1",
            "status": "passed",
            "materialization_revision": "v8",
            "materialization_timestamp": timestamp,
            "preflight_receipt_sha256": preflight_receipt["sha256"],
            "source_inheritance_receipt_sha256": inheritance_receipt[
                "sha256"
            ],
            "repair_semantics_receipt_sha256": repair_semantics_receipt[
                "sha256"
            ],
            "g8_protocol_guard_receipt_sha256": (
                g8_protocol_guard_receipt["sha256"]
            ),
            "study1_objective_gate_authority": staged_authority_binding,
            "staged_loader_validation_sha256": staged_loader_receipt[
                "sha256"
            ],
            "canonical_path_projected_loader_validation_sha256": (
                final_loader_receipt["sha256"]
            ),
            "bundles": bundle_summaries,
            "implementation_inventory": {
                "preflight_sha256": implementation_preflight["sha256"],
                "post_staged_loader_sha256": (
                    implementation_post_staged_loader["sha256"]
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
            "post_publish_write_count": 0,
            "execution_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )
    tree_before_final = support._snapshot_roots(
        (staging_root,), relative_to=staging_root
    )
    final_receipt = _write_receipt(
        staging_root / "final_materialization_receipt.json",
        {
            "schema": "ra_adapt_final_materialization_receipt_v1",
            "status": "passed",
            "campaign_id": "paper_i_ra_adapt_stationarity_comparison_v1",
            "run_class": "candidate",
            "materialization_revision": "v8",
            "finalized_utc": support._utc_now(),
            "atomic_publish": {
                "method": "darwin_renameatx_np_RENAME_EXCL_v1",
                "no_replace": True,
                "unsupported_platform_behavior": "fail_closed",
                "all_final_files_staged_before_publish": True,
                "post_publish_write_count": 0,
                "staged_materialization_receipt_sha256": staged_receipt[
                    "sha256"
                ],
                "staged_loader_validation_sha256": staged_loader_receipt[
                    "sha256"
                ],
                "final_loader_validation_sha256": final_loader_receipt[
                    "sha256"
                ],
                "staged_and_final_loader_rows": EXPECTED_PROTOCOL_COUNT,
            },
            "bundles": final_bundle_summaries,
            "source_materialization": {
                "path": (
                    V8_ROOT / "source_materialization"
                ).relative_to(REPO_ROOT).as_posix(),
                "status": "inherited_byte_identical",
                "source_revision": "v7",
                "file_count": source_staged["file_count"],
                "relative_tree_sha256": source_staged["tree_sha256"],
                "inheritance_receipt_sha256": inheritance_receipt[
                    "sha256"
                ],
                "snapshots_exactly_equal": True,
            },
            "corrected_repair_semantics": {
                "path": "v7_v8_repair_semantics_receipt.json",
                "canonical_sha256": repair_semantics_receipt["sha256"],
                "file_sha256": _sha256_file(
                    staging_root / "v7_v8_repair_semantics_receipt.json"
                ),
                "status": "passed",
            },
            "study1_objective_gate_authority": (
                staged_authority_binding
            ),
            "study1_g8_protocol_guard": {
                "path": "study1_g8_protocol_guard_receipt.json",
                "canonical_sha256": g8_protocol_guard_receipt["sha256"],
                "file_sha256": _sha256_file(
                    staging_root / "study1_g8_protocol_guard_receipt.json"
                ),
                "protocol_count": EXPECTED_PROTOCOL_COUNT,
                "status": "passed",
            },
            "loader_validation": {
                "path": "cross_file_loader_validation.json",
                "sha256": final_loader_receipt["sha256"],
                "total_count": EXPECTED_PROTOCOL_COUNT,
                "passed_count": EXPECTED_PROTOCOL_COUNT,
                "failed_count": 0,
            },
            "implementation_inventory": {
                "root_count": implementation_preflight["root_count"],
                "file_count": implementation_preflight["file_count"],
                "preflight_sha256": implementation_preflight["sha256"],
                "post_staged_loader_sha256": (
                    implementation_post_staged_loader["sha256"]
                ),
                "post_final_loader_sha256": (
                    implementation_post_final_loader["sha256"]
                ),
                "stable": True,
            },
            "older_materialization_preservation": (
                _preservation_comparison(
                    historical_before, historical_pre_publish
                )
            ),
            "v8_tree_before_final_receipt": _summary(tree_before_final),
            "supersession_chain": supersession_chain,
            "stationarity_winner_selected": False,
            "user_decision_required_after_study_1": True,
            "execution_authorized": False,
            "submission_state": "not_submitted",
            "submitted": False,
        },
    )
    persisted_final_receipt = _load_canonical_digested(
        staging_root / "final_materialization_receipt.json",
        label="staged final materialization receipt",
    )
    _assert_equal(
        persisted_final_receipt,
        final_receipt,
        label="staged final materialization receipt payload",
    )
    complete_staged_tree = support._snapshot_roots(
        (staging_root,), relative_to=staging_root
    )
    historical_final_pre_publish = _historical_snapshots()
    _assert_historical_unchanged(
        historical_before,
        historical_final_pre_publish,
        label="Complete staged v8 materialization",
    )
    implementation_final_pre_publish = _implementation_source_inventory(
        REPO_ROOT
    )
    _assert_equal(
        implementation_final_pre_publish,
        implementation_preflight,
        label="complete-staging implementation inventory",
    )

    # This is the sole publication mutation.  Every final file, including the
    # final receipt, already exists in the private sibling staging tree.
    support._atomic_rename_no_replace(staging_root, V8_ROOT)

    # Everything below is read-only.  A successful publication never exposes
    # an incomplete canonical v8 and never requires cleanup to retry.
    source_post_publish = support._snapshot_roots(
        (V8_ROOT / "source_materialization",),
        relative_to=V8_ROOT / "source_materialization",
    )
    _assert_equal(
        source_post_publish,
        source_before,
        label="published inherited source-materialization",
    )
    final_authority_binding = _validate_authority_binding(
        V8_ROOT / AUTHORITY_NAME,
        expected=authority_receipt,
    )
    _assert_equal(
        final_authority_binding,
        staged_authority_binding,
        label="staged-to-final authority binding",
    )
    published_final_receipt = _load_canonical_digested(
        V8_ROOT / "final_materialization_receipt.json",
        label="published final materialization receipt",
    )
    _assert_equal(
        published_final_receipt,
        final_receipt,
        label="staged-to-published final receipt",
    )
    post_publish_loader = _run_loader_validation(
        V8_ROOT,
        display_root=V8_ROOT,
        phase="canonical_post_publish_read_only",
    )
    _assert_equal(
        post_publish_loader["passed_count"],
        EXPECTED_PROTOCOL_COUNT,
        label="post-publish read-only loader count",
    )
    published_bundle_summaries = [
        support._validate_bundle_surface(
            V8_ROOT / bundle_id,
            expected_implementation_inventory=implementation_preflight,
        )
        for bundle_id, _policy in STUDY1_BUNDLE_POLICIES
    ]
    _assert_equal(
        published_bundle_summaries,
        final_bundle_summaries,
        label="staged-to-published bundle summaries",
    )
    _assert_equal(
        _validate_v7_v8_repair_semantics(
            V8_ROOT,
            implementation_inventory=implementation_preflight,
        ),
        repair_semantics_receipt,
        label="staged-to-published v7-to-v8 repair semantics",
    )
    _assert_equal(
        _validate_g8_materialized_protocol_guard(V8_ROOT),
        g8_protocol_guard_receipt,
        label="staged-to-published G8 protocol guard",
    )
    historical_after = _historical_snapshots()
    _assert_historical_unchanged(
        historical_before,
        historical_after,
        label="Published v8 materialization",
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
        (V8_ROOT,), relative_to=V8_ROOT
    )
    _assert_equal(
        published_tree,
        complete_staged_tree,
        label="staged-to-published complete tree",
    )
    print(
        json.dumps(
            {
                "destination": V8_ROOT.relative_to(REPO_ROOT).as_posix(),
                "status": "passed",
                "bundle_count": len(final_bundle_summaries),
                "cell_count_per_bundle": 58,
                "loader_validation": "116/116",
                "implementation_inventory_sha256": (
                    implementation_preflight["sha256"]
                ),
                "objective_gate_authority_sha256": authority_receipt[
                    "sha256"
                ],
                "staged_materialization_receipt_sha256": staged_receipt[
                    "sha256"
                ],
                "final_receipt_sha256": final_receipt["sha256"],
                "final_receipt_file_sha256": _sha256_file(
                    V8_ROOT / "final_materialization_receipt.json"
                ),
                "published_tree_sha256": published_tree["tree_sha256"],
                "published_file_count": published_tree["file_count"],
                "execution_authorized": False,
                "submitted": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
