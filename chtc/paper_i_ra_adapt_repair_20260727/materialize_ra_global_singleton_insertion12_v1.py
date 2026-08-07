#!/usr/bin/env python3
"""Materialize the inert 12-cell global-singleton insertion comparison.

The six exact singleton-plateau source locks are inherited from the sealed
stationary-core v13 materialization.  Each canonical Paper-I regime is crossed
with commutation-reduced append and the existing commutation-reduced plateau
policy.  Stationarity and all-phase resource weighting are fixed.  This
command has no execution, scheduler, package, or activation seam and refuses
to overwrite an existing materialization.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"

from chtc.paper_i_ra_adapt_repair_20260727 import (  # noqa: E402
    materialize_stationary_core_v12 as v12,
)
from pipelines.static_adapt.ra_adapt.adapters import (  # noqa: E402
    GLOBAL_SINGLE_PAULI_ADAPTER_ID,
    PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
    PHASE_I_VISIBILITY_ALL_EXECUTABLE,
    PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
)
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    CLAIM_FACING_REGIME_CUTOFF_PAIRS,
    GLOBAL_SINGLETON_BUNDLE_ID,
    GLOBAL_SINGLETON_CAMPAIGN_ID,
    GLOBAL_SINGLETON_INSERTION_ROUTE_IDS,
    GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME,
    GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH,
    GLOBAL_SINGLETON_RUN_CLASS,
    ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED,
    ROUTE_RA_SINGLETON_PLATEAU,
    SINGLETON_PARENT_MEMBERSHIP_BY_NPH,
    build_global_singleton_insertion_cell_specs,
    materialize_global_singleton_insertion_bundle,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    ACTIVE_GRADIENT_STATIONARY,
    RESOURCE_WEIGHTING_ALL_PHASE,
    canonical_json_bytes,
    canonical_sha256,
)
from pipelines.static_adapt.sr_snake.contracts import (  # noqa: E402
    AppendCommutationReducedInsertion,
    PlateauCommutationInsertion,
)


REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MATERIALIZATIONS_ROOT = REPAIR_ROOT / "bundles/materializations"
MATERIALIZATION_ID = "ra_adapt_global_singleton_insertion12_v1"
MATERIALIZATION_ROOT = MATERIALIZATIONS_ROOT / MATERIALIZATION_ID
V13_MATERIALIZATION_ID = "ra_adapt_stationary_late_core_v13"
V13_ROOT = MATERIALIZATIONS_ROOT / V13_MATERIALIZATION_ID
V13_SOURCE_LOCKS = (
    V13_ROOT / "source_materialization/source_locks_input.json"
)
V13_PROBLEM_BASELINES = (
    V13_ROOT / "source_materialization/problem_baselines.json"
)
V13_FINAL_RECEIPT = V13_ROOT / "final_publication_receipt.json"

EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256 = (
    "c361f55eb5551e9c11251ea01fce0059521c48d9fa0398d865b02a6bd29df656"
)
EXPECTED_V13_PROBLEM_BASELINES_FILE_SHA256 = (
    "a12a36c3f2c8bfe74e4c8a0c9db1d1baecf3b100b00480c5386e903d973c4015"
)
EXPECTED_V13_FINAL_FILE_SHA256 = (
    "d9219d94db6f75d65842b828642e833d3c17eef0534516d0ef6dc2de08f6b415"
)
EXPECTED_CELL_COUNT = 12
EXPECTED_SOURCE_CELL_COUNT = 6
FINAL_RECEIPT_NAME = (
    "global_singleton_insertion12_materialization_receipt.json"
)

COMMON_DELTA_ROWS = (
    {
        "binding": "fixed_campaign_policy",
        "classification": "explicit_user_requested_fixed_policy_v1",
        "field": "resource_weighting_scope",
        "id": "D5",
        "to": RESOURCE_WEIGHTING_ALL_PHASE,
    },
    {
        "binding": "fixed_campaign_candidate_supply",
        "classification": "explicit_user_requested_global_singleton_v1",
        "field": "candidate_adapter_id",
        "id": "global_singleton_candidate_adapter",
        "to": GLOBAL_SINGLE_PAULI_ADAPTER_ID,
    },
    {
        "binding": "fixed_campaign_candidate_supply",
        "classification": "explicit_user_requested_global_singleton_v1",
        "field": "phase_i_candidate_supply",
        "id": "global_singleton_phase_i_candidate_supply",
        "to": PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON,
    },
    {
        "binding": "fixed_campaign_candidate_supply",
        "classification": "explicit_user_requested_global_singleton_v1",
        "field": "phase_i_candidate_visibility",
        "id": "global_singleton_phase_i_candidate_visibility",
        "to": PHASE_I_VISIBILITY_ALL_EXECUTABLE,
    },
    {
        "binding": "fixed_campaign_candidate_supply",
        "classification": "explicit_user_requested_global_singleton_v1",
        "field": "phase_ii_candidate_exposure",
        "id": "global_singleton_phase_ii_candidate_exposure",
        "to": PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY,
    },
)


class GlobalSingletonMaterializationError(ValueError):
    """Raised when the immutable campaign materialization does not close."""


def _sha256_file(path: Path) -> str:
    return v12.support._hash_file(path)


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise GlobalSingletonMaterializationError(
            f"{label} is missing or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GlobalSingletonMaterializationError(
            f"{label} is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise GlobalSingletonMaterializationError(
            f"{label} must be a JSON object."
        )
    return payload


def _digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("sha256", None)
    result["sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    v12.support._write_bytes_atomic_no_replace(
        path, canonical_json_bytes(payload) + b"\n"
    )


def _binding(path: Path, *, relative_to: Path) -> dict[str, Any]:
    payload = _load_mapping(path, label=path.name)
    return {
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": _sha256_file(path),
        "canonical_sha256": payload.get("sha256"),
        "size_bytes": path.stat().st_size,
    }


def _repository_state() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    return {
        "git_commit": commit,
        "dirty_working_tree": dirty,
        "cwd": REPO_ROOT.as_posix(),
    }


def _sealed_plateau_source_locks() -> dict[str, Any]:
    if _sha256_file(V13_SOURCE_LOCKS) != (
        EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
    ):
        raise GlobalSingletonMaterializationError(
            "The sealed v13 source-lock input drifted."
        )
    source = _load_mapping(
        V13_SOURCE_LOCKS, label="sealed v13 source locks"
    )
    raw_cells = source.get("cell_locks")
    if not isinstance(raw_cells, Mapping):
        raise GlobalSingletonMaterializationError(
            "The sealed v13 source lock has no cell map."
        )
    selected = {
        lock_id: copy.deepcopy(row)
        for lock_id, row in sorted(raw_cells.items())
        if isinstance(row, Mapping)
        and row.get("route_id") == ROUTE_RA_SINGLETON_PLATEAU
    }
    if len(selected) != EXPECTED_SOURCE_CELL_COUNT:
        raise GlobalSingletonMaterializationError(
            "The sealed v13 source lock does not contain exactly six "
            "singleton-plateau cells."
        )
    expected_pairs = set(CLAIM_FACING_REGIME_CUTOFF_PAIRS)
    observed_pairs = {
        (str(row.get("regime_id")), int(row.get("nph", -1)))
        for row in selected.values()
    }
    if observed_pairs != expected_pairs:
        raise GlobalSingletonMaterializationError(
            "The sealed v13 singleton-plateau regime/cutoff matrix drifted."
        )
    for lock_id, row in selected.items():
        changes = row.get("resolver_trace", {}).get(
            "settings_changed", []
        )
        insertion = [
            change
            for change in changes
            if isinstance(change, Mapping)
            and change.get("field") == "insertion_policy"
        ]
        if (
            len(insertion) != 1
            or insertion[0].get("to")
            != PlateauCommutationInsertion.kind
        ):
            raise GlobalSingletonMaterializationError(
                f"Source lock is not singleton plateau: {lock_id}."
            )
    return {
        "schema": source.get("schema"),
        "global_sources": copy.deepcopy(source.get("global_sources")),
        "cell_locks": selected,
    }


def _derive_campaign_source_locks(
    sealed: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sealed_cells = sealed.get("cell_locks")
    if not isinstance(sealed_cells, Mapping):
        raise GlobalSingletonMaterializationError(
            "Sealed singleton-plateau locks have no cell map."
        )
    cells = build_global_singleton_insertion_cell_specs()
    derived_cells: dict[str, Any] = {}
    delta_rows: list[dict[str, Any]] = []
    for cell in cells:
        predecessor_id = (
            f"{cell.regime_id}__nph{cell.nph}__"
            f"{ROUTE_RA_SINGLETON_PLATEAU}"
        )
        predecessor = sealed_cells.get(predecessor_id)
        if not isinstance(predecessor, Mapping):
            raise GlobalSingletonMaterializationError(
                f"Missing sealed predecessor {predecessor_id}."
            )
        derived = copy.deepcopy(dict(predecessor))
        trace = derived.get("resolver_trace")
        if not isinstance(trace, dict):
            raise GlobalSingletonMaterializationError(
                f"Predecessor has no mutable resolver trace: {predecessor_id}."
            )
        predecessor_changes = trace.get("settings_changed")
        if not isinstance(predecessor_changes, list):
            raise GlobalSingletonMaterializationError(
                f"Predecessor has no settings-change list: {predecessor_id}."
            )
        predecessor_core_anchor = trace.pop("core_source_anchor", None)
        if not isinstance(predecessor_core_anchor, Mapping):
            raise GlobalSingletonMaterializationError(
                f"Predecessor has no core source anchor: {predecessor_id}."
            )

        derived["route_id"] = cell.route_id
        trace["method"] = cell.route_id
        route_row = {
            "binding": "insertion_comparison_route_identity",
            "classification": "explicit_user_requested_route_identity_v1",
            "field": "route_id",
            "from": ROUTE_RA_SINGLETON_PLATEAU,
            "id": "global_singleton_route_identity",
            "to": cell.route_id,
        }
        new_rows = [
            *copy.deepcopy(predecessor_changes),
            *copy.deepcopy(COMMON_DELTA_ROWS),
            route_row,
        ]
        target_insertion = PlateauCommutationInsertion.kind
        declared_delta_ids = [
            str(row["id"]) for row in COMMON_DELTA_ROWS
        ] + ["global_singleton_route_identity"]
        if cell.route_id == ROUTE_RA_GLOBAL_SINGLETON_APPEND_REDUCED:
            target_insertion = AppendCommutationReducedInsertion.kind
            new_rows.append(
                {
                    "binding": "insertion_comparison_axis",
                    "classification": (
                        "explicit_user_requested_insertion_axis_v1"
                    ),
                    "field": "insertion_policy",
                    "from": PlateauCommutationInsertion.kind,
                    "id": (
                        "global_singleton_insertion_policy_variant"
                    ),
                    "to": target_insertion,
                }
            )
            declared_delta_ids.append(
                "global_singleton_insertion_policy_variant"
            )
        trace["settings_changed"] = new_rows
        trace["global_singleton_source_anchor"] = {
            "schema": (
                "paper_i_ra_adapt_global_singleton_source_anchor_v1"
            ),
            "anchor_family": (
                "sealed_stationary_core_v13_singleton_plateau_v1"
            ),
            "regime_id": cell.regime_id,
            "nph": cell.nph,
            "scientific_result_anchor_claimed": False,
            "predecessor": {
                "materialization_id": V13_MATERIALIZATION_ID,
                "source_lock_id": predecessor_id,
                "source_lock_canonical_sha256": canonical_sha256(
                    predecessor
                ),
                "source_route_id": ROUTE_RA_SINGLETON_PLATEAU,
                "source_insertion_policy": (
                    PlateauCommutationInsertion.kind
                ),
                "archive": copy.deepcopy(predecessor.get("archive")),
                "member": copy.deepcopy(predecessor.get("member")),
                "core_source_anchor": copy.deepcopy(
                    predecessor_core_anchor
                ),
            },
            "route_derivation": {
                "target_route_id": cell.route_id,
                "target_insertion_policy": target_insertion,
                "declared_delta_ids": declared_delta_ids,
            },
        }
        if (
            derived.get("archive") != predecessor.get("archive")
            or derived.get("member") != predecessor.get("member")
        ):
            raise GlobalSingletonMaterializationError(
                f"Source-byte binding changed for {cell.cell_id}."
            )
        derived_cells[cell.source_lock_id] = derived
        delta_rows.append(
            {
                "cell_id": cell.cell_id,
                "source_lock_id": cell.source_lock_id,
                "predecessor_source_lock_id": predecessor_id,
                "target_route_id": cell.route_id,
                "target_insertion_policy": target_insertion,
                "declared_delta_ids": declared_delta_ids,
                "archive_binding_preserved": True,
                "member_binding_preserved": True,
                "scientific_result_anchor_claimed": False,
            }
        )

    if (
        len(derived_cells) != EXPECTED_CELL_COUNT
        or len(delta_rows) != EXPECTED_CELL_COUNT
    ):
        raise GlobalSingletonMaterializationError(
            "Derived source-lock cardinality drifted."
        )
    derived_source_locks = {
        "schema": sealed.get("schema"),
        "global_sources": copy.deepcopy(sealed.get("global_sources")),
        "cell_locks": derived_cells,
    }
    delta_receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_global_singleton_insertion_"
                "source_lock_delta_v1"
            ),
            "status": "passed",
            "source_materialization": (
                V13_ROOT.relative_to(REPO_ROOT).as_posix()
            ),
            "source_locks_file_sha256": (
                EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
            ),
            "source_cell_count": EXPECTED_SOURCE_CELL_COUNT,
            "derived_cell_count": EXPECTED_CELL_COUNT,
            "fixed_policy_fields": {
                "active_gradient_policy": (
                    ACTIVE_GRADIENT_STATIONARY
                ),
                "resource_weighting_scope": (
                    RESOURCE_WEIGHTING_ALL_PHASE
                ),
                "candidate_adapter_id": (
                    GLOBAL_SINGLE_PAULI_ADAPTER_ID
                ),
                "phase_i_candidate_supply": (
                    PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
                ),
                "phase_i_candidate_visibility": (
                    PHASE_I_VISIBILITY_ALL_EXECUTABLE
                ),
                "phase_ii_candidate_exposure": (
                    PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
                ),
            },
            "insertion_axis": {
                "route_ids": list(
                    GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
                ),
                "typed_policies": [
                    AppendCommutationReducedInsertion.kind,
                    PlateauCommutationInsertion.kind,
                ],
            },
            "rows": delta_rows,
            "all_archive_bindings_preserved": True,
            "all_member_bindings_preserved": True,
            "all_global_source_bindings_preserved": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    return derived_source_locks, delta_receipt


def main() -> int:
    if MATERIALIZATION_ROOT.exists() or MATERIALIZATION_ROOT.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite global-singleton materialization: "
            f"{MATERIALIZATION_ROOT}"
        )
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise GlobalSingletonMaterializationError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )
    if _sha256_file(V13_PROBLEM_BASELINES) != (
        EXPECTED_V13_PROBLEM_BASELINES_FILE_SHA256
    ):
        raise GlobalSingletonMaterializationError(
            "The sealed v13 problem baselines drifted."
        )
    if _sha256_file(V13_FINAL_RECEIPT) != EXPECTED_V13_FINAL_FILE_SHA256:
        raise GlobalSingletonMaterializationError(
            "The sealed v13 final receipt drifted."
        )

    sealed = _sealed_plateau_source_locks()
    source_locks, delta_receipt = _derive_campaign_source_locks(sealed)
    baselines = _load_mapping(
        V13_PROBLEM_BASELINES, label="sealed v13 problem baselines"
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{MATERIALIZATION_ID}.staging.",
            dir=MATERIALIZATIONS_ROOT,
        )
    )
    try:
        source_dir = staging / "source_materialization"
        source_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            source_dir / "sealed_plateau_source_locks.json", sealed
        )
        _write_json(
            source_dir / "source_locks_input.json", source_locks
        )
        _write_json(
            source_dir / "source_lock_delta_receipt.json",
            delta_receipt,
        )
        v12.support._write_bytes_atomic_no_replace(
            source_dir / "problem_baselines.json",
            V13_PROBLEM_BASELINES.read_bytes(),
        )

        receipt = materialize_global_singleton_insertion_bundle(
            staging,
            problem_resolver=v12.support._problem_resolver_from(
                baselines
            ),
            source_locks=source_locks,
            repository_state=_repository_state(),
            repo_root=REPO_ROOT,
            horizon=50,
            dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
            materialization_timestamp=v12.support._utc_now(),
            verify_source_files=True,
        )
        if (
            receipt.bundle_id != GLOBAL_SINGLETON_BUNDLE_ID
            or receipt.materialization_status != "passed"
            or int(receipt.cell_count) != EXPECTED_CELL_COUNT
        ):
            raise GlobalSingletonMaterializationError(
                "Global-singleton bundle receipt cardinality/status drifted."
            )

        bundle_root = staging / GLOBAL_SINGLETON_BUNDLE_ID
        final = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_global_singleton_insertion12_"
                    "materialization_receipt_v1"
                ),
                "status": "passed",
                "materialization_id": MATERIALIZATION_ID,
                "campaign_id": GLOBAL_SINGLETON_CAMPAIGN_ID,
                "bundle_id": GLOBAL_SINGLETON_BUNDLE_ID,
                "run_class": GLOBAL_SINGLETON_RUN_CLASS,
                "source_anchor": {
                    "materialization": (
                        V13_ROOT.relative_to(REPO_ROOT).as_posix()
                    ),
                    "source_locks_file_sha256": (
                        EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
                    ),
                    "problem_baselines_file_sha256": (
                        EXPECTED_V13_PROBLEM_BASELINES_FILE_SHA256
                    ),
                    "final_receipt_file_sha256": (
                        EXPECTED_V13_FINAL_FILE_SHA256
                    ),
                    "selected_source_lock_count": (
                        EXPECTED_SOURCE_CELL_COUNT
                    ),
                    "source_route_id": ROUTE_RA_SINGLETON_PLATEAU,
                    "source_lock_delta_receipt": _binding(
                        source_dir / "source_lock_delta_receipt.json",
                        relative_to=staging,
                    ),
                },
                "fixed_scientific_contract": {
                    "active_gradient_policy": (
                        ACTIVE_GRADIENT_STATIONARY
                    ),
                    "resource_weighting_scope": (
                        RESOURCE_WEIGHTING_ALL_PHASE
                    ),
                    "phase1_cost_term": "enabled_v1",
                    "candidate_adapter_id": (
                        GLOBAL_SINGLE_PAULI_ADAPTER_ID
                    ),
                    "phase_i_candidate_supply": (
                        PHASE_I_SUPPLY_GLOBAL_GUARDED_SINGLETON
                    ),
                    "phase_i_candidate_visibility": (
                        PHASE_I_VISIBILITY_ALL_EXECUTABLE
                    ),
                    "phase_i_shortlist_size": 24,
                    "phase_ii_candidate_exposure": (
                        PHASE_II_EXPOSURE_RETAINED_SINGLETON_IDENTITY
                    ),
                    "phase_ii_shortlist_size": 12,
                    "phase_iii_admission_cardinality": 1,
                    "horizon": 50,
                },
                "insertion_axis": {
                    "route_ids": list(
                        GLOBAL_SINGLETON_INSERTION_ROUTE_IDS
                    ),
                    "append_commutation_reduced": {
                        "kind": (
                            AppendCommutationReducedInsertion.kind
                        ),
                        "runtime_mode": (
                            AppendCommutationReducedInsertion.runtime_mode
                        ),
                        "position_scope": (
                            AppendCommutationReducedInsertion.position_scope
                        ),
                        "equivalence_policy": (
                            AppendCommutationReducedInsertion.equivalence_policy
                        ),
                    },
                    "plateau_commutation": {
                        "kind": PlateauCommutationInsertion.kind,
                        "runtime_mode": (
                            "insertion_commutation_plateau_v1"
                        ),
                        "energy_decrease_threshold": 1e-8,
                        "threshold_comparison": "strictly_below_v1",
                        "patience": 1,
                        "hysteresis_active": False,
                    },
                },
                "pool_authority": {
                    "parent_membership_by_nph": {
                        str(nph): dict(contract)
                        for nph, contract in (
                            SINGLETON_PARENT_MEMBERSHIP_BY_NPH.items()
                        )
                    },
                    "global_executable_membership_by_nph": {
                        str(nph): dict(contract)
                        for nph, contract in (
                            GLOBAL_SINGLETON_POOL_MEMBERSHIP_BY_NPH.items()
                        )
                    },
                    "ordered_pool_sha256_by_regime": dict(
                        GLOBAL_SINGLETON_ORDERED_POOL_SHA256_BY_REGIME
                    ),
                },
                "cell_count": EXPECTED_CELL_COUNT,
                "bundle": {
                    "bundle_manifest": _binding(
                        bundle_root / "bundle_manifest.json",
                        relative_to=staging,
                    ),
                    "source_locks": _binding(
                        bundle_root / "source_locks.json",
                        relative_to=staging,
                    ),
                    "expected_artifacts": _binding(
                        bundle_root / "expected_artifacts.json",
                        relative_to=staging,
                    ),
                    "validation_report": _binding(
                        bundle_root / "validation_report.json",
                        relative_to=staging,
                    ),
                },
                "cross_arm_equality": (
                    "all_common_scientific_fields_equal_outside_"
                    "insertion_v1"
                ),
                "execution_authorized": False,
                "submission_authorized": False,
                "submission_state": "not_submitted",
                "submitted": False,
                "remote_stage": False,
                "condor_submit": False,
            }
        )
        _write_json(staging / FINAL_RECEIPT_NAME, final)

        v12.support._darwin_renameatx_np()
        v12.support._atomic_rename_no_replace(
            staging, MATERIALIZATION_ROOT
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "materialization_root": (
                        MATERIALIZATION_ROOT.relative_to(
                            REPO_ROOT
                        ).as_posix()
                    ),
                    "final_receipt_sha256": final["sha256"],
                    "cell_count": EXPECTED_CELL_COUNT,
                    "execution_authorized": False,
                    "submission_authorized": False,
                    "submitted": False,
                },
                sort_keys=True,
            )
        )
        return 0
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
