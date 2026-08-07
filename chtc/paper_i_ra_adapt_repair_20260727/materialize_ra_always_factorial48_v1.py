#!/usr/bin/env python3
"""Materialize the inert 48-cell corrected-always factorial.

The source lock is the sealed v13 corrected-always subset.  The four bundles
cross only the active-gradient policy and the Phase-I resource-weighting
scope.  This command has no scientific-execution or scheduler-submission
seam and refuses to overwrite an existing materialization.
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
from pipelines.static_adapt.ra_adapt.bundles import (  # noqa: E402
    FACTORIAL_BUNDLE_POLICIES,
    FACTORIAL_CAMPAIGN_ID,
    FACTORIAL_RUN_CLASS,
    materialize_factorial_always_bundles,
)
from pipelines.static_adapt.ra_adapt.contracts import (  # noqa: E402
    canonical_json_bytes,
    canonical_sha256,
)


REPAIR_ROOT = REPO_ROOT / "chtc/paper_i_ra_adapt_repair_20260727"
MATERIALIZATIONS_ROOT = REPAIR_ROOT / "bundles/materializations"
MATERIALIZATION_ID = "ra_adapt_always_factorial48_v1"
MATERIALIZATION_ROOT = MATERIALIZATIONS_ROOT / MATERIALIZATION_ID
V13_ROOT = MATERIALIZATIONS_ROOT / "ra_adapt_stationary_late_core_v13"
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
EXPECTED_ARM_COUNT = 4
EXPECTED_CELL_COUNT_PER_ARM = 12
EXPECTED_TOTAL_CELL_COUNT = 48
FINAL_RECEIPT_NAME = "factorial_materialization_receipt.json"


class FactorialMaterializationError(ValueError):
    """Raised when the factorial materialization does not close."""


def _sha256_file(path: Path) -> str:
    return v12.support._hash_file(path)


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FactorialMaterializationError(
            f"{label} is missing or unsafe: {path}"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FactorialMaterializationError(
            f"{label} is not valid JSON."
        ) from exc
    if not isinstance(payload, dict):
        raise FactorialMaterializationError(
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


def _factorial_source_locks() -> dict[str, Any]:
    if _sha256_file(V13_SOURCE_LOCKS) != (
        EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
    ):
        raise FactorialMaterializationError(
            "The sealed v13 source-lock input drifted."
        )
    source = _load_mapping(
        V13_SOURCE_LOCKS, label="sealed v13 source locks"
    )
    cell_locks = source.get("cell_locks")
    if not isinstance(cell_locks, Mapping):
        raise FactorialMaterializationError(
            "The sealed v13 source lock has no cell map."
        )
    selected = {
        lock_id: row
        for lock_id, row in sorted(cell_locks.items())
        if isinstance(row, Mapping)
        and row.get("route_id")
        in {"ra_macro_always", "ra_singleton_always"}
    }
    if len(selected) != EXPECTED_CELL_COUNT_PER_ARM:
        raise FactorialMaterializationError(
            "The sealed v13 source lock does not contain exactly twelve "
            "corrected-always cells."
        )
    for lock_id, row in selected.items():
        changes = row.get("resolver_trace", {}).get("settings_changed", [])
        insertion = [
            change
            for change in changes
            if isinstance(change, Mapping)
            and change.get("field") == "insertion_policy"
        ]
        if (
            len(insertion) != 1
            or insertion[0].get("to")
            != "always_commutation_reduced"
        ):
            raise FactorialMaterializationError(
                f"Source lock is not corrected-always: {lock_id}."
            )
    return {
        "schema": source.get("schema"),
        "global_sources": source.get("global_sources"),
        "cell_locks": selected,
    }


def _is_factor_axis_row(row: Any) -> bool:
    return isinstance(row, Mapping) and row.get("field") in {
        "active_gradient_policy",
        "resource_weighting_scope",
    }


def _derive_source_locks_by_bundle(
    base: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    base_cells = base.get("cell_locks")
    if not isinstance(base_cells, Mapping):
        raise FactorialMaterializationError(
            "Factorial base source locks have no cell map."
        )
    by_bundle: dict[str, dict[str, Any]] = {}
    delta_rows: list[dict[str, Any]] = []
    for (
        bundle_id,
        active_gradient_policy,
        resource_weighting_scope,
    ) in FACTORIAL_BUNDLE_POLICIES:
        derived = copy.deepcopy(dict(base))
        derived_cells = derived.get("cell_locks")
        if not isinstance(derived_cells, dict):
            raise FactorialMaterializationError(
                "Derived source locks have no mutable cell map."
            )
        for lock_id in sorted(derived_cells):
            before = base_cells[lock_id]
            after = derived_cells[lock_id]
            if not isinstance(before, Mapping) or not isinstance(after, dict):
                raise FactorialMaterializationError(
                    f"Malformed factorial source lock: {lock_id}."
                )
            before_trace = before.get("resolver_trace")
            after_trace = after.get("resolver_trace")
            if (
                not isinstance(before_trace, Mapping)
                or not isinstance(after_trace, dict)
            ):
                raise FactorialMaterializationError(
                    f"Source lock has no resolver trace: {lock_id}."
                )
            before_changes = before_trace.get("settings_changed")
            after_changes = after_trace.get("settings_changed")
            if not isinstance(before_changes, list) or not isinstance(
                after_changes, list
            ):
                raise FactorialMaterializationError(
                    f"Source lock has no settings change list: {lock_id}."
                )
            predecessor_axis = [
                copy.deepcopy(row)
                for row in before_changes
                if _is_factor_axis_row(row)
            ]
            non_axis_before = [
                copy.deepcopy(row)
                for row in before_changes
                if not _is_factor_axis_row(row)
            ]
            predecessor_gradient_rows = [
                row
                for row in before_changes
                if isinstance(row, Mapping)
                and row.get("field") == "active_gradient_policy"
            ]
            if len(predecessor_gradient_rows) != 1:
                raise FactorialMaterializationError(
                    "Expected exactly one active-gradient declaration for "
                    f"{lock_id}."
                )
            gradient_id = predecessor_gradient_rows[0].get("id")
            if gradient_id not in {
                "study1_axis",
                "core_stationary_gradient_policy",
            }:
                raise FactorialMaterializationError(
                    f"Unexpected active-gradient declaration: {lock_id}."
                )
            predecessor_d5_rows = [
                row
                for row in before_changes
                if isinstance(row, Mapping) and row.get("id") == "D5"
            ]
            if len(predecessor_d5_rows) > 1:
                raise FactorialMaterializationError(
                    f"Duplicate D5 declaration: {lock_id}."
                )
            after_trace["settings_changed"] = [
                *copy.deepcopy(non_axis_before),
                {
                    "binding": "factorial_bundle_axis",
                    "classification": (
                        "explicit_user_requested_factorial_axis_v1"
                    ),
                    "field": "active_gradient_policy",
                    "id": gradient_id,
                    "to": active_gradient_policy,
                },
                {
                    "binding": "factorial_bundle_axis",
                    "classification": (
                        "explicit_user_requested_factorial_axis_v1"
                    ),
                    "field": "resource_weighting_scope",
                    "id": "D5",
                    "to": resource_weighting_scope,
                },
            ]
            derived_axis = [
                copy.deepcopy(row)
                for row in after_trace["settings_changed"]
                if _is_factor_axis_row(row)
            ]

            before_without_axes = copy.deepcopy(dict(before))
            after_without_axes = copy.deepcopy(dict(after))
            before_without_axes["resolver_trace"]["settings_changed"] = (
                non_axis_before
            )
            after_without_axes["resolver_trace"]["settings_changed"] = (
                copy.deepcopy(non_axis_before)
            )
            if before_without_axes != after_without_axes:
                raise FactorialMaterializationError(
                    "Factor derivation changed a non-axis source-lock field: "
                    f"{bundle_id}:{lock_id}."
                )
            if (
                after.get("archive") != before.get("archive")
                or after.get("member") != before.get("member")
            ):
                raise FactorialMaterializationError(
                    f"Factor derivation changed source bytes: {lock_id}."
                )
            delta_rows.append(
                {
                    "bundle_id": bundle_id,
                    "source_lock_id": lock_id,
                    "active_gradient_policy": active_gradient_policy,
                    "resource_weighting_scope": (
                        resource_weighting_scope
                    ),
                    "predecessor_axis_declarations": predecessor_axis,
                    "derived_axis_declarations": derived_axis,
                    "non_axis_source_lock_equal": True,
                    "archive_binding_preserved": True,
                    "member_binding_preserved": True,
                }
            )
        if derived.get("global_sources") != base.get("global_sources"):
            raise FactorialMaterializationError(
                f"Global source locks drifted for {bundle_id}."
            )
        by_bundle[bundle_id] = derived

    if (
        len(by_bundle) != EXPECTED_ARM_COUNT
        or len(delta_rows) != EXPECTED_TOTAL_CELL_COUNT
    ):
        raise FactorialMaterializationError(
            "Factor source-lock delta cardinality drifted."
        )
    receipt = _digested(
        {
            "schema": (
                "paper_i_ra_adapt_always_factorial_source_lock_delta_v1"
            ),
            "status": "passed",
            "source_materialization": (
                V13_ROOT.relative_to(REPO_ROOT).as_posix()
            ),
            "source_locks_file_sha256": (
                EXPECTED_V13_SOURCE_LOCKS_FILE_SHA256
            ),
            "allowed_changed_fields": [
                "active_gradient_policy",
                "resource_weighting_scope",
            ],
            "arm_count": EXPECTED_ARM_COUNT,
            "row_count": EXPECTED_TOTAL_CELL_COUNT,
            "rows": delta_rows,
            "all_non_axis_source_lock_fields_equal": True,
            "all_archive_bindings_preserved": True,
            "all_member_bindings_preserved": True,
            "all_global_source_bindings_preserved": True,
            "execution_authorized": False,
            "submission_authorized": False,
            "submitted": False,
        }
    )
    return by_bundle, receipt


def main() -> int:
    if MATERIALIZATION_ROOT.exists() or MATERIALIZATION_ROOT.is_symlink():
        raise FileExistsError(
            "Refusing to overwrite factorial materialization: "
            f"{MATERIALIZATION_ROOT}"
        )
    if not MATERIALIZATIONS_ROOT.is_dir():
        raise FactorialMaterializationError(
            f"Materializations root is missing: {MATERIALIZATIONS_ROOT}"
        )
    if _sha256_file(V13_PROBLEM_BASELINES) != (
        EXPECTED_V13_PROBLEM_BASELINES_FILE_SHA256
    ):
        raise FactorialMaterializationError(
            "The sealed v13 problem baselines drifted."
        )
    if _sha256_file(V13_FINAL_RECEIPT) != EXPECTED_V13_FINAL_FILE_SHA256:
        raise FactorialMaterializationError(
            "The sealed v13 final receipt drifted."
        )

    source_locks = _factorial_source_locks()
    source_locks_by_bundle, factor_delta = (
        _derive_source_locks_by_bundle(source_locks)
    )
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
        _write_json(source_dir / "source_locks_input.json", source_locks)
        inputs_dir = source_dir / "source_locks_by_bundle"
        inputs_dir.mkdir(parents=True, exist_ok=False)
        for bundle_id, arm_locks in source_locks_by_bundle.items():
            _write_json(inputs_dir / f"{bundle_id}.json", arm_locks)
        _write_json(
            source_dir / "factor_delta_receipt.json", factor_delta
        )
        v12.support._write_bytes_atomic_no_replace(
            source_dir / "problem_baselines.json",
            V13_PROBLEM_BASELINES.read_bytes(),
        )

        receipts = materialize_factorial_always_bundles(
            staging,
            problem_resolver=v12.support._problem_resolver_from(baselines),
            source_locks_by_bundle=source_locks_by_bundle,
            repository_state=_repository_state(),
            repo_root=REPO_ROOT,
            horizon=50,
            dependency_lock_paths=(REPO_ROOT / "requirements.txt",),
            materialization_timestamp=v12.support._utc_now(),
            verify_source_files=True,
        )
        if (
            len(receipts) != EXPECTED_ARM_COUNT
            or sum(int(receipt.cell_count) for receipt in receipts)
            != EXPECTED_TOTAL_CELL_COUNT
            or any(
                receipt.materialization_status != "passed"
                or int(receipt.cell_count) != EXPECTED_CELL_COUNT_PER_ARM
                for receipt in receipts
            )
        ):
            raise FactorialMaterializationError(
                "Factorial bundle receipt cardinality/status drifted."
            )

        arm_rows: list[dict[str, Any]] = []
        for (
            bundle_id,
            active_gradient_policy,
            resource_weighting_scope,
        ), receipt in zip(FACTORIAL_BUNDLE_POLICIES, receipts):
            if receipt.bundle_id != bundle_id:
                raise FactorialMaterializationError(
                    "Factorial bundle ordering drifted."
                )
            bundle_root = staging / bundle_id
            arm_rows.append(
                {
                    "bundle_id": bundle_id,
                    "active_gradient_policy": active_gradient_policy,
                    "resource_weighting_scope": resource_weighting_scope,
                    "cell_count": int(receipt.cell_count),
                    "bundle_manifest": _binding(
                        bundle_root / "bundle_manifest.json",
                        relative_to=staging,
                    ),
                    "source_locks": _binding(
                        bundle_root / "source_locks.json",
                        relative_to=staging,
                    ),
                    "validation_report": _binding(
                        bundle_root / "validation_report.json",
                        relative_to=staging,
                    ),
                }
            )

        final = _digested(
            {
                "schema": (
                    "paper_i_ra_adapt_always_factorial48_"
                    "materialization_receipt_v1"
                ),
                "status": "passed",
                "materialization_id": MATERIALIZATION_ID,
                "campaign_id": FACTORIAL_CAMPAIGN_ID,
                "run_class": FACTORIAL_RUN_CLASS,
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
                        EXPECTED_CELL_COUNT_PER_ARM
                    ),
                    "insertion_policy": "always_commutation_reduced",
                    "factor_delta_receipt": _binding(
                        source_dir / "factor_delta_receipt.json",
                        relative_to=staging,
                    ),
                },
                "factorial_axes": {
                    "active_gradient_policy": [
                        "stationary_source_response_v1",
                        "measured_residual_response_v1",
                    ],
                    "resource_weighting_scope": [
                        "late_resource_weighting_v1",
                        "all_phase_resource_weighting_v1",
                    ],
                    "phase1_cost_off_semantics": (
                        "unit_phase1_effective_burden_with_raw_telemetry_"
                        "preserved_v1"
                    ),
                    "phase1_cost_on_semantics": (
                        "raw_phase1_burden_applied_v1"
                    ),
                    "all_non_axis_scientific_fields_source_locked": True,
                },
                "arm_count": EXPECTED_ARM_COUNT,
                "cell_count_per_arm": EXPECTED_CELL_COUNT_PER_ARM,
                "total_cell_count": EXPECTED_TOTAL_CELL_COUNT,
                "arms": arm_rows,
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
                    "arm_count": EXPECTED_ARM_COUNT,
                    "total_cell_count": EXPECTED_TOTAL_CELL_COUNT,
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
