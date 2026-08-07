#!/usr/bin/env python3
"""Validate and receipt the two Study-1 shared Append result references.

No measured-policy Append artifact is copied or fabricated.  Each receipt
authenticates the equality projection of the two immutable Append protocols and
maps the measured logical artifact roles to hashes of the single canonical
stationary-policy execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    EXPECTED_ARTIFACT_ROLES,
    PACKAGE_ID,
    SHARED_APPEND_EQUIVALENCE_SCHEMA,
    STATIONARY_BUNDLE_ID,
    PackageContractError,
    atomic_write_json,
    canonical_sha256,
    digested,
    load_json_object,
    safe_relative_path,
    sha256_file,
    shared_append_rows,
    verify_self_digest,
)


def _plan_logical_rows(
    plan: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    raw = plan.get("logical_cells")
    if not isinstance(raw, list):
        raise PackageContractError("Execution plan has no logical cells.")
    rows: dict[str, Mapping[str, Any]] = {}
    for row in raw:
        if not isinstance(row, Mapping):
            raise PackageContractError("Execution plan logical row is invalid.")
        key = str(row.get("logical_key", ""))
        if key in rows:
            raise PackageContractError(f"Duplicate logical plan row: {key}")
        rows[key] = row
    return rows


def build_shared_append_receipts(
    *,
    source_root: Path,
    fetched_root: Path,
    output_dir: Path,
    plan: Mapping[str, Any],
    dedupe: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate both shared Append references and write two receipt files."""

    verify_self_digest(plan, label="execution plan")
    verify_self_digest(dedupe, label="Study-1 dedupe contract")
    if (
        plan.get("package_id") != PACKAGE_ID
        or plan.get("study1_dedupe_sha256") != dedupe.get("sha256")
    ):
        raise PackageContractError("Plan/dedupe authority drifted.")
    projection = dedupe.get("scientific_equivalence_projection")
    if not isinstance(projection, Mapping):
        raise PackageContractError("Dedupe equivalence projection is missing.")
    required_equal_fields = projection.get("required_equal_fields")
    if not isinstance(required_equal_fields, list) or not required_equal_fields:
        raise PackageContractError("Dedupe equality field list is empty.")
    logical_rows = _plan_logical_rows(plan)
    direct_rows = {
        str(row["execution_id"]): row
        for row in plan.get("direct_executions", [])
        if isinstance(row, Mapping)
    }
    receipts: list[dict[str, Any]] = []
    for shared in shared_append_rows():
        canonical_key = (
            f"{STATIONARY_BUNDLE_ID}::{shared['cell_id']}"
        )
        reference_key = shared["reference_logical_key"]
        canonical_row = logical_rows.get(canonical_key)
        reference_row = logical_rows.get(reference_key)
        direct = direct_rows.get(shared["canonical_execution_id"])
        if (
            not isinstance(canonical_row, Mapping)
            or not isinstance(reference_row, Mapping)
            or not isinstance(direct, Mapping)
            or canonical_row.get("direct_execution_required") is not True
            or reference_row.get("direct_execution_required") is not False
            or reference_row.get("canonical_execution_id")
            != shared["canonical_execution_id"]
            or reference_row.get("execution_fulfillment", {}).get(
                "fulfillment_kind"
            )
            != "shared_result_reference_v1"
        ):
            raise PackageContractError(
                f"Shared Append plan identity drifted: {reference_key}"
            )

        canonical_protocol_path = source_root / safe_relative_path(
            canonical_row["protocol"]["path"],
            label="canonical Append protocol path",
        )
        reference_protocol_path = source_root / safe_relative_path(
            reference_row["protocol"]["path"],
            label="reference Append protocol path",
        )
        canonical_protocol = load_json_object(
            canonical_protocol_path, label="canonical Append protocol"
        )
        reference_protocol = load_json_object(
            reference_protocol_path, label="reference Append protocol"
        )
        verify_self_digest(
            canonical_protocol, label="canonical Append protocol"
        )
        verify_self_digest(
            reference_protocol, label="reference Append protocol"
        )
        if (
            sha256_file(canonical_protocol_path)
            != canonical_row["protocol"]["file_sha256"]
            or sha256_file(reference_protocol_path)
            != reference_row["protocol"]["file_sha256"]
        ):
            raise PackageContractError("Append protocol file hash drifted.")
        comparisons = []
        for field in required_equal_fields:
            if not isinstance(field, str) or not field:
                raise PackageContractError(
                    "Dedupe required-equal field is invalid."
                )
            canonical_value = canonical_protocol.get(field)
            reference_value = reference_protocol.get(field)
            equal = canonical_value == reference_value
            comparisons.append(
                {
                    "field": field,
                    "canonical_value_sha256": canonical_sha256(
                        canonical_value
                    ),
                    "reference_value_sha256": canonical_sha256(
                        reference_value
                    ),
                    "equal": equal,
                }
            )
            if not equal:
                raise PackageContractError(
                    f"Shared Append protocol projection differs at {field}: "
                    f"{shared['regime_id']}"
                )

        output_comparisons = []
        for role in EXPECTED_ARTIFACT_ROLES:
            canonical_relative = safe_relative_path(
                direct["artifact_paths"][role],
                label=f"canonical Append {role}",
            ).as_posix()
            reference_expected = reference_row[
                "expected_run_artifacts"
            ][role]
            if (
                reference_expected.get("direct_file_required") is not False
                or reference_expected.get("reference_receipt_required")
                is not True
                or reference_expected.get("fulfillment_kind")
                != "shared_result_reference_v1"
            ):
                raise PackageContractError(
                    f"Measured Append {role} is not reference-only."
                )
            measured_relative = (
                f"{reference_row['protocol']['path'].rsplit('/protocols/', 1)[0]}/"
                f"{reference_expected['path']}"
            )
            measured_path = fetched_root / safe_relative_path(
                measured_relative,
                label=f"measured Append {role} physical path",
            )
            if measured_path.exists():
                raise PackageContractError(
                    "Measured Append reference fulfillment must not have a "
                    f"copied/fabricated physical artifact: {measured_relative}"
                )
            canonical_path = fetched_root / canonical_relative
            if not canonical_path.is_file() or canonical_path.is_symlink():
                raise PackageContractError(
                    f"Canonical Append output is missing: {canonical_relative}"
                )
            output_hash = sha256_file(canonical_path)
            output_comparisons.append(
                {
                    "role": role,
                    "canonical_path": canonical_relative,
                    "canonical_sha256": output_hash,
                    "reference_expected_path": measured_relative,
                    "reference_sha256": output_hash,
                    "equal": True,
                    "reference_file_materialized": False,
                }
            )

        receipt = digested(
            {
                "schema": SHARED_APPEND_EQUIVALENCE_SCHEMA,
                "package_id": PACKAGE_ID,
                "study1_dedupe_sha256": dedupe["sha256"],
                "group_id": reference_row["execution_fulfillment"][
                    "group_id"
                ],
                "regime_id": shared["regime_id"],
                "cell_id": shared["cell_id"],
                "canonical_execution_id": shared[
                    "canonical_execution_id"
                ],
                "canonical_logical_key": canonical_key,
                "reference_logical_key": reference_key,
                "canonical_protocol": dict(canonical_row["protocol"]),
                "reference_protocol": dict(reference_row["protocol"]),
                "required_equal_fields": list(required_equal_fields),
                "field_comparisons": comparisons,
                "all_required_fields_equal": True,
                "output_hash_comparisons": output_comparisons,
                "all_output_hashes_equal": True,
                "reference_fulfillment": {
                    "fulfillment_kind": "shared_result_reference_v1",
                    "physical_files_materialized": False,
                    "completion_matrix_status": "done",
                },
            }
        )
        destination = (
            output_dir
            / "shared_references"
            / f"{shared['regime_id']}__append_macro_equivalence.json"
        )
        atomic_write_json(destination, receipt)
        receipts.append(
            {
                "regime_id": shared["regime_id"],
                "reference_logical_key": reference_key,
                "path": destination.relative_to(output_dir).as_posix(),
                "canonical_sha256": receipt["sha256"],
                "file_sha256": sha256_file(destination),
            }
        )
    return receipts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fetched-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-plan", type=Path, required=True)
    parser.add_argument("--dedupe-contract", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.output_dir.exists():
            raise PackageContractError(
                f"Refusing to overwrite output directory: {args.output_dir}"
            )
        args.output_dir.mkdir(parents=True)
        plan = load_json_object(
            args.execution_plan, label="execution plan"
        )
        dedupe = load_json_object(
            args.dedupe_contract, label="dedupe contract"
        )
        receipts = build_shared_append_receipts(
            source_root=args.source_root.resolve(),
            fetched_root=args.fetched_root.resolve(),
            output_dir=args.output_dir.resolve(),
            plan=plan,
            dedupe=dedupe,
        )
        print(json.dumps(receipts, indent=2, sort_keys=True))
        return 0
    except (PackageContractError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
