#!/usr/bin/env python3
"""Build an explicit, self-digested selector for preserved CHTC attempts.

Fetched archives are immutable and attempt-qualified.  This helper binds one
cluster/proc identity (and, when present, its exact bytes) for every direct
execution.  Validation never chooses an attempt by mtime or filename order.
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
    ATTEMPT_SELECTION_SCHEMA,
    EXECUTION_PLAN_SCHEMA,
    PACKAGE_ID,
    PackageContractError,
    atomic_write_json,
    digested,
    direct_execution_ids,
    load_json_object,
    sha256_file,
    verify_exact_key_set,
    verify_self_digest,
)


ATTEMPT_DISPOSITIONS = frozenset({"validate", "blocked", "superseded"})


def attempt_archive_name(
    execution_id: str,
    *,
    cluster_id: int,
    proc_id: int,
) -> str:
    if execution_id not in direct_execution_ids():
        raise PackageContractError(
            f"Unknown direct execution ID: {execution_id}"
        )
    for label, value in (("cluster_id", cluster_id), ("proc_id", proc_id)):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise PackageContractError(
                f"Attempt {label} must be a nonnegative integer."
            )
    return (
        f"{execution_id}__cluster_{int(cluster_id)}"
        f"__proc_{int(proc_id)}.tar.gz"
    )


def _attempt_map_from_file(path: Path) -> Mapping[str, Any]:
    payload = load_json_object(path, label="attempt map")
    rows = payload.get("selections", payload)
    if not isinstance(rows, Mapping):
        raise PackageContractError("Attempt map selections must be an object.")
    return rows


def build_attempt_selection(
    *,
    plan_path: Path,
    fetched_dir: Path,
    cluster_id: int | None = None,
    attempt_map: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind exactly one explicit attempt identity for every direct job."""

    if (cluster_id is None) == (attempt_map is None):
        raise PackageContractError(
            "Select exactly one of cluster_id or attempt_map."
        )
    plan = load_json_object(plan_path, label="execution plan")
    verify_self_digest(plan, label="execution plan")
    if (
        plan.get("schema") != EXECUTION_PLAN_SCHEMA
        or plan.get("package_id") != PACKAGE_ID
        or [
            row.get("execution_id")
            for row in plan.get("direct_executions", ())
            if isinstance(row, Mapping)
        ]
        != list(direct_execution_ids())
    ):
        raise PackageContractError(
            "Attempt selector received a drifted execution plan."
        )
    if not fetched_dir.is_dir() or fetched_dir.is_symlink():
        raise PackageContractError(
            f"Fetched directory is unavailable or unsafe: {fetched_dir}"
        )

    if cluster_id is not None:
        if (
            isinstance(cluster_id, bool)
            or int(cluster_id) != cluster_id
            or int(cluster_id) < 0
        ):
            raise PackageContractError(
                "Initial cluster ID must be a nonnegative integer."
            )
        resolved_map: Mapping[str, Any] = {
            execution_id: {
                "cluster_id": int(cluster_id),
                "proc_id": proc_id,
                "disposition": "validate",
            }
            for proc_id, execution_id in enumerate(direct_execution_ids())
        }
        selection_kind = "full_queue_cluster_process_order_v1"
    else:
        assert attempt_map is not None
        verify_exact_key_set(
            attempt_map,
            direct_execution_ids(),
            label="explicit attempt-map execution IDs",
        )
        resolved_map = attempt_map
        selection_kind = "explicit_mixed_attempt_map_v1"

    selections: list[dict[str, Any]] = []
    for execution_id in direct_execution_ids():
        raw = resolved_map.get(execution_id)
        if not isinstance(raw, Mapping):
            raise PackageContractError(
                f"Attempt selection is invalid for {execution_id}."
            )
        cluster = raw.get("cluster_id")
        proc = raw.get("proc_id")
        if (
            isinstance(cluster, bool)
            or isinstance(proc, bool)
            or not isinstance(cluster, int)
            or not isinstance(proc, int)
        ):
            raise PackageContractError(
                f"Attempt identity must be integral for {execution_id}."
            )
        disposition = str(raw.get("disposition", "validate"))
        if disposition not in ATTEMPT_DISPOSITIONS:
            raise PackageContractError(
                f"Unknown attempt disposition for {execution_id}: "
                f"{disposition!r}."
            )
        archive_name = attempt_archive_name(
            execution_id,
            cluster_id=cluster,
            proc_id=proc,
        )
        archive_path = fetched_dir / archive_name
        archive_present = archive_path.is_file() and not archive_path.is_symlink()
        if archive_path.exists() and not archive_present:
            raise PackageContractError(
                f"Attempt archive is not a regular file: {archive_path}"
            )
        selections.append(
            {
                "execution_id": execution_id,
                "cluster_id": cluster,
                "proc_id": proc,
                "archive_name": archive_name,
                "disposition": disposition,
                "archive_present": archive_present,
                "archive_sha256": (
                    sha256_file(archive_path) if archive_present else None
                ),
                "archive_size_bytes": (
                    archive_path.stat().st_size if archive_present else None
                ),
            }
        )
    return digested(
        {
            "schema": ATTEMPT_SELECTION_SCHEMA,
            "package_id": PACKAGE_ID,
            "execution_plan_sha256": plan["sha256"],
            "selection_kind": selection_kind,
            "selection_policy": (
                "explicit_identity_never_mtime_or_lexical_latest_v1"
            ),
            "direct_execution_count": 18,
            "selections": selections,
            "status": (
                "ready"
                if all(
                    row["disposition"] == "validate"
                    and row["archive_present"] is True
                    for row in selections
                )
                else "incomplete"
            ),
        }
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plan",
        type=Path,
        default=PACKAGE_DIR / "execution_plan.json",
    )
    parser.add_argument("--fetched-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--cluster-id", type=int)
    group.add_argument("--attempt-map", type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        attempt_map = (
            None
            if args.attempt_map is None
            else _attempt_map_from_file(args.attempt_map.resolve())
        )
        receipt = build_attempt_selection(
            plan_path=args.plan.resolve(),
            fetched_dir=args.fetched_dir.resolve(),
            cluster_id=args.cluster_id,
            attempt_map=attempt_map,
        )
        output = args.output.resolve()
        if output.exists():
            raise PackageContractError(
                f"Refusing to overwrite attempt selector: {output}"
            )
        atomic_write_json(output, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (OSError, PackageContractError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
