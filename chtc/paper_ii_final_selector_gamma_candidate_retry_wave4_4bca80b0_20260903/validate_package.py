#!/usr/bin/env python3
"""Fail-closed validation for selector retry wave 4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile


PACKAGE = Path(__file__).resolve().parent
EXPECTED_ACTION_COUNT = 1
EXPECTED_PARENT_CLUSTER_ID = 9718529
EXPECTED_CODE_REVISION = "4bca80b01f3d4e20e74f21b21ec36d23374b3392"
EXPECTED_SOURCE_SHA256 = (
    "8f12755668b793062650801eab5755b81ff64c20e851f15e9b0ec18813e047d3"
)
EXPECTED_ROUTES = {
    "paper_ii_exchange_q_compiled_v1",
    "paper_ii_exchange_gamma_compiled_v1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    image = Path(args[0]) if args else None
    manifest = json.loads((PACKAGE / "input/manifest.json").read_text(encoding="utf-8"))
    require(
        manifest.get("schema") == "paper_ii_final_selector_gamma_candidate_retry_package_v1",
        "manifest schema",
    )
    require(manifest["scope"]["action_count"] == EXPECTED_ACTION_COUNT, "manifest action count")
    require(manifest["scope"]["apm_action_count"] == EXPECTED_ACTION_COUNT, "manifest APM count")
    require(manifest["scope"]["avqds_action_count"] == 0, "manifest AVQDS count")
    require(manifest["scope"]["precursor_action_count"] == 0, "manifest precursor count")
    require(manifest["scope"]["terminal_parent_failures_only"] is True, "retry scope")
    require(manifest["parent"]["cluster_id"] == EXPECTED_PARENT_CLUSTER_ID, "parent cluster")
    require(manifest["source"]["code_revision"] == EXPECTED_CODE_REVISION, "repair revision")
    require(manifest["source"]["archive_sha256"] == EXPECTED_SOURCE_SHA256, "source digest")
    require(
        bool(set(manifest["routes"]))
        and set(manifest["routes"]).issubset(EXPECTED_ROUTES),
        "retry route set",
    )
    require(manifest["inputs"]["report_refresh_resolution"]["active_blocks"] == [], "active block")
    require(manifest["inputs"]["calibration_receipt"]["error_count"] == 0, "calibration error")

    for entry in manifest["inputs"].values():
        path = PACKAGE / entry["path"]
        require(path.is_file(), f"missing input: {path}")
        require(sha256(path) == entry["sha256"], f"input sha drift: {path}")
    for relative, expected in manifest["package_files"].items():
        path = PACKAGE / relative
        require(path.is_file(), f"missing package file: {path}")
        require(sha256(path) == expected, f"package-file sha drift: {path}")
    source = PACKAGE / manifest["source"]["archive"]
    require(source.is_file(), "source archive absent")
    require(sha256(source) == manifest["source"]["archive_sha256"], "source archive sha drift")
    with tarfile.open(source, "r:gz") as handle:
        members = set(handle.getnames())
    for required in (
        "pipelines/time_dynamics/runners/ap_append_from_adapt_artifact.py",
        "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
        "pipelines/time_dynamics/ap_mclachlan/exchange_structural.py",
        "pipelines/time_dynamics/ap_mclachlan/compiled_patch_cost.py",
        "pipelines/time_dynamics/ap_mclachlan/objective_ordinates.py",
        "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json",
    ):
        require(required in members, f"source archive misses {required}")

    rows = [
        json.loads(line)
        for line in (PACKAGE / "input/actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = (PACKAGE / "input/action_ids.txt").read_text(encoding="utf-8").splitlines()
    require(len(rows) == len(ids) == EXPECTED_ACTION_COUNT, "action row count")
    require([row["action_id"] for row in rows] == ids, "action order/id drift")
    require(len(set(ids)) == EXPECTED_ACTION_COUNT, "duplicate action ids")
    require({row["method"] for row in rows} == {"apm"}, "non-APM action")
    require(all(row["comparator"]["action"] is None for row in rows), "comparator action")
    require(
        {row["exchange_route_bundle"] for row in rows} == set(manifest["routes"]),
        "route set drift",
    )
    require(
        all(row["code_revision"] == manifest["source"]["code_revision"] for row in rows),
        "code revision drift",
    )
    require(
        all(row["source_package_sha256"] == manifest["source"]["archive_sha256"] for row in rows),
        "source digest not bound to every action",
    )
    require(
        all("--support-patch-scoring-workers" in row["runner_argv"] for row in rows),
        "worker flag absent",
    )
    require(
        all("--resume-from-run-json" not in row["runner_argv"] for row in rows),
        "precursor resume action present",
    )
    require(
        all(
            row["runner_argv"][row["runner_argv"].index("--t-final") + 1] == "20.0"
            and row["runner_argv"][row["runner_argv"].index("--num-times") + 1] == "501"
            for row in rows
        ),
        "time-grid drift",
    )

    parent = json.loads((PACKAGE / "input/retry_parent_failures.json").read_text(encoding="utf-8"))
    require(parent.get("schema") == "paper_ii_final_selector_retry_parent_failures_v1", "parent schema")
    require(parent.get("failure_count") == EXPECTED_ACTION_COUNT, "parent failure count")
    require(parent["parent"]["cluster_id"] == EXPECTED_PARENT_CLUSTER_ID, "parent cluster receipt")
    require(parent["repair"]["code_revision"] == EXPECTED_CODE_REVISION, "parent repair revision")
    require(parent["repair"]["source_archive_sha256"] == EXPECTED_SOURCE_SHA256, "parent source digest")
    failures = {str(item["action_id"]): item for item in parent["failures"]}
    require(len(failures) == EXPECTED_ACTION_COUNT, "parent failure ids")
    require(set(failures) == set(ids), "retry/parent action mismatch")
    require(
        all(
            item["job_status"] == 4
            and item["exit_code"] == 2
            and item["exit_by_signal"] is False
            and item["final_error"].startswith(
                "error: ideal Gamma violated nested monotonicity for a pure insertion:"
            )
            for item in failures.values()
        ),
        "unverified parent failure",
    )
    require(
        all(
            row["retry_parent_cluster_id"] == EXPECTED_PARENT_CLUSTER_ID
            and row["retry_parent_proc_id"] == failures[row["action_id"]]["proc_id"]
            and row["retry_parent_stderr_sha256"]
            == failures[row["action_id"]]["stderr_sha256"]
            and row["retry_wave"] == 4
            for row in rows
        ),
        "retry provenance drift",
    )
    if image is not None:
        require(image.is_file(), f"container absent: {image}")
        require(sha256(image) == manifest["runtime"]["container_sha256"], "container sha drift")
    print(
        json.dumps(
            {
                "status": "OK",
                "action_count": len(rows),
                "parent_cluster_id": EXPECTED_PARENT_CLUSTER_ID,
                "source_sha256": sha256(source),
                "container_checked": image is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
