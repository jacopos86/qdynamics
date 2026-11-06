#!/usr/bin/env python3
"""Fail-closed validation for the 45-action selector package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile


PACKAGE = Path(__file__).resolve().parent


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
    require(manifest.get("schema") == "paper_ii_final_selector_comparison_package_v1", "manifest schema")
    require(manifest["scope"]["action_count"] == 45, "manifest action count")
    require(manifest["scope"]["apm_action_count"] == 45, "manifest APM count")
    require(manifest["scope"]["avqds_action_count"] == 0, "manifest AVQDS count")
    require(manifest["scope"]["precursor_action_count"] == 0, "manifest precursor count")
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
        "pipelines/time_dynamics/ap_mclachlan/compiled_patch_cost.py",
        "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json",
    ):
        require(required in members, f"source archive misses {required}")

    rows = [
        json.loads(line)
        for line in (PACKAGE / "input/actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = (PACKAGE / "input/action_ids.txt").read_text(encoding="utf-8").splitlines()
    require(len(rows) == len(ids) == 45, "action row count")
    require([row["action_id"] for row in rows] == ids, "action order/id drift")
    require(len(set(ids)) == 45, "duplicate action ids")
    require({row["method"] for row in rows} == {"apm"}, "non-APM action")
    require(all(row["comparator"]["action"] is None for row in rows), "comparator action")
    require(
        {row["exchange_route_bundle"] for row in rows}
        == set(manifest["routes"]),
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
    if image is not None:
        require(image.is_file(), f"container absent: {image}")
        require(sha256(image) == manifest["runtime"]["container_sha256"], "container sha drift")
    print(
        json.dumps(
            {
                "status": "OK",
                "action_count": len(rows),
                "source_sha256": sha256(source),
                "container_checked": image is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
