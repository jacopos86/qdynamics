#!/usr/bin/env python3
"""Fail-closed validation for the 60-action exact-estimator package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile


PACKAGE = Path(__file__).resolve().parent
EXPECTED_CODE_REVISION = "9e31f4d7f3859110aa016d8eaefd4c50ac62fde8"
EXPECTED_SOURCE_SHA256 = (
    "f1e2d49de5b929de9715fbbf23ad3a64a4041a4b7fb29fe4f16e8f82bf7b995c"
)
EXPECTED_ROUTES = {
    "paper_ii_exchange_q_compiled_v1",
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


def flag_pair(argv: list[str], flag: str) -> tuple[str, str] | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    require(index + 1 < len(argv), f"{flag} has no value")
    return flag, argv[index + 1]


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    image = Path(args[0]) if args else None
    manifest = json.loads((PACKAGE / "input/manifest.json").read_text(encoding="utf-8"))
    require(
        manifest.get("schema") == "paper_ii_final_estimator_telemetry_package_v1",
        "manifest schema",
    )
    scope = manifest["scope"]
    require(scope["action_count"] == 2, "manifest action count")
    require(scope["apm_action_count"] == 1, "manifest APM count")
    require(scope["avqds_action_count"] == 1, "manifest AVQDS count")
    require(scope["precursor_action_count"] == 0, "manifest precursor count")
    calibration = manifest["inputs"]["calibration_receipt"]
    require(calibration["error_count"] == 0, "calibration ERROR")
    require(calibration["status"] == "INCOMPLETE", "calibration status drift")
    require(
        calibration["not_checked"]
        == ["benchmark_point_sameness", "response_normalized_matched_accuracy"],
        "calibration missing-check disclosure drift",
    )
    require(not calibration["allow_unverified_response_scales_used"], "degraded scales")
    resolution = manifest["inputs"]["report_refresh_resolution"]
    require(resolution["active_blocks"] == [], "active submission block")
    require(resolution["authorizes_new_campaign_submission"] is True, "submission not authorized")

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
    require(sha256(source) == EXPECTED_SOURCE_SHA256, "source archive sha drift")
    with tarfile.open(source, "r:gz") as handle:
        members = set(handle.getnames())
    for required in (
        "pipelines/time_dynamics/campaigns/paper_ii_final_estimator_telemetry_v1.py",
        "pipelines/time_dynamics/ap_mclachlan/avqds.py",
        "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
        "pipelines/time_dynamics/ap_mclachlan/exchange_certification.py",
        "pipelines/time_dynamics/ap_mclachlan/compiled_patch_cost.py",
        "pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py",
        "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json",
    ):
        require(required in members, f"source archive misses {required}")

    rows = [
        json.loads(line)
        for line in (PACKAGE / "input/actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = (PACKAGE / "input/action_ids.txt").read_text(encoding="utf-8").splitlines()
    apm_ids = (PACKAGE / "input/action_ids_apm.txt").read_text(encoding="utf-8").splitlines()
    avqds_ids = (PACKAGE / "input/action_ids_avqds.txt").read_text(encoding="utf-8").splitlines()
    require(len(rows) == len(ids) == 2, "action row count")
    require([row["action_id"] for row in rows] == ids, "action order/id drift")
    require(len(set(ids)) == 2, "duplicate action ids")
    require(len(apm_ids) == 1 and len(avqds_ids) == 1, "queue split drift")
    require(set(apm_ids).isdisjoint(avqds_ids), "queue split overlaps")
    require(set(apm_ids) | set(avqds_ids) == set(ids), "queue split incomplete")
    require(
        all(row["code_revision"] == EXPECTED_CODE_REVISION for row in rows),
        "code revision drift",
    )
    require(
        all(row["source_package_sha256"] == EXPECTED_SOURCE_SHA256 for row in rows),
        "source digest is not bound to every action",
    )
    require(all(row["precursor"] is False for row in rows), "precursor row")
    require(
        all(
            row["estimator_telemetry_contract"]
            == "paper_ii_exact_logical_estimator_telemetry_20260903"
            for row in rows
        ),
        "telemetry contract drift",
    )

    apm = [row for row in rows if row["method"] == "apm"]
    avqds = [row for row in rows if row["method"] == "avqds"]
    require(len(apm) == 1 and len(avqds) == 1, "method counts drift")
    require(
        {row["exchange_route_bundle"] for row in apm} == EXPECTED_ROUTES,
        "APM route set drift",
    )
    by_cell: dict[str, list[dict]] = {}
    for row in rows:
        by_cell.setdefault(str(row["cell_id"]), []).append(row)
        action_id = str(row["action_id"])
        require(
            row["trajectory_json"]
            == f"raw_outputs/paper_ii_corrected_exchange_strongfast_state_motion_v1/{action_id}/run.json",
            "trajectory path drift",
        )
        runner = list(row["runner_argv"])
        require(flag_pair(runner, "--t-final") == ("--t-final", "20.0"), "t-final drift")
        require(flag_pair(runner, "--num-times") == ("--num-times", "501"), "grid drift")
        require("--record-statevector" in runner, "statevector receipt disabled")
        require("--resume-from-run-json" not in runner, "resume action present")
    require(len(by_cell) == 1, "cell count drift")
    require(all(len(cell_rows) == 4 for cell_rows in by_cell.values()), "cell width drift")

    for row in apm:
        runner = list(row["runner_argv"])
        require(flag_pair(runner, "--compile-workers") == ("--compile-workers", "8"), "compile pool drift")
        require(
            flag_pair(runner, "--support-patch-scoring-workers")
            == ("--support-patch-scoring-workers", "4"),
            "scoring pool drift",
        )
        require(row["comparator"]["action"] in avqds_ids, "fresh comparator not linked")
    for row in avqds:
        runner = list(row["runner_argv"])
        require(row["arm"] == "avqds", "AVQDS arm drift")
        require(flag_pair(runner, "--dynamics-policy") == ("--dynamics-policy", "avqds"), "AVQDS policy drift")
        require(flag_pair(runner, "--avqds-l2-cut") is not None, "AVQDS cut absent")
        require("--exchange-route-bundle" not in runner, "exchange route leaked into AVQDS")
        require("--insertion-gate-mode" not in runner, "APM gate leaked into AVQDS")
        require("--compile-workers" not in runner, "compile pool leaked into AVQDS")

    if image is not None:
        require(image.is_file(), f"container absent: {image}")
        require(sha256(image) == manifest["runtime"]["container_sha256"], "container sha drift")
    print(
        json.dumps(
            {
                "status": "OK",
                "action_count": len(rows),
                "apm_action_count": len(apm),
                "avqds_action_count": len(avqds),
                "source_sha256": sha256(source),
                "container_checked": image is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
