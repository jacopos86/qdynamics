#!/usr/bin/env python3
"""Fail-closed validation for the 11-action corrected-exchange package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tarfile


PACKAGE = Path(__file__).resolve().parent
EXPECTED_ACTION_COUNT = 1
EXPECTED_PERMISSION_RAY = "5e-3"
CAMPAIGN_ID = "paper_ii_permission_threshold_5em03_v1"
EXPECTED_CODE_REVISION = "1cdce1a0cc38cf8f9a60cca86fe592358ce36d3a"
EXPECTED_SOURCE_SHA256 = (
    "5e79c267b85190c99a377b973f5b9f30e0971b7ca6d53a8a57c9d402024dc702"
)
EXPECTED_ROUTES = {"paper_ii_exchange_q_compiled_v1"}
EXPECTED_SEARCH_PROFILE = "paper_declared_exchange_v1"
COMPARATOR_SOURCE_CAMPAIGN_ID = "paper_ii_final_estimator_telemetry_v1"


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
        manifest.get("schema") == "paper_ii_corrected_exchange_package_v1",
        "manifest schema",
    )
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "manifest campaign id")
    scope = manifest["scope"]
    require(scope["action_count"] == EXPECTED_ACTION_COUNT, "manifest action count")
    require(scope["apm_action_count"] == EXPECTED_ACTION_COUNT, "manifest APM count")
    require(scope["avqds_action_count"] == 0, "manifest AVQDS count")
    require(scope["precursor_action_count"] == 0, "manifest precursor count")
    require(
        scope["comparator_mode"] == "reuse_locked_exact_estimator_avqds",
        "comparator mode drift",
    )
    require(
        scope["comparator_source_campaign_id"] == COMPARATOR_SOURCE_CAMPAIGN_ID,
        "comparator source campaign drift",
    )
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
        "pipelines/time_dynamics/campaigns/paper_ii_corrected_exchange_cells_v1.json",
        "pipelines/time_dynamics/campaigns/paper_ii_permission_threshold_5em03_cells_v1.json",
        "pipelines/time_dynamics/ap_mclachlan/exchange_integration.py",
        "pipelines/time_dynamics/ap_mclachlan/exchange_certification.py",
        "pipelines/time_dynamics/ap_mclachlan/compiled_patch_cost.py",
        "pipelines/time_dynamics/diagnostics/paper_ii_four_route_comparison.py",
        "chtc/paper_ii_production_v2/input/seeds/hh_snake_nph1.json",
    ):
        require(required in members, f"source archive misses {required}")

    lock = json.loads(
        (PACKAGE / "input/matched_cells_lock.json").read_text(encoding="utf-8")
    )
    require(lock["campaign_id"] == CAMPAIGN_ID, "lock campaign drift")
    require(lock["status"] == "PREPARED_NOT_SUBMITTED", "lock status drift")
    require(
        lock["registry_bindings"]["search_profile"] == EXPECTED_SEARCH_PROFILE,
        "lock search-profile drift",
    )
    require(len(lock["cells"]) == EXPECTED_ACTION_COUNT, "lock cell count drift")
    require(set(lock["selector_routes"]) == EXPECTED_ROUTES, "lock route drift")
    grid = {
        (drive, controller)
        for drive in lock["grid"]["drives"]
        for controller in lock["grid"]["controllers"]
    }
    cells = {(row["drive"], row["controller"]) for row in lock["cells"]}
    excluded = {(row["drive"], row["controller"]) for row in lock["exclusions"]}
    require(cells | excluded == grid, "lock does not partition the grid")
    require(not cells & excluded, "lock cell is also excluded")

    rows = [
        json.loads(line)
        for line in (PACKAGE / "input/actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    ids = (PACKAGE / "input/action_ids.txt").read_text(encoding="utf-8").splitlines()
    apm_ids = (PACKAGE / "input/action_ids_apm.txt").read_text(encoding="utf-8").splitlines()
    require(len(rows) == len(ids) == EXPECTED_ACTION_COUNT, "action row count")
    require([row["action_id"] for row in rows] == ids, "action order/id drift")
    require(len(set(ids)) == EXPECTED_ACTION_COUNT, "duplicate action ids")
    require(apm_ids == ids, "queue split drift")
    require(
        all(row["campaign_id"] == CAMPAIGN_ID for row in rows), "campaign id drift"
    )
    require(
        all(row["code_revision"] == EXPECTED_CODE_REVISION for row in rows),
        "code revision drift",
    )
    require(
        all(row["source_package_sha256"] == EXPECTED_SOURCE_SHA256 for row in rows),
        "source digest is not bound to every action",
    )
    require(
        all(row["matched_cell_lock_sha256"] == manifest["inputs"]["matched_cells_lock"]["sha256"] for row in rows),
        "lock digest is not bound to every action",
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
    require(len(apm) == EXPECTED_ACTION_COUNT, "method counts drift")
    require(not [row for row in rows if row["method"] != "apm"], "non-APM action present")
    require(
        {row["exchange_route_bundle"] for row in apm} == EXPECTED_ROUTES,
        "APM route set drift",
    )
    cells_seen: set[str] = set()
    for row in rows:
        cells_seen.add(str(row["cell_id"]))
        action_id = str(row["action_id"])
        require(
            row["trajectory_json"] == f"raw_outputs/{CAMPAIGN_ID}/{action_id}/run.json",
            "trajectory path drift",
        )
        runner = list(row["runner_argv"])
        require(flag_pair(runner, "--t-final") == ("--t-final", "20.0"), "t-final drift")
        require(flag_pair(runner, "--num-times") == ("--num-times", "501"), "grid drift")
        require("--record-statevector" in runner, "statevector receipt disabled")
        require("--resume-from-run-json" not in runner, "resume action present")
        require(
            flag_pair(runner, "--exchange-route-bundle")
            == ("--exchange-route-bundle", "paper_ii_exchange_q_compiled_v1"),
            "route bundle drift",
        )
        require(
            flag_pair(runner, "--exchange-insertion-frontier")
            == ("--exchange-insertion-frontier", "8"),
            "declared exchange frontier absent",
        )
        require(
            flag_pair(runner, "--exchange-permission-ray-distance-max")
            == ("--exchange-permission-ray-distance-max", EXPECTED_PERMISSION_RAY),
            "declared pre-permission threshold absent",
        )
        require(
            flag_pair(runner, "--insertion-gate-mode")
            == ("--insertion-gate-mode", "mclachlan_l2"),
            "insertion gate drift",
        )
        require(
            flag_pair(runner, "--compile-workers") == ("--compile-workers", "8"),
            "compile pool drift",
        )
        require(
            flag_pair(runner, "--support-patch-scoring-workers")
            == ("--support-patch-scoring-workers", "4"),
            "scoring pool drift",
        )
        comparator = row["comparator"]
        require(
            comparator["reuse"] == "reuse_locked_exact_estimator_avqds",
            "comparator is not the reused locked comparator",
        )
        require(
            comparator["source_campaign_id"] == COMPARATOR_SOURCE_CAMPAIGN_ID,
            "comparator source campaign drift",
        )
    require(len(cells_seen) == EXPECTED_ACTION_COUNT, "cell count drift")

    if image is not None:
        require(image.is_file(), f"container absent: {image}")
        require(sha256(image) == manifest["runtime"]["container_sha256"], "container sha drift")
    print(
        json.dumps(
            {
                "status": "OK",
                "action_count": len(rows),
                "apm_action_count": len(apm),
                "avqds_action_count": 0,
                "source_sha256": sha256(source),
                "container_checked": image is not None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
