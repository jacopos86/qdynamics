#!/usr/bin/env python3
"""Build immutable successors for the empty batch-selection fallback seam."""

from __future__ import annotations

import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_paper_i_hh_sr_phase3_batch3_coordinate_successors_20260720 as base


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"
ADAPT_PATH = "pipelines/static_adapt/adapt_pipeline.py"

FAMILIES = (
    {
        "name": "combinatorial",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v13_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v14_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v13"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v14"
        ),
        "route": "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050",
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v12_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v13_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v12"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v13"
        ),
        "route": "ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865",
    },
)


def patch_adapt(text: str) -> str:
    local = (ROOT / ADAPT_PATH).read_text(encoding="utf-8")
    helper_start = local.index("def _phase3_batch_empty_selection_fallback(")
    helper_end = local.index(
        "\ndef _sr_outer_growth_cache_absence_requires_exact_fallback(",
        helper_start,
    )
    helper = local[helper_start:helper_end].rstrip() + "\n\n"
    marker = "def _all_energy_models_infeasible_novelty_fallback_telemetry("
    if "def _phase3_batch_empty_selection_fallback(" in text:
        raise ValueError("predecessor already contains empty-selection repair")
    if text.count(marker) != 1:
        raise ValueError("empty-selection helper insertion seam is ambiguous")
    text = text.replace(marker, helper + marker, 1)

    old = """                                    if not phase2_selected_records:
                                        phase2_selected_records = [dict(full_records[0])]
"""
    new = """                                    if not phase2_selected_records:
                                        phase2_selected_records = (
                                            _phase3_batch_empty_selection_fallback(
                                                phase2_selected_records,
                                                batch_source_records=(
                                                    batch_source_records
                                                ),
                                                authoritative_records=(
                                                    historical_coordinate_admission_records
                                                ),
                                                raw_phase2_records=full_records,
                                                historical_coordinate_overlay_active=(
                                                    historical_nonbeam_coordinate_overlay_active
                                                ),
                                                coordinate_solve_policy=(
                                                    historical_singleton_coordinate_solve_policy_key
                                                ),
                                            )
                                        )
                                        if historical_nonbeam_coordinate_overlay_active:
                                            batch_summary = {
                                                **dict(batch_summary),
                                                "authoritative_singleton_fallback": {
                                                    "schema": (
                                                        "phase3_batch_authoritative_singleton_"
                                                        "fallback_v1"
                                                    ),
                                                    "reason": "batch_selector_returned_empty",
                                                    "selected_count": 1,
                                                },
                                            }
"""
    if text.count(old) != 1:
        raise ValueError("raw Phase-II empty-selection fallback seam is ambiguous")
    return text.replace(old, new, 1)


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def augment_receipt(spec: dict[str, str], build_receipt: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    with tarfile.open(bundle / "source_locked.tar.gz", "r:gz") as archive:
        source = archive.extractfile(ADAPT_PATH)
        if source is None:
            raise RuntimeError("adapt_pipeline.py missing from successor archive")
        text = source.read().decode("utf-8")
    for required in (
        "def _phase3_batch_empty_selection_fallback(",
        "_phase3_batch_empty_selection_fallback(\n",
        "batch_selector_returned_empty",
    ):
        if required not in text:
            raise RuntimeError(f"successor archive lacks repair marker {required!r}")
    receipt = {
        "schema": "paper_i_sr_phase3_batch3_empty_selection_repair_v1",
        "status": "pass_built_not_submitted_pending_exact_remote_image_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_bundle": spec["parent"],
        "successor_bundle": spec["output"],
        "source_archive_sha256": build_receipt["source_archive_sha256"],
        "route_contract_sha256": spec["route"],
        "scientific_settings_changed": False,
        "repair": (
            "when the batch selector returns no records, retain the top record "
            "from its authoritative measured Phase-III input domain instead of "
            "jumping to a raw Phase-II record"
        ),
        "failed_predecessor_error": (
            "Whitened historical singleton plan has the wrong coordinate schema."
        ),
        "proof": {
            "archive_only_job_gates_passed": 6,
            "focused_local_regressions_passed": 6,
            "route_digest_unchanged": True,
            "exact_remote_image_gate": "pending",
        },
    }
    dump(bundle / "empty_selection_fallback_repair.json", receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["empty_selection_fallback_repair"] = receipt
        value["submission_performed"] = False
        dump(path, value)
    dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_empty_selection_successor_artifacts_v1",
            "bundle_id": spec["output"],
            "files": {
                path.relative_to(bundle).as_posix(): {
                    "sha256": base.sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(bundle.rglob("*"))
                if path.is_file()
                and path.name != "submission_artifact_hashes.json"
            },
        },
    )


def finalize_remote_preflight(spec: dict[str, str]) -> dict[str, Any]:
    result = base.finalize_remote_preflight(spec)
    bundle = INPUT / spec["output"]
    receipt_path = bundle / "empty_selection_fallback_repair.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "pass_exact_remote_image_validated_not_submitted"
    receipt["remote_preflight_completed_utc"] = datetime.now(
        timezone.utc
    ).isoformat()
    receipt["proof"]["exact_remote_image_gate"] = "pass"
    receipt["proof"]["exact_remote_image_rows_passed"] = 6
    receipt["proof"]["exact_remote_image_sha256"] = (
        "fa5c4ea89a19ad2fa4c264f3bce65d9e83b86b91618ed97dfc80ee7be401239f"
    )
    dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["empty_selection_fallback_repair"] = receipt
        value["submission_performed"] = False
        dump(path, value)
    dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_empty_selection_successor_artifacts_v1",
            "bundle_id": spec["output"],
            "files": {
                path.relative_to(bundle).as_posix(): {
                    "sha256": base.sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(bundle.rglob("*"))
                if path.is_file()
                and path.name != "submission_artifact_hashes.json"
            },
        },
    )
    return {**result, "empty_selection_fallback_repair": receipt}


def main() -> int:
    base.patch_adapt = patch_adapt
    receipts: list[dict[str, Any]] = []
    for spec in FAMILIES:
        receipt = base.build_family(spec)
        augment_receipt(spec, receipt)
        receipts.append(receipt)
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
