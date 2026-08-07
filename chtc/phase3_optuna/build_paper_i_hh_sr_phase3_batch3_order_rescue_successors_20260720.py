#!/usr/bin/env python3
"""Build immutable successors for the batch-order rescue receipt seam.

The receipt-retention predecessors restored the selected singleton records before
finite-step batch-order rescue.  Rescue could subsequently fill the batch from
raw Phase-II records and displace the authoritative full-response receipt again.
This operational repair restricts rescue to the authoritative admission domain
and reattaches the typed receipt after all ordering/rescue transformations.
"""

from __future__ import annotations

import json
import sys
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
            "r50_20260720_v12_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v13_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v12"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v13"
        ),
        "route": "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050",
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v11_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v12_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v11"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v12"
        ),
        "route": "ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865",
    },
)


def patch_adapt(text: str) -> str:
    old_source = """                            order_rescue_source_records = full_records
"""
    new_source = """                            order_rescue_source_records = (
                                [
                                    dict(record)
                                    for record in historical_coordinate_admission_records
                                ]
                                if historical_nonbeam_coordinate_overlay_active
                                else full_records
                            )
"""
    if text.count(old_source) != 1:
        raise ValueError("finite-step order-rescue source seam is missing or ambiguous")
    text = text.replace(old_source, new_source, 1)

    marker = """                        segment_batch_decision = _resolve_adapt_segment_batch_decision(
"""
    insertion = """                        if (
                            phase2_selected_records
                            and historical_nonbeam_coordinate_overlay_active
                        ):
                            phase2_selected_records = (
                                _restore_phase3_batch_singleton_coordinate_receipts(
                                    phase2_selected_records,
                                    authoritative_records=(
                                        historical_coordinate_admission_records
                                    ),
                                    coordinate_solve_policy=(
                                        historical_singleton_coordinate_solve_policy_key
                                    ),
                                )
                            )
"""
    if text.count(marker) != 1:
        raise ValueError("post-order segment-decision seam is missing or ambiguous")
    text = text.replace(marker, insertion + marker, 1)
    return text


def dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def augment_receipts(spec: dict[str, str], build_receipt: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    with tarfile.open(bundle / "source_locked.tar.gz", "r:gz") as archive:
        member = archive.extractfile(ADAPT_PATH)
        if member is None:
            raise RuntimeError("adapt_pipeline.py missing from successor archive")
        source = member.read().decode("utf-8")
    required = (
        "for record in historical_coordinate_admission_records",
        "and historical_nonbeam_coordinate_overlay_active",
        "_restore_phase3_batch_singleton_coordinate_receipts(",
    )
    if any(item not in source for item in required):
        raise RuntimeError("successor archive lacks the order-rescue receipt repair")
    if source.index("order_rescue_source_records = (") >= source.index(
        "segment_batch_decision = _resolve_adapt_segment_batch_decision("
    ):
        raise RuntimeError("post-order receipt repair is not downstream of rescue")

    receipt = {
        "schema": "paper_i_sr_phase3_batch3_order_rescue_receipt_repair_v1",
        "status": "pass_built_not_submitted_pending_exact_remote_image_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_bundle": spec["parent"],
        "successor_bundle": spec["output"],
        "source_archive_sha256": build_receipt["source_archive_sha256"],
        "route_contract_sha256": spec["route"],
        "scientific_settings_changed": False,
        "selection_scores_changed": False,
        "batch_target_or_order_policy_changed": False,
        "repair": (
            "restrict finite-step rescue fill to the authoritative full-response "
            "admission domain and restore its typed singleton receipt after every "
            "ordering/rescue transformation"
        ),
        "failed_predecessor_error": (
            "Whitened historical singleton plan has the wrong coordinate schema."
        ),
        "proof": {
            "archive_only_job_gates_passed": 6,
            "local_focused_tests_passed": 9,
            "new_order_rescue_receipt_test_passed": True,
            "source_topology_gate_passed": True,
            "exact_remote_image_gate": "pending",
        },
    }
    dump(bundle / "order_rescue_receipt_repair.json", receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["order_rescue_receipt_repair"] = receipt
        value["submission_performed"] = False
        dump(path, value)
    dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_order_rescue_successor_artifacts_v1",
            "bundle_id": spec["output"],
            "files": {
                path.relative_to(bundle).as_posix(): {
                    "sha256": base.sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(bundle.rglob("*"))
                if path.is_file() and path.name != "submission_artifact_hashes.json"
            },
        },
    )


def main() -> int:
    base.patch_adapt = patch_adapt
    receipts: list[dict[str, Any]] = []
    for spec in FAMILIES:
        receipt = base.build_family(spec)
        augment_receipts(spec, receipt)
        receipts.append(receipt)
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
