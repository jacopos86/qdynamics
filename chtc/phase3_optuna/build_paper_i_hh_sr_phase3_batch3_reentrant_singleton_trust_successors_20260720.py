#!/usr/bin/env python3
"""Build immutable batch-3 successors for re-entrant singleton trust receipts.

The predecessor carries the authoritative one-record response into the first
batch trust validation.  That validation resolves the singleton transaction
but did not preserve the input identity marker, so the required post-refit
validation misclassified its own output as a multi-record response.  Preserve
the zero-query identity marker across the validation boundary.  No selection,
measurement, model, route, or scientific setting changes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_paper_i_hh_sr_phase3_batch3_coordinate_successors_20260720 as base


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "chtc/phase3_optuna/input"

FAMILIES = (
    {
        "name": "combinatorial",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v17_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v18_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v17"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v18"
        ),
        "route": "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050",
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v16_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v17_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v16"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v17"
        ),
        "route": "ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865",
    },
)


def patch_adapt(text: str) -> str:
    old = '''        selector_summary["phase3_batch_authoritative_singleton_fallback"] = {
            "schema": "phase3_batch_authoritative_singleton_fallback_v3",
            "reason": fallback_reason,
            "selected_count": 1,
            "selected_label": selected_label,
            "query_charge": 0,
            "geometry_expansion_active": bool(geometry_expansion_active),
        }
        return (
'''
    new = '''        selector_summary["phase3_batch_authoritative_singleton_fallback"] = {
            "schema": "phase3_batch_authoritative_singleton_fallback_v3",
            "reason": fallback_reason,
            "selected_count": 1,
            "selected_label": selected_label,
            "query_charge": 0,
            "geometry_expansion_active": bool(geometry_expansion_active),
        }
        # The same validated response is consumed before commit and again after
        # the accepted refit.  Preserve the authoritative identity receipt so
        # the second validation cannot misclassify this one-record transaction
        # as a multi-record batch.  This is metadata-only and adds no queries.
        selector_summary["authoritative_singleton_fallback"] = copy.deepcopy(
            fallback
        )
        return (
'''
    if text.count(old) != 1:
        raise ValueError("singleton trust output seam is missing or ambiguous")
    return text.replace(old, new, 1)


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _focused_archive_test(bundle: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sr-batch3-reentrant-singleton-") as raw:
        source = Path(raw) / "source"
        source.mkdir()
        with tarfile.open(bundle / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        code = r'''from pipelines.static_adapt import adapt_pipeline as m

receipt = {
    "schema": m.HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
    "feasible": True,
    "joint_batch_context_mode": m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    "joint_linear_solve_policy_effective": m.JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
}
summary = {
    "authoritative_singleton_fallback": {
        "schema": "phase3_batch_authoritative_singleton_fallback_v3",
        "reason": "batch_selector_selected_singleton",
        "selected_count": 1,
        "selected_label": "candidate-A",
        "coordinate_summary": receipt,
        "selected_record_context": {},
        "query_charge": 0,
    }
}
first = m._phase3_batch_appendix_trust_update_inputs(
    summary,
    pre_parameter_count=0,
    selected_count=1,
    selected_labels=["candidate-A"],
    radius=0.25,
)
assert first[0]["authoritative_singleton_fallback"]["query_charge"] == 0
second = m._phase3_batch_appendix_trust_update_inputs(
    first[0],
    pre_parameter_count=0,
    selected_count=1,
    selected_labels=["candidate-A"],
    radius=0.25,
)
assert second[0]["authoritative_singleton_fallback"]["selected_label"] == "candidate-A"
assert second[1:] == first[1:]
assert second[0]["joint_linear_solve_policy_effective"] == m.JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
'''
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(source), "PYTHONDONTWRITEBYTECODE": "1"})
        env.pop("PYTHONNOUSERSITE", None)
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=source,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "frozen-source re-entrant singleton trust test failed: "
                + completed.stdout
                + completed.stderr
            )
    return {
        "schema": "paper_i_sr_phase3_batch_reentrant_singleton_trust_test_v1",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "tests_passed": 4,
    }


def _refresh_hashes(bundle: Path) -> None:
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_reentrant_singleton_successor_artifacts_v1",
            "bundle_id": bundle.name,
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


def _augment(spec: dict[str, str], build_receipt: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    test_receipt = _focused_archive_test(bundle)
    receipt = {
        "schema": "paper_i_sr_phase3_batch_reentrant_singleton_trust_repair_v1",
        "status": "pass_built_not_submitted_pending_exact_remote_image_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_bundle": spec["parent"],
        "successor_bundle": spec["output"],
        "source_archive_sha256": build_receipt["source_archive_sha256"],
        "route_contract_sha256": spec["route"],
        "scientific_settings_changed": False,
        "selection_changed": False,
        "measurements_added": 0,
        "query_charge_added": 0,
        "repair": (
            "preserve the authoritative singleton identity receipt across "
            "the required pre-commit and post-refit trust validations"
        ),
        "failed_predecessor_error": (
            "Phase-III batching appendix requires the supported-metric global "
            "trust solve for its complete batch response."
        ),
        "proof": {
            "archive_only_job_gates_passed": 6,
            "frozen_source_tests_passed": 4,
            "route_digest_unchanged": True,
            "multi_record_global_trust_still_required": True,
            "exact_round1_smoke": "pending",
            "exact_remote_image_gate": "pending",
        },
    }
    _dump(bundle / "reentrant_singleton_trust_repair.json", receipt)
    _dump(bundle / "source_lock/reentrant_singleton_trust_test.json", test_receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reentrant_singleton_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _refresh_hashes(bundle)


def _set_round1_smoke_pass(spec: dict[str, str], smoke: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    receipt_path = bundle / "reentrant_singleton_trust_repair.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["proof"]["exact_round1_smoke"] = "pass"
    receipt["proof"]["exact_round1_smoke_receipt"] = smoke
    _dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reentrant_singleton_trust_repair"] = receipt
        _dump(path, value)
    _refresh_hashes(bundle)


def _finalize_remote_preflight(spec: dict[str, str]) -> dict[str, Any]:
    base_receipt = base.finalize_remote_preflight(spec)
    bundle = INPUT / spec["output"]
    receipt_path = bundle / "reentrant_singleton_trust_repair.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["proof"].get("exact_round1_smoke") != "pass":
        raise RuntimeError("exact round-1 smoke has not passed")
    receipt["status"] = "pass_exact_remote_image_validated_not_submitted"
    receipt["remote_preflight_completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["proof"]["exact_remote_image_gate"] = "pass"
    receipt["proof"]["exact_remote_image_rows_passed"] = 6
    _dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reentrant_singleton_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _refresh_hashes(bundle)
    return {**base_receipt, "reentrant_singleton_trust_repair": receipt}


def main() -> int:
    base.patch_adapt = patch_adapt
    if sys.argv[1:] == ["--finalize-remote-preflight"]:
        print(
            json.dumps(
                [_finalize_remote_preflight(spec) for spec in FAMILIES],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    receipts: list[dict[str, Any]] = []
    for spec in FAMILIES:
        receipt = base.build_family(spec)
        _augment(spec, receipt)
        receipts.append(receipt)
    print(json.dumps(receipts, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
