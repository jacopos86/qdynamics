#!/usr/bin/env python3
"""Build immutable batch-3 successors for singleton-fallback trust receipts.

When the batch selector legitimately returns an empty batch, the predecessor
keeps the top record from the authoritative measured Phase-III admission
domain.  It did not, however, replace the rejected joint-batch summary used by
the later adaptive-trust transaction.  This repair carries the selected
record's already measured full-active-plus-singleton response receipt through
that degenerate one-record batch path.  No selector, model, query, or route
setting changes.
"""

from __future__ import annotations

import json
import os
import shutil
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
ADAPT_PATH = "pipelines/static_adapt/adapt_pipeline.py"

FAMILIES = (
    {
        "name": "combinatorial",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v14_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v15_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v14"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v15"
        ),
        "route": "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050",
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v13_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v14_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v13"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v14"
        ),
        "route": "ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865",
    },
)


def patch_adapt(text: str) -> str:
    old_trust = """    if not bool(summary.get("feasible", False)):
        raise RuntimeError(
            "Phase-III batching appendix trust update received an infeasible "
            f"response summary: {summary.get('reason', 'unknown')!r}."
        )
"""
    new_trust = """    fallback_raw = summary.get("authoritative_singleton_fallback")
    if isinstance(fallback_raw, Mapping):
        fallback = dict(fallback_raw)
        committed_labels = [str(value) for value in selected_labels]
        selected_label = str(fallback.get("selected_label", ""))
        if (
            str(fallback.get("schema", ""))
            != "phase3_batch_authoritative_singleton_fallback_v2"
            or str(fallback.get("reason", "")) != "batch_selector_returned_empty"
            or batch_count != 1
            or int(fallback.get("selected_count", -1)) != 1
            or committed_labels != [selected_label]
        ):
            raise RuntimeError(
                "Phase-III batching appendix singleton fallback identity is "
                "missing or inconsistent with the committed admission."
            )
        coordinate_summary_raw = fallback.get("coordinate_summary")
        if not isinstance(coordinate_summary_raw, Mapping):
            raise RuntimeError(
                "Phase-III batching appendix singleton fallback is missing its "
                "measured full-response coordinate receipt."
            )
        selector_summary, context_mode, allow_historical = (
            _historical_singleton_trust_update_inputs(
                {
                    "phase2_joint_geometry_reuse": dict(
                        coordinate_summary_raw
                    )
                },
                whitening_active=True,
                sr_escape_active=False,
                coordinate_solve_policy=(
                    JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
                ),
                radius=float(radius),
                metric_floor=0.0,
                reduced_metric_collapse_rel_tol=0.0,
            )
        )
        selector_summary = dict(selector_summary)
        selector_summary["phase3_batch_authoritative_singleton_fallback"] = {
            "schema": "phase3_batch_authoritative_singleton_fallback_v2",
            "reason": "batch_selector_returned_empty",
            "selected_count": 1,
            "selected_label": selected_label,
            "query_charge": 0,
        }
        return selector_summary, context_mode, allow_historical, False
    if not bool(summary.get("feasible", False)):
        raise RuntimeError(
            "Phase-III batching appendix trust update received an infeasible "
            f"response summary: {summary.get('reason', 'unknown')!r}."
        )
"""
    if text.count(old_trust) != 1:
        raise ValueError("batch trust rejection seam is missing or ambiguous")
    text = text.replace(old_trust, new_trust, 1)

    old_marker = """                                        if historical_nonbeam_coordinate_overlay_active:
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
    new_marker = """                                        if historical_nonbeam_coordinate_overlay_active:
                                            selected_fallback_record = dict(
                                                phase2_selected_records[0]
                                            )
                                            selected_fallback_feature = (
                                                selected_fallback_record.get("feature")
                                            )
                                            if not isinstance(
                                                selected_fallback_feature,
                                                CandidateFeatures,
                                            ):
                                                raise RuntimeError(
                                                    "Authoritative Phase-III batch singleton "
                                                    "fallback requires typed CandidateFeatures."
                                                )
                                            selected_fallback_response = (
                                                selected_fallback_feature.phase2_joint_geometry_reuse
                                            )
                                            if not isinstance(
                                                selected_fallback_response,
                                                Mapping,
                                            ):
                                                raise RuntimeError(
                                                    "Authoritative Phase-III batch singleton "
                                                    "fallback is missing its measured response."
                                                )
                                            batch_summary = {
                                                **dict(batch_summary),
                                                "authoritative_singleton_fallback": {
                                                    "schema": (
                                                        "phase3_batch_authoritative_singleton_"
                                                        "fallback_v2"
                                                    ),
                                                    "reason": "batch_selector_returned_empty",
                                                    "selected_count": 1,
                                                    "selected_label": str(
                                                        _record_candidate_label(
                                                            selected_fallback_record
                                                        )
                                                    ),
                                                    "coordinate_summary": copy.deepcopy(
                                                        dict(selected_fallback_response)
                                                    ),
                                                },
                                            }
"""
    if text.count(old_marker) != 1:
        raise ValueError("authoritative singleton fallback marker seam is missing or ambiguous")
    return text.replace(old_marker, new_marker, 1)


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _additional_archive_test(bundle: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sr-batch3-singleton-trust-") as raw:
        source = Path(raw) / "source"
        source.mkdir()
        with tarfile.open(bundle / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        code = r'''from pipelines.static_adapt import adapt_pipeline as m
summary = {
    "feasible": False,
    "reason": "singleton_shell",
    "authoritative_singleton_fallback": {
        "schema": "phase3_batch_authoritative_singleton_fallback_v2",
        "reason": "batch_selector_returned_empty",
        "selected_count": 1,
        "selected_label": "candidate-A",
        "coordinate_summary": {
            "schema": m.HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
            "feasible": True,
            "joint_linear_solve_policy_effective": m.JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
            "joint_batch_context_mode": m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
        },
    },
}
resolved, context, historical, expansion = m._phase3_batch_appendix_trust_update_inputs(
    summary,
    pre_parameter_count=0,
    selected_count=1,
    selected_labels=["candidate-A"],
    radius=0.2,
)
assert resolved["feasible"] is True
assert resolved["phase3_batch_authoritative_singleton_fallback"]["query_charge"] == 0
assert context == m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
assert historical is False and expansion is False
try:
    m._phase3_batch_appendix_trust_update_inputs(
        summary,
        pre_parameter_count=0,
        selected_count=1,
        selected_labels=["candidate-B"],
        radius=0.2,
    )
except RuntimeError as exc:
    assert "identity" in str(exc)
else:
    raise AssertionError("mismatched singleton fallback identity was accepted")
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
                "frozen-source singleton fallback trust test failed: "
                + completed.stdout
                + completed.stderr
            )
    return {
        "schema": "paper_i_sr_phase3_batch_singleton_trust_test_v1",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "tests_passed": 2,
    }


def _augment(spec: dict[str, str], build_receipt: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    test_receipt = _additional_archive_test(bundle)
    receipt = {
        "schema": "paper_i_sr_phase3_batch_singleton_trust_repair_v1",
        "status": "pass_built_not_submitted_pending_exact_remote_image_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_bundle": spec["parent"],
        "successor_bundle": spec["output"],
        "source_archive_sha256": build_receipt["source_archive_sha256"],
        "route_contract_sha256": spec["route"],
        "scientific_settings_changed": False,
        "query_measurements_added": 0,
        "repair": (
            "carry the selected authoritative singleton's already measured "
            "full-response coordinate receipt into the degenerate one-record "
            "batch adaptive-trust transaction"
        ),
        "failed_predecessor_error": (
            "Phase-III batching appendix trust update received an infeasible "
            "response summary: 'singleton_shell'."
        ),
        "proof": {
            "archive_only_job_gates_passed": 6,
            "frozen_source_tests_passed": 2,
            "route_digest_unchanged": True,
            "exact_remote_image_gate": "pending",
        },
    }
    _dump(bundle / "singleton_fallback_trust_repair.json", receipt)
    _dump(bundle / "source_lock/singleton_fallback_trust_test.json", test_receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["singleton_fallback_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_singleton_trust_successor_artifacts_v1",
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


def _finalize_remote_preflight(spec: dict[str, str]) -> dict[str, Any]:
    base_receipt = base.finalize_remote_preflight(spec)
    bundle = INPUT / spec["output"]
    receipt_path = bundle / "singleton_fallback_trust_repair.json"
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
    _dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["singleton_fallback_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_singleton_trust_successor_artifacts_v1",
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
    return {**base_receipt, "singleton_fallback_trust_repair": receipt}


def main() -> int:
    base.patch_adapt = patch_adapt
    if sys.argv[1:] == ["--finalize-remote-preflight"]:
        receipts = [_finalize_remote_preflight(spec) for spec in FAMILIES]
        print(json.dumps(receipts, indent=2, sort_keys=True))
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
