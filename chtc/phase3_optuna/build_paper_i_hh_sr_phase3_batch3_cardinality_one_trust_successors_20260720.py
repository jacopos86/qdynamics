#!/usr/bin/env python3
"""Build immutable batch-3 successors for cardinality-one trust handoff.

The predecessor correctly selects from the authoritative measured Phase-III
domain, but its adaptive-trust handoff treats every selected batch as a
multi-record batch.  A cardinality-one selection instead has an authoritative
full-active-plus-singleton response receipt already.  Reuse that receipt,
including the existing all-infeasible geometry-expansion marker, without
changing selection, models, measurements, routes, or scientific settings.
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
            "r50_20260720_v15_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v16_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v15"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v16"
        ),
        "route": "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050",
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v14_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v15_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v14"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v15"
        ),
        "route": "ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865",
    },
)


def patch_adapt(text: str) -> str:
    old_trust = '''    fallback_raw = summary.get("authoritative_singleton_fallback")
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
'''
    new_trust = '''    fallback_raw = summary.get("authoritative_singleton_fallback")
    if isinstance(fallback_raw, Mapping):
        fallback = dict(fallback_raw)
        committed_labels = [str(value) for value in selected_labels]
        selected_label = str(fallback.get("selected_label", ""))
        fallback_reason = str(fallback.get("reason", ""))
        if (
            str(fallback.get("schema", ""))
            != "phase3_batch_authoritative_singleton_fallback_v3"
            or fallback_reason not in {
                "batch_selector_returned_empty",
                "batch_selector_selected_singleton",
            }
            or batch_count != 1
            or int(fallback.get("selected_count", -1)) != 1
            or committed_labels != [selected_label]
        ):
            raise RuntimeError(
                "Phase-III batching appendix singleton trust identity is "
                "missing or inconsistent with the committed admission."
            )
        coordinate_summary_raw = fallback.get("coordinate_summary")
        selected_record_context_raw = fallback.get("selected_record_context", {})
        if not isinstance(coordinate_summary_raw, Mapping) or not isinstance(
            selected_record_context_raw, Mapping
        ):
            raise RuntimeError(
                "Phase-III batching appendix singleton trust handoff is missing "
                "its measured response or selected-record context."
            )
        (
            selector_summary,
            context_mode,
            allow_historical,
            geometry_expansion_active,
        ) = _historical_singleton_trust_update_inputs_or_geometry_expansion(
            {
                "phase2_joint_geometry_reuse": dict(coordinate_summary_raw),
            },
            selected_record=dict(selected_record_context_raw),
            whitening_active=True,
            sr_escape_active=False,
            coordinate_solve_policy=(
                JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1
            ),
            radius=float(radius),
            metric_floor=0.0,
            reduced_metric_collapse_rel_tol=0.0,
        )
        selector_summary = dict(selector_summary)
        selector_summary["phase3_batch_authoritative_singleton_fallback"] = {
            "schema": "phase3_batch_authoritative_singleton_fallback_v3",
            "reason": fallback_reason,
            "selected_count": 1,
            "selected_label": selected_label,
            "query_charge": 0,
            "geometry_expansion_active": bool(geometry_expansion_active),
        }
        return (
            selector_summary,
            context_mode,
            allow_historical,
            bool(geometry_expansion_active),
        )
'''
    if text.count(old_trust) != 1:
        raise ValueError("v2 singleton trust handoff seam is missing or ambiguous")
    text = text.replace(old_trust, new_trust, 1)

    old_before_penalty = '''                                    phase2_last_batch_penalty_total = float(
                                        batch_summary.get("additivity_defect", 0.0)
                                    )
'''
    new_before_penalty = '''                                    if (
                                        historical_nonbeam_coordinate_overlay_active
                                        and len(phase2_selected_records) == 1
                                    ):
                                        selected_singleton_record = dict(
                                            phase2_selected_records[0]
                                        )
                                        selected_singleton_feature = (
                                            selected_singleton_record.get("feature")
                                        )
                                        if not isinstance(
                                            selected_singleton_feature,
                                            CandidateFeatures,
                                        ):
                                            raise RuntimeError(
                                                "Authoritative Phase-III batch singleton "
                                                "requires typed CandidateFeatures."
                                            )
                                        selected_singleton_response = (
                                            selected_singleton_feature.phase2_joint_geometry_reuse
                                        )
                                        if not isinstance(
                                            selected_singleton_response,
                                            Mapping,
                                        ):
                                            raise RuntimeError(
                                                "Authoritative Phase-III batch singleton is "
                                                "missing its measured response."
                                            )
                                        batch_summary = {
                                            **dict(batch_summary),
                                            "authoritative_singleton_fallback": {
                                                "schema": (
                                                    "phase3_batch_authoritative_singleton_"
                                                    "fallback_v3"
                                                ),
                                                "reason": (
                                                    "batch_selector_selected_singleton"
                                                ),
                                                "selected_count": 1,
                                                "selected_label": str(
                                                    _record_candidate_label(
                                                        selected_singleton_record
                                                    )
                                                ),
                                                "coordinate_summary": copy.deepcopy(
                                                    dict(selected_singleton_response)
                                                ),
                                                "selected_record_context": {
                                                    str(key): copy.deepcopy(value)
                                                    for key, value in (
                                                        selected_singleton_record.items()
                                                    )
                                                    if str(key).startswith(
                                                        ("route_a_", "sr_escape_")
                                                    )
                                                },
                                            },
                                        }
                                    phase2_last_batch_penalty_total = float(
                                        batch_summary.get("additivity_defect", 0.0)
                                    )
'''
    if text.count(old_before_penalty) != 1:
        raise ValueError("cardinality-one batch marker seam is missing or ambiguous")
    text = text.replace(old_before_penalty, new_before_penalty, 1)

    old_empty_marker = '''                                                    "schema": (
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
'''
    new_empty_marker = '''                                                    "schema": (
                                                        "phase3_batch_authoritative_singleton_"
                                                        "fallback_v3"
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
                                                    "selected_record_context": {
                                                        str(key): copy.deepcopy(value)
                                                        for key, value in (
                                                            selected_fallback_record.items()
                                                        )
                                                        if str(key).startswith(
                                                            ("route_a_", "sr_escape_")
                                                        )
                                                    },
'''
    if text.count(old_empty_marker) != 1:
        raise ValueError("empty singleton v2 marker seam is missing or ambiguous")
    return text.replace(old_empty_marker, new_empty_marker, 1)


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _additional_archive_test(bundle: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sr-batch3-cardinality-one-") as raw:
        source = Path(raw) / "source"
        source.mkdir()
        with tarfile.open(bundle / "source_locked.tar.gz", "r:gz") as archive:
            archive.extractall(source, filter="data")
        code = r'''from pipelines.static_adapt import adapt_pipeline as m

def receipt(*, feasible=True, reason="ok"):
    return {
        "schema": m.HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA,
        "feasible": feasible,
        "reason": reason,
        "joint_linear_solve_policy_effective": m.JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
        "joint_batch_context_mode": m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
    }

def summary(*, reason, coordinate, context=None):
    return {
        "feasible": False,
        "reason": "degenerate_batch_shell",
        "authoritative_singleton_fallback": {
            "schema": "phase3_batch_authoritative_singleton_fallback_v3",
            "reason": reason,
            "selected_count": 1,
            "selected_label": "candidate-A",
            "coordinate_summary": coordinate,
            "selected_record_context": dict(context or {}),
        },
    }

for reason in ("batch_selector_returned_empty", "batch_selector_selected_singleton"):
    resolved, context, historical, expansion = m._phase3_batch_appendix_trust_update_inputs(
        summary(reason=reason, coordinate=receipt()),
        pre_parameter_count=3,
        selected_count=1,
        selected_labels=["candidate-A"],
        radius=0.2,
    )
    assert resolved["feasible"] is True
    assert resolved["phase3_batch_authoritative_singleton_fallback"]["query_charge"] == 0
    assert context == m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
    assert historical is False and expansion is False

resolved, context, historical, expansion = m._phase3_batch_appendix_trust_update_inputs(
    summary(
        reason="batch_selector_returned_empty",
        coordinate=receipt(feasible=False, reason="rank_gate"),
        context={"route_a_geometry_expansion_mode": "collective_span_novelty_over_cost_v1"},
    ),
    pre_parameter_count=3,
    selected_count=1,
    selected_labels=["candidate-A"],
    radius=0.2,
)
assert expansion is True and historical is False
assert context == m.HISTORICAL_SINGLETON_GEOMETRY_EXPANSION_CONTEXT_V1
assert resolved["phase3_batch_authoritative_singleton_fallback"]["query_charge"] == 0

try:
    m._phase3_batch_appendix_trust_update_inputs(
        summary(reason="batch_selector_selected_singleton", coordinate=receipt()),
        pre_parameter_count=3,
        selected_count=1,
        selected_labels=["candidate-B"],
        radius=0.2,
    )
except RuntimeError as exc:
    assert "identity" in str(exc)
else:
    raise AssertionError("mismatched singleton identity was accepted")

try:
    m._phase3_batch_appendix_trust_update_inputs(
        {"feasible": True, "joint_batch_context_mode": m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1,
         "joint_linear_solve_policy_effective": m.JOINT_LINEAR_SOLVE_SUPPORTED_METRIC_WHITENED_EIGH_V1,
         "selected_count": 2, "selected_labels": ["A", "B"]},
        pre_parameter_count=3,
        selected_count=2,
        selected_labels=["A", "B"],
        radius=0.2,
    )
except RuntimeError as exc:
    assert "global trust" in str(exc)
else:
    raise AssertionError("multi-record batch escaped global-trust enforcement")
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
                "frozen-source cardinality-one trust test failed: "
                + completed.stdout
                + completed.stderr
            )
    return {
        "schema": "paper_i_sr_phase3_batch_cardinality_one_trust_test_v1",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "tests_passed": 5,
    }


def _augment(spec: dict[str, str], build_receipt: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    test_receipt = _additional_archive_test(bundle)
    receipt = {
        "schema": "paper_i_sr_phase3_batch_cardinality_one_trust_repair_v1",
        "status": "pass_built_not_submitted_pending_exact_remote_image_gate",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "predecessor_bundle": spec["parent"],
        "successor_bundle": spec["output"],
        "source_archive_sha256": build_receipt["source_archive_sha256"],
        "route_contract_sha256": spec["route"],
        "scientific_settings_changed": False,
        "candidate_selection_changed": False,
        "query_measurements_added": 0,
        "repair": (
            "route an authoritative cardinality-one batch through its already "
            "measured singleton response; retain the existing geometry-expansion "
            "flag for all-infeasible novelty fallback"
        ),
        "failed_predecessor_errors": [
            "Phase-III batching appendix requires the supported-metric global trust solve for its complete batch response.",
            "Whitened historical singleton trust update received an infeasible energy-model summary outside geometry-expansion mode: 'rank_gate'.",
        ],
        "proof": {
            "archive_only_job_gates_passed": 6,
            "frozen_source_tests_passed": 5,
            "route_digest_unchanged": True,
            "multi_record_global_trust_still_required": True,
            "exact_remote_image_gate": "pending",
        },
    }
    _dump(bundle / "cardinality_one_trust_repair.json", receipt)
    _dump(bundle / "source_lock/cardinality_one_trust_test.json", test_receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cardinality_one_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_cardinality_one_successor_artifacts_v1",
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
    receipt_path = bundle / "cardinality_one_trust_repair.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "pass_exact_remote_image_validated_not_submitted"
    receipt["remote_preflight_completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["proof"]["exact_remote_image_gate"] = "pass"
    receipt["proof"]["exact_remote_image_rows_passed"] = 6
    _dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["cardinality_one_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_cardinality_one_successor_artifacts_v1",
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
    return {**base_receipt, "cardinality_one_trust_repair": receipt}


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
