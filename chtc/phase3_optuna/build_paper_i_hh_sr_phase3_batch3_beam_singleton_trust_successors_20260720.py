#!/usr/bin/env python3
"""Build immutable batch-3 successors for beam-plan singleton trust receipts.

The predecessor correctly retained singleton response receipts in the
non-beam admission path.  The effective 1x1 beam planner rebuilt the same
authoritative one-record proposal without attaching that receipt to the plan,
so adaptive trust incorrectly demanded a multi-record global solve.  This
repair propagates the already measured receipt through both beam proposal and
beam singleton-fallback plans.  It changes no selection, measurement, route,
or scientific setting.
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
            "r50_20260720_v16_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v17_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v16"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v17"
        ),
        "route": "27df701ab280c02422e7030ec60a77d37ff20b73132ae4824cc41017f93fa050",
    },
    {
        "name": "greedy",
        "parent": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v15_chtc"
        ),
        "output": (
            "paper_i_hh_sr_snake_appendix_phase3_greedy_batch3_full_response_"
            "symmetric_cost_noprune_nobeam_no_ordinary_novelty_all_six_"
            "r50_20260720_v16_chtc"
        ),
        "parent_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v15"
        ),
        "output_batch": (
            "paper-i-hh-sr-appendix-phase3-greedy-batch3-fullresp-symcost-"
            "noprune-nobeam-nonovelty-six-r50-20260720-v16"
        ),
        "route": "ed3afaeffa076cfce6b3eb63bf912dd182768fd109b3cc2be5878f23b335c865",
    },
)


def patch_adapt(text: str) -> str:
    helper_marker = "def _phase3_batch_appendix_trust_update_inputs("
    helper = '''def _phase3_batch_beam_singleton_trust_summary(
    batch_summary: Mapping[str, Any],
    *,
    selected_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach an already measured singleton response to a 1x1 beam plan."""

    selected = dict(selected_record)
    feature_raw = selected.get("feature")
    if isinstance(feature_raw, CandidateFeatures):
        coordinate_summary_raw = feature_raw.phase2_joint_geometry_reuse
    elif isinstance(feature_raw, Mapping):
        coordinate_summary_raw = feature_raw.get("phase2_joint_geometry_reuse")
    else:
        raise RuntimeError(
            "Phase-III batch beam singleton lost its typed feature receipt."
        )
    if not (
        isinstance(coordinate_summary_raw, Mapping)
        and str(coordinate_summary_raw.get("schema", ""))
        == HISTORICAL_SINGLETON_COORDINATE_MODEL_SCHEMA
        and bool(coordinate_summary_raw.get("feasible", False))
        and str(coordinate_summary_raw.get("joint_batch_context_mode", ""))
        == BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
    ):
        raise RuntimeError(
            "Phase-III batch beam singleton lacks its feasible measured "
            "full-active-plus-singleton response."
        )
    return {
        **dict(batch_summary),
        "authoritative_singleton_fallback": {
            "schema": "phase3_batch_authoritative_singleton_fallback_v3",
            "reason": "batch_selector_selected_singleton",
            "selected_count": 1,
            "selected_label": str(_record_candidate_label(selected)),
            "coordinate_summary": copy.deepcopy(dict(coordinate_summary_raw)),
            "selected_record_context": {
                str(key): copy.deepcopy(value)
                for key, value in selected.items()
                if str(key).startswith(("route_a_", "sr_escape_"))
            },
            "query_charge": 0,
            "source": "beam_plan_authoritative_singleton_receipt",
        },
    }


'''
    if text.count(helper_marker) != 1:
        raise ValueError("batch trust helper insertion seam is missing or ambiguous")
    if "def _phase3_batch_beam_singleton_trust_summary(" in text:
        raise ValueError("predecessor already contains beam singleton trust repair")
    text = text.replace(helper_marker, helper + helper_marker, 1)

    old_proposal = '''                        candidate_plans.append(
                            _BranchExpansionPlan(
                                candidate_pool_index=int(first_rec.get("candidate_pool_index", best_idx_local)),
                                position_id=int(first_rec.get("position_id", append_position_local)),
                                selection_mode=str(selection_mode_local),
                                candidate_label=str(candidate_term.label),
                                candidate_term=candidate_term,
                                feature_row=feat_row,
                                init_theta=0.0,
                                batch_records=tuple(dict(rec) for rec in ordered_records),
                                batch_summary={
                                    **dict(batch_summary_local),
                                    **dict(proposal.summary),
                                    "batch_order_summary": dict(order_summary),
                                    "route_a_round_trust_region": (
                                        None
                                        if route_a_round_trust_receipt_local is None
                                        else dict(
                                            route_a_round_trust_receipt_local
                                        )
                                    ),
                                },
'''
    new_proposal = '''                        proposal_batch_summary_local = {
                            **dict(batch_summary_local),
                            **dict(proposal.summary),
                            "batch_order_summary": dict(order_summary),
                            "route_a_round_trust_region": (
                                None
                                if route_a_round_trust_receipt_local is None
                                else dict(route_a_round_trust_receipt_local)
                            ),
                        }
                        if (
                            phase3_only_batch_appendix_active
                            and historical_nonbeam_coordinate_overlay_active
                            and len(ordered_records) == 1
                        ):
                            proposal_batch_summary_local = (
                                _phase3_batch_beam_singleton_trust_summary(
                                    proposal_batch_summary_local,
                                    selected_record=ordered_records[0],
                                )
                            )
                        candidate_plans.append(
                            _BranchExpansionPlan(
                                candidate_pool_index=int(first_rec.get("candidate_pool_index", best_idx_local)),
                                position_id=int(first_rec.get("position_id", append_position_local)),
                                selection_mode=str(selection_mode_local),
                                candidate_label=str(candidate_term.label),
                                candidate_term=candidate_term,
                                feature_row=feat_row,
                                init_theta=0.0,
                                batch_records=tuple(dict(rec) for rec in ordered_records),
                                batch_summary=proposal_batch_summary_local,
'''
    if text.count(old_proposal) != 1:
        raise ValueError("beam proposal plan seam is missing or ambiguous")
    text = text.replace(old_proposal, new_proposal, 1)

    old_fallback = '''                    for rec in beam_selected_records_local:
                        feat_row = _beam_feature_row(rec)
                        saddle_plan_local = bool(
'''
    new_fallback = '''                    for rec in beam_selected_records_local:
                        feat_row = _beam_feature_row(rec)
                        singleton_plan_summary_local = None
                        singleton_plan_records_local: tuple[dict[str, Any], ...] = ()
                        if (
                            phase3_only_batch_appendix_active
                            and historical_nonbeam_coordinate_overlay_active
                        ):
                            singleton_plan_records_local = (dict(rec),)
                            singleton_plan_summary_local = (
                                _phase3_batch_beam_singleton_trust_summary(
                                    {},
                                    selected_record=rec,
                                )
                            )
                        saddle_plan_local = bool(
'''
    if text.count(old_fallback) != 1:
        raise ValueError("beam singleton fallback seam is missing or ambiguous")
    text = text.replace(old_fallback, new_fallback, 1)

    old_fallback_plan = '''                                batch_records=(
                                    (dict(rec),)
                                    if sr_escape_active
                                    else ()
                                ),
                                batch_summary=(
                                    coordinate_summary_for_plan_local
                                    if saddle_plan_local
                                    else None
                                ),
'''
    new_fallback_plan = '''                                batch_records=(
                                    singleton_plan_records_local
                                    if singleton_plan_records_local
                                    else (
                                        (dict(rec),)
                                        if sr_escape_active
                                        else ()
                                    )
                                ),
                                batch_summary=(
                                    singleton_plan_summary_local
                                    if singleton_plan_summary_local is not None
                                    else (
                                        coordinate_summary_for_plan_local
                                        if saddle_plan_local
                                        else None
                                    )
                                ),
'''
    if text.count(old_fallback_plan) != 1:
        raise ValueError("beam singleton plan payload seam is missing or ambiguous")
    return text.replace(old_fallback_plan, new_fallback_plan, 1)


def _dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _focused_archive_test(bundle: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="sr-batch3-beam-singleton-") as raw:
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
record = {
    "candidate_pool_index": 7,
    "position_id": 2,
    "candidate_label": "candidate-A",
    "feature": {"phase2_joint_geometry_reuse": receipt},
}
summary = m._phase3_batch_beam_singleton_trust_summary({}, selected_record=record)
fallback = summary["authoritative_singleton_fallback"]
assert fallback["query_charge"] == 0
assert fallback["source"] == "beam_plan_authoritative_singleton_receipt"
resolved, context, historical, expansion = m._phase3_batch_appendix_trust_update_inputs(
    summary,
    pre_parameter_count=3,
    selected_count=1,
    selected_labels=["candidate-A"],
    radius=0.2,
)
assert resolved["feasible"] is True
assert context == m.BATCH_JOINT_CONTEXT_FULL_ANSATZ_V1
assert historical is False and expansion is False
try:
    m._phase3_batch_beam_singleton_trust_summary(
        {},
        selected_record={**record, "feature": {"phase2_joint_geometry_reuse": {**receipt, "feasible": False}}},
    )
except RuntimeError as exc:
    assert "feasible measured" in str(exc)
else:
    raise AssertionError("infeasible singleton response was accepted")
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
                "frozen-source beam singleton trust test failed: "
                + completed.stdout
                + completed.stderr
            )
    return {
        "schema": "paper_i_sr_phase3_batch_beam_singleton_trust_test_v1",
        "status": "pass",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "tests_passed": 3,
    }


def _augment(spec: dict[str, str], build_receipt: dict[str, Any]) -> None:
    bundle = INPUT / spec["output"]
    test_receipt = _focused_archive_test(bundle)
    receipt = {
        "schema": "paper_i_sr_phase3_batch_beam_singleton_trust_repair_v1",
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
            "propagate the authoritative one-record full-response receipt "
            "through 1x1 beam proposal and singleton-fallback plans"
        ),
        "failed_predecessor_error": (
            "Phase-III batching appendix requires the supported-metric global "
            "trust solve for its complete batch response."
        ),
        "proof": {
            "archive_only_job_gates_passed": 6,
            "frozen_source_tests_passed": 3,
            "route_digest_unchanged": True,
            "multi_record_global_trust_still_required": True,
            "exact_remote_image_gate": "pending",
        },
    }
    _dump(bundle / "beam_singleton_trust_repair.json", receipt)
    _dump(bundle / "source_lock/beam_singleton_trust_test.json", test_receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["beam_singleton_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _refresh_hashes(bundle)


def _refresh_hashes(bundle: Path) -> None:
    _dump(
        bundle / "submission_artifact_hashes.json",
        {
            "schema": "paper_i_sr_batch3_beam_singleton_successor_artifacts_v1",
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


def _finalize_remote_preflight(spec: dict[str, str]) -> dict[str, Any]:
    base_receipt = base.finalize_remote_preflight(spec)
    bundle = INPUT / spec["output"]
    receipt_path = bundle / "beam_singleton_trust_repair.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["status"] = "pass_exact_remote_image_validated_not_submitted"
    receipt["remote_preflight_completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["proof"]["exact_remote_image_gate"] = "pass"
    receipt["proof"]["exact_remote_image_rows_passed"] = 6
    _dump(receipt_path, receipt)
    for name in ("preflight.json", "bundle_manifest.json"):
        path = bundle / name
        value = json.loads(path.read_text(encoding="utf-8"))
        value["beam_singleton_trust_repair"] = receipt
        value["submission_performed"] = False
        _dump(path, value)
    _refresh_hashes(bundle)
    return {**base_receipt, "beam_singleton_trust_repair": receipt}


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
