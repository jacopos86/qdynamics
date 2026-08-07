#!/usr/bin/env python3
"""Build source-anchored CHTC workflows for plateau-triggered insertion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from chtc.phase3_optuna.generate_paper_i_hh_commutation_reduced_insertion_r50_bundles import (
    MACRO_PARENT_ID,
    SINGLETON_PARENT_ID,
    _build_bundle,
)
from pipelines.reporting.build_paper_i_hh_tracking_plateau_costs import (
    _read_source_result,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    canonical_sr_snake_contract,
    canonical_sr_snake_contract_sha256,
    normalize_sr_route_profile_request,
)


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "chtc" / "phase3_optuna" / "input"
TRACKER = ROOT / (
    "output/pdf/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715/"
    "paper_i_hh_sr_snake_no_prune_no_beam_no_ordinary_novelty_tracking_20260715.json"
)
REPORT_PROVENANCE = ROOT / (
    "MATH/paper_details/figures/paper_i_hh_macro_common_accuracy_20260723/"
    "paper_i_hh_macro_common_accuracy_20260723_provenance.json"
)
VALIDATOR = ROOT / "chtc/phase3_optuna/validate_paper_i_source_locked_anchor.py"

MACRO_ANCHOR_ID = (
    "paper_i_hh_sr_snake_macro_plateau_insertion_source_anchor_"
    "weak_weak_r9_20260725_v1_chtc"
)
MACRO_CHILD_ID = (
    "paper_i_hh_sr_snake_macro_insertion_commutation_plateau_"
    "all_six_r50_20260725_v1_chtc"
)
SINGLETON_ANCHOR_ID = (
    "paper_i_hh_sr_snake_singleton_plateau_insertion_source_anchor_"
    "weak_weak_r29_20260725_v1_chtc"
)
SINGLETON_CHILD_ID = (
    "paper_i_hh_sr_snake_singleton_insertion_commutation_plateau_"
    "all_six_r50_20260725_v2_chtc"
)

MACRO_PARENT_PROFILE = "sr_snake_macro_only_physical_lanes_v1"
MACRO_CHILD_PROFILE = (
    "sr_snake_macro_only_physical_lanes_insertion_commutation_plateau_v1"
)
SINGLETON_PARENT_PROFILE = (
    "sr_snake_no_prune_symmetric_cost_projected_phase3_no_overlap_trust_v1"
)
SINGLETON_CHILD_PROFILE = "insertion_commutation_plateau_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _patch_ledger_schema_compatibility(bundle_id: str) -> None:
    """Accept both complete estimator-ledger sidecar payload versions."""

    path = INPUT_ROOT / bundle_id / "evidence_validation.py"
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        'LEDGER_SCHEMA = "paper_i_estimator_call_ledger_sidecar_v1"',
        (
            "LEDGER_SCHEMAS = frozenset({\n"
            '    "paper_i_estimator_call_ledger_sidecar_v1",\n'
            '    "paper_i_estimator_call_ledger_sidecar_v2",\n'
            "})"
        ),
        1,
    )
    text = text.replace(
        'if ledger_sidecar.get("schema") != LEDGER_SCHEMA:',
        'if ledger_sidecar.get("schema") not in LEDGER_SCHEMAS:',
        1,
    )
    if "LEDGER_SCHEMA =" in text or "!= LEDGER_SCHEMA" in text:
        raise RuntimeError(f"{bundle_id}: estimator-ledger schema patch failed")
    path.write_text(text, encoding="utf-8")


def _adapt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    direct = payload.get("adapt_vqe")
    if isinstance(direct, Mapping):
        return direct
    wrapped = payload.get("result")
    if isinstance(wrapped, Mapping) and isinstance(wrapped.get("adapt_vqe"), Mapping):
        return wrapped["adapt_vqe"]
    raise ValueError("source payload lacks adapt_vqe")


def _expectation(
    *,
    tracker: Mapping[str, Any],
    route_id: str,
    depth: int,
    profile_request: str,
    visible_page: int,
    visible_error: float,
) -> dict[str, Any]:
    route = next(item for item in tracker["routes"] if item["id"] == route_id)
    source = route["results"]["weak_weak"]["source"]
    payload, _runtime_seed, receipt = _read_source_result(
        source,
        need_runtime_seed=False,
    )
    history = list(_adapt(payload)["history"])
    if len(history) < depth:
        raise ValueError(f"{route_id}: source history ends before k={depth}")
    prefix = history[:depth]
    final_error = float(prefix[-1]["delta_abs_current"])
    if abs(final_error - visible_error) > 1.0e-12:
        raise ValueError(f"{route_id}: visible/source prefix error drift")
    profile_resolved = normalize_sr_route_profile_request(profile_request)
    return {
        "schema": "paper_i_source_value_anchor_expectation_v1",
        "visible_artifact": (
            "MATH/paper_details/figures/"
            "paper_i_hh_macro_common_accuracy_20260723/"
            "paper_i_hh_macro_common_accuracy_20260723.pdf"
        ),
        "visible_page": visible_page,
        "route_id": route_id,
        "regime": "weak_weak",
        "depth": depth,
        "selected_labels": [str(row["selected_op"]) for row in prefix],
        "energies": [float(row["energy_after_opt"]) for row in prefix],
        "final_error": final_error,
        "energy_abs_tolerance": 1.0e-10,
        "final_error_abs_tolerance": 1.0e-10,
        "source": receipt,
        "profile_request": profile_request,
        "profile_resolved": profile_resolved,
        "profile_contract_sha256": canonical_sr_snake_contract_sha256(
            profile_request
        ),
        "execution_settings_sha256": _canonical_sha256(
            canonical_sr_snake_contract(profile_request)["execution_settings"]
        ),
    }


def _patch_anchor_bundle(
    *,
    bundle_id: str,
    expectation: Mapping[str, Any],
    child_bundle_id: str,
) -> Path:
    bundle = INPUT_ROOT / bundle_id
    expectation_path = bundle / "anchor_expectation.json"
    _dump(expectation_path, expectation)
    audit_path = bundle / "source_locked_sensitivity_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit.update(
        {
            "status": "pending_anchor",
            "source": {
                "table_or_figure": expectation["visible_artifact"],
                "page": expectation["visible_page"],
                "method": "SNAKE",
                "regime_or_case": "weak_weak",
                "source_json": expectation["source"].get("path"),
                "source_sha256": expectation["source"].get("sha256"),
                "runner_mode": "direct_source_profile",
                "route_or_profile_id": expectation["profile_request"],
                "settings_hash": expectation["execution_settings_sha256"],
                "source_variable_value": "append_only",
            },
            "sweep": {
                "run_class": "diagnostic",
                "variable": "adapt_insertion_mode",
                "grid": ["append_only"],
                "runner_mode": "direct_source_profile",
                "wrapper_used": False,
                "wrapper_kind": None,
                "baseline_materialization_status": "complete",
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "settings_changed": [],
            },
            "planned_rows": [
                {
                    "value": "append_only",
                    "settings_hash": expectation["execution_settings_sha256"],
                    "changed_fields_vs_source": [],
                    "non_swept_settings_diff": [],
                }
            ],
            "anchor": {
                "value": "append_only",
                "anchor_result_json": None,
                "anchor_reproduces_source": False,
                "non_swept_settings_diff": [],
            },
            "child_bundle": child_bundle_id,
        }
    )
    _dump(audit_path, audit)

    for directory in (bundle / "jobs", bundle / "normalized_manifests"):
        for path in directory.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sensitivity_study"] = {
                "schema": "source_locked_sensitivity_anchor_v1",
                "swept_field": "adapt_insertion_mode",
                "source_value": "append_only",
                "changed_execution_fields": [],
                "non_swept_settings_diff": [],
                "status": "pending_anchor",
            }
            _dump(path, payload)

    result_path = bundle / "anchor_result/weak_weak_transfer.tar.gz"
    result_path.parent.mkdir()
    submit_path = bundle / "submit.sub"
    lines = submit_path.read_text(encoding="utf-8").splitlines()
    replacement = (
        'transfer_output_remaps = "'
        f"raw_outputs/{bundle_id}/$(regime_slug)_transfer.tar.gz = "
        f"{result_path.relative_to(ROOT)}"
        '"'
    )
    lines = [
        replacement if line.startswith("transfer_output_remaps =") else line
        for line in lines
    ]
    submit_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return expectation_path


def _patch_child_audit(
    *,
    child_bundle_id: str,
    anchor_bundle_id: str,
    parent_profile: str,
    child_profile: str,
) -> None:
    bundle = INPUT_ROOT / child_bundle_id
    audit_path = bundle / "source_locked_sensitivity_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    parent_settings = canonical_sr_snake_contract(parent_profile)[
        "execution_settings"
    ]
    child_settings = canonical_sr_snake_contract(child_profile)[
        "execution_settings"
    ]
    differences = sorted(
        key
        for key in set(parent_settings) | set(child_settings)
        if parent_settings.get(key) != child_settings.get(key)
    )
    if differences != ["adapt_insertion_mode"]:
        raise RuntimeError(
            f"{child_bundle_id}: non-swept settings drift: {differences}"
        )
    planned_rows = []
    for path in sorted((bundle / "normalized_manifests").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        planned_rows.append(
            {
                "regime": path.stem,
                "value": "insertion_commutation_plateau_v1",
                "settings_hash": _canonical_sha256(
                    payload["route_identity"]["profile_contract"][
                        "execution_settings"
                    ]
                ),
                "changed_fields_vs_source": ["adapt_insertion_mode"],
                "non_swept_settings_diff": [],
            }
        )
    audit.update(
        {
            "status": "pending_anchor",
            "sweep": {
                **audit["sweep"],
                "runner_mode": "direct_source_profile",
                "wrapper_used": False,
                "wrapper_kind": None,
                "baseline_materialization_status": "complete",
                "unresolved_source_fields": [],
                "fields_added_by_current_defaults": [],
                "settings_changed": ["adapt_insertion_mode"],
            },
            "planned_rows": planned_rows,
            "anchor": {
                "anchor_bundle": anchor_bundle_id,
                "anchor_result_json": None,
                "anchor_reproduces_source": False,
                "non_swept_settings_diff": [],
            },
        }
    )
    _dump(audit_path, audit)


def _write_dag(
    *,
    anchor_bundle_id: str,
    child_bundle_id: str,
    expectation_path: Path,
) -> Path:
    anchor = INPUT_ROOT / anchor_bundle_id
    child = INPUT_ROOT / child_bundle_id
    dag = INPUT_ROOT / f"{child_bundle_id}.dag"
    dag.write_text(
        "\n".join(
            (
                f"JOB SOURCE_ANCHOR {anchor.relative_to(ROOT)}/submit.sub",
                (
                    "SCRIPT POST SOURCE_ANCHOR python3 "
                    f"{VALIDATOR.relative_to(ROOT)} "
                    f"--transfer-tar "
                    f"{anchor.relative_to(ROOT)}/anchor_result/weak_weak_transfer.tar.gz "
                    f"--expectation {expectation_path.relative_to(ROOT)} "
                    f"--anchor-audit "
                    f"{anchor.relative_to(ROOT)}/source_locked_sensitivity_audit.json "
                    f"--child-audit "
                    f"{child.relative_to(ROOT)}/source_locked_sensitivity_audit.json "
                    f"--child-bundle {child.relative_to(ROOT)}"
                ),
                f"JOB PLATEAU_FANOUT {child.relative_to(ROOT)}/submit.sub",
                "PARENT SOURCE_ANCHOR CHILD PLATEAU_FANOUT",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return dag


def _build_workflow(
    *,
    parent_id: str,
    anchor_id: str,
    child_id: str,
    parent_profile: str,
    child_profile: str,
    route_id: str,
    visible_page: int,
    visible_depth: int,
    visible_error: float,
    representation: str,
    memory_mb: int,
) -> dict[str, Any]:
    _build_bundle(
        parent_id=parent_id,
        bundle_id=anchor_id,
        profile_request=parent_profile,
        batch_name=f"Paper-I HH {representation} source-value anchor",
        segment_tag=f"sr-{representation}-plateau-insertion-anchor",
        condor_batch_name=f"paper-i-hh-{representation}-plateau-anchor-20260725-v1",
        memory_mb=memory_mb,
        child_insertion_mode="append_only",
        campaign_date="20260725",
        insertion_overlay_kind="plateau_insertion_source_anchor_runtime_v1",
        regime_slugs=frozenset({"weak_weak"}),
        anchor_status="pending_anchor",
        target_depth=visible_depth,
    )
    _patch_ledger_schema_compatibility(anchor_id)
    _build_bundle(
        parent_id=parent_id,
        bundle_id=child_id,
        profile_request=child_profile,
        batch_name=f"Paper-I HH {representation} plateau insertion k=50",
        segment_tag=f"sr-{representation}-insertion-commutation-plateau",
        condor_batch_name=f"paper-i-hh-{representation}-plateau-six-r50-20260725-v1",
        memory_mb=memory_mb,
        child_insertion_mode="insertion_commutation_plateau_v1",
        campaign_date="20260725",
        insertion_overlay_kind="plateau_triggered_commutation_insertion_runtime_v1",
        anchor_status="pending_anchor",
        anchor_bundle_id=anchor_id,
        target_depth=50,
    )
    _patch_ledger_schema_compatibility(child_id)
    expectation = _expectation(
        tracker=json.loads(TRACKER.read_text(encoding="utf-8")),
        route_id=route_id,
        depth=visible_depth,
        profile_request=parent_profile,
        visible_page=visible_page,
        visible_error=visible_error,
    )
    expectation_path = _patch_anchor_bundle(
        bundle_id=anchor_id,
        expectation=expectation,
        child_bundle_id=child_id,
    )
    _patch_child_audit(
        child_bundle_id=child_id,
        anchor_bundle_id=anchor_id,
        parent_profile=parent_profile,
        child_profile=child_profile,
    )
    dag = _write_dag(
        anchor_bundle_id=anchor_id,
        child_bundle_id=child_id,
        expectation_path=expectation_path,
    )
    return {
        "anchor_bundle": anchor_id,
        "child_bundle": child_id,
        "dag": str(dag.relative_to(ROOT)),
        "visible_page": visible_page,
        "visible_depth": visible_depth,
        "visible_error": visible_error,
        "anchor_submit_sha256": _sha256(INPUT_ROOT / anchor_id / "submit.sub"),
        "child_submit_sha256": _sha256(INPUT_ROOT / child_id / "submit.sub"),
    }


def main() -> int:
    provenance = json.loads(REPORT_PROVENANCE.read_text(encoding="utf-8"))
    macro_row = next(
        row
        for row in provenance["rows"]
        if row["regime"] == "weak_weak" and row["method"] == "snake"
    )
    singleton_row = next(
        row
        for row in provenance["singleton_own_plateau_common_accuracy"]["rows"]
        if row["regime"] == "weak_weak"
        and row["method"] == "snake_singleton"
    )
    workflows = [
        _build_workflow(
            parent_id=MACRO_PARENT_ID,
            anchor_id=MACRO_ANCHOR_ID,
            child_id=MACRO_CHILD_ID,
            parent_profile=MACRO_PARENT_PROFILE,
            child_profile=MACRO_CHILD_PROFILE,
            route_id="sr_macro_physical_lanes_nph3_7",
            visible_page=2,
            visible_depth=int(macro_row["k_cross"]),
            visible_error=float(macro_row["crossing_error"]),
            representation="macro",
            memory_mb=32768,
        ),
        _build_workflow(
            parent_id=SINGLETON_PARENT_ID,
            anchor_id=SINGLETON_ANCHOR_ID,
            child_id=SINGLETON_CHILD_ID,
            parent_profile=SINGLETON_PARENT_PROFILE,
            child_profile=SINGLETON_CHILD_PROFILE,
            route_id="no_overlap_trust_projected_phase3_nph3_7",
            visible_page=4,
            visible_depth=int(singleton_row["k_cross"]),
            visible_error=float(singleton_row["crossing_error"]),
            representation="singleton",
            memory_mb=40960,
        ),
    ]
    print(json.dumps(workflows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
