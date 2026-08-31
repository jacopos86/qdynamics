"""Fail-closed bindings for the canonical Paper-II results manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pypdf import PdfWriter

from pipelines.time_dynamics.diagnostics.canonical_results_manifest import (
    bind_canonical_results_manifest,
)
from pipelines.time_dynamics.diagnostics.promotion_contract import (
    PROMOTION_PREVIEW_SCHEMA,
    REPORT_SCHEMA,
)
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellKey,
    METHOD_APM,
    METHOD_AVQDS,
    RungRecord,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_pdf(path: Path, *, pages: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


def _record(
    *, method: str, horizon: str, policy: AccuracyPolicy, qualified: bool
) -> RungRecord:
    value = 5.0e-5 if qualified else 2.0e-4
    return RungRecord.completed(
        key=CellKey("hh", "weakslow", method, "delta_theta_5e-5"),
        horizon=horizon,
        threshold=1.0e-6,
        run_path=f"output/{method}/{horizon}/run.json",
        metrics=AccuracyMetrics(value, value, value, value),
        policy=policy,
        resume_from_run_path=(
            None if horizon == "t5" else f"output/{method}/t5/run.json"
        ),
    )


@pytest.fixture
def canonical_binding_case(tmp_path: Path) -> dict[str, Any]:
    policy = AccuracyPolicy()
    ledger = CampaignLedger(qualification_policy=policy.to_json_dict())
    for record in (
        _record(method=METHOD_APM, horizon="t5", policy=policy, qualified=True),
        _record(
            method=METHOD_AVQDS,
            horizon="t5",
            policy=policy,
            qualified=False,
        ),
        _record(method=METHOD_APM, horizon="t20", policy=policy, qualified=True),
        _record(method=METHOD_AVQDS, horizon="t20", policy=policy, qualified=True),
    ):
        ledger = ledger.append(record)

    ledger_path = tmp_path / "output/ledger.json"
    ledger.write(ledger_path)
    preview_fragment = tmp_path / "report/promotion_preview_sections.tex"
    preview_fragment.parent.mkdir(parents=True, exist_ok=True)
    preview_fragment.write_text("planned matrix\n", encoding="utf-8")
    preview_path = tmp_path / "report/promotion_preview.json"
    preview = {
        "schema": PROMOTION_PREVIEW_SCHEMA,
        "status": "promotion_preview_with_response_normalized_pair_states",
        "qualification_policy": policy.to_json_dict(),
        "ledger": {
            "path": str(ledger_path.relative_to(tmp_path)),
            "sha256": _sha256(ledger_path),
            "record_count": 4,
        },
        "planned_tuple_count": 1,
        "planned_tuples": [
            {
                "exact_reference": {"sha256": "a" * 64},
                "methods": {
                    METHOD_APM: {"paper_status": "qualified"},
                    METHOD_AVQDS: {"paper_status": "qualified"},
                },
            }
        ],
        "pair_count": 1,
        "pairs": [
            {
                "regime": "hh",
                "drive": "weakslow",
                "controller": "delta_theta_5e-5",
                "horizon": "t20",
                "rows": {
                    METHOD_APM: {"status": "accuracy_qualified"},
                    METHOD_AVQDS: {"status": "accuracy_qualified"},
                },
            }
        ],
        "tex_fragment": {
            "path": str(preview_fragment.relative_to(tmp_path)),
            "sha256": _sha256(preview_fragment),
        },
    }
    _write_json(preview_path, preview)

    report_tex = tmp_path / "report/paper_ii_staged_report.tex"
    report_tex.write_text("report source\n", encoding="utf-8")
    report_pdf = tmp_path / "report/paper_ii_staged_report.pdf"
    _write_pdf(report_pdf)
    report_markdown = tmp_path / "report/paper_ii_staged_report.md"
    report_markdown.write_text("report summary\n", encoding="utf-8")
    report_manifest = tmp_path / "report/paper_ii_staged_report.json"
    _write_json(
        report_manifest,
        {
            "schema": REPORT_SCHEMA,
            "horizon": "t5_and_t20",
            "markdown": str(report_markdown.relative_to(tmp_path)),
            "preserved_field": "retained",
        },
    )

    scale_sidecar = tmp_path / "policy/response_scales.json"
    _write_json(scale_sidecar, {"status": "verified"})
    policy_source = tmp_path / "policy/source.py"
    policy_source.write_text("POLICY = 'fixture'\n", encoding="utf-8")
    transition_path = tmp_path / "policy/transition.json"
    _write_json(
        transition_path,
        {
            "status": "PASS",
            "ledger_before": {
                "path": "policy/ledger_before.json",
                "sha256": "b" * 64,
            },
            "ledger_after": {"sha256": _sha256(ledger_path), "record_count": 4},
            "policy_after": {
                "qualification_policy_sha256": _mapping_sha256(
                    policy.to_json_dict()
                ),
                "response_scale_sidecar": {
                    "path": str(scale_sidecar.relative_to(tmp_path)),
                    "sha256": _sha256(scale_sidecar),
                },
            },
            "policy_source_artifacts": [
                {
                    "path": str(policy_source.relative_to(tmp_path)),
                    "sha256": _sha256(policy_source),
                }
            ],
            "decision_handoff": {"status": "recorded"},
            "source_revision_before_uncommitted_repair": "fixture-revision",
        },
    )
    return {
        "workspace_root": tmp_path,
        "ledger": ledger_path,
        "preview": preview_path,
        "preview_fragment": preview_fragment,
        "report_manifest": report_manifest,
        "report_tex": report_tex,
        "report_pdf": report_pdf,
        "transition": transition_path,
        "policy": policy.to_json_dict(),
    }


def _bind(case: dict[str, Any]) -> dict[str, Any]:
    return bind_canonical_results_manifest(
        ledger_path=case["ledger"],
        preview_receipt_path=case["preview"],
        report_manifest_path=case["report_manifest"],
        report_tex_path=case["report_tex"],
        report_pdf_path=case["report_pdf"],
        policy_transition_receipt_path=case["transition"],
        workspace_root=case["workspace_root"],
    )


def test_bind_canonical_results_manifest_writes_the_verified_digest_chain(
    canonical_binding_case: dict[str, Any],
) -> None:
    case = canonical_binding_case

    manifest = _bind(case)

    assert json.loads(case["report_manifest"].read_text(encoding="utf-8")) == manifest
    assert manifest["preserved_field"] == "retained"
    assert manifest["ledger_record_count"] == 4
    assert manifest["source_ledger_sha256"] == _sha256(case["ledger"])
    assert manifest["qualified_pair_count"] == 1
    assert manifest["t20_accuracy_qualified_method_arm_count"] == 2
    assert manifest["t20_candidate_method_arm_count"] == 0
    assert manifest["t20_pending_method_arm_count"] == 0
    assert manifest["t20_evaluated_trajectory_count"] == 2
    assert manifest["t5_accuracy_qualified_apm_count"] == 1
    assert manifest["t5_accuracy_qualified_avqds_count"] == 0
    assert manifest["t5_requires_threshold_extension_count"] == 1
    assert manifest["qualification_policy"] == case["policy"]
    assert manifest["promotion_preview"]["receipt_sha256"] == _sha256(
        case["preview"]
    )
    assert manifest["policy_transition"]["receipt_sha256"] == _sha256(
        case["transition"]
    )
    assert manifest["tex_sha256"] == _sha256(case["report_tex"])
    assert manifest["pdf_sha256"] == _sha256(case["report_pdf"])
    assert manifest["page_count"] == 1


def test_bind_canonical_results_manifest_rejects_a_changed_preview_fragment(
    canonical_binding_case: dict[str, Any],
) -> None:
    case = canonical_binding_case
    manifest_before = case["report_manifest"].read_bytes()
    case["preview_fragment"].write_text("changed matrix\n", encoding="utf-8")

    with pytest.raises(ValueError, match="TeX fragment digest mismatch"):
        _bind(case)

    assert case["report_manifest"].read_bytes() == manifest_before
    assert not case["report_manifest"].with_suffix(".json.part").exists()


def test_bind_canonical_results_manifest_rejects_a_preview_record_count_mismatch(
    canonical_binding_case: dict[str, Any],
) -> None:
    case = canonical_binding_case
    preview = json.loads(case["preview"].read_text(encoding="utf-8"))
    preview["ledger"]["record_count"] = 5
    _write_json(case["preview"], preview)

    with pytest.raises(ValueError, match="record count differs from the ledger"):
        _bind(case)


def test_bind_canonical_results_manifest_rejects_a_transition_for_another_policy(
    canonical_binding_case: dict[str, Any],
) -> None:
    case = canonical_binding_case
    transition = json.loads(case["transition"].read_text(encoding="utf-8"))
    transition["policy_after"]["qualification_policy_sha256"] = "0" * 64
    _write_json(case["transition"], transition)

    with pytest.raises(ValueError, match="does not bind the exact ledger policy"):
        _bind(case)


def test_bind_canonical_results_manifest_rejects_a_preview_for_another_policy(
    canonical_binding_case: dict[str, Any],
) -> None:
    case = canonical_binding_case
    preview = json.loads(case["preview"].read_text(encoding="utf-8"))
    preview["qualification_policy"] = {
        **preview["qualification_policy"],
        "ceiling": 9.0e-4,
    }
    _write_json(case["preview"], preview)

    with pytest.raises(ValueError, match="preview policy differs from the ledger"):
        _bind(case)
