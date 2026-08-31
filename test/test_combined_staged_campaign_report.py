"""Fail-closed combined t5+t20 Paper-II report builder tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pypdf import PdfWriter
import pytest

from pipelines.time_dynamics.diagnostics import combined_staged_campaign_report as report
from pipelines.time_dynamics.diagnostics.promotion_contract import (
    APM_METHOD,
    AVQDS_METHOD,
    PROMOTION_PREVIEW_SCHEMA,
    planned_tuple_id,
)
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellKey,
    RungRecord,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    policy = AccuracyPolicy()
    ledger = CampaignLedger(qualification_policy=policy.to_json_dict())
    for horizon, method, cut, cluster in (
        ("t5", "avqds", 3.0e-6, 1001),
        ("t20", "exchange", 1.0e-6, 1002),
    ):
        ledger = ledger.append(
            RungRecord.completed(
                key=CellKey("hh", "weakslow", method, "state_motion_5e-5"),
                horizon=horizon,
                threshold=cut,
                run_path=f"output/attempts/cluster_{cluster}/run.json",
                resume_from_run_path=(
                    None
                    if horizon == "t5"
                    else "output/attempts/cluster_1001/run.json"
                ),
                metrics=AccuracyMetrics(1.0e-5, 2.0e-5, 3.0e-5, 4.0e-5),
                policy=policy,
            )
        )
    ledger_path = tmp_path / "output/ledger.json"
    ledger.write(ledger_path)

    output_dir = tmp_path / "output/pdf/report"
    fragment = output_dir / "promotion_preview_sections.tex"
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(
        "\\section*{Planned t20 exact/APM/AVQDS tuple matrix}\n",
        encoding="utf-8",
    )
    exact = tmp_path / "output/exact/reference.json"
    exact.parent.mkdir(parents=True, exist_ok=True)
    exact.write_text("{}\n", encoding="utf-8")
    latest_artifacts = {}
    for name in ("run", "screen_run", "cost"):
        artifact = tmp_path / f"output/evidence/{name}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(f'{{"kind": "{name}"}}\n', encoding="utf-8")
        latest_artifacts[f"{name}_path"] = str(artifact.relative_to(tmp_path))
        latest_artifacts[f"{name}_sha256"] = _sha256(artifact)
    planned = {
        "regime": "hh",
        "drive": "weakslow",
        "controller": "state_motion_5e-5",
        "horizon": "t20",
        "exact_reference": {
            "path": str(exact.relative_to(tmp_path)),
            "sha256": _sha256(exact),
        },
        "methods": {
            APM_METHOD: {
                "paper_status": "candidate",
                "latest_row": {
                    "method": APM_METHOD,
                    "status": "requires_threshold_extension",
                    **latest_artifacts,
                },
            },
            AVQDS_METHOD: {"paper_status": "pending", "latest_row": None},
        },
    }
    planned["tuple_id"] = planned_tuple_id(planned)
    preview_path = output_dir / "promotion_preview.json"
    _write_json(
        preview_path,
        {
            "schema": PROMOTION_PREVIEW_SCHEMA,
            "status": "fixture",
            "qualification_policy": policy.to_json_dict(),
            "ledger": {
                "path": str(ledger_path.relative_to(tmp_path)),
                "sha256": _sha256(ledger_path),
                "record_count": 2,
            },
            "planned_tuple_count": 1,
            "planned_tuples": [planned],
            "pair_count": 0,
            "pairs": [],
            "tex_fragment": {
                "path": str(fragment.relative_to(tmp_path)),
                "sha256": _sha256(fragment),
            },
        },
    )

    supplement = tmp_path / "evidence/locked_t5.tex"
    supplement.parent.mkdir(parents=True, exist_ok=True)
    supplement.write_text(
        "\\section*{Archived pre-change t5 screening pairs}\n",
        encoding="utf-8",
    )
    frontier_path = tmp_path / "output/frontier.json"
    _write_json(
        frontier_path,
        {
            "schema": report.COMBINED_FRONTIER_SCHEMA,
            "snapshot_id": "snapshot-001",
            "snapshot_utc": "2026-08-28T18:00:00+00:00",
            "audit_status": "PASS_FIXTURE",
            "scheduler_summary": "one running action; no holds",
            "ledger": {"sha256": _sha256(ledger_path), "record_count": 2},
            "promotion_preview": {
                "sha256": _sha256(preview_path),
                "planned_tuple_count": 1,
            },
            "cluster_ids": [1001, 1002, 1003],
            "active_clusters": [{"cluster_id": 1003, "running_process_count": 1}],
            "excluded_cluster_ids": [999],
            "incremental_ingestion_receipts": ["receipts/increment.json"],
            "submission_receipts": ["receipts/submission.json"],
            "scheduler_rows": [
                {
                    "job_id": "1003.0",
                    "method": "AVQDS",
                    "controller": "state motion",
                    "drive": "strongfast",
                    "threshold": 9.0e-6,
                    "state": "running",
                    "progress": "t=3.0/5.0",
                    "available_data": "checkpoint only",
                }
            ],
            "supplemental_sections": [
                {
                    "role": "locked_t5_screening",
                    "tex_fragment": {
                        "path": str(supplement.relative_to(tmp_path)),
                        "sha256": _sha256(supplement),
                    },
                    "figures": [],
                    "markdown_summary": "Two historical screening pairs remain visible.",
                }
            ],
        },
    )
    return {
        "workspace": tmp_path,
        "ledger": ledger_path,
        "preview": preview_path,
        "frontier": frontier_path,
        "output": output_dir,
        "fragment": fragment,
    }


def test_combined_builder_preserves_both_horizons_slots_and_frontier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path)

    def compile_fixture(tex_path: Path, *, source_date_epoch: int):
        assert source_date_epoch == 1787940000
        pdf = tex_path.parent / "fixture.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=200)
        with pdf.open("wb") as handle:
            writer.write(handle)
        return pdf, "fixture"

    monkeypatch.setattr(report, "_compile_pdf", compile_fixture)
    manifest = report.build_combined_report(
        ledger_path=case["ledger"],
        promotion_preview_path=case["preview"],
        frontier_metadata_path=case["frontier"],
        output_dir=case["output"],
        workspace_root=case["workspace"],
    )

    tex = (case["output"] / "paper_ii_staged_report.tex").read_text(encoding="utf-8")
    markdown = (case["output"] / "paper_ii_staged_report.md").read_text(
        encoding="utf-8"
    )
    assert manifest["horizon"] == "t5_and_t20"
    assert manifest["builder_schema"] == report.COMBINED_REPORT_BUILDER_SCHEMA
    assert manifest["ledger_record_count"] == 2
    assert manifest["t20_candidate_method_arm_count"] == 1
    assert manifest["t20_pending_method_arm_count"] == 1
    assert manifest["scheduler_row_count"] == 1
    assert manifest["active_clusters"] == [
        {"cluster_id": 1003, "running_process_count": 1}
    ]
    assert "Current t20 frontier" in tex
    assert "Current t5 frontier" in tex
    assert "1003.0" in tex
    assert "Planned t20 exact/APM/AVQDS tuple matrix" in tex
    assert "Archived pre-change t5 screening pairs" in tex
    assert "Current t20 frontier" in markdown
    assert "Current t5 frontier" in markdown
    assert (case["output"] / "paper_ii_staged_report.pdf").is_file()
    assert not list(case["output"].glob("*.part"))


def test_combined_builder_fails_closed_on_stale_preview_link(tmp_path: Path) -> None:
    case = _fixture(tmp_path)
    frontier = json.loads(case["frontier"].read_text(encoding="utf-8"))
    frontier["promotion_preview"]["sha256"] = "0" * 64
    _write_json(case["frontier"], frontier)
    sentinel = case["output"] / "paper_ii_staged_report.tex"
    sentinel.write_text("preserve me\n", encoding="utf-8")

    with pytest.raises(report.CombinedReportContractError, match="stale.*preview"):
        report.build_combined_report(
            ledger_path=case["ledger"],
            promotion_preview_path=case["preview"],
            frontier_metadata_path=case["frontier"],
            output_dir=case["output"],
            workspace_root=case["workspace"],
        )
    assert sentinel.read_text(encoding="utf-8") == "preserve me\n"


def test_compiler_failure_does_not_overwrite_published_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path)
    published = case["output"] / "paper_ii_staged_report.pdf"
    published.write_bytes(b"previous-pdf")

    def fail_compile(_tex_path: Path, *, source_date_epoch: int):
        raise RuntimeError(f"compiler failed at {source_date_epoch}")

    monkeypatch.setattr(report, "_compile_pdf", fail_compile)
    with pytest.raises(RuntimeError, match="compiler failed"):
        report.build_combined_report(
            ledger_path=case["ledger"],
            promotion_preview_path=case["preview"],
            frontier_metadata_path=case["frontier"],
            output_dir=case["output"],
            workspace_root=case["workspace"],
        )
    assert published.read_bytes() == b"previous-pdf"
    assert not list(case["output"].glob("*.part"))


def test_combined_builder_accepts_terminal_unresolved_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = _fixture(tmp_path)
    preview = json.loads(case["preview"].read_text(encoding="utf-8"))
    slot = preview["planned_tuples"][0]["methods"][AVQDS_METHOD]
    slot.update(
        {
            "paper_status": "unresolved",
            "state_reason": "resource_retry_exhausted",
            "latest_evidence_status": "operationally_infeasible",
            "latest_row": None,
        }
    )
    preview["planned_tuples"][0]["tuple_id"] = planned_tuple_id(
        preview["planned_tuples"][0]
    )
    _write_json(case["preview"], preview)
    frontier = json.loads(case["frontier"].read_text(encoding="utf-8"))
    frontier["promotion_preview"]["sha256"] = _sha256(case["preview"])
    _write_json(case["frontier"], frontier)

    def compile_fixture(tex_path: Path, *, source_date_epoch: int):
        pdf = tex_path.parent / "fixture.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=300, height=200)
        with pdf.open("wb") as handle:
            writer.write(handle)
        return pdf, str(source_date_epoch)

    monkeypatch.setattr(report, "_compile_pdf", compile_fixture)
    manifest = report.build_combined_report(
        ledger_path=case["ledger"],
        promotion_preview_path=case["preview"],
        frontier_metadata_path=case["frontier"],
        output_dir=case["output"],
        workspace_root=case["workspace"],
    )

    assert manifest["t20_unresolved_method_arm_count"] == 1
