"""Fail-closed bindings for the Paper-II manuscript promotion receipt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest
from pypdf import PdfWriter

from pipelines.time_dynamics.diagnostics.manuscript_promotion_receipt import (
    RECEIPT_SCHEMA,
    TABLE_SCHEMA,
    bind_manuscript_promotion_receipt,
)
from pipelines.time_dynamics.diagnostics.promotion_contract import REPORT_SCHEMA


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_pdf(path: Path, *, pages: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=72, height=72)
    with path.open("wb") as handle:
        writer.write(handle)


@pytest.fixture
def manuscript_binding_case(tmp_path: Path) -> dict[str, Any]:
    ledger = tmp_path / "output/ledger.json"
    _write_json(ledger, {"schema": "fixture_ledger", "records": [{}]})
    preview = tmp_path / "report/promotion_preview.json"
    _write_json(preview, {"schema": "fixture_preview"})
    report_pdf = tmp_path / "report/paper_ii_staged_report.pdf"
    _write_pdf(report_pdf, pages=2)
    policy_transition = tmp_path / "policy/transition.json"
    _write_json(policy_transition, {"status": "PASS"})

    report_manifest = tmp_path / "report/paper_ii_staged_report.json"
    _write_json(
        report_manifest,
        {
            "schema": REPORT_SCHEMA,
            "ledger_record_count": 1,
            "source_ledger_sha256": _sha256(ledger),
            "qualification_policy": {
                "mode": "response_normalized_matched_pair",
                "matched_accuracy_factor": 2.0,
            },
            "promotion_preview": {
                "receipt": str(preview.relative_to(tmp_path)),
                "receipt_sha256": _sha256(preview),
            },
            "policy_transition": {
                "receipt": str(policy_transition.relative_to(tmp_path)),
                "receipt_sha256": _sha256(policy_transition),
            },
            "pdf": str(report_pdf.relative_to(tmp_path)),
            "pdf_sha256": _sha256(report_pdf),
            "page_count": 2,
        },
    )

    table = tmp_path / "paper/generated/frontier_table.tex"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("verified table\n", encoding="utf-8")
    table_receipt = tmp_path / "paper/table_receipt.json"
    _write_json(
        table_receipt,
        {
            "schema": TABLE_SCHEMA,
            "results_pdf": {
                "report_manifest_sha256": _sha256(report_manifest),
                "preview_receipt_sha256": _sha256(preview),
            },
            "table": {
                "path": str(table.relative_to(tmp_path)),
                "sha256": _sha256(table),
            },
            "tuple_count": 1,
            "tuple_ids": ["tuple-1"],
            "row_count": 2,
            "qualified_row_count": 1,
            "candidate_row_count": 0,
            "pending_row_count": 1,
        },
    )

    manuscript_source = tmp_path / "paper/manuscript.tex"
    manuscript_source.write_text(
        "\\input{generated/frontier_table.tex}\n", encoding="utf-8"
    )
    manuscript_pdf = tmp_path / "paper/manuscript.pdf"
    _write_pdf(manuscript_pdf, pages=3)
    figure = tmp_path / "paper/representative.pdf"
    _write_pdf(figure, pages=1)
    return {
        "workspace_root": tmp_path,
        "ledger": ledger,
        "preview": preview,
        "report_pdf": report_pdf,
        "policy_transition": policy_transition,
        "report_manifest": report_manifest,
        "table": table,
        "table_receipt": table_receipt,
        "manuscript_source": manuscript_source,
        "manuscript_pdf": manuscript_pdf,
        "figure": figure,
        "output": tmp_path / "paper/manuscript_receipt.json",
    }


def _bind(
    case: dict[str, Any],
    *,
    rendered_pages: Sequence[int] = (2, 3, 2),
    render_finding: str = "Pages 2-3 are legible and unclipped.",
) -> dict[str, Any]:
    return bind_manuscript_promotion_receipt(
        table_receipt_path=case["table_receipt"],
        report_manifest_path=case["report_manifest"],
        ledger_path=case["ledger"],
        manuscript_source_path=case["manuscript_source"],
        manuscript_pdf_path=case["manuscript_pdf"],
        output_receipt_path=case["output"],
        representative_figure_path=case["figure"],
        rendered_pages=rendered_pages,
        render_finding=render_finding,
        workspace_root=case["workspace_root"],
    )


def test_bind_manuscript_promotion_receipt_writes_the_verified_digest_chain(
    manuscript_binding_case: dict[str, Any],
) -> None:
    case = manuscript_binding_case

    receipt = _bind(case)

    assert json.loads(case["output"].read_text(encoding="utf-8")) == receipt
    assert receipt["schema"] == RECEIPT_SCHEMA
    assert receipt["ledger"] == {
        "path": "output/ledger.json",
        "sha256": _sha256(case["ledger"]),
        "record_count": 1,
    }
    assert receipt["results_pdf"]["manifest"]["sha256"] == _sha256(
        case["report_manifest"]
    )
    assert receipt["results_pdf"]["sha256"] == _sha256(case["report_pdf"])
    assert receipt["results_pdf"]["page_count"] == 2
    assert receipt["table"]["sha256"] == _sha256(case["table"])
    assert receipt["tuple_count"] == 1
    assert receipt["table_row_count"] == 2
    assert receipt["qualified_row_count"] == 1
    assert receipt["pending_row_count"] == 1
    assert receipt["manuscript"]["pdf"]["page_count"] == 3
    assert receipt["manuscript"]["render_check"] == {
        "status": "PASS_CHANGED_PAGES_INSPECTED",
        "pages": [2, 3],
        "finding": "Pages 2-3 are legible and unclipped.",
    }
    assert receipt["observable_plot_policy"]["preserved_representative_figure"][
        "sha256"
    ] == _sha256(case["figure"])


def test_bind_manuscript_promotion_receipt_rejects_a_changed_results_pdf(
    manuscript_binding_case: dict[str, Any],
) -> None:
    case = manuscript_binding_case
    _write_pdf(case["report_pdf"], pages=1)

    with pytest.raises(ValueError, match="campaign results PDF digest mismatch"):
        _bind(case)

    assert not case["output"].exists()


def test_bind_manuscript_promotion_receipt_rejects_an_inconsistent_tuple_count(
    manuscript_binding_case: dict[str, Any],
) -> None:
    case = manuscript_binding_case
    table_receipt = json.loads(case["table_receipt"].read_text(encoding="utf-8"))
    table_receipt["tuple_count"] = 2
    _write_json(case["table_receipt"], table_receipt)

    with pytest.raises(ValueError, match="tuple count is inconsistent"):
        _bind(case)


def test_bind_manuscript_promotion_receipt_requires_two_rows_per_tuple(
    manuscript_binding_case: dict[str, Any],
) -> None:
    case = manuscript_binding_case
    table_receipt = json.loads(case["table_receipt"].read_text(encoding="utf-8"))
    table_receipt["row_count"] = 3
    _write_json(case["table_receipt"], table_receipt)

    with pytest.raises(ValueError, match="two method slots per tuple"):
        _bind(case)


@pytest.mark.parametrize(
    ("rendered_pages", "message"),
    [
        ((), "requires a rendered-page check"),
        ((0,), "outside the PDF"),
        ((4,), "outside the PDF"),
    ],
)
def test_bind_manuscript_promotion_receipt_rejects_missing_or_invalid_render_pages(
    manuscript_binding_case: dict[str, Any],
    rendered_pages: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _bind(manuscript_binding_case, rendered_pages=rendered_pages)


def test_bind_manuscript_promotion_receipt_requires_a_render_finding(
    manuscript_binding_case: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="render inspection has no finding"):
        _bind(manuscript_binding_case, render_finding="  ")


def test_bind_manuscript_promotion_receipt_rejects_inconsistent_row_state_counts(
    manuscript_binding_case: dict[str, Any],
) -> None:
    case = manuscript_binding_case
    table_receipt = json.loads(case["table_receipt"].read_text(encoding="utf-8"))
    table_receipt["pending_row_count"] = 2
    _write_json(case["table_receipt"], table_receipt)

    with pytest.raises(ValueError, match="row-state counts do not sum"):
        _bind(case)
