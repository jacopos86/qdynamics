"""Bind the Paper-II manuscript to a verified results-PDF-first table."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from pypdf import PdfReader

from pipelines.time_dynamics.diagnostics.promotion_contract import REPORT_SCHEMA


TABLE_SCHEMA = "paper_ii_results_pdf_first_table_v2"
RECEIPT_SCHEMA = "paper_ii_results_pdf_first_manuscript_promotion_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return payload


def _resolve(path: str | Path, *, workspace_root: Path) -> Path:
    target = Path(path)
    return (target if target.is_absolute() else workspace_root / target).resolve()


def _verify(
    artifact: Mapping[str, Any], *, workspace_root: Path, label: str
) -> Path:
    try:
        path = _resolve(str(artifact["path"]), workspace_root=workspace_root)
        expected = str(artifact["sha256"])
    except KeyError as exc:
        raise ValueError(f"{label} lacks path/SHA-256") from exc
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"{label} is missing or empty: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(
            f"{label} digest mismatch: expected {expected}, observed {observed}"
        )
    return path


def bind_manuscript_promotion_receipt(
    *,
    table_receipt_path: Path,
    report_manifest_path: Path,
    ledger_path: Path,
    manuscript_source_path: Path,
    manuscript_pdf_path: Path,
    output_receipt_path: Path,
    representative_figure_path: Path,
    rendered_pages: Sequence[int],
    render_finding: str,
    workspace_root: Path = Path("."),
) -> dict[str, Any]:
    """Write the digest chain only after all promotion inputs agree."""

    workspace_root = workspace_root.resolve()
    paths = [
        table_receipt_path,
        report_manifest_path,
        ledger_path,
        manuscript_source_path,
        manuscript_pdf_path,
        output_receipt_path,
        representative_figure_path,
    ]
    (
        table_receipt_path,
        report_manifest_path,
        ledger_path,
        manuscript_source_path,
        manuscript_pdf_path,
        output_receipt_path,
        representative_figure_path,
    ) = tuple(path.resolve() for path in paths)

    table_receipt = _read_object(table_receipt_path, label="table receipt")
    report = _read_object(report_manifest_path, label="report manifest")
    if table_receipt.get("schema") != TABLE_SCHEMA:
        raise ValueError("table receipt does not use the results-PDF-first schema")
    if report.get("schema") != REPORT_SCHEMA:
        raise ValueError("report manifest does not use the canonical schema")
    if table_receipt.get("results_pdf", {}).get(
        "report_manifest_sha256"
    ) != _sha256(report_manifest_path):
        raise ValueError("table receipt is stale against the report manifest")
    if table_receipt.get("results_pdf", {}).get(
        "preview_receipt_sha256"
    ) != report.get("promotion_preview", {}).get("receipt_sha256"):
        raise ValueError("table and report bind different promotion previews")
    table_path = _verify(
        table_receipt.get("table") or {},
        workspace_root=workspace_root,
        label="generated Paper-II table",
    )
    report_pdf = _verify(
        {"path": report.get("pdf"), "sha256": report.get("pdf_sha256")},
        workspace_root=workspace_root,
        label="campaign results PDF",
    )
    preview_path = _verify(
        {
            "path": report.get("promotion_preview", {}).get("receipt"),
            "sha256": report.get("promotion_preview", {}).get("receipt_sha256"),
        },
        workspace_root=workspace_root,
        label="promotion-preview receipt",
    )
    if report.get("source_ledger_sha256") != _sha256(ledger_path):
        raise ValueError("report manifest is stale against the canonical ledger")
    if not manuscript_source_path.is_file() or not manuscript_pdf_path.is_file():
        raise ValueError("manuscript source/PDF is missing")
    source = manuscript_source_path.read_text(encoding="utf-8")
    table_relative = table_path.relative_to(manuscript_source_path.parent)
    expected_input = rf"\input{{{table_relative.as_posix()}}}"
    if expected_input not in source:
        raise ValueError("manuscript source does not input the verified table")
    if not rendered_pages:
        raise ValueError("a changed reader-facing table requires a rendered-page check")
    page_count = len(PdfReader(manuscript_pdf_path).pages)
    pages = sorted(set(int(page) for page in rendered_pages))
    if pages[0] < 1 or pages[-1] > page_count:
        raise ValueError("rendered manuscript page is outside the PDF")
    if not str(render_finding).strip():
        raise ValueError("render inspection has no finding")

    policy_transition = dict(report.get("policy_transition") or {})
    policy_path = _verify(
        {
            "path": policy_transition.get("receipt"),
            "sha256": policy_transition.get("receipt_sha256"),
        },
        workspace_root=workspace_root,
        label="policy transition receipt",
    )
    tuple_ids = list(table_receipt.get("tuple_ids") or ())
    if int(table_receipt.get("tuple_count", -1)) != len(tuple_ids):
        raise ValueError("table receipt tuple count is inconsistent")
    row_count = int(table_receipt.get("row_count", -1))
    if row_count != 2 * len(tuple_ids):
        raise ValueError("table receipt does not contain two method slots per tuple")
    row_state_counts = {
        "qualified": int(table_receipt.get("qualified_row_count", -1)),
        "candidate": int(table_receipt.get("candidate_row_count", -1)),
        "pending": int(table_receipt.get("pending_row_count", -1)),
    }
    if any(value < 0 for value in row_state_counts.values()):
        raise ValueError("table receipt row-state counts must be nonnegative")
    if sum(row_state_counts.values()) != row_count:
        raise ValueError("table receipt row-state counts do not sum to row_count")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "response_normalized_matched_accuracy_promoted",
        "evidence_label": "artifact_verified_response_normalized_qualified_or_pending",
        "accuracy_claim": (
            "Observable errors are normalized by exact induced RMS response. "
            "Completed pairwise cost comparisons require both the aggregate "
            "accuracy burdens and every deadbanded primitive q component to "
            "agree within the declared factor-two rule; only unfinished arms "
            "remain pending."
        ),
        "cell_identity": [
            "physical_regime_and_seed_provenance",
            "drive",
            "time_step_controller",
            "t20_horizon",
        ],
        "ledger": {
            "path": str(ledger_path.relative_to(workspace_root)),
            "sha256": _sha256(ledger_path),
            "record_count": int(report["ledger_record_count"]),
        },
        "qualification_policy": dict(report.get("qualification_policy") or {}),
        "policy_transition": {
            "path": str(policy_path.relative_to(workspace_root)),
            "sha256": _sha256(policy_path),
        },
        "results_pdf": {
            "path": str(report_pdf.relative_to(workspace_root)),
            "sha256": _sha256(report_pdf),
            "page_count": int(report["page_count"]),
            "manifest": {
                "path": str(report_manifest_path.relative_to(workspace_root)),
                "sha256": _sha256(report_manifest_path),
            },
            "tuple_receipt": {
                "path": str(preview_path.relative_to(workspace_root)),
                "sha256": _sha256(preview_path),
            },
        },
        "table": {
            "path": str(table_path.relative_to(workspace_root)),
            "sha256": _sha256(table_path),
            "coverage_receipt": {
                "path": str(table_receipt_path.relative_to(workspace_root)),
                "sha256": _sha256(table_receipt_path),
            },
        },
        "tuple_count": len(tuple_ids),
        "tuple_ids": tuple_ids,
        "table_row_count": row_count,
        "qualified_row_count": row_state_counts["qualified"],
        "completed_candidate_row_count": row_state_counts["candidate"],
        "pending_row_count": row_state_counts["pending"],
        "reader_facing_evidence_column": False,
        "reader_facing_table_columns": [
            "drive",
            "time_step_controller",
            "method",
            "mean_infidelity",
            "mean_abs_energy_error",
            "mean_abs_doublon_error",
            "mean_abs_charge_imbalance_error",
            "accepted_integration_steps",
            "terminal_active_coordinate_count",
            "terminal_graph_cost_tuple",
        ],
        "observable_plot_policy": {
            "automatic_new_manuscript_plots": False,
            "campaign_results_pdf_holds_all_candidate_plots": True,
            "publication_subset_selected_by_author": True,
            "preserved_representative_figure": {
                "path": str(representative_figure_path.relative_to(workspace_root)),
                "sha256": _sha256(representative_figure_path),
            },
        },
        "promotion_order": [
            "results_pdf_planned_matrix_generated",
            "results_pdf_latex_and_digest_verified",
            "response_normalized_and_pair_matched_states_verified",
            "manuscript_table_generated",
            "manuscript_built_and_changed_pages_inspected",
        ],
        "manuscript": {
            "source": {
                "path": str(manuscript_source_path.relative_to(workspace_root)),
                "sha256": _sha256(manuscript_source_path),
            },
            "pdf": {
                "path": str(manuscript_pdf_path.relative_to(workspace_root)),
                "sha256": _sha256(manuscript_pdf_path),
                "page_count": page_count,
            },
            "render_check": {
                "status": "PASS_CHANGED_PAGES_INSPECTED",
                "pages": pages,
                "finding": str(render_finding).strip(),
            },
        },
    }
    output_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_receipt_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_receipt_path)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--table-receipt", required=True, type=Path)
    parser.add_argument("--report-manifest", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--manuscript-source", required=True, type=Path)
    parser.add_argument("--manuscript-pdf", required=True, type=Path)
    parser.add_argument("--output-receipt", required=True, type=Path)
    parser.add_argument("--representative-figure", required=True, type=Path)
    parser.add_argument("--rendered-page", action="append", type=int, default=[])
    parser.add_argument("--render-finding", required=True)
    parser.add_argument("--workspace-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    receipt = bind_manuscript_promotion_receipt(
        table_receipt_path=args.table_receipt,
        report_manifest_path=args.report_manifest,
        ledger_path=args.ledger,
        manuscript_source_path=args.manuscript_source,
        manuscript_pdf_path=args.manuscript_pdf,
        output_receipt_path=args.output_receipt,
        representative_figure_path=args.representative_figure,
        rendered_pages=args.rendered_page,
        render_finding=args.render_finding,
        workspace_root=args.workspace_root,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["bind_manuscript_promotion_receipt", "main"]
