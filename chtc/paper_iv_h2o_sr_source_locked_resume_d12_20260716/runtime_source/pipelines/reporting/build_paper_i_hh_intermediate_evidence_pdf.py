#!/usr/bin/env python3
"""Build the Paper-I HH SNAKE intermediate evidence PDF.

This report is deliberately separate from manuscript/source-map promotion.  It
summarizes completed and live-status Hubbard-Holstein SNAKE evidence with the
same-cutoff error as the primary completed-row metric.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_JSON = (
    REPO_ROOT
    / "MATH/paper_facing/paper_I_static_scaffold/"
    / "paper_i_hh_snake_intermediate_evidence_pass1_sources_20260613.json"
)
DEFAULT_OUTPUT_PDF = REPO_ROOT / "output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.pdf"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "output/pdf/paper_i_hh_snake_intermediate_evidence_pass1_20260613.csv"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path | str | None) -> str:
    if path in (None, ""):
        return ""
    try:
        p = Path(str(path))
        if p.is_absolute():
            return str(p.relative_to(REPO_ROOT))
        return str(p)
    except Exception:
        return str(path)


def _fmt_sci(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "--"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    return f"{x:.{digits}e}"


def _fmt_plain(value: Any) -> str:
    if value in (None, ""):
        return "--"
    return str(value)


def _safe(text: Any) -> str:
    return html.escape(str(text), quote=False)


def _p(text: Any, style: ParagraphStyle) -> Paragraph:
    return Paragraph(_safe(text), style)


def _path_p(text: Any, style: ParagraphStyle) -> Paragraph:
    safe = _safe(_rel(text))
    safe = safe.replace("/", "/<br/>").replace("_", "_<br/>")
    return Paragraph(safe, style)


def _table_style(header_rows: int = 1, font_size: float = 7.0) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#E9EDF3")),
            ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.HexColor("#1F2937")),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("FONTNAME", (0, header_rows), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("LEADING", (0, 0), (-1, -1), font_size + 1.7),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#BAC4D0")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
        ]
    )


def _footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(0.55 * inch, 0.32 * inch, "Hubbard-Holstein SNAKE results snapshot")
    canvas.drawRightString(10.45 * inch, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _validate_source(source_path: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    rows = source.get("rows")
    selections = source.get("display_selections")
    if source.get("schema") != "paper_i_hh_snake_intermediate_evidence_sources_v1":
        errors.append("unexpected source schema")
    if not isinstance(rows, list) or not rows:
        errors.append("source rows missing or empty")
        rows = []
    if not isinstance(selections, list) or not selections:
        errors.append("display selections missing or empty")
        selections = []

    by_id = {str(row.get("row_id")): row for row in rows if isinstance(row, Mapping)}
    if len(by_id) != len(rows):
        errors.append("duplicate or missing row_id in source rows")

    for selection in selections:
        row_id = str((selection or {}).get("selected_row_id"))
        if row_id not in by_id:
            errors.append(f"display selection references missing row_id {row_id}")

    for row in rows:
        if not isinstance(row, Mapping):
            continue
        row_id = str(row.get("row_id"))
        sector = row.get("sector")
        status = str(row.get("evidence_status"))
        metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
        if sector == "table_iii_hh" and str(row.get("regime", "")).startswith("u8"):
            errors.append(f"{row_id} mixes U/t=8 row into table_iii_hh")
        if status == "completed_local_result_json":
            src = row.get("source_json")
            expected_sha = row.get("source_sha256")
            if not src:
                errors.append(f"{row_id} completed row lacks source_json")
                continue
            src_path = (REPO_ROOT / str(src)).resolve()
            if not src_path.exists():
                errors.append(f"{row_id} source_json missing: {src}")
                continue
            actual_sha = _sha256(src_path)
            if actual_sha != expected_sha:
                errors.append(f"{row_id} sha mismatch: expected {expected_sha}, got {actual_sha}")
            if metrics.get("same_cutoff_abs_delta_e") is None:
                errors.append(f"{row_id} completed row lacks same_cutoff_abs_delta_e")
        elif "live" in status:
            if metrics.get("same_cutoff_abs_delta_e") is not None:
                warnings.append(f"{row_id} live row has same_cutoff_abs_delta_e set; check status contract")
        else:
            warnings.append(f"{row_id} has unrecognized evidence_status {status}")

    source_map = ((source.get("table_context") or {}).get("source_map")) if isinstance(source.get("table_context"), Mapping) else None
    source_map_sha = ((source.get("table_context") or {}).get("source_map_sha256")) if isinstance(source.get("table_context"), Mapping) else None
    if source_map:
        p = REPO_ROOT / str(source_map)
        if p.exists():
            actual = _sha256(p)
            if source_map_sha and actual != source_map_sha:
                errors.append(f"source_map sha mismatch: expected {source_map_sha}, got {actual}")
        else:
            warnings.append(f"source_map path not found: {source_map}")

    return {
        "source_json": _rel(source_path),
        "source_sha256": _sha256(source_path),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }


def _selected_rows(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = {str(row["row_id"]): dict(row) for row in source["rows"]}
    selected: list[dict[str, Any]] = []
    for selection in source["display_selections"]:
        row = dict(rows[str(selection["selected_row_id"])])
        row["selection_status"] = selection.get("selection_status")
        row["tied_row_ids"] = selection.get("tied_row_ids", [])
        selected.append(row)
    return selected


def _aggregate(source_path: Path, source: Mapping[str, Any], validation: Mapping[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in source.get("rows", [])]
    selected = _selected_rows(source)
    row_counts = Counter(str(row.get("evidence_status")) for row in rows)
    sector_counts = Counter(str(row.get("sector")) for row in rows)
    return {
        "schema": "paper_i_hh_snake_intermediate_evidence_report_v1",
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_json": _rel(source_path),
        "source_sha256": validation["source_sha256"],
        "artifact_role": source.get("artifact_role"),
        "primary_error_metric": source.get("primary_error_metric"),
        "primary_error_formula": source.get("primary_error_formula"),
        "live_status_policy": source.get("live_status_policy"),
        "promotion_decision": source.get("promotion_decision"),
        "manuscript_edit_policy": source.get("manuscript_edit_policy"),
        "canonical_source_map_edit_policy": source.get("canonical_source_map_edit_policy"),
        "table_context": source.get("table_context", {}),
        "row_counts_by_status": dict(row_counts),
        "row_counts_by_sector": dict(sector_counts),
        "selected_rows": selected,
        "rows": rows,
        "caveats": list(source.get("caveats", [])),
        "validation": dict(validation),
    }


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sector",
        "regime",
        "method",
        "variant",
        "evidence_status",
        "same_cutoff_abs_delta_e",
        "ansatz_depth",
        "trial",
        "target_hit_success",
        "visible_same_cutoff_plateau_abs_delta_e",
        "comparison_direction",
        "source_json",
        "source_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in report["rows"]:
            metrics = row.get("metrics") or {}
            visible = row.get("visible_baseline") or {}
            comparison = row.get("comparison_to_visible") or {}
            writer.writerow(
                {
                    "sector": row.get("sector"),
                    "regime": row.get("regime"),
                    "method": row.get("method"),
                    "variant": row.get("variant"),
                    "evidence_status": row.get("evidence_status"),
                    "same_cutoff_abs_delta_e": metrics.get("same_cutoff_abs_delta_e"),
                    "ansatz_depth": metrics.get("ansatz_depth"),
                    "trial": metrics.get("trial"),
                    "target_hit_success": metrics.get("target_hit_success"),
                    "visible_same_cutoff_plateau_abs_delta_e": visible.get("visible_same_cutoff_plateau_abs_delta_e"),
                    "comparison_direction": comparison.get("direction"),
                    "source_json": row.get("source_json"),
                    "source_sha256": row.get("source_sha256"),
                }
            )


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    base = styles["BodyText"]
    return {
        "title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            spaceAfter=10,
            textColor=colors.HexColor("#0F172A"),
        ),
        "h1": ParagraphStyle(
            "Heading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor("#172554"),
        ),
        "body": ParagraphStyle("Body", parent=base, fontSize=8.7, leading=11.5, alignment=TA_LEFT),
        "small": ParagraphStyle("Small", parent=base, fontSize=7.0, leading=8.6, alignment=TA_LEFT),
        "tiny": ParagraphStyle("Tiny", parent=base, fontSize=5.8, leading=7.0, alignment=TA_LEFT),
    }


def _manifest_table(report: Mapping[str, Any], s: Mapping[str, ParagraphStyle]) -> Table:
    rows = [
        ["Field", "Value"],
        ["Generated UTC", report["generated_utc"]],
        ["Source JSON", report["source_json"]],
        ["Source SHA256", report["source_sha256"]],
        ["Primary error", f"{report['primary_error_metric']} = {report['primary_error_formula']}"],
        ["Role", report["artifact_role"]],
        ["Promotion decision", report["promotion_decision"]],
        ["Manuscript edit policy", report["manuscript_edit_policy"]],
        ["Canonical source-map edit policy", report["canonical_source_map_edit_policy"]],
        ["Validation", "passed" if report["validation"]["passed"] else "FAILED"],
    ]
    table = Table(
        [[_p(c, s["small"]) for c in row] for row in rows],
        colWidths=[1.9 * inch, 7.6 * inch],
        repeatRows=1,
    )
    table.setStyle(_table_style(font_size=7.0))
    return table


def _find_selected(rows: Sequence[Mapping[str, Any]], regime: str) -> Mapping[str, Any]:
    for row in rows:
        if row.get("regime") == regime:
            return row
    raise KeyError(regime)


def _nph_label(row: Mapping[str, Any]) -> str:
    params = row.get("parameters") or {}
    work = params.get("n_ph_work")
    if work is None:
        return "--"
    return f"{work}"


def _current_cell(row: Mapping[str, Any]) -> str:
    visible = row.get("visible_baseline") or {}
    value = visible.get("visible_same_cutoff_plateau_abs_delta_e")
    k = visible.get("visible_k_pl")
    if value is None:
        return "--"
    return f"k={k}; |Delta E|={_fmt_sci(value)}"


def _new_cell(row: Mapping[str, Any]) -> str:
    status = str(row.get("evidence_status", ""))
    metrics = row.get("metrics") or {}
    if status == "completed_local_result_json":
        k = metrics.get("ansatz_depth")
        value = metrics.get("same_cutoff_abs_delta_e")
        state = "completed"
        tied = row.get("tied_row_ids") or []
        if tied:
            state = "completed; exponent=flat"
        return f"k={k}; |Delta E|={_fmt_sci(value)}; {state}"
    if "live" in status:
        return "running; same-cutoff |Delta E| pending"
    return status.replace("_", " ")


def _direction_cell(row: Mapping[str, Any]) -> str:
    direction = str((row.get("comparison_to_visible") or {}).get("direction") or "")
    if direction == "lower_than_visible_on_same_cutoff_metric":
        return "lower than current Paper I"
    if direction == "higher_than_visible_on_same_cutoff_metric":
        return "higher than current Paper I"
    if "live" in direction or "not_compared" in direction:
        return "pending"
    if "separate_new_sector" in direction:
        return "new sector"
    return "--"


def _paper_i_format_table(selected: Sequence[Mapping[str, Any]], s: Mapping[str, ParagraphStyle]) -> Table:
    regimes = ["weak_weak", "strong_weak", "weak_strong", "strong_strong"]
    labels = {
        "weak_weak": "weak-weak\n(U/t, lambda)=(0.25,0.25)",
        "strong_weak": "strong-weak\n(U/t, lambda)=(1.25,0.25)",
        "weak_strong": "weak-strong\n(U/t, lambda)=(0.25,1.25)",
        "strong_strong": "strong-strong\n(U/t, lambda)=(1.25,1.25)",
    }
    by_regime = {regime: _find_selected(selected, regime) for regime in regimes}
    data = [
        [_p("SNAKE row", s["small"])] + [_p(labels[regime], s["small"]) for regime in regimes],
        [_p("phonon cutoff used for |Delta E|", s["small"])]
        + [_p(_nph_label(by_regime[regime]), s["small"]) for regime in regimes],
        [_p("current Paper I", s["small"])]
        + [_p(_current_cell(by_regime[regime]), s["small"]) for regime in regimes],
        [_p("new CHTC batch", s["small"])]
        + [_p(_new_cell(by_regime[regime]), s["small"]) for regime in regimes],
        [_p("same-cutoff comparison", s["small"])]
        + [_p(_direction_cell(by_regime[regime]), s["small"]) for regime in regimes],
    ]
    table = Table(data, colWidths=[1.65 * inch, 2.05 * inch, 2.05 * inch, 2.05 * inch, 2.05 * inch], repeatRows=1)
    table.setStyle(_table_style(font_size=7.4))
    return table


def _u8_table(selected: Sequence[Mapping[str, Any]], s: Mapping[str, ParagraphStyle]) -> Table:
    rows = [row for row in selected if row.get("sector") == "u8_diagnostic_hh"]
    data = [
        [_p("U/t=8 sector", s["small"]), _p("phonon cutoff", s["small"]), _p("SNAKE result/status", s["small"])],
    ]
    for row in rows:
        data.append(
            [
                _p(str(row.get("regime_label", row.get("regime"))).replace(" (U/t, lambda)=", "\n(U/t, lambda)="), s["small"]),
                _p(_nph_label(row), s["small"]),
                _p(_new_cell(row), s["small"]),
            ]
        )
    table = Table(data, colWidths=[3.0 * inch, 1.5 * inch, 5.3 * inch], repeatRows=1)
    table.setStyle(_table_style(font_size=7.4))
    return table


def _definition_table(report: Mapping[str, Any], s: Mapping[str, ParagraphStyle]) -> Table:
    rows = [
        ["Quantity", "Meaning"],
        ["same-cutoff |Delta E|", "|E_alg(n_ph) - E_ED(n_ph)|. This is the Paper-I HH Table-III error."],
    ]
    table = Table([[_p(c, s["small"]) for c in row] for row in rows], colWidths=[2.0 * inch, 7.8 * inch], repeatRows=1)
    table.setStyle(_table_style(font_size=7.4))
    return table


def _selected_table(rows: Sequence[Mapping[str, Any]], title: str, s: Mapping[str, ParagraphStyle]) -> list[Any]:
    data: list[list[Any]] = [
        [
            _p("Regime", s["small"]),
            _p("Status", s["small"]),
            _p("same-cutoff deltaE", s["small"]),
            _p("Depth", s["small"]),
            _p("Trial", s["small"]),
            _p("Visible same-cutoff", s["small"]),
            _p("Comparison note", s["small"]),
        ]
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        visible = row.get("visible_baseline") or {}
        comparison = row.get("comparison_to_visible") or {}
        status = str(row.get("evidence_status", ""))
        if "live" in status:
            status_note = "live status; same-cutoff pending"
        else:
            status_note = status.replace("_", " ")
        data.append(
            [
                _p(row.get("regime_label", row.get("regime")), s["small"]),
                _p(status_note, s["small"]),
                _p(_fmt_sci(metrics.get("same_cutoff_abs_delta_e")), s["small"]),
                _p(_fmt_plain(metrics.get("ansatz_depth")), s["small"]),
                _p(_fmt_plain(metrics.get("trial")), s["small"]),
                _p(_fmt_sci(visible.get("visible_same_cutoff_plateau_abs_delta_e")), s["small"]),
                _p(str(comparison.get("direction", "--")).replace("_", " "), s["small"]),
            ]
        )
    table = Table(
        data,
        colWidths=[1.75 * inch, 1.25 * inch, 1.05 * inch, 0.55 * inch, 0.5 * inch, 1.1 * inch, 2.55 * inch],
        repeatRows=1,
    )
    table.setStyle(_table_style(font_size=6.7))
    return [_p(title, s["h1"]), table]


def _source_audit_table(rows: Sequence[Mapping[str, Any]], s: Mapping[str, ParagraphStyle]) -> Table:
    data: list[list[Any]] = [
        [
            _p("Row", s["tiny"]),
            _p("Sector", s["tiny"]),
            _p("Variant", s["tiny"]),
            _p("Status", s["tiny"]),
            _p("same-cutoff", s["tiny"]),
            _p("Depth/trial", s["tiny"]),
            _p("Source ref", s["tiny"]),
            _p("SHA256", s["tiny"]),
        ]
    ]
    for row in rows:
        metrics = row.get("metrics") or {}
        data.append(
            [
                _p(row.get("row_id"), s["tiny"]),
                _p(row.get("sector"), s["tiny"]),
                _p(row.get("variant"), s["tiny"]),
                _p(str(row.get("evidence_status")).replace("_", " "), s["tiny"]),
                _p(_fmt_sci(metrics.get("same_cutoff_abs_delta_e")), s["tiny"]),
                _p(f"{_fmt_plain(metrics.get('ansatz_depth'))}/{_fmt_plain(metrics.get('trial'))}", s["tiny"]),
                _p(row.get("source_kind") or "full path in sidecar", s["tiny"]),
                _p(str(row.get("source_sha256") or "--")[:14], s["tiny"]),
            ]
        )
    table = Table(
        data,
        colWidths=[1.9 * inch, 0.95 * inch, 0.95 * inch, 1.2 * inch, 0.9 * inch, 0.7 * inch, 1.75 * inch, 0.85 * inch],
        repeatRows=1,
    )
    table.setStyle(_table_style(font_size=5.7))
    return table


def _build_pdf(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    s = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.55 * inch,
    )
    story: list[Any] = []
    story.append(_p("Hubbard-Holstein SNAKE Results Snapshot", s["title"]))
    story.append(Spacer(1, 0.08 * inch))
    selected = report["selected_rows"]
    story.append(_definition_table(report, s))
    story.append(Spacer(1, 0.12 * inch))
    story.append(_p("Current Paper-I Table-III sectors", s["h1"]))
    story.append(_paper_i_format_table(selected, s))
    story.append(Spacer(1, 0.16 * inch))
    story.append(_p("Additional U/t=8 sector", s["h1"]))
    story.append(_u8_table(selected, s))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def build(source_json: Path, output_pdf: Path, output_json: Path, output_csv: Path) -> dict[str, Any]:
    source_json = source_json.resolve()
    source = _load_json(source_json)
    validation = _validate_source(source_json, source)
    if not validation["passed"]:
        raise SystemExit("source validation failed: " + "; ".join(validation["errors"]))
    report = _aggregate(source_json, source, validation)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_csv, report)
    _build_pdf(output_pdf, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-json", type=Path, default=DEFAULT_SOURCE_JSON)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build(args.source_json, args.output_pdf, args.output_json, args.output_csv)
    print(json.dumps({
        "pdf": _rel(args.output_pdf),
        "json": _rel(args.output_json),
        "csv": _rel(args.output_csv),
        "source_sha256": report["source_sha256"],
        "selected_rows": len(report["selected_rows"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
