#!/usr/bin/env python3
"""Build a compact PDF comparing Paper-I repeat-policy comparator suites."""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE_A = REPO_ROOT / "raw_outputs" / "paper_i_tables_i_ii_no_repeat_comparator_capmatch_20260610_v1" / "suite_A_validation_summary.json"
DEFAULT_SUITE_B = REPO_ROOT / "raw_outputs" / "paper_i_tables_i_ii_repeat_enabled_comparator_capmatch_20260610_v1" / "suite_B_validation_summary.json"
DEFAULT_OUTPUT_PDF = REPO_ROOT / "output" / "pdf" / "paper_i_hubbard_spinboson_repeat_policy_suite_comparison_20260610.pdf"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "output" / "pdf" / "paper_i_hubbard_spinboson_repeat_policy_suite_comparison_20260610.json"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "output" / "pdf" / "paper_i_hubbard_spinboson_repeat_policy_suite_comparison_20260610.csv"
TARGET_LINE = 2.0e-4

CASE_LABELS = {
    "hubbard_L2_three_model_weak": "Hubbard weak",
    "hubbard_L2_three_model_strong": "Hubbard strong",
    "spin_boson_L2_nph1_three_model_weak": "Spin-boson weak",
    "spin_boson_L2_nph2_three_model_strong": "Spin-boson strong",
}
CASE_ORDER = tuple(CASE_LABELS)
METHOD_LABELS = {
    "append": "Append",
    "tetris": "TETRIS",
    "geo": "Geo",
    "qubit_qeb": "Qubit/QEB",
}
METHOD_ORDER = ("append", "tetris", "geo", "qubit_qeb")
SUITE_LABELS = {
    "suite_A_no_repeat": "A no-repeat",
    "suite_B_repeat_enabled": "B repeat-enabled",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_sci(value: Any, digits: int = 3) -> str:
    if value in (None, ""):
        return "--"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "--"
    if x == 0.0:
        return "0"
    return f"{x:.{digits}e}"


def _fmt_int(value: Any) -> str:
    if value in (None, ""):
        return "--"
    try:
        x = float(value)
    except Exception:
        return str(value)
    if not math.isfinite(x):
        return "--"
    return str(int(round(x)))


def _read_result_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(row.get("result_json", "")))
    payload: dict[str, Any] = {}
    result: dict[str, Any] = {}
    if path.exists():
        payload = _load_json(path)
        raw = payload.get("result")
        result = raw if isinstance(raw, dict) else {}
    out = dict(row)
    out.update(
        {
            "result_json_exists": path.exists(),
            "top_generic_adapt_stop_policy": payload.get("generic_adapt_stop_policy"),
            "metadata_generic_adapt_stop_policy": (payload.get("metadata") or {}).get("generic_adapt_stop_policy")
            if isinstance(payload.get("metadata"), dict)
            else None,
            "abs_delta_e": result.get("abs_delta_e", row.get("abs_delta_e")),
            "abs_delta_e_same_cutoff": result.get("abs_delta_e_same_cutoff", row.get("abs_delta_e_same_cutoff")),
            "adapt_depth_reached": result.get("adapt_depth_reached", row.get("adapt_depth_reached")),
            "selected_operator_count": result.get("selected_operator_count", row.get("selected_operator_count")),
            "selected_unique_operator_count": result.get("selected_unique_operator_count", row.get("selected_unique_operator_count")),
            "S_alg": result.get("S_alg", row.get("S_alg")),
            "shots_total": result.get("shots_total", row.get("shots_total")),
            "compiled_count_2q_total": result.get("compiled_count_2q_total", row.get("compiled_count_2q_total")),
            "compiled_depth_2q_total": result.get("compiled_depth_2q_total"),
            "compiled_depth_total": result.get("compiled_depth_total", row.get("compiled_depth_total")),
            "compiled_resource_qiskit_validated": result.get("compiled_resource_qiskit_validated"),
            "compiled_resource_source_kind": result.get("compiled_resource_source_kind"),
            "benchmark_energy_stop_target": result.get("benchmark_energy_stop_target"),
        }
    )
    return out


def _load_suite(path: Path) -> dict[str, Any]:
    summary = _load_json(path)
    rows = [_read_result_fields(row) for row in summary.get("rows", [])]
    return {**summary, "rows": rows, "summary_path": str(path), "summary_sha256": _sha256(path)}


def _float_or_inf(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.inf
    return x if math.isfinite(x) else math.inf


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("case_id")), str(row.get("method_key"))


def _aggregate(suite_a: dict[str, Any], suite_b: dict[str, Any]) -> dict[str, Any]:
    by_suite: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for suite in (suite_a, suite_b):
        suite_id = str(suite.get("suite_id"))
        by_suite[suite_id] = {_row_key(row): row for row in suite.get("rows", [])}
    comparisons: list[dict[str, Any]] = []
    for case_id in CASE_ORDER:
        for method_key in METHOD_ORDER:
            key = (case_id, method_key)
            a = by_suite.get("suite_A_no_repeat", {}).get(key)
            b = by_suite.get("suite_B_repeat_enabled", {}).get(key)
            err_a = _float_or_inf((a or {}).get("abs_delta_e_same_cutoff", (a or {}).get("abs_delta_e")))
            err_b = _float_or_inf((b or {}).get("abs_delta_e_same_cutoff", (b or {}).get("abs_delta_e")))
            s_a = _float_or_inf((a or {}).get("S_alg"))
            s_b = _float_or_inf((b or {}).get("S_alg"))
            if math.isinf(err_a) and math.isinf(err_b):
                error_winner = "none"
            elif abs(err_a - err_b) <= max(1e-12, 1e-3 * min(abs(err_a), abs(err_b), 1.0)):
                error_winner = "tie"
            else:
                error_winner = "A" if err_a < err_b else "B"
            if math.isinf(s_a) and math.isinf(s_b):
                cost_winner = "none"
            elif s_a == s_b:
                cost_winner = "tie"
            else:
                cost_winner = "A" if s_a < s_b else "B"
            comparisons.append(
                {
                    "case_id": case_id,
                    "case_label": CASE_LABELS[case_id],
                    "method_key": method_key,
                    "method_label": METHOD_LABELS[method_key],
                    "suite_A": a,
                    "suite_B": b,
                    "error_winner": error_winner,
                    "cost_winner": cost_winner,
                    "suite_A_hits_target_line": bool(err_a <= TARGET_LINE),
                    "suite_B_hits_target_line": bool(err_b <= TARGET_LINE),
                }
            )
    return {
        "schema": "paper_i_repeat_policy_suite_comparison_v1",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "target_line": TARGET_LINE,
        "suite_summaries": [
            {
                "suite_id": suite_a.get("suite_id"),
                "summary_path": suite_a.get("summary_path"),
                "summary_sha256": suite_a.get("summary_sha256"),
                "all_checks_pass": suite_a.get("all_checks_pass"),
                "record_count": suite_a.get("record_count"),
            },
            {
                "suite_id": suite_b.get("suite_id"),
                "summary_path": suite_b.get("summary_path"),
                "summary_sha256": suite_b.get("summary_sha256"),
                "all_checks_pass": suite_b.get("all_checks_pass"),
                "record_count": suite_b.get("record_count"),
            },
        ],
        "comparisons": comparisons,
    }


def _write_csv(path: Path, comparison: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "case_id",
        "method_key",
        "suite",
        "abs_delta_e_same_cutoff",
        "hit_2e_4",
        "S_alg",
        "compiled_count_2q_total",
        "compiled_depth_2q_total",
        "compiled_depth_total",
        "adapt_depth_reached",
        "selected_operator_count",
        "selected_unique_operator_count",
        "adapt_stop_reason",
        "adapt_allow_repeats_override",
        "geo_replacement_policy",
        "result_json",
        "result_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for comp in comparison["comparisons"]:
            for suite_key, suite_label in (("suite_A", "suite_A_no_repeat"), ("suite_B", "suite_B_repeat_enabled")):
                row = comp.get(suite_key) or {}
                err = row.get("abs_delta_e_same_cutoff", row.get("abs_delta_e"))
                writer.writerow(
                    {
                        "case_id": comp["case_id"],
                        "method_key": comp["method_key"],
                        "suite": suite_label,
                        "abs_delta_e_same_cutoff": err,
                        "hit_2e_4": _float_or_inf(err) <= TARGET_LINE,
                        "S_alg": row.get("S_alg"),
                        "compiled_count_2q_total": row.get("compiled_count_2q_total"),
                        "compiled_depth_2q_total": row.get("compiled_depth_2q_total"),
                        "compiled_depth_total": row.get("compiled_depth_total"),
                        "adapt_depth_reached": row.get("adapt_depth_reached"),
                        "selected_operator_count": row.get("selected_operator_count"),
                        "selected_unique_operator_count": row.get("selected_unique_operator_count"),
                        "adapt_stop_reason": row.get("adapt_stop_reason"),
                        "adapt_allow_repeats_override": row.get("adapt_allow_repeats_override"),
                        "geo_replacement_policy": row.get("geo_replacement_policy"),
                        "result_json": row.get("result_json"),
                        "result_sha256": row.get("result_sha256"),
                    }
                )


def _short_path(value: Any) -> str:
    text = str(value or "")
    try:
        path = Path(text)
        if path.is_absolute():
            return str(path.relative_to(REPO_ROOT))
    except Exception:
        pass
    return text


def _p(text: str, style: ParagraphStyle) -> Paragraph:
    # ReportLab paragraph parser treats bare ampersands as entities.
    return Paragraph(str(text).replace("&", "&amp;"), style)


def _path_p(text: str, style: ParagraphStyle) -> Paragraph:
    safe = str(text).replace("&", "&amp;")
    safe = safe.replace("/", "/<br/>").replace("_", "_<br/>")
    return Paragraph(safe, style)


def _table_style(header_rows: int = 1, font_size: float = 6.7) -> TableStyle:
    return TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#1f2937")),
            ("TEXTCOLOR", (0, 0), (-1, header_rows - 1), colors.white),
            ("FONTNAME", (0, 0), (-1, header_rows - 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), font_size),
            ("FONTNAME", (0, header_rows), (-1, -1), "Helvetica"),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ALIGN", (0, 0), (0, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
            ("ROWBACKGROUNDS", (0, header_rows), (-1, -1), [colors.white, colors.HexColor("#f9fafb")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3),
            ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]
    )


def _page_footer(canvas: Any, doc: Any) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#6b7280"))
    canvas.drawString(0.55 * inch, 0.35 * inch, "Paper I repeat-policy comparator suite comparison")
    canvas.drawRightString(10.45 * inch, 0.35 * inch, f"Page {doc.page}")
    canvas.restoreState()


def _build_pdf(path: Path, comparison: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title2", parent=styles["Title"], fontSize=17, leading=20, spaceAfter=8)
    h1 = ParagraphStyle("H1x", parent=styles["Heading1"], fontSize=11, leading=13, spaceBefore=8, spaceAfter=5)
    h2 = ParagraphStyle("H2x", parent=styles["Heading2"], fontSize=9.5, leading=11, spaceBefore=6, spaceAfter=4)
    body = ParagraphStyle("BodyX", parent=styles["BodyText"], fontSize=8, leading=10, alignment=TA_LEFT)
    small = ParagraphStyle("SmallX", parent=styles["BodyText"], fontSize=6.8, leading=8.2, alignment=TA_LEFT)

    doc = SimpleDocTemplate(
        str(path),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.42 * inch,
        bottomMargin=0.55 * inch,
    )
    story: list[Any] = []
    story.append(_p("Paper I Hubbard / Spin-Boson Repeat-Policy Suite Comparison", title))
    story.append(
        _p(
            "This evidence PDF compares Suite A no-repeat/native comparator rows against Suite B repeat-enabled diagnostic rows. "
            "It reports observed same-cutoff errors and costs only; it does not make a promotion decision.",
            body,
        )
    )
    story.append(Spacer(1, 0.08 * inch))

    manifest_rows = [["Field", "Value"]]
    manifest_rows.extend(
        [
            ["Generated UTC", comparison["generated_utc"]],
            ["Target line", _fmt_sci(comparison["target_line"])],
            ["Suite A", "no-repeat/native: selected_labels_excluded; Geo forced no-repeat"],
            ["Suite B", "repeat-enabled diagnostic: append_new_occurrence; block immediate repeat only"],
            ["Stop policy", "fixed_horizon_no_target_v1; energy_stop_target blank; first_hit_thresholds reporting-only"],
            ["Metric", "abs_delta_e_same_cutoff; spin-boson higher-cutoff references retained only in sidecar CSV/JSON"],
        ]
    )
    t = Table(manifest_rows, colWidths=[1.55 * inch, 8.0 * inch], repeatRows=1)
    t.setStyle(_table_style())
    story.append(t)
    story.append(Spacer(1, 0.1 * inch))

    source_rows = [["Suite", "Checks", "Records", "Summary path", "Summary SHA256"]]
    for s in comparison["suite_summaries"]:
        source_rows.append(
            [
                _p(str(s["suite_id"]), small),
                _p("pass" if s.get("all_checks_pass") else "FAIL", small),
                _p(str(s.get("record_count")), small),
                _p(f"{Path(str(s.get('summary_path'))).parent.name}/suite validation", small),
                _p(str(s.get("summary_sha256"))[:16] + "...", small),
            ]
        )
    t = Table(source_rows, colWidths=[1.3 * inch, 0.65 * inch, 0.5 * inch, 5.85 * inch, 1.25 * inch], repeatRows=1)
    t.setStyle(_table_style(font_size=6.1))
    story.append(t)
    story.append(Spacer(1, 0.12 * inch))

    story.append(_p("Executive scoreboard", h1))
    score_rows = [["Case", "Method", "Err A", "Err B", "Hit A/B", "S_alg A", "S_alg B", "Error winner", "Cost winner"]]
    for comp in comparison["comparisons"]:
        a = comp.get("suite_A") or {}
        b = comp.get("suite_B") or {}
        err_a = a.get("abs_delta_e_same_cutoff", a.get("abs_delta_e"))
        err_b = b.get("abs_delta_e_same_cutoff", b.get("abs_delta_e"))
        score_rows.append(
            [
                comp["case_label"],
                comp["method_label"],
                _fmt_sci(err_a),
                _fmt_sci(err_b),
                f"{'Y' if comp['suite_A_hits_target_line'] else 'N'}/{'Y' if comp['suite_B_hits_target_line'] else 'N'}",
                _fmt_int(a.get("S_alg")),
                _fmt_int(b.get("S_alg")),
                comp["error_winner"],
                comp["cost_winner"],
            ]
        )
    t = Table(
        score_rows,
        colWidths=[1.35 * inch, 0.85 * inch, 0.82 * inch, 0.82 * inch, 0.55 * inch, 0.75 * inch, 0.75 * inch, 0.85 * inch, 0.8 * inch],
        repeatRows=1,
    )
    t.setStyle(_table_style())
    story.append(t)
    story.append(PageBreak())

    for case_id in CASE_ORDER:
        story.append(_p(CASE_LABELS[case_id], h1))
        story.append(
            _p(
                "Costs are reported as S_alg / N2q / D2q / Dcirc. D2q is the compiled two-qubit depth. "
                "A = no-repeat/native; B = repeat-enabled diagnostic.",
                small,
            )
        )
        rows = [["Method", "Suite", "Err", "Hit", "Depth", "Ops", "Unique", "S_alg", "N2q", "D2q", "Dcirc", "Stop"]]
        for method_key in METHOD_ORDER:
            comp = next(c for c in comparison["comparisons"] if c["case_id"] == case_id and c["method_key"] == method_key)
            for suite_key, suite_name in (("suite_A", "A no-repeat"), ("suite_B", "B repeat")):
                row = comp.get(suite_key) or {}
                err = row.get("abs_delta_e_same_cutoff", row.get("abs_delta_e"))
                rows.append(
                    [
                        comp["method_label"],
                        suite_name,
                        _fmt_sci(err),
                        "Y" if _float_or_inf(err) <= TARGET_LINE else "N",
                        _fmt_int(row.get("adapt_depth_reached")),
                        _fmt_int(row.get("selected_operator_count")),
                        _fmt_int(row.get("selected_unique_operator_count")),
                        _fmt_int(row.get("S_alg")),
                        _fmt_int(row.get("compiled_count_2q_total")),
                        _fmt_int(row.get("compiled_depth_2q_total")),
                        _fmt_int(row.get("compiled_depth_total")),
                        str(row.get("adapt_stop_reason") or "--"),
                    ]
                )
        t = Table(
            rows,
            colWidths=[0.8 * inch, 0.82 * inch, 0.72 * inch, 0.35 * inch, 0.45 * inch, 0.45 * inch, 0.48 * inch, 0.65 * inch, 0.45 * inch, 0.45 * inch, 0.5 * inch, 1.55 * inch],
            repeatRows=1,
        )
        t.setStyle(_table_style())
        story.append(t)
        story.append(Spacer(1, 0.09 * inch))
        notes: list[str] = []
        for comp in [c for c in comparison["comparisons"] if c["case_id"] == case_id]:
            a = comp.get("suite_A") or {}
            b = comp.get("suite_B") or {}
            if comp["suite_A_hits_target_line"] != comp["suite_B_hits_target_line"]:
                notes.append(
                    f"{comp['method_label']}: target-line status differs (A={'hit' if comp['suite_A_hits_target_line'] else 'miss'}, "
                    f"B={'hit' if comp['suite_B_hits_target_line'] else 'miss'})."
                )
            if _float_or_inf(b.get("S_alg")) > 5 * max(_float_or_inf(a.get("S_alg")), 1.0):
                notes.append(f"{comp['method_label']}: repeat-enabled S_alg is >5x no-repeat S_alg.")
        if notes:
            story.append(_p("Notes: " + " ".join(notes), small))
        story.append(Spacer(1, 0.12 * inch))
        if case_id != CASE_ORDER[-1]:
            story.append(PageBreak())

    story.append(PageBreak())
    story.append(_p("Provenance and validation notes", h1))
    story.append(
        _p(
            "Every row in both suites passed validation: fixed-horizon/no-target policy at top and row levels, expected repeat override, expected replacement policy, null benchmark_energy_stop_target, and no benchmark_abs_delta_e_target stop. Full row-level paths and SHA256 hashes are in the sidecar CSV/JSON next to this PDF.",
            body,
        )
    )
    provenance_rows = [["Suite", "Output root", "Validation summary"]]
    roots = defaultdict(set)
    for comp in comparison["comparisons"]:
        for suite_key in ("suite_A", "suite_B"):
            row = comp.get(suite_key) or {}
            result_path = row.get("result_json")
            if result_path:
                roots[suite_key].add(str(Path(str(result_path)).parents[2]))
    for suite_key, label in (("suite_A", "A no-repeat"), ("suite_B", "B repeat-enabled")):
        summary = comparison["suite_summaries"][0 if suite_key == "suite_A" else 1]
        root_names = ", ".join(Path(item).name for item in sorted(roots[suite_key]))
        summary_path = Path(str(summary["summary_path"]))
        provenance_rows.append([
            _p(label, small),
            _p(root_names, small),
            _p(summary_path.name, small),
        ])
    t = Table(provenance_rows, colWidths=[1.05 * inch, 4.65 * inch, 4.25 * inch], repeatRows=1)
    t.setStyle(_table_style(font_size=6.1))
    story.append(t)
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


def build(*, suite_a_path: Path, suite_b_path: Path, output_pdf: Path, output_json: Path, output_csv: Path) -> dict[str, Any]:
    suite_a = _load_suite(suite_a_path)
    suite_b = _load_suite(suite_b_path)
    comparison = _aggregate(suite_a, suite_b)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_csv, comparison)
    _build_pdf(output_pdf, comparison)
    result = {
        "pdf": str(output_pdf),
        "pdf_sha256": _sha256(output_pdf),
        "json": str(output_json),
        "json_sha256": _sha256(output_json),
        "csv": str(output_csv),
        "csv_sha256": _sha256(output_csv),
        "suite_a": str(suite_a_path),
        "suite_b": str(suite_b_path),
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-a", type=Path, default=DEFAULT_SUITE_A)
    parser.add_argument("--suite-b", type=Path, default=DEFAULT_SUITE_B)
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_PDF)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = build(
        suite_a_path=args.suite_a,
        suite_b_path=args.suite_b,
        output_pdf=args.output_pdf,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
