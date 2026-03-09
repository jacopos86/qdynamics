from __future__ import annotations

import json
from typing import Any, Iterable, Sequence

from docs.reports.pdf_utils import render_info_page, render_text_page

SectionRows = Sequence[tuple[str, Any]]
SectionSpec = Sequence[tuple[str, SectionRows]]


def _format_value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (list, tuple, set)):
        rendered = ", ".join(_format_value(item) for item in value)
        return rendered or "none"
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except TypeError:
            return str(value)
    return str(value)


"""page_lines = title + optional subtitle/statement + grouped key-value sections + notes"""
def build_sectioned_report_lines(
    *,
    title: str,
    subtitle: str | None = None,
    experiment_statement: str | None = None,
    sections: SectionSpec = (),
    notes: Iterable[str] | None = None,
) -> list[str]:
    lines: list[str] = [title]
    if subtitle:
        lines += ["", str(subtitle)]
    if experiment_statement:
        lines += ["", f"Experiment: {experiment_statement}"]

    for header, rows in sections:
        lines += ["", f"{header}:"]
        for label, value in rows:
            lines.append(f"  - {label}: {_format_value(value)}")

    note_lines = [str(line) for line in (notes or []) if str(line).strip()]
    if note_lines:
        lines += ["", "Notes:"]
        lines.extend(f"  - {line}" for line in note_lines)
    return lines


"""render_page(pdf) = render_text_page(pdf, build_sectioned_report_lines(...))"""
def render_sectioned_report_page(
    pdf: Any,
    *,
    title: str,
    subtitle: str | None = None,
    experiment_statement: str | None = None,
    sections: SectionSpec = (),
    notes: Iterable[str] | None = None,
    fontsize: int = 10,
    max_line_width: int = 110,
) -> None:
    render_text_page(
        pdf,
        build_sectioned_report_lines(
            title=title,
            subtitle=subtitle,
            experiment_statement=experiment_statement,
            sections=sections,
            notes=notes,
        ),
        fontsize=fontsize,
        line_spacing=0.028,
        max_line_width=max_line_width,
    )


"""manifest_page = sectioned_report_page(front_matter grouped for reproducibility)"""
def render_manifest_overview_page(
    pdf: Any,
    *,
    title: str,
    experiment_statement: str,
    sections: SectionSpec,
    notes: Iterable[str] | None = None,
) -> None:
    render_sectioned_report_page(
        pdf,
        title=title,
        subtitle="Parameter manifest",
        experiment_statement=experiment_statement,
        sections=sections,
        notes=notes,
        fontsize=10,
        max_line_width=110,
    )


"""summary_page = sectioned_report_page(headline results before dense plots)"""
def render_executive_summary_page(
    pdf: Any,
    *,
    title: str,
    experiment_statement: str,
    sections: SectionSpec,
    notes: Iterable[str] | None = None,
) -> None:
    render_sectioned_report_page(
        pdf,
        title=title,
        subtitle="Executive summary",
        experiment_statement=experiment_statement,
        sections=sections,
        notes=notes,
        fontsize=10,
        max_line_width=110,
    )


"""divider_text = section_purpose + focus bullets + appendix/science boundary"""
def build_section_divider_text(
    *,
    title: str,
    summary: str,
    bullets: Iterable[str] | None = None,
) -> str:
    parts = [summary.strip()]
    bullet_lines = [str(item).strip() for item in (bullets or []) if str(item).strip()]
    if bullet_lines:
        parts.append("")
        parts.append("Focus:")
        parts.extend(f"- {line}" for line in bullet_lines)
    return "\n".join(parts)


"""divider_page(pdf) = info_page(section boundary, not data dump)"""
def render_section_divider_page(
    pdf: Any,
    *,
    title: str,
    summary: str,
    bullets: Iterable[str] | None = None,
) -> None:
    render_info_page(
        pdf,
        build_section_divider_text(title=title, summary=summary, bullets=bullets),
        title=title,
    )
