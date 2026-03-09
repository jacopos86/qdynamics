from docs.reports.report_labels import (
    report_branch_label,
    report_method_label,
    report_metric_label,
    report_stage_label,
)
from pathlib import Path

from docs.reports.report_pages import (
    build_section_divider_text,
    build_sectioned_report_lines,
)


def test_report_label_registry_returns_readable_display_names() -> None:
    assert report_method_label("cfqm4") == "CFQM4"
    assert report_stage_label("adapt_pool_b") == "ADAPT Pool-B"
    assert report_branch_label("trotter_hva") == "Trotter HVA replay"
    assert report_metric_label("max_abs_delta_over_stderr") == "Max |Δ| / stderr"


def test_sectioned_report_lines_use_grouped_structure() -> None:
    lines = build_sectioned_report_lines(
        title="Executive summary",
        subtitle="Executive summary",
        experiment_statement="Prepared-state quality before dynamics.",
        sections=(
            ("Headline results", (("Final |ΔE|", 1.2e-3), ("Completed modes", "2 / 3"))),
            ("Coverage", (("Benchmark rows", 4),)),
        ),
        notes=["Appendix holds the full command."],
    )

    assert lines[0] == "Executive summary"
    assert "Experiment: Prepared-state quality before dynamics." in lines
    assert "Headline results:" in lines
    assert "  - Final |ΔE|: 0.0012" in lines
    assert "  - Completed modes: 2 / 3" in lines
    assert "Notes:" in lines


def test_section_divider_text_keeps_summary_and_focus_bullets() -> None:
    text = build_section_divider_text(
        title="Technical appendix",
        summary="Supporting diagnostics and reproducibility material.",
        bullets=["3D surfaces.", "Executed command."],
    )

    assert "Supporting diagnostics and reproducibility material." in text
    assert "Focus:" in text
    assert "- 3D surfaces." in text
    assert "- Executed command." in text


def test_sectioned_report_lines_handle_non_json_native_dict_values() -> None:
    lines = build_sectioned_report_lines(
        title="Manifest",
        sections=(("Configs", (("Mitigation", {"path": Path("artifact.json")}),)),),
    )

    assert any("artifact.json" in line for line in lines)
