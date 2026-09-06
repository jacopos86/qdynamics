"""Small pure helpers of the corrected-vs-reported report generator."""

from __future__ import annotations

from pathlib import Path

from pipelines.time_dynamics.diagnostics.paper_ii_corrected_campaign_report import (
    _fallback_avqds_row,
    _match_text,
)


def test_fallback_row_reads_the_manuscript_avqds_line(tmp_path: Path) -> None:
    table = tmp_path / "table.tex"
    table.write_text(
        "\n".join(
            [
                r"$(2.4,\,3)$ & $\delta\theta$ & APM & 3.50 & 1.44 & 4.71 & 44 & 144 & 113 & 432 & 14.0 & 1227 \\",
                r" &  & AVQDS & 20.7 & 2.13 & 7.41 & 555 & 3123 & 2893 & 10979 & 346 & 441 \\",
            ]
        ),
        encoding="utf-8",
    )
    assert _fallback_avqds_row(table, "strongfast", "delta_theta_5e-5") == [
        "20.7", "2.13", "7.41", "555", "3123", "2893", "10979", "346", "441",
    ]
    assert _fallback_avqds_row(table, "weakslow", "delta_theta_5e-5") is None


def test_match_text_reports_the_verdict_and_worst_factor() -> None:
    assert _match_text({"passes": True, "worst_component_ratio": 1.3209}) == "yes (1.32)"
    assert _match_text({"passes": False, "worst_component_ratio": 2.82}) == "no (2.82)"
    assert _match_text(None) == "--"
    assert _match_text({"passes": False, "worst_component_ratio": None}) == "--"
