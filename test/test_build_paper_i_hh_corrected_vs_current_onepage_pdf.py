from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.reporting import build_paper_i_hh_corrected_parent_comparator_page13_pdf as base
from pipelines.reporting import build_paper_i_hh_corrected_vs_current_onepage_pdf as report


def _display_row(regime: str, method: str) -> base.DisplayRow:
    return base.DisplayRow(
        regime=regime,
        method=method,
        k_pl=1,
        history_position=0,
        logical_depth=1,
        abs_delta_e=0.1,
        n2q=1,
        d2q=1,
        dc=2,
        s_alg=3,
        s_components={"N_H_outer": 1, "N_H_refit": 1, "N_grad_selector": 1, "S_alg": 3},
        curve=[base.CurvePoint(0, 1.0), base.CurvePoint(1, 0.1)],
        source_json=f"{regime}/{method}.json",
        source_sha256="a" * 64,
        cost_source="test_qiskit",
        cost_metadata={},
        validation={"ok": True},
    )


def _synthetic_corrected_rows() -> list[base.DisplayRow]:
    rows = [_display_row(regime, "snake") for regime in base.REGIME_ORDER]
    rows.extend(
        _display_row(regime, method)
        for regime in base.REGIME_ORDER
        for method in ("geo", "append")
        if (regime, method) not in report.DEFERRED_PAIRS
    )
    return rows


def test_active_page13_parser_reads_all_rows_and_strong_strong_override() -> None:
    cells = report._current_paper_cells()

    assert len(cells) == 18
    assert cells[("strong-strong", "append")] == {
        "k_pl": 8,
        "abs_delta_e": pytest.approx(3.616e-05),
        "N2q": 7976,
        "D2q": 7833,
        "Dc": 26219,
        "S_alg": 6378,
    }


def test_current_paper_rows_lock_curves_tables_and_qiskit_costs() -> None:
    rows = report.build_current_paper_rows()
    by_key = {(row.regime, row.method): row for row in rows}

    assert len(rows) == 12
    assert all(all(row.validation.values()) for row in rows)
    assert all(row.curve[0].k == 0 for row in rows)
    assert by_key[("weak-weak", "append")].plot_marker_error != pytest.approx(
        by_key[("weak-weak", "append")].table_error
    )
    strong_strong_append = by_key[("strong-strong", "append")]
    assert strong_strong_append.k_pl == 8
    assert strong_strong_append.n2q == 7976
    assert strong_strong_append.cost_source == "qiskit_prefix_recompile_at_plot_iteration_k8"


def test_validation_requires_all_nondeferred_corrected_rows() -> None:
    current = report.build_current_paper_rows()
    corrected = _synthetic_corrected_rows()

    checks = report.validate_rows(
        corrected,
        current,
        deferred_pairs=report.DEFERRED_PAIRS,
    )
    assert all(checks.values())

    running_omitted = [
        row
        for row in corrected
        if (row.regime, row.method) not in report.RUNNING_PAIRS
    ]
    running_checks = report.validate_rows(
        running_omitted,
        current,
        deferred_pairs=report.DEFERRED_PAIRS,
    )
    assert all(running_checks.values())

    invalid_missing = [
        row
        for row in corrected
        if (row.regime, row.method) != ("weak-weak", "geo")
    ]
    with pytest.raises(ValueError, match="only_running_rows_lack_completed_results"):
        report.validate_rows(
            invalid_missing,
            current,
            deferred_pairs=report.DEFERRED_PAIRS,
        )


def test_strong_strong_table_marks_corrected_rows_deferred() -> None:
    current = report.build_current_paper_rows()
    tex = report._table_tex("strong-strong", _synthetic_corrected_rows(), current)

    assert "SNAKE P-I" in tex
    assert "Geo P-I" in tex
    assert "App. P-I" in tex
    assert tex.count("deferred") == 2
    assert "7,976" in tex

    running_rows = [
        row
        for row in _synthetic_corrected_rows()
        if (row.regime, row.method) != ("weak-strong", "append")
    ]
    running_tex = report._table_tex("weak-strong", running_rows, current)
    assert running_tex.count("running") == 1


def test_onepage_tex_has_manifest_and_no_page_break(tmp_path: Path) -> None:
    current = report.build_current_paper_rows()
    corrected = _synthetic_corrected_rows()
    figures = [
        {"regime": regime, "pdf": str(tmp_path / f"{regime}.pdf")}
        for regime in base.REGIME_ORDER
    ]
    tex_path = tmp_path / "report.tex"

    report._write_onepage_tex(
        tex_path,
        corrected_rows=corrected,
        current_rows=current,
        figures=figures,
        report_json=tmp_path / "report.json",
        report_csv=tmp_path / "report.csv",
        generated_utc="2026-07-10T00:00:00+00:00",
    )
    source = tex_path.read_text(encoding="utf-8")

    assert "Manifest & Candidate report" in source
    assert "Strong--strong corrected Geo/Append were not run" in source
    assert "\\clearpage" not in source
    assert source.count("\\includegraphics") == 6
