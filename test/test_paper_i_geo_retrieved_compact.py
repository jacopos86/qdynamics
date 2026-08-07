from __future__ import annotations

from pathlib import Path

import pytest

from pipelines.reporting.build_paper_i_geo_retrieved_compact import (
    DEFAULT_GROUPED_EXACT_MAX_ACTIVE_QUBITS,
    EXPECTED_CASE_IDS,
    EXPECTED_MAIN_CASE_IDS,
    PAGE_CASE_COUNTS,
    filter_overlay_methods,
    ordered_inventory_rows,
    page_chunks,
    parse_args,
    plot_compact_case,
    write_report_tex,
)
from pipelines.reporting.paper_i_geo_compact_method_overlays import write_overlay_csv


def _inventory() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for index, case_id in enumerate(EXPECTED_CASE_IDS):
        rows.append(
            {
                "case_id": case_id,
                "paper_placement": (
                    "main_results_hubbard_holstein_L2"
                    if index < len(EXPECTED_MAIN_CASE_IDS)
                    else "appendix_scaling_results"
                ),
            }
        )
    return {"rows": rows}


def _report_row(index: int, output_dir: Path) -> dict[str, object]:
    return {
        "order_index": index,
        "case_id": EXPECTED_CASE_IDS[index],
        "family": "hh",
        "L": 2,
        "display_regime": "weak-weak",
        "compact_title_tex": f"Case {index + 1}",
        "compact_plot": {
            "pdf": str(output_dir / "plots" / f"case_{index:02d}.pdf")
        },
        "marker": {"k": 2, "error_raw": 1.0e-5},
        "query_ledger": {"S": 1234},
        "qiskit_prefix_cost": {
            "status": "ok",
            "N2q": 10,
            "D2q": 8,
            "Dcirc": 20,
        },
    }


def _overlay_report_row(index: int, output_dir: Path) -> dict[str, object]:
    row = _report_row(index, output_dir)
    row["overlay_mode"] = True
    row["method_overlays"] = {
        method: {
            "status": "ok",
            "trajectory_status": "ok",
            "cost_status": "ok",
            "marker": {
                "k": 10 + offset,
                "error_raw": 1.0e-4 / (offset + 1),
                "error_plotted": 1.0e-4 / (offset + 1),
            },
            "curve": [
                {"k": 0, "error_raw": 1.0, "error_plotted": 1.0},
                {
                    "k": 10 + offset,
                    "error_raw": 1.0e-4 / (offset + 1),
                    "error_plotted": 1.0e-4 / (offset + 1),
                },
            ],
            "qiskit_cost": {"N2q": 10 + offset, "D2q": 8 + offset, "Dcirc": 20 + offset},
            "query_ledger": {"S": 1000 + offset},
        }
        for offset, method in enumerate(("Append-ADAPT", "Geo-ADAPT", "SNAKE"))
    }
    row["method_overlays"]["SNAKE"].update(
        {
            "status": "ok",
            "trajectory_status": "ok",
            "trajectory_diagnostic": {
                "history_round_count": 50,
            },
        }
    )
    return row


def test_inventory_contract_is_exactly_six_main_then_thirty_four_scaling() -> None:
    rows = ordered_inventory_rows(_inventory())
    assert len(rows) == 40
    assert tuple(row["case_id"] for row in rows[:6]) == EXPECTED_MAIN_CASE_IDS
    assert tuple(row["case_id"] for row in rows) == EXPECTED_CASE_IDS

    broken = _inventory()
    broken["rows"][0], broken["rows"][1] = broken["rows"][1], broken["rows"][0]
    with pytest.raises(ValueError, match="exact ordered 40-case Geo contract"):
        ordered_inventory_rows(broken)


def test_page_contract_is_exactly_five_pages_with_requested_panel_counts() -> None:
    rows = [{"case_id": case_id} for case_id in EXPECTED_CASE_IDS]
    chunks = page_chunks(rows)
    assert tuple(map(len, chunks)) == PAGE_CASE_COUNTS == (6, 9, 9, 9, 7)
    assert chunks[0][0]["case_id"] == EXPECTED_MAIN_CASE_IDS[0]
    assert chunks[1][0]["case_id"] == EXPECTED_CASE_IDS[6]


def test_report_tex_has_manifest_five_page_break_contract_and_forty_tables(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "paper_i_geo_retrieved_compact_20260711"
    output_dir.mkdir()
    rows = [_report_row(index, output_dir) for index in range(40)]
    tex_path = output_dir / "report.tex"
    write_report_tex(
        tex_path,
        rows=rows,
        inventory_path=tmp_path / "inventory.json",
        inventory_sha256="a" * 64,
        provenance_json=output_dir / "provenance.json",
        provenance_csv=output_dir / "provenance.csv",
        manifest_json=output_dir / "manifest.json",
    )
    source = tex_path.read_text(encoding="utf-8")
    assert "BEGIN_MACHINE_READABLE_GEO_RETRIEVED_COMPACT" in source
    assert "Parameter" not in source  # manifest is rendered directly, not a placeholder
    assert "40 completed rows" in source
    assert source.count(r"\begin{minipage}[t][2.68in][t]") == 40
    assert source.count(r"\clearpage") == 4
    assert source.count(r"$k_{\rm pl}$ & $|\Delta E|$") == 40
    assert "all 40 rows validated" in source


def test_cli_defaults_to_grouped_exact_active_qubit_cap_eight() -> None:
    args = parse_args([])
    assert DEFAULT_GROUPED_EXACT_MAX_ACTIVE_QUBITS == 8
    assert args.grouped_exact_max_active_qubits == 8
    assert args.skip_qiskit is False


def test_overlay_tex_keeps_five_page_layout_and_three_terminal_cost_rows(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "comparator"
    output_dir.mkdir()
    rows = [_overlay_report_row(index, output_dir) for index in range(40)]
    tex_path = output_dir / "report.tex"
    write_report_tex(
        tex_path,
        rows=rows,
        inventory_path=tmp_path / "inventory.json",
        inventory_sha256="b" * 64,
        provenance_json=output_dir / "provenance.json",
        provenance_csv=output_dir / "provenance.csv",
        manifest_json=output_dir / "manifest.json",
        overlay_summary={
            "complete_case_count": 0,
            "displayable_three_method_case_count": 40,
            "snake_history_round_audit": {
                "audited_row_count": 40,
            },
        },
        append_roots=[tmp_path / "append"],
        snake_roots=[tmp_path / "snake"],
    )
    source = tex_path.read_text(encoding="utf-8")
    assert source.count(r"\clearpage") == 4
    assert source.count(r"Method & index") == 40
    assert source.count(r"    Append ($k_{\rm pl}$) & ") == 40
    assert source.count(r"    Geo ($k_{\rm pl}$) & ") == 40
    assert source.count(r"    SNAKE ($r_{\rm hist}$) & ") == 40
    assert "Displayable three-method panels: 40/40" in source
    assert "SNAKE curves use source history rounds" in source
    assert "overlay_mode=true" in source


def test_overlay_cli_accepts_repeatable_explicit_roots_and_completion_gate() -> None:
    args = parse_args(
        [
            "--append-source-root",
            "append-a",
            "--append-source-root",
            "append-b",
            "--snake-source-root",
            "snake",
            "--require-complete-overlays",
            "--audit-overlays-only",
        ]
    )
    assert args.append_source_root == [Path("append-a"), Path("append-b")]
    assert args.snake_source_root == [Path("snake")]
    assert args.require_complete_overlays is True
    assert args.audit_overlays_only is True


def test_append_geo_filter_removes_snake_from_tex_csv_and_summary(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "comparator"
    output_dir.mkdir()
    rows = [_overlay_report_row(index, output_dir) for index in range(40)]
    rows, summary = filter_overlay_methods(
        rows,
        {
            "schema": "paper_i_geo_compact_overlay_summary_v1",
            "append_root_audit": [{"status": "ok"}],
            "snake_root_audit": [{"status": "ok"}],
        },
        include_append=True,
        include_snake=False,
    )
    assert summary["active_methods"] == ["Append-ADAPT", "Geo-ADAPT"]
    assert "snake_root_audit" not in summary
    assert all(
        tuple(row["method_overlays"]) == ("Append-ADAPT", "Geo-ADAPT")
        for row in rows
    )

    tex_path = output_dir / "report.tex"
    write_report_tex(
        tex_path,
        rows=rows,
        inventory_path=tmp_path / "inventory.json",
        inventory_sha256="c" * 64,
        provenance_json=output_dir / "provenance.json",
        provenance_csv=output_dir / "provenance.csv",
        manifest_json=output_dir / "manifest.json",
        overlay_summary=summary,
        append_roots=[tmp_path / "append"],
    )
    source = tex_path.read_text(encoding="utf-8")
    assert "SNAKE" not in source
    assert source.count(r"    Append ($k_{\rm pl}$) & ") == 40
    assert source.count(r"    Geo ($k_{\rm pl}$) & ") == 40
    assert "Displayable two-method panels: 40/40" in source

    csv_path = output_dir / "provenance.csv"
    write_overlay_csv(csv_path, rows)
    csv_source = csv_path.read_text(encoding="utf-8")
    assert "SNAKE" not in csv_source
    assert csv_source.count("Append-ADAPT") == 40
    assert csv_source.count("Geo-ADAPT") == 40

    plot = plot_compact_case(rows[0], plot_dir=output_dir / "plots")
    assert plot["pdf"].endswith("__append_geo_compact.pdf")
    assert plot["display_method_count"] == 2
    assert tuple(plot["markers"]) == ("Append-ADAPT", "Geo-ADAPT")
