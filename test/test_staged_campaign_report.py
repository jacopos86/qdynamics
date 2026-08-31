"""Compact Markdown/PNG/LaTeX-PDF reporting for the staged campaign."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pypdf import PdfReader

from pipelines.time_dynamics.diagnostics import staged_campaign_report as report
from pipelines.time_dynamics.staged_campaign import (
    AccuracyMetrics,
    AccuracyPolicy,
    CampaignLedger,
    CellKey,
    RungRecord,
)


def _row(time: float, offset: float = 0.0) -> dict:
    return {
        "time": time,
        "energy_expectation": 1.0 + time + offset,
        "doublon": 0.2 + 0.01 * time + offset,
        "primary_density": 0.4 + 0.01 * time + offset,
        "site_occupations": [0.4 + 0.01 * time + offset, 0.6 - 0.01 * time],
        "psi_real": [1.0, 0.0],
        "psi_imag": [0.0, 0.0],
    }


def _write_run(path: Path, times, *, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "plot_rows": [_row(time, offset) for time in times],
                "run_lock": {"schema": "lock"},
                "summary": {
                    "accepted_internal_substep_count": 2,
                    "attempted_internal_step_count": 3,
                    "rejected_internal_step_count": 1,
                    "rhs_evaluation_count": 4,
                    "accepted_insert_count": 1,
                    "accepted_delete_count": 1,
                    "accepted_exchange_count": 0,
                    "runtime_parameter_count_final": 12,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_cost(run_path: Path, shift: int) -> None:
    row = {
        "graph_proxy_terminal_c_hat_2q": 100 + shift,
        "graph_proxy_terminal_c_hat_d": 80 + shift,
        "graph_proxy_terminal_c_hat_1q": 120 + shift,
        "graph_proxy_terminal_theta_count": 12,
        "graph_proxy_delta_c_hat_2q": 10 + shift,
        "graph_proxy_delta_c_hat_d": 8 + shift,
        "graph_proxy_delta_c_hat_1q": 12 + shift,
        "graph_proxy_delta_theta_count": 2 + shift,
        "N2q": 150 + shift,
        "Dc": 300 + shift,
        "D2q": 90 + shift,
        "terminal_reconstruction_parity": {"passed": True},
    }
    run_path.with_name("terminal_qiskit_cost.json").write_text(
        json.dumps({"rows": [row]}),
        encoding="utf-8",
    )


def test_report_places_compact_plot_before_pair_tables(tmp_path, monkeypatch) -> None:
    ledger = CampaignLedger()
    policy = AccuracyPolicy()
    metrics = AccuracyMetrics(4.0e-5, 3.0e-5, 2.0e-5, 1.0e-5)
    for method, offset, shift in (("avqds", 1.0e-5, 1), ("exchange", 0.0, 0)):
        screen = tmp_path / method / "t5" / "run.json"
        promoted = tmp_path / method / "t20" / "run.json"
        _write_run(screen, (0.0, 5.0), offset=offset)
        _write_run(promoted, (5.0, 20.0), offset=offset)
        _write_cost(promoted, shift)
        ledger = ledger.append(
            RungRecord.completed(
                key=CellKey("hh", "fastweak", method, "delta_theta"),
                horizon="t20",
                threshold=1.0e-6,
                run_path=str(promoted),
                metrics=metrics,
                policy=policy,
                resume_from_run_path=str(screen),
            )
        )

    monkeypatch.setattr(report, "assert_comparable", lambda _locks: None)

    def exact(_seed, _run, *, times, substeps):
        assert substeps == 4
        return [
            {
                "time": float(time),
                "energy": 1.0 + float(time),
                "doublon": 0.2 + 0.01 * float(time),
                "primary_density": 0.4 + 0.01 * float(time),
                "site_occupations": [
                    0.4 + 0.01 * float(time),
                    0.6 - 0.01 * float(time),
                ],
                "statevector": np.asarray([1.0, 0.0], dtype=complex),
            }
            for time in times
        ]

    monkeypatch.setattr(report, "exact_rows_on_times", exact)
    output = tmp_path / "report"
    manifest = report.write_report(
        ledger=ledger,
        seed_path=tmp_path / "seed.json",
        output_dir=output,
        reference_substeps=4,
        source_ledger_path=tmp_path / "ledger.json",
        cluster_ids=(9697391,),
        audit_status="PASS_FIXTURE",
    )
    markdown = (output / "paper_ii_staged_report.md").read_text(encoding="utf-8")
    image = output / "plots" / "fastweak__delta_theta__t20.png"
    pdf = output / "paper_ii_staged_report.pdf"
    tex = output / "paper_ii_staged_report.tex"
    assert manifest["qualified_pair_count"] == 1
    assert manifest["ledger_record_count"] == 2
    assert image.is_file() and image.stat().st_size > 0
    assert pdf.is_file() and pdf.stat().st_size > 0
    assert tex.is_file() and tex.stat().st_size > 0
    assert manifest["pdf_engine"] in {"latexmk", "tectonic", "pdflatex"}
    assert manifest["tex"] == str(tex)
    tex_source = tex.read_text(encoding="utf-8")
    assert "fastweak - delta theta - t20" in tex_source
    assert "fastweak - delta\\_theta - t20" not in tex_source
    assert markdown.index("![Observable trajectories]") < markdown.index("mean |ΔE|")
    assert markdown.index("Campaign ledger overview") < markdown.index(
        "![Observable trajectories]"
    )
    assert "Qiskit" not in markdown
    pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    assert "CHTC clusters" in pdf_text
    assert "9697391" in pdf_text
    assert "PASS FIXTURE" in pdf_text
    assert "Qiskit" not in pdf_text
    assert manifest["pdf_sha256"]


def test_results_report_refuses_a_ledger_without_an_exchange_pair(tmp_path) -> None:
    with pytest.raises(ValueError, match="APM/AVQDS pair"):
        report.write_report(
            ledger=CampaignLedger(),
            seed_path=tmp_path / "seed.json",
            output_dir=tmp_path / "report",
            audit_status="PARTIAL",
        )
    assert not (tmp_path / "report" / "paper_ii_staged_report.pdf").exists()
