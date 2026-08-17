from __future__ import annotations

import importlib.util
import csv
import io
import math
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    REPO_ROOT
    / "pipelines/reporting/build_paper_i_local_vs_chtc_replacement_report.py"
)


def _module():
    name = "test_paper_i_local_vs_chtc_replacement_report_module"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _metrics(**overrides):
    value = {
        "delta_e": 1.0e-6,
        "N2q": 100,
        "D2q": 80,
        "Dc": 400,
        "W1q": 200,
        "S_alg": 1000,
        "exact_same_cutoff_energy": -1.0,
    }
    value.update(overrides)
    return value


def test_cell_matrix_is_exact_full_24_curve_paper_i_scope():
    report = _module()
    rows = report.cell_specs()

    assert len(rows) == 24
    assert len({row.key for row in rows}) == 24
    assert sum(row.nph == 3 for row in rows) == 12
    assert sum(row.nph == 7 for row in rows) == 12
    assert sum(row.method == "ra_plateau" for row in rows) == 6
    assert sum(row.method == "ra_append_only" for row in rows) == 6
    assert sum(row.method == "ra_always_cr" for row in rows) == 6
    assert sum(row.method == "append_conventional" for row in rows) == 6
    for regime, _label, _nph in report.REGIMES:
        assert {
            row.method for row in rows if row.regime == regime
        } == {
            "ra_plateau",
            "ra_append_only",
            "ra_always_cr",
            "append_conventional",
        }


def test_comparison_classifies_pareto_dominance_and_tradeoff():
    report = _module()
    historical = _metrics()

    local_better = _metrics(
        delta_e=5.0e-7, N2q=99, D2q=79, Dc=399, W1q=199, S_alg=999
    )
    assert (
        report.compare_metrics(local_better, historical)["classification"]
        == "local_pareto_better"
    )

    local_worse = _metrics(
        delta_e=2.0e-6, N2q=101, D2q=81, Dc=401, W1q=201, S_alg=1001
    )
    assert (
        report.compare_metrics(local_worse, historical)["classification"]
        == "chtc_dominates"
    )

    local_tradeoff = _metrics(delta_e=5.0e-7, N2q=101)
    assert (
        report.compare_metrics(local_tradeoff, historical)["classification"]
        == "tradeoff"
    )


def test_delta_e_tie_uses_declared_ulp_tolerance():
    report = _module()
    historical = _metrics(delta_e=0.0)
    tolerance = 128.0 * math.ulp(1.0)
    local = _metrics(delta_e=tolerance)

    comparison = report.compare_metrics(local, historical)

    assert comparison["metric_directions"]["delta_e"] == "tie"
    assert comparison["energy_comparison"] == "delta_e_equivalent_within_tolerance"


def test_latex_starts_with_combined_six_regime_plot_then_consolidated_table():
    report = _module()
    cells = []
    for spec in report.cell_specs():
        historical = {
            **_metrics(),
            "execution_id": f"chtc::{spec.key}",
            "qiskit_version": "2.3.1",
        }
        cells.append(
            {
                "key": spec.key,
                "group": spec.group,
                "method": spec.method,
                "method_label": spec.method_label,
                "regime": spec.regime,
                "regime_label": spec.regime_label,
                "nph": spec.nph,
                "local_state": "pending",
                "historical": historical,
                "local": None,
                "comparison": report.compare_metrics(None, historical),
            }
        )
    payload = {
        "cells": cells,
        "campaign": {"state": "running"},
        "counts": {
            "completed": 5,
            "local_pareto_better": 0,
            "chtc_dominates": 0,
        },
        "evidence_revision_sha256": "a" * 64,
        "generated_at_utc": "2026-08-15T00:00:00Z",
    }

    latex = report.render_latex(payload)

    assert "PROVISIONAL DIAGNOSTIC" in latex
    assert "Local closures" not in latex
    assert "Decision basis." not in latex
    assert latex.count("\\includegraphics") == 1
    assert "six-regime convergence" in latex
    assert "Top row: weak Holstein sector" in latex
    assert "bottom row: strong Holstein sector" in latex
    assert "Every panel contains the complete four-method Paper-I set" in latex
    assert latex.index("\\includegraphics") < latex.index("Regime / method & Origin")
    assert latex.index("Regime / method & Origin") < latex.index(
        "Parameter and provenance"
    )
    for _regime, regime_label, _nph in report.REGIMES:
        assert regime_label in latex
    assert "N_{2q}" in latex
    assert "D_{2q}" in latex
    assert "D_c" in latex
    assert "W_{1q}" in latex
    assert "S_{\\rm alg}" in latex
    assert "paper\\_adoption\\_authorized" in latex
    assert "paper\\_evidence\\_adoption\\_authorized" in latex
    assert "RA append-only" in latex
    assert "RA always-open" in latex


def test_combined_plot_source_uses_paper_i_two_by_three_shared_row_layout():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "plt.subplots(2, 3" in source
    assert 'sharey="row"' in source
    assert '"two_holstein_rows_by_three_hubbard_columns"' in source
    assert '"explicit_user_requested_combined_six_regime_layout": True' in source


def test_position_aware_phase0_overlay_is_supplemental_not_k50_decision():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "def _position_aware_phase0_overlay()" in source
    assert '"target_horizon": 15' in source
    assert '"decision_table_inclusion": False' in source
    assert '"supplemental_curves": supplemental_curves' in source
    assert "the supplemental position-aware local curve marks its effective plateau" in source
    assert "supplemental_authenticated_k15_diagnostic_plotted" in source


def test_csv_records_position_aware_overlay_as_nondecision_diagnostic():
    report = _module()
    payload = {
        "cells": [],
        "supplemental_curves": [
            {
                "key": "position_aware_phase0::ra_always_cr::strong_weak_u8",
                "method": "ra_always_cr",
                "regime": "strong_weak_u8",
                "nph": 3,
                "origin": "new local diagnostic",
                "execution_id": "diagnostic-k15",
                "terminal": {
                    "delta_e": 8.28e-7,
                    "N2q": 72,
                    "D2q": 56,
                    "Dc": 258,
                    "W1q": None,
                    "S_alg": 68380,
                },
            }
        ],
    }

    rows = list(csv.DictReader(io.StringIO(report._csv_bytes(payload).decode())))

    assert len(rows) == 1
    assert rows[0]["local_state"] == "authenticated_k15_diagnostic_not_k50_decision"
    assert rows[0]["classification"] == "supplemental_not_in_k50_decision"
    assert rows[0]["W1q"] == ""


def test_trajectory_requires_contiguous_fixed_k50_prefix():
    report = _module()
    valid = [
        {"controller_round": round_, "absolute_energy_error": 1.0 / round_}
        for round_ in range(1, 51)
    ]

    points = report._trajectory(
        valid,
        label="test",
        round_key="controller_round",
        error_key="absolute_energy_error",
    )

    assert points[0]["k"] == 1
    assert points[-1] == {"k": 50, "delta_e": 0.02}

    with pytest.raises(report.ReportError, match="not contiguous"):
        report._trajectory(
            [row for row in valid if row["controller_round"] != 17],
            label="gap",
            round_key="controller_round",
            error_key="absolute_energy_error",
        )


def test_plot_marker_rejects_error_drift():
    report = _module()
    trajectory = [{"k": round_, "delta_e": 1.0 / round_} for round_ in range(1, 51)]

    with pytest.raises(report.ReportError, match="marker error drifted"):
        report._trajectory_marker(
            trajectory,
            marker={
                "k": 20,
                "error": 0.5,
                "policy": "first_effective_plateau_prefix",
            },
        )


def test_report_source_has_no_manuscript_mutation_target():
    source = SCRIPT.read_text(encoding="utf-8")

    assert "MATH/paper_details/Paper_I.tex" not in source
    assert "paper_i_ra_vs_append_matched_singleton_plateau.pdf" not in source
    assert "paper_adoption_authorized\": False" in source
    assert "paper_evidence_adoption_authorized\": False" in source


def test_compile_tex_exposes_mactex_children_to_launchd_path(tmp_path, monkeypatch):
    report = _module()
    tex_path = tmp_path / "report.tex"
    tex_path.write_text("report", encoding="utf-8")
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0].endswith("latexmk"):
            (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 10_000)
        return SimpleNamespace(returncode=0, stdout="Pages: 1\n", stderr="")

    monkeypatch.setattr(report.subprocess, "run", fake_run)
    monkeypatch.setattr(
        report.shutil,
        "which",
        lambda command: "/usr/bin/pdfinfo" if command == "pdfinfo" else None,
    )

    built = report._compile_tex(tex_path, output_dir=tmp_path)

    assert built == tmp_path / "report.pdf"
    latex_env = calls[0][1]["env"]
    assert latex_env["PATH"].split(report.os.pathsep)[0] == "/Library/TeX/texbin"
