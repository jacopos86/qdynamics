from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from pipelines.reporting import (
    build_paper_i_hh_weak_weak_fm_outer_reuse_error_cost as report,
)


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _qiskit(
    path: Path,
    *,
    k: int,
    n2q: int,
    d2q: int,
    dcirc: int,
    error: float,
) -> Path:
    return _write_json(
        path,
        {
            "schema": "paper_i_selected_prefix_qiskit_cost_sidecar_v1",
            "compile_convention": "table_i_basis_gate_transpile_v1",
            "compiled_circuit_stats_status": "ok",
            "compiled_resource_qiskit_validated": True,
            "qiskit_transpile_optimization_level": 0,
            "qiskit_transpile_seed": 7,
            "history_position": k,
            "compiled_count_2q_total": n2q,
            "compiled_depth_2q_total": d2q,
            "compiled_depth_total": dcirc,
            "primary_error_at_prefix": error,
        },
    )


def _closure(
    *,
    winning_s_alg: int,
    discarded_s_alg: int,
    stored_nfev: int,
    nfev_correction: int,
    winning_nfev: int,
    discarded_nfev: int,
) -> dict[str, object]:
    def counts(total: int) -> dict[str, int]:
        return {
            "N_E": total // 2,
            "N_grad": total // 10,
            "N_G": total // 5,
            "N_Hv": 0,
            "N_Q": total - (total // 2 + total // 10 + total // 5),
            "N_cross": 0,
        }

    corrected_nfev = stored_nfev + nfev_correction
    assert winning_nfev + discarded_nfev == corrected_nfev
    return {
        "query_closure": {
            "winning_branch": {
                **counts(winning_s_alg),
                "counts": counts(winning_s_alg),
                "S_alg": winning_s_alg,
            },
            "discarded_branch_operational_overhead": {
                **counts(discarded_s_alg),
                "counts": counts(discarded_s_alg),
                "S_alg": discarded_s_alg,
            },
            "stored_nfev_total": stored_nfev,
            "corrected_nfev_total": corrected_nfev,
            "nfev_correction": nfev_correction,
            "nfev_winning_lineage": winning_nfev,
            "nfev_discarded_operational_overhead": discarded_nfev,
        },
        "correction_receipt": {
            "stored_optimizer_and_guard_nfev": stored_nfev,
            "corrected_optimizer_and_guard_nfev": corrected_nfev,
            "nfev_correction": nfev_correction,
            "correction_reason": "synthetic authoritative-ledger correction",
            "unique_query_oracle_work_changed": False,
        },
    }


def _synthetic_inputs(tmp_path: Path) -> dict[str, Path]:
    reference_pdf = tmp_path / "Paper_I_no_ordinary_novelty_sr_snake_20260717.pdf"
    reference_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    baseline_source = _write_json(
        tmp_path / "baseline_result.json",
        {
            "adapt_vqe": {
                "exact_gs_energy": -1.0,
                "history": [
                    {"depth": 1, "energy_before_opt": 0.0, "delta_abs_current": 0.2},
                    {"depth": 2, "delta_abs_current": 0.01},
                    {"depth": 3, "delta_abs_current": 1.0e-4},
                ],
            }
        },
    )
    baseline_qiskit = _qiskit(
        tmp_path / "baseline_qiskit.json",
        k=3,
        n2q=40,
        d2q=24,
        dcirc=130,
        error=1.0e-4,
    )
    baseline_ledger = _write_json(
        tmp_path / "baseline_estimator_call_ledger.json",
        {"schema": "synthetic_estimator_call_ledger_v1"},
    )
    baseline_evidence = _write_json(
        tmp_path / "baseline_evidence.json",
        {
            "schema": "paper_i_no_ordinary_novelty_sr_snake_evidence_copy_v1",
            "rows": [
                {
                    "regime": "weak_weak",
                    "k_eval": 3,
                    "history_position": 3,
                    "absolute_same_cutoff_error": 1.0e-4,
                    "s_alg": 450,
                    "source": {
                        "path": str(baseline_source),
                        "sha256": _sha256(baseline_source),
                    },
                    "qiskit_sidecar": {
                        "path": str(baseline_qiskit),
                        "sha256": _sha256(baseline_qiskit),
                    },
                    "s_alg_source": {
                        "policy": "canonical_same_state_unique_primitive_v1",
                        "path": str(baseline_ledger),
                        "sha256": _sha256(baseline_ledger),
                    },
                }
            ],
        },
    )
    baseline_accounting = _write_json(
        tmp_path / "baseline_accounting_reclosed.json",
        {
            "schema": "paper_i_fm_query_accounting_correction_sidecar_v1",
            "status": "passed_hash_linked_posthoc_correction",
            "source": {
                "result": {
                    "path": str(baseline_source),
                    "sha256": _sha256(baseline_source),
                },
                "estimator_ledger_sidecar": {
                    "path": str(baseline_ledger),
                    "sha256": _sha256(baseline_ledger),
                },
            },
            **_closure(
                winning_s_alg=450,
                discarded_s_alg=0,
                stored_nfev=100,
                nfev_correction=0,
                winning_nfev=100,
                discarded_nfev=0,
            ),
            "validation": {"passed": True},
        },
    )
    fm_source = _write_json(
        tmp_path / "fm_result.json",
        {
            "adapt_vqe": {
                "exact_gs_energy": -1.0,
                "ansatz_depth": 2,
                "abs_delta_e": 2.0e-6,
                "adapt_reoptimization_route": "formal_manifold_warm_start_v1",
                "history": [
                    {"depth": 1, "energy_before_opt": 0.0, "delta_abs_current": 0.15},
                    {"depth": 2, "delta_abs_current": 0.002},
                    {"depth": 3, "delta_abs_current": 2.1e-6},
                ],
            }
        },
    )
    fm_accounting = _write_json(
        tmp_path / "fm_corrected_accounting.json",
        {
            "schema": "paper_i_fm_query_accounting_correction_sidecar_v1",
            "status": "passed_hash_linked_posthoc_correction",
            "source": {
                "result": {"path": str(fm_source), "sha256": _sha256(fm_source)}
            },
            **_closure(
                winning_s_alg=800,
                discarded_s_alg=100,
                stored_nfev=290,
                nfev_correction=10,
                winning_nfev=200,
                discarded_nfev=100,
            ),
            "validation": {"passed": True},
        },
    )
    settings_drift = _write_json(
        tmp_path / "settings_drift.json",
        {
            "schema": "synthetic_settings_drift_audit_v1",
            "causal_reuse_off_control_present": False,
            "settings_differences": [
                {
                    "field": "route/profile",
                    "baseline": "SR-SNAKE no ordinary novelty",
                    "fm": "formal_manifold_warm_start_v1",
                    "classification": "route-level",
                },
                {
                    "field": "outer geometry reuse",
                    "baseline": "off",
                    "fm": "on",
                    "classification": "not isolated",
                },
            ],
        },
    )
    return {
        "reference_pdf": reference_pdf,
        "baseline_source": baseline_source,
        "baseline_accounting": baseline_accounting,
        "baseline_evidence": baseline_evidence,
        "baseline_qiskit": baseline_qiskit,
        "fm_accounting": fm_accounting,
        "fm_qiskit": _qiskit(
            tmp_path / "fm_qiskit.json",
            k=3,
            n2q=52,
            d2q=31,
            dcirc=170,
            error=2.0e-6,
        ),
        "settings_drift": settings_drift,
    }


def test_builds_explicit_contextual_latex_report_seam(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    output_dir = tmp_path / "report"

    outputs = report.build_report(
        baseline_label="SR-SNAKE no ordinary novelty (2026-07-17)",
        baseline_reference_pdf=inputs["reference_pdf"],
        baseline_source_json=inputs["baseline_source"],
        baseline_accounting_audit_json=inputs["baseline_accounting"],
        baseline_evidence_audit_json=inputs["baseline_evidence"],
        baseline_qiskit_sidecar_json=inputs["baseline_qiskit"],
        fm_corrected_accounting_json=inputs["fm_accounting"],
        fm_qiskit_sidecar_json=inputs["fm_qiskit"],
        settings_drift_audit_json=inputs["settings_drift"],
        output_dir=output_dir,
        compile_pdf=False,
    )

    assert "pdf" not in outputs
    assert outputs["manifest_json"].is_file()
    assert outputs["manifest_csv"].is_file()
    assert outputs["trajectory_csv"].is_file()
    assert outputs["comparison_csv"].is_file()
    assert outputs["error_vs_round_png"].stat().st_size > 0
    assert outputs["error_vs_closed_winning_s_alg_png"].stat().st_size > 0

    payload = json.loads(outputs["report_json"].read_text(encoding="utf-8"))
    assert payload["scope"]["comparison_classification"] == "contextual_route_level_comparison_v1"
    assert payload["scope"]["causal_reuse_off_control_present"] is False
    assert payload["scope"]["prefix_query_trajectory_status"] == (
        "unavailable_no_round_boundary_ledger_checkpoints"
    )
    assert payload["methods"]["baseline"]["reported_abs_delta_e"] == pytest.approx(1.0e-4)
    assert payload["methods"]["fm"]["reported_abs_delta_e"] == pytest.approx(2.0e-6)
    assert payload["methods"]["fm"]["reported_k"] == 3
    assert payload["methods"]["fm"]["reported_winning_s_alg"] == 800
    assert payload["methods"]["baseline"]["reported_discarded_s_alg"] == 0
    assert payload["methods"]["fm"]["reported_discarded_s_alg"] == 100
    assert payload["query_accounting"]["scientific_x_coordinate"] == (
        "winning_branch.S_alg"
    )
    assert payload["query_accounting"]["fm"]["winning_branch"]["counts"] == {
        "N_E": 400,
        "N_grad": 80,
        "N_G": 160,
        "N_Hv": 0,
        "N_Q": 160,
        "N_cross": 0,
    }
    assert payload["query_accounting"]["fm"]["nfev"] == {
        "stored_total": 290,
        "corrected_total": 300,
        "correction": 10,
        "winning_lineage": 200,
        "discarded_operational_overhead": 100,
    }
    assert payload["qiskit_contract"] == {
        "compile_convention": "table_i_basis_gate_transpile_v1",
        "optimization_level": 0,
        "transpile_seed": 7,
    }
    manifest = json.loads(outputs["manifest_json"].read_text(encoding="utf-8"))
    assert manifest["query_accounting"] == payload["query_accounting"]

    tex = outputs["tex"].read_text(encoding="utf-8")
    assert r"\documentclass[10pt,twocolumn]{article}" in tex
    assert "Compact parameter and provenance manifest" in tex
    assert tex.index("Compact parameter and provenance manifest") < tex.index("error_vs_round.png")
    assert "contextual route-level comparison" in tex
    assert "missing: no matched FM outer-reuse-off control" in tex
    assert "$|\\Delta E|$ & Route / compiled point" in tex
    assert "discard $S_{\\rm alg}$" in tex
    assert "Settings-drift audit (exact executed-command differences).}\\par" in tex
    assert "Error is listed first" in tex
    assert r"\clearpage" not in tex

    with outputs["comparison_csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = list(reader)
    assert header[0] == "abs_delta_e"
    assert "discarded_s_alg" in header
    assert len(rows) == 2


def test_fails_closed_on_unmatched_qiskit_convention(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    qiskit = json.loads(inputs["fm_qiskit"].read_text(encoding="utf-8"))
    qiskit["qiskit_transpile_optimization_level"] = 1
    _write_json(inputs["fm_qiskit"], qiskit)

    with pytest.raises(ValueError, match="opt0"):
        report.build_report(
            baseline_label="SR-SNAKE no ordinary novelty",
            baseline_reference_pdf=inputs["reference_pdf"],
            baseline_source_json=inputs["baseline_source"],
            baseline_accounting_audit_json=inputs["baseline_accounting"],
            baseline_evidence_audit_json=inputs["baseline_evidence"],
            baseline_qiskit_sidecar_json=inputs["baseline_qiskit"],
            fm_corrected_accounting_json=inputs["fm_accounting"],
            fm_qiskit_sidecar_json=inputs["fm_qiskit"],
            settings_drift_audit_json=inputs["settings_drift"],
            output_dir=tmp_path / "bad_report",
            compile_pdf=False,
        )


def test_fails_closed_when_fm_query_accounting_is_not_closed(tmp_path: Path) -> None:
    inputs = _synthetic_inputs(tmp_path)
    fm = json.loads(inputs["fm_accounting"].read_text(encoding="utf-8"))
    fm["formal_manifold_query_accounting_complete"] = False
    _write_json(inputs["fm_accounting"], fm)

    with pytest.raises(ValueError, match="does not declare complete closure"):
        report.build_report(
            baseline_label="SR-SNAKE no ordinary novelty",
            baseline_reference_pdf=inputs["reference_pdf"],
            baseline_source_json=inputs["baseline_source"],
            baseline_accounting_audit_json=inputs["baseline_accounting"],
            baseline_evidence_audit_json=inputs["baseline_evidence"],
            baseline_qiskit_sidecar_json=inputs["baseline_qiskit"],
            fm_corrected_accounting_json=inputs["fm_accounting"],
            fm_qiskit_sidecar_json=inputs["fm_qiskit"],
            settings_drift_audit_json=inputs["settings_drift"],
            output_dir=tmp_path / "unclosed_report",
            compile_pdf=False,
        )


def test_fails_closed_when_corrected_nfev_partition_does_not_reconcile(
    tmp_path: Path,
) -> None:
    inputs = _synthetic_inputs(tmp_path)
    fm = json.loads(inputs["fm_accounting"].read_text(encoding="utf-8"))
    fm["query_closure"]["corrected_nfev_total"] = 301
    _write_json(inputs["fm_accounting"], fm)

    with pytest.raises(ValueError, match="stored nfev plus correction"):
        report.build_report(
            baseline_label="SR-SNAKE no ordinary novelty",
            baseline_reference_pdf=inputs["reference_pdf"],
            baseline_source_json=inputs["baseline_source"],
            baseline_accounting_audit_json=inputs["baseline_accounting"],
            baseline_evidence_audit_json=inputs["baseline_evidence"],
            baseline_qiskit_sidecar_json=inputs["baseline_qiskit"],
            fm_corrected_accounting_json=inputs["fm_accounting"],
            fm_qiskit_sidecar_json=inputs["fm_qiskit"],
            settings_drift_audit_json=inputs["settings_drift"],
            output_dir=tmp_path / "bad_nfev_report",
            compile_pdf=False,
        )
