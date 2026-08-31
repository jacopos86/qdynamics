"""Results-PDF-first promotion contracts."""

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.time_dynamics.diagnostics.promotion_contract import (
    PromotionCoverageError,
    manuscript_rows_from_results_pdf,
    planned_tuple_id,
    tuple_id,
    verified_results_pdf_tuples,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": _sha256(path)}


def _coverage_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    ledger = tmp_path / "ledger.json"
    _write_json(ledger, {"records": []})
    exact = tmp_path / "exact" / "manifest.json"
    _write_json(
        exact,
        {
            "schema": "paper_ii_exact_reference_store_v1",
            "seed_sha256": "seed-digest",
            "times": [index * 0.04 for index in range(501)],
        },
    )
    figure = tmp_path / "tuple.png"
    figure.write_bytes(b"figure")
    fragment = tmp_path / "promotion_preview_sections.tex"
    fragment.write_text("preview", encoding="utf-8")
    results_pdf = tmp_path / "paper_ii_staged_report.pdf"
    results_pdf.write_bytes(b"results-pdf")

    rows: dict[str, dict[str, object]] = {}
    for method in ("exchange", "avqds"):
        run = tmp_path / method / "t20" / "run.json"
        screen = tmp_path / method / "t5" / "run.json"
        cost = tmp_path / method / "t20" / "terminal_qiskit_cost.json"
        _write_json(run, {"method": method, "horizon": "t20"})
        _write_json(screen, {"method": method, "horizon": "t5"})
        _write_json(cost, {"method": method})
        rows[method] = {
            "method": method,
            "status": "accuracy_qualified",
            "run_path": str(run),
            "run_sha256": _sha256(run),
            "screen_run_path": str(screen),
            "screen_run_sha256": _sha256(screen),
            "cost_path": str(cost),
            "cost_sha256": _sha256(cost),
            "mean_infidelity": 2.0e-6,
            "mean_abs_energy_error": 3.0e-5,
            "mean_abs_doublon_error": 4.0e-5,
            "mean_abs_charge_imbalance_error": 5.0e-5,
            "time_step_count": 123,
            "n_theta": 40,
            "graph": [100, 200, 300],
        }
    pair: dict[str, object] = {
        "regime": "hh_snake_nph1",
        "drive": "weakslow",
        "controller": "state_motion_1e-3_plus_parameter_5e-5",
        "horizon": "t20",
        "source_comparison_fingerprint": "comparison-digest",
        "rows": rows,
        "exact_reference": _artifact(exact),
        "figure": _artifact(figure),
        "figure_sources": [_artifact(exact)],
    }
    pair["tuple_id"] = tuple_id(pair)
    planned: dict[str, object] = {
        "regime": pair["regime"],
        "drive": pair["drive"],
        "controller": pair["controller"],
        "horizon": "t20",
        "exact_reference": pair["exact_reference"],
        "methods": {
            method: {
                "paper_status": "qualified",
                "pending_reason": "",
                "latest_evidence_status": "accuracy_qualified",
                "latest_row": rows[method],
            }
            for method in ("exchange", "avqds")
        },
    }
    planned["tuple_id"] = planned_tuple_id(planned)
    preview = {
        "schema": "paper_ii_promotion_preview_v4",
        "ledger": {
            **_artifact(ledger),
            "record_count": 1,
        },
        "pair_count": 1,
        "pairs": [pair],
        "planned_tuple_count": 1,
        "planned_tuples": [planned],
        "tex_fragment": _artifact(fragment),
    }
    preview_path = tmp_path / "promotion_preview.json"
    _write_json(preview_path, preview)
    report = {
        "schema": "paper_ii_staged_campaign_report_v1",
        "source_ledger": str(ledger),
        "source_ledger_sha256": _sha256(ledger),
        "ledger_record_count": 1,
        "pdf": str(results_pdf),
        "pdf_sha256": _sha256(results_pdf),
        "promotion_preview": {
            "receipt": str(preview_path),
            "receipt_sha256": _sha256(preview_path),
            "pair_count": 1,
            "planned_tuple_count": 1,
        },
    }
    report_path = tmp_path / "paper_ii_staged_report.json"
    _write_json(report_path, report)
    return preview_path, report_path, preview


def _rewire_report(report_path: Path, preview_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["promotion_preview"]["receipt_sha256"] = _sha256(preview_path)
    _write_json(report_path, report)


def test_verified_results_pdf_tuple_preserves_time_step_controller(tmp_path) -> None:
    preview_path, report_path, _preview = _coverage_fixture(tmp_path)

    tuples = verified_results_pdf_tuples(
        preview_receipt_path=preview_path,
        report_manifest_path=report_path,
        workspace_root=tmp_path,
    )
    rows = manuscript_rows_from_results_pdf(tuples)

    assert len(tuples) == 1
    assert tuples[0].horizon == "t20"
    assert tuples[0].controller == "state_motion_1e-3_plus_parameter_5e-5"
    assert {row["method"] for row in rows} == {"apm", "avqds"}
    assert {row["controller"] for row in rows} == {
        "state_motion_1e-3_plus_parameter_5e-5"
    }
    assert len({row["results_pdf_tuple_id"] for row in rows}) == 1


def test_completed_unqualified_arm_is_a_visible_candidate(tmp_path) -> None:
    preview_path, report_path, preview = _coverage_fixture(tmp_path)
    slot = preview["planned_tuples"][0]["methods"]["exchange"]
    slot["paper_status"] = "candidate"
    slot["pending_reason"] = "accuracy_gate_pending_threshold_extension"
    slot["latest_evidence_status"] = "requires_threshold_extension"
    slot["latest_row"]["status"] = "requires_threshold_extension"
    preview["planned_tuples"][0]["tuple_id"] = planned_tuple_id(
        preview["planned_tuples"][0]
    )
    _write_json(preview_path, preview)
    _rewire_report(report_path, preview_path)

    tuples = verified_results_pdf_tuples(
        preview_receipt_path=preview_path,
        report_manifest_path=report_path,
        workspace_root=tmp_path,
    )
    rows = manuscript_rows_from_results_pdf(tuples)

    apm = next(row for row in rows if row["method"] == "apm")
    assert apm["paper_status"] == "candidate"
    assert apm["mean_infidelity"] == 2.0e-6
    assert apm["mean_abs_energy_error"] == 3.0e-5
    assert apm["mean_abs_doublon_error"] == 4.0e-5
    assert apm["mean_abs_charge_imbalance_error"] == 5.0e-5


def test_terminal_unresolved_arm_retains_its_last_numerical_evidence(tmp_path) -> None:
    preview_path, report_path, preview = _coverage_fixture(tmp_path)
    slot = preview["planned_tuples"][0]["methods"]["exchange"]
    slot["paper_status"] = "unresolved"
    slot["state_reason"] = "ladder_or_safeguard_exhausted"
    slot["latest_evidence_status"] = "safeguard_tripped"
    slot["latest_row"]["status"] = "requires_threshold_extension"
    preview["planned_tuples"][0]["tuple_id"] = planned_tuple_id(
        preview["planned_tuples"][0]
    )
    _write_json(preview_path, preview)
    _rewire_report(report_path, preview_path)

    tuples = verified_results_pdf_tuples(
        preview_receipt_path=preview_path,
        report_manifest_path=report_path,
        workspace_root=tmp_path,
    )
    rows = manuscript_rows_from_results_pdf(tuples)

    apm = next(row for row in rows if row["method"] == "apm")
    assert apm["paper_status"] == "unresolved"
    assert apm["state_reason"] == "ladder_or_safeguard_exhausted"
    assert apm["mean_infidelity"] == 2.0e-6


def test_promotion_rejects_tuple_without_time_step_controller(tmp_path) -> None:
    preview_path, report_path, preview = _coverage_fixture(tmp_path)
    preview["pairs"][0]["controller"] = ""
    _write_json(preview_path, preview)
    _rewire_report(report_path, preview_path)

    with pytest.raises(PromotionCoverageError, match="controller"):
        verified_results_pdf_tuples(
            preview_receipt_path=preview_path,
            report_manifest_path=report_path,
            workspace_root=tmp_path,
        )


def test_promotion_rejects_run_changed_after_results_pdf_build(tmp_path) -> None:
    preview_path, report_path, preview = _coverage_fixture(tmp_path)
    apm_run = Path(preview["pairs"][0]["rows"]["exchange"]["run_path"])
    _write_json(apm_run, {"changed": True})

    with pytest.raises(PromotionCoverageError, match="digest mismatch"):
        verified_results_pdf_tuples(
            preview_receipt_path=preview_path,
            report_manifest_path=report_path,
            workspace_root=tmp_path,
        )


def test_promotion_rejects_preview_not_bound_to_current_results_pdf(tmp_path) -> None:
    preview_path, report_path, _preview = _coverage_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["promotion_preview"]["receipt_sha256"] = "stale"
    _write_json(report_path, report)

    with pytest.raises(PromotionCoverageError, match="digest mismatch"):
        verified_results_pdf_tuples(
            preview_receipt_path=preview_path,
            report_manifest_path=report_path,
            workspace_root=tmp_path,
        )
