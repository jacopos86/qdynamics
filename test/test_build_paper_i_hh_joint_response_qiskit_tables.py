from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from pipelines.reporting import build_paper_i_hh_joint_response_qiskit_tables as report


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reference_row(regime: str, method: str, base: int) -> dict[str, Any]:
    row = {
        "regime": regime,
        "method": method,
        "method_display": {"snake": "SNAKE", "geo": "Geo", "append": "Append"}[method],
        "k_pl": base,
        "N2q": base + 10,
        "D2q": base + 20,
        "Dc": base + 30,
        "S_alg": base + 40,
        "source_json": f"locked/{regime}/{method}.json",
        "source_sha256": f"{base:064x}"[-64:],
        "validation": {"locked": True},
    }
    if method == "snake":
        row["abs_delta_e"] = base / 1000.0
    else:
        row["table_abs_delta_e"] = base / 1000.0
        row["plot_marker_abs_delta_e"] = base / 900.0
        row["role"] = "current_paper_i"
    return row


def _reference_fixture(path: Path) -> dict[str, Any]:
    corrected: list[dict[str, Any]] = []
    comparators: list[dict[str, Any]] = []
    for regime_index, spec in enumerate(report.REGIME_SPECS, start=1):
        corrected.append(_reference_row(spec.regime, "snake", regime_index * 10 + 1))
        comparators.append(_reference_row(spec.regime, "geo", regime_index * 10 + 2))
        comparators.append(_reference_row(spec.regime, "append", regime_index * 10 + 3))
    payload = {
        "schema": "synthetic_locked_reference_v1",
        "corrected_and_snake_rows": corrected,
        "current_paper_i_comparator_rows": comparators,
    }
    _write_json(path, payload)
    return payload


def _campaign_fixture(root: Path) -> None:
    errors = [0.9, 0.2, 0.105, 0.1, 0.099]
    for index, spec in enumerate(report.REGIME_SPECS, start=1):
        result = {
            "summary": {"success": True},
            "settings": {
                "problem": "hh",
                "u": float(index),
                "g_ep": 0.35,
                "omega0": 1.0,
                "n_ph_max": 2,
                "static_route_id": "route_a",
                "static_meta_feature_profile": "paper_i_production_v1",
            },
            "adapt_vqe": {
                "success": True,
                "history_checkpoint_complete": True,
                "method": "static_family_native_adapt_phase3",
                "history": [
                    {
                        "delta_abs_current": value,
                        "energy_after_opt": -float(index) - value,
                        "selected_op": f"op_{position}",
                        "selected_position": position - 1,
                    }
                    for position, value in enumerate(errors, start=1)
                ],
            },
        }
        _write_json(root / spec.campaign_dir / "result.json", result)
        _write_json(
            root / spec.campaign_dir / "plan.json",
            {
                "scientific_settings_hash": f"hash-{spec.regime}",
                "scientific_settings": {
                    "regime": {"u": float(index), "lambda": 0.25, "n_ph_work": 2},
                    "route_a_invocation": {
                        "route_id": "route_a",
                        "profile": "canonical",
                        "optimizer": {
                            "inner_optimizer": "POWELL",
                            "maxiter": 50,
                            "scipy_maxfev": 200,
                        },
                        "shortlists": {
                            "phase1_size": 32,
                            "phase2_size": 24,
                            "child_phase1_size": 32,
                            "child_phase2_size": 25,
                        },
                        "mechanisms": {
                            "phase3_candidate_population": "global_child_only_after_phase2_v1",
                            "batch_selection_mode": "combinatorial_reduced_plane",
                            "batch_size_cap": 2,
                            "batch_search_pool_size": 15,
                            "joint_batch_context_mode": "full_ansatz_v1",
                        },
                    }
                },
            },
        )


def _fake_sidecar_builder(
    *,
    result_json: Path,
    history_position: int,
    output_json: Path,
    threshold: float,
) -> dict[str, Any]:
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    error = payload["adapt_vqe"]["history"][history_position - 1]["delta_abs_current"]
    sidecar = {
        "schema": "paper_i_selected_prefix_qiskit_cost_sidecar_v1",
        "source_result_path": str(result_json),
        "history_position": history_position,
        "k_pl": history_position,
        "threshold_reference": threshold,
        "primary_error_at_prefix": error,
        "compiled_resource_qiskit_validated": True,
        "compiled_circuit_stats_status": "ok",
        "compile_convention": report.QISKIT_COMPILE_CONVENTION,
        "compile_convention_expected": report.QISKIT_COMPILE_CONVENTION,
        "compiled_count_2q_total": 100 + history_position,
        "compiled_depth_2q_total": 200 + history_position,
        "compiled_depth_total": 300 + history_position,
        "replay": {"replayed_operator_count": 2 * history_position},
        "mechanism_formula_S": 400 + history_position,
        "mechanism_formula_status": "ok_test_formula_v1",
        "instrumented_runtime_S": 900 + history_position,
        "instrumented_runtime_status": "ok",
    }
    _write_json(output_json, sidecar)
    return sidecar


def _fake_compile_latex(tex_path: Path) -> Path:
    pdf_path = tex_path.with_suffix(".pdf")
    pdf_path.write_bytes(b"%PDF-1.4\n% synthetic test PDF\n")
    return pdf_path


def test_plateau_selection_uses_first_row_within_ten_percent_of_minimum() -> None:
    payload = {
        "adapt_vqe": {
            "history": [
                {"delta_abs_current": value}
                for value in (0.9, 0.2, 0.105, 0.1, 0.099)
            ]
        }
    }

    selection = report.select_plateau_prefix(payload)

    assert selection.trajectory_minimum == pytest.approx(0.099)
    assert selection.threshold == pytest.approx(0.1089)
    assert selection.history_position == 3
    assert selection.k_pl == 3
    assert selection.error == pytest.approx(0.105)
    assert selection.error_field == "delta_abs_current"


def test_plateau_selection_uses_terminal_winner_error_for_final_prefix() -> None:
    payload = {
        "adapt_vqe": {
            "abs_delta_e": 0.08,
            "history": [
                {"delta_abs_current": 0.20},
                {"delta_abs_current": 0.10},
            ],
        }
    }

    selection = report.select_plateau_prefix(payload)

    assert selection.history_position == 2
    assert selection.error == pytest.approx(0.08)
    assert selection.error_field == "adapt_vqe.abs_delta_e_terminal_winner"


def test_reference_loader_prefers_corrected_comparator_rows() -> None:
    corrected = []
    fallback = []
    for index, spec in enumerate(report.REGIME_SPECS, start=1):
        corrected.append(_reference_row(spec.regime, "snake", 10 * index + 1))
        corrected.append(_reference_row(spec.regime, "geo", 10 * index + 2))
        corrected.append(_reference_row(spec.regime, "append", 10 * index + 3))
        fallback.append(_reference_row(spec.regime, "geo", 100 * index + 2))
        fallback.append(_reference_row(spec.regime, "append", 100 * index + 3))
    selected, exact = report._load_reference_rows(
        {
            "corrected_and_snake_rows": corrected,
            "current_paper_i_comparator_rows": fallback,
        }
    )

    row, collection = selected[("weak-weak", "geo")]
    assert collection == "corrected_and_snake_rows"
    assert row["k_pl"] == 12
    assert exact == corrected


def test_report_rejects_partial_or_failed_result() -> None:
    with pytest.raises(ValueError, match="successful complete result"):
        report._validate_completed_result(
            {
                "summary": {"success": False},
                "adapt_vqe": {
                    "success": False,
                    "history_checkpoint_complete": False,
                },
            },
            source=Path("failed.json"),
        )


def test_build_aligns_rows_preserves_reference_objects_and_orders_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root = tmp_path / "campaign"
    output_dir = tmp_path / "report"
    reference_json = tmp_path / "reference.json"
    reference_payload = _reference_fixture(reference_json)
    _campaign_fixture(campaign_root)
    fallback_spec = report.REGIME_SPECS[0]
    fallback_dir = campaign_root / fallback_spec.campaign_dir
    (fallback_dir / "plan.json").rename(fallback_dir / "plan_new_route.json")
    monkeypatch.setattr(report, "build_selected_prefix_sidecar", _fake_sidecar_builder)
    monkeypatch.setattr(report, "compile_latex", _fake_compile_latex)

    payload = report.build(
        campaign_root=campaign_root,
        output_dir=output_dir,
        stem="synthetic_joint_response_tables",
        reference_json=reference_json,
    )

    rows = payload["rows"]
    assert [(row["regime"], row["method"]) for row in rows] == [
        (spec.regime, method)
        for spec in report.REGIME_SPECS
        for method in report.METHOD_ORDER
    ]
    for spec in report.REGIME_SPECS:
        current = next(
            row for row in rows if row["regime"] == spec.regime and row["method"] == report.CURRENT_METHOD
        )
        assert current["k_pl"] == 3
        assert current["ansatz_depth"] == 6
        assert current["abs_delta_e"] == pytest.approx(0.105)
        assert current["N2q"] == 103
        assert current["D2q"] == 203
        assert current["Dc"] == 303
        assert current["S"] == 403
        assert current["S_source"] == "mechanism_formula_S"

    expected_exact: list[dict[str, Any]] = []
    for spec in report.REGIME_SPECS:
        expected_exact.append(
            next(
                row
                for row in reference_payload["corrected_and_snake_rows"]
                if row["regime"] == spec.regime and row["method"] == "snake"
            )
        )
        expected_exact.extend(
            next(
                row
                for row in reference_payload["current_paper_i_comparator_rows"]
                if row["regime"] == spec.regime and row["method"] == method
            )
            for method in ("geo", "append")
        )
    assert payload["paper_i_reference_rows_exact"] == expected_exact

    csv_path = output_dir / "synthetic_joint_response_tables.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [(row["regime"], row["method"]) for row in csv_rows] == [
        (spec.regime, method)
        for spec in report.REGIME_SPECS
        for method in report.METHOD_ORDER
    ]
    assert all(payload["validation"].values())
    fallback_evidence = payload["current_evidence"][fallback_spec.regime]
    assert fallback_evidence["plan_json"].endswith("/plan_new_route.json")
    assert fallback_evidence["route_settings"]["plan_route_a_invocation"]["route_id"] == "route_a"


def test_tex_is_table_only_and_starts_with_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root = tmp_path / "campaign"
    output_dir = tmp_path / "report"
    reference_json = tmp_path / "reference.json"
    _reference_fixture(reference_json)
    _campaign_fixture(campaign_root)
    monkeypatch.setattr(report, "build_selected_prefix_sidecar", _fake_sidecar_builder)
    monkeypatch.setattr(report, "compile_latex", _fake_compile_latex)

    report.build(
        campaign_root=campaign_root,
        output_dir=output_dir,
        stem="table_only",
        reference_json=reference_json,
    )
    source = (output_dir / "table_only.tex").read_text(encoding="ascii")

    document_start = source.index(r"\begin{document}")
    manifest_start = source.index("Normalized parameter and provenance manifest")
    report_tables_start = source.index("Paper-I-style selected-prefix resource tables")
    assert document_start < manifest_start < report_tables_start
    assert r"\includegraphics" not in source
    assert "graphicx" not in source
    assert "tabularx" not in source
    assert source.count(r"Method & $k_{\rm pl}$") == len(report.REGIME_SPECS) == 6
    assert "Joint-response SNAKE" in source
    assert "Paper-I SNAKE" in source
    assert "Geo-ADAPT" in source
    assert "Append-ADAPT" in source


def test_selected_prefix_replay_failure_is_contextual_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_root = tmp_path / "campaign"
    output_dir = tmp_path / "report"
    reference_json = tmp_path / "reference.json"
    _reference_fixture(reference_json)
    _campaign_fixture(campaign_root)

    def fail_replay(**_: Any) -> dict[str, Any]:
        raise ValueError(
            "missing Pauli groups for selected labels: "
            "paop_full:paop_disp(site=1)::child_set[1]::legal_projected"
        )

    monkeypatch.setattr(report, "build_selected_prefix_sidecar", fail_replay)
    monkeypatch.setattr(report, "compile_latex", _fake_compile_latex)

    with pytest.raises(report.SelectedPrefixCompilationError) as caught:
        report.build(
            campaign_root=campaign_root,
            output_dir=output_dir,
            stem="must_not_exist",
            reference_json=reference_json,
        )

    message = str(caught.value)
    assert "regime=weak-weak" in message
    assert "history_position=3" in message
    assert "missing Pauli groups" in message
    assert "legal_projected" in message
    assert not (output_dir / "must_not_exist.tex").exists()
    assert not (output_dir / "must_not_exist.pdf").exists()
