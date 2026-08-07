from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.reporting import build_paper_i_hh_joint_response_six_regime_overlay as report


def test_repaired_l25_falls_back_to_prior_report_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pipelines.reporting import (
        build_paper_i_hh_joint_response_six_regime_overlay_l25_repaired as repaired,
    )

    monkeypatch.setattr(
        repaired,
        "_load_evidence",
        lambda _output_dir: (_ for _ in ()).throw(FileNotFoundError("cleaned")),
    )
    regimes = []
    for index, regime in enumerate(report.REGIME_ORDER):
        regimes.append(
            {
                "regime": regime,
                "curves": {
                    "jr_l25": {
                        "points": [{"k": 0, "error": 1.0}, {"k": 1, "error": 0.1}],
                        "marker_k": 1,
                        "marker_error": 0.1,
                        "source_json": f"cleaned/{regime}/result.json",
                        "source_sha256": f"{index + 1:064x}",
                    }
                },
                "resource_table_rows": [
                    {
                        "regime": regime,
                        "method": "repaired_l25_snake",
                        "N2q": index + 1,
                    }
                ],
            }
        )
    output_dir = tmp_path / "stage"
    output_dir.mkdir()
    prior_report = tmp_path / "immutable_prior_overlay.json"
    prior_report.write_text(
        json.dumps(
            {
                "repaired_l25_campaign": {
                    "schema": "repaired-v1",
                    "status": "retrieved_6_of_6_r15_capped",
                    "cluster": 8775444,
                    "scientific_contract_hash": "a" * 64,
                    "execution_profile": "wave11_legal_fixed_policy_v1",
                    "policy": {"batch_search_pool_size": 25},
                },
                "pages": {"jr_policies": {"regimes": regimes}},
            }
        ),
        encoding="utf-8",
    )

    evidence = report._repaired_l25_evidence(
        output_dir,
        prior_report_json=prior_report,
    )

    assert evidence["cluster"] == 8775444
    assert [row["regime"] for row in evidence["regimes"]] == list(
        report.REGIME_ORDER
    )
    assert evidence["regimes"][0]["resource_table_rows"][0]["N2q"] == 1
    assert evidence["source_recovery"]["mode"] == "prior_report_provenance_v1"
    assert evidence["source_recovery"]["report_sha256"] == report._sha256(
        prior_report
    )
    assert Path(evidence["source_recovery"]["report_json"]).resolve() == (
        prior_report.resolve()
    )


def test_history_curve_includes_reference_state_round_zero(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.04,
                    "history": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.2},
                        {"energy_before_opt": -0.8, "delta_abs_current": 0.05},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    curve = report._history_curve(path, role="current")

    assert curve.points == ((0, 1.5), (1, 0.2), (2, 0.04))
    assert curve.marker_k == 2
    assert curve.marker_error == pytest.approx(0.04)


def test_history_curve_accepts_selected_prefix_marker_override(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.04,
                    "history": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.2},
                        {"energy_before_opt": -0.8, "delta_abs_current": 0.05},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    curve = report._history_curve(path, role="current", marker_k=1, marker_error=0.2)

    assert curve.marker_k == 1
    assert curve.marker_error == pytest.approx(0.2)


def test_history_curve_accepts_complete_checkpoint_tail(tmp_path: Path) -> None:
    path = tmp_path / "current.json"
    path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.04,
                    "history_count": 2,
                    "history_tail_count": 2,
                    "history_tail": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.2},
                        {"energy_before_opt": -0.8, "delta_abs_current": 0.05},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    curve = report._history_curve(path, role="diagnostic")

    assert curve.points == ((0, 1.5), (1, 0.2), (2, 0.04))


def test_stitched_history_curve_offsets_continuation_rounds(tmp_path: Path) -> None:
    paths = []
    for index, (before, errors, terminal) in enumerate(
        (
            (0.5, [0.2, 0.1], 0.1),
            (-0.9, [0.05], 0.05),
            (-0.95, [0.02, 0.01], 0.009),
        )
    ):
        path = tmp_path / f"segment-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "adapt_vqe": {
                        "exact_gs_energy": -1.0,
                        "abs_delta_e": terminal,
                        "history": [
                            {
                                "energy_before_opt": before if row_index == 0 else -0.9,
                                "delta_abs_current": error,
                            }
                            for row_index, error in enumerate(errors)
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)

    curve = report._stitched_history_curve(paths, role="l10_live")

    assert curve.points == (
        (0, 1.5),
        (1, 0.2),
        (2, 0.1),
        (3, 0.05),
        (4, 0.02),
        (5, 0.009),
    )
    assert curve.marker_k == 5
    assert [segment["controller_round_offset"] for segment in curve.source_segments] == [0, 2, 3]


def test_stitched_query_work_quarantines_sidecar_without_primitive_ids(
    tmp_path: Path,
) -> None:
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.1,
                    "history": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.1}
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_work_sidecar.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "query_work_status": "ok",
                "query_work_scope": "winner_lineage_terminal",
                "query_work_total": 15.0,
                "query_work_components": {
                    "N_H_outer": 1.0,
                    "N_H_refit": 2.0,
                    "N_grad": 4.0,
                    "N_metric": 8.0,
                },
            }
        ),
        encoding="utf-8",
    )

    stitched = report._stitched_winning_lineage_query_work((result_path,))

    assert stitched["schema"] == "jr_snake_stitched_query_accounting_v3"
    assert stitched["status"] == "unavailable_raw_occurrence_stitching"
    assert stitched["S_alg"] is None
    assert stitched["components"] is None
    assert stitched["unique_primitive_union_validated"] is False
    assert stitched["legacy_proxy"]["S_alg"] == 15.0
    assert stitched["legacy_proxy"]["components"] == {
        "S_alg_N_H_outer_eval": 1.0,
        "S_alg_N_H_refit_eval": 2.0,
        "S_alg_N_grad_probe": 4.0,
        "S_alg_N_metric_probe": 8.0,
        "S_alg_N_other_quantum": 0.0,
    }
    assert stitched["segments"][0]["source_kind"] == "adjacent_query_work_sidecar"
    assert report._stitched_query_resource_override(stitched) == (
        None,
        "unavailable_raw_occurrence_stitching",
        None,
    )


def _write_stitched_estimator_segment(
    root: Path,
    *,
    name: str,
    calls: list[tuple[str, str, str]],
    winning_branch_id: str,
) -> Path:
    from pipelines.static_adapt.estimator_call_ledger import (
        EstimatorCallKey,
        EstimatorCallLedger,
    )

    segment_dir = root / name
    segment_dir.mkdir()
    result_path = segment_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.1,
                    "history": [
                        {
                            "energy_before_opt": 0.5,
                            "energy_after_opt": -0.9,
                            "delta_abs_current": 0.1,
                            "branch_id": winning_branch_id,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    ledger = EstimatorCallLedger()
    for label, component, branch_id in calls:
        ledger.record_call(
            EstimatorCallKey(
                projective_state_fingerprint=f"state:{label}",
                hamiltonian_fingerprint="hamiltonian:shared",
                backend_fingerprint="backend:exact",
                precision_contract="precision:exact",
                primitive_kind=(
                    "energy" if component.startswith("N_H") else "coordinate_gradient"
                ),
                observable_or_formula_identity=f"observable:{label}",
            ),
            component=component,
            consumer_scope=f"scope:{name}",
            branch_id=branch_id,
        )
    all_summary = ledger.summary()
    winning = ledger.summary(
        branch_ids=[winning_branch_id], include_unbranched=True
    )
    all_ids = set(all_summary["primitive_ids"])
    winning_ids = set(winning["primitive_ids"])
    discarded_ids = all_ids.difference(winning_ids)
    all_components = all_summary["component_by_primitive_id"]
    discarded_components = {
        component: sum(
            all_components[primitive_id] == component
            for primitive_id in discarded_ids
        )
        for component in ("N_H_outer", "N_H_refit", "N_grad", "N_metric")
    }
    (segment_dir / "estimator_call_ledger.json").write_text(
        json.dumps(
            {
                "schema": "paper_i_estimator_call_ledger_sidecar_v1",
                "accounting": {
                    "complete": True,
                    "winning_branch_ids": [winning_branch_id],
                    "winning_lineage": winning,
                    "all_branch_search_work": all_summary,
                    "discarded_branch_only_by_unique_set_difference": {
                        "S_alg": len(discarded_ids),
                        "primitive_ids": sorted(discarded_ids),
                        "components": discarded_components,
                    },
                },
                "ledger": ledger.to_payload(),
                "adapt_success": True,
                "adapt_error": None,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result_path


def test_stitched_query_work_unions_primitive_ids_across_continuation_boundary(
    tmp_path: Path,
) -> None:
    first = _write_stitched_estimator_segment(
        tmp_path,
        name="first",
        winning_branch_id="winner-1",
        calls=[
            ("a", "N_H_outer", "winner-1"),
            ("boundary", "N_H_refit", "winner-1"),
            ("discard-1", "N_grad", "discard-1"),
        ],
    )
    second = _write_stitched_estimator_segment(
        tmp_path,
        name="second",
        winning_branch_id="winner-2",
        calls=[
            # Same physical primitive, new consumer component and branch label.
            ("boundary", "N_H_outer", "winner-2"),
            ("c", "N_grad", "winner-2"),
            # Reusing a prior winner in a discarded branch must not become
            # discarded-only scientific work.
            ("a", "N_H_outer", "discard-2"),
            ("discard-2", "N_metric", "discard-2"),
        ],
    )

    stitched = report._stitched_winning_lineage_query_work((first, second))

    assert stitched["schema"] == "jr_snake_stitched_query_accounting_v3"
    assert stitched["status"] == "unavailable_raw_occurrence_stitching"
    assert stitched["unique_primitive_union_validated"] is True
    assert stitched["S_alg"] is None
    assert stitched["S_unique"] == 3
    assert stitched["S_unique_components"] == {
        "S_alg_N_H_outer_eval": 1,
        "S_alg_N_H_refit_eval": 1,
        "S_alg_N_grad_probe": 1,
        "S_alg_N_metric_probe": 0,
        "S_alg_N_other_quantum": 0,
    }
    assert stitched["continuation_boundary_deduplication"][
        "deduplicated_boundary_primitive_count"
    ] == 1
    assert len(
        stitched["continuation_boundary_deduplication"][
            "winning_cross_component_reuse_primitive_ids"
        ]
    ) == 1
    discarded = stitched["discarded_branch_operational_overhead"]
    assert discarded["S_unique"] == 2
    assert len(discarded["primitive_ids"]) == 2
    assert set(discarded["primitive_ids"]).isdisjoint(
        stitched["winning_primitive_ids"]
    )
    assert report._stitched_query_resource_override(stitched) == (
        None,
        "unavailable_raw_occurrence_stitching",
        None,
    )

    declaration_only = dict(stitched)
    declaration_only.pop("winning_primitive_ids")
    assert report._stitched_query_resource_override(declaration_only) == (
        None,
        "unavailable_raw_occurrence_stitching",
        None,
    )


def test_terminal_estimator_summary_is_rebuilt_from_winning_branch_ids(
    tmp_path: Path,
) -> None:
    result_path = _write_stitched_estimator_segment(
        tmp_path,
        name="terminal-lineage",
        winning_branch_id="winner",
        calls=[
            ("winning", "N_H_outer", "winner"),
            ("discarded", "N_H_refit", "discarded"),
        ],
    )
    sidecar_path = result_path.with_name("estimator_call_ledger.json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar["accounting"]["winning_lineage"] = sidecar["accounting"][
        "all_branch_search_work"
    ]
    sidecar_path.write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="does not match authenticated winning_branch_ids",
    ):
        report._verified_segment_estimator_ledger(
            path=result_path,
            payload=json.loads(result_path.read_text(encoding="utf-8")),
        )


def _write_fm_checkpoint_estimator_segment(
    root: Path,
    *,
    include_incomplete_terminal: bool,
) -> Path:
    from pipelines.static_adapt.estimator_call_ledger import (
        EstimatorCallKey,
        EstimatorCallLedger,
    )

    segment_dir = root / "fm-checkpoint"
    segment_dir.mkdir()
    ledger = EstimatorCallLedger()
    for label, branch_id in (
        ("beam-winner", "beam_branch:7"),
        ("single-frontier-shared", "single_frontier:0"),
        ("beam-loser", "beam_branch:8"),
    ):
        ledger.record_call(
            EstimatorCallKey(
                projective_state_fingerprint=f"state:{label}",
                hamiltonian_fingerprint="hamiltonian:shared",
                backend_fingerprint="backend:exact",
                precision_contract="precision:exact",
                primitive_kind="energy",
                observable_or_formula_identity=f"observable:{label}",
            ),
            component="N_H_outer",
            consumer_scope="scope:fm-checkpoint",
            branch_id=branch_id,
        )
    ledger_payload = ledger.to_payload()
    checkpoint_sidecar = segment_dir / "checkpoint-ledger.json"
    checkpoint_sidecar.write_text(
        json.dumps(
            {
                "schema": "paper_i_estimator_call_ledger_checkpoint_sidecar_v1",
                "checkpoint": {
                    "reason": "iteration_done",
                    "depth": 1,
                    "branch_id": 7,
                    "parent_branch_id": None,
                    "current_round_finalized": True,
                },
                "ledger": ledger_payload,
                "ledger_fingerprint": ledger_payload["ledger_fingerprint"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    result_path = segment_dir / "current.json"
    result_path.write_text(
        json.dumps(
            {
                "route_id": report.FM_LIVE_ROUTE_ID,
                "adapt_vqe": {
                    "adapt_reoptimization_route": report.FM_LIVE_ROUTE_ID,
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.1,
                    "history": [
                        {
                            "energy_before_opt": 0.5,
                            "energy_after_opt": -0.9,
                            "delta_abs_current": 0.1,
                            "branch_id": 7,
                        }
                    ],
                    "estimator_call_ledger_checkpoint": {
                        "schema": (
                            "paper_i_estimator_call_ledger_checkpoint_pointer_v1"
                        ),
                        "enabled": True,
                        "status": "complete",
                        "path": checkpoint_sidecar.name,
                        "sha256": report._sha256(checkpoint_sidecar),
                        "current_round_finalized": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    if include_incomplete_terminal:
        (segment_dir / "estimator_call_ledger.json").write_text(
            json.dumps(
                {
                    "schema": "paper_i_estimator_call_ledger_sidecar_v1",
                    "accounting": {
                        "complete": False,
                        "winning_branch_ids": ["beam_branch:7"],
                    },
                    "ledger": ledger_payload,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    return result_path


def test_fm_checkpoint_lineage_adds_beam_and_single_frontier_aliases(
    tmp_path: Path,
) -> None:
    result_path = _write_fm_checkpoint_estimator_segment(
        tmp_path,
        include_incomplete_terminal=False,
    )

    exact, blocker = report._verified_segment_estimator_ledger(
        path=result_path,
        payload=json.loads(result_path.read_text(encoding="utf-8")),
    )

    assert blocker is None
    assert exact is not None
    assert exact["source_kind"] == (
        "completed_round_estimator_call_ledger_checkpoint"
    )
    assert exact["winning_summary"]["S_unique"] == 2
    assert exact["all_summary"]["S_unique"] == 3
    assert len(exact["discarded_ids"]) == 1


def test_incomplete_terminal_uses_authenticated_checkpoint_explicitly(
    tmp_path: Path,
) -> None:
    result_path = _write_fm_checkpoint_estimator_segment(
        tmp_path,
        include_incomplete_terminal=True,
    )

    exact, blocker = report._verified_segment_estimator_ledger(
        path=result_path,
        payload=json.loads(result_path.read_text(encoding="utf-8")),
    )

    assert blocker is None
    assert exact is not None
    assert exact["terminal_fallback_used"] is True
    assert exact["terminal_fallback_blocker"] == (
        "terminal_estimator_accounting_not_complete"
    )
    stitched = report._stitched_winning_lineage_query_work((result_path,))
    assert stitched["status"] == "unavailable_raw_occurrence_stitching"
    assert stitched["segments"][0]["terminal_fallback_used"] is True
    assert stitched["segments"][0]["terminal_fallback_blocker"] == (
        "terminal_estimator_accounting_not_complete"
    )


def test_incomplete_terminal_does_not_accept_unauthenticated_checkpoint(
    tmp_path: Path,
) -> None:
    result_path = _write_fm_checkpoint_estimator_segment(
        tmp_path,
        include_incomplete_terminal=True,
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["adapt_vqe"]["estimator_call_ledger_checkpoint"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="sidecar hash mismatch"):
        report._verified_segment_estimator_ledger(
            path=result_path,
            payload=payload,
        )


def test_exact_stitched_validation_allows_independent_component_charging(
    tmp_path: Path,
) -> None:
    first = _write_stitched_estimator_segment(
        tmp_path,
        name="discarded-first",
        winning_branch_id="winner-1",
        calls=[
            ("first-winner", "N_grad", "winner-1"),
            ("reused", "N_H_refit", "discarded-1"),
        ],
    )
    second = _write_stitched_estimator_segment(
        tmp_path,
        name="winning-later",
        winning_branch_id="winner-2",
        calls=[("reused", "N_H_outer", "winner-2")],
    )

    stitched = report._stitched_winning_lineage_query_work((first, second))
    reused_id = next(
        primitive_id
        for primitive_id in stitched["winning_primitive_ids"]
        if stitched["winning_component_by_primitive_id"][primitive_id]
        == "N_H_outer"
    )
    assert stitched["all_component_by_primitive_id"][reused_id] == "N_H_refit"
    assert report._stitched_query_resource_override(stitched) == (
        None,
        "unavailable_raw_occurrence_stitching",
        None,
    )


def test_paper_reference_rows_use_corrected_then_current_fallback() -> None:
    payload = {
        "corrected_and_snake_rows": [
            {"regime": "weak-weak", "method": "snake", "source": "corrected"},
        ],
        "current_paper_i_comparator_rows": [
            {"regime": "weak-weak", "method": "snake", "source": "current"},
            {"regime": "weak-weak", "method": "geo", "source": "current"},
        ],
    }
    original_order = report.REGIME_ORDER
    original_methods = report.PAPER_METHODS
    try:
        report.REGIME_ORDER = ("weak-weak",)
        report.PAPER_METHODS = ("snake", "geo")
        rows = report._paper_reference_rows(payload)
    finally:
        report.REGIME_ORDER = original_order
        report.PAPER_METHODS = original_methods

    assert rows[("weak-weak", "snake")]["source"] == "corrected"
    assert rows[("weak-weak", "geo")]["source"] == "current"


def test_resource_rows_require_core_methods_and_preserve_optional_route() -> None:
    rows = []
    for regime in report.REGIME_ORDER:
        for index, method in enumerate(report.RESOURCE_METHODS, start=1):
            rows.append(
                {
                    "regime": regime,
                    "method": method,
                    "method_display": method,
                    "k_pl": index,
                    "abs_delta_e": index / 1000.0,
                    "N2q": index + 10,
                    "D2q": index + 20,
                    "Dc": index + 30,
                    "S": index + 40,
                }
            )
    rows.append(
        {
            "regime": "weak-weak",
            "method": "formal_manifold_snake",
            "method_display": "FM-SNAKE",
            "k_pl": 5,
            "abs_delta_e": 1.0e-5,
            "N2q": 15,
            "D2q": 12,
            "Dc": 50,
            "S": 100,
        }
    )

    grouped = report._resource_rows_by_regime(
        {"validation": {"all_checks": True}, "rows": rows}
    )

    assert [row["method"] for row in grouped["weak-weak"]] == [
        *report.RESOURCE_METHODS,
        "formal_manifold_snake",
    ]
    assert [row["method"] for row in grouped["strong-strong"]] == list(
        report.RESOURCE_METHODS
    )


def test_fm_adapter_adds_valid_complete_row_and_explicit_pending_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_root = tmp_path / "campaign"
    campaign_root.mkdir()
    complete_dir = tmp_path / "complete"
    complete_dir.mkdir()
    result_path = complete_dir / "result.json"
    qiskit_path = complete_dir / "campaign-qiskit.json"
    query_path = complete_dir / "query.json"
    result_path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "success": True,
                    "adapt_reoptimization_route": "formal_manifold_warm_start_v1",
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.02,
                    "history": [
                        {
                            "energy_before_opt": 0.5,
                            "delta_abs_current": 0.02,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    result_hash = report._sha256(result_path)
    qiskit_path.write_text(
        json.dumps(
            {
                "compiled_resource_qiskit_validated": True,
                "source_result_sha256": result_hash,
            }
        ),
        encoding="utf-8",
    )
    query_path.write_text(
        json.dumps(
            {
                "science_valid": True,
                "joint_response_selector_invoked": False,
                "source_result_sha256": result_hash,
                "winning_branch": {"expanded_query_work": 321},
            }
        ),
        encoding="utf-8",
    )
    provenance = {
        "whitening_id": "w",
        "frame_id": "f",
        "logical_range_id": "l",
        "curvature_whitening_id": "w",
        "curvature_frame_id": "f",
        "qbroyd_whitening_id": "w",
        "qbroyd_logical_range_id": "l",
    }
    cells = [
        {
            "cell_id": "complete-cell",
            "status": "complete",
            "scientific_settings_sha256": "a" * 64,
            "paths": {
                "result_json": str(result_path),
                "qiskit_sidecar": str(qiskit_path),
                "query_work_sidecar": str(query_path),
            },
            "evidence": {
                "result_sha256": result_hash,
                "whitening_provenance": provenance,
            },
        },
        {
            "cell_id": "queued-cell",
            "status": "queued",
            "scientific_settings_sha256": "b" * 64,
        },
    ]
    (campaign_root / "pareto_ledger.json").write_text(
        json.dumps({"cells": cells}), encoding="utf-8"
    )

    monkeypatch.setattr(report, "REGIME_ORDER", ("weak-weak", "weak-strong"))
    monkeypatch.setattr(
        report,
        "FM_CELL_ID",
        {"weak-weak": "complete-cell", "weak-strong": "queued-cell"},
    )

    def fake_supplemental(**kwargs: object) -> dict[str, object]:
        sidecar_path = Path(str(kwargs["sidecar_json"]))
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text("{}", encoding="utf-8")
        return {
            "regime": kwargs["regime"],
            "method": kwargs["method"],
            "k_pl": kwargs["history_position"],
            "abs_delta_e": kwargs["expected_error"],
            "N2q": 10,
            "D2q": 8,
            "Dc": 40,
            "S": None,
        }

    monkeypatch.setattr(report, "_supplemental_resource_row", fake_supplemental)

    rows, metadata = report._fm_rows_by_regime(
        campaign_root=campaign_root,
        supplemental_dir=tmp_path / "supplemental",
    )

    assert rows["weak-weak"]["status"] == "complete"
    assert rows["weak-weak"]["curve"].marker_k == 1
    assert rows["weak-weak"]["resource"]["S"] == 321
    assert rows["weak-weak"]["resource"]["coordinate_provenance_checks"] == {
        "curvature_whitening": True,
        "curvature_frame": True,
        "qbroyd_whitening": True,
        "qbroyd_logical_range": True,
    }
    assert rows["weak-strong"]["resource"]["status"] == "queued"
    assert rows["weak-strong"]["resource"]["abs_delta_e"] is None
    assert metadata["pending_regimes"] == ["weak-strong"]


def test_model_tex_builds_five_comparison_pages_and_final_provenance(
    tmp_path: Path,
) -> None:
    tex_path = tmp_path / "report.tex"
    pages = {}
    panel_paths = {}
    for page_key, roles in (
        ("model", ["jr_selected", "fm_qbroyd_off", "snake", "geo", "append"]),
        ("jr_policies", ["jr_selected", "jr_baseline"]),
        ("fm_policies", ["fm_qbroyd_default", "fm_qbroyd_off"]),
        ("paper_i_route4", ["paper_i_route4", "snake"]),
        ("original_route", ["snake", "geo", "append"]),
    ):
        pages[page_key] = {
            "title": page_key,
            "subtitle": "comparison",
            "roles": roles,
            "regimes": [
                {
                    "regime": regime,
                    "display": regime,
                    "resource_table_rows": [
                        {
                            "method": "jr_snake_whitened_l10",
                            "k_pl": 5,
                            "abs_delta_e": 1.0e-4,
                            "N2q": 10,
                            "D2q": 8,
                            "Dc": 40,
                            "S": 100,
                        },
                        {
                            "method": "fm_qbroyd_off",
                            "status": "queued",
                        },
                    ],
                }
                for regime in report.REGIME_ORDER
            ],
        }
        panel_paths[page_key] = {
            regime: tmp_path / f"{page_key}-{regime}.png"
            for regime in report.REGIME_ORDER
        }
    pages["model"]["roles"] = [
        "paper_i_route4_live",
        "paper_i_route4",
        *pages["model"]["roles"],
    ]
    for regime_row in pages["model"]["regimes"]:
        regime_row["resource_table_rows"] = [
            {
                "method": "paper_i_route4_live_checkpoint_snake",
                "status": "running_checkpoint_not_terminal",
                "k_pl": 31,
                "abs_delta_e": 5.3e-3,
                "N2q": None,
                "D2q": None,
                "Dc": None,
                "S": None,
            },
            {
                "method": "paper_i_route4_snake",
                "status": "failed_partial_round21",
                "k_pl": 21,
                "abs_delta_e": 1.9e-2,
                "N2q": 120,
                "D2q": 80,
                "Dc": 400,
                "S": None,
            },
            {
                "method": "jr_snake_whitened_l10",
                "status": "complete",
                "k_pl": 5,
                "abs_delta_e": 1.0e-4,
                "N2q": 10,
                "D2q": 8,
                "Dc": 40,
                "S": 100,
            },
            {"method": "fm_qbroyd_off", "status": "queued"},
        ]
    evidence = {
        "schema": report.SCHEMA,
        "paper_i_reference_json": "reference.json",
        "paper_i_reference_sha256": "c" * 64,
        "formal_manifold_campaign": {
            "ledger_sha256": "d" * 64,
            "pending_regimes": ["weak-strong"],
        },
        "formal_manifold_ablation_campaign": {
            "ledger_json": "fm-ledger.json",
            "ledger_sha256": "d" * 64,
            "completed_variant_cells": 3,
        },
        "formal_manifold_live_status_campaign": {
            "captured_at_local": "2026-07-13 09:54 CDT",
            "status_json": "fm-live-status.json",
            "status_sha256": "e" * 64,
        },
        "jr_l10_campaign": {
            "policy": "L10-B2",
            "regime_rounds": {regime: 9 for regime in report.REGIME_ORDER},
            "selector_exhausted_regimes": ["strong-weak"],
        },
        "repaired_l25_campaign": {
            "cluster": 8775444,
            "status": "retrieved_6_of_6_r15_capped",
        },
        "paper_i_route4_campaign": {
            "completed_regimes": ["weak-weak"],
            "partial_regimes": ["weak-strong"],
            "not_run_regimes": ["strong-strong"],
        },
        "paper_i_route4_live_snapshot_campaign": {
            "captured_at_utc": "2026-07-13T20:58:44Z",
            "manifest_json": "route4-live.json",
            "manifest_sha256": "f" * 64,
        },
        "run_setting_caveat_ledger": [dict(report.RUN_SETTING_LEDGER[0])],
        "pages": pages,
    }
    report._write_model_tex(
        tex_path,
        panel_paths=panel_paths,
        provenance_path=tmp_path / "provenance.json",
        evidence=evidence,
    )
    source = tex_path.read_text(encoding="utf-8")

    assert source.count("\\includegraphics") == 30
    assert source.count("\\newpage") == 5
    assert "Model comparison" not in source
    assert "Human caveats and machine provenance" in source
    assert source.index("Parameter manifest") < source.index("\\includegraphics")
    assert '"page_contract":["parameter_manifest_human_caveats_and_machine_provenance"' in source
    assert "HH model parameters" in source
    assert "n\\_ph\\_max=2" in source
    assert "same-cutoff quantities" in source
    assert "JR-L10" in source
    assert "queued" in source
    assert "adaptive, zero lower bound" in source
    assert "supported-metric whitened eigensolve" in source
    assert '"algorithm_setting_ledger"' in source
    assert '"page_contract"' in source
    assert '"formal_manifold_live_status_campaign"' in source
    assert "FM status endpoints" in source
    assert "2026-07-13 09:54 CDT" in source
    assert "starred rows are last verified checkpoints, not terminal metrics" in source
    assert "SR-SNAKE recovery evidence" in source
    audit = source.split("Selected model & Regime", maxsplit=1)[1].split(
        "\\bottomrule", maxsplit=1
    )[0]
    assert "SR recovery [nonterminal]" in audit
    assert "SR-SNAKE [partial r21]" in audit
    assert "JR-L10" in audit
    assert "FM qB off" not in audit

    validation_path = _write_sr_expanded_chart_whitening_validation(tmp_path)
    whitening_campaign = report._load_sr_expanded_chart_whitening_validation(
        validation_path
    )
    whitening_campaign["support_rank_sequence"] = list(range(1, 31))
    evidence["sr_expanded_chart_whitening_campaign"] = whitening_campaign
    evidence["pages"][report.SR_EXPANDED_CHART_WHITENING_PAGE_KEY] = {
        "page_type": report.SR_EXPANDED_CHART_WHITENING_PAGE_KEY,
        "title": "expanded whitening page",
        "subtitle": "opt-in",
        "campaign": whitening_campaign,
    }
    panel_paths[report.SR_EXPANDED_CHART_WHITENING_PAGE_KEY] = {
        "wide": tmp_path / "expanded-whitening.png"
    }
    report._write_model_tex(
        tex_path,
        panel_paths=panel_paths,
        provenance_path=tmp_path / "provenance.json",
        evidence=evidence,
    )
    opt_in_source = tex_path.read_text(encoding="utf-8")
    assert opt_in_source.count("\\includegraphics") == 31
    assert opt_in_source.count("\\newpage") == 6
    assert opt_in_source.index("paper\\_i\\_route4") < opt_in_source.index(
        "expanded whitening page"
    ) < opt_in_source.index("original\\_route")
    assert "sr_expanded_chart_whitening_weak_weak_diagnostic" in opt_in_source
    assert "Gram matrix and whitening map are rebuilt after every" in opt_in_source
    assert "Support rank, rounds 1--15" in opt_in_source
    assert "Support rank, rounds 16--30" in opt_in_source
    assert "Support rank, rounds 1--30" not in opt_in_source
    assert r"expanded\_\allowbreak{}runtime\_\allowbreak{}projected" in opt_in_source
    assert r"/\allowbreak{}" in opt_in_source
    assert "expanded_runtime_projected_logical_v1" in opt_in_source
    assert str(validation_path) in opt_in_source


def test_partial_route4_resource_row_keeps_compiled_costs_and_unresolved_s() -> None:
    source = report._resource_table_tex(
        [
            {
                "method": "paper_i_route4_snake",
                "status": "failed_partial_round21",
                "k_pl": 21,
                "abs_delta_e": 0.019,
                "N2q": 120,
                "D2q": 80,
                "Dc": 400,
                "S": None,
            }
        ]
    )

    assert "partial r21" in source
    assert "120" in source
    assert "80" in source
    assert "400" in source
    assert "n/a" in source


def test_route4_accounting_summary_drops_large_identity_maps(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "accounting": {
                    "complete": True,
                    "definition": "S_alg = N_H_outer + N_H_refit + N_grad + N_metric",
                    "status": "resolved_from_live_state_keyed_instrumentation",
                    "exact_blockers": [],
                    "winning_lineage": {
                        "N_H_outer": 1,
                        "N_H_refit": 2,
                        "N_grad": 3,
                        "N_metric": 4,
                        "S_alg": 10,
                        "component_by_primitive_id": {"large": "N_grad"},
                    },
                    "discarded_branch_only_by_unique_set_difference": {"S_alg": 7},
                    "all_branch_search_work": {
                        "N_H_outer": 1,
                        "N_H_refit": 4,
                        "N_grad": 5,
                        "N_metric": 7,
                        "S_alg": 17,
                    },
                },
                "adapt_success": True,
                "adapt_error": None,
            }
        ),
        encoding="utf-8",
    )

    summary = report._route4_accounting_summary(ledger)

    assert summary["winning_lineage"]["S_alg"] == 10
    assert summary["discarded_branch_only"]["S_alg"] == 7
    assert "component_by_primitive_id" not in summary["winning_lineage"]


def _write_route4_live_bundle(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_path = tmp_path / "weak_strong_current.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "settings": {
                    "historical_singleton_coordinate_trust_overlay": {
                        "active": True,
                        "whitening_active": True,
                        "adaptive_trust_active": True,
                        "coordinate_solve_policy": (
                            "supported_metric_whitened_eigh_v1"
                        ),
                        "trust_region_update": {
                            "policy": "displacement_calibrated_unbounded_v2"
                        },
                        "phase0_pilot_enabled": False,
                        "phase2_batching_enabled": False,
                        "phase3_batching_enabled": False,
                        "route_a_funnel_active": False,
                        "child_padding_contract": {
                            "projection_active": True,
                            "satisfied": True,
                            "source": "exact_projected_grouped_v1",
                        },
                    }
                },
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "energy": -0.975,
                    "abs_delta_e": 0.025,
                    "ansatz_depth": 2,
                    "branch_id": 8,
                    "parent_branch_id": 7,
                    "history_checkpoint_complete": True,
                    "partial_checkpoint": True,
                    "success": False,
                    "history_count": 2,
                    "history": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.2},
                        {"energy_before_opt": -0.8, "delta_abs_current": 0.025},
                    ],
                },
                "checkpoint": {
                    "complete": False,
                    "depth": 2,
                    "ansatz_depth": 2,
                    "branch_id": 8,
                    "parent_branch_id": 7,
                },
            }
        ),
        encoding="utf-8",
    )
    queue_path = tmp_path / "queue_state.json"
    queue_path.write_text('{"status":"running"}\n', encoding="utf-8")
    command_path = tmp_path / "command.json"
    command_path.write_text(
        json.dumps(
            {
                "argv": [
                    "python",
                    "--phase0-no-pilot",
                    "--phase2-no-batching",
                    "--phase3-no-batching",
                    "--phase3-runtime-split-child-set-symmetry-policy",
                    "hard_guard",
                    "--phase3-runtime-split-child-padding-policy",
                    "exact_projected_grouped_v1",
                    "--historical-singleton-coordinate-solve-policy",
                    "supported_metric_whitened_eigh_v1",
                    "--historical-singleton-trust-region-update-policy",
                    "displacement_calibrated_unbounded_v2",
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )
    normalized_path = tmp_path / "normalized_manifest.json"
    normalized_path.write_text(
        json.dumps(
            {
                "scientific_contract": {
                    "phase0_enabled": False,
                    "phase2_batching_enabled": False,
                    "phase3_batching_enabled": False,
                    "symmetry_policy": "hard_guard",
                    "padding_policy": "exact_projected_grouped_v1",
                    "singleton_subset_size": 1,
                    "coordinate_solve_policy": "supported_metric_whitened_eigh_v1",
                    "trust_region_update_policy": (
                        "displacement_calibrated_unbounded_v2"
                    ),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_lock_path = tmp_path / "source_lock_and_settings_diff.json"
    source_lock_path.write_text('{"status":"pass"}\n', encoding="utf-8")
    snapshot_hash = report._sha256(snapshot_path)
    queue_hash = report._sha256(queue_path)
    manifest_path = tmp_path / "paper_i_route4_live_snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_route4_live_snapshot_bundle_v1",
                "captured_at_utc": "2026-07-13T20:58:44Z",
                "campaign_root": "raw_outputs/route4-recovery",
                "route": "route_4_whitened_adaptive",
                "evidence_status": "running_checkpoint_not_terminal",
                "terminal": False,
                "queue_state_snapshot": queue_path.name,
                "queue_state_snapshot_sha256": queue_hash,
                "queue_state_source": "raw_outputs/route4-recovery/queue_state.json",
                "queue_state_source_sha256_at_capture": queue_hash,
                "source_lock_and_settings_diff_json": str(source_lock_path),
                "source_lock_and_settings_diff_sha256": report._sha256(
                    source_lock_path
                ),
                "policy": {
                    "coordinate_solve": "supported_metric_whitened_eigh_v1",
                    "trust_region_update": "displacement_calibrated_unbounded_v2",
                    "phase0": "off",
                    "phase2_batching": "off",
                    "phase3_batching": "off",
                    "prune_policy": "recoverability_ladder_v1",
                },
                "entries": [
                    {
                        "regime": "weak-strong",
                        "status": "running_checkpoint_not_terminal",
                        "terminal": False,
                        "snapshot_json": snapshot_path.name,
                        "snapshot_sha256": snapshot_hash,
                        "source_current_json": (
                            "raw_outputs/route4-recovery/weak_strong/full/json/current.json"
                        ),
                        "source_current_sha256": snapshot_hash,
                        "command_json": str(command_path),
                        "command_sha256": report._sha256(command_path),
                        "normalized_manifest_json": str(normalized_path),
                        "normalized_manifest_sha256": report._sha256(
                            normalized_path
                        ),
                        "controller_round": 2,
                        "ansatz_depth": 2,
                        "branch_id": 8,
                        "parent_branch_id": 7,
                        "energy": -0.975,
                        "exact_same_cutoff_energy": -1.0,
                        "abs_delta_e": 0.025,
                        "history_checkpoint_complete": True,
                        "source_kind": "immutable_current_json_snapshot",
                        "evidence_relation": (
                            "best_branch_recovery_checkpoint_additive_to_"
                            "preserved_round21_row"
                        ),
                        "pending_fields": ["N2q", "D2q", "Dc", "S_alg"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return snapshot_path, manifest_path


def test_route4_live_manifest_adds_hash_validated_marker_only_pending_row(
    tmp_path: Path,
) -> None:
    _, manifest_path = _write_route4_live_bundle(tmp_path)

    rows, campaign = report._load_paper_i_route4_live_snapshot_manifest(
        manifest_path
    )

    live = rows["weak-strong"]
    curve = report._curve_payload(live["curve"])
    resource = live["resource"]
    assert curve["points"] == []
    assert curve["marker_k"] == 2
    assert curve["marker_error"] == pytest.approx(0.025)
    assert curve["status_endpoint_only"] is True
    assert curve["trajectory_relation"] == "same_route_later_checkpoint"
    assert resource["status"] == "running_checkpoint_not_terminal"
    assert resource["terminal"] is False
    assert resource["exact_same_cutoff_energy"] == pytest.approx(-1.0)
    assert [resource[key] for key in ("N2q", "D2q", "Dc", "S")] == [
        None,
        None,
        None,
        None,
    ]
    assert campaign["schema"] == "paper_i_route4_live_snapshot_bundle_v1"
    assert campaign["endpoint_semantics"] == "marker_only_no_trajectory_interpolation"
    assert campaign["preserved_row_relation"] == "additive_to_preserved_round21_rows"
    assert all(live["provenance"]["command_policy_checks"].values())
    assert all(live["provenance"]["normalized_policy_checks"].values())
    assert all(live["provenance"]["snapshot_overlay_checks"].values())
    assert resource["symmetry_evidence_status"].endswith(
        "checkpoint_leakage_unresolved"
    )
    table = report._resource_table_tex([resource])
    assert "SR recovery [nonterminal]" in table
    assert "2 & 2.50e-02" in table
    assert "pending & pending & pending & pending" in table


def _upgrade_route4_bundle_with_terminal_recovery(
    tmp_path: Path,
) -> Path:
    _, manifest_path = _write_route4_live_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    result_path = tmp_path / "intermediate_strong_result.json"
    result_path.write_text('{"status":"complete"}\n', encoding="utf-8")
    qiskit_path = tmp_path / "intermediate_strong_qiskit.json"
    qiskit_path.write_text(
        json.dumps(
            {
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "compiled_resource_qiskit_validated": True,
                "compiled_count_2q_total": 492,
                "compiled_depth_2q_total": 395,
                "compiled_depth_total": 2014,
                "history_position": 45,
                "logical_operator_count": 43,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    validation_path = tmp_path / "intermediate_strong_validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_hh_route4_round45_validation_v1",
                "status": "validated",
                "controller_rounds": 45,
                "active_ansatz_depth": 43,
                "energy": -0.6238176328457341,
                "exact_same_cutoff_energy": -0.6239104048313422,
                "absolute_error": 9.277198560808664e-05,
                "result_json": str(result_path.resolve()),
                "result_sha256": report._sha256(result_path),
                "qiskit": {
                    "path": str(qiskit_path.resolve()),
                    "sha256": report._sha256(qiskit_path),
                    "N2q": 492,
                    "D2q": 395,
                    "circuit_depth": 2014,
                },
                "leakage": {
                    "tolerance": 1.0e-10,
                    "maximum_sector_leakage": 8.1e-15,
                    "maximum_padding_leakage": 7.8e-15,
                },
                "resume": {"prefix_replay_abs_discrepancy": 4.0e-15},
                "estimator_accounting": {
                    "complete": True,
                    "scope": "continuation_segment_only",
                    "winning_lineage_S_alg": 484694,
                    "cumulative_rounds_1_to_final_S_alg": None,
                    "exact_blockers": [
                        "source_rounds_1_21_state_keyed_ledger_missing_after_failed_run"
                    ],
                },
                "stop_reason": "segment_target_controller_round",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipt_path = tmp_path / "strong_strong_receipt.json"
    receipt_path.write_text('{"cluster_id":8778975}\n', encoding="utf-8")

    live_entry = manifest["entries"][0]
    manifest.update(
        {
            "schema": "paper_i_sr_snake_recovery_snapshot_bundle_v2",
            "stable_family_id": "singleton_response_snake",
            "evidence_status": "mixed_terminal_and_nonterminal_recovery",
            "terminal": None,
            "strong_strong_submission": {
                "status": "submitted_pending_chtc",
                "cluster_id": 8778975,
                "batch_name": "paper-i-hh-sr-ss-r45-20260714-v2",
                "submission_receipt": str(receipt_path),
                "submission_receipt_sha256": report._sha256(receipt_path),
            },
        }
    )
    manifest["entries"].append(
        {
            "regime": "intermediate-strong",
            "status": "validated_terminal_recovery",
            "terminal": True,
            "source_kind": "validated_recovery_endpoint",
            "validation_json": str(validation_path),
            "validation_sha256": report._sha256(validation_path),
            "result_json": str(result_path),
            "result_sha256": report._sha256(result_path),
            "qiskit_json": str(qiskit_path),
            "qiskit_sha256": report._sha256(qiskit_path),
            "command_json": live_entry["command_json"],
            "command_sha256": live_entry["command_sha256"],
            "normalized_manifest_json": live_entry["normalized_manifest_json"],
            "normalized_manifest_sha256": live_entry[
                "normalized_manifest_sha256"
            ],
            "controller_round": 45,
            "ansatz_depth": 43,
            "energy": -0.6238176328457341,
            "exact_same_cutoff_energy": -0.6239104048313422,
            "abs_delta_e": 9.277198560808664e-05,
            "N2q": 492,
            "D2q": 395,
            "Dc": 2014,
            "winning_lineage_S_alg": 484694,
            "S_alg_scope": "continuation_segment_rounds_22_to_45_only",
            "cumulative_rounds_1_to_45_S_alg": None,
            "exact_blockers": [
                "source_rounds_1_21_state_keyed_ledger_missing_after_failed_run"
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_sr_recovery_manifest_mixes_open_running_and_filled_validated_endpoint(
    tmp_path: Path,
) -> None:
    manifest_path = _upgrade_route4_bundle_with_terminal_recovery(tmp_path)

    rows, campaign = report._load_paper_i_route4_live_snapshot_manifest(
        manifest_path
    )

    running_curve = report._curve_payload(rows["weak-strong"]["curve"])
    terminal_curve = report._curve_payload(rows["intermediate-strong"]["curve"])
    terminal = rows["intermediate-strong"]["resource"]
    assert running_curve["status_endpoint_only"] is True
    assert "status_endpoint_only" not in terminal_curve
    assert terminal_curve["marker_k"] == 45
    assert terminal_curve["marker_error"] == pytest.approx(9.277198560808664e-05)
    assert terminal["status"] == "validated_terminal_recovery"
    assert [terminal[key] for key in ("N2q", "D2q", "Dc", "S")] == [
        492,
        395,
        2014,
        484694,
    ]
    assert terminal["S_scope"] == "continuation_segment_rounds_22_to_45_only"
    assert terminal["cumulative_rounds_1_to_45_S_alg"] is None
    assert terminal["exact_blockers"] == [
        "source_rounds_1_21_state_keyed_ledger_missing_after_failed_run"
    ]
    assert terminal["maximum_sector_leakage"] < terminal["leakage_tolerance"]
    assert terminal["maximum_padding_leakage"] < terminal["leakage_tolerance"]
    assert campaign["schema"] == "paper_i_sr_snake_recovery_snapshot_bundle_v2"
    assert campaign["running_regimes"] == ["weak-strong"]
    assert campaign["terminal_regimes"] == ["intermediate-strong"]
    assert campaign["strong_strong_submission"]["cluster_id"] == 8778975
    table = report._resource_table_tex([terminal])
    assert "SR recovery [validated; S r22--45]" in table
    assert "492" in table and "395" in table and "2,014" in table
    assert "484,694" in table


def test_sr_terminal_recovery_loads_hash_linked_complete_trajectory(
    tmp_path: Path,
) -> None:
    manifest_path = _upgrade_route4_bundle_with_terminal_recovery(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal_entry = manifest["entries"][1]
    source_current_path = tmp_path / "intermediate_strong_current.json"
    source_current_path.write_text('{"checkpoint":"round45"}\n', encoding="utf-8")
    result_path = Path(terminal_entry["result_json"])
    validation_path = Path(terminal_entry["validation_json"])
    exact = float(terminal_entry["exact_same_cutoff_energy"])
    terminal_energy = float(terminal_entry["energy"])
    terminal_error = float(terminal_entry["abs_delta_e"])
    trajectory_semantics = (
        "complete_controller_history_rounds_1_to_45_"
        "with_validated_terminal_refit_endpoint"
    )
    history = [
        {
            "depth": round_index,
            "energy_before_opt": 2.25 if round_index == 1 else exact + 1.0 / round_index,
            "energy_after_opt": exact + 0.5 / round_index,
            "delta_abs_current": 0.5 / round_index,
        }
        for round_index in range(1, 46)
    ]
    trajectory_path = tmp_path / "intermediate_strong_terminal_trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_sr_snake_terminal_trajectory_v1",
                "regime": "intermediate-strong",
                "trajectory_semantics": trajectory_semantics,
                "controller_rounds": 45,
                "trajectory_point_count": 46,
                "source_current_json": str(source_current_path),
                "source_current_sha256": report._sha256(source_current_path),
                "source_result_json": str(result_path),
                "source_result_sha256": report._sha256(result_path),
                "validation_json": str(validation_path),
                "validation_sha256": report._sha256(validation_path),
                "validated_terminal_energy": terminal_energy,
                "validated_terminal_abs_delta_e": terminal_error,
                "adapt_vqe": {
                    "exact_gs_energy": exact,
                    "abs_delta_e": terminal_error,
                    "history_count": 45,
                    "history_checkpoint_complete": True,
                    "history": history,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    terminal_entry.update(
        {
            "source_current_json": str(source_current_path),
            "source_current_sha256": report._sha256(source_current_path),
            "trajectory_json": str(trajectory_path),
            "trajectory_sha256": report._sha256(trajectory_path),
            "trajectory_point_count": 46,
            "trajectory_semantics": trajectory_semantics,
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    rows, campaign = report._load_paper_i_route4_live_snapshot_manifest(
        manifest_path
    )

    curve = report._curve_payload(rows["intermediate-strong"]["curve"])
    assert curve["point_count"] == 46
    assert curve["points"][0] == {
        "k": 0,
        "error": pytest.approx(abs(2.25 - exact)),
    }
    assert curve["points"][-1] == {"k": 45, "error": pytest.approx(terminal_error)}
    assert curve["source_sha256"] == report._sha256(trajectory_path)
    assert curve["source_segments"][-1]["trajectory_point_count"] == 46
    assert campaign["complete_terminal_trajectory_regimes"] == [
        "intermediate-strong"
    ]
    assert campaign["endpoint_semantics"] == (
        "mixed_complete_checkpoint_and_terminal_trajectories_with_validated_endpoints"
    )


def test_sr_recovery_manifest_accepts_fresh_full_horizon_accounting(
    tmp_path: Path,
) -> None:
    manifest_path = _upgrade_route4_bundle_with_terminal_recovery(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    terminal_entry = manifest["entries"][1]
    validation_path = Path(terminal_entry["validation_json"])
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["estimator_accounting"].update(
        {
            "scope": "full_horizon",
            "winning_lineage_S_alg": 135686,
            "cumulative_rounds_1_to_final_S_alg": 135686,
            "exact_blockers": [],
        }
    )
    validation["resume"] = None
    validation["fixed_prefix_reconstruction"] = {
        "prefix_replay_abs_discrepancy": 4.0e-15
    }
    validation_path.write_text(json.dumps(validation) + "\n", encoding="utf-8")
    terminal_entry.update(
        {
            "validation_sha256": report._sha256(validation_path),
            "winning_lineage_S_alg": 135686,
            "S_alg_scope": "full_horizon_rounds_1_to_45",
            "cumulative_rounds_1_to_45_S_alg": 135686,
            "exact_blockers": [],
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    rows, campaign = report._load_paper_i_route4_live_snapshot_manifest(manifest_path)

    terminal = rows["intermediate-strong"]["resource"]
    assert terminal["S"] == 135686
    assert terminal["S_scope"] == "full_horizon_rounds_1_to_45"
    assert terminal["cumulative_rounds_1_to_45_S_alg"] == 135686
    assert terminal["exact_blockers"] == []
    table = report._resource_table_tex([terminal])
    assert "SR recovery [validated; full S r1--45]" in table
    summary = report._sr_recovery_summary(campaign)
    assert "full S r1--45" in summary


def test_route4_nonterminal_checkpoint_can_be_labeled_stopped(
    tmp_path: Path,
) -> None:
    _, manifest_path = _write_route4_live_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["status"] = "stopped_checkpoint_not_terminal"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    rows, campaign = report._load_paper_i_route4_live_snapshot_manifest(
        manifest_path
    )

    resource = rows["weak-strong"]["resource"]
    assert resource["status"] == "stopped_checkpoint_not_terminal"
    assert campaign["running_regimes"] == []
    assert campaign["stopped_regimes"] == ["weak-strong"]
    assert "SR recovery [stopped nonterminal]" in report._resource_table_tex(
        [resource]
    )


def _enrich_stopped_route4_checkpoint_with_fixed_prefix_sidecars(
    tmp_path: Path,
) -> tuple[Path, Path]:
    manifest_path = _upgrade_route4_bundle_with_terminal_recovery(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    snapshot_path = tmp_path / str(entry["snapshot_json"])

    qiskit_path = tmp_path / "weak_strong_stopped_qiskit.json"
    qiskit_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_selected_prefix_qiskit_cost_sidecar_v1",
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "compiled_resource_qiskit_validated": True,
                "compiled_count_2q_total": 12,
                "compiled_depth_2q_total": 10,
                "compiled_depth_total": 50,
                "history_position": 2,
                "logical_operator_count": 2,
                "energy_after_opt_at_prefix": -0.975,
                "primary_error_at_prefix": 0.025,
                "instrumented_runtime_S": 50,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    replay_path = tmp_path / "weak_strong_stopped_fixed_prefix_replay.json"
    replay_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_hh_fixed_prefix_reconstruction_v1",
                "status": "validated",
                "source_result_sha256": report._sha256(snapshot_path),
                "controller_round": 2,
                "operator_count": 2,
                "ordered_labels_exact_match": True,
                "logical_parameters_exact_match": True,
                "runtime_parameters_exact_match": True,
                "saved_energy": -0.975,
                "replayed_energy": -0.975,
                "prefix_replay_abs_discrepancy": 0.0,
                "energy_tolerance": 1.0e-10,
                "saved_absolute_error": 0.025,
                "qiskit": {
                    "path": str(qiskit_path.resolve()),
                    "sha256": report._sha256(qiskit_path),
                    "N2q": 12,
                    "D2q": 10,
                    "circuit_depth": 50,
                },
                "leakage": {
                    "tolerance": 1.0e-10,
                    "maximum_sector_leakage": 0.0,
                    "maximum_padding_leakage": 0.0,
                },
                "estimator_accounting": {
                    "complete": True,
                    "scope": (
                        "display_prefix_rounds_1_to_2_"
                        "retained_history_reconstruction"
                    ),
                    "components": {
                        "N_H_outer": 0,
                        "N_H_refit": 40,
                        "N_grad": 4,
                        "N_metric": 6,
                    },
                    "winning_lineage_S_alg": 50,
                    "exact_blockers": [
                        "raw_state_keyed_call_occurrences_unavailable"
                    ],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    accounting_path = tmp_path / "weak_strong_stopped_accounting.json"
    accounting_path.write_text(
        json.dumps(
            {
                "schema": (
                    "paper_i_sr_snake_stopped_checkpoint_"
                    "accounting_trajectory_v1"
                ),
                "retained_history_reconstruction": {
                    "components": {
                        "N_H_outer": 0,
                        "N_H_refit": 40,
                        "N_grad": 4,
                        "N_metric": 6,
                        "S_alg": 50,
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    entry.update(
        {
            "status": "stopped_checkpoint_not_terminal",
            "pending_fields": [],
            "qiskit_json": str(qiskit_path),
            "qiskit_sha256": report._sha256(qiskit_path),
            "fixed_prefix_replay_json": str(replay_path),
            "fixed_prefix_replay_sha256": report._sha256(replay_path),
            "accounting_and_trajectory_json": str(accounting_path),
            "accounting_and_trajectory_sha256": report._sha256(
                accounting_path
            ),
            "N2q": 12,
            "D2q": 10,
            "Dc": 50,
            "winning_lineage_S_alg": 50,
            "S_alg_scope": (
                "display_prefix_rounds_1_to_2_"
                "retained_history_reconstruction"
            ),
            "exact_blockers": [
                "raw_state_keyed_call_occurrences_unavailable"
            ],
        }
    )
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    return manifest_path, qiskit_path


def test_sr_stopped_checkpoint_loads_full_history_and_fixed_prefix_resources(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _enrich_stopped_route4_checkpoint_with_fixed_prefix_sidecars(
        tmp_path
    )

    rows, campaign = report._load_paper_i_route4_live_snapshot_manifest(
        manifest_path
    )

    stopped = rows["weak-strong"]
    curve = report._curve_payload(stopped["curve"])
    resource = stopped["resource"]
    assert curve["points"] == [
        {"k": 0, "error": pytest.approx(1.5)},
        {"k": 1, "error": pytest.approx(0.2)},
        {"k": 2, "error": pytest.approx(0.025)},
    ]
    assert curve["point_count"] == 3
    assert "status_endpoint_only" not in curve
    assert curve["source_segments"][-1]["fixed_prefix_resources_validated"] is True
    assert [resource[key] for key in ("N2q", "D2q", "Dc", "S")] == [
        12,
        10,
        50,
        50,
    ]
    assert resource["resource_status"] == "validated_stopped_fixed_prefix"
    assert resource["maximum_sector_leakage"] == 0.0
    assert resource["maximum_padding_leakage"] == 0.0
    assert resource["exact_blockers"] == [
        "raw_state_keyed_call_occurrences_unavailable"
    ]
    assert campaign["fixed_prefix_checkpoint_regimes"] == ["weak-strong"]
    assert campaign["endpoint_semantics"] == (
        "mixed_complete_checkpoint_trajectories_and_filled_"
        "validated_terminal_endpoints"
    )
    assert "stopped fixed prefix" in report._sr_recovery_summary(campaign)
    table = report._resource_table_tex([resource])
    assert "SR recovery [stopped r2; fixed prefix]" in table
    assert "12 & 10 & 50 & 50" in table


def test_sr_stopped_fixed_prefix_sidecar_hash_fails_closed(tmp_path: Path) -> None:
    manifest_path, qiskit_path = (
        _enrich_stopped_route4_checkpoint_with_fixed_prefix_sidecars(tmp_path)
    )
    qiskit_path.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="stopped-prefix Qiskit sidecar hash mismatch"):
        report._load_paper_i_route4_live_snapshot_manifest(manifest_path)


def test_sr_recovery_submission_can_be_completed_and_fetched(
    tmp_path: Path,
) -> None:
    manifest_path = _upgrade_route4_bundle_with_terminal_recovery(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["strong_strong_submission"]["status"] = "completed_fetched"
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    _, campaign = report._load_paper_i_route4_live_snapshot_manifest(manifest_path)

    assert campaign["strong_strong_submission"]["status"] == "completed_fetched"


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        ("hash", "snapshot hash mismatch"),
        ("round", "controller-round mismatch"),
        ("overlay", "snapshot overlay mismatch"),
    ),
)
def test_route4_live_manifest_fails_closed_on_hash_or_checkpoint_drift(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    snapshot_path, manifest_path = _write_route4_live_bundle(tmp_path)
    if mutation == "hash":
        snapshot_path.write_text('{"changed":true}\n', encoding="utf-8")
    elif mutation == "round":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["entries"][0]["controller_round"] = 3
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["settings"]["historical_singleton_coordinate_trust_overlay"][
            "phase3_batching_enabled"
        ] = True
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
        snapshot_hash = report._sha256(snapshot_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["entries"][0]["snapshot_sha256"] = snapshot_hash
        manifest["entries"][0]["source_current_sha256"] = snapshot_hash
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        report._load_paper_i_route4_live_snapshot_manifest(manifest_path)


def test_route4_live_manifest_cli_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "route4-live.json"
    captured: dict[str, object] = {}

    def fake_build(*args: object) -> dict[str, str]:
        captured["args"] = args
        return {"status": "ok"}

    monkeypatch.setattr(report, "build", fake_build)

    assert report.main(
        ["--paper-i-route4-live-snapshot-manifest", str(manifest_path)]
    ) == 0
    assert captured["args"][-2] == manifest_path


def _write_jr_chtc_live_bundle(tmp_path: Path) -> tuple[Path, Path]:
    snapshot_path = tmp_path / "weak-weak.current.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "settings": {
                    "static_route_id": "route_a",
                    "route_a_trust_region_state": {
                        "last_update": {
                            "policy": "displacement_calibrated_unbounded_v2"
                        }
                    },
                },
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "energy": -0.975,
                    "abs_delta_e": 0.025,
                    "ansatz_depth": 2,
                    "history_count": 2,
                    "history_checkpoint_complete": True,
                    "partial_checkpoint": True,
                    "stop_reason": None,
                    "history": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.2},
                        {"energy_before_opt": -0.8, "delta_abs_current": 0.025},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    batch_id = "paper-i-hh-jr-l10-rollback-free-r50-test"
    manifest_path = tmp_path / "jr_chtc_live_snapshot_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "jr_snake_chtc_live_snapshot_bundle_v1",
                "batch_id": batch_id,
                "cluster_id": 8776170,
                "captured_at": "2026-07-13T21:04:03Z",
                "status": "running_snapshot_bundle",
                "policy": {
                    "route": "route_a",
                    "batch_search_pool_size": 10,
                    "batch_size_cap": 2,
                    "inner_optimizer": "POWELL",
                    "powell_maxfev": 200,
                    "joint_linear_solve": "supported_metric_whitened_eigh_v1",
                    "trust_region_update": "displacement_calibrated_unbounded_v2",
                    "structural_rollback_enabled": False,
                },
                "entries": [
                    {
                        "proc_id": 0,
                        "row_id": f"{batch_id}__weak_weak",
                        "regime": "weak-weak",
                        "scheduler_state": "running_snapshot",
                        "source_kind": "live_current_json",
                        "snapshot_json": snapshot_path.name,
                        "source_sha256": report._sha256(snapshot_path),
                        "terminal": False,
                        "structural_rollback_enabled": False,
                        "controller_round": 2,
                        "ansatz_depth": 2,
                        "energy": -0.975,
                        "exact_same_cutoff_energy": -1.0,
                        "abs_delta_e": 0.025,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return snapshot_path, manifest_path


def test_jr_chtc_live_manifest_loads_hash_validated_pending_trajectory(
    tmp_path: Path,
) -> None:
    _, manifest_path = _write_jr_chtc_live_bundle(tmp_path)

    rows, campaign = report._load_jr_chtc_live_snapshot_manifest(manifest_path)

    live = rows["weak-weak"]
    curve = report._curve_payload(live["curve"])
    resource = live["resource"]
    assert curve["points"] == [
        {"k": 0, "error": 1.5},
        {"k": 1, "error": 0.2},
        {"k": 2, "error": 0.025},
    ]
    assert curve["live_snapshot"] is True
    assert curve["marker_k"] == 2
    assert resource["status"] == "running_snapshot"
    assert resource["terminal"] is False
    assert [resource[key] for key in ("N2q", "D2q", "Dc", "S")] == [
        None,
        None,
        None,
        None,
    ]
    assert campaign["cluster_id"] == "8776170"
    assert campaign["structural_rollback_enabled"] is False
    assert "JR CHTC [running]" in report._resource_table_tex(
        [resource]
    )


def test_jr_chtc_terminal_manifest_loads_validated_resource_sidecars(
    tmp_path: Path,
) -> None:
    snapshot_path, manifest_path = _write_jr_chtc_live_bundle(tmp_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["adapt_vqe"].update(
        partial_checkpoint=False,
        stop_reason="joint_geometry_selector_exhausted",
    )
    snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
    qiskit_path = tmp_path / "qiskit.json"
    qiskit_path.write_text(
        json.dumps(
            {
                "compiled_resource_qiskit_validated": True,
                "compiled_circuit_stats_status": "ok",
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "primary_error_at_prefix": 0.025,
                "compiled_count_2q_total": 101,
                "compiled_depth_2q_total": 71,
                "compiled_depth_total": 303,
                "replay": {"replayed_operator_count": 2},
            }
        ),
        encoding="utf-8",
    )
    query_path = tmp_path / "query.json"
    query_path.write_text(
        json.dumps(
            {
                "schema": "jr_snake_stitched_winning_lineage_query_work_v2",
                "status": "ok",
                "S_alg_work_scope": "winning_lineage_unique_primitive_union",
                "S_alg": 4,
                "components": {
                    "S_alg_N_H_outer_eval": 1,
                    "S_alg_N_H_refit_eval": 1,
                    "S_alg_N_grad_probe": 1,
                    "S_alg_N_metric_probe": 1,
                    "S_alg_N_other_quantum": 0,
                },
                "primitive_union_validated": True,
                "winning_primitive_ids": ["p_outer", "p_refit", "p_grad", "p_metric"],
                "all_executed_primitive_ids": [
                    "p_outer",
                    "p_refit",
                    "p_grad",
                    "p_metric",
                    "p_discarded",
                ],
                "winning_component_by_primitive_id": {
                    "p_outer": "N_H_outer",
                    "p_refit": "N_H_refit",
                    "p_grad": "N_grad",
                    "p_metric": "N_metric",
                },
                "all_component_by_primitive_id": {
                    "p_outer": "N_H_outer",
                    "p_refit": "N_H_refit",
                    "p_grad": "N_grad",
                    "p_metric": "N_metric",
                    "p_discarded": "N_H_refit",
                },
                "discarded_branch_operational_overhead": {
                    "definition": (
                        "all_executed_unique_ids_minus_winning_lineage_unique_ids"
                    ),
                    "S_alg": 1,
                    "components": {
                        "S_alg_N_H_outer_eval": 0,
                        "S_alg_N_H_refit_eval": 1,
                        "S_alg_N_grad_probe": 0,
                        "S_alg_N_metric_probe": 0,
                        "S_alg_N_other_quantum": 0,
                    },
                    "primitive_ids": ["p_discarded"],
                },
                "discarded_branch_search_work_included": False,
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    entry.update(
        scheduler_state="completed_snapshot_pending_qiskit",
        source_kind="completed_result_json",
        source_sha256=report._sha256(snapshot_path),
        terminal=True,
        qiskit_sidecar_json=qiskit_path.name,
        qiskit_sidecar_sha256=report._sha256(qiskit_path),
        query_work_sidecar_json=query_path.name,
        query_work_sidecar_sha256=report._sha256(query_path),
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows, campaign = report._load_jr_chtc_live_snapshot_manifest(manifest_path)

    resource = rows["weak-weak"]["resource"]
    assert [resource[key] for key in ("N2q", "D2q", "Dc", "S")] == [
        101,
        71,
        303,
        None,
    ]
    assert resource["S_scope"] == "withdrawn_unique_primitive_union_is_not_S_alg"
    assert resource["S_status"] == "unavailable_raw_occurrence_stitching"
    assert campaign["resource_fields"] == "per_entry_pending_or_validated_sidecars"
    assert "JR CHTC [done]" in report._resource_table_tex([resource])


def test_jr_chtc_live_manifest_rejects_structural_rollback(
    tmp_path: Path,
) -> None:
    snapshot_path, manifest_path = _write_jr_chtc_live_bundle(tmp_path)
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    payload["settings"]["adapt_rollback_mode"] = "structural"
    snapshot_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["source_sha256"] = report._sha256(snapshot_path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="structural rollback"):
        report._load_jr_chtc_live_snapshot_manifest(manifest_path)


def test_jr_chtc_held_snapshot_is_nonterminal_and_resource_pending(
    tmp_path: Path,
) -> None:
    _, manifest_path = _write_jr_chtc_live_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["scheduler_state"] = "held_snapshot"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows, _ = report._load_jr_chtc_live_snapshot_manifest(manifest_path)

    resource = rows["weak-weak"]["resource"]
    assert resource["terminal"] is False
    assert resource["status"] == "held_snapshot"
    assert "JR CHTC [held]" in report._resource_table_tex([resource])


def test_jr_chtc_stopped_snapshot_accepts_validated_fixed_prefix_sidecars(
    tmp_path: Path,
) -> None:
    snapshot_path, manifest_path = _write_jr_chtc_live_bundle(tmp_path)
    qiskit_path = tmp_path / "qiskit.json"
    qiskit_path.write_text(
        json.dumps(
            {
                "compiled_resource_qiskit_validated": True,
                "compiled_circuit_stats_status": "ok",
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "primary_error_at_prefix": 0.025,
                "compiled_count_2q_total": 101,
                "compiled_depth_2q_total": 71,
                "compiled_depth_total": 303,
                "replay": {"replayed_operator_count": 2},
            }
        ),
        encoding="utf-8",
    )
    query_path = tmp_path / "query.json"
    query_path.write_text(
        json.dumps(
            {
                "schema": "jr_snake_stitched_winning_lineage_query_work_v1",
                "status": "ok",
                "S_alg_work_scope": "winning_lineage_stitched_segments",
                "S_alg": 4567.0,
                "discarded_branch_search_work_included": False,
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest["entries"][0]
    entry.update(
        scheduler_state="stopped_snapshot",
        terminal=False,
        qiskit_sidecar_json=qiskit_path.name,
        qiskit_sidecar_sha256=report._sha256(qiskit_path),
        query_work_sidecar_json=query_path.name,
        query_work_sidecar_sha256=report._sha256(query_path),
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows, campaign = report._load_jr_chtc_live_snapshot_manifest(manifest_path)

    resource = rows["weak-weak"]["resource"]
    assert resource["terminal"] is False
    assert resource["status"] == "stopped_snapshot"
    assert resource["prefix_semantics"] == (
        "immutable_stopped_checkpoint_with_validated_snapshot_sidecars"
    )
    assert [resource[key] for key in ("N2q", "D2q", "Dc", "S")] == [
        101,
        71,
        303,
        None,
    ]
    assert resource["S_status"] == "legacy_proxy_not_exact"
    assert resource["legacy_proxy_S"] == 4567
    assert campaign["stopped_regimes"] == ["weak-weak"]
    assert campaign["running_regimes"] == []
    assert "JR CHTC [stopped snapshot]" in report._resource_table_tex([resource])


def test_jr_chtc_caveat_records_archived_radius_scope_and_round_30_target() -> None:
    row = next(
        item
        for item in report.RUN_SETTING_LEDGER
        if item["curve"].startswith("Dark teal dashed")
    )

    assert "final selector adaptive" in row["rho"]
    assert "Phase 1/2 static rho=0.25" in row["rho"]
    assert "recovery target round 30" in row["optimizer"]
    assert "stopped round-29 fixed prefixes" in row["caveat"]
    assert "predate unified-rho plumbing" in row["caveat"]
    source = Path(report.__file__).read_text(encoding="utf-8")
    assert "per-entry fixed-prefix Qiskit/S" in source
    assert "Qiskit and finalized S pending" not in source


def test_jr_chtc_recovery_queued_snapshot_is_nonterminal_and_resource_pending(
    tmp_path: Path,
) -> None:
    _, manifest_path = _write_jr_chtc_live_bundle(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"][0]["scheduler_state"] = "recovery_queued_snapshot"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    rows, _ = report._load_jr_chtc_live_snapshot_manifest(manifest_path)

    resource = rows["weak-weak"]["resource"]
    assert resource["terminal"] is False
    assert resource["status"] == "recovery_queued_snapshot"
    assert "JR CHTC [recovery queued]" in report._resource_table_tex([resource])


def test_jr_chtc_live_manifest_cli_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "jr-chtc-live.json"
    captured: dict[str, object] = {}

    def fake_build(*args: object) -> dict[str, str]:
        captured["args"] = args
        return {"status": "ok"}

    monkeypatch.setattr(report, "build", fake_build)

    assert report.main(
        ["--jr-chtc-live-snapshot-manifest", str(manifest_path)]
    ) == 0
    assert captured["args"][-1] == manifest_path


def _write_fm_live_snapshot(
    path: Path,
    *,
    qbroyd_epsilon0: float,
    rollback_mode: str | None = None,
) -> None:
    settings = {
        "adapt_reoptimization_route": "formal_manifold_warm_start_v1",
        "adapt_formal_manifold_config": {
            "qbroyd_epsilon0": qbroyd_epsilon0,
        },
    }
    if rollback_mode is not None:
        settings["adapt_rollback_mode"] = rollback_mode
    path.write_text(
        json.dumps(
            {
                "settings": settings,
                "adapt_vqe": {
                    "exact_gs_energy": -1.0,
                    "abs_delta_e": 0.025,
                    "ansatz_depth": 2,
                    "history_checkpoint_complete": True,
                    "history": [
                        {"energy_before_opt": 0.5, "delta_abs_current": 0.2},
                        {"energy_before_opt": -0.8, "delta_abs_current": 0.025},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )


def _write_fm_live_manifest(
    path: Path,
    *,
    snapshot_path: Path,
    policy: str = "qbroyd_on",
    proc_id: int = 0,
    source_sha256: str | None = None,
) -> None:
    batch_id = "fm-live-batch"
    path.write_text(
        json.dumps(
            {
                "schema": "formal_manifold_live_snapshot_bundle_v1",
                "batch_id": batch_id,
                "cluster_id": 8776119,
                "captured_at": "2026-07-12T23:00:00Z",
                "status": "running_snapshot_bundle",
                "entries": [
                    {
                        "proc_id": proc_id,
                        "row_id": f"{batch_id}__weak_weak__{policy}__depth30__powell200",
                        "regime": "weak-weak",
                        "policy": policy,
                        "scheduler_state": "running_snapshot",
                        "source_kind": "live_current_json",
                        "snapshot_json": snapshot_path.name,
                        "source_sha256": source_sha256 or report._sha256(snapshot_path),
                        "route_id": "formal_manifold_warm_start_v1",
                        "structural_rollback_enabled": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_fm_live_manifest_loads_hash_validated_nonterminal_curve(
    tmp_path: Path,
) -> None:
    snapshot_path = tmp_path / "current.json"
    manifest_path = tmp_path / "manifest.json"
    _write_fm_live_snapshot(snapshot_path, qbroyd_epsilon0=0.15)
    _write_fm_live_manifest(manifest_path, snapshot_path=snapshot_path)

    rows, metadata = report._load_fm_live_snapshot_manifest(manifest_path)

    live = rows["weak-weak"]["fm_qbroyd_default"]
    assert live["curve"].points == ((0, 1.5), (1, 0.2), (2, 0.025))
    assert live["resource"]["status"] == "running_snapshot"
    assert live["resource"]["k_pl"] == 2
    assert live["resource"]["ansatz_depth"] == 2
    assert live["resource"]["abs_delta_e"] == pytest.approx(0.025)
    assert live["resource"]["N2q"] is None
    assert live["resource"]["D2q"] is None
    assert live["resource"]["Dc"] is None
    assert live["resource"]["S"] is None
    assert live["resource"]["terminal"] is False
    assert metadata["cluster_id"] == "8776119"
    assert metadata["evidence_class"] == "matched_within_batch_diagnostic"
    assert metadata["source_value_anchor"] == "absent_not_claimed"
    assert metadata["source_locked_sensitivity"] is False
    assert metadata["terminal_evidence"] is False
    assert "FM qB on [running]" in report._resource_table_tex(
        [live["resource"]]
    )
    assert "pending & pending & pending & pending" in report._resource_table_tex(
        [live["resource"]]
    )
    assert "qB default" not in report.RESOURCE_METHOD_DISPLAY["fm_qbroyd_default"]


def test_fm_failed_progress_uses_last_complete_ansatz_depth(tmp_path: Path) -> None:
    snapshot_path = tmp_path / "failure_progress.json"
    manifest_path = tmp_path / "manifest.json"
    snapshot_path.write_text(
        json.dumps(
            {
                "route_id": "formal_manifold_warm_start_v1",
                "qbroyd_epsilon0": 0.15,
                "outer_structural_rollback_active": False,
                "progress": [
                    {"round": 11, "ansatz_depth": 11, "abs_delta_e": 2.7e-5},
                    {"round": 12, "ansatz_depth": 12, "abs_delta_e": 2.5e-5},
                ],
            }
        ),
        encoding="utf-8",
    )
    batch_id = "fm-live-batch"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "formal_manifold_live_snapshot_bundle_v1",
                "batch_id": batch_id,
                "cluster_id": 8776119,
                "captured_at": "2026-07-12T23:00:00Z",
                "status": "partial_failure_bundle",
                "entries": [
                    {
                        "proc_id": 2,
                        "row_id": f"{batch_id}__intermediate_weak__qbroyd_on__depth30__powell200",
                        "regime": "intermediate-weak",
                        "policy": "qbroyd_on",
                        "scheduler_state": "failed_partial",
                        "source_kind": "failure_progress_log",
                        "snapshot_json": snapshot_path.name,
                        "source_sha256": report._sha256(snapshot_path),
                        "route_id": "formal_manifold_warm_start_v1",
                        "structural_rollback_enabled": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows, _ = report._load_fm_live_snapshot_manifest(manifest_path)

    resource = rows["intermediate-weak"]["fm_qbroyd_default"]["resource"]
    assert resource["status"] == "failed_partial"
    assert resource["k_pl"] == 12
    assert resource["ansatz_depth"] == 12
    assert resource["terminal"] is False


def test_fm_live_merge_preserves_prior_terminal_and_exposes_live_prefix() -> None:
    prior_curve = report.Curve(
        role="fm_qbroyd_off",
        points=((0, 1.0), (1, 0.1)),
        marker_k=1,
        marker_error=0.1,
        source_json="prior.json",
        source_sha256="a" * 64,
    )
    live_curve = report.Curve(
        role="fm_qbroyd_off",
        points=((0, 1.0), (1, 0.1), (2, 0.01)),
        marker_k=2,
        marker_error=0.01,
        source_json="live.json",
        source_sha256="b" * 64,
    )
    prior_resource = {
        "method": "formal_manifold_snake",
        "status": "complete",
        "k_pl": 1,
        "abs_delta_e": 0.1,
        "N2q": 10,
        "D2q": 9,
        "Dc": 20,
        "S": 30,
    }
    live_resource = {
        "method": "fm_qbroyd_off",
        "status": "running_snapshot",
        "k_pl": 2,
        "abs_delta_e": 0.01,
        "N2q": None,
        "D2q": None,
        "Dc": None,
        "S": None,
    }

    curves, resources = report._merge_fm_model_live_evidence(
        {"curve": prior_curve, "resource": prior_resource},
        {"curve": live_curve, "resource": live_resource},
    )

    assert curves["fm_qbroyd_off"]["marker_k"] == 2
    assert curves["fm_qbroyd_off_prior"]["marker_k"] == 1
    assert [row["method"] for row in resources] == [
        "fm_qbroyd_off",
        "fm_qbroyd_off_prior",
    ]
    assert resources[0]["status"] == "running_snapshot"
    assert resources[1]["status"] == "complete"
    assert resources[1]["N2q"] == 10


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("hash", "hash mismatch"),
        ("policy", "qB-off snapshot has nonzero"),
        ("proc", "proc/policy mapping mismatch"),
        ("rollback", "structural rollback"),
    ),
)
def test_fm_live_manifest_rejects_mismatched_provenance(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    snapshot_path = tmp_path / "current.json"
    manifest_path = tmp_path / "manifest.json"
    _write_fm_live_snapshot(
        snapshot_path,
        qbroyd_epsilon0=0.15,
        rollback_mode="structural" if failure == "rollback" else None,
    )
    _write_fm_live_manifest(
        manifest_path,
        snapshot_path=snapshot_path,
        policy="qbroyd_off" if failure == "policy" else "qbroyd_on",
        proc_id=7 if failure == "proc" else (1 if failure == "policy" else 0),
        source_sha256="0" * 64 if failure == "hash" else None,
    )

    with pytest.raises(ValueError, match=match):
        report._load_fm_live_snapshot_manifest(manifest_path)


def test_fm_lightweight_status_adds_marker_only_endpoints_and_packaging_star(
    tmp_path: Path,
) -> None:
    live_rows = {regime: {} for regime in report.REGIME_ORDER}
    for regime in report.REGIME_ORDER:
        for policy in ("qbroyd_on", "qbroyd_off"):
            role = report._fm_live_policy_role(policy)
            live_rows[regime][role] = {
                "curve": report.Curve(
                    role=role,
                    points=((0, 1.0), (1, 0.1)),
                    marker_k=1,
                    marker_error=0.1,
                    source_json=f"{regime}-{policy}.json",
                    source_sha256="b" * 64,
                ),
                "resource": {
                    "method": role,
                    "status": "running_snapshot",
                    "k_pl": 1,
                    "ansatz_depth": 1,
                    "abs_delta_e": 0.1,
                    "N2q": None,
                    "D2q": None,
                    "Dc": None,
                    "S": None,
                },
                "provenance": {},
            }
    repair_path = tmp_path / "repair.json"
    repair_path.write_text('{"schema":"repair"}\n', encoding="utf-8")
    entries = []
    for regime in report.REGIME_ORDER:
        for policy in ("qbroyd_on", "qbroyd_off"):
            packaging = regime == "weak-weak" or (
                regime == "strong-weak" and policy == "qbroyd_off"
            )
            restart = regime == "intermediate-weak"
            entries.append(
                {
                    "regime": regime,
                    "policy": policy,
                    "cluster_id": "8776378" if restart else "8776119",
                    "state": (
                        "science_complete_packaging_failed"
                        if packaging
                        else "running_status_endpoint"
                    ),
                    "controller_round": 13 if packaging else 5,
                    "ansatz_depth": 12 if packaging else 4,
                    "abs_delta_e": 2.9e-6 if packaging else 2.0e-4,
                    "metric_source": (
                        "last_verified_prior_checkpoint"
                        if packaging
                        else "live_checkpoint_status_observation"
                    ),
                    "terminal_metric_validated": False,
                    "trajectory_relation": (
                        "replacement_restart_after_parent_failure"
                        if restart
                        else "same_row_later_observation"
                    ),
                }
            )
    status_path = tmp_path / "status.json"
    status_path.write_text(
        json.dumps(
            {
                "schema": "formal_manifold_lightweight_status_snapshot_v1",
                "captured_at": "2026-07-13T14:54:00Z",
                "captured_at_local": "2026-07-13 15:55 CDT",
                "route_id": "formal_manifold_warm_start_v1",
                "structural_rollback_enabled": False,
                "prior_live_snapshot": {"manifest_sha256": "a" * 64},
                "replacement_repair": {
                    "cluster_id": "8776378",
                    "manifest": str(repair_path),
                    "manifest_sha256": report._sha256(repair_path),
                },
                "entries": entries,
            }
        ),
        encoding="utf-8",
    )

    campaign = report._load_fm_live_status_snapshot(
        status_path,
        live_rows=live_rows,
        live_campaign={"manifest_sha256": "a" * 64},
    )

    iw = live_rows["intermediate-weak"]["fm_qbroyd_default"]
    assert iw["curve"].points == ((0, 1.0), (1, 0.1))
    assert iw["curve"].marker_k == 5
    assert iw["resource"]["trajectory_relation"] == "replacement_restart_after_parent_failure"
    assert iw["resource"]["status_endpoint_only"] is True
    weak = live_rows["weak-weak"]["fm_qbroyd_default"]["resource"]
    assert weak["checkpoint_asterisk"] is True
    assert weak["terminal_metric_validated"] is False
    tex = report._resource_table_tex([weak])
    assert "done/pkg fail*" in tex
    assert "13*" in tex
    assert "2.90e-06*" in tex
    assert "pending & pending & pending & pending" in tex
    running = live_rows["strong-strong"]["fm_qbroyd_default"]["resource"]
    assert running["status_capture_label"] == "15:55 CDT"
    assert "run@15:55 CDT" in report._resource_table_tex([running])
    assert campaign["endpoint_semantics"] == "marker_only_no_trajectory_interpolation"


def _write_fm_stopped_snapshot_bundle(
    root: Path,
) -> tuple[Path, dict[int, Path]]:
    batch_id = "fm-stopped-batch"
    cluster_id = "8776119"
    regimes = (
        "weak-strong",
        "weak-strong",
        "intermediate-strong",
        "intermediate-strong",
        "strong-strong-u8",
        "strong-strong-u8",
    )
    rows = []
    current_paths: dict[int, Path] = {}
    for proc_id, regime in zip(range(6, 12), regimes):
        qbroyd_on = proc_id % 2 == 0
        policy = "qbroyd_on" if qbroyd_on else "qbroyd_off"
        policy_id = f"inverse_rbfgs_{policy}_v1"
        proc_dir = root / f"proc{proc_id}"
        proc_dir.mkdir(parents=True)
        current_path = proc_dir / "current.json"
        error = 0.01 * (proc_id + 1)
        current_path.write_text(
            json.dumps(
                {
                    "settings": {
                        "adapt_reoptimization_route": "formal_manifold_warm_start_v1",
                        "adapt_formal_manifold_config": {
                            "qbroyd_epsilon0": 0.15 if qbroyd_on else 0.0,
                        },
                    },
                    "no_credentials_serialized": True,
                    "checkpoint": {
                        "complete": False,
                        "reason": "beam_round_done",
                    },
                    "adapt_vqe": {
                        "exact_gs_energy": -1.0,
                        "abs_delta_e": error,
                        "ansatz_depth": 2,
                        "branch_id": proc_id + 10,
                        "history_checkpoint_complete": True,
                        "partial_checkpoint": True,
                        "history": [
                            {
                                "energy_before_opt": 0.5,
                                "delta_abs_current": 0.2,
                            },
                            {
                                "energy_before_opt": -0.8,
                                "delta_abs_current": error,
                            },
                        ],
                        "formal_manifold_warm_state_checkpoint": {
                            "route": "formal_manifold_warm_start_v1",
                            "rank": 2,
                            "trust_radius": 0.125,
                            "curvature_branch": (
                                "inverse_rbfgs_raised_covariant_hessian_v1"
                            ),
                        },
                        "formal_manifold_query_closure_checkpoint": {
                            "current_round_finalized": True,
                        },
                        "formal_manifold_query_closure": {
                            "route": "formal_manifold_warm_start_v1",
                            "joint_response_selector_invoked": False,
                        },
                        "route_a_trust_region_state": {
                            "schema": "route_a_trust_region_state_v1",
                            "radius": 0.25,
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        settings_hash = f"{proc_id + 1:064x}"
        plan_path = proc_dir / "plan.json"
        plan_path.write_text(
            json.dumps({"scientific_settings_hash": settings_hash}),
            encoding="utf-8",
        )
        regime_token = regime.replace("-", "_")
        row_id = (
            f"{batch_id}__{regime_token}__{policy}__depth30__powell200"
        )
        (proc_dir / "runner_manifest.json").write_text(
            json.dumps(
                {
                    "batch_id": batch_id,
                    "regime": regime,
                    "policy_id": policy_id,
                    "row_id": row_id,
                    "remote_dry_run_plan_sha256": report._sha256(plan_path),
                    "expected_scientific_settings_hash": settings_hash,
                }
            ),
            encoding="utf-8",
        )
        rows.append(
            {
                "proc_id": proc_id,
                "regime": regime,
                "policy_id": policy_id,
                "ansatz_depth": 2,
                "abs_delta_e": error,
                "branch_id": proc_id + 10,
                "metric_rank": 2,
                "formal_trust_radius": 0.125,
                "curvature_branch": (
                    "inverse_rbfgs_raised_covariant_hessian_v1"
                ),
                "current_json": str(current_path.relative_to(root)),
                "current_sha256": report._sha256(current_path),
            }
        )
        current_paths[proc_id] = current_path
    manifest_path = root / "retrieval_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "formal_manifold_stop_retrieval_manifest_v1",
                "batch_id": batch_id,
                "cluster_id": cluster_id,
                "snapshot_stage_created_utc": "2026-07-14T20:11:58Z",
                "stop_requested_utc": "2026-07-14T20:25:08Z",
                "stop_scope": [f"{cluster_id}.{proc_id}" for proc_id in range(6, 12)],
                "scheduler_status_after_stop": {
                    "job_status": 3,
                    "meaning": "removed",
                    "unrelated_jobs_touched": False,
                },
                "validation": {
                    "archive_sha256_matches_access_point": True,
                    "gzip_streams_valid": True,
                    "all_json_parse": True,
                    "runner_expected_settings_hash_matches_plan": True,
                    "all_current_checkpoints_are_completed_beam_rounds": True,
                    "all_formal_warm_state_checkpoints_present": True,
                    "all_query_closure_checkpoints_present": True,
                    "all_route_trust_states_present": True,
                    "credentials_serialized": False,
                    "evidence_class": "matched_within_batch_diagnostic",
                    "source_value_anchor": "not_run_user_approved_parallel_fanout",
                    "source_locked_sensitivity": False,
                },
                "rows": rows,
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, current_paths


def test_fm_stopped_snapshot_loads_and_replaces_only_proc6_to_11(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, _ = _write_fm_stopped_snapshot_bundle(tmp_path)
    stopped_rows, campaign = report._load_fm_stopped_snapshot_manifest(
        manifest_path
    )

    assert campaign["entry_count"] == 6
    assert campaign["overlay_scope"] == "proc6-11_replaces_prior_live_status_rows_only"
    stopped = stopped_rows["weak-strong"]["fm_qbroyd_default"]
    assert stopped["curve"].points[-1] == pytest.approx((2, 0.07))
    assert stopped["resource"]["status"] == "stopped_snapshot"
    assert stopped["resource"]["terminal"] is False
    assert stopped["resource"]["N2q"] is None

    compile_values = {
        "fm_qbroyd_default": (42, 28, 146),
        "fm_qbroyd_off": (50, 35, 204),
    }

    def fake_supplemental(**kwargs: object) -> dict[str, object]:
        role = str(kwargs["method"])
        n2q, d2q, dc = compile_values[role]
        return {
            "regime": kwargs["regime"],
            "method": role,
            "k_pl": kwargs["history_position"],
            "abs_delta_e": kwargs["expected_error"],
            "N2q": n2q,
            "D2q": d2q,
            "Dc": dc,
            "S": 999,
            "qiskit_sidecar": str(kwargs["sidecar_json"]),
        }

    monkeypatch.setattr(report, "_supplemental_resource_row", fake_supplemental)
    report._compile_fm_stopped_snapshot_resources(
        stopped_rows,
        supplemental_dir=tmp_path / "qiskit",
    )
    compiled = stopped_rows["weak-strong"]["fm_qbroyd_default"]["resource"]
    assert (compiled["N2q"], compiled["D2q"], compiled["Dc"]) == (42, 28, 146)
    assert compiled["S"] is None
    assert compiled["terminal"] is False
    assert "stopped fixed-prefix" in report._resource_table_tex([compiled])
    assert "42 & 28 & 146 & n/a" in report._resource_table_tex([compiled])

    sentinel = {"resource": {"status": "science_complete_packaging_failed"}}
    live_rows = {regime: {} for regime in report.REGIME_ORDER}
    live_rows["weak-weak"]["fm_qbroyd_default"] = sentinel
    report._overlay_fm_stopped_snapshot_rows(live_rows, stopped_rows)
    assert live_rows["weak-weak"]["fm_qbroyd_default"] is sentinel
    assert live_rows["weak-strong"]["fm_qbroyd_default"] is stopped


@pytest.mark.parametrize(
    ("failure", "match"),
    (
        ("hash", "hash mismatch"),
        ("route", "route mismatch"),
        ("qbroyd", "qB-on checkpoint has disabled"),
        ("rollback", "structural rollback"),
        ("checkpoint", "completed accepted round"),
    ),
)
def test_fm_stopped_snapshot_rejects_invalid_checkpoint_evidence(
    tmp_path: Path,
    failure: str,
    match: str,
) -> None:
    manifest_path, current_paths = _write_fm_stopped_snapshot_bundle(tmp_path)
    current_path = current_paths[6]
    payload = json.loads(current_path.read_text(encoding="utf-8"))
    if failure == "hash":
        current_path.write_text(
            current_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
    else:
        if failure == "route":
            payload["settings"]["adapt_reoptimization_route"] = "wrong_route"
        elif failure == "qbroyd":
            payload["settings"]["adapt_formal_manifold_config"][
                "qbroyd_epsilon0"
            ] = 0.0
        elif failure == "rollback":
            payload["settings"]["adapt_rollback_mode"] = "structural"
        elif failure == "checkpoint":
            payload["adapt_vqe"]["history_checkpoint_complete"] = False
        current_path.write_text(json.dumps(payload), encoding="utf-8")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["rows"][0]["current_sha256"] = report._sha256(current_path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        report._load_fm_stopped_snapshot_manifest(manifest_path)


def test_fm_stopped_snapshot_cli_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "fm-stopped.json"
    captured: dict[str, object] = {}

    def fake_build(*args: object) -> dict[str, str]:
        captured["args"] = args
        return {"status": "ok"}

    monkeypatch.setattr(report, "build", fake_build)

    assert report.main(
        ["--fm-stopped-snapshot-manifest", str(manifest_path)]
    ) == 0
    assert captured["args"][-3] == manifest_path


def _write_fm_completed_resource_recovery_bundle(
    tmp_path: Path,
) -> tuple[
    Path,
    dict[str, dict[str, dict[str, object]]],
    dict[str, object],
]:
    batch_id = "fm-completed-resource-recovery-test"
    source_retrieval = tmp_path / "retrieval_manifest.json"
    source_retrieval.write_text("{}\n", encoding="utf-8")
    live_rows: dict[str, dict[str, dict[str, object]]] = {
        regime: {} for regime in report.REGIME_ORDER
    }
    rows = []
    for regime in ("weak-weak", "intermediate-weak", "strong-weak"):
        manifest_regime = (
            regime.replace("-", "_") + ("_u8" if regime == "strong-weak" else "")
        )
        for policy in ("qbroyd_on", "qbroyd_off"):
            role = report._fm_live_policy_role(policy)
            row_id = f"{batch_id}__{manifest_regime}__{policy}"
            live_rows[regime][role] = {
                "curve": report.Curve(
                    role=role,
                    points=((0, 1.0), (1, 0.1)),
                    marker_k=1,
                    marker_error=0.1,
                    source_json=f"{row_id}.current.json",
                    source_sha256="a" * 64,
                ),
                "resource": {
                    "regime": regime,
                    "method": role,
                    "status": "science_complete_packaging_failed",
                    "k_pl": 1,
                    "ansatz_depth": 1,
                    "abs_delta_e": 0.1,
                    "N2q": None,
                    "D2q": None,
                    "Dc": None,
                    "S": None,
                    "resource_status": "pending_terminal_packaging_and_sidecars",
                },
                "provenance": {"row_id": row_id},
            }
            row_dir = tmp_path / row_id
            row_dir.mkdir()
            full_hash = ("1" if policy == "qbroyd_on" else "2") * 64
            query_path = row_dir / "query.json"
            query_path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "formal_manifold_terminal_query_work_stdout_recovery_v1"
                        ),
                        "query_work_status": "ok",
                        "query_work_scope": "accepted_terminal_lineage",
                        "query_work_total": 123.0,
                        "winning_branch": {"expanded_query_work": 123.0},
                        "source_full_result_sha256": full_hash,
                    }
                ),
                encoding="utf-8",
            )
            endpoint_missing = regime == "strong-weak" and policy == "qbroyd_on"
            qiskit_path = row_dir / "qiskit.json"
            endpoint = None
            qiskit_status = "unavailable_terminal_operator_sequence_omitted"
            if not endpoint_missing:
                endpoint = {
                    "controller_round": 3,
                    "ansatz_depth": 2,
                    "abs_delta_e": 1.0e-5,
                }
                qiskit_status = "validated_report_qiskit"
                qiskit_path.write_text(
                    json.dumps(
                        {
                            "compiled_resource_qiskit_validated": True,
                            "compiled_circuit_stats_status": "ok",
                            "compile_convention": "table_i_basis_gate_transpile_v1",
                            "source_full_result_sha256": full_hash,
                            "primary_error_at_prefix": 1.0e-5,
                            "instrumented_runtime_S": 123.0,
                            "compiled_count_2q_total": 10,
                            "compiled_depth_2q_total": 8,
                            "compiled_depth_total": 40,
                            "replay": {"replayed_operator_count": 2},
                        }
                    ),
                    encoding="utf-8",
                )
            rows.append(
                {
                    "row_id": row_id,
                    "regime": manifest_regime,
                    "policy": policy,
                    "qiskit_status": qiskit_status,
                    "query_work_total": 123.0,
                    "endpoint": endpoint,
                    "omitted_full_result_sha256": full_hash,
                    "query_work_sidecar": str(query_path),
                    "query_work_sidecar_sha256": report._sha256(query_path),
                    "qiskit_sidecar": (
                        None if endpoint_missing else str(qiskit_path)
                    ),
                    "qiskit_sidecar_sha256": (
                        None
                        if endpoint_missing
                        else report._sha256(qiskit_path)
                    ),
                }
            )
    manifest_path = tmp_path / "recovery_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": report.FM_COMPLETED_RESOURCE_RECOVERY_SCHEMA,
                "batch_id": batch_id,
                "created_utc": "2026-07-14T00:00:00+00:00",
                "source_retrieval_manifest": str(source_retrieval),
                "source_retrieval_manifest_sha256": report._sha256(
                    source_retrieval
                ),
                "rows": rows,
                "validation": {
                    "exact_terminal_query_work_rows": 6,
                    "validated_qiskit_rows": 5,
                    "qiskit_compile_convention": (
                        "table_i_basis_gate_transpile_v1"
                    ),
                    "terminal_operator_sequence_unavailable_rows": [
                        rows[-2]["row_id"]
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path, live_rows, {"batch_id": batch_id}


def test_fm_completed_resource_recovery_overlays_qiskit_and_query_work(
    tmp_path: Path,
) -> None:
    manifest, live_rows, campaign = _write_fm_completed_resource_recovery_bundle(
        tmp_path
    )

    metadata = report._overlay_fm_completed_resource_recovery(
        manifest,
        live_rows=live_rows,
        live_campaign=campaign,
    )

    assert metadata["validated_qiskit_rows"] == 5
    assert metadata["exact_terminal_query_work_rows"] == 6
    recovered = live_rows["weak-weak"]["fm_qbroyd_off"]
    assert recovered["curve"].marker_k == 3
    assert recovered["curve"].marker_error == pytest.approx(1.0e-5)
    assert recovered["resource"]["N2q"] == 10
    assert recovered["resource"]["D2q"] == 8
    assert recovered["resource"]["Dc"] == 40
    assert recovered["resource"]["S"] == 123
    unavailable = live_rows["strong-weak"]["fm_qbroyd_default"]["resource"]
    assert unavailable["N2q"] is None
    assert unavailable["S"] == 123
    table = report._resource_table_tex([unavailable])
    assert "Qiskit unavailable" in table
    assert "123\\textsuperscript{\\dag}" in table


def test_fm_completed_resource_recovery_rejects_sidecar_hash_drift(
    tmp_path: Path,
) -> None:
    manifest, live_rows, campaign = _write_fm_completed_resource_recovery_bundle(
        tmp_path
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["rows"][0]["query_work_sidecar_sha256"] = "f" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="query sidecar hash mismatch"):
        report._overlay_fm_completed_resource_recovery(
            manifest,
            live_rows=live_rows,
            live_campaign=campaign,
        )


def test_fm_completed_resource_recovery_cli_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "fm-recovery.json"
    captured: dict[str, object] = {}

    def fake_build(*args: object) -> dict[str, str]:
        captured["args"] = args
        return {"status": "ok"}

    monkeypatch.setattr(report, "build", fake_build)

    assert report.main(
        ["--fm-completed-resource-recovery-manifest", str(manifest_path)]
    ) == 0
    assert captured["args"][-4] == manifest_path


def _write_sr_expanded_chart_whitening_validation(
    tmp_path: Path,
) -> Path:
    source_archive = tmp_path / "source-tree.tar.gz"
    source_archive.write_bytes(b"source-lock")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text('{"locked":true}\n', encoding="utf-8")

    def accepted_refit(round_index: int) -> dict[str, object]:
        logical_count = round_index
        base_count = round_index + 1
        occurrences = logical_count * (logical_count + 1) // 2
        return {
            "policy": "supported_fs_whitened_fixed_v1",
            "base_chart_policy": "expanded_runtime_projected_logical_v1",
            "base_coordinate_kind": "expanded_runtime_projected_logical",
            "logical_parameter_count": logical_count,
            "base_parameter_count": base_count,
            "metric_support_rank": round_index,
            "chart_fixed_within_powell_invocation": True,
            "chart_recomputed_after_next_admission": True,
            "accepted_refit_invocation": {
                "metric_query_accounting": {
                    "symmetric_metric_element_occurrences": occurrences,
                    "new_unique_metric_elements_charged": logical_count,
                    "deduplicated_or_ledger_disabled_count": (
                        occurrences - logical_count
                    ),
                }
            },
        }

    rows = (
        (
            "historical_high_accuracy_sr_baseline",
            "expanded_runtime_projected_logical_v1",
            4.472864776339236e-7,
            None,
        ),
        (
            "wrong_reduced_chart_whitened_r22",
            "logical_shared_reduced_v1",
            3.6989258474240394e-4,
            3.6989219635752413e-4,
        ),
        (
            "good_expanded_chart_whitened_r22",
            "expanded_runtime_projected_logical_v1",
            1.50837715207075e-7,
            5.7635175610970535e-6,
        ),
        (
            "good_expanded_chart_whitened_r30",
            "expanded_runtime_projected_logical_v1",
            1.0373365499916076e-9,
            1.7400664686917366e-8,
        ),
    )
    comparisons = []
    result_paths: dict[str, Path] = {}
    for label, chart, finalized, preterminal in rows:
        result_path = tmp_path / f"{label}.json"
        final_history_error = finalized if preterminal is None else preterminal
        history = [
            {
                "energy_before_opt": 0.0,
                "delta_abs_current": 0.1,
                "accepted_refit": accepted_refit(1),
            },
            {
                "energy_before_opt": -0.9,
                "delta_abs_current": final_history_error,
                "accepted_refit": accepted_refit(2),
            },
        ]
        adapt: dict[str, object] = {
            "exact_gs_energy": -1.0,
            "abs_delta_e": finalized,
            "history": history,
            "optimizer_coordinate_chart": {
                "powell_coordinate_chart_policy": chart,
            },
        }
        if label == "good_expanded_chart_whitened_r30":
            terminal_refit = accepted_refit(2)
            terminal_refit["accepted_refit_invocation"] = {
                "metric_query_accounting": {
                    "symmetric_metric_element_occurrences": 3,
                    "new_unique_metric_elements_charged": 3,
                    "deduplicated_or_ledger_disabled_count": 0,
                }
            }
            adapt["final_full_refit"] = {"accepted_refit": terminal_refit}
        result_path.write_text(
            json.dumps({"adapt_vqe": adapt}), encoding="utf-8"
        )
        result_paths[label] = result_path
        comparison = {
            "label": label,
            "powell_base_chart": chart,
            "rounds": 2,
            "finalized_abs_error": finalized,
            "result_path": str(result_path),
            "result_sha256": report._sha256(result_path),
        }
        if preterminal is not None:
            comparison["pre_terminal_checkpoint_abs_error"] = preterminal
        comparisons.append(comparison)

    r30_path = result_paths["good_expanded_chart_whitened_r30"]
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": report.SR_EXPANDED_CHART_WHITENING_VALIDATION_SCHEMA,
                "status": "validated_with_terminal_action_disclosure",
                "regime": "weak-weak",
                "artifacts": {
                    "result_path": str(r30_path),
                    "result_sha256": report._sha256(r30_path),
                },
                "comparisons": comparisons,
                "reference": {
                    "n_ph_work": 2,
                    "n_ph_ref": 2,
                    "same_cutoff_energy": -1.0,
                },
                "route": {
                    "family": "singleton_response_snake",
                    "profile": "supported_whitened_adaptive_trust_v1",
                    "powell_base_chart": "expanded_runtime_projected_logical_v1",
                    "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
                    "all_history_rows_used_expected_base_chart": True,
                    "all_history_rows_used_expected_whitening": True,
                },
                "result": {
                    "pre_terminal_checkpoint_abs_error": 1.7400664686917366e-8,
                    "pre_terminal_checkpoint_active_depth": 2,
                    "finalized_abs_error": 1.0373365499916076e-9,
                    "finalized_active_depth": 1,
                    "terminal_actions": {
                        "final_full_refit_executed": True,
                        "final_full_refit_nfev": 5,
                        "post_refit_phase1_prune_accepted_count_from_log": 1,
                    },
                },
                "validation": {
                    "estimator_ledger_complete": True,
                    "strict_terminal_replay_passed": True,
                    "strict_terminal_replay_fidelity": 1.0,
                    "maximum_fixed_sector_illegal_probability": 1e-15,
                    "maximum_binary_padding_illegal_probability": 1e-15,
                    "winning_lineage_s_alg": {"S_alg": 11},
                    "discarded_branch_unique_work": 2,
                    "all_branch_search_s_alg": 13,
                },
                "source_lock": {
                    "archive_path": str(source_archive),
                    "archive_sha256": report._sha256(source_archive),
                    "manifest_path": str(source_manifest),
                    "manifest_sha256": report._sha256(source_manifest),
                    "non_swept_settings_diff": [],
                    "approved_executable_diff": {"adapt_max_depth": {"from": 22, "to": 30}},
                },
            }
        ),
        encoding="utf-8",
    )
    return validation_path


def test_sr_expanded_chart_whitening_validation_loads_hash_closed_evidence(
    tmp_path: Path,
) -> None:
    path = _write_sr_expanded_chart_whitening_validation(tmp_path)

    campaign = report._load_sr_expanded_chart_whitening_validation(path)

    assert campaign["support_rank_sequence"] == [1, 2]
    assert campaign["round30_metric_accounting"] == {
        "logical_dimension": 2,
        "expanded_runtime_dimension": 3,
        "retained_support_rank": 2,
        "symmetric_metric_element_occurrences": 3,
        "new_unique_metric_elements_charged": 2,
        "deduplicated_metric_elements": 1,
    }
    assert campaign["terminal_refit_metric_accounting"]["fresh_chart"] is True
    assert campaign["terminal_refit_metric_accounting"][
        "new_unique_metric_elements_charged"
    ] == 3
    assert campaign["comparisons"][-1]["preterminal_checkpoint_error"] == pytest.approx(
        1.7400664686917366e-8
    )


def test_sr_expanded_chart_whitening_validation_fails_closed_on_result_hash_drift(
    tmp_path: Path,
) -> None:
    path = _write_sr_expanded_chart_whitening_validation(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    result_path = Path(payload["comparisons"][-1]["result_path"])
    result_path.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        report._load_sr_expanded_chart_whitening_validation(path)


def test_sr_expanded_chart_whitening_cli_is_opt_in_keyword_plumbing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.json"
    captured: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object) -> dict[str, str]:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(report, "build", fake_build)

    assert report.main(
        ["--sr-expanded-chart-whitening-validation-json", str(validation_path)]
    ) == 0
    assert captured["kwargs"] == {
        "sr_expanded_chart_whitening_validation_json": validation_path
    }


def test_build_page_count_is_six_then_additive_for_weak_weak_and_iw_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_campaign_presence: list[tuple[bool, bool]] = []

    def fake_build_evidence(*args: object, **kwargs: object) -> dict[str, object]:
        validation = kwargs.get("sr_expanded_chart_whitening_validation_json")
        iw_validation = kwargs.get(
            "sr_expanded_chart_whitening_intermediate_weak_validation_json"
        )
        campaign = (
            None
            if validation is None
            else {
                "validation_json": str(validation),
                "validation_sha256": "a" * 64,
            }
        )
        return {
            "pages": {},
            "status": "ok",
            "formal_manifold_campaign": {"pending_regimes": []},
            "sr_expanded_chart_whitening_campaign": campaign,
            "sr_expanded_chart_whitening_intermediate_weak_campaign": (
                None
                if iw_validation is None
                else {
                    "validation_json": str(iw_validation),
                    "validation_sha256": "b" * 64,
                    "qiskit": None,
                }
            ),
        }

    def fake_compile(tex_path: Path) -> Path:
        pdf_path = tex_path.with_suffix(".pdf")
        pdf_path.write_bytes(b"pdf")
        return pdf_path

    def fake_page_count(_path: Path) -> int:
        weak_weak_present, iw_present = observed_campaign_presence[-1]
        return 6 + int(weak_weak_present) + int(iw_present)

    monkeypatch.setattr(report, "build_evidence", fake_build_evidence)
    monkeypatch.setattr(report, "_plot_report_page", lambda *args, **kwargs: {})
    monkeypatch.setattr(report, "_write_model_tex", lambda *args, **kwargs: None)
    monkeypatch.setattr(report, "_compile_latex", fake_compile)
    monkeypatch.setattr(report, "_page_count", fake_page_count)

    observed_campaign_presence.append((False, False))
    report.build(output_dir=tmp_path / "six", stem="report")
    observed_campaign_presence.append((True, True))
    result = report.build(
        output_dir=tmp_path / "eight",
        stem="report",
        sr_expanded_chart_whitening_validation_json=tmp_path / "validation.json",
        sr_expanded_chart_whitening_intermediate_weak_validation_json=(
            tmp_path / "iw-validation.json"
        ),
    )

    assert result["sr_expanded_chart_whitening_validation_json"].endswith(
        "validation.json"
    )
    assert result[
        "sr_expanded_chart_whitening_intermediate_weak_validation_json"
    ].endswith("iw-validation.json")


def _write_sr_intermediate_weak_completed_validation(
    tmp_path: Path,
) -> tuple[Path, Path, Path]:
    from pipelines.exact_bench.generic_static_metric_enrichment import (
        _sha256_json_without_snake_sidecars,
    )

    source_archive = tmp_path / "source-tree.tar.gz"
    source_archive.write_bytes(b"source-tree")
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text('{"locked":true}\n', encoding="utf-8")
    reference_energy = -1.0

    def accepted_refit(index: int) -> dict[str, object]:
        return {
            "policy": "supported_fs_whitened_fixed_v1",
            "base_chart_policy": "expanded_runtime_projected_logical_v1",
            "supported_metric_whitening_policy": (
                "supported_metric_whitened_eigh_v1"
            ),
            "logical_parameter_count": index,
            "base_parameter_count": index + 1,
            "metric_support_rank": index,
            "chart_fixed_within_powell_invocation": True,
            "chart_recomputed_after_next_admission": True,
        }

    history = []
    for index in range(1, 31):
        error = 1.0e-8 if index == 30 else 0.1 / index
        history.append(
            {
                "energy_before_opt": 0.0 if index == 1 else history[-1]["energy_after_opt"],
                "energy_after_opt": reference_energy + error,
                "delta_abs_current": error,
                "depth_cumulative": index,
                "accepted_refit": accepted_refit(index),
            }
        )
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "adapt_vqe": {
                    "history": history,
                    "energy": reference_energy + 1.0e-9,
                    "abs_delta_e": 1.0e-9,
                    "exact_gs_energy": reference_energy,
                    "final_full_refit": {
                        "nfev": 365,
                        "accepted_refit": accepted_refit(30),
                    },
                    "prune_summary": {
                        "candidate_count": 1,
                        "accepted_count": 0,
                    },
                    "post_prune_refit": {
                        "executed": False,
                        "energy": reference_energy + 1.0e-9,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    artifact_paths = {
        "command": tmp_path / "command.json",
        "estimator_ledger": tmp_path / "estimator.json",
        "exit_status": tmp_path / "exit.json",
        "result": result_path,
        "settings_diff": tmp_path / "settings-diff.json",
    }
    for label, artifact_path in artifact_paths.items():
        if label != "result":
            artifact_path.write_text(json.dumps({"artifact": label}), encoding="utf-8")
    artifacts = {
        label: {"path": str(artifact_path), "sha256": report._sha256(artifact_path)}
        for label, artifact_path in artifact_paths.items()
    }
    validation_path = tmp_path / "iw-validation.json"
    validation_path.write_text(
        json.dumps(
            {
                "schema": report.SR_COMPLETED_RUN_VALIDATION_SCHEMA,
                "status": "validated",
                "regime": "intermediate-weak",
                "blockers": [],
                "artifacts": artifacts,
                "reference": {
                    "n_ph_work": 2,
                    "n_ph_ref": 2,
                    "same_cutoff_energy": reference_energy,
                },
                "result": {
                    "displayed_absolute_error": 1.0e-9,
                    "displayed_energy": reference_energy + 1.0e-9,
                    "displayed_replayed_energy_discrepancy": 0.0,
                    "finalized_active_depth": 30,
                    "pre_terminal_checkpoint_active_depth": 30,
                    "pre_terminal_checkpoint_replayed_absolute_error": 1.0e-8,
                    "pre_terminal_checkpoint_replayed_energy": reference_energy + 1.0e-8,
                    "replayed_absolute_error": 1.0e-9,
                    "replayed_energy": reference_energy + 1.0e-9,
                    "same_cutoff_reference_energy": reference_energy,
                    "stop_reason": "max_depth",
                },
                "route": {
                    "accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
                    "accepted_refit_whitening_policy": "supported_metric_whitened_eigh_v1",
                    "adaptive_trust_policy": "displacement_calibrated_unbounded_v2",
                    "all_history_rows_used_expected_base_chart": True,
                    "all_history_rows_used_expected_whitening": True,
                    "family": "singleton_response_snake",
                    "outer_round_horizon": 30,
                    "powell_base_chart": "expanded_runtime_projected_logical_v1",
                    "profile": "supported_whitened_adaptive_trust_v1",
                },
                "checkpoint_replay": {
                    "active_checkpoint_count": 30,
                    "leakage_tolerance": 1.0e-9,
                    "maximum_binary_padding_illegal_probability": 1.0e-15,
                    "maximum_fixed_sector_illegal_probability": 1.0e-15,
                    "rows": [{"outer_iteration": index} for index in range(1, 31)],
                    "terminal": {"checkpoint_kind": "terminal"},
                    "terminal_active_depth": 30,
                    "terminal_checkpoint_sha256": "a" * 64,
                    "terminal_state_fidelity_to_serialized_final_state": 1.0,
                },
                "estimator_accounting": {
                    "complete": True,
                    "status": "resolved_from_live_state_keyed_instrumentation",
                    "discarded_branch_unique_work": 10,
                    "winning_lineage": {
                        "N_H_outer": 1,
                        "N_H_refit": 20,
                        "N_grad": 30,
                        "N_metric": 49,
                        "S_alg": 100,
                    },
                    "all_branch_search_work": {
                        "N_H_outer": 1,
                        "N_H_refit": 30,
                        "N_grad": 30,
                        "N_metric": 49,
                        "S_alg": 110,
                    },
                },
                "source_lock": {
                    "archive_path": str(source_archive),
                    "archive_sha256": report._sha256(source_archive),
                    "manifest_path": str(source_manifest),
                    "manifest_sha256": report._sha256(source_manifest),
                    "runtime_tree": str(tmp_path / "runtime-tree"),
                    "verified_file_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    qiskit_path = tmp_path / "qiskit-sidecar.json"
    qiskit_path.write_text(
        json.dumps(
            {
                "schema": "paper_i_selected_prefix_qiskit_cost_sidecar_v1",
                "compile_convention": "table_i_basis_gate_transpile_v1",
                "compiled_resource_qiskit_validated": True,
                "compiled_circuit_stats_status": "ok",
                "source_result_path": str(result_path),
                "source_result_sha256": _sha256_json_without_snake_sidecars(result_path),
                "source_result_hash_convention": (
                    "canonical_json_without_snake_sidecars_v1"
                ),
                "history_position": 30,
                "logical_operator_count": 30,
                "runtime_rotation_count": 31,
                "energy_after_opt_at_prefix": reference_energy + 1.0e-9,
                "compiled_count_2q_total": 218,
                "compiled_depth_2q_total": 201,
                "compiled_depth_total": 1065,
            }
        ),
        encoding="utf-8",
    )
    return validation_path, qiskit_path, result_path


def test_sr_intermediate_weak_completed_validation_and_qiskit_are_hash_closed(
    tmp_path: Path,
) -> None:
    validation_path, qiskit_path, _ = _write_sr_intermediate_weak_completed_validation(
        tmp_path
    )

    campaign = report._load_sr_expanded_chart_whitening_intermediate_weak_validation(
        validation_path,
        qiskit_sidecar_path=qiskit_path,
    )

    assert len(campaign["trajectory_points"]) == 31
    assert campaign["result"]["displayed_absolute_error"] == pytest.approx(1.0e-9)
    assert campaign["qiskit"]["N2q"] == 218
    assert campaign["qiskit"]["D2q"] == 201
    assert campaign["qiskit"]["Dc"] == 1065
    assert campaign["qiskit"]["primary_error_at_prefix_ignored"] is True
    assert campaign["qiskit"]["sha256"] == report._sha256(qiskit_path)
    assert set(campaign["verified_artifacts"]) == {
        "command",
        "estimator_ledger",
        "exit_status",
        "result",
        "settings_diff",
    }


def test_sr_intermediate_weak_completed_validation_rejects_artifact_hash_drift(
    tmp_path: Path,
) -> None:
    validation_path, _, _ = _write_sr_intermediate_weak_completed_validation(tmp_path)
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    payload["artifacts"]["command"]["sha256"] = "f" * 64
    validation_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        report._load_sr_expanded_chart_whitening_intermediate_weak_validation(
            validation_path
        )


def test_sr_intermediate_weak_qiskit_rejects_source_hash_drift(
    tmp_path: Path,
) -> None:
    validation_path, qiskit_path, _ = _write_sr_intermediate_weak_completed_validation(
        tmp_path
    )
    payload = json.loads(qiskit_path.read_text(encoding="utf-8"))
    payload["source_result_sha256"] = "f" * 64
    qiskit_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="source-result hash mismatch"):
        report._load_sr_expanded_chart_whitening_intermediate_weak_validation(
            validation_path,
            qiskit_sidecar_path=qiskit_path,
        )


def test_sr_intermediate_weak_validation_and_qiskit_cli_are_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_path = tmp_path / "validation.json"
    qiskit_path = tmp_path / "qiskit.json"
    captured: dict[str, object] = {}

    def fake_build(*args: object, **kwargs: object) -> dict[str, str]:
        captured["kwargs"] = kwargs
        return {"status": "ok"}

    monkeypatch.setattr(report, "build", fake_build)

    assert report.main(
        [
            "--sr-expanded-chart-whitening-intermediate-weak-validation-json",
            str(validation_path),
            "--sr-expanded-chart-whitening-intermediate-weak-qiskit-json",
            str(qiskit_path),
        ]
    ) == 0
    assert captured["kwargs"] == {
        "sr_expanded_chart_whitening_intermediate_weak_validation_json": (
            validation_path
        ),
        "sr_expanded_chart_whitening_intermediate_weak_qiskit_json": qiskit_path,
    }


def test_sr_intermediate_weak_page_tex_discloses_terminal_and_qiskit_sources(
    tmp_path: Path,
) -> None:
    validation_path, qiskit_path, _ = _write_sr_intermediate_weak_completed_validation(
        tmp_path
    )
    campaign = report._load_sr_expanded_chart_whitening_intermediate_weak_validation(
        validation_path,
        qiskit_sidecar_path=qiskit_path,
    )
    source = report._sr_expanded_chart_whitening_intermediate_weak_page_tex(
        {
            "title": "IW validation",
            "subtitle": "diagnostic only",
            "campaign": campaign,
        },
        image_path=tmp_path / "plot.png",
    )

    assert "2.000000000000e+02" not in source
    assert "2q}=218" in source
    assert "D_c=1065" in source
    assert "final full refit 365 nfev" in source
    assert "terminal prune nominated 1 and accepted 0" in source
    assert campaign["validation_sha256"] in source.replace(r"\allowbreak{}", "")
