from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pipelines.reporting import build_paper_i_hh_sr_no_prune_no_beam_tracking_pdf as tracker
from pipelines.reporting.build_paper_i_hh_cost_arm_tracking_summary import (
    SCHEMA,
    build_tracking_summary,
)
from pipelines.reporting import build_paper_i_hh_tracking_plateau_costs as plateau
from pipelines.reporting import build_paper_i_hh_tracking_target_costs as target


DIGEST = "a05ecc8b709db8beac9115d9d0ca39f4faf09e1cbaa10e57bdd674abef9215f0"
RESULT_SHA = "f" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _synthetic_pass(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    row_dir = tmp_path / "v9_revalidated/8900509.0__weak_weak"
    row_dir.mkdir(parents=True)
    archive = tmp_path / "8900509.0__weak_weak_transfer.tar.gz"
    archive.write_bytes(b"immutable raw archive placeholder")
    archive_sha = _sha(archive)

    checkpoint = {
        "schema": "paper_i_checkpoint_execution_order_repair_v1",
        "repair": {
            "status": "repaired_permutation_only",
            "substantive_term_changes": False,
            "source_checkpoint_sha256": "b" * 64,
            "repaired_checkpoint_sha256": "c" * 64,
        },
        "source": {"result_sha256": RESULT_SHA, "outer_iteration": 50},
        "repaired_checkpoint": {
            "schema": "paper_i_signed_active_prefix_checkpoint_v1",
            "checkpoint_sha256": "c" * 64,
            "outer_iteration": 50,
            "active_ansatz_depth": 49,
            "sr_route_profile_contract_sha256": DIGEST,
            "ordered_active_operator_labels": [f"g{index}" for index in range(49)],
            "ordered_active_operators": [[{"label": "x", "coeff": 1.0}]] * 49,
            "signed_unwrapped_logical_parameters": [0.1] * 49,
            "signed_unwrapped_runtime_parameters": [0.1] * 60,
            "estimator_ledger_receipt": {
                "status": "complete",
                "outer_iteration": 50,
            },
        },
    }
    validation = {
        "schema": "paper_i_hh_sr_fs_prune_nodamping_validation_v1",
        "status": "pass",
        "controller_horizon_round": 50,
        "selected_winner_round": 50,
        "result_sha256": RESULT_SHA,
        "same_cutoff_exact_energy": -1.0,
        "fixed_prefix_replay": {
            "status": "pass",
            "reported_energy": -0.99,
            "active_ansatz_depth": 49,
        },
        "current_fake_marrakesh_metrics": {"N2q": 101, "D2q": 77, "Dc": 222},
        "historical_metrics": {"N2q": 61, "D2q": 44, "Dc": 155},
    }
    qiskit = {
        "schema": "paper_i_hh_recovery_prune_aware_qiskit_sidecar_v1",
        "status": "ok",
        "source": {"result_sha256": RESULT_SHA},
        "current_jr_fake_marrakesh_convention": {
            "metrics": {"N2q": 101, "D2q": 77, "Dc": 222}
        },
        "historical_displayed_convention": {
            "metrics": {"N2q": 61, "D2q": 44, "Dc": 155}
        },
    }
    fidelity = {
        "schema": "paper_i_hh_sr_post_run_projector_fidelity_receipt_v1",
        "status": "pass",
        "source_result_sha256": RESULT_SHA,
        "fidelity": 0.999,
        "ground_space_fidelity": {
            "same_cutoff_verified": True,
            "working_cutoff": 3,
            "reference_cutoff": 3,
        },
    }
    paths = {
        "terminal_checkpoint.execution_order_repaired.json": checkpoint,
        "validation.json": validation,
        "qiskit_cost_sidecar.json": qiskit,
        "ground_space_projector_fidelity.json": fidelity,
    }
    for name, payload in paths.items():
        _write(row_dir / name, payload)

    generated = {
        name: {"sha256": _sha(row_dir / name), "size_bytes": (row_dir / name).stat().st_size}
        for name in paths
    }
    receipt = {
        "schema": "paper_i_sr_macro_beam_cost_v9_v6_archive_revalidation_v1",
        "status": "pass",
        "scientific_rerun_required": False,
        "raw_transfer_archive_preserved": True,
        "raw_transfer_archive": str(archive),
        "raw_transfer_archive_sha256_before": archive_sha,
        "raw_transfer_archive_sha256_after": archive_sha,
        "regime_slug": "weak_weak",
        "profile_contract_sha256": DIGEST,
        "generated_reporting_artifacts": generated,
        "source_only_runtime_settings_receipt": {
            "status": "pass",
            "profile_contract_sha256": DIGEST,
            "phase_live_hysteresis_disabled": True,
            "behavioral_closure": "full_response_validated_each_controller_round_v1",
            "source_only_runtime_settings": {
                "adapt_beam_live_branches": 3,
                "adapt_beam_children_per_parent": 2,
                "phase0_pilot_enabled": False,
                "phase3_enable_batching": False,
                "adapt_accepted_refit_coordinate_chart": "supported_fs_whitened_fixed_v1",
            },
        },
        "scientific_evidence_validation": {
            "controller_rounds": 50,
            "compact_current_history_receipt": {"status": "pass", "rounds": 50},
            "active_prefix_estimator_ledger_receipts": {
                "closure_passed": True,
                "controller_horizon_rounds": 50,
                "all_branch_S_alg": 1000,
                "receipt_count": 50,
            },
            "ledger": {
                "all_branch_s_alg": 1000,
                "winning_lineage_s_alg": 900,
                "finite_angle_guard_occurrence_count": 0,
                "ledger_fingerprint": "a" * 64,
            },
            "expected_cost_mode": "family_robust_symmetric_arctan_v1",
            "expected_fallback_policy": "collective_span_novelty_over_symmetric_cost_v1",
            "max_binary_padding_leakage": 1.0e-15,
            "max_fixed_sector_leakage": 2.0e-15,
            "selected_final_controller_round": 50,
            "selected_final_active_depth": 49,
            "selected_prune_rounds_executed": 1,
            "selected_prune_rounds_accepted": 1,
            "selected_terminal_checkpoint_sha256": "b" * 64,
        },
    }
    receipt_path = row_dir / "revalidation_receipt.json"
    _write(receipt_path, receipt)

    trajectory = {
        "schema": "paper_i_sr_macro_beam_cost_compact_trajectory_v1",
        "status": "pass",
        "regime_slug": "weak_weak",
        "profile_contract_sha256": DIGEST,
        "raw_transfer_archive": {
            "path": str(archive),
            "sha256": archive_sha,
            "size_bytes": archive.stat().st_size,
        },
        "result_member": {
            "name": "payload/weak_weak/json/result.json",
            "sha256": RESULT_SHA,
            "size_bytes": 123456,
        },
        "controller_rounds": 50,
        "selected_final_controller_round": 50,
        "selected_final_active_depth": 49,
        "all_branch_S_alg": 1000,
        "winning_lineage_S_alg": 900,
        "construction_receipt": {
            "all_branch_ledger_closure_matches_v9": True,
            "winning_lineage_ledger_closure_matches_v9": True,
            "archive_hash_matches_v9_before_and_after": True,
            "trajectory_rounds_exactly_1_through_50": True,
            "all_branch_unique_checkpoint_receipt_count": 50,
            "estimator_ledger_unique_primitive_entry_count": 1000,
        },
        "selected_prefix_identity": {
            "ledger_fingerprint": "a" * 64,
            "terminal_checkpoint_sha256": "b" * 64,
            "repaired_terminal_checkpoint_sha256": _sha(
                row_dir / "terminal_checkpoint.execution_order_repaired.json"
            ),
        },
        "trajectory": [
            {
                "round": index,
                "error": 0.5 - (0.49 * index / 50),
                "active_depth": 49 if index == 50 else index,
                "prune_accepted": index == 50,
                "S_alg": 18 * index,
                "winning_lineage_S_alg": 18 * index,
                "all_branch_S_alg": 20 * index,
            }
            for index in range(1, 51)
        ],
    }
    compact_path = row_dir / "compact_trajectory.json"
    _write(compact_path, trajectory)
    summary_path = row_dir / "tracking_summary.json"
    return archive, receipt_path, compact_path, summary_path


def test_cost_arm_summary_is_pass_only_and_uses_small_executable_source(tmp_path) -> None:
    archive, receipt, compact, summary_path = _synthetic_pass(tmp_path)

    summary = build_tracking_summary(
        archive_path=archive,
        revalidation_receipt_path=receipt,
        compact_trajectory_path=compact,
        output_json=summary_path,
    )

    assert summary["schema"] == SCHEMA
    assert summary["status"] == "pass"
    assert summary["archive"]["sha256"] == _sha(archive)
    assert summary["result"]["terminal_error"] == pytest.approx(0.01)
    assert summary["result"]["trajectory"][-1]["S_alg"] == 1000
    assert summary["terminal_prefix_qiskit"] == {"N2q": 61, "D2q": 44, "Dc": 155}
    assert summary["executable_source"]["path"].endswith(
        "terminal_checkpoint.execution_order_repaired.json"
    )


def test_tracker_ingests_pass_summary_without_hashing_raw_archive(
    tmp_path, monkeypatch
) -> None:
    archive, receipt, compact, summary_path = _synthetic_pass(tmp_path)
    build_tracking_summary(
        archive_path=archive,
        revalidation_receipt_path=receipt,
        compact_trajectory_path=compact,
        output_json=summary_path,
    )
    monkeypatch.setattr(tracker, "REPO_ROOT", tmp_path)
    original_sha = tracker._sha256

    def guarded_sha(path: Path) -> str:
        if path.resolve() == archive.resolve():
            raise AssertionError("tracker must not reopen the raw cost-arm archive")
        return original_sha(path)

    monkeypatch.setattr(tracker, "_sha256", guarded_sha)
    result, costs, sources = tracker._cost_arm_tracking_summary(
        summary_path=summary_path,
        regime="weak_weak",
        expected_arm="symmetric",
        expected_route_digest=DIGEST,
        expected_cost_mode="family_robust_symmetric_arctan_v1",
        expected_fallback_policy="collective_span_novelty_over_symmetric_cost_v1",
    )

    assert result["status"] == "complete"
    assert result["source"]["path"].endswith(
        "terminal_checkpoint.execution_order_repaired.json"
    )
    assert result["raw_archive_provenance"]["raw_archive_not_reopened_by_tracker"] is True
    assert costs == {"N2q": 101, "D2q": 77, "Dc": 222}
    assert any(source["path"].endswith("revalidation_receipt.json") for source in sources)


def test_pending_cost_rows_are_not_added(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(tracker, "COMPARATOR_LATE_FETCH", tmp_path / "missing")
    routes = tracker._build_pass_only_cost_arm_routes([])
    assert routes == []


def test_pass_only_upsert_adds_only_validated_cells_and_reconciles_note(
    tmp_path, monkeypatch
) -> None:
    archive, receipt, compact, summary_path = _synthetic_pass(tmp_path)
    build_tracking_summary(
        archive_path=archive,
        revalidation_receipt_path=receipt,
        compact_trajectory_path=compact,
        output_json=summary_path,
    )
    monkeypatch.setattr(tracker, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(tracker, "COMPARATOR_LATE_FETCH", tmp_path)

    routes = tracker._upsert_pass_only_cost_arm_routes([], sources=[])
    notes = tracker._reconcile_cost_arm_pending_notes(
        tracker.PENDING_COST_ARM_NOTES,
        routes=routes,
    )

    assert [route["id"] for route in routes] == [
        "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7"
    ]
    assert routes[0]["results"]["weak_weak"]["status"] == "complete"
    assert routes[0]["results"]["intermediate_weak"]["status"] != "complete"
    symmetric_note = next(
        note for note in notes if note["route_id"] == routes[0]["id"]
    )
    assert symmetric_note["status"] == "partial_pass_only_evidence"
    assert symmetric_note["passed_regimes"] == ["weak_weak"]
    assert "intermediate_weak" in symmetric_note["pending_regimes"]


def test_nonpass_summary_fails_closed(tmp_path) -> None:
    archive, receipt, compact, summary_path = _synthetic_pass(tmp_path)
    summary = build_tracking_summary(
        archive_path=archive,
        revalidation_receipt_path=receipt,
        compact_trajectory_path=compact,
        output_json=summary_path,
    )
    summary["status"] = "pending"
    _write(summary_path, summary)
    with pytest.raises(RuntimeError, match="not a pass"):
        tracker._cost_arm_tracking_summary(
            summary_path=summary_path,
            regime="weak_weak",
            expected_arm="symmetric",
            expected_route_digest=DIGEST,
            expected_cost_mode="family_robust_symmetric_arctan_v1",
            expected_fallback_policy="collective_span_novelty_over_symmetric_cost_v1",
        )


def test_cost_arm_prefix_builders_use_terminal_checkpoint_and_never_raw_archive(
    tmp_path, monkeypatch
) -> None:
    archive, receipt, compact, summary_path = _synthetic_pass(tmp_path)
    build_tracking_summary(
        archive_path=archive,
        revalidation_receipt_path=receipt,
        compact_trajectory_path=compact,
        output_json=summary_path,
    )
    monkeypatch.setattr(tracker, "REPO_ROOT", tmp_path)
    result, _costs, _sources = tracker._cost_arm_tracking_summary(
        summary_path=summary_path,
        regime="weak_weak",
        expected_arm="symmetric",
        expected_route_digest=DIGEST,
        expected_cost_mode="family_robust_symmetric_arctan_v1",
        expected_fallback_policy="collective_span_novelty_over_symmetric_cost_v1",
    )
    tracker_json = tmp_path / "tracker.json"
    _write(
        tracker_json,
        {
            "schema": "test_tracker",
            "routes": [
                {
                    "id": "sr_macro_beam3x2_fs_prune_symmetric_cost_nph3_7",
                    "results": {"weak_weak": result},
                }
            ],
        },
    )
    monkeypatch.setattr(plateau, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(target, "REPO_ROOT", tmp_path)

    plateau_payload = plateau.build_plateau_costs(
        tracker_json=tracker_json,
        output_json=tmp_path / "plateau.json",
    )
    target_payload = target.build_target_costs(
        tracker_json=tracker_json,
        output_json=tmp_path / "target.json",
    )

    assert plateau_payload["rows"][0]["k_pl"] == 50
    assert plateau_payload["rows"][0]["S_alg"] == 1000
    assert plateau_payload["rows"][0]["qiskit"] == {
        "N2q": 61,
        "D2q": 44,
        "Dc": 155,
    }
    assert plateau_payload["rows"][0]["prefix_source"]["raw_archive_reopened"] is False
    assert target_payload["rows"] == []
    assert target_payload["unresolved"][0]["status"] == "threshold_not_reached"
