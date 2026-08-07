from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILDER_DIR = ROOT / "chtc" / "phase3_optuna"
if str(BUILDER_DIR) not in sys.path:
    sys.path.insert(0, str(BUILDER_DIR))

import build_paper_i_hh_sr_material_window_fanout_20260721 as builder


def _extension_module(tmp_path: Path):
    path = tmp_path / "evidence_validation.py"
    path.write_text(
        builder.anchor.RECOVERED_VALIDATOR.read_text(encoding="utf-8").rstrip()
        + "\n"
        + builder._material_validator_extension(),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("material_window_test_evidence", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rank_gain_one_receipt(module):
    receipt = {
        "receipt_version": "phase3_material_window_receipt_v1",
        "policy": copy.deepcopy(module.MATERIAL_WINDOW_POLICY),
        "active_indices": [0],
        "prior_active_nullity": 0,
        "prior_joint_nullity": 0,
        "gram_normalized_scores": [1.0],
        "hessian_normalized_scores": [1.0],
        "initial_gram_mask": [True],
        "initial_hessian_mask": [True],
        "initial_union_mask": [True],
        "final_retained_mask": [True],
        "closure_added_indices": [],
        "retained_indices": [0],
        "omitted_indices": [],
        "initial_gram_omitted_l2_ratio": 0.0,
        "initial_hessian_omitted_l2_ratio": 0.0,
        "final_gram_omitted_l2_ratio": 0.0,
        "final_hessian_omitted_l2_ratio": 0.0,
        "gram_entry_threshold": 4.0e-3,
        "hessian_entry_threshold": 2.0e-22,
        "gram_omitted_l2_tolerance": 1.0,
        "hessian_omitted_l2_tolerance": 1.0,
        "inputs_finite": True,
        "closure_satisfied": True,
        "closure_reason": "satisfied_by_threshold_union",
        "measured_active_supported_rank": 1,
        "measured_joint_supported_rank": 2,
        "measured_active_nullity": 0,
        "measured_joint_nullity": 0,
        "measured_rank_gain": 1,
        "support_nullity_drift": False,
        "requires_full_geometry_refresh": False,
        "refresh_reasons": [],
    }
    receipt["receipt_sha256"] = module._material_receipt_digest(receipt)
    return receipt


def _actual_anchor_validation_schema_fixture(result_sha256: str) -> dict:
    """Exact fetched-anchor shape: scientific metrics have no nested status."""
    fallback_rounds = [
        14, 15, 31, 32, 34, 36, 37, 39, 40,
        42, 43, 44, 45, 46, 47, 48, 49, 50,
    ]
    return {
        "schema": "paper_i_hh_sr_symcost_noprune_fetched_validation_v1",
        "status": "pass",
        "result_sha256": result_sha256,
        "ledger_schema": "paper_i_estimator_call_ledger_sidecar_v1",
        "profile_contract_sha256": builder.anchor.PARENT_DIGEST,
        "target_controller_round": 50,
        "scientific_evidence_validation": {
            "controller_rounds": 50,
            "new_admissions": 50,
            "final_active_depth": 50,
            "adaptive_trust_updates": 50,
            "phase3_response_scope": "full_active_plus_singleton_v1",
            "supported_rank_recorded_each_round": True,
            "terminal_state_unchanged_from_last_ordinary_round": True,
            "terminal_checkpoint_sha256": "8" * 64,
            "prune_rounds_executed": 0,
            "ordinary_phase2_novelty_multiplier_active": False,
            "ordinary_phase3_novelty_multiplier_active": False,
            "phase1_lambda_f_proxy_occurrences": 0,
            "phase2_lambda_f_proxy_occurrences": 0,
            "phase2_missing_curvature_fallback_occurrences": 0,
            "phase2_full_candidate_occurrences": 3076,
            "validated_phase2_curvature_receipt_occurrences": 3076,
            "max_binary_padding_leakage": 1.4e-14,
            "max_fixed_sector_leakage": 1.5e-14,
            "infeasible_model_fallback_activation_count": len(fallback_rounds),
            "infeasible_model_fallback_controller_rounds": fallback_rounds,
            "infeasible_model_fallback_enabled": True,
            "infeasible_model_fallback_fired": True,
            "active_prefix_estimator_ledger_receipts": {
                "schema": "paper_i_active_prefix_estimator_ledger_closure_v1",
                "closure_passed": True,
                "S_alg": 233458,
                "raw_occurrence_count": 338610,
                "round_receipt_count": 50,
                "terminal_receipt_count": 1,
                "receipt_count": 51,
            },
            "ledger": {
                "all_branch_s_alg": 233458,
                "winning_lineage_s_alg": 233458,
                "raw_entry_count": 233458,
                "raw_occurrence_count": 338610,
                "finite_angle_guard_occurrence_count": 0,
                "ledger_fingerprint": "1" * 64,
            },
        },
        "projected_generalized_phase3_validation": {
            "schema": "paper_i_sr_projected_generalized_phase3_evidence_v1",
            "status": "pass",
            "controller_rounds": 50,
            "projected_solver_receipt_count": 50,
            "feasible_solver_receipt_count": 32,
            "infeasible_solver_receipt_count": 18,
            "projection_provenance_count": 32,
            "supported_metric_whitening_active": False,
            "accepted_powell_refit_whitening_active": True,
            "classical_quantum_query_charge": 0,
        },
        "no_overlap_trust_validation": {
            "schema": "paper_i_sr_source_metric_no_overlap_trust_evidence_v1",
            "status": "pass",
            "controller_rounds": 50,
            "expansion_count": 2,
            "contraction_count": 17,
            "hold_count": 31,
            "geometry_expansion_no_overlap_hold_count": 18,
            "initial_zero_active_no_overlap_hold_count": 1,
            "source_metric_receipt_count": 31,
            "source_metric_displacement_unresolved_hold_count": 0,
            "source_metric_transaction_failure_hold_count": 0,
            "endpoint_overlap_measurement_count": 0,
            "endpoint_overlap_query_charge": 0,
            "accepted_powell_refit_whitening_active": True,
        },
    }


def test_exact_locked_anchor_source_is_authority(tmp_path: Path) -> None:
    archive, files, overlays, contracts = builder._locked_anchor_source(tmp_path)
    assert builder.anchor.sha256(archive) == builder.ANCHOR_SOURCE_ARCHIVE_SHA256
    assert len(files) == 395
    assert overlays
    assert contracts[builder.anchor.PARENT_ALIAS]["digest"] == builder.anchor.PARENT_DIGEST
    assert contracts[builder.anchor.CHILD_ALIAS]["digest"] == builder.anchor.CHILD_DIGEST


def test_actual_anchor_validation_schema_closes_without_nested_status() -> None:
    result_sha = "a" * 64
    receipt = _actual_anchor_validation_schema_fixture(result_sha)
    assert "status" not in receipt["scientific_evidence_validation"]
    checked = builder._validate_anchor_validation_receipt(
        validation=receipt, result_sha256=result_sha
    )
    assert checked == {
        "schema": "paper_i_material_window_anchor_validation_contract_v1",
        "status": "pass",
        "validation_schema": "paper_i_hh_sr_symcost_noprune_fetched_validation_v1",
        "controller_rounds": 50,
        "S_alg": 233458,
        "raw_occurrence_count": 338610,
        "fallback_count": 18,
        "projected_feasible_count": 32,
        "endpoint_overlap_query_charge": 0,
    }

    runtime = copy.deepcopy(receipt)
    runtime["schema"] = "paper_i_hh_sr_symcost_noprune_validation_v1"
    for key in (
        "ledger_schema", "profile_contract_sha256", "target_controller_round",
    ):
        runtime.pop(key)
    runtime_checked = builder._validate_anchor_validation_receipt(
        validation=runtime, result_sha256=result_sha
    )
    assert runtime_checked["status"] == "pass"
    assert runtime_checked["validation_schema"] == (
        "paper_i_hh_sr_symcost_noprune_validation_v1"
    )

    broken = copy.deepcopy(receipt)
    broken["scientific_evidence_validation"][
        "active_prefix_estimator_ledger_receipts"
    ]["closure_passed"] = False
    with pytest.raises(ValueError, match="estimator-ledger closure drift"):
        builder._validate_anchor_validation_receipt(
            validation=broken, result_sha256=result_sha
        )


def test_exact_six_regime_matrix_and_cutoffs() -> None:
    jobs = sorted(builder.BASE.joinpath("jobs").glob("*.json"))
    assert len(jobs) == 6
    for path in jobs:
        builder._validate_regime_contract(json.loads(path.read_text(encoding="utf-8")))
    weak_weak = next(path for path in jobs if path.stem == "weak_weak")
    drifted = json.loads(weak_weak.read_text(encoding="utf-8"))
    drifted["physics"]["n_ph_work"] = 7
    with pytest.raises(ValueError, match="same-cutoff"):
        builder._validate_regime_contract(drifted)


def test_wrapper_patches_use_material_scope_and_real_validators(tmp_path: Path) -> None:
    _archive, _files, _overlays, contracts = builder._locked_anchor_source(
        tmp_path / "locked"
    )
    run_job = tmp_path / "run_job.py"
    run_job.write_bytes((builder.BASE / "run_job.py").read_bytes())
    builder._patch_run_job(
        run_job,
        parent_profile=contracts[builder.anchor.PARENT_ALIAS]["contract"]["route_profile"],
        child_profile=contracts[builder.anchor.CHILD_ALIAS]["contract"]["route_profile"],
    )
    run_text = run_job.read_text(encoding="utf-8")
    compile(run_text, str(run_job), "exec")
    assert '"phase3_response_coordinate_scope": MATERIAL_WINDOW_SCOPE' in run_text
    required_execution = run_text.split("required_execution = {", 1)[1].split("}", 1)[0]
    required_semantics = run_text.split("required_semantics = {", 1)[1].split("}", 1)[0]
    assert "phase3_material_window_support_change_policy" not in required_execution
    assert "phase3_material_window_support_change_policy" in required_semantics
    assert 'f"{slug}-sr-material-window-r0-r{target}-20260721-v1"' in run_text
    assert 'f"{slug}-sr-no-overlap-trust-r0-r{target}-20260720-v2"' not in run_text
    assert "material_no_overlap_validation_view(result)" in run_text
    assert "paper_i_sr_material_window_no_overlap_trust_evidence_v1" not in run_text

    fetched = tmp_path / "validate_fetched.py"
    fetched.write_bytes((builder.BASE / "validate_fetched.py").read_bytes())
    builder._patch_validate_fetched(
        fetched,
        child_profile=contracts[builder.anchor.CHILD_ALIAS]["contract"]["route_profile"],
    )
    fetched_text = fetched.read_text(encoding="utf-8")
    compile(fetched_text, str(fetched), "exec")
    assert "material_window_validation" in fetched_text
    assert "runtime/fetched material-window validation mismatch" in fetched_text
    assert "material_no_overlap_validation_view(result)" in fetched_text


def test_rank_gain_one_is_valid_and_receipt_hash_is_fail_closed(tmp_path: Path) -> None:
    module = _extension_module(tmp_path)
    receipt = _rank_gain_one_receipt(module)
    checked = module._validate_material_receipt(
        receipt, active=[0], field="rank-gain-one receipt"
    )
    assert checked["rank_gain"] == 1
    assert checked["drift"] is False
    assert receipt["requires_full_geometry_refresh"] is False

    tampered = copy.deepcopy(receipt)
    tampered["measured_rank_gain"] = 0
    with pytest.raises(ValueError, match="identity/digest drift"):
        module._validate_material_receipt(
            tampered, active=[0], field="tampered receipt"
        )


def test_projected_solver_classifies_exact_zero_query_fallback(tmp_path: Path) -> None:
    module = _extension_module(tmp_path)
    fallback = {
        "all_energy_models_infeasible_novelty_fallback_fired": True,
        "all_energy_models_infeasible_novelty_fallback_enabled": True,
        "all_energy_models_infeasible_novelty_fallback_query_charge": 0,
    }
    assert module._validate_material_projected_solver_receipt(
        row=fallback, summary={}, expected_round=14
    ) is True

    charged = copy.deepcopy(fallback)
    charged["all_energy_models_infeasible_novelty_fallback_query_charge"] = 1
    with pytest.raises(ValueError, match="zero-query fallback receipt drift"):
        module._validate_material_projected_solver_receipt(
            row=charged, summary={}, expected_round=14
        )


def test_validation_views_preserve_nonfinite_diagnostic_values(tmp_path: Path) -> None:
    module = _extension_module(tmp_path)
    payload = {
        "settings": {},
        "adapt_vqe": {"history": []},
        "diagnostic_only": float("-inf"),
    }
    assert module.material_parent_validation_view(payload)["diagnostic_only"] == float(
        "-inf"
    )
    assert module.material_no_overlap_validation_view(payload)[
        "diagnostic_only"
    ] == float("-inf")
