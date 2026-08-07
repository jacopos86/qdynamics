from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    HIGH_MISS_NO_ADMIT_POLICY_DEFAULT,
    HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON,
    HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING,
    CheckpointLedgerEntry,
    GeometryValueKey,
    OracleValueKey,
    RealtimeCheckpointConfig,
    dataclass_to_payload,
    full_horizon_completion_fields,
    hash_statevector,
    hash_theta_vector,
    high_miss_no_admit_diagnostic_counts,
    high_miss_no_admit_soft_fallback_counts,
    normalize_high_miss_no_admit_policy,
    validate_scaffold_acceptance,
)


def test_validate_scaffold_acceptance_accepts_readapt_5op_artifact() -> None:
    payload = json.loads(
        (REPO_ROOT / "artifacts" / "json" / "hh_prune_nighthawk_readapt_5op.json").read_text(encoding="utf-8")
    )
    result = validate_scaffold_acceptance(payload)
    assert bool(result.accepted) is True
    assert str(result.reason) == "accepted"


def test_validate_scaffold_acceptance_rejects_locked_7term_artifact() -> None:
    payload = json.loads(
        (REPO_ROOT / "artifacts" / "json" / "hh_prune_nighthawk_gate_pruned_7term.json").read_text(encoding="utf-8")
    )
    result = validate_scaffold_acceptance(payload)
    assert bool(result.accepted) is False
    assert bool(result.structure_locked) is True


def test_high_miss_no_admit_policy_default_and_alias_normalization() -> None:
    assert RealtimeCheckpointConfig().high_miss_no_admit_policy == HIGH_MISS_NO_ADMIT_POLICY_DEFAULT
    assert normalize_high_miss_no_admit_policy(None) == "bounded_stay_advance"
    assert normalize_high_miss_no_admit_policy("") == "bounded_stay_advance"
    assert normalize_high_miss_no_admit_policy(" bounded_stay_advance ") == "bounded_stay_advance"
    assert normalize_high_miss_no_admit_policy("legacy_advance_stay") == "bounded_stay_advance"
    assert normalize_high_miss_no_admit_policy("repair_stop") == "repair_stop"
    assert normalize_high_miss_no_admit_policy("repair_retry") == "repair_retry"
    with pytest.raises(ValueError, match="high_miss_no_admit_policy"):
        normalize_high_miss_no_admit_policy("ordinary_stay")


def test_high_miss_no_admit_soft_fallback_counts() -> None:
    rows = [
        {"action_kind": "stay", "high_miss_no_admit_soft_fallback": False},
        {
            "action_kind": "stay",
            "high_miss_no_admit_soft_fallback": True,
            "high_miss_no_admit_soft_fallback_reason": HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON,
        },
        {"action_kind": "append_candidate"},
    ]
    counts = high_miss_no_admit_soft_fallback_counts(rows)
    assert counts["high_miss_no_admit_soft_fallback_count"] == 1
    assert counts["high_miss_no_admit_soft_fallback_warning_count"] == 1
    assert counts["ordinary_stay_count"] == 1
    assert counts["high_miss_no_admit_soft_fallback_reason_counts"] == {
        HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON: 1
    }


def test_high_miss_no_admit_diagnostic_counts_promotes_first_bad_checkpoint() -> None:
    rows = [
        {
            "checkpoint_index": 0,
            "time": 0.0,
            "action_kind": "append_candidate",
            "controller_lane": "append",
            "candidate_label": "op_x",
        },
        {
            "checkpoint_index": 1,
            "time": 1.0,
            "action_kind": "stay",
            "proposed_action_kind": "append_candidate",
            "controller_lane": "append",
            "controller_lane_reason": "exact_rho_miss_above_threshold",
            "rho_miss": 0.2,
            "append_no_harm_veto_reason": "no_harm_condition_worse",
            "repair_no_admit_diagnostics": {
                "strict_no_admit_reason": "no_harm_condition_worse",
                "no_admit_resolution": "bounded_stay_advance",
            },
        },
        {"checkpoint_index": 2, "time": 2.0, "action_kind": "stay", "controller_lane": "stay"},
    ]

    counts = high_miss_no_admit_diagnostic_counts(rows)

    assert counts["high_miss_count"] == 2
    assert counts["high_miss_fraction"] == pytest.approx(2.0 / 3.0)
    assert counts["high_miss_no_admit_count"] == 1
    assert counts["high_miss_no_admit_fraction"] == pytest.approx(1.0 / 3.0)
    assert counts["high_miss_no_admit_reason_counts"] == {"no_harm_condition_worse": 1}
    assert counts["high_miss_no_admit_resolution_counts"] == {"bounded_stay_advance": 1}
    assert counts["append_no_harm_veto_count"] == 1
    assert counts["append_no_harm_veto_reason_counts"] == {"no_harm_condition_worse": 1}
    first = counts["first_bad_high_miss_no_admit_checkpoint_diagnostic"]
    assert first["checkpoint_index"] == 1
    assert first["high_miss_no_admit_reason"] == "no_harm_condition_worse"
    assert first["repair_no_admit_diagnostics"]["no_admit_resolution"] == "bounded_stay_advance"


def test_full_horizon_completion_fields_gate_uses_time_rows_and_early_stop() -> None:
    passed = full_horizon_completion_fields(
        [{"time": 0.0}, {"time": 1.0}, {"time": 2.0}],
        expected_t_final=2.0,
        expected_row_count=3,
    )
    stopped = full_horizon_completion_fields(
        [{"time": 0.0}, {"time": 1.0}],
        expected_t_final=2.0,
        expected_row_count=3,
        early_stop_reason="repair_required_high_miss_no_admit",
    )

    assert passed["full_horizon_gate_passed"] is True
    assert passed["full_horizon_gate_reason"] == "passed"
    assert passed["full_horizon_completion_kind"] == "completed"
    assert passed["full_horizon_successful_early_stop"] is False
    assert stopped["full_horizon_gate_passed"] is False
    assert stopped["full_horizon_gate_reason"] == "early_stop:repair_required_high_miss_no_admit"
    assert stopped["full_horizon_completion_kind"] == "failed"
    assert stopped["full_horizon_successful_early_stop"] is False
    assert stopped["full_horizon_reached_final_time"] is False
    assert stopped["full_horizon_reached_expected_rows"] is False


def test_full_horizon_completion_fields_accepts_stable_observable_early_stop() -> None:
    stable = full_horizon_completion_fields(
        [{"time": 0.0}, {"time": 4.0}],
        expected_t_final=8.0,
        expected_row_count=321,
        early_stop_reason="progress_observables_stable:window=16",
        stable_early_stop_accepted=True,
    )

    assert stable["full_horizon_gate_passed"] is True
    assert stable["full_horizon_gate_reason"] == "stable_early_stop:progress_observables_stable:window=16"
    assert stable["full_horizon_completion_kind"] == "stable_early_stop"
    assert stable["full_horizon_successful_early_stop"] is True
    assert stable["full_horizon_reached_final_time"] is False
    assert stable["full_horizon_reached_expected_rows"] is False


def test_full_horizon_completion_fields_does_not_infer_stable_stop_from_prefix_alone() -> None:
    fields = full_horizon_completion_fields(
        [{"time": 0.0}, {"time": 4.0}],
        expected_t_final=8.0,
        expected_row_count=321,
        early_stop_reason="progress_observables_stable:window=16",
    )

    assert fields["full_horizon_gate_passed"] is False
    assert fields["full_horizon_gate_reason"] == "early_stop:progress_observables_stable:window=16"
    assert fields["full_horizon_completion_kind"] == "failed"
    assert fields["full_horizon_successful_early_stop"] is False


def test_hash_helpers_are_stable_for_equal_values() -> None:
    theta_a = np.array([0.1, -0.2, 0.3], dtype=float)
    theta_b = np.array([0.1, -0.2, 0.3], dtype=float)
    psi_a = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    psi_b = np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    assert hash_theta_vector(theta_a) == hash_theta_vector(theta_b)
    assert hash_statevector(psi_a) == hash_statevector(psi_b)


def test_geometry_value_key_has_no_tier_identity() -> None:
    key_a = GeometryValueKey(
        checkpoint_id="ckpt",
        observable_family="candidate_insert_tangent_block",
        candidate_label="op_y",
        position_id=1,
        runtime_indices=(2, 3),
        group_key=None,
        grouping_mode="qwc_basis_cover_reuse",
    )
    key_b = GeometryValueKey(
        checkpoint_id="ckpt",
        observable_family="candidate_insert_tangent_block",
        candidate_label="op_y",
        position_id=1,
        runtime_indices=(2, 3),
        group_key=None,
        grouping_mode="qwc_basis_cover_reuse",
    )
    assert key_a == key_b
    assert hash(key_a) == hash(key_b)


def test_oracle_value_key_includes_tier_identity() -> None:
    key_a = OracleValueKey(
        checkpoint_id="ckpt",
        tier_name="confirm",
        observable_family="candidate_step_energy",
        candidate_label="op_y",
        position_id=1,
    )
    key_b = OracleValueKey(
        checkpoint_id="ckpt",
        tier_name="commit",
        observable_family="candidate_step_energy",
        candidate_label="op_y",
        position_id=1,
    )
    assert key_a != key_b
    assert hash(key_a) != hash(key_b)


def test_realtime_checkpoint_types_payloads_include_analytic_noise_fields() -> None:
    cfg_payload = dataclass_to_payload(
        RealtimeCheckpointConfig(
            analytic_noise_std=0.25,
            analytic_noise_seed=13,
            analytic_noise_model="hybrid_qpu_proxy_v1",
            analytic_noise_nominal_shots=4096,
            analytic_noise_nominal_repeats=2,
            analytic_noise_shot_scale=1.25,
            analytic_noise_two_qubit_depth_scale=0.4,
            analytic_noise_groups_new_scale=0.3,
            analytic_noise_time_corr=0.6,
            analytic_noise_bias_energy=0.02,
            analytic_noise_bias_doublon=0.01,
            analytic_noise_bias_staggered=-0.03,
            analytic_noise_metric_scale=1.5,
            analytic_noise_force_psd=False,
            high_miss_no_admit_policy="repair_stop",
            repair_retry_admission_policy="rescue_best_confirmed_append_v1",
            repair_retry_rescue_min_gain_ratio=0.25,
            repair_retry_rescue_attempt="terminal_attempt_only",
            miss_abs_threshold=0.125,
            miss_persistence_window=3,
            miss_persistence_count=2,
            integrator_policy="rk4",
            integrator_columnarity_threshold=0.9,
            integrator_curvature_threshold=0.2,
            integrator_euler_fs_error_threshold=0.004,
            integrator_condition_max=1234.5,
        )
    )
    ledger_payload = dataclass_to_payload(
        CheckpointLedgerEntry(
            checkpoint_index=0,
            time=0.0,
            action_kind="stay",
            candidate_label=None,
            position_id=None,
            rho_miss=0.0,
            rho_real=0.0,
            rho_num=0.0,
            gain_ratio_selected=0.0,
            shortlist_size=0,
            tier_reached="scout",
            logical_block_count_before=1,
            logical_block_count_after=1,
            runtime_parameter_count_before=1,
            runtime_parameter_count_after=1,
            rate_change_l2=None,
            integrator_policy="rk4",
            integrator_used="rk4",
            integrator_columnarity=0.95,
            integrator_curvature=0.05,
            integrator_euler_fs_error=0.002,
            integrator_condition_number=12.0,
            integrator_condition_pass=True,
            integrator_rho_miss_pass=False,
            analytic_noise_std=0.25,
            analytic_noise_seed=13,
            repair_no_admit_diagnostics={"confirmed_candidate_count": 0},
            repair_rescue_candidate_label="op_y",
            repair_rescue_reason="no_confirmed_candidates",
            repair_rescue_admitted=False,
            high_miss_no_admit_soft_fallback=True,
            high_miss_no_admit_soft_fallback_policy="bounded_stay_advance",
            high_miss_no_admit_soft_fallback_reason=HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON,
            high_miss_no_admit_soft_fallback_warning=HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING,
        )
    )
    assert float(cfg_payload["analytic_noise_std"]) == pytest.approx(0.25)
    assert int(cfg_payload["analytic_noise_seed"]) == 13
    assert str(cfg_payload["analytic_noise_model"]) == "hybrid_qpu_proxy_v1"
    assert int(cfg_payload["analytic_noise_nominal_shots"]) == 4096
    assert int(cfg_payload["analytic_noise_nominal_repeats"]) == 2
    assert float(cfg_payload["analytic_noise_shot_scale"]) == pytest.approx(1.25)
    assert float(cfg_payload["analytic_noise_two_qubit_depth_scale"]) == pytest.approx(0.4)
    assert float(cfg_payload["analytic_noise_groups_new_scale"]) == pytest.approx(0.3)
    assert float(cfg_payload["analytic_noise_time_corr"]) == pytest.approx(0.6)
    assert float(cfg_payload["analytic_noise_bias_energy"]) == pytest.approx(0.02)
    assert float(cfg_payload["analytic_noise_bias_doublon"]) == pytest.approx(0.01)
    assert float(cfg_payload["analytic_noise_bias_staggered"]) == pytest.approx(-0.03)
    assert float(cfg_payload["analytic_noise_metric_scale"]) == pytest.approx(1.5)
    assert bool(cfg_payload["analytic_noise_force_psd"]) is False
    assert str(cfg_payload["high_miss_no_admit_policy"]) == "repair_stop"
    assert str(cfg_payload["repair_retry_admission_policy"]) == "rescue_best_confirmed_append_v1"
    assert float(cfg_payload["repair_retry_rescue_min_gain_ratio"]) == pytest.approx(0.25)
    assert str(cfg_payload["repair_retry_rescue_attempt"]) == "terminal_attempt_only"
    assert float(cfg_payload["miss_abs_threshold"]) == pytest.approx(0.125)
    assert int(cfg_payload["miss_persistence_window"]) == 3
    assert int(cfg_payload["miss_persistence_count"]) == 2
    assert str(cfg_payload["integrator_policy"]) == "rk4"
    assert float(cfg_payload["integrator_columnarity_threshold"]) == pytest.approx(0.9)
    assert float(cfg_payload["integrator_curvature_threshold"]) == pytest.approx(0.2)
    assert float(cfg_payload["integrator_euler_fs_error_threshold"]) == pytest.approx(0.004)
    assert float(cfg_payload["integrator_condition_max"]) == pytest.approx(1234.5)
    assert float(ledger_payload["rho_real"]) == pytest.approx(0.0)
    assert float(ledger_payload["rho_num"]) == pytest.approx(0.0)
    assert str(ledger_payload["integrator_policy"]) == "rk4"
    assert str(ledger_payload["integrator_used"]) == "rk4"
    assert ledger_payload["repair_no_admit_diagnostics"] == {"confirmed_candidate_count": 0}
    assert str(ledger_payload["repair_rescue_candidate_label"]) == "op_y"
    assert str(ledger_payload["repair_rescue_reason"]) == "no_confirmed_candidates"
    assert bool(ledger_payload["repair_rescue_admitted"]) is False
    assert bool(ledger_payload["high_miss_no_admit_soft_fallback"]) is True
    assert str(ledger_payload["high_miss_no_admit_soft_fallback_policy"]) == "bounded_stay_advance"
    assert str(ledger_payload["high_miss_no_admit_soft_fallback_reason"]) == HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_REASON
    assert str(ledger_payload["high_miss_no_admit_soft_fallback_warning"]) == HIGH_MISS_NO_ADMIT_SOFT_FALLBACK_WARNING
    assert float(ledger_payload["integrator_columnarity"]) == pytest.approx(0.95)
    assert float(ledger_payload["integrator_curvature"]) == pytest.approx(0.05)
    assert float(ledger_payload["integrator_euler_fs_error"]) == pytest.approx(0.002)
    assert float(ledger_payload["integrator_condition_number"]) == pytest.approx(12.0)
    assert bool(ledger_payload["integrator_condition_pass"]) is True
    assert bool(ledger_payload["integrator_rho_miss_pass"]) is False
    assert float(ledger_payload["analytic_noise_std"]) == pytest.approx(0.25)
    assert int(ledger_payload["analytic_noise_seed"]) == 13
