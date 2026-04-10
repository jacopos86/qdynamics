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
    CheckpointLedgerEntry,
    GeometryValueKey,
    OracleValueKey,
    RealtimeCheckpointConfig,
    dataclass_to_payload,
    hash_statevector,
    hash_theta_vector,
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
            gain_ratio_selected=0.0,
            shortlist_size=0,
            tier_reached="scout",
            logical_block_count_before=1,
            logical_block_count_after=1,
            runtime_parameter_count_before=1,
            runtime_parameter_count_after=1,
            rate_change_l2=None,
            analytic_noise_std=0.25,
            analytic_noise_seed=13,
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
    assert float(ledger_payload["analytic_noise_std"]) == pytest.approx(0.25)
    assert int(ledger_payload["analytic_noise_seed"]) == 13
