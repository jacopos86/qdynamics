from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    GeometryValueKey,
    OracleValueKey,
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
