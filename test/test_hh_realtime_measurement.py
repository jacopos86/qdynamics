from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.exact_bench.noise_oracle_runtime import OracleConfig
from pipelines.hardcoded.hh_realtime_checkpoint_types import GeometryValueKey, MeasurementTierConfig, OracleValueKey
from pipelines.hardcoded.hh_realtime_measurement import (
    ExactCheckpointValueCache,
    OracleCheckpointValueCache,
    build_controller_oracle_tier_configs,
    planning_stats_for_term,
    validate_controller_oracle_base_config,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


def _simple_term() -> AnsatzTerm:
    return AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )


def test_exact_checkpoint_value_cache_reuses_same_key_and_extends_tier() -> None:
    cache = ExactCheckpointValueCache(checkpoint_id="ckpt0", grouping_mode="qwc_basis_cover_reuse")
    key = GeometryValueKey(
        checkpoint_id="ckpt0",
        observable_family="baseline_h_apply",
        candidate_label=None,
        position_id=None,
        runtime_indices=tuple(),
        group_key=None,
        grouping_mode="qwc_basis_cover_reuse",
    )
    calls = {"count": 0}

    def _compute() -> tuple[int, str]:
        calls["count"] += 1
        return (7, "value")

    value_1, reused_1 = cache.get_or_compute(key, tier_name="scout", compute=_compute)
    value_2, reused_2 = cache.get_or_compute(key, tier_name="confirm", compute=_compute)

    assert value_1 == (7, "value")
    assert value_2 == (7, "value")
    assert bool(reused_1) is False
    assert bool(reused_2) is True
    assert int(calls["count"]) == 1
    assert int(cache.summary()["extensions"]) == 1


def test_exact_checkpoint_value_cache_rejects_checkpoint_mismatch() -> None:
    cache = ExactCheckpointValueCache(checkpoint_id="ckpt0", grouping_mode="qwc_basis_cover_reuse")
    key = GeometryValueKey(
        checkpoint_id="ckpt1",
        observable_family="baseline_h_apply",
        candidate_label=None,
        position_id=None,
        runtime_indices=tuple(),
        group_key=None,
        grouping_mode="qwc_basis_cover_reuse",
    )
    with pytest.raises(ValueError, match="checkpoint mismatch"):
        cache.get_or_compute(key, tier_name="scout", compute=lambda: 1)


def test_planning_audit_stays_separate_from_exact_cache() -> None:
    term = _simple_term()
    audit = MeasurementCacheAudit()
    stats = planning_stats_for_term(term, audit)
    cache = ExactCheckpointValueCache(checkpoint_id="ckpt0", grouping_mode="qwc_basis_cover_reuse")
    key = GeometryValueKey(
        checkpoint_id="ckpt0",
        observable_family="candidate_insert_tangent_block",
        candidate_label="op_x",
        position_id=0,
        runtime_indices=(0,),
        group_key=None,
        grouping_mode="qwc_basis_cover_reuse",
    )
    cache.get_or_compute(key, tier_name="scout", compute=lambda: 123)

    assert int(stats.groups_new) >= 1
    assert int(cache.summary()["entries"]) == 1
    assert int(cache.summary()["hits"]) == 0


def test_oracle_checkpoint_value_cache_is_tier_aware() -> None:
    cache = OracleCheckpointValueCache(checkpoint_id="ckpt0")
    key_confirm = OracleValueKey(
        checkpoint_id="ckpt0",
        tier_name="confirm",
        observable_family="candidate_step_energy",
        candidate_label="op_x",
        position_id=0,
    )
    key_commit = OracleValueKey(
        checkpoint_id="ckpt0",
        tier_name="commit",
        observable_family="candidate_step_energy",
        candidate_label="op_x",
        position_id=0,
    )
    calls = {"count": 0}

    def _compute() -> int:
        calls["count"] += 1
        return 7

    value_confirm, reused_confirm = cache.get_or_compute(key_confirm, compute=_compute)
    value_commit, reused_commit = cache.get_or_compute(key_commit, compute=_compute)

    assert int(value_confirm) == 7
    assert int(value_commit) == 7
    assert bool(reused_confirm) is False
    assert bool(reused_commit) is False
    assert int(calls["count"]) == 2


def test_build_controller_oracle_tier_configs_clones_mean_only_defaults() -> None:
    base = OracleConfig(
        noise_mode="shots",
        shots=2000,
        oracle_repeats=3,
        oracle_aggregate="mean",
    )
    tiers = (
        MeasurementTierConfig(tier_name="scout", exact_mode_behavior="proxy_only"),
        MeasurementTierConfig(tier_name="confirm", exact_mode_behavior="incremental_exact"),
        MeasurementTierConfig(tier_name="commit", exact_mode_behavior="commit_exact"),
    )
    configs = build_controller_oracle_tier_configs(base, tiers)

    assert int(configs["scout"].shots) == 500
    assert int(configs["scout"].oracle_repeats) == 1
    assert int(configs["confirm"].shots) == 1024
    assert int(configs["confirm"].oracle_repeats) == 3
    assert int(configs["commit"].shots) == 2000
    assert int(configs["commit"].oracle_repeats) == 3
    assert str(configs["commit"].oracle_aggregate) == "mean"


def test_validate_controller_oracle_base_config_accepts_runtime() -> None:
    validate_controller_oracle_base_config(
        OracleConfig(
            noise_mode="runtime",
            oracle_aggregate="mean",
            backend_name="ibm_marrakesh",
        )
    )


def test_validate_controller_oracle_base_config_rejects_backend_scheduled_readout_with_active_symmetry() -> None:
    with pytest.raises(ValueError, match="active symmetry mitigation"):
        validate_controller_oracle_base_config(
            OracleConfig(
                noise_mode="backend_scheduled",
                oracle_aggregate="mean",
                use_fake_backend=True,
                mitigation={"mode": "readout", "local_readout_strategy": "mthree"},
                symmetry_mitigation={"mode": "postselect_diag_v1", "num_sites": 2, "sector_n_up": 1, "sector_n_dn": 1},
            )
        )
