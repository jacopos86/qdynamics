from __future__ import annotations

from pathlib import Path
import sys

import pytest
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_continuation_scoring import MeasurementCacheAudit
from pipelines.exact_bench.noise_oracle_runtime import (
    OracleConfig,
    normalize_mitigation_config,
    normalize_runtime_estimator_profile_config,
)
from pipelines.hardcoded.hh_fixed_manifold_measured import (
    FixedManifoldMeasuredConfig,
    assemble_measured_geometry,
)
from pipelines.hardcoded.hh_fixed_manifold_observables import (
    build_checkpoint_observable_plan_from_layout,
)
from pipelines.hardcoded.hh_realtime_checkpoint_controller import (
    RealtimeCheckpointController,
)
from pipelines.hardcoded.hh_realtime_checkpoint_types import (
    DerivedGeometryKey,
    GeometryValueKey,
    MeasurementTierConfig,
    OracleValueKey,
    RealtimeCheckpointConfig,
    make_checkpoint_context,
)
from pipelines.hardcoded.hh_realtime_measurement import (
    BackendScheduledRawGroupPool,
    controller_oracle_supports_raw_group_sampling,
    DerivedGeometryMemo,
    ExactCheckpointValueCache,
    OracleCheckpointValueCache,
    TemporalMeasurementLedger,
    build_controller_oracle_tier_configs,
    estimate_grouped_raw_mclachlan_incremental_block,
    planning_stats_for_term,
    validate_controller_oracle_base_config,
)
from pipelines.hardcoded.hh_vqe_from_adapt_family import ReplayScaffoldContext
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    PauliPolynomial,
    PauliTerm,
    expval_pauli_polynomial,
)
from types import SimpleNamespace


def _simple_term() -> AnsatzTerm:
    return AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )


def _composite_candidate_context() -> tuple[ReplayScaffoldContext, np.ndarray, np.ndarray, np.ndarray]:
    x_term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    yz_term = AnsatzTerm(
        label="op_yz",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(1, ps="y", pc=0.7),
                PauliTerm(1, ps="z", pc=-0.4),
            ],
        ),
    )
    h_poly = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])
    hmat = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    psi_ref = np.array([1.0, 0.0], dtype=complex)
    base_layout = build_parameter_layout(
        [x_term],
        ignore_identity=True,
        coefficient_tolerance=1.0e-12,
        sort_terms=True,
    )
    executor = CompiledAnsatzExecutor(
        [x_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    best_theta = np.array([0.2], dtype=float)
    psi_initial = executor.prepare_state(best_theta, psi_ref)
    replay_context = ReplayScaffoldContext(
        cfg=SimpleNamespace(reps=1),
        h_poly=h_poly,
        psi_ref=psi_ref,
        payload_in={"adapt_vqe": {"pool_type": "phase3_v1"}},
        family_info={"resolved": "toy_composite_pool"},
        family_pool=(x_term, yz_term),
        pool_meta={"candidate_pool_complete": True},
        replay_terms=(x_term,),
        base_layout=base_layout,
        adapt_theta_runtime=np.array([0.2], dtype=float),
        adapt_theta_logical=np.array([0.2], dtype=float),
        adapt_depth=1,
        handoff_state_kind="prepared_state",
        provenance_source="explicit",
        family_terms_count=2,
    )
    return replay_context, h_poly, hmat, psi_initial


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


def test_normalize_mitigation_config_records_two_qubit_twirling_scope() -> None:
    payload = normalize_mitigation_config(
        {
            "mode": "readout",
            "local_readout_strategy": "mthree",
            "local_gate_twirling": True,
        }
    )

    assert payload["local_gate_twirling"] is True
    assert payload["local_gate_twirling_scope"] == "2q_only"


def test_normalize_runtime_estimator_profile_config_records_two_qubit_twirling_scope() -> None:
    payload = normalize_runtime_estimator_profile_config({"name": "main_twirled_readout_v1"})

    assert payload["gate_twirling"] is True
    assert payload["gate_twirling_scope"] == "2q_only"


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


def test_derived_geometry_memo_reuses_same_key() -> None:
    memo = DerivedGeometryMemo(checkpoint_id="ckpt0")
    key = DerivedGeometryKey(
        checkpoint_id="ckpt0",
        memo_family="baseline_geometry",
        candidate_label=None,
        position_id=None,
    )
    calls = {"count": 0}

    def _compute() -> dict[str, int]:
        calls["count"] += 1
        return {"value": 7}

    value_1, reused_1 = memo.get_or_compute(key, compute=_compute)
    value_2, reused_2 = memo.get_or_compute(key, compute=_compute)

    assert dict(value_1) == {"value": 7}
    assert dict(value_2) == {"value": 7}
    assert bool(reused_1) is False
    assert bool(reused_2) is True
    assert int(calls["count"]) == 1
    assert int(memo.summary()["hits"]) == 1


def test_temporal_measurement_ledger_bonus_is_only_a_low_displacement_prior() -> None:
    ledger = TemporalMeasurementLedger()
    ledger.record_checkpoint(
        checkpoint_index=0,
        selected_candidate_identity="op_x__pool1",
        selected_position_id=1,
        selected_groups_new=2.0,
        selected_gain_ratio=0.15,
        predicted_displacement=0.02,
        refresh_pressure="low",
    )

    low_bonus = ledger.candidate_probe_bonus(
        candidate_identity="op_x__pool1",
        position_id=1,
        predicted_displacement=0.01,
    )
    high_bonus = ledger.candidate_probe_bonus(
        candidate_identity="op_x__pool1",
        position_id=1,
        predicted_displacement=0.20,
    )
    missing_bonus = ledger.candidate_probe_bonus(
        candidate_identity="op_y__pool2",
        position_id=1,
        predicted_displacement=0.01,
    )

    assert float(low_bonus) > 0.0
    assert float(high_bonus) == 0.0
    assert float(missing_bonus) == 0.0
    assert int(ledger.summary()["prior_entries"]) == 1


def test_validate_controller_oracle_base_config_accepts_runtime_raw_sampler_profile() -> None:
    cfg = OracleConfig(
        noise_mode="runtime",
        oracle_aggregate="mean",
        backend_name="ibm_marrakesh",
        runtime_profile={"name": "main_twirled_readout_v1"},
        runtime_raw_profile={"name": "raw_sampler_twirled_v1"},
    )
    validate_controller_oracle_base_config(cfg)


def test_validate_controller_oracle_base_config_rejects_runtime_estimator_profile_as_raw_profile() -> None:
    cfg = OracleConfig(
        noise_mode="runtime",
        oracle_aggregate="mean",
        backend_name="ibm_marrakesh",
        runtime_raw_profile={"name": "main_twirled_readout_v1"},
    )
    with pytest.raises(ValueError, match="runtime_raw_profile"):
        validate_controller_oracle_base_config(cfg)


def test_controller_oracle_supports_raw_group_sampling_runtime_requires_valid_raw_sampler_profile() -> None:
    valid_cfg = OracleConfig(
        noise_mode="runtime",
        oracle_aggregate="mean",
        backend_name="ibm_marrakesh",
        mitigation={"mode": "none"},
        symmetry_mitigation={"mode": "verify_only"},
        runtime_raw_profile={"name": "raw_sampler_dd_probe_v1"},
    )
    invalid_cfg = OracleConfig(
        noise_mode="runtime",
        oracle_aggregate="mean",
        backend_name="ibm_marrakesh",
        mitigation={"mode": "none"},
        symmetry_mitigation={"mode": "verify_only"},
        runtime_raw_profile={"name": "main_twirled_readout_v1"},
    )

    assert controller_oracle_supports_raw_group_sampling(valid_cfg) is True
    assert controller_oracle_supports_raw_group_sampling(invalid_cfg) is False


def test_backend_scheduled_raw_group_pool_reuses_and_extends_same_term_key() -> None:
    class _ObservableStub:
        def to_list(self) -> list[tuple[str, complex]]:
            return [("ZI", 1.0), ("ZZ", 0.5)]

    class _OracleStub:
        def __init__(self) -> None:
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()
            self.calls = 0

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            self.calls += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": 128,
                "counts": {"00": 128},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0, 1],
                "quasi_probs": None,
                "term_details": {
                    "active_logical_qubits": [0, 1],
                    "active_physical_qubits": [0, 1],
                    "pauli_weight": 2,
                    "label": str(pauli_label_ixyz),
                },
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

    pool = BackendScheduledRawGroupPool(checkpoint_id="ckpt0")
    oracle = _OracleStub()
    first = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=_ObservableStub(),
        observable_family="candidate_step_energy",
        candidate_label="op_x",
        position_id=0,
        min_total_shots=128,
        min_samples=1,
    )
    second = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=_ObservableStub(),
        observable_family="candidate_step_energy",
        candidate_label="op_x",
        position_id=0,
        min_total_shots=256,
        min_samples=2,
    )

    assert float(first["mean"]) == pytest.approx(1.5)
    assert float(second["mean"]) == pytest.approx(1.5)
    assert int(oracle.calls) == 2
    assert int(pool.summary()["hits"]) == 1
    assert int(pool.summary()["misses"]) == 1
    assert int(pool.summary()["extensions"]) == 2


def test_backend_scheduled_raw_group_pool_uses_full_tier_shot_budget_for_top_up() -> None:
    class _ObservableStub:
        def to_list(self) -> list[tuple[str, complex]]:
            return [("Z", 1.0)]

    class _OracleStub:
        def __init__(self) -> None:
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()
            self.calls = 0

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            self.calls += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": 100,
                "counts": {"0": 100},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {"active_logical_qubits": [0], "label": str(pauli_label_ixyz)},
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

    pool = BackendScheduledRawGroupPool(checkpoint_id="ckpt0")
    oracle = _OracleStub()
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=_ObservableStub(),
        observable_family="stay_step_energy",
        candidate_label=None,
        position_id=None,
        min_total_shots=100,
        min_samples=1,
    )
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=_ObservableStub(),
        observable_family="stay_step_energy",
        candidate_label=None,
        position_id=None,
        min_total_shots=300,
        min_samples=2,
    )

    assert int(oracle.calls) == 3
    assert int(pool.summary()["total_shots"]) == 300


def test_backend_scheduled_raw_group_pool_without_state_key_keeps_observable_family_isolation() -> None:
    class _ObservableStub:
        def to_list(self) -> list[tuple[str, complex]]:
            return [("Z", 1.0)]

    class _OracleStub:
        def __init__(self) -> None:
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()
            self.calls = 0

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            self.calls += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": 128,
                "counts": {"0": 128},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {"active_logical_qubits": [0], "label": str(pauli_label_ixyz)},
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

    pool = BackendScheduledRawGroupPool(checkpoint_id="ckpt0")
    oracle = _OracleStub()
    observable = _ObservableStub()
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=observable,
        observable_family="energy",
        candidate_label=None,
        position_id=None,
        min_total_shots=128,
        min_samples=1,
    )
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=observable,
        observable_family="generator_mean",
        candidate_label=None,
        position_id=None,
        min_total_shots=128,
        min_samples=1,
    )

    assert int(oracle.calls) == 2
    assert int(pool.summary()["hits"]) == 0
    assert int(pool.summary()["misses"]) == 2


def test_backend_scheduled_raw_group_pool_with_state_key_shares_across_observable_families() -> None:
    class _ObservableStub:
        def to_list(self) -> list[tuple[str, complex]]:
            return [("Z", 1.0)]

    class _OracleStub:
        def __init__(self) -> None:
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()
            self.calls = 0

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            self.calls += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": 128,
                "counts": {"0": 128},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {"active_logical_qubits": [0], "label": str(pauli_label_ixyz)},
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

    pool = BackendScheduledRawGroupPool(checkpoint_id="ckpt0")
    oracle = _OracleStub()
    observable = _ObservableStub()
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=observable,
        observable_family="energy",
        candidate_label=None,
        position_id=None,
        min_total_shots=128,
        min_samples=1,
        state_key="state_a",
    )
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=observable,
        observable_family="generator_mean",
        candidate_label=None,
        position_id=None,
        min_total_shots=128,
        min_samples=1,
        state_key="state_a",
    )
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.3]},
        observable=observable,
        observable_family="generator_mean",
        candidate_label=None,
        position_id=None,
        min_total_shots=128,
        min_samples=1,
        state_key="state_b",
    )

    assert int(oracle.calls) == 2
    assert int(pool.summary()["hits"]) == 1
    assert int(pool.summary()["misses"]) == 2


def test_backend_scheduled_raw_group_pool_with_state_key_shares_across_candidates() -> None:
    class _ObservableStub:
        def to_list(self) -> list[tuple[str, complex]]:
            return [("Z", 1.0)]

    class _OracleStub:
        def __init__(self) -> None:
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()
            self.calls = 0

        def collect_group_sample(self, circuit, pauli_label_ixyz: str, *, repeat_idx: int = 0):
            self.calls += 1
            return {
                "repeat_index": int(repeat_idx),
                "shots": 128,
                "counts": {"0": 128},
                "basis_label": str(pauli_label_ixyz),
                "measured_logical_qubits": [0],
                "quasi_probs": None,
                "term_details": {"active_logical_qubits": [0], "label": str(pauli_label_ixyz)},
                "readout_mitigation": {"mode": "none", "applied": False},
                "local_gate_twirling": {"requested": False, "applied": False},
                "local_dynamical_decoupling": {"requested": False, "applied": False},
            }

    pool = BackendScheduledRawGroupPool(checkpoint_id="ckpt0")
    oracle = _OracleStub()
    observable = _ObservableStub()
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=observable,
        observable_family="candidate_incremental_block:A_1",
        candidate_label="cand_a__pool1",
        position_id=0,
        min_total_shots=128,
        min_samples=1,
        state_key="state_shared",
    )
    _ = pool.estimate_observable(
        oracle=oracle,
        circuit={"theta": [0.2]},
        observable=observable,
        observable_family="candidate_incremental_block:A_2",
        candidate_label="cand_b__pool2",
        position_id=1,
        min_total_shots=128,
        min_samples=1,
        state_key="state_shared",
    )

    assert int(oracle.calls) == 1
    assert int(pool.summary()["hits"]) == 1
    assert int(pool.summary()["misses"]) == 1


def test_incremental_grouped_measurement_matches_exact_block_and_skips_baseline_only_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_context, h_poly, hmat, psi_initial = _composite_candidate_context()
    cfg = RealtimeCheckpointConfig(
        mode="exact_v1",
        miss_threshold=0.0,
        gain_ratio_threshold=1.0e-12,
        append_margin_abs=1.0e-12,
        regularization_lambda=1.0e-8,
        candidate_regularization_lambda=1.0e-8,
    )
    controller = RealtimeCheckpointController(
        cfg=cfg,
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_composite_pool",
        grouping_mode=str(cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline_exact = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    exact_block = controller._compute_candidate_incremental_block(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline_exact,
        candidate_term=replay_context.family_pool[1],
        candidate_pool_index=1,
        position_id=1,
    )
    candidate_data = dict(exact_block["candidate_data"])
    assert len(candidate_data["runtime_block_indices"]) == 2

    baseline_plan = build_checkpoint_observable_plan_from_layout(
        controller.current_layout,
        controller.current_theta,
        psi_ref=replay_context.psi_ref,
        h_poly=h_poly,
        drop_abs_tol=1.0e-12,
        hermiticity_tol=1.0e-10,
        max_observable_terms=128,
    )
    psi_baseline = controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref)
    baseline_energy = float(expval_pauli_polynomial(psi_baseline, baseline_plan.energy.poly))
    baseline_h2 = float(expval_pauli_polynomial(psi_baseline, baseline_plan.variance_h2.poly))
    baseline_generator_means = [
        float(expval_pauli_polynomial(psi_baseline, spec.poly)) for spec in baseline_plan.generator_means
    ]
    baseline_pair_expectations = {
        tuple(pair): (0.0 if spec.is_zero else float(expval_pauli_polynomial(psi_baseline, spec.poly)))
        for pair, spec in baseline_plan.pair_anticommutators.items()
    }
    baseline_force_expectations = [
        0.0 if spec.is_zero else float(expval_pauli_polynomial(psi_baseline, spec.poly))
        for spec in baseline_plan.force_anticommutators
    ]
    baseline_measured = assemble_measured_geometry(
        plan=baseline_plan,
        energy=baseline_energy,
        h2=baseline_h2,
        generator_means=baseline_generator_means,
        pair_expectations=baseline_pair_expectations,
        force_expectations=baseline_force_expectations,
        geom_cfg=FixedManifoldMeasuredConfig(
            regularization_lambda=float(cfg.regularization_lambda),
            pinv_rcond=float(cfg.pinv_rcond),
        ),
    )

    psi_aug = candidate_data["aug_executor"].prepare_state(
        candidate_data["theta_aug"],
        replay_context.psi_ref,
    )
    observed_names: list[str] = []

    def _exact_observable_spec_mean(
        *,
        raw_group_pool,
        oracle,
        circuit,
        spec,
        observable_family,
        candidate_label,
        position_id,
        state_key,
        min_total_shots,
        min_samples,
    ):
        observed_names.append(str(spec.name))
        mean = 0.0 if spec.is_zero else float(expval_pauli_polynomial(psi_aug, spec.poly))
        return {
            "mean": float(mean),
            "stderr": 0.0,
            "std": 0.0,
            "stdev": 0.0,
            "n_samples": 1,
            "aggregate": "mean",
            "backend_info": {
                "noise_mode": "ideal_stub",
                "estimator_kind": "exact_spec_stub",
                "backend_name": "stub",
                "using_fake_backend": False,
                "details": {},
            },
            "term_payloads": [],
        }

    monkeypatch.setattr(
        "pipelines.hardcoded.hh_realtime_measurement._observable_spec_mean",
        _exact_observable_spec_mean,
    )

    class _PoolStub:
        def summary(self) -> dict[str, int]:
            return {"entries": 0, "hits": 0, "misses": 0, "stores": 0, "extensions": 0}

    oracle = SimpleNamespace(
        backend_info=SimpleNamespace(
            noise_mode="ideal_stub",
            estimator_kind="exact_spec_stub",
            backend_name="stub",
            using_fake_backend=False,
            details={},
        )
    )
    measured = estimate_grouped_raw_mclachlan_incremental_block(
        oracle=oracle,
        raw_group_pool=_PoolStub(),
        baseline_measured=baseline_measured,
        layout=candidate_data["aug_layout"],
        theta_runtime=np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1),
        psi_ref=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
        h_poly=h_poly,
        candidate_runtime_indices=tuple(candidate_data["runtime_block_indices"]),
        runtime_insert_position=int(candidate_data["runtime_insert_position"]),
        geom_cfg=FixedManifoldMeasuredConfig(
            regularization_lambda=float(cfg.regularization_lambda),
            pinv_rcond=float(cfg.pinv_rcond),
        ),
        candidate_regularization_lambda=float(cfg.candidate_regularization_lambda),
        pinv_rcond=float(cfg.pinv_rcond),
        observable_family_prefix="candidate_incremental_block",
        candidate_label="op_yz__pool1",
        position_id=1,
        state_key="state_exact",
        min_total_shots=64,
        min_samples=1,
    )
    block = measured["incremental_block"]

    assert "energy" not in observed_names
    assert "h2" not in observed_names
    assert "A_0" not in observed_names
    assert "AHsym_0" not in observed_names
    assert set(observed_names) == {"A_1", "A_2", "AHsym_1", "AHsym_2", "AAsym_0_1", "AAsym_0_2", "AAsym_1_2"}
    assert np.allclose(np.asarray(block["B"], dtype=float), np.asarray(exact_block["B"], dtype=float), atol=1.0e-10)
    assert np.allclose(np.asarray(block["C"], dtype=float), np.asarray(exact_block["C"], dtype=float), atol=1.0e-10)
    assert np.allclose(np.asarray(block["q"], dtype=float), np.asarray(exact_block["q"], dtype=float), atol=1.0e-10)
    assert np.allclose(
        np.asarray(block["theta_dot_aug_existing"], dtype=float),
        np.asarray(exact_block["theta_dot_aug_existing"], dtype=float),
        atol=1.0e-10,
    )
    assert np.allclose(
        np.asarray(block["theta_dot_step"], dtype=float),
        np.asarray(exact_block["theta_dot_aug"], dtype=float),
        atol=1.0e-10,
    )
    assert float(block["gain_exact"]) == pytest.approx(float(exact_block["gain_exact"]), abs=1.0e-10)
    assert float(block["gain_ratio"]) == pytest.approx(float(exact_block["gain_ratio"]), abs=1.0e-8)


def test_incremental_grouped_measurement_validates_contiguous_zero_insert_block() -> None:
    replay_context, h_poly, hmat, psi_initial = _composite_candidate_context()
    controller = RealtimeCheckpointController(
        cfg=RealtimeCheckpointConfig(mode="exact_v1"),
        replay_context=replay_context,
        h_poly=h_poly,
        hmat=hmat,
        psi_initial=psi_initial,
        best_theta=[0.2],
        allow_repeats=False,
        t_final=0.1,
        num_times=2,
    )
    checkpoint_ctx = make_checkpoint_context(
        checkpoint_index=0,
        time_start=0.0,
        time_stop=0.1,
        scaffold_labels=[carrier.label for carrier in controller.current_terms],
        theta=controller.current_theta,
        psi=controller.current_executor.prepare_state(controller.current_theta, replay_context.psi_ref),
        logical_count=int(controller.current_layout.logical_parameter_count),
        runtime_count=int(controller.current_layout.runtime_parameter_count),
        resolved_family="toy_composite_pool",
        grouping_mode=str(controller.cfg.grouping_mode),
        structure_locked=False,
    )
    cache = ExactCheckpointValueCache(
        checkpoint_id=str(checkpoint_ctx.checkpoint_id),
        grouping_mode=str(controller.cfg.grouping_mode),
    )
    geometry_memo = DerivedGeometryMemo(checkpoint_id=str(checkpoint_ctx.checkpoint_id))
    baseline_exact = controller._baseline_geometry(checkpoint_ctx, cache, geometry_memo)
    exact_block = controller._compute_candidate_incremental_block(
        checkpoint_ctx=checkpoint_ctx,
        cache=cache,
        geometry_memo=geometry_memo,
        baseline=baseline_exact,
        candidate_term=replay_context.family_pool[1],
        candidate_pool_index=1,
        position_id=1,
    )
    candidate_data = dict(exact_block["candidate_data"])
    oracle = SimpleNamespace(
        backend_info=SimpleNamespace(
            noise_mode="ideal_stub",
            estimator_kind="exact_spec_stub",
            backend_name="stub",
            using_fake_backend=False,
            details={},
        )
    )

    class _PoolStub:
        def summary(self) -> dict[str, int]:
            return {}

    with pytest.raises(ValueError, match="contiguous inserted runtime block"):
        estimate_grouped_raw_mclachlan_incremental_block(
            oracle=oracle,
            raw_group_pool=_PoolStub(),
            baseline_measured={},
            layout=candidate_data["aug_layout"],
            theta_runtime=np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1),
            psi_ref=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
            h_poly=h_poly,
            candidate_runtime_indices=(0, 2),
            runtime_insert_position=int(candidate_data["runtime_insert_position"]),
            geom_cfg=FixedManifoldMeasuredConfig(),
            candidate_regularization_lambda=1.0e-8,
            pinv_rcond=1.0e-10,
            observable_family_prefix="candidate_incremental_block",
            candidate_label="op_yz__pool1",
            position_id=1,
            state_key="state_exact",
            min_total_shots=64,
            min_samples=1,
        )

    theta_nonzero = np.asarray(candidate_data["theta_aug"], dtype=float).reshape(-1)
    theta_nonzero[int(candidate_data["runtime_block_indices"][0])] = 0.1
    with pytest.raises(ValueError, match="zero inserted candidate angles"):
        estimate_grouped_raw_mclachlan_incremental_block(
            oracle=oracle,
            raw_group_pool=_PoolStub(),
            baseline_measured={},
            layout=candidate_data["aug_layout"],
            theta_runtime=theta_nonzero,
            psi_ref=np.asarray(replay_context.psi_ref, dtype=complex).reshape(-1),
            h_poly=h_poly,
            candidate_runtime_indices=tuple(candidate_data["runtime_block_indices"]),
            runtime_insert_position=int(candidate_data["runtime_insert_position"]),
            geom_cfg=FixedManifoldMeasuredConfig(),
            candidate_regularization_lambda=1.0e-8,
            pinv_rcond=1.0e-10,
            observable_family_prefix="candidate_incremental_block",
            candidate_label="op_yz__pool1",
            position_id=1,
            state_key="state_exact",
            min_total_shots=64,
            min_samples=1,
        )


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


def test_controller_oracle_supports_raw_group_sampling_runtime_only_for_none_mitigation() -> None:
    assert controller_oracle_supports_raw_group_sampling(
        OracleConfig(
            noise_mode="runtime",
            oracle_aggregate="mean",
            backend_name="ibm_test",
            mitigation="none",
        )
    )
    assert not controller_oracle_supports_raw_group_sampling(
        OracleConfig(
            noise_mode="runtime",
            oracle_aggregate="mean",
            backend_name="ibm_test",
            mitigation={"mode": "readout", "local_readout_strategy": "mthree"},
        )
    )


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
