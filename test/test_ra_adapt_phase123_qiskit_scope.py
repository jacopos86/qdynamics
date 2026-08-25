from __future__ import annotations

from types import SimpleNamespace

import pytest

import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.scaffold.hh_continuation_scoring import (
    FullScoreConfig,
    HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1,
    SimpleScoreConfig,
    rescore_hardware_cost_family,
)
from pipelines.scaffold.hh_continuation_types import CandidateFeatures
from pipelines.static_adapt.hh_backend_compile_oracle import (
    BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
    BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1,
    backend_compile_scope_uses_qiskit_for_stage,
)


def _compiled_feature(
    *,
    label: str,
    pool_index: int,
    raw_delta: float,
) -> CandidateFeatures:
    clipped = max(0.0, float(raw_delta))
    generator_id = f"generator::{label}"
    base_structure_key = "1" * 64
    trial_structure_key = f"{pool_index + 2:x}" * 64
    compile_cache_identity = {
        "schema": "phase123_qiskit_candidate_position_compile_cache_v1",
        "scope": BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        "candidate_label": label,
        "generator_id": f"{generator_id}::pool[{pool_index}]",
        "position_id": int(pool_index),
        "base_structure_key": base_structure_key,
        "trial_structure_key": trial_structure_key,
    }
    return CandidateFeatures(
        stage_name="fixture",
        candidate_label=label,
        candidate_family="fixture",
        candidate_pool_index=pool_index,
        position_id=pool_index,
        append_position=2,
        positions_considered=[pool_index],
        g_signed=1.0,
        g_abs=1.0,
        g_lcb=1.0,
        sigma_hat=0.0,
        F=1.0,
        novelty=1.0,
        curvature_mode="fixture",
        novelty_mode="fixture",
        refit_window_indices=[],
        compiled_position_cost_proxy={},
        measurement_cache_stats={},
        leakage_penalty=0.0,
        stage_gate_open=True,
        leakage_gate_open=True,
        trough_probe_triggered=False,
        trough_detected=False,
        simple_score=1.0,
        score_version="fixture_v1",
        c_hat_2q=clipped,
        c_hat_d=clipped,
        c_hat_1q=clipped,
        c_hat_theta=0.0,
        c_hat_shot=0.0,
        hardware_cost_source="backend_transpile_v1",
        compile_cost_source="backend_transpile_v1",
        compile_gate_open=True,
        generator_id=generator_id,
        phase2_raw_trust_gain=1.0,
        phase3_reduced_trust_gain=1.0,
        phase3_full_joint_trust_gain=1.0,
        phase3_incremental_candidate_gain=1.0,
        compiled_position_cost_backend={
            "selected_backend_name": "FakeMarrakesh",
            "selected_resolution_kind": "fake_exact",
            "raw_delta_compiled_count_2q": float(raw_delta),
            "raw_delta_compiled_depth_2q": float(raw_delta),
            "raw_delta_compiled_count_1q": float(raw_delta),
            "negative_delta_reward_enabled": True,
            "base_structure_key": base_structure_key,
            "trial_structure_key": trial_structure_key,
            "compile_cache_identity": compile_cache_identity,
            "compile_cache_identity_sha256": (
                adapt_pipeline._candidate_record_payload_digest(
                    compile_cache_identity
                )
            ),
            "base_initial_layout": None,
            "trial_initial_layout": None,
            "base_logical_to_physical": [0, 1],
            "trial_logical_to_physical": [0, 1],
            "base_trial_layout_coupling_policy": (
                "independent_unconstrained_full_transpiles_v1"
            ),
            "position_id": int(pool_index),
            "candidate_label": label,
        },
    )


def _normalized_population() -> list[dict[str, object]]:
    return rescore_hardware_cost_family(
        [
            {"feature": _compiled_feature(label="cancel", pool_index=0, raw_delta=-4.0)},
            {"feature": _compiled_feature(label="add", pool_index=1, raw_delta=4.0)},
        ],
        FullScoreConfig(
            hardware_cost_normalization_mode=(
                HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
            ),
            lambda_theta=0.0,
            lambda_shot=0.0,
        ),
    )


def _winner_label_at_production_rank_seam(
    records: list[dict[str, object]],
    *,
    phase: str,
) -> str:
    if phase == "phase_i":
        payload = adapt_pipeline._phase1_eval_payload_from_records(
            records,
            append_position_value=2,
        )
        best_feature = payload["best_feat"]
        assert isinstance(best_feature, dict)
        return str(best_feature["candidate_label"])
    if phase == "phase_ii":
        selected = min(records, key=adapt_pipeline._phase2_record_sort_key)
    elif phase == "phase_iii":
        selected = min(
            records,
            key=adapt_pipeline._default_no_prune_phase3_record_sort_key,
        )
    else:  # pragma: no cover - the parametrization below is exhaustive.
        raise AssertionError(f"Unknown fixture phase: {phase!r}")
    feature = selected.get("feature")
    assert isinstance(feature, CandidateFeatures)
    return str(feature.candidate_label)


def test_phase123_scope_literal_and_stage_routing_are_exact() -> None:
    assert BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1 == (
        "phase0_proxy_or_off_phase_i_phase_ii_phase_iii_qiskit_transpile_v1"
    )
    assert not backend_compile_scope_uses_qiskit_for_stage(
        BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        "phase0",
    )
    for stage in ("phase1", "phase2", "phase3", "full"):
        assert backend_compile_scope_uses_qiskit_for_stage(
            BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
            stage,
        )

    graph_oracle = object()
    qiskit_oracle = object()
    context = SimpleNamespace(
        backend_compile_scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        backend_compile_oracle=graph_oracle,
        phase3_backend_compile_oracle=qiskit_oracle,
    )
    pending = SimpleNamespace(
        backend_compile_snapshot="graph-snapshot",
        phase3_backend_compile_snapshot="qiskit-snapshot",
    )
    assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
        context=context,
        pending=pending,
        evaluation_stage="phase0",
    ) == (graph_oracle, "graph-snapshot", False)
    for stage in ("phase1", "phase2", "phase3"):
        assert adapt_pipeline._default_no_prune_compile_oracle_for_stage(
            context=context,
            pending=pending,
            evaluation_stage=stage,
        ) == (qiskit_oracle, "qiskit-snapshot", True)


def test_phase123_scope_builds_signed_full_trial_qiskit_config() -> None:
    config = adapt_pipeline._default_no_prune_staged_qiskit_compile_config(
        scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        weight_2q=1.0,
        weight_depth=0.1,
        weight_size=0.01,
    )

    assert config.mode == "transpile_single_v1"
    assert config.requested_backend_name == "FakeMarrakesh"
    assert config.optimization_level == 1
    assert config.seed_transpiler == 7
    assert config.reward_negative_deltas is True
    assert config.allow_preferred_fallback is False
    assert config.one_qubit_coordinate_policy == "compiled_positive_delta_v1"


def test_phase1_transaction_records_qiskit_population_on_transaction_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the live session-to-transaction Phase-I ownership seam."""

    normalized = _normalized_population()
    pre_snapshot = object()
    context = SimpleNamespace(
        backend_compile_scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        candidate_adapter=object(),
        max_depth=1,
    )
    cursor = SimpleNamespace(
        phase1_stage=SimpleNamespace(
            pre_step_snapshot=lambda **_kwargs: pre_snapshot,
        ),
    )
    pending = SimpleNamespace(
        append_position=0,
        controller_pre_snapshot_dict={},
        depth=0,
        insertion_mode="append_only",
        qiskit_population_normalization_receipts={},
    )
    session_type = adapt_pipeline._DefaultNoPruneNumericalSession
    session = session_type(
        context=context,
        cursor=cursor,
        estimator_service=object(),
        candidate_sector_auditor=object(),
        initial_accepted_state=object(),
    )
    monkeypatch.setattr(
        adapt_pipeline,
        "_controller_snapshot_dict",
        lambda _snapshot: {},
    )
    monkeypatch.setattr(
        session_type,
        "_evaluate_default_phase1_positions",
        lambda _self, _pending: {
            "append_best_score": 1.0,
            "best_feat": object(),
            "best_idx": 0,
            "best_position": 0,
            "records": normalized,
        },
    )
    monkeypatch.setattr(
        session_type,
        "_default_cheap_selection_mode",
        lambda _self, _feature, *, probe: (
            "test_probe" if probe else "test_append"
        ),
    )
    monkeypatch.setattr(
        session_type,
        "_finalize_default_phase1_score_surface",
        lambda _self, _pending, **_kwargs: {
            "controller_snapshot": pre_snapshot,
            "phase1_records": normalized,
            "phase1_shortlisted_records": normalized,
            "adaptive_shortlist_receipt": None,
        },
    )

    result = adapt_pipeline._DefaultNoPruneSelectionTransaction(
        session=session,
        pending=pending,
    ).run_phase_i()

    receipt = pending.qiskit_population_normalization_receipts["phase_i"]
    assert receipt["population_count"] == len(normalized)
    assert result["phase1_records"] == normalized


@pytest.mark.parametrize("phase", ["phase_i", "phase_ii", "phase_iii"])
def test_phase123_population_receipt_closes_signed_normalization(
    phase: str,
) -> None:
    normalized = _normalized_population()
    receipt = adapt_pipeline._default_no_prune_phase123_qiskit_population_receipt(
        normalized,
        phase=phase,
        scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
    )

    assert receipt["phase"] == phase
    assert receipt["population_count"] == 2
    assert receipt["normalization_count"] == 1
    assert receipt["normalization_policy"] == (
        HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
    )
    assert receipt["negative_delta_reward_enabled"] is True
    factors = {
        row["candidate_label"]: row["hardware_cost_score_factor"]
        for row in receipt["rows"]
    }
    assert factors["cancel"] > 1.0
    assert factors["add"] < 1.0
    for row in receipt["rows"]:
        assert "::pool[" not in row["generator_id"]
        assert row["compile_cache_generator_id"] == (
            f"{row['generator_id']}::pool[{row['candidate_pool_index']}]"
        )
        assert row["compile_cache_identity"]["generator_id"] == row[
            "compile_cache_generator_id"
        ]
    assert receipt["excluded_from_s_alg"] is True


@pytest.mark.parametrize(
    ("phase", "score_key"),
    [
        ("phase_i", "phase1_active_score"),
        ("phase_ii", "phase2_raw_score"),
        ("phase_iii", "phase3_primary_score"),
    ],
)
def test_signed_qiskit_cost_can_reverse_raw_benefit_order_in_every_phase(
    phase: str,
    score_key: str,
) -> None:
    cancel = adapt_pipeline.candidate_feature_with_updates(
        _compiled_feature(label="cancel", pool_index=0, raw_delta=-4.0),
        {
            "g_signed": 0.9,
            "g_abs": 0.9,
            "g_lcb": 0.9,
            "phase2_raw_trust_gain": 0.9,
            "phase3_reduced_trust_gain": 0.9,
            "phase3_full_joint_trust_gain": 0.9,
            "phase3_incremental_candidate_gain": 0.9,
        },
    )
    add = adapt_pipeline.candidate_feature_with_updates(
        _compiled_feature(label="add", pool_index=1, raw_delta=4.0),
        {
            "g_signed": 1.0,
            "g_abs": 1.0,
            "g_lcb": 1.0,
            "phase2_raw_trust_gain": 1.0,
            "phase3_reduced_trust_gain": 1.0,
            "phase3_full_joint_trust_gain": 1.0,
            "phase3_incremental_candidate_gain": 1.0,
        },
    )
    common = {
        "hardware_cost_normalization_mode": (
            HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
        ),
        "lambda_theta": 0.0,
        "lambda_shot": 0.0,
    }
    cfg = (
        SimpleScoreConfig(**common)
        if phase == "phase_i"
        else FullScoreConfig(**common)
    )
    raw_records = [
        {
            "feature": cancel,
            "phase1_active_score": 0.9,
            "phase2_raw_score": 0.9,
            "full_v2_score": 0.9,
            "phase3_primary_score": 0.9,
            "phase3_tie_break_score": 0.9,
            "simple_score": 0.9,
            "candidate_pool_index": 0,
            "position_id": 0,
        },
        {
            "feature": add,
            "phase1_active_score": 1.0,
            "phase2_raw_score": 1.0,
            "full_v2_score": 1.0,
            "phase3_primary_score": 1.0,
            "phase3_tie_break_score": 1.0,
            "simple_score": 1.0,
            "candidate_pool_index": 1,
            "position_id": 1,
        },
    ]
    assert _winner_label_at_production_rank_seam(
        raw_records,
        phase=phase,
    ) == "add"
    rescored = rescore_hardware_cost_family(
        raw_records,
        cfg,
    )

    # The additive candidate has the larger raw benefit (1.0 > 0.9), but the
    # compiled cancellation is a reward and therefore wins after cost shaping.
    assert float(rescored[0][score_key]) > float(rescored[1][score_key])
    assert _winner_label_at_production_rank_seam(
        rescored,
        phase=phase,
    ) == "cancel"
    receipt = adapt_pipeline._default_no_prune_phase123_qiskit_population_receipt(
        rescored,
        phase=phase,
        scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
    )
    assert receipt["population_count"] == 2


@pytest.mark.parametrize("mutation", ["missing_hash", "mixed_hash", "unsigned"])
def test_phase123_population_receipt_rejects_missing_or_mixed_telemetry(
    mutation: str,
) -> None:
    normalized = _normalized_population()
    first = normalized[0]["feature"]
    second = normalized[1]["feature"]
    assert isinstance(first, CandidateFeatures)
    assert isinstance(second, CandidateFeatures)
    if mutation == "missing_hash":
        normalized[0]["feature"] = adapt_pipeline.candidate_feature_with_updates(
            first,
            {"hardware_cost_population_hash": None},
        )
    elif mutation == "mixed_hash":
        normalized[1]["feature"] = adapt_pipeline.candidate_feature_with_updates(
            second,
            {"hardware_cost_population_hash": "f" * 64},
        )
    else:
        backend = dict(first.compiled_position_cost_backend or {})
        backend["negative_delta_reward_enabled"] = False
        normalized[0]["feature"] = adapt_pipeline.candidate_feature_with_updates(
            first,
            {"compiled_position_cost_backend": backend},
        )

    with pytest.raises(RuntimeError, match="population|signed|normalization"):
        adapt_pipeline._default_no_prune_phase123_qiskit_population_receipt(
            normalized,
            phase="phase_ii",
            scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        )


def test_phase123_accounting_lists_qiskit_phases_and_excludes_s_alg() -> None:
    qiskit = SimpleNamespace(
        config=SimpleNamespace(
            mode="transpile_single_v1",
            requested_backend_name="FakeMarrakesh",
            optimization_level=1,
            seed_transpiler=7,
            structure_theta_value=1.0,
            reward_negative_deltas=True,
            one_qubit_coordinate_policy="compiled_positive_delta_v1",
            allow_preferred_fallback=False,
        ),
        targets=(),
        resolution_audit=(),
        cache_summary=lambda: {"estimate_count": 6},
    )
    context = SimpleNamespace(
        backend_compile_scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        backend_compile_oracle=object(),
        phase3_backend_compile_oracle=qiskit,
    )
    accounting = adapt_pipeline._default_no_prune_selector_compile_cost_accounting(
        context=context,
    )

    assert accounting["qiskit_applied_phases"] == [
        "phase_i",
        "phase_ii",
        "phase_iii",
    ]
    assert accounting["phase_i_phase_ii"] is None
    assert accounting["phase_iii"]["role"] == "phase_i_phase_ii_phase_iii"
    assert accounting["excluded_from_s_alg"] is True
    assert "S_alg" not in accounting


def _projected_phase3_qiskit_receipt(scope: str) -> dict[str, object]:
    oracle = SimpleNamespace(estimate_count=0)
    features = [
        _compiled_feature(label="cancel", pool_index=0, raw_delta=-4.0),
        _compiled_feature(label="add", pool_index=1, raw_delta=4.0),
    ]
    retained = [
        {
            "feature": feature,
            "candidate_label": feature.candidate_label,
            "candidate_pool_index": feature.candidate_pool_index,
            "position_id": feature.position_id,
        }
        for feature in features
    ]
    factories = {}
    for row, feature in zip(retained, features, strict=True):
        key = adapt_pipeline._batch_admission_record_key(row)

        def _factory(feature=feature, row=row):
            oracle.estimate_count += 1
            return ([{**row, "feature": feature}], {})

        factories[key] = _factory

    _records, projected, _shortlisted = (
        adapt_pipeline._default_no_prune_projected_phase3_population(
            phase2_shortlisted_records=retained,
            archival_phase3_factory_by_parent_key=factories,
            archival_phase2_parent_expansions=[],
            phase2_full_records_evaluated=retained,
            controller_snapshot=None,
            phase3_live=False,
            pool_generator_registry={},
            phase2_score_cfg_round=FullScoreConfig(
                hardware_cost_normalization_mode=(
                    HARDWARE_COST_NORMALIZATION_ZERO_CENTERED_SIGNED_ARCTAN_V1
                ),
                lambda_theta=0.0,
                lambda_shot=0.0,
            ),
            phase3_runtime_split_summary={},
            phase_shortlist_runtime=None,
            phase3_shortlist_size=None,
            phase3_backend_compile_oracle=oracle,
            backend_compile_scope=scope,
        )
    )
    receipt = projected.get("phase3_qiskit_selector_cost_receipt")
    assert isinstance(receipt, dict)
    return receipt


def test_phase123_projected_phase3_emits_signed_population_receipt() -> None:
    receipt = _projected_phase3_qiskit_receipt(
        BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1
    )

    assert receipt["scope"] == BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1
    assert receipt["negative_delta_reward_enabled"] is True
    assert receipt["selector_circuit_coordinates"] == [
        "raw_signed_delta_N2q",
        "raw_signed_delta_D2q",
        "raw_signed_delta_N1q",
    ]
    normalized = receipt["phase123_population_normalization_receipt"]
    assert normalized["phase"] == "phase_iii"
    assert normalized["population_count"] == 2
    assert normalized["normalization_count"] == 1


def test_phase23_projected_phase3_closes_signed_qiskit_population_without_phase123_label() -> None:
    receipt = _projected_phase3_qiskit_receipt(
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
    )

    assert receipt["scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
    )
    assert receipt["phase_cost_sources"] == {
        "phase_i": "structural_proxy_v1",
        "phase_ii": "backend_transpile_v1",
        "phase_iii": "backend_transpile_v1",
    }
    assert receipt["negative_delta_reward_enabled"] is True
    assert receipt["selector_circuit_coordinates"] == [
        "raw_signed_delta_N2q",
        "raw_signed_delta_D2q",
        "raw_signed_delta_N1q",
    ]
    assert "phase123_population_normalization_receipt" not in receipt
    normalized = receipt["phase23_population_normalization_receipt"]
    assert normalized["schema"] == (
        "paper_i_phase23_qiskit_population_normalization_v1"
    )
    assert normalized["scope"] == (
        BACKEND_COMPILE_SCOPE_PHASE2_PHASE3_QISKIT_ONLY_V1
    )
    assert normalized["phase"] == "phase_iii"
    assert normalized["population_count"] == 2
    assert normalized["normalization_count"] == 1
    assert all(
        "compile_cache_generator_id" not in row
        for row in normalized["rows"]
    )


def test_phase123_qiskit_base_snapshot_is_lazy_until_phase_i() -> None:
    class _QiskitOracle:
        def __init__(self) -> None:
            self.snapshot_calls: list[tuple[object, ...]] = []

        def snapshot_base(self, ops: tuple[object, ...]) -> object:
            self.snapshot_calls.append(tuple(ops))
            return {"base": tuple(ops)}

    qiskit = _QiskitOracle()
    proxy = object()
    proxy_snapshot = object()
    context = SimpleNamespace(
        backend_compile_scope=BACKEND_COMPILE_SCOPE_PHASE123_QISKIT_V1,
        phase3_backend_compile_oracle=qiskit,
        backend_compile_oracle=proxy,
    )
    pending = SimpleNamespace(
        phase3_backend_compile_snapshot=None,
        backend_compile_snapshot=proxy_snapshot,
        compile_base_ops=("accepted:G0",),
    )

    phase0 = adapt_pipeline._default_no_prune_compile_oracle_for_stage(
        context=context,
        pending=pending,
        evaluation_stage="phase0",
    )

    assert phase0 == (proxy, proxy_snapshot, False)
    assert qiskit.snapshot_calls == []

    phase_i = adapt_pipeline._default_no_prune_compile_oracle_for_stage(
        context=context,
        pending=pending,
        evaluation_stage="phase1",
    )
    phase_ii = adapt_pipeline._default_no_prune_compile_oracle_for_stage(
        context=context,
        pending=pending,
        evaluation_stage="phase2",
    )

    assert phase_i[0] is qiskit and phase_i[2] is True
    assert phase_ii[1] is phase_i[1]
    assert qiskit.snapshot_calls == [("accepted:G0",)]
