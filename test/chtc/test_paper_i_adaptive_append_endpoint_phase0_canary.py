from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np

from pipelines.scaffold.hh_continuation_scoring import (
    SimpleScoreConfig,
    hardware_cost_candidate_record_denominators,
)

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / (
    "chtc/paper_i_ra_adapt_repair_20260727/"
    "adaptive_append_endpoint_phase0_canary_20260816.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "adaptive_append_endpoint_phase0_canary_test",
        MODULE_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixed_arm_uses_graph_weighted_top24_and_records_adaptive_shadow() -> None:
    module = load_module()
    rows = [
        {
            "pool_index": 0,
            "generator_id": "g0",
            "pool_label": "G0",
            "append_gradient_signed": 3.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "pool_index": 1,
            "generator_id": "g1",
            "pool_label": "G1",
            "append_gradient_signed": 1.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "pool_index": 2,
            "generator_id": "g2",
            "pool_label": "G2",
            "append_gradient_signed": 0.5,
            "graph_proxy_denominator": 1.0,
        },
    ]

    decision = module.select_append_endpoint_phase0_rows(
        rows,
        mode=module.MODE_FIXED24_SHADOW,
        cap=24,
    )

    assert decision["ranked_pool_indices"] == [0, 1, 2]
    assert decision["retained_pool_indices"] == [0, 1, 2]
    assert decision["active_shortlist_policy"] == "fixed_top_k_by_utility_v1"
    assert decision["adaptive_decision"]["retained_generator_indices"] == [0]
    assert decision["adaptive_decision_role"] == "shadow"


def test_adaptive_arm_filters_only_generator_identity_not_position() -> None:
    module = load_module()
    rows = [
        {
            "pool_index": 0,
            "generator_id": "g0",
            "pool_label": "G0",
            "append_gradient_signed": 3.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "pool_index": 1,
            "generator_id": "g1",
            "pool_label": "G1",
            "append_gradient_signed": 1.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "pool_index": 2,
            "generator_id": "g2",
            "pool_label": "G2",
            "append_gradient_signed": 0.5,
            "graph_proxy_denominator": 1.0,
        },
    ]
    domain = tuple(
        SimpleNamespace(
            pool_index=pool_index,
            insertion_position=position,
            domain_record_id=f"g{pool_index}@{position}",
        )
        for pool_index, position in ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))
    )

    decision = module.select_append_endpoint_phase0_rows(
        rows,
        mode=module.MODE_ACTIVE_ADAPTIVE,
        cap=24,
    )
    filtered = module.filter_position_domain_by_retained_generators(
        domain,
        ranked_pool_indices=decision["ranked_pool_indices"],
        retained_pool_indices=decision["retained_pool_indices"],
    )

    assert decision["retained_pool_indices"] == [0]
    assert decision["adaptive_decision_role"] == "active"
    assert [(row.pool_index, row.insertion_position) for row in filtered] == [
        (0, 0),
        (0, 2),
    ]


def test_receipt_is_append_scoped_closes_gradient_work_and_recomputes() -> None:
    module = load_module()
    compile_rows = [
        {
            "label": f"G{pool_index}",
            "candidate_pool_index": pool_index,
            "position_id": 4,
            "c_hat_2q": 2.0 + pool_index,
            "c_hat_d": 3.0 + pool_index,
            "c_hat_1q": 4.0 + pool_index,
            "c_hat_theta": 1.0,
            "c_hat_shot": 0.0,
        }
        for pool_index in range(3)
    ]
    normalization = hardware_cost_candidate_record_denominators(
        compile_rows,
        SimpleScoreConfig(),
    )
    gradients = (3.0, 1.0, 0.5)
    rows = [
        {
            "pool_index": pool_index,
            "generator_id": f"g{pool_index}",
            "pool_label": f"G{pool_index}",
            "append_position": 4,
            "append_gradient_signed": gradients[pool_index],
            "graph_proxy_source": "proxy_logical_ladder_span_v1",
            "graph_proxy_raw": dict(normalization["rows"][pool_index]["raw"]),
            "graph_proxy_bars": dict(normalization["rows"][pool_index]["bars"]),
            "graph_proxy_cost_excess_sum": normalization["rows"][pool_index][
                "hardware_cost_excess_sum"
            ],
            "graph_proxy_denominator": normalization["denominators"][pool_index],
        }
        for pool_index in range(3)
    ]
    event_ids = ["estimator:1:g0", "estimator:2:g1", "estimator:3:g2"]

    receipt = module.build_append_endpoint_phase0_receipt(
        rows,
        graph_proxy_normalization=normalization,
        estimator_event_ids=event_ids,
        mode=module.MODE_FIXED24_SHADOW,
        cap=24,
    )

    assert module.validate_append_endpoint_phase0_receipt(receipt) == receipt
    assert receipt["gradient_surface"] == "append_endpoint_generators_v1"
    assert receipt["position_aware_gradient_surface"] is False
    assert receipt["qiskit_compile_cost_policy"] == "off"
    assert receipt["qiskit_compile_cost_scope"] == "phase0_only_v1"
    assert receipt["metric_policy"] == "off"
    assert receipt["semantic_ownership_scope"] == (
        "phase0_append_endpoint_shortlist_only_v1"
    )
    assert receipt["later_phase_semantics_ownership"] == (
        "external_source_locked_route_contract_v1"
    )
    assert receipt["later_phase_qiskit_semantics_claimed"] is False
    assert receipt["later_phase_zero_centered_semantics_claimed"] is False
    assert receipt["execution_authorized"] is False
    assert receipt["native_semantic_closure_required"] is True
    assert receipt["execution_authority"] == "none_inert_implementation_only_v1"
    assert receipt["estimator_accounting"]["N_grad"] == 3
    assert receipt["estimator_accounting"]["S_alg"] == 3
    assert receipt["adaptive_shadow_accounting"]["S_alg"] == 0
    assert receipt["estimator_event_ids"] == event_ids

    tampered = copy.deepcopy(receipt)
    tampered["ranking"][0]["utility"] = 0.0
    unsigned = dict(tampered)
    unsigned.pop("sha256")
    tampered["sha256"] = module.canonical_sha256(unsigned)
    with pytest.raises(RuntimeError, match="recomputation"):
        module.validate_append_endpoint_phase0_receipt(tampered)


def test_zero_utility_surface_is_stationary_and_boolean_cap_is_rejected() -> None:
    module = load_module()
    rows = [
        {
            "pool_index": 0,
            "generator_id": "g0",
            "pool_label": "G0",
            "append_gradient_signed": 0.0,
            "graph_proxy_denominator": 1.0,
        },
        {
            "pool_index": 1,
            "generator_id": "g1",
            "pool_label": "G1",
            "append_gradient_signed": -0.0,
            "graph_proxy_denominator": 2.0,
        },
    ]

    fixed = module.select_append_endpoint_phase0_rows(
        rows,
        mode=module.MODE_FIXED24_SHADOW,
        cap=24,
    )
    adaptive = module.select_append_endpoint_phase0_rows(
        rows,
        mode=module.MODE_ACTIVE_ADAPTIVE,
        cap=24,
    )

    assert fixed["status"] == "stationary"
    assert fixed["retained_pool_indices"] == []
    assert adaptive["status"] == "stationary"
    assert adaptive["retained_pool_indices"] == []
    with pytest.raises(ValueError, match="integer, not bool"):
        module.select_append_endpoint_phase0_rows(
            rows,
            mode=module.MODE_FIXED24_SHADOW,
            cap=True,
        )


def test_runtime_executes_one_append_gradient_per_generator_and_keeps_positions() -> None:
    module = load_module()

    class CompileOracle:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def estimate(self, **kwargs):
            self.calls.append(dict(kwargs))
            return SimpleNamespace(
                c_hat_2q=2.0,
                c_hat_d=3.0,
                c_hat_1q=4.0,
                c_hat_theta=1.0,
                hardware_cost_source="proxy_logical_ladder_span_v1",
                source_mode="proxy",
            )

    compile_oracle = CompileOracle()
    pending = SimpleNamespace(
        phase0_gradient_shortlist_receipt=None,
        append_position=2,
        theta_logical_current=np.asarray([0.1, 0.2]),
        gradients=np.zeros(3, dtype=float),
        grad_magnitudes=np.zeros(3, dtype=float),
        available_sorted=[0, 1, 2],
        shortlist=[0, 1, 2],
        phase2_score_cfg_round=SimpleScoreConfig(),
    )
    context = SimpleNamespace(
        pool=tuple(SimpleNamespace(label=f"G{index}") for index in range(3)),
        compiled_pool=tuple(SimpleNamespace(terms=(object(),)) for _ in range(3)),
        phase1_compile_oracle=compile_oracle,
        reoptimization_policy="windowed",
        reoptimization_window_size=2,
        reoptimization_window_topk=2,
        transition_services=SimpleNamespace(controller_noise_runtime=None),
    )
    cursor = SimpleNamespace(
        available_indices={0, 1, 2},
        selected_ops=[object(), object()],
    )

    class Session:
        def __init__(self) -> None:
            self.context = context
            self.cursor = cursor
            self.gradient_calls = 0
            self.occurrences: list[dict[str, object]] = []

        def _evaluate_default_candidate_gradient_surface(
            self,
            pending_arg,
            *,
            consumer_scope: str,
        ) -> None:
            self.gradient_calls += 1
            pending_arg.gradients[:] = np.asarray([3.0, 1.0, 0.5])
            pending_arg.grad_magnitudes[:] = np.abs(pending_arg.gradients)
            self.occurrences.extend(
                {
                    "component": "N_grad",
                    "consumer_scope": consumer_scope,
                    "sequence": index + 1,
                    "primitive_id": f"g{index}",
                }
                for index in range(3)
            )

        def _refresh_default_candidate_gradient_summaries(self, _pending) -> None:
            return None

    session = Session()
    transaction = SimpleNamespace(
        session=session,
        pending=pending,
        context=context,
        cursor=cursor,
        ledger_occurrences=lambda: list(session.occurrences),
    )
    pipeline = SimpleNamespace(
        _predict_nested_refit_window_for_position=lambda **_kwargs: object(),
        build_nested_window_accounting=lambda *_args, **_kwargs: SimpleNamespace(
            compile_proxy_refit_count=2
        ),
        _ShortlistRankReceipt=lambda **kwargs: SimpleNamespace(**kwargs),
        _PhaseSelectionReceipt=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    domain = tuple(
        SimpleNamespace(
            pool_index=pool_index,
            generator_id=f"g{pool_index}",
            pool_label=f"G{pool_index}",
            insertion_position=position,
            domain_record_id=f"g{pool_index}@{position}",
        )
        for pool_index, position in ((0, 0), (0, 2), (1, 0), (1, 2), (2, 1))
    )

    result = module.execute_adaptive_append_endpoint_phase0(
        transaction,
        pipeline_module=pipeline,
        admissible_domain=domain,
        shortlist_size=24,
        policy=module.PARENT_ABSOLUTE_GRADIENT_PHASE0_POLICY,
        receipt_schema=module.PARENT_GLOBAL_SINGLETON_PHASE0_RECEIPT_SCHEMA,
        consumer_scope="phase0_global_singleton_gradient_surface",
        population_scope="current_available_global_guarded_singletons_v1",
        mode=module.MODE_ACTIVE_ADAPTIVE,
    )

    assert session.gradient_calls == 1
    assert len(session.occurrences) == 3
    assert [call["position_id"] for call in compile_oracle.calls] == [2, 2, 2]
    assert [call["append_position"] for call in compile_oracle.calls] == [2, 2, 2]
    assert [(row.pool_index, row.insertion_position) for row in result.shortlist] == [
        (0, 0),
        (0, 2),
    ]
    assert pending.phase0_gradient_shortlist_receipt["estimator_accounting"][
        "N_grad"
    ] == 3
    assert pending.phase0_gradient_shortlist_receipt[
        "position_aware_gradient_surface"
    ] is False
    project = lambda row: {  # noqa: E731 - compact local receipt projection
        "domain_record_id": row.domain_record_id,
        "generator_id": row.generator_id,
        "pool_index": row.pool_index,
        "pool_label": row.pool_label,
        "insertion_position": row.insertion_position,
        "position_class": (
            "interior" if row.insertion_position < pending.append_position else "append"
        ),
    }
    population_projection = [project(row) for row in domain]
    shortlist_projection = [project(row) for row in result.shortlist]
    screen = {
        "schema": "paper_i_scored_gradient_phase0_population_v1",
        "population_count": len(population_projection),
        "population": population_projection,
        "ordered_population_sha256": module.canonical_sha256(population_projection),
        "shortlist_count": len(shortlist_projection),
        "shortlist": shortlist_projection,
        "ordered_shortlist_sha256": module.canonical_sha256(shortlist_projection),
    }
    assert module.validate_append_endpoint_phase0_receipt(
        pending.phase0_gradient_shortlist_receipt,
        scored_population={"phase0_gradient_screen": screen},
    ) == pending.phase0_gradient_shortlist_receipt

    position_tamper = copy.deepcopy(screen)
    position_tamper["shortlist"] = position_tamper["shortlist"][:-1]
    position_tamper["shortlist_count"] = len(position_tamper["shortlist"])
    position_tamper["ordered_shortlist_sha256"] = module.canonical_sha256(
        position_tamper["shortlist"]
    )
    with pytest.raises(RuntimeError, match="position domain"):
        module.validate_append_endpoint_phase0_receipt(
            pending.phase0_gradient_shortlist_receipt,
            scored_population={"phase0_gradient_screen": position_tamper},
        )


def test_temporary_overlay_refuses_installation_without_native_semantic_closure() -> None:
    module = load_module()

    class Transaction:
        def run_absolute_gradient_phase0(self, **_kwargs):
            return "original-phase0"

    def original_validator(*_args, **_kwargs):
        return {"original": True}

    original_phase0 = Transaction.run_absolute_gradient_phase0
    pipeline = SimpleNamespace(_DefaultNoPruneSelectionTransaction=Transaction)
    engine = SimpleNamespace(
        _validated_gradient_phase0_round_receipt=original_validator
    )

    with pytest.raises(RuntimeError, match="native semantic-closure route"):
        module.install_adaptive_append_endpoint_phase0_overlay(
            mode=module.MODE_ACTIVE_ADAPTIVE,
            pipeline_module=pipeline,
            engine_module=engine,
        )

    assert Transaction.run_absolute_gradient_phase0 is original_phase0
    assert engine._validated_gradient_phase0_round_receipt is original_validator
