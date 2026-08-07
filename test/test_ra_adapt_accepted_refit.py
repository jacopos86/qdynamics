from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
import pipelines.static_adapt.adapt_pipeline as adapt_pipeline
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    MacroCandidateAdapter,
    RAAdaptRequest,
    SinglePauliWordCandidateAdapter,
    run_ra_adapt,
)
from pipelines.static_adapt.sr_snake import (
    AlwaysCommutationReducedInsertion,
    SRExecutionPolicy,
    SRMethodPolicy,
    SRStopPolicy,
)


def _problem() -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def test_both_adapters_use_one_common_fixed_refit_chart_per_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original: Callable[..., Any] = (
        adapt_pipeline.build_supported_fs_powell_chart
    )
    calls: list[str] = []

    def _record_common_builder(*args: Any, **kwargs: Any) -> Any:
        calls.append(f"{original.__module__}.{original.__qualname__}")
        return original(*args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "build_supported_fs_powell_chart",
        _record_common_builder,
    )
    problem = _problem()
    builder_counts: list[int] = []
    receipt_schemas: list[str] = []
    for adapter in (
        SinglePauliWordCandidateAdapter(),
        MacroCandidateAdapter(),
    ):
        count_before = len(calls)
        result = run_ra_adapt(
            problem,
            RAAdaptRequest(
                adapter=adapter,
                execution=SRExecutionPolicy(
                    stop=SRStopPolicy(maximum_controller_rounds=1)
                ),
            ),
        )
        builder_counts.append(len(calls) - count_before)
        accepted_rounds = result.scientific_receipts[
            "accepted_round_receipts"
        ]
        assert len(accepted_rounds) == len(
            result.run.accepted_transitions
        ) == 1
        accepted = accepted_rounds[0]
        assert (
            list(accepted).count(
                "accepted_refit_fixed_chart_receipt"
            )
            == 1
        )
        chart = accepted["accepted_refit_fixed_chart_receipt"]
        receipt_schemas.append(str(chart["schema"]))
        assert chart["scope"] == "full_ansatz_v1"
        assert chart["coordinate_chart"] == (
            "supported_fs_whitened_fixed_v1"
        )
        assert chart["chart_lifetime"] == (
            "fixed_for_one_optimizer_invocation_then_discarded_v1"
        )
        assert chart["sha256"] == accepted[
            "accepted_refit_fixed_chart_sha256"
        ]
        initialization = accepted[
            "accepted_refit_initialization"
        ]
        assert initialization["policy"] == (
            "exact_applied_joint_step_guarded_v1"
        )
        assert initialization["status"] in {"accepted", "rejected"}
        assert initialization["guard_objective_evals"] == 1
        assert initialization["guard_objective_stage"] == (
            "accepted_refit_joint_response_guard"
        )
        assert accepted[
            "accepted_refit_initialization_guard_nfev"
        ] == 1
        mapping = initialization["supported_fs_mapping"]
        assert mapping["source_step_within_supported_chart"] is True
        assert mapping["classical_quantum_query_charge"] == 0
        gain = initialization["phase3_candidate_gain_receipt"]
        assert gain["policy"] == (
            "joint_minus_active_only_supported_trust_v1"
        )
        assert gain["active_only_baseline"][
            "classical_quantum_query_charge"
        ] == 0
        assert gain["full_joint_trust_gain"] >= gain[
            "active_only_trust_gain"
        ] - gain["comparison_tolerance"]
        components = result.run.estimator_accounting.all_work.components
        assert result.run.estimator_accounting.all_work.s_alg == (
            components.n_h_outer
            + components.n_h_refit
            + components.n_grad
            + components.n_metric
        )

    assert builder_counts == [1, 1]
    assert len(set(calls)) == 1
    assert calls[0] == (
        "pipelines.static_adapt.accepted_refit."
        "build_supported_fs_powell_chart"
    )
    assert receipt_schemas == [
        "accepted_refit_fixed_chart_receipt_v1",
        "accepted_refit_fixed_chart_receipt_v1",
    ]


def test_round_two_seed_carries_existing_and_new_coordinates() -> None:
    result = run_ra_adapt(
        _problem(),
        RAAdaptRequest(
            adapter=SinglePauliWordCandidateAdapter(),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=2)
            ),
        ),
    )

    rounds = result.scientific_receipts["accepted_round_receipts"]
    assert len(rounds) == 2
    second = rounds[1]
    initialization = second["accepted_refit_initialization"]
    mapping = initialization["supported_fs_mapping"]
    gain = initialization["phase3_candidate_gain_receipt"]
    baseline = gain["active_only_baseline"]

    assert baseline["active_coordinate_count"] == 1
    assert len(second["accepted_candidate_lineage"]) == 1
    assert mapping["logical_parameter_count"] == 2
    assert len(mapping["phase_order_joint_step"]) == 2
    assert sorted(mapping["phase3_to_post_logical_permutation"]) == [0, 1]
    assert mapping["source_coordinate_order"] == (
        "phase3_active_then_selected_batch_v1"
    )
    assert initialization[
        "mapped_seed_predicted_full_joint_reduction"
    ] == pytest.approx(gain["full_joint_trust_gain"])
    assert gain["incremental_candidate_gain_raw"] == pytest.approx(
        gain["full_joint_trust_gain"]
        - gain["active_only_trust_gain"]
    )
    assert baseline["classical_quantum_query_charge"] == 0
    assert sum(
        row["accepted_refit_initialization_guard_nfev"]
        for row in rounds
    ) == 2


def test_macro_always_insertion_three_rounds_close_full_response_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise incremental selection and full old-plus-new refit together."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(
        adapt_pipeline,
        "_ai_log",
        lambda *_args, **_kwargs: None,
    )
    result = run_ra_adapt(
        _problem(),
        RAAdaptRequest(
            adapter=MacroCandidateAdapter(),
            method=SRMethodPolicy(
                insertion=AlwaysCommutationReducedInsertion()
            ),
            execution=SRExecutionPolicy(
                stop=SRStopPolicy(maximum_controller_rounds=3)
            ),
        ),
    )

    assert result.protocol.algorithm_id == (
        "paper_i_ra_adapt_nonstationary_incremental_full_response_v2"
    )
    assert result.protocol.schema == "paper_i_ra_adapt_resolved_protocol_v2"
    assert result.schema == "paper_i_ra_adapt_result_v2"
    assert result.protocol.route_contract is not None
    assert result.protocol.route_contract["schema"] == (
        "paper_i_ra_adapt_route_contract_v2"
    )
    assert result.protocol.candidate_representation == "macro_generator_v1"
    assert result.run.route.insertion_policy == (
        "always_commutation_reduced"
    )
    assert result.policy.phase3_candidate_gain_policy == (
        "joint_minus_active_only_supported_trust_v1"
    )
    assert result.policy.accepted_refit_initialization_policy == (
        "exact_applied_joint_step_guarded_v1"
    )
    assert result.run.stop.completed_controller_rounds == 3

    rounds = result.scientific_receipts["accepted_round_receipts"]
    assert len(rounds) == len(result.run.accepted_transitions) == 3
    for active_count, accepted in enumerate(rounds):
        reduction = accepted["insertion_commutation_reduced"]
        assert reduction["policy"] == "always_commutation_reduced"
        assert reduction["domain_open"] is True
        assert reduction["requested_positions"] == list(
            range(active_count + 1)
        )

        initialization = accepted["accepted_refit_initialization"]
        mapping = initialization["supported_fs_mapping"]
        gain = initialization["phase3_candidate_gain_receipt"]
        baseline = gain["active_only_baseline"]
        tolerance = max(float(gain["comparison_tolerance"]), 1.0e-12)

        assert baseline["active_coordinate_count"] == active_count
        assert baseline["candidate_independent"] is True
        assert baseline["classical_quantum_query_charge"] == 0
        assert gain["policy"] == (
            "joint_minus_active_only_supported_trust_v1"
        )
        assert gain["joint_gain_semantics"] == (
            "incremental_candidate_gain_v1"
        )
        assert gain["incremental_candidate_gain_raw"] == pytest.approx(
            gain["full_joint_trust_gain"]
            - gain["active_only_trust_gain"],
            abs=tolerance,
        )
        assert gain["incremental_candidate_gain"] == pytest.approx(
            max(0.0, gain["incremental_candidate_gain_raw"]),
            abs=tolerance,
        )
        assert gain["selected_gain"] == pytest.approx(
            gain["incremental_candidate_gain"],
            abs=tolerance,
        )

        logical_count = active_count + 1
        assert len(accepted["accepted_candidate_lineage"]) == 1
        assert mapping["logical_parameter_count"] == logical_count
        assert len(mapping["phase_order_joint_step"]) == logical_count
        assert sorted(mapping["phase3_to_post_logical_permutation"]) == (
            list(range(logical_count))
        )
        assert mapping["source_coordinate_order"] == (
            "phase3_active_then_selected_batch_v1"
        )
        assert mapping["source_step_within_supported_chart"] is True
        assert mapping["classical_quantum_query_charge"] == 0
        assert initialization[
            "mapped_seed_predicted_full_joint_reduction"
        ] == pytest.approx(
            gain["full_joint_trust_gain"],
            abs=tolerance,
        )
        assert initialization["status"] in {"accepted", "rejected"}
        assert initialization["guard_objective_evals"] == 1
        assert accepted["accepted_refit_initialization_guard_nfev"] == 1
        typed_refit = result.run.scientific_replay[
            active_count
        ].accepted_refit
        assert typed_refit.initialization_policy == (
            "exact_applied_joint_step_guarded_v1"
        )
        assert typed_refit.initialization_status in {"accepted", "rejected"}
        assert typed_refit.initialization_guard_nfev == 1

        chart = accepted["accepted_refit_fixed_chart_receipt"]
        assert chart["scope"] == "full_ansatz_v1"
        assert chart["coordinate_chart"] == (
            "supported_fs_whitened_fixed_v1"
        )
        metric_accounting = accepted[
            "accepted_refit_metric_query_accounting"
        ]
        assert metric_accounting["status"] == (
            "reused_external_logical_fs_gram_receipt"
        )
        assert metric_accounting["incremental_quantum_query_charge"] == 0

    assert sum(
        row["accepted_refit_initialization_guard_nfev"] for row in rounds
    ) == 3
    components = result.run.estimator_accounting.all_work.components
    assert result.run.estimator_accounting.all_work.s_alg == (
        components.n_h_outer
        + components.n_h_refit
        + components.n_grad
        + components.n_metric
    )
