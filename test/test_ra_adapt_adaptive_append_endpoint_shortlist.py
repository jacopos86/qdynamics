from __future__ import annotations

import hashlib
import json
import math

import pytest

from pipelines.static_adapt.ra_adapt.adaptive_append_endpoint_shortlist import (
    AdaptivePhase0ActiveScore,
    AppendEndpointGeneratorScore,
    select_adaptive_phase0_active_score_shortlist,
    select_adaptive_append_endpoint_shortlist,
)


def test_adaptive_shortlist_retains_both_competition_champions() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        (
            AppendEndpointGeneratorScore(
                generator_index=0,
                append_gradient=4.0,
                graph_cost=16.0,
            ),
            AppendEndpointGeneratorScore(
                generator_index=1,
                append_gradient=3.0,
                graph_cost=1.0,
            ),
            AppendEndpointGeneratorScore(
                generator_index=2,
                append_gradient=1.0,
                graph_cost=1.0,
            ),
        ),
        cap=24,
    )

    assert decision.ranked_generator_indices == (1, 0, 2)
    assert decision.retained_generator_indices == (1, 0)
    assert decision.effective_shortlist_size == 2
    assert decision.effective_competitor_count == pytest.approx(121.0 / 83.0)
    assert decision.raw_gradient_champion_index == 0
    assert decision.weighted_utility_champion_index == 1
    assert decision.status == "competitive"
    assert decision.frontier_saturated is False


def test_adaptive_shortlist_closes_an_exact_boundary_tie() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        tuple(
            AppendEndpointGeneratorScore(
                generator_index=index,
                append_gradient=gradient,
                graph_cost=1.0,
            )
            for index, gradient in enumerate((3.0, 1.0, -1.0, 1.0))
        ),
        cap=24,
    )

    # The worked utilities are (9, 1, 1, 1): N_eff = 12^2 / 84,
    # which rounds to two.  The second-ranked boundary is an exact three-way
    # tie, so the complete competitive shell is retained.
    assert decision.effective_competitor_count == pytest.approx(12.0 / 7.0)
    assert decision.retained_generator_indices == (0, 1, 2, 3)
    assert decision.effective_shortlist_size == 4
    assert decision.frontier_saturated is False


def test_adaptive_shortlist_receipt_is_order_independent_and_append_scoped() -> None:
    population = (
        AppendEndpointGeneratorScore(7, -4.0, 16.0),
        AppendEndpointGeneratorScore(2, 3.0, 1.0),
        AppendEndpointGeneratorScore(5, 1.0, 2.0),
    )

    forward = select_adaptive_append_endpoint_shortlist(population, cap=24)
    reversed_input = select_adaptive_append_endpoint_shortlist(
        tuple(reversed(population)),
        cap=24,
    )
    receipt = forward.to_receipt()

    assert forward == reversed_input
    assert receipt == reversed_input.to_receipt()
    assert receipt["population_scope"] == "append_endpoint_generators_v1"
    assert receipt["position_aware_gradient_surface"] is False
    assert receipt["graph_proxy_cost_policy"] == "weighted_v1"
    assert receipt["qiskit_compile_cost_policy"] == "off"
    assert receipt["adaptive_law"] == "inverse_simpson_effective_population_v1"
    assert receipt["rounding_policy"] == "floor_n_eff_plus_one_half_v1"
    assert receipt["champion_policy"] == "raw_and_weighted_champions_v1"
    assert receipt["tie_policy"] == "exact_tie_closure_cap_saturation_v1"
    assert receipt["retained_generator_indices"] == [2, 7]
    unsigned = dict(receipt)
    observed_sha256 = unsigned.pop("sha256")
    expected_sha256 = hashlib.sha256(
        json.dumps(
            unsigned,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert observed_sha256 == expected_sha256


def test_adaptive_shortlist_rejects_a_cap_that_cannot_retain_both_champions() -> None:
    population = (
        AppendEndpointGeneratorScore(0, 4.0, 100.0),
        AppendEndpointGeneratorScore(1, 3.0, 1.0),
    )

    with pytest.raises(ValueError, match="cannot retain both champions"):
        select_adaptive_append_endpoint_shortlist(population, cap=1)


def test_adaptive_shortlist_marks_a_boundary_tie_cut_by_the_cap() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        tuple(
            AppendEndpointGeneratorScore(index, gradient, 1.0)
            for index, gradient in enumerate((3.0, 1.0, -1.0, 1.0))
        ),
        cap=3,
    )

    assert decision.retained_generator_indices == (0, 1, 2)
    assert decision.effective_shortlist_size == 3
    assert decision.frontier_saturated is True


def test_adaptive_shortlist_reports_an_all_zero_surface_as_stationary() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        (
            AppendEndpointGeneratorScore(4, 0.0, 2.0),
            AppendEndpointGeneratorScore(1, -0.0, 1.0),
        ),
        cap=24,
    )

    assert decision.ranked_generator_indices == (1, 4)
    assert decision.retained_generator_indices == ()
    assert decision.effective_shortlist_size == 0
    assert decision.effective_competitor_count == 0.0
    assert decision.raw_gradient_champion_index is None
    assert decision.weighted_utility_champion_index is None
    assert decision.status == "stationary"
    assert decision.to_receipt()["retained_generator_indices"] == []


def test_adaptive_shortlist_keeps_tiny_nonzero_gradients_competitive() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        (
            AppendEndpointGeneratorScore(4, 1.0e-200, 4.0),
            AppendEndpointGeneratorScore(1, -2.0e-200, 1.0),
            AppendEndpointGeneratorScore(7, 5.0e-201, 1.0),
        ),
        cap=24,
    )

    assert decision.status == "competitive"
    assert decision.ranked_generator_indices == (1, 4, 7)
    assert decision.weighted_utility_champion_index == 1
    assert decision.raw_gradient_champion_index == 1
    assert decision.retained_generator_indices == (1,)
    receipt = decision.to_receipt()
    assert receipt["positive_utility_candidate_count"] == 3
    assert receipt["ranking"][0]["utility_relative_to_champion"] == 1.0
    assert all(
        row["utility_log"] is not None for row in receipt["ranking"]
    )


def test_adaptive_shortlist_expands_for_raw_champion_without_dropping_tie() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        (
            AppendEndpointGeneratorScore(0, 10.0, 100.0),
            AppendEndpointGeneratorScore(1, 9.0, 1.0),
            AppendEndpointGeneratorScore(2, 9.0, 1.0),
        ),
        cap=24,
    )

    # Weighted utilities are (1, 81, 81).  N_eff rounds to two and the
    # weighted boundary is an exact tie.  The distinct raw-gradient champion
    # expands the retained set because the cap has room; it must not replace
    # either member of the tied weighted frontier.
    assert decision.ranked_generator_indices == (1, 2, 0)
    assert decision.retained_generator_indices == (1, 2, 0)
    assert decision.raw_gradient_champion_index == 0
    assert decision.weighted_utility_champion_index == 1
    assert decision.frontier_saturated is False


def test_adaptive_shortlist_contracts_to_one_clear_champion() -> None:
    decision = select_adaptive_append_endpoint_shortlist(
        (
            AppendEndpointGeneratorScore(0, 3.0, 1.0),
            AppendEndpointGeneratorScore(1, 1.0, 1.0),
            AppendEndpointGeneratorScore(2, 0.5, 1.0),
        ),
        cap=24,
    )

    assert decision.effective_competitor_count == pytest.approx(1681.0 / 1313.0)
    assert decision.retained_generator_indices == (0,)


def test_adaptive_shortlist_rejects_nonpositive_graph_cost() -> None:
    with pytest.raises(ValueError, match="graph costs must be positive"):
        select_adaptive_append_endpoint_shortlist(
            (AppendEndpointGeneratorScore(0, 1.0, 0.0),),
            cap=24,
        )


def test_adaptive_shortlist_rejects_duplicate_generator_indices() -> None:
    with pytest.raises(ValueError, match="indices must be unique"):
        select_adaptive_append_endpoint_shortlist(
            (
                AppendEndpointGeneratorScore(0, 1.0, 1.0),
                AppendEndpointGeneratorScore(0, 2.0, 1.0),
            ),
            cap=24,
        )


def test_adaptive_shortlist_rejects_overflowing_derived_utility() -> None:
    with pytest.raises(ValueError, match="utilities must be finite"):
        select_adaptive_append_endpoint_shortlist(
            (AppendEndpointGeneratorScore(0, 1.0e308, 1.0),),
            cap=24,
        )


def test_adaptive_shortlist_rejects_boolean_cap() -> None:
    with pytest.raises(ValueError, match="cap must be an integer"):
        select_adaptive_append_endpoint_shortlist(
            (AppendEndpointGeneratorScore(0, 1.0, 1.0),),
            cap=True,
        )


def test_v2_adaptive_cardinality_consumes_the_active_score_without_rescoring() -> None:
    decision = select_adaptive_phase0_active_score_shortlist(
        (
            AdaptivePhase0ActiveScore(0, 4.0),
            AdaptivePhase0ActiveScore(1, 3.0),
            AdaptivePhase0ActiveScore(2, 1.0),
        ),
        cap=24,
        active_score_policy="absolute_append_endpoint_generator_gradient_v1",
    )

    # Worked independently from the active scores: N_eff = 8^2 / 26.
    assert decision.effective_competitor_count == pytest.approx(32.0 / 13.0)
    assert decision.rounded_effective_competitor_count == 2
    assert decision.ranked_generator_indices == (0, 1, 2)
    assert decision.retained_generator_indices == (0, 1)
    receipt = decision.to_receipt()
    assert receipt["score"] == (
        "absolute_append_endpoint_generator_gradient_v1"
    )
    assert [row["active_score"] for row in receipt["ranking"]] == [
        4.0,
        3.0,
        1.0,
    ]


def test_v2_adaptive_cardinality_is_a_ranked_prefix_without_champion_injection() -> None:
    decision = select_adaptive_phase0_active_score_shortlist(
        (
            AdaptivePhase0ActiveScore(0, 1.0),
            AdaptivePhase0ActiveScore(1, 81.0),
            AdaptivePhase0ActiveScore(2, 81.0),
        ),
        cap=24,
        active_score_policy=(
            "absolute_append_gradient_over_graph_proxy_cost_v2"
        ),
    )

    assert decision.ranked_generator_indices == (1, 2, 0)
    assert decision.retained_generator_indices == (1, 2)
    assert decision.effective_shortlist_size == 2
    assert decision.frontier_saturated is False


def test_v2_adaptive_cardinality_closes_an_exact_boundary_tie() -> None:
    decision = select_adaptive_phase0_active_score_shortlist(
        tuple(
            AdaptivePhase0ActiveScore(index, score)
            for index, score in enumerate((9.0, 1.0, 1.0, 1.0))
        ),
        cap=24,
        active_score_policy="worked_active_score_v1",
    )

    # N_eff = 12^2 / 84, which rounds to two; the complete tied shell closes.
    assert decision.effective_competitor_count == pytest.approx(12.0 / 7.0)
    assert decision.retained_generator_indices == (0, 1, 2, 3)
    assert decision.effective_shortlist_size == 4


def test_v2_boundary_ties_compare_exact_scores_not_rounded_logs() -> None:
    lower = 1.0e100
    higher = math.nextafter(lower, math.inf)
    assert higher > lower
    assert math.log(higher) == math.log(lower)

    decision = select_adaptive_phase0_active_score_shortlist(
        (
            AdaptivePhase0ActiveScore(0, 3.0e100),
            AdaptivePhase0ActiveScore(1, lower),
            AdaptivePhase0ActiveScore(2, higher),
        ),
        cap=24,
        active_score_policy="worked_active_score_v1",
    )

    assert decision.rounded_effective_competitor_count == 2
    assert decision.ranked_generator_indices == (0, 2, 1)
    assert decision.retained_generator_indices == (0, 2)


def test_v2_exact_tie_closure_is_subject_to_the_hard_cap() -> None:
    decision = select_adaptive_phase0_active_score_shortlist(
        tuple(
            AdaptivePhase0ActiveScore(index, score)
            for index, score in enumerate((9.0, 1.0, 1.0, 1.0))
        ),
        cap=3,
        active_score_policy="worked_active_score_v1",
    )

    assert decision.retained_generator_indices == (0, 1, 2)
    assert decision.frontier_saturated is True
    assert decision.to_receipt()["retention_policy"] == (
        "ranked_prefix_with_exact_boundary_tie_closure_subject_to_hard_cap_v2"
    )


def test_v2_adaptive_cardinality_all_zero_surface_is_stationary() -> None:
    decision = select_adaptive_phase0_active_score_shortlist(
        (
            AdaptivePhase0ActiveScore(4, 0.0),
            AdaptivePhase0ActiveScore(1, 0.0),
        ),
        cap=24,
        active_score_policy="absolute_append_endpoint_generator_gradient_v1",
    )

    assert decision.status == "stationary"
    assert decision.ranked_generator_indices == (1, 4)
    assert decision.retained_generator_indices == ()
    assert decision.to_receipt()["retained_generator_indices"] == []
