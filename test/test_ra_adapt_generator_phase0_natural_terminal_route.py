from __future__ import annotations

from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1,
    ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1,
)
from pipelines.static_adapt.ra_adapt import (
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1,
    PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS,
    build_paper_i_ra_all_phase_adaptive_natural_terminal_request,
    build_paper_i_ra_all_phase_adaptive_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
)


def test_generator_phase0_natural_terminal_v2_is_distinct_and_authenticated(
) -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    v1 = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_adaptive_request(
            insertion_policy="plateau_commutation",
            maximum_controller_rounds=50,
        ),
    )
    v2 = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_adaptive_natural_terminal_request(
            insertion_policy="plateau_commutation",
            maximum_controller_rounds=50,
        ),
    )

    assert v1.route_contract["native_semantic_contract"]["route_variant"] == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_V1
    )
    assert v2.route_contract["native_semantic_contract"]["route_variant"] == (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    )
    assert v1.algorithm_id != v2.algorithm_id
    assert v1.bundle_id != v2.bundle_id
    assert v1.route_contract["sha256"] != v2.route_contract["sha256"]
    assert (
        PAPER_I_RA_ALL_PHASE_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
        in PAPER_I_RA_PHASE3_NATURAL_TERMINAL_ROUTE_VARIANTS
    )

    native = v2.route_contract["native_semantic_contract"]
    assert native["phase0_policy"]["population"] == (
        "same_ordered_append_endpoint_generator_population_v1"
    )
    assert native["phase0_policy"]["graph_proxy_cost"] == "off"
    assert native["phase0_policy"]["fubini_study_metric"] == "off"
    assert native["phase0_policy"]["qiskit_compile"] == "off"
    assert native["phase0_estimator_components"] == ["N_grad"]
    assert native["qiskit_active_phases"] == [
        "phase_i", "phase_ii", "phase_iii"
    ]
    assert native["phase_shortlist_maxima"] == {
        "phase_i": 24, "phase_ii": 12, "phase_iii": 12
    }
    assert native["phase_frontier_ratios"] == {
        "phase_i": 0.9, "phase_ii": 0.9, "phase_iii": 0.9
    }
    assert native["phase3_no_positive_policy"] == (
        ADAPTIVE_PHASE3_NO_POSITIVE_POLICY_TYPED_TERMINAL_V1
    )
    assert native["controller_horizon_policy"] == (
        ADAPTIVE_HORIZON_POLICY_MAXIMUM_V1
    )
