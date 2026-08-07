from __future__ import annotations

import pytest

from pipelines.contracts.static_provenance import (
    classify_hh_physical_operator_lane,
    classify_static_physical_operator_lane,
    summarize_static_physical_operator_pool_labels,
)
from pipelines.static_adapt.lane_routes import (
    STATIC_LANE_ROUTE_CHOICES,
    STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
    clamp_controller_cap_pair_for_lane_route,
    normalize_static_lane_route,
    physical_lane_route_variant_id_for_problem,
    resolve_static_shortlist_lane_spec,
)


def test_retired_algebraic_lane_route_is_not_a_runtime_choice() -> None:
    assert STATIC_LANE_ROUTE_CHOICES == (
        STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
    )
    assert normalize_static_lane_route(None) == (
        STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE
    )
    with pytest.raises(ValueError, match="static_lane_route"):
        normalize_static_lane_route("algebraic")


def test_physical_lane_route_fixes_controller_caps_to_effective_cap() -> None:
    assert clamp_controller_cap_pair_for_lane_route(
        route="physical_operator_type",
        cap_min=24,
        cap_max=24,
        effective_cap=8,
    ) == (8, 8)
    assert clamp_controller_cap_pair_for_lane_route(
        route="physical_operator_type",
        cap_min=12,
        cap_max=12,
        effective_cap=4,
    ) == (4, 4)


def test_physical_lane_route_ignores_smaller_legacy_maturity_caps() -> None:
    assert clamp_controller_cap_pair_for_lane_route(
        route="physical_operator_type",
        cap_min=2,
        cap_max=6,
        effective_cap=8,
    ) == (8, 8)


def test_hubbard_physical_lane_route_includes_qeb_and_hva_lanes() -> None:
    spec = resolve_static_shortlist_lane_spec("physical_operator_type", problem="hubbard")
    assert "uccsd_single" in spec.lanes
    assert "uccsd_double" in spec.lanes
    assert "qeb_excitation" in spec.lanes
    assert "hva_hamiltonian_blocks" in spec.lanes
    assert "other" in spec.lanes
    assert physical_lane_route_variant_id_for_problem("hubbard") == "route_a_hubbard_physical_operator_lanes_v3_uccsd_qeb_hva_blocks"


def test_hubbard_qeb_hva_physical_operator_lane_classifier_and_audit() -> None:
    assert (
        classify_static_physical_operator_lane(
            "qeb_pair(0,1)",
            problem="hubbard",
        )["physical_operator_lane"]
        == "qeb_excitation"
    )
    assert (
        classify_static_physical_operator_lane(
            "qeb_double(0,3->1,2)",
            problem="hubbard",
        )["physical_operator_lane"]
        == "qeb_excitation"
    )
    assert (
        classify_static_physical_operator_lane(
            "hva_block::hop_layer",
            problem="hubbard",
        )["physical_operator_lane"]
        == "hva_hamiltonian_blocks"
    )
    assert (
        classify_static_physical_operator_lane(
            "hva_block::onsite_layer",
            problem="hubbard",
        )["physical_operator_lane"]
        == "hva_hamiltonian_blocks"
    )
    assert (
        classify_static_physical_operator_lane(
            "uccsd_sing(alpha:0->1)::child_set[0,1]",
            problem="hubbard",
        )["physical_operator_lane"]
        == "uccsd_single"
    )
    assert (
        classify_static_physical_operator_lane(
            "uccsd_dbl(ab:0,2->1,3)::child_set[0,1,2]",
            problem="hubbard",
        )["physical_operator_lane"]
        == "uccsd_double"
    )
    assert (
        classify_static_physical_operator_lane(
            "qeb_pair_alt(0,1)",
            problem="hubbard",
        )["physical_operator_lane"]
        == "other"
    )

    audit = summarize_static_physical_operator_pool_labels(
        [
            "uccsd_sing(alpha:0->1)",
            "uccsd_dbl(ab:0,2->1,3)",
            "qeb_pair(0,1)",
            "qeb_double(0,3->1,2)::child_set[0,2,3]",
            "hva_block::hop_layer",
            "hva_block::onsite_layer",
            "hva_block::hop_layer::child_set[0,1]",
        ],
        problem="hubbard",
    )
    assert audit["other_count"] == 0
    assert audit["exact_other_labels"] == []
    assert audit["lane_counts"]["uccsd_single"] == 1
    assert audit["lane_counts"]["uccsd_double"] == 1
    assert audit["lane_counts"]["qeb_excitation"] == 2
    assert audit["lane_counts"]["hva_hamiltonian_blocks"] == 3


def test_hh_physical_operator_lane_classifier_keeps_families_separate() -> None:
    assert (
        classify_hh_physical_operator_lane(
            "uccsd_ferm_lifted::uccsd_dbl(i=0,j=1)"
        )["physical_operator_lane"]
        == "uccsd_double"
    )
    assert (
        classify_hh_physical_operator_lane("hh_phonon::x(site=0)")[
            "physical_operator_lane"
        ]
        == "phonon_displacement"
    )
    assert (
        classify_hh_physical_operator_lane("hh_phonon::x_sq(site=0)")[
            "physical_operator_lane"
        ]
        == "phonon_squeeze_relaxation"
    )
    assert (
        classify_hh_physical_operator_lane(
            "paop_0:paop_hopdrag(i=0,j=1)"
        )["physical_operator_lane"]
        == "dressed_phonon_correlation"
    )
    assert (
        classify_hh_physical_operator_lane("ham_block::phonon")[
            "physical_operator_lane"
        ]
        == "hva_hamiltonian_blocks"
    )
