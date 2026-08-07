"""Static ADAPT shortlist lane route contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipelines.contracts.static_provenance import (
    HH_PHYSICAL_OPERATOR_LANE_OTHER,
    normalize_static_physical_operator_problem,
    physical_operator_lanes_for_problem,
)
from pipelines.static_adapt.algebraic_metadata import LANES_PHASE1, LANE_MIX


STATIC_LANE_ROUTE_ALGEBRAIC = "algebraic"
STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE = "physical_operator_type"
STATIC_LANE_ROUTE_CHOICES = (
    STATIC_LANE_ROUTE_ALGEBRAIC,
    STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
)

PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES = (2, 3)
PHYSICAL_LANE_ROUTE_VARIANT_ID = "route_a_physical_operator_lanes_v2_uccsd_split"
PHYSICAL_LANE_ROUTE_VARIANT_IDS_BY_PROBLEM = {
    "hh": PHYSICAL_LANE_ROUTE_VARIANT_ID,
    "hubbard": "route_a_hubbard_physical_operator_lanes_v3_uccsd_qeb_hva_blocks",
    "spin_boson": "route_a_spin_boson_physical_operator_lanes_v2_full_meta_hamiltonian_blocks",
    "bose_hubbard": "route_a_bose_hubbard_physical_operator_lanes_v2_full_meta_hamiltonian_blocks",
    "molecular_restricted_closed_shell": "route_a_molecular_restricted_physical_operator_lanes_v1",
    "molecular_vibronic_h2o_linear_fd": "route_a_h2o_linear_fd_physical_operator_lanes_v1",
}


@dataclass(frozen=True)
class StaticShortlistLaneSpec:
    route: str
    lane_key: str
    lanes: tuple[str, ...]
    fallback_lane: str
    health_key_prefix: str


def normalize_static_lane_route(value: Any, *, default: str = STATIC_LANE_ROUTE_ALGEBRAIC) -> str:
    raw = str(default if value in {None, ""} else value).strip().lower().replace("-", "_")
    aliases = {
        "physical": STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
        "operator_type": STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
        "physical_operator": STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
        "physical_operator_lanes": STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
        "physical_operator_type_lanes": STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
    }
    route = aliases.get(raw, raw)
    if route not in STATIC_LANE_ROUTE_CHOICES:
        raise ValueError(
            "static_lane_route must be one of "
            f"{list(STATIC_LANE_ROUTE_CHOICES)}; got {value!r}."
        )
    return str(route)


def normalize_physical_lane_shortlist_aggressiveness(value: Any, *, default: int = 3) -> int:
    raw = int(default if value in {None, ""} else value)
    if raw not in set(PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES):
        raise ValueError(
            "physical_lane_shortlist_aggressiveness must be one of "
            f"{list(PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES)}; got {value!r}."
        )
    return int(raw)


def physical_lane_route_variant_id_for_problem(problem: Any) -> str:
    problem_key = normalize_static_physical_operator_problem(problem)
    return str(PHYSICAL_LANE_ROUTE_VARIANT_IDS_BY_PROBLEM[problem_key])


def resolve_static_shortlist_lane_spec(
    route: Any,
    *,
    problem: Any = "hh",
) -> StaticShortlistLaneSpec:
    route_key = normalize_static_lane_route(route)
    if route_key == STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE:
        return StaticShortlistLaneSpec(
            route=STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE,
            lane_key="physical_operator_lane",
            lanes=tuple(str(x) for x in physical_operator_lanes_for_problem(problem)),
            fallback_lane=str(HH_PHYSICAL_OPERATOR_LANE_OTHER),
            health_key_prefix="physical_operator",
        )
    return StaticShortlistLaneSpec(
        route=STATIC_LANE_ROUTE_ALGEBRAIC,
        lane_key="algebraic_lane",
        lanes=tuple(str(x) for x in LANES_PHASE1),
        fallback_lane=str(LANE_MIX),
        health_key_prefix="algebraic",
    )


def clamp_controller_cap_pair_for_lane_route(
    *,
    route: Any,
    cap_min: int,
    cap_max: int,
    effective_cap: int,
) -> tuple[int, int]:
    """Resolve phase caps while keeping maturity scheduling legacy-only.

    Physical-operator lanes use the explicit phase shortlist cap at every
    iteration. Algebraic routes retain their historical maturity-cap range.
    """

    route_key = normalize_static_lane_route(route)
    cap_min_val = int(cap_min)
    cap_max_val = int(cap_max)
    if route_key != STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE:
        return int(cap_min_val), int(cap_max_val)
    effective_cap_val = int(max(1, int(effective_cap)))
    return int(effective_cap_val), int(effective_cap_val)


__all__ = [
    "PHYSICAL_LANE_ROUTE_VARIANT_ID",
    "PHYSICAL_LANE_ROUTE_VARIANT_IDS_BY_PROBLEM",
    "PHYSICAL_LANE_SHORTLIST_AGGRESSIVENESS_CHOICES",
    "STATIC_LANE_ROUTE_ALGEBRAIC",
    "STATIC_LANE_ROUTE_CHOICES",
    "STATIC_LANE_ROUTE_PHYSICAL_OPERATOR_TYPE",
    "StaticShortlistLaneSpec",
    "clamp_controller_cap_pair_for_lane_route",
    "normalize_physical_lane_shortlist_aggressiveness",
    "normalize_static_lane_route",
    "physical_lane_route_variant_id_for_problem",
    "resolve_static_shortlist_lane_spec",
]
