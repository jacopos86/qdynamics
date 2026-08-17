"""Actual-position insertion geometry for canonical Paper-I RA-ADAPT.

This module is the public owner of candidate-position enumeration and exact
ordered zero-angle insertion geometry.  The numerical kernels remain shared
with the characterized static-ADAPT implementation and are imported lazily to
avoid making this neutral boundary depend on the monolithic pipeline at import
time.

Every geometry call crosses :class:`InsertionGeometryRequest`, which fixes the
chart and records the actual insertion position.  Every returned receipt
revalidates those same fields against the delegated numerical payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from pipelines.static_adapt.ra_adapt.contracts import (
    CANDIDATE_REPRESENTATIONS,
    EXACT_ORDERED_INSERTION_CHART,
)


POSITION_DOMAIN_SCHEMA = "ra_adapt_actual_position_domain_v1"
CANDIDATE_POSITION_PLAN_SCHEMA = "ra_adapt_candidate_position_plans_v1"
COMMUTATION_REDUCED_POSITION_PLAN_SCHEMA = (
    "commutation_reduced_insertion_positions_v1"
)
COMMUTATION_REDUCED_DOMAIN_RECEIPT_SCHEMA = (
    "commutation_reduced_insertion_domain_receipt_v1"
)
APPEND_COMMUTATION_REDUCED_POLICY = "append_commutation_reduced"
APPEND_COMMUTATION_REDUCED_MODE = "append_commutation_reduced"
APPEND_ENDPOINT_POSITION_SCOPE = "append_endpoint_only_every_depth_v1"
EXACT_TERM_COMMUTATION_EQUIVALENCE = (
    "termwise_cross_component_commutation_earliest_representative_v1"
)
INSERTION_GEOMETRY_REQUEST_SCHEMA = "ra_adapt_insertion_geometry_request_v1"
INSERTION_GEOMETRY_RECEIPT_SCHEMA = "ra_adapt_insertion_geometry_receipt_v1"

FIRST_ORDER_GEOMETRY_KIND = "first_order_gradient_metric_v1"
JOINT_GEOMETRY_KIND = "joint_gradient_metric_hessian_v1"
FRESH_PHASE3_GEOMETRY_KIND = "fresh_phase3_joint_geometry_receipt_v1"


def _require_position(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    position = int(value)
    if position < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return position


def _require_representation(value: str) -> str:
    representation = str(value).strip()
    if representation not in CANDIDATE_REPRESENTATIONS:
        raise ValueError(
            "candidate_representation must be a canonical macro or "
            "single-Pauli-word representation."
        )
    return representation


def _require_exact_chart(value: str) -> str:
    chart = str(value).strip()
    if chart != EXACT_ORDERED_INSERTION_CHART:
        raise ValueError(
            "Canonical RA-ADAPT insertion geometry requires "
            f"{EXACT_ORDERED_INSERTION_CHART!r}; received {value!r}."
        )
    return chart


@dataclass(frozen=True, slots=True)
class InsertionGeometryRequest:
    """One candidate evaluated in the exact chart at its actual position."""

    candidate_term: Any
    insertion_position: int
    candidate_representation: str
    coordinate_chart: str = EXACT_ORDERED_INSERTION_CHART

    def __post_init__(self) -> None:
        if not hasattr(self.candidate_term, "polynomial"):
            raise TypeError(
                "candidate_term must expose the canonical AnsatzTerm "
                "polynomial interface."
            )
        object.__setattr__(
            self,
            "insertion_position",
            _require_position(
                self.insertion_position,
                name="insertion_position",
            ),
        )
        object.__setattr__(
            self,
            "candidate_representation",
            _require_representation(self.candidate_representation),
        )
        object.__setattr__(
            self,
            "coordinate_chart",
            _require_exact_chart(self.coordinate_chart),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": INSERTION_GEOMETRY_REQUEST_SCHEMA,
            "coordinate_chart": str(self.coordinate_chart),
            "candidate_representation": str(self.candidate_representation),
            "candidate_label": str(
                getattr(self.candidate_term, "label", "")
            ),
            "insertion_position": int(self.insertion_position),
        }


@dataclass(frozen=True, slots=True)
class InsertionGeometryReceipt:
    """Validated exact-geometry payload with explicit chart and position."""

    geometry_kind: str
    coordinate_chart: str
    candidate_representation: str
    insertion_position: int
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        kind = str(self.geometry_kind).strip()
        if kind not in {
            FIRST_ORDER_GEOMETRY_KIND,
            JOINT_GEOMETRY_KIND,
            FRESH_PHASE3_GEOMETRY_KIND,
        }:
            raise ValueError(f"Unknown insertion geometry kind: {kind!r}.")
        object.__setattr__(self, "geometry_kind", kind)
        object.__setattr__(
            self,
            "coordinate_chart",
            _require_exact_chart(self.coordinate_chart),
        )
        object.__setattr__(
            self,
            "candidate_representation",
            _require_representation(self.candidate_representation),
        )
        position = _require_position(
            self.insertion_position,
            name="insertion_position",
        )
        object.__setattr__(self, "insertion_position", position)
        payload = dict(self.payload)
        payload_chart = payload.get(
            "coordinate_chart",
            payload.get("coordinate_chart_id"),
        )
        if _require_exact_chart(str(payload_chart)) != self.coordinate_chart:
            raise ValueError(
                "Insertion geometry payload chart differs from its receipt."
            )
        payload_position = payload.get(
            "candidate_position_id",
            payload.get("position_id"),
        )
        if payload_position is None or int(payload_position) != position:
            raise ValueError(
                "Insertion geometry payload position differs from its receipt."
            )
        object.__setattr__(self, "payload", payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": INSERTION_GEOMETRY_RECEIPT_SCHEMA,
            "geometry_kind": str(self.geometry_kind),
            "coordinate_chart": str(self.coordinate_chart),
            "candidate_representation": str(
                self.candidate_representation
            ),
            "insertion_position": int(self.insertion_position),
            "payload": dict(self.payload),
        }


def exact_ordered_insertion_request(
    *,
    record: Any,
    insertion_position: int,
    representation_id: str,
) -> InsertionGeometryRequest:
    """Create the canonical geometry request from an adapter candidate.

    Candidate adapters traffic in lineage-bearing records, while the geometry
    kernel intentionally depends only on the live numerical term.  Keeping
    this projection here prevents either adapter from inventing a second chart
    or position convention.
    """

    candidate_term = getattr(record, "term", None)
    if candidate_term is None:
        raise TypeError("record must expose the live candidate term.")
    record_representation = getattr(record, "representation_id", None)
    if (
        record_representation is not None
        and str(record_representation) != str(representation_id)
    ):
        raise ValueError(
            "Candidate record and adapter representation identities differ."
        )
    return InsertionGeometryRequest(
        candidate_term=candidate_term,
        insertion_position=int(insertion_position),
        candidate_representation=str(representation_id),
    )


@dataclass(frozen=True, slots=True)
class ActualPositionDomain:
    """Resolved Phase-I position domain before candidate-specific reduction."""

    positions: tuple[int, ...]
    append_position: int
    probing_triggered: bool
    reason: str
    coordinate_chart: str = EXACT_ORDERED_INSERTION_CHART

    def __post_init__(self) -> None:
        append_position = _require_position(
            self.append_position,
            name="append_position",
        )
        positions = tuple(
            _require_position(value, name="position") for value in self.positions
        )
        if not positions:
            raise ValueError("An actual-position domain cannot be empty.")
        if len(set(positions)) != len(positions):
            raise ValueError(
                "An actual-position domain cannot contain duplicates."
            )
        if tuple(sorted(positions)) != positions:
            raise ValueError(
                "An actual-position domain must be in ascending order."
            )
        if any(position > append_position for position in positions):
            raise ValueError(
                "An actual-position domain cannot extend past append position."
            )
        if append_position not in positions:
            raise ValueError(
                "An actual-position domain must include the append position."
            )
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "append_position", append_position)
        object.__setattr__(
            self,
            "coordinate_chart",
            _require_exact_chart(self.coordinate_chart),
        )
        if not str(self.reason).strip():
            raise ValueError("Position-domain reason must be nonempty.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": POSITION_DOMAIN_SCHEMA,
            "coordinate_chart": str(self.coordinate_chart),
            "positions": [int(value) for value in self.positions],
            "append_position": int(self.append_position),
            "probing_triggered": bool(self.probing_triggered),
            "reason": str(self.reason),
        }


def append_commutation_reduced_position_domain(
    *,
    append_position: int,
    n_params: int,
) -> ActualPositionDomain:
    """Return the singleton endpoint domain for exact reduction.

    This is intentionally distinct from the historical append-only policy:
    callers must still pass the returned domain through
    :func:`enumerate_candidate_position_plans`, which invokes the same exact
    termwise cross-component reducer used by the always-open and plateau
    routes.
    """

    append = _require_position(
        append_position,
        name="append_position",
    )
    parameter_count = _require_position(n_params, name="n_params")
    if append != parameter_count:
        raise ValueError(
            "Commutation-reduced append insertion requires "
            "append_position == n_params."
        )
    return ActualPositionDomain(
        positions=(append,),
        append_position=append,
        probing_triggered=False,
        reason=APPEND_COMMUTATION_REDUCED_MODE,
    )


def append_commutation_reduced_receipt_header(
    *,
    append_position: int,
) -> dict[str, Any]:
    """Return the authenticated header for an endpoint reduction receipt."""

    append = _require_position(
        append_position,
        name="append_position",
    )
    return {
        "schema": COMMUTATION_REDUCED_DOMAIN_RECEIPT_SCHEMA,
        "policy": APPEND_COMMUTATION_REDUCED_POLICY,
        "domain_state": "closed",
        "domain_open": False,
        "effective_insertion_mode": APPEND_COMMUTATION_REDUCED_MODE,
        "append_position": append,
    }


def enumerate_actual_insertion_positions(
    *,
    insertion_mode: str,
    append_eval: Mapping[str, Any],
    append_position: int,
    n_params: int,
    active_window_indices: Sequence[int],
    stage_name: str,
    drop_plateau_hits: int,
    max_grad: float,
    eps_grad: float,
    finite_angle_fallback: bool,
    repeated_family_flat: bool,
    stage_controller_config: Any,
) -> ActualPositionDomain:
    """Resolve the actual Phase-I position domain with canonical telemetry."""

    append = _require_position(append_position, name="append_position")
    parameter_count = _require_position(n_params, name="n_params")
    if append != parameter_count:
        raise ValueError(
            "Canonical ordered insertion requires append_position == n_params."
        )
    for name, value in (("max_grad", max_grad), ("eps_grad", eps_grad)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite.")

    from pipelines.static_adapt.adapt_pipeline import (
        _phase1_position_probe_plan,
    )

    positions, triggered, reason = _phase1_position_probe_plan(
        insertion_mode=str(insertion_mode),
        append_eval=dict(append_eval),
        append_position=append,
        n_params=parameter_count,
        active_window_indices=[
            _require_position(value, name="active_window_index")
            for value in active_window_indices
        ],
        stage_name=str(stage_name),
        drop_plateau_hits=int(drop_plateau_hits),
        max_grad=float(max_grad),
        eps_grad=float(eps_grad),
        finite_angle_fallback=bool(finite_angle_fallback),
        repeated_family_flat=bool(repeated_family_flat),
        cfg=stage_controller_config,
    )
    return ActualPositionDomain(
        positions=tuple(int(value) for value in positions),
        append_position=append,
        probing_triggered=bool(triggered),
        reason=str(reason),
    )


def enumerate_candidate_position_plans(
    *,
    pool: Sequence[Any],
    candidate_indices: Sequence[int],
    selected_ops: Sequence[Any],
    domain: ActualPositionDomain,
) -> dict[int, dict[str, Any]]:
    """Return candidate-specific actual-position representatives.

    Each row preserves the characterized commutation-reduction payload and
    adds the exact chart plus explicit candidate identity used by RA-ADAPT.
    """

    if not isinstance(domain, ActualPositionDomain):
        raise TypeError("domain must be an ActualPositionDomain.")

    from pipelines.static_adapt.adapt_pipeline import (
        _candidate_insertion_position_plans,
    )

    plans = _candidate_insertion_position_plans(
        pool=pool,
        candidate_indices=[int(value) for value in candidate_indices],
        selected_ops=selected_ops,
        positions=domain.positions,
    )
    out: dict[int, dict[str, Any]] = {}
    for raw_index, raw_plan in plans.items():
        candidate_index = int(raw_index)
        if candidate_index < 0 or candidate_index >= len(pool):
            raise ValueError(
                "Candidate-position plan index is outside the supplied pool."
            )
        plan = dict(raw_plan)
        representatives = tuple(
            int(value) for value in plan.get("representative_positions", ())
        )
        if any(value not in domain.positions for value in representatives):
            raise RuntimeError(
                "Delegated candidate-position plan escaped its resolved domain."
            )
        out[candidate_index] = {
            **plan,
            "owner_schema": CANDIDATE_POSITION_PLAN_SCHEMA,
            "coordinate_chart": EXACT_ORDERED_INSERTION_CHART,
            "candidate_pool_index": candidate_index,
            "candidate_label": str(
                getattr(pool[candidate_index], "label", "")
            ),
            "append_position": int(domain.append_position),
        }
    return out


def _position_list(value: Any, *, owner: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{owner} must be a position list.")
    positions = [
        _require_position(position, name=f"{owner} position")
        for position in value
    ]
    if positions != sorted(set(positions)):
        raise ValueError(f"{owner} must be sorted and duplicate-free.")
    return positions


def _position_map(value: Any, *, owner: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} must be a position mapping.")
    normalized: dict[int, int] = {}
    for raw_key, raw_value in value.items():
        try:
            key = int(raw_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{owner} has a non-integer key.") from exc
        if str(raw_key).strip() != str(key) and not isinstance(
            raw_key, (int, np.integer)
        ):
            raise ValueError(f"{owner} has a non-canonical integer key.")
        key = _require_position(key, name=f"{owner} key")
        normalized[key] = _require_position(
            raw_value,
            name=f"{owner} value",
        )
    if len(normalized) != len(value):
        raise ValueError(f"{owner} repeats a normalized position key.")
    return normalized


def validate_commutation_reduced_insertion_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_policy: str | None = None,
    expected_requested_positions: Sequence[int] | None = None,
    scored_population: Mapping[str, Any] | None = None,
    expected_representative_pairs: Sequence[tuple[int, int]] | None = None,
    expected_phase_i_pairs: Sequence[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """Prove exact closure of a commutation-reduced insertion domain.

    The receipt must partition each candidate's complete requested position
    domain into disjoint exact-commutation classes, retain the earliest member
    of each class, and close all aggregate counts.  When a scored population
    is supplied, Phase I must equal the representative domain exactly and all
    later scored records must remain inside it.  A position-record Phase-0
    caller may instead bind the complete representative domain and its
    retained Phase-I subset separately; this preserves authentication of the
    original commutation-reduced domain without re-expanding the shortlist.
    """

    if not isinstance(receipt, Mapping):
        raise ValueError("Reduced insertion receipt must be a mapping.")
    policy = str(receipt.get("policy", "")).strip()
    if expected_policy is not None and policy != str(expected_policy):
        raise ValueError("Reduced insertion receipt policy drifted.")
    schema = str(receipt.get("schema", "")).strip()
    append_position: int | None = None
    if policy == "always_commutation_reduced":
        if schema != COMMUTATION_REDUCED_DOMAIN_RECEIPT_SCHEMA:
            raise ValueError("Always-reduced receipt schema drifted.")
        if (
            receipt.get("domain_open") is not True
            or receipt.get("domain_state") != "open"
            or receipt.get("effective_insertion_mode")
            != "full_commutation_reduced"
        ):
            raise ValueError("Always-reduced receipt domain state drifted.")
    elif policy in {
        "insertion_commutation_plateau_v1",
        "insertion_commutation_plateau_v2",
    }:
        expected_schema = {
            "insertion_commutation_plateau_v1": (
                "insertion_commutation_plateau_round_policy_v1"
            ),
            "insertion_commutation_plateau_v2": (
                "insertion_commutation_plateau_round_policy_v2"
            ),
        }[policy]
        if schema != expected_schema:
            raise ValueError("Plateau-reduced receipt schema drifted.")
        domain_open = receipt.get("domain_open")
        expected_mode = (
            "full_commutation_reduced" if domain_open is True else "append_only"
        )
        if (
            not isinstance(domain_open, bool)
            or receipt.get("domain_state")
            != ("open" if domain_open else "closed")
            or receipt.get("effective_insertion_mode") != expected_mode
        ):
            raise ValueError("Plateau-reduced receipt domain state drifted.")
    elif policy == APPEND_COMMUTATION_REDUCED_POLICY:
        if schema != COMMUTATION_REDUCED_DOMAIN_RECEIPT_SCHEMA:
            raise ValueError("Append-reduced receipt schema drifted.")
        if (
            receipt.get("domain_open") is not False
            or receipt.get("domain_state") != "closed"
            or receipt.get("effective_insertion_mode")
            != APPEND_COMMUTATION_REDUCED_MODE
        ):
            raise ValueError("Append-reduced receipt domain state drifted.")
        append_position = _require_position(
            receipt.get("append_position"),
            name="append_position",
        )
    else:
        raise ValueError("Reduced insertion receipt policy is unsupported.")

    requested = _position_list(
        receipt.get("requested_positions"),
        owner="Reduced receipt requested_positions",
    )
    if not requested:
        raise ValueError("Reduced receipt requested domain cannot be empty.")
    if (
        append_position is not None
        and requested != [append_position]
    ):
        raise ValueError(
            "Append-reduced receipt must request exactly the append endpoint."
        )
    if expected_requested_positions is not None:
        expected = _position_list(
            list(expected_requested_positions),
            owner="Expected requested_positions",
        )
        if requested != expected:
            raise ValueError("Reduced receipt requested domain drifted.")
    if int(receipt.get("requested_position_count", -1)) != len(requested):
        raise ValueError("Reduced receipt requested-position count drifted.")

    raw_plans = receipt.get("candidate_position_plans")
    if not isinstance(raw_plans, list) or not raw_plans:
        raise ValueError("Reduced receipt candidate plans are missing.")
    plans_by_pool_index: dict[int, dict[str, Any]] = {}
    representatives_by_pool_index: dict[int, set[int]] = {}
    retained_count = 0
    collapsed_count = 0
    for raw_plan in raw_plans:
        if not isinstance(raw_plan, Mapping):
            raise ValueError("Reduced insertion plan must be a mapping.")
        plan = dict(raw_plan)
        if plan.get("schema") != COMMUTATION_REDUCED_POSITION_PLAN_SCHEMA:
            raise ValueError("Reduced insertion plan schema drifted.")
        pool_index = _require_position(
            plan.get("candidate_pool_index"),
            name="candidate_pool_index",
        )
        if pool_index in plans_by_pool_index:
            raise ValueError("Reduced insertion plans repeat a candidate.")
        plan_requested = _position_list(
            plan.get("requested_positions"),
            owner=f"Candidate {pool_index} requested_positions",
        )
        representatives = _position_list(
            plan.get("representative_positions"),
            owner=f"Candidate {pool_index} representative_positions",
        )
        if plan_requested != requested or not representatives:
            raise ValueError(
                "Reduced insertion plan does not cover the requested domain."
            )
        if not set(representatives).issubset(requested):
            raise ValueError(
                "Reduced insertion representative escaped the requested domain."
            )
        representative_by_position = _position_map(
            plan.get("representative_by_position"),
            owner=f"Candidate {pool_index} representative_by_position",
        )
        raw_members = plan.get("members_by_representative")
        if not isinstance(raw_members, Mapping):
            raise ValueError(
                "Reduced insertion members_by_representative is missing."
            )
        members_by_representative: dict[int, list[int]] = {}
        for raw_representative, raw_class_members in raw_members.items():
            try:
                representative = int(raw_representative)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Reduced insertion class has a non-integer representative."
                ) from exc
            representative = _require_position(
                representative,
                name="class representative",
            )
            if representative in members_by_representative:
                raise ValueError(
                    "Reduced insertion classes repeat a representative."
                )
            members_by_representative[representative] = _position_list(
                raw_class_members,
                owner=(
                    f"Candidate {pool_index} class {representative} members"
                ),
            )
        if set(members_by_representative) != set(representatives):
            raise ValueError(
                "Reduced insertion class keys do not match representatives."
            )
        flattened = [
            member
            for representative in representatives
            for member in members_by_representative[representative]
        ]
        if sorted(flattened) != requested or len(flattened) != len(
            set(flattened)
        ):
            raise ValueError(
                "Reduced insertion classes do not form an exact disjoint "
                "partition."
            )
        for representative, members in members_by_representative.items():
            if not members or representative != min(members):
                raise ValueError(
                    "Reduced insertion must retain the earliest class member."
                )
        expected_map = {
            member: representative
            for representative, members in members_by_representative.items()
            for member in members
        }
        if representative_by_position != expected_map:
            raise ValueError(
                "Reduced insertion representative map does not close."
            )
        crossings = plan.get("commuting_crossings")
        if (
            not isinstance(crossings, list)
            or len(crossings) != max(requested)
            or any(not isinstance(value, bool) for value in crossings)
        ):
            raise ValueError(
                "Reduced insertion commuting-crossing certificate drifted."
            )
        class_start_by_position: dict[int, int] = {0: 0}
        class_start = 0
        for crossing_index, crossing in enumerate(crossings):
            if not crossing:
                class_start = crossing_index + 1
            class_start_by_position[crossing_index + 1] = class_start
        requested_by_class: dict[int, list[int]] = {}
        for position in requested:
            requested_by_class.setdefault(
                class_start_by_position[position],
                [],
            ).append(position)
        expected_members_by_representative = {
            min(members): members for members in requested_by_class.values()
        }
        if members_by_representative != expected_members_by_representative:
            raise ValueError(
                "Reduced insertion class partition disagrees with its "
                "commuting-crossing certificate."
            )
        plan_collapsed = len(requested) - len(representatives)
        if int(plan.get("collapsed_position_count", -1)) != plan_collapsed:
            raise ValueError("Reduced insertion collapsed count drifted.")
        plans_by_pool_index[pool_index] = plan
        representatives_by_pool_index[pool_index] = set(representatives)
        retained_count += len(representatives)
        collapsed_count += plan_collapsed

    candidate_count = len(plans_by_pool_index)
    if (
        int(receipt.get("candidate_count", -1)) != candidate_count
        or int(receipt.get("retained_representative_count", -1))
        != retained_count
        or int(receipt.get("collapsed_position_count", -1))
        != collapsed_count
        or retained_count + collapsed_count
        != candidate_count * len(requested)
    ):
        raise ValueError("Reduced insertion aggregate count closure failed.")
    if append_position is not None and (
        retained_count != candidate_count or collapsed_count != 0
    ):
        raise ValueError(
            "Append-reduced receipt must retain one endpoint representative "
            "and collapse zero positions per candidate."
        )

    raw_retained = receipt.get("retained_representatives")
    if not isinstance(raw_retained, list) or len(raw_retained) != candidate_count:
        raise ValueError("Reduced insertion retained count rows drifted.")
    retained_by_pool_index: dict[int, list[int]] = {}
    for raw_row in raw_retained:
        if not isinstance(raw_row, Mapping):
            raise ValueError(
                "Reduced insertion retained representative row is malformed."
            )
        pool_index = _require_position(
            raw_row.get("candidate_pool_index"),
            name="retained candidate_pool_index",
        )
        if pool_index in retained_by_pool_index:
            raise ValueError(
                "Reduced insertion retained rows repeat a candidate."
            )
        retained_by_pool_index[pool_index] = _position_list(
            raw_row.get("positions"),
            owner=f"Candidate {pool_index} retained positions",
        )
    if {
        pool_index: set(positions)
        for pool_index, positions in retained_by_pool_index.items()
    } != representatives_by_pool_index:
        raise ValueError(
            "Reduced insertion retained representatives do not close."
        )

    if scored_population is not None:
        if not isinstance(scored_population, Mapping):
            raise ValueError("Reduced insertion scored population is malformed.")
        phases = scored_population.get("phases")
        if not isinstance(phases, list) or not phases:
            raise ValueError("Reduced insertion scored phases are missing.")
        phase_i_pairs: set[tuple[int, int]] | None = None
        expected_pairs = {
            (pool_index, position)
            for pool_index, positions in representatives_by_pool_index.items()
            for position in positions
        }
        if expected_representative_pairs is not None:
            normalized_representative_pairs = [
                (
                    _require_position(
                        pair[0],
                        name="expected representative pool_index",
                    ),
                    _require_position(
                        pair[1],
                        name="expected representative insertion_position",
                    ),
                )
                for pair in expected_representative_pairs
            ]
            if (
                len(normalized_representative_pairs)
                != len(set(normalized_representative_pairs))
                or set(normalized_representative_pairs) != expected_pairs
            ):
                raise ValueError(
                    "Reduced insertion representative domain disagrees with "
                    "the authenticated Phase-0 population."
                )
        if expected_phase_i_pairs is None:
            phase_i_expected_pairs = expected_pairs
        else:
            normalized_phase_i_pairs = [
                (
                    _require_position(
                        pair[0],
                        name="expected Phase-I pool_index",
                    ),
                    _require_position(
                        pair[1],
                        name="expected Phase-I insertion_position",
                    ),
                )
                for pair in expected_phase_i_pairs
            ]
            if (
                len(normalized_phase_i_pairs)
                != len(set(normalized_phase_i_pairs))
                or not set(normalized_phase_i_pairs).issubset(expected_pairs)
            ):
                raise ValueError(
                    "Reduced insertion Phase-I domain is not an exact retained "
                    "subset of its representatives."
                )
            phase_i_expected_pairs = set(normalized_phase_i_pairs)
        for phase in phases:
            if not isinstance(phase, Mapping):
                raise ValueError(
                    "Reduced insertion scored phase is malformed."
                )
            records = phase.get("records")
            if not isinstance(records, list):
                raise ValueError(
                    "Reduced insertion scored records are malformed."
                )
            pairs: list[tuple[int, int]] = []
            for record in records:
                if not isinstance(record, Mapping):
                    raise ValueError(
                        "Reduced insertion scored record is malformed."
                    )
                pair = (
                    _require_position(
                        record.get("pool_index"),
                        name="scored pool_index",
                    ),
                    _require_position(
                        record.get("insertion_position"),
                        name="scored insertion_position",
                    ),
                )
                pairs.append(pair)
            if not set(pairs).issubset(phase_i_expected_pairs):
                raise ValueError(
                    "Reduced insertion scored population escaped its "
                    "authenticated Phase-I domain."
                )
            if phase.get("phase") == "phase_i":
                if len(pairs) != len(set(pairs)):
                    raise ValueError(
                        "Reduced insertion Phase-I scored population repeats "
                        "a representative."
                    )
                phase_i_pairs = set(pairs)
        if phase_i_pairs != phase_i_expected_pairs:
            raise ValueError(
                "Reduced insertion Phase-I scored domain does not equal its "
                "authenticated input domain."
            )

    return dict(receipt)


def splice_candidate_at_position(
    *,
    ops: Sequence[Any],
    theta: np.ndarray | Sequence[float],
    candidate_term: Any,
    insertion_position: int,
    initial_coordinate: float = 0.0,
) -> tuple[list[Any], np.ndarray]:
    """Materialize one candidate at an explicit logical position.

    Canonical geometry always calls this with ``initial_coordinate=0``.  The
    argument remains explicit because the characterized helper is also the
    admission-boundary primitive and callers may use it to validate that a
    nonzero proposal is reset before accepted refit.
    """

    position = _require_position(
        insertion_position,
        name="insertion_position",
    )
    initial = float(initial_coordinate)
    if not math.isfinite(initial):
        raise ValueError("initial_coordinate must be finite.")

    from pipelines.static_adapt.adapt_pipeline import (
        _splice_candidate_at_position,
    )

    return _splice_candidate_at_position(
        ops=list(ops),
        theta=np.asarray(theta, dtype=float),
        op=candidate_term,
        position_id=position,
        init_theta=initial,
    )


def prepare_exact_insertion_first_order_context(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray | Sequence[float],
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    hpsi_state: np.ndarray,
    pauli_action_cache: dict[str, Any] | None = None,
    state_consistency_tolerance: float = 1.0e-10,
) -> Any:
    """Prepare one reusable accepted-state context for position probes."""

    from pipelines.scaffold.hh_continuation_scoring import (
        _prepare_exact_insertion_first_order_context,
    )

    return _prepare_exact_insertion_first_order_context(
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        hpsi_state=np.asarray(hpsi_state, dtype=complex),
        pauli_action_cache=pauli_action_cache,
        state_consistency_tolerance=float(state_consistency_tolerance),
    )


def evaluate_exact_insertion_first_order(
    *,
    context: Any,
    request: InsertionGeometryRequest,
    candidate_compiled: Any | None = None,
    pauli_action_cache: dict[str, Any] | None = None,
) -> InsertionGeometryReceipt:
    """Evaluate exact gradient and Fubini--Study diagonal at one position."""

    if not isinstance(request, InsertionGeometryRequest):
        raise TypeError("request must be an InsertionGeometryRequest.")

    from pipelines.scaffold.hh_continuation_scoring import (
        _exact_insertion_first_order_candidate_geometry,
    )

    delegated = dict(
        _exact_insertion_first_order_candidate_geometry(
            context=context,
            candidate_term=request.candidate_term,
            position_id=int(request.insertion_position),
            candidate_compiled=candidate_compiled,
            pauli_action_cache=pauli_action_cache,
        )
    )
    payload = {
        **delegated,
        "coordinate_chart": EXACT_ORDERED_INSERTION_CHART,
        "candidate_position_id": int(request.insertion_position),
    }
    return InsertionGeometryReceipt(
        geometry_kind=FIRST_ORDER_GEOMETRY_KIND,
        coordinate_chart=EXACT_ORDERED_INSERTION_CHART,
        candidate_representation=request.candidate_representation,
        insertion_position=int(request.insertion_position),
        payload=payload,
    )


def prepare_exact_insertion_joint_context(
    *,
    selected_ops: Sequence[Any],
    theta: np.ndarray | Sequence[float],
    psi_ref: np.ndarray,
    psi_state: np.ndarray,
    active_indices: Sequence[int],
    h_compiled: Any,
    measure_old_old_geometry: bool = True,
) -> Any:
    """Prepare the exact shared old-coordinate blocks for joint geometry."""

    from pipelines.scaffold.hh_continuation_scoring import (
        _selector_scaffold_context,
    )

    return _selector_scaffold_context(
        selected_ops=selected_ops,
        theta=np.asarray(theta, dtype=float),
        psi_ref=np.asarray(psi_ref, dtype=complex),
        psi_state=np.asarray(psi_state, dtype=complex),
        active_indices=tuple(
            _require_position(value, name="active_index")
            for value in active_indices
        ),
        h_compiled=h_compiled,
        measure_old_old_geometry=bool(measure_old_old_geometry),
    )


def evaluate_exact_insertion_joint_geometry(
    *,
    context: Any,
    request: InsertionGeometryRequest,
    h_compiled: Any,
    pauli_action_cache: dict[str, Any] | None = None,
    state_consistency_tolerance: float = 1.0e-10,
    old_old_geometry_prior: Any | None = None,
    acquisition_mode: str = "default_full_v1",
) -> InsertionGeometryReceipt:
    """Evaluate exact active-plus-candidate Gram and Hessian blocks."""

    if not isinstance(request, InsertionGeometryRequest):
        raise TypeError("request must be an InsertionGeometryRequest.")

    from pipelines.scaffold.hh_continuation_scoring import (
        _exact_insertion_joint_geometry_payload,
    )

    payload = dict(
        _exact_insertion_joint_geometry_payload(
            scaffold_context=context,
            candidate_term=request.candidate_term,
            position_id=int(request.insertion_position),
            h_compiled=h_compiled,
            pauli_action_cache=pauli_action_cache,
            state_consistency_tolerance=float(
                state_consistency_tolerance
            ),
            old_old_geometry_prior=old_old_geometry_prior,
            acquisition_mode=str(acquisition_mode),
        )
    )
    return InsertionGeometryReceipt(
        geometry_kind=JOINT_GEOMETRY_KIND,
        coordinate_chart=EXACT_ORDERED_INSERTION_CHART,
        candidate_representation=request.candidate_representation,
        insertion_position=int(request.insertion_position),
        payload=payload,
    )


def promote_fresh_phase3_joint_geometry(
    *,
    acquired_payload: Mapping[str, Any],
    context: Any,
    request: InsertionGeometryRequest,
    h_compiled: Any,
    state_consistency_tolerance: float = 1.0e-10,
) -> InsertionGeometryReceipt:
    """Promote already acquired Phase-III blocks into the canonical receipt."""

    if not isinstance(request, InsertionGeometryRequest):
        raise TypeError("request must be an InsertionGeometryRequest.")

    from pipelines.scaffold.hh_continuation_scoring import (
        _promote_fresh_phase3_joint_geometry_receipt,
    )

    payload = dict(
        _promote_fresh_phase3_joint_geometry_receipt(
            acquired_payload=dict(acquired_payload),
            scaffold_context=context,
            candidate_term=request.candidate_term,
            position_id=int(request.insertion_position),
            h_compiled=h_compiled,
            state_consistency_tolerance=float(
                state_consistency_tolerance
            ),
        )
    )
    return InsertionGeometryReceipt(
        geometry_kind=FRESH_PHASE3_GEOMETRY_KIND,
        coordinate_chart=EXACT_ORDERED_INSERTION_CHART,
        candidate_representation=request.candidate_representation,
        insertion_position=int(request.insertion_position),
        payload=payload,
    )


__all__ = [
    "APPEND_COMMUTATION_REDUCED_MODE",
    "APPEND_COMMUTATION_REDUCED_POLICY",
    "APPEND_ENDPOINT_POSITION_SCOPE",
    "ActualPositionDomain",
    "CANDIDATE_POSITION_PLAN_SCHEMA",
    "COMMUTATION_REDUCED_DOMAIN_RECEIPT_SCHEMA",
    "COMMUTATION_REDUCED_POSITION_PLAN_SCHEMA",
    "EXACT_TERM_COMMUTATION_EQUIVALENCE",
    "FIRST_ORDER_GEOMETRY_KIND",
    "FRESH_PHASE3_GEOMETRY_KIND",
    "INSERTION_GEOMETRY_RECEIPT_SCHEMA",
    "INSERTION_GEOMETRY_REQUEST_SCHEMA",
    "InsertionGeometryReceipt",
    "InsertionGeometryRequest",
    "JOINT_GEOMETRY_KIND",
    "POSITION_DOMAIN_SCHEMA",
    "append_commutation_reduced_position_domain",
    "append_commutation_reduced_receipt_header",
    "enumerate_actual_insertion_positions",
    "enumerate_candidate_position_plans",
    "exact_ordered_insertion_request",
    "evaluate_exact_insertion_first_order",
    "evaluate_exact_insertion_joint_geometry",
    "prepare_exact_insertion_first_order_context",
    "prepare_exact_insertion_joint_context",
    "promote_fresh_phase3_joint_geometry",
    "splice_candidate_at_position",
    "validate_commutation_reduced_insertion_receipt",
]
