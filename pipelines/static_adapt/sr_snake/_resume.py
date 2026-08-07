"""Strict accepted-state hydration for the typed Paper-I SR-SNAKE route.

The direct route publishes an accepted-prefix ``current.json`` plus a
hash-linked estimator-ledger sidecar.  This module is the only canonical
reader for that format.  It deliberately does not call the legacy resume
scaffold, infer a route, repair incomplete payloads, or fall back to another
checkpoint representation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from pipelines.contracts.problem import ResolvedProblemContext
from pipelines.scaffold.handoff_state_bundle import build_statevector_manifest
from pipelines.scaffold.hh_continuation_types import PhaseControllerSnapshot
from pipelines.static_adapt.estimator_call_ledger import (
    EstimatorCallLedger,
    S_ALG_COMPONENTS,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    AppendCommutationReducedInsertion,
    ResolvedProblemReceipt,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1,
    RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
)
from pipelines.static_adapt.sr_snake_route_profile import (
    INSERTION_COMMUTATION_PLATEAU_CALIBRATION_STATUS,
    INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS,
    INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD,
)


_CHECKPOINT_SCHEMA = "static_adapt_current_checkpoint_v1"
_SIGNED_PREFIX_SCHEMA = "paper_i_signed_active_prefix_checkpoint_v1"
_LEDGER_POINTER_SCHEMA = (
    "paper_i_estimator_call_ledger_checkpoint_pointer_v2"
)
_LEDGER_SIDECAR_SCHEMA = (
    "paper_i_estimator_call_ledger_checkpoint_sidecar_v2"
)
_ACCOUNTING_SCHEMA = "paper_i_current_s_alg_accounting_v2"
_WORK_SCHEMA = "paper_i_executed_logical_scalar_estimator_work_v2"
_PREFIX_RECEIPT_SCHEMA = (
    "paper_i_active_prefix_estimator_ledger_receipt_v2"
)
_ROUTE_CONTRACT_SCHEMA = "sr_snake_route_profile_contract_v1"
_RA_ROUTE_CONTRACT_SCHEMAS = frozenset(
    {
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V1,
        RA_ADAPT_ROUTE_CONTRACT_SCHEMA_V2,
    }
)
_ROUTE_FAMILIES = frozenset(
    {
        "singleton_response_snake",
        "greedy_batch_response_snake",
        "combinatorial_batch_response_snake",
        "ra_adapt",
    }
)
_PROJECTIVE_FINGERPRINT_PREFIX = "projective_state_v1:"


class CanonicalResumeError(ValueError):
    """The requested artifact is not a complete canonical accepted state."""


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    options: dict[str, Any] = {
        "sort_keys": True,
        "ensure_ascii": True,
        "allow_nan": False,
    }
    if pretty:
        options["indent"] = 2
    else:
        options["separators"] = (",", ":")
    return json.dumps(value, **options).encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise CanonicalResumeError(
                f"Canonical accepted checkpoint repeats JSON field {key!r}."
            )
        payload[key] = value
    return payload


def _reject_nonfinite_json(value: str) -> None:
    raise CanonicalResumeError(
        "Canonical accepted checkpoint contains a non-finite JSON number: "
        f"{value}."
    )


def _load_json_object(path: Path, *, owner: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise CanonicalResumeError(
            f"Cannot read {owner} as UTF-8 JSON: {path}."
        ) from exc
    try:
        payload = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except CanonicalResumeError:
        raise
    except json.JSONDecodeError as exc:
        raise CanonicalResumeError(
            f"{owner} is not valid JSON: {path}."
        ) from exc
    if not isinstance(payload, dict):
        raise CanonicalResumeError(f"{owner} must be a JSON object.")
    return payload


def _mapping(value: Any, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalResumeError(f"{owner} must be an object.")
    return dict(value)


def _sequence(value: Any, *, owner: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise CanonicalResumeError(f"{owner} must be an array.")
    return list(value)


def _text(value: Any, *, owner: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CanonicalResumeError(f"{owner} must be nonempty text.")
    return value.strip()


def _integer(
    value: Any,
    *,
    owner: str,
    minimum: int | None = None,
) -> int:
    if isinstance(value, bool):
        raise CanonicalResumeError(f"{owner} must be an integer.")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalResumeError(f"{owner} must be an integer.") from exc
    if converted != value:
        raise CanonicalResumeError(f"{owner} must be an exact integer.")
    if minimum is not None and converted < minimum:
        raise CanonicalResumeError(
            f"{owner} must be at least {minimum}; got {converted}."
        )
    return converted


def _finite(value: Any, *, owner: str) -> float:
    if isinstance(value, bool):
        raise CanonicalResumeError(f"{owner} must be a finite number.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CanonicalResumeError(
            f"{owner} must be a finite number."
        ) from exc
    if not math.isfinite(converted):
        raise CanonicalResumeError(f"{owner} must be finite.")
    return converted


def _validate_maturity_controller_snapshot(
    value: Any,
    *,
    owner: str,
) -> dict[str, Any]:
    """Require the exact typed snapshot reconstructed by resume hydration."""

    payload = _mapping(value, owner=owner)
    try:
        snapshot = PhaseControllerSnapshot(**payload)
    except (TypeError, ValueError) as exc:
        raise CanonicalResumeError(
            f"{owner} cannot reconstruct PhaseControllerSnapshot."
        ) from exc
    for field_name in ("step_index", "depth_local", "depth_left"):
        _integer(
            getattr(snapshot, field_name),
            owner=f"{owner}.{field_name}",
            minimum=0,
        )
    for field_name in (
        "runway_ratio",
        "early_coordinate",
        "late_coordinate",
        "frontier_ratio",
    ):
        _finite(
            getattr(snapshot, field_name),
            owner=f"{owner}.{field_name}",
        )
    mapping_fields = (
        "phase_thresholds",
        "phase_caps",
        "phase_shots",
        "phase_uncertainty",
        "phase_live",
        "phase_null_reasons",
        "phase_null_streaks",
        "phase_caps_scheduled",
        "phase_shots_maturity_floor",
        "phase_shots_scheduled",
        "phase_shots_snr",
        "phase_shots_effective",
        "phase_shot_uplift",
        "phase_shot_fraction",
        "phase_signal",
        "phase_signal_floor",
    )
    for field_name in mapping_fields:
        _mapping(
            getattr(snapshot, field_name),
            owner=f"{owner}.{field_name}",
        )
    phase_live = _mapping(snapshot.phase_live, owner=f"{owner}.phase_live")
    if not phase_live or any(
        not isinstance(flag, bool) for flag in phase_live.values()
    ):
        raise CanonicalResumeError(
            f"{owner}.phase_live must contain Boolean phase flags."
        )
    _text(snapshot.snapshot_version, owner=f"{owner}.snapshot_version")
    return payload


def _sha256(value: Any, *, owner: str) -> str:
    digest = _text(value, owner=owner).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise CanonicalResumeError(f"{owner} must be a lowercase SHA-256.")
    return digest


def _integer_key_mapping(
    value: Any,
    *,
    owner: str,
) -> dict[int, Any]:
    """Parse a JSON object whose canonical keys are nonnegative integers."""

    payload = _mapping(value, owner=owner)
    parsed: dict[int, Any] = {}
    for raw_key, child in payload.items():
        if isinstance(raw_key, bool):
            raise CanonicalResumeError(
                f"{owner} keys must be canonical nonnegative integers."
            )
        if isinstance(raw_key, int):
            key = raw_key
        elif isinstance(raw_key, str):
            try:
                key = int(raw_key)
            except (TypeError, ValueError, OverflowError) as exc:
                raise CanonicalResumeError(
                    f"{owner} keys must be canonical nonnegative integers."
                ) from exc
            if raw_key != str(key):
                raise CanonicalResumeError(
                    f"{owner} keys must be canonical nonnegative integers."
                )
        else:
            raise CanonicalResumeError(
                f"{owner} keys must be canonical nonnegative integers."
            )
        if key < 0 or key in parsed:
            raise CanonicalResumeError(
                f"{owner} keys must be unique nonnegative integers."
            )
        parsed[key] = child
    return parsed


def _resolve_authenticated_insertion_policy(
    route_contract: Mapping[str, Any],
) -> str:
    """Resolve only executable insertion identities accepted by hydration."""

    semantic_invariants = _mapping(
        route_contract.get("semantic_invariants"),
        owner="route semantic invariants",
    )
    execution_settings = _mapping(
        route_contract.get("execution_settings"),
        owner="route execution settings",
    )
    insertion_policy_raw = semantic_invariants.get(
        "canonical_insertion_policy"
    )
    insertion_mode_raw = execution_settings.get("adapt_insertion_mode")
    insertion_mode = (
        None
        if insertion_mode_raw is None
        else _text(
            insertion_mode_raw,
            owner="route insertion mode",
        )
    )
    if insertion_mode == "full":
        raise CanonicalResumeError(
            "Authenticated resume rejects the retired unreduced full "
            "insertion mode."
        )
    if insertion_policy_raw is None:
        if insertion_mode is None:
            raise CanonicalResumeError(
                "Authenticated route does not declare an insertion mode."
            )
        insertion_policy_raw = {
            "append_only": "append_only",
            AppendCommutationReducedInsertion.runtime_mode: (
                AppendCommutationReducedInsertion.kind
            ),
            "full_commutation_reduced": "always_commutation_reduced",
            "insertion_commutation_plateau_v1": "plateau_commutation",
            "insertion_commutation_plateau_v2": "plateau_commutation",
        }.get(insertion_mode, insertion_mode)
    insertion_policy = _text(
        insertion_policy_raw,
        owner="route canonical insertion policy",
    )
    if insertion_policy == "full_commutation":
        raise CanonicalResumeError(
            "Authenticated resume rejects the retired unreduced "
            "full_commutation insertion policy."
        )
    if insertion_policy not in {
        "append_only",
        AppendCommutationReducedInsertion.kind,
        "always_commutation_reduced",
        "plateau_commutation",
    }:
        raise CanonicalResumeError(
            "Authenticated route insertion policy is unsupported."
        )
    expected_modes = {
        "append_only": frozenset({"append_only"}),
        AppendCommutationReducedInsertion.kind: frozenset(
            {AppendCommutationReducedInsertion.runtime_mode}
        ),
        "always_commutation_reduced": frozenset(
            {"full_commutation_reduced"}
        ),
        "plateau_commutation": frozenset(
            {
                "insertion_commutation_plateau_v1",
                "insertion_commutation_plateau_v2",
            }
        ),
    }[insertion_policy]
    if insertion_mode is None or insertion_mode not in expected_modes:
        raise CanonicalResumeError(
            "Authenticated route insertion policy and execution mode "
            "disagree."
        )
    if insertion_policy == AppendCommutationReducedInsertion.kind and (
        semantic_invariants.get("insertion_position_scope")
        != AppendCommutationReducedInsertion.position_scope
        or semantic_invariants.get("insertion_equivalence_policy")
        != AppendCommutationReducedInsertion.equivalence_policy
    ):
        raise CanonicalResumeError(
            "Authenticated append-reduced route does not bind the endpoint "
            "scope and exact commutation equivalence."
        )
    return insertion_policy


def _validate_scored_insertion_population(
    value: Any,
    *,
    owner: str,
    append_position: int,
    representatives_by_pool_index: Mapping[int, tuple[int, ...]],
) -> None:
    """Bind every scored position to an authenticated representative."""

    payload = _mapping(value, owner=owner)
    phase_order = ("phase_i", "phase_ii", "phase_iii")
    if (
        payload.get("schema")
        != "paper_i_scored_insertion_position_population_v1"
        or payload.get("coordinate_chart")
        != "exact_ordered_insertion_zero_angle_v1"
        or _integer(
            payload.get("append_position"),
            owner=f"{owner}.append_position",
            minimum=0,
        )
        != append_position
        or tuple(
            _text(phase, owner=f"{owner}.phase_order")
            for phase in _sequence(
                payload.get("phase_order"),
                owner=f"{owner}.phase_order",
            )
        )
        != phase_order
    ):
        raise CanonicalResumeError(
            f"{owner} does not identify the exact ordered insertion chart."
        )
    phases = [
        _mapping(phase, owner=f"{owner}.phases[{index}]")
        for index, phase in enumerate(
            _sequence(payload.get("phases"), owner=f"{owner}.phases")
        )
    ]
    if len(phases) != len(phase_order):
        raise CanonicalResumeError(
            f"{owner} must contain exactly the three scored phases."
        )

    all_records: list[dict[str, Any]] = []
    phase_i_pairs: set[tuple[int, int]] | None = None
    for expected_phase, phase in zip(
        phase_order,
        phases,
        strict=True,
    ):
        if phase.get("phase") != expected_phase:
            raise CanonicalResumeError(
                f"{owner} scored phases are not in canonical order."
            )
        records = [
            _mapping(
                record,
                owner=f"{owner}.{expected_phase}.records[{index}]",
            )
            for index, record in enumerate(
                _sequence(
                    phase.get("records"),
                    owner=f"{owner}.{expected_phase}.records",
                )
            )
        ]
        if (
            not records
            or _integer(
                phase.get("population_count"),
                owner=f"{owner}.{expected_phase}.population_count",
                minimum=1,
            )
            != len(records)
            or _sha256(
                phase.get("ordered_population_sha256"),
                owner=(
                    f"{owner}.{expected_phase}."
                    "ordered_population_sha256"
                ),
            )
            != _digest_json(records)
        ):
            raise CanonicalResumeError(
                f"{owner} {expected_phase} scored population is incomplete."
            )
        observed_pairs: set[tuple[int, int]] = set()
        observed_identities: set[tuple[str, str]] = set()
        for record in records:
            domain_record_id = _text(
                record.get("domain_record_id"),
                owner=f"{owner} scored domain record id",
            )
            generator_id = _text(
                record.get("generator_id"),
                owner=f"{owner} scored generator id",
            )
            pool_index = _integer(
                record.get("pool_index"),
                owner=f"{owner} scored pool index",
                minimum=0,
            )
            _text(
                record.get("pool_label"),
                owner=f"{owner} scored pool label",
            )
            position = _integer(
                record.get("insertion_position"),
                owner=f"{owner} scored insertion position",
                minimum=0,
            )
            representatives = representatives_by_pool_index.get(pool_index)
            expected_class = (
                "interior" if position < append_position else "append"
            )
            if (
                representatives is None
                or position not in representatives
                or position > append_position
                or record.get("position_class") != expected_class
                or (
                    expected_phase == "phase_i"
                    and (pool_index, position) in observed_pairs
                )
                or (domain_record_id, generator_id) in observed_identities
            ):
                raise CanonicalResumeError(
                    f"{owner} contains a scored position outside its "
                    "authenticated representatives."
                )
            observed_pairs.add((pool_index, position))
            observed_identities.add((domain_record_id, generator_id))
        if expected_phase == "phase_i":
            phase_i_pairs = observed_pairs
        all_records.extend(records)

    expected_phase_i_pairs = {
        (pool_index, position)
        for pool_index, positions in representatives_by_pool_index.items()
        for position in positions
    }
    if phase_i_pairs != expected_phase_i_pairs:
        raise CanonicalResumeError(
            f"{owner} Phase-I scored positions do not close over every "
            "authenticated representative."
        )
    interior_count = sum(
        record["position_class"] == "interior"
        for record in all_records
    )
    append_count = len(all_records) - interior_count
    unsigned = dict(payload)
    supplied_sha = _sha256(
        unsigned.pop("sha256", None),
        owner=f"{owner}.sha256",
    )
    if (
        _integer(
            payload.get("scored_record_count"),
            owner=f"{owner}.scored_record_count",
            minimum=1,
        )
        != len(all_records)
        or _integer(
            payload.get("interior_scored_count"),
            owner=f"{owner}.interior_scored_count",
            minimum=0,
        )
        != interior_count
        or _integer(
            payload.get("append_scored_count"),
            owner=f"{owner}.append_scored_count",
            minimum=0,
        )
        != append_count
        or supplied_sha != _digest_json(unsigned)
    ):
        raise CanonicalResumeError(
            f"{owner} scored population counts or digest disagree."
        )


def _validate_append_only_scored_population(
    value: Any,
    *,
    owner: str,
    append_position: int,
) -> None:
    """Require every scored record to remain at the append endpoint."""

    payload = _mapping(value, owner=owner)
    phase_order = ("phase_i", "phase_ii", "phase_iii")
    if (
        payload.get("schema")
        != "paper_i_scored_insertion_position_population_v1"
        or payload.get("coordinate_chart")
        != "exact_ordered_insertion_zero_angle_v1"
        or _integer(
            payload.get("append_position"),
            owner=f"{owner}.append_position",
            minimum=0,
        )
        != append_position
        or tuple(
            _text(phase, owner=f"{owner}.phase_order")
            for phase in _sequence(
                payload.get("phase_order"),
                owner=f"{owner}.phase_order",
            )
        )
        != phase_order
    ):
        raise CanonicalResumeError(
            f"{owner} append-only scoring identity is incomplete."
        )
    phases = [
        _mapping(phase, owner=f"{owner}.phases[{index}]")
        for index, phase in enumerate(
            _sequence(payload.get("phases"), owner=f"{owner}.phases")
        )
    ]
    if len(phases) != len(phase_order):
        raise CanonicalResumeError(
            f"{owner} append-only scoring must contain three phases."
        )
    all_records: list[dict[str, Any]] = []
    for expected_phase, phase in zip(
        phase_order,
        phases,
        strict=True,
    ):
        records = [
            _mapping(
                record,
                owner=f"{owner}.{expected_phase}.records[{index}]",
            )
            for index, record in enumerate(
                _sequence(
                    phase.get("records"),
                    owner=f"{owner}.{expected_phase}.records",
                )
            )
        ]
        if (
            phase.get("phase") != expected_phase
            or not records
            or _integer(
                phase.get("population_count"),
                owner=f"{owner}.{expected_phase}.population_count",
                minimum=1,
            )
            != len(records)
            or _sha256(
                phase.get("ordered_population_sha256"),
                owner=(
                    f"{owner}.{expected_phase}."
                    "ordered_population_sha256"
                ),
            )
            != _digest_json(records)
        ):
            raise CanonicalResumeError(
                f"{owner} append-only {expected_phase} population is "
                "incomplete."
            )
        identities: set[tuple[str, str]] = set()
        for record in records:
            identity = (
                _text(
                    record.get("domain_record_id"),
                    owner=f"{owner} append-only domain record id",
                ),
                _text(
                    record.get("generator_id"),
                    owner=f"{owner} append-only generator id",
                ),
            )
            _integer(
                record.get("pool_index"),
                owner=f"{owner} append-only pool index",
                minimum=0,
            )
            _text(
                record.get("pool_label"),
                owner=f"{owner} append-only pool label",
            )
            position = _integer(
                record.get("insertion_position"),
                owner=f"{owner} append-only insertion position",
                minimum=0,
            )
            if (
                identity in identities
                or position != append_position
                or record.get("position_class") != "append"
            ):
                raise CanonicalResumeError(
                    f"{owner} append-only scoring contains an interior "
                    "or repeated identity."
                )
            identities.add(identity)
        all_records.extend(records)
    unsigned = dict(payload)
    supplied_sha = _sha256(
        unsigned.pop("sha256", None),
        owner=f"{owner}.sha256",
    )
    if (
        _integer(
            payload.get("scored_record_count"),
            owner=f"{owner}.scored_record_count",
            minimum=1,
        )
        != len(all_records)
        or _integer(
            payload.get("interior_scored_count"),
            owner=f"{owner}.interior_scored_count",
            minimum=0,
        )
        != 0
        or _integer(
            payload.get("append_scored_count"),
            owner=f"{owner}.append_scored_count",
            minimum=1,
        )
        != len(all_records)
        or supplied_sha != _digest_json(unsigned)
    ):
        raise CanonicalResumeError(
            f"{owner} append-only scored counts or digest disagree."
        )


def _validate_commutation_reduced_insertion_round(
    value: Any,
    *,
    owner: str,
    expected_schema: str,
    expected_policy: str,
    expected_requested_positions: Sequence[int],
    expected_domain_open: bool,
    scored_population: Any,
    expected_effective_mode: str | None = None,
) -> dict[int, Mapping[str, Any]]:
    """Validate the exact representative partition for one accepted round."""

    receipt = _mapping(value, owner=owner)
    requested_domain = tuple(
        _integer(
            position,
            owner=f"{owner} expected requested position",
            minimum=0,
        )
        for position in expected_requested_positions
    )
    if (
        not requested_domain
        or requested_domain
        != tuple(sorted(set(requested_domain)))
    ):
        raise CanonicalResumeError(
            f"{owner} expected requested domain is not canonical."
        )
    resolved_effective_mode = (
        str(expected_effective_mode)
        if expected_effective_mode is not None
        else (
            "full_commutation_reduced"
            if expected_domain_open
            else "append_only"
        )
    )
    if (
        receipt.get("schema") != expected_schema
        or receipt.get("policy") != expected_policy
        or receipt.get("domain_open") is not expected_domain_open
        or receipt.get("domain_state")
        != ("open" if expected_domain_open else "closed")
        or receipt.get("effective_insertion_mode")
        != resolved_effective_mode
    ):
        raise CanonicalResumeError(
            f"{owner} insertion-policy identity is incomplete."
        )
    if expected_policy == AppendCommutationReducedInsertion.kind and (
        _integer(
            receipt.get("append_position"),
            owner=f"{owner}.append_position",
            minimum=0,
        )
        != max(requested_domain)
        or requested_domain != (max(requested_domain),)
    ):
        raise CanonicalResumeError(
            f"{owner} append-reduced domain is not exactly its endpoint."
        )

    plan_rows = [
        _mapping(
            plan,
            owner=f"{owner}.candidate_position_plans[{index}]",
        )
        for index, plan in enumerate(
            _sequence(
                receipt.get("candidate_position_plans"),
                owner=f"{owner}.candidate_position_plans",
            )
        )
    ]
    if not plan_rows:
        raise CanonicalResumeError(
            f"{owner} has no candidate-position plans."
        )
    plans_by_pool_index: dict[int, Mapping[str, Any]] = {}
    representatives_by_pool_index: dict[int, tuple[int, ...]] = {}
    labels_by_pool_index: dict[int, str] = {}
    retained_total = 0
    collapsed_total = 0
    for plan_index, plan in enumerate(plan_rows):
        pool_index = _integer(
            plan.get("candidate_pool_index"),
            owner=f"{owner} plan {plan_index} pool index",
            minimum=0,
        )
        candidate_label = _text(
            plan.get("candidate_label"),
            owner=f"{owner} plan {plan_index} candidate label",
        )
        if (
            pool_index in plans_by_pool_index
            or plan.get("schema")
            != "commutation_reduced_insertion_positions_v1"
        ):
            raise CanonicalResumeError(
                f"{owner} repeats a pool index or uses an unreduced "
                "insertion plan schema."
            )
        requested = tuple(
            _integer(
                position,
                owner=f"{owner} plan {plan_index} requested position",
                minimum=0,
            )
            for position in _sequence(
                plan.get("requested_positions"),
                owner=f"{owner} plan {plan_index} requested positions",
            )
        )
        representatives = tuple(
            _integer(
                position,
                owner=f"{owner} plan {plan_index} representative",
                minimum=0,
            )
            for position in _sequence(
                plan.get("representative_positions"),
                owner=f"{owner} plan {plan_index} representatives",
            )
        )
        if (
            requested != requested_domain
            or representatives
            != tuple(sorted(set(representatives)))
            or not representatives
        ):
            raise CanonicalResumeError(
                f"{owner} plan {plan_index} does not cover the requested "
                "domain exactly once."
            )
        representative_by_position_raw = _integer_key_mapping(
            plan.get("representative_by_position"),
            owner=f"{owner} plan {plan_index} representative map",
        )
        representative_by_position = {
            position: _integer(
                representative,
                owner=(
                    f"{owner} plan {plan_index} mapped representative"
                ),
                minimum=0,
            )
            for position, representative
            in representative_by_position_raw.items()
        }
        members_raw = _integer_key_mapping(
            plan.get("members_by_representative"),
            owner=f"{owner} plan {plan_index} member partition",
        )
        members_by_representative = {
            representative: tuple(
                _integer(
                    position,
                    owner=f"{owner} plan {plan_index} class member",
                    minimum=0,
                )
                for position in _sequence(
                    members,
                    owner=f"{owner} plan {plan_index} class members",
                )
            )
            for representative, members in members_raw.items()
        }
        partition_members: list[int] = []
        derived_map: dict[int, int] = {}
        if set(members_by_representative) != set(representatives):
            raise CanonicalResumeError(
                f"{owner} plan {plan_index} class keys disagree with its "
                "representatives."
            )
        for representative, members in members_by_representative.items():
            if (
                not members
                or members != tuple(sorted(set(members)))
                or representative != min(members)
            ):
                raise CanonicalResumeError(
                    f"{owner} plan {plan_index} does not retain the earliest "
                    "position in each commutation class."
                )
            partition_members.extend(members)
            for position in members:
                if position in derived_map:
                    raise CanonicalResumeError(
                        f"{owner} plan {plan_index} commutation classes "
                        "overlap."
                    )
                derived_map[position] = representative
        if (
            tuple(sorted(partition_members)) != requested_domain
            or representative_by_position != derived_map
            or tuple(sorted(representative_by_position))
            != requested_domain
        ):
            raise CanonicalResumeError(
                f"{owner} plan {plan_index} representative maps do not "
                "close over the requested domain."
            )
        crossings = _sequence(
            plan.get("commuting_crossings"),
            owner=f"{owner} plan {plan_index} commuting crossings",
        )
        if (
            len(crossings) != max(requested_domain)
            or any(not isinstance(crossing, bool) for crossing in crossings)
        ):
            raise CanonicalResumeError(
                f"{owner} plan {plan_index} commuting-crossing receipt "
                "does not match the pre-round prefix."
            )
        class_start_by_position: dict[int, int] = {0: 0}
        class_start = 0
        for crossing_index, crossing in enumerate(crossings):
            if not crossing:
                class_start = crossing_index + 1
            class_start_by_position[crossing_index + 1] = class_start
        requested_by_class: dict[int, list[int]] = {}
        for position in requested_domain:
            requested_by_class.setdefault(
                class_start_by_position[position],
                [],
            ).append(position)
        expected_members_by_representative = {
            min(members): tuple(members)
            for members in requested_by_class.values()
        }
        if (
            members_by_representative
            != expected_members_by_representative
        ):
            raise CanonicalResumeError(
                f"{owner} plan {plan_index} class partition disagrees "
                "with its commuting-crossing certificate."
            )
        collapsed = _integer(
            plan.get("collapsed_position_count"),
            owner=f"{owner} plan {plan_index} collapsed count",
            minimum=0,
        )
        if collapsed != len(requested_domain) - len(representatives):
            raise CanonicalResumeError(
                f"{owner} plan {plan_index} collapsed-position count "
                "disagrees."
            )
        plans_by_pool_index[pool_index] = plan
        representatives_by_pool_index[pool_index] = representatives
        labels_by_pool_index[pool_index] = candidate_label
        retained_total += len(representatives)
        collapsed_total += collapsed

    if tuple(plans_by_pool_index) != tuple(sorted(plans_by_pool_index)):
        raise CanonicalResumeError(
            f"{owner} candidate-position plans are not canonically ordered."
        )
    retained_rows = [
        _mapping(
            row,
            owner=f"{owner}.retained_representatives[{index}]",
        )
        for index, row in enumerate(
            _sequence(
                receipt.get("retained_representatives"),
                owner=f"{owner}.retained_representatives",
            )
        )
    ]
    expected_retained_rows = [
        {
            "candidate_pool_index": pool_index,
            "candidate_label": labels_by_pool_index[pool_index],
            "positions": list(
                representatives_by_pool_index[pool_index]
            ),
        }
        for pool_index in plans_by_pool_index
    ]
    observed_requested = tuple(
        _integer(
            position,
            owner=f"{owner} requested position",
            minimum=0,
        )
        for position in _sequence(
            receipt.get("requested_positions"),
            owner=f"{owner}.requested_positions",
        )
    )
    candidate_count = len(plans_by_pool_index)
    if (
        retained_rows != expected_retained_rows
        or observed_requested != requested_domain
        or _integer(
            receipt.get("candidate_count"),
            owner=f"{owner}.candidate_count",
            minimum=1,
        )
        != candidate_count
        or _integer(
            receipt.get("requested_position_count"),
            owner=f"{owner}.requested_position_count",
            minimum=1,
        )
        != len(requested_domain)
        or _integer(
            receipt.get("retained_representative_count"),
            owner=f"{owner}.retained_representative_count",
            minimum=1,
        )
        != retained_total
        or _integer(
            receipt.get("collapsed_position_count"),
            owner=f"{owner}.collapsed_position_count",
            minimum=0,
        )
        != collapsed_total
        or retained_total + collapsed_total
        != candidate_count * len(requested_domain)
    ):
        raise CanonicalResumeError(
            f"{owner} global representative closure is incomplete."
        )
    if (
        expected_policy == AppendCommutationReducedInsertion.kind
        and (
            retained_total != candidate_count
            or collapsed_total != 0
        )
    ):
        raise CanonicalResumeError(
            f"{owner} append-reduced closure must retain one endpoint and "
            "collapse zero positions per candidate."
        )
    _validate_scored_insertion_population(
        scored_population,
        owner=f"{owner} scored population",
        append_position=max(requested_domain),
        representatives_by_pool_index=representatives_by_pool_index,
    )
    return plans_by_pool_index


def _projective_fingerprint(value: Any, *, owner: str) -> str:
    fingerprint = _text(value, owner=owner)
    if not fingerprint.startswith(_PROJECTIVE_FINGERPRINT_PREFIX):
        raise CanonicalResumeError(
            f"{owner} is not a projective-state-v1 fingerprint."
        )
    _sha256(
        fingerprint[len(_PROJECTIVE_FINGERPRINT_PREFIX) :],
        owner=owner,
    )
    return fingerprint


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(child)
                for key, child in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child) for child in value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(
        "Canonical hydration can freeze only JSON-compatible values; "
        f"got {type(value).__name__}."
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _thaw_json(child)
            for key, child in value.items()
        }
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


@dataclass(frozen=True, slots=True)
class CanonicalRuntimePauliTermHydration:
    """One ordered runtime Pauli term needed for deterministic replay."""

    pauli_exyz: str
    coefficient_real: float
    coefficient_imaginary: float
    qubit_count: int


@dataclass(frozen=True, slots=True)
class CanonicalParameterBlockHydration:
    """One accepted logical generator and its runtime-coordinate block."""

    candidate_label: str
    logical_index: int
    runtime_start: int
    runtime_count: int
    execution_mode: str
    runtime_terms: tuple[CanonicalRuntimePauliTermHydration, ...]


@dataclass(frozen=True, slots=True)
class CanonicalOperatorHydration:
    """One active operator in exact accepted-prefix execution order."""

    active_position: int
    label: str
    generator_id: str
    parent_generator_id: str | None
    execution_mode: str
    admission_round: int
    runtime_terms: tuple[CanonicalRuntimePauliTermHydration, ...]


@dataclass(frozen=True, slots=True)
class CanonicalPruneStateHydration:
    """Recoverable peer-pruning trust state at the accepted prefix."""

    radius: float
    metric_damping: float
    update_count: int


@dataclass(frozen=True, slots=True)
class CanonicalAcceptedStateHydration:
    """Immutable direct-session state authenticated by one checkpoint."""

    source_path: Path
    source_sha256: str
    problem_request_sha256: str
    problem_binding_sha256: str
    route_family: str
    route_profile: str
    route_contract_sha256: str
    winning_beam_branch_ids: tuple[str, ...]
    beam_search_diagnostics: Mapping[str, Any]
    beam_fork_local_lineage_s_alg: int
    controller_round: int
    accepted_energy: float
    accepted_state_fingerprint: str
    operators: tuple[CanonicalOperatorHydration, ...]
    parameter_blocks: tuple[CanonicalParameterBlockHydration, ...]
    logical_parameters: tuple[float, ...]
    runtime_parameters: tuple[float, ...]
    selection_counts_by_pool_index: tuple[int, ...]
    available_pool_indices: tuple[int, ...]
    selected_parent_pool_indices: tuple[int, ...]
    nfev_total: int
    s_alg: int
    s_unique: int
    estimator_prefix_checkpoint_cursor: Mapping[str, Any]
    route_a_trust_region_state: Mapping[str, Any]
    prune_trust_state: CanonicalPruneStateHydration | None
    maturity_controller_snapshot: Mapping[str, Any]
    history: tuple[Mapping[str, Any], ...]
    active_prefix_estimator_receipts: tuple[Mapping[str, Any], ...]
    parameterization: Mapping[str, Any]
    route_contract: Mapping[str, Any]
    estimator_call_ledger_payload: Mapping[str, Any]
    terminal_signed_checkpoint: Mapping[str, Any]

    def mutable_history(self) -> list[dict[str, Any]]:
        return [
            dict(_thaw_json(row))
            for row in self.history
        ]

    def mutable_parameterization(self) -> dict[str, Any]:
        return dict(_thaw_json(self.parameterization))

    def mutable_route_contract(self) -> dict[str, Any]:
        return dict(_thaw_json(self.route_contract))

    def mutable_beam_search_diagnostics(self) -> dict[str, Any]:
        return dict(_thaw_json(self.beam_search_diagnostics))

    def mutable_estimator_call_ledger_payload(self) -> dict[str, Any]:
        return dict(_thaw_json(self.estimator_call_ledger_payload))

    def mutable_terminal_signed_checkpoint(self) -> dict[str, Any]:
        return dict(_thaw_json(self.terminal_signed_checkpoint))


def _validate_problem_binding(
    *,
    envelope: Mapping[str, Any],
    adapt: Mapping[str, Any],
    route_contract: Mapping[str, Any],
    expected_problem: ResolvedProblemContext,
) -> tuple[str, str]:
    receipt = ResolvedProblemReceipt.from_problem(expected_problem)
    request = expected_problem.request
    if (
        str(expected_problem.family_key) != "hh"
        or str(request.problem_key) != "hh"
        or int(request.num_sites) != 2
    ):
        raise CanonicalResumeError(
            "Canonical accepted-state resume is restricted to L=2 "
            "Hubbard--Holstein problems."
        )
    expected_particles = tuple(expected_problem.default_num_particles)
    if (
        float(request.v_nn) != 0.0
        or float(request.t_prime) != 0.0
        or (
            request.n_fermions is not None
            and int(request.n_fermions) != sum(expected_particles)
        )
        or request.molecular_problem_json is not None
        or request.molecular_vibronic_h2_fixture_json is not None
        or request.molecular_vibronic_h2o_fixture_json is not None
        or request.molecular_vibronic_h2o_linear_fd_fixture_json is not None
    ):
        raise CanonicalResumeError(
            "Canonical HH resume cannot attest noncanonical problem "
            "extensions omitted by the direct checkpoint schema."
        )

    settings = _mapping(envelope.get("settings"), owner="settings")
    expected_settings: dict[str, Any] = {
        "problem": str(request.problem_key),
        "L": int(request.num_sites),
        "t": float(request.t),
        "u": float(request.u),
        "dv": float(request.dv),
        "omega0": float(request.omega0),
        "g_ep": float(request.g_ep),
        "n_ph_max": int(request.n_ph_max),
        "boson_encoding": str(request.boson_encoding),
        "ordering": str(request.ordering),
        "boundary": str(request.boundary),
        "include_zero_point": bool(request.include_zero_point),
    }
    mismatches = [
        key
        for key, expected_value in expected_settings.items()
        if settings.get(key) != expected_value
    ]
    if mismatches:
        raise CanonicalResumeError(
            "Canonical accepted checkpoint describes a different physical "
            "problem: "
            + ", ".join(mismatches)
            + "."
        )
    particles_raw = adapt.get("num_particles")
    if particles_raw is not None:
        particles = _mapping(
            particles_raw,
            owner="adapt_vqe.num_particles",
        )
        if particles != {
            "n_up": int(expected_particles[0]),
            "n_dn": int(expected_particles[1]),
        }:
            raise CanonicalResumeError(
                "Canonical accepted checkpoint sector particle counts "
                "disagree with the resolved problem."
            )

    execution = _mapping(
        route_contract.get("execution_settings"),
        owner="route contract execution_settings",
    )
    required_route_settings = {
        "problem": "hh",
        "adapt_pool": "full_meta",
        "adapt_inner_optimizer": "POWELL",
        "adapt_maxiter": 200,
        "adapt_seed": 7,
        "phase0_pilot_enabled": False,
        "phase3_response_coordinate_scope": (
            "full_active_plus_singleton_v1"
        ),
        "phase3_runtime_split_child_set_symmetry_policy": "hard_guard",
        "phase3_runtime_split_max_subset_size": 1,
        "phase3_runtime_split_subset_sizes": "1",
        "adapt_pool_class_filter_json": None,
        "adapt_pool_label_filter_json": None,
    }
    route_mismatches = [
        key
        for key, expected_value in required_route_settings.items()
        if execution.get(key) != expected_value
    ]
    if route_mismatches:
        raise CanonicalResumeError(
            "Checkpoint route is not the canonical unfiltered guarded "
            "Paper-I HH route: "
            + ", ".join(route_mismatches)
            + "."
        )

    reference_manifest = _mapping(
        envelope.get("ansatz_input_state"),
        owner="ansatz_input_state",
    )
    expected_reference_manifest = build_statevector_manifest(
        psi_state=expected_problem.reference_state.build_state(),
        source=str(expected_problem.reference_state.source_label),
        handoff_state_kind="reference_state",
        amplitude_cutoff=1.0e-12,
    )
    if reference_manifest != expected_reference_manifest:
        raise CanonicalResumeError(
            "Canonical accepted checkpoint reference state disagrees with "
            "the resolved physical problem."
        )

    binding_payload = {
        "problem_request_sha256": receipt.problem_request_sha256,
        "settings": expected_settings,
        "sector_label": receipt.sector_label,
        "comparison_space_label": receipt.comparison_space_label,
        "reference_state": reference_manifest,
    }
    return receipt.problem_request_sha256, _digest_json(binding_payload)


def _validate_route_binding(
    *,
    envelope: Mapping[str, Any],
    adapt: Mapping[str, Any],
    expected_route_profile: str,
    expected_route_contract_sha256: str,
) -> tuple[str, dict[str, Any]]:
    profile = _text(
        expected_route_profile,
        owner="expected_route_profile",
    )
    expected_digest = _sha256(
        expected_route_contract_sha256,
        owner="expected_route_contract_sha256",
    )
    settings = _mapping(envelope.get("settings"), owner="settings")
    settings_contract = _mapping(
        settings.get("sr_route_profile_contract"),
        owner="settings.sr_route_profile_contract",
    )
    adapt_contract = _mapping(
        adapt.get("sr_route_profile_contract"),
        owner="adapt_vqe.sr_route_profile_contract",
    )
    if settings_contract != adapt_contract:
        raise CanonicalResumeError(
            "Checkpoint settings and accepted state carry different route "
            "contracts."
        )
    route_schema = settings_contract.get("schema")
    if route_schema not in {
        _ROUTE_CONTRACT_SCHEMA,
        *_RA_ROUTE_CONTRACT_SCHEMAS,
    }:
        raise CanonicalResumeError(
            "Canonical accepted checkpoint route-contract schema is "
            "unsupported."
        )
    observed_digest = _digest_json(settings_contract)
    declared_digests = {
        str(settings.get("sr_route_profile_contract_sha256", "")).lower(),
        str(adapt.get("sr_route_profile_contract_sha256", "")).lower(),
        str(
            _mapping(
                envelope.get("checkpoint"),
                owner="checkpoint",
            ).get("sr_route_profile_contract_sha256", "")
        ).lower(),
    }
    if declared_digests != {expected_digest} or observed_digest != expected_digest:
        raise CanonicalResumeError(
            "Canonical accepted checkpoint route-contract digest disagrees "
            "with the requested typed route."
        )
    embedded_profile = str(settings_contract.get("route_profile", ""))
    profile_fields = {
        embedded_profile,
        str(settings.get("sr_route_profile_request", "")),
        str(settings.get("sr_route_profile_resolved", "")),
        str(adapt.get("route_profile", "")),
        str(adapt.get("sr_route_profile_request", "")),
        str(adapt.get("sr_route_profile_resolved", "")),
    }
    if profile_fields != {profile}:
        raise CanonicalResumeError(
            "Canonical accepted checkpoint route profile disagrees with the "
            "requested typed route."
        )
    route_family = str(settings_contract.get("route_family", ""))
    if (
        route_family not in _ROUTE_FAMILIES
        or str(adapt.get("route_family", "")) != route_family
    ):
        raise CanonicalResumeError(
            "Canonical accepted checkpoint has an incompatible route family."
        )
    if route_schema in _RA_ROUTE_CONTRACT_SCHEMAS:
        invariants = _mapping(
            settings_contract.get("semantic_invariants"),
            owner="RA route contract semantic_invariants",
        )
        if (
            route_family != "ra_adapt"
            or invariants.get("canonical_interface")
            != "run_ra_adapt_problem_request_v1"
            or invariants.get("selector_identity")
            != "ra_adapt_staged_phase_i_ii_iii_funnel_v1"
            or _integer(
                invariants.get("admission_cardinality"),
                owner="RA route admission cardinality",
                minimum=1,
            )
            != 1
        ):
            raise CanonicalResumeError(
                "Canonical accepted checkpoint RA route identity is "
                "incompatible."
            )
    elif route_family == "ra_adapt":
        raise CanonicalResumeError(
            "An RA route family requires the authenticated RA route schema."
        )
    return route_family, settings_contract


def _validate_beam_route_binding(
    *,
    route_contract: Mapping[str, Any],
    declared_beam_enabled: bool,
) -> None:
    invariants = _mapping(
        route_contract.get("semantic_invariants"),
        owner="route contract semantic_invariants",
    )
    policy = invariants.get("canonical_beam_policy")
    if policy not in {None, "off", "fork_local"}:
        raise CanonicalResumeError(
            "Canonical route contract has an unsupported beam policy."
        )
    contract_beam_enabled = policy == "fork_local"
    if contract_beam_enabled != declared_beam_enabled:
        raise CanonicalResumeError(
            "Checkpoint beam declaration disagrees with the authenticated "
            "route contract."
        )
    if contract_beam_enabled:
        execution = _mapping(
            route_contract.get("execution_settings"),
            owner="route contract execution_settings",
        )
        if (
            _integer(
                execution.get("adapt_beam_live_branches"),
                owner="beam live parent branches",
                minimum=1,
            )
            != _integer(
                invariants.get("beam_live_parent_branches"),
                owner="beam invariant live parent branches",
                minimum=1,
            )
            or _integer(
                execution.get("adapt_beam_children_per_parent"),
                owner="beam children per parent",
                minimum=1,
            )
            != _integer(
                invariants.get("beam_admission_children_per_parent"),
                owner="beam invariant children per parent",
                minimum=1,
            )
            or invariants.get("beam_global_accounting")
            != "all_executed_branch_occurrences_in_global_s_alg_v1"
            or invariants.get("beam_unchanged_parent_survival") is not False
        ):
            raise CanonicalResumeError(
                "Canonical beam route contract is internally inconsistent."
            )


def _validate_state_manifest(
    value: Any,
    *,
    total_qubits: int,
) -> dict[str, Any]:
    manifest = _mapping(value, owner="initial_state")
    if (
        manifest.get("handoff_state_kind") != "prepared_state"
        or manifest.get("source")
        != "active_sr_snake_accepted_checkpoint"
        or _integer(
            manifest.get("nq_total"),
            owner="initial_state.nq_total",
            minimum=1,
        )
        != int(total_qubits)
        or not math.isclose(
            _finite(manifest.get("norm"), owner="initial_state.norm"),
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-10,
        )
    ):
        raise CanonicalResumeError(
            "Accepted-state manifest is incompatible with the resolved "
            "register."
        )
    amplitudes = _mapping(
        manifest.get("amplitudes_qn_to_q0"),
        owner="initial_state.amplitudes_qn_to_q0",
    )
    if not amplitudes:
        raise CanonicalResumeError(
            "Accepted-state manifest has no retained amplitudes."
        )
    probability = 0.0
    for basis, amplitude_raw in amplitudes.items():
        if (
            len(str(basis)) != int(total_qubits)
            or set(str(basis)).difference({"0", "1"})
        ):
            raise CanonicalResumeError(
                "Accepted-state manifest basis labels are incompatible with "
                "the resolved register."
            )
        amplitude = _mapping(
            amplitude_raw,
            owner=f"initial_state amplitude {basis}",
        )
        real = _finite(amplitude.get("re"), owner=f"amplitude {basis}.re")
        imaginary = _finite(
            amplitude.get("im"),
            owner=f"amplitude {basis}.im",
        )
        probability += real * real + imaginary * imaginary
    if not math.isclose(
        probability,
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-10,
    ):
        raise CanonicalResumeError(
            "Accepted-state manifest retained amplitudes are not normalized."
        )
    return manifest


def _validate_beam_declaration(
    *,
    adapt: Mapping[str, Any],
    route_contract: Mapping[str, Any],
    controller_round: int,
    beam_enabled: bool,
) -> tuple[tuple[str, ...], dict[str, Any]]:
    if not beam_enabled:
        return (), {}
    diagnostics = _mapping(
        adapt.get("beam_search_diagnostics"),
        owner="adapt_vqe.beam_search_diagnostics",
    )
    invariants = _mapping(
        route_contract.get("semantic_invariants"),
        owner="beam route semantic_invariants",
    )
    execution = _mapping(
        route_contract.get("execution_settings"),
        owner="beam route execution_settings",
    )
    if (
        diagnostics.get("schema")
        != "paper_i_canonical_fork_local_beam_search_v1"
        or diagnostics.get("comparison")
        != "accepted_energy_plus_weight_times_lineage_s_alg"
        or invariants.get("beam_comparison")
        != "accepted_energy_plus_weight_times_fork_local_s_alg_v1"
        or diagnostics.get("s_alg_scope")
        != "fork_local_cumulative_lineage"
        or _finite(
            diagnostics.get("s_alg_weight"),
            owner="beam diagnostics s_alg_weight",
        )
        != _finite(
            invariants.get("beam_s_alg_weight"),
            owner="beam invariant s_alg_weight",
        )
        or _finite(
            diagnostics.get("s_alg_weight"),
            owner="beam diagnostics s_alg_weight",
        )
        != _finite(
            execution.get("adapt_beam_lambda"),
            owner="beam execution s_alg_weight",
        )
        or diagnostics.get("calibration_status")
        != invariants.get("beam_calibration_status")
        or _integer(
            diagnostics.get("live_parent_branches"),
            owner="beam diagnostics live_parent_branches",
            minimum=1,
        )
        != _integer(
            invariants.get("beam_live_parent_branches"),
            owner="beam invariant live_parent_branches",
            minimum=1,
        )
        or _integer(
            diagnostics.get("admission_children_per_parent"),
            owner="beam diagnostics admission_children_per_parent",
            minimum=1,
        )
        != _integer(
            invariants.get("beam_admission_children_per_parent"),
            owner="beam invariant admission_children_per_parent",
            minimum=1,
        )
        or _integer(
            diagnostics.get("maximum_admission_children_per_round"),
            owner=(
                "beam diagnostics maximum_admission_children_per_round"
            ),
            minimum=1,
        )
        != _integer(
            invariants.get("beam_maximum_admission_children_per_round"),
            owner="beam invariant maximum children per round",
            minimum=1,
        )
        or diagnostics.get("unchanged_parent_survival") is not False
        or diagnostics.get("unchanged_parent_survival")
        != invariants.get("beam_unchanged_parent_survival")
        or diagnostics.get("phase_live_hysteresis") is not False
        or diagnostics.get("phase_live_hysteresis")
        != invariants.get("beam_phase_live_hysteresis")
    ):
        raise CanonicalResumeError(
            "Beam accepted checkpoint diagnostics are incomplete or "
            "noncanonical."
        )
    winning_ids = tuple(
        _text(value, owner="beam winning branch id")
        for value in _sequence(
            diagnostics.get("winning_branch_ids"),
            owner="beam winning_branch_ids",
        )
    )
    if (
        not winning_ids
        or len(winning_ids) != controller_round
        or len(set(winning_ids)) != len(winning_ids)
        or adapt.get("branch_id") != winning_ids[-1]
        or adapt.get("parent_branch_id")
        != (winning_ids[-2] if len(winning_ids) > 1 else None)
    ):
        raise CanonicalResumeError(
            "Beam winner declaration does not identify one exact history "
            "lineage."
        )
    round_rows = [
        _mapping(row, owner=f"beam diagnostics round {index}")
        for index, row in enumerate(
            _sequence(
                diagnostics.get("rounds"),
                owner="beam diagnostics rounds",
            )
        )
    ]
    if len(round_rows) != len(winning_ids):
        raise CanonicalResumeError(
            "Beam diagnostics do not cover every winning history round."
        )
    first_beam_round = 1
    for offset, (branch_id, round_row) in enumerate(
        zip(winning_ids, round_rows, strict=True)
    ):
        round_index = first_beam_round + offset
        expected_parent = (
            None if offset == 0 else winning_ids[offset - 1]
        )
        child_rows = [
            _mapping(
                row,
                owner=f"beam diagnostics round {round_index} child",
            )
            for row in _sequence(
                round_row.get("children"),
                owner=f"beam diagnostics round {round_index} children",
            )
        ]
        matching_children = [
            child
            for child in child_rows
            if child.get("branch_id") == branch_id
            and child.get("parent_branch_id") == expected_parent
        ]
        survivor_ids = tuple(
            _text(
                value,
                owner=(
                    f"beam diagnostics round {round_index} survivor id"
                ),
            )
            for value in _sequence(
                round_row.get("survivor_branch_ids"),
                owner=(
                    f"beam diagnostics round {round_index} survivors"
                ),
            )
        )
        if (
            _integer(
                round_row.get("controller_round"),
                owner=(
                    f"beam diagnostics round {round_index} controller_round"
                ),
                minimum=1,
            )
            != round_index
            or len(matching_children) != 1
            or branch_id not in survivor_ids
        ):
            raise CanonicalResumeError(
                f"Beam diagnostics round {round_index} does not authenticate "
                "its winning child."
            )
    _integer(
        diagnostics.get("initial_unbranched_s_alg"),
        owner="beam diagnostics initial_unbranched_s_alg",
        minimum=0,
    )
    _integer(
        diagnostics.get("all_executed_s_alg"),
        owner="beam diagnostics all_executed_s_alg",
        minimum=0,
    )
    _integer(
        diagnostics.get("winning_lineage_s_alg"),
        owner="beam diagnostics winning_lineage_s_alg",
        minimum=0,
    )
    return winning_ids, diagnostics


def _runtime_term(
    raw: Any,
    *,
    owner: str,
    total_qubits: int,
) -> CanonicalRuntimePauliTermHydration:
    value = _mapping(raw, owner=owner)
    pauli = _text(value.get("pauli_exyz"), owner=f"{owner}.pauli_exyz")
    if (
        pauli != pauli.lower()
        or len(pauli) != int(total_qubits)
        or set(pauli).difference({"e", "x", "y", "z"})
        or set(pauli) == {"e"}
    ):
        raise CanonicalResumeError(
            f"{owner}.pauli_exyz violates the internal e/x/y/z convention."
        )
    qubits = _integer(
        value.get("nq"),
        owner=f"{owner}.nq",
        minimum=1,
    )
    if qubits != int(total_qubits):
        raise CanonicalResumeError(
            f"{owner}.nq disagrees with the resolved register."
        )
    return CanonicalRuntimePauliTermHydration(
        pauli_exyz=pauli,
        coefficient_real=_finite(
            value.get("coeff_re"),
            owner=f"{owner}.coeff_re",
        ),
        coefficient_imaginary=_finite(
            value.get("coeff_im"),
            owner=f"{owner}.coeff_im",
        ),
        qubit_count=qubits,
    )


def _validate_prefix_receipt(
    raw: Any,
    *,
    owner: str,
) -> dict[str, Any]:
    receipt = _mapping(raw, owner=owner)
    if (
        receipt.get("schema") != _PREFIX_RECEIPT_SCHEMA
        or receipt.get("enabled") is not True
        or receipt.get("status") != "complete"
    ):
        raise CanonicalResumeError(
            f"{owner} is not a complete active-prefix estimator receipt."
        )
    return receipt


@dataclass(frozen=True, slots=True)
class _ValidatedSignedPrefix:
    payload: dict[str, Any]
    operator_labels: tuple[str, ...]
    operator_rows: tuple[
        tuple[
            str,
            str,
            str | None,
            str,
            tuple[CanonicalRuntimePauliTermHydration, ...],
        ],
        ...,
    ]
    parameter_blocks: tuple[CanonicalParameterBlockHydration, ...]
    logical_parameters: tuple[float, ...]
    runtime_parameters: tuple[float, ...]
    state_fingerprint: str
    ledger_receipt: dict[str, Any]


def _runtime_term_sort_key(
    item: CanonicalRuntimePauliTermHydration,
) -> tuple[str, float, float, int]:
    return (
        item.pauli_exyz,
        item.coefficient_real,
        item.coefficient_imaginary,
        item.qubit_count,
    )


def _validate_signed_prefix(
    raw: Any,
    *,
    owner: str,
    expected_round: int,
    expected_route_profile: str,
    expected_route_contract_sha256: str,
    total_qubits: int,
    allow_multi_term_operators: bool = False,
) -> _ValidatedSignedPrefix:
    checkpoint = _mapping(raw, owner=owner)
    if checkpoint.get("schema") != _SIGNED_PREFIX_SCHEMA:
        raise CanonicalResumeError(f"{owner} schema is unsupported.")
    declared_sha = _sha256(
        checkpoint.get("checkpoint_sha256"),
        owner=f"{owner}.checkpoint_sha256",
    )
    unsigned = dict(checkpoint)
    unsigned.pop("checkpoint_sha256", None)
    if _digest_json(unsigned) != declared_sha:
        raise CanonicalResumeError(
            f"{owner} signed-checkpoint SHA-256 mismatch."
        )
    if _integer(
        checkpoint.get("outer_iteration"),
        owner=f"{owner}.outer_iteration",
        minimum=1,
    ) != int(expected_round):
        raise CanonicalResumeError(
            f"{owner} outer iteration disagrees with accepted history."
        )
    if (
        checkpoint.get("sr_route_profile") != expected_route_profile
        or str(
            checkpoint.get("sr_route_profile_contract_sha256", "")
        ).lower()
        != expected_route_contract_sha256
    ):
        raise CanonicalResumeError(
            f"{owner} route binding disagrees with the requested typed route."
        )

    labels_raw = _sequence(
        checkpoint.get("ordered_active_operator_labels"),
        owner=f"{owner}.ordered_active_operator_labels",
    )
    labels = tuple(
        _text(label, owner=f"{owner} operator label")
        for label in labels_raw
    )
    depth = _integer(
        checkpoint.get("active_ansatz_depth"),
        owner=f"{owner}.active_ansatz_depth",
        minimum=1,
    )
    if depth != len(labels):
        raise CanonicalResumeError(
            f"{owner} active depth and operator labels disagree."
        )

    operator_rows_raw = _sequence(
        checkpoint.get("ordered_active_operators"),
        owner=f"{owner}.ordered_active_operators",
    )
    if len(operator_rows_raw) != depth:
        raise CanonicalResumeError(
            f"{owner} active operator records are incomplete."
        )
    operator_rows: list[
        tuple[
            str,
            str,
            str | None,
            str,
            tuple[CanonicalRuntimePauliTermHydration, ...],
        ]
    ] = []
    for position, row_raw in enumerate(operator_rows_raw):
        row = _mapping(
            row_raw,
            owner=f"{owner}.ordered_active_operators[{position}]",
        )
        if _integer(
            row.get("active_position"),
            owner=f"{owner} operator position",
            minimum=0,
        ) != position:
            raise CanonicalResumeError(
                f"{owner} active operator positions are not contiguous."
            )
        label = _text(row.get("label"), owner=f"{owner} operator label")
        if label != labels[position]:
            raise CanonicalResumeError(
                f"{owner} operator row and ordered label disagree."
            )
        generator_id = _text(
            row.get("generator_id"),
            owner=f"{owner} generator_id",
        )
        parent_raw = row.get("parent_generator_id")
        parent_id = (
            None
            if parent_raw is None
            else _text(parent_raw, owner=f"{owner} parent_generator_id")
        )
        execution_mode = _text(
            row.get("execution_mode"),
            owner=f"{owner} execution_mode",
        )
        term_rows = _sequence(
            row.get("serialized_terms_exyz_in_execution_order"),
            owner=f"{owner} serialized runtime terms",
        )
        if not term_rows:
            raise CanonicalResumeError(
                f"{owner} operator {label!r} has no replayable runtime terms."
            )
        terms = tuple(
            _runtime_term(
                term,
                owner=f"{owner} operator {position} term {term_index}",
                total_qubits=total_qubits,
            )
            for term_index, term in enumerate(term_rows)
        )
        if not allow_multi_term_operators and len(terms) != 1:
            raise CanonicalResumeError(
                f"{owner} operator {label!r} is not a cardinality-one "
                "canonical Pauli child."
            )
        operator_rows.append(
            (label, generator_id, parent_id, execution_mode, terms)
        )

    logical_parameters = tuple(
        _finite(value, owner=f"{owner} logical parameter")
        for value in _sequence(
            checkpoint.get("signed_unwrapped_logical_parameters"),
            owner=f"{owner}.signed_unwrapped_logical_parameters",
        )
    )
    runtime_parameters = tuple(
        _finite(value, owner=f"{owner} runtime parameter")
        for value in _sequence(
            checkpoint.get("signed_unwrapped_runtime_parameters"),
            owner=f"{owner}.signed_unwrapped_runtime_parameters",
        )
    )
    if len(logical_parameters) != depth or (
        not allow_multi_term_operators
        and len(runtime_parameters) != depth
    ):
        raise CanonicalResumeError(
            f"{owner} accepted parameter dimensions are incomplete."
        )

    parameterization = _mapping(
        checkpoint.get("parameterization"),
        owner=f"{owner}.parameterization",
    )
    if (
        _integer(
            parameterization.get("logical_operator_count"),
            owner=f"{owner} logical parameter count",
            minimum=1,
        )
        != depth
        or _integer(
            parameterization.get("runtime_parameter_count"),
            owner=f"{owner} runtime parameter count",
            minimum=1,
        )
        != len(runtime_parameters)
    ):
        raise CanonicalResumeError(
            f"{owner} parameterization dimensions disagree."
        )
    blocks_raw = _sequence(
        parameterization.get("blocks"),
        owner=f"{owner}.parameterization.blocks",
    )
    if len(blocks_raw) != depth:
        raise CanonicalResumeError(
            f"{owner} parameterization blocks are incomplete."
        )
    parameter_blocks: list[CanonicalParameterBlockHydration] = []
    next_runtime_start = 0
    for logical_index, block_raw in enumerate(blocks_raw):
        block = _mapping(
            block_raw,
            owner=f"{owner} parameter block {logical_index}",
        )
        runtime_start = _integer(
            block.get("runtime_start"),
            owner=f"{owner} block runtime_start",
            minimum=0,
        )
        runtime_count = _integer(
            block.get("runtime_count"),
            owner=f"{owner} block runtime_count",
            minimum=1,
        )
        if (
            _integer(
                block.get("logical_index"),
                owner=f"{owner} block logical_index",
                minimum=0,
            )
            != logical_index
            or runtime_start != next_runtime_start
            or (
                not allow_multi_term_operators
                and runtime_count != 1
            )
            or block.get("candidate_label") != labels[logical_index]
            or block.get("execution_mode")
            != operator_rows[logical_index][3]
        ):
            raise CanonicalResumeError(
                f"{owner} parameter block order or identity disagrees."
            )
        terms = tuple(
            _runtime_term(
                term,
                owner=(
                    f"{owner} parameter block {logical_index} "
                    f"term {term_index}"
                ),
                total_qubits=total_qubits,
            )
            for term_index, term in enumerate(
                _sequence(
                    block.get("runtime_terms_exyz"),
                    owner=f"{owner} block runtime terms",
                )
            )
        )
        operator_terms = operator_rows[logical_index][4]
        if allow_multi_term_operators:
            terms_close = sorted(
                terms,
                key=_runtime_term_sort_key,
            ) == sorted(
                operator_terms,
                key=_runtime_term_sort_key,
            )
        else:
            terms_close = terms == operator_terms
        if (
            len(terms) != runtime_count
            or not terms_close
        ):
            raise CanonicalResumeError(
                f"{owner} operator and parameter-block runtime terms disagree."
            )
        parameter_blocks.append(
            CanonicalParameterBlockHydration(
                candidate_label=labels[logical_index],
                logical_index=logical_index,
                runtime_start=runtime_start,
                runtime_count=runtime_count,
                execution_mode=operator_rows[logical_index][3],
                runtime_terms=terms,
            )
        )
        next_runtime_start += runtime_count
    if next_runtime_start != len(runtime_parameters):
        raise CanonicalResumeError(
            f"{owner} runtime parameter blocks do not close."
        )

    strict = _mapping(
        checkpoint.get("strict_replay"),
        owner=f"{owner}.strict_replay",
    )
    if (
        strict.get("schema") != "static_adapt_strict_state_replay_v1"
        or strict.get("passed") is not True
    ):
        raise CanonicalResumeError(
            f"{owner} strict projective-state replay did not pass."
        )
    tolerance = _finite(
        strict.get("tolerance"),
        owner=f"{owner} replay tolerance",
    )
    phase_l2 = _finite(
        strict.get("phase_aligned_l2"),
        owner=f"{owner} replay phase-aligned L2",
    )
    fidelity = _finite(
        strict.get("fidelity"),
        owner=f"{owner} replay fidelity",
    )
    if (
        tolerance <= 0.0
        or phase_l2 < 0.0
        or phase_l2 > tolerance
        or fidelity < 0.0
        or fidelity > 1.0 + tolerance
        or 1.0 - fidelity > max(1.0e-12, 10.0 * tolerance)
    ):
        raise CanonicalResumeError(
            f"{owner} strict replay receipt is inconsistent."
        )
    active_sector = _mapping(
        checkpoint.get("active_generator_sector_contract"),
        owner=f"{owner}.active_generator_sector_contract",
    )
    state_sector = _mapping(
        checkpoint.get("state_sector_contract"),
        owner=f"{owner}.state_sector_contract",
    )
    if (
        active_sector.get("passed_with_parameterization") is not True
        or state_sector.get("passed") is not True
    ):
        raise CanonicalResumeError(
            f"{owner} sector or padding guard did not pass."
        )
    fingerprint = _projective_fingerprint(
        checkpoint.get("projective_state_fingerprint"),
        owner=f"{owner}.projective_state_fingerprint",
    )
    ledger_receipt = _validate_prefix_receipt(
        checkpoint.get("estimator_ledger_receipt"),
        owner=f"{owner}.estimator_ledger_receipt",
    )
    return _ValidatedSignedPrefix(
        payload=checkpoint,
        operator_labels=labels,
        operator_rows=tuple(operator_rows),
        parameter_blocks=tuple(parameter_blocks),
        logical_parameters=logical_parameters,
        runtime_parameters=runtime_parameters,
        state_fingerprint=fingerprint,
        ledger_receipt=ledger_receipt,
    )


def _validate_history_and_admissions(
    *,
    adapt: Mapping[str, Any],
    controller_round: int,
    route_family: str,
    route_contract: Mapping[str, Any],
    winning_beam_branch_ids: tuple[str, ...],
    expected_route_profile: str,
    expected_route_contract_sha256: str,
    total_qubits: int,
    allow_multi_term_operators: bool,
) -> tuple[
    list[dict[str, Any]],
    list[_ValidatedSignedPrefix],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    Mapping[str, Any],
    CanonicalPruneStateHydration | None,
    tuple[str, ...],
    tuple[str, ...],
]:
    history_raw = _sequence(
        adapt.get("history"),
        owner="adapt_vqe.history",
    )
    history = [
        _mapping(row, owner=f"adapt_vqe.history[{index}]")
        for index, row in enumerate(history_raw)
    ]
    if len(history) != controller_round:
        raise CanonicalResumeError(
            "Accepted checkpoint history is partial or disagrees with its "
            "controller round."
        )
    tail = _sequence(
        adapt.get("history_tail"),
        owner="adapt_vqe.history_tail",
    )
    if tail != history or _integer(
        adapt.get("history_tail_count"),
        owner="adapt_vqe.history_tail_count",
        minimum=1,
    ) != controller_round:
        raise CanonicalResumeError(
            "Accepted checkpoint does not retain a complete authenticated "
            "history."
        )

    active_labels: list[str] = []
    active_generator_ids: list[str] = []
    active_admission_rounds: list[int] = []
    active_retained_parent_owners: list[
        Mapping[str, Any] | None
    ] = []
    pool_size = _integer(
        adapt.get("pool_size"),
        owner="adapt_vqe.pool_size",
        minimum=1,
    )
    selection_counts = [0 for _ in range(pool_size)]
    selected_parent_indices: list[int] = []
    signed_prefixes: list[_ValidatedSignedPrefix] = []
    maturity_snapshot: Mapping[str, Any] | None = None
    prune_update_count = 0
    prune_radius: float | None = None
    prune_metric_damping: float | None = None
    executed_prune_branch_ids: list[str] = []
    accepted_prune_branch_ids: list[str] = []
    semantic_invariants = _mapping(
        route_contract.get("semantic_invariants"),
        owner="route semantic invariants",
    )
    pruning_active = semantic_invariants.get("pruning_active")
    if not isinstance(pruning_active, bool):
        raise CanonicalResumeError(
            "Authenticated route does not declare a Boolean pruning mode."
        )
    insertion_policy = _resolve_authenticated_insertion_policy(
        route_contract
    )
    route_execution_settings = _mapping(
        route_contract.get("execution_settings"),
        owner="route execution settings",
    )
    insertion_mode = _text(
        route_execution_settings.get("adapt_insertion_mode"),
        owner="route insertion mode",
    )
    plateau_receipt_identities = {
        "insertion_commutation_plateau_v1": (
            "insertion_commutation_plateau_round_policy_v1",
            INSERTION_COMMUTATION_PLATEAU_CALIBRATION_STATUS,
        ),
        "insertion_commutation_plateau_v2": (
            "insertion_commutation_plateau_round_policy_v2",
            INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_CALIBRATION_STATUS,
        ),
    }
    plateau_receipt_identity = plateau_receipt_identities.get(insertion_mode)
    if (
        insertion_policy == "plateau_commutation"
        and plateau_receipt_identity is None
    ):
        raise CanonicalResumeError(
            "Authenticated plateau route has no supported receipt identity."
        )
    batch_maximum: int | None = None
    batch_search_window: int | None = None
    if route_family != "singleton_response_snake":
        batch_prefix = (
            "greedy"
            if route_family == "greedy_batch_response_snake"
            else "combinatorial"
        )
        batch_maximum = _integer(
            semantic_invariants.get(
                f"{batch_prefix}_batch_maximum_size"
            ),
            owner=f"route {batch_prefix} batch maximum size",
            minimum=1,
        )
        search_window_raw = semantic_invariants.get(
            f"{batch_prefix}_batch_search_window_size"
        )
        batch_search_window = (
            None
            if search_window_raw is None
            else _integer(
                search_window_raw,
                owner=f"route {batch_prefix} batch search window",
                minimum=1,
            )
        )
    first_beam_round = (
        controller_round - len(winning_beam_branch_ids) + 1
        if winning_beam_branch_ids
        else controller_round + 1
    )

    for round_index, row in enumerate(history, start=1):
        if _integer(
            row.get("depth"),
            owner=f"history[{round_index - 1}].depth",
            minimum=1,
        ) != round_index:
            raise CanonicalResumeError(
                "Accepted checkpoint history rounds are not contiguous."
            )
        expected_branch_id = (
            None
            if round_index < first_beam_round
            else winning_beam_branch_ids[round_index - first_beam_round]
        )
        expected_parent_branch_id = (
            None
            if expected_branch_id is None
            or round_index == first_beam_round
            else winning_beam_branch_ids[
                round_index - first_beam_round - 1
            ]
        )
        if (
            row.get("branch_id") != expected_branch_id
            or row.get("parent_branch_id") != expected_parent_branch_id
        ):
            raise CanonicalResumeError(
                f"Accepted round {round_index} does not belong to the "
                "declared winning beam lineage."
            )
        selected_labels = tuple(
            _text(
                value,
                owner=f"history[{round_index - 1}] selected label",
            )
            for value in _sequence(
                row.get("selected_batch_labels", row.get("selected_ops")),
                owner=f"history[{round_index - 1}] selected labels",
            )
        )
        selected_indices = tuple(
            _integer(
                value,
                owner=f"history[{round_index - 1}] pool index",
                minimum=0,
            )
            for value in _sequence(
                row.get(
                    "selected_pool_indices",
                    [row.get("pool_index")],
                ),
                owner=f"history[{round_index - 1}] pool indices",
            )
        )
        effective_positions = tuple(
            _integer(
                value,
                owner=f"history[{round_index - 1}] effective position",
                minimum=0,
            )
            for value in _sequence(
                row.get(
                    "selected_batch_effective_positions",
                    row.get(
                        "selected_effective_positions",
                        [row.get("selected_position")],
                    ),
                ),
                owner=f"history[{round_index - 1}] effective positions",
            )
        )
        original_positions = tuple(
            _integer(
                value,
                owner=f"history[{round_index - 1}] original position",
                minimum=0,
            )
            for value in _sequence(
                row.get(
                    "selected_batch_positions",
                    row.get(
                        "selected_positions",
                        [row.get("selected_position")],
                    ),
                ),
                owner=f"history[{round_index - 1}] original positions",
            )
        )
        pre_round_depth = len(active_labels)
        expected_effective_positions: list[int] = []
        prior_original_positions: list[int] = []
        for position in original_positions:
            if position > pre_round_depth:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} original insertion "
                    "position is outside the pre-round prefix."
                )
            expected_effective_positions.append(
                position
                + sum(
                    1
                    for prior in prior_original_positions
                    if prior <= position
                )
            )
            prior_original_positions.append(position)
        if tuple(expected_effective_positions) != effective_positions:
            raise CanonicalResumeError(
                f"Accepted round {round_index} effective insertion "
                "positions do not follow the atomic insertion contract."
            )
        if (
            not selected_labels
            or len(selected_labels) != len(selected_indices)
            or len(selected_labels) != len(effective_positions)
            or len(selected_labels) != len(original_positions)
            or _integer(
                row.get("selected_logical_size"),
                owner=f"history[{round_index - 1}] selected_logical_size",
                minimum=1,
            )
            != len(selected_labels)
            or _sequence(
                row.get("selected_ops"),
                owner=f"history[{round_index - 1}] selected_ops",
            )
            != list(selected_labels)
            or _sequence(
                row.get("selected_logical_pool_indices"),
                owner=(
                    f"history[{round_index - 1}] "
                    "selected_logical_pool_indices"
                ),
            )
            != list(selected_indices)
        ):
            raise CanonicalResumeError(
                f"Accepted round {round_index} admission cardinalities "
                "disagree."
            )
        insertion_receipt_raw = row.get("insertion_commutation_plateau")
        reduced_receipt_raw = row.get("insertion_commutation_reduced")
        if insertion_policy == "append_only":
            if (
                insertion_receipt_raw is not None
                or reduced_receipt_raw is not None
                or any(
                    position != pre_round_depth
                    for position in original_positions
                )
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} does not obey the "
                    "authenticated append-only insertion policy."
                )
            _validate_append_only_scored_population(
                row.get("scored_insertion_position_population"),
                owner=(
                    f"history[{round_index - 1}]."
                    "scored_insertion_position_population"
                ),
                append_position=pre_round_depth,
            )
        elif insertion_policy == "always_commutation_reduced":
            if insertion_receipt_raw is not None:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} always-insertion must "
                    "not carry a plateau-policy receipt."
                )
            plans_by_pool_index = (
                _validate_commutation_reduced_insertion_round(
                    reduced_receipt_raw,
                    owner=(
                        f"history[{round_index - 1}]."
                        "insertion_commutation_reduced"
                    ),
                    expected_schema=(
                        "commutation_reduced_insertion_domain_receipt_v1"
                    ),
                    expected_policy="always_commutation_reduced",
                    expected_requested_positions=tuple(
                        range(pre_round_depth + 1)
                    ),
                    expected_domain_open=True,
                    scored_population=row.get(
                        "scored_insertion_position_population"
                    ),
                )
            )
            for pool_index, original_position in zip(
                selected_indices,
                original_positions,
                strict=True,
            ):
                plan = plans_by_pool_index.get(pool_index)
                if (
                    plan is None
                    or original_position
                    not in _sequence(
                        plan.get("representative_positions"),
                        owner=(
                            f"history[{round_index - 1}] selected insertion "
                            "representatives"
                        ),
                    )
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} chosen insertion "
                        "does not bind its authenticated reduced-position "
                        "plan."
                    )
        elif insertion_policy == AppendCommutationReducedInsertion.kind:
            if insertion_receipt_raw is not None:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} append-reduced insertion "
                    "must not carry a plateau-policy receipt."
                )
            plans_by_pool_index = (
                _validate_commutation_reduced_insertion_round(
                    reduced_receipt_raw,
                    owner=(
                        f"history[{round_index - 1}]."
                        "insertion_commutation_reduced"
                    ),
                    expected_schema=(
                        "commutation_reduced_insertion_domain_receipt_v1"
                    ),
                    expected_policy=(
                        AppendCommutationReducedInsertion.kind
                    ),
                    expected_requested_positions=(pre_round_depth,),
                    expected_domain_open=False,
                    expected_effective_mode=(
                        AppendCommutationReducedInsertion.runtime_mode
                    ),
                    scored_population=row.get(
                        "scored_insertion_position_population"
                    ),
                )
            )
            if any(
                position != pre_round_depth
                for position in original_positions
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} append-reduced insertion "
                    "selected a non-endpoint position."
                )
            for pool_index, original_position in zip(
                selected_indices,
                original_positions,
                strict=True,
            ):
                plan = plans_by_pool_index.get(pool_index)
                if (
                    plan is None
                    or original_position
                    not in _sequence(
                        plan.get("representative_positions"),
                        owner=(
                            f"history[{round_index - 1}] selected insertion "
                            "representatives"
                        ),
                    )
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} chosen append endpoint "
                        "does not bind its authenticated reduced-position "
                        "plan."
                    )
        else:
            if plateau_receipt_identity is None:
                raise CanonicalResumeError(
                    "Authenticated plateau route has no receipt identity."
                )
            plateau_receipt_schema, plateau_calibration_status = (
                plateau_receipt_identity
            )
            insertion_receipt = _mapping(
                insertion_receipt_raw,
                owner=(
                    f"history[{round_index - 1}]."
                    "insertion_commutation_plateau"
                ),
            )
            if reduced_receipt_raw is not None:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} plateau insertion must "
                    "not carry an always-insertion receipt."
                )
            domain_open = insertion_receipt.get("domain_open")
            if not isinstance(domain_open, bool):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} plateau receipt does "
                    "not declare a Boolean domain state."
                )
            _validate_commutation_reduced_insertion_round(
                insertion_receipt,
                owner=(
                    f"history[{round_index - 1}]."
                    "insertion_commutation_plateau"
                ),
                expected_schema=plateau_receipt_schema,
                expected_policy=insertion_mode,
                expected_requested_positions=(
                    tuple(range(pre_round_depth + 1))
                    if domain_open
                    else (pre_round_depth,)
                ),
                expected_domain_open=domain_open,
                scored_population=row.get(
                    "scored_insertion_position_population"
                ),
            )
            if (
                not domain_open
                and any(
                    position != pre_round_depth
                    for position in original_positions
                )
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} closed plateau domain "
                    "is not append-only."
                )
            plan_rows = [
                _mapping(
                    plan,
                    owner=(
                        f"history[{round_index - 1}] insertion "
                        f"candidate-position plan {plan_index}"
                    ),
                )
                for plan_index, plan in enumerate(
                    _sequence(
                        insertion_receipt.get(
                            "candidate_position_plans"
                        ),
                        owner=(
                            f"history[{round_index - 1}] insertion "
                            "candidate-position plans"
                        ),
                    )
                )
            ]
            plans_by_pool_index: dict[int, Mapping[str, Any]] = {}
            all_requested_positions: set[int] = set()
            retained_representative_count = 0
            collapsed_position_count = 0
            for plan_index, plan in enumerate(plan_rows):
                pool_index = _integer(
                    plan.get("candidate_pool_index"),
                    owner=(
                        f"history[{round_index - 1}] insertion plan "
                        f"{plan_index} pool index"
                    ),
                    minimum=0,
                )
                if pool_index in plans_by_pool_index:
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} insertion receipt "
                        "repeats a candidate pool index."
                    )
                requested = tuple(
                    _integer(
                        position,
                        owner=(
                            f"history[{round_index - 1}] insertion "
                            "requested position"
                        ),
                        minimum=0,
                    )
                    for position in _sequence(
                        plan.get("requested_positions"),
                        owner=(
                            f"history[{round_index - 1}] insertion "
                            "requested positions"
                        ),
                    )
                )
                representatives = tuple(
                    _integer(
                        position,
                        owner=(
                            f"history[{round_index - 1}] insertion "
                            "representative position"
                        ),
                        minimum=0,
                    )
                    for position in _sequence(
                        plan.get("representative_positions"),
                        owner=(
                            f"history[{round_index - 1}] insertion "
                            "representative positions"
                        ),
                    )
                )
                if (
                    plan.get("schema")
                    != "commutation_reduced_insertion_positions_v1"
                    or not requested
                    or not representatives
                    or not set(representatives).issubset(requested)
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} insertion position "
                        "plan is incomplete."
                    )
                plans_by_pool_index[pool_index] = plan
                all_requested_positions.update(requested)
                retained_representative_count += len(representatives)
                collapsed_position_count += _integer(
                    plan.get("collapsed_position_count"),
                    owner=(
                        f"history[{round_index - 1}] insertion collapsed "
                        "position count"
                    ),
                    minimum=0,
                )
            if insertion_mode == "insertion_commutation_plateau_v2":
                _integer(
                    insertion_receipt.get(
                        "prior_accepted_transition_count"
                    ),
                    owner=(
                        f"history[{round_index - 1}] insertion prior "
                        "accepted transition count"
                    ),
                    minimum=0,
                )
                if (
                    _finite(
                        insertion_receipt.get(
                            "prior_mean_decrease_ratio_threshold"
                        ),
                        owner=(
                            f"history[{round_index - 1}] insertion prior "
                            "mean decrease ratio threshold"
                        ),
                    )
                    != INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD
                    or insertion_receipt.get("threshold_comparison")
                    != "marginal_to_prior_mean_strictly_below_v2"
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} prior-mean plateau "
                        "receipt is incomplete."
                    )
            if (
                insertion_receipt.get("schema")
                != plateau_receipt_schema
                or insertion_receipt.get("policy")
                != insertion_mode
                or insertion_receipt.get("calibration_status")
                != plateau_calibration_status
                or insertion_receipt.get("exact_reference_used") is not False
                or _integer(
                    insertion_receipt.get("candidate_count"),
                    owner=(
                        f"history[{round_index - 1}] insertion candidate "
                        "count"
                    ),
                    minimum=1,
                )
                != len(plan_rows)
                or _integer(
                    insertion_receipt.get("requested_position_count"),
                    owner=(
                        f"history[{round_index - 1}] insertion requested "
                        "position count"
                    ),
                    minimum=1,
                )
                != len(all_requested_positions)
                or tuple(
                    _integer(
                        position,
                        owner=(
                            f"history[{round_index - 1}] insertion domain "
                            "position"
                        ),
                        minimum=0,
                    )
                    for position in _sequence(
                        insertion_receipt.get("requested_positions"),
                        owner=(
                            f"history[{round_index - 1}] insertion domain "
                            "positions"
                        ),
                    )
                )
                != tuple(sorted(all_requested_positions))
                or _integer(
                    insertion_receipt.get(
                        "retained_representative_count"
                    ),
                    owner=(
                        f"history[{round_index - 1}] insertion retained "
                        "representative count"
                    ),
                    minimum=1,
                )
                != retained_representative_count
                or _integer(
                    insertion_receipt.get("collapsed_position_count"),
                    owner=(
                        f"history[{round_index - 1}] insertion collapsed "
                        "position total"
                    ),
                    minimum=0,
                )
                != collapsed_position_count
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} insertion-policy "
                    "receipt is incomplete."
                )
            insertion_feature_rows = [
                _mapping(
                    value,
                    owner=(
                        f"history[{round_index - 1}] selected insertion "
                        "feature"
                    ),
                )
                for value in _sequence(
                    row.get("selected_feature_rows"),
                    owner=(
                        f"history[{round_index - 1}] "
                        "selected_feature_rows"
                    ),
                )
            ]
            if len(insertion_feature_rows) != len(selected_labels):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} selected feature receipts "
                    "are incomplete."
                )
            for (
                label,
                pool_index,
                original_position,
                selected_feature,
            ) in zip(
                selected_labels,
                selected_indices,
                original_positions,
                insertion_feature_rows,
                strict=True,
            ):
                plan = plans_by_pool_index.get(pool_index)
                feature_label = _text(
                    selected_feature.get("candidate_label"),
                    owner=(
                        f"history[{round_index - 1}] selected feature "
                        "candidate label"
                    ),
                )
                feature_pool_index = _integer(
                    selected_feature.get("candidate_pool_index"),
                    owner=(
                        f"history[{round_index - 1}] selected feature "
                        "candidate pool index"
                    ),
                    minimum=0,
                )
                runtime_parent_raw = selected_feature.get(
                    "runtime_split_parent_label"
                )
                classifier_raw = selected_feature.get(
                    "physical_operator_classifier_label"
                )
                runtime_parent = (
                    None
                    if runtime_parent_raw in (None, "")
                    else _text(
                        runtime_parent_raw,
                        owner=(
                            f"history[{round_index - 1}] runtime-split "
                            "parent label"
                        ),
                    )
                )
                classifier = (
                    None
                    if classifier_raw in (None, "")
                    else _text(
                        classifier_raw,
                        owner=(
                            f"history[{round_index - 1}] physical "
                            "operator classifier label"
                        ),
                    )
                )
                if (
                    runtime_parent is not None
                    and classifier is not None
                    and runtime_parent != classifier
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} selected child "
                        "disagrees with its authenticated physical parent."
                    )
                plan_candidate_label = (
                    runtime_parent or classifier or feature_label
                )
                if (
                    plan is None
                    or feature_label != label
                    or feature_pool_index != pool_index
                    or plan.get("candidate_label")
                    != plan_candidate_label
                    or original_position
                    not in _sequence(
                        plan.get("representative_positions"),
                        owner=(
                            f"history[{round_index - 1}] selected insertion "
                            "representatives"
                        ),
                    )
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} chosen insertion "
                        "does not bind its authenticated position plan."
                    )
        if route_family == "singleton_response_snake" and len(
            selected_labels
        ) != 1:
            raise CanonicalResumeError(
                "Singleton route checkpoint contains a batch admission."
            )
        feature_rows = [
            _mapping(
                value,
                owner=f"history[{round_index - 1}] selected feature",
            )
            for value in _sequence(
                row.get("selected_feature_rows"),
                owner=f"history[{round_index - 1}] selected_feature_rows",
            )
        ]
        if len(feature_rows) != len(selected_labels):
            raise CanonicalResumeError(
                f"Accepted round {round_index} selected feature receipts "
                "are incomplete."
            )
        selected_retained_parent_owners: list[
            Mapping[str, Any] | None
        ] = []
        selected_generator_ids_list: list[str] = []
        for feature, label in zip(
            feature_rows,
            selected_labels,
            strict=True,
        ):
            retained_parent_owner: Mapping[str, Any] | None = None
            runtime_split_mode = str(
                feature.get("runtime_split_mode", "off")
            )
            metadata_raw = feature.get("generator_metadata")
            metadata = (
                {}
                if metadata_raw is None
                else _mapping(
                    metadata_raw,
                    owner=(
                        f"history[{round_index - 1}] generator metadata"
                    ),
                )
            )
            is_ra_guarded_singleton = bool(
                runtime_split_mode
                == "guarded_singleton_children_only_v1"
                and metadata.get("ra_candidate_representation")
                == "single_pauli_word_v1"
                and isinstance(metadata.get("ra_adapter_id"), str)
                and bool(str(metadata.get("ra_adapter_id")).strip())
            )
            retained_parent_raw = metadata.get(
                "ra_retained_parent_owner"
            )
            if is_ra_guarded_singleton:
                retained_parent = _mapping(
                    retained_parent_raw,
                    owner=(
                        f"history[{round_index - 1}] retained-parent "
                        "owner receipt"
                    ),
                )
                receipt_sha = _sha256(
                    retained_parent.get("sha256"),
                    owner=(
                        f"history[{round_index - 1}] retained-parent "
                        "owner receipt SHA-256"
                    ),
                )
                unsigned_receipt = dict(retained_parent)
                unsigned_receipt.pop("sha256", None)
                parent_identity = _text(
                    retained_parent.get(
                        "parent_generator_identity"
                    ),
                    owner=(
                        f"history[{round_index - 1}] retained-parent "
                        "generator identity"
                    ),
                )
                parent_label = _text(
                    retained_parent.get("parent_label"),
                    owner=(
                        f"history[{round_index - 1}] retained-parent label"
                    ),
                )
                child_identity = _text(
                    retained_parent.get(
                        "child_generator_identity"
                    ),
                    owner=(
                        f"history[{round_index - 1}] retained child "
                        "generator identity"
                    ),
                )
                child_label = _text(
                    retained_parent.get("child_label"),
                    owner=(
                        f"history[{round_index - 1}] retained child label"
                    ),
                )
                manifest_sha = _sha256(
                    retained_parent.get(
                        "candidate_manifest_sha256"
                    ),
                    owner=(
                        f"history[{round_index - 1}] retained child "
                        "manifest SHA-256"
                    ),
                )
                manifest = _mapping(
                    metadata.get("ra_candidate_manifest"),
                    owner=(
                        f"history[{round_index - 1}] RA candidate manifest"
                    ),
                )
                parent_identities = tuple(
                    _text(
                        value,
                        owner=(
                            f"history[{round_index - 1}] candidate "
                            "parent identity"
                        ),
                    )
                    for value in _sequence(
                        manifest.get("parent_identities"),
                        owner=(
                            f"history[{round_index - 1}] candidate "
                            "parent identities"
                        ),
                    )
                )
                shared_contract = _mapping(
                    metadata.get("shared_pauli_pool_contract"),
                    owner=(
                        f"history[{round_index - 1}] shared-Pauli "
                        "parent contract"
                    ),
                )
                parent_labels = tuple(
                    _text(
                        value,
                        owner=(
                            f"history[{round_index - 1}] shared-Pauli "
                            "parent label"
                        ),
                    )
                    for value in _sequence(
                        shared_contract.get("parent_labels"),
                        owner=(
                            f"history[{round_index - 1}] shared-Pauli "
                            "parent labels"
                        ),
                    )
                )
                owner_indices = [
                    index
                    for index, value in enumerate(parent_identities)
                    if value == parent_identity
                ]
                if (
                    retained_parent.get("schema")
                    != "ra_adapt_retained_parent_owner_v1"
                    or _digest_json(unsigned_receipt) != receipt_sha
                    or len(parent_identities) != len(parent_labels)
                    or len(owner_indices) != 1
                    or parent_labels[owner_indices[0]] != parent_label
                    or metadata.get("ra_parent_generator_ids")
                    != list(parent_identities)
                    or metadata.get("parent_generator_id")
                    != parent_identity
                    or feature.get("parent_generator_id")
                    != parent_identity
                    or metadata.get("generator_id") != child_identity
                    or manifest.get("generator_identity")
                    != child_identity
                    or manifest.get("label") != child_label
                    or child_label != label
                    or _digest_json(manifest) != manifest_sha
                    or metadata.get(
                        "ra_candidate_manifest_sha256"
                    )
                    != manifest_sha
                    or feature.get(
                        "runtime_split_parent_label"
                    )
                    != parent_label
                    or feature.get(
                        "physical_operator_classifier_label"
                    )
                    != parent_label
                ):
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} retained-parent "
                        "owner is not bound to its authenticated child, "
                        "ancestry, classifier, and runtime parent."
                    )
                retained_parent_owner = retained_parent
            elif retained_parent_raw is not None:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} carries a retained-parent "
                    "owner outside the RA guarded-singleton route."
                )
            selected_retained_parent_owners.append(
                retained_parent_owner
            )
            generator_raw = feature.get("generator_id")
            if generator_raw is None or generator_raw == "":
                generator_raw = metadata.get("generator_id", label)
            selected_generator_ids_list.append(
                _text(
                    generator_raw,
                    owner=f"history[{round_index - 1}] generator id",
                )
            )
        selected_generator_ids = tuple(selected_generator_ids_list)
        if route_family == "singleton_response_snake":
            if (
                row.get("greedy_batch_admission") is not None
                or row.get("combinatorial_batch_admission") is not None
                or row.get("generator_id") != selected_generator_ids[0]
                or row.get("selected_op") != selected_labels[0]
                or _integer(
                    row.get("selected_position"),
                    owner=(
                        f"history[{round_index - 1}] selected_position"
                    ),
                    minimum=0,
                )
                != original_positions[0]
            ):
                raise CanonicalResumeError(
                    f"Accepted singleton round {round_index} admission "
                    "identity is inconsistent."
                )
        else:
            admission_field = (
                "greedy_batch_admission"
                if route_family == "greedy_batch_response_snake"
                else "combinatorial_batch_admission"
            )
            incompatible_field = (
                "combinatorial_batch_admission"
                if admission_field == "greedy_batch_admission"
                else "greedy_batch_admission"
            )
            admission = _mapping(
                row.get(admission_field),
                owner=f"history[{round_index - 1}].{admission_field}",
            )
            expected_admission_schema = (
                "sr_snake_greedy_batch_admission_v1"
                if admission_field == "greedy_batch_admission"
                else "sr_snake_combinatorial_batch_admission_v1"
            )
            if batch_maximum is None:
                raise CanonicalResumeError(
                    "Authenticated batch route lacks its cardinality limit."
                )
            admission_search_window_raw = admission.get(
                "search_window_size"
            )
            admission_search_window = (
                None
                if admission_search_window_raw is None
                else _integer(
                    admission_search_window_raw,
                    owner=(
                        f"history[{round_index - 1}] admission search "
                        "window"
                    ),
                    minimum=1,
                )
            )
            selected_record_ids = tuple(
                _text(
                    value,
                    owner=(
                        f"history[{round_index - 1}] selected record id"
                    ),
                )
                for value in _sequence(
                    admission.get("selected_record_ids"),
                    owner=(
                        f"history[{round_index - 1}] selected record ids"
                    ),
                )
            )
            if (
                row.get(incompatible_field) is not None
                or admission.get("schema") != expected_admission_schema
                or len(selected_labels) > batch_maximum
                or _integer(
                    admission.get("maximum_size"),
                    owner=(
                        f"history[{round_index - 1}] admission maximum "
                        "size"
                    ),
                    minimum=1,
                )
                != batch_maximum
                or admission_search_window != batch_search_window
                or tuple(
                    _text(
                        value,
                        owner=(
                            f"history[{round_index - 1}] admission "
                            "generator id"
                        ),
                    )
                    for value in _sequence(
                        admission.get("selected_generator_ids"),
                        owner=(
                            f"history[{round_index - 1}] admission "
                            "generator ids"
                        ),
                    )
                )
                != selected_generator_ids
                or tuple(
                    _integer(
                        value,
                        owner=(
                            f"history[{round_index - 1}] admission "
                            "original position"
                        ),
                        minimum=0,
                    )
                    for value in _sequence(
                        admission.get("selected_original_positions"),
                        owner=(
                            f"history[{round_index - 1}] admission "
                            "original positions"
                        ),
                    )
                )
                != original_positions
                or tuple(
                    _integer(
                        value,
                        owner=(
                            f"history[{round_index - 1}] admission "
                            "effective position"
                        ),
                        minimum=0,
                    )
                    for value in _sequence(
                        admission.get("selected_effective_positions"),
                        owner=(
                            f"history[{round_index - 1}] admission "
                            "effective positions"
                        ),
                    )
                )
                != effective_positions
                or len(selected_record_ids) != len(selected_labels)
                or len(set(selected_record_ids)) != len(selected_record_ids)
                or len(set(selected_generator_ids))
                != len(selected_generator_ids)
            ):
                raise CanonicalResumeError(
                    f"Accepted batch round {round_index} admission receipt "
                    "does not close to its selected members."
                )
            if route_family == "combinatorial_batch_response_snake":
                ranked_population = _integer(
                    admission.get("ranked_population_count"),
                    owner=(
                        f"history[{round_index - 1}] combinatorial ranked "
                        "population"
                    ),
                    minimum=1,
                )
                expected_ranked_window = min(
                    ranked_population,
                    (
                        batch_search_window
                        if batch_search_window is not None
                        else min(2 * batch_maximum, 10)
                    ),
                )
                if (
                    ranked_population < len(selected_labels)
                    or _integer(
                        admission.get("ranked_window_count"),
                        owner=(
                            f"history[{round_index - 1}] combinatorial "
                            "ranked window"
                        ),
                        minimum=1,
                    )
                    != expected_ranked_window
                ):
                    raise CanonicalResumeError(
                        f"Accepted combinatorial round {round_index} search "
                        "population disagrees with its route contract."
                    )
        for (
            label,
            generator_id,
            pool_index,
            position,
            retained_parent_owner,
        ) in zip(
            selected_labels,
            selected_generator_ids,
            selected_indices,
            effective_positions,
            selected_retained_parent_owners,
            strict=True,
        ):
            if pool_index >= pool_size:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} pool index is out of range."
                )
            if position > len(active_labels):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} insertion position is "
                    "outside the active prefix."
                )
            active_labels.insert(position, label)
            active_generator_ids.insert(position, generator_id)
            active_admission_rounds.insert(position, round_index)
            active_retained_parent_owners.insert(
                position,
                retained_parent_owner,
            )
            selection_counts[pool_index] += 1
            selected_parent_indices.append(pool_index)

        snapshots = [
            _mapping(
                feature.get("controller_snapshot"),
                owner=(
                    f"history[{round_index - 1}] controller_snapshot"
                ),
            )
            for feature in feature_rows
        ]
        if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
            raise CanonicalResumeError(
                f"Accepted round {round_index} batch members disagree on "
                "their controller snapshot."
            )
        maturity_snapshot = snapshots[0]

        prune = _mapping(
            row.get("post_admission_prune"),
            owner=f"history[{round_index - 1}].post_admission_prune",
        )
        if prune.get("enabled") is not pruning_active:
            raise CanonicalResumeError(
                f"Accepted round {round_index} prune mode disagrees with "
                "the authenticated route contract."
            )
        accepted_count = _integer(
            prune.get("accepted_count", 0),
            owner=f"history[{round_index - 1}] accepted prune count",
            minimum=0,
        )
        deleted_indices = tuple(
            _integer(
                value,
                owner=f"history[{round_index - 1}] deleted index",
                minimum=0,
            )
            for value in _sequence(
                prune.get("deleted_indices", []),
                owner=f"history[{round_index - 1}] deleted indices",
            )
        )
        deleted_labels = tuple(
            _text(
                value,
                owner=f"history[{round_index - 1}] deleted label",
            )
            for value in _sequence(
                prune.get("deleted_labels", []),
                owner=f"history[{round_index - 1}] deleted labels",
            )
        )
        if (
            accepted_count not in {0, 1}
            or accepted_count != len(deleted_indices)
            or accepted_count != len(deleted_labels)
        ):
            raise CanonicalResumeError(
                f"Accepted round {round_index} prune deletion receipt is "
                "incomplete."
            )
        if accepted_count == 1:
            deleted_index = deleted_indices[0]
            if (
                deleted_index >= len(active_labels)
                or deleted_labels[0]
                not in {
                    active_labels[deleted_index],
                    active_generator_ids[deleted_index],
                }
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} prune deletion does not "
                    "match the active prefix."
                )
            del active_labels[deleted_index]
            del active_generator_ids[deleted_index]
            del active_admission_rounds[deleted_index]
            del active_retained_parent_owners[deleted_index]
        if prune.get("enabled") is True and prune.get("trial_executed") is True:
            prune_update_count += 1
            trial_branch_id = _text(
                prune.get("trial_branch_id"),
                owner=f"history[{round_index - 1}] prune trial branch id",
            )
            prune_work = _mapping(
                prune.get("phase1_prune_exact_refit_work_accounting"),
                owner=(
                    f"history[{round_index - 1}] prune estimator work"
                ),
            )
            expected_classification = (
                "committed_prune"
                if accepted_count == 1
                else "discarded_prune"
            )
            if (
                prune_work.get("schema")
                != "sr_v4_prune_exact_refit_work_accounting_v1"
                or prune_work.get("classification")
                != expected_classification
                or prune_work.get("estimator_trial_branch_id")
                != trial_branch_id
                or prune_work.get("included_in_all_branch_search_work")
                is not True
                or prune_work.get("included_in_winning_lineage")
                != (accepted_count == 1)
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} prune estimator branch "
                    "classification is inconsistent."
                )
            executed_prune_branch_ids.append(trial_branch_id)
            if accepted_count == 1:
                accepted_prune_branch_ids.append(trial_branch_id)
        elif prune.get("trial_branch_id") is not None:
            raise CanonicalResumeError(
                f"Accepted round {round_index} carries a prune branch without "
                "an executed trial."
            )
        if prune.get("enabled") is True:
            prune_radius = _finite(
                prune.get("trust_radius_after"),
                owner=f"history[{round_index - 1}] prune trust radius",
            )
            prune_metric_damping = _finite(
                prune.get("metric_damping"),
                owner=f"history[{round_index - 1}] prune metric damping",
            )
            if prune_radius <= 0.0 or prune_metric_damping < 0.0:
                raise CanonicalResumeError(
                    f"Accepted round {round_index} prune trust state is "
                    "not construction-safe."
                )

        prefix = _validate_signed_prefix(
            row.get("active_prefix_checkpoint"),
            owner=f"history[{round_index - 1}].active_prefix_checkpoint",
            expected_round=round_index,
            expected_route_profile=expected_route_profile,
            expected_route_contract_sha256=(
                expected_route_contract_sha256
            ),
            total_qubits=total_qubits,
            allow_multi_term_operators=allow_multi_term_operators,
        )
        if list(prefix.operator_labels) != active_labels:
            raise CanonicalResumeError(
                f"Accepted round {round_index} admission/prune lineage does "
                "not close to its signed active prefix."
            )
        if [
            row[1] for row in prefix.operator_rows
        ] != active_generator_ids:
            raise CanonicalResumeError(
                f"Accepted round {round_index} generator lineage does not "
                "close to its signed active prefix."
            )
        prefix_operator_rows = [
            _mapping(
                value,
                owner=(
                    f"history[{round_index - 1}] signed active operator "
                    f"{position}"
                ),
            )
            for position, value in enumerate(
                _sequence(
                    prefix.payload.get("ordered_active_operators"),
                    owner=(
                        f"history[{round_index - 1}] signed active "
                        "operators"
                    ),
                )
            )
        ]
        for position, (
            expected_owner,
            signed_operator,
            parsed_operator,
        ) in enumerate(
            zip(
                active_retained_parent_owners,
                prefix_operator_rows,
                prefix.operator_rows,
                strict=True,
            )
        ):
            signed_owner_raw = signed_operator.get(
                "ra_retained_parent_owner"
            )
            if expected_owner is None:
                if signed_owner_raw is not None:
                    raise CanonicalResumeError(
                        f"Accepted round {round_index} signed operator "
                        f"{position} invents a retained-parent owner."
                    )
                continue
            signed_owner = _mapping(
                signed_owner_raw,
                owner=(
                    f"history[{round_index - 1}] signed operator "
                    f"{position} retained-parent owner"
                ),
            )
            if (
                signed_owner != expected_owner
                or parsed_operator[2]
                != expected_owner.get(
                    "parent_generator_identity"
                )
            ):
                raise CanonicalResumeError(
                    f"Accepted round {round_index} selected retained-parent "
                    "owner does not close to its signed active prefix."
                )
        if (
            prefix.ledger_receipt.get("branch_id") != expected_branch_id
            or prefix.ledger_receipt.get("parent_branch_id")
            != expected_parent_branch_id
        ):
            raise CanonicalResumeError(
                f"Accepted round {round_index} signed estimator receipt does "
                "not bind the winning history branch."
            )
        signed_prefixes.append(prefix)

    if maturity_snapshot is None:
        raise CanonicalResumeError(
            "Accepted checkpoint has no controller maturity snapshot."
        )
    maturity_snapshot = _validate_maturity_controller_snapshot(
        maturity_snapshot,
        owner="accepted controller maturity snapshot",
    )
    if pruning_active and (
        prune_radius is None or prune_metric_damping is None
    ):
        raise CanonicalResumeError(
            "Pruning-enabled accepted checkpoint lacks its trust state."
        )
    prune_state = (
        None
        if prune_radius is None or prune_metric_damping is None
        else CanonicalPruneStateHydration(
            radius=prune_radius,
            metric_damping=prune_metric_damping,
            update_count=prune_update_count,
        )
    )
    return (
        history,
        signed_prefixes,
        tuple(active_admission_rounds),
        tuple(selection_counts),
        tuple(selected_parent_indices),
        _freeze_json(maturity_snapshot),
        prune_state,
        tuple(executed_prune_branch_ids),
        tuple(accepted_prune_branch_ids),
    )


def _load_and_validate_ledger(
    *,
    source_path: Path,
    envelope: Mapping[str, Any],
    adapt: Mapping[str, Any],
    controller_round: int,
    beam_enabled: bool,
    winning_beam_branch_ids: tuple[str, ...],
    beam_diagnostics: Mapping[str, Any],
    executed_prune_branch_ids: tuple[str, ...],
    accepted_prune_branch_ids: tuple[str, ...],
    history_prefixes: Sequence[_ValidatedSignedPrefix],
    terminal_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], int, int, int]:
    expected_ledger_scope = (
        "all_executed_branches" if beam_enabled else "single_route"
    )
    pointer = _mapping(
        adapt.get("estimator_call_ledger_checkpoint"),
        owner="adapt_vqe.estimator_call_ledger_checkpoint",
    )
    if (
        pointer.get("schema") != _LEDGER_POINTER_SCHEMA
        or pointer.get("enabled") is not True
        or pointer.get("status") != "complete"
        or pointer.get("current_round_finalized") is not True
        or pointer.get("checkpoint_reason") != "iteration_done"
        or pointer.get("ledger_scope") != expected_ledger_scope
        or pointer.get("beam_enabled") is not beam_enabled
        or pointer.get("checkpoint_branch_policy")
        != (
            "canonical_terminal_winning_lineage"
            if beam_enabled
            else None
        )
        or pointer.get("branch_id") != adapt.get("branch_id")
        or pointer.get("parent_branch_id")
        != adapt.get("parent_branch_id")
        or _integer(
            pointer.get("checkpoint_depth"),
            owner="ledger pointer checkpoint_depth",
            minimum=1,
        )
        != controller_round
    ):
        raise CanonicalResumeError(
            "Accepted checkpoint estimator-ledger pointer is partial or "
            "unsupported."
        )
    top_checkpoint = _mapping(
        envelope.get("checkpoint"),
        owner="checkpoint",
    )
    if _mapping(
        top_checkpoint.get("estimator_call_ledger_checkpoint"),
        owner="checkpoint estimator-ledger pointer",
    ) != pointer:
        raise CanonicalResumeError(
            "Top-level and accepted-state estimator-ledger pointers disagree."
        )
    relative_name = _text(
        pointer.get("path"),
        owner="ledger pointer path",
    )
    relative_path = Path(relative_name)
    if (
        relative_path.is_absolute()
        or relative_path.name != relative_name
        or relative_path.parts != (relative_name,)
    ):
        raise CanonicalResumeError(
            "Estimator-ledger pointer must name one sibling file."
        )
    sidecar_path = source_path.with_name(relative_name)
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise CanonicalResumeError(
            "Estimator-ledger pointer does not resolve to a regular sibling "
            "file."
        )
    expected_sidecar_sha = _sha256(
        pointer.get("sha256"),
        owner="ledger pointer sha256",
    )
    if _file_sha256(sidecar_path) != expected_sidecar_sha:
        raise CanonicalResumeError(
            "Accepted checkpoint estimator-ledger sidecar SHA-256 mismatch."
        )
    sidecar = _load_json_object(
        sidecar_path,
        owner="accepted checkpoint estimator-ledger sidecar",
    )
    if (
        sidecar.get("schema") != _LEDGER_SIDECAR_SCHEMA
        or sidecar.get("no_credentials_serialized") is not True
        or sidecar.get("ledger_scope") != expected_ledger_scope
    ):
        raise CanonicalResumeError(
            "Accepted checkpoint estimator-ledger sidecar schema is "
            "unsupported."
        )
    sidecar_checkpoint = _mapping(
        sidecar.get("checkpoint"),
        owner="ledger sidecar checkpoint",
    )
    if (
        sidecar_checkpoint.get("reason") != "iteration_done"
        or sidecar_checkpoint.get("current_round_finalized") is not True
        or sidecar_checkpoint.get("ledger_scope")
        != expected_ledger_scope
        or sidecar_checkpoint.get("beam_enabled") is not beam_enabled
        or sidecar_checkpoint.get("checkpoint_branch_policy")
        != (
            "canonical_terminal_winning_lineage"
            if beam_enabled
            else None
        )
        or sidecar_checkpoint.get("branch_id") != adapt.get("branch_id")
        or sidecar_checkpoint.get("parent_branch_id")
        != adapt.get("parent_branch_id")
        or _integer(
            sidecar_checkpoint.get("depth"),
            owner="ledger sidecar depth",
            minimum=1,
        )
        != controller_round
    ):
        raise CanonicalResumeError(
            "Estimator-ledger sidecar is not closed at the accepted round."
        )
    ledger_payload = _mapping(
        sidecar.get("ledger"),
        owner="ledger sidecar ledger",
    )
    try:
        rebuilt_ledger = EstimatorCallLedger.from_payload(ledger_payload)
    except (TypeError, ValueError) as exc:
        raise CanonicalResumeError(
            "Accepted checkpoint estimator ledger failed deterministic "
            "reconstruction."
        ) from exc
    rebuilt_payload = rebuilt_ledger.to_payload()
    if rebuilt_payload != ledger_payload:
        raise CanonicalResumeError(
            "Accepted checkpoint estimator ledger is not its exact "
            "deterministic canonical payload."
        )
    occurrence = _mapping(
        rebuilt_payload.get("occurrence_summary"),
        owner="rebuilt occurrence summary",
    )
    unique = _mapping(
        rebuilt_payload.get("summary"),
        owner="rebuilt unique summary",
    )
    s_alg = _integer(
        occurrence.get("total_call_occurrences"),
        owner="ledger S_alg",
        minimum=0,
    )
    s_unique = _integer(
        unique.get("S_unique"),
        owner="ledger S_unique",
        minimum=0,
    )
    ledger_fingerprint = _sha256(
        rebuilt_payload.get("ledger_fingerprint"),
        owner="ledger fingerprint",
    )
    pointer_values = {
        "ledger_fingerprint": ledger_fingerprint,
        "raw_occurrence_count": s_alg,
        "S_alg": s_alg,
        "S_unique": s_unique,
    }
    for field, expected in pointer_values.items():
        if pointer.get(field) != expected or sidecar.get(field) != expected:
            raise CanonicalResumeError(
                "Estimator-ledger pointer/sidecar accounting does not close "
                f"for {field}."
            )
    if (
        pointer.get("ledger_schema") != rebuilt_payload.get("schema")
        or sidecar.get("unique_primitive_count")
        != unique.get("unique_primitive_count")
        or pointer.get("unique_primitive_count")
        != unique.get("unique_primitive_count")
    ):
        raise CanonicalResumeError(
            "Estimator-ledger pointer/sidecar primitive metadata does not "
            "close."
        )
    _mapping(
        sidecar.get("consumer_complete_projection"),
        owner="ledger sidecar consumer_complete_projection",
    )

    accounting = _mapping(
        adapt.get("estimator_call_accounting"),
        owner="adapt_vqe.estimator_call_accounting",
    )
    components = {
        component: _integer(
            _mapping(
                occurrence.get("component_occurrence_counts"),
                owner="ledger occurrence components",
            ).get(component, 0),
            owner=f"ledger {component}",
            minimum=0,
        )
        for component in S_ALG_COMPONENTS
    }
    if (
        accounting.get("schema") != _ACCOUNTING_SCHEMA
        or accounting.get("enabled") is not True
        or accounting.get("complete") is not True
        or accounting.get("exact_blockers") != []
        or accounting.get("definition")
        != "S_alg = N_H_outer + N_H_refit + N_grad + N_metric"
        or accounting.get("components") != components
        or accounting.get("S_alg") != s_alg
        or accounting.get("S_unique") != s_unique
        or adapt.get("S_alg") != s_alg
        or adapt.get("S_unique") != s_unique
        or adapt.get("S_alg_components") != components
        or sum(components.values()) != s_alg
    ):
        raise CanonicalResumeError(
            "Accepted checkpoint canonical estimator accounting is partial "
            "or does not close to the full ledger."
        )
    all_work = _mapping(
        accounting.get("all_branch_search_work"),
        owner="all-branch estimator work",
    )
    if (
        all_work.get("schema") != _WORK_SCHEMA
        or all_work.get("components") != components
        or all_work.get("S_alg") != s_alg
        or all_work.get("includes_rejected_evaluated_candidates") is not True
        or all_work.get("persistent_or_prior_run_cache_reductions_allowed")
        is not False
    ):
        raise CanonicalResumeError(
            "Accepted checkpoint all-branch S_alg work is incomplete."
        )
    occurrence_rows = [
        _mapping(row, owner="ledger occurrence")
        for row in _sequence(
            rebuilt_payload.get("occurrences"),
            owner="ledger occurrences",
        )
    ]
    ledger_branch_ids = {
        _text(row.get("branch_id"), owner="ledger occurrence branch_id")
        for row in occurrence_rows
        if row.get("branch_id") is not None
    }
    diagnostic_beam_child_ids = {
        _text(
            child.get("branch_id"),
            owner="beam diagnostic child branch id",
        )
        for round_row in (
            _sequence(
                beam_diagnostics.get("rounds"),
                owner="beam diagnostic rounds",
            )
            if beam_enabled
            else []
        )
        for child in (
            _mapping(
                child_raw,
                owner="beam diagnostic child",
            )
            for child_raw in _sequence(
                _mapping(
                    round_row,
                    owner="beam diagnostic round",
                ).get("children"),
                owner="beam diagnostic children",
            )
        )
    }
    if (
        not set(accepted_prune_branch_ids).issubset(
            executed_prune_branch_ids
        )
        or ledger_branch_ids
        != diagnostic_beam_child_ids.union(
            executed_prune_branch_ids
        )
    ):
        raise CanonicalResumeError(
            "Authenticated ledger branch identities do not equal the "
            "declared beam children and executed prune trials."
        )
    beam_child_delta_by_branch: dict[str, int] = {}
    if beam_enabled:
        initial_unbranched_s_alg = _integer(
            beam_diagnostics.get("initial_unbranched_s_alg"),
            owner="beam diagnostic initial unbranched S_alg",
            minimum=0,
        )
        raw_continuation = _mapping(
            adapt.get("continuation"),
            owner="adapt_vqe.continuation",
        )
        for receipt_index, receipt_raw in enumerate(
            _sequence(
                raw_continuation.get(
                    "all_active_prefix_estimator_ledger_receipts"
                ),
                owner="active-prefix estimator receipts",
            )
        ):
            receipt = _mapping(
                receipt_raw,
                owner=f"beam estimator receipt {receipt_index}",
            )
            if receipt.get("checkpoint_kind") != "post_admission_prune":
                continue
            branch_id = _text(
                receipt.get("branch_id"),
                owner=f"beam estimator receipt {receipt_index} branch_id",
            )
            if (
                branch_id not in diagnostic_beam_child_ids
                or branch_id in beam_child_delta_by_branch
            ):
                raise CanonicalResumeError(
                    "Beam estimator receipts do not uniquely bind every "
                    "diagnostic child."
                )
            receipt_delta = _integer(
                _mapping(
                    receipt.get("raw_occurrence_delta"),
                    owner=(
                        f"beam estimator receipt {receipt_index} raw delta"
                    ),
                ).get("total"),
                owner=f"beam branch {branch_id} receipt S_alg delta",
                minimum=0,
            )
            occurrence_start = _integer(
                receipt.get("occurrence_sequence_start_exclusive"),
                owner=f"beam branch {branch_id} receipt occurrence start",
                minimum=0,
            )
            occurrence_end = _integer(
                receipt.get("occurrence_sequence_end_inclusive"),
                owner=f"beam branch {branch_id} receipt occurrence end",
                minimum=0,
            )
            if (
                occurrence_end < occurrence_start
                or occurrence_end > len(occurrence_rows)
                or receipt_delta != occurrence_end - occurrence_start
            ):
                raise CanonicalResumeError(
                    "Beam estimator receipt interval does not close to its "
                    "raw occurrence delta."
                )
            receipt_occurrences = occurrence_rows[
                occurrence_start:occurrence_end
            ]
            unbranched_count = sum(
                row.get("branch_id") is None
                for row in receipt_occurrences
            )
            expected_unbranched = (
                initial_unbranched_s_alg
                if not beam_child_delta_by_branch
                else 0
            )
            if unbranched_count != expected_unbranched:
                raise CanonicalResumeError(
                    "Beam estimator receipt includes unexpected unbranched "
                    "work."
                )
            receipt_branch_ids = {
                _text(
                    row.get("branch_id"),
                    owner=(
                        f"beam branch {branch_id} receipt occurrence "
                        "branch_id"
                    ),
                )
                for row in receipt_occurrences
                if row.get("branch_id") is not None
            }
            if not receipt_branch_ids.issubset(
                {branch_id, *executed_prune_branch_ids}
            ):
                raise CanonicalResumeError(
                    "Beam estimator receipt contains work from an unrelated "
                    "branch."
                )
            beam_child_delta_by_branch[branch_id] = (
                receipt_delta - unbranched_count
            )
        if set(beam_child_delta_by_branch) != diagnostic_beam_child_ids:
            raise CanonicalResumeError(
                "Beam child diagnostics and estimator-receipt intervals "
                "do not close."
            )
        lineage_s_alg_by_branch: dict[str, int] = {}
        comparison_by_branch: dict[str, float] = {}
        history_rows = [
            _mapping(row, owner=f"beam history row {index}")
            for index, row in enumerate(
                _sequence(
                    adapt.get("history"),
                    owner="beam history",
                )
            )
        ]
        beam_weight = _finite(
            beam_diagnostics.get("s_alg_weight"),
            owner="beam diagnostic s_alg_weight",
        )
        for round_offset, round_raw in enumerate(
            _sequence(
                beam_diagnostics.get("rounds"),
                owner="beam diagnostic rounds",
            )
        ):
            round_index = round_offset + 1
            round_row = _mapping(
                round_raw,
                owner=f"beam diagnostic round {round_index}",
            )
            winning_branch_id = winning_beam_branch_ids[round_offset]
            winning_child: Mapping[str, Any] | None = None
            for child_raw in _sequence(
                round_row.get("children"),
                owner=f"beam diagnostic round {round_index} children",
            ):
                child = _mapping(
                    child_raw,
                    owner=f"beam diagnostic round {round_index} child",
                )
                branch_id = _text(
                    child.get("branch_id"),
                    owner=f"beam round {round_index} branch_id",
                )
                parent_raw = child.get("parent_branch_id")
                parent_id = (
                    None
                    if parent_raw is None
                    else _text(
                        parent_raw,
                        owner=f"beam round {round_index} parent_branch_id",
                    )
                )
                parent_lineage_s_alg = (
                    0
                    if parent_id is None
                    else lineage_s_alg_by_branch.get(parent_id)
                )
                if parent_lineage_s_alg is None:
                    raise CanonicalResumeError(
                        "Beam child refers to an unauthenticated parent "
                        "lineage."
                    )
                branch_occurrence = rebuilt_ledger.occurrence_summary(
                    branch_ids=(branch_id,),
                    include_unbranched=False,
                )
                direct_branch_delta = _integer(
                    branch_occurrence.get("total_call_occurrences"),
                    owner=f"beam branch {branch_id} S_alg delta",
                    minimum=0,
                )
                branch_delta = beam_child_delta_by_branch[branch_id]
                if direct_branch_delta > branch_delta:
                    raise CanonicalResumeError(
                        "Beam child receipt omits direct branch estimator "
                        "work."
                    )
                lineage_s_alg = parent_lineage_s_alg + branch_delta
                accepted_energy = _finite(
                    child.get("accepted_energy"),
                    owner=f"beam branch {branch_id} accepted_energy",
                )
                comparison_score = accepted_energy + (
                    beam_weight * lineage_s_alg
                )
                if (
                    child.get("fork_local_s_alg_delta") != branch_delta
                    or child.get("lineage_s_alg") != lineage_s_alg
                    or not math.isclose(
                        _finite(
                            child.get("comparison_score"),
                            owner=(
                                f"beam branch {branch_id} comparison_score"
                            ),
                        ),
                        comparison_score,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                ):
                    raise CanonicalResumeError(
                        "Beam child cost or comparison score does not close "
                        "to the authenticated branch ledger."
                    )
                lineage_s_alg_by_branch[branch_id] = lineage_s_alg
                comparison_by_branch[branch_id] = comparison_score
                if branch_id == winning_branch_id:
                    winning_child = child
            if winning_child is None:
                raise CanonicalResumeError(
                    f"Beam round {round_index} has no declared winning child."
                )
            history_row = history_rows[round_index - 1]
            if (
                not math.isclose(
                    _finite(
                        winning_child.get("accepted_energy"),
                        owner=(
                            f"beam round {round_index} winning energy"
                        ),
                    ),
                    _finite(
                        history_row.get("energy_after_opt"),
                        owner=f"beam history round {round_index} energy",
                    ),
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                or _sequence(
                    winning_child.get("selected_pool_indices"),
                    owner=(
                        f"beam round {round_index} winning pool indices"
                    ),
                )
                != _sequence(
                    history_row.get("selected_pool_indices"),
                    owner=(
                        f"beam history round {round_index} pool indices"
                    ),
                )
            ):
                raise CanonicalResumeError(
                    f"Beam round {round_index} winning child does not bind "
                    "the accepted history row."
                )
        if not math.isclose(
            _finite(
                beam_diagnostics.get("winning_comparison_score"),
                owner="beam terminal winning comparison score",
            ),
            comparison_by_branch[winning_beam_branch_ids[-1]],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise CanonicalResumeError(
                "Beam terminal comparison score does not match the "
                "authenticated winning child."
            )
    winning_accounting_branch_ids = tuple(
        sorted(
            set(winning_beam_branch_ids).union(
                accepted_prune_branch_ids
            )
        )
    )
    if not set(winning_accounting_branch_ids).issubset(
        ledger_branch_ids
    ):
        raise CanonicalResumeError(
            "Accepted winner lineage names estimator branches absent from "
            "the authenticated full ledger."
        )
    if ledger_branch_ids:
        winning_occurrence = rebuilt_ledger.occurrence_summary(
            branch_ids=winning_accounting_branch_ids,
            include_unbranched=True,
        )
        winning_unique = rebuilt_ledger.summary(
            branch_ids=winning_accounting_branch_ids,
            include_unbranched=True,
        )
    else:
        winning_occurrence = occurrence
        winning_unique = unique
    winning_components = {
        component: _integer(
            _mapping(
                winning_occurrence.get("component_occurrence_counts"),
                owner="winning occurrence components",
            ).get(component, 0),
            owner=f"winning {component}",
            minimum=0,
        )
        for component in S_ALG_COMPONENTS
    }
    winning_s_alg = _integer(
        winning_occurrence.get("total_call_occurrences"),
        owner="winning S_alg",
        minimum=0,
    )
    winning_s_unique = _integer(
        winning_occurrence.get("unique_primitive_count"),
        owner="winning S_unique",
        minimum=0,
    )
    winning_work = _mapping(
        accounting.get("winning_lineage"),
        owner="winning-lineage estimator work",
    )
    if (
        winning_work.get("schema") != _WORK_SCHEMA
        or winning_work.get("components") != winning_components
        or winning_work.get("S_alg") != winning_s_alg
        or winning_work.get("S_unique") != winning_s_unique
        or winning_work.get(
            "includes_rejected_evaluated_candidates"
        )
        is not True
        or winning_work.get(
            "persistent_or_prior_run_cache_reductions_allowed"
        )
        is not False
        or accounting.get(
            "winning_lineage_unique_primitive_diagnostic"
        )
        != winning_unique
    ):
        raise CanonicalResumeError(
            "Accepted checkpoint winner-lineage accounting does not close "
            "to the authenticated full ledger."
        )
    prune_accounting_raw = accounting.get(
        "recoverability_prune_accounting"
    )
    if prune_accounting_raw is not None:
        prune_accounting = _mapping(
            prune_accounting_raw,
            owner="recoverability prune accounting",
        )
        rejected_prune_branch_ids = [
            branch_id
            for branch_id in executed_prune_branch_ids
            if branch_id not in set(accepted_prune_branch_ids)
        ]
        if (
            prune_accounting.get("schema")
            != "sr_v4_prune_estimator_accounting_views_v1"
            or prune_accounting.get("accepted_trial_branch_ids")
            != list(accepted_prune_branch_ids)
            or prune_accounting.get("rejected_trial_branch_ids")
            != rejected_prune_branch_ids
            or prune_accounting.get("winning_branch_ids")
            != list(winning_accounting_branch_ids)
            or prune_accounting.get("all_work") != unique
            or prune_accounting.get("winning_lineage")
            != winning_unique
        ):
            raise CanonicalResumeError(
                "Recoverability-prune branch accounting does not close to "
                "the authenticated history and ledger."
            )
    elif executed_prune_branch_ids:
        raise CanonicalResumeError(
            "Executed recoverability-prune branches lack accounting views."
        )
    beam_accounting_raw = accounting.get("beam_accounting")
    beam_fork_local_lineage_s_alg = 0
    if beam_enabled:
        beam_branch_occurrence = rebuilt_ledger.occurrence_summary(
            branch_ids=winning_accounting_branch_ids,
            include_unbranched=False,
        )
        unbranched_occurrence = rebuilt_ledger.occurrence_summary(
            branch_ids=(),
            include_unbranched=True,
        )
        beam_fork_local_lineage_s_alg = _integer(
            beam_branch_occurrence.get("total_call_occurrences"),
            owner="beam fork-local winning-lineage S_alg",
            minimum=0,
        )
        unbranched_s_alg = _integer(
            unbranched_occurrence.get("total_call_occurrences"),
            owner="beam unbranched S_alg",
            minimum=0,
        )
        beam_accounting = _mapping(
            beam_accounting_raw,
            owner="beam accounting",
        )
        if (
            beam_accounting.get("schema")
            != "paper_i_fork_local_beam_accounting_v1"
            or beam_accounting.get(
                "all_executed_search_work_included"
            )
            is not True
            or beam_accounting.get("winning_branch_ids")
            != list(winning_beam_branch_ids)
            or beam_accounting.get("winning_lineage") != winning_work
            or beam_accounting.get("discarded_s_alg")
            != s_alg - winning_s_alg
            or beam_accounting.get("unchanged_parent_survival") is not False
            or beam_diagnostics.get("all_executed_s_alg") != s_alg
            or beam_diagnostics.get("initial_unbranched_s_alg")
            != unbranched_s_alg
            or beam_diagnostics.get("winning_lineage_s_alg")
            != beam_fork_local_lineage_s_alg
            or winning_s_alg
            != unbranched_s_alg + beam_fork_local_lineage_s_alg
        ):
            raise CanonicalResumeError(
                "Accepted checkpoint beam accounting does not close to its "
                "global and winner ledgers."
            )
    elif beam_accounting_raw is not None:
        raise CanonicalResumeError(
            "Non-beam accepted checkpoint carries beam accounting."
        )
    executed = _mapping(
        accounting.get("executed_occurrence_accounting"),
        owner="executed occurrence accounting",
    )
    if executed.get("all_execution") != occurrence:
        raise CanonicalResumeError(
            "Accepted checkpoint occurrence accounting differs from the "
            "authenticated full ledger."
        )
    continuation = _mapping(
        adapt.get("continuation"),
        owner="adapt_vqe.continuation",
    )
    if continuation.get("estimator_call_accounting") != accounting:
        raise CanonicalResumeError(
            "Accepted checkpoint continuation accounting disagrees with the "
            "accepted state."
        )

    receipts = [
        _validate_prefix_receipt(
            raw,
            owner=f"active prefix estimator receipt {index}",
        )
        for index, raw in enumerate(
            _sequence(
                continuation.get(
                    "all_active_prefix_estimator_ledger_receipts"
                ),
                owner="active-prefix estimator receipts",
            )
        )
    ]
    if not receipts:
        raise CanonicalResumeError(
            "Accepted checkpoint has no estimator-prefix closure receipts."
        )
    terminal_checkpoint_kind = _text(
        terminal_receipt.get("checkpoint_kind"),
        owner="terminal estimator receipt checkpoint_kind",
    )
    # Cadence checkpoints are published immediately after an accepted round.
    # Their pointer and sibling sidecar close the full ledger, but they reuse
    # the last signed round prefix and do not synthesize the later zero-delta
    # terminal receipt/closure summary.  Keep that exact single-route shape
    # distinct from finalized, beam, and prune-bearing checkpoints.
    round_finalized_current_checkpoint = bool(
        not beam_enabled
        and not executed_prune_branch_ids
        and not accepted_prune_branch_ids
        and terminal_checkpoint_kind == "post_admission_prune"
    )
    minimum_receipt_count = len(history_prefixes) + (
        0 if round_finalized_current_checkpoint else 1
    )
    if len(receipts) < minimum_receipt_count:
        raise CanonicalResumeError(
            "Accepted checkpoint estimator receipts omit a finalized history "
            "or terminal prefix."
        )
    for round_index, prefix in enumerate(history_prefixes, start=1):
        signed_receipt = prefix.ledger_receipt
        signed_sequence = _integer(
            signed_receipt.get("checkpoint_sequence"),
            owner=(
                f"history prefix {round_index} estimator checkpoint_sequence"
            ),
            minimum=1,
        )
        if (
            signed_sequence > len(receipts)
            or receipts[signed_sequence - 1] != signed_receipt
            or _integer(
                signed_receipt.get("outer_iteration"),
                owner=f"history prefix {round_index} outer_iteration",
                minimum=1,
            )
            != round_index
            or signed_receipt.get("checkpoint_kind")
            != prefix.payload.get("checkpoint_kind")
        ):
            raise CanonicalResumeError(
                f"Accepted round {round_index} signed estimator receipt is "
                "not the corresponding full-ledger closure receipt."
            )
    beam_children: list[tuple[int, str, str | None]] = []
    if beam_enabled:
        seen_beam_children: set[str] = set()
        for round_row_raw in _sequence(
            beam_diagnostics.get("rounds"),
            owner="beam diagnostics rounds",
        ):
            round_row = _mapping(
                round_row_raw,
                owner="beam diagnostics round",
            )
            child_round = _integer(
                round_row.get("controller_round"),
                owner="beam child controller round",
                minimum=1,
            )
            for child_raw in _sequence(
                round_row.get("children"),
                owner=f"beam round {child_round} children",
            ):
                child = _mapping(
                    child_raw,
                    owner=f"beam round {child_round} child",
                )
                child_id = _text(
                    child.get("branch_id"),
                    owner=f"beam round {child_round} child branch_id",
                )
                parent_raw = child.get("parent_branch_id")
                parent_id = (
                    None
                    if parent_raw is None
                    else _text(
                        parent_raw,
                        owner=(
                            f"beam round {child_round} child "
                            "parent_branch_id"
                        ),
                    )
                )
                if child_id in seen_beam_children:
                    raise CanonicalResumeError(
                        "Beam diagnostics repeat a child branch identity."
                    )
                seen_beam_children.add(child_id)
                beam_children.append(
                    (child_round, child_id, parent_id)
                )
    cumulative_raw = 0
    cumulative_unique = 0
    raw_components = {component: 0 for component in S_ALG_COMPONENTS}
    unique_components = {component: 0 for component in S_ALG_COMPONENTS}
    previous_outer_iteration = 0
    beam_child_receipt_index = 0
    first_beam_round = (
        controller_round - len(winning_beam_branch_ids) + 1
        if beam_enabled
        else controller_round + 1
    )
    for index, receipt in enumerate(receipts, start=1):
        if _integer(
            receipt.get("checkpoint_sequence"),
            owner="prefix receipt checkpoint_sequence",
            minimum=1,
        ) != index:
            raise CanonicalResumeError(
                "Estimator-prefix checkpoint sequences are not contiguous."
            )
        if _integer(
            receipt.get("occurrence_sequence_start_exclusive"),
            owner="prefix receipt occurrence start",
            minimum=0,
        ) != cumulative_raw:
            raise CanonicalResumeError(
                "Estimator-prefix occurrence intervals are not contiguous."
            )
        raw_delta = _mapping(
            receipt.get("raw_occurrence_delta"),
            owner="prefix raw occurrence delta",
        )
        executed_delta = _mapping(
            receipt.get("executed_query_delta"),
            owner="prefix executed-query delta",
        )
        unique_delta = _mapping(
            receipt.get("unique_primitive_delta"),
            owner="prefix unique primitive delta",
        )
        raw_delta_components = _mapping(
            raw_delta.get("components"),
            owner="prefix raw delta components",
        )
        executed_delta_components = _mapping(
            executed_delta.get("components"),
            owner="prefix executed-query delta components",
        )
        unique_delta_components = _mapping(
            unique_delta.get("components"),
            owner="prefix unique delta components",
        )
        raw_total_delta = _integer(
            raw_delta.get("total"),
            owner="prefix raw delta total",
            minimum=0,
        )
        unique_total_delta = _integer(
            unique_delta.get("S_unique"),
            owner="prefix unique delta total",
            minimum=0,
        )
        for component in S_ALG_COMPONENTS:
            raw_components[component] += _integer(
                raw_delta_components.get(component, 0),
                owner=f"prefix raw {component}",
                minimum=0,
            )
            unique_components[component] += _integer(
                unique_delta_components.get(component, 0),
                owner=f"prefix unique {component}",
                minimum=0,
            )
        if (
            sum(
                _integer(
                    raw_delta_components.get(component, 0),
                    owner=f"prefix raw {component}",
                    minimum=0,
                )
                for component in S_ALG_COMPONENTS
            )
            != raw_total_delta
            or sum(
                _integer(
                    unique_delta_components.get(component, 0),
                    owner=f"prefix unique {component}",
                    minimum=0,
                )
                for component in S_ALG_COMPONENTS
            )
            != unique_total_delta
        ):
            raise CanonicalResumeError(
                "Estimator-prefix component deltas do not close."
            )
        if (
            executed_delta_components != raw_delta_components
            or _integer(
                executed_delta.get("S_alg"),
                owner="prefix executed-query delta S_alg",
                minimum=0,
            )
            != raw_total_delta
        ):
            raise CanonicalResumeError(
                "Estimator-prefix executed-query and raw-occurrence deltas "
                "disagree."
            )
        cumulative_raw += raw_total_delta
        cumulative_unique += unique_total_delta
        if _integer(
            receipt.get("occurrence_sequence_end_inclusive"),
            owner="prefix receipt occurrence end",
            minimum=0,
        ) != cumulative_raw:
            raise CanonicalResumeError(
                "Estimator-prefix occurrence endpoint does not close."
            )
        cumulative_raw_receipt = _mapping(
            receipt.get("cumulative_raw_occurrences"),
            owner="prefix cumulative raw occurrences",
        )
        cumulative_executed_receipt = _mapping(
            receipt.get("cumulative_executed_queries"),
            owner="prefix cumulative executed queries",
        )
        cumulative_unique_receipt = _mapping(
            receipt.get("cumulative_unique_primitives"),
            owner="prefix cumulative unique primitives",
        )
        if (
            _mapping(
                cumulative_raw_receipt.get("components"),
                owner="prefix cumulative raw components",
            )
            != raw_components
            or _integer(
                cumulative_raw_receipt.get("total"),
                owner="prefix cumulative raw total",
                minimum=0,
            )
            != cumulative_raw
            or _mapping(
                cumulative_executed_receipt.get("components"),
                owner="prefix cumulative executed components",
            )
            != raw_components
            or _integer(
                cumulative_executed_receipt.get("S_alg"),
                owner="prefix cumulative executed S_alg",
                minimum=0,
            )
            != cumulative_raw
            or cumulative_executed_receipt.get("unit")
            != "executed_logical_scalar_estimator_invocation"
            or _mapping(
                cumulative_unique_receipt.get("components"),
                owner="prefix cumulative unique components",
            )
            != unique_components
            or _integer(
                cumulative_unique_receipt.get("S_unique"),
                owner="prefix cumulative unique S_unique",
                minimum=0,
            )
            != cumulative_unique
        ):
            raise CanonicalResumeError(
                "Estimator-prefix cumulative counters do not close to their "
                "deltas."
            )
        outer_iteration = _integer(
            receipt.get("outer_iteration"),
            owner="prefix receipt outer_iteration",
            minimum=1,
        )
        checkpoint_kind = _text(
            receipt.get("checkpoint_kind"),
            owner="prefix receipt checkpoint_kind",
        )
        if (
            outer_iteration < previous_outer_iteration
            or outer_iteration > controller_round
            or checkpoint_kind
            not in {
                "post_admission_prune",
                "terminal_post_final_refit_and_prune",
            }
            or receipt.get("runtime_estimator_occurrence_contract")
            != "all_instrumented_logical_scalar_estimator_calls_v1"
            or receipt.get(
                "physical_identity_collapse_is_diagnostic_only"
            )
            is not True
            or receipt.get("raw_occurrences_preserved") is not True
        ):
            raise CanonicalResumeError(
                "Estimator-prefix receipt provenance or accounting contract "
                "is incompatible with the canonical route."
            )
        receipt_branch_id = receipt.get("branch_id")
        receipt_parent_id = receipt.get("parent_branch_id")
        if not beam_enabled:
            if receipt_branch_id is not None or receipt_parent_id is not None:
                raise CanonicalResumeError(
                    "Non-beam estimator-prefix receipt carries branch "
                    "provenance."
                )
        elif checkpoint_kind == "post_admission_prune":
            if beam_child_receipt_index >= len(beam_children):
                raise CanonicalResumeError(
                    "Beam estimator receipts contain an undeclared child."
                )
            expected_child = beam_children[beam_child_receipt_index]
            beam_child_receipt_index += 1
            if (
                expected_child
                != (
                    outer_iteration,
                    receipt_branch_id,
                    receipt_parent_id,
                )
            ):
                raise CanonicalResumeError(
                    "Beam estimator receipt order or branch provenance "
                    "disagrees with diagnostics."
                )
        else:
            winner_index = outer_iteration - first_beam_round
            expected_terminal_branch = (
                None
                if winner_index < 0
                else winning_beam_branch_ids[winner_index]
            )
            expected_terminal_parent = (
                None
                if winner_index <= 0
                else winning_beam_branch_ids[winner_index - 1]
            )
            if (
                receipt_branch_id != expected_terminal_branch
                or receipt_parent_id != expected_terminal_parent
            ):
                raise CanonicalResumeError(
                    "Beam terminal estimator receipt does not bind the "
                    "winning lineage at that round."
                )
        previous_outer_iteration = outer_iteration
    if beam_child_receipt_index != len(beam_children):
        raise CanonicalResumeError(
            "Beam diagnostics contain children without global estimator "
            "prefix receipts."
        )
    ledger_unique_components = {
        component: int(
            _mapping(
                unique.get("components"),
                owner="ledger unique components",
            ).get(component, 0)
        )
        for component in S_ALG_COMPONENTS
    }
    terminal_closure_raw = continuation.get(
        "active_prefix_estimator_ledger_closure"
    )
    if terminal_closure_raw is None:
        if not round_finalized_current_checkpoint:
            raise CanonicalResumeError(
                "Accepted checkpoint estimator-prefix closure summary is "
                "incomplete or inconsistent."
            )
    else:
        terminal_closure = _mapping(
            terminal_closure_raw,
            owner="active-prefix estimator-ledger closure",
        )
        if (
            terminal_closure.get("schema")
            != "paper_i_active_prefix_estimator_ledger_closure_v1"
            or terminal_closure.get("enabled") is not True
            or terminal_closure.get("status") != "complete"
            or terminal_closure.get("passed") is not True
            or terminal_closure.get("receipt_count") != len(receipts)
            or terminal_closure.get("summed_raw_occurrences")
            != {
                "components": raw_components,
                "total": cumulative_raw,
            }
            or terminal_closure.get("terminal_raw_occurrences")
            != {
                "components": components,
                "total": s_alg,
            }
            or terminal_closure.get("summed_unique_primitives")
            != {
                "components": unique_components,
                "S_unique": cumulative_unique,
            }
            or terminal_closure.get("terminal_unique_primitives")
            != {
                "components": ledger_unique_components,
                "S_unique": s_unique,
            }
            or terminal_closure.get(
                "includes_discarded_branch_checkpoints"
            )
            is not beam_enabled
        ):
            raise CanonicalResumeError(
                "Accepted checkpoint estimator-prefix closure summary is "
                "incomplete or inconsistent."
            )
    if (
        cumulative_raw != s_alg
        or cumulative_unique != s_unique
        or raw_components != components
        or unique_components != ledger_unique_components
        or receipts[-1] != dict(terminal_receipt)
        or receipts[-1].get("checkpoint_kind")
        != (
            "post_admission_prune"
            if round_finalized_current_checkpoint
            else "terminal_post_final_refit_and_prune"
        )
        or _integer(
            receipts[-1].get("outer_iteration"),
            owner="terminal estimator receipt outer_iteration",
            minimum=1,
        )
        != controller_round
    ):
        raise CanonicalResumeError(
            "Accepted checkpoint active-prefix receipts do not close to the "
            "terminal full ledger."
        )
    return (
        rebuilt_payload,
        receipts,
        s_alg,
        s_unique,
        beam_fork_local_lineage_s_alg,
    )


def _estimator_prefix_cursor(
    *,
    receipt: Mapping[str, Any],
) -> Mapping[str, Any]:
    cumulative_raw = _mapping(
        receipt.get("cumulative_raw_occurrences"),
        owner="terminal cumulative raw occurrences",
    )
    cumulative_unique = _mapping(
        receipt.get("cumulative_unique_primitives"),
        owner="terminal cumulative unique primitives",
    )
    return _freeze_json(
        {
            "checkpoint_sequence": _integer(
                receipt.get("checkpoint_sequence"),
                owner="terminal checkpoint sequence",
                minimum=1,
            ),
            "raw_occurrence_count": _integer(
                cumulative_raw.get("total"),
                owner="terminal raw occurrence count",
                minimum=0,
            ),
            "unique_primitive_count": _integer(
                cumulative_unique.get("S_unique"),
                owner="terminal unique primitive count",
                minimum=0,
            ),
            "occurrence_components": {
                component: _integer(
                    _mapping(
                        cumulative_raw.get("components"),
                        owner="terminal raw components",
                    ).get(component, 0),
                    owner=f"terminal raw {component}",
                    minimum=0,
                )
                for component in S_ALG_COMPONENTS
            },
            "unique_components": {
                component: _integer(
                    _mapping(
                        cumulative_unique.get("components"),
                        owner="terminal unique components",
                    ).get(component, 0),
                    owner=f"terminal unique {component}",
                    minimum=0,
                )
                for component in S_ALG_COMPONENTS
            },
        }
    )


def _validate_checkpoint_envelope(
    *,
    envelope: Mapping[str, Any],
) -> tuple[dict[str, Any], int, bool]:
    if (
        envelope.get("schema_version") != _CHECKPOINT_SCHEMA
        or envelope.get("no_credentials_serialized") is not True
    ):
        raise CanonicalResumeError(
            "Accepted-state resume supports only the canonical direct "
            "checkpoint envelope."
    )
    checkpoint = _mapping(envelope.get("checkpoint"), owner="checkpoint")
    adapt = _mapping(envelope.get("adapt_vqe"), owner="adapt_vqe")
    beam_enabled = checkpoint.get("beam_enabled")
    if (
        checkpoint.get("reason") != "iteration_done"
        or checkpoint.get("complete") is not False
        or not isinstance(beam_enabled, bool)
        or adapt.get("adapt_beam_enabled") is not beam_enabled
        or adapt.get("partial_checkpoint") is not True
        or adapt.get("checkpoint_reason") != "iteration_done"
        or adapt.get("history_checkpoint_complete") is not True
        or adapt.get("success") is not False
        or adapt.get("stop_reason") not in {None, ""}
    ):
        raise CanonicalResumeError(
            "Accepted-state checkpoint is partial, branched, terminal, or "
            "otherwise incompatible with deterministic hydration."
        )
    if beam_enabled:
        for owner, branch in (
            ("checkpoint.branch_id", checkpoint.get("branch_id")),
            ("adapt_vqe.branch_id", adapt.get("branch_id")),
        ):
            _text(branch, owner=owner)
        for owner, parent in (
            (
                "checkpoint.parent_branch_id",
                checkpoint.get("parent_branch_id"),
            ),
            (
                "adapt_vqe.parent_branch_id",
                adapt.get("parent_branch_id"),
            ),
        ):
            if parent is not None:
                _text(parent, owner=owner)
        if (
            checkpoint.get("branch_id") != adapt.get("branch_id")
            or checkpoint.get("parent_branch_id")
            != adapt.get("parent_branch_id")
            or checkpoint.get("checkpoint_branch_policy")
            != "canonical_terminal_winning_lineage"
        ):
            raise CanonicalResumeError(
                "Beam accepted checkpoint top-level winner binding is "
                "inconsistent."
            )
    elif (
        checkpoint.get("checkpoint_branch_policy") not in {None, ""}
        or checkpoint.get("branch_id") is not None
        or checkpoint.get("parent_branch_id") is not None
        or adapt.get("branch_id") is not None
        or adapt.get("parent_branch_id") is not None
        or adapt.get("beam_search_diagnostics") is not None
    ):
        raise CanonicalResumeError(
            "Non-beam accepted checkpoint carries beam lineage state."
        )
    controller_round = _integer(
        checkpoint.get("depth"),
        owner="checkpoint.depth",
        minimum=1,
    )
    if (
        _integer(
            adapt.get("history_count"),
            owner="adapt_vqe.history_count",
            minimum=1,
        )
        != controller_round
    ):
        raise CanonicalResumeError(
            "Accepted-state checkpoint round and history count disagree."
        )
    for field in (
        "formal_manifold_runtime_checkpoint",
        "formal_manifold_warm_state_checkpoint",
        "formal_manifold_query_closure_checkpoint",
        "beam_replay_telemetry",
    ):
        if adapt.get(field) is not None:
            raise CanonicalResumeError(
                "Canonical accepted-state resume rejects compatibility "
                f"runtime field adapt_vqe.{field}."
            )
    final_refit = _mapping(
        adapt.get("final_full_refit"),
        owner="adapt_vqe.final_full_refit",
    )
    if (
        final_refit.get("attempted") is not False
        or final_refit.get("executed") is not False
        or final_refit.get("nfev") != 0
        or final_refit.get("skipped_reason")
        != "checkpoint_before_final_refit"
    ):
        raise CanonicalResumeError(
            "Accepted-state checkpoint final-refit boundary is incompatible."
        )
    return adapt, controller_round, beam_enabled


def load_canonical_accepted_state_resume(
    resume: AcceptedStateResume,
    *,
    expected_problem: ResolvedProblemContext,
    expected_route_profile: str,
    expected_route_contract_sha256: str,
) -> CanonicalAcceptedStateHydration:
    """Authenticate one direct-route accepted prefix for session hydration.

    The caller must independently reconstruct the accepted ansatz from
    ``operators`` and ``runtime_parameters`` against the resolved problem's
    reference state, then require its projective fingerprint to equal
    ``accepted_state_fingerprint`` before allowing another controller round.
    This reader supplies every deterministic replay input; it does not execute
    or silently weaken that final numerical replay guard.
    """

    if not isinstance(resume, AcceptedStateResume):
        raise TypeError("resume must be an AcceptedStateResume contract.")
    if not isinstance(expected_problem, ResolvedProblemContext):
        raise TypeError(
            "expected_problem must be a ResolvedProblemContext."
        )
    source_path = Path(resume.checkpoint_path)
    if source_path.is_symlink() or not source_path.is_file():
        raise CanonicalResumeError(
            "Accepted-state checkpoint must be a regular, non-symlink file."
        )
    source_sha = _file_sha256(source_path)
    if source_sha != resume.checkpoint_sha256:
        raise CanonicalResumeError(
            "AcceptedStateResume checkpoint SHA-256 mismatch."
        )
    envelope = _load_json_object(
        source_path,
        owner="canonical accepted checkpoint",
    )
    adapt, controller_round, beam_enabled = _validate_checkpoint_envelope(
        envelope=envelope
    )
    expected_digest = _sha256(
        expected_route_contract_sha256,
        owner="expected_route_contract_sha256",
    )
    route_family, route_contract = _validate_route_binding(
        envelope=envelope,
        adapt=adapt,
        expected_route_profile=expected_route_profile,
        expected_route_contract_sha256=expected_digest,
    )
    _validate_beam_route_binding(
        route_contract=route_contract,
        declared_beam_enabled=beam_enabled,
    )
    problem_sha, problem_binding_sha = _validate_problem_binding(
        envelope=envelope,
        adapt=adapt,
        route_contract=route_contract,
        expected_problem=expected_problem,
    )
    _validate_state_manifest(
        envelope.get("initial_state"),
        total_qubits=int(expected_problem.layout.total_qubits),
    )
    winning_beam_branch_ids, beam_diagnostics = (
        _validate_beam_declaration(
            adapt=adapt,
            route_contract=route_contract,
            controller_round=controller_round,
            beam_enabled=beam_enabled,
        )
    )
    (
        history,
        history_prefixes,
        admission_rounds,
        selection_counts,
        selected_parent_indices,
        maturity_snapshot,
        prune_state,
        executed_prune_branch_ids,
        accepted_prune_branch_ids,
    ) = _validate_history_and_admissions(
        adapt=adapt,
        controller_round=controller_round,
        route_family=(
            "singleton_response_snake"
            if route_family == "ra_adapt"
            else route_family
        ),
        route_contract=route_contract,
        winning_beam_branch_ids=winning_beam_branch_ids,
        expected_route_profile=expected_route_profile,
        expected_route_contract_sha256=expected_digest,
        total_qubits=int(expected_problem.layout.total_qubits),
        allow_multi_term_operators=(
            route_family == "ra_adapt"
            and _mapping(
                route_contract.get("semantic_invariants"),
                owner="RA route semantic invariants",
            ).get("candidate_representation")
            == "macro_generator_v1"
        ),
    )
    declared_prefixes = _sequence(
        adapt.get("active_prefix_checkpoints"),
        owner="adapt_vqe.active_prefix_checkpoints",
    )
    if declared_prefixes != [
        prefix.payload for prefix in history_prefixes
    ]:
        raise CanonicalResumeError(
            "Accepted checkpoint history and active-prefix list disagree."
        )
    continuation = _mapping(
        adapt.get("continuation"),
        owner="adapt_vqe.continuation",
    )
    if continuation.get("active_prefix_checkpoints") != declared_prefixes:
        raise CanonicalResumeError(
            "Accepted checkpoint continuation prefix list disagrees."
        )
    terminal_prefix = _validate_signed_prefix(
        adapt.get("terminal_active_prefix_checkpoint"),
        owner="adapt_vqe.terminal_active_prefix_checkpoint",
        expected_round=controller_round,
        expected_route_profile=expected_route_profile,
        expected_route_contract_sha256=expected_digest,
        total_qubits=int(expected_problem.layout.total_qubits),
        allow_multi_term_operators=(
            route_family == "ra_adapt"
            and _mapping(
                route_contract.get("semantic_invariants"),
                owner="RA route semantic invariants",
            ).get("candidate_representation")
            == "macro_generator_v1"
        ),
    )
    if (
        continuation.get("terminal_active_prefix_checkpoint")
        != terminal_prefix.payload
        or terminal_prefix.operator_labels
        != history_prefixes[-1].operator_labels
        or terminal_prefix.operator_rows
        != history_prefixes[-1].operator_rows
        or terminal_prefix.parameter_blocks
        != history_prefixes[-1].parameter_blocks
        or terminal_prefix.logical_parameters
        != history_prefixes[-1].logical_parameters
        or terminal_prefix.runtime_parameters
        != history_prefixes[-1].runtime_parameters
        or terminal_prefix.state_fingerprint
        != history_prefixes[-1].state_fingerprint
        or terminal_prefix.payload.get("parameterization")
        != history_prefixes[-1].payload.get("parameterization")
        or terminal_prefix.payload.get("ordered_active_operators")
        != history_prefixes[-1].payload.get("ordered_active_operators")
        or len(admission_rounds) != len(terminal_prefix.operator_rows)
    ):
        raise CanonicalResumeError(
            "Terminal signed accepted prefix disagrees with the finalized "
            "history state."
        )
    labels = tuple(
        _text(value, owner="adapt_vqe operator label")
        for value in _sequence(
            adapt.get("operators"),
            owner="adapt_vqe.operators",
        )
    )
    if (
        labels != terminal_prefix.operator_labels
        or _integer(
            adapt.get("ansatz_depth"),
            owner="adapt_vqe.ansatz_depth",
            minimum=1,
        )
        != len(labels)
        or _integer(
            _mapping(envelope.get("checkpoint"), owner="checkpoint").get(
                "ansatz_depth"
            ),
            owner="checkpoint.ansatz_depth",
            minimum=1,
        )
        != len(labels)
        or _sequence(
            adapt.get("optimal_point"),
            owner="adapt_vqe.optimal_point",
        )
        != list(terminal_prefix.runtime_parameters)
        or _sequence(
            adapt.get("logical_optimal_point"),
            owner="adapt_vqe.logical_optimal_point",
        )
        != list(terminal_prefix.logical_parameters)
        or adapt.get("parameterization")
        != terminal_prefix.payload["parameterization"]
    ):
        raise CanonicalResumeError(
            "Accepted-state optimizer coordinates disagree with the terminal "
            "signed prefix."
        )
    accepted_energy = _finite(
        adapt.get("energy"),
        owner="adapt_vqe.energy",
    )
    last_energy = _finite(
        history[-1].get("energy_after_opt"),
        owner="terminal history energy_after_opt",
    )
    if not math.isclose(
        accepted_energy,
        last_energy,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise CanonicalResumeError(
            "Accepted-state energy disagrees with finalized history."
        )
    nfev_total = _integer(
        adapt.get("nfev_total"),
        owner="adapt_vqe.nfev_total",
        minimum=0,
    )
    if nfev_total != _integer(
        history[-1].get("nfev_total_after_step"),
        owner="terminal history nfev_total_after_step",
        minimum=0,
    ):
        raise CanonicalResumeError(
            "Accepted-state optimizer work disagrees with finalized history."
        )
    (
        ledger_payload,
        ledger_receipts,
        s_alg,
        s_unique,
        beam_fork_local_lineage_s_alg,
    ) = (
        _load_and_validate_ledger(
            source_path=source_path,
            envelope=envelope,
            adapt=adapt,
            controller_round=controller_round,
            beam_enabled=beam_enabled,
            winning_beam_branch_ids=winning_beam_branch_ids,
            beam_diagnostics=beam_diagnostics,
            executed_prune_branch_ids=executed_prune_branch_ids,
            accepted_prune_branch_ids=accepted_prune_branch_ids,
            history_prefixes=history_prefixes,
            terminal_receipt=terminal_prefix.ledger_receipt,
        )
    )
    operators = tuple(
        CanonicalOperatorHydration(
            active_position=position,
            label=row[0],
            generator_id=row[1],
            parent_generator_id=row[2],
            execution_mode=row[3],
            admission_round=admission_rounds[position],
            runtime_terms=row[4],
        )
        for position, row in enumerate(terminal_prefix.operator_rows)
    )
    allow_repeats = bool(
        _mapping(
            route_contract.get("execution_settings"),
            owner="route execution settings",
        ).get("adapt_allow_repeats", False)
    )
    available_indices = tuple(
        index
        for index, count in enumerate(selection_counts)
        if allow_repeats or count == 0
    )
    trust_state = _mapping(
        adapt.get("route_a_trust_region_state"),
        owner="adapt_vqe.route_a_trust_region_state",
    )
    last_trust_update = trust_state.get("last_update")
    if last_trust_update is not None:
        _mapping(
            last_trust_update,
            owner="adapt_vqe.route_a_trust_region_state.last_update",
        )
    if (
        trust_state.get("schema") != "route_a_trust_region_state_v1"
        or _finite(
            trust_state.get("radius"),
            owner="route trust radius",
        )
        <= 0.0
        or _finite(
            trust_state.get("reference_radius"),
            owner="route trust reference radius",
        )
        <= 0.0
        or _integer(
            trust_state.get("update_count"),
            owner="route trust update_count",
            minimum=0,
        )
        < 0
    ):
        raise CanonicalResumeError(
            "Accepted-state route trust state is incomplete."
        )
    _text(
        trust_state.get("initialization_reason"),
        owner="route trust initialization_reason",
    )
    return CanonicalAcceptedStateHydration(
        source_path=source_path,
        source_sha256=source_sha,
        problem_request_sha256=problem_sha,
        problem_binding_sha256=problem_binding_sha,
        route_family=route_family,
        route_profile=expected_route_profile,
        route_contract_sha256=expected_digest,
        winning_beam_branch_ids=winning_beam_branch_ids,
        beam_search_diagnostics=_freeze_json(beam_diagnostics),
        beam_fork_local_lineage_s_alg=(
            beam_fork_local_lineage_s_alg
        ),
        controller_round=controller_round,
        accepted_energy=accepted_energy,
        accepted_state_fingerprint=(
            terminal_prefix.state_fingerprint
        ),
        operators=operators,
        parameter_blocks=terminal_prefix.parameter_blocks,
        logical_parameters=terminal_prefix.logical_parameters,
        runtime_parameters=terminal_prefix.runtime_parameters,
        selection_counts_by_pool_index=selection_counts,
        available_pool_indices=available_indices,
        selected_parent_pool_indices=selected_parent_indices,
        nfev_total=nfev_total,
        s_alg=s_alg,
        s_unique=s_unique,
        estimator_prefix_checkpoint_cursor=_estimator_prefix_cursor(
            receipt=terminal_prefix.ledger_receipt
        ),
        route_a_trust_region_state=_freeze_json(trust_state),
        prune_trust_state=prune_state,
        maturity_controller_snapshot=maturity_snapshot,
        history=tuple(_freeze_json(row) for row in history),
        active_prefix_estimator_receipts=tuple(
            _freeze_json(receipt) for receipt in ledger_receipts
        ),
        parameterization=_freeze_json(
            terminal_prefix.payload["parameterization"]
        ),
        route_contract=_freeze_json(route_contract),
        estimator_call_ledger_payload=_freeze_json(ledger_payload),
        terminal_signed_checkpoint=_freeze_json(
            terminal_prefix.payload
        ),
    )


__all__ = [
    "CanonicalAcceptedStateHydration",
    "CanonicalOperatorHydration",
    "CanonicalParameterBlockHydration",
    "CanonicalPruneStateHydration",
    "CanonicalResumeError",
    "CanonicalRuntimePauliTermHydration",
    "load_canonical_accepted_state_resume",
]
