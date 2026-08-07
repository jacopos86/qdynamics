"""Pure Stage-B core for SR-SNAKE modeled-local-minimum exploration.

The module owns no circuit, optimizer, or mutable pipeline state.  It consumes
state-bound certificates and implements only the mathematical Stage-B gates:

* complete exposed-family PSD/redundancy eligibility;
* canonical countable action enumeration and Calkin--Wilf radii;
* exact Fubini--Study exclusion and certified path/barrier provenance;
* bounded barrier/distance utility with exact symbolic fair entitlements;
* activation-frozen move/refinement service;
* disposable, seed-preserving Powell promotion; and
* the atomic incumbent/working-state transaction.

Missing evidence is serviced as refinement.  Failed, stale, nonfinite,
non-simultaneous, physically collapsed, or provenance-mismatched evidence is
invalid and cannot create a move.  Public state is frozen and serialized with
canonical JSON.  Checkpoints are content-addressed and reject tampering.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from fractions import Fraction
import hashlib
import json
import math
from typing import TypeAlias

from pipelines.static_adapt.sr_snake_escape_controller import (
    PsdCertificate,
    QuotientRedundantCertificate,
    ReachablePopulationAudit,
    reachable_population_digest,
)


_FS_DIAMETER = math.pi / 2.0
_RUNTIME_SCHEMA = "sr_snake_modeled_minimum_core_scheduler_state_v2"
_CHECKPOINT_SCHEMA = "sr_snake_modeled_minimum_core_scheduler_checkpoint_v2"
_CHECKPOINT_SCOPE = "pure_core_scheduler_only"
_ACTION_RECEIPT_SCHEMA = "sr_snake_modeled_minimum_action_receipt_v1"
ACTION_INDEX_SCHEMA = "sr_snake_action_index_cantor_calkin_wilf_v1"


class _DeterministicSerializable:
    def to_dict(self) -> dict[str, object]:  # pragma: no cover - interface
        raise NotImplementedError

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(name: str, value: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be nonempty.")
    return result


def _positive_index(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer.")
    result = int(value)
    if result != value or result <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _nonnegative_index(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result != value or result < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result


def _encode_nonnegative_integer(value: int) -> str:
    """Serialize arbitrary-size integers without decimal digit-limit failures."""

    resolved = _nonnegative_index("serialized_integer", value)
    return f"0x{resolved:x}"


def _decode_nonnegative_integer(name: str, value: object) -> int:
    """Decode the canonical lower-case hexadecimal integer representation."""

    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{name} must use canonical hexadecimal serialization.")
    digits = value[2:]
    if not digits or any(character not in "0123456789abcdef" for character in digits):
        raise ValueError(f"{name} must use canonical hexadecimal serialization.")
    if len(digits) > 1 and digits[0] == "0":
        raise ValueError(f"{name} has a noncanonical leading zero.")
    result = int(digits, 16)
    if value != _encode_nonnegative_integer(result):
        raise ValueError(f"{name} is not canonically serialized.")
    return result


def _decode_positive_integer(name: str, value: object) -> int:
    result = _decode_nonnegative_integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _finite(name: str, value: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite real data.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative(name: str, value: float) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive(name: str, value: float) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _is_finite_real(value: object) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _log_positive_integer(value: int) -> float:
    integer = _positive_index("positive_integer", value)
    result = math.log(integer)
    if not math.isfinite(result):
        raise ValueError("positive integer logarithm is not finite.")
    return result


def _finite_fraction(name: str, value: float) -> Fraction:
    """Return the exact rational represented by one finite machine float."""

    return Fraction.from_float(_finite(name, value))


def _fraction_log(value: Fraction) -> float:
    if value <= 0:
        raise ValueError("fraction logarithm requires positive data.")
    return _log_positive_integer(value.numerator) - _log_positive_integer(
        value.denominator
    )


def _fraction_to_dict(value: Fraction) -> dict[str, str]:
    return {
        "numerator": _encode_nonnegative_integer(value.numerator),
        "denominator": _encode_nonnegative_integer(value.denominator),
    }


def _fraction_from_dict(name: str, data: object) -> Fraction:
    if not isinstance(data, dict):
        raise ValueError(f"{name} must be a rational object.")
    numerator = _decode_positive_integer(f"{name}.numerator", data.get("numerator"))
    denominator = _decode_positive_integer(
        f"{name}.denominator", data.get("denominator")
    )
    value = Fraction(numerator, denominator)
    if _fraction_to_dict(value) != data:
        raise ValueError(f"{name} is not reduced canonical rational data.")
    return value


def _log_logistic(log_x: float) -> float:
    """Return ``log(x/(1+x))`` from ``log(x)`` without underflow."""

    log_value = _finite("log_x", log_x)
    if log_value >= 0.0:
        return -math.log1p(math.exp(-log_value))
    return log_value - math.log1p(math.exp(log_value))


class CertificateState(str, Enum):
    PASSED = "passed"
    UNRESOLVED = "unresolved"
    FAILED = "failed"


class ResolutionKind(str, Enum):
    CERTIFIED = "certified"
    REFINEMENT = "refinement"
    INVALID = "invalid"


class PathOrientation(IntEnum):
    NEGATIVE = -1
    POSITIVE = 1


class ServiceTag(str, Enum):
    MOVE = "move"
    REFINEMENT = "ref"


class ControllerMode(str, Enum):
    ORDINARY = "ord"
    ESCAPE_STATIONARY = "esc:stat"
    ESCAPE_UNRESOLVED = "esc:unres"
    ESCAPE_FUNNEL = "esc:funnel"
    EXPLORE = "explore"


@dataclass(frozen=True)
class EligibilityStateToken(_DeterministicSerializable):
    """Full fixed-state/fixed-population token for Stage-B evidence."""

    working_state_fingerprint: str
    reachable_record_ids: tuple[str, ...]
    reachable_population_digest: str
    comparison_epoch: str
    support_provenance_digest: str
    trust_provenance_digest: str
    trust_radius: float
    stationarity_margin: float

    def __post_init__(self) -> None:
        for name in (
            "working_state_fingerprint",
            "reachable_population_digest",
            "comparison_epoch",
            "support_provenance_digest",
            "trust_provenance_digest",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        records = tuple(str(value) for value in self.reachable_record_ids)
        if not records or any(not value.strip() for value in records):
            raise ValueError("reachable_record_ids must be nonempty identifiers.")
        if len(set(records)) != len(records):
            raise ValueError("reachable_record_ids must be unique and ordered.")
        object.__setattr__(self, "reachable_record_ids", records)
        expected_population_digest = reachable_population_digest(records)
        if self.reachable_population_digest != expected_population_digest:
            raise ValueError(
                "reachable_population_digest does not match ordered record IDs."
            )
        _positive("trust_radius", self.trust_radius)
        if _finite("stationarity_margin", self.stationarity_margin) > 0.0:
            raise ValueError("stationarity_margin must certify stationarity.")

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "working_state_fingerprint": self.working_state_fingerprint,
            "reachable_record_ids": list(self.reachable_record_ids),
            "reachable_population_digest": self.reachable_population_digest,
            "comparison_epoch": self.comparison_epoch,
            "support_provenance_digest": self.support_provenance_digest,
            "trust_provenance_digest": self.trust_provenance_digest,
            "trust_radius": self.trust_radius,
            "stationarity_margin": self.stationarity_margin,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EligibilityStateToken":
        return cls(
            working_state_fingerprint=str(data["working_state_fingerprint"]),
            reachable_record_ids=tuple(data["reachable_record_ids"]),
            reachable_population_digest=str(data["reachable_population_digest"]),
            comparison_epoch=str(data["comparison_epoch"]),
            support_provenance_digest=str(data["support_provenance_digest"]),
            trust_provenance_digest=str(data["trust_provenance_digest"]),
            trust_radius=float(data["trust_radius"]),
            stationarity_margin=float(data["stationarity_margin"]),
        )


@dataclass(frozen=True)
class ExposedFamilyEligibility(_DeterministicSerializable):
    """Result of the state-bound reachable-family PSD/redundancy gate."""

    eligible: bool
    reason: str
    reachable_record_ids: tuple[str, ...] = ()
    psd_record_ids: tuple[str, ...] = ()
    redundant_record_ids: tuple[str, ...] = ()
    working_state_fingerprint: str | None = None
    reachable_population_digest: str | None = None
    comparison_epoch: str | None = None
    support_provenance_digest: str | None = None
    trust_provenance_digest: str | None = None
    trust_radius: float | None = None
    stationarity_margin: float | None = None

    def __post_init__(self) -> None:
        _nonempty("reason", self.reason)
        for name in (
            "reachable_record_ids",
            "psd_record_ids",
            "redundant_record_ids",
        ):
            object.__setattr__(
                self,
                name,
                tuple(str(value) for value in getattr(self, name)),
            )
        if self.eligible:
            reachable = self.reachable_record_ids
            psd = self.psd_record_ids
            redundant = self.redundant_record_ids
            if not reachable or len(set(reachable)) != len(reachable):
                raise ValueError(
                    "eligible result requires unique ordered reachable records."
                )
            if set(psd) & set(redundant):
                raise ValueError("PSD and redundant record sets must be disjoint.")
            if set(psd) | set(redundant) != set(reachable):
                raise ValueError(
                    "eligible result must classify every reachable record."
                )
            _ = self.state_token

    @property
    def state_token(self) -> EligibilityStateToken:
        if not self.eligible:
            raise ValueError("ineligible audit has no Stage-B state token.")
        values = (
            self.working_state_fingerprint,
            self.reachable_population_digest,
            self.comparison_epoch,
            self.support_provenance_digest,
            self.trust_provenance_digest,
            self.trust_radius,
            self.stationarity_margin,
        )
        if any(value is None for value in values):
            raise ValueError(
                "eligible exposed-family result requires a complete state token."
            )
        return EligibilityStateToken(
            working_state_fingerprint=str(self.working_state_fingerprint),
            reachable_record_ids=self.reachable_record_ids,
            reachable_population_digest=str(self.reachable_population_digest),
            comparison_epoch=str(self.comparison_epoch),
            support_provenance_digest=str(self.support_provenance_digest),
            trust_provenance_digest=str(self.trust_provenance_digest),
            trust_radius=float(self.trust_radius),
            stationarity_margin=float(self.stationarity_margin),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "eligible": self.eligible,
            "reason": self.reason,
            "reachable_record_ids": list(self.reachable_record_ids),
            "psd_record_ids": list(self.psd_record_ids),
            "redundant_record_ids": list(self.redundant_record_ids),
            "working_state_fingerprint": self.working_state_fingerprint,
            "reachable_population_digest": self.reachable_population_digest,
            "comparison_epoch": self.comparison_epoch,
            "support_provenance_digest": self.support_provenance_digest,
            "trust_provenance_digest": self.trust_provenance_digest,
            "trust_radius": self.trust_radius,
            "stationarity_margin": self.stationarity_margin,
        }


def assess_exposed_family_psd(
    audit: ReachablePopulationAudit | None,
) -> ExposedFamilyEligibility:
    """Require complete PSD/redundancy and state stationarity at one token."""

    if audit is None:
        return ExposedFamilyEligibility(
            eligible=False,
            reason="reachable_population_audit_missing",
        )
    reachable = tuple(audit.reachable_record_ids)
    if not reachable:
        return ExposedFamilyEligibility(
            eligible=False,
            reason="reachable_population_is_empty",
        )
    if not audit.complete:
        return ExposedFamilyEligibility(
            eligible=False,
            reason="reachable_population_audit_incomplete",
            reachable_record_ids=reachable,
        )
    by_record = {
        certificate.record_id: certificate for certificate in audit.certificates
    }
    psd = tuple(
        record_id
        for record_id in reachable
        if isinstance(by_record[record_id], PsdCertificate)
    )
    redundant = tuple(
        record_id
        for record_id in reachable
        if isinstance(by_record[record_id], QuotientRedundantCertificate)
    )
    if len(psd) + len(redundant) != len(reachable):
        return ExposedFamilyEligibility(
            eligible=False,
            reason="reachable_population_not_all_psd_or_redundant",
            reachable_record_ids=reachable,
            psd_record_ids=psd,
            redundant_record_ids=redundant,
        )
    if not audit.state_stationarity_certified:
        return ExposedFamilyEligibility(
            eligible=False,
            reason="state_stationarity_certificate_missing_or_population_stale",
            reachable_record_ids=reachable,
            psd_record_ids=psd,
            redundant_record_ids=redundant,
        )
    state = audit.state_stationarity
    assert state is not None
    return ExposedFamilyEligibility(
        eligible=True,
        reason=(
            "complete_reachable_population_is_psd_or_redundant_and_"
            "state_stationary"
        ),
        reachable_record_ids=reachable,
        psd_record_ids=psd,
        redundant_record_ids=redundant,
        working_state_fingerprint=state.state_fingerprint,
        reachable_population_digest=state.reachable_population_digest,
        comparison_epoch=state.comparison_epoch,
        support_provenance_digest=state.support_provenance_digest,
        trust_provenance_digest=state.trust_provenance_digest,
        trust_radius=float(state.trust_radius),
        stationarity_margin=float(state.stationarity_margin),
    )


@dataclass(frozen=True)
class PositiveRational(_DeterministicSerializable):
    """Reduced positive rational and its Calkin--Wilf index."""

    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        numerator = _positive_index("numerator", self.numerator)
        denominator = _positive_index("denominator", self.denominator)
        if math.gcd(numerator, denominator) != 1:
            raise ValueError("positive rational must be in reduced form.")

    @property
    def index(self) -> int:
        return calkin_wilf_index(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, object]:
        return {
            "numerator": _encode_nonnegative_integer(self.numerator),
            "denominator": _encode_nonnegative_integer(self.denominator),
            "calkin_wilf_index": _encode_nonnegative_integer(self.index),
        }


def calkin_wilf_rational(index: int) -> PositiveRational:
    """Map a positive index to the canonical positive rational ``q_j``."""

    resolved = _positive_index("calkin_wilf_index", index)
    numerator = 1
    denominator = 1
    for bit in bin(resolved)[3:]:
        if bit == "0":
            denominator = numerator + denominator
        else:
            numerator = numerator + denominator
    return PositiveRational(numerator=numerator, denominator=denominator)


def calkin_wilf_index(numerator: int, denominator: int) -> int:
    """Invert :func:`calkin_wilf_rational` for a reduced positive rational."""

    p = _positive_index("numerator", numerator)
    q = _positive_index("denominator", denominator)
    if math.gcd(p, q) != 1:
        raise ValueError("positive rational must be in reduced form.")
    reverse_runs: list[tuple[int, int]] = []
    while p != q:
        if p < q:
            count = (q - 1) // p
            reverse_runs.append((0, count))
            q -= count * p
        else:
            count = (p - 1) // q
            reverse_runs.append((1, count))
            p -= count * q
    index = 1
    for bit, count in reversed(reverse_runs):
        for _ in range(count):
            index = (index << 1) | bit
    return index


def _cantor_pair(first: int, second: int) -> int:
    a = _nonnegative_index("first_pair_coordinate", first)
    b = _nonnegative_index("second_pair_coordinate", second)
    diagonal = a + b
    return diagonal * (diagonal + 1) // 2 + b


def _cantor_unpair(value: int) -> tuple[int, int]:
    paired = _nonnegative_index("paired_coordinate", value)
    diagonal = (math.isqrt(8 * paired + 1) - 1) // 2
    offset = paired - diagonal * (diagonal + 1) // 2
    return diagonal - offset, offset


def canonical_action_index(
    *,
    record_order: int,
    record_count: int,
    orientation: PathOrientation | int,
    radius_index: int,
    path_index: int,
) -> int:
    """Canonical bijection from ``(r, sigma, j, ell)`` to positive integers."""

    count = _positive_index("record_count", record_count)
    order = _positive_index("record_order", record_order)
    if order > count:
        raise ValueError("record_order cannot exceed record_count.")
    sign = PathOrientation(orientation)
    sign_offset = 0 if sign is PathOrientation.NEGATIVE else 1
    lane = 2 * (order - 1) + sign_offset
    pair = _cantor_pair(
        _positive_index("radius_index", radius_index) - 1,
        _positive_index("path_index", path_index) - 1,
    )
    return pair * (2 * count) + lane + 1


@dataclass(frozen=True)
class CanonicalActionCoordinates(_DeterministicSerializable):
    record_id: str
    record_order: int
    record_count: int
    orientation: PathOrientation
    radius_index: int
    path_index: int

    def __post_init__(self) -> None:
        _nonempty("record_id", self.record_id)
        count = _positive_index("record_count", self.record_count)
        order = _positive_index("record_order", self.record_order)
        if order > count:
            raise ValueError("record_order cannot exceed record_count.")
        object.__setattr__(self, "orientation", PathOrientation(self.orientation))
        _positive_index("radius_index", self.radius_index)
        _positive_index("path_index", self.path_index)

    @property
    def action_index(self) -> int:
        return canonical_action_index(
            record_order=self.record_order,
            record_count=self.record_count,
            orientation=self.orientation,
            radius_index=self.radius_index,
            path_index=self.path_index,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "record_order": self.record_order,
            "record_count": self.record_count,
            "orientation": int(self.orientation),
            "radius_index": _encode_nonnegative_integer(self.radius_index),
            "path_index": _encode_nonnegative_integer(self.path_index),
            "action_index": _encode_nonnegative_integer(self.action_index),
            "action_index_schema": ACTION_INDEX_SCHEMA,
        }


def inverse_action_index(
    action_index: int,
    reachable_record_ids: tuple[str, ...],
) -> CanonicalActionCoordinates:
    """Invert the canonical action bijection for one finite record order."""

    index = _positive_index("action_index", action_index)
    records = tuple(str(value) for value in reachable_record_ids)
    if not records or len(set(records)) != len(records):
        raise ValueError("reachable_record_ids must be nonempty and unique.")
    pair, lane = divmod(index - 1, 2 * len(records))
    radius_zero, path_zero = _cantor_unpair(pair)
    record_order = lane // 2 + 1
    orientation = (
        PathOrientation.NEGATIVE
        if lane % 2 == 0
        else PathOrientation.POSITIVE
    )
    return CanonicalActionCoordinates(
        record_id=records[record_order - 1],
        record_order=record_order,
        record_count=len(records),
        orientation=orientation,
        radius_index=radius_zero + 1,
        path_index=path_zero + 1,
    )


@dataclass(frozen=True)
class PathActionKey(_DeterministicSerializable):
    """Canonical ``(r, sigma, j, ell)`` identity; index is always derived."""

    record_id: str
    record_order: int
    record_count: int
    orientation: PathOrientation
    radius_index: int
    path_index: int

    def __post_init__(self) -> None:
        _ = CanonicalActionCoordinates(
            record_id=self.record_id,
            record_order=self.record_order,
            record_count=self.record_count,
            orientation=self.orientation,
            radius_index=self.radius_index,
            path_index=self.path_index,
        )
        object.__setattr__(self, "record_id", _nonempty("record_id", self.record_id))
        object.__setattr__(self, "orientation", PathOrientation(self.orientation))

    @property
    def action_index(self) -> int:
        return canonical_action_index(
            record_order=self.record_order,
            record_count=self.record_count,
            orientation=self.orientation,
            radius_index=self.radius_index,
            path_index=self.path_index,
        )

    @property
    def radius(self) -> PositiveRational:
        return calkin_wilf_rational(self.radius_index)

    @property
    def deterministic_order_key(self) -> tuple[object, ...]:
        return (
            self.action_index,
            self.record_order,
            int(self.orientation),
            self.radius_index,
            self.path_index,
            self.record_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index_schema": ACTION_INDEX_SCHEMA,
            "record_id": self.record_id,
            "record_order": self.record_order,
            "record_count": self.record_count,
            "orientation": int(self.orientation),
            "radius_index": _encode_nonnegative_integer(self.radius_index),
            "radius": self.radius.to_dict(),
            "path_index": _encode_nonnegative_integer(self.path_index),
            "action_index": _encode_nonnegative_integer(self.action_index),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "PathActionKey":
        if data.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("serialized action-index schema is unsupported.")
        key = cls(
            record_id=str(data["record_id"]),
            record_order=int(data["record_order"]),
            record_count=int(data["record_count"]),
            orientation=PathOrientation(int(data["orientation"])),
            radius_index=_decode_positive_integer(
                "radius_index", data["radius_index"]
            ),
            path_index=_decode_positive_integer("path_index", data["path_index"]),
        )
        if _decode_positive_integer("action_index", data["action_index"]) != key.action_index:
            raise ValueError("serialized action_index fails canonical bijection.")
        if data.get("radius") != key.radius.to_dict():
            raise ValueError("serialized radius fails Calkin--Wilf binding.")
        return key


def canonical_action_receipt_digest(
    key: PathActionKey,
    eligibility_token_digest: str,
) -> str:
    """Bind every physical receipt to one canonical action and state token."""

    token_digest = _nonempty(
        "eligibility_token_digest", eligibility_token_digest
    )
    return _digest(
        {
            "schema": _ACTION_RECEIPT_SCHEMA,
            "action_index_schema": ACTION_INDEX_SCHEMA,
            "action_key": key.to_dict(),
            "eligibility_token_digest": token_digest,
        }
    )


@dataclass(frozen=True)
class CanonicalActionMass(_DeterministicSerializable):
    action_index: int

    def __post_init__(self) -> None:
        _positive_index("action_index", self.action_index)

    @property
    def scheduling_coefficient(self) -> Fraction:
        """Exact coefficient ``pi^2 a_i = 6 / i^2``."""

        return Fraction(6, self.action_index * self.action_index)

    @property
    def log_value(self) -> float:
        return _fraction_log(self.scheduling_coefficient) - 2.0 * math.log(
            math.pi
        )

    @property
    def symbolic_expression(self) -> str:
        return (
            "6/(pi^2*i^2),i="
            f"{_encode_nonnegative_integer(self.action_index)}"
        )

    def as_float(self) -> float:
        return math.exp(self.log_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "action_index_schema": ACTION_INDEX_SCHEMA,
            "action_index": _encode_nonnegative_integer(self.action_index),
            "scheduling_coefficient": _fraction_to_dict(
                self.scheduling_coefficient
            ),
            "log_value": self.log_value,
            "symbolic_expression": self.symbolic_expression,
            "float_diagnostic": self.as_float(),
        }


def canonical_action_mass(action_index: int) -> CanonicalActionMass:
    """Return ``6/(pi^2 i^2)`` without evaluating the mass for scheduling."""

    index = _positive_index("action_index", action_index)
    return CanonicalActionMass(action_index=index)


@dataclass(frozen=True)
class LogEntitlement(_DeterministicSerializable):
    """Exact fair coefficient with a logarithm used only for diagnostics.

    The physical entitlement equals ``scheduling_coefficient / pi**2``.  The
    common ``pi**-2`` factor cancels from every fair virtual-finish comparison.
    """

    coefficient_numerator: int
    coefficient_denominator: int
    symbolic_expression: str

    def __post_init__(self) -> None:
        numerator = _positive_index(
            "coefficient_numerator", self.coefficient_numerator
        )
        denominator = _positive_index(
            "coefficient_denominator", self.coefficient_denominator
        )
        if math.gcd(numerator, denominator) != 1:
            raise ValueError("fair entitlement coefficient must be reduced.")
        _nonempty("symbolic_expression", self.symbolic_expression)
        if self.log_value > 0.0:
            raise ValueError("fair entitlement cannot exceed one.")

    @classmethod
    def from_coefficient(
        cls,
        coefficient: Fraction,
        *,
        symbolic_expression: str,
    ) -> "LogEntitlement":
        if coefficient <= 0:
            raise ValueError("fair entitlement coefficient must be positive.")
        return cls(
            coefficient_numerator=coefficient.numerator,
            coefficient_denominator=coefficient.denominator,
            symbolic_expression=symbolic_expression,
        )

    @property
    def scheduling_coefficient(self) -> Fraction:
        return Fraction(self.coefficient_numerator, self.coefficient_denominator)

    @property
    def log_value(self) -> float:
        return _fraction_log(self.scheduling_coefficient) - 2.0 * math.log(
            math.pi
        )

    def as_float(self) -> float:
        return math.exp(self.log_value)

    def to_dict(self) -> dict[str, object]:
        return {
            "scheduling_coefficient": _fraction_to_dict(
                self.scheduling_coefficient
            ),
            "log_value": self.log_value,
            "symbolic_expression": self.symbolic_expression,
            "float_diagnostic": self.as_float(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "LogEntitlement":
        coefficient = _fraction_from_dict(
            "scheduling_coefficient", data.get("scheduling_coefficient")
        )
        entitlement = cls.from_coefficient(
            coefficient,
            symbolic_expression=str(data["symbolic_expression"]),
        )
        if float(data["log_value"]) != entitlement.log_value:
            raise ValueError("serialized entitlement logarithm is inconsistent.")
        if float(data["float_diagnostic"]) != entitlement.as_float():
            raise ValueError("serialized entitlement diagnostic is inconsistent.")
        return entitlement


@dataclass(frozen=True)
class RunEnergyUnit(_DeterministicSerializable):
    """Fixed run-level physical energy unit ``E0``."""

    run_id: str
    unit_id: str
    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _nonempty("run_id", self.run_id))
        object.__setattr__(self, "unit_id", _nonempty("unit_id", self.unit_id))
        _positive("value", self.value)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "unit_id": self.unit_id,
            "value": self.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RunEnergyUnit":
        return cls(
            run_id=str(data["run_id"]),
            unit_id=str(data["unit_id"]),
            value=float(data["value"]),
        )


@dataclass(frozen=True)
class EnergyInterval(_DeterministicSerializable):
    """One state energy on a declared simultaneous comparison epoch."""

    state_id: str
    energy_estimate: float
    energy_error_bound: float
    comparison_epoch: str
    simultaneous: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _nonempty("state_id", self.state_id))
        estimate = _finite("energy_estimate", self.energy_estimate)
        error = _nonnegative("energy_error_bound", self.energy_error_bound)
        if not math.isfinite(estimate - error) or not math.isfinite(
            estimate + error
        ):
            raise ValueError("derived energy interval bounds must be finite.")
        object.__setattr__(
            self,
            "comparison_epoch",
            _nonempty("comparison_epoch", self.comparison_epoch),
        )

    @property
    def lower_bound(self) -> float:
        return float(self.energy_estimate - self.energy_error_bound)

    @property
    def upper_bound(self) -> float:
        return float(self.energy_estimate + self.energy_error_bound)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "energy_estimate": self.energy_estimate,
            "energy_error_bound": self.energy_error_bound,
            "comparison_epoch": self.comparison_epoch,
            "simultaneous": self.simultaneous,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "EnergyInterval":
        interval = cls(
            state_id=str(data["state_id"]),
            energy_estimate=float(data["energy_estimate"]),
            energy_error_bound=float(data["energy_error_bound"]),
            comparison_epoch=str(data["comparison_epoch"]),
            simultaneous=bool(data["simultaneous"]),
        )
        if float(data["lower_bound"]) != interval.lower_bound:
            raise ValueError("serialized energy lower bound is inconsistent.")
        if float(data["upper_bound"]) != interval.upper_bound:
            raise ValueError("serialized energy upper bound is inconsistent.")
        return interval


@dataclass(frozen=True)
class FSExclusionEvidence:
    """Overlap and connected-path evidence for physical FS exclusion."""

    witness_id: str
    action_receipt_digest: str
    path_id: str
    component_id: str
    comparison_epoch: str
    path_origin_state_id: str
    path_endpoint_state_id: str
    incumbent_state_id: str
    subject_state_id: str
    overlap_amplitude_estimate: float | None
    overlap_error_bound: float | None
    current_exclusion_radius: float | None
    path_distance_lower_bound: float | None
    overlap_status: CertificateState
    path_status: CertificateState
    component_status: CertificateState
    simultaneous: bool

    def __post_init__(self) -> None:
        for name in ("overlap_status", "path_status", "component_status"):
            object.__setattr__(self, name, CertificateState(getattr(self, name)))


@dataclass(frozen=True)
class FSExclusionCertificate(_DeterministicSerializable):
    witness_id: str
    action_receipt_digest: str
    path_id: str
    component_id: str
    comparison_epoch: str
    path_origin_state_id: str
    path_endpoint_state_id: str
    incumbent_state_id: str
    subject_state_id: str
    overlap_amplitude_upper_bound: float
    endpoint_distance_lower_bound: float
    current_exclusion_radius: float
    path_distance_lower_bound: float

    def __post_init__(self) -> None:
        for name in (
            "witness_id",
            "action_receipt_digest",
            "path_id",
            "component_id",
            "comparison_epoch",
            "path_origin_state_id",
            "path_endpoint_state_id",
            "incumbent_state_id",
            "subject_state_id",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.subject_state_id != self.path_endpoint_state_id:
            raise ValueError("FS subject must equal the bound path endpoint.")
        overlap = _nonnegative(
            "overlap_amplitude_upper_bound",
            self.overlap_amplitude_upper_bound,
        )
        endpoint = _positive(
            "endpoint_distance_lower_bound",
            self.endpoint_distance_lower_bound,
        )
        current = _nonnegative(
            "current_exclusion_radius", self.current_exclusion_radius
        )
        path_lower = _nonnegative(
            "path_distance_lower_bound", self.path_distance_lower_bound
        )
        if overlap > 1.0:
            raise ValueError("overlap upper bound cannot exceed one.")
        if max(endpoint, current, path_lower) > _FS_DIAMETER:
            raise ValueError("Fubini--Study distances cannot exceed pi/2.")
        if endpoint < current or path_lower < current:
            raise ValueError("subject/path does not satisfy exclusion radius.")

    def to_dict(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "action_receipt_digest": self.action_receipt_digest,
            "path_id": self.path_id,
            "component_id": self.component_id,
            "comparison_epoch": self.comparison_epoch,
            "path_origin_state_id": self.path_origin_state_id,
            "path_endpoint_state_id": self.path_endpoint_state_id,
            "incumbent_state_id": self.incumbent_state_id,
            "subject_state_id": self.subject_state_id,
            "overlap_amplitude_upper_bound": self.overlap_amplitude_upper_bound,
            "endpoint_distance_lower_bound": self.endpoint_distance_lower_bound,
            "current_exclusion_radius": self.current_exclusion_radius,
            "path_distance_lower_bound": self.path_distance_lower_bound,
        }


@dataclass(frozen=True)
class FSExclusionAssessment(_DeterministicSerializable):
    kind: ResolutionKind
    reason: str
    certificate: FSExclusionCertificate | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ResolutionKind(self.kind))
        _nonempty("reason", self.reason)
        if (self.kind is ResolutionKind.CERTIFIED) != (
            self.certificate is not None
        ):
            raise ValueError("only certified FS assessment carries a certificate.")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "certificate": (
                None if self.certificate is None else self.certificate.to_dict()
            ),
        }


def assess_fs_exclusion(
    evidence: FSExclusionEvidence | None,
) -> FSExclusionAssessment:
    """Apply ``acos(min(1,c_hat+eps_c))`` with full path binding."""

    if evidence is None:
        return FSExclusionAssessment(
            kind=ResolutionKind.REFINEMENT,
            reason="fs_exclusion_evidence_missing",
        )
    statuses = (
        ("overlap", evidence.overlap_status),
        ("path", evidence.path_status),
        ("component", evidence.component_status),
    )
    for name, status in statuses:
        if status is CertificateState.FAILED:
            return FSExclusionAssessment(
                kind=ResolutionKind.INVALID,
                reason=f"fs_{name}_certificate_failed",
            )
        if status is CertificateState.UNRESOLVED:
            return FSExclusionAssessment(
                kind=ResolutionKind.REFINEMENT,
                reason=f"fs_{name}_certificate_unresolved",
            )
    if not evidence.simultaneous:
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_exclusion_not_simultaneous",
        )
    for name in (
        "witness_id",
        "action_receipt_digest",
        "path_id",
        "component_id",
        "comparison_epoch",
        "path_origin_state_id",
        "path_endpoint_state_id",
        "incumbent_state_id",
        "subject_state_id",
    ):
        if not str(getattr(evidence, name)).strip():
            return FSExclusionAssessment(
                kind=ResolutionKind.INVALID,
                reason=f"fs_{name}_missing",
            )
    if evidence.subject_state_id != evidence.path_endpoint_state_id:
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_subject_path_endpoint_mismatch",
        )
    numeric = (
        evidence.overlap_amplitude_estimate,
        evidence.overlap_error_bound,
        evidence.current_exclusion_radius,
        evidence.path_distance_lower_bound,
    )
    if any(value is None for value in numeric):
        return FSExclusionAssessment(
            kind=ResolutionKind.REFINEMENT,
            reason="fs_exclusion_numeric_bound_missing",
        )
    if not all(_is_finite_real(value) for value in numeric):
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_exclusion_nonfinite_data",
        )
    overlap = float(evidence.overlap_amplitude_estimate)
    error = float(evidence.overlap_error_bound)
    current = float(evidence.current_exclusion_radius)
    path_lower = float(evidence.path_distance_lower_bound)
    if (
        overlap < 0.0
        or overlap > 1.0
        or error < 0.0
        or current < 0.0
        or path_lower < 0.0
        or current > _FS_DIAMETER
        or path_lower > _FS_DIAMETER
    ):
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_exclusion_bound_out_of_range",
        )
    overlap_upper = min(1.0, overlap + error)
    endpoint_lower = math.acos(overlap_upper)
    if endpoint_lower <= 0.0:
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_endpoint_separation_nonpositive",
        )
    if endpoint_lower < current or path_lower < current:
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_exclusion_radius_not_satisfied",
        )
    try:
        certificate = FSExclusionCertificate(
            witness_id=evidence.witness_id,
            action_receipt_digest=evidence.action_receipt_digest,
            path_id=evidence.path_id,
            component_id=evidence.component_id,
            comparison_epoch=evidence.comparison_epoch,
            path_origin_state_id=evidence.path_origin_state_id,
            path_endpoint_state_id=evidence.path_endpoint_state_id,
            incumbent_state_id=evidence.incumbent_state_id,
            subject_state_id=evidence.subject_state_id,
            overlap_amplitude_upper_bound=overlap_upper,
            endpoint_distance_lower_bound=endpoint_lower,
            current_exclusion_radius=current,
            path_distance_lower_bound=path_lower,
        )
    except ValueError as exc:
        return FSExclusionAssessment(
            kind=ResolutionKind.INVALID,
            reason=f"fs_exclusion_invalid:{exc}",
        )
    return FSExclusionAssessment(
        kind=ResolutionKind.CERTIFIED,
        reason="fs_exclusion_certified",
        certificate=certificate,
    )


@dataclass(frozen=True)
class StabilizedTrustPathEvidence:
    witness_id: str
    action_receipt_digest: str
    path_id: str
    comparison_epoch: str
    origin_state_id: str
    endpoint_state_id: str
    trust_provenance_digest: str
    reference_trust_radius: float | None
    scheduled_trust_radius: float | None
    schedule_error_bound: float | None
    certified_trust_arclength: float | None
    arclength_error_bound: float | None
    status: CertificateState
    simultaneous: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CertificateState(self.status))


@dataclass(frozen=True)
class StabilizedTrustPathCertificate(_DeterministicSerializable):
    witness_id: str
    action_receipt_digest: str
    path_id: str
    comparison_epoch: str
    origin_state_id: str
    endpoint_state_id: str
    trust_provenance_digest: str
    reference_trust_radius: float
    scheduled_trust_radius: float
    certified_trust_arclength: float
    schedule_error_bound: float
    arclength_error_bound: float
    radius: PositiveRational

    def to_dict(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "action_receipt_digest": self.action_receipt_digest,
            "path_id": self.path_id,
            "comparison_epoch": self.comparison_epoch,
            "origin_state_id": self.origin_state_id,
            "endpoint_state_id": self.endpoint_state_id,
            "trust_provenance_digest": self.trust_provenance_digest,
            "reference_trust_radius": self.reference_trust_radius,
            "scheduled_trust_radius": self.scheduled_trust_radius,
            "certified_trust_arclength": self.certified_trust_arclength,
            "schedule_error_bound": self.schedule_error_bound,
            "arclength_error_bound": self.arclength_error_bound,
            "radius": self.radius.to_dict(),
        }


@dataclass(frozen=True)
class UniformBarrierEvidence:
    witness_id: str
    action_receipt_digest: str
    enclosure_id: str
    path_id: str
    origin_state_id: str
    comparison_epoch: str
    incumbent_energy: EnergyInterval | None
    barrier_upper_bound: float | None
    comparison_energy_width: float | None
    incumbent_referenced: bool
    status: CertificateState
    simultaneous: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CertificateState(self.status))


@dataclass(frozen=True)
class UniformBarrierCertificate(_DeterministicSerializable):
    witness_id: str
    action_receipt_digest: str
    enclosure_id: str
    path_id: str
    origin_state_id: str
    comparison_epoch: str
    incumbent_energy: EnergyInterval
    barrier_upper_bound: float
    comparison_energy_width: float

    def to_dict(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "action_receipt_digest": self.action_receipt_digest,
            "enclosure_id": self.enclosure_id,
            "path_id": self.path_id,
            "origin_state_id": self.origin_state_id,
            "comparison_epoch": self.comparison_epoch,
            "incumbent_energy": self.incumbent_energy.to_dict(),
            "barrier_upper_bound": self.barrier_upper_bound,
            "comparison_energy_width": self.comparison_energy_width,
            "incumbent_referenced": True,
            "uniform_path_enclosure": True,
        }


@dataclass(frozen=True)
class EndpointDistanceEvidence:
    witness_id: str
    action_receipt_digest: str
    path_id: str
    endpoint_state_id: str
    comparison_epoch: str
    active_manifold_digest: str
    trust_radius: float | None
    distance_lower_bound: float | None
    status: CertificateState
    simultaneous: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CertificateState(self.status))


@dataclass(frozen=True)
class EndpointDistanceCertificate(_DeterministicSerializable):
    witness_id: str
    action_receipt_digest: str
    path_id: str
    endpoint_state_id: str
    comparison_epoch: str
    active_manifold_digest: str
    trust_radius: float
    distance_lower_bound: float

    def to_dict(self) -> dict[str, object]:
        return {
            "witness_id": self.witness_id,
            "action_receipt_digest": self.action_receipt_digest,
            "path_id": self.path_id,
            "endpoint_state_id": self.endpoint_state_id,
            "comparison_epoch": self.comparison_epoch,
            "active_manifold_digest": self.active_manifold_digest,
            "trust_radius": self.trust_radius,
            "distance_lower_bound": self.distance_lower_bound,
        }


@dataclass(frozen=True)
class BarrierDistanceUtility(_DeterministicSerializable):
    endpoint_distance_lower_bound: float
    distance_lower_bound_squared: float
    distance_lower_bound_squared_numerator: int
    distance_lower_bound_squared_denominator: int
    barrier_upper_bound: float
    comparison_energy_width: float
    energy_unit: RunEnergyUnit
    log_raw_utility: float | None
    log_compactified_utility: float
    canonical_mass: CanonicalActionMass
    phase3_cost: float
    live_entitlement: LogEntitlement

    @property
    def exact_distance_lower_bound_squared(self) -> Fraction:
        return Fraction(
            self.distance_lower_bound_squared_numerator,
            self.distance_lower_bound_squared_denominator,
        )

    @property
    def denominator(self) -> float:
        return float(self.barrier_upper_bound + self.comparison_energy_width)

    @property
    def raw_utility(self) -> float:
        if self.log_raw_utility is None:
            return math.inf
        try:
            return math.exp(self.log_raw_utility)
        except OverflowError:
            return math.inf

    @property
    def compactified_utility(self) -> float:
        return math.exp(self.log_compactified_utility)

    @property
    def live_weight(self) -> float:
        return self.live_entitlement.as_float()

    def to_dict(self) -> dict[str, object]:
        return {
            "endpoint_distance_lower_bound": self.endpoint_distance_lower_bound,
            "distance_lower_bound_squared": self.distance_lower_bound_squared,
            "distance_lower_bound_squared_exact": _fraction_to_dict(
                self.exact_distance_lower_bound_squared
            ),
            "barrier_upper_bound": self.barrier_upper_bound,
            "comparison_energy_width": self.comparison_energy_width,
            "energy_unit": self.energy_unit.to_dict(),
            "log_raw_utility": (
                "infinity"
                if self.log_raw_utility is None
                else self.log_raw_utility
            ),
            "log_compactified_utility": self.log_compactified_utility,
            "compactified_utility_diagnostic": self.compactified_utility,
            "canonical_mass": self.canonical_mass.to_dict(),
            "phase3_cost": self.phase3_cost,
            "live_entitlement": self.live_entitlement.to_dict(),
        }


def compute_barrier_distance_utility(
    *,
    endpoint_distance_lower_bound: float,
    barrier_upper_bound: float,
    comparison_energy_width: float,
    energy_unit: RunEnergyUnit,
    action_index: int,
    phase3_cost: float,
) -> BarrierDistanceUtility:
    """Compute compact utility and log-symbolic live entitlement."""

    distance = _positive(
        "endpoint_distance_lower_bound", endpoint_distance_lower_bound
    )
    if distance > _FS_DIAMETER:
        raise ValueError("endpoint distance cannot exceed pi/2.")
    barrier = _nonnegative("barrier_upper_bound", barrier_upper_bound)
    width = _nonnegative("comparison_energy_width", comparison_energy_width)
    cost = _nonnegative("phase3_cost", phase3_cost)
    distance_fraction = _finite_fraction(
        "endpoint_distance_lower_bound", distance
    )
    distance_squared_exact = distance_fraction * distance_fraction
    distance_squared = float(distance_squared_exact)
    denominator = barrier + width
    if not math.isfinite(denominator):
        raise ValueError("barrier denominator must be finite.")
    if denominator == 0.0:
        log_raw: float | None = None
        log_compact = 0.0
        compactified_exact = Fraction(1, 1)
    else:
        denominator_exact = _finite_fraction("barrier_upper_bound", barrier) + (
            _finite_fraction("comparison_energy_width", width)
        )
        raw_utility_exact = distance_squared_exact / denominator_exact
        scaled_exact = (
            _finite_fraction("energy_unit", energy_unit.value)
            * raw_utility_exact
        )
        compactified_exact = scaled_exact / (1 + scaled_exact)
        log_raw = _fraction_log(raw_utility_exact)
        log_compact = _fraction_log(compactified_exact)
    mass = canonical_action_mass(action_index)
    cost_exact = _finite_fraction("phase3_cost", cost)
    scheduling_coefficient = (
        mass.scheduling_coefficient
        * compactified_exact
        / (1 + cost_exact)
    )
    entitlement = LogEntitlement.from_coefficient(
        scheduling_coefficient,
        symbolic_expression=(
            f"{mass.symbolic_expression}*phi(E0*D2/(B+eps))/(1+K3)"
        ),
    )
    return BarrierDistanceUtility(
        endpoint_distance_lower_bound=distance,
        distance_lower_bound_squared=distance_squared,
        distance_lower_bound_squared_numerator=distance_squared_exact.numerator,
        distance_lower_bound_squared_denominator=distance_squared_exact.denominator,
        barrier_upper_bound=barrier,
        comparison_energy_width=width,
        energy_unit=energy_unit,
        log_raw_utility=log_raw,
        log_compactified_utility=log_compact,
        canonical_mass=mass,
        phase3_cost=cost,
        live_entitlement=entitlement,
    )


@dataclass(frozen=True)
class PathActionEvidence:
    key: PathActionKey
    phase3_cost: float
    eligibility_token_digest: str
    energy_unit_digest: str
    endpoint_seed_energy: EnergyInterval | None
    numerical_status: CertificateState
    map_status: CertificateState
    symmetry_status: CertificateState
    padding_status: CertificateState
    trust_path_evidence: StabilizedTrustPathEvidence | None
    barrier_evidence: UniformBarrierEvidence | None
    endpoint_distance_evidence: EndpointDistanceEvidence | None
    exclusion_evidence: FSExclusionEvidence | None

    def __post_init__(self) -> None:
        for name in (
            "numerical_status",
            "map_status",
            "symmetry_status",
            "padding_status",
        ):
            object.__setattr__(self, name, CertificateState(getattr(self, name)))


@dataclass(frozen=True)
class CertifiedPathAction(_DeterministicSerializable):
    key: PathActionKey
    eligibility_token: EligibilityStateToken
    energy_unit: RunEnergyUnit
    endpoint_seed_energy: EnergyInterval
    trust_path_certificate: StabilizedTrustPathCertificate
    barrier_certificate: UniformBarrierCertificate
    endpoint_distance_certificate: EndpointDistanceCertificate
    exclusion_certificate: FSExclusionCertificate
    utility: BarrierDistanceUtility

    @property
    def origin_state_id(self) -> str:
        return self.trust_path_certificate.origin_state_id

    @property
    def endpoint_state_id(self) -> str:
        return self.trust_path_certificate.endpoint_state_id

    @property
    def comparison_epoch(self) -> str:
        return self.trust_path_certificate.comparison_epoch

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "eligibility_token": self.eligibility_token.to_dict(),
            "eligibility_token_digest": self.eligibility_token.digest,
            "energy_unit": self.energy_unit.to_dict(),
            "energy_unit_digest": self.energy_unit.digest,
            "endpoint_seed_energy": self.endpoint_seed_energy.to_dict(),
            "trust_path_certificate": self.trust_path_certificate.to_dict(),
            "barrier_certificate": self.barrier_certificate.to_dict(),
            "endpoint_distance_certificate": (
                self.endpoint_distance_certificate.to_dict()
            ),
            "exclusion_certificate": self.exclusion_certificate.to_dict(),
            "utility": self.utility.to_dict(),
        }


@dataclass(frozen=True)
class RefinementAction(_DeterministicSerializable):
    key: PathActionKey
    phase3_cost: float
    eligibility_token_digest: str
    energy_unit_digest: str
    reason: str

    def __post_init__(self) -> None:
        _nonnegative("phase3_cost", self.phase3_cost)
        _nonempty("eligibility_token_digest", self.eligibility_token_digest)
        _nonempty("energy_unit_digest", self.energy_unit_digest)
        _nonempty("reason", self.reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key.to_dict(),
            "phase3_cost": self.phase3_cost,
            "eligibility_token_digest": self.eligibility_token_digest,
            "energy_unit_digest": self.energy_unit_digest,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PathActionAssessment(_DeterministicSerializable):
    kind: ResolutionKind
    reason: str
    certified_action: CertifiedPathAction | None = None
    refinement_action: RefinementAction | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ResolutionKind(self.kind))
        _nonempty("reason", self.reason)
        if self.kind is ResolutionKind.CERTIFIED:
            if self.certified_action is None or self.refinement_action is not None:
                raise ValueError("certified assessment carries only a move action.")
        elif self.kind is ResolutionKind.REFINEMENT:
            if self.refinement_action is None or self.certified_action is not None:
                raise ValueError("refinement assessment carries only a ref action.")
        elif self.certified_action is not None or self.refinement_action is not None:
            raise ValueError("invalid assessment cannot carry service action.")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "reason": self.reason,
            "certified_action": (
                None
                if self.certified_action is None
                else self.certified_action.to_dict()
            ),
            "refinement_action": (
                None
                if self.refinement_action is None
                else self.refinement_action.to_dict()
            ),
        }


def _refinement_assessment(
    evidence: PathActionEvidence,
    reason: str,
) -> PathActionAssessment:
    return PathActionAssessment(
        kind=ResolutionKind.REFINEMENT,
        reason=reason,
        refinement_action=RefinementAction(
            key=evidence.key,
            phase3_cost=float(evidence.phase3_cost),
            eligibility_token_digest=evidence.eligibility_token_digest,
            energy_unit_digest=evidence.energy_unit_digest,
            reason=reason,
        ),
    )


def assess_path_action(
    *,
    evidence: PathActionEvidence,
    eligibility: ExposedFamilyEligibility,
    energy_unit: RunEnergyUnit,
) -> PathActionAssessment:
    """Reduce fully bound action evidence to move, refinement, or invalid."""

    if not eligibility.eligible:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason=f"exposed_family_not_eligible:{eligibility.reason}",
        )
    token = eligibility.state_token
    if evidence.eligibility_token_digest != token.digest:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_action_eligibility_token_mismatch",
        )
    if evidence.energy_unit_digest != energy_unit.digest:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_action_energy_unit_token_mismatch",
        )
    key = evidence.key
    if key.record_count != len(token.reachable_record_ids):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_action_record_count_mismatch",
        )
    if token.reachable_record_ids[key.record_order - 1] != key.record_id:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_action_record_order_mismatch",
        )
    expected_action_receipt = canonical_action_receipt_digest(
        key,
        token.digest,
    )
    for receipt_name, receipt in (
        ("trust_path", evidence.trust_path_evidence),
        ("uniform_barrier", evidence.barrier_evidence),
        ("endpoint_distance", evidence.endpoint_distance_evidence),
        ("fs_exclusion", evidence.exclusion_evidence),
    ):
        if (
            receipt is not None
            and receipt.action_receipt_digest != expected_action_receipt
        ):
            return PathActionAssessment(
                kind=ResolutionKind.INVALID,
                reason=f"{receipt_name}_action_receipt_mismatch",
            )
    if not _is_finite_real(evidence.phase3_cost) or float(evidence.phase3_cost) < 0:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="phase3_cost_invalid",
        )
    for name, status in (
        ("numerical", evidence.numerical_status),
        ("map", evidence.map_status),
        ("symmetry", evidence.symmetry_status),
        ("padding", evidence.padding_status),
    ):
        if status is CertificateState.FAILED:
            return PathActionAssessment(
                kind=ResolutionKind.INVALID,
                reason=f"{name}_certificate_failed",
            )
        if status is CertificateState.UNRESOLVED:
            return _refinement_assessment(
                evidence, f"{name}_certificate_unresolved"
            )
    if evidence.endpoint_seed_energy is None:
        return _refinement_assessment(evidence, "endpoint_seed_energy_missing")
    seed = evidence.endpoint_seed_energy
    if not seed.simultaneous:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="endpoint_seed_energy_not_simultaneous",
        )

    trust = evidence.trust_path_evidence
    if trust is None or trust.status is CertificateState.UNRESOLVED:
        return _refinement_assessment(evidence, "trust_path_unresolved")
    if trust.status is CertificateState.FAILED or not trust.simultaneous:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="trust_path_failed_or_not_simultaneous",
        )
    trust_numeric = (
        trust.reference_trust_radius,
        trust.scheduled_trust_radius,
        trust.schedule_error_bound,
        trust.certified_trust_arclength,
        trust.arclength_error_bound,
    )
    if any(value is None for value in trust_numeric):
        return _refinement_assessment(evidence, "trust_path_numeric_bound_missing")
    if not all(_is_finite_real(value) for value in trust_numeric):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="trust_path_nonfinite_data",
        )
    reference_radius = float(trust.reference_trust_radius)
    scheduled_radius = float(trust.scheduled_trust_radius)
    schedule_error = float(trust.schedule_error_bound)
    arclength = float(trust.certified_trust_arclength)
    arclength_error = float(trust.arclength_error_bound)
    if (
        reference_radius <= 0.0
        or scheduled_radius <= 0.0
        or arclength <= 0.0
        or schedule_error < 0.0
        or arclength_error < 0.0
    ):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="trust_path_bound_out_of_range",
        )
    radius = key.radius
    try:
        ratio = radius.numerator / radius.denominator
        expected_radius = reference_radius * ratio
    except (OverflowError, ZeroDivisionError):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="scheduled_radius_not_finitely_representable",
        )
    if not math.isfinite(expected_radius):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="scheduled_radius_not_finitely_representable",
        )
    if abs(scheduled_radius - expected_radius) > schedule_error:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="scheduled_radius_calkin_wilf_mismatch",
        )
    if abs(arclength - scheduled_radius) > arclength_error:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="trust_arclength_radius_mismatch",
        )
    if reference_radius != token.trust_radius:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="reference_trust_radius_eligibility_mismatch",
        )
    if trust.trust_provenance_digest != token.trust_provenance_digest:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="trust_path_provenance_mismatch",
        )
    try:
        trust_certificate = StabilizedTrustPathCertificate(
            witness_id=_nonempty("trust_witness_id", trust.witness_id),
            action_receipt_digest=_nonempty(
                "trust_action_receipt_digest", trust.action_receipt_digest
            ),
            path_id=_nonempty("trust_path_id", trust.path_id),
            comparison_epoch=_nonempty(
                "trust_comparison_epoch", trust.comparison_epoch
            ),
            origin_state_id=_nonempty(
                "trust_origin_state_id", trust.origin_state_id
            ),
            endpoint_state_id=_nonempty(
                "trust_endpoint_state_id", trust.endpoint_state_id
            ),
            trust_provenance_digest=_nonempty(
                "trust_provenance_digest",
                trust.trust_provenance_digest,
            ),
            reference_trust_radius=reference_radius,
            scheduled_trust_radius=scheduled_radius,
            certified_trust_arclength=arclength,
            schedule_error_bound=schedule_error,
            arclength_error_bound=arclength_error,
            radius=radius,
        )
    except ValueError as exc:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason=f"trust_path_invalid:{exc}",
        )

    barrier = evidence.barrier_evidence
    if barrier is None or barrier.status is CertificateState.UNRESOLVED:
        return _refinement_assessment(evidence, "uniform_barrier_unresolved")
    if barrier.status is CertificateState.FAILED:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="uniform_barrier_certificate_failed",
        )
    if not barrier.simultaneous or not barrier.incumbent_referenced:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="uniform_barrier_not_simultaneous_or_incumbent_referenced",
        )
    if barrier.incumbent_energy is None:
        return _refinement_assessment(evidence, "barrier_incumbent_energy_missing")
    if (
        not barrier.incumbent_energy.simultaneous
        or barrier.incumbent_energy.comparison_epoch
        != barrier.comparison_epoch
    ):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="barrier_incumbent_energy_not_simultaneous_or_bound",
        )
    if (
        barrier.barrier_upper_bound is None
        or barrier.comparison_energy_width is None
    ):
        return _refinement_assessment(evidence, "uniform_barrier_bound_missing")
    if not _is_finite_real(barrier.barrier_upper_bound) or not _is_finite_real(
        barrier.comparison_energy_width
    ):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="uniform_barrier_nonfinite_data",
        )
    barrier_upper = float(barrier.barrier_upper_bound)
    comparison_width = float(barrier.comparison_energy_width)
    if barrier_upper < 0.0 or comparison_width < 0.0:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="uniform_barrier_bound_negative",
        )
    try:
        barrier_certificate = UniformBarrierCertificate(
            witness_id=_nonempty("barrier_witness_id", barrier.witness_id),
            action_receipt_digest=_nonempty(
                "barrier_action_receipt_digest", barrier.action_receipt_digest
            ),
            enclosure_id=_nonempty(
                "barrier_enclosure_id", barrier.enclosure_id
            ),
            path_id=_nonempty("barrier_path_id", barrier.path_id),
            origin_state_id=_nonempty(
                "barrier_origin_state_id", barrier.origin_state_id
            ),
            comparison_epoch=_nonempty(
                "barrier_comparison_epoch", barrier.comparison_epoch
            ),
            incumbent_energy=barrier.incumbent_energy,
            barrier_upper_bound=barrier_upper,
            comparison_energy_width=comparison_width,
        )
    except ValueError as exc:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason=f"uniform_barrier_invalid:{exc}",
        )

    endpoint = evidence.endpoint_distance_evidence
    if endpoint is None or endpoint.status is CertificateState.UNRESOLVED:
        return _refinement_assessment(evidence, "endpoint_distance_unresolved")
    if endpoint.status is CertificateState.FAILED or not endpoint.simultaneous:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="endpoint_distance_failed_or_not_simultaneous",
        )
    if endpoint.trust_radius is None or endpoint.distance_lower_bound is None:
        return _refinement_assessment(evidence, "endpoint_distance_bound_missing")
    if not _is_finite_real(endpoint.trust_radius) or not _is_finite_real(
        endpoint.distance_lower_bound
    ):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="endpoint_distance_nonfinite_data",
        )
    endpoint_radius = float(endpoint.trust_radius)
    endpoint_lower = float(endpoint.distance_lower_bound)
    if endpoint_radius != scheduled_radius:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="endpoint_distance_trust_radius_mismatch",
        )
    if endpoint_lower <= 0.0 or endpoint_lower > _FS_DIAMETER:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_action_endpoint_separation_nonpositive_or_invalid",
        )
    if endpoint.active_manifold_digest != token.support_provenance_digest:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="endpoint_active_manifold_provenance_mismatch",
        )
    try:
        endpoint_certificate = EndpointDistanceCertificate(
            witness_id=_nonempty("endpoint_witness_id", endpoint.witness_id),
            action_receipt_digest=_nonempty(
                "endpoint_action_receipt_digest", endpoint.action_receipt_digest
            ),
            path_id=_nonempty("endpoint_path_id", endpoint.path_id),
            endpoint_state_id=_nonempty(
                "endpoint_state_id", endpoint.endpoint_state_id
            ),
            comparison_epoch=_nonempty(
                "endpoint_comparison_epoch", endpoint.comparison_epoch
            ),
            active_manifold_digest=_nonempty(
                "active_manifold_digest", endpoint.active_manifold_digest
            ),
            trust_radius=endpoint_radius,
            distance_lower_bound=endpoint_lower,
        )
    except ValueError as exc:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason=f"endpoint_distance_invalid:{exc}",
        )

    exclusion = assess_fs_exclusion(evidence.exclusion_evidence)
    if exclusion.kind is ResolutionKind.REFINEMENT:
        return _refinement_assessment(evidence, exclusion.reason)
    if exclusion.kind is ResolutionKind.INVALID:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason=exclusion.reason,
        )
    assert exclusion.certificate is not None
    exclusion_certificate = exclusion.certificate

    path_ids = {
        trust_certificate.path_id,
        barrier_certificate.path_id,
        endpoint_certificate.path_id,
        exclusion_certificate.path_id,
    }
    action_receipts = {
        trust_certificate.action_receipt_digest,
        barrier_certificate.action_receipt_digest,
        endpoint_certificate.action_receipt_digest,
        exclusion_certificate.action_receipt_digest,
    }
    epochs = {
        trust_certificate.comparison_epoch,
        barrier_certificate.comparison_epoch,
        endpoint_certificate.comparison_epoch,
        exclusion_certificate.comparison_epoch,
        seed.comparison_epoch,
    }
    if len(path_ids) != 1:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_witness_id_binding_mismatch",
        )
    if action_receipts != {expected_action_receipt}:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_action_receipt_binding_mismatch",
        )
    if epochs != {token.comparison_epoch}:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_witness_comparison_epoch_mismatch",
        )
    if trust_certificate.origin_state_id != token.working_state_fingerprint:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_origin_working_state_mismatch",
        )
    if barrier_certificate.origin_state_id != trust_certificate.origin_state_id:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="barrier_origin_path_origin_mismatch",
        )
    endpoint_state = trust_certificate.endpoint_state_id
    if (
        endpoint_certificate.endpoint_state_id != endpoint_state
        or exclusion_certificate.path_endpoint_state_id != endpoint_state
        or seed.state_id != endpoint_state
    ):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="path_endpoint_witness_binding_mismatch",
        )
    if exclusion_certificate.path_origin_state_id != trust_certificate.origin_state_id:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="fs_path_origin_binding_mismatch",
        )
    if (
        exclusion_certificate.incumbent_state_id
        != barrier_certificate.incumbent_energy.state_id
    ):
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason="barrier_incumbent_fs_incumbent_mismatch",
        )

    try:
        utility = compute_barrier_distance_utility(
            endpoint_distance_lower_bound=endpoint_lower,
            barrier_upper_bound=barrier_upper,
            comparison_energy_width=comparison_width,
            energy_unit=energy_unit,
            action_index=key.action_index,
            phase3_cost=float(evidence.phase3_cost),
        )
        action = CertifiedPathAction(
            key=key,
            eligibility_token=token,
            energy_unit=energy_unit,
            endpoint_seed_energy=seed,
            trust_path_certificate=trust_certificate,
            barrier_certificate=barrier_certificate,
            endpoint_distance_certificate=endpoint_certificate,
            exclusion_certificate=exclusion_certificate,
            utility=utility,
        )
    except ValueError as exc:
        return PathActionAssessment(
            kind=ResolutionKind.INVALID,
            reason=f"path_action_invalid:{exc}",
        )
    return PathActionAssessment(
        kind=ResolutionKind.CERTIFIED,
        reason="path_action_move_certified",
        certified_action=action,
    )


ServiceAction: TypeAlias = CertifiedPathAction | RefinementAction


@dataclass(frozen=True)
class FrozenServiceItem(_DeterministicSerializable):
    """Tagged service clock with an exact activation-frozen entitlement."""

    tag: ServiceTag
    action_key: PathActionKey
    frozen_entitlement: LogEntitlement
    service_count: int
    service_epoch: str
    eligibility_token_digest: str
    energy_unit_digest: str
    activation_reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tag", ServiceTag(self.tag))
        _nonnegative_index("service_count", self.service_count)
        for name in (
            "service_epoch",
            "eligibility_token_digest",
            "energy_unit_digest",
            "activation_reason",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))

    @property
    def virtual_finish_log(self) -> float:
        return _log_positive_integer(self.service_count + 1) - (
            self.frozen_entitlement.log_value
        )

    @property
    def scaled_virtual_finish(self) -> Fraction:
        """Exact virtual finish with the common ``pi**2`` factor removed."""

        return Fraction(self.service_count + 1, 1) / (
            self.frozen_entitlement.scheduling_coefficient
        )

    @property
    def deterministic_order_key(self) -> tuple[object, ...]:
        tag_order = 0 if self.tag is ServiceTag.MOVE else 1
        return (*self.action_key.deterministic_order_key, tag_order)

    def to_dict(self) -> dict[str, object]:
        return {
            "tag": self.tag.value,
            "action_key": self.action_key.to_dict(),
            "frozen_entitlement": self.frozen_entitlement.to_dict(),
            "service_count": _encode_nonnegative_integer(self.service_count),
            "service_epoch": self.service_epoch,
            "eligibility_token_digest": self.eligibility_token_digest,
            "energy_unit_digest": self.energy_unit_digest,
            "activation_reason": self.activation_reason,
            "scaled_virtual_finish": _fraction_to_dict(
                self.scaled_virtual_finish
            ),
            "virtual_finish_log": self.virtual_finish_log,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "FrozenServiceItem":
        item = cls(
            tag=ServiceTag(str(data["tag"])),
            action_key=PathActionKey.from_dict(data["action_key"]),
            frozen_entitlement=LogEntitlement.from_dict(
                data["frozen_entitlement"]
            ),
            service_count=_decode_nonnegative_integer(
                "service_count", data["service_count"]
            ),
            service_epoch=str(data["service_epoch"]),
            eligibility_token_digest=str(data["eligibility_token_digest"]),
            energy_unit_digest=str(data["energy_unit_digest"]),
            activation_reason=str(data["activation_reason"]),
        )
        if float(data["virtual_finish_log"]) != item.virtual_finish_log:
            raise ValueError("serialized virtual finish fails clock binding.")
        if _fraction_from_dict(
            "scaled_virtual_finish", data.get("scaled_virtual_finish")
        ) != item.scaled_virtual_finish:
            raise ValueError("serialized exact virtual finish fails clock binding.")
        return item


def activate_service_item(
    assessment: PathActionAssessment,
    *,
    service_epoch: str,
) -> FrozenServiceItem:
    """Freeze move/refinement entitlement for a fixed state/population epoch."""

    epoch = _nonempty("service_epoch", service_epoch)
    if assessment.kind is ResolutionKind.CERTIFIED:
        action = assessment.certified_action
        assert action is not None
        return FrozenServiceItem(
            tag=ServiceTag.MOVE,
            action_key=action.key,
            frozen_entitlement=action.utility.live_entitlement,
            service_count=0,
            service_epoch=epoch,
            eligibility_token_digest=action.eligibility_token.digest,
            energy_unit_digest=action.energy_unit.digest,
            activation_reason=assessment.reason,
        )
    if assessment.kind is ResolutionKind.REFINEMENT:
        action = assessment.refinement_action
        assert action is not None
        mass = canonical_action_mass(action.key.action_index)
        cost = _finite_fraction("phase3_cost", action.phase3_cost)
        entitlement = LogEntitlement.from_coefficient(
            mass.scheduling_coefficient / (1 + cost),
            symbolic_expression=f"{mass.symbolic_expression}/(1+K3)",
        )
        return FrozenServiceItem(
            tag=ServiceTag.REFINEMENT,
            action_key=action.key,
            frozen_entitlement=entitlement,
            service_count=0,
            service_epoch=epoch,
            eligibility_token_digest=action.eligibility_token_digest,
            energy_unit_digest=action.energy_unit_digest,
            activation_reason=assessment.reason,
        )
    raise ValueError("invalid action assessments cannot enter service.")


def relabel_refinement_as_move(
    item: FrozenServiceItem,
    action: CertifiedPathAction,
) -> FrozenServiceItem:
    """Transfer the service count when unresolved evidence becomes a move."""

    if item.tag is not ServiceTag.REFINEMENT:
        raise ValueError("only refinement service may be relabeled.")
    if item.action_key != action.key:
        raise ValueError("resolved action identity differs from refinement.")
    if item.eligibility_token_digest != action.eligibility_token.digest:
        raise ValueError("resolved action eligibility token changed.")
    if item.energy_unit_digest != action.energy_unit.digest:
        raise ValueError("resolved action energy unit changed.")
    return FrozenServiceItem(
        tag=ServiceTag.MOVE,
        action_key=item.action_key,
        frozen_entitlement=action.utility.live_entitlement,
        service_count=item.service_count,
        service_epoch=item.service_epoch,
        eligibility_token_digest=item.eligibility_token_digest,
        energy_unit_digest=item.energy_unit_digest,
        activation_reason="refinement_resolved_to_move",
    )


@dataclass(frozen=True)
class FairServiceDecision(_DeterministicSerializable):
    selected: tuple[FrozenServiceItem, ...]
    updated_population: tuple[FrozenServiceItem, ...]
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected", tuple(self.selected))
        object.__setattr__(
            self, "updated_population", tuple(self.updated_population)
        )
        _nonempty("reason", self.reason)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [item.to_dict() for item in self.selected],
            "updated_population": [
                item.to_dict() for item in self.updated_population
            ],
            "reason": self.reason,
        }


def serve_fair_service(
    population: tuple[FrozenServiceItem, ...],
    *,
    batch_size: int = 1,
) -> FairServiceDecision:
    """Apply ``FirstDistinct_B`` using exact virtual-finish comparisons."""

    items = tuple(population)
    if not items:
        return FairServiceDecision(
            selected=(),
            updated_population=(),
            reason="represented_action_population_empty",
        )
    capacity = _positive_index("batch_size", batch_size)
    epochs = {item.service_epoch for item in items}
    eligibility_tokens = {item.eligibility_token_digest for item in items}
    energy_units = {item.energy_unit_digest for item in items}
    if len(epochs) != 1:
        raise ValueError("service population spans multiple epochs.")
    if len(eligibility_tokens) != 1:
        raise ValueError("service population spans multiple eligibility tokens.")
    if len(energy_units) != 1:
        raise ValueError("service population spans multiple energy units.")
    keys = tuple(item.action_key for item in items)
    if len(set(keys)) != len(keys):
        raise ValueError("an action may have only one live service tag.")
    indices = tuple(key.action_index for key in keys)
    if len(set(indices)) != len(indices):
        raise ValueError("canonical action indices must be unique.")
    ordered = sorted(
        items,
        key=lambda item: (
            item.scaled_virtual_finish,
            item.deterministic_order_key,
        ),
    )
    selected = tuple(ordered[: min(capacity, len(ordered))])
    selected_keys = {item.action_key for item in selected}
    updated = tuple(
        replace(item, service_count=item.service_count + 1)
        if item.action_key in selected_keys
        else item
        for item in items
    )
    return FairServiceDecision(
        selected=selected,
        updated_population=updated,
        reason="activation_frozen_fair_service_selected",
    )


@dataclass(frozen=True)
class DisposablePowellProbe:
    """Seed-preserving Powell outcomes evaluated only on a disposable copy."""

    completed: bool
    simultaneous: bool
    comparison_epoch: str | None
    one_sided_error_bound: float | None
    outcomes: tuple[EnergyInterval, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcomes", tuple(self.outcomes))


@dataclass(frozen=True)
class PowellPromotionDecision(_DeterministicSerializable):
    promote: bool
    comparison_valid: bool
    reason: str
    incumbent_state_id: str
    constrained_working_state_id: str
    selected_state: EnergyInterval
    one_sided_error_bound: float | None
    promotion_margin_lower_bound: float | None

    def __post_init__(self) -> None:
        _nonempty("reason", self.reason)
        _nonempty("incumbent_state_id", self.incumbent_state_id)
        _nonempty(
            "constrained_working_state_id", self.constrained_working_state_id
        )
        if self.promote and not self.comparison_valid:
            raise ValueError("promotion requires valid simultaneous comparison.")

    def to_dict(self) -> dict[str, object]:
        return {
            "promote": self.promote,
            "comparison_valid": self.comparison_valid,
            "reason": self.reason,
            "incumbent_state_id": self.incumbent_state_id,
            "constrained_working_state_id": self.constrained_working_state_id,
            "selected_state": self.selected_state.to_dict(),
            "one_sided_error_bound": self.one_sided_error_bound,
            "promotion_margin_lower_bound": self.promotion_margin_lower_bound,
        }


def decide_disposable_powell_promotion(
    *,
    incumbent: EnergyInterval,
    constrained_working: EnergyInterval,
    probe: DisposablePowellProbe,
) -> PowellPromotionDecision:
    """Retain the seed and promote only under the strict simultaneous margin."""

    def fail(reason: str) -> PowellPromotionDecision:
        return PowellPromotionDecision(
            promote=False,
            comparison_valid=False,
            reason=reason,
            incumbent_state_id=incumbent.state_id,
            constrained_working_state_id=constrained_working.state_id,
            selected_state=constrained_working,
            one_sided_error_bound=None,
            promotion_margin_lower_bound=None,
        )

    if not probe.completed:
        return fail("powell_probe_failed_or_incomplete")
    if not probe.simultaneous:
        return fail("powell_probe_comparison_not_simultaneous")
    if probe.comparison_epoch is None or not str(probe.comparison_epoch).strip():
        return fail("powell_probe_comparison_epoch_missing")
    if probe.one_sided_error_bound is None:
        return fail("powell_one_sided_error_unresolved")
    if not _is_finite_real(probe.one_sided_error_bound):
        return fail("powell_one_sided_error_nonfinite")
    powell_error = float(probe.one_sided_error_bound)
    if powell_error < 0.0:
        return fail("powell_one_sided_error_negative")
    candidates = (constrained_working, *probe.outcomes)
    if not incumbent.simultaneous or any(
        not state.simultaneous for state in candidates
    ):
        return fail("state_energy_intervals_not_simultaneous")
    epoch = str(probe.comparison_epoch)
    if incumbent.comparison_epoch != epoch or any(
        state.comparison_epoch != epoch for state in candidates
    ):
        return fail("state_energy_comparison_epoch_mismatch")
    selected = min(
        enumerate(candidates),
        key=lambda indexed: (
            indexed[1].upper_bound,
            indexed[0],
            indexed[1].state_id,
        ),
    )[1]
    margin = incumbent.lower_bound - (selected.upper_bound + powell_error)
    if not math.isfinite(margin):
        return fail("powell_promotion_margin_nonfinite")
    promote = margin > 0.0
    return PowellPromotionDecision(
        promote=promote,
        comparison_valid=True,
        reason=(
            "powell_outcome_strictly_defeats_incumbent"
            if promote
            else "powell_outcome_does_not_strictly_defeat_incumbent"
        ),
        incumbent_state_id=incumbent.state_id,
        constrained_working_state_id=constrained_working.state_id,
        selected_state=selected,
        one_sided_error_bound=powell_error,
        promotion_margin_lower_bound=margin,
    )


@dataclass(frozen=True)
class BranchStateSnapshot(_DeterministicSerializable):
    """Branch-local scientific incumbent and exploratory working state."""

    incumbent: EnergyInterval
    working: EnergyInterval
    exclusion_radius: float

    def __post_init__(self) -> None:
        radius = _nonnegative("exclusion_radius", self.exclusion_radius)
        if radius > _FS_DIAMETER:
            raise ValueError("exclusion_radius cannot exceed pi/2.")
        same_state = self.incumbent.state_id == self.working.state_id
        if radius == 0.0 and not same_state:
            raise ValueError("zero exclusion radius requires X=I.")
        if radius > 0.0 and same_state:
            raise ValueError("positive exclusion radius requires X distinct from I.")

    def to_dict(self) -> dict[str, object]:
        return {
            "incumbent": self.incumbent.to_dict(),
            "working": self.working.to_dict(),
            "exclusion_radius": self.exclusion_radius,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "BranchStateSnapshot":
        return cls(
            incumbent=EnergyInterval.from_dict(data["incumbent"]),
            working=EnergyInterval.from_dict(data["working"]),
            exclusion_radius=float(data["exclusion_radius"]),
        )


@dataclass(frozen=True)
class ConstrainedWorkingState(_DeterministicSerializable):
    """Seed-retaining constrained refit in one certified FS component."""

    seed: EnergyInterval
    state: EnergyInterval
    action_path_id: str
    component_id: str
    refit_witness_id: str
    refit_completed: bool
    seed_retained: bool
    simultaneous: bool
    exclusion_certificate: FSExclusionCertificate

    def __post_init__(self) -> None:
        for name in ("action_path_id", "component_id", "refit_witness_id"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if not self.seed_retained:
            raise ValueError("constrained refit must retain the feasible seed.")
        if (
            not self.simultaneous
            or not self.seed.simultaneous
            or not self.state.simultaneous
        ):
            raise ValueError("constrained refit comparison must be simultaneous.")
        if self.seed.comparison_epoch != self.state.comparison_epoch:
            raise ValueError("seed/refit energy comparison epoch mismatch.")
        if not self.refit_completed and self.state != self.seed:
            raise ValueError("failed refit must return the retained seed exactly.")
        if self.state.upper_bound > self.seed.upper_bound:
            raise ValueError("chosen constrained refit is worse than feasible seed.")
        certificate = self.exclusion_certificate
        if certificate.path_origin_state_id != self.seed.state_id:
            raise ValueError("refit exclusion path must begin at retained seed.")
        if certificate.path_endpoint_state_id != self.state.state_id:
            raise ValueError("refit exclusion path must end at chosen state.")
        if certificate.component_id != self.component_id:
            raise ValueError("refit exclusion component identifier mismatch.")
        if certificate.witness_id != self.refit_witness_id:
            raise ValueError("refit exclusion witness identifier mismatch.")
        if certificate.comparison_epoch != self.state.comparison_epoch:
            raise ValueError("refit exclusion comparison epoch mismatch.")

    @property
    def seed_state_id(self) -> str:
        return self.seed.state_id

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed.to_dict(),
            "state": self.state.to_dict(),
            "action_path_id": self.action_path_id,
            "component_id": self.component_id,
            "refit_witness_id": self.refit_witness_id,
            "refit_completed": self.refit_completed,
            "seed_retained": self.seed_retained,
            "simultaneous": self.simultaneous,
            "exclusion_certificate": self.exclusion_certificate.to_dict(),
        }


@dataclass(frozen=True)
class ExploratoryTransaction(_DeterministicSerializable):
    accepted: bool
    promoted: bool
    reason: str
    previous_state: BranchStateSnapshot
    next_state: BranchStateSnapshot
    promotion_decision: PowellPromotionDecision | None

    def __post_init__(self) -> None:
        _nonempty("reason", self.reason)
        if self.promoted and not self.accepted:
            raise ValueError("rejected transition cannot promote.")

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "promoted": self.promoted,
            "reason": self.reason,
            "previous_state": self.previous_state.to_dict(),
            "next_state": self.next_state.to_dict(),
            "promotion_decision": (
                None
                if self.promotion_decision is None
                else self.promotion_decision.to_dict()
            ),
        }


def execute_exploratory_transaction(
    *,
    branch: BranchStateSnapshot,
    action: CertifiedPathAction,
    constrained_working: ConstrainedWorkingState,
    probe: DisposablePowellProbe,
) -> ExploratoryTransaction:
    """Apply the exact immutable ``(I,X,chi)`` Stage-B transaction."""

    def reject(reason: str) -> ExploratoryTransaction:
        return ExploratoryTransaction(
            accepted=False,
            promoted=False,
            reason=reason,
            previous_state=branch,
            next_state=branch,
            promotion_decision=None,
        )

    if action.eligibility_token.working_state_fingerprint != branch.working.state_id:
        return reject("action_eligibility_working_state_mismatch")
    expected_action_receipt = canonical_action_receipt_digest(
        action.key,
        action.eligibility_token.digest,
    )
    action_receipts = {
        action.trust_path_certificate.action_receipt_digest,
        action.barrier_certificate.action_receipt_digest,
        action.endpoint_distance_certificate.action_receipt_digest,
        action.exclusion_certificate.action_receipt_digest,
    }
    if action_receipts != {expected_action_receipt}:
        return reject("action_certificate_receipt_binding_mismatch")
    if action.origin_state_id != branch.working.state_id:
        return reject("action_origin_does_not_match_working_state")
    if action.barrier_certificate.incumbent_energy != branch.incumbent:
        return reject("barrier_incumbent_energy_does_not_match_branch")
    action_exclusion = action.exclusion_certificate
    if action_exclusion.incumbent_state_id != branch.incumbent.state_id:
        return reject("action_exclusion_incumbent_mismatch")
    if action_exclusion.current_exclusion_radius != branch.exclusion_radius:
        return reject("action_exclusion_radius_epoch_mismatch")
    proposed_exclusion = (
        branch.exclusion_radius
        if branch.exclusion_radius > 0.0
        else action_exclusion.endpoint_distance_lower_bound
    )
    if constrained_working.seed != action.endpoint_seed_energy:
        return reject("constrained_refit_seed_energy_mismatch")
    if constrained_working.action_path_id != action.trust_path_certificate.path_id:
        return reject("constrained_refit_action_path_mismatch")
    if constrained_working.component_id != action_exclusion.component_id:
        return reject("constrained_refit_component_mismatch")
    working_exclusion = constrained_working.exclusion_certificate
    if working_exclusion.action_receipt_digest != expected_action_receipt:
        return reject("working_exclusion_action_receipt_mismatch")
    if working_exclusion.incumbent_state_id != branch.incumbent.state_id:
        return reject("working_exclusion_incumbent_mismatch")
    if working_exclusion.current_exclusion_radius != proposed_exclusion:
        return reject("working_exclusion_radius_mismatch")
    if constrained_working.state.state_id == branch.incumbent.state_id:
        return reject("constrained_working_state_collapsed_to_incumbent")
    promotion = decide_disposable_powell_promotion(
        incumbent=branch.incumbent,
        constrained_working=constrained_working.state,
        probe=probe,
    )
    if promotion.promote:
        selected = promotion.selected_state
        next_state = BranchStateSnapshot(
            incumbent=selected,
            working=selected,
            exclusion_radius=0.0,
        )
        return ExploratoryTransaction(
            accepted=True,
            promoted=True,
            reason="exploratory_working_state_promoted",
            previous_state=branch,
            next_state=next_state,
            promotion_decision=promotion,
        )
    next_state = BranchStateSnapshot(
        incumbent=branch.incumbent,
        working=constrained_working.state,
        exclusion_radius=proposed_exclusion,
    )
    return ExploratoryTransaction(
        accepted=True,
        promoted=False,
        reason="exploratory_working_state_preserved_without_promotion",
        previous_state=branch,
        next_state=next_state,
        promotion_decision=promotion,
    )


@dataclass(frozen=True)
class RuntimeHistoryEvent(_DeterministicSerializable):
    event_index: int
    event_kind: str
    working_state_fingerprint: str
    details_digest: str
    action_index: int | None = None

    def __post_init__(self) -> None:
        _nonnegative_index("event_index", self.event_index)
        for name in (
            "event_kind",
            "working_state_fingerprint",
            "details_digest",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if self.action_index is not None:
            _positive_index("action_index", self.action_index)

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": _encode_nonnegative_integer(self.event_index),
            "event_kind": self.event_kind,
            "working_state_fingerprint": self.working_state_fingerprint,
            "details_digest": self.details_digest,
            "action_index": (
                None
                if self.action_index is None
                else _encode_nonnegative_integer(self.action_index)
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "RuntimeHistoryEvent":
        return cls(
            event_index=_decode_nonnegative_integer(
                "event_index", data["event_index"]
            ),
            event_kind=str(data["event_kind"]),
            working_state_fingerprint=str(data["working_state_fingerprint"]),
            details_digest=str(data["details_digest"]),
            action_index=(
                None
                if data["action_index"] is None
                else _decode_positive_integer(
                    "action_index", data["action_index"]
                )
            ),
        )


@dataclass(frozen=True)
class ModeledMinimumRuntimeState(_DeterministicSerializable):
    """Immutable pure-core scheduler state, queue, history, and clocks.

    This is not a replayable circuit/optimizer checkpoint.  It deliberately
    omits ansatz parameters, exact states, optimizer internals, path-enclosure
    payloads, and integration-owned runtime handles.
    """

    eligibility_token: EligibilityStateToken
    energy_unit: RunEnergyUnit
    branch: BranchStateSnapshot
    ordinary_trust_radius: float
    mode: ControllerMode
    service_epoch: str
    service_population: tuple[FrozenServiceItem, ...]
    history: tuple[RuntimeHistoryEvent, ...]
    next_event_index: int
    schema_version: str = field(default=_RUNTIME_SCHEMA, init=False)
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    checkpoint_scope: str = field(default=_CHECKPOINT_SCOPE, init=False)
    runtime_resume_complete: bool = field(default=False, init=False)
    integration_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _positive("ordinary_trust_radius", self.ordinary_trust_radius)
        object.__setattr__(self, "mode", ControllerMode(self.mode))
        object.__setattr__(
            self, "service_epoch", _nonempty("service_epoch", self.service_epoch)
        )
        population = tuple(self.service_population)
        history = tuple(self.history)
        object.__setattr__(self, "service_population", population)
        object.__setattr__(self, "history", history)
        next_index = _nonnegative_index("next_event_index", self.next_event_index)
        if (
            self.branch.working.state_id
            != self.eligibility_token.working_state_fingerprint
        ):
            raise ValueError("runtime branch is stale relative to eligibility token.")
        if self.ordinary_trust_radius != self.eligibility_token.trust_radius:
            raise ValueError("runtime trust radius differs from eligibility token.")
        if (
            self.branch.exclusion_radius > 0.0
            and self.mode is not ControllerMode.EXPLORE
        ):
            raise ValueError("positive exclusion radius requires explore mode.")
        keys = tuple(item.action_key for item in population)
        if len(set(keys)) != len(keys):
            raise ValueError("runtime queue contains duplicate action tags.")
        indices = tuple(key.action_index for key in keys)
        if len(set(indices)) != len(indices):
            raise ValueError("runtime queue contains duplicate action indices.")
        for item in population:
            if item.service_epoch != self.service_epoch:
                raise ValueError("queue item service epoch mismatch.")
            if item.eligibility_token_digest != self.eligibility_token.digest:
                raise ValueError("queue item eligibility token mismatch.")
            if item.energy_unit_digest != self.energy_unit.digest:
                raise ValueError("queue item energy unit mismatch.")
            key = item.action_key
            if key.record_count != len(self.eligibility_token.reachable_record_ids):
                raise ValueError("queue action record count mismatch.")
            if (
                self.eligibility_token.reachable_record_ids[key.record_order - 1]
                != key.record_id
            ):
                raise ValueError("queue action record order mismatch.")
        event_indices = tuple(event.event_index for event in history)
        if event_indices != tuple(sorted(set(event_indices))):
            raise ValueError("history event indices must be unique and ordered.")
        if history and next_index <= history[-1].event_index:
            raise ValueError("next_event_index must exceed history.")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_index_schema": self.action_index_schema,
            "checkpoint_scope": self.checkpoint_scope,
            "runtime_resume_complete": self.runtime_resume_complete,
            "integration_ready": self.integration_ready,
            "eligibility_token": self.eligibility_token.to_dict(),
            "energy_unit": self.energy_unit.to_dict(),
            "branch": self.branch.to_dict(),
            "ordinary_trust_radius": self.ordinary_trust_radius,
            "mode": self.mode.value,
            "service_epoch": self.service_epoch,
            "service_population": [
                item.to_dict() for item in self.service_population
            ],
            "history": [event.to_dict() for event in self.history],
            "next_event_index": _encode_nonnegative_integer(
                self.next_event_index
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ModeledMinimumRuntimeState":
        if data.get("schema_version") != _RUNTIME_SCHEMA:
            raise ValueError("unsupported modeled-minimum runtime schema.")
        if data.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("unsupported modeled-minimum action-index schema.")
        if data.get("checkpoint_scope") != _CHECKPOINT_SCOPE:
            raise ValueError("unsupported modeled-minimum checkpoint scope.")
        if data.get("runtime_resume_complete") is not False:
            raise ValueError("pure-core checkpoint cannot claim runtime resume.")
        if data.get("integration_ready") is not False:
            raise ValueError("pure-core checkpoint cannot claim integration.")
        return cls(
            eligibility_token=EligibilityStateToken.from_dict(
                data["eligibility_token"]
            ),
            energy_unit=RunEnergyUnit.from_dict(data["energy_unit"]),
            branch=BranchStateSnapshot.from_dict(data["branch"]),
            ordinary_trust_radius=float(data["ordinary_trust_radius"]),
            mode=ControllerMode(str(data["mode"])),
            service_epoch=str(data["service_epoch"]),
            service_population=tuple(
                FrozenServiceItem.from_dict(item)
                for item in data["service_population"]
            ),
            history=tuple(
                RuntimeHistoryEvent.from_dict(event)
                for event in data["history"]
            ),
            next_event_index=_decode_nonnegative_integer(
                "next_event_index", data["next_event_index"]
            ),
        )


@dataclass(frozen=True)
class ModeledMinimumCheckpoint(_DeterministicSerializable):
    """Content-addressed pure-core scheduler checkpoint, not runtime resume."""

    runtime: ModeledMinimumRuntimeState
    content_digest: str
    schema_version: str = field(default=_CHECKPOINT_SCHEMA, init=False)
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    checkpoint_scope: str = field(default=_CHECKPOINT_SCOPE, init=False)
    runtime_resume_complete: bool = field(default=False, init=False)
    integration_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "content_digest",
            _nonempty("content_digest", self.content_digest),
        )
        if self.content_digest != _digest(self.runtime.to_dict()):
            raise ValueError("checkpoint content digest mismatch.")

    @classmethod
    def create(
        cls,
        runtime: ModeledMinimumRuntimeState,
    ) -> "ModeledMinimumCheckpoint":
        return cls(runtime=runtime, content_digest=_digest(runtime.to_dict()))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_index_schema": self.action_index_schema,
            "checkpoint_scope": self.checkpoint_scope,
            "runtime_resume_complete": self.runtime_resume_complete,
            "integration_ready": self.integration_ready,
            "runtime": self.runtime.to_dict(),
            "content_digest": self.content_digest,
        }

    @classmethod
    def from_json(cls, payload: str) -> "ModeledMinimumCheckpoint":
        try:
            data = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint is not valid JSON.") from exc
        if (
            not isinstance(data, dict)
            or data.get("schema_version") != _CHECKPOINT_SCHEMA
        ):
            raise ValueError("unsupported modeled-minimum checkpoint schema.")
        if data.get("checkpoint_scope") != _CHECKPOINT_SCOPE:
            raise ValueError("unsupported modeled-minimum checkpoint scope.")
        if data.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("unsupported modeled-minimum action-index schema.")
        if data.get("runtime_resume_complete") is not False:
            raise ValueError("pure-core checkpoint cannot claim runtime resume.")
        if data.get("integration_ready") is not False:
            raise ValueError("pure-core checkpoint cannot claim integration.")
        runtime_data = data.get("runtime")
        if not isinstance(runtime_data, dict):
            raise ValueError("checkpoint runtime payload is missing.")
        expected = _digest(runtime_data)
        if data.get("content_digest") != expected:
            raise ValueError("checkpoint content digest mismatch.")
        try:
            runtime = ModeledMinimumRuntimeState.from_dict(runtime_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("checkpoint runtime payload is invalid.") from exc
        if runtime.to_dict() != runtime_data:
            raise ValueError("checkpoint runtime canonical round-trip mismatch.")
        return cls(runtime=runtime, content_digest=expected)


__all__ = [
    "ACTION_INDEX_SCHEMA",
    "BarrierDistanceUtility",
    "BranchStateSnapshot",
    "CanonicalActionCoordinates",
    "CanonicalActionMass",
    "CertificateState",
    "CertifiedPathAction",
    "ConstrainedWorkingState",
    "ControllerMode",
    "DisposablePowellProbe",
    "EndpointDistanceCertificate",
    "EndpointDistanceEvidence",
    "EnergyInterval",
    "EligibilityStateToken",
    "ExposedFamilyEligibility",
    "ExploratoryTransaction",
    "FSExclusionAssessment",
    "FSExclusionCertificate",
    "FSExclusionEvidence",
    "FairServiceDecision",
    "FrozenServiceItem",
    "LogEntitlement",
    "ModeledMinimumCheckpoint",
    "ModeledMinimumRuntimeState",
    "PathActionAssessment",
    "PathActionEvidence",
    "PathActionKey",
    "PathOrientation",
    "PositiveRational",
    "PowellPromotionDecision",
    "RefinementAction",
    "ResolutionKind",
    "RunEnergyUnit",
    "RuntimeHistoryEvent",
    "ServiceTag",
    "StabilizedTrustPathCertificate",
    "StabilizedTrustPathEvidence",
    "UniformBarrierCertificate",
    "UniformBarrierEvidence",
    "activate_service_item",
    "assess_exposed_family_psd",
    "assess_fs_exclusion",
    "assess_path_action",
    "calkin_wilf_index",
    "calkin_wilf_rational",
    "canonical_action_index",
    "canonical_action_mass",
    "canonical_action_receipt_digest",
    "compute_barrier_distance_utility",
    "decide_disposable_powell_promotion",
    "execute_exploratory_transaction",
    "inverse_action_index",
    "relabel_refinement_as_move",
    "serve_fair_service",
]
