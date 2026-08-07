"""Fail-closed Stage-B active-manifold distance envelope.

This module deliberately covers one narrow exact-simulator case only.  The
active-only reference must be a computational-basis state and every active
primitive must be a Pauli word.  In that case, Pauli evolution cannot leave the
affine computational-basis orbit

``b_ref xor span_GF(2)({X/Y flip masks})``.

The orbit span is a *linear superset* of the nonlinear active-only manifold.
Consequently the Fubini--Study distance from an endpoint to that span is a
rigorous lower bound on its distance to the nonlinear manifold.  The bound is
conservative: endpoint-vector error is added to the projected norm before the
inverse cosine is taken.  Any unsupported reference, stale receipt binding,
resource-cap breach, or saturated error bound remains unresolved.

Nothing here mutates SR-SNAKE runtime state or enables combined Stage B.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import hashlib
import json
import math
from typing import Iterable, Sequence

from pipelines.static_adapt.sr_snake_modeled_minimum import (
    EligibilityStateToken,
    PathActionKey,
    canonical_action_receipt_digest,
)


ACTIVE_MANIFOLD_DISTANCE_SCHEMA = (
    "sr_snake_active_manifold_affine_pauli_envelope_v1"
)
ACTIVE_MANIFOLD_EXECUTION_MODE = (
    "computational_basis_primitive_pauli_support_envelope_v1"
)
COMPUTATIONAL_BASIS_REFERENCE_KIND = "computational_basis"
PAULI_WORD_ORDERING = "left_to_right_q_n_minus_1_to_q_0"
_SUPPORTED_PAULI_LETTERS = frozenset("exyz")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _nonempty(name: str, value: object) -> str:
    resolved = str(value).strip()
    if not resolved:
        raise ValueError(f"{name} must be nonempty.")
    return resolved


def _nonnegative_integer(name: str, value: object) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a nonnegative integer.")
    try:
        resolved = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a nonnegative integer.") from exc
    if resolved != value or resolved < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return resolved


def _positive_integer(name: str, value: object) -> int:
    resolved = _nonnegative_integer(name, value)
    if resolved <= 0:
        raise ValueError(f"{name} must be positive.")
    return resolved


def _finite_nonnegative(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and nonnegative.")
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and nonnegative.") from exc
    if not math.isfinite(resolved) or resolved < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return resolved


def _hex_integer(value: int) -> str:
    return f"0x{_nonnegative_integer('integer', value):x}"


def _parse_hex_integer(name: str, value: object) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{name} must use canonical hexadecimal encoding.")
    digits = value[2:]
    if not digits or any(character not in "0123456789abcdef" for character in digits):
        raise ValueError(f"{name} must use canonical hexadecimal encoding.")
    if len(digits) > 1 and digits.startswith("0"):
        raise ValueError(f"{name} has a noncanonical leading zero.")
    resolved = int(digits, 16)
    if _hex_integer(resolved) != value:
        raise ValueError(f"{name} is not canonical hexadecimal data.")
    return resolved


def _fraction_to_dict(value: Fraction | None) -> dict[str, str] | None:
    if value is None:
        return None
    if value < 0:
        raise ValueError("serialized probability must be nonnegative.")
    return {
        "numerator": _hex_integer(value.numerator),
        "denominator": _hex_integer(value.denominator),
    }


def _fraction_from_dict(name: str, value: object) -> Fraction | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be canonical rational data.")
    if set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} must be canonical rational data.")
    numerator = _parse_hex_integer(f"{name}.numerator", value["numerator"])
    denominator = _parse_hex_integer(f"{name}.denominator", value["denominator"])
    if denominator <= 0:
        raise ValueError(f"{name}.denominator must be positive.")
    resolved = Fraction(numerator, denominator)
    if _fraction_to_dict(resolved) != value:
        raise ValueError(f"{name} is not reduced canonical rational data.")
    return resolved


@dataclass(frozen=True)
class PrimitivePauliSupport:
    """One nonzero primitive term in the active support envelope.

    The coefficient is validated as finite and nonzero, then intentionally
    omitted from canonical support identity.  Thus nonzero coefficient
    rescaling cannot change the affine-support certificate.
    """

    pauli_word: str
    coefficient: complex = 1.0 + 0.0j


def _validated_support_words(
    primitives: Iterable[PrimitivePauliSupport],
    *,
    qubit_count: int,
) -> tuple[str, ...]:
    count = _positive_integer("qubit_count", qubit_count)
    words: set[str] = set()
    for term in primitives:
        if not isinstance(term, PrimitivePauliSupport):
            raise ValueError("active support contains a non-primitive Pauli term.")
        word = str(term.pauli_word)
        if len(word) != count or any(letter not in _SUPPORTED_PAULI_LETTERS for letter in word):
            raise ValueError(
                "primitive Pauli words must use exactly n lower-case e/x/y/z letters."
            )
        try:
            coefficient = complex(term.coefficient)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("primitive Pauli coefficient is invalid.") from exc
        if (
            not math.isfinite(coefficient.real)
            or not math.isfinite(coefficient.imag)
            or coefficient == 0.0
        ):
            raise ValueError("primitive Pauli coefficients must be finite and nonzero.")
        words.add(word)
    return tuple(sorted(words))


def canonical_active_support_digest(
    primitives: Iterable[PrimitivePauliSupport],
    *,
    qubit_count: int,
) -> str:
    """Return order/duplicate/nonzero-rescaling invariant support identity."""

    count = _positive_integer("qubit_count", qubit_count)
    words = _validated_support_words(primitives, qubit_count=count)
    return _digest(
        {
            "schema": ACTIVE_MANIFOLD_DISTANCE_SCHEMA,
            "kind": "canonical_primitive_pauli_support",
            "qubit_count": count,
            "pauli_word_ordering": PAULI_WORD_ORDERING,
            "canonical_pauli_words": list(words),
        }
    )


def canonical_active_layout_digest(*, qubit_count: int) -> str:
    count = _positive_integer("qubit_count", qubit_count)
    return _digest(
        {
            "schema": ACTIVE_MANIFOLD_DISTANCE_SCHEMA,
            "kind": "computational_basis_pauli_layout",
            "qubit_count": count,
            "pauli_word_ordering": PAULI_WORD_ORDERING,
        }
    )


def canonical_active_execution_mode_digest() -> str:
    return _digest(
        {
            "schema": ACTIVE_MANIFOLD_DISTANCE_SCHEMA,
            "execution_mode": ACTIVE_MANIFOLD_EXECUTION_MODE,
        }
    )


def canonical_active_radius_digest(
    *,
    action_key: PathActionKey,
    eligibility_token: EligibilityStateToken,
) -> str:
    return _digest(
        {
            "schema": ACTIVE_MANIFOLD_DISTANCE_SCHEMA,
            "kind": "action_radius_binding",
            "action_key": action_key.to_dict(),
            "radius": action_key.radius.to_dict(),
            "reference_trust_radius": eligibility_token.trust_radius,
            "trust_provenance_digest": eligibility_token.trust_provenance_digest,
        }
    )


@dataclass(frozen=True)
class ActiveManifoldDistanceBindings:
    """Frozen Stage-B provenance required by the narrow provider."""

    eligibility_token: EligibilityStateToken
    working_state_fingerprint: str
    reference_state_fingerprint: str
    endpoint_state_fingerprint: str
    comparison_epoch: str
    branch_epoch: str
    active_support_digest: str
    layout_digest: str
    execution_mode_digest: str
    support_provenance_digest: str
    trust_provenance_digest: str
    radius_digest: str
    action_receipt_digest: str
    path_digest: str
    sector_digest: str
    padding_digest: str
    action_key: PathActionKey

    def __post_init__(self) -> None:
        for name in (
            "working_state_fingerprint",
            "reference_state_fingerprint",
            "endpoint_state_fingerprint",
            "comparison_epoch",
            "branch_epoch",
            "active_support_digest",
            "layout_digest",
            "execution_mode_digest",
            "support_provenance_digest",
            "trust_provenance_digest",
            "radius_digest",
            "action_receipt_digest",
            "path_digest",
            "sector_digest",
            "padding_digest",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "eligibility_token": self.eligibility_token.to_dict(),
            "eligibility_token_digest": self.eligibility_token.digest,
            "working_state_fingerprint": self.working_state_fingerprint,
            "reference_state_fingerprint": self.reference_state_fingerprint,
            "endpoint_state_fingerprint": self.endpoint_state_fingerprint,
            "comparison_epoch": self.comparison_epoch,
            "branch_epoch": self.branch_epoch,
            "active_support_digest": self.active_support_digest,
            "layout_digest": self.layout_digest,
            "execution_mode_digest": self.execution_mode_digest,
            "support_provenance_digest": self.support_provenance_digest,
            "trust_provenance_digest": self.trust_provenance_digest,
            "radius_digest": self.radius_digest,
            "action_receipt_digest": self.action_receipt_digest,
            "path_digest": self.path_digest,
            "sector_digest": self.sector_digest,
            "padding_digest": self.padding_digest,
            "action_key": self.action_key.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ActiveManifoldDistanceBindings":
        token = EligibilityStateToken.from_dict(dict(data["eligibility_token"]))
        if data.get("eligibility_token_digest") != token.digest:
            raise ValueError("eligibility token digest is stale or tampered.")
        return cls(
            eligibility_token=token,
            working_state_fingerprint=str(data["working_state_fingerprint"]),
            reference_state_fingerprint=str(data["reference_state_fingerprint"]),
            endpoint_state_fingerprint=str(data["endpoint_state_fingerprint"]),
            comparison_epoch=str(data["comparison_epoch"]),
            branch_epoch=str(data["branch_epoch"]),
            active_support_digest=str(data["active_support_digest"]),
            layout_digest=str(data["layout_digest"]),
            execution_mode_digest=str(data["execution_mode_digest"]),
            support_provenance_digest=str(data["support_provenance_digest"]),
            trust_provenance_digest=str(data["trust_provenance_digest"]),
            radius_digest=str(data["radius_digest"]),
            action_receipt_digest=str(data["action_receipt_digest"]),
            path_digest=str(data["path_digest"]),
            sector_digest=str(data["sector_digest"]),
            padding_digest=str(data["padding_digest"]),
            action_key=PathActionKey.from_dict(dict(data["action_key"])),
        )


@dataclass(frozen=True)
class ActiveManifoldDistanceRequest:
    bindings: ActiveManifoldDistanceBindings
    qubit_count: int
    reference_kind: str
    reference_bitstring: str | None
    primitive_support: tuple[PrimitivePauliSupport, ...]
    endpoint_amplitudes: tuple[complex, ...]
    endpoint_l2_error_bound: float
    max_gf2_rank: int
    max_orbit_size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "primitive_support", tuple(self.primitive_support))
        object.__setattr__(self, "endpoint_amplitudes", tuple(self.endpoint_amplitudes))


class ActiveManifoldDistanceStatus(str, Enum):
    CERTIFIED_POSITIVE = "certified_positive"
    CERTIFIED_ZERO = "certified_zero"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ActiveManifoldDistanceResult:
    """Content-addressed result/receipt for one endpoint and action."""

    status: ActiveManifoldDistanceStatus
    reason: str
    bindings: ActiveManifoldDistanceBindings
    reference_kind: str
    reference_bitstring: str | None
    qubit_count: int
    max_gf2_rank: int
    max_orbit_size: int
    canonical_pauli_words: tuple[str, ...] = ()
    gf2_basis_masks: tuple[int, ...] = ()
    affine_orbit_indices: tuple[int, ...] = ()
    gf2_rank: int | None = None
    endpoint_l2_error_bound: float | None = None
    endpoint_norm_squared_exact: Fraction | None = None
    projection_norm_squared_exact: Fraction | None = None
    projection_norm_squared_upper_bound: float | None = None
    projection_norm_upper_bound: float | None = None
    distance_lower_bound: float | None = None
    distance_lower_bound_squared: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", ActiveManifoldDistanceStatus(self.status))
        _nonempty("reason", self.reason)
        if self.status is ActiveManifoldDistanceStatus.CERTIFIED_POSITIVE:
            if (
                self.distance_lower_bound is None
                or self.distance_lower_bound_squared is None
                or self.projection_norm_upper_bound is None
                or self.projection_norm_squared_upper_bound is None
            ):
                raise ValueError("positive result requires complete numerical bounds.")
            if not (0.0 < self.distance_lower_bound <= math.pi / 2.0):
                raise ValueError("certified distance lower bound is out of range.")
            if not (0.0 <= self.projection_norm_upper_bound < 1.0):
                raise ValueError("positive result requires projected norm below one.")
        elif self.status is ActiveManifoldDistanceStatus.CERTIFIED_ZERO:
            if self.distance_lower_bound != 0.0 or self.distance_lower_bound_squared != 0.0:
                raise ValueError("zero certificate must carry exact zero distance.")
        elif self.distance_lower_bound is not None or self.distance_lower_bound_squared is not None:
            raise ValueError("unresolved result cannot carry a distance certificate.")

    @property
    def certified(self) -> bool:
        return self.status is not ActiveManifoldDistanceStatus.UNRESOLVED

    @property
    def positive(self) -> bool:
        return self.status is ActiveManifoldDistanceStatus.CERTIFIED_POSITIVE

    @property
    def receipt_digest(self) -> str:
        return _digest(self._payload_dict())

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema": ACTIVE_MANIFOLD_DISTANCE_SCHEMA,
            "status": self.status.value,
            "reason": self.reason,
            "bindings": self.bindings.to_dict(),
            "reference_kind": self.reference_kind,
            "reference_bitstring": self.reference_bitstring,
            "qubit_count": self.qubit_count,
            "max_gf2_rank": self.max_gf2_rank,
            "max_orbit_size": self.max_orbit_size,
            "canonical_pauli_words": list(self.canonical_pauli_words),
            "gf2_basis_masks": [_hex_integer(value) for value in self.gf2_basis_masks],
            "affine_orbit_indices": [
                _hex_integer(value) for value in self.affine_orbit_indices
            ],
            "gf2_rank": self.gf2_rank,
            "endpoint_l2_error_bound": self.endpoint_l2_error_bound,
            "endpoint_norm_squared_exact": _fraction_to_dict(
                self.endpoint_norm_squared_exact
            ),
            "projection_norm_squared_exact": _fraction_to_dict(
                self.projection_norm_squared_exact
            ),
            "projection_norm_squared_upper_bound": (
                self.projection_norm_squared_upper_bound
            ),
            "projection_norm_upper_bound": self.projection_norm_upper_bound,
            "distance_lower_bound": self.distance_lower_bound,
            "distance_lower_bound_squared": self.distance_lower_bound_squared,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self._payload_dict()
        return {**payload, "receipt_digest": self.receipt_digest}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "ActiveManifoldDistanceResult":
        if data.get("schema") != ACTIVE_MANIFOLD_DISTANCE_SCHEMA:
            raise ValueError("active-manifold receipt schema is unsupported.")
        result = cls(
            status=ActiveManifoldDistanceStatus(str(data["status"])),
            reason=str(data["reason"]),
            bindings=ActiveManifoldDistanceBindings.from_dict(dict(data["bindings"])),
            reference_kind=str(data["reference_kind"]),
            reference_bitstring=(
                None
                if data.get("reference_bitstring") is None
                else str(data["reference_bitstring"])
            ),
            qubit_count=int(data["qubit_count"]),
            max_gf2_rank=int(data["max_gf2_rank"]),
            max_orbit_size=int(data["max_orbit_size"]),
            canonical_pauli_words=tuple(data["canonical_pauli_words"]),
            gf2_basis_masks=tuple(
                _parse_hex_integer("gf2_basis_mask", value)
                for value in data["gf2_basis_masks"]
            ),
            affine_orbit_indices=tuple(
                _parse_hex_integer("affine_orbit_index", value)
                for value in data["affine_orbit_indices"]
            ),
            gf2_rank=(None if data.get("gf2_rank") is None else int(data["gf2_rank"])),
            endpoint_l2_error_bound=(
                None
                if data.get("endpoint_l2_error_bound") is None
                else float(data["endpoint_l2_error_bound"])
            ),
            endpoint_norm_squared_exact=_fraction_from_dict(
                "endpoint_norm_squared_exact", data.get("endpoint_norm_squared_exact")
            ),
            projection_norm_squared_exact=_fraction_from_dict(
                "projection_norm_squared_exact",
                data.get("projection_norm_squared_exact"),
            ),
            projection_norm_squared_upper_bound=(
                None
                if data.get("projection_norm_squared_upper_bound") is None
                else float(data["projection_norm_squared_upper_bound"])
            ),
            projection_norm_upper_bound=(
                None
                if data.get("projection_norm_upper_bound") is None
                else float(data["projection_norm_upper_bound"])
            ),
            distance_lower_bound=(
                None
                if data.get("distance_lower_bound") is None
                else float(data["distance_lower_bound"])
            ),
            distance_lower_bound_squared=(
                None
                if data.get("distance_lower_bound_squared") is None
                else float(data["distance_lower_bound_squared"])
            ),
        )
        if data.get("receipt_digest") != result.receipt_digest:
            raise ValueError("active-manifold receipt digest is stale or tampered.")
        if result.to_dict() != data:
            raise ValueError("active-manifold receipt is not canonical.")
        return result


def _canonical_flip_basis(words: Sequence[str]) -> tuple[int, ...]:
    """Return the unique reduced GF(2) row basis, highest pivot first."""

    basis: dict[int, int] = {}
    masks = sorted(
        {
            int("".join("1" if letter in "xy" else "0" for letter in word), 2)
            for word in words
        }
        - {0}
    )
    for original in masks:
        value = original
        for pivot in sorted(basis, reverse=True):
            if value & (1 << pivot):
                value ^= basis[pivot]
        if value == 0:
            continue
        pivot = value.bit_length() - 1
        for other_pivot in tuple(basis):
            if basis[other_pivot] & (1 << pivot):
                basis[other_pivot] ^= value
        basis[pivot] = value
    return tuple(basis[pivot] for pivot in sorted(basis, reverse=True))


def _affine_orbit(reference_index: int, basis: Sequence[int]) -> tuple[int, ...]:
    orbit = [reference_index]
    for mask in basis:
        orbit.extend(value ^ mask for value in tuple(orbit))
    return tuple(sorted(orbit))


def _probability(value: complex) -> Fraction:
    try:
        amplitude = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("endpoint amplitude is invalid.") from exc
    if not math.isfinite(amplitude.real) or not math.isfinite(amplitude.imag):
        raise ValueError("endpoint amplitudes must be finite.")
    real = Fraction.from_float(amplitude.real)
    imaginary = Fraction.from_float(amplitude.imag)
    return real * real + imaginary * imaginary


def _sqrt_bounds(value: Fraction) -> tuple[float, float]:
    """Outward machine-float bounds on the square root of a rational."""

    if value < 0:
        raise ValueError("square-root argument must be nonnegative.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("square-root argument is not finitely representable.")
    root = math.sqrt(converted)
    lower = root
    while Fraction.from_float(lower) ** 2 > value:
        lower = math.nextafter(lower, -math.inf)
    upper = root
    while Fraction.from_float(upper) ** 2 < value:
        upper = math.nextafter(upper, math.inf)
    return max(0.0, lower), upper


def _binding_failure_reason(
    request: ActiveManifoldDistanceRequest,
    canonical_words: tuple[str, ...],
) -> str | None:
    bindings = request.bindings
    token = bindings.eligibility_token
    key = bindings.action_key
    if bindings.working_state_fingerprint != token.working_state_fingerprint:
        return "working_state_fingerprint_eligibility_mismatch"
    if bindings.comparison_epoch != token.comparison_epoch:
        return "comparison_epoch_eligibility_mismatch"
    if bindings.support_provenance_digest != token.support_provenance_digest:
        return "support_provenance_eligibility_mismatch"
    if bindings.trust_provenance_digest != token.trust_provenance_digest:
        return "trust_provenance_eligibility_mismatch"
    if key.record_count != len(token.reachable_record_ids):
        return "action_record_count_eligibility_mismatch"
    if key.record_order > len(token.reachable_record_ids):
        return "action_record_order_eligibility_mismatch"
    if token.reachable_record_ids[key.record_order - 1] != key.record_id:
        return "action_record_id_eligibility_mismatch"
    if bindings.action_receipt_digest != canonical_action_receipt_digest(
        key, token.digest
    ):
        return "canonical_action_receipt_mismatch"
    if bindings.active_support_digest != canonical_active_support_digest(
        tuple(PrimitivePauliSupport(word) for word in canonical_words),
        qubit_count=request.qubit_count,
    ):
        return "active_support_digest_mismatch"
    if bindings.layout_digest != canonical_active_layout_digest(
        qubit_count=request.qubit_count
    ):
        return "layout_digest_mismatch"
    if bindings.execution_mode_digest != canonical_active_execution_mode_digest():
        return "execution_mode_digest_mismatch"
    if bindings.radius_digest != canonical_active_radius_digest(
        action_key=key,
        eligibility_token=token,
    ):
        return "radius_digest_mismatch"
    return None


def _result(
    request: ActiveManifoldDistanceRequest,
    *,
    status: ActiveManifoldDistanceStatus,
    reason: str,
    canonical_words: tuple[str, ...] = (),
    basis: tuple[int, ...] = (),
    orbit: tuple[int, ...] = (),
    endpoint_error: float | None = None,
    endpoint_norm_squared: Fraction | None = None,
    projection_norm_squared: Fraction | None = None,
    projection_squared_upper: float | None = None,
    projection_upper: float | None = None,
    distance: float | None = None,
    distance_squared: float | None = None,
) -> ActiveManifoldDistanceResult:
    return ActiveManifoldDistanceResult(
        status=status,
        reason=reason,
        bindings=request.bindings,
        reference_kind=str(request.reference_kind),
        reference_bitstring=request.reference_bitstring,
        qubit_count=int(request.qubit_count),
        max_gf2_rank=int(request.max_gf2_rank),
        max_orbit_size=int(request.max_orbit_size),
        canonical_pauli_words=canonical_words,
        gf2_basis_masks=basis,
        affine_orbit_indices=orbit,
        gf2_rank=(len(basis) if orbit or canonical_words or basis else None),
        endpoint_l2_error_bound=endpoint_error,
        endpoint_norm_squared_exact=endpoint_norm_squared,
        projection_norm_squared_exact=projection_norm_squared,
        projection_norm_squared_upper_bound=projection_squared_upper,
        projection_norm_upper_bound=projection_upper,
        distance_lower_bound=distance,
        distance_lower_bound_squared=distance_squared,
    )


def certify_active_manifold_distance(
    request: ActiveManifoldDistanceRequest,
) -> ActiveManifoldDistanceResult:
    """Certify a conservative endpoint-to-active-manifold FS lower bound.

    A positive result is possible only when every provenance binding is current,
    the GF(2) orbit fits the declared cap, the approximate endpoint can represent
    a normalized state within its declared L2 error, and the outward projected
    norm remains strictly below one.
    """

    try:
        qubit_count = _positive_integer("qubit_count", request.qubit_count)
        max_rank = _nonnegative_integer("max_gf2_rank", request.max_gf2_rank)
        max_orbit = _positive_integer("max_orbit_size", request.max_orbit_size)
        endpoint_error = _finite_nonnegative(
            "endpoint_l2_error_bound", request.endpoint_l2_error_bound
        )
    except ValueError as exc:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason=f"invalid_resource_or_error_contract:{exc}",
        )

    if str(request.reference_kind) != COMPUTATIONAL_BASIS_REFERENCE_KIND:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="reference_kind_not_supported",
            endpoint_error=endpoint_error,
        )
    bitstring = request.reference_bitstring
    if (
        not isinstance(bitstring, str)
        or len(bitstring) != qubit_count
        or any(bit not in "01" for bit in bitstring)
    ):
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="computational_basis_reference_invalid",
            endpoint_error=endpoint_error,
        )

    try:
        canonical_words = _validated_support_words(
            request.primitive_support,
            qubit_count=qubit_count,
        )
    except ValueError as exc:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason=f"primitive_pauli_support_unresolved:{exc}",
            endpoint_error=endpoint_error,
        )
    binding_failure = _binding_failure_reason(request, canonical_words)
    if binding_failure is not None:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason=binding_failure,
            canonical_words=canonical_words,
            endpoint_error=endpoint_error,
        )

    basis = _canonical_flip_basis(canonical_words)
    rank = len(basis)
    if rank > max_rank:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="gf2_rank_resource_cap_exceeded",
            canonical_words=canonical_words,
            basis=basis,
            endpoint_error=endpoint_error,
        )
    orbit_size = 1 << rank
    if orbit_size > max_orbit:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="affine_orbit_resource_cap_exceeded",
            canonical_words=canonical_words,
            basis=basis,
            endpoint_error=endpoint_error,
        )
    orbit = _affine_orbit(int(bitstring, 2), basis)

    expected_dimension = 1 << qubit_count
    if len(request.endpoint_amplitudes) != expected_dimension:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="endpoint_dimension_layout_mismatch",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
        )
    try:
        probabilities = tuple(_probability(value) for value in request.endpoint_amplitudes)
    except ValueError as exc:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason=f"endpoint_amplitudes_unresolved:{exc}",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
        )
    endpoint_norm_squared = sum(probabilities, Fraction(0, 1))
    projection_norm_squared = sum(
        (probabilities[index] for index in orbit), Fraction(0, 1)
    )
    try:
        norm_lower, norm_upper = _sqrt_bounds(endpoint_norm_squared)
        _, projection_upper_exact_vector = _sqrt_bounds(projection_norm_squared)
    except (ValueError, OverflowError) as exc:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason=f"endpoint_norm_not_finitely_resolved:{exc}",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
            endpoint_norm_squared=endpoint_norm_squared,
            projection_norm_squared=projection_norm_squared,
        )

    if norm_lower > 1.0 + endpoint_error or norm_upper < 1.0 - endpoint_error:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="endpoint_normalization_outside_declared_error",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
            endpoint_norm_squared=endpoint_norm_squared,
            projection_norm_squared=projection_norm_squared,
        )

    # Full affine closure equals the entire Hilbert space.  Its distance lower
    # bound is exactly zero, independent of endpoint uncertainty.
    if rank == qubit_count:
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.CERTIFIED_ZERO,
            reason="active_flip_span_is_full_hilbert_space",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
            endpoint_norm_squared=endpoint_norm_squared,
            projection_norm_squared=projection_norm_squared,
            projection_squared_upper=1.0,
            projection_upper=1.0,
            distance=0.0,
            distance_squared=0.0,
        )

    projection_upper = projection_upper_exact_vector + endpoint_error
    if endpoint_error > 0.0 and math.isfinite(projection_upper):
        projection_upper = math.nextafter(projection_upper, math.inf)
    if not math.isfinite(projection_upper):
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.UNRESOLVED,
            reason="projected_norm_error_enclosure_nonfinite",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
            endpoint_norm_squared=endpoint_norm_squared,
            projection_norm_squared=projection_norm_squared,
        )
    capped_projection = min(1.0, projection_upper)
    projection_squared_upper = capped_projection * capped_projection
    if 0.0 < projection_squared_upper < 1.0:
        projection_squared_upper = min(
            1.0, math.nextafter(projection_squared_upper, math.inf)
        )

    if capped_projection >= 1.0:
        if endpoint_error > 0.0:
            return _result(
                request,
                status=ActiveManifoldDistanceStatus.UNRESOLVED,
                reason="endpoint_error_precludes_positive_distance",
                canonical_words=canonical_words,
                basis=basis,
                orbit=orbit,
                endpoint_error=endpoint_error,
                endpoint_norm_squared=endpoint_norm_squared,
                projection_norm_squared=projection_norm_squared,
                projection_squared_upper=1.0,
                projection_upper=1.0,
            )
        return _result(
            request,
            status=ActiveManifoldDistanceStatus.CERTIFIED_ZERO,
            reason="endpoint_projection_saturates_affine_envelope",
            canonical_words=canonical_words,
            basis=basis,
            orbit=orbit,
            endpoint_error=endpoint_error,
            endpoint_norm_squared=endpoint_norm_squared,
            projection_norm_squared=projection_norm_squared,
            projection_squared_upper=1.0,
            projection_upper=1.0,
            distance=0.0,
            distance_squared=0.0,
        )

    distance = math.acos(capped_projection)
    if distance > 0.0:
        distance = max(0.0, math.nextafter(distance, -math.inf))
    distance_squared = distance * distance
    if distance_squared > 0.0:
        distance_squared = max(
            0.0, math.nextafter(distance_squared, -math.inf)
        )
    return _result(
        request,
        status=ActiveManifoldDistanceStatus.CERTIFIED_POSITIVE,
        reason="affine_pauli_support_projection_separates_endpoint",
        canonical_words=canonical_words,
        basis=basis,
        orbit=orbit,
        endpoint_error=endpoint_error,
        endpoint_norm_squared=endpoint_norm_squared,
        projection_norm_squared=projection_norm_squared,
        projection_squared_upper=projection_squared_upper,
        projection_upper=capped_projection,
        distance=distance,
        distance_squared=distance_squared,
    )


__all__ = [
    "ACTIVE_MANIFOLD_DISTANCE_SCHEMA",
    "ACTIVE_MANIFOLD_EXECUTION_MODE",
    "COMPUTATIONAL_BASIS_REFERENCE_KIND",
    "ActiveManifoldDistanceBindings",
    "ActiveManifoldDistanceRequest",
    "ActiveManifoldDistanceResult",
    "ActiveManifoldDistanceStatus",
    "PrimitivePauliSupport",
    "canonical_active_execution_mode_digest",
    "canonical_active_layout_digest",
    "canonical_active_radius_digest",
    "canonical_active_support_digest",
    "certify_active_manifold_distance",
]
