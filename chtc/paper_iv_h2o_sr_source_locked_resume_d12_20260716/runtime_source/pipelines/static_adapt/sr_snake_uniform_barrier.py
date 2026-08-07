"""Rigorous global spectral fallback for Stage-B uniform path barriers.

For a finite real Pauli Hamiltonian

``H = c_I I + sum_{P != I} h_P P``

every normalized-state expectation lies in
``[c_I - h, c_I + h]``, where ``h = sum_P |h_P|`` after exact
combination of duplicate Pauli words.  Consequently the incumbent-referenced
energy rise along *any* path is at most ``2 h``.  This module computes ``h``
as an exact :class:`fractions.Fraction` from the stored finite binary64
coefficients and converts ``2 h`` to binary64 only by outward rounding.

This is deliberately a conservative global fallback.  It performs no path
sampling and certifies neither a node/Taylor enclosure nor Fubini--Study
exclusion.  It does not enable or execute combined Stage-B mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real

from pipelines.static_adapt.sr_snake_modeled_minimum import (
    ACTION_INDEX_SCHEMA,
    CertificateState,
    EligibilityStateToken,
    EnergyInterval,
    PathActionKey,
    UniformBarrierEvidence,
    canonical_action_receipt_digest,
)
from pipelines.static_adapt.sr_snake_modeled_minimum_runtime import (
    ConfigurationBinding,
    ProviderIdentity,
    ProviderRole,
    SourceBinding,
)


METHOD = "spectral_l1_global_fallback"
NUMERICAL_SCHEMA = "sr_snake_spectral_l1_exact_fraction_binary64_outward_v1"
HAMILTONIAN_SCHEMA = "sr_snake_real_pauli_hamiltonian_binding_v1"
PATH_BINDING_SCHEMA = "sr_snake_canonical_path_barrier_binding_v1"
INCUMBENT_BINDING_SCHEMA = "sr_snake_incumbent_barrier_binding_v1"
CONTEXT_SCHEMA = "sr_snake_spectral_l1_barrier_context_v1"
CERTIFICATE_SCHEMA = "sr_snake_spectral_l1_uniform_barrier_certificate_v1"
UNIFORMITY_SCOPE = "all_normalized_states_hence_any_bound_canonical_path"


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
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string.")
    return value


def _sha256(name: str, value: object) -> str:
    digest = _nonempty(name, value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest.")
    return digest


def _positive_integer(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean.")
    return value


def _finite_stored_float(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real data.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite real data.")
    # Canonicalize signed zero; it has no Hamiltonian semantics.
    return 0.0 if result == 0.0 else result


def _encode_nonnegative_integer(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("serialized integer must be nonnegative.")
    return f"0x{value:x}"


def _decode_nonnegative_integer(name: str, value: object) -> int:
    if not isinstance(value, str) or not value.startswith("0x"):
        raise ValueError(f"{name} must use canonical hexadecimal serialization.")
    digits = value[2:]
    if not digits or any(character not in "0123456789abcdef" for character in digits):
        raise ValueError(f"{name} must use canonical hexadecimal serialization.")
    if len(digits) > 1 and digits[0] == "0":
        raise ValueError(f"{name} has a noncanonical leading zero.")
    result = int(digits, 16)
    if _encode_nonnegative_integer(result) != value:
        raise ValueError(f"{name} is not canonically serialized.")
    return result


def _encode_signed_integer(value: int) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("serialized integer must be integral.")
    if value < 0:
        return f"-0x{-value:x}"
    return _encode_nonnegative_integer(value)


def _decode_signed_integer(name: str, value: object) -> int:
    if not isinstance(value, str):
        raise ValueError(f"{name} must use canonical signed hexadecimal serialization.")
    if value.startswith("-0x"):
        magnitude = _decode_nonnegative_integer(name, "0x" + value[3:])
        if magnitude == 0:
            raise ValueError(f"{name} has a noncanonical negative zero.")
        result = -magnitude
    else:
        result = _decode_nonnegative_integer(name, value)
    if _encode_signed_integer(result) != value:
        raise ValueError(f"{name} is not canonically serialized.")
    return result


def _fraction_to_dict(value: Fraction) -> dict[str, str]:
    return {
        "numerator": _encode_signed_integer(value.numerator),
        "denominator": _encode_nonnegative_integer(value.denominator),
    }


def _fraction_from_dict(name: str, value: object) -> Fraction:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a rational object.")
    if set(value) != {"numerator", "denominator"}:
        raise ValueError(f"{name} has unsupported rational fields.")
    numerator = _decode_signed_integer(f"{name}.numerator", value["numerator"])
    denominator = _decode_nonnegative_integer(
        f"{name}.denominator", value["denominator"]
    )
    if denominator <= 0:
        raise ValueError(f"{name}.denominator must be positive.")
    result = Fraction(numerator, denominator)
    if _fraction_to_dict(result) != value:
        raise ValueError(f"{name} is not reduced canonical rational data.")
    return result


def _outward_upper_float(name: str, value: Fraction) -> float:
    """Return the least adjacent binary64 we can prove is not below ``value``."""

    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")
    try:
        rounded = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} has no finite binary64 upper enclosure.") from exc
    if not math.isfinite(rounded):
        raise ValueError(f"{name} has no finite binary64 upper enclosure.")
    if Fraction.from_float(rounded) < value:
        rounded = math.nextafter(rounded, math.inf)
    if not math.isfinite(rounded) or Fraction.from_float(rounded) < value:
        raise ValueError(f"{name} outward rounding failed.")
    return rounded


@dataclass(frozen=True)
class StoredRealPauliTerm:
    """One stored finite-real Pauli coefficient in the internal convention."""

    pauli_word: str
    coefficient: float

    def __post_init__(self) -> None:
        word = _nonempty("pauli_word", self.pauli_word)
        if any(character not in "exyz" for character in word):
            raise ValueError("pauli_word must use the internal e/x/y/z convention.")
        object.__setattr__(self, "pauli_word", word)
        object.__setattr__(
            self,
            "coefficient",
            _finite_stored_float("coefficient", self.coefficient),
        )

    @property
    def exact_coefficient(self) -> Fraction:
        return Fraction.from_float(self.coefficient)

    def to_dict(self) -> dict[str, object]:
        return {
            "pauli_word": self.pauli_word,
            "coefficient_hex": self.coefficient.hex(),
            "exact_coefficient": _fraction_to_dict(self.exact_coefficient),
        }

    @classmethod
    def from_dict(cls, data: object) -> "StoredRealPauliTerm":
        if not isinstance(data, dict):
            raise ValueError("stored Pauli term must be an object.")
        try:
            coefficient = float.fromhex(str(data["coefficient_hex"]))
        except (KeyError, ValueError) as exc:
            raise ValueError("stored Pauli coefficient is not canonical hex data.") from exc
        term = cls(pauli_word=str(data["pauli_word"]), coefficient=coefficient)
        if term.to_dict() != data:
            raise ValueError("stored Pauli term failed canonical round-trip.")
        return term


@dataclass(frozen=True)
class FrozenRealPauliHamiltonian:
    """Finite ordered compiled payload plus its canonical combined semantics."""

    hamiltonian_id: str
    qubit_count: int
    terms: tuple[StoredRealPauliTerm, ...]
    schema_version: str = field(default=HAMILTONIAN_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "hamiltonian_id",
            _nonempty("hamiltonian_id", self.hamiltonian_id),
        )
        width = _positive_integer("qubit_count", self.qubit_count)
        terms = tuple(self.terms)
        if any(len(term.pauli_word) != width for term in terms):
            raise ValueError("every Pauli word must match qubit_count.")
        object.__setattr__(self, "terms", terms)

    @property
    def combined_terms(self) -> tuple[tuple[str, Fraction], ...]:
        combined: dict[str, Fraction] = {}
        for term in self.terms:
            combined[term.pauli_word] = (
                combined.get(term.pauli_word, Fraction(0, 1))
                + term.exact_coefficient
            )
        return tuple(
            (word, coefficient)
            for word, coefficient in sorted(combined.items())
            if coefficient != 0
        )

    @property
    def identity_word(self) -> str:
        return "e" * self.qubit_count

    @property
    def nonidentity_l1(self) -> Fraction:
        return sum(
            (
                abs(coefficient)
                for word, coefficient in self.combined_terms
                if word != self.identity_word
            ),
            Fraction(0, 1),
        )

    @property
    def exact_barrier_upper_bound(self) -> Fraction:
        return 2 * self.nonidentity_l1

    @property
    def compiled_digest(self) -> str:
        return _digest(
            {
                "schema_version": self.schema_version,
                "qubit_count": self.qubit_count,
                "ordered_stored_terms": [term.to_dict() for term in self.terms],
            }
        )

    @property
    def semantic_digest(self) -> str:
        return _digest(
            {
                "schema_version": self.schema_version,
                "qubit_count": self.qubit_count,
                "canonical_combined_terms": [
                    {
                        "pauli_word": word,
                        "exact_coefficient": _fraction_to_dict(coefficient),
                    }
                    for word, coefficient in self.combined_terms
                ],
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "hamiltonian_id": self.hamiltonian_id,
            "qubit_count": self.qubit_count,
            "terms": [term.to_dict() for term in self.terms],
            "canonical_combined_terms": [
                {
                    "pauli_word": word,
                    "exact_coefficient": _fraction_to_dict(coefficient),
                }
                for word, coefficient in self.combined_terms
            ],
            "nonidentity_l1": _fraction_to_dict(self.nonidentity_l1),
            "exact_barrier_upper_bound": _fraction_to_dict(
                self.exact_barrier_upper_bound
            ),
            "semantic_digest": self.semantic_digest,
            "compiled_digest": self.compiled_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "FrozenRealPauliHamiltonian":
        if not isinstance(data, dict) or data.get("schema_version") != HAMILTONIAN_SCHEMA:
            raise ValueError("unsupported real-Pauli Hamiltonian schema.")
        value = cls(
            hamiltonian_id=str(data["hamiltonian_id"]),
            qubit_count=int(data["qubit_count"]),
            terms=tuple(
                StoredRealPauliTerm.from_dict(item)
                for item in data["terms"]  # type: ignore[union-attr]
            ),
        )
        if value.to_dict() != data:
            raise ValueError("Hamiltonian binding failed canonical digest round-trip.")
        return value


@dataclass(frozen=True)
class BoundCanonicalPath:
    """Frozen identity of the canonical path serviced for one full action."""

    path_id: str
    descriptor_digest: str
    origin_state_fingerprint: str
    eligibility_token_digest: str
    action_receipt_digest: str
    schema_version: str = field(default=PATH_BINDING_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", _nonempty("path_id", self.path_id))
        object.__setattr__(
            self,
            "origin_state_fingerprint",
            _nonempty("origin_state_fingerprint", self.origin_state_fingerprint),
        )
        for name in (
            "descriptor_digest",
            "eligibility_token_digest",
            "action_receipt_digest",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "path_id": self.path_id,
            "descriptor_digest": self.descriptor_digest,
            "origin_state_fingerprint": self.origin_state_fingerprint,
            "eligibility_token_digest": self.eligibility_token_digest,
            "action_receipt_digest": self.action_receipt_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "BoundCanonicalPath":
        if not isinstance(data, dict) or data.get("schema_version") != PATH_BINDING_SCHEMA:
            raise ValueError("unsupported canonical-path binding schema.")
        value = cls(
            path_id=str(data["path_id"]),
            descriptor_digest=str(data["descriptor_digest"]),
            origin_state_fingerprint=str(data["origin_state_fingerprint"]),
            eligibility_token_digest=str(data["eligibility_token_digest"]),
            action_receipt_digest=str(data["action_receipt_digest"]),
        )
        if value.to_dict() != data:
            raise ValueError("canonical-path binding failed canonical round-trip.")
        return value


@dataclass(frozen=True)
class IncumbentBarrierReference:
    """Frozen incumbent snapshot and simultaneous comparison-energy binding."""

    snapshot_digest: str
    state_id: str
    energy: EnergyInterval
    schema_version: str = field(default=INCUMBENT_BINDING_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "snapshot_digest",
            _sha256("snapshot_digest", self.snapshot_digest),
        )
        object.__setattr__(self, "state_id", _nonempty("state_id", self.state_id))
        if self.energy.state_id != self.state_id:
            raise ValueError("incumbent energy must identify the bound incumbent state.")
        if not self.energy.simultaneous:
            raise ValueError("incumbent energy must be simultaneous.")

    @property
    def comparison_epoch(self) -> str:
        return self.energy.comparison_epoch

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_digest": self.snapshot_digest,
            "state_id": self.state_id,
            "comparison_epoch": self.comparison_epoch,
            "energy": self.energy.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> "IncumbentBarrierReference":
        if not isinstance(data, dict) or data.get("schema_version") != INCUMBENT_BINDING_SCHEMA:
            raise ValueError("unsupported incumbent barrier-binding schema.")
        value = cls(
            snapshot_digest=str(data["snapshot_digest"]),
            state_id=str(data["state_id"]),
            energy=EnergyInterval.from_dict(data["energy"]),  # type: ignore[arg-type]
        )
        if value.to_dict() != data:
            raise ValueError("incumbent binding failed canonical round-trip.")
        return value


@dataclass(frozen=True)
class SpectralL1BarrierContext:
    """Authoritative frozen inputs against which one certificate is current."""

    eligibility_token: EligibilityStateToken
    action_key: PathActionKey
    path: BoundCanonicalPath
    incumbent: IncumbentBarrierReference
    hamiltonian: FrozenRealPauliHamiltonian
    source: SourceBinding
    config: ConfigurationBinding
    provider: ProviderIdentity
    numerical_schema: str = field(default=NUMERICAL_SCHEMA, init=False)
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    schema_version: str = field(default=CONTEXT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if self.provider.role is not ProviderRole.UNIFORM_INCUMBENT_BARRIER:
            raise ValueError("provider role must be uniform_incumbent_barrier.")
        token = self.eligibility_token
        key = self.action_key
        if key.record_count != len(token.reachable_record_ids):
            raise ValueError("action record count does not match eligibility token.")
        if token.reachable_record_ids[key.record_order - 1] != key.record_id:
            raise ValueError("action record order does not match eligibility token.")
        expected_receipt = canonical_action_receipt_digest(key, token.digest)
        if self.path.eligibility_token_digest != token.digest:
            raise ValueError("canonical path has a stale eligibility token.")
        if self.path.action_receipt_digest != expected_receipt:
            raise ValueError("canonical path does not bind the full action key.")
        if self.path.origin_state_fingerprint != token.working_state_fingerprint:
            raise ValueError("canonical path origin is stale for the working state.")
        if self.incumbent.comparison_epoch != token.comparison_epoch:
            raise ValueError("incumbent comparison epoch is stale for eligibility.")

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_index_schema": self.action_index_schema,
            "numerical_schema": self.numerical_schema,
            "eligibility_token": self.eligibility_token.to_dict(),
            "eligibility_token_digest": self.eligibility_token.digest,
            "action_key": self.action_key.to_dict(),
            "action_receipt_digest": canonical_action_receipt_digest(
                self.action_key, self.eligibility_token.digest
            ),
            "path": self.path.to_dict(),
            "path_binding_digest": self.path.content_digest,
            "incumbent": self.incumbent.to_dict(),
            "incumbent_binding_digest": self.incumbent.content_digest,
            "hamiltonian": self.hamiltonian.to_dict(),
            "hamiltonian_semantic_digest": self.hamiltonian.semantic_digest,
            "hamiltonian_compiled_digest": self.hamiltonian.compiled_digest,
            "source": self.source.to_dict(),
            "source_binding_digest": self.source.content_digest,
            "config": self.config.to_dict(),
            "config_binding_digest": self.config.content_digest,
            "provider": self.provider.to_dict(),
            "provider_identity_digest": self.provider.content_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "SpectralL1BarrierContext":
        if not isinstance(data, dict) or data.get("schema_version") != CONTEXT_SCHEMA:
            raise ValueError("unsupported spectral-l1 barrier-context schema.")
        if data.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("barrier-context action-index schema drift.")
        if data.get("numerical_schema") != NUMERICAL_SCHEMA:
            raise ValueError("barrier-context numerical schema drift.")
        value = cls(
            eligibility_token=EligibilityStateToken.from_dict(data["eligibility_token"]),  # type: ignore[arg-type]
            action_key=PathActionKey.from_dict(data["action_key"]),  # type: ignore[arg-type]
            path=BoundCanonicalPath.from_dict(data["path"]),
            incumbent=IncumbentBarrierReference.from_dict(data["incumbent"]),
            hamiltonian=FrozenRealPauliHamiltonian.from_dict(data["hamiltonian"]),
            source=SourceBinding.from_dict(data["source"]),
            config=ConfigurationBinding.from_dict(data["config"]),
            provider=ProviderIdentity.from_dict(data["provider"]),
        )
        if value.to_dict() != data:
            raise ValueError("spectral-l1 barrier context failed digest round-trip.")
        return value


@dataclass(frozen=True)
class SpectralL1UniformBarrierCertificate:
    """Passed, content-addressed global spectral barrier certificate."""

    context: SpectralL1BarrierContext
    nonidentity_l1: Fraction
    exact_barrier_upper_bound: Fraction
    barrier_upper_bound: float
    exact_comparison_energy_width: Fraction
    comparison_energy_width: float
    method: str = field(default=METHOD, init=False)
    numerical_schema: str = field(default=NUMERICAL_SCHEMA, init=False)
    uniformity_scope: str = field(default=UNIFORMITY_SCOPE, init=False)
    status: CertificateState = field(default=CertificateState.PASSED, init=False)
    global_uniformity_certified: bool = field(default=True, init=False)
    incumbent_referenced: bool = field(default=True, init=False)
    path_sampling_used: bool = field(default=False, init=False)
    node_taylor_enclosure_certified: bool = field(default=False, init=False)
    fs_exclusion_certified: bool = field(default=False, init=False)
    schema_version: str = field(default=CERTIFICATE_SCHEMA, init=False)

    def __post_init__(self) -> None:
        expected_l1 = self.context.hamiltonian.nonidentity_l1
        expected_barrier = 2 * expected_l1
        incumbent_error = Fraction.from_float(
            self.context.incumbent.energy.energy_error_bound
        )
        expected_width = 2 * incumbent_error
        if self.nonidentity_l1 != expected_l1:
            raise ValueError("certificate l1 norm does not match Hamiltonian semantics.")
        if self.exact_barrier_upper_bound != expected_barrier:
            raise ValueError("certificate exact barrier is not twice the l1 norm.")
        if self.exact_comparison_energy_width != expected_width:
            raise ValueError("certificate comparison width is not incumbent-bound.")
        expected_float = _outward_upper_float(
            "barrier_upper_bound", expected_barrier
        )
        expected_width_float = _outward_upper_float(
            "comparison_energy_width", expected_width
        )
        if self.barrier_upper_bound != expected_float:
            raise ValueError("certificate barrier is not the canonical outward rounding.")
        if self.comparison_energy_width != expected_width_float:
            raise ValueError("certificate comparison width is not outward rounded.")

    @property
    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "numerical_schema": self.numerical_schema,
            "uniformity_scope": self.uniformity_scope,
            "status": self.status.value,
            "global_uniformity_certified": self.global_uniformity_certified,
            "incumbent_referenced": self.incumbent_referenced,
            "path_sampling_used": self.path_sampling_used,
            "node_taylor_enclosure_certified": self.node_taylor_enclosure_certified,
            "fs_exclusion_certified": self.fs_exclusion_certified,
            "context": self.context.to_dict(),
            "context_digest": self.context.content_digest,
            "nonidentity_l1": _fraction_to_dict(self.nonidentity_l1),
            "exact_barrier_upper_bound": _fraction_to_dict(
                self.exact_barrier_upper_bound
            ),
            "barrier_upper_bound": self.barrier_upper_bound,
            "exact_comparison_energy_width": _fraction_to_dict(
                self.exact_comparison_energy_width
            ),
            "comparison_energy_width": self.comparison_energy_width,
        }

    @property
    def content_digest(self) -> str:
        return _digest(self._payload_dict)

    @property
    def witness_id(self) -> str:
        return f"{METHOD}:{self.content_digest}"

    def to_dict(self) -> dict[str, object]:
        return {**self._payload_dict, "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, data: object) -> "SpectralL1UniformBarrierCertificate":
        if not isinstance(data, dict) or data.get("schema_version") != CERTIFICATE_SCHEMA:
            raise ValueError("unsupported spectral-l1 barrier-certificate schema.")
        if data.get("method") != METHOD:
            raise ValueError("spectral-l1 barrier method drift.")
        if data.get("numerical_schema") != NUMERICAL_SCHEMA:
            raise ValueError("spectral-l1 numerical schema drift.")
        if data.get("uniformity_scope") != UNIFORMITY_SCOPE:
            raise ValueError("spectral-l1 uniformity-scope drift.")
        if data.get("status") != CertificateState.PASSED.value:
            raise ValueError("only passed spectral-l1 certificates are serializable.")
        expected_flags = {
            "global_uniformity_certified": True,
            "incumbent_referenced": True,
            "path_sampling_used": False,
            "node_taylor_enclosure_certified": False,
            "fs_exclusion_certified": False,
        }
        for name, expected in expected_flags.items():
            if _strict_bool(name, data.get(name)) is not expected:
                raise ValueError(f"spectral-l1 certificate flag drift: {name}.")
        value = cls(
            context=SpectralL1BarrierContext.from_dict(data["context"]),
            nonidentity_l1=_fraction_from_dict(
                "nonidentity_l1", data["nonidentity_l1"]
            ),
            exact_barrier_upper_bound=_fraction_from_dict(
                "exact_barrier_upper_bound", data["exact_barrier_upper_bound"]
            ),
            barrier_upper_bound=float(data["barrier_upper_bound"]),
            exact_comparison_energy_width=_fraction_from_dict(
                "exact_comparison_energy_width",
                data["exact_comparison_energy_width"],
            ),
            comparison_energy_width=float(data["comparison_energy_width"]),
        )
        if value.to_dict() != data:
            raise ValueError("spectral-l1 certificate failed content-addressed round-trip.")
        return value

    def assert_current(self, current: SpectralL1BarrierContext) -> None:
        """Fail closed if any current action/path/state/code binding has moved."""

        if current.content_digest != self.context.content_digest:
            raise ValueError("spectral-l1 certificate is stale for the current context.")

    def to_uniform_barrier_evidence(
        self,
        *,
        current: SpectralL1BarrierContext,
    ) -> UniformBarrierEvidence:
        """Project a still-current passed certificate into the existing core API."""

        self.assert_current(current)
        # A canonical round-trip is an inexpensive final tamper/schema guard.
        SpectralL1UniformBarrierCertificate.from_dict(self.to_dict())
        return UniformBarrierEvidence(
            witness_id=self.witness_id,
            action_receipt_digest=canonical_action_receipt_digest(
                current.action_key, current.eligibility_token.digest
            ),
            enclosure_id=self.content_digest,
            path_id=current.path.path_id,
            origin_state_id=current.path.origin_state_fingerprint,
            comparison_epoch=current.incumbent.comparison_epoch,
            incumbent_energy=current.incumbent.energy,
            barrier_upper_bound=self.barrier_upper_bound,
            comparison_energy_width=self.comparison_energy_width,
            incumbent_referenced=True,
            status=CertificateState.PASSED,
            simultaneous=True,
        )


def certify_spectral_l1_uniform_barrier(
    context: SpectralL1BarrierContext,
) -> SpectralL1UniformBarrierCertificate:
    """Create a passed certificate or raise without emitting core evidence."""

    exact_barrier = context.hamiltonian.exact_barrier_upper_bound
    exact_width = 2 * Fraction.from_float(
        context.incumbent.energy.energy_error_bound
    )
    return SpectralL1UniformBarrierCertificate(
        context=context,
        nonidentity_l1=context.hamiltonian.nonidentity_l1,
        exact_barrier_upper_bound=exact_barrier,
        barrier_upper_bound=_outward_upper_float(
            "barrier_upper_bound", exact_barrier
        ),
        exact_comparison_energy_width=exact_width,
        comparison_energy_width=_outward_upper_float(
            "comparison_energy_width", exact_width
        ),
    )


__all__ = [
    "BoundCanonicalPath",
    "FrozenRealPauliHamiltonian",
    "IncumbentBarrierReference",
    "METHOD",
    "NUMERICAL_SCHEMA",
    "SpectralL1BarrierContext",
    "SpectralL1UniformBarrierCertificate",
    "StoredRealPauliTerm",
    "certify_spectral_l1_uniform_barrier",
]
