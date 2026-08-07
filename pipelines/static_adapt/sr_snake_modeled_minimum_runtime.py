"""Replay-safe production envelope for SR-SNAKE Stage-B exploration.

This module deliberately does not execute modeled-minimum exploration.  It
defines the immutable state, provider contracts, service plans, and checkpoint
validation required before a production integration is allowed to do so.

Two state objects are always explicit:

``I``
    The externally visible incumbent.
``X``
    The branch-local exploratory working state.

The external view is always derived from ``I``.  Checkpoint loading never
reconstructs a missing ``X`` from ``I`` and a pure-core scheduler checkpoint is
never accepted as a replay-complete production checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from typing import Protocol, runtime_checkable

from pipelines.static_adapt.sr_snake_modeled_minimum import (
    ACTION_INDEX_SCHEMA,
    ConstrainedWorkingState,
    DisposablePowellProbe,
    EligibilityStateToken,
    EnergyInterval,
    EndpointDistanceEvidence,
    FrozenServiceItem,
    ModeledMinimumCheckpoint,
    ModeledMinimumRuntimeState,
    PathActionKey,
    RunEnergyUnit,
    StabilizedTrustPathEvidence,
    UniformBarrierEvidence,
    canonical_action_receipt_digest,
)


_PAYLOAD_SCHEMA = "sr_snake_stage_b_replayable_state_payload_v1"
_SNAPSHOT_SCHEMA = "sr_snake_stage_b_replayable_state_snapshot_v1"
_REPLAY_RECEIPT_SCHEMA = "sr_snake_stage_b_strict_replay_receipt_v1"
_EXECUTION_SCHEMA = "sr_snake_stage_b_single_branch_execution_state_v1"
_CHECKPOINT_SCHEMA = "sr_snake_stage_b_production_checkpoint_v1"
_CHECKPOINT_SCOPE = "stage_b_production_replay_envelope_v1"
_SERVICE_PLAN_SCHEMA = "sr_snake_stage_b_action_service_plan_v1"
_READINESS_SCHEMA = "sr_snake_stage_b_readiness_assessment_v1"
_EXTERNAL_VIEW_SCHEMA = "sr_snake_stage_b_external_incumbent_view_v1"
_FS_DIAMETER = math.pi / 2.0


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
    result = _nonempty(name, value)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256 digest.")
    return result


def _finite(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite real data.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite real data.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative(name: str, value: object) -> float:
    result = _finite(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive(name: str, value: object) -> float:
    result = _finite(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return result


def _integer(name: str, value: object, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum == 1 else "nonnegative"
        raise ValueError(f"{name} must be a {qualifier} integer.")
    return value


def _encode_nonnegative_integer(value: int) -> str:
    """Serialize arbitrary-size counters without decimal digit-limit failures."""

    resolved = _integer("serialized_integer", value)
    return f"0x{resolved:x}"


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


def _decode_positive_integer(name: str, value: object) -> int:
    result = _decode_nonnegative_integer(name, value)
    if result <= 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _strict_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be Boolean.")
    return value


def _require_dict(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object.")
    return value


def _require_list(name: str, value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list.")
    return value


class ProviderRole(str, Enum):
    CANONICAL_PATH = "canonical_path"
    UNIFORM_INCUMBENT_BARRIER = "uniform_incumbent_barrier"
    NONLINEAR_ACTIVE_MANIFOLD_DISTANCE = "nonlinear_active_manifold_distance"
    CONNECTED_COMPONENT_REFIT = "connected_exclusion_component_refit"
    DISPOSABLE_POWELL = "disposable_powell"
    STATE_REPLAY = "state_replay"


_EXECUTION_PROVIDER_ROLES = (
    ProviderRole.CANONICAL_PATH,
    ProviderRole.UNIFORM_INCUMBENT_BARRIER,
    ProviderRole.NONLINEAR_ACTIVE_MANIFOLD_DISTANCE,
    ProviderRole.CONNECTED_COMPONENT_REFIT,
    ProviderRole.DISPOSABLE_POWELL,
)

_BOUND_PROVIDER_ROLES = (*_EXECUTION_PROVIDER_ROLES, ProviderRole.STATE_REPLAY)


class OperatorExecutionMode(str, Enum):
    TERMWISE_PRODUCT = "termwise_product"
    GROUPED_EXACT = "grouped_exact"


class ParameterizationMode(str, Enum):
    LOGICAL_SHARED = "logical_shared"
    PER_PAULI_TERM = "per_pauli_term"


@dataclass(frozen=True)
class ProviderIdentity:
    """Immutable identity for one evidence-producing implementation."""

    role: ProviderRole
    provider_id: str
    version: str
    implementation_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", ProviderRole(self.role))
        object.__setattr__(self, "provider_id", _nonempty("provider_id", self.provider_id))
        object.__setattr__(self, "version", _nonempty("version", self.version))
        object.__setattr__(
            self,
            "implementation_digest",
            _sha256("implementation_digest", self.implementation_digest),
        )

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "provider_id": self.provider_id,
            "version": self.version,
            "implementation_digest": self.implementation_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ProviderIdentity":
        value = _require_dict("provider_identity", data)
        return cls(
            role=ProviderRole(str(value["role"])),
            provider_id=str(value["provider_id"]),
            version=str(value["version"]),
            implementation_digest=str(value["implementation_digest"]),
        )


@dataclass(frozen=True)
class ExactHexFloat:
    """One finite IEEE-754 value serialized without decimal roundoff."""

    hex_value: str

    def __post_init__(self) -> None:
        encoded = _nonempty("hex_value", self.hex_value)
        try:
            value = float.fromhex(encoded)
        except ValueError as exc:
            raise ValueError("hex_value is not a finite Python hex float.") from exc
        if not math.isfinite(value) or value.hex() != encoded:
            raise ValueError("hex_value must be canonical finite Python hex data.")

    @property
    def value(self) -> float:
        return float.fromhex(self.hex_value)

    def to_dict(self) -> dict[str, object]:
        return {"hex_value": self.hex_value}

    @classmethod
    def from_float(cls, value: float) -> "ExactHexFloat":
        return cls(_finite("value", value).hex())

    @classmethod
    def from_dict(cls, data: object) -> "ExactHexFloat":
        value = _require_dict("exact_hex_float", data)
        return cls(hex_value=str(value["hex_value"]))


@dataclass(frozen=True)
class ExactComplexCoefficient:
    real: ExactHexFloat
    imag: ExactHexFloat

    def __post_init__(self) -> None:
        if self.real.value == 0.0 and self.imag.value == 0.0:
            raise ValueError("canonical operator terms must omit zero coefficients.")

    def to_dict(self) -> dict[str, object]:
        return {"real": self.real.to_dict(), "imag": self.imag.to_dict()}

    @classmethod
    def from_complex(cls, value: complex) -> "ExactComplexCoefficient":
        return cls(
            real=ExactHexFloat.from_float(float(value.real)),
            imag=ExactHexFloat.from_float(float(value.imag)),
        )

    @classmethod
    def from_dict(cls, data: object) -> "ExactComplexCoefficient":
        value = _require_dict("exact_complex_coefficient", data)
        return cls(
            real=ExactHexFloat.from_dict(value["real"]),
            imag=ExactHexFloat.from_dict(value["imag"]),
        )


@dataclass(frozen=True)
class ExactOperatorTerm:
    """One exact internal-convention Pauli term."""

    term_id: str
    pauli_word: str
    coefficient: ExactComplexCoefficient

    def __post_init__(self) -> None:
        object.__setattr__(self, "term_id", _nonempty("term_id", self.term_id))
        word = _nonempty("pauli_word", self.pauli_word)
        if any(character not in "exyz" for character in word):
            raise ValueError("pauli_word must use the internal e/x/y/z convention.")

    @property
    def qubit_count(self) -> int:
        return len(self.pauli_word)

    def to_dict(self) -> dict[str, object]:
        return {
            "term_id": self.term_id,
            "pauli_word": self.pauli_word,
            "coefficient": self.coefficient.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> "ExactOperatorTerm":
        value = _require_dict("operator_term", data)
        return cls(
            term_id=str(value["term_id"]),
            pauli_word=str(value["pauli_word"]),
            coefficient=ExactComplexCoefficient.from_dict(value["coefficient"]),
        )


@dataclass(frozen=True)
class ExactOperatorPayload:
    """One semantic operator with its exact, ordered Pauli-term payload."""

    operator_id: str
    semantic_operator_id: str
    execution_mode: OperatorExecutionMode
    terms: tuple[ExactOperatorTerm, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "operator_id", _nonempty("operator_id", self.operator_id))
        object.__setattr__(
            self,
            "semantic_operator_id",
            _nonempty("semantic_operator_id", self.semantic_operator_id),
        )
        object.__setattr__(
            self,
            "execution_mode",
            OperatorExecutionMode(self.execution_mode),
        )
        terms = tuple(self.terms)
        if not terms:
            raise ValueError("operator payload must contain at least one exact term.")
        if len({term.term_id for term in terms}) != len(terms):
            raise ValueError("operator term identifiers must be unique.")
        if len({term.qubit_count for term in terms}) != 1:
            raise ValueError("all terms in one operator must have the same width.")
        if any(term.coefficient.imag.value != 0.0 for term in terms):
            raise ValueError("ansatz generators require real Pauli coefficients.")
        object.__setattr__(self, "terms", terms)

    @property
    def qubit_count(self) -> int:
        return self.terms[0].qubit_count

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "operator_id": self.operator_id,
            "semantic_operator_id": self.semantic_operator_id,
            "execution_mode": self.execution_mode.value,
            "terms": [term.to_dict() for term in self.terms],
        }

    @classmethod
    def from_dict(cls, data: object) -> "ExactOperatorPayload":
        value = _require_dict("operator_payload", data)
        return cls(
            operator_id=str(value["operator_id"]),
            semantic_operator_id=str(value["semantic_operator_id"]),
            execution_mode=OperatorExecutionMode(str(value["execution_mode"])),
            terms=tuple(
                ExactOperatorTerm.from_dict(item)
                for item in _require_list("operator_terms", value["terms"])
            ),
        )


class ThetaSpace(str, Enum):
    RUNTIME = "runtime"
    LOGICAL = "logical"


@dataclass(frozen=True)
class ExactThetaVector:
    space: ThetaSpace
    parameter_ids: tuple[str, ...]
    values: tuple[ExactHexFloat, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "space", ThetaSpace(self.space))
        identifiers = tuple(_nonempty("parameter_id", value) for value in self.parameter_ids)
        values = tuple(self.values)
        if len(identifiers) != len(values):
            raise ValueError("theta parameter identifiers and values must align.")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("theta parameter identifiers must be unique.")
        object.__setattr__(self, "parameter_ids", identifiers)
        object.__setattr__(self, "values", values)

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "space": self.space.value,
            "parameter_ids": list(self.parameter_ids),
            "values": [value.to_dict() for value in self.values],
        }

    @classmethod
    def from_floats(
        cls,
        *,
        space: ThetaSpace,
        parameter_ids: tuple[str, ...],
        values: tuple[float, ...],
    ) -> "ExactThetaVector":
        return cls(
            space=space,
            parameter_ids=parameter_ids,
            values=tuple(ExactHexFloat.from_float(value) for value in values),
        )

    @classmethod
    def from_dict(cls, data: object) -> "ExactThetaVector":
        value = _require_dict("theta_vector", data)
        return cls(
            space=ThetaSpace(str(value["space"])),
            parameter_ids=tuple(str(item) for item in _require_list("parameter_ids", value["parameter_ids"])),
            values=tuple(
                ExactHexFloat.from_dict(item)
                for item in _require_list("theta_values", value["values"])
            ),
        )


@dataclass(frozen=True)
class ParameterLayout:
    """Exact runtime/logical parameter and operator layout."""

    runtime_parameter_ids: tuple[str, ...]
    logical_parameter_ids: tuple[str, ...]
    runtime_to_logical: tuple[int, ...]
    operator_to_logical: tuple[int, ...]

    def __post_init__(self) -> None:
        runtime = tuple(_nonempty("runtime_parameter_id", item) for item in self.runtime_parameter_ids)
        logical = tuple(_nonempty("logical_parameter_id", item) for item in self.logical_parameter_ids)
        if len(set(runtime)) != len(runtime) or len(set(logical)) != len(logical):
            raise ValueError("parameter layout identifiers must be unique within each space.")
        runtime_map = tuple(_integer("runtime_to_logical", item) for item in self.runtime_to_logical)
        operator_map = tuple(_integer("operator_to_logical", item) for item in self.operator_to_logical)
        if len(runtime_map) != len(runtime):
            raise ValueError("runtime_to_logical must cover every runtime parameter.")
        if (runtime_map or operator_map) and not logical:
            raise ValueError("mapped parameters require a nonempty logical layout.")
        if any(item >= len(logical) for item in (*runtime_map, *operator_map)):
            raise ValueError("parameter layout index exceeds logical dimension.")
        object.__setattr__(self, "runtime_parameter_ids", runtime)
        object.__setattr__(self, "logical_parameter_ids", logical)
        object.__setattr__(self, "runtime_to_logical", runtime_map)
        object.__setattr__(self, "operator_to_logical", operator_map)

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "runtime_parameter_ids": list(self.runtime_parameter_ids),
            "logical_parameter_ids": list(self.logical_parameter_ids),
            "runtime_to_logical": list(self.runtime_to_logical),
            "operator_to_logical": list(self.operator_to_logical),
        }

    @classmethod
    def from_dict(cls, data: object) -> "ParameterLayout":
        value = _require_dict("parameter_layout", data)
        return cls(
            runtime_parameter_ids=tuple(str(item) for item in _require_list("runtime_parameter_ids", value["runtime_parameter_ids"])),
            logical_parameter_ids=tuple(str(item) for item in _require_list("logical_parameter_ids", value["logical_parameter_ids"])),
            runtime_to_logical=tuple(_integer("runtime_to_logical", item) for item in _require_list("runtime_to_logical", value["runtime_to_logical"])),
            operator_to_logical=tuple(_integer("operator_to_logical", item) for item in _require_list("operator_to_logical", value["operator_to_logical"])),
        )


@dataclass(frozen=True)
class PreparedStateManifest:
    state_fingerprint: str
    prepared_state_digest: str
    statevector_digest: str
    preparation_manifest_digest: str
    qubit_count: int
    normalized: bool
    finite: bool
    norm_error_bound: float
    phase_convention: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_fingerprint", _nonempty("state_fingerprint", self.state_fingerprint))
        for name in (
            "prepared_state_digest",
            "statevector_digest",
            "preparation_manifest_digest",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        _integer("qubit_count", self.qubit_count, minimum=1)
        _strict_bool("normalized", self.normalized)
        _strict_bool("finite", self.finite)
        _nonnegative("norm_error_bound", self.norm_error_bound)
        object.__setattr__(self, "phase_convention", _nonempty("phase_convention", self.phase_convention))

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "state_fingerprint": self.state_fingerprint,
            "prepared_state_digest": self.prepared_state_digest,
            "statevector_digest": self.statevector_digest,
            "preparation_manifest_digest": self.preparation_manifest_digest,
            "qubit_count": self.qubit_count,
            "normalized": self.normalized,
            "finite": self.finite,
            "norm_error_bound": self.norm_error_bound,
            "phase_convention": self.phase_convention,
        }

    @classmethod
    def from_dict(cls, data: object) -> "PreparedStateManifest":
        value = _require_dict("prepared_state_manifest", data)
        return cls(
            state_fingerprint=str(value["state_fingerprint"]),
            prepared_state_digest=str(value["prepared_state_digest"]),
            statevector_digest=str(value["statevector_digest"]),
            preparation_manifest_digest=str(value["preparation_manifest_digest"]),
            qubit_count=_integer("qubit_count", value["qubit_count"], minimum=1),
            normalized=_strict_bool("normalized", value["normalized"]),
            finite=_strict_bool("finite", value["finite"]),
            norm_error_bound=float(value["norm_error_bound"]),
            phase_convention=str(value["phase_convention"]),
        )


@dataclass(frozen=True)
class ReplayableStatePayload:
    """Exact ansatz/prepared-state payload, excluding replay testimony."""

    operators: tuple[ExactOperatorPayload, ...]
    parameterization_mode: ParameterizationMode
    runtime_theta: ExactThetaVector
    logical_theta: ExactThetaVector
    layout: ParameterLayout
    prepared_state: PreparedStateManifest
    schema_version: str = field(default=_PAYLOAD_SCHEMA, init=False)

    def __post_init__(self) -> None:
        operators = tuple(self.operators)
        object.__setattr__(
            self,
            "parameterization_mode",
            ParameterizationMode(self.parameterization_mode),
        )
        if len({operator.operator_id for operator in operators}) != len(operators):
            raise ValueError("operator identifiers must be unique and ordered.")
        if any(operator.qubit_count != self.prepared_state.qubit_count for operator in operators):
            raise ValueError("operator width differs from the prepared-state manifest.")
        if self.runtime_theta.space is not ThetaSpace.RUNTIME:
            raise ValueError("runtime_theta must use runtime space.")
        if self.logical_theta.space is not ThetaSpace.LOGICAL:
            raise ValueError("logical_theta must use logical space.")
        if self.runtime_theta.parameter_ids != self.layout.runtime_parameter_ids:
            raise ValueError("runtime theta order differs from parameter layout.")
        if self.logical_theta.parameter_ids != self.layout.logical_parameter_ids:
            raise ValueError("logical theta order differs from parameter layout.")
        if len(self.layout.operator_to_logical) != len(operators):
            raise ValueError("operator_to_logical must cover the exact operator sequence.")
        if (
            self.parameterization_mode is ParameterizationMode.PER_PAULI_TERM
            and any(
                operator.execution_mode is OperatorExecutionMode.GROUPED_EXACT
                for operator in operators
            )
        ):
            raise ValueError(
                "grouped_exact execution is incompatible with per_pauli_term parameterization."
            )
        runtime_values = tuple(value.value for value in self.runtime_theta.values)
        logical_values = tuple(value.value for value in self.logical_theta.values)
        for logical_index, logical_value in enumerate(logical_values):
            block = tuple(
                runtime_values[runtime_index]
                for runtime_index, mapped_index in enumerate(
                    self.layout.runtime_to_logical
                )
                if mapped_index == logical_index
            )
            if not block:
                raise ValueError("every logical coordinate requires runtime support.")
            projected = math.fsum(block) / len(block)
            if logical_value != projected:
                raise ValueError(
                    "logical theta differs from the runtime block-mean projection."
                )
        object.__setattr__(self, "operators", operators)

    @property
    def operator_sequence_digest(self) -> str:
        return _digest([operator.to_dict() for operator in self.operators])

    @property
    def content_digest(self) -> str:
        return _digest(self._payload_dict())

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operators": [operator.to_dict() for operator in self.operators],
            "parameterization_mode": self.parameterization_mode.value,
            "runtime_theta": self.runtime_theta.to_dict(),
            "logical_theta": self.logical_theta.to_dict(),
            "layout": self.layout.to_dict(),
            "prepared_state": self.prepared_state.to_dict(),
            "operator_sequence_digest": self.operator_sequence_digest,
            "runtime_theta_digest": self.runtime_theta.content_digest,
            "logical_theta_digest": self.logical_theta.content_digest,
            "layout_digest": self.layout.content_digest,
            "prepared_state_manifest_digest": self.prepared_state.content_digest,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload_dict(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, data: object) -> "ReplayableStatePayload":
        value = _require_dict("replayable_state_payload", data)
        if value.get("schema_version") != _PAYLOAD_SCHEMA:
            raise ValueError("unsupported replayable state payload schema.")
        payload = cls(
            operators=tuple(
                ExactOperatorPayload.from_dict(item)
                for item in _require_list("operators", value["operators"])
            ),
            parameterization_mode=ParameterizationMode(
                str(value["parameterization_mode"])
            ),
            runtime_theta=ExactThetaVector.from_dict(value["runtime_theta"]),
            logical_theta=ExactThetaVector.from_dict(value["logical_theta"]),
            layout=ParameterLayout.from_dict(value["layout"]),
            prepared_state=PreparedStateManifest.from_dict(value["prepared_state"]),
        )
        if payload.to_dict() != value:
            raise ValueError("replayable state payload failed canonical digest round-trip.")
        return payload


@dataclass(frozen=True)
class StateLineageEvent:
    event_index: int
    event_kind: str
    state_fingerprint: str
    parent_state_fingerprint: str | None
    details_digest: str
    action_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        _integer("event_index", self.event_index)
        object.__setattr__(self, "event_kind", _nonempty("event_kind", self.event_kind))
        object.__setattr__(self, "state_fingerprint", _nonempty("state_fingerprint", self.state_fingerprint))
        if self.parent_state_fingerprint is not None:
            object.__setattr__(self, "parent_state_fingerprint", _nonempty("parent_state_fingerprint", self.parent_state_fingerprint))
        object.__setattr__(self, "details_digest", _sha256("details_digest", self.details_digest))
        if self.action_receipt_digest is not None:
            object.__setattr__(self, "action_receipt_digest", _sha256("action_receipt_digest", self.action_receipt_digest))

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "state_fingerprint": self.state_fingerprint,
            "parent_state_fingerprint": self.parent_state_fingerprint,
            "details_digest": self.details_digest,
            "action_receipt_digest": self.action_receipt_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "StateLineageEvent":
        value = _require_dict("state_lineage_event", data)
        parent = value["parent_state_fingerprint"]
        action = value["action_receipt_digest"]
        return cls(
            event_index=_integer("event_index", value["event_index"]),
            event_kind=str(value["event_kind"]),
            state_fingerprint=str(value["state_fingerprint"]),
            parent_state_fingerprint=None if parent is None else str(parent),
            details_digest=str(value["details_digest"]),
            action_receipt_digest=None if action is None else str(action),
        )


@dataclass(frozen=True)
class StrictReplayReceipt:
    """Independent replay testimony bound to every exact state payload field."""

    receipt_id: str
    replay_provider: ProviderIdentity
    source_digest: str
    config_digest: str
    payload_digest: str
    energy_interval_digest: str
    replayed_state_fingerprint: str
    prepared_state_digest: str
    operator_sequence_digest: str
    runtime_theta_digest: str
    logical_theta_digest: str
    layout_digest: str
    verification_result_digest: str
    finite: bool
    normalized: bool
    phase_aligned: bool
    projective_distance: float
    state_consistency_tolerance: float
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    schema_version: str = field(default=_REPLAY_RECEIPT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "receipt_id", _nonempty("receipt_id", self.receipt_id))
        if self.replay_provider.role is not ProviderRole.STATE_REPLAY:
            raise ValueError("strict replay receipt requires a state-replay provider.")
        for name in (
            "source_digest",
            "config_digest",
            "payload_digest",
            "energy_interval_digest",
            "prepared_state_digest",
            "operator_sequence_digest",
            "runtime_theta_digest",
            "logical_theta_digest",
            "layout_digest",
            "verification_result_digest",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(self, "replayed_state_fingerprint", _nonempty("replayed_state_fingerprint", self.replayed_state_fingerprint))
        for name in ("finite", "normalized", "phase_aligned"):
            _strict_bool(name, getattr(self, name))
        distance = _nonnegative("projective_distance", self.projective_distance)
        tolerance = _nonnegative("state_consistency_tolerance", self.state_consistency_tolerance)
        if distance > _FS_DIAMETER or tolerance > _FS_DIAMETER:
            raise ValueError("replay distance and tolerance cannot exceed the FS diameter.")
        if distance > tolerance:
            raise ValueError("replay projective distance exceeds its declared tolerance.")

    @property
    def strict_passed(self) -> bool:
        return self.finite and self.normalized and self.phase_aligned

    def assert_matches(self, payload: ReplayableStatePayload, energy: EnergyInterval) -> None:
        expected = {
            "payload_digest": payload.content_digest,
            "energy_interval_digest": _digest(energy.to_dict()),
            "replayed_state_fingerprint": payload.prepared_state.state_fingerprint,
            "prepared_state_digest": payload.prepared_state.prepared_state_digest,
            "operator_sequence_digest": payload.operator_sequence_digest,
            "runtime_theta_digest": payload.runtime_theta.content_digest,
            "logical_theta_digest": payload.logical_theta.content_digest,
            "layout_digest": payload.layout.content_digest,
        }
        mismatches = [name for name, expected_value in expected.items() if getattr(self, name) != expected_value]
        if mismatches:
            raise ValueError("strict replay receipt payload mismatch: " + ",".join(mismatches))

    @classmethod
    def record_verified_result(
        cls,
        *,
        receipt_id: str,
        replay_provider: ProviderIdentity,
        source_digest: str,
        config_digest: str,
        payload: ReplayableStatePayload,
        energy: EnergyInterval,
        projective_distance: float,
        state_consistency_tolerance: float,
        verification_result_digest: str,
        finite: bool,
        normalized: bool,
        phase_aligned: bool,
    ) -> "StrictReplayReceipt":
        """Record a replay provider result; this method does not execute replay."""

        return cls(
            receipt_id=receipt_id,
            replay_provider=replay_provider,
            source_digest=source_digest,
            config_digest=config_digest,
            payload_digest=payload.content_digest,
            energy_interval_digest=_digest(energy.to_dict()),
            replayed_state_fingerprint=payload.prepared_state.state_fingerprint,
            prepared_state_digest=payload.prepared_state.prepared_state_digest,
            operator_sequence_digest=payload.operator_sequence_digest,
            runtime_theta_digest=payload.runtime_theta.content_digest,
            logical_theta_digest=payload.logical_theta.content_digest,
            layout_digest=payload.layout.content_digest,
            verification_result_digest=verification_result_digest,
            finite=finite,
            normalized=normalized,
            phase_aligned=phase_aligned,
            projective_distance=projective_distance,
            state_consistency_tolerance=state_consistency_tolerance,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_index_schema": self.action_index_schema,
            "receipt_id": self.receipt_id,
            "replay_provider": self.replay_provider.to_dict(),
            "source_digest": self.source_digest,
            "config_digest": self.config_digest,
            "payload_digest": self.payload_digest,
            "energy_interval_digest": self.energy_interval_digest,
            "replayed_state_fingerprint": self.replayed_state_fingerprint,
            "prepared_state_digest": self.prepared_state_digest,
            "operator_sequence_digest": self.operator_sequence_digest,
            "runtime_theta_digest": self.runtime_theta_digest,
            "logical_theta_digest": self.logical_theta_digest,
            "layout_digest": self.layout_digest,
            "verification_result_digest": self.verification_result_digest,
            "finite": self.finite,
            "normalized": self.normalized,
            "phase_aligned": self.phase_aligned,
            "projective_distance": self.projective_distance,
            "state_consistency_tolerance": self.state_consistency_tolerance,
        }

    @classmethod
    def from_dict(cls, data: object) -> "StrictReplayReceipt":
        value = _require_dict("strict_replay_receipt", data)
        if value.get("schema_version") != _REPLAY_RECEIPT_SCHEMA:
            raise ValueError("unsupported strict replay receipt schema.")
        if value.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("strict replay receipt action-index schema drift.")
        return cls(
            receipt_id=str(value["receipt_id"]),
            replay_provider=ProviderIdentity.from_dict(value["replay_provider"]),
            source_digest=str(value["source_digest"]),
            config_digest=str(value["config_digest"]),
            payload_digest=str(value["payload_digest"]),
            energy_interval_digest=str(value["energy_interval_digest"]),
            replayed_state_fingerprint=str(value["replayed_state_fingerprint"]),
            prepared_state_digest=str(value["prepared_state_digest"]),
            operator_sequence_digest=str(value["operator_sequence_digest"]),
            runtime_theta_digest=str(value["runtime_theta_digest"]),
            logical_theta_digest=str(value["logical_theta_digest"]),
            layout_digest=str(value["layout_digest"]),
            verification_result_digest=str(value["verification_result_digest"]),
            finite=_strict_bool("finite", value["finite"]),
            normalized=_strict_bool("normalized", value["normalized"]),
            phase_aligned=_strict_bool("phase_aligned", value["phase_aligned"]),
            projective_distance=float(value["projective_distance"]),
            state_consistency_tolerance=float(value["state_consistency_tolerance"]),
        )


@dataclass(frozen=True)
class ReplayableStateSnapshot:
    payload: ReplayableStatePayload
    energy: EnergyInterval
    lineage: tuple[StateLineageEvent, ...]
    replay_receipt: StrictReplayReceipt | None
    schema_version: str = field(default=_SNAPSHOT_SCHEMA, init=False)

    def __post_init__(self) -> None:
        fingerprint = self.payload.prepared_state.state_fingerprint
        if self.energy.state_id != fingerprint:
            raise ValueError("energy interval state_id differs from prepared-state fingerprint.")
        lineage = tuple(self.lineage)
        if not lineage:
            raise ValueError("replayable state requires explicit lineage history.")
        if tuple(event.event_index for event in lineage) != tuple(range(len(lineage))):
            raise ValueError("state lineage event indices must be contiguous from zero.")
        if lineage[-1].state_fingerprint != fingerprint:
            raise ValueError("state lineage does not terminate at the snapshot state.")
        object.__setattr__(self, "lineage", lineage)
        if self.replay_receipt is not None:
            self.replay_receipt.assert_matches(self.payload, self.energy)

    @property
    def state_fingerprint(self) -> str:
        return self.payload.prepared_state.state_fingerprint

    @property
    def strict_replay_complete(self) -> bool:
        return (
            self.replay_receipt is not None
            and self.replay_receipt.strict_passed
            and self.payload.prepared_state.finite
            and self.payload.prepared_state.normalized
        )

    @property
    def content_digest(self) -> str:
        return _digest(self._payload_dict())

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "payload": self.payload.to_dict(),
            "energy": self.energy.to_dict(),
            "lineage": [event.to_dict() for event in self.lineage],
            "replay_receipt": None if self.replay_receipt is None else self.replay_receipt.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload_dict(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, data: object) -> "ReplayableStateSnapshot":
        value = _require_dict("replayable_state_snapshot", data)
        if value.get("schema_version") != _SNAPSHOT_SCHEMA:
            raise ValueError("unsupported replayable state snapshot schema.")
        receipt_data = value["replay_receipt"]
        snapshot = cls(
            payload=ReplayableStatePayload.from_dict(value["payload"]),
            energy=EnergyInterval.from_dict(_require_dict("energy", value["energy"])),
            lineage=tuple(
                StateLineageEvent.from_dict(item)
                for item in _require_list("lineage", value["lineage"])
            ),
            replay_receipt=None if receipt_data is None else StrictReplayReceipt.from_dict(receipt_data),
        )
        if snapshot.to_dict() != value:
            raise ValueError("replayable state snapshot failed canonical digest round-trip.")
        return snapshot


@dataclass(frozen=True)
class SourceBinding:
    repository_id: str
    revision: str
    source_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository_id", _nonempty("repository_id", self.repository_id))
        object.__setattr__(self, "revision", _nonempty("revision", self.revision))
        object.__setattr__(self, "source_digest", _sha256("source_digest", self.source_digest))

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "repository_id": self.repository_id,
            "revision": self.revision,
            "source_digest": self.source_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "SourceBinding":
        value = _require_dict("source_binding", data)
        return cls(
            repository_id=str(value["repository_id"]),
            revision=str(value["revision"]),
            source_digest=str(value["source_digest"]),
        )


@dataclass(frozen=True)
class ConfigurationBinding:
    config_id: str
    route_family: str
    route_profile: str
    config_digest: str
    state_replay_tolerance: float
    state_norm_error_tolerance: float

    def __post_init__(self) -> None:
        for name in ("config_id", "route_family", "route_profile"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(self, "config_digest", _sha256("config_digest", self.config_digest))
        replay_tolerance = _nonnegative(
            "state_replay_tolerance", self.state_replay_tolerance
        )
        norm_tolerance = _nonnegative(
            "state_norm_error_tolerance", self.state_norm_error_tolerance
        )
        if replay_tolerance > _FS_DIAMETER:
            raise ValueError("state replay tolerance cannot exceed the FS diameter.")
        if norm_tolerance > 1.0:
            raise ValueError("state norm-error tolerance cannot exceed one.")

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "config_id": self.config_id,
            "route_family": self.route_family,
            "route_profile": self.route_profile,
            "config_digest": self.config_digest,
            "state_replay_tolerance": self.state_replay_tolerance,
            "state_norm_error_tolerance": self.state_norm_error_tolerance,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ConfigurationBinding":
        value = _require_dict("configuration_binding", data)
        return cls(
            config_id=str(value["config_id"]),
            route_family=str(value["route_family"]),
            route_profile=str(value["route_profile"]),
            config_digest=str(value["config_digest"]),
            state_replay_tolerance=float(value["state_replay_tolerance"]),
            state_norm_error_tolerance=float(value["state_norm_error_tolerance"]),
        )


@dataclass(frozen=True)
class StageBProviderBindings:
    canonical_path: ProviderIdentity | None = None
    uniform_incumbent_barrier: ProviderIdentity | None = None
    nonlinear_active_manifold_distance: ProviderIdentity | None = None
    connected_component_refit: ProviderIdentity | None = None
    disposable_powell: ProviderIdentity | None = None
    state_replay: ProviderIdentity | None = None

    def __post_init__(self) -> None:
        expected = {
            "canonical_path": ProviderRole.CANONICAL_PATH,
            "uniform_incumbent_barrier": ProviderRole.UNIFORM_INCUMBENT_BARRIER,
            "nonlinear_active_manifold_distance": ProviderRole.NONLINEAR_ACTIVE_MANIFOLD_DISTANCE,
            "connected_component_refit": ProviderRole.CONNECTED_COMPONENT_REFIT,
            "disposable_powell": ProviderRole.DISPOSABLE_POWELL,
            "state_replay": ProviderRole.STATE_REPLAY,
        }
        for name, role in expected.items():
            provider = getattr(self, name)
            if provider is not None and provider.role is not role:
                raise ValueError(f"{name} provider role mismatch.")

    @property
    def missing_roles(self) -> tuple[ProviderRole, ...]:
        by_role = self.by_role
        return tuple(role for role in _BOUND_PROVIDER_ROLES if role not in by_role)

    @property
    def complete(self) -> bool:
        return not self.missing_roles

    @property
    def by_role(self) -> dict[ProviderRole, ProviderIdentity]:
        values = (
            self.canonical_path,
            self.uniform_incumbent_barrier,
            self.nonlinear_active_manifold_distance,
            self.connected_component_refit,
            self.disposable_powell,
            self.state_replay,
        )
        return {provider.role: provider for provider in values if provider is not None}

    @property
    def content_digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_path": None if self.canonical_path is None else self.canonical_path.to_dict(),
            "uniform_incumbent_barrier": None if self.uniform_incumbent_barrier is None else self.uniform_incumbent_barrier.to_dict(),
            "nonlinear_active_manifold_distance": None if self.nonlinear_active_manifold_distance is None else self.nonlinear_active_manifold_distance.to_dict(),
            "connected_component_refit": None if self.connected_component_refit is None else self.connected_component_refit.to_dict(),
            "disposable_powell": None if self.disposable_powell is None else self.disposable_powell.to_dict(),
            "state_replay": None if self.state_replay is None else self.state_replay.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> "StageBProviderBindings":
        value = _require_dict("provider_bindings", data)

        def optional(name: str) -> ProviderIdentity | None:
            item = value[name]
            return None if item is None else ProviderIdentity.from_dict(item)

        return cls(
            canonical_path=optional("canonical_path"),
            uniform_incumbent_barrier=optional("uniform_incumbent_barrier"),
            nonlinear_active_manifold_distance=optional("nonlinear_active_manifold_distance"),
            connected_component_refit=optional("connected_component_refit"),
            disposable_powell=optional("disposable_powell"),
            state_replay=optional("state_replay"),
        )


@dataclass(frozen=True)
class StageBActionServicePlan:
    """Provider-call plan only; it contains no fabricated certificate data."""

    plan_id: str
    action_key: PathActionKey
    action_receipt_digest: str
    eligibility_token_digest: str
    energy_unit_digest: str
    incumbent_snapshot_digest: str
    working_snapshot_digest: str
    source_binding_digest: str
    config_binding_digest: str
    provider_bindings_digest: str
    service_epoch: str
    service_ordinal: int
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    schema_version: str = field(default=_SERVICE_PLAN_SCHEMA, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _nonempty("plan_id", self.plan_id))
        for name in (
            "action_receipt_digest",
            "eligibility_token_digest",
            "energy_unit_digest",
            "incumbent_snapshot_digest",
            "working_snapshot_digest",
            "source_binding_digest",
            "config_binding_digest",
            "provider_bindings_digest",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(self, "service_epoch", _nonempty("service_epoch", self.service_epoch))
        _integer("service_ordinal", self.service_ordinal)
        expected = canonical_action_receipt_digest(self.action_key, self.eligibility_token_digest)
        if self.action_receipt_digest != expected:
            raise ValueError("service plan action receipt does not bind its full action key.")

    @classmethod
    def create(
        cls,
        *,
        plan_id: str,
        action_key: PathActionKey,
        eligibility_token: EligibilityStateToken,
        energy_unit: RunEnergyUnit,
        incumbent: ReplayableStateSnapshot,
        working: ReplayableStateSnapshot,
        source: SourceBinding,
        config: ConfigurationBinding,
        providers: StageBProviderBindings,
        service_epoch: str,
        service_ordinal: int,
    ) -> "StageBActionServicePlan":
        token_digest = eligibility_token.digest
        return cls(
            plan_id=plan_id,
            action_key=action_key,
            action_receipt_digest=canonical_action_receipt_digest(action_key, token_digest),
            eligibility_token_digest=token_digest,
            energy_unit_digest=energy_unit.digest,
            incumbent_snapshot_digest=incumbent.content_digest,
            working_snapshot_digest=working.content_digest,
            source_binding_digest=source.content_digest,
            config_binding_digest=config.content_digest,
            provider_bindings_digest=providers.content_digest,
            service_epoch=service_epoch,
            service_ordinal=service_ordinal,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "action_index_schema": self.action_index_schema,
            "plan_id": self.plan_id,
            "action_key": self.action_key.to_dict(),
            "action_receipt_digest": self.action_receipt_digest,
            "eligibility_token_digest": self.eligibility_token_digest,
            "energy_unit_digest": self.energy_unit_digest,
            "incumbent_snapshot_digest": self.incumbent_snapshot_digest,
            "working_snapshot_digest": self.working_snapshot_digest,
            "source_binding_digest": self.source_binding_digest,
            "config_binding_digest": self.config_binding_digest,
            "provider_bindings_digest": self.provider_bindings_digest,
            "service_epoch": self.service_epoch,
            "service_ordinal": self.service_ordinal,
        }

    @classmethod
    def from_dict(cls, data: object) -> "StageBActionServicePlan":
        value = _require_dict("service_plan", data)
        if value.get("schema_version") != _SERVICE_PLAN_SCHEMA:
            raise ValueError("unsupported Stage-B service-plan schema.")
        if value.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("Stage-B service-plan action-index schema drift.")
        return cls(
            plan_id=str(value["plan_id"]),
            action_key=PathActionKey.from_dict(_require_dict("action_key", value["action_key"])),
            action_receipt_digest=str(value["action_receipt_digest"]),
            eligibility_token_digest=str(value["eligibility_token_digest"]),
            energy_unit_digest=str(value["energy_unit_digest"]),
            incumbent_snapshot_digest=str(value["incumbent_snapshot_digest"]),
            working_snapshot_digest=str(value["working_snapshot_digest"]),
            source_binding_digest=str(value["source_binding_digest"]),
            config_binding_digest=str(value["config_binding_digest"]),
            provider_bindings_digest=str(value["provider_bindings_digest"]),
            service_epoch=str(value["service_epoch"]),
            service_ordinal=_integer("service_ordinal", value["service_ordinal"]),
        )


@runtime_checkable
class CanonicalPathProvider(Protocol):
    provider_identity: ProviderIdentity

    def canonical_path(self, plan: StageBActionServicePlan) -> StabilizedTrustPathEvidence: ...


@runtime_checkable
class UniformIncumbentBarrierProvider(Protocol):
    provider_identity: ProviderIdentity

    def uniform_incumbent_barrier(self, plan: StageBActionServicePlan) -> UniformBarrierEvidence: ...


@runtime_checkable
class NonlinearActiveManifoldDistanceProvider(Protocol):
    provider_identity: ProviderIdentity

    def nonlinear_active_manifold_distance(self, plan: StageBActionServicePlan) -> EndpointDistanceEvidence: ...


@runtime_checkable
class ConnectedExclusionComponentRefitProvider(Protocol):
    provider_identity: ProviderIdentity

    def connected_component_constrained_refit(self, plan: StageBActionServicePlan) -> ConstrainedWorkingState: ...


@runtime_checkable
class DisposablePowellProvider(Protocol):
    provider_identity: ProviderIdentity

    def disposable_powell(self, plan: StageBActionServicePlan) -> DisposablePowellProbe: ...


@dataclass(frozen=True)
class ServiceCursor:
    service_epoch: str
    next_action_index: int
    expansion_count: int
    completed_services: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "service_epoch", _nonempty("service_epoch", self.service_epoch))
        _integer("next_action_index", self.next_action_index, minimum=1)
        _integer("expansion_count", self.expansion_count)
        _integer("completed_services", self.completed_services)
        if self.next_action_index != self.expansion_count + 1:
            raise ValueError(
                "countable action cursor must advance one canonical index per expansion."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "service_epoch": self.service_epoch,
            "next_action_index": _encode_nonnegative_integer(
                self.next_action_index
            ),
            "expansion_count": _encode_nonnegative_integer(
                self.expansion_count
            ),
            "completed_services": _encode_nonnegative_integer(
                self.completed_services
            ),
        }

    @classmethod
    def from_dict(cls, data: object) -> "ServiceCursor":
        value = _require_dict("service_cursor", data)
        return cls(
            service_epoch=str(value["service_epoch"]),
            next_action_index=_decode_positive_integer(
                "next_action_index", value["next_action_index"]
            ),
            expansion_count=_decode_nonnegative_integer(
                "expansion_count", value["expansion_count"]
            ),
            completed_services=_decode_nonnegative_integer(
                "completed_services", value["completed_services"]
            ),
        )


@dataclass(frozen=True)
class ExecutionHistoryEvent:
    event_index: int
    event_kind: str
    incumbent_snapshot_digest: str
    working_snapshot_digest: str
    chi: float
    rho: float
    completed_services: int
    next_action_index: int
    details_digest: str
    action_receipt_digest: str | None = None

    def __post_init__(self) -> None:
        _integer("event_index", self.event_index)
        object.__setattr__(self, "event_kind", _nonempty("event_kind", self.event_kind))
        for name in ("incumbent_snapshot_digest", "working_snapshot_digest", "details_digest"):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        chi = _nonnegative("chi", self.chi)
        if chi > _FS_DIAMETER:
            raise ValueError("chi cannot exceed the Fubini--Study diameter.")
        _positive("rho", self.rho)
        _integer("completed_services", self.completed_services)
        _integer("next_action_index", self.next_action_index, minimum=1)
        if self.action_receipt_digest is not None:
            object.__setattr__(self, "action_receipt_digest", _sha256("action_receipt_digest", self.action_receipt_digest))

    def to_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_kind": self.event_kind,
            "incumbent_snapshot_digest": self.incumbent_snapshot_digest,
            "working_snapshot_digest": self.working_snapshot_digest,
            "chi": self.chi,
            "rho": self.rho,
            "completed_services": _encode_nonnegative_integer(
                self.completed_services
            ),
            "next_action_index": _encode_nonnegative_integer(
                self.next_action_index
            ),
            "details_digest": self.details_digest,
            "action_receipt_digest": self.action_receipt_digest,
        }

    @classmethod
    def from_dict(cls, data: object) -> "ExecutionHistoryEvent":
        value = _require_dict("execution_history_event", data)
        action = value["action_receipt_digest"]
        return cls(
            event_index=_integer("event_index", value["event_index"]),
            event_kind=str(value["event_kind"]),
            incumbent_snapshot_digest=str(value["incumbent_snapshot_digest"]),
            working_snapshot_digest=str(value["working_snapshot_digest"]),
            chi=float(value["chi"]),
            rho=float(value["rho"]),
            completed_services=_decode_nonnegative_integer(
                "completed_services", value["completed_services"]
            ),
            next_action_index=_decode_positive_integer(
                "next_action_index", value["next_action_index"]
            ),
            details_digest=str(value["details_digest"]),
            action_receipt_digest=None if action is None else str(action),
        )


@dataclass(frozen=True)
class SingleBranchState:
    branch_id: str
    incumbent: ReplayableStateSnapshot
    working: ReplayableStateSnapshot
    chi: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch_id", _nonempty("branch_id", self.branch_id))
        chi = _nonnegative("chi", self.chi)
        if chi > _FS_DIAMETER:
            raise ValueError("chi cannot exceed the Fubini--Study diameter.")
        same_state = self.incumbent.state_fingerprint == self.working.state_fingerprint
        if same_state and self.incumbent.content_digest != self.working.content_digest:
            raise ValueError("equal I/X fingerprints require identical replay snapshots.")
        if same_state and chi != 0.0:
            raise ValueError("I=X requires chi=0.")
        if not same_state and chi <= 0.0:
            raise ValueError("distinct I/X states require positive chi.")

    @property
    def exploring(self) -> bool:
        return self.incumbent.state_fingerprint != self.working.state_fingerprint

    def to_dict(self) -> dict[str, object]:
        return {
            "branch_id": self.branch_id,
            "incumbent": self.incumbent.to_dict(),
            "working": self.working.to_dict(),
            "chi": self.chi,
        }

    @classmethod
    def from_dict(cls, data: object) -> "SingleBranchState":
        value = _require_dict("single_branch_state", data)
        if "working" not in value:
            raise ValueError("production checkpoint is missing X; I-only fallback is forbidden.")
        return cls(
            branch_id=str(value["branch_id"]),
            incumbent=ReplayableStateSnapshot.from_dict(value["incumbent"]),
            working=ReplayableStateSnapshot.from_dict(value["working"]),
            chi=float(value["chi"]),
        )


@dataclass(frozen=True)
class StageBExecutionState:
    """Exactly one branch with replayable I/X and frozen runtime bindings."""

    branch: SingleBranchState
    rho: float
    core_token: EligibilityStateToken
    energy_unit: RunEnergyUnit
    queue: tuple[StageBActionServicePlan, ...]
    service_population: tuple[FrozenServiceItem, ...]
    cursor: ServiceCursor
    providers: StageBProviderBindings
    source: SourceBinding
    config: ConfigurationBinding
    history: tuple[ExecutionHistoryEvent, ...]
    schema_version: str = field(default=_EXECUTION_SCHEMA, init=False)
    branch_count: int = field(default=1, init=False)
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    combined_execution_enabled: bool = field(default=False, init=False)
    integration_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        rho = _positive("rho", self.rho)
        if rho != self.core_token.trust_radius:
            raise ValueError("rho differs from the Stage-B eligibility token.")
        if self.branch.working.state_fingerprint != self.core_token.working_state_fingerprint:
            raise ValueError("X differs from the state bound into the Stage-B core token.")
        if (
            self.branch.incumbent.energy.comparison_epoch
            != self.core_token.comparison_epoch
            or self.branch.working.energy.comparison_epoch
            != self.core_token.comparison_epoch
        ):
            raise ValueError("I/X energy epochs differ from the Stage-B core token.")
        queue = tuple(self.queue)
        service_population = tuple(self.service_population)
        history = tuple(self.history)
        if self.cursor.service_epoch != (queue[0].service_epoch if queue else self.cursor.service_epoch):
            raise ValueError("cursor service epoch differs from queue epoch.")
        if tuple(plan.service_ordinal for plan in queue) != tuple(range(len(queue))):
            raise ValueError("service-plan ordinals must be contiguous from zero.")
        if len({plan.plan_id for plan in queue}) != len(queue):
            raise ValueError("service-plan identifiers must be unique.")
        records = self.core_token.reachable_record_ids
        for plan in queue:
            if plan.service_epoch != self.cursor.service_epoch:
                raise ValueError("service plan epoch differs from cursor epoch.")
            if plan.eligibility_token_digest != self.core_token.digest:
                raise ValueError("service plan core-token binding is stale.")
            if plan.energy_unit_digest != self.energy_unit.digest:
                raise ValueError("service plan energy-unit binding is stale.")
            if plan.incumbent_snapshot_digest != self.branch.incumbent.content_digest:
                raise ValueError("service plan incumbent snapshot binding is stale.")
            if plan.working_snapshot_digest != self.branch.working.content_digest:
                raise ValueError("service plan working snapshot binding is stale.")
            if plan.source_binding_digest != self.source.content_digest:
                raise ValueError("service plan source binding is stale.")
            if plan.config_binding_digest != self.config.content_digest:
                raise ValueError("service plan config binding is stale.")
            if plan.provider_bindings_digest != self.providers.content_digest:
                raise ValueError("service plan provider binding is stale.")
            key = plan.action_key
            if key.record_count != len(records) or records[key.record_order - 1] != key.record_id:
                raise ValueError("service plan action differs from core-token population order.")
            if key.action_index >= self.cursor.next_action_index:
                raise ValueError("service plan lies beyond the countable-action cursor.")
        if len(service_population) != len(queue):
            raise ValueError("every service plan requires one frozen fair-service clock.")
        for plan, clock in zip(queue, service_population):
            if clock.action_key != plan.action_key:
                raise ValueError("service plan and fair-service clock action mismatch.")
            if clock.service_epoch != self.cursor.service_epoch:
                raise ValueError("fair-service clock epoch mismatch.")
            if clock.eligibility_token_digest != self.core_token.digest:
                raise ValueError("fair-service clock core-token binding is stale.")
            if clock.energy_unit_digest != self.energy_unit.digest:
                raise ValueError("fair-service clock energy-unit binding is stale.")
        if self.cursor.completed_services != sum(
            item.service_count for item in service_population
        ):
            raise ValueError("global completed-service count differs from fair clocks.")
        if tuple(event.event_index for event in history) != tuple(range(len(history))):
            raise ValueError("execution history indices must be contiguous from zero.")
        if not history:
            raise ValueError("execution checkpoint requires nonempty bound history.")
        terminal = history[-1]
        if (
            terminal.incumbent_snapshot_digest
            != self.branch.incumbent.content_digest
            or terminal.working_snapshot_digest != self.branch.working.content_digest
            or terminal.chi != self.branch.chi
            or terminal.rho != self.rho
            or terminal.completed_services != self.cursor.completed_services
            or terminal.next_action_index != self.cursor.next_action_index
        ):
            raise ValueError("terminal history event does not bind the live branch/cursor.")
        object.__setattr__(self, "queue", queue)
        object.__setattr__(self, "service_population", service_population)
        object.__setattr__(self, "history", history)

    @property
    def top_level_incumbent(self) -> ReplayableStateSnapshot:
        return self.branch.incumbent

    @property
    def content_digest(self) -> str:
        return _digest(self._payload_dict())

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "branch_count": self.branch_count,
            "action_index_schema": self.action_index_schema,
            "combined_execution_enabled": self.combined_execution_enabled,
            "integration_ready": self.integration_ready,
            "branch": self.branch.to_dict(),
            "rho": self.rho,
            "core_token": self.core_token.to_dict(),
            "energy_unit": self.energy_unit.to_dict(),
            "queue": [plan.to_dict() for plan in self.queue],
            "service_population": [
                item.to_dict() for item in self.service_population
            ],
            "cursor": self.cursor.to_dict(),
            "providers": self.providers.to_dict(),
            "source": self.source.to_dict(),
            "config": self.config.to_dict(),
            "history": [event.to_dict() for event in self.history],
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload_dict(), "content_digest": self.content_digest}

    @classmethod
    def from_dict(cls, data: object) -> "StageBExecutionState":
        value = _require_dict("stage_b_execution_state", data)
        if value.get("schema_version") != _EXECUTION_SCHEMA:
            raise ValueError("unsupported Stage-B execution-state schema.")
        if value.get("branch_count") != 1:
            raise ValueError("Stage-B production envelope requires exactly one branch.")
        if value.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("Stage-B execution-state action-index schema drift.")
        if value.get("combined_execution_enabled") is not False:
            raise ValueError("combined Stage-B execution is not enabled by this envelope.")
        if value.get("integration_ready") is not False:
            raise ValueError("Stage-B runtime integration is not enabled by this envelope.")
        state = cls(
            branch=SingleBranchState.from_dict(value["branch"]),
            rho=float(value["rho"]),
            core_token=EligibilityStateToken.from_dict(_require_dict("core_token", value["core_token"])),
            energy_unit=RunEnergyUnit.from_dict(
                _require_dict("energy_unit", value["energy_unit"])
            ),
            queue=tuple(
                StageBActionServicePlan.from_dict(item)
                for item in _require_list("queue", value["queue"])
            ),
            service_population=tuple(
                FrozenServiceItem.from_dict(
                    _require_dict("service_clock", item)
                )
                for item in _require_list(
                    "service_population", value["service_population"]
                )
            ),
            cursor=ServiceCursor.from_dict(value["cursor"]),
            providers=StageBProviderBindings.from_dict(value["providers"]),
            source=SourceBinding.from_dict(value["source"]),
            config=ConfigurationBinding.from_dict(value["config"]),
            history=tuple(
                ExecutionHistoryEvent.from_dict(item)
                for item in _require_list("history", value["history"])
            ),
        )
        if state.to_dict() != value:
            raise ValueError("Stage-B execution state failed canonical digest round-trip.")
        return state


@dataclass(frozen=True)
class StageBReadinessAssessment:
    runtime_resume_complete: bool
    blockers: tuple[str, ...]
    state_digest: str
    providers_complete: bool
    incumbent_replay_complete: bool
    working_replay_complete: bool
    service_plan_complete: bool
    schema_version: str = field(default=_READINESS_SCHEMA, init=False)
    combined_execution_enabled: bool = field(default=False, init=False)
    integration_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        blockers = tuple(_nonempty("readiness_blocker", item) for item in self.blockers)
        if len(set(blockers)) != len(blockers):
            raise ValueError("readiness blockers must be unique and ordered.")
        object.__setattr__(self, "blockers", blockers)
        object.__setattr__(self, "state_digest", _sha256("state_digest", self.state_digest))
        for name in (
            "runtime_resume_complete",
            "providers_complete",
            "incumbent_replay_complete",
            "working_replay_complete",
            "service_plan_complete",
        ):
            _strict_bool(name, getattr(self, name))
        if self.runtime_resume_complete != (not blockers):
            raise ValueError(
                "runtime-resume completeness must equal absence of blockers."
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "combined_execution_enabled": self.combined_execution_enabled,
            "integration_ready": self.integration_ready,
            "runtime_resume_complete": self.runtime_resume_complete,
            "blockers": list(self.blockers),
            "state_digest": self.state_digest,
            "providers_complete": self.providers_complete,
            "incumbent_replay_complete": self.incumbent_replay_complete,
            "working_replay_complete": self.working_replay_complete,
            "service_plan_complete": self.service_plan_complete,
        }


def assess_stage_b_readiness(state: StageBExecutionState) -> StageBReadinessAssessment:
    """Fail-closed production-envelope audit; this function executes nothing."""

    blockers: list[str] = []
    for role in state.providers.missing_roles:
        blockers.append(f"provider_missing:{role.value}")

    incumbent_complete = state.branch.incumbent.strict_replay_complete
    working_complete = state.branch.working.strict_replay_complete
    if not incumbent_complete:
        blockers.append("incumbent_strict_replay_missing_or_failed")
    if not working_complete:
        blockers.append("working_strict_replay_missing_or_failed")

    for label, snapshot in (
        ("incumbent", state.branch.incumbent),
        ("working", state.branch.working),
    ):
        receipt = snapshot.replay_receipt
        if receipt is None:
            continue
        if receipt.source_digest != state.source.source_digest:
            blockers.append(f"{label}_replay_source_binding_mismatch")
        if receipt.config_digest != state.config.config_digest:
            blockers.append(f"{label}_replay_config_binding_mismatch")
        if receipt.replay_provider != state.providers.state_replay:
            blockers.append(f"{label}_replay_provider_binding_mismatch")
        if (
            receipt.state_consistency_tolerance
            != state.config.state_replay_tolerance
        ):
            blockers.append(f"{label}_replay_tolerance_binding_mismatch")
        if (
            snapshot.payload.prepared_state.norm_error_bound
            > state.config.state_norm_error_tolerance
        ):
            blockers.append(f"{label}_state_norm_error_exceeds_config")

    plan_complete = bool(state.queue) and len(state.queue) == len(
        state.service_population
    )
    if not plan_complete:
        blockers.append("service_queue_empty")

    return StageBReadinessAssessment(
        runtime_resume_complete=not blockers,
        blockers=tuple(blockers),
        state_digest=state.content_digest,
        providers_complete=state.providers.complete,
        incumbent_replay_complete=incumbent_complete,
        working_replay_complete=working_complete,
        service_plan_complete=plan_complete,
    )


@dataclass(frozen=True)
class StageBExecutionCheckpoint:
    """Content-addressed checkpoint for the one-branch production envelope."""

    state: StageBExecutionState
    readiness: StageBReadinessAssessment
    content_digest: str
    schema_version: str = field(default=_CHECKPOINT_SCHEMA, init=False)
    checkpoint_scope: str = field(default=_CHECKPOINT_SCOPE, init=False)
    action_index_schema: str = field(default=ACTION_INDEX_SCHEMA, init=False)
    combined_execution_enabled: bool = field(default=False, init=False)
    integration_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.readiness != assess_stage_b_readiness(self.state):
            raise ValueError("checkpoint readiness assessment is stale.")
        object.__setattr__(self, "content_digest", _sha256("content_digest", self.content_digest))
        if self.content_digest != _digest(self._payload_dict()):
            raise ValueError("Stage-B checkpoint content digest mismatch.")

    @property
    def runtime_resume_complete(self) -> bool:
        return self.readiness.runtime_resume_complete

    @classmethod
    def create(
        cls,
        state: StageBExecutionState | ModeledMinimumCheckpoint | ModeledMinimumRuntimeState,
        *,
        require_replay_complete: bool = False,
    ) -> "StageBExecutionCheckpoint":
        if isinstance(state, (ModeledMinimumCheckpoint, ModeledMinimumRuntimeState)):
            raise ValueError("pure-core scheduler state is not a replay-complete production checkpoint.")
        readiness = assess_stage_b_readiness(state)
        if require_replay_complete and not readiness.runtime_resume_complete:
            raise ValueError("Stage-B production checkpoint is not replay-complete: " + ",".join(readiness.blockers))
        provisional = {
            "schema_version": _CHECKPOINT_SCHEMA,
            "checkpoint_scope": _CHECKPOINT_SCOPE,
            "action_index_schema": ACTION_INDEX_SCHEMA,
            "combined_execution_enabled": False,
            "integration_ready": False,
            "runtime_resume_complete": readiness.runtime_resume_complete,
            "state": state.to_dict(),
            "readiness": readiness.to_dict(),
        }
        return cls(state=state, readiness=readiness, content_digest=_digest(provisional))

    @classmethod
    def create_replay_complete(cls, state: StageBExecutionState) -> "StageBExecutionCheckpoint":
        return cls.create(state, require_replay_complete=True)

    def _payload_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_scope": self.checkpoint_scope,
            "action_index_schema": self.action_index_schema,
            "combined_execution_enabled": self.combined_execution_enabled,
            "integration_ready": self.integration_ready,
            "runtime_resume_complete": self.runtime_resume_complete,
            "state": self.state.to_dict(),
            "readiness": self.readiness.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload_dict(), "content_digest": self.content_digest}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(
        cls,
        payload: str,
        *,
        expected_source: SourceBinding | None = None,
        expected_config: ConfigurationBinding | None = None,
        expected_providers: StageBProviderBindings | None = None,
        expected_action_index_schema: str = ACTION_INDEX_SCHEMA,
        require_replay_complete: bool = False,
    ) -> "StageBExecutionCheckpoint":
        try:
            raw = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Stage-B checkpoint is not valid JSON.") from exc
        data = _require_dict("stage_b_checkpoint", raw)
        if data.get("checkpoint_scope") == "pure_core_scheduler_only" or data.get("schema_version") in {
            "sr_snake_modeled_minimum_core_scheduler_checkpoint_v2",
            "sr_snake_modeled_minimum_core_scheduler_state_v2",
        }:
            raise ValueError("pure-core scheduler checkpoint is not replay-complete.")
        if data.get("schema_version") != _CHECKPOINT_SCHEMA or data.get("checkpoint_scope") != _CHECKPOINT_SCOPE:
            raise ValueError("unsupported Stage-B production checkpoint schema.")
        if expected_action_index_schema != ACTION_INDEX_SCHEMA or data.get("action_index_schema") != ACTION_INDEX_SCHEMA:
            raise ValueError("Stage-B checkpoint action-index schema drift.")
        if data.get("combined_execution_enabled") is not False:
            raise ValueError("combined Stage-B execution is not enabled by this envelope.")
        if data.get("integration_ready") is not False:
            raise ValueError("Stage-B runtime integration is not enabled by this envelope.")
        try:
            state = StageBExecutionState.from_dict(data["state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Stage-B checkpoint state is invalid.") from exc
        readiness = assess_stage_b_readiness(state)
        if data.get("readiness") != readiness.to_dict():
            raise ValueError("Stage-B checkpoint readiness payload is stale or tampered.")
        if (
            data.get("runtime_resume_complete")
            is not readiness.runtime_resume_complete
        ):
            raise ValueError("Stage-B checkpoint replay-complete flag is inconsistent.")
        expected_digest = _digest({key: value for key, value in data.items() if key != "content_digest"})
        if data.get("content_digest") != expected_digest:
            raise ValueError("Stage-B checkpoint content digest mismatch.")
        checkpoint = cls(state=state, readiness=readiness, content_digest=expected_digest)
        if checkpoint.to_dict() != data:
            raise ValueError("Stage-B checkpoint failed canonical round-trip.")
        if expected_source is not None and state.source != expected_source:
            raise ValueError("Stage-B checkpoint source binding drift.")
        if expected_config is not None and state.config != expected_config:
            raise ValueError("Stage-B checkpoint config binding drift.")
        if expected_providers is not None and state.providers != expected_providers:
            raise ValueError("Stage-B checkpoint provider binding drift.")
        if require_replay_complete and not readiness.runtime_resume_complete:
            raise ValueError("Stage-B checkpoint is not replay-complete: " + ",".join(readiness.blockers))
        return checkpoint


@dataclass(frozen=True)
class ExternalIncumbentView:
    """Reader-facing/top-level view that intentionally contains no X payload."""

    incumbent: ReplayableStateSnapshot
    source_binding_digest: str
    config_binding_digest: str
    execution_state_digest: str
    checkpoint_digest: str | None
    schema_version: str = field(default=_EXTERNAL_VIEW_SCHEMA, init=False)

    def __post_init__(self) -> None:
        for name in ("source_binding_digest", "config_binding_digest", "execution_state_digest"):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        if self.checkpoint_digest is not None:
            object.__setattr__(self, "checkpoint_digest", _sha256("checkpoint_digest", self.checkpoint_digest))

    @property
    def energy(self) -> EnergyInterval:
        return self.incumbent.energy

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "incumbent": self.incumbent.to_dict(),
            "source_binding_digest": self.source_binding_digest,
            "config_binding_digest": self.config_binding_digest,
            "execution_state_digest": self.execution_state_digest,
            "checkpoint_digest": self.checkpoint_digest,
        }


def external_incumbent_view(
    state_or_checkpoint: StageBExecutionState | StageBExecutionCheckpoint,
) -> ExternalIncumbentView:
    if isinstance(state_or_checkpoint, StageBExecutionCheckpoint):
        state = state_or_checkpoint.state
        checkpoint_digest: str | None = state_or_checkpoint.content_digest
    elif isinstance(state_or_checkpoint, StageBExecutionState):
        state = state_or_checkpoint
        checkpoint_digest = None
    else:  # pragma: no cover - defensive public boundary
        raise TypeError("external incumbent view requires a Stage-B execution state or checkpoint.")
    return ExternalIncumbentView(
        incumbent=state.branch.incumbent,
        source_binding_digest=state.source.content_digest,
        config_binding_digest=state.config.content_digest,
        execution_state_digest=state.content_digest,
        checkpoint_digest=checkpoint_digest,
    )


__all__ = [
    "CanonicalPathProvider",
    "ConfigurationBinding",
    "ConnectedExclusionComponentRefitProvider",
    "DisposablePowellProvider",
    "ExactComplexCoefficient",
    "ExactHexFloat",
    "ExactOperatorPayload",
    "ExactOperatorTerm",
    "ExactThetaVector",
    "ExecutionHistoryEvent",
    "ExternalIncumbentView",
    "NonlinearActiveManifoldDistanceProvider",
    "OperatorExecutionMode",
    "ParameterLayout",
    "ParameterizationMode",
    "PreparedStateManifest",
    "ProviderIdentity",
    "ProviderRole",
    "ReplayableStatePayload",
    "ReplayableStateSnapshot",
    "ServiceCursor",
    "SingleBranchState",
    "SourceBinding",
    "StageBActionServicePlan",
    "StageBExecutionCheckpoint",
    "StageBExecutionState",
    "StageBProviderBindings",
    "StageBReadinessAssessment",
    "StateLineageEvent",
    "StrictReplayReceipt",
    "ThetaSpace",
    "UniformIncumbentBarrierProvider",
    "assess_stage_b_readiness",
    "external_incumbent_view",
]
