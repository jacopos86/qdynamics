"""Thread-safe estimator-invocation accounting with identity diagnostics.

The ledger records every instrumented logical scalar-estimator invocation, not
optimizer iterations or formula-cardinality proxies.  Physical primitive
identity is retained as a separate diagnostic.  Paper-I clean-algorithm
reporting can therefore retain required repeated optimizer calls while
explicitly reconciling same-iteration reuse and implementation-only duplicate
bridges.  Branches and controller scopes are consumers of a primitive and
deliberately do not participate in its identity.

This module is intentionally standalone.  Call sites in the adaptive pipeline
can be instrumented incrementally without importing controller or reporting
machinery here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import threading
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LEDGER_SCHEMA = "estimator_call_ledger_v1"
SUMMARY_SCHEMA = "estimator_call_ledger_unique_primitive_summary_v2"
PRIMITIVE_SET_SUMMARY_SCHEMA = (
    "estimator_call_ledger_unique_primitive_set_summary_v2"
)
OCCURRENCE_PREFIX_SUMMARY_SCHEMA = (
    "estimator_call_ledger_occurrence_prefix_summary_v1"
)
CALL_KEY_SCHEMA = "estimator_call_key_v1"
CALL_KEY_SCHEMA_V2 = "estimator_call_key_v2"
PHYSICAL_TANGENT_OPERAND_SCHEMA = "physical_tangent_operand_identity_v2"
S_ALG_COMPONENTS = (
    "N_H_outer",
    "N_H_refit",
    "N_grad",
    "N_metric",
)
_S_ALG_COMPONENT_SET = frozenset(S_ALG_COMPONENTS)
_UNBRANCHED_LABEL = "__unbranched__"

# Paper-facing formal-manifold query-oracle coordinates.  The estimator ledger
# intentionally retains the older four-way execution component partition; the
# map below is the authoritative bridge from physical primitive identity to the
# finer scientific query coordinates used by FM-SNAKE reports.
FORMAL_QUERY_CATEGORIES = (
    "N_E",
    "N_grad",
    "N_G",
    "N_Q",
    "N_Hv",
    "N_cross",
)
FORMAL_QUERY_CATEGORY_BY_PRIMITIVE_KIND = {
    "energy": "N_E",
    "hamiltonian_expectation": "N_E",
    "coordinate_gradient": "N_grad",
    "metric_element": "N_G",
    "directional_metric_bilinear": "N_G",
    "tangent_or_metric": "N_G",
    "hessian_element": "N_Q",
    "coordinate_second_derivative": "N_Q",
    "hessian_vector": "N_Hv",
    "cross_state_tangent": "N_cross",
    "state_overlap": "N_cross",
}
FORMAL_QUERY_ALLOWED_LEGACY_COMPONENTS = {
    "N_E": frozenset(("N_H_outer", "N_H_refit")),
    "N_grad": frozenset(("N_grad",)),
    "N_G": frozenset(("N_metric",)),
    "N_Q": frozenset(("N_metric",)),
    "N_Hv": frozenset(("N_metric",)),
    "N_cross": frozenset(("N_metric",)),
}
FORMAL_QUERY_CLOSURE_SCHEMA = (
    "formal_manifold_estimator_ledger_query_closure_v1"
)


def _require_text(name: str, value: Any) -> str:
    if value is None:
        raise ValueError(f"{name} must be nonempty text.")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be nonempty text.")
    return text


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(schema: str, payload: Mapping[str, Any]) -> str:
    envelope = {"schema": str(schema), "payload": dict(payload)}
    return hashlib.sha256(_canonical_json_bytes(envelope)).hexdigest()


@dataclass(frozen=True)
class PhysicalTangentOperandIdentity:
    """Canonical identity of one physical parameter tangent.

    Candidate labels, accepted-coordinate labels, route names, branch IDs,
    whitening-frame labels, and optimizer scopes are intentionally absent.
    They describe consumers or representations of a tangent, not the physical
    derivative circuit itself.  Conversely, every field retained here can
    change the physical tangent and therefore participates in ``operand_id``.

    ``derivative_circuit_fingerprint`` must identify the ordered derivative
    circuit at the parameter point represented by the enclosing projective
    state.  ``generator_fingerprint`` and ``insertion_position`` preserve the
    generator occurrence even when the same generator appears more than once.
    ``parameterization_tie_map_fingerprint`` distinguishes raw, tied, and
    otherwise reparameterized derivatives.  ``tangent_convention`` separates
    horizontal/projective tangents from any future derivative convention.
    """

    derivative_circuit_fingerprint: str
    generator_fingerprint: str
    insertion_position: int
    parameterization_tie_map_fingerprint: str
    tangent_convention: str = "horizontal_projective_parameter_derivative_v1"
    schema: str = PHYSICAL_TANGENT_OPERAND_SCHEMA

    def __post_init__(self) -> None:
        if str(self.schema) != PHYSICAL_TANGENT_OPERAND_SCHEMA:
            raise ValueError(
                f"schema must be {PHYSICAL_TANGENT_OPERAND_SCHEMA!r}."
            )
        for name in (
            "derivative_circuit_fingerprint",
            "generator_fingerprint",
            "parameterization_tie_map_fingerprint",
            "tangent_convention",
        ):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        position = int(self.insertion_position)
        if position < 0:
            raise ValueError("insertion_position must be nonnegative.")
        object.__setattr__(self, "insertion_position", position)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": PHYSICAL_TANGENT_OPERAND_SCHEMA,
            "derivative_circuit_fingerprint": self.derivative_circuit_fingerprint,
            "generator_fingerprint": self.generator_fingerprint,
            "insertion_position": int(self.insertion_position),
            "parameterization_tie_map_fingerprint": (
                self.parameterization_tie_map_fingerprint
            ),
            "tangent_convention": self.tangent_convention,
        }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any]
    ) -> "PhysicalTangentOperandIdentity":
        data = dict(payload)
        return cls(
            derivative_circuit_fingerprint=data.get(
                "derivative_circuit_fingerprint", ""
            ),
            generator_fingerprint=data.get("generator_fingerprint", ""),
            insertion_position=data.get("insertion_position", -1),
            parameterization_tie_map_fingerprint=data.get(
                "parameterization_tie_map_fingerprint", ""
            ),
            tangent_convention=data.get(
                "tangent_convention",
                "horizontal_projective_parameter_derivative_v1",
            ),
            schema=data.get("schema", PHYSICAL_TANGENT_OPERAND_SCHEMA),
        )

    @property
    def operand_id(self) -> str:
        return (
            f"physical_tangent_operand_v2:"
            f"{_digest(PHYSICAL_TANGENT_OPERAND_SCHEMA, self.as_dict())}"
        )


def canonical_physical_operand_id(value: Any) -> str:
    """Return canonical text for a physical estimator operand.

    The typed tangent identity is preferred for new call sites.  Plain text is
    accepted so existing v1 coordinate identities and serialized ledgers
    remain usable during incremental migration.
    """

    if isinstance(value, PhysicalTangentOperandIdentity):
        return value.operand_id
    return _require_text("physical_operand_identity", value)


def canonical_symmetric_pair(left: Any, right: Any) -> tuple[str, str]:
    """Return an order-independent pair of nonempty physical operand IDs."""

    pair = (
        canonical_physical_operand_id(left),
        canonical_physical_operand_id(right),
    )
    return tuple(sorted(pair))


def projective_state_fingerprint(
    statevector: Sequence[complex] | np.ndarray,
    *,
    quantization_decimals: int = 14,
) -> str:
    """Fingerprint a normalized state modulo norm and global phase.

    The largest-magnitude amplitude fixes the global-phase gauge.  Decimal
    quantization makes the gauge fixing stable against roundoff introduced by
    multiplying the input by a unit complex phase.  The quantization contract
    is included in the digest and should remain fixed within one run.
    """

    decimals = int(quantization_decimals)
    if decimals < 0 or decimals > 16:
        raise ValueError("quantization_decimals must be between 0 and 16.")
    vector = np.asarray(statevector, dtype=np.complex128)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("statevector must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(vector.real)) or not np.all(np.isfinite(vector.imag)):
        raise ValueError("statevector must contain only finite amplitudes.")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("statevector norm must be finite and positive.")

    normalized = np.asarray(vector / norm, dtype=np.complex128)
    # Quantize magnitudes before choosing the pivot so equal-magnitude
    # amplitudes retain the same lowest-index tie break after rephasing.
    pivot_magnitudes = np.round(np.abs(normalized), decimals=decimals)
    pivot_index = int(np.argmax(pivot_magnitudes))
    pivot = complex(normalized[pivot_index])
    if abs(pivot) <= 0.0:
        raise ValueError("statevector has no nonzero amplitude.")
    gauge_fixed = normalized / (pivot / abs(pivot))
    # Pin the gauge amplitude exactly onto the positive real axis before
    # quantization, then normalize signed zero for deterministic JSON.
    gauge_fixed[pivot_index] = complex(abs(gauge_fixed[pivot_index]), 0.0)
    real = np.round(gauge_fixed.real, decimals=decimals)
    imag = np.round(gauge_fixed.imag, decimals=decimals)
    real[real == 0.0] = 0.0
    imag[imag == 0.0] = 0.0
    payload = {
        "dimension": int(vector.size),
        "quantization_decimals": decimals,
        "amplitudes": [
            [float(real[index]), float(imag[index])]
            for index in range(vector.size)
        ],
    }
    return f"projective_state_v1:{_digest('projective_state_v1', payload)}"


@dataclass(frozen=True)
class EstimatorCallKey:
    """Branch-independent logical identity of one estimator primitive.

    The v1 schema keeps its original serialized form.  New unary tangent calls
    use ``operand_identity`` with the v2 schema; new symmetric bilinear calls
    put typed physical tangent identities in ``symmetric_pair``.
    """

    projective_state_fingerprint: str
    hamiltonian_fingerprint: str
    backend_fingerprint: str
    precision_contract: str
    primitive_kind: str
    observable_or_formula_identity: str
    symmetric_pair: tuple[str, str] | None = None
    schema: str = CALL_KEY_SCHEMA
    operand_identity: str | PhysicalTangentOperandIdentity | None = None

    def __post_init__(self) -> None:
        for name in (
            "projective_state_fingerprint",
            "hamiltonian_fingerprint",
            "backend_fingerprint",
            "precision_contract",
            "primitive_kind",
            "observable_or_formula_identity",
        ):
            object.__setattr__(self, name, _require_text(name, getattr(self, name)))
        schema = str(self.schema)
        if schema not in {CALL_KEY_SCHEMA, CALL_KEY_SCHEMA_V2}:
            raise ValueError(
                "schema must be one of "
                f"{(CALL_KEY_SCHEMA, CALL_KEY_SCHEMA_V2)!r}."
            )
        object.__setattr__(self, "schema", schema)
        if self.operand_identity is not None:
            object.__setattr__(
                self,
                "operand_identity",
                canonical_physical_operand_id(self.operand_identity),
            )
        if self.symmetric_pair is not None:
            if len(self.symmetric_pair) != 2:
                raise ValueError("symmetric_pair must contain exactly two identities.")
            object.__setattr__(
                self,
                "symmetric_pair",
                canonical_symmetric_pair(*self.symmetric_pair),
            )
        if schema == CALL_KEY_SCHEMA and self.operand_identity is not None:
            raise ValueError(
                "operand_identity requires estimator_call_key_v2; v1 parsing "
                "remains unchanged."
            )
        if self.operand_identity is not None and self.symmetric_pair is not None:
            raise ValueError(
                "a call key may carry one unary operand or one symmetric pair, "
                "not both."
            )

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "projective_state_fingerprint": self.projective_state_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "backend_fingerprint": self.backend_fingerprint,
            "precision_contract": self.precision_contract,
            "primitive_kind": self.primitive_kind,
            "observable_or_formula_identity": self.observable_or_formula_identity,
            "symmetric_pair": (
                None if self.symmetric_pair is None else list(self.symmetric_pair)
            ),
        }
        if self.schema == CALL_KEY_SCHEMA_V2:
            payload["operand_identity"] = self.operand_identity
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EstimatorCallKey":
        data = dict(payload)
        pair = data.get("symmetric_pair")
        if pair is not None and (not isinstance(pair, (list, tuple)) or len(pair) != 2):
            raise ValueError("serialized symmetric_pair must contain two identities.")
        return cls(
            projective_state_fingerprint=data.get("projective_state_fingerprint", ""),
            hamiltonian_fingerprint=data.get("hamiltonian_fingerprint", ""),
            backend_fingerprint=data.get("backend_fingerprint", ""),
            precision_contract=data.get("precision_contract", ""),
            primitive_kind=data.get("primitive_kind", ""),
            observable_or_formula_identity=data.get(
                "observable_or_formula_identity", ""
            ),
            symmetric_pair=(None if pair is None else (pair[0], pair[1])),
            schema=data.get("schema", CALL_KEY_SCHEMA),
            operand_identity=data.get("operand_identity"),
        )

    @property
    def primitive_id(self) -> str:
        return _digest(self.schema, self.as_dict())


@dataclass(frozen=True)
class EstimatorCallReceipt:
    """Result of recording one logical estimator-call occurrence."""

    primitive_id: str
    charged: bool
    charged_component: str
    consumer_component: str
    consumer_scope: str
    branch_id: str | None
    occurrence_sequence: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "charged": bool(self.charged),
            "charged_component": self.charged_component,
            "consumer_component": self.consumer_component,
            "consumer_scope": self.consumer_scope,
            "branch_id": self.branch_id,
            "occurrence_sequence": int(self.occurrence_sequence),
        }


@dataclass
class _ConsumerAggregate:
    component: str
    scope: str
    branch_id: str | None
    first_seen_sequence: int
    last_seen_sequence: int
    occurrence_count: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": self.component,
            "scope": self.scope,
            "branch_id": self.branch_id,
            "first_seen_sequence": int(self.first_seen_sequence),
            "last_seen_sequence": int(self.last_seen_sequence),
            "occurrence_count": int(self.occurrence_count),
        }


@dataclass
class _PrimitiveRecord:
    identity: EstimatorCallKey
    charged_component: str
    first_seen_sequence: int
    last_seen_sequence: int
    occurrence_count: int = 1
    consumers: dict[tuple[str, str, str | None], _ConsumerAggregate] = field(
        default_factory=dict
    )


@dataclass(frozen=True)
class _OccurrenceRecord:
    """One executed logical estimator request before identity deduplication."""

    sequence: int
    primitive_id: str
    component: str
    consumer_scope: str
    branch_id: str | None
    charged: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "primitive_id": str(self.primitive_id),
            "component": str(self.component),
            "consumer_scope": str(self.consumer_scope),
            "branch_id": self.branch_id,
            "charged": bool(self.charged),
        }


def summarize_estimator_occurrence_prefix(
    occurrences: Iterable[Mapping[str, Any]],
    *,
    occurrence_sequence_end_inclusive: int | None = None,
) -> dict[str, Any]:
    """Reconstruct one cumulative executed-query prefix from ordered calls.

    The function accepts an in-memory ledger iterator or a bounded-memory
    stream from a post-run sidecar.  It deliberately does not trust serialized
    cumulative counters: executed calls are recounted, while primitive IDs are
    separately deduplicated for diagnostics and provenance.
    """

    requested_end = (
        None
        if occurrence_sequence_end_inclusive is None
        else int(occurrence_sequence_end_inclusive)
    )
    if requested_end is not None and requested_end < 0:
        raise ValueError(
            "occurrence_sequence_end_inclusive must be nonnegative."
        )

    raw_components = {name: 0 for name in S_ALG_COMPONENTS}
    unique_components = {name: 0 for name in S_ALG_COMPONENTS}
    seen_primitive_ids: set[str] = set()
    expected_sequence = 1
    observed_end = 0

    for serialized in occurrences:
        if not isinstance(serialized, Mapping):
            raise TypeError("each estimator occurrence must be a mapping.")
        row = dict(serialized)
        sequence = int(row.get("sequence", 0))
        if requested_end is not None and sequence > requested_end:
            break
        if sequence != expected_sequence:
            raise ValueError(
                "estimator occurrence sequence is not contiguous at "
                f"{sequence}; expected {expected_sequence}."
            )
        primitive_id = _require_text(
            "occurrence.primitive_id", row.get("primitive_id")
        )
        component = _require_text(
            "occurrence.component", row.get("component")
        )
        if component not in _S_ALG_COMPONENT_SET:
            raise ValueError(
                f"occurrence component is not in {list(S_ALG_COMPONENTS)!r}."
            )
        charged = row.get("charged")
        if not isinstance(charged, bool):
            raise ValueError("occurrence.charged must be Boolean.")

        first_occurrence = primitive_id not in seen_primitive_ids
        if charged is not first_occurrence:
            raise ValueError(
                "occurrence charged flag disagrees with first physical "
                f"identity use at sequence {sequence}."
            )
        raw_components[component] += 1
        if first_occurrence:
            seen_primitive_ids.add(primitive_id)
            unique_components[component] += 1
        observed_end = sequence
        expected_sequence += 1

    if requested_end is not None and observed_end != requested_end:
        raise ValueError(
            "estimator occurrence stream ended before the requested prefix: "
            f"observed={observed_end}, requested={requested_end}."
        )

    raw_total = int(sum(raw_components.values()))
    unique_total = int(sum(unique_components.values()))
    if raw_total != observed_end:
        raise AssertionError("raw occurrence component closure failed.")
    if unique_total != len(seen_primitive_ids):
        raise AssertionError("unique estimator component closure failed.")

    primitive_set_sha256 = _digest(
        PRIMITIVE_SET_SUMMARY_SCHEMA,
        {"primitive_ids": sorted(seen_primitive_ids)},
    )
    return {
        "schema": OCCURRENCE_PREFIX_SUMMARY_SCHEMA,
        "component_contract": list(S_ALG_COMPONENTS),
        "occurrence_sequence_end_inclusive": int(observed_end),
        "cumulative_raw_occurrences": {
            "components": dict(raw_components),
            "total": int(raw_total),
        },
        "cumulative_executed_queries": {
            "components": dict(raw_components),
            "S_alg": int(raw_total),
            "unit": "executed_logical_scalar_estimator_invocation",
        },
        "cumulative_unique_primitives": {
            "components": dict(unique_components),
            "S_unique": int(unique_total),
        },
        "unique_primitive_count": int(unique_total),
        "primitive_set_sha256": str(primitive_set_sha256),
    }


class EstimatorCallLedger:
    """Thread-safe execution ledger with branch-aware identity diagnostics."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._records: dict[str, _PrimitiveRecord] = {}
        self._occurrences: list[_OccurrenceRecord] = []
        self._next_sequence = 1

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    def record_call(
        self,
        identity: EstimatorCallKey,
        *,
        component: str,
        consumer_scope: str,
        branch_id: str | None = None,
    ) -> EstimatorCallReceipt:
        """Record one call occurrence and charge its identity at most once."""

        if not isinstance(identity, EstimatorCallKey):
            raise TypeError("identity must be an EstimatorCallKey.")
        component_text = _require_text("component", component)
        if component_text not in _S_ALG_COMPONENT_SET:
            raise ValueError(
                f"component must be one of {list(S_ALG_COMPONENTS)!r}."
            )
        scope = _require_text("consumer_scope", consumer_scope)
        branch = None if branch_id is None else _require_text("branch_id", branch_id)
        primitive_id = identity.primitive_id

        with self._lock:
            sequence = int(self._next_sequence)
            self._next_sequence += 1
            record = self._records.get(primitive_id)
            charged = record is None
            consumer_key = (component_text, scope, branch)
            if record is None:
                consumer = _ConsumerAggregate(
                    component=component_text,
                    scope=scope,
                    branch_id=branch,
                    first_seen_sequence=sequence,
                    last_seen_sequence=sequence,
                )
                record = _PrimitiveRecord(
                    identity=identity,
                    charged_component=component_text,
                    first_seen_sequence=sequence,
                    last_seen_sequence=sequence,
                    consumers={consumer_key: consumer},
                )
                self._records[primitive_id] = record
            else:
                if record.identity != identity:
                    raise ValueError("primitive ID collision with unequal identities.")
                record.last_seen_sequence = sequence
                record.occurrence_count += 1
                consumer = record.consumers.get(consumer_key)
                if consumer is None:
                    record.consumers[consumer_key] = _ConsumerAggregate(
                        component=component_text,
                        scope=scope,
                        branch_id=branch,
                        first_seen_sequence=sequence,
                        last_seen_sequence=sequence,
                    )
                else:
                    consumer.last_seen_sequence = sequence
                    consumer.occurrence_count += 1
            self._occurrences.append(
                _OccurrenceRecord(
                    sequence=sequence,
                    primitive_id=str(primitive_id),
                    component=str(component_text),
                    consumer_scope=str(scope),
                    branch_id=branch,
                    charged=bool(charged),
                )
            )
            return EstimatorCallReceipt(
                primitive_id=primitive_id,
                charged=charged,
                charged_component=record.charged_component,
                consumer_component=component_text,
                consumer_scope=scope,
                branch_id=branch,
                occurrence_sequence=sequence,
            )

    def occurrence_summary(
        self,
        *,
        branch_ids: Iterable[str] | None = None,
        include_unbranched: bool = True,
        consumer_scopes: Iterable[str] | None = None,
        components: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Return executed calls without physical-identity collapse.

        This is the runtime occurrence audit consumed by the Paper-I
        clean-algorithm recount.  :meth:`summary` retains the historical
        unique-primitive diagnostic.  This method retains every objective,
        guard, refit, gradient, and geometry request, including repeated calls
        at the same projective state.
        """

        branch_filter = (
            None
            if branch_ids is None
            else frozenset(_require_text("branch_id", value) for value in branch_ids)
        )
        scope_filter = (
            None
            if consumer_scopes is None
            else frozenset(
                _require_text("consumer_scope", value) for value in consumer_scopes
            )
        )
        component_filter = (
            None
            if components is None
            else frozenset(_require_text("component", value) for value in components)
        )
        if component_filter is not None and not component_filter.issubset(
            _S_ALG_COMPONENT_SET
        ):
            raise ValueError(
                f"components must be drawn from {list(S_ALG_COMPONENTS)!r}."
            )
        with self._lock:
            selected: list[_OccurrenceRecord] = []
            for occurrence in self._occurrences:
                if scope_filter is not None and occurrence.consumer_scope not in scope_filter:
                    continue
                if component_filter is not None and occurrence.component not in component_filter:
                    continue
                if occurrence.branch_id is None and not include_unbranched:
                    continue
                if branch_filter is not None:
                    if occurrence.branch_id is None:
                        if not include_unbranched:
                            continue
                    elif occurrence.branch_id not in branch_filter:
                        continue
                selected.append(occurrence)

            component_counts = {name: 0 for name in S_ALG_COMPONENTS}
            by_scope: dict[str, int] = {}
            by_branch: dict[str, int] = {}
            primitive_ids: set[str] = set()
            for occurrence in selected:
                component_counts[occurrence.component] += 1
                by_scope[occurrence.consumer_scope] = (
                    int(by_scope.get(occurrence.consumer_scope, 0)) + 1
                )
                branch_label = (
                    _UNBRANCHED_LABEL
                    if occurrence.branch_id is None
                    else occurrence.branch_id
                )
                by_branch[branch_label] = int(by_branch.get(branch_label, 0)) + 1
                primitive_ids.add(str(occurrence.primitive_id))
            occurrence_count = int(len(selected))
            unique_count = int(len(primitive_ids))
            return {
                "schema": "estimator_call_occurrence_summary_v1",
                "selection": {
                    "branch_ids": (
                        None if branch_filter is None else sorted(branch_filter)
                    ),
                    "include_unbranched": bool(include_unbranched),
                    "consumer_scopes": (
                        None if scope_filter is None else sorted(scope_filter)
                    ),
                    "components": (
                        None if component_filter is None else sorted(component_filter)
                    ),
                },
                "component_occurrence_counts": dict(component_counts),
                "components": dict(component_counts),
                **component_counts,
                "S_alg": occurrence_count,
                "unit": "executed_logical_scalar_estimator_invocation",
                "total_call_occurrences": occurrence_count,
                "unique_primitive_count": unique_count,
                "same_identity_reuse_occurrence_count": int(
                    occurrence_count - unique_count
                ),
                "occurrence_count_by_consumer_scope": dict(sorted(by_scope.items())),
                "occurrence_count_by_consumer_branch": dict(sorted(by_branch.items())),
                "primitive_ids": sorted(primitive_ids),
                "occurrence_sequences": [
                    int(occurrence.sequence) for occurrence in selected
                ],
            }

    def closed_occurrence_prefix_summary(self) -> dict[str, Any]:
        """Return the canonical executed-query prefix from ordered calls.

        This is the compact runtime counterpart of the post-run streaming
        audit.  Both paths use :func:`summarize_estimator_occurrence_prefix`,
        so a route receipt and a later sidecar reconstruction share one
        counting implementation.
        """

        with self._lock:
            return summarize_estimator_occurrence_prefix(
                occurrence.as_dict() for occurrence in self._occurrences
            )

    def summary(
        self,
        *,
        branch_ids: Iterable[str] | None = None,
        include_unbranched: bool = True,
        consumer_scopes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Build an all-branch or selected-lineage unique-identity diagnostic.

        ``branch_ids=None`` selects every branch.  For a winning lineage, pass
        every branch/ancestor id in that lineage; unbranched setup calls are
        included by default.  Scope and branch breakdowns are overlapping
        unique-primitive views, so their values need not sum to
        ``S_unique``.
        """

        branch_filter = (
            None
            if branch_ids is None
            else frozenset(_require_text("branch_id", value) for value in branch_ids)
        )
        scope_filter = (
            None
            if consumer_scopes is None
            else frozenset(
                _require_text("consumer_scope", value) for value in consumer_scopes
            )
        )
        with self._lock:
            return self._summary_locked(
                branch_filter=branch_filter,
                include_unbranched=bool(include_unbranched),
                scope_filter=scope_filter,
            )

    def primitive_kind_by_id(self) -> dict[str, str]:
        """Return the immutable physical primitive-kind registry."""

        with self._lock:
            return {
                str(primitive_id): str(record.identity.primitive_kind)
                for primitive_id, record in sorted(self._records.items())
            }

    def summary_for_primitive_ids(
        self,
        primitive_ids: Iterable[str],
    ) -> dict[str, Any]:
        """Summarize an explicit unique-primitive set by global charge.

        Unlike :meth:`summary`, this view does not reassign a shared primitive
        to the earliest consumer selected by a branch/scope filter.  Every
        primitive retains the component charged by its first occurrence in the
        complete ledger, so independently derived primitive-set partitions
        reconcile component by component.  Duplicate input IDs are collapsed
        as set members; IDs absent from the ledger fail closed.
        """

        if isinstance(primitive_ids, (str, bytes, bytearray)):
            raise TypeError(
                "primitive_ids must be an iterable of primitive IDs, not text."
            )
        try:
            selected_ids = frozenset(
                _require_text("primitive_id", value) for value in primitive_ids
            )
        except TypeError as exc:
            raise TypeError(
                "primitive_ids must be an iterable of primitive IDs."
            ) from exc

        with self._lock:
            missing_ids = sorted(selected_ids.difference(self._records))
            if missing_ids:
                raise ValueError(
                    "primitive-set summary references IDs absent from the "
                    f"ledger: {missing_ids[:5]!r}."
                )

            ordered_ids = sorted(selected_ids)
            component_counts = {name: 0 for name in S_ALG_COMPONENTS}
            component_by_primitive_id: dict[str, str] = {}
            for primitive_id in ordered_ids:
                charged_component = str(
                    self._records[primitive_id].charged_component
                )
                if charged_component not in _S_ALG_COMPONENT_SET:
                    raise RuntimeError(
                        "ledger primitive has an unsupported globally charged "
                        f"component: primitive_id={primitive_id!r}, "
                        f"component={charged_component!r}."
                    )
                component_counts[charged_component] += 1
                component_by_primitive_id[primitive_id] = charged_component

            s_unique = int(sum(component_counts.values()))
            if s_unique != len(ordered_ids):
                raise AssertionError(
                    "primitive-set S_unique component reconciliation failed."
                )
            primitive_set_sha256 = _digest(
                PRIMITIVE_SET_SUMMARY_SCHEMA,
                {"primitive_ids": ordered_ids},
            )
            return {
                "schema": PRIMITIVE_SET_SUMMARY_SCHEMA,
                "component_contract": list(S_ALG_COMPONENTS),
                "component_assignment": "ledger_global_charged_component_v1",
                "components": dict(component_counts),
                **component_counts,
                "S_unique": int(s_unique),
                "unique_primitive_count": int(len(ordered_ids)),
                "primitive_ids": ordered_ids,
                "primitive_set_sha256": str(primitive_set_sha256),
                "component_by_primitive_id": dict(
                    sorted(component_by_primitive_id.items())
                ),
            }

    def _summary_locked(
        self,
        *,
        branch_filter: frozenset[str] | None,
        include_unbranched: bool,
        scope_filter: frozenset[str] | None,
    ) -> dict[str, Any]:
        component_counts = {name: 0 for name in S_ALG_COMPONENTS}
        primitive_ids: list[str] = []
        component_by_primitive_id: dict[str, str] = {}
        by_scope: dict[str, set[str]] = {}
        by_branch: dict[str, set[str]] = {}
        cross_component_ids: list[str] = []
        selected_occurrences = 0

        for primitive_id, record in sorted(self._records.items()):
            consumers = []
            for consumer in record.consumers.values():
                if scope_filter is not None and consumer.scope not in scope_filter:
                    continue
                if consumer.branch_id is None and not include_unbranched:
                    continue
                if branch_filter is not None:
                    branch_selected = consumer.branch_id in branch_filter
                    if consumer.branch_id is None:
                        branch_selected = bool(include_unbranched)
                    if not branch_selected:
                        continue
                consumers.append(consumer)
            if not consumers:
                continue

            consumers.sort(
                key=lambda item: (
                    item.first_seen_sequence,
                    item.component,
                    item.scope,
                    "" if item.branch_id is None else item.branch_id,
                )
            )
            charged_component = consumers[0].component
            component_counts[charged_component] += 1
            primitive_ids.append(primitive_id)
            component_by_primitive_id[primitive_id] = charged_component
            selected_occurrences += sum(item.occurrence_count for item in consumers)
            if len({item.component for item in consumers}) > 1:
                cross_component_ids.append(primitive_id)
            for consumer in consumers:
                by_scope.setdefault(consumer.scope, set()).add(primitive_id)
                branch_label = (
                    _UNBRANCHED_LABEL
                    if consumer.branch_id is None
                    else consumer.branch_id
                )
                by_branch.setdefault(branch_label, set()).add(primitive_id)

        unique_count = len(primitive_ids)
        payload: dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "selection": {
                "branch_ids": (
                    None if branch_filter is None else sorted(branch_filter)
                ),
                "include_unbranched": bool(include_unbranched),
                "consumer_scopes": (
                    None if scope_filter is None else sorted(scope_filter)
                ),
            },
            "components": dict(component_counts),
            **component_counts,
            "S_unique": int(sum(component_counts.values())),
            "unique_primitive_count": int(unique_count),
            "selected_call_occurrence_count": int(selected_occurrences),
            "deduplicated_reuse_occurrence_count": int(
                selected_occurrences - unique_count
            ),
            "primitive_ids": primitive_ids,
            "component_by_primitive_id": dict(sorted(component_by_primitive_id.items())),
            "cross_component_reuse_primitive_ids": sorted(cross_component_ids),
            "unique_primitive_count_by_consumer_scope": {
                key: len(value) for key, value in sorted(by_scope.items())
            },
            "unique_primitive_count_by_consumer_branch": {
                key: len(value) for key, value in sorted(by_branch.items())
            },
        }
        if payload["S_unique"] != payload["unique_primitive_count"]:
            raise AssertionError("S_unique component reconciliation failed.")
        return payload

    def to_payload(self) -> dict[str, Any]:
        """Return a deterministic, JSON-serializable full-ledger payload."""

        with self._lock:
            entries = []
            for primitive_id, record in sorted(self._records.items()):
                consumers = sorted(
                    record.consumers.values(),
                    key=lambda item: (
                        item.first_seen_sequence,
                        item.component,
                        item.scope,
                        "" if item.branch_id is None else item.branch_id,
                    ),
                )
                entries.append(
                    {
                        "primitive_id": primitive_id,
                        "identity": record.identity.as_dict(),
                        "charged_component": record.charged_component,
                        "first_seen_sequence": int(record.first_seen_sequence),
                        "last_seen_sequence": int(record.last_seen_sequence),
                        "occurrence_count": int(record.occurrence_count),
                        "reuse_count": int(record.occurrence_count - 1),
                        "consumers": [consumer.as_dict() for consumer in consumers],
                    }
                )
            fingerprint_payload = {
                "component_contract": list(S_ALG_COMPONENTS),
                "entries": entries,
                "occurrences": [
                    occurrence.as_dict() for occurrence in self._occurrences
                ],
            }
            return {
                "schema": LEDGER_SCHEMA,
                "component_contract": list(S_ALG_COMPONENTS),
                "ledger_fingerprint": _digest(LEDGER_SCHEMA, fingerprint_payload),
                "entries": entries,
                "occurrences": [
                    occurrence.as_dict() for occurrence in self._occurrences
                ],
                "summary": self._summary_locked(
                    branch_filter=None,
                    include_unbranched=True,
                    scope_filter=None,
                ),
                "occurrence_summary": self.occurrence_summary(),
            }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EstimatorCallLedger":
        """Restore and validate a full-ledger payload."""

        data = dict(payload)
        if data.get("schema") != LEDGER_SCHEMA:
            raise ValueError("unsupported estimator-call ledger schema.")
        if list(data.get("component_contract", [])) != list(S_ALG_COMPONENTS):
            raise ValueError("estimator-call component contract mismatch.")
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise ValueError("ledger entries must be a list.")

        ledger = cls()
        max_sequence = 0
        with ledger._lock:
            for serialized_record in entries:
                if not isinstance(serialized_record, Mapping):
                    raise ValueError("each ledger entry must be an object.")
                row = dict(serialized_record)
                identity_payload = row.get("identity")
                if not isinstance(identity_payload, Mapping):
                    raise ValueError("ledger entry identity must be an object.")
                identity = EstimatorCallKey.from_dict(identity_payload)
                primitive_id = _require_text("primitive_id", row.get("primitive_id", ""))
                if primitive_id != identity.primitive_id:
                    raise ValueError("serialized primitive_id does not match its identity.")
                if primitive_id in ledger._records:
                    raise ValueError("ledger payload contains a duplicate primitive_id.")
                charged_component = _require_text(
                    "charged_component", row.get("charged_component", "")
                )
                if charged_component not in _S_ALG_COMPONENT_SET:
                    raise ValueError("serialized charged_component is invalid.")
                first_seen = int(row.get("first_seen_sequence", 0))
                last_seen = int(row.get("last_seen_sequence", 0))
                occurrence_count = int(row.get("occurrence_count", 0))
                reuse_count = int(row.get("reuse_count", -1))
                if (
                    first_seen <= 0
                    or last_seen < first_seen
                    or occurrence_count <= 0
                    or reuse_count != occurrence_count - 1
                ):
                    raise ValueError("serialized primitive occurrence counts are invalid.")

                serialized_consumers = row.get("consumers", [])
                if not isinstance(serialized_consumers, list) or not serialized_consumers:
                    raise ValueError("each primitive must retain at least one consumer.")
                consumers: dict[
                    tuple[str, str, str | None], _ConsumerAggregate
                ] = {}
                for serialized_consumer in serialized_consumers:
                    if not isinstance(serialized_consumer, Mapping):
                        raise ValueError("serialized consumer must be an object.")
                    consumer_row = dict(serialized_consumer)
                    component = _require_text(
                        "consumer.component", consumer_row.get("component", "")
                    )
                    if component not in _S_ALG_COMPONENT_SET:
                        raise ValueError("serialized consumer component is invalid.")
                    scope = _require_text(
                        "consumer.scope", consumer_row.get("scope", "")
                    )
                    raw_branch = consumer_row.get("branch_id")
                    branch = (
                        None
                        if raw_branch is None
                        else _require_text("consumer.branch_id", raw_branch)
                    )
                    consumer_first = int(
                        consumer_row.get("first_seen_sequence", 0)
                    )
                    consumer_last = int(consumer_row.get("last_seen_sequence", 0))
                    consumer_count = int(consumer_row.get("occurrence_count", 0))
                    if (
                        consumer_first < first_seen
                        or consumer_last > last_seen
                        or consumer_last < consumer_first
                        or consumer_count <= 0
                    ):
                        raise ValueError("serialized consumer occurrence data are invalid.")
                    consumer_key = (component, scope, branch)
                    if consumer_key in consumers:
                        raise ValueError("ledger payload contains a duplicate consumer.")
                    consumers[consumer_key] = _ConsumerAggregate(
                        component=component,
                        scope=scope,
                        branch_id=branch,
                        first_seen_sequence=consumer_first,
                        last_seen_sequence=consumer_last,
                        occurrence_count=consumer_count,
                    )

                if sum(item.occurrence_count for item in consumers.values()) != occurrence_count:
                    raise ValueError("consumer occurrence counts do not reconcile.")
                if min(item.first_seen_sequence for item in consumers.values()) != first_seen:
                    raise ValueError("consumer first-seen sequence does not reconcile.")
                if max(item.last_seen_sequence for item in consumers.values()) != last_seen:
                    raise ValueError("consumer last-seen sequence does not reconcile.")
                first_consumer = min(
                    consumers.values(), key=lambda item: item.first_seen_sequence
                )
                if first_consumer.component != charged_component:
                    raise ValueError("charged_component is not the first consumer component.")

                ledger._records[primitive_id] = _PrimitiveRecord(
                    identity=identity,
                    charged_component=charged_component,
                    first_seen_sequence=first_seen,
                    last_seen_sequence=last_seen,
                    occurrence_count=occurrence_count,
                    consumers=consumers,
                )
                max_sequence = max(max_sequence, last_seen)
            ledger._next_sequence = max_sequence + 1

            serialized_occurrences = data.get("occurrences", [])
            if not isinstance(serialized_occurrences, list):
                raise ValueError("ledger occurrences must be a list.")
            occurrences: list[_OccurrenceRecord] = []
            for serialized_occurrence in serialized_occurrences:
                if not isinstance(serialized_occurrence, Mapping):
                    raise ValueError("serialized occurrence must be an object.")
                occurrence_row = dict(serialized_occurrence)
                sequence = int(occurrence_row.get("sequence", 0))
                primitive_id = _require_text(
                    "occurrence.primitive_id",
                    occurrence_row.get("primitive_id", ""),
                )
                component = _require_text(
                    "occurrence.component", occurrence_row.get("component", "")
                )
                scope = _require_text(
                    "occurrence.consumer_scope",
                    occurrence_row.get("consumer_scope", ""),
                )
                raw_branch = occurrence_row.get("branch_id")
                branch = (
                    None
                    if raw_branch is None
                    else _require_text("occurrence.branch_id", raw_branch)
                )
                if sequence <= 0 or primitive_id not in ledger._records:
                    raise ValueError("serialized occurrence identity is invalid.")
                if component not in _S_ALG_COMPONENT_SET:
                    raise ValueError("serialized occurrence component is invalid.")
                occurrences.append(
                    _OccurrenceRecord(
                        sequence=sequence,
                        primitive_id=primitive_id,
                        component=component,
                        consumer_scope=scope,
                        branch_id=branch,
                        charged=bool(occurrence_row.get("charged", False)),
                    )
                )
            if [item.sequence for item in occurrences] != list(
                range(1, len(occurrences) + 1)
            ):
                raise ValueError("serialized occurrence sequence is not contiguous.")
            if max_sequence != len(occurrences):
                raise ValueError("occurrence and aggregate sequence counts do not reconcile.")

            occurrences_by_primitive: dict[str, list[_OccurrenceRecord]] = {}
            occurrences_by_consumer: dict[
                tuple[str, str, str, str | None], list[_OccurrenceRecord]
            ] = {}
            consumer_keys_by_primitive: dict[
                str, set[tuple[str, str, str, str | None]]
            ] = {}
            for occurrence in occurrences:
                occurrences_by_primitive.setdefault(
                    occurrence.primitive_id, []
                ).append(occurrence)
                consumer_key = (
                    occurrence.primitive_id,
                    occurrence.component,
                    occurrence.consumer_scope,
                    occurrence.branch_id,
                )
                occurrences_by_consumer.setdefault(consumer_key, []).append(
                    occurrence
                )
                consumer_keys_by_primitive.setdefault(
                    occurrence.primitive_id, set()
                ).add(consumer_key)

            if set(occurrences_by_primitive) != set(ledger._records):
                raise ValueError(
                    "serialized occurrence primitive set does not reconcile "
                    "with aggregate records."
                )
            for primitive_id, record in ledger._records.items():
                primitive_occurrences = occurrences_by_primitive[primitive_id]
                primitive_sequences = [
                    item.sequence for item in primitive_occurrences
                ]
                if (
                    len(primitive_occurrences) != record.occurrence_count
                    or min(primitive_sequences) != record.first_seen_sequence
                    or max(primitive_sequences) != record.last_seen_sequence
                ):
                    raise ValueError(
                        "serialized occurrences do not reconcile with primitive "
                        "occurrence aggregates."
                    )

                charged_occurrences = [
                    item for item in primitive_occurrences if item.charged
                ]
                if (
                    len(charged_occurrences) != 1
                    or charged_occurrences[0].sequence
                    != record.first_seen_sequence
                    or charged_occurrences[0].component
                    != record.charged_component
                ):
                    raise ValueError(
                        "serialized charged occurrence does not reconcile with "
                        "the primitive aggregate."
                    )

                expected_consumer_keys = {
                    (primitive_id, component, scope, branch)
                    for component, scope, branch in record.consumers
                }
                observed_consumer_keys = consumer_keys_by_primitive[primitive_id]
                if observed_consumer_keys != expected_consumer_keys:
                    raise ValueError(
                        "serialized occurrence consumers do not reconcile with "
                        "consumer aggregates."
                    )
                for (
                    component,
                    scope,
                    branch,
                ), consumer in record.consumers.items():
                    consumer_occurrences = occurrences_by_consumer[
                        (primitive_id, component, scope, branch)
                    ]
                    consumer_sequences = [
                        item.sequence for item in consumer_occurrences
                    ]
                    if (
                        len(consumer_occurrences) != consumer.occurrence_count
                        or min(consumer_sequences)
                        != consumer.first_seen_sequence
                        or max(consumer_sequences)
                        != consumer.last_seen_sequence
                    ):
                        raise ValueError(
                            "serialized occurrences do not reconcile with "
                            "consumer occurrence aggregates."
                        )
            ledger._occurrences = occurrences
            ledger._next_sequence = len(occurrences) + 1

        rebuilt = ledger.to_payload()
        if data.get("ledger_fingerprint") != rebuilt["ledger_fingerprint"]:
            raise ValueError("serialized ledger fingerprint mismatch.")
        if data.get("summary") != rebuilt["summary"]:
            raise ValueError("serialized ledger summary mismatch.")
        if data.get("occurrence_summary") != rebuilt["occurrence_summary"]:
            raise ValueError("serialized ledger occurrence summary mismatch.")
        return ledger


def formal_query_category_for_primitive_kind(primitive_kind: Any) -> str:
    """Return the formal query coordinate for one chargeable primitive.

    Unknown kinds fail closed.  Silently assigning a new estimator primitive
    to an existing category would change the paper-facing query oracle.
    """

    kind = _require_text("primitive_kind", primitive_kind)
    category = FORMAL_QUERY_CATEGORY_BY_PRIMITIVE_KIND.get(kind)
    if category is None:
        raise RuntimeError(
            "Formal-manifold estimator ledger contains an unsupported "
            f"chargeable primitive kind: {kind!r}."
        )
    return category


def is_optimizer_or_guard_energy_scope(scope: Any) -> bool:
    """Whether a Hamiltonian occurrence belongs in optimizer/guard nfev.

    State refreshes, prune-surrogate anchors, and final verification remain
    chargeable ``N_E`` work but are not optimizer objective evaluations.
    """

    value = _require_text("consumer_scope", scope)
    return value.startswith("energy:") or value == "finite_angle_objective_guard"


def is_optimizer_energy_scope(scope: Any) -> bool:
    """Whether one Hamiltonian occurrence is a raw optimizer evaluation.

    The legacy ``nfev_total`` counter records objective-function evaluations
    performed by the optimizer.  Finite-angle warm-start guards are separately
    chargeable quantum-query work, but they are not optimizer ``nfev`` and must
    not be folded into that legacy counter.
    """

    value = _require_text("consumer_scope", scope)
    return value.startswith("energy:")


def optimizer_nfev_from_occurrence_summary(
    occurrence_summary: Mapping[str, Any],
) -> int:
    """Derive raw optimizer-only nfev from an occurrence summary."""

    scope_counts = occurrence_summary.get(
        "occurrence_count_by_consumer_scope", {}
    )
    if not isinstance(scope_counts, Mapping):
        raise ValueError(
            "occurrence_count_by_consumer_scope must be a mapping."
        )
    return int(
        sum(
            int(count)
            for scope, count in scope_counts.items()
            if is_optimizer_energy_scope(scope)
        )
    )


def optimizer_or_guard_nfev_from_occurrence_summary(
    occurrence_summary: Mapping[str, Any],
) -> int:
    """Derive optimizer/guard nfev from an occurrence summary."""

    scope_counts = occurrence_summary.get(
        "occurrence_count_by_consumer_scope", {}
    )
    if not isinstance(scope_counts, Mapping):
        raise ValueError(
            "occurrence_count_by_consumer_scope must be a mapping."
        )
    return int(
        sum(
            int(count)
            for scope, count in scope_counts.items()
            if is_optimizer_or_guard_energy_scope(scope)
        )
    )


def _formal_query_unique_view(
    *,
    primitive_kind_by_id: Mapping[str, str],
    primitive_ids: Iterable[str],
) -> dict[str, Any]:
    selected_ids = frozenset(str(value) for value in primitive_ids)
    counts = {name: 0 for name in FORMAL_QUERY_CATEGORIES}
    category_by_id: dict[str, str] = {}
    available_ids = set(str(value) for value in primitive_kind_by_id)
    for primitive_id, primitive_kind in primitive_kind_by_id.items():
        if primitive_id not in selected_ids:
            continue
        category = formal_query_category_for_primitive_kind(
            primitive_kind
        )
        counts[category] += 1
        category_by_id[primitive_id] = category
    missing = sorted(selected_ids.difference(available_ids))
    if missing:
        raise ValueError(
            "formal query view references primitive IDs absent from the ledger: "
            f"{missing[:5]!r}."
        )
    total = int(sum(counts.values()))
    if total != len(selected_ids):
        raise AssertionError("formal query-category reconciliation failed.")
    return {
        "counts": dict(counts),
        **counts,
        "S_alg": total,
        "unique_primitive_count": int(len(selected_ids)),
        "primitive_ids": sorted(selected_ids),
        "query_category_by_primitive_id": dict(sorted(category_by_id.items())),
    }


def build_formal_manifold_query_closure_from_estimator_ledger(
    ledger: EstimatorCallLedger,
    *,
    winning_branch_ids: Iterable[str] | None,
    stored_nfev_total: int | None = None,
) -> dict[str, Any]:
    """Build a disjoint, query-oracle-closed FM accounting view.

    Unbranched work is shared by the winning lineage.  Discarded scientific
    overhead is the exact unique-set difference from the winning/shared set;
    it therefore never double-charges a primitive reused by both consumers.
    Execution occurrences remain disjoint by consumer branch and are used to
    derive the raw optimizer/guard nfev separately from unique ``S_alg``.
    """

    if not isinstance(ledger, EstimatorCallLedger):
        raise TypeError("ledger must be an EstimatorCallLedger.")
    winning_ids_arg = (
        None
        if winning_branch_ids is None
        else tuple(_require_text("branch_id", value) for value in winning_branch_ids)
    )
    all_unique_summary = ledger.summary()
    winning_unique_summary = (
        dict(all_unique_summary)
        if winning_ids_arg is None
        else ledger.summary(
            branch_ids=winning_ids_arg,
            include_unbranched=True,
        )
    )
    all_ids = set(str(value) for value in all_unique_summary["primitive_ids"])
    winning_ids = set(
        str(value) for value in winning_unique_summary["primitive_ids"]
    )
    discarded_only_ids = all_ids.difference(winning_ids)
    primitive_kinds = ledger.primitive_kind_by_id()
    legacy_component_by_id = {
        str(primitive_id): str(component)
        for primitive_id, component in dict(
            all_unique_summary.get("component_by_primitive_id", {})
        ).items()
    }
    missing_component_ids = sorted(all_ids.difference(legacy_component_by_id))
    if missing_component_ids:
        raise RuntimeError(
            "FM estimator ledger lacks legacy charge components for primitive "
            f"IDs: {missing_component_ids[:5]!r}."
        )
    incompatible_components: list[dict[str, str]] = []
    for primitive_id in sorted(all_ids):
        category = formal_query_category_for_primitive_kind(
            primitive_kinds[primitive_id]
        )
        legacy_component = legacy_component_by_id[primitive_id]
        if legacy_component not in FORMAL_QUERY_ALLOWED_LEGACY_COMPONENTS[
            category
        ]:
            incompatible_components.append(
                {
                    "primitive_id": primitive_id,
                    "primitive_kind": primitive_kinds[primitive_id],
                    "formal_query_category": category,
                    "legacy_component": legacy_component,
                }
            )
    if incompatible_components:
        raise RuntimeError(
            "FM estimator primitive-kind/legacy-component compatibility "
            f"failed: {incompatible_components[:5]!r}."
        )
    winning_view = _formal_query_unique_view(
        primitive_kind_by_id=primitive_kinds,
        primitive_ids=winning_ids,
    )
    discarded_view = _formal_query_unique_view(
        primitive_kind_by_id=primitive_kinds,
        primitive_ids=discarded_only_ids,
    )
    all_view = _formal_query_unique_view(
        primitive_kind_by_id=primitive_kinds,
        primitive_ids=all_ids,
    )

    all_occurrences = ledger.occurrence_summary()
    all_branch_labels = sorted(
        str(value)
        for value in dict(
            all_occurrences.get("occurrence_count_by_consumer_branch", {})
        )
        if str(value) != _UNBRANCHED_LABEL
    )
    if winning_ids_arg is None:
        winning_execution = dict(all_occurrences)
        discarded_branch_labels: list[str] = []
    else:
        winning_execution = ledger.occurrence_summary(
            branch_ids=winning_ids_arg,
            include_unbranched=True,
        )
        discarded_branch_labels = sorted(
            set(all_branch_labels).difference(winning_ids_arg)
        )
    discarded_execution = ledger.occurrence_summary(
        branch_ids=discarded_branch_labels,
        include_unbranched=False,
    )
    shared_execution = ledger.occurrence_summary(
        branch_ids=[],
        include_unbranched=True,
    )

    h_components = ("N_H_outer", "N_H_refit")
    all_energy_occurrences = ledger.occurrence_summary(
        components=h_components,
    )
    winning_energy_execution = ledger.occurrence_summary(
        branch_ids=(None if winning_ids_arg is None else winning_ids_arg),
        include_unbranched=True,
        components=h_components,
    )
    discarded_energy_execution = ledger.occurrence_summary(
        branch_ids=discarded_branch_labels,
        include_unbranched=False,
        components=h_components,
    )
    corrected_nfev = optimizer_or_guard_nfev_from_occurrence_summary(
        all_energy_occurrences
    )
    winning_nfev = optimizer_or_guard_nfev_from_occurrence_summary(
        winning_energy_execution
    )
    discarded_nfev = optimizer_or_guard_nfev_from_occurrence_summary(
        discarded_energy_execution
    )
    all_raw_optimizer_nfev = optimizer_nfev_from_occurrence_summary(
        all_energy_occurrences
    )
    winning_raw_optimizer_nfev = optimizer_nfev_from_occurrence_summary(
        winning_energy_execution
    )
    discarded_raw_optimizer_nfev = optimizer_nfev_from_occurrence_summary(
        discarded_energy_execution
    )
    nfev_reconciled = bool(
        corrected_nfev == winning_nfev + discarded_nfev
        and all_raw_optimizer_nfev
        == winning_raw_optimizer_nfev + discarded_raw_optimizer_nfev
    )
    if not nfev_reconciled:
        raise RuntimeError(
            "FM estimator occurrence nfev failed winning/discarded partition."
        )

    unique_reconciled = bool(
        winning_ids.isdisjoint(discarded_only_ids)
        and winning_ids.union(discarded_only_ids) == all_ids
    )
    if not unique_reconciled:
        raise RuntimeError(
            "FM estimator unique primitives failed disjoint-union reconciliation."
        )
    stored = None if stored_nfev_total is None else int(stored_nfev_total)
    return {
        "schema": FORMAL_QUERY_CLOSURE_SCHEMA,
        "primitive_kind_to_query_category": dict(
            sorted(FORMAL_QUERY_CATEGORY_BY_PRIMITIVE_KIND.items())
        ),
        "formal_query_allowed_legacy_components": {
            category: sorted(components)
            for category, components in sorted(
                FORMAL_QUERY_ALLOWED_LEGACY_COMPONENTS.items()
            )
        },
        "primitive_kind_legacy_component_compatible": True,
        "winning_branch_ids": (
            all_branch_labels if winning_ids_arg is None else list(winning_ids_arg)
        ),
        "discarded_execution_branch_ids": list(discarded_branch_labels),
        "winning_branch": winning_view,
        "discarded_branch_operational_overhead": discarded_view,
        "all_executed": all_view,
        "executed_occurrence_accounting": {
            "all_execution": dict(all_occurrences),
            "winning_plus_shared_execution": dict(winning_execution),
            "discarded_branch_execution": dict(discarded_execution),
            "shared_unbranched_execution": dict(shared_execution),
            "all_hamiltonian_execution": dict(all_energy_occurrences),
            "winning_plus_shared_hamiltonian_execution": dict(
                winning_energy_execution
            ),
            "discarded_branch_hamiltonian_execution": dict(
                discarded_energy_execution
            ),
        },
        "primitive_set_reconciliation": {
            "winning_discarded_disjoint": True,
            "union_equals_all_executed": True,
            "winning_count": int(len(winning_ids)),
            "discarded_count": int(len(discarded_only_ids)),
            "all_executed_count": int(len(all_ids)),
        },
        "occurrence_reconciliation": {
            "all_occurrence_count": int(
                all_occurrences["total_call_occurrences"]
            ),
            "winning_plus_shared_occurrence_count": int(
                winning_execution["total_call_occurrences"]
            ),
            "discarded_occurrence_count": int(
                discarded_execution["total_call_occurrences"]
            ),
            "all_equals_winning_plus_discarded": bool(
                int(all_occurrences["total_call_occurrences"])
                == int(winning_execution["total_call_occurrences"])
                + int(discarded_execution["total_call_occurrences"])
            ),
            "same_identity_reuse_occurrence_count": int(
                all_occurrences["same_identity_reuse_occurrence_count"]
            ),
        },
        "chargeable_unique_energy_evaluation_ids": sorted(
            primitive_id
            for primitive_id, category in all_view[
                "query_category_by_primitive_id"
            ].items()
            if category == "N_E"
        ),
        "chargeable_unique_energy_evaluations": int(all_view["N_E"]),
        "energy_evaluation_occurrence_ids": [
            str(value)
            for value in all_energy_occurrences["occurrence_sequences"]
        ],
        "raw_energy_evaluation_occurrence_count": int(
            all_energy_occurrences["total_call_occurrences"]
        ),
        "raw_minus_unique_energy_evaluations": int(
            all_energy_occurrences["total_call_occurrences"] - all_view["N_E"]
        ),
        "stored_nfev_total": stored,
        "raw_optimizer_nfev_all_execution": int(all_raw_optimizer_nfev),
        "raw_optimizer_nfev_winning_lineage": int(
            winning_raw_optimizer_nfev
        ),
        "raw_optimizer_nfev_discarded_operational_overhead": int(
            discarded_raw_optimizer_nfev
        ),
        "stored_nfev_matches_winning_raw_optimizer": (
            None
            if stored is None
            else bool(stored == winning_raw_optimizer_nfev)
        ),
        "corrected_nfev_total": int(corrected_nfev),
        "nfev_correction": (
            None if stored is None else int(corrected_nfev - stored)
        ),
        "nfev_winning_lineage": int(winning_nfev),
        "nfev_discarded_operational_overhead": int(discarded_nfev),
        "nfev_reconciled": True,
    }


__all__ = [
    "CALL_KEY_SCHEMA",
    "CALL_KEY_SCHEMA_V2",
    "LEDGER_SCHEMA",
    "PHYSICAL_TANGENT_OPERAND_SCHEMA",
    "PRIMITIVE_SET_SUMMARY_SCHEMA",
    "SUMMARY_SCHEMA",
    "S_ALG_COMPONENTS",
    "FORMAL_QUERY_CATEGORIES",
    "FORMAL_QUERY_CATEGORY_BY_PRIMITIVE_KIND",
    "FORMAL_QUERY_ALLOWED_LEGACY_COMPONENTS",
    "FORMAL_QUERY_CLOSURE_SCHEMA",
    "EstimatorCallKey",
    "EstimatorCallLedger",
    "EstimatorCallReceipt",
    "PhysicalTangentOperandIdentity",
    "build_formal_manifold_query_closure_from_estimator_ledger",
    "canonical_physical_operand_id",
    "canonical_symmetric_pair",
    "formal_query_category_for_primitive_kind",
    "is_optimizer_energy_scope",
    "is_optimizer_or_guard_energy_scope",
    "optimizer_nfev_from_occurrence_summary",
    "optimizer_or_guard_nfev_from_occurrence_summary",
    "projective_state_fingerprint",
]
