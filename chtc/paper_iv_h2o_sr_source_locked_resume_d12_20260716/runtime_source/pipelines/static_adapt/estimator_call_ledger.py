"""Thread-safe, identity-deduplicated estimator-call accounting.

The ledger records logical estimator primitives, not optimizer iterations or
formula-cardinality proxies.  A primitive is charged once for a projective
physical state, Hamiltonian, backend/precision contract, primitive kind, and
observable/formula identity.  Branches and controller scopes are consumers of
that primitive; they deliberately do not participate in its identity.

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
SUMMARY_SCHEMA = "estimator_call_ledger_summary_v1"
CALL_KEY_SCHEMA = "estimator_call_key_v1"
S_ALG_COMPONENTS = (
    "N_H_outer",
    "N_H_refit",
    "N_grad",
    "N_metric",
)
_S_ALG_COMPONENT_SET = frozenset(S_ALG_COMPONENTS)
_UNBRANCHED_LABEL = "__unbranched__"


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


def canonical_symmetric_pair(left: Any, right: Any) -> tuple[str, str]:
    """Return an order-independent pair of nonempty coordinate identities."""

    pair = (
        _require_text("symmetric_pair.left", left),
        _require_text("symmetric_pair.right", right),
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
    """Branch-independent logical identity of one estimator primitive."""

    projective_state_fingerprint: str
    hamiltonian_fingerprint: str
    backend_fingerprint: str
    precision_contract: str
    primitive_kind: str
    observable_or_formula_identity: str
    symmetric_pair: tuple[str, str] | None = None
    schema: str = CALL_KEY_SCHEMA

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
        if str(self.schema) != CALL_KEY_SCHEMA:
            raise ValueError(f"schema must be {CALL_KEY_SCHEMA!r}.")
        if self.symmetric_pair is not None:
            if len(self.symmetric_pair) != 2:
                raise ValueError("symmetric_pair must contain exactly two identities.")
            object.__setattr__(
                self,
                "symmetric_pair",
                canonical_symmetric_pair(*self.symmetric_pair),
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": CALL_KEY_SCHEMA,
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
        )

    @property
    def primitive_id(self) -> str:
        return _digest(CALL_KEY_SCHEMA, self.as_dict())


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


class EstimatorCallLedger:
    """Thread-safe unique-primitive ledger with branch-aware consumers."""

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
        """Return an execution-occurrence view without unique-charge collapse.

        This view is deliberately separate from :meth:`summary`.  ``summary``
        remains the canonical unique-primitive ``S_alg`` contract; this method
        retains every objective, guard, refit, gradient, and geometry request,
        including repeated calls at the same projective state.
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
                **component_counts,
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

    def summary(
        self,
        *,
        branch_ids: Iterable[str] | None = None,
        include_unbranched: bool = True,
        consumer_scopes: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        """Build an all-branch or selected-lineage unique-call summary.

        ``branch_ids=None`` selects every branch.  For a winning lineage, pass
        every branch/ancestor id in that lineage; unbranched setup calls are
        included by default.  Scope and branch breakdowns are overlapping
        unique-primitive views, so their values need not sum to ``S_alg``.
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
            "S_alg": int(sum(component_counts.values())),
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
        if payload["S_alg"] != payload["unique_primitive_count"]:
            raise AssertionError("S_alg component reconciliation failed.")
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


__all__ = [
    "CALL_KEY_SCHEMA",
    "LEDGER_SCHEMA",
    "SUMMARY_SCHEMA",
    "S_ALG_COMPONENTS",
    "EstimatorCallKey",
    "EstimatorCallLedger",
    "EstimatorCallReceipt",
    "canonical_symmetric_pair",
    "projective_state_fingerprint",
]
