"""Private accepted-singleton transition boundary for the default SR route.

The numerical kernel remains in :mod:`pipelines.static_adapt.adapt_pipeline`.
This module owns the immutable controller transaction and validates that the
kernel consumes exactly the selected generator-position record, commits one
zero-angle admission, performs the characterized full supported-FS refit, and
closes one estimator-ledger prefix.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Protocol

from pipelines.static_adapt.sr_snake._selection import (
    _CombinatorialBatchAdmissionDecision,
    _GreedyBatchAdmissionDecision,
    _SingletonAdmissionDecision,
)

_LEDGER_COMPONENTS = ("N_H_outer", "N_H_refit", "N_grad", "N_metric")


def _sha256_json(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class _AcceptedStateSnapshot:
    """Portable identity and numerical snapshot of one accepted state."""

    controller_round: int
    accepted_operator_ids: tuple[str, ...]
    accepted_insertion_positions: tuple[int, ...]
    logical_parameter_ids: tuple[str, ...]
    logical_parameter_values: tuple[float, ...]
    runtime_parameter_ids: tuple[str, ...]
    runtime_parameter_values: tuple[float, ...]
    accepted_energy: float
    accepted_state_fingerprint: str
    available_generator_ids: tuple[str, ...]
    selection_counts: tuple[tuple[str, int], ...]
    trust_state_identity: str
    optimizer_memory_identity: str
    estimator_prefix_identity: str

    def __post_init__(self) -> None:
        if self.controller_round < 0:
            raise ValueError("controller_round must be non-negative")
        if len(self.accepted_operator_ids) != len(
            self.accepted_insertion_positions
        ):
            raise ValueError(
                "accepted operator and insertion-position identities disagree"
            )
        if self.accepted_insertion_positions != tuple(
            range(len(self.accepted_operator_ids))
        ):
            raise ValueError(
                "accepted insertion positions must identify current coordinates"
            )
        if len(self.logical_parameter_ids) != len(
            self.logical_parameter_values
        ):
            raise ValueError("logical parameter identities and values disagree")
        if len(self.runtime_parameter_ids) != len(
            self.runtime_parameter_values
        ):
            raise ValueError("runtime parameter identities and values disagree")
        if len(set(self.logical_parameter_ids)) != len(
            self.logical_parameter_ids
        ):
            raise ValueError("logical parameter identities must be unique")
        if len(set(self.runtime_parameter_ids)) != len(
            self.runtime_parameter_ids
        ):
            raise ValueError("runtime parameter identities must be unique")
        if any(
            not math.isfinite(float(value))
            for value in (
                *self.logical_parameter_values,
                *self.runtime_parameter_values,
                self.accepted_energy,
            )
        ):
            raise ValueError("accepted numerical state must be finite")
        if not self.accepted_state_fingerprint:
            raise ValueError("accepted_state_fingerprint must be non-empty")
        if len(set(self.available_generator_ids)) != len(
            self.available_generator_ids
        ):
            duplicate_ids = sorted(
                identity
                for identity in set(self.available_generator_ids)
                if self.available_generator_ids.count(identity) > 1
            )
            raise ValueError(
                "available generator identities must be unique; "
                f"duplicates={duplicate_ids!r}"
            )
        selection_ids = tuple(identity for identity, _ in self.selection_counts)
        if len(set(selection_ids)) != len(selection_ids):
            raise ValueError("selection-count identities must be unique")
        if any(count < 0 for _, count in self.selection_counts):
            raise ValueError("selection counts must be non-negative")
        if not self.trust_state_identity:
            raise ValueError("trust state identity must be non-empty")
        if not self.optimizer_memory_identity:
            raise ValueError("optimizer memory identity must be non-empty")
        if not self.estimator_prefix_identity:
            raise ValueError("estimator prefix identity must be non-empty")


@dataclass(frozen=True, slots=True)
class _AdmissionReceipt:
    selected_domain_record_id: str
    generator_id: str
    pool_index: int
    insertion_position: int
    initial_logical_value: float
    logical_parameter_count_before: int
    logical_parameter_count_after: int
    runtime_parameter_count_before: int
    runtime_parameter_count_after: int
    old_to_new_logical_indices: tuple[int, ...]
    old_to_new_runtime_indices: tuple[int, ...]
    inserted_runtime_indices: tuple[int, ...]
    optimizer_memory_identity_before: str
    optimizer_memory_identity_after: str
    selection_count_before: int
    selection_count_after: int
    available_before: bool
    available_after: bool
    source_identity: str
    child_identity: str

    def __post_init__(self) -> None:
        if not self.selected_domain_record_id or not self.generator_id:
            raise ValueError("admission identities must be non-empty")
        if self.pool_index < 0 or self.insertion_position < 0:
            raise ValueError("admission pool index and position must be non-negative")
        if not math.isfinite(self.initial_logical_value):
            raise ValueError("initial admitted value must be finite")
        if self.logical_parameter_count_after != (
            self.logical_parameter_count_before + 1
        ):
            raise ValueError("singleton admission must add one logical parameter")
        if self.runtime_parameter_count_after <= self.runtime_parameter_count_before:
            raise ValueError(
                "singleton admission must add at least one runtime parameter"
            )
        expected_logical_mapping = tuple(
            index
            if index < self.insertion_position
            else index + 1
            for index in range(self.logical_parameter_count_before)
        )
        if self.old_to_new_logical_indices != expected_logical_mapping:
            raise ValueError("logical insertion mapping is inconsistent")
        added_runtime_count = (
            self.runtime_parameter_count_after
            - self.runtime_parameter_count_before
        )
        if len(self.old_to_new_runtime_indices) != (
            self.runtime_parameter_count_before
        ):
            raise ValueError("runtime insertion mapping is incomplete")
        if len(self.inserted_runtime_indices) != added_runtime_count:
            raise ValueError("inserted runtime identity count is inconsistent")
        mapped_runtime_set = set(self.old_to_new_runtime_indices)
        inserted_runtime_set = set(self.inserted_runtime_indices)
        if mapped_runtime_set.intersection(inserted_runtime_set):
            raise ValueError("old and inserted runtime coordinates overlap")
        if mapped_runtime_set.union(inserted_runtime_set) != set(
            range(self.runtime_parameter_count_after)
        ):
            raise ValueError("runtime insertion mapping does not cover the layout")
        if self.selection_count_after != self.selection_count_before + 1:
            raise ValueError("singleton admission must increment selection count once")
        if not self.available_before:
            raise ValueError("admitted generator must be available before admission")
        if not self.optimizer_memory_identity_before:
            raise ValueError("pre-admission optimizer identity must be non-empty")
        if not self.optimizer_memory_identity_after:
            raise ValueError("post-admission optimizer identity must be non-empty")
        if not self.source_identity or not self.child_identity:
            raise ValueError("source and child admission identities must be non-empty")


@dataclass(frozen=True, slots=True)
class _GreedyBatchAdmissionReceipt:
    """Composed zero-angle admission receipt for one ordered greedy batch."""

    selected_domain_record_ids: tuple[str, ...]
    generator_ids: tuple[str, ...]
    pool_indices: tuple[int, ...]
    original_insertion_positions: tuple[int, ...]
    effective_insertion_positions: tuple[int, ...]
    initial_logical_values: tuple[float, ...]
    logical_parameter_count_before: int
    logical_parameter_count_after: int
    old_to_new_logical_indices: tuple[int, ...]
    inserted_logical_indices: tuple[int, ...]
    admitted_runtime_counts: tuple[int, ...]
    runtime_insert_positions: tuple[int, ...]
    runtime_parameter_count_before: int
    runtime_parameter_count_after: int
    old_to_new_runtime_indices: tuple[int, ...]
    inserted_runtime_indices: tuple[int, ...]
    optimizer_memory_identity_before: str
    optimizer_memory_identity_after: str
    selection_counts_before: tuple[int, ...]
    selection_counts_after: tuple[int, ...]
    available_before: tuple[bool, ...]
    available_after: tuple[bool, ...]
    source_identities: tuple[str, ...]
    child_identities: tuple[str, ...]

    def __post_init__(self) -> None:
        batch_size = len(self.selected_domain_record_ids)
        if not 1 <= batch_size <= 5:
            raise ValueError("greedy batch admission size must lie in 1..5")
        ordered_member_fields = (
            self.generator_ids,
            self.pool_indices,
            self.original_insertion_positions,
            self.effective_insertion_positions,
            self.initial_logical_values,
            self.admitted_runtime_counts,
            self.runtime_insert_positions,
            self.selection_counts_before,
            self.selection_counts_after,
            self.available_before,
            self.available_after,
            self.source_identities,
            self.child_identities,
        )
        if any(len(values) != batch_size for values in ordered_member_fields):
            raise ValueError(
                "greedy batch admission member receipts must have one "
                "entry per selected record"
            )
        identity_fields = (
            self.selected_domain_record_ids,
            self.generator_ids,
            self.source_identities,
            self.child_identities,
        )
        if any(any(not value for value in values) for values in identity_fields):
            raise ValueError(
                "greedy batch admission identities must be non-empty"
            )
        if len(set(self.selected_domain_record_ids)) != batch_size:
            raise ValueError(
                "greedy batch admission record identities must be unique"
            )
        if len(set(self.generator_ids)) != batch_size:
            raise ValueError(
                "greedy batch admission generator identities must be unique"
            )
        if any(index < 0 for index in self.pool_indices):
            raise ValueError(
                "greedy batch admission pool indices must be non-negative"
            )
        if self.logical_parameter_count_before < 0:
            raise ValueError(
                "greedy batch logical pre-count must be non-negative"
            )
        if self.logical_parameter_count_after != (
            self.logical_parameter_count_before + batch_size
        ):
            raise ValueError(
                "greedy batch admission must add one logical parameter "
                "per selected record"
            )
        if any(
            position < 0
            or position > self.logical_parameter_count_before
            for position in self.original_insertion_positions
        ):
            raise ValueError(
                "greedy batch original insertion positions are invalid"
            )
        expected_effective: list[int] = []
        original_seen: list[int] = []
        for position in self.original_insertion_positions:
            expected_effective.append(
                int(
                    position
                    + sum(
                        1
                        for previous in original_seen
                        if previous <= position
                    )
                )
            )
            original_seen.append(int(position))
        if self.effective_insertion_positions != tuple(expected_effective):
            raise ValueError(
                "greedy batch effective insertion positions are inconsistent"
            )
        expected_old_mapping = tuple(
            index
            + sum(
                1
                for position in self.original_insertion_positions
                if position <= index
            )
            for index in range(self.logical_parameter_count_before)
        )
        if self.old_to_new_logical_indices != expected_old_mapping:
            raise ValueError(
                "greedy batch logical insertion mapping is inconsistent"
            )
        expected_inserted = tuple(
            position
            + sum(
                1
                for other in self.original_insertion_positions
                if other < position
            )
            + sum(
                1
                for previous_index in range(member_index)
                if self.original_insertion_positions[previous_index]
                == position
            )
            for member_index, position in enumerate(
                self.original_insertion_positions
            )
        )
        if self.inserted_logical_indices != expected_inserted:
            raise ValueError(
                "greedy batch final inserted logical indices are inconsistent"
            )
        if any(value != 0.0 for value in self.initial_logical_values):
            raise ValueError(
                "greedy batch admission requires exact zero amplitudes"
            )
        if any(count <= 0 for count in self.admitted_runtime_counts):
            raise ValueError(
                "each greedy batch member must add runtime coordinates"
            )
        if self.runtime_parameter_count_before < 0:
            raise ValueError(
                "greedy batch runtime pre-count must be non-negative"
            )
        if self.runtime_parameter_count_after != (
            self.runtime_parameter_count_before
            + sum(self.admitted_runtime_counts)
        ):
            raise ValueError(
                "greedy batch runtime counts do not reconcile"
            )
        running_runtime_count = self.runtime_parameter_count_before
        for position, count in zip(
            self.runtime_insert_positions,
            self.admitted_runtime_counts,
            strict=True,
        ):
            if position < 0 or position > running_runtime_count:
                raise ValueError(
                    "greedy batch runtime insertion position is invalid"
                )
            running_runtime_count += count
        if len(self.old_to_new_runtime_indices) != (
            self.runtime_parameter_count_before
        ):
            raise ValueError(
                "greedy batch runtime insertion mapping is incomplete"
            )
        inserted_runtime_count = sum(self.admitted_runtime_counts)
        if len(self.inserted_runtime_indices) != inserted_runtime_count:
            raise ValueError(
                "greedy batch inserted runtime identity count is inconsistent"
            )
        old_runtime_set = set(self.old_to_new_runtime_indices)
        inserted_runtime_set = set(self.inserted_runtime_indices)
        if old_runtime_set.intersection(inserted_runtime_set):
            raise ValueError(
                "greedy batch old and inserted runtime coordinates overlap"
            )
        if old_runtime_set.union(inserted_runtime_set) != set(
            range(self.runtime_parameter_count_after)
        ):
            raise ValueError(
                "greedy batch runtime insertion mapping does not cover "
                "the final layout"
            )
        if any(
            after != before + 1
            for before, after in zip(
                self.selection_counts_before,
                self.selection_counts_after,
                strict=True,
            )
        ):
            raise ValueError(
                "greedy batch admission must increment every source once"
            )
        if not all(self.available_before):
            raise ValueError(
                "every greedy batch source must be available before admission"
            )
        if (
            not self.optimizer_memory_identity_before
            or not self.optimizer_memory_identity_after
        ):
            raise ValueError(
                "greedy batch optimizer identities must be non-empty"
            )

    @property
    def composition_identity(self) -> str:
        return _sha256_json(
            {
                "selected_domain_record_ids": self.selected_domain_record_ids,
                "generator_ids": self.generator_ids,
                "original_insertion_positions": (
                    self.original_insertion_positions
                ),
                "effective_insertion_positions": (
                    self.effective_insertion_positions
                ),
                "old_to_new_logical_indices": (
                    self.old_to_new_logical_indices
                ),
                "inserted_logical_indices": self.inserted_logical_indices,
                "old_to_new_runtime_indices": (
                    self.old_to_new_runtime_indices
                ),
                "inserted_runtime_indices": self.inserted_runtime_indices,
            }
        )


@dataclass(frozen=True, slots=True)
class _CombinatorialBatchAdmissionReceipt(_GreedyBatchAdmissionReceipt):
    """Mechanical zero-angle receipt for one combinatorial subset."""


@dataclass(frozen=True, slots=True)
class _SupportedFSRefitReceipt:
    policy_identity: str
    scope_identity: str
    optimizer_identity: str
    chart_identity: str
    chart_dimension: int
    supported_rank: int
    active_logical_indices: tuple[int, ...]
    external_gram_receipt_identity: str
    external_gram_reused: bool
    initialization_policy_identity: str
    initialization_status: str
    initialization_guard_nfev: int
    optimizer_success: bool
    optimizer_nfev: int
    optimizer_nit: int
    optimizer_message: str

    def __post_init__(self) -> None:
        if self.policy_identity != "supported_fs_whitened_fixed_v1":
            raise ValueError("default transition requires supported-FS whitening")
        if self.scope_identity != "full_ansatz_v1":
            raise ValueError("default transition requires a full-ansatz refit")
        if self.optimizer_identity != "POWELL":
            raise ValueError("default transition requires the Powell optimizer")
        if not self.chart_identity or self.chart_dimension <= 0:
            raise ValueError("supported-FS chart identity and dimension are required")
        if not 0 < self.supported_rank <= self.chart_dimension:
            raise ValueError("supported-FS rank must lie within the chart")
        if self.active_logical_indices != tuple(range(self.chart_dimension)):
            raise ValueError(
                "full accepted refit must cover every logical coordinate"
            )
        if (
            not self.external_gram_reused
            or not self.external_gram_receipt_identity
        ):
            raise ValueError(
                "default transition requires same-iteration external Gram reuse"
            )
        if not self.initialization_policy_identity:
            raise ValueError(
                "accepted-refit initialization policy must be explicit"
            )
        if self.initialization_status not in {
            "disabled",
            "accepted",
            "rejected",
            "error",
            "unavailable",
        }:
            raise ValueError("accepted-refit initialization status is invalid")
        if self.initialization_guard_nfev < 0:
            raise ValueError(
                "accepted-refit initialization counter must be non-negative"
            )
        if self.optimizer_nfev < 0 or self.optimizer_nit < 0:
            raise ValueError("optimizer counters must be non-negative")


@dataclass(frozen=True, slots=True)
class _AdaptiveTrustUpdateReceipt:
    policy_identity: str
    trust_state_identity_before: str
    trust_state_identity_after: str
    update_count_before: int
    update_count_after: int
    payload_identity: str
    endpoint_overlap_query_charge: int

    def __post_init__(self) -> None:
        if not self.policy_identity:
            raise ValueError("adaptive trust policy identity must be non-empty")
        if (
            not self.trust_state_identity_before
            or not self.trust_state_identity_after
            or not self.payload_identity
        ):
            raise ValueError("adaptive trust identities must be non-empty")
        if self.update_count_after != self.update_count_before + 1:
            raise ValueError("adaptive trust state must update exactly once")
        expected_overlap_charge = {
            "source_metric_inverse_sqrt_no_overlap_v1": 0,
            "displacement_calibrated_unbounded_v2": 1,
        }.get(self.policy_identity)
        if (
            expected_overlap_charge is None
            or self.endpoint_overlap_query_charge
            != expected_overlap_charge
        ):
            raise ValueError(
                "adaptive trust endpoint-overlap charge disagrees with its "
                "policy"
            )


@dataclass(frozen=True, slots=True)
class _NonWorseningReceipt:
    energy_before: float
    energy_after: float
    absolute_tolerance: float
    comparison_semantics: str
    accepted: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.energy_before) or not math.isfinite(
            self.energy_after
        ):
            raise ValueError("non-worsening energies must be finite")
        if not math.isfinite(self.absolute_tolerance) or self.absolute_tolerance < 0:
            raise ValueError("non-worsening tolerance must be finite and non-negative")
        if not self.comparison_semantics:
            raise ValueError("non-worsening comparison semantics must be explicit")
        expected = bool(
            self.energy_after <= self.energy_before + self.absolute_tolerance
        )
        if self.accepted != expected:
            raise ValueError(
                "non-worsening decision disagrees with energies and tolerance"
            )


@dataclass(frozen=True, slots=True)
class _RoundLedgerClosure:
    controller_round: int
    checkpoint_sequence: int
    prefix_identity_before: str
    prefix_identity_after: str
    sequence_start_exclusive: int
    first_sequence_index: int
    sequence_indices: tuple[int, ...]
    occurrence_ids: tuple[str, ...]
    reuse_identities: tuple[str | None, ...]
    round_s_alg_components: tuple[tuple[str, int], ...]
    round_s_unique_components: tuple[tuple[str, int], ...]
    cumulative_s_alg: int
    cumulative_s_alg_components: tuple[tuple[str, int], ...]
    cumulative_s_unique: int
    cumulative_s_unique_components: tuple[tuple[str, int], ...]
    close_count: int

    def __post_init__(self) -> None:
        if (
            self.controller_round < 0
            or self.checkpoint_sequence <= 0
            or self.sequence_start_exclusive < 0
            or self.first_sequence_index < 0
        ):
            raise ValueError("ledger round and first sequence must be non-negative")
        if not self.prefix_identity_before or not self.prefix_identity_after:
            raise ValueError("ledger prefix identities must be non-empty")
        if len(self.sequence_indices) != len(self.occurrence_ids):
            raise ValueError(
                "ledger sequence and occurrence identity counts must agree"
            )
        if len(self.occurrence_ids) != len(self.reuse_identities):
            raise ValueError(
                "ledger occurrence and reuse identity counts must agree"
            )
        if self.sequence_indices:
            if self.first_sequence_index != self.sequence_start_exclusive + 1:
                raise ValueError(
                    "ledger first sequence must follow its closed prefix"
                )
            if self.sequence_indices != tuple(
                range(
                    self.first_sequence_index,
                    self.first_sequence_index
                    + len(self.sequence_indices),
                )
            ):
                raise ValueError(
                    "ledger occurrence sequence must be exact and contiguous"
                )
        if any(not occurrence_id for occurrence_id in self.occurrence_ids):
            raise ValueError("ledger occurrence identities must be non-empty")
        if len(set(self.occurrence_ids)) != len(self.occurrence_ids):
            raise ValueError("ledger occurrence identities must be unique")
        component_receipts = (
            self.round_s_alg_components,
            self.round_s_unique_components,
            self.cumulative_s_alg_components,
            self.cumulative_s_unique_components,
        )
        if any(
            tuple(name for name, _ in receipt) != _LEDGER_COMPONENTS
            for receipt in component_receipts
        ):
            raise ValueError(
                "ledger component receipts must use the canonical order"
            )
        if any(
            value < 0
            for receipt in component_receipts
            for _, value in receipt
        ):
            raise ValueError("ledger component counts must be non-negative")
        if sum(value for _, value in self.round_s_alg_components) != len(
            self.occurrence_ids
        ):
            raise ValueError("round S_alg components do not match occurrences")
        if sum(value for _, value in self.round_s_unique_components) > len(
            self.occurrence_ids
        ):
            raise ValueError("round S_unique exceeds executed occurrences")
        if self.cumulative_s_alg < len(self.occurrence_ids):
            raise ValueError("cumulative S_alg cannot precede the round delta")
        if sum(value for _, value in self.cumulative_s_alg_components) != (
            self.cumulative_s_alg
        ):
            raise ValueError("cumulative S_alg components do not reconcile")
        if not 0 <= self.cumulative_s_unique <= self.cumulative_s_alg:
            raise ValueError("cumulative S_unique must lie within S_alg")
        if sum(value for _, value in self.cumulative_s_unique_components) != (
            self.cumulative_s_unique
        ):
            raise ValueError("cumulative S_unique components do not reconcile")
        if self.close_count != 1:
            raise ValueError("accepted round ledger must close exactly once")

    @property
    def closure_identity(self) -> str:
        return _sha256_json(
            {
                "controller_round": self.controller_round,
                "checkpoint_sequence": self.checkpoint_sequence,
                "prefix_identity_before": self.prefix_identity_before,
                "prefix_identity_after": self.prefix_identity_after,
                "sequence_start_exclusive": self.sequence_start_exclusive,
                "first_sequence_index": self.first_sequence_index,
                "sequence_indices": self.sequence_indices,
                "occurrence_ids": self.occurrence_ids,
                "reuse_identities": self.reuse_identities,
                "round_s_alg_components": self.round_s_alg_components,
                "round_s_unique_components": self.round_s_unique_components,
                "cumulative_s_alg": self.cumulative_s_alg,
                "cumulative_s_alg_components": (
                    self.cumulative_s_alg_components
                ),
                "cumulative_s_unique": self.cumulative_s_unique,
                "cumulative_s_unique_components": (
                    self.cumulative_s_unique_components
                ),
            }
        )

    def active_prefix_receipt(self) -> dict[str, object]:
        round_s_alg_components = dict(self.round_s_alg_components)
        round_s_unique_components = dict(self.round_s_unique_components)
        cumulative_s_alg_components = dict(
            self.cumulative_s_alg_components
        )
        cumulative_s_unique_components = dict(
            self.cumulative_s_unique_components
        )
        return {
            "schema": "paper_i_active_prefix_estimator_ledger_receipt_v2",
            "enabled": True,
            "status": "complete",
            "checkpoint_sequence": int(self.checkpoint_sequence),
            "occurrence_sequence_start_exclusive": int(
                self.sequence_start_exclusive
            ),
            "occurrence_sequence_end_inclusive": int(
                self.sequence_indices[-1]
                if self.sequence_indices
                else self.sequence_start_exclusive
            ),
            "raw_occurrence_delta": {
                "components": round_s_alg_components,
                "total": int(len(self.occurrence_ids)),
            },
            "executed_query_delta": {
                "components": round_s_alg_components,
                "S_alg": int(len(self.occurrence_ids)),
            },
            "unique_primitive_delta": {
                "components": round_s_unique_components,
                "S_unique": int(
                    sum(round_s_unique_components.values())
                ),
            },
            "cumulative_raw_occurrences": {
                "components": cumulative_s_alg_components,
                "total": int(self.cumulative_s_alg),
            },
            "cumulative_executed_queries": {
                "components": cumulative_s_alg_components,
                "S_alg": int(self.cumulative_s_alg),
                "unit": (
                    "executed_logical_scalar_estimator_invocation"
                ),
            },
            "cumulative_unique_primitives": {
                "components": cumulative_s_unique_components,
                "S_unique": int(self.cumulative_s_unique),
            },
            "runtime_estimator_occurrence_contract": (
                "all_instrumented_logical_scalar_estimator_calls_v1"
            ),
            "physical_identity_collapse_is_diagnostic_only": True,
            "raw_occurrences_preserved": True,
        }


@dataclass(frozen=True, slots=True)
class _TransitionOperationAudit:
    admission_calls: int
    supported_fs_chart_calls: int
    optimizer_dispatch_calls: int
    trust_update_calls: int
    ledger_close_calls: int
    checkpoint_event_count: int
    prune_nomination_calls: int = 0
    prune_verification_calls: int = 0

    @classmethod
    def from_operation_sequence(
        cls,
        operations: tuple[str, ...],
    ) -> _TransitionOperationAudit:
        no_prune_expected = (
            "admission",
            "supported_fs_chart",
            "optimizer_dispatch",
            "trust_update",
            "ledger_close",
            "checkpoint_event",
        )
        prune_expected = (
            "admission",
            "supported_fs_chart",
            "optimizer_dispatch",
            "trust_update",
            "prune_nomination",
            "prune_verification",
            "ledger_close",
            "checkpoint_event",
        )
        if operations not in {no_prune_expected, prune_expected}:
            raise ValueError(
                "default accepted transition operation order changed: "
                f"expected={no_prune_expected!r} or {prune_expected!r}, "
                f"observed={operations!r}"
            )
        return cls(
            admission_calls=operations.count("admission"),
            supported_fs_chart_calls=operations.count(
                "supported_fs_chart"
            ),
            optimizer_dispatch_calls=operations.count(
                "optimizer_dispatch"
            ),
            trust_update_calls=operations.count("trust_update"),
            ledger_close_calls=operations.count("ledger_close"),
            checkpoint_event_count=operations.count("checkpoint_event"),
            prune_nomination_calls=operations.count("prune_nomination"),
            prune_verification_calls=operations.count(
                "prune_verification"
            ),
        )

    def __post_init__(self) -> None:
        observed = (
            self.admission_calls,
            self.supported_fs_chart_calls,
            self.optimizer_dispatch_calls,
            self.trust_update_calls,
            self.prune_nomination_calls,
            self.prune_verification_calls,
            self.ledger_close_calls,
            self.checkpoint_event_count,
        )
        if observed not in {
            (1, 1, 1, 1, 0, 0, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1),
        }:
            raise ValueError(
                "default accepted transition operations must execute once"
            )


@dataclass(frozen=True, slots=True)
class _CheckpointReadyAcceptedStateEvent:
    controller_round: int
    accepted_state_fingerprint: str
    accepted_operator_ids: tuple[str, ...]
    accepted_insertion_positions: tuple[int, ...]
    logical_parameter_ids: tuple[str, ...]
    logical_parameter_values: tuple[float, ...]
    runtime_parameter_ids: tuple[str, ...]
    runtime_parameter_values: tuple[float, ...]
    accepted_energy: float
    trust_state_identity: str
    estimator_prefix_identity: str
    ledger_closure: _RoundLedgerClosure

    @property
    def ledger_closure_identity(self) -> str:
        return self.ledger_closure.closure_identity


@dataclass(frozen=True, slots=True)
class _RecoverabilityPruneReceipt:
    """Typed intermediate-to-final receipt for optional measured deletion."""

    status: str
    reason: str
    source_state_fingerprint: str
    pre_prune_operator_ids: tuple[str, ...]
    post_prune_operator_ids: tuple[str, ...]
    pre_prune_logical_parameter_count: int
    post_prune_logical_parameter_count: int
    pre_prune_runtime_parameter_count: int
    post_prune_runtime_parameter_count: int
    optimizer_memory_identity_before: str
    optimizer_memory_identity_after: str
    trust_radius_before: float
    trust_radius_after: float
    metric_damping: float
    endpoint_overlap_query_charge: int
    terminal_prune_active: bool
    nomination_index: int | None
    nomination_label: str | None
    predicted_energy_change: float | None
    surrogate_used_for_acceptance: bool | None
    trial_executed: bool
    trial_branch_id: str | None
    trial_classification: str | None
    trial_s_alg: int
    measured_energy_before: float | None
    measured_energy_after: float | None
    accepted: bool | None
    deleted_index: int | None
    deleted_label: str | None
    final_state_fingerprint: str
    policy_identity: str = "recoverability_ladder_v1"
    nomination_policy_identity: str = (
        "full_logical_fs_trust_delete_refit_v1"
    )

    def __post_init__(self) -> None:
        if self.status not in {"not_executed", "accepted", "rejected"}:
            raise ValueError("prune receipt status is invalid")
        if not self.reason:
            raise ValueError("prune receipt reason must be non-empty")
        if self.policy_identity != "recoverability_ladder_v1":
            raise ValueError("prune policy identity changed")
        if self.nomination_policy_identity not in {
            "full_logical_fs_trust_delete_refit_v1",
            "metric_regularized_v1",
        }:
            raise ValueError("prune nomination policy identity changed")
        if self.endpoint_overlap_query_charge != 0:
            raise ValueError("pruning must not charge endpoint overlap")
        if self.metric_damping != 0.0 or self.terminal_prune_active:
            raise ValueError("pruning damping and terminal prune remain off")
        if (
            not math.isfinite(self.trust_radius_before)
            or not math.isfinite(self.trust_radius_after)
            or self.trust_radius_before <= 0.0
            or self.trust_radius_after <= 0.0
        ):
            raise ValueError("prune trust radii must be finite and positive")
        if self.trial_s_alg < 0:
            raise ValueError("prune trial work must be non-negative")
        if (
            not self.source_state_fingerprint
            or not self.final_state_fingerprint
            or not self.optimizer_memory_identity_before
            or not self.optimizer_memory_identity_after
        ):
            raise ValueError("prune state and optimizer identities are required")
        if (
            len(self.pre_prune_operator_ids)
            != self.pre_prune_logical_parameter_count
            or len(self.post_prune_operator_ids)
            != self.post_prune_logical_parameter_count
            or self.pre_prune_runtime_parameter_count < 0
            or self.post_prune_runtime_parameter_count < 0
        ):
            raise ValueError(
                "prune operator and parameter cardinalities disagree"
            )
        if not self.trial_executed:
            if self.status != "not_executed":
                raise ValueError("no-trial prune receipt must be not_executed")
            trial_only = (
                self.nomination_index,
                self.nomination_label,
                self.predicted_energy_change,
                self.surrogate_used_for_acceptance,
                self.trial_branch_id,
                self.trial_classification,
                self.measured_energy_before,
                self.measured_energy_after,
                self.accepted,
                self.deleted_index,
                self.deleted_label,
            )
            if any(value is not None for value in trial_only):
                raise ValueError(
                    "no-trial prune receipt carries trial-only fields"
                )
            if self.trial_s_alg != 0:
                raise ValueError("no-trial prune receipt must report zero work")
            if (
                self.pre_prune_operator_ids
                != self.post_prune_operator_ids
                or self.pre_prune_logical_parameter_count
                != self.post_prune_logical_parameter_count
                or self.pre_prune_runtime_parameter_count
                != self.post_prune_runtime_parameter_count
                or self.optimizer_memory_identity_before
                != self.optimizer_memory_identity_after
                or self.source_state_fingerprint
                != self.final_state_fingerprint
            ):
                raise ValueError(
                    "no-trial prune receipt must preserve the source state"
                )
            if not math.isclose(
                self.trust_radius_after,
                self.trust_radius_before,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    "no-trial prune receipt must preserve trust radius"
                )
            return

        required = (
            self.nomination_index,
            self.nomination_label,
            self.predicted_energy_change,
            self.trial_branch_id,
            self.trial_classification,
            self.measured_energy_before,
            self.measured_energy_after,
            self.accepted,
        )
        if any(value is None for value in required):
            raise ValueError("measured prune trial receipt is incomplete")
        if self.surrogate_used_for_acceptance is not False:
            raise ValueError("surrogate evidence cannot authorize deletion")
        if self.nomination_index is None or self.nomination_index < 0:
            raise ValueError("measured prune nominee index must be non-negative")
        if not self.nomination_label or not self.trial_branch_id:
            raise ValueError(
                "measured prune nominee and branch identities are required"
            )
        if any(
            value is None or not math.isfinite(float(value))
            for value in (
                self.predicted_energy_change,
                self.measured_energy_before,
                self.measured_energy_after,
            )
        ):
            raise ValueError(
                "prune prediction and measured energies must be finite"
            )
        if self.accepted:
            if (
                self.status != "accepted"
                or self.trial_classification != "committed_prune"
            ):
                raise ValueError(
                    "accepted prune trial requires committed classification"
                )
            if (
                self.deleted_index is None
                or self.deleted_label is None
                or self.deleted_index != self.nomination_index
                or self.deleted_label != self.nomination_label
            ):
                raise ValueError(
                    "accepted prune trial requires nominated deletion identity"
                )
            if not (
                0 <= self.deleted_index < len(self.pre_prune_operator_ids)
                and self.pre_prune_operator_ids[self.deleted_index]
                == self.deleted_label
                and self.post_prune_operator_ids
                == (
                    self.pre_prune_operator_ids[: self.deleted_index]
                    + self.pre_prune_operator_ids[
                        self.deleted_index + 1 :
                    ]
                )
                and self.post_prune_logical_parameter_count
                == self.pre_prune_logical_parameter_count - 1
                and self.post_prune_runtime_parameter_count
                < self.pre_prune_runtime_parameter_count
            ):
                raise ValueError(
                    "accepted prune trial deletion remap is inconsistent"
                )
            if not math.isclose(
                self.trust_radius_after,
                self.trust_radius_before,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError(
                    "accepted prune trial must preserve trust radius"
                )
        else:
            if (
                self.status != "rejected"
                or self.trial_classification != "discarded_prune"
            ):
                raise ValueError(
                    "rejected prune trial requires discarded classification"
                )
            if self.deleted_index is not None or self.deleted_label is not None:
                raise ValueError(
                    "rejected prune trial cannot carry deletion identity"
                )
            if (
                self.pre_prune_operator_ids
                != self.post_prune_operator_ids
                or self.pre_prune_logical_parameter_count
                != self.post_prune_logical_parameter_count
                or self.pre_prune_runtime_parameter_count
                != self.post_prune_runtime_parameter_count
                or self.optimizer_memory_identity_before
                != self.optimizer_memory_identity_after
                or self.source_state_fingerprint
                != self.final_state_fingerprint
            ):
                raise ValueError(
                    "rejected prune trial must preserve the source state"
                )
            if self.trust_radius_after > self.trust_radius_before:
                raise ValueError(
                    "rejected prune trial cannot expand trust radius"
                )


@dataclass(frozen=True, slots=True)
class _TransitionEvaluation:
    """Validated numerical result supplied by the exact-route kernel."""

    next_state: _AcceptedStateSnapshot
    admission: _AdmissionReceipt
    refit: _SupportedFSRefitReceipt
    trust: _AdaptiveTrustUpdateReceipt
    non_worsening: _NonWorseningReceipt
    ledger: _RoundLedgerClosure
    checkpoint_event: _CheckpointReadyAcceptedStateEvent
    operation_audit: _TransitionOperationAudit
    pruning: _RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True, slots=True)
class _GreedyBatchTransitionEvaluation:
    """Validated result of one atomic greedy-batch numerical transaction."""

    next_state: _AcceptedStateSnapshot
    admission: _GreedyBatchAdmissionReceipt
    refit: _SupportedFSRefitReceipt
    trust: _AdaptiveTrustUpdateReceipt
    non_worsening: _NonWorseningReceipt
    ledger: _RoundLedgerClosure
    checkpoint_event: _CheckpointReadyAcceptedStateEvent
    operation_audit: _TransitionOperationAudit
    pruning: _RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True, slots=True)
class _CombinatorialBatchTransitionEvaluation:
    """Validated result of one atomic combinatorial-batch transaction."""

    next_state: _AcceptedStateSnapshot
    admission: _CombinatorialBatchAdmissionReceipt
    refit: _SupportedFSRefitReceipt
    trust: _AdaptiveTrustUpdateReceipt
    non_worsening: _NonWorseningReceipt
    ledger: _RoundLedgerClosure
    checkpoint_event: _CheckpointReadyAcceptedStateEvent
    operation_audit: _TransitionOperationAudit
    pruning: _RecoverabilityPruneReceipt | None = None


class _TransitionNumericalRuntime(Protocol):
    """Live arrays, executors, memories, and accepted-state cursor."""

    def accepted_state_snapshot(self) -> _AcceptedStateSnapshot: ...


class _TransitionKernel(Protocol):
    def execute(
        self,
        decision: (
            _SingletonAdmissionDecision
            | _GreedyBatchAdmissionDecision
            | _CombinatorialBatchAdmissionDecision
        ),
        live_record: object,
        runtime: _TransitionNumericalRuntime,
    ) -> (
        _TransitionEvaluation
        | _GreedyBatchTransitionEvaluation
        | _CombinatorialBatchTransitionEvaluation
    ): ...


@dataclass(frozen=True, slots=True)
class _TransitionWorkspace:
    """Cohesive live workspace; arrays and executors stay behind the kernel."""

    runtime_sidecar: Mapping[str, object]
    numerical_runtime: _TransitionNumericalRuntime
    kernel: _TransitionKernel


@dataclass(frozen=True, slots=True)
class _AcceptedSingletonTransition:
    preceding_state: _AcceptedStateSnapshot
    decision: _SingletonAdmissionDecision
    next_state: _AcceptedStateSnapshot
    admission: _AdmissionReceipt
    refit: _SupportedFSRefitReceipt
    trust: _AdaptiveTrustUpdateReceipt
    non_worsening: _NonWorseningReceipt
    ledger: _RoundLedgerClosure
    checkpoint_event: _CheckpointReadyAcceptedStateEvent
    operation_audit: _TransitionOperationAudit
    pruning: _RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True, slots=True)
class _AcceptedGreedyBatchTransition:
    preceding_state: _AcceptedStateSnapshot
    decision: _GreedyBatchAdmissionDecision
    next_state: _AcceptedStateSnapshot
    admission: _GreedyBatchAdmissionReceipt
    refit: _SupportedFSRefitReceipt
    trust: _AdaptiveTrustUpdateReceipt
    non_worsening: _NonWorseningReceipt
    ledger: _RoundLedgerClosure
    checkpoint_event: _CheckpointReadyAcceptedStateEvent
    operation_audit: _TransitionOperationAudit
    pruning: _RecoverabilityPruneReceipt | None = None


@dataclass(frozen=True, slots=True)
class _AcceptedCombinatorialBatchTransition:
    preceding_state: _AcceptedStateSnapshot
    decision: _CombinatorialBatchAdmissionDecision
    next_state: _AcceptedStateSnapshot
    admission: _CombinatorialBatchAdmissionReceipt
    refit: _SupportedFSRefitReceipt
    trust: _AdaptiveTrustUpdateReceipt
    non_worsening: _NonWorseningReceipt
    ledger: _RoundLedgerClosure
    checkpoint_event: _CheckpointReadyAcceptedStateEvent
    operation_audit: _TransitionOperationAudit
    pruning: _RecoverabilityPruneReceipt | None = None


def _selection_count(
    state: _AcceptedStateSnapshot,
    generator_id: str,
) -> int:
    return dict(state.selection_counts).get(generator_id, 0)


def _accepted_positions_after_insertion(
    positions: tuple[int, ...],
    insertion_position: int,
) -> tuple[int, ...]:
    remapped = [
        int(position)
        if int(position) < int(insertion_position)
        else int(position) + 1
        for position in positions
    ]
    remapped.insert(int(insertion_position), int(insertion_position))
    return tuple(remapped)


def _assert_checkpoint_event(
    event: _CheckpointReadyAcceptedStateEvent,
    state: _AcceptedStateSnapshot,
    ledger: _RoundLedgerClosure,
) -> None:
    expected = (
        state.controller_round,
        state.accepted_state_fingerprint,
        state.accepted_operator_ids,
        state.accepted_insertion_positions,
        state.logical_parameter_ids,
        state.logical_parameter_values,
        state.runtime_parameter_ids,
        state.runtime_parameter_values,
        state.accepted_energy,
        state.trust_state_identity,
        state.estimator_prefix_identity,
        ledger.closure_identity,
    )
    observed = (
        event.controller_round,
        event.accepted_state_fingerprint,
        event.accepted_operator_ids,
        event.accepted_insertion_positions,
        event.logical_parameter_ids,
        event.logical_parameter_values,
        event.runtime_parameter_ids,
        event.runtime_parameter_values,
        event.accepted_energy,
        event.trust_state_identity,
        event.estimator_prefix_identity,
        event.ledger_closure_identity,
    )
    if observed != expected:
        raise RuntimeError(
            "checkpoint-ready event does not identify the next accepted state"
        )


def _transition_singleton(
    preceding_state: _AcceptedStateSnapshot,
    decision: _SingletonAdmissionDecision,
    workspace: _TransitionWorkspace,
) -> _AcceptedSingletonTransition:
    """Execute and validate one accepted singleton transition."""

    if decision.controller_round != preceding_state.controller_round:
        raise ValueError("decision and preceding controller rounds disagree")
    if (
        decision.controller_state_fingerprint
        != preceding_state.accepted_state_fingerprint
    ):
        raise ValueError("decision identifies a different preceding state")

    selected_record_id = decision.selected.domain_record_id
    sidecar_keys = tuple(workspace.runtime_sidecar)
    if sidecar_keys != (selected_record_id,):
        raise ValueError(
            "sole runtime sidecar key must equal the decision record"
        )
    live_record = workspace.runtime_sidecar[selected_record_id]

    live_before = workspace.numerical_runtime.accepted_state_snapshot()
    if live_before != preceding_state:
        differing_fields = tuple(
            item.name
            for item in fields(_AcceptedStateSnapshot)
            if getattr(live_before, item.name)
            != getattr(preceding_state, item.name)
        )
        raise RuntimeError(
            "live accepted state disagrees with the preceding snapshot; "
            f"differing_fields={differing_fields!r}"
        )
    evaluation = workspace.kernel.execute(
        decision,
        live_record,
        workspace.numerical_runtime,
    )
    live_after = workspace.numerical_runtime.accepted_state_snapshot()
    if live_after != evaluation.next_state:
        raise RuntimeError(
            "live accepted state disagrees with the returned next snapshot"
        )

    selected = decision.selected
    admission = evaluation.admission
    if (
        admission.selected_domain_record_id != selected.domain_record_id
        or admission.generator_id != selected.generator_id
        or admission.pool_index != selected.pool_index
        or admission.insertion_position != selected.insertion_position
    ):
        raise RuntimeError(
            "transition admitted a record other than the immutable decision"
        )
    if admission.source_identity != selected.lineage_identity[0]:
        raise RuntimeError("admission changed the source generator identity")
    if admission.child_identity != selected.generator_id:
        raise RuntimeError("admission changed the selected child identity")
    if admission.initial_logical_value != 0.0:
        raise RuntimeError("default singleton must be admitted at zero amplitude")
    if admission.logical_parameter_count_before != len(
        preceding_state.logical_parameter_ids
    ):
        raise RuntimeError("admission pre-count disagrees with preceding state")
    pruning = evaluation.pruning
    expected_post_admission_logical_count = (
        len(evaluation.next_state.logical_parameter_ids)
        if pruning is None
        else pruning.pre_prune_logical_parameter_count
    )
    if (
        admission.logical_parameter_count_after
        != expected_post_admission_logical_count
    ):
        raise RuntimeError(
            "admission post-count disagrees with the post-admission state"
        )
    if admission.runtime_parameter_count_before != len(
        preceding_state.runtime_parameter_ids
    ):
        raise RuntimeError(
            "runtime admission pre-count disagrees with preceding state"
        )
    expected_post_admission_runtime_count = (
        len(evaluation.next_state.runtime_parameter_ids)
        if pruning is None
        else pruning.pre_prune_runtime_parameter_count
    )
    if (
        admission.runtime_parameter_count_after
        != expected_post_admission_runtime_count
    ):
        raise RuntimeError(
            "runtime admission post-count disagrees with the post-admission state"
        )
    if admission.optimizer_memory_identity_before != (
        preceding_state.optimizer_memory_identity
    ):
        raise RuntimeError("admission optimizer source identity changed")
    expected_post_admission_optimizer_identity = (
        evaluation.next_state.optimizer_memory_identity
        if pruning is None
        else pruning.optimizer_memory_identity_before
    )
    if (
        admission.optimizer_memory_identity_after
        != expected_post_admission_optimizer_identity
    ):
        raise RuntimeError("admission optimizer destination identity changed")
    if admission.selection_count_before != _selection_count(
        preceding_state, admission.source_identity
    ):
        raise RuntimeError("admission selection count source changed")
    if admission.selection_count_after != _selection_count(
        evaluation.next_state, admission.source_identity
    ):
        raise RuntimeError("admission selection count destination changed")
    if admission.available_before != (
        admission.source_identity in preceding_state.available_generator_ids
    ):
        raise RuntimeError("admission availability source changed")
    if admission.available_after != (
        admission.source_identity in evaluation.next_state.available_generator_ids
    ):
        raise RuntimeError("admission availability destination changed")

    next_state = evaluation.next_state
    if next_state.controller_round != preceding_state.controller_round + 1:
        raise RuntimeError("accepted transition must advance one controller round")
    expected_operator_ids = list(preceding_state.accepted_operator_ids)
    expected_operator_ids.insert(
        selected.insertion_position,
        admission.child_identity,
    )
    post_admission_operator_ids = (
        next_state.accepted_operator_ids
        if pruning is None
        else pruning.pre_prune_operator_ids
    )
    if post_admission_operator_ids != tuple(expected_operator_ids):
        raise RuntimeError(
            "post-admission operators do not reflect the authorized insertion"
        )
    expected_positions = _accepted_positions_after_insertion(
        preceding_state.accepted_insertion_positions,
        selected.insertion_position,
    )
    if pruning is None and (
        next_state.accepted_insertion_positions != expected_positions
    ):
        raise RuntimeError(
            "next insertion-position identities changed outside admission"
        )
    if evaluation.refit.chart_dimension != len(
        (
            next_state.logical_parameter_ids
            if pruning is None
            else tuple(
                range(pruning.pre_prune_logical_parameter_count)
            )
        )
    ):
        raise RuntimeError(
            "supported-FS chart does not cover the post-admission ansatz"
        )
    if pruning is not None:
        if pruning.post_prune_operator_ids != next_state.accepted_operator_ids:
            raise RuntimeError(
                "prune receipt final operators differ from the next state"
            )
        if pruning.post_prune_logical_parameter_count != len(
            next_state.logical_parameter_ids
        ) or pruning.post_prune_runtime_parameter_count != len(
            next_state.runtime_parameter_ids
        ):
            raise RuntimeError(
                "prune receipt final parameter counts differ from next state"
            )
        if pruning.optimizer_memory_identity_after != (
            next_state.optimizer_memory_identity
        ):
            raise RuntimeError(
                "prune receipt optimizer destination differs from next state"
            )
        if pruning.accepted:
            if pruning.deleted_index is None or pruning.deleted_label is None:
                raise RuntimeError("accepted prune lacks deletion identity")
            expected_pruned_ids = list(pruning.pre_prune_operator_ids)
            if not 0 <= pruning.deleted_index < len(expected_pruned_ids):
                raise RuntimeError("accepted prune deletion index is invalid")
            if (
                expected_pruned_ids[pruning.deleted_index]
                != pruning.deleted_label
            ):
                raise RuntimeError("accepted prune deletion label changed")
            del expected_pruned_ids[pruning.deleted_index]
            if tuple(expected_pruned_ids) != pruning.post_prune_operator_ids:
                raise RuntimeError(
                    "accepted prune did not return the measured reduced ansatz"
                )
        elif (
            pruning.pre_prune_operator_ids
            != pruning.post_prune_operator_ids
        ):
            raise RuntimeError("rejected/no-nominee prune mutated keep state")
    if evaluation.trust.trust_state_identity_before != (
        preceding_state.trust_state_identity
    ):
        raise RuntimeError("adaptive trust receipt changed its source state")
    if evaluation.trust.trust_state_identity_after != (
        next_state.trust_state_identity
    ):
        raise RuntimeError("adaptive trust receipt changed its destination state")
    if evaluation.non_worsening.energy_before != (
        preceding_state.accepted_energy
    ):
        raise RuntimeError("non-worsening receipt changed preceding energy")
    if evaluation.non_worsening.energy_after != next_state.accepted_energy:
        raise RuntimeError("non-worsening receipt changed accepted energy")
    if evaluation.ledger.controller_round != preceding_state.controller_round:
        raise RuntimeError("ledger closure identifies a different round")
    if evaluation.ledger.prefix_identity_before != (
        preceding_state.estimator_prefix_identity
    ):
        raise RuntimeError("ledger closure changed its source prefix")
    if evaluation.ledger.prefix_identity_after != (
        next_state.estimator_prefix_identity
    ):
        raise RuntimeError("ledger closure changed its destination prefix")
    _assert_checkpoint_event(
        evaluation.checkpoint_event,
        next_state,
        evaluation.ledger,
    )

    return _AcceptedSingletonTransition(
        preceding_state=preceding_state,
        decision=decision,
        next_state=next_state,
        admission=admission,
        refit=evaluation.refit,
        trust=evaluation.trust,
        non_worsening=evaluation.non_worsening,
        ledger=evaluation.ledger,
        checkpoint_event=evaluation.checkpoint_event,
        operation_audit=evaluation.operation_audit,
        pruning=evaluation.pruning,
    )


def _transition_ordered_batch(
    preceding_state: _AcceptedStateSnapshot,
    decision: (
        _GreedyBatchAdmissionDecision
        | _CombinatorialBatchAdmissionDecision
    ),
    workspace: _TransitionWorkspace,
    *,
    strategy_label: str,
    evaluation_type: type[
        _GreedyBatchTransitionEvaluation
        | _CombinatorialBatchTransitionEvaluation
    ],
    accepted_type: type[
        _AcceptedGreedyBatchTransition
        | _AcceptedCombinatorialBatchTransition
    ],
) -> _AcceptedGreedyBatchTransition | _AcceptedCombinatorialBatchTransition:
    """Execute and validate one fixed-order batch as one accepted round."""

    if decision.controller_round != preceding_state.controller_round:
        raise ValueError("decision and preceding controller rounds disagree")
    if (
        decision.controller_state_fingerprint
        != preceding_state.accepted_state_fingerprint
    ):
        raise ValueError("decision identifies a different preceding state")
    selected_record_ids = tuple(
        record.domain_record_id for record in decision.selected
    )
    if tuple(workspace.runtime_sidecar) != selected_record_ids:
        raise ValueError(
            "runtime sidecar keys must equal the ordered "
            f"{strategy_label} decision"
        )
    live_records = tuple(
        workspace.runtime_sidecar[record_id]
        for record_id in selected_record_ids
    )

    live_before = workspace.numerical_runtime.accepted_state_snapshot()
    if live_before != preceding_state:
        differing_fields = tuple(
            item.name
            for item in fields(_AcceptedStateSnapshot)
            if getattr(live_before, item.name)
            != getattr(preceding_state, item.name)
        )
        raise RuntimeError(
            "live accepted state disagrees with the preceding snapshot; "
            f"differing_fields={differing_fields!r}"
        )
    evaluation = workspace.kernel.execute(
        decision,
        live_records,
        workspace.numerical_runtime,
    )
    if not isinstance(evaluation, evaluation_type):
        raise TypeError(
            f"{strategy_label} batch kernel returned the wrong transition "
            "evaluation"
        )
    live_after = workspace.numerical_runtime.accepted_state_snapshot()
    if live_after != evaluation.next_state:
        raise RuntimeError(
            "live accepted state disagrees with the returned next snapshot"
        )

    admission = evaluation.admission
    selected = decision.selected
    expected_member_fields = (
        selected_record_ids,
        tuple(record.generator_id for record in selected),
        tuple(record.pool_index for record in selected),
        tuple(record.insertion_position for record in selected),
        tuple(record.lineage_identity[0] for record in selected),
        tuple(record.generator_id for record in selected),
    )
    observed_member_fields = (
        admission.selected_domain_record_ids,
        admission.generator_ids,
        admission.pool_indices,
        admission.original_insertion_positions,
        admission.source_identities,
        admission.child_identities,
    )
    if observed_member_fields != expected_member_fields:
        raise RuntimeError(
            "transition admitted records other than the immutable "
            f"ordered {strategy_label} decision"
        )
    pruning = evaluation.pruning
    expected_post_admission_logical_count = (
        len(evaluation.next_state.logical_parameter_ids)
        if pruning is None
        else pruning.pre_prune_logical_parameter_count
    )
    if admission.logical_parameter_count_before != len(
        preceding_state.logical_parameter_ids
    ) or admission.logical_parameter_count_after != (
        expected_post_admission_logical_count
    ):
        raise RuntimeError(
            f"{strategy_label} admission logical counts disagree with "
            "accepted states"
        )
    expected_post_admission_runtime_count = (
        len(evaluation.next_state.runtime_parameter_ids)
        if pruning is None
        else pruning.pre_prune_runtime_parameter_count
    )
    if admission.runtime_parameter_count_before != len(
        preceding_state.runtime_parameter_ids
    ) or admission.runtime_parameter_count_after != (
        expected_post_admission_runtime_count
    ):
        raise RuntimeError(
            f"{strategy_label} admission runtime counts disagree with "
            "accepted states"
        )
    expected_post_admission_optimizer_identity = (
        evaluation.next_state.optimizer_memory_identity
        if pruning is None
        else pruning.optimizer_memory_identity_before
    )
    if admission.optimizer_memory_identity_before != (
        preceding_state.optimizer_memory_identity
    ) or admission.optimizer_memory_identity_after != (
        expected_post_admission_optimizer_identity
    ):
        raise RuntimeError(
            f"{strategy_label} admission optimizer identity changed outside "
            "the "
            "atomic transaction"
        )
    for member_index, source_identity in enumerate(
        admission.source_identities
    ):
        if admission.selection_counts_before[member_index] != (
            _selection_count(preceding_state, source_identity)
        ):
            raise RuntimeError(
                f"{strategy_label} admission selection-count source changed"
            )
        if admission.selection_counts_after[member_index] != (
            _selection_count(evaluation.next_state, source_identity)
        ):
            raise RuntimeError(
                f"{strategy_label} admission selection-count destination "
                "changed"
            )
        if admission.available_before[member_index] != (
            source_identity in preceding_state.available_generator_ids
        ):
            raise RuntimeError(
                f"{strategy_label} admission availability source changed"
            )
        if admission.available_after[member_index] != (
            source_identity in evaluation.next_state.available_generator_ids
        ):
            raise RuntimeError(
                f"{strategy_label} admission availability destination "
                "changed"
            )

    next_state = evaluation.next_state
    if next_state.controller_round != preceding_state.controller_round + 1:
        raise RuntimeError(
            f"accepted {strategy_label} batch must advance one controller "
            "round"
        )
    expected_operator_ids = list(preceding_state.accepted_operator_ids)
    for child_identity, effective_position in zip(
        admission.child_identities,
        admission.effective_insertion_positions,
        strict=True,
    ):
        expected_operator_ids.insert(effective_position, child_identity)
    post_admission_operator_ids = (
        next_state.accepted_operator_ids
        if pruning is None
        else pruning.pre_prune_operator_ids
    )
    if post_admission_operator_ids != tuple(expected_operator_ids):
        raise RuntimeError(
            "next operators do not reflect the authorized ordered batch"
        )
    if next_state.accepted_insertion_positions != tuple(
        range(len(next_state.accepted_operator_ids))
    ):
        raise RuntimeError(
            "next insertion-position identities changed outside admission"
        )
    if evaluation.refit.chart_dimension != (
        len(next_state.logical_parameter_ids)
        if pruning is None
        else pruning.pre_prune_logical_parameter_count
    ):
        raise RuntimeError(
            "supported-FS chart does not cover the post-batch ansatz"
        )
    if pruning is not None:
        if pruning.post_prune_operator_ids != next_state.accepted_operator_ids:
            raise RuntimeError(
                "batch prune final operators differ from the next state"
            )
        if pruning.post_prune_logical_parameter_count != len(
            next_state.logical_parameter_ids
        ) or pruning.post_prune_runtime_parameter_count != len(
            next_state.runtime_parameter_ids
        ):
            raise RuntimeError(
                "batch prune final parameter counts differ from next state"
            )
        if pruning.optimizer_memory_identity_after != (
            next_state.optimizer_memory_identity
        ):
            raise RuntimeError(
                "batch prune optimizer destination differs from next state"
            )
        if pruning.accepted:
            if pruning.deleted_index is None or pruning.deleted_label is None:
                raise RuntimeError(
                    "accepted batch prune lacks deletion identity"
                )
            expected_pruned_ids = list(pruning.pre_prune_operator_ids)
            if not 0 <= pruning.deleted_index < len(expected_pruned_ids):
                raise RuntimeError(
                    "accepted batch prune deletion index is invalid"
                )
            if (
                expected_pruned_ids[pruning.deleted_index]
                != pruning.deleted_label
            ):
                raise RuntimeError(
                    "accepted batch prune deletion label changed"
                )
            del expected_pruned_ids[pruning.deleted_index]
            if tuple(expected_pruned_ids) != pruning.post_prune_operator_ids:
                raise RuntimeError(
                    "accepted batch prune did not return the measured reduced "
                    "ansatz"
                )
        elif (
            pruning.pre_prune_operator_ids
            != pruning.post_prune_operator_ids
        ):
            raise RuntimeError(
                "rejected/no-nominee batch prune mutated keep state"
            )
    if evaluation.trust.trust_state_identity_before != (
        preceding_state.trust_state_identity
    ) or evaluation.trust.trust_state_identity_after != (
        next_state.trust_state_identity
    ):
        raise RuntimeError(
            "adaptive trust receipt changed the accepted-state boundary"
        )
    if evaluation.non_worsening.energy_before != (
        preceding_state.accepted_energy
    ) or evaluation.non_worsening.energy_after != next_state.accepted_energy:
        raise RuntimeError(
            "non-worsening receipt changed the accepted energies"
        )
    if evaluation.ledger.controller_round != preceding_state.controller_round:
        raise RuntimeError("ledger closure identifies a different round")
    if evaluation.ledger.prefix_identity_before != (
        preceding_state.estimator_prefix_identity
    ) or evaluation.ledger.prefix_identity_after != (
        next_state.estimator_prefix_identity
    ):
        raise RuntimeError(
            "ledger closure changed the accepted estimator prefix"
        )
    _assert_checkpoint_event(
        evaluation.checkpoint_event,
        next_state,
        evaluation.ledger,
    )

    return accepted_type(
        preceding_state=preceding_state,
        decision=decision,
        next_state=next_state,
        admission=admission,
        refit=evaluation.refit,
        trust=evaluation.trust,
        non_worsening=evaluation.non_worsening,
        ledger=evaluation.ledger,
        checkpoint_event=evaluation.checkpoint_event,
        operation_audit=evaluation.operation_audit,
        pruning=evaluation.pruning,
    )


def _transition_greedy_batch(
    preceding_state: _AcceptedStateSnapshot,
    decision: _GreedyBatchAdmissionDecision,
    workspace: _TransitionWorkspace,
) -> _AcceptedGreedyBatchTransition:
    """Execute and validate one ordered greedy batch as one accepted round."""

    transition = _transition_ordered_batch(
        preceding_state,
        decision,
        workspace,
        strategy_label="greedy",
        evaluation_type=_GreedyBatchTransitionEvaluation,
        accepted_type=_AcceptedGreedyBatchTransition,
    )
    if not isinstance(transition, _AcceptedGreedyBatchTransition):
        raise RuntimeError("greedy transition returned the wrong type")
    return transition


def _transition_combinatorial_batch(
    preceding_state: _AcceptedStateSnapshot,
    decision: _CombinatorialBatchAdmissionDecision,
    workspace: _TransitionWorkspace,
) -> _AcceptedCombinatorialBatchTransition:
    """Execute one fixed-record combinatorial subset as one accepted round."""

    transition = _transition_ordered_batch(
        preceding_state,
        decision,
        workspace,
        strategy_label="combinatorial",
        evaluation_type=_CombinatorialBatchTransitionEvaluation,
        accepted_type=_AcceptedCombinatorialBatchTransition,
    )
    if not isinstance(transition, _AcceptedCombinatorialBatchTransition):
        raise RuntimeError("combinatorial transition returned the wrong type")
    return transition
