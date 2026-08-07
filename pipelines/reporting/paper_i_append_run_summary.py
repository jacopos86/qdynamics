"""Pure, typed summary projection for conventional Paper-I Append-ADAPT.

The Append executor already owns every estimator call, controller transition,
and Qiskit compilation used by this summary.  This module only validates and
projects the completed result payload; it must never acquire new scientific
data or compile another circuit.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping

from pipelines.static_adapt.ra_adapt.contracts import (
    APPEND_ADAPT_PROTOCOL_SCHEMA,
    APPEND_CONVENTIONAL_SELECTOR_ID,
    APPEND_CONVENTIONAL_SELECTOR_SCOPE,
    CanonicalContract,
    ResolvedRAAdaptProtocol,
    canonical_json_bytes,
    canonical_sha256,
)


PAPER_I_APPEND_RUN_SUMMARY_SCHEMA = "paper_i_append_run_summary_v1"
PAPER_I_APPEND_ACCEPTED_ROUND_SCHEMA = (
    "paper_i_append_accepted_round_summary_v1"
)
PAPER_I_APPEND_ACCOUNTING_SUMMARY_SCHEMA = (
    "paper_i_append_accounting_summary_v1"
)
PAPER_I_APPEND_RESOURCE_SUMMARY_SCHEMA = (
    "paper_i_append_resource_summary_v1"
)
PAPER_I_APPEND_RESOURCE_ROW_SCHEMA = (
    "paper_i_append_resource_row_summary_v1"
)
PAPER_I_APPEND_SUMMARY_DERIVATION = (
    "pure_projection_from_completed_append_result_payload_v1"
)
_ACCOUNTING_COMPONENTS = (
    "N_H_outer",
    "N_H_refit",
    "N_grad",
    "N_metric",
)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _mapping_copy(value: Any, *, name: str) -> dict[str, Any]:
    payload = _mapping(value, name=name)
    return json.loads(canonical_json_bytes(dict(payload)).decode("utf-8"))


def _text(value: Any, *, name: str) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _finite(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _sequence(value: Any, *, name: str) -> tuple[Any, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a sequence.")
    return tuple(value)


def _compiled_observation_status(resources: Mapping[str, Any]) -> str:
    status = str(resources.get("compiled_circuit_stats_status", "")).strip()
    return status or "unreported"


@dataclass(frozen=True)
class PaperIAppendAcceptedRoundSummary(CanonicalContract):
    schema: str
    controller_round: int
    selected_label: str
    selected_generator_identity: str
    insertion_position: int
    energy_before: float
    energy_after: float
    selected_abs_commutator_gradient: float

    def __post_init__(self) -> None:
        if self.schema != PAPER_I_APPEND_ACCEPTED_ROUND_SCHEMA:
            raise ValueError("Unknown Append accepted-round summary schema.")
        if self.controller_round < 1:
            raise ValueError("Append accepted-round ordinal must be positive.")
        if not self.selected_label or not self.selected_generator_identity:
            raise ValueError("Append accepted-round identities are required.")
        if self.insertion_position < 0:
            raise ValueError("Append insertion position cannot be negative.")
        for name in (
            "energy_before",
            "energy_after",
            "selected_abs_commutator_gradient",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"Append accepted-round {name} must be finite.")


@dataclass(frozen=True)
class PaperIAppendAccountingSummary(CanonicalContract):
    schema: str
    convention: str
    N_H_outer: int
    N_H_refit: int
    N_grad: int
    N_metric: int
    S_alg: int
    closed_occurrence_reconciliation: bool
    closed_occurrence_prefix_S_alg: int
    ledger_schema: str
    ledger_fingerprint: str
    ledger_sha256: str
    ledger_occurrence_count: int

    def __post_init__(self) -> None:
        if self.schema != PAPER_I_APPEND_ACCOUNTING_SUMMARY_SCHEMA:
            raise ValueError("Unknown Append accounting-summary schema.")
        if not self.convention:
            raise ValueError("Append accounting convention is required.")
        components = (
            self.N_H_outer,
            self.N_H_refit,
            self.N_grad,
            self.N_metric,
        )
        if any(value < 0 for value in components):
            raise ValueError("Append accounting components cannot be negative.")
        if self.S_alg != sum(components):
            raise ValueError("Append summary S_alg does not close by component.")
        if not self.closed_occurrence_reconciliation:
            raise ValueError("Append summary requires closed occurrence accounting.")
        if (
            self.closed_occurrence_prefix_S_alg != self.S_alg
            or self.ledger_occurrence_count != self.S_alg
        ):
            raise ValueError("Append ledger occurrence prefix does not close.")
        if self.ledger_schema != "estimator_call_ledger_v1":
            raise ValueError("Append summary received a foreign ledger schema.")
        for name in ("ledger_fingerprint", "ledger_sha256"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Append {name} must be a SHA-256 digest.")


@dataclass(frozen=True)
class PaperIAppendCompiledResourceRow(CanonicalContract):
    schema: str
    controller_round: int
    accepted_prefix_length: int
    observation_status: str
    compiled_resources: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.schema != PAPER_I_APPEND_RESOURCE_ROW_SCHEMA:
            raise ValueError("Unknown Append resource-row summary schema.")
        if self.controller_round < 1 or self.accepted_prefix_length < 1:
            raise ValueError("Append resource-row ordinals must be positive.")
        if not self.observation_status:
            raise ValueError("Append resource-row observation status is required.")
        if not isinstance(self.compiled_resources, Mapping):
            raise TypeError("Append compiled resources must be a mapping.")


@dataclass(frozen=True)
class PaperIAppendResourceSummary(CanonicalContract):
    schema: str
    terminal_observation_status: str
    terminal_compiled_resources: Mapping[str, Any]
    accepted_prefix_observation_status: str
    requested_controller_rounds: tuple[int, ...]
    materialized_controller_rounds: tuple[int, ...]
    unmaterialized_controller_rounds: tuple[int, ...]
    compiled_resources_by_round: tuple[
        PaperIAppendCompiledResourceRow, ...
    ]

    def __post_init__(self) -> None:
        if self.schema != PAPER_I_APPEND_RESOURCE_SUMMARY_SCHEMA:
            raise ValueError("Unknown Append resource-summary schema.")
        if not self.terminal_observation_status:
            raise ValueError("Append terminal resource status is required.")
        if not isinstance(self.terminal_compiled_resources, Mapping):
            raise TypeError("Append terminal resources must be a mapping.")
        if self.accepted_prefix_observation_status not in {
            "complete",
            "not_requested",
            "partial",
        }:
            raise ValueError("Unknown Append accepted-prefix resource status.")
        requested = set(self.requested_controller_rounds)
        materialized = set(self.materialized_controller_rounds)
        unmaterialized = set(self.unmaterialized_controller_rounds)
        if (
            tuple(sorted(requested)) != self.requested_controller_rounds
            or tuple(sorted(materialized))
            != self.materialized_controller_rounds
            or tuple(sorted(unmaterialized))
            != self.unmaterialized_controller_rounds
            or materialized.intersection(unmaterialized)
            or materialized.union(unmaterialized) != requested
            or tuple(
                row.controller_round
                for row in self.compiled_resources_by_round
            )
            != self.materialized_controller_rounds
            or any(
                row.accepted_prefix_length != row.controller_round
                for row in self.compiled_resources_by_round
            )
        ):
            raise ValueError("Append resource observation rounds do not close.")
        expected_status = (
            "not_requested"
            if not requested
            else "complete"
            if not unmaterialized
            else "partial"
        )
        if self.accepted_prefix_observation_status != expected_status:
            raise ValueError("Append resource observation status is inconsistent.")


@dataclass(frozen=True)
class PaperIAppendRunSummary(CanonicalContract):
    schema: str
    source_result_payload_sha256: str
    protocol_schema: str
    protocol_sha256: str
    bundle_id: str
    bundle_manifest_sha256: str
    algorithm_id: str
    candidate_representation: str
    adapter_id: str
    active_gradient_policy: str
    resource_weighting_scope: str
    optimizer: str
    optimizer_maxiter: int
    protocol_horizon: int
    stopping_rule: Mapping[str, Any]
    seeds: Mapping[str, int]
    selector_identity: str
    selector_scope: str
    selector_source_id: str
    selection_with_replacement: bool
    append_position_only: bool
    compile_identity: Mapping[str, Any]
    accepted_refit_scope: str
    accepted_refit_coordinate_chart: str
    controller_rounds_completed: int
    stop_reason: str
    final_energy: float
    accepted_operator_labels: tuple[str, ...]
    accepted_generator_identities: tuple[str, ...]
    accepted_history: tuple[PaperIAppendAcceptedRoundSummary, ...]
    estimator_accounting: PaperIAppendAccountingSummary
    resources: PaperIAppendResourceSummary
    derivation_policy: str
    additional_estimator_acquisitions: int
    additional_controller_rounds: int

    @property
    def available_controller_rounds(self) -> int:
        return self.controller_rounds_completed

    def __post_init__(self) -> None:
        if self.schema != PAPER_I_APPEND_RUN_SUMMARY_SCHEMA:
            raise ValueError("Unknown Paper-I Append run-summary schema.")
        for name in (
            "source_result_payload_sha256",
            "protocol_sha256",
            "bundle_manifest_sha256",
        ):
            value = str(getattr(self, name))
            if len(value) != 64 or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"Append summary {name} must be SHA-256.")
        if self.protocol_schema != APPEND_ADAPT_PROTOCOL_SCHEMA:
            raise ValueError("Append summary protocol schema drifted.")
        if self.selector_identity != APPEND_CONVENTIONAL_SELECTOR_ID:
            raise ValueError("Append summary selector identity drifted.")
        if self.selector_scope != APPEND_CONVENTIONAL_SELECTOR_SCOPE:
            raise ValueError("Append summary selector scope drifted.")
        if (
            not self.selection_with_replacement
            or not self.append_position_only
        ):
            raise ValueError("Append summary selector semantics drifted.")
        if self.optimizer_maxiter < 1 or self.protocol_horizon < 1:
            raise ValueError("Append summary protocol bounds must be positive.")
        if not all(
            (
                self.algorithm_id,
                self.candidate_representation,
                self.adapter_id,
                self.active_gradient_policy,
                self.resource_weighting_scope,
                self.optimizer,
                self.selector_source_id,
                self.accepted_refit_scope,
                self.accepted_refit_coordinate_chart,
                self.stop_reason,
            )
        ):
            raise ValueError("Append summary provenance fields are required.")
        for name in (
            "compile_identity",
            "stopping_rule",
            "seeds",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"Append summary {name} must be a mapping.")
        if self.controller_rounds_completed < 0:
            raise ValueError("Append summary round count cannot be negative.")
        if (
            len(self.accepted_history) != self.controller_rounds_completed
            or len(self.accepted_operator_labels)
            != self.controller_rounds_completed
            or len(self.accepted_generator_identities)
            != self.controller_rounds_completed
        ):
            raise ValueError("Append summary accepted history is incomplete.")
        if tuple(
            row.controller_round for row in self.accepted_history
        ) != tuple(range(1, self.controller_rounds_completed + 1)):
            raise ValueError("Append summary controller rounds are not contiguous.")
        if tuple(
            row.selected_label for row in self.accepted_history
        ) != self.accepted_operator_labels:
            raise ValueError("Append summary accepted labels disagree.")
        if tuple(
            row.selected_generator_identity for row in self.accepted_history
        ) != self.accepted_generator_identities:
            raise ValueError("Append summary generator identities disagree.")
        if tuple(
            row.insertion_position for row in self.accepted_history
        ) != tuple(range(self.controller_rounds_completed)):
            raise ValueError("Append summary insertion positions are not append-only.")
        if self.accepted_history and not math.isclose(
            self.accepted_history[-1].energy_after,
            self.final_energy,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Append summary final energy disagrees with history.")
        if (
            self.derivation_policy != PAPER_I_APPEND_SUMMARY_DERIVATION
            or self.additional_estimator_acquisitions != 0
            or self.additional_controller_rounds != 0
        ):
            raise ValueError("Append summary must be a zero-work projection.")


def _accepted_history(
    result_payload: Mapping[str, Any],
) -> tuple[PaperIAppendAcceptedRoundSummary, ...]:
    rows = _sequence(result_payload.get("history"), name="result history")
    result: list[PaperIAppendAcceptedRoundSummary] = []
    for index, raw in enumerate(rows, start=1):
        row = _mapping(raw, name=f"result history[{index - 1}]")
        result.append(
            PaperIAppendAcceptedRoundSummary(
                schema=PAPER_I_APPEND_ACCEPTED_ROUND_SCHEMA,
                controller_round=_integer(
                    row.get("controller_round"),
                    name=f"history[{index - 1}].controller_round",
                    minimum=1,
                ),
                selected_label=_text(
                    row.get("selected_label"),
                    name=f"history[{index - 1}].selected_label",
                ),
                selected_generator_identity=_text(
                    row.get("selected_generator_identity"),
                    name=(
                        f"history[{index - 1}]."
                        "selected_generator_identity"
                    ),
                ),
                insertion_position=_integer(
                    row.get("insertion_position"),
                    name=f"history[{index - 1}].insertion_position",
                ),
                energy_before=_finite(
                    row.get("energy_before"),
                    name=f"history[{index - 1}].energy_before",
                ),
                energy_after=_finite(
                    row.get("energy_after"),
                    name=f"history[{index - 1}].energy_after",
                ),
                selected_abs_commutator_gradient=_finite(
                    row.get("selected_abs_commutator_gradient"),
                    name=(
                        f"history[{index - 1}]."
                        "selected_abs_commutator_gradient"
                    ),
                ),
            )
        )
    return tuple(result)


def _accounting_summary(
    result_payload: Mapping[str, Any],
) -> PaperIAppendAccountingSummary:
    accounting = _mapping(
        result_payload.get("estimator_accounting"),
        name="result estimator_accounting",
    )
    components_raw = accounting.get("components", accounting)
    components = _mapping(
        components_raw,
        name="result estimator_accounting components",
    )
    values = {
        key: _integer(
            components.get(key, accounting.get(key)),
            name=f"estimator_accounting.{key}",
        )
        for key in _ACCOUNTING_COMPONENTS
    }
    for key, expected in values.items():
        if _integer(
            accounting.get(key),
            name=f"estimator_accounting top-level {key}",
        ) != expected:
            raise ValueError(
                f"Append accounting component {key} disagrees with components."
            )
    s_alg = _integer(
        accounting.get("S_alg"),
        name="estimator_accounting.S_alg",
    )
    occurrence = _mapping(
        accounting.get("occurrence_summary"),
        name="estimator_accounting.occurrence_summary",
    )
    if _integer(
        occurrence.get("S_alg"),
        name="occurrence_summary.S_alg",
    ) != s_alg:
        raise ValueError("Append occurrence summary does not close to S_alg.")
    for key, expected in values.items():
        if _integer(
            occurrence.get(key),
            name=f"occurrence_summary.{key}",
        ) != expected:
            raise ValueError(
                f"Append occurrence component {key} does not reconcile."
            )
    prefix = _mapping(
        accounting.get("closed_occurrence_prefix"),
        name="estimator_accounting.closed_occurrence_prefix",
    )
    cumulative = _mapping(
        prefix.get("cumulative_executed_queries"),
        name="closed occurrence cumulative queries",
    )
    cumulative_components = _mapping(
        cumulative.get("components"),
        name="closed occurrence cumulative query components",
    )
    prefix_s_alg = _integer(
        cumulative.get("S_alg"),
        name="closed occurrence prefix S_alg",
    )
    for key, expected in values.items():
        if _integer(
            cumulative_components.get(key),
            name=f"closed occurrence prefix {key}",
        ) != expected:
            raise ValueError(
                f"Append occurrence prefix component {key} does not reconcile."
            )
    ledger = _mapping(
        result_payload.get("estimator_call_ledger"),
        name="result estimator_call_ledger",
    )
    ledger_occurrences = _sequence(
        ledger.get("occurrences"),
        name="estimator_call_ledger.occurrences",
    )
    ledger_occurrence_summary = _mapping(
        ledger.get("occurrence_summary"),
        name="estimator_call_ledger.occurrence_summary",
    )
    if _integer(
        ledger_occurrence_summary.get("S_alg"),
        name="ledger occurrence_summary.S_alg",
    ) != s_alg:
        raise ValueError("Append serialized ledger does not close to S_alg.")
    for key, expected in values.items():
        if _integer(
            ledger_occurrence_summary.get(key),
            name=f"ledger occurrence_summary.{key}",
        ) != expected:
            raise ValueError(
                f"Append ledger occurrence component {key} does not reconcile."
            )
    return PaperIAppendAccountingSummary(
        schema=PAPER_I_APPEND_ACCOUNTING_SUMMARY_SCHEMA,
        convention=_text(
            accounting.get("convention"),
            name="estimator_accounting.convention",
        ),
        **values,
        S_alg=s_alg,
        closed_occurrence_reconciliation=_boolean(
            accounting.get("closed_occurrence_reconciliation"),
            name="estimator_accounting.closed_occurrence_reconciliation",
        ),
        closed_occurrence_prefix_S_alg=prefix_s_alg,
        ledger_schema=_text(
            ledger.get("schema"),
            name="estimator_call_ledger.schema",
        ),
        ledger_fingerprint=_text(
            ledger.get("ledger_fingerprint"),
            name="estimator_call_ledger.ledger_fingerprint",
        ),
        ledger_sha256=canonical_sha256(ledger),
        ledger_occurrence_count=len(ledger_occurrences),
    )


def _resource_summary(
    result_payload: Mapping[str, Any],
) -> PaperIAppendResourceSummary:
    terminal = _mapping_copy(
        result_payload.get("compiled_resources"),
        name="result compiled_resources",
    )
    observation = _mapping(
        result_payload.get("resource_observation"),
        name="result resource_observation",
    )
    requested = tuple(
        _integer(value, name="requested resource round", minimum=1)
        for value in _sequence(
            observation.get("requested_resource_rounds"),
            name="requested_resource_rounds",
        )
    )
    materialized = tuple(
        _integer(value, name="materialized resource round", minimum=1)
        for value in _sequence(
            observation.get("materialized_resource_rounds"),
            name="materialized_resource_rounds",
        )
    )
    unmaterialized = tuple(
        _integer(value, name="unmaterialized resource round", minimum=1)
        for value in _sequence(
            observation.get("unmaterialized_resource_rounds"),
            name="unmaterialized_resource_rounds",
        )
    )
    rows: list[PaperIAppendCompiledResourceRow] = []
    for index, raw in enumerate(
        _sequence(
            result_payload.get("compiled_resources_by_round"),
            name="result compiled_resources_by_round",
        )
    ):
        row = _mapping(raw, name=f"compiled resource row {index}")
        resources = _mapping_copy(
            row.get("compiled_resources"),
            name=f"compiled resource row {index} resources",
        )
        rows.append(
            PaperIAppendCompiledResourceRow(
                schema=PAPER_I_APPEND_RESOURCE_ROW_SCHEMA,
                controller_round=_integer(
                    row.get("controller_round"),
                    name=f"compiled resource row {index} round",
                    minimum=1,
                ),
                accepted_prefix_length=_integer(
                    row.get("accepted_prefix_length"),
                    name=f"compiled resource row {index} prefix length",
                    minimum=1,
                ),
                observation_status=_compiled_observation_status(resources),
                compiled_resources=resources,
            )
        )
    prefix_status = (
        "not_requested"
        if not requested
        else "complete"
        if not unmaterialized
        else "partial"
    )
    return PaperIAppendResourceSummary(
        schema=PAPER_I_APPEND_RESOURCE_SUMMARY_SCHEMA,
        terminal_observation_status=_compiled_observation_status(terminal),
        terminal_compiled_resources=terminal,
        accepted_prefix_observation_status=prefix_status,
        requested_controller_rounds=requested,
        materialized_controller_rounds=materialized,
        unmaterialized_controller_rounds=unmaterialized,
        compiled_resources_by_round=tuple(rows),
    )


def summarize_paper_i_append_run(
    *,
    protocol: ResolvedRAAdaptProtocol,
    selector_identity: str,
    result_payload: Mapping[str, Any],
) -> PaperIAppendRunSummary:
    """Build one zero-work canonical summary from a completed Append payload."""

    if not isinstance(protocol, ResolvedRAAdaptProtocol):
        raise TypeError("protocol must be a ResolvedRAAdaptProtocol.")
    if protocol.schema != APPEND_ADAPT_PROTOCOL_SCHEMA:
        raise ValueError("Append summary requires an Append protocol.")
    payload = _mapping(result_payload, name="result_payload")
    if _text(
        payload.get("protocol_sha256"),
        name="result_payload.protocol_sha256",
    ) != protocol.sha256:
        raise ValueError("Append summary protocol digest drifted.")
    selector = _text(selector_identity, name="selector_identity")
    if (
        selector != APPEND_CONVENTIONAL_SELECTOR_ID
        or _text(
            payload.get("selector_identity"),
            name="result_payload.selector_identity",
        )
        != selector
    ):
        raise ValueError("Append summary selector identity drifted.")
    selector_scope = _text(
        payload.get("selector_scope"),
        name="result_payload.selector_scope",
    )
    if selector_scope != APPEND_CONVENTIONAL_SELECTOR_SCOPE:
        raise ValueError("Append summary selector scope drifted.")
    expected_payload_fields = {
        "algorithm_id": protocol.algorithm_id,
        "candidate_representation": protocol.candidate_representation,
        "compile_identity": dict(protocol.compile_identity),
        "accepted_refit_scope": protocol.accepted_refit_scope,
        "accepted_refit_coordinate_chart": (
            protocol.accepted_refit_coordinate_chart
        ),
    }
    for name, expected in expected_payload_fields.items():
        observed = payload.get(name)
        if observed != expected:
            raise ValueError(
                f"Append summary result payload {name} drifted from protocol."
            )
    history = _accepted_history(payload)
    completed = _integer(
        payload.get("controller_rounds_completed"),
        name="result_payload.controller_rounds_completed",
    )
    if completed > int(protocol.horizon):
        raise ValueError("Append completed rounds exceed the protocol horizon.")
    accepted_labels = tuple(
        _text(value, name="accepted operator label")
        for value in _sequence(
            payload.get("accepted_operator_labels"),
            name="result_payload.accepted_operator_labels",
        )
    )
    accepted_generators = tuple(
        _text(value, name="accepted generator identity")
        for value in _sequence(
            payload.get("accepted_generator_identities"),
            name="result_payload.accepted_generator_identities",
        )
    )
    accounting_summary = _accounting_summary(payload)
    if accounting_summary.convention != protocol.estimator_accounting_convention:
        raise ValueError(
            "Append summary accounting convention drifted from protocol."
        )
    return PaperIAppendRunSummary(
        schema=PAPER_I_APPEND_RUN_SUMMARY_SCHEMA,
        source_result_payload_sha256=canonical_sha256(payload),
        protocol_schema=protocol.schema,
        protocol_sha256=protocol.sha256,
        bundle_id=_text(protocol.bundle_id, name="protocol.bundle_id"),
        bundle_manifest_sha256=_text(
            protocol.bundle_manifest_sha256,
            name="protocol.bundle_manifest_sha256",
        ),
        algorithm_id=_text(
            payload.get("algorithm_id"),
            name="result_payload.algorithm_id",
        ),
        candidate_representation=_text(
            payload.get("candidate_representation"),
            name="result_payload.candidate_representation",
        ),
        adapter_id=_text(protocol.adapter_id, name="protocol.adapter_id"),
        active_gradient_policy=_text(
            protocol.active_gradient_policy,
            name="protocol.active_gradient_policy",
        ),
        resource_weighting_scope=_text(
            protocol.resource_weighting_scope,
            name="protocol.resource_weighting_scope",
        ),
        optimizer=_text(protocol.optimizer, name="protocol.optimizer"),
        optimizer_maxiter=_integer(
            protocol.optimizer_maxiter,
            name="protocol.optimizer_maxiter",
            minimum=1,
        ),
        protocol_horizon=_integer(
            protocol.horizon,
            name="protocol.horizon",
            minimum=1,
        ),
        stopping_rule=_mapping_copy(
            protocol.stopping_rule,
            name="protocol.stopping_rule",
        ),
        seeds=_mapping_copy(protocol.seeds, name="protocol.seeds"),
        selector_identity=selector,
        selector_scope=selector_scope,
        selector_source_id=_text(
            payload.get("selector_source_id"),
            name="result_payload.selector_source_id",
        ),
        selection_with_replacement=_boolean(
            payload.get("selection_with_replacement"),
            name="result_payload.selection_with_replacement",
        ),
        append_position_only=_boolean(
            payload.get("append_position_only"),
            name="result_payload.append_position_only",
        ),
        compile_identity=_mapping_copy(
            payload.get("compile_identity"),
            name="result_payload.compile_identity",
        ),
        accepted_refit_scope=_text(
            payload.get("accepted_refit_scope"),
            name="result_payload.accepted_refit_scope",
        ),
        accepted_refit_coordinate_chart=_text(
            payload.get("accepted_refit_coordinate_chart"),
            name="result_payload.accepted_refit_coordinate_chart",
        ),
        controller_rounds_completed=completed,
        stop_reason=_text(
            payload.get("stop_reason"),
            name="result_payload.stop_reason",
        ),
        final_energy=_finite(
            payload.get("final_energy"),
            name="result_payload.final_energy",
        ),
        accepted_operator_labels=accepted_labels,
        accepted_generator_identities=accepted_generators,
        accepted_history=history,
        estimator_accounting=accounting_summary,
        resources=_resource_summary(payload),
        derivation_policy=PAPER_I_APPEND_SUMMARY_DERIVATION,
        additional_estimator_acquisitions=0,
        additional_controller_rounds=0,
    )


__all__ = [
    "PAPER_I_APPEND_ACCOUNTING_SUMMARY_SCHEMA",
    "PAPER_I_APPEND_ACCEPTED_ROUND_SCHEMA",
    "PAPER_I_APPEND_RESOURCE_ROW_SCHEMA",
    "PAPER_I_APPEND_RESOURCE_SUMMARY_SCHEMA",
    "PAPER_I_APPEND_RUN_SUMMARY_SCHEMA",
    "PAPER_I_APPEND_SUMMARY_DERIVATION",
    "PaperIAppendAcceptedRoundSummary",
    "PaperIAppendAccountingSummary",
    "PaperIAppendCompiledResourceRow",
    "PaperIAppendResourceSummary",
    "PaperIAppendRunSummary",
    "summarize_paper_i_append_run",
]
