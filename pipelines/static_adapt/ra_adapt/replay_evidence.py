"""Authenticated controller-prefix evidence for Paper-I replay checks.

The numerical controllers already construct the state needed for strict
accepted-prefix replay.  This module projects that state into one common,
result-facing receipt without changing selection, optimization, stopping, or
estimator accounting.  Every controller-round prefix is signed independently;
the run-level replay identity and resume-sidecar closure are signed
independently as well.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
from typing import Any

from pipelines.static_adapt.adaptive_phase_contracts import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256


CONTROLLER_REPLAY_EVIDENCE_SCHEMA = (
    "paper_i_controller_replay_evidence_v1"
)
SIGNED_CONTROLLER_PREFIX_SCHEMA = (
    "paper_i_signed_controller_round_prefix_v1"
)
BOUNDED_REPLAY_IDENTITY_SCHEMA = (
    "paper_i_bounded_deterministic_replay_identity_v1"
)
BOUNDED_REPLAY_COMPARISON_SCHEMA = (
    "paper_i_bounded_deterministic_replay_comparison_v1"
)
RESUME_SIDECAR_CLOSURE_SCHEMA = (
    "paper_i_authenticated_resume_sidecar_closure_v1"
)
APPEND_SIGNED_PREFIX_SCHEMA = (
    "paper_i_signed_append_active_prefix_checkpoint_v1"
)
RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA = (
    "paper_i_ra_phase3_no_positive_controller_replay_terminal_v1"
)
RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA_V2 = (
    "paper_i_ra_phase3_no_positive_controller_replay_terminal_v2"
)

_RA_METHOD_FAMILY = "ra_adapt"
_APPEND_METHOD_FAMILY = "append_adapt"
_RA_PHASE3_NO_POSITIVE_SELECTION_SCHEMA = (
    "paper_i_ra_phase3_no_positive_selection_terminal_v1"
)
_RA_PHASE3_NO_POSITIVE_SELECTION_FIELDS = frozenset(
    {
        "schema",
        "terminal_controller_outcome",
        "accepted_controller_round",
        "attempted_controller_round",
        "accepted_state_fingerprint",
        "accepted_operator_count",
        "accepted_state_unchanged",
        "final_admission_record_id",
        "phase0_gradient_shortlist",
        "insertion_mode",
        "insertion_commutation_plateau",
        "insertion_commutation_reduced",
        "phase3_population_activation",
        "controller_measurement_work_proxy",
        "scored_insertion_position_population",
        "projected_phase3_population_receipt",
        "phase123_qiskit_population_normalization_receipts",
        "estimator_event_ids",
        "estimator_event_count",
        "estimator_event_ids_sha256",
        "terminal_active_prefix_checkpoint_sha256",
        "terminal_estimator_prefix_receipt",
        "terminal_estimator_prefix_receipt_sha256",
        "sha256",
    }
)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return copy.deepcopy(dict(value))


def _sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(
        value, (str, bytes, bytearray)
    ):
        raise TypeError(f"{name} must be a sequence.")
    return copy.deepcopy(list(value))


def _require_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef"
        for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return normalized


def _require_positive_round(value: Any, *, name: str) -> int:
    resolved = int(value)
    if isinstance(value, bool) or resolved < 1 or resolved != value:
        raise ValueError(f"{name} must be a positive controller round.")
    return resolved


def _require_nonnegative_round(value: Any, *, name: str) -> int:
    resolved = int(value)
    if isinstance(value, bool) or resolved < 0 or resolved != value:
        raise ValueError(f"{name} must be a nonnegative controller round.")
    return resolved


def _require_finite_sequence(
    value: Any,
    *,
    name: str,
) -> list[float]:
    values = _sequence(value, name=name)
    resolved = [float(item) for item in values]
    if any(not math.isfinite(item) for item in resolved):
        raise ValueError(f"{name} must contain only finite values.")
    return resolved


def _receipt_payload(value: Any, *, name: str) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        payload = value.to_dict()
    elif isinstance(value, Mapping):
        payload = dict(value)
    else:
        raise TypeError(f"{name} must provide to_dict() or be a mapping.")
    return _mapping(payload, name=name)


def _signed(
    payload: Mapping[str, Any],
    *,
    signature_field: str = "sha256",
) -> dict[str, Any]:
    signed = copy.deepcopy(dict(payload))
    if signature_field in signed:
        raise ValueError(
            f"Unsigned payload already contains {signature_field!r}."
        )
    signed[signature_field] = canonical_sha256(signed)
    return signed


def _verify_signed(
    value: Any,
    *,
    name: str,
    signature_field: str = "sha256",
) -> dict[str, Any]:
    payload = _mapping(value, name=name)
    observed = _require_sha256(
        payload.pop(signature_field, None),
        name=f"{name}.{signature_field}",
    )
    expected = canonical_sha256(payload)
    if observed != expected:
        raise ValueError(f"{name} SHA-256 does not match its payload.")
    payload[signature_field] = observed
    return payload


def _problem_request_sha256(protocol: Any) -> str:
    problem = getattr(protocol, "problem", None)
    return _require_sha256(
        getattr(problem, "problem_request_sha256", ""),
        name="protocol.problem.problem_request_sha256",
    )


def _scientific_input_identity(
    *,
    protocol: Any,
    method_family: str,
    route_identity: Mapping[str, Any],
) -> str:
    """Hash scientific inputs while excluding horizon and resume mechanics."""

    payload = {
        "schema": "paper_i_replay_scientific_input_identity_v1",
        "method_family": str(method_family),
        "algorithm_id": str(getattr(protocol, "algorithm_id", "")),
        "problem_request_sha256": _problem_request_sha256(protocol),
        "candidate_representation": str(
            getattr(protocol, "candidate_representation", "")
        ),
        "selector_identity": str(
            getattr(protocol, "selector_identity", "")
        ),
        "selector_scope": getattr(protocol, "selector_scope", None),
        "active_gradient_policy": str(
            getattr(protocol, "active_gradient_policy", "")
        ),
        "resource_weighting_scope": str(
            getattr(protocol, "resource_weighting_scope", "")
        ),
        "derivative_chart_id": str(
            getattr(protocol, "derivative_chart_id", "")
        ),
        "trust_policy_id": str(
            getattr(protocol, "trust_policy_id", "")
        ),
        "phase3_solver_id": str(
            getattr(protocol, "phase3_solver_id", "")
        ),
        "accepted_refit_scope": str(
            getattr(protocol, "accepted_refit_scope", "")
        ),
        "accepted_refit_coordinate_chart": str(
            getattr(protocol, "accepted_refit_coordinate_chart", "")
        ),
        "accepted_refit_base_chart_policy": str(
            getattr(protocol, "accepted_refit_base_chart_policy", "")
        ),
        "executable_pool_sha256": str(
            getattr(
                getattr(protocol, "executable_pool", None),
                "ordered_pool_sha256",
                "",
            )
        ),
        "optimizer": str(getattr(protocol, "optimizer", "")),
        "optimizer_maxiter": int(
            getattr(protocol, "optimizer_maxiter", 0)
        ),
        "seeds": dict(getattr(protocol, "seeds", {})),
        "estimator_accounting_convention": str(
            getattr(protocol, "estimator_accounting_convention", "")
        ),
        "compile_identity": dict(
            getattr(protocol, "compile_identity", {})
        ),
        "route_identity": dict(route_identity),
    }
    return canonical_sha256(payload)


def _prefix_replay_identity(
    *,
    method_family: str,
    problem_request_sha256: str,
    route_identity: Mapping[str, Any],
    controller_round: int,
    operator_labels: Sequence[str],
    logical_parameters: Sequence[float],
    runtime_parameters: Sequence[float],
    state_fingerprint: str,
    accepted_energy: float,
) -> str:
    energy = float(accepted_energy)
    if not math.isfinite(energy):
        raise ValueError("Accepted prefix energy must be finite.")
    payload = {
        "schema": "paper_i_controller_prefix_replay_identity_v1",
        "method_family": str(method_family),
        "problem_request_sha256": str(problem_request_sha256),
        "route_identity": dict(route_identity),
        "controller_round": int(controller_round),
        "ordered_operator_labels": [str(value) for value in operator_labels],
        "logical_parameters": [float(value) for value in logical_parameters],
        "runtime_parameters": [float(value) for value in runtime_parameters],
        "projective_state_fingerprint": str(state_fingerprint),
        "accepted_energy": energy,
    }
    return canonical_sha256(payload)


def build_signed_append_prefix_checkpoint(
    *,
    protocol: Any,
    controller_round: int,
    accepted_operator_labels: Sequence[str],
    accepted_generator_identities: Sequence[str],
    logical_parameters: Sequence[float],
    runtime_parameters: Sequence[float],
    projective_state_fingerprint: str,
    accepted_energy: float,
    accepted_refit: Mapping[str, Any],
    estimator_prefix: Mapping[str, Any],
) -> dict[str, Any]:
    """Sign one complete Append accepted prefix without enabling resume."""

    round_index = _require_positive_round(
        controller_round,
        name="controller_round",
    )
    labels = [str(value) for value in accepted_operator_labels]
    generators = [str(value) for value in accepted_generator_identities]
    if len(labels) != round_index or len(generators) != len(labels):
        raise ValueError(
            "Append signed prefix must contain one accepted generator per "
            "completed controller round."
        )
    logical = _require_finite_sequence(
        logical_parameters,
        name="logical_parameters",
    )
    runtime = _require_finite_sequence(
        runtime_parameters,
        name="runtime_parameters",
    )
    fingerprint = str(projective_state_fingerprint).strip()
    if not fingerprint:
        raise ValueError(
            "Append signed prefix requires a state fingerprint."
        )
    energy = float(accepted_energy)
    if not math.isfinite(energy):
        raise ValueError("Append signed prefix energy must be finite.")
    prefix = _mapping(
        estimator_prefix,
        name="estimator_prefix",
    )
    if (
        prefix.get("schema")
        != "estimator_call_ledger_occurrence_prefix_summary_v1"
    ):
        raise ValueError(
            "Append signed prefix requires the canonical occurrence-prefix "
            "ledger summary."
        )
    payload = {
        "schema": APPEND_SIGNED_PREFIX_SCHEMA,
        "controller_round": round_index,
        "protocol_sha256": _require_sha256(
            getattr(protocol, "sha256", ""),
            name="protocol.sha256",
        ),
        "problem_request_sha256": _problem_request_sha256(protocol),
        "selector_identity": str(
            getattr(protocol, "selector_identity", "")
        ),
        "selector_scope": str(getattr(protocol, "selector_scope", "")),
        "accepted_operator_labels": labels,
        "accepted_generator_identities": generators,
        "logical_parameters": logical,
        "runtime_parameters": runtime,
        "projective_state_fingerprint": fingerprint,
        "accepted_energy": energy,
        "accepted_refit": _mapping(
            accepted_refit,
            name="accepted_refit",
        ),
        "estimator_prefix": prefix,
    }
    return _signed(payload, signature_field="checkpoint_sha256")


def _append_prefix_wrapper(
    *,
    protocol: Any,
    checkpoint: Mapping[str, Any],
    previous_sha256: str | None,
) -> dict[str, Any]:
    signed_checkpoint = _verify_signed(
        checkpoint,
        name="Append active-prefix checkpoint",
        signature_field="checkpoint_sha256",
    )
    if signed_checkpoint.get("schema") != APPEND_SIGNED_PREFIX_SCHEMA:
        raise ValueError("Unknown Append active-prefix checkpoint schema.")
    round_index = _require_positive_round(
        signed_checkpoint.get("controller_round"),
        name="Append active-prefix controller_round",
    )
    protocol_sha256 = _require_sha256(
        getattr(protocol, "sha256", ""),
        name="protocol.sha256",
    )
    problem_sha256 = _problem_request_sha256(protocol)
    if (
        signed_checkpoint.get("protocol_sha256") != protocol_sha256
        or signed_checkpoint.get("problem_request_sha256") != problem_sha256
        or signed_checkpoint.get("selector_identity")
        != getattr(protocol, "selector_identity", None)
        or signed_checkpoint.get("selector_scope")
        != getattr(protocol, "selector_scope", None)
    ):
        raise ValueError(
            "Append active-prefix checkpoint is not bound to its protocol."
        )
    route_identity = {
        "selector_identity": str(getattr(protocol, "selector_identity", "")),
        "selector_scope": str(getattr(protocol, "selector_scope", "")),
    }
    prefix_identity = _prefix_replay_identity(
        method_family=_APPEND_METHOD_FAMILY,
        problem_request_sha256=problem_sha256,
        route_identity=route_identity,
        controller_round=round_index,
        operator_labels=signed_checkpoint["accepted_operator_labels"],
        logical_parameters=signed_checkpoint["logical_parameters"],
        runtime_parameters=signed_checkpoint["runtime_parameters"],
        state_fingerprint=signed_checkpoint[
            "projective_state_fingerprint"
        ],
        accepted_energy=float(signed_checkpoint["accepted_energy"]),
    )
    return _signed(
        {
            "schema": SIGNED_CONTROLLER_PREFIX_SCHEMA,
            "method_family": _APPEND_METHOD_FAMILY,
            "controller_round": round_index,
            "protocol_sha256": protocol_sha256,
            "problem_request_sha256": problem_sha256,
            "route_identity": route_identity,
            "prefix_replay_identity_sha256": prefix_identity,
            "source_checkpoint_sha256": signed_checkpoint[
                "checkpoint_sha256"
            ],
            "preceding_signed_prefix_sha256": previous_sha256,
            "active_prefix_checkpoint": signed_checkpoint,
        }
    )


def build_append_controller_replay_evidence(
    *,
    protocol: Any,
    history: Sequence[Mapping[str, Any]],
    estimator_ledger: Mapping[str, Any],
    estimator_accounting: Mapping[str, Any],
) -> dict[str, Any]:
    """Build result-facing replay evidence for a completed Append run."""

    history_rows = [
        _mapping(row, name=f"Append history[{index}]")
        for index, row in enumerate(history)
    ]
    wrappers: list[dict[str, Any]] = []
    previous: str | None = None
    for index, row in enumerate(history_rows, start=1):
        checkpoint = _mapping(
            row.get("active_prefix_checkpoint"),
            name=f"Append history[{index - 1}].active_prefix_checkpoint",
        )
        wrapper = _append_prefix_wrapper(
            protocol=protocol,
            checkpoint=checkpoint,
            previous_sha256=previous,
        )
        if wrapper["controller_round"] != index:
            raise ValueError("Append prefix rounds must be contiguous.")
        wrappers.append(wrapper)
        previous = str(wrapper["sha256"])

    ledger = _mapping(estimator_ledger, name="Append estimator ledger")
    accounting = _mapping(
        estimator_accounting,
        name="Append estimator accounting",
    )
    terminal_prefix = _mapping(
        accounting.get("closed_occurrence_prefix"),
        name="Append terminal occurrence prefix",
    )
    if wrappers and (
        wrappers[-1]["active_prefix_checkpoint"]["estimator_prefix"]
        != terminal_prefix
    ):
        raise ValueError(
            "Append terminal signed prefix does not close to estimator "
            "accounting."
        )
    route_identity = {
        "selector_identity": str(getattr(protocol, "selector_identity", "")),
        "selector_scope": str(getattr(protocol, "selector_scope", "")),
    }
    scientific_input_sha256 = _scientific_input_identity(
        protocol=protocol,
        method_family=_APPEND_METHOD_FAMILY,
        route_identity=route_identity,
    )
    replay_identity = _signed(
        {
            "schema": BOUNDED_REPLAY_IDENTITY_SCHEMA,
            "method_family": _APPEND_METHOD_FAMILY,
            "scientific_input_sha256": scientific_input_sha256,
            "accepted_prefix_replay_identity_sha256s": [
                row["prefix_replay_identity_sha256"] for row in wrappers
            ],
            "signed_controller_prefix_sha256s": [
                row["sha256"] for row in wrappers
            ],
            "history_projection_sha256": canonical_sha256(history_rows),
        }
    )
    resume_closure = _signed(
        {
            "schema": RESUME_SIDECAR_CLOSURE_SCHEMA,
            "method_family": _APPEND_METHOD_FAMILY,
            "resume_mode": "authenticated_reconstruction_only_v1",
            "public_resume_execution_supported": False,
            "continuation_execution_status": (
                "not_authorized_append_resume_contract"
            ),
            "problem_request_sha256": _problem_request_sha256(protocol),
            "protocol_sha256": _require_sha256(
                getattr(protocol, "sha256", ""),
                name="protocol.sha256",
            ),
            "route_identity": route_identity,
            "terminal_signed_prefix_sha256": (
                None if not wrappers else wrappers[-1]["sha256"]
            ),
            "terminal_source_checkpoint_sha256": (
                None
                if not wrappers
                else wrappers[-1]["source_checkpoint_sha256"]
            ),
            "estimator_ledger_sha256": canonical_sha256(ledger),
            "terminal_estimator_prefix_sha256": canonical_sha256(
                terminal_prefix
            ),
            "signed_prefix_count": len(wrappers),
            "zero_acceptance_terminal": not wrappers,
            "reconstruction_fields_complete": True,
        }
    )
    return _signed(
        {
            "schema": CONTROLLER_REPLAY_EVIDENCE_SCHEMA,
            "method_family": _APPEND_METHOD_FAMILY,
            "protocol_sha256": _require_sha256(
                getattr(protocol, "sha256", ""),
                name="protocol.sha256",
            ),
            "problem_request_sha256": _problem_request_sha256(protocol),
            "signed_controller_round_prefixes": wrappers,
            "bounded_replay_identity": replay_identity,
            "resume_sidecar_closure": resume_closure,
        }
    )


def _ra_route_identity(run: Any) -> dict[str, Any]:
    route = getattr(run, "route", None)
    return {
        "family": str(getattr(route, "family", "")),
        "profile": str(getattr(route, "profile", "")),
        "route_contract_sha256": _require_sha256(
            getattr(route, "contract_sha256", ""),
            name="run.route.contract_sha256",
        ),
    }


def _ra_prefix_wrapper(
    *,
    protocol: Any,
    run: Any,
    checkpoint: Mapping[str, Any],
    previous_sha256: str | None,
) -> dict[str, Any]:
    signed_checkpoint = _verify_signed(
        checkpoint,
        name="RA active-prefix checkpoint",
        signature_field="checkpoint_sha256",
    )
    if (
        signed_checkpoint.get("schema")
        != "paper_i_signed_active_prefix_checkpoint_v1"
    ):
        raise ValueError("Unknown RA active-prefix checkpoint schema.")
    round_index = _require_positive_round(
        signed_checkpoint.get("outer_iteration"),
        name="RA active-prefix outer_iteration",
    )
    route_identity = _ra_route_identity(run)
    if (
        signed_checkpoint.get("sr_route_profile")
        != route_identity["profile"]
        or signed_checkpoint.get("sr_route_profile_contract_sha256")
        != route_identity["route_contract_sha256"]
    ):
        raise ValueError(
            "RA active-prefix checkpoint is not bound to the result route."
        )
    strict_replay = _mapping(
        signed_checkpoint.get("strict_replay"),
        name="RA active-prefix strict replay",
    )
    if strict_replay.get("passed") is not True:
        raise ValueError("RA active-prefix strict state replay did not pass.")
    states = {
        int(getattr(state, "controller_round")): state
        for state in getattr(run, "accepted_trajectory", ())
    }
    transitions = {
        int(getattr(row, "controller_round")): row
        for row in getattr(run, "accepted_transitions", ())
    }
    replay_rows = {
        int(getattr(row, "controller_round")): row
        for row in getattr(run, "scientific_replay", ())
    }
    if (
        round_index not in states
        or round_index not in transitions
        or round_index not in replay_rows
    ):
        raise ValueError(
            "RA active-prefix checkpoint has no aligned typed result row."
        )
    state_payload = _receipt_payload(
        states[round_index],
        name=f"RA accepted state round {round_index}",
    )
    if (
        signed_checkpoint.get("projective_state_fingerprint")
        != state_payload.get("projective_state_fingerprint")
        or list(
            signed_checkpoint.get(
                "signed_unwrapped_logical_parameters", ()
            )
        )
        != list(state_payload.get("logical_parameters", ()))
        or list(
            signed_checkpoint.get(
                "signed_unwrapped_runtime_parameters", ()
            )
        )
        != list(state_payload.get("runtime_parameters", ()))
        or list(
            signed_checkpoint.get("ordered_active_operator_labels", ())
        )
        != list(state_payload.get("operators", ()))
    ):
        raise ValueError(
            "RA signed checkpoint disagrees with its accepted state."
        )
    problem_sha256 = _problem_request_sha256(protocol)
    prefix_identity = _prefix_replay_identity(
        method_family=_RA_METHOD_FAMILY,
        problem_request_sha256=problem_sha256,
        route_identity=route_identity,
        controller_round=round_index,
        operator_labels=state_payload["operators"],
        logical_parameters=state_payload["logical_parameters"],
        runtime_parameters=state_payload["runtime_parameters"],
        state_fingerprint=state_payload["projective_state_fingerprint"],
        accepted_energy=float(state_payload["energy"]),
    )
    return _signed(
        {
            "schema": SIGNED_CONTROLLER_PREFIX_SCHEMA,
            "method_family": _RA_METHOD_FAMILY,
            "controller_round": round_index,
            "protocol_sha256": _require_sha256(
                getattr(protocol, "sha256", ""),
                name="protocol.sha256",
            ),
            "problem_request_sha256": problem_sha256,
            "route_identity": route_identity,
            "prefix_replay_identity_sha256": prefix_identity,
            "source_checkpoint_sha256": signed_checkpoint[
                "checkpoint_sha256"
            ],
            "preceding_signed_prefix_sha256": previous_sha256,
            "accepted_state": state_payload,
            "accepted_state_sha256": canonical_sha256(state_payload),
            "accepted_transition_sha256": canonical_sha256(
                _receipt_payload(
                    transitions[round_index],
                    name=f"RA transition round {round_index}",
                )
            ),
            "scientific_replay_sha256": canonical_sha256(
                _receipt_payload(
                    replay_rows[round_index],
                    name=f"RA replay round {round_index}",
                )
            ),
            "active_prefix_checkpoint": signed_checkpoint,
        }
    )


def _validate_ra_phase3_no_positive_replay_terminal(
    value: Any,
    *,
    prefixes: Sequence[Mapping[str, Any]],
    terminal_checkpoint: Mapping[str, Any],
    estimator_prefix_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Cross-bind one failed Phase-III admission to its accepted prefix."""

    terminal = _verify_signed(
        value,
        name="RA Phase-III no-positive replay terminal",
    )
    common_fields = {
        "schema",
        "terminal_controller_outcome",
        "accepted_controller_round",
        "attempted_controller_round",
        "accepted_state_unchanged",
        "accepted_state_sha256",
        "natural_terminal_route_contract",
        "natural_terminal_route_contract_sha256",
        "terminal_phase3_selection_receipt",
        "terminal_phase3_selection_receipt_sha256",
        "terminal_active_prefix_checkpoint",
        "terminal_active_prefix_checkpoint_sha256",
        "terminal_estimator_prefix_receipt",
        "terminal_estimator_prefix_receipt_sha256",
        "sha256",
    }
    schema = terminal.get("schema")
    round_zero_terminal = bool(
        schema == RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA_V2
    )
    expected_fields = common_fields.union(
        {
            "round_zero_accepted_state",
            "round_zero_accepted_state_sha256",
        }
        if round_zero_terminal
        else {"accepted_signed_prefix_sha256"}
    )
    if set(terminal) != expected_fields:
        raise ValueError(
            "RA Phase-III no-positive replay terminal fields drifted."
        )
    accepted_round = _require_nonnegative_round(
        terminal.get("accepted_controller_round"),
        name="accepted controller round",
    )
    attempted_round = _require_positive_round(
        terminal.get("attempted_controller_round"),
        name="attempted controller round",
    )
    if attempted_round != accepted_round + 1:
        raise ValueError(
            "RA Phase-III attempted controller round must equal accepted "
            "round plus one."
        )
    if (
        schema
        not in {
            RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA,
            RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA_V2,
        }
        or terminal.get("terminal_controller_outcome")
        != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        or terminal.get("accepted_state_unchanged") is not True
        or (
            round_zero_terminal
            and (accepted_round != 0 or prefixes)
        )
        or (
            not round_zero_terminal
            and (
                accepted_round < 1
                or not prefixes
                or int(prefixes[-1].get("controller_round", -1))
                != accepted_round
                or terminal.get("accepted_signed_prefix_sha256")
                != prefixes[-1].get("sha256")
            )
        )
    ):
        raise ValueError(
            "RA Phase-III no-positive replay terminal identity drifted."
        )

    accepted_state = _mapping(
        terminal.get("round_zero_accepted_state")
        if round_zero_terminal
        else prefixes[-1].get("accepted_state"),
        name="terminal accepted state",
    )
    if (
        canonical_sha256(accepted_state)
        != terminal.get("accepted_state_sha256")
        or (
            round_zero_terminal
            and canonical_sha256(accepted_state)
            != terminal.get("round_zero_accepted_state_sha256")
        )
    ):
        raise ValueError(
            "RA Phase-III terminal accepted-state digest drifted."
        )

    natural_terminal_route = _mapping(
        terminal.get("natural_terminal_route_contract"),
        name="RA Phase-III natural-terminal route contract",
    )
    natural_terminal_route_sha256 = _require_sha256(
        terminal.get("natural_terminal_route_contract_sha256"),
        name="RA Phase-III natural-terminal route contract SHA-256",
    )
    from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
        validate_semantic_phase3_natural_terminal_route_contract,
    )

    validate_semantic_phase3_natural_terminal_route_contract(
        natural_terminal_route,
        expected_route_contract_sha256=natural_terminal_route_sha256,
    )
    terminal_route_identity = (
        {
            "profile": natural_terminal_route.get("route_profile"),
            "route_contract_sha256": natural_terminal_route_sha256,
        }
        if round_zero_terminal
        else _mapping(
            prefixes[-1].get("route_identity"),
            name="RA Phase-III terminal prefix route identity",
        )
    )
    if (
        natural_terminal_route_sha256
        != terminal_route_identity.get("route_contract_sha256")
        or natural_terminal_route_sha256
        != terminal_checkpoint.get("sr_route_profile_contract_sha256")
        or natural_terminal_route.get("route_profile")
        != terminal_route_identity.get("profile")
        or natural_terminal_route.get("route_profile")
        != terminal_checkpoint.get("sr_route_profile")
    ):
        raise ValueError(
            "RA Phase-III natural-terminal route provenance is detached."
        )

    selection = _verify_signed(
        terminal.get("terminal_phase3_selection_receipt"),
        name="RA Phase-III terminal selection receipt",
    )
    if set(selection) != _RA_PHASE3_NO_POSITIVE_SELECTION_FIELDS:
        raise ValueError(
            "RA Phase-III terminal selection receipt fields drifted."
        )
    event_ids = _sequence(
        selection.get("estimator_event_ids"),
        name="RA Phase-III terminal estimator event IDs",
    )
    insertion_mode = selection.get("insertion_mode")
    plateau = selection.get("insertion_commutation_plateau")
    reduced = selection.get("insertion_commutation_reduced")
    activation = selection.get("phase3_population_activation")
    controller_work = selection.get("controller_measurement_work_proxy")
    insertion_evidence_valid = bool(
        (
            insertion_mode == "append_only"
            and plateau is None
            and reduced is None
        )
        or (
            insertion_mode
            in {
                "insertion_commutation_plateau_v1",
                "insertion_commutation_plateau_v2",
            }
            and isinstance(plateau, Mapping)
            and plateau.get("policy") == insertion_mode
            and reduced is None
        )
        or (
            insertion_mode == "full_commutation_reduced"
            and plateau is None
            and isinstance(reduced, Mapping)
            and reduced.get("policy") == "always_commutation_reduced"
        )
        or (
            insertion_mode == "append_commutation_reduced"
            and plateau is None
            and isinstance(reduced, Mapping)
            and reduced.get("policy")
            == "append_commutation_reduced"
        )
    )
    if (
        selection.get("schema")
        != _RA_PHASE3_NO_POSITIVE_SELECTION_SCHEMA
        or selection.get("terminal_controller_outcome")
        != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        or selection.get("accepted_controller_round") != accepted_round
        or selection.get("attempted_controller_round") != attempted_round
        or selection.get("accepted_state_unchanged") is not True
        or selection.get("final_admission_record_id") is not None
        or selection.get("accepted_state_fingerprint")
        != accepted_state.get("projective_state_fingerprint")
        or selection.get("accepted_operator_count")
        != len(accepted_state.get("operators", ()))
        or terminal.get("terminal_phase3_selection_receipt_sha256")
        != selection.get("sha256")
        or selection.get("estimator_event_count") != len(event_ids)
        or any(not isinstance(value, str) or not value for value in event_ids)
        or len(set(event_ids)) != len(event_ids)
        or selection.get("estimator_event_ids_sha256")
        != canonical_sha256(event_ids)
        or any(
            not isinstance(selection.get(field), Mapping)
            for field in (
                "phase0_gradient_shortlist",
                "phase3_population_activation",
                "controller_measurement_work_proxy",
                "scored_insertion_position_population",
                "projected_phase3_population_receipt",
                "phase123_qiskit_population_normalization_receipts",
            )
        )
        or not insertion_evidence_valid
        or not isinstance(
            activation.get("competitive_population_live"), bool
        )
        or controller_work.get("schema")
        != "controller_measurement_work_proxy_v1"
    ):
        raise ValueError(
            "RA Phase-III terminal selection receipt is detached from the "
            "accepted prefix."
        )

    checkpoint = _verify_signed(
        terminal.get("terminal_active_prefix_checkpoint"),
        name="RA Phase-III terminal active-prefix checkpoint",
        signature_field="checkpoint_sha256",
    )
    expected_checkpoint = _mapping(
        terminal_checkpoint,
        name="RA resume terminal active-prefix checkpoint",
    )
    if (
        checkpoint != expected_checkpoint
        or checkpoint.get("checkpoint_kind")
        != "terminal_phase3_no_positive"
        or checkpoint.get("outer_iteration") != accepted_round
        or checkpoint.get("projective_state_fingerprint")
        != accepted_state.get("projective_state_fingerprint")
        or list(checkpoint.get("ordered_active_operator_labels", ()))
        != list(accepted_state.get("operators", ()))
        or list(checkpoint.get("signed_unwrapped_logical_parameters", ()))
        != list(accepted_state.get("logical_parameters", ()))
        or list(checkpoint.get("signed_unwrapped_runtime_parameters", ()))
        != list(accepted_state.get("runtime_parameters", ()))
        or terminal.get("terminal_active_prefix_checkpoint_sha256")
        != canonical_sha256(checkpoint)
        or selection.get("terminal_active_prefix_checkpoint_sha256")
        != canonical_sha256(checkpoint)
    ):
        raise ValueError(
            "RA Phase-III terminal checkpoint changed the accepted state."
        )

    estimator_prefix = _mapping(
        terminal.get("terminal_estimator_prefix_receipt"),
        name="RA Phase-III terminal estimator-prefix receipt",
    )
    selection_estimator_prefix = _mapping(
        selection.get("terminal_estimator_prefix_receipt"),
        name="RA Phase-III selection estimator-prefix receipt",
    )
    if not estimator_prefix_receipts:
        raise ValueError(
            "RA Phase-III terminal lacks estimator-prefix receipts."
        )
    final_estimator_prefix = _mapping(
        estimator_prefix_receipts[-1],
        name="RA final estimator-prefix receipt",
    )
    if (
        estimator_prefix != selection_estimator_prefix
        or estimator_prefix != final_estimator_prefix
        or estimator_prefix.get("checkpoint_kind")
        != "terminal_phase3_no_positive"
        or terminal.get("terminal_estimator_prefix_receipt_sha256")
        != canonical_sha256(estimator_prefix)
        or selection.get("terminal_estimator_prefix_receipt_sha256")
        != canonical_sha256(estimator_prefix)
    ):
        raise ValueError(
            "RA Phase-III terminal estimator-prefix receipt is detached."
        )
    return terminal


def _build_ra_phase3_no_positive_replay_terminal(
    *,
    protocol: Any,
    run: Any,
    finalization: Mapping[str, Any],
    continuation: Mapping[str, Any],
    prefixes: Sequence[Mapping[str, Any]],
    terminal_checkpoint: Mapping[str, Any],
    estimator_prefix_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Build the conditional replay terminal without changing other receipts."""

    final_outcome = finalization.get("terminal_controller_outcome")
    run_outcome = getattr(
        getattr(run, "stop", None),
        "terminal_controller_outcome",
        None,
    )
    final_selection = finalization.get("terminal_phase3_selection_receipt")
    continuation_selection = continuation.get(
        "terminal_phase3_selection_receipt"
    )
    phase3_terminal_present = bool(
        final_outcome == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        or run_outcome == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        or final_selection is not None
        or continuation_selection is not None
    )
    if not phase3_terminal_present:
        return None
    if (
        final_outcome != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        or run_outcome != ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        or not isinstance(final_selection, Mapping)
        or not isinstance(continuation_selection, Mapping)
        or dict(final_selection) != dict(continuation_selection)
    ):
        raise ValueError(
            "RA Phase-III terminal outcome and selection evidence disagree."
        )

    route_contract = _mapping(
        finalization.get("sr_route_profile_contract"),
        name="RA Phase-III natural-terminal route contract",
    )
    route_contract_sha256 = _require_sha256(
        finalization.get("sr_route_profile_contract_sha256"),
        name="RA Phase-III natural-terminal route contract SHA-256",
    )
    from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
        validate_semantic_phase3_natural_terminal_route_contract,
    )

    validate_semantic_phase3_natural_terminal_route_contract(
        route_contract,
        expected_route_contract_sha256=route_contract_sha256,
    )
    run_route = getattr(run, "route", None)
    if (
        route_contract.get("algorithm_id")
        != getattr(protocol, "algorithm_id", None)
        or route_contract.get("route_profile")
        != getattr(run_route, "profile", None)
        or route_contract_sha256
        != getattr(run_route, "contract_sha256", None)
        or route_contract_sha256
        != terminal_checkpoint.get("sr_route_profile_contract_sha256")
    ):
        raise ValueError(
            "RA Phase-III natural-terminal route provenance is detached."
        )

    run_trajectory = tuple(getattr(run, "accepted_trajectory", ()))
    run_transitions = tuple(getattr(run, "accepted_transitions", ()))
    run_replay = tuple(getattr(run, "scientific_replay", ()))
    accepted_round = len(run_trajectory)
    round_zero_terminal = accepted_round == 0
    if len(run_transitions) != accepted_round or len(run_replay) != accepted_round:
        raise ValueError(
            "RA Phase-III terminal accepted evidence cardinalities drifted."
        )
    if round_zero_terminal:
        if prefixes:
            raise ValueError(
                "Round-zero Phase-III terminal cannot claim an accepted prefix."
            )
        accepted_state = _receipt_payload(
            getattr(run, "final_state", None),
            name="RA Phase-III round-zero accepted state",
        )
    else:
        if (
            not prefixes
            or int(getattr(run_trajectory[-1], "controller_round", -1))
            != accepted_round
            or int(prefixes[-1].get("controller_round", -1))
            != accepted_round
        ):
            raise ValueError(
                "RA Phase-III terminal requires its complete accepted prefix."
            )
        accepted_state = _receipt_payload(
            run_trajectory[-1],
            name="RA Phase-III terminal accepted state",
        )
    if _receipt_payload(
        getattr(run, "final_state", None),
        name="RA Phase-III terminal final state",
    ) != accepted_state:
        raise ValueError(
            "RA Phase-III terminal changed the final accepted state."
        )
    selection = _verify_signed(
        final_selection,
        name="RA Phase-III terminal selection receipt",
    )
    estimator_prefix = _mapping(
        selection.get("terminal_estimator_prefix_receipt"),
        name="RA Phase-III terminal estimator-prefix receipt",
    )
    terminal = _signed(
        {
            "schema": (
                RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA_V2
                if round_zero_terminal
                else RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA
            ),
            "terminal_controller_outcome": (
                ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
            ),
            "accepted_controller_round": accepted_round,
            "attempted_controller_round": accepted_round + 1,
            "accepted_state_unchanged": True,
            "accepted_state_sha256": canonical_sha256(accepted_state),
            **(
                {
                    "round_zero_accepted_state": accepted_state,
                    "round_zero_accepted_state_sha256": canonical_sha256(
                        accepted_state
                    ),
                }
                if round_zero_terminal
                else {
                    "accepted_signed_prefix_sha256": prefixes[-1]["sha256"]
                }
            ),
            "natural_terminal_route_contract": route_contract,
            "natural_terminal_route_contract_sha256": (
                route_contract_sha256
            ),
            "terminal_phase3_selection_receipt": selection,
            "terminal_phase3_selection_receipt_sha256": selection["sha256"],
            "terminal_active_prefix_checkpoint": copy.deepcopy(
                dict(terminal_checkpoint)
            ),
            "terminal_active_prefix_checkpoint_sha256": canonical_sha256(
                terminal_checkpoint
            ),
            "terminal_estimator_prefix_receipt": estimator_prefix,
            "terminal_estimator_prefix_receipt_sha256": canonical_sha256(
                estimator_prefix
            ),
        }
    )
    return _validate_ra_phase3_no_positive_replay_terminal(
        terminal,
        prefixes=prefixes,
        terminal_checkpoint=terminal_checkpoint,
        estimator_prefix_receipts=estimator_prefix_receipts,
    )


def build_ra_controller_replay_evidence(
    *,
    protocol: Any,
    run: Any,
    finalization: Mapping[str, Any],
) -> dict[str, Any]:
    """Expose the controller's already-authenticated RA replay state."""

    protocol_problem_sha256 = _problem_request_sha256(protocol)
    if (
        getattr(
            getattr(run, "problem", None),
            "problem_request_sha256",
            None,
        )
        != protocol_problem_sha256
    ):
        raise ValueError(
            "RA typed result and resolved protocol describe different "
            "physical problems."
        )
    final = _mapping(finalization, name="RA finalization")
    history = _sequence(final.get("history"), name="RA finalization.history")
    continuation = _mapping(
        final.get("continuation"),
        name="RA finalization.continuation",
    )
    declared_prefixes = _sequence(
        continuation.get("active_prefix_checkpoints"),
        name="RA continuation.active_prefix_checkpoints",
    )
    history_prefixes = [
        _mapping(
            _mapping(row, name=f"RA history[{index}]").get(
                "active_prefix_checkpoint"
            ),
            name=f"RA history[{index}].active_prefix_checkpoint",
        )
        for index, row in enumerate(history)
    ]
    if declared_prefixes != history_prefixes:
        raise ValueError(
            "RA history and continuation signed-prefix lists disagree."
        )
    stationary_without_prefix = bool(
        not declared_prefixes
        and final.get("terminal_controller_outcome")
        == "phase0_stationary_no_competitive_candidate_v1"
        and getattr(
            getattr(run, "stop", None),
            "terminal_controller_outcome",
            None,
        )
        == "phase0_stationary_no_competitive_candidate_v1"
        and not getattr(run, "accepted_trajectory", ())
    )
    phase3_without_prefix = bool(
        not declared_prefixes
        and final.get("terminal_controller_outcome")
        == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        and getattr(
            getattr(run, "stop", None),
            "terminal_controller_outcome",
            None,
        )
        == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        and not getattr(run, "accepted_trajectory", ())
        and isinstance(
            final.get("terminal_phase3_selection_receipt"),
            Mapping,
        )
    )
    if not declared_prefixes and not (
        stationary_without_prefix or phase3_without_prefix
    ):
        raise ValueError("RA replay evidence requires accepted prefixes.")

    wrappers: list[dict[str, Any]] = []
    previous: str | None = None
    expected_round: int | None = None
    for prefix_checkpoint in declared_prefixes:
        wrapper = _ra_prefix_wrapper(
            protocol=protocol,
            run=run,
            checkpoint=_mapping(
                prefix_checkpoint,
                name="RA active-prefix checkpoint",
            ),
            previous_sha256=previous,
        )
        if expected_round is None:
            expected_round = int(wrapper["controller_round"])
        if int(wrapper["controller_round"]) != expected_round:
            raise ValueError("RA signed-prefix rounds must be contiguous.")
        expected_round += 1
        wrappers.append(wrapper)
        previous = str(wrapper["sha256"])

    continuation_terminal = continuation.get(
        "terminal_active_prefix_checkpoint"
    )
    terminal_source = continuation_terminal
    if (
        isinstance(continuation_terminal, Mapping)
        and continuation_terminal.get("schema")
        == "paper_i_signed_active_prefix_checkpoint_binding_v1"
    ):
        candidate_terminal = _mapping(
            final.get("terminal_active_prefix_checkpoint"),
            name="RA full terminal active-prefix checkpoint",
        )
        controller_noise = candidate_terminal.get("controller_noise")
        if (
            set(continuation_terminal) != {"schema", "checkpoint_sha256"}
            or continuation_terminal.get("checkpoint_sha256")
            != candidate_terminal.get("checkpoint_sha256")
            or not isinstance(controller_noise, Mapping)
            or controller_noise.get("schema")
            != "paper_i_pure_hubbard_controller_noise_checkpoint_v1"
        ):
            raise ValueError(
                "RA terminal active-prefix checkpoint binding is invalid."
            )
        terminal_source = candidate_terminal
    terminal_checkpoint = _verify_signed(
        terminal_source,
        name="RA terminal active-prefix checkpoint",
        signature_field="checkpoint_sha256",
    )
    if (
        terminal_checkpoint.get("schema")
        != "paper_i_signed_active_prefix_checkpoint_v1"
        or terminal_checkpoint.get("sr_route_profile")
        != getattr(run.route, "profile", None)
        or terminal_checkpoint.get("sr_route_profile_contract_sha256")
        != getattr(run.route, "contract_sha256", None)
    ):
        raise ValueError(
            "RA terminal checkpoint is not bound to its result route."
        )
    ledger_receipts = _sequence(
        continuation.get(
            "all_active_prefix_estimator_ledger_receipts"
        ),
        name="RA continuation estimator-prefix receipts",
    )
    ledger_closure = _mapping(
        continuation.get("active_prefix_estimator_ledger_closure"),
        name="RA continuation estimator-prefix closure",
    )
    if (
        ledger_closure.get("schema")
        != "paper_i_active_prefix_estimator_ledger_closure_v1"
        or ledger_closure.get("enabled") is not True
        or ledger_closure.get("status") != "complete"
        or ledger_closure.get("passed") is not True
        or int(ledger_closure.get("receipt_count", -1))
        != len([row for row in ledger_receipts if row.get("enabled") is True])
    ):
        raise ValueError(
            "RA active-prefix estimator sidecar closure is incomplete."
        )
    phase3_no_positive_terminal = (
        _build_ra_phase3_no_positive_replay_terminal(
            protocol=protocol,
            run=run,
            finalization=final,
            continuation=continuation,
            prefixes=wrappers,
            terminal_checkpoint=terminal_checkpoint,
            estimator_prefix_receipts=ledger_receipts,
        )
    )
    terminal_controller_outcome = (
        "phase0_stationary_no_competitive_candidate_v1"
        if stationary_without_prefix
        else (
            ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
            if phase3_no_positive_terminal is not None
            else None
        )
    )
    route_identity = _ra_route_identity(run)
    scientific_input_sha256 = _scientific_input_identity(
        protocol=protocol,
        method_family=_RA_METHOD_FAMILY,
        route_identity=route_identity,
    )
    run_trajectory = [
        _receipt_payload(row, name="RA accepted trajectory row")
        for row in getattr(run, "accepted_trajectory", ())
    ]
    run_transitions = [
        _receipt_payload(row, name="RA accepted transition row")
        for row in getattr(run, "accepted_transitions", ())
    ]
    run_replay = [
        _receipt_payload(row, name="RA scientific replay row")
        for row in getattr(run, "scientific_replay", ())
    ]
    replay_identity = _signed(
        {
            "schema": BOUNDED_REPLAY_IDENTITY_SCHEMA,
            "method_family": _RA_METHOD_FAMILY,
            "scientific_input_sha256": scientific_input_sha256,
            "accepted_prefix_replay_identity_sha256s": [
                row["prefix_replay_identity_sha256"] for row in wrappers
            ],
            "signed_controller_prefix_sha256s": [
                row["sha256"] for row in wrappers
            ],
            "accepted_trajectory_sha256": canonical_sha256(run_trajectory),
            "accepted_transitions_sha256": canonical_sha256(
                run_transitions
            ),
            "scientific_replay_sha256": canonical_sha256(run_replay),
        }
    )
    observation_artifacts = [
        _receipt_payload(row, name="RA observation artifact")
        for row in getattr(
            getattr(run, "observation", None),
            "artifacts",
            (),
        )
    ]
    artifact_by_kind = {
        str(row.get("kind")): row for row in observation_artifacts
    }
    checkpoint_artifact = artifact_by_kind.get(
        "accepted_state_checkpoint"
    )
    estimator_artifact = artifact_by_kind.get("estimator_ledger")
    resume_closure = _signed(
        {
            "schema": RESUME_SIDECAR_CLOSURE_SCHEMA,
            "method_family": _RA_METHOD_FAMILY,
            "resume_mode": (
                "not_applicable_phase0_stationary_v1"
                if stationary_without_prefix
                else (
                    "not_applicable_phase3_natural_terminal_v1"
                    if phase3_no_positive_terminal is not None
                    else "canonical_accepted_state_resume_v1"
                )
            ),
            "public_resume_execution_supported": bool(
                not stationary_without_prefix
                and phase3_no_positive_terminal is None
            ),
            "problem_request_sha256": _problem_request_sha256(protocol),
            "protocol_sha256": _require_sha256(
                getattr(protocol, "sha256", ""),
                name="protocol.sha256",
            ),
            "route_identity": route_identity,
            "terminal_signed_prefix_checkpoint_sha256": (
                terminal_checkpoint["checkpoint_sha256"]
            ),
            "terminal_signed_prefix_checkpoint": terminal_checkpoint,
            "all_estimator_prefix_receipts_sha256": canonical_sha256(
                ledger_receipts
            ),
            "all_estimator_prefix_receipts": ledger_receipts,
            "estimator_prefix_closure_sha256": canonical_sha256(
                ledger_closure
            ),
            "estimator_prefix_closure": ledger_closure,
            "signed_prefix_count": len(wrappers),
            "checkpoint_artifact": checkpoint_artifact,
            "estimator_ledger_artifact": estimator_artifact,
            "checkpoint_artifact_available": (
                checkpoint_artifact is not None
            ),
            "estimator_ledger_artifact_available": (
                estimator_artifact is not None
            ),
            "authentication_binding_complete": True,
            "terminal_controller_outcome": terminal_controller_outcome,
            **(
                {
                    "phase3_no_positive_terminal_sha256": (
                        phase3_no_positive_terminal["sha256"]
                    )
                }
                if phase3_no_positive_terminal is not None
                else {}
            ),
        }
    )
    return _signed(
        {
            "schema": CONTROLLER_REPLAY_EVIDENCE_SCHEMA,
            "method_family": _RA_METHOD_FAMILY,
            "protocol_sha256": _require_sha256(
                getattr(protocol, "sha256", ""),
                name="protocol.sha256",
            ),
            "problem_request_sha256": _problem_request_sha256(protocol),
            "signed_controller_round_prefixes": wrappers,
            "bounded_replay_identity": replay_identity,
            "resume_sidecar_closure": resume_closure,
            "terminal_controller_outcome": terminal_controller_outcome,
            **(
                {
                    "phase3_no_positive_terminal": (
                        phase3_no_positive_terminal
                    )
                }
                if phase3_no_positive_terminal is not None
                else {}
            ),
        }
    )


def validate_controller_replay_evidence(
    value: Any,
) -> dict[str, Any]:
    """Fail closed on a serialized RA or Append replay-evidence receipt."""

    evidence = _verify_signed(
        value,
        name="controller replay evidence",
    )
    if evidence.get("schema") != CONTROLLER_REPLAY_EVIDENCE_SCHEMA:
        raise ValueError("Unknown controller replay-evidence schema.")
    method = str(evidence.get("method_family", ""))
    if method not in {_RA_METHOD_FAMILY, _APPEND_METHOD_FAMILY}:
        raise ValueError("Unknown controller replay-evidence method family.")
    protocol_sha256 = _require_sha256(
        evidence.get("protocol_sha256"),
        name="controller replay evidence protocol_sha256",
    )
    problem_sha256 = _require_sha256(
        evidence.get("problem_request_sha256"),
        name="controller replay evidence problem_request_sha256",
    )
    prefixes = _sequence(
        evidence.get("signed_controller_round_prefixes"),
        name="signed_controller_round_prefixes",
    )
    terminal_controller_outcome = evidence.get(
        "terminal_controller_outcome"
    )
    stationary_without_prefix = bool(
        method == _RA_METHOD_FAMILY
        and not prefixes
        and terminal_controller_outcome
        == "phase0_stationary_no_competitive_candidate_v1"
    )
    phase3_no_positive_terminal = bool(
        method == _RA_METHOD_FAMILY
        and terminal_controller_outcome
        == ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
        and isinstance(
            evidence.get("phase3_no_positive_terminal"),
            Mapping,
        )
    )
    if method == _RA_METHOD_FAMILY and terminal_controller_outcome not in {
        None,
        "phase0_stationary_no_competitive_candidate_v1",
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
    }:
        raise ValueError("Unknown RA replay terminal controller outcome.")
    if not prefixes and method == _RA_METHOD_FAMILY and not (
        stationary_without_prefix or phase3_no_positive_terminal
    ):
        raise ValueError(
            "RA controller replay evidence requires signed prefixes."
        )
    previous: str | None = None
    rounds: list[int] = []
    prefix_identity_sha256s: list[str] = []
    prefix_sha256s: list[str] = []
    for index, raw in enumerate(prefixes):
        prefix = _verify_signed(
            raw,
            name=f"signed_controller_round_prefixes[{index}]",
        )
        if (
            prefix.get("schema") != SIGNED_CONTROLLER_PREFIX_SCHEMA
            or prefix.get("method_family") != method
            or prefix.get("protocol_sha256") != protocol_sha256
            or prefix.get("problem_request_sha256") != problem_sha256
            or prefix.get("preceding_signed_prefix_sha256") != previous
        ):
            raise ValueError(
                "Signed controller prefix binding or chain is invalid."
            )
        rounds.append(
            _require_positive_round(
                prefix.get("controller_round"),
                name=f"signed prefix {index} controller_round",
            )
        )
        prefix_identity_sha256s.append(
            _require_sha256(
                prefix.get("prefix_replay_identity_sha256"),
                name=f"signed prefix {index} replay identity",
            )
        )
        prefix_sha256s.append(str(prefix["sha256"]))
        checkpoint = _mapping(
            prefix.get("active_prefix_checkpoint"),
            name=f"signed prefix {index} checkpoint",
        )
        _verify_signed(
            checkpoint,
            name=f"signed prefix {index} checkpoint",
            signature_field="checkpoint_sha256",
        )
        if checkpoint.get("checkpoint_sha256") != prefix.get(
            "source_checkpoint_sha256"
        ):
            raise ValueError(
                "Signed controller prefix source-checkpoint binding drifted."
            )
        route_identity = _mapping(
            prefix.get("route_identity"),
            name=f"signed prefix {index} route_identity",
        )
        if method == _RA_METHOD_FAMILY:
            if (
                checkpoint.get("schema")
                != "paper_i_signed_active_prefix_checkpoint_v1"
                or checkpoint.get("sr_route_profile")
                != route_identity.get("profile")
                or checkpoint.get(
                    "sr_route_profile_contract_sha256"
                )
                != route_identity.get("route_contract_sha256")
            ):
                raise ValueError(
                    "RA controller prefix checkpoint route binding drifted."
                )
            accepted_state = _mapping(
                prefix.get("accepted_state"),
                name=f"signed prefix {index} accepted_state",
            )
            if (
                canonical_sha256(accepted_state)
                != prefix.get("accepted_state_sha256")
                or checkpoint.get("projective_state_fingerprint")
                != accepted_state.get("projective_state_fingerprint")
                or list(
                    checkpoint.get(
                        "signed_unwrapped_logical_parameters", ()
                    )
                )
                != list(accepted_state.get("logical_parameters", ()))
                or list(
                    checkpoint.get(
                        "signed_unwrapped_runtime_parameters", ()
                    )
                )
                != list(accepted_state.get("runtime_parameters", ()))
                or list(
                    checkpoint.get(
                        "ordered_active_operator_labels", ()
                    )
                )
                != list(accepted_state.get("operators", ()))
            ):
                raise ValueError(
                    "RA controller prefix accepted-state binding drifted."
                )
            replay_operator_labels = accepted_state.get("operators", ())
            replay_logical_parameters = accepted_state.get(
                "logical_parameters", ()
            )
            replay_runtime_parameters = accepted_state.get(
                "runtime_parameters", ()
            )
            replay_state_fingerprint = accepted_state.get(
                "projective_state_fingerprint", ""
            )
            replay_energy = accepted_state.get("energy")
        else:
            if checkpoint.get("schema") != APPEND_SIGNED_PREFIX_SCHEMA:
                raise ValueError(
                    "Append controller prefix checkpoint schema drifted."
                )
            replay_operator_labels = checkpoint.get(
                "accepted_operator_labels", ()
            )
            replay_logical_parameters = checkpoint.get(
                "logical_parameters", ()
            )
            replay_runtime_parameters = checkpoint.get(
                "runtime_parameters", ()
            )
            replay_state_fingerprint = checkpoint.get(
                "projective_state_fingerprint", ""
            )
            replay_energy = checkpoint.get("accepted_energy")
        expected_prefix_identity = _prefix_replay_identity(
            method_family=method,
            problem_request_sha256=problem_sha256,
            route_identity=route_identity,
            controller_round=rounds[-1],
            operator_labels=replay_operator_labels,
            logical_parameters=replay_logical_parameters,
            runtime_parameters=replay_runtime_parameters,
            state_fingerprint=str(replay_state_fingerprint),
            accepted_energy=float(replay_energy),
        )
        if expected_prefix_identity != prefix_identity_sha256s[-1]:
            raise ValueError(
                "Signed controller prefix replay identity drifted."
            )
        previous = str(prefix["sha256"])
    if any(
        right != left + 1 for left, right in zip(rounds, rounds[1:])
    ):
        raise ValueError("Signed controller prefix rounds are not contiguous.")

    replay = _verify_signed(
        evidence.get("bounded_replay_identity"),
        name="bounded replay identity",
    )
    if (
        replay.get("schema") != BOUNDED_REPLAY_IDENTITY_SCHEMA
        or replay.get("method_family") != method
        or replay.get("accepted_prefix_replay_identity_sha256s")
        != prefix_identity_sha256s
        or replay.get("signed_controller_prefix_sha256s")
        != prefix_sha256s
    ):
        raise ValueError("Bounded replay identity does not close.")
    _require_sha256(
        replay.get("scientific_input_sha256"),
        name="bounded replay scientific_input_sha256",
    )
    resume = _verify_signed(
        evidence.get("resume_sidecar_closure"),
        name="resume sidecar closure",
    )
    if (
        resume.get("schema") != RESUME_SIDECAR_CLOSURE_SCHEMA
        or resume.get("method_family") != method
        or resume.get("protocol_sha256") != protocol_sha256
        or resume.get("problem_request_sha256") != problem_sha256
        or int(resume.get("signed_prefix_count", -1)) != len(prefixes)
    ):
        raise ValueError("Resume-sidecar closure does not close.")
    if method == _RA_METHOD_FAMILY:
        expected_terminal_outcome = (
            "phase0_stationary_no_competitive_candidate_v1"
            if stationary_without_prefix
            else (
                ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
                if phase3_no_positive_terminal
                else None
            )
        )
        expected_resume_mode = (
            "not_applicable_phase0_stationary_v1"
            if stationary_without_prefix
            else (
                "not_applicable_phase3_natural_terminal_v1"
                if phase3_no_positive_terminal
                else "canonical_accepted_state_resume_v1"
            )
        )
        expected_resume_supported = bool(
            not stationary_without_prefix
            and not phase3_no_positive_terminal
        )
        if (
            resume.get("public_resume_execution_supported")
            is not expected_resume_supported
            or resume.get("resume_mode") != expected_resume_mode
            or resume.get("authentication_binding_complete") is not True
            or resume.get("terminal_controller_outcome")
            != expected_terminal_outcome
        ):
            raise ValueError(
                "RA replay evidence lacks authenticated resume closure."
            )
        terminal = _verify_signed(
            resume.get("terminal_signed_prefix_checkpoint"),
            name="RA resume terminal signed prefix checkpoint",
            signature_field="checkpoint_sha256",
        )
        if terminal.get("checkpoint_sha256") != resume.get(
            "terminal_signed_prefix_checkpoint_sha256"
        ):
            raise ValueError(
                "RA terminal signed-prefix checkpoint binding drifted."
            )
        estimator_receipts = _sequence(
            resume.get("all_estimator_prefix_receipts"),
            name="RA resume estimator-prefix receipts",
        )
        estimator_closure = _mapping(
            resume.get("estimator_prefix_closure"),
            name="RA resume estimator-prefix closure",
        )
        if (
            canonical_sha256(estimator_receipts)
            != resume.get("all_estimator_prefix_receipts_sha256")
            or canonical_sha256(estimator_closure)
            != resume.get("estimator_prefix_closure_sha256")
            or estimator_closure.get("passed") is not True
        ):
            raise ValueError(
                "RA estimator-prefix resume sidecar binding drifted."
            )
        raw_phase3_terminal = evidence.get("phase3_no_positive_terminal")
        resume_phase3_sha256 = resume.get(
            "phase3_no_positive_terminal_sha256"
        )
        if phase3_no_positive_terminal:
            terminal_phase3 = _validate_ra_phase3_no_positive_replay_terminal(
                raw_phase3_terminal,
                prefixes=prefixes,
                terminal_checkpoint=terminal,
                estimator_prefix_receipts=estimator_receipts,
            )
            if resume_phase3_sha256 != terminal_phase3["sha256"]:
                raise ValueError(
                    "RA Phase-III replay terminal is detached from resume "
                    "closure."
                )
        elif (
            raw_phase3_terminal is not None
            or resume_phase3_sha256 is not None
        ):
            raise ValueError(
                "Non-Phase-III replay evidence contains a Phase-III "
                "terminal receipt."
            )
    else:
        if (
            resume.get("public_resume_execution_supported") is not False
            or resume.get("reconstruction_fields_complete") is not True
            or resume.get("continuation_execution_status")
            != "not_authorized_append_resume_contract"
            or resume.get("zero_acceptance_terminal") is not (
                len(prefixes) == 0
            )
        ):
            raise ValueError(
                "Append replay evidence misstates its reconstruction-only "
                "resume boundary."
            )
    return evidence


def bounded_prefix_replay_identity(
    value: Any,
    *,
    controller_round: int,
) -> str:
    """Return the method-neutral identity compared by bounded replay checks."""

    evidence = validate_controller_replay_evidence(value)
    requested = _require_positive_round(
        controller_round,
        name="controller_round",
    )
    for prefix in evidence["signed_controller_round_prefixes"]:
        if int(prefix["controller_round"]) == requested:
            return str(prefix["prefix_replay_identity_sha256"])
    raise ValueError(
        f"Controller replay evidence has no round {requested} prefix."
    )


def compare_bounded_controller_replays(
    first: Any,
    second: Any,
    *,
    controller_round: int,
) -> dict[str, Any]:
    """Authenticate two independent executions of one scientific prefix."""

    first_evidence = validate_controller_replay_evidence(first)
    second_evidence = validate_controller_replay_evidence(second)
    first_method = str(first_evidence["method_family"])
    second_method = str(second_evidence["method_family"])
    if first_method != second_method:
        raise ValueError(
            "Bounded replay comparison requires the same method family."
        )
    first_replay = _mapping(
        first_evidence["bounded_replay_identity"],
        name="first bounded replay identity",
    )
    second_replay = _mapping(
        second_evidence["bounded_replay_identity"],
        name="second bounded replay identity",
    )
    first_scientific_input = _require_sha256(
        first_replay.get("scientific_input_sha256"),
        name="first replay scientific_input_sha256",
    )
    second_scientific_input = _require_sha256(
        second_replay.get("scientific_input_sha256"),
        name="second replay scientific_input_sha256",
    )
    if first_scientific_input != second_scientific_input:
        raise ValueError(
            "Bounded replay executions use different scientific inputs."
        )
    round_index = _require_positive_round(
        controller_round,
        name="controller_round",
    )
    first_prefix = bounded_prefix_replay_identity(
        first_evidence,
        controller_round=round_index,
    )
    second_prefix = bounded_prefix_replay_identity(
        second_evidence,
        controller_round=round_index,
    )
    if first_prefix != second_prefix:
        raise ValueError(
            "Bounded replay accepted-prefix identities do not match."
        )
    return _signed(
        {
            "schema": BOUNDED_REPLAY_COMPARISON_SCHEMA,
            "method_family": first_method,
            "controller_round": round_index,
            "scientific_input_sha256": first_scientific_input,
            "first_controller_replay_evidence_sha256": (
                first_evidence["sha256"]
            ),
            "second_controller_replay_evidence_sha256": (
                second_evidence["sha256"]
            ),
            "first_prefix_replay_identity_sha256": first_prefix,
            "second_prefix_replay_identity_sha256": second_prefix,
            "matched": True,
        }
    )


__all__ = [
    "APPEND_SIGNED_PREFIX_SCHEMA",
    "BOUNDED_REPLAY_COMPARISON_SCHEMA",
    "BOUNDED_REPLAY_IDENTITY_SCHEMA",
    "CONTROLLER_REPLAY_EVIDENCE_SCHEMA",
    "RA_PHASE3_NO_POSITIVE_REPLAY_TERMINAL_SCHEMA",
    "RESUME_SIDECAR_CLOSURE_SCHEMA",
    "SIGNED_CONTROLLER_PREFIX_SCHEMA",
    "bounded_prefix_replay_identity",
    "build_append_controller_replay_evidence",
    "build_ra_controller_replay_evidence",
    "build_signed_append_prefix_checkpoint",
    "compare_bounded_controller_replays",
    "validate_controller_replay_evidence",
]
