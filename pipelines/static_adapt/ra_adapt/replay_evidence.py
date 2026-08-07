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

_RA_METHOD_FAMILY = "ra_adapt"
_APPEND_METHOD_FAMILY = "append_adapt"


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
    if not declared_prefixes:
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

    terminal_checkpoint = _verify_signed(
        continuation.get("terminal_active_prefix_checkpoint"),
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
            "resume_mode": "canonical_accepted_state_resume_v1",
            "public_resume_execution_supported": True,
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
    if not prefixes and method == _RA_METHOD_FAMILY:
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
        if (
            resume.get("public_resume_execution_supported") is not True
            or resume.get("authentication_binding_complete") is not True
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
    "RESUME_SIDECAR_CLOSURE_SCHEMA",
    "SIGNED_CONTROLLER_PREFIX_SCHEMA",
    "bounded_prefix_replay_identity",
    "build_append_controller_replay_evidence",
    "build_ra_controller_replay_evidence",
    "build_signed_append_prefix_checkpoint",
    "compare_bounded_controller_replays",
    "validate_controller_replay_evidence",
]
