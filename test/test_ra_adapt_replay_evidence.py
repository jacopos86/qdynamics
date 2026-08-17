from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pipelines.contracts.problem import ProblemRequest
from pipelines.static_adapt.builders.problem_registry import (
    resolve_problem_context,
)
from pipelines.static_adapt.ra_adapt import (
    AppendAdaptRequest,
    MacroCandidateAdapter,
    RAAdaptRequest,
    run_append_adapt,
    run_ra_adapt,
)
from pipelines.static_adapt.ra_adapt import engine as ra_engine
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    APPEND_SIGNED_PREFIX_SCHEMA,
    CONTROLLER_REPLAY_EVIDENCE_SCHEMA,
    SIGNED_CONTROLLER_PREFIX_SCHEMA,
    bounded_prefix_replay_identity,
    build_ra_controller_replay_evidence,
    compare_bounded_controller_replays,
    validate_controller_replay_evidence,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2,
    build_paper_i_ra_all_phase_position_adaptive_request,
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    RAAdaptOperationalControls,
)
from pipelines.static_adapt.sr_snake import (
    CheckpointObservation,
    EstimatorLedgerObservation,
    SRExecutionPolicy,
    SRObservationPolicy,
    SRStopPolicy,
)


def _hh_problem() -> Any:
    return resolve_problem_context(
        ProblemRequest(
            problem_key="hh",
            num_sites=2,
            t=1.0,
            u=0.5,
            dv=0.0,
            omega0=1.0,
            g_ep=0.2,
            n_ph_max=1,
            boson_encoding="binary",
            ordering="blocked",
            boundary="open",
        )
    )


def _execution() -> SRExecutionPolicy:
    return SRExecutionPolicy(
        stop=SRStopPolicy(maximum_controller_rounds=1)
    )


def _observation(tmp_path: Path, *, stem: str) -> SRObservationPolicy:
    return SRObservationPolicy(
        checkpoint=CheckpointObservation(
            path=tmp_path / f"{stem}.current.json",
            every_controller_rounds=1,
            keep_history_tail=2,
        ),
        estimator_ledger=EstimatorLedgerObservation(
            path=tmp_path / f"{stem}.ledger.json"
        ),
    )


def _assert_outer_checkpoint_is_signed(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    observed = payload.pop("sha256", None)
    assert observed == canonical_sha256(payload)
    payload["sha256"] = observed
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


_PHASE3_NO_POSITIVE_TERMINAL = (
    "phase_iii_no_positive_feasible_candidate_v1"
)


def _phase3_no_positive_finalization(
    result: Any,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Project one accepted result into an authenticated failed next attempt."""

    evidence = result.scientific_receipts["controller_replay_evidence"]
    prefix = copy.deepcopy(evidence["signed_controller_round_prefixes"][-1])
    checkpoint = copy.deepcopy(prefix["active_prefix_checkpoint"])
    checkpoint.pop("checkpoint_sha256")
    checkpoint["checkpoint_kind"] = "terminal_phase3_no_positive"
    checkpoint["checkpoint_sha256"] = canonical_sha256(checkpoint)

    resume = evidence["resume_sidecar_closure"]
    estimator_receipts = copy.deepcopy(
        resume["all_estimator_prefix_receipts"]
    )
    terminal_estimator_prefix = copy.deepcopy(estimator_receipts[-1])
    terminal_estimator_prefix["checkpoint_kind"] = (
        "terminal_phase3_no_positive"
    )
    terminal_estimator_prefix["checkpoint_sequence"] = (
        len(estimator_receipts) + 1
    )
    estimator_receipts.append(terminal_estimator_prefix)
    estimator_closure = copy.deepcopy(resume["estimator_prefix_closure"])
    estimator_closure["receipt_count"] = len(
        [row for row in estimator_receipts if row.get("enabled") is True]
    )

    accepted_count = len(result.run.accepted_trajectory)
    accepted_state = result.run.final_state
    event_ids = ["terminal:phase_i", "terminal:phase_iii"]
    terminal_receipt = {
        "schema": "paper_i_ra_phase3_no_positive_selection_terminal_v1",
        "terminal_controller_outcome": _PHASE3_NO_POSITIVE_TERMINAL,
        "accepted_controller_round": accepted_count,
        "attempted_controller_round": accepted_count + 1,
        "accepted_state_fingerprint": (
            accepted_state.projective_state_fingerprint
        ),
        "accepted_operator_count": len(accepted_state.operators),
        "accepted_state_unchanged": True,
        "final_admission_record_id": None,
        "phase0_gradient_shortlist": {"fixture": "full_terminal_receipt"},
        "insertion_mode": "append_only",
        "insertion_commutation_plateau": None,
        "insertion_commutation_reduced": None,
        "phase3_population_activation": {
            "competitive_population_live": True,
        },
        "controller_measurement_work_proxy": {
            "schema": "controller_measurement_work_proxy_v1",
        },
        "scored_insertion_position_population": {
            "fixture": "full_terminal_receipt"
        },
        "projected_phase3_population_receipt": {
            "fixture": "full_terminal_receipt"
        },
        "phase123_qiskit_population_normalization_receipts": {
            "fixture": "full_terminal_receipt"
        },
        "estimator_event_ids": event_ids,
        "estimator_event_count": len(event_ids),
        "estimator_event_ids_sha256": canonical_sha256(event_ids),
        "terminal_active_prefix_checkpoint_sha256": canonical_sha256(
            checkpoint
        ),
        "terminal_estimator_prefix_receipt": terminal_estimator_prefix,
        "terminal_estimator_prefix_receipt_sha256": canonical_sha256(
            terminal_estimator_prefix
        ),
    }
    terminal_receipt["sha256"] = canonical_sha256(terminal_receipt)
    route_contract = copy.deepcopy(dict(result.protocol.route_contract))
    route_contract_sha256 = route_contract.pop("sha256")
    finalization = {
        "sr_route_profile_contract": route_contract,
        "sr_route_profile_contract_sha256": route_contract_sha256,
        "history": [{"active_prefix_checkpoint": copy.deepcopy(
            row["active_prefix_checkpoint"]
        )} for row in evidence["signed_controller_round_prefixes"]],
        "terminal_controller_outcome": _PHASE3_NO_POSITIVE_TERMINAL,
        "terminal_phase3_selection_receipt": copy.deepcopy(terminal_receipt),
        "terminal_active_prefix_checkpoint": copy.deepcopy(checkpoint),
        "continuation": {
            "active_prefix_checkpoints": [copy.deepcopy(
                row["active_prefix_checkpoint"]
            ) for row in evidence["signed_controller_round_prefixes"]],
            "terminal_active_prefix_checkpoint": copy.deepcopy(checkpoint),
            "all_active_prefix_estimator_ledger_receipts": estimator_receipts,
            "active_prefix_estimator_ledger_closure": estimator_closure,
            "terminal_phase3_selection_receipt": copy.deepcopy(
                terminal_receipt
            ),
        },
    }
    stop = replace(
        result.run.stop,
        primary_reason="phase_iii_no_positive_feasible_candidate",
        fired_reasons=("phase_iii_no_positive_feasible_candidate",),
        terminal_controller_outcome=_PHASE3_NO_POSITIVE_TERMINAL,
    )
    return replace(result.run, stop=stop), finalization, terminal_receipt


def test_phase3_no_positive_terminal_is_bound_into_public_replay_evidence() -> None:
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=1,
        ),
    )
    accepted = ra_engine.run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=1,
        ),
    )
    terminal_run, finalization, terminal_receipt = (
        _phase3_no_positive_finalization(accepted)
    )

    evidence = validate_controller_replay_evidence(
        build_ra_controller_replay_evidence(
            protocol=accepted.protocol,
            run=terminal_run,
            finalization=finalization,
        )
    )

    terminal = evidence["phase3_no_positive_terminal"]
    terminal_route = terminal["natural_terminal_route_contract"]
    assert terminal_route["native_semantic_contract"]["route_variant"] == (
        PAPER_I_RA_ALL_PHASE_POSITION_ADAPTIVE_SHORTLIST_NATURAL_TERMINAL_V2
    )
    assert terminal["natural_terminal_route_contract_sha256"] == (
        accepted.run.route.contract_sha256
    )
    assert terminal["terminal_controller_outcome"] == (
        _PHASE3_NO_POSITIVE_TERMINAL
    )
    assert terminal["accepted_controller_round"] == 1
    assert terminal["attempted_controller_round"] == 2
    assert terminal["accepted_state_unchanged"] is True
    assert terminal["terminal_phase3_selection_receipt"] == terminal_receipt
    assert terminal["terminal_phase3_selection_receipt"][
        "insertion_mode"
    ] == "append_only"
    assert terminal["terminal_phase3_selection_receipt"][
        "phase3_population_activation"
    ]["competitive_population_live"] is True
    assert terminal["terminal_phase3_selection_receipt_sha256"] == (
        terminal_receipt["sha256"]
    )
    assert terminal["terminal_active_prefix_checkpoint"] == (
        finalization["terminal_active_prefix_checkpoint"]
    )
    assert terminal["terminal_estimator_prefix_receipt"] == (
        terminal_receipt["terminal_estimator_prefix_receipt"]
    )
    assert evidence["resume_sidecar_closure"][
        "phase3_no_positive_terminal_sha256"
    ] == terminal["sha256"]
    assert evidence["resume_sidecar_closure"]["resume_mode"] == (
        "not_applicable_phase3_natural_terminal_v1"
    )
    assert evidence["resume_sidecar_closure"][
        "public_resume_execution_supported"
    ] is False

    tampered = copy.deepcopy(evidence)
    tampered["phase3_no_positive_terminal"][
        "attempted_controller_round"
    ] = 3
    tampered["phase3_no_positive_terminal"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered[
                "phase3_no_positive_terminal"
            ].items()
            if key != "sha256"
        }
    )
    tampered["resume_sidecar_closure"][
        "phase3_no_positive_terminal_sha256"
    ] = tampered["phase3_no_positive_terminal"]["sha256"]
    tampered["resume_sidecar_closure"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in tampered["resume_sidecar_closure"].items()
            if key != "sha256"
        }
    )
    tampered["sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "sha256"}
    )
    with pytest.raises(ValueError, match="attempted controller round"):
        validate_controller_replay_evidence(tampered)

    v1_protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=1,
        ),
    )
    v1_route = copy.deepcopy(dict(v1_protocol.route_contract))
    v1_route_sha256 = v1_route.pop("sha256")
    cross_version = copy.deepcopy(evidence)
    cross_version_terminal = cross_version[
        "phase3_no_positive_terminal"
    ]
    cross_version_terminal["natural_terminal_route_contract"] = v1_route
    cross_version_terminal[
        "natural_terminal_route_contract_sha256"
    ] = v1_route_sha256
    cross_version_terminal["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in cross_version_terminal.items()
            if key != "sha256"
        }
    )
    cross_version["resume_sidecar_closure"][
        "phase3_no_positive_terminal_sha256"
    ] = cross_version_terminal["sha256"]
    cross_version["resume_sidecar_closure"]["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in cross_version[
                "resume_sidecar_closure"
            ].items()
            if key != "sha256"
        }
    )
    cross_version["sha256"] = canonical_sha256(
        {
            key: value
            for key, value in cross_version.items()
            if key != "sha256"
        }
    )
    with pytest.raises(ValueError, match="V2 natural-terminal route"):
        validate_controller_replay_evidence(cross_version)


def test_ra_result_exposes_signed_prefix_and_resume_sidecar_closure(
    tmp_path: Path,
) -> None:
    result = run_ra_adapt(
        _hh_problem(),
        RAAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=_execution(),
            observation=_observation(tmp_path, stem="ra"),
        ),
    )

    evidence = validate_controller_replay_evidence(
        result.scientific_receipts["controller_replay_evidence"]
    )
    assert evidence["schema"] == CONTROLLER_REPLAY_EVIDENCE_SCHEMA
    assert evidence["method_family"] == "ra_adapt"
    assert evidence["sha256"] == result.scientific_receipts[
        "controller_replay_evidence_sha256"
    ]
    prefixes = evidence["signed_controller_round_prefixes"]
    assert len(prefixes) == 1
    assert prefixes[0]["schema"] == SIGNED_CONTROLLER_PREFIX_SCHEMA
    assert prefixes[0]["controller_round"] == 1
    assert prefixes[0]["active_prefix_checkpoint"][
        "strict_replay"
    ]["passed"] is True
    assert bounded_prefix_replay_identity(
        evidence,
        controller_round=1,
    ) == prefixes[0]["prefix_replay_identity_sha256"]
    repeated = run_ra_adapt(
        _hh_problem(),
        RAAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=_execution(),
        ),
    )
    repeated_evidence = validate_controller_replay_evidence(
        repeated.scientific_receipts["controller_replay_evidence"]
    )
    assert bounded_prefix_replay_identity(
        evidence,
        controller_round=1,
    ) == bounded_prefix_replay_identity(
        repeated_evidence,
        controller_round=1,
    )
    assert evidence["bounded_replay_identity"][
        "scientific_input_sha256"
    ] == repeated_evidence["bounded_replay_identity"][
        "scientific_input_sha256"
    ]
    replay_comparison = compare_bounded_controller_replays(
        evidence,
        repeated_evidence,
        controller_round=1,
    )
    assert replay_comparison["matched"] is True
    comparison_digest = replay_comparison.pop("sha256")
    assert comparison_digest == canonical_sha256(replay_comparison)

    resume = evidence["resume_sidecar_closure"]
    assert evidence["terminal_controller_outcome"] is None
    assert "phase3_no_positive_terminal" not in evidence
    assert resume["resume_mode"] == "canonical_accepted_state_resume_v1"
    assert resume["public_resume_execution_supported"] is True
    assert resume["terminal_controller_outcome"] is None
    assert "phase3_no_positive_terminal_sha256" not in resume
    assert resume["authentication_binding_complete"] is True
    assert resume["checkpoint_artifact_available"] is True
    assert resume["estimator_ledger_artifact_available"] is True
    checkpoint_path = tmp_path / "ra.current.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert checkpoint["adapt_vqe"]["active_prefix_checkpoints"]
    assert _file_sha256(checkpoint_path) == resume[
        "checkpoint_artifact"
    ]["sha256"]

    tampered = copy.deepcopy(evidence)
    tampered["signed_controller_round_prefixes"][0][
        "active_prefix_checkpoint"
    ]["signed_unwrapped_logical_parameters"][0] += 0.25
    with pytest.raises(ValueError, match="SHA-256"):
        validate_controller_replay_evidence(tampered)


def test_append_replay_identity_is_deterministic_and_resume_boundary_is_explicit(
    tmp_path: Path,
) -> None:
    problem = _hh_problem()
    request = AppendAdaptRequest(
        adapter=MacroCandidateAdapter(),
        execution=_execution(),
        observation=_observation(tmp_path, stem="append-first"),
    )
    first = run_append_adapt(problem, request)
    second = run_append_adapt(
        problem,
        AppendAdaptRequest(
            adapter=MacroCandidateAdapter(),
            execution=_execution(),
        ),
    )

    first_evidence = validate_controller_replay_evidence(
        first.result_payload["controller_replay_evidence"]
    )
    second_evidence = validate_controller_replay_evidence(
        second.result_payload["controller_replay_evidence"]
    )
    assert first_evidence == first.scientific_receipts[
        "controller_replay_evidence"
    ]
    assert first.scientific_receipts[
        "controller_replay_evidence_sha256"
    ] == first_evidence["sha256"]
    assert bounded_prefix_replay_identity(
        first_evidence,
        controller_round=1,
    ) == bounded_prefix_replay_identity(
        second_evidence,
        controller_round=1,
    )
    assert compare_bounded_controller_replays(
        first_evidence,
        second_evidence,
        controller_round=1,
    )["matched"] is True

    prefix = first_evidence["signed_controller_round_prefixes"][0]
    assert prefix["active_prefix_checkpoint"]["schema"] == (
        APPEND_SIGNED_PREFIX_SCHEMA
    )
    resume = first_evidence["resume_sidecar_closure"]
    assert resume["public_resume_execution_supported"] is False
    assert resume["reconstruction_fields_complete"] is True
    assert resume["continuation_execution_status"] == (
        "not_authorized_append_resume_contract"
    )
    outer_checkpoint = _assert_outer_checkpoint_is_signed(
        tmp_path / "append-first.current.json"
    )
    assert outer_checkpoint["controller_replay_evidence"] == first_evidence
