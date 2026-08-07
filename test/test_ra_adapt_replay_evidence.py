from __future__ import annotations

import copy
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
from pipelines.static_adapt.ra_adapt.contracts import canonical_sha256
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    APPEND_SIGNED_PREFIX_SCHEMA,
    CONTROLLER_REPLAY_EVIDENCE_SCHEMA,
    SIGNED_CONTROLLER_PREFIX_SCHEMA,
    bounded_prefix_replay_identity,
    compare_bounded_controller_replays,
    validate_controller_replay_evidence,
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
    assert resume["public_resume_execution_supported"] is True
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
