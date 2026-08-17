from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt import engine as ra_engine
from pipelines.static_adapt.ra_adapt.adaptive_phase_shortlist import (
    ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1,
)
from pipelines.static_adapt.ra_adapt.contracts import (
    RAAdaptOperationalControls,
    canonical_sha256,
)
from pipelines.static_adapt.ra_adapt.replay_evidence import (
    validate_controller_replay_evidence,
)
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    build_paper_i_ra_all_phase_adaptive_natural_terminal_request,
    build_paper_i_ra_all_phase_position_adaptive_request,
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
    validate_semantic_phase3_no_positive_terminal_receipt,
)
from pipelines.static_adapt.sr_snake._resume import (
    CanonicalResumeError,
    load_canonical_accepted_state_resume,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AcceptedStateResume,
    CheckpointObservation,
    EstimatorLedgerObservation,
    SRObservationPolicy,
)


def _without_sha256(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "sha256"}


@pytest.mark.parametrize(
    "request_builder",
    (
        build_paper_i_ra_all_phase_adaptive_natural_terminal_request,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    ),
    ids=("append-endpoint-phase0", "position-record-phase0"),
)
def test_public_run_returns_authenticated_terminal_without_fake_round_two(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    request_builder: Any,
) -> None:
    """A scored, nonempty zero Phase III terminates after accepted round one."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    real_shortlist = adapt_pipeline._adaptive_phase_shortlist_with_receipt
    phase3_input_counts: list[int] = []
    forced_phase3_scores: list[tuple[float, ...]] = []

    def force_second_phase3_to_zero(
        records: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("phase") == "phase_iii":
            phase3_input_counts.append(len(records))
            if len(phase3_input_counts) == 2:
                assert records
                score_key = str(kwargs["score_key"])
                for record in records:
                    assert isinstance(record, dict)
                    record[score_key] = 0.0
                forced_phase3_scores.append(
                    tuple(float(record[score_key]) for record in records)
                )
        return real_shortlist(records, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "_adaptive_phase_shortlist_with_receipt",
        force_second_phase3_to_zero,
    )

    checkpoint_path = tmp_path / "phase3-terminal.current.json"
    ledger_path = tmp_path / "phase3-terminal.ledger.json"
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        request_builder(
            insertion_policy="append_only",
            maximum_controller_rounds=3,
        ),
    )

    result = ra_engine.run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=3,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                    every_controller_rounds=1,
                    keep_history_tail=3,
                ),
                estimator_ledger=EstimatorLedgerObservation(path=ledger_path),
                resource_rounds=(1, 2, 3),
            ),
        ),
    )

    assert phase3_input_counts[0] > 0
    assert phase3_input_counts[1] > 0
    assert len(phase3_input_counts) == 2
    assert forced_phase3_scores and set(forced_phase3_scores[0]) == {0.0}

    run = result.run
    assert run.stop.terminal_controller_outcome == (
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    assert run.stop.completed_controller_rounds == 1
    assert tuple(row.controller_round for row in run.accepted_trajectory) == (1,)
    assert tuple(row.controller_round for row in run.accepted_transitions) == (1,)
    assert tuple(row.controller_round for row in run.scientific_replay) == (1,)
    assert len(run.canonical_reporting.accepted_prefix_work) == 1
    assert run.final_state == run.accepted_trajectory[0]
    assert run.estimator_accounting.all_work.s_alg >= (
        run.canonical_reporting.accepted_prefix_work[0].s_alg
    )

    accepted = result.scientific_receipts["accepted_round_receipts"]
    assert [row["accepted_round_ordinal"] for row in accepted] == [1]
    terminal = result.scientific_receipts[
        "terminal_phase3_selection_receipt"
    ]
    assert terminal["schema"] == (
        "paper_i_ra_phase3_no_positive_selection_terminal_v1"
    )
    assert terminal["terminal_controller_outcome"] == (
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    assert terminal["accepted_controller_round"] == 1
    assert terminal["attempted_controller_round"] == 2
    assert terminal["accepted_state_unchanged"] is True
    assert terminal["final_admission_record_id"] is None
    assert terminal["accepted_state_fingerprint"] == (
        run.final_state.projective_state_fingerprint
    )
    assert terminal["insertion_mode"] == "append_only"
    assert terminal["insertion_commutation_plateau"] is None
    assert terminal["insertion_commutation_reduced"] is None
    activation = terminal["phase3_population_activation"]
    assert activation["schema"] == "ra_phase3_population_activation_receipt_v1"
    assert activation["competitive_population_live"] is True
    controller_work = terminal["controller_measurement_work_proxy"]
    assert controller_work["schema"] == "controller_measurement_work_proxy_v1"
    assert controller_work["events_count"] > 0
    assert controller_work["candidate_work_event_count"] > 0
    assert controller_work["candidate_work_missing_event_count"] == 0
    assert controller_work["controller_numeric_validation_status"] == "ok"
    assert terminal["sha256"] == canonical_sha256(_without_sha256(terminal))

    phase3 = terminal["scored_insertion_position_population"]["phases"][2]
    adaptive = phase3["adaptive_shortlist"]
    assert phase3["phase"] == "phase_iii"
    assert phase3["terminal_outcome"] == (
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    assert phase3["shortlist_count"] == 0
    assert phase3["shortlist_records"] == []
    assert adaptive["input_population_count"] > 0
    assert adaptive["status"] == "no_positive_population"
    assert adaptive["retained_count"] == 0
    assert adaptive["retained_record_ids"] == []
    assert {
        float(row["active_score"])
        for row in phase3["adaptive_population_scores"]
    } == {0.0}

    closure = result.scientific_receipts[
        "semantic_selector_accounting_closure"
    ]
    assert closure["validated_round_count"] == 1
    assert closure["terminal_attempted_controller_round"] == 2
    assert closure["terminal_phase3_selection_receipt_sha256"] == (
        terminal["sha256"]
    )

    summary = run.paper_i_summary
    assert summary is not None
    assert summary.available_controller_rounds == 1
    assert tuple(row.controller_round for row in summary.accepted_error_trace) == (
        1,
    )
    assert tuple(row.controller_round for row in summary.requested_rounds) == (1,)

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    adapt = checkpoint["adapt_vqe"]
    assert len(adapt["history"]) == 1
    assert adapt["terminal_controller_outcome"] == (
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    assert adapt["terminal_phase3_selection_receipt"] == terminal
    terminal_checkpoint = adapt["terminal_active_prefix_checkpoint"]
    assert terminal["terminal_active_prefix_checkpoint_sha256"] == (
        canonical_sha256(terminal_checkpoint)
    )
    assert terminal["terminal_estimator_prefix_receipt_sha256"] == (
        canonical_sha256(terminal["terminal_estimator_prefix_receipt"])
    )

    tampered_terminals: list[dict[str, Any]] = []
    tampered_mode = deepcopy(terminal)
    tampered_mode["insertion_mode"] = "full_commutation_reduced"
    tampered_terminals.append(tampered_mode)
    tampered_plateau = deepcopy(terminal)
    tampered_plateau["insertion_commutation_plateau"] = {}
    tampered_terminals.append(tampered_plateau)
    tampered_reduced = deepcopy(terminal)
    tampered_reduced["insertion_commutation_reduced"] = {}
    tampered_terminals.append(tampered_reduced)
    tampered_activation = deepcopy(terminal)
    tampered_activation["phase3_population_activation"][
        "competitive_population_live"
    ] = False
    tampered_terminals.append(tampered_activation)
    tampered_work = deepcopy(terminal)
    tampered_work["controller_measurement_work_proxy"]["events_count"] += 1
    tampered_terminals.append(tampered_work)

    for tampered in tampered_terminals:
        tampered["sha256"] = canonical_sha256(_without_sha256(tampered))
        tampered_finalization = deepcopy(adapt)
        tampered_finalization["terminal_phase3_selection_receipt"] = tampered
        tampered_finalization["continuation"][
            "terminal_phase3_selection_receipt"
        ] = tampered
        with pytest.raises(
            ValueError,
            match="Invalid semantic Phase-III no-positive terminal receipt",
        ):
            validate_semantic_phase3_no_positive_terminal_receipt(
                tampered,
                route_variant=closure["route_variant"],
                route_contract=result.scientific_receipts[
                    "resolved_route_contract"
                ],
                expected_route_contract_sha256=(
                    closure["route_contract_sha256"]
                ),
                accepted_round_count=1,
                terminal_active_prefix_checkpoint=terminal_checkpoint,
                finalization=tampered_finalization,
            )

    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    with pytest.raises(
        CanonicalResumeError,
        match="natural terminal.*non-resumable",
    ):
        load_canonical_accepted_state_resume(
            AcceptedStateResume(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
            ),
            expected_problem=problem,
            expected_route_profile=run.route.profile,
            expected_route_contract_sha256=run.route.contract_sha256,
        )
    artifacts = {artifact.kind: artifact for artifact in run.observation.artifacts}
    assert artifacts["accepted_state_checkpoint"].sha256 == checkpoint_sha256
    assert artifacts["estimator_ledger"].size_bytes == ledger_path.stat().st_size


def test_public_v1_route_rejects_zero_score_phase3_population(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The legacy exact-target route keeps its fail-closed Phase-III seam."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    real_shortlist = adapt_pipeline._adaptive_phase_shortlist_with_receipt

    def force_phase3_to_zero(
        records: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("phase") == "phase_iii":
            score_key = str(kwargs["score_key"])
            for record in records:
                assert isinstance(record, dict)
                record[score_key] = 0.0
        return real_shortlist(records, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "_adaptive_phase_shortlist_with_receipt",
        force_phase3_to_zero,
    )
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_request(
            insertion_policy="append_only",
            maximum_controller_rounds=1,
        ),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "Adaptive phase_iii shortlist has no positive feasible candidate"
        ),
    ):
        ra_engine.run_ra_adapt(
            problem,
            protocol,
            operational_controls=RAAdaptOperationalControls(
                maximum_controller_rounds=1,
                observation=SRObservationPolicy(
                    checkpoint=CheckpointObservation(
                        path=tmp_path / "v1.current.json",
                        every_controller_rounds=1,
                        keep_history_tail=1,
                    ),
                    estimator_ledger=EstimatorLedgerObservation(
                        path=tmp_path / "v1.ledger.json"
                    ),
                    resource_rounds=(1,),
                ),
            ),
        )


def test_public_v2_route_authenticates_round_zero_natural_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """First-round Phase-III exhaustion preserves a signed zero prefix."""

    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)
    real_shortlist = adapt_pipeline._adaptive_phase_shortlist_with_receipt
    phase3_input_counts: list[int] = []

    def force_first_phase3_to_zero(
        records: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if kwargs.get("phase") == "phase_iii":
            phase3_input_counts.append(len(records))
            assert len(phase3_input_counts) == 1
            assert records
            score_key = str(kwargs["score_key"])
            for record in records:
                assert isinstance(record, dict)
                record[score_key] = 0.0
        return real_shortlist(records, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "_adaptive_phase_shortlist_with_receipt",
        force_first_phase3_to_zero,
    )
    checkpoint_path = tmp_path / "phase3-round-zero.current.json"
    ledger_path = tmp_path / "phase3-round-zero.ledger.json"
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    protocol = materialize_paper_i_ra_semantic_protocol(
        problem,
        build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
            insertion_policy="append_only",
            maximum_controller_rounds=2,
        ),
    )

    result = ra_engine.run_ra_adapt(
        problem,
        protocol,
        operational_controls=RAAdaptOperationalControls(
            maximum_controller_rounds=2,
            observation=SRObservationPolicy(
                checkpoint=CheckpointObservation(
                    path=checkpoint_path,
                    every_controller_rounds=1,
                    keep_history_tail=2,
                ),
                estimator_ledger=EstimatorLedgerObservation(path=ledger_path),
                resource_rounds=(1, 2),
            ),
        ),
    )

    assert phase3_input_counts and phase3_input_counts[0] > 0
    run = result.run
    assert run.stop.terminal_controller_outcome == (
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    assert run.stop.completed_controller_rounds == 0
    assert run.final_state.controller_round == 0
    assert run.final_state.operators == ()
    assert run.accepted_trajectory == ()
    assert run.accepted_transitions == ()
    assert run.scientific_replay == ()
    assert run.canonical_reporting.accepted_prefix_work == ()
    assert run.paper_i_summary is None
    assert run.estimator_accounting.all_work.s_alg > 0

    assert result.scientific_receipts["accepted_round_receipts"] == []
    terminal = result.scientific_receipts[
        "terminal_phase3_selection_receipt"
    ]
    assert terminal["accepted_controller_round"] == 0
    assert terminal["attempted_controller_round"] == 1
    assert terminal["accepted_operator_count"] == 0
    assert terminal["accepted_state_fingerprint"] == (
        run.final_state.projective_state_fingerprint
    )
    closure = result.scientific_receipts[
        "semantic_selector_accounting_closure"
    ]
    assert closure["validated_round_count"] == 0
    assert closure["terminal_attempted_controller_round"] == 1

    replay = validate_controller_replay_evidence(
        result.scientific_receipts["controller_replay_evidence"]
    )
    assert replay["signed_controller_round_prefixes"] == []
    replay_terminal = replay["phase3_no_positive_terminal"]
    assert replay_terminal["schema"] == (
        "paper_i_ra_phase3_no_positive_controller_replay_terminal_v2"
    )
    assert replay_terminal["accepted_controller_round"] == 0
    assert replay_terminal["attempted_controller_round"] == 1
    assert "accepted_signed_prefix_sha256" not in replay_terminal
    assert replay_terminal["round_zero_accepted_state"] == (
        run.final_state.to_dict()
    )
    assert replay["resume_sidecar_closure"][
        "public_resume_execution_supported"
    ] is False

    tampered_replay = deepcopy(replay)
    tampered_replay_terminal = tampered_replay[
        "phase3_no_positive_terminal"
    ]
    tampered_replay_terminal["round_zero_accepted_state"]["energy"] += 1.0
    tampered_replay_terminal["sha256"] = canonical_sha256(
        _without_sha256(tampered_replay_terminal)
    )
    tampered_replay["resume_sidecar_closure"][
        "phase3_no_positive_terminal_sha256"
    ] = tampered_replay_terminal["sha256"]
    tampered_replay["resume_sidecar_closure"]["sha256"] = canonical_sha256(
        _without_sha256(tampered_replay["resume_sidecar_closure"])
    )
    tampered_replay["sha256"] = canonical_sha256(
        _without_sha256(tampered_replay)
    )
    with pytest.raises(ValueError, match="accepted-state digest"):
        validate_controller_replay_evidence(tampered_replay)

    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    adapt = checkpoint["adapt_vqe"]
    assert adapt["history"] == []
    assert adapt["terminal_controller_outcome"] == (
        ADAPTIVE_PHASE3_NO_POSITIVE_TERMINAL_OUTCOME_V1
    )
    assert adapt["terminal_phase3_selection_receipt"] == terminal
    assert adapt["terminal_active_prefix_checkpoint"]["outer_iteration"] == 0
    assert adapt["terminal_active_prefix_checkpoint"]["active_ansatz_depth"] == 0
    assert "verified_singleton_resume_sidecar" not in adapt

    checkpoint_sha256 = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    with pytest.raises(
        CanonicalResumeError,
        match="natural terminal.*non-resumable",
    ):
        load_canonical_accepted_state_resume(
            AcceptedStateResume(
                checkpoint_path=checkpoint_path,
                checkpoint_sha256=checkpoint_sha256,
            ),
            expected_problem=problem,
            expected_route_profile=run.route.profile,
            expected_route_contract_sha256=run.route.contract_sha256,
        )

    tampered_checkpoint = deepcopy(checkpoint)
    tampered_terminal = tampered_checkpoint["adapt_vqe"][
        "terminal_phase3_selection_receipt"
    ]
    tampered_terminal["accepted_operator_count"] = 1
    tampered_terminal["sha256"] = canonical_sha256(
        _without_sha256(tampered_terminal)
    )
    tampered_checkpoint["adapt_vqe"]["continuation"][
        "terminal_phase3_selection_receipt"
    ] = deepcopy(tampered_terminal)
    tampered_path = tmp_path / "phase3-round-zero.tampered.current.json"
    tampered_path.write_text(
        json.dumps(tampered_checkpoint, sort_keys=True),
        encoding="utf-8",
    )
    tampered_sha256 = hashlib.sha256(tampered_path.read_bytes()).hexdigest()
    with pytest.raises(
        CanonicalResumeError,
        match="natural-terminal evidence is invalid",
    ):
        load_canonical_accepted_state_resume(
            AcceptedStateResume(
                checkpoint_path=tampered_path,
                checkpoint_sha256=tampered_sha256,
            ),
            expected_problem=problem,
            expected_route_profile=run.route.profile,
            expected_route_contract_sha256=run.route.contract_sha256,
        )
    artifacts = {artifact.kind: artifact for artifact in run.observation.artifacts}
    assert artifacts["accepted_state_checkpoint"].sha256 == checkpoint_sha256
    assert artifacts["estimator_ledger"].size_bytes == ledger_path.stat().st_size
