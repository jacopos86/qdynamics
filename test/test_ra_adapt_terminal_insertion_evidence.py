from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import pytest

from pipelines.static_adapt import adapt_pipeline
from pipelines.static_adapt.ra_adapt import engine as ra_engine
from pipelines.static_adapt.ra_adapt.semantic_closure_routes import (
    build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request,
    build_paper_i_ra_hh_regime_problem,
    materialize_paper_i_ra_semantic_protocol,
)
from pipelines.static_adapt.sr_snake.contracts import (
    AlwaysCommutationReducedInsertion,
)


def _run_terminal_attempt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    insertion_policy: str,
    terminal_attempt: int,
    always_open: bool = False,
) -> tuple[dict[str, Any], int]:
    monkeypatch.setenv("STATIC_ADAPT_CANDIDATE_RECORD_CACHE", "off")
    monkeypatch.setenv("STATIC_ADAPT_HH_POOL_CACHE", "off")
    monkeypatch.setattr(adapt_pipeline, "_ai_log", lambda *_args, **_kwargs: None)

    real_shortlist = adapt_pipeline._adaptive_phase_shortlist_with_receipt
    phase3_calls = 0

    def force_terminal_phase3(
        records: Sequence[Mapping[str, Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal phase3_calls
        if kwargs.get("phase") == "phase_iii":
            phase3_calls += 1
            if phase3_calls == terminal_attempt:
                assert records
                score_key = str(kwargs["score_key"])
                for record in records:
                    assert isinstance(record, dict)
                    record[score_key] = 0.0
        return real_shortlist(records, *args, **kwargs)

    monkeypatch.setattr(
        adapt_pipeline,
        "_adaptive_phase_shortlist_with_receipt",
        force_terminal_phase3,
    )
    problem = build_paper_i_ra_hh_regime_problem("weak_weak")
    request = build_paper_i_ra_all_phase_position_adaptive_natural_terminal_request(
        insertion_policy=insertion_policy,
        maximum_controller_rounds=terminal_attempt,
    )
    if always_open:
        request = replace(
            request,
            method=replace(
                request.method,
                insertion=AlwaysCommutationReducedInsertion(),
            ),
        )
    protocol = materialize_paper_i_ra_semantic_protocol(problem, request)
    result = ra_engine.run_ra_adapt(problem, protocol)
    terminal = result.scientific_receipts[
        "terminal_phase3_selection_receipt"
    ]
    assert terminal["attempted_controller_round"] == terminal_attempt
    assert result.run.stop.completed_controller_rounds == terminal_attempt - 1
    assert len(result.scientific_receipts["accepted_round_receipts"]) == (
        terminal_attempt - 1
    )
    return terminal, phase3_calls


def test_terminal_attempt_binds_plateau_closed_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal, phase3_calls = _run_terminal_attempt(
        monkeypatch,
        insertion_policy="plateau_commutation",
        terminal_attempt=2,
    )

    assert phase3_calls == 2
    assert terminal["insertion_mode"] == "insertion_commutation_plateau_v2"
    plateau = terminal["insertion_commutation_plateau"]
    assert terminal["insertion_commutation_reduced"] is None
    assert plateau["policy"] == "insertion_commutation_plateau_v2"
    assert plateau["domain_state"] == "closed"
    assert plateau["domain_open"] is False
    assert plateau["requested_positions"] == [1]
    assert terminal["phase3_population_activation"][
        "competitive_population_live"
    ] is True
    assert terminal["controller_measurement_work_proxy"]["events_count"] > 0


def test_terminal_attempt_binds_plateau_open_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapt_pipeline,
        "INSERTION_COMMUTATION_PLATEAU_PRIOR_MEAN_DECREASE_RATIO_THRESHOLD",
        100.0,
    )
    terminal, phase3_calls = _run_terminal_attempt(
        monkeypatch,
        insertion_policy="plateau_commutation",
        terminal_attempt=3,
    )

    assert phase3_calls == 3
    assert terminal["insertion_mode"] == "insertion_commutation_plateau_v2"
    plateau = terminal["insertion_commutation_plateau"]
    assert terminal["insertion_commutation_reduced"] is None
    assert plateau["domain_state"] == "open"
    assert plateau["domain_open"] is True
    assert plateau["requested_positions"] == [0, 1, 2]
    assert plateau["candidate_count"] > 0
    assert plateau["retained_representative_count"] > 0
    assert terminal["controller_measurement_work_proxy"]["events_count"] > 0


def test_terminal_attempt_binds_always_open_reduced_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal, phase3_calls = _run_terminal_attempt(
        monkeypatch,
        insertion_policy="append_only",
        terminal_attempt=2,
        always_open=True,
    )

    assert phase3_calls == 2
    assert terminal["insertion_mode"] == "full_commutation_reduced"
    assert terminal["insertion_commutation_plateau"] is None
    reduced = terminal["insertion_commutation_reduced"]
    assert reduced["policy"] == "always_commutation_reduced"
    assert reduced["domain_state"] == "open"
    assert reduced["domain_open"] is True
    assert reduced["requested_positions"] == [0, 1]
    assert terminal["controller_measurement_work_proxy"]["events_count"] > 0
