from __future__ import annotations

import numpy as np
import pytest

from paper5.stability import DimerParameters
from paper5.stability.exact_reference import _build_exact_dimer_model, _ground_state
from paper5.stability.reachability_observability import (
    build_drive_aware_word_envelope,
    drive_aware_word_hankel_rank_audit,
)


def test_uncapped_component_word_envelope_matches_nested_rank_audit() -> None:
    envelope = build_drive_aware_word_envelope(
        DimerParameters(lambda_ep=1.5, gamma=0.5),
        phonon_cutoff=2,
        maximum_word_depth=2,
        rank_tolerance=1e-10,
    )

    assert envelope.layer_dimensions == (19, 19, 36)
    assert envelope.cumulative_dimensions == (19, 38, 74)
    assert len(envelope.hidden_observables) == 74


def test_preparation_residual_enters_word_envelope_as_initial_slip() -> None:
    parameters = DimerParameters(lambda_ep=1.5, gamma=0.5)
    model = _build_exact_dimer_model(parameters, phonon_cutoff=2)
    _, ground_state = _ground_state(model, eigensolver_tolerance=1e-12)

    envelope = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=2,
        maximum_word_depth=2,
        rank_tolerance=1e-10,
        preparation_state_vectors=(ground_state,),
    )
    rescaled = build_drive_aware_word_envelope(
        parameters,
        phonon_cutoff=2,
        maximum_word_depth=0,
        rank_tolerance=1e-10,
        preparation_state_vectors=(3.0j * ground_state,),
    )

    assert envelope.layer_dimensions == (20, 21, 40)
    assert envelope.cumulative_dimensions == (20, 41, 81)
    assert rescaled.layer_dimensions == (20,)


def test_drive_aware_word_rank_is_nested_and_matches_small_cutoff() -> None:
    result = drive_aware_word_hankel_rank_audit(
        DimerParameters(lambda_ep=1.5, gamma=0.5),
        phonon_cutoff=2,
        maximum_word_depth=2,
        rank_tolerance=1e-10,
        practical_hidden_budget=200,
    )

    assert result.new_ranks == (19, 19, 36)
    assert result.cumulative_ranks == (19, 38, 74)
    assert not result.crossed_budget
    assert result.first_budget_crossing_depth is None
    assert result.drive_force_norm < 1e-12 * result.static_force_norm
    assert all(
        np.all(np.asarray(values) > 0.0)
        for values in result.frontier_singular_values
    )


def test_word_rank_stops_at_first_budget_crossing() -> None:
    result = drive_aware_word_hankel_rank_audit(
        DimerParameters(lambda_ep=1.5, gamma=0.5),
        phonon_cutoff=2,
        maximum_word_depth=5,
        rank_tolerance=1e-10,
        practical_hidden_budget=50,
    )

    assert result.cumulative_ranks == (19, 38, 74)
    assert result.crossed_budget
    assert result.first_budget_crossing_depth == 2
    assert result.hankel_rank_lower_bound == 74


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("phonon_cutoff", 1),
        ("maximum_word_depth", -1),
        ("rank_tolerance", 0.0),
        ("practical_hidden_budget", 0),
        ("drive_component_scale", 0.0),
    ),
)
def test_word_rank_rejects_invalid_settings(keyword: str, value: float) -> None:
    arguments = {
        "phonon_cutoff": 2,
        "maximum_word_depth": 1,
        "rank_tolerance": 1e-10,
        "practical_hidden_budget": 96,
        "drive_component_scale": 1.0,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError):
        drive_aware_word_hankel_rank_audit(
            DimerParameters(lambda_ep=1.5, gamma=0.5),
            **arguments,
        )
