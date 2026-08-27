"""Behavior locks for the record-to-residual hybrid QSE trace."""

from __future__ import annotations

from typing import Any

import pytest

from pipelines.exact_bench.paper_iii_growth_trace_campaign import trace_hybrid
from pipelines.qse_spectra.core import QSEBasisVectorPolicy
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)


@pytest.fixture(scope="module")
def intermediate_strong_record_stop() -> tuple[Any, list[int], dict[str, Any]]:
    problem = load_problem(
        regime="intermediate_strong",
        u=1.25,
        g_ep=0.790569415042,
        n_ph_max=3,
        target_roots=6,
    )
    policy = QSEBasisVectorPolicy(
        reference_projection="q0",
        basis_vector_normalization="raw_projected",
    )
    selection = select_static_qse_records(
        problem.basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=len(problem.basis),
            geometry_target_roots=6,
            geometry_cost_discount_alpha=1.0,
            geometry_residual_stop=1.0e-3,
        ),
        hamiltonian=problem.hamiltonian,
        prepared_state=problem.ground,
        basis_vector_policy=policy,
        compiled_costs=problem.costs,
    )
    assert selection.geometry_stop is not None
    return (
        problem,
        [int(index) for index in selection.selected_original_indices],
        dict(selection.geometry_stop),
    )


@pytest.fixture(scope="module")
def intermediate_strong_hybrid_trace(
    intermediate_strong_record_stop: tuple[Any, list[int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    problem, record_indices, record_stop = intermediate_strong_record_stop
    return trace_hybrid(
        problem,
        record_indices,
        k_max=64,
        stride=len(record_indices),
        record_stop=record_stop,
    )


def test_hybrid_handoff_uses_maintained_record_support(
    intermediate_strong_hybrid_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    _rows, handoff = intermediate_strong_hybrid_trace

    assert handoff["fired"] is True
    assert handoff["phase_a_support_indices"] == handoff["seed_indices"]
    assert handoff["seed_size"] == len(handoff["seed_indices"])


def test_hybrid_trigger_is_the_record_selector_terminal_receipt(
    intermediate_strong_hybrid_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    _rows, handoff = intermediate_strong_hybrid_trace

    assert handoff["trigger_source"] == "record_selector_terminal_receipt"


def test_hybrid_charges_only_directions_synthesized_after_handoff(
    intermediate_strong_hybrid_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    rows, handoff = intermediate_strong_hybrid_trace
    synthesis_rows = [row for row in rows if row.get("phase") == "synthesis"]
    first_synthesis = synthesis_rows[0]

    assert first_synthesis["support_size"] == handoff["initial_adaptive_dimension"] + 1
    for resource in ("n2q", "d2q", "dc"):
        incremental = first_synthesis[resource] - handoff["phase_a_resources"][resource]
        assert incremental == pytest.approx(
            handoff["per_synthesized_direction"][resource]
        )


def test_hybrid_resolves_the_seed_orthogonal_triplet_root(
    intermediate_strong_hybrid_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    _rows, handoff = intermediate_strong_hybrid_trace

    assert handoff["terminal_window_error"] < 1.0e-12
    assert handoff["terminal_root_energies"][3] == pytest.approx(1.0, abs=1.0e-12)


def test_hybrid_preserves_an_unresolved_phase_a_window(
    intermediate_strong_record_stop: tuple[Any, list[int], dict[str, Any]],
) -> None:
    problem, record_indices, _record_stop = intermediate_strong_record_stop

    _rows, handoff = trace_hybrid(
        problem,
        record_indices[:1],
        k_max=2,
        stride=1,
        record_stop={"stop_reason": "pool_exhausted"},
    )

    assert handoff["phase_a_error"] is None
