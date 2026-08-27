"""Behavior locks for the canonical record-first adaptive QSE trace."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pipelines.exact_bench import paper_iii_growth_trace_campaign as growth_campaign
from pipelines.exact_bench.paper_iii_growth_trace_campaign import (
    _mixed_resource_triple,
    trace_record_first_adaptive_qse,
)
from pipelines.reporting.build_paper_iii_growth_fragments import METHODS
from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    pauli_string_basis_element,
)
from pipelines.qse_spectra.paper_iii_problem import load_problem
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def test_adaptive_resource_accounting_separates_record_seed_from_synthesis() -> None:
    resources = _mixed_resource_triple(
        seed_resources={"n2q": 11.0, "d2q": 13.0, "dc": 17.0},
        per_synthesized_direction={"n2q": 5.0, "d2q": 7.0, "dc": 9.0},
        synthesized_direction_count=3,
    )

    assert resources == {"n2q": 26.0, "d2q": 34.0, "dc": 44.0}


def test_growth_report_names_the_canonical_cost_weighted_method() -> None:
    assert METHODS[0] == (
        "record_first_adaptive_qse",
        "cost-weighted QSE",
    )


def test_canonical_dimension_cap_reaches_the_observed_record_exhaustion() -> None:
    assert growth_campaign.DEFAULT_DIMENSION_CAP == 64


def test_adaptive_trace_charges_seed_as_records_only(monkeypatch: pytest.MonkeyPatch) -> None:
    hamiltonian = PauliPolynomial("JW")
    hamiltonian.add_term(PauliTerm(4, ps="eeez", pc=1.0))
    basis = (
        pauli_string_basis_element("IIII", nq=4, name="identity"),
        pauli_string_basis_element("IIIX", nq=4, name="hh_phonon::x"),
    )

    def resource_triple(indices: list[int]) -> dict[str, float]:
        return {
            "n2q": 11.0 * len(indices),
            "d2q": 13.0 * len(indices),
            "dc": 17.0 * len(indices),
        }

    problem = SimpleNamespace(
        basis=basis,
        hamiltonian=hamiltonian,
        ground=np.eye(1, 16, 0, dtype=complex).reshape(-1),
        num_qubits=4,
        references=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        resource_triple=resource_triple,
    )

    def fake_adaptive(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        roots = [0.0, *problem.references]
        return {
            "seed_policy": {
                "seed_rows": [{"admitted": False}, {"admitted": True}],
            },
            "iterations": [
                {"dimension": 2, "retained_rank": 2, "root_energies": roots},
                {"dimension": 4, "retained_rank": 4, "root_energies": roots},
            ],
        }

    monkeypatch.setattr(
        growth_campaign, "run_adaptive_qse_benchmark", fake_adaptive
    )
    rows = growth_campaign.trace_adaptive_qse(problem, k_max=4)

    assert rows[0]["seed_record_count"] == 1
    assert rows[0]["synthesized_direction_count"] == 0
    assert rows[0]["n2q"] == 11.0
    assert rows[1]["synthesized_direction_count"] == 2
    assert rows[1]["n2q"] == 11.0 + 2 * rows[1]["per_synthesized_direction"]["n2q"]


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
def intermediate_strong_record_first_trace(
    intermediate_strong_record_stop: tuple[Any, list[int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    problem, record_indices, record_stop = intermediate_strong_record_stop
    return trace_record_first_adaptive_qse(
        problem,
        record_indices,
        k_max=64,
        stride=len(record_indices),
        record_stop=record_stop,
    )


def test_record_first_handoff_uses_maintained_record_support(
    intermediate_strong_record_first_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    _rows, handoff = intermediate_strong_record_first_trace

    assert handoff["fired"] is True
    assert handoff["method_id"] == "record_first_adaptive_qse_v2"
    assert handoff["reason"] == "record_pool_exhausted"
    assert handoff["phase_a_support_indices"] == handoff["seed_indices"]
    assert handoff["seed_size"] == len(handoff["seed_indices"])


def test_record_first_trigger_is_the_record_selector_terminal_receipt(
    intermediate_strong_record_first_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    _rows, handoff = intermediate_strong_record_first_trace

    assert handoff["trigger_source"] == "record_selector_terminal_receipt"


def test_record_first_charges_only_directions_synthesized_after_handoff(
    intermediate_strong_record_first_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    rows, handoff = intermediate_strong_record_first_trace
    synthesis_rows = [row for row in rows if row.get("phase") == "synthesis"]
    first_synthesis = synthesis_rows[0]

    assert first_synthesis["support_size"] == handoff["initial_adaptive_dimension"] + 1
    for resource in ("n2q", "d2q", "dc"):
        incremental = first_synthesis[resource] - handoff["phase_a_resources"][resource]
        assert incremental == pytest.approx(
            handoff["per_synthesized_direction"][resource]
        )


def test_record_first_resolves_the_seed_orthogonal_triplet_root(
    intermediate_strong_record_first_trace: tuple[list[dict[str, Any]], dict[str, Any]],
) -> None:
    _rows, handoff = intermediate_strong_record_first_trace

    assert handoff["terminal_window_error"] < 1.0e-12
    assert handoff["terminal_root_energies"][3] == pytest.approx(1.0, abs=1.0e-12)


def test_record_first_preserves_an_unresolved_phase_a_window(
    intermediate_strong_record_stop: tuple[Any, list[int], dict[str, Any]],
) -> None:
    problem, record_indices, _record_stop = intermediate_strong_record_stop

    _rows, handoff = trace_record_first_adaptive_qse(
        problem,
        record_indices[:1],
        k_max=2,
        stride=1,
        record_stop={"stop_reason": "pool_exhausted"},
    )

    assert handoff["phase_a_error"] is None


def test_record_first_stops_without_synthesis_after_record_residual_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_row = {
        "k": 1,
        "support_size": 1,
        "support_indices": [3],
        "exchanged": True,
        "retained_rank": 1,
        "max_root_abs_error": 1.0e-8,
        "n2q": 5.0,
        "d2q": 7.0,
        "dc": 11.0,
    }
    monkeypatch.setattr(
        growth_campaign,
        "trace_arm",
        lambda *_args, **_kwargs: [dict(record_row)],
    )

    def fail_if_synthesis_runs(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise AssertionError("residual synthesis must not run after record convergence")

    monkeypatch.setattr(
        growth_campaign,
        "run_adaptive_qse_benchmark",
        fail_if_synthesis_runs,
    )

    rows, handoff = trace_record_first_adaptive_qse(
        SimpleNamespace(),
        [3],
        k_max=4,
        stride=1,
        record_stop={"stop_reason": "residual_converged"},
    )

    assert rows[0]["phase"] == "records"
    assert handoff == {
        "method_id": "record_first_adaptive_qse_v2",
        "fired": False,
        "reason": "record_residual_converged",
        "trigger_source": "record_selector_terminal_receipt",
        "record_stop": {"stop_reason": "residual_converged"},
    }
