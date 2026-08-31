"""Shared Paper-I graph-span resource estimator contracts."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.resource_estimation.pauli_graph_span import (
    PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1,
    compare_pauli_graph_span_ansatzes,
    evaluate_pauli_graph_span_ansatz,
)
from src.quantum.vqe_latex_python_pairs import (
    AnsatzTerm,
    PauliPolynomial,
    PauliTerm,
)


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial(
            "JW",
            [PauliTerm(len(pauli), ps=pauli, pc=1.0)],
        ),
    )


def test_complete_ansatz_ledger_uses_graph_span_without_transpilation() -> None:
    ledger = evaluate_pauli_graph_span_ansatz(
        (_term("wide", "xeex"), _term("local", "ezee")),
        num_qubits=4,
        embedding=(0, 1, 2, 3),
        coupling_edges=((0, 1), (1, 2), (2, 3)),
        backend_name="FakeMarrakesh",
        graph_snapshot={"coupling_edges": [[0, 1], [1, 2], [2, 3]]},
        independent_parameter_count=2,
    ).to_json_dict()

    assert ledger["profile_id"] == PAPER_I_MARRAKESH_GRAPH_SPAN_PROFILE_V1
    assert ledger["no_transpile"] is True
    assert ledger["routing_seed_used"] is False
    assert ledger["ordered_generator_occurrence_count"] == 2
    assert ledger["nonidentity_pauli_rotation_count"] == 2
    assert ledger["absolute_c_hat_2q"] == 2.0
    assert ledger["absolute_c_hat_d"] == 6.0
    assert ledger["absolute_c_hat_1q"] == 6.0
    assert ledger["absolute_theta_count"] == 2.0
    assert len(ledger["ordered_support_sha256"]) == 64


def test_trajectory_comparison_reports_absolute_and_seed_relative_cost() -> None:
    seed = _term("seed", "xeex")
    inserted = _term("inserted", "ezee")
    comparison = compare_pauli_graph_span_ansatzes(
        initial_terms=(seed,),
        terminal_terms=(seed, inserted),
        num_qubits=4,
        embedding=(0, 1, 2, 3),
        coupling_edges=((0, 1), (1, 2), (2, 3)),
        backend_name="FakeMarrakesh",
        graph_snapshot={"coupling_edges": [[0, 1], [1, 2], [2, 3]]},
        initial_parameter_count=1,
        terminal_parameter_count=2,
    )

    assert comparison["reporting_role"] == (
        "routing_independent_structural_cost_primary_v1"
    )
    assert comparison["initial"]["absolute_c_hat_2q"] == 2.0
    assert comparison["terminal"]["absolute_c_hat_2q"] == 2.0
    assert comparison["terminal_minus_initial"] == {
        "absolute_c_hat_2q": 0.0,
        "absolute_c_hat_d": 0.0,
        "absolute_c_hat_1q": 1.0,
        "absolute_theta_count": 1.0,
    }


def test_repeated_positioned_occurrences_are_counted_repeatedly() -> None:
    occurrence = _term("same-word", "xeex")
    one = evaluate_pauli_graph_span_ansatz(
        (occurrence,),
        num_qubits=4,
        embedding=(0, 1, 2, 3),
        coupling_edges=((0, 1), (1, 2), (2, 3)),
        backend_name="FakeMarrakesh",
        graph_snapshot={},
        independent_parameter_count=1,
    )
    two = evaluate_pauli_graph_span_ansatz(
        (occurrence, occurrence),
        num_qubits=4,
        embedding=(0, 1, 2, 3),
        coupling_edges=((0, 1), (1, 2), (2, 3)),
        backend_name="FakeMarrakesh",
        graph_snapshot={},
        independent_parameter_count=2,
    )

    assert two.absolute_c_hat_2q == 2.0 * one.absolute_c_hat_2q
    assert two.absolute_c_hat_d == 2.0 * one.absolute_c_hat_d
    assert two.absolute_c_hat_1q == 2.0 * one.absolute_c_hat_1q
