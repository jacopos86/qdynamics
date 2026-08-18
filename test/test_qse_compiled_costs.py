from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_BACKEND_TRANSPILE,
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    QSE_COMPILED_COSTS_SCHEMA_VERSION,
    annotate_basis_with_compiled_costs,
    compiled_costs_manifest_payload,
    qse_basis_element_to_ansatz_term,
)
from pipelines.qse_spectra.core import pauli_string_basis_element, polynomial_basis_element
from pipelines.qse_spectra.io import load_operator_basis_json
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

GOLDEN_QSE_RESULT = (
    REPO_ROOT
    / "output/diagnostics/paper_iii_hh_advisor_demo_20260802_a005/qse_result.json"
)


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def test_basis_element_to_ansatz_term_conversion() -> None:
    string_term = qse_basis_element_to_ansatz_term(
        pauli_string_basis_element("xy", nq=2, name="string_direction")
    )
    assert string_term.label == "string_direction"
    terms = list(string_term.polynomial.return_polynomial())
    assert len(terms) == 1
    assert str(terms[0].pw2strng()) == "xy"
    assert complex(terms[0].p_coeff) == pytest.approx(1.0 + 0.0j)

    poly = _poly(2, [("yx", 0.5), ("xy", -0.5)])
    poly_term = qse_basis_element_to_ansatz_term(
        polynomial_basis_element(poly, name="poly_direction")
    )
    assert poly_term.label == "poly_direction"
    assert poly_term.polynomial is poly


def test_graph_span_annotation_orders_and_prices_support() -> None:
    # The graph-span embedding supports fixed HH register sizes (4/6/8/...).
    basis = [
        pauli_string_basis_element("eeee", nq=4, name="identity"),
        pauli_string_basis_element("eeex", nq=4, name="one_qubit"),
        pauli_string_basis_element("eexy", nq=4, name="two_qubit"),
    ]

    rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=4,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    )

    assert [row.name for row in rows] == ["identity", "one_qubit", "two_qubit"]
    assert rows[0].estimate.c_hat_2q == pytest.approx(0.0)
    assert rows[2].estimate.c_hat_2q > 0.0
    assert rows[2].scalarized_canonical_cost > rows[1].scalarized_canonical_cost
    cumulative = [row.cumulative_scalarized_cost for row in rows]
    assert cumulative == sorted(cumulative)
    assert len({row.hardware_cost_source for row in rows}) == 1
    assert len({row.source_mode for row in rows}) == 1

    payload = compiled_costs_manifest_payload(
        rows, oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN, num_qubits=4
    )
    assert payload["schema_version"] == QSE_COMPILED_COSTS_SCHEMA_VERSION
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["shot_component_annotated"] is False
    assert payload["summary"]["row_count"] == 3
    assert payload["summary"]["total_scalarized_cost"] == pytest.approx(cumulative[-1])
    assert len(payload["summary"]["hardware_cost_sources"]) == 1


@pytest.mark.skipif(not GOLDEN_QSE_RESULT.is_file(), reason="stored 20260802 advisor-demo basis not present")
def test_golden_advisor_demo_basis_annotates_completely() -> None:
    basis, provenance = load_operator_basis_json(GOLDEN_QSE_RESULT, nq=8)
    assert len(basis) == 158

    rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=8,
        oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    )

    assert len(rows) == len(basis)
    assert all(math.isfinite(row.scalarized_canonical_cost) for row in rows)
    assert all(row.scalarized_canonical_cost >= 0.0 for row in rows)
    cumulative = [row.cumulative_scalarized_cost for row in rows]
    assert cumulative == sorted(cumulative)
    assert len({row.hardware_cost_source for row in rows}) == 1
    assert len({row.source_mode for row in rows}) == 1
    identity_rows = [row for row in rows if row.name == "identity"]
    assert identity_rows and identity_rows[0].scalarized_canonical_cost == pytest.approx(0.0)
    assert any(row.estimate.c_hat_2q > 0.0 for row in rows)

    payload = compiled_costs_manifest_payload(
        rows, oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN, num_qubits=8
    )
    assert payload["summary"]["row_count"] == 158
    assert payload["summary"]["hardware_cost_sources"] == sorted(
        {row.hardware_cost_source for row in rows}
    )


def test_backend_transpile_annotation_small_basis() -> None:
    basis = [
        pauli_string_basis_element("ex", nq=2, name="one_qubit"),
        pauli_string_basis_element("xy", nq=2, name="two_qubit"),
    ]

    rows = annotate_basis_with_compiled_costs(
        basis,
        num_qubits=2,
        oracle_kind=ORACLE_KIND_BACKEND_TRANSPILE,
    )

    assert len(rows) == 2
    assert all(math.isfinite(row.scalarized_canonical_cost) for row in rows)
    assert rows[1].estimate.c_hat_2q > rows[0].estimate.c_hat_2q
    assert len({row.hardware_cost_source for row in rows}) == 1
    assert rows[0].source_mode != "proxy"
