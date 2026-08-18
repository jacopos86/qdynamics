from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.__main__ import main as qse_main
from pipelines.qse_spectra.compiled_costs import (
    ORACLE_KIND_MARRAKESH_GRAPH_SPAN,
    annotate_basis_with_compiled_costs,
)
from pipelines.qse_spectra.core import compute_qse_spectra, computational_basis_state
from pipelines.qse_spectra.io import (
    load_operator_basis_json,
    load_polynomial_json,
    load_state_json,
)
from pipelines.qse_spectra.record_selection import (
    StaticRecordSelectionConfig,
    select_static_qse_records,
    static_record_selection_payload,
)
from pipelines.qse_spectra.core import pauli_string_basis_element
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm

GOLDEN_DIR = REPO_ROOT / "output/diagnostics/paper_iii_hh_advisor_demo_20260802_a005"
GOLDEN_QSE_RESULT = GOLDEN_DIR / "qse_result.json"
GOLDEN_SOURCE_SEED = GOLDEN_DIR / "source_seed.json"

_golden = pytest.mark.skipif(
    not (GOLDEN_QSE_RESULT.is_file() and GOLDEN_SOURCE_SEED.is_file()),
    reason="stored 20260802 advisor-demo inputs not present",
)


def _load_golden():
    hamiltonian, _prov = load_polynomial_json(GOLDEN_SOURCE_SEED)
    state, _state_prov = load_state_json(GOLDEN_SOURCE_SEED, state_key="auto")
    basis, _basis_prov = load_operator_basis_json(GOLDEN_QSE_RESULT, nq=8)
    return hamiltonian, state, basis


@_golden
def test_compiled_cost_mode_unbounded_reproduces_input_order_spectra() -> None:
    hamiltonian, state, basis = _load_golden()
    rows = annotate_basis_with_compiled_costs(
        basis, num_qubits=8, oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN
    )
    costs = tuple(row.scalarized_canonical_cost for row in rows)

    input_order = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="input_order", max_records=len(basis)),
    )
    compiled = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="compiled_cost", max_records=len(basis)),
        compiled_costs=costs,
    )

    assert sorted(compiled.selected_original_indices) == sorted(input_order.selected_original_indices)
    result_input = compute_qse_spectra(hamiltonian, state, input_order.selected_basis_elements)
    result_compiled = compute_qse_spectra(hamiltonian, state, compiled.selected_basis_elements)
    assert np.allclose(
        np.asarray(result_input.eigenvalues, dtype=float),
        np.asarray(result_compiled.eigenvalues, dtype=float),
        atol=1.0e-8,
    )


@_golden
def test_compiled_cost_truncation_is_cheaper_than_input_order() -> None:
    _hamiltonian, _state, basis = _load_golden()
    rows = annotate_basis_with_compiled_costs(
        basis, num_qubits=8, oracle_kind=ORACLE_KIND_MARRAKESH_GRAPH_SPAN
    )
    costs = tuple(row.scalarized_canonical_cost for row in rows)
    max_records = 40

    compiled = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="compiled_cost", max_records=max_records),
        compiled_costs=costs,
    )
    input_order = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="input_order", max_records=max_records),
    )

    compiled_total = sum(costs[index] for index in compiled.selected_original_indices)
    input_total = sum(costs[index] for index in input_order.selected_original_indices)
    assert compiled_total <= input_total
    assert compiled_total < input_total

    payload = static_record_selection_payload(compiled)
    assert payload["compiled_costs_present"] is True
    assert all("compiled_cost" in row["features"] or "compiled_cost" in row for row in payload["candidates"])


def test_geometry_mode_consumes_compiled_costs_and_validates_alpha() -> None:
    hamiltonian = PauliPolynomial("JW")
    hamiltonian.add_term(PauliTerm(1, ps="z", pc=1.0))
    hamiltonian.add_term(PauliTerm(1, ps="x", pc=0.5))
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("I", nq=1, name="identity"),
        pauli_string_basis_element("Z", nq=1, name="parallel_z"),
        pauli_string_basis_element("X", nq=1, name="residual_flip"),
    ]
    from pipelines.qse_spectra.core import QSEBasisVectorPolicy

    result = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(mode="geometry_selected", max_records=1),
        hamiltonian=hamiltonian,
        prepared_state=psi,
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        compiled_costs=(0.1, 0.2, 0.3),
    )
    payload = static_record_selection_payload(result)
    assert result.geometry_cost_source == "compiled"
    assert payload["geometry_cost_source"] == "compiled"
    assert payload["candidates"][2]["geometry"]["cost_source"] == "compiled"
    assert result.selected_original_indices == (2,)

    with pytest.raises(ValueError, match="geometry_cost_discount_alpha"):
        StaticRecordSelectionConfig(
            mode="geometry_selected", max_records=1, geometry_cost_discount_alpha=-1.0
        )
    discounted = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected", max_records=1, geometry_cost_discount_alpha=1.0
        ),
        hamiltonian=hamiltonian,
        prepared_state=psi,
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0",
            basis_vector_normalization="raw_projected",
        ),
        compiled_costs=(0.1, 0.2, 0.3),
    )
    assert discounted.selected_original_indices == (2,)


def test_cli_compiled_cost_mode_emits_costs_and_frontier(tmp_path: Path) -> None:
    ham_path = tmp_path / "four_qubit_ham.json"
    out_path = tmp_path / "qse_compiled_cost.json"
    ham_path.write_text(
        json.dumps(
            {
                "terms": [
                    {"pauli_exyz": "eeez", "coeff_re": -1.0, "coeff_im": 0.0},
                    {"pauli_exyz": "eeze", "coeff_re": -0.5, "coeff_im": 0.0},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert qse_main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0000",
            "--operator-basis-label",
            "IIII",
            "--operator-basis-label",
            "IIIX",
            "--operator-basis-label",
            "IIXY",
            "--static-record-selection-mode",
            "compiled_cost",
            "--static-record-selection-max-records",
            "2",
            "--static-record-selection-cost-frontier",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    costs = data["qse_compiled_costs_v1"]
    assert costs["schema_version"] == "qse_compiled_costs_v1"
    assert costs["oracle_kind"] == "marrakesh_graph_span_v1"
    assert costs["summary"]["row_count"] == 3
    assert costs["controller_boundary"]["feeds_controller_decisions"] is False

    selection = data["static_record_selection"]
    assert selection["compiled_costs_present"] is True
    assert selection["summary"]["selected_basis_size"] == 2
    selected_names = [row["name"] for row in selection["selected_records"]]
    assert "op_2" not in selected_names  # the two-qubit-support label is the most expensive

    frontier = selection["accuracy_cost_frontier"]
    assert frontier["schema_version"] == "qse_accuracy_cost_frontier_v1"
    assert len(frontier["rows"]) == 2
    cumulative = [row["cumulative_scalarized_cost"] for row in frontier["rows"]]
    assert cumulative == sorted(cumulative)
    assert all(row["solve_status"] in {"solved", "failed"} for row in frontier["rows"])


def test_geometry_residual_stop_terminates_before_budget() -> None:
    # H = z + 0.5x on |0>: after accepting the flip direction the R=1 Ritz
    # residual (projected off the reference) vanishes, so the residual-norm
    # stopping rule must end selection with one operator despite budget 3.
    hamiltonian = PauliPolynomial("JW")
    hamiltonian.add_term(PauliTerm(1, ps="z", pc=1.0))
    hamiltonian.add_term(PauliTerm(1, ps="x", pc=0.5))
    psi = computational_basis_state(1, "0")
    basis = [
        pauli_string_basis_element("X", nq=1, name="flip"),
        pauli_string_basis_element("Y", nq=1, name="flip_dup"),
        pauli_string_basis_element("Z", nq=1, name="dead"),
    ]
    from pipelines.qse_spectra.core import QSEBasisVectorPolicy

    result = select_static_qse_records(
        basis,
        config=StaticRecordSelectionConfig(
            mode="geometry_selected",
            max_records=3,
            geometry_target_roots=1,
            geometry_residual_stop=1.0e-8,
        ),
        hamiltonian=hamiltonian,
        prepared_state=psi,
        basis_vector_policy=QSEBasisVectorPolicy(
            reference_projection="q0", basis_vector_normalization="raw_projected"
        ),
        compiled_costs=(1.0, 1.0, 1.0),
    )

    assert len(result.selected_original_indices) == 1
    assert result.geometry_stop is not None
    assert result.geometry_stop["stop_reason"] == "residual_converged"
    assert result.geometry_stop["final_max_target_residual_norm"] < 1.0e-8
    payload = static_record_selection_payload(result)
    assert payload["geometry_stop"]["stop_reason"] == "residual_converged"
    with pytest.raises(ValueError, match="geometry_residual_stop"):
        StaticRecordSelectionConfig(
            mode="geometry_selected", max_records=1, geometry_residual_stop=0.0
        )
