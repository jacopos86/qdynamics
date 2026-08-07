from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.qse_spectra.core import QSEBasisElement
from pipelines.scaffold.qse_compact_root_refit import (
    compose_base_scaffold_and_excitation,
    fit_compact_greedy_pauli_ansatz,
)
from pipelines.scaffold.qse_root_refit import PauliRotationRefitResult
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(
        label=label,
        polynomial=PauliPolynomial("JW", [PauliTerm(len(pauli), ps=pauli, pc=1.0)]),
        execution_mode="termwise_product",
    )


def _fidelity(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=complex).reshape(-1)
    b = np.asarray(right, dtype=complex).reshape(-1)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(abs(np.vdot(a, b)) ** 2)


def test_compact_greedy_refit_selects_one_pauli_for_one_qubit_root() -> None:
    basis = (
        QSEBasisElement(name="identity", kind="pauli_string", pauli_label_exyz="e"),
        QSEBasisElement(name="flip", kind="pauli_string", pauli_label_exyz="x"),
    )
    hamiltonian = PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])

    fit, diagnostics = fit_compact_greedy_pauli_ansatz(
        target_state=np.asarray([0.0, 1.0], dtype=complex),
        prepared_state=np.asarray([1.0, 0.0], dtype=complex),
        basis=basis,
        nq=1,
        hamiltonian=hamiltonian,
        qse_energy=-1.0,
        max_selected_paulis=1,
        target_infidelity=1.0e-12,
        max_energy_error=1.0e-12,
        max_physical_residual=1.0e-12,
        optimizer_maxiter=100,
    )

    assert diagnostics.selected_labels == ("x",)
    assert fit.layout.runtime_parameter_count == 1
    assert fit.fidelity == pytest.approx(1.0, abs=1.0e-12)
    assert diagnostics.physical_residual_norm <= 1.0e-12


def test_base_scaffold_composition_replays_reference_to_suffix_state() -> None:
    reference = np.asarray([1.0, 0.0], dtype=complex)
    base_term = _term("base_x", "x")
    base_layout = build_parameter_layout([base_term])
    base_executor = CompiledAnsatzExecutor(
        [base_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=base_layout,
    )
    base_theta = np.asarray([0.37], dtype=float)
    base_initial = base_executor.prepare_state(base_theta, reference)

    suffix_term = _term("suffix_z", "z")
    suffix_layout = build_parameter_layout([suffix_term])
    suffix_executor = CompiledAnsatzExecutor(
        [suffix_term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=suffix_layout,
    )
    suffix_theta = np.asarray([-0.23], dtype=float)
    suffix_state = suffix_executor.prepare_state(suffix_theta, base_initial)
    suffix_fit = PauliRotationRefitResult(
        terms=(suffix_term,),
        layout=suffix_layout,
        theta_runtime=suffix_theta,
        theta_logical=suffix_theta,
        fitted_state=suffix_state,
        fidelity=1.0,
        infidelity=0.0,
        optimizer_summary={"method": "unit_test"},
    )
    runtime_input = SimpleNamespace(
        selected_terms=(base_term,),
        base_layout=base_layout,
        theta_runtime=base_theta,
        theta_logical=base_theta,
        psi_ref=reference,
        psi_initial=base_initial,
        provenance={"loader_mode": "unit_test"},
    )

    combined, prepared, composition = compose_base_scaffold_and_excitation(
        runtime_input=runtime_input,
        qse_prepared_state=base_initial,
        excitation_fit=suffix_fit,
    )

    assert len(combined.terms) == 2
    assert combined.layout.runtime_parameter_count == 2
    assert prepared.state.tolist() == reference.tolist()
    assert _fidelity(combined.fitted_state, suffix_state) == pytest.approx(1.0, abs=1.0e-12)
    assert composition["prepared_state_injection_used"] is False
    assert composition["base_runtime_parameter_count"] == 1
    assert composition["excitation_runtime_parameter_count"] == 1
