from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra.core import (
    QSEBasisVectorPolicy,
    QSEMatrices,
    QSEResult,
    computational_basis_state,
    compute_qse_spectra,
    pauli_string_basis_element,
    pauli_string_observable,
)
from pipelines.qse_spectra.spectral_functions import (
    BroadeningKernelConfig,
    CutoffBoundaryLayout,
    SpectralGrid,
    SpectralReference,
    SpectralWindow,
    build_cutoff_boundary_diagnostics,
    build_spectral_function_payload,
    build_spectral_window_metrics_payload,
    gaussian_kernel,
    lorentzian_kernel,
    parse_spectral_window_spec,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))


def _one_qubit_gap_result():
    hamiltonian = _poly(1, [("z", -1.0)])
    psi = computational_basis_state(1, "0")
    policy = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")
    return compute_qse_spectra(
        hamiltonian,
        psi,
        [pauli_string_basis_element("X", nq=1, name="x_excitation")],
        basis_vector_policy=policy,
        transition_observables=[pauli_string_observable("X", nq=1, name="dipole")],
    )


def test_lorentzian_and_gaussian_kernels_are_finite_and_unit_area_convention() -> None:
    grid = np.linspace(-10.0, 10.0, 20001)
    lorentzian = lorentzian_kernel(grid, eta=0.2)
    gaussian = gaussian_kernel(grid, eta=0.2)

    assert np.all(np.isfinite(lorentzian))
    assert np.all(np.isfinite(gaussian))
    assert lorentzian_kernel(np.asarray([0.0]), eta=0.5)[0] == pytest.approx(2.0 / np.pi)
    assert gaussian_kernel(np.asarray([0.0]), eta=0.5)[0] == pytest.approx(1.0 / (0.5 * np.sqrt(2.0 * np.pi)))
    assert 0.98 < _trapz(lorentzian, grid) < 1.0
    assert _trapz(gaussian, grid) == pytest.approx(1.0, abs=1.0e-12)


def test_spectral_function_payload_peaks_at_one_qubit_qse_gap() -> None:
    result = _one_qubit_gap_result()
    payload = build_spectral_function_payload(
        result,
        grid=SpectralGrid(0.0, 4.0, 401),
        kernel_config=BroadeningKernelConfig("lorentzian", eta=0.05),
    )

    assert payload["schema_version"] == "qse_spectral_functions_v1"
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["reference_energy"] == pytest.approx(-1.0)
    observable = payload["observables"][0]
    assert observable["name"] == "dipole"
    assert observable["roots"][0]["energy"] == pytest.approx(1.0)
    assert observable["roots"][0]["omega"] == pytest.approx(2.0)
    assert observable["roots"][0]["transition_strength"] == pytest.approx(1.0)
    assert observable["peak_omega"] == pytest.approx(2.0)
    assert observable["peak_value"] > 6.0
    assert len(observable["values"]) == 401


def test_spectral_window_metrics_and_identical_reference_comparison() -> None:
    result = _one_qubit_gap_result()
    spectral = build_spectral_function_payload(
        result,
        grid=SpectralGrid(0.0, 4.0, 401),
        kernel_config=BroadeningKernelConfig("lorentzian", eta=0.05),
    )
    observable = spectral["observables"][0]
    reference = SpectralReference(
        observable_name="dipole",
        grid=spectral["grid"]["values"],
        values=observable["values"],
        label="self",
    )

    metrics = build_spectral_window_metrics_payload(
        spectral,
        windows=[SpectralWindow("gap", 1.5, 2.5)],
        references=[reference],
    )

    assert metrics["schema_version"] == "qse_spectral_window_metrics_v1"
    assert metrics["controller_boundary"]["feeds_controller_decisions"] is False
    record = metrics["observables"][0]["window_metrics"][0]
    assert record["window_name"] == "gap"
    assert record["integrated_weight"] > 0.9
    assert record["centroid"] == pytest.approx(2.0, abs=1.0e-12)
    assert record["peak_omega"] == pytest.approx(2.0)
    comparison = record["reference_comparison"]
    assert comparison["reference_label"] == "self"
    assert comparison["feeds_controller_decisions"] is False
    assert comparison["l1_error"] == pytest.approx(0.0, abs=1.0e-12)
    assert comparison["l2_error"] == pytest.approx(0.0, abs=1.0e-12)
    assert comparison["max_abs_error"] == pytest.approx(0.0, abs=1.0e-12)


def test_spectral_window_parser_accepts_named_and_unnamed_specs() -> None:
    assert parse_spectral_window_spec("1.5:2.5", index=3) == SpectralWindow("window_3", 1.5, 2.5)
    assert parse_spectral_window_spec("gap:1.5:2.5") == SpectralWindow("gap", 1.5, 2.5)


def test_binary_cutoff_boundary_diagnostic_for_one_qubit_root() -> None:
    result = _one_qubit_gap_result()
    payload = build_cutoff_boundary_diagnostics(
        result,
        layout=CutoffBoundaryLayout(num_sites=1, n_ph_max=1, boson_encoding="binary", fermion_qubits=0),
    )

    assert payload["schema_version"] == "qse_cutoff_boundary_diagnostics_v1"
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["layout"]["qubits_per_boson_site"] == 1
    root = payload["roots"][0]
    assert root["omega"] == pytest.approx(2.0)
    assert root["ell_cut"] == pytest.approx(1.0)
    assert root["boundary_probability_by_site"] == pytest.approx([1.0])
    assert root["legal_probability_by_site"] == pytest.approx([1.0])
    assert root["illegal_probability_by_site"] == pytest.approx([0.0])


def _single_vector_qse_result(*, nq: int, occupied_index: int) -> QSEResult:
    basis_vector = np.zeros(1 << int(nq), dtype=complex)
    basis_vector[int(occupied_index)] = 1.0
    matrices = QSEMatrices(
        nq=int(nq),
        hilbert_dim=int(1 << int(nq)),
        basis_elements=(pauli_string_basis_element("e" * int(nq), nq=int(nq), name="explicit_root"),),
        reference_energy=0.0,
        reference_energy_imag_abs=0.0,
        basis_vector_norms=(1.0,),
        overlap=np.asarray([[1.0 + 0.0j]]),
        hamiltonian=np.asarray([[1.0 + 0.0j]]),
        overlap_hermitian_residual_max_abs_raw=0.0,
        hamiltonian_hermitian_residual_max_abs_raw=0.0,
        hamiltonian_coeff_imag_max_abs=0.0,
        basis_matrix_vectors=(basis_vector,),
    )
    return QSEResult(
        matrices=matrices,
        eigenvalues=np.asarray([1.0]),
        eigenvectors_basis=np.asarray([[1.0 + 0.0j]]),
        overlap_eigenvalues_raw=np.asarray([1.0]),
        overlap_eigenvalues_clamped=np.asarray([1.0]),
        retained_overlap_indices=(0,),
        overlap_pruning_threshold=1.0e-12,
        retained_rank=1,
        discarded_rank=0,
        overlap_condition_estimate=1.0,
        overlap_min_eigenvalue_raw=1.0,
        overlap_max_eigenvalue_raw=1.0,
        generalized_residual_norms=(0.0,),
        solver_status="unit_test",
    )


def test_unary_cutoff_boundary_diagnostic_for_explicit_root_at_code_10() -> None:
    result = _single_vector_qse_result(nq=2, occupied_index=2)

    payload = build_cutoff_boundary_diagnostics(
        result,
        layout=CutoffBoundaryLayout(num_sites=1, n_ph_max=1, boson_encoding="unary", fermion_qubits=0),
    )

    root = payload["roots"][0]
    assert payload["layout"]["qubits_per_boson_site"] == 2
    assert root["ell_cut"] == pytest.approx(1.0)
    assert root["boundary_probability_by_site"] == pytest.approx([1.0])
    assert root["legal_probability_by_site"] == pytest.approx([1.0])
    assert root["illegal_probability_by_site"] == pytest.approx([0.0])


def test_binary_cutoff_offsets_respect_fermion_qubits_and_multiple_sites() -> None:
    # q0 is a fermion qubit; q1 is site 0; q2 is site 1.  Index 0b011 has
    # site 0 at the boundary and site 1 below the boundary.
    result = _single_vector_qse_result(nq=3, occupied_index=0b011)

    payload = build_cutoff_boundary_diagnostics(
        result,
        layout=CutoffBoundaryLayout(num_sites=2, n_ph_max=1, boson_encoding="binary", fermion_qubits=1),
    )

    root = payload["roots"][0]
    assert payload["layout"]["qubits_per_boson_site"] == 1
    assert root["ell_cut"] == pytest.approx(1.0)
    assert root["boundary_probability_by_site"] == pytest.approx([1.0, 0.0])
    assert root["legal_probability_by_site"] == pytest.approx([1.0, 1.0])
    assert root["illegal_probability_by_site"] == pytest.approx([0.0, 0.0])


def test_cutoff_layout_mismatch_is_rejected() -> None:
    result = _one_qubit_gap_result()

    with pytest.raises(ValueError, match="expects"):
        build_cutoff_boundary_diagnostics(
            result,
            layout=CutoffBoundaryLayout(num_sites=2, n_ph_max=1, boson_encoding="binary", fermion_qubits=0),
        )
