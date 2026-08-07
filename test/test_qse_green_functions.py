from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra import (
    BroadeningKernelConfig,
    GreenFunctionMode,
    SpectralGrid,
    build_green_function_payload,
    computational_basis_state,
    compute_qse_spectra,
    jw_ladder_source_state,
    pauli_string_basis_element,
)
from pipelines.qse_spectra.__main__ import main as qse_main
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def _number_poly(epsilon: float = 1.0) -> PauliPolynomial:
    # n_0 = (I - Z_0) / 2 in the repo convention.
    return _poly(1, [("e", 0.5 * float(epsilon)), ("z", -0.5 * float(epsilon))])


def _one_mode_result(bitstring: str):
    hamiltonian = _number_poly(1.0)
    psi = computational_basis_state(1, bitstring)
    basis = [pauli_string_basis_element("I", nq=1, name="identity")]
    result = compute_qse_spectra(hamiltonian, psi, basis)
    return hamiltonian, psi, basis, result


def _green_payload(bitstring: str):
    hamiltonian, psi, basis, result = _one_mode_result(bitstring)
    payload = build_green_function_payload(
        result,
        hamiltonian=hamiltonian,
        prepared_state=psi,
        modes=[GreenFunctionMode("a0", 0)],
        grid=SpectralGrid(omega_min=0.0, omega_max=2.0, num_points=3),
        kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
        fermion_mode_count=1,
    )
    return hamiltonian, psi, basis, result, payload


def test_jw_ladder_sources_and_empty_one_mode_green_function() -> None:
    hamiltonian, empty, basis, result, payload = _green_payload("0")
    mode = GreenFunctionMode("a0", 0)

    add_source = jw_ladder_source_state(empty, mode=mode, operation="addition", expected_nq=1)
    rem_source = jw_ladder_source_state(empty, mode=mode, operation="removal", expected_nq=1)
    assert np.allclose(add_source.source_state, computational_basis_state(1, "1"), atol=1.0e-12)
    assert np.allclose(rem_source.source_state, np.zeros(2, dtype=complex), atol=1.0e-12)

    assert payload["schema_version"] == "qse_green_function_v1"
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["sector_policy"]["source_specific_qse_solves"] is True
    assert payload["sector_policy"]["neutral_qse_matrices_reused_for_green_sectors"] is False
    assert payload["sector_policy"]["explicit_particle_number_projection"] is False
    assert payload["sector_policy"]["operator_basis_reused_from_parent_qse"] is True
    assert payload["mode_domain"]["fermion_mode_count"] == 1
    assert payload["mode_domain"]["fermion_mode_count_source"] == "caller_supplied"
    assert payload["frequency_convention"]["reference_energy"] == pytest.approx(0.0)

    mode_payload = payload["modes"][0]
    addition = mode_payload["addition"]
    removal = mode_payload["removal"]
    assert addition["source_norm"] == pytest.approx(1.0)
    assert addition["qse_solve_status"] == "solved_source_specific_qse_sector"
    assert addition["sector_policy"]["sector_projection"] == "identity"
    assert addition["sector_policy"]["reference_projection"] == "none"
    assert addition["roots"][0]["sector_energy"] == pytest.approx(1.0)
    assert addition["roots"][0]["energy_offset_from_reference"] == pytest.approx(1.0)
    assert addition["roots"][0]["retarded_pole_omega"] == pytest.approx(1.0)
    assert addition["roots"][0]["residue"] == pytest.approx([1.0, 0.0])

    assert removal["zero_source_sector"] is True
    assert removal["qse_solve_status"] == "skipped_zero_source"
    assert removal["roots"] == []
    assert all(value == pytest.approx([0.0, 0.0]) for value in removal["retarded_green_function"]["values"])

    # G^R(omega=epsilon) = 1/(i eta) = -i/eta for the empty one-mode addition pole.
    assert mode_payload["retarded_green_function"]["values"][1] == pytest.approx([0.0, -10.0], abs=1.0e-12)
    assert mode_payload["diagonal_spectral_function"]["values"][1] == pytest.approx(10.0 / math.pi)

    sum_rule = mode_payload["diagonal_sum_rule_diagnostics"]
    assert sum_rule["addition_source_norm_squared"] == pytest.approx(1.0)
    assert sum_rule["removal_source_norm_squared"] == pytest.approx(0.0)
    assert sum_rule["total_residue_sum"] == pytest.approx([1.0, 0.0])
    assert sum_rule["source_norm_canonical_deficit_abs"] == pytest.approx(0.0, abs=1.0e-12)
    assert sum_rule["residue_canonical_deficit_abs"] == pytest.approx(0.0, abs=1.0e-12)


def test_jw_ladder_source_uses_repo_two_qubit_ordering_and_lower_mode_parity() -> None:
    mode_1 = GreenFunctionMode("mode1", 1)
    lower_occupied = computational_basis_state(2, "01")
    both_occupied = computational_basis_state(2, "11")

    add_source = jw_ladder_source_state(lower_occupied, mode=mode_1, operation="addition", expected_nq=2)
    rem_source = jw_ladder_source_state(both_occupied, mode=mode_1, operation="removal", expected_nq=2)

    assert np.allclose(add_source.source_state, -both_occupied, atol=1.0e-12)
    assert np.allclose(rem_source.source_state, -lower_occupied, atol=1.0e-12)


def test_jw_ladder_sources_and_occupied_one_mode_green_function() -> None:
    _hamiltonian, occupied, _basis, _result, payload = _green_payload("1")
    mode = GreenFunctionMode("a0", 0)

    add_source = jw_ladder_source_state(occupied, mode=mode, operation="addition", expected_nq=1)
    rem_source = jw_ladder_source_state(occupied, mode=mode, operation="removal", expected_nq=1)
    assert np.allclose(add_source.source_state, np.zeros(2, dtype=complex), atol=1.0e-12)
    assert np.allclose(rem_source.source_state, computational_basis_state(1, "0"), atol=1.0e-12)

    mode_payload = payload["modes"][0]
    addition = mode_payload["addition"]
    removal = mode_payload["removal"]
    assert addition["zero_source_sector"] is True
    assert addition["qse_solve_status"] == "skipped_zero_source"
    assert removal["source_norm"] == pytest.approx(1.0)
    assert removal["qse_solve_status"] == "solved_source_specific_qse_sector"
    assert removal["roots"][0]["sector_energy"] == pytest.approx(0.0)
    assert removal["roots"][0]["energy_offset_from_reference"] == pytest.approx(-1.0)
    assert removal["roots"][0]["retarded_pole_omega"] == pytest.approx(1.0)
    assert removal["roots"][0]["residue"] == pytest.approx([1.0, 0.0])
    assert mode_payload["retarded_green_function"]["values"][1] == pytest.approx([0.0, -10.0], abs=1.0e-12)
    assert mode_payload["diagonal_sum_rule_diagnostics"]["zero_source_sectors"] == {
        "addition": True,
        "removal": False,
    }


def test_green_function_rejects_invalid_mode_and_duplicate_labels() -> None:
    hamiltonian, psi, basis, result = _one_mode_result("0")

    with pytest.raises(ValueError, match="valid range"):
        build_green_function_payload(
            result,
            hamiltonian=hamiltonian,
            prepared_state=psi,
            modes=[GreenFunctionMode("bad", 1)],
            grid=SpectralGrid(omega_min=0.0, omega_max=2.0, num_points=3),
            kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
            basis_elements=basis,
            fermion_mode_count=1,
        )

    with pytest.raises(ValueError, match="unique"):
        build_green_function_payload(
            result,
            hamiltonian=hamiltonian,
            prepared_state=psi,
            modes=[GreenFunctionMode("dup", 0), GreenFunctionMode("dup", 0)],
            grid=SpectralGrid(omega_min=0.0, omega_max=2.0, num_points=3),
            kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
            basis_elements=basis,
            fermion_mode_count=1,
        )


def test_green_function_requires_lorentzian_kernel() -> None:
    hamiltonian, psi, basis, result = _one_mode_result("0")

    with pytest.raises(ValueError, match="lorentzian"):
        build_green_function_payload(
            result,
            hamiltonian=hamiltonian,
            prepared_state=psi,
            modes=[GreenFunctionMode("a0", 0)],
            grid=SpectralGrid(omega_min=0.0, omega_max=2.0, num_points=3),
            kernel_config=BroadeningKernelConfig(kernel="gaussian", eta=0.1),
            basis_elements=basis,
            fermion_mode_count=1,
        )


def test_cli_threads_green_function_payload_additively_without_neutral_response(tmp_path: Path) -> None:
    ham_path = tmp_path / "number_ham.json"
    out_path = tmp_path / "qse_green.json"
    ham_path.write_text(
        json.dumps(
            {
                "terms": [
                    {"pauli_exyz": "e", "coeff_re": 0.5, "coeff_im": 0.0},
                    {"pauli_exyz": "z", "coeff_re": -0.5, "coeff_im": 0.0},
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
            "0",
            "--operator-basis-label",
            "I",
            "--green-functions",
            "--green-function-mode",
            "a0=0",
            "--green-function-fermion-qubits",
            "1",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "2",
            "--spectral-grid-num",
            "3",
            "--spectral-eta",
            "0.1",
            "--spectral-kernel",
            "lorentzian",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["settings"]["green_functions_enabled"] is True
    assert data["settings"]["response_functions_enabled"] is False
    assert data["settings"]["conductivity_response_enabled"] is False
    assert data["settings"]["green_function_modes"] == [{"label": "a0", "mode_index": 0}]
    assert data["settings"]["green_function_fermion_qubits"] == 1
    assert data["matrices"] == {"included": False}
    assert "qse_response_functions_v1" not in data
    assert "qse_conductivity_response_v1" not in data

    green = data["qse_green_function_v1"]
    assert green["schema_version"] == "qse_green_function_v1"
    assert green["controller_boundary"]["post_run_diagnostic_only"] is True
    assert green["frequency_grid"]["values"] == pytest.approx([0.0, 1.0, 2.0])
    assert green["kernel"]["name"] == "lorentzian"
    assert green["mode_domain"]["fermion_mode_count"] == 1
    assert green["sector_policy"]["operator_basis_reused_from_parent_qse"] is True
    mode_payload = green["modes"][0]
    assert mode_payload["label"] == "a0"
    assert mode_payload["addition"]["roots"][0]["retarded_pole_omega"] == pytest.approx(1.0)
    assert mode_payload["removal"]["zero_source_sector"] is True
    assert mode_payload["retarded_green_function"]["values"][1] == pytest.approx([0.0, -10.0], abs=1.0e-12)

