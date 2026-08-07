from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.qse_spectra import (
    BroadeningKernelConfig,
    ConductivityChannel,
    SpectralGrid,
    build_conductivity_response_payload,
    computational_basis_state,
    compute_qse_spectra,
    pauli_string_basis_element,
    pauli_string_observable,
    polynomial_observable,
)
from pipelines.qse_spectra.__main__ import main as qse_main
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def _minus_z_current_result(*, zero_current: bool = False):
    hamiltonian = _poly(1, [("z", -1.0)])
    psi = computational_basis_state(1, "0")
    if zero_current:
        zero = PauliPolynomial("JW", [PauliTerm(1, ps="e", pc=0.0)])
        current = polynomial_observable(zero, name="J_zero", metadata={"source": "unit_test_zero"})
        observables = [current]
    else:
        observables = [
            pauli_string_observable("Y", nq=1, name="J", metadata={"source": "unit_test_current"}),
            pauli_string_observable("Z", nq=1, name="K", metadata={"source": "unit_test_contact"}),
        ]
    result = compute_qse_spectra(
        hamiltonian,
        psi,
        [
            pauli_string_basis_element("I", nq=1, name="identity"),
            pauli_string_basis_element("X", nq=1, name="x_excitation"),
        ],
        transition_observables=observables,
    )
    return hamiltonian, psi, result


def test_conductivity_payload_preserves_complex_current_and_records_contact_policy() -> None:
    _hamiltonian, psi, result = _minus_z_current_result()

    payload = build_conductivity_response_payload(
        result,
        grid=SpectralGrid(omega_min=0.0, omega_max=4.0, num_points=5),
        kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
        channels=[ConductivityChannel(current_label="J", contact_label="K")],
        prepared_state=psi,
    )

    assert payload["schema_version"] == "qse_conductivity_response_v1"
    assert payload["complex_scalar_encoding"] == "array_real_imag"
    assert payload["controller_boundary"]["feeds_controller_decisions"] is False
    assert payload["contact_policy"]["combines_contact_into_drude_delta"] is False
    assert payload["regular_conductivity_policy"]["zero_or_negative_frequency_handling"]

    channel = payload["channels"][0]
    assert channel["current_label"] == "J"
    assert channel["contact_term"]["status"] == "evaluated"
    assert channel["contact_term"]["expectation"] == pytest.approx([1.0, 0.0])
    assert channel["drude_weight"]["status"] == "not_evaluated"
    assert channel["current_source"]["source_norm"] == pytest.approx(1.0)
    assert channel["current_source"]["zero_current_source"] is False

    excited_root = next(root for root in channel["roots"] if root["omega"] == pytest.approx(2.0))
    assert abs(excited_root["current_amplitude"][1]) == pytest.approx(1.0)
    assert excited_root["current_weight"] == pytest.approx([1.0, 0.0])
    assert channel["paramagnetic_current_response"]["values"][2][0] > 0.0
    assert channel["regular_conductivity"]["values"][0] == pytest.approx([0.0, 0.0])
    assert channel["regular_conductivity"]["values"][2][0] > 0.0


def test_zero_current_source_is_explicit_not_omitted() -> None:
    _hamiltonian, psi, result = _minus_z_current_result(zero_current=True)

    payload = build_conductivity_response_payload(
        result,
        grid=SpectralGrid(omega_min=0.0, omega_max=2.0, num_points=3),
        kernel_config=BroadeningKernelConfig(kernel="gaussian", eta=0.2),
        channels=[ConductivityChannel(current_label="J_zero")],
        prepared_state=psi,
    )

    channel = payload["channels"][0]
    assert channel["current_label"] == "J_zero"
    assert channel["current_source"]["status"] == "evaluated"
    assert channel["current_source"]["zero_current_source"] is True
    assert channel["current_source"]["source_norm"] == pytest.approx(0.0)
    assert len(channel["roots"]) == len(result.eigenvalues)
    assert all(root["current_weight"] == pytest.approx([0.0, 0.0]) for root in channel["roots"])
    assert all(value == pytest.approx([0.0, 0.0]) for value in channel["paramagnetic_current_response"]["values"])
    assert channel["contact_term"]["status"] == "not_supplied"


def test_regular_conductivity_excludes_elastic_zero_frequency_weight() -> None:
    hamiltonian = _poly(1, [("z", -1.0)])
    psi = computational_basis_state(1, "0")
    result = compute_qse_spectra(
        hamiltonian,
        psi,
        [
            pauli_string_basis_element("I", nq=1, name="identity"),
            pauli_string_basis_element("X", nq=1, name="x_excitation"),
        ],
        transition_observables=[
            pauli_string_observable("I", nq=1, name="J_elastic", metadata={"source": "unit_test_elastic"})
        ],
    )

    payload = build_conductivity_response_payload(
        result,
        grid=SpectralGrid(omega_min=0.0, omega_max=1.0, num_points=3),
        kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
        channels=[ConductivityChannel(current_label="J_elastic")],
        prepared_state=psi,
    )

    channel = payload["channels"][0]
    elastic_root = next(root for root in channel["roots"] if root["omega"] == pytest.approx(0.0))
    assert elastic_root["current_weight"] == pytest.approx([1.0, 0.0])
    assert elastic_root["included_in_regular_conductivity_sum"] is False
    assert channel["paramagnetic_current_response"]["values"][1][0] > 0.0
    assert all(value == pytest.approx([0.0, 0.0]) for value in channel["regular_conductivity"]["values"])


def test_cli_threads_conductivity_payload_additively_without_neutral_response(tmp_path: Path) -> None:
    ham_path = tmp_path / "minus_z_ham.json"
    out_path = tmp_path / "qse_conductivity.json"
    ham_path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]}),
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
            "--operator-basis-label",
            "X",
            "--transition-observable-label",
            "current=Y",
            "--transition-observable-label",
            "contact=Z",
            "--conductivity-response",
            "--conductivity-current-label",
            "current",
            "--conductivity-contact-label",
            "contact",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "41",
            "--spectral-eta",
            "0.05",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["settings"]["conductivity_response_enabled"] is True
    assert data["settings"]["response_functions_enabled"] is False
    assert data["matrices"] == {"included": False}
    assert "qse_response_functions_v1" not in data
    assert "spectral_functions" in data  # grid postprocessing remains additive under the existing CLI contract.

    conductivity = data["qse_conductivity_response_v1"]
    assert conductivity["schema_version"] == "qse_conductivity_response_v1"
    assert conductivity["controller_boundary"]["post_run_diagnostic_only"] is True
    channel = conductivity["channels"][0]
    assert channel["current_label"] == "current"
    assert channel["contact_label"] == "contact"
    assert channel["contact_term"]["expectation"] == pytest.approx([1.0, 0.0])
    excited_root = next(root for root in channel["roots"] if math.isclose(root["omega"], 2.0))
    assert excited_root["current_strength"] == pytest.approx(1.0)
