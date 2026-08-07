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
    QSEBasisVectorPolicy,
    ResponseChannel,
    ResponseTimeGrid,
    SpectralGrid,
    build_response_functions_payload,
    computational_basis_state,
    compute_qse_spectra,
    pauli_string_basis_element,
    pauli_string_observable,
)
from pipelines.qse_spectra.__main__ import main as qse_main
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm


def _poly(nq: int, terms: list[tuple[str, complex]]) -> PauliPolynomial:
    out = PauliPolynomial("JW")
    for label, coeff in terms:
        out.add_term(PauliTerm(int(nq), ps=str(label), pc=complex(coeff)))
    return out


def _minus_z_response_result():
    hamiltonian = _poly(1, [("z", -1.0)])
    psi = computational_basis_state(1, "0")
    policy = QSEBasisVectorPolicy(reference_projection="q0", basis_vector_normalization="raw_projected")
    result = compute_qse_spectra(
        hamiltonian,
        psi,
        [pauli_string_basis_element("X", nq=1, name="x_excitation")],
        basis_vector_policy=policy,
        transition_observables=[pauli_string_observable("X", nq=1, name="probe", metadata={"source": "unit_test"})],
    )
    return hamiltonian, psi, result


def test_response_payload_computes_frequency_time_moments_and_sum_rule_deficits() -> None:
    hamiltonian, psi, result = _minus_z_response_result()

    payload = build_response_functions_payload(
        result,
        grid=SpectralGrid(omega_min=1.5, omega_max=2.5, num_points=3),
        kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
        time_grid=ResponseTimeGrid(t_min=0.0, t_max=math.pi / 4.0, num_points=2),
        channels=[ResponseChannel("probe", "probe", "XX")],
        max_moment_order=2,
        hamiltonian=hamiltonian,
        prepared_state=psi,
    )

    assert payload["schema_version"] == "qse_response_functions_v1"
    assert payload["complex_scalar_convention"] == "[real, imag]"
    assert payload["frequency_convention"]["reference_energy"] == pytest.approx(-1.0)
    assert payload["frequency_grid"]["values"] == pytest.approx([1.5, 2.0, 2.5])
    assert payload["time_grid"]["units"] == "inverse_hamiltonian_energy"

    channel = payload["channels"][0]
    assert channel["A_label"] == "probe"
    assert channel["B_label"] == "probe"
    assert channel["A_operator_source"] == "unit_test"
    assert channel["B_operator_source"] == "unit_test"
    assert channel["channel_kind"] == "XX"
    assert channel["roots"][0]["omega"] == pytest.approx(2.0)
    assert channel["roots"][0]["residue"] == pytest.approx([1.0, 0.0])

    # Unit-area Lorentzian at zero offset: (eta/pi)/(eta^2) = 1/(pi*eta).
    assert channel["frequency_response"]["quantity"] == "S_AB(omega)"
    assert channel["frequency_response"]["values"][1] == pytest.approx([1.0 / (math.pi * 0.1), 0.0])
    assert channel["time_correlation"]["quantity"] == "C_AB(t)"
    assert channel["time_correlation"]["values"][0] == pytest.approx([1.0, 0.0])
    assert channel["time_correlation"]["values"][1] == pytest.approx([0.0, -1.0], abs=1.0e-12)

    moments = {item["order"]: item["value"] for item in channel["moments"]}
    assert moments[0] == pytest.approx([1.0, 0.0])
    assert moments[1] == pytest.approx([2.0, 0.0])
    assert moments[2] == pytest.approx([4.0, 0.0])

    deficits = channel["sum_rule_deficits"]
    assert deficits["status"] == "evaluated"
    assert deficits["m0"]["target"] == pytest.approx([1.0, 0.0])
    assert deficits["m0"]["deficit_abs"] == pytest.approx(0.0, abs=1.0e-12)
    assert deficits["m1"]["target"] == pytest.approx([2.0, 0.0])
    assert deficits["m1"]["deficit_abs"] == pytest.approx(0.0, abs=1.0e-12)


def test_sum_rule_m1_deficit_uses_actual_m1_when_only_m0_is_serialized() -> None:
    hamiltonian, psi, result = _minus_z_response_result()

    payload = build_response_functions_payload(
        result,
        grid=SpectralGrid(omega_min=1.5, omega_max=2.5, num_points=3),
        kernel_config=BroadeningKernelConfig(kernel="lorentzian", eta=0.1),
        time_grid=ResponseTimeGrid(t_min=0.0, t_max=0.0, num_points=1),
        channels=[ResponseChannel("probe", "probe", "XX")],
        max_moment_order=0,
        hamiltonian=hamiltonian,
        prepared_state=psi,
    )

    channel = payload["channels"][0]
    assert [item["order"] for item in channel["moments"]] == [0]
    assert channel["sum_rule_deficits"]["m1"]["qse"] == pytest.approx([2.0, 0.0])
    assert channel["sum_rule_deficits"]["m1"]["deficit_abs"] == pytest.approx(0.0, abs=1.0e-12)


def test_response_sum_rule_deficits_are_not_evaluated_without_direct_inputs() -> None:
    _, _, result = _minus_z_response_result()

    payload = build_response_functions_payload(
        result,
        grid=SpectralGrid(omega_min=1.5, omega_max=2.5, num_points=3),
        kernel_config=BroadeningKernelConfig(kernel="gaussian", eta=0.2),
        time_grid=ResponseTimeGrid(t_min=0.0, t_max=0.0, num_points=1),
        channels=[ResponseChannel("probe", "probe", "custom")],
        max_moment_order=1,
    )

    deficits = payload["channels"][0]["sum_rule_deficits"]
    assert deficits["status"] == "not_evaluated"
    assert "reason" in deficits


def test_cli_threads_response_payload_additively_without_changing_base_schema(tmp_path: Path) -> None:
    ham_path = tmp_path / "minus_z_ham.json"
    out_path = tmp_path / "qse_response.json"
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
            "probe=X",
            "--spectral-grid-min",
            "0",
            "--spectral-grid-max",
            "4",
            "--spectral-grid-num",
            "81",
            "--spectral-eta",
            "0.05",
            "--spectral-kernel",
            "lorentzian",
            "--response-functions",
            "--response-channel",
            "probe:probe:XX",
            "--response-time-grid-min",
            "0",
            "--response-time-grid-max",
            str(math.pi / 4.0),
            "--response-time-grid-num",
            "2",
            "--response-moment-max-order",
            "1",
            "--output-json",
            str(out_path),
            "--omit-matrices",
        ]
    ) == 0

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "qse_spectra_v1"
    assert data["settings"]["response_functions_enabled"] is True
    assert data["matrices"] == {"included": False}
    assert "spectral_functions" in data  # existing spectral postprocessing remains additive.

    response = data["qse_response_functions_v1"]
    assert response["schema_version"] == "qse_response_functions_v1"
    assert response["controller_boundary"]["feeds_controller_decisions"] is False
    assert response["complex_scalar_encoding"] == "array_real_imag"
    channel = response["channels"][0]
    assert channel["A_label"] == "probe"
    assert channel["B_label"] == "probe"
    assert channel["channel_kind"] == "XX"
    assert channel["roots"][0]["omega"] == pytest.approx(0.0)
    assert channel["roots"][1]["omega"] == pytest.approx(2.0)
    assert channel["moments"][0]["value"] == pytest.approx([1.0, 0.0])
    assert channel["moments"][1]["value"] == pytest.approx([2.0, 0.0])
    assert channel["sum_rule_deficits"]["status"] == "evaluated"
    assert channel["sum_rule_deficits"]["m0"]["deficit_abs"] == pytest.approx(0.0, abs=1.0e-12)
