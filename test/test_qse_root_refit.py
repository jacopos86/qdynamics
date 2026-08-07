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
from pipelines.scaffold.qse_root_refit import (
    QSERootRefitConfig,
    QSERootRefitError,
    reconstruct_ansatz_state_from_payload,
    run_qse_root_refit,
)


def _state_fidelity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=complex).reshape(-1)
    bb = np.asarray(b, dtype=complex).reshape(-1)
    aa = aa / np.linalg.norm(aa)
    bb = bb / np.linalg.norm(bb)
    return float(abs(np.vdot(aa, bb)) ** 2)


def _write_minus_z_hamiltonian(path: Path) -> Path:
    path.write_text(
        json.dumps({"terms": [{"pauli_exyz": "z", "coeff_re": -1.0, "coeff_im": 0.0}]}),
        encoding="utf-8",
    )
    return path


def _generate_one_qubit_qse(tmp_path: Path) -> tuple[Path, Path]:
    ham_path = _write_minus_z_hamiltonian(tmp_path / "minus_z_ham.json")
    qse_path = tmp_path / "qse_source.json"
    rc = qse_main(
        [
            "--hamiltonian-json",
            str(ham_path),
            "--state-bitstring",
            "0",
            "--operator-basis-label",
            "I",
            "--operator-basis-label",
            "X",
            "--output-json",
            str(qse_path),
            "--omit-matrices",
        ]
    )
    assert rc == 0
    return ham_path, qse_path


def test_one_qubit_qse_root_refit_reaches_near_unity_fidelity(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)
    out_path = tmp_path / "qse_root_refit.json"

    artifact = run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=qse_path,
            state_index=1,
            output_json=out_path,
            hamiltonian_json=ham_path,
            max_infidelity=1.0e-10,
            max_energy_error=1.0e-10,
            maxiter=0,
        )
    )

    assert out_path.exists()
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "qse_root_refit_v1"
    assert artifact["fit_summary"]["fidelity"] == pytest.approx(1.0, abs=1.0e-12)
    assert artifact["fit_summary"]["infidelity"] <= 1.0e-12
    energy = artifact["fit_summary"]["energy_diagnostics"]
    assert energy["available"] is True
    assert energy["fitted_energy"] == pytest.approx(1.0, abs=1.0e-12)
    assert energy["abs_energy_error_vs_qse"] <= 1.0e-12
    assert artifact["fit_summary"]["passes"]["all_thresholds"] is True

    replayed = reconstruct_ansatz_state_from_payload(artifact)
    assert _state_fidelity(replayed, np.asarray([0.0, 1.0], dtype=complex)) == pytest.approx(1.0, abs=1.0e-12)


def test_artifact_visibility_and_ansatz_payload_lengths_are_valid(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)
    artifact = run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=qse_path,
            state_index=1,
            output_json=tmp_path / "refit.json",
            hamiltonian_json=ham_path,
            maxiter=0,
        )
    )

    boundary = artifact["controller_boundary"]
    assert boundary["controller_usable"] is False
    assert boundary["feeds_controller_decisions"] is False
    assert boundary["decision_path_allowed"] is False
    assert boundary["realtime_wiring"] is False
    assert boundary["matches_scaffold_runtime_contract"] is False

    visibility = artifact["visibility"]
    assert visibility["controller_visible_payload_refs"] == []
    assert visibility["potentially_promotable_payload_refs"] == ["ansatz_payload"]
    assert "qse_ritz_diagnostics.basis_coefficients" in visibility["forbidden_to_controller_refs"]
    assert "target_state_diagnostics" in visibility["forbidden_to_controller_refs"]

    ansatz = artifact["ansatz_payload"]
    parameterization = ansatz["parameterization"]
    assert ansatz["ansatz_schema"] == "pauli_rotation_ansatz_v1"
    assert ansatz["qpu_preparable_in_principle"] is True
    assert ansatz["matches_scaffold_runtime_contract"] is False
    assert parameterization["runtime_parameter_count"] == len(ansatz["theta_runtime"])
    assert parameterization["logical_operator_count"] == len(ansatz["theta_logical"])
    assert parameterization["runtime_parameter_count"] >= 1
    assert "x" in ansatz["selected_operator_labels"]
    assert "e" not in ansatz["selected_operator_labels"]


def test_ground_state_rejected_unless_allow_flag(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)

    with pytest.raises(QSERootRefitError, match="state_index=0"):
        run_qse_root_refit(
            QSERootRefitConfig(
                qse_result_json=qse_path,
                state_index=0,
                output_json=tmp_path / "rejected.json",
                hamiltonian_json=ham_path,
                maxiter=0,
            )
        )

    artifact = run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=qse_path,
            state_index=0,
            output_json=tmp_path / "allowed.json",
            hamiltonian_json=ham_path,
            allow_ground_state=True,
            maxiter=0,
        )
    )
    assert artifact["qse_ritz_diagnostics"]["state_index"] == 0
    assert "ground_qse_root_refit_allowed_by_explicit_flag" in artifact["warnings"]


def test_q0_projected_root_zero_refits_without_ground_override(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)
    payload = json.loads(qse_path.read_text(encoding="utf-8"))
    policy = {
        "reference_projection": "q0",
        "basis_vector_normalization": "raw_projected",
        "sector_projection": "identity",
        "sector_label": "unit_test_sector",
    }
    payload["settings"]["basis_vector_policy"] = dict(policy)
    payload["diagnostics"]["basis_vector_policy"] = dict(policy)
    payload["eigenvalues"][0]["energy_relative_to_reference"] = 1.0
    q0_path = tmp_path / "qse_q0.json"
    q0_path.write_text(json.dumps(payload), encoding="utf-8")

    artifact = run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=q0_path,
            state_index=0,
            output_json=tmp_path / "q0_refit.json",
            hamiltonian_json=ham_path,
            max_infidelity=1.0,
            maxiter=0,
        )
    )

    assert artifact["qse_ritz_diagnostics"]["root_role"] == "lowest_orthogonal_ritz_root"
    assert "q0_root_zero_is_lowest_orthogonal_ritz_root_not_ground_state" in artifact["warnings"]


def test_missing_prepared_state_provenance_fails_clearly(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)
    payload = json.loads(qse_path.read_text(encoding="utf-8"))
    payload["input"]["state"] = {"source_schema": "state_json_without_path"}
    bad_qse_path = tmp_path / "qse_missing_state.json"
    bad_qse_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(QSERootRefitError, match="prepared state provenance.*--prepared-state-json"):
        run_qse_root_refit(
            QSERootRefitConfig(
                qse_result_json=bad_qse_path,
                state_index=1,
                output_json=tmp_path / "bad_refit.json",
                hamiltonian_json=ham_path,
                maxiter=0,
            )
        )

    artifact = run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=bad_qse_path,
            state_index=1,
            output_json=tmp_path / "override_refit.json",
            prepared_state_bitstring="0",
            hamiltonian_json=ham_path,
            maxiter=0,
        )
    )
    assert artifact["source"]["prepared_state_override_used"] is True
    assert artifact["fit_summary"]["fidelity"] == pytest.approx(1.0, abs=1.0e-12)


def test_bad_thresholds_and_basis_coefficient_coverage_raise(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)
    with pytest.raises(QSERootRefitError, match="max_infidelity"):
        run_qse_root_refit(
            QSERootRefitConfig(
                qse_result_json=qse_path,
                state_index=1,
                output_json=tmp_path / "bad_threshold.json",
                hamiltonian_json=ham_path,
                max_infidelity=-1.0,
            )
        )

    payload = json.loads(qse_path.read_text(encoding="utf-8"))
    payload["eigenvalues"][1]["basis_coefficients"].pop()
    bad_qse_path = tmp_path / "qse_bad_coeffs.json"
    bad_qse_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(QSERootRefitError, match="basis_coefficients length"):
        run_qse_root_refit(
            QSERootRefitConfig(
                qse_result_json=bad_qse_path,
                state_index=1,
                output_json=tmp_path / "bad_coeffs_refit.json",
                hamiltonian_json=ham_path,
                maxiter=0,
            )
        )


def test_diagnostic_p2_sections_are_ignored_for_fitting(tmp_path: Path) -> None:
    ham_path, qse_path = _generate_one_qubit_qse(tmp_path)
    payload = json.loads(qse_path.read_text(encoding="utf-8"))
    payload["spectral_functions"] = {
        "schema_version": "qse_spectral_functions_v1",
        "controller_boundary": {"feeds_controller_decisions": False},
        "poison_if_used_for_fit": 123456,
    }
    payload["spectral_window_metrics"] = {"poison_if_used_for_fit": 789}
    payload["cutoff_boundary_diagnostics"] = {"poison_if_used_for_fit": 42}
    qse_with_p2 = tmp_path / "qse_with_p2.json"
    qse_with_p2.write_text(json.dumps(payload), encoding="utf-8")

    artifact = run_qse_root_refit(
        QSERootRefitConfig(
            qse_result_json=qse_with_p2,
            state_index=1,
            output_json=tmp_path / "refit_with_p2.json",
            hamiltonian_json=ham_path,
            maxiter=0,
        )
    )
    assert artifact["fit_summary"]["fidelity"] == pytest.approx(1.0, abs=1.0e-12)
    assert "spectral_functions" not in artifact
    assert "spectral_window_metrics" not in artifact
    assert "cutoff_boundary_diagnostics" not in artifact
