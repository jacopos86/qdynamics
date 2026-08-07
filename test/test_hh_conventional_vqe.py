#!/usr/bin/env python3
"""Focused tests for HH conventional/fixed-operator VQE helpers."""

from __future__ import annotations

import numpy as np
import pytest

from pipelines.exact_bench import hh_conventional_vqe as conventional
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm, VQEResult


def _toy_operator_terms() -> list[AnsatzTerm]:
    return [
        AnsatzTerm(label="toy_x", polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)])),
        AnsatzTerm(label="toy_y", polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="y", pc=1.0)])),
    ]


def _toy_y_operator_terms() -> list[AnsatzTerm]:
    return [_toy_operator_terms()[1]]


def _toy_x_hamiltonian() -> PauliPolynomial:
    return PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)])


def _toy_two_qubit_operator_terms() -> list[AnsatzTerm]:
    return [
        AnsatzTerm(label="toy_x_q0", polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="ex", pc=1.0)])),
        AnsatzTerm(label="toy_x_q1", polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xe", pc=1.0)])),
        AnsatzTerm(label="toy_xx", polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xx", pc=1.0)])),
    ]


class _FakeQiskitHeaAnsatz:
    num_parameters = 3

    def prepare_state(self, theta, psi_ref):  # noqa: ANN001, ANN201 - mirrors ansatz protocol
        theta_arr = np.asarray(theta, dtype=float)
        assert theta_arr.shape == (3,)
        return np.asarray(psi_ref, dtype=complex)


def test_qiskit_hea_support_probe_is_dependency_guarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(conventional, "_import_qiskit_hea_components", lambda: (object, object, object))
    assert conventional.has_qiskit_hea_support() is True

    def _missing_qiskit():
        raise ImportError("Qiskit benchmark-only HEA support unavailable")

    monkeypatch.setattr(conventional, "_import_qiskit_hea_components", _missing_qiskit)
    assert conventional.has_qiskit_hea_support() is False


def test_qiskit_hea_trial_uses_repo_vqe_minimize(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_build_qiskit_hea_ansatz(*, num_qubits: int, reps: int):
        captured["num_qubits"] = num_qubits
        captured["reps"] = reps
        return _FakeQiskitHeaAnsatz()

    def _fake_vqe_minimize(H, ansatz, psi_ref, *, restarts, seed, maxiter, method):  # noqa: ANN001, ANN002, ANN003
        captured["H"] = H
        captured["ansatz"] = ansatz
        captured["psi_ref"] = np.asarray(psi_ref, dtype=complex)
        captured["restarts"] = restarts
        captured["seed"] = seed
        captured["maxiter"] = maxiter
        captured["method"] = method
        assert int(ansatz.num_parameters) == 3
        return VQEResult(
            energy=-0.875,
            theta=np.array([0.1, -0.2, 0.3], dtype=float),
            success=True,
            message="mocked ok",
            nfev=7,
            nit=3,
            best_restart=1,
            restart_summaries=[{"restart": 1}],
        )

    monkeypatch.setattr(conventional, "_build_qiskit_hea_ansatz", _fake_build_qiskit_hea_ansatz)
    monkeypatch.setattr(conventional, "vqe_minimize", _fake_vqe_minimize)

    payload = conventional.run_hh_conventional_vqe_trial(
        ansatz_kind="qiskit_hea",
        h_poly="fake-h-poly",
        exact_gs=-1.0,
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        boundary="open",
        ordering="blocked",
        reps=2,
        optimizer="COBYLA",
        maxiter=11,
        restarts=2,
        seed=42,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )

    assert captured["num_qubits"] == 1
    assert captured["reps"] == 2
    assert captured["H"] == "fake-h-poly"
    assert captured["restarts"] == 2
    assert captured["seed"] == 42
    assert captured["maxiter"] == 11
    assert captured["method"] == "COBYLA"
    assert payload["success"] is True
    assert payload["method_kind"] == "conventional_vqe"
    assert payload["display_name"] == "HH-HEA-Qiskit"
    assert payload["ansatz_kind"] == "qiskit_hea"
    assert payload["ansatz_name"] == "hh_hea_qiskit"
    assert payload["vqe_reps"] == 2
    assert payload["vqe_restarts"] == 2
    assert payload["vqe_maxiter"] == 11
    assert payload["num_parameters"] == 3
    assert payload["nfev"] == 7
    assert payload["nit"] == 3
    assert np.isfinite(payload["abs_delta_e"])
    assert payload["abs_delta_e"] == 0.125
    assert payload["optimizer_success"] is True
    assert payload["optimizer_message"] == "mocked ok"
    assert np.isclose(np.linalg.norm(payload["_psi_vqe"]), 1.0)


def test_hh_conventional_vqe_decision_noise_reports_exact_final_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_build_qiskit_hea_ansatz(*, num_qubits: int, reps: int):  # noqa: ARG001
        return _FakeQiskitHeaAnsatz()

    def _fake_vqe_minimize(H, ansatz, psi_ref, **kwargs):  # noqa: ANN001, ANN003
        transform = kwargs.get("objective_value_transform")
        assert callable(transform)
        decision_energy = float(
            transform(
                {
                    "energy_ideal": -0.875,
                    "restart_index": 1,
                    "nfev_restart": 1,
                    "nfev_total_estimate": 1,
                }
            )
        )
        return VQEResult(
            energy=decision_energy,
            theta=np.array([0.1, -0.2, 0.3], dtype=float),
            success=True,
            message="mocked ok",
            nfev=1,
            nit=1,
            best_restart=0,
        )

    monkeypatch.setattr(conventional, "_build_qiskit_hea_ansatz", _fake_build_qiskit_hea_ansatz)
    monkeypatch.setattr(conventional, "vqe_minimize", _fake_vqe_minimize)

    payload = conventional.run_hh_conventional_vqe_trial(
        ansatz_kind="qiskit_hea",
        h_poly=_toy_x_hamiltonian(),
        exact_gs=-1.0,
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        boundary="open",
        ordering="blocked",
        reps=2,
        optimizer="COBYLA",
        maxiter=11,
        restarts=2,
        seed=42,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        benchmark_decision_noise_config={
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "0.5",
            "benchmark_decision_noise_seed": "20260515",
        },
        benchmark_decision_noise_scope={"case_id": "hh_L2", "algorithm_id": "static_hva_vqe"},
    )

    meta = payload["benchmark_decision_noise"]
    assert payload["energy"] == pytest.approx(0.0)
    assert payload["abs_delta_e"] == pytest.approx(1.0)
    assert payload["optimizer_decision_energy"] == pytest.approx(meta["trace_preview"][0]["value_decision"])
    assert meta["surfaces_affected"] == ["hh_vqe_objective"]
    assert meta["draw_count_total"] == 1
    assert payload["shots_total"] == 1024


def test_qiskit_hea_trial_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing_builder(*, num_qubits: int, reps: int):
        raise ImportError("Qiskit benchmark-only HEA support unavailable")

    monkeypatch.setattr(conventional, "_build_qiskit_hea_ansatz", _missing_builder)

    with pytest.raises(ImportError, match="Qiskit.*benchmark-only HEA support"):
        conventional.run_hh_conventional_vqe_trial(
            ansatz_kind="qiskit_hea",
            h_poly="fake-h-poly",
            exact_gs=-1.0,
            num_sites=2,
            t=1.0,
            u=4.0,
            dv=0.0,
            omega0=1.0,
            g_ep=0.5,
            n_ph_max=1,
            boson_encoding="binary",
            boundary="open",
            ordering="blocked",
            reps=2,
            optimizer="COBYLA",
            maxiter=11,
            restarts=2,
            seed=42,
            psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        )


def test_compiled_operator_vqe_trial_requires_nonempty_operator_list() -> None:
    with pytest.raises(ValueError, match="operator_terms"):
        conventional.run_compiled_operator_vqe_trial(
            operator_terms=[],
            ansatz_name="empty",
            display_name="Empty",
            h_poly="fake-h-poly",
            exact_gs=-1.0,
            psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
            optimizer="COBYLA",
            maxiter=10,
            restarts=1,
            seed=1,
        )


def test_compiled_operator_vqe_trial_uses_logical_shared_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_vqe_minimize(H, ansatz, psi_ref, *, restarts, seed, maxiter, method):  # noqa: ANN001, ANN002, ANN003
        captured["H"] = H
        captured["ansatz"] = ansatz
        captured["psi_ref"] = np.asarray(psi_ref, dtype=complex)
        captured["restarts"] = restarts
        captured["seed"] = seed
        captured["maxiter"] = maxiter
        captured["method"] = method
        assert int(ansatz.num_parameters) == 2
        assert int(ansatz.logical_parameter_count) == 2
        assert int(ansatz.runtime_parameter_count) >= 2
        prepared = np.asarray(ansatz.prepare_state(np.zeros(2, dtype=float), psi_ref), dtype=complex)
        assert prepared.shape == (2,)
        return VQEResult(
            energy=-0.75,
            theta=np.array([0.1, -0.2], dtype=float),
            success=False,
            message="mocked maxiter",
            nfev=5,
            nit=2,
            best_restart=0,
            restart_summaries=[{"restart": 0}],
        )

    monkeypatch.setattr(conventional, "vqe_minimize", _fake_vqe_minimize)

    payload = conventional.run_compiled_operator_vqe_trial(
        operator_terms=_toy_operator_terms(),
        ansatz_name="hh_uccsd_lifted",
        display_name="HH-UCCSD-Lifted",
        h_poly="fake-h-poly",
        exact_gs=-1.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        optimizer="COBYLA",
        maxiter=11,
        restarts=3,
        seed=9,
        parameterization_mode="logical_shared",
    )

    assert captured["H"] == "fake-h-poly"
    assert captured["restarts"] == 3
    assert captured["seed"] == 9
    assert captured["maxiter"] == 11
    assert captured["method"] == "COBYLA"
    assert payload["success"] is True
    assert payload["method_kind"] == "conventional_vqe"
    assert payload["display_name"] == "HH-UCCSD-Lifted"
    assert payload["ansatz_name"] == "hh_uccsd_lifted"
    assert payload["energy"] == -0.75
    assert payload["exact_gs_energy"] == -1.0
    assert payload["abs_delta_e"] == 0.25
    assert payload["delta_E_abs"] == 0.25
    assert payload["nfev"] == 5
    assert payload["nit"] == 2
    assert payload["num_parameters"] == 2
    assert payload["logical_parameter_count"] == 2
    assert int(payload["runtime_parameter_count"]) >= 2
    assert payload["vqe_reps"] is None
    assert payload["vqe_restarts"] == 3
    assert payload["vqe_maxiter"] == 11
    assert payload["optimizer_success"] is False
    assert payload["optimizer_message"] == "mocked maxiter"
    assert payload["converged"] is False
    assert payload["parameterization_mode"] == "logical_shared"
    assert payload["selected_operator_labels"] == ["toy_x", "toy_y"]
    assert payload["selected_operator_count"] == 2
    assert np.isclose(np.linalg.norm(payload["_psi_vqe"]), 1.0)


def test_compiled_operator_vqe_decision_noise_metadata_and_exact_final_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_vqe_minimize(H, ansatz, psi_ref, **kwargs):  # noqa: ANN001, ANN003
        transform = kwargs.get("objective_value_transform")
        assert callable(transform)
        decision_energy = float(
            transform(
                {
                    "energy_ideal": -0.75,
                    "restart_index": 1,
                    "nfev_restart": 1,
                    "nfev_total_estimate": 1,
                }
            )
        )
        return VQEResult(
            energy=decision_energy,
            theta=np.array([0.1, -0.2], dtype=float),
            success=True,
            message="mocked ok",
            nfev=1,
            nit=1,
            best_restart=0,
            restart_summaries=[{"restart": 0}],
        )

    monkeypatch.setattr(conventional, "vqe_minimize", _fake_vqe_minimize)

    payload = conventional.run_compiled_operator_vqe_trial(
        operator_terms=_toy_operator_terms(),
        ansatz_name="hh_uccsd_lifted",
        display_name="HH-UCCSD-Lifted",
        h_poly=_toy_x_hamiltonian(),
        exact_gs=-1.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        optimizer="COBYLA",
        maxiter=11,
        restarts=3,
        seed=9,
        parameterization_mode="logical_shared",
        benchmark_decision_noise_config={
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "0.5",
            "benchmark_decision_noise_seed": "20260515",
        },
        benchmark_decision_noise_scope={"case_id": "hh_L2", "algorithm_id": "static_uccsd_vqe"},
    )

    meta = payload["benchmark_decision_noise"]
    assert np.isfinite(payload["energy"])
    assert payload["delta_E_abs"] == pytest.approx(abs(payload["energy"] - payload["exact_energy"]))
    assert payload["optimizer_decision_energy"] == pytest.approx(meta["trace_preview"][0]["value_decision"])
    assert meta["surfaces_affected"] == ["hh_compiled_vqe_objective"]
    assert meta["draw_count_total"] == 1


def test_compiled_operator_qsci_trial_requires_nonempty_operator_list() -> None:
    with pytest.raises(ValueError, match="operator_terms"):
        conventional.run_compiled_operator_qsci_trial(
            operator_terms=[],
            ansatz_name="empty",
            display_name="Empty",
            sector_hamiltonian=np.eye(4, dtype=complex),
            sector_basis_full_indices=[0, 1, 2, 3],
            exact_gs=-1.0,
            psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        )


def test_compiled_operator_qsci_exact_gs_is_reporting_only() -> None:
    common = {
        "operator_terms": _toy_two_qubit_operator_terms(),
        "ansatz_name": "hh_qsci_sq_lf_std",
        "display_name": "HH-QSCI-SQ-LF-Std",
        "sector_hamiltonian": np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        "sector_basis_full_indices": [0, 1, 2, 3],
        "psi_ref": np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        "basis_probe_angle": np.pi / 2,
        "basis_amp_cutoff": 1e-9,
        "qsci_max_basis_states": 3,
    }

    payload_a = conventional.run_compiled_operator_qsci_trial(exact_gs=-3.0, **common)
    payload_b = conventional.run_compiled_operator_qsci_trial(exact_gs=9.0, **common)

    assert payload_a["energy"] == pytest.approx(payload_b["energy"])
    assert payload_a["selected_basis_full_indices"] == payload_b["selected_basis_full_indices"]
    assert payload_a["subspace_dimension"] == payload_b["subspace_dimension"]
    assert payload_a["qsci_basis_probe_count"] == payload_b["qsci_basis_probe_count"]
    assert payload_a["exact_gs_energy"] == -3.0
    assert payload_b["exact_gs_energy"] == 9.0
    assert payload_a["abs_delta_e"] != payload_b["abs_delta_e"]


def test_compiled_operator_qsci_support_union_ranking_and_cap() -> None:
    payload = conventional.run_compiled_operator_qsci_trial(
        operator_terms=_toy_two_qubit_operator_terms(),
        ansatz_name="hh_qsci_sq_lf_std",
        display_name="HH-QSCI-SQ-LF-Std",
        sector_hamiltonian=np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        sector_basis_full_indices=[0, 1, 2, 3],
        exact_gs=-3.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        basis_probe_angle=np.pi / 2,
        basis_amp_cutoff=1e-9,
        qsci_max_basis_states=3,
    )

    assert payload["full_sector_dimension"] == 4
    assert payload["subspace_dimension"] == 3
    assert payload["subspace_dimension"] < payload["full_sector_dimension"]
    assert payload["selected_basis_full_indices"] == [0, 1, 2]
    assert payload["selected_sector_indices"] == [0, 1, 2]
    assert payload["qsci_candidate_basis_count"] == 4


def test_compiled_operator_qsci_projected_diagonalization_payload_contract() -> None:
    payload = conventional.run_compiled_operator_qsci_trial(
        operator_terms=_toy_two_qubit_operator_terms(),
        operator_labels=["op0", "op1", "op2"],
        ansatz_name="hh_qsci_sq_lf_std",
        display_name="HH-QSCI-SQ-LF-Std",
        sector_hamiltonian=np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        sector_basis_full_indices=[0, 1, 2, 3],
        exact_gs=-3.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        basis_probe_angle=np.pi / 2,
        basis_amp_cutoff=1e-9,
        qsci_max_basis_states=3,
    )

    assert payload["success"] is True
    assert payload["method_kind"] == "qsci"
    assert payload["ansatz_kind"] == "compiled_operator_qsci"
    assert payload["ansatz_name"] == "hh_qsci_sq_lf_std"
    assert payload["energy"] == pytest.approx(-2.0)
    assert payload["exact_gs_energy"] == -3.0
    assert payload["abs_delta_e"] == pytest.approx(1.0)
    assert payload["delta_E_abs"] == pytest.approx(1.0)
    assert payload["nfev"] == 3
    assert payload["nfev_total"] == 3
    assert payload["nit"] == 0
    assert payload["num_parameters"] is None
    assert payload["selected_operator_count"] == 3
    assert payload["selected_operator_labels"] == ["op0", "op1", "op2"]
    assert payload["subspace_dimension"] == 3
    assert payload["full_sector_dimension"] == 4
    assert payload["qsci_basis_probe_count"] == 3
    assert payload["qsci_basis_selection_mode"] == "single_operator_support_union_top_amp"
    assert payload["qsci_stop_reason"] == "projected_diag"
    assert np.isclose(np.linalg.norm(payload["_psi_vqe"]), 1.0)


def test_compiled_operator_sqd_trial_requires_nonempty_operator_list() -> None:
    with pytest.raises(ValueError, match="operator_terms"):
        conventional.run_compiled_operator_sqd_trial(
            operator_terms=[],
            ansatz_name="empty",
            display_name="Empty",
            sector_hamiltonian=np.eye(4, dtype=complex),
            sector_basis_full_indices=[0, 1, 2, 3],
            exact_gs=-1.0,
            psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        )


def test_compiled_operator_sqd_sampling_is_deterministic_with_fixed_seed() -> None:
    common = {
        "operator_terms": _toy_two_qubit_operator_terms(),
        "ansatz_name": "hh_sqd_sq_lf_std",
        "display_name": "HH-SQD-SQ-LF-Std",
        "sector_hamiltonian": np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        "sector_basis_full_indices": [0, 1, 2, 3],
        "exact_gs": -3.0,
        "psi_ref": np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        "basis_probe_angle": np.pi / 2,
        "sqd_shots_per_probe": 19,
        "sqd_max_basis_states": 3,
        "sqd_seed": 11,
    }

    payload_a = conventional.run_compiled_operator_sqd_trial(**common)
    payload_b = conventional.run_compiled_operator_sqd_trial(**common)

    assert payload_a["energy"] == pytest.approx(payload_b["energy"])
    assert payload_a["selected_basis_full_indices"] == payload_b["selected_basis_full_indices"]
    assert payload_a["selected_sector_indices"] == payload_b["selected_sector_indices"]
    assert payload_a["sqd_sample_counts_by_full_index"] == payload_b["sqd_sample_counts_by_full_index"]
    assert payload_a["sqd_max_observed_probability_by_full_index"] == payload_b[
        "sqd_max_observed_probability_by_full_index"
    ]


def test_compiled_operator_sqd_exact_gs_is_reporting_only() -> None:
    common = {
        "operator_terms": _toy_two_qubit_operator_terms(),
        "ansatz_name": "hh_sqd_sq_lf_std",
        "display_name": "HH-SQD-SQ-LF-Std",
        "sector_hamiltonian": np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        "sector_basis_full_indices": [0, 1, 2, 3],
        "psi_ref": np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        "basis_probe_angle": np.pi / 2,
        "sqd_shots_per_probe": 23,
        "sqd_max_basis_states": 3,
        "sqd_seed": 7,
    }

    payload_a = conventional.run_compiled_operator_sqd_trial(exact_gs=-3.0, **common)
    payload_b = conventional.run_compiled_operator_sqd_trial(exact_gs=9.0, **common)

    assert payload_a["energy"] == pytest.approx(payload_b["energy"])
    assert payload_a["selected_basis_full_indices"] == payload_b["selected_basis_full_indices"]
    assert payload_a["subspace_dimension"] == payload_b["subspace_dimension"]
    assert payload_a["sqd_sample_counts_by_full_index"] == payload_b["sqd_sample_counts_by_full_index"]
    assert payload_a["shots_total"] == payload_b["shots_total"]
    assert payload_a["exact_gs_energy"] == -3.0
    assert payload_b["exact_gs_energy"] == 9.0
    assert payload_a["abs_delta_e"] != payload_b["abs_delta_e"]


def test_compiled_operator_sqd_sampling_ranking_and_cap() -> None:
    payload = conventional.run_compiled_operator_sqd_trial(
        operator_terms=_toy_two_qubit_operator_terms(),
        ansatz_name="hh_sqd_sq_lf_std",
        display_name="HH-SQD-SQ-LF-Std",
        sector_hamiltonian=np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        sector_basis_full_indices=[0, 1, 2, 3],
        exact_gs=-3.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        basis_probe_angle=np.pi / 2,
        sqd_shots_per_probe=64,
        sqd_max_basis_states=99,
        sqd_seed=7,
    )

    assert payload["full_sector_dimension"] == 4
    assert payload["subspace_dimension"] == 3
    assert payload["subspace_dimension"] < payload["full_sector_dimension"]
    assert payload["selected_basis_full_indices"][0] == 0
    assert payload["sqd_candidate_basis_count"] <= payload["full_sector_dimension"]
    assert payload["sqd_sampled_sector_shots"] <= payload["shots_total"]


def test_compiled_operator_sqd_projected_diagonalization_payload_contract() -> None:
    payload = conventional.run_compiled_operator_sqd_trial(
        operator_terms=_toy_two_qubit_operator_terms(),
        operator_labels=["op0", "op1", "op2"],
        ansatz_name="hh_sqd_sq_lf_std",
        display_name="HH-SQD-SQ-LF-Std",
        sector_hamiltonian=np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex),
        sector_basis_full_indices=[0, 1, 2, 3],
        exact_gs=-3.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        basis_probe_angle=np.pi / 2,
        sqd_shots_per_probe=17,
        sqd_max_basis_states=3,
        sqd_seed=5,
    )

    assert payload["success"] is True
    assert payload["method_kind"] == "sqd"
    assert payload["ansatz_kind"] == "compiled_operator_sqd"
    assert payload["ansatz_name"] == "hh_sqd_sq_lf_std"
    assert np.isfinite(payload["energy"])
    assert payload["exact_gs_energy"] == -3.0
    assert np.isfinite(payload["abs_delta_e"])
    assert payload["delta_E_abs"] == pytest.approx(payload["abs_delta_e"])
    assert payload["nfev"] == 3
    assert payload["nfev_total"] == 3
    assert payload["shots_total"] == 51
    assert payload["nit"] == 0
    assert payload["num_parameters"] is None
    assert payload["selected_operator_count"] == 3
    assert payload["selected_operator_labels"] == ["op0", "op1", "op2"]
    assert payload["subspace_dimension"] < payload["full_sector_dimension"]
    assert payload["sqd_basis_probe_count"] == 3
    assert payload["sqd_shots_per_probe"] == 17
    assert payload["sqd_seed"] == 5
    assert payload["sqd_stop_reason"] in {"projected_diag", "reference_only"}
    assert payload["sqd_basis_selection_mode"] == "single_operator_probe_shot_counts"
    assert isinstance(payload["sqd_sample_counts_by_full_index"], dict)
    assert np.isclose(np.linalg.norm(payload["_psi_vqe"]), 1.0)



def test_compiled_operator_avqite_trial_requires_nonempty_operator_list() -> None:
    with pytest.raises(ValueError, match="operator_terms"):
        conventional.run_compiled_operator_avqite_trial(
            operator_terms=[],
            ansatz_name="empty",
            display_name="Empty",
            h_poly=_toy_x_hamiltonian(),
            exact_gs=-1.0,
            psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        )


def test_compiled_operator_avqite_exact_gs_is_reporting_only() -> None:
    common = {
        "operator_terms": _toy_y_operator_terms(),
        "ansatz_name": "toy_y",
        "display_name": "Toy-Y-AVQITE",
        "h_poly": _toy_x_hamiltonian(),
        "psi_ref": np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        "avqite_step_size": 0.1,
        "avqite_max_steps": 4,
        "avqite_energy_tol": 0.0,
        "avqite_residual_tol": 0.0,
        "avqite_derivative_eps": 1e-5,
    }

    payload_a = conventional.run_compiled_operator_avqite_trial(exact_gs=-1.0, **common)
    payload_b = conventional.run_compiled_operator_avqite_trial(exact_gs=9.0, **common)

    assert payload_a["energy"] == pytest.approx(payload_b["energy"])
    assert payload_a["nfev_total"] == payload_b["nfev_total"]
    assert payload_a["avqite_steps_completed"] == payload_b["avqite_steps_completed"]
    assert payload_a["avqite_stop_reason"] == payload_b["avqite_stop_reason"]
    assert payload_a["theta"] == pytest.approx(payload_b["theta"])
    assert payload_a["exact_gs_energy"] == -1.0
    assert payload_b["exact_gs_energy"] == 9.0
    assert payload_a["abs_delta_e"] != payload_b["abs_delta_e"]


def test_compiled_operator_avqite_backtracks_to_monotone_step() -> None:
    payload = conventional.run_compiled_operator_avqite_trial(
        operator_terms=_toy_y_operator_terms(),
        ansatz_name="toy_y",
        display_name="Toy-Y-AVQITE",
        h_poly=_toy_x_hamiltonian(),
        exact_gs=-1.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        avqite_step_size=2.0,
        avqite_max_steps=1,
        avqite_energy_tol=0.0,
        avqite_residual_tol=0.0,
        avqite_derivative_eps=1e-5,
        avqite_max_backtracks=3,
    )

    accepted_steps = [entry for entry in payload["history"] if entry.get("event") == "accepted_step"]
    assert len(accepted_steps) == 1
    assert accepted_steps[0]["backtracks"] == 1
    assert accepted_steps[0]["energy"] <= accepted_steps[0]["energy_before"] + 1e-10
    assert accepted_steps[0]["backtrack_trials"][0]["accepted"] is False
    assert accepted_steps[0]["backtrack_trials"][1]["accepted"] is True
    assert payload["avqite_stop_reason"] == "max_steps"


def test_compiled_operator_avqite_payload_contract() -> None:
    payload = conventional.run_compiled_operator_avqite_trial(
        operator_terms=_toy_y_operator_terms(),
        ansatz_name="toy_y",
        display_name="Toy-Y-AVQITE",
        h_poly=_toy_x_hamiltonian(),
        exact_gs=-1.0,
        psi_ref=np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        avqite_step_size=0.1,
        avqite_max_steps=2,
        avqite_energy_tol=0.0,
        avqite_residual_tol=0.0,
    )

    assert payload["success"] is True
    assert payload["method_kind"] == "avqite"
    assert payload["selected_operator_count"] == 1
    assert payload["selected_operator_labels"] == ["toy_y"]
    assert payload["num_parameters"] == 1
    assert payload["parameterization_mode"] == "logical_shared"
    assert payload["avqite_steps_completed"] == 2
    assert payload["avqite_stop_reason"] == "max_steps"
    assert payload["nfev_total"] >= payload["energy_evaluations_total"] >= 1
    assert np.isfinite(payload["abs_delta_e"])
    assert np.isclose(np.linalg.norm(payload["_psi_vqe"]), 1.0)
