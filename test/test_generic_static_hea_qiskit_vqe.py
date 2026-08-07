#!/usr/bin/env python3
"""Tests for the generic exact-bench Qiskit HEA static VQE suite."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_hea_qiskit_vqe as hea
from pipelines.exact_bench.paper_i_main_tables_spsa_profile import PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
from pipelines.exact_bench.table_i_canonical_cases import TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY
from pipelines.exact_bench.qiskit_hea_adapter import build_qiskit_hea_ansatz
from pipelines.static_adapt.builders.problem_registry import (
    FixedCountConstraint,
    SectorSelection,
    TruncationConstraint,
    WeightedChargeConstraint,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import VQEResult


class _BlockLayout:
    total_qubits = 4
    fermion_qubits = 4

    def block(self, name: str):  # noqa: ANN201 - tiny fake layout
        if name == "fermion":
            return SimpleNamespace(start_qubit=0, stop_qubit=4)
        return None


class _SpinlessLayout:
    total_qubits = 2
    fermion_qubits = 2

    def block(self, name: str):  # noqa: ANN201 - tiny fake layout
        if name == "fermion":
            return SimpleNamespace(start_qubit=0, stop_qubit=2)
        return None


class _SpinBosonLayout:
    total_qubits = 4
    fermion_qubits = 2
    boson_qubits = 2
    boson_encoding = "binary"

    def block(self, name: str):  # noqa: ANN201 - tiny fake layout
        if name == "emitter":
            return SimpleNamespace(start_qubit=0, stop_qubit=2)
        if name == "boson":
            return SimpleNamespace(start_qubit=2, stop_qubit=4)
        return None


class _FakeAnsatz:
    ansatz_name = "qiskit_hea_linear_ryrz_cx"
    num_parameters = 1

    def circuit_stats(self):  # noqa: ANN201 - mirrors adapter stats contract
        return SimpleNamespace(depth=3, count_2q=2, op_counts={"cx": 2, "ry": 2})

    def prepare_state(self, theta, psi_ref):  # noqa: ANN001, ANN201 - ansatz protocol
        out = np.zeros_like(np.asarray(psi_ref, dtype=complex))
        out[5] = 1.0  # q0=1, q2=1 => n_up=1, n_dn=1 for L=2 blocked Hubbard.
        return out


def _fake_context() -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(num_sites=2, ordering="blocked"),
        layout=_BlockLayout(),
        hamiltonian="fake-hamiltonian",
        reference_state=SimpleNamespace(build_state=lambda: np.eye(16, dtype=complex)[5]),
        sector=SectorSelection(
            label="half_filled_spin_sector",
            comparison_space_label="full_register",
            constraints=(
                FixedCountConstraint(quantity="n_up", value=1, scope="full_register"),
                FixedCountConstraint(quantity="n_dn", value=1, scope="full_register"),
            ),
            num_particles=(1, 1),
        ),
    )


def _fake_spinless_context() -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(num_sites=2, ordering="blocked"),
        layout=_SpinlessLayout(),
        hamiltonian="fake-spinless-hamiltonian",
        reference_state=SimpleNamespace(build_state=lambda: np.eye(4, dtype=complex)[1]),
        sector=SectorSelection(
            label="fixed_one_spinless_fermion",
            comparison_space_label="full_register",
            constraints=(FixedCountConstraint(quantity="n_f", value=1, scope="fermion_register"),),
            num_particles=(1,),
        ),
    )


def _fake_spin_boson_context() -> SimpleNamespace:
    return SimpleNamespace(
        request=SimpleNamespace(num_sites=1, ordering="blocked", n_ph_max=2, boson_encoding="binary"),
        layout=_SpinBosonLayout(),
        hamiltonian="fake-spin-boson-hamiltonian",
        reference_state=SimpleNamespace(build_state=lambda: np.eye(16, dtype=complex)[1]),
        sector=SectorSelection(
            label="single_emitter_truncated_boson_sector",
            comparison_space_label="one_emitter_truncated_boson_register",
            constraints=(
                WeightedChargeConstraint(
                    quantity="n_emitter",
                    weights=(("g", 1), ("e", 1)),
                    value=1,
                    scope="emitter_register",
                ),
                TruncationConstraint(
                    quantity="boson_occupancy",
                    max_local_occupancy=2,
                    scope="boson_register",
                ),
            ),
            num_particles=None,
        ),
    )


def test_default_static_hea_case_ids_cover_table_i_canonical_suite() -> None:
    for family, case_ids in TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY.items():
        assert hea.default_static_hea_case_ids(family) == tuple(case_ids)

def test_missing_qiskit_writes_controlled_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hea, "has_qiskit_hea_support", lambda: False)

    payload = hea.run_static_hea_qiskit_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_optional_dependency"
    assert payload["qiskit_available"] is False
    row = payload["rows"][0]
    assert row["execution_surface"] == "qiskit_circuit_statevector_ansatz_with_repo_vqe_optimizer"
    assert row["execution_surface_role"] == "primary_execution_surface"
    assert payload["comparator_source"]["external_reference_status"] == "primary_execution_surface"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()


def test_legacy_hea_optimizer_passthrough_is_preserved_without_overlay() -> None:
    settings = hea._normalize_hea_optimizer_settings(optimizer="SLSQP", maxiter=5, seed=7)

    assert settings["optimizer"] == "SLSQP"
    assert settings["optimizer_kind"] == "slsqp"
    assert settings["optimizer_profile"] is None
    assert settings["maxiter"] == 5

    with pytest.raises(ValueError, match="optimizer overlay"):
        hea._normalize_hea_optimizer_settings(
            optimizer="COBYLA",
            maxiter=5,
            seed=7,
            hea_optimizer="SLSQP",
        )


def test_qiskit_algorithms_spsa_missing_or_incompatible_fails_closed(monkeypatch) -> None:
    def _raise_import(name: str):  # noqa: ANN202
        raise ImportError(f"missing {name}")

    monkeypatch.setattr(hea.importlib, "import_module", _raise_import)
    with pytest.raises(RuntimeError, match="qiskit_algorithms.optimizers.SPSA"):
        hea._load_qiskit_algorithms_spsa_class()

    monkeypatch.setattr(hea.importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="SPSA is missing"):
        hea._load_qiskit_algorithms_spsa_class()

    class _BadSPSA:
        def __init__(self, *, maxiter: int) -> None:
            self.maxiter = int(maxiter)

    monkeypatch.setattr(hea.importlib, "import_module", lambda name: SimpleNamespace(SPSA=_BadSPSA))
    with pytest.raises(RuntimeError, match="does not expose minimize"):
        hea._make_qiskit_algorithms_spsa(maxiter=1)

    class _RejectsScheduleSPSA:
        def __init__(self, *, maxiter: int) -> None:
            self.maxiter = int(maxiter)

        def minimize(self, *, fun, x0, bounds=None):  # noqa: ANN001, ANN201
            return SimpleNamespace(x=x0, fun=float(fun(x0)), nfev=1, nit=1, success=True, message="ok")

    monkeypatch.setattr(hea.importlib, "import_module", lambda name: SimpleNamespace(SPSA=_RejectsScheduleSPSA))
    with pytest.raises(RuntimeError, match="learning_rate=0.04"):
        hea._make_qiskit_algorithms_spsa(maxiter=1, learning_rate=0.04, perturbation=0.01)


def test_hea_spsa_schedule_pair_validation() -> None:
    with pytest.raises(ValueError, match="provided together"):
        hea._normalize_hea_optimizer_settings(
            optimizer="COBYLA",
            maxiter=5,
            seed=7,
            hea_optimizer="spsa",
            hea_spsa_learning_rate=0.04,
        )

    with pytest.raises(ValueError, match="optimizer=spsa"):
        hea._normalize_hea_optimizer_settings(
            optimizer="COBYLA",
            maxiter=5,
            seed=7,
            hea_optimizer="cobyla",
            hea_spsa_learning_rate=0.04,
            hea_spsa_perturbation=0.01,
        )


def test_sector_probability_flags_known_leaking_state() -> None:
    context = _fake_context()
    leaking = np.eye(16, dtype=complex)[0]
    good = np.eye(16, dtype=complex)[5]

    leak_diag = hea.sector_probability(context, leaking)
    good_diag = hea.sector_probability(context, good)

    assert leak_diag["sector_leak_flag"] is True
    assert leak_diag["sector_probability"] == 0.0
    assert good_diag["sector_leak_flag"] is False
    assert good_diag["sector_probability"] == 1.0


def test_sector_probability_handles_spinless_fixed_count_register() -> None:
    context = _fake_spinless_context()
    vacuum = np.eye(4, dtype=complex)[0]
    one_particle = np.eye(4, dtype=complex)[1]
    two_particles = np.eye(4, dtype=complex)[3]

    vacuum_diag = hea.sector_probability(context, vacuum)
    one_particle_diag = hea.sector_probability(context, one_particle)
    two_particle_diag = hea.sector_probability(context, two_particles)

    assert vacuum_diag["sector_leak_flag"] is True
    assert vacuum_diag["sector_probability"] == 0.0
    assert one_particle_diag["sector_leak_flag"] is False
    assert one_particle_diag["sector_probability"] == 1.0
    assert one_particle_diag["constraints_evaluated"] == [
        {"quantity": "n_f", "scope": "fermion_register", "value": 1, "qubits": [0, 1]}
    ]
    assert two_particle_diag["sector_leak_flag"] is True
    assert two_particle_diag["sector_probability"] == 0.0


def test_sector_probability_reports_illegal_binary_boson_occupancy() -> None:
    context = _fake_spin_boson_context()
    legal = np.eye(16, dtype=complex)[1]  # one emitter bit set, boson code 0b00.
    illegal_boson = np.eye(16, dtype=complex)[13]  # one emitter bit set, illegal boson code 0b11.

    legal_diag = hea.sector_probability(context, legal)
    illegal_diag = hea.sector_probability(context, illegal_boson)

    assert legal_diag["sector_leak_flag"] is False
    assert legal_diag["sector_probability"] == 1.0
    assert legal_diag["boson_illegal_probability_max"] == 0.0
    assert legal_diag["boson_truncation_leak_flag"] is False
    assert legal_diag["policy"] == "diagnostic_only_fixed_count_and_truncation_probability"

    assert illegal_diag["sector_leak_flag"] is True
    assert illegal_diag["sector_probability"] == 0.0
    assert illegal_diag["boson_legal_probability_min"] == 0.0
    assert illegal_diag["boson_illegal_probability_max"] == 1.0
    assert illegal_diag["boson_truncation_leak_flag"] is True
    truncation = illegal_diag["truncation_constraints_evaluated"]
    assert truncation == [
        {
            "quantity": "boson_occupancy",
            "scope": "boson_register",
            "max_local_occupancy": 2,
            "qubits": [2, 3],
            "start_qubit": 2,
            "stop_qubit": 4,
            "num_sites": 1,
            "boson_encoding": "binary",
            "bits_per_site": 2,
            "legal_basis_count": 3,
            "legal_probability": 0.0,
            "illegal_probability": 1.0,
        }
    ]
    assert illegal_diag["boson_subspace_diagnostics"]["policy"] == "reporting_only_after_optimizer"


def test_qiskit_adapter_preserves_qubit0_lsb_basis_convention() -> None:
    pytest.importorskip("qiskit")
    ansatz = build_qiskit_hea_ansatz(num_qubits=2, reps=1)
    psi_ref = np.zeros(4, dtype=complex)
    psi_ref[1] = 1.0  # |q1 q0> = |0 1>, so q0 is the rightmost/LSB bit.
    evolved = ansatz.prepare_state(np.zeros(ansatz.num_parameters), psi_ref)

    # With all angles zero, the single linear CX layer maps q0=1 control onto
    # q1, so |01> becomes |11> if Qiskit/repo qubit indexing agree.
    assert np.argmax(np.abs(evolved)) == 3


def test_runner_uses_repo_vqe_and_resolves_exact_after_optimizer(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    context = _fake_context()
    fake_spec = SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
    )

    def _fake_vqe_minimize(H, ansatz, psi_ref, **kwargs):  # noqa: ANN001, ANN003
        events.append("optimizer")
        assert H == "fake-hamiltonian"
        assert isinstance(ansatz, _FakeAnsatz)
        assert kwargs["method"] == "COBYLA"
        return VQEResult(
            energy=-1.25,
            theta=np.array([0.1]),
            success=True,
            message="ok",
            nfev=4,
            nit=2,
            best_restart=0,
        )

    def _fake_exact_energy(ctx):  # noqa: ANN001
        assert events == ["optimizer"]
        events.append("exact")
        return -1.30

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        assert events == ["optimizer", "exact"]
        events.append("sector")
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    monkeypatch.setattr(hea, "has_qiskit_hea_support", lambda: True)
    monkeypatch.setattr(hea, "_spec_by_case_id", lambda family, case_id: fake_spec)
    monkeypatch.setattr(hea, "_resolve_context_from_spec", lambda spec: context)
    monkeypatch.setattr(hea, "build_qiskit_hea_ansatz", lambda *, num_qubits, reps: _FakeAnsatz())
    monkeypatch.setattr(hea, "vqe_minimize", _fake_vqe_minimize)
    monkeypatch.setattr(hea, "_safe_exact_energy", _fake_exact_energy)
    monkeypatch.setattr(hea, "sector_probability", _fake_sector_probability)

    payload = hea.run_static_hea_qiskit_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        reps=1,
        restarts=1,
        maxiter=5,
        optimizer="COBYLA",
    )

    assert events == ["optimizer", "exact", "sector"]
    assert payload["status"] == "completed"
    row = payload["rows"][0]
    assert row["delta_E_abs"] == abs(-1.25 - (-1.30))
    assert row["optimizer"] == "COBYLA"
    assert row["optimizer_kind"] == "cobyla"
    assert row["optimizer_profile"] is None
    assert row["phase3_controller_called"] is False
    assert row["uses_exact_for_decision"] is False
    assert row["infidelity_status"] == "not_available_exact_state_not_exposed_by_problem_context"
    assert row["qiskit_boundary"] == "pipelines.exact_bench_only"
    assert row["sector_leak_flag"] is False
    assert (tmp_path / "generic_static_single.json").exists()
    assert (tmp_path / "metrics_proxy_runs.jsonl").exists()


def test_runner_uses_qiskit_algorithms_spsa_and_records_profile(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    context = _fake_context()
    fake_spec = SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
    )

    class _FakeSPSA:
        def minimize(self, *, fun, x0, bounds=None):  # noqa: ANN001, ANN201
            events.append("optimizer")
            assert bounds == [(-np.pi, np.pi)]
            x = np.asarray(x0, dtype=float).reshape(-1) + 0.05
            return SimpleNamespace(x=x, fun=float(fun(x)), nfev=1, nit=1, success=True, message="ok")

    def _fake_exact_energy(ctx):  # noqa: ANN001
        assert events == ["optimizer"]
        events.append("exact")
        return -1.30

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        assert events == ["optimizer", "exact"]
        events.append("sector")
        return {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        }

    monkeypatch.setattr(hea, "has_qiskit_hea_support", lambda: True)
    monkeypatch.setattr(hea, "_spec_by_case_id", lambda family, case_id: fake_spec)
    monkeypatch.setattr(hea, "_resolve_context_from_spec", lambda spec: context)
    monkeypatch.setattr(hea, "build_qiskit_hea_ansatz", lambda *, num_qubits, reps: _FakeAnsatz())
    def _fake_make_spsa(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs == {"maxiter": 3, "learning_rate": 0.04, "perturbation": 0.01}
        return _FakeSPSA()

    monkeypatch.setattr(hea, "_make_qiskit_algorithms_spsa", _fake_make_spsa)
    monkeypatch.setattr(hea, "_set_qiskit_algorithms_seed", lambda seed: True)
    monkeypatch.setattr(hea, "expval_pauli_polynomial", lambda psi, H: -1.25)
    monkeypatch.setattr(hea, "_safe_exact_energy", _fake_exact_energy)
    monkeypatch.setattr(hea, "sector_probability", _fake_sector_probability)

    payload = hea.run_static_hea_qiskit_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        reps=1,
        restarts=1,
        maxiter=800,
        optimizer_profile=PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID,
        optimizer_profile_source="env",
        hea_optimizer="spsa",
        hea_spsa_maxiter=3,
        hea_spsa_seed=99,
        hea_spsa_learning_rate=0.04,
        hea_spsa_perturbation=0.01,
        optimizer_overlay_source="test",
    )

    assert events == ["optimizer", "exact", "sector"]
    assert payload["status"] == "completed"
    row = payload["rows"][0]
    assert row["optimizer"] == "qiskit_algorithms.optimizers.SPSA"
    assert row["optimizer_kind"] == "spsa"
    assert row["optimizer_profile"] == PAPER_I_MAIN_TABLES_SPSA_PROFILE_ID
    assert row["optimizer_profile_source"] == "env"
    assert row["optimizer_overlay_source"] == "test"
    assert row["hea_spsa_maxiter"] == 3
    assert row["hea_spsa_seed"] == 99
    assert row["hea_spsa_learning_rate"] == pytest.approx(0.04)
    assert row["hea_spsa_perturbation"] == pytest.approx(0.01)
    assert row["spsa_learning_rate"] == pytest.approx(0.04)
    assert row["spsa_perturbation"] == pytest.approx(0.01)
    assert row["vqe_maxiter"] == 3
    assert row["nfev"] == 1
    assert row["delta_E_abs"] == abs(-1.25 - (-1.30))


def test_decision_noise_hook_metadata_and_final_exact_energy(monkeypatch, tmp_path: Path) -> None:
    events: list[str] = []
    context = _fake_context()
    context.hamiltonian = PauliPolynomial("JW", [PauliTerm(4, ps="eeee", pc=-1.20)])
    fake_spec = SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
    )

    def _fake_vqe_minimize(H, ansatz, psi_ref, **kwargs):  # noqa: ANN001, ANN003
        events.append("optimizer")
        transform = kwargs.get("objective_value_transform")
        assert callable(transform)
        decision_energy = float(
            transform(
                {
                    "energy_ideal": -2.0,
                    "restart_index": 1,
                    "nfev_restart": 1,
                    "nfev_total_estimate": 1,
                    "progress_label": "test",
                }
            )
        )
        return VQEResult(
            energy=decision_energy,
            theta=np.array([0.1]),
            success=True,
            message="ok",
            nfev=1,
            nit=1,
            best_restart=0,
        )

    monkeypatch.setattr(hea, "has_qiskit_hea_support", lambda: True)
    monkeypatch.setattr(hea, "_spec_by_case_id", lambda family, case_id: fake_spec)
    monkeypatch.setattr(hea, "_resolve_context_from_spec", lambda spec: context)
    monkeypatch.setattr(hea, "build_qiskit_hea_ansatz", lambda *, num_qubits, reps: _FakeAnsatz())
    monkeypatch.setattr(hea, "vqe_minimize", _fake_vqe_minimize)
    monkeypatch.setattr(hea, "_safe_exact_energy", lambda ctx: -1.30)
    monkeypatch.setattr(
        hea,
        "sector_probability",
        lambda ctx, psi: {
            "sector_probability": 1.0,
            "sector_leak_probability": 0.0,
            "sector_leak_flag": False,
            "sector_leak_threshold": 1e-8,
            "boson_legal_probability_min": None,
            "boson_illegal_probability_max": None,
            "boson_truncation_leak_flag": False,
            "boson_subspace_diagnostics": None,
            "truncation_constraints_evaluated": [],
        },
    )

    payload = hea.run_static_hea_qiskit_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        reps=1,
        restarts=1,
        maxiter=5,
        optimizer="COBYLA",
        benchmark_decision_noise_config={
            "benchmark_decision_noise_model": "gaussian_iid_v1",
            "benchmark_decision_noise_std": "0.5",
            "benchmark_decision_noise_seed": "20260515",
        },
    )

    assert events == ["optimizer"]
    row = payload["rows"][0]
    meta = row["benchmark_decision_noise"]
    assert row["energy"] == pytest.approx(-1.20)
    assert row["delta_E_abs"] == pytest.approx(abs(-1.20 - (-1.30)))
    assert row["optimizer_decision_energy"] == pytest.approx(meta["trace_preview"][0]["value_decision"])
    assert meta["draw_count_total"] == 1
    assert meta["surfaces_affected"] == ["vqe_objective"]
    assert meta["physical_shots_unchanged"] is True
    assert meta["algorithmic_measurement_work_unchanged"] is True
    rows_payload = json.loads((tmp_path / "rows.json").read_text(encoding="utf-8"))
    assert rows_payload["benchmark_decision_noise_status"] == "ok"


def test_runner_failure_path_emits_normalized_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(hea, "has_qiskit_hea_support", lambda: True)
    monkeypatch.setattr(hea, "_spec_by_case_id", lambda family, case_id: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = hea.run_static_hea_qiskit_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "failed"
    assert payload["exception_type"] == "RuntimeError"
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()
