#!/usr/bin/env python3
"""Tests for the generic exact-bench Qiskit AdaptVQE static suite."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pipelines.exact_bench import generic_static_qiskit_adapt_vqe as qadapt
from pipelines.exact_bench.table_i_canonical_cases import (
    TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY,
    TABLE_I_STATIC_SUITE_PROFILE_ENV,
    TABLE_I_THREE_MODEL_MAIN_PROFILE,
)
from pipelines.exact_bench import qiskit_adaptvqe_adapter as adapter


class _FakeTerm:
    def __init__(self, label: str, coeff: complex = 1.0) -> None:
        self._label = str(label)
        self.p_coeff = coeff

    def pw2strng(self) -> str:
        return self._label

    def nqubit(self) -> int:
        return len(self._label)


class _FakePoly:
    def __init__(self, labels: tuple[str, ...] = ("ze", "xz", "ee")) -> None:
        self._terms = [_FakeTerm(label) for label in labels]

    def return_polynomial(self):  # noqa: ANN201 - repo PauliPolynomial protocol
        return list(self._terms)


class _FakeSparsePauliOp:
    def __init__(self, terms):  # noqa: ANN001
        self.terms = list(terms)
        self.num_qubits = len(str(self.terms[0][0])) if self.terms else 0

    @classmethod
    def from_list(cls, terms):  # noqa: ANN001, ANN201
        return cls(terms)

    def simplify(self, atol: float = 1e-12):  # noqa: ANN001, ANN201
        return self

    def __repr__(self) -> str:
        return f"_FakeSparsePauliOp({self.terms!r})"


class _FakeQuantumCircuit:
    def __init__(self, num_qubits: int) -> None:
        self.num_qubits = int(num_qubits)
        self.num_parameters = 0
        self.parameters = []
        self.data = []
        self._ops: list[str] = []
        self.layout = None

    def x(self, qubit: int) -> None:
        self._ops.append(f"x{qubit}")

    def initialize(self, state, qubits) -> None:  # noqa: ANN001
        self._ops.append(f"initialize:{len(state)}:{len(qubits)}")

    def depth(self) -> int:
        return len(self._ops)

    def count_ops(self) -> dict[str, int]:
        return {"x": sum(1 for op in self._ops if op.startswith("x"))}

    def assign_parameters(self, values, inplace: bool = False):  # noqa: ANN001, ANN201
        return self

    def copy(self):  # noqa: ANN201
        return self


class _FakeStatevector:
    def __init__(self, data) -> None:  # noqa: ANN001
        self.data = np.asarray(data, dtype=complex).reshape(-1)

    @classmethod
    def from_instruction(cls, circuit):  # noqa: ANN001, ANN201
        dim = 1 << int(circuit.num_qubits)
        data = np.zeros(dim, dtype=complex)
        data[1 if dim > 1 else 0] = 1.0
        return cls(data)

    def expectation_value(self, operator) -> complex:  # noqa: ANN001
        return -0.75 + 0.0j


class _FakeEstimator:
    pass


class _FakeCOBYLA:
    def __init__(self, maxiter: int = 200) -> None:
        self.maxiter = int(maxiter)


class _FakeVQE:
    def __init__(self, estimator, ansatz, optimizer):  # noqa: ANN001
        self.estimator = estimator
        self.ansatz = ansatz
        self.optimizer = optimizer


class _FakeAdaptVQE:
    last_instance = None
    events: list[str] = []

    def __init__(self, *, solver, operators, initial_state, max_iterations, **kwargs):  # noqa: ANN001, ANN003
        self.solver = solver
        self.operators = list(operators)
        self.initial_state = initial_state
        self.max_iterations = int(max_iterations)
        self.kwargs = kwargs
        self._excitation_list = []
        _FakeAdaptVQE.last_instance = self

    def compute_minimum_eigenvalue(self, operator):  # noqa: ANN001, ANN201
        _FakeAdaptVQE.events.append("optimizer")
        assert isinstance(operator, _FakeSparsePauliOp)
        self._excitation_list = self.operators[:1]
        return SimpleNamespace(
            eigenvalue=-1.25,
            optimal_circuit=self.initial_state,
            optimal_point=np.array([0.2]),
            optimal_parameters={},
            cost_function_evals=5,
            optimizer_result=SimpleNamespace(nit=2),
            num_iterations=1,
            final_max_gradient=0.01,
            termination_criterion="MAXIMUM",
            eigenvalue_history=[-1.25],
        )


_FAKE_COMPONENTS = adapter.QiskitAdaptVQEComponents(
    QuantumCircuit=_FakeQuantumCircuit,
    SparsePauliOp=_FakeSparsePauliOp,
    Statevector=_FakeStatevector,
    StatevectorEstimator=_FakeEstimator,
    AdaptVQE=_FakeAdaptVQE,
    VQE=_FakeVQE,
    COBYLA=_FakeCOBYLA,
)


def _fake_context(*, num_qubits: int = 2) -> SimpleNamespace:
    events = _FakeAdaptVQE.events

    def _resolve_energy(ai_log=None):  # noqa: ANN001
        assert events == ["optimizer"]
        events.append("exact")
        return -1.30

    return SimpleNamespace(
        request=SimpleNamespace(num_sites=2, ordering="blocked"),
        layout=SimpleNamespace(total_qubits=int(num_qubits)),
        hamiltonian=_FakePoly(),
        reference_state=SimpleNamespace(build_state=lambda: np.eye(1 << int(num_qubits), dtype=complex)[1]),
        exact_target=SimpleNamespace(resolve_energy=_resolve_energy),
        sector=SimpleNamespace(constraints=()),
    )


def _fake_full_meta_pool(context, *, max_terms=qadapt._POOL_TERM_CAP):  # noqa: ANN001
    return (
        SimpleNamespace(
            label="full_meta::fake_a",
            polynomial=_FakePoly(labels=("ze",)),
            pauli_labels_exyz=("ze",),
        ),
        SimpleNamespace(
            label="full_meta::fake_b",
            polynomial=_FakePoly(labels=("xz",)),
            pauli_labels_exyz=("xz",),
        ),
    )


def test_default_static_qiskit_adapt_vqe_case_ids_cover_table_i_canonical_suite() -> None:
    for family, case_ids in TABLE_I_CANONICAL_CASE_IDS_BY_FAMILY.items():
        assert qadapt.default_static_qiskit_adapt_vqe_case_ids(family) == tuple(case_ids)


def test_default_static_qiskit_adapt_vqe_case_ids_honor_suite_profile(monkeypatch) -> None:
    monkeypatch.setenv(TABLE_I_STATIC_SUITE_PROFILE_ENV, TABLE_I_THREE_MODEL_MAIN_PROFILE)

    assert qadapt.default_static_qiskit_adapt_vqe_case_ids("hubbard") == (
        "hubbard_L2_three_model_weak",
        "hubbard_L2_three_model_strong",
    )
    assert qadapt.default_static_qiskit_adapt_vqe_case_ids("spin_boson") == (
        "spin_boson_L2_nph1_three_model_weak",
        "spin_boson_L2_nph2_three_model_strong",
    )



def test_adapter_builds_unit_coefficient_nonidentity_pool_and_preserves_labels() -> None:
    poly = _FakePoly(labels=("ee", "ze", "ze", "xy"))

    labels = adapter.hamiltonian_term_pool_labels(poly)
    ops, op_labels = adapter.hamiltonian_term_pool_to_sparse_pauli_ops(
        poly,
        sparse_pauli_op_cls=_FakeSparsePauliOp,
    )

    assert labels == ("ze", "xy")
    assert op_labels == labels
    assert [op.terms for op in ops] == [[("ZI", 1.0 + 0.0j)], [("XY", 1.0 + 0.0j)]]


def test_full_meta_candidate_conversion_accepts_multi_term_sparse_pauli_ops() -> None:
    op = qadapt.pauli_poly_to_sparse_pauli_op(
        _FakePoly(labels=("ze", "xz")),
        sparse_pauli_op_cls=_FakeSparsePauliOp,
    )

    assert op.terms == [("ZI", 1.0 + 0.0j), ("XZ", 1.0 + 0.0j)]


def test_missing_qiskit_writes_controlled_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qadapt, "has_qiskit_adaptvqe_support", lambda: False)

    payload = qadapt.run_static_qiskit_adapt_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_optional_dependency"
    assert payload["qiskit_available"] is False
    row = payload["rows"][0]
    assert row["phase3_controller_called"] is False
    assert row["qiskit_algorithms_boundary"] == "pipelines.exact_bench_only"
    assert row["execution_surface_role"] == "primary_execution_surface"
    assert row["external_reference_status"] == "primary_execution_surface"
    assert payload["comparator_source"]["parity_reference_algorithm_id"] == "static_full_meta_append_adapt_vqe"
    assert (tmp_path / "result.json").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "generic_static_single.json").exists()
    assert (tmp_path / "metrics_proxy_summary.json").exists()


def test_resource_guard_writes_normalized_artifacts(monkeypatch, tmp_path: Path) -> None:
    fake_spec = SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
        features=None,
    )

    monkeypatch.setattr(qadapt, "has_qiskit_adaptvqe_support", lambda: True)
    monkeypatch.setattr(qadapt, "import_qiskit_adaptvqe_components", lambda: _FAKE_COMPONENTS)
    monkeypatch.setattr(qadapt, "_spec_by_case_id", lambda family, case_id: fake_spec)
    monkeypatch.setattr(qadapt, "_resolve_context_from_spec", lambda spec: _fake_context(num_qubits=qadapt._QUBIT_CAP + 1))
    monkeypatch.setattr(qadapt, "build_full_meta_candidate_pool", _fake_full_meta_pool)

    payload = qadapt.run_static_qiskit_adapt_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_resource_guard"
    assert payload["resource_guard"]["resource_guard_kind"] == "qiskit_adaptvqe_qubit_cap"
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert (tmp_path / "rows.json").exists()
    assert (tmp_path / "metrics_proxy_runs.jsonl").exists()


def test_runner_uses_qiskit_adaptvqe_and_resolves_exact_after_optimizer(monkeypatch, tmp_path: Path) -> None:
    _FakeAdaptVQE.events = []
    _FakeAdaptVQE.last_instance = None
    fake_spec = SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
        features=None,
    )

    def _fake_sector_probability(ctx, psi):  # noqa: ANN001
        assert _FakeAdaptVQE.events == ["optimizer", "exact"]
        _FakeAdaptVQE.events.append("sector")
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

    monkeypatch.setattr(qadapt, "has_qiskit_adaptvqe_support", lambda: True)
    monkeypatch.setattr(qadapt, "import_qiskit_adaptvqe_components", lambda: _FAKE_COMPONENTS)
    monkeypatch.setattr(qadapt, "_spec_by_case_id", lambda family, case_id: fake_spec)
    monkeypatch.setattr(qadapt, "_resolve_context_from_spec", lambda spec: _fake_context(num_qubits=2))
    monkeypatch.setattr(qadapt, "build_full_meta_candidate_pool", _fake_full_meta_pool)
    monkeypatch.setattr(qadapt, "sector_probability", _fake_sector_probability)
    monkeypatch.setattr(
        qadapt,
        "_compiled_circuit_stats",
        lambda circuit: {
            "compiled_depth_total": 11,
            "compiled_count_2q_total": 4,
            "compiled_op_counts": {"cx": 4, "rz": 6},
            "compiled_circuit_stats_status": "ok",
            "compiled_circuit_stats_error": None,
            "compiled_basis_gates": ["rz", "cx"],
        },
    )

    payload = qadapt.run_static_qiskit_adapt_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
        max_adapt_iterations=1,
        optimizer_maxiter=7,
    )

    assert _FakeAdaptVQE.events == ["optimizer", "exact", "sector"]
    assert payload["status"] == "completed"
    assert _FakeAdaptVQE.last_instance is not None
    assert _FakeAdaptVQE.last_instance.max_iterations == 1
    assert _FakeAdaptVQE.last_instance.solver.optimizer.maxiter == 7
    row = payload["rows"][0]
    assert row["method_id"] == "static_qiskit_adapt_vqe"
    assert row["method_kind"] == "adapt_reference"
    assert row["algorithm_origin"] == "qiskit_algorithms_adaptvqe_full_meta_exact_bench"
    assert row["phase3_controller_called"] is False
    assert row["uses_exact_for_decision"] is False
    assert row["exact_reference_usage"] == "reporting_only_after_optimization"
    assert row["qiskit_boundary"] == "pipelines.exact_bench_only"
    assert row["qiskit_algorithms_boundary"] == "pipelines.exact_bench_only"
    assert row["adapt_append_only"] is True
    assert row["phase3_emulation"] is False
    assert row["pool_source"] == "problem_local_full_meta_pool"
    assert row["pool_name"] == "full_meta"
    assert row["taxonomy_role"] == "same_pool_controller_comparator"
    assert row["delta_E_abs"] == abs(-1.25 - (-1.30))
    assert row["selected_operator_count"] == 1
    assert row["pool_labels"] == ["full_meta::fake_a", "full_meta::fake_b"]
    assert row["pool_operator_pauli_labels_exyz"] == {
        "full_meta::fake_a": ["ze"],
        "full_meta::fake_b": ["xz"],
    }
    assert row["shots_total"] == 1024 * 2 * (5 + 2)
    assert row["static_shot_estimate_status"] == "deterministic_proxy_not_physical_shots"
    assert row["shots_per_pauli_term_proxy"] == 1024
    assert row["hamiltonian_pauli_term_count"] == 2
    assert row["pool_term_count"] == 2
    assert row["energy_eval_count_proxy"] == 5
    assert row["gradient_scan_count_proxy"] == 1
    assert row["gradient_operator_probe_count_proxy"] == 2
    assert "shots_total = shots_per_pauli_term_proxy" in row["shot_proxy_formula"]
    assert row["compiled_depth_total"] == 11
    assert row["compiled_count_2q_total"] == 4
    assert row["compiled_op_counts"] == {"cx": 4, "rz": 6}
    assert row["compiled_circuit_stats_status"] == "ok"
    assert row["compiled_circuit_stats_error"] is None
    assert (tmp_path / "generic_static_single.json").exists()
    assert (tmp_path / "metrics_proxy_runs.csv").exists()


def test_initial_gradient_convergence_is_normalized_as_completed(monkeypatch, tmp_path: Path) -> None:
    class _ZeroGradientAdaptVQE(_FakeAdaptVQE):
        events: list[str] = []

        def compute_minimum_eigenvalue(self, operator):  # noqa: ANN001, ANN201
            _ZeroGradientAdaptVQE.events.append("optimizer")
            raise RuntimeError(
                "All gradients have been evaluated to lie below the convergence threshold "
                "during the first iteration of the algorithm."
            )

    components = adapter.QiskitAdaptVQEComponents(
        QuantumCircuit=_FakeQuantumCircuit,
        SparsePauliOp=_FakeSparsePauliOp,
        Statevector=_FakeStatevector,
        StatevectorEstimator=_FakeEstimator,
        AdaptVQE=_ZeroGradientAdaptVQE,
        VQE=_FakeVQE,
        COBYLA=_FakeCOBYLA,
    )
    fake_spec = SimpleNamespace(
        benchmark_id="hubbard_L2",
        family="hubbard",
        base_pipeline_args=("--problem", "hubbard", "--L", "2"),
        split="train",
        tags=(),
        features=None,
    )
    context = _fake_context(num_qubits=2)

    def _fake_exact_energy(ctx):  # noqa: ANN001
        assert _ZeroGradientAdaptVQE.events == ["optimizer"]
        _ZeroGradientAdaptVQE.events.append("exact")
        return -1.0

    monkeypatch.setattr(qadapt, "has_qiskit_adaptvqe_support", lambda: True)
    monkeypatch.setattr(qadapt, "import_qiskit_adaptvqe_components", lambda: components)
    monkeypatch.setattr(qadapt, "_spec_by_case_id", lambda family, case_id: fake_spec)
    monkeypatch.setattr(qadapt, "_resolve_context_from_spec", lambda spec: context)
    monkeypatch.setattr(qadapt, "build_full_meta_candidate_pool", _fake_full_meta_pool)
    monkeypatch.setattr(qadapt, "_safe_exact_energy", _fake_exact_energy)
    monkeypatch.setattr(qadapt, "sector_probability", lambda ctx, psi: {"sector_probability": 1.0})
    monkeypatch.setattr(
        qadapt,
        "_compiled_circuit_stats",
        lambda circuit: {
            "compiled_depth_total": 1,
            "compiled_count_2q_total": 0,
            "compiled_op_counts": {"x": 1},
            "compiled_circuit_stats_status": "ok",
            "compiled_circuit_stats_error": None,
            "compiled_basis_gates": ["x"],
        },
    )

    payload = qadapt.run_static_qiskit_adapt_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path,
    )

    assert _ZeroGradientAdaptVQE.events == ["optimizer", "exact"]
    assert payload["status"] == "completed"
    row = payload["rows"][0]
    assert row["energy"] == -0.75
    assert row["exact_energy"] == -1.0
    assert row["delta_E_abs"] == 0.25
    assert row["initial_gradient_converged"] is True
    assert row["adapt_depth_reached"] == 0
    assert row["selected_operator_count"] == 0
    assert row["adapt_stop_reason"] == "initial_gradients_below_threshold"
    assert row["shots_total"] == 1024 * 2 * (1 + 2)
    assert row["energy_eval_count_proxy"] == 1
    assert row["gradient_scan_count_proxy"] == 1
    assert row["gradient_operator_probe_count_proxy"] == 2
    assert row["compiled_depth_total"] == 1
    assert row["compiled_count_2q_total"] == 0


def test_local_full_meta_append_matches_qiskit_adaptvqe_one_iteration_noiseless_when_available(tmp_path: Path) -> None:
    pytest.importorskip("qiskit")
    pytest.importorskip("qiskit_algorithms")

    from pipelines.exact_bench.generic_static_adapt_variants import run_generic_static_adapt_variant_single

    qiskit_payload = qadapt.run_static_qiskit_adapt_vqe_single(
        family="hubbard",
        case_id="hubbard_L2",
        output_dir=tmp_path / "qiskit",
        max_adapt_iterations=1,
        optimizer_maxiter=200,
        seed=123,
    )
    local_payload = run_generic_static_adapt_variant_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_full_meta_append_adapt_vqe",
        output_dir=tmp_path / "local",
        max_adapt_iterations=1,
        optimizer_maxiter=200,
    )

    assert qiskit_payload["status"] == "completed"
    assert local_payload["status"] == "completed"
    qiskit_row = qiskit_payload["rows"][0]
    local_row = local_payload["rows"][0]
    assert qiskit_row["selected_operator_count"] == local_row["selected_operator_count"] == 1
    assert abs(float(qiskit_row["energy"]) - float(local_row["energy"])) < 1e-6
    assert qiskit_row["pool_name"] == local_row["pool_name"] == "full_meta"
    assert qiskit_row["pool_source"] == local_row["pool_source"] == "problem_local_full_meta_pool"



def test_runner_failure_path_emits_normalized_artifacts(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(qadapt, "has_qiskit_adaptvqe_support", lambda: True)
    monkeypatch.setattr(qadapt, "_spec_by_case_id", lambda family, case_id: (_ for _ in ()).throw(RuntimeError("boom")))

    payload = qadapt.run_static_qiskit_adapt_vqe_single(
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
