from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.hardcoded.hh_backend_compile_oracle import BackendCompileConfig, BackendCompileOracle
from pipelines.hardcoded.hh_continuation_types import CompileCostEstimate
from pipelines.qiskit_backend_tools import ResolvedBackendTarget
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


class _BackendStub:
    def __init__(self, name: str):
        self.name = str(name)
        self.num_qubits = 6


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(label=str(label), polynomial=PauliPolynomial("JW", [PauliTerm(len(pauli), ps=pauli, pc=1.0)]))


def test_backend_compile_oracle_prefers_lower_penalty_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="ibm_boston",
                    resolved_name="FakeNighthawk",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeNighthawk"),
                    target_snapshot={"backend_name": "FakeNighthawk"},
                ),
                ResolvedBackendTarget(
                    requested_name="ibm_miami",
                    resolved_name="FakeFez",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeFez"),
                    target_snapshot={"backend_name": "FakeFez"},
                ),
            ),
            [
                {"requested_name": "ibm_boston", "resolved_name": "FakeNighthawk", "success": True},
                {"requested_name": "ibm_miami", "resolved_name": "FakeFez", "success": True},
            ],
        )

    def _fake_compile(circuit, backend, *, seed_transpiler: int, optimization_level: int = 1):
        compiled = QuantumCircuit(circuit.num_qubits)
        compiled.metadata = {
            "backend_name": str(backend.name),
            "instruction_count": len(circuit.data),
        }
        return {
            "compiled": compiled,
            "logical_to_physical": tuple(range(circuit.num_qubits)),
            "compiled_num_qubits": int(circuit.num_qubits),
        }

    def _fake_depth(compiled: QuantumCircuit) -> int:
        instr = int(compiled.metadata.get("instruction_count", 0))
        if str(compiled.metadata.get("backend_name")) == "FakeNighthawk":
            return 10 + 2 * instr
        return 12 + 3 * instr

    def _fake_stats(compiled: QuantumCircuit) -> dict[str, object]:
        instr = int(compiled.metadata.get("instruction_count", 0))
        if str(compiled.metadata.get("backend_name")) == "FakeNighthawk":
            return {
                "compiled_count_2q": 6 + 2 * instr,
                "compiled_cx_count": 4 + instr,
                "compiled_ecr_count": 0,
                "compiled_op_counts": {"swap": 0, "cx": 4 + instr},
            }
        return {
            "compiled_count_2q": 7 + 3 * instr,
            "compiled_cx_count": 5 + 2 * instr,
            "compiled_ecr_count": 0,
            "compiled_op_counts": {"swap": 1, "cx": 5 + 2 * instr},
        }

    import pipelines.hardcoded.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _fake_compile)
    monkeypatch.setattr(oracle_mod, "safe_circuit_depth", _fake_depth)
    monkeypatch.setattr(oracle_mod, "compiled_gate_stats", _fake_stats)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(
            mode="transpile_shortlist_v1",
            requested_backend_shortlist=("ibm_boston", "ibm_miami"),
        ),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    op_a = _term("a", "xeeeee")
    op_b = _term("b", "zxeeee")
    snapshot = oracle.snapshot_base([op_a])
    proxy = CompileCostEstimate(
        new_pauli_actions=1.0,
        new_rotation_steps=1.0,
        position_shift_span=1.0,
        refit_active_count=1.0,
        proxy_total=9.0,
        cx_proxy_total=3.0,
        sq_proxy_total=6.0,
        gate_proxy_total=9.0,
        max_pauli_weight=2.0,
    )
    estimate = oracle.estimate_insertion(snapshot, candidate_term=op_b, position_id=1, proxy_baseline=proxy)

    assert estimate.compile_gate_open is True
    assert estimate.selected_backend_name == "FakeNighthawk"
    assert estimate.penalty_total is not None and estimate.penalty_total >= 0.0
    assert estimate.proxy_baseline is not None
    assert estimate.proxy_baseline["proxy_total"] == pytest.approx(9.0)
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["transpile_backend"] == "FakeNighthawk"

    summary = oracle.final_scaffold_summary([op_a, op_b])
    assert summary["selected_backend"]["transpile_backend"] == "FakeNighthawk"
    assert summary["selected_backend"]["absolute_burden_score_v1"] >= 0.0


def test_backend_compile_oracle_closes_gate_when_all_targets_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="ibm_boston",
                    resolved_name="FakeNighthawk",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeNighthawk"),
                    target_snapshot={"backend_name": "FakeNighthawk"},
                ),
            ),
            [{"requested_name": "ibm_boston", "resolved_name": "FakeNighthawk", "success": True}],
        )

    def _always_fail(*args, **kwargs):
        raise RuntimeError("transpile failed")

    import pipelines.hardcoded.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _always_fail)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(mode="transpile_single_v1", requested_backend_name="ibm_boston"),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    op_a = _term("a", "xeeeee")
    snapshot = oracle.snapshot_base([op_a])
    estimate = oracle.estimate_insertion(snapshot, candidate_term=op_a, position_id=0, proxy_baseline=None)

    assert estimate.compile_gate_open is False
    assert estimate.failure_reason == "all_targets_failed"
    assert estimate.successful_target_count == 0
    assert estimate.failed_target_count >= 1
