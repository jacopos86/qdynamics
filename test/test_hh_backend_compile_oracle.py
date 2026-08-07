from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines.static_adapt.hh_backend_compile_oracle import (
    BackendCompileConfig,
    BackendCompileOracle,
    MARRAKESH_GRAPH_SPAN_MODE,
    MarrakeshGraphSpanCostOracle,
    ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1,
    marrakesh_graph_span_edges_for_support,
    marrakesh_logical_embedding_v1,
    marrakesh_pauli_support_from_label,
)
from pipelines.scaffold.hh_continuation_types import CompileCostEstimate
from pipelines.qiskit_backend_tools import (
    ResolvedBackendTarget,
    backend_coupling_graph_snapshot,
    undirected_coupling_edges,
)
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, PauliPolynomial, PauliTerm


class _BackendStub:
    def __init__(self, name: str):
        self.name = str(name)
        self.num_qubits = 6


class _CouplingMapStub:
    def __init__(self, edges):
        self._edges = tuple(edges)

    def get_edges(self):
        return list(self._edges)


class _GraphBackendStub:
    def __init__(self, name: str, edges, *, num_qubits: int = 6):
        self.name = str(name)
        self.num_qubits = int(num_qubits)
        self.coupling_map = _CouplingMapStub(edges)


def _term(label: str, pauli: str) -> AnsatzTerm:
    return AnsatzTerm(label=str(label), polynomial=PauliPolynomial("JW", [PauliTerm(len(pauli), ps=pauli, pc=1.0)]))


def test_backend_compile_config_rejects_nonfinite_weights() -> None:
    with pytest.raises(ValueError, match="backend compile cost weights must be finite and nonnegative"):
        BackendCompileConfig(weight_depth=float("nan"))


def test_backend_compile_default_preserves_proxy_one_qubit_coordinate() -> None:
    oracle = object.__new__(BackendCompileOracle)
    oracle.config = BackendCompileConfig(
        mode="transpile_single_v1",
        requested_backend_name="FakeMarrakesh",
    )
    oracle.targets = (SimpleNamespace(resolved_name="FakeMarrakesh"),)
    proxy = CompileCostEstimate(
        new_pauli_actions=1.0,
        new_rotation_steps=1.0,
        position_shift_span=0.0,
        refit_active_count=1.0,
        proxy_total=1.0,
        c_hat_1q=19.0,
    )

    estimate = oracle._estimate_from_rows(
        base_rows=[
            {
                "transpile_backend": "FakeMarrakesh",
                "transpile_status": "ok",
                "compiled_count_1q": 3,
                "compiled_count_2q": 2,
                "compiled_depth": 4,
                "compiled_depth_2q": 2,
                "compiled_size": 7,
            }
        ],
        trial_rows=[
            {
                "transpile_backend": "FakeMarrakesh",
                "transpile_status": "ok",
                "resolution_kind": "fake_exact",
                "compiled_count_1q": 8,
                "compiled_count_2q": 3,
                "compiled_depth": 6,
                "compiled_depth_2q": 3,
                "compiled_size": 11,
            }
        ],
        proxy_baseline=proxy,
    )

    assert oracle.config.one_qubit_coordinate_policy == "proxy_baseline_v1"
    assert estimate.c_hat_1q == pytest.approx(19.0)
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["raw_delta_compiled_count_1q"] == 5
    assert estimate.selected_backend_row["delta_compiled_count_1q"] == 5


def test_coupling_graph_helpers_symmetrize_and_snapshot_edges() -> None:
    backend = _GraphBackendStub("FakeMarrakesh", [(2, 1), (1, 2), (2, 2), (3, 2)], num_qubits=4)

    assert undirected_coupling_edges(backend) == ((1, 2), (2, 3))
    snapshot = backend_coupling_graph_snapshot(backend)
    assert snapshot["backend_name"] == "FakeMarrakesh"
    assert snapshot["num_qubits"] == 4
    assert snapshot["coupling_edge_count"] == 2
    assert snapshot["coupling_edges"] == [[1, 2], [2, 3]]
    assert snapshot["graph_symmetrized"] is True


def test_marrakesh_pauli_support_convention_and_embedding_sizes() -> None:
    assert marrakesh_pauli_support_from_label("xeee", num_qubits=4) == (3,)
    assert marrakesh_pauli_support_from_label("eeex", num_qubits=4) == (0,)
    assert marrakesh_logical_embedding_v1(4) == (0, 1, 2, 3)
    assert len(marrakesh_logical_embedding_v1(6)) == 6
    assert len(marrakesh_logical_embedding_v1(8)) == 8
    assert len(marrakesh_logical_embedding_v1(9)) == 9
    assert len(marrakesh_logical_embedding_v1(10)) == 10
    assert len(marrakesh_logical_embedding_v1(12)) == 12
    with pytest.raises(ValueError, match="unsupported_marrakesh_graph_span_embedding_v1_size:7"):
        marrakesh_logical_embedding_v1(7)
    with pytest.raises(ValueError, match="pauli_label_length_mismatch"):
        marrakesh_pauli_support_from_label("xe", num_qubits=4)
    with pytest.raises(ValueError, match="invalid_pauli_label"):
        marrakesh_pauli_support_from_label("aeex", num_qubits=4)


def test_marrakesh_graph_span_exact_simple_graph() -> None:
    edges = [(0, 1), (1, 2), (1, 3)]

    assert marrakesh_graph_span_edges_for_support([], coupling_edges=edges) == 0
    assert marrakesh_graph_span_edges_for_support([0], coupling_edges=edges) == 0
    assert marrakesh_graph_span_edges_for_support([0, 2], coupling_edges=edges) == 2
    assert marrakesh_graph_span_edges_for_support([0, 2, 3], coupling_edges=edges) == 3
    with pytest.raises(ValueError, match="marrakesh_graph_span_terminal_not_in_graph:99"):
        marrakesh_graph_span_edges_for_support([99], coupling_edges=edges)


def test_marrakesh_graph_span_oracle_estimate_uses_support_span_and_proxy_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_resolve(**kwargs):
        assert kwargs["requested_names"] == ("FakeMarrakesh",)
        assert kwargs["preferred_fake_backends"] == ("FakeMarrakesh",)
        assert kwargs["allow_preferred_fallback"] is False
        assert kwargs["fallback_mode"] == "single"
        return (
            (
                ResolvedBackendTarget(
                    requested_name="FakeMarrakesh",
                    resolved_name="FakeMarrakesh",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_GraphBackendStub("FakeMarrakesh", [(0, 1), (1, 2), (2, 3)], num_qubits=4),
                    target_snapshot={"backend_name": "FakeMarrakesh"},
                ),
            ),
            [{"requested_name": "FakeMarrakesh", "resolved_name": "FakeMarrakesh", "success": True}],
        )

    def _forbidden_compile(*args, **kwargs):
        raise AssertionError("marrakesh_graph_span_v1 must not transpile")

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _forbidden_compile)

    oracle = MarrakeshGraphSpanCostOracle(
        config=BackendCompileConfig(
            mode=MARRAKESH_GRAPH_SPAN_MODE,
            requested_backend_name="FakeMarrakesh",
            weight_2q=1.0,
            weight_depth=0.1,
        ),
        num_qubits=4,
        ref_state=np.array([1.0] + [0.0] * 15, dtype=complex),
    )
    candidate = AnsatzTerm(
        label="multi",
        polynomial=PauliPolynomial(
            "JW",
            [
                PauliTerm(4, ps="xeex", pc=1.0),
                PauliTerm(4, ps="eeee", pc=1.0),
                PauliTerm(4, ps="zzzz", pc=1.0e-15),
            ],
        ),
    )
    proxy = CompileCostEstimate(
        new_pauli_actions=2.0,
        new_rotation_steps=3.0,
        position_shift_span=4.0,
        refit_active_count=5.0,
        proxy_total=6.0,
        cx_proxy_total=7.0,
        sq_proxy_total=8.0,
        gate_proxy_total=9.0,
        max_pauli_weight=2.0,
        c_hat_2q=55.0,
        c_hat_d=66.0,
        c_hat_1q=12.0,
        c_hat_theta=7.0,
    )

    snapshot = oracle.snapshot_base([])
    estimate = oracle.estimate_insertion(snapshot, candidate_term=candidate, position_id=3, proxy_baseline=proxy)

    assert estimate.hardware_cost_source == MARRAKESH_GRAPH_SPAN_MODE
    assert estimate.source_mode == MARRAKESH_GRAPH_SPAN_MODE
    assert estimate.selected_backend_name == "FakeMarrakesh"
    assert estimate.target_backend_names == ["FakeMarrakesh"]
    assert estimate.c_hat_2q == pytest.approx(2.0)
    assert estimate.c_hat_d == pytest.approx(6.0)
    assert estimate.c_hat_1q == pytest.approx(12.0)
    assert estimate.c_hat_theta == pytest.approx(7.0)
    assert estimate.penalty_total == pytest.approx(2.6)
    assert estimate.raw_delta_compiled_count_2q is None
    assert estimate.raw_delta_compiled_depth is None
    assert estimate.raw_delta_compiled_depth_2q is None
    assert estimate.raw_delta_compiled_size is None
    assert estimate.selected_backend_row is not None
    row = estimate.selected_backend_row
    assert row["no_transpile"] is True
    assert row["transpile_status"] == "not_run"
    assert row["source_mode"] == MARRAKESH_GRAPH_SPAN_MODE
    assert len(row["pauli_terms"]) == 2
    nonidentity = next(term for term in row["pauli_terms"] if int(term["support_size"]) > 0)
    identity = next(term for term in row["pauli_terms"] if int(term["support_size"]) == 0)
    assert nonidentity["logical_support"] == [3, 0]
    assert nonidentity["physical_support"] == [3, 0]
    assert nonidentity["span_edges"] == 3
    assert identity["support_size"] == 0
    assert estimate.proxy_baseline is not None
    assert estimate.proxy_baseline["c_hat_1q"] == pytest.approx(12.0)
    summary = oracle.final_scaffold_summary([candidate])
    assert summary["schema"] == "marrakesh_graph_span_final_summary_v1"
    assert summary["no_transpile"] is True
    assert summary["selected_backend"] == "FakeMarrakesh"
    assert summary["absolute_c_hat_2q"] == pytest.approx(2.0)
    assert summary["absolute_c_hat_d"] == pytest.approx(6.0)
    assert oracle.cache_summary()["mode"] == MARRAKESH_GRAPH_SPAN_MODE


def test_marrakesh_graph_span_oracle_strict_backend_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="requires phase3_backend_name='FakeMarrakesh'"):
        MarrakeshGraphSpanCostOracle(
            config=BackendCompileConfig(mode=MARRAKESH_GRAPH_SPAN_MODE, requested_backend_name="FakeFez"),
            num_qubits=4,
            ref_state=None,
        )
    with pytest.raises(ValueError, match="does not accept --phase3-backend-shortlist"):
        MarrakeshGraphSpanCostOracle(
            config=BackendCompileConfig(
                mode=MARRAKESH_GRAPH_SPAN_MODE,
                requested_backend_name="FakeMarrakesh",
                requested_backend_shortlist=("FakeMarrakesh",),
            ),
            num_qubits=4,
            ref_state=None,
        )

    def _wrong_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="FakeMarrakesh",
                    resolved_name="FakeFez",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_GraphBackendStub("FakeFez", [(0, 1), (1, 2), (2, 3)], num_qubits=4),
                    target_snapshot={"backend_name": "FakeFez"},
                ),
            ),
            [{"requested_name": "FakeMarrakesh", "resolved_name": "FakeFez", "success": True}],
        )

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _wrong_resolve)
    with pytest.raises(RuntimeError, match="resolved unexpected backend"):
        MarrakeshGraphSpanCostOracle(
            config=BackendCompileConfig(mode=MARRAKESH_GRAPH_SPAN_MODE, requested_backend_name="FakeMarrakesh"),
            num_qubits=4,
            ref_state=None,
        )


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

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

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
    assert estimate.hardware_cost_source == "backend_transpile_v1"
    assert estimate.c_hat_2q >= 0.0
    assert estimate.c_hat_d >= 0.0
    assert estimate.raw_delta_compiled_depth_2q is not None
    assert estimate.delta_compiled_depth_2q is not None
    assert estimate.selected_backend_row["delta_compiled_depth_2q"] == pytest.approx(estimate.delta_compiled_depth_2q)

    summary = oracle.final_scaffold_summary([op_a, op_b])
    assert summary["selected_backend"]["transpile_backend"] == "FakeNighthawk"
    assert summary["selected_backend"]["absolute_burden_score_v1"] >= 0.0


def test_backend_compile_oracle_uses_signed_penalty_only_after_clipped_burden_tie(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="fake_a",
                    resolved_name="FakeA",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeA"),
                    target_snapshot={"backend_name": "FakeA"},
                ),
                ResolvedBackendTarget(
                    requested_name="fake_z",
                    resolved_name="FakeZ",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeZ"),
                    target_snapshot={"backend_name": "FakeZ"},
                ),
            ),
            [
                {"requested_name": "fake_a", "resolved_name": "FakeA", "success": True},
                {"requested_name": "fake_z", "resolved_name": "FakeZ", "success": True},
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
        if str(compiled.metadata.get("backend_name")) == "FakeZ":
            return 100 - instr
        return 10

    def _fake_stats(compiled: QuantumCircuit) -> dict[str, object]:
        instr = int(compiled.metadata.get("instruction_count", 0))
        if str(compiled.metadata.get("backend_name")) == "FakeZ":
            return {
                "compiled_count_2q": 100 - instr,
                "compiled_cx_count": 100 - instr,
                "compiled_ecr_count": 0,
                "compiled_op_counts": {"cx": 100 - instr},
            }
        return {
            "compiled_count_2q": 10,
            "compiled_cx_count": 10,
            "compiled_ecr_count": 0,
            "compiled_op_counts": {"cx": 10},
        }

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _fake_compile)
    monkeypatch.setattr(oracle_mod, "safe_circuit_depth", _fake_depth)
    monkeypatch.setattr(oracle_mod, "compiled_gate_stats", _fake_stats)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(
            mode="transpile_shortlist_v1",
            requested_backend_shortlist=("fake_a", "fake_z"),
            reward_negative_deltas=True,
        ),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    op_a = _term("a", "xeeeee")
    op_b = _term("b", "zxeeee")
    snapshot = oracle.snapshot_base([op_a])
    estimate = oracle.estimate_insertion(snapshot, candidate_term=op_b, position_id=1, proxy_baseline=None)

    assert estimate.compile_gate_open is True
    assert estimate.selected_backend_name == "FakeZ"
    assert estimate.penalty_total is not None and estimate.penalty_total < 0.0
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["clipped_penalty_total"] == pytest.approx(0.0)
    assert estimate.selected_backend_row["signed_penalty_total"] < 0.0
    assert estimate.delta_compiled_count_2q == pytest.approx(0.0)
    assert estimate.delta_compiled_depth_2q == pytest.approx(0.0)
    assert estimate.delta_compiled_size == pytest.approx(0.0)
    assert estimate.hardware_cost_source == "backend_transpile_v1"
    assert estimate.c_hat_2q == pytest.approx(0.0)
    assert estimate.c_hat_d == pytest.approx(0.0)
    assert estimate.selected_backend_row["c_hat_2q"] == pytest.approx(0.0)
    assert estimate.selected_backend_row["c_hat_d"] == pytest.approx(0.0)


def test_backend_compile_oracle_ranks_nonnegative_burden_before_signed_reward() -> None:
    oracle = object.__new__(BackendCompileOracle)
    oracle.config = BackendCompileConfig(reward_negative_deltas=True)
    oracle.targets = [
        SimpleNamespace(resolved_name="worse_signed"),
        SimpleNamespace(resolved_name="better_clipped"),
    ]
    estimate = oracle._estimate_from_rows(
        base_rows=[
            {
                "transpile_backend": "worse_signed",
                "transpile_status": "ok",
                "compiled_count_2q": 10,
                "compiled_depth": 100,
                "compiled_depth_2q": 100,
                "compiled_size": 200,
                "compiled_cx_count": 10,
                "compiled_ecr_count": 0,
            },
            {
                "transpile_backend": "better_clipped",
                "transpile_status": "ok",
                "compiled_count_2q": 10,
                "compiled_depth": 10,
                "compiled_depth_2q": 10,
                "compiled_size": 20,
                "compiled_cx_count": 10,
                "compiled_ecr_count": 0,
            },
        ],
        trial_rows=[
            {
                "transpile_backend": "worse_signed",
                "transpile_status": "ok",
                "compiled_count_2q": 11,
                "compiled_depth": 0,
                "compiled_depth_2q": 0,
                "compiled_size": 0,
                "compiled_cx_count": 11,
                "compiled_ecr_count": 0,
            },
            {
                "transpile_backend": "better_clipped",
                "transpile_status": "ok",
                "compiled_count_2q": 10,
                "compiled_depth": 10,
                "compiled_depth_2q": 10,
                "compiled_size": 20,
                "compiled_cx_count": 10,
                "compiled_ecr_count": 0,
            },
        ],
        proxy_baseline=None,
    )

    assert estimate.selected_backend_name == "better_clipped"
    assert estimate.delta_compiled_count_2q == pytest.approx(0.0)
    assert estimate.delta_compiled_depth_2q == pytest.approx(0.0)
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["signed_penalty_total"] == pytest.approx(0.0)


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

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

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


def test_backend_compile_oracle_can_reward_negative_marginal_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="FakeMarrakesh",
                    resolved_name="FakeMarrakesh",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeMarrakesh"),
                    target_snapshot={"backend_name": "FakeMarrakesh"},
                ),
            ),
            [{"requested_name": "FakeMarrakesh", "resolved_name": "FakeMarrakesh", "success": True}],
        )

    def _fake_compile(circuit, backend, *, seed_transpiler: int, optimization_level: int = 1):
        compiled = QuantumCircuit(circuit.num_qubits)
        compiled.metadata = {"instruction_count": len(circuit.data)}
        return {"compiled": compiled, "logical_to_physical": tuple(range(circuit.num_qubits)), "compiled_num_qubits": int(circuit.num_qubits)}

    def _fake_depth(compiled: QuantumCircuit) -> int:
        instr = int(compiled.metadata.get("instruction_count", 0))
        return 100 - instr

    def _fake_stats(compiled: QuantumCircuit) -> dict[str, object]:
        instr = int(compiled.metadata.get("instruction_count", 0))
        return {
            "compiled_count_2q": 100 - instr,
            "compiled_cx_count": 100 - instr,
            "compiled_ecr_count": 0,
            "compiled_op_counts": {"cx": 100 - instr},
        }

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _fake_compile)
    monkeypatch.setattr(oracle_mod, "safe_circuit_depth", _fake_depth)
    monkeypatch.setattr(oracle_mod, "compiled_gate_stats", _fake_stats)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(mode="transpile_single_v1", requested_backend_name="FakeMarrakesh"),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    op_a = _term("a", "xeeeee")
    op_b = _term("b", "zxeeee")
    snapshot = oracle.snapshot_base([op_a])
    estimate = oracle.estimate_insertion(snapshot, candidate_term=op_b, position_id=1, proxy_baseline=None)

    assert estimate.raw_delta_compiled_count_2q is not None
    assert estimate.raw_delta_compiled_count_2q < 0.0
    assert estimate.delta_compiled_count_2q == 0.0
    assert estimate.penalty_total is not None and estimate.penalty_total < 0.0
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["negative_delta_reward_enabled"] is True
    assert estimate.selected_backend_row["signed_penalty_total"] < 0.0


def test_backend_compile_oracle_uses_transpiled_one_qubit_delta_instead_of_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="FakeMarrakesh",
                    resolved_name="FakeMarrakesh",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeMarrakesh"),
                    target_snapshot={"backend_name": "FakeMarrakesh"},
                ),
            ),
            [
                {
                    "requested_name": "FakeMarrakesh",
                    "resolved_name": "FakeMarrakesh",
                    "success": True,
                }
            ],
        )

    calls = {"n": 0}

    def _fake_compile(
        circuit,
        backend,
        *,
        seed_transpiler: int,
        optimization_level: int = 1,
    ):
        calls["n"] += 1
        compiled = QuantumCircuit(circuit.num_qubits)
        compiled.metadata = {"call": calls["n"]}
        return {
            "compiled": compiled,
            "logical_to_physical": tuple(range(circuit.num_qubits)),
            "compiled_num_qubits": int(circuit.num_qubits),
        }

    def _fake_stats(compiled: QuantumCircuit) -> dict[str, object]:
        call = int(compiled.metadata["call"])
        one_qubit_count = {1: 3, 2: 7}[call]
        return {
            "compiled_count_1q": one_qubit_count,
            "compiled_count_1q_semantics": (
                "post_transpile_one_qubit_quantum_ops_"
                "excluding_barrier_delay_id_measure_reset"
            ),
            "compiled_count_2q": 0,
            "compiled_depth_2q": 0,
            "compiled_cx_count": 0,
            "compiled_ecr_count": 0,
            "compiled_op_counts": {"sx": one_qubit_count},
        }

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _fake_compile)
    monkeypatch.setattr(oracle_mod, "compiled_gate_stats", _fake_stats)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(
            mode="transpile_single_v1",
            requested_backend_name="FakeMarrakesh",
            one_qubit_coordinate_policy=(
                ONE_QUBIT_COORDINATE_COMPILED_POSITIVE_DELTA_V1
            ),
        ),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    proxy = CompileCostEstimate(
        new_pauli_actions=1.0,
        new_rotation_steps=1.0,
        position_shift_span=0.0,
        refit_active_count=1.0,
        proxy_total=99.0,
        c_hat_1q=99.0,
    )
    snapshot = oracle.snapshot_base([_term("a", "xeeeee")])
    estimate = oracle.estimate_insertion(
        snapshot,
        candidate_term=_term("b", "zxeeee"),
        position_id=1,
        proxy_baseline=proxy,
    )

    assert estimate.c_hat_1q == pytest.approx(4.0)
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["base_compiled_count_1q"] == 3
    assert estimate.selected_backend_row["compiled_count_1q"] == 7
    assert estimate.selected_backend_row["raw_delta_compiled_count_1q"] == 4
    assert estimate.selected_backend_row["delta_compiled_count_1q"] == 4
    assert estimate.selected_backend_row["compiled_count_1q_semantics"] == (
        "post_transpile_one_qubit_quantum_ops_"
        "excluding_barrier_delay_id_measure_reset"
    )


def test_backend_compile_oracle_uses_configured_depth_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="FakeMarrakesh",
                    resolved_name="FakeMarrakesh",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeMarrakesh"),
                    target_snapshot={"backend_name": "FakeMarrakesh"},
                ),
            ),
            [{"requested_name": "FakeMarrakesh", "resolved_name": "FakeMarrakesh", "success": True}],
        )

    calls = {"n": 0}

    def _fake_compile(circuit, backend, *, seed_transpiler: int, optimization_level: int = 1):
        calls["n"] += 1
        compiled = QuantumCircuit(circuit.num_qubits)
        compiled.metadata = {"call": calls["n"]}
        return {"compiled": compiled, "logical_to_physical": tuple(range(circuit.num_qubits)), "compiled_num_qubits": int(circuit.num_qubits)}

    def _fake_depth(compiled: QuantumCircuit) -> int:
        return 10 if int(compiled.metadata["call"]) == 1 else 15

    def _fake_stats(compiled: QuantumCircuit) -> dict[str, object]:
        return {"compiled_count_2q": 20, "compiled_cx_count": 20, "compiled_ecr_count": 0, "compiled_op_counts": {"cx": 20}}

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _fake_compile)
    monkeypatch.setattr(oracle_mod, "safe_circuit_depth", _fake_depth)
    monkeypatch.setattr(oracle_mod, "compiled_gate_stats", _fake_stats)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(
            mode="transpile_single_v1",
            requested_backend_name="FakeMarrakesh",
            weight_2q=1.0,
            weight_depth=2.0,
            weight_size=0.0,
        ),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    op_a = _term("a", "xeeeee")
    op_b = _term("b", "zxeeee")
    snapshot = oracle.snapshot_base([op_a])
    estimate = oracle.estimate_insertion(snapshot, candidate_term=op_b, position_id=1, proxy_baseline=None)

    assert estimate.raw_delta_compiled_depth == pytest.approx(5.0)
    assert estimate.penalty_total == pytest.approx(10.0)
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["penalty_weight_depth"] == pytest.approx(2.0)


def test_backend_compile_oracle_incremental_prefix_suffix_uses_prefix_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve(**kwargs):
        return (
            (
                ResolvedBackendTarget(
                    requested_name="FakeMarrakesh",
                    resolved_name="FakeMarrakesh",
                    resolution_kind="fake_exact",
                    using_fake_backend=True,
                    backend_obj=_BackendStub("FakeMarrakesh"),
                    target_snapshot={"backend_name": "FakeMarrakesh"},
                ),
            ),
            [{"requested_name": "FakeMarrakesh", "resolved_name": "FakeMarrakesh", "success": True}],
        )

    calls: list[dict[str, object]] = []

    def _fake_compile(
        circuit,
        backend,
        *,
        seed_transpiler: int,
        optimization_level: int = 1,
        initial_layout=None,
    ):
        call_index = len(calls) + 1
        layout = None if initial_layout is None else tuple(int(q) for q in initial_layout)
        calls.append(
            {
                "call": call_index,
                "instruction_count": len(circuit.data),
                "initial_layout": layout,
            }
        )
        compiled = QuantumCircuit(circuit.num_qubits)
        compiled.metadata = {"call": call_index}
        final_layout = tuple(reversed(range(circuit.num_qubits))) if layout is None else layout
        return {
            "compiled": compiled,
            "logical_to_physical": final_layout,
            "compiled_num_qubits": int(circuit.num_qubits),
        }

    def _fake_depth(compiled: QuantumCircuit) -> int:
        call = int(compiled.metadata["call"])
        return {1: 100, 2: 0, 3: 7}[call]

    def _fake_stats(compiled: QuantumCircuit) -> dict[str, object]:
        call = int(compiled.metadata["call"])
        count_2q = {1: 100, 2: 0, 3: 7}[call]
        return {
            "compiled_count_2q": count_2q,
            "compiled_cx_count": count_2q,
            "compiled_ecr_count": 0,
            "compiled_op_counts": {"cx": count_2q},
        }

    import pipelines.static_adapt.hh_backend_compile_oracle as oracle_mod

    monkeypatch.setattr(oracle_mod, "resolve_backend_targets", _fake_resolve)
    monkeypatch.setattr(oracle_mod, "compile_circuit_for_backend", _fake_compile)
    monkeypatch.setattr(oracle_mod, "safe_circuit_depth", _fake_depth)
    monkeypatch.setattr(oracle_mod, "compiled_gate_stats", _fake_stats)

    oracle = BackendCompileOracle(
        config=BackendCompileConfig(mode="incremental_prefix_suffix_v1", requested_backend_name="FakeMarrakesh"),
        num_qubits=6,
        ref_state=np.array([1.0] + [0.0] * 63, dtype=complex),
    )
    op_a = _term("a", "xeeeee")
    op_b = _term("b", "zxeeee")
    op_c = _term("c", "zzeeee")
    snapshot = oracle.snapshot_base([op_a, op_c])
    assert len(calls) == 1

    estimate = oracle.estimate_insertion(snapshot, candidate_term=op_b, position_id=2, proxy_baseline=None)

    prefix_layout = tuple(reversed(range(6)))
    assert len(calls) == 3
    assert calls[0]["initial_layout"] is None
    assert calls[1]["initial_layout"] == prefix_layout
    assert calls[2]["initial_layout"] == prefix_layout
    assert estimate.source_mode == "backend_incremental_prefix_suffix_v1"
    assert estimate.hardware_cost_source == "backend_incremental_prefix_suffix_v1"
    assert estimate.raw_delta_compiled_count_2q == pytest.approx(7.0)
    assert estimate.selected_backend_row is not None
    assert estimate.selected_backend_row["source_mode"] == "backend_incremental_prefix_suffix_v1"
    meta = estimate.selected_backend_row["incremental_prefix_suffix"]
    assert meta["strict_no_proxy_fallback"] is True
    assert meta["prefix_depth"] == 2
    assert meta["base_tail_depth"] == 0
    assert meta["trial_tail_depth"] == 1
