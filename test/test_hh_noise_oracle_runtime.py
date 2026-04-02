from __future__ import annotations

import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.noise_oracle_runtime as nor
from pipelines.exact_bench.noise_oracle_runtime import (
    ExpectationOracle,
    OracleConfig,
    RawMeasurementOracle,
    _ansatz_to_circuit,
    _pauli_poly_to_sparse_pauli_op,
    build_runtime_layout_circuit,
    compile_circuit_for_local_backend,
    normalize_ideal_reference_symmetry_mitigation,
    normalize_runtime_estimator_profile_config,
    normalize_runtime_session_policy_config,
    normalize_sampler_raw_runtime_config,
    normalize_symmetry_mitigation_config,
    preflight_backend_scheduled_fake_backend_environment,
)
from pipelines.hardcoded.adapt_circuit_execution import (
    bind_parameterized_ansatz_circuit,
    build_ansatz_circuit,
    build_parameterized_ansatz_plan,
)
from src.quantum.ansatz_parameterization import build_parameter_layout
from src.quantum.compiled_ansatz import CompiledAnsatzExecutor
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.qubitization_module import PauliTerm
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import AnsatzTerm, HubbardHolsteinTermwiseAnsatz


def test_pauli_poly_to_sparse_pauli_op_preserves_exyz_to_ixyz_mapping() -> None:
    poly = PauliPolynomial(3)
    poly.add_term(PauliTerm(3, ps="xez", pc=0.5))
    poly.add_term(PauliTerm(3, ps="eey", pc=-0.25))
    poly.add_term(PauliTerm(3, ps="xez", pc=0.5))

    qop = _pauli_poly_to_sparse_pauli_op(poly)
    coeffs = {lbl: complex(c) for lbl, c in qop.to_list()}

    assert "XIZ" in coeffs
    assert "IIY" in coeffs
    assert coeffs["XIZ"] == pytest.approx(1.0 + 0.0j)
    assert coeffs["IIY"] == pytest.approx(-0.25 + 0.0j)


def test_ansatz_to_circuit_matches_prepare_state_for_hh_termwise_small_case() -> None:
    num_sites = 2
    num_particles = (1, 1)
    ansatz = HubbardHolsteinTermwiseAnsatz(
        dims=num_sites,
        J=1.0,
        U=4.0,
        omega0=1.0,
        g=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        pbc=True,
    )
    psi_ref = hubbard_holstein_reference_state(
        dims=num_sites,
        num_particles=num_particles,
        n_ph_max=1,
        boson_encoding="binary",
        indexing="blocked",
    )
    rng = np.random.default_rng(123)
    theta = 0.05 * rng.normal(size=int(ansatz.num_parameters))

    qc = _ansatz_to_circuit(
        ansatz,
        theta,
        num_qubits=int(ansatz.nq),
        reference_state=np.asarray(psi_ref, dtype=complex),
    )
    psi_circuit = np.asarray(Statevector.from_instruction(qc).data, dtype=complex).reshape(-1)
    psi_expected = np.asarray(ansatz.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)

    fidelity = float(abs(np.vdot(psi_expected, psi_circuit)) ** 2)
    assert fidelity > 1.0 - 1e-10


def test_build_runtime_layout_circuit_matches_compiled_executor_small_case() -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    theta = np.asarray([0.2], dtype=float)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    qc = build_runtime_layout_circuit(layout, theta, 1, reference_state=psi_ref)
    psi_circuit = np.asarray(Statevector.from_instruction(qc).data, dtype=complex).reshape(-1)
    executor = CompiledAnsatzExecutor(
        [term],
        parameterization_mode="per_pauli_term",
        parameterization_layout=layout,
    )
    psi_expected = np.asarray(executor.prepare_state(theta, psi_ref), dtype=complex).reshape(-1)
    fidelity = float(abs(np.vdot(psi_expected, psi_circuit)) ** 2)
    assert fidelity > 1.0 - 1e-10


def test_ideal_oracle_matches_statevector_expectation() -> None:
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(noise_mode="ideal", oracle_repeats=3, oracle_aggregate="mean")
    ) as oracle:
        est = oracle.evaluate(qc, obs)

    exact = float(np.real(Statevector.from_instruction(qc).expectation_value(obs)))
    assert est.mean == pytest.approx(exact, abs=1e-10)
    assert est.std == pytest.approx(0.0, abs=1e-12)
    assert est.stdev == pytest.approx(0.0, abs=1e-12)
    assert est.stderr == pytest.approx(0.0, abs=1e-12)
    assert str(est.aggregate) == "mean"


def test_oracle_mitigation_config_is_normalized_and_recorded() -> None:
    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("I", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            mitigation={"mode": "zne", "zne_scales": "1.0,2.0,3.0", "dd_sequence": "XY4"},
        )
    ) as oracle:
        _ = oracle.evaluate(qc, obs)
        mit = oracle.config.mitigation
        assert isinstance(mit, dict)
        assert mit["mode"] == "zne"
        assert mit["zne_scales"] == [1.0, 2.0, 3.0]
        assert mit["dd_sequence"] == "XY4"
        assert mit["local_readout_strategy"] is None
        assert dict(oracle.backend_info.details.get("mitigation", {})) == mit


def test_normalize_mitigation_config_accepts_local_gate_twirling_flag() -> None:
    cfg = nor.normalize_mitigation_config(
        {
            "mode": "readout",
            "local_readout_strategy": "mthree",
            "local_gate_twirling": True,
        }
    )

    assert cfg["mode"] == "readout"
    assert cfg["local_readout_strategy"] == "mthree"
    assert cfg["local_gate_twirling"] is True


def test_backend_scheduled_rejects_active_symmetry_with_readout() -> None:
    with pytest.raises(ValueError, match="readout mitigation is not combinable"):
        ExpectationOracle(
            OracleConfig(
                noise_mode="backend_scheduled",
                shots=128,
                seed=7,
                backend_name="FakeGuadalupeV2",
                use_fake_backend=True,
                mitigation={"mode": "readout", "local_readout_strategy": "mthree"},
                symmetry_mitigation={
                    "mode": "postselect_diag_v1",
                    "num_sites": 1,
                    "ordering": "blocked",
                    "sector_n_up": 1,
                    "sector_n_dn": 0,
                },
            )
        )


def test_backend_scheduled_deterministic_with_fixed_seed_and_fake_backend() -> None:
    pytest.importorskip("mthree")
    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    cfg = OracleConfig(
        noise_mode="backend_scheduled",
        shots=256,
        seed=111,
        oracle_repeats=4,
        oracle_aggregate="mean",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
    )

    with ExpectationOracle(cfg) as oracle_a:
        est_a = oracle_a.evaluate(qc, obs)
    with ExpectationOracle(cfg) as oracle_b:
        est_b = oracle_b.evaluate(qc, obs)

    assert est_a.mean == pytest.approx(est_b.mean, abs=1e-12)
    assert est_a.stderr == pytest.approx(est_b.stderr, abs=1e-12)


def test_backend_scheduled_collect_term_sample_returns_raw_payload() -> None:
    qc = QuantumCircuit(1)
    qc.x(0)
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        group_payload = oracle.collect_backend_scheduled_group_sample(qc, "Z", repeat_idx=0)
        payload = oracle.collect_backend_scheduled_term_sample(qc, "Z", repeat_idx=0)

    assert group_payload["basis_label"] == "Z"
    assert group_payload["measured_logical_qubits"] == [0]
    assert isinstance(group_payload["counts"], dict)
    assert int(payload["repeat_index"]) == 0
    assert int(payload["shots"]) == 64
    assert np.isfinite(float(payload["expectation"]))
    assert isinstance(payload["counts"], dict)
    assert isinstance(payload["term_details"], dict)
    assert "active_physical_qubits" in payload["term_details"]
    assert isinstance(oracle.backend_info.details.get("readout_mitigation", {}), dict)
    assert isinstance(oracle.backend_info.details.get("local_gate_twirling", {}), dict)
    assert isinstance(oracle.backend_info.details.get("local_dynamical_decoupling", {}), dict)


def test_backend_scheduled_mthree_readout_records_details() -> None:
    pytest.importorskip("mthree")
    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=128,
            seed=7,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            mitigation={"mode": "readout", "local_readout_strategy": "mthree"},
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        details = dict(oracle.backend_info.details.get("readout_mitigation", {}))

    assert np.isfinite(est.mean)
    assert details.get("mode") == "readout"
    assert details.get("strategy") == "mthree"
    assert details.get("applied") is True
    assert int(details.get("calibration_cache_size", 0)) >= 1


def test_backend_scheduled_local_gate_twirling_records_details() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=128,
            seed=7,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            mitigation={"mode": "none", "local_gate_twirling": True},
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        details = dict(oracle.backend_info.details.get("local_gate_twirling", {}))

    assert np.isfinite(est.mean)
    assert details.get("requested") is True
    assert details.get("applied") is True
    assert details.get("engine") == "qiskit.circuit.pauli_twirl_2q_gates"
    assert int(details.get("seed")) == 8
    assert int(details.get("twirled_metrics", {}).get("compiled_two_qubit_count", 0)) >= 1


def test_backend_scheduled_local_dd_with_gate_twirling_records_details() -> None:
    pytest.importorskip("mthree")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            mitigation={
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "local_gate_twirling": True,
                "dd_sequence": "XpXm",
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        twirling = dict(oracle.backend_info.details.get("local_gate_twirling", {}))
        dd = dict(oracle.backend_info.details.get("local_dynamical_decoupling", {}))

    assert np.isfinite(est.mean)
    assert twirling.get("requested") is True
    assert twirling.get("applied") is True
    assert dd.get("requested") is True
    assert dd.get("sequence") == "XPXM"


def test_backend_scheduled_local_dd_with_gate_twirling_records_details_without_readout() -> None:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            mitigation={
                "mode": "none",
                "local_gate_twirling": True,
                "dd_sequence": "XpXm",
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        twirling = dict(oracle.backend_info.details.get("local_gate_twirling", {}))
        dd = dict(oracle.backend_info.details.get("local_dynamical_decoupling", {}))
        readout = dict(oracle.backend_info.details.get("readout_mitigation", {}))

    assert np.isfinite(est.mean)
    assert twirling.get("requested") is True
    assert twirling.get("applied") is True
    assert dd.get("requested") is True
    assert dd.get("sequence") == "XPXM"
    assert readout.get("mode") == "none"
    assert readout.get("applied") is False


def test_backend_scheduled_rejects_unsupported_local_dd_sequence() -> None:
    pytest.importorskip("mthree")
    with pytest.raises(ValueError, match="only dd_sequence='XpXm'"):
        ExpectationOracle(
            OracleConfig(
                noise_mode="backend_scheduled",
                shots=64,
                seed=7,
                oracle_repeats=1,
                backend_name="FakeGuadalupeV2",
                use_fake_backend=True,
                mitigation={
                    "mode": "readout",
                    "local_readout_strategy": "mthree",
                    "dd_sequence": "XY4",
                },
            )
        )


def test_backend_scheduled_local_dd_records_details() -> None:
    pytest.importorskip("mthree")
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    obs = SparsePauliOp.from_list([("ZZ", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            mitigation={
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "dd_sequence": "XpXm",
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        details = dict(oracle.backend_info.details.get("local_dynamical_decoupling", {}))

    assert np.isfinite(est.mean)
    assert details.get("requested") is True
    assert details.get("sequence") == "XPXM"
    assert details.get("scheduling_method") == "alap"
    assert "applied" in details


def test_backend_scheduled_local_dd_invalid_t2_retries_with_sanitized_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Backend:
        target = object()
        name = "FakeUnit"

    class _FailingTarget:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, circuit, *, shots=None, seed_simulator=None):
            self.calls += 1
            raise RuntimeError(
                "NoiseError: 'Invalid T_2 relaxation time parameter: T_2 greater than 2 * T_1.'"
            )

    class _Job:
        def __init__(self, counts: dict[str, int]) -> None:
            self._counts = counts

        def result(self):
            return self

        def get_counts(self):
            return dict(self._counts)

    class _SuccessTarget:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, circuit, *, shots=None, seed_simulator=None):
            self.calls += 1
            return _Job({"0": int(shots or 0)})

    failing_target = _FailingTarget()
    success_target = _SuccessTarget()
    monkeypatch.setattr(nor, "_resolve_mthree", lambda: object())
    monkeypatch.setattr(
        nor,
        "_resolve_noise_backend",
        lambda cfg: (_Backend(), "FakeUnit", True),
    )
    monkeypatch.setattr(
        nor,
        "_apply_mthree_readout_correction",
        lambda **kwargs: (
            {"0": 1.0},
            {
                "method": "stub",
                "time": 0.0,
                "dimension": 1,
                "mitigation_overhead": 1.0,
                "negative_mass": 0.0,
                "shots": 64,
            },
        ),
    )

    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeUnit",
            use_fake_backend=True,
            mitigation={
                "mode": "readout",
                "local_readout_strategy": "mthree",
                "dd_sequence": "XpXm",
            },
        )
    ) as oracle:
        oracle._backend_target = failing_target
        monkeypatch.setattr(
            oracle,
            "_get_backend_scheduled_base",
            lambda circuit: {"compiled": QuantumCircuit(1), "logical_to_physical": (0,)},
        )
        monkeypatch.setattr(oracle, "_ensure_mthree_calibration", lambda active_physical_qubits: None)
        monkeypatch.setattr(
            oracle,
            "_maybe_apply_local_dd_to_measured_term_circuit",
            lambda circuit, *, logical_to_physical, dd_sequence: (
                circuit,
                {
                    "requested": True,
                    "applied": True,
                    "sequence": "XPXM",
                    "reason": "",
                    "scheduling_method": "alap",
                    "timing_supported": True,
                    "dd_qubits": [0],
                    "added_x": 2,
                    "added_y": 0,
                },
            ),
        )
        monkeypatch.setattr(
            oracle,
            "_get_backend_scheduled_local_dd_retry_target",
            lambda: (
                success_target,
                {
                    "execution_target_kind": "aer_sanitized_from_backend",
                    "noise_model_sanitization": {
                        "applied": True,
                        "strategy": "clamp_t2_to_2t1",
                        "affected_qubits": [100],
                        "retry_trigger": "invalid_t2_gt_2t1",
                    },
                },
            ),
        )
        est = oracle.evaluate(qc, obs)
        details = dict(oracle.backend_info.details.get("local_dynamical_decoupling", {}))

    assert np.isfinite(est.mean)
    assert est.mean == pytest.approx(1.0, abs=1e-12)
    assert failing_target.calls == 1
    assert success_target.calls == 1
    assert details.get("execution_target_kind") == "aer_sanitized_from_backend"
    assert dict(details.get("noise_model_sanitization", {})).get("applied") is True


def test_backend_scheduled_non_dd_path_does_not_invoke_local_dd_retry_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Backend:
        target = object()
        name = "FakeUnit"

    class _Job:
        def __init__(self, counts: dict[str, int]) -> None:
            self._counts = counts

        def result(self):
            return self

        def get_counts(self):
            return dict(self._counts)

    class _SuccessTarget:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, circuit, *, shots=None, seed_simulator=None):
            self.calls += 1
            return _Job({"0": int(shots or 0)})

    success_target = _SuccessTarget()
    helper_calls = {"count": 0}
    monkeypatch.setattr(nor, "_resolve_mthree", lambda: object())
    monkeypatch.setattr(
        nor,
        "_resolve_noise_backend",
        lambda cfg: (_Backend(), "FakeUnit", True),
    )
    monkeypatch.setattr(
        nor,
        "_apply_mthree_readout_correction",
        lambda **kwargs: (
            {"0": 1.0},
            {
                "method": "stub",
                "time": 0.0,
                "dimension": 1,
                "mitigation_overhead": 1.0,
                "negative_mass": 0.0,
                "shots": 64,
            },
        ),
    )

    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeUnit",
            use_fake_backend=True,
            mitigation={"mode": "readout", "local_readout_strategy": "mthree"},
        )
    ) as oracle:
        oracle._backend_target = success_target
        monkeypatch.setattr(
            oracle,
            "_get_backend_scheduled_base",
            lambda circuit: {"compiled": QuantumCircuit(1), "logical_to_physical": (0,)},
        )
        monkeypatch.setattr(oracle, "_ensure_mthree_calibration", lambda active_physical_qubits: None)
        monkeypatch.setattr(
            oracle,
            "_maybe_apply_local_dd_to_measured_term_circuit",
            lambda circuit, *, logical_to_physical, dd_sequence: (
                circuit,
                {
                    "requested": False,
                    "applied": False,
                    "sequence": None,
                    "reason": "disabled",
                    "scheduling_method": "alap",
                    "timing_supported": False,
                },
            ),
        )

        def _unexpected_retry():
            helper_calls["count"] += 1
            raise AssertionError("DD retry helper should not be called for non-DD path.")

        monkeypatch.setattr(oracle, "_get_backend_scheduled_local_dd_retry_target", _unexpected_retry)
        est = oracle.evaluate(qc, obs)

    assert np.isfinite(est.mean)
    assert est.mean == pytest.approx(1.0, abs=1e-12)
    assert success_target.calls == 1
    assert helper_calls["count"] == 0


def test_backend_scheduled_attribution_rejects_local_gate_twirling() -> None:
    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            mitigation={"mode": "none", "local_gate_twirling": True},
        )
    ) as oracle:
        with pytest.raises(ValueError, match="local gate twirling off"):
            oracle.evaluate_backend_scheduled_attribution(qc, obs)


def test_backend_scheduled_attribution_reuses_single_compile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nor, "_load_fake_backend", lambda name: (object(), str(name or "FakeGuadalupeV2")))
    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        calls = {"base": 0, "eval": []}
        compiled = QuantumCircuit(1)
        readout_target = object()
        gate_target = object()

        def _fake_base(circuit: QuantumCircuit) -> dict[str, object]:
            calls["base"] += 1
            return {"compiled": compiled, "logical_to_physical": (0,)}

        def _fake_target(slice_name: str) -> tuple[object, dict[str, object]]:
            components = {
                "readout_only": {"gate_stateprep": False, "readout": True},
                "gate_stateprep_only": {"gate_stateprep": True, "readout": False},
                "full": {"gate_stateprep": True, "readout": True},
            }
            target = {"readout_only": readout_target, "gate_stateprep_only": gate_target, "full": object()}[slice_name]
            return target, {
                "components": dict(components[slice_name]),
                "execution_target_kind": "stub",
                "attribution_slice": str(slice_name),
                "shared_compile_reused": True,
            }

        def _fake_eval(
            compiled_base: QuantumCircuit,
            logical_to_physical: tuple[int, ...],
            observable: SparsePauliOp,
            *,
            execution_target: object | None = None,
            attribution_slice: str | None = None,
            target_details: dict[str, object] | None = None,
        ) -> tuple[nor.OracleEstimate, dict[str, object]]:
            calls["eval"].append(
                {
                    "compiled_base": compiled_base,
                    "logical_to_physical": tuple(logical_to_physical),
                    "execution_target": execution_target,
                    "attribution_slice": attribution_slice,
                    "target_details": dict(target_details or {}),
                }
            )
            return (
                nor.OracleEstimate(
                    mean=0.25,
                    std=0.0,
                    stdev=0.0,
                    stderr=0.01,
                    n_samples=2,
                    raw_values=[0.25, 0.25],
                    aggregate="mean",
                ),
                {
                    "readout_mitigation": {"mode": "none", "applied": False},
                    "attribution_slice": attribution_slice,
                    "shared_compile_reused": True,
                    **dict(target_details or {}),
                },
            )

        monkeypatch.setattr(oracle, "_get_backend_scheduled_base", _fake_base)
        monkeypatch.setattr(oracle, "_get_backend_scheduled_attribution_target", _fake_target)
        monkeypatch.setattr(oracle, "_evaluate_backend_scheduled_with_target", _fake_eval)

        payload = oracle.evaluate_backend_scheduled_attribution(qc, obs)

    assert calls["base"] == 1
    assert len(calls["eval"]) == 3
    assert payload["shared_compile"]["requested_slices"] == [
        "readout_only",
        "gate_stateprep_only",
        "full",
    ]
    assert calls["eval"][0]["compiled_base"] is compiled
    assert calls["eval"][1]["compiled_base"] is compiled
    assert calls["eval"][2]["compiled_base"] is compiled
    assert calls["eval"][0]["execution_target"] is readout_target
    assert calls["eval"][1]["execution_target"] is gate_target
    assert calls["eval"][2]["execution_target"] is None
    assert payload["slices"]["readout_only"]["components"]["readout"] is True
    assert payload["slices"]["gate_stateprep_only"]["components"]["gate_stateprep"] is True
    assert payload["slices"]["full"]["backend_info"]["details"]["shared_compile_reused"] is True


def test_compile_circuit_for_local_backend_returns_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    qc = QuantumCircuit(2)
    compiled = SimpleNamespace(
        num_qubits=3,
        layout=SimpleNamespace(final_index_layout=lambda: [5, 4, 3]),
    )
    monkeypatch.setattr(
        nor,
        "_compile_circuit_for_backend_shared",
        lambda circuit, backend, *, seed_transpiler, optimization_level=1: {
            "compiled": compiled,
            "logical_to_physical": (5, 4),
            "compiled_num_qubits": 3,
        },
    )

    out = compile_circuit_for_local_backend(qc, object(), seed_transpiler=17, optimization_level=2)

    assert out["compiled"] is compiled
    assert out["logical_to_physical"] == (5, 4)
    assert out["compiled_num_qubits"] == 3


def test_backend_scheduled_base_uses_shared_compile_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nor, "_load_fake_backend", lambda name: (object(), str(name or "FakeGuadalupeV2")))
    calls: list[dict[str, object]] = []

    def _fake_compile(circuit: QuantumCircuit, backend: object, *, seed_transpiler: int, optimization_level: int = 1) -> dict[str, object]:
        calls.append(
            {
                "circuit": circuit,
                "backend": backend,
                "seed_transpiler": seed_transpiler,
                "optimization_level": optimization_level,
            }
        )
        return {
            "compiled": QuantumCircuit(1),
            "logical_to_physical": (2,),
            "compiled_num_qubits": 5,
        }

    monkeypatch.setattr(nor, "compile_circuit_for_local_backend", _fake_compile)
    qc = QuantumCircuit(1)
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=13,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        first = oracle._get_backend_scheduled_base(qc)
        second = oracle._get_backend_scheduled_base(qc)

    assert len(calls) == 1
    assert first is second
    assert calls[0]["seed_transpiler"] == 13
    assert calls[0]["optimization_level"] == 1
    assert first["logical_to_physical"] == (2,)


def test_backend_scheduled_base_honors_explicit_compile_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nor, "_load_fake_backend", lambda name: (object(), str(name or "FakeGuadalupeV2")))
    calls: list[dict[str, object]] = []

    def _fake_compile(circuit: QuantumCircuit, backend: object, *, seed_transpiler: int, optimization_level: int = 1) -> dict[str, object]:
        compiled = QuantumCircuit(1)
        compiled.x(0)
        calls.append(
            {
                "seed_transpiler": seed_transpiler,
                "optimization_level": optimization_level,
                "compiled": compiled,
            }
        )
        return {
            "compiled": compiled,
            "logical_to_physical": (4,),
            "compiled_num_qubits": 7,
        }

    monkeypatch.setattr(nor, "compile_circuit_for_local_backend", _fake_compile)
    qc = QuantumCircuit(1)
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=13,
            seed_transpiler=29,
            transpile_optimization_level=2,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        base = oracle._get_backend_scheduled_base(qc)
        details = dict(oracle.backend_info.details)

    assert len(calls) == 1
    assert calls[0]["seed_transpiler"] == 29
    assert calls[0]["optimization_level"] == 2
    assert base["logical_to_physical"] == (4,)
    assert details["transpile_seed"] == 29
    assert details["transpile_optimization_level"] == 2
    assert details["compiled_num_qubits"] == 7
    assert details["compiled_two_qubit_count"] == 0
    assert details["compiled_depth"] >= 1


def test_symmetry_mitigation_config_is_normalized() -> None:
    cfg = normalize_symmetry_mitigation_config(
        {
            "mode": "postselect_diag_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        }
    )
    assert cfg == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }


def test_ideal_reference_symmetry_mitigation_downgrades_runtime_to_verify_only() -> None:
    cfg = normalize_ideal_reference_symmetry_mitigation(
        {
            "mode": "projector_renorm_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        },
        noise_mode="runtime",
    )
    assert cfg["mode"] == "verify_only"
    assert cfg["num_sites"] == 2
    assert cfg["sector_n_up"] == 1
    assert cfg["sector_n_dn"] == 1


def test_postselect_diag_v1_filters_to_target_sector_for_diagonal_observable() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            oracle_repeats=2,
            oracle_aggregate="mean",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert est.mean == pytest.approx(1.0, abs=1e-10)
    assert sym.get("applied_mode") == "postselect_diag_v1"
    assert sym.get("retained_fraction_mean") == pytest.approx(0.5, abs=1e-10)
    assert sym.get("sector_probability_mean") == pytest.approx(0.5, abs=1e-10)


def test_projector_renorm_v1_matches_postselect_diag_for_diagonal_case() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])
    sym_cfg = {
        "mode": "postselect_diag_v1",
        "num_sites": 1,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 0,
    }

    with ExpectationOracle(OracleConfig(noise_mode="ideal", symmetry_mitigation=sym_cfg)) as post_oracle:
        post_est = post_oracle.evaluate(qc, obs)

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            symmetry_mitigation={**sym_cfg, "mode": "projector_renorm_v1"},
        )
    ) as proj_oracle:
        proj_est = proj_oracle.evaluate(qc, obs)
        sym = dict(proj_oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert proj_est.mean == pytest.approx(post_est.mean, abs=1e-10)
    assert proj_est.mean == pytest.approx(1.0, abs=1e-10)
    assert sym.get("applied_mode") == "projector_renorm_v1"
    assert sym.get("estimator_form") == "projector_ratio_diag_v1"


def test_symmetry_mitigation_non_diagonal_falls_back_to_verify_only() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("XI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 1,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert np.isfinite(est.mean)
    assert sym.get("applied_mode") == "verify_only"
    assert sym.get("fallback_reason") == "observable_not_diagonal"


def test_symmetry_mitigation_zero_retained_probability_falls_back_explicitly() -> None:
    qc = QuantumCircuit(2)
    qc.x(0)
    qc.h(1)
    obs = SparsePauliOp.from_list([("ZI", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="ideal",
            symmetry_mitigation={
                "mode": "postselect_diag_v1",
                "num_sites": 1,
                "ordering": "blocked",
                "sector_n_up": 0,
                "sector_n_dn": 0,
            },
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)
        sym = dict(oracle.backend_info.details.get("symmetry_mitigation", {}))

    assert np.isfinite(est.mean)
    assert sym.get("applied_mode") == "verify_only"
    assert "zero probability mass" in str(sym.get("fallback_reason", ""))


def test_shots_oracle_standard_error_improves_with_more_repeats() -> None:
    pytest.importorskip("qiskit_aer")
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    errs_r2: list[float] = []
    errs_r8: list[float] = []
    for seed in range(30, 40):
        with ExpectationOracle(
            OracleConfig(
                noise_mode="shots",
                shots=128,
                seed=seed,
                oracle_repeats=2,
                oracle_aggregate="mean",
            )
        ) as o2:
            e2 = o2.evaluate(qc, obs)
        with ExpectationOracle(
            OracleConfig(
                noise_mode="shots",
                shots=128,
                seed=seed,
                oracle_repeats=8,
                oracle_aggregate="mean",
            )
        ) as o8:
            e8 = o8.evaluate(qc, obs)

        errs_r2.append(abs(float(e2.mean)))
        errs_r8.append(abs(float(e8.mean)))
        assert float(e8.stderr) <= float(e2.stderr) + 1e-9

    assert float(np.mean(errs_r8)) <= float(np.mean(errs_r2)) + 1e-9


def test_aer_noise_mode_deterministic_with_fixed_seed_and_fake_backend() -> None:
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("qiskit_ibm_runtime")

    qc = QuantumCircuit(1)
    qc.x(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])
    cfg = OracleConfig(
        noise_mode="aer_noise",
        shots=256,
        seed=111,
        oracle_repeats=4,
        oracle_aggregate="mean",
        backend_name="FakeManilaV2",
        use_fake_backend=True,
    )

    with ExpectationOracle(cfg) as oracle_a:
        est_a = oracle_a.evaluate(qc, obs)
    with ExpectationOracle(cfg) as oracle_b:
        est_b = oracle_b.evaluate(qc, obs)

    assert est_a.raw_values == pytest.approx(est_b.raw_values)
    assert est_a.mean == pytest.approx(est_b.mean)
    assert est_a.std == pytest.approx(est_b.std)
    assert est_a.stdev == pytest.approx(est_b.stdev)
    assert est_a.stderr == pytest.approx(est_b.stderr)


def test_forced_aer_failure_triggers_sampler_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_cfg):
        raise RuntimeError("OMP: Error #178: Function Can't open SHM2 failed")

    monkeypatch.setattr(nor, "_build_estimator", _boom)
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    with ExpectationOracle(
        OracleConfig(
            noise_mode="shots",
            shots=128,
            seed=9,
            oracle_repeats=3,
            allow_aer_fallback=True,
        )
    ) as oracle:
        est = oracle.evaluate(qc, obs)

    assert np.isfinite(est.mean)
    assert bool(oracle.backend_info.details.get("fallback_used")) is True
    assert bool(oracle.backend_info.details.get("aer_failed")) is True
    assert str(oracle.backend_info.estimator_kind).startswith("qiskit.primitives.StatevectorSampler")


def test_sampler_fallback_deterministic_with_fixed_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(_cfg):
        raise RuntimeError("OMP: Error #178: Function Can't open SHM2 failed")

    monkeypatch.setattr(nor, "_build_estimator", _boom)
    qc = QuantumCircuit(1)
    qc.h(0)
    obs = SparsePauliOp.from_list([("Z", 1.0)])

    cfg = OracleConfig(
        noise_mode="shots",
        shots=128,
        seed=77,
        oracle_repeats=4,
        allow_aer_fallback=True,
    )
    with ExpectationOracle(cfg) as oa:
        ea = oa.evaluate(qc, obs)
    with ExpectationOracle(cfg) as ob:
        eb = ob.evaluate(qc, obs)

    assert ea.raw_values == pytest.approx(eb.raw_values)
    assert ea.mean == pytest.approx(eb.mean)


def test_backend_scheduled_preflight_detects_openmp_shm_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nor._BACKEND_SCHEDULED_PREFLIGHT_OK_CACHE.clear()

    def _fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=134,
            stdout="",
            stderr="OMP: Error #178: Function Can't open SHM2 failed:\nOMP: System error #2: No such file or directory\n",
        )

    monkeypatch.setattr(nor.subprocess, "run", _fake_run)
    cfg = OracleConfig(
        noise_mode="backend_scheduled",
        shots=128,
        seed=7,
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
    )

    with pytest.raises(RuntimeError, match="backend_scheduled preflight failed due to OpenMP shared-memory restrictions"):
        preflight_backend_scheduled_fake_backend_environment(cfg)


def test_backend_scheduled_preflight_caches_success(monkeypatch: pytest.MonkeyPatch) -> None:
    nor._BACKEND_SCHEDULED_PREFLIGHT_OK_CACHE.clear()
    calls = {"count": 0}

    def _fake_run(*args, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            returncode=0,
            stdout="BACKEND_SCHEDULED_PREFLIGHT_OK 2\n",
            stderr="",
        )

    monkeypatch.setattr(nor.subprocess, "run", _fake_run)
    cfg = OracleConfig(
        noise_mode="backend_scheduled",
        shots=128,
        seed=7,
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
    )

    preflight_backend_scheduled_fake_backend_environment(cfg)
    preflight_backend_scheduled_fake_backend_environment(cfg)

    assert calls["count"] == 1


def test_runtime_profile_main_twirled_readout_v1_defaults() -> None:
    cfg = normalize_runtime_estimator_profile_config("main_twirled_readout_v1")

    assert cfg["name"] == "main_twirled_readout_v1"
    assert cfg["resilience_level"] == 1
    assert cfg["measure_mitigation"] is True
    assert cfg["measure_twirling"] is True
    assert cfg["gate_twirling"] is True
    assert cfg["twirling_strategy"] == "active-accum"
    assert cfg["dd_enable"] is False
    assert cfg["zne_mitigation"] is False
    assert cfg["pec_mitigation"] is False


def test_runtime_profile_dd_probe_defaults() -> None:
    cfg = normalize_runtime_estimator_profile_config("dd_probe_twirled_readout_v1")

    assert cfg["name"] == "dd_probe_twirled_readout_v1"
    assert cfg["measure_mitigation"] is True
    assert cfg["measure_twirling"] is True
    assert cfg["gate_twirling"] is False
    assert cfg["dd_enable"] is True
    assert cfg["dd_sequence"] == "XpXm"
    assert cfg["zne_mitigation"] is False


def test_runtime_profile_final_audit_zne_defaults() -> None:
    cfg = normalize_runtime_estimator_profile_config("final_audit_zne_twirled_readout_v1")

    assert cfg["name"] == "final_audit_zne_twirled_readout_v1"
    assert cfg["measure_mitigation"] is True
    assert cfg["measure_twirling"] is True
    assert cfg["gate_twirling"] is True
    assert cfg["dd_enable"] is False
    assert cfg["zne_mitigation"] is True
    assert cfg["zne_noise_factors"] == [1.0, 3.0, 5.0]
    assert cfg["zne_extrapolator"] == ["exponential", "linear"]
    assert cfg["pec_mitigation"] is False


def test_runtime_session_policy_defaults() -> None:
    assert normalize_runtime_session_policy_config(None) == {"mode": "prefer_session"}
    assert normalize_runtime_session_policy_config("require_session") == {
        "mode": "require_session"
    }


def test_configure_runtime_estimator_options_applies_explicit_profile() -> None:
    class _Node:
        pass

    class _Estimator:
        def __init__(self) -> None:
            self.options = _Node()
            self.options.execution = _Node()
            self.options.resilience = _Node()
            self.options.resilience.zne = _Node()
            self.options.twirling = _Node()
            self.options.dynamical_decoupling = _Node()
            self.options.default_shots = None
            self.options.default_precision = None
            self.options.max_execution_time = None
            self.options.seed_estimator = None
            self.options.resilience_level = None
            self.options.execution.init_qubits = None
            self.options.resilience.measure_mitigation = None
            self.options.resilience.zne_mitigation = None
            self.options.resilience.pec_mitigation = None
            self.options.twirling.enable_measure = None
            self.options.twirling.enable_gates = None
            self.options.twirling.strategy = None
            self.options.dynamical_decoupling.enable = None
            self.options.dynamical_decoupling.sequence_type = None
            self.options.resilience.zne.noise_factors = None
            self.options.resilience.zne.extrapolator = None

    estimator = _Estimator()
    cfg = OracleConfig(noise_mode="runtime", shots=4096, seed=7)
    mitigation_cfg = nor.normalize_mitigation_config({"mode": "readout"})
    runtime_profile_cfg = normalize_runtime_estimator_profile_config(
        "main_twirled_readout_v1"
    )

    details = nor._configure_runtime_estimator_options(
        estimator,
        cfg=cfg,
        mitigation_cfg=mitigation_cfg,
        runtime_profile_cfg=runtime_profile_cfg,
    )

    assert estimator.options.default_shots == 4096
    assert estimator.options.seed_estimator == 7
    assert estimator.options.execution.init_qubits is True
    assert estimator.options.resilience_level == 1
    assert estimator.options.resilience.measure_mitigation is True
    assert estimator.options.resilience.zne_mitigation is False
    assert estimator.options.resilience.pec_mitigation is False
    assert estimator.options.twirling.enable_measure is True
    assert estimator.options.twirling.enable_gates is True
    assert estimator.options.twirling.strategy == "active-accum"
    assert estimator.options.dynamical_decoupling.enable is False
    assert details["engine"] == "runtime_profile"
    assert details["provider_strategy"] == "explicit_profile"
    assert details["applied"] is True


def test_parameterized_ansatz_plan_digest_stable_and_theta_independent() -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    plan_a = build_parameterized_ansatz_plan(layout, nq=1, ref_state=psi_ref)
    plan_b = build_parameterized_ansatz_plan(layout, nq=1, ref_state=psi_ref)

    assert plan_a.structure_digest == plan_b.structure_digest
    assert plan_a.reference_state_digest == plan_b.reference_state_digest
    assert plan_a.plan_digest == plan_b.plan_digest

    theta_a = np.asarray([0.1], dtype=float)
    theta_b = np.asarray([-0.2], dtype=float)
    _ = bind_parameterized_ansatz_circuit(plan_a, theta_a)
    _ = bind_parameterized_ansatz_circuit(plan_a, theta_b)

    assert plan_a.plan_digest == plan_b.plan_digest


def test_build_ansatz_circuit_matches_parameterized_plan_binding() -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    theta = np.asarray([0.2], dtype=float)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)

    plan = build_parameterized_ansatz_plan(layout, nq=1, ref_state=psi_ref)
    qc_bound = bind_parameterized_ansatz_circuit(plan, theta)
    qc_direct = build_ansatz_circuit(layout, theta, 1, ref_state=psi_ref)

    psi_bound = np.asarray(Statevector.from_instruction(qc_bound).data, dtype=complex).reshape(-1)
    psi_direct = np.asarray(Statevector.from_instruction(qc_direct).data, dtype=complex).reshape(-1)
    fidelity = float(abs(np.vdot(psi_bound, psi_direct)) ** 2)
    assert fidelity > 1.0 - 1e-10


def test_parameterized_measured_template_keeps_parameters_after_transpile() -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    psi_ref = np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)
    plan = build_parameterized_ansatz_plan(layout, nq=1, ref_state=psi_ref)
    measured, _active = nor._sparse_term_measurement_circuit(plan.circuit, "X")
    backend, _name = nor._load_fake_backend("FakeGuadalupeV2")
    compiled = compile_circuit_for_local_backend(
        measured,
        backend,
        seed_transpiler=0,
        optimization_level=1,
    )

    assert len(tuple(compiled["compiled"].parameters)) == 1


def test_bind_group_template_allows_compiled_parameter_subset() -> None:
    term = AnsatzTerm(
        label="op_z",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=1,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )

    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=32,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        bound = oracle._bind_group_template(
            {"compiled": QuantumCircuit(1, 1)},
            plan,
            np.asarray([0.3], dtype=float),
        )

    assert isinstance(bound, QuantumCircuit)
    assert len(tuple(bound.parameters)) == 0


def _raw_record(
    *,
    basis_label: str,
    repeat_index: int,
    counts: dict[str, int],
    measured_logical_qubits: tuple[int, ...],
    group_terms: tuple[dict[str, Any], ...],
    num_qubits: int = 1,
    evaluation_id: str = "eval-1",
    observable_family: str = "unit_test",
    semantic_tags: dict[str, Any] | None = None,
) -> nor.RawMeasurementRecord:
    shots = int(sum(int(v) for v in counts.values()))
    measured_physical_qubits = tuple(int(q) for q in measured_logical_qubits)
    logical_to_physical = tuple(range(int(num_qubits)))
    physical_to_logical = {str(int(q)): int(q) for q in logical_to_physical}
    return nor.RawMeasurementRecord(
        schema_version="raw_measurement_record_v1",
        record_id=f"record-{basis_label}-{repeat_index}",
        evaluation_id=str(evaluation_id),
        execution_surface="raw_measurement_v1",
        observable_family=str(observable_family),
        semantic_tags=dict(semantic_tags or {}),
        group_index=0,
        basis_label=str(basis_label),
        group_terms=group_terms,
        plan_digest="plan",
        structure_digest="structure",
        reference_state_digest=None,
        theta_runtime=(0.0,),
        theta_digest="theta",
        logical_parameter_count=1,
        runtime_parameter_count=1,
        num_qubits=int(num_qubits),
        measured_logical_qubits=tuple(measured_logical_qubits),
        measured_physical_qubits=measured_physical_qubits,
        logical_to_physical=logical_to_physical,
        physical_to_logical=physical_to_logical,
        compile_signature={"compiled_depth": 1},
        backend_snapshot={"backend_name": "stub"},
        transport="backend_run",
        call_path="backend_run_counts",
        job_records=tuple(),
        repeat_index=int(repeat_index),
        shots_requested=shots,
        shots_completed=shots,
        counts=dict(counts),
        requested_mitigation={"mode": "none"},
        requested_symmetry_mitigation={"mode": "off"},
        requested_runtime_profile={"name": "legacy_runtime_v0"},
        requested_runtime_session={"mode": "prefer_session"},
        parent_record_ids=tuple(),
        emitted_utc="2026-03-26T00:00:00Z",
    )


def test_reduce_grouped_counts_repeat_aligned_returns_raw_values() -> None:
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    records = [
        _raw_record(
            basis_label="Z",
            repeat_index=0,
            counts={"0": 16},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
        ),
        _raw_record(
            basis_label="Z",
            repeat_index=1,
            counts={"1": 16},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
        ),
    ]

    estimate = nor._reduce_grouped_counts_to_observable(
        observable,
        records,
        aggregate="mean",
    )

    assert estimate.reduction_mode == "repeat_aligned_full_observable"
    assert estimate.n_samples == 2
    assert estimate.raw_values == pytest.approx((1.0, -1.0))
    assert estimate.mean == pytest.approx(0.0, abs=1e-12)


def test_reduce_grouped_counts_falls_back_when_repeat_grid_incomplete() -> None:
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    records = [
        _raw_record(
            basis_label="Z",
            repeat_index=0,
            counts={"0": 16},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
        ),
        _raw_record(
            basis_label="Z",
            repeat_index=0,
            counts={"1": 16},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
        ),
    ]

    estimate = nor._reduce_grouped_counts_to_observable(
        observable,
        records,
        aggregate="mean",
    )

    assert estimate.reduction_mode == "weighted_term_fallback"
    assert estimate.aggregate == "weighted_term_fallback"


def test_reduce_grouped_counts_missing_expected_repeat_falls_back() -> None:
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    records = [
        _raw_record(
            basis_label="Z",
            repeat_index=0,
            counts={"0": 16},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
        ),
        _raw_record(
            basis_label="Z",
            repeat_index=2,
            counts={"1": 16},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
        ),
    ]

    estimate = nor._reduce_grouped_counts_to_observable(
        observable,
        records,
        aggregate="mean",
        expected_repeat_count=3,
    )

    assert estimate.reduction_mode == "weighted_term_fallback"
    assert estimate.aggregate == "weighted_term_fallback"


def test_raw_measurement_oracle_backend_scheduled_emits_records_and_estimate() -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=1,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=64,
            seed=7,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            raw_store_memory=True,
        )
    ) as oracle:
        bundle = oracle.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=observable,
            observable_family="unit_test",
        )

    assert bundle.transport == "backend_run"
    assert bundle.estimate.record_count == 2
    assert bundle.estimate.group_count == 1
    assert len(bundle.records) == 2
    assert len(oracle.memory_records) == 2
    assert bundle.records[0].basis_label == "Z"
    assert bundle.records[0].measured_logical_qubits == (0,)
    assert isinstance(bundle.records[0].compile_signature, dict)


def test_raw_measurement_oracle_compile_cache_reused_across_theta_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=1,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    compile_calls = {"count": 0}
    orig_compile = nor.compile_circuit_for_local_backend

    def _counting_compile(*args, **kwargs):
        compile_calls["count"] += 1
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(nor, "compile_circuit_for_local_backend", _counting_compile)
    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=32,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        _ = oracle.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=observable,
            observable_family="unit_test",
        )
        _ = oracle.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.2], dtype=float),
            observable=observable,
            observable_family="unit_test",
        )

    assert compile_calls["count"] == 1


def test_expectation_oracle_parameterized_compile_cache_reused_across_theta_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=1,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    compile_calls = {"count": 0}
    orig_compile = nor.compile_circuit_for_local_backend

    def _counting_compile(*args, **kwargs):
        compile_calls["count"] += 1
        return orig_compile(*args, **kwargs)

    monkeypatch.setattr(nor, "compile_circuit_for_local_backend", _counting_compile)
    with ExpectationOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=32,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        _ = oracle.evaluate_parameterized(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=observable,
        )
        _ = oracle.evaluate_parameterized(
            plan=plan,
            theta_runtime=np.asarray([0.2], dtype=float),
            observable=observable,
        )

    assert compile_calls["count"] == 1


def test_summarize_hh_full_register_z_records_reports_sector_weight_and_doublon() -> None:
    tags = {
        "symmetry_num_sites": 2,
        "symmetry_ordering": "blocked",
        "symmetry_sector_n_up": 1,
        "symmetry_sector_n_dn": 1,
    }
    records = [
        _raw_record(
            basis_label="ZZZZZZ",
            repeat_index=0,
            counts={"000101": 4, "001010": 4},
            measured_logical_qubits=(0, 1, 2, 3, 4, 5),
            group_terms=({"label": "ZZZZZZ", "coeff_re": 1.0, "coeff_im": 0.0},),
            num_qubits=6,
            observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            semantic_tags=tags,
        ),
        _raw_record(
            basis_label="ZZZZZZ",
            repeat_index=1,
            counts={"000101": 4, "000001": 4},
            measured_logical_qubits=(0, 1, 2, 3, 4, 5),
            group_terms=({"label": "ZZZZZZ", "coeff_re": 1.0, "coeff_im": 0.0},),
            num_qubits=6,
            observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            semantic_tags=tags,
        ),
    ]

    summary = nor._summarize_hh_full_register_z_records(records, expected_repeat_count=2)

    assert summary["source"] == "raw_all_z_full_register_v1"
    assert summary["repeat_grid_complete"] is True
    assert summary["sector_weight_mean"] == pytest.approx(0.75)
    assert summary["doublon_total_mean"] == pytest.approx(0.75)
    assert summary["sector_distribution"]["sorted_by"] == "n_up_then_n_dn"
    assert summary["sector_distribution"]["weights_by_sector"] == [
        {"n_up": 1, "n_dn": 0, "mean": pytest.approx(0.25), "std": pytest.approx(0.3535533905932738), "stderr": pytest.approx(0.25)},
        {"n_up": 1, "n_dn": 1, "mean": pytest.approx(0.75), "std": pytest.approx(0.3535533905932738), "stderr": pytest.approx(0.25)},
    ]
    assert summary["site_observables"]["site_index_order"] == "physical_site_0_to_n_minus_1"
    assert summary["site_observables"]["doublon_by_site_mean"] == pytest.approx([0.5, 0.25])
    assert summary["site_observables"]["charge_by_site_mean"] == pytest.approx([1.25, 0.5])
    assert summary["observable_span"]["supports_postselected_energy"] is False
    assert summary["observable_span"]["supports_projector_renorm_energy"] is False


def test_summarize_hh_exact_diagonal_reference_reports_sector_weight_and_doublon() -> None:
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(2)

    summary = nor._summarize_hh_exact_diagonal_reference(
        qc,
        num_sites=2,
        ordering="blocked",
        sector_n_up=1,
        sector_n_dn=1,
    )

    assert summary["source"] == "ideal_diagonal_v1"
    assert summary["sector_weight"] == pytest.approx(1.0)
    assert summary["doublon_total"] == pytest.approx(1.0)
    assert summary["target_sector"] == {"n_up": 1, "n_dn": 1}


def test_bootstrap_hh_full_register_z_records_reports_confidence_intervals() -> None:
    tags = {
        "symmetry_num_sites": 2,
        "symmetry_ordering": "blocked",
        "symmetry_sector_n_up": 1,
        "symmetry_sector_n_dn": 1,
    }
    records = [
        _raw_record(
            basis_label="ZZZZZZ",
            repeat_index=0,
            counts={"000101": 4, "001010": 4},
            measured_logical_qubits=(0, 1, 2, 3, 4, 5),
            group_terms=({"label": "ZZZZZZ", "coeff_re": 1.0, "coeff_im": 0.0},),
            num_qubits=6,
            observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            semantic_tags=tags,
        ),
        _raw_record(
            basis_label="ZZZZZZ",
            repeat_index=1,
            counts={"000101": 4, "000001": 4},
            measured_logical_qubits=(0, 1, 2, 3, 4, 5),
            group_terms=({"label": "ZZZZZZ", "coeff_re": 1.0, "coeff_im": 0.0},),
            num_qubits=6,
            observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            semantic_tags=tags,
        ),
    ]

    summary = nor._bootstrap_hh_full_register_z_records(
        records,
        expected_repeat_count=2,
        bootstrap_repetitions=32,
        seed=11,
    )

    assert summary["source"] == "hh_full_register_z_bootstrap_v1"
    assert summary["bootstrap_repetitions"] == 32
    assert summary["metrics"]["sector_weight"]["point_estimate"] == pytest.approx(0.75)
    assert (
        summary["metrics"]["sector_weight"]["ci_lower"]
        <= summary["metrics"]["sector_weight"]["point_estimate"]
        <= summary["metrics"]["sector_weight"]["ci_upper"]
    )
    assert len(summary["metrics"]["sector_distribution"]) == 2
    assert summary["metrics"]["doublon_by_site"][0]["site"] == 0
    assert summary["metrics"]["charge_by_site"][1]["point_estimate"] == pytest.approx(0.5)


def test_summarize_hh_full_register_z_records_rejects_partial_register_measurement() -> None:
    tags = {
        "symmetry_num_sites": 2,
        "symmetry_ordering": "blocked",
        "symmetry_sector_n_up": 1,
        "symmetry_sector_n_dn": 1,
    }
    record = _raw_record(
        basis_label="ZZIIII",
        repeat_index=0,
        counts={"000101": 8},
        measured_logical_qubits=(0, 2),
        group_terms=({"label": "ZZIIII", "coeff_re": 1.0, "coeff_im": 0.0},),
        num_qubits=6,
        observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
        semantic_tags=tags,
    )

    with pytest.raises(ValueError, match="full-register"):
        nor._summarize_hh_full_register_z_records([record], expected_repeat_count=1)


def test_summarize_hh_full_register_z_artifact_filters_gzip_records(tmp_path: Path) -> None:
    raw_path = tmp_path / "records.ndjson.gz"
    diag_tags = {
        "symmetry_num_sites": 2,
        "symmetry_ordering": "blocked",
        "symmetry_sector_n_up": 1,
        "symmetry_sector_n_dn": 1,
    }
    for record in (
        _raw_record(
            basis_label="Z",
            repeat_index=0,
            counts={"0": 8},
            measured_logical_qubits=(0,),
            group_terms=({"label": "Z", "coeff_re": 1.0, "coeff_im": 0.0},),
            evaluation_id="eval-energy",
            observable_family="adapt_phase3_oracle_gradient",
        ),
        _raw_record(
            basis_label="ZZZZZZ",
            repeat_index=0,
            counts={"000101": 8},
            measured_logical_qubits=(0, 1, 2, 3, 4, 5),
            group_terms=({"label": "ZZZZZZ", "coeff_re": 1.0, "coeff_im": 0.0},),
            num_qubits=6,
            evaluation_id="eval-diag",
            observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            semantic_tags=diag_tags,
        ),
        _raw_record(
            basis_label="ZZZZZZ",
            repeat_index=1,
            counts={"000001": 8},
            measured_logical_qubits=(0, 1, 2, 3, 4, 5),
            group_terms=({"label": "ZZZZZZ", "coeff_re": 1.0, "coeff_im": 0.0},),
            num_qubits=6,
            evaluation_id="eval-diag",
            observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            semantic_tags=diag_tags,
        ),
    ):
        nor._write_raw_measurement_record(str(raw_path), record)

    summary = nor._summarize_hh_full_register_z_artifact(
        str(raw_path),
        evaluation_id="eval-diag",
        observable_family="fixed_scaffold_runtime_all_z_symmetry_diagnostic",
        expected_repeat_count=2,
    )

    assert summary["evaluation_id"] == "eval-diag"
    assert summary["raw_artifact_path"] == str(raw_path)
    assert summary["sector_weight_mean"] == pytest.approx(0.5)


def test_raw_measurement_oracle_all_z_diagnostic_measures_full_register() -> None:
    term = AnsatzTerm(
        label="op_xy",
        polynomial=PauliPolynomial("JW", [PauliTerm(2, ps="xy", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=2,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )
    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=32,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle:
        bundle = oracle.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=nor._all_z_full_register_qop(2),
            observable_family="all_z_diag_test",
        )

    assert bundle.records[0].basis_label == "ZZ"
    assert bundle.records[0].measured_logical_qubits == (0, 1)
    assert bundle.estimate.group_count == 1
    assert "ZZ" in bundle.compile_signatures_by_basis


def test_raw_measurement_oracle_ndjson_writer_appends_records(tmp_path: Path) -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=1,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )
    observable = SparsePauliOp.from_list([("Z", 1.0)])
    raw_path = tmp_path / "records.ndjson.gz"
    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=32,
            seed=7,
            oracle_repeats=2,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
            raw_artifact_path=str(raw_path),
        )
    ) as oracle:
        bundle = oracle.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=observable,
            observable_family="unit_test",
        )

    with gzip.open(raw_path, "rt", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]

    assert len(rows) == len(bundle.records)
    assert rows[0]["schema_version"] == "raw_measurement_record_v1"
    assert rows[0]["basis_label"] == "Z"


def test_raw_measurement_oracle_default_evaluation_ids_are_unique_across_instances() -> None:
    term = AnsatzTerm(
        label="op_x",
        polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)]),
    )
    layout = build_parameter_layout([term], ignore_identity=True, coefficient_tolerance=1e-12, sort_terms=True)
    plan = build_parameterized_ansatz_plan(
        layout,
        nq=1,
        ref_state=np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
    )
    observable = SparsePauliOp.from_list([("Z", 1.0)])

    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=16,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle_a:
        bundle_a = oracle_a.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=observable,
            observable_family="unit_test",
        )
    with RawMeasurementOracle(
        OracleConfig(
            noise_mode="backend_scheduled",
            shots=16,
            seed=7,
            oracle_repeats=1,
            backend_name="FakeGuadalupeV2",
            use_fake_backend=True,
        )
    ) as oracle_b:
        bundle_b = oracle_b.measure_observable(
            plan=plan,
            theta_runtime=np.asarray([0.0], dtype=float),
            observable=observable,
            observable_family="unit_test",
        )

    assert bundle_a.evaluation_id != bundle_b.evaluation_id
    assert bundle_a.records[0].record_id != bundle_b.records[0].record_id


def test_resolve_raw_transport_runtime_auto_prefers_sampler_v2() -> None:
    IBMBackendStub = type("IBMBackend", (), {"run": lambda self, *args, **kwargs: None})
    IBMBackendStub.__module__ = "qiskit_ibm_runtime.ibm_backend"
    backend = IBMBackendStub()
    cfg = OracleConfig(noise_mode="runtime", shots=128, raw_transport="auto")

    assert nor._resolve_raw_transport(cfg, backend) == "sampler_v2"


def test_resolve_raw_transport_runtime_rejects_backend_run_for_ibm_backend() -> None:
    IBMBackendStub = type("IBMBackend", (), {"run": lambda self, *args, **kwargs: None})
    IBMBackendStub.__module__ = "qiskit_ibm_runtime.ibm_backend"
    backend = IBMBackendStub()
    cfg = OracleConfig(noise_mode="runtime", shots=128, raw_transport="backend_run")

    with pytest.raises(ValueError, match="backend_run"):
        nor._resolve_raw_transport(cfg, backend)


def test_resolve_raw_transport_runtime_allows_backend_run_for_direct_backend() -> None:
    DirectBackendStub = type("DirectBackend", (), {"run": lambda self, *args, **kwargs: None})
    DirectBackendStub.__module__ = "custom.runtime_backend"
    backend = DirectBackendStub()
    cfg = OracleConfig(noise_mode="runtime", shots=128, raw_transport="backend_run")

    assert nor._resolve_raw_transport(cfg, backend) == "backend_run"


def test_normalize_sampler_raw_runtime_config_accepts_runtime_sampler_contract() -> None:
    cfg = normalize_sampler_raw_runtime_config(
        OracleConfig(
            noise_mode="runtime",
            shots=128,
            backend_name="ibm_marrakesh",
            execution_surface="raw_measurement_v1",
            mitigation={"mode": "none"},
            symmetry_mitigation={"mode": "off"},
            runtime_profile={"name": "legacy_runtime_v0"},
            raw_transport="auto",
        )
    )

    assert cfg.noise_mode == "runtime"
    assert cfg.execution_surface == "raw_measurement_v1"
    assert cfg.raw_transport == "auto"


def test_normalize_sampler_raw_runtime_config_rejects_backend_run_transport() -> None:
    with pytest.raises(ValueError, match="raw_transport"):
        normalize_sampler_raw_runtime_config(
            OracleConfig(
                noise_mode="runtime",
                shots=128,
                backend_name="ibm_marrakesh",
                execution_surface="raw_measurement_v1",
                mitigation={"mode": "none"},
                symmetry_mitigation={"mode": "off"},
                runtime_profile={"name": "legacy_runtime_v0"},
                raw_transport="backend_run",
            )
        )


def test_normalize_sampler_raw_runtime_config_rejects_verify_only_symmetry() -> None:
    with pytest.raises(ValueError, match="symmetry_mitigation='off'"):
        normalize_sampler_raw_runtime_config(
            OracleConfig(
                noise_mode="runtime",
                shots=128,
                backend_name="ibm_marrakesh",
                execution_surface="raw_measurement_v1",
                mitigation={"mode": "none"},
                symmetry_mitigation={"mode": "verify_only"},
                runtime_profile={"name": "legacy_runtime_v0"},
                raw_transport="sampler_v2",
            )
        )


def test_expectation_oracle_marks_legacy_execution_surface() -> None:
    qc = QuantumCircuit(1)
    obs = SparsePauliOp.from_list([("I", 1.0)])

    with ExpectationOracle(OracleConfig(noise_mode="ideal")) as oracle:
        _ = oracle.evaluate(qc, obs)

    assert oracle.backend_info.details.get("execution_surface") == "expectation_v1"
