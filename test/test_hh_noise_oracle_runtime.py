from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector

import pipelines.exact_bench.noise_oracle_runtime as nor
from pipelines.exact_bench.noise_oracle_runtime import (
    ExpectationOracle,
    OracleConfig,
    _ansatz_to_circuit,
    _pauli_poly_to_sparse_pauli_op,
    compile_circuit_for_local_backend,
    normalize_ideal_reference_symmetry_mitigation,
    normalize_symmetry_mitigation_config,
)
from src.quantum.hartree_fock_reference_state import hubbard_holstein_reference_state
from src.quantum.qubitization_module import PauliTerm
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.vqe_latex_python_pairs import HubbardHolsteinTermwiseAnsatz


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
