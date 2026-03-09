from __future__ import annotations

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
        assert dict(oracle.backend_info.details.get("mitigation", {})) == mit


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
