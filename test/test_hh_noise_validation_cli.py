from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_noise_hardware_validation as _noise_mod
from pipelines.exact_bench.hh_noise_hardware_validation import (
    _build_mitigation_config_from_args,
    _build_symmetry_mitigation_config_from_args,
    _apply_defaults_and_minimums,
    parse_args,
)
from src.quantum.pauli_polynomial_class import PauliPolynomial
from src.quantum.qubitization_module import PauliTerm
from src.quantum.vqe_latex_python_pairs import AnsatzTerm


def _toy_pool() -> list[AnsatzTerm]:
    return [
        AnsatzTerm(label="cand_a", polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="x", pc=1.0)])),
        AnsatzTerm(label="cand_b", polynomial=PauliPolynomial("JW", [PauliTerm(1, ps="z", pc=1.0)])),
    ]


def _toy_args(**overrides) -> SimpleNamespace:
    base = dict(
        adapt_inner_optimizer="SPSA",
        adapt_spsa_a=0.2,
        adapt_spsa_c=0.1,
        adapt_spsa_alpha=0.602,
        adapt_spsa_gamma=0.101,
        adapt_spsa_A=10.0,
        adapt_spsa_avg_last=0,
        adapt_spsa_eval_repeats=1,
        adapt_spsa_eval_agg="mean",
        adapt_seed=7,
        adapt_max_depth=1,
        adapt_gradient_step=0.5,
        adapt_eps_grad=1e-9,
        adapt_eps_energy=-1.0,
        adapt_min_confidence=0.0,
        adapt_allow_repeats=False,
        adapt_maxiter=1,
        adapt_pool="toy_pool",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _oracle_estimate(*, mean: float, std: float, stderr: float, n_samples: int = 7, aggregate: str = "mean") -> SimpleNamespace:
    return SimpleNamespace(
        mean=float(mean),
        std=float(std),
        stdev=float(std),
        stderr=float(stderr),
        n_samples=int(n_samples),
        raw_values=[],
        aggregate=str(aggregate),
    )


def _fake_adapt_circuit(ops, theta, *, num_qubits, reference_state):
    return {
        "labels": [str(op.label) for op in ops],
        "theta": [float(x) for x in np.asarray(theta, dtype=float).tolist()],
        "num_qubits": int(num_qubits),
    }


def _fake_spsa_minimize(fun, x0, **kwargs):
    x = np.zeros_like(np.asarray(x0, dtype=float))
    value = float(fun(x))
    return SimpleNamespace(x=x, fun=float(value), nfev=1, nit=1, success=True, message="stub")


def test_hh_defaults_applied_from_minimum_table_l2_nph1() -> None:
    args = parse_args(["--L", "2"])
    args = _apply_defaults_and_minimums(args)

    assert int(args.vqe_reps) == 2
    assert int(args.vqe_restarts) == 3
    assert int(args.vqe_maxiter) == 800
    assert int(args.trotter_steps) == 64


def test_hh_under_minimum_rejected_without_smoke_flag() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "8",
        ]
    )
    with pytest.raises(ValueError):
        _apply_defaults_and_minimums(args)


def test_hh_under_minimum_allowed_with_smoke_flag() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "8",
            "--smoke-test-intentionally-weak",
        ]
    )
    args = _apply_defaults_and_minimums(args)

    assert int(args.vqe_reps) == 1
    assert int(args.vqe_restarts) == 1
    assert int(args.vqe_maxiter) == 100
    assert int(args.trotter_steps) == 8


def test_hubbard_under_minimum_rejected_without_smoke_flag() -> None:
    args = parse_args(
        [
            "--problem",
            "hubbard",
            "--ansatz",
            "hva",
            "--L",
            "4",
            "--vqe-reps",
            "1",
            "--vqe-restarts",
            "1",
            "--vqe-maxiter",
            "100",
            "--trotter-steps",
            "16",
        ]
    )
    with pytest.raises(ValueError):
        _apply_defaults_and_minimums(args)


def test_cli_parses_fallback_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--no-allow-aer-fallback",
            "--no-omp-shm-workaround",
        ]
    )
    assert bool(args.allow_aer_fallback) is False
    assert bool(args.omp_shm_workaround) is False


def test_cli_parses_legacy_parity_flags() -> None:
    args = parse_args(
        [
            "--L",
            "2",
            "--legacy-reference-json",
            "artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json",
            "--legacy-parity-tol",
            "1e-10",
            "--output-compare-plot",
            "artifacts/pdf/hh_noise_cmp.png",
            "--compare-observables",
            "energy_static_trotter,doublon_trotter",
        ]
    )
    assert str(args.legacy_reference_json).endswith(
        "artifacts/json/hc_hh_L2_static_t1.0_U2.0_g1.0_nph1.json"
    )
    assert float(args.legacy_parity_tol) == pytest.approx(1e-10)
    assert str(args.output_compare_plot).endswith("artifacts/pdf/hh_noise_cmp.png")
    assert str(args.compare_observables) == "energy_static_trotter,doublon_trotter"


def test_cli_parses_mitigation_flags_defaults_and_values() -> None:
    args = parse_args(["--L", "2"])
    assert str(args.mitigation) == "none"
    assert args.zne_scales is None
    assert args.dd_sequence is None
    assert _build_mitigation_config_from_args(args) == {
        "mode": "none",
        "zne_scales": [],
        "dd_sequence": None,
        "local_readout_strategy": None,
    }

    args = parse_args(
        [
            "--L",
            "2",
            "--mitigation",
            "zne",
            "--zne-scales",
            "1.0,2.0,3.0",
            "--dd-sequence",
            "XY4",
        ]
    )
    assert str(args.mitigation) == "zne"
    assert str(args.zne_scales) == "1.0,2.0,3.0"
    assert str(args.dd_sequence) == "XY4"


def test_cli_parses_symmetry_mitigation_flags_defaults_and_values() -> None:
    args = parse_args(["--L", "2"])
    assert str(args.symmetry_mitigation_mode) == "off"
    assert _build_symmetry_mitigation_config_from_args(args) == {
        "mode": "off",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }

    args = parse_args(
        [
            "--L",
            "2",
            "--symmetry-mitigation-mode",
            "postselect_diag_v1",
        ]
    )
    assert str(args.symmetry_mitigation_mode) == "postselect_diag_v1"
    assert _build_symmetry_mitigation_config_from_args(args) == {
        "mode": "postselect_diag_v1",
        "num_sites": 2,
        "ordering": "blocked",
        "sector_n_up": 1,
        "sector_n_dn": 1,
    }


def test_run_noisy_adapt_emits_sigma_hat_from_oracle_fd_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    grad_step = 0.5
    pool = _toy_pool()

    def _noisy_eval(circuit, _observable):
        labels = tuple(circuit["labels"])
        theta = tuple(circuit["theta"])
        if labels == ("cand_a",) and theta == (grad_step,):
            return _oracle_estimate(mean=1.0, std=0.8, stderr=0.3, n_samples=11)
        if labels == ("cand_a",) and theta == (-grad_step,):
            return _oracle_estimate(mean=0.0, std=0.6, stderr=0.4, n_samples=13)
        if labels == ("cand_b",) and theta == (grad_step,):
            return _oracle_estimate(mean=0.7, std=0.2, stderr=0.05, n_samples=9)
        if labels == ("cand_b",) and theta == (-grad_step,):
            return _oracle_estimate(mean=0.1, std=0.2, stderr=0.05, n_samples=9)
        return _oracle_estimate(mean=0.25, std=0.1, stderr=0.02, n_samples=5)

    noisy_oracle = SimpleNamespace(evaluate=_noisy_eval)
    ideal_oracle = SimpleNamespace(evaluate=lambda circuit, obs: _oracle_estimate(mean=0.2, std=0.0, stderr=0.0, n_samples=1))

    monkeypatch.setattr(_noise_mod, "_adapt_ops_to_circuit", _fake_adapt_circuit)
    monkeypatch.setattr(_noise_mod, "spsa_minimize", _fake_spsa_minimize)

    payload, _ops, _theta = _noise_mod._run_noisy_adapt(
        args=_toy_args(adapt_gradient_step=grad_step),
        pool=pool,
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        h_qop=SparsePauliOp.from_list([("I", 1.0)]),
        noisy_oracle=noisy_oracle,
        ideal_oracle=ideal_oracle,
    )

    assert payload["gradient_uncertainty_source"] == "oracle_fd_stderr_v1"
    assert payload["phase1_score_z_alpha_used"] == pytest.approx(0.0)
    assert payload["selection_metric_name"] == "g_abs"
    assert payload["gradient_confidence_mode_used"] == "std"
    assert payload["history"][0]["selected_label"] == "cand_a"
    assert payload["history"][0]["sigma_hat"] == pytest.approx(0.5)
    assert payload["history"][0]["g_lcb"] == pytest.approx(payload["history"][0]["max_gradient_abs"])
    assert payload["history"][0]["gradient_confidence"] == pytest.approx(payload["history"][0]["gradient_confidence_std"])
    assert payload["history"][0]["gradient_confidence_stderr"] == pytest.approx(2.0)
    selected_scout = next(row for row in payload["history"][0]["candidate_gradient_scout"] if row["selected_for_optimization"])
    assert selected_scout["candidate_label"] == "cand_a"
    assert selected_scout["sigma_hat"] == pytest.approx(0.5)
    assert selected_scout["oracle_samples_plus"] == 11
    assert selected_scout["oracle_samples_minus"] == 13


def test_run_noisy_adapt_keeps_raw_gradient_selection_when_sigma_is_large(monkeypatch: pytest.MonkeyPatch) -> None:
    grad_step = 0.5
    pool = _toy_pool()

    def _noisy_eval(circuit, _observable):
        labels = tuple(circuit["labels"])
        theta = tuple(circuit["theta"])
        if labels == ("cand_a",) and theta == (grad_step,):
            return _oracle_estimate(mean=1.0, std=0.1, stderr=2.0, n_samples=15)
        if labels == ("cand_a",) and theta == (-grad_step,):
            return _oracle_estimate(mean=0.0, std=0.1, stderr=3.0, n_samples=15)
        if labels == ("cand_b",) and theta == (grad_step,):
            return _oracle_estimate(mean=0.9, std=0.1, stderr=0.01, n_samples=15)
        if labels == ("cand_b",) and theta == (-grad_step,):
            return _oracle_estimate(mean=0.1, std=0.1, stderr=0.01, n_samples=15)
        return _oracle_estimate(mean=0.3, std=0.1, stderr=0.02, n_samples=5)

    noisy_oracle = SimpleNamespace(evaluate=_noisy_eval)
    ideal_oracle = SimpleNamespace(evaluate=lambda circuit, obs: _oracle_estimate(mean=0.2, std=0.0, stderr=0.0, n_samples=1))

    monkeypatch.setattr(_noise_mod, "_adapt_ops_to_circuit", _fake_adapt_circuit)
    monkeypatch.setattr(_noise_mod, "spsa_minimize", _fake_spsa_minimize)

    payload, _ops, _theta = _noise_mod._run_noisy_adapt(
        args=_toy_args(adapt_gradient_step=grad_step),
        pool=pool,
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        h_qop=SparsePauliOp.from_list([("I", 1.0)]),
        noisy_oracle=noisy_oracle,
        ideal_oracle=ideal_oracle,
    )

    scout_rows = {row["candidate_label"]: row for row in payload["history"][0]["candidate_gradient_scout"]}
    assert payload["history"][0]["selected_label"] == "cand_a"
    assert scout_rows["cand_a"]["sigma_hat"] > scout_rows["cand_b"]["sigma_hat"]
    assert scout_rows["cand_a"]["selection_metric_value"] > scout_rows["cand_b"]["selection_metric_value"]
    assert scout_rows["cand_a"]["g_lcb"] == pytest.approx(scout_rows["cand_a"]["gradient_abs"])
    assert scout_rows["cand_b"]["g_lcb"] == pytest.approx(scout_rows["cand_b"]["gradient_abs"])


def test_run_noisy_adapt_retains_last_sigma_telemetry_on_low_confidence_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    grad_step = 0.5
    pool = _toy_pool()

    def _noisy_eval(circuit, _observable):
        labels = tuple(circuit["labels"])
        theta = tuple(circuit["theta"])
        if labels == ("cand_a",) and theta == (grad_step,):
            return _oracle_estimate(mean=1.0, std=0.8, stderr=0.3, n_samples=11)
        if labels == ("cand_a",) and theta == (-grad_step,):
            return _oracle_estimate(mean=0.0, std=0.6, stderr=0.4, n_samples=13)
        if labels == ("cand_b",) and theta == (grad_step,):
            return _oracle_estimate(mean=0.7, std=0.2, stderr=0.05, n_samples=9)
        if labels == ("cand_b",) and theta == (-grad_step,):
            return _oracle_estimate(mean=0.1, std=0.2, stderr=0.05, n_samples=9)
        return _oracle_estimate(mean=0.25, std=0.1, stderr=0.02, n_samples=5)

    noisy_oracle = SimpleNamespace(evaluate=_noisy_eval)
    ideal_oracle = SimpleNamespace(evaluate=lambda circuit, obs: _oracle_estimate(mean=0.2, std=0.0, stderr=0.0, n_samples=1))

    monkeypatch.setattr(_noise_mod, "_adapt_ops_to_circuit", _fake_adapt_circuit)
    monkeypatch.setattr(_noise_mod, "spsa_minimize", _fake_spsa_minimize)

    payload, _ops, _theta = _noise_mod._run_noisy_adapt(
        args=_toy_args(adapt_gradient_step=grad_step, adapt_min_confidence=100.0),
        pool=pool,
        psi_ref=np.array([1.0, 0.0], dtype=complex),
        h_qop=SparsePauliOp.from_list([("I", 1.0)]),
        noisy_oracle=noisy_oracle,
        ideal_oracle=ideal_oracle,
    )

    assert payload["stop_reason"] == "low_gradient_confidence"
    assert payload["history"] == []
    assert payload["last_selected_label"] == "cand_a"
    assert payload["last_sigma_hat"] == pytest.approx(0.5)
    assert payload["last_g_lcb"] == pytest.approx(1.0)
    assert payload["last_gradient_confidence"] == pytest.approx(1.0)
    assert payload["last_gradient_confidence_stderr"] == pytest.approx(2.0)
    assert any(row["selected_for_optimization"] for row in payload["last_candidate_gradient_scout"])
