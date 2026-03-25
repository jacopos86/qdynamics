from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_noise_robustness_seq_report as report


class _LayoutStub:
    logical_parameter_count = 2
    runtime_parameter_count = 3


def test_fixed_lean_noisy_replay_unavailable_without_ansatz_input_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "ansatz_input_state_meta": {
                "available": False,
                "reason": "missing_ansatz_input_state_provenance",
                "error": None,
                "source": None,
                "handoff_state_kind": None,
            },
        },
    )

    payload = report._run_imported_fixed_lean_noisy_replay(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        optimizer_method="SPSA",
        optimizer_seed=19,
        optimizer_maxiter=3,
        optimizer_wallclock_cap_s=30,
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "missing_ansatz_input_state_provenance"


def test_fixed_lean_noisy_replay_rejects_non_lean_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "payload": {"adapt_vqe": {"pool_type": "full_meta"}},
            "settings": {},
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "hf",
                "handoff_state_kind": "reference_state",
            },
        },
    )

    payload = report._run_imported_fixed_lean_noisy_replay(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        optimizer_method="SPSA",
        optimizer_seed=19,
        optimizer_maxiter=3,
        optimizer_wallclock_cap_s=30,
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is False
    assert payload["reason"] == "fixed_lean_replay_requires_pareto_lean_l2_source"


def test_fixed_lean_noisy_replay_reports_locked_structure_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "payload": {"adapt_vqe": {"pool_type": "pareto_lean_l2"}},
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "theta_runtime": np.asarray([0.1, -0.2, 0.3], dtype=float),
            "ansatz_input_state": np.asarray([1.0, 0.0], dtype=complex),
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "hf",
                "handoff_state_kind": "reference_state",
            },
            "num_qubits": 1,
            "saved_energy": 0.25,
        },
    )
    monkeypatch.setattr(
        report,
        "_build_ansatz_circuit",
        lambda layout, theta_runtime, nq, ref_state=None: {"theta": [float(x) for x in np.asarray(theta_runtime).tolist()], "nq": nq},
    )

    def _fake_spsa_minimize(**kwargs):
        return type(
            "SPSAResultStub",
            (),
            {
                "x": np.asarray([0.05, -0.1, 0.2], dtype=float),
                "fun": 0.11,
                "nfev": 4,
                "nit": 1,
                "success": True,
                "message": "done",
            },
        )()

    monkeypatch.setattr(report, "spsa_minimize", _fake_spsa_minimize)
    monkeypatch.setattr(report, "project_runtime_theta_block_mean", lambda theta, layout: np.asarray([0.0, 0.0], dtype=float))

    eval_counter = {"count": 0}

    def _fake_eval_locked(**kwargs):
        eval_counter["count"] += 1
        noisy_mean = 0.4 if eval_counter["count"] == 1 else 0.3
        ideal_mean = 0.2 if eval_counter["count"] == 1 else 0.1
        return {
            "noisy_mean": noisy_mean,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
            "ideal_mean": ideal_mean,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": noisy_mean - ideal_mean,
            "delta_stderr": 0.01,
            "backend_info": {"noise_mode": "backend_scheduled"},
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy", _fake_eval_locked)

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def evaluate(self, circuit, observable):
            theta = circuit["theta"]
            return type(
                "EstimateStub",
                (),
                {
                    "mean": float(np.sum(np.asarray(theta, dtype=float) ** 2)),
                    "std": 0.0,
                    "stdev": 0.0,
                    "stderr": 0.02,
                    "n_samples": 1,
                    "raw_values": [float(np.sum(np.asarray(theta, dtype=float) ** 2))],
                    "aggregate": "mean",
                },
            )()

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)
    monkeypatch.setattr(report, "build_time_dependent_sparse_qop", lambda **kwargs: object())

    payload = report._run_imported_fixed_lean_noisy_replay(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        optimizer_method="SPSA",
        optimizer_seed=19,
        optimizer_maxiter=3,
        optimizer_wallclock_cap_s=30,
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["structure_locked"] is True
    assert payload["matched_family_replay"] is False
    assert payload["reps"] == 1
    assert payload["optimizer"]["method"] == "SPSA"
    assert payload["parameterization"]["runtime_parameter_count"] == 3
    assert payload["energies"]["best_noisy_minus_ideal"] == pytest.approx(0.2)


def test_fixed_lean_noisy_replay_accepts_powell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "payload": {"adapt_vqe": {"pool_type": "pareto_lean_l2"}},
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "theta_runtime": np.asarray([0.1, -0.2, 0.3], dtype=float),
            "ansatz_input_state": np.asarray([1.0, 0.0], dtype=complex),
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "hf",
                "handoff_state_kind": "reference_state",
            },
            "num_qubits": 1,
            "saved_energy": 0.25,
        },
    )
    monkeypatch.setattr(
        report,
        "_build_ansatz_circuit",
        lambda layout, theta_runtime, nq, ref_state=None: {"theta": [float(x) for x in np.asarray(theta_runtime).tolist()], "nq": nq},
    )
    monkeypatch.setattr(report, "project_runtime_theta_block_mean", lambda theta, layout: np.asarray([0.0, 0.0], dtype=float))
    monkeypatch.setattr(report, "build_time_dependent_sparse_qop", lambda **kwargs: object())

    def _fake_eval_locked(**kwargs):
        return {
            "noisy_mean": 0.25,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
            "ideal_mean": 0.15,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.10,
            "delta_stderr": 0.01,
            "backend_info": {"noise_mode": "backend_scheduled"},
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy", _fake_eval_locked)

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeGuadalupeV2",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def evaluate(self, circuit, observable):
            theta = np.asarray(circuit["theta"], dtype=float)
            return type(
                "EstimateStub",
                (),
                {
                    "mean": float(np.sum(theta**2)),
                    "std": 0.0,
                    "stdev": 0.0,
                    "stderr": 0.02,
                    "n_samples": 1,
                    "raw_values": [float(np.sum(theta**2))],
                    "aggregate": "mean",
                },
            )()

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)

    import scipy.optimize

    def _fake_minimize(fun, x0, method=None, options=None):
        assert method == "Powell"
        val = fun(np.asarray(x0, dtype=float))
        return type(
            "PowellResultStub",
            (),
            {
                "x": np.asarray([0.0, 0.0, 0.0], dtype=float),
                "fun": float(val),
                "nfev": 5,
                "nit": 2,
                "success": True,
                "message": "powell done",
            },
        )()

    monkeypatch.setattr(scipy.optimize, "minimize", _fake_minimize)

    payload = report._run_imported_fixed_lean_noisy_replay(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        optimizer_method="Powell",
        optimizer_seed=19,
        optimizer_maxiter=3,
        optimizer_wallclock_cap_s=30,
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is True
    assert payload["optimizer"]["method"] == "Powell"
    assert payload["optimizer"]["iterations_completed"] == 2
    assert payload["optimizer"]["objective_calls_total"] == 5


def test_fixed_scaffold_noisy_replay_accepts_locked_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "payload": {
                "adapt_vqe": {
                    "pool_type": "fixed_scaffold_locked",
                    "structure_locked": True,
                    "fixed_scaffold_kind": "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1",
                    "fixed_scaffold_metadata": {
                        "route_family": "locked_imported_scaffold_v1",
                        "term_order_id": "source_order_pruned",
                        "operator_count": 5,
                        "runtime_term_count": 6,
                    },
                }
            },
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "theta_runtime": np.asarray([0.1, -0.2, 0.3], dtype=float),
            "ansatz_input_state": np.asarray([1.0, 0.0], dtype=complex),
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "hf",
                "handoff_state_kind": "reference_state",
            },
            "num_qubits": 1,
            "saved_energy": 0.25,
        },
    )
    monkeypatch.setattr(
        report,
        "_build_ansatz_circuit",
        lambda layout, theta_runtime, nq, ref_state=None: {"theta": [float(x) for x in np.asarray(theta_runtime).tolist()], "nq": nq},
    )

    def _fake_spsa_minimize(**kwargs):
        return type(
            "SPSAResultStub",
            (),
            {
                "x": np.asarray([0.05, -0.1, 0.2], dtype=float),
                "fun": 0.11,
                "nfev": 4,
                "nit": 1,
                "success": True,
                "message": "done",
            },
        )()

    monkeypatch.setattr(report, "spsa_minimize", _fake_spsa_minimize)
    monkeypatch.setattr(report, "project_runtime_theta_block_mean", lambda theta, layout: np.asarray([0.0, 0.0], dtype=float))
    logged_events: list[str] = []
    monkeypatch.setattr(report, "_ai_log", lambda event, **fields: logged_events.append(str(event)))

    eval_counter = {"count": 0}

    def _fake_eval_locked(**kwargs):
        eval_counter["count"] += 1
        noisy_mean = 0.4 if eval_counter["count"] == 1 else 0.3
        ideal_mean = 0.2 if eval_counter["count"] == 1 else 0.1
        return {
            "noisy_mean": noisy_mean,
            "noisy_std": 0.0,
            "noisy_stdev": 0.0,
            "noisy_stderr": 0.01,
            "ideal_mean": ideal_mean,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": noisy_mean - ideal_mean,
            "delta_stderr": 0.01,
            "backend_info": {"noise_mode": "backend_scheduled"},
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy", _fake_eval_locked)

    class _OracleStub:
        def __init__(self, config):
            self.backend_info = type(
                "NoiseBackendInfoStub",
                (),
                {
                    "noise_mode": "backend_scheduled",
                    "estimator_kind": "fake_backend.run(counts)",
                    "backend_name": "FakeMarrakesh",
                    "using_fake_backend": True,
                    "details": {},
                },
            )()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def evaluate(self, circuit, observable):
            theta = circuit["theta"]
            return type(
                "EstimateStub",
                (),
                {
                    "mean": float(np.sum(np.asarray(theta, dtype=float) ** 2)),
                    "std": 0.0,
                    "stdev": 0.0,
                    "stderr": 0.02,
                    "n_samples": 1,
                    "raw_values": [float(np.sum(np.asarray(theta, dtype=float) ** 2))],
                    "aggregate": "mean",
                },
            )()

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)
    monkeypatch.setattr(report, "build_time_dependent_sparse_qop", lambda **kwargs: object())

    payload = report._run_imported_fixed_scaffold_noisy_replay(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        optimizer_method="SPSA",
        optimizer_seed=19,
        optimizer_maxiter=3,
        optimizer_wallclock_cap_s=30,
        progress_every_s=1.0,
        spsa_a=0.2,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=10.0,
        spsa_avg_last=0,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        local_dd_probe_sequence="XPXM",
        mitigation_config={"mode": "readout", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is True
    assert payload["route"] == "fixed_scaffold_noisy_replay"
    assert payload["structure_locked"] is True
    assert payload["subject_kind"] == "hh_marrakesh_gate_pruned_6term_drop_eyezee_v1"
    assert payload["term_order_id"] == "source_order_pruned"
    assert payload["theta_source"] == "imported_theta_runtime"
    assert payload["execution_mode"] == "backend_scheduled"
    assert payload["local_mitigation_label"] == "readout_only"
    assert "saved_theta_local_mitigation_ablation" in payload
    assert payload["saved_theta_local_mitigation_ablation"]["results"]["readout_plus_local_dd"]["success"] is True
    assert "fixed_scaffold_replay_started" in logged_events
    assert "fixed_scaffold_replay_completed" in logged_events


def test_fixed_scaffold_noise_attribution_reports_component_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit_sentinel = object()
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "payload": {
                "adapt_vqe": {
                    "pool_type": "fixed_scaffold_locked",
                    "structure_locked": True,
                    "fixed_scaffold_kind": "hh_nighthawk_gate_pruned_7term_v1",
                    "fixed_scaffold_metadata": {
                        "route_family": "locked_imported_scaffold_v1",
                        "term_order_id": "source_order",
                        "operator_count": 5,
                        "runtime_term_count": 7,
                    },
                }
            },
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "theta_runtime": np.asarray([0.1, -0.2, 0.3], dtype=float),
            "ansatz_input_state": np.asarray([1.0, 0.0], dtype=complex),
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "hf",
                "handoff_state_kind": "reference_state",
            },
            "num_qubits": 1,
            "saved_energy": 0.25,
            "circuit": circuit_sentinel,
        },
    )
    monkeypatch.setattr(report, "_build_ansatz_circuit", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rebuild circuit")))
    monkeypatch.setattr(report, "build_time_dependent_sparse_qop", lambda **kwargs: object())

    class _OracleStub:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def evaluate(self, circuit, observable):
            assert circuit is circuit_sentinel
            return type(
                "EstimateStub",
                (),
                {
                    "mean": 0.2,
                    "std": 0.0,
                    "stdev": 0.0,
                    "stderr": 0.0,
                    "n_samples": 1,
                    "raw_values": [0.2],
                    "aggregate": "mean",
                },
            )()

        def evaluate_backend_scheduled_attribution(self, circuit, observable, *, slices):
            assert circuit is circuit_sentinel
            assert tuple(slices) == (
                "readout_only",
                "gate_stateprep_only",
                "full",
            )
            def _rec(mean: float, slice_name: str, gate: bool, readout: bool) -> dict[str, object]:
                return {
                    "success": True,
                    "slice": slice_name,
                    "components": {"gate_stateprep": gate, "readout": readout},
                    "estimate": type(
                        "EstimateStub",
                        (),
                        {
                            "mean": mean,
                            "std": 0.0,
                            "stdev": 0.0,
                            "stderr": 0.01,
                            "n_samples": 1,
                            "raw_values": [mean],
                            "aggregate": "mean",
                        },
                    )(),
                    "backend_info": {
                        "noise_mode": "backend_scheduled",
                        "estimator_kind": "fake_backend.run(counts)",
                        "backend_name": "FakeNighthawk",
                        "using_fake_backend": True,
                        "details": {"shared_compile_reused": True, "attribution_slice": slice_name},
                    },
                    "reason": None,
                    "error": None,
                }
            return {
                "shared_compile": {
                    "shared_transpile": True,
                    "layout_physical_qubits": [0],
                    "requested_slices": ["readout_only", "gate_stateprep_only", "full"],
                },
                "slices": {
                    "readout_only": _rec(0.35, "readout_only", False, True),
                    "gate_stateprep_only": _rec(1.40, "gate_stateprep_only", True, False),
                    "full": _rec(1.50, "full", True, True),
                },
            }

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)

    payload = report._run_imported_fixed_scaffold_noise_attribution(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeNighthawk",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation_config={"mode": "off"},
        slices=("readout_only", "gate_stateprep_only", "full"),
    )

    assert payload["success"] is True
    assert payload["route"] == "fixed_scaffold_noise_attribution"
    assert payload["subject_kind"] == "hh_nighthawk_gate_pruned_7term_v1"
    assert payload["term_order_id"] == "source_order"
    assert payload["shared_compile"]["shared_transpile"] is True


def test_fixed_lean_noise_attribution_reports_component_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    circuit_sentinel = object()
    monkeypatch.setattr(
        report,
        "_load_imported_artifact_context",
        lambda artifact_json: {
            "path": Path(artifact_json),
            "source_kind": "direct_adapt_artifact",
            "payload": {"adapt_vqe": {"pool_type": "pareto_lean_l2"}},
            "settings": {"L": 2, "ordering": "blocked"},
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "theta_runtime": np.asarray([0.1, -0.2, 0.3], dtype=float),
            "ansatz_input_state": np.asarray([1.0, 0.0], dtype=complex),
            "ansatz_input_state_meta": {
                "available": True,
                "reason": None,
                "error": None,
                "source": "hf",
                "handoff_state_kind": "reference_state",
            },
            "num_qubits": 1,
            "saved_energy": 0.25,
            "circuit": circuit_sentinel,
        },
    )
    monkeypatch.setattr(report, "_build_ansatz_circuit", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not rebuild circuit")))
    monkeypatch.setattr(report, "build_time_dependent_sparse_qop", lambda **kwargs: object())

    class _OracleStub:
        def __init__(self, config):
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def evaluate(self, circuit, observable):
            assert circuit is circuit_sentinel
            return type(
                "EstimateStub",
                (),
                {
                    "mean": 0.2,
                    "std": 0.0,
                    "stdev": 0.0,
                    "stderr": 0.0,
                    "n_samples": 1,
                    "raw_values": [0.2],
                    "aggregate": "mean",
                },
            )()

        def evaluate_backend_scheduled_attribution(self, circuit, observable, *, slices):
            assert circuit is circuit_sentinel
            assert tuple(slices) == (
                "readout_only",
                "gate_stateprep_only",
                "full",
            )
            def _rec(mean: float, slice_name: str, gate: bool, readout: bool) -> dict[str, object]:
                return {
                    "success": True,
                    "slice": slice_name,
                    "components": {"gate_stateprep": gate, "readout": readout},
                    "estimate": type(
                        "EstimateStub",
                        (),
                        {
                            "mean": mean,
                            "std": 0.0,
                            "stdev": 0.0,
                            "stderr": 0.01,
                            "n_samples": 1,
                            "raw_values": [mean],
                            "aggregate": "mean",
                        },
                    )(),
                    "backend_info": {
                        "noise_mode": "backend_scheduled",
                        "estimator_kind": "fake_backend.run(counts)",
                        "backend_name": "FakeGuadalupeV2",
                        "using_fake_backend": True,
                        "details": {"shared_compile_reused": True, "attribution_slice": slice_name},
                    },
                    "reason": None,
                    "error": None,
                }
            return {
                "shared_compile": {
                    "shared_transpile": True,
                    "layout_physical_qubits": [0],
                    "requested_slices": ["readout_only", "gate_stateprep_only", "full"],
                },
                "slices": {
                    "readout_only": _rec(0.35, "readout_only", False, True),
                    "gate_stateprep_only": _rec(1.40, "gate_stateprep_only", True, False),
                    "full": _rec(1.50, "full", True, True),
                },
            }

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)

    payload = report._run_imported_fixed_lean_noise_attribution(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeGuadalupeV2",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none", "zne_scales": [], "dd_sequence": None, "local_readout_strategy": None},
        symmetry_mitigation_config={"mode": "off"},
        slices=("readout_only", "gate_stateprep_only", "full"),
    )

    assert payload["success"] is True
    assert payload["route"] == "fixed_lean_noise_attribution"
    assert payload["structure_locked"] is True
    assert payload["parameter_optimization"] is False
    assert set(payload["slices"].keys()) == {"readout_only", "gate_stateprep_only", "full"}
    assert payload["shared_compile"]["shared_transpile"] is True
    assert payload["noise_config"]["mitigation"]["mode"] == "none"
    assert payload["noise_config"]["symmetry_mitigation"]["mode"] == "off"
    assert payload["slices"]["full"]["delta_mean"] == pytest.approx(1.3)
    assert payload["slice_comparisons"]["full_minus_readout_only"] == pytest.approx(1.15)
