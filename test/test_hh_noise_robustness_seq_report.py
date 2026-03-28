from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pipelines.exact_bench.hh_noise_robustness_seq_report as report


class _LayoutStub:
    logical_parameter_count = 2
    runtime_parameter_count = 2


def _locked_fixed_scaffold_ctx(
    tmp_path: Path,
) -> tuple[dict[str, object], dict[str, object], report.LockedImportedSubject, None]:
    return (
        {
            "path": tmp_path / "fixed_scaffold.json",
            "source_kind": "direct_adapt_artifact",
            "ordered_labels_exyz": ["ze"],
            "static_coeff_map_exyz": {"ze": 1.0 + 0.0j},
            "layout": _LayoutStub(),
            "num_qubits": 1,
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
            "theta_runtime": np.asarray([0.1, -0.2], dtype=float),
            "ansatz_input_state": np.asarray([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex),
        },
        {
            "source": "hf",
            "handoff_state_kind": "reference_state",
        },
        report.LockedImportedSubject(
            family="fixed_scaffold",
            pool_type="fixed_scaffold_locked",
            subject_kind="hh_nighthawk_gate_pruned_7term_v1",
            structure_locked=True,
            term_order_id="source_order",
            operator_count=7,
            runtime_term_count=7,
        ),
        None,
    )


def test_fixed_scaffold_runtime_raw_baseline_returns_report_friendly_energy_audits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )
    monkeypatch.setattr(
        report,
        "build_parameterized_ansatz_plan",
        lambda layout, nq, ref_state=None: SimpleNamespace(
            layout=layout,
            nq=int(nq),
            plan_digest="plan",
            structure_digest="structure",
            reference_state_digest="ref",
        ),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_energy",
        lambda **kwargs: {
            "noisy_mean": 1.2,
            "noisy_std": 0.1,
            "noisy_stdev": 0.1,
            "noisy_stderr": 0.05,
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.2,
            "delta_stderr": 0.05,
            "backend_info": {
                "noise_mode": "runtime",
                "estimator_kind": "raw_measurement_oracle",
                "backend_name": "ibm_test_backend",
                "using_fake_backend": False,
                "details": {
                    "execution_surface": "raw_measurement_v1",
                    "transport": "sampler_v2",
                    "raw_artifact_path": "artifacts/raw.ndjson.gz",
                    "record_count": 4,
                    "group_count": 2,
                    "term_count": 2,
                    "reduction_mode": "repeat_aligned_full_observable",
                    "plan_digest": "plan",
                    "structure_digest": "structure",
                    "reference_state_digest": "ref",
                    "compile_signatures_by_basis": {"Z": {"compiled_depth": 3}},
                    "backend_snapshot": {"backend_name": "ibm_test_backend"},
                    "seed_transpiler": 7,
                    "transpile_optimization_level": 1,
                },
            },
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_diagnostic",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": "diag-eval-1",
            "transport": "sampler_v2",
            "raw_artifact_path": "artifacts/raw.ndjson.gz",
            "record_count": 2,
            "group_count": 1,
            "term_count": 1,
            "compile_signatures_by_basis": {"ZZ": {"compiled_depth": 2}},
            "summary": {
                "sector_weight_mean": 0.75,
                "doublon_total_mean": 0.5,
                "sector_distribution": {
                    "weights_by_sector": [
                        {"n_up": 1, "n_dn": 1, "mean": 0.75, "std": 0.0, "stderr": 0.0}
                    ],
                    "sorted_by": "n_up_then_n_dn",
                },
                "site_observables": {
                    "site_index_order": "physical_site_0_to_n_minus_1",
                    "doublon_by_site_mean": [0.5, 0.0],
                    "doublon_by_site_std": [0.0, 0.0],
                    "doublon_by_site_stderr": [0.0, 0.0],
                    "charge_by_site_mean": [1.0, 0.0],
                    "charge_by_site_std": [0.0, 0.0],
                    "charge_by_site_stderr": [0.0, 0.0],
                },
                "observable_span": {
                    "supports_postselected_energy": False,
                    "supports_projector_renorm_energy": False,
                },
            },
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_validation",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "reference_source": "ideal_diagonal_v1",
            "metrics": {
                "sector_weight": {
                    "raw_mean": 0.75,
                    "raw_stderr": 0.1,
                    "reference": 1.0,
                    "delta": -0.25,
                },
                "doublon_total": {
                    "raw_mean": 0.5,
                    "raw_stderr": 0.1,
                    "reference": 0.6,
                    "delta": -0.1,
                },
            },
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_bootstrap",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "summary": {
                "source": "hh_full_register_z_bootstrap_v1",
                "bootstrap_repetitions": 32,
                "metrics": {
                    "sector_weight": {
                        "point_estimate": 0.75,
                        "bootstrap_mean": 0.74,
                        "bootstrap_std": 0.08,
                        "ci_lower": 0.55,
                        "ci_upper": 0.88,
                    }
                },
            },
            "error_type": None,
            "error_message": None,
        },
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="auto",
        raw_store_memory=False,
        raw_artifact_path="artifacts/raw.ndjson.gz",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["route"] == "fixed_scaffold_runtime_raw_baseline"
    assert payload["energy_audits"]["main"]["evaluation"]["delta_mean"] == pytest.approx(0.2)
    assert payload["backend_info"]["details"]["execution_surface"] == "raw_measurement_v1"
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["available"] is True
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["summary"]["sector_weight_mean"] == pytest.approx(0.75)
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["summary"]["sector_distribution"]["weights_by_sector"][0]["n_up"] == 1
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["summary"]["site_observables"]["doublon_by_site_mean"] == pytest.approx([0.5, 0.0])
    assert payload["energy_audits"]["main"]["symmetry_validation"]["available"] is True
    assert payload["energy_audits"]["main"]["symmetry_bootstrap"]["available"] is True
    assert payload["energy_audits"]["main"]["symmetry_bootstrap"]["summary"]["metrics"]["sector_weight"]["point_estimate"] == pytest.approx(0.75)
    assert payload["energy_audits"]["main"]["symmetry_validation"]["metrics"]["sector_weight"]["reference"] == pytest.approx(1.0)
    assert payload["energy_audits"]["main"]["evaluation"]["delta_mean"] == pytest.approx(0.2)
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["summary"]["observable_span"]["supports_postselected_energy"] is False
    assert payload["backend_info"]["details"]["symmetry_diagnostic"]["evaluation_id"] == "diag-eval-1"
    assert payload["compile_control"]["backend_name"] == "ibm_test_backend"
    assert payload["compile_control"]["seed_transpiler"] == 7
    assert payload["compile_control"]["transpile_optimization_level"] == 1
    assert payload["compile_control"]["source"] == "fixed_scaffold_runtime_transpile_cli"
    assert payload["compile_observation"]["available"] is True
    assert payload["compile_observation"]["matches_requested"] is True
    assert payload["compile_observation"]["observed"]["seed_transpiler"] == 7
    assert payload["compile_observation"]["observed"]["transpile_optimization_level"] == 1
    assert payload["energy_audits"]["dd_probe"]["reason"] == "raw_acquisition_only_v1"
    assert payload["energy_audits"]["final_audit_zne"]["reason"] == "raw_acquisition_only_v1"


def test_run_noisy_mode_isolated_nonzero_exit_returns_env_blocked_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQueue:
        def empty(self) -> bool:
            return True

    class _FakeProc:
        exitcode = -6

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return False

        def terminate(self) -> None:
            raise AssertionError("terminate should not be called for a finished child")

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_noisy_mode_isolated(kwargs={"noise_mode": "shots"}, timeout_s=30)

    assert payload["success"] is False
    assert payload["env_blocked"] is True
    assert payload["reason"] == "subprocess_nonzero_exit"
    assert payload["exitcode"] == -6


def test_fixed_scaffold_runtime_raw_baseline_keeps_main_success_when_symmetry_diagnostic_degrades(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )
    monkeypatch.setattr(
        report,
        "build_parameterized_ansatz_plan",
        lambda layout, nq, ref_state=None: SimpleNamespace(
            layout=layout,
            nq=int(nq),
            plan_digest="plan",
            structure_digest="structure",
            reference_state_digest="ref",
        ),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_energy",
        lambda **kwargs: {
            "noisy_mean": 1.2,
            "noisy_std": 0.1,
            "noisy_stdev": 0.1,
            "noisy_stderr": 0.05,
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.2,
            "delta_stderr": 0.05,
            "backend_info": {"details": {"execution_surface": "raw_measurement_v1"}},
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_diagnostic",
        lambda **kwargs: {
            "success": False,
            "available": False,
            "reason": "summary_failed",
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": "diag-eval-1",
            "transport": "sampler_v2",
            "raw_artifact_path": "artifacts/raw.ndjson.gz",
            "record_count": 2,
            "group_count": 1,
            "term_count": 1,
            "compile_signatures_by_basis": {"ZZ": {"compiled_depth": 2}},
            "summary": None,
            "error_type": "ValueError",
            "error_message": "synthetic summary failure",
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_validation",
        lambda **kwargs: {
            "success": False,
            "available": False,
            "reason": "symmetry_diagnostic_unavailable",
            "reference_source": None,
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_bootstrap",
        lambda **kwargs: {
            "success": False,
            "available": False,
            "reason": "symmetry_diagnostic_unavailable",
            "summary": None,
            "error_type": None,
            "error_message": None,
        },
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="auto",
        raw_store_memory=False,
        raw_artifact_path="artifacts/raw.ndjson.gz",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["available"] is False
    assert payload["energy_audits"]["main"]["symmetry_diagnostic"]["reason"] == "summary_failed"
    assert payload["energy_audits"]["main"]["symmetry_validation"]["available"] is False
    assert payload["energy_audits"]["main"]["symmetry_validation"]["reason"] == "symmetry_diagnostic_unavailable"
    assert payload["energy_audits"]["main"]["symmetry_bootstrap"]["available"] is False
    assert payload["energy_audits"]["main"]["symmetry_bootstrap"]["reason"] == "symmetry_diagnostic_unavailable"


def test_fixed_scaffold_runtime_raw_baseline_keeps_main_success_when_reference_validation_degrades(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )
    monkeypatch.setattr(
        report,
        "build_parameterized_ansatz_plan",
        lambda layout, nq, ref_state=None: SimpleNamespace(
            layout=layout,
            nq=int(nq),
            plan_digest="plan",
            structure_digest="structure",
            reference_state_digest="ref",
        ),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_energy",
        lambda **kwargs: {
            "noisy_mean": 1.2,
            "noisy_std": 0.1,
            "noisy_stdev": 0.1,
            "noisy_stderr": 0.05,
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.2,
            "delta_stderr": 0.05,
            "backend_info": {"details": {"execution_surface": "raw_measurement_v1"}},
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_diagnostic",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": "diag-eval-1",
            "transport": "sampler_v2",
            "raw_artifact_path": "artifacts/raw.ndjson.gz",
            "record_count": 2,
            "group_count": 1,
            "term_count": 1,
            "compile_signatures_by_basis": {"ZZ": {"compiled_depth": 2}},
            "summary": {
                "sector_weight_mean": 0.75,
                "sector_weight_stderr": 0.1,
                "doublon_total_mean": 0.5,
                "doublon_total_stderr": 0.1,
            },
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_validation",
        lambda **kwargs: {
            "success": False,
            "available": False,
            "reason": "reference_unavailable",
            "reference_source": "ideal_diagonal_v1",
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": "RuntimeError",
            "error_message": "synthetic reference failure",
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_bootstrap",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "summary": {
                "source": "hh_full_register_z_bootstrap_v1",
                "bootstrap_repetitions": 32,
                "metrics": {},
            },
            "error_type": None,
            "error_message": None,
        },
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="auto",
        raw_store_memory=False,
        raw_artifact_path="artifacts/raw.ndjson.gz",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["energy_audits"]["main"]["evaluation"]["delta_mean"] == pytest.approx(0.2)
    assert payload["energy_audits"]["main"]["symmetry_validation"]["available"] is False
    assert payload["energy_audits"]["main"]["symmetry_validation"]["reason"] == "reference_unavailable"


def test_fixed_scaffold_runtime_raw_baseline_keeps_main_success_when_symmetry_bootstrap_degrades(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )
    monkeypatch.setattr(
        report,
        "build_parameterized_ansatz_plan",
        lambda layout, nq, ref_state=None: SimpleNamespace(
            layout=layout,
            nq=int(nq),
            plan_digest="plan",
            structure_digest="structure",
            reference_state_digest="ref",
        ),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_energy",
        lambda **kwargs: {
            "noisy_mean": 1.2,
            "noisy_std": 0.1,
            "noisy_stdev": 0.1,
            "noisy_stderr": 0.05,
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.2,
            "delta_stderr": 0.05,
            "backend_info": {"details": {"execution_surface": "raw_measurement_v1"}},
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_diagnostic",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": "diag-eval-1",
            "transport": "sampler_v2",
            "raw_artifact_path": "artifacts/raw.ndjson.gz",
            "record_count": 2,
            "group_count": 1,
            "term_count": 1,
            "compile_signatures_by_basis": {"ZZ": {"compiled_depth": 2}},
            "summary": {
                "sector_weight_mean": 0.75,
                "sector_weight_stderr": 0.1,
                "doublon_total_mean": 0.5,
                "doublon_total_stderr": 0.1,
            },
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_validation",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "reference_source": "ideal_diagonal_v1",
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_bootstrap",
        lambda **kwargs: {
            "success": False,
            "available": False,
            "reason": "bootstrap_failed",
            "summary": None,
            "error_type": "ValueError",
            "error_message": "synthetic bootstrap failure",
        },
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="auto",
        raw_store_memory=False,
        raw_artifact_path="artifacts/raw.ndjson.gz",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["energy_audits"]["main"]["evaluation"]["delta_mean"] == pytest.approx(0.2)
    assert payload["energy_audits"]["main"]["symmetry_bootstrap"]["available"] is False
    assert payload["energy_audits"]["main"]["symmetry_bootstrap"]["reason"] == "bootstrap_failed"


def test_fixed_scaffold_runtime_raw_baseline_allows_fake_backend_local_diagonal_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )
    monkeypatch.setattr(
        report,
        "build_parameterized_ansatz_plan",
        lambda layout, nq, ref_state=None: SimpleNamespace(
            layout=layout,
            nq=int(nq),
            plan_digest="plan",
            structure_digest="structure",
            reference_state_digest="ref",
        ),
    )

    def _fake_energy(**kwargs):
        assert kwargs["use_fake_backend"] is True
        assert kwargs["backend_name"] == "FakeNighthawk"
        assert dict(kwargs["mitigation_config"])["mode"] == "none"
        assert dict(kwargs["symmetry_mitigation_config"])["mode"] == "off"
        return {
            "noisy_mean": 1.2,
            "noisy_std": 0.1,
            "noisy_stdev": 0.1,
            "noisy_stderr": 0.05,
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.2,
            "delta_stderr": 0.05,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "raw_measurement_oracle",
                "backend_name": "FakeNighthawk",
                "using_fake_backend": True,
                "details": {
                    "execution_surface": "raw_measurement_v1",
                    "transport": "backend_run",
                    "raw_artifact_path": "artifacts/raw.ndjson.gz",
                    "transpile_seed": 7,
                    "transpile_optimization_level": 1,
                },
            },
        }

    def _fake_symmetry_diag(**kwargs):
        assert kwargs["use_fake_backend"] is True
        assert dict(kwargs["mitigation_config"])["mode"] == "none"
        assert dict(kwargs["symmetry_mitigation_config"])["mode"] == "off"
        assert dict(kwargs["diagonal_postprocessing_mitigation_config"])["mode"] == "readout"
        assert (
            dict(kwargs["diagonal_postprocessing_mitigation_config"])["local_readout_strategy"]
            == "mthree"
        )
        assert (
            dict(kwargs["diagonal_postprocessing_symmetry_config"])["mode"]
            == "projector_renorm_v1"
        )
        return {
            "success": True,
            "available": True,
            "reason": None,
            "observable_family": "fixed_scaffold_runtime_all_z_symmetry_diagnostic",
            "evaluation_id": "diag-eval-local-1",
            "transport": "backend_run",
            "raw_artifact_path": "artifacts/raw.ndjson.gz",
            "record_count": 2,
            "group_count": 1,
            "term_count": 1,
            "compile_signatures_by_basis": {"ZZ": {"compiled_depth": 2}},
            "summary": {"sector_weight_mean": 0.7},
            "diagonal_postprocessing": {
                "success": True,
                "available": True,
                "reason": None,
                "summary": {
                    "source": "raw_full_register_z_postprocessed_v1",
                    "pipeline": {
                        "readout_mode": "readout",
                        "symmetry_mode": "projector_renorm_v1",
                    },
                    "sector_weight_mean": 0.72,
                    "doublon_total_mean": 0.48,
                },
                "readout_details": {"strategy": "mthree"},
                "error_type": None,
                "error_message": None,
            },
            "error_type": None,
            "error_message": None,
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_raw_energy", _fake_energy)
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_diagnostic",
        _fake_symmetry_diag,
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_validation",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "reference_source": "ideal_diagonal_v1",
            "metrics": {},
            "notes": ["diagnostic_only", "no_energy_correction"],
            "error_type": None,
            "error_message": None,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_raw_symmetry_bootstrap",
        lambda **kwargs: {
            "success": True,
            "available": True,
            "reason": None,
            "summary": {"source": "hh_full_register_z_bootstrap_v1", "metrics": {}},
            "error_type": None,
            "error_message": None,
        },
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeNighthawk",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout"},
        symmetry_mitigation_config={"mode": "projector_renorm_v1"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["noise_config"]["noise_mode"] == "backend_scheduled"
    assert payload["noise_config"]["mitigation"]["mode"] == "none"
    assert payload["noise_config"]["symmetry_mitigation"]["mode"] == "off"
    assert payload["noise_config"]["requested_diagonal_postprocessing"]["mitigation"]["mode"] == "readout"
    assert payload["noise_config"]["requested_diagonal_postprocessing"]["mitigation"]["local_readout_strategy"] == "mthree"
    assert payload["noise_config"]["requested_diagonal_postprocessing"]["symmetry_mitigation"]["mode"] == "projector_renorm_v1"
    assert payload["energy_audits"]["main"]["diagonal_postprocessing"]["available"] is True
    assert payload["backend_info"]["details"]["symmetry_diagnostic"]["diagonal_postprocessing_available"] is True
    assert payload["compile_control"]["backend_name"] == "FakeNighthawk"
    assert payload["compile_control"]["seed_transpiler"] == 7
    assert payload["compile_control"]["transpile_optimization_level"] == 1
    assert payload["compile_observation"]["available"] is True
    assert payload["compile_observation"]["matches_requested"] is True


def test_fixed_scaffold_runtime_raw_baseline_rejects_non_none_mitigation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_scaffold_runtime_raw_baseline_requires_no_mitigation"


def test_fixed_scaffold_runtime_raw_baseline_rejects_nonlegacy_runtime_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "main_twirled_readout_v1"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_scaffold_runtime_raw_baseline_requires_legacy_runtime_profile"


def test_fixed_scaffold_runtime_raw_baseline_rejects_verify_only_symmetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "verify_only"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="sampler_v2",
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_scaffold_runtime_raw_baseline_requires_symmetry_off"


def test_imported_fixed_scaffold_noisy_replay_allows_symmetry_only_backend_scheduled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )
    monkeypatch.setattr(
        report,
        "build_time_dependent_sparse_qop",
        lambda ordered_labels_exyz, static_coeff_map_exyz, drive_coeff_map_exyz=None: "qop",
    )
    monkeypatch.setattr(report, "_build_ansatz_circuit", lambda layout, theta, nq, ref_state=None: "qc")
    monkeypatch.setattr(
        report,
        "project_runtime_theta_block_mean",
        lambda theta, layout: np.asarray(theta, dtype=float),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_energy",
        lambda **kwargs: {
            "noisy_mean": -0.71,
            "noisy_stderr": 0.01,
            "ideal_mean": -0.73,
            "ideal_stderr": 0.0,
            "delta_mean": 0.02,
            "delta_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "expectation_oracle",
                "backend_name": "FakeMarrakesh",
                "using_fake_backend": True,
                "details": {},
            },
        },
    )
    monkeypatch.setattr(
        report,
        "_run_saved_theta_local_mitigation_ablation",
        lambda **kwargs: {"success": True, "available": True, "results": {}},
    )

    class _OracleStub:
        def __init__(self, config):
            self.config = config
            assert str(config.noise_mode) == "backend_scheduled"
            assert dict(config.mitigation).get("mode") == "none"
            assert dict(config.symmetry_mitigation).get("mode") == "projector_renorm_v1"
            self.backend_info = SimpleNamespace(
                noise_mode="backend_scheduled",
                estimator_kind="expectation_oracle",
                backend_name="FakeMarrakesh",
                using_fake_backend=True,
                details={},
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def evaluate(self, qc, qop, runtime_job_observer=None, runtime_trace_context=None):
            return SimpleNamespace(mean=-0.71, stderr=0.01)

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)
    monkeypatch.setattr(
        report,
        "spsa_minimize",
        lambda **kwargs: SimpleNamespace(
            x=np.asarray(kwargs["x0"], dtype=float),
            nit=1,
            nfev=1,
            success=True,
            message="ok",
        ),
    )

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
        optimizer_maxiter=10,
        optimizer_wallclock_cap_s=60,
        spsa_a=0.1,
        spsa_c=0.1,
        spsa_alpha=0.602,
        spsa_gamma=0.101,
        spsa_A=5.0,
        spsa_avg_last=1,
        spsa_eval_repeats=1,
        spsa_eval_agg="mean",
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={
            "mode": "projector_renorm_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        },
        transpile_optimization_level=2,
        seed_transpiler=5,
        include_dd_probe=False,
        include_final_zne_audit=False,
        progress_every_s=60.0,
        local_dd_probe_sequence=None,
    )

    assert payload["success"] is True
    assert payload["noise_config"]["mitigation"]["mode"] == "none"
    assert payload["noise_config"]["symmetry_mitigation"]["mode"] == "projector_renorm_v1"
    assert payload["local_mitigation_label"] == "none"


def test_fixed_scaffold_runtime_raw_baseline_rejects_backend_run_transport(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_resolve_locked_imported_fixed_scaffold_context",
        lambda artifact_json, nonfixed_reason: _locked_fixed_scaffold_ctx(tmp_path),
    )

    payload = report._run_imported_fixed_scaffold_runtime_raw_baseline(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
        runtime_profile_config={"name": "legacy_runtime_v0"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="backend_run",
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_scaffold_runtime_raw_baseline_requires_sampler_transport"
