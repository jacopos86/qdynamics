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
            "exact_energy": 1.08,
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


def test_fixed_scaffold_runtime_raw_baseline_rejects_estimator_style_runtime_profile(
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
    assert payload["reason"] == "fixed_scaffold_runtime_raw_baseline_requires_sampler_safe_runtime_profile"


def test_fixed_scaffold_runtime_raw_baseline_accepts_verify_only_symmetry(
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
            "evaluation_id": "diag-eval-verify-only",
            "transport": "sampler_v2",
            "raw_artifact_path": "artifacts/raw.ndjson.gz",
            "record_count": 2,
            "group_count": 1,
            "term_count": 1,
            "compile_signatures_by_basis": {"ZZ": {"compiled_depth": 2}},
            "summary": {"sector_weight_mean": 0.75},
            "diagonal_postprocessing": {
                "success": False,
                "available": False,
                "reason": "local_fake_backend_only",
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
        backend_name="ibm_test_backend",
        use_fake_backend=False,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        mitigation_config={"mode": "none"},
        symmetry_mitigation_config={"mode": "verify_only"},
        runtime_profile_config={"name": "raw_sampler_twirled_v1"},
        runtime_session_config={"mode": "prefer_session"},
        transpile_optimization_level=1,
        seed_transpiler=7,
        raw_transport="sampler_v2",
    )

    assert payload["success"] is True
    assert payload["available"] is True
    assert payload["noise_config"]["symmetry_mitigation"]["mode"] == "verify_only"
    assert payload["noise_config"]["runtime_profile"]["name"] == "raw_sampler_twirled_v1"


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


def test_imported_fixed_scaffold_noisy_replay_emits_partial_handoff_progress(
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
                "details": {"seed_transpiler": 5, "transpile_optimization_level": 2},
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
            self.backend_info = SimpleNamespace(
                noise_mode="backend_scheduled",
                estimator_kind="expectation_oracle",
                backend_name="FakeMarrakesh",
                using_fake_backend=True,
                details={"seed_transpiler": 5, "transpile_optimization_level": 2},
            )

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def evaluate(self, qc, qop, runtime_job_observer=None, runtime_trace_context=None):
            assert qc == "qc"
            assert qop == "qop"
            assert runtime_trace_context["route"] == "fixed_scaffold_noisy_replay"
            if runtime_job_observer is not None:
                runtime_job_observer({"job": {"job_id": "job-1", "status": "DONE"}})
            return SimpleNamespace(mean=-0.71, stderr=0.01)

    def _fake_spsa_minimize(**kwargs):
        x0 = np.asarray(kwargs["x0"], dtype=float)
        value = float(kwargs["fun"](x0))
        callback = kwargs.get("callback")
        if callable(callback):
            callback({"iter": 1, "nfev_so_far": 1, "grad_norm": 0.0, "fx": value})
        return SimpleNamespace(
            x=x0,
            nit=1,
            nfev=1,
            success=True,
            message="ok",
        )

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)
    monkeypatch.setattr(report, "spsa_minimize", _fake_spsa_minimize)

    progress_events: list[dict[str, object]] = []
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
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        transpile_optimization_level=2,
        seed_transpiler=5,
        include_dd_probe=False,
        include_final_zne_audit=False,
        progress_every_s=60.0,
        local_dd_probe_sequence=None,
        progress_callback=lambda msg: progress_events.append(dict(msg)),
    )

    event_names = [str(rec["event"]) for rec in progress_events]
    completed = [
        rec for rec in progress_events if rec.get("event") == "fixed_scaffold_noisy_replay_objective_completed"
    ]
    assert payload["success"] is True
    assert payload["partial"] is False
    assert payload["best_so_far"]["call_index"] == 1
    assert event_names == [
        "fixed_scaffold_noisy_replay_initialized",
        "fixed_scaffold_noisy_replay_objective_completed",
    ]
    assert len(completed) == 1
    assert completed[0]["current_energy_noisy_mean"] == pytest.approx(-0.71)
    assert completed[0]["partial_payload"]["partial"] is True
    assert completed[0]["partial_payload"]["objective_trace"][0]["runtime_jobs"][0]["job_id"] == "job-1"
    assert completed[0]["partial_payload"]["best_so_far"]["call_index"] == 1
    assert completed[0]["partial_payload"]["theta"]["best_runtime"] == pytest.approx([0.1, -0.2])


def test_fixed_scaffold_noisy_replay_mode_isolated_timeout_preserves_partial_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQueue:
        def __init__(self) -> None:
            self._msgs = [
                {
                    "kind": "progress",
                    "payload": {
                        "event": "fixed_scaffold_noisy_replay_objective_completed",
                        "partial_payload": {
                            "route": "fixed_scaffold_noisy_replay",
                            "partial": True,
                            "theta_source": "imported_theta_runtime",
                            "execution_mode": "backend_scheduled",
                            "local_mitigation_label": "readout_only",
                            "objective_trace": [
                                {
                                    "call_index": 1,
                                    "energy_noisy_mean": -0.71,
                                    "energy_noisy_stderr": 0.01,
                                }
                            ],
                            "best_so_far": {
                                "call_index": 1,
                                "energy_noisy_mean": -0.71,
                                "energy_noisy_stderr": 0.01,
                            },
                            "optimizer": {
                                "method": "SPSA",
                                "stop_reason": "in_progress",
                                "objective_calls_total": 1,
                            },
                        },
                    },
                }
            ]

        def get_nowait(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise report.pyqueue.Empty()

    class _FakeProc:
        exitcode = None

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monotonic_values = iter([100.0, 100.0, 100.0, 100.5, 100.5, 100.5])
    monkeypatch.setattr(report.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_imported_fixed_scaffold_noisy_replay_mode_isolated(
        kwargs={"artifact_json": "artifacts/json/example.json"},
        timeout_s=0,
    )

    assert payload["success"] is False
    assert payload["env_blocked"] is True
    assert payload["partial"] is True
    assert payload["reason"] == "timeout_after_0s"
    assert payload["objective_trace"][0]["energy_noisy_mean"] == pytest.approx(-0.71)
    assert payload["best_so_far"]["call_index"] == 1
    assert payload["theta_source"] == "imported_theta_runtime"
    assert payload["execution_mode"] == "backend_scheduled"
    assert payload["local_mitigation_label"] == "readout_only"
    assert payload["last_progress_event"] == "fixed_scaffold_noisy_replay_objective_completed"
    assert payload["optimizer"]["stop_reason"] == "timeout_after_0s"


def test_fixed_scaffold_noisy_replay_mode_isolated_timeout_preserves_initialized_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQueue:
        def __init__(self) -> None:
            self._msgs = [
                {
                    "kind": "progress",
                    "payload": {
                        "event": "fixed_scaffold_noisy_replay_initialized",
                        "partial_payload": {
                            "route": "fixed_scaffold_noisy_replay",
                            "partial": True,
                            "theta_source": "imported_theta_runtime",
                            "execution_mode": "backend_scheduled",
                            "local_mitigation_label": "readout_only",
                            "objective_trace": [],
                            "runtime_job_ids": [],
                            "optimizer": {
                                "method": "SPSA",
                                "stop_reason": "in_progress",
                                "objective_calls_total": 0,
                            },
                        },
                    },
                }
            ]

        def get_nowait(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise report.pyqueue.Empty()

    class _FakeProc:
        exitcode = None

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monotonic_values = iter([200.0, 200.0, 200.0, 200.5, 200.5, 200.5])
    monkeypatch.setattr(report.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_imported_fixed_scaffold_noisy_replay_mode_isolated(
        kwargs={"artifact_json": "artifacts/json/example.json"},
        timeout_s=0,
    )

    assert payload["success"] is False
    assert payload["env_blocked"] is True
    assert payload["partial"] is True
    assert payload["reason"] == "timeout_after_0s"
    assert payload["objective_trace"] == []
    assert payload["runtime_job_ids"] == []
    assert payload["theta_source"] == "imported_theta_runtime"
    assert payload["execution_mode"] == "backend_scheduled"
    assert payload["last_progress_event"] == "fixed_scaffold_noisy_replay_initialized"
    assert payload["optimizer"]["stop_reason"] == "timeout_after_0s"


def test_fixed_scaffold_noisy_replay_mode_isolated_missing_result_preserves_partial_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQueue:
        def __init__(self) -> None:
            self._msgs = [
                {
                    "kind": "progress",
                    "payload": {
                        "event": "fixed_scaffold_noisy_replay_objective_completed",
                        "partial_payload": {
                            "route": "fixed_scaffold_noisy_replay",
                            "partial": True,
                            "theta_source": "imported_theta_runtime",
                            "execution_mode": "backend_scheduled",
                            "local_mitigation_label": "readout_only",
                            "objective_trace": [{"call_index": 1, "energy_noisy_mean": -0.71}],
                            "best_so_far": {"call_index": 1, "energy_noisy_mean": -0.71},
                            "optimizer": {
                                "method": "SPSA",
                                "stop_reason": "in_progress",
                                "objective_calls_total": 1,
                            },
                        },
                    },
                }
            ]

        def get_nowait(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise report.pyqueue.Empty()

    class _FakeProc:
        exitcode = 0

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

    monotonic_values = iter([300.0, 300.0, 300.5, 300.5, 300.5, 300.6, 300.6, 300.7])
    monkeypatch.setattr(report.time, "monotonic", lambda: next(monotonic_values, 301.0))
    monkeypatch.setattr(report.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_imported_fixed_scaffold_noisy_replay_mode_isolated(
        kwargs={"artifact_json": "artifacts/json/example.json"},
        timeout_s=30,
    )

    assert payload["success"] is False
    assert payload["env_blocked"] is True
    assert payload["partial"] is True
    assert payload["reason"] == "subprocess_completed_without_payload"
    assert payload["best_so_far"]["call_index"] == 1
    assert payload["last_progress_event"] == "fixed_scaffold_noisy_replay_objective_completed"
    assert payload["optimizer"]["stop_reason"] == "subprocess_completed_without_payload"


def test_fixed_scaffold_noisy_replay_mode_isolated_waits_for_delayed_result_after_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_payload = {
        "success": True,
        "route": "fixed_scaffold_noisy_replay",
        "theta_source": "imported_theta_runtime",
        "execution_mode": "backend_scheduled",
        "local_mitigation_label": "readout_only",
    }

    class _FakeQueue:
        def __init__(self) -> None:
            self._call_index = 0

        def get_nowait(self):
            self._call_index += 1
            if self._call_index == 1:
                return {
                    "kind": "progress",
                    "payload": {
                        "event": "fixed_scaffold_noisy_replay_objective_completed",
                        "partial_payload": {
                            "route": "fixed_scaffold_noisy_replay",
                            "partial": True,
                            "objective_trace": [{"call_index": 1, "energy_noisy_mean": -0.71}],
                            "optimizer": {"stop_reason": "in_progress"},
                        },
                    },
                }
            if self._call_index == 2:
                raise report.pyqueue.Empty()
            if self._call_index == 3:
                raise report.pyqueue.Empty()
            if self._call_index == 4:
                return {"kind": "result", "ok": True, "payload": dict(result_payload)}
            raise report.pyqueue.Empty()

    class _FakeProc:
        exitcode = 0

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

    monotonic_values = iter([400.0, 400.0, 400.1, 400.1, 400.2, 400.2])
    monkeypatch.setattr(report.time, "monotonic", lambda: next(monotonic_values, 401.0))
    monkeypatch.setattr(report.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_imported_fixed_scaffold_noisy_replay_mode_isolated(
        kwargs={"artifact_json": "artifacts/json/example.json"},
        timeout_s=30,
    )

    assert payload == result_payload


def test_fixed_scaffold_noisy_replay_mode_isolated_prefers_cached_result_before_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_payload = {
        "success": True,
        "route": "fixed_scaffold_noisy_replay",
        "theta_source": "imported_theta_runtime",
    }

    class _FakeQueue:
        def __init__(self) -> None:
            self._emitted = False

        def get_nowait(self):
            if not self._emitted:
                self._emitted = True
                return {"kind": "result", "ok": True, "payload": dict(result_payload)}
            raise report.pyqueue.Empty()

    class _FakeProc:
        def __init__(self) -> None:
            self.exitcode = None
            self._alive = True

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return self._alive

        def terminate(self) -> None:
            self._alive = False
            self.exitcode = 0

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monotonic_values = iter([500.0, 500.0, 500.0])
    monkeypatch.setattr(report.time, "monotonic", lambda: next(monotonic_values, 501.0))
    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_imported_fixed_scaffold_noisy_replay_mode_isolated(
        kwargs={"artifact_json": "artifacts/json/example.json"},
        timeout_s=0,
    )

    assert payload == result_payload


def test_imported_fixed_scaffold_noisy_replay_preserves_partial_handoff_on_optimizer_exception(
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
            if runtime_job_observer is not None:
                runtime_job_observer({"job": {"job_id": "job-1", "status": "DONE"}})
            return SimpleNamespace(mean=-0.71, stderr=0.01)

    def _raising_spsa(**kwargs):
        kwargs["fun"](np.asarray(kwargs["x0"], dtype=float))
        raise RuntimeError("synthetic optimizer failure")

    monkeypatch.setattr(report, "ExpectationOracle", _OracleStub)
    monkeypatch.setattr(report, "spsa_minimize", _raising_spsa)

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
        symmetry_mitigation_config={"mode": "off"},
        transpile_optimization_level=2,
        seed_transpiler=5,
        include_dd_probe=False,
        include_final_zne_audit=False,
        progress_every_s=60.0,
        local_dd_probe_sequence=None,
    )

    assert payload["success"] is False
    assert payload["reason"] == "optimizer_exception"
    assert payload["partial"] is True
    assert payload["objective_trace"][0]["energy_noisy_mean"] == pytest.approx(-0.71)
    assert payload["objective_trace"][0]["runtime_jobs"][0]["job_id"] == "job-1"
    assert payload["runtime_job_ids"] == ["job-1"]
    assert payload["best_so_far"]["call_index"] == 1
    assert payload["optimizer"]["stop_reason"] == "optimizer_exception"
    assert "synthetic optimizer failure" in str(payload["error"])


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


def test_saved_theta_local_mitigation_ablation_adds_twirling_plus_dd_phase(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_energy",
        lambda **kwargs: {
            "noisy_mean": -0.9,
            "noisy_stderr": 0.01,
            "ideal_mean": -1.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.1,
            "delta_stderr": 0.01,
            "backend_info": {},
        },
    )
    qc = report.QuantumCircuit(1)
    subject = _locked_fixed_scaffold_ctx(tmp_path)[2]

    payload = report._run_saved_theta_local_mitigation_ablation(
        circuit=qc,
        artifact_json="artifacts/json/example.json",
        subject=subject,
        ordered_labels_exyz=["ze"],
        static_coeff_map_exyz={"ze": 1.0 + 0.0j},
        shots=128,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        seed_transpiler=4,
        transpile_optimization_level=1,
        local_dd_probe_sequence="XpXm",
    )

    results = payload["results"]
    assert "readout_only" in results
    assert "readout_plus_gate_twirling" in results
    assert "readout_plus_local_dd" in results
    assert "readout_plus_gate_twirling_plus_local_dd" in results
    twirl_dd = results["readout_plus_gate_twirling_plus_local_dd"]
    assert twirl_dd["mitigation_config"]["local_gate_twirling"] is True
    assert twirl_dd["mitigation_config"]["dd_sequence"] == "XpXm"


def test_evaluate_locked_imported_circuit_energy_local_zne_records_per_factor_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}
    noisy_means = [1.1, 1.3, 1.5]

    def _fake_noisy_eval(**kwargs):
        idx = calls["count"]
        calls["count"] += 1
        return {
            "noisy_mean": float(noisy_means[idx]),
            "noisy_std": 0.02,
            "noisy_stdev": 0.02,
            "noisy_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "expectation_oracle",
                "backend_name": "FakeMarrakesh",
                "using_fake_backend": True,
                "details": {
                    "seed_transpiler": kwargs["seed_transpiler"],
                    "transpile_optimization_level": kwargs["transpile_optimization_level"],
                    "compiled_two_qubit_count": 14,
                    "compiled_depth": 48,
                    "compiled_size": 80,
                },
            },
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_noisy_energy", _fake_noisy_eval)

    reference_circuit = report.QuantumCircuit(1)
    body_circuit = report.QuantumCircuit(1)
    body_circuit.rz(0.2, 0)

    payload = report._evaluate_locked_imported_circuit_energy_local_zne(
        reference_circuit=reference_circuit,
        body_circuit=body_circuit,
        ordered_labels_exyz=["ze"],
        static_coeff_map_exyz={"ze": 1.0 + 0.0j},
        shots=128,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        mitigation_config={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        seed_transpiler=4,
        transpile_optimization_level=1,
        zne_scales=[1.0, 3.0, 5.0],
        ideal_eval={
            "ideal_mean": 0.8,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
        exact_energy=0.6,
    )

    assert payload["success"] is True
    assert payload["zne_enabled"] is True
    assert payload["zne_scales"] == [1.0, 3.0, 5.0]
    assert len(payload["per_factor_results"]) == 3
    assert payload["compile_observation"]["matches_requested"] is True
    assert payload["compiled_two_qubit_count"] == 14
    assert payload["compiled_depth"] == 48
    assert payload["compiled_size"] == 80
    assert payload["extrapolator"] == "linear"
    assert payload["delta_mean"] == pytest.approx(0.2, abs=1e-9)
    assert payload["delta_to_ideal_mean"] == pytest.approx(0.2, abs=1e-9)
    assert payload["delta_to_exact_mean"] == pytest.approx(0.4, abs=1e-9)
    assert payload["zne_fold_scope"] == "prepared_circuit_full"


def test_build_local_zne_folded_circuit_marks_body_only_when_full_prepared_fold_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(report, "_prepared_circuit_inverse_or_none", lambda circuit: None)
    reference_circuit = report.QuantumCircuit(1)
    body_circuit = report.QuantumCircuit(1)
    body_circuit.rz(0.2, 0)

    _, metadata = report._build_local_zne_folded_circuit(
        reference_circuit=reference_circuit,
        body_circuit=body_circuit,
        noise_scale=3.0,
    )

    assert metadata["zne_fold_scope"] == "body_only"
    assert metadata["warning"] == "reference_state_prep_not_folded"


def test_fixed_scaffold_saved_theta_mitigation_matrix_emits_12_cells_and_ranks_best(
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
        "_build_ansatz_circuit",
        lambda layout, theta, nq, ref_state=None: report.QuantumCircuit(int(nq)),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        lambda **kwargs: {
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
    )

    noisy_delta_map = {
        (1, 4, "readout_plus_gate_twirling"): 0.10,
        (1, 4, "readout_plus_local_dd"): 0.08,
        (1, 4, "readout_plus_gate_twirling_plus_local_dd"): 0.06,
        (2, 0, "readout_plus_gate_twirling"): 0.12,
        (2, 0, "readout_plus_local_dd"): 0.09,
        (2, 0, "readout_plus_gate_twirling_plus_local_dd"): 0.11,
    }
    zne_delta_map = {
        (1, 4, "readout_plus_gate_twirling"): 0.09,
        (1, 4, "readout_plus_local_dd"): 0.07,
        (1, 4, "readout_plus_gate_twirling_plus_local_dd"): 0.05,
        (2, 0, "readout_plus_gate_twirling"): 0.06,
        (2, 0, "readout_plus_local_dd"): 0.08,
        (2, 0, "readout_plus_gate_twirling_plus_local_dd"): 0.04,
    }

    def _fake_noisy_eval(**kwargs):
        opt = int(kwargs["transpile_optimization_level"])
        seed_trans = int(kwargs["seed_transpiler"])
        mit = kwargs["mitigation_config"]
        if bool(mit.get("local_gate_twirling", False)) and mit.get("dd_sequence", None) not in {None, "", "none"}:
            stack = "readout_plus_gate_twirling_plus_local_dd"
        elif bool(mit.get("local_gate_twirling", False)):
            stack = "readout_plus_gate_twirling"
        else:
            stack = "readout_plus_local_dd"
        delta = float(noisy_delta_map[(opt, seed_trans, stack)])
        depth = 48 if opt == 1 else 38
        size = 80 if opt == 1 else 76
        return {
            "noisy_mean": 1.0 + delta,
            "noisy_std": 0.01,
            "noisy_stdev": 0.01,
            "noisy_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "estimator_kind": "expectation_oracle",
                "backend_name": "FakeMarrakesh",
                "using_fake_backend": True,
                "details": {
                    "seed_transpiler": seed_trans,
                    "transpile_optimization_level": opt,
                    "compiled_two_qubit_count": 14,
                    "compiled_depth": depth,
                    "compiled_size": size,
                },
            },
        }

    def _fake_zne_eval(**kwargs):
        opt = int(kwargs["transpile_optimization_level"])
        seed_trans = int(kwargs["seed_transpiler"])
        mit = kwargs["mitigation_config"]
        if bool(mit.get("local_gate_twirling", False)) and mit.get("dd_sequence", None) not in {None, "", "none"}:
            stack = "readout_plus_gate_twirling_plus_local_dd"
        elif bool(mit.get("local_gate_twirling", False)):
            stack = "readout_plus_gate_twirling"
        else:
            stack = "readout_plus_local_dd"
        delta = float(zne_delta_map[(opt, seed_trans, stack)])
        depth = 48 if opt == 1 else 38
        size = 80 if opt == 1 else 76
        compile_request = report._build_compile_request_payload(
            backend_name="FakeMarrakesh",
            seed_transpiler=seed_trans,
            transpile_optimization_level=opt,
            source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
        )
        backend_info = {
            "noise_mode": "backend_scheduled",
            "estimator_kind": "expectation_oracle",
            "backend_name": "FakeMarrakesh",
            "using_fake_backend": True,
            "details": {
                "seed_transpiler": seed_trans,
                "transpile_optimization_level": opt,
                "compiled_two_qubit_count": 14,
                "compiled_depth": depth,
                "compiled_size": size,
            },
        }
        return {
            "success": True,
            "zne_enabled": True,
            "zne_scales": [1.0, 3.0, 5.0],
            "per_factor_results": [
                {"noise_scale": 1.0, "delta_mean": delta + 0.01},
                {"noise_scale": 3.0, "delta_mean": delta + 0.03},
                {"noise_scale": 5.0, "delta_mean": delta + 0.05},
            ],
            "extrapolator": "linear",
            "compile_request": dict(compile_request),
            "compile_observation": report._build_compile_observation_payload(
                requested=compile_request,
                backend_info=backend_info,
            ),
            "matches_requested": True,
            "transpile_seed": seed_trans,
            "transpile_optimization_level": opt,
            "compiled_two_qubit_count": 14,
            "compiled_depth": depth,
            "compiled_size": size,
            "backend_info": backend_info,
            "noisy_mean": 1.0 + delta,
            "noisy_stderr": 0.02,
            "ideal_mean": 1.0,
            "ideal_stderr": 0.0,
            "delta_mean": delta,
            "delta_stderr": 0.02,
            "extrapolated_energy_mean": 1.0 + delta,
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_noisy_energy", _fake_noisy_eval)
    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy_local_zne", _fake_zne_eval)

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        compile_presets=(
            {"label": "opt1_seed4", "transpile_optimization_level": 1, "seed_transpiler": 4},
            {"label": "opt2_seed0", "transpile_optimization_level": 2, "seed_transpiler": 0},
        ),
        zne_scales=(1.0, 3.0, 5.0),
        suppression_labels=(
            "readout_plus_gate_twirling",
            "readout_plus_local_dd",
            "readout_plus_gate_twirling_plus_local_dd",
        ),
        mitigation_config_base={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    labels = [str(rec["label"]) for rec in payload["cells"]]
    assert payload["success"] is True
    assert payload["route"] == "fixed_scaffold_saved_theta_mitigation_matrix"
    assert len(payload["cells"]) == 12
    assert labels == [
        "opt1_seed4__zne_off__twirl",
        "opt1_seed4__zne_off__dd",
        "opt1_seed4__zne_off__twirl_dd",
        "opt1_seed4__zne_on__twirl",
        "opt1_seed4__zne_on__dd",
        "opt1_seed4__zne_on__twirl_dd",
        "opt2_seed0__zne_off__twirl",
        "opt2_seed0__zne_off__dd",
        "opt2_seed0__zne_off__twirl_dd",
        "opt2_seed0__zne_on__twirl",
        "opt2_seed0__zne_on__dd",
        "opt2_seed0__zne_on__twirl_dd",
    ]
    assert payload["best_cell"]["label"] == "opt2_seed0__zne_on__twirl_dd"
    assert payload["best_cell_by_ideal_abs"]["label"] == "opt2_seed0__zne_on__twirl_dd"
    assert payload["best_cell_by_exact_abs"]["label"] == "opt2_seed0__zne_on__dd"
    assert payload["best_by_compile_preset"]["opt1_seed4"]["label"] == "opt1_seed4__zne_on__twirl_dd"
    assert payload["best_by_compile_preset"]["opt2_seed0"]["label"] == "opt2_seed0__zne_on__twirl_dd"
    assert payload["best_by_compile_preset_by_exact_abs"]["opt2_seed0"]["label"] == "opt2_seed0__zne_on__dd"
    assert payload["best_by_zne_toggle"]["zne_on"]["label"] == "opt2_seed0__zne_on__twirl_dd"
    assert payload["best_by_zne_toggle_by_exact_abs"]["zne_on"]["label"] == "opt2_seed0__zne_on__dd"
    assert payload["best_by_suppression_stack"]["readout_plus_gate_twirling_plus_local_dd"]["label"] == (
        "opt2_seed0__zne_on__twirl_dd"
    )
    assert payload["best_by_suppression_stack_by_exact_abs"]["readout_plus_local_dd"]["label"] == (
        "opt2_seed0__zne_on__dd"
    )


def test_rank_imported_compile_control_candidates_uses_absolute_delta() -> None:
    ranked = report._rank_imported_compile_control_candidates(
        [
            {
                "label": "large_negative",
                "delta_mean": -0.2,
                "compiled_two_qubit_count": 10,
                "compiled_depth": 20,
                "compiled_size": 30,
                "transpile_optimization_level": 1,
                "transpile_seed": 4,
            },
            {
                "label": "small_positive",
                "delta_mean": 0.01,
                "compiled_two_qubit_count": 10,
                "compiled_depth": 20,
                "compiled_size": 30,
                "transpile_optimization_level": 1,
                "transpile_seed": 4,
            },
        ],
        rank_policy="delta_mean_then_two_qubit_then_depth_then_size",
    )

    assert [str(rec["label"]) for rec in ranked] == ["small_positive", "large_negative"]


def test_fixed_scaffold_saved_theta_mitigation_matrix_selected_cells_filters_to_shortlist(
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
        "_build_ansatz_circuit",
        lambda layout, theta, nq, ref_state=None: report.QuantumCircuit(int(nq)),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        lambda **kwargs: {
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_noisy_energy",
        lambda **kwargs: {
            "noisy_mean": 1.2,
            "noisy_std": 0.01,
            "noisy_stdev": 0.01,
            "noisy_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "backend_name": "FakeMarrakesh",
                "details": {
                    "seed_transpiler": kwargs["seed_transpiler"],
                    "transpile_optimization_level": kwargs["transpile_optimization_level"],
                    "compiled_two_qubit_count": 18,
                    "compiled_depth": 43,
                    "compiled_size": 83,
                },
            },
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_energy_local_zne",
        lambda **kwargs: {
            "success": True,
            "zne_enabled": True,
            "zne_scales": [1.0, 3.0, 5.0],
            "per_factor_results": [{"noise_scale": 1.0, "delta_mean": 0.05}],
            "extrapolator": "linear",
            "compile_request": report._build_compile_request_payload(
                backend_name="FakeMarrakesh",
                seed_transpiler=kwargs["seed_transpiler"],
                transpile_optimization_level=kwargs["transpile_optimization_level"],
                source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
            ),
            "compile_observation": {
                "available": True,
                "requested": {},
                "observed": {},
                "matches_requested": True,
                "mismatch_fields": [],
            },
            "matches_requested": True,
            "transpile_seed": kwargs["seed_transpiler"],
            "transpile_optimization_level": kwargs["transpile_optimization_level"],
            "compiled_two_qubit_count": 18,
            "compiled_depth": 43,
            "compiled_size": 83,
            "backend_info": {},
            "noisy_mean": 1.05,
            "noisy_stderr": 0.02,
            "ideal_mean": 1.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.05,
            "delta_stderr": 0.02,
            "extrapolated_energy_mean": 1.05,
        },
    )

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        compile_presets=(
            {"label": "opt2_seed5", "transpile_optimization_level": 2, "seed_transpiler": 5},
        ),
        zne_scales=(1.0, 3.0, 5.0),
        suppression_labels=(
            "readout_plus_gate_twirling",
            "readout_plus_local_dd",
            "readout_plus_gate_twirling_plus_local_dd",
        ),
        selected_cells=(
            "opt2_seed5__zne_on__twirl_dd",
            "opt2_seed5__zne_on__twirl",
            "opt2_seed5__zne_on__dd",
            "opt2_seed5__zne_off__twirl_dd",
        ),
        mitigation_config_base={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is True
    assert payload["selected_cells"] == [
        "opt2_seed5__zne_on__twirl_dd",
        "opt2_seed5__zne_on__twirl",
        "opt2_seed5__zne_on__dd",
        "opt2_seed5__zne_off__twirl_dd",
    ]
    assert payload["cell_counts"] == {
        "total": 4,
        "completed": 4,
        "successful": 4,
        "failed": 0,
    }
    assert [str(rec["label"]) for rec in payload["cells"]] == [
        "opt2_seed5__zne_off__twirl_dd",
        "opt2_seed5__zne_on__twirl",
        "opt2_seed5__zne_on__dd",
        "opt2_seed5__zne_on__twirl_dd",
    ]


def test_fixed_scaffold_saved_theta_mitigation_matrix_rejects_unsupported_selected_cells(
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
        "_build_ansatz_circuit",
        lambda layout, theta, nq, ref_state=None: report.QuantumCircuit(int(nq)),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        lambda **kwargs: {
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
    )

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        compile_presets=(
            {"label": "opt2_seed5", "transpile_optimization_level": 2, "seed_transpiler": 5},
        ),
        selected_cells=("opt2_seed5__zne_on__bogus",),
        mitigation_config_base={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is False
    assert payload["available"] is False
    assert payload["reason"] == "fixed_scaffold_saved_theta_mitigation_matrix_unsupported_selected_cells"
    assert payload["unsupported_selected_cells"] == ["opt2_seed5__zne_on__bogus"]


def test_fixed_scaffold_saved_theta_mitigation_matrix_allows_none_base_mitigation(
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
        "_build_ansatz_circuit",
        lambda layout, theta, nq, ref_state=None: report.QuantumCircuit(int(nq)),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        lambda **kwargs: {
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
    )

    def _fake_zne_eval(**kwargs):
        mit = dict(kwargs["mitigation_config"])
        assert mit["mode"] == "none"
        assert mit["local_readout_strategy"] is None
        assert bool(mit.get("local_gate_twirling", False)) is True
        assert str(mit.get("dd_sequence")) == "XpXm"
        compile_request = report._build_compile_request_payload(
            backend_name="FakeMarrakesh",
            seed_transpiler=kwargs["seed_transpiler"],
            transpile_optimization_level=kwargs["transpile_optimization_level"],
            source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
        )
        backend_info = {
            "noise_mode": "backend_scheduled",
            "backend_name": "FakeMarrakesh",
            "details": {
                "seed_transpiler": kwargs["seed_transpiler"],
                "transpile_optimization_level": kwargs["transpile_optimization_level"],
                "compiled_two_qubit_count": 14,
                "compiled_depth": 38,
                "compiled_size": 76,
            },
        }
        return {
            "success": True,
            "zne_enabled": True,
            "zne_scales": [1.0, 3.0, 5.0],
            "per_factor_results": [{"noise_scale": 1.0, "delta_mean": 0.02}],
            "extrapolator": "linear",
            "compile_request": dict(compile_request),
            "compile_observation": report._build_compile_observation_payload(
                requested=compile_request,
                backend_info=backend_info,
            ),
            "matches_requested": True,
            "transpile_seed": kwargs["seed_transpiler"],
            "transpile_optimization_level": kwargs["transpile_optimization_level"],
            "compiled_two_qubit_count": 14,
            "compiled_depth": 38,
            "compiled_size": 76,
            "backend_info": backend_info,
            "noisy_mean": 1.01,
            "noisy_stderr": 0.01,
            "ideal_mean": 1.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.01,
            "delta_stderr": 0.01,
            "extrapolated_energy_mean": 1.01,
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy_local_zne", _fake_zne_eval)

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        compile_presets=(
            {"label": "opt2_seed0", "transpile_optimization_level": 2, "seed_transpiler": 0},
        ),
        suppression_labels=("readout_plus_gate_twirling_plus_local_dd",),
        selected_cells=("opt2_seed0__zne_on__twirl_dd",),
        mitigation_config_base={"mode": "none"},
        symmetry_mitigation_config={"mode": "off"},
    )

    assert payload["success"] is True
    assert payload["noise_config"]["mitigation_base"]["mode"] == "none"
    assert payload["cells"][0]["label"] == "opt2_seed0__zne_on__twirl_dd"
    assert payload["cells"][0]["suppression_stack"] == "gate_twirling_plus_local_dd"
    assert payload["cells"][0]["mitigation_config"]["mode"] == "none"
    assert payload["best_by_suppression_stack"]["gate_twirling_plus_local_dd"]["label"] == (
        "opt2_seed0__zne_on__twirl_dd"
    )


def test_fixed_scaffold_saved_theta_mitigation_matrix_allows_projector_symmetry(
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
        "_build_ansatz_circuit",
        lambda layout, theta, nq, ref_state=None: report.QuantumCircuit(int(nq)),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        lambda **kwargs: {
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
    )

    def _fake_zne_eval(**kwargs):
        symmetry_cfg = dict(kwargs["symmetry_mitigation_config"])
        assert symmetry_cfg["mode"] == "projector_renorm_v1"
        assert symmetry_cfg["num_sites"] == 2
        assert symmetry_cfg["sector_n_up"] == 1
        assert symmetry_cfg["sector_n_dn"] == 1
        return {
            "success": True,
            "zne_enabled": True,
            "zne_scales": [1.0, 3.0, 5.0],
            "per_factor_results": [{"noise_scale": 1.0, "delta_mean": 0.02}],
            "extrapolator": "linear",
            "compile_request": report._build_compile_request_payload(
                backend_name="FakeMarrakesh",
                seed_transpiler=0,
                transpile_optimization_level=2,
                source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
            ),
            "compile_observation": {
                "available": True,
                "requested": {},
                "observed": {},
                "matches_requested": True,
                "mismatch_fields": [],
            },
            "matches_requested": True,
            "transpile_seed": 0,
            "transpile_optimization_level": 2,
            "compiled_two_qubit_count": 14,
            "compiled_depth": 38,
            "compiled_size": 76,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "backend_name": "FakeMarrakesh",
                "details": {},
            },
            "noisy_mean": 1.02,
            "noisy_stderr": 0.01,
            "ideal_mean": 1.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.02,
            "delta_stderr": 0.01,
            "extrapolated_energy_mean": 1.02,
        }

    monkeypatch.setattr(report, "_evaluate_locked_imported_circuit_energy_local_zne", _fake_zne_eval)

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        compile_presets=(
            {"label": "opt2_seed0", "transpile_optimization_level": 2, "seed_transpiler": 0},
        ),
        suppression_labels=("readout_plus_gate_twirling_plus_local_dd",),
        selected_cells=("opt2_seed0__zne_on__twirl_dd",),
        mitigation_config_base={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={
            "mode": "projector_renorm_v1",
            "num_sites": 2,
            "ordering": "blocked",
            "sector_n_up": 1,
            "sector_n_dn": 1,
        },
    )

    assert payload["success"] is True
    assert payload["noise_config"]["symmetry_mitigation"]["mode"] == "projector_renorm_v1"
    assert payload["cells"][0]["label"] == "opt2_seed0__zne_on__twirl_dd"


def test_fixed_scaffold_saved_theta_mitigation_matrix_progress_emits_delta_payloads(
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
        "_build_ansatz_circuit",
        lambda layout, theta, nq, ref_state=None: report.QuantumCircuit(int(nq)),
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_ideal_energy",
        lambda **kwargs: {
            "ideal_mean": 1.0,
            "ideal_std": 0.0,
            "ideal_stdev": 0.0,
            "ideal_stderr": 0.0,
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_noisy_energy",
        lambda **kwargs: {
            "noisy_mean": 1.1,
            "noisy_std": 0.01,
            "noisy_stdev": 0.01,
            "noisy_stderr": 0.01,
            "backend_info": {
                "noise_mode": "backend_scheduled",
                "backend_name": "FakeMarrakesh",
                "details": {
                    "seed_transpiler": 4,
                    "transpile_optimization_level": 1,
                    "compiled_two_qubit_count": 14,
                    "compiled_depth": 48,
                    "compiled_size": 80,
                },
            },
        },
    )
    monkeypatch.setattr(
        report,
        "_evaluate_locked_imported_circuit_energy_local_zne",
        lambda **kwargs: {
            "success": True,
            "zne_enabled": True,
            "zne_scales": [1.0, 3.0, 5.0],
            "per_factor_results": [{"noise_scale": 1.0, "delta_mean": 0.09}],
            "extrapolator": "linear",
            "compile_request": report._build_compile_request_payload(
                backend_name="FakeMarrakesh",
                seed_transpiler=4,
                transpile_optimization_level=1,
                source="fixed_scaffold_saved_theta_mitigation_matrix_cell",
            ),
            "compile_observation": {
                "available": True,
                "requested": {},
                "observed": {},
                "matches_requested": True,
                "mismatch_fields": [],
            },
            "matches_requested": True,
            "transpile_seed": 4,
            "transpile_optimization_level": 1,
            "compiled_two_qubit_count": 14,
            "compiled_depth": 48,
            "compiled_size": 80,
            "backend_info": {},
            "noisy_mean": 1.09,
            "noisy_stderr": 0.02,
            "ideal_mean": 1.0,
            "ideal_stderr": 0.0,
            "delta_mean": 0.09,
            "delta_stderr": 0.02,
            "extrapolated_energy_mean": 1.09,
        },
    )

    ai_events: list[dict[str, object]] = []
    progress_events: list[dict[str, object]] = []
    monkeypatch.setattr(
        report,
        "_ai_log",
        lambda event, **fields: ai_events.append({"event": event, **fields}),
    )

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix(
        artifact_json="artifacts/json/example.json",
        shots=256,
        seed=7,
        oracle_repeats=2,
        oracle_aggregate="mean",
        backend_name="FakeMarrakesh",
        use_fake_backend=True,
        allow_aer_fallback=True,
        omp_shm_workaround=True,
        compile_presets=(
            {"label": "opt1_seed4", "transpile_optimization_level": 1, "seed_transpiler": 4},
        ),
        zne_scales=(1.0, 3.0, 5.0),
        suppression_labels=("readout_plus_gate_twirling",),
        mitigation_config_base={"mode": "readout", "local_readout_strategy": "mthree"},
        symmetry_mitigation_config={"mode": "off"},
        progress_callback=lambda msg: progress_events.append(dict(msg)),
    )

    completed_events = [
        rec for rec in progress_events if rec.get("event") == "fixed_scaffold_saved_theta_mitigation_matrix_cell_completed"
    ]
    assert payload["success"] is True
    assert len(completed_events) == 2
    assert completed_events[0]["cell"]["delta_mean"] == pytest.approx(0.1)
    assert completed_events[1]["cell"]["delta_mean"] == pytest.approx(0.09)
    assert completed_events[1]["cell"]["delta_to_ideal_mean"] == pytest.approx(0.09)
    assert completed_events[1]["cell"]["delta_to_exact_mean"] == pytest.approx(0.01)
    assert completed_events[1]["partial_payload"]["best_cell"]["delta_mean"] == pytest.approx(0.09)
    assert completed_events[1]["partial_payload"]["best_cell_by_exact_abs"]["delta_to_exact_mean"] == pytest.approx(0.01)
    done_ai = [
        rec for rec in ai_events if rec.get("event") == "fixed_scaffold_saved_theta_mitigation_matrix_cell_done"
    ]
    assert len(done_ai) == 2
    assert done_ai[0]["delta_mean"] == pytest.approx(0.1)
    assert done_ai[1]["delta_mean"] == pytest.approx(0.09)
    assert done_ai[1]["delta_to_exact_mean"] == pytest.approx(0.01)


def test_fixed_scaffold_saved_theta_mitigation_matrix_mode_isolated_timeout_preserves_partial_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeQueue:
        def __init__(self) -> None:
            self._msgs = [
                {
                    "kind": "progress",
                    "payload": {
                        "event": "fixed_scaffold_saved_theta_mitigation_matrix_cell_completed",
                        "partial_payload": {
                            "route": "fixed_scaffold_saved_theta_mitigation_matrix",
                            "cell_counts": {
                                "total": 12,
                                "completed": 2,
                                "successful": 2,
                                "failed": 0,
                            },
                            "best_cell": {
                                "label": "opt1_seed4__zne_off__dd",
                                "delta_mean": 0.08,
                            },
                            "last_cell_label": "opt1_seed4__zne_off__dd",
                            "last_cell_index": 1,
                        },
                    },
                }
            ]

        def get_nowait(self):
            if self._msgs:
                return self._msgs.pop(0)
            raise report.pyqueue.Empty()

    class _FakeProc:
        exitcode = None

        def start(self) -> None:
            return None

        def join(self, timeout: float | None = None) -> None:
            _ = timeout
            return None

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return None

    class _FakeContext:
        def Queue(self) -> _FakeQueue:
            return _FakeQueue()

        def Process(self, target, args, daemon=False) -> _FakeProc:
            assert callable(target)
            assert isinstance(args, tuple)
            assert bool(daemon) is False
            return _FakeProc()

    monotonic_values = iter([100.0, 100.0, 100.0, 100.5, 100.5, 100.5])
    monkeypatch.setattr(report.time, "monotonic", lambda: next(monotonic_values))
    monkeypatch.setattr(
        report.mp,
        "get_context",
        lambda method: (_FakeContext() if str(method) == "spawn" else None),
    )

    payload = report._run_imported_fixed_scaffold_saved_theta_mitigation_matrix_mode_isolated(
        kwargs={"artifact_json": "artifacts/json/example.json"},
        timeout_s=0,
    )

    assert payload["success"] is False
    assert payload["env_blocked"] is True
    assert payload["reason"] == "timeout_after_0s"
    assert payload["cell_counts"]["completed"] == 2
    assert payload["best_cell"]["label"] == "opt1_seed4__zne_off__dd"
    assert payload["last_cell_label"] == "opt1_seed4__zne_off__dd"
    assert payload["last_progress_event"] == "fixed_scaffold_saved_theta_mitigation_matrix_cell_completed"
