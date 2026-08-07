#!/usr/bin/env python3
"""Tests for benchmark-local external ADAPT public-code adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pipelines.exact_bench.external_adapt import external_static_adapt_benchmark as ext
from pipelines.exact_bench.external_adapt.provenance import CEO_ADAPT_VQE_PINNED_COMMIT


def test_default_external_static_adapt_case_ids_include_parameterized_tetris_hubbard_cases() -> None:
    assert ext.default_external_static_adapt_case_ids("hubbard", "static_ceo_adapt_phase3") == (
        "hubbard_L2",
    )
    assert ext.default_external_static_adapt_case_ids("hh", "static_ceo_adapt_phase3") == ()
    assert ext.default_external_static_adapt_case_ids("hubbard", "static_tetris_adapt_phase3") == (
        "hubbard_L2",
        "hubbard_L2_three_model_weak",
        "hubbard_L2_three_model_strong",
    )
    assert ext.default_external_static_adapt_case_ids("hh", "static_tetris_adapt_phase3") == ()


def test_external_adapt_python_resolution_prefers_env_then_cache_venv_then_current(monkeypatch, tmp_path: Path) -> None:
    env_python = tmp_path / "env-python"
    default_python = tmp_path / "default-python"
    for path in (env_python, default_python):
        path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        path.chmod(0o755)

    monkeypatch.setattr(ext, "_DEFAULT_CEO_VENV_PYTHON", default_python)
    monkeypatch.setenv(ext._EXTERNAL_ADAPT_PYTHON_ENV, str(env_python))
    assert ext._resolve_external_adapt_python() == (env_python, f"env:{ext._EXTERNAL_ADAPT_PYTHON_ENV}")

    monkeypatch.delenv(ext._EXTERNAL_ADAPT_PYTHON_ENV)
    assert ext._resolve_external_adapt_python() == (default_python, "default_ceo_cache_venv")

    default_python.unlink()
    assert ext._resolve_external_adapt_python() == (Path(sys.executable), "current_python")


def test_missing_ceo_checkout_writes_controlled_skip(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ext, "checkout_dir_for", lambda reference_id: tmp_path / "missing")

    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_ceo_adapt_phase3",
        output_dir=tmp_path / "out",
    )

    assert payload["status"] == "skipped_optional_dependency"
    assert "missing" in payload["reason"]
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert payload["comparator_source"]["same_suite_contract_id"].endswith("first_slice_only_v1")
    assert payload["rows"][0]["external_reference_license_status"] == "not_checked_checkout_missing"
    assert (tmp_path / "out" / "result.json").exists()
    assert (tmp_path / "out" / "rows.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert (tmp_path / "out" / "generic_static_single.json").exists()
    assert (tmp_path / "out" / "metrics_proxy_summary.json").exists()
    assert (tmp_path / "out" / "external_static_adapt_skip.json").exists()


def test_commit_mismatch_writes_provenance_skip(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "ceo_adapt_vqe"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(ext, "checkout_dir_for", lambda reference_id: checkout)
    monkeypatch.setattr(ext, "_resolved_git_commit", lambda path: "badcommit")

    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_ceo_adapt_phase3",
        output_dir=tmp_path / "out",
    )

    assert payload["status"] == "skipped_provenance_mismatch"
    assert CEO_ADAPT_VQE_PINNED_COMMIT in payload["reason"]
    assert payload["external_reference"]["resolved_commit"] is None
    assert payload["rows"][0]["phase3_controller_called"] is False


def test_worker_dependency_skip_is_normalized_by_parent(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "ceo_adapt_vqe"
    alg_dir = checkout / "adaptvqe" / "algorithms"
    alg_dir.mkdir(parents=True)
    (checkout / "adaptvqe" / "__init__.py").write_text("", encoding="utf-8")
    (alg_dir / "__init__.py").write_text("", encoding="utf-8")
    (alg_dir / "adapt_vqe.py").write_text("import openfermion_missing_for_external_skip_test\n", encoding="utf-8")
    monkeypatch.setattr(ext, "_validate_ceo_checkout", lambda: (checkout, CEO_ADAPT_VQE_PINNED_COMMIT))
    monkeypatch.setenv(ext._EXTERNAL_ADAPT_PYTHON_ENV, sys.executable)

    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_ceo_adapt_phase3",
        output_dir=tmp_path / "out",
    )

    assert payload["status"] == "skipped_optional_dependency"
    assert "openfermion_missing_for_external_skip_test" in payload["reason"]
    assert "interpreter=" in payload["reason"]
    assert payload["external_reference"]["resolved_commit"] == CEO_ADAPT_VQE_PINNED_COMMIT
    assert payload["guardrails"]["phase3_controller_called"] is False
    assert (tmp_path / "out" / "external_static_adapt_skip.json").exists()


def test_success_path_emits_normalized_external_artifacts(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "ceo_adapt_vqe"
    checkout.mkdir()
    (checkout / "LICENSE").write_text("license\n", encoding="utf-8")
    monkeypatch.setattr(ext, "_validate_ceo_checkout", lambda: (checkout, CEO_ADAPT_VQE_PINNED_COMMIT))
    monkeypatch.setattr(
        ext,
        "_run_ceo_hubbard_l2_public_code",
        lambda *, checkout_dir, case_settings: ext.ExternalAdaptRunSummary(
            energy=-1.23,
            exact_energy=-1.25,
            initial_energy=0.0,
            selected_operator_count=2,
            num_parameters=2,
            nfev=11,
            ngev=7,
            nit=5,
            adapt_iterations=2,
            adapt_success=True,
            adapt_stop_reason="converged",
            pool_name="OVP_CEO",
            pool_size=6,
            selected_indices=(0, 3),
            coefficients=(0.1, -0.2),
            gradient_norms=(0.5, 0.01),
            selected_gradients=(0.4, 0.02),
            raw_stdout_tail="ok",
            worker_mode="ceo",
            tetris_enabled=False,
            tetris_batching_enabled=False,
            operators_added_per_iteration=(1, 1),
            max_operators_added_per_iteration=1,
            batch_iterations=0,
            selected_indices_by_iteration=((0,), (0, 3)),
            worker_python="/tmp/external-python",
            worker_python_source=f"env:{ext._EXTERNAL_ADAPT_PYTHON_ENV}",
            worker_schema="ceo_public_code_worker_v1",
            worker_returncode=0,
        ),
    )

    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2",
        algorithm_id="static_ceo_adapt_phase3",
        output_dir=tmp_path / "out",
    )

    assert payload["status"] == "completed"
    assert payload["external_reference"]["resolved_commit"] == CEO_ADAPT_VQE_PINNED_COMMIT
    assert payload["external_reference"]["license_files"] == ["LICENSE"]
    assert payload["guardrails"]["uses_exact_for_decision"] is False
    row = payload["rows"][0]
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == abs(-1.23 - (-1.25))
    assert row["phase3_controller_called"] is False
    assert row["algorithm_origin"] == "external_public_code_ceo_adapt_vqe"
    assert row["execution_surface_role"] == "primary_execution_surface"
    assert row["parity_status"] == "first_slice_conformance_only_not_full_paper_i_suite"
    assert payload["comparator_source"]["external_reference_resolved_commit"] == CEO_ADAPT_VQE_PINNED_COMMIT
    assert row["worker_mode"] == "ceo"
    assert row["tetris_enabled"] is False
    assert row["hubbard_u"] == 4.0
    assert row["adapt_threshold"] == 1e-3
    assert row["adapt_max_adapt_iter"] == 6
    assert row["external_adapt_python"] == "/tmp/external-python"
    assert row["external_adapt_worker_schema"] == "ceo_public_code_worker_v1"
    assert (tmp_path / "out" / "result.json").exists()
    assert (tmp_path / "out" / "rows.json").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    assert (tmp_path / "out" / "generic_static_single.json").exists()
    assert (tmp_path / "out" / "metrics_proxy_runs.csv").exists()
    written = json.loads((tmp_path / "out" / "generic_static_single.json").read_text())
    assert written["status"] == "completed"


def test_tetris_success_path_emits_distinct_public_code_mode_telemetry(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "ceo_adapt_vqe"
    checkout.mkdir()
    monkeypatch.setattr(ext, "_validate_ceo_checkout", lambda: (checkout, CEO_ADAPT_VQE_PINNED_COMMIT))
    monkeypatch.setattr(
        ext,
        "_run_tetris_hubbard_l2_public_code",
        lambda *, checkout_dir, case_settings: ext.ExternalAdaptRunSummary(
            energy=-0.9,
            exact_energy=-1.0,
            initial_energy=0.0,
            selected_operator_count=3,
            num_parameters=3,
            nfev=13,
            ngev=9,
            nit=6,
            adapt_iterations=2,
            adapt_success=False,
            adapt_stop_reason="max_adapt_iter_6",
            pool_name="OVP_CEO",
            pool_size=6,
            selected_indices=(0, 3, 5),
            coefficients=(0.1, -0.2, 0.3),
            gradient_norms=(0.5, 0.02),
            selected_gradients=(0.4, 0.03, 0.01),
            raw_stdout_tail="tetris ok",
            worker_mode="tetris",
            tetris_enabled=True,
            tetris_batching_enabled=True,
            tetris_progressive_opt=False,
            tetris_candidate_window="full_pool_nonzero_gradient_window",
            tetris_screening_rule="disjoint_qubit_support_via_pool_get_qubits",
            operators_added_per_iteration=(2, 1),
            max_operators_added_per_iteration=2,
            batch_iterations=1,
            selected_indices_by_iteration=((0, 3), (0, 3, 5)),
            worker_python="/tmp/external-python",
            worker_python_source=f"env:{ext._EXTERNAL_ADAPT_PYTHON_ENV}",
            worker_schema="ceo_public_code_worker_v1",
            worker_returncode=0,
            external_case_profile="paper_i_hubbard_L2_three_model_weak_tetris_diagnostic",
            hubbard_x_dim=2,
            hubbard_y_dim=1,
            hubbard_t=1.0,
            hubbard_u=0.5,
            hubbard_periodic=True,
            hubbard_particle_hole_symmetry=False,
            adapt_threshold=1e-8,
            adapt_max_adapt_iter=80,
            adapt_max_opt_iter=300,
        ),
    )

    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2_three_model_weak",
        algorithm_id="static_tetris_adapt_phase3",
        output_dir=tmp_path / "out",
    )

    assert payload["status"] == "completed"
    assert payload["dispatch"] == "external_static_adapt_tetris_public_code"
    assert payload["guardrails"]["tetris_row_promoted"] is True
    assert payload["guardrails"]["phase3_controller_called"] is False
    row = payload["rows"][0]
    assert row["algorithm_origin"] == "external_public_code_ceo_adapt_vqe_tetris"
    assert row["parity_reference_algorithm_id"] == "static_tetris_qubit_adapt_vqe"
    assert payload["comparator_source"]["execution_surface"].endswith("tetris_hubbard_L2_parameterized_cases")
    assert row["worker_mode"] == "tetris"
    assert row["tetris_enabled"] is True
    assert row["tetris_batching_enabled"] is True
    assert row["operators_added_per_iteration"] == [2, 1]
    assert row["max_operators_added_per_iteration"] == 2
    assert row["batch_iterations"] == 1
    assert row["external_case_profile"] == "paper_i_hubbard_L2_three_model_weak_tetris_diagnostic"
    assert row["hubbard_u"] == 0.5
    assert row["adapt_threshold"] == 1e-8
    assert row["adapt_max_adapt_iter"] == 80
    assert payload["table_i"]["first_slice"] is False


def test_tetris_worker_call_receives_registered_case_settings(monkeypatch, tmp_path: Path) -> None:
    checkout = tmp_path / "ceo_adapt_vqe"
    checkout.mkdir()
    captured: dict[str, object] = {}
    monkeypatch.setattr(ext, "_validate_ceo_checkout", lambda: (checkout, CEO_ADAPT_VQE_PINNED_COMMIT))

    def fake_runner(*, checkout_dir, case_settings):  # noqa: ANN001
        captured["checkout_dir"] = checkout_dir
        captured["case_settings"] = case_settings
        return ext.ExternalAdaptRunSummary(
            energy=-0.9,
            exact_energy=-1.0,
            initial_energy=0.0,
            selected_operator_count=1,
            num_parameters=1,
            nfev=1,
            ngev=1,
            nit=1,
            adapt_iterations=1,
            adapt_success=False,
            adapt_stop_reason="max_adapt_iter_20",
            pool_name="OVP_CEO",
            pool_size=6,
            selected_indices=(0,),
            coefficients=(0.1,),
            gradient_norms=(0.5,),
            selected_gradients=(0.4,),
            worker_mode="tetris",
            tetris_enabled=True,
            tetris_batching_enabled=True,
            worker_schema="ceo_public_code_worker_v1",
            worker_returncode=0,
            external_case_profile=case_settings.case_profile,
            hubbard_x_dim=case_settings.x_dim,
            hubbard_y_dim=case_settings.y_dim,
            hubbard_t=case_settings.t,
            hubbard_u=case_settings.u,
            hubbard_periodic=case_settings.periodic,
            hubbard_particle_hole_symmetry=case_settings.particle_hole_symmetry,
            adapt_threshold=case_settings.threshold,
            adapt_max_adapt_iter=case_settings.max_adapt_iter,
            adapt_max_opt_iter=case_settings.max_opt_iter,
        )

    monkeypatch.setattr(ext, "_run_tetris_hubbard_l2_public_code", fake_runner)

    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2_three_model_strong",
        algorithm_id="static_tetris_adapt_phase3",
        output_dir=tmp_path / "out",
    )

    settings = captured["case_settings"]
    assert settings.u == 1.5
    assert settings.threshold == 1e-8
    assert settings.max_adapt_iter == 20
    assert payload["rows"][0]["hubbard_u"] == 1.5
    assert payload["rows"][0]["adapt_max_adapt_iter"] == 20


def test_unsupported_external_rows_remain_explicit_skip(tmp_path: Path) -> None:
    payload = ext.run_external_static_adapt_single(
        family="hubbard",
        case_id="hubbard_L2_default",
        algorithm_id="static_tetris_adapt_phase3",
        output_dir=tmp_path,
    )

    assert payload["status"] == "skipped_not_implemented"
    assert payload["guardrails"]["tetris_row_promoted"] is False
    assert payload["guardrails"]["overlap_row_promoted"] is False
    assert (tmp_path / "external_static_adapt_skip.json").exists()
