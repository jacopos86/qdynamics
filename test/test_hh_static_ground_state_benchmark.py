#!/usr/bin/env python3
"""Focused tests for the static HH ground-state benchmark runner."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pipelines.exact_bench import hh_static_ground_state_benchmark as bench


def _case(case_id: str = "hh_test_case") -> bench.HHBenchmarkCase:
    return bench.HHBenchmarkCase(
        case_id=case_id,
        num_sites=2,
        t=1.0,
        u=4.0,
        dv=0.0,
        omega0=1.0,
        g_ep=0.5,
        n_ph_max=1,
        boson_encoding="binary",
        ordering="blocked",
        boundary="open",
        include_zero_point=True,
    )


def _algorithm(
    algorithm_id: str = "hh_adapt_fake_legacy",
    adapt_pool: str = "fake_pool",
) -> bench.HHBenchmarkAlgorithmSpec:
    return bench.HHBenchmarkAlgorithmSpec(
        algorithm_id=algorithm_id,
        adapt_pool=adapt_pool,
        continuation_mode="legacy",
        max_depth=2,
        eps_grad=1.0e-5,
        eps_energy=1.0e-8,
        maxiter=20,
        seed=7,
        allow_repeats=True,
        finite_angle_fallback=True,
        finite_angle=0.1,
        finite_angle_min_improvement=1.0e-12,
        adapt_reopt_policy="full",
        paop_r=1,
        paop_split_paulis=False,
        paop_prune_eps=0.0,
        paop_normalization="none",
    )


def _conventional_algorithm(
    algorithm_id: str = "hh_hva_termwise_vqe",
    ansatz_kind: str = "termwise",
    ansatz_name: str = "hh_hva_termwise",
    display_name: str = "HH-Termwise",
) -> bench.HHBenchmarkAlgorithmSpec:
    return bench.HHBenchmarkAlgorithmSpec(
        algorithm_id=algorithm_id,
        runner_kind="conventional_vqe",
        display_name=display_name,
        ansatz_kind=ansatz_kind,
        ansatz_name=ansatz_name,
        optimizer="COBYLA",
        seed=42,
    )


def _compiled_operator_algorithm(
    algorithm_id: str = "hh_uccsd_lifted_vqe",
    operator_source: str = "hh_uccsd_lifted",
    *,
    display_name: str = "HH-UCCSD-Lifted",
    ansatz_name: str = "hh_uccsd_lifted",
) -> bench.HHBenchmarkAlgorithmSpec:
    return bench.HHBenchmarkAlgorithmSpec(
        algorithm_id=algorithm_id,
        runner_kind="compiled_operator_vqe",
        display_name=display_name,
        ansatz_name=ansatz_name,
        operator_source=operator_source,
        parameterization_mode="logical_shared",
        optimizer="COBYLA",
        seed=42,
    )


def _compiled_operator_avqite_algorithm(
    algorithm_id: str = "hh_avqite_uccsd_lifted",
    operator_source: str = "hh_uccsd_lifted",
    *,
    display_name: str = "HH-AVQITE-UCCSD-Lifted",
    ansatz_name: str = "hh_uccsd_lifted",
) -> bench.HHBenchmarkAlgorithmSpec:
    return bench.HHBenchmarkAlgorithmSpec(
        algorithm_id=algorithm_id,
        runner_kind="compiled_operator_avqite",
        display_name=display_name,
        ansatz_name=ansatz_name,
        operator_source=operator_source,
        parameterization_mode="logical_shared",
        avqite_step_size=0.1,
        avqite_max_steps=80,
        avqite_energy_tol=1e-8,
        avqite_residual_tol=1e-6,
    )


def _compiled_operator_qsci_algorithm(
    algorithm_id: str = "hh_qsci_sq_lf_std",
    operator_source: str = "hh_sq_lf_std_pool",
    *,
    display_name: str = "HH-QSCI-SQ-LF-Std",
    ansatz_name: str = "hh_qsci_sq_lf_std",
) -> bench.HHBenchmarkAlgorithmSpec:
    return bench.HHBenchmarkAlgorithmSpec(
        algorithm_id=algorithm_id,
        runner_kind="compiled_operator_qsci",
        display_name=display_name,
        ansatz_name=ansatz_name,
        operator_source=operator_source,
        basis_probe_angle=float(np.pi / 2),
        basis_amp_cutoff=1e-9,
        qsci_max_basis_states=32,
    )


def _compiled_operator_sqd_algorithm(
    algorithm_id: str = "hh_sqd_sq_lf_std",
    operator_source: str = "hh_sq_lf_std_pool",
    *,
    display_name: str = "HH-SQD-SQ-LF-Std",
    ansatz_name: str = "hh_sqd_sq_lf_std",
) -> bench.HHBenchmarkAlgorithmSpec:
    return bench.HHBenchmarkAlgorithmSpec(
        algorithm_id=algorithm_id,
        runner_kind="compiled_operator_sqd",
        display_name=display_name,
        ansatz_name=ansatz_name,
        operator_source=operator_source,
        basis_probe_angle=float(np.pi / 2),
        sqd_shots_per_probe=256,
        sqd_max_basis_states=32,
        sqd_seed=7,
    )


def _compiled_operator_lang_firsov_algorithm() -> bench.HHBenchmarkAlgorithmSpec:
    return _compiled_operator_algorithm(
        algorithm_id="hh_lang_firsov_sq_lf_vqe",
        operator_source="hh_sq_lf_std_lf_only",
        display_name="HH-LangFirsov-SQ-LF-VQE",
        ansatz_name="hh_lang_firsov_sq_lf",
    )


class _FakeExactTarget:
    kind = "exact_ground_energy_sector_hh"

    def resolve_energy(self, *, ai_log=None) -> float:  # noqa: ANN001 - mirrors production callback
        return -1.25


class _FakeReferenceState:
    def build_state(self):  # noqa: ANN201 - mirrors production callback
        return np.array([1.0 + 0.0j, 0.0 + 0.0j], dtype=complex)


class _FakeResolvedProblem:
    hamiltonian = "fake-h-poly"
    exact_target = _FakeExactTarget()
    reference_state = _FakeReferenceState()


def _patch_problem_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_resolve_problem_context(request):  # noqa: ANN001 - production signature is broad
        assert request.problem_key == "hh"
        assert request.ordering == "blocked"
        assert request.boundary == "open"
        assert request.boson_encoding == "binary"
        assert request.include_zero_point is True
        return _FakeResolvedProblem()

    monkeypatch.setattr(bench, "resolve_problem_context", _fake_resolve_problem_context)


def test_case_manifest_matches_scoreboard_contract() -> None:
    cases = bench.canonical_hh_benchmark_cases()
    assert [case.case_id for case in cases] == [
        "hh_L2_strong_canonical",
        "hh_L2_weak_diagnostic",
        "hh_L3_weak_current_success",
        "hh_L3_strong_historical_anchor",
    ]
    for case in cases:
        assert case.t == 1.0
        assert case.dv == 0.0
        assert case.omega0 == 1.0
        assert case.n_ph_max == 1
        assert case.boson_encoding == "binary"
        assert case.ordering == "blocked"
        assert case.boundary == "open"
        assert case.include_zero_point is True


def test_algorithm_manifest_without_qiskit_keeps_native_inventory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "has_qiskit_hea_support", lambda: False)

    algorithms = bench.default_hh_benchmark_algorithms()
    assert len(algorithms) == 17
    assert [algorithm.algorithm_id for algorithm in algorithms] == [
        "hh_adapt_full_hamiltonian_legacy",
        "hh_adapt_hva_legacy",
        "hh_adapt_paop_lf_std_legacy",
        "hh_adapt_qeb_sq_lf_std_legacy",
        "hh_adapt_overlap_paop_lf_std_phase3",
        "hh_adapt_ceo_paop_lf_std_phase3",
        "hh_adapt_uccsd_otimes_paop_lf_std_legacy",
        "hh_adapt_full_meta_legacy",
        "hh_adapt_pareto_lean_legacy",
        "hh_hva_termwise_vqe",
        "hh_hva_layerwise_vqe",
        "hh_uccsd_lifted_vqe",
        "hh_avqite_uccsd_lifted",
        "hh_qsci_sq_lf_std",
        "hh_sqd_sq_lf_std",
        "hh_puccd_lifted_vqe",
        "hh_lang_firsov_sq_lf_vqe",
    ]
    assert [algorithm.runner_kind for algorithm in algorithms] == [
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "adapt_vqe",
        "conventional_vqe",
        "conventional_vqe",
        "compiled_operator_vqe",
        "compiled_operator_avqite",
        "compiled_operator_qsci",
        "compiled_operator_sqd",
        "compiled_operator_vqe",
        "compiled_operator_vqe",
    ]
    assert [algorithm.adapt_pool for algorithm in algorithms[:9]] == [
        "full_hamiltonian",
        "hva",
        "paop_lf_std",
        "sq_lf_std",
        "paop_lf_std",
        "paop_lf_std",
        "uccsd_otimes_paop_lf_std",
        "full_meta",
        "pareto_lean",
    ]
    assert [algorithm.ansatz_name for algorithm in algorithms[-8:]] == [
        "hh_hva_termwise",
        "hh_hva_layerwise",
        "hh_uccsd_lifted",
        "hh_uccsd_lifted",
        "hh_qsci_sq_lf_std",
        "hh_sqd_sq_lf_std",
        "hh_puccd_lifted",
        "hh_lang_firsov_sq_lf",
    ]
    assert [algorithm.seed for algorithm in algorithms[-8:]] == [42, 42, 42, 7, 7, 7, 42, 42]
    uccsd_lifted_row = algorithms[-6]
    assert uccsd_lifted_row.operator_source == "hh_uccsd_lifted"
    assert uccsd_lifted_row.parameterization_mode == "logical_shared"
    avqite_row = algorithms[-5]
    assert avqite_row.algorithm_id == "hh_avqite_uccsd_lifted"
    assert avqite_row.runner_kind == "compiled_operator_avqite"
    assert avqite_row.operator_source == "hh_uccsd_lifted"
    assert avqite_row.parameterization_mode == "logical_shared"
    assert avqite_row.avqite_step_size == 0.1
    assert avqite_row.avqite_max_steps == 80
    assert avqite_row.avqite_energy_tol == 1e-8
    assert avqite_row.avqite_residual_tol == 1e-6
    qsci_row = algorithms[-4]
    assert qsci_row.algorithm_id == "hh_qsci_sq_lf_std"
    assert qsci_row.runner_kind == "compiled_operator_qsci"
    assert qsci_row.operator_source == "hh_sq_lf_std_pool"
    assert qsci_row.ansatz_name == "hh_qsci_sq_lf_std"
    assert qsci_row.basis_probe_angle == pytest.approx(float(np.pi / 2))
    assert qsci_row.basis_amp_cutoff == 1e-9
    assert qsci_row.qsci_max_basis_states == 32
    sqd_row = algorithms[-3]
    assert sqd_row.algorithm_id == "hh_sqd_sq_lf_std"
    assert sqd_row.runner_kind == "compiled_operator_sqd"
    assert sqd_row.operator_source == "hh_sq_lf_std_pool"
    assert sqd_row.ansatz_name == "hh_sqd_sq_lf_std"
    assert sqd_row.basis_probe_angle == pytest.approx(float(np.pi / 2))
    assert sqd_row.sqd_shots_per_probe == 256
    assert sqd_row.sqd_max_basis_states == 32
    assert sqd_row.sqd_seed == 7
    puccd_lifted_row = algorithms[-2]
    assert puccd_lifted_row.runner_kind == "compiled_operator_vqe"
    assert puccd_lifted_row.operator_source == "hh_puccd_lifted"
    assert puccd_lifted_row.parameterization_mode == "logical_shared"
    lang_firsov_row = algorithms[-1]
    assert lang_firsov_row.algorithm_id == "hh_lang_firsov_sq_lf_vqe"
    assert lang_firsov_row.runner_kind == "compiled_operator_vqe"
    assert lang_firsov_row.display_name == "HH-LangFirsov-SQ-LF-VQE"
    assert lang_firsov_row.ansatz_name == "hh_lang_firsov_sq_lf"
    assert lang_firsov_row.operator_source == "hh_sq_lf_std_lf_only"
    assert lang_firsov_row.parameterization_mode == "logical_shared"
    qeb_row = algorithms[3]
    assert qeb_row.runner_kind == "adapt_vqe"
    assert qeb_row.adapt_pool == "sq_lf_std"
    assert qeb_row.continuation_mode == "legacy"
    assert qeb_row.phase2_batch_selection_mode == ""
    assert qeb_row.adapt_reopt_policy == "full"
    overlap_row = algorithms[4]
    assert overlap_row.runner_kind == "adapt_vqe"
    assert overlap_row.adapt_pool == "paop_lf_std"
    assert overlap_row.continuation_mode == "phase3_v1"
    assert overlap_row.phase2_batch_selection_mode == "overlap_orthogonal_benchmark"
    ceo_row = algorithms[5]
    assert ceo_row.runner_kind == "adapt_vqe"
    assert ceo_row.adapt_pool == "paop_lf_std"
    assert ceo_row.continuation_mode == "phase3_v1"
    assert ceo_row.phase2_batch_selection_mode == "ceo_commuting_benchmark"
    uccsd_row = algorithms[6]
    assert uccsd_row.runner_kind == "adapt_vqe"
    assert uccsd_row.adapt_pool == "uccsd_otimes_paop_lf_std"
    assert uccsd_row.continuation_mode == "legacy"
    legacy_adapt_rows = [*algorithms[:4], *algorithms[6:9]]
    assert all(algorithm.continuation_mode == "legacy" for algorithm in legacy_adapt_rows)
    assert all(algorithm.adapt_reopt_policy == "full" for algorithm in algorithms[:9])


def test_algorithm_manifest_with_qiskit_appends_hea_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bench, "has_qiskit_hea_support", lambda: True)

    algorithms = bench.default_hh_benchmark_algorithms()
    assert len(algorithms) == 18
    assert [algorithm.algorithm_id for algorithm in algorithms[-9:]] == [
        "hh_hva_termwise_vqe",
        "hh_hva_layerwise_vqe",
        "hh_uccsd_lifted_vqe",
        "hh_avqite_uccsd_lifted",
        "hh_qsci_sq_lf_std",
        "hh_sqd_sq_lf_std",
        "hh_puccd_lifted_vqe",
        "hh_lang_firsov_sq_lf_vqe",
        "hh_hea_qiskit_vqe",
    ]
    hea = algorithms[-1]
    assert hea.runner_kind == "conventional_vqe"
    assert hea.display_name == "HH-HEA-Qiskit"
    assert hea.ansatz_kind == "qiskit_hea"
    assert hea.ansatz_name == "hh_hea_qiskit"
    assert hea.vqe_reps == 2
    assert hea.optimizer == "COBYLA"
    assert hea.seed == 42


def test_hh_static_benchmark_decision_noise_rejects_unsupported_direct_hh_algorithm(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="unsupported HH algorithm_ids"):
        bench.run_hh_static_ground_state_benchmark(
            output_dir=tmp_path,
            cases=[_case("hh_L2_strong_canonical")],
            algorithms=[
                _compiled_operator_algorithm(
                    "hh_puccd_lifted_vqe",
                    "hh_puccd_lifted",
                    display_name="HH-pUCCD-Lifted",
                    ansatz_name="hh_puccd_lifted",
                )
            ],
            benchmark_decision_noise_config={
                "enabled": True,
                "model": "gaussian_iid_v1",
                "std": 0.25,
                "seed": 11,
            },
        )

    assert not (tmp_path / "hh_static_benchmark_manifest.json").exists()


class _FakeOperatorTerm:
    def __init__(self, label: str):
        self.label = label
        self.polynomial = object()


def test_hh_puccd_lifted_operator_source_filters_paired_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed_terms = [
        _FakeOperatorTerm("uccsd_ferm_lifted::uccsd_sing(alpha:0->1)"),
        _FakeOperatorTerm("uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,2)"),
        _FakeOperatorTerm("uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,3)"),
    ]
    captured_builder: list[dict[str, object]] = []

    def _fake_build_pool(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors production builder
        captured_builder.append({"args": args, "kwargs": kwargs})
        assert args == (2, 1, "binary", "blocked", "open")
        assert kwargs["num_particles"] == (1, 1)
        return mixed_terms

    monkeypatch.setattr(bench, "_build_hh_uccsd_fermion_lifted_pool", _fake_build_pool)

    selected = bench._build_hh_puccd_lifted_terms(case=_case("hh_L2_strong_canonical"))

    assert selected == [mixed_terms[2]]
    assert len(captured_builder) == 1

    def _fake_empty_build_pool(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors production builder
        return mixed_terms[:2]

    monkeypatch.setattr(bench, "_build_hh_uccsd_fermion_lifted_pool", _fake_empty_build_pool)
    with pytest.raises(ValueError, match="selected zero paired-double"):
        bench._build_hh_puccd_lifted_terms(case=_case("hh_L2_strong_canonical"))


def test_hh_sq_lf_std_lf_only_operator_source_filters_stable_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mixed_terms = [
        _FakeOperatorTerm("sq_lf_std:lf_disp(site=0)"),
        _FakeOperatorTerm("sq_lf_std:sq(site=0)"),
        _FakeOperatorTerm("sq_lf_std:dens_sq(site=0)"),
        _FakeOperatorTerm("sq_lf_std:lf_disp(site=1)[0]_exyz"),
    ]
    captured_plan_kwargs: list[dict[str, object]] = []

    def _fake_resolve_pool_plan(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_plan_kwargs.append(dict(kwargs))
        assert kwargs["adapt_pool"] == "sq_lf_std"
        assert kwargs["continuation_mode"] == "legacy"
        assert kwargs["phase3_symmetry_mitigation_mode"] == "off"
        return type("_FakePoolPlan", (), {"pool": mixed_terms})()

    monkeypatch.setattr(bench, "resolve_pool_plan", _fake_resolve_pool_plan)

    selected = bench._build_hh_sq_lf_std_lf_only_terms(
        case=_case("hh_L2_strong_canonical"),
        algorithm=_compiled_operator_lang_firsov_algorithm(),
        resolved_problem=_FakeResolvedProblem(),
    )

    assert selected == [mixed_terms[0], mixed_terms[3]]
    assert len(captured_plan_kwargs) == 1
    assert all(str(term.label).startswith("sq_lf_std:lf_disp(") for term in selected)
    assert not any(str(term.label).startswith("sq_lf_std:sq(") for term in selected)
    assert not any(str(term.label).startswith("sq_lf_std:dens_sq(") for term in selected)

    def _fake_squeeze_only_plan(**kwargs):  # noqa: ANN003 - production signature is broad
        return type("_FakePoolPlan", (), {"pool": mixed_terms[1:3]})()

    monkeypatch.setattr(bench, "resolve_pool_plan", _fake_squeeze_only_plan)
    with pytest.raises(ValueError, match="zero LF displacement operators"):
        bench._build_hh_sq_lf_std_lf_only_terms(
            case=_case("hh_L2_strong_canonical"),
            algorithm=_compiled_operator_lang_firsov_algorithm(),
            resolved_problem=_FakeResolvedProblem(),
        )


def test_mocked_success_writes_rows_and_proxy_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run_adapt(**kwargs):  # noqa: ANN003 - production signature is large
        captured_kwargs.append(dict(kwargs))
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs_override"] == -1.25
        assert kwargs["boundary"] == "open"
        return (
            {
                "success": True,
                "energy": -1.0,
                "exact_gs_energy": -1.25,
                "abs_delta_e": 0.25,
                "nfev_total": 12,
                "num_parameters": 2,
                "ansatz_depth": 2,
                "pool_type": kwargs["adapt_pool"],
                "stop_reason": "max_depth",
            },
            object(),
        )

    monkeypatch.setattr(
        bench,
        "_run_hardcoded_adapt_vqe_compatibility",
        _fake_run_adapt,
    )
    monkeypatch.setattr(
        bench,
        "_case_reference_energy_audit",
        lambda **kwargs: {
            "reference_energy_status": "ok",
            "reference_state_energy": -0.75,
            "reference_abs_delta_e": 0.5,
            "reference_state_source": "test",
        },
    )

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_algorithm("hh_adapt_fake_legacy", "fake_pool")],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_adapt_fake_legacy"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["status"] == "ok"
    assert row["quality_status"] == "ok_large_error"
    assert row["algorithm_spec_optimizer"] == "COBYLA"
    assert row["configured_optimizer"] == "POWELL"
    assert row["actual_optimizer"] == "POWELL"
    assert row["delta_E_abs"] == 0.25
    assert row["abs_delta_e"] == 0.25
    assert row["nfev"] == 12
    assert row["reference_energy_status"] == "ok"
    assert row["reference_state_energy"] == -0.75
    assert row["reference_abs_delta_e"] == 0.5
    assert row["improvement_over_reference_abs_delta_e"] == pytest.approx(0.25)
    assert row["beats_reference_state"] is True
    assert row["benchmark_audit_flags"] == ["large_energy_error_gt_0p1"]
    assert row["adapt_depth_reached"] == 2
    assert row["adapt_stop_reason"] == "max_depth"

    rows_json = json.loads((tmp_path / "hh_static_benchmark_rows.json").read_text(encoding="utf-8"))
    assert rows_json == result["rows"]
    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert artifact_payload["benchmark_reference_energy_audit"]["reference_energy_status"] == "ok"

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert proxy_rows[0]["delta_E_abs"] == "0.25"
    assert proxy_rows[0]["nfev"] == "12"
    assert proxy_rows[0]["adapt_depth_reached"] == "2"
    assert proxy_rows[0]["adapt_stop_reason"] == "max_depth"

    summary = json.loads((tmp_path / "summary" / "metrics_proxy_summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "hh_bench_metrics_v5"
    assert summary["benchmark_schema"] == bench.RUNNER_SCHEMA_VERSION
    assert summary["row_count"] == 1
    assert summary["status_counts"] == {"ok": 1}


def test_static_benchmark_audit_flags_reference_and_optimizer_pathologies() -> None:
    assert bench._quality_status_from_flags(status="failed", flags=[]) == "failed"
    assert (
        bench._quality_status_from_flags(
            status="ok",
            flags=["does_not_improve_reference_state", "large_energy_error_gt_0p1"],
        )
        == "ok_reference_not_improved"
    )
    assert (
        bench._quality_status_from_flags(status="ok", flags=["optimizer_not_converged"])
        == "ok_optimizer_suspect"
    )

    flags = bench._benchmark_audit_flags(
        payload={"success": True, "energy": -1.0, "optimizer_success": False, "converged": False},
        runner_kind="conventional_vqe",
        abs_delta_e=0.6,
        exact_energy=-1.25,
        reference_abs_delta_e=0.5,
    )

    assert flags == [
        "does_not_improve_reference_state",
        "large_energy_error_gt_0p1",
        "not_converged",
        "optimizer_not_converged",
    ]


def test_mocked_uccsd_product_success_maps_adapt_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)

    def _fake_run_adapt(**kwargs):  # noqa: ANN003 - production signature is large
        return (
            {
                "success": True,
                "energy": -1.20,
                "exact_gs_energy": -1.25,
                "abs_delta_e": 0.05,
                "nfev_total": 17,
                "num_parameters": 3,
                "ansatz_depth": 3,
                "pool_type": kwargs["adapt_pool"],
                "stop_reason": "eps_energy",
            },
            object(),
        )

    monkeypatch.setattr(
        bench,
        "_run_hardcoded_adapt_vqe_compatibility",
        _fake_run_adapt,
    )

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[
            _algorithm(
                "hh_adapt_uccsd_otimes_paop_lf_std_legacy",
                "uccsd_otimes_paop_lf_std",
            )
        ],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    row = result["rows"][0]
    assert row["method_id"] == "hh_adapt_uccsd_otimes_paop_lf_std_legacy"
    assert row["method_kind"] == "adapt_vqe"
    assert row["pool_name"] == "uccsd_otimes_paop_lf_std"
    assert row["adapt_pool"] == "uccsd_otimes_paop_lf_std"
    assert row["continuation_mode"] == "legacy"
    assert row["delta_E_abs"] == 0.05
    assert row["adapt_depth_reached"] == 3
    assert row["adapt_stop_reason"] == "eps_energy"


def test_mocked_qeb_sq_lf_std_success_maps_adapt_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    monkeypatch.setattr(bench, "has_qiskit_hea_support", lambda: False)
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run_adapt(**kwargs):  # noqa: ANN003 - production signature is large
        captured_kwargs.append(dict(kwargs))
        return (
            {
                "success": True,
                "energy": -1.21,
                "exact_gs_energy": -1.25,
                "abs_delta_e": 0.04,
                "nfev_total": 18,
                "num_parameters": 2,
                "ansatz_depth": 2,
                "pool_type": kwargs["adapt_pool"],
                "stop_reason": "eps_grad",
            },
            object(),
        )

    monkeypatch.setattr(
        bench,
        "_run_hardcoded_adapt_vqe_compatibility",
        _fake_run_adapt,
    )
    algorithms = [
        algorithm
        for algorithm in bench.default_hh_benchmark_algorithms()
        if algorithm.algorithm_id == "hh_adapt_qeb_sq_lf_std_legacy"
    ]

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=algorithms,
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["adapt_pool"] == "sq_lf_std"
    assert captured_kwargs[0]["adapt_continuation_mode"] == "legacy"
    assert captured_kwargs[0]["phase3_backend_cost_mode"] == "proxy"
    assert "phase2_batch_selection_mode" not in captured_kwargs[0]
    row = result["rows"][0]
    assert row["method_id"] == "hh_adapt_qeb_sq_lf_std_legacy"
    assert row["method_kind"] == "adapt_vqe"
    assert row["pool_name"] == "sq_lf_std"
    assert row["adapt_pool"] == "sq_lf_std"
    assert row["continuation_mode"] == "legacy"
    assert row["delta_E_abs"] == 0.04
    assert row["adapt_depth_reached"] == 2
    assert row["adapt_stop_reason"] == "eps_grad"
    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_algorithm"]["algorithm_id"] == "hh_adapt_qeb_sq_lf_std_legacy"
    assert artifact_payload["benchmark_algorithm"]["adapt_pool"] == "sq_lf_std"
    assert artifact_payload["benchmark_algorithm"]["continuation_mode"] == "legacy"


def test_mocked_overlap_phase3_row_threads_batch_selection_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    monkeypatch.setattr(bench, "has_qiskit_hea_support", lambda: False)
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run_adapt(**kwargs):  # noqa: ANN003 - production signature is large
        captured_kwargs.append(dict(kwargs))
        return (
            {
                "success": True,
                "energy": -1.24,
                "exact_gs_energy": -1.25,
                "abs_delta_e": 0.01,
                "nfev_total": 22,
                "num_parameters": 5,
                "ansatz_depth": 5,
                "pool_type": kwargs["adapt_pool"],
                "stop_reason": "max_depth",
            },
            object(),
        )

    monkeypatch.setattr(
        bench,
        "_run_hardcoded_adapt_vqe_compatibility",
        _fake_run_adapt,
    )
    algorithms = [
        algorithm
        for algorithm in bench.default_hh_benchmark_algorithms()
        if algorithm.algorithm_id == "hh_adapt_overlap_paop_lf_std_phase3"
    ]

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=algorithms,
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["adapt_pool"] == "paop_lf_std"
    assert captured_kwargs[0]["adapt_continuation_mode"] == "phase3_v1"
    assert captured_kwargs[0]["phase3_backend_cost_mode"] == "proxy"
    assert captured_kwargs[0]["phase2_batch_selection_mode"] == "overlap_orthogonal_benchmark"
    row = result["rows"][0]
    assert row["method_id"] == "hh_adapt_overlap_paop_lf_std_phase3"
    assert row["continuation_mode"] == "phase3_v1"
    assert row["adapt_pool"] == "paop_lf_std"
    assert row["delta_E_abs"] == 0.01
    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_algorithm"]["phase2_batch_selection_mode"] == "overlap_orthogonal_benchmark"
    assert artifact_payload["benchmark_algorithm"]["continuation_mode"] == "phase3_v1"


def test_mocked_ceo_phase3_row_threads_batch_selection_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    monkeypatch.setattr(bench, "has_qiskit_hea_support", lambda: False)
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run_adapt(**kwargs):  # noqa: ANN003 - production signature is large
        captured_kwargs.append(dict(kwargs))
        return (
            {
                "success": True,
                "energy": -1.245,
                "exact_gs_energy": -1.25,
                "abs_delta_e": 0.005,
                "nfev_total": 23,
                "num_parameters": 6,
                "ansatz_depth": 6,
                "pool_type": kwargs["adapt_pool"],
                "stop_reason": "max_depth",
            },
            object(),
        )

    monkeypatch.setattr(
        bench,
        "_run_hardcoded_adapt_vqe_compatibility",
        _fake_run_adapt,
    )
    algorithms = [
        algorithm
        for algorithm in bench.default_hh_benchmark_algorithms()
        if algorithm.algorithm_id == "hh_adapt_ceo_paop_lf_std_phase3"
    ]

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=algorithms,
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["adapt_pool"] == "paop_lf_std"
    assert captured_kwargs[0]["adapt_continuation_mode"] == "phase3_v1"
    assert captured_kwargs[0]["phase3_backend_cost_mode"] == "proxy"
    assert captured_kwargs[0]["phase2_batch_selection_mode"] == "ceo_commuting_benchmark"
    row = result["rows"][0]
    assert row["method_id"] == "hh_adapt_ceo_paop_lf_std_phase3"
    assert row["continuation_mode"] == "phase3_v1"
    assert row["adapt_pool"] == "paop_lf_std"
    assert row["delta_E_abs"] == 0.005
    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_algorithm"]["phase2_batch_selection_mode"] == "ceo_commuting_benchmark"
    assert artifact_payload["benchmark_algorithm"]["continuation_mode"] == "phase3_v1"


def test_mocked_conventional_success_writes_rows_manifest_and_artifact_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run_conventional(**kwargs):  # noqa: ANN003 - production signature is large
        captured_kwargs.append(dict(kwargs))
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["ansatz_kind"] == "termwise"
        assert kwargs["boundary"] == "open"
        assert kwargs["ordering"] == "blocked"
        assert kwargs["boson_encoding"] == "binary"
        assert kwargs["seed"] == 42
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-Termwise",
            "ansatz_name": "hh_hva_termwise",
            "energy": -1.10,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.15,
            "delta_E_abs": 0.15,
            "nfev": 33,
            "nit": 4,
            "num_parameters": 6,
            "vqe_reps_used": 2,
            "vqe_restarts": 3,
            "vqe_maxiter_used": 800,
            "optimizer": "COBYLA",
            "optimizer_success": False,
            "optimizer_message": "mocked maxiter",
            "converged": False,
        }

    monkeypatch.setattr(bench, "run_hh_conventional_vqe_trial", _fake_run_conventional)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_conventional_algorithm()],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_hva_termwise_vqe"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "conventional_vqe"
    assert row["ansatz_name"] == "hh_hva_termwise"
    assert row["pool_name"] == ""
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.15
    assert row["abs_delta_e"] == 0.15
    assert row["nfev"] == 33
    assert row["nit"] == 4
    assert row["num_parameters"] == 6
    assert row["vqe_reps"] == 2
    assert row["vqe_restarts"] == 3
    assert row["vqe_maxiter"] == 800
    assert row["optimizer_success"] is False
    assert row["optimizer_message"] == "mocked maxiter"
    assert row["converged"] is False

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "conventional_vqe"
    assert manifest["algorithms"][0]["ansatz_name"] == "hh_hva_termwise"

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["optimizer_success"] is False
    assert artifact_payload["optimizer_message"] == "mocked maxiter"
    assert artifact_payload["converged"] is False

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "conventional_vqe"
    assert proxy_rows[0]["ansatz_name"] == "hh_hva_termwise"
    assert proxy_rows[0]["vqe_reps"] == "2"
    assert proxy_rows[0]["vqe_restarts"] == "3"
    assert proxy_rows[0]["vqe_maxiter"] == "800"
    assert proxy_rows[0]["operator_family_proxy"] == "conventional_vqe+hva"


def test_mocked_qiskit_hea_success_maps_conventional_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    captured_kwargs: list[dict[str, object]] = []

    def _fake_run_conventional(**kwargs):  # noqa: ANN003 - production signature is large
        captured_kwargs.append(dict(kwargs))
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["ansatz_kind"] == "qiskit_hea"
        assert kwargs["reps"] == 2
        assert kwargs["boundary"] == "open"
        assert kwargs["ordering"] == "blocked"
        assert kwargs["boson_encoding"] == "binary"
        assert kwargs["seed"] == 42
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-HEA-Qiskit",
            "ansatz_name": "hh_hea_qiskit",
            "energy": -1.18,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.07,
            "delta_E_abs": 0.07,
            "nfev": 55,
            "nit": 6,
            "num_parameters": 12,
            "vqe_reps": 2,
            "vqe_restarts": 3,
            "vqe_maxiter": 800,
            "optimizer": "COBYLA",
            "optimizer_success": True,
            "optimizer_message": "mocked ok",
            "converged": True,
        }

    monkeypatch.setattr(bench, "run_hh_conventional_vqe_trial", _fake_run_conventional)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[
            bench.HHBenchmarkAlgorithmSpec(
                algorithm_id="hh_hea_qiskit_vqe",
                runner_kind="conventional_vqe",
                display_name="HH-HEA-Qiskit",
                ansatz_kind="qiskit_hea",
                ansatz_name="hh_hea_qiskit",
                vqe_reps=2,
                optimizer="COBYLA",
                seed=42,
            )
        ],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["method_id"] == "hh_hea_qiskit_vqe"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "conventional_vqe"
    assert row["display_name"] == "HH-HEA-Qiskit"
    assert row["ansatz_name"] == "hh_hea_qiskit"
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.07
    assert row["nfev"] == 55
    assert row["num_parameters"] == 12
    assert row["vqe_reps"] == 2
    assert row["vqe_restarts"] == 3
    assert row["vqe_maxiter"] == 800
    assert row["optimizer_success"] is True
    assert row["optimizer_message"] == "mocked ok"
    assert row["converged"] is True

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["algorithm_id"] == "hh_hea_qiskit_vqe"
    assert manifest["algorithms"][0]["ansatz_kind"] == "qiskit_hea"
    assert manifest["algorithms"][0]["vqe_reps"] == 2

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert artifact_payload["ansatz_name"] == "hh_hea_qiskit"
    assert artifact_payload["vqe_reps"] == 2

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "conventional_vqe"
    assert proxy_rows[0]["ansatz_name"] == "hh_hea_qiskit"
    assert proxy_rows[0]["vqe_reps"] == "2"
    assert proxy_rows[0]["vqe_restarts"] == "3"
    assert proxy_rows[0]["vqe_maxiter"] == "800"



def test_mocked_compiled_operator_success_maps_conventional_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    fake_terms = [object(), object(), object()]
    captured_builder: list[dict[str, object]] = []
    captured_kwargs: list[dict[str, object]] = []

    def _fake_build_pool(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors production builder
        captured_builder.append({"args": args, "kwargs": kwargs})
        assert args == (2, 1, "binary", "blocked", "open")
        assert kwargs["num_particles"] == (1, 1)
        return fake_terms

    def _fake_run_compiled(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_kwargs.append(dict(kwargs))
        assert kwargs["operator_terms"] == fake_terms
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["ansatz_name"] == "hh_uccsd_lifted"
        assert kwargs["display_name"] == "HH-UCCSD-Lifted"
        assert kwargs["parameterization_mode"] == "logical_shared"
        assert kwargs["seed"] == 42
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-UCCSD-Lifted",
            "ansatz_name": "hh_uccsd_lifted",
            "energy": -1.23,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.02,
            "delta_E_abs": 0.02,
            "nfev": 44,
            "nit": 5,
            "num_parameters": 3,
            "runtime_parameter_count": 9,
            "vqe_reps": None,
            "vqe_restarts": 3,
            "vqe_maxiter": 800,
            "optimizer": "COBYLA",
            "optimizer_success": True,
            "optimizer_message": "mocked ok",
            "converged": True,
            "parameterization_mode": "logical_shared",
            "selected_operator_labels": ["op0", "op1", "op2"],
            "selected_operator_count": 3,
        }

    monkeypatch.setattr(bench, "_build_hh_uccsd_fermion_lifted_pool", _fake_build_pool)
    monkeypatch.setattr(bench, "run_compiled_operator_vqe_trial", _fake_run_compiled)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_compiled_operator_algorithm()],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_builder) == 1
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_uccsd_lifted_vqe"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "conventional_vqe"
    assert row["ansatz_name"] == "hh_uccsd_lifted"
    assert row["pool_name"] == ""
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.02
    assert row["vqe_reps"] is None
    assert row["vqe_restarts"] == 3
    assert row["vqe_maxiter"] == 800
    assert row["selected_operator_count"] == 3
    assert row["adapt_depth_reached"] is None
    assert row["adapt_stop_reason"] == ""
    assert row["continuation_mode"] == ""
    assert row["adapt_pool"] == ""

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "compiled_operator_vqe"
    assert manifest["algorithms"][0]["operator_source"] == "hh_uccsd_lifted"
    assert manifest["algorithms"][0]["parameterization_mode"] == "logical_shared"

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["selected_operator_count"] == 3
    assert artifact_payload["parameterization_mode"] == "logical_shared"
    assert artifact_payload["runtime_parameter_count"] == 9

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "conventional_vqe"
    assert proxy_rows[0]["ansatz_name"] == "hh_uccsd_lifted"
    assert proxy_rows[0]["vqe_reps"] == ""
    assert proxy_rows[0]["vqe_restarts"] == "3"
    assert proxy_rows[0]["vqe_maxiter"] == "800"
    assert proxy_rows[0]["selected_operator_count"] == "3"
    assert proxy_rows[0]["operator_family_proxy"] == "conventional_vqe+uccsd"


def test_mocked_lang_firsov_sq_lf_compiled_operator_row_preserves_pool_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    mixed_terms = [
        _FakeOperatorTerm("sq_lf_std:lf_disp(site=0)"),
        _FakeOperatorTerm("sq_lf_std:sq(site=0)"),
        _FakeOperatorTerm("sq_lf_std:dens_sq(site=0)"),
        _FakeOperatorTerm("sq_lf_std:lf_disp(site=1)"),
    ]
    expected_terms = [mixed_terms[0], mixed_terms[3]]
    captured_plan_kwargs: list[dict[str, object]] = []
    captured_kwargs: list[dict[str, object]] = []

    def _fake_resolve_pool_plan(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_plan_kwargs.append(dict(kwargs))
        assert kwargs["adapt_pool"] == "sq_lf_std"
        assert kwargs["continuation_mode"] == "legacy"
        assert kwargs["paop_r"] == 1
        assert kwargs["paop_split_paulis"] is False
        assert kwargs["paop_prune_eps"] == 0.0
        assert kwargs["paop_normalization"] == "none"
        assert kwargs["phase3_symmetry_mitigation_mode"] == "off"
        return type("_FakePoolPlan", (), {"pool": mixed_terms})()

    def _fake_run_compiled(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_kwargs.append(dict(kwargs))
        assert kwargs["operator_terms"] == expected_terms
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["ansatz_name"] == "hh_lang_firsov_sq_lf"
        assert kwargs["display_name"] == "HH-LangFirsov-SQ-LF-VQE"
        assert kwargs["parameterization_mode"] == "logical_shared"
        assert kwargs["seed"] == 42
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-LangFirsov-SQ-LF-VQE",
            "ansatz_name": "hh_lang_firsov_sq_lf",
            "energy": -1.24,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.01,
            "delta_E_abs": 0.01,
            "nfev": 31,
            "nit": 4,
            "num_parameters": 2,
            "runtime_parameter_count": 2,
            "vqe_reps": None,
            "vqe_restarts": 3,
            "vqe_maxiter": 800,
            "optimizer": "COBYLA",
            "optimizer_success": True,
            "optimizer_message": "mocked ok",
            "converged": True,
            "parameterization_mode": "logical_shared",
            "selected_operator_labels": [term.label for term in expected_terms],
            "selected_operator_count": 2,
        }

    monkeypatch.setattr(bench, "resolve_pool_plan", _fake_resolve_pool_plan)
    monkeypatch.setattr(bench, "run_compiled_operator_vqe_trial", _fake_run_compiled)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_compiled_operator_lang_firsov_algorithm()],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_plan_kwargs) == 1
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_lang_firsov_sq_lf_vqe"
    assert row["method_id"] == "hh_lang_firsov_sq_lf_vqe"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "conventional_vqe"
    assert row["display_name"] == "HH-LangFirsov-SQ-LF-VQE"
    assert row["ansatz_name"] == "hh_lang_firsov_sq_lf"
    assert row["pool_name"] == "sq_lf_std"
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.01
    assert row["nfev"] == 31
    assert row["num_parameters"] == 2
    assert row["selected_operator_count"] == 2
    assert row["vqe_reps"] is None
    assert row["vqe_restarts"] == 3
    assert row["vqe_maxiter"] == 800
    assert row["adapt_depth_reached"] is None
    assert row["adapt_stop_reason"] == ""
    assert row["continuation_mode"] == ""
    assert row["adapt_pool"] == ""

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "compiled_operator_vqe"
    assert manifest["algorithms"][0]["operator_source"] == "hh_sq_lf_std_lf_only"
    assert manifest["algorithms"][0]["ansatz_name"] == "hh_lang_firsov_sq_lf"
    assert manifest["algorithms"][0]["parameterization_mode"] == "logical_shared"

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["selected_operator_count"] == 2
    assert artifact_payload["selected_operator_labels"] == [term.label for term in expected_terms]
    assert artifact_payload["parameterization_mode"] == "logical_shared"

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "conventional_vqe"
    assert proxy_rows[0]["ansatz_name"] == "hh_lang_firsov_sq_lf"
    assert proxy_rows[0]["pool_name"] == "sq_lf_std"
    assert proxy_rows[0]["num_parameters"] == "2"
    assert proxy_rows[0]["selected_operator_count"] == "2"
    assert proxy_rows[0]["pool_family_proxy"] == "sq_lf_std"


def test_mocked_avqite_compiled_operator_success_maps_avqite_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    fake_terms = [object(), object(), object()]
    captured_builder: list[dict[str, object]] = []
    captured_kwargs: list[dict[str, object]] = []

    def _fake_build_pool(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors production builder
        captured_builder.append({"args": args, "kwargs": kwargs})
        assert args == (2, 1, "binary", "blocked", "open")
        assert kwargs["num_particles"] == (1, 1)
        return fake_terms

    def _fake_run_avqite(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_kwargs.append(dict(kwargs))
        assert kwargs["operator_terms"] == fake_terms
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["ansatz_name"] == "hh_uccsd_lifted"
        assert kwargs["display_name"] == "HH-AVQITE-UCCSD-Lifted"
        assert kwargs["parameterization_mode"] == "logical_shared"
        assert kwargs["avqite_step_size"] == 0.1
        assert kwargs["avqite_max_steps"] == 80
        assert kwargs["avqite_energy_tol"] == 1e-8
        assert kwargs["avqite_residual_tol"] == 1e-6
        assert np.asarray(kwargs["psi_ref"], dtype=complex).shape == (2,)
        assert "optimizer" not in kwargs
        assert "seed" not in kwargs
        return {
            "success": True,
            "method_kind": "avqite",
            "display_name": "HH-AVQITE-UCCSD-Lifted",
            "ansatz_name": "hh_uccsd_lifted",
            "energy": -1.22,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.03,
            "delta_E_abs": 0.03,
            "nfev_total": 77,
            "nit": 8,
            "num_parameters": 3,
            "runtime_parameter_count": 9,
            "vqe_reps": None,
            "vqe_restarts": None,
            "vqe_maxiter": None,
            "optimizer": "AVQITE",
            "optimizer_success": False,
            "optimizer_message": "max_steps",
            "converged": False,
            "parameterization_mode": "logical_shared",
            "selected_operator_labels": ["op0", "op1", "op2"],
            "selected_operator_count": 3,
            "avqite_steps_completed": 8,
            "imaginary_time_total": 0.8,
            "avqite_stop_reason": "max_steps",
            "history": [{"event": "initial", "energy": -1.0}],
        }

    monkeypatch.setattr(bench, "_build_hh_uccsd_fermion_lifted_pool", _fake_build_pool)
    monkeypatch.setattr(bench, "run_compiled_operator_avqite_trial", _fake_run_avqite)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_compiled_operator_avqite_algorithm()],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_builder) == 1
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_avqite_uccsd_lifted"
    assert row["method_id"] == "hh_avqite_uccsd_lifted"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "avqite"
    assert row["ansatz_name"] == "hh_uccsd_lifted"
    assert row["pool_name"] == ""
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.03
    assert row["nfev"] == 77
    assert row["num_parameters"] == 3
    assert row["vqe_reps"] is None
    assert row["vqe_restarts"] is None
    assert row["vqe_maxiter"] is None
    assert row["selected_operator_count"] == 3
    assert row["avqite_steps_completed"] == 8
    assert row["avqite_stop_reason"] == "max_steps"
    assert row["imaginary_time_total"] == 0.8
    assert row["adapt_depth_reached"] is None
    assert row["adapt_stop_reason"] == ""
    assert row["continuation_mode"] == ""
    assert row["adapt_pool"] == ""
    assert np.isfinite(row["abs_delta_e"])

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "compiled_operator_avqite"
    assert manifest["algorithms"][0]["operator_source"] == "hh_uccsd_lifted"
    assert manifest["algorithms"][0]["parameterization_mode"] == "logical_shared"
    assert manifest["algorithms"][0]["avqite_max_steps"] == 80

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["method_kind"] == "avqite"
    assert artifact_payload["selected_operator_count"] == 3
    assert artifact_payload["parameterization_mode"] == "logical_shared"
    assert artifact_payload["runtime_parameter_count"] == 9
    assert artifact_payload["avqite_steps_completed"] == 8
    assert artifact_payload["avqite_stop_reason"] == "max_steps"
    assert artifact_payload["imaginary_time_total"] == 0.8

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "avqite"
    assert proxy_rows[0]["ansatz_name"] == "hh_uccsd_lifted"
    assert proxy_rows[0]["vqe_reps"] == ""
    assert proxy_rows[0]["vqe_restarts"] == ""
    assert proxy_rows[0]["vqe_maxiter"] == ""
    assert proxy_rows[0]["selected_operator_count"] == "3"
    assert proxy_rows[0]["avqite_steps_completed"] == "8"
    assert proxy_rows[0]["avqite_stop_reason"] == "max_steps"
    assert proxy_rows[0]["imaginary_time_total"] == "0.8"
    assert proxy_rows[0]["operator_family_proxy"] == "avqite+uccsd"


def test_mocked_qsci_compiled_operator_success_maps_qsci_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    fake_terms = [
        _FakeOperatorTerm("sq_lf_std:op0"),
        _FakeOperatorTerm("sq_lf_std:op1"),
        _FakeOperatorTerm("sq_lf_std:op2"),
    ]
    fake_sector = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)
    captured_plan_kwargs: list[dict[str, object]] = []
    captured_sector_kwargs: list[dict[str, object]] = []
    captured_kwargs: list[dict[str, object]] = []

    def _fake_resolve_pool_plan(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_plan_kwargs.append(dict(kwargs))
        assert kwargs["adapt_pool"] == "sq_lf_std"
        assert kwargs["continuation_mode"] == "legacy"
        assert kwargs["paop_r"] == 1
        assert kwargs["paop_split_paulis"] is False
        assert kwargs["paop_prune_eps"] == 0.0
        assert kwargs["paop_normalization"] == "none"
        assert kwargs["phase3_symmetry_mitigation_mode"] == "off"
        return type("_FakePoolPlan", (), {"pool": fake_terms})()

    def _fake_sector_data(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_sector_kwargs.append(dict(kwargs))
        assert kwargs["case"].case_id == "hh_L2_strong_canonical"
        assert kwargs["resolved_problem"] is not None
        return fake_sector, [0, 1, 2, 3]

    def _fake_run_qsci(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_kwargs.append(dict(kwargs))
        assert kwargs["operator_terms"] == fake_terms
        assert kwargs["operator_labels"] == ["sq_lf_std:op0", "sq_lf_std:op1", "sq_lf_std:op2"]
        assert kwargs["ansatz_name"] == "hh_qsci_sq_lf_std"
        assert kwargs["display_name"] == "HH-QSCI-SQ-LF-Std"
        assert kwargs["sector_hamiltonian"] is fake_sector
        assert kwargs["sector_basis_full_indices"] == [0, 1, 2, 3]
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["basis_probe_angle"] == pytest.approx(float(np.pi / 2))
        assert kwargs["basis_amp_cutoff"] == 1e-9
        assert kwargs["qsci_max_basis_states"] == 32
        assert np.asarray(kwargs["psi_ref"], dtype=complex).shape == (2,)
        assert "h_poly" not in kwargs
        assert "optimizer" not in kwargs
        assert "seed" not in kwargs
        return {
            "success": True,
            "method_kind": "qsci",
            "display_name": "HH-QSCI-SQ-LF-Std",
            "ansatz_name": "hh_qsci_sq_lf_std",
            "energy": -1.20,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.05,
            "delta_E_abs": 0.05,
            "nfev_total": 3,
            "nfev": 3,
            "nit": 0,
            "num_parameters": None,
            "vqe_reps": None,
            "vqe_restarts": None,
            "vqe_maxiter": None,
            "optimizer": "projected_diagonalization",
            "optimizer_success": True,
            "optimizer_message": "projected_diag",
            "converged": True,
            "selected_operator_labels": ["sq_lf_std:op0", "sq_lf_std:op1", "sq_lf_std:op2"],
            "selected_operator_count": 3,
            "selected_basis_full_indices": [0, 1, 2],
            "selected_sector_indices": [0, 1, 2],
            "subspace_dimension": 3,
            "full_sector_dimension": 4,
            "qsci_basis_probe_count": 3,
            "qsci_stop_reason": "projected_diag",
        }

    monkeypatch.setattr(bench, "resolve_pool_plan", _fake_resolve_pool_plan)
    monkeypatch.setattr(bench, "_build_hh_sector_projection_data", _fake_sector_data)
    monkeypatch.setattr(bench, "run_compiled_operator_qsci_trial", _fake_run_qsci)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_compiled_operator_qsci_algorithm()],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_plan_kwargs) == 1
    assert len(captured_sector_kwargs) == 1
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_qsci_sq_lf_std"
    assert row["method_id"] == "hh_qsci_sq_lf_std"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "qsci"
    assert row["ansatz_name"] == "hh_qsci_sq_lf_std"
    assert row["pool_name"] == "sq_lf_std"
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.05
    assert row["nfev"] == 3
    assert row["nit"] == 0
    assert row["num_parameters"] is None
    assert row["vqe_reps"] is None
    assert row["vqe_restarts"] is None
    assert row["vqe_maxiter"] is None
    assert row["selected_operator_count"] == 3
    assert row["subspace_dimension"] == 3
    assert row["adapt_depth_reached"] is None
    assert row["adapt_stop_reason"] == ""
    assert row["continuation_mode"] == ""
    assert row["adapt_pool"] == ""
    assert np.isfinite(row["abs_delta_e"])

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "compiled_operator_qsci"
    assert manifest["algorithms"][0]["operator_source"] == "hh_sq_lf_std_pool"
    assert manifest["algorithms"][0]["basis_probe_angle"] == pytest.approx(float(np.pi / 2))
    assert manifest["algorithms"][0]["basis_amp_cutoff"] == 1e-9
    assert manifest["algorithms"][0]["qsci_max_basis_states"] == 32

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["method_kind"] == "qsci"
    assert artifact_payload["selected_operator_count"] == 3
    assert artifact_payload["subspace_dimension"] == 3
    assert artifact_payload["full_sector_dimension"] == 4
    assert artifact_payload["qsci_basis_probe_count"] == 3
    assert artifact_payload["qsci_stop_reason"] == "projected_diag"

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "qsci"
    assert proxy_rows[0]["ansatz_name"] == "hh_qsci_sq_lf_std"
    assert proxy_rows[0]["pool_name"] == "sq_lf_std"
    assert proxy_rows[0]["nfev"] == "3"
    assert proxy_rows[0]["vqe_reps"] == ""
    assert proxy_rows[0]["vqe_restarts"] == ""
    assert proxy_rows[0]["vqe_maxiter"] == ""
    assert proxy_rows[0]["selected_operator_count"] == "3"
    assert proxy_rows[0]["subspace_dimension"] == "3"
    assert proxy_rows[0]["pool_family_proxy"] == "sq_lf_std"



def test_mocked_sqd_compiled_operator_success_maps_sqd_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    fake_terms = [
        _FakeOperatorTerm("sq_lf_std:op0"),
        _FakeOperatorTerm("sq_lf_std:op1"),
        _FakeOperatorTerm("sq_lf_std:op2"),
    ]
    fake_sector = np.diag([0.0, -1.0, -2.0, -3.0]).astype(complex)
    captured_plan_kwargs: list[dict[str, object]] = []
    captured_sector_kwargs: list[dict[str, object]] = []
    captured_kwargs: list[dict[str, object]] = []

    def _fake_resolve_pool_plan(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_plan_kwargs.append(dict(kwargs))
        assert kwargs["adapt_pool"] == "sq_lf_std"
        assert kwargs["continuation_mode"] == "legacy"
        assert kwargs["phase3_symmetry_mitigation_mode"] == "off"
        return type("_FakePoolPlan", (), {"pool": fake_terms})()

    def _fake_sector_data(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_sector_kwargs.append(dict(kwargs))
        assert kwargs["case"].case_id == "hh_L2_strong_canonical"
        assert kwargs["resolved_problem"] is not None
        return fake_sector, [0, 1, 2, 3]

    def _fake_run_sqd(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_kwargs.append(dict(kwargs))
        assert kwargs["operator_terms"] == fake_terms
        assert kwargs["operator_labels"] == ["sq_lf_std:op0", "sq_lf_std:op1", "sq_lf_std:op2"]
        assert kwargs["ansatz_name"] == "hh_sqd_sq_lf_std"
        assert kwargs["display_name"] == "HH-SQD-SQ-LF-Std"
        assert kwargs["sector_hamiltonian"] is fake_sector
        assert kwargs["sector_basis_full_indices"] == [0, 1, 2, 3]
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["basis_probe_angle"] == pytest.approx(float(np.pi / 2))
        assert kwargs["sqd_shots_per_probe"] == 256
        assert kwargs["sqd_max_basis_states"] == 32
        assert kwargs["sqd_seed"] == 7
        assert np.asarray(kwargs["psi_ref"], dtype=complex).shape == (2,)
        assert "h_poly" not in kwargs
        assert "optimizer" not in kwargs
        assert "seed" not in kwargs
        return {
            "success": True,
            "method_kind": "sqd",
            "display_name": "HH-SQD-SQ-LF-Std",
            "ansatz_name": "hh_sqd_sq_lf_std",
            "energy": -1.19,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.06,
            "delta_E_abs": 0.06,
            "nfev_total": 3,
            "nfev": 3,
            "shots_total": 768,
            "nit": 0,
            "num_parameters": None,
            "vqe_reps": None,
            "vqe_restarts": None,
            "vqe_maxiter": None,
            "optimizer": "projected_diagonalization",
            "optimizer_success": True,
            "optimizer_message": "projected_diag",
            "converged": True,
            "selected_operator_labels": ["sq_lf_std:op0", "sq_lf_std:op1", "sq_lf_std:op2"],
            "selected_operator_count": 3,
            "selected_basis_full_indices": [0, 1, 2],
            "selected_sector_indices": [0, 1, 2],
            "subspace_dimension": 3,
            "full_sector_dimension": 4,
            "sqd_basis_probe_count": 3,
            "sqd_seed": 7,
            "sqd_stop_reason": "projected_diag",
            "sqd_sample_counts_by_full_index": {"0": 12, "1": 250, "2": 240},
        }

    monkeypatch.setattr(bench, "resolve_pool_plan", _fake_resolve_pool_plan)
    monkeypatch.setattr(bench, "_build_hh_sector_projection_data", _fake_sector_data)
    monkeypatch.setattr(bench, "run_compiled_operator_sqd_trial", _fake_run_sqd)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[_compiled_operator_sqd_algorithm()],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_plan_kwargs) == 1
    assert len(captured_sector_kwargs) == 1
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_sqd_sq_lf_std"
    assert row["method_id"] == "hh_sqd_sq_lf_std"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "sqd"
    assert row["ansatz_name"] == "hh_sqd_sq_lf_std"
    assert row["pool_name"] == "sq_lf_std"
    assert row["status"] == "ok"
    assert row["delta_E_abs"] == 0.06
    assert row["nfev"] == 3
    assert row["shots_total"] == 768
    assert row["nit"] == 0
    assert row["num_parameters"] is None
    assert row["vqe_reps"] is None
    assert row["vqe_restarts"] is None
    assert row["vqe_maxiter"] is None
    assert row["selected_operator_count"] == 3
    assert row["subspace_dimension"] == 3
    assert row["adapt_depth_reached"] is None
    assert row["adapt_stop_reason"] == ""
    assert row["continuation_mode"] == ""
    assert row["adapt_pool"] == ""
    assert np.isfinite(row["abs_delta_e"])

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "compiled_operator_sqd"
    assert manifest["algorithms"][0]["operator_source"] == "hh_sq_lf_std_pool"
    assert manifest["algorithms"][0]["basis_probe_angle"] == pytest.approx(float(np.pi / 2))
    assert manifest["algorithms"][0]["sqd_shots_per_probe"] == 256
    assert manifest["algorithms"][0]["sqd_max_basis_states"] == 32
    assert manifest["algorithms"][0]["sqd_seed"] == 7

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["method_kind"] == "sqd"
    assert artifact_payload["selected_operator_count"] == 3
    assert artifact_payload["subspace_dimension"] == 3
    assert artifact_payload["full_sector_dimension"] == 4
    assert artifact_payload["shots_total"] == 768
    assert artifact_payload["sqd_basis_probe_count"] == 3
    assert artifact_payload["sqd_seed"] == 7
    assert artifact_payload["sqd_stop_reason"] == "projected_diag"
    assert artifact_payload["sqd_sample_counts_by_full_index"] == {"0": 12, "1": 250, "2": 240}

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["method_kind"] == "sqd"
    assert proxy_rows[0]["ansatz_name"] == "hh_sqd_sq_lf_std"
    assert proxy_rows[0]["pool_name"] == "sq_lf_std"
    assert proxy_rows[0]["nfev"] == "3"
    assert proxy_rows[0]["shots_total"] == "768"
    assert proxy_rows[0]["vqe_reps"] == ""
    assert proxy_rows[0]["vqe_restarts"] == ""
    assert proxy_rows[0]["vqe_maxiter"] == ""
    assert proxy_rows[0]["selected_operator_count"] == "3"
    assert proxy_rows[0]["subspace_dimension"] == "3"
    assert proxy_rows[0]["pool_family_proxy"] == "sq_lf_std"



def test_mocked_puccd_compiled_operator_success_maps_conventional_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)
    mixed_terms = [
        _FakeOperatorTerm("uccsd_ferm_lifted::uccsd_sing(alpha:0->1)"),
        _FakeOperatorTerm("uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,2)"),
        _FakeOperatorTerm("uccsd_ferm_lifted::uccsd_dbl(ab:0,2->1,3)"),
    ]
    captured_kwargs: list[dict[str, object]] = []

    def _fake_build_pool(*args, **kwargs):  # noqa: ANN002, ANN003 - mirrors production builder
        assert args == (2, 1, "binary", "blocked", "open")
        assert kwargs["num_particles"] == (1, 1)
        return mixed_terms

    def _fake_run_compiled(**kwargs):  # noqa: ANN003 - production signature is broad
        captured_kwargs.append(dict(kwargs))
        assert kwargs["operator_terms"] == [mixed_terms[2]]
        assert kwargs["h_poly"] == "fake-h-poly"
        assert kwargs["exact_gs"] == -1.25
        assert kwargs["ansatz_name"] == "hh_puccd_lifted"
        assert kwargs["display_name"] == "HH-pUCCD-Lifted"
        assert kwargs["parameterization_mode"] == "logical_shared"
        assert kwargs["seed"] == 42
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-pUCCD-Lifted",
            "ansatz_name": "hh_puccd_lifted",
            "energy": -1.24,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.01,
            "delta_E_abs": 0.01,
            "nfev": 12,
            "nit": 3,
            "num_parameters": 1,
            "runtime_parameter_count": 4,
            "vqe_reps": None,
            "vqe_restarts": 3,
            "vqe_maxiter": 800,
            "optimizer": "COBYLA",
            "optimizer_success": True,
            "optimizer_message": "mocked ok",
            "converged": True,
            "parameterization_mode": "logical_shared",
            "selected_operator_labels": [mixed_terms[2].label],
            "selected_operator_count": 1,
        }

    monkeypatch.setattr(bench, "_build_hh_uccsd_fermion_lifted_pool", _fake_build_pool)
    monkeypatch.setattr(bench, "run_compiled_operator_vqe_trial", _fake_run_compiled)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[
            _compiled_operator_algorithm(
                "hh_puccd_lifted_vqe",
                "hh_puccd_lifted",
                display_name="HH-pUCCD-Lifted",
                ansatz_name="hh_puccd_lifted",
            )
        ],
    )

    assert result["row_count"] == 1
    assert result["failed_row_count"] == 0
    assert len(captured_kwargs) == 1
    row = result["rows"][0]
    assert row["run_id"] == "hh_L2_strong_canonical__hh_puccd_lifted_vqe"
    assert row["method_id"] == "hh_puccd_lifted_vqe"
    assert row["hamiltonian_id"] == "hh_L2_strong_canonical"
    assert row["method_kind"] == "conventional_vqe"
    assert row["ansatz_name"] == "hh_puccd_lifted"
    assert row["vqe_reps"] is None
    assert row["selected_operator_count"] == 1
    assert np.isfinite(row["abs_delta_e"])

    manifest = json.loads((tmp_path / "hh_static_benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["algorithms"][0]["runner_kind"] == "compiled_operator_vqe"
    assert manifest["algorithms"][0]["operator_source"] == "hh_puccd_lifted"
    assert manifest["algorithms"][0]["parameterization_mode"] == "logical_shared"

    artifact_payload = json.loads(Path(row["artifact_json"]).read_text(encoding="utf-8"))
    assert artifact_payload["benchmark_status"] == "ok"
    assert artifact_payload["selected_operator_count"] == 1
    assert artifact_payload["selected_operator_labels"] == [mixed_terms[2].label]

    with (tmp_path / "summary" / "metrics_proxy_runs.csv").open("r", encoding="utf-8", newline="") as f_csv:
        proxy_rows = list(csv.DictReader(f_csv))
    assert proxy_rows[0]["selected_operator_count"] == "1"



def test_row_failure_is_logged_and_later_rows_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)

    def _fake_run_adapt(**kwargs):  # noqa: ANN003 - production signature is large
        if kwargs["adapt_pool"] == "bad_pool":
            raise RuntimeError("mocked ADAPT failure")
        return (
            {
                "success": True,
                "energy": -1.2,
                "exact_gs_energy": -1.25,
                "abs_delta_e": 0.05,
                "nfev_total": 7,
                "num_parameters": 1,
                "ansatz_depth": 1,
                "pool_type": kwargs["adapt_pool"],
                "stop_reason": "eps_energy",
            },
            object(),
        )

    monkeypatch.setattr(
        bench,
        "_run_hardcoded_adapt_vqe_compatibility",
        _fake_run_adapt,
    )

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[
            _algorithm("hh_adapt_bad_legacy", "bad_pool"),
            _algorithm("hh_adapt_good_legacy", "good_pool"),
        ],
    )

    assert result["row_count"] == 2
    assert result["failed_row_count"] == 1
    assert [row["status"] for row in result["rows"]] == ["failed", "ok"]
    assert result["rows"][0]["delta_E_abs"] is None
    assert result["rows"][1]["delta_E_abs"] == 0.05

    failure_payload = json.loads(Path(result["rows"][0]["artifact_json"]).read_text(encoding="utf-8"))
    assert failure_payload["benchmark_status"] == "failed"
    assert failure_payload["benchmark_stage"] == "adapt_run"
    assert failure_payload["error_type"] == "RuntimeError"
    assert "mocked ADAPT failure" in failure_payload["error_message"]

    success_payload = json.loads(Path(result["rows"][1]["artifact_json"]).read_text(encoding="utf-8"))
    assert success_payload["benchmark_status"] == "ok"

    summary = json.loads((tmp_path / "summary" / "metrics_proxy_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["status_counts"] == {"failed": 1, "ok": 1}


def test_compiled_operator_failure_is_logged_and_later_rows_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)

    def _fake_run_compiled_operator(**kwargs):  # noqa: ANN003 - production signature is broad
        raise RuntimeError("mocked compiled-operator failure")

    def _fake_run_conventional(**kwargs):  # noqa: ANN003 - production signature is broad
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-Termwise",
            "ansatz_name": "hh_hva_termwise",
            "energy": -1.20,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.05,
            "delta_E_abs": 0.05,
            "nfev": 9,
            "nit": 2,
            "num_parameters": 4,
            "vqe_reps_used": 2,
            "vqe_restarts": 3,
            "vqe_maxiter_used": 800,
            "optimizer": "COBYLA",
            "optimizer_success": True,
            "optimizer_message": "ok",
            "converged": True,
        }

    monkeypatch.setattr(bench, "_run_one_compiled_operator_algorithm", _fake_run_compiled_operator)
    monkeypatch.setattr(bench, "run_hh_conventional_vqe_trial", _fake_run_conventional)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[
            _compiled_operator_algorithm(),
            _conventional_algorithm("hh_hva_termwise_vqe", "termwise", "hh_hva_termwise", "HH-Termwise"),
        ],
    )

    assert result["row_count"] == 2
    assert result["failed_row_count"] == 1
    assert [row["status"] for row in result["rows"]] == ["failed", "ok"]
    assert [row["method_kind"] for row in result["rows"]] == ["conventional_vqe", "conventional_vqe"]
    assert result["rows"][0]["ansatz_name"] == "hh_uccsd_lifted"
    assert result["rows"][0]["vqe_reps"] is None
    assert result["rows"][0]["vqe_restarts"] == 3
    assert result["rows"][0]["vqe_maxiter"] == 800
    assert result["rows"][1]["delta_E_abs"] == 0.05

    failure_payload = json.loads(Path(result["rows"][0]["artifact_json"]).read_text(encoding="utf-8"))
    assert failure_payload["benchmark_status"] == "failed"
    assert failure_payload["benchmark_stage"] == "compiled_operator_run"
    assert failure_payload["error_type"] == "RuntimeError"
    assert "mocked compiled-operator failure" in failure_payload["error_message"]

    success_payload = json.loads(Path(result["rows"][1]["artifact_json"]).read_text(encoding="utf-8"))
    assert success_payload["benchmark_status"] == "ok"

    summary = json.loads((tmp_path / "summary" / "metrics_proxy_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["status_counts"] == {"failed": 1, "ok": 1}


def test_conventional_failure_is_logged_and_later_rows_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_problem_resolution(monkeypatch)

    def _fake_run_conventional(**kwargs):  # noqa: ANN003 - production signature is large
        if kwargs["ansatz_kind"] == "termwise":
            raise RuntimeError("mocked conventional failure")
        return {
            "success": True,
            "method_kind": "conventional_vqe",
            "display_name": "HH-Layerwise",
            "ansatz_name": "hh_hva_layerwise",
            "energy": -1.20,
            "exact_gs_energy": -1.25,
            "abs_delta_e": 0.05,
            "delta_E_abs": 0.05,
            "nfev": 9,
            "nit": 2,
            "num_parameters": 4,
            "vqe_reps_used": 2,
            "vqe_restarts": 3,
            "vqe_maxiter_used": 800,
            "optimizer": "COBYLA",
            "optimizer_success": True,
            "optimizer_message": "ok",
            "converged": True,
        }

    monkeypatch.setattr(bench, "run_hh_conventional_vqe_trial", _fake_run_conventional)

    result = bench.run_hh_static_ground_state_benchmark(
        output_dir=tmp_path,
        cases=[_case("hh_L2_strong_canonical")],
        algorithms=[
            _conventional_algorithm("hh_hva_termwise_vqe", "termwise", "hh_hva_termwise", "HH-Termwise"),
            _conventional_algorithm("hh_hva_layerwise_vqe", "layerwise", "hh_hva_layerwise", "HH-Layerwise"),
        ],
    )

    assert result["row_count"] == 2
    assert result["failed_row_count"] == 1
    assert [row["status"] for row in result["rows"]] == ["failed", "ok"]
    assert [row["method_kind"] for row in result["rows"]] == ["conventional_vqe", "conventional_vqe"]
    assert result["rows"][0]["delta_E_abs"] is None
    assert result["rows"][0]["vqe_reps"] == 2
    assert result["rows"][0]["vqe_restarts"] == 3
    assert result["rows"][0]["vqe_maxiter"] == 800
    assert result["rows"][1]["delta_E_abs"] == 0.05

    failure_payload = json.loads(Path(result["rows"][0]["artifact_json"]).read_text(encoding="utf-8"))
    assert failure_payload["benchmark_status"] == "failed"
    assert failure_payload["benchmark_stage"] == "conventional_run"
    assert failure_payload["error_type"] == "RuntimeError"
    assert "mocked conventional failure" in failure_payload["error_message"]

    success_payload = json.loads(Path(result["rows"][1]["artifact_json"]).read_text(encoding="utf-8"))
    assert success_payload["benchmark_status"] == "ok"
    assert success_payload["optimizer_success"] is True

    summary = json.loads((tmp_path / "summary" / "metrics_proxy_summary.json").read_text(encoding="utf-8"))
    assert summary["row_count"] == 2
    assert summary["status_counts"] == {"failed": 1, "ok": 1}
